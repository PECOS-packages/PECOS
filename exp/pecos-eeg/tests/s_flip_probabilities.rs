// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! S injections are physical independent flips in both noise views.

use pecos_core::Gate;
use pecos_core::gate_type::GateType;
use pecos_core::pauli::pauli_bitmask::BitmaskStorage;
use pecos_eeg::Bm;
use pecos_eeg::circuit::analyze_with_noise;
use pecos_eeg::coherent_dem::{build_coherent_dem, build_coherent_dem_decomposable};
use pecos_eeg::dem_mapping::{Detector, build_dem};
use pecos_eeg::eeg::EegType;
use pecos_eeg::expand::{EegBuildError, GateIndex, expand_circuit, make_gate};
use pecos_eeg::heisenberg::{
    heisenberg_detection_probability, heisenberg_exact_from_circuit, heisenberg_sparse,
};
use pecos_eeg::noise::{NoiseInjection, NoiseSpec, UniformNoise};
use pecos_eeg::noise_compression::{CompressedNoiseSpec, compress_noise_to_boundaries};
use pecos_eeg::stabilizer::StabilizerGroup;

fn detector(qubits: &[usize]) -> Detector {
    let mut stabilizer = Bm::default();
    for &q in qubits {
        stabilizer.z_bits.set_bit(q);
    }
    Detector { id: 0, stabilizer }
}

/// Compare both backward builders and all three physical walks to a hand oracle.
fn check_backward(gates: &[Gate], noise: &dyn NoiseSpec, records: &[usize], expected: f64) {
    let expanded = expand_circuit(gates).unwrap();
    let qubits: Vec<_> = records
        .iter()
        .map(|&r| expanded.measurement_qubit[r])
        .collect();
    let det = detector(&qubits);
    let detectors = [det];
    let entries = build_coherent_dem(
        &expanded.gates,
        noise,
        &detectors,
        &[],
        &expanded.expansion_gates,
    );
    let decomposed = build_coherent_dem_decomposable(
        &expanded.gates,
        noise,
        &detectors,
        &[],
        &expanded.expansion_gates,
    );
    if expected == 0.0 {
        assert!(entries.is_empty());
        assert!(decomposed.is_empty());
    } else {
        assert_eq!(entries.len(), 1);
        assert_eq!(decomposed.len(), 1);
        assert_eq!(entries[0].event.detectors.as_slice(), &[0]);
        assert_eq!(decomposed[0].event, entries[0].event);
        assert!((entries[0].probability - expected).abs() < 1e-12);
        assert!((decomposed[0].probability - expected).abs() < 1e-12);
    }
    let init = [make_gate(
        GateType::PZ,
        &(0..expanded.num_qubits).collect::<Vec<_>>(),
    )];
    let stab = StabilizerGroup::from_circuit(&init, expanded.num_qubits);
    let index = GateIndex::build(
        &expanded.gates,
        expanded.num_qubits,
        noise,
        &expanded.expansion_gates,
    );
    let walk = heisenberg_detection_probability(
        &expanded.gates,
        &detectors[0].stabilizer,
        noise,
        &stab,
        0.0,
        &expanded.expansion_gates,
    );
    let sparse = heisenberg_sparse(
        &expanded.gates,
        &detectors[0].stabilizer,
        noise,
        &stab,
        0.0,
        &index,
        None,
    );
    let dense =
        heisenberg_exact_from_circuit(gates, records, noise, expanded.num_original_qubits).unwrap();
    for probability in [walk, sparse, dense] {
        assert!(
            (probability - expected).abs() < 1e-12,
            "{probability} != {expected}"
        );
    }
}

#[test]
fn backward_measurement_flip_probability() {
    let gates = [make_gate(GateType::PZ, &[0]), make_gate(GateType::MZ, &[0])];
    let noise = UniformNoise {
        p_meas: 0.25,
        ..UniformNoise::coherent_only(0.0)
    };
    check_backward(&gates, &noise, &[0], 0.25);
}

#[test]
fn backward_preparation_flip_probability() {
    for prep in [GateType::PZ, GateType::QAlloc] {
        let gates = [make_gate(prep, &[0]), make_gate(GateType::MZ, &[0])];
        let noise = UniformNoise {
            p_prep: 0.25,
            ..UniformNoise::coherent_only(0.0)
        };
        check_backward(&gates, &noise, &[0], 0.25);
    }
}

#[test]
fn backward_two_measurements_compose_independently() {
    let gates = [
        make_gate(GateType::PZ, &[0, 1]),
        make_gate(GateType::MZ, &[0]),
        make_gate(GateType::MZ, &[1]),
    ];
    let noise = UniformNoise {
        p_meas: 0.25,
        ..UniformNoise::coherent_only(0.0)
    };
    // Odd parity: 2 * 0.25 * 0.75 = 0.375, including merging different labels.
    check_backward(&gates, &noise, &[0, 1], 0.375);
}

struct Flips {
    gate_type: GateType,
    rates: Vec<f64>,
}

impl NoiseSpec for Flips {
    fn noise_after_gate(
        &self,
        _: usize,
        gate_type: GateType,
        qubits: &[usize],
    ) -> Vec<NoiseInjection> {
        if gate_type != self.gate_type {
            return Vec::new();
        }
        self.rates
            .iter()
            .map(|&rate| NoiseInjection {
                eeg_type: EegType::S,
                label: Bm::x(qubits[0]),
                label2: None,
                rate,
            })
            .collect()
    }
}

#[test]
fn backward_custom_flips_cover_full_probability_range() {
    let gates = [make_gate(GateType::PZ, &[0]), make_gate(GateType::MZ, &[0])];
    for (rates, expected) in [
        (vec![0.0], 0.0),
        (vec![-0.5], 0.5),
        (vec![-0.75], 0.75),
        (vec![-1.0], 1.0),
        (vec![-1.0, -1.0], 0.0),
        (vec![-0.25, -0.125], 0.3125),
    ] {
        check_backward(
            &gates,
            &Flips {
                gate_type: GateType::MZ,
                rates,
            },
            &[0],
            expected,
        );
    }
}

#[test]
fn forward_measurement_flip_probability() {
    let gates = [make_gate(GateType::PZ, &[0]), make_gate(GateType::MZ, &[0])];
    let noise = UniformNoise {
        p_meas: 0.25,
        ..UniformNoise::coherent_only(0.0)
    };
    let result = analyze_with_noise(&gates, &noise, &[false; 2]).unwrap();
    let entries = build_dem(&result.generators, &[detector(&[0])], &[]);
    assert_eq!(entries.len(), 1);
    assert!((entries[0].probability - 0.25).abs() < 1e-12);
}

#[test]
fn forward_rejects_unrepresentable_s_probabilities() {
    let gates = [make_gate(GateType::MZ, &[0])];
    let noise = UniformNoise {
        p_meas: 0.5,
        ..UniformNoise::coherent_only(0.0)
    };
    assert!(matches!(
        analyze_with_noise(&gates, &noise, &[false]),
        Err(EegBuildError::InvalidStochasticProbability {
            gate_index: 0,
            probability: 0.5
        })
    ));
    for rate in [
        0.25,
        -0.5,
        -0.75,
        -1.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
    ] {
        let noise = Flips {
            gate_type: GateType::MZ,
            rates: vec![rate],
        };
        assert!(matches!(
            analyze_with_noise(&gates, &noise, &[false]),
            Err(EegBuildError::InvalidStochasticProbability { gate_index: 0, .. })
        ));
    }
}

// Sources at distinct locations propagate to the same boundary label.
struct BoundaryFlips(Vec<f64>);

impl NoiseSpec for BoundaryFlips {
    fn noise_after_gate(
        &self,
        gate_index: usize,
        gate_type: GateType,
        qubits: &[usize],
    ) -> Vec<NoiseInjection> {
        if gate_type != GateType::I {
            return Vec::new();
        }
        Flips {
            gate_type,
            rates: vec![self.0[gate_index - 1]],
        }
        .noise_after_gate(gate_index, gate_type, qubits)
    }
}

#[test]
fn compression_composes_s_flips() {
    let gates = [
        make_gate(GateType::PZ, &[0]),
        make_gate(GateType::I, &[0]),
        make_gate(GateType::I, &[0]),
        make_gate(GateType::MZ, &[0]),
    ];
    // (1 - (1 - 2*0.25)(1 - 2*0.125))/2 = 0.3125.
    // Also cover product zero (p=0.5), negative products, and exact cancellation.
    for (rates, expected) in [
        (vec![-0.25, -0.125], 0.3125),
        (vec![-0.5, -0.125], 0.5),
        (vec![-0.75, -0.125], 0.6875),
        (vec![-1.0, -1.0], 0.0),
    ] {
        let noise = BoundaryFlips(rates);
        let compressed = compress_noise_to_boundaries(&gates, &noise, &[false; 4]);
        if expected == 0.0 {
            assert!(compressed.boundary_sources.is_empty());
        } else {
            assert_eq!(compressed.boundary_sources.len(), 1);
            assert!((compressed.boundary_sources[0].value + expected).abs() < 1e-12);
        }
        let spec = CompressedNoiseSpec::from_compressed(&compressed);
        let detectors = [detector(&[0])];
        let original = build_coherent_dem(&gates, &noise, &detectors, &[], &[false; 4]);
        let combined = build_coherent_dem(&gates, &spec, &detectors, &[], &[false; 4]);
        assert_eq!(original.len(), combined.len());
        if expected != 0.0 {
            assert_eq!(original[0].event, combined[0].event);
            assert!((original[0].probability - expected).abs() < 1e-12);
            assert!((combined[0].probability - expected).abs() < 1e-12);
        }
    }
}

#[test]
fn builder_propagates_unrepresentable_s_probability() {
    use pecos_eeg::builder::EegDemBuilder;
    use pecos_eeg::circuit::NoiseModel;
    use pecos_quantum::TickCircuit;

    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0]);
    circuit.tick().mz(&[0]);
    let builder = EegDemBuilder::from_tick_circuit(&circuit).noise(NoiseModel {
        p_meas: 0.5,
        ..NoiseModel::coherent_only(0.0)
    });
    assert!(matches!(
        builder.build(),
        Err(EegBuildError::InvalidStochasticProbability { .. })
    ));
    assert!(matches!(
        builder.summary(),
        Err(EegBuildError::InvalidStochasticProbability { .. })
    ));
    assert!(matches!(
        builder.build_dem_string(),
        Err(EegBuildError::InvalidStochasticProbability { .. })
    ));
}

#[test]
fn compression_preserves_measurement_and_preparation_injections() {
    let gates = [make_gate(GateType::PZ, &[0]), make_gate(GateType::MZ, &[0])];
    let noise = UniformNoise {
        p_prep: 0.75,
        p_meas: 0.5,
        ..UniformNoise::coherent_only(0.0)
    };
    let compressed = compress_noise_to_boundaries(&gates, &noise, &[false; 2]);
    assert!(compressed.boundary_sources.is_empty());
    assert_eq!(compressed.preparation_sources.len(), 1);
    assert_eq!(compressed.measurement_sources.len(), 1);
    let spec = CompressedNoiseSpec::from_compressed(&compressed);
    for (i, gate) in gates.iter().enumerate() {
        let injections = spec.noise_after_gate(i, gate.gate_type, &[0]);
        assert_eq!(injections.len(), 1);
        assert_eq!(injections[0].label, Bm::x(0));
        assert_eq!(injections[0].eeg_type, EegType::S);
        assert!(injections[0].label2.is_none());
        assert_eq!(injections[0].rate, if i == 0 { -0.75 } else { -0.5 });
    }
}
