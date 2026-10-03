// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Expansion provenance distinguishes virtual gates from identical user gates.

use pecos_core::Gate;
use pecos_core::gate_type::GateType;
use pecos_eeg::circuit::analyze_with_noise;
use pecos_eeg::eeg::EegType;
use pecos_eeg::expand::{GateIndex, expand_circuit, make_gate};
use pecos_eeg::heisenberg::{
    build_noise_map, heisenberg_detection_probability,
    heisenberg_detection_probability_from_circuit, heisenberg_exact_from_circuit,
    heisenberg_sparse, heisenberg_with_noise_map,
};
use pecos_eeg::stabilizer::StabilizerGroup;
use pecos_eeg::{Bm, NoiseInjection, NoiseSpec, UniformNoise};

fn reproducer() -> [Gate; 4] {
    [
        Gate::pz(&[0]),
        make_gate(GateType::QAlloc, &[1]),
        Gate::cx(&[(0, 1)]),
        Gate::mz(&[1]),
    ]
}

fn p2_noise(p2: f64) -> UniformNoise {
    UniformNoise {
        p2,
        idle_rz: 0.0,
        p1: 0.0,
        p_meas: 0.0,
        p_prep: 0.0,
    }
}

/// Include every walk in the diagnostic, so a shared regression cannot hide
/// behind the first failing implementation.
fn assert_walks(gates: &[Gate], record: usize, noise: &dyn NoiseSpec, expected: f64) {
    let expanded = expand_circuit(gates).unwrap();
    let detector = Bm::z(expanded.aux_qubit_for_record(record).unwrap());
    let initial_gates: Vec<_> = (0..expanded.num_original_qubits)
        .map(|q| Gate::pz(&[q]))
        .collect();
    let initial = StabilizerGroup::from_circuit(&initial_gates, expanded.num_qubits);
    let flags = &expanded.expansion_gates;
    let index = GateIndex::build(&expanded.gates, expanded.num_qubits, noise, flags);
    let noise_map = build_noise_map(&expanded.gates, noise, flags);
    let results = [
        (
            "windowed",
            heisenberg_detection_probability(
                &expanded.gates,
                &detector,
                noise,
                &initial,
                0.0,
                flags,
            ),
        ),
        (
            "precomputed",
            heisenberg_with_noise_map(&expanded.gates, &detector, &noise_map, &initial, 0.0),
        ),
        (
            "sparse",
            heisenberg_sparse(
                &expanded.gates,
                &detector,
                noise,
                &initial,
                0.0,
                &index,
                None,
            ),
        ),
        (
            "sparse precomputed",
            heisenberg_sparse(
                &expanded.gates,
                &detector,
                noise,
                &initial,
                0.0,
                &index,
                Some(&noise_map),
            ),
        ),
        (
            "from circuit",
            heisenberg_detection_probability_from_circuit(
                gates,
                &[record],
                noise,
                expanded.num_original_qubits,
                0.0,
            )
            .unwrap(),
        ),
        (
            "matrix",
            heisenberg_exact_from_circuit(gates, &[record], noise, expanded.num_original_qubits)
                .unwrap(),
        ),
    ];
    assert!(
        results.iter().all(|(_, p)| (p - expected).abs() < 1e-12),
        "expected {expected}; walks: {results:?}"
    );
}

#[test]
fn user_cx_keeps_builtin_noise() {
    for p2 in [0.1, 0.3, 0.6] {
        assert_walks(&reproducer(), 0, &p2_noise(p2), 8.0 * p2 / 15.0);
    }
}

/// A user allocation is a preparation and carries built-in prep noise. The
/// allocation inserted for the measurement does not: if it did, the record
/// would flip with probability 2p(1-p) instead of p.
#[test]
fn user_qalloc_keeps_builtin_prep_noise() {
    for p_prep in [0.1, 0.3] {
        let noise = UniformNoise {
            p_prep,
            idle_rz: 0.0,
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.0,
        };
        for prep in [GateType::PZ, GateType::QAlloc] {
            assert_walks(&[make_gate(prep, &[0]), Gate::mz(&[0])], 0, &noise, p_prep);
        }
    }
}

struct FlipAt {
    index: usize,
    qubit: usize,
    probability: f64,
}

impl NoiseSpec for FlipAt {
    fn noise_after_gate(&self, index: usize, _: GateType, _: &[usize]) -> Vec<NoiseInjection> {
        if index == self.index {
            vec![NoiseInjection {
                eeg_type: EegType::S,
                label: Bm::x(self.qubit),
                label2: None,
                rate: -self.probability,
            }]
        } else {
            Vec::new()
        }
    }
}

#[test]
fn user_qalloc_keeps_custom_noise() {
    let gates = [make_gate(GateType::QAlloc, &[1]), Gate::mz(&[1])];
    let expanded = expand_circuit(&gates).unwrap();
    let index = expanded
        .gates
        .iter()
        .position(|g| g.gate_type == GateType::QAlloc && g.qubits[0].index() == 1)
        .unwrap();
    assert_walks(
        &gates,
        0,
        &FlipAt {
            index,
            qubit: 1,
            probability: 0.3,
        },
        0.3,
    );
}

#[test]
fn user_reset_keeps_custom_noise() {
    let gates = [
        Gate::pz(&[0]),
        make_gate(GateType::QAlloc, &[1]),
        Gate::cx(&[(0, 1)]),
        Gate::pz(&[0]),
        Gate::mz(&[0]),
    ];
    assert_walks(
        &gates,
        0,
        &FlipAt {
            index: 3,
            qubit: 0,
            probability: 0.3,
        },
        0.3,
    );
}

fn assert_inserted_noise_skipped(index: usize, gate_type: GateType, qubit: usize, record: usize) {
    // Expanded: PZ(0), QAlloc(1), CX(0,1), PZ(0), QAlloc(2), CX(0,2), MZ(1), MZ(2).
    // The MPZ reset is followed by use of qubit 0, without another reset to
    // erase an erroneous X injection after the inserted PZ.
    let gates = [
        Gate::pz(&[0]),
        make_gate(GateType::MPZ, &[0]),
        Gate::mz(&[0]),
    ];
    let expanded = expand_circuit(&gates).unwrap();
    assert_eq!(expanded.gates[index].gate_type, gate_type);
    assert_eq!(expanded.gates[4].gate_type, GateType::QAlloc);
    assert_eq!(expanded.gates[5].gate_type, GateType::CX);
    assert_walks(&gates, record, &p2_noise(0.0), 0.0);
    assert_walks(
        &gates,
        record,
        &FlipAt {
            index,
            qubit,
            probability: 0.3,
        },
        0.0,
    );
}

#[test]
fn inserted_qalloc_skips_noise() {
    assert_inserted_noise_skipped(1, GateType::QAlloc, 1, 0);
}

#[test]
fn inserted_cx_skips_noise() {
    assert_inserted_noise_skipped(2, GateType::CX, 1, 0);
}

#[test]
fn inserted_reset_skips_noise() {
    assert_inserted_noise_skipped(3, GateType::PZ, 0, 1);
}

#[test]
fn forward_eeg_keeps_user_cx_generators() {
    let expanded = expand_circuit(&reproducer()).unwrap();
    let result = analyze_with_noise(&expanded.gates, &p2_noise(0.3), &expanded.expansion_gates);
    assert_eq!(result.generators.len(), 15);
}

#[test]
#[should_panic(expected = "expansion_gates length 0 must equal gates length 1")]
fn walk_rejects_wrong_flag_length() {
    let gates = [Gate::pz(&[0])];
    let initial = StabilizerGroup::from_circuit(&gates, 1);
    heisenberg_detection_probability(&gates, &Bm::z(0), &p2_noise(0.0), &initial, 0.0, &[]);
}

#[test]
#[should_panic(expected = "expansion_gates length 0 must equal gates length 1")]
fn index_rejects_wrong_flag_length() {
    let _ = GateIndex::build(&[Gate::pz(&[0])], 1, &p2_noise(0.0), &[]);
}

#[test]
#[should_panic(expected = "expansion_gates length 0 must equal gates length 1")]
fn noise_map_rejects_wrong_flag_length() {
    build_noise_map(&[Gate::pz(&[0])], &p2_noise(0.0), &[]);
}

#[test]
#[should_panic(expected = "noise_map length 0 must equal gates length 1")]
fn walk_rejects_wrong_noise_map_length() {
    let gates = [Gate::pz(&[0])];
    let initial = StabilizerGroup::from_circuit(&gates, 1);
    let _ = heisenberg_with_noise_map(&gates, &Bm::z(0), &[], &initial, 0.0);
}

#[test]
#[should_panic(expected = "gate_index.expansion_gates length 0 must equal gates length 1")]
fn sparse_rejects_wrong_index_length() {
    let gates = [Gate::pz(&[0])];
    let initial = StabilizerGroup::from_circuit(&gates, 1);
    let index = GateIndex::build(&[], 1, &p2_noise(0.0), &[]);
    heisenberg_sparse(
        &gates,
        &Bm::z(0),
        &p2_noise(0.0),
        &initial,
        0.0,
        &index,
        None,
    );
}

#[test]
#[should_panic(expected = "noise_map length 0 must equal gates length 1")]
fn sparse_rejects_wrong_noise_map_length() {
    let gates = [Gate::pz(&[0])];
    let initial = StabilizerGroup::from_circuit(&gates, 1);
    let index = GateIndex::build(&gates, 1, &p2_noise(0.0), &[false]);
    heisenberg_sparse(
        &gates,
        &Bm::z(0),
        &p2_noise(0.0),
        &initial,
        0.0,
        &index,
        Some(&[]),
    );
}

#[test]
fn coherent_dem_and_compression_skip_noise_on_inserted_gates() {
    use pecos_eeg::coherent_dem::build_coherent_dem;
    use pecos_eeg::dem_mapping::Detector;
    use pecos_eeg::noise_compression::compress_noise_to_boundaries;

    // Expanded: PZ(0), H(0), H(0), QAlloc(1), CX(0,1), MZ(1). An X0 flip
    // after the second H and an X1 flip after the inserted CX both flip the
    // record, so only the provenance flags decide whether these consumers see
    // them.
    let gates = [Gate::pz(&[0]), Gate::h(&[0]), Gate::h(&[0]), Gate::mz(&[0])];
    let expanded = expand_circuit(&gates).unwrap();
    assert_eq!(expanded.gates[4].gate_type, GateType::CX);
    assert!(expanded.expansion_gates[4]);
    let detector = Detector {
        id: 0,
        stabilizer: Bm::z(expanded.aux_qubit_for_record(0).unwrap()),
    };
    for (index, qubit, inserted) in [(2, 0, false), (4, 1, true)] {
        let noise = FlipAt {
            index,
            qubit,
            probability: 0.2,
        };
        let dem = build_coherent_dem(
            &expanded.gates,
            &noise,
            std::slice::from_ref(&detector),
            &[],
            &expanded.expansion_gates,
        );
        let compressed =
            compress_noise_to_boundaries(&expanded.gates, &noise, &expanded.expansion_gates);
        if inserted {
            assert!(
                dem.is_empty(),
                "noise after the inserted CX reached the DEM"
            );
            assert_eq!(compressed.original_count, 0);
        } else {
            // One mechanism; its probability follows the DEM's own S-type
            // generator approximation, which this test does not pin.
            assert_eq!(dem.len(), 1);
            assert!(dem[0].probability > 0.0);
            assert_eq!(compressed.original_count, 1);
        }
    }
}
