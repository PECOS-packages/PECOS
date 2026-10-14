// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Analytic regressions for categorical depolarizing channels (issue #943).

use pecos_core::Gate;
use pecos_core::gate_type::GateType;
use pecos_eeg::expand::GateIndex;
use pecos_eeg::heisenberg::{
    build_noise_map, heisenberg_detection_probability, heisenberg_sparse, heisenberg_with_noise_map,
};
use pecos_eeg::stabilizer::StabilizerGroup;
use pecos_eeg::{Bm, GateNoise, NoiseInjection, NoiseSpec, UniformNoise};

/// Check every walker against a channel eigenvalue derived analytically.
/// Comparing walkers with one another would preserve their shared model bug.
fn assert_detection_probability(
    gates: &[Gate],
    detector: &Bm,
    noise: &dyn NoiseSpec,
    num_qubits: usize,
    expected: f64,
) {
    let initial = StabilizerGroup::from_circuit(
        &[Gate::pz(&(0..num_qubits).collect::<Vec<_>>())],
        num_qubits,
    );
    assert_walkers(gates, detector, noise, &initial, num_qubits, expected);
}

fn assert_walkers(
    gates: &[Gate],
    detector: &Bm,
    noise: &dyn NoiseSpec,
    initial: &StabilizerGroup,
    num_qubits: usize,
    expected: f64,
) {
    let index = GateIndex::build(gates, num_qubits, noise, &vec![false; gates.len()]);
    let noise_map = build_noise_map(gates, noise, &index.expansion_gates);
    let results = [
        (
            "windowed",
            heisenberg_detection_probability(
                gates,
                detector,
                noise,
                initial,
                0.0,
                &vec![false; gates.len()],
            ),
        ),
        (
            "precomputed",
            heisenberg_with_noise_map(gates, detector, &noise_map, initial, 0.0),
        ),
        (
            "sparse",
            heisenberg_sparse(gates, detector, noise, initial, 0.0, &index, None),
        ),
        (
            "sparse precomputed",
            heisenberg_sparse(
                gates,
                detector,
                noise,
                initial,
                0.0,
                &index,
                Some(&noise_map),
            ),
        ),
    ];
    assert!(
        results
            .iter()
            .all(|(_, actual)| (actual - expected).abs() < 1e-12),
        "expected {expected}; walkers: {results:?}",
    );
}

#[test]
fn single_qubit_depolarizing_has_categorical_eigenvalue() {
    // Z preserves |0>; one depolarizing location attenuates Z by 1-4p/3.
    let gates = [Gate::pz(&[0]), Gate::z(&[0]), Gate::mz(&[0])];
    for p in [0.0, 0.01, 0.1, 0.3, 0.75, 1.0] {
        let noise = UniformNoise {
            p1: p,
            ..UniformNoise::coherent_only(0.0)
        };
        assert_detection_probability(&gates, &Bm::z(0), &noise, 1, 2.0 * p / 3.0);
    }
}

#[test]
fn two_hadamards_compose_categorical_depolarizing_channels() {
    // Issue reproducer: two locations have eigenvalue lambda=1-4p/3.
    // The backward Pauli is only X or Z, avoiding the separate Y/Y bug #942.
    let gates = [Gate::pz(&[0]), Gate::h(&[0]), Gate::h(&[0]), Gate::mz(&[0])];
    for p in [0.0_f64, 0.01, 0.1, 0.3, 0.75, 1.0] {
        let noise = UniformNoise {
            p1: p,
            ..UniformNoise::coherent_only(0.0)
        };
        let expected = (1.0 - (1.0 - 4.0 * p / 3.0).powi(2)) / 2.0;
        assert_detection_probability(&gates, &Bm::z(0), &noise, 1, expected);
    }
}

#[test]
fn fully_depolarizing_single_qubit_erases_the_detector_bias() {
    let gates = [Gate::pz(&[0]), Gate::h(&[0]), Gate::h(&[0]), Gate::mz(&[0])];
    let noise = UniformNoise {
        p1: 0.75,
        ..UniformNoise::coherent_only(0.0)
    };
    assert_detection_probability(&gates, &Bm::z(0), &noise, 1, 0.5);
}

#[test]
fn two_qubit_depolarizing_has_categorical_eigenvalue() {
    // CX preserves |00>. Each nontrivial Z-type term anticommutes with
    // eight of the fifteen mutually exclusive errors, giving 1-16p/15.
    let gates = [Gate::pz(&[0, 1]), Gate::cx(&[(0, 1)]), Gate::mz(&[0, 1])];
    for p in [0.0, 0.01, 0.1, 0.3, 0.9375, 1.0] {
        let noise = UniformNoise {
            p2: p,
            ..UniformNoise::coherent_only(0.0)
        };
        for detector in [Bm::z(0), Bm::z(1), Bm::z(0).multiply(&Bm::z(1))] {
            assert_detection_probability(&gates, &detector, &noise, 2, 8.0 * p / 15.0);
        }
    }
}

/// Place `UniformNoise`'s channels at an identity gate so every Pauli can be
/// tested without depending on a Clifford conjugation or measurement basis.
struct NoiseAtIdentity {
    noise: UniformNoise,
    source_gate: GateType,
}

impl NoiseSpec for NoiseAtIdentity {
    fn noise_after_gate(&self, index: usize, _: GateType, qubits: &[usize]) -> Vec<NoiseInjection> {
        self.noise.noise_after_gate(index, self.source_gate, qubits)
    }

    fn exact_noise_after_gate(&self, index: usize, _: GateType, qubits: &[usize]) -> GateNoise {
        self.noise
            .exact_noise_after_gate(index, self.source_gate, qubits)
    }
}

#[test]
fn categorical_channels_have_the_analytic_eigenvalue_for_every_pauli() {
    for num_qubits in [1, 2] {
        let qubits: Vec<_> = (0..num_qubits).collect();
        let gates = [pecos_eeg::expand::make_gate(GateType::I, &qubits)];
        for p in [0.0, 0.1, 0.75, 0.9375, 1.0] {
            let noise = NoiseAtIdentity {
                noise: UniformNoise {
                    p1: p,
                    p2: p,
                    ..UniformNoise::coherent_only(0.0)
                },
                source_gate: if num_qubits == 1 {
                    GateType::H
                } else {
                    GateType::CX
                },
            };
            for label in 0..(1 << (2 * num_qubits)) {
                let mut detector = Bm::default();
                let mut preparation = vec![Gate::pz(&qubits)];
                for q in 0..num_qubits {
                    let axis = (label >> (2 * q)) & 3;
                    let pauli = match axis {
                        1 => Bm::x(q),
                        2 => Bm::y(q),
                        3 => Bm::z(q),
                        _ => Bm::default(),
                    };
                    detector = detector.multiply(&pauli);
                    if axis == 1 || axis == 2 {
                        preparation.push(Gate::h(&[q]));
                    }
                    if axis == 2 {
                        preparation.push(pecos_eeg::expand::make_gate(GateType::SZ, &[q]));
                    }
                }
                let initial = StabilizerGroup::from_circuit(&preparation, num_qubits);
                let expected = if label == 0 {
                    0.0
                } else if num_qubits == 1 {
                    2.0 * p / 3.0
                } else {
                    8.0 * p / 15.0
                };
                assert_walkers(&gates, &detector, &noise, &initial, num_qubits, expected);
            }
        }
    }
}

#[test]
fn categorical_noise_covers_every_operand_in_gate_batches() {
    // Z and CX preserve these |0000> observables. This isolates noise-channel
    // batching from the separate question of batched Clifford conjugation.
    for p in [0.1_f64, 0.75, 0.9375, 1.0] {
        for (gate, noise, eigenvalue) in [
            (
                Gate::z(&[0, 1, 2, 3]),
                UniformNoise {
                    p1: p,
                    ..UniformNoise::coherent_only(0.0)
                },
                1.0 - 4.0 * p / 3.0,
            ),
            (
                Gate::cx(&[(0, 1), (2, 3)]),
                UniformNoise {
                    p2: p,
                    ..UniformNoise::coherent_only(0.0)
                },
                1.0 - 16.0 * p / 15.0,
            ),
        ] {
            let gates = [Gate::pz(&[0, 1, 2, 3]), gate, Gate::mz(&[0, 1, 2, 3])];
            assert_detection_probability(&gates, &Bm::z(3), &noise, 4, (1.0 - eigenvalue) / 2.0);
            assert_detection_probability(
                &gates,
                &Bm::z(1).multiply(&Bm::z(3)),
                &noise,
                4,
                (1.0 - eigenvalue.powi(2)) / 2.0,
            );
            assert_detection_probability(&gates, &Bm::default(), &noise, 4, 0.0);
        }
    }
}

struct IndependentInjections(Vec<NoiseInjection>);

impl NoiseSpec for IndependentInjections {
    fn noise_after_gate(&self, _: usize, _: GateType, _: &[usize]) -> Vec<NoiseInjection> {
        self.0.clone()
    }
}

fn injection(eeg_type: pecos_eeg::eeg::EegType, label: Bm, rate: f64) -> NoiseInjection {
    NoiseInjection {
        eeg_type,
        label,
        label2: None,
        rate,
    }
}

#[test]
fn equal_rate_custom_s_injections_remain_independent() {
    use pecos_eeg::eeg::EegType;
    let gates = [pecos_eeg::expand::make_gate(GateType::I, &[0])];
    let p = 0.2_f64;
    for (labels, anticommuting) in [
        (vec![Bm::x(0), Bm::y(0), Bm::z(0)], 2),
        (vec![Bm::x(0); 3], 3),
        (vec![Bm::x(0); 15], 15),
    ] {
        let noise = IndependentInjections(
            labels
                .into_iter()
                .map(|pauli| injection(EegType::S, pauli, -p))
                .collect(),
        );
        let expected = (1.0 - (1.0 - 2.0 * p).powi(anticommuting)) / 2.0;
        assert_detection_probability(&gates, &Bm::z(0), &noise, 1, expected);
    }
}

#[test]
fn single_qubit_custom_s_set_on_a_two_qubit_gate_leaves_the_other_qubit() {
    // Issue #995: X/Y/Z on q0 of a two-qubit gate were once batched as if
    // they were two-qubit depolarizing, attenuating terms supported on q1.
    use pecos_eeg::eeg::EegType;
    let gates = [pecos_eeg::expand::make_gate(GateType::I, &[0, 1])];
    let noise = IndependentInjections(
        [Bm::x(0), Bm::y(0), Bm::z(0)]
            .into_iter()
            .map(|pauli| injection(EegType::S, pauli, -0.1))
            .collect(),
    );
    assert_detection_probability(&gates, &Bm::z(1), &noise, 2, 0.0);
}

#[test]
fn custom_injection_order_is_preserved_in_noise_maps() {
    use pecos_eeg::eeg::EegType;
    let theta = 0.37_f64;
    let p = 0.2;
    let noise = IndependentInjections(vec![
        injection(EegType::S, Bm::x(0), -p),
        injection(EegType::H, Bm::z(0), theta / 2.0),
    ]);
    let gates = [pecos_eeg::expand::make_gate(GateType::I, &[0])];
    let initial = StabilizerGroup::from_circuit(&[Gate::pz(&[0]), Gate::h(&[0])], 1);
    // Forward S_X leaves |+> unchanged in both Pauli branches, then RZ(theta)
    // rotates X toward Y, giving <Y> = sin(theta), independent of p.
    let expected = (1.0 - theta.sin()) / 2.0;
    assert_walkers(&gates, &Bm::y(0), &noise, &initial, 1, expected);
}

#[test]
fn coherent_and_categorical_noise_match_bell_parity_formula() {
    let gates = [Gate::pz(&[0, 1]), Gate::h(&[0]), Gate::cx(&[(0, 1)])];
    let detector = Bm::x(0).multiply(&Bm::x(1));
    for theta in [0.0_f64, 0.17, 0.6] {
        for p1 in [0.0, 0.3, 0.9] {
            for p2 in [0.0, 0.4, 0.9375, 1.0] {
                let noise = UniformNoise {
                    idle_rz: theta,
                    p1,
                    p2,
                    p_meas: 0.0,
                    p_prep: 0.0,
                };
                // RZ(theta) on both Bell qubits gives <XX>=cos(2theta).
                // Each categorical location multiplies that Pauli expectation
                // by its analytic eigenvalue, including negative eigenvalues.
                let expected = (1.0
                    - (1.0 - 4.0 * p1 / 3.0) * (1.0 - 16.0 * p2 / 15.0) * (2.0 * theta).cos())
                    / 2.0;
                assert_detection_probability(&gates, &detector, &noise, 2, expected);
            }
        }
    }
}

#[test]
fn compressed_mechanism_structure_retains_exact_categorical_targets() {
    use pecos_eeg::dem_mapping::Detector;
    use pecos_eeg::noise_characterization::{NoiseCharacterization, NoiseCharacterizationInput};
    use pecos_eeg::noise_compression::{CompressedNoiseSpec, compress_noise_to_boundaries};

    let gates = [Gate::pz(&[0]), Gate::h(&[0]), Gate::h(&[0]), Gate::mz(&[0])];
    let noise = UniformNoise {
        p1: 0.75,
        ..UniformNoise::coherent_only(0.0)
    };
    let expansion_gates = vec![false; gates.len()];
    let compressed = compress_noise_to_boundaries(&gates, &noise, &expansion_gates);
    assert!(compressed.compressed_count < compressed.original_count);
    let structure = CompressedNoiseSpec::from_compressed(&compressed);
    let detectors = [Detector {
        id: 0,
        stabilizer: Bm::z(0),
    }];
    let initial = StabilizerGroup::from_circuit(&[Gate::pz(&[0])], 1);
    for structure_noise in [None, Some(&structure as &dyn NoiseSpec)] {
        let characterization = NoiseCharacterization::build(NoiseCharacterizationInput {
            gates: &gates,
            expansion_gates: &expansion_gates,
            noise: &noise,
            structure_noise,
            detectors: &detectors,
            observables: &[],
            initial_stab: &initial,
            num_qubits: 1,
            max_order: 1,
            prune_threshold: 0.0,
            detector_meas_ids: &[],
            observable_meas_ids: &[],
        });
        assert_eq!(characterization.correlations.len(), 1);
        assert_eq!(characterization.correlations[0].labels, ["D0"]);
        assert!((characterization.correlations[0].probability - 0.5).abs() < 1e-12);
    }
}

#[test]
fn independent_depolarizing_flips_have_categorical_eigenvalues() {
    for (gate_type, qubits, boundary, anticommuting) in [
        (GateType::H, vec![0], 0.75, 2),
        (GateType::CX, vec![0, 1], 0.9375, 8),
    ] {
        for p in [1e-20, 1e-6, 0.1, 0.3, boundary - 1e-6, boundary] {
            let noise = UniformNoise {
                p1: if qubits.len() == 1 { p } else { 0.0 },
                p2: if qubits.len() == 2 { p } else { 0.0 },
                ..UniformNoise::coherent_only(0.0)
            };
            let injections = noise.noise_after_gate(0, gate_type, &qubits);
            assert_eq!(injections.len(), (1 << (2 * qubits.len())) - 1);
            let expected = 1.0 - p / boundary;
            for label in 1..(1 << (2 * qubits.len())) {
                let mut pauli = Bm::default();
                for &q in &qubits {
                    pauli = pauli.multiply(&match (label >> (2 * q)) & 3 {
                        1 => Bm::x(q),
                        2 => Bm::y(q),
                        3 => Bm::z(q),
                        _ => Bm::default(),
                    });
                }
                let factors: Vec<_> = injections
                    .iter()
                    .filter(|inj| !inj.label.commutes_with(&pauli))
                    .map(|inj| 1.0 + 2.0 * inj.rate)
                    .collect();
                assert_eq!(factors.len(), anticommuting);
                let actual: f64 = factors.iter().product();
                // At the boundary require exact zero; otherwise relative 1e-12.
                assert!(
                    (actual - expected).abs() <= 1e-12 * expected.abs(),
                    "{gate_type:?}, p={p}, Pauli={pauli:?}: {actual} != {expected}"
                );
            }
            if p < 1e-16 {
                // Eigenvalues round to one here; check the small-p asymptote
                // q = p/(4^n - 1) to catch cancellation in the rates themselves.
                let leading_rate = -p / f64::from(if qubits.len() == 1 { 3 } else { 15 });
                for inj in &injections {
                    assert!((inj.rate / leading_rate - 1.0).abs() < 1e-12);
                }
            }
        }
    }
}

struct UniformInjections(UniformNoise);

impl NoiseSpec for UniformInjections {
    fn noise_after_gate(
        &self,
        index: usize,
        gate: GateType,
        qubits: &[usize],
    ) -> Vec<NoiseInjection> {
        self.0.noise_after_gate(index, gate, qubits)
    }

    fn exact_noise_after_gate(&self, index: usize, gate: GateType, qubits: &[usize]) -> GateNoise {
        GateNoise {
            injections: self.noise_after_gate(index, gate, qubits),
            depolarizing: Vec::new(),
        }
    }
}

#[test]
fn independent_depolarizing_flips_match_channel_walks() {
    // Overlapping H and CX noise locations prepare a noisy Bell state.
    // RZ on both Bell qubits gives <XX> = lambda1 * lambda2 * cos(2*theta).
    let gates = [Gate::pz(&[0, 1]), Gate::h(&[0]), Gate::cx(&[(0, 1)])];
    let detector = Bm::x(0).multiply(&Bm::x(1));
    for p1 in [1e-6, 0.3, 0.75] {
        for p2 in [1e-6, 0.4, 0.9375] {
            for theta in [0.0_f64, 0.17] {
                let channels = UniformNoise {
                    p1,
                    p2,
                    ..UniformNoise::coherent_only(theta)
                };
                let initial = StabilizerGroup::from_circuit(&[Gate::pz(&[0, 1])], 2);
                let channel_probability = heisenberg_detection_probability(
                    &gates,
                    &detector,
                    &channels,
                    &initial,
                    0.0,
                    &[false; 3],
                );
                let analytic = (1.0
                    - (1.0 - 4.0 * p1 / 3.0) * (1.0 - 16.0 * p2 / 15.0) * (2.0 * theta).cos())
                    / 2.0;
                assert!((channel_probability - analytic).abs() < 1e-12);
                assert_walkers(
                    &gates,
                    &detector,
                    &UniformInjections(channels),
                    &initial,
                    2,
                    channel_probability,
                );
            }
        }
    }
}

#[test]
#[should_panic(expected = "generator view requires p1 <= 3/4")]
fn independent_depolarizing_rejects_p1_above_boundary() {
    UniformNoise {
        p1: 0.75_f64.next_up(),
        ..UniformNoise::coherent_only(0.0)
    }
    .noise_after_gate(0, GateType::H, &[0]);
}

#[test]
#[should_panic(expected = "generator view requires p2 <= 15/16")]
fn independent_depolarizing_rejects_p2_above_boundary() {
    UniformNoise {
        p2: 0.9375_f64.next_up(),
        ..UniformNoise::coherent_only(0.0)
    }
    .noise_after_gate(0, GateType::CX, &[0, 1]);
}

#[test]
fn independent_depolarizing_boundaries_and_nonpositive_rates() {
    for p in [-0.1, 0.0] {
        let noise = UniformNoise::depolarizing(p);
        assert!(noise.noise_after_gate(0, GateType::H, &[0]).is_empty());
        assert!(noise.noise_after_gate(0, GateType::CX, &[0, 1]).is_empty());
    }
    let noise = UniformNoise {
        p1: 0.75,
        p2: 0.9375,
        ..UniformNoise::coherent_only(0.0)
    };
    for (gate, qubits) in [(GateType::H, vec![0]), (GateType::CX, vec![0, 1])] {
        for injection in noise.noise_after_gate(0, gate, &qubits) {
            assert_eq!(injection.rate, -0.5);
        }
        for p in [0.75, 0.9375, 1.0] {
            let channels = UniformNoise::depolarizing(p).exact_noise_after_gate(0, gate, &qubits);
            assert_eq!(channels.depolarizing.len(), 1);
            assert!(channels.injections.is_empty());
        }
    }
}
