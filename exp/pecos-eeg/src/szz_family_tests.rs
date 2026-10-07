// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Forward conjugation, reference states and backward walks agree on the
//! square roots of two-qubit Paulis (issue #1012).

use crate::circuit::analyze_with_noise;
use crate::dem_mapping::{Detector, EegConfig, build_dem_configured};
use crate::eeg::EegType;
use crate::expand::make_gate;
use crate::heisenberg::{SparsePauli, heisenberg_detection_probability, sparse_conjugate};
use crate::stabilizer::StabilizerGroup;
use crate::{Bm, NoiseInjection, NoiseSpec};
use pecos_core::Gate;
use pecos_core::gate_type::GateType;
use pecos_core::pauli::pauli_bitmask::{self, Conjugated};

const GATES: [(GateType, GateType); 6] = [
    (GateType::SZZ, GateType::SZZdg),
    (GateType::SZZdg, GateType::SZZ),
    (GateType::SXX, GateType::SXXdg),
    (GateType::SXXdg, GateType::SXX),
    (GateType::SYY, GateType::SYYdg),
    (GateType::SYYdg, GateType::SYY),
];

fn forward(p: &Bm, gate: GateType, a: usize, b: usize) -> Conjugated<smallvec::SmallVec<[u64; 8]>> {
    match gate {
        GateType::SZZ => pauli_bitmask::conjugate_szz(p, a, b),
        GateType::SZZdg => pauli_bitmask::conjugate_szzdg(p, a, b),
        GateType::SXX => pauli_bitmask::conjugate_sxx(p, a, b),
        GateType::SXXdg => pauli_bitmask::conjugate_sxxdg(p, a, b),
        GateType::SYY => pauli_bitmask::conjugate_syy(p, a, b),
        GateType::SYYdg => pauli_bitmask::conjugate_syydg(p, a, b),
        other => panic!("unexpected test gate {other:?}"),
    }
}

fn paulis(a: usize, b: usize) -> Vec<Bm> {
    let local = |q| [Bm::default(), Bm::x(q), Bm::y(q), Bm::z(q)];
    local(a)
        .into_iter()
        .flat_map(|p| local(b).map(|q| p.multiply(&q)))
        .collect()
}

struct Injection {
    label: Bm,
    kind: EegType,
}

impl NoiseSpec for Injection {
    fn noise_after_gate(&self, index: usize, _: GateType, _: &[usize]) -> Vec<NoiseInjection> {
        if index != 0 {
            return vec![];
        }
        vec![NoiseInjection {
            eeg_type: self.kind,
            label: self.label.clone(),
            label2: None,
            rate: 0.125,
        }]
    }
}

#[test]
fn szz_family_forward_injections_and_adjoint_walks() {
    for (gate, adjoint) in GATES {
        for [a, b] in [[0, 1], [65, 2]] {
            let gates = [make_gate(GateType::I, &[a]), make_gate(gate, &[a, b])];
            let mut nontrivial = 0;
            for p in paulis(a, b) {
                let expected = forward(&p, gate, a, b);
                nontrivial += usize::from(expected.label != p);
                let noise = Injection {
                    label: p.clone(),
                    kind: EegType::H,
                };
                let result = analyze_with_noise(&gates, &noise, &[false; 2]);
                assert_eq!(result.generators.len(), 1);
                assert_eq!(result.generators[0].label, expected.label, "{gate:?} {p:?}");
                let sign = if expected.sign_negative { -1.0 } else { 1.0 };
                assert!(
                    (result.generators[0].coeff - sign * 0.125).abs() < 1e-12,
                    "{gate:?} {p:?}"
                );
                let mut sparse = SparsePauli::from_bm(&p);
                let negative = sparse_conjugate(&mut sparse, &make_gate(adjoint, &[a, b])).unwrap();
                assert_eq!(sparse.to_bm(), expected.label, "{gate:?} {p:?}");
                assert_eq!(negative, expected.sign_negative, "{gate:?} {p:?}");
            }
            assert!(nontrivial > 0, "{gate:?}");
        }
    }
}

#[test]
fn szz_family_stabilizers_include_y_inputs_and_batched_pairs() {
    for (gate, _) in GATES {
        for y_input in [false, true] {
            let mut preparation = vec![Gate::pz(&[0, 1, 2, 65]), Gate::h(&[0, 65])];
            if y_input {
                preparation.push(Gate::sz(&[0, 65]));
            }
            preparation.push(make_gate(gate, &[0, 1, 65, 2]));
            let group = StabilizerGroup::from_circuit(&preparation, 66);
            for [a, b] in [[0, 1], [65, 2]] {
                let first = if y_input { Bm::y(a) } else { Bm::x(a) };
                let members = [
                    Bm::default(),
                    first.clone(),
                    Bm::z(b),
                    first.multiply(&Bm::z(b)),
                ];
                let expected: Vec<_> = members.iter().map(|p| forward(p, gate, a, b)).collect();
                for p in paulis(a, b) {
                    let sign = expected
                        .iter()
                        .find(|r| r.label == p)
                        .map(|r| !r.sign_negative);
                    assert_eq!(group.is_stabilizer(&p), sign, "{gate:?} y={y_input} {p:?}");
                }
            }
        }
    }
}

#[test]
fn szz_family_reversed_operands_reproducer() {
    let gates = [
        Gate::pz(&[0, 1]),
        make_gate(GateType::SZZ, &[1, 0]),
        make_gate(GateType::SXX, &[1, 0]),
    ];
    let group = StabilizerGroup::from_circuit(&gates, 2);
    let expected: Vec<_> = [
        Bm::default(),
        Bm::z(0),
        Bm::z(1),
        Bm::z(0).multiply(&Bm::z(1)),
    ]
    .iter()
    .map(|p| {
        let first = forward(p, GateType::SZZ, 1, 0);
        let mut second = forward(&first.label, GateType::SXX, 1, 0);
        second.sign_negative ^= first.sign_negative;
        second
    })
    .collect();
    for p in paulis(0, 1) {
        assert_eq!(
            group.is_stabilizer(&p),
            expected
                .iter()
                .find(|r| r.label == p)
                .map(|r| !r.sign_negative),
            "{p:?}"
        );
    }
    let noise = Injection {
        label: Bm::y(0),
        kind: EegType::H,
    };
    let result = analyze_with_noise(&gates, &noise, &[false; 3]);
    let first = forward(&noise.label, GateType::SZZ, 1, 0);
    let second = forward(&first.label, GateType::SXX, 1, 0);
    assert_eq!(result.generators[0].label, second.label);
    let sign = if first.sign_negative ^ second.sign_negative {
        -1.0
    } else {
        1.0
    };
    assert!((result.generators[0].coeff - sign * 0.125).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "EEG forward: unsupported gate type F")]
fn forward_rejects_unsupported_clifford() {
    let gates = [make_gate(GateType::I, &[0]), make_gate(GateType::F, &[0])];
    let noise = Injection {
        label: Bm::x(0),
        kind: EegType::H,
    };
    analyze_with_noise(&gates, &noise, &[false; 2]);
}

#[test]
#[should_panic(expected = "EEG stabilizer: unsupported gate type F")]
fn stabilizer_rejects_unsupported_clifford() {
    let _ = StabilizerGroup::from_circuit(&[make_gate(GateType::F, &[0])], 1);
}

fn noop_gate(gate: GateType) -> Gate {
    if gate == GateType::RZ {
        Gate::rz(pecos_core::Angle64::from_radians(0.01), &[0])
    } else {
        make_gate(gate, &[0])
    }
}

#[test]
fn forward_listed_noops_preserve_injections() {
    for gate in [
        GateType::MZ,
        GateType::MeasureFree,
        GateType::I,
        GateType::Idle,
        GateType::QFree,
        GateType::RZ,
    ] {
        let gates = [make_gate(GateType::I, &[0]), noop_gate(gate)];
        let noise = Injection {
            label: Bm::y(0),
            kind: EegType::H,
        };
        let result = analyze_with_noise(&gates, &noise, &[false; 2]);
        assert_eq!(
            result.generators.len(),
            1 + usize::from(gate == GateType::RZ),
            "{gate:?}"
        );
        assert_eq!(result.generators[0].label, noise.label, "{gate:?}");
        assert!(
            (result.generators[0].coeff - 0.125).abs() < 1e-12,
            "{gate:?}"
        );
    }
}

#[test]
fn stabilizer_listed_noops_preserve_preparation() {
    let preparation = [
        Gate::pz(&[0, 1]),
        Gate::h(&[0]),
        Gate::sz(&[0]),
        Gate::cx(&[(0, 1)]),
    ];
    let plain = StabilizerGroup::from_circuit(&preparation, 2);
    let mut interleaved = vec![];
    for (gate, noop) in
        preparation
            .into_iter()
            .zip([GateType::I, GateType::Idle, GateType::QFree, GateType::RZ])
    {
        interleaved.push(gate);
        interleaved.push(noop_gate(noop));
    }
    let group = StabilizerGroup::from_circuit(&interleaved, 2);
    for p in paulis(0, 1) {
        assert_eq!(group.is_stabilizer(&p), plain.is_stabilizer(&p));
    }
}

#[test]
fn szz_family_forward_detection_matches_walk() {
    // SZZ maps Y0 to -X0 Z1, which flips the final Y0 Z1 detector.
    // Ignoring SZZ leaves Y0 unchanged and incorrectly predicts no detection.
    let gates = [
        make_gate(GateType::I, &[0]),
        make_gate(GateType::SZZ, &[0, 1]),
    ];
    let preparation = [Gate::pz(&[0, 1]), Gate::h(&[0])];
    let initial = StabilizerGroup::from_circuit(&preparation, 2);
    let mut noiseless = preparation.to_vec();
    noiseless.extend_from_slice(&gates);
    let reference = StabilizerGroup::from_circuit(&noiseless, 2);
    let noise = Injection {
        label: Bm::y(0),
        kind: EegType::H,
    };
    // The final noiseless stabilizer is Y0 Z1, obtained from X0 by SZZ.
    let detector = forward(&Bm::x(0), GateType::SZZ, 0, 1).label;
    let result = analyze_with_noise(&gates, &noise, &[false; 2]);
    let entries = build_dem_configured(
        &result.generators,
        &[Detector {
            id: 0,
            stabilizer: detector.clone(),
        }],
        &[],
        Some(&reference),
        &EegConfig::new().sin_squared(),
    );
    let forward_rate: f64 = entries.iter().map(|entry| entry.probability).sum();
    let walk =
        heisenberg_detection_probability(&gates, &detector, &noise, &initial, 0.0, &[false; 2]);
    assert!((walk - 0.125_f64.sin().powi(2)).abs() < 1e-12);
    assert!(
        (forward_rate - walk).abs() < 1e-12,
        "forward={forward_rate} walk={walk}"
    );
}

#[test]
fn forward_rejects_other_resets_and_measurements() {
    for gate in [GateType::PX, GateType::MX, GateType::MPZ, GateType::Fdg] {
        let gates = [make_gate(GateType::I, &[0]), make_gate(gate, &[0])];
        let noise = Injection {
            label: Bm::x(0),
            kind: EegType::H,
        };
        assert!(
            std::panic::catch_unwind(|| analyze_with_noise(&gates, &noise, &[false; 2])).is_err(),
            "{gate:?}"
        );
    }
}

#[test]
fn stabilizer_rejects_other_resets_measurements_and_metadata() {
    for gate in [
        GateType::PX,
        GateType::MX,
        GateType::MPZ,
        GateType::MeasureFree,
        GateType::Fdg,
    ] {
        assert!(
            std::panic::catch_unwind(|| StabilizerGroup::from_circuit(&[make_gate(gate, &[0])], 1))
                .is_err(),
            "{gate:?}"
        );
    }
}

#[test]
fn tracked_pauli_marker_is_a_noop_everywhere() {
    // `TickCircuit` inserts a `TrackedPauliMeta` gate on the tracked qubits;
    // the EEG builder ignores tracked-Pauli annotations, so the marker must
    // not change a label, the reference state, or a walk.
    let marker = make_gate(GateType::TrackedPauliMeta, &[0]);

    let gates = [make_gate(GateType::I, &[0]), marker.clone()];
    let noise = Injection {
        label: Bm::y(0),
        kind: EegType::H,
    };
    let result = analyze_with_noise(&gates, &noise, &[false; 2]);
    assert_eq!(result.generators.len(), 1);
    assert_eq!(result.generators[0].label, noise.label);

    let preparation = [Gate::pz(&[0, 1]), Gate::h(&[0]), Gate::cx(&[(0, 1)])];
    let plain = StabilizerGroup::from_circuit(&preparation, 2);
    let mut marked = preparation.to_vec();
    marked.insert(2, marker.clone());
    let group = StabilizerGroup::from_circuit(&marked, 2);
    for p in paulis(0, 1) {
        assert_eq!(group.is_stabilizer(&p), plain.is_stabilizer(&p));
    }

    let mut label = SparsePauli::from_bm(&Bm::x(0));
    assert_eq!(sparse_conjugate(&mut label, &marker), None);
    assert_eq!(label.to_bm(), Bm::x(0));
}
