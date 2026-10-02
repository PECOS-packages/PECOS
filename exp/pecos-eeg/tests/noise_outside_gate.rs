// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Noise whose support lies outside its gate's qubits, such as crosstalk onto
//! a neighbour (issue #997). The walks must decide relevance by the noise's own
//! support, not by the gate's.

use pecos_core::Gate;
use pecos_core::gate_type::GateType;
use pecos_eeg::eeg::EegType;
use pecos_eeg::expand::{GateIndex, expand_circuit, make_gate};
use pecos_eeg::heisenberg::{
    build_noise_map, heisenberg_detection_probability, heisenberg_exact_from_circuit,
    heisenberg_sparse, heisenberg_with_noise_map,
};
use pecos_eeg::stabilizer::StabilizerGroup;
use pecos_eeg::{Bm, DepolarizingChannel, GateNoise, NoiseInjection, NoiseSpec};

/// Run every Pauli-tracking walk and return each result by name.
fn walks(
    gates: &[Gate],
    detector: &Bm,
    noise: &dyn NoiseSpec,
    initial: &StabilizerGroup,
    num_qubits: usize,
    prune: f64,
) -> [(&'static str, f64); 4] {
    let index = GateIndex::build(gates, num_qubits, noise);
    let noise_map = build_noise_map(gates, noise, &index.expansion_gates);
    [
        (
            "windowed",
            heisenberg_detection_probability(gates, detector, noise, initial, prune),
        ),
        (
            "precomputed",
            heisenberg_with_noise_map(gates, detector, &noise_map, initial, prune),
        ),
        (
            "sparse",
            heisenberg_sparse(gates, detector, noise, initial, prune, &index, None),
        ),
        (
            "sparse precomputed",
            heisenberg_sparse(
                gates,
                detector,
                noise,
                initial,
                prune,
                &index,
                Some(&noise_map),
            ),
        ),
    ]
}

fn assert_walks(
    gates: &[Gate],
    detector: &Bm,
    noise: &dyn NoiseSpec,
    initial: &StabilizerGroup,
    num_qubits: usize,
    expected: f64,
) {
    let results = walks(gates, detector, noise, initial, num_qubits, 0.0);
    assert!(
        results
            .iter()
            .all(|(_, actual)| (actual - expected).abs() < 1e-12),
        "expected {expected}; walks: {results:?}",
    );
}

/// The given exact noise after gate `at`, nothing elsewhere.
struct NoiseAt {
    at: usize,
    noise: GateNoise,
}

impl NoiseSpec for NoiseAt {
    fn noise_after_gate(&self, i: usize, _: GateType, _: &[usize]) -> Vec<NoiseInjection> {
        if i == self.at {
            self.noise.injections.clone()
        } else {
            Vec::new()
        }
    }

    fn exact_noise_after_gate(&self, i: usize, _: GateType, _: &[usize]) -> GateNoise {
        if i == self.at {
            self.noise.clone()
        } else {
            GateNoise::default()
        }
    }
}

fn injection(eeg_type: EegType, label: Bm, rate: f64) -> NoiseInjection {
    NoiseInjection {
        eeg_type,
        label,
        label2: None,
        rate,
    }
}

/// Noise on qubit 1 after an identity gate on qubit 0. No gate touches qubit 1,
/// so this also covers active-qubit bitmaps sized from gate qubits alone.
fn noise_on_qubit_one_after_gate_on_qubit_zero(noise: GateNoise, expected: f64) {
    let gates = [make_gate(GateType::I, &[0])];
    let initial = StabilizerGroup::from_circuit(&[Gate::pz(&[0, 1])], 2);
    let noise = NoiseAt { at: 0, noise };
    assert_walks(&gates, &Bm::z(1), &noise, &initial, 2, expected);
    // A detector that also covers the gate's qubit must see the same flip.
    let detector = Bm::z(0).multiply(&Bm::z(1));
    assert_walks(&gates, &detector, &noise, &initial, 2, expected);
}

#[test]
fn stochastic_injection_outside_the_gate() {
    let p = 0.1;
    noise_on_qubit_one_after_gate_on_qubit_zero(
        GateNoise {
            injections: vec![injection(EegType::S, Bm::x(1), -p)],
            depolarizing: Vec::new(),
        },
        p,
    );
}

#[test]
fn coherent_injection_outside_the_gate() {
    // exp(-i h X1) on |0> flips qubit 1 with probability sin²h.
    let h = 0.3_f64;
    noise_on_qubit_one_after_gate_on_qubit_zero(
        GateNoise {
            injections: vec![injection(EegType::H, Bm::x(1), h)],
            depolarizing: Vec::new(),
        },
        h.sin().powi(2),
    );
}

#[test]
fn depolarizing_channel_outside_the_gate() {
    // Z1 anticommutes with two of the three exclusive Paulis.
    let p = 0.1;
    noise_on_qubit_one_after_gate_on_qubit_zero(
        GateNoise {
            injections: Vec::new(),
            depolarizing: vec![DepolarizingChannel::OneQubit {
                qubit: 1,
                probability: p,
            }],
        },
        2.0 * p / 3.0,
    );
}

/// Coherent injections after gates 2-8 of the issue #997 circuit.
struct Issue997Noise;

impl NoiseSpec for Issue997Noise {
    fn noise_after_gate(&self, i: usize, _: GateType, _: &[usize]) -> Vec<NoiseInjection> {
        let (label, rate) = match i {
            2 => (Bm::y(0), 0.23),
            3 => (Bm::x(1), -0.31),
            4 => (Bm::z(0), 0.41),
            5 => (Bm::y(0).multiply(&Bm::z(1)), -0.19),
            6 => (Bm::x(0).multiply(&Bm::x(1)), 0.27),
            7 => (Bm::z(1), -0.37),
            8 => (Bm::y(0).multiply(&Bm::y(1)), 0.17),
            _ => return Vec::new(),
        };
        vec![injection(EegType::H, label, rate)]
    }
}

#[test]
fn issue_997_walks_match_the_matrix_reference() {
    // The Y0Y1 injection follows the final H1. The backward term there is on
    // q0 only, so a walk that judges relevance by the gate's qubits skips it,
    // and the term it branches onto q1 must still be conjugated by H1.
    let gates = [
        Gate::pz(&[0]),
        Gate::pz(&[1]),
        Gate::h(&[0]),
        Gate::cx(&[(0, 1)]),
        Gate::h(&[1]),
        Gate::cx(&[(1, 0)]),
        Gate::h(&[0]),
        Gate::cx(&[(0, 1)]),
        Gate::h(&[1]),
        Gate::mz(&[0]),
        Gate::mz(&[1]),
    ];
    let exact = heisenberg_exact_from_circuit(&gates, &[0], &Issue997Noise, 2).unwrap();
    // Value from the issue, confirmed there by a forward state-vector run.
    assert!((exact - 0.280_215).abs() < 1e-6, "matrix reference {exact}");

    let expanded = expand_circuit(&gates).unwrap();
    let mut detector = Bm::default();
    detector = detector.multiply(&Bm::z(expanded.aux_qubit_for_record(0).unwrap()));
    let initial = StabilizerGroup::from_circuit(&[Gate::pz(&[0, 1])], expanded.num_qubits);
    assert_walks(
        &expanded.gates,
        &detector,
        &Issue997Noise,
        &initial,
        expanded.num_qubits,
        exact,
    );
}

/// Exact noise listed per gate index; gates past the list are noiseless.
struct NoisePerGate(Vec<GateNoise>);

impl NoiseSpec for NoisePerGate {
    fn noise_after_gate(&self, i: usize, _: GateType, _: &[usize]) -> Vec<NoiseInjection> {
        self.0
            .get(i)
            .map(|noise| noise.injections.clone())
            .unwrap_or_default()
    }

    fn exact_noise_after_gate(&self, i: usize, _: GateType, _: &[usize]) -> GateNoise {
        self.0.get(i).cloned().unwrap_or_default()
    }
}

fn injections(list: Vec<NoiseInjection>) -> GateNoise {
    GateNoise {
        injections: list,
        depolarizing: Vec::new(),
    }
}

#[test]
fn repeated_noise_outside_the_gate_composes() {
    // Twelve X1 rotations on gates that never touch q1 compose to one
    // rotation by 12h. Each one branches every term, so the walks must keep
    // applying (and merging) noise on gates whose own qubits stay inactive.
    let (count, h) = (12_u32, 0.05_f64);
    let gates: Vec<Gate> = (0..count).map(|_| make_gate(GateType::I, &[0])).collect();
    let noise = NoisePerGate(
        (0..count)
            .map(|_| injections(vec![injection(EegType::H, Bm::x(1), h)]))
            .collect(),
    );
    let initial = StabilizerGroup::from_circuit(&[Gate::pz(&[0, 1])], 2);
    let expected = (f64::from(count) * h).sin().powi(2);
    assert_walks(&gates, &Bm::z(1), &noise, &initial, 2, expected);
}

#[test]
fn pruning_applies_after_noise_outside_the_gate() {
    // S(X1) at p=0.49 leaves Z1 with coefficient 0.02, below the 0.1
    // threshold. Every walk prunes it, so <Z1> = 0 and p = 0.5.
    let gates = [make_gate(GateType::I, &[0])];
    let noise = NoisePerGate(vec![injections(vec![injection(
        EegType::S,
        Bm::x(1),
        -0.49,
    )])]);
    let initial = StabilizerGroup::from_circuit(&[Gate::pz(&[0, 1])], 2);
    let results = walks(&gates, &Bm::z(1), &noise, &initial, 2, 0.1);
    assert!(
        results
            .iter()
            .all(|(_, actual)| (actual - 0.5).abs() < 1e-12),
        "walks: {results:?}",
    );
}

#[test]
fn noise_reaches_a_far_qubit_and_schedules_earlier_noise_through_it() {
    // Gates act on q1, which never becomes active. Backward, the first
    // X0X130 rotation branches the Z0 term onto q130, beyond any gate qubit;
    // the Z130 flip on the middle gate is reachable only through that qubit.
    // Forward: the two rotations compose to sin²(2h), unless the Z130 flip
    // between them (probability p) makes them cancel, so p_det = (1-p) sin²(2h).
    let (h, p) = (0.3_f64, 0.2);
    let far = 130;
    let rotation = || injection(EegType::H, Bm::x(0).multiply(&Bm::x(far)), h);
    let gates: Vec<Gate> = (0..3).map(|_| make_gate(GateType::I, &[1])).collect();
    let noise = NoisePerGate(vec![
        injections(vec![rotation()]),
        injections(vec![injection(EegType::S, Bm::z(far), -p)]),
        injections(vec![rotation()]),
    ]);
    let qubits: Vec<usize> = (0..=far).collect();
    let initial = StabilizerGroup::from_circuit(&[Gate::pz(&qubits)], far + 1);
    let expected = (1.0 - p) * (2.0 * h).sin().powi(2);
    assert_walks(&gates, &Bm::z(0), &noise, &initial, far + 1, expected);
}
