// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Batched gates act on every operand group, and the groups' signs multiply
//! (issue #1007). A valid batched gate has disjoint groups (`Gate::validate`).
//! Each walk is checked against the dense-matrix reference or an analytic
//! value.

use pecos_core::Gate;
use pecos_core::gate_type::GateType;
use pecos_eeg::circuit::analyze_with_noise;
use pecos_eeg::eeg::EegType;
use pecos_eeg::expand::{GateIndex, make_gate};
use pecos_eeg::heisenberg::{
    build_noise_map, heisenberg_detection_probability, heisenberg_exact_from_circuit,
    heisenberg_sparse, heisenberg_with_noise_map,
};
use pecos_eeg::noise::UniformNoise;
use pecos_eeg::stabilizer::StabilizerGroup;
use pecos_eeg::{Bm, NoiseInjection, NoiseSpec};

/// Every Pauli-tracking walk on an unexpanded circuit, by name.
fn walks(
    gates: &[Gate],
    detector: &Bm,
    noise: &dyn NoiseSpec,
    initial: &StabilizerGroup,
    num_qubits: usize,
) -> [(&'static str, f64); 4] {
    // Unexpanded circuits: no gate was inserted by measurement expansion.
    let unexpanded = vec![false; gates.len()];
    let index = GateIndex::build(gates, num_qubits, noise, &unexpanded);
    let noise_map = build_noise_map(gates, noise, &index.expansion_gates);
    [
        (
            "windowed",
            heisenberg_detection_probability(gates, detector, noise, initial, 0.0, &unexpanded),
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
    ]
}

fn assert_walks(
    gates: &[Gate],
    detector: &Bm,
    initial: &StabilizerGroup,
    num_qubits: usize,
    expected: f64,
) {
    let noise = UniformNoise::coherent_only(0.0);
    let results = walks(gates, detector, &noise, initial, num_qubits);
    assert!(
        results
            .iter()
            .all(|(_, actual)| (actual - expected).abs() < 1e-12),
        "expected {expected}; walks: {results:?}",
    );
}

fn zeros(num_qubits: usize) -> StabilizerGroup {
    StabilizerGroup::from_circuit(
        &[Gate::pz(&(0..num_qubits).collect::<Vec<_>>())],
        num_qubits,
    )
}

#[test]
fn batched_hadamard_acts_on_every_qubit() {
    // Qubit 1 sees the batched H and then a second H, so it is back in |0>.
    // Qubit 0 sees only the batched H, so it is a fair coin.
    let circuit = [
        Gate::pz(&[0, 1]),
        Gate::h(&[0, 1]),
        Gate::h(&[1]),
        Gate::mz(&[0]),
        Gate::mz(&[1]),
    ];
    let noise = UniformNoise::coherent_only(0.0);
    for (record, expected) in [(0, 0.5), (1, 0.0)] {
        let exact = heisenberg_exact_from_circuit(&circuit, &[record], &noise, 2).unwrap();
        assert!((exact - expected).abs() < 1e-12, "matrix: {exact}");
    }

    let gates = [Gate::h(&[0, 1]), Gate::h(&[1])];
    assert_walks(&gates, &Bm::z(0), &zeros(2), 2, 0.5);
    assert_walks(&gates, &Bm::z(1), &zeros(2), 2, 0.0);
}

#[test]
fn batched_cx_acts_on_every_pair() {
    // H [0, 2] then CX [(0,1), (2,3)] makes two Bell pairs: each qubit is a
    // fair coin and each pair's parity is 0. The second pair is only right
    // if the walk conjugates it too.
    let gates = [Gate::h(&[0, 2]), Gate::cx(&[(0, 1), (2, 3)])];
    assert_walks(&gates, &Bm::z(3), &zeros(4), 4, 0.5);
    assert_walks(&gates, &Bm::z(0).multiply(&Bm::z(1)), &zeros(4), 4, 0.0);
    assert_walks(&gates, &Bm::z(2).multiply(&Bm::z(3)), &zeros(4), 4, 0.0);
}

#[test]
fn batched_gate_signs_multiply() {
    // X on both qubits flips Z0 and Z1, so Z0 Z1 keeps its sign and the
    // parity of |00> is unchanged. Keeping only one group's sign gives 1.
    let gates = [Gate::x(&[0, 1])];
    assert_walks(&gates, &Bm::z(0).multiply(&Bm::z(1)), &zeros(2), 2, 0.0);
    assert_walks(&gates, &Bm::z(1), &zeros(2), 2, 1.0);
}

#[test]
fn batched_preparation_builds_every_pair_into_the_stabilizer_group() {
    // H [0, 2] then CX [(0,1), (2,3)] prepares two Bell pairs, so Z2Z3 is a
    // stabilizer and its detector never fires. Building the group from the
    // first pair only leaves Z2Z3 outside it.
    let preparation = [
        Gate::pz(&[0, 1, 2, 3]),
        Gate::h(&[0, 2]),
        Gate::cx(&[(0, 1), (2, 3)]),
    ];
    let initial = StabilizerGroup::from_circuit(&preparation, 4);
    let gates = [make_gate(GateType::I, &[0])];
    assert_walks(&gates, &Bm::z(2).multiply(&Bm::z(3)), &initial, 4, 0.0);
    assert_walks(&gates, &Bm::z(0).multiply(&Bm::z(1)), &initial, 4, 0.0);
    assert_walks(&gates, &Bm::z(3), &initial, 4, 0.5);
}

/// One coherent injection, with the given label and rate 0.1, after gate 0.
struct InjectionAfterFirstGate(Bm);

impl NoiseSpec for InjectionAfterFirstGate {
    fn noise_after_gate(&self, i: usize, _: GateType, _: &[usize]) -> Vec<NoiseInjection> {
        if i == 0 {
            vec![NoiseInjection {
                eeg_type: EegType::H,
                label: self.0.clone(),
                label2: None,
                rate: 0.1,
            }]
        } else {
            Vec::new()
        }
    }
}

#[test]
fn forward_eeg_propagates_through_every_operand() {
    // X2 is untouched by the first pair of CX [(0,1), (2,3)] and spreads to
    // X2 X3 through the second.
    let gates = [make_gate(GateType::I, &[2]), Gate::cx(&[(0, 1), (2, 3)])];
    let result = analyze_with_noise(
        &gates,
        &InjectionAfterFirstGate(Bm::x(2)),
        &vec![false; gates.len()],
    );
    assert_eq!(result.generators.len(), 1);
    assert_eq!(result.generators[0].label, Bm::x(2).multiply(&Bm::x(3)));

    // A batched H on both qubits turns X0 X1 into Z0 Z1.
    let gates = [
        make_gate(GateType::I, &[0]),
        Gate::cx(&[(0, 1)]),
        Gate::h(&[0, 1]),
    ];
    let result = analyze_with_noise(
        &gates,
        &InjectionAfterFirstGate(Bm::x(0)),
        &vec![false; gates.len()],
    );
    assert_eq!(result.generators[0].label, Bm::z(0).multiply(&Bm::z(1)));
}

#[test]
fn forward_eeg_multiplies_the_signs_of_batched_groups() {
    // X on both qubits negates Z0 and Z1, so H(Z0 Z1) keeps its coefficient.
    let zz = Bm::z(0).multiply(&Bm::z(1));
    let gates = [make_gate(GateType::I, &[0, 1]), Gate::x(&[0, 1])];
    let result = analyze_with_noise(
        &gates,
        &InjectionAfterFirstGate(zz.clone()),
        &vec![false; gates.len()],
    );
    assert_eq!(result.generators[0].label, zz);
    assert!(
        (result.generators[0].coeff - 0.1).abs() < 1e-15,
        "{}",
        result.generators[0].coeff
    );
}

#[test]
fn overlapping_batched_gate_is_rejected_at_expansion() {
    // CX [(0,1), (1,2)] repeats qubit 1, which `Gate::validate` forbids.
    let gates = [
        Gate::pz(&[0, 1, 2]),
        Gate::cx(&[(0, 1), (1, 2)]),
        Gate::mz(&[2]),
    ];
    assert!(matches!(
        pecos_eeg::expand::expand_circuit(&gates),
        Err(pecos_eeg::expand::EegBuildError::InvalidGate { index: 1, .. })
    ));
}

#[test]
#[should_panic(expected = "acts on 3 qubits, not a multiple of its arity 2")]
fn walk_rejects_a_two_qubit_gate_on_an_odd_qubit_count() {
    let gates = [make_gate(GateType::CX, &[0, 1, 2])];
    heisenberg_detection_probability(
        &gates,
        &Bm::z(2),
        &UniformNoise::coherent_only(0.0),
        &zeros(3),
        0.0,
        &[false],
    );
}

#[test]
#[should_panic(expected = "acts on 3 qubits, not a multiple of its arity 2")]
fn forward_eeg_rejects_a_two_qubit_gate_on_an_odd_qubit_count() {
    let gates = [
        make_gate(GateType::I, &[0]),
        make_gate(GateType::CX, &[0, 1, 2]),
    ];
    analyze_with_noise(
        &gates,
        &InjectionAfterFirstGate(Bm::x(0)),
        &vec![false; gates.len()],
    );
}

#[test]
#[should_panic(expected = "acts on 3 qubits, not a multiple of its arity 2")]
fn stabilizer_group_rejects_a_two_qubit_gate_on_an_odd_qubit_count() {
    let _ = StabilizerGroup::from_circuit(&[make_gate(GateType::CX, &[0, 1, 2])], 3);
}
