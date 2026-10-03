// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Batched gates act on every operand, and a batched two-qubit gate applies
//! its pairs in order (issue #1007). Each walk is checked against the
//! dense-matrix reference or an analytic value.

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
    let index = GateIndex::build(gates, num_qubits, noise);
    let noise_map = build_noise_map(gates, noise, &index.expansion_gates);
    [
        (
            "windowed",
            heisenberg_detection_probability(gates, detector, noise, initial, 0.0),
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
fn batched_cx_applies_its_pairs_in_order() {
    // CX [(0,1), (1,2)] after H0 prepares a GHZ state: CX(0,1) then CX(1,2).
    // In the other order CX(1,2) acts on |0> first and qubit 2 stays 0.
    let gates = [Gate::h(&[0]), Gate::cx(&[(0, 1), (1, 2)])];
    assert_walks(&gates, &Bm::z(2), &zeros(3), 3, 0.5);
    assert_walks(&gates, &Bm::z(1).multiply(&Bm::z(2)), &zeros(3), 3, 0.0);
    assert_walks(&gates, &Bm::z(0).multiply(&Bm::z(2)), &zeros(3), 3, 0.0);
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

/// One coherent X0 injection after gate 0.
struct X0AfterFirstGate;

impl NoiseSpec for X0AfterFirstGate {
    fn noise_after_gate(&self, i: usize, _: GateType, _: &[usize]) -> Vec<NoiseInjection> {
        if i == 0 {
            vec![NoiseInjection {
                eeg_type: EegType::H,
                label: Bm::x(0),
                label2: None,
                rate: 0.1,
            }]
        } else {
            Vec::new()
        }
    }
}

#[test]
fn forward_eeg_propagates_through_every_operand_in_order() {
    // X0 through CX [(0,1), (1,2)] becomes X0 X1 X2 when the pairs act in
    // order (X0 -> X0X1 -> X0X1X2); the reverse order gives X0 X1.
    let gates = [make_gate(GateType::I, &[0]), Gate::cx(&[(0, 1), (1, 2)])];
    let result = analyze_with_noise(&gates, &X0AfterFirstGate);
    assert_eq!(result.generators.len(), 1);
    let expected = Bm::x(0).multiply(&Bm::x(1)).multiply(&Bm::x(2));
    assert_eq!(result.generators[0].label, expected);

    // A batched H turns X0 into Z0 and leaves the other operand's X alone.
    let gates = [
        make_gate(GateType::I, &[0]),
        Gate::cx(&[(0, 1)]),
        Gate::h(&[0, 1]),
    ];
    let result = analyze_with_noise(&gates, &X0AfterFirstGate);
    assert_eq!(
        result.generators[0].label,
        Bm::z(0).multiply(&Bm::z(1)),
        "X0 X1 after H on both qubits"
    );
}

#[test]
#[should_panic(expected = "acts on 3 qubits, not a multiple of its arity 2")]
fn two_qubit_gate_on_an_odd_qubit_count_is_rejected() {
    let gates = [make_gate(GateType::CX, &[0, 1, 2])];
    heisenberg_detection_probability(
        &gates,
        &Bm::z(2),
        &UniformNoise::coherent_only(0.0),
        &zeros(3),
        0.0,
    );
}
