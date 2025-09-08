//! Basic tests for the `QuEST` wrapper using PECOS-style API

use approx::assert_relative_eq;
use pecos_quest::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, QuestStateVec};
use rand_chacha::ChaCha8Rng;

#[test]
fn test_state_creation() {
    let state = QuestStateVec::new(5);
    assert_eq!(state.num_qubits(), 5);

    // Check that initial state is |00000>
    let prob = state.probability(0);
    assert_relative_eq!(prob, 1.0, epsilon = 1e-10);
}

#[test]
fn test_state_with_seed() {
    let state1: QuestStateVec<ChaCha8Rng> = QuestStateVec::with_seed(3, 42);
    let state2: QuestStateVec<ChaCha8Rng> = QuestStateVec::with_seed(3, 42);

    assert_eq!(state1.num_qubits(), 3);
    assert_eq!(state2.num_qubits(), 3);

    // Both should be in the same initial state
    assert_relative_eq!(
        state1.probability(0),
        state2.probability(0),
        epsilon = 1e-10
    );
}

#[test]
fn test_computational_basis_preparation() {
    let mut state = QuestStateVec::new(2);

    // Prepare |01> (binary 10 = decimal 2)
    state.prepare_computational_basis(0b10);

    assert_relative_eq!(state.probability(0b00), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(0b01), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(0b10), 1.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(0b11), 0.0, epsilon = 1e-10);
}

#[test]
fn test_plus_state_preparation() {
    let mut state = QuestStateVec::new(2);
    state.prepare_plus_state();

    // Each basis state should have probability 1/4
    let expected_prob = 0.25;
    for i in 0..4 {
        assert_relative_eq!(state.probability(i), expected_prob, epsilon = 1e-10);
    }
}

#[test]
fn test_state_access() {
    let state = QuestStateVec::new(2);

    // Initially |00>
    // Check amplitude of |00>
    let amp0 = state.get_amplitude(0);
    assert_relative_eq!(amp0.re, 1.0, epsilon = 1e-10);
    assert_relative_eq!(amp0.im, 0.0, epsilon = 1e-10);

    // Check other amplitudes are zero
    for i in 1..4 {
        let amp = state.get_amplitude(i);
        assert_relative_eq!(amp.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(amp.im, 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_reset() {
    let mut state = QuestStateVec::new(2);

    // Change the state
    state.prepare_computational_basis(3);
    assert_relative_eq!(state.probability(3), 1.0, epsilon = 1e-10);

    // Reset should bring back to |00>
    state.reset();
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(3), 0.0, epsilon = 1e-10);
}

#[test]
fn test_pauli_gates() {
    let mut state = QuestStateVec::new(1);

    // Test Pauli-X: |0> -> |1>
    state.reset();
    state.x(0);
    assert_relative_eq!(state.probability(0), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(1), 1.0, epsilon = 1e-10);

    // Test Pauli-Z on |1>: should add phase but not change probabilities
    state.z(0);
    assert_relative_eq!(state.probability(0), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(1), 1.0, epsilon = 1e-10);

    // Test Pauli-Y: X*Z = iY, so after X then Z, we should have i|1>
    // Probability should still be 1 for |1>
    state.reset().x(0).y(0);
    // Y|1> = -i|0>, so we should be in |0>
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(1), 0.0, epsilon = 1e-10);
}

#[test]
fn test_hadamard_gate() {
    let mut state = QuestStateVec::new(1);

    // H|0> = |+> = (|0> + |1>)/sqrt(2)
    state.h(0);

    let expected_prob = 0.5;
    assert_relative_eq!(state.probability(0), expected_prob, epsilon = 1e-10);
    assert_relative_eq!(state.probability(1), expected_prob, epsilon = 1e-10);
}

#[test]
fn test_s_gates() {
    let mut state = QuestStateVec::new(1);

    // S|0> = |0>, probability unchanged
    state.sz(0);
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);

    // S†S = I, so applying S then S† should be identity
    state.szdg(0);
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);
}

#[test]
fn test_cnot_gate() {
    let mut state = QuestStateVec::new(2);

    // CNOT|00> = |00>
    state.cx(0, 1);
    assert_relative_eq!(state.probability(0b00), 1.0, epsilon = 1e-10);

    // Prepare |10> and apply CNOT(0,1) -> |11>
    state.prepare_computational_basis(0b10); // This is |10> with our qubit ordering
    state.cx(0, 1);
    assert_relative_eq!(state.probability(0b11), 1.0, epsilon = 1e-10);

    // Prepare |01> and apply CNOT(0,1) -> |01>
    state.prepare_computational_basis(0b01); // This is |01> with our qubit ordering
    state.cx(0, 1);
    assert_relative_eq!(state.probability(0b01), 1.0, epsilon = 1e-10);
}

#[test]
fn test_cz_gate() {
    let mut state = QuestStateVec::new(2);

    // CZ|00> = |00>
    state.cz(0, 1);
    assert_relative_eq!(state.probability(0b00), 1.0, epsilon = 1e-10);

    // CZ|11> = -|11> (same probability)
    state.prepare_computational_basis(0b11);
    state.cz(0, 1);
    assert_relative_eq!(state.probability(0b11), 1.0, epsilon = 1e-10);
}

#[test]
fn test_bell_state_creation() {
    let mut state = QuestStateVec::new(2);

    // Create Bell state: H(0) then CNOT(0,1)
    state.h(0).cx(0, 1);

    // Should have equal probability for |00> and |11>
    assert_relative_eq!(state.probability(0b00), 0.5, epsilon = 1e-10);
    assert_relative_eq!(state.probability(0b01), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(0b10), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(0b11), 0.5, epsilon = 1e-10);
}

#[test]
fn test_measurement() {
    let mut state = QuestStateVec::new(1);

    // Measure |0>
    let result = state.mz(0);
    assert!(!result.outcome); // |0> corresponds to false
    assert!(result.is_deterministic);

    // Measure |1>
    state.prepare_computational_basis(1);
    let result = state.mz(0);
    assert!(result.outcome); // |1> corresponds to true
    assert!(result.is_deterministic);

    // Measure superposition state
    state.reset().h(0);
    let result = state.mz(0);
    // Should not be deterministic (though this is probabilistic)
    // For a superposition state, measurement is non-deterministic
    assert!(!result.is_deterministic);
}

#[test]
fn test_rotation_gates() {
    use std::f64::consts::PI;

    let mut state = QuestStateVec::new(1);

    // RX(π) = -iX, so RX(π)|0> should give |1>
    state.rx(PI, 0);
    assert_relative_eq!(state.probability(0), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(1), 1.0, epsilon = 1e-10);

    // RZ doesn't change computational basis probabilities
    state.reset();
    state.rz(PI / 2.0, 0);
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);

    // RY(π/2) should create superposition
    state.reset();
    state.ry(PI / 2.0, 0);
    assert_relative_eq!(state.probability(0), 0.5, epsilon = 1e-10);
    assert_relative_eq!(state.probability(1), 0.5, epsilon = 1e-10);
}

#[test]
fn test_t_gates() {
    let mut state = QuestStateVec::new(1);

    // T|0> = |0>, probability unchanged
    state.t(0);
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);

    // T†T = I, so applying T then T† should be identity
    state.tdg(0);
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);
}

#[test]
fn test_rzz_gate() {
    use std::f64::consts::PI;

    let mut state = QuestStateVec::new(2);

    // RZZ doesn't change computational basis probabilities
    state.rzz(PI / 2.0, 0, 1);
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);

    // Test on |11> state
    state.prepare_computational_basis(0b11);
    state.rzz(PI / 2.0, 0, 1);
    assert_relative_eq!(state.probability(0b11), 1.0, epsilon = 1e-10);
}

#[test]
fn test_method_chaining() {
    let mut state = QuestStateVec::new(2);

    // Test that all methods return &mut Self for chaining
    state
        .reset()
        .h(0)
        .cx(0, 1)
        .z(1)
        .rx(std::f64::consts::PI / 4.0, 0);

    // Just verify it compiles and runs
    assert_eq!(state.num_qubits(), 2);
}
