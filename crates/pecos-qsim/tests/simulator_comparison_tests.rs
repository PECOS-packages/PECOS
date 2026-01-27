use pecos_core::{Angle64, qid, qid2};
use pecos_qsim::ArbitraryRotationGateable;
use pecos_qsim::CliffordGateable;
use pecos_qsim::DensityMatrix;
use pecos_qsim::QuantumSimulator;
use pecos_qsim::StateVec;
use std::f64::consts::PI;

// Helper function to check if two probabilities are close enough
fn assert_probs_equal(p1: f64, p2: f64) {
    assert!(
        (p1 - p2).abs() < 1e-10,
        "Probabilities differ: {p1} vs {p2}"
    );
}

// Helper function to compare the results of multiple basis states between simulators
fn compare_probabilities(sv: &mut StateVec, dm: &mut DensityMatrix, num_qubits: usize) {
    for i in 0..(1 << num_qubits) {
        let sv_prob = sv.probability(i);
        let dm_prob = dm.probability(i);
        assert_probs_equal(sv_prob, dm_prob);
    }
}

#[test]
fn test_compare_initial_state() {
    // Test that both simulators start in the |0...0⟩ state
    let num_qubits = 2;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    compare_probabilities(&mut sv, &mut dm, num_qubits);
}

#[test]
fn test_compare_x_gate() {
    // Test X gates give identical results
    let num_qubits = 1;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply X to qubit 0
    sv.x(&qid(0));
    dm.x(&qid(0));

    compare_probabilities(&mut sv, &mut dm, num_qubits);
}

#[test]
fn test_compare_hadamard() {
    // Test Hadamard gates give identical results
    let num_qubits = 1;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply H to qubit 0
    sv.h(&qid(0));
    dm.h(&qid(0));

    compare_probabilities(&mut sv, &mut dm, num_qubits);
}

#[test]
fn test_compare_multiple_gates() {
    // Test multiple gates in sequence give identical results
    let num_qubits = 2;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply sequence of gates to create a Bell state
    sv.h(&qid(0)).cx(&qid2(0, 1));
    dm.h(&qid(0)).cx(&qid2(0, 1));

    compare_probabilities(&mut sv, &mut dm, num_qubits);
}

#[test]
fn test_compare_rotations() {
    // Test rotation gates give identical results
    let num_qubits = 1;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply various rotations
    sv.rx(Angle64::from_radians(PI / 4.0), &qid(0));
    dm.rx(Angle64::from_radians(PI / 4.0), &qid(0));

    compare_probabilities(&mut sv, &mut dm, num_qubits);

    // Apply another rotation
    sv.rz(Angle64::from_radians(PI / 3.0), &qid(0));
    dm.rz(Angle64::from_radians(PI / 3.0), &qid(0));

    compare_probabilities(&mut sv, &mut dm, num_qubits);
}

#[test]
fn test_compare_two_qubit_rotations() {
    // Test two-qubit rotation gates give identical results
    let num_qubits = 2;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Create superposition first
    sv.h(&qid(0)).h(&qid(1));
    dm.h(&qid(0)).h(&qid(1));

    // Apply ZZ rotation
    sv.rzz(Angle64::from_radians(PI / 4.0), &qid2(0, 1));
    dm.rzz(Angle64::from_radians(PI / 4.0), &qid2(0, 1));

    compare_probabilities(&mut sv, &mut dm, num_qubits);
}

#[test]
fn test_compare_complex_circuit() {
    // Test a more complex quantum circuit
    let num_qubits = 3;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Create a GHZ state
    sv.h(&qid(0)).cx(&qid2(0, 1)).cx(&qid2(1, 2));
    dm.h(&qid(0)).cx(&qid2(0, 1)).cx(&qid2(1, 2));

    compare_probabilities(&mut sv, &mut dm, num_qubits);

    // Apply more gates
    sv.x(&qid(0))
        .h(&qid(1))
        .rz(Angle64::from_radians(PI / 3.0), &qid(2));
    dm.x(&qid(0))
        .h(&qid(1))
        .rz(Angle64::from_radians(PI / 3.0), &qid(2));

    compare_probabilities(&mut sv, &mut dm, num_qubits);
}

#[test]
fn test_compare_measurements() {
    // Test measurement behavior
    // Note: Since measurement is probabilistic and involves state collapse,
    // we can only test this properly with a fixed seed

    let num_qubits = 1;
    let seed = 42; // Fixed seed for deterministic behavior

    let mut sv = StateVec::with_seed(num_qubits, seed);
    let mut dm = DensityMatrix::with_seed(num_qubits, seed);

    // Put qubits in superposition
    sv.h(&qid(0));
    dm.h(&qid(0));

    // Both should report the same probabilities
    assert_probs_equal(sv.probability(0), dm.probability(0));
    assert_probs_equal(sv.probability(1), dm.probability(1));

    // With identical seeds, measurements should give identical results
    let sv_result = sv.mz(&qid(0)).into_iter().next().unwrap();
    let dm_result = dm.mz(&qid(0)).into_iter().next().unwrap();

    assert_eq!(
        sv_result.outcome, dm_result.outcome,
        "Measurement outcomes differ despite using the same seed"
    );
    assert_eq!(
        sv_result.is_deterministic, dm_result.is_deterministic,
        "Determinism flags differ despite using the same seed"
    );

    // After measurement, both should be in the same state
    compare_probabilities(&mut sv, &mut dm, num_qubits);
}

#[test]
fn test_compare_prepare_states() {
    // Test preparation methods give identical results
    let num_qubits = 2;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Test computational basis preparation
    sv.prepare_computational_basis(2); // Prepare |10⟩
    dm.prepare_computational_basis(2); // Prepare |10⟩

    compare_probabilities(&mut sv, &mut dm, num_qubits);

    // Test plus state preparation
    // For plus state, we'll prepare it the same way in both simulators
    // by starting from |0⟩ and applying Hadamard gates
    sv.prepare_computational_basis(0);
    dm.prepare_computational_basis(0);

    for i in 0..num_qubits {
        sv.h(&qid(i));
        dm.h(&qid(i));
    }

    // Compare probabilities - we expect them to be identical since we applied the same operations
    compare_probabilities(&mut sv, &mut dm, num_qubits);
}

#[test]
fn test_builtin_prepare_plus_state_consistency() {
    // This test verifies that all methods of preparing the |+...+⟩ state
    // produce the same correctly normalized result.

    let num_qubits = 2;

    // Method 1: Using StateVec's direct prepare_plus_state
    let mut sv1 = StateVec::new(num_qubits);
    sv1.prepare_plus_state();

    // Method 2: Using Hadamard gates on StateVec
    let mut sv2 = StateVec::new(num_qubits);
    sv2.prepare_computational_basis(0);
    for i in 0..num_qubits {
        sv2.h(&qid(i));
    }

    // Method 3: Using DensityMatrix's prepare_plus_state
    let mut dm = DensityMatrix::new(num_qubits);
    dm.prepare_plus_state();

    // All methods should produce the correct |+...+⟩ state with uniform probabilities
    // For n qubits, each basis state should have probability 1/2^n
    let expected_prob = 1.0 / f64::from(1 << num_qubits); // 0.25 for 2 qubits

    for i in 0..(1 << num_qubits) {
        let p1 = sv1.probability(i);
        let p2 = sv2.probability(i);
        let p3 = dm.probability(i);

        assert!(
            (p1 - expected_prob).abs() < 1e-10,
            "StateVec direct: {p1} should be {expected_prob}"
        );
        assert!(
            (p2 - expected_prob).abs() < 1e-10,
            "StateVec with H: {p2} should be {expected_prob}"
        );
        assert!(
            (p3 - expected_prob).abs() < 1e-10,
            "DensityMatrix: {p3} should be {expected_prob}"
        );
    }
}

#[test]
fn test_compare_reset() {
    // Test reset behavior
    let num_qubits = 2;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply some gates to get to a non-trivial state
    sv.h(&qid(0)).cx(&qid2(0, 1));
    dm.h(&qid(0)).cx(&qid2(0, 1));

    // Reset both simulators
    sv.reset();
    dm.reset();

    // Both should be in the |0...0⟩ state
    compare_probabilities(&mut sv, &mut dm, num_qubits);
}

// Test comparing pure states created by rotation gates
#[test]
fn test_compare_pure_rotated_states() {
    let num_qubits = 1;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply a sequence of rotation gates
    sv.rx(Angle64::from_radians(PI / 8.0), &qid(0))
        .rz(Angle64::from_radians(PI / 6.0), &qid(0))
        .rx(Angle64::from_radians(PI / 4.0), &qid(0));
    dm.rx(Angle64::from_radians(PI / 8.0), &qid(0))
        .rz(Angle64::from_radians(PI / 6.0), &qid(0))
        .rx(Angle64::from_radians(PI / 4.0), &qid(0));

    compare_probabilities(&mut sv, &mut dm, num_qubits);
}

// Test comparing entangled states
#[test]
fn test_compare_entangled_states() {
    let num_qubits = 2;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Create different entangled states

    // Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    sv.h(&qid(0)).cx(&qid2(0, 1));
    dm.h(&qid(0)).cx(&qid2(0, 1));
    compare_probabilities(&mut sv, &mut dm, num_qubits);

    // Reset and create Bell state |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    sv.reset().h(&qid(0)).cx(&qid2(0, 1)).z(&qid(1));
    dm.reset().h(&qid(0)).cx(&qid2(0, 1)).z(&qid(1));
    compare_probabilities(&mut sv, &mut dm, num_qubits);

    // Reset and create Bell state |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    sv.reset().h(&qid(0)).cx(&qid2(0, 1)).x(&qid(1));
    dm.reset().h(&qid(0)).cx(&qid2(0, 1)).x(&qid(1));
    compare_probabilities(&mut sv, &mut dm, num_qubits);

    // Reset and create Bell state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    sv.reset().h(&qid(0)).cx(&qid2(0, 1)).z(&qid(0)).x(&qid(1));
    dm.reset().h(&qid(0)).cx(&qid2(0, 1)).z(&qid(0)).x(&qid(1));
    compare_probabilities(&mut sv, &mut dm, num_qubits);
}
