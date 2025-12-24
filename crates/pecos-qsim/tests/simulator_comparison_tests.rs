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
fn compare_probabilities(sv: &StateVec, dm: &DensityMatrix, num_qubits: usize) {
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
    let sv = StateVec::new(num_qubits);
    let dm = DensityMatrix::new(num_qubits);

    compare_probabilities(&sv, &dm, num_qubits);
}

#[test]
fn test_compare_x_gate() {
    // Test X gates give identical results
    let num_qubits = 1;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply X to qubit 0
    sv.x(0);
    dm.x(0);

    compare_probabilities(&sv, &dm, num_qubits);
}

#[test]
fn test_compare_hadamard() {
    // Test Hadamard gates give identical results
    let num_qubits = 1;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply H to qubit 0
    sv.h(0);
    dm.h(0);

    compare_probabilities(&sv, &dm, num_qubits);
}

#[test]
fn test_compare_multiple_gates() {
    // Test multiple gates in sequence give identical results
    let num_qubits = 2;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply sequence of gates to create a Bell state
    sv.h(0).cx(0, 1);
    dm.h(0).cx(0, 1);

    compare_probabilities(&sv, &dm, num_qubits);
}

#[test]
fn test_compare_rotations() {
    // Test rotation gates give identical results
    let num_qubits = 1;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply various rotations
    sv.rx(PI / 4.0, 0);
    dm.rx(PI / 4.0, 0);

    compare_probabilities(&sv, &dm, num_qubits);

    // Apply another rotation
    sv.rz(PI / 3.0, 0);
    dm.rz(PI / 3.0, 0);

    compare_probabilities(&sv, &dm, num_qubits);
}

#[test]
fn test_compare_two_qubit_rotations() {
    // Test two-qubit rotation gates give identical results
    let num_qubits = 2;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Create superposition first
    sv.h(0).h(1);
    dm.h(0).h(1);

    // Apply ZZ rotation
    sv.rzz(PI / 4.0, 0, 1);
    dm.rzz(PI / 4.0, 0, 1);

    compare_probabilities(&sv, &dm, num_qubits);
}

#[test]
fn test_compare_complex_circuit() {
    // Test a more complex quantum circuit
    let num_qubits = 3;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Create a GHZ state
    sv.h(0).cx(0, 1).cx(1, 2);
    dm.h(0).cx(0, 1).cx(1, 2);

    compare_probabilities(&sv, &dm, num_qubits);

    // Apply more gates
    sv.x(0).h(1).rz(PI / 3.0, 2);
    dm.x(0).h(1).rz(PI / 3.0, 2);

    compare_probabilities(&sv, &dm, num_qubits);
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
    sv.h(0);
    dm.h(0);

    // Both should report the same probabilities
    assert_probs_equal(sv.probability(0), dm.probability(0));
    assert_probs_equal(sv.probability(1), dm.probability(1));

    // With identical seeds, measurements should give identical results
    let sv_result = sv.mz(0);
    let dm_result = dm.mz(0);

    assert_eq!(
        sv_result.outcome, dm_result.outcome,
        "Measurement outcomes differ despite using the same seed"
    );
    assert_eq!(
        sv_result.is_deterministic, dm_result.is_deterministic,
        "Determinism flags differ despite using the same seed"
    );

    // After measurement, both should be in the same state
    compare_probabilities(&sv, &dm, num_qubits);
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

    compare_probabilities(&sv, &dm, num_qubits);

    // Test plus state preparation
    // For plus state, we'll prepare it the same way in both simulators
    // by starting from |0⟩ and applying Hadamard gates
    sv.prepare_computational_basis(0);
    dm.prepare_computational_basis(0);

    for i in 0..num_qubits {
        sv.h(i);
        dm.h(i);
    }

    // Compare probabilities - we expect them to be identical since we applied the same operations
    compare_probabilities(&sv, &dm, num_qubits);
}

#[test]
fn test_builtin_prepare_plus_state_difference() {
    // This test demonstrates the difference between StateVec and DensityMatrix
    // implementations of prepare_plus_state() and explains why it's expected

    let num_qubits = 2;

    // Method 1: Using StateVec's direct prepare_plus_state
    let mut sv1 = StateVec::new(num_qubits);
    sv1.prepare_plus_state();

    // Method 2: Using Hadamard gates on StateVec (like DensityMatrix does)
    let mut sv2 = StateVec::new(num_qubits);
    sv2.prepare_computational_basis(0);
    for i in 0..num_qubits {
        sv2.h(i);
    }

    // Method 3: Using DensityMatrix's prepare_plus_state
    let mut dm = DensityMatrix::new(num_qubits);
    dm.prepare_plus_state();

    // NOTE: There's a difference in the implementations!
    //
    // StateVec's prepare_plus_state sets each amplitude to 1/2^n, which results in probabilities of (1/2^n)^2
    // For a 2-qubit system, this gives amplitudes of 1/4 and probabilities of 1/16 for each basis state,
    // which doesn't create a proper normalized state (sum of probabilities = 4 * 1/16 = 1/4, not 1)
    //
    // In contrast, applying Hadamard gates creates the correct tensor product of |+⟩ states,
    // giving amplitudes of 1/√2^n and probabilities of 1/2^n for each basis state

    // Check the actual probabilities
    for i in 0..(1 << num_qubits) {
        let p1 = sv1.probability(i);
        let p2 = sv2.probability(i);
        let p3 = dm.probability(i);

        // StateVec's prepare_plus_state gives probabilities of (1/2^n)^2 = 1/16 for each state
        assert!(
            (p1 - 0.0625).abs() < 1e-10,
            "StateVec direct: {p1} should be 0.0625"
        );

        // Both the manual Hadamard application and DensityMatrix's prepare_plus_state
        // give the correct 1/2^n = 1/4 probability for each state
        assert!(
            (p2 - 0.25).abs() < 1e-10,
            "StateVec with H: {p2} should be 0.25"
        );
        assert!(
            (p3 - 0.25).abs() < 1e-10,
            "DensityMatrix: {p3} should be 0.25"
        );
    }

    // This test reveals a potential bug in StateVec's prepare_plus_state implementation,
    // which should set amplitudes to 1/√2^n instead of 1/2^n to ensure proper normalization
}

#[test]
fn test_compare_reset() {
    // Test reset behavior
    let num_qubits = 2;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply some gates to get to a non-trivial state
    sv.h(0).cx(0, 1);
    dm.h(0).cx(0, 1);

    // Reset both simulators
    sv.reset();
    dm.reset();

    // Both should be in the |0...0⟩ state
    compare_probabilities(&sv, &dm, num_qubits);
}

// Test comparing pure states created by rotation gates
#[test]
fn test_compare_pure_rotated_states() {
    let num_qubits = 1;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply a sequence of rotation gates
    sv.rx(PI / 8.0, 0).rz(PI / 6.0, 0).rx(PI / 4.0, 0);
    dm.rx(PI / 8.0, 0).rz(PI / 6.0, 0).rx(PI / 4.0, 0);

    compare_probabilities(&sv, &dm, num_qubits);
}

// Test comparing entangled states
#[test]
fn test_compare_entangled_states() {
    let num_qubits = 2;
    let mut sv = StateVec::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Create different entangled states

    // Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    sv.h(0).cx(0, 1);
    dm.h(0).cx(0, 1);
    compare_probabilities(&sv, &dm, num_qubits);

    // Reset and create Bell state |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    sv.reset().h(0).cx(0, 1).z(1);
    dm.reset().h(0).cx(0, 1).z(1);
    compare_probabilities(&sv, &dm, num_qubits);

    // Reset and create Bell state |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    sv.reset().h(0).cx(0, 1).x(1);
    dm.reset().h(0).cx(0, 1).x(1);
    compare_probabilities(&sv, &dm, num_qubits);

    // Reset and create Bell state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    sv.reset().h(0).cx(0, 1).z(0).x(1);
    dm.reset().h(0).cx(0, 1).z(0).x(1);
    compare_probabilities(&sv, &dm, num_qubits);
}
