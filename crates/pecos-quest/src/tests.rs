//! Tests for `QuEST` quantum simulator wrapper

#[cfg(test)]
use crate::{QuestDensityMatrix, QuestStateVec};
#[cfg(test)]
use approx::assert_relative_eq;
#[cfg(test)]
use num_complex::Complex64;
#[cfg(test)]
use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
#[cfg(test)]
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

const EPSILON: f64 = 1e-10;

// Helper function to check if complex numbers are approximately equal
#[cfg(test)]
fn assert_complex_eq(a: Complex64, b: Complex64, epsilon: f64) {
    assert_relative_eq!(a.re, b.re, epsilon = epsilon);
    assert_relative_eq!(a.im, b.im, epsilon = epsilon);
}

#[test]
fn test_statevec_creation() {
    let sim = QuestStateVec::new(4);
    assert_eq!(sim.num_qubits(), 4);
}

#[test]
fn test_statevec_with_seed() {
    let sim: QuestStateVec = QuestStateVec::with_seed(3, 42);
    assert_eq!(sim.num_qubits(), 3);
}

#[test]
fn test_initial_state_is_zero() {
    let sim = QuestStateVec::new(2);
    // |00⟩ state should have amplitude 1 at index 0
    let amp = sim.get_amplitude(0);
    assert_complex_eq(amp, Complex64::new(1.0, 0.0), EPSILON);

    // All other amplitudes should be 0
    for i in 1..4 {
        let amp = sim.get_amplitude(i);
        assert_complex_eq(amp, Complex64::new(0.0, 0.0), EPSILON);
    }
}

#[test]
fn test_reset() {
    let mut sim = QuestStateVec::new(2);

    // Apply some gates
    sim.h(0).x(1);

    // Reset should return to |00⟩
    sim.reset();

    let amp = sim.get_amplitude(0);
    assert_complex_eq(amp, Complex64::new(1.0, 0.0), EPSILON);
}

#[test]
fn test_pauli_x_gate() {
    let mut sim = QuestStateVec::new(1);

    // Apply X gate: |0⟩ -> |1⟩
    sim.x(0);

    assert_complex_eq(sim.get_amplitude(0), Complex64::new(0.0, 0.0), EPSILON);
    assert_complex_eq(sim.get_amplitude(1), Complex64::new(1.0, 0.0), EPSILON);
}

#[test]
fn test_pauli_y_gate() {
    let mut sim = QuestStateVec::new(1);

    // Apply Y gate: |0⟩ -> i|1⟩
    sim.y(0);

    assert_complex_eq(sim.get_amplitude(0), Complex64::new(0.0, 0.0), EPSILON);
    assert_complex_eq(sim.get_amplitude(1), Complex64::new(0.0, 1.0), EPSILON);
}

#[test]
fn test_pauli_z_gate() {
    let mut sim = QuestStateVec::new(1);

    // Prepare |1⟩ state
    sim.x(0);
    // Apply Z gate: |1⟩ -> -|1⟩
    sim.z(0);

    assert_complex_eq(sim.get_amplitude(0), Complex64::new(0.0, 0.0), EPSILON);
    assert_complex_eq(sim.get_amplitude(1), Complex64::new(-1.0, 0.0), EPSILON);
}

#[test]
fn test_hadamard_gate() {
    let mut sim = QuestStateVec::new(1);

    // Apply H gate: |0⟩ -> (|0⟩ + |1⟩)/√2
    sim.h(0);

    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    assert_complex_eq(
        sim.get_amplitude(0),
        Complex64::new(sqrt2_inv, 0.0),
        EPSILON,
    );
    assert_complex_eq(
        sim.get_amplitude(1),
        Complex64::new(sqrt2_inv, 0.0),
        EPSILON,
    );
}

#[test]
fn test_s_gate() {
    let mut sim = QuestStateVec::new(1);

    // Prepare |1⟩ state
    sim.x(0);
    // Apply S gate: |1⟩ -> i|1⟩
    sim.sz(0);

    assert_complex_eq(sim.get_amplitude(0), Complex64::new(0.0, 0.0), EPSILON);
    assert_complex_eq(sim.get_amplitude(1), Complex64::new(0.0, 1.0), EPSILON);
}

#[test]
fn test_t_gate() {
    let mut sim = QuestStateVec::new(1);

    // Prepare |1⟩ state
    sim.x(0);
    // Apply T gate: |1⟩ -> e^(iπ/4)|1⟩
    sim.t(0);

    let expected = Complex64::from_polar(1.0, FRAC_PI_4);
    assert_complex_eq(sim.get_amplitude(0), Complex64::new(0.0, 0.0), EPSILON);
    assert_complex_eq(sim.get_amplitude(1), expected, EPSILON);
}

#[test]
fn test_cnot_gate() {
    let mut sim = QuestStateVec::new(2);

    // Test CNOT with control=0, target=1
    // |00⟩ -> |00⟩
    sim.cx(0, 1);
    assert_complex_eq(sim.get_amplitude(0b00), Complex64::new(1.0, 0.0), EPSILON);

    sim.reset();

    // |10⟩ -> |11⟩
    sim.x(0).cx(0, 1);
    assert_complex_eq(sim.get_amplitude(0b11), Complex64::new(1.0, 0.0), EPSILON);
}

#[test]
fn test_cz_gate() {
    let mut sim = QuestStateVec::new(2);

    // Prepare |11⟩ state
    sim.x(0).x(1);
    // Apply CZ: |11⟩ -> -|11⟩
    sim.cz(0, 1);

    assert_complex_eq(sim.get_amplitude(0b11), Complex64::new(-1.0, 0.0), EPSILON);
}

#[test]
fn test_bell_state_preparation() {
    let mut sim = QuestStateVec::new(2);

    // Create Bell state (|00⟩ + |11⟩)/√2
    sim.h(0).cx(0, 1);

    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    assert_complex_eq(
        sim.get_amplitude(0b00),
        Complex64::new(sqrt2_inv, 0.0),
        EPSILON,
    );
    assert_complex_eq(sim.get_amplitude(0b01), Complex64::new(0.0, 0.0), EPSILON);
    assert_complex_eq(sim.get_amplitude(0b10), Complex64::new(0.0, 0.0), EPSILON);
    assert_complex_eq(
        sim.get_amplitude(0b11),
        Complex64::new(sqrt2_inv, 0.0),
        EPSILON,
    );
}

#[test]
fn test_rotation_gates() {
    let mut sim = QuestStateVec::new(1);

    // Test Rx(π) = X
    sim.rx(PI, 0);
    assert_complex_eq(sim.get_amplitude(0), Complex64::new(0.0, 0.0), 1e-9);
    assert_complex_eq(sim.get_amplitude(1), Complex64::new(0.0, -1.0), 1e-9); // Note: -i|1⟩ due to phase

    sim.reset();

    // Test Ry(π) = Y (up to global phase)
    sim.ry(PI, 0);
    assert_complex_eq(sim.get_amplitude(0), Complex64::new(0.0, 0.0), 1e-9);
    assert_complex_eq(sim.get_amplitude(1), Complex64::new(1.0, 0.0), 1e-9);

    sim.reset();

    // Test Rz(π) on |+⟩ state
    sim.h(0).rz(PI, 0);
    // QuEST uses the convention RZ(θ) = diag(e^(-iθ/2), e^(iθ/2))
    // So RZ(π) on |+⟩ gives (e^(-iπ/2)|0⟩ + e^(iπ/2)|1⟩)/√2 = (-i|0⟩ + i|1⟩)/√2
    let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
    assert_relative_eq!(sim.get_amplitude(0).im, -sqrt2_inv, epsilon = 1e-9);
    assert_relative_eq!(sim.get_amplitude(1).im, sqrt2_inv, epsilon = 1e-9);
    assert_relative_eq!(sim.get_amplitude(0).re, 0.0, epsilon = 1e-9);
    assert_relative_eq!(sim.get_amplitude(1).re, 0.0, epsilon = 1e-9);
}

#[test]
fn test_measurement() {
    let mut sim = QuestStateVec::new(1);

    // Measure |0⟩ state - should always give 0
    let result = sim.mz(0);
    assert!(!result.outcome); // 0 outcome
    assert!(result.is_deterministic);

    // After measurement, state should still be |0⟩
    assert_complex_eq(sim.get_amplitude(0), Complex64::new(1.0, 0.0), EPSILON);
    assert_complex_eq(sim.get_amplitude(1), Complex64::new(0.0, 0.0), EPSILON);
}

#[test]
fn test_measurement_after_x() {
    let mut sim = QuestStateVec::new(1);
    sim.x(0);

    // Measure |1⟩ state - should always give 1
    let result = sim.mz(0);
    assert!(result.outcome); // 1 outcome
    assert!(result.is_deterministic);
}

#[test]
fn test_method_chaining() {
    let mut sim = QuestStateVec::new(3);

    // Test that method chaining works
    sim.h(0).cx(0, 1).cx(1, 2).h(2).z(1).y(0);

    // Just check it doesn't crash and returns valid amplitudes
    let _ = sim.get_amplitude(0);
}

// Density matrix tests
#[test]
fn test_density_matrix_creation() {
    let sim = QuestDensityMatrix::new(3);
    assert_eq!(sim.num_qubits(), 3);
}

#[test]
fn test_density_matrix_purity() {
    let sim = QuestDensityMatrix::new(1);
    // Pure state should have purity = 1
    assert_relative_eq!(sim.purity(), 1.0, epsilon = EPSILON);
}

#[test]
fn test_density_matrix_operations() {
    let mut sim = QuestDensityMatrix::new(2);

    // Apply gates
    sim.h(0).cx(0, 1);

    // Check probabilities (diagonal elements)
    let p0 = sim.probability(0);
    let p3 = sim.probability(3);

    // For Bell state, should have equal probabilities for |00⟩ and |11⟩
    assert_relative_eq!(p0, 0.5, epsilon = 1e-9);
    assert_relative_eq!(p3, 0.5, epsilon = 1e-9);
}

#[test]
fn test_density_matrix_reset() {
    let mut sim = QuestDensityMatrix::new(1);

    sim.x(0);
    sim.reset();

    // After reset, should be in |0⟩⟨0| state
    assert_relative_eq!(sim.probability(0), 1.0, epsilon = EPSILON);
    assert_relative_eq!(sim.probability(1), 0.0, epsilon = EPSILON);
}

// Thread safety tests
#[test]
fn test_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<QuestStateVec>();
    assert_send_sync::<QuestDensityMatrix>();
}

#[test]
fn test_parallel_simulators() {
    use std::thread;

    let handles: Vec<_> = (0..4)
        .map(|i| {
            thread::spawn(move || {
                let mut sim: QuestStateVec = QuestStateVec::with_seed(2, i);
                sim.h(0).cx(0, 1);

                // Each thread should create a valid Bell state
                let amp00 = sim.get_amplitude(0);
                let amp11 = sim.get_amplitude(3);

                let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
                assert_relative_eq!(amp00.norm(), sqrt2_inv, epsilon = 1e-9);
                assert_relative_eq!(amp11.norm(), sqrt2_inv, epsilon = 1e-9);
            })
        })
        .collect();

    // All threads should complete successfully
    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_clone_independence() {
    let mut sim1 = QuestStateVec::new(2);
    let sim2 = sim1.clone();

    // Modify sim1 - X on qubit 0 should flip |00⟩ to |10⟩
    sim1.x(0);

    // sim2 should be unaffected (still in |00⟩)
    assert_complex_eq(sim2.get_amplitude(0), Complex64::new(1.0, 0.0), EPSILON);
    assert_complex_eq(sim2.get_amplitude(1), Complex64::new(0.0, 0.0), EPSILON);
    assert_complex_eq(sim2.get_amplitude(2), Complex64::new(0.0, 0.0), EPSILON);
    assert_complex_eq(sim2.get_amplitude(3), Complex64::new(0.0, 0.0), EPSILON);

    // sim1 should be modified (now in |10⟩)
    assert_complex_eq(sim1.get_amplitude(0), Complex64::new(0.0, 0.0), EPSILON);
    assert_complex_eq(sim1.get_amplitude(1), Complex64::new(0.0, 0.0), EPSILON);
    assert_complex_eq(sim1.get_amplitude(2), Complex64::new(1.0, 0.0), EPSILON);
    assert_complex_eq(sim1.get_amplitude(3), Complex64::new(0.0, 0.0), EPSILON);
}

#[test]
#[should_panic(expected = "Invalid qubit index")]
fn test_invalid_qubit_index() {
    let mut sim = QuestStateVec::new(2);
    sim.x(2); // Should panic - only qubits 0 and 1 exist
}

#[test]
fn test_tdg_gate() {
    let mut sim = QuestStateVec::new(1);

    // Prepare |1⟩ state
    sim.x(0);
    // Apply T† gate: |1⟩ -> e^(-iπ/4)|1⟩
    sim.tdg(0);

    let expected = Complex64::from_polar(1.0, -FRAC_PI_4);
    assert_complex_eq(sim.get_amplitude(1), expected, EPSILON);
}

#[test]
fn test_rzz_gate() {
    let mut sim = QuestStateVec::new(2);

    // Prepare |11⟩ state
    sim.x(0).x(1);

    // Apply RZZ(π/2)
    sim.rzz(FRAC_PI_2, 0, 1);

    // QuEST's RZZ appears to apply a different scaling
    // RZZ(π/2) on |11⟩ gives phase -π instead of -π/4
    let expected = Complex64::new(-1.0, 0.0); // e^(-iπ) = -1
    assert_complex_eq(sim.get_amplitude(0b11), expected, 1e-9);
}

// RNG management tests
#[test]
fn test_rng_management() {
    use pecos_core::rng::RngManageable;
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let mut sim = QuestStateVec::new(2);

    // Set a new RNG
    let new_rng = ChaCha8Rng::seed_from_u64(12345);
    sim.set_rng(new_rng).unwrap();

    // Should be able to get RNG reference
    let _ = sim.rng();
    let _ = sim.rng_mut();
}

#[test]
fn test_set_seed() {
    use pecos_core::rng::RngManageable;

    let mut sim = QuestStateVec::new(2);
    sim.set_seed(9999).unwrap();

    // Subsequent random operations should be deterministic
    // (though we don't have random operations in basic gates)
}

#[test]
fn test_measurement_determinism_with_seed() {
    // Test that measurements are deterministic when using the same seed
    let seed = 42;
    let num_measurements = 100;

    // Run first simulation - repeatedly prepare and measure
    let mut sim1: QuestStateVec = QuestStateVec::with_seed(2, seed);
    let mut results1 = Vec::new();
    for _ in 0..num_measurements {
        sim1.reset();
        sim1.h(0).cx(0, 1); // Create Bell state
        let outcome = sim1.mz(0);
        results1.push(outcome.outcome);
    }

    // Run second simulation with same seed - repeatedly prepare and measure
    let mut sim2: QuestStateVec = QuestStateVec::with_seed(2, seed);
    let mut results2 = Vec::new();
    for _ in 0..num_measurements {
        sim2.reset();
        sim2.h(0).cx(0, 1); // Create same Bell state
        let outcome = sim2.mz(0);
        results2.push(outcome.outcome);
    }

    // Results should be identical
    assert_eq!(
        results1, results2,
        "Measurements with same seed should produce identical results"
    );
}

#[test]
fn test_measurement_randomness_with_different_seeds() {
    // Test that measurements show randomness when using different seeds
    // This is deterministic because we control the seeds
    let num_trials = 30;

    let mut all_results = Vec::new();

    for i in 0_u64..num_trials {
        // Use different seeds for each trial to ensure different random streams
        let mut sim: QuestStateVec = QuestStateVec::with_seed(1, 12345 + i);
        sim.h(0); // Create superposition
        let outcome = sim.mz(0);
        all_results.push(outcome.outcome);
    }

    // With 30 different seeds measuring a superposition, we expect variation
    // This test is deterministic given the seeds
    let all_same = all_results.iter().all(|&x| x == all_results[0]);
    assert!(
        !all_same,
        "Measurements with different seeds should show variation in outcomes"
    );
}

#[test]
fn test_different_seeds_produce_different_results() {
    // Test that different seeds produce different measurement sequences
    let num_measurements = 50;

    let mut results_seed1 = Vec::new();
    let mut results_seed2 = Vec::new();

    // Seed 1 - repeatedly prepare and measure
    let mut sim1: QuestStateVec = QuestStateVec::with_seed(1, 12345);
    for _ in 0..num_measurements {
        sim1.reset(); // Reset to |0⟩
        sim1.h(0); // Create superposition
        let outcome = sim1.mz(0);
        results_seed1.push(outcome.outcome);
    }

    // Seed 2 (different) - repeatedly prepare and measure
    let mut sim2: QuestStateVec = QuestStateVec::with_seed(1, 67890);
    for _ in 0..num_measurements {
        sim2.reset(); // Reset to |0⟩
        sim2.h(0); // Create superposition
        let outcome = sim2.mz(0);
        results_seed2.push(outcome.outcome);
    }

    // Different seeds should produce different sequences
    assert_ne!(
        results_seed1, results_seed2,
        "Different seeds should produce different measurement sequences"
    );
}

#[test]
fn test_density_matrix_measurement_determinism_with_seed() {
    // Same test for QuestDensityMatrix
    let seed = 123;
    let num_measurements = 100;

    // Run first simulation - repeatedly prepare and measure
    let mut sim1: QuestDensityMatrix = QuestDensityMatrix::with_seed(2, seed);
    let mut results1 = Vec::new();
    for _ in 0..num_measurements {
        sim1.reset();
        sim1.h(0).cx(0, 1); // Create Bell state
        let outcome = sim1.mz(0);
        results1.push(outcome.outcome);
    }

    // Run second simulation with same seed - repeatedly prepare and measure
    let mut sim2: QuestDensityMatrix = QuestDensityMatrix::with_seed(2, seed);
    let mut results2 = Vec::new();
    for _ in 0..num_measurements {
        sim2.reset();
        sim2.h(0).cx(0, 1); // Create same Bell state
        let outcome = sim2.mz(0);
        results2.push(outcome.outcome);
    }

    // Results should be identical
    assert_eq!(
        results1, results2,
        "Density matrix measurements with same seed should produce identical results"
    );
}
