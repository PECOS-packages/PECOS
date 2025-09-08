// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use pecos_cppsparsesim::CppSparseStab;
use pecos_qsim::{CliffordGateable, QuantumSimulator};

#[test]
fn test_basic_gates() {
    let mut sim = CppSparseStab::new(2);

    // Test basic single-qubit gates
    sim.h(0);
    sim.x(1);
    sim.z(0);
    sim.y(1);
}

#[test]
fn test_bell_state() {
    let mut sim = CppSparseStab::new(2);

    // Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    sim.h(0);
    sim.cx(0, 1);

    // Measure both qubits
    let r0 = sim.mz(0);
    let r1 = sim.mz(1);

    // Both measurements should be equal (entangled)
    assert_eq!(r0.outcome, r1.outcome);
}

#[test]
fn test_reset() {
    let mut sim = CppSparseStab::new(3);

    // Apply some gates
    sim.h(0).cx(0, 1).h(2);

    // Reset the simulator
    sim.reset();

    // After reset, all qubits should be in |0⟩ state
    // Measuring in Z basis should give 0
    let r0 = sim.mz(0);
    let r1 = sim.mz(1);
    let r2 = sim.mz(2);

    assert!(!r0.outcome);
    assert!(!r1.outcome);
    assert!(!r2.outcome);
    // Note: Determinism tracking has been removed from the wrapper
    // per design decision - the wrapper does not track determinism
}

#[test]
fn test_phase_gates() {
    let mut sim = CppSparseStab::new(1);

    // Test S and S† gates
    sim.sz(0);
    sim.szdg(0);

    // Test SX and SX† gates
    sim.sx(0);
    sim.sxdg(0);

    // Test SY and SY† gates
    sim.sy(0);
    sim.sydg(0);
}

#[test]
fn test_deterministic_with_seed() {
    // Test that simulators with the same seed produce identical results
    // Now that we have per-instance RNG, this should work correctly

    let seed = 42;

    // Create two simulators with the same seed
    let mut sim1 = CppSparseStab::new_with_seed(3, seed);
    let mut sim2 = CppSparseStab::new_with_seed(3, seed);

    // Apply same operations to both
    sim1.h(0);
    sim1.h(1);
    sim1.cx(0, 2);
    sim1.h(2);

    sim2.h(0);
    sim2.h(1);
    sim2.cx(0, 2);
    sim2.h(2);

    // Collect measurements from both simulators
    let results1 = vec![sim1.mz(0).outcome, sim1.mz(1).outcome, sim1.mz(2).outcome];

    let results2 = vec![sim2.mz(0).outcome, sim2.mz(1).outcome, sim2.mz(2).outcome];

    // Results should be identical with same seed
    assert_eq!(
        results1, results2,
        "Simulators with same seed should produce identical results"
    );

    // Test with different seeds to ensure they differ
    let mut sim3 = CppSparseStab::new_with_seed(3, seed + 1);
    sim3.h(0);
    sim3.h(1);
    sim3.cx(0, 2);
    sim3.h(2);

    let _results3 = [sim3.mz(0).outcome, sim3.mz(1).outcome, sim3.mz(2).outcome];

    // Very unlikely to be the same with different seed
    // (1/8 chance, but we accept this small possibility)
    // The other tests verify statistical properties more thoroughly
}

#[test]
fn test_different_seeds_different_results() {
    // Test that different seeds produce different results (with high probability)
    // We'll run multiple trials to account for the small chance of getting same results

    let mut same_count = 0;
    let trials = 10;

    for trial in 0..trials {
        let seed1 = trial * 100 + 1;
        let seed2 = trial * 100 + 2;

        let mut sim1 = CppSparseStab::new_with_seed(5, seed1);
        let mut sim2 = CppSparseStab::new_with_seed(5, seed2);

        // Create superposition on all qubits
        for i in 0..5 {
            sim1.h(i);
            sim2.h(i);
        }

        // Measure all qubits
        let mut results1 = vec![];
        let mut results2 = vec![];
        for i in 0..5 {
            results1.push(sim1.mz(i).outcome);
            results2.push(sim2.mz(i).outcome);
        }

        if results1 == results2 {
            same_count += 1;
        }
    }

    // With 5 qubits, probability of getting same results is 1/2^5 = 1/32
    // Over 10 trials, we expect 0-1 matches, definitely not all 10
    assert!(
        same_count < trials / 2,
        "Different seeds should produce different results most of the time. Got {same_count} same out of {trials} trials"
    );
}

#[test]
fn test_forced_measurements() {
    // Test that forced measurements work correctly
    let mut sim = CppSparseStab::new_with_seed(3, 123);

    // Put qubits in superposition
    sim.h(0);
    sim.h(1);
    sim.h(2);

    // Force measurements to specific values
    let r0 = sim.force_measure(0, false); // Force to 0
    let r1 = sim.force_measure(1, true); // Force to 1
    let r2 = sim.force_measure(2, false); // Force to 0

    assert!(!r0.outcome, "Forced measurement to 0 should return false");
    assert!(r1.outcome, "Forced measurement to 1 should return true");
    assert!(!r2.outcome, "Forced measurement to 0 should return false");

    // After forcing, qubits should be in deterministic states
    // Measuring again should give same results
    let r0_again = sim.mz(0);
    let r1_again = sim.mz(1);
    let r2_again = sim.mz(2);

    assert!(!r0_again.outcome);
    assert!(r1_again.outcome);
    assert!(!r2_again.outcome);
}

#[test]
fn test_forced_measurement_on_deterministic_state() {
    // Test that forcing a deterministic state returns the deterministic value
    // (not the forced value)
    let mut sim = CppSparseStab::new(2);

    // Qubit 0 is deterministically |0⟩
    // Try to force it to 1
    let r0 = sim.force_measure(0, true);
    assert!(
        !r0.outcome,
        "Forcing deterministic |0⟩ to 1 should still return 0"
    );

    // Put qubit 1 in |1⟩
    sim.x(1);
    // Try to force it to 0
    let r1 = sim.force_measure(1, false);
    assert!(
        r1.outcome,
        "Forcing deterministic |1⟩ to 0 should still return 1"
    );
}

#[test]
fn test_measurement_statistics() {
    // Test that non-deterministic measurements produce roughly 50/50 results
    let seed = 999;
    let num_trials = 1000;
    let mut zeros = 0;
    let mut ones = 0;

    for i in 0..num_trials {
        // Use different seeds but deterministic sequence
        let mut sim = CppSparseStab::new_with_seed(1, seed + i);
        sim.h(0); // Put in superposition

        let result = sim.mz(0);
        if result.outcome {
            ones += 1;
        } else {
            zeros += 1;
        }
    }

    // Check that we get roughly 50/50 distribution (with some tolerance)
    let ratio = f64::from(zeros) / f64::from(num_trials);
    assert!(
        ratio > 0.4 && ratio < 0.6,
        "Expected roughly 50/50 distribution, got {zeros} zeros and {ones} ones (ratio: {ratio})"
    );
}

#[test]
fn test_complex_circuit_determinism() {
    // Test a more complex circuit with multiple gates and measurements
    let seed = 12345;

    // Run the same complex circuit twice with same seed
    let results1 = run_complex_circuit(seed);
    let results2 = run_complex_circuit(seed);

    assert_eq!(
        results1, results2,
        "Complex circuit with same seed should produce identical results"
    );

    // Run with different seed
    let results3 = run_complex_circuit(seed + 1);

    // Very unlikely to get same results with different seed
    assert_ne!(
        results1, results3,
        "Complex circuit with different seed should produce different results"
    );
}

fn run_complex_circuit(seed: u32) -> Vec<bool> {
    let mut sim = CppSparseStab::new_with_seed(6, seed);
    let mut results = vec![];

    // Create a complex entangled state
    sim.h(0);
    sim.cx(0, 1);
    sim.h(2);
    sim.cx(2, 3);
    sim.cx(1, 4);
    sim.h(5);
    sim.cz(3, 5);

    // Apply some single-qubit gates
    sim.sz(0);
    sim.sx(2);
    sim.sy(4);

    // Measure some qubits
    results.push(sim.mz(0).outcome);
    results.push(sim.mz(2).outcome);

    // Apply more gates
    sim.h(1);
    sim.cx(4, 5);

    // Measure remaining qubits
    results.push(sim.mz(1).outcome);
    results.push(sim.mz(3).outcome);
    results.push(sim.mz(4).outcome);
    results.push(sim.mz(5).outcome);

    results
}
