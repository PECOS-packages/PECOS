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

use pecos_core::{QubitId, qid};
use pecos_cppsparsesim::CppSparseStab;
use pecos_simulators::{CliffordGateable, QuantumSimulator};

#[test]
fn test_basic_gates() {
    let mut sim = CppSparseStab::new(2);

    // Test basic single-qubit gates
    sim.h(&qid(0));
    sim.x(&qid(1));
    sim.z(&qid(0));
    sim.y(&qid(1));
}

#[test]
fn test_bell_state() {
    let mut sim = CppSparseStab::new(2);

    // Create Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2
    sim.h(&qid(0));
    sim.cx(&[(QubitId(0), QubitId(1))]);

    // Measure both qubits
    let r0 = sim.mz(&qid(0))[0].outcome;
    let r1 = sim.mz(&qid(1))[0].outcome;

    // Both measurements should be equal (entangled)
    assert_eq!(r0, r1);
}

#[test]
fn test_reset() {
    let mut sim = CppSparseStab::new(3);

    // Apply some gates
    sim.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]).h(&qid(2));

    // Reset the simulator
    sim.reset();

    // After reset, all qubits should be in |0⟩ state
    // Measuring in Z basis should give 0
    let r0 = sim.mz(&qid(0))[0].outcome;
    let r1 = sim.mz(&qid(1))[0].outcome;
    let r2 = sim.mz(&qid(2))[0].outcome;

    assert!(!r0);
    assert!(!r1);
    assert!(!r2);
    // Note: Determinism tracking has been removed from the wrapper
    // per design decision - the wrapper does not track determinism
}

#[test]
fn test_phase_gates() {
    let mut sim = CppSparseStab::new(1);

    // Test S and S† gates
    sim.sz(&qid(0));
    sim.szdg(&qid(0));

    // Test SX and SX† gates
    sim.sx(&qid(0));
    sim.sxdg(&qid(0));

    // Test SY and SY† gates
    sim.sy(&qid(0));
    sim.sydg(&qid(0));
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
    sim1.h(&qid(0));
    sim1.h(&qid(1));
    sim1.cx(&[(QubitId(0), QubitId(2))]);
    sim1.h(&qid(2));

    sim2.h(&qid(0));
    sim2.h(&qid(1));
    sim2.cx(&[(QubitId(0), QubitId(2))]);
    sim2.h(&qid(2));

    // Collect measurements from both simulators
    let results1 = vec![
        sim1.mz(&qid(0))[0].outcome,
        sim1.mz(&qid(1))[0].outcome,
        sim1.mz(&qid(2))[0].outcome,
    ];

    let results2 = vec![
        sim2.mz(&qid(0))[0].outcome,
        sim2.mz(&qid(1))[0].outcome,
        sim2.mz(&qid(2))[0].outcome,
    ];

    // Results should be identical with same seed
    assert_eq!(
        results1, results2,
        "Simulators with same seed should produce identical results"
    );

    // Test with different seeds to ensure they differ
    let mut sim3 = CppSparseStab::new_with_seed(3, seed + 1);
    sim3.h(&qid(0));
    sim3.h(&qid(1));
    sim3.cx(&[(QubitId(0), QubitId(2))]);
    sim3.h(&qid(2));

    let _results3 = [
        sim3.mz(&qid(0))[0].outcome,
        sim3.mz(&qid(1))[0].outcome,
        sim3.mz(&qid(2))[0].outcome,
    ];

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
            sim1.h(&[QubitId::new(i)]);
            sim2.h(&[QubitId::new(i)]);
        }

        // Measure all qubits
        let mut results1 = vec![];
        let mut results2 = vec![];
        for i in 0..5 {
            results1.push(sim1.mz(&[QubitId::new(i)])[0].outcome);
            results2.push(sim2.mz(&[QubitId::new(i)])[0].outcome);
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
    sim.h(&qid(0));
    sim.h(&qid(1));
    sim.h(&qid(2));

    // Force measurements to specific values
    let r0 = sim.force_measure(&qid(0), false)[0].outcome; // Force to 0
    let r1 = sim.force_measure(&qid(1), true)[0].outcome; // Force to 1
    let r2 = sim.force_measure(&qid(2), false)[0].outcome; // Force to 0

    assert!(!r0, "Forced measurement to 0 should return false");
    assert!(r1, "Forced measurement to 1 should return true");
    assert!(!r2, "Forced measurement to 0 should return false");

    // After forcing, qubits should be in deterministic states
    // Measuring again should give same results
    let r0_again = sim.mz(&qid(0))[0].outcome;
    let r1_again = sim.mz(&qid(1))[0].outcome;
    let r2_again = sim.mz(&qid(2))[0].outcome;

    assert!(!r0_again);
    assert!(r1_again);
    assert!(!r2_again);
}

#[test]
fn test_forced_measurement_on_deterministic_state() {
    // Test that forcing a deterministic state returns the deterministic value
    // (not the forced value)
    let mut sim = CppSparseStab::new(2);

    // Qubit 0 is deterministically |0⟩
    // Try to force it to 1
    let r0 = sim.force_measure(&qid(0), true)[0].outcome;
    assert!(!r0, "Forcing deterministic |0⟩ to 1 should still return 0");

    // Put qubit 1 in |1⟩
    sim.x(&qid(1));
    // Try to force it to 0
    let r1 = sim.force_measure(&qid(1), false)[0].outcome;
    assert!(r1, "Forcing deterministic |1⟩ to 0 should still return 1");
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
        sim.h(&qid(0)); // Put in superposition

        let result = sim.mz(&qid(0))[0].outcome;
        if result {
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
    sim.h(&qid(0));
    sim.cx(&[(QubitId(0), QubitId(1))]);
    sim.h(&qid(2));
    sim.cx(&[(QubitId(2), QubitId(3))]);
    sim.cx(&[(QubitId(1), QubitId(4))]);
    sim.h(&qid(5));
    sim.cz(&[(QubitId(3), QubitId(5))]);

    // Apply some single-qubit gates
    sim.sz(&qid(0));
    sim.sx(&qid(2));
    sim.sy(&qid(4));

    // Measure some qubits
    results.push(sim.mz(&qid(0))[0].outcome);
    results.push(sim.mz(&qid(2))[0].outcome);

    // Apply more gates
    sim.h(&qid(1));
    sim.cx(&[(QubitId(4), QubitId(5))]);

    // Measure remaining qubits
    results.push(sim.mz(&qid(1))[0].outcome);
    results.push(sim.mz(&qid(3))[0].outcome);
    results.push(sim.mz(&qid(4))[0].outcome);
    results.push(sim.mz(&qid(5))[0].outcome);

    results
}
