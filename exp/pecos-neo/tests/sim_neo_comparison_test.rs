// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

// statistical tests use count as f64
#![allow(clippy::cast_precision_loss)]
//! Comparison tests between `sim_neo()` (Tool architecture) and `sim()` (pecos-engines).
//!
//! These tests verify that the new Tool architecture produces equivalent results
//! to the established pecos-engines simulation system.

use pecos_core::QubitId;
use pecos_engines::GeneralNoiseModelBuilder as EnginesNoiseBuilder;
use pecos_engines::sim;
use pecos_neo::noise::GeneralNoiseModelBuilder;
use pecos_neo::prelude::*;
use pecos_neo::tool::sim_neo;
use pecos_qasm::qasm_engine;
use std::collections::BTreeMap;

const NUM_SHOTS: usize = 1000;
const TOLERANCE_PERCENT: f64 = 10.0;

// --- Helper Functions ---

/// Extract measurement outcomes from pecos-engines `ShotVec`.
fn extract_engines_outcomes(
    results: &pecos_engines::ShotVec,
    register_name: &str,
    num_bits: usize,
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();

    for shot in &results.shots {
        let value = shot
            .data
            .get(register_name)
            .and_then(pecos_engines::prelude::Data::as_u32)
            .unwrap_or(0);

        let bits: String = (0..num_bits)
            .map(|i| if (value >> i) & 1 == 1 { '1' } else { '0' })
            .collect();

        *counts.entry(bits).or_insert(0) += 1;
    }

    counts
}

/// Extract measurement outcomes from `sim_neo` `SimulationResults`.
fn extract_neo_outcomes(
    results: &pecos_neo::tool::SimulationResults,
    qubit_ids: &[QubitId],
) -> BTreeMap<String, usize> {
    let mut counts = BTreeMap::new();

    for outcome in &results.outcomes {
        let bits: String = qubit_ids
            .iter()
            .map(|&q| {
                if outcome.get_bit(q).unwrap_or(false) {
                    '1'
                } else {
                    '0'
                }
            })
            .collect();

        *counts.entry(bits).or_insert(0) += 1;
    }

    counts
}

/// Compare two outcome distributions within tolerance.
fn distributions_match(
    dist1: &BTreeMap<String, usize>,
    dist2: &BTreeMap<String, usize>,
    total_shots: usize,
    tolerance_percent: f64,
) -> bool {
    let mut all_keys: Vec<_> = dist1.keys().chain(dist2.keys()).cloned().collect();
    all_keys.sort();
    all_keys.dedup();

    for key in all_keys {
        let count1 = *dist1.get(&key).unwrap_or(&0) as f64;
        let count2 = *dist2.get(&key).unwrap_or(&0) as f64;

        let rate1 = count1 / total_shots as f64;
        let rate2 = count2 / total_shots as f64;

        let diff = (rate1 - rate2).abs();
        let tolerance = tolerance_percent / 100.0;

        if diff > tolerance {
            eprintln!(
                "Distribution mismatch for '{key}': {rate1:.4} vs {rate2:.4} (diff: {diff:.4})"
            );
            return false;
        }
    }

    true
}

// --- Basic Circuit Tests (No Noise) ---

#[test]
fn test_sim_neo_vs_sim_deterministic_x() {
    // Deterministic circuit: prep |0>, X, measure -> always 1
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
    "#;

    // Run with sim() (pecos-engines)
    let engines_results = sim(qasm_engine().qasm(qasm))
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();
    let engines_counts = extract_engines_outcomes(&engines_results, "c", 1);

    // Run with sim_neo() (Tool architecture)
    let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

    let neo_results = sim_neo(circuit).shots(NUM_SHOTS).seed(42).build().run();
    let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0)]);

    // Both should produce all 1s
    assert_eq!(
        engines_counts.get("1").copied().unwrap_or(0),
        NUM_SHOTS,
        "engines: X gate should always produce 1"
    );
    assert_eq!(
        neo_counts.get("1").copied().unwrap_or(0),
        NUM_SHOTS,
        "neo: X gate should always produce 1"
    );

    println!("sim() results: {engines_counts:?}");
    println!("sim_neo() results: {neo_counts:?}");
}

#[test]
fn test_sim_neo_vs_sim_hadamard() {
    // Hadamard: should produce ~50/50 distribution
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
    "#;

    // Run with sim()
    let engines_results = sim(qasm_engine().qasm(qasm))
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();
    let engines_counts = extract_engines_outcomes(&engines_results, "c", 1);

    // Run with sim_neo()
    let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

    let neo_results = sim_neo(circuit).shots(NUM_SHOTS).seed(42).build().run();
    let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0)]);

    // Both should be roughly 50/50
    let engines_ones = *engines_counts.get("1").unwrap_or(&0) as f64 / NUM_SHOTS as f64;
    let neo_ones = *neo_counts.get("1").unwrap_or(&0) as f64 / NUM_SHOTS as f64;

    println!("sim() ones rate: {engines_ones:.4}");
    println!("sim_neo() ones rate: {neo_ones:.4}");

    assert!(
        (engines_ones - 0.5).abs() < TOLERANCE_PERCENT / 100.0,
        "engines: Hadamard should give ~50% ones"
    );
    assert!(
        (neo_ones - 0.5).abs() < TOLERANCE_PERCENT / 100.0,
        "neo: Hadamard should give ~50% ones"
    );

    // Both distributions should be similar
    assert!(
        distributions_match(&engines_counts, &neo_counts, NUM_SHOTS, TOLERANCE_PERCENT),
        "Distributions should match"
    );
}

#[test]
fn test_sim_neo_vs_sim_bell_state() {
    // Bell state: should produce correlated 00 and 11 only
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    // Run with sim()
    let engines_results = sim(qasm_engine().qasm(qasm))
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();
    let engines_counts = extract_engines_outcomes(&engines_results, "c", 2);

    // Run with sim_neo()
    let circuit = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    let neo_results = sim_neo(circuit).shots(NUM_SHOTS).seed(42).build().run();
    let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0), QubitId(1)]);

    // Both should only have 00 and 11
    let engines_valid =
        engines_counts.get("00").unwrap_or(&0) + engines_counts.get("11").unwrap_or(&0);
    let neo_valid = neo_counts.get("00").unwrap_or(&0) + neo_counts.get("11").unwrap_or(&0);

    println!("sim() counts: {engines_counts:?}");
    println!("sim_neo() counts: {neo_counts:?}");

    assert_eq!(
        engines_valid, NUM_SHOTS,
        "engines: Bell state should only produce 00 or 11"
    );
    assert_eq!(
        neo_valid, NUM_SHOTS,
        "neo: Bell state should only produce 00 or 11"
    );

    // Distributions should be similar (both ~50/50 for 00 and 11)
    assert!(
        distributions_match(&engines_counts, &neo_counts, NUM_SHOTS, TOLERANCE_PERCENT),
        "Bell state distributions should match"
    );
}

#[test]
fn test_sim_neo_vs_sim_depolarizing_noise() {
    // Test single-qubit depolarizing noise
    let p1 = 0.05;

    // pecos-engines noise model builder
    let engines_noise = EnginesNoiseBuilder::new().with_average_p1_probability(p1 / 1.5); // Scale factor for engines

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
    "#;

    // Run with sim()
    let engines_results = sim(qasm_engine().qasm(qasm))
        .noise(engines_noise)
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();
    let engines_counts = extract_engines_outcomes(&engines_results, "c", 1);

    // Run with sim_neo()
    let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

    let neo_noise = GeneralNoiseModelBuilder::new().with_p1(p1).build();

    let neo_results = sim_neo(circuit)
        .noise(neo_noise)
        .shots(NUM_SHOTS)
        .seed(42)
        .build()
        .run();
    let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0)]);

    // Both should have mostly 1s with some errors
    let engines_error_rate = *engines_counts.get("0").unwrap_or(&0) as f64 / NUM_SHOTS as f64;
    let neo_error_rate = *neo_counts.get("0").unwrap_or(&0) as f64 / NUM_SHOTS as f64;

    println!("sim() error rate: {engines_error_rate:.4}");
    println!("sim_neo() error rate: {neo_error_rate:.4}");

    // Both should have some errors but not too many
    assert!(
        engines_error_rate > 0.0 && engines_error_rate < 0.3,
        "engines should have some errors"
    );
    assert!(
        neo_error_rate > 0.0 && neo_error_rate < 0.3,
        "neo should have some errors"
    );

    // Error rates should be similar (with wider tolerance due to noise model differences)
    let diff = (engines_error_rate - neo_error_rate).abs();
    assert!(
        diff < 0.15, // 15% tolerance for noise model differences
        "Error rates should be similar: {engines_error_rate:.4} vs {neo_error_rate:.4}"
    );
}

#[test]
fn test_sim_neo_vs_sim_measurement_noise() {
    // Test measurement errors
    let p_meas = 0.10;

    // pecos-engines noise model
    let engines_noise = EnginesNoiseBuilder::new()
        .with_meas_0_probability(p_meas)
        .with_meas_1_probability(p_meas);

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        measure q[0] -> c[0];
    "#;

    // Run with sim()
    let engines_results = sim(qasm_engine().qasm(qasm))
        .noise(engines_noise)
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();
    let engines_counts = extract_engines_outcomes(&engines_results, "c", 1);

    // Run with sim_neo()
    let circuit = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

    let neo_noise = GeneralNoiseModelBuilder::new()
        .with_p_meas(p_meas, p_meas)
        .build();

    let neo_results = sim_neo(circuit)
        .noise(neo_noise)
        .shots(NUM_SHOTS)
        .seed(42)
        .build()
        .run();
    let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0)]);

    // Error rate should be ~p_meas (measuring |0> but getting 1)
    let engines_error_rate = *engines_counts.get("1").unwrap_or(&0) as f64 / NUM_SHOTS as f64;
    let neo_error_rate = *neo_counts.get("1").unwrap_or(&0) as f64 / NUM_SHOTS as f64;

    println!("Expected error rate: {p_meas:.4}");
    println!("sim() error rate: {engines_error_rate:.4}");
    println!("sim_neo() error rate: {neo_error_rate:.4}");

    // Both should be close to p_meas
    assert!(
        (engines_error_rate - p_meas).abs() < TOLERANCE_PERCENT / 100.0,
        "engines error rate should be close to p_meas"
    );
    assert!(
        (neo_error_rate - p_meas).abs() < TOLERANCE_PERCENT / 100.0,
        "neo error rate should be close to p_meas"
    );
}

// --- Conditional Execution Tests ---

#[test]
fn test_sim_neo_vs_sim_conditional_x_correction() {
    use pecos_neo::program::{ConditionalProgram, ProgramRunner};
    use pecos_simulators::SparseStab;

    // Test: measure qubit, if result is 1, apply X to flip it back
    // QASM: if (c[0] == 1) x q[0];
    //
    // This simulates a simple error correction pattern

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        if (c[0] == 1) x q[0];
        measure q[0] -> c[0];
    "#;

    // Run with sim() - uses QASM conditional
    let engines_results = sim(qasm_engine().qasm(qasm))
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();

    // Count final measurement outcomes (second measurement stored in c[0])
    let mut engines_zeros = 0;
    for shot in &engines_results.shots {
        if let Some(val) = shot
            .data
            .get("c")
            .and_then(pecos_engines::prelude::Data::as_u32)
            && val == 0
        {
            engines_zeros += 1;
        }
    }

    // Run with pecos-neo ProgramRunner + ConditionalProgram
    let initial = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

    let branch = |outcomes: &pecos_neo::outcome::MeasurementOutcomes| {
        // If measured 1, apply X to flip back to 0, then measure again
        if outcomes.get_bit(QubitId(0)) == Some(true) {
            Some(CommandBuilder::new().x(&[0]).mz(&[0]).build())
        } else {
            // If measured 0, just measure again
            Some(CommandBuilder::new().mz(&[0]).build())
        }
    };

    let mut neo_zeros = 0;
    for shot_idx in 0..NUM_SHOTS {
        let mut program = ConditionalProgram::new(initial.clone(), branch, 1);
        let seed = 42u64.wrapping_add(shot_idx as u64);
        let mut runner = ProgramRunner::new(SparseStab::new(1)).with_seed(seed);

        let result = runner.run_shot(&mut program);

        // The final measurement should always be 0 (after correction)
        // Get the last measurement outcome
        if result.outcomes.len() >= 2 {
            // Second measurement is at index 1
            if !result.outcomes.get_bit(QubitId(0)).unwrap_or(true) {
                neo_zeros += 1;
            }
        }
    }

    // After X correction, we should always get 0
    // (The conditional flips 1->0, and 0 stays 0)
    let engines_zero_rate = f64::from(engines_zeros) / NUM_SHOTS as f64;
    let neo_zero_rate = f64::from(neo_zeros) / NUM_SHOTS as f64;

    println!("Conditional X correction:");
    println!("  engines zero rate: {engines_zero_rate:.4}");
    println!("  neo zero rate: {neo_zero_rate:.4}");

    // Both should produce 100% zeros after correction
    assert!(
        engines_zero_rate > 0.95,
        "engines should produce mostly zeros after correction: {engines_zero_rate:.4}"
    );
    assert!(
        neo_zero_rate > 0.95,
        "neo should produce mostly zeros after correction: {neo_zero_rate:.4}"
    );
}

#[test]
fn test_sim_neo_vs_sim_conditional_with_noise() {
    use pecos_neo::program::{ConditionalProgram, ProgramRunner};
    use pecos_simulators::SparseStab;

    // Test conditional with noise - the correction may fail due to noise
    let p1 = 0.10;

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
        if (c[0] == 1) x q[0];
        measure q[0] -> c[0];
    "#;

    let engines_noise = EnginesNoiseBuilder::new().with_average_p1_probability(p1 / 1.5);

    // Run with sim()
    let engines_results = sim(qasm_engine().qasm(qasm))
        .noise(engines_noise)
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();

    let mut engines_zeros = 0;
    for shot in &engines_results.shots {
        if let Some(val) = shot
            .data
            .get("c")
            .and_then(pecos_engines::prelude::Data::as_u32)
            && val == 0
        {
            engines_zeros += 1;
        }
    }

    // Run with pecos-neo
    let initial = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

    let branch = |outcomes: &pecos_neo::outcome::MeasurementOutcomes| {
        if outcomes.get_bit(QubitId(0)) == Some(true) {
            Some(CommandBuilder::new().x(&[0]).mz(&[0]).build())
        } else {
            Some(CommandBuilder::new().mz(&[0]).build())
        }
    };

    let mut neo_zeros = 0;
    for shot_idx in 0..NUM_SHOTS {
        let mut program = ConditionalProgram::new(initial.clone(), branch, 1);
        let seed = 42u64.wrapping_add(shot_idx as u64);
        // Create fresh noise model for each shot
        let neo_noise = GeneralNoiseModelBuilder::new().with_p1(p1).build();
        let mut runner = ProgramRunner::new(SparseStab::new(1))
            .with_noise(neo_noise)
            .with_seed(seed);

        let result = runner.run_shot(&mut program);

        if !result.outcomes.get_bit(QubitId(0)).unwrap_or(true) {
            neo_zeros += 1;
        }
    }

    let engines_zero_rate = f64::from(engines_zeros) / NUM_SHOTS as f64;
    let neo_zero_rate = f64::from(neo_zeros) / NUM_SHOTS as f64;

    println!("Conditional with noise (p1={p1}):");
    println!("  engines zero rate: {engines_zero_rate:.4}");
    println!("  neo zero rate: {neo_zero_rate:.4}");

    // With noise, we expect less than 100% zeros
    // But both should have similar rates
    let diff = (engines_zero_rate - neo_zero_rate).abs();
    assert!(
        diff < 0.15,
        "Conditional noise results should be similar: {engines_zero_rate:.4} vs {neo_zero_rate:.4}"
    );
}

#[test]
fn test_sim_neo_vs_sim_teleportation_style() {
    // Test a teleportation-style pattern where we apply corrections based on measurements
    //
    // Pattern:
    // 1. Prepare Bell pair on qubits 1,2
    // 2. Prepare qubit 0 in |+> state
    // 3. CNOT 0->1, H on 0
    // 4. Measure 0 and 1
    // 5. Apply corrections on qubit 2 based on measurements

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];

        // Create Bell pair on q[1], q[2]
        h q[1];
        cx q[1], q[2];

        // Prepare q[0] in |+>
        h q[0];

        // Bell measurement on q[0], q[1]
        cx q[0], q[1];
        h q[0];
        measure q[0] -> c[0];
        measure q[1] -> c[1];

        // Corrections on q[2]
        if (c[1] == 1) x q[2];
        if (c[0] == 1) z q[2];

        // Final measurement
        h q[2];
        measure q[2] -> c[2];
    "#;

    // Run with sim()
    let engines_results = sim(qasm_engine().qasm(qasm))
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();

    // After teleportation, q[2] should be in |+> state
    // H|+> = |0>, so we should always measure 0
    let mut engines_zeros = 0;
    for shot in &engines_results.shots {
        if let Some(val) = shot
            .data
            .get("c")
            .and_then(pecos_engines::prelude::Data::as_u32)
        {
            // c[2] is the bit at position 2 (bit 2)
            if (val >> 2) & 1 == 0 {
                engines_zeros += 1;
            }
        }
    }

    let engines_zero_rate = f64::from(engines_zeros) / NUM_SHOTS as f64;

    println!("Teleportation-style conditional:");
    println!("  engines final zero rate: {engines_zero_rate:.4}");

    // The teleported state should give |0> after H (from |+>)
    assert!(
        engines_zero_rate > 0.95,
        "Teleportation should preserve state: {engines_zero_rate:.4}"
    );
}

// --- Ergonomic Noise API Tests ---

#[test]
fn test_sim_neo_ergonomic_builder_direct() {
    // Test passing GeneralNoiseModelBuilder directly without .build()
    let p1 = 0.10;

    // pecos-engines
    let engines_noise = EnginesNoiseBuilder::new().with_average_p1_probability(p1 / 1.5);

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
    "#;

    let engines_results = sim(qasm_engine().qasm(qasm))
        .noise(engines_noise)
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();
    let engines_counts = extract_engines_outcomes(&engines_results, "c", 1);

    // pecos-neo - pass builder directly, no .build()!
    let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

    let neo_results = sim_neo(circuit)
        .noise(GeneralNoiseModelBuilder::new().with_p1(p1)) // No .build()!
        .shots(NUM_SHOTS)
        .seed(42)
        .build()
        .run();
    let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0)]);

    // Both should have similar error rates
    let engines_error = *engines_counts.get("0").unwrap_or(&0) as f64 / NUM_SHOTS as f64;
    let neo_error = *neo_counts.get("0").unwrap_or(&0) as f64 / NUM_SHOTS as f64;

    let diff = (engines_error - neo_error).abs();
    assert!(
        diff < 0.10,
        "Ergonomic API should produce similar results: {engines_error:.4} vs {neo_error:.4}"
    );
}

#[test]
fn test_sim_neo_convenience_methods() {
    // Test the convenience methods like .depolarizing(p)
    // .depolarizing(p) applies noise to gates, prep, and measurement
    let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

    // Using .depolarizing() convenience method
    let results_convenience = sim_neo(circuit.clone())
        .depolarizing(0.05)
        .shots(500)
        .seed(42)
        .build()
        .run();

    // Using explicit GeneralNoiseModelBuilder - must match what .depolarizing() does
    let results_explicit = sim_neo(circuit)
        .noise(
            GeneralNoiseModelBuilder::new()
                .with_p1(0.05)
                .with_p2(0.05)
                .with_p_prep(0.05)
                .with_p_meas_symmetric(0.05),
        )
        .shots(500)
        .seed(42)
        .build()
        .run();

    // Both should produce similar results
    let conv_errors: usize = results_convenience
        .outcomes
        .iter()
        .filter(|o| !o.get_bit(QubitId(0)).unwrap_or(true))
        .count();
    let explicit_errors: usize = results_explicit
        .outcomes
        .iter()
        .filter(|o| !o.get_bit(QubitId(0)).unwrap_or(true))
        .count();

    // Same seed = same results
    assert_eq!(
        conv_errors, explicit_errors,
        "Convenience methods should produce identical results to explicit configuration"
    );
}

// --- Reusability Tests ---

#[test]
fn test_sim_neo_reusable() {
    // Test that sim_neo Simulation handle can be rerun with different configs
    let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

    let mut sim = sim_neo(circuit).shots(100).seed(42).build();

    // First run
    let results1 = sim.run();
    assert_eq!(results1.len(), 100);

    // Reconfigure and run again
    sim.shots(200).seed(123);
    let results2 = sim.run();
    assert_eq!(results2.len(), 200);

    // Results should be different (different seeds)
    let ones1 = results1
        .outcomes
        .iter()
        .filter(|o| o.get_bit(QubitId(0)).unwrap_or(false))
        .count();
    let ones2 = results2
        .outcomes
        .iter()
        .filter(|o| o.get_bit(QubitId(0)).unwrap_or(false))
        .count();

    // Both should be roughly 50%
    let rate1 = ones1 as f64 / 100.0;
    let rate2 = ones2 as f64 / 200.0;

    assert!(
        (rate1 - 0.5).abs() < 0.2,
        "First run should be ~50%: {rate1:.4}"
    );
    assert!(
        (rate2 - 0.5).abs() < 0.2,
        "Second run should be ~50%: {rate2:.4}"
    );
}

#[test]
fn test_sim_neo_determinism() {
    // Same seed should produce identical results
    let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

    let results1 = sim_neo(circuit.clone()).shots(100).seed(42).build().run();

    let results2 = sim_neo(circuit).shots(100).seed(42).build().run();

    // Results should be identical
    for (o1, o2) in results1.outcomes.iter().zip(results2.outcomes.iter()) {
        assert_eq!(
            o1.get_bit(QubitId(0)),
            o2.get_bit(QubitId(0)),
            "Same seed should produce identical results"
        );
    }
}

// --- Noiseless vs Noisy Comparison Tests ---

#[test]
fn test_sim_neo_vs_sim_noiseless_exact() {
    // Verify that both systems produce identical results without noise
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    // Run with sim() - no noise
    let engines_results = sim(qasm_engine().qasm(qasm))
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();
    let engines_counts = extract_engines_outcomes(&engines_results, "c", 2);

    // Run with sim_neo() - no noise
    let circuit = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .x(&[0])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    let neo_results = sim_neo(circuit).shots(NUM_SHOTS).seed(42).build().run();
    let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0), QubitId(1)]);

    // Both should produce 100% |11>
    assert_eq!(
        engines_counts.get("11").copied().unwrap_or(0),
        NUM_SHOTS,
        "engines: X-CX should always produce |11>"
    );
    assert_eq!(
        neo_counts.get("11").copied().unwrap_or(0),
        NUM_SHOTS,
        "neo: X-CX should always produce |11>"
    );
}

#[test]
fn test_sim_neo_noise_level_scaling() {
    // Test that error rates scale appropriately with noise level
    let noise_levels = [0.01, 0.05, 0.10, 0.20];
    let mut engines_errors = Vec::new();
    let mut neo_errors = Vec::new();

    for &p1 in &noise_levels {
        // pecos-engines
        let engines_noise = EnginesNoiseBuilder::new().with_average_p1_probability(p1 / 1.5);

        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            x q[0];
            measure q[0] -> c[0];
        "#;

        let engines_results = sim(qasm_engine().qasm(qasm))
            .noise(engines_noise)
            .seed(42)
            .run(NUM_SHOTS)
            .unwrap();
        let engines_counts = extract_engines_outcomes(&engines_results, "c", 1);
        let eng_err = *engines_counts.get("0").unwrap_or(&0) as f64 / NUM_SHOTS as f64;

        // pecos-neo
        let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();
        let neo_noise = GeneralNoiseModelBuilder::new().with_p1(p1).build();

        let neo_results = sim_neo(circuit)
            .noise(neo_noise)
            .shots(NUM_SHOTS)
            .seed(42)
            .build()
            .run();
        let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0)]);
        let neo_err = *neo_counts.get("0").unwrap_or(&0) as f64 / NUM_SHOTS as f64;

        engines_errors.push(eng_err);
        neo_errors.push(neo_err);

        println!("p1={p1:.2}: engines={eng_err:.4}, neo={neo_err:.4}");
    }

    // Verify error rates increase with noise level
    for i in 1..noise_levels.len() {
        assert!(
            engines_errors[i] >= engines_errors[i - 1] * 0.8, // Allow some variance
            "engines errors should increase with noise level"
        );
        assert!(
            neo_errors[i] >= neo_errors[i - 1] * 0.8,
            "neo errors should increase with noise level"
        );
    }

    // Verify engines and neo track each other
    for i in 0..noise_levels.len() {
        let diff = (engines_errors[i] - neo_errors[i]).abs();
        assert!(
            diff < 0.10,
            "At p1={}, error rates should match: {:.4} vs {:.4}",
            noise_levels[i],
            engines_errors[i],
            neo_errors[i]
        );
    }
}

#[test]
fn test_sim_neo_vs_sim_zero_noise() {
    // Explicitly test with noise model but zero error rates
    let engines_noise = EnginesNoiseBuilder::new()
        .with_prep_probability(0.0)
        .with_average_p1_probability(0.0)
        .with_average_p2_probability(0.0)
        .with_meas_0_probability(0.0)
        .with_meas_1_probability(0.0);

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    // Run with sim() with zero-noise model
    let engines_results = sim(qasm_engine().qasm(qasm))
        .noise(engines_noise)
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();
    let engines_counts = extract_engines_outcomes(&engines_results, "c", 2);

    // Run with sim_neo() with zero-noise model
    let circuit = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    let neo_noise = GeneralNoiseModelBuilder::new()
        .with_p_prep(0.0)
        .with_p1(0.0)
        .with_p2(0.0)
        .with_p_meas(0.0, 0.0)
        .build();

    let neo_results = sim_neo(circuit)
        .noise(neo_noise)
        .shots(NUM_SHOTS)
        .seed(42)
        .build()
        .run();
    let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0), QubitId(1)]);

    // Bell state should be 100% correlated (only |00> and |11>)
    let engines_correlated =
        engines_counts.get("00").unwrap_or(&0) + engines_counts.get("11").unwrap_or(&0);
    let neo_correlated = neo_counts.get("00").unwrap_or(&0) + neo_counts.get("11").unwrap_or(&0);

    assert_eq!(
        engines_correlated, NUM_SHOTS,
        "engines: Bell state with zero noise should be 100% correlated"
    );
    assert_eq!(
        neo_correlated, NUM_SHOTS,
        "neo: Bell state with zero noise should be 100% correlated"
    );
}

#[test]
fn test_sim_neo_high_noise_chaos() {
    // Test behavior at high noise levels (near 50% depolarizing)
    let p1 = 0.40; // 40% depolarizing - very noisy

    let engines_noise = EnginesNoiseBuilder::new().with_average_p1_probability(p1 / 1.5);

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
    "#;

    let engines_results = sim(qasm_engine().qasm(qasm))
        .noise(engines_noise)
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();
    let engines_counts = extract_engines_outcomes(&engines_results, "c", 1);

    let circuit = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();
    let neo_noise = GeneralNoiseModelBuilder::new().with_p1(p1).build();

    let neo_results = sim_neo(circuit)
        .noise(neo_noise)
        .shots(NUM_SHOTS)
        .seed(42)
        .build()
        .run();
    let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0)]);

    // At high noise, we expect significant errors but not 50/50
    // (X and Y errors flip, Z keeps same - so 2/3 of errors flip)
    let engines_ones = *engines_counts.get("1").unwrap_or(&0) as f64 / NUM_SHOTS as f64;
    let neo_ones = *neo_counts.get("1").unwrap_or(&0) as f64 / NUM_SHOTS as f64;

    println!("High noise (p1={p1}): engines={engines_ones:.4}, neo={neo_ones:.4}");

    // Both should show significant deviation from ideal (100% ones)
    assert!(
        engines_ones < 0.9 && engines_ones > 0.3,
        "engines should have significant errors at high noise"
    );
    assert!(
        neo_ones < 0.9 && neo_ones > 0.3,
        "neo should have significant errors at high noise"
    );

    // Should still be similar to each other
    let diff = (engines_ones - neo_ones).abs();
    assert!(
        diff < 0.15,
        "High noise rates should match: {engines_ones:.4} vs {neo_ones:.4}"
    );
}

// --- Additional Noise Tests (using GeneralNoiseModel on both sides) ---

#[test]
fn test_sim_neo_vs_sim_two_qubit_noise() {
    // Test two-qubit depolarizing noise
    let p2 = 0.10;

    // pecos-engines noise model with scaling factor
    let engines_noise = EnginesNoiseBuilder::new().with_average_p2_probability(p2 / 1.25); // Scale factor for engines

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    // Run with sim()
    let engines_results = sim(qasm_engine().qasm(qasm))
        .noise(engines_noise)
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();
    let engines_counts = extract_engines_outcomes(&engines_results, "c", 2);

    // Run with sim_neo()
    let circuit = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .x(&[0])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    let neo_noise = GeneralNoiseModelBuilder::new().with_p2(p2).build();

    let neo_results = sim_neo(circuit)
        .noise(neo_noise)
        .shots(NUM_SHOTS)
        .seed(42)
        .build()
        .run();
    let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0), QubitId(1)]);

    // Expected: |11> with some errors from CX noise
    let engines_correct = *engines_counts.get("11").unwrap_or(&0) as f64 / NUM_SHOTS as f64;
    let neo_correct = *neo_counts.get("11").unwrap_or(&0) as f64 / NUM_SHOTS as f64;

    println!("sim() correct rate (|11>): {engines_correct:.4}");
    println!("sim_neo() correct rate (|11>): {neo_correct:.4}");

    // Both should have some errors
    assert!(
        engines_correct < 1.0 && engines_correct > 0.5,
        "engines should have some errors: {engines_correct:.4}"
    );
    assert!(
        neo_correct < 1.0 && neo_correct > 0.5,
        "neo should have some errors: {neo_correct:.4}"
    );

    // Error rates should be similar
    let diff = (engines_correct - neo_correct).abs();
    assert!(
        diff < 0.15,
        "Correct rates should be similar: {engines_correct:.4} vs {neo_correct:.4}"
    );
}

#[test]
fn test_sim_neo_vs_sim_preparation_noise() {
    // Test preparation errors
    let p_prep = 0.15;

    // pecos-engines noise model
    let engines_noise = EnginesNoiseBuilder::new().with_prep_probability(p_prep);

    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        reset q[0];
        measure q[0] -> c[0];
    "#;

    // Run with sim()
    let engines_results = sim(qasm_engine().qasm(qasm))
        .noise(engines_noise)
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();
    let engines_counts = extract_engines_outcomes(&engines_results, "c", 1);

    // Run with sim_neo()
    let circuit = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

    let neo_noise = GeneralNoiseModelBuilder::new().with_p_prep(p_prep).build();

    let neo_results = sim_neo(circuit)
        .noise(neo_noise)
        .shots(NUM_SHOTS)
        .seed(42)
        .build()
        .run();
    let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0)]);

    // Error rate should be ~p_prep (preparing |0> but getting |1>)
    let engines_error_rate = *engines_counts.get("1").unwrap_or(&0) as f64 / NUM_SHOTS as f64;
    let neo_error_rate = *neo_counts.get("1").unwrap_or(&0) as f64 / NUM_SHOTS as f64;

    println!("Expected error rate: {p_prep:.4}");
    println!("sim() error rate: {engines_error_rate:.4}");
    println!("sim_neo() error rate: {neo_error_rate:.4}");

    // Both should be close to p_prep
    assert!(
        (engines_error_rate - p_prep).abs() < TOLERANCE_PERCENT / 100.0,
        "engines error rate should be close to p_prep"
    );
    assert!(
        (neo_error_rate - p_prep).abs() < TOLERANCE_PERCENT / 100.0,
        "neo error rate should be close to p_prep"
    );
}

#[test]
fn test_sim_neo_vs_sim_combined_noise() {
    // Test with multiple noise sources active at once
    let p_prep = 0.02;
    let p1 = 0.05;
    let p2 = 0.10;
    let p_meas = 0.02;

    // pecos-engines noise model (with scaling factors)
    let engines_noise = EnginesNoiseBuilder::new()
        .with_prep_probability(p_prep)
        .with_average_p1_probability(p1 / 1.5)
        .with_average_p2_probability(p2 / 1.25)
        .with_meas_0_probability(p_meas)
        .with_meas_1_probability(p_meas);

    // Bell state circuit with noise
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        reset q[0];
        reset q[1];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    // Run with sim()
    let engines_results = sim(qasm_engine().qasm(qasm))
        .noise(engines_noise)
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();
    let engines_counts = extract_engines_outcomes(&engines_results, "c", 2);

    // Run with sim_neo()
    let circuit = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    let neo_noise = GeneralNoiseModelBuilder::new()
        .with_p_prep(p_prep)
        .with_p1(p1)
        .with_p2(p2)
        .with_p_meas(p_meas, p_meas)
        .build();

    let neo_results = sim_neo(circuit)
        .noise(neo_noise)
        .shots(NUM_SHOTS)
        .seed(42)
        .build()
        .run();
    let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0), QubitId(1)]);

    // For Bell state, ideal is 50% |00> and 50% |11> (correlated)
    // With noise, we expect some |01> and |10> (anti-correlated)
    let engines_correlated = (*engines_counts.get("00").unwrap_or(&0)
        + *engines_counts.get("11").unwrap_or(&0)) as f64
        / NUM_SHOTS as f64;
    let neo_correlated = (*neo_counts.get("00").unwrap_or(&0) + *neo_counts.get("11").unwrap_or(&0))
        as f64
        / NUM_SHOTS as f64;

    println!("sim() correlated rate: {engines_correlated:.4}");
    println!("sim_neo() correlated rate: {neo_correlated:.4}");

    // Both should have mostly correlated outcomes with some errors
    assert!(
        engines_correlated > 0.7 && engines_correlated < 1.0,
        "engines should be mostly correlated: {engines_correlated:.4}"
    );
    assert!(
        neo_correlated > 0.7 && neo_correlated < 1.0,
        "neo should be mostly correlated: {neo_correlated:.4}"
    );

    // Correlated rates should be similar (wider tolerance due to multiple noise sources)
    let diff = (engines_correlated - neo_correlated).abs();
    assert!(
        diff < 0.20,
        "Correlated rates should be similar: {engines_correlated:.4} vs {neo_correlated:.4}"
    );
}

// --- Multi-qubit Tests ---

#[test]
fn test_sim_neo_vs_sim_ghz_state() {
    // GHZ state: |000> + |111>
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        cx q[0], q[1];
        cx q[1], q[2];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];
    "#;

    // Run with sim()
    let engines_results = sim(qasm_engine().qasm(qasm))
        .seed(42)
        .run(NUM_SHOTS)
        .unwrap();
    let engines_counts = extract_engines_outcomes(&engines_results, "c", 3);

    // Run with sim_neo()
    let circuit = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .pz(&[2])
        .h(&[0])
        .cx(&[(0, 1)])
        .cx(&[(1, 2)])
        .mz(&[0])
        .mz(&[1])
        .mz(&[2])
        .build();

    let neo_results = sim_neo(circuit).shots(NUM_SHOTS).seed(42).build().run();
    let neo_counts = extract_neo_outcomes(&neo_results, &[QubitId(0), QubitId(1), QubitId(2)]);

    // Both should only have 000 and 111
    let engines_valid =
        engines_counts.get("000").unwrap_or(&0) + engines_counts.get("111").unwrap_or(&0);
    let neo_valid = neo_counts.get("000").unwrap_or(&0) + neo_counts.get("111").unwrap_or(&0);

    println!("sim() GHZ counts: {engines_counts:?}");
    println!("sim_neo() GHZ counts: {neo_counts:?}");

    assert_eq!(
        engines_valid, NUM_SHOTS,
        "engines: GHZ should only produce 000 or 111"
    );
    assert_eq!(
        neo_valid, NUM_SHOTS,
        "neo: GHZ should only produce 000 or 111"
    );

    assert!(
        distributions_match(&engines_counts, &neo_counts, NUM_SHOTS, TOLERANCE_PERCENT),
        "GHZ distributions should match"
    );
}
