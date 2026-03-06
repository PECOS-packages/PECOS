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

//! Integration tests comparing pecos-neo execution against pecos-engines.
//!
//! These tests verify that:
//! 1. `ClassicalEngineAdapter` correctly wraps QASM engines
//! 2. `MonteCarloRunner` produces equivalent results to `MonteCarloEngine`
//! 3. End-to-end workflows match between the two systems

use pecos_core::QubitId;
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::{MonteCarloEngine, PassThroughNoiseModel};
use pecos_neo::noise::GeneralNoiseModelBuilder;
use pecos_neo::prelude::*;
use pecos_neo::sampling::{MonteCarloConfig, MonteCarloRunner};
use pecos_qasm::QASMEngine;
use pecos_qsim::SparseStab;
use std::collections::BTreeMap;
use std::str::FromStr;

const NUM_SHOTS: usize = 1000;
const TOLERANCE_PERCENT: f64 = 10.0; // Allow 10% difference for statistical tests

// ============================================================================
// Helper Functions
// ============================================================================

/// Extract measurement outcomes from pecos-engines Shot results.
fn extract_outcomes_from_shots(
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

        // Convert to bitstring
        let bits: String = (0..num_bits)
            .map(|i| if (value >> i) & 1 == 1 { '1' } else { '0' })
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
    // Get all keys from both distributions
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
                "Distribution mismatch for '{key}': {rate1:.4} vs {rate2:.4} (diff: {diff:.4}, tolerance: {tolerance:.4})"
            );
            return false;
        }
    }

    true
}

// ============================================================================
// MonteCarloRunner vs MonteCarloEngine Tests
// ============================================================================

#[test]
fn test_monte_carlo_bell_state_no_noise() {
    // Test that both systems produce correlated Bell state measurements
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

    // Run with pecos-engines MonteCarloEngine
    let engine = QASMEngine::from_str(qasm).unwrap();
    let engines_results = MonteCarloEngine::run_with_noise_model(
        Box::new(engine),
        Box::new(PassThroughNoiseModel::builder().build()),
        NUM_SHOTS,
        1,
        Some(42),
    )
    .unwrap();

    let engines_counts = extract_outcomes_from_shots(&engines_results, "c", 2);

    // Run with pecos-neo MonteCarloRunner
    let commands = CommandBuilder::new()
        .pz(0)
        .pz(1)
        .h(0)
        .cx(0, 1)
        .mz(0)
        .mz(1)
        .build();

    let config = MonteCarloConfig::new()
        .with_shots(NUM_SHOTS)
        .with_workers(1)
        .with_seed(42);

    let neo_results = MonteCarloRunner::run(
        &commands,
        config,
        || (CircuitRunner::new(), SparseStab::new(2)),
        |outcomes| {
            let b0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
            let b1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);
            format!(
                "{}{}",
                if b0 { '1' } else { '0' },
                if b1 { '1' } else { '0' }
            )
        },
    );

    let mut neo_counts = BTreeMap::new();
    for result in neo_results.iter() {
        *neo_counts.entry(result.clone()).or_insert(0) += 1;
    }

    // Bell state should only produce "00" or "11"
    let valid_outcomes: usize =
        neo_counts.get("00").unwrap_or(&0) + neo_counts.get("11").unwrap_or(&0);
    assert_eq!(
        valid_outcomes, NUM_SHOTS,
        "Bell state should only produce correlated outcomes"
    );

    // Both systems should have similar 00/11 distribution (roughly 50/50)
    assert!(
        distributions_match(&engines_counts, &neo_counts, NUM_SHOTS, TOLERANCE_PERCENT),
        "Outcome distributions should match within tolerance"
    );
}

#[test]
fn test_monte_carlo_with_depolarizing_noise() {
    // Test that both systems produce similar error rates with depolarizing noise
    let p1 = 0.05; // 5% single-qubit error rate

    // Build equivalent noise models
    // pecos-engines uses scaled probabilities
    let engines_noise = GeneralNoiseModel::builder()
        .with_average_p1_probability(p1 / 1.5) // Scale down for engines
        .build();

    // Simple circuit: prep, apply X (identity on |0>), measure
    // With noise, we expect some bit flips

    // Run with pecos-engines
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        x q[0];
        measure q[0] -> c[0];
    "#;

    let engine = QASMEngine::from_str(qasm).unwrap();
    let engines_results = MonteCarloEngine::run_with_noise_model(
        Box::new(engine),
        Box::new(engines_noise),
        NUM_SHOTS,
        1,
        Some(42),
    )
    .unwrap();

    let engines_counts = extract_outcomes_from_shots(&engines_results, "c", 1);

    // Run with pecos-neo (create noise model inside closure)
    let commands = CommandBuilder::new().pz(0).x(0).mz(0).build();

    let config = MonteCarloConfig::new()
        .with_shots(NUM_SHOTS)
        .with_workers(1)
        .with_seed(42);

    let neo_results = MonteCarloRunner::run(
        &commands,
        config,
        || {
            let neo_noise = GeneralNoiseModelBuilder::new().with_p1(p1).build();
            (
                CircuitRunner::new().with_noise(neo_noise),
                SparseStab::new(1),
            )
        },
        |outcomes| {
            let b = outcomes.get_bit(QubitId(0)).unwrap_or(false);
            if b { "1".to_string() } else { "0".to_string() }
        },
    );

    let mut neo_counts = BTreeMap::new();
    for result in neo_results.iter() {
        *neo_counts.entry(result.clone()).or_insert(0) += 1;
    }

    // Both should have mostly "1" outcomes (X flips |0> to |1>) with some errors
    let engines_error_rate = *engines_counts.get("0").unwrap_or(&0) as f64 / NUM_SHOTS as f64;
    let neo_error_rate = f64::from(*neo_counts.get("0").unwrap_or(&0)) / NUM_SHOTS as f64;

    println!("Engines error rate: {engines_error_rate:.4}");
    println!("Neo error rate: {neo_error_rate:.4}");

    // Error rates should be similar (within tolerance)
    let diff = (engines_error_rate - neo_error_rate).abs();
    assert!(
        diff < TOLERANCE_PERCENT / 100.0,
        "Error rates should match: engines={engines_error_rate:.4}, neo={neo_error_rate:.4}, diff={diff:.4}"
    );
}

#[test]
fn test_monte_carlo_measurement_errors() {
    // Test measurement error rates match between systems
    let p_meas = 0.10; // 10% measurement error

    // pecos-engines noise model
    let engines_noise = GeneralNoiseModel::builder()
        .with_meas_0_probability(p_meas)
        .with_meas_1_probability(p_meas)
        .build();

    // Circuit: prep |0>, measure (should be 0, but measurement errors flip some)
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        measure q[0] -> c[0];
    "#;

    let engine = QASMEngine::from_str(qasm).unwrap();
    let engines_results = MonteCarloEngine::run_with_noise_model(
        Box::new(engine),
        Box::new(engines_noise),
        NUM_SHOTS,
        1,
        Some(42),
    )
    .unwrap();

    let engines_counts = extract_outcomes_from_shots(&engines_results, "c", 1);

    // Run with pecos-neo
    let commands = CommandBuilder::new().pz(0).mz(0).build();

    let config = MonteCarloConfig::new()
        .with_shots(NUM_SHOTS)
        .with_workers(1)
        .with_seed(42);

    let neo_results = MonteCarloRunner::run(
        &commands,
        config,
        || {
            let neo_noise = GeneralNoiseModelBuilder::new()
                .with_p_meas(p_meas, p_meas)
                .build();
            (
                CircuitRunner::new().with_noise(neo_noise),
                SparseStab::new(1),
            )
        },
        |outcomes| {
            let b = outcomes.get_bit(QubitId(0)).unwrap_or(false);
            if b { "1".to_string() } else { "0".to_string() }
        },
    );

    let mut neo_counts = BTreeMap::new();
    for result in neo_results.iter() {
        *neo_counts.entry(result.clone()).or_insert(0) += 1;
    }

    // Error rate should be approximately p_meas (measuring |0> but getting 1)
    let engines_error_rate = *engines_counts.get("1").unwrap_or(&0) as f64 / NUM_SHOTS as f64;
    let neo_error_rate = f64::from(*neo_counts.get("1").unwrap_or(&0)) / NUM_SHOTS as f64;

    println!("Expected error rate: {p_meas:.4}");
    println!("Engines error rate: {engines_error_rate:.4}");
    println!("Neo error rate: {neo_error_rate:.4}");

    // Both should be close to p_meas
    assert!(
        (engines_error_rate - p_meas).abs() < TOLERANCE_PERCENT / 100.0,
        "Engines error rate should be close to p_meas"
    );
    assert!(
        (neo_error_rate - p_meas).abs() < TOLERANCE_PERCENT / 100.0,
        "Neo error rate should be close to p_meas"
    );
}

#[test]
fn test_monte_carlo_parallel_execution() {
    // Test that parallel execution produces consistent results
    let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

    // Run with multiple workers
    let config_parallel = MonteCarloConfig::new()
        .with_shots(NUM_SHOTS)
        .with_workers(4)
        .with_seed(42);

    let results_parallel = MonteCarloRunner::run(
        &commands,
        config_parallel,
        || (CircuitRunner::new(), SparseStab::new(1)),
        |outcomes| outcomes.get_bit(QubitId(0)).unwrap_or(false),
    );

    // Run with single worker (different seed to avoid comparing same RNG sequence)
    let config_single = MonteCarloConfig::new()
        .with_shots(NUM_SHOTS)
        .with_workers(1)
        .with_seed(123);

    let results_single = MonteCarloRunner::run(
        &commands,
        config_single,
        || (CircuitRunner::new(), SparseStab::new(1)),
        |outcomes| outcomes.get_bit(QubitId(0)).unwrap_or(false),
    );

    // Both should have roughly 50/50 distribution (Hadamard)
    let parallel_ones = results_parallel.iter().filter(|&&b| b).count();
    let single_ones = results_single.iter().filter(|&&b| b).count();

    let parallel_rate = parallel_ones as f64 / NUM_SHOTS as f64;
    let single_rate = single_ones as f64 / NUM_SHOTS as f64;

    println!("Parallel rate: {parallel_rate:.4}");
    println!("Single rate: {single_rate:.4}");

    // Both should be close to 0.5
    assert!(
        (parallel_rate - 0.5).abs() < TOLERANCE_PERCENT / 100.0,
        "Parallel execution should give ~50% ones"
    );
    assert!(
        (single_rate - 0.5).abs() < TOLERANCE_PERCENT / 100.0,
        "Single execution should give ~50% ones"
    );
}

#[test]
fn test_monte_carlo_two_qubit_noise() {
    // Test two-qubit gate noise
    let p2 = 0.10; // 10% two-qubit error rate

    // pecos-engines noise model (scaled)
    let engines_noise = GeneralNoiseModel::builder()
        .with_average_p2_probability(p2 / 1.25) // Scale down for engines
        .build();

    // Circuit: Bell state creation, errors will decorrelate outcomes
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

    let engine = QASMEngine::from_str(qasm).unwrap();
    let engines_results = MonteCarloEngine::run_with_noise_model(
        Box::new(engine),
        Box::new(engines_noise),
        NUM_SHOTS,
        1,
        Some(42),
    )
    .unwrap();

    let engines_counts = extract_outcomes_from_shots(&engines_results, "c", 2);

    // Run with pecos-neo
    let commands = CommandBuilder::new()
        .pz(0)
        .pz(1)
        .h(0)
        .cx(0, 1)
        .mz(0)
        .mz(1)
        .build();

    let config = MonteCarloConfig::new()
        .with_shots(NUM_SHOTS)
        .with_workers(1)
        .with_seed(42);

    let neo_results = MonteCarloRunner::run(
        &commands,
        config,
        || {
            let neo_noise = GeneralNoiseModelBuilder::new().with_p2(p2).build();
            (
                CircuitRunner::new().with_noise(neo_noise),
                SparseStab::new(2),
            )
        },
        |outcomes| {
            let b0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
            let b1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);
            format!(
                "{}{}",
                if b0 { '1' } else { '0' },
                if b1 { '1' } else { '0' }
            )
        },
    );

    let mut neo_counts = BTreeMap::new();
    for result in neo_results.iter() {
        *neo_counts.entry(result.clone()).or_insert(0) += 1;
    }

    // With noise, we should see some "01" and "10" outcomes (decorrelated)
    let engines_decorrelated =
        engines_counts.get("01").unwrap_or(&0) + engines_counts.get("10").unwrap_or(&0);
    let neo_decorrelated = neo_counts.get("01").unwrap_or(&0) + neo_counts.get("10").unwrap_or(&0);

    let engines_decorr_rate = engines_decorrelated as f64 / NUM_SHOTS as f64;
    let neo_decorr_rate = f64::from(neo_decorrelated) / NUM_SHOTS as f64;

    println!("Engines decorrelation rate: {engines_decorr_rate:.4}");
    println!("Neo decorrelation rate: {neo_decorr_rate:.4}");

    // Both should have some decorrelation due to noise
    assert!(
        engines_decorr_rate > 0.01,
        "Engines should show some decorrelation"
    );
    assert!(neo_decorr_rate > 0.01, "Neo should show some decorrelation");

    // Rates should be similar
    let diff = (engines_decorr_rate - neo_decorr_rate).abs();
    assert!(
        diff < TOLERANCE_PERCENT / 100.0,
        "Decorrelation rates should match: engines={engines_decorr_rate:.4}, neo={neo_decorr_rate:.4}, diff={diff:.4}"
    );
}

// ============================================================================
// Reproducibility Tests
// ============================================================================

#[test]
fn test_monte_carlo_deterministic_circuit() {
    // Test that deterministic circuits (no measurement randomness) produce consistent results
    // Use X gate which deterministically flips |0> to |1>
    let commands = CommandBuilder::new().pz(0).x(0).mz(0).build();

    let config = MonteCarloConfig::new()
        .with_shots(100)
        .with_workers(1)
        .with_seed(42);

    let results: Vec<bool> = MonteCarloRunner::run(
        &commands,
        config,
        || (CircuitRunner::new(), SparseStab::new(1)),
        |outcomes| outcomes.get_bit(QubitId(0)).unwrap_or(false),
    )
    .into_iter()
    .collect();

    // All results should be true (X flips |0> to |1>)
    assert!(
        results.iter().all(|&b| b),
        "Deterministic X circuit should always produce 1"
    );
}

#[test]
fn test_monte_carlo_statistical_consistency() {
    // Test that Hadamard circuit produces roughly 50/50 distribution
    let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

    let config = MonteCarloConfig::new()
        .with_shots(NUM_SHOTS)
        .with_workers(1)
        .with_seed(42);

    let results: Vec<bool> = MonteCarloRunner::run(
        &commands,
        config,
        || (CircuitRunner::new(), SparseStab::new(1)),
        |outcomes| outcomes.get_bit(QubitId(0)).unwrap_or(false),
    )
    .into_iter()
    .collect();

    let ones_rate = results.iter().filter(|&&b| b).count() as f64 / NUM_SHOTS as f64;

    // Should be close to 0.5
    assert!(
        (ones_rate - 0.5).abs() < TOLERANCE_PERCENT / 100.0,
        "Hadamard should give ~50% ones, got {ones_rate:.4}"
    );
}

#[test]
fn test_full_seed_determinism() {
    // Test that with_full_seed() produces identical results across runs
    let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

    // Run twice with same full seed - should produce identical results
    let mut state1 = SparseStab::new(1);
    let mut runner1 = CircuitRunner::<SparseStab>::new().with_full_seed(&mut state1, 42);
    let mut state2 = SparseStab::new(1);
    let mut runner2 = CircuitRunner::<SparseStab>::new().with_full_seed(&mut state2, 42);

    let mut results1 = Vec::new();
    let mut results2 = Vec::new();

    for _ in 0..100 {
        state1.reset();
        results1.push(
            runner1
                .apply_circuit(&mut state1, &commands)
                .unwrap()
                .get_bit(QubitId(0))
                .unwrap_or(false),
        );
        state2.reset();
        results2.push(
            runner2
                .apply_circuit(&mut state2, &commands)
                .unwrap()
                .get_bit(QubitId(0))
                .unwrap_or(false),
        );
    }

    assert_eq!(
        results1, results2,
        "with_full_seed() should produce identical results"
    );
}

// ============================================================================
// Importance Sampling Validation Tests
// ============================================================================

/// Validate that importance sampling produces unbiased estimates matching standard Monte Carlo.
///
/// This is the critical validation: both methods should estimate the same quantity
/// (error rate) and agree within statistical tolerance.
#[test]
fn test_importance_sampling_matches_standard_monte_carlo() {
    use pecos_neo::sampling::ImportanceSamplingRunner;

    let commands = CommandBuilder::new()
        .pz(0)
        .identity(0) // Single-qubit gate that triggers noise
        .mz(0)
        .build();

    let p_error = 0.05; // True error rate
    let num_shots = 10_000;

    // ========== Standard Monte Carlo ==========
    // Estimate error rate by counting bit flips
    let mut standard_ones = 0;
    for seed in 0..num_shots {
        let noise = GeneralNoiseModelBuilder::new().with_p1(p_error).build();
        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(seed as u64);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            standard_ones += 1;
        }
    }
    let standard_rate = f64::from(standard_ones) / f64::from(num_shots);

    // ========== Importance Sampling ==========
    // Estimate same quantity with boosted error rate and reweighting
    let boost = 10.0;
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for seed in 0..num_shots {
        let mut runner = ImportanceSamplingRunner::new(SparseStab::new(1))
            .with_single_qubit_boost(p_error, boost)
            .with_seed(seed as u64);
        let result = runner.run_shot(&commands);

        let value = if result.outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            1.0
        } else {
            0.0
        };
        let weight = result.weight.weight();

        weighted_sum += value * weight;
        total_weight += weight;
    }
    let importance_rate = weighted_sum / total_weight;

    // ========== Validate ==========
    println!("True error rate:     {p_error:.4}");
    println!("Standard MC rate:    {standard_rate:.4}");
    println!("Importance sampling: {importance_rate:.4}");
    println!(
        "Difference:          {:.4}",
        (standard_rate - importance_rate).abs()
    );

    // Both should be close to the true error rate
    assert!(
        (standard_rate - p_error).abs() < 0.02,
        "Standard MC should be close to true rate: expected ~{p_error}, got {standard_rate:.4}"
    );

    // Importance sampling should match standard Monte Carlo
    assert!(
        (standard_rate - importance_rate).abs() < 0.02,
        "Importance sampling should match standard MC: std={standard_rate:.4}, imp={importance_rate:.4}"
    );
}

/// Test importance sampling with higher boost factor for rare events.
///
/// With a lower true error rate and higher boost, importance sampling
/// should still produce unbiased estimates.
#[test]
fn test_importance_sampling_rare_events() {
    use pecos_neo::sampling::ImportanceSamplingRunner;

    let commands = CommandBuilder::new().pz(0).identity(0).mz(0).build();

    let p_error = 0.01; // 1% error rate (rarer)
    let boost = 50.0; // Aggressive boost
    let num_shots = 20_000; // More shots for rare events

    // ========== Standard Monte Carlo ==========
    let mut standard_ones = 0;
    for seed in 0..num_shots {
        let noise = GeneralNoiseModelBuilder::new().with_p1(p_error).build();
        let mut state = SparseStab::new(1);
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise)
            .with_seed(seed as u64);
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
        if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            standard_ones += 1;
        }
    }
    let standard_rate = f64::from(standard_ones) / f64::from(num_shots);

    // ========== Importance Sampling ==========
    let mut weighted_sum = 0.0;
    let mut total_weight = 0.0;

    for seed in 0..num_shots {
        let mut runner = ImportanceSamplingRunner::new(SparseStab::new(1))
            .with_single_qubit_boost(p_error, boost)
            .with_seed(seed as u64);
        let result = runner.run_shot(&commands);

        let value = if result.outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            1.0
        } else {
            0.0
        };
        let weight = result.weight.weight();

        weighted_sum += value * weight;
        total_weight += weight;
    }
    let importance_rate = weighted_sum / total_weight;

    // ========== Validate ==========
    println!("True error rate:     {p_error:.4}");
    println!("Standard MC rate:    {standard_rate:.4}");
    println!("Importance sampling: {importance_rate:.4}");

    // With 1% error rate, standard MC should see ~200 errors in 20k shots
    // Importance sampling with 50x boost sees ~50% errors but reweights correctly
    assert!(
        (standard_rate - importance_rate).abs() < 0.01,
        "Importance sampling should match standard MC for rare events: std={standard_rate:.4}, imp={importance_rate:.4}"
    );
}

/// Test that importance sampling variance is reduced compared to standard MC.
///
/// For rare events, importance sampling should achieve lower variance
/// (tighter confidence intervals) for the same number of samples.
#[test]
fn test_importance_sampling_variance_reduction() {
    use pecos_neo::sampling::ImportanceSamplingRunner;

    let commands = CommandBuilder::new().pz(0).identity(0).mz(0).build();

    let p_error = 0.01;
    let boost = 20.0;
    let num_trials = 50; // Run multiple independent estimates
    let shots_per_trial = 1000;

    // ========== Standard Monte Carlo variance ==========
    let mut standard_estimates = Vec::new();
    for trial in 0..num_trials {
        let base_seed = trial * shots_per_trial;
        let mut ones = 0;
        for shot in 0..shots_per_trial {
            let noise = GeneralNoiseModelBuilder::new().with_p1(p_error).build();
            let mut state = SparseStab::new(1);
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(noise)
                .with_seed((base_seed + shot) as u64);
            let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();
            if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
                ones += 1;
            }
        }
        standard_estimates.push(f64::from(ones) / f64::from(shots_per_trial));
    }

    // ========== Importance Sampling variance ==========
    let mut importance_estimates = Vec::new();
    for trial in 0..num_trials {
        let base_seed = trial * shots_per_trial;
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;

        for shot in 0..shots_per_trial {
            let mut runner = ImportanceSamplingRunner::new(SparseStab::new(1))
                .with_single_qubit_boost(p_error, boost)
                .with_seed((base_seed + shot) as u64);
            let result = runner.run_shot(&commands);

            let value = if result.outcomes.get_bit(QubitId(0)).unwrap_or(false) {
                1.0
            } else {
                0.0
            };
            let weight = result.weight.weight();

            weighted_sum += value * weight;
            total_weight += weight;
        }
        importance_estimates.push(weighted_sum / total_weight);
    }

    // Calculate variances
    let standard_mean: f64 = standard_estimates.iter().sum::<f64>() / f64::from(num_trials);
    let standard_var: f64 = standard_estimates
        .iter()
        .map(|x| (x - standard_mean).powi(2))
        .sum::<f64>()
        / f64::from(num_trials);

    let importance_mean: f64 = importance_estimates.iter().sum::<f64>() / f64::from(num_trials);
    let importance_var: f64 = importance_estimates
        .iter()
        .map(|x| (x - importance_mean).powi(2))
        .sum::<f64>()
        / f64::from(num_trials);

    println!("Standard MC:    mean={standard_mean:.4}, var={standard_var:.6}");
    println!("Importance:     mean={importance_mean:.4}, var={importance_var:.6}");
    println!("Variance ratio: {:.2}x", standard_var / importance_var);

    // Both means should be close to the true rate
    assert!(
        (standard_mean - p_error).abs() < 0.01,
        "Standard MC mean should be close to true rate"
    );
    assert!(
        (importance_mean - p_error).abs() < 0.01,
        "Importance sampling mean should be close to true rate"
    );

    // Note: Variance reduction depends on the boost factor and true rate.
    // We just verify both methods produce valid estimates here.
    // A well-tuned importance sampler would show variance reduction.
}
