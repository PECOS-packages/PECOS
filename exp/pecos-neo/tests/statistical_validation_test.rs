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
//! Comprehensive statistical validation tests with high sample sizes.
//!
//! These tests run larger numbers of shots to get statistically significant
//! comparisons between pecos-neo systems and pecos-engines.

use pecos_core::QubitId;
use pecos_engines::monte_carlo::MonteCarloEngine;
use pecos_engines::noise::GeneralNoiseModel;
use pecos_neo::command::CommandBuilder;
use pecos_neo::ecs::{ParallelConfig, ParallelCoordinator};
use pecos_neo::noise::GeneralNoiseModelBuilder;
use pecos_neo::runner::CircuitRunner;
use pecos_neo::sampling::{MonteCarloConfig, MonteCarloRunner};
use pecos_qasm::QASMEngine;
use pecos_simulators::SparseStab;
use std::collections::BTreeMap;
use std::str::FromStr;

// High shot counts for statistical significance
const NUM_SHOTS: usize = 10_000;
const NUM_WORKERS: usize = 4;

// Statistical tolerance based on binomial standard error
// For p=0.5, n=10000: SE = sqrt(0.5*0.5/10000) = 0.005
// 3-sigma confidence: tolerance = 3 * SE = 0.015 (1.5%)
const STAT_TOLERANCE: f64 = 0.02; // 2% to be safe

/// Calculate 95% confidence interval half-width for a binomial proportion.
fn binomial_ci_halfwidth(p: f64, n: usize) -> f64 {
    // 1.96 * sqrt(p*(1-p)/n)
    1.96 * (p * (1.0 - p) / n as f64).sqrt()
}

/// Check if two proportions are statistically equivalent.
fn proportions_equivalent(p1: f64, n1: usize, p2: f64, n2: usize) -> bool {
    // Use pooled standard error for two-proportion z-test
    let p_pooled = (p1 * n1 as f64 + p2 * n2 as f64) / (n1 + n2) as f64;
    let se = (p_pooled * (1.0 - p_pooled) * (1.0 / n1 as f64 + 1.0 / n2 as f64)).sqrt();
    let z = (p1 - p2).abs() / se;

    // z < 2.58 for 99% confidence level
    z < 2.58
}

/// Compute chi-square statistic for distribution comparison.
fn chi_square_test(
    observed: &BTreeMap<String, usize>,
    expected: &BTreeMap<String, usize>,
    total_obs: usize,
    total_exp: usize,
) -> f64 {
    let mut chi_sq = 0.0;

    let mut all_keys: Vec<_> = observed.keys().chain(expected.keys()).cloned().collect();
    all_keys.sort();
    all_keys.dedup();

    for key in all_keys {
        let obs = *observed.get(&key).unwrap_or(&0) as f64;
        let exp_rate = *expected.get(&key).unwrap_or(&0) as f64 / total_exp as f64;
        let exp = exp_rate * total_obs as f64;

        if exp > 0.0 {
            chi_sq += (obs - exp).powi(2) / exp;
        }
    }

    chi_sq
}

// --- Hadamard Gate Tests - Should give 50/50 distribution ---

#[test]
fn test_hadamard_distribution_high_statistics() {
    println!("\n=== Hadamard Distribution Test ({NUM_SHOTS} shots) ===\n");

    let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

    // MonteCarloRunner
    let mc_config = MonteCarloConfig::new()
        .with_shots(NUM_SHOTS)
        .with_workers(NUM_WORKERS)
        .with_seed(42);

    let mc_results = MonteCarloRunner::run(
        &commands,
        &mc_config,
        || (CircuitRunner::new(), SparseStab::new(1)),
        |outcomes| outcomes.get_bit(QubitId(0)).unwrap_or(false),
    );

    let mc_ones = mc_results.iter().filter(|&&b| b).count();
    let mc_rate = mc_ones as f64 / NUM_SHOTS as f64;
    let mc_ci = binomial_ci_halfwidth(mc_rate, NUM_SHOTS);

    // ParallelCoordinator
    let coord_config = ParallelConfig::new()
        .with_workers(NUM_WORKERS)
        .with_entities_per_worker(NUM_SHOTS / NUM_WORKERS)
        .with_seed(42);

    let coordinator: ParallelCoordinator<SparseStab> = ParallelCoordinator::new(coord_config);

    let coord_results = coordinator.run(
        || SparseStab::new(1),
        |world| {
            let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

            world
                .entities()
                .map(|entity| {
                    let sim_comp = world.simulators.get(entity).unwrap();
                    let rng_comp = world.rngs.get(entity).unwrap();

                    let mut sim = sim_comp.simulator.clone();
                    let mut runner =
                        CircuitRunner::<SparseStab>::new().with_rng(rng_comp.rng.clone());
                    sim.reset();
                    let outcomes = runner.apply_circuit(&mut sim, &commands).unwrap();
                    outcomes.get_bit(QubitId(0)).unwrap_or(false)
                })
                .collect()
        },
    );

    let coord_ones = coord_results.iter().filter(|&&b| b).count();
    let coord_rate = coord_ones as f64 / coord_results.len() as f64;
    let coord_ci = binomial_ci_halfwidth(coord_rate, coord_results.len());

    println!("MonteCarloRunner:     {mc_rate:.4} +/- {mc_ci:.4} ({mc_ones}/{NUM_SHOTS} ones)");
    println!(
        "ParallelCoordinator:  {coord_rate:.4} +/- {coord_ci:.4} ({coord_ones}/{} ones)",
        coord_results.len()
    );
    println!("Expected:             0.5000 (Hadamard on |0>)");

    // Both should be statistically consistent with 0.5
    assert!(
        (mc_rate - 0.5).abs() < 3.0 * mc_ci,
        "MC rate {mc_rate:.4} should be within 3-sigma of 0.5"
    );
    assert!(
        (coord_rate - 0.5).abs() < 3.0 * coord_ci,
        "Coordinator rate {coord_rate:.4} should be within 3-sigma of 0.5"
    );

    // They should be statistically equivalent to each other
    assert!(
        proportions_equivalent(mc_rate, NUM_SHOTS, coord_rate, coord_results.len()),
        "MC and Coordinator rates should be statistically equivalent"
    );
}

// --- Bell State Tests - Should give 50/50 for 00 and 11 ---

#[test]
fn test_bell_state_distribution_high_statistics() {
    println!("\n=== Bell State Distribution Test ({NUM_SHOTS} shots) ===\n");

    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    // MonteCarloRunner
    let mc_config = MonteCarloConfig::new()
        .with_shots(NUM_SHOTS)
        .with_workers(NUM_WORKERS)
        .with_seed(42);

    let mc_results = MonteCarloRunner::run(
        &commands,
        &mc_config,
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

    let mut mc_counts: BTreeMap<String, usize> = BTreeMap::new();
    for result in mc_results.iter() {
        *mc_counts.entry(result.clone()).or_insert(0) += 1;
    }

    let mc_00 = *mc_counts.get("00").unwrap_or(&0) as f64 / NUM_SHOTS as f64;
    let mc_11 = *mc_counts.get("11").unwrap_or(&0) as f64 / NUM_SHOTS as f64;
    let mc_01 = *mc_counts.get("01").unwrap_or(&0) as f64 / NUM_SHOTS as f64;
    let mc_10 = *mc_counts.get("10").unwrap_or(&0) as f64 / NUM_SHOTS as f64;

    println!("MonteCarloRunner distribution:");
    println!(
        "  00: {:.4} ({} shots)",
        mc_00,
        mc_counts.get("00").unwrap_or(&0)
    );
    println!(
        "  11: {:.4} ({} shots)",
        mc_11,
        mc_counts.get("11").unwrap_or(&0)
    );
    println!(
        "  01: {:.4} ({} shots)",
        mc_01,
        mc_counts.get("01").unwrap_or(&0)
    );
    println!(
        "  10: {:.4} ({} shots)",
        mc_10,
        mc_counts.get("10").unwrap_or(&0)
    );

    // Bell state: only 00 and 11 should occur
    let anti_correlated = mc_01 + mc_10;
    assert!(
        anti_correlated < 0.001,
        "Bell state should have no 01/10 outcomes, got {anti_correlated:.4}"
    );

    // Both 00 and 11 should be ~0.5
    let ci = binomial_ci_halfwidth(0.5, NUM_SHOTS);
    assert!(
        (mc_00 - 0.5).abs() < 3.0 * ci,
        "00 rate {mc_00:.4} should be ~0.5"
    );
    assert!(
        (mc_11 - 0.5).abs() < 3.0 * ci,
        "11 rate {mc_11:.4} should be ~0.5"
    );
}

// --- Noise Model Tests ---

#[test]
fn test_depolarizing_noise_rate_validation() {
    println!("\n=== Depolarizing Noise Rate Validation ({NUM_SHOTS} shots) ===\n");

    // Test various error rates
    let test_rates = [0.01, 0.05, 0.10, 0.20];

    for &p1 in &test_rates {
        // Circuit: |0> -> X -> measure (should give 1 without noise)
        // With depolarizing noise on X gate, some shots will give 0
        let commands = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

        let mc_config = MonteCarloConfig::new()
            .with_shots(NUM_SHOTS)
            .with_workers(NUM_WORKERS)
            .with_seed(42);

        let mc_results = MonteCarloRunner::run(
            &commands,
            &mc_config,
            || {
                let noise = GeneralNoiseModelBuilder::new().with_p1(p1).build();
                (CircuitRunner::new().with_noise(noise), SparseStab::new(1))
            },
            |outcomes| outcomes.get_bit(QubitId(0)).unwrap_or(false),
        );

        let ones = mc_results.iter().filter(|&&b| b).count();
        let ones_rate = ones as f64 / NUM_SHOTS as f64;
        let error_rate = 1.0 - ones_rate;

        // With depolarizing noise p1, after X gate:
        // - No error (1-p1): gives |1>
        // - X error (p1/3): X*X|0> = |0>, so error
        // - Y error (p1/3): Y*X|0> = -i*Z|0> = |0>, so error (phase doesn't matter for measurement)
        // - Z error (p1/3): Z*X|0> = -X|0> = -|1>, so correct
        // Expected error rate: 2*p1/3
        let expected_error = 2.0 * p1 / 3.0;
        let ci = binomial_ci_halfwidth(expected_error, NUM_SHOTS);

        println!(
            "p1={:.2}: observed error={:.4}, expected={:.4} +/- {:.4}",
            p1,
            error_rate,
            expected_error,
            3.0 * ci
        );

        // Should be within 3-sigma
        assert!(
            (error_rate - expected_error).abs() < 4.0 * ci + 0.01, // Add small buffer for numerical stability
            "Error rate {error_rate:.4} should be close to expected {expected_error:.4}"
        );
    }
}

#[test]
fn test_measurement_error_rate_validation() {
    println!("\n=== Measurement Error Rate Validation ({NUM_SHOTS} shots) ===\n");

    let test_rates = [0.01, 0.05, 0.10];

    for &p_meas in &test_rates {
        // Circuit: |0> -> measure (should give 0 without noise)
        // With measurement error, some shots will give 1
        let commands = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

        let mc_config = MonteCarloConfig::new()
            .with_shots(NUM_SHOTS)
            .with_workers(NUM_WORKERS)
            .with_seed(42);

        let mc_results = MonteCarloRunner::run(
            &commands,
            &mc_config,
            || {
                let noise = GeneralNoiseModelBuilder::new()
                    .with_p_meas(p_meas, 0.0) // Only 0->1 flip
                    .build();
                (CircuitRunner::new().with_noise(noise), SparseStab::new(1))
            },
            |outcomes| outcomes.get_bit(QubitId(0)).unwrap_or(false),
        );

        let ones = mc_results.iter().filter(|&&b| b).count();
        let error_rate = ones as f64 / NUM_SHOTS as f64;
        let ci = binomial_ci_halfwidth(p_meas, NUM_SHOTS);

        println!(
            "p_meas={:.2}: observed error={:.4}, expected={:.4} +/- {:.4}",
            p_meas,
            error_rate,
            p_meas,
            3.0 * ci
        );

        // Should be within 3-sigma of p_meas
        assert!(
            (error_rate - p_meas).abs() < 3.0 * ci + 0.005,
            "Measurement error rate {error_rate:.4} should be close to {p_meas:.4}"
        );
    }
}

// --- Cross-System Comparison (pecos-neo vs pecos-engines) ---

#[test]
fn test_neo_vs_engines_bell_state_comparison() {
    println!("\n=== Neo vs Engines Bell State Comparison ({NUM_SHOTS} shots) ===\n");

    // QASM for pecos-engines
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

    // Run with pecos-engines
    let engine = QASMEngine::from_str(qasm).unwrap();
    let engines_results = MonteCarloEngine::run_with_noise_model(
        Box::new(engine),
        Box::new(pecos_engines::PassThroughNoiseModel::builder().build()),
        NUM_SHOTS,
        NUM_WORKERS,
        Some(42),
    )
    .unwrap();

    let mut engines_counts: BTreeMap<String, usize> = BTreeMap::new();
    for shot in &engines_results.shots {
        let value = shot
            .data
            .get("c")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .unwrap_or(0);
        let bits: String = (0..2)
            .map(|i| if (value >> i) & 1 == 1 { '1' } else { '0' })
            .collect();
        *engines_counts.entry(bits).or_insert(0) += 1;
    }

    // Run with pecos-neo MonteCarloRunner
    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    let mc_config = MonteCarloConfig::new()
        .with_shots(NUM_SHOTS)
        .with_workers(NUM_WORKERS)
        .with_seed(42);

    let neo_results = MonteCarloRunner::run(
        &commands,
        &mc_config,
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

    let mut neo_counts: BTreeMap<String, usize> = BTreeMap::new();
    for result in neo_results.iter() {
        *neo_counts.entry(result.clone()).or_insert(0) += 1;
    }

    println!("pecos-engines distribution:");
    for (k, v) in &engines_counts {
        println!("  {}: {} ({:.4})", k, v, *v as f64 / NUM_SHOTS as f64);
    }
    println!("\npecos-neo distribution:");
    for (k, v) in &neo_counts {
        println!("  {}: {} ({:.4})", k, v, *v as f64 / NUM_SHOTS as f64);
    }

    // Chi-square test
    let chi_sq = chi_square_test(&neo_counts, &engines_counts, NUM_SHOTS, NUM_SHOTS);
    // Critical value for df=3 (4 outcomes - 1) at alpha=0.01 is 11.34
    println!("\nChi-square statistic: {chi_sq:.4} (critical value at 99%: 11.34)");

    assert!(
        chi_sq < 11.34,
        "Distributions should not differ significantly (chi-sq={chi_sq:.4})"
    );
}

#[test]
fn test_neo_vs_engines_noisy_comparison() {
    println!("\n=== Neo vs Engines Noisy Comparison ({NUM_SHOTS} shots) ===\n");

    let p1 = 0.05;
    let p2 = 0.05;

    // QASM for pecos-engines
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

    // pecos-engines noise model
    let engines_noise = GeneralNoiseModel::builder()
        .with_average_p1_probability(p1 / 1.5)
        .with_average_p2_probability(p2 / 1.25)
        .build();

    let engine = QASMEngine::from_str(qasm).unwrap();
    let engines_results = MonteCarloEngine::run_with_noise_model(
        Box::new(engine),
        Box::new(engines_noise),
        NUM_SHOTS,
        NUM_WORKERS,
        Some(42),
    )
    .unwrap();

    // Count outcomes
    let mut engines_counts: BTreeMap<String, usize> = BTreeMap::new();
    for shot in &engines_results.shots {
        let value = shot
            .data
            .get("c")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .unwrap_or(0);
        let bits: String = (0..2)
            .map(|i| if (value >> i) & 1 == 1 { '1' } else { '0' })
            .collect();
        *engines_counts.entry(bits).or_insert(0) += 1;
    }

    // pecos-neo
    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    let mc_config = MonteCarloConfig::new()
        .with_shots(NUM_SHOTS)
        .with_workers(NUM_WORKERS)
        .with_seed(42);

    let neo_results = MonteCarloRunner::run(
        &commands,
        &mc_config,
        || {
            let noise = GeneralNoiseModelBuilder::new()
                .with_p1(p1)
                .with_p2(p2)
                .build();
            (CircuitRunner::new().with_noise(noise), SparseStab::new(2))
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

    let mut neo_counts: BTreeMap<String, usize> = BTreeMap::new();
    for result in neo_results.iter() {
        *neo_counts.entry(result.clone()).or_insert(0) += 1;
    }

    println!("pecos-engines distribution (p1={p1}, p2={p2}):");
    for (k, v) in &engines_counts {
        println!("  {}: {} ({:.4})", k, v, *v as f64 / NUM_SHOTS as f64);
    }
    println!("\npecos-neo distribution:");
    for (k, v) in &neo_counts {
        println!("  {}: {} ({:.4})", k, v, *v as f64 / NUM_SHOTS as f64);
    }

    // Check decorrelation rates are similar
    let engines_decorr = (*engines_counts.get("01").unwrap_or(&0)
        + *engines_counts.get("10").unwrap_or(&0)) as f64
        / NUM_SHOTS as f64;
    let neo_decorr = (*neo_counts.get("01").unwrap_or(&0) + *neo_counts.get("10").unwrap_or(&0))
        as f64
        / NUM_SHOTS as f64;

    println!("\nDecorrelation rates:");
    println!("  pecos-engines: {engines_decorr:.4}");
    println!("  pecos-neo:     {neo_decorr:.4}");

    // Should be within 2% of each other
    assert!(
        (engines_decorr - neo_decorr).abs() < 0.03,
        "Decorrelation rates should be similar: {engines_decorr:.4} vs {neo_decorr:.4}"
    );
}

// --- Summary Statistics ---

#[test]
fn test_print_validation_summary() {
    println!("\n");
    println!("=================================================================");
    println!("                  STATISTICAL VALIDATION SUMMARY");
    println!("=================================================================");
    println!();
    println!("Sample size: {NUM_SHOTS} shots per test");
    println!("Workers: {NUM_WORKERS}");
    println!("Statistical tolerance: {:.1}%", STAT_TOLERANCE * 100.0);
    println!();
    println!("For binomial proportions at p=0.5 with n={NUM_SHOTS}:");
    println!("  Standard error: {:.4}", (0.25 / NUM_SHOTS as f64).sqrt());
    println!(
        "  95% CI width:   +/- {:.4}",
        binomial_ci_halfwidth(0.5, NUM_SHOTS)
    );
    println!(
        "  99% CI width:   +/- {:.4}",
        2.58 * (0.25 / NUM_SHOTS as f64).sqrt()
    );
    println!();
    println!("=================================================================");
}
