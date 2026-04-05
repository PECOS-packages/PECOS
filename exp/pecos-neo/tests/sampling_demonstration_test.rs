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

//! Tests for sampling techniques: path exploration, importance sampling,
//! outcome biasing, classical control flow integration, and subset simulation.
#![allow(clippy::float_cmp)]

use pecos_core::QubitId;
use pecos_neo::command::CommandBuilder;
use pecos_neo::noise::{ComposableNoiseModel, SingleQubitChannel};
use pecos_neo::outcome::MeasurementOutcomes;
use pecos_neo::program::{CommandSource, ConditionalProgram, ProgramRunner};
use pecos_neo::runner::CircuitRunner;
use pecos_neo::sampling::path::{EnumeratedPath, PathEnumerator, PathExplorer, PathStatistics};
use pecos_neo::sampling::weight::WeightedStatistics;
use pecos_neo::sampling::{
    BernoulliSubsetSimulation, ImportanceSamplingRunner, OutcomeBiasConfig, ProperSubsetSimulation,
    SubsetConfig,
};
use pecos_simulators::SparseStab;

// --- Part 1: Path Exploration on Static Circuits ---

/// Demonstrate path recording - running a circuit and recording which path was taken.
#[test]
fn demo_path_recording() {
    // A simple circuit with non-deterministic measurement
    let circuit = CommandBuilder::new()
        .pz(&[0])
        .h(&[0]) // Creates |+> superposition
        .mz(&[0])
        .build();

    let mut explorer = PathExplorer::new(SparseStab::new(1)).with_seed(42);

    // Run and record the path
    let result = explorer.run_and_record(&circuit);

    println!("Path Recording Demo:");
    println!("  Outcome: {:?}", result.outcomes.get_bit(QubitId(0)));
    println!("  Path length: {}", result.path.len());
    println!(
        "  Was deterministic: {:?}",
        result.path.get(0).map(|p| p.is_deterministic)
    );
    println!("  Path probability: {}", result.path.probability_f64());

    // H|0> creates superposition, so measurement is NOT deterministic
    assert!(!result.path.get(0).unwrap().is_deterministic);
    assert_eq!(result.path.probability_f64(), 0.5); // 50% for either outcome
}

/// Demonstrate path replay - forcing specific measurement outcomes.
#[test]
fn demo_path_replay() {
    let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

    let mut explorer = PathExplorer::new(SparseStab::new(1));

    println!("\nPath Replay Demo:");

    // Force outcome 0
    let path_zero = EnumeratedPath::new(0, 1);
    let result0 = explorer.run_with_path(&circuit, &path_zero);
    println!(
        "  Forced path '0': got outcome {:?}",
        result0.outcomes.get_bit(QubitId(0))
    );
    assert_eq!(result0.outcomes.get_bit(QubitId(0)), Some(false));

    // Force outcome 1
    let path_one = EnumeratedPath::new(1, 1);
    let result1 = explorer.run_with_path(&circuit, &path_one);
    println!(
        "  Forced path '1': got outcome {:?}",
        result1.outcomes.get_bit(QubitId(0))
    );
    assert_eq!(result1.outcomes.get_bit(QubitId(0)), Some(true));
}

/// Demonstrate systematic path enumeration with weighted statistics.
#[test]
fn demo_path_enumeration() {
    // A circuit with 2 non-deterministic measurements
    let circuit = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0])
        .h(&[1])
        .mz(&[0])
        .mz(&[1])
        .build();

    let mut explorer = PathExplorer::new(SparseStab::new(2));
    let mut stats = PathStatistics::new();

    println!("\nPath Enumeration Demo:");
    println!("  Enumerating all 2^2 = 4 paths:");

    for path in PathEnumerator::new(2) {
        let result = explorer.run_with_path(&circuit, &path);
        let o0 = result.outcomes.get_bit(QubitId(0)).unwrap_or(false);
        let o1 = result.outcomes.get_bit(QubitId(1)).unwrap_or(false);

        // Example predicate: count paths where both qubits measure 1
        let both_one = if o0 && o1 { 1.0 } else { 0.0 };
        stats.add(both_one, path.probability());

        println!(
            "    Path '{}': outcomes ({}, {}), prob={}",
            path.to_binary_string(),
            u8::from(o0),
            u8::from(o1),
            path.probability()
        );
    }

    println!("  P(both=1) = {} (expected 0.25)", stats.mean());
    println!("  Total weight = {} (should be 1.0)", stats.total_weight());

    assert!((stats.mean() - 0.25).abs() < 1e-10);
    assert!((stats.total_weight() - 1.0).abs() < 1e-10);
}

/// Demonstrate Bell state path exploration - showing correlated outcomes.
#[test]
fn demo_bell_state_paths() {
    let bell_circuit = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0])
        .cx(&[(0, 1)]) // Entangle
        .mz(&[0])
        .mz(&[1])
        .build();

    let mut explorer = PathExplorer::new(SparseStab::new(2));

    println!("\nBell State Path Demo:");
    println!("  In a Bell state, q1's measurement is DETERMINISTIC after q0 is measured.");

    // Only need to enumerate 1 non-deterministic measurement
    // The second measurement is deterministically correlated
    for path in PathEnumerator::new(1) {
        let result = explorer.run_with_path(&bell_circuit, &path);
        let o0 = result.outcomes.get_bit(QubitId(0)).unwrap();
        let o1 = result.outcomes.get_bit(QubitId(1)).unwrap();

        // Check path details
        let det0 = result.path.get(0).is_none_or(|p| p.is_deterministic);
        let det1 = result.path.get(1).is_none_or(|p| p.is_deterministic);

        println!(
            "    Path '{}': q0={} (det={}), q1={} (det={})",
            path.to_binary_string(),
            u8::from(o0),
            det0,
            u8::from(o1),
            det1
        );

        assert_eq!(o0, o1, "Bell state outcomes must be correlated");
        assert!(!det0, "First measurement should be non-deterministic");
        assert!(
            det1,
            "Second measurement should be deterministic (correlated)"
        );
    }
}

// --- Part 2: Importance Sampling for Rare Events (Error Rate Boosting) ---

/// Demonstrate importance sampling with error rate boosting.
#[test]
fn demo_importance_sampling_boosted_errors() {
    // Circuit that might see an error
    let circuit = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

    // True error rate is low (0.1%)
    let p_true: f64 = 0.001;
    // We boost it 50x for sampling efficiency
    let boost: f64 = 50.0;
    let p_proposal = (p_true * boost).min(0.5);

    println!("\nImportance Sampling Demo (Error Boosting):");
    println!("  True error rate: {p_true}");
    println!("  Proposal rate: {p_proposal}");
    println!("  Boost factor: {boost}x");

    // Run many shots with boosted error rate
    let mut runner = ImportanceSamplingRunner::new(SparseStab::new(1))
        .with_single_qubit_boost(p_true, boost)
        .with_seed(42);

    let noise =
        ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(p_proposal));
    runner = runner.with_noise(noise);

    let num_shots = 1000;
    let mut stats = WeightedStatistics::new();

    for _ in 0..num_shots {
        let shot = runner.run_shot(&circuit);
        // Count if error occurred (measuring 1 when we prepped 0)
        let error = shot.outcomes.get_bit(QubitId(0)).unwrap_or(false);
        stats.add(if error { 1.0 } else { 0.0 }, &shot.weight);
    }

    println!("  Estimated error rate: {:.6}", stats.mean());
    println!(
        "  Effective sample size: {:.1}",
        stats.effective_sample_size()
    );

    // The weighted estimate should be close to the true error rate
    // (within statistical uncertainty)
}

/// Compare importance sampling vs standard Monte Carlo efficiency.
#[test]
#[allow(clippy::cast_sign_loss)] // loop counters (trial) are non-negative, cast to u64
fn demo_variance_comparison() {
    let circuit = CommandBuilder::new().pz(&[0]).mz(&[0]).build();

    let p_true: f64 = 0.01;
    let boost: f64 = 20.0;

    println!("\nVariance Comparison Demo:");
    println!("  Comparing standard MC vs importance sampling for p={p_true}");

    // Run both methods multiple times to compare variance
    let num_trials = 10;
    let shots_per_trial = 1000;

    let mut mc_estimates = Vec::new();
    let mut is_estimates = Vec::new();

    for trial in 0..num_trials {
        // Standard Monte Carlo (true error rate)
        let mut mc_state = SparseStab::new(1);
        let mut mc_runner = CircuitRunner::<SparseStab>::new()
            .with_noise(
                ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(p_true)),
            )
            .with_seed(trial as u64);

        let mut mc_count = 0;
        for _ in 0..shots_per_trial {
            mc_state.reset();
            let outcomes = mc_runner.apply_circuit(&mut mc_state, &circuit).unwrap();
            if outcomes.get_bit(QubitId(0)).unwrap_or(false) {
                mc_count += 1;
            }
        }
        mc_estimates.push(f64::from(mc_count) / f64::from(shots_per_trial));

        // Importance sampling (boosted)
        let p_proposal = (p_true * boost).min(0.5_f64);
        let mut is_runner = ImportanceSamplingRunner::new(SparseStab::new(1))
            .with_single_qubit_boost(p_true, boost)
            .with_noise(
                ComposableNoiseModel::new()
                    .add_channel(SingleQubitChannel::depolarizing(p_proposal)),
            )
            .with_seed(1000 + trial as u64);

        let mut is_stats = WeightedStatistics::new();
        for _ in 0..shots_per_trial {
            let shot = is_runner.run_shot(&circuit);
            let error = shot.outcomes.get_bit(QubitId(0)).unwrap_or(false);
            is_stats.add(if error { 1.0 } else { 0.0 }, &shot.weight);
        }
        is_estimates.push(is_stats.mean());
    }

    // Compute means and standard deviations
    let mc_mean: f64 = mc_estimates.iter().sum::<f64>() / f64::from(num_trials);
    let is_mean: f64 = is_estimates.iter().sum::<f64>() / f64::from(num_trials);

    let mc_var: f64 = mc_estimates
        .iter()
        .map(|x| (x - mc_mean).powi(2))
        .sum::<f64>()
        / f64::from(num_trials);
    let is_var: f64 = is_estimates
        .iter()
        .map(|x| (x - is_mean).powi(2))
        .sum::<f64>()
        / f64::from(num_trials);

    println!(
        "  Standard MC: mean={:.6}, std={:.6}",
        mc_mean,
        mc_var.sqrt()
    );
    println!(
        "  Importance Sampling: mean={:.6}, std={:.6}",
        is_mean,
        is_var.sqrt()
    );
    println!("  True value: {p_true}");

    // Both should estimate the true value
    assert!((mc_mean - p_true).abs() < 0.02);
    assert!((is_mean - p_true).abs() < 0.02);
}

// --- Part 3: Measurement Outcome Biasing for Branch Exploration ---

/// Demonstrate measurement outcome biasing for exploring rare branches.
#[test]
fn demo_outcome_biasing() {
    // Circuit that branches based on measurement
    let prep_and_measure = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

    println!("\nOutcome Biasing Demo:");
    println!("  Biasing measurements to explore rare branches.");

    // Without biasing: 50/50 outcomes
    let mut unbiased_runner = ImportanceSamplingRunner::new(SparseStab::new(1)).with_seed(42);

    let mut count_one = 0;
    for _ in 0..1000 {
        let shot = unbiased_runner.run_shot(&prep_and_measure);
        if shot.outcomes.get_bit(QubitId(0)).unwrap_or(false) {
            count_one += 1;
        }
    }
    println!("  Unbiased: {}% measured '1'", f64::from(count_one) / 10.0);

    // With biasing toward '1': more '1' outcomes (but with weights)
    let bias_config = OutcomeBiasConfig::bias_toward_one(0.9);
    let mut biased_runner = ImportanceSamplingRunner::new(SparseStab::new(1))
        .with_outcome_bias(bias_config)
        .with_seed(42);

    let mut biased_count_one = 0;
    let mut weighted_stats = WeightedStatistics::new();

    for _ in 0..1000 {
        let shot = biased_runner.run_shot_biased(&prep_and_measure);
        let is_one = shot.outcomes.get_bit(QubitId(0)).unwrap_or(false);
        if is_one {
            biased_count_one += 1;
        }
        // The weight corrects for the bias
        weighted_stats.add(if is_one { 1.0 } else { 0.0 }, &shot.weight);
    }

    println!(
        "  Biased (p=0.9): {}% measured '1' (raw)",
        f64::from(biased_count_one) / 10.0
    );
    println!(
        "  Weighted estimate: {:.4} (should be ~0.50)",
        weighted_stats.mean()
    );

    // Raw count is biased high, but weighted estimate is correct
    assert!(biased_count_one > 700); // Should be ~90%
    assert!((weighted_stats.mean() - 0.5).abs() < 0.1); // But weighted mean ~0.5
}

// --- Part 4: Programs with Classical Control Flow (CommandSource) ---

/// Demonstrate running a conditional program with feedback.
#[test]
fn demo_conditional_program() {
    // Initial circuit: prepare and measure
    let initial = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

    // Branch function: if measured 1, apply X correction
    let branch_fn = |outcomes: &MeasurementOutcomes| {
        if outcomes.get_bit(QubitId(0)) == Some(true) {
            Some(
                CommandBuilder::new()
                    .x(&[0]) // Correction
                    .mz(&[0])
                    .build(),
            )
        } else {
            None // No correction needed
        }
    };

    let mut program = ConditionalProgram::new(initial, branch_fn, 1);
    let mut runner = ProgramRunner::new(SparseStab::new(1)).with_seed(42);

    println!("\nConditional Program Demo:");

    // Run multiple shots to see both branches
    let mut corrected_count = 0;
    for shot in 0..10 {
        let result = runner.run_shot(&mut program);
        let num_batches = result.num_batches;
        let corrected = num_batches == 2;
        if corrected {
            corrected_count += 1;
        }
        println!(
            "  Shot {}: {} batches ({})",
            shot,
            num_batches,
            if corrected {
                "corrected"
            } else {
                "no correction"
            }
        );
    }
    println!("  Total corrections: {corrected_count}/10");
}

/// Demonstrate a repeat-until-success pattern.
#[test]
fn demo_repeat_until_success() {
    // A program that tries to prepare a specific state and retries on failure
    struct RepeatUntilZero {
        max_attempts: usize,
        current_attempt: usize,
        success: bool,
    }

    impl CommandSource for RepeatUntilZero {
        fn next_commands(
            &mut self,
            outcomes: Option<&MeasurementOutcomes>,
        ) -> Option<pecos_neo::command::CommandQueue> {
            // Check if previous attempt succeeded (measured 0)
            if let Some(o) = outcomes
                && o.get_bit(QubitId(0)) == Some(false)
            {
                self.success = true;
                return None; // Success!
            }

            if self.current_attempt >= self.max_attempts {
                return None; // Give up
            }

            self.current_attempt += 1;

            // Try again: prepare, rotate slightly, measure
            Some(
                CommandBuilder::new()
                    .pz(&[0])
                    .h(&[0]) // 50% chance of 0
                    .mz(&[0])
                    .build(),
            )
        }

        fn is_complete(&self) -> bool {
            self.success || self.current_attempt >= self.max_attempts
        }

        fn reset(&mut self) {
            self.current_attempt = 0;
            self.success = false;
        }

        fn num_qubits(&self) -> usize {
            1
        }
    }

    let mut program = RepeatUntilZero {
        max_attempts: 10,
        current_attempt: 0,
        success: false,
    };

    let mut runner = ProgramRunner::new(SparseStab::new(1)).with_seed(42);

    println!("\nRepeat-Until-Success Demo:");

    for trial in 0..5 {
        let result = runner.run_shot(&mut program);
        println!(
            "  Trial {}: {} attempts, success={}",
            trial, result.num_batches, program.success
        );
        assert!(program.success || result.num_batches == 10);
    }
}

// --- Part 5: Combining Techniques - Path Enumeration with Error Analysis ---

/// Demonstrate combining path enumeration with error rate analysis.
#[test]
fn demo_combined_path_and_error_analysis() {
    // A syndrome extraction circuit: 2 data qubits + 1 ancilla
    // Measure ancilla to detect errors
    let syndrome_circuit = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .pz(&[2]) // ancilla
        .h(&[0])
        .h(&[1])
        .cx(&[(0, 2)]) // CNOT data -> ancilla
        .cx(&[(1, 2)])
        .mz(&[2]) // Syndrome measurement
        .mz(&[0])
        .mz(&[1])
        .build();

    let mut explorer = PathExplorer::new(SparseStab::new(3));
    let mut error_stats = PathStatistics::new();

    println!("\nCombined Path + Error Analysis Demo:");
    println!("  Analyzing a simple syndrome extraction circuit.");

    // The syndrome measurement (q2) might be non-deterministic
    // Data measurements (q0, q1) are non-deterministic after H
    // In practice we have 3 potentially non-deterministic measurements

    for path in PathEnumerator::new(3) {
        let result = explorer.run_with_path(&syndrome_circuit, &path);

        let syndrome = result.outcomes.get_bit(QubitId(2)).unwrap_or(false);
        let d0 = result.outcomes.get_bit(QubitId(0)).unwrap_or(false);
        let d1 = result.outcomes.get_bit(QubitId(1)).unwrap_or(false);

        // Check how many random measurements actually occurred
        let num_random = result.path.num_random_measurements();

        // Count paths where syndrome indicates an error
        let error_indicated = if syndrome { 1.0 } else { 0.0 };
        error_stats.add(error_indicated, path.probability());

        if path.index() < 8 {
            // Only print first few
            println!(
                "    Path '{}': syn={}, d0={}, d1={}, random_meas={}",
                path.to_binary_string(),
                u8::from(syndrome),
                u8::from(d0),
                u8::from(d1),
                num_random
            );
        }
    }

    println!("  ...");
    println!(
        "  P(syndrome=1) = {:.4} across {} paths",
        error_stats.mean(),
        error_stats.num_paths()
    );
    println!(
        "  Total probability weight: {:.4}",
        error_stats.total_weight()
    );
}

// --- Part 6: Subset Simulation for Very Rare Events ---

/// Demonstrate subset simulation for estimating rare event probabilities.
///
/// Subset simulation (Au & Beck algorithm) is ideal when:
/// - Events are too rare for standard Monte Carlo (probability < 1e-4)
/// - Importance sampling causes weight explosion
/// - There's a natural "progress" metric toward failure
#[test]
fn demo_subset_simulation_bernoulli() {
    println!("\nSubset Simulation Demo (Bernoulli Model):");
    println!("  Estimating P(damage >= threshold) for a damage accumulation process.");

    // Configuration: 100 steps, 10% damage probability per step, failure if damage >= 20
    let p_damage = 0.10;
    let num_steps = 100;
    let threshold = 20.0;

    // First, compute the analytical probability for validation
    let sim = BernoulliSubsetSimulation::new(p_damage, num_steps, threshold).with_config(
        SubsetConfig::new()
            .with_samples_per_level(1000)
            .with_seed(42),
    );

    let analytical = sim.analytical_probability();
    println!("  Parameters: p={p_damage}, n={num_steps}, threshold={threshold}");
    println!("  Analytical P(failure): {analytical:.6}");

    // Run direct Monte Carlo for comparison
    let direct_mc = sim.run_direct_mc(10000, 12345);
    println!("  Direct MC P(failure):  {direct_mc:.6} (10k samples)");

    // Run subset simulation
    let result = sim.run();
    println!("  Subset result:         {:.6}", result.probability());
    println!(
        "  Coefficient of variation: {:.4}",
        result.coefficient_of_variation
    );

    // Verify results are reasonable
    let rel_error_mc = (analytical - direct_mc).abs() / analytical;
    let rel_error_ss = (analytical - result.probability()).abs() / analytical;

    println!("  Direct MC relative error: {:.1}%", rel_error_mc * 100.0);
    println!("  Subset relative error:    {:.1}%", rel_error_ss * 100.0);

    // Note: These are demonstration tests with moderate sample sizes.
    // Statistical variation means errors can be significant for rare events.
    // Real applications would use more samples for production estimates.
    assert!(
        rel_error_mc < 0.50,
        "Direct MC should be within 50% of analytical"
    );
    assert!(
        rel_error_ss < 0.60,
        "Subset sim should be within 60% of analytical"
    );
}

/// Demonstrate proper subset simulation with checkpoint-based resampling.
///
/// This shows the Au & Beck algorithm which:
/// 1. Runs all trajectories to completion
/// 2. Sorts by score and finds adaptive thresholds
/// 3. Resamples below-threshold trajectories from above-threshold parents
/// 4. Repeats until threshold reaches failure criterion
#[test]
fn demo_proper_subset_simulation() {
    println!("\nProper Subset Simulation Demo (Au & Beck Algorithm):");

    let p_damage = 0.15;
    let damage_increment = 1.0;
    let failure_threshold = 8.0;
    let num_rounds = 20;

    let config = SubsetConfig::new()
        .with_samples_per_level(500)
        .with_threshold_fraction(0.2) // Keep top 20% at each level
        .with_max_levels(10)
        .with_seed(42);

    let sim = ProperSubsetSimulation::new(
        p_damage,
        damage_increment,
        failure_threshold,
        num_rounds,
        config,
    );

    // Use BernoulliSubsetSimulation to compute analytical probability
    // (same underlying binomial model)
    let bernoulli =
        BernoulliSubsetSimulation::new(p_damage, num_rounds, failure_threshold / damage_increment);
    let analytical = bernoulli.analytical_probability();

    println!("  Parameters: p={p_damage}, threshold={failure_threshold}, rounds={num_rounds}");
    println!("  Analytical P(failure): {analytical:.6}");

    // Run direct Monte Carlo
    let direct_mc = sim.run_direct_mc(5000);
    println!("  Direct MC P(failure):  {direct_mc:.6}");

    // Run proper subset simulation
    let result = sim.run();
    println!("  Subset result:         {:.6}", result.probability());
    println!("  Levels used: {}", result.levels.len());

    // Show level progression
    println!("  Level progression:");
    for level in &result.levels {
        println!(
            "    Level {}: threshold={:.1}, P(exceed|prev)={:.4}",
            level.level, level.threshold, level.conditional_prob
        );
    }

    // The product of conditional probabilities gives the final estimate
    let product: f64 = result.levels.iter().map(|l| l.conditional_prob).product();
    println!("  Product of conditionals: {product:.6}");

    // Verify accuracy
    let rel_error = (analytical - result.probability()).abs() / analytical;
    println!("  Relative error: {:.1}%", rel_error * 100.0);

    // For this moderate probability case, expect reasonable accuracy
    assert!(
        rel_error < 0.50,
        "Should be within 50% of analytical, got {}%",
        rel_error * 100.0
    );
}

// --- Part 7: Summary - Which Technique When? ---

/// Summary test showing when to use each technique.
#[test]
fn demo_technique_summary() {
    let sep = "=".repeat(70);
    println!("\n{sep}");
    println!("SAMPLING TECHNIQUES SUMMARY");
    println!("{sep}");

    println!("\n1. PATH EXPLORATION (path.rs)");
    println!("   Use when: Systematic analysis of all execution paths");
    println!("   Best for:");
    println!("   - Small bounded programs (< ~20 non-deterministic measurements)");
    println!("   - Exact probability computation");
    println!("   - Debugging specific execution traces");
    println!("   - QEC circuits with syndrome-dependent feedback");

    println!("\n2. IMPORTANCE SAMPLING - Error Boosting (importance.rs)");
    println!("   Use when: Estimating rare event probabilities");
    println!("   Best for:");
    println!("   - Low physical error rates (p << 1)");
    println!("   - Variance reduction in Monte Carlo");
    println!("   - Logical error rate estimation");
    println!("   - Any rare-event simulation");

    println!("\n3. OUTCOME BIASING (importance_runner.rs)");
    println!("   Use when: Exploring rare branches in control flow");
    println!("   Best for:");
    println!("   - Programs where interesting behavior is in rare branches");
    println!("   - Teleportation-like protocols");
    println!("   - Distillation circuits");
    println!("   - Any measurement-conditioned protocol");

    println!("\n4. PROGRAM RUNNER (program.rs)");
    println!("   Use when: Running programs with classical feedback");
    println!("   Best for:");
    println!("   - QASM programs with if-statements");
    println!("   - QEC with real-time decoding");
    println!("   - Repeat-until-success protocols");
    println!("   - Any hybrid quantum-classical program");

    println!("\n5. SUBSET SIMULATION (subset.rs + ecs/)");
    println!("   Use when: Estimating VERY rare events (1e-6 to 1e-12)");
    println!("   Best for:");
    println!("   - Logical error rates in high-distance codes");
    println!("   - Events too rare for importance sampling alone");
    println!("   - Processes with intermediate 'criticality' levels");
    println!("   - When simple error boosting causes weight explosion");
    println!("   How it works:");
    println!("   - Define intermediate thresholds (syndrome weight, damage)");
    println!("   - Clone promising trajectories that cross thresholds");
    println!("   - Prune trajectories that don't progress");
    println!("   - P(rare) = product of conditional probabilities at each level");

    println!("\n{sep}");
    println!("COMBINATIONS:");
    println!("- Path exploration + noise: Enumerate paths, apply noise at each");
    println!("- Importance sampling + outcome bias: Boost errors AND branch exploration");
    println!("- Program runner + importance sampling: Dynamic programs with rare events");
    println!("- Subset sim + ECS World: Clone simulator states for trajectory splitting");
    println!("{sep}");

    println!("\nSELECTION GUIDE:");
    println!("  Error rate 1e-2 to 1e-4: Standard Monte Carlo or Importance Sampling");
    println!("  Error rate 1e-4 to 1e-7: Importance Sampling with boosting");
    println!("  Error rate 1e-7 to 1e-12: Subset Simulation / Multilevel Splitting");
    println!("  Exact analysis needed: Path Enumeration (if feasible)");
}
