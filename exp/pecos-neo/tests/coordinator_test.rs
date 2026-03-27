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

//! Tests for `ParallelCoordinator` comparing against `MonteCarloRunner`.
//!
//! These tests verify that:
//! 1. `ParallelCoordinator` produces correct results
//! 2. Results are statistically equivalent to `MonteCarloRunner`
//! 3. Synchronization points work correctly for rare event simulation

use pecos_core::QubitId;
use pecos_neo::command::CommandBuilder;
use pecos_neo::ecs::{ParallelConfig, ParallelCoordinator};
use pecos_neo::noise::{ComposableNoiseModel, SingleQubitChannel};
use pecos_neo::runner::CircuitRunner;
use pecos_neo::sampling::{MonteCarloConfig, MonteCarloRunner};
use pecos_simulators::SparseStab;
use std::collections::BTreeMap;

const NUM_SHOTS: usize = 1000;
const TOLERANCE_PERCENT: f64 = 10.0;

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
                "Distribution mismatch for '{key}': {rate1:.4} vs {rate2:.4} (diff: {diff:.4}, tolerance: {tolerance:.4})"
            );
            return false;
        }
    }

    true
}

#[test]
fn test_coordinator_vs_monte_carlo_bell_state() {
    // Compare ParallelCoordinator against MonteCarloRunner for Bell state
    let commands = CommandBuilder::new()
        .pz(&[0])
        .pz(&[1])
        .h(&[0])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    // Run with MonteCarloRunner
    let mc_config = MonteCarloConfig::new()
        .with_shots(NUM_SHOTS)
        .with_workers(2)
        .with_seed(42);

    let mc_results = MonteCarloRunner::run(
        &commands,
        mc_config,
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

    let mut mc_counts = BTreeMap::new();
    for result in mc_results.iter() {
        *mc_counts.entry(result.clone()).or_insert(0) += 1;
    }

    // Run with ParallelCoordinator
    // Use 2 workers with 500 entities each = 1000 total
    let coord_config = ParallelConfig::new()
        .with_workers(2)
        .with_entities_per_worker(NUM_SHOTS / 2)
        .with_seed(42);

    let coordinator: ParallelCoordinator<SparseStab> = ParallelCoordinator::new(coord_config);

    let coord_results = coordinator.run(
        || SparseStab::new(2),
        |world| {
            let commands = CommandBuilder::new()
                .pz(&[0])
                .pz(&[1])
                .h(&[0])
                .cx(&[(0, 1)])
                .mz(&[0])
                .mz(&[1])
                .build();

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
                    let b0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
                    let b1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);
                    format!(
                        "{}{}",
                        if b0 { '1' } else { '0' },
                        if b1 { '1' } else { '0' }
                    )
                })
                .collect()
        },
    );

    let mut coord_counts = BTreeMap::new();
    for result in coord_results.iter() {
        *coord_counts.entry(result.clone()).or_insert(0) += 1;
    }

    // Bell state should only produce "00" or "11"
    let mc_valid = mc_counts.get("00").unwrap_or(&0) + mc_counts.get("11").unwrap_or(&0);
    let coord_valid = coord_counts.get("00").unwrap_or(&0) + coord_counts.get("11").unwrap_or(&0);

    assert_eq!(
        mc_valid, NUM_SHOTS,
        "MC: Bell state should only produce correlated outcomes"
    );
    assert_eq!(
        coord_valid, NUM_SHOTS,
        "Coord: Bell state should only produce correlated outcomes"
    );

    // Distributions should match
    assert!(
        distributions_match(&mc_counts, &coord_counts, NUM_SHOTS, TOLERANCE_PERCENT),
        "Coordinator should match MonteCarloRunner distribution"
    );
}

#[test]
fn test_coordinator_vs_monte_carlo_with_noise() {
    // Compare with depolarizing noise
    let p1 = 0.05;

    let commands = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

    // Run with MonteCarloRunner
    let mc_config = MonteCarloConfig::new()
        .with_shots(NUM_SHOTS)
        .with_workers(2)
        .with_seed(42);

    let mc_results = MonteCarloRunner::run(
        &commands,
        mc_config,
        || {
            let noise =
                ComposableNoiseModel::new().add_channel(SingleQubitChannel::depolarizing(p1));
            (CircuitRunner::new().with_noise(noise), SparseStab::new(1))
        },
        |outcomes| outcomes.get_bit(QubitId(0)).unwrap_or(false),
    );

    let mc_ones = mc_results.iter().filter(|&&b| b).count();
    let mc_rate = mc_ones as f64 / NUM_SHOTS as f64;

    // Run with ParallelCoordinator
    let coord_config = ParallelConfig::new()
        .with_workers(2)
        .with_entities_per_worker(NUM_SHOTS / 2)
        .with_seed(42);

    let coordinator: ParallelCoordinator<SparseStab> = ParallelCoordinator::new(coord_config);

    let coord_results = coordinator.run(
        || SparseStab::new(1),
        |world| {
            let commands = CommandBuilder::new().pz(&[0]).x(&[0]).mz(&[0]).build();

            world
                .entities()
                .map(|entity| {
                    let sim_comp = world.simulators.get(entity).unwrap();
                    let rng_comp = world.rngs.get(entity).unwrap();

                    let noise = ComposableNoiseModel::new()
                        .add_channel(SingleQubitChannel::depolarizing(p1));
                    let mut sim = sim_comp.simulator.clone();
                    let mut runner = CircuitRunner::<SparseStab>::new()
                        .with_noise(noise)
                        .with_rng(rng_comp.rng.clone());

                    sim.reset();
                    let outcomes = runner.apply_circuit(&mut sim, &commands).unwrap();
                    outcomes.get_bit(QubitId(0)).unwrap_or(false)
                })
                .collect()
        },
    );

    let coord_ones = coord_results.iter().filter(|&&b| b).count();
    let coord_rate = coord_ones as f64 / NUM_SHOTS as f64;

    println!("MC ones rate: {mc_rate:.4}");
    println!("Coordinator ones rate: {coord_rate:.4}");

    // Both should be high (X flips |0> to |1>, with some noise errors)
    assert!(mc_rate > 0.8, "MC should have mostly ones");
    assert!(coord_rate > 0.8, "Coordinator should have mostly ones");

    // Rates should be similar
    let diff = (mc_rate - coord_rate).abs();
    assert!(
        diff < TOLERANCE_PERCENT / 100.0,
        "Rates should match: mc={mc_rate:.4}, coord={coord_rate:.4}, diff={diff:.4}"
    );
}

#[test]
fn test_coordinator_determinism() {
    // Test that coordinator produces identical results with same seed
    let config = ParallelConfig::new()
        .with_workers(2)
        .with_entities_per_worker(50)
        .with_seed(42);

    // Run twice
    let coord1: ParallelCoordinator<SparseStab> = ParallelCoordinator::new(config.clone());
    let coord2: ParallelCoordinator<SparseStab> = ParallelCoordinator::new(config);

    let results1: Vec<bool> = coord1
        .run(
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
        )
        .into_iter()
        .collect();

    let results2: Vec<bool> = coord2
        .run(
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
        )
        .into_iter()
        .collect();

    // Results should be identical
    assert_eq!(
        results1, results2,
        "Same seed should produce identical results"
    );
}

#[test]
fn test_coordinator_sync_points() {
    // Test that sync points are called correctly
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    let config = ParallelConfig::new()
        .with_workers(2)
        .with_entities_per_worker(5)
        .with_seed(42)
        .with_sync_interval(3);

    let coordinator: ParallelCoordinator<SparseStab> = ParallelCoordinator::new(config);

    let sync_count = Arc::new(AtomicUsize::new(0));
    let sync_count_clone = Arc::clone(&sync_count);

    let result = coordinator.run_with_sync::<_, _, _, ()>(
        || SparseStab::new(1),
        10, // 10 steps
        |_world, _step| {
            // Do nothing per step
        },
        move |_workers, step| {
            sync_count_clone.fetch_add(1, Ordering::SeqCst);
            println!("Sync at step {step}");
        },
    );

    // With sync_interval=3 and 10 steps:
    // Steps 0,1,2 -> sync at 3
    // Steps 3,4,5 -> sync at 6
    // Steps 6,7,8 -> sync at 9
    // Step 9 -> end (no sync at end)
    assert_eq!(sync_count.load(Ordering::SeqCst), 3);
    assert_eq!(result.stats.sync_points, 3);
}

#[test]
fn test_coordinator_hadamard_distribution() {
    // Test that Hadamard produces ~50/50 distribution
    let config = ParallelConfig::new()
        .with_workers(4)
        .with_entities_per_worker(250)
        .with_seed(42);

    let coordinator: ParallelCoordinator<SparseStab> = ParallelCoordinator::new(config);

    let results = coordinator.run(
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

    let total = results.len();
    let ones = results.iter().filter(|&&b| b).count();
    let rate = ones as f64 / total as f64;

    println!("Hadamard distribution: {ones}/{total} = {rate:.4}");

    // Should be close to 0.5
    assert!(
        (rate - 0.5).abs() < 0.1,
        "Hadamard should give ~50%, got {rate:.4}"
    );
}
