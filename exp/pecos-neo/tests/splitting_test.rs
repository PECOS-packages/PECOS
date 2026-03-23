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

//! Tests for splitting and resampling functionality.

use pecos_neo::ecs::{SplitDecision, SplitStats, SubsetLevel, World};
use pecos_neo::sampling::SampleWeight;
use pecos_random::PecosRng;
use pecos_simulators::SparseStab;

#[test]
fn test_apply_split_decisions_prune() {
    let mut world: World<SparseStab> = World::new(42);

    // Create some entities
    let e1 = world.spawn_with_simulator(SparseStab::new(1));
    let e2 = world.spawn_with_simulator(SparseStab::new(1));
    let e3 = world.spawn_with_simulator(SparseStab::new(1));

    assert_eq!(world.active_entities().len(), 3);

    // Prune e2
    let decisions = vec![(e1, 1), (e2, 0), (e3, 1)];
    let created = world.apply_split_decisions(&decisions);

    assert_eq!(created, 0);
    assert_eq!(world.active_entities().len(), 2);
    assert!(world.active_entities().contains(&e1));
    assert!(!world.active_entities().contains(&e2));
    assert!(world.active_entities().contains(&e3));
}

#[test]
fn test_apply_split_decisions_split() {
    let mut world: World<SparseStab> = World::new(42);

    // Create one entity
    let e1 = world.spawn_with_simulator(SparseStab::new(1));

    assert_eq!(world.active_entities().len(), 1);

    // Split into 4 copies
    let decisions = vec![(e1, 4)];
    let created = world.apply_split_decisions(&decisions);

    assert_eq!(created, 3); // 3 new + 1 original = 4 total
    assert_eq!(world.active_entities().len(), 4);

    // Check weights are split
    for entity in world.active_entities() {
        let weight = world.weights.get(entity).unwrap().weight.weight();
        assert!(
            (weight - 0.25).abs() < 0.01,
            "Weight should be ~0.25, got {weight}"
        );
    }
}

#[test]
fn test_apply_split_decisions_mixed() {
    let mut world: World<SparseStab> = World::new(42);

    let e1 = world.spawn_with_simulator(SparseStab::new(1));
    let e2 = world.spawn_with_simulator(SparseStab::new(1));
    let e3 = world.spawn_with_simulator(SparseStab::new(1));

    // e1: keep, e2: prune, e3: split into 3
    let decisions = vec![(e1, 1), (e2, 0), (e3, 3)];
    let created = world.apply_split_decisions(&decisions);

    assert_eq!(created, 2); // 2 new from e3
    assert_eq!(world.active_entities().len(), 4); // e1 + (e3 + 2 clones)
}

#[test]
fn test_resample_by_weight_preserves_total_weight() {
    let mut world: World<SparseStab> = World::new(42);

    // Create entities with different weights
    let e1 = world.spawn_with_simulator(SparseStab::new(1));
    let e2 = world.spawn_with_simulator(SparseStab::new(1));
    let e3 = world.spawn_with_simulator(SparseStab::new(1));

    // Set weights: e1=1.0, e2=2.0, e3=3.0, total=6.0
    world.weights.get_mut(e1).unwrap().weight = SampleWeight::from_linear(1.0);
    world.weights.get_mut(e2).unwrap().weight = SampleWeight::from_linear(2.0);
    world.weights.get_mut(e3).unwrap().weight = SampleWeight::from_linear(3.0);

    let total_before = world.total_weight();
    assert!((total_before - 6.0).abs() < 1e-10);

    // Resample to 10 entities
    let mut rng = PecosRng::seed_from_u64(42);
    let count = world.resample_by_weight(10, &mut rng);

    assert_eq!(count, 10);

    // Total weight should be preserved
    let total_after = world.total_weight();
    assert!(
        (total_before - total_after).abs() < 0.1,
        "Total weight should be preserved: before={total_before}, after={total_after}"
    );
}

#[test]
fn test_resample_by_weight_respects_probabilities() {
    // Run multiple trials to verify that higher-weight entities are selected more often
    let mut selection_counts = [0usize; 3];
    let trials = 1000;

    for trial in 0..trials {
        let mut world: World<SparseStab> = World::new(trial);

        let e1 = world.spawn_with_simulator(SparseStab::new(1));
        let e2 = world.spawn_with_simulator(SparseStab::new(1));
        let e3 = world.spawn_with_simulator(SparseStab::new(1));

        // Weights: e1=1, e2=2, e3=7 (e3 should be selected ~70% of time)
        world.weights.get_mut(e1).unwrap().weight = SampleWeight::from_linear(1.0);
        world.weights.get_mut(e2).unwrap().weight = SampleWeight::from_linear(2.0);
        world.weights.get_mut(e3).unwrap().weight = SampleWeight::from_linear(7.0);

        let mut rng = PecosRng::seed_from_u64(trial + 1000);
        world.resample_by_weight(1, &mut rng);

        // Count which entity survived
        let active = world.active_entities();
        assert_eq!(active.len(), 1);
        let survivor = *active.first().unwrap();

        if survivor == e1 {
            selection_counts[0] += 1;
        } else if survivor == e2 {
            selection_counts[1] += 1;
        } else if survivor == e3 {
            selection_counts[2] += 1;
        }
    }

    // Check proportions (with generous tolerance for statistical variation)
    let p1 = selection_counts[0] as f64 / trials as f64;
    let p2 = selection_counts[1] as f64 / trials as f64;
    let p3 = selection_counts[2] as f64 / trials as f64;

    println!("Selection proportions: e1={p1:.3}, e2={p2:.3}, e3={p3:.3}");

    // Expected: p1~0.1, p2~0.2, p3~0.7 (within reasonable tolerance)
    assert!(
        p1 < 0.25,
        "e1 should be selected ~10%, got {:.1}%",
        p1 * 100.0
    );
    assert!(
        p3 > 0.5,
        "e3 should be selected ~70%, got {:.1}%",
        p3 * 100.0
    );
}

#[test]
fn test_subset_level() {
    let level = SubsetLevel::new(0.5, 100);
    assert!((level.threshold - 0.5).abs() < 1e-10);
    assert_eq!(level.target_count, 100);
}

#[test]
fn test_split_decision_constructors() {
    let prune = SplitDecision::prune(pecos_neo::ecs::EntityId(1));
    assert_eq!(prune.copies, 0);

    let keep = SplitDecision::keep(pecos_neo::ecs::EntityId(2));
    assert_eq!(keep.copies, 1);

    let split = SplitDecision::split(
        pecos_neo::ecs::EntityId(3),
        4,
        SampleWeight::from_linear(0.25),
    );
    assert_eq!(split.copies, 4);
    assert!((split.new_weight.weight() - 0.25).abs() < 1e-10);
}

#[test]
fn test_split_stats_weight_preservation() {
    let stats = SplitStats {
        entities_before: 100,
        entities_after: 150,
        pruned: 20,
        split: 70,
        total_weight_before: 100.0,
        total_weight_after: 100.0,
    };

    assert!(stats.weight_preserved(1e-10));

    let bad_stats = SplitStats {
        total_weight_before: 100.0,
        total_weight_after: 90.0,
        ..stats
    };

    assert!(!bad_stats.weight_preserved(1e-10));
}

#[test]
fn test_entity_transfer() {
    let mut world1: World<SparseStab> = World::new(42);
    let mut world2: World<SparseStab> = World::new(43);

    // Create entity in world1 with specific weight
    let e1 = world1.spawn_with_simulator(SparseStab::new(1));
    world1.weights.get_mut(e1).unwrap().weight = SampleWeight::from_linear(5.0);

    assert_eq!(world1.active_entities().len(), 1);
    assert_eq!(world2.active_entities().len(), 0);

    // Extract entity from world1
    let transfer = world1.extract_entity(e1).expect("Should extract");
    assert_eq!(world1.active_entities().len(), 0);
    assert!((transfer.weight.weight() - 5.0).abs() < 1e-10);

    // Import into world2
    let e2 = world2.import_entity(transfer);
    assert_eq!(world2.active_entities().len(), 1);

    // Weight should be preserved
    let imported_weight = world2.weights.get(e2).unwrap().weight.weight();
    assert!((imported_weight - 5.0).abs() < 1e-10);
}

#[test]
fn test_redistribution_at_sync_points() {
    use pecos_neo::ecs::{WorkerState, redistribute_by_weight};

    // Create workers manually (simulating what run_with_sync does internally)
    let mut workers: Vec<WorkerState<SparseStab>> = (0..2)
        .map(|id| {
            let mut worker = WorkerState::new(id, 42);
            // Each worker gets 5 entities with varying weights
            for i in 0..5 {
                let e = worker.world.spawn_with_simulator(SparseStab::new(1));
                worker.world.weights.get_mut(e).unwrap().weight =
                    SampleWeight::from_linear(f64::from(i + 1));
            }
            worker
        })
        .collect();

    // Total weight should be 2 * (1+2+3+4+5) = 30
    let total_before: f64 = workers.iter().map(|w| w.world.total_weight()).sum();
    assert!((total_before - 30.0).abs() < 1e-10);

    // Simulate a sync point with redistribution
    let mut rng = PecosRng::seed_from_u64(123);
    let stats = redistribute_by_weight(&mut workers, 5, &mut rng);

    // Weight should be preserved
    assert!(
        stats.weight_preserved(0.1),
        "Weight not preserved: {} -> {}",
        stats.weight_before,
        stats.weight_after
    );

    // Should have 10 entities total (5 per worker)
    assert_eq!(stats.entities_after, 10);

    // Each entity should have weight = 30/10 = 3.0
    for worker in &workers {
        for entity in worker.world.active_entities() {
            let w = worker.world.weights.get(entity).unwrap().weight.weight();
            assert!(
                (w - 3.0).abs() < 1e-10,
                "Entity weight should be 3.0, got {w}"
            );
        }
    }
}

/// Test demonstrating subset simulation for rare event estimation.
///
/// This simulates a simplified model where trajectories accumulate "damage"
/// (analogous to syndrome weight in QEC). We use subset simulation to
/// efficiently estimate the probability of reaching a high damage level.
#[test]
fn test_subset_simulation_workflow() {
    use pecos_neo::ecs::{WorkerState, redistribute_by_weight};
    use rand::RngExt;

    // Configuration
    let num_workers = 2;
    let entities_per_worker = 50;
    let total_entities = num_workers * entities_per_worker;
    let num_steps = 10; // Steps per level
    let damage_probability = 0.1; // Probability of damage per step
    let levels = [1.0, 2.0, 3.0]; // Damage thresholds for each level

    // Track damage for each entity using a simple approach:
    // We'll store damage in a separate structure keyed by entity ID
    // In a real implementation, this would be part of the simulator state

    let mut workers: Vec<WorkerState<SparseStab>> = (0..num_workers)
        .map(|id| {
            let mut worker = WorkerState::new(id, 42 + id as u64);
            for _ in 0..entities_per_worker {
                worker.world.spawn_with_simulator(SparseStab::new(1));
            }
            worker
        })
        .collect();

    // Track damage per entity (entity_id.0 -> damage)
    let mut damage: std::collections::BTreeMap<u64, f64> = std::collections::BTreeMap::new();
    for worker in &workers {
        for entity in worker.world.active_entities() {
            damage.insert(entity.0, 0.0);
        }
    }

    let mut rng = PecosRng::seed_from_u64(12345);
    let mut level_probabilities: Vec<f64> = Vec::new();

    // Run subset simulation
    for (level_idx, &threshold) in levels.iter().enumerate() {
        // Run simulation steps
        for _ in 0..num_steps {
            for worker in &mut workers {
                for entity in worker.world.active_entities() {
                    if let Some(d) = damage.get_mut(&entity.0) {
                        // Accumulate damage with some probability
                        if rng.random::<f64>() < damage_probability {
                            *d += 0.5;
                        }
                    }
                }
            }
        }

        // Count entities that crossed the threshold
        let mut above_threshold = 0;
        let mut total_weight = 0.0;
        let mut weight_above = 0.0;

        for worker in &workers {
            for entity in worker.world.active_entities() {
                let w = worker
                    .world
                    .weights
                    .get(entity)
                    .map_or(1.0, |wc| wc.weight.weight());
                total_weight += w;

                if let Some(&d) = damage.get(&entity.0)
                    && d >= threshold
                {
                    above_threshold += 1;
                    weight_above += w;
                }
            }
        }

        // Conditional probability for this level
        let p_level = if total_weight > 0.0 {
            weight_above / total_weight
        } else {
            0.0
        };
        level_probabilities.push(p_level);

        println!(
            "Level {}: threshold={:.1}, above={}/{}, p={:.4}",
            level_idx + 1,
            threshold,
            above_threshold,
            total_entities,
            p_level
        );

        // If no entities crossed, we can't continue
        if above_threshold == 0 {
            break;
        }

        // Prune entities below threshold before redistribution.
        // We need to manually remove them since SampleWeight doesn't support zero.
        // First collect survivors' damage values (entities above threshold)
        let mut survivor_damage: std::collections::BTreeMap<u64, f64> =
            std::collections::BTreeMap::new();
        for worker in &workers {
            for entity in worker.world.active_entities() {
                if let Some(&d) = damage.get(&entity.0)
                    && d >= threshold
                {
                    survivor_damage.insert(entity.0, d);
                }
            }
        }

        // Now despawn entities below threshold
        for worker in &mut workers {
            let to_despawn: Vec<_> = worker
                .world
                .active_entities()
                .into_iter()
                .filter(|e| damage.get(&e.0).is_none_or(|&d| d < threshold))
                .collect();

            for entity in to_despawn {
                worker.world.despawn(entity);
            }
        }

        // Redistribute survivors to restore population
        let stats = redistribute_by_weight(&mut workers, entities_per_worker, &mut rng);

        // Update damage map for new entities (inherit from source)
        // For simplicity in this test, we assign the average damage to new entities
        let avg_damage = if survivor_damage.is_empty() {
            threshold
        } else {
            survivor_damage.values().sum::<f64>() / survivor_damage.len() as f64
        };

        damage.clear();
        for worker in &workers {
            for entity in worker.world.active_entities() {
                // New entities get average damage of survivors
                damage.insert(entity.0, avg_damage);
            }
        }

        println!(
            "  After resample: {} entities, weight preserved: {}",
            stats.entities_after,
            stats.weight_preserved(0.01)
        );
    }

    // Compute overall rare event probability
    let overall_probability: f64 = level_probabilities.iter().product();
    println!("\nOverall rare event probability: {overall_probability:.6}");

    // Verify the simulation ran correctly
    assert!(
        !level_probabilities.is_empty(),
        "Should have computed at least one level"
    );

    // Each level probability should be between 0 and 1
    for (i, &p) in level_probabilities.iter().enumerate() {
        assert!(
            (0.0..=1.0).contains(&p),
            "Level {i} probability {p} out of range"
        );
    }

    // The overall probability should be positive, indicating we found rare events
    assert!(
        overall_probability > 0.0,
        "Should have found some rare events"
    );

    // All three levels should have been reached in this simulation
    assert_eq!(
        level_probabilities.len(),
        3,
        "Should have computed all 3 levels"
    );
}

// ============================================================================
// Additional Validation Tests for Rare Event Infrastructure
// ============================================================================

/// Verify that weight is preserved exactly (within floating point tolerance)
/// across multiple redistribution operations.
#[test]
fn test_redistribution_exact_weight_preservation() {
    use pecos_neo::ecs::{WorkerState, redistribute_by_weight};

    let mut workers: Vec<WorkerState<SparseStab>> = (0..4)
        .map(|id| {
            let mut worker = WorkerState::new(id, 100 + id as u64);
            for i in 0..10 {
                let e = worker.world.spawn_with_simulator(SparseStab::new(1));
                // Set varying weights
                worker.world.weights.get_mut(e).unwrap().weight =
                    SampleWeight::from_linear(0.1 * f64::from(i + 1));
            }
            worker
        })
        .collect();

    let initial_weight: f64 = workers.iter().map(|w| w.world.total_weight()).sum();

    // Perform multiple redistributions
    let mut rng = PecosRng::seed_from_u64(999);
    for _ in 0..5 {
        let stats = redistribute_by_weight(&mut workers, 10, &mut rng);
        assert!(
            stats.weight_preserved(1e-10),
            "Weight not preserved: {} -> {}",
            stats.weight_before,
            stats.weight_after
        );
    }

    let final_weight: f64 = workers.iter().map(|w| w.world.total_weight()).sum();
    assert!(
        (initial_weight - final_weight).abs() < 1e-10,
        "Weight drift after multiple redistributions: {initial_weight} -> {final_weight}"
    );
}

/// Test redistribution with extreme weight distributions.
#[test]
fn test_redistribution_extreme_weights() {
    use pecos_neo::ecs::{WorkerState, redistribute_by_weight};

    let mut workers: Vec<WorkerState<SparseStab>> = (0..2)
        .map(|id| WorkerState::new(id, 42 + id as u64))
        .collect();

    // One entity with very high weight, rest with very low
    let e1 = workers[0].world.spawn_with_simulator(SparseStab::new(1));
    workers[0].world.weights.get_mut(e1).unwrap().weight = SampleWeight::from_linear(100.0);

    for _ in 0..9 {
        let e = workers[0].world.spawn_with_simulator(SparseStab::new(1));
        workers[0].world.weights.get_mut(e).unwrap().weight = SampleWeight::from_linear(0.001);
    }

    let weight_before: f64 = workers.iter().map(|w| w.world.total_weight()).sum();

    let mut rng = PecosRng::seed_from_u64(12345);
    let stats = redistribute_by_weight(&mut workers, 5, &mut rng);

    // Weight should be preserved
    assert!(
        stats.weight_preserved(1e-10),
        "Weight not preserved with extreme weights: {} -> {}",
        stats.weight_before,
        stats.weight_after
    );

    // Total weight should match
    let weight_after: f64 = workers.iter().map(|w| w.world.total_weight()).sum();
    assert!(
        (weight_before - weight_after).abs() < 1e-10,
        "Total weight changed: {weight_before} -> {weight_after}"
    );
}

/// Test that `resample_by_weight` converges to correct proportions statistically.
#[test]
fn test_resampling_statistical_correctness() {
    // Run many trials and check that selection frequencies match weight proportions
    let trials = 500;
    let mut selection_counts = [0usize; 3];

    for trial in 0..trials {
        let mut world: World<SparseStab> = World::new(1000 + trial);

        let e1 = world.spawn_with_simulator(SparseStab::new(1));
        let e2 = world.spawn_with_simulator(SparseStab::new(1));
        let e3 = world.spawn_with_simulator(SparseStab::new(1));

        // Weights: 1, 3, 6 (proportions: 0.1, 0.3, 0.6)
        world.weights.get_mut(e1).unwrap().weight = SampleWeight::from_linear(1.0);
        world.weights.get_mut(e2).unwrap().weight = SampleWeight::from_linear(3.0);
        world.weights.get_mut(e3).unwrap().weight = SampleWeight::from_linear(6.0);

        let mut rng = PecosRng::seed_from_u64(trial + 5000);
        world.resample_by_weight(1, &mut rng);

        // Count which entity survived
        let active = world.active_entities();
        assert_eq!(
            active.len(),
            1,
            "Should have exactly 1 entity after resampling to 1"
        );

        let survivor = active[0];
        if survivor == e1 {
            selection_counts[0] += 1;
        } else if survivor == e2 {
            selection_counts[1] += 1;
        } else if survivor == e3 {
            selection_counts[2] += 1;
        }
    }

    // Check proportions (with tolerance for statistical variation)
    let p1 = selection_counts[0] as f64 / trials as f64;
    let p2 = selection_counts[1] as f64 / trials as f64;
    let p3 = selection_counts[2] as f64 / trials as f64;

    println!(
        "Resampling proportions: e1={p1:.3} (expected 0.1), e2={p2:.3} (expected 0.3), e3={p3:.3} (expected 0.6)"
    );

    // Allow 10% deviation from expected
    assert!(
        (p1 - 0.1).abs() < 0.1,
        "e1 proportion {p1:.3} too far from expected 0.1"
    );
    assert!(
        (p2 - 0.3).abs() < 0.1,
        "e2 proportion {p2:.3} too far from expected 0.3"
    );
    assert!(
        (p3 - 0.6).abs() < 0.1,
        "e3 proportion {p3:.3} too far from expected 0.6"
    );
}

/// Test that entity state (including simulator) is preserved through transfer.
#[test]
fn test_entity_state_preservation_through_transfer() {
    use pecos_core::QubitId;
    use pecos_simulators::CliffordGateable;

    let mut world1: World<SparseStab> = World::new(42);

    // Create entity and modify its simulator state
    let e1 = world1.spawn_with_simulator(SparseStab::new(2));

    // Apply some gates to create a non-trivial state
    if let Some(sim_comp) = world1.simulators.get_mut(e1) {
        sim_comp.simulator.h(&[QubitId(0)]);
        sim_comp.simulator.cx(&[QubitId(0), QubitId(1)]);
    }

    // Mark a qubit as leaked in noise context
    if let Some(ctx_comp) = world1.noise_contexts.get_mut(e1) {
        ctx_comp.context.mark_leaked(QubitId(1));
    }

    // Set a non-default weight
    world1.weights.get_mut(e1).unwrap().weight = SampleWeight::from_linear(3.5);

    // Extract and import into another world
    let transfer = world1.extract_entity(e1).expect("Should extract");

    assert_eq!(world1.active_entities().len(), 0);
    assert!((transfer.weight.weight() - 3.5).abs() < 1e-10);
    assert!(transfer.noise_context.is_leaked(QubitId(1)));

    let mut world2: World<SparseStab> = World::new(43);
    let e2 = world2.import_entity(transfer);

    // Verify state was preserved
    assert_eq!(world2.active_entities().len(), 1);

    let imported_weight = world2.weights.get(e2).unwrap().weight.weight();
    assert!(
        (imported_weight - 3.5).abs() < 1e-10,
        "Weight not preserved: expected 3.5, got {imported_weight}"
    );

    let imported_ctx = &world2.noise_contexts.get(e2).unwrap().context;
    assert!(
        imported_ctx.is_leaked(QubitId(1)),
        "Leakage state not preserved"
    );
}

/// Test that splitting maintains correct weight invariants.
#[test]
fn test_splitting_weight_invariants() {
    let mut world: World<SparseStab> = World::new(42);

    // Create entity with weight 1.0
    let e1 = world.spawn_with_simulator(SparseStab::new(1));
    let initial_weight = world.total_weight();
    assert!((initial_weight - 1.0).abs() < 1e-10);

    // Split into 4 - total weight should remain 1.0
    let clones = world.split_entity(e1, 4);
    assert_eq!(clones.len(), 3);
    assert_eq!(world.active_entities().len(), 4);

    let total_after_split = world.total_weight();
    assert!(
        (total_after_split - initial_weight).abs() < 1e-10,
        "Weight not preserved after split: {initial_weight} -> {total_after_split}"
    );

    // Each entity should have weight 0.25
    for entity in world.active_entities() {
        let w = world.weights.get(entity).unwrap().weight.weight();
        assert!(
            (w - 0.25).abs() < 1e-10,
            "Entity weight should be 0.25, got {w}"
        );
    }

    // Now apply decisions: prune 2, keep 1, split 1 into 3
    let active = world.active_entities();
    let decisions = vec![
        (active[0], 0), // prune
        (active[1], 0), // prune
        (active[2], 1), // keep
        (active[3], 3), // split into 3
    ];

    world.apply_split_decisions(&decisions);

    // After: 1 kept (0.25) + 3 from split (0.25/3 each = 0.0833... each, total 0.25)
    // Total should still be ~0.5 (2 entities were pruned)
    let weight_after = world.total_weight();

    // Note: apply_split_decisions splits weight for split entities
    // So we have: 1 entity at 0.25 + 3 entities at 0.25/3 each
    let expected_weight = 0.25 + 0.25; // kept + split total
    assert!(
        (weight_after - expected_weight).abs() < 1e-10,
        "Weight after apply_split_decisions: expected {expected_weight}, got {weight_after}"
    );
}

/// Test edge case: resampling when all entities are pruned.
#[test]
fn test_resampling_edge_cases() {
    // Test with empty world
    let mut world: World<SparseStab> = World::new(42);
    let mut rng = PecosRng::seed_from_u64(123);
    let count = world.resample_by_weight(10, &mut rng);
    assert_eq!(count, 0, "Empty world should return 0");

    // Test resampling to 0 entities
    let e = world.spawn_with_simulator(SparseStab::new(1));
    assert!(world.is_alive(e));

    let count = world.resample_by_weight(0, &mut rng);
    assert_eq!(count, 0, "Resampling to 0 should return 0");
}

/// Test that determinism is preserved across redistribution operations.
#[test]
fn test_redistribution_determinism() {
    use pecos_neo::ecs::{WorkerState, redistribute_by_weight};

    fn create_workers() -> Vec<WorkerState<SparseStab>> {
        (0..2)
            .map(|id| {
                let mut worker = WorkerState::new(id, 42);
                for i in 0..5 {
                    let e = worker.world.spawn_with_simulator(SparseStab::new(1));
                    worker.world.weights.get_mut(e).unwrap().weight =
                        SampleWeight::from_linear(f64::from(i + 1));
                }
                worker
            })
            .collect()
    }

    // Run redistribution with same seed twice
    let mut workers1 = create_workers();
    let mut rng1 = PecosRng::seed_from_u64(999);
    let stats1 = redistribute_by_weight(&mut workers1, 5, &mut rng1);

    let mut workers2 = create_workers();
    let mut rng2 = PecosRng::seed_from_u64(999);
    let stats2 = redistribute_by_weight(&mut workers2, 5, &mut rng2);

    // Results should be identical
    assert_eq!(stats1.entities_after, stats2.entities_after);
    assert!(
        (stats1.weight_after - stats2.weight_after).abs() < 1e-10,
        "Weights differ: {} vs {}",
        stats1.weight_after,
        stats2.weight_after
    );

    // Entity counts per worker should match
    for (w1, w2) in workers1.iter().zip(workers2.iter()) {
        assert_eq!(
            w1.world.active_entities().len(),
            w2.world.active_entities().len(),
            "Entity counts differ"
        );
    }
}

// ============================================================================
// Quantum Circuit Subset Simulation Demonstration
// ============================================================================

/// Demonstrates ECS-based trajectory tracking for rare event estimation.
///
/// This test shows how to use the ECS World infrastructure to track multiple
/// quantum trajectories in parallel, which is the foundation for subset simulation.
///
/// The approach:
/// 1. Run N trajectories with noise, tracking syndrome/error state
/// 2. At each level, compute "criticality" scores (syndrome weight)
/// 3. Demonstrate entity cloning for promising trajectories
/// 4. Validate weight preservation through splitting
///
/// This demonstrates the infrastructure that enables proper subset simulation.
#[test]
fn test_quantum_circuit_subset_simulation() {
    use pecos_core::QubitId;
    use pecos_neo::command::CommandBuilder;
    use pecos_neo::noise::{ComposableNoiseModel, SingleQubitChannel};
    use pecos_neo::runner::CircuitRunner;

    // Configuration
    let num_rounds = 3; // Error correction rounds
    let p_error = 0.05; // Physical error rate per gate
    let num_samples = 100;
    let failure_threshold = 2; // Logical failure if >= 2 syndrome detections

    // Noise model factory (creates fresh noise model each time)
    fn make_noise(p: f64) -> ComposableNoiseModel {
        ComposableNoiseModel::new().add_channel(SingleQubitChannel::bit_flip(p))
    }

    // Syndrome extraction circuit for simple error detection
    // Data qubit: 0
    // Ancilla qubits: 1, 2
    fn syndrome_circuit() -> pecos_neo::command::CommandQueue {
        CommandBuilder::new()
            .pz(1) // Reset ancilla
            .cx(0, 1) // CNOT data -> ancilla
            .mz(1)
            .build()
    }

    // Direct Monte Carlo for comparison - run full shots
    let mut direct_failures = 0;

    for sample in 0..num_samples {
        let mut syndrome_detections = 0;
        let mut state = SparseStab::new(2);
        let mut runner =
            CircuitRunner::<SparseStab>::new().with_rng(PecosRng::seed_from_u64(sample));

        for round in 0..num_rounds {
            // Create fresh noise model for this round
            let noise = make_noise(p_error);
            runner = runner.with_noise(noise);

            // Build circuit for this round - need to reset simulator for rounds after first
            let circuit = if round == 0 {
                CommandBuilder::new().pz(0).pz(1).cx(0, 1).mz(1).build()
            } else {
                syndrome_circuit()
            };

            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &circuit).unwrap();

            if outcomes.get_bit(QubitId(1)).unwrap_or(false) {
                syndrome_detections += 1;
            }
        }

        if syndrome_detections >= failure_threshold {
            direct_failures += 1;
        }
    }

    let direct_mc_probability = f64::from(direct_failures) / num_samples as f64;

    // Now demonstrate ECS-based trajectory tracking with World
    let mut world: World<SparseStab> = World::new(12345);

    // Spawn entities (trajectories)
    for _ in 0..num_samples {
        world.spawn_with_simulator(SparseStab::new(2));
    }

    // Track syndrome detections per entity
    let mut syndrome_counts: std::collections::BTreeMap<u64, usize> =
        std::collections::BTreeMap::new();
    for entity in world.active_entities() {
        syndrome_counts.insert(entity.0, 0);
    }

    let mut level_probs: Vec<f64> = Vec::new();
    let initial_weight = world.total_weight();

    // Run rounds and track statistics
    for round in 0..num_rounds {
        for entity in world.active_entities() {
            // Get a unique seed for this entity/round combination
            let seed = world.base_seed() + entity.0 * 1000 + round as u64;

            if let Some(sim_comp) = world.simulators.get_mut(entity) {
                // Run circuit directly on the simulator
                let noise = make_noise(p_error);
                let mut runner = CircuitRunner::<SparseStab>::new()
                    .with_noise(noise)
                    .with_rng(PecosRng::seed_from_u64(seed));

                let circuit = if round == 0 {
                    CommandBuilder::new().pz(0).pz(1).cx(0, 1).mz(1).build()
                } else {
                    syndrome_circuit()
                };

                sim_comp.simulator.reset();
                let outcomes = runner
                    .apply_circuit(&mut sim_comp.simulator, &circuit)
                    .unwrap();

                // Track syndrome detection
                if outcomes.get_bit(QubitId(1)).unwrap_or(false)
                    && let Some(count) = syndrome_counts.get_mut(&entity.0)
                {
                    *count += 1;
                }
            }
        }

        // Compute level statistics
        let total_weight = world.total_weight();
        let threshold = round;

        let weight_above: f64 = world
            .active_entities()
            .iter()
            .filter(|e| syndrome_counts.get(&e.0).copied().unwrap_or(0) > threshold)
            .map(|e| world.weights.get(*e).map_or(1.0, |w| w.weight.weight()))
            .sum();

        let p_level = if total_weight > 0.0 {
            weight_above / total_weight
        } else {
            0.0
        };
        level_probs.push(p_level);
    }

    // Verify weight preservation
    let final_weight = world.total_weight();
    assert!(
        (initial_weight - final_weight).abs() < 1e-10,
        "Weight should be preserved: {initial_weight} -> {final_weight}"
    );

    // Count failures using ECS tracking
    let failures = world
        .active_entities()
        .iter()
        .filter(|e| syndrome_counts.get(&e.0).copied().unwrap_or(0) >= failure_threshold)
        .count();

    let ecs_probability = failures as f64 / num_samples as f64;

    // Report results
    println!("\n=== ECS-Based Trajectory Tracking Results ===");
    println!("Configuration:");
    println!("  Physical error rate: {p_error}");
    println!("  Rounds: {num_rounds}");
    println!("  Samples: {num_samples}");
    println!("  Failure threshold: {failure_threshold} syndrome detections");
    println!();
    println!("Level statistics:");
    for (i, p) in level_probs.iter().enumerate() {
        println!("  Round {i}: P(syndrome > {i}) = {p:.4}");
    }
    println!();
    println!("Results:");
    println!("  Direct Monte Carlo: P(failure) = {direct_mc_probability:.4}");
    println!("  ECS tracking:       P(failure) = {ecs_probability:.4}");

    // Test entity splitting (key for subset simulation)
    let test_entity = world.active_entities()[0];
    let clones = world.split_entity(test_entity, 4);
    assert_eq!(clones.len(), 3, "Should create 3 clones + original = 4");

    // Verify weight preservation after split
    let weight_after_split = world.total_weight();
    assert!(
        (initial_weight - weight_after_split).abs() < 1e-10,
        "Total weight should be preserved after split"
    );

    println!("  Entity splitting: OK (4 copies, weight preserved)");
}

/// Test the Bernoulli simulation validation framework.
///
/// This verifies that the Monte Carlo probability estimates match the
/// analytical binomial distribution. Note: `BernoulliSubsetSimulation::run()`
/// uses direct Monte Carlo (not subset simulation) for validation purposes.
#[test]
fn test_bernoulli_subset_validation() {
    use pecos_neo::sampling::SubsetConfig;
    use pecos_neo::sampling::subset::BernoulliSubsetSimulation;

    // Moderate probability case (easier to validate with smaller samples)
    let sim = BernoulliSubsetSimulation::new(
        0.1,  // 10% damage per step
        50,   // 50 steps
        10.0, // Failure if damage >= 10
    )
    .with_config(
        SubsetConfig::new()
            .with_samples_per_level(5000)
            .with_seed(42),
    );

    let analytical = sim.analytical_probability();
    let direct_mc = sim.run_direct_mc(10000, 12345);

    println!("\n=== Bernoulli Validation ===");
    println!("Parameters: p=0.1, n=50, threshold=10");
    println!("  Analytical P(failure): {analytical:.6}");
    println!("  Direct MC P(failure):  {direct_mc:.6}");
    println!(
        "  Relative error: {:.2}%",
        ((analytical - direct_mc).abs() / analytical) * 100.0
    );

    // Direct MC should be within 10% of analytical (with these sample sizes)
    let rel_error = (analytical - direct_mc).abs() / analytical;
    assert!(
        rel_error < 0.15,
        "Direct MC should match analytical: {direct_mc} vs {analytical}"
    );

    // The direct MC result via run() (returns SubsetResult for API consistency)
    let result = sim.run();
    println!("  MC result (run()):     {:.6}", result.probability());

    // Direct MC via run() should also be close to analytical
    // (Using relaxed tolerance since sample size is smaller)
    let mc_rel_error = (analytical - result.probability()).abs() / analytical;
    assert!(
        mc_rel_error < 0.25,
        "Direct MC should be close to analytical: {} vs {}",
        result.probability(),
        analytical
    );
}

/// Test `SubsetSimulation` with noise integration.
///
/// This verifies that the `with_noise_builder` API works correctly
/// by running a simple circuit with depolarizing noise and measurement error.
#[test]
fn test_subset_simulation_with_noise() {
    use pecos_core::QubitId;
    use pecos_neo::command::CommandBuilder;
    use pecos_neo::noise::ComposableNoiseModel;
    use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
    use pecos_neo::outcome::MeasurementOutcomes;
    use pecos_neo::sampling::subset::{SubsetConfig, SubsetSimulation};

    // Build a circuit with identity gates that can accumulate errors.
    // Without noise, this always measures 0.
    let circuit = CommandBuilder::new()
        .pz(0)
        .identity(0) // Gate that can have depolarizing error
        .identity(0)
        .identity(0)
        .mz(0)
        .build();

    // Score function: 1.0 if measured 1 (error), 0.0 if measured 0 (correct)
    let qubit = QubitId(0);
    let score_fn = move |outcomes: &MeasurementOutcomes| {
        outcomes
            .get_bit(qubit)
            .map_or(0.0, |b| if b { 1.0 } else { 0.0 })
    };

    // Failure predicate: fails if we measure 1 (which would indicate an error)
    let is_failure_fn =
        move |outcomes: &MeasurementOutcomes| outcomes.get_bit(qubit).unwrap_or(false);

    // Create noise builder function that returns noise with measurement error.
    // Measurement error is guaranteed to affect the result.
    let noise_builder = || -> Option<ComposableNoiseModel> {
        Some(
            CompositeNoiseModelBuilder::new()
                .with_p1(0.05) // 5% single-qubit gate error
                .with_p_meas(0.01, 0.01) // 1% measurement error (symmetric)
                .build(),
        )
    };

    let config = SubsetConfig::new()
        .with_samples_per_level(1000)
        .with_seed(42);

    let sim = SubsetSimulation::new(circuit, 1, score_fn, is_failure_fn)
        .with_noise_builder(noise_builder)
        .with_config(config);

    let result = sim.run();

    println!("\n=== SubsetSimulation with Noise Test ===");
    println!("  Probability: {:.6}", result.probability());
    println!("  Samples:     {}", result.total_samples);
    println!("  Failures:    {}", result.direct_failures);

    // Verify the simulation runs and produces a result.
    // With 1% measurement error and 5% gate errors, we should see some failures.
    assert!(result.total_samples > 0, "Should have run some samples");

    // The probability should be non-zero if noise is being applied correctly.
    // With 3 identity gates at 5% error and 1% measurement error,
    // we expect a small but non-zero error rate.
    assert!(
        result.direct_failures > 0,
        "Expected some failures from noise, got 0. Noise model may not be applied correctly."
    );
    assert!(
        result.probability() > 0.0,
        "Expected non-zero probability from noise-induced errors"
    );
}
