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

//! Entity redistribution for parallel rare event simulation.
//!
//! This module provides helpers for redistributing entities across workers
//! at synchronization points. This is essential for:
//!
//! - **Load balancing**: After splitting, some workers may have more entities
//! - **Global resampling**: Making resampling decisions based on all entities
//! - **Weight-based pruning**: Removing low-weight entities globally
//!
//! ## Usage
//!
//! ```no_run
//! use pecos_neo::ecs::{ParallelCoordinator, ParallelConfig, redistribute_by_weight};
//! use pecos_simulators::SparseStab;
//! use pecos_random::PecosRng;
//! use rand_core::SeedableRng;
//!
//! let config = ParallelConfig::new().with_workers(2).with_seed(42);
//! let coordinator: ParallelCoordinator<SparseStab> = ParallelCoordinator::new(config);
//! let mut rng = PecosRng::seed_from_u64(42);
//! let target_per_worker = 10;
//!
//! coordinator.run_with_sync::<_, _, _, ()>(
//!     || SparseStab::new(1),
//!     5,
//!     |_world, _step| {},
//!     |workers, _step| {
//!         // Redistribute entities based on weights
//!         redistribute_by_weight(workers, target_per_worker, &mut rng);
//!     },
//! );
//! ```

use super::coordinator::WorkerState;
use super::entity::EntityId;
use super::world::EntityTransfer;
use crate::sampling::SampleWeight;
use pecos_random::PecosRng;
use pecos_simulators::CliffordGateable;
use rand::RngExt;

/// Statistics from a redistribution operation.
#[derive(Debug, Clone, Default)]
pub struct RedistributionStats {
    /// Total entities before redistribution.
    pub entities_before: usize,
    /// Total entities after redistribution.
    pub entities_after: usize,
    /// Entities transferred between workers.
    pub transfers: usize,
    /// Total weight before redistribution.
    pub weight_before: f64,
    /// Total weight after redistribution.
    pub weight_after: f64,
}

impl RedistributionStats {
    /// Check if weight was preserved within tolerance.
    #[must_use]
    pub fn weight_preserved(&self, tolerance: f64) -> bool {
        (self.weight_before - self.weight_after).abs() < tolerance
    }
}

/// Collect all entity weights from all workers.
///
/// Returns a vector of (`worker_idx`, `entity_id`, weight) tuples.
#[must_use]
pub fn collect_weights<S: CliffordGateable>(
    workers: &[WorkerState<S>],
) -> Vec<(usize, EntityId, f64)> {
    let mut weights = Vec::new();
    for (worker_idx, worker) in workers.iter().enumerate() {
        for entity in worker.world.active_entities() {
            let weight = worker
                .world
                .weights
                .get(entity)
                .map_or(1.0, |w| w.weight.weight());
            weights.push((worker_idx, entity, weight));
        }
    }
    weights
}

/// Compute the total weight across all workers.
#[must_use]
pub fn total_weight<S: CliffordGateable>(workers: &[WorkerState<S>]) -> f64 {
    workers.iter().map(|w| w.world.total_weight()).sum()
}

/// Redistribute entities across workers using multinomial resampling.
///
/// This performs global resampling based on entity weights:
/// 1. Collects all active entities and their weights
/// 2. Samples `target_per_worker * num_workers` entities with replacement
/// 3. Distributes the selected entities evenly across workers
///
/// After redistribution:
/// - Each worker has approximately `target_per_worker` entities
/// - Each entity has weight `total_weight / total_entities`
/// - Total weight is preserved
///
/// Returns statistics about the redistribution.
pub fn redistribute_by_weight<S: CliffordGateable + Clone>(
    workers: &mut [WorkerState<S>],
    target_per_worker: usize,
    rng: &mut PecosRng,
) -> RedistributionStats {
    let num_workers = workers.len();
    if num_workers == 0 || target_per_worker == 0 {
        return RedistributionStats::default();
    }

    let total_target = target_per_worker * num_workers;

    // Collect all entities and weights
    let entity_info = collect_weights(workers);
    if entity_info.is_empty() {
        return RedistributionStats::default();
    }

    let entities_before = entity_info.len();
    let weight_before: f64 = entity_info.iter().map(|(_, _, w)| w).sum();

    // Build CDF for weighted sampling
    let total_w: f64 = entity_info.iter().map(|(_, _, w)| w).sum();
    if total_w <= 0.0 {
        return RedistributionStats::default();
    }

    let mut cdf = Vec::with_capacity(entity_info.len());
    let mut cumsum = 0.0;
    for (_, _, w) in &entity_info {
        cumsum += w / total_w;
        cdf.push(cumsum);
    }

    // Sample entities with replacement
    let mut selection_counts = vec![0usize; entity_info.len()];
    for _ in 0..total_target {
        let u: f64 = rng.random();
        let idx = cdf.partition_point(|&x| x < u).min(cdf.len() - 1);
        selection_counts[idx] += 1;
    }

    // Extract ALL entities from all workers first (this removes them)
    // We'll only re-import the ones that were selected
    let mut extracted: Vec<Option<EntityTransfer<S>>> = Vec::with_capacity(entity_info.len());
    for (worker_idx, entity, _) in &entity_info {
        extracted.push(workers[*worker_idx].world.extract_entity(*entity));
    }

    // Build the list of transfers (only entities with count > 0)
    let mut transfers: Vec<(EntityTransfer<S>, usize)> = Vec::new();
    let mut transfer_count = 0;

    for (idx, &count) in selection_counts.iter().enumerate() {
        if count == 0 {
            // This entity was not selected - it's already been removed, so it's pruned
            continue;
        }

        if let Some(transfer) = extracted[idx].take() {
            transfers.push((transfer, count));
            transfer_count += count;
        }
    }

    // Distribute extracted entities across workers
    // Each entity gets weight = total_weight / total_entities
    let new_weight = SampleWeight::from_linear(weight_before / total_target as f64);

    let mut worker_idx = 0;
    let mut entities_in_current_worker = 0;

    for (mut transfer, count) in transfers {
        // Update weight for the transfer
        transfer.weight = new_weight;

        for _ in 0..count {
            // Clone the transfer for each copy
            let t = transfer.clone();

            // Import into current worker
            workers[worker_idx].world.import_entity(t);
            entities_in_current_worker += 1;

            // Move to next worker if this one is full
            if entities_in_current_worker >= target_per_worker && worker_idx + 1 < num_workers {
                worker_idx += 1;
                entities_in_current_worker = 0;
            }
        }
    }

    // Count entities after redistribution
    let entities_after: usize = workers
        .iter()
        .map(|w| w.world.active_entities().len())
        .sum();
    let weight_after = total_weight(workers);

    RedistributionStats {
        entities_before,
        entities_after,
        transfers: transfer_count,
        weight_before,
        weight_after,
    }
}

/// Balance entity counts across workers without resampling.
///
/// This moves entities from workers with more than average to workers
/// with fewer than average, without changing weights or sampling.
///
/// Use this for simple load balancing after splitting operations.
pub fn balance_entity_counts<S: CliffordGateable + Clone>(workers: &mut [WorkerState<S>]) -> usize {
    let num_workers = workers.len();
    if num_workers <= 1 {
        return 0;
    }

    // Count entities per worker
    let counts: Vec<usize> = workers
        .iter()
        .map(|w| w.world.active_entities().len())
        .collect();
    let total: usize = counts.iter().sum();
    let target = total / num_workers;

    let mut transfers = 0;

    // Collect excess entities from workers with too many
    let mut excess: Vec<EntityTransfer<S>> = Vec::new();
    for (worker_idx, &count) in counts.iter().enumerate() {
        if count > target + 1 {
            // Take excess entities
            let to_take = count - target - 1;
            let active = workers[worker_idx].world.active_entities();
            let entities_to_take: Vec<EntityId> = active.into_iter().take(to_take).collect();

            for entity in entities_to_take {
                if let Some(transfer) = workers[worker_idx].world.extract_entity(entity) {
                    excess.push(transfer);
                }
            }
        }
    }

    // Distribute excess to workers with too few
    let mut excess_iter = excess.into_iter();
    for (worker_idx, &count) in counts.iter().enumerate() {
        if count < target {
            let needed = target - count;
            for _ in 0..needed {
                if let Some(transfer) = excess_iter.next() {
                    workers[worker_idx].world.import_entity(transfer);
                    transfers += 1;
                }
            }
        }
    }

    // Any remaining excess goes to the last worker
    for transfer in excess_iter {
        workers[num_workers - 1].world.import_entity(transfer);
        transfers += 1;
    }

    transfers
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_simulators::SparseStab;

    #[test]
    fn test_collect_weights() {
        let mut worker0: WorkerState<SparseStab> = WorkerState::new(0, 42);
        let mut worker1: WorkerState<SparseStab> = WorkerState::new(1, 42);

        worker0.world.spawn_with_simulator(SparseStab::new(1));
        worker0.world.spawn_with_simulator(SparseStab::new(1));
        worker1.world.spawn_with_simulator(SparseStab::new(1));

        let workers = vec![worker0, worker1];
        let weights = collect_weights(&workers);

        assert_eq!(weights.len(), 3);
        // Default weight is 1.0
        for (_, _, w) in &weights {
            assert!((w - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_redistribute_preserves_weight() {
        let mut workers: Vec<WorkerState<SparseStab>> = (0..2)
            .map(|id| {
                let mut worker = WorkerState::new(id, 42);
                // Each worker gets 5 entities
                for _ in 0..5 {
                    worker.world.spawn_with_simulator(SparseStab::new(1));
                }
                worker
            })
            .collect();

        // Set different weights
        for worker in &mut workers {
            for (i, entity) in worker.world.active_entities().into_iter().enumerate() {
                if let Some(w) = worker.world.weights.get_mut(entity) {
                    w.weight = SampleWeight::from_linear((i + 1) as f64);
                }
            }
        }

        let mut rng = PecosRng::seed_from_u64(42);
        let stats = redistribute_by_weight(&mut workers, 5, &mut rng);

        // Weight should be preserved
        assert!(
            stats.weight_preserved(0.1),
            "Weight not preserved: before={}, after={}",
            stats.weight_before,
            stats.weight_after
        );

        // Should have target entities per worker
        assert_eq!(stats.entities_after, 10);
    }

    #[test]
    fn test_balance_entity_counts() {
        let mut workers: Vec<WorkerState<SparseStab>> = vec![
            WorkerState::new(0, 42),
            WorkerState::new(1, 42),
            WorkerState::new(2, 42),
        ];

        // Worker 0 gets 10 entities, others get 1 each
        for _ in 0..10 {
            workers[0].world.spawn_with_simulator(SparseStab::new(1));
        }
        workers[1].world.spawn_with_simulator(SparseStab::new(1));
        workers[2].world.spawn_with_simulator(SparseStab::new(1));

        let transfers = balance_entity_counts(&mut workers);

        // Should have transferred some entities
        assert!(transfers > 0);

        // Counts should be more balanced now
        let counts: Vec<usize> = workers
            .iter()
            .map(|w| w.world.active_entities().len())
            .collect();

        // All workers should have at least 3 entities (12 total / 3 = 4 target)
        for count in &counts {
            assert!(*count >= 3, "Worker has only {count} entities");
        }
    }
}
