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

//! Parallel coordinator for ECS-based quantum simulation.
//!
//! The coordinator manages parallel execution across multiple workers, where each
//! worker owns its own [`World`] containing simulation entities. This enables:
//!
//! - **Standard Monte Carlo**: Embarrassingly parallel shot execution
//! - **Rare Event Simulation**: Periodic synchronization for splitting/pruning
//! - **Adaptive Load Balancing**: Entity redistribution between workers (future)
//!
//! ## Architecture
//!
//! ```text
//! ParallelCoordinator
//!   ├── Worker 0
//!   │   └── World (entities, components, resources)
//!   ├── Worker 1
//!   │   └── World (entities, components, resources)
//!   └── Worker N
//!       └── World (entities, components, resources)
//! ```
//!
//! ## Seed Hierarchy
//!
//! Seeds are derived hierarchically for determinism and independence:
//!
//! ```text
//! coordinator.base_seed
//! └── worker_{id}
//!     └── entity_{id}
//!         ├── noise
//!         └── simulator
//! ```
//!
//! ## Example
//!
//! ```no_run
//! use pecos_neo::ecs::{ParallelCoordinator, ParallelConfig};
//! use pecos_simulators::SparseStab;
//!
//! let config = ParallelConfig::new()
//!     .with_workers(4)
//!     .with_entities_per_worker(100)
//!     .with_seed(42);
//!
//! let coordinator: ParallelCoordinator<SparseStab> = ParallelCoordinator::new(config);
//!
//! // Run simulation with a step function
//! let results = coordinator.run(
//!     || SparseStab::new(1),
//!     |world| {
//!         // Execute one step of simulation on each entity
//!         world.entities().map(|e| e.0).collect()
//!     },
//! );
//! ```

use super::component::StatusComponent;
use super::world::World;
use pecos_core::rng::rng_manageable::{RngManageable, derive_seed};
use pecos_random::PecosRng;
use pecos_simulators::CliffordGateable;
use rayon::prelude::*;
use std::sync::{Arc, Mutex};

/// Configuration for parallel simulation.
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Number of worker threads.
    pub num_workers: usize,
    /// Number of entities (shots) per worker.
    pub entities_per_worker: usize,
    /// Base seed for deterministic execution.
    pub seed: u64,
    /// Synchronization interval for rare event simulation (None = no sync).
    pub sync_interval: Option<usize>,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            num_workers: num_cpus::get().max(1),
            entities_per_worker: 100,
            seed: 0,
            sync_interval: None,
        }
    }
}

impl ParallelConfig {
    /// Create a new configuration with default values.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the number of workers.
    #[must_use]
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.num_workers = workers.max(1);
        self
    }

    /// Set the number of entities per worker.
    #[must_use]
    pub fn with_entities_per_worker(mut self, count: usize) -> Self {
        self.entities_per_worker = count;
        self
    }

    /// Set the base seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the synchronization interval for rare event simulation.
    ///
    /// When set, workers will synchronize after every `interval` steps
    /// to allow for splitting, pruning, and redistribution.
    #[must_use]
    pub fn with_sync_interval(mut self, interval: usize) -> Self {
        self.sync_interval = Some(interval);
        self
    }

    /// Total number of entities across all workers.
    #[must_use]
    pub fn total_entities(&self) -> usize {
        self.num_workers * self.entities_per_worker
    }
}

/// Statistics from parallel execution.
#[derive(Debug, Clone, Default)]
pub struct ExecutionStats {
    /// Total entities processed.
    pub total_entities: usize,
    /// Entities that completed successfully.
    pub completed: usize,
    /// Entities that failed.
    pub failed: usize,
    /// Entities that were pruned.
    pub pruned: usize,
    /// Number of synchronization points executed.
    pub sync_points: usize,
}

impl ExecutionStats {
    /// Create new empty stats.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Merge stats from another execution.
    pub fn merge(&mut self, other: &Self) {
        self.total_entities += other.total_entities;
        self.completed += other.completed;
        self.failed += other.failed;
        self.pruned += other.pruned;
        self.sync_points = self.sync_points.max(other.sync_points);
    }
}

/// Per-worker state containing a World and local statistics.
pub struct WorkerState<S: CliffordGateable> {
    /// The worker's world containing entities.
    pub world: World<S>,
    /// Worker ID for seed derivation.
    pub worker_id: usize,
    /// Local statistics.
    pub stats: ExecutionStats,
}

impl<S: CliffordGateable> WorkerState<S> {
    /// Create a new worker state with derived seed.
    #[must_use]
    pub fn new(worker_id: usize, base_seed: u64) -> Self {
        let worker_seed = derive_seed(base_seed, &format!("worker_{worker_id}"));
        Self {
            world: World::new(worker_seed),
            worker_id,
            stats: ExecutionStats::new(),
        }
    }

    /// Get the number of active entities.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.world.active_entities().len()
    }

    /// Collect statistics from the world.
    pub fn collect_stats(&mut self) {
        self.stats.total_entities = self.world.entity_count();
        self.stats.completed = self
            .world
            .entities_with_status(StatusComponent::Complete)
            .len();
        self.stats.failed = self
            .world
            .entities_with_status(StatusComponent::Failed)
            .len();
        self.stats.pruned = self
            .world
            .entities_with_status(StatusComponent::Pruned)
            .len();
    }
}

/// Aggregated results from parallel execution.
#[derive(Debug, Clone)]
pub struct ParallelResult<T> {
    /// Results from each entity, in deterministic order.
    pub results: Vec<T>,
    /// Execution statistics.
    pub stats: ExecutionStats,
    /// Seed used for the execution.
    pub seed: u64,
}

impl<T> ParallelResult<T> {
    /// Create new results.
    #[must_use]
    pub fn new(results: Vec<T>, stats: ExecutionStats, seed: u64) -> Self {
        Self {
            results,
            stats,
            seed,
        }
    }

    /// Get the number of results.
    #[must_use]
    pub fn len(&self) -> usize {
        self.results.len()
    }

    /// Check if results are empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.results.is_empty()
    }

    /// Iterate over results.
    pub fn iter(&self) -> impl Iterator<Item = &T> {
        self.results.iter()
    }
}

impl<T> IntoIterator for ParallelResult<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.results.into_iter()
    }
}

/// Parallel coordinator for ECS-based quantum simulation.
///
/// Manages multiple workers, each with its own [`World`] containing
/// simulation entities. Supports both standard Monte Carlo (embarrassingly
/// parallel) and rare event simulation (periodic synchronization).
pub struct ParallelCoordinator<S: CliffordGateable> {
    config: ParallelConfig,
    /// Marker for simulator type (workers are created on demand).
    _marker: std::marker::PhantomData<S>,
}

impl<S: CliffordGateable + Clone + Send + Sync> ParallelCoordinator<S> {
    /// Create a new coordinator with the given configuration.
    #[must_use]
    pub fn new(config: ParallelConfig) -> Self {
        Self {
            config,
            _marker: std::marker::PhantomData,
        }
    }

    /// Get the configuration.
    #[must_use]
    pub fn config(&self) -> &ParallelConfig {
        &self.config
    }

    /// Run a parallel simulation with a step function.
    ///
    /// The step function is called for each worker's world and should execute
    /// one shot of simulation, returning a result for each active entity.
    ///
    /// # Arguments
    /// * `make_simulator` - Factory function to create simulators
    /// * `step` - Function to execute one simulation step, returns results
    ///
    /// # Returns
    /// Aggregated results from all entities in deterministic order.
    ///
    /// # Panics
    /// Panics if an internal mutex is poisoned.
    pub fn run<F, G, T>(&self, make_simulator: F, step: G) -> ParallelResult<T>
    where
        F: Fn() -> S + Send + Sync,
        G: Fn(&mut World<S>) -> Vec<T> + Send + Sync,
        T: Send,
        S: RngManageable<Rng = PecosRng>,
    {
        // Collect results with (worker_id, entity_idx, result) for ordering
        let results: Arc<Mutex<Vec<(usize, usize, T)>>> =
            Arc::new(Mutex::new(Vec::with_capacity(self.config.total_entities())));
        let stats: Arc<Mutex<ExecutionStats>> = Arc::new(Mutex::new(ExecutionStats::new()));

        // Execute in parallel
        (0..self.config.num_workers)
            .into_par_iter()
            .for_each(|worker_id| {
                let mut worker = WorkerState::<S>::new(worker_id, self.config.seed);

                // Spawn entities with simulators
                for _ in 0..self.config.entities_per_worker {
                    let sim = make_simulator();
                    worker.world.spawn_with_full_seeding(sim);
                }

                // Execute step function
                let worker_results = step(&mut worker.world);

                // Collect stats
                worker.collect_stats();

                // Store results with ordering info
                let mut all_results = results.lock().expect("results lock poisoned");
                for (entity_idx, result) in worker_results.into_iter().enumerate() {
                    all_results.push((worker_id, entity_idx, result));
                }

                // Merge stats
                let mut all_stats = stats.lock().expect("stats lock poisoned");
                all_stats.merge(&worker.stats);
            });

        // Sort results by (worker_id, entity_idx) for deterministic ordering
        let mut sorted_results = results.lock().expect("results lock poisoned");
        sorted_results.sort_by(|(w1, e1, _), (w2, e2, _)| w1.cmp(w2).then(e1.cmp(e2)));

        let final_results: Vec<T> = sorted_results.drain(..).map(|(_, _, r)| r).collect();
        drop(sorted_results);

        let final_stats = stats.lock().expect("stats lock poisoned").clone();

        ParallelResult::new(final_results, final_stats, self.config.seed)
    }

    /// Run a parallel simulation with synchronization points for rare event simulation.
    ///
    /// This method executes the step function multiple times, synchronizing between
    /// workers at intervals to allow for:
    /// - Weight-based pruning of low-probability trajectories
    /// - Splitting of high-importance trajectories
    /// - Entity redistribution for load balancing
    ///
    /// # Arguments
    /// * `make_simulator` - Factory function to create simulators
    /// * `num_steps` - Total number of steps to execute
    /// * `step` - Function to execute one step on a world
    /// * `on_sync` - Called at each synchronization point with all worker states
    ///
    /// # Returns
    /// Final results from all active entities.
    pub fn run_with_sync<F, G, H, T>(
        &self,
        make_simulator: F,
        num_steps: usize,
        step: G,
        mut on_sync: H,
    ) -> ParallelResult<T>
    where
        F: Fn() -> S + Send + Sync,
        G: Fn(&mut World<S>, usize) + Send + Sync,
        H: FnMut(&mut [WorkerState<S>], usize),
        T: Send + Clone,
        S: RngManageable<Rng = PecosRng>,
    {
        let sync_interval = self.config.sync_interval.unwrap_or(num_steps);
        let mut sync_points = 0;

        // Create workers
        let mut workers: Vec<WorkerState<S>> = (0..self.config.num_workers)
            .map(|worker_id| {
                let mut worker = WorkerState::<S>::new(worker_id, self.config.seed);
                // Spawn entities
                for _ in 0..self.config.entities_per_worker {
                    let sim = make_simulator();
                    worker.world.spawn_with_full_seeding(sim);
                }
                worker
            })
            .collect();

        // Execute steps with periodic synchronization
        let mut current_step = 0;
        while current_step < num_steps {
            let steps_until_sync = sync_interval.min(num_steps - current_step);

            // Parallel execution phase
            workers.par_iter_mut().for_each(|worker| {
                for step_offset in 0..steps_until_sync {
                    step(&mut worker.world, current_step + step_offset);
                }
            });

            current_step += steps_until_sync;

            // Synchronization phase (if not at the end)
            if current_step < num_steps {
                on_sync(&mut workers, current_step);
                sync_points += 1;
            }
        }

        // Collect final stats
        let mut final_stats = ExecutionStats::new();
        final_stats.sync_points = sync_points;

        for worker in &mut workers {
            worker.collect_stats();
            final_stats.merge(&worker.stats);
        }

        // Note: This returns empty results - actual result collection should be
        // done via the step function or on_sync callback for rare event simulation
        ParallelResult::new(Vec::new(), final_stats, self.config.seed)
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)] // statistical tests use count as f64
mod tests {
    use super::*;
    use crate::command::CommandBuilder;
    use crate::runner::CircuitRunner;
    use pecos_core::QubitId;
    use pecos_simulators::SparseStab;

    #[test]
    fn test_parallel_config_defaults() {
        let config = ParallelConfig::new();
        assert!(config.num_workers >= 1);
        assert_eq!(config.entities_per_worker, 100);
        assert_eq!(config.seed, 0);
        assert!(config.sync_interval.is_none());
    }

    #[test]
    fn test_parallel_config_builder() {
        let config = ParallelConfig::new()
            .with_workers(4)
            .with_entities_per_worker(50)
            .with_seed(42)
            .with_sync_interval(10);

        assert_eq!(config.num_workers, 4);
        assert_eq!(config.entities_per_worker, 50);
        assert_eq!(config.seed, 42);
        assert_eq!(config.sync_interval, Some(10));
        assert_eq!(config.total_entities(), 200);
    }

    #[test]
    fn test_worker_state_creation() {
        let worker: WorkerState<SparseStab> = WorkerState::new(0, 42);
        assert_eq!(worker.worker_id, 0);
        assert_eq!(worker.world.entity_count(), 0);
    }

    #[test]
    fn test_worker_state_with_entities() {
        let mut worker: WorkerState<SparseStab> = WorkerState::new(0, 42);

        worker.world.spawn_with_simulator(SparseStab::new(2));
        worker.world.spawn_with_simulator(SparseStab::new(2));

        assert_eq!(worker.active_count(), 2);

        worker.collect_stats();
        assert_eq!(worker.stats.total_entities, 2);
    }

    #[test]
    fn test_execution_stats_merge() {
        let mut stats1 = ExecutionStats::new();
        stats1.total_entities = 10;
        stats1.completed = 8;
        stats1.failed = 1;
        stats1.pruned = 1;

        let mut stats2 = ExecutionStats::new();
        stats2.total_entities = 5;
        stats2.completed = 4;
        stats2.failed = 0;
        stats2.pruned = 1;

        stats1.merge(&stats2);

        assert_eq!(stats1.total_entities, 15);
        assert_eq!(stats1.completed, 12);
        assert_eq!(stats1.failed, 1);
        assert_eq!(stats1.pruned, 2);
    }

    #[test]
    fn test_parallel_coordinator_basic() {
        let config = ParallelConfig::new()
            .with_workers(2)
            .with_entities_per_worker(10)
            .with_seed(42);

        let coordinator: ParallelCoordinator<SparseStab> = ParallelCoordinator::new(config);

        // Simple test: return entity count from each world
        let results = coordinator.run(
            || SparseStab::new(1),
            |world| {
                // Just return the entity IDs as results
                world.entities().map(|e| e.0).collect()
            },
        );

        // Should have 20 results (2 workers * 10 entities)
        assert_eq!(results.len(), 20);
        assert_eq!(results.stats.total_entities, 20);
    }

    #[test]
    fn test_parallel_coordinator_determinism() {
        let config = ParallelConfig::new()
            .with_workers(2)
            .with_entities_per_worker(5)
            .with_seed(42);

        // Run twice with same config
        let coordinator1: ParallelCoordinator<SparseStab> =
            ParallelCoordinator::new(config.clone());
        let coordinator2: ParallelCoordinator<SparseStab> = ParallelCoordinator::new(config);

        let results1 = coordinator1.run(
            || SparseStab::new(1),
            |world| world.entities().map(|e| world.base_seed() + e.0).collect(),
        );

        let results2 = coordinator2.run(
            || SparseStab::new(1),
            |world| world.entities().map(|e| world.base_seed() + e.0).collect(),
        );

        // Results should be identical
        assert_eq!(results1.results, results2.results);
    }

    #[test]
    fn test_parallel_coordinator_with_simulation() {
        // Test running actual quantum simulation through the coordinator
        let config = ParallelConfig::new()
            .with_workers(2)
            .with_entities_per_worker(10)
            .with_seed(42);

        let coordinator: ParallelCoordinator<SparseStab> = ParallelCoordinator::new(config);

        let commands = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();

        let results = coordinator.run(
            || SparseStab::new(1),
            |world| {
                // Run one shot per entity
                world
                    .entities()
                    .map(|entity| {
                        // Get simulator and RNG
                        let sim_comp = world
                            .simulators
                            .get(entity)
                            .expect("entity must have simulator");
                        let rng_comp = world.rngs.get(entity).expect("entity must have rng");

                        // Create a runner with the entity's components
                        let mut sim = sim_comp.simulator.clone();
                        let mut runner =
                            CircuitRunner::<SparseStab>::new().with_rng(rng_comp.rng.clone());

                        let outcomes = runner
                            .apply_circuit(&mut sim, &commands)
                            .expect("circuit execution failed");
                        outcomes.get_bit(QubitId(0)).unwrap_or(false)
                    })
                    .collect()
            },
        );

        // Should have 20 results
        assert_eq!(results.len(), 20);

        // With H gate, should have roughly 50/50 distribution
        let ones = results.iter().filter(|&&b| b).count();
        let rate = ones as f64 / results.len() as f64;

        // Allow for statistical variation (should be ~0.5)
        assert!(
            rate > 0.2 && rate < 0.8,
            "H gate should give roughly 50/50, got {rate:.2}"
        );
    }

    #[test]
    fn test_parallel_result_iteration() {
        let stats = ExecutionStats::new();
        let result = ParallelResult::new(vec![1, 2, 3, 4, 5], stats, 42);

        assert_eq!(result.len(), 5);
        assert!(!result.is_empty());

        let sum: i32 = result.iter().sum();
        assert_eq!(sum, 15);

        let sum: i32 = result.into_iter().sum();
        assert_eq!(sum, 15);
    }

    #[test]
    fn test_run_with_sync_basic() {
        let config = ParallelConfig::new()
            .with_workers(2)
            .with_entities_per_worker(5)
            .with_seed(42)
            .with_sync_interval(2);

        let coordinator: ParallelCoordinator<SparseStab> = ParallelCoordinator::new(config);

        let mut sync_count = 0;

        let result: ParallelResult<()> = coordinator.run_with_sync(
            || SparseStab::new(1),
            5, // 5 steps
            |_world, _step| {
                // Do nothing per step
            },
            |_workers, _step| {
                sync_count += 1;
            },
        );

        // With sync_interval=2 and 5 steps:
        // Steps 0,1 -> sync at 2
        // Steps 2,3 -> sync at 4
        // Step 4 -> end (no sync at end)
        assert_eq!(sync_count, 2);
        assert_eq!(result.stats.sync_points, 2);
    }
}
