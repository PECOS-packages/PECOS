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

//! Lightweight ECS-inspired infrastructure for quantum simulation.
//!
//! This module provides a data-oriented foundation for managing populations of
//! simulation instances, particularly useful for:
//!
//! - **Splitting/Cloning**: Rare event simulation where trajectories branch
//! - **Subset Simulation**: Populations that resample at level crossings
//! - **Branching Programs**: Multiple paths through a QEC program graph
//!
//! ## Design Philosophy
//!
//! This is *not* a full ECS framework like Bevy. Instead, it's a lightweight,
//! quantum-simulation-focused approach that borrows key ideas:
//!
//! 1. **Entities** are just IDs (cheap to create, copy, compare)
//! 2. **Components** are plain data stored separately (Structure of Arrays)
//! 3. **Resources** are shared state (seed, program, noise config)
//! 4. **Systems** are functions that operate on the world (user-defined)
//!
//! ## Determinism
//!
//! All data structures use deterministic ordering (`BTreeMap`, `BTreeSet`).
//! Seed derivation is centralized: each entity gets a deterministic seed
//! derived from the world's base seed and the entity's ID.
//!
//! ## Example
//!
//! ```
//! use pecos_neo::ecs::{World, EntityId};
//! use pecos_qsim::SparseStab;
//!
//! // Create a world with base seed for determinism
//! let mut world: World<SparseStab> = World::new(42);
//!
//! // Spawn simulation instances
//! let e1 = world.spawn_with_simulator(SparseStab::new(2));
//! let e2 = world.spawn_with_simulator(SparseStab::new(2));
//!
//! // Each entity has its own deterministic RNG
//! assert!(world.rngs.get(e1).is_some());
//! assert!(world.rngs.get(e2).is_some());
//!
//! // Clone/split a trajectory for rare event simulation
//! let clones = world.split_entity(e1, 4);
//! assert_eq!(clones.len(), 3); // 3 new + 1 original = 4 total
//! ```
//!
//! ## Module Structure
//!
//! - `entity`: Entity identifiers
//! - `component`: Component types and storage
//! - `resource`: Shared resources (seed management)
//! - `world`: The main World container
//! - `coordinator`: Parallel execution coordinator

mod component;
mod coordinator;
mod entity;
mod redistribution;
mod resource;
mod splitting;
mod world;

pub use component::{
    ComponentStorage, NoiseContextComponent, OutcomeComponent, PathComponent, RngComponent,
    SimulatorComponent, StatusComponent, WeightComponent,
};
pub use coordinator::{
    ExecutionStats, ParallelConfig, ParallelCoordinator, ParallelResult, WorkerState,
};
pub use entity::EntityId;
pub use redistribution::{
    RedistributionStats, balance_entity_counts, collect_weights, redistribute_by_weight,
    total_weight,
};
pub use resource::{Resources, SeedResource};
pub use splitting::{
    CustomScoreCriterion, ScoreFn, SplitDecision, SplitStats, SplittingCriterion, SubsetLevel,
    ThresholdCriterion,
};
pub use world::{EntityTransfer, World};
