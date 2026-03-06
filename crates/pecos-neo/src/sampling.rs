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

//! Advanced sampling methods for rare event simulation.
//!
//! This module provides infrastructure for sampling methods beyond standard Monte Carlo:
//!
//! - **Importance Sampling**: Bias toward rare events, correct with weights
//! - **Path Exploration**: Systematically enumerate and analyze execution paths
//! - **Splitting**: Clone promising trajectories
//! - **Subset Simulation**: Condition on intermediate events
//!
//! These methods are essential for efficiently estimating rare event probabilities
//! like logical error rates in quantum error correction.
//!
//! ## Path Exploration
//!
//! For programs with measurement-dependent branching (like QEC with feedback),
//! you can systematically explore all paths:
//!
//! ```no_run
//! use pecos_neo::sampling::path::{PathExplorer, PathEnumerator, PathStatistics};
//! use pecos_neo::prelude::*;
//! use pecos_qsim::SparseStab;
//!
//! let n = 5;
//! let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();
//! let mut explorer = PathExplorer::new(SparseStab::new(n));
//! let mut stats = PathStatistics::new();
//!
//! // Enumerate all paths up to 10 non-deterministic measurements
//! for path in PathEnumerator::new(10) {
//!     let result = explorer.run_with_path(&commands, &path);
//!     // Check outcomes and accumulate statistics
//!     stats.add(0.0, path.probability());
//! }
//!
//! println!("Logical error rate: {}", stats.mean());
//! ```

pub mod importance;
pub mod importance_runner;
pub mod monte_carlo;
pub mod path;
pub mod subset;
pub mod weight;

pub use importance::{ImportanceConfig, ImportanceSamplingNoise};
pub use importance_runner::{ImportanceSampledShot, ImportanceSamplingRunner, OutcomeBiasConfig};
pub use monte_carlo::{
    ImportanceSamplingResults, MonteCarloConfig, MonteCarloResults, MonteCarloRunner,
};
pub use path::{
    EnumeratedPath, MeasurementPath, PathEnumerator, PathExplorer, PathOutcome, PathRecordedResult,
    PathSignature, PathStatistics,
};
pub use subset::{
    BernoulliSubsetSimulation, EcsSubsetSimulation, HistoryTrajectory, ProperSubsetSimulation,
    QecCheckpoint, QecHistoryTrajectory, QecSubsetConfig, QecSubsetSimulation, QecTrajectory,
    RoundResult, SubsetConfig, SubsetResult, SubsetSimulation, SyndromeScore, Trajectory,
    TrajectoryCheckpoint, bit_flip_syndrome_circuit, phase_flip_syndrome_circuit,
};
pub use weight::{SampleWeight, WeightedOutcome, WeightedStatistics};
