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

//! Reusable quantum simulations with owned state and seeded shot loops.
//!
//! Configure a program, backend, noise model and sampling strategy with
//! [`sim_neo`], then build a [`Simulation`] to run and reuse.
//! Monte Carlo supports static circuits and classical engines; importance
//! sampling, path enumeration and subset simulation operate on static circuits.
//! Runtime failures are returned to the caller.
//!
//! ```
//! use pecos_neo::prelude::*;
//! use pecos_neo::tool::{monte_carlo, sim_neo, sparse_stab};
//!
//! let circuit = CommandBuilder::new().pz(&[0]).h(&[0]).mz(&[0]).build();
//! let mut sim = sim_neo(circuit)
//!     .quantum(sparse_stab())
//!     .sampling(monte_carlo(100))
//!     .seed(42)
//!     .build();
//! let results = sim.run().expect("simulation should succeed");
//! assert_eq!(results.len(), 100);
//! ```

mod simulation;

pub use crate::sampling::subset::{LevelStats, SubsetResult};
pub use pecos_results::{Data, Shot, ShotMap, ShotVec};
pub use simulation::{
    Circuit, CustomBackendBuilder, ImportanceSamplingBuilder, MonteCarloBuilder,
    PathEnumerationBuilder, QuantumBackend, Sampling, SimConfig, SimNeoBuilder, SimNeoInput,
    Simulation, SimulationResults, SimulatorFactory, SparseStabBuilder, StabilizerBuilder,
    StateVecBuilder, StoredOverrides, SubsetFailureFn, SubsetScoreFn, SubsetSimulationBuilder,
    custom_backend, custom_backend_from_factory, custom_backend_with_rotations,
    importance_sampling, monte_carlo, path_enumeration, sim_neo, sim_neo_builder, sparse_stab,
    stabilizer, state_vector, subset_simulation,
};
pub use simulation::{PendingEngineBuilder, TypedProgram};
