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

//! Bevy-inspired Tool architecture for quantum simulation and validation.
//!
//! This module provides a flexible, plugin-based system for building various quantum tools.
//! See `design.md` for the full architecture documentation.
//!
//! # Overview
//!
//! The architecture consists of:
//!
//! - [`Tool`] - Generic Bevy-like foundation with plugins, systems, and resources
//! - [`Plugin`] - Trait for bundling functionality
//! - [`Stage`] - Execution stages (Startup, `PreShot`, Execute, `PostShot`, Finish)
//! - [`Resources`] - Typed singleton storage
//!
//! # Example
//!
//! ```
//! use pecos_neo::tool::{Tool, Stage, Resources};
//!
//! // Create a simple counter tool
//! let mut tool = Tool::new()
//!     .insert_resource(0u32)
//!     .add_system(Stage::Execute, |res: &mut Resources| {
//!         *res.get_mut::<u32>() += 1;
//!     });
//!
//! tool.run();
//! assert_eq!(*tool.resource::<u32>(), 1);
//!
//! tool.run();
//! assert_eq!(*tool.resource::<u32>(), 2);
//! ```
//!
//! # Stages
//!
//! Tools execute systems in stages:
//!
//! - **Startup**: Runs once at the beginning (initialize simulators, compile circuits)
//! - `PreShot`: Before each shot (reset state, derive seeds)
//! - **Execute**: Run the main logic (circuit execution with noise)
//! - `PostShot`: After each shot (collect outcomes, update weights)
//! - **Finish**: Runs once at the end (aggregate results, compute statistics)
//!
//! # Plugins
//!
//! Plugins bundle related resources and systems:
//!
//! ```
//! use pecos_neo::tool::{Tool, Plugin, Stage, Resources};
//!
//! struct CounterPlugin {
//!     initial: u32,
//! }
//!
//! impl Plugin for CounterPlugin {
//!     fn build(&self, tool: &mut Tool) {
//!         tool.insert_resource_mut(self.initial);
//!         tool.add_system_mut(Stage::Execute, |res: &mut Resources| {
//!             *res.get_mut::<u32>() += 1;
//!         });
//!     }
//! }
//!
//! let mut tool = Tool::new()
//!     .add_plugin(&CounterPlugin { initial: 10 });
//!
//! tool.run();
//! assert_eq!(*tool.resource::<u32>(), 11);
//! ```

mod core;
mod importance;
mod plugin;
mod resource;
mod simulation;
mod system;

// Re-export core types
pub use self::core::Tool;
pub use importance::{
    CurrentShotWeight, ImportanceSamplingConfig, ImportanceSamplingPlugin,
    ImportanceSamplingResults,
};
pub use plugin::{Plugin, PluginGroup};
pub use resource::{Resource, Resources};
pub use simulation::{
    Circuit, CustomBackendBuilder, ImportanceSamplingBuilder, NoiseResource, Orchestrator,
    QuantumBackend, SimConfig, SimNeoBuilder, SimNeoInput, Simulation, SimulationResults,
    SimulatorFactory, SparseStabBuilder, StateVecBuilder, StoredOverrides, custom_backend,
    custom_backend_with_rotations, importance_sampling, sim_neo, sim_neo_builder, sparse_stab,
    state_vector,
};
#[cfg(feature = "engines-adapter")]
pub use simulation::{PendingEngineBuilder, TypedProgram};
pub use system::{IntoSystem, Schedule, System};

/// Execution stages for quantum tool workflows.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Stage {
    /// Once at beginning (init simulators, compile circuits)
    Startup,
    /// Before each shot (reset state, derive seed)
    PreShot,
    /// Run the circuit with noise
    Execute,
    /// After each shot (collect outcomes, update weights)
    PostShot,
    /// Once at end (aggregate results, compute statistics)
    Finish,
}
