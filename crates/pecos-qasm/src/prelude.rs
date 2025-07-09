// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! A prelude for users of the `pecos-qasm` crate.
//!
//! This prelude re-exports the most commonly used types, traits, and functions
//! needed for working with `OpenQASM` in PECOS. It's designed to be imported directly
//! in documentation, tests, and examples for the `pecos-qasm` crate, where using
//! the main `pecos::prelude` would create circular dependencies.
//!
//! ## Usage
//!
//! ```
//! use pecos_qasm::prelude::*;
//!
//! // Now you can use all QASM-related PECOS types and functions
//! ```
//!
//! ## Contents
//!
//! This prelude includes:
//!
//! * Standard library types needed for QASM operations (`FromStr`, `BTreeMap`)
//! * QASM engine types (`QASMEngine`, `QASMEngineBuilder`, `QASMProgram`)
//! * QASM simulation function (`run_qasm`)
//! * Result types (`Shot`, `ShotVec`, `ShotMap`) from pecos-engines
//! * Engine traits (`ClassicalEngine`) for accessing engine methods
//! * Noise models and quantum engines from `pecos-engines`
//! * Error types and random number generator traits
//!
//! ## Note on Meta-Crate Usage
//!
//! When writing application code that uses PECOS, prefer importing from the main
//! `pecos::prelude` instead, which re-exports this prelude along with other PECOS
//! functionality.

// Standard library imports
pub use std::collections::BTreeMap;
pub use std::str::FromStr;

// Re-export engine types
pub use crate::engine::QASMEngine;
pub use crate::engine_builder::QASMEngineBuilder;
pub use crate::program::QASMProgram;

// Re-export run function
pub use crate::run::run_qasm;

// Re-export simulation module types and functions
pub use crate::simulation::{
    NoiseModelType, QasmSimulation, QasmSimulationBuilder, QuantumEngineType, qasm_sim,
};

// Re-export config module types
pub use crate::config::{NoiseConfig, QuantumEngineConfig};

// Re-export setup function
pub use crate::setup_qasm_engine;

// Re-export engine traits and types from pecos-engines
pub use pecos_engines::{
    BitVecDisplayFormat, ClassicalEngine, MonteCarloEngine, Shot, ShotMap, ShotMapDisplayExt,
    ShotMapDisplayOptions, ShotVec,
};

// Re-export core error type and traits
pub use pecos_core::RngManageable;
pub use pecos_core::errors::PecosError;
// Re-export noise models and builders from pecos-engines
pub use pecos_engines::noise::{
    BiasedDepolarizingNoiseModel, BiasedDepolarizingNoiseModelBuilder, DepolarizingNoiseModel,
    DepolarizingNoiseModelBuilder, GeneralNoiseModel, GeneralNoiseModelBuilder, NoiseModel,
    PassThroughNoiseModel, PassThroughNoiseModelBuilder,
};
// Re-export noise models from pecos-engines
pub use pecos_engines::quantum::{
    QuantumEngine, SparseStabEngine, StateVecEngine, new_stabilizer_engine,
    new_stabilizer_engine_with_seed,
};

// Re-export result formatting utilities
pub use crate::result_formatter::{
    QASMResultFormatter, format_as_binary_strings, format_as_decimal_arrays,
};
