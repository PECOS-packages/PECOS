// Copyright 2024 The PECOS Developers
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

//! A prelude for PECOS users.
//!
//! This prelude re-exports the most commonly used types, traits, and functions
//! from all PECOS component crates. By importing this prelude with
//! `use pecos::prelude::*;`, you get access to the complete PECOS API without
//! having to manually import from each component crate.
//!
//! ## Contents
//!
//! This prelude includes re-exports from:
//!
//! * `pecos_core`: Core types, traits, and error handling
//! * `pecos_engines`: Simulation engines for quantum and classical processing
//! * `pecos_phir`: PECOS High-level Intermediate Representation
//! * `pecos_qasm`: `OpenQASM` language support
//! * `pecos_qir`: Quantum Intermediate Representation support
//! * `pecos_qsim`: Quantum simulation implementations
//!
//! It also includes key functionality from the top-level PECOS crate:
//!
//! * Simulation functions (`run_sim`)
//! * Engine setup functions (`setup_qasm_engine`, `setup_qir_engine`)
//! * Program type detection and handling
//!
//! ## Usage
//!
//! ```rust
//! use pecos::prelude::*;
//!
//! // Now you can use all common PECOS types and functions without additional imports
//! ```

// Re-export preludes from component crates
pub use pecos_core::prelude::*;
pub use pecos_engines::prelude::*;
pub use pecos_phir::prelude::*;
pub use pecos_qasm::prelude::*;
pub use pecos_qir::prelude::*;
pub use pecos_qsim::prelude::*;

// Re-export ShotVec directly from pecos_engines for easier access
pub use pecos_engines::shot_results::ShotVec;

// Re-export crate-specific utilities
pub use crate::program::{
    ProgramType, detect_program_type, get_program_path, setup_engine_for_program,
};

// Re-export setup functions from format-specific crates
pub use pecos_phir::setup_phir_engine;
pub use pecos_qasm::setup_qasm_engine;
pub use pecos_qir::setup_qir_engine;

// Re-export run_sim from pecos-engines
pub use pecos_engines::run_sim;

// Re-export PCG RNG functions
pub use pecos_rng::rng_pcg;
