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
//! This prelude re-exports the preludes from all PECOS component crates,
//! plus pecos-specific functionality like the unified simulation API.
//!
//! ## Recommended Usage
//!
//! ```rust,no_run
//! use pecos::prelude::*;
//!
//! let qasm_code = r#"
//!     OPENQASM 2.0;
//!     include "qelib1.inc";
//!     qreg q[2];
//!     h q[0];
//!     cx q[0], q[1];
//! "#;
//! let program = Qasm::from_string(qasm_code);
//!
//! let results = sim(program)
//!     .quantum(sparse_stabilizer())
//!     .seed(42)
//!     .run(1000)?;
//! # Ok::<(), pecos_core::errors::PecosError>(())
//! ```
//!
//! ## What's Included
//!
//! This prelude includes everything from:
//!
//! - `pecos_core::prelude` - Core types, traits, and error handling
//! - `pecos_engines::prelude` - Simulation engines and builders
//! - `pecos_qasm::prelude` - `OpenQASM` language support
//! - `pecos_qsim::prelude` - Quantum simulation implementations
//! - `pecos_qis_core::prelude` - QIS control engine
//! - `pecos_qis_selene::prelude` - Selene-based QIS interface (when `selene` feature enabled)
//! - `pecos_programs::prelude` - Program type definitions
//! - `pecos_rng::prelude` - Random number generation
//! - `pecos_num::prelude` - Numerical computing (scipy.optimize replacement)
//! - `pecos_hugr_qis::prelude` - HUGR to QIS compilation
//! - `pecos_phir_json::prelude` - PHIR-JSON format support
//!
//! Plus pecos-specific items:
//!
//! - Unified simulation API: `sim()`, `SimBuilderExt`
//! - Program utilities: `detect_program_type()`, etc.
//! - Feature-gated quantum backends: `CppSparseStab`, `QuestStateVec`, etc.
//!
//! For organized access to specific functionality, use the namespace modules:
//!
//! - [`crate::engines`] - Classical control engines
//! - [`crate::quantum`] - Quantum simulation backends
//! - [`crate::noise`] - Noise models
//! - [`crate::runtime`] - QIS runtimes

// ============================================================================
// Re-export preludes from component crates
// ============================================================================

pub use pecos_core::prelude::*;
pub use pecos_engines::prelude::*;
#[cfg(feature = "qasm")]
pub use pecos_qasm::prelude::*;
pub use pecos_qsim::prelude::*;

// Re-export pecos_qis_core prelude
// Note: Shot and Value from pecos_qis_core are not included (removed from its prelude)
// Re-export QIS core prelude (when qis feature is enabled)
#[cfg(feature = "qis")]
pub use pecos_qis_core::prelude::*;

// Re-export Selene QIS interface when feature is enabled
#[cfg(feature = "qis")]
pub use pecos_qis_selene::prelude::*;

// Re-export program types prelude
pub use pecos_programs::prelude::*;

// Re-export RNG prelude
pub use pecos_rng::prelude::*;

// Re-export numerical computing prelude
pub use pecos_num::prelude::*;

// Re-export HUGR compiler prelude
#[cfg(feature = "hugr")]
pub use pecos_hugr_qis::prelude::*;

// Re-export LLVM IR generation prelude
#[cfg(feature = "llvm")]
pub use pecos_llvm::prelude::*;

// Re-export PHIR-JSON prelude
#[cfg(feature = "phir")]
pub use pecos_phir_json::prelude::*;

// Re-export PHIR configuration (not commonly used, but available)
pub use pecos_phir::PhirConfig;

// ============================================================================
// Pecos-specific items (unified API and utilities)
// ============================================================================

// Re-export crate-specific utilities from pecos crate itself
pub use crate::program::{
    ProgramType, detect_program_type, get_program_path, setup_engine_for_program,
};

// Re-export unified simulation API from pecos crate
pub use crate::unified_sim::{ProgrammedSimBuilder, SimBuilderExt, sim};

// ============================================================================
// Feature-gated quantum simulator backends
// ============================================================================

#[cfg(feature = "cppsparsesim")]
pub use pecos_cppsparsesim::CppSparseStab;

#[cfg(feature = "quest")]
pub use pecos_quest::{QuestDensityMatrix, QuestStateVec};

#[cfg(feature = "qulacs")]
pub use pecos_qulacs::QulacsStateVec;

// ============================================================================
// WebAssembly foreign object support
// ============================================================================

#[cfg(feature = "wasm")]
pub use pecos_wasm::{ForeignObject, WasmForeignObject};

// ============================================================================
// Decoder support
// ============================================================================

// Re-export core decoder traits (always available)
#[cfg(any(feature = "ldpc", feature = "all-decoders"))]
pub use pecos_decoders::{BatchDecoder, CssDecoder, Decoder, DecoderError, SoftDecoder};
