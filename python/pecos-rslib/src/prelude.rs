//! Local prelude re-exporting from direct crate dependencies.
//!
//! Replaces `use pecos::prelude::*` now that pecos-rslib depends on
//! individual crates instead of the `pecos` metacrate.

// Core types, traits, error handling
pub use pecos_core::prelude::*;

// Simulation engines and builders
pub use pecos_engines::prelude::*;

// Quantum simulation implementations
pub use pecos_simulators::prelude::*;

// Program type definitions
pub use pecos_programs::prelude::*;

// Random number generation
pub use pecos_random::prelude::*;

// Numerical computing (Array1, math traits, etc.)
pub use pecos_num::prelude::*;

// QIS / LLVM IR execution
pub use pecos_qis::prelude::*;

// HUGR compilation
pub use pecos_hugr_qis::prelude::*;

// PHIR-JSON format
pub use pecos_phir_json::prelude::*;

// C++ simulator backends
pub use pecos_quest::{QuestDensityMatrix, QuestStateVec};
pub use pecos_qulacs::QulacsStateVec;

// WASM types (feature-gated)
#[cfg(feature = "wasm")]
pub use pecos_wasm::ForeignObject;
