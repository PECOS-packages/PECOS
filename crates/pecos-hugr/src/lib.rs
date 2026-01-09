// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Direct HUGR interpreter engine for PECOS.
//!
//! This crate provides [`HugrEngine`], a classical control engine that directly
//! interprets HUGR (Hierarchical Unified Graph Representation) programs without
//! requiring LLVM compilation.
//!
//! # Overview
//!
//! The `HugrEngine` walks a HUGR graph in topological order, emitting quantum
//! commands via [`ByteMessage`] and handling measurement results. This is similar
//! to how [`QASMEngine`] interprets `OpenQASM` programs.
//!
//! # Quick Start
//!
//! Load a HUGR file and build an engine:
//!
//! ```
//! use pecos_hugr::hugr_engine;
//! use pecos_engines::{ClassicalControlEngineBuilder, ClassicalEngine};
//!
//! // Load a HUGR circuit
//! let hugr_path = concat!(
//!     env!("CARGO_MANIFEST_DIR"),
//!     "/../pecos/tests/test_data/hugr/single_hadamard.hugr"
//! );
//! let engine = hugr_engine()
//!     .hugr_file(hugr_path)
//!     .build()
//!     .expect("Failed to load HUGR");
//!
//! // Check the circuit was loaded
//! assert!(engine.num_qubits() >= 1);
//! ```
//!
//! For full simulation with quantum execution (requires a quantum backend):
//!
//! ```no_run
//! use pecos_hugr::hugr_sim;
//!
//! // Run 100 shots of a HUGR circuit
//! let hugr_path = concat!(
//!     env!("CARGO_MANIFEST_DIR"),
//!     "/../pecos/tests/test_data/hugr/bell_state.hugr"
//! );
//! let results = hugr_sim(hugr_path)
//!     .seed(42)
//!     .run(100)
//!     .unwrap();
//!
//! for shot in &results.shots {
//!     println!("Measurement: {:?}", shot.data);
//! }
//! ```
//!
//! # Supported Operations
//!
//! Currently supports quantum circuits with:
//! - Single-qubit gates: H, X, Y, Z, S, Sdg, T, Tdg, Rx, Ry, Rz
//! - Two-qubit gates: CX, CY, CZ, `ZZMax` (SZZ)
//! - Lifecycle: `QAlloc`, `QFree`, Measure, `MeasureFree`, Reset
//! - Control flow: Conditional nodes (if/else based on measurement results)

mod builder;
mod engine;
mod loader;

pub use builder::{HugrEngineBuilder, hugr_engine, hugr_sim};
pub use engine::{CapturedResult, ClassicalValue, FutureId, HugrEngine, ResultValue, RngContextId};
pub use loader::{load_hugr_from_bytes, load_hugr_from_file};

// Re-export key types for convenience
pub use pecos_engines::prelude::{ByteMessage, ClassicalEngine, ControlEngine, Engine};
pub use tket::hugr::Hugr;
