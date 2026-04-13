//! QASM parser and engine for PECOS
//!
//! This crate provides a complete QASM 2.0 parser and execution engine,
//! with several enhancements:
//!
//! - Scientific notation support for floating-point numbers
//! - Mathematical functions (sin, cos, tan, exp, ln, sqrt)
//! - Power operator (**) for exponentiation
//! - Include file preprocessing with support for:
//!   - Custom include search paths
//!   - Virtual includes (in-memory content)
//!   - Circular dependency detection
//!
//! # Example: Using the Simplified API
//!
//! ## Parsing from a string
//!
//! ```
//! use pecos_qasm::QASMEngine;
//! use pecos_engines::{ClassicalEngine, ClassicalControlEngine};
//! use std::str::FromStr;
//!
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     include "qelib1.inc";
//!     qreg q[2];
//!     h q[0];
//! "#;
//!
//! let engine = QASMEngine::from_str(qasm)?;
//! assert_eq!(engine.num_qubits(), 2);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```
//!
//! ## Using the builder API
//!
//! ```
//! use pecos_qasm::QASMEngine;
//! use pecos_engines::{ClassicalEngine, ClassicalControlEngine};
//!
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     include "custom.inc";
//!     qreg q[1];
//!     my_gate q[0];
//! "#;
//!
//! let engine = QASMEngine::builder()
//!     .with_virtual_include("custom.inc", "gate my_gate a { H a; }")
//!     .allow_complex_conditionals(true)
//!     .build_from_str(qasm)?;
//! assert_eq!(engine.num_qubits(), 1);
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

pub mod ast;
pub mod bitvec_expression;
// pub mod dag_bridge; // TODO: requires DagCircuit classical bit API
// pub mod config; // TODO: Update to use unified API types
pub mod engine;
pub mod engine_builder;
pub mod foreign_objects;
pub mod includes;
pub mod parser;
pub mod prelude;
pub mod preprocessor;
pub mod program;
pub mod result_formatter;
pub mod run;
pub mod simulation;
pub mod unified_engine_builder;
pub mod util;

#[cfg(feature = "phir")]
pub mod qasm_to_phir;
pub mod qasm_to_phir_json;

#[cfg(feature = "wasm")]
pub mod wasm_foreign_object;

pub use crate::run::run_qasm;
pub use ast::{Expression, GateOperation, Operation, OperationDisplay};
pub use engine::QASMEngine;
pub use engine_builder::QASMEngineBuilder;
pub use parser::{ParseConfig, QASMParser};
pub use preprocessor::Preprocessor;
pub use program::QASMProgram;
#[cfg(feature = "wasm")]
pub use program::QasmEngineWasm;
#[cfg(feature = "phir")]
pub use qasm_to_phir::{qasm_program_to_phir_module, qasm_to_phir_module, qasm_to_ron};
pub use qasm_to_phir_json::{program_to_phir_json, qasm_to_phir_json};
pub use unified_engine_builder::{QasmEngineBuilder, qasm_engine};
pub use util::{count_qubits_in_file, count_qubits_in_str};

/// List of built-in mathematical functions that cannot be overridden by WASM
pub const BUILTIN_FUNCTIONS: [&str; 6] = ["sin", "cos", "tan", "exp", "ln", "sqrt"];
pub const PLATFORM_FUNCTIONS: [&str; 4] = ["RNGseed", "RNGbound", "RNGindex", "RNGnum"];

use log::debug;
use pecos_core::errors::PecosError;
use pecos_engines::ClassicalControlEngine;
use std::path::Path;

/// Sets up a basic QASM engine.
///
/// This function creates a QASM engine from the provided path.
///
/// # Parameters
///
/// - `program_path`: A reference to the path of the QASM program file
/// - `seed`: Optional seed value for deterministic execution
///
/// # Returns
///
/// Returns a `Box<dyn ClassicalEngine>` containing the QASM engine
///
/// # Errors
///
/// This function may return the following errors:
/// - `PecosError::IO`: If the QASM file cannot be read
/// - `PecosError::Processing`: If the QASM engine creation fails or if parsing fails
pub fn setup_qasm_engine(
    program_path: &Path,
    seed: Option<u64>,
) -> Result<Box<dyn ClassicalControlEngine>, PecosError> {
    debug!("Setting up QASM engine for: {}", program_path.display());

    // Note: The seed parameter is unused as QASMEngine doesn't handle randomness.
    // Randomness is managed by the QuantumEngine in MonteCarloEngine.
    // The seed parameter is kept for API consistency with other engines.
    let _ = seed;

    // Use the QASMEngine from the pecos-qasm crate
    let engine = QASMEngine::from_file(program_path).map_err(|e| {
        PecosError::Processing(format!(
            "QASM engine setup failed: Could not create engine: {e}"
        ))
    })?;

    Ok(Box::new(engine))
}
