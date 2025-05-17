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
//! ```no_run
//! use pecos_qasm::QASMEngine;
//! use std::str::FromStr;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Simple case - parse from string or file
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     include "qelib1.inc";
//!     qreg q[2];
//!     h q[0];
//! "#;
//!
//! // From string
//! let engine1 = QASMEngine::from_str(qasm)?;
//!
//! // From file
//! let engine2 = QASMEngine::from_file("circuit.qasm")?;
//!
//! // Complex case - use builder for virtual includes and custom paths
//! let engine3 = QASMEngine::builder()
//!     .with_virtual_include("custom.inc", "gate my_gate a { h a; }")
//!     .with_include_path("/custom/includes")
//!     .allow_complex_conditionals(true)
//!     .build_from_str(qasm)?;
//! # Ok(())
//! # }
//! ```

pub mod ast;
pub mod engine;
pub mod engine_builder;
pub mod includes;
pub mod parser;
pub mod preprocessor;
pub mod util;

pub use ast::{Expression, GateOperation, Operation, OperationDisplay};
pub use engine::QASMEngine;
pub use engine_builder::QASMEngineBuilder;
pub use parser::{ParseConfig, QASMParser};
pub use preprocessor::Preprocessor;
pub use util::{count_qubits_in_file, count_qubits_in_str};
