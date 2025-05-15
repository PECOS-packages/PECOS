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
//! # Example: Using Custom Include Paths
//!
//! ```no_run
//! use pecos_qasm::{QASMParser, QASMEngine};
//! use std::path::PathBuf;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Parse with custom include paths
//! let qasm = r#"
//!     OPENQASM 2.0;
//!     include "custom_gates.inc";
//!     qreg q[1];
//!     my_gate q[0];
//! "#;
//!
//! let include_paths = vec![
//!     PathBuf::from("/custom/includes"),
//!     PathBuf::from("./local/qasm")
//! ];
//!
//! let program = QASMParser::parse_str_with_include_paths(qasm, include_paths)?;
//!
//! // Or use with the engine
//! let mut engine = QASMEngine::new()?;
//! engine.from_str_with_include_paths(qasm, vec!["/custom/includes"])?;
//! # Ok(())
//! # }
//! ```

pub mod ast;
pub mod engine;
pub mod parser;
pub mod preprocessor;
pub mod util;
pub mod includes;

pub use ast::{Expression, Operation};
pub use engine::QASMEngine;
pub use parser::QASMParser;
pub use preprocessor::Preprocessor;
pub use util::{count_qubits_in_file, count_qubits_in_str};
