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

//! QASM program representation and parsing utilities.

use crate::engine::QASMEngine;
use crate::parser::Program;
use pecos_core::errors::PecosError;
use pecos_engines::ClassicalEngine;
use std::fs::read_to_string;
use std::path::Path;
use std::str::FromStr;

/// A parsed QASM program.
///
/// This type represents a parsed QASM program that can be used to create a `QASMEngine`
/// or inspected directly. It provides a more type-safe way to handle QASM programs
/// compared to using raw strings.
///
/// # Examples
///
/// Basic usage:
///
/// ```
/// use pecos_qasm::QASMProgram;
/// use std::str::FromStr;
///
/// // Parse a QASM program from a string
/// let qasm = r#"
///     OPENQASM 2.0;
///     include "qelib1.inc";
///     qreg q[2];
///     h q[0];
/// "#;
///
/// let program = QASMProgram::from_str(qasm).unwrap();
///
/// // Get information about the program
/// println!("Total qubits: {}", program.num_qubits());
///
/// // Convert to a QASMEngine for execution
/// let engine = program.into_engine();
/// ```
///
/// Using with the PECOS simulation API:
///
/// ```
/// use pecos_qasm::QASMProgram;
/// use pecos_engines::ClassicalEngine;
/// use std::str::FromStr;
///
/// // Parse a QASM program
/// let qasm = r#"
///     OPENQASM 2.0;
///     include "qelib1.inc";
///     qreg q[2];
///     creg c[2];
///     h q[0];
///     cx q[0], q[1];
///     measure q -> c;
/// "#;
///
/// let program = QASMProgram::from_str(qasm)?;
///
/// // Convert to engine and verify properties
/// let engine = program.into_engine();
/// assert_eq!(engine.num_qubits(), 2);
///
/// // The engine is ready for simulation
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Debug, Clone)]
pub struct QASMProgram {
    /// The parsed QASM program
    program: Program,
    /// The original QASM code
    source: String,
}

impl QASMProgram {
    /// Creates a new `QASMProgram` from a parsed Program and source code.
    ///
    /// This is generally used internally - users should prefer `from_str` or `from_file`.
    #[must_use]
    pub fn new(program: Program, source: String) -> Self {
        Self { program, source }
    }

    /// Get the number of qubits in the program.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.program.total_qubits
    }

    /// Get a reference to the internal Program AST.
    #[must_use]
    pub fn program(&self) -> &Program {
        &self.program
    }

    /// Get the original source code of the program.
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    /// Convert this program into a `QASMEngine` for execution.
    #[must_use]
    pub fn into_engine(self) -> QASMEngine {
        QASMEngine::new(self)
    }

    /// Convert this program into a boxed `QASMEngine` ready for simulation.
    ///
    /// This is particularly convenient when using the `run_sim` function from the
    /// pecos crate, which takes a `Box<dyn ClassicalEngine>`.
    #[must_use]
    pub fn into_engine_box(self) -> Box<dyn ClassicalEngine> {
        Box::new(self.into_engine())
    }

    /// Parse a QASM program from a file.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or the QASM code is invalid.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, PecosError> {
        let source = read_to_string(path).map_err(|e| PecosError::Input(e.to_string()))?;
        Self::from_str(&source)
    }

    /// Get all function calls used in the program that are not built-in functions
    #[must_use]
    pub fn get_non_builtin_function_calls(&self) -> Vec<String> {
        use crate::ast::{Expression, Operation};
        use std::collections::BTreeSet;

        // Helper function to extract function calls from expressions
        fn extract_function_calls(expr: &Expression, calls: &mut BTreeSet<String>) {
            match expr {
                Expression::FunctionCall { name, args } => {
                    if !crate::BUILTIN_FUNCTIONS.contains(&name.as_str()) {
                        calls.insert(name.clone());
                    }
                    // Recursively check arguments
                    for arg in args {
                        extract_function_calls(arg, calls);
                    }
                }
                Expression::BinaryOp { left, right, .. } => {
                    extract_function_calls(left, calls);
                    extract_function_calls(right, calls);
                }
                Expression::UnaryOp { op: _, expr } => {
                    extract_function_calls(expr, calls);
                }
                _ => {}
            }
        }

        let mut function_calls = BTreeSet::new();

        // Check all operations
        for op in &self.program.operations {
            if let Operation::ClassicalAssignment { expression, .. } = op {
                extract_function_calls(expression, &mut function_calls);
            }
        }

        // Note: Gate parameters are f64 values, not expressions in the current AST
        // So we don't need to check them for function calls

        function_calls.into_iter().collect()
    }
}

impl FromStr for QASMProgram {
    type Err = PecosError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let program = crate::parser::QASMParser::parse_str(s)?;
        Ok(Self::new(program, s.to_string()))
    }
}

impl std::fmt::Display for QASMProgram {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.source)
    }
}
