pub mod comparison;
pub mod config;
pub mod constant_folding;
pub mod errors;
pub mod expressions;
pub mod gates;
pub mod native_gates;
pub mod operations;
pub mod register_manager;
pub mod registers;
pub mod statements;
pub mod utils;

// Re-export commonly used types
pub use config::ParseConfig;

use pecos_core::errors::PecosError;
use pest::iterators::Pair;
use pest_derive::Parser;
use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::Path;

use crate::ast::{GateDefinition, Operation, OperationDisplay};
use crate::parser::gates::parse_gate_definition;
use crate::parser::operations::{parse_classical_operation, parse_if_statement, parse_quantum_op};
use crate::parser::registers::parse_register;
use crate::parser::statements::parse_statement;
use crate::parser::utils::{expand_gates, validate_no_opaque_gate_usage};
use crate::preprocessor::Preprocessor;

#[derive(Parser)]
#[grammar = "qasm.pest"]
#[allow(clippy::too_many_lines)] // Generated code from pest
pub struct QASMParser;

/// Native gates that PECOS can execute directly through `ByteMessage`
/// These gates don't need to be expanded and can be handled by the quantum engine
pub const PECOS_NATIVE_GATES: &[&str] = &[
    // Quantum gates from ByteMessage::GateType
    "X", "Y", "Z", "H", "CX", "SZZ", "RZ", "RX", "RY", "R1XY", "RZZ", "SZZdg", "U",
    // Special operations (these are handled differently but treated as "native")
    "barrier", "reset", "opaque", "measure",
];

impl Operation {
    /// Display this operation with proper register names using the qubit mapping
    #[must_use]
    pub fn display_with_map<'a>(
        &'a self,
        qubit_map: &'a BTreeMap<usize, (String, usize)>,
    ) -> OperationDisplay<'a> {
        OperationDisplay {
            operation: self,
            qubit_map,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Program {
    pub version: String,
    pub operations: Vec<Operation>,
    pub gate_definitions: BTreeMap<String, GateDefinition>,
    pub quantum_registers: BTreeMap<String, Vec<usize>>, // register_name -> vec of global qubit IDs
    pub classical_registers: BTreeMap<String, usize>,    // register_name -> size
    pub total_qubits: usize,
    pub qubit_map: BTreeMap<usize, (String, usize)>, // global_id -> (register_name, index)
}

impl QASMParser {
    /// Constant for QASM operation error context
    const QASM_OPERATION: &'static str = "QASM operation";

    /// Create a `CompileInvalidOperation` error with QASM context
    pub(crate) fn error(reason: impl Into<String>) -> PecosError {
        PecosError::CompileInvalidOperation {
            operation: Self::QASM_OPERATION.to_string(),
            reason: reason.into(),
        }
    }

    /// Get the standard includes directory path
    fn get_standard_includes_path() -> std::path::PathBuf {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        std::path::Path::new(manifest_dir).join("includes")
    }

    /// Parse QASM source with default configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the QASM source cannot be parsed.
    pub fn parse_str(source: &str) -> Result<Program, PecosError> {
        Self::parse_with_config(source, &ParseConfig::default())
    }

    /// Main parsing method using configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the QASM source cannot be parsed with the given configuration.
    pub fn parse_with_config(source: &str, config: &ParseConfig) -> Result<Program, PecosError> {
        // Create preprocessor
        let mut preprocessor = Preprocessor::new();
        for (name, content) in &config.includes {
            preprocessor.add_include(name, content);
        }
        for path in &config.search_paths {
            preprocessor.add_path(path);
        }
        if let Some(path_str) = Self::get_standard_includes_path().to_str() {
            preprocessor.add_path(path_str);
        }

        // Preprocess the source
        let preprocessed_source = preprocessor.preprocess_str(source)?;

        // Parse the preprocessed source
        let mut program = Self::parse_str_raw(&preprocessed_source)?;

        // Expand gates if requested
        if config.expand_gates {
            expand_gates(&mut program)?;
        }

        // Validate if requested
        if config.validate_gates {
            validate_no_opaque_gate_usage(&program)?;
        }

        Ok(program)
    }

    /// Parse a file with default configuration
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Program, PecosError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path)?;

        // Add the directory of the file to search paths for relative includes
        let mut config = ParseConfig::default();
        if let Some(parent) = path.parent() {
            config.search_paths.push(parent.to_path_buf());

            // Also check for an includes subdirectory
            let includes_dir = parent.join("includes");
            if includes_dir.is_dir() {
                config.search_paths.push(includes_dir);
            }
        }

        Self::parse_with_config(&content, &config)
    }

    /// Get the preprocessed QASM (after phase 1 - include resolution)
    ///
    /// # Errors
    ///
    /// Returns an error if preprocessing fails.
    pub fn preprocess(source: &str) -> Result<String, PecosError> {
        let mut preprocessor = Preprocessor::new();
        // Add standard includes path as fallback for filesystem includes
        if let Some(path_str) = Self::get_standard_includes_path().to_str() {
            preprocessor.add_path(path_str);
        }
        preprocessor.preprocess(source)
    }

    /// Get the preprocessed and expanded QASM (after phases 1 and 2)
    ///
    /// # Errors
    ///
    /// Returns an error if preprocessing or expansion fails.
    pub fn preprocess_and_expand(source: &str) -> Result<String, PecosError> {
        // Phase 1: Preprocess includes
        let preprocessed = Self::preprocess(source)?;

        // Phase 2: Expand gates to native operations
        Self::expand_all_gate_definitions(&preprocessed)
    }

    /// Expand all gate definitions in QASM source to native gates only.
    ///
    /// # Errors
    ///
    /// Returns an error if gate expansion fails.
    pub fn expand_all_gate_definitions(source: &str) -> Result<String, PecosError> {
        // Parse the source to get gate definitions and operations
        let mut program = Self::parse_phase1(source)?;

        // Expand all gates
        expand_gates(&mut program)?;

        // Convert back to QASM string with expanded operations only (no gate definitions)
        Ok(Self::program_to_qasm_expanded(&program))
    }

    /// Parse only phase 1 - just enough to get gate definitions and operations
    fn parse_phase1(source: &str) -> Result<Program, PecosError> {
        let mut program = Program::default();
        let mut pairs =
            <Self as pest::Parser<Rule>>::parse(Rule::program, source).map_err(|e| {
                PecosError::ParseSyntax {
                    language: "QASM".to_string(),
                    message: e.to_string(),
                }
            })?;

        let program_pair = pairs.next().ok_or_else(|| Self::error("Empty program"))?;

        for pair in program_pair.into_inner() {
            match pair.as_rule() {
                Rule::oqasm => {
                    // Version declaration
                    if let Some(version_pair) = pair.into_inner().next() {
                        program.version = version_pair.as_str().to_string();
                    }
                }
                Rule::statement => {
                    for inner_pair in pair.into_inner() {
                        match inner_pair.as_rule() {
                            Rule::register_decl => parse_register(inner_pair, &mut program)?,
                            Rule::gate_def => {
                                parse_gate_definition(inner_pair, &mut program)?;
                            }
                            Rule::quantum_op => {
                                if let Some(op) = parse_quantum_op(inner_pair, &program)? {
                                    program.operations.push(op);
                                }
                            }
                            Rule::classical_op => {
                                if let Some(op) = parse_classical_operation(inner_pair, &program)? {
                                    program.operations.push(op);
                                }
                            }
                            Rule::if_stmt => {
                                if let Some(op) = parse_if_statement(inner_pair, &program)? {
                                    program.operations.push(op);
                                }
                            }
                            _ => {} // Skip other operations for phase 1
                        }
                    }
                }
                _ => {} // Skip other rules
            }
        }

        Ok(program)
    }

    /// Convert a Program back to QASM string with only expanded operations (no gate definitions)
    fn program_to_qasm_expanded(program: &Program) -> String {
        let mut qasm = String::new();

        // Version
        if !program.version.is_empty() {
            writeln!(qasm, "OPENQASM {};", program.version).unwrap();
        }

        // Quantum registers
        for (name, qubits) in &program.quantum_registers {
            writeln!(qasm, "qreg {}[{}];", name, qubits.len()).unwrap();
        }

        // Classical registers
        for (name, size) in &program.classical_registers {
            writeln!(qasm, "creg {name}[{size}];").unwrap();
        }

        // Operations (expanded) - no gate definitions
        for op in &program.operations {
            qasm.push_str(&Self::format_operation(op, &program.qubit_map));
            qasm.push_str(";\n");
        }

        qasm
    }

    /// Format an operation with proper qubit register names
    fn format_operation(op: &Operation, qubit_map: &BTreeMap<usize, (String, usize)>) -> String {
        // Use the display wrapper to properly format with register names
        format!("{}", op.display_with_map(qubit_map))
    }

    /// Parse QASM with virtual includes but without gate expansion (for testing)
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    #[cfg(test)]
    pub fn parse_str_with_virtual_includes_no_expansion(
        source: &str,
        virtual_includes: impl IntoIterator<Item = (String, String)>,
    ) -> Result<Program, PecosError> {
        let config = ParseConfig {
            includes: virtual_includes.into_iter().collect(),
            expand_gates: false,
            validate_gates: false,
            ..Default::default()
        };

        Self::parse_with_config(source, &config)
    }

    /// Parse QASM source string without preprocessing includes
    ///
    /// This follows recursive descent principles:
    /// - Clear top-down parsing structure
    /// - Each grammar rule has a corresponding parse function
    /// - Direct AST construction
    ///
    /// # Errors
    ///
    /// Returns an error if parsing fails.
    pub fn parse_str_raw(source: &str) -> Result<Program, PecosError> {
        // Parse with Pest
        let mut pairs = Self::parse_pest(Rule::program, source)?;
        let program_pair = pairs.next().ok_or_else(|| Self::error("Empty program"))?;

        // Build program using recursive descent style
        let mut program = Self::build_program(program_pair)?;

        // Post-processing: expand gates
        expand_gates(&mut program)?;
        Ok(program)
    }

    /// Parse using Pest and convert errors
    fn parse_pest(
        rule: Rule,
        source: &str,
    ) -> Result<pest::iterators::Pairs<'_, Rule>, PecosError> {
        <Self as pest::Parser<Rule>>::parse(rule, source).map_err(|e| {
            // Extract line/column information if available
            let (line, col) = match e.line_col {
                pest::error::LineColLocation::Pos((l, c)) => (Some(l), Some(c)),
                pest::error::LineColLocation::Span((l1, _), _) => (Some(l1), None),
            };

            let mut message = e.to_string();
            if let (Some(l), Some(c)) = (line, col) {
                message = format!("at line {l}, column {c}: {message}");
            }

            PecosError::ParseSyntax {
                language: "QASM".to_string(),
                message,
            }
        })
    }

    /// Build program from parsed pairs (recursive descent style)
    fn build_program(program_pair: Pair<Rule>) -> Result<Program, PecosError> {
        let mut program = Program::default();

        for pair in program_pair.into_inner() {
            match pair.as_rule() {
                Rule::oqasm => Self::parse_version_declaration(pair, &mut program)?,
                Rule::statement => parse_statement(pair, &mut program)?,
                Rule::EOI => break,
                _ => {} // Skip other rules
            }
        }

        Ok(program)
    }

    /// Parse OPENQASM version declaration
    fn parse_version_declaration(
        pair: Pair<Rule>,
        program: &mut Program,
    ) -> Result<(), PecosError> {
        for inner in pair.into_inner() {
            if inner.as_rule() == Rule::version_num {
                let version = inner.as_str();
                if version != "2.0" {
                    return Err(PecosError::ParseInvalidVersion {
                        language: "QASM".to_string(),
                        version: format!("Unsupported version: {version} (only 2.0 is supported)"),
                    });
                }
                program.version = version.to_string();
            }
        }
        Ok(())
    }
}
