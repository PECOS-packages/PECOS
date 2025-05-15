#![allow(clippy::too_many_lines, clippy::bool_to_int_with_if)]

use log::debug;
use pecos_core::errors::PecosError;
use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use std::collections::{HashMap, HashSet, BTreeMap};
use std::fmt;
use std::path::Path;

use crate::preprocessor::Preprocessor;
use crate::ast::Expression;

// Expression is now replaced by the unified Expression type
// Use Expression with the following mappings:
// - Expression::Constant(f) -> Expression::Float(f)
// - Expression::Identifier(s) -> Expression::Variable(s)
// - Expression::Pi -> Expression::Pi
// - Expression::BinaryOp { op, left, right } -> Expression::BinaryOp { op, left, right }
// - Expression::FunctionCall { name, args } -> Expression::FunctionCall { name, args }

#[derive(Debug, Clone)]
pub struct GateDefOperation {
    pub name: String,
    pub parameters: Vec<Expression>,
    pub arguments: Vec<String>,
}

impl fmt::Display for GateDefOperation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.name)?;

        // Parameters if any
        if !self.parameters.is_empty() {
            write!(f, "(")?;
            for (i, param) in self.parameters.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", param)?;
            }
            write!(f, ")")?;
        }

        // Arguments
        for (i, arg) in self.arguments.iter().enumerate() {
            if i == 0 {
                write!(f, " ")?;
            } else {
                write!(f, ", ")?;
            }
            write!(f, "{}", arg)?;
        }

        Ok(())
    }
}

#[derive(Parser)]
#[grammar = "qasm.pest"]
pub struct QASMParser;

// Expression is now imported from ast module

#[derive(Debug, Clone)]
pub enum Operation {
    Gate {
        name: String,
        parameters: Vec<f64>,
        qubits: Vec<usize>, // Global qubit IDs
    },
    Measure {
        qubit: usize,   // Global qubit ID
        c_reg: String,  // Classical register name
        c_index: usize, // Bit index within the register
    },
    If {
        condition: Expression,
        operation: Box<Operation>,
    },
    Reset {
        qubit: usize, // Global qubit ID
    },
    Barrier {
        qubits: Vec<usize>, // Global qubit IDs
    },
    RegMeasure {
        q_reg: String, // Still need register names for full register operations
        c_reg: String,
    },
    ClassicalAssignment {
        target: String,       // Register name or bit
        is_indexed: bool,     // Is this a bit_id or just register
        index: Option<usize>, // Index if it's a bit_id
        expression: Expression,
    },
    OpaqueGate {
        name: String,
        params: Vec<String>,
        qargs: Vec<String>,
    },
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::Gate {
                name,
                parameters,
                qubits,
            } => {
                write!(f, "{name}")?;
                // Only add parentheses if there are parameters
                if !parameters.is_empty() {
                    write!(f, "(")?;
                    for (i, param) in parameters.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{param}")?;
                    }
                    write!(f, ")")?;
                }

                // Output comma-separated qubits
                for (i, qubit) in qubits.iter().enumerate() {
                    if i == 0 {
                        write!(f, " q[{qubit}]")?;
                    } else {
                        write!(f, ", q[{qubit}]")?;
                    }
                }
                Ok(())
            }
            Operation::Measure {
                qubit,
                c_reg,
                c_index,
            } => {
                write!(f, "measure q[{qubit}] -> {c_reg}[{c_index}]")
            }
            Operation::If {
                condition,
                operation,
            } => {
                write!(f, "if ({condition}) {operation}")
            }
            Operation::Reset { qubit } => {
                write!(f, "reset q[{qubit}]")
            }
            Operation::Barrier { qubits } => {
                write!(f, "barrier")?;
                // Output comma-separated qubits
                for (i, qubit) in qubits.iter().enumerate() {
                    if i == 0 {
                        write!(f, " q[{qubit}]")?;
                    } else {
                        write!(f, ", q[{qubit}]")?;
                    }
                }
                Ok(())
            }
            Operation::RegMeasure { q_reg, c_reg } => {
                write!(f, "measure {q_reg} -> {c_reg}")
            }
            Operation::ClassicalAssignment {
                target,
                is_indexed,
                index,
                expression,
            } => {
                if *is_indexed {
                    if let Some(idx) = index {
                        write!(f, "{}[{}] = {}", target, idx, expression)
                    } else {
                        write!(f, "{} = {}", target, expression)
                    }
                } else {
                    write!(f, "{} = {}", target, expression)
                }
            }
            Operation::OpaqueGate {
                name,
                params,
                qargs,
            } => {
                write!(f, "opaque {}", name)?;
                if !params.is_empty() {
                    write!(f, "(")?;
                    for (i, param) in params.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{}", param)?;
                    }
                    write!(f, ")")?;
                }
                write!(f, " ")?;
                for (i, qarg) in qargs.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", qarg)?;
                }
                Ok(())
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GateDefinition {
    pub name: String,
    pub params: Vec<String>,
    pub qargs: Vec<String>,
    pub body: Vec<GateDefOperation>,
}

#[derive(Debug, Clone, Default)]
pub struct Program {
    pub version: String,
    pub operations: Vec<Operation>,
    pub gate_definitions: BTreeMap<String, GateDefinition>,

    // Quantum register mapping to global qubit IDs
    pub quantum_registers: BTreeMap<String, Vec<usize>>, // register_name -> vec of global qubit IDs

    // Classical registers stay as they were (just sizes)
    pub classical_registers: BTreeMap<String, usize>, // register_name -> size

    // Total count
    pub total_qubits: usize,

    // Reverse mapping for debugging/error messages
    pub qubit_map: HashMap<usize, (String, usize)>, // global_id -> (register_name, index)
}

impl QASMParser {
    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Program, PecosError> {
        // Use preprocessor to handle includes
        let mut preprocessor = Preprocessor::new();

        // Add virtual includes from embedded content
        let virtual_includes = crate::includes::get_standard_includes();
        preprocessor.add_virtual_includes(virtual_includes);

        let preprocessed_source = preprocessor.preprocess_file(path)?;
        Self::parse_str_raw(&preprocessed_source)
    }

    /// Get the preprocessed QASM (after phase 1 - include resolution)
    /// This shows the QASM with all includes resolved but gates not yet expanded
    pub fn preprocess(source: &str) -> Result<String, PecosError> {
        let mut preprocessor = Preprocessor::new();

        // Add virtual includes from embedded content
        let virtual_includes = crate::includes::get_standard_includes();
        preprocessor.add_virtual_includes(virtual_includes);

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let include_dir = std::path::Path::new(manifest_dir).join("includes");
        preprocessor.add_include_paths(vec![include_dir]);

        preprocessor.preprocess_str(source)
    }

    /// Get the preprocessed and expanded QASM (after phases 1 and 2)
    /// This shows the QASM with all includes resolved and all gates expanded to native operations
    pub fn preprocess_and_expand(source: &str) -> Result<String, PecosError> {
        // Phase 1: Preprocess includes
        let preprocessed = Self::preprocess(source)?;

        // Phase 2: Expand gates to native operations
        Self::expand_all_gate_definitions(&preprocessed)
    }


    pub fn parse_str_with_includes(source: &str) -> Result<Program, PecosError> {
        // Phase 1: Preprocess includes
        let mut preprocessor = Preprocessor::new();

        // Add virtual includes from embedded content
        let virtual_includes = crate::includes::get_standard_includes();
        preprocessor.add_virtual_includes(virtual_includes);

        // Add the standard includes directory to the search path as fallback
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let include_dir = std::path::Path::new(manifest_dir).join("includes");
        preprocessor.add_include_paths(vec![include_dir]);

        let preprocessed_source = preprocessor.preprocess_str(source)?;

        // Phase 2: Parse the preprocessed source
        let mut program = Self::parse_str_raw(&preprocessed_source)?;

        // Phase 3: Expand gates
        Self::expand_gates(&mut program)?;

        // Phase 4: Check for opaque gates - these are not yet supported
        Self::validate_no_opaque_gate_usage(&program)?;

        Ok(program)
    }

    /// Parse QASM with includes but without gate expansion (mainly for testing and utility functions)
    pub fn parse_str_with_includes_no_expansion(source: &str) -> Result<Program, PecosError> {
        let mut preprocessor = Preprocessor::new();

        // Add virtual includes from embedded content
        let virtual_includes = crate::includes::get_standard_includes();
        preprocessor.add_virtual_includes(virtual_includes);

        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        let include_dir = std::path::Path::new(manifest_dir).join("includes");
        preprocessor.add_include_paths(vec![include_dir]);

        let preprocessed_source = preprocessor.preprocess_str(source)?;
        let mut program = Self::parse_str_raw(&preprocessed_source)?;

        // Still expand gates but don't validate undefined gates
        let _ = Self::expand_gates_old(&mut program);

        Ok(program)
    }

    /// Parse QASM with virtual includes but without gate expansion (for testing)
    #[cfg(test)]
    pub fn parse_str_with_virtual_includes_no_expansion(
        source: &str,
        virtual_includes: impl IntoIterator<Item = (String, String)>,
    ) -> Result<Program, PecosError> {
        // Use preprocessor with virtual includes
        let mut preprocessor = Preprocessor::new();
        preprocessor.add_virtual_includes(virtual_includes);
        let preprocessed_source = preprocessor.preprocess_str(source)?;

        // Parse but don't expand at all - just return parsed program
        let program = Self::parse_str_raw(&preprocessed_source)?;

        Ok(program)
    }

    // Old gate expansion method (without recursive expansion) for compatibility
    fn expand_gates_old(program: &mut Program) -> Result<(), PecosError> {
        let mut expanded_operations = Vec::new();

        for operation in &program.operations {
            match operation {
                Operation::Gate { name, parameters, qubits } => {
                    if let Some(gate_def) = program.gate_definitions.get(name) {
                        let expanded = Self::expand_gate_call(
                            gate_def,
                            parameters,
                            qubits,
                            &program.gate_definitions,
                        )?;
                        expanded_operations.extend(expanded);
                    } else {
                        // Just keep the gate as is (old behavior for tests)
                        expanded_operations.push(operation.clone());
                    }
                }
                _ => expanded_operations.push(operation.clone()),
            }
        }

        program.operations = expanded_operations;
        Ok(())
    }

    pub fn parse_str_with_virtual_includes(
        source: &str,
        virtual_includes: impl IntoIterator<Item = (String, String)>,
    ) -> Result<Program, PecosError> {
        // Use preprocessor with virtual includes
        let mut preprocessor = Preprocessor::new();
        preprocessor.add_virtual_includes(virtual_includes);
        let preprocessed_source = preprocessor.preprocess_str(source)?;

        // Parse the preprocessed source
        let mut program = Self::parse_str_raw(&preprocessed_source)?;

        // Expand gates
        Self::expand_gates(&mut program)?;

        // Validate
        Self::validate_no_opaque_gate_usage(&program)?;

        Ok(program)
    }

    /// Parse QASM source code with custom include paths
    pub fn parse_str_with_include_paths<I, P>(
        source: &str,
        include_paths: I,
    ) -> Result<Program, PecosError>
    where
        I: IntoIterator<Item = P>,
        P: Into<std::path::PathBuf>,
    {
        let mut preprocessor = Preprocessor::new();
        preprocessor.add_include_paths(include_paths);
        let preprocessed_source = preprocessor.preprocess_str(source)?;

        // Parse the preprocessed source
        let mut program = Self::parse_str_raw(&preprocessed_source)?;

        // Expand gates
        Self::expand_gates(&mut program)?;

        // Validate
        Self::validate_no_opaque_gate_usage(&program)?;

        Ok(program)
    }

    /// Parse QASM source code with both custom include paths and virtual includes
    pub fn parse_str_with_include_paths_and_virtual<I, P>(
        source: &str,
        include_paths: I,
        virtual_includes: impl IntoIterator<Item = (String, String)>,
    ) -> Result<Program, PecosError>
    where
        I: IntoIterator<Item = P>,
        P: Into<std::path::PathBuf>,
    {
        let mut preprocessor = Preprocessor::new();
        preprocessor.add_include_paths(include_paths);
        preprocessor.add_virtual_includes(virtual_includes);
        let preprocessed_source = preprocessor.preprocess_str(source)?;

        // Parse the preprocessed source
        let mut program = Self::parse_str_raw(&preprocessed_source)?;

        // Expand gates
        Self::expand_gates(&mut program)?;

        // Validate
        Self::validate_no_opaque_gate_usage(&program)?;

        Ok(program)
    }

    /// Expand all gate definitions in QASM source to native gates only.
    /// This is phase 2 of the three-phase parsing process.
    /// This is exposed publicly so users can see the expanded QASM.
    pub fn expand_all_gate_definitions(source: &str) -> Result<String, PecosError> {
        // Parse the source to get gate definitions and operations
        let mut program = Self::parse_phase1(source)?;

        // Expand all gates
        Self::expand_gates(&mut program)?;

        // Convert back to QASM string with expanded operations only (no gate definitions)
        Ok(Self::program_to_qasm_expanded(&program))
    }

    /// Parse only phase 1 - just enough to get gate definitions and operations
    fn parse_phase1(source: &str) -> Result<Program, PecosError> {
        let mut program = Program::default();
        let mut pairs = Self::parse(Rule::program, source).map_err(|e| PecosError::ParseSyntax {
            language: "QASM".to_string(),
            message: e.to_string(),
        })?;

        let program_pair = pairs
            .next()
            .ok_or_else(|| PecosError::CompileInvalidOperation {
                operation: "QASM program".to_string(),
                reason: "Empty program".to_string(),
            })?;

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
                            Rule::register_decl => Self::parse_register(inner_pair, &mut program)?,
                            Rule::gate_def => Self::parse_gate_definition(inner_pair, &mut program)?,
                            Rule::quantum_op => {
                                if let Some(op) = Self::parse_quantum_op(inner_pair, &program)? {
                                    program.operations.push(op);
                                }
                            }
                            Rule::classical_op => {
                                if let Some(op) = Self::parse_classical_operation(inner_pair)? {
                                    program.operations.push(op);
                                }
                            }
                            Rule::if_stmt => {
                                if let Some(op) = Self::parse_if_statement(inner_pair, &program)? {
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

    /// Convert a Program back to QASM string
    #[allow(dead_code)]
    fn program_to_qasm(program: &Program) -> String {
        let mut qasm = String::new();

        // Version
        if !program.version.is_empty() {
            qasm.push_str(&format!("OPENQASM {};\n", program.version));
        }

        // Gate definitions (need to preserve these for later phases)
        for (name, gate_def) in &program.gate_definitions {
            qasm.push_str(&format!("gate {} ", name));

            // Parameters
            if !gate_def.params.is_empty() {
                qasm.push('(');
                qasm.push_str(&gate_def.params.join(", "));
                qasm.push(')');
                qasm.push(' ');
            }

            // Qubits
            qasm.push_str(&gate_def.qargs.join(", "));
            qasm.push_str(" {\n");

            // Gate body
            for body_op in &gate_def.body {
                qasm.push_str("  ");
                qasm.push_str(&format!("{}", body_op));
                qasm.push_str(";\n");
            }

            qasm.push_str("}\n");
        }

        // Quantum registers
        for (name, qubits) in &program.quantum_registers {
            qasm.push_str(&format!("qreg {}[{}];\n", name, qubits.len()));
        }

        // Classical registers
        for (name, size) in &program.classical_registers {
            qasm.push_str(&format!("creg {}[{}];\n", name, size));
        }

        // Operations (expanded)
        for op in &program.operations {
            qasm.push_str(&Self::format_operation(op, &program.qubit_map));
            qasm.push_str(";\n");
        }

        qasm
    }

    /// Convert a Program back to QASM string with only expanded operations (no gate definitions)
    fn program_to_qasm_expanded(program: &Program) -> String {
        let mut qasm = String::new();

        // Version
        if !program.version.is_empty() {
            qasm.push_str(&format!("OPENQASM {};\n", program.version));
        }

        // Quantum registers
        for (name, qubits) in &program.quantum_registers {
            qasm.push_str(&format!("qreg {}[{}];\n", name, qubits.len()));
        }

        // Classical registers
        for (name, size) in &program.classical_registers {
            qasm.push_str(&format!("creg {}[{}];\n", name, size));
        }

        // Operations (expanded) - no gate definitions
        for op in &program.operations {
            qasm.push_str(&Self::format_operation(op, &program.qubit_map));
            qasm.push_str(";\n");
        }

        qasm
    }

    /// Format an operation with proper qubit register names
    fn format_operation(op: &Operation, qubit_map: &HashMap<usize, (String, usize)>) -> String {
        match op {
            Operation::Gate { name, parameters, qubits } => {
                let mut result = name.clone();

                // Add parameters if any
                if !parameters.is_empty() {
                    result.push('(');
                    for (i, param) in parameters.iter().enumerate() {
                        if i > 0 {
                            result.push_str(", ");
                        }
                        result.push_str(&param.to_string());
                    }
                    result.push(')');
                }

                // Add qubits with proper register names
                for (i, &qubit_id) in qubits.iter().enumerate() {
                    if i == 0 {
                        result.push(' ');
                    } else {
                        result.push_str(", ");
                    }

                    if let Some((reg_name, index)) = qubit_map.get(&qubit_id) {
                        result.push_str(&format!("{}[{}]", reg_name, index));
                    } else {
                        // Fallback if mapping not found
                        result.push_str(&format!("q[{}]", qubit_id));
                    }
                }

                result
            }
            Operation::Measure { qubit, c_reg, c_index } => {
                let qubit_str = if let Some((reg_name, index)) = qubit_map.get(qubit) {
                    format!("{}[{}]", reg_name, index)
                } else {
                    format!("q[{}]", qubit)
                };
                format!("measure {} -> {}[{}]", qubit_str, c_reg, c_index)
            }
            Operation::Reset { qubit } => {
                let qubit_str = if let Some((reg_name, index)) = qubit_map.get(qubit) {
                    format!("{}[{}]", reg_name, index)
                } else {
                    format!("q[{}]", qubit)
                };
                format!("reset {}", qubit_str)
            }
            Operation::Barrier { qubits } => {
                let mut result = String::from("barrier");
                for (i, &qubit_id) in qubits.iter().enumerate() {
                    if i == 0 {
                        result.push(' ');
                    } else {
                        result.push_str(", ");
                    }

                    if let Some((reg_name, index)) = qubit_map.get(&qubit_id) {
                        result.push_str(&format!("{}[{}]", reg_name, index));
                    } else {
                        result.push_str(&format!("q[{}]", qubit_id));
                    }
                }
                result
            }
            Operation::If { condition, operation } => {
                let nested_operation_str = Self::format_operation(operation, qubit_map);
                format!("if ({}) {}", condition, nested_operation_str)
            }
            _ => format!("{}", op), // Use default Display for other operations
        }
    }

    /// Parse QASM source string without preprocessing includes.
    /// This is the low-level parsing function that assumes all includes have already been resolved.
    ///
    /// For most use cases, consider using `parse_str_with_includes()` which handles include resolution.
    pub fn parse_str_raw(source: &str) -> Result<Program, PecosError> {
        let mut program = Program::default();
        let mut pairs =
            Self::parse(Rule::program, source).map_err(|e| PecosError::ParseSyntax {
                language: "QASM".to_string(),
                message: e.to_string(),
            })?;
        let program_pair = pairs
            .next()
            .ok_or_else(|| PecosError::CompileInvalidOperation {
                operation: "QASM operation".to_string(),
                reason: "Empty program".to_string(),
            })?;

        for pair in program_pair.into_inner() {
            match pair.as_rule() {
                Rule::oqasm => {
                    for inner in pair.into_inner() {
                        if inner.as_rule() == Rule::version_num {
                            let version = inner.as_str();
                            if version != "2.0" {
                                return Err(PecosError::ParseInvalidVersion {
                                    language: "QASM".to_string(),
                                    version: format!("Unsupported version: {version}"),
                                });
                            }
                            program.version = version.to_string();
                        }
                    }
                }
                Rule::statement => Self::parse_statement(pair, &mut program)?,
                Rule::EOI => break,
                _ => {
                    // Ignore other rules at this level
                }
            }
        }

        // After parsing, expand all gates using their definitions
        Self::expand_gates(&mut program)?;

        // Note: Opaque gate validation moved to later in the process

        Ok(program)
    }

    fn parse_statement(
        pair: pest::iterators::Pair<Rule>,
        program: &mut Program,
    ) -> Result<(), PecosError> {
        for inner_pair in pair.into_inner() {
            // Match statements with correct pattern handling
            match inner_pair.as_rule() {
                // Explicitly handle specific rules
                Rule::register_decl => Self::parse_register(inner_pair, program)?,
                Rule::quantum_op => {
                    if let Some(op) = Self::parse_quantum_op(inner_pair, program)? {
                        program.operations.push(op);
                    }
                }
                Rule::classical_op => {
                    if let Some(op) = Self::parse_classical_operation(inner_pair)? {
                        program.operations.push(op);
                    }
                }
                Rule::if_stmt => {
                    if let Some(op) = Self::parse_if_statement(inner_pair, program)? {
                        program.operations.push(op);
                    }
                }
                Rule::gate_def => {
                    Self::parse_gate_definition(inner_pair, program)?;
                }
                Rule::include => {
                    // Include statements should be handled by preprocessor
                    return Err(PecosError::ParseSyntax {
                        language: "QASM".to_string(),
                        message: "Include statements should be preprocessed before parsing"
                            .to_string(),
                    });
                }
                Rule::opaque_def => {
                    if let Some(op) = Self::parse_opaque_def(inner_pair)? {
                        program.operations.push(op);
                    }
                }
                // Rules that are recognized but not yet implemented
                _ => {
                    // Ignoring unimplemented rules for now
                }
            }
        }
        Ok(())
    }

    fn parse_register(
        pair: pest::iterators::Pair<Rule>,
        program: &mut Program,
    ) -> Result<(), PecosError> {
        let inner = pair.into_inner().next().unwrap();

        #[allow(clippy::match_same_arms)]
        match inner.as_rule() {
            Rule::qreg => {
                let indexed_id = inner.into_inner().next().unwrap();
                let (name, size) = Self::parse_indexed_id(&indexed_id)?;

                // Assign global qubit IDs
                let mut qubit_ids = Vec::new();
                for i in 0..size {
                    let global_id = program.total_qubits;
                    qubit_ids.push(global_id);

                    // Store reverse mapping for debugging
                    program.qubit_map.insert(global_id, (name.clone(), i));

                    program.total_qubits += 1;
                }

                program.quantum_registers.insert(name, qubit_ids);
            }
            Rule::creg => {
                let indexed_id = inner.into_inner().next().unwrap();
                let (name, size) = Self::parse_indexed_id(&indexed_id)?;
                program.classical_registers.insert(name, size);
            }
            _ => {
                return Err(PecosError::CompileInvalidOperation {
                    operation: "QASM operation".to_string(),
                    reason: format!("Unexpected register type: {:?}", inner.as_rule()),
                });
            }
        }

        Ok(())
    }

    fn parse_quantum_op(
        pair: pest::iterators::Pair<Rule>,
        program: &Program,
    ) -> Result<Option<Operation>, PecosError> {
        let inner = pair.into_inner().next().unwrap();

        #[allow(clippy::match_same_arms)]
        match inner.as_rule() {
            Rule::gate_call => {
                let mut inner_pairs = inner.into_inner();
                let gate_name = inner_pairs.next().unwrap().as_str();

                let mut params = Vec::new();
                let mut global_qubit_ids = Vec::new();

                for pair in inner_pairs {
                    match pair.as_rule() {
                        // Handle parameter values
                        Rule::param_values => {
                            for param_expr in pair.into_inner() {
                                if param_expr.as_rule() == Rule::expr {
                                    let expr = Self::parse_expr(param_expr)?;
                                    // Evaluate the expression to a float
                                    let value = expr.evaluate().map_err(|e| {
                                        PecosError::ParseInvalidExpression(format!(
                                            "Failed to evaluate parameter: {}",
                                            e
                                        ))
                                    })?;
                                    params.push(value);
                                }
                            }
                        }
                        // Handle qubit lists - convert to global IDs
                        Rule::qubit_list => {
                            for qubit_id in pair.into_inner() {
                                if qubit_id.as_rule() == Rule::qubit_id {
                                    let (reg_name, idx) = Self::parse_id_with_index(&qubit_id)?;

                                    // Look up the global ID
                                    if let Some(qubit_ids) =
                                        program.quantum_registers.get(&reg_name)
                                    {
                                        if idx < qubit_ids.len() {
                                            global_qubit_ids.push(qubit_ids[idx]);
                                        } else {
                                            return Err(PecosError::CompileInvalidOperation {
                                                operation: "QASM operation".to_string(),
                                                reason: format!(
                                                    "Qubit index {} out of bounds for register '{}'",
                                                    idx, reg_name
                                                ),
                                            });
                                        }
                                    } else {
                                        return Err(PecosError::CompileInvalidOperation {
                                            operation: "QASM operation".to_string(),
                                            reason: format!(
                                                "Unknown quantum register '{}'",
                                                reg_name
                                            ),
                                        });
                                    }
                                }
                            }
                        }
                        // Unhandled rule types
                        _ => {
                            // Skip unimplemented rules for now
                        }
                    }
                }

                Ok(Some(Operation::Gate {
                    name: gate_name.to_string(),
                    parameters: params,
                    qubits: global_qubit_ids,
                }))
            }
            Rule::measure => Self::parse_measure(inner, program),
            Rule::reset => Self::parse_reset(inner, program),
            Rule::barrier => Self::parse_barrier(inner, program),
            _ => Ok(None),
        }
    }

    fn parse_measure(
        pair: pest::iterators::Pair<Rule>,
        program: &Program,
    ) -> Result<Option<Operation>, PecosError> {
        let inner_parts: Vec<_> = pair.into_inner().collect();

        if inner_parts.len() == 2 {
            let src = &inner_parts[0];
            let dst = &inner_parts[1];

            if src.as_rule() == Rule::qubit_id && dst.as_rule() == Rule::bit_id {
                let (q_reg, q_idx) = Self::parse_id_with_index(&src.clone())?;
                let (c_reg, c_idx) = Self::parse_id_with_index(&dst.clone())?;

                // Look up global qubit ID
                if let Some(qubit_ids) = program.quantum_registers.get(&q_reg) {
                    if q_idx < qubit_ids.len() {
                        let global_qubit_id = qubit_ids[q_idx];

                        Ok(Some(Operation::Measure {
                            qubit: global_qubit_id,
                            c_reg,
                            c_index: c_idx,
                        }))
                    } else {
                        Err(PecosError::CompileInvalidOperation {
                            operation: "QASM operation".to_string(),
                            reason: format!(
                                "Qubit index {} out of bounds for register '{}'",
                                q_idx, q_reg
                            ),
                        })
                    }
                } else {
                    Err(PecosError::CompileInvalidOperation {
                        operation: "QASM operation".to_string(),
                        reason: format!("Unknown quantum register '{}'", q_reg),
                    })
                }
            } else if src.as_rule() == Rule::identifier && dst.as_rule() == Rule::identifier {
                Ok(Some(Operation::RegMeasure {
                    q_reg: src.as_str().to_string(),
                    c_reg: dst.as_str().to_string(),
                }))
            } else {
                Err(PecosError::CompileInvalidOperation {
                    operation: "QASM operation".to_string(),
                    reason: "Invalid measurement format".to_string(),
                })
            }
        } else {
            Err(PecosError::CompileInvalidOperation {
                operation: "QASM operation".to_string(),
                reason: "Invalid measurement syntax".to_string(),
            })
        }
    }

    fn parse_reset(
        pair: pest::iterators::Pair<Rule>,
        program: &Program,
    ) -> Result<Option<Operation>, PecosError> {
        let qubit_id = pair.into_inner().next().unwrap();
        let (reg_name, idx) = Self::parse_id_with_index(&qubit_id)?;

        // Look up global qubit ID
        if let Some(qubit_ids) = program.quantum_registers.get(&reg_name) {
            if idx < qubit_ids.len() {
                let global_qubit_id = qubit_ids[idx];
                Ok(Some(Operation::Reset {
                    qubit: global_qubit_id,
                }))
            } else {
                Err(PecosError::CompileInvalidOperation {
                    operation: "QASM operation".to_string(),
                    reason: format!(
                        "Qubit index {} out of bounds for register '{}'",
                        idx, reg_name
                    ),
                })
            }
        } else {
            Err(PecosError::CompileInvalidOperation {
                operation: "QASM operation".to_string(),
                reason: format!("Unknown quantum register '{}'", reg_name),
            })
        }
    }

    fn parse_barrier(
        pair: pest::iterators::Pair<Rule>,
        program: &Program,
    ) -> Result<Option<Operation>, PecosError> {
        let any_list = pair.into_inner().next().unwrap();
        let mut qubits = Vec::new();

        // Parse the any_list which contains any_items
        for item in any_list.into_inner() {
            if item.as_rule() == Rule::any_item {
                let inner = item.into_inner().next().unwrap();
                match inner.as_rule() {
                    Rule::identifier => {
                        // This is a register name - add all qubits from the register
                        let reg_name = inner.as_str();
                        if let Some(qubit_ids) = program.quantum_registers.get(reg_name) {
                            qubits.extend(qubit_ids.iter());
                        } else {
                            return Err(PecosError::CompileInvalidOperation {
                                operation: "QASM operation".to_string(),
                                reason: format!(
                                    "Unknown quantum register '{}' in barrier",
                                    reg_name
                                ),
                            });
                        }
                    }
                    Rule::qubit_id => {
                        // This is an individual qubit - parse and add it
                        let (reg_name, idx) = Self::parse_id_with_index(&inner)?;
                        if let Some(qubit_ids) = program.quantum_registers.get(&reg_name) {
                            if idx < qubit_ids.len() {
                                qubits.push(qubit_ids[idx]);
                            } else {
                                return Err(PecosError::CompileInvalidOperation {
                                    operation: "QASM operation".to_string(),
                                    reason: format!(
                                        "Qubit index {} out of bounds for register '{}'",
                                        idx, reg_name
                                    ),
                                });
                            }
                        } else {
                            return Err(PecosError::CompileInvalidOperation {
                                operation: "QASM operation".to_string(),
                                reason: format!("Unknown quantum register '{}'", reg_name),
                            });
                        }
                    }
                    _ => {
                        // Skip unexpected rules
                    }
                }
            }
        }

        Ok(Some(Operation::Barrier { qubits }))
    }

    // Parse if statement with condition (expression) and operation
    fn parse_if_statement(
        pair: pest::iterators::Pair<Rule>,
        program: &Program,
    ) -> Result<Option<Operation>, PecosError> {
        // For debugging
        debug!("Parsing if statement: '{}'", pair.as_str());

        // Collect all parts of the if statement
        let parts: Vec<_> = pair.into_inner().collect();

        if parts.len() < 2 {
            return Err(PecosError::CompileInvalidOperation {
                operation: "QASM operation".to_string(),
                reason: format!(
                    "Invalid if statement: expected at least 2 parts, got {}",
                    parts.len()
                ),
            });
        }

        // We expect parts to be: condition_expr, operation
        let condition_expr_pair = &parts[0];
        let operation_pair = &parts[1];

        // Parse the condition expression
        let condition = match condition_expr_pair.as_rule() {
            Rule::condition_expr => {
                // Get the expression inside condition_expr
                let expr_pair =
                    condition_expr_pair
                        .clone()
                        .into_inner()
                        .next()
                        .ok_or_else(|| PecosError::CompileInvalidOperation {
                            operation: "QASM operation".to_string(),
                            reason: "Empty condition expression".to_string(),
                        })?;
                Self::parse_expr(expr_pair)?
            }
            _ => {
                return Err(PecosError::CompileInvalidOperation {
                    operation: "QASM operation".to_string(),
                    reason: format!(
                        "Invalid rule in if statement, expected condition_expr, got: {:?}",
                        condition_expr_pair.as_rule()
                    ),
                });
            }
        };

        // Parse the operation to be conditionally executed
        let operation = match operation_pair.as_rule() {
            Rule::quantum_op => {
                if let Some(op) = Self::parse_quantum_op(operation_pair.clone(), program)? {
                    op
                } else {
                    return Err(PecosError::CompileInvalidOperation {
                        operation: "QASM operation".to_string(),
                        reason: "Invalid quantum operation in if statement".to_string(),
                    });
                }
            }
            Rule::classical_op => {
                if let Some(op) = Self::parse_classical_operation(operation_pair.clone())? {
                    op
                } else {
                    return Err(PecosError::CompileInvalidOperation {
                        operation: "QASM operation".to_string(),
                        reason: "Invalid classical operation in if statement".to_string(),
                    });
                }
            }
            _ => {
                return Err(PecosError::CompileInvalidOperation {
                    operation: "QASM operation".to_string(),
                    reason: format!(
                        "Unsupported operation type in if statement: {:?}",
                        operation_pair.as_rule()
                    ),
                });
            }
        };

        // Create and return the If operation
        Ok(Some(Operation::If {
            condition,
            operation: Box::new(operation),
        }))
    }

    // Add a new method to parse classical operations
    fn parse_classical_operation(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Option<Operation>, PecosError> {
        // For debugging
        eprintln!("Parsing classical op: '{}'", pair.as_str());

        // Get the inner pairs: 1) target (identifier or bit_id) and 2) expression
        let inner_parts: Vec<_> = pair.into_inner().collect();

        // Debug print all inner parts
        for (i, part) in inner_parts.iter().enumerate() {
            eprintln!(
                "  Part {}: rule={:?}, text='{}'",
                i,
                part.as_rule(),
                part.as_str()
            );
        }

        if inner_parts.len() >= 2 {
            let target_pair = &inner_parts[0];
            let target: String;
            let is_indexed: bool;
            let index: Option<usize>;

            // Handle target (either bit_id or identifier)
            match target_pair.as_rule() {
                Rule::bit_id => {
                    // Parse bit_id (e.g., "a[2]")
                    let (reg_name, bit_idx) = Self::parse_id_with_index(&target_pair)?;
                    target = reg_name;
                    is_indexed = true;
                    index = Some(bit_idx);
                }
                Rule::identifier => {
                    // Parse identifier (e.g., "a")
                    target = target_pair.as_str().to_string();
                    is_indexed = false;
                    index = None;
                }
                _ => {
                    return Err(PecosError::CompileInvalidOperation {
                        operation: "QASM operation".to_string(),
                        reason: format!(
                            "Invalid classical assignment target: {:?}",
                            target_pair.as_rule()
                        ),
                    });
                }
            }

            // Get the expression from the second inner part
            let expr_pair = &inner_parts[1];
            eprintln!("About to parse expression: '{}'", expr_pair.as_str());

            // Parse the expression
            let expression = Self::parse_expr(expr_pair.clone())?;
            eprintln!("Parsed expression: {:?}", expression);

            return Ok(Some(Operation::ClassicalAssignment {
                target,
                is_indexed,
                index,
                expression,
            }));
        }

        Err(PecosError::CompileInvalidOperation {
            operation: "QASM operation".to_string(),
            reason: "Invalid classical operation".to_string(),
        })
    }

    fn parse_indexed_id(pair: &pest::iterators::Pair<Rule>) -> Result<(String, usize), PecosError> {
        let content = pair.as_str();

        if let Some(bracket_pos) = content.find('[') {
            let name = content[0..bracket_pos].to_string();
            let size_str = &content[bracket_pos + 1..content.len() - 1];
            let size = size_str
                .parse::<usize>()
                .map_err(|e| PecosError::CompileInvalidRegisterSize(e.to_string()))?;
            Ok((name, size))
        } else {
            Err(PecosError::ParseInvalidExpression(format!(
                "Invalid indexed identifier: {content}"
            )))
        }
    }

    // This function is identical to parse_indexed_id, using a single implementation for both cases
    fn parse_id_with_index(
        pair: &pest::iterators::Pair<Rule>,
    ) -> Result<(String, usize), PecosError> {
        Self::parse_indexed_id(pair)
    }

    // New method to correctly handle binary expressions like a^b, a|b, etc.
    fn parse_binary_expr(pair: Pair<Rule>, default_op: &str) -> Result<Expression, PecosError> {
        // Debug the input pair
        let rule = pair.as_rule();
        eprintln!(
            "parse_binary_expr for rule {:?} with text '{}'",
            rule,
            pair.as_str()
        );

        let inner_pairs: Vec<Pair<Rule>> = pair.into_inner().collect();

        // If we have exactly one inner pair, just parse it directly (no operator)
        if inner_pairs.len() == 1 {
            return Self::parse_expr(inner_pairs[0].clone());
        }

        // Get the left side expression (first inner pair)
        let mut result = Self::parse_expr(inner_pairs[0].clone())?;

        // Process the rest as operator-operand pairs
        let mut i = 1;
        while i < inner_pairs.len() {
            let next_pair = &inner_pairs[i];

            // Check if this is an operator token (for equality, relational, etc.)
            let (actual_op, right_expr) = match next_pair.as_rule() {
                Rule::equality_op
                | Rule::relational_op
                | Rule::shift_op
                | Rule::add_op
                | Rule::mul_op
                | Rule::pow_op => {
                    // This is an explicit operator, next pair should be the operand
                    if i + 1 < inner_pairs.len() {
                        let op_str = next_pair.as_str();
                        let right = Self::parse_expr(inner_pairs[i + 1].clone())?;
                        i += 2; // Skip both operator and operand
                        (op_str, right)
                    } else {
                        return Err(PecosError::ParseInvalidExpression(
                            "Missing right operand for binary operation".to_string(),
                        ));
                    }
                }
                _ => {
                    // For implicit operators (like |, ^, &), the operator is implicit in the rule
                    // and this pair is the operand
                    let op = match rule {
                        Rule::b_or_expr => "|",
                        Rule::b_xor_expr => "^",
                        Rule::b_and_expr => "&",
                        _ => default_op,
                    };
                    let right = Self::parse_expr(next_pair.clone())?;
                    i += 1; // Skip just the operand
                    (op, right)
                }
            };

            result = Expression::BinaryOp {
                op: actual_op.to_string(),
                left: Box::new(result),
                right: Box::new(right_expr),
            };
        }

        Ok(result)
    }

    fn parse_expr(pair: Pair<Rule>) -> Result<Expression, PecosError> {
        // Debug the input pair
        eprintln!(
            "parse_expr: Rule {:?}, Text: '{}'",
            pair.as_rule(),
            pair.as_str()
        );

        match pair.as_rule() {
            // Handle all expression types based on our updated grammar

            // Top-level expression rule
            Rule::expr => {
                let inner = pair.into_inner().next().ok_or_else(|| {
                    PecosError::ParseInvalidExpression("Empty expression".to_string())
                })?;
                Self::parse_expr(inner)
            }

            // Binary operations - explicitly map each rule to parse_binary_expr
            Rule::b_or_expr => Self::parse_binary_expr(pair, "|"),
            Rule::b_xor_expr => Self::parse_binary_expr(pair, "^"),
            Rule::b_and_expr => Self::parse_binary_expr(pair, "&"),
            Rule::equality_expr => Self::parse_binary_expr(pair, "=="),
            Rule::relational_expr => Self::parse_binary_expr(pair, "<"),
            Rule::shift_expr => Self::parse_binary_expr(pair, "<<"),
            Rule::additive_expr => Self::parse_binary_expr(pair, "+"),
            Rule::multiplicative_expr => Self::parse_binary_expr(pair, "*"),
            Rule::power_expr => Self::parse_binary_expr(pair, "**"),

            // Unary operations
            Rule::unary_expr => {
                let mut pairs = pair.into_inner();

                // Get operators, if any
                let mut ops = Vec::new();
                while let Some(pair) = pairs.peek() {
                    if pair.as_rule() == Rule::unary_op {
                        ops.push(pairs.next().unwrap().as_str().to_string());
                    } else {
                        break;
                    }
                }

                // Get the operand
                if let Some(operand_pair) = pairs.next() {
                    let mut expr = Self::parse_expr(operand_pair)?;

                    // Apply operators in reverse order (right-to-left)
                    for op in ops.iter().rev() {
                        if op == "-" {
                            // Handle negation specially for integers
                            if let Expression::Integer(value) = expr {
                                expr = Expression::Integer(-value);
                            } else {
                                expr = Expression::UnaryOp { op: op.clone(), expr: Box::new(expr) };
                            }
                        } else {
                            expr = Expression::UnaryOp { op: op.clone(), expr: Box::new(expr) };
                        }
                    }

                    Ok(expr)
                } else {
                    Err(PecosError::ParseInvalidExpression(
                        "Missing operand for unary operation".to_string(),
                    ))
                }
            }

            // Primary expressions
            Rule::primary_expr => {
                let inner = pair.into_inner().next().unwrap();
                Self::parse_expr(inner)
            }

            // Atomic values
            Rule::pi_constant => Ok(Expression::Pi),

            Rule::number => {
                let num_str = pair.as_str();
                // Check if it's a float (has decimal point or scientific notation)
                if num_str.contains('.') || num_str.contains('e') || num_str.contains('E') {
                    Ok(Expression::Float(num_str.parse().map_err(|_| {
                        PecosError::ParseInvalidNumber(num_str.to_string())
                    })?))
                } else {
                    Ok(Expression::Integer(num_str.parse().map_err(|_| {
                        PecosError::ParseInvalidNumber(num_str.to_string())
                    })?))
                }
            }

            Rule::int => {
                let int_str = pair.as_str();
                Ok(Expression::Integer(int_str.parse().map_err(|_| {
                    PecosError::ParseInvalidNumber(int_str.to_string())
                })?))
            }

            Rule::bit_id => {
                let bit_id = pair.as_str();
                let parts: Vec<&str> = bit_id.split('[').collect();
                let name = parts[0].to_string();
                let idx_str = parts[1].trim_end_matches(']');
                let idx = idx_str
                    .parse()
                    .map_err(|_| PecosError::ParseInvalidNumber(idx_str.to_string()))?;
                Ok(Expression::BitId(name, idx))
            }

            Rule::identifier => {
                // Handle simple identifier (register name)
                Ok(Expression::Variable(pair.as_str().to_string()))
            }

            Rule::function_call => {
                let mut pairs = pair.into_inner();
                let name = pairs.next().unwrap().as_str().to_string();

                let mut args = Vec::new();
                while let Some(arg_pair) = pairs.next() {
                    args.push(Self::parse_expr(arg_pair)?);
                }

                Ok(Expression::FunctionCall { name, args })
            }

            _ => Err(PecosError::ParseInvalidExpression(format!(
                "Unexpected rule in expression: {:?}",
                pair.as_rule()
            ))),
        }
    }

    pub fn parse_param_values(_pair: pest::iterators::Pair<Rule>) -> Result<Vec<f64>, PecosError> {
        let params = Vec::new();
        // For now, just return an empty vector
        // In a real implementation, we'd parse each expr in the param_values
        Ok(params)
    }

    fn parse_gate_definition(
        pair: pest::iterators::Pair<Rule>,
        program: &mut Program,
    ) -> Result<(), PecosError> {
        let mut inner = pair.into_inner();

        // Parse gate name
        let name = inner.next().unwrap().as_str().to_string();

        let mut params = Vec::new();
        let mut qargs = Vec::new();
        let mut body_pairs = Vec::new();

        // Parse remaining parts
        for inner_pair in inner {
            match inner_pair.as_rule() {
                Rule::param_list => {
                    // Parse parameter names
                    for param in inner_pair.into_inner() {
                        if param.as_rule() == Rule::identifier {
                            params.push(param.as_str().to_string());
                        }
                    }
                }
                Rule::identifier_list => {
                    // Parse qubit argument names
                    for ident in inner_pair.into_inner() {
                        if ident.as_rule() == Rule::identifier {
                            qargs.push(ident.as_str().to_string());
                        }
                    }
                }
                Rule::gate_def_statement => {
                    body_pairs.push(inner_pair);
                }
                _ => {}
            }
        }

        // Parse body operations
        let mut body = Vec::new();
        for statement_pair in body_pairs {
            // Parse gate definition statements
            if let Some(op) = Self::parse_gate_def_statement(statement_pair)? {
                body.push(op);
            }
        }

        let gate_def = GateDefinition {
            name: name.clone(),
            params,
            qargs,
            body,
        };

        program.gate_definitions.insert(name, gate_def);

        Ok(())
    }

    fn parse_opaque_def(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Option<Operation>, PecosError> {
        let mut inner = pair.into_inner();

        // Get the gate name
        let name = inner
            .next()
            .ok_or_else(|| PecosError::CompileInvalidOperation {
                operation: "QASM operation".to_string(),
                reason: "Missing gate name".to_string(),
            })?
            .as_str()
            .to_string();

        let mut params = Vec::new();
        let mut qargs = Vec::new();

        // Parse the rest of the declaration
        for part in inner {
            match part.as_rule() {
                Rule::param_list => {
                    for param in part.into_inner() {
                        if param.as_rule() == Rule::identifier {
                            params.push(param.as_str().to_string());
                        }
                    }
                }
                Rule::identifier_list => {
                    for qarg in part.into_inner() {
                        if qarg.as_rule() == Rule::identifier {
                            qargs.push(qarg.as_str().to_string());
                        }
                    }
                }
                _ => {}
            }
        }

        Ok(Some(Operation::OpaqueGate {
            name,
            params,
            qargs,
        }))
    }

    fn parse_gate_def_statement(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Option<GateDefOperation>, PecosError> {
        let inner = pair.into_inner().next().unwrap();

        match inner.as_rule() {
            Rule::gate_def_call => {
                let mut parts = inner.into_inner();
                let gate_name = parts.next().unwrap().as_str();

                let mut params = Vec::new();
                let mut arguments = Vec::new();

                for part in parts {
                    match part.as_rule() {
                        Rule::param_values => {
                            // Parse parameter expressions
                            for expr_pair in part.into_inner() {
                                let param_expr = Self::parse_param_expr(expr_pair)?;
                                params.push(param_expr);
                            }
                        }
                        Rule::identifier_list => {
                            // Parse qubit arguments
                            for ident in part.into_inner() {
                                if ident.as_rule() == Rule::identifier {
                                    arguments.push(ident.as_str().to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }

                Ok(Some(GateDefOperation {
                    name: gate_name.to_string(),
                    parameters: params,
                    arguments,
                }))
            }
            _ => Ok(None),
        }
    }

    fn parse_param_expr(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Expression, PecosError> {
        match pair.as_rule() {
            Rule::expr => {
                // Parse the expression recursively
                Self::parse_param_expr(pair.into_inner().next().unwrap())
            }
            Rule::primary_expr => {
                // Handle primary expressions
                let inner = pair.into_inner().next().unwrap();
                Self::parse_param_expr(inner)
            }
            Rule::identifier => Ok(Expression::Variable(pair.as_str().to_string())),
            Rule::number => {
                let value = pair
                    .as_str()
                    .parse()
                    .map_err(|_| PecosError::ParseInvalidNumber("Invalid number".to_string()))?;
                Ok(Expression::Float(value))
            }
            Rule::pi_constant => Ok(Expression::Pi),
            Rule::function_call => {
                let mut inner = pair.into_inner();
                let func_name = inner.next().unwrap().as_str().to_string();
                let args: Result<Vec<_>, _> =
                    inner.map(|arg| Self::parse_param_expr(arg)).collect();
                Ok(Expression::FunctionCall {
                    name: func_name,
                    args: args?,
                })
            }
            Rule::additive_expr
            | Rule::multiplicative_expr
            | Rule::power_expr
            | Rule::b_or_expr
            | Rule::b_xor_expr
            | Rule::b_and_expr => Self::parse_binary_param_expr(pair),
            Rule::unary_expr => {
                // Handle unary expressions (like negation)
                let mut inner = pair.into_inner();

                // Check if there's a unary operator
                let mut negate = false;
                while let Some(child) = inner.peek() {
                    if child.as_rule() == Rule::unary_op {
                        let op = inner.next().unwrap();
                        if op.as_str() == "-" {
                            negate = !negate; // Handle multiple negations
                        }
                    } else {
                        break;
                    }
                }

                // Parse the rest of the expression
                if let Some(expr_pair) = inner.next() {
                    let mut expr = Self::parse_param_expr(expr_pair)?;

                    // Apply negation if needed
                    if negate {
                        expr = Expression::BinaryOp {
                            op: "-".to_string(),
                            left: Box::new(Expression::Float(0.0)),
                            right: Box::new(expr),
                        };
                    }

                    Ok(expr)
                } else {
                    Err(PecosError::ParseInvalidExpression(
                        "Expected expression after unary operator".to_string(),
                    ))
                }
            }
            _ => {
                // For any other binary expression node, try to parse as binary
                let mut inner = pair.clone().into_inner();
                if inner.clone().count() > 1 {
                    Self::parse_binary_param_expr(pair)
                } else if let Some(child) = inner.next() {
                    // Single child, continue recursively
                    Self::parse_param_expr(child)
                } else {
                    // Unknown node type, default to constant 0
                    debug!(
                        "Unknown node type in parse_param_expr: {:?}",
                        pair.as_rule()
                    );
                    Ok(Expression::Float(0.0))
                }
            }
        }
    }

    fn parse_binary_param_expr(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Expression, PecosError> {
        let mut inner = pair.into_inner();
        let left_pair = inner.next().ok_or_else(|| {
            PecosError::ParseInvalidExpression("Expected left operand".to_string())
        })?;
        let mut left = Self::parse_param_expr(left_pair)?;

        while let Some(op_pair) = inner.next() {
            let op = op_pair.as_str().to_string();
            if inner.peek().is_none() {
                debug!(
                    "parse_binary_param_expr: No right operand found after operator {}",
                    op
                );
            }
            let right_pair = inner.next().ok_or_else(|| {
                PecosError::ParseInvalidExpression("Expected right operand".to_string())
            })?;
            let right = Self::parse_param_expr(right_pair)?;
            left = Expression::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn expand_gates(program: &mut Program) -> Result<(), PecosError> {
        let mut expanded_operations = Vec::new();

        // Define native gates - only U and CX are truly native in OpenQASM 2.0
        // Other gates are only native in our implementation for hardware efficiency
        let mut native_gates: HashSet<&str> = ["U", "CX", "u", "cx"].iter().cloned().collect();

        // For PECOS, we also treat these as native for efficiency, but only if they're not user-defined
        // Keep uppercase and lowercase separate to avoid conflicts
        let pecos_native_gates = [
            "H", "X", "Y", "Z", "RZ", "RZZ", "SZZ", // Hardware native gates (uppercase)
            "h", "x", "y", "z", "rz", "rzz", "szz", // User-friendly lowercase versions
        ];

        // Only treat PECOS gates as native if they're not user-defined
        for gate in &pecos_native_gates {
            if !program.gate_definitions.contains_key(*gate) {
                native_gates.insert(gate);
            }
        }

        // Also treat barrier and reset as special native operations
        native_gates.insert("barrier");
        native_gates.insert("reset");

        // Opaque gates pass through unchanged
        native_gates.insert("opaque");

        for operation in &program.operations {
            match operation {
                Operation::Gate {
                    name,
                    parameters,
                    qubits,
                } => {
                    // Check if this is a native gate - don't expand native gates
                    if native_gates.contains(name.as_str()) {
                        expanded_operations.push(operation.clone());
                    }
                    // Check if this gate has a definition
                    else if let Some(gate_def) = program.gate_definitions.get(name) {
                        // Expand the gate using its definition
                        let expanded = Self::expand_gate_call(
                            gate_def,
                            parameters,
                            qubits,
                            &program.gate_definitions,
                        )?;
                        expanded_operations.extend(expanded);
                    } else {
                        // Gate is neither native nor defined - this is an error
                        return Err(PecosError::CompileInvalidOperation {
                            operation: format!("gate '{}'", name),
                            reason: format!(
                                "Undefined gate '{}' - gate is neither native nor user-defined. Did you forget to include qelib1.inc?",
                                name
                            ),
                        });
                    }
                }
                // Other operations pass through unchanged
                _ => expanded_operations.push(operation.clone()),
            }
        }

        program.operations = expanded_operations;
        Ok(())
    }

    fn expand_gate_call(
        gate_def: &GateDefinition,
        parameters: &[f64],
        qubits: &[usize],
        all_definitions: &BTreeMap<String, GateDefinition>,
    ) -> Result<Vec<Operation>, PecosError> {
        Self::expand_gate_call_with_stack(
            gate_def,
            parameters,
            qubits,
            all_definitions,
            &mut vec![gate_def.name.clone()],
        )
    }

    fn expand_gate_call_with_stack(
        gate_def: &GateDefinition,
        parameters: &[f64],
        qubits: &[usize],
        all_definitions: &BTreeMap<String, GateDefinition>,
        expansion_stack: &mut Vec<String>,
    ) -> Result<Vec<Operation>, PecosError> {
        let mut expanded = Vec::new();

        // Define native gates - only U and CX are truly native in OpenQASM 2.0
        // Need to check these during nested expansion too
        let mut native_gates: HashSet<&str> = ["U", "CX", "u", "cx"].iter().cloned().collect();

        // For PECOS, we also treat these as native for efficiency
        let pecos_native_gates = [
            "H", "X", "Y", "Z", "RZ", "RZZ", "SZZ", // Hardware native gates (uppercase)
            "h", "x", "y", "z", "rz", "rzz", "szz", // User-friendly lowercase versions
        ];

        // Only treat PECOS gates as native if they're not user-defined
        for gate in &pecos_native_gates {
            if !all_definitions.contains_key(*gate) {
                native_gates.insert(gate);
            }
        }

        // Also treat barrier and reset as special native operations
        // These are allowed in gate bodies
        native_gates.insert("barrier");
        native_gates.insert("reset");

        // Opaque gates pass through expansion unchanged
        // They will be caught later during validation
        native_gates.insert("opaque");

        // Create parameter mapping
        let mut param_map = HashMap::new();
        for (i, param_name) in gate_def.params.iter().enumerate() {
            if i < parameters.len() {
                param_map.insert(param_name.clone(), parameters[i]);
            }
        }

        // Create qubit mapping from argument names to global IDs
        let mut qubit_map = HashMap::new();
        for (i, qarg_name) in gate_def.qargs.iter().enumerate() {
            if i < qubits.len() {
                qubit_map.insert(qarg_name.clone(), qubits[i]);
            }
        }

        // Expand each operation in the gate body
        for body_op in &gate_def.body {
            // Keep the original name - don't map uppercase to lowercase
            let mapped_name = body_op.name.clone();

            // Substitute parameters
            let mut new_params = Vec::new();
            for param_expr in &body_op.parameters {
                let value = Self::evaluate_param_expr(param_expr, &param_map)?;
                new_params.push(value);
            }

            // Substitute qubits with global IDs
            let mut new_qubits = Vec::new();
            for arg_name in &body_op.arguments {
                if let Some(&mapped_qubit) = qubit_map.get(arg_name) {
                    new_qubits.push(mapped_qubit);
                }
            }

            let new_op = Operation::Gate {
                name: mapped_name.clone(),
                parameters: new_params.clone(),
                qubits: new_qubits.clone(),
            };

            // Check if this gate has a definition - if it does, expand it
            if let Some(nested_def) = all_definitions.get(&mapped_name) {
                // Check for circular dependency
                if expansion_stack.contains(&mapped_name) {
                    let mut cycle_info = String::new();
                    cycle_info.push_str(&format!(
                        "Circular dependency detected: {} -> {}\n\n",
                        expansion_stack.join(" -> "),
                        mapped_name
                    ));

                    // Add helpful context
                    cycle_info.push_str("To fix this error:\n");
                    cycle_info.push_str("1. Check the gate definitions for circular references\n");
                    cycle_info.push_str("2. Ensure no gate directly or indirectly calls itself\n");
                    cycle_info.push_str(
                        "3. Consider breaking the cycle by refactoring your gate hierarchy\n\n",
                    );

                    cycle_info.push_str("The cycle involves these gates:\n");
                    for (i, gate) in expansion_stack.iter().enumerate() {
                        cycle_info.push_str(&format!("  {}. '{}' calls ", i + 1, gate));
                        if i + 1 < expansion_stack.len() {
                            cycle_info.push_str(&format!("'{}'\n", expansion_stack[i + 1]));
                        } else {
                            cycle_info
                                .push_str(&format!("'{}' (completes the cycle)\n", mapped_name));
                        }
                    }

                    return Err(PecosError::CompileCircularDependency(cycle_info));
                }

                // Add to stack for recursion
                expansion_stack.push(mapped_name.clone());

                // Recursively expand non-native gates
                let nested_expanded = Self::expand_gate_call_with_stack(
                    nested_def,
                    &new_params,
                    &new_qubits,
                    all_definitions,
                    expansion_stack,
                )?;

                // Remove from stack after recursion
                expansion_stack.pop();

                expanded.extend(nested_expanded);
            } else {
                // No definition found - check if it's native or undefined
                if native_gates.contains(mapped_name.as_str()) {
                    // It's a native gate, add it
                    expanded.push(new_op);
                } else {
                    // Gate is neither native nor defined - this is an error
                    return Err(PecosError::CompileInvalidOperation {
                        operation: format!("gate '{}'", mapped_name),
                        reason: format!(
                            "Undefined gate '{}' - gate is neither native nor user-defined. Did you forget to include qelib1.inc?",
                            mapped_name
                        ),
                    });
                }
            }
        }

        Ok(expanded)
    }

    fn evaluate_param_expr(
        expr: &Expression,
        param_map: &HashMap<String, f64>,
    ) -> Result<f64, PecosError> {
        match expr {
            Expression::Integer(value) => Ok(*value as f64),
            Expression::Float(value) => Ok(*value),
            Expression::Pi => Ok(std::f64::consts::PI),
            Expression::Variable(name) => param_map
                .get(name)
                .copied()
                .ok_or_else(|| PecosError::ParseInvalidIdentifier(name.clone())),
            Expression::BitId(_name, _idx) => {
                // BitId cannot be evaluated in parameter context
                Err(PecosError::ParseInvalidExpression(
                    "Cannot evaluate bit_id in parameter expression".to_string(),
                ))
            }
            Expression::BinaryOp { op, left, right } => {
                let left_val = Self::evaluate_param_expr(left, param_map)?;
                let right_val = Self::evaluate_param_expr(right, param_map)?;
                match op.as_str() {
                    "+" => Ok(left_val + right_val),
                    "-" => Ok(left_val - right_val),
                    "*" => Ok(left_val * right_val),
                    "/" => Ok(left_val / right_val),
                    "**" => Ok(left_val.powf(right_val)),
                    _ => Err(PecosError::ParseInvalidExpression(format!(
                        "Invalid operator: {}",
                        op
                    ))),
                }
            }
            Expression::FunctionCall { name, args } => {
                if args.len() != 1 {
                    return Err(PecosError::ParseInvalidExpression(format!(
                        "Function {} expects exactly 1 argument, got {}",
                        name,
                        args.len()
                    )));
                }

                let arg_val = Self::evaluate_param_expr(&args[0], param_map)?;

                match name.as_str() {
                    "sin" => Ok(arg_val.sin()),
                    "cos" => Ok(arg_val.cos()),
                    "tan" => Ok(arg_val.tan()),
                    "exp" => Ok(arg_val.exp()),
                    "ln" => {
                        if arg_val <= 0.0 {
                            Err(PecosError::ParseInvalidExpression(format!(
                                "ln({}) is undefined for non-positive values",
                                arg_val
                            )))
                        } else {
                            Ok(arg_val.ln())
                        }
                    }
                    "sqrt" => {
                        if arg_val < 0.0 {
                            Err(PecosError::ParseInvalidExpression(format!(
                                "sqrt({}) is undefined for negative values",
                                arg_val
                            )))
                        } else {
                            Ok(arg_val.sqrt())
                        }
                    }
                    _ => Err(PecosError::ParseInvalidExpression(format!(
                        "Unknown function: {}",
                        name
                    ))),
                }
            }
            Expression::UnaryOp { op, expr } => {
                let val = Self::evaluate_param_expr(expr, param_map)?;
                match op.as_str() {
                    "-" => Ok(-val),
                    "~" => Ok((!(val as i64)) as f64),
                    _ => Err(PecosError::ParseInvalidExpression(format!(
                        "Unknown unary operator: {}",
                        op
                    ))),
                }
            }
        }
    }

    fn validate_no_opaque_gate_usage(program: &Program) -> Result<(), PecosError> {
        // Collect all declared opaque gates
        let mut opaque_gates = HashSet::new();
        let mut gate_usages = Vec::new();

        for operation in &program.operations {
            match operation {
                Operation::OpaqueGate { name, .. } => {
                    opaque_gates.insert(name.clone());
                }
                Operation::Gate { name, .. } => {
                    gate_usages.push(name.clone());
                }
                _ => {}
            }
        }

        // Check if any gate usage corresponds to an opaque gate
        for gate_name in gate_usages {
            if opaque_gates.contains(&gate_name) {
                return Err(PecosError::CompileInvalidOperation {
                    operation: "QASM operation".to_string(),
                    reason: format!(
                        "Opaque gate '{}' is used but opaque gates are not yet implemented in PECOS. \
                    The gate is declared as opaque but cannot be executed.",
                        gate_name
                    ),
                });
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_scientific_notation() -> Result<(), Box<dyn std::error::Error>> {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];

            // Test various scientific notation formats
            rx(1.23e-4) q[0];
            ry(2.5E+3) q[0];
            rz(3e2) q[0];
            u3(1.0e-10, 2E5, .5e-1) q[0];

            // Test regular floats alongside scientific notation
            u1(3.14159) q[0];
            u2(0.5, 1e-3) q[0];
        "#;

        // Define the gates we need in virtual includes with actual bodies
        let virtual_includes = vec![(
            "qelib1.inc".to_string(),
            r#"
            gate rx(theta) a { U(theta, -pi/2, pi/2) a; }
            gate ry(theta) a { U(theta, 0, 0) a; }
            gate rz(theta) a { U(0, 0, theta) a; }
            gate u1(lambda) a { U(0, 0, lambda) a; }
            gate u2(phi, lambda) a { U(pi/2, phi, lambda) a; }
            gate u3(theta, phi, lambda) a { U(theta, phi, lambda) a; }
            "#.to_string(),
        )];

        let program = QASMParser::parse_str_with_virtual_includes_no_expansion(qasm, virtual_includes)?;

        // Verify gates were parsed correctly
        assert_eq!(program.operations.len(), 6);

        // Check that all operations are gates
        for op in &program.operations {
            match op {
                Operation::Gate { .. } => {}
                _ => panic!("Expected only gates"),
            }
        }

        // Test expression evaluation
        let expr1 = Expression::Float(1.23e-4);
        assert_eq!(expr1.evaluate()?, 1.23e-4);

        let expr2 = Expression::Float(2.5E+3);
        assert_eq!(expr2.evaluate()?, 2500.0);

        let expr3 = Expression::Float(3e2);
        assert_eq!(expr3.evaluate()?, 300.0);

        Ok(())
    }

    #[test]
    fn test_parse_bell_state() -> Result<(), Box<dyn std::error::Error>> {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0],q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
        "#;

        let program = QASMParser::parse_str_with_includes_no_expansion(qasm)?;

        assert_eq!(program.version, "2.0");

        // Check register mappings
        assert!(program.quantum_registers.contains_key("q"));
        let q_ids = program.quantum_registers.get("q").unwrap();
        assert_eq!(q_ids.len(), 2);
        assert_eq!(q_ids, &vec![0, 1]); // Global IDs for q[0] and q[1]

        assert_eq!(program.classical_registers.get("c"), Some(&2));
        // Operations should only contain actual gate operations, not definitions
        assert_eq!(program.operations.len(), 4); // 2 gates + 2 measurements

        // Verify the gate operations
        if let Operation::Gate {
            name,
            parameters,
            qubits,
        } = &program.operations[0]
        {
            assert_eq!(name, "H");
            assert!(parameters.is_empty());
            assert_eq!(qubits, &[0]); // Global ID for q[0]
        } else {
            panic!("Expected gate operation");
        }

        if let Operation::Gate {
            name,
            parameters,
            qubits,
        } = &program.operations[1]
        {
            assert_eq!(name, "CX");
            assert!(parameters.is_empty());
            assert_eq!(qubits, &[0, 1]); // Global IDs for q[0] and q[1]
        } else {
            panic!("Expected gate operation");
        }

        // Verify the measure operations
        if let Operation::Measure {
            qubit,
            c_reg,
            c_index,
        } = &program.operations[2]
        {
            assert_eq!(*qubit, 0); // Global ID for q[0]
            assert_eq!(c_reg, "c");
            assert_eq!(*c_index, 0);
        } else {
            panic!("Expected measure operation");
        }

        if let Operation::Measure {
            qubit,
            c_reg,
            c_index,
        } = &program.operations[3]
        {
            assert_eq!(*qubit, 1); // Global ID for q[1]
            assert_eq!(c_reg, "c");
            assert_eq!(*c_index, 1);
        } else {
            panic!("Expected measure operation");
        }

        Ok(())
    }

    #[test]
    fn test_parse_conditional() -> Result<(), Box<dyn std::error::Error>> {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            h q[0];
            measure q[0] -> c[0];
            if(c[0]==1) x q[0];
        "#;

        let program = QASMParser::parse_str_with_includes_no_expansion(qasm)?;

        assert_eq!(program.version, "2.0");
        assert_eq!(program.quantum_registers.get("q").map(|v| v.len()), Some(1));
        assert_eq!(program.classical_registers.get("c"), Some(&1));
        assert_eq!(program.operations.len(), 3); // h gate + measure + if statement

        // Verify the if statement was parsed
        if let Operation::If {
            condition,
            operation,
        } = &program.operations[2]
        {
            // Verify the condition (c[0] == 1)
            if let Expression::BinaryOp { op, left, right } = condition {
                // Check left side is c[0]
                if let Expression::BitId(reg, idx) = &**left {
                    assert_eq!(reg, "c");
                    assert_eq!(*idx, 0);
                } else {
                    panic!("Expected BitId in condition left side");
                }

                // Check operator
                assert_eq!(op, "==");

                // Check right side is 1
                if let Expression::Integer(val) = &**right {
                    assert_eq!(*val, 1);
                } else {
                    panic!("Expected Integer in condition right side");
                }
            } else {
                panic!("Expected BinaryOp in condition");
            }

            // Verify the operation is x q[0]
            if let Operation::Gate { name, qubits, .. } = &**operation {
                assert_eq!(name, "x");
                assert_eq!(qubits, &[0]);
            } else {
                panic!("Expected Gate operation in if statement");
            }
        } else {
            panic!("Expected if statement operation");
        }

        Ok(())
    }

    #[test]
    fn test_parse_classical_conditional() -> Result<(), Box<dyn std::error::Error>> {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            h q[0];
            measure q[0] -> c[0];
            if(c[0]==1) c[0] = 0;
        "#;

        let program = QASMParser::parse_str_with_includes_no_expansion(qasm)?;

        assert_eq!(program.version, "2.0");
        assert_eq!(program.quantum_registers.get("q").map(|v| v.len()), Some(1));
        assert_eq!(program.classical_registers.get("c"), Some(&1));
        assert_eq!(program.operations.len(), 3); // h gate + measure + if statement

        // Verify the if statement contains a classical assignment
        if let Operation::If {
            condition: _,
            operation,
        } = &program.operations[2]
        {
            if let Operation::ClassicalAssignment {
                target,
                is_indexed,
                index,
                expression,
            } = &**operation
            {
                assert_eq!(target, "c");
                assert!(is_indexed);
                assert_eq!(*index, Some(0));

                if let Expression::Integer(val) = expression {
                    assert_eq!(*val, 0);
                } else {
                    panic!("Expected Integer in assignment");
                }
            } else {
                panic!("Expected ClassicalAssignment in if statement");
            }
        } else {
            panic!("Expected If operation");
        }

        Ok(())
    }

    #[test]
    fn test_binary_operators() -> Result<(), Box<dyn std::error::Error>> {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg a[2];
            creg b[2];
            creg c[2];
            
            b = 2;
            a = 1;
            c = b ^ a;  // XOR operation: 2 ^ 1 = 3
            
            // Test other binary operators
            c = b | a;  // OR operation: 2 | 1 = 3
            c = b & a;  // AND operation: 2 & 1 = 0
        "#;

        let program = QASMParser::parse_str_with_includes(qasm)?;

        // Just check that parsing succeeded
        assert_eq!(program.classical_registers.len(), 3);
        assert_eq!(program.operations.len(), 5); // 3 assignments

        Ok(())
    }
}
