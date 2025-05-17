use log::debug;
use pecos_core::errors::PecosError;
use pest::iterators::Pair;
use pest_derive::Parser;
use std::collections::{BTreeMap, HashMap, HashSet};
use std::fmt::Write;
use std::path::Path;

use crate::ast::{Expression, GateDefinition, GateOperation, Operation, OperationDisplay};
use crate::preprocessor::Preprocessor;

#[derive(Parser)]
#[grammar = "qasm.pest"]
#[allow(clippy::too_many_lines)] // Generated code from pest
pub struct QASMParser;

/// Native gates that PECOS can execute directly through `ByteMessage`
/// These gates don't need to be expanded and can be handled by the quantum engine
const PECOS_NATIVE_GATES: &[&str] = &[
    // Quantum gates from ByteMessage::GateType
    "X", "Y", "Z", "H", "CX", "SZZ", "RZ", "R1XY", "RZZ", "SZZdg", "U",
    // Special operations (these are handled differently but treated as "native")
    "barrier", "reset", "opaque", "measure",
];

impl Operation {
    /// Display this operation with proper register names using the qubit mapping
    #[must_use]
    pub fn display_with_map<'a>(
        &'a self,
        qubit_map: &'a HashMap<usize, (String, usize)>,
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
    pub qubit_map: HashMap<usize, (String, usize)>, // global_id -> (register_name, index)
}

/// Simple configuration for parsing
#[derive(Clone)]
pub struct ParseConfig {
    pub includes: Vec<(String, String)>,
    pub search_paths: Vec<std::path::PathBuf>,
    pub expand_gates: bool,
    pub validate_gates: bool,
}

impl Default for ParseConfig {
    fn default() -> Self {
        Self {
            includes: vec![],
            search_paths: vec![],
            expand_gates: true,
            validate_gates: true,
        }
    }
}

impl QASMParser {
    const QASM_OPERATION: &'static str = "QASM operation";

    /// Create a `CompileInvalidOperation` error with standard QASM operation context
    fn invalid_operation_error(reason: impl Into<String>) -> PecosError {
        PecosError::CompileInvalidOperation {
            operation: Self::QASM_OPERATION.to_string(),
            reason: reason.into(),
        }
    }

    /// Create a `CompileInvalidOperation` error for unknown register
    fn unknown_register_error(register_type: &str, register_name: &str) -> PecosError {
        PecosError::CompileInvalidOperation {
            operation: Self::QASM_OPERATION.to_string(),
            reason: format!("Unknown {register_type} register '{register_name}'"),
        }
    }

    /// Create a `CompileInvalidOperation` error for register index out of bounds
    fn register_index_error(register_name: &str, index: usize, reason: &str) -> PecosError {
        PecosError::CompileInvalidOperation {
            operation: Self::QASM_OPERATION.to_string(),
            reason: format!(
                "{} index {} {} for register '{}'",
                if register_name.starts_with('c') {
                    "Bit"
                } else {
                    "Qubit"
                },
                index,
                reason,
                register_name
            ),
        }
    }

    /// Get the standard includes directory path
    fn get_standard_includes_path() -> std::path::PathBuf {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        std::path::Path::new(manifest_dir).join("includes")
    }

    /// Parse QASM source with default configuration
    pub fn parse_str(source: &str) -> Result<Program, PecosError> {
        Self::parse_with_config(source, &ParseConfig::default())
    }

    /// Main parsing method using configuration
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
            Self::expand_gates(&mut program)?;
        }

        // Validate if requested
        if config.validate_gates {
            Self::validate_no_opaque_gate_usage(&program)?;
        }

        Ok(program)
    }

    /// Parse a file with default configuration
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
    pub fn preprocess(source: &str) -> Result<String, PecosError> {
        let mut preprocessor = Preprocessor::new();
        // Add standard includes path as fallback for filesystem includes
        if let Some(path_str) = Self::get_standard_includes_path().to_str() {
            preprocessor.add_path(path_str);
        }
        preprocessor.preprocess(source)
    }

    /// Get the preprocessed and expanded QASM (after phases 1 and 2)
    pub fn preprocess_and_expand(source: &str) -> Result<String, PecosError> {
        // Phase 1: Preprocess includes
        let preprocessed = Self::preprocess(source)?;

        // Phase 2: Expand gates to native operations
        Self::expand_all_gate_definitions(&preprocessed)
    }

    /// Expand all gate definitions in QASM source to native gates only.
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
        let mut pairs =
            <Self as pest::Parser<Rule>>::parse(Rule::program, source).map_err(|e| {
                PecosError::ParseSyntax {
                    language: "QASM".to_string(),
                    message: e.to_string(),
                }
            })?;

        let program_pair = pairs
            .next()
            .ok_or_else(|| Self::invalid_operation_error("Empty program"))?;

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
                            Rule::gate_def => {
                                Self::parse_gate_definition(inner_pair, &mut program)?;
                            }
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
    fn format_operation(op: &Operation, qubit_map: &HashMap<usize, (String, usize)>) -> String {
        // Use the display wrapper to properly format with register names
        format!("{}", op.display_with_map(qubit_map))
    }

    /// Parse QASM with virtual includes but without gate expansion (for testing)
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
    pub fn parse_str_raw(source: &str) -> Result<Program, PecosError> {
        let mut program = Program::default();
        let mut pairs =
            <Self as pest::Parser<Rule>>::parse(Rule::program, source).map_err(|e| {
                PecosError::ParseSyntax {
                    language: "QASM".to_string(),
                    message: e.to_string(),
                }
            })?;
        let program_pair = pairs
            .next()
            .ok_or_else(|| Self::invalid_operation_error("Empty program"))?;

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
                _ => {}
            }
        }

        // After parsing, expand all gates using their definitions
        Self::expand_gates(&mut program)?;
        Ok(program)
    }

    fn parse_statement(
        pair: pest::iterators::Pair<Rule>,
        program: &mut Program,
    ) -> Result<(), PecosError> {
        for inner_pair in pair.into_inner() {
            match inner_pair.as_rule() {
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
                _ => {}
            }
        }
        Ok(())
    }

    fn parse_register(
        pair: pest::iterators::Pair<Rule>,
        program: &mut Program,
    ) -> Result<(), PecosError> {
        let inner = pair.into_inner().next().unwrap();

        match inner.as_rule() {
            Rule::qreg => {
                let indexed_id = inner.into_inner().next().unwrap();
                let (name, size) = Self::parse_indexed_id(&indexed_id)?;

                // Assign global qubit IDs
                let mut qubit_ids = Vec::new();
                for i in 0..size {
                    let global_id = program.total_qubits;
                    qubit_ids.push(global_id);
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
                return Err(Self::invalid_operation_error(format!(
                    "Unexpected register type: {:?}",
                    inner.as_rule()
                )));
            }
        }

        Ok(())
    }

    // Consolidated method for parsing indexed identifiers (replaces duplicate methods)
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

    // Simplified binary expression parser
    fn parse_binary_expr(pair: Pair<Rule>) -> Result<Expression, PecosError> {
        let rule = pair.as_rule();
        let inner_pairs: Vec<Pair<Rule>> = pair.into_inner().collect();

        // Single element - no operator
        if inner_pairs.len() == 1 {
            return Self::parse_expr(inner_pairs[0].clone());
        }

        // Get default operator for the current rule
        let default_op = match rule {
            Rule::b_or_expr => "|",
            Rule::b_xor_expr => "^",
            Rule::b_and_expr => "&",
            Rule::equality_expr => "==",
            Rule::relational_expr => "<",
            Rule::shift_expr => "<<",
            Rule::additive_expr => "+",
            Rule::multiplicative_expr => "*",
            Rule::power_expr => "**",
            _ => {
                return Err(PecosError::ParseInvalidExpression(
                    "Unknown binary rule".to_string(),
                ));
            }
        };

        // Build expression tree
        let mut result = Self::parse_expr(inner_pairs[0].clone())?;
        let mut i = 1;

        while i < inner_pairs.len() {
            let next_pair = &inner_pairs[i];

            let (op, right_expr) = match next_pair.as_rule() {
                // Explicit operator
                Rule::equality_op
                | Rule::relational_op
                | Rule::shift_op
                | Rule::add_op
                | Rule::mul_op
                | Rule::pow_op => {
                    if i + 1 < inner_pairs.len() {
                        let op_str = next_pair.as_str();
                        let right = Self::parse_expr(inner_pairs[i + 1].clone())?;
                        i += 2;
                        (op_str, right)
                    } else {
                        return Err(PecosError::ParseInvalidExpression(
                            "Missing right operand".to_string(),
                        ));
                    }
                }
                // Implicit operator
                _ => {
                    let right = Self::parse_expr(next_pair.clone())?;
                    i += 1;
                    (default_op, right)
                }
            };

            result = Expression::BinaryOp {
                op: op.to_string(),
                left: Box::new(result),
                right: Box::new(right_expr),
            };
        }

        Ok(result)
    }

    // Main expression parser
    fn parse_expr(pair: Pair<Rule>) -> Result<Expression, PecosError> {
        match pair.as_rule() {
            Rule::expr => {
                let inner = pair.into_inner().next().ok_or_else(|| {
                    PecosError::ParseInvalidExpression("Empty expression".to_string())
                })?;
                Self::parse_expr(inner)
            }

            // Binary operations - use consolidated parser
            Rule::b_or_expr
            | Rule::b_xor_expr
            | Rule::b_and_expr
            | Rule::equality_expr
            | Rule::relational_expr
            | Rule::shift_expr
            | Rule::additive_expr
            | Rule::multiplicative_expr
            | Rule::power_expr => Self::parse_binary_expr(pair),

            // Unary operations
            Rule::unary_expr => {
                let mut pairs = pair.into_inner();
                let mut ops = Vec::new();

                // Collect operators
                while let Some(pair) = pairs.peek() {
                    if pair.as_rule() == Rule::unary_op {
                        ops.push(pairs.next().unwrap().as_str().to_string());
                    } else {
                        break;
                    }
                }

                // Get operand
                let operand_pair = pairs.next().ok_or_else(|| {
                    PecosError::ParseInvalidExpression(
                        "Missing operand for unary operation".to_string(),
                    )
                })?;
                let mut expr = Self::parse_expr(operand_pair)?;

                // Apply operators in reverse order
                for op in ops.iter().rev() {
                    match (&op[..], &expr) {
                        ("-", Expression::Integer(value)) => {
                            expr = Expression::Integer(-value);
                        }
                        _ => {
                            expr = Expression::UnaryOp {
                                op: op.clone(),
                                expr: Box::new(expr),
                            };
                        }
                    }
                }

                Ok(expr)
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
            Rule::identifier => Ok(Expression::Variable(pair.as_str().to_string())),
            Rule::function_call => {
                let mut pairs = pair.into_inner();
                let name = pairs.next().unwrap().as_str().to_string();
                let args: Result<Vec<_>, _> = pairs.map(Self::parse_expr).collect();
                Ok(Expression::FunctionCall { name, args: args? })
            }
            _ => Err(PecosError::ParseInvalidExpression(format!(
                "Unexpected rule in expression: {:?}",
                pair.as_rule()
            ))),
        }
    }

    #[allow(clippy::too_many_lines)]
    fn parse_quantum_op(
        pair: pest::iterators::Pair<Rule>,
        program: &Program,
    ) -> Result<Option<Operation>, PecosError> {
        let inner = pair.into_inner().next().unwrap();

        match inner.as_rule() {
            Rule::gate_call => {
                let mut inner_pairs = inner.into_inner();
                let gate_name = inner_pairs.next().unwrap().as_str();

                let mut params = Vec::new();
                let mut register_or_qubits = Vec::new();

                for pair in inner_pairs {
                    match pair.as_rule() {
                        Rule::param_values => {
                            for param_expr in pair.into_inner() {
                                if param_expr.as_rule() == Rule::expr {
                                    let expr = Self::parse_expr(param_expr)?;
                                    let value = expr.evaluate_with_context(None).map_err(|e| {
                                        PecosError::ParseInvalidExpression(format!(
                                            "Failed to evaluate parameter: {e}"
                                        ))
                                    })?;
                                    params.push(value);
                                }
                            }
                        }
                        Rule::any_list => {
                            for item in pair.into_inner() {
                                if item.as_rule() == Rule::any_item {
                                    let inner = item.into_inner().next().unwrap();
                                    match inner.as_rule() {
                                        Rule::identifier => {
                                            // Handle register name - expand to all qubits in register
                                            let reg_name = inner.as_str();
                                            if let Some(qubit_ids) =
                                                program.quantum_registers.get(reg_name)
                                            {
                                                register_or_qubits.push((
                                                    reg_name.to_string(),
                                                    qubit_ids.clone(),
                                                ));
                                            } else {
                                                return Err(Self::unknown_register_error(
                                                    "quantum", reg_name,
                                                ));
                                            }
                                        }
                                        Rule::qubit_id => {
                                            // Handle individual qubit
                                            let (reg_name, idx) = Self::parse_indexed_id(&inner)?;
                                            if let Some(qubit_ids) =
                                                program.quantum_registers.get(&reg_name)
                                            {
                                                if idx < qubit_ids.len() {
                                                    register_or_qubits.push((
                                                        format!("{reg_name}[{idx}]"),
                                                        vec![qubit_ids[idx]],
                                                    ));
                                                } else {
                                                    return Err(Self::register_index_error(
                                                        &reg_name,
                                                        idx,
                                                        "out of bounds",
                                                    ));
                                                }
                                            } else {
                                                return Err(Self::unknown_register_error(
                                                    "quantum", &reg_name,
                                                ));
                                            }
                                        }
                                        _ => {}
                                    }
                                }
                            }
                        }
                        _ => {}
                    }
                }

                // Now handle the expansion of registers into individual gate operations
                let num_operands = register_or_qubits.len();

                // Check if any of the operands are actually full registers
                let has_register = register_or_qubits
                    .iter()
                    .any(|(_, qubits)| qubits.len() > 1);

                if !has_register {
                    // All operands are individual qubits, no expansion needed
                    let mut all_qubits = Vec::new();
                    for (_, qubits) in &register_or_qubits {
                        all_qubits.extend(qubits);
                    }

                    Ok(Some(Operation::Gate {
                        name: gate_name.to_string(),
                        parameters: params,
                        qubits: all_qubits,
                    }))
                } else if num_operands == 1 {
                    // Single operand that is a register - expand to individual gates
                    let (_name, qubits) = &register_or_qubits[0];

                    // For phase 2 expansion, create a single gate with multiple qubits
                    // PECOS will handle the expansion later
                    Ok(Some(Operation::Gate {
                        name: gate_name.to_string(),
                        parameters: params,
                        qubits: qubits.clone(),
                    }))
                } else if num_operands == 2 {
                    // For two-qubit gates, handle register sizes
                    let (_name1, qubits1) = &register_or_qubits[0];
                    let (_name2, qubits2) = &register_or_qubits[1];

                    // If both are single qubits, no special handling needed
                    if qubits1.len() == 1 && qubits2.len() == 1 {
                        Ok(Some(Operation::Gate {
                            name: gate_name.to_string(),
                            parameters: params,
                            qubits: vec![qubits1[0], qubits2[0]],
                        }))
                    } else if qubits1.len() == qubits2.len() {
                        // Both are registers of the same size - apply pairwise
                        // For now, we'll create a special marker for this case
                        // that the expansion phase will handle
                        let mut all_qubits = Vec::new();
                        for i in 0..qubits1.len() {
                            all_qubits.push(qubits1[i]);
                            all_qubits.push(qubits2[i]);
                        }

                        Ok(Some(Operation::Gate {
                            name: gate_name.to_string(),
                            parameters: params,
                            qubits: all_qubits,
                        }))
                    } else {
                        // Register size mismatch
                        return Err(PecosError::CompileInvalidOperation {
                            operation: Self::QASM_OPERATION.to_string(),
                            reason: format!(
                                "Register size mismatch for gate {}: first operand has {} qubits, second has {}",
                                gate_name,
                                qubits1.len(),
                                qubits2.len()
                            ),
                        });
                    }
                } else {
                    // For gates with more than 2 operands, just collect all qubits
                    let mut all_qubits = Vec::new();
                    for (_name, qubits) in &register_or_qubits {
                        all_qubits.extend(qubits);
                    }

                    Ok(Some(Operation::Gate {
                        name: gate_name.to_string(),
                        parameters: params,
                        qubits: all_qubits,
                    }))
                }
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
                let (q_reg, q_idx) = Self::parse_indexed_id(&src.clone())?;
                let (c_reg, c_idx) = Self::parse_indexed_id(&dst.clone())?;

                if let Some(qubit_ids) = program.quantum_registers.get(&q_reg) {
                    if q_idx < qubit_ids.len() {
                        let global_qubit_id = qubit_ids[q_idx];

                        Ok(Some(Operation::Measure {
                            qubit: global_qubit_id,
                            c_reg,
                            c_index: c_idx,
                        }))
                    } else {
                        Err(Self::register_index_error(&q_reg, q_idx, "out of bounds"))
                    }
                } else {
                    Err(Self::unknown_register_error("quantum", &q_reg))
                }
            } else if src.as_rule() == Rule::identifier && dst.as_rule() == Rule::identifier {
                Ok(Some(Operation::RegMeasure {
                    q_reg: src.as_str().to_string(),
                    c_reg: dst.as_str().to_string(),
                }))
            } else {
                Err(Self::invalid_operation_error("Invalid measurement format"))
            }
        } else {
            Err(Self::invalid_operation_error("Invalid measurement syntax"))
        }
    }

    fn parse_reset(
        pair: pest::iterators::Pair<Rule>,
        program: &Program,
    ) -> Result<Option<Operation>, PecosError> {
        let qubit_id = pair.into_inner().next().unwrap();
        let (reg_name, idx) = Self::parse_indexed_id(&qubit_id)?;

        if let Some(qubit_ids) = program.quantum_registers.get(&reg_name) {
            if idx < qubit_ids.len() {
                let global_qubit_id = qubit_ids[idx];
                Ok(Some(Operation::Reset {
                    qubit: global_qubit_id,
                }))
            } else {
                Err(Self::register_index_error(&reg_name, idx, "out of bounds"))
            }
        } else {
            Err(Self::unknown_register_error("quantum", &reg_name))
        }
    }

    fn parse_barrier(
        pair: pest::iterators::Pair<Rule>,
        program: &Program,
    ) -> Result<Option<Operation>, PecosError> {
        let any_list = pair.into_inner().next().unwrap();
        let mut qubits = Vec::new();

        for item in any_list.into_inner() {
            if item.as_rule() == Rule::any_item {
                let inner = item.into_inner().next().unwrap();
                match inner.as_rule() {
                    Rule::identifier => {
                        let reg_name = inner.as_str();
                        if let Some(qubit_ids) = program.quantum_registers.get(reg_name) {
                            qubits.extend(qubit_ids.iter());
                        } else {
                            return Err(Self::unknown_register_error("quantum", reg_name));
                        }
                    }
                    Rule::qubit_id => {
                        let (reg_name, idx) = Self::parse_indexed_id(&inner)?;
                        if let Some(qubit_ids) = program.quantum_registers.get(&reg_name) {
                            if idx < qubit_ids.len() {
                                qubits.push(qubit_ids[idx]);
                            } else {
                                return Err(Self::register_index_error(
                                    &reg_name,
                                    idx,
                                    "out of bounds",
                                ));
                            }
                        } else {
                            return Err(Self::unknown_register_error("quantum", &reg_name));
                        }
                    }
                    _ => {}
                }
            }
        }

        Ok(Some(Operation::Barrier { qubits }))
    }

    // Helper functions remain largely the same with minimal refactoring...

    // Continued with the rest of the parser implementation...
    fn parse_if_statement(
        pair: pest::iterators::Pair<Rule>,
        program: &Program,
    ) -> Result<Option<Operation>, PecosError> {
        debug!("Parsing if statement: '{}'", pair.as_str());

        let parts: Vec<_> = pair.into_inner().collect();

        if parts.len() < 2 {
            return Err(PecosError::CompileInvalidOperation {
                operation: Self::QASM_OPERATION.to_string(),
                reason: format!(
                    "Invalid if statement: expected at least 2 parts, got {}",
                    parts.len()
                ),
            });
        }

        let condition_expr_pair = &parts[0];
        let operation_pair = &parts[1];

        let condition = match condition_expr_pair.as_rule() {
            Rule::condition_expr => {
                let expr_pair =
                    condition_expr_pair
                        .clone()
                        .into_inner()
                        .next()
                        .ok_or_else(|| PecosError::CompileInvalidOperation {
                            operation: Self::QASM_OPERATION.to_string(),
                            reason: "Empty condition expression".to_string(),
                        })?;
                Self::parse_expr(expr_pair)?
            }
            _ => {
                return Err(PecosError::CompileInvalidOperation {
                    operation: Self::QASM_OPERATION.to_string(),
                    reason: format!(
                        "Invalid rule in if statement, expected condition_expr, got: {:?}",
                        condition_expr_pair.as_rule()
                    ),
                });
            }
        };

        let operation = match operation_pair.as_rule() {
            Rule::quantum_op => {
                if let Some(op) = Self::parse_quantum_op(operation_pair.clone(), program)? {
                    op
                } else {
                    return Err(PecosError::CompileInvalidOperation {
                        operation: Self::QASM_OPERATION.to_string(),
                        reason: "Invalid quantum operation in if statement".to_string(),
                    });
                }
            }
            Rule::classical_op => {
                if let Some(op) = Self::parse_classical_operation(operation_pair.clone())? {
                    op
                } else {
                    return Err(PecosError::CompileInvalidOperation {
                        operation: Self::QASM_OPERATION.to_string(),
                        reason: "Invalid classical operation in if statement".to_string(),
                    });
                }
            }
            _ => {
                return Err(PecosError::CompileInvalidOperation {
                    operation: Self::QASM_OPERATION.to_string(),
                    reason: format!(
                        "Unsupported operation type in if statement: {:?}",
                        operation_pair.as_rule()
                    ),
                });
            }
        };

        Ok(Some(Operation::If {
            condition,
            operation: Box::new(operation),
        }))
    }

    fn parse_classical_operation(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Option<Operation>, PecosError> {
        let inner_parts: Vec<_> = pair.into_inner().collect();

        if inner_parts.len() >= 2 {
            let target_pair = &inner_parts[0];
            let target: String;
            let is_indexed: bool;
            let index: Option<usize>;

            match target_pair.as_rule() {
                Rule::bit_id => {
                    let (reg_name, bit_idx) = Self::parse_indexed_id(target_pair)?;
                    target = reg_name;
                    is_indexed = true;
                    index = Some(bit_idx);
                }
                Rule::identifier => {
                    target = target_pair.as_str().to_string();
                    is_indexed = false;
                    index = None;
                }
                _ => {
                    return Err(PecosError::CompileInvalidOperation {
                        operation: Self::QASM_OPERATION.to_string(),
                        reason: format!(
                            "Invalid classical assignment target: {:?}",
                            target_pair.as_rule()
                        ),
                    });
                }
            }

            let expr_pair = &inner_parts[1];
            let expression = Self::parse_expr(expr_pair.clone())?;

            return Ok(Some(Operation::ClassicalAssignment {
                target,
                is_indexed,
                index,
                expression,
            }));
        }

        Err(PecosError::CompileInvalidOperation {
            operation: Self::QASM_OPERATION.to_string(),
            reason: "Invalid classical operation".to_string(),
        })
    }

    fn parse_gate_definition(
        pair: pest::iterators::Pair<Rule>,
        program: &mut Program,
    ) -> Result<(), PecosError> {
        let mut inner = pair.into_inner();

        let name = inner.next().unwrap().as_str().to_string();

        let mut params = Vec::new();
        let mut qargs = Vec::new();
        let mut body_pairs = Vec::new();

        for inner_pair in inner {
            match inner_pair.as_rule() {
                Rule::param_list => {
                    for param in inner_pair.into_inner() {
                        if param.as_rule() == Rule::identifier {
                            params.push(param.as_str().to_string());
                        }
                    }
                }
                Rule::identifier_list => {
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

        let mut body = Vec::new();
        for statement_pair in body_pairs {
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

        let name = inner
            .next()
            .ok_or_else(|| PecosError::CompileInvalidOperation {
                operation: Self::QASM_OPERATION.to_string(),
                reason: "Missing gate name".to_string(),
            })?
            .as_str()
            .to_string();

        let mut params = Vec::new();
        let mut qargs = Vec::new();

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
    ) -> Result<Option<GateOperation>, PecosError> {
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
                            for expr_pair in part.into_inner() {
                                let param_expr = Self::parse_expr(expr_pair)?;
                                params.push(param_expr);
                            }
                        }
                        Rule::identifier_list => {
                            for ident in part.into_inner() {
                                if ident.as_rule() == Rule::identifier {
                                    arguments.push(ident.as_str().to_string());
                                }
                            }
                        }
                        _ => {}
                    }
                }

                Ok(Some(GateOperation {
                    name: gate_name.to_string(),
                    params,
                    qargs: arguments,
                }))
            }
            _ => Ok(None),
        }
    }

    // Simplified gate expansion
    #[allow(clippy::too_many_lines)]
    fn expand_gates(program: &mut Program) -> Result<(), PecosError> {
        let mut expanded_operations = Vec::new();

        // Create a set of native gates from our constant
        let mut native_gates: HashSet<&str> = HashSet::new();

        // Only add native gates that aren't user-defined
        for &gate in PECOS_NATIVE_GATES {
            if !program.gate_definitions.contains_key(gate) {
                native_gates.insert(gate);
            }
        }

        for operation in &program.operations {
            match operation {
                Operation::Gate {
                    name,
                    parameters,
                    qubits,
                } => {
                    // Gate names in QASM files are lowercase, but we need to check against PECOS native gates
                    let uppercase_name = name.to_uppercase();

                    // Check if this is a register-level gate operation that needs expansion
                    // Only handle PECOS native gates
                    let needs_register_expansion = match uppercase_name.as_str() {
                        // Single-qubit gates that can be applied to registers
                        "H" | "X" | "Y" | "Z" | "RZ" | "U" | "R1XY" => qubits.len() > 1,
                        // Two-qubit gates need pairwise expansion
                        "CX" | "SZZ" | "RZZ" | "SZZDG" => qubits.len() > 2,
                        _ => false,
                    };

                    if needs_register_expansion {
                        // Handle register expansion based on gate type
                        match uppercase_name.as_str() {
                            // Single-qubit native gates: apply to each qubit individually
                            "H" | "X" | "Y" | "Z" | "RZ" | "R1XY" | "U" => {
                                for &qubit in qubits {
                                    expanded_operations.push(Operation::Gate {
                                        name: name.clone(), // Keep original name casing
                                        parameters: parameters.clone(),
                                        qubits: vec![qubit],
                                    });
                                }
                            }
                            // Two-qubit native gates: apply pairwise
                            "CX" | "SZZ" | "RZZ" | "SZZDG" => {
                                if qubits.len() % 2 != 0 {
                                    return Err(PecosError::CompileInvalidOperation {
                                        operation: format!("gate '{name}'"),
                                        reason: format!(
                                            "Two-qubit gate '{}' applied to {} qubits (must be even number)",
                                            name,
                                            qubits.len()
                                        ),
                                    });
                                }

                                // Apply gate pairwise
                                for i in (0..qubits.len()).step_by(2) {
                                    expanded_operations.push(Operation::Gate {
                                        name: name.clone(),
                                        parameters: parameters.clone(),
                                        qubits: vec![qubits[i], qubits[i + 1]],
                                    });
                                }
                            }
                            _ => {
                                // For other gates, just pass through
                                expanded_operations.push(operation.clone());
                            }
                        }
                    } else if native_gates.contains(name.as_str()) {
                        expanded_operations.push(operation.clone());
                    } else if let Some(gate_def) = program.gate_definitions.get(name) {
                        let expanded = Self::expand_gate_call(
                            gate_def,
                            parameters,
                            qubits,
                            &program.gate_definitions,
                        )?;
                        expanded_operations.extend(expanded);
                    } else {
                        return Err(PecosError::CompileInvalidOperation {
                            operation: format!("gate '{name}'"),
                            reason: format!(
                                "Undefined gate '{name}' - gate is neither native nor user-defined. Did you forget to include qelib1.inc?"
                            ),
                        });
                    }
                }
                Operation::RegMeasure { q_reg, c_reg } => {
                    // Expand register-level measurement to individual measurements
                    let q_qubits = program.quantum_registers.get(q_reg).ok_or_else(|| {
                        PecosError::CompileInvalidOperation {
                            operation: format!("measure {q_reg} -> {c_reg}"),
                            reason: format!("Unknown quantum register: {q_reg}"),
                        }
                    })?;

                    let c_size = program.classical_registers.get(c_reg).ok_or_else(|| {
                        PecosError::CompileInvalidOperation {
                            operation: format!("measure {q_reg} -> {c_reg}"),
                            reason: format!("Unknown classical register: {c_reg}"),
                        }
                    })?;

                    if q_qubits.len() != *c_size {
                        return Err(PecosError::CompileInvalidOperation {
                            operation: format!("measure {q_reg} -> {c_reg}"),
                            reason: format!(
                                "Register size mismatch: quantum register {} has {} qubits, classical register {} has {} bits",
                                q_reg,
                                q_qubits.len(),
                                c_reg,
                                c_size
                            ),
                        });
                    }

                    // Expand to individual measurements
                    for (i, &qubit) in q_qubits.iter().enumerate() {
                        expanded_operations.push(Operation::Measure {
                            qubit,
                            c_reg: c_reg.clone(),
                            c_index: i,
                        });
                    }
                }
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

        // Create a set of native gates from our constant
        let mut native_gates: HashSet<&str> = HashSet::new();

        // Only add native gates that aren't user-defined
        for &gate in PECOS_NATIVE_GATES {
            if !all_definitions.contains_key(gate) {
                native_gates.insert(gate);
            }
        }

        // Create parameter mapping
        let mut param_map = HashMap::new();
        for (i, param_name) in gate_def.params.iter().enumerate() {
            if i < parameters.len() {
                param_map.insert(param_name.clone(), parameters[i]);
            }
        }

        // Create qubit mapping
        let mut qubit_map = HashMap::new();
        for (i, qarg_name) in gate_def.qargs.iter().enumerate() {
            if i < qubits.len() {
                qubit_map.insert(qarg_name.clone(), qubits[i]);
            }
        }

        // Expand each operation in the gate body
        for body_op in &gate_def.body {
            let mapped_name = body_op.name.clone();

            // Substitute parameters
            let mut new_params = Vec::new();
            for param_expr in &body_op.params {
                let value = Self::evaluate_param_expr(param_expr, &param_map)?;
                new_params.push(value);
            }

            // Substitute qubits
            let mut new_qubits = Vec::new();
            for arg_name in &body_op.qargs {
                if let Some(&mapped_qubit) = qubit_map.get(arg_name) {
                    new_qubits.push(mapped_qubit);
                }
            }

            let new_op = Operation::Gate {
                name: mapped_name.clone(),
                parameters: new_params.clone(),
                qubits: new_qubits.clone(),
            };

            // Check for circular dependency
            if let Some(nested_def) = all_definitions.get(&mapped_name) {
                if expansion_stack.contains(&mapped_name) {
                    let mut cycle_info = String::new();
                    write!(
                        cycle_info,
                        "Circular dependency detected: {} -> {}\n\n",
                        expansion_stack.join(" -> "),
                        mapped_name
                    )
                    .unwrap();

                    cycle_info.push_str("To fix this error:\n");
                    cycle_info.push_str("1. Check the gate definitions for circular references\n");
                    cycle_info.push_str("2. Ensure no gate directly or indirectly calls itself\n");
                    cycle_info.push_str(
                        "3. Consider breaking the cycle by refactoring your gate hierarchy\n\n",
                    );
                    cycle_info.push_str("The cycle involves these gates:\n");

                    for (i, gate) in expansion_stack.iter().enumerate() {
                        write!(cycle_info, "  {}. '{}' calls ", i + 1, gate).unwrap();
                        if i + 1 < expansion_stack.len() {
                            writeln!(cycle_info, "'{}'", expansion_stack[i + 1]).unwrap();
                        } else {
                            writeln!(cycle_info, "'{mapped_name}' (completes the cycle)").unwrap();
                        }
                    }

                    return Err(PecosError::CompileCircularDependency(cycle_info));
                }

                expansion_stack.push(mapped_name.clone());

                let nested_expanded = Self::expand_gate_call_with_stack(
                    nested_def,
                    &new_params,
                    &new_qubits,
                    all_definitions,
                    expansion_stack,
                )?;

                expansion_stack.pop();
                expanded.extend(nested_expanded);
            } else if native_gates.contains(mapped_name.as_str()) {
                expanded.push(new_op);
            } else {
                return Err(PecosError::CompileInvalidOperation {
                    operation: format!("gate '{mapped_name}'"),
                    reason: format!(
                        "Undefined gate '{mapped_name}' - gate is neither native nor user-defined. Did you forget to include qelib1.inc?"
                    ),
                });
            }
        }

        Ok(expanded)
    }

    fn evaluate_param_expr(
        expr: &Expression,
        param_map: &HashMap<String, f64>,
    ) -> Result<f64, PecosError> {
        use crate::ast::EvaluationCtx;
        let context = EvaluationCtx {
            params: Some(param_map),
        };
        expr.evaluate(Some(&context))
    }

    fn validate_no_opaque_gate_usage(program: &Program) -> Result<(), PecosError> {
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

        for gate_name in gate_usages {
            if opaque_gates.contains(&gate_name) {
                return Err(PecosError::CompileInvalidOperation {
                    operation: Self::QASM_OPERATION.to_string(),
                    reason: format!(
                        "Opaque gate '{gate_name}' is used but opaque gates are not yet implemented in PECOS. \
                    The gate is declared as opaque but cannot be executed."
                    ),
                });
            }
        }

        Ok(())
    }
}
