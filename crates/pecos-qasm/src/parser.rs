use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;
use log::debug;

#[derive(Debug, Clone)]
pub enum ParameterExpression {
    Constant(f64),
    Identifier(String),
    Pi,
    BinaryOp {
        op: String,
        left: Box<ParameterExpression>,
        right: Box<ParameterExpression>,
    },
}

#[derive(Debug, Clone)]
pub struct GateDefOperation {
    pub name: String,
    pub parameters: Vec<ParameterExpression>,
    pub arguments: Vec<String>,
}

#[derive(Parser)]
#[grammar = "qasm.pest"]
pub struct QASMParser;

#[derive(Debug)]
pub enum ParseError {
    IoError(std::io::Error),
    PestError(Box<pest::error::Error<Rule>>),
    InvalidVersion(String),
    InvalidRegisterSize(String),
    InvalidOperation(String),
    InvalidExpression(String),
    InvalidFloat(String),
    InvalidInt(String),
    InvalidExpr(String),
    InvalidParameter(String),
    InvalidOperator(String),
    InvalidNumber,
    InvalidConstant(String),
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ParseError::IoError(err) => write!(f, "IO error: {err}"),
            ParseError::PestError(err) => write!(f, "Parse error: {err}"),
            ParseError::InvalidVersion(msg) => write!(f, "Invalid version: {msg}"),
            ParseError::InvalidRegisterSize(msg) => write!(f, "Invalid register size: {msg}"),
            ParseError::InvalidOperation(msg) => write!(f, "Invalid operation: {msg}"),
            ParseError::InvalidExpression(msg) | ParseError::InvalidExpr(msg) => {
                write!(f, "Invalid expression: {msg}")
            }
            ParseError::InvalidFloat(msg) => write!(f, "Invalid float: {msg}"),
            ParseError::InvalidInt(msg) => write!(f, "Invalid int: {msg}"),
            ParseError::InvalidParameter(name) => write!(f, "Invalid parameter: {name}"),
            ParseError::InvalidOperator(op) => write!(f, "Invalid operator: {op}"),
            ParseError::InvalidNumber => write!(f, "Invalid number"),
            ParseError::InvalidConstant(msg) => write!(f, "Invalid constant: {msg}"),
        }
    }
}

impl std::error::Error for ParseError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            ParseError::IoError(err) => Some(err),
            ParseError::PestError(err) => Some(&**err),
            _ => None,
        }
    }
}

impl From<std::io::Error> for ParseError {
    fn from(err: std::io::Error) -> Self {
        ParseError::IoError(err)
    }
}

impl From<pest::error::Error<Rule>> for ParseError {
    fn from(err: pest::error::Error<Rule>) -> Self {
        ParseError::PestError(Box::new(err))
    }
}

impl From<std::num::ParseIntError> for ParseError {
    fn from(err: std::num::ParseIntError) -> Self {
        ParseError::InvalidRegisterSize(err.to_string())
    }
}

#[derive(Debug, Clone)]
pub enum Expression {
    Integer(i64),
    Float(f64),
    Pi,
    BinaryOp(Box<Expression>, String, Box<Expression>),
    UnaryOp(String, Box<Expression>),
    BitId(String, i64),
    Variable(String),
    FunctionCall {
        name: String,
        args: Vec<Expression>,
    },
}

impl Expression {
    pub fn evaluate(&self) -> Result<f64, Box<dyn Error>> {
        match self {
            #[allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]
            Expression::Integer(i) => {
                // i64 to f64 conversion can lose precision for values > 2^53
                // For QASM integer literals, this is an acceptable tradeoff as such large
                // integers are unlikely in quantum circuit descriptions

                // Perform the conversion and check if precision was lost
                let value = *i as f64;

                // Check if the roundtrip conversion preserves the value
                if *i != (value as i64) {
                    // This warning is important for debugging but doesn't affect correctness
                    // QASM rarely uses integers large enough to cause precision loss
                    eprintln!(
                        "Warning: Precision loss in converting integer {} to float {}",
                        *i, value
                    );
                }

                Ok(value)
            }
            Expression::Float(f) => Ok(*f),
            Expression::Pi => Ok(std::f64::consts::PI),
            Expression::BinaryOp(left, op, right) => {
                let left_val = left.evaluate()?;
                let right_val = right.evaluate()?;
                match op.as_str() {
                    "+" => Ok(left_val + right_val),
                    "-" => Ok(left_val - right_val),
                    "*" => Ok(left_val * right_val),
                    "/" => Ok(left_val / right_val),
                    // Add more binary operators
                    "&" => Ok((left_val as i64 & right_val as i64) as f64),
                    "|" => Ok((left_val as i64 | right_val as i64) as f64),
                    "^" => Ok((left_val as i64 ^ right_val as i64) as f64),
                    "==" => Ok(if left_val == right_val { 1.0 } else { 0.0 }),
                    "!=" => Ok(if left_val != right_val { 1.0 } else { 0.0 }),
                    "<" => Ok(if left_val < right_val { 1.0 } else { 0.0 }),
                    ">" => Ok(if left_val > right_val { 1.0 } else { 0.0 }),
                    "<=" => Ok(if left_val <= right_val { 1.0 } else { 0.0 }),
                    ">=" => Ok(if left_val >= right_val { 1.0 } else { 0.0 }),
                    "<<" => Ok(((left_val as i64) << (right_val as i64)) as f64),
                    ">>" => Ok(((left_val as i64) >> (right_val as i64)) as f64),
                    _ => Err(format!("Unsupported binary operation: {op}").into()),
                }
            }
            Expression::UnaryOp(op, expr) => {
                let val = expr.evaluate()?;
                match op.as_str() {
                    "-" => Ok(-val),
                    "~" => Ok((!(val as i64)) as f64),
                    _ => Err(format!("Unsupported unary operation: {op}").into()),
                }
            }
            Expression::BitId(reg_name, idx) => {
                // We can't evaluate BitId directly because it requires register state
                // This is used in if conditions, so add debugging
                debug!("Cannot evaluate BitId({}, {}) directly - the engine needs to handle this", reg_name, idx);
                Err("Cannot evaluate bit_id directly".into())
            },
            Expression::Variable(_) => Err("Cannot evaluate variable directly".into()),
            Expression::FunctionCall { name, args: _ } => {
                Err(format!("Function calls not implemented yet: {name}").into())
            },
        }
    }
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Integer(i) => write!(f, "{i}"),
            Expression::Float(float_val) => write!(f, "{float_val}"),
            Expression::Pi => write!(f, "pi"),
            Expression::BinaryOp(left, op, right) => write!(f, "({left} {op} {right})"),
            Expression::UnaryOp(op, expr) => write!(f, "({op}{expr})"),
            Expression::BitId(name, idx) => write!(f, "{name}[{idx}]"),
            Expression::Variable(name) => write!(f, "{name}"),
            Expression::FunctionCall { name, args } => {
                write!(f, "{name}(")?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{arg}")?;
                }
                write!(f, ")")
            }
        }
    }
}

#[derive(Debug, Clone)]
pub enum Operation {
    Gate {
        name: String,
        parameters: Vec<f64>,
        arguments: Vec<usize>,
        // Add register name for each qubit
        registers: Vec<String>,
    },
    Measure {
        qubit: usize,
        q_reg: String,
        bit: usize,
        c_reg: String,
    },
    If {
        condition: Expression,
        operation: Box<Operation>,
    },
    Reset {
        qubit: usize,
    },
    Barrier {
        qubits: Vec<usize>,
    },
    RegMeasure {
        q_reg: String,
        c_reg: String,
    },
    // Added to support classical operations
    ClassicalAssignment {
        target: String,        // Register name or bit
        is_indexed: bool,      // Is this a bit_id or just register
        index: Option<usize>,  // Index if it's a bit_id
        expression: Expression,
    },
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::Gate {
                name,
                parameters,
                arguments,
                registers,
            } => {
                write!(f, "{name}(")?;
                for (i, param) in parameters.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{param}")?;
                }
                write!(f, ")")?;

                // Use actual register names if available
                for (i, arg) in arguments.iter().enumerate() {
                    let reg_name = if i < registers.len() {
                        &registers[i]
                    } else {
                        "q" // Fallback to "q" if register name isn't available
                    };
                    write!(f, " {reg_name}[{arg}]")?;
                }
                Ok(())
            }
            Operation::Measure {
                qubit,
                q_reg: _,
                bit,
                c_reg: _,
            } => {
                write!(f, "measure q[{qubit}] -> c[{bit}]")
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
                for qubit in qubits {
                    write!(f, " q[{qubit}]")?;
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
    pub quantum_registers: HashMap<String, usize>,
    pub classical_registers: HashMap<String, usize>,
    pub operations: Vec<Operation>,
    pub gate_definitions: HashMap<String, GateDefinition>,
}

impl QASMParser {
    pub fn parse_file<P: AsRef<Path>>(path: P) -> Result<Program, ParseError> {
        let source = fs::read_to_string(path)?;
        Self::parse_str(&source)
    }

    pub fn parse_str(source: &str) -> Result<Program, ParseError> {
        let mut program = Program::default();
        let mut pairs = Self::parse(Rule::program, source)?;
        let program_pair = pairs
            .next()
            .ok_or_else(|| ParseError::InvalidOperation("Empty program".into()))?;

        for pair in program_pair.into_inner() {
            match pair.as_rule() {
                Rule::oqasm => {
                    for inner in pair.into_inner() {
                        if inner.as_rule() == Rule::version_num {
                            let version = inner.as_str();
                            if version != "2.0" {
                                return Err(ParseError::InvalidVersion(format!(
                                    "Unsupported version: {version}"
                                )));
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

        Ok(program)
    }

    fn parse_statement(
        pair: pest::iterators::Pair<Rule>,
        program: &mut Program,
    ) -> Result<(), ParseError> {
        for inner_pair in pair.into_inner() {
            // Match statements with correct pattern handling
            match inner_pair.as_rule() {
                // Explicitly handle specific rules
                Rule::register_decl => Self::parse_register(inner_pair, program)?,
                Rule::quantum_op => {
                    if let Some(op) = Self::parse_quantum_op(inner_pair)? {
                        program.operations.push(op);
                    }
                }
                Rule::classical_op => {
                    if let Some(op) = Self::parse_classical_operation(inner_pair)? {
                        program.operations.push(op);
                    }
                }
                Rule::if_stmt => {
                    if let Some(op) = Self::parse_if_statement(inner_pair)? {
                        program.operations.push(op);
                    }
                }
                Rule::gate_def => {
                    Self::parse_gate_definition(inner_pair, program)?;
                }
                Rule::include => {
                    Self::parse_include(inner_pair, program)?;
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
    ) -> Result<(), ParseError> {
        let inner = pair.into_inner().next().unwrap();

        #[allow(clippy::match_same_arms)]
        match inner.as_rule() {
            Rule::qreg => {
                let indexed_id = inner.into_inner().next().unwrap();
                let (name, size) = Self::parse_indexed_id(&indexed_id)?;
                program.quantum_registers.insert(name, size);
            }
            Rule::creg => {
                let indexed_id = inner.into_inner().next().unwrap();
                let (name, size) = Self::parse_indexed_id(&indexed_id)?;
                program.classical_registers.insert(name, size);
            }
            _ => {
                return Err(ParseError::InvalidOperation(format!(
                    "Unexpected register type: {:?}",
                    inner.as_rule()
                )));
            }
        }

        Ok(())
    }

    fn parse_quantum_op(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Option<Operation>, ParseError> {
        let inner = pair.into_inner().next().unwrap();

        #[allow(clippy::match_same_arms)]
        match inner.as_rule() {
            Rule::gate_call => {
                let mut inner_pairs = inner.into_inner();
                let gate_name = inner_pairs.next().unwrap().as_str();

                let mut params = Vec::new();
                let mut arguments = Vec::new();
                let mut registers = Vec::new();

                for pair in inner_pairs {
                    match pair.as_rule() {
                        // Handle parameter values
                        Rule::param_values => {
                            for param_expr in pair.into_inner() {
                                if param_expr.as_rule() == Rule::expr {
                                    let expr = Self::parse_expr(param_expr)?;
                                    // Evaluate the expression to a float
                                    let value = expr.evaluate()
                                        .map_err(|e| ParseError::InvalidExpression(format!("Failed to evaluate parameter: {}", e)))?;
                                    params.push(value);
                                }
                            }
                        }
                        // Handle qubit lists - add arguments from qubit IDs
                        Rule::qubit_list => {
                            for qubit_id in pair.into_inner() {
                                if qubit_id.as_rule() == Rule::qubit_id {
                                    let (reg_name, idx) = Self::parse_id_with_index(&qubit_id)?;
                                    arguments.push(idx);
                                    registers.push(reg_name);
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
                    arguments,
                    registers,
                }))
            }
            Rule::measure => Self::parse_measure(inner),
            Rule::reset => Self::parse_reset(inner),
            Rule::barrier => Self::parse_barrier(inner),
            _ => Ok(None),
        }
    }

    fn parse_measure(pair: pest::iterators::Pair<Rule>) -> Result<Option<Operation>, ParseError> {
        let inner_parts: Vec<_> = pair.into_inner().collect();

        if inner_parts.len() == 2 {
            let src = &inner_parts[0];
            let dst = &inner_parts[1];

            if src.as_rule() == Rule::qubit_id && dst.as_rule() == Rule::bit_id {
                let (q_reg, qubit) = Self::parse_id_with_index(&src.clone())?;
                let (c_reg, bit) = Self::parse_id_with_index(&dst.clone())?;

                Ok(Some(Operation::Measure {
                    qubit,
                    q_reg,
                    bit,
                    c_reg,
                }))
            } else if src.as_rule() == Rule::identifier && dst.as_rule() == Rule::identifier {
                Ok(Some(Operation::RegMeasure {
                    q_reg: src.as_str().to_string(),
                    c_reg: dst.as_str().to_string(),
                }))
            } else {
                Err(ParseError::InvalidOperation(
                    "Invalid measurement format".into(),
                ))
            }
        } else {
            Err(ParseError::InvalidOperation(
                "Invalid measurement syntax".into(),
            ))
        }
    }

    fn parse_reset(pair: pest::iterators::Pair<Rule>) -> Result<Option<Operation>, ParseError> {
        let qubit_id = pair.into_inner().next().unwrap();
        let (_, qubit) = Self::parse_id_with_index(&qubit_id)?;

        Ok(Some(Operation::Reset { qubit }))
    }

    fn parse_barrier(pair: pest::iterators::Pair<Rule>) -> Result<Option<Operation>, ParseError> {
        let qubit_list = pair.into_inner().next().unwrap();
        let qubits = Self::parse_qubit_list(qubit_list)?;

        Ok(Some(Operation::Barrier { qubits }))
    }

    // Parse if statement with condition (expression) and operation
    fn parse_if_statement(pair: pest::iterators::Pair<Rule>) -> Result<Option<Operation>, ParseError> {
        // For debugging
        debug!("Parsing if statement: '{}'", pair.as_str());

        // Collect all parts of the if statement
        let parts: Vec<_> = pair.into_inner().collect();

        if parts.len() < 2 {
            return Err(ParseError::InvalidOperation(
                format!("Invalid if statement: expected at least 2 parts, got {}", parts.len())
            ));
        }

        // We expect parts to be: condition_expr, operation
        let condition_expr_pair = &parts[0];
        let operation_pair = &parts[1];

        // Parse the condition expression
        let condition = match condition_expr_pair.as_rule() {
            Rule::condition_expr => {
                // Get the expression inside condition_expr
                let expr_pair = condition_expr_pair.clone().into_inner().next()
                    .ok_or_else(|| ParseError::InvalidOperation("Empty condition expression".to_string()))?;
                Self::parse_expr(expr_pair)?
            },
            _ => {
                return Err(ParseError::InvalidOperation(format!(
                    "Invalid rule in if statement, expected condition_expr, got: {:?}",
                    condition_expr_pair.as_rule()
                )));
            }
        };

        // Parse the operation to be conditionally executed
        let operation = match operation_pair.as_rule() {
                Rule::quantum_op => {
                    if let Some(op) = Self::parse_quantum_op(operation_pair.clone())? {
                        op
                    } else {
                        return Err(ParseError::InvalidOperation(
                            "Invalid quantum operation in if statement".into()
                        ));
                    }
                },
                Rule::classical_op => {
                    if let Some(op) = Self::parse_classical_operation(operation_pair.clone())? {
                        op
                    } else {
                        return Err(ParseError::InvalidOperation(
                            "Invalid classical operation in if statement".into()
                        ));
                    }
                },
                _ => {
                    return Err(ParseError::InvalidOperation(format!(
                        "Unsupported operation type in if statement: {:?}",
                        operation_pair.as_rule()
                    )));
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
    ) -> Result<Option<Operation>, ParseError> {
        // For debugging
        eprintln!("Parsing classical op: '{}'", pair.as_str());

        // Get the inner pairs: 1) target (identifier or bit_id) and 2) expression
        let inner_parts: Vec<_> = pair.into_inner().collect();

        // Debug print all inner parts
        for (i, part) in inner_parts.iter().enumerate() {
            eprintln!("  Part {}: rule={:?}, text='{}'", i, part.as_rule(), part.as_str());
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
                    return Err(ParseError::InvalidOperation(format!(
                        "Invalid classical assignment target: {:?}",
                        target_pair.as_rule()
                    )));
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

        Err(ParseError::InvalidOperation("Invalid classical operation".into()))
    }

    fn parse_qubit_list(pair: pest::iterators::Pair<Rule>) -> Result<Vec<usize>, ParseError> {
        let mut qubits = Vec::new();

        for qubit_id in pair.into_inner() {
            if qubit_id.as_rule() == Rule::qubit_id {
                let (_, index) = Self::parse_id_with_index(&qubit_id)?;
                qubits.push(index);
            }
        }

        Ok(qubits)
    }

    fn parse_indexed_id(pair: &pest::iterators::Pair<Rule>) -> Result<(String, usize), ParseError> {
        let content = pair.as_str();

        if let Some(bracket_pos) = content.find('[') {
            let name = content[0..bracket_pos].to_string();
            let size_str = &content[bracket_pos + 1..content.len() - 1];
            let size = size_str.parse::<usize>()?;
            Ok((name, size))
        } else {
            Err(ParseError::InvalidExpression(format!(
                "Invalid indexed identifier: {content}"
            )))
        }
    }

    // This function is identical to parse_indexed_id, using a single implementation for both cases
    fn parse_id_with_index(
        pair: &pest::iterators::Pair<Rule>,
    ) -> Result<(String, usize), ParseError> {
        Self::parse_indexed_id(pair)
    }

    // New method to correctly handle binary expressions like a^b, a|b, etc.
    fn parse_binary_expr(pair: Pair<Rule>, default_op: &str) -> Result<Expression, ParseError> {
        // Debug the input pair
        let rule = pair.as_rule();
        eprintln!("parse_binary_expr for rule {:?} with text '{}'", rule, pair.as_str());

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
                Rule::equality_op | Rule::relational_op | Rule::shift_op | Rule::add_op | Rule::mul_op => {
                    // This is an explicit operator, next pair should be the operand
                    if i + 1 < inner_pairs.len() {
                        let op_str = next_pair.as_str();
                        let right = Self::parse_expr(inner_pairs[i + 1].clone())?;
                        i += 2; // Skip both operator and operand
                        (op_str, right)
                    } else {
                        return Err(ParseError::InvalidExpression("Missing right operand for binary operation".into()));
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

            result = Expression::BinaryOp(Box::new(result), actual_op.to_string(), Box::new(right_expr));
        }

        Ok(result)
    }

    fn parse_expr(pair: Pair<Rule>) -> Result<Expression, ParseError> {
        // Debug the input pair
        eprintln!("parse_expr: Rule {:?}, Text: '{}'", pair.as_rule(), pair.as_str());
        
        match pair.as_rule() {
            // Handle all expression types based on our updated grammar

            // Top-level expression rule
            Rule::expr => {
                let inner = pair.into_inner().next().ok_or_else(||
                    ParseError::InvalidExpression("Empty expression".into()))?;
                Self::parse_expr(inner)
            },

            // Binary operations - explicitly map each rule to parse_binary_expr
            Rule::b_or_expr => Self::parse_binary_expr(pair, "|"),
            Rule::b_xor_expr => Self::parse_binary_expr(pair, "^"),
            Rule::b_and_expr => Self::parse_binary_expr(pair, "&"),
            Rule::equality_expr => Self::parse_binary_expr(pair, "=="),
            Rule::relational_expr => Self::parse_binary_expr(pair, "<"),
            Rule::shift_expr => Self::parse_binary_expr(pair, "<<"),
            Rule::additive_expr => Self::parse_binary_expr(pair, "+"),
            Rule::multiplicative_expr => Self::parse_binary_expr(pair, "*"),

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
                                expr = Expression::UnaryOp(op.clone(), Box::new(expr));
                            }
                        } else {
                            expr = Expression::UnaryOp(op.clone(), Box::new(expr));
                        }
                    }

                    Ok(expr)
                } else {
                    Err(ParseError::InvalidExpression("Missing operand for unary operation".into()))
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
                if num_str.contains('.') {
                    Ok(Expression::Float(num_str.parse().map_err(|_| {
                        ParseError::InvalidFloat(num_str.to_string())
                    })?))
                } else {
                    Ok(Expression::Integer(num_str.parse().map_err(|_| {
                        ParseError::InvalidInt(num_str.to_string())
                    })?))
                }
            }

            Rule::int => {
                let int_str = pair.as_str();
                Ok(Expression::Integer(int_str.parse().map_err(|_| {
                    ParseError::InvalidInt(int_str.to_string())
                })?))
            }

            Rule::bit_id => {
                let bit_id = pair.as_str();
                let parts: Vec<&str> = bit_id.split('[').collect();
                let name = parts[0].to_string();
                let idx_str = parts[1].trim_end_matches(']');
                let idx = idx_str
                    .parse()
                    .map_err(|_| ParseError::InvalidInt(idx_str.to_string()))?;
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

            _ => Err(ParseError::InvalidExpr(format!(
                "Unexpected rule in expression: {:?}",
                pair.as_rule()
            ))),
        }
    }

    pub fn parse_param_values(_pair: pest::iterators::Pair<Rule>) -> Result<Vec<f64>, ParseError> {
        let params = Vec::new();
        // For now, just return an empty vector
        // In a real implementation, we'd parse each expr in the param_values
        Ok(params)
    }

    fn parse_gate_definition(
        pair: pest::iterators::Pair<Rule>,
        program: &mut Program,
    ) -> Result<(), ParseError> {
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

    fn parse_gate_def_statement(
        pair: pest::iterators::Pair<Rule>,
    ) -> Result<Option<GateDefOperation>, ParseError> {
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

    fn parse_param_expr(pair: pest::iterators::Pair<Rule>) -> Result<ParameterExpression, ParseError> {
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
            Rule::identifier => {
                Ok(ParameterExpression::Identifier(pair.as_str().to_string()))
            }
            Rule::number => {
                let value = pair.as_str().parse().map_err(|_| ParseError::InvalidNumber)?;
                Ok(ParameterExpression::Constant(value))
            }
            Rule::pi_constant => {
                Ok(ParameterExpression::Pi)
            }
            Rule::additive_expr | Rule::multiplicative_expr | Rule::b_or_expr | Rule::b_xor_expr | Rule::b_and_expr => {
                Self::parse_binary_param_expr(pair)
            }
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
                        expr = ParameterExpression::BinaryOp {
                            op: "-".to_string(),
                            left: Box::new(ParameterExpression::Constant(0.0)),
                            right: Box::new(expr),
                        };
                    }

                    Ok(expr)
                } else {
                    Err(ParseError::InvalidExpression("Expected expression after unary operator".to_string()))
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
                    debug!("Unknown node type in parse_param_expr: {:?}", pair.as_rule());
                    Ok(ParameterExpression::Constant(0.0))
                }
            }
        }
    }

    fn parse_binary_param_expr(pair: pest::iterators::Pair<Rule>) -> Result<ParameterExpression, ParseError> {
        let mut inner = pair.into_inner();
        let left_pair = inner.next().ok_or_else(|| ParseError::InvalidExpression("Expected left operand".to_string()))?;
        let mut left = Self::parse_param_expr(left_pair)?;

        while let Some(op_pair) = inner.next() {
            let op = op_pair.as_str().to_string();
            if inner.peek().is_none() {
                debug!("parse_binary_param_expr: No right operand found after operator {}", op);
            }
            let right_pair = inner.next().ok_or_else(|| ParseError::InvalidExpression("Expected right operand".to_string()))?;
            let right = Self::parse_param_expr(right_pair)?;
            left = ParameterExpression::BinaryOp {
                op,
                left: Box::new(left),
                right: Box::new(right),
            };
        }

        Ok(left)
    }

    fn parse_include(
        pair: pest::iterators::Pair<Rule>,
        program: &mut Program,
    ) -> Result<(), ParseError> {
        let mut inner = pair.into_inner();

        if let Some(string_pair) = inner.next() {
            let filename = string_pair.as_str().trim_matches('"');

            // Try to load the include file
            // First check in the includes directory relative to the source
            let include_paths = vec![
                Path::new("includes").join(filename),
                Path::new(filename).to_path_buf(),
            ];

            for include_path in include_paths {
                if include_path.exists() {
                    let include_content = fs::read_to_string(&include_path)?;

                    // Parse the included file
                    let include_program = Self::parse_str(&include_content)?;

                    // Merge gate definitions
                    for (name, def) in include_program.gate_definitions {
                        program.gate_definitions.insert(name, def);
                    }

                    // Don't include operations from the include file
                    // Only gate definitions should be used

                    break;
                }
            }
        }

        Ok(())
    }

    fn expand_gates(program: &mut Program) -> Result<(), ParseError> {
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

        for operation in &program.operations {
            match operation {
                Operation::Gate { name, parameters, arguments, registers } => {
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
                            arguments,
                            registers,
                            &program.gate_definitions,
                        )?;
                        expanded_operations.extend(expanded);
                    } else {
                        // Keep the original gate if no definition exists
                        expanded_operations.push(operation.clone());
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
        arguments: &[usize],
        registers: &[String],
        all_definitions: &HashMap<String, GateDefinition>,
    ) -> Result<Vec<Operation>, ParseError> {
        let mut expanded = Vec::new();

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
            if i < arguments.len() && i < registers.len() {
                qubit_map.insert(qarg_name.clone(), (arguments[i], registers[i].clone()));
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

            // Substitute qubits
            let mut new_args = Vec::new();
            let mut new_regs = Vec::new();

            for arg_name in &body_op.arguments {
                if let Some((mapped_arg, mapped_reg)) = qubit_map.get(arg_name) {
                    new_args.push(*mapped_arg);
                    new_regs.push(mapped_reg.clone());
                }
            }

            let new_op = Operation::Gate {
                name: mapped_name.clone(),
                parameters: new_params.clone(),
                arguments: new_args.clone(),
                registers: new_regs.clone(),
            };

            // Check if this gate has a definition - if it does, expand it
            if let Some(nested_def) = all_definitions.get(&mapped_name) {
                // Recursively expand non-native gates
                let nested_expanded = Self::expand_gate_call(
                    nested_def,
                    &new_params,
                    &new_args,
                    &new_regs,
                    all_definitions,
                )?;
                expanded.extend(nested_expanded);
            } else {
                // No definition found - keep as is
                expanded.push(new_op);
            }
        }

        Ok(expanded)
    }

    fn evaluate_param_expr(expr: &ParameterExpression, param_map: &HashMap<String, f64>) -> Result<f64, ParseError> {
        match expr {
            ParameterExpression::Constant(value) => Ok(*value),
            ParameterExpression::Pi => Ok(std::f64::consts::PI),
            ParameterExpression::Identifier(name) => {
                param_map.get(name).copied().ok_or_else(|| ParseError::InvalidParameter(name.clone()))
            }
            ParameterExpression::BinaryOp { op, left, right } => {
                let left_val = Self::evaluate_param_expr(left, param_map)?;
                let right_val = Self::evaluate_param_expr(right, param_map)?;
                match op.as_str() {
                    "+" => Ok(left_val + right_val),
                    "-" => Ok(left_val - right_val),
                    "*" => Ok(left_val * right_val),
                    "/" => Ok(left_val / right_val),
                    _ => Err(ParseError::InvalidOperator(op.clone())),
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

        let program = QASMParser::parse_str(qasm)?;

        assert_eq!(program.version, "2.0");
        assert_eq!(program.quantum_registers.get("q"), Some(&2));
        assert_eq!(program.classical_registers.get("c"), Some(&2));
        assert_eq!(program.operations.len(), 4); // 2 gates + 2 measurements

        // Verify the gate operations
        if let Operation::Gate {
            name,
            parameters,
            arguments,
            registers,
        } = &program.operations[0]
        {
            assert_eq!(name, "H");
            assert!(parameters.is_empty());
            assert_eq!(arguments, &[0]);
            assert_eq!(registers, &["q".to_string()]);
        } else {
            panic!("Expected gate operation");
        }

        if let Operation::Gate {
            name,
            parameters,
            arguments,
            registers,
        } = &program.operations[1]
        {
            assert_eq!(name, "cx");
            assert!(parameters.is_empty());
            assert_eq!(arguments, &[0, 1]);
            assert_eq!(registers, &["q".to_string(), "q".to_string()]);
        } else {
            panic!("Expected gate operation");
        }

        // Verify the measure operations
        if let Operation::Measure {
            qubit,
            q_reg,
            bit,
            c_reg,
        } = &program.operations[2]
        {
            assert_eq!(*qubit, 0);
            assert_eq!(*q_reg, "q");
            assert_eq!(*bit, 0);
            assert_eq!(*c_reg, "c");
        } else {
            panic!("Expected measure operation");
        }

        if let Operation::Measure {
            qubit,
            q_reg,
            bit,
            c_reg,
        } = &program.operations[3]
        {
            assert_eq!(*qubit, 1);
            assert_eq!(*q_reg, "q");
            assert_eq!(*bit, 1);
            assert_eq!(*c_reg, "c");
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

        let program = QASMParser::parse_str(qasm)?;

        assert_eq!(program.version, "2.0");
        assert_eq!(program.quantum_registers.get("q"), Some(&1));
        assert_eq!(program.classical_registers.get("c"), Some(&1));
        assert_eq!(program.operations.len(), 3); // h gate + measure + if statement

        // Verify the if statement was parsed
        if let Operation::If { condition, operation } = &program.operations[2] {
            // Verify the condition (c[0] == 1)
            if let Expression::BinaryOp(left, op, right) = condition {
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
            if let Operation::Gate { name, arguments, registers, .. } = &**operation {
                assert_eq!(name, "x");
                assert_eq!(arguments, &[0]);
                assert_eq!(registers, &["q".to_string()]);
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

        let program = QASMParser::parse_str(qasm)?;

        assert_eq!(program.version, "2.0");
        assert_eq!(program.quantum_registers.get("q"), Some(&1));
        assert_eq!(program.classical_registers.get("c"), Some(&1));
        assert_eq!(program.operations.len(), 3); // h gate + measure + if statement

        // Verify the if statement contains a classical assignment
        if let Operation::If { condition: _, operation } = &program.operations[2] {
            if let Operation::ClassicalAssignment { target, is_indexed, index, expression } = &**operation {
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
        
        let program = QASMParser::parse_str(qasm)?;
        
        // Just check that parsing succeeded
        assert_eq!(program.classical_registers.len(), 3);
        assert_eq!(program.operations.len(), 5); // 3 assignments
        
        Ok(())
    }
}