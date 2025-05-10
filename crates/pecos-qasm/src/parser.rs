use pest::Parser;
use pest::iterators::Pair;
use pest_derive::Parser;
use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::Path;

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
    BitId(String, i64),
    FunctionCall(String, Vec<Expression>),
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
                    _ => Err(format!("Unsupported binary operation: {op}").into()),
                }
            }
            Expression::BitId(_, _) => Err("Cannot evaluate bit_id directly".into()),
            Expression::FunctionCall(_, _) => Err("Function calls not implemented yet".into()),
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
            Expression::BitId(name, idx) => write!(f, "{name}[{idx}]"),
            Expression::FunctionCall(name, args) => {
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
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::Gate {
                name,
                parameters,
                arguments,
            } => {
                write!(f, "{name}(")?;
                for (i, param) in parameters.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{param}")?;
                }
                write!(f, ")")?;
                for arg in arguments {
                    write!(f, " q[{arg}]")?;
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
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct Program {
    pub version: String,
    pub quantum_registers: HashMap<String, usize>,
    pub classical_registers: HashMap<String, usize>,
    pub operations: Vec<Operation>,
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
                // Rules that are recognized but not yet implemented (including Rule::include)
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

                let params = Vec::new();
                let mut arguments = Vec::new();

                for pair in inner_pairs {
                    match pair.as_rule() {
                        // Handle qubit lists - add arguments from qubit IDs
                        Rule::qubit_list => {
                            for qubit_id in pair.into_inner() {
                                if qubit_id.as_rule() == Rule::qubit_id {
                                    let (_, idx) = Self::parse_id_with_index(&qubit_id)?;
                                    arguments.push(idx);
                                }
                            }
                        }
                        // Unhandled rule types (including param_values which we'll implement later)
                        _ => {
                            // Skip unimplemented rules for now
                        }
                    }
                }

                Ok(Some(Operation::Gate {
                    name: gate_name.to_string(),
                    parameters: params,
                    arguments,
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

    #[allow(dead_code)]
    fn parse_expr(pair: Pair<Rule>) -> Result<Expression, ParseError> {
        match pair.as_rule() {
            Rule::expr => {
                let mut pairs = pair.into_inner();
                let mut left = Self::parse_expr(pairs.next().unwrap())?;

                while let Some(op_pair) = pairs.next() {
                    let op = op_pair.as_str().to_string();
                    let right = Self::parse_expr(pairs.next().unwrap())?;
                    left = Expression::BinaryOp(Box::new(left), op, Box::new(right));
                }

                Ok(left)
            }
            Rule::term => {
                let mut pairs = pair.into_inner();
                let mut left = Self::parse_expr(pairs.next().unwrap())?;

                while let Some(op_pair) = pairs.next() {
                    let op = op_pair.as_str().to_string();
                    let right = Self::parse_expr(pairs.next().unwrap())?;
                    left = Expression::BinaryOp(Box::new(left), op, Box::new(right));
                }

                Ok(left)
            }
            Rule::factor => {
                let inner = pair.into_inner().next().unwrap();
                Self::parse_expr(inner)
            }
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
        } = &program.operations[0]
        {
            assert_eq!(name, "h");
            assert!(parameters.is_empty());
            assert_eq!(arguments, &[0]);
        } else {
            panic!("Expected gate operation");
        }

        if let Operation::Gate {
            name,
            parameters,
            arguments,
        } = &program.operations[1]
        {
            assert_eq!(name, "cx");
            assert!(parameters.is_empty());
            assert_eq!(arguments, &[0, 1]);
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
        assert_eq!(program.operations.len(), 2); // 1 gate + 1 measurement (if statement is not parsed yet)

        Ok(())
    }
}
