use std::collections::HashMap;
use std::fmt;
use pecos_core::errors::PecosError;

/// Represents a complete QASM program
#[derive(Debug, Clone)]
pub struct QASMProgram {
    /// QASM version (e.g., "2.0")
    pub version: String,
    /// List of included files
    pub includes: Vec<String>,
    /// Quantum register declarations
    pub quantum_registers: HashMap<String, usize>,
    /// Classical register declarations
    pub classical_registers: HashMap<String, usize>,
    /// List of operations in the program
    pub operations: Vec<Operation>,
    /// Gate definitions from included files
    pub gate_definitions: HashMap<String, GateDefinition>,
    /// Opaque gate declarations
    pub opaque_gates: HashMap<String, OpaqueGateDefinition>,
}

/// Represents a gate definition
#[derive(Debug, Clone)]
pub struct GateDefinition {
    /// Name of the gate
    pub name: String,
    /// Parameter names (if any)
    pub params: Vec<String>,
    /// Qubit argument names
    pub qargs: Vec<String>,
    /// Gate body (list of operations)
    pub body: Vec<GateOperation>,
}

/// Represents an opaque gate declaration
#[derive(Debug, Clone)]
pub struct OpaqueGateDefinition {
    /// Name of the gate
    pub name: String,
    /// Parameter names (if any)
    pub params: Vec<String>,
    /// Qubit argument names
    pub qargs: Vec<String>,
}

/// Represents an operation within a gate definition
#[derive(Debug, Clone)]
pub enum GateOperation {
    /// A gate call within the definition
    GateCall {
        name: String,
        params: Vec<Expression>,
        qargs: Vec<String>,
    },
}

// GateExpression is now replaced by the unified Expression type

/// Represents different types of operations in a QASM program
#[derive(Debug, Clone)]
pub enum Operation {
    /// Quantum gate operation
    QuantumGate {
        /// Name of the gate
        name: String,
        /// List of qubit arguments (register name, index)
        qubits: Vec<String>,
        /// Optional parameters for parameterized gates
        params: Vec<Expression>,
    },
    /// Measurement operation
    Measure {
        /// Qubit to measure (register name, index)
        qubit: String,
        /// Classical bit to store result (register name, index)
        classical: String,
    },
    /// Conditional operation block
    If {
        /// Condition expression
        condition: Expression,
        /// Operations in the true branch
        operations: Vec<Operation>,
    },
    /// Classical operation
    Classical {
        /// Expression to evaluate
        expr: Expression,
    },
}

/// Represents expressions in classical operations
#[derive(Debug, Clone)]
pub enum Expression {
    /// Integer literal
    Integer(i64),
    /// Float literal
    Float(f64),
    /// Mathematical constant pi
    Pi,
    /// Variable reference (parameter or register name)
    Variable(String),
    /// Register bit reference (register name, index)
    BitId(String, i64),
    /// Binary operation
    BinaryOp {
        /// Operation type (e.g., "+", "-", "==", etc.)
        op: String,
        /// Left operand
        left: Box<Expression>,
        /// Right operand
        right: Box<Expression>,
    },
    /// Unary operation
    UnaryOp {
        /// Operation type (e.g., "~", "-", etc.)
        op: String,
        /// Operand
        expr: Box<Expression>,
    },
    /// Function call
    FunctionCall {
        /// Function name
        name: String,
        /// Arguments
        args: Vec<Expression>,
    },
}

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expression::Integer(val) => write!(f, "{}", val),
            Expression::Float(val) => write!(f, "{}", val),
            Expression::Pi => write!(f, "pi"),
            Expression::Variable(name) => write!(f, "{}", name),
            Expression::BitId(reg_name, idx) => write!(f, "{}[{}]", reg_name, idx),
            Expression::BinaryOp { op, left, right } => write!(f, "({} {} {})", left, op, right),
            Expression::UnaryOp { op, expr } => write!(f, "{}({})", op, expr),
            Expression::FunctionCall { name, args } => {
                write!(f, "{}(", name)?;
                for (i, arg) in args.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{}", arg)?;
                }
                write!(f, ")")
            }
        }
    }
}

impl Expression {
    pub fn evaluate(&self) -> Result<f64, PecosError> {
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
            Expression::BinaryOp { op, left, right } => {
                let left_val = left.evaluate()?;
                let right_val = right.evaluate()?;
                match op.as_str() {
                    "+" => Ok(left_val + right_val),
                    "-" => Ok(left_val - right_val),
                    "*" => Ok(left_val * right_val),
                    "/" => Ok(left_val / right_val),
                    "**" => Ok(left_val.powf(right_val)),
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
                    _ => Err(PecosError::ParseInvalidExpression(format!(
                        "Unsupported binary operation: {}",
                        op
                    ))),
                }
            }
            Expression::UnaryOp { op, expr } => {
                let val = expr.evaluate()?;
                match op.as_str() {
                    "-" => Ok(-val),
                    "~" => Ok((!(val as i64)) as f64),
                    _ => Err(PecosError::ParseInvalidExpression(format!(
                        "Unsupported unary operation: {}",
                        op
                    ))),
                }
            }
            Expression::BitId(reg_name, idx) => {
                // We can't evaluate BitId directly because it requires register state
                // This is used in if conditions
                Err(PecosError::ParseInvalidExpression(format!(
                    "Cannot evaluate BitId({}, {}) directly - requires register state",
                    reg_name, idx
                )))
            }
            Expression::Variable(_) => Err(PecosError::ParseInvalidExpression(
                "Cannot evaluate variable directly".to_string(),
            )),
            Expression::FunctionCall { name, args } => {
                if args.len() != 1 {
                    return Err(PecosError::ParseInvalidExpression(format!(
                        "Function {} expects exactly 1 argument, got {}",
                        name,
                        args.len()
                    )));
                }

                let arg_val = args[0].evaluate()?;

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
        }
    }
}

impl QASMProgram {
    /// Creates a new empty QASM program
    #[must_use]
    pub fn new(version: String) -> Self {
        Self {
            version,
            includes: Vec::new(),
            quantum_registers: HashMap::new(),
            classical_registers: HashMap::new(),
            operations: Vec::new(),
            gate_definitions: HashMap::new(),
            opaque_gates: HashMap::new(),
        }
    }

    /// Adds a quantum register declaration
    pub fn add_quantum_register(&mut self, name: String, size: usize) {
        self.quantum_registers.insert(name, size);
    }

    /// Adds a classical register declaration
    pub fn add_classical_register(&mut self, name: String, size: usize) {
        self.classical_registers.insert(name, size);
    }

    /// Adds an operation to the program
    pub fn add_operation(&mut self, operation: Operation) {
        self.operations.push(operation);
    }

    /// Adds an opaque gate declaration
    pub fn add_opaque_gate(&mut self, name: String, params: Vec<String>, qargs: Vec<String>) {
        let opaque_gate = OpaqueGateDefinition {
            name: name.clone(),
            params,
            qargs,
        };
        self.opaque_gates.insert(name, opaque_gate);
    }
}

impl Default for QASMProgram {
    fn default() -> Self {
        Self::new("2.0".to_string())
    }
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Operation::QuantumGate {
                name,
                params,
                qubits,
            } => {
                write!(f, "{name}")?;
                if !params.is_empty() {
                    write!(f, "(")?;
                    for (i, param) in params.iter().enumerate() {
                        if i > 0 {
                            write!(f, ", ")?;
                        }
                        write!(f, "{param}")?;
                    }
                    write!(f, ")")?;
                }
                write!(f, " ")?;
                for (i, qubit) in qubits.iter().enumerate() {
                    if i > 0 {
                        write!(f, ", ")?;
                    }
                    write!(f, "{qubit}")?;
                }
                Ok(())
            }
            Operation::Measure { qubit, classical } => {
                write!(f, "measure {qubit} -> {classical}")
            }
            Operation::If {
                condition,
                operations,
            } => {
                write!(f, "if ({condition}) {{")?;
                for op in operations {
                    write!(f, " {op};")?;
                }
                write!(f, " }}")
            }
            Operation::Classical { expr } => {
                write!(f, "{expr}")
            }
        }
    }
}
