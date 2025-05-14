use std::collections::HashMap;
use std::fmt;

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

/// Represents an operation within a gate definition
#[derive(Debug, Clone)]
pub enum GateOperation {
    /// A gate call within the definition
    GateCall {
        name: String,
        params: Vec<GateExpression>,
        qargs: Vec<String>,
    },
}

/// Represents an expression within a gate definition
#[derive(Debug, Clone)]
pub enum GateExpression {
    /// A parameter reference
    Parameter(String),
    /// A constant value
    Constant(f64),
    /// A binary operation
    BinaryOp {
        op: String,
        left: Box<GateExpression>,
        right: Box<GateExpression>,
    },
    /// Pi constant
    Pi,
}

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
    /// Variable reference (register name, index)
    Variable(String),
    /// Numeric literal
    Literal(f64),
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
            Expression::Variable(name) => write!(f, "{name}"),
            Expression::Literal(value) => write!(f, "{value}"),
            Expression::BinaryOp { op, left, right } => write!(f, "({left} {op} {right})"),
            Expression::UnaryOp { op, expr } => write!(f, "{op}({expr})"),
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
