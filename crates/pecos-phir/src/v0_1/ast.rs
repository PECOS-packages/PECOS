use serde::Deserialize;
use std::collections::HashMap;

/// Program structure for PHIR (PECOS High-level Intermediate Representation)
#[derive(Debug, Deserialize, Clone)]
pub struct PHIRProgram {
    pub format: String,
    pub version: String,
    pub metadata: HashMap<String, serde_json::Value>,
    pub ops: Vec<Operation>,
}

/// Represents an operation in the PHIR program
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum Operation {
    /// Variable definition for quantum or classical variables
    VariableDefinition {
        data: String,
        data_type: String,
        variable: String,
        size: usize,
    },
    /// Quantum operation (gates, measurements)
    QuantumOp {
        qop: String,
        #[serde(default)]
        angles: Option<(Vec<f64>, String)>,
        args: Vec<QubitArg>,
        #[serde(default)]
        returns: Vec<(String, usize)>,
        #[serde(default)]
        metadata: Option<HashMap<String, serde_json::Value>>,
    },
    /// Classical operation (e.g., Result for exporting values)
    ClassicalOp {
        cop: String,
        #[serde(default)]
        args: Vec<ArgItem>,
        #[serde(default)]
        returns: Vec<ArgItem>,
        #[serde(default)]
        metadata: Option<HashMap<String, serde_json::Value>>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        function: Option<String>, // For ffcall
    },
    /// Block operation (e.g., sequence, qparallel, if)
    Block {
        block: String,
        #[serde(default)]
        ops: Vec<Operation>,
        #[serde(default)]
        condition: Option<Expression>,
        #[serde(default)]
        true_branch: Option<Vec<Operation>>,
        #[serde(default)]
        false_branch: Option<Vec<Operation>>,
        #[serde(default)]
        metadata: Option<HashMap<String, serde_json::Value>>,
    },
    /// Machine operation (e.g., Idle, Transport)
    MachineOp {
        mop: String,
        #[serde(default)]
        args: Option<Vec<QubitArg>>,
        #[serde(default)]
        duration: Option<(f64, String)>,
        #[serde(default)]
        metadata: Option<HashMap<String, serde_json::Value>>,
    },
    /// Meta instruction (e.g., barrier)
    MetaInstruction {
        meta: String,
        #[serde(default)]
        args: Vec<(String, usize)>,
        #[serde(default)]
        metadata: Option<HashMap<String, serde_json::Value>>,
    },
    /// Comment
    Comment {
        #[serde(rename = "//")]
        comment: String,
    },
}

/// Represents an argument to a quantum operation
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum QubitArg {
    /// Single qubit (var, idx)
    SingleQubit((String, usize)),
    /// Multiple qubits for multi-qubit gates [(var, idx), ...]
    MultipleQubits(Vec<(String, usize)>),
}

/// Represents an argument to a classical operation
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ArgItem {
    /// Indexed argument (var, idx)
    Indexed((String, usize)),
    /// Simple argument (entire register)
    Simple(String),
    /// Integer literal
    Integer(i64),
    /// Expression (for nested expressions)
    Expression(Box<Expression>),
}

/// Represents a classical expression
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum Expression {
    /// Operation with operator and arguments
    Operation { cop: String, args: Vec<ArgItem> },
    /// Variable reference
    Variable(String),
    /// Bit reference
    BitIndex((String, usize)),
    /// Integer literal
    Integer(i64),
}

// Constants for internal register naming
pub const MEASUREMENT_PREFIX: &str = "measurement_";
