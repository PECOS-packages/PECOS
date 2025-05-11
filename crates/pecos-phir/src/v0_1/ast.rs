use serde::Deserialize;
use std::collections::HashMap;

/// Program structure for PHIR (PECOS High-level Intermediate Representation)
#[derive(Debug, Deserialize, Clone)]
pub struct PHIRProgram {
    pub format: String,
    pub version: String,
    pub metadata: HashMap<String, String>,
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
        args: Vec<(String, usize)>,
        #[serde(default)]
        returns: Vec<(String, usize)>,
    },
    /// Classical operation (e.g., Result for exporting values)
    ClassicalOp {
        cop: String,
        #[serde(default)]
        args: Vec<ArgItem>,
        #[serde(default)]
        returns: Vec<ArgItem>,
    },
}

/// Represents an argument to a classical operation
#[derive(Debug, Deserialize, Clone)]
#[serde(untagged)]
pub enum ArgItem {
    /// Indexed argument (var, idx)
    Indexed((String, usize)),
    /// Simple argument (entire register)
    Simple(String),
}

// Constants for internal register naming
pub const MEASUREMENT_PREFIX: &str = "measurement_";
