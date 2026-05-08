use serde::Deserialize;
use std::collections::BTreeMap;
use std::f64::consts::PI;

/// Program structure for PHIR (PECOS High-level Intermediate Representation)
#[derive(Debug, Deserialize, Clone)]
pub struct PHIRProgram {
    pub format: String,
    pub version: String,
    #[serde(default)]
    pub metadata: BTreeMap<String, serde_json::Value>,
    pub ops: Vec<Operation>,
}

/// Represents an operation in the PHIR program.
///
/// Deserialized via a manual `Deserialize` impl that inspects which discriminating
/// key is present (`qop`, `cop`, `block`, `mop`, `meta`, `data`, `//`).
/// This avoids `#[serde(untagged)]` whose `ContentDeserializer` can silently fail
/// with certain nested types (serde issue 1183).
#[derive(Debug, Clone)]
pub enum Operation {
    /// Variable definition for quantum or classical variables
    VariableDefinition {
        data: String,
        data_type: String,
        variable: String,
        /// Size in bits. Optional -- if omitted, inferred from `data_type`.
        size: Option<usize>,
    },
    /// Quantum operation (gates, measurements)
    QuantumOp {
        qop: String,
        /// Angles in radians (converted from the JSON `[[values...], "unit"]` format)
        angles: Option<Vec<f64>>,
        args: Vec<QubitArg>,
        returns: Vec<(String, usize)>,
        metadata: Option<BTreeMap<String, serde_json::Value>>,
    },
    /// Classical operation (e.g., Result for exporting values)
    ClassicalOp {
        cop: String,
        args: Vec<ArgItem>,
        returns: Vec<ArgItem>,
        metadata: Option<BTreeMap<String, serde_json::Value>>,
        function: Option<String>,
    },
    /// Block operation (e.g., sequence, qparallel, if)
    Block {
        block: String,
        ops: Vec<Operation>,
        condition: Option<Expression>,
        true_branch: Option<Vec<Operation>>,
        false_branch: Option<Vec<Operation>>,
        metadata: Option<BTreeMap<String, serde_json::Value>>,
    },
    /// Machine operation (e.g., Idle, Transport)
    MachineOp {
        mop: String,
        args: Option<Vec<QubitArg>>,
        duration: Option<(f64, String)>,
        metadata: Option<BTreeMap<String, serde_json::Value>>,
    },
    /// Meta instruction (e.g., barrier)
    MetaInstruction {
        meta: String,
        args: Vec<(String, usize)>,
        metadata: Option<BTreeMap<String, serde_json::Value>>,
    },
    /// Data export (`cvar_export`) -- specifies which variables to export
    DataExport {
        data: String,
        variables: Vec<String>,
    },
    /// Comment
    Comment { comment: String },
}

// ---------------------------------------------------------------------------
// Manual Deserialize for Operation -- key-based dispatch on serde_json::Value
// ---------------------------------------------------------------------------

/// Convert raw JSON angles `[[values...], "unit"]` to radians.
fn convert_angles(raw: &serde_json::Value) -> Result<Option<Vec<f64>>, String> {
    if raw.is_null() {
        return Ok(None);
    }
    let arr = raw.as_array().ok_or("angles: expected array")?;
    if arr.len() != 2 {
        return Err(format!(
            "angles: expected [values, unit], got {} elements",
            arr.len()
        ));
    }
    let values = arr[0]
        .as_array()
        .ok_or("angles: first element must be an array of numbers")?
        .iter()
        .map(|v| {
            v.as_f64()
                .ok_or_else(|| format!("angles: expected number, got {v}"))
        })
        .collect::<Result<Vec<f64>, _>>()?;
    let unit = arr[1]
        .as_str()
        .ok_or("angles: second element must be a string")?;
    match unit {
        "rad" => Ok(Some(values)),
        "deg" => Ok(Some(values.into_iter().map(|v| v * PI / 180.0).collect())),
        "pi" => Ok(Some(values.into_iter().map(|v| v * PI).collect())),
        _ => Err(format!("Unsupported angle unit: {unit}")),
    }
}

/// Helper: extract optional metadata from a JSON object.
fn extract_metadata(
    obj: &serde_json::Map<String, serde_json::Value>,
) -> Option<BTreeMap<String, serde_json::Value>> {
    obj.get("metadata").and_then(|v| {
        if v.is_null() {
            None
        } else {
            serde_json::from_value(v.clone()).ok()
        }
    })
}

impl<'de> Deserialize<'de> for Operation {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;

        let val = serde_json::Value::deserialize(deserializer)?;
        let obj = val
            .as_object()
            .ok_or_else(|| D::Error::custom("operation must be a JSON object"))?;

        // Dispatch on the discriminating key
        if let Some(qop_val) = obj.get("qop") {
            // QuantumOp
            let qop = qop_val
                .as_str()
                .ok_or_else(|| D::Error::custom("qop must be a string"))?
                .to_string();
            let angles = obj
                .get("angles")
                .map_or(Ok(None), convert_angles)
                .map_err(D::Error::custom)?;
            let args: Vec<QubitArg> = obj
                .get("args")
                .map_or(Ok(vec![]), |v| serde_json::from_value(v.clone()))
                .map_err(|e| D::Error::custom(format!("args: {e}")))?;
            let returns: Vec<(String, usize)> = obj
                .get("returns")
                .map_or(Ok(vec![]), |v| serde_json::from_value(v.clone()))
                .map_err(|e| D::Error::custom(format!("returns: {e}")))?;
            let metadata = extract_metadata(obj);
            Ok(Operation::QuantumOp {
                qop,
                angles,
                args,
                returns,
                metadata,
            })
        } else if let Some(cop_val) = obj.get("cop") {
            // ClassicalOp
            let cop = cop_val
                .as_str()
                .ok_or_else(|| D::Error::custom("cop must be a string"))?
                .to_string();
            let args: Vec<ArgItem> = obj
                .get("args")
                .map_or(Ok(vec![]), |v| serde_json::from_value(v.clone()))
                .map_err(|e| D::Error::custom(format!("args: {e}")))?;
            let returns: Vec<ArgItem> = obj
                .get("returns")
                .map_or(Ok(vec![]), |v| serde_json::from_value(v.clone()))
                .map_err(|e| D::Error::custom(format!("returns: {e}")))?;
            let metadata = extract_metadata(obj);
            let function: Option<String> = obj
                .get("function")
                .and_then(|v| v.as_str().map(String::from));
            Ok(Operation::ClassicalOp {
                cop,
                args,
                returns,
                metadata,
                function,
            })
        } else if let Some(block_val) = obj.get("block") {
            // Block
            let block = block_val
                .as_str()
                .ok_or_else(|| D::Error::custom("block must be a string"))?
                .to_string();
            let ops: Vec<Operation> = obj
                .get("ops")
                .map_or(Ok(vec![]), |v| serde_json::from_value(v.clone()))
                .map_err(|e| D::Error::custom(format!("ops: {e}")))?;
            let condition: Option<Expression> = obj
                .get("condition")
                .filter(|v| !v.is_null())
                .map(|v| serde_json::from_value(v.clone()))
                .transpose()
                .map_err(|e| D::Error::custom(format!("condition: {e}")))?;
            let true_branch: Option<Vec<Operation>> = obj
                .get("true_branch")
                .filter(|v| !v.is_null())
                .map(|v| serde_json::from_value(v.clone()))
                .transpose()
                .map_err(|e| D::Error::custom(format!("true_branch: {e}")))?;
            let false_branch: Option<Vec<Operation>> = obj
                .get("false_branch")
                .filter(|v| !v.is_null())
                .map(|v| serde_json::from_value(v.clone()))
                .transpose()
                .map_err(|e| D::Error::custom(format!("false_branch: {e}")))?;
            let metadata = extract_metadata(obj);
            Ok(Operation::Block {
                block,
                ops,
                condition,
                true_branch,
                false_branch,
                metadata,
            })
        } else if let Some(mop_val) = obj.get("mop") {
            // MachineOp
            let mop = mop_val
                .as_str()
                .ok_or_else(|| D::Error::custom("mop must be a string"))?
                .to_string();
            let args: Option<Vec<QubitArg>> = obj.get("args").and_then(|v| {
                if v.is_null() {
                    None
                } else {
                    serde_json::from_value(v.clone()).ok()
                }
            });
            let duration: Option<(f64, String)> = obj.get("duration").and_then(|v| {
                if v.is_null() {
                    None
                } else {
                    serde_json::from_value(v.clone()).ok()
                }
            });
            let metadata = extract_metadata(obj);
            Ok(Operation::MachineOp {
                mop,
                args,
                duration,
                metadata,
            })
        } else if let Some(meta_val) = obj.get("meta") {
            // MetaInstruction
            let meta = meta_val
                .as_str()
                .ok_or_else(|| D::Error::custom("meta must be a string"))?
                .to_string();
            let args: Vec<(String, usize)> = obj
                .get("args")
                .map_or(Ok(vec![]), |v| serde_json::from_value(v.clone()))
                .map_err(|e| D::Error::custom(format!("args: {e}")))?;
            let metadata = extract_metadata(obj);
            Ok(Operation::MetaInstruction {
                meta,
                args,
                metadata,
            })
        } else if let Some(comment_val) = obj.get("//") {
            // Comment
            let comment = comment_val
                .as_str()
                .ok_or_else(|| D::Error::custom("comment must be a string"))?
                .to_string();
            Ok(Operation::Comment { comment })
        } else if let Some(data_val) = obj.get("data") {
            let data = data_val
                .as_str()
                .ok_or_else(|| D::Error::custom("data must be a string"))?
                .to_string();
            if obj.contains_key("variables") {
                // DataExport
                let variables: Vec<String> = serde_json::from_value(obj["variables"].clone())
                    .map_err(|e| D::Error::custom(format!("variables: {e}")))?;
                Ok(Operation::DataExport { data, variables })
            } else {
                // VariableDefinition
                let data_type: String = obj
                    .get("data_type")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| D::Error::custom("missing data_type"))?
                    .to_string();
                let variable: String = obj
                    .get("variable")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| D::Error::custom("missing variable"))?
                    .to_string();
                let size: Option<usize> = obj
                    .get("size")
                    .and_then(serde_json::Value::as_u64)
                    .and_then(|n| usize::try_from(n).ok());
                Ok(Operation::VariableDefinition {
                    data,
                    data_type,
                    variable,
                    size,
                })
            }
        } else {
            Err(D::Error::custom(format!(
                "unknown operation: no recognized key (qop, cop, block, mop, meta, //, data) found in {:?}",
                obj.keys().collect::<Vec<_>>()
            )))
        }
    }
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
#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum ArgItem {
    /// Indexed argument (var, idx)
    Indexed((String, usize)),
    /// Simple argument (entire register)
    Simple(String),
    /// Integer literal (signed, covers most cases)
    Integer(i64),
    /// Unsigned integer literal (for values > `i64::MAX`, e.g. `u64::MAX`)
    UInteger(u64),
    /// Expression (for nested expressions)
    Expression(Box<Expression>),
}

/// Represents a classical expression
#[derive(Debug, Deserialize, Clone, PartialEq)]
#[serde(untagged)]
pub enum Expression {
    /// Operation with operator and arguments
    Operation { cop: String, args: Vec<ArgItem> },
    /// Variable reference
    Variable(String),
    /// Integer literal
    Integer(i64),
}

// Constants for internal register naming
pub const MEASUREMENT_PREFIX: &str = "measurement_";

/// Infer variable size from data type when not explicitly provided.
///
/// For types like "i32", "u64", extracts the bit width from the name.
/// For "qubits", returns 0 (size must be explicit).
#[must_use]
pub fn infer_size(data_type: &str, explicit_size: Option<usize>) -> usize {
    if let Some(s) = explicit_size {
        return s;
    }
    // Try to extract bit width from type name (e.g., "i32" -> 32, "u64" -> 64)
    let digits: String = data_type.chars().filter(char::is_ascii_digit).collect();
    digits.parse().unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_qparallel_with_angles() {
        let json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {},
            "ops": [
                {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
                {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 2},
                {"qop": "RZ", "angles": [[1.0], "pi"], "args": [["q", 0], ["q", 1]]},
                {"block": "qparallel", "ops": [
                    {"qop": "R1XY", "angles": [[0.5, 0.5], "pi"], "args": [["q", 0]]},
                    {"qop": "R1XY", "angles": [[1.5, 0.5], "pi"], "args": [["q", 1]]}
                ]},
                {"qop": "RZZ", "angles": [[0.5], "pi"], "args": [[["q", 0], ["q", 1]]]},
                {"qop": "Measure", "args": [["q", 0], ["q", 1]], "returns": [["m", 0], ["m", 1]]}
            ]
        }"#;
        let program: PHIRProgram = serde_json::from_str(json).expect("should parse");
        assert_eq!(program.ops.len(), 6);

        // Verify angles were converted to radians (pi units)
        if let Operation::QuantumOp { angles, .. } = &program.ops[2] {
            let a = angles.as_ref().unwrap();
            assert!((a[0] - PI).abs() < 1e-10, "RZ angle should be pi");
        } else {
            panic!("Expected QuantumOp");
        }

        // Verify inner ops of qparallel block
        if let Operation::Block { ops, .. } = &program.ops[3] {
            assert_eq!(ops.len(), 2);
            if let Operation::QuantumOp { qop, angles, .. } = &ops[0] {
                assert_eq!(qop, "R1XY");
                let a = angles.as_ref().unwrap();
                assert_eq!(a.len(), 2);
                assert!((a[0] - 0.5 * PI).abs() < 1e-10);
                assert!((a[1] - 0.5 * PI).abs() < 1e-10);
            } else {
                panic!("Expected QuantumOp inside block");
            }
        } else {
            panic!("Expected Block");
        }
    }

    #[test]
    fn test_parse_bell_qparallel_compact() {
        // Simulate what Python json.dumps produces (compact, single-line)
        let json = r#"{"format": "PHIR/JSON", "version": "0.1.0", "metadata": {"source": "pytket-phir v0.2.0", "strict_parallelism": "true"}, "ops": [{"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2}, {"data": "cvar_define", "data_type": "u32", "variable": "m", "size": 2}, {"qop": "RZ", "angles": [[1.0], "pi"], "args": [["q", 0], ["q", 1]]}, {"block": "qparallel", "ops": [{"qop": "R1XY", "angles": [[0.5, 0.5], "pi"], "args": [["q", 0]]}, {"qop": "R1XY", "angles": [[1.5, 0.5], "pi"], "args": [["q", 1]]}]}, {"qop": "RZ", "angles": [[1.0], "pi"], "args": [["q", 0]]}, {"qop": "RZZ", "angles": [[0.5], "pi"], "args": [[["q", 0], ["q", 1]]]}, {"block": "qparallel", "ops": [{"qop": "RZ", "angles": [[1.5], "pi"], "args": [["q", 0]]}, {"qop": "RZ", "angles": [[0.5], "pi"], "args": [["q", 1]]}]}, {"qop": "R1XY", "angles": [[1.5, 0.5], "pi"], "args": [["q", 1]]}, {"qop": "Measure", "args": [["q", 0], ["q", 1]], "returns": [["m", 0], ["m", 1]]}]}"#;
        let program: PHIRProgram = serde_json::from_str(json).expect("should parse");
        assert_eq!(program.ops.len(), 9);
    }

    #[test]
    fn test_angle_units() {
        // Test all three angle unit types
        let json = r#"{
            "format": "PHIR/JSON", "version": "0.1.0", "metadata": {},
            "ops": [
                {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
                {"qop": "RZ", "angles": [[1.5707963267948966], "rad"], "args": [["q", 0]]},
                {"qop": "RZ", "angles": [[90.0], "deg"], "args": [["q", 0]]},
                {"qop": "RZ", "angles": [[0.5], "pi"], "args": [["q", 0]]}
            ]
        }"#;
        let program: PHIRProgram = serde_json::from_str(json).expect("should parse");

        // All three should produce the same angle (pi/2)
        for i in 1..=3 {
            if let Operation::QuantumOp { angles, .. } = &program.ops[i] {
                let a = angles.as_ref().unwrap();
                assert!(
                    (a[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-10,
                    "op {i}: expected pi/2, got {}",
                    a[0]
                );
            }
        }
    }

    #[test]
    fn test_no_angles() {
        let json = r#"{
            "format": "PHIR/JSON", "version": "0.1.0", "metadata": {},
            "ops": [
                {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
                {"qop": "H", "args": [["q", 0]]},
                {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]}
            ]
        }"#;
        let program: PHIRProgram = serde_json::from_str(json).expect("should parse");
        if let Operation::QuantumOp { angles, .. } = &program.ops[1] {
            assert!(angles.is_none());
        }
    }

    #[test]
    fn test_comment() {
        let json = r#"{
            "format": "PHIR/JSON", "version": "0.1.0", "metadata": {},
            "ops": [{"//": "this is a comment"}]
        }"#;
        let program: PHIRProgram = serde_json::from_str(json).expect("should parse");
        if let Operation::Comment { comment } = &program.ops[0] {
            assert_eq!(comment, "this is a comment");
        } else {
            panic!("Expected Comment");
        }
    }

    #[test]
    fn test_data_export() {
        let json = r#"{
            "format": "PHIR/JSON", "version": "0.1.0", "metadata": {},
            "ops": [{"data": "cvar_export", "variables": ["m", "n"]}]
        }"#;
        let program: PHIRProgram = serde_json::from_str(json).expect("should parse");
        if let Operation::DataExport { data, variables } = &program.ops[0] {
            assert_eq!(data, "cvar_export");
            assert_eq!(variables, &["m", "n"]);
        } else {
            panic!("Expected DataExport");
        }
    }
}
