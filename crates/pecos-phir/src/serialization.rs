/*!
Serialization support for PHIR to/from PHIR-JSON format

This module provides conversion between PHIR's in-memory representation
and PHIR-JSON, which serves as the stable, human-readable serialization
format for PHIR.

Key principles:
- PHIR-JSON is a direct serialization of PHIR concepts
- The JSON format is versioned and stable across PHIR internal changes
- Human-readable while maintaining full fidelity
- Bidirectional conversion without information loss
*/

use crate::phir::{Module, Region, Block, Instruction, AttributeValue};
use crate::builtin_ops::{ModuleOp, FuncOp, BuiltinOp};
use crate::ops::{Operation, QuantumOp, ClassicalOp, SSAValue};
use crate::types::Type;
use serde::{Serialize, Deserialize};
use serde_json::Value as JsonValue;
use std::collections::BTreeMap;

/// PHIR-JSON format version
pub const PHIR_JSON_VERSION: &str = "0.2.0";

/// Top-level PHIR-JSON structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhirJson {
    /// Format identifier
    pub format: String,
    /// Version of the PHIR-JSON format
    pub version: String,
    /// Optional metadata
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, JsonValue>>,
    /// The PHIR module
    pub module: PhirModule,
}

/// PHIR representation of a PHIR Module
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhirModule {
    pub name: String,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub attributes: BTreeMap<String, JsonValue>,
    pub body: PhirRegion,
}

/// PHIR representation of a PHIR Region
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhirRegion {
    pub kind: String,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub attributes: BTreeMap<String, JsonValue>,
    pub blocks: Vec<PhirBlock>,
}

/// PHIR representation of a PHIR Block
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhirBlock {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub arguments: Vec<PhirBlockArg>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub attributes: BTreeMap<String, JsonValue>,
    pub ops: Vec<PhirOperation>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub terminator: Option<PhirTerminator>,
}

/// Block argument in PHIR
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhirBlockArg {
    pub name: String,
    #[serde(rename = "type")]
    pub ty: String,
}

/// PHIR representation of a PHIR Operation
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PhirOperation {
    /// Quantum operation
    Quantum {
        qop: String,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        args: Vec<String>,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        returns: Vec<String>,
        #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
        attributes: BTreeMap<String, JsonValue>,
    },
    /// Classical operation
    Classical {
        cop: String,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        args: Vec<String>,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        returns: Vec<String>,
        #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
        attributes: BTreeMap<String, JsonValue>,
    },
    /// Function definition
    Function {
        #[serde(rename = "function")]
        func: PhirFunction,
    },
    /// Block (for nested regions)
    Block {
        block: PhirBlock,
    },
    /// Comment
    Comment {
        #[serde(rename = "//")]
        text: String,
    },
}

/// PHIR representation of a function
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhirFunction {
    pub name: String,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub inputs: Vec<PhirType>,
    #[serde(skip_serializing_if = "Vec::is_empty", default)]
    pub outputs: Vec<PhirType>,
    #[serde(skip_serializing_if = "BTreeMap::is_empty", default)]
    pub attributes: BTreeMap<String, JsonValue>,
    pub body: Vec<PhirRegion>,
}

/// PHIR representation of types
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PhirType {
    /// Simple type name
    Simple(String),
    /// Parameterized type
    Parameterized {
        name: String,
        params: Vec<PhirType>,
    },
}

/// PHIR representation of terminators
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "terminator")]
pub enum PhirTerminator {
    #[serde(rename = "return")]
    Return {
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        values: Vec<String>,
    },
    #[serde(rename = "branch")]
    Branch {
        target: String,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        args: Vec<String>,
    },
    #[serde(rename = "cond_branch")]
    ConditionalBranch {
        condition: String,
        true_target: String,
        false_target: String,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        true_args: Vec<String>,
        #[serde(skip_serializing_if = "Vec::is_empty", default)]
        false_args: Vec<String>,
    },
}

/// Convert PHIR Module to PHIR-JSON
pub fn module_to_phir_json(module: &ModuleOp) -> Result<PhirJson, serde_json::Error> {
    Ok(PhirJson {
        format: "PHIR/JSON".to_string(),
        version: PHIR_JSON_VERSION.to_string(),
        metadata: None,
        module: PhirModule {
            name: module.name.clone(),
            attributes: attributes_to_json(&module.attributes),
            body: region_to_phir(&module.body)?,
        },
    })
}

/// Convert PHIR Region to PHIR representation
fn region_to_phir(region: &Region) -> Result<PhirRegion, serde_json::Error> {
    Ok(PhirRegion {
        kind: format!("{:?}", region.kind),
        attributes: attributes_to_json(&region.attributes),
        blocks: region.blocks.iter()
            .map(block_to_phir)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

/// Convert PHIR Block to PHIR representation
fn block_to_phir(block: &Block) -> Result<PhirBlock, serde_json::Error> {
    Ok(PhirBlock {
        label: block.label.clone(),
        arguments: block.arguments.iter()
            .map(|arg| PhirBlockArg {
                name: format!("{}", arg.value),
                ty: format!("{}", arg.ty),
            })
            .collect(),
        attributes: attributes_to_json(&block.attributes),
        ops: block.operations.iter()
            .map(instruction_to_phir)
            .collect::<Result<Vec<_>, _>>()?,
        terminator: block.terminator.as_ref().map(|t| terminator_to_phir(t)).transpose()?,
    })
}

/// Convert PHIR Instruction to PHIR operation
fn instruction_to_phir(inst: &Instruction) -> Result<PhirOperation, serde_json::Error> {
    match &inst.operation {
        Operation::Quantum(qop) => {
            Ok(PhirOperation::Quantum {
                qop: quantum_op_name(qop),
                args: inst.operands.iter().map(|v| format!("{}", v)).collect(),
                returns: inst.results.iter().map(|v| format!("{}", v)).collect(),
                attributes: BTreeMap::new(), // TODO: Add instruction attributes
            })
        }
        Operation::Classical(cop) => {
            Ok(PhirOperation::Classical {
                cop: classical_op_name(cop),
                args: inst.operands.iter().map(|v| format!("{}", v)).collect(),
                returns: inst.results.iter().map(|v| format!("{}", v)).collect(),
                attributes: BTreeMap::new(),
            })
        }
        Operation::Builtin(BuiltinOp::Func(func)) => {
            Ok(PhirOperation::Function {
                func: function_to_phir(func)?,
            })
        }
        // TODO: Handle other operation types
        _ => {
            // For now, represent unknown ops as comments
            Ok(PhirOperation::Comment {
                text: format!("TODO: Serialize {:?}", inst.operation),
            })
        }
    }
}

/// Convert function to PHIR representation
fn function_to_phir(func: &FuncOp) -> Result<PhirFunction, serde_json::Error> {
    Ok(PhirFunction {
        name: func.name.clone(),
        inputs: func.function_type.inputs.iter()
            .map(|t| PhirType::Simple(format!("{}", t)))
            .collect(),
        outputs: func.function_type.outputs.iter()
            .map(|t| PhirType::Simple(format!("{}", t)))
            .collect(),
        attributes: attributes_to_json(&func.attributes),
        body: func.body.iter()
            .map(region_to_phir)
            .collect::<Result<Vec<_>, _>>()?,
    })
}

/// Convert terminator to PHIR representation
fn terminator_to_phir(term: &crate::phir::Terminator) -> Result<PhirTerminator, serde_json::Error> {
    use crate::phir::Terminator;
    match term {
        Terminator::Return { values } => {
            Ok(PhirTerminator::Return {
                values: values.iter().map(|v| format!("{}", v)).collect(),
            })
        }
        Terminator::Branch { target, args } => {
            Ok(PhirTerminator::Branch {
                target: target.to_string(),
                args: args.iter().map(|v| format!("{}", v)).collect(),
            })
        }
        Terminator::ConditionalBranch { condition, true_target, true_args, false_target, false_args } => {
            Ok(PhirTerminator::ConditionalBranch {
                condition: format!("{}", condition),
                true_target: true_target.to_string(),
                false_target: false_target.to_string(),
                true_args: true_args.iter().map(|v| format!("{}", v)).collect(),
                false_args: false_args.iter().map(|v| format!("{}", v)).collect(),
            })
        }
    }
}

/// Convert attributes to JSON
fn attributes_to_json(attrs: &BTreeMap<String, AttributeValue>) -> BTreeMap<String, JsonValue> {
    attrs.iter()
        .map(|(k, v)| (k.clone(), attribute_value_to_json(v)))
        .collect()
}

/// Convert AttributeValue to JSON
fn attribute_value_to_json(value: &AttributeValue) -> JsonValue {
    match value {
        AttributeValue::Bool(b) => JsonValue::Bool(*b),
        AttributeValue::Int(i) => JsonValue::Number((*i).into()),
        AttributeValue::Float(f) => JsonValue::Number(
            serde_json::Number::from_f64(*f).unwrap_or_else(|| serde_json::Number::from(0))
        ),
        AttributeValue::String(s) => JsonValue::String(s.clone()),
        AttributeValue::Array(arr) => JsonValue::Array(
            arr.iter().map(attribute_value_to_json).collect()
        ),
        AttributeValue::Dict(map) => JsonValue::Object(
            map.iter()
                .map(|(k, v)| (k.clone(), attribute_value_to_json(v)))
                .collect()
        ),
    }
}

/// Get quantum operation name
fn quantum_op_name(qop: &QuantumOp) -> String {
    match qop {
        QuantumOp::H => "H".to_string(),
        QuantumOp::X => "X".to_string(),
        QuantumOp::Y => "Y".to_string(),
        QuantumOp::Z => "Z".to_string(),
        QuantumOp::CNOT => "CNOT".to_string(),
        QuantumOp::Measure => "Measure".to_string(),
        QuantumOp::StatePrep => "StatePrep".to_string(),
    }
}

/// Get classical operation name
fn classical_op_name(cop: &ClassicalOp) -> String {
    match cop {
        ClassicalOp::Add => "Add".to_string(),
        ClassicalOp::Sub => "Sub".to_string(),
        ClassicalOp::Mul => "Mul".to_string(),
        ClassicalOp::Div => "Div".to_string(),
        ClassicalOp::Eq => "Eq".to_string(),
        ClassicalOp::Lt => "Lt".to_string(),
        ClassicalOp::And => "And".to_string(),
        ClassicalOp::Or => "Or".to_string(),
        ClassicalOp::Not => "Not".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::builtin_ops::ModuleOp;
    use crate::region_kinds::RegionKind;

    #[test]
    fn test_module_to_phir_json() {
        let module = ModuleOp::new("test_module");
        let phir = module_to_phir_json(&module).unwrap();

        assert_eq!(phir.format, "PHIR/JSON");
        assert_eq!(phir.version, PHIR_JSON_VERSION);
        assert_eq!(phir.module.name, "test_module");
    }
}