/*!
RON (Rusty Object Notation) serialization for PHIR

This module provides direct serialization/deserialization between PHIR and RON format.
RON provides a format that more closely matches the underlying PHIR structure,
serving as a bridge between the stable, versioned PHIR-JSON format and the
internal PHIR representation. It's particularly useful for debugging and understanding
the IR structure.

Architecture:
    PHIR (in-memory) ←→ PHIR-RON ←→ PHIR-JSON
         ↑                ↑           ↑
    Internal IR    Debug/Bridge   Stable API
*/

use crate::phir::{Module, Region, Block, Instruction, AttributeValue, Terminator, BlockArg};
use crate::builtin_ops::{ModuleOp, FuncOp, ReturnOp, BuiltinOp};
use crate::ops::{Operation, QuantumOp, ClassicalOp, ControlFlowOp, MemoryOp, CustomOp, SSAValue};
use crate::parsing_ops::{ParsingOp, UnresolvedCall, UnresolvedRef, ForLoop, IfElse};
use crate::types::{Type, FunctionType, IntWidth, FloatWidth};
use crate::region_kinds::RegionKind;
use serde::{Serialize, Deserialize};
use std::collections::BTreeMap;

/// PHIR-RON format version
pub const PHIR_RON_VERSION: &str = "0.2.0";

/// Top-level PHIR-RON structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhirRon {
    pub format: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<BTreeMap<String, AttributeValue>>,
    pub module: ModuleOp,
}

/// Direct serialization of PHIR to RON
impl PhirRon {
    /// Create from a PHIR module
    pub fn from_module(module: ModuleOp) -> Self {
        Self {
            format: "PHIR/RON".to_string(),
            version: PHIR_RON_VERSION.to_string(),
            metadata: None,
            module,
        }
    }

    /// Extract the module
    pub fn into_module(self) -> ModuleOp {
        self.module
    }

    /// Serialize to RON string
    pub fn to_ron_string(&self) -> Result<String, ron::Error> {
        ron::to_string(self)
    }

    /// Pretty-print to RON string
    pub fn to_ron_pretty(&self) -> Result<String, ron::Error> {
        let pretty = ron::ser::PrettyConfig::default()
            .with_separate_tuple_members(true)
            .with_enumerate_arrays(true);
        ron::ser::to_string_pretty(self, pretty)
    }

    /// Deserialize from RON string
    pub fn from_ron_str(s: &str) -> Result<Self, ron::de::Error> {
        ron::from_str(s)
    }
}

/// Make all PHIR types directly serializable with RON
/// This is the key advantage - RON can handle Rust enums naturally

impl Serialize for RegionKind {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            RegionKind::SSACFG => serializer.serialize_str("SSACFG"),
            RegionKind::Graph => serializer.serialize_str("Graph"),
            RegionKind::Custom(s) => serializer.serialize_str(s),
        }
    }
}

impl<'de> Deserialize<'de> for RegionKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(match s.as_str() {
            "SSACFG" => RegionKind::SSACFG,
            "Graph" => RegionKind::Graph,
            other => RegionKind::Custom(other.to_string()),
        })
    }
}

/// Example RON representation of a quantum circuit
///
/// This is much more concise than JSON and naturally represents Rust types:
///
/// ```ron
/// PhirRon(
///     format: "PHIR/RON",
///     version: "0.2.0",
///     module: ModuleOp(
///         name: "bell_circuit",
///         attributes: {},
///         body: Region(
///             blocks: [
///                 Block(
///                     label: Some("entry"),
///                     operations: [
///                         Instruction(
///                             operation: Quantum(H),
///                             operands: [SSAValue(0)],
///                             results: [SSAValue(1)],
///                             result_types: [Qubit],
///                         ),
///                         Instruction(
///                             operation: Quantum(CNOT),
///                             operands: [SSAValue(1), SSAValue(2)],
///                             results: [SSAValue(3), SSAValue(4)],
///                             result_types: [Qubit, Qubit],
///                         ),
///                     ],
///                 ),
///             ],
///             kind: SSACFG,
///             attributes: {},
///         ),
///     ),
/// )
/// ```

/// Convert between RON and JSON representations
pub mod conversion {
    use super::*;
    use crate::serialization::{PhirJson, PhirModule, PhirRegion, PhirBlock, PhirOperation};

    /// Convert PHIR-RON to PHIR-JSON
    pub fn ron_to_json(ron: PhirRon) -> Result<PhirJson, serde_json::Error> {
        // This is where we translate from RON's natural Rust representation
        // to JSON's more structured format
        crate::serialization::module_to_phir_json(&ron.module)
    }

    /// Convert PHIR-JSON to PHIR-RON
    pub fn json_to_ron(json: PhirJson) -> Result<PhirRon, Box<dyn std::error::Error>> {
        // This would need to parse the JSON structure back into PHIR types
        // For now, this is a placeholder
        todo!("Implement JSON to RON conversion")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phir::*;
    use crate::ops::SSAValue;

    #[test]
    fn test_ron_serialization() {
        // Create a simple module
        let mut module = ModuleOp::new("test");

        // Create a function
        let func_type = FunctionType {
            inputs: vec![Type::Qubit, Type::Qubit],
            outputs: vec![Type::Bit, Type::Bit],
            variadic: false,
        };
        let func = FuncOp::new("bell_circuit", func_type);

        // Add to module
        let func_inst = Instruction::new(
            Operation::Builtin(BuiltinOp::Func(func)),
            vec![],
            vec![],
            vec![],
        );
        module.add_operation(func_inst);

        // Create PHIR-RON
        let phir_ron = PhirRon::from_module(module);

        // Serialize to RON
        let ron_string = phir_ron.to_ron_pretty().unwrap();
        println!("RON representation:\n{}", ron_string);

        // Verify it contains expected content
        assert!(ron_string.contains("PHIR/RON"));
        assert!(ron_string.contains("bell_circuit"));

        // Test round-trip
        let deserialized = PhirRon::from_ron_str(&ron_string).unwrap();
        assert_eq!(deserialized.module.name, "test");
    }

    #[test]
    fn test_ron_quantum_ops() {
        let mut block = Block::new(Some("quantum_ops"));

        // Add some quantum operations
        let h_op = Instruction::new(
            Operation::Quantum(QuantumOp::H),
            vec![SSAValue::new(0)],
            vec![SSAValue::new(1)],
            vec![Type::Qubit],
        );
        block.add_instruction(h_op);

        // RON can naturally represent the enum variants
        let ron_string = ron::to_string(&block).unwrap();
        assert!(ron_string.contains("Quantum(H)"));
    }

    #[test]
    fn test_ron_attributes() {
        let mut attrs = BTreeMap::new();
        attrs.insert("qec.code".to_string(), AttributeValue::String("steane".to_string()));
        attrs.insert("qec.distance".to_string(), AttributeValue::Int(7));
        attrs.insert("verified".to_string(), AttributeValue::Bool(true));

        let ron_string = ron::ser::to_string_pretty(&attrs, Default::default()).unwrap();
        println!("Attributes in RON:\n{}", ron_string);

        // RON handles nested structures elegantly
        assert!(ron_string.contains("String(\"steane\")"));
        assert!(ron_string.contains("Int(7)"));
    }
}