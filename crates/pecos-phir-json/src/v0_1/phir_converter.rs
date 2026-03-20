/*!
Improved PHIR-JSON to PHIR Module converter with explicit bit operations

This module converts PHIR-JSON to PHIR Module structures and generates
explicit bit-combining operations for measurements that write to bit indices.

For example, when measurements write to [["m", 0]] and [["m", 1]], this
generates explicit shift and OR operations to combine the bits.
*/

use pecos_core::errors::PecosError;
use pecos_phir::{
    Module,
    builtin_ops::{BuiltinOp, VarDefineOp},
    ops::{ClassicalOp, Operation, QuantumOp},
    phir::{Block, Instruction, Region, SSAValue},
    region_kinds::RegionKind,
    types::{IntWidth, Type},
};
use serde_json::Value;
use std::collections::BTreeMap;

/// Information about a bit-indexed write
#[derive(Debug, Clone)]
struct BitIndexedWrite {
    bit_index: u32,
    ssa_value: SSAValue,
}

/// Convert PHIR-JSON string to PHIR Module with explicit bit operations
///
/// # Errors
///
/// Returns an error if JSON parsing fails or the structure is invalid
pub fn phir_json_to_module(json_str: &str) -> Result<Module, PecosError> {
    // Parse JSON
    let json_value: Value = serde_json::from_str(json_str)
        .map_err(|e| PecosError::Input(format!("Failed to parse PHIR-JSON: {e}")))?;

    let obj = json_value
        .as_object()
        .ok_or_else(|| PecosError::Input("PHIR-JSON must be an object".to_string()))?;

    // Validate format and version
    let format = obj
        .get("format")
        .and_then(|v| v.as_str())
        .ok_or_else(|| PecosError::Input("Missing 'format' field".to_string()))?;

    if format != "PHIR/JSON" {
        return Err(PecosError::Input(format!(
            "Invalid format: expected 'PHIR/JSON', got '{format}'"
        )));
    }

    let version = obj
        .get("version")
        .and_then(|v| v.as_str())
        .ok_or_else(|| PecosError::Input("Missing 'version' field".to_string()))?;

    if version != "0.1.0" {
        return Err(PecosError::Input(format!(
            "Unsupported version: expected '0.1.0', got '{version}'"
        )));
    }

    // Extract module name from metadata
    let module_name = obj
        .get("metadata")
        .and_then(|m| m.as_object())
        .and_then(|m| m.get("name"))
        .and_then(|n| n.as_str())
        .unwrap_or("phir_module");

    // Convert operations
    let ops = obj
        .get("ops")
        .and_then(|v| v.as_array())
        .ok_or_else(|| PecosError::Input("Missing 'ops' array".to_string()))?;

    let mut converter = ImprovedConverter::new();
    let instructions = converter.convert_operations(ops)?;

    // Create main block
    let main_block = Block {
        label: None,
        arguments: vec![],
        operations: instructions,
        terminator: None,
        attributes: BTreeMap::new(),
    };

    // Create main region
    let main_region = Region {
        blocks: vec![main_block],
        kind: RegionKind::SSACFG,
        attributes: BTreeMap::new(),
    };

    // Create module
    let module = Module {
        name: module_name.to_string(),
        attributes: BTreeMap::new(),
        body: main_region,
    };

    Ok(module)
}

struct ImprovedConverter {
    next_ssa_id: u32,
    variable_map: BTreeMap<String, u32>,
    variable_types: BTreeMap<String, Type>,
    bit_indexed_writes: BTreeMap<String, Vec<BitIndexedWrite>>,
}

impl ImprovedConverter {
    fn new() -> Self {
        Self {
            next_ssa_id: 0,
            variable_map: BTreeMap::new(),
            variable_types: BTreeMap::new(),
            bit_indexed_writes: BTreeMap::new(),
        }
    }

    fn get_ssa_id(&mut self, var: &str) -> u32 {
        if let Some(&id) = self.variable_map.get(var) {
            id
        } else {
            let id = self.next_ssa_id;
            self.next_ssa_id += 1;
            self.variable_map.insert(var.to_string(), id);
            id
        }
    }

    fn new_ssa_id(&mut self) -> u32 {
        let id = self.next_ssa_id;
        self.next_ssa_id += 1;
        id
    }

    fn convert_operations(&mut self, ops: &[Value]) -> Result<Vec<Instruction>, PecosError> {
        let mut instructions = Vec::new();
        let mut result_operations = Vec::new();

        // First pass: convert all operations except Result operations
        for op in ops {
            if let Some(cop) = op
                .as_object()
                .and_then(|o| o.get("cop"))
                .and_then(|v| v.as_str())
                && cop == "Result"
            {
                // Save Result operations for later
                result_operations.push(op.clone());
                continue;
            }

            if let Some(instruction) = self.convert_operation(op)? {
                instructions.push(instruction);
            }
        }

        // Second pass: generate bit-combining operations for variables with bit-indexed writes
        let bit_indexed_writes = self.bit_indexed_writes.clone();
        for (var_name, writes) in &bit_indexed_writes {
            if writes.len() > 1 {
                // Multiple bit writes to the same variable - generate combining operations
                let mut combining_instructions = Vec::new();
                let combined_ssa = self.generate_bit_combining_operations(
                    var_name,
                    writes,
                    &mut combining_instructions,
                );

                // Add the combining instructions
                instructions.extend(combining_instructions);

                // Update the variable's SSA mapping to point to the combined value
                self.variable_map.insert(var_name.clone(), combined_ssa.id);
            } else if writes.len() == 1 {
                // Single bit write - cast the measurement Bool to int and update mapping
                let bit_as_int = SSAValue {
                    id: self.new_ssa_id(),
                    version: 0,
                };
                let cast_instruction = Instruction {
                    operation: Operation::Classical(ClassicalOp::Bitcast),
                    operands: vec![writes[0].ssa_value],
                    results: vec![bit_as_int],
                    result_types: vec![Type::UInt(IntWidth::I32)],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                };
                instructions.push(cast_instruction);
                self.variable_map.insert(var_name.clone(), bit_as_int.id);
            }
        }

        // Third pass: now process Result operations with updated variable mappings
        for result_op in &result_operations {
            if let Some(instruction) = self.convert_operation(result_op)? {
                instructions.push(instruction);
            }
        }

        Ok(instructions)
    }

    fn generate_bit_combining_operations(
        &mut self,
        _var_name: &str,
        writes: &[BitIndexedWrite],
        instructions: &mut Vec<Instruction>,
    ) -> SSAValue {
        // Sort writes by bit index
        let mut sorted_writes = writes.to_vec();
        sorted_writes.sort_by_key(|w| w.bit_index);

        // Start with zero
        let zero_ssa = SSAValue {
            id: self.new_ssa_id(),
            version: 0,
        };
        let zero_instruction = Instruction {
            operation: Operation::Classical(ClassicalOp::ConstInt(0)),
            operands: vec![],
            results: vec![zero_ssa],
            result_types: vec![Type::UInt(IntWidth::I32)],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: None,
        };
        instructions.push(zero_instruction);

        let mut current_value = zero_ssa;

        // For each bit write, shift and OR
        for write in &sorted_writes {
            // Convert bool to int if needed
            let bit_as_int = SSAValue {
                id: self.new_ssa_id(),
                version: 0,
            };
            let cast_instruction = Instruction {
                operation: Operation::Classical(ClassicalOp::Bitcast),
                operands: vec![write.ssa_value],
                results: vec![bit_as_int],
                result_types: vec![Type::UInt(IntWidth::I32)],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            };
            instructions.push(cast_instruction);

            if write.bit_index > 0 {
                // Shift the bit to its position
                let shifted_ssa = SSAValue {
                    id: self.new_ssa_id(),
                    version: 0,
                };
                let shift_instruction = Instruction {
                    operation: Operation::Classical(ClassicalOp::Shl(write.bit_index)),
                    operands: vec![bit_as_int],
                    results: vec![shifted_ssa],
                    result_types: vec![Type::UInt(IntWidth::I32)],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                };
                instructions.push(shift_instruction);

                // OR with current value
                let or_ssa = SSAValue {
                    id: self.new_ssa_id(),
                    version: 0,
                };
                let or_instruction = Instruction {
                    operation: Operation::Classical(ClassicalOp::Or),
                    operands: vec![current_value, shifted_ssa],
                    results: vec![or_ssa],
                    result_types: vec![Type::UInt(IntWidth::I32)],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                };
                instructions.push(or_instruction);
                current_value = or_ssa;
            } else {
                // Bit 0 - just OR with current value
                let or_ssa = SSAValue {
                    id: self.new_ssa_id(),
                    version: 0,
                };
                let or_instruction = Instruction {
                    operation: Operation::Classical(ClassicalOp::Or),
                    operands: vec![current_value, bit_as_int],
                    results: vec![or_ssa],
                    result_types: vec![Type::UInt(IntWidth::I32)],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                };
                instructions.push(or_instruction);
                current_value = or_ssa;
            }
        }

        current_value
    }

    fn convert_operation(&mut self, op: &Value) -> Result<Option<Instruction>, PecosError> {
        let obj = op
            .as_object()
            .ok_or_else(|| PecosError::Input("Operation must be an object".to_string()))?;

        // Variable definition
        if let Some(data) = obj.get("data").and_then(|v| v.as_str()) {
            return Ok(self.convert_variable_definition(obj, data));
        }

        // Quantum operation
        if let Some(qop) = obj.get("qop").and_then(|v| v.as_str()) {
            return self.convert_quantum_operation(obj, qop);
        }

        // Classical operation
        if let Some(cop) = obj.get("cop").and_then(|v| v.as_str()) {
            return Ok(self.convert_classical_operation(obj, cop));
        }

        // Skip unknown operations
        Ok(None)
    }

    fn convert_variable_definition(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        data: &str,
    ) -> Option<Instruction> {
        let data_type = obj.get("data_type").and_then(|v| v.as_str()).unwrap_or("");
        let variable = obj.get("variable").and_then(|v| v.as_str()).unwrap_or("");
        let size = obj
            .get("size")
            .and_then(serde_json::Value::as_u64)
            .and_then(|v| usize::try_from(v).ok())
            .unwrap_or(0);

        match data {
            "qvar_define" | "cvar_define" => {
                let var_define_op =
                    VarDefineOp::new(variable.to_string(), data_type.to_string(), size);

                let var_id = self.get_ssa_id(variable);

                let result_type = match data {
                    "qvar_define" => Type::QuantumReg(size),
                    "cvar_define" => match data_type {
                        "i8" => Type::Int(IntWidth::I8),
                        "i16" => Type::Int(IntWidth::I16),
                        "i32" => Type::Int(IntWidth::I32),
                        "u8" => Type::UInt(IntWidth::I8),
                        "u16" => Type::UInt(IntWidth::I16),
                        "u32" => Type::UInt(IntWidth::I32),
                        "u64" => Type::UInt(IntWidth::I64),
                        "bool" => Type::Bool,
                        _ => Type::Int(IntWidth::I64), // Default fallback (includes "i64")
                    },
                    _ => Type::Unknown,
                };

                // Store the type for later use
                self.variable_types
                    .insert(variable.to_string(), result_type.clone());

                let instruction = Instruction {
                    operation: Operation::Builtin(BuiltinOp::VarDefine(var_define_op)),
                    operands: vec![],
                    results: vec![SSAValue {
                        id: var_id,
                        version: 0,
                    }],
                    result_types: vec![result_type],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                };

                Some(instruction)
            }
            _ => None, // Skip unknown variable definitions
        }
    }

    fn convert_quantum_operation(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        qop: &str,
    ) -> Result<Option<Instruction>, PecosError> {
        let quantum_op = match qop {
            "H" => QuantumOp::H,
            "X" => QuantumOp::X,
            "Y" => QuantumOp::Y,
            "Z" => QuantumOp::Z,
            "S" => QuantumOp::S,
            "T" => QuantumOp::T,
            "CX" | "CNOT" => QuantumOp::CX,
            "CZ" => QuantumOp::CZ,
            "Measure" => QuantumOp::Measure,
            _ => {
                return Err(PecosError::Input(format!(
                    "Unknown quantum operation: {qop}"
                )));
            }
        };

        // Convert operands
        let mut operands = Vec::new();
        if let Some(args) = obj.get("args").and_then(|v| v.as_array()) {
            for arg in args {
                if let Some(arr) = arg.as_array()
                    && arr.len() == 2
                    && let (Some(_var), Some(idx)) = (arr[0].as_str(), arr[1].as_u64())
                {
                    // For quantum operations, the operand is the qubit index directly
                    operands.push(SSAValue {
                        id: u32::try_from(idx).unwrap_or(0),
                        version: 0,
                    });
                }
            }
        }

        // Convert results
        let mut results = Vec::new();
        let mut result_types = Vec::new();

        if let Some(returns) = obj.get("returns").and_then(|v| v.as_array()) {
            for ret in returns {
                if let Some(arr) = ret.as_array() {
                    if arr.len() == 2
                        && let (Some(var), Some(idx)) = (arr[0].as_str(), arr[1].as_u64())
                    {
                        // For measurements with bit-indexed returns, allocate a new SSA ID
                        if qop == "Measure" {
                            let result_ssa = SSAValue {
                                id: self.new_ssa_id(),
                                version: 0,
                            };
                            results.push(result_ssa);
                            result_types.push(Type::Bit);

                            // Track this bit-indexed write
                            let write = BitIndexedWrite {
                                bit_index: u32::try_from(idx).unwrap_or(0),
                                ssa_value: result_ssa,
                            };
                            self.bit_indexed_writes
                                .entry(var.to_string())
                                .or_default()
                                .push(write);
                        } else {
                            // Non-measurement operations
                            let ssa_id = self.get_ssa_id(var);
                            results.push(SSAValue {
                                id: ssa_id + u32::try_from(idx).unwrap_or(0),
                                version: 0,
                            });
                            result_types.push(Type::Qubit);
                        }
                    }
                } else if let Some(_var) = ret.as_str() {
                    // Simple variable return
                    let result_ssa = SSAValue {
                        id: self.new_ssa_id(),
                        version: 0,
                    };
                    results.push(result_ssa);
                    result_types.push(if qop == "Measure" {
                        Type::Bit
                    } else {
                        Type::Qubit
                    });
                }
            }
        } else if qop != "Measure" {
            // Generate result for non-measurement operations
            let result_id = self.new_ssa_id();
            results.push(SSAValue {
                id: result_id,
                version: 0,
            });
            result_types.push(Type::Qubit);
        }

        let instruction = Instruction {
            operation: Operation::Quantum(quantum_op),
            operands,
            results,
            result_types,
            regions: vec![],
            attributes: BTreeMap::new(),
            location: None,
        };

        Ok(Some(instruction))
    }

    fn convert_classical_operation(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        cop: &str,
    ) -> Option<Instruction> {
        match cop {
            "Result" => {
                let classical_op = ClassicalOp::Result;

                // Convert operands (source variables)
                let mut operands = Vec::new();
                if let Some(args) = obj.get("args").and_then(|v| v.as_array()) {
                    for arg in args {
                        if let Some(var_name) = arg.as_str() {
                            // Use the current SSA ID for this variable
                            // It may have been updated by bit-combining operations
                            let ssa_id = self.get_ssa_id(var_name);
                            operands.push(SSAValue {
                                id: ssa_id,
                                version: 0,
                            });
                        }
                    }
                }

                // Convert results (destination variables)
                let mut results = Vec::new();
                if let Some(returns) = obj.get("returns").and_then(|v| v.as_array()) {
                    for ret in returns {
                        if let Some(var_name) = ret.as_str() {
                            let ssa_id = self.get_ssa_id(var_name);
                            results.push(SSAValue {
                                id: ssa_id,
                                version: 0,
                            });
                        }
                    }
                }

                // Create attributes to store the export names
                let mut attributes = BTreeMap::new();
                if let Some(returns) = obj.get("returns").and_then(|v| v.as_array())
                    && let Some(export_name) = returns.first().and_then(|v| v.as_str())
                {
                    attributes.insert(
                        "export_name".to_string(),
                        pecos_phir::phir::AttributeValue::String(export_name.to_string()),
                    );
                }

                let instruction = Instruction {
                    operation: Operation::Classical(classical_op),
                    operands,
                    results,
                    result_types: vec![Type::UInt(IntWidth::I32)], // Result operations typically return integers
                    regions: vec![],
                    attributes,
                    location: None,
                };

                Some(instruction)
            }
            _ => None, // Skip unknown classical operations
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_missing_format_field() {
        let json = r#"{"version": "0.1.0", "ops": []}"#;
        let result = phir_json_to_module(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("format"));
    }

    #[test]
    fn test_invalid_format_value() {
        let json = r#"{"format": "WRONG", "version": "0.1.0", "ops": []}"#;
        let result = phir_json_to_module(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("PHIR/JSON"));
    }

    #[test]
    fn test_missing_version_field() {
        let json = r#"{"format": "PHIR/JSON", "ops": []}"#;
        let result = phir_json_to_module(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("version"));
    }

    #[test]
    fn test_unsupported_version() {
        let json = r#"{"format": "PHIR/JSON", "version": "99.0.0", "ops": []}"#;
        let result = phir_json_to_module(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("99.0.0"));
    }

    #[test]
    fn test_missing_ops_array() {
        let json = r#"{"format": "PHIR/JSON", "version": "0.1.0"}"#;
        let result = phir_json_to_module(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("ops"));
    }

    #[test]
    fn test_root_not_object() {
        let result = phir_json_to_module("[1, 2, 3]");
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("object"));
    }

    #[test]
    fn test_invalid_json() {
        let result = phir_json_to_module("not json at all");
        assert!(result.is_err());
    }

    #[test]
    fn test_unknown_quantum_op() {
        let json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "ops": [
                {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
                {"qop": "UnknownGate", "args": [["q", 0]]}
            ]
        }"#;
        let result = phir_json_to_module(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("UnknownGate"));
    }

    #[test]
    fn test_operation_not_object() {
        let json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "ops": ["not_an_object"]
        }"#;
        let result = phir_json_to_module(json);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("object"));
    }

    #[test]
    fn test_empty_ops() {
        let json = r#"{"format": "PHIR/JSON", "version": "0.1.0", "ops": []}"#;
        let module = phir_json_to_module(json).unwrap();
        assert!(module.body.blocks[0].operations.is_empty());
    }

    #[test]
    fn test_unknown_classical_op_skipped() {
        let json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "ops": [
                {"data": "cvar_define", "data_type": "i64", "variable": "x", "size": 1},
                {"cop": "=", "args": [1], "returns": [["x", 0]]}
            ]
        }"#;
        // Unknown classical ops should be silently skipped, not error
        let module = phir_json_to_module(json).unwrap();
        // Should have the variable definition but the cop="=" should be skipped
        assert_eq!(module.body.blocks[0].operations.len(), 1);
    }

    #[test]
    fn test_unknown_data_type_skipped() {
        let json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "ops": [
                {"data": "unknown_define", "data_type": "qubits", "variable": "x", "size": 1}
            ]
        }"#;
        // Unknown data definitions should be skipped
        let module = phir_json_to_module(json).unwrap();
        assert!(module.body.blocks[0].operations.is_empty());
    }

    #[test]
    fn test_bell_state_conversion() {
        let bell_json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "metadata": {"description": "Bell state"},
            "ops": [
                {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
                {"data": "cvar_define", "data_type": "i64", "variable": "m", "size": 2},
                {"qop": "H", "args": [["q", 0]]},
                {"qop": "CX", "args": [["q", 0], ["q", 1]]},
                {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
                {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
                {"cop": "Result", "args": ["m"], "returns": ["c"]}
            ]
        }"#;

        let module = phir_json_to_module(bell_json).unwrap();

        // Should have more than 7 operations due to bit combining
        assert!(module.body.blocks[0].operations.len() > 7);

        // Check that we have Cast, Shl, Or operations
        let ops = &module.body.blocks[0].operations;
        let has_bitcast = ops
            .iter()
            .any(|i| matches!(i.operation, Operation::Classical(ClassicalOp::Bitcast)));
        let has_shift = ops
            .iter()
            .any(|i| matches!(i.operation, Operation::Classical(ClassicalOp::Shl(_))));
        let has_or = ops
            .iter()
            .any(|i| matches!(i.operation, Operation::Classical(ClassicalOp::Or)));

        assert!(has_bitcast, "Should have Bitcast operations");
        assert!(has_shift, "Should have Shift operations");
        assert!(has_or, "Should have Or operations");
    }
}
