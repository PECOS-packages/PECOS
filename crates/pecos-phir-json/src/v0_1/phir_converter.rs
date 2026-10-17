/*!
Improved PHIR-JSON to PHIR Module converter with explicit bit operations

This module converts PHIR-JSON to PHIR Module structures and generates
explicit bit-combining operations for measurements that write to bit indices.

For example, when measurements write to [["m", 0]] and [["m", 1]], this
generates explicit shift and OR operations to combine the bits.
*/

use pecos_core::{Angle64, errors::PecosError};
use pecos_phir::{
    Module,
    builtin_ops::{BuiltinOp, VarDefineOp},
    ops::{ClassicalOp, Operation, QuantumOp},
    phir::{Block, Instruction, Region, SSAValue},
    region_kinds::RegionKind,
    types::{IntWidth, Type},
};
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};

use super::declarations::{DeclarationKind, Declarations};
use super::environment::Environment;

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
    // Emit definitions once at the program boundary, after all uses have been resolved.
    let instructions = converter
        .resolved_qubits
        .iter()
        .map(|&id| {
            Instruction::new(
                Operation::Classical(ClassicalOp::ConstInt(i64::from(id))),
                vec![],
                vec![SSAValue::new(id)],
                vec![Type::Int(IntWidth::I64)],
            )
        })
        .chain(instructions)
        .collect();

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
    environment: Environment,
    declarations: Declarations,
    next_ssa_id: u32,
    resolved_qubits: BTreeSet<u32>,
    variable_map: BTreeMap<String, u32>,
    variable_types: BTreeMap<String, Type>,
    bit_indexed_writes: BTreeMap<String, Vec<BitIndexedWrite>>,
}

impl ImprovedConverter {
    fn new() -> Self {
        Self {
            environment: Environment::new(),
            declarations: Declarations::default(),
            next_ssa_id: 0,
            resolved_qubits: BTreeSet::new(),
            variable_map: BTreeMap::new(),
            variable_types: BTreeMap::new(),
            bit_indexed_writes: BTreeMap::new(),
        }
    }

    fn get_ssa_id(&mut self, var: &str) -> Result<u32, PecosError> {
        if let Some(&id) = self.variable_map.get(var) {
            Ok(id)
        } else {
            let id = self.new_ssa_id()?;
            self.variable_map.insert(var.to_string(), id);
            Ok(id)
        }
    }

    fn new_ssa_id(&mut self) -> Result<u32, PecosError> {
        let id = self.next_ssa_id;
        self.next_ssa_id = id
            .checked_add(1)
            .ok_or_else(|| PecosError::Input("PHIR SSA ID space exhausted".to_string()))?;
        Ok(id)
    }

    fn convert_operations(&mut self, ops: &[Value]) -> Result<Vec<Instruction>, PecosError> {
        super::declarations::validate_json_operations(ops)?;
        // Register definitions own the global qubit numbering, as in PyPHIR.
        // Reserve those IDs before allocating SSA IDs for registers and values.
        for op in ops {
            if op.get("data").and_then(Value::as_str) == Some("qvar_define") {
                let name = op.get("variable").and_then(Value::as_str).ok_or_else(|| {
                    PecosError::Input("Quantum register requires a variable name".to_string())
                })?;
                let data_type = op
                    .get("data_type")
                    .map(|value| {
                        value.as_str().ok_or_else(|| {
                            PecosError::Input(format!(
                                "Register '{name}' has invalid data_type {value}"
                            ))
                        })
                    })
                    .transpose()?;
                let (name, size) = super::ast::validate_quantum_declaration(
                    name,
                    data_type,
                    op.get("size").and_then(Value::as_u64),
                )?;
                self.environment.add_quantum_register(name, size)?;
            }
        }
        self.next_ssa_id = u32::try_from(self.environment.count_qubits())
            .map_err(|_| PecosError::Input("Too many qubits for PHIR SSA IDs".to_string()))?;

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
                if let Operation::Quantum(ref op) = instruction.operation {
                    let arity = op
                        .operand_count()
                        .expect("converted quantum operations have fixed positive arity");
                    for (index, operands) in instruction.operands.chunks_exact(arity).enumerate() {
                        let mut split = instruction.clone();
                        split.operands = operands.to_vec();
                        let result_arity = if matches!(op, QuantumOp::Measure) {
                            1
                        } else {
                            arity
                        };
                        if !instruction.results.is_empty() {
                            let start = index * result_arity;
                            let end = start + result_arity;
                            split.results = instruction.results[start..end].to_vec();
                            split.result_types = instruction.result_types[start..end].to_vec();
                        }
                        instructions.push(split);
                    }
                } else {
                    instructions.push(instruction);
                }
            }
        }

        // Second pass: generate bit-combining operations for variables with bit-indexed writes
        let bit_indexed_writes = self.bit_indexed_writes.clone();
        for (var_name, writes) in &bit_indexed_writes {
            let combined_ssa =
                self.generate_bit_combining_operations(var_name, writes, &mut instructions)?;

            // Update the variable's SSA mapping to point to the combined value
            self.variable_map.insert(var_name.clone(), combined_ssa.id);
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
    ) -> Result<SSAValue, PecosError> {
        // Sort writes by bit index
        let mut sorted_writes = writes.to_vec();
        sorted_writes.sort_by_key(|w| w.bit_index);

        // Start with zero
        let zero_ssa = SSAValue {
            id: self.new_ssa_id()?,
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
                id: self.new_ssa_id()?,
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
                    id: self.new_ssa_id()?,
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
                    id: self.new_ssa_id()?,
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
                    id: self.new_ssa_id()?,
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

        Ok(current_value)
    }

    fn unsupported_operation(kind: &str, name: &str) -> PecosError {
        PecosError::Input(format!(
            "PHIR {kind} {name}: the JSON-to-module path does not support it yet; use the native PHIR-JSON engine for supported PHIR operations"
        ))
    }

    fn convert_operation(&mut self, op: &Value) -> Result<Option<Instruction>, PecosError> {
        let obj = op.as_object().ok_or_else(|| {
            PecosError::Input(format!(
                "Invalid PHIR operation {op}; supply a JSON object with a supported operation key"
            ))
        })?;

        // Variable definition
        if let Some(data) = obj.get("data").and_then(|v| v.as_str()) {
            return self.convert_variable_definition(obj, data);
        }

        // Quantum operation
        if let Some(qop) = obj.get("qop").and_then(|v| v.as_str()) {
            return self.convert_quantum_operation(obj, qop);
        }

        // Classical operation
        if let Some(cop) = obj.get("cop").and_then(|v| v.as_str()) {
            return self.convert_classical_operation(obj, cop);
        }

        for kind in ["block", "mop", "meta"] {
            if let Some(name) = obj.get(kind) {
                let name = match name {
                    Value::String(name) => name.clone(),
                    value => value.to_string(),
                };
                return Err(Self::unsupported_operation(kind, &name));
            }
        }
        // A PHIR comment carries no execution semantics.
        if obj.contains_key("//") {
            return Ok(None);
        }
        Err(PecosError::Input(format!(
            "Unrecognized PHIR operation {op}; supply a supported data, qop or cop operation, or use the native PHIR-JSON engine"
        )))
    }

    fn convert_variable_definition(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        data: &str,
    ) -> Result<Option<Instruction>, PecosError> {
        if data != "qvar_define" && data != "cvar_define" {
            return Err(Self::unsupported_operation("data", data));
        }
        let variable = obj.get("variable").and_then(Value::as_str).unwrap_or("");
        let data_type = obj
            .get("data_type")
            .map(|value| {
                value.as_str().ok_or_else(|| {
                    PecosError::Input(format!(
                        "Register '{variable}' has invalid data_type {value}"
                    ))
                })
            })
            .transpose()?;
        let raw_size = obj.get("size").and_then(Value::as_u64);
        let (data_type, size) = if data == "qvar_define" {
            let (_, size) =
                super::ast::validate_quantum_declaration(variable, data_type, raw_size)?;
            ("qubits", size)
        } else {
            let data_type = data_type.ok_or_else(|| {
                PecosError::Input(format!("Register '{variable}' requires data_type"))
            })?;
            (
                data_type,
                raw_size.and_then(|n| usize::try_from(n).ok()).unwrap_or(0),
            )
        };

        match data {
            "qvar_define" | "cvar_define" => {
                let var_define_op =
                    VarDefineOp::new(variable.to_string(), data_type.to_string(), size);

                let kind = if data == "qvar_define" {
                    DeclarationKind::Quantum
                } else {
                    DeclarationKind::Classical
                };
                self.declarations.register(variable, kind)?;
                let var_id = self.get_ssa_id(variable)?;

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

                Ok(Some(instruction))
            }
            _ => Err(Self::unsupported_operation("data", data)),
        }
    }

    fn convert_quantum_operation(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        qop: &str,
    ) -> Result<Option<Instruction>, PecosError> {
        use super::ast::{Operation as JsonOperation, QubitArg};
        use pecos_core::gate_type::GateType;
        let mut quantum_fields = obj.clone();
        quantum_fields.remove("returns");
        let ast: JsonOperation = serde_json::from_value(Value::Object(quantum_fields))
            .map_err(|error| PecosError::Input(format!("Invalid {qop} operation {obj:?}: {error}; supply valid PHIR arguments and angles")))?;
        let JsonOperation::QuantumOp { angles, args, .. } = ast else {
            unreachable!("convert_quantum_operation receives a qop object")
        };
        let arity = if qop == "CPhase" {
            2
        } else {
            super::operations::quantum_gate_type(qop)?.quantum_arity()
        };
        super::operations::validate_quantum_args(qop, arity, &args)?;
        if qop != "CPhase" {
            let qubit_count = args
                .iter()
                .map(|arg| match arg {
                    QubitArg::SingleQubit(_) => 1,
                    QubitArg::MultipleQubits(qubits) => qubits.len(),
                })
                .sum();
            super::operations::validate_quantum_signature(
                qop,
                super::operations::quantum_gate_type(qop)?,
                qubit_count,
                angles.as_ref().map_or(0, Vec::len),
            )?;
        }
        let angle = |count: usize, index: usize| -> Result<Angle64, PecosError> {
            let values = angles.as_deref().unwrap_or_default();
            if values.len() != count {
                return Err(PecosError::Input(format!(
                    "{qop}: angle count {} is invalid; supply {count} angles",
                    values.len()
                )));
            }
            Angle64::try_from_radians(values[index])
                .map_err(|error| PecosError::Input(error.to_string()))
        };
        let quantum_op = if qop == "CPhase" {
            QuantumOp::CPhase(angle(1, 0)?)
        } else {
            let gate = super::operations::quantum_gate_type(qop)?;
            match gate {
                GateType::I => QuantumOp::I,
                GateType::H => QuantumOp::H,
                GateType::X => QuantumOp::X,
                GateType::Y => QuantumOp::Y,
                GateType::Z => QuantumOp::Z,
                GateType::T => QuantumOp::T,
                GateType::Tdg => QuantumOp::Tdg,
                GateType::SX => QuantumOp::SX,
                GateType::SXdg => QuantumOp::SXdg,
                GateType::SY => QuantumOp::SY,
                GateType::SYdg => QuantumOp::SYdg,
                GateType::F => QuantumOp::F,
                GateType::Fdg => QuantumOp::Fdg,
                GateType::CX => QuantumOp::CX,
                GateType::CY => QuantumOp::CY,
                GateType::CZ => QuantumOp::CZ,
                GateType::SWAP => QuantumOp::SWAP,
                GateType::SXX => QuantumOp::SXX,
                GateType::SXXdg => QuantumOp::SXXdg,
                GateType::SYY => QuantumOp::SYY,
                GateType::SYYdg => QuantumOp::SYYdg,
                GateType::SZZ => QuantumOp::SZZ,
                GateType::SZZdg => QuantumOp::SZZdg,
                GateType::SZ => QuantumOp::S,
                GateType::SZdg => QuantumOp::Sdg,
                GateType::MZ => QuantumOp::Measure,
                GateType::PZ => QuantumOp::Reset,
                GateType::RX => QuantumOp::RX(angle(1, 0)?),
                GateType::RY => QuantumOp::RY(angle(1, 0)?),
                GateType::RZ => QuantumOp::RZ(angle(1, 0)?),
                GateType::RXX => QuantumOp::RXX(angle(1, 0)?),
                GateType::RYY => QuantumOp::RYY(angle(1, 0)?),
                GateType::RZZ => QuantumOp::RZZ(angle(1, 0)?),
                GateType::RXY1Q => QuantumOp::RXY1Q(angle(2, 0)?, angle(2, 1)?),
                GateType::RXYXY2Q => QuantumOp::RXYXY2Q(angle(2, 0)?, angle(2, 1)?),
                GateType::U => QuantumOp::U3(angle(3, 0)?, angle(3, 1)?, angle(3, 2)?),
                GateType::RXXRYYRZZ => {
                    QuantumOp::RXXRYYRZZ(angle(3, 0)?, angle(3, 1)?, angle(3, 2)?)
                }
                _ => {
                    return Err(PecosError::Input(format!(
                        "Quantum operation {qop} has no PHIR module representation; use a representable gate"
                    )));
                }
            }
        };
        let arity = quantum_op
            .operand_count()
            .expect("converted quantum operations have fixed positive arity");
        let mut operands = Vec::new();
        for arg in args {
            let qubits = match arg {
                QubitArg::SingleQubit(qubit) => vec![qubit],
                QubitArg::MultipleQubits(qubits) => qubits,
            };
            for (var, index) in qubits {
                operands.push(SSAValue::new(self.resolve_qubit(&var, index as u64)?));
            }
        }
        if operands.is_empty() || !operands.len().is_multiple_of(arity) {
            return Err(PecosError::Input(format!(
                "{qop}: operand count {} is invalid; supply a nonzero multiple of {arity} qubits",
                operands.len()
            )));
        }
        let qubit_ids: Vec<_> = operands.iter().map(|operand| operand.id as usize).collect();
        super::operations::validate_quantum_qubits(qop, &qubit_ids)?;
        let mut results = Vec::new();
        let mut result_types = Vec::new();
        if let Some(raw_returns) = obj.get("returns") {
            let returns = raw_returns.as_array().ok_or_else(|| {
                PecosError::Input(format!(
                    "{qop}: invalid returns {raw_returns}; supply an array of return destinations"
                ))
            })?;
            for ret in returns {
                if let Some(arr) = ret.as_array() {
                    if arr.len() == 2
                        && let (Some(var), Some(idx)) = (arr[0].as_str(), arr[1].as_u64())
                    {
                        // For measurements with bit-indexed returns, allocate a new SSA ID
                        if qop == "Measure" {
                            let result_ssa = SSAValue {
                                id: self.new_ssa_id()?,
                                version: 0,
                            };
                            results.push(result_ssa);
                            result_types.push(Type::Bit);

                            // Track this bit-indexed write
                            let write = BitIndexedWrite {
                                bit_index: u32::try_from(idx).map_err(|_| {
                                    PecosError::Input(format!(
                                        "Measure: bit index {idx} is too large; use a u32 index"
                                    ))
                                })?,
                                ssa_value: result_ssa,
                            };
                            self.bit_indexed_writes
                                .entry(var.to_string())
                                .or_default()
                                .push(write);
                        } else {
                            // Non-measurement operations
                            results.push(SSAValue {
                                id: self.resolve_qubit(var, idx)?,
                                version: 0,
                            });
                            result_types.push(Type::Qubit);
                        }
                    } else {
                        return Err(PecosError::Input(format!(
                            "{qop}: invalid return {ret}; supply [register, nonnegative index]"
                        )));
                    }
                } else if let Some(_var) = ret.as_str().filter(|_| qop != "Measure") {
                    // Simple variable return
                    let result_ssa = SSAValue {
                        id: self.new_ssa_id()?,
                        version: 0,
                    };
                    results.push(result_ssa);
                    result_types.push(Type::Qubit);
                } else {
                    return Err(PecosError::Input(format!(
                        "{qop}: invalid return {ret}; supply [register, nonnegative index]"
                    )));
                }
            }
        } else if qop != "Measure" {
            // Generate result for non-measurement operations
            for _ in &operands {
                results.push(SSAValue::new(self.new_ssa_id()?));
                result_types.push(Type::Qubit);
            }
        }

        if !results.is_empty() && results.len() != operands.len() {
            return Err(PecosError::Input(format!(
                "{qop}: result count {} does not match {} operands; supply one return per qubit or omit returns",
                results.len(),
                operands.len()
            )));
        }
        Ok(Some(Instruction::new(
            Operation::Quantum(quantum_op),
            operands,
            results,
            result_types,
        )))
    }

    fn resolve_qubit(&mut self, var: &str, index: u64) -> Result<u32, PecosError> {
        let index = usize::try_from(index).map_err(|_| {
            PecosError::Input(format!(
                "Qubit index {index} too large for register '{var}'"
            ))
        })?;
        let id = self.environment.resolve_qubit(var, index)?;
        let id = u32::try_from(id)
            .map_err(|_| PecosError::Input(format!("Qubit ID {id} too large for PHIR SSA")))?;
        self.resolved_qubits.insert(id);
        Ok(id)
    }

    fn convert_classical_operation(
        &mut self,
        obj: &serde_json::Map<String, Value>,
        cop: &str,
    ) -> Result<Option<Instruction>, PecosError> {
        match cop {
            "Result" => {
                let classical_op = ClassicalOp::Result;

                let names = |field: &str| -> Result<Vec<&str>, PecosError> {
                    let value = obj.get(field).ok_or_else(|| {
                        PecosError::Input(format!(
                            "Result: missing {field}; supply an array of variable names"
                        ))
                    })?;
                    let values = value.as_array().ok_or_else(|| {
                        PecosError::Input(format!(
                            "Result: invalid {field} {value}; supply an array of variable names"
                        ))
                    })?;
                    values
                        .iter()
                        .map(|value| {
                            value.as_str().ok_or_else(|| PecosError::Input(format!(
                        "Result: invalid {field} entry {value}; supply a variable name as a string"
                    )))
                        })
                        .collect()
                };
                let args = names("args")?;
                let returns = names("returns")?;

                // Resolve sources using mappings updated by bit-combining operations.
                let mut operands = Vec::new();
                for name in args {
                    operands.push(SSAValue::new(self.get_ssa_id(name)?));
                }
                let mut results = Vec::new();
                for name in returns {
                    results.push(SSAValue::new(self.get_ssa_id(name)?));
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

                Ok(Some(instruction))
            }
            _ => Err(Self::unsupported_operation("cop", cop)),
        }
    }
}

#[cfg(test)]
mod tests {

    #[test]
    fn declaration_ssa_aliasing_regression() {
        use serde_json::json;
        for (first, second) in [
            ("qvar_define", "qvar_define"),
            ("cvar_define", "cvar_define"),
            ("qvar_define", "cvar_define"),
            ("cvar_define", "qvar_define"),
        ] {
            let declaration = |data: &str, name: &str| {
                json!({
                    "data":data, "variable":name, "size":1,
                    "data_type":if data == "qvar_define" {"qubits"} else {"u32"},
                })
            };
            let mut converter = ImprovedConverter::new();
            let a = converter
                .convert_operation(&declaration(first, "a"))
                .unwrap()
                .unwrap();
            let b = converter
                .convert_operation(&declaration(second, "b"))
                .unwrap()
                .unwrap();
            assert_ne!(
                a.results[0], b.results[0],
                "distinct declared names must have distinct SSA identities"
            );
            let before = (
                converter.next_ssa_id,
                converter.variable_map.clone(),
                converter.variable_types.clone(),
            );
            let error = converter
                .convert_operation(&declaration(second, "a"))
                .unwrap_err();
            let first_kind = if first == "qvar_define" {
                "quantum"
            } else {
                "classical"
            };
            let second_kind = if second == "qvar_define" {
                "quantum"
            } else {
                "classical"
            };
            assert_eq!(
                error.to_string(),
                format!(
                    "Input error: Variable 'a' is already declared as {first_kind}; cannot redeclare as {second_kind}"
                )
            );
            assert_eq!(
                (
                    converter.next_ssa_id,
                    converter.variable_map.clone(),
                    converter.variable_types.clone()
                ),
                before,
                "rejection must preserve SSA allocation and type state"
            );
            // References still resolve to the first declaration's identity.
            assert_eq!(converter.get_ssa_id("a").unwrap(), a.results[0].id);
        }
    }

    #[test]
    fn redeclaration_precedes_ssa_lookup() {
        use serde_json::json;
        let mut converter = ImprovedConverter::new();
        converter
            .convert_operation(&json!({"data":"qvar_define", "variable":"a", "size":1}))
            .unwrap();
        // A reference mapping can change independently of the declaration registry.
        // Force the lookup to allocate, making its ordering observable even though
        // a normal duplicate lookup would return an existing id without side effects.
        converter.variable_map.remove("a");
        converter.next_ssa_id = u32::MAX;
        let error = converter
            .convert_operation(
                &json!({"data":"cvar_define", "data_type":"u32", "variable":"a", "size":1}),
            )
            .unwrap_err();
        assert_eq!(
            error.to_string(),
            "Input error: Variable 'a' is already declared as quantum; cannot redeclare as classical"
        );
    }

    #[test]
    fn program_redeclaration_precedes_qubit_and_ssa_reservation() {
        use serde_json::json;
        let mut converter = ImprovedConverter::new();
        let error = converter
            .convert_operations(&[
                json!({"data":"qvar_define", "variable":"a", "size":1}),
                json!({"data":"cvar_define", "data_type":"u32", "variable":"a", "size":1}),
            ])
            .unwrap_err();
        assert!(error.to_string().contains("cannot redeclare as classical"));
        assert_eq!(
            (
                converter.environment.count_qubits(),
                converter.next_ssa_id,
                converter.variable_map.len()
            ),
            (0, 0, 0)
        );
    }
    use super::*;

    #[test]
    fn all_named_two_qubit_roots_convert_to_typed_phir_with_tuple_args() {
        let json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "ops": [
                {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
                {"qop": "SXX", "args": [[["q", 0], ["q", 1]]]},
                {"qop": "SXXdg", "args": [[["q", 0], ["q", 1]]]},
                {"qop": "SYY", "args": [[["q", 0], ["q", 1]]]},
                {"qop": "SYYdg", "args": [[["q", 0], ["q", 1]]]},
                {"qop": "SZZ", "args": [[["q", 0], ["q", 1]]]},
                {"qop": "SZZdg", "args": [[["q", 0], ["q", 1]]]}
            ]
        }"#;
        let module = phir_json_to_module(json).expect("all roots should convert");
        let quantum: Vec<_> = module.body.blocks[0]
            .operations
            .iter()
            .filter_map(|instruction| match &instruction.operation {
                Operation::Quantum(op) => Some((op.clone(), instruction.operands.clone())),
                _ => None,
            })
            .collect();
        assert_eq!(
            quantum.iter().map(|(op, _)| op).collect::<Vec<_>>(),
            [
                &QuantumOp::SXX,
                &QuantumOp::SXXdg,
                &QuantumOp::SYY,
                &QuantumOp::SYYdg,
                &QuantumOp::SZZ,
                &QuantumOp::SZZdg,
            ]
        );
        assert!(quantum.iter().all(|(_, operands)| operands.len() == 2));
    }

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
        assert_eq!(module.body.blocks[0].operations, []);
    }

    #[test]
    fn test_unsupported_classical_op_rejected() {
        let json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "ops": [
                {"data": "cvar_define", "data_type": "i64", "variable": "x", "size": 1},
                {"cop": "=", "args": [1], "returns": [["x", 0]]}
            ]
        }"#;
        let error = phir_json_to_module(json).unwrap_err().to_string();
        assert!(error.contains("cop =") && error.contains("JSON-to-module"));
    }

    #[test]
    fn test_unknown_data_directive_rejected() {
        let json = r#"{
            "format": "PHIR/JSON",
            "version": "0.1.0",
            "ops": [
                {"data": "unknown_define", "data_type": "qubits", "variable": "x", "size": 1}
            ]
        }"#;
        let error = phir_json_to_module(json).unwrap_err().to_string();
        assert!(error.contains("unknown_define") && error.contains("JSON-to-module"));
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
