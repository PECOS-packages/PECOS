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
use super::environment::{DataType, Environment, VariableInfo};

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

    super::classical_validation::validate_classical_inputs(&json_value)?;

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
                    super::ast::register_size(name, op.get("size"))?,
                )?;
                self.environment.add_quantum_register(name, size)?;
            }
        }
        self.next_ssa_id = u32::try_from(self.environment.count_qubits())
            .map_err(|_| PecosError::Input("Too many qubits for PHIR SSA IDs".to_string()))?;

        let mut instructions = Vec::new();
        let mut export_names = Vec::new();

        // Convert operations in source order, flushing measurements before Result copies.
        for op in ops {
            if let Some(cop) = op
                .as_object()
                .and_then(|o| o.get("cop"))
                .and_then(|v| v.as_str())
                && cop == "Result"
            {
                self.combine_pending_measurements(&mut instructions)?;
                export_names.extend(self.convert_result_operation(op, &mut instructions)?);
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

        self.combine_pending_measurements(&mut instructions)?;
        if export_names.is_empty() {
            export_names.extend(
                self.environment
                    .get_all_variables()
                    .iter()
                    .filter(|info| info.data_type != DataType::Qubits)
                    .map(|info| info.name.clone()),
            );
        }
        for name in export_names {
            let source = self.classical_value(&name)?;
            let width = if Self::register_width(self.environment.classical_info(&name)?) <= 32 {
                IntWidth::I32
            } else {
                IntWidth::I64
            };
            let source = self.append_classical(
                ClassicalOp::Bitcast,
                vec![source],
                &Type::UInt(width),
                &mut instructions,
            )?;
            let mut instruction = Instruction::new(
                Operation::Classical(ClassicalOp::Result),
                vec![source],
                vec![],
                vec![],
            );
            instruction.attributes.insert(
                "export_name".to_string(),
                pecos_phir::phir::AttributeValue::String(name),
            );
            instructions.push(instruction);
        }

        Ok(instructions)
    }

    fn register_width(info: &VariableInfo) -> usize {
        info.size + usize::from(info.data_type.is_signed())
    }

    fn storage_type(info: &VariableInfo) -> Type {
        match info.data_type {
            DataType::I8 => Type::Int(IntWidth::I8),
            DataType::I16 => Type::Int(IntWidth::I16),
            DataType::I32 => Type::Int(IntWidth::I32),
            DataType::I64 => Type::Int(IntWidth::I64),
            DataType::U8 => Type::UInt(IntWidth::I8),
            DataType::U16 => Type::UInt(IntWidth::I16),
            DataType::U32 => Type::UInt(IntWidth::I32),
            DataType::U64 => Type::UInt(IntWidth::I64),
            DataType::Bool => Type::Bool,
            DataType::Qubits => Type::QuantumReg(info.size),
        }
    }

    fn classical_value(&self, name: &str) -> Result<SSAValue, PecosError> {
        self.environment.classical_info(name)?;
        self.variable_map.get(name).copied().map(SSAValue::new).ok_or_else(|| PecosError::Input(format!(
            "Result or measurement: unknown classical variable {}; declare it with cvar_define before use", serde_json::json!(name)
        )))
    }

    fn append_classical(
        &mut self,
        op: ClassicalOp,
        operands: Vec<SSAValue>,
        value_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> Result<SSAValue, PecosError> {
        let result = SSAValue::new(self.new_ssa_id()?);
        instructions.push(Instruction::new(
            Operation::Classical(op),
            operands,
            vec![result],
            vec![value_type.clone()],
        ));
        Ok(result)
    }

    fn combine_pending_measurements(
        &mut self,
        instructions: &mut Vec<Instruction>,
    ) -> Result<(), PecosError> {
        for (name, writes) in std::mem::take(&mut self.bit_indexed_writes) {
            for write in &writes {
                self.environment
                    .validate_measurement_write(&name, write.bit_index as usize)?;
            }
            let mut current = self.classical_value(&name)?;
            let value_type = Self::storage_type(self.environment.classical_info(&name)?);
            for write in writes {
                current = self.replace_bit(current, &write, &value_type, instructions)?;
            }
            self.variable_map.insert(name, current.id);
        }
        Ok(())
    }

    fn replace_bit(
        &mut self,
        current: SSAValue,
        write: &BitIndexedWrite,
        value_type: &Type,
        instructions: &mut Vec<Instruction>,
    ) -> Result<SSAValue, PecosError> {
        let bit = self.append_classical(
            ClassicalOp::Bitcast,
            vec![write.ssa_value],
            value_type,
            instructions,
        )?;
        let shifted = self.append_classical(
            ClassicalOp::Shl(write.bit_index),
            vec![bit],
            value_type,
            instructions,
        )?;
        let mask = i64::from_ne_bytes((!(1_u64 << write.bit_index)).to_ne_bytes());
        let mask = self.append_classical(
            ClassicalOp::ConstInt(mask),
            vec![],
            value_type,
            instructions,
        )?;
        let cleared = self.append_classical(
            ClassicalOp::And,
            vec![current, mask],
            value_type,
            instructions,
        )?;
        self.append_classical(
            ClassicalOp::Or,
            vec![cleared, shifted],
            value_type,
            instructions,
        )
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
            return Err(Self::unsupported_operation("cop", cop));
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
        let raw_size = super::ast::register_size(variable, obj.get("size"))?;
        let (data_type, size) = if data == "qvar_define" {
            let (_, size) =
                super::ast::validate_quantum_declaration(variable, data_type, raw_size)?;
            ("qubits", size)
        } else {
            let data_type = data_type.ok_or_else(|| {
                PecosError::Input(format!("Register '{variable}' requires data_type"))
            })?;
            (data_type, super::ast::infer_size(data_type, raw_size))
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

                if data == "cvar_define" {
                    let native_type: super::environment::DataType = data_type.parse()?;
                    self.environment
                        .add_classical_variable(variable, native_type.clone(), size)?;
                }
                let result_type = if data == "qvar_define" {
                    Type::QuantumReg(size)
                } else {
                    Self::storage_type(self.environment.classical_info(variable)?)
                };

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
        let gate = super::operations::quantum_gate_type(qop)?;
        super::operations::validate_quantum_args(qop, gate.quantum_arity(), &args)?;
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
        super::operations::validate_quantum_signature(
            qop,
            gate,
            operands.len(),
            angles.as_ref().map_or(0, Vec::len),
        )?;
        let qubit_ids: Vec<_> = operands.iter().map(|operand| operand.id as usize).collect();
        super::operations::validate_quantum_qubits(qop, &qubit_ids)?;
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
        let quantum_op = match gate {
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
            GateType::RXXRYYRZZ => QuantumOp::RXXRYYRZZ(angle(3, 0)?, angle(3, 1)?, angle(3, 2)?),
            _ => {
                return Err(PecosError::Input(format!(
                    "Quantum operation {qop} has no PHIR module representation; use a representable gate"
                )));
            }
        };
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

    // SSA register values contain declared-width bits. Native signed values also
    // extend the sign bit to a complete i64 before a whole-register copy.
    fn result_source_bits(
        &mut self,
        value: SSAValue,
        info: &VariableInfo,
        instructions: &mut Vec<Instruction>,
    ) -> Result<SSAValue, PecosError> {
        let width = Self::register_width(info);
        if !info.data_type.is_signed() || width == 64 {
            return Ok(value);
        }
        let shift = u32::try_from(64 - width).map_err(|_| {
            PecosError::Input(format!(
                "Result: register {} has invalid width {width}; use at most 64 bits",
                serde_json::json!(info.name)
            ))
        })?;
        let storage = Type::Int(IntWidth::I64);
        let value =
            self.append_classical(ClassicalOp::Bitcast, vec![value], &storage, instructions)?;
        let value =
            self.append_classical(ClassicalOp::Shl(shift), vec![value], &storage, instructions)?;
        self.append_classical(ClassicalOp::Shr(shift), vec![value], &storage, instructions)
    }

    fn convert_result_operation(
        &mut self,
        op: &Value,
        instructions: &mut Vec<Instruction>,
    ) -> Result<Vec<String>, PecosError> {
        let args = super::classical_validation::result_field(op, "args")?;
        let returns = super::classical_validation::result_field(op, "returns")?;
        let pairs = self.environment.validate_result(&args, &returns)?;
        let mut export_names = Vec::new();
        for (source, destination) in pairs {
            let mut value = self.classical_value(&source.name)?;
            let source_info = self.environment.classical_info(&source.name)?.clone();
            let source_type = Self::storage_type(&source_info);
            let source_width = if source_info.data_type.is_signed() {
                64
            } else {
                Self::register_width(&source_info)
            };
            if source.index.is_none() && destination.index.is_none() {
                value = self.result_source_bits(value, &source_info, instructions)?;
            }
            if let Some(index) = source.index {
                value = self.append_classical(
                    ClassicalOp::LShr(index),
                    vec![value],
                    &source_type,
                    instructions,
                )?;
            }
            if source.index.is_some() || destination.index.is_some() {
                let one = self.append_classical(
                    ClassicalOp::ConstInt(1),
                    vec![],
                    &source_type,
                    instructions,
                )?;
                value = self.append_classical(
                    ClassicalOp::And,
                    vec![value, one],
                    &source_type,
                    instructions,
                )?;
            }
            self.environment
                .prepare_result_destination(&source, &destination)?;
            let destination_info = self.environment.classical_info(&destination.name)?;
            let value_type = Self::storage_type(destination_info);
            let width = Self::register_width(destination_info);
            if let Some(index) = destination.index {
                let current = if self.variable_map.contains_key(&destination.name) {
                    self.classical_value(&destination.name)?
                } else {
                    self.append_classical(
                        ClassicalOp::ConstInt(0),
                        vec![],
                        &value_type,
                        instructions,
                    )?
                };
                value = self.replace_bit(
                    current,
                    &BitIndexedWrite {
                        bit_index: index,
                        ssa_value: value,
                    },
                    &value_type,
                    instructions,
                )?;
            } else if width < source_width {
                let mask = i64::from_ne_bytes(((1_u64 << width) - 1).to_ne_bytes());
                let mask = self.append_classical(
                    ClassicalOp::ConstInt(mask),
                    vec![],
                    &value_type,
                    instructions,
                )?;
                value = self.append_classical(
                    ClassicalOp::And,
                    vec![value, mask],
                    &value_type,
                    instructions,
                )?;
            }
            self.variable_map.insert(destination.name.clone(), value.id);
            export_names.push(destination.name);
        }
        Ok(export_names)
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
                converter
                    .environment
                    .get_all_variables()
                    .iter()
                    .map(|info| (info.name.clone(), info.data_type.clone(), info.size))
                    .collect::<Vec<_>>(),
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
                    converter
                        .environment
                        .get_all_variables()
                        .iter()
                        .map(|info| (info.name.clone(), info.data_type.clone(), info.size))
                        .collect::<Vec<_>>()
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
