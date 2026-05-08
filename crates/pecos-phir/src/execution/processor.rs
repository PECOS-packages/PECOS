/*!
PHIR Operation Processor

Processes PHIR operations and converts them to quantum instructions.
This is the core component that interprets PHIR operations and generates
the appropriate quantum gates and classical computations.
*/

use super::environment::{DataType, Environment, TypedValue};
use crate::builtin_ops::BuiltinOp;
use crate::error::{PhirError, Result};
use crate::ops::{ClassicalOp, MemoryOp, Operation, QuantumOp};
use crate::phir::{Block, Module};
use pecos_core::Gate;
use pecos_engines::byte_message::builder::ByteMessageBuilder;
use std::collections::BTreeMap;

/// PHIR operation processor - converts PHIR operations to quantum instructions
#[derive(Debug, Clone)]
pub struct PhirProcessor {
    /// Execution environment for classical variables
    pub environment: Environment,
    /// Current instruction pointer within the current block
    instruction_pointer: usize,
    /// Current block being executed
    current_block: usize,
    /// Current region being executed
    current_region: usize,
    /// Measurement index to SSA ID mapping
    pub measurement_mappings: Vec<u32>, // SSA IDs that will receive measurement results
    /// Export mappings from Result operations (source SSA ID to export name)
    pub export_mappings: BTreeMap<u32, String>,
    /// SSA value storage (SSA ID to typed value)
    pub ssa_values: BTreeMap<u32, TypedValue>,
    /// Variable name to SSA ID mapping
    pub variable_ssa_map: BTreeMap<String, u32>,
    /// Final export values that persist across reset (export name to value)
    pub final_exports: BTreeMap<String, TypedValue>,
    /// Memory store for alloca/load/store operations (pointer SSA ID -> value)
    memory: BTreeMap<u32, TypedValue>,
    /// Number of qubits in the program
    qubit_count: usize,
}

impl PhirProcessor {
    /// Create a new PHIR processor
    #[must_use]
    pub fn new() -> Self {
        let environment = Environment::new();

        Self {
            environment,
            instruction_pointer: 0,
            current_block: 0,
            current_region: 0,
            measurement_mappings: Vec::new(),
            export_mappings: BTreeMap::new(),
            ssa_values: BTreeMap::new(),
            variable_ssa_map: BTreeMap::new(),
            final_exports: BTreeMap::new(),
            memory: BTreeMap::new(),
            qubit_count: 0,
        }
    }

    /// Reset the processor state
    pub fn reset(&mut self) {
        self.instruction_pointer = 0;
        self.current_block = 0;
        self.current_region = 0;
        self.measurement_mappings.clear();

        // Reset SSA values to defaults but keep variable definitions
        // We don't reset the environment completely because we need to preserve variable definitions
        for (var_name, &ssa_id) in &self.variable_ssa_map {
            if let Ok(Some(value)) = self.environment.get_variable(var_name) {
                // Reset to default value based on the variable's type
                let default_value = match value {
                    TypedValue::I64(_) => TypedValue::I64(0),
                    TypedValue::U32(_) => TypedValue::U32(0),
                    TypedValue::U64(_) => TypedValue::U64(0),
                    TypedValue::Bool(_) => TypedValue::Bool(false),
                    TypedValue::BitVec(bv) => TypedValue::BitVec(vec![false; bv.len()]),
                    _ => value.clone(),
                };
                self.ssa_values.insert(ssa_id, default_value.clone());
                // Also reset the environment variable to default
                let _ = self.environment.set_variable(var_name, default_value);
            }
        }

        // Clear memory store
        self.memory.clear();

        // Don't clear export_mappings, variable_ssa_map, or final_exports - they persist across shots
    }

    /// Process a PHIR module and generate quantum operations
    ///
    /// # Errors
    ///
    /// Returns an error if processing fails
    pub fn process_module(
        &mut self,
        module: &Module,
        message_builder: &mut ByteMessageBuilder,
    ) -> Result<bool> {
        // Start with the main function if it exists
        if let Some(main_block) = module.body.blocks.first() {
            self.process_block(main_block, message_builder)
        } else {
            Ok(false) // No operations to process
        }
    }

    /// Process a single block
    ///
    /// # Errors
    ///
    /// Returns an error if block processing fails
    pub fn process_block(
        &mut self,
        block: &Block,
        message_builder: &mut ByteMessageBuilder,
    ) -> Result<bool> {
        let mut has_quantum_ops = false;

        // Process operations starting from current instruction pointer
        while self.instruction_pointer < block.operations.len() {
            let instruction = &block.operations[self.instruction_pointer];

            let processed_quantum = self.process_instruction(instruction, message_builder)?;
            has_quantum_ops = has_quantum_ops || processed_quantum;

            self.instruction_pointer += 1;
        }

        Ok(has_quantum_ops)
    }

    /// Process a single instruction
    ///
    /// # Errors
    ///
    /// Returns an error if instruction processing fails
    pub fn process_instruction(
        &mut self,
        instruction: &crate::phir::Instruction,
        message_builder: &mut ByteMessageBuilder,
    ) -> Result<bool> {
        match &instruction.operation {
            Operation::Quantum(quantum_op) => {
                self.process_quantum_operation(quantum_op, instruction, message_builder)
            }
            Operation::Classical(classical_op) => {
                self.process_classical_operation(classical_op, instruction)?;
                Ok(false) // Classical operations don't generate quantum instructions
            }
            Operation::Builtin(builtin_op) => {
                self.process_builtin_operation(builtin_op, instruction, message_builder)
            }
            Operation::Custom(_) => {
                // For now, skip custom/dialect operations
                // TODO: Implement custom operation processing
                Ok(false)
            }
            Operation::ControlFlow(_) => {
                // Control flow is handled at the engine level, not here
                Ok(false)
            }
            Operation::Memory(mem_op) => {
                self.process_memory_operation(mem_op, instruction);
                Ok(false)
            }
            Operation::Parsing(_) => {
                // Skip parsing operations during execution
                Ok(false)
            }
        }
    }

    /// Process a quantum operation
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Required operands are missing
    /// - Operand indices are invalid
    /// - SSA values cannot be resolved
    pub fn process_quantum_operation(
        &mut self,
        quantum_op: &crate::ops::QuantumOp,
        instruction: &crate::phir::Instruction,
        message_builder: &mut ByteMessageBuilder,
    ) -> Result<bool> {
        match quantum_op {
            // Fixed single-qubit gates
            QuantumOp::H => self.process_single_qubit_gate("H", instruction, message_builder),
            QuantumOp::X => self.process_single_qubit_gate("X", instruction, message_builder),
            QuantumOp::Y => self.process_single_qubit_gate("Y", instruction, message_builder),
            QuantumOp::Z => self.process_single_qubit_gate("Z", instruction, message_builder),
            QuantumOp::S => self.process_single_qubit_gate("S", instruction, message_builder),
            QuantumOp::Sdg => self.process_single_qubit_gate("Sdg", instruction, message_builder),
            QuantumOp::T => self.process_single_qubit_gate("T", instruction, message_builder),
            QuantumOp::Tdg => self.process_single_qubit_gate("Tdg", instruction, message_builder),

            // Parameterized single-qubit gates
            QuantumOp::RX(angle) => {
                let qubit_id = self.extract_single_qubit(instruction, "RX")?;
                message_builder.rx(*angle, &[qubit_id]);
                Ok(true)
            }
            QuantumOp::RY(angle) => {
                let qubit_id = self.extract_single_qubit(instruction, "RY")?;
                message_builder.ry(*angle, &[qubit_id]);
                Ok(true)
            }
            QuantumOp::RZ(angle) => {
                let qubit_id = self.extract_single_qubit(instruction, "RZ")?;
                message_builder.rz(*angle, &[qubit_id]);
                Ok(true)
            }
            QuantumOp::R1XY(theta, phi) => {
                let qubit_id = self.extract_single_qubit(instruction, "R1XY")?;
                message_builder.r1xy(*theta, *phi, &[qubit_id]);
                Ok(true)
            }
            QuantumOp::U3(theta, phi, lambda) => {
                let qubit_id = self.extract_single_qubit(instruction, "U3")?;
                message_builder.u(*theta, *phi, *lambda, &[qubit_id]);
                Ok(true)
            }

            // Two-qubit gates
            QuantumOp::CX => self.process_two_qubit_gate("CX", instruction, message_builder),
            QuantumOp::CZ => self.process_two_qubit_gate("CZ", instruction, message_builder),
            QuantumOp::SWAP => {
                let (q1, q2) = self.extract_two_qubits(instruction, "SWAP")?;
                let gate = Gate::swap(&[(q1, q2)]);
                message_builder.add_gate_command(&gate);
                Ok(true)
            }
            QuantumOp::RZZ(angle) => {
                let (q1, q2) = self.extract_two_qubits(instruction, "RZZ")?;
                message_builder.rzz(*angle, &[(q1, q2)]);
                Ok(true)
            }
            QuantumOp::CPhase(angle) => {
                let (q1, q2) = self.extract_two_qubits(instruction, "CPhase")?;
                let gate = Gate::crz(*angle, &[(q1, q2)]);
                message_builder.add_gate_command(&gate);
                Ok(true)
            }

            // Measurement
            QuantumOp::Measure => self.process_measurement(instruction, message_builder),

            // Resource management
            QuantumOp::Alloc => {
                if !instruction.results.is_empty() {
                    let qubit_id = usize::try_from(instruction.results[0].id).unwrap_or(usize::MAX);
                    self.qubit_count = self.qubit_count.max(qubit_id + 1);
                    let gate = Gate::qalloc(&[qubit_id]);
                    message_builder.add_gate_command(&gate);
                }
                Ok(true)
            }
            QuantumOp::Dealloc => {
                if !instruction.operands.is_empty() {
                    let qubit_id =
                        usize::try_from(instruction.operands[0].id).unwrap_or(usize::MAX);
                    let gate = Gate::qfree(&[qubit_id]);
                    message_builder.add_gate_command(&gate);
                }
                Ok(true)
            }
            QuantumOp::Reset => {
                if !instruction.operands.is_empty() {
                    let qubit_id =
                        usize::try_from(instruction.operands[0].id).unwrap_or(usize::MAX);
                    message_builder.pz(&[qubit_id]);
                }
                Ok(true)
            }
            QuantumOp::InitZero => {
                // InitZero is equivalent to Reset (prepare |0>)
                if !instruction.operands.is_empty() {
                    let qubit_id =
                        usize::try_from(instruction.operands[0].id).unwrap_or(usize::MAX);
                    message_builder.pz(&[qubit_id]);
                }
                Ok(true)
            }

            _ => Err(PhirError::internal(format!(
                "Quantum operation not yet implemented: {quantum_op:?}"
            ))),
        }
    }

    /// Extract a single qubit ID from instruction operands
    fn extract_single_qubit(
        &mut self,
        instruction: &crate::phir::Instruction,
        gate_name: &str,
    ) -> Result<usize> {
        if instruction.operands.len() != 1 {
            return Err(PhirError::internal(format!(
                "{gate_name} gate requires exactly 1 operand, got {}",
                instruction.operands.len()
            )));
        }
        let qubit_id = usize::try_from(instruction.operands[0].id).unwrap_or(usize::MAX);
        self.qubit_count = self.qubit_count.max(qubit_id + 1);
        Ok(qubit_id)
    }

    /// Extract two qubit IDs from instruction operands
    fn extract_two_qubits(
        &mut self,
        instruction: &crate::phir::Instruction,
        gate_name: &str,
    ) -> Result<(usize, usize)> {
        if instruction.operands.len() != 2 {
            return Err(PhirError::internal(format!(
                "{gate_name} gate requires exactly 2 operands, got {}",
                instruction.operands.len()
            )));
        }
        let q1 = usize::try_from(instruction.operands[0].id).unwrap_or(usize::MAX);
        let q2 = usize::try_from(instruction.operands[1].id).unwrap_or(usize::MAX);
        self.qubit_count = self.qubit_count.max(q1 + 1);
        self.qubit_count = self.qubit_count.max(q2 + 1);
        Ok((q1, q2))
    }

    /// Process a single-qubit gate
    fn process_single_qubit_gate(
        &mut self,
        gate_name: &str,
        instruction: &crate::phir::Instruction,
        message_builder: &mut ByteMessageBuilder,
    ) -> Result<bool> {
        let qubit_id = self.extract_single_qubit(instruction, gate_name)?;

        match gate_name {
            "H" => {
                message_builder.h(&[qubit_id]);
            }
            "X" => {
                message_builder.x(&[qubit_id]);
            }
            "Y" => {
                message_builder.y(&[qubit_id]);
            }
            "Z" => {
                message_builder.z(&[qubit_id]);
            }
            "S" => {
                message_builder.sz(&[qubit_id]);
            }
            "Sdg" => {
                message_builder.szdg(&[qubit_id]);
            }
            "T" => {
                message_builder.t(&[qubit_id]);
            }
            "Tdg" => {
                message_builder.tdg(&[qubit_id]);
            }
            _ => {
                return Err(PhirError::internal(format!(
                    "Unknown single-qubit gate: {gate_name}"
                )));
            }
        }

        Ok(true)
    }

    /// Process a two-qubit gate
    fn process_two_qubit_gate(
        &mut self,
        gate_name: &str,
        instruction: &crate::phir::Instruction,
        message_builder: &mut ByteMessageBuilder,
    ) -> Result<bool> {
        let (q1, q2) = self.extract_two_qubits(instruction, gate_name)?;

        match gate_name {
            "CX" => {
                message_builder.cx(&[(q1, q2)]);
            }
            "CZ" => {
                message_builder.cz(&[(q1, q2)]);
            }
            _ => {
                return Err(PhirError::internal(format!(
                    "Unknown two-qubit gate: {gate_name}"
                )));
            }
        }

        Ok(true)
    }

    /// Process a measurement operation
    fn process_measurement(
        &mut self,
        instruction: &crate::phir::Instruction,
        message_builder: &mut ByteMessageBuilder,
    ) -> Result<bool> {
        if instruction.operands.is_empty() {
            return Err(PhirError::internal(
                "Measurement requires at least 1 operand",
            ));
        }

        // For now, process single-qubit measurements
        // TODO: Support multi-qubit measurements
        let qubit_id = usize::try_from(instruction.operands[0].id).unwrap_or(usize::MAX);

        // Track maximum qubit index
        self.qubit_count = self.qubit_count.max(qubit_id + 1);

        message_builder.mz(&[qubit_id]);

        // Track measurement mapping for later processing
        // The measurement index maps to which variable should receive the result
        let _measurement_index = self.measurement_mappings.len();

        // Store the SSA ID that will receive this measurement result
        if !instruction.results.is_empty() {
            let result_ssa_id = instruction.results[0].id;
            self.measurement_mappings.push(result_ssa_id);
        }

        Ok(true)
    }

    /// Process a classical operation
    ///
    /// # Errors
    ///
    /// Returns an error if type conversion fails
    ///
    /// # Panics
    ///
    /// Panics if a shift amount or constant value doesn't fit in the expected type
    pub fn process_classical_operation(
        &mut self,
        classical_op: &crate::ops::ClassicalOp,
        instruction: &crate::phir::Instruction,
    ) -> Result<()> {
        match classical_op {
            // Constants
            ClassicalOp::ConstInt(value) => {
                self.process_const_int_operation(*value, instruction);
                Ok(())
            }
            ClassicalOp::ConstFloat(value) => {
                if !instruction.results.is_empty() {
                    let ssa_id = instruction.results[0].id;
                    self.ssa_values.insert(ssa_id, TypedValue::F64(*value));
                }
                Ok(())
            }
            ClassicalOp::ConstBool(value) => {
                if !instruction.results.is_empty() {
                    let ssa_id = instruction.results[0].id;
                    self.ssa_values.insert(ssa_id, TypedValue::Bool(*value));
                }
                Ok(())
            }

            // Binary arithmetic
            ClassicalOp::Add => {
                self.process_binary_int_op(
                    instruction,
                    "add",
                    i64::wrapping_add,
                    u64::wrapping_add,
                );
                Ok(())
            }
            ClassicalOp::Sub => {
                self.process_binary_int_op(
                    instruction,
                    "sub",
                    i64::wrapping_sub,
                    u64::wrapping_sub,
                );
                Ok(())
            }
            ClassicalOp::Mul => {
                self.process_binary_int_op(
                    instruction,
                    "mul",
                    i64::wrapping_mul,
                    u64::wrapping_mul,
                );
                Ok(())
            }
            ClassicalOp::Div => {
                self.process_binary_int_op(
                    instruction,
                    "div",
                    |a, b| a.checked_div(b).unwrap_or(0),
                    |a, b| a.checked_div(b).unwrap_or(0),
                );
                Ok(())
            }
            ClassicalOp::Mod => {
                self.process_binary_int_op(
                    instruction,
                    "mod",
                    |a, b| if b == 0 { 0 } else { a % b },
                    |a, b| if b == 0 { 0 } else { a % b },
                );
                Ok(())
            }

            // Bitwise
            ClassicalOp::And => {
                self.process_binary_int_op(instruction, "and", |a, b| a & b, |a, b| a & b);
                Ok(())
            }
            ClassicalOp::Or => {
                self.process_binary_int_op(instruction, "or", |a, b| a | b, |a, b| a | b);
                Ok(())
            }
            ClassicalOp::Xor => {
                self.process_binary_int_op(instruction, "xor", |a, b| a ^ b, |a, b| a ^ b);
                Ok(())
            }
            ClassicalOp::Shl(shift) => {
                if instruction.operands.len() >= 2 {
                    // Binary mode: shift amount from second operand (used by QIS parser)
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    // shift amount is non-negative
                    self.process_binary_int_op(
                        instruction,
                        "shl",
                        |a, b| a.wrapping_shl(b as u32),
                        |a, b| a.wrapping_shl(b as u32),
                    );
                } else {
                    let s = *shift;
                    self.process_unary_int_op(
                        instruction,
                        move |v: i64| v.wrapping_shl(s),
                        move |v: u64| v.wrapping_shl(s),
                    );
                }
                Ok(())
            }
            ClassicalOp::Shr(shift) => {
                if instruction.operands.len() >= 2 {
                    // Binary mode: shift amount from second operand (used by QIS parser)
                    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                    // shift amount is non-negative
                    self.process_binary_int_op(
                        instruction,
                        "shr",
                        |a, b| a.wrapping_shr(b as u32),
                        |a, b| a.wrapping_shr(b as u32),
                    );
                } else {
                    let s = *shift;
                    self.process_unary_int_op(
                        instruction,
                        move |v: i64| v.wrapping_shr(s),
                        move |v: u64| v.wrapping_shr(s),
                    );
                }
                Ok(())
            }
            ClassicalOp::Not => {
                if !instruction.operands.is_empty() && !instruction.results.is_empty() {
                    let op_id = instruction.operands[0].id;
                    let res_id = instruction.results[0].id;
                    if let Some(val) = self.ssa_values.get(&op_id) {
                        let result = match val {
                            TypedValue::Bool(v) => TypedValue::Bool(!v),
                            TypedValue::I32(v) => TypedValue::I32(!v),
                            TypedValue::I64(v) => TypedValue::I64(!v),
                            TypedValue::U32(v) => TypedValue::U32(!v),
                            TypedValue::U64(v) => TypedValue::U64(!v),
                            other => other.clone(),
                        };
                        self.ssa_values.insert(res_id, result);
                    }
                }
                Ok(())
            }
            ClassicalOp::Neg => {
                if !instruction.operands.is_empty() && !instruction.results.is_empty() {
                    let op_id = instruction.operands[0].id;
                    let res_id = instruction.results[0].id;
                    if let Some(val) = self.ssa_values.get(&op_id) {
                        let result = match val {
                            TypedValue::I32(v) => TypedValue::I32(v.wrapping_neg()),
                            TypedValue::I64(v) => TypedValue::I64(v.wrapping_neg()),
                            TypedValue::F64(v) => TypedValue::F64(-v),
                            other => other.clone(),
                        };
                        self.ssa_values.insert(res_id, result);
                    }
                }
                Ok(())
            }

            // Comparisons
            ClassicalOp::Eq => {
                self.process_comparison(instruction, |ord| ord == std::cmp::Ordering::Equal);
                Ok(())
            }
            ClassicalOp::Ne => {
                self.process_comparison(instruction, |ord| ord != std::cmp::Ordering::Equal);
                Ok(())
            }
            ClassicalOp::Lt => {
                self.process_comparison(instruction, |ord| ord == std::cmp::Ordering::Less);
                Ok(())
            }
            ClassicalOp::Le => {
                self.process_comparison(instruction, |ord| ord != std::cmp::Ordering::Greater);
                Ok(())
            }
            ClassicalOp::Gt => {
                self.process_comparison(instruction, |ord| ord == std::cmp::Ordering::Greater);
                Ok(())
            }
            ClassicalOp::Ge => {
                self.process_comparison(instruction, |ord| ord != std::cmp::Ordering::Less);
                Ok(())
            }

            // Select (ternary)
            ClassicalOp::Select => {
                if instruction.operands.len() >= 3 && !instruction.results.is_empty() {
                    let cond_id = instruction.operands[0].id;
                    let true_id = instruction.operands[1].id;
                    let false_id = instruction.operands[2].id;
                    let res_id = instruction.results[0].id;

                    let cond = self.ssa_values.get(&cond_id).is_some_and(|v| match v {
                        TypedValue::Bool(b) => *b,
                        TypedValue::U32(v) => *v != 0,
                        TypedValue::I64(v) => *v != 0,
                        _ => false,
                    });

                    let chosen_id = if cond { true_id } else { false_id };
                    if let Some(val) = self.ssa_values.get(&chosen_id) {
                        self.ssa_values.insert(res_id, val.clone());
                    }
                }
                Ok(())
            }

            // Type conversions
            ClassicalOp::Bitcast => {
                self.process_bitcast_operation(instruction);
                Ok(())
            }

            // Assignment
            ClassicalOp::Assign => {
                if !instruction.operands.is_empty() && !instruction.results.is_empty() {
                    let src_id = instruction.operands[0].id;
                    let dst_id = instruction.results[0].id;
                    if let Some(val) = self.ssa_values.get(&src_id) {
                        self.ssa_values.insert(dst_id, val.clone());
                    }
                }
                Ok(())
            }

            // Result export
            ClassicalOp::Result => {
                self.process_result_operation(instruction);
                Ok(())
            }

            // Float arithmetic
            ClassicalOp::FAdd => {
                self.process_binary_float_op(instruction, |a, b| a + b);
                Ok(())
            }
            ClassicalOp::FSub => {
                self.process_binary_float_op(instruction, |a, b| a - b);
                Ok(())
            }
            ClassicalOp::FMul => {
                self.process_binary_float_op(instruction, |a, b| a * b);
                Ok(())
            }
            ClassicalOp::FDiv => {
                self.process_binary_float_op(
                    instruction,
                    |a, b| if b == 0.0 { 0.0 } else { a / b },
                );
                Ok(())
            }
            ClassicalOp::FNeg => {
                if !instruction.operands.is_empty() && !instruction.results.is_empty() {
                    let op_id = instruction.operands[0].id;
                    let res_id = instruction.results[0].id;
                    if let Some(TypedValue::F64(v)) = self.ssa_values.get(&op_id) {
                        self.ssa_values.insert(res_id, TypedValue::F64(-v));
                    }
                }
                Ok(())
            }

            _ => {
                // Skip unimplemented classical ops without error
                Ok(())
            }
        }
    }

    /// Helper: process a binary integer operation on two SSA operands
    fn process_binary_int_op(
        &mut self,
        instruction: &crate::phir::Instruction,
        _name: &str,
        signed_op: impl Fn(i64, i64) -> i64,
        unsigned_op: impl Fn(u64, u64) -> u64,
    ) {
        if instruction.operands.len() < 2 || instruction.results.is_empty() {
            return;
        }
        let left_id = instruction.operands[0].id;
        let right_id = instruction.operands[1].id;
        let res_id = instruction.results[0].id;

        let left = self.ssa_values.get(&left_id).cloned();
        let right = self.ssa_values.get(&right_id).cloned();

        if let (Some(l), Some(r)) = (left, right) {
            let result = match (&l, &r) {
                (TypedValue::I32(a), TypedValue::I32(b)) =>
                {
                    #[allow(clippy::cast_possible_truncation)]
                    TypedValue::I32(signed_op(i64::from(*a), i64::from(*b)) as i32)
                }
                (TypedValue::I64(a), TypedValue::I64(b)) => TypedValue::I64(signed_op(*a, *b)),
                (TypedValue::U32(a), TypedValue::U32(b)) =>
                {
                    #[allow(clippy::cast_possible_truncation)]
                    TypedValue::U32(unsigned_op(u64::from(*a), u64::from(*b)) as u32)
                }
                (TypedValue::U64(a), TypedValue::U64(b)) => TypedValue::U64(unsigned_op(*a, *b)),
                // Mixed types: coerce to I64
                _ => {
                    let a = l.to_u64().unwrap_or(0);
                    let b = r.to_u64().unwrap_or(0);
                    #[allow(clippy::cast_possible_wrap)]
                    TypedValue::I64(signed_op(a as i64, b as i64))
                }
            };
            self.ssa_values.insert(res_id, result);
        }
    }

    /// Helper: process a unary integer operation
    fn process_unary_int_op(
        &mut self,
        instruction: &crate::phir::Instruction,
        signed_op: impl Fn(i64) -> i64,
        unsigned_op: impl Fn(u64) -> u64,
    ) {
        if instruction.operands.is_empty() || instruction.results.is_empty() {
            return;
        }
        let op_id = instruction.operands[0].id;
        let res_id = instruction.results[0].id;

        if let Some(val) = self.ssa_values.get(&op_id).cloned() {
            let result = match val {
                #[allow(clippy::cast_possible_truncation)]
                TypedValue::I32(v) => TypedValue::I32(signed_op(i64::from(v)) as i32),
                TypedValue::I64(v) => TypedValue::I64(signed_op(v)),
                #[allow(clippy::cast_possible_truncation)]
                TypedValue::U32(v) => TypedValue::U32(unsigned_op(u64::from(v)) as u32),
                TypedValue::U64(v) => TypedValue::U64(unsigned_op(v)),
                other => other,
            };
            self.ssa_values.insert(res_id, result);
        }
    }

    /// Helper: process a binary float operation
    fn process_binary_float_op(
        &mut self,
        instruction: &crate::phir::Instruction,
        op: impl Fn(f64, f64) -> f64,
    ) {
        if instruction.operands.len() < 2 || instruction.results.is_empty() {
            return;
        }
        let left_id = instruction.operands[0].id;
        let right_id = instruction.operands[1].id;
        let res_id = instruction.results[0].id;

        if let (Some(TypedValue::F64(a)), Some(TypedValue::F64(b))) = (
            self.ssa_values.get(&left_id),
            self.ssa_values.get(&right_id),
        ) {
            self.ssa_values.insert(res_id, TypedValue::F64(op(*a, *b)));
        }
    }

    /// Helper: process a comparison operation returning Bool
    fn process_comparison(
        &mut self,
        instruction: &crate::phir::Instruction,
        cmp_fn: impl Fn(std::cmp::Ordering) -> bool,
    ) {
        if instruction.operands.len() < 2 || instruction.results.is_empty() {
            return;
        }
        let left_id = instruction.operands[0].id;
        let right_id = instruction.operands[1].id;
        let res_id = instruction.results[0].id;

        let left = self.ssa_values.get(&left_id).cloned();
        let right = self.ssa_values.get(&right_id).cloned();

        if let (Some(l), Some(r)) = (left, right) {
            let ordering = match (&l, &r) {
                (TypedValue::I32(a), TypedValue::I32(b)) => a.cmp(b),
                (TypedValue::I64(a), TypedValue::I64(b)) => a.cmp(b),
                (TypedValue::U32(a), TypedValue::U32(b)) => a.cmp(b),
                (TypedValue::U64(a), TypedValue::U64(b)) => a.cmp(b),
                (TypedValue::Bool(a), TypedValue::Bool(b)) => a.cmp(b),
                // Mixed: coerce to i64
                _ => {
                    let a = l.to_u64().unwrap_or(0);
                    let b = r.to_u64().unwrap_or(0);
                    a.cmp(&b)
                }
            };
            self.ssa_values
                .insert(res_id, TypedValue::Bool(cmp_fn(ordering)));
        }
    }

    /// Process a builtin operation
    ///
    /// # Errors
    ///
    /// Returns an error if builtin operation processing fails
    pub fn process_builtin_operation(
        &mut self,
        builtin_op: &crate::builtin_ops::BuiltinOp,
        instruction: &crate::phir::Instruction,
        _message_builder: &mut ByteMessageBuilder,
    ) -> Result<bool> {
        match builtin_op {
            BuiltinOp::VarDefine(var_def) => {
                // Handle variable definition
                self.process_var_define(var_def, instruction)?;
                Ok(false) // Variable definitions don't generate quantum operations
            }
            BuiltinOp::Module(_) | BuiltinOp::Func(_) | BuiltinOp::Return(_) => {
                // Skip structural operations during execution
                Ok(false)
            }
        }
    }

    /// Handle measurement results by updating SSA values
    /// For measurements into bit-indexed variables, combine results into single integer
    ///
    /// # Errors
    ///
    /// Returns an error if measurement result handling fails
    pub fn handle_measurement_results(&mut self, outcomes: &[u8]) -> Result<()> {
        // Process measurement outcomes

        // Create a map to track which base variable each measurement SSA ID belongs to
        let mut measurement_to_base: BTreeMap<u32, (String, u32, usize)> = BTreeMap::new();

        // For each variable, check if any measurement SSA IDs are offsets of it
        for (var_name, &base_ssa_id) in &self.variable_ssa_map {
            for &meas_ssa_id in &self.measurement_mappings {
                // Check if this measurement SSA ID is base_ssa_id + offset (0-9)
                if meas_ssa_id >= base_ssa_id && meas_ssa_id < base_ssa_id + 10 {
                    let offset = usize::try_from(meas_ssa_id - base_ssa_id).unwrap_or(0);
                    measurement_to_base
                        .insert(meas_ssa_id, (var_name.clone(), base_ssa_id, offset));
                    // Map measurement SSA to variable bit offset
                }
            }
        }

        // First, store individual measurement outcomes as bools
        for (i, &outcome) in outcomes.iter().enumerate() {
            if i < self.measurement_mappings.len() {
                let ssa_id = self.measurement_mappings[i];
                let value = TypedValue::Bool(outcome != 0);
                // Store measurement outcome
                self.ssa_values.insert(ssa_id, value);
            }

            // Also store in standard measurement variable for compatibility
            let standard_var = format!("measurement_{i}");
            let value = TypedValue::U8(outcome);
            if !self.environment.has_variable(&standard_var) {
                self.environment
                    .add_variable(&standard_var, DataType::U8, 1)?;
            }
            self.environment.set_variable(&standard_var, value)?;
        }

        // Now combine measurement results for integer variables
        let mut combined_values: BTreeMap<u32, u32> = BTreeMap::new();

        // Process each measurement and accumulate bits for its base variable
        for (i, &outcome) in outcomes.iter().enumerate() {
            if i < self.measurement_mappings.len() {
                let meas_ssa_id = self.measurement_mappings[i];

                if let Some((var_name, base_ssa_id, bit_offset)) =
                    measurement_to_base.get(&meas_ssa_id)
                {
                    // Measurement contributes to variable bit

                    // Only process if it's an integer variable
                    if let Ok(Some(
                        TypedValue::I64(_)
                        | TypedValue::U32(_)
                        | TypedValue::U64(_)
                        | TypedValue::I32(_),
                    )) = self.environment.get_variable(var_name)
                    {
                        let current_value = combined_values.entry(*base_ssa_id).or_insert(0);
                        if outcome != 0 {
                            *current_value |= 1 << bit_offset;
                        }
                    }
                }
            }
        }

        // Store the combined values for integer variables
        // Store the combined values for integer variables
        for (base_ssa_id, combined_value) in combined_values {
            // Find the variable name for this SSA ID
            if let Some((var_name, _)) = self
                .variable_ssa_map
                .iter()
                .find(|(_, id)| **id == base_ssa_id)
            {
                // Check if it's an integer type
                if let Ok(Some(
                    TypedValue::I64(_)
                    | TypedValue::U32(_)
                    | TypedValue::U64(_)
                    | TypedValue::I32(_),
                )) = self.environment.get_variable(var_name)
                {
                    let new_value = TypedValue::U32(combined_value);
                    // Set variable to combined value
                    self.ssa_values.insert(base_ssa_id, new_value.clone());
                    // Also update environment
                    let _ = self.environment.set_variable(var_name, new_value);
                } else {
                    // Could not get variable from environment
                }
            } else {
                // Could not find variable name for SSA ID
            }
        }

        Ok(())
    }

    /// Finalize export values after measurements are processed
    /// This should be called after `handle_measurement_results` to prepare exports
    pub fn finalize_exports(&mut self) {
        // Don't clear previous exports - they should persist and be updated
        // self.final_exports.clear();

        // Process export mappings

        // Process each export mapping
        for (src_ssa_id, export_name) in &self.export_mappings {
            // Process export from SSA ID

            // Check if this is a base SSA ID for an integer variable that should have combined bits
            if let Some((_var_name, _)) = self
                .variable_ssa_map
                .iter()
                .find(|(_, id)| **id == *src_ssa_id)
            {
                // SSA belongs to a variable

                // Look for measurement SSA IDs that are offsets of this base SSA ID
                let mut combined_value = 0u32;
                let mut found_bits = false;

                for &meas_ssa_id in &self.measurement_mappings {
                    if meas_ssa_id >= *src_ssa_id && meas_ssa_id < *src_ssa_id + 10 {
                        found_bits = true;
                        let bit_offset = usize::try_from(meas_ssa_id - src_ssa_id).unwrap_or(0);

                        // Get the Bool value from the measurement SSA ID
                        if let Some(TypedValue::Bool(bit_value)) = self.ssa_values.get(&meas_ssa_id)
                            && *bit_value
                        {
                            combined_value |= 1 << bit_offset;
                        }
                        // Found bit value for variable
                    }
                }

                if found_bits {
                    // We found measurement bits - export the combined value
                    let export_value = TypedValue::U32(combined_value);
                    // Export the combined bit value
                    self.final_exports.insert(export_name.clone(), export_value);
                    continue;
                }
            }

            // Fall back to exporting the SSA value directly
            if let Some(value) = self.ssa_values.get(src_ssa_id) {
                // Export the SSA value directly
                self.final_exports
                    .insert(export_name.clone(), value.clone());
            } else {
                // SSA not found for export
            }
        }
        // Export processing complete
    }

    /// Get the number of qubits used in the program
    #[must_use]
    pub fn get_qubit_count(&self) -> usize {
        self.qubit_count
    }

    /// Add a variable definition
    ///
    /// # Errors
    ///
    /// Returns an error if the variable cannot be added
    pub fn add_variable(&mut self, name: &str, data_type: DataType, size: usize) -> Result<()> {
        self.environment.add_variable(name, data_type, size)
    }

    /// Extract variable definitions from PHIR module during initialization
    /// This follows `PhirJsonEngine` pattern of processing variables upfront
    ///
    /// # Errors
    ///
    /// Returns an error if variable extraction fails
    pub fn extract_variable_definitions(&mut self, module: &crate::phir::Module) -> Result<()> {
        // First look for VarDefine operations in the top-level blocks
        self.extract_variable_definitions_from_region(&module.body)?;

        // Also look inside function bodies
        for block in &module.body.blocks {
            for instruction in &block.operations {
                if let crate::ops::Operation::Builtin(crate::builtin_ops::BuiltinOp::Func(
                    func_op,
                )) = &instruction.operation
                {
                    // Process each region in the function body
                    for region in &func_op.body {
                        self.extract_variable_definitions_from_region(region)?;
                    }
                }
            }
        }

        Ok(())
    }

    /// Extract variable definitions from a region
    fn extract_variable_definitions_from_region(
        &mut self,
        region: &crate::phir::Region,
    ) -> Result<()> {
        for block in &region.blocks {
            for instruction in &block.operations {
                if let crate::ops::Operation::Builtin(crate::builtin_ops::BuiltinOp::VarDefine(
                    var_def,
                )) = &instruction.operation
                {
                    // Map PHIR type strings to DataType enum
                    let data_type = match var_def.var_type.as_str() {
                        "qubits" => DataType::Qubits,
                        "i8" => DataType::I8,
                        "i16" => DataType::I16,
                        "i32" => DataType::I32,
                        "u8" => DataType::U8,
                        "u16" => DataType::U16,
                        "u32" => DataType::U32,
                        "u64" => DataType::U64,
                        "f64" => DataType::F64,
                        "bool" => DataType::Bool,
                        _ => DataType::I64, // Default to I64 (includes "i64")
                    };

                    // Add the variable to the environment
                    // Add variable to environment
                    self.environment
                        .add_variable(&var_def.name, data_type, var_def.size)?;

                    // Track qubit count
                    if data_type == DataType::Qubits {
                        self.qubit_count = self.qubit_count.max(var_def.size);
                    }

                    // Also create an SSA value for this variable if it has a result
                    if !instruction.results.is_empty() {
                        let ssa_id = instruction.results[0].id;
                        // Map variable name to SSA ID
                        self.variable_ssa_map.insert(var_def.name.clone(), ssa_id);

                        // Initialize with default value based on type
                        let default_value = match data_type {
                            DataType::I8 | DataType::I16 | DataType::I32 | DataType::I64 => {
                                TypedValue::I64(0)
                            }
                            DataType::U8 | DataType::U16 | DataType::U32 | DataType::U64 => {
                                if var_def.size > 1 {
                                    TypedValue::U32(0)
                                } else {
                                    TypedValue::U64(0)
                                }
                            }
                            DataType::F64 => TypedValue::F64(0.0),
                            DataType::Bool => TypedValue::Bool(false),
                            DataType::Qubits => TypedValue::BitVec(vec![false; var_def.size]),
                        };
                        // Initialize SSA value
                        self.ssa_values.insert(ssa_id, default_value.clone());

                        // Also set the initial value in the environment
                        let _ = self.environment.set_variable(&var_def.name, default_value);
                    }
                }
            }
        }
        Ok(())
    }

    /// Get all results for export
    #[must_use]
    pub fn get_results(&self) -> BTreeMap<String, TypedValue> {
        self.environment.get_all_variables()
    }

    /// Get export results based on finalized exports
    /// Returns the final export values that were computed after measurements
    #[must_use]
    pub fn get_export_results(&self) -> BTreeMap<String, TypedValue> {
        self.final_exports.clone()
    }

    /// Process a variable definition operation
    fn process_var_define(
        &mut self,
        var_def: &crate::builtin_ops::VarDefineOp,
        _instruction: &crate::phir::Instruction,
    ) -> Result<()> {
        // Map PHIR type strings to DataType enum
        let data_type = match var_def.var_type.as_str() {
            "qubits" => DataType::Qubits,
            "i8" => DataType::I8,
            "i16" => DataType::I16,
            "i32" => DataType::I32,
            "i64" => DataType::I64,
            "u8" => DataType::U8,
            "u16" => DataType::U16,
            "u32" => DataType::U32,
            "u64" => DataType::U64,
            "f64" => DataType::F64,
            "bool" => DataType::Bool,
            _ => {
                return Err(PhirError::internal(format!(
                    "Unknown variable type: {}",
                    var_def.var_type
                )));
            }
        };

        // Track qubit count
        if data_type == DataType::Qubits {
            self.qubit_count = self.qubit_count.max(var_def.size);
        }

        // Add the variable to the environment
        self.environment
            .add_variable(&var_def.name, data_type, var_def.size)
    }

    /// Process a memory operation (alloca, load, store)
    fn process_memory_operation(
        &mut self,
        mem_op: &MemoryOp,
        instruction: &crate::phir::Instruction,
    ) {
        match mem_op {
            MemoryOp::Alloc(alloc_type) if !instruction.results.is_empty() => {
                let ptr_id = instruction.results[0].id;
                let default = match alloc_type {
                    #[allow(clippy::match_same_arms)]
                    crate::ops::AllocType::Scalar(ty) => match ty {
                        crate::types::Type::Int(
                            crate::types::IntWidth::I8
                            | crate::types::IntWidth::I16
                            | crate::types::IntWidth::I32,
                        ) => TypedValue::I32(0),
                        crate::types::Type::Int(_) => TypedValue::I64(0),
                        crate::types::Type::UInt(
                            crate::types::IntWidth::I8
                            | crate::types::IntWidth::I16
                            | crate::types::IntWidth::I32,
                        ) => TypedValue::U32(0),
                        crate::types::Type::UInt(_) => TypedValue::U64(0),
                        crate::types::Type::Bool => TypedValue::Bool(false),
                        crate::types::Type::Float(_) => TypedValue::F64(0.0),
                        _ => TypedValue::I64(0),
                    },
                    _ => TypedValue::I64(0),
                };
                self.memory.insert(ptr_id, default);
            }
            MemoryOp::Load
                if !instruction.operands.is_empty() && !instruction.results.is_empty() =>
            {
                let ptr_id = instruction.operands[0].id;
                let res_id = instruction.results[0].id;
                if let Some(val) = self.memory.get(&ptr_id) {
                    self.ssa_values.insert(res_id, val.clone());
                }
            }
            MemoryOp::Store if instruction.operands.len() >= 2 => {
                let val_id = instruction.operands[0].id;
                let ptr_id = instruction.operands[1].id;
                if let Some(val) = self.ssa_values.get(&val_id) {
                    self.memory.insert(ptr_id, val.clone());
                }
            }
            _ => {} // Skip other memory ops
        }
    }

    /// Process a Result operation - immediately export the value
    fn process_result_operation(&mut self, instruction: &crate::phir::Instruction) {
        // Result operations export values immediately
        // {"cop": "Result", "args": ["m"], "returns": ["bell_result"]}

        if !instruction.operands.is_empty() {
            let operand_ssa_id = instruction.operands[0].id;

            // Get the export name from attributes
            let mut export_name = None;
            for (key, value) in &instruction.attributes {
                if key.starts_with("export_name")
                    && let crate::phir::AttributeValue::String(name) = value
                {
                    export_name = Some(name.clone());
                    break;
                }
            }

            if let Some(name) = export_name {
                // Get the value to export
                if let Some(value) = self.ssa_values.get(&operand_ssa_id) {
                    self.final_exports.insert(name, value.clone());
                }
            }
        }
    }

    /// Process a `ConstInt` operation - creates an integer constant
    fn process_const_int_operation(&mut self, value: i64, instruction: &crate::phir::Instruction) {
        if !instruction.results.is_empty() {
            let result_ssa_id = instruction.results[0].id;
            // Store the constant value as U32 for bit operations
            // Quantum operations typically use small constants, wrapping is intentional
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let value_u32 = value as u32;
            self.ssa_values
                .insert(result_ssa_id, TypedValue::U32(value_u32));
        }
    }

    /// Process a Bitcast operation - converts bool to int
    fn process_bitcast_operation(&mut self, instruction: &crate::phir::Instruction) {
        if !instruction.operands.is_empty() && !instruction.results.is_empty() {
            let operand_ssa_id = instruction.operands[0].id;
            let result_ssa_id = instruction.results[0].id;

            // Get the bool value and convert to int
            if let Some(TypedValue::Bool(bool_val)) = self.ssa_values.get(&operand_ssa_id) {
                let int_val = u32::from(*bool_val);
                self.ssa_values
                    .insert(result_ssa_id, TypedValue::U32(int_val));
            }
        }
    }
}

impl Default for PhirProcessor {
    fn default() -> Self {
        Self::new()
    }
}
