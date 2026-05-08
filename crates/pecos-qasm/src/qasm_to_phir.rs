//! QASM to PHIR conversion
//!
//! Converts a parsed QASM `Program` into a PHIR `Module` that can be executed
//! by the `PhirEngine` or serialized to RON for debugging.

use pecos_core::Angle64;
use pecos_core::prelude::GateType;
use pecos_phir::Result;
use pecos_phir::builtin_ops::{BuiltinOp, FuncOp, VarDefineOp};
use pecos_phir::ops::{ClassicalOp, Operation, QuantumOp, SSAValue};
use pecos_phir::phir::{AttributeValue, Block, Instruction, Module};
use pecos_phir::types::{FunctionType, Type};

use crate::ast::Operation as QasmOp;
use crate::parser::{Program, QASMParser};

/// Convert a QASM string to a PHIR Module.
///
/// # Errors
///
/// Returns an error if parsing or conversion fails.
pub fn qasm_to_phir_module(qasm_str: &str) -> Result<Module> {
    let program = QASMParser::parse_str(qasm_str)
        .map_err(|e| pecos_phir::PhirError::internal(format!("QASM parse error: {e}")))?;
    qasm_program_to_phir_module(&program)
}

/// Convert a parsed QASM `Program` to a PHIR Module.
///
/// # Errors
///
/// Returns an error if the program contains unsupported operations.
pub fn qasm_program_to_phir_module(program: &Program) -> Result<Module> {
    let mut converter = Converter::new();
    converter.convert(program)
}

/// Convert a QASM string to RON text (for debugging / round-trip testing).
///
/// # Errors
///
/// Returns an error if parsing, conversion, or serialization fails.
pub fn qasm_to_ron(qasm_str: &str) -> Result<String> {
    let module = qasm_to_phir_module(qasm_str)?;
    pecos_phir::to_ron(&module)
}

/// Internal converter state.
struct Converter {
    next_ssa: u32,
    /// Deferred measurements: (`qubit_ssa`, `classical_reg_name`, `bit_index`)
    deferred_measurements: Vec<(SSAValue, String, usize)>,
}

impl Converter {
    fn new() -> Self {
        Self {
            next_ssa: 0,
            deferred_measurements: Vec::new(),
        }
    }

    fn new_ssa(&mut self) -> SSAValue {
        let ssa = SSAValue::new(self.next_ssa);
        self.next_ssa += 1;
        ssa
    }

    fn convert(&mut self, program: &Program) -> Result<Module> {
        let mut module = Module::new("qasm_module");

        // Build main function
        let func_type = FunctionType {
            inputs: vec![],
            outputs: vec![],
            variadic: false,
        };
        let mut func = FuncOp::new("main", func_type);

        let block = func
            .entry_region_mut()
            .and_then(|r| r.entry_block_mut())
            .ok_or_else(|| pecos_phir::PhirError::internal("No entry block"))?;

        // 1) Emit VarDefine for each quantum register
        for (name, qubit_ids) in &program.quantum_registers {
            let var_def = VarDefineOp::new(name.clone(), "qubits".to_string(), qubit_ids.len());
            block.add_instruction(Instruction::new(
                Operation::Builtin(BuiltinOp::VarDefine(var_def)),
                vec![],
                vec![],
                vec![],
            ));
        }

        // 2) Emit VarDefine for each classical register
        for (name, size) in &program.classical_registers {
            let var_def = VarDefineOp::new(name.clone(), "i64".to_string(), *size);
            block.add_instruction(Instruction::new(
                Operation::Builtin(BuiltinOp::VarDefine(var_def)),
                vec![],
                vec![],
                vec![],
            ));
        }

        // 3) Assign SSA values to each qubit (VarDefine handles allocation)
        let mut qubit_ssa: Vec<SSAValue> = Vec::with_capacity(program.total_qubits);
        for _ in 0..program.total_qubits {
            qubit_ssa.push(self.new_ssa());
        }

        // 4) Convert operations (measurements are deferred)
        for op in &program.operations {
            self.convert_operation(op, block, &qubit_ssa)?;
        }

        // 5) Emit all deferred measurements at the end so they land in one
        //    PhirEngine batch (avoids the measurement-mapping indexing bug).
        let measurements = std::mem::take(&mut self.deferred_measurements);
        self.emit_measurements(block, &measurements)?;

        module.add_function(func);
        Ok(module)
    }

    fn convert_operation(
        &mut self,
        op: &QasmOp,
        block: &mut Block,
        qubit_ssa: &[SSAValue],
    ) -> Result<()> {
        match op {
            QasmOp::Gate {
                name,
                parameters,
                qubits,
            } => {
                let quantum_op = gate_name_to_quantum_op(name, parameters)?;
                let operands: Vec<SSAValue> = qubits.iter().map(|&q| qubit_ssa[q]).collect();
                let results: Vec<SSAValue> = operands.iter().map(|_| self.new_ssa()).collect();
                let result_types = vec![Type::Qubit; results.len()];

                block.add_instruction(Instruction::new(
                    Operation::Quantum(quantum_op),
                    operands,
                    results,
                    result_types,
                ));
            }

            QasmOp::NativeGate(gate) => {
                let quantum_op = gate_type_to_quantum_op(gate.gate_type, &gate.params)?;
                let operands: Vec<SSAValue> = gate.qubits.iter().map(|q| qubit_ssa[q.0]).collect();
                let results: Vec<SSAValue> = operands.iter().map(|_| self.new_ssa()).collect();
                let result_types = vec![Type::Qubit; results.len()];

                if gate.gate_type == GateType::PZ {
                    // Prep/reset: emit Reset instead of a gate
                    block.add_instruction(Instruction::new(
                        Operation::Quantum(QuantumOp::Reset),
                        operands,
                        results,
                        result_types,
                    ));
                } else {
                    block.add_instruction(Instruction::new(
                        Operation::Quantum(quantum_op),
                        operands,
                        results,
                        result_types,
                    ));
                }
            }

            QasmOp::MeasureWithMapping {
                gate,
                c_reg,
                c_index,
            } => {
                if let Some(qubit) = gate.qubits.first() {
                    self.deferred_measurements
                        .push((qubit_ssa[qubit.0], c_reg.clone(), *c_index));
                }
            }

            QasmOp::RegMeasure { q_reg, c_reg } => {
                if let Some(qubit_ids) = program_qreg_ids(q_reg) {
                    // We don't have access to the program here, so RegMeasure
                    // should have been expanded by the parser already.
                    // If we encounter it, return an error.
                    let _ = qubit_ids;
                }
                return Err(pecos_phir::PhirError::internal(format!(
                    "RegMeasure should be expanded by the parser: {q_reg} -> {c_reg}"
                )));
            }

            QasmOp::If { .. } => {
                return Err(pecos_phir::PhirError::internal(
                    "Conditional (if) operations are not yet supported in QASM-to-PHIR conversion",
                ));
            }

            QasmOp::Barrier { .. }
            | QasmOp::ClassicalAssignment { .. }
            | QasmOp::VoidFunctionCall { .. }
            | QasmOp::OpaqueGate { .. } => {
                // Skip barriers, classical-only, and opaque operations
            }
        }
        Ok(())
    }

    /// Emit Measure + Bitcast + Shl + Or + Result instructions for all
    /// deferred measurements, grouped by classical register.
    fn emit_measurements(
        &mut self,
        block: &mut Block,
        measurements: &[(SSAValue, String, usize)],
    ) -> Result<()> {
        if measurements.is_empty() {
            return Ok(());
        }

        // Step 1: Emit all Measure instructions first
        let mut measure_results: Vec<(SSAValue, String, usize)> =
            Vec::with_capacity(measurements.len());

        for (qubit_ssa, reg_name, bit_idx) in measurements {
            let meas_id = self.new_ssa();
            block.add_instruction(Instruction::new(
                Operation::Quantum(QuantumOp::Measure),
                vec![*qubit_ssa],
                vec![meas_id],
                vec![Type::Bit],
            ));
            measure_results.push((meas_id, reg_name.clone(), *bit_idx));
        }

        // Step 2: Group by register and combine bits
        // Collect unique register names in order
        let mut reg_order: Vec<String> = Vec::new();
        for (_, name, _) in &measure_results {
            if !reg_order.contains(name) {
                reg_order.push(name.clone());
            }
        }

        for reg_name in &reg_order {
            // Gather bits for this register
            let bits: Vec<(SSAValue, usize)> = measure_results
                .iter()
                .filter(|(_, n, _)| n == reg_name)
                .map(|(ssa, _, idx)| (*ssa, *idx))
                .collect();

            // Start with ConstInt(0)
            let zero_ssa = self.new_ssa();
            block.add_instruction(Instruction::new(
                Operation::Classical(ClassicalOp::ConstInt(0)),
                vec![],
                vec![zero_ssa],
                vec![Type::Int(pecos_phir::types::IntWidth::I64)],
            ));

            let mut accum = zero_ssa;

            for (meas_ssa, bit_idx) in &bits {
                // Bitcast measurement bit to i64
                let cast_ssa = self.new_ssa();
                block.add_instruction(Instruction::new(
                    Operation::Classical(ClassicalOp::Bitcast),
                    vec![*meas_ssa],
                    vec![cast_ssa],
                    vec![Type::Int(pecos_phir::types::IntWidth::I64)],
                ));

                // Shift left by bit_idx
                let shifted_ssa = self.new_ssa();
                let shift_amount = u32::try_from(*bit_idx)
                    .map_err(|_| pecos_phir::PhirError::internal("bit index too large"))?;
                block.add_instruction(Instruction::new(
                    Operation::Classical(ClassicalOp::Shl(shift_amount)),
                    vec![cast_ssa],
                    vec![shifted_ssa],
                    vec![Type::Int(pecos_phir::types::IntWidth::I64)],
                ));

                // Or with accumulator
                let or_ssa = self.new_ssa();
                block.add_instruction(Instruction::new(
                    Operation::Classical(ClassicalOp::Or),
                    vec![accum, shifted_ssa],
                    vec![or_ssa],
                    vec![Type::Int(pecos_phir::types::IntWidth::I64)],
                ));

                accum = or_ssa;
            }

            // Emit Result instruction with export_name attribute
            let result_ssa = self.new_ssa();
            let mut result_instr = Instruction::new(
                Operation::Classical(ClassicalOp::Result),
                vec![accum],
                vec![result_ssa],
                vec![Type::Int(pecos_phir::types::IntWidth::I64)],
            );
            result_instr.attributes.insert(
                "export_name".to_string(),
                AttributeValue::String(reg_name.clone()),
            );
            block.add_instruction(result_instr);
        }

        Ok(())
    }
}

/// Map a QASM gate name (string) + parameters to a PHIR `QuantumOp`.
fn gate_name_to_quantum_op(name: &str, params: &[f64]) -> Result<QuantumOp> {
    match name.to_lowercase().as_str() {
        "h" => Ok(QuantumOp::H),
        "x" => Ok(QuantumOp::X),
        "y" => Ok(QuantumOp::Y),
        "z" => Ok(QuantumOp::Z),
        "s" => Ok(QuantumOp::S),
        "sdg" => Ok(QuantumOp::Sdg),
        "t" => Ok(QuantumOp::T),
        "tdg" => Ok(QuantumOp::Tdg),
        "cx" | "cnot" => Ok(QuantumOp::CX),
        "cz" => Ok(QuantumOp::CZ),
        "swap" => Ok(QuantumOp::SWAP),
        "rx" => Ok(QuantumOp::RX(angle_param(params, 0))),
        "ry" => Ok(QuantumOp::RY(angle_param(params, 0))),
        "rz" => Ok(QuantumOp::RZ(angle_param(params, 0))),
        "rzz" => Ok(QuantumOp::RZZ(angle_param(params, 0))),
        "r1xy" => Ok(QuantumOp::R1XY(
            angle_param(params, 0),
            angle_param(params, 1),
        )),
        "u" | "u3" => Ok(QuantumOp::U3(
            angle_param(params, 0),
            angle_param(params, 1),
            angle_param(params, 2),
        )),
        "reset" => Ok(QuantumOp::Reset),
        _ => Err(pecos_phir::PhirError::internal(format!(
            "Unsupported gate: {name}"
        ))),
    }
}

/// Map a `GateType` enum + angles to a PHIR `QuantumOp`.
fn gate_type_to_quantum_op(gate_type: GateType, params: &[f64]) -> Result<QuantumOp> {
    match gate_type {
        GateType::H => Ok(QuantumOp::H),
        GateType::X => Ok(QuantumOp::X),
        GateType::Y => Ok(QuantumOp::Y),
        GateType::Z => Ok(QuantumOp::Z),
        GateType::SZ => Ok(QuantumOp::S),
        GateType::SZdg => Ok(QuantumOp::Sdg),
        GateType::T => Ok(QuantumOp::T),
        GateType::Tdg => Ok(QuantumOp::Tdg),
        GateType::CX => Ok(QuantumOp::CX),
        GateType::CZ => Ok(QuantumOp::CZ),
        GateType::RX => Ok(QuantumOp::RX(angle_param(params, 0))),
        GateType::RY => Ok(QuantumOp::RY(angle_param(params, 0))),
        GateType::RZ => Ok(QuantumOp::RZ(angle_param(params, 0))),
        GateType::RZZ => Ok(QuantumOp::RZZ(angle_param(params, 0))),
        GateType::R1XY => Ok(QuantumOp::R1XY(
            angle_param(params, 0),
            angle_param(params, 1),
        )),
        GateType::MZ => Ok(QuantumOp::Measure),
        GateType::PZ => Ok(QuantumOp::Reset),
        _ => Err(pecos_phir::PhirError::internal(format!(
            "Unsupported gate type: {gate_type:?}"
        ))),
    }
}

// Stub -- RegMeasure is expected to be expanded by the parser.
fn program_qreg_ids(_name: &str) -> Option<Vec<usize>> {
    None
}

/// Extract a radians parameter from a slice and convert to `Angle64`.
fn angle_param(params: &[f64], index: usize) -> Angle64 {
    Angle64::from_radians(params.get(index).copied().unwrap_or(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_and_convert(qasm: &str) -> Module {
        qasm_to_phir_module(qasm).expect("conversion should succeed")
    }

    fn get_main_block(module: &Module) -> &Block {
        let block = module
            .body
            .blocks
            .first()
            .expect("module should have a block");
        let func_instr = block.operations.first().expect("should have a function");
        if let Operation::Builtin(BuiltinOp::Func(func)) = &func_instr.operation {
            func.body
                .first()
                .and_then(|r| r.blocks.first())
                .expect("function should have entry block")
        } else {
            panic!("first operation should be a function");
        }
    }

    #[test]
    fn single_h_conversion() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            h q[0];
            measure q[0] -> c[0];
        "#;
        let module = parse_and_convert(qasm);
        let block = get_main_block(&module);

        // Should have: VarDefine(q), VarDefine(c), Alloc, H, Measure, ConstInt, Bitcast, Shl, Or, Result
        let ops: Vec<String> = block
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        assert!(
            ops.contains(&"quantum.h".to_string()),
            "should contain H gate: {ops:?}"
        );
        assert!(
            ops.contains(&"quantum.measure".to_string()),
            "should contain Measure: {ops:?}"
        );
        assert!(
            ops.contains(&"arith.result".to_string()),
            "should contain Result: {ops:?}"
        );
    }

    #[test]
    fn bell_state_conversion() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
            h q[0];
            cx q[0], q[1];
            measure q[0] -> c[0];
            measure q[1] -> c[1];
        "#;
        let module = parse_and_convert(qasm);
        let block = get_main_block(&module);

        let ops: Vec<String> = block
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        assert!(ops.contains(&"quantum.h".to_string()));
        assert!(ops.contains(&"quantum.cx".to_string()));
        assert!(ops.contains(&"quantum.measure".to_string()));
    }

    #[test]
    fn rz_conversion() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            rz(1.5707963267948966) q[0];
        "#;
        let module = parse_and_convert(qasm);
        let block = get_main_block(&module);

        let has_rz = block
            .operations
            .iter()
            .any(|i| matches!(&i.operation, Operation::Quantum(QuantumOp::RZ(_))));
        assert!(has_rz, "should contain RZ gate");
    }

    #[test]
    fn qasm_to_ron_roundtrip() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            h q[0];
            measure q[0] -> c[0];
        "#;
        let ron_str = qasm_to_ron(qasm).expect("should produce RON");
        let module: Module = pecos_phir::from_ron(&ron_str).expect("should parse RON");
        assert_eq!(module.name, "qasm_module");
    }

    #[test]
    fn var_define_emitted() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg c[2];
        "#;
        let module = parse_and_convert(qasm);
        let block = get_main_block(&module);

        let var_defs: Vec<&VarDefineOp> = block
            .operations
            .iter()
            .filter_map(|i| {
                if let Operation::Builtin(BuiltinOp::VarDefine(v)) = &i.operation {
                    Some(v)
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(var_defs.len(), 2, "should have 2 VarDefine ops");
        let names: Vec<&str> = var_defs.iter().map(|v| v.name.as_str()).collect();
        assert!(names.contains(&"q"));
        assert!(names.contains(&"c"));
    }

    #[test]
    fn no_alloc_emitted() {
        // VarDefine handles qubit allocation; no explicit Alloc instructions
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[3];
        "#;
        let module = parse_and_convert(qasm);
        let block = get_main_block(&module);

        let alloc_count = block
            .operations
            .iter()
            .filter(|i| matches!(&i.operation, Operation::Quantum(QuantumOp::Alloc)))
            .count();
        assert_eq!(
            alloc_count, 0,
            "should have no Alloc ops (VarDefine handles allocation)"
        );
    }

    #[test]
    fn result_ops_emitted() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            creg a[1];
            creg b[1];
            measure q[0] -> a[0];
            measure q[1] -> b[0];
        "#;
        let module = parse_and_convert(qasm);
        let block = get_main_block(&module);

        let result_count = block
            .operations
            .iter()
            .filter(|i| matches!(&i.operation, Operation::Classical(ClassicalOp::Result)))
            .count();
        assert_eq!(result_count, 2, "should have 1 Result per register");
    }

    #[test]
    fn empty_qasm_conversion() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
        "#;
        let module = parse_and_convert(qasm);
        assert_eq!(module.name, "qasm_module");
    }

    #[test]
    fn unsupported_if_returns_error() {
        // We can't easily construct an If operation via QASM text since the parser
        // expands them, but we can test the converter directly.
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];
            measure q[0] -> c[0];
            if(c==1) x q[0];
        "#;
        let result = qasm_to_phir_module(qasm);
        assert!(result.is_err(), "if-statements should return an error");
    }

    #[test]
    fn reset_conversion() {
        let qasm = r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            reset q[0];
        "#;
        let module = parse_and_convert(qasm);
        let block = get_main_block(&module);

        let has_reset = block
            .operations
            .iter()
            .any(|i| matches!(&i.operation, Operation::Quantum(QuantumOp::Reset)));
        assert!(has_reset, "should contain Reset");
    }
}
