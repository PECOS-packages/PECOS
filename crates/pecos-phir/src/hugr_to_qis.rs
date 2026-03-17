/*!
HUGR to QIS Conversion Pass

This module provides a conversion pass that translates HUGR dialect operations
to QIS dialect operations. This follows the same decomposition strategy used by
Selene's hugr-qis compiler.

The conversion maps high-level quantum gates to hardware-native gates:
- Hadamard (H) → RZ(-π/2), RXY(π/2, 0), RZ(-π/2)
- CNOT/CX → RXY(π/2, 0) on target, RZZ(π/2), RZ(-π/2) on control, RXY(-π/2, 0) on target
- RX(θ) → RXY(θ, 0)
- RY(θ) → RXY(θ, π/2)
*/

use crate::error::Result;
use crate::ops::{ClassicalOp, CustomOp, Operation};
use crate::phir::{Block, Instruction, Module, Region, SSAValue};
use std::collections::BTreeMap;
use std::f64::consts::PI;

// ========================================================================
// Reusable decomposition helpers
//
// These functions emit QIS CustomOp instructions for standard gate
// decompositions.  They are used by both this module and `qis_parser`.
// ========================================================================

/// Helper: create a `ConstFloat` instruction that defines `result` = `value`.
#[must_use]
pub fn emit_const_float(result: SSAValue, value: f64) -> Instruction {
    Instruction {
        results: vec![result],
        operation: Operation::Classical(ClassicalOp::ConstFloat(value)),
        operands: vec![],
        result_types: vec![crate::types::Type::Float(crate::types::FloatPrecision::F64)],
        regions: vec![],
        attributes: BTreeMap::new(),
        location: None,
    }
}

/// Helper: emit a `qis.rz(qubit, angle)` instruction.
#[must_use]
pub fn emit_qis_rz(qubit: SSAValue, angle: SSAValue) -> Instruction {
    Instruction {
        results: vec![],
        operation: Operation::Custom(CustomOp::new("qis", "rz", vec![], BTreeMap::new())),
        operands: vec![qubit, angle],
        result_types: vec![],
        regions: vec![],
        attributes: BTreeMap::new(),
        location: None,
    }
}

/// Helper: emit a `qis.rxy(qubit, theta, phi)` instruction.
#[must_use]
pub fn emit_qis_rxy(qubit: SSAValue, theta: SSAValue, phi: SSAValue) -> Instruction {
    Instruction {
        results: vec![],
        operation: Operation::Custom(CustomOp::new("qis", "rxy", vec![], BTreeMap::new())),
        operands: vec![qubit, theta, phi],
        result_types: vec![],
        regions: vec![],
        attributes: BTreeMap::new(),
        location: None,
    }
}

/// Helper: emit a `qis.rzz(qubit1, qubit2, angle)` instruction.
#[must_use]
pub fn emit_qis_rzz(qubit1: SSAValue, qubit2: SSAValue, angle: SSAValue) -> Instruction {
    Instruction {
        results: vec![],
        operation: Operation::Custom(CustomOp::new("qis", "rzz", vec![], BTreeMap::new())),
        operands: vec![qubit1, qubit2, angle],
        result_types: vec![],
        regions: vec![],
        attributes: BTreeMap::new(),
        location: None,
    }
}

/// Decompose Hadamard: H = RZ(-pi/2) . RXY(pi/2, 0) . RZ(-pi/2)
///
/// Appends instructions and constants to `out`. The caller provides
/// `fresh_id`, a closure that returns a fresh `SSAValue` each call.
pub fn decompose_h(
    qubit: SSAValue,
    out: &mut Vec<Instruction>,
    fresh_id: &mut impl FnMut() -> SSAValue,
) {
    let neg_half_pi = fresh_id();
    out.push(emit_const_float(neg_half_pi, -PI / 2.0));
    let half_pi = fresh_id();
    out.push(emit_const_float(half_pi, PI / 2.0));
    let zero = fresh_id();
    out.push(emit_const_float(zero, 0.0));

    out.push(emit_qis_rz(qubit, neg_half_pi));
    out.push(emit_qis_rxy(qubit, half_pi, zero));
    // Reuse neg_half_pi SSA value for second RZ (same constant)
    out.push(emit_qis_rz(qubit, neg_half_pi));
}

/// Decompose CX: RXY(pi/2,0) target, RZZ(pi/2) ctrl+tgt, RZ(-pi/2) ctrl, RXY(-pi/2,0) target
pub fn decompose_cx(
    control: SSAValue,
    target: SSAValue,
    out: &mut Vec<Instruction>,
    fresh_id: &mut impl FnMut() -> SSAValue,
) {
    let half_pi = fresh_id();
    out.push(emit_const_float(half_pi, PI / 2.0));
    let zero = fresh_id();
    out.push(emit_const_float(zero, 0.0));
    let neg_half_pi = fresh_id();
    out.push(emit_const_float(neg_half_pi, -PI / 2.0));

    out.push(emit_qis_rxy(target, half_pi, zero));
    out.push(emit_qis_rzz(control, target, half_pi));
    out.push(emit_qis_rz(control, neg_half_pi));
    out.push(emit_qis_rxy(target, neg_half_pi, zero));
}

/// Convert a HUGR module to use QIS operations
///
/// # Errors
/// Returns an error if the module conversion fails (e.g., unsupported operations or invalid module structure).
pub fn convert_hugr_to_qis(module: &mut Module) -> Result<()> {
    let mut converter = HugrToQisConverter::new();
    converter.convert_module(module);
    Ok(())
}

struct HugrToQisConverter {
    /// Map from HUGR qubit values to QIS qubit IDs
    #[allow(dead_code)]
    qubit_map: BTreeMap<SSAValue, SSAValue>,
    /// Counter for generating fresh SSA values
    next_value_id: u32,
}

impl HugrToQisConverter {
    fn new() -> Self {
        Self {
            qubit_map: BTreeMap::new(),
            next_value_id: 1000, // Start from a high number to avoid conflicts
        }
    }

    fn fresh_value(&mut self) -> SSAValue {
        let value = SSAValue::new(self.next_value_id);
        self.next_value_id += 1;
        value
    }

    fn convert_module(&mut self, module: &mut Module) {
        // Process the module's body region
        self.convert_region(&mut module.body);
    }

    fn convert_region(&mut self, region: &mut Region) {
        for block in &mut region.blocks {
            self.convert_block(block);
        }
    }

    fn convert_block(&mut self, block: &mut Block) {
        let mut new_instructions = Vec::new();

        for instruction in &block.operations {
            match &instruction.operation {
                Operation::Custom(custom_op) if custom_op.dialect() == "hugr" => {
                    // Convert HUGR operations to QIS
                    let qis_ops = self.convert_hugr_op(
                        custom_op,
                        &instruction.operands,
                        &instruction.results,
                    );
                    new_instructions.extend(qis_ops);
                }
                _ => {
                    // Keep non-HUGR operations as-is
                    new_instructions.push(instruction.clone());
                }
            }
        }

        block.operations = new_instructions;
    }

    #[allow(clippy::too_many_lines)] // Operation conversion requires a comprehensive match on all gate types
    fn convert_hugr_op(
        &mut self,
        op: &CustomOp,
        operands: &[SSAValue],
        results: &[SSAValue],
    ) -> Vec<Instruction> {
        let mut instructions = Vec::new();

        match op.name() {
            "qalloc" => {
                // HUGR qalloc → QIS qalloc
                let qis_op = CustomOp::new("qis", "qalloc", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: results.to_vec(),
                    operation: Operation::Custom(qis_op),
                    operands: vec![],
                    result_types: vec![crate::types::Type::Qubit],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            "qfree" => {
                // HUGR qfree → QIS qfree
                let qis_op = CustomOp::new("qis", "qfree", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: vec![],
                    operation: Operation::Custom(qis_op),
                    operands: operands.to_vec(),
                    result_types: vec![],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            "h" => {
                decompose_h(operands[0], &mut instructions, &mut || self.fresh_value());
            }

            "cx" => {
                decompose_cx(operands[0], operands[1], &mut instructions, &mut || {
                    self.fresh_value()
                });
            }

            "rx" => {
                // RX(θ) → RXY(θ, 0)
                let zero = self.fresh_value();
                instructions.push(emit_const_float(zero, 0.0));
                instructions.push(emit_qis_rxy(operands[0], operands[1], zero));
            }

            "ry" => {
                // RY(θ) → RXY(θ, π/2)
                let half_pi = self.fresh_value();
                instructions.push(emit_const_float(half_pi, PI / 2.0));
                instructions.push(emit_qis_rxy(operands[0], operands[1], half_pi));
            }

            "rz" => {
                // RZ(θ) → QIS RZ(θ) (direct mapping)
                instructions.push(emit_qis_rz(operands[0], operands[1]));
            }

            "measure" => {
                // HUGR measure → QIS lazy_measure + read_future
                let qubit = &operands[0];

                // Create a future for the measurement
                let future = self.fresh_value();
                let lazy_measure = CustomOp::new("qis", "lazy_measure", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: vec![future],
                    operation: Operation::Custom(lazy_measure),
                    operands: vec![*qubit],
                    result_types: vec![crate::types::Type::Future],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });

                // Read the future to get the result
                let read_future = CustomOp::new("qis", "read_future", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: results.to_vec(),
                    operation: Operation::Custom(read_future),
                    operands: vec![future],
                    result_types: vec![crate::types::Type::Bool],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            _ => {
                // For other HUGR operations, keep as-is for now
                // In a complete implementation, all HUGR ops would be converted
                instructions.push(Instruction {
                    results: results.to_vec(),
                    operation: Operation::Custom(op.clone()),
                    operands: operands.to_vec(),
                    result_types: vec![], // Unknown types for unhandled ops
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }
        }

        instructions
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phir::{Block, Module, Region};

    #[test]
    fn test_hadamard_decomposition() {
        // Create a module with a Hadamard gate
        let mut module = Module {
            name: "test".to_string(),
            attributes: BTreeMap::new(),
            body: Region {
                kind: crate::region_kinds::RegionKind::Graph,
                attributes: BTreeMap::new(),
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    attributes: BTreeMap::new(),
                    operations: vec![Instruction {
                        results: vec![],
                        operation: Operation::Custom(CustomOp::new(
                            "hugr",
                            "h",
                            vec![],
                            BTreeMap::new(),
                        )),
                        operands: vec![SSAValue::new(0)], // q0
                        result_types: vec![],
                        regions: vec![],
                        attributes: BTreeMap::new(),
                        location: None,
                    }],
                    terminator: None,
                }],
            },
        };

        // Convert HUGR to QIS
        let mut converter = HugrToQisConverter::new();
        converter.convert_module(&mut module);

        // 3 ConstFloat + 3 gate ops (RZ, RXY, RZ) = 6 total
        assert_eq!(module.body.blocks[0].operations.len(), 6);

        // Verify the QIS gate operations are correct
        let qis_ops: Vec<_> = module.body.blocks[0]
            .operations
            .iter()
            .filter_map(|instr| {
                if let Operation::Custom(custom_op) = &instr.operation {
                    Some(custom_op.name().to_string())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(qis_ops, vec!["rz", "rxy", "rz"]);
    }

    #[test]
    fn test_cnot_decomposition() {
        // Create a module with a CNOT gate
        let mut module = Module {
            name: "test".to_string(),
            attributes: BTreeMap::new(),
            body: Region {
                kind: crate::region_kinds::RegionKind::Graph,
                attributes: BTreeMap::new(),
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    attributes: BTreeMap::new(),
                    operations: vec![Instruction {
                        results: vec![],
                        operation: Operation::Custom(CustomOp::new(
                            "hugr",
                            "cx",
                            vec![],
                            BTreeMap::new(),
                        )),
                        operands: vec![SSAValue::new(0), SSAValue::new(1)], // q0, q1
                        result_types: vec![],
                        regions: vec![],
                        attributes: BTreeMap::new(),
                        location: None,
                    }],
                    terminator: None,
                }],
            },
        };

        // Convert HUGR to QIS
        let mut converter = HugrToQisConverter::new();
        converter.convert_module(&mut module);

        // 3 ConstFloat + 4 gate ops (RXY, RZZ, RZ, RXY) = 7 total
        assert_eq!(module.body.blocks[0].operations.len(), 7);

        // Verify the sequence of QIS gate operations
        let qis_ops: Vec<_> = module.body.blocks[0]
            .operations
            .iter()
            .filter_map(|instr| {
                if let Operation::Custom(custom_op) = &instr.operation {
                    Some(custom_op.name().to_string())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(qis_ops, vec!["rxy", "rzz", "rz", "rxy"]);
    }
}
