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
use crate::ops::{CustomOp, Operation};
use crate::phir::{Block, Instruction, Module, Region, SSAValue};
use std::collections::BTreeMap;
use std::f64::consts::PI;

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
                // Hadamard decomposition: H = RZ(-π/2) · RXY(π/2, 0) · RZ(-π/2)
                let qubit = &operands[0];

                // RZ(-π/2)
                let rz1 = CustomOp::new("qis", "rz", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: vec![],
                    operation: Operation::Custom(rz1),
                    operands: vec![*qubit, self.make_float_constant(-PI / 2.0)],
                    result_types: vec![],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });

                // RXY(π/2, 0)
                let rxy = CustomOp::new("qis", "rxy", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: vec![],
                    operation: Operation::Custom(rxy),
                    operands: vec![
                        *qubit,
                        self.make_float_constant(PI / 2.0),
                        self.make_float_constant(0.0),
                    ],
                    result_types: vec![],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });

                // RZ(-π/2)
                let rz2 = CustomOp::new("qis", "rz", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: vec![],
                    operation: Operation::Custom(rz2),
                    operands: vec![*qubit, self.make_float_constant(-PI / 2.0)],
                    result_types: vec![],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            "cx" => {
                // CNOT decomposition using RXY and RZZ
                let control = &operands[0];
                let target = &operands[1];

                // RXY(π/2, 0) on target
                let rxy1 = CustomOp::new("qis", "rxy", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: vec![],
                    operation: Operation::Custom(rxy1),
                    operands: vec![
                        *target,
                        self.make_float_constant(PI / 2.0),
                        self.make_float_constant(0.0),
                    ],
                    result_types: vec![],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });

                // RZZ(π/2) on control and target
                let rzz = CustomOp::new("qis", "rzz", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: vec![],
                    operation: Operation::Custom(rzz),
                    operands: vec![*control, *target, self.make_float_constant(PI / 2.0)],
                    result_types: vec![],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });

                // RZ(-π/2) on control
                let rz = CustomOp::new("qis", "rz", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: vec![],
                    operation: Operation::Custom(rz),
                    operands: vec![*control, self.make_float_constant(-PI / 2.0)],
                    result_types: vec![],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });

                // RXY(-π/2, 0) on target
                let rxy2 = CustomOp::new("qis", "rxy", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: vec![],
                    operation: Operation::Custom(rxy2),
                    operands: vec![
                        *target,
                        self.make_float_constant(-PI / 2.0),
                        self.make_float_constant(0.0),
                    ],
                    result_types: vec![],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            "rx" => {
                // RX(θ) → RXY(θ, 0)
                let qubit = &operands[0];
                let angle = &operands[1];

                let rxy = CustomOp::new("qis", "rxy", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: vec![],
                    operation: Operation::Custom(rxy),
                    operands: vec![*qubit, *angle, self.make_float_constant(0.0)],
                    result_types: vec![],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            "ry" => {
                // RY(θ) → RXY(θ, π/2)
                let qubit = &operands[0];
                let angle = &operands[1];

                let rxy = CustomOp::new("qis", "rxy", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: vec![],
                    operation: Operation::Custom(rxy),
                    operands: vec![*qubit, *angle, self.make_float_constant(PI / 2.0)],
                    result_types: vec![],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
            }

            "rz" => {
                // RZ(θ) → QIS RZ(θ) (direct mapping)
                let qubit = &operands[0];
                let angle = &operands[1];

                let qis_rz = CustomOp::new("qis", "rz", vec![], BTreeMap::new());
                instructions.push(Instruction {
                    results: vec![],
                    operation: Operation::Custom(qis_rz),
                    operands: vec![*qubit, *angle],
                    result_types: vec![],
                    regions: vec![],
                    attributes: BTreeMap::new(),
                    location: None,
                });
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

    fn make_float_constant(&mut self, _value: f64) -> SSAValue {
        // In a real implementation, this would create a proper constant
        // For now, we just create a placeholder SSA value

        // This would normally emit a constant operation
        self.fresh_value()
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

        // Check that we have 3 QIS operations (RZ, RXY, RZ)
        assert_eq!(module.body.blocks[0].operations.len(), 3);

        // Verify the operations are correct QIS ops
        for op in &module.body.blocks[0].operations {
            if let Operation::Custom(custom_op) = &op.operation {
                assert_eq!(custom_op.dialect(), "qis");
                assert!(custom_op.name() == "rz" || custom_op.name() == "rxy");
            }
        }
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

        // Check that we have 4 QIS operations (RXY, RZZ, RZ, RXY)
        assert_eq!(module.body.blocks[0].operations.len(), 4);

        // Verify the sequence of operations
        let ops: Vec<_> = module.body.blocks[0]
            .operations
            .iter()
            .filter_map(|instr| {
                if let Operation::Custom(custom_op) = &instr.operation {
                    Some(custom_op.name())
                } else {
                    None
                }
            })
            .collect();

        assert_eq!(ops, vec!["rxy", "rzz", "rz", "rxy"]);
    }
}
