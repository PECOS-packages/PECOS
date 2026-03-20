/*!
QIS to Quantum Conversion Pass

Lowers QIS dialect `CustomOps` to standard PHIR `QuantumOps`:
- `qis.qalloc` -> `QuantumOp::Alloc`
- `qis.qfree` -> `QuantumOp::Dealloc`
- `qis.reset` -> `QuantumOp::Reset`
- `qis.rxy` -> `QuantumOp::R1XY(theta, phi)`
- `qis.rz` -> `QuantumOp::RZ(angle)`
- `qis.rzz` -> `QuantumOp::RZZ(angle)`
- `qis.measure` / `qis.lazy_measure` -> `QuantumOp::Measure`
- `qis.read_future` -> elided (results forwarded)

Angle values are resolved from `ClassicalOp::ConstFloat` instructions via an
SSA constant map built in a preliminary scan.
*/

use crate::error::{PhirError, Result};
use crate::ops::{ClassicalOp, CustomOp, Operation, QuantumOp};
use crate::phir::{Block, Instruction, Module, Region, SSAValue};
use pecos_core::Angle64;
use std::collections::BTreeMap;

/// Convert QIS dialect `CustomOps` in `module` to standard `QuantumOps` in-place.
///
/// # Errors
///
/// Returns an error if an angle operand cannot be resolved to a constant.
pub fn convert_qis_to_quantum(module: &mut Module) -> Result<()> {
    let const_map = build_constant_map(&module.body);
    convert_region(&mut module.body, &const_map)
}

/// Scan a region for `ConstFloat` instructions and map their result SSA values
/// to the float value.
fn build_constant_map(region: &Region) -> BTreeMap<SSAValue, f64> {
    let mut map = BTreeMap::new();
    for block in &region.blocks {
        for instr in &block.operations {
            if let Operation::Classical(ClassicalOp::ConstFloat(val)) = &instr.operation
                && let Some(&result) = instr.results.first()
            {
                map.insert(result, *val);
            }
        }
    }
    map
}

fn convert_region(region: &mut Region, const_map: &BTreeMap<SSAValue, f64>) -> Result<()> {
    for block in &mut region.blocks {
        convert_block(block, const_map)?;
    }
    Ok(())
}

fn convert_block(block: &mut Block, const_map: &BTreeMap<SSAValue, f64>) -> Result<()> {
    let mut new_ops = Vec::with_capacity(block.operations.len());

    for instr in &block.operations {
        match &instr.operation {
            Operation::Custom(custom) if custom.dialect() == "qis" => {
                if let Some(converted) = convert_qis_op(custom, instr, const_map)? {
                    new_ops.push(converted);
                }
                // else: elided (e.g. read_future)
            }
            Operation::Classical(ClassicalOp::ConstFloat(_)) => {
                // Keep ConstFloat instructions -- they may be used by non-quantum code
                // or by QuantumOps that embed angles. We could DCE them later.
                new_ops.push(instr.clone());
            }
            _ => {
                new_ops.push(instr.clone());
            }
        }
    }

    block.operations = new_ops;
    Ok(())
}

fn convert_qis_op(
    custom: &CustomOp,
    instr: &Instruction,
    const_map: &BTreeMap<SSAValue, f64>,
) -> Result<Option<Instruction>> {
    match custom.name() {
        "qalloc" => Ok(Some(Instruction {
            results: instr.results.clone(),
            operation: Operation::Quantum(QuantumOp::Alloc),
            operands: vec![],
            result_types: vec![crate::types::Type::Qubit],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: instr.location.clone(),
        })),

        "qfree" => Ok(Some(Instruction {
            results: vec![],
            operation: Operation::Quantum(QuantumOp::Dealloc),
            operands: instr.operands.clone(),
            result_types: vec![],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: instr.location.clone(),
        })),

        "reset" => Ok(Some(Instruction {
            results: vec![],
            operation: Operation::Quantum(QuantumOp::Reset),
            operands: instr.operands.clone(),
            result_types: vec![],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: instr.location.clone(),
        })),

        "rxy" => {
            // operands: [qubit, theta_ssa, phi_ssa]
            let theta = resolve_angle(&instr.operands, 1, const_map, "rxy theta")?;
            let phi = resolve_angle(&instr.operands, 2, const_map, "rxy phi")?;
            let qubit = instr.operands[0];
            Ok(Some(Instruction {
                results: vec![],
                operation: Operation::Quantum(QuantumOp::R1XY(theta, phi)),
                operands: vec![qubit],
                result_types: vec![],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: instr.location.clone(),
            }))
        }

        "rz" => {
            // operands: [qubit, angle_ssa]
            let angle = resolve_angle(&instr.operands, 1, const_map, "rz angle")?;
            let qubit = instr.operands[0];
            Ok(Some(Instruction {
                results: vec![],
                operation: Operation::Quantum(QuantumOp::RZ(angle)),
                operands: vec![qubit],
                result_types: vec![],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: instr.location.clone(),
            }))
        }

        "rzz" => {
            // operands: [qubit1, qubit2, angle_ssa]
            let angle = resolve_angle(&instr.operands, 2, const_map, "rzz angle")?;
            let q1 = instr.operands[0];
            let q2 = instr.operands[1];
            Ok(Some(Instruction {
                results: vec![],
                operation: Operation::Quantum(QuantumOp::RZZ(angle)),
                operands: vec![q1, q2],
                result_types: vec![],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: instr.location.clone(),
            }))
        }

        "cz" => {
            let q1 = instr.operands[0];
            let q2 = instr.operands[1];
            Ok(Some(Instruction {
                results: vec![],
                operation: Operation::Quantum(QuantumOp::CZ),
                operands: vec![q1, q2],
                result_types: vec![],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: instr.location.clone(),
            }))
        }

        "swap" => {
            let q1 = instr.operands[0];
            let q2 = instr.operands[1];
            Ok(Some(Instruction {
                results: vec![],
                operation: Operation::Quantum(QuantumOp::SWAP),
                operands: vec![q1, q2],
                result_types: vec![],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: instr.location.clone(),
            }))
        }

        "cphase" => {
            let angle = resolve_angle(&instr.operands, 2, const_map, "cphase angle")?;
            let q1 = instr.operands[0];
            let q2 = instr.operands[1];
            Ok(Some(Instruction {
                results: vec![],
                operation: Operation::Quantum(QuantumOp::CPhase(angle)),
                operands: vec![q1, q2],
                result_types: vec![],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: instr.location.clone(),
            }))
        }

        "measure" | "lazy_measure" => {
            let qubit = instr.operands[0];
            Ok(Some(Instruction {
                results: instr.results.clone(),
                operation: Operation::Quantum(QuantumOp::Measure),
                operands: vec![qubit],
                result_types: vec![crate::types::Type::Bool],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: instr.location.clone(),
            }))
        }

        "read_future" => {
            // Elide -- the measurement result is already captured by the
            // measure instruction's result SSA value.
            Ok(None)
        }

        "initialize" => {
            // Runtime init -- elide at this level
            Ok(None)
        }

        other => Err(PhirError::internal(format!(
            "qis_to_quantum: unknown QIS operation: qis.{other}"
        ))),
    }
}

/// Look up the angle at `operands[index]` in the constant map.
fn resolve_angle(
    operands: &[SSAValue],
    index: usize,
    const_map: &BTreeMap<SSAValue, f64>,
    context: &str,
) -> Result<Angle64> {
    let ssa = operands.get(index).ok_or_else(|| {
        PhirError::internal(format!(
            "qis_to_quantum: missing operand {index} for {context}"
        ))
    })?;
    let radians = const_map.get(ssa).copied().ok_or_else(|| {
        PhirError::internal(format!(
            "qis_to_quantum: cannot resolve {context} (SSA {ssa}) to a constant"
        ))
    })?;
    Ok(Angle64::from_radians(radians))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hugr_to_qis::emit_const_float;
    use crate::ops::CustomOp;
    use crate::phir::{Block, Module, Region};
    use std::f64::consts::FRAC_PI_2;

    fn make_module(instructions: Vec<Instruction>) -> Module {
        Module {
            name: "test".to_string(),
            attributes: BTreeMap::new(),
            body: Region {
                kind: crate::region_kinds::RegionKind::Graph,
                attributes: BTreeMap::new(),
                blocks: vec![Block {
                    label: None,
                    arguments: vec![],
                    attributes: BTreeMap::new(),
                    operations: instructions,
                    terminator: None,
                }],
            },
        }
    }

    #[test]
    fn test_qalloc_qfree() {
        let q = SSAValue::new(0);
        let mut module = make_module(vec![
            Instruction {
                results: vec![q],
                operation: Operation::Custom(CustomOp::new(
                    "qis",
                    "qalloc",
                    vec![],
                    BTreeMap::new(),
                )),
                operands: vec![],
                result_types: vec![crate::types::Type::Qubit],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            },
            Instruction {
                results: vec![],
                operation: Operation::Custom(CustomOp::new(
                    "qis",
                    "qfree",
                    vec![],
                    BTreeMap::new(),
                )),
                operands: vec![q],
                result_types: vec![],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            },
        ]);

        convert_qis_to_quantum(&mut module).unwrap();

        let ops: Vec<_> = module.body.blocks[0]
            .operations
            .iter()
            .map(|i| i.operation.name())
            .collect();
        assert_eq!(ops, vec!["quantum.alloc", "quantum.dealloc"]);
    }

    #[test]
    fn test_rz_conversion() {
        let q = SSAValue::new(0);
        let angle_ssa = SSAValue::new(1);
        let mut module = make_module(vec![
            emit_const_float(angle_ssa, FRAC_PI_2),
            Instruction {
                results: vec![],
                operation: Operation::Custom(CustomOp::new("qis", "rz", vec![], BTreeMap::new())),
                operands: vec![q, angle_ssa],
                result_types: vec![],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            },
        ]);

        convert_qis_to_quantum(&mut module).unwrap();

        // Should have ConstFloat + RZ
        let quantum_ops: Vec<_> = module.body.blocks[0]
            .operations
            .iter()
            .filter(|i| matches!(i.operation, Operation::Quantum(_)))
            .collect();
        assert_eq!(quantum_ops.len(), 1);
        assert!(matches!(
            quantum_ops[0].operation,
            Operation::Quantum(QuantumOp::RZ(v)) if v == Angle64::from_radians(FRAC_PI_2)
        ));
    }

    #[test]
    fn test_rxy_conversion() {
        let q = SSAValue::new(0);
        let theta_ssa = SSAValue::new(1);
        let phi_ssa = SSAValue::new(2);
        let mut module = make_module(vec![
            emit_const_float(theta_ssa, FRAC_PI_2),
            emit_const_float(phi_ssa, 0.0),
            Instruction {
                results: vec![],
                operation: Operation::Custom(CustomOp::new("qis", "rxy", vec![], BTreeMap::new())),
                operands: vec![q, theta_ssa, phi_ssa],
                result_types: vec![],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            },
        ]);

        convert_qis_to_quantum(&mut module).unwrap();

        let quantum_ops: Vec<_> = module.body.blocks[0]
            .operations
            .iter()
            .filter(|i| matches!(i.operation, Operation::Quantum(_)))
            .collect();
        assert_eq!(quantum_ops.len(), 1);
        assert!(matches!(
            quantum_ops[0].operation,
            Operation::Quantum(QuantumOp::R1XY(theta, phi))
                if theta == Angle64::from_radians(FRAC_PI_2) && phi == Angle64::ZERO
        ));
    }

    #[test]
    fn test_measure_conversion() {
        let q = SSAValue::new(0);
        let m = SSAValue::new(1);
        let mut module = make_module(vec![Instruction {
            results: vec![m],
            operation: Operation::Custom(CustomOp::new("qis", "measure", vec![], BTreeMap::new())),
            operands: vec![q],
            result_types: vec![crate::types::Type::Bool],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: None,
        }]);

        convert_qis_to_quantum(&mut module).unwrap();
        assert!(matches!(
            module.body.blocks[0].operations[0].operation,
            Operation::Quantum(QuantumOp::Measure)
        ));
    }

    #[test]
    fn test_read_future_elided() {
        let future = SSAValue::new(0);
        let result = SSAValue::new(1);
        let mut module = make_module(vec![Instruction {
            results: vec![result],
            operation: Operation::Custom(CustomOp::new(
                "qis",
                "read_future",
                vec![],
                BTreeMap::new(),
            )),
            operands: vec![future],
            result_types: vec![crate::types::Type::Bool],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: None,
        }]);

        convert_qis_to_quantum(&mut module).unwrap();
        // read_future should be elided
        assert!(module.body.blocks[0].operations.is_empty());
    }

    #[test]
    fn test_full_pipeline_parse_then_convert() {
        // Parse QIS IR, then convert to QuantumOps
        let ir = r"
declare i64 @___qalloc()
declare void @___rz(i64, double)
declare void @___qfree(i64)

define void @main() {
entry:
  %q = call i64 @___qalloc()
  call void @___rz(i64 %q, double 0x3FF921FB54442D18)
  call void @___qfree(i64 %q)
  ret void
}
";
        let mut module = crate::qis_parser::parse_qis_llvm_ir(ir).unwrap();
        convert_qis_to_quantum(&mut module).unwrap();

        let quantum_ops: Vec<_> = module.body.blocks[0]
            .operations
            .iter()
            .filter(|i| matches!(i.operation, Operation::Quantum(_)))
            .map(|i| i.operation.name())
            .collect();
        assert_eq!(
            quantum_ops,
            vec!["quantum.alloc", "quantum.rz", "quantum.dealloc"]
        );
    }

    // ──────────────────────────────────────────────────────────────────
    // RZZ conversion
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_rzz_conversion() {
        let q1 = SSAValue::new(0);
        let q2 = SSAValue::new(1);
        let angle_ssa = SSAValue::new(2);
        let mut module = make_module(vec![
            emit_const_float(angle_ssa, FRAC_PI_2),
            Instruction {
                results: vec![],
                operation: Operation::Custom(CustomOp::new("qis", "rzz", vec![], BTreeMap::new())),
                operands: vec![q1, q2, angle_ssa],
                result_types: vec![],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            },
        ]);

        convert_qis_to_quantum(&mut module).unwrap();

        let quantum_ops: Vec<_> = module.body.blocks[0]
            .operations
            .iter()
            .filter(|i| matches!(i.operation, Operation::Quantum(_)))
            .collect();
        assert_eq!(quantum_ops.len(), 1);
        assert!(matches!(
            quantum_ops[0].operation,
            Operation::Quantum(QuantumOp::RZZ(v)) if v == Angle64::from_radians(FRAC_PI_2)
        ));
        assert_eq!(quantum_ops[0].operands.len(), 2);
    }

    // ──────────────────────────────────────────────────────────────────
    // CZ conversion
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_cz_conversion() {
        let q1 = SSAValue::new(0);
        let q2 = SSAValue::new(1);
        let mut module = make_module(vec![Instruction {
            results: vec![],
            operation: Operation::Custom(CustomOp::new("qis", "cz", vec![], BTreeMap::new())),
            operands: vec![q1, q2],
            result_types: vec![],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: None,
        }]);

        convert_qis_to_quantum(&mut module).unwrap();
        assert!(matches!(
            module.body.blocks[0].operations[0].operation,
            Operation::Quantum(QuantumOp::CZ)
        ));
        assert_eq!(module.body.blocks[0].operations[0].operands.len(), 2);
    }

    // ──────────────────────────────────────────────────────────────────
    // SWAP conversion
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_swap_conversion() {
        let q1 = SSAValue::new(0);
        let q2 = SSAValue::new(1);
        let mut module = make_module(vec![Instruction {
            results: vec![],
            operation: Operation::Custom(CustomOp::new("qis", "swap", vec![], BTreeMap::new())),
            operands: vec![q1, q2],
            result_types: vec![],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: None,
        }]);

        convert_qis_to_quantum(&mut module).unwrap();
        assert!(matches!(
            module.body.blocks[0].operations[0].operation,
            Operation::Quantum(QuantumOp::SWAP)
        ));
        assert_eq!(module.body.blocks[0].operations[0].operands.len(), 2);
    }

    // ──────────────────────────────────────────────────────────────────
    // CPhase conversion
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_cphase_conversion() {
        let q1 = SSAValue::new(0);
        let q2 = SSAValue::new(1);
        let angle_ssa = SSAValue::new(2);
        let mut module = make_module(vec![
            emit_const_float(angle_ssa, std::f64::consts::PI),
            Instruction {
                results: vec![],
                operation: Operation::Custom(CustomOp::new(
                    "qis",
                    "cphase",
                    vec![],
                    BTreeMap::new(),
                )),
                operands: vec![q1, q2, angle_ssa],
                result_types: vec![],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            },
        ]);

        convert_qis_to_quantum(&mut module).unwrap();

        let quantum_ops: Vec<_> = module.body.blocks[0]
            .operations
            .iter()
            .filter(|i| matches!(i.operation, Operation::Quantum(_)))
            .collect();
        assert_eq!(quantum_ops.len(), 1);
        assert!(matches!(
            quantum_ops[0].operation,
            Operation::Quantum(QuantumOp::CPhase(v)) if v == Angle64::from_radians(std::f64::consts::PI)
        ));
        assert_eq!(quantum_ops[0].operands.len(), 2);
    }

    // ──────────────────────────────────────────────────────────────────
    // lazy_measure conversion
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_lazy_measure_conversion() {
        let q = SSAValue::new(0);
        let m = SSAValue::new(1);
        let mut module = make_module(vec![Instruction {
            results: vec![m],
            operation: Operation::Custom(CustomOp::new(
                "qis",
                "lazy_measure",
                vec![],
                BTreeMap::new(),
            )),
            operands: vec![q],
            result_types: vec![crate::types::Type::Bool],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: None,
        }]);

        convert_qis_to_quantum(&mut module).unwrap();
        assert!(matches!(
            module.body.blocks[0].operations[0].operation,
            Operation::Quantum(QuantumOp::Measure)
        ));
        assert_eq!(module.body.blocks[0].operations[0].results, vec![m]);
    }

    // ──────────────────────────────────────────────────────────────────
    // Reset conversion
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_reset_conversion() {
        let q = SSAValue::new(0);
        let mut module = make_module(vec![Instruction {
            results: vec![],
            operation: Operation::Custom(CustomOp::new("qis", "reset", vec![], BTreeMap::new())),
            operands: vec![q],
            result_types: vec![],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: None,
        }]);

        convert_qis_to_quantum(&mut module).unwrap();
        assert!(matches!(
            module.body.blocks[0].operations[0].operation,
            Operation::Quantum(QuantumOp::Reset)
        ));
        assert_eq!(module.body.blocks[0].operations[0].operands, vec![q]);
    }

    // ──────────────────────────────────────────────────────────────────
    // Initialize elision
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_initialize_elided() {
        let mut module = make_module(vec![Instruction {
            results: vec![],
            operation: Operation::Custom(CustomOp::new(
                "qis",
                "initialize",
                vec![],
                BTreeMap::new(),
            )),
            operands: vec![],
            result_types: vec![],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: None,
        }]);

        convert_qis_to_quantum(&mut module).unwrap();
        assert!(module.body.blocks[0].operations.is_empty());
    }

    // ──────────────────────────────────────────────────────────────────
    // Error paths
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_unknown_qis_op_error() {
        let mut module = make_module(vec![Instruction {
            results: vec![],
            operation: Operation::Custom(CustomOp::new(
                "qis",
                "nonexistent_gate",
                vec![],
                BTreeMap::new(),
            )),
            operands: vec![],
            result_types: vec![],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: None,
        }]);

        let err = convert_qis_to_quantum(&mut module).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("nonexistent_gate"),
            "Error should mention the op name: {msg}"
        );
    }

    #[test]
    fn test_missing_angle_operand_error() {
        let q = SSAValue::new(0);
        // rz needs [qubit, angle_ssa] but we only provide [qubit]
        let mut module = make_module(vec![Instruction {
            results: vec![],
            operation: Operation::Custom(CustomOp::new("qis", "rz", vec![], BTreeMap::new())),
            operands: vec![q],
            result_types: vec![],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: None,
        }]);

        let err = convert_qis_to_quantum(&mut module).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("rz angle"),
            "Error should mention context: {msg}"
        );
    }

    #[test]
    fn test_unresolved_angle_constant_error() {
        let q = SSAValue::new(0);
        let angle_ssa = SSAValue::new(99); // not in const map
        let mut module = make_module(vec![
            // No ConstFloat for angle_ssa=99
            Instruction {
                results: vec![],
                operation: Operation::Custom(CustomOp::new("qis", "rz", vec![], BTreeMap::new())),
                operands: vec![q, angle_ssa],
                result_types: vec![],
                regions: vec![],
                attributes: BTreeMap::new(),
                location: None,
            },
        ]);

        let err = convert_qis_to_quantum(&mut module).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("cannot resolve"),
            "Error should mention unresolved constant: {msg}"
        );
    }

    // ──────────────────────────────────────────────────────────────────
    // Non-QIS ops are preserved
    // ──────────────────────────────────────────────────────────────────

    #[test]
    fn test_non_qis_custom_ops_preserved() {
        let mut module = make_module(vec![Instruction {
            results: vec![],
            operation: Operation::Custom(CustomOp::new(
                "other_dialect",
                "some_op",
                vec![],
                BTreeMap::new(),
            )),
            operands: vec![],
            result_types: vec![],
            regions: vec![],
            attributes: BTreeMap::new(),
            location: None,
        }]);

        convert_qis_to_quantum(&mut module).unwrap();
        assert_eq!(module.body.blocks[0].operations.len(), 1);
        assert!(matches!(
            module.body.blocks[0].operations[0].operation,
            Operation::Custom(_)
        ));
    }

    #[test]
    fn test_const_float_preserved() {
        let ssa = SSAValue::new(0);
        let mut module = make_module(vec![emit_const_float(ssa, 1.5)]);

        convert_qis_to_quantum(&mut module).unwrap();
        // ConstFloat should be kept
        assert_eq!(module.body.blocks[0].operations.len(), 1);
        assert!(matches!(
            module.body.blocks[0].operations[0].operation,
            Operation::Classical(ClassicalOp::ConstFloat(_))
        ));
    }
}
