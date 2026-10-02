//! Helpers for emitting QIS instructions.

use crate::ops::{ClassicalOp, CustomOp, Operation};
use crate::phir::{Instruction, SSAValue};
use std::collections::BTreeMap;

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

/// Helper: emit a `qis.rz(qubit, &[angle])` instruction.
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
