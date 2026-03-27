/*!
Helper functions for translating from SLR/qeclib patterns to PHIR

This module provides convenience functions that make it easier to translate
quantum programs written in PECOS's SLR (Simple Logical Representation) and
qeclib to PHIR. The functions mirror SLR's compositional patterns.
*/

use crate::ops::{Operation, QuantumOp, SSAValue};
use crate::phir::{AttributeValue, Block, Instruction, Region};
use crate::types::Type;

/// Create a comment instruction (similar to SLR's Comment)
#[must_use]
pub fn comment(text: &str) -> Instruction {
    // Comments can be represented as attributes on a no-op
    Instruction::new(
        Operation::Custom(crate::ops::CustomOp {
            dialect: "slr".to_string(),
            name: "comment".to_string(),
            operands: vec![],
            attributes: vec![("text".to_string(), AttributeValue::String(text.to_string()))]
                .into_iter()
                .collect(),
        }),
        vec![],
        vec![],
        vec![],
    )
}

/// Create a quantum X gate instruction
#[must_use]
pub fn quantum_x(qubit: SSAValue) -> Instruction {
    Instruction::new(
        Operation::Quantum(QuantumOp::X),
        vec![qubit],
        vec![qubit],
        vec![Type::Qubit],
    )
}

/// Create a quantum Y gate instruction
#[must_use]
pub fn quantum_y(qubit: SSAValue) -> Instruction {
    Instruction::new(
        Operation::Quantum(QuantumOp::Y),
        vec![qubit],
        vec![qubit],
        vec![Type::Qubit],
    )
}

/// Create a quantum Z gate instruction
#[must_use]
pub fn quantum_z(qubit: SSAValue) -> Instruction {
    Instruction::new(
        Operation::Quantum(QuantumOp::Z),
        vec![qubit],
        vec![qubit],
        vec![Type::Qubit],
    )
}

/// Create a quantum H gate instruction
#[must_use]
pub fn quantum_h(qubit: SSAValue) -> Instruction {
    Instruction::new(
        Operation::Quantum(QuantumOp::H),
        vec![qubit],
        vec![qubit],
        vec![Type::Qubit],
    )
}

/// Create a CNOT gate instruction
#[must_use]
pub fn quantum_cx(control: SSAValue, target: SSAValue) -> Instruction {
    Instruction::new(
        Operation::Quantum(QuantumOp::CX),
        vec![control, target],
        vec![control, target],
        vec![Type::Qubit, Type::Qubit],
    )
}

/// Create a measurement instruction
#[must_use]
pub fn mz(qubit: SSAValue) -> (Instruction, SSAValue) {
    let result = SSAValue {
        id: qubit.id + 1000,
        version: 0,
    }; // Simple ID generation
    let inst = Instruction::new(
        Operation::Quantum(QuantumOp::Measure),
        vec![qubit],
        vec![result],
        vec![Type::Bit],
    );
    (inst, result)
}

/// Create a logical Pauli X gate block (Steane code example)
/// This mirrors the pattern from `qeclib/steane/gates_sq/paulis.py`
///
/// # Panics
///
/// Panics if `data_qubits` does not contain exactly 7 qubits
#[must_use]
pub fn logical_x_steane(data_qubits: &[SSAValue]) -> Block {
    assert_eq!(data_qubits.len(), 7, "Steane code requires 7 qubits");

    Block::new(Some("logical_x".to_string()))
        .with_instruction(comment("Logical X"))
        .with_instruction(quantum_x(data_qubits[4]))
        .with_instruction(quantum_x(data_qubits[5]))
        .with_instruction(quantum_x(data_qubits[6]))
        .with_attr("qec.logical_gate", AttributeValue::String("X".to_string()))
        .with_attr("qec.code", AttributeValue::String("steane".to_string()))
}

/// Create a logical Pauli Z gate block (Steane code example)
///
/// # Panics
///
/// Panics if `data_qubits` does not contain exactly 7 qubits
#[must_use]
pub fn logical_z_steane(data_qubits: &[SSAValue]) -> Block {
    assert_eq!(data_qubits.len(), 7, "Steane code requires 7 qubits");

    Block::new(Some("logical_z".to_string()))
        .with_instruction(comment("Logical Z"))
        .with_instruction(quantum_z(data_qubits[0]))
        .with_instruction(quantum_z(data_qubits[1]))
        .with_instruction(quantum_z(data_qubits[2]))
        .with_attr("qec.logical_gate", AttributeValue::String("Z".to_string()))
        .with_attr("qec.code", AttributeValue::String("steane".to_string()))
}

/// Create a syndrome extraction block
/// This is a simplified example - real syndrome extraction would be more complex
#[must_use]
pub fn syndrome_extraction(data_qubits: &[SSAValue], ancilla_qubits: &[SSAValue]) -> Region {
    let region = Region::new(crate::region_kinds::RegionKind::SSACFG);

    // X stabilizer measurements
    let mut x_stabilizers = Block::new(Some("x_stabilizers".to_string()))
        .with_instruction(comment("Measure X stabilizers"));

    // Add X stabilizer measurements using ancilla qubits
    for (i, &ancilla) in ancilla_qubits.iter().enumerate() {
        if i < data_qubits.len() / 2 {
            x_stabilizers = x_stabilizers
                .with_instruction(quantum_h(ancilla))
                .with_instruction(comment(&format!("X stabilizer {i} with data qubits")));
        }
    }
    x_stabilizers =
        x_stabilizers.with_attr("stabilizer.type", AttributeValue::String("X".to_string()));

    // Z stabilizer measurements
    let mut z_stabilizers = Block::new(Some("z_stabilizers".to_string()))
        .with_instruction(comment("Measure Z stabilizers"));

    // Add Z stabilizer measurements using remaining ancilla qubits
    for (i, &_ancilla) in ancilla_qubits.iter().enumerate() {
        if i >= data_qubits.len() / 2 && i < ancilla_qubits.len() {
            z_stabilizers = z_stabilizers.with_instruction(comment(&format!(
                "Z stabilizer {} with data qubit {}",
                i - data_qubits.len() / 2,
                data_qubits[i % data_qubits.len()].id
            )));
        }
    }
    z_stabilizers =
        z_stabilizers.with_attr("stabilizer.type", AttributeValue::String("Z".to_string()));

    region
        .with_block(x_stabilizers)
        .with_block(z_stabilizers)
        .with_attr(
            "protocol",
            AttributeValue::String("syndrome_extraction".to_string()),
        )
}

/// Helper to create a QEC cycle (syndrome extraction + correction)
#[must_use]
pub fn qec_cycle(data_qubits: &[SSAValue], ancilla_qubits: &[SSAValue]) -> Region {
    Region::new(crate::region_kinds::RegionKind::SSACFG)
        .with_block(
            Block::new(Some("extraction".to_string())).with_instruction(comment(&format!(
                "Extract syndrome for {} data qubits using {} ancillas",
                data_qubits.len(),
                ancilla_qubits.len()
            ))),
        )
        .with_block(
            Block::new(Some("decode".to_string())).with_instruction(comment("Decode syndrome")),
        )
        .with_block(
            Block::new(Some("correct".to_string())).with_instruction(comment("Apply corrections")),
        )
        .with_attr("protocol", AttributeValue::String("qec_cycle".to_string()))
}

/// Create a repeat-until-success block pattern (similar to SLR's Repeat)
pub fn repeat_until_success<F>(condition_check: F) -> Region
where
    F: FnOnce() -> SSAValue,
{
    // Get the condition check result
    let condition = condition_check();

    // This would need more sophisticated lowering, but shows the pattern
    Region::new(crate::region_kinds::RegionKind::SSACFG)
        .with_block(
            Block::new(Some("check_condition".to_string())).with_instruction(comment(&format!(
                "Check condition using SSA value {}",
                condition.id
            ))),
        )
        .with_attr(
            "slr.pattern",
            AttributeValue::String("repeat_until_success".to_string()),
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_logical_gates() {
        let qubits: Vec<_> = (0..7).map(SSAValue::new).collect();

        let logical_x = logical_x_steane(&qubits);
        assert_eq!(logical_x.operations.len(), 4); // comment + 3 X gates

        let logical_z = logical_z_steane(&qubits);
        assert_eq!(logical_z.operations.len(), 4); // comment + 3 Z gates
    }
}
