use pecos_engines::ByteMessage;
use pecos_phir::{
    execution::processor::PhirProcessor,
    ops::{Operation, QuantumOp},
    phir::{Instruction, SSAValue},
};

#[test]
fn resource_operations_reject_extra_operands_and_results() {
    for op in [
        QuantumOp::Measure,
        QuantumOp::Reset,
        QuantumOp::InitZero,
        QuantumOp::InitOne,
        QuantumOp::InitPlus,
        QuantumOp::InitMinus,
        QuantumOp::Dealloc,
        QuantumOp::H,
        QuantumOp::CX,
        QuantumOp::Alloc,
    ] {
        for (operands, results) in [(4, 0), (if op == QuantumOp::CX { 2 } else { 1 }, 4)] {
            let instruction = Instruction::new(
                Operation::Quantum(op.clone()),
                (0..operands).map(SSAValue::new).collect(),
                (10..10 + results).map(SSAValue::new).collect(),
                vec![],
            );
            let mut processor = PhirProcessor::new();
            for id in 0..4 {
                processor.ssa_values.insert(
                    id,
                    pecos_phir::execution::environment::TypedValue::I64(i64::from(id)),
                );
            }
            let error = processor
                .process_instruction(&instruction, &mut ByteMessage::quantum_operations_builder())
                .expect_err("invalid arity must fail")
                .to_string();
            assert!(error.contains("count"), "{op:?}: {error}");
            assert!(error.contains('4'), "{op:?}: {error}");
        }
    }
}

#[test]
fn classical_and_memory_operations_reject_consumed_prefixes() {
    use pecos_phir::ops::{ClassicalOp, MemoryOp};
    for operation in [
        Operation::Classical(ClassicalOp::ConstBool(true)),
        Operation::Classical(ClassicalOp::Assign),
        Operation::Classical(ClassicalOp::Add),
        Operation::Classical(ClassicalOp::Select),
        Operation::Classical(ClassicalOp::Result),
        Operation::Memory(MemoryOp::Load),
        Operation::Memory(MemoryOp::Store),
    ] {
        let instruction = Instruction::new(
            operation,
            (0..4).map(SSAValue::new).collect(),
            vec![SSAValue::new(10)],
            vec![],
        );
        let error = PhirProcessor::new()
            .process_instruction(&instruction, &mut ByteMessage::quantum_operations_builder())
            .expect_err("extra operands must fail before reading SSA values")
            .to_string();
        assert!(error.contains("operand count 4"), "{error}");
    }
}

#[test]
fn two_qubit_operations_reject_aliased_physical_operands() {
    use pecos_core::Angle64;
    use pecos_phir::execution::environment::TypedValue;
    let angle = Angle64::QUARTER_TURN;
    for op in [
        QuantumOp::RXXRYYRZZ(angle, angle, angle),
        QuantumOp::RXX(angle),
        QuantumOp::RYY(angle),
        QuantumOp::RZZ(angle),
        QuantumOp::CX,
        QuantumOp::CY,
        QuantumOp::CZ,
        QuantumOp::SWAP,
        QuantumOp::SZZ,
        QuantumOp::RXYXY2Q(angle, angle),
    ] {
        let instruction = Instruction::new(
            Operation::Quantum(op.clone()),
            vec![SSAValue::new(0), SSAValue::new(1)],
            vec![SSAValue::new(2), SSAValue::new(3)],
            vec![],
        );
        let mut processor = PhirProcessor::new();
        processor.ssa_values.insert(0, TypedValue::U32(7));
        processor.ssa_values.insert(1, TypedValue::U32(7));
        let error = processor
            .process_instruction(&instruction, &mut ByteMessage::quantum_operations_builder())
            .unwrap_err()
            .to_string();
        assert!(
            error.contains('7') && error.contains("distinct"),
            "{op:?}: {error}"
        );
    }
}
