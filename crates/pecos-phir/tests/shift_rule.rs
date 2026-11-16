use pecos_phir::error::{PhirError, RuntimeError, SourceLocation};
use pecos_phir::execution::{environment::TypedValue, processor::PhirProcessor};
use pecos_phir::ops::{ClassicalOp, Operation, SSAValue};
use pecos_phir::phir::Instruction;

fn run(
    left: TypedValue,
    count: Option<TypedValue>,
    op: &ClassicalOp,
) -> Result<TypedValue, PhirError> {
    let mut processor = PhirProcessor::new();
    processor.ssa_values.insert(0, left);
    let mut operands = vec![SSAValue::new(0)];
    if let Some(count) = count {
        processor.ssa_values.insert(1, count);
        operands.push(SSAValue::new(1));
    }
    let mut instruction = Instruction::new(
        Operation::Classical(op.clone()),
        operands,
        vec![SSAValue::new(2)],
        vec![],
    );
    instruction.location = Some(SourceLocation::new("shift.phir", 7, 3));
    processor.process_classical_operation(op, &instruction)?;
    Ok(processor.ssa_values[&2].clone())
}

// Fix demonstration: both instruction modes saturate, preserving 32-bit truncation.
#[test]
fn shift_rule_processor_boundaries() {
    for count in [31, 32, 33, 63, 64, 65, 65535, 65536] {
        for binary in [false, true] {
            let c = binary.then_some(TypedValue::I64(i64::from(count)));
            assert_eq!(
                run(TypedValue::I64(-8), c, &ClassicalOp::Shr(count)).unwrap(),
                TypedValue::I64(-1)
            );
            let c = binary.then_some(TypedValue::U64(u64::from(count)));
            let expected = if count < 64 { 1_u64 << count } else { 0 };
            assert_eq!(
                run(TypedValue::U64(1), c.clone(), &ClassicalOp::Shl(count)).unwrap(),
                TypedValue::U64(expected)
            );
            let expected = if count < 64 {
                (1_u64 << 63) >> count
            } else {
                0
            };
            assert_eq!(
                run(TypedValue::U64(1 << 63), c, &ClassicalOp::Shr(count)).unwrap(),
                TypedValue::U64(expected)
            );
            let c = binary.then_some(TypedValue::U32(count));
            let expected = if count < 32 { 1_u32 << count } else { 0 };
            assert_eq!(
                run(TypedValue::U32(1), c.clone(), &ClassicalOp::Shl(count)).unwrap(),
                TypedValue::U32(expected)
            );
            let expected = if count < 32 {
                (1_u32 << 31) >> count
            } else {
                0
            };
            assert_eq!(
                run(TypedValue::U32(1 << 31), c, &ClassicalOp::Shr(count)).unwrap(),
                TypedValue::U32(expected)
            );
            let c = binary.then_some(TypedValue::I32(i32::try_from(count).unwrap()));
            assert_eq!(
                run(TypedValue::I32(1), c.clone(), &ClassicalOp::Shl(count)).unwrap(),
                TypedValue::I32(if count == 31 { i32::MIN } else { 0 }),
            );
            assert_eq!(
                run(TypedValue::I32(-8), c, &ClassicalOp::Shr(count)).unwrap(),
                TypedValue::I32(-1)
            );
        }
    }
}

// Fix demonstration: reject by the original tag, including mixed operand types.
#[test]
fn shift_rule_processor_count_tags() {
    for op in [ClassicalOp::Shl(0), ClassicalOp::Shr(0)] {
        for count in [
            TypedValue::I8(-1),
            TypedValue::I16(-1),
            TypedValue::I32(-1),
            TypedValue::I64(-1),
        ] {
            let err = run(TypedValue::U64(8), Some(count), &op).unwrap_err();
            let PhirError::Runtime(err) = err else {
                panic!("runtime error required")
            };
            assert!(
                matches!(*err, RuntimeError::ExecutionFailed { message, location } if message == "Negative shift amount: -1" && location == SourceLocation::new("shift.phir", 7, 3))
            );
        }
    }
}

// Compatibility pin: boolean counts and mixed-result tags are unchanged.
#[test]
fn shift_rule_processor_bool() {
    assert_eq!(
        run(
            TypedValue::U64(8),
            Some(TypedValue::Bool(true)),
            &ClassicalOp::Shr(0)
        )
        .unwrap(),
        TypedValue::I64(4)
    );
    assert_eq!(
        run(
            TypedValue::U64(8),
            Some(TypedValue::Bool(false)),
            &ClassicalOp::Shl(0)
        )
        .unwrap(),
        TypedValue::I64(8)
    );
}

// Fix demonstration: an unsigned top bit never turns a count negative.
#[test]
fn shift_rule_processor_unsigned_high_count() {
    for op in [ClassicalOp::Shl(0), ClassicalOp::Shr(0)] {
        for left in [TypedValue::U64(8), TypedValue::I64(8)] {
            let expected = if matches!(left, TypedValue::U64(_)) {
                TypedValue::U64(0)
            } else {
                TypedValue::I64(0)
            };
            assert_eq!(
                run(left, Some(TypedValue::U64(1 << 63)), &op).unwrap(),
                expected
            );
        }
    }
}

#[test]
fn noninteger_unary_operand_is_an_input_validation_error() {
    let error = run(TypedValue::F64(1.5), None, &ClassicalOp::Shl(1)).unwrap_err();
    assert!(matches!(error, PhirError::Validation(_)), "{error}");
    let message = error.to_string();
    assert!(
        message.contains("1.5") && message.contains("use an integer operand"),
        "{message}"
    );
}
