//! Fix demonstrations for #1217. These tests use only APIs and variants present
//! on the base commit, so their computed-value assertions can be checked there.
use pecos_engines::hybrid::builder::HybridEngineBuilder;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{ClassicalEngine, ControlEngine, EngineStage, EngineSystem};
use pecos_phir::PhirEngine;
use pecos_phir::error::PhirError;
use pecos_phir::execution::{environment::TypedValue, processor::PhirProcessor};
use pecos_phir::ops::{ClassicalOp, MemoryOp, Operation};
use pecos_phir::phir::{AttributeValue, Instruction, SSAValue};
use pecos_phir::types::{IntWidth, Type};

fn module(body: &str) -> pecos_phir::phir::Module {
    pecos_phir::parse_qis_to_quantum(&format!("define void @main() {{\n{body}\nret void\n}}\n"))
        .unwrap()
}

fn value(body: &str) -> TypedValue {
    value_with_qubits(body, 0)
}

fn value_with_qubits(body: &str, qubits: usize) -> TypedValue {
    let module = module(body);
    let id = module.body.blocks[0].operations.last().unwrap().results[0].id;
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(PhirEngine::new(module).unwrap()))
        .with_quantum_engine(Box::new(StateVecEngine::new(qubits)))
        .build();
    hybrid.run_shot().unwrap();
    let engine = hybrid
        .controller()
        .as_any()
        .downcast_ref::<PhirEngine>()
        .unwrap();
    assert!(engine.finished);
    engine
        .processor
        .ssa_values
        .get(&id)
        .expect("computed SSA result")
        .clone()
}

// Expected integers below are hand calculated with wrapping 64-bit arithmetic.
// In particular (2^64 - 8) / 7 = 2_635_249_153_387_078_801, remainder 1.
macro_rules! binary_cases {
    ($name:ident, $op:literal, $a:literal, $b:literal, $expected:literal) => {
        #[test]
        fn $name() {
            for (a, b, expected) in [(12_i64, 3_i64, positive_expected($op)), ($a, $b, $expected)] {
                for mixed in [false, true] {
                    let body = if mixed {
                        format!("%a = add i64 {a}, 0\n%r = {} i64 %a, {b}", $op)
                    } else {
                        format!("%r = {} i64 {a}, {b}", $op)
                    };
                    assert_eq!(value(&body), TypedValue::I64(expected), "{body}");
                }
            }
        }
    };
}

fn positive_expected(op: &str) -> i64 {
    match op {
        "add" | "or" | "xor" => 15,
        "sub" => 9,
        "mul" => 36,
        "sdiv" | "udiv" => 4,
        "srem" | "urem" | "and" => 0,
        "shl" => 96,
        "ashr" | "lshr" => 1,
        _ => unreachable!(),
    }
}

binary_cases!(qis_add_values, "add", -8, 7, -1);
binary_cases!(qis_sub_values, "sub", -8, 7, -15);
binary_cases!(qis_mul_values, "mul", -8, 7, -56);
binary_cases!(qis_udiv_values, "udiv", -8, 7, 2_635_249_153_387_078_801);
binary_cases!(qis_sdiv_values, "sdiv", -8, 7, -1);
binary_cases!(qis_urem_values, "urem", -8, 7, 1);
binary_cases!(qis_srem_values, "srem", -8, 7, -1);
binary_cases!(qis_and_values, "and", -8, 7, 0);
binary_cases!(qis_or_values, "or", -8, 7, -1);
binary_cases!(qis_xor_values, "xor", -8, 7, -1);
binary_cases!(qis_shl_values, "shl", -8, 2, -32);
binary_cases!(qis_lshr_values, "lshr", -8, 2, 4_611_686_018_427_387_902);
binary_cases!(qis_ashr_values, "ashr", -8, 2, -2);

macro_rules! comparison_cases {
    ($name:ident, $pred:literal, $positive:literal, $negative:literal, $equal:literal) => {
        #[test]
        fn $name() {
            for (a, b, expected) in [(2, 3, $positive), (-1, 0, $negative), (-1, -1, $equal)] {
                for mixed in [false, true] {
                    let body = if mixed {
                        format!("%a = add i64 {a}, 0\n%r = icmp {} i64 %a, {b}", $pred)
                    } else {
                        format!("%r = icmp {} i64 {a}, {b}", $pred)
                    };
                    assert_eq!(value(&body), TypedValue::Bool(expected), "{body}");
                }
            }
        }
    };
}
comparison_cases!(qis_icmp_eq_values, "eq", false, false, true);
comparison_cases!(qis_icmp_ne_values, "ne", true, true, false);
comparison_cases!(qis_icmp_slt_values, "slt", true, true, false);
comparison_cases!(qis_icmp_sle_values, "sle", true, true, true);
comparison_cases!(qis_icmp_sgt_values, "sgt", false, false, false);
comparison_cases!(qis_icmp_sge_values, "sge", false, false, true);
comparison_cases!(qis_icmp_ult_values, "ult", true, false, false);
comparison_cases!(qis_icmp_ule_values, "ule", true, false, true);
comparison_cases!(qis_icmp_ugt_values, "ugt", false, true, false);
comparison_cases!(qis_icmp_uge_values, "uge", false, true, true);

#[test]
fn qis_select_typed_literals() {
    for (ty, a, b, yes, no) in [
        (
            "double",
            "0x3FF8000000000000",
            "2.5",
            TypedValue::F64(1.5),
            TypedValue::F64(2.5),
        ),
        ("i64", "-8", "7", TypedValue::I64(-8), TypedValue::I64(7)),
        (
            "i1",
            "true",
            "false",
            TypedValue::Bool(true),
            TypedValue::Bool(false),
        ),
        (
            "float",
            "1.5",
            "2.5",
            TypedValue::F64(1.5),
            TypedValue::F64(2.5),
        ),
    ] {
        for (cond, expected) in [("true", yes), ("false", no)] {
            assert_eq!(
                value(&format!("%r = select i1 {cond}, {ty} {a}, {ty} {b}")),
                expected
            );
        }
    }
}

#[test]
fn qis_undef_poison_zero() {
    for literal in ["undef", "poison"] {
        for (ty, expected) in [
            ("i8", TypedValue::I64(0)),
            ("i1", TypedValue::Bool(false)),
            ("double", TypedValue::F64(0.0)),
        ] {
            assert_eq!(
                value(&format!(
                    "%r = select i1 true, {ty} {literal}, {ty} {literal}"
                )),
                expected
            );
        }
        assert_eq!(
            value(&format!("%r = select i1 {literal}, i64 -8, i64 7")),
            TypedValue::I64(7)
        );
        assert_eq!(
            value(&format!("%r = add i64 -8, {literal}")),
            TypedValue::I64(-8)
        );
    }
    for ty in ["ptr", "i8*", "{ i64, i64 }", "[2 x i64]", "<2 x i64>"] {
        for literal in ["null", "undef", "poison", "zeroinitializer"] {
            assert_eq!(
                value(&format!(
                    "%r = select i1 true, {ty} {literal}, {ty} {literal}"
                )),
                TypedValue::I64(0)
            );
        }
    }
}

#[test]
fn qis_signed_overflow_zero() {
    for op in ["sdiv", "srem"] {
        assert_eq!(
            value(&format!("%r = {op} i64 -9223372036854775808, -1")),
            TypedValue::I64(0)
        );
    }
}

#[test]
fn qis_zero_divisors() {
    for op in ["udiv", "urem", "sdiv", "srem"] {
        assert_eq!(value(&format!("%r = {op} i64 -8, 0")), TypedValue::I64(0));
    }
}

// Fix demonstration: this lossless chain agrees with LLVM at every width.
// Width-changing semantics remain tracked by #1220.
#[test]
fn qis_hugr_cast_chain() {
    assert_eq!(
        value(
            "%a = add i64 8, 0\n%b = trunc i64 %a to i16\n%c = zext i16 %b to i32\n%r = sext i32 %c to i64"
        ),
        TypedValue::I64(8)
    );
}

#[test]
fn qis_boolean_xor() {
    for literal in ["true", "1", "-1"] {
        assert_eq!(
            value(&format!("%a = icmp eq i64 1, 1\n%r = xor i1 %a, {literal}")),
            TypedValue::Bool(false)
        );
    }
    assert_eq!(value("%r = add i1 true, true"), TypedValue::Bool(false));
}

#[test]
fn qis_store_literals() {
    for (ty, literal, expected) in [
        ("i64", "-8", TypedValue::I64(-8)),
        ("double", "0x3FF8000000000000", TypedValue::F64(1.5)),
        ("i1", "true", TypedValue::Bool(true)),
    ] {
        assert_eq!(
            value(&format!(
                "%p = alloca {ty}\nstore {ty} {literal}, ptr %p\n%r = load {ty}, ptr %p"
            )),
            expected
        );
    }
}

#[test]
fn qis_invalid_literals_error() {
    for text in [
        "9223372036854775808",
        "-9223372036854775809",
        "nonsense",
        "null",
    ] {
        assert!(
            pecos_phir::parse_qis_to_quantum(&format!(
                "define void @main() {{\n%r = add i64 {text}, 1\n}}\n"
            ))
            .is_err()
        );
    }
    assert!(
        pecos_phir::parse_qis_to_quantum(
            "define void @main() {\n%r = select i1 true, double nonsense, double 1.0\n}\n"
        )
        .is_err()
    );
}

#[test]
fn qis_call_arg_literals() {
    // Inspect emitted argument constants even for quantum operations whose IDs
    // are resolved statically during conversion.
    for (ty, literal, expected) in [
        ("i1", "true", ClassicalOp::ConstBool(true)),
        ("i1", "false", ClassicalOp::ConstBool(false)),
        ("i1", "1", ClassicalOp::ConstInt(1)),
        ("i1", "-1", ClassicalOp::ConstInt(-1)),
        ("i8", "7", ClassicalOp::ConstInt(7)),
        ("i16", "7", ClassicalOp::ConstInt(7)),
        ("i128", "7", ClassicalOp::ConstInt(7)),
    ] {
        let module = module(&format!(
            "%unused = call i1 @___read_future({ty} {literal})"
        ));
        assert_eq!(
            module.body.blocks[0].operations[0].operation,
            Operation::Classical(expected)
        );
        if ty == "i1" {
            assert_eq!(
                module.body.blocks[0].operations[0].result_types,
                vec![Type::Bool]
            );
        }
    }
}

fn classical(op: ClassicalOp, count: usize) -> Instruction {
    Instruction::new(
        Operation::Classical(op),
        (0..count)
            .map(|i| SSAValue::new(u32::try_from(i).unwrap()))
            .collect(),
        vec![SSAValue::new(9)],
        vec![Type::Int(IntWidth::I64)],
    )
}

// Fix demonstration: verify every changed read, including both binary sides,
// both selected values, and missing memory. No new operation names needed.
#[test]
fn processor_missing_operands_error() {
    let mut instructions = vec![
        classical(ClassicalOp::Shl(0), 2),
        classical(ClassicalOp::Shr(0), 2),
        classical(ClassicalOp::Shl(2), 1),
        classical(ClassicalOp::Not, 1),
        classical(ClassicalOp::Neg, 1),
        classical(ClassicalOp::Select, 3),
        classical(ClassicalOp::Assign, 1),
        classical(ClassicalOp::FNeg, 1),
        classical(ClassicalOp::Add, 2),
        classical(ClassicalOp::FAdd, 2),
        classical(ClassicalOp::Eq, 2),
        classical(ClassicalOp::Bitcast, 1),
        classical(ClassicalOp::Result, 1),
        Instruction::new(
            Operation::Memory(MemoryOp::Store),
            vec![SSAValue::new(0), SSAValue::new(1)],
            vec![],
            vec![],
        ),
        Instruction::new(
            Operation::Memory(MemoryOp::Load),
            vec![SSAValue::new(0)],
            vec![SSAValue::new(9)],
            vec![],
        ),
    ];
    // Locate the unsigned variants through the parser, keeping this test
    // compilable on base (where these lower to existing signed variants).
    for body in [
        "%r = udiv i64 1, 2",
        "%r = urem i64 1, 2",
        "%r = lshr i64 1, 2",
        "%r = icmp ult i64 1, 2",
        "%r = icmp ule i64 1, 2",
        "%r = icmp ugt i64 1, 2",
        "%r = icmp uge i64 1, 2",
    ] {
        let parsed = module(body);
        let op = parsed.body.blocks[0]
            .operations
            .last()
            .unwrap()
            .operation
            .clone();
        instructions.push(Instruction::new(
            op,
            vec![SSAValue::new(0), SSAValue::new(1)],
            vec![SSAValue::new(9)],
            vec![Type::Int(IntWidth::I64)],
        ));
    }
    for mut instruction in instructions {
        for operand in &mut instruction.operands {
            operand.id += 100;
        }
        instruction.attributes.insert(
            "export_name".into(),
            AttributeValue::String("answer".into()),
        );
        let reads = if matches!(instruction.operation, Operation::Memory(MemoryOp::Store)) {
            1
        } else {
            instruction.operands.len()
        };
        for missing in 0..reads {
            for condition in [false, true] {
                if matches!(
                    instruction.operation,
                    Operation::Classical(ClassicalOp::Select)
                ) && missing != 0
                    && missing != if condition { 1 } else { 2 }
                {
                    continue;
                }
                let mut processor = PhirProcessor::new();
                for index in 0..instruction.operands.len() {
                    if index != missing {
                        let v = if matches!(
                            instruction.operation,
                            Operation::Classical(ClassicalOp::Select)
                        ) && index == 0
                        {
                            TypedValue::Bool(condition)
                        } else if matches!(
                            instruction.operation,
                            Operation::Classical(ClassicalOp::FAdd | ClassicalOp::FNeg)
                        ) {
                            TypedValue::F64(1.0)
                        } else {
                            TypedValue::I64(1)
                        };
                        processor
                            .ssa_values
                            .insert(instruction.operands[index].id, v);
                    }
                }
                let error = processor
                    .process_instruction(
                        &instruction,
                        &mut pecos_engines::ByteMessageBuilder::new(),
                    )
                    .unwrap_err();
                let PhirError::Internal(message) = error else {
                    panic!("internal error required: {error}");
                };
                assert!(message.contains(&instruction.operation.name()), "{message}");
                assert!(message.contains(&format!("operand {missing}")), "{message}");
                assert!(
                    message.contains(&format!("SSA id {}", instruction.operands[missing].id)),
                    "{message}"
                );
            }
        }
    }
}

#[test]
fn processor_mixed_patterns() {
    for right in [
        TypedValue::U8(0),
        TypedValue::U16(0),
        TypedValue::U32(0),
        TypedValue::U64(0),
        TypedValue::Bool(false),
    ] {
        let mut processor = PhirProcessor::new();
        processor.ssa_values.insert(0, TypedValue::I64(-8));
        processor.ssa_values.insert(1, right);
        let add = classical(ClassicalOp::Add, 2);
        processor
            .process_classical_operation(&ClassicalOp::Add, &add)
            .unwrap();
        assert_eq!(processor.ssa_values[&9], TypedValue::I64(-8));
        let lt = classical(ClassicalOp::Lt, 2);
        processor
            .process_classical_operation(&ClassicalOp::Lt, &lt)
            .unwrap();
        assert_eq!(processor.ssa_values[&9], TypedValue::Bool(true));
        for (predicate, expected) in [("ult", false), ("ule", false), ("ugt", true), ("uge", true)]
        {
            let parsed = module(&format!("%r = icmp {predicate} i64 -8, 0"));
            let Operation::Classical(op) =
                &parsed.body.blocks[0].operations.last().unwrap().operation
            else {
                panic!()
            };
            let instruction = classical(op.clone(), 2);
            processor
                .process_classical_operation(op, &instruction)
                .unwrap();
            assert_eq!(processor.ssa_values[&9], TypedValue::Bool(expected));
        }
    }
}

#[test]
fn processor_declared_integer_storage() {
    for (ty, expected) in [
        (Type::Int(IntWidth::I8), TypedValue::I64(-8)),
        (Type::Int(IntWidth::I64), TypedValue::I64(-8)),
        (Type::UInt(IntWidth::I32), TypedValue::U32(4_294_967_288)),
        (
            Type::UInt(IntWidth::I64),
            TypedValue::U64(18_446_744_073_709_551_608),
        ),
        (Type::Bool, TypedValue::Bool(false)),
    ] {
        let mut processor = PhirProcessor::new();
        let mut instruction = classical(ClassicalOp::ConstInt(-8), 0);
        instruction.result_types = vec![ty.clone()];
        processor
            .process_classical_operation(&ClassicalOp::ConstInt(-8), &instruction)
            .unwrap();
        assert_eq!(processor.ssa_values[&9], expected);
        for source in [
            TypedValue::I64(-8),
            TypedValue::U64(18_446_744_073_709_551_608),
        ] {
            processor.ssa_values.insert(0, source);
            let mut cast = classical(ClassicalOp::Bitcast, 1);
            cast.result_types = vec![ty.clone()];
            processor
                .process_classical_operation(&ClassicalOp::Bitcast, &cast)
                .unwrap();
            assert_eq!(processor.ssa_values[&9], expected);
        }
    }
}

#[test]
fn qis_logical_shift_count_rule() {
    for count in [0, 1, 63, 64, 65, 65536] {
        let expected = match count {
            0 => -8,
            1 => 9_223_372_036_854_775_804,
            63 => 1,
            _ => 0,
        };
        assert_eq!(
            value(&format!("%r = lshr i64 -8, {count}")),
            TypedValue::I64(expected)
        );
        let parsed = module(&format!("%r = lshr i64 -8, {count}"));
        let Operation::Classical(op) = &parsed.body.blocks[0].operations.last().unwrap().operation
        else {
            panic!()
        };
        // The variant's embedded count is obtained without mentioning a new
        // variant, via serde's existing RON representation.
        let text = ron::to_string(op).unwrap();
        let unary_op: ClassicalOp =
            ron::from_str(&text.replace("(0)", &format!("({count})"))).unwrap();
        let instruction = classical(unary_op.clone(), 1);
        let mut processor = PhirProcessor::new();
        processor.ssa_values.insert(0, TypedValue::I64(-8));
        processor
            .process_classical_operation(&unary_op, &instruction)
            .unwrap();
        assert_eq!(processor.ssa_values[&9], TypedValue::I64(expected));
    }
    let mut engine = PhirEngine::new(module("%r = lshr i64 -8, -1")).unwrap();
    assert!(
        engine
            .generate_commands()
            .err()
            .expect("missing operand error")
            .to_string()
            .contains("Negative shift amount")
    );
}

// Compatibility pin: the existing signed operators already implement LLVM's
// truncation toward zero, dividend-signed remainder and sign-filling shift.
#[test]
fn processor_signed_semantics_compatibility() {
    for (op, b, expected) in [
        (ClassicalOp::Div, 7, -1),
        (ClassicalOp::Mod, 7, -1),
        (ClassicalOp::Shr(0), 2, -2),
    ] {
        let mut processor = PhirProcessor::new();
        processor.ssa_values.insert(0, TypedValue::I64(-8));
        processor.ssa_values.insert(1, TypedValue::I64(b));
        processor
            .process_classical_operation(&op, &classical(op.clone(), 2))
            .unwrap();
        assert_eq!(processor.ssa_values[&9], TypedValue::I64(expected));
    }
}

// Compatibility pin: QASM and PHIR-JSON's declared u32 packing stays u32.
#[test]
fn processor_u32_packing_compatibility() {
    let mut processor = PhirProcessor::new();
    let mut constant = classical(ClassicalOp::ConstInt(0), 0);
    constant.result_types = vec![Type::UInt(IntWidth::I32)];
    processor
        .process_classical_operation(&ClassicalOp::ConstInt(0), &constant)
        .unwrap();
    assert_eq!(processor.ssa_values[&9], TypedValue::U32(0));
    processor.ssa_values.insert(0, TypedValue::Bool(true));
    let mut cast = classical(ClassicalOp::Bitcast, 1);
    cast.result_types = vec![Type::UInt(IntWidth::I32)];
    processor
        .process_classical_operation(&ClassicalOp::Bitcast, &cast)
        .unwrap();
    assert_eq!(processor.ssa_values[&9], TypedValue::U32(1));
}

// Fix demonstration: producers intentionally left for follow-up issues fail at
// their first consumer, including lines the parser cannot lower yet.
#[test]
fn qis_unimplemented_producers_error() {
    for producer in [
        "%a = or disjoint i64 1, 2",
        "%a = trunc nuw i64 1 to i32",
        "%a = fadd double 1.0, 2.0",
        "%a = getelementptr i64, ptr null, i64 0",
        "%a = call i64 @non_qis()",
        "%a = phi i64 [ 1, %entry ]",
        "%a = call i1 @___read_future(ptr null)",
        "call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)\n%a = call i1 @__quantum__qis__read_result__body(%Result* null)",
        "%a = insertvalue { i64 } poison, i64 1, 0",
    ] {
        let body = format!("{producer}\n%r = add i64 %a, 1");
        let mut engine = PhirEngine::new(module(&body)).unwrap();
        let error = engine
            .generate_commands()
            .err()
            .expect("missing operand error")
            .to_string();
        assert!(
            error.contains("Internal error")
                && error.contains("operand 0")
                && error.contains("SSA id"),
            "{producer}: {error}"
        );
    }
}

// Fix demonstration: cast literals use the same type mapping. Values are
// lossless or truncate to i1, so expectations do not pin the #1220 defect.
#[test]
fn qis_cast_literals() {
    assert_eq!(value("%r = trunc i64 8 to i16"), TypedValue::I64(8));
    assert_eq!(value("%r = zext i1 true to i64"), TypedValue::I64(1));
    assert_eq!(value("%r = trunc i64 1 to i1"), TypedValue::Bool(true));
    assert_eq!(value("%r = trunc i64 -8 to i1"), TypedValue::Bool(false));
}

// Fix demonstration: memory consumers wait for pending measurement values.
#[test]
fn qis_store_measurement_value() {
    assert_eq!(
        value_with_qubits(
            "%p = alloca i1\n%m = call i1 @___measure(i64 0)\nstore i1 %m, ptr %p\n%r = load i1, ptr %p",
            1
        ),
        TypedValue::Bool(false)
    );
}

// Compatibility pin: select reads only its chosen value.
#[test]
fn processor_select_unchosen_compatibility() {
    for condition in [false, true] {
        let mut processor = PhirProcessor::new();
        processor.ssa_values.insert(0, TypedValue::Bool(condition));
        processor
            .ssa_values
            .insert(if condition { 1 } else { 2 }, TypedValue::I64(7));
        processor
            .process_classical_operation(&ClassicalOp::Select, &classical(ClassicalOp::Select, 3))
            .unwrap();
        assert_eq!(processor.ssa_values[&9], TypedValue::I64(7));
    }
}

// Fix demonstration: signed remainder overflow must not panic, independently
// of parser constants and the division overflow case.
#[test]
fn processor_signed_remainder_overflow() {
    let mut processor = PhirProcessor::new();
    processor.ssa_values.insert(0, TypedValue::I64(i64::MIN));
    processor.ssa_values.insert(1, TypedValue::I64(-1));
    processor
        .process_classical_operation(&ClassicalOp::Mod, &classical(ClassicalOp::Mod, 2))
        .unwrap();
    assert_eq!(processor.ssa_values[&9], TypedValue::I64(0));
}

// Fix demonstration: unsigned predicates interpret every integer storage tag
// and combination as a 64-bit pattern, including same-type signed pairs.
#[test]
fn processor_unsigned_comparison_tags() {
    let nonzero = [
        TypedValue::I8(-1),
        TypedValue::I16(-1),
        TypedValue::I32(-1),
        TypedValue::I64(-1),
        TypedValue::U8(255),
        TypedValue::U16(65535),
        TypedValue::U32(u32::MAX),
        TypedValue::U64(u64::MAX),
        TypedValue::Bool(true),
    ];
    let zero = [
        TypedValue::I8(0),
        TypedValue::I16(0),
        TypedValue::I32(0),
        TypedValue::I64(0),
        TypedValue::U8(0),
        TypedValue::U16(0),
        TypedValue::U32(0),
        TypedValue::U64(0),
        TypedValue::Bool(false),
    ];
    for (predicate, expected) in [("ult", false), ("ule", false), ("ugt", true), ("uge", true)] {
        let parsed = module(&format!("%r = icmp {predicate} i64 -1, 0"));
        let Operation::Classical(op) = &parsed.body.blocks[0].operations.last().unwrap().operation
        else {
            panic!()
        };
        for left in &nonzero {
            for right in &zero {
                let mut processor = PhirProcessor::new();
                processor.ssa_values.insert(0, left.clone());
                processor.ssa_values.insert(1, right.clone());
                processor
                    .process_classical_operation(op, &classical(op.clone(), 2))
                    .unwrap();
                assert_eq!(
                    processor.ssa_values[&9],
                    TypedValue::Bool(expected),
                    "{predicate} {left:?} {right:?}"
                );
            }
        }
    }
}

// Compatibility pin: erased runtime output bookkeeping does not parse or
// materialise unused string-pointer constant expressions.
#[test]
fn qis_erased_runtime_arguments_compatibility() {
    let module = module(
        "call void @__quantum__rt__result_record_output(i64 0, i8* getelementptr inbounds ([2 x i8], [2 x i8]* @label, i32 0, i32 0))",
    );
    assert_eq!(module.body.blocks[0].operations.len(), 0);
}

// Fix demonstration: delivery must assign each batch's results, without
// overwriting the preceding measurement. Repeating shots also exercises reset.
#[test]
fn qis_measure_consume_measure_consume() {
    let module = module(
        "%q0 = call i64 @__quantum__rt__qubit_allocate()\n\
         %q1 = call i64 @__quantum__rt__qubit_allocate()\n\
         call void @___x(i64 %q1)\n\
         %m = call i1 @___measure(i64 %q0)\n\
         %z = zext i1 %m to i64\n\
         %m2 = call i1 @___measure(i64 %q1)\n\
         %r = zext i1 %m2 to i64",
    );
    let casts: Vec<_> = module.body.blocks[0]
        .operations
        .iter()
        .filter(|instruction| instruction.operation == Operation::Classical(ClassicalOp::Bitcast))
        .map(|instruction| instruction.results[0].id)
        .collect();
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(PhirEngine::new(module).unwrap()))
        .with_quantum_engine(Box::new(StateVecEngine::new(2)))
        .build();
    for _ in 0..2 {
        hybrid.run_shot().unwrap();
        let engine = hybrid
            .controller()
            .as_any()
            .downcast_ref::<PhirEngine>()
            .unwrap();
        assert!(engine.finished);
        assert_eq!(engine.processor.ssa_values[&casts[0]], TypedValue::I64(0));
        assert_eq!(engine.processor.ssa_values[&casts[1]], TypedValue::I64(1));
        let measurements = &engine.processor.measurement_mappings;
        assert_eq!(
            engine.processor.ssa_values[&measurements[0]],
            TypedValue::Bool(false)
        );
        assert_eq!(
            engine.processor.ssa_values[&measurements[1]],
            TypedValue::Bool(true)
        );
        pecos_engines::Engine::reset(&mut hybrid).unwrap();
    }
}

// Fix demonstration: qalloc produces the index used by HUGR's integer casts.
#[test]
fn qis_alloc_integer_value() {
    assert_eq!(
        value_with_qubits(
            "%q0 = call i64 @__quantum__rt__qubit_allocate()\n%q1 = call i64 @__quantum__rt__qubit_allocate()\n%r = trunc i64 %q1 to i16",
            2
        ),
        TypedValue::I64(1)
    );
}

// Fix demonstration against ac44a7631: attributed integer/float/pointer
// literals must parse exactly like their unadorned counterparts.
#[test]
fn qis_call_parameter_attributes() {
    for (attributed, plain) in [
        (
            "call void @___rz(i64 noundef zeroext 0, double noundef 1.5)",
            "call void @___rz(i64 0, double 1.5)",
        ),
        (
            "call void @___h(ptr noundef noalias align 8 dereferenceable_or_null(8) null)",
            "call void @___h(ptr null)",
        ),
        (
            "call void @___h(%Qubit* noundef noalias align 8 dereferenceable_or_null(8) null)",
            "call void @___h(%Qubit* null)",
        ),
        (
            "%q = call i64 @__quantum__rt__qubit_allocate()\ncall void @___h(ptr nonnull dereferenceable(8) %q)\ncall void @___rz(i64 signext 0, double noundef 1.5)",
            "%q = call i64 @__quantum__rt__qubit_allocate()\ncall void @___h(ptr %q)\ncall void @___rz(i64 0, double 1.5)",
        ),
    ] {
        assert_eq!(module(attributed), module(plain));
    }
}

// Fix demonstration: unmatched deliveries must error before changing any SSA
// values; discarded measurements still occupy their position in the batch.
#[test]
fn processor_measurement_delivery_validation() {
    use pecos_engines::byte_message::builder::ByteMessageBuilder;
    use pecos_phir::ops::QuantumOp;
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();
    let mut measurement = classical(ClassicalOp::Assign, 1);
    measurement.operation = Operation::Quantum(QuantumOp::Measure);
    measurement.results.clear();
    processor
        .process_instruction(&measurement, &mut builder)
        .unwrap();
    measurement.results = vec![SSAValue::new(9)];
    processor
        .process_instruction(&measurement, &mut builder)
        .unwrap();
    for outcomes in [vec![1], vec![1, 0, 1]] {
        let error = processor.handle_measurement_results(&outcomes).unwrap_err();
        assert!(matches!(error, PhirError::Internal(_)));
        assert!(error.to_string().contains("pending measurements"));
        assert!(!processor.ssa_values.contains_key(&9));
    }
    processor.handle_measurement_results(&[0, 1]).unwrap();
    assert_eq!(processor.ssa_values[&9], TypedValue::Bool(true));
    assert!(processor.handle_measurement_results(&[0]).is_err());
    processor
        .process_instruction(&measurement, &mut builder)
        .unwrap();
    processor.reset();
    assert!(processor.handle_measurement_results(&[0]).is_err());
    processor.handle_measurement_results(&[]).unwrap();
}

// Fix demonstration: packed variables retain earlier batches' bits, and a
// later zero outcome clears just the bit delivered in that batch.
#[test]
fn processor_measurement_packing_batches() {
    use pecos_engines::byte_message::builder::ByteMessageBuilder;
    use pecos_phir::execution::environment::DataType;
    use pecos_phir::ops::QuantumOp;
    let mut processor = PhirProcessor::new();
    processor
        .environment
        .add_variable("register", DataType::U32, 1)
        .unwrap();
    processor
        .environment
        .set_variable("register", TypedValue::U32(0))
        .unwrap();
    processor.variable_ssa_map.insert("register".into(), 100);
    processor.export_mappings.insert(100, "register".into());
    let mut builder = ByteMessageBuilder::new();
    let mut measurement = classical(ClassicalOp::Assign, 1);
    measurement.operation = Operation::Quantum(QuantumOp::Measure);
    for (id, outcome, expected) in [(101, 1, 2), (102, 0, 2), (101, 0, 0)] {
        measurement.results = vec![SSAValue::new(id)];
        processor
            .process_instruction(&measurement, &mut builder)
            .unwrap();
        processor.handle_measurement_results(&[outcome]).unwrap();
        assert_eq!(processor.ssa_values[&100], TypedValue::U32(expected));
        assert_eq!(
            processor.environment.get_variable("register").unwrap(),
            Some(&TypedValue::U32(expected))
        );
        processor.finalize_exports();
        assert_eq!(
            processor.final_exports["register"],
            TypedValue::U32(expected)
        );
    }
}

#[test]
fn qis_final_measurement_after_consumer_is_delivered() {
    // The last batch holds a measurement that follows a classical consumer;
    // it must be sent and its outcome delivered even though the program has
    // run out of operations by the time that batch is built.
    let module = module(
        "%q0 = call i64 @__quantum__rt__qubit_allocate()\n\
         %q1 = call i64 @__quantum__rt__qubit_allocate()\n\
         call void @___x(i64 %q1)\n\
         %m = call i1 @___measure(i64 %q0)\n\
         %z = zext i1 %m to i64\n\
         %m2 = call i1 @___measure(i64 %q1)",
    );
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(PhirEngine::new(module).unwrap()))
        .with_quantum_engine(Box::new(StateVecEngine::new(2)))
        .build();
    hybrid.run_shot().unwrap();
    let engine = hybrid
        .controller()
        .as_any()
        .downcast_ref::<PhirEngine>()
        .unwrap();
    let measurements = &engine.processor.measurement_mappings;
    assert_eq!(measurements.len(), 2);
    assert_eq!(
        engine.processor.ssa_values.get(&measurements[1]),
        Some(&TypedValue::Bool(true))
    );
}

#[test]
fn qis_call_argument_spellings() {
    // Typed pointers may span tokens, a quoted attribute may hold `*`, and
    // tabs separate type and value.
    for call in [
        "call void @___h(%Qubit addrspace(1)* null)",
        "call void @___h(i8 addrspace(1)* null)",
        "call void @___h(%Qubit * null)",
        "call void @___h(%Qubit addrspace (1)* null)",
        "call void @___h(void (i64)* null)",
        "call void @___rz(i64\t0, double\t1.5)",
        "call void @___rz(i64 0, double \"note\"=\"*\" 1.5)",
    ] {
        pecos_phir::parse_qis_to_quantum(&format!(
            "declare void @___h(i64)\ndeclare void @___rz(i64, double)\ndefine void @main() {{\n{call}\nret void\n}}\n"
        ))
        .unwrap_or_else(|e| panic!("{call}: {e}"));
    }
    // A malformed argument is a parse error, not a later out-of-bounds panic.
    assert!(
        pecos_phir::parse_qis_to_quantum(
            "define void @main() {\ncall void @___rz(i64 0, garbage)\nret void\n}\n"
        )
        .is_err()
    );
}

#[test]
fn qis_classical_only_program_completes_at_start() {
    // A program without quantum operations has nothing to send, so start()
    // completes instead of handing a header-only batch to the quantum engine.
    let mut engine = PhirEngine::new(module("%r = add i64 1, 2")).unwrap();
    assert!(matches!(
        ControlEngine::start(&mut engine, ()).unwrap(),
        EngineStage::Complete(_)
    ));
}
