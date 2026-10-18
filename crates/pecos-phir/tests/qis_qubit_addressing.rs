//! Qubit operands name physical qubits by their runtime integer values.
use pecos_engines::hybrid::builder::HybridEngineBuilder;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{ByteMessage, ClassicalEngine, EngineSystem};
use pecos_phir::PhirEngine;
use pecos_phir::execution::{environment::TypedValue, processor::PhirProcessor};
use pecos_phir::ops::{Operation, QuantumOp};
use pecos_phir::phir::{Instruction, Module, SSAValue};
use pecos_phir::types::{IntWidth, Type};

fn module(body: &str) -> Module {
    pecos_phir::parse_qis_to_quantum(&format!("define void @main() {{\n{body}\nret void\n}}\n"))
        .unwrap()
}

fn run(body: &str, qubits: usize, seed: u64) -> (Vec<bool>, usize) {
    try_run(body, qubits, seed).unwrap()
}

fn try_run(
    body: &str,
    qubits: usize,
    seed: u64,
) -> Result<(Vec<bool>, usize), pecos_core::errors::PecosError> {
    // Bound simulator size before running regressions against SSA-id addressing.
    let mut engine = PhirEngine::new(module(body)).unwrap();
    engine.generate_commands().unwrap();
    assert!(
        engine.processor.get_qubit_count() <= 16,
        "unexpected simulator size"
    );
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(engine))
        .with_quantum_engine(Box::new(StateVecEngine::with_seed(qubits, seed)))
        .build();
    hybrid.run_shot()?;
    let engine = hybrid
        .controller()
        .as_any()
        .downcast_ref::<PhirEngine>()
        .unwrap();
    let outcomes = engine
        .processor
        .measurement_mappings
        .iter()
        .map(|id| match engine.processor.ssa_values[id] {
            TypedValue::Bool(outcome) => outcome,
            ref value => panic!("expected measurement Bool, got {value:?}"),
        })
        .collect();
    Ok((outcomes, engine.processor.get_qubit_count()))
}

#[test]
fn allocation_after_consumed_measurement_preserves_state() {
    let body = "%q0 = call i64 @___qalloc()\n\
        call void @___x(i64 %q0)\n\
        %m = call i1 @___measure(i64 %q0)\n\
        %z = zext i1 %m to i64\n\
        %q1 = call i64 @___qalloc()\n\
        %m2 = call i1 @___measure(i64 %q0)";
    let error = try_run(body, 1, 42).unwrap_err().to_string();
    assert!(
        error.contains("1 qubits") && error.contains("qubit 1"),
        "{error}"
    );
    assert_eq!(run(body, 2, 42), (vec![true, true], 2));
}

#[test]
fn literal_addressing() {
    assert_eq!(
        run(
            "call void @___x(i64 0)\n%m = call i1 @___measure(i64 0)",
            1,
            0
        ),
        (vec![true], 1)
    );
}

#[test]
fn qir_null_addressing() {
    assert_eq!(
        run(
            "call void @__quantum__qis__x__body(%Qubit* null)\n\
             call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)\n\
             call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 1 to %Qubit*), %Result* null)",
            2,
            0,
        ),
        (vec![true, false], 2)
    );
}

#[test]
fn qir_inttoptr_addressing() {
    assert_eq!(
        run(
            "call void @__quantum__qis__x__body(%Qubit* inttoptr (i64 1 to %Qubit*))\n\
             call void @__quantum__qis__mz__body(%Qubit* inttoptr (i64 1 to %Qubit*), %Result* null)\n\
             call void @__quantum__qis__mz__body(%Qubit* null, %Result* null)",
            2,
            0,
        ),
        (vec![true, false], 2)
    );
}

const MEMORY_PREFIX: &str = "
    %unused0 = add i64 17, 19
    %unused1 = add i64 23, 29
    %unused2 = add i64 31, 37
    %scratch0 = alloca i64
    %scratch1 = alloca i64
    %slot0 = alloca i16
    %slot1 = alloca i16
    %q0 = call i64 @__quantum__rt__qubit_allocate()
    %small0 = trunc i64 %q0 to i16
    store i16 %small0, i16* %slot0
";

#[test]
fn allocated_index_through_memory() {
    assert_eq!(
        run(
            &format!(
                "{MEMORY_PREFIX}\n\
                 %loaded = load i16, i16* %slot0\n\
                 %wide = zext i16 %loaded to i64\n\
                 call void @___x(i64 %wide)\n\
                 %loaded2 = load i16, i16* %slot0\n\
                 %wide2 = zext i16 %loaded2 to i64\n\
                 %m = call i1 @___measure(i64 %wide2)"
            ),
            1,
            0,
        ),
        (vec![true], 1)
    );
}

#[test]
fn bell_indices_through_memory() {
    let body = format!(
        "{MEMORY_PREFIX}\n\
         %q1 = call i64 @__quantum__rt__qubit_allocate()\n\
         %small1 = trunc i64 %q1 to i16\n\
         store i16 %small1, i16* %slot1\n\
         %hload = load i16, i16* %slot0\n\
         %hindex = zext i16 %hload to i64\n\
         call void @___h(i64 %hindex)\n\
         %cload = load i16, i16* %slot0\n\
         %tload = load i16, i16* %slot1\n\
         %cindex = zext i16 %cload to i64\n\
         %tindex = zext i16 %tload to i64\n\
         call void @___cx(i64 %cindex, i64 %tindex)\n\
         %mload0 = load i16, i16* %slot0\n\
         %mindex0 = zext i16 %mload0 to i64\n\
         %m0 = call i1 @___measure(i64 %mindex0)\n\
         %mload1 = load i16, i16* %slot1\n\
         %mindex1 = zext i16 %mload1 to i64\n\
         %m1 = call i1 @___measure(i64 %mindex1)"
    );
    let mut seen = [false; 2];
    for seed in 0..32 {
        let (outcomes, count) = run(&body, 2, seed);
        assert_eq!(count, 2);
        assert_eq!(outcomes.len(), 2);
        assert_eq!(outcomes[0], outcomes[1], "seed {seed}");
        seen[usize::from(outcomes[0])] = true;
    }
    assert_eq!(seen, [true, true]);
}

fn generated(body: &str) -> PhirEngine {
    let mut engine = PhirEngine::new(module(body)).unwrap();
    engine.generate_commands().unwrap();
    engine
}

#[test]
fn allocation_reuses_freed_index() {
    let module = module(
        "%q0 = call i64 @___qalloc()\n\
         call void @___qfree(i64 %q0)\n\
         %q1 = call i64 @___qalloc()",
    );
    let last = module.body.blocks[0].operations.last().unwrap().results[0].id;
    let mut engine = PhirEngine::new(module).unwrap();
    engine.generate_commands().unwrap();
    assert_eq!(engine.processor.ssa_values[&last], TypedValue::I64(0));
    assert_eq!(engine.processor.get_qubit_count(), 1);
}

#[test]
fn deallocation_frees_only_its_index() {
    // Index 3 stays allocated across the free of 0, so dynamic allocation
    // skips it.
    let module = module(
        "call void @___qalloc(i64 3)\n\
         %a = call i64 @___qalloc()\n\
         call void @___qfree(i64 %a)\n\
         %b = call i64 @___qalloc()\n\
         %c = call i64 @___qalloc()\n\
         %d = call i64 @___qalloc()\n\
         %e = call i64 @___qalloc()",
    );
    let allocations: Vec<u32> = module.body.blocks[0]
        .operations
        .iter()
        .filter(|op| matches!(op.operation, Operation::Quantum(QuantumOp::Alloc)))
        .map(|op| op.results[0].id)
        .collect();
    let mut engine = PhirEngine::new(module).unwrap();
    engine.generate_commands().unwrap();
    let indices: Vec<TypedValue> = allocations
        .iter()
        .map(|id| engine.processor.ssa_values[id].clone())
        .collect();
    assert_eq!(indices, [3, 0, 0, 1, 2, 4].map(TypedValue::I64));
}

#[test]
fn explicit_allocation_index() {
    for spelling in ["___qalloc", "__quantum__rt__qubit_allocate"] {
        let engine = generated(&format!("call void @{spelling}(i64 3)"));
        assert_eq!(engine.processor.get_qubit_count(), 4);
        assert!(
            engine
                .processor
                .ssa_values
                .values()
                .all(|v| *v == TypedValue::I64(3))
        );
    }
}

#[test]
fn duplicate_allocation_errors() {
    let mut engine = PhirEngine::new(module(
        "call void @___qalloc(i64 3)\ncall void @___qalloc(i64 3)",
    ))
    .unwrap();
    let error = engine
        .generate_commands()
        .err()
        .expect("invalid qubit must fail")
        .to_string();
    assert!(error.contains("quantum.alloc"), "{error}");
}

#[test]
fn negative_index_errors() {
    let mut engine = PhirEngine::new(module("call void @___x(i64 -1)")).unwrap();
    let error = engine
        .generate_commands()
        .err()
        .expect("invalid qubit must fail")
        .to_string();
    assert!(error.contains("quantum.x"), "{error}");
}

#[test]
fn float_index_errors() {
    let mut engine = PhirEngine::new(module("call void @___x(double 1.0)")).unwrap();
    let error = engine
        .generate_commands()
        .err()
        .expect("invalid qubit must fail")
        .to_string();
    assert!(error.contains("quantum.x"), "{error}");
}

fn quantum(op: QuantumOp, operands: &[u32], results: &[u32]) -> Instruction {
    Instruction::new(
        Operation::Quantum(op),
        operands.iter().copied().map(SSAValue::new).collect(),
        results.iter().copied().map(SSAValue::new).collect(),
        results.iter().map(|_| Type::Int(IntWidth::I64)).collect(),
    )
}

#[test]
fn index_types_and_all_operand_sites() {
    let valid = [
        TypedValue::I8(2),
        TypedValue::I16(2),
        TypedValue::I32(2),
        TypedValue::I64(2),
        TypedValue::U8(2),
        TypedValue::U16(2),
        TypedValue::U32(2),
        TypedValue::U64(2),
    ];
    let invalid = [
        TypedValue::I8(-1),
        TypedValue::I16(-1),
        TypedValue::I32(-1),
        TypedValue::I64(-1),
        TypedValue::Bool(false),
        TypedValue::F64(2.0),
        TypedValue::BitVec(vec![false]),
        TypedValue::U64(u64::MAX),
        TypedValue::U64(u64::try_from(usize::MAX).unwrap()),
    ];
    for op in [
        QuantumOp::Alloc,
        QuantumOp::Dealloc,
        QuantumOp::Reset,
        QuantumOp::InitZero,
        QuantumOp::X,
        QuantumOp::CX,
        QuantumOp::Measure,
    ] {
        let operands = if op == QuantumOp::CX {
            vec![50, 51]
        } else {
            vec![50]
        };
        let instruction = quantum(op, &operands, &[]);
        for value in valid.iter().chain(&invalid) {
            for &operand in &operands {
                let mut processor = PhirProcessor::new();
                processor.ssa_values.insert(50, TypedValue::I64(0));
                processor.ssa_values.insert(51, TypedValue::I64(1));
                processor.ssa_values.insert(operand, value.clone());
                let mut builder = ByteMessage::quantum_operations_builder();
                let result = processor.process_instruction(&instruction, &mut builder);
                if valid.contains(value) {
                    result.unwrap();
                    assert_eq!(processor.get_qubit_count(), 3);
                    let gates = builder.build().quantum_ops().unwrap();
                    assert!(gates.iter().any(|gate| gate.qubits.contains(&2.into())));
                } else {
                    let error = result.unwrap_err().to_string();
                    assert!(error.contains(&instruction.operation.name()), "{error}");
                }
            }
        }
        let mut processor = PhirProcessor::new();
        let error = processor
            .process_instruction(&instruction, &mut ByteMessage::quantum_operations_builder())
            .unwrap_err()
            .to_string();
        assert!(error.contains("missing operand 0, SSA id 50"), "{error}");
    }
}

#[test]
fn allocation_lifecycle_and_reset() {
    assert_eq!(QuantumOp::Alloc.operand_count(), None);
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessage::quantum_operations_builder();
    processor.ssa_values.insert(50, TypedValue::U16(1));
    for instruction in [
        quantum(QuantumOp::Alloc, &[], &[]),
        quantum(QuantumOp::Alloc, &[50], &[]),
        quantum(QuantumOp::Alloc, &[], &[60]),
        quantum(QuantumOp::Alloc, &[], &[61]),
    ] {
        processor
            .process_instruction(&instruction, &mut builder)
            .unwrap();
    }
    assert_eq!(processor.ssa_values[&60], TypedValue::I64(0));
    assert_eq!(processor.ssa_values[&61], TypedValue::I64(2));
    let gates = builder.build().quantum_ops().unwrap();
    assert_eq!(
        gates,
        vec![
            pecos_core::Gate::qalloc(&[1]),
            pecos_core::Gate::qalloc(&[0]),
            pecos_core::Gate::qalloc(&[2])
        ]
    );
    processor.reset();
    processor
        .process_instruction(&quantum(QuantumOp::Alloc, &[], &[62]), &mut builder)
        .unwrap();
    assert_eq!(processor.ssa_values[&62], TypedValue::I64(0));
    processor
        .process_instruction(&quantum(QuantumOp::Alloc, &[50], &[]), &mut builder)
        .unwrap();
    for _ in 0..2 {
        processor
            .process_instruction(&quantum(QuantumOp::Dealloc, &[50], &[]), &mut builder)
            .unwrap();
    }
}

#[test]
fn index_boundary_and_allocation_result() {
    // Exercise command encoding and decoding without a state-vector backend:
    // u32::MAX is a valid address, but simulating 2^32 qubits is not feasible.
    let index = usize::try_from(u32::MAX).unwrap();
    for op in [
        QuantumOp::X,
        QuantumOp::CX,
        QuantumOp::Measure,
        QuantumOp::Reset,
        QuantumOp::InitZero,
        QuantumOp::Alloc,
        QuantumOp::Dealloc,
    ] {
        let operands = if op == QuantumOp::CX {
            vec![50, 51]
        } else {
            vec![50]
        };
        let results = if op == QuantumOp::Dealloc {
            vec![]
        } else {
            vec![60]
        };
        let instruction = quantum(op, &operands, &results);
        for &operand in &operands {
            for value in [
                TypedValue::U32(u32::MAX),
                TypedValue::I64(i64::from(u32::MAX)),
            ] {
                let mut processor = PhirProcessor::new();
                processor.ssa_values.insert(50, TypedValue::I64(0));
                processor.ssa_values.insert(51, TypedValue::I64(1));
                processor.ssa_values.insert(operand, value);
                let mut builder = ByteMessage::quantum_operations_builder();
                processor
                    .process_instruction(&instruction, &mut builder)
                    .unwrap();
                assert_eq!(processor.get_qubit_count(), index + 1);
                let gates = builder.build().quantum_ops().unwrap();
                assert_eq!(gates.len(), 1);
                assert!(gates[0].qubits.contains(&index.into()));
                if matches!(instruction.operation, Operation::Quantum(QuantumOp::Alloc)) {
                    assert_eq!(
                        processor.ssa_values[&60],
                        TypedValue::I64(i64::from(u32::MAX))
                    );
                }
            }
            for value in [
                TypedValue::U64(u64::from(u32::MAX) + 1),
                TypedValue::I64(i64::from(u32::MAX) + 1),
            ] {
                let mut processor = PhirProcessor::new();
                processor.ssa_values.insert(50, TypedValue::I64(0));
                processor.ssa_values.insert(51, TypedValue::I64(1));
                processor.ssa_values.insert(operand, value);
                let mut builder = ByteMessage::quantum_operations_builder();
                let error = processor
                    .process_instruction(&instruction, &mut builder)
                    .unwrap_err()
                    .to_string();
                assert!(error.contains(&instruction.operation.name()), "{error}");
                assert_eq!(
                    builder.build().quantum_ops().unwrap(),
                    Vec::<pecos_core::Gate>::new()
                );
            }
        }
    }
}

#[test]
fn allocation_arity() {
    let error = pecos_phir::parse_qis_to_quantum(
        "define void @main() {\ncall void @___qalloc(i64 0, i64 1)\nret void\n}",
    )
    .unwrap_err()
    .to_string();
    assert!(error.contains("qis.qalloc"), "{error}");
    let error = PhirProcessor::new()
        .process_instruction(
            &quantum(QuantumOp::Alloc, &[0, 1], &[]),
            &mut ByteMessage::quantum_operations_builder(),
        )
        .unwrap_err()
        .to_string();
    assert!(error.contains("quantum.alloc"), "{error}");
}

#[test]
fn measurement_derived_index_through_memory() {
    assert_eq!(
        run(
            "%slot = alloca i64\n\
         call void @___x(i64 0)\n\
         %m0 = call i1 @___measure(i64 0)\n\
         %index = zext i1 %m0 to i64\n\
         store i64 %index, i64* %slot\n\
         %loaded = load i64, i64* %slot\n\
         call void @___x(i64 %loaded)\n\
         %m1 = call i1 @___measure(i64 1)",
            2,
            0,
        ),
        (vec![true, true], 2)
    );
}
