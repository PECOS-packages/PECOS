/*!
Basic tests for `PhirEngine`

Tests to verify that `PhirEngine` basic functionality works correctly.
*/

use super::engine::PhirEngine;
use super::processor::PhirProcessor;
use crate::ops::{AllocType, ClassicalOp, MemoryOp, Operation, QuantumOp};
use crate::phir::{Block, Instruction, Module, Region, SSAValue};
use crate::region_kinds::RegionKind;
use crate::types::Type;
use pecos_core::Angle64;
use pecos_engines::ClassicalEngine;
use pecos_engines::byte_message::builder::ByteMessageBuilder;
use std::collections::BTreeMap;

/// Create a simple PHIR module for testing
fn create_test_module() -> Module {
    // Create a simple module with an H gate
    let h_instruction = Instruction {
        operation: Operation::Quantum(QuantumOp::H),
        operands: vec![SSAValue { id: 0, version: 0 }],
        results: vec![SSAValue { id: 1, version: 0 }],
        result_types: vec![Type::Qubit],
        regions: vec![],
        attributes: BTreeMap::new(),
        location: None,
    };

    let main_block = Block {
        label: None,
        arguments: vec![],
        operations: vec![h_instruction],
        terminator: None,
        attributes: BTreeMap::new(),
    };

    Module {
        name: "test_module".to_string(),
        attributes: BTreeMap::new(),
        body: Region {
            blocks: vec![main_block],
            kind: RegionKind::SSACFG,
            attributes: BTreeMap::new(),
        },
    }
}

/// Create a Bell state PHIR module for testing
fn create_bell_state_module() -> Module {
    let h_instruction = Instruction {
        operation: Operation::Quantum(QuantumOp::H),
        operands: vec![SSAValue { id: 0, version: 0 }],
        results: vec![SSAValue { id: 2, version: 0 }],
        result_types: vec![Type::Qubit],
        regions: vec![],
        attributes: BTreeMap::new(),
        location: None,
    };

    let cx_instruction = Instruction {
        operation: Operation::Quantum(QuantumOp::CX),
        operands: vec![
            SSAValue { id: 0, version: 0 },
            SSAValue { id: 1, version: 0 },
        ],
        results: vec![SSAValue { id: 3, version: 0 }],
        result_types: vec![Type::Qubit],
        regions: vec![],
        attributes: BTreeMap::new(),
        location: None,
    };

    let measure1_instruction = Instruction {
        operation: Operation::Quantum(QuantumOp::Measure),
        operands: vec![SSAValue { id: 0, version: 0 }],
        results: vec![SSAValue { id: 4, version: 0 }],
        result_types: vec![Type::Bit],
        regions: vec![],
        attributes: BTreeMap::new(),
        location: None,
    };

    let measure2_instruction = Instruction {
        operation: Operation::Quantum(QuantumOp::Measure),
        operands: vec![SSAValue { id: 1, version: 0 }],
        results: vec![SSAValue { id: 5, version: 0 }],
        result_types: vec![Type::Bit],
        regions: vec![],
        attributes: BTreeMap::new(),
        location: None,
    };

    let main_block = Block {
        label: None,
        arguments: vec![],
        operations: vec![
            h_instruction,
            cx_instruction,
            measure1_instruction,
            measure2_instruction,
        ],
        terminator: None,
        attributes: BTreeMap::new(),
    };

    Module {
        name: "bell_state".to_string(),
        attributes: BTreeMap::new(),
        body: Region {
            blocks: vec![main_block],
            kind: RegionKind::SSACFG,
            attributes: BTreeMap::new(),
        },
    }
}

/// Test basic `PhirEngine` functionality
#[test]
fn test_phir_engine_basic() -> Result<(), Box<dyn std::error::Error>> {
    let module = create_test_module();
    let mut engine = PhirEngine::new(module)?;

    // Initially qubit count is 0
    assert_eq!(engine.num_qubits(), 0);

    // Test command generation
    let _commands = engine.generate_commands()?;

    // After processing operations, qubit count should be 1 (only qubit 0 used)
    assert_eq!(engine.num_qubits(), 1);
    println!("Generated commands for H gate");

    // Test that compilation works
    assert!(engine.compile().is_ok());

    // Test that we can get results (even if empty)
    let results = engine.get_results()?;
    println!("Results: {:?}", results.data);

    Ok(())
}

/// Test Bell state circuit
#[test]
fn test_bell_state_circuit() -> Result<(), Box<dyn std::error::Error>> {
    let module = create_bell_state_module();
    let mut engine = PhirEngine::new(module)?;

    // Test that the engine recognizes this as a Bell state circuit
    assert_eq!(engine.module().unwrap().name, "bell_state");

    // Generate commands for the Bell state
    let _commands = engine.generate_commands()?;
    println!("Generated Bell state commands");

    // Test compilation
    assert!(engine.compile().is_ok());

    Ok(())
}

/// Test empty module
#[test]
fn test_empty_module() -> Result<(), Box<dyn std::error::Error>> {
    let empty_module = Module {
        name: "empty".to_string(),
        attributes: BTreeMap::new(),
        body: Region {
            blocks: vec![Block {
                label: None,
                arguments: vec![],
                operations: vec![],
                terminator: None,
                attributes: BTreeMap::new(),
            }],
            kind: RegionKind::SSACFG,
            attributes: BTreeMap::new(),
        },
    };

    let mut engine = PhirEngine::new(empty_module)?;

    // Should handle empty modules gracefully
    let _commands = engine.generate_commands()?;
    println!("Generated commands for empty module");

    assert!(engine.compile().is_ok());

    Ok(())
}

/// Test engine reset functionality
#[test]
fn test_engine_reset() -> Result<(), Box<dyn std::error::Error>> {
    let module = create_test_module();
    let mut engine = PhirEngine::new(module)?;

    // Generate commands
    let _commands1 = engine.generate_commands()?;

    // Reset the engine
    engine.reset()?;

    // Should be able to generate commands again
    let _commands2 = engine.generate_commands()?;

    Ok(())
}

/// Test cloning functionality
#[test]
fn test_engine_clone() -> Result<(), Box<dyn std::error::Error>> {
    let module = create_test_module();
    let engine1 = PhirEngine::new(module)?;

    // Clone the engine
    let engine2 = engine1.clone();

    // Both should have the same module
    assert_eq!(
        engine1.module().unwrap().name,
        engine2.module().unwrap().name
    );
    assert_eq!(engine1.num_qubits(), engine2.num_qubits());

    Ok(())
}

/// Test measurement handling
#[test]
fn test_measurement_handling() -> Result<(), Box<dyn std::error::Error>> {
    let module = create_bell_state_module();
    let mut engine = PhirEngine::new(module)?;

    // Generate commands that include measurements
    let _commands = engine.generate_commands()?;

    // Create a mock measurement message
    let mut builder = ByteMessageBuilder::new();
    let _ = builder.for_outcomes();
    builder.add_outcomes(&[1, 0]); // Mock measurement results
    let measurement_msg = builder.build();

    // Handle the measurements
    let result = engine.handle_measurements(measurement_msg);

    // Should not error (even if results aren't processed perfectly yet)
    assert!(result.is_ok());

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// Helpers for processor-level tests
// ──────────────────────────────────────────────────────────────────────

/// Create a single instruction
fn instr(
    op: Operation,
    operands: Vec<u32>,
    results: Vec<u32>,
    result_types: Vec<Type>,
) -> Instruction {
    Instruction {
        operation: op,
        operands: operands
            .into_iter()
            .map(|id| SSAValue { id, version: 0 })
            .collect(),
        results: results
            .into_iter()
            .map(|id| SSAValue { id, version: 0 })
            .collect(),
        result_types,
        regions: vec![],
        attributes: BTreeMap::new(),
        location: None,
    }
}

/// Run a sequence of instructions through the processor, returning the processor state
fn run_instructions(instructions: Vec<Instruction>) -> PhirProcessor {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();
    for instr in &instructions {
        processor
            .process_instruction(instr, &mut builder)
            .expect("instruction should succeed");
    }
    processor
}

// ──────────────────────────────────────────────────────────────────────
// Rotation gate processor tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_rz_gate() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    let rz_instr = instr(
        Operation::Quantum(QuantumOp::RZ(Angle64::from_radians(
            std::f64::consts::FRAC_PI_2,
        ))),
        vec![0],
        vec![1],
        vec![Type::Qubit],
    );

    let result = processor.process_instruction(&rz_instr, &mut builder);
    assert!(result.is_ok());
    assert!(result.unwrap()); // Should generate quantum instructions
    assert_eq!(processor.get_qubit_count(), 1);
}

#[test]
fn test_processor_r1xy_gate() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    let r1xy_instr = instr(
        Operation::Quantum(QuantumOp::R1XY(
            Angle64::from_radians(std::f64::consts::FRAC_PI_2),
            Angle64::ZERO,
        )),
        vec![0],
        vec![1],
        vec![Type::Qubit],
    );

    let result = processor.process_instruction(&r1xy_instr, &mut builder);
    assert!(result.is_ok());
    assert!(result.unwrap());
}

#[test]
fn test_processor_rx_ry_gates() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    let rx_instr = instr(
        Operation::Quantum(QuantumOp::RX(Angle64::from_radians(std::f64::consts::PI))),
        vec![0],
        vec![1],
        vec![Type::Qubit],
    );
    let ry_instr = instr(
        Operation::Quantum(QuantumOp::RY(Angle64::from_radians(
            std::f64::consts::FRAC_PI_4,
        ))),
        vec![0],
        vec![2],
        vec![Type::Qubit],
    );

    assert!(
        processor
            .process_instruction(&rx_instr, &mut builder)
            .unwrap()
    );
    assert!(
        processor
            .process_instruction(&ry_instr, &mut builder)
            .unwrap()
    );
}

#[test]
fn test_processor_u3_gate() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    let u3_instr = instr(
        Operation::Quantum(QuantumOp::U3(
            Angle64::from_radians(std::f64::consts::FRAC_PI_2),
            Angle64::ZERO,
            Angle64::from_radians(std::f64::consts::PI),
        )),
        vec![0],
        vec![1],
        vec![Type::Qubit],
    );

    assert!(
        processor
            .process_instruction(&u3_instr, &mut builder)
            .unwrap()
    );
}

// ──────────────────────────────────────────────────────────────────────
// Two-qubit gate processor tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_swap_gate() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    let swap_instr = instr(
        Operation::Quantum(QuantumOp::SWAP),
        vec![0, 1],
        vec![2],
        vec![Type::Qubit],
    );

    assert!(
        processor
            .process_instruction(&swap_instr, &mut builder)
            .unwrap()
    );
    assert_eq!(processor.get_qubit_count(), 2);
}

#[test]
fn test_processor_rzz_gate() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    let rzz_instr = instr(
        Operation::Quantum(QuantumOp::RZZ(Angle64::from_radians(
            std::f64::consts::FRAC_PI_4,
        ))),
        vec![0, 1],
        vec![2],
        vec![Type::Qubit],
    );

    assert!(
        processor
            .process_instruction(&rzz_instr, &mut builder)
            .unwrap()
    );
    assert_eq!(processor.get_qubit_count(), 2);
}

#[test]
fn test_processor_cphase_gate() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    let cp_instr = instr(
        Operation::Quantum(QuantumOp::CPhase(Angle64::from_radians(
            std::f64::consts::PI,
        ))),
        vec![0, 1],
        vec![2],
        vec![Type::Qubit],
    );

    assert!(
        processor
            .process_instruction(&cp_instr, &mut builder)
            .unwrap()
    );
}

#[test]
fn test_processor_cz_gate() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    let cz_instr = instr(
        Operation::Quantum(QuantumOp::CZ),
        vec![0, 1],
        vec![2],
        vec![Type::Qubit],
    );

    assert!(
        processor
            .process_instruction(&cz_instr, &mut builder)
            .unwrap()
    );
}

// ──────────────────────────────────────────────────────────────────────
// Fixed single-qubit gate tests (S, Sdg, T, Tdg)
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_s_sdg_t_tdg_gates() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    for (op, name) in [
        (QuantumOp::S, "S"),
        (QuantumOp::Sdg, "Sdg"),
        (QuantumOp::T, "T"),
        (QuantumOp::Tdg, "Tdg"),
    ] {
        let gate_instr = instr(Operation::Quantum(op), vec![0], vec![10], vec![Type::Qubit]);
        let result = processor.process_instruction(&gate_instr, &mut builder);
        assert!(result.is_ok(), "{name} gate failed: {:?}", result.err());
        assert!(
            result.unwrap(),
            "{name} gate should produce quantum instructions"
        );
    }
}

// ──────────────────────────────────────────────────────────────────────
// Resource management tests (Alloc, Dealloc, Reset, InitZero)
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_alloc_dealloc() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    let alloc_instr = instr(
        Operation::Quantum(QuantumOp::Alloc),
        vec![],
        vec![0],
        vec![Type::Qubit],
    );
    assert!(
        processor
            .process_instruction(&alloc_instr, &mut builder)
            .unwrap()
    );
    assert_eq!(processor.get_qubit_count(), 1);

    let dealloc_instr = instr(
        Operation::Quantum(QuantumOp::Dealloc),
        vec![0],
        vec![],
        vec![],
    );
    assert!(
        processor
            .process_instruction(&dealloc_instr, &mut builder)
            .unwrap()
    );
}

#[test]
fn test_processor_reset() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    let reset_instr = instr(
        Operation::Quantum(QuantumOp::Reset),
        vec![0],
        vec![],
        vec![],
    );
    assert!(
        processor
            .process_instruction(&reset_instr, &mut builder)
            .unwrap()
    );
}

#[test]
fn test_processor_init_zero() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    let init_instr = instr(
        Operation::Quantum(QuantumOp::InitZero),
        vec![0],
        vec![],
        vec![],
    );
    assert!(
        processor
            .process_instruction(&init_instr, &mut builder)
            .unwrap()
    );
}

// ──────────────────────────────────────────────────────────────────────
// Classical operation processor tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_const_int() {
    let processor = run_instructions(vec![instr(
        Operation::Classical(ClassicalOp::ConstInt(42)),
        vec![],
        vec![0],
        vec![Type::Int(crate::types::IntWidth::I64)],
    )]);
    assert_eq!(
        processor.ssa_values.get(&0),
        Some(&super::environment::TypedValue::U32(42))
    );
}

#[test]
fn test_processor_const_float() {
    let processor = run_instructions(vec![instr(
        Operation::Classical(ClassicalOp::ConstFloat(1.234)),
        vec![],
        vec![0],
        vec![Type::Float(crate::types::FloatPrecision::F64)],
    )]);
    assert_eq!(
        processor.ssa_values.get(&0),
        Some(&super::environment::TypedValue::F64(1.234))
    );
}

#[test]
fn test_processor_const_bool() {
    let processor = run_instructions(vec![instr(
        Operation::Classical(ClassicalOp::ConstBool(true)),
        vec![],
        vec![0],
        vec![Type::Bool],
    )]);
    assert_eq!(
        processor.ssa_values.get(&0),
        Some(&super::environment::TypedValue::Bool(true))
    );
}

#[test]
fn test_processor_add() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstInt(10)),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(20)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Add),
            vec![0, 1],
            vec![2],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&2),
        Some(&super::environment::TypedValue::U32(30))
    );
}

#[test]
fn test_processor_sub() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstInt(30)),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(12)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Sub),
            vec![0, 1],
            vec![2],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&2),
        Some(&super::environment::TypedValue::U32(18))
    );
}

#[test]
fn test_processor_mul() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstInt(6)),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(7)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Mul),
            vec![0, 1],
            vec![2],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&2),
        Some(&super::environment::TypedValue::U32(42))
    );
}

#[test]
fn test_processor_div_and_mod() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstInt(17)),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(5)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Div),
            vec![0, 1],
            vec![2],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Mod),
            vec![0, 1],
            vec![3],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&2),
        Some(&super::environment::TypedValue::U32(3))
    );
    assert_eq!(
        processor.ssa_values.get(&3),
        Some(&super::environment::TypedValue::U32(2))
    );
}

#[test]
fn test_processor_div_by_zero() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstInt(42)),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(0)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Div),
            vec![0, 1],
            vec![2],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    // Division by zero should produce 0
    assert_eq!(
        processor.ssa_values.get(&2),
        Some(&super::environment::TypedValue::U32(0))
    );
}

// ──────────────────────────────────────────────────────────────────────
// Bitwise operation tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_bitwise_and_or_xor() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstInt(0b1100)),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(0b1010)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::And),
            vec![0, 1],
            vec![2],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Or),
            vec![0, 1],
            vec![3],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Xor),
            vec![0, 1],
            vec![4],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&2),
        Some(&super::environment::TypedValue::U32(0b1000))
    ); // AND
    assert_eq!(
        processor.ssa_values.get(&3),
        Some(&super::environment::TypedValue::U32(0b1110))
    ); // OR
    assert_eq!(
        processor.ssa_values.get(&4),
        Some(&super::environment::TypedValue::U32(0b0110))
    ); // XOR
}

#[test]
fn test_processor_shl_shr() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstInt(1)),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Shl(3)),
            vec![0],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(16)),
            vec![],
            vec![2],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Shr(2)),
            vec![2],
            vec![3],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&1),
        Some(&super::environment::TypedValue::U32(8))
    ); // 1 << 3
    assert_eq!(
        processor.ssa_values.get(&3),
        Some(&super::environment::TypedValue::U32(4))
    ); // 16 >> 2
}

#[test]
fn test_processor_shl_shr_binary_mode() {
    // Test Shl/Shr with two operands (binary mode), matching how the QIS parser emits them
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstInt(1)),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(3)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        // Shl(0) with two operands: value=SSA0, shift_amount=SSA1
        instr(
            Operation::Classical(ClassicalOp::Shl(0)),
            vec![0, 1],
            vec![2],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(16)),
            vec![],
            vec![3],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(2)),
            vec![],
            vec![4],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        // Shr(0) with two operands: value=SSA3, shift_amount=SSA4
        instr(
            Operation::Classical(ClassicalOp::Shr(0)),
            vec![3, 4],
            vec![5],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&2),
        Some(&super::environment::TypedValue::U32(8))
    ); // 1 << 3
    assert_eq!(
        processor.ssa_values.get(&5),
        Some(&super::environment::TypedValue::U32(4))
    ); // 16 >> 2
}

#[test]
fn test_processor_not() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstBool(true)),
            vec![],
            vec![0],
            vec![Type::Bool],
        ),
        instr(
            Operation::Classical(ClassicalOp::Not),
            vec![0],
            vec![1],
            vec![Type::Bool],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&1),
        Some(&super::environment::TypedValue::Bool(false))
    );
}

#[test]
fn test_processor_neg_float() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstFloat(42.0)),
            vec![],
            vec![0],
            vec![Type::Float(crate::types::FloatPrecision::F64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Neg),
            vec![0],
            vec![1],
            vec![Type::Float(crate::types::FloatPrecision::F64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&1),
        Some(&super::environment::TypedValue::F64(-42.0))
    );
}

// ──────────────────────────────────────────────────────────────────────
// Comparison operation tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_comparisons() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstInt(10)),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(20)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Eq),
            vec![0, 1],
            vec![2],
            vec![Type::Bool],
        ),
        instr(
            Operation::Classical(ClassicalOp::Ne),
            vec![0, 1],
            vec![3],
            vec![Type::Bool],
        ),
        instr(
            Operation::Classical(ClassicalOp::Lt),
            vec![0, 1],
            vec![4],
            vec![Type::Bool],
        ),
        instr(
            Operation::Classical(ClassicalOp::Le),
            vec![0, 1],
            vec![5],
            vec![Type::Bool],
        ),
        instr(
            Operation::Classical(ClassicalOp::Gt),
            vec![0, 1],
            vec![6],
            vec![Type::Bool],
        ),
        instr(
            Operation::Classical(ClassicalOp::Ge),
            vec![0, 1],
            vec![7],
            vec![Type::Bool],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&2),
        Some(&super::environment::TypedValue::Bool(false))
    ); // 10 == 20
    assert_eq!(
        processor.ssa_values.get(&3),
        Some(&super::environment::TypedValue::Bool(true))
    ); // 10 != 20
    assert_eq!(
        processor.ssa_values.get(&4),
        Some(&super::environment::TypedValue::Bool(true))
    ); // 10 < 20
    assert_eq!(
        processor.ssa_values.get(&5),
        Some(&super::environment::TypedValue::Bool(true))
    ); // 10 <= 20
    assert_eq!(
        processor.ssa_values.get(&6),
        Some(&super::environment::TypedValue::Bool(false))
    ); // 10 > 20
    assert_eq!(
        processor.ssa_values.get(&7),
        Some(&super::environment::TypedValue::Bool(false))
    ); // 10 >= 20
}

#[test]
fn test_processor_comparisons_equal() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstInt(5)),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(5)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Eq),
            vec![0, 1],
            vec![2],
            vec![Type::Bool],
        ),
        instr(
            Operation::Classical(ClassicalOp::Le),
            vec![0, 1],
            vec![3],
            vec![Type::Bool],
        ),
        instr(
            Operation::Classical(ClassicalOp::Ge),
            vec![0, 1],
            vec![4],
            vec![Type::Bool],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&2),
        Some(&super::environment::TypedValue::Bool(true))
    ); // 5 == 5
    assert_eq!(
        processor.ssa_values.get(&3),
        Some(&super::environment::TypedValue::Bool(true))
    ); // 5 <= 5
    assert_eq!(
        processor.ssa_values.get(&4),
        Some(&super::environment::TypedValue::Bool(true))
    ); // 5 >= 5
}

// ──────────────────────────────────────────────────────────────────────
// Select operation test
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_select() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstBool(true)),
            vec![],
            vec![0],
            vec![Type::Bool],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(100)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(200)),
            vec![],
            vec![2],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Select),
            vec![0, 1, 2],
            vec![3],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&3),
        Some(&super::environment::TypedValue::U32(100))
    );
}

#[test]
fn test_processor_select_false() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstBool(false)),
            vec![],
            vec![0],
            vec![Type::Bool],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(100)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(200)),
            vec![],
            vec![2],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Select),
            vec![0, 1, 2],
            vec![3],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&3),
        Some(&super::environment::TypedValue::U32(200))
    );
}

// ──────────────────────────────────────────────────────────────────────
// Float operation tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_float_arithmetic() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstFloat(3.0)),
            vec![],
            vec![0],
            vec![Type::Float(crate::types::FloatPrecision::F64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstFloat(2.0)),
            vec![],
            vec![1],
            vec![Type::Float(crate::types::FloatPrecision::F64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::FAdd),
            vec![0, 1],
            vec![2],
            vec![Type::Float(crate::types::FloatPrecision::F64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::FSub),
            vec![0, 1],
            vec![3],
            vec![Type::Float(crate::types::FloatPrecision::F64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::FMul),
            vec![0, 1],
            vec![4],
            vec![Type::Float(crate::types::FloatPrecision::F64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::FDiv),
            vec![0, 1],
            vec![5],
            vec![Type::Float(crate::types::FloatPrecision::F64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&2),
        Some(&super::environment::TypedValue::F64(5.0))
    ); // 3 + 2
    assert_eq!(
        processor.ssa_values.get(&3),
        Some(&super::environment::TypedValue::F64(1.0))
    ); // 3 - 2
    assert_eq!(
        processor.ssa_values.get(&4),
        Some(&super::environment::TypedValue::F64(6.0))
    ); // 3 * 2
    assert_eq!(
        processor.ssa_values.get(&5),
        Some(&super::environment::TypedValue::F64(1.5))
    ); // 3 / 2
}

#[test]
fn test_processor_fneg() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstFloat(7.5)),
            vec![],
            vec![0],
            vec![Type::Float(crate::types::FloatPrecision::F64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::FNeg),
            vec![0],
            vec![1],
            vec![Type::Float(crate::types::FloatPrecision::F64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&1),
        Some(&super::environment::TypedValue::F64(-7.5))
    );
}

#[test]
fn test_processor_fdiv_by_zero() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstFloat(1.0)),
            vec![],
            vec![0],
            vec![Type::Float(crate::types::FloatPrecision::F64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstFloat(0.0)),
            vec![],
            vec![1],
            vec![Type::Float(crate::types::FloatPrecision::F64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::FDiv),
            vec![0, 1],
            vec![2],
            vec![Type::Float(crate::types::FloatPrecision::F64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&2),
        Some(&super::environment::TypedValue::F64(0.0))
    );
}

// ──────────────────────────────────────────────────────────────────────
// Memory operation tests (alloca/load/store)
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_memory_alloc_store_load() {
    let processor = run_instructions(vec![
        // alloca i64 -> ptr %0
        instr(
            Operation::Memory(MemoryOp::Alloc(AllocType::Scalar(Type::Int(
                crate::types::IntWidth::I64,
            )))),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        // store 42 into %0
        instr(
            Operation::Classical(ClassicalOp::ConstInt(42)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Memory(MemoryOp::Store),
            vec![1, 0], // value, pointer
            vec![],
            vec![],
        ),
        // load from %0 -> %2
        instr(
            Operation::Memory(MemoryOp::Load),
            vec![0], // pointer
            vec![2],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&2),
        Some(&super::environment::TypedValue::U32(42))
    );
}

#[test]
fn test_processor_memory_overwrite() {
    let processor = run_instructions(vec![
        instr(
            Operation::Memory(MemoryOp::Alloc(AllocType::Scalar(Type::Int(
                crate::types::IntWidth::I64,
            )))),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        // Store 10
        instr(
            Operation::Classical(ClassicalOp::ConstInt(10)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Memory(MemoryOp::Store),
            vec![1, 0],
            vec![],
            vec![],
        ),
        // Store 99 (overwrite)
        instr(
            Operation::Classical(ClassicalOp::ConstInt(99)),
            vec![],
            vec![2],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Memory(MemoryOp::Store),
            vec![2, 0],
            vec![],
            vec![],
        ),
        // Load - should get 99
        instr(
            Operation::Memory(MemoryOp::Load),
            vec![0],
            vec![3],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&3),
        Some(&super::environment::TypedValue::U32(99))
    );
}

// ──────────────────────────────────────────────────────────────────────
// Assign operation test
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_assign() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstInt(42)),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Assign),
            vec![0],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&1),
        Some(&super::environment::TypedValue::U32(42))
    );
}

// ──────────────────────────────────────────────────────────────────────
// Bitcast operation tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_bitcast_bool_to_int_true() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstBool(true)),
            vec![],
            vec![0],
            vec![Type::Bool],
        ),
        instr(
            Operation::Classical(ClassicalOp::Bitcast),
            vec![0],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&1),
        Some(&super::environment::TypedValue::U32(1))
    );
}

#[test]
fn test_processor_bitcast_bool_to_int_false() {
    let processor = run_instructions(vec![
        instr(
            Operation::Classical(ClassicalOp::ConstBool(false)),
            vec![],
            vec![0],
            vec![Type::Bool],
        ),
        instr(
            Operation::Classical(ClassicalOp::Bitcast),
            vec![0],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ]);
    assert_eq!(
        processor.ssa_values.get(&1),
        Some(&super::environment::TypedValue::U32(0))
    );
}

// ──────────────────────────────────────────────────────────────────────
// Result/export operation tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_result_export() {
    use crate::phir::AttributeValue;

    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    // Create a constant
    let const_instr = instr(
        Operation::Classical(ClassicalOp::ConstInt(42)),
        vec![],
        vec![0],
        vec![Type::Int(crate::types::IntWidth::I64)],
    );
    processor
        .process_instruction(&const_instr, &mut builder)
        .unwrap();

    // Create a Result instruction with export_name attribute
    let mut attrs = BTreeMap::new();
    attrs.insert(
        "export_name".to_string(),
        AttributeValue::String("my_result".to_string()),
    );
    let result_instr = Instruction {
        operation: Operation::Classical(ClassicalOp::Result),
        operands: vec![SSAValue { id: 0, version: 0 }],
        results: vec![],
        result_types: vec![],
        regions: vec![],
        attributes: attrs,
        location: None,
    };
    processor
        .process_instruction(&result_instr, &mut builder)
        .unwrap();

    let exports = processor.get_export_results();
    assert_eq!(
        exports.get("my_result"),
        Some(&super::environment::TypedValue::U32(42))
    );
}

// ──────────────────────────────────────────────────────────────────────
// Operand error tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_wrong_operand_count_single_qubit() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    // H gate with 2 operands should fail
    let bad_instr = instr(
        Operation::Quantum(QuantumOp::H),
        vec![0, 1],
        vec![2],
        vec![Type::Qubit],
    );
    assert!(
        processor
            .process_instruction(&bad_instr, &mut builder)
            .is_err()
    );
}

#[test]
fn test_processor_wrong_operand_count_two_qubit() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    // CX gate with 1 operand should fail
    let bad_instr = instr(
        Operation::Quantum(QuantumOp::CX),
        vec![0],
        vec![2],
        vec![Type::Qubit],
    );
    assert!(
        processor
            .process_instruction(&bad_instr, &mut builder)
            .is_err()
    );
}

// ──────────────────────────────────────────────────────────────────────
// Unimplemented quantum op error tests
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_processor_unimplemented_quantum_op() {
    let mut processor = PhirProcessor::new();
    let mut builder = ByteMessageBuilder::new();

    let toffoli_instr = instr(
        Operation::Quantum(QuantumOp::Toffoli),
        vec![0, 1, 2],
        vec![3],
        vec![Type::Qubit],
    );
    let result = processor.process_instruction(&toffoli_instr, &mut builder);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("not yet implemented"),
        "Error should mention unimplemented: {msg}"
    );
}

// ──────────────────────────────────────────────────────────────────────
// Compile error path test
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_engine_compile_empty_name_error() -> Result<(), Box<dyn std::error::Error>> {
    let module = Module {
        name: String::new(),
        attributes: BTreeMap::new(),
        body: Region {
            blocks: vec![Block {
                label: None,
                arguments: vec![],
                operations: vec![],
                terminator: None,
                attributes: BTreeMap::new(),
            }],
            kind: RegionKind::SSACFG,
            attributes: BTreeMap::new(),
        },
    };

    let engine = PhirEngine::new(module)?;
    let result = engine.compile();
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("empty"),
        "Error should mention empty name: {msg}"
    );

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// Classical-only module (no quantum messages)
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_engine_classical_only_module() -> Result<(), Box<dyn std::error::Error>> {
    let instructions = vec![
        instr(
            Operation::Classical(ClassicalOp::ConstInt(10)),
            vec![],
            vec![0],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::ConstInt(20)),
            vec![],
            vec![1],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
        instr(
            Operation::Classical(ClassicalOp::Add),
            vec![0, 1],
            vec![2],
            vec![Type::Int(crate::types::IntWidth::I64)],
        ),
    ];

    let module = Module {
        name: "classical_only".to_string(),
        attributes: BTreeMap::new(),
        body: Region {
            blocks: vec![Block {
                label: None,
                arguments: vec![],
                operations: instructions,
                terminator: None,
                attributes: BTreeMap::new(),
            }],
            kind: RegionKind::SSACFG,
            attributes: BTreeMap::new(),
        },
    };

    let mut engine = PhirEngine::new(module)?;
    let commands = engine.generate_commands()?;
    // Classical-only module should finish after one generate_commands call
    assert!(engine.finished);
    // The message should contain no quantum gate operations
    assert!(
        commands.is_empty().unwrap_or(true),
        "Classical-only module should not produce quantum operations"
    );
    assert!(engine.compile().is_ok());

    // Verify classical ops were processed
    assert_eq!(
        engine.processor.ssa_values.get(&2),
        Some(&super::environment::TypedValue::U32(30))
    );

    Ok(())
}

// ──────────────────────────────────────────────────────────────────────
// Module-level engine test with rotation gates
// ──────────────────────────────────────────────────────────────────────

#[test]
fn test_engine_rz_rxy_module() -> Result<(), Box<dyn std::error::Error>> {
    // Build a module with Alloc, RZ, R1XY, Measure, Dealloc
    let instructions = vec![
        instr(
            Operation::Quantum(QuantumOp::Alloc),
            vec![],
            vec![0],
            vec![Type::Qubit],
        ),
        instr(
            Operation::Quantum(QuantumOp::RZ(Angle64::from_radians(
                std::f64::consts::FRAC_PI_2,
            ))),
            vec![0],
            vec![1],
            vec![Type::Qubit],
        ),
        instr(
            Operation::Quantum(QuantumOp::R1XY(
                Angle64::from_radians(std::f64::consts::FRAC_PI_2),
                Angle64::ZERO,
            )),
            vec![0],
            vec![2],
            vec![Type::Qubit],
        ),
        instr(
            Operation::Quantum(QuantumOp::RZ(Angle64::from_radians(
                std::f64::consts::FRAC_PI_2,
            ))),
            vec![0],
            vec![3],
            vec![Type::Qubit],
        ),
        instr(
            Operation::Quantum(QuantumOp::Measure),
            vec![0],
            vec![4],
            vec![Type::Bit],
        ),
        instr(
            Operation::Quantum(QuantumOp::Dealloc),
            vec![0],
            vec![],
            vec![],
        ),
    ];

    let main_block = Block {
        label: None,
        arguments: vec![],
        operations: instructions,
        terminator: None,
        attributes: BTreeMap::new(),
    };

    let module = Module {
        name: "rz_rxy_test".to_string(),
        attributes: BTreeMap::new(),
        body: Region {
            blocks: vec![main_block],
            kind: RegionKind::SSACFG,
            attributes: BTreeMap::new(),
        },
    };

    let mut engine = PhirEngine::new(module)?;
    let _commands = engine.generate_commands()?;
    assert!(engine.compile().is_ok());
    assert_eq!(engine.num_qubits(), 1);

    Ok(())
}
