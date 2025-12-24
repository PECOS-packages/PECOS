/*!
Basic tests for `PhirEngine`

Tests to verify that `PhirEngine` basic functionality works correctly.
*/

use super::engine::PhirEngine;
use crate::ops::{Operation, QuantumOp};
use crate::phir::{Block, Instruction, Module, Region, SSAValue};
use crate::region_kinds::RegionKind;
use crate::types::Type;
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
