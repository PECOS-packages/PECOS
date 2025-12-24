/*!
Test the improved PHIR-JSON to PHIR converter functionality

This test was converted from examples/test_improved_converter.rs
*/

use pecos_phir_json::phir_json_to_module;
use pecos_core::errors::PecosError;

#[test]
fn test_converter_bell_state_ssa_flow() -> Result<(), PecosError> {
    let bell_json = r#"{
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {"description": "Bell state"},
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "i64", "variable": "m", "size": 2},
            {"qop": "H", "args": [["q", 0]]},
            {"qop": "CX", "args": [["q", 0], ["q", 1]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
            {"cop": "Result", "args": ["m"], "returns": ["c"]}
        ]
    }"#;

    let module = phir_json_to_module(bell_json)?;

    // Verify the module structure
    assert!(!module.body.blocks.is_empty(), "Module should have at least one block");
    let operations = &module.body.blocks[0].operations;

    // The converter should generate additional operations for bit combining
    // Original has 7 ops, but converter adds bitwise operations for measurements
    assert!(operations.len() > 7, "Converter should add bit-combining operations");

    // Count measurement operations and verify they have proper SSA values
    let mut measure_count = 0;
    let mut has_bitcast = false;
    let mut has_shift = false;
    let mut has_or = false;

    for op in operations {
        match &op.operation {
            pecos_phir::ops::Operation::Quantum(pecos_phir::ops::QuantumOp::Measure) => {
                measure_count += 1;
                // Each measure should have one operand (qubit) and one result
                assert_eq!(op.operands.len(), 1, "Measure should have one operand");
                assert_eq!(op.results.len(), 1, "Measure should have one result");
            }
            pecos_phir::ops::Operation::Classical(classical_op) => {
                match classical_op {
                    pecos_phir::ops::ClassicalOp::Bitcast => has_bitcast = true,
                    pecos_phir::ops::ClassicalOp::Shl(_) => has_shift = true,
                    pecos_phir::ops::ClassicalOp::Or => has_or = true,
                    _ => {}
                }
            }
            _ => {}
        }
    }

    assert_eq!(measure_count, 2, "Should have 2 measurement operations");
    assert!(has_bitcast, "Should have bitcast operations for type conversion");
    assert!(has_shift, "Should have shift operation for bit positioning");
    assert!(has_or, "Should have OR operation for bit combining");

    Ok(())
}

#[test]
fn test_converter_single_qubit_circuit() -> Result<(), PecosError> {
    let single_qubit_json = r#"{
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {"description": "Single qubit circuit"},
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 1},
            {"data": "cvar_define", "data_type": "i64", "variable": "m", "size": 1},
            {"qop": "H", "args": [["q", 0]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"cop": "Result", "args": ["m"], "returns": ["result"]}
        ]
    }"#;

    let module = phir_json_to_module(single_qubit_json)?;

    // Verify basic structure
    assert!(!module.body.blocks.is_empty());
    let operations = &module.body.blocks[0].operations;

    // Should have at least the original operations
    assert!(operations.len() >= 5, "Should have at least 5 operations");

    // Verify we have the expected quantum operations
    let quantum_ops: Vec<_> = operations.iter()
        .filter_map(|op| match &op.operation {
            pecos_phir::ops::Operation::Quantum(q) => Some(q),
            _ => None
        })
        .collect();

    assert!(quantum_ops.iter().any(|op| matches!(op, pecos_phir::ops::QuantumOp::H)));
    assert!(quantum_ops.iter().any(|op| matches!(op, pecos_phir::ops::QuantumOp::Measure)));

    Ok(())
}

#[test]
fn test_converter_invalid_json() {
    let invalid_json = r#"{
        "format": "PHIR/JSON",
        "version": "999.0.0",
        "ops": "not an array"
    }"#;

    let result = phir_json_to_module(invalid_json);
    assert!(result.is_err(), "Should fail on invalid JSON structure");
}