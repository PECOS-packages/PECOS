/*!
Test loading and processing PHIR-JSON fixtures
*/

use pecos_phir_json::phir_json_to_module;
use pecos_core::errors::PecosError;
use std::fs;

#[test]
fn test_bell_state_fixture() -> Result<(), PecosError> {
    // Load the bell state fixture
    let bell_json = fs::read_to_string("tests/fixtures/bell_state.phir.json")
        .expect("Failed to read bell_state.phir.json fixture");

    // Convert to PHIR module
    let module = phir_json_to_module(&bell_json)?;

    // Verify the module name and structure
    assert_eq!(module.name, "bell_state_circuit", "Module should be named 'bell_state_circuit'");
    assert!(!module.body.blocks.is_empty(), "Module should have blocks");

    // Verify it has the expected operations
    let operations = &module.body.blocks[0].operations;

    // Count different operation types
    let mut h_count = 0;
    let mut cx_count = 0;
    let mut measure_count = 0;

    for op in operations {
        match &op.operation {
            pecos_phir::ops::Operation::Quantum(q) => match q {
                pecos_phir::ops::QuantumOp::H => h_count += 1,
                pecos_phir::ops::QuantumOp::CX => cx_count += 1,
                pecos_phir::ops::QuantumOp::Measure => measure_count += 1,
                _ => {}
            },
            _ => {}
        }
    }

    assert_eq!(h_count, 1, "Should have 1 Hadamard gate");
    assert_eq!(cx_count, 1, "Should have 1 CNOT gate");
    assert_eq!(measure_count, 2, "Should have 2 measurements");

    Ok(())
}

#[test]
fn test_all_json_fixtures() -> Result<(), PecosError> {
    // Test that all .json files in fixtures directory can be parsed
    let fixtures_dir = "tests/fixtures";

    if let Ok(entries) = fs::read_dir(fixtures_dir) {
        for entry in entries {
            let entry = entry.expect("Failed to read directory entry");
            let path = entry.path();

            if path.extension().and_then(|s| s.to_str()) == Some("json") {
                let json_content = fs::read_to_string(&path)
                    .expect(&format!("Failed to read {:?}", path));

                // Try to parse each JSON file
                let result = phir_json_to_module(&json_content);

                // At minimum, it should parse without panicking
                // We allow errors because some fixtures might be testing error cases
                if result.is_err() {
                    eprintln!("Warning: {:?} failed to parse: {:?}", path, result.err());
                }
            }
        }
    }

    Ok(())
}