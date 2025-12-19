#![cfg(feature = "runtime")]

use pecos::prelude::*;
use std::fs;

#[test]
fn test_setup_engine_for_program() -> Result<(), PecosError> {
    // Create temporary directories for our files
    let temp_dir = tempfile::tempdir().map_err(|e| PecosError::IO(std::io::Error::other(e)))?;

    // Create QASM file with proper extension
    let qasm_path = temp_dir.path().join("test_program.qasm");
    fs::write(
        &qasm_path,
        r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        measure q -> c;
    "#,
    )
    .map_err(PecosError::IO)?;

    // Create JSON/PHIR file with proper extension
    let phir_path = temp_dir.path().join("test_program.phir.json");
    fs::write(
        &phir_path,
        r#"{
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {
            "description": "Test PHIR program"
        },
        "ops": [
            {
                "data": "qvar_define",
                "data_type": "qubits",
                "variable": "q",
                "size": 2
            },
            {
                "data": "cvar_define",
                "data_type": "i64",
                "variable": "c",
                "size": 2
            },
            {
                "cop": "Result",
                "args": ["c"],
                "returns": ["result"]
            }
        ]
    }"#,
    )
    .map_err(PecosError::IO)?;

    // Detect program types
    let qasm_type = detect_program_type(&qasm_path)?;
    let phir_type = detect_program_type(&phir_path)?;

    // Setup engines
    let qasm_engine = setup_engine_for_program(qasm_type, &qasm_path, Some(42))?;
    let phir_json_engine = setup_engine_for_program(phir_type, &phir_path, None)?;

    // Verify engine setup
    assert_eq!(qasm_engine.num_qubits(), 2);
    assert_eq!(phir_json_engine.num_qubits(), 2);

    Ok(())
}
