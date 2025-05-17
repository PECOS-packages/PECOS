use pecos::prelude::*;

#[test]
fn test_setup_qasm_engine() -> Result<(), PecosError> {
    // Create a temporary file with a simple QASM program
    let mut file =
        tempfile::NamedTempFile::new().map_err(|e| PecosError::IO(std::io::Error::other(e)))?;
    let qasm_content = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
    "#;
    std::io::Write::write_all(&mut file, qasm_content.as_bytes()).map_err(PecosError::IO)?;

    // Set up the QASM engine without a seed
    let engine = setup_qasm_engine(file.path(), None)?;

    // Verify that we can query the number of qubits
    let num_qubits = engine.num_qubits();
    assert_eq!(num_qubits, 1, "Should have 1 qubit");

    // Set up the QASM engine with a specific seed
    let engine_with_seed = setup_qasm_engine(file.path(), Some(42))?;

    // Verify that the seeded engine also reports correct qubit count
    let seeded_num_qubits = engine_with_seed.num_qubits();
    assert_eq!(seeded_num_qubits, 1, "Seeded engine should have 1 qubit");

    Ok(())
}
