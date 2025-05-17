use pecos::prelude::*;
use pecos_qasm::QASMEngine;
use std::str::FromStr;

#[test]
fn test_qelib1_inc_available_from_external_crate() -> Result<(), PecosError> {
    // Test that qelib1.inc is available when used from an external crate
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0],q[1];
        sdg q[1];
        cx q[0],q[1];
        h q[0];
        measure q -> c;
    "#;

    // Create engine and load QASM with qelib1.inc
    let engine = QASMEngine::from_str(qasm)?;

    // Verify the engine loaded successfully with 2 qubits
    assert_eq!(engine.num_qubits(), 2);

    Ok(())
}

#[test]
fn test_custom_includes_with_embedded_standard() -> Result<(), PecosError> {
    // Test that both embedded standard includes and custom includes work together
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        gate bell a,b {
            h a;
            cx a,b;
        }
        qreg q[2];
        creg c[2];
        bell q[0],q[1];
        measure q -> c;
    "#;

    let engine = QASMEngine::from_str(qasm)?;

    assert_eq!(engine.num_qubits(), 2);

    Ok(())
}

#[test]
fn test_pecos_inc_available() -> Result<(), PecosError> {
    // Test that pecos.inc is also available
    let qasm = r#"
        OPENQASM 2.0;
        include "pecos.inc";
        qreg q[2];
        creg c[2];
        // Use a gate from pecos.inc if any specific ones exist
        // For now just verify the include works
        H q[0];
        CX q[0],q[1];
        measure q -> c;
    "#;

    let engine = QASMEngine::from_str(qasm)?;

    assert_eq!(engine.num_qubits(), 2);

    Ok(())
}
