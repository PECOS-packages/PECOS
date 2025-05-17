// Test extended gate support in PECOS QASM
use pecos_qasm::QASMEngine;
use std::str::FromStr;

#[test]
fn test_basic_rotation_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Test RZ gate
        RZ(pi/2) q[0];

        // Test S and T gates
        s q[0];
        sdg q[0];
        t q[0];
        tdg q[0];
    "#;

    let result = QASMEngine::from_str(qasm);

    assert!(result.is_ok(), "Should successfully parse rotation gates");
}

#[test]
fn test_two_qubit_rotations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];

        // Test RZZ gate with parameter
        RZZ(pi/4) q[0], q[1];

        // Test SZZ gate
        SZZ q[0], q[1];
    "#;

    let result = QASMEngine::from_str(qasm);

    assert!(
        result.is_ok(),
        "Should successfully parse two-qubit rotation gates"
    );
}

#[test]
fn test_decomposed_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];

        // Test gates that are decomposed from the qelib1 library
        cz q[0], q[1];
        cy q[0], q[1];
        swap q[0], q[1];
    "#;

    let result = QASMEngine::from_str(qasm);

    assert!(result.is_ok(), "Should successfully parse decomposed gates");
}

#[test]
fn test_parameterized_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Test parameterized gates
        RZ(pi) q[0];
        RZ(pi/2) q[0];
        RZ(0.7854) q[0];  // pi/4 in decimal
    "#;

    let result = QASMEngine::from_str(qasm);

    assert!(
        result.is_ok(),
        "Should successfully parse parameterized gates"
    );
}

#[test]
fn test_unsupported_gate_error() {
    let qasm = r"
        OPENQASM 2.0;
        qreg q[3];

        // This should fail during parsing - Toffoli is not defined
        ccx q[0], q[1], q[2];
    ";

    let result = QASMEngine::from_str(qasm);

    // With stricter parsing, this should now fail at parse time
    assert!(result.is_err(), "Should fail on undefined gate");

    if let Err(e) = result {
        let error_msg = e.to_string();
        assert!(
            error_msg.contains("Undefined") && error_msg.contains("ccx"),
            "Error should mention undefined gate ccx: {error_msg}"
        );
    }
}
