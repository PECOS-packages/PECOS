use pecos_qasm::QASMParser;

#[test]
fn test_undefined_gate_fails() {
    // Test with rx gate which is NOT in the native gates list
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];
        rx(pi/2) q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);

    // This should fail because rx is not native and not defined
    assert!(result.is_err());

    if let Err(e) = result {
        let error_msg = e.to_string();
        assert!(error_msg.contains("rx"));
        assert!(error_msg.contains("Undefined"));
        assert!(error_msg.contains("qelib1.inc"));
    }
}

#[test]
fn test_native_gates_pass() {
    // Test with gates that ARE in the native list
    let qasm = r"
        OPENQASM 2.0;
        qreg q[2];
        H q[0];
        CX q[0], q[1];
        RZ(pi) q[1];
    ";

    let result = QASMParser::parse_str_raw(qasm);

    // This should pass because these are native gates
    assert!(result.is_ok());
}

#[test]
fn test_defined_gates_pass() {
    // Test with user-defined gates
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];
        
        gate mygate a {
            H a;
            X a;
        }
        
        mygate q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);

    // This should pass because mygate is defined
    assert!(result.is_ok());
}

#[test]
fn test_gates_in_definitions_only() {
    // Test that gates used only in definitions don't cause errors
    // until the definition is actually used
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];
        
        gate uses_undefined a {
            rx(pi) a;  // rx is not native
        }
        
        // Don't use the gate - should still pass
        H q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);

    // This should pass because uses_undefined is never used
    assert!(result.is_ok());
}

#[test]
fn test_using_gate_with_undefined_gates() {
    // Test that using a gate that contains undefined gates fails
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];

        gate uses_undefined a {
            undefined_gate a;  // This gate doesn't exist anywhere
        }

        uses_undefined q[0];  // This should trigger expansion and fail
    ";

    let result = QASMParser::parse_str_raw(qasm);

    // This should fail when expanding uses_undefined
    assert!(result.is_err());

    if let Err(e) = result {
        let error_msg = e.to_string();
        assert!(error_msg.contains("undefined_gate"));
        assert!(error_msg.contains("Undefined"));
    }
}
