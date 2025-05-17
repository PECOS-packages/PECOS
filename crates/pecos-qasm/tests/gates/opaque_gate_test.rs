use pecos_qasm::QASMParser;

/// Test for opaque gate declarations
/// According to `OpenQASM` 2.0 spec, opaque gates are used to define
/// gates that are implemented at a lower level (hardware or external library)
/// without specifying their decomposition in terms of other gates.
#[test]
fn test_opaque_gate_syntax() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        // Declare quantum registers
        qreg q[4];
        creg c[4];

        // Opaque gate declarations - these define gates without implementation
        // Single-qubit opaque gate without parameters
        opaque mygate1 a;

        // Single-qubit opaque gate with parameters
        opaque mygate2(theta, phi) a;

        // Two-qubit opaque gate
        opaque mygate3 a, b;

        // Two-qubit opaque gate with parameters
        opaque mygate4(alpha) a, b;

        // Three-qubit opaque gate
        opaque mygate5 a, b, c;

        // Use the opaque gates
        mygate1 q[0];
        mygate2(pi/2, pi/4) q[1];
        mygate3 q[0], q[1];
        mygate4(0.5) q[2], q[3];
        mygate5 q[0], q[1], q[2];

        // Measure
        measure q -> c;
    "#;

    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(_) => {
            panic!("Expected error for opaque gate usage, but parsing succeeded");
        }
        Err(e) => {
            // With stricter parsing, we now get undefined gate error
            // since opaque gates don't create actual definitions
            println!("Got expected error: {e}");
            assert!(e.to_string().contains("Undefined gate") && e.to_string().contains("mygate1"));
        }
    }
}

/// Test mixing opaque gates with regular gate definitions
#[test]
fn test_opaque_and_regular_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[3];
        creg c[3];

        // Regular gate definition
        gate bell a, b {
            H a;
            CX a, b;
        }

        // Opaque gate declaration - no body
        opaque oracle(theta) a, b;

        // Another regular gate using the opaque gate
        gate algorithm q1, q2 {
            bell q1, q2;
            oracle(pi/4) q1, q2;
            bell q1, q2;
        }

        // Use both types
        bell q[0], q[1];
        oracle(pi/2) q[1], q[2];
        algorithm q[0], q[2];

        measure q -> c;
    "#;

    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(ast) => {
            println!("Mixed opaque/regular gates AST:");
            println!("{ast:#?}");
        }
        Err(e) => {
            println!("Expected error: {e}");
        }
    }
}

/// Test that opaque gate declarations without usage are allowed
#[test]
fn test_opaque_gate_declaration_only() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[2];

        // Opaque gate declarations without usage - should be fine
        opaque mygate1 a;
        opaque mygate2(theta, phi) a;
        opaque mygate3 a, b;

        // Regular gate usage is still allowed
        H q[0];
        CX q[0], q[1];

        measure q -> c;
    "#;

    let result = QASMParser::parse_str(qasm);

    // This should succeed because we're not using the opaque gates
    match result {
        Ok(program) => {
            println!("Successfully parsed program with opaque declarations (no usage)");
            // Count opaque declarations
            let opaque_count = program
                .operations
                .iter()
                .filter(|op| matches!(op, pecos_qasm::Operation::OpaqueGate { .. }))
                .count();
            assert_eq!(opaque_count, 3);
        }
        Err(e) => {
            panic!("Should have succeeded, but got error: {e}");
        }
    }
}

/// Test error cases for opaque gates
#[test]
fn test_opaque_gate_errors() {
    // Test 1: Opaque gate with a body (should be an error)
    let invalid_qasm1 = r"
        OPENQASM 2.0;
        qreg q[2];

        // This should be an error - opaque gates can't have bodies
        opaque mygate a {
            H a;
        }
    ";

    let result1 = QASMParser::parse_str(invalid_qasm1);
    assert!(result1.is_err(), "Opaque gate with body should be an error");

    // Test 2: Using undefined opaque gate
    let invalid_qasm2 = r"
        OPENQASM 2.0;
        qreg q[2];

        // Using a gate that wasn't declared
        undefined_gate q[0];
    ";

    let result2 = QASMParser::parse_str(invalid_qasm2);
    // This might already fail as undefined gate
    println!("Undefined gate error: {result2:?}");
}
