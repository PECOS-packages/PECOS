use pecos_qasm::QASMParser;

/// Test all operations allowed at the top level of a QASM program
#[test]
fn test_allowed_top_level_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        // Register declarations
        qreg q[4];
        creg c[4];

        // Quantum operations
        H q[0];                    // Gate call
        CX q[0], q[1];            // Two-qubit gate
        rx(pi/2) q[2];            // Parameterized gate
        barrier q[0], q[1];       // Barrier
        reset q[3];               // Reset
        measure q[0] -> c[0];     // Measurement
        measure q -> c;           // Full register measurement

        // Classical operations
        c[1] = 1;                 // Bit assignment
        c = 5;                    // Register assignment
        c[2] = c[0] & c[1];      // Expression

        // Conditional operations
        if (c[0] == 1) H q[1];    // Conditional gate
        if (c > 3) X q[2];        // Conditional with comparison

        // Gate definitions
        gate mygate a {
            H a;
            X a;
        }

        // Opaque gate declarations
        opaque oracle(theta) a, b;

        // Using defined gates
        mygate q[0];
    "#;

    let result = QASMParser::parse_str(qasm);
    if let Err(ref e) = result {
        eprintln!("Error during parsing: {e}");

        // Try just phase 1
        if let Ok(preprocessed) = QASMParser::preprocess(qasm) {
            eprintln!("Phase 1 (preprocessed) succeeded");

            // Try phase 2
            match QASMParser::expand_all_gate_definitions(&preprocessed) {
                Ok(expanded) => {
                    eprintln!("Phase 2 (expanded) succeeded:");
                    eprintln!("Expanded QASM:\n{expanded}");
                }
                Err(e) => eprintln!("Phase 2 (expansion) failed: {e}"),
            }
        }
    }
    assert!(
        result.is_ok(),
        "All these operations should be allowed at top level"
    );
}

/// Test operations that should NOT be allowed at the top level
#[test]
fn test_disallowed_top_level_operations() {
    // Test 1: Nested gate definitions (gates can't be defined inside other structures)
    let qasm1 = r"
        OPENQASM 2.0;
        qreg q[1];

        if (1) {
            gate bad a { H a; }  // Can't define gates inside if
        }
    ";

    let result1 = QASMParser::parse_str_raw(qasm1);
    assert!(result1.is_err(), "Gate definitions inside if should fail");

    // Test 2: Invalid measurement syntax
    let qasm2 = r"
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];

        measure q[0] c[0];  // Missing arrow
    ";

    let result2 = QASMParser::parse_str_raw(qasm2);
    assert!(result2.is_err(), "Measurement without arrow should fail");
}

/// Test operations allowed inside gate definitions
#[test]
fn test_allowed_gate_body_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];

        gate allowed_ops a, b, c {
            // Basic gates
            H a;
            X b;
            y c;
            Z a;

            // Two-qubit gates
            CX a, b;
            cz b, c;

            // Parameterized gates
            rx(pi/4) a;
            ry(pi/2) b;
            RZ(pi) c;

            // Composite gates (defined elsewhere)
            ccx a, b, c;

            // Special operations now allowed in gate bodies
            barrier a, b;
            reset a;
        }

        allowed_ops q[0], q[1], q[2];
    "#;

    let result = QASMParser::parse_str(qasm);
    match result {
        Ok(_) => (),
        Err(e) => {
            eprintln!("Original QASM:\n{qasm}");
            panic!("Failed to parse: {e}")
        }
    }
}

/// Test that barrier and reset are now allowed in gate bodies
#[test]
fn test_barrier_reset_in_gate_body() {
    // Test 1: Barrier in gate body should now succeed
    let qasm_barrier = r"
        OPENQASM 2.0;
        qreg q[2];

        gate valid_gate a, b {
            H a;
            barrier a, b;  // This is now allowed
            X b;
        }

        valid_gate q[0], q[1];
    ";

    let result = QASMParser::parse_str(qasm_barrier);
    assert!(result.is_ok(), "Barrier should be allowed in gate bodies");

    // Test 2: Reset in gate body should now succeed
    let qasm_reset = r"
        OPENQASM 2.0;
        qreg q[1];

        gate valid_gate a {
            H a;
            reset a;  // This is now allowed
            X a;
        }

        valid_gate q[0];
    ";

    let result = QASMParser::parse_str(qasm_reset);
    assert!(result.is_ok(), "Reset should be allowed in gate bodies");
}

/// Test operations that should NOT be allowed in gate definitions
#[test]
fn test_disallowed_gate_body_operations() {
    // Test 1: Measurements in gate body
    let qasm1 = r"
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];

        gate bad_gate a {
            measure a -> c[0];  // Measurements not allowed
        }
    ";

    let result1 = QASMParser::parse_str_raw(qasm1);
    assert!(result1.is_err(), "Measurements in gate body should fail");

    // Test 2: Classical operations in gate body
    let qasm2 = r"
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];

        gate bad_gate a {
            c[0] = 1;  // Classical ops not allowed
        }
    ";

    let result2 = QASMParser::parse_str_raw(qasm2);
    assert!(
        result2.is_err(),
        "Classical operations in gate body should fail"
    );

    // Test 3: If statements in gate body
    let qasm3 = r"
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];

        gate bad_gate a {
            if (c[0] == 1) H a;  // Conditionals not allowed
        }
    ";

    let result3 = QASMParser::parse_str_raw(qasm3);
    assert!(result3.is_err(), "If statements in gate body should fail");

    // Test 4: Nested gate definitions
    let qasm4 = r"
        OPENQASM 2.0;
        qreg q[1];

        gate outer a {
            gate inner b { H b; }  // Can't define gates inside gates
        }
    ";

    let result4 = QASMParser::parse_str_raw(qasm4);
    assert!(result4.is_err(), "Nested gate definitions should fail");
}

/// Test operations allowed in if statement bodies
#[test]
fn test_allowed_if_body_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        // Single quantum operation
        if (c[0] == 1) H q[0];

        // Single classical operation
        if (c[0] == 0) c[1] = 1;

        // QASM doesn't support block if statements, only single operations
    "#;

    let result = QASMParser::parse_str(qasm);
    assert!(
        result.is_ok(),
        "These operations should be allowed in if statements"
    );
}

/// Test operations that are context-dependent
#[test]
fn test_context_dependent_operations() {
    // Barriers: allowed at top level and (currently) in gate bodies
    let qasm1 = r"
        OPENQASM 2.0;
        qreg q[2];

        barrier q[0], q[1];  // OK at top level

        gate with_barrier a, b {
            barrier a, b;    // Currently allowed (but maybe shouldn't be)
        }
    ";

    let result1 = QASMParser::parse_str_raw(qasm1);
    assert!(result1.is_ok());

    // Reset: similar to barriers
    let qasm2 = r"
        OPENQASM 2.0;
        qreg q[1];

        reset q[0];  // OK at top level

        gate with_reset a {
            reset a;     // Currently allowed (but shouldn't be)
        }
    ";

    let result2 = QASMParser::parse_str_raw(qasm2);
    assert!(result2.is_ok());
}
