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
        h q[0];                    // Gate call
        cx q[0], q[1];            // Two-qubit gate
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
        if (c[0] == 1) h q[1];    // Conditional gate
        if (c > 3) x q[2];        // Conditional with comparison
        
        // Gate definitions
        gate mygate a {
            h a;
            x a;
        }
        
        // Opaque gate declarations
        opaque oracle(theta) a, b;
        
        // Using defined gates
        mygate q[0];
    "#;
    
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_ok(), "All these operations should be allowed at top level");
}

/// Test operations that should NOT be allowed at the top level
#[test]
fn test_disallowed_top_level_operations() {
    // Test 1: Nested gate definitions (gates can't be defined inside other structures)
    let qasm1 = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        if (1) {
            gate bad a { h a; }  // Can't define gates inside if
        }
    "#;
    
    let result1 = QASMParser::parse_str(qasm1);
    assert!(result1.is_err(), "Gate definitions inside if should fail");
    
    // Test 2: Invalid measurement syntax
    let qasm2 = r#"
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];

        measure q[0] c[0];  // Missing arrow
    "#;

    let result2 = QASMParser::parse_str(qasm2);
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
            h a;
            x b;
            y c;
            z a;
            
            // Two-qubit gates
            cx a, b;
            cz b, c;
            
            // Parameterized gates
            rx(pi/4) a;
            ry(pi/2) b;
            rz(pi) c;
            
            // Composite gates (defined elsewhere)
            ccx a, b, c;
            
            // Currently also accepts (but shouldn't):
            barrier a, b;  // This works but shouldn't
            reset a;       // This works but shouldn't
        }
        
        allowed_ops q[0], q[1], q[2];
    "#;
    
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_ok(), "These operations are currently allowed in gate bodies");
}

/// Test operations that should NOT be allowed in gate definitions
#[test]
fn test_disallowed_gate_body_operations() {
    // Test 1: Measurements in gate body
    let qasm1 = r#"
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];
        
        gate bad_gate a {
            measure a -> c[0];  // Measurements not allowed
        }
    "#;
    
    let result1 = QASMParser::parse_str(qasm1);
    assert!(result1.is_err(), "Measurements in gate body should fail");
    
    // Test 2: Classical operations in gate body
    let qasm2 = r#"
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];
        
        gate bad_gate a {
            c[0] = 1;  // Classical ops not allowed
        }
    "#;
    
    let result2 = QASMParser::parse_str(qasm2);
    assert!(result2.is_err(), "Classical operations in gate body should fail");
    
    // Test 3: If statements in gate body
    let qasm3 = r#"
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];
        
        gate bad_gate a {
            if (c[0] == 1) h a;  // Conditionals not allowed
        }
    "#;
    
    let result3 = QASMParser::parse_str(qasm3);
    assert!(result3.is_err(), "If statements in gate body should fail");
    
    // Test 4: Nested gate definitions
    let qasm4 = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        gate outer a {
            gate inner b { h b; }  // Can't define gates inside gates
        }
    "#;
    
    let result4 = QASMParser::parse_str(qasm4);
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
        if (c[0] == 1) h q[0];
        
        // Single classical operation  
        if (c[0] == 0) c[1] = 1;
        
        // QASM doesn't support block if statements, only single operations
    "#;
    
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_ok(), "These operations should be allowed in if statements");
}

/// Test operations that are context-dependent
#[test]
fn test_context_dependent_operations() {
    // Barriers: allowed at top level and (currently) in gate bodies
    let qasm1 = r#"
        OPENQASM 2.0;
        qreg q[2];
        
        barrier q[0], q[1];  // OK at top level
        
        gate with_barrier a, b {
            barrier a, b;    // Currently allowed (but maybe shouldn't be)
        }
    "#;
    
    let result1 = QASMParser::parse_str(qasm1);
    assert!(result1.is_ok());
    
    // Reset: similar to barriers
    let qasm2 = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        reset q[0];  // OK at top level
        
        gate with_reset a {
            reset a;     // Currently allowed (but shouldn't be)
        }
    "#;
    
    let result2 = QASMParser::parse_str(qasm2);
    assert!(result2.is_ok());
}