use pecos_qasm::QASMParser;

#[test]
fn test_circular_dependency_detection() {
    // Test direct circular dependency
    let qasm_direct = r#"
        OPENQASM 2.0;
        qreg q[1];
        gate g1 q { g1 q; }
        g1 q[0];
    "#;
    
    match QASMParser::parse_str(qasm_direct) {
        Err(e) => {
            assert!(e.to_string().contains("Circular dependency"));
            assert!(e.to_string().contains("g1 -> g1"));
        }
        Ok(_) => panic!("Expected error due to circular dependency"),
    }
}

#[test]
fn test_indirect_circular_dependency_detection() {
    // Test indirect circular dependency (A -> B -> A)
    let qasm_indirect = r#"
        OPENQASM 2.0;
        qreg q[1];
        gate g1 q { g2 q; }
        gate g2 q { g1 q; }
        g1 q[0];
    "#;
    
    match QASMParser::parse_str(qasm_indirect) {
        Err(e) => {
            assert!(e.to_string().contains("Circular dependency"));
            // Either g1 -> g2 -> g1 or g2 -> g1 -> g2 is valid depending on which gets expanded first
            assert!(
                e.to_string().contains("g1 -> g2 -> g1") || 
                e.to_string().contains("g2 -> g1 -> g2")
            );
        }
        Ok(_) => panic!("Expected error due to circular dependency"),
    }
}

#[test]
fn test_complex_circular_dependency_detection() {
    // Test complex circular dependency (A -> B -> C -> A)
    let qasm_complex = r#"
        OPENQASM 2.0;
        qreg q[1];
        gate g1 q { g2 q; }
        gate g2 q { g3 q; }
        gate g3 q { g1 q; }
        g1 q[0];
    "#;
    
    match QASMParser::parse_str(qasm_complex) {
        Err(e) => {
            assert!(e.to_string().contains("Circular dependency"));
            assert!(e.to_string().contains("g1 -> g2 -> g3 -> g1"));
        }
        Ok(_) => panic!("Expected error due to circular dependency"),
    }
}

#[test]
fn test_valid_deep_nesting() {
    // Test that valid deep nesting still works
    let qasm_valid = r#"
        OPENQASM 2.0;
        qreg q[1];
        gate g1 q { h q; }
        gate g2 q { g1 q; }
        gate g3 q { g2 q; }
        gate g4 q { g3 q; }
        gate g5 q { g4 q; }
        g5 q[0];
    "#;
    
    match QASMParser::parse_str(qasm_valid) {
        Ok(_) => { /* Success */ }
        Err(e) => panic!("Valid deep nesting failed with error: {}", e),
    }
}

#[test]
fn test_circular_dependency_with_parameters() {
    // Test circular dependency with parameterized gates
    let qasm_param = r#"
        OPENQASM 2.0;
        qreg q[1];
        gate rot(theta) q { rot(theta) q; }
        rot(pi/2) q[0];
    "#;
    
    match QASMParser::parse_str(qasm_param) {
        Err(e) => {
            assert!(e.to_string().contains("Circular dependency"));
            assert!(e.to_string().contains("rot -> rot"));
        }
        Ok(_) => panic!("Expected error due to circular dependency"),
    }
}

#[test]
fn test_circular_dependency_without_usage() {
    // Test that circular dependencies can be defined but not used
    let qasm_unused = r#"
        OPENQASM 2.0;
        qreg q[2];
        gate g1 q { g2 q; }
        gate g2 q { g1 q; }
        CX q[0], q[1];  // Use a different gate
    "#;
    
    // This should succeed since we never actually use the circular gates
    assert!(QASMParser::parse_str(qasm_unused).is_ok());
}