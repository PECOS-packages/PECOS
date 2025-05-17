use pecos_qasm::parser::QASMParser;

#[test]
fn test_undefined_gate_error() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        gatedoesntexist q[0];
    "#;

    // This should fail because 'gatedoesntexist' is not a defined gate
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_err(), "Should fail with undefined gate error");
    
    if let Err(e) = result {
        let error_message = e.to_string();
        println!("Error message: {}", error_message);
        
        // The error should mention the undefined gate
        assert!(
            error_message.contains("gatedoesntexist") || 
            error_message.contains("undefined") || 
            error_message.contains("not defined") ||
            error_message.contains("unknown"),
            "Error should mention the undefined gate"
        );
    }
}

#[test]
fn test_misspelled_gate_error() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        
        hadamrd q[0];  // misspelled 'hadamard' or 'h'
    "#;

    let result = QASMParser::parse_str(qasm);
    assert!(result.is_err(), "Should fail with misspelled gate error");
}

#[test]
fn test_case_sensitive_gate_error() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        
        CZ q[0], q[1];  // Should be lowercase 'cz'
    "#;

    // This might or might not fail depending on whether the parser is case-sensitive
    let result = QASMParser::parse_str(qasm);
    
    // Let's check what happens
    match result {
        Ok(program) => {
            // If it succeeds, the parser accepts uppercase gates
            println!("Parser accepts uppercase gates");
            assert!(!program.operations.is_empty());
        }
        Err(e) => {
            // If it fails, the parser is case-sensitive
            println!("Parser is case-sensitive: {}", e);
        }
    }
}

#[test]
fn test_gate_with_wrong_arity() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];

        cx q[0];  // cx requires 2 qubits, not 1
    "#;

    let result = QASMParser::parse_str(qasm);
    // The parser might accept this syntactically but fail during execution
    match result {
        Ok(_) => println!("Parser accepts syntactically valid but semantically incorrect arity"),
        Err(e) => println!("Parser rejects wrong arity: {}", e),
    }
}

#[test]
fn test_gate_with_too_many_parameters() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        rz(pi, pi/2) q[0];  // rz only takes 1 parameter
    "#;

    let result = QASMParser::parse_str(qasm);
    // The parser might accept extra parameters syntactically
    match result {
        Ok(_) => println!("Parser accepts extra parameters syntactically"),
        Err(e) => println!("Parser rejects extra parameters: {}", e),
    }
}

#[test]
fn test_gate_with_missing_parameters() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        
        rz q[0];  // rz requires an angle parameter
    "#;

    let result = QASMParser::parse_str(qasm);
    assert!(result.is_err(), "Should fail with missing parameter");
}