use pecos_qasm::parser::QASMParser;

#[test]
fn test_qubit_index_out_of_bounds() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        rz(1.5*pi) q[4];
    "#;

    // This should fail because qubit 4 doesn't exist (only 0, 1, 2 are valid)
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_err(), "Should fail with out-of-bounds qubit index");
    
    if let Err(e) = result {
        let error_message = e.to_string();
        println!("Error message: {}", error_message);
        // The error should mention the out-of-bounds qubit
        assert!(
            error_message.contains("4") || 
            error_message.contains("out of bounds") || 
            error_message.contains("does not exist") ||
            error_message.contains("undefined"),
            "Error should mention the invalid qubit index"
        );
    }
}

#[test]
fn test_valid_qubit_indices() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        rz(1.5*pi) q[0];
        rz(1.5*pi) q[1];
        rz(1.5*pi) q[2];
    "#;

    // This should succeed - all indices are valid
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_ok(), "Should succeed with valid qubit indices");
    
    let program = result.unwrap();
    // Check that we have gates on the correct qubits
    let mut qubit_indices = Vec::new();
    for op in &program.operations {
        if let pecos_qasm::Operation::Gate { qubits, .. } = op {
            for &qubit in qubits {
                qubit_indices.push(qubit);
            }
        }
    }
    
    // All indices should be within bounds
    for &idx in &qubit_indices {
        assert!(idx < 3, "Qubit index {} should be less than 3", idx);
    }
}

#[test]
fn test_multiple_registers_index_error() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        qreg r[2];
        cx q[0], r[2];  // r[2] is out of bounds
    "#;

    // This should fail because r[2] doesn't exist (only r[0], r[1] are valid)
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_err(), "Should fail with out-of-bounds qubit index in second register");
}

#[test]
fn test_negative_index_error() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        rz(pi) q[-1];  // negative index should be invalid
    "#;

    // This should fail because negative indices are not allowed
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_err(), "Should fail with negative qubit index");
}

#[test]
fn test_register_boundary() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[5];
        rz(pi) q[0];  // valid
        rz(pi) q[4];  // valid (last index)
        rz(pi) q[5];  // invalid (out of bounds)
    "#;

    // This should fail at the last gate
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_err(), "Should fail with qubit index 5 in register of size 5");
}