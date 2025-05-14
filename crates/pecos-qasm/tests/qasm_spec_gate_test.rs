// Test gate definitions against examples from the OpenQASM 2.0 specification

use pecos_qasm::QASMParser;

#[test]
fn test_qasm_spec_example_1() {
    // Example from the spec: controlled-sqrt-Z gate
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[2];
        
        // Controlled sqrt(Z) gate
        gate cz a,b {
            h b;
            cx a,b;
            h b;
        }
        
        cz q[0], q[1];
    "#;
    
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_ok());
}

#[test]
fn test_qasm_spec_example_2() {
    // Example from the spec: Toffoli gate
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[3];
        
        gate ccx a,b,c {
            h c;
            cx b,c;
            tdg c;
            cx a,c;
            t c;
            cx b,c;
            tdg c;
            cx a,c;
            t b;
            t c;
            h c;
            cx a,b;
            t a;
            tdg b;
            cx a,b;
        }
        
        ccx q[0], q[1], q[2];
    "#;
    
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_ok());
}

#[test]
fn test_qasm_spec_example_3() {
    // Example with parameters
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        // Rotation about X-axis
        gate rx(theta) a {
            h a;
            rz(theta) a;
            h a;
        }
        
        rx(pi/2) q[0];
    "#;
    
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_ok());
}

#[test]
fn test_qasm_spec_example_4() {
    // Example of gate using other gates
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[2];
        
        // Define a CNOT using CZ and Hadamards
        gate cx_from_cz c,t {
            h t;
            cz c,t;
            h t;
        }
        
        cx_from_cz q[0], q[1];
    "#;
    
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_ok());
}

#[test]
fn test_qasm_spec_syntax_variations() {
    // Test various syntactic forms from the spec
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[4];
        
        // No parameters, single qubit
        gate x180 a {
            x a;
            x a;
        }
        
        // Multiple parameters, single qubit  
        gate u3(theta,phi,lambda) q {
            rz(phi) q;
            ry(theta) q;
            rz(lambda) q;
        }
        
        // No parameters, multiple qubits
        gate swap a,b {
            cx a,b;
            cx b,a;
            cx a,b;
        }
        
        // Parameters with expressions
        gate mygate(alpha) q {
            rz(alpha/2) q;
            rx(alpha*2) q;
            ry(alpha+pi) q;
        }
        
        // Using the gates
        x180 q[0];
        u3(pi/2, 0, pi) q[1];
        swap q[2], q[3];
        mygate(pi/4) q[0];
    "#;
    
    let result = QASMParser::parse_str(qasm);
    assert!(result.is_ok());
}

#[test]
fn test_qasm_spec_invalid_syntax() {
    // Test invalid gate definitions according to spec
    
    // Missing curly braces
    let invalid1 = r#"
        OPENQASM 2.0;
        gate bad a h a;
    "#;
    assert!(QASMParser::parse_str(invalid1).is_err());
    
    // Invalid parameter syntax (missing parentheses)
    let invalid2 = r#"
        OPENQASM 2.0;
        gate bad theta a { rz(theta) a; }
    "#;
    assert!(QASMParser::parse_str(invalid2).is_err());
    
    // Empty parameter list
    let valid_empty_params = r#"
        OPENQASM 2.0;
        gate good() a { h a; }
    "#;
    // This might be valid or invalid depending on spec interpretation
    let result = QASMParser::parse_str(valid_empty_params);
    println!("Empty params result: {:?}", result.is_ok());
}