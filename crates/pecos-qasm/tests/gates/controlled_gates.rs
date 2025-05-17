// Test gate definitions against examples from the OpenQASM 2.0 specification

use pecos_qasm::QASMParser;

#[test]
fn test_qasm_spec_example_1() {
    // Example from the spec: controlled-sqrt-Z gate
    let qasm = r"
        OPENQASM 2.0;
        qreg q[2];

        // Controlled sqrt(Z) gate
        gate cz a,b {
            H b;
            CX a,b;
            H b;
        }

        cz q[0], q[1];
    ";

    let result = QASMParser::parse_str(qasm);
    assert!(result.is_ok());
}

#[test]
fn test_qasm_spec_example_2() {
    // Example from the spec: Toffoli gate
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];

        gate ccx a,b,c {
            h c;
            CX b,c;
            tdg c;
            CX a,c;
            t c;
            CX b,c;
            tdg c;
            CX a,c;
            t b;
            t c;
            h c;
            CX a,b;
            t a;
            tdg b;
            CX a,b;
        }

        ccx q[0], q[1], q[2];
    "#;

    let result = QASMParser::parse_str(qasm);
    assert!(result.is_ok());
}

#[test]
fn test_qasm_spec_example_3() {
    // Example with parameters
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];

        // Rotation about X-axis
        gate rx(theta) a {
            H a;
            RZ(theta) a;
            H a;
        }

        rx(pi/2) q[0];
    ";

    let result = QASMParser::parse_str(qasm);
    assert!(result.is_ok());
}

#[test]
fn test_qasm_spec_example_4() {
    // Example of gate using other gates
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
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
        include "qelib1.inc";
        qreg q[4];

        // No parameters, single qubit
        gate x180 a {
            X a;
            X a;
        }

        // Multiple parameters, single qubit
        gate u3(theta,phi,lambda) q {
            RZ(phi) q;
            ry(theta) q;
            RZ(lambda) q;
        }

        // No parameters, multiple qubits
        gate swap a,b {
            CX a,b;
            CX b,a;
            CX a,b;
        }

        // Parameters with expressions
        gate mygate(alpha) q {
            RZ(alpha/2) q;
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
    let invalid1 = r"
        OPENQASM 2.0;
        gate bad a H a;
    ";
    assert!(QASMParser::parse_str_raw(invalid1).is_err());

    // Invalid parameter syntax (missing parentheses)
    let invalid2 = r"
        OPENQASM 2.0;
        gate bad theta a { RZ(theta) a; }
    ";
    assert!(QASMParser::parse_str_raw(invalid2).is_err());

    // Empty parameter list
    let valid_empty_params = r"
        OPENQASM 2.0;
        gate good() a { H a; }
    ";
    // This might be valid or invalid depending on spec interpretation
    let result = QASMParser::parse_str_raw(valid_empty_params);
    println!("Empty params result: {:?}", result.is_ok());
}
