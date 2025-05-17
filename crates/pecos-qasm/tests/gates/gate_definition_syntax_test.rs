use pecos_qasm::QASMParser;

#[test]
fn test_basic_gate_definition() {
    let qasm = r"
        OPENQASM 2.0;
        qreg q[2];

        // Basic gate with no parameters
        gate mygate a {
            H a;
            X a;
        }

        mygate q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);
    assert!(result.is_ok());

    let program = result.unwrap();
    assert!(program.gate_definitions.contains_key("mygate"));
    assert_eq!(program.gate_definitions["mygate"].params.len(), 0);
    assert_eq!(program.gate_definitions["mygate"].qargs.len(), 1);
}

#[test]
fn test_gate_with_single_parameter() {
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];

        gate phase_gate(lambda) q {
            RZ(lambda) q;
        }

        phase_gate(pi/4) q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);
    assert!(result.is_ok());

    let program = result.unwrap();
    assert!(program.gate_definitions.contains_key("phase_gate"));
    assert_eq!(
        program.gate_definitions["phase_gate"].params,
        vec!["lambda"]
    );
}

#[test]
fn test_gate_with_multiple_parameters() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        gate u3(theta, phi, lambda) q {
            RZ(phi) q;
            rx(theta) q;
            RZ(lambda) q;
        }

        u3(pi/2, pi/4, pi/8) q[0];
    "#;

    let result = QASMParser::parse_str(qasm);
    if let Err(e) = &result {
        eprintln!("Error in test_gate_with_multiple_parameters: {e}");
    }
    assert!(result.is_ok());

    let program = result.unwrap();
    assert!(program.gate_definitions.contains_key("u3"));
    assert_eq!(
        program.gate_definitions["u3"].params,
        vec!["theta", "phi", "lambda"]
    );
}

#[test]
fn test_gate_with_multiple_qubits() {
    let qasm = r"
        OPENQASM 2.0;
        qreg q[3];

        gate three_way a, b, c {
            CX a, b;
            CX b, c;
            CX a, c;
        }

        three_way q[0], q[1], q[2];
    ";

    let result = QASMParser::parse_str_raw(qasm);
    assert!(result.is_ok());

    let program = result.unwrap();
    assert!(program.gate_definitions.contains_key("three_way"));
    assert_eq!(
        program.gate_definitions["three_way"].qargs,
        vec!["a", "b", "c"]
    );
}

#[test]
fn test_parameter_expressions_in_gate_body() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        gate complex_gate(theta) q {
            RZ(theta/2) q;
            rx(theta*2) q;
            ry(theta + pi/4) q;
            RZ(theta - pi/2) q;
        }

        complex_gate(pi) q[0];
    "#;

    let result = QASMParser::parse_str(qasm);
    if let Err(e) = &result {
        eprintln!("Error in test_gate_with_multiple_parameters: {e}");
    }
    assert!(result.is_ok());
}

#[test]
fn test_nested_gate_calls() {
    let qasm = r"
        OPENQASM 2.0;
        qreg q[2];

        gate inner a {
            H a;
            X a;
        }

        gate outer(theta) a, b {
            inner a;
            RZ(theta) a;
            inner b;
            CX a, b;
        }

        outer(pi/3) q[0], q[1];
    ";

    let result = QASMParser::parse_str_raw(qasm);
    assert!(result.is_ok());
}

#[test]
fn test_empty_gate_body() {
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];

        gate do_nothing a {
            // Empty body - should be valid
        }

        do_nothing q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);
    assert!(result.is_ok());
}

#[test]
fn test_gate_name_conflicts() {
    // Test that we can redefine gates from the standard library
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Redefine the h gate with a simple implementation
        gate H a {
            RZ(pi/2) a;
            X a;
            RZ(pi/2) a;
        }

        H q[0];
    "#;

    let result = QASMParser::parse_str(qasm);
    if let Err(e) = &result {
        eprintln!("Error in test_gate_with_multiple_parameters: {e}");
    }
    assert!(result.is_ok());

    let program = result.unwrap();
    // Our custom h should override the library version
    assert!(program.gate_definitions.contains_key("h"));
}

#[test]
fn test_invalid_gate_syntax() {
    // Missing body braces
    let qasm1 = r"
        OPENQASM 2.0;
        gate bad a H a;
    ";

    let result1 = QASMParser::parse_str_raw(qasm1);
    assert!(result1.is_err());

    // Missing parameter list parentheses
    let qasm2 = r"
        OPENQASM 2.0;
        gate bad theta a { RZ(theta) a; }
    ";

    let result2 = QASMParser::parse_str_raw(qasm2);
    assert!(result2.is_err());
}
