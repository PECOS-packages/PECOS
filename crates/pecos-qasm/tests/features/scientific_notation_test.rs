use pecos_qasm::parser::QASMParser;

#[test]
fn test_scientific_notation_formats() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];

        // Basic scientific notation
        rx(1.5e-3) q[0];
        rx(1.5E-3) q[0];
        rx(2e4) q[0];
        rx(2E4) q[0];

        // With explicit sign
        rx(1.5e+3) q[0];
        rx(1.5E+3) q[0];
        rx(2e-4) q[0];
        rx(2E-4) q[0];

        // Without decimal part
        rx(5e2) q[0];
        rx(5E2) q[0];

        // With decimal but no fractional part
        rx(5.e2) q[0];
        rx(5.E2) q[0];

        // With no integer part
        rx(.5e2) q[0];
        rx(.5E2) q[0];

        // Regular decimal numbers still work
        rx(3.14159) q[0];
        rx(0.123) q[0];
        rx(.456) q[0];
        rx(789.) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // After expansion, we'll have more operations than just the original gates
    assert!(!program.operations.is_empty());

    // All operations should be gate calls
    for op in &program.operations {
        match op {
            pecos_qasm::Operation::Gate { .. } => {
                // Gate expanded into native operations
            }
            _ => panic!("Expected only gate calls"),
        }
    }
}

#[test]
fn test_scientific_notation_in_expressions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Scientific notation in expressions
        rx(1e-3 + 2e-3) q[0];
        rx(5e2 * 2) q[0];
        rx(1.5E3 / 3) q[0];
        rx(-2.5e-2) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_scientific_notation_edge_cases() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Very small numbers
        rx(1e-308) q[0];

        // Very large numbers
        rx(1e308) q[0];

        // Zero with scientific notation
        rx(0e0) q[0];
        rx(0.0e0) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_scientific_notation_with_pi() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Scientific notation mixed with pi
        rx(pi * 1e-3) q[0];
        rx(2e2 * pi) q[0];
        rx(pi / 1.5e1) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_scientific_notation_in_gate_definitions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        gate mygate(a, b) q {
            rx(a * 1e-3) q;
            ry(b * 2.5E2) q;
        }

        mygate(3.14, 1.5e-1) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // Should have our custom gate definition
    assert!(program.gate_definitions.contains_key("mygate"));

    // The custom gate should have parameters with expressions containing scientific notation
    let gate_def = &program.gate_definitions["mygate"];
    assert_eq!(gate_def.params.len(), 2);

    // The main thing is that the program parses successfully with scientific notation
    // in gate parameters and definitions
}
