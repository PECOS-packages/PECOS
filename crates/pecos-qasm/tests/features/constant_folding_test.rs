//! Tests for constant folding optimization

use pecos_qasm::QASMParser;

#[test]
fn test_float_constant_folding() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // These should be folded at compile time
        rz(pi/2) q[0];
        rx(2*pi) q[0];
        ry(pi/4 + pi/4) q[0];
        u(sin(pi/2), cos(0), sqrt(4)) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // Check that the parameters have been folded to constants
    for op in &program.operations {
        if let pecos_qasm::ast::Operation::Gate {
            name, parameters, ..
        } = op
        {
            match name.as_str() {
                "rz" | "ry" => {
                    assert_eq!(parameters.len(), 1);
                    assert!((parameters[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-10);
                }
                "rx" => {
                    assert_eq!(parameters.len(), 1);
                    assert!((parameters[0] - 2.0 * std::f64::consts::PI).abs() < 1e-10);
                }
                "u" => {
                    assert_eq!(parameters.len(), 3);
                    assert!((parameters[0] - 1.0).abs() < 1e-10); // sin(pi/2) = 1
                    assert!((parameters[1] - 1.0).abs() < 1e-10); // cos(0) = 1
                    assert!((parameters[2] - 2.0).abs() < 1e-10); // sqrt(4) = 2
                }
                _ => {}
            }
        }
    }
}

#[test]
fn test_integer_constant_folding() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg a[8];
        creg b[8];

        // Arithmetic folding
        a = 5 + 3;         // Should fold to 8
        b = 10 - 7;        // Should fold to 3

        // Conditional with constant folding
        if(5 == 5) x q[0]; // Should always execute
        if(3 > 5) y q[0];  // Should never execute
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // The constant expressions should be folded
    let mut x_count = 0;
    let mut y_count = 0;

    for op in &program.operations {
        match op {
            pecos_qasm::ast::Operation::ClassicalAssignment { expression, .. } => {
                // Check that expressions are folded to constants
                match expression {
                    pecos_qasm::ast::Expression::Integer(bv) => {
                        let value = pecos_core::bitvec::to_decimal_string(bv);
                        assert!(value == "8" || value == "3");
                    }
                    _ => panic!("Expected folded integer constant"),
                }
            }
            pecos_qasm::ast::Operation::If {
                condition: pecos_qasm::ast::Expression::Integer(bv),
                operation,
            } => {
                // Check that conditions are folded
                let value = pecos_core::bitvec::to_decimal_string(bv);
                if value == "1" {
                    // Condition is true, operation should be X
                    match &**operation {
                        pecos_qasm::ast::Operation::Gate { name, .. } => {
                            if name == "x" {
                                x_count += 1;
                            }
                        }
                        pecos_qasm::ast::Operation::NativeGate(gate) => {
                            if matches!(gate.gate_type, pecos_engines::GateType::X) {
                                x_count += 1;
                            }
                        }
                        _ => {}
                    }
                }
            }
            pecos_qasm::ast::Operation::Gate { name, .. } => {
                if name == "y" {
                    y_count += 1;
                }
            }
            _ => {}
        }
    }

    assert_eq!(
        x_count, 1,
        "X gate should appear once (from true condition)"
    );
    assert_eq!(
        y_count, 0,
        "Y gate should not appear (from false condition)"
    );
}

#[test]
fn test_bitwise_constant_folding() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        creg a[8];
        creg b[8];
        creg c[8];

        // Bitwise operations
        a = 5 & 3;    // 0b101 & 0b011 = 0b001 = 1
        b = 5 | 3;    // 0b101 | 0b011 = 0b111 = 7
        c = 5 ^ 3;    // 0b101 ^ 0b011 = 0b110 = 6
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    let mut values = Vec::new();
    for op in &program.operations {
        if let pecos_qasm::ast::Operation::ClassicalAssignment {
            expression: pecos_qasm::ast::Expression::Integer(bv),
            ..
        } = op
        {
            values.push(pecos_core::bitvec::to_decimal_string(bv));
        }
    }

    assert_eq!(values, vec!["1", "7", "6"]);
}

#[test]
fn test_complex_expression_folding() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg a[8];

        // Complex nested expression
        a = (2 + 1) * 2;  // Should fold to 6

        // Mixed float expression in gate
        rz((pi/2 + pi/2) * sin(pi/2)) q[0];  // Should fold to pi
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // Check integer folding
    for op in &program.operations {
        match op {
            pecos_qasm::ast::Operation::ClassicalAssignment {
                expression: pecos_qasm::ast::Expression::Integer(bv),
                ..
            } => {
                assert_eq!(pecos_core::bitvec::to_decimal_string(bv), "6");
            }
            pecos_qasm::ast::Operation::Gate {
                name, parameters, ..
            } => {
                if name == "rz" {
                    assert_eq!(parameters.len(), 1);
                    // (pi/2 + pi/2) * sin(pi/2) = pi * 1 = pi
                    assert!((parameters[0] - std::f64::consts::PI).abs() < 1e-10);
                }
            }
            _ => {}
        }
    }
}
