use pecos_qasm::parser::QASMParser;

#[test]
fn test_power_operator_basic() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Test basic power operations
        rx(2**3) q[0];    // 2^3 = 8
        ry(3**2) q[0];    // 3^2 = 9
        RZ(10**0) q[0];   // 10^0 = 1
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    // After expansion, we'll have more than 3 operations
    assert!(!program.operations.is_empty());
}

#[test]
fn test_power_operator_with_floats() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Test power with floating point numbers
        rx(2.0**3.0) q[0];    // 2.0^3.0 = 8.0
        ry(4.0**0.5) q[0];    // 4.0^0.5 = 2.0 (square root)
        RZ(2.718281828**1) q[0]; // e^1 = e
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_power_operator_precedence() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Test operator precedence - power should bind tighter than multiplication
        rx(2*3**2) q[0];     // 2*(3^2) = 2*9 = 18, not (2*3)^2 = 36
        ry(2**3*2) q[0];     // (2^3)*2 = 8*2 = 16
        RZ(2+3**2) q[0];     // 2+(3^2) = 2+9 = 11
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_power_with_pi() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Test power with pi
        rx(pi**2) q[0];      // pi^2
        ry(2**pi) q[0];      // 2^pi
        RZ(pi**(1/2)) q[0];  // sqrt(pi)
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_power_negative_base() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Test power with negative base
        rx((-2)**3) q[0];    // (-2)^3 = -8
        ry((-1)**2) q[0];    // (-1)^2 = 1
        RZ((-3)**2) q[0];    // (-3)^2 = 9
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_power_in_gate_definitions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        gate powgate(a, b) q {
            rx(a**2) q;
            ry(2**b) q;
            RZ(a**b) q;
        }

        powgate(2, 3) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert!(program.gate_definitions.contains_key("powgate"));
}

#[test]
fn test_power_evaluation_accuracy() {
    use pecos_qasm::Expression;

    // Test 2^3
    let expr = Expression::BinaryOp {
        op: "**".to_string(),
        left: Box::new(Expression::Float(2.0)),
        right: Box::new(Expression::Float(3.0)),
    };
    assert!((expr.evaluate_with_context(None).unwrap() - 8.0).abs() < 1e-10);

    // Test 4^0.5 (square root)
    let expr = Expression::BinaryOp {
        op: "**".to_string(),
        left: Box::new(Expression::Float(4.0)),
        right: Box::new(Expression::Float(0.5)),
    };
    assert!((expr.evaluate_with_context(None).unwrap() - 2.0).abs() < 1e-10);

    // Test 10^0
    let expr = Expression::BinaryOp {
        op: "**".to_string(),
        left: Box::new(Expression::Float(10.0)),
        right: Box::new(Expression::Float(0.0)),
    };
    assert!((expr.evaluate_with_context(None).unwrap() - 1.0).abs() < 1e-10);
}
