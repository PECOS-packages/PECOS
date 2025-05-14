use pecos_qasm::parser::QASMParser;
use std::f64::consts::PI;

#[test]
fn test_trig_functions() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        // Test trigonometric functions
        rx(sin(pi/2)) q[0];  // sin(pi/2) = 1
        ry(cos(0)) q[0];     // cos(0) = 1
        rz(tan(pi/4)) q[0];  // tan(pi/4) = 1
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert_eq!(program.operations.len(), 3);
}

#[test]
fn test_exp_ln_functions() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        // Test exponential and logarithm
        rx(exp(0)) q[0];     // exp(0) = 1
        ry(ln(1)) q[0];      // ln(1) = 0
        rz(exp(ln(2))) q[0]; // exp(ln(2)) = 2
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert_eq!(program.operations.len(), 3);
}

#[test]
fn test_sqrt_function() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        // Test square root
        rx(sqrt(4)) q[0];    // sqrt(4) = 2
        ry(sqrt(0.25)) q[0]; // sqrt(0.25) = 0.5
        rz(sqrt(9)) q[0];    // sqrt(9) = 3
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert_eq!(program.operations.len(), 3);
}

#[test]
fn test_nested_functions() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        // Test nested mathematical functions
        rx(sin(cos(0))) q[0];        // sin(cos(0)) = sin(1)
        ry(sqrt(exp(ln(4)))) q[0];   // sqrt(exp(ln(4))) = sqrt(4) = 2
        rz(cos(sin(pi/2))) q[0];     // cos(sin(pi/2)) = cos(1)
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert_eq!(program.operations.len(), 3);
}

#[test]
fn test_functions_with_expressions() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        // Test functions with complex expressions
        rx(sin(pi/6 + pi/3)) q[0];    // sin(pi/2) = 1
        ry(cos(2*pi - pi)) q[0];      // cos(pi) = -1
        rz(sqrt(2*2 + 3*3)) q[0];     // sqrt(13)
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert_eq!(program.operations.len(), 3);
}

#[test]
fn test_error_cases() {
    // Test ln of negative number - parsing should succeed
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        rx(ln(-1)) q[0];
    "#;

    let result = QASMParser::parse_str(qasm);
    // The parsing should fail because ln(-1) is evaluated during parsing for gate parameters
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("ln(-1) is undefined"));
    }

    // Test sqrt of negative number
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        rx(sqrt(-4)) q[0];
    "#;

    let result = QASMParser::parse_str(qasm);
    // The parsing should fail because sqrt(-4) is evaluated during parsing for gate parameters
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("sqrt(-4) is undefined"));
    }
}

#[test]
fn test_functions_in_gate_definitions() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        gate mygate(theta) q {
            rx(sin(theta)) q;
            ry(cos(theta)) q;
            rz(sqrt(theta)) q;
        }
        
        mygate(pi/4) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert!(program.gate_definitions.contains_key("mygate"));
}

#[test]
fn test_all_math_functions() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        // Test all mathematical functions
        rx(sin(pi/2)) q[0];
        rx(cos(pi)) q[0];
        rx(tan(pi/4)) q[0];
        rx(exp(1)) q[0];
        rx(ln(2.718281828)) q[0];
        rx(sqrt(2)) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert_eq!(program.operations.len(), 6);
}

#[test]
fn test_evaluation_accuracy() {
    use pecos_qasm::parser::Expression;
    
    // Test sin
    let expr = Expression::FunctionCall {
        name: "sin".to_string(),
        args: vec![Expression::Float(PI / 2.0)],
    };
    assert!((expr.evaluate().unwrap() - 1.0).abs() < 1e-10);
    
    // Test cos
    let expr = Expression::FunctionCall {
        name: "cos".to_string(),
        args: vec![Expression::Float(0.0)],
    };
    assert!((expr.evaluate().unwrap() - 1.0).abs() < 1e-10);
    
    // Test tan
    let expr = Expression::FunctionCall {
        name: "tan".to_string(),
        args: vec![Expression::Float(PI / 4.0)],
    };
    assert!((expr.evaluate().unwrap() - 1.0).abs() < 1e-10);
    
    // Test exp
    let expr = Expression::FunctionCall {
        name: "exp".to_string(),
        args: vec![Expression::Float(0.0)],
    };
    assert!((expr.evaluate().unwrap() - 1.0).abs() < 1e-10);
    
    // Test ln
    let expr = Expression::FunctionCall {
        name: "ln".to_string(),
        args: vec![Expression::Float(std::f64::consts::E)],
    };
    assert!((expr.evaluate().unwrap() - 1.0).abs() < 1e-10);
    
    // Test sqrt
    let expr = Expression::FunctionCall {
        name: "sqrt".to_string(),
        args: vec![Expression::Float(4.0)],
    };
    assert!((expr.evaluate().unwrap() - 2.0).abs() < 1e-10);
}