use pecos_qasm::Expression;
use pecos_qasm::{Operation, QASMParser};
use std::f64::consts::PI;

#[test]
fn test_trig_functions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Test trigonometric functions
        rx(sin(pi/2)) q[0];  // sin(pi/2) = 1
        ry(cos(0)) q[0];     // cos(0) = 1
        RZ(tan(pi/4)) q[0];  // tan(pi/4) = 1
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    // Just verify the program compiles successfully
    assert!(!program.operations.is_empty());
}

#[test]
fn test_exp_ln_functions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Test exponential and logarithm
        rx(exp(0)) q[0];     // exp(0) = 1
        ry(ln(1)) q[0];      // ln(1) = 0
        RZ(exp(ln(2))) q[0]; // exp(ln(2)) = 2
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_sqrt_function() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Test square root
        rx(sqrt(4)) q[0];    // sqrt(4) = 2
        ry(sqrt(0.25)) q[0]; // sqrt(0.25) = 0.5
        RZ(sqrt(9)) q[0];    // sqrt(9) = 3
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // After includes, the high-level gates are expanded into native gates
    // rx, ry, and rz are all expanded, so we expect more than 3 operations
    // We should just verify that the program compiles correctly

    assert!(!program.operations.is_empty());

    // Verify all operations are gates
    for op in &program.operations {
        assert!(matches!(op, Operation::Gate { .. }));
    }
}

#[test]
fn test_nested_functions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Test nested mathematical functions
        rx(sin(cos(0))) q[0];        // sin(cos(0)) = sin(1)
        ry(sqrt(exp(ln(4)))) q[0];   // sqrt(exp(ln(4))) = sqrt(4) = 2
        RZ(cos(sin(pi/2))) q[0];     // cos(sin(pi/2)) = cos(1)
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_functions_with_expressions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // Test functions with complex expressions
        rx(sin(pi/6 + pi/3)) q[0];    // sin(pi/2) = 1
        ry(cos(2*pi - pi)) q[0];      // cos(pi) = -1
        RZ(sqrt(2*2 + 3*3)) q[0];     // sqrt(13)
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_error_cases() {
    // Test ln of negative number - parsing should succeed
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];
        rx(ln(-1)) q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);
    // The parsing should fail because ln(-1) is evaluated during parsing for gate parameters
    assert!(result.is_err());
    if let Err(e) = result {
        assert!(e.to_string().contains("ln(-1) is undefined"));
    }

    // Test sqrt of negative number
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];
        rx(sqrt(-4)) q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);
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
        include "qelib1.inc";
        qreg q[1];

        gate mygate(theta) q {
            rx(sin(theta)) q;
            ry(cos(theta)) q;
            RZ(sqrt(theta)) q;
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
        include "qelib1.inc";
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
    assert!(!program.operations.is_empty());
}

#[test]
fn test_evaluation_accuracy() {
    // Expression is already imported at the top of the file

    // Test sin
    let expr = Expression::FunctionCall {
        name: "sin".to_string(),
        args: vec![Expression::Float(PI / 2.0)],
    };
    assert!((expr.evaluate_with_context(None).unwrap() - 1.0).abs() < 1e-10);

    // Test cos
    let expr = Expression::FunctionCall {
        name: "cos".to_string(),
        args: vec![Expression::Float(0.0)],
    };
    assert!((expr.evaluate_with_context(None).unwrap() - 1.0).abs() < 1e-10);

    // Test tan
    let expr = Expression::FunctionCall {
        name: "tan".to_string(),
        args: vec![Expression::Float(PI / 4.0)],
    };
    assert!((expr.evaluate_with_context(None).unwrap() - 1.0).abs() < 1e-10);

    // Test exp
    let expr = Expression::FunctionCall {
        name: "exp".to_string(),
        args: vec![Expression::Float(0.0)],
    };
    assert!((expr.evaluate_with_context(None).unwrap() - 1.0).abs() < 1e-10);

    // Test ln
    let expr = Expression::FunctionCall {
        name: "ln".to_string(),
        args: vec![Expression::Float(std::f64::consts::E)],
    };
    assert!((expr.evaluate_with_context(None).unwrap() - 1.0).abs() < 1e-10);

    // Test sqrt
    let expr = Expression::FunctionCall {
        name: "sqrt".to_string(),
        args: vec![Expression::Float(4.0)],
    };
    assert!((expr.evaluate_with_context(None).unwrap() - 2.0).abs() < 1e-10);
}

#[test]
fn test_trig_identity_with_measurement() {
    use pecos_engines::{MonteCarloEngine, PassThroughNoiseModel};
    use pecos_qasm::QASMEngine;
    use std::str::FromStr;

    // Test that sin²(π/6) + cos²(π/6) = 1 through quantum measurement
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        // sin²(π/6) + cos²(π/6) = 0.25 + 0.75 = 1.0
        // To test, we'll multiply by π to get a π rotation
        rx((sin(pi/6)**2 + cos(pi/6)**2) * pi) q[0];

        // Measure the qubit (after π rotation, should see state |1⟩)
        measure q[0] -> c[0];
    "#;

    // Run the simulation with multiple shots
    let engine = QASMEngine::from_str(qasm).unwrap();

    let results = MonteCarloEngine::run_with_noise_model(
        Box::new(engine),
        Box::new(PassThroughNoiseModel),
        100, // 100 shots
        1,
        Some(42), // Fixed seed for deterministic results
    )
    .unwrap()
    .register_shots;

    // Assert we have results
    assert!(results.contains_key("c"));
    assert_eq!(results["c"].len(), 100);

    // Since sin²(π/6) + cos²(π/6) = 1.0, and we're doing rx(1.0 * π) = rx(π)
    // The qubit should be in state |1⟩, so all measurements should be 1
    for &value in &results["c"] {
        assert_eq!(value, 1, "Expected all measurements to be 1 after rx(π)");
    }

    println!("Trigonometric identity verified: all measurements are 1");
}

#[test]
fn test_trig_identity_various_angles() {
    use pecos_engines::{MonteCarloEngine, PassThroughNoiseModel};
    use pecos_qasm::QASMEngine;
    use std::str::FromStr;

    // Test multiple angles to verify sin²(x) + cos²(x) = 1 always holds
    let test_angles = ["pi/4", "pi/3", "2*pi/3", "3*pi/4"];

    for angle in &test_angles {
        let qasm = format!(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];

            // sin²({angle}) + cos²({angle}) should = 1.0
            rx((sin({angle})**2 + cos({angle})**2) * pi) q[0];

            // Measure the qubit (after π rotation, should see state |1⟩)
            measure q[0] -> c[0];
        "#
        );

        // Run the simulation
        let engine = QASMEngine::from_str(&qasm).unwrap();

        let results = MonteCarloEngine::run_with_noise_model(
            Box::new(engine),
            Box::new(PassThroughNoiseModel),
            50, // 50 shots per angle
            1,
            Some(42), // Fixed seed for deterministic results
        )
        .unwrap()
        .register_shots;

        // Assert we have results
        assert!(results.contains_key("c"));
        assert_eq!(results["c"].len(), 50);

        // For rx(π), all measurements should be 1
        for &value in &results["c"] {
            assert_eq!(
                value, 1,
                "Expected all measurements to be 1 for angle {angle} after rx(π)"
            );
        }

        println!("Trigonometric identity verified for angle {angle}: all measurements are 1");
    }
}

#[test]
fn test_trig_identity_exact_value() {
    // Test that the expression evaluates to exactly 1.0
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];

        // Test exact evaluation
        rx(sin(pi/3)**2 + cos(pi/3)**2) q[0];
    "#;

    let _program = QASMParser::parse_str(qasm).unwrap();

    // For direct evaluation, let's create an Expression manually

    // Create the trigonometric identity expression: sin²(π/3) + cos²(π/3)
    let sin_expr = Expression::FunctionCall {
        name: "sin".to_string(),
        args: vec![Expression::BinaryOp {
            op: "/".to_string(),
            left: Box::new(Expression::Pi),
            right: Box::new(Expression::Float(3.0)),
        }],
    };

    let sin_squared = Expression::BinaryOp {
        op: "**".to_string(),
        left: Box::new(sin_expr),
        right: Box::new(Expression::Float(2.0)),
    };

    let cos_expr = Expression::FunctionCall {
        name: "cos".to_string(),
        args: vec![Expression::BinaryOp {
            op: "/".to_string(),
            left: Box::new(Expression::Pi),
            right: Box::new(Expression::Float(3.0)),
        }],
    };

    let cos_squared = Expression::BinaryOp {
        op: "**".to_string(),
        left: Box::new(cos_expr),
        right: Box::new(Expression::Float(2.0)),
    };

    let trig_identity = Expression::BinaryOp {
        op: "+".to_string(),
        left: Box::new(sin_squared),
        right: Box::new(cos_squared),
    };

    // Evaluate the expression
    let value = evaluate_param_expr(&trig_identity);

    // Should be exactly 1.0 (within floating point precision)
    assert!(
        (value - 1.0).abs() < 1e-10,
        "sin²(π/3) + cos²(π/3) should equal 1.0, got {value}"
    );
    println!("Exact evaluation: sin²(π/3) + cos²(π/3) = {value}");
}

// Helper function to evaluate an Expression
fn evaluate_param_expr(expr: &Expression) -> f64 {
    // Since this is a test helper and we don't have parameters,
    // use evaluate_with_context() which handles basic evaluation
    expr.evaluate_with_context(None)
        .expect("Failed to evaluate expression")
}
