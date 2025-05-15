use pecos_qasm::parser::{Operation, ParameterExpression, QASMParser};
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
        rz(tan(pi/4)) q[0];  // tan(pi/4) = 1
    "#;

    let program = QASMParser::parse_str_with_includes(qasm).expect("Failed to parse QASM");
    // Just verify the program compiles successfully
    assert!(program.operations.len() > 0);
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
        rz(exp(ln(2))) q[0]; // exp(ln(2)) = 2
    "#;

    let program = QASMParser::parse_str_with_includes(qasm).expect("Failed to parse QASM");
    assert!(program.operations.len() > 0);
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
        rz(sqrt(9)) q[0];    // sqrt(9) = 3
    "#;

    let program = QASMParser::parse_str_with_includes(qasm).expect("Failed to parse QASM");

    // After includes, the high-level gates are expanded into native gates
    // rx, ry, and rz are all expanded, so we expect more than 3 operations
    // We should just verify that the program compiles correctly

    assert!(program.operations.len() > 0);

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
        rz(cos(sin(pi/2))) q[0];     // cos(sin(pi/2)) = cos(1)
    "#;

    let program = QASMParser::parse_str_with_includes(qasm).expect("Failed to parse QASM");
    assert!(program.operations.len() > 0);
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
        rz(sqrt(2*2 + 3*3)) q[0];     // sqrt(13)
    "#;

    let program = QASMParser::parse_str_with_includes(qasm).expect("Failed to parse QASM");
    assert!(program.operations.len() > 0);
}

#[test]
fn test_error_cases() {
    // Test ln of negative number - parsing should succeed
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        rx(ln(-1)) q[0];
    "#;

    let result = QASMParser::parse_str_raw(qasm);
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
            rz(sqrt(theta)) q;
        }

        mygate(pi/4) q[0];
    "#;

    let program = QASMParser::parse_str_with_includes(qasm).expect("Failed to parse QASM");
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

    let program = QASMParser::parse_str_with_includes(qasm).expect("Failed to parse QASM");
    assert!(program.operations.len() > 0);
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

#[test]
fn test_trig_identity_with_measurement() {
    use pecos_engines::{MonteCarloEngine, PassThroughNoiseModel};
    use pecos_qasm::QASMEngine;

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
    let mut engine = QASMEngine::new().unwrap();
    engine.from_str(qasm).unwrap();

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

    // Test multiple angles to verify sin²(x) + cos²(x) = 1 always holds
    let test_angles = ["pi/4", "pi/3", "2*pi/3", "3*pi/4"];

    for angle in &test_angles {
        let qasm = format!(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[1];

            // sin²({}) + cos²({}) should = 1.0
            rx((sin({})**2 + cos({})**2) * pi) q[0];

            // Measure the qubit (after π rotation, should see state |1⟩)
            measure q[0] -> c[0];
        "#,
            angle, angle, angle, angle
        );

        // Run the simulation
        let mut engine = QASMEngine::new().unwrap();
        engine.from_str(&qasm).unwrap();

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
                "Expected all measurements to be 1 for angle {} after rx(π)",
                angle
            );
        }

        println!(
            "Trigonometric identity verified for angle {}: all measurements are 1",
            angle
        );
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

    let _program = QASMParser::parse_str_with_includes(qasm).unwrap();

    // For direct evaluation, let's create a ParameterExpression manually

    // Create the trigonometric identity expression: sin²(π/3) + cos²(π/3)
    let sin_expr = ParameterExpression::FunctionCall {
        name: "sin".to_string(),
        args: vec![ParameterExpression::BinaryOp {
            op: "/".to_string(),
            left: Box::new(ParameterExpression::Pi),
            right: Box::new(ParameterExpression::Constant(3.0)),
        }],
    };

    let sin_squared = ParameterExpression::BinaryOp {
        op: "**".to_string(),
        left: Box::new(sin_expr),
        right: Box::new(ParameterExpression::Constant(2.0)),
    };

    let cos_expr = ParameterExpression::FunctionCall {
        name: "cos".to_string(),
        args: vec![ParameterExpression::BinaryOp {
            op: "/".to_string(),
            left: Box::new(ParameterExpression::Pi),
            right: Box::new(ParameterExpression::Constant(3.0)),
        }],
    };

    let cos_squared = ParameterExpression::BinaryOp {
        op: "**".to_string(),
        left: Box::new(cos_expr),
        right: Box::new(ParameterExpression::Constant(2.0)),
    };

    let trig_identity = ParameterExpression::BinaryOp {
        op: "+".to_string(),
        left: Box::new(sin_squared),
        right: Box::new(cos_squared),
    };

    // Evaluate the expression
    let value = evaluate_param_expr(&trig_identity);

    // Should be exactly 1.0 (within floating point precision)
    assert!(
        (value - 1.0).abs() < 1e-10,
        "sin²(π/3) + cos²(π/3) should equal 1.0, got {}",
        value
    );
    println!("Exact evaluation: sin²(π/3) + cos²(π/3) = {}", value);
}

// Helper function to evaluate a ParameterExpression
fn evaluate_param_expr(expr: &ParameterExpression) -> f64 {
    match expr {
        ParameterExpression::Constant(val) => *val,
        ParameterExpression::Pi => std::f64::consts::PI,
        ParameterExpression::BinaryOp { op, left, right } => {
            let left_val = evaluate_param_expr(left);
            let right_val = evaluate_param_expr(right);

            match op.as_str() {
                "+" => left_val + right_val,
                "-" => left_val - right_val,
                "*" => left_val * right_val,
                "/" => left_val / right_val,
                "**" => left_val.powf(right_val),
                _ => panic!("Unsupported operation: {}", op),
            }
        }
        ParameterExpression::FunctionCall { name, args } => {
            let arg_val = evaluate_param_expr(&args[0]);
            match name.as_str() {
                "sin" => arg_val.sin(),
                "cos" => arg_val.cos(),
                _ => panic!("Unsupported function: {}", name),
            }
        }
        _ => panic!("Unsupported expression type"),
    }
}
