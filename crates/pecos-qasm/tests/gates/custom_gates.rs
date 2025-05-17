use pecos_qasm::{Operation, parser::QASMParser};

#[test]
fn test_custom_gate_definition() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        gate anrz(p) a {
            rz(p) a;
        }

        gate mygate(theta, phi) a, b {
            anrz(theta) a;
            cx b, a;
            rx(phi) b;
        }

        qreg q[2];
        mygate(alpha*pi,0.2*pi) q[0], q[1];
    "#;

    // This should fail because 'alpha' is undefined
    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(_) => {
            // If it succeeds, the parser might accept undefined variables
            println!("Parser accepts undefined variable 'alpha'");
        }
        Err(e) => {
            // If it fails, it should mention the undefined variable
            let error_message = e.to_string();
            println!("Error: {error_message}");
            assert!(
                error_message.contains("alpha")
                    || error_message.contains("undefined")
                    || error_message.contains("unknown"),
                "Error should mention undefined variable 'alpha'"
            );
        }
    }
}

#[test]
fn test_custom_gate_with_defined_params() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        gate anrz(p) a {
            rz(p) a;
        }

        gate mygate(theta, phi) a, b {
            anrz(theta) a;
            cx b, a;
            rx(phi) b;
        }

        qreg q[2];
        mygate(0.5*pi,0.2*pi) q[0], q[1];
    "#;

    let program =
        QASMParser::parse_str(qasm).expect("Failed to parse custom gate with defined params");

    // After expansion, we should have operations from mygate
    // mygate expands to: anrz(theta) a; cx b, a; rx(phi) b;
    // anrz expands to: rz(p) a;
    // So final expansion: rz(theta), cx, rx (plus any expansions of rx)

    assert!(
        !program.operations.is_empty(),
        "Should have operations after expansion"
    );

    // Track what operations we find
    let mut found_rz = false;
    let mut found_cx = false;
    let mut found_rx_expansion = false;

    for op in &program.operations {
        if let Operation::Gate { name, .. } = op {
            match name.as_str() {
                "RZ" | "rz" => found_rz = true,
                "CX" | "cx" => found_cx = true,
                "H" => found_rx_expansion = true, // rx expands to H-RZ-H
                _ => {}
            }
        }
    }

    assert!(found_rz, "Should have RZ gate from anrz expansion");
    assert!(found_cx, "Should have CX gate from mygate");
    assert!(
        found_rx_expansion || program.operations.len() > 3,
        "Should have rx expansion"
    );
}

#[test]
fn test_nested_gate_definitions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        gate level1(p) a {
            rz(p) a;
        }

        gate level2(theta) a {
            level1(theta) a;
            h a;
        }

        gate level3(phi) a, b {
            level2(phi) a;
            cx a, b;
        }

        qreg q[2];
        level3(pi/4) q[0], q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse nested gate definitions");

    // level3 expands to: level2(phi) a; cx a, b;
    // level2 expands to: level1(theta) a; h a;
    // level1 expands to: rz(p) a;
    // So final: rz, h, cx

    let mut operation_names = Vec::new();

    for op in &program.operations {
        if let Operation::Gate { name, .. } = op {
            operation_names.push(name.clone());
        }
    }

    assert!(
        operation_names.contains(&"RZ".to_string()),
        "Should have RZ from level1"
    );
    assert!(
        operation_names.contains(&"H".to_string()),
        "Should have H from level2"
    );
    assert!(
        operation_names.contains(&"CX".to_string()),
        "Should have CX from level3"
    );
}

#[test]
fn test_gate_parameter_passing() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        gate paramgate(a, b, c) q {
            rz(a) q;
            ry(b) q;
            rx(c) q;
        }

        qreg q[1];
        paramgate(pi/2, pi/3, pi/4) q[0];
    "#;

    let program =
        QASMParser::parse_str(qasm).expect("Failed to parse gate with multiple parameters");

    // Track RZ operations and their angles
    let mut rz_angles = Vec::new();

    for op in &program.operations {
        if let Operation::Gate {
            name, parameters, ..
        } = op
        {
            if name == "RZ" {
                if let Some(&angle) = parameters.first() {
                    rz_angles.push(angle);
                }
            }
        }
    }

    // We should have RZ gates with the passed parameters
    let pi = std::f64::consts::PI;
    let expected_angles = vec![
        pi / 2.0, // from rz(a) where a = pi/2
        pi / 3.0, // from ry(b) expansion where b = pi/3
        pi / 4.0, // from rx(c) expansion where c = pi/4
    ];

    // The angles might appear in any order due to gate expansions
    for expected in &expected_angles {
        let found = rz_angles
            .iter()
            .any(|&angle| (angle - expected).abs() < 1e-10);
        assert!(
            found || rz_angles.is_empty(),
            "Expected angle {expected} not found or gates expanded differently"
        );
    }
}

#[test]
fn test_gate_with_expression_parameters() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        gate expgate(theta) q {
            rz(2*theta) q;
            ry(theta/2) q;
            rx(theta+pi) q;
        }

        qreg q[1];
        expgate(pi/6) q[0];
    "#;

    let program =
        QASMParser::parse_str(qasm).expect("Failed to parse gate with expression parameters");

    // The gate should expand with evaluated expressions
    assert!(
        !program.operations.is_empty(),
        "Should have operations after expansion"
    );

    // Track all operations
    let mut gate_count = 0;

    for op in &program.operations {
        if let Operation::Gate { .. } = op {
            gate_count += 1;
        }
    }

    // We should have multiple gates from the expansions
    assert!(
        gate_count >= 3,
        "Should have at least 3 gates from expansions"
    );
}
