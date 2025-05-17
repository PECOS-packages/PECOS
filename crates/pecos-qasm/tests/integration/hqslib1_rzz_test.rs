use pecos_qasm::{Operation, QASMParser};

#[test]
fn test_hqslib1_rzz_sequence() {
    // Test RZZ gate sequence from hqslib1 with various parameter values
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[2];
        RZZ(0.3*pi) q[0],q[1];
        RZZ(0.4*pi) q[0],q[1];
        RZZ(-0.6*pi) q[0],q[1];
        RZZ(1.0*pi) q[0],q[1];
        RZZ(-0.2999999999999998*pi) q[0],q[1];
        RZZ(0.6*pi) q[0],q[1];
        RZZ(1.0*pi) q[0],q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM with RZZ gates");

    // All operations should be gate operations
    let gate_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::Gate { .. }))
        .count();

    assert_eq!(gate_count, 7, "Expected 7 RZZ gates");

    // Verify all gates are RZZ
    let rzz_gates: Vec<_> = program
        .operations
        .iter()
        .filter_map(|op| match op {
            Operation::Gate {
                name,
                parameters,
                qubits,
            } => {
                if name == "RZZ" {
                    Some((parameters.clone(), qubits.clone()))
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect();

    assert_eq!(rzz_gates.len(), 7, "All gates should be RZZ");

    // Check each gate has correct structure
    for (i, (params, qubits)) in rzz_gates.iter().enumerate() {
        assert_eq!(params.len(), 1, "RZZ gate {i} should have 1 parameter");
        assert_eq!(qubits.len(), 2, "RZZ gate {i} should have 2 qubits");
        assert_eq!(qubits[0], 0, "RZZ gate {i} first qubit should be q[0]");
        assert_eq!(qubits[1], 1, "RZZ gate {i} second qubit should be q[1]");
    }

    // Verify the parameter values (approximate due to pi calculations)
    let expected_params = [
        0.3 * std::f64::consts::PI,
        0.4 * std::f64::consts::PI,
        -0.6 * std::f64::consts::PI,
        1.0 * std::f64::consts::PI,
        -0.299_999_999_999_999_8 * std::f64::consts::PI,
        0.6 * std::f64::consts::PI,
        1.0 * std::f64::consts::PI,
    ];

    for (i, ((params, _), expected)) in rzz_gates.iter().zip(expected_params.iter()).enumerate() {
        let delta = (params[0] - expected).abs();
        assert!(
            delta < 1e-10,
            "RZZ gate {} parameter mismatch: expected {}, got {}",
            i,
            expected,
            params[0]
        );
    }
}

#[test]
fn test_rzz_with_negative_parameters() {
    // Test that RZZ handles negative parameters correctly
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[2];
        RZZ(-pi/2) q[0],q[1];
        RZZ(-pi) q[0],q[1];
        RZZ(-2*pi) q[0],q[1];
    "#;

    let program =
        QASMParser::parse_str(qasm).expect("Failed to parse QASM with negative RZZ parameters");

    let rzz_parameters: Vec<f64> = program
        .operations
        .iter()
        .filter_map(|op| match op {
            Operation::Gate {
                name, parameters, ..
            } => {
                if name == "RZZ" {
                    Some(parameters[0])
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect();

    assert_eq!(rzz_parameters.len(), 3);

    // Check negative values are preserved
    assert!(
        rzz_parameters[0] < 0.0,
        "First parameter should be negative"
    );
    assert!(
        rzz_parameters[1] < 0.0,
        "Second parameter should be negative"
    );
    assert!(
        rzz_parameters[2] < 0.0,
        "Third parameter should be negative"
    );

    // Check approximate values
    assert!((rzz_parameters[0] - (-std::f64::consts::PI / 2.0)).abs() < 1e-10);
    assert!((rzz_parameters[1] - (-std::f64::consts::PI)).abs() < 1e-10);
    assert!((rzz_parameters[2] - (-2.0 * std::f64::consts::PI)).abs() < 1e-10);
}

#[test]
fn test_rzz_mixed_with_other_gates() {
    // Test RZZ gates mixed with other operations
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[3];
        creg c[3];

        h q[0];
        h q[1];

        RZZ(pi/4) q[0],q[1];
        cx q[1],q[2];
        RZZ(pi/3) q[1],q[2];

        measure q -> c;
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse mixed gate QASM");

    // Count different operation types
    let h_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::Gate { name, .. } if name == "H"))
        .count();
    let rzz_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::Gate { name, .. } if name == "RZZ"))
        .count();
    let cx_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::Gate { name, .. } if name == "CX"))
        .count();
    let measure_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::Measure { .. }))
        .count();

    assert_eq!(h_count, 2, "Expected 2 Hadamard gates");
    assert_eq!(rzz_count, 2, "Expected 2 RZZ gates");
    assert_eq!(cx_count, 1, "Expected 1 CX gate");
    assert_eq!(measure_count, 3, "Expected 3 measurements");

    // Verify the sequence order
    let gate_sequence: Vec<&str> = program
        .operations
        .iter()
        .filter_map(|op| match op {
            Operation::Gate { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(gate_sequence, vec!["H", "H", "RZZ", "CX", "RZZ"]);
}
