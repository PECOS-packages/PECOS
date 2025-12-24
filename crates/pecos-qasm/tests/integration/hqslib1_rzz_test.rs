use pecos_core::prelude::GateType;
use pecos_qasm::{Operation, QASMParser};

// Helper function to count operations by gate type
fn count_gates_by_name(operations: &[Operation], gate_name: &str) -> usize {
    operations
        .iter()
        .filter(|op| match op {
            Operation::Gate { name, .. } => name.eq_ignore_ascii_case(gate_name),
            Operation::NativeGate(gate) => {
                let gate_type_str = format!("{:?}", gate.gate_type);
                gate_type_str.eq_ignore_ascii_case(gate_name)
                    || (gate_name.eq_ignore_ascii_case("h")
                        && matches!(gate.gate_type, GateType::H))
            }
            _ => false,
        })
        .count()
}

// Helper function to extract gate parameters
// Note: For NativeGate, rotation angles are now stored in gate.angles as Angle64
fn extract_gate_parameters(operations: &[Operation], gate_name: &str) -> Vec<f64> {
    operations
        .iter()
        .filter_map(|op| match op {
            Operation::Gate {
                name, parameters, ..
            } if name.eq_ignore_ascii_case(gate_name) => parameters.first().copied(),
            Operation::NativeGate(gate) => {
                let gate_type_str = format!("{:?}", gate.gate_type);
                if gate_type_str.eq_ignore_ascii_case(gate_name) {
                    // Rotation gate angles are stored in gate.angles as Angle64
                    gate.angles.first().map(pecos_core::Angle::to_radians)
                } else {
                    None
                }
            }
            _ => None,
        })
        .collect()
}

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
        .filter(|op| matches!(op, Operation::Gate { .. } | Operation::NativeGate(_)))
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
            Operation::NativeGate(gate) => {
                let gate_type_str = format!("{:?}", gate.gate_type);
                if gate_type_str == "RZZ" {
                    let qubits = gate.qubits.iter().map(|q| q.0).collect();
                    // Rotation gate angles are now stored in gate.angles as Angle64
                    let params: Vec<f64> = gate
                        .angles
                        .iter()
                        .map(pecos_core::Angle::to_radians)
                        .collect();
                    Some((params, qubits))
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
    // Note: Angle64 normalizes angles to [0, 2π), so negative angles become positive
    // -0.6*PI becomes 1.4*PI, -0.3*PI becomes 1.7*PI
    let pi = std::f64::consts::PI;
    let expected_params = [
        0.3 * pi,                             // 0.3*pi
        0.4 * pi,                             // 0.4*pi
        1.4 * pi,                             // -0.6*pi normalized to 1.4*pi
        1.0 * pi,                             // 1.0*pi
        (2.0 - 0.299_999_999_999_999_8) * pi, // -0.3*pi normalized to ~1.7*pi
        0.6 * pi,                             // 0.6*pi
        1.0 * pi,                             // 1.0*pi
    ];

    for (i, ((params, _), expected)) in rzz_gates.iter().zip(expected_params.iter()).enumerate() {
        let delta = (params[0] - expected).abs();
        assert!(
            delta < 1e-6, // Relaxed tolerance for angle normalization
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
    // Note: Angle64 normalizes angles to [0, 2π), so negative angles become positive
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

    let rzz_parameters = extract_gate_parameters(&program.operations, "RZZ");

    assert_eq!(rzz_parameters.len(), 3);

    // Angle64 normalizes to [0, 2π), so:
    // -π/2 becomes 3π/2
    // -π becomes π
    // -2π becomes 0
    let pi = std::f64::consts::PI;

    // Check normalized values
    assert!(
        (rzz_parameters[0] - 3.0 * pi / 2.0).abs() < 1e-6,
        "First parameter should be 3π/2 (normalized from -π/2), got {}",
        rzz_parameters[0]
    );
    assert!(
        (rzz_parameters[1] - pi).abs() < 1e-6,
        "Second parameter should be π (normalized from -π), got {}",
        rzz_parameters[1]
    );
    assert!(
        rzz_parameters[2].abs() < 1e-6,
        "Third parameter should be 0 (normalized from -2π), got {}",
        rzz_parameters[2]
    );
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
    let h_count = count_gates_by_name(&program.operations, "H");
    let rzz_count = count_gates_by_name(&program.operations, "RZZ");
    let cx_count = count_gates_by_name(&program.operations, "CX");
    let measure_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::MeasureWithMapping { .. }))
        .count();

    assert_eq!(h_count, 2, "Expected 2 Hadamard gates");
    assert_eq!(rzz_count, 2, "Expected 2 RZZ gates");
    assert_eq!(cx_count, 1, "Expected 1 CX gate");
    assert_eq!(measure_count, 3, "Expected 3 measurements");

    // Verify the sequence order
    let gate_sequence: Vec<String> = program
        .operations
        .iter()
        .filter_map(|op| match op {
            Operation::Gate { name, .. } => Some(name.clone()),
            Operation::NativeGate(gate) => Some(format!("{:?}", gate.gate_type)),
            _ => None,
        })
        .collect();

    let expected_names = ["H", "H", "RZZ", "CX", "RZZ"];
    for (i, expected) in expected_names.iter().enumerate() {
        if i < gate_sequence.len() {
            assert!(
                gate_sequence[i] == *expected
                    || (expected == &"H" && gate_sequence[i] == "Hadamard")
                    || (expected == &"CX" && gate_sequence[i] == "CNOT"),
                "Expected {} at position {}, got {}",
                expected,
                i,
                gate_sequence[i]
            );
        }
    }
}
