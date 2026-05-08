use pecos_qasm::{Operation, parser::QASMParser};

// Helper function to extract gate name from operation
fn get_gate_name(op: &Operation) -> Option<String> {
    match op {
        Operation::Gate { name, .. } => Some(name.clone()),
        Operation::NativeGate(gate) => Some(format!("{:?}", gate.gate_type)),
        _ => None,
    }
}

// Helper function to extract qubits from operation
fn get_gate_qubits(op: &Operation) -> Vec<usize> {
    match op {
        Operation::Gate { qubits, .. } => qubits.clone(),
        Operation::NativeGate(gate) => gate.qubits.iter().map(|q| q.0).collect(),
        _ => vec![],
    }
}

#[test]
fn test_mixed_gates_circuit() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[10];
        creg c[4];
        rz(1.5*pi) q[4];
        rx(0.085*pi) q[7];
        rz(0.5*pi) q[3];
        cx q[0], q[3];
        rz(1.5*pi) q[3];
        rx(2.25*pi) q[3];
        cz q[0] ,q[5];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse mixed gates circuit");

    // Count gate types and track operations
    let mut gate_count = 0;
    let mut gate_types = std::collections::BTreeMap::new();
    let mut qubit_usage = std::collections::BTreeSet::new();

    for op in &program.operations {
        if let Some(name) = get_gate_name(op) {
            gate_count += 1;
            *gate_types.entry(name.to_lowercase()).or_insert(0) += 1;

            for qubit in get_gate_qubits(op) {
                qubit_usage.insert(qubit);
            }
        }
    }

    // These gates will be expanded
    // rz stays as rz (or RZ)
    // rx expands to H-RZ-H
    // cx stays as cx (or CX)
    // cz expands to H-CX-H

    // Since we don't know the exact expansion pattern, let's check broadly
    assert!(
        gate_count > 7,
        "Should have more than 7 operations after expansion"
    );

    // Check that all used qubits are within bounds
    for &qubit in &qubit_usage {
        assert!(qubit < 10, "All qubits should be within register bounds");
    }

    // Verify that specific qubits were used
    assert!(qubit_usage.contains(&0), "Qubit 0 should be used");
    assert!(qubit_usage.contains(&3), "Qubit 3 should be used");
    assert!(qubit_usage.contains(&4), "Qubit 4 should be used");
    assert!(qubit_usage.contains(&5), "Qubit 5 should be used");
    assert!(qubit_usage.contains(&7), "Qubit 7 should be used");

    // Check that classical register is not used in quantum operations
    for op in &program.operations {
        if let Operation::Gate { .. } = op {
            // This is a quantum operation, should not involve classical registers
            // (This is implicitly true since Gate operations only have qubit indices)
        }
    }
}

#[test]
fn test_angle_precision() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[10];
        rz(1.5*pi) q[4];
        rx(0.085*pi) q[7];
        rz(0.5*pi) q[3];
        rx(2.25*pi) q[3];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse angle precision test");

    // Track the RZ gates and their angles after expansion
    let mut rz_angles = Vec::new();

    for op in &program.operations {
        match op {
            Operation::Gate {
                name, parameters, ..
            } if name == "RZ" => {
                if let Some(&angle) = parameters.first() {
                    rz_angles.push(angle);
                }
            }
            Operation::NativeGate(gate)
                if matches!(gate.gate_type, pecos_core::gate_type::GateType::RZ) =>
            {
                // Rotation gate angles are now stored in gate.angles as Angle64
                if let Some(angle) = gate.angles.first() {
                    rz_angles.push(angle.to_radians());
                }
            }
            _ => {}
        }
    }

    // After expansion, we should have RZ gates with various angles
    assert!(
        !rz_angles.is_empty(),
        "Should have RZ gates after expansion"
    );

    // Check that angles are preserved with reasonable precision
    // Note: Angle64 normalizes angles to [0, 2π), so 2.25*pi becomes 0.25*pi
    let pi = std::f64::consts::PI;
    let expected_angles = vec![
        1.5 * pi, // rz(1.5*pi)
        0.5 * pi, // rz(0.5*pi)
        // rx gates will contribute their angles too
        0.085 * pi, // from rx(0.085*pi)
        0.25 * pi,  // from rx(2.25*pi) - normalized: 2.25*pi mod 2*pi = 0.25*pi
    ];

    // The angles might not be in the same order after expansion
    for expected in &expected_angles {
        let found = rz_angles
            .iter()
            .any(|&angle| (angle - expected).abs() < 1e-6); // Relaxed tolerance for angle normalization
        assert!(found, "Expected angle {expected} not found in RZ gates");
    }
}

#[test]
fn test_gate_sequence() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[5];
        rz(pi) q[3];
        cx q[0], q[3];
        rz(pi) q[3];
        rx(pi) q[3];
        cz q[0], q[3];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse gate sequence");

    // Track operations on qubit 3
    let mut q3_operations = Vec::new();

    for op in &program.operations {
        match op {
            Operation::Gate { name, qubits, .. } if qubits.contains(&3) => {
                q3_operations.push(name.clone());
            }
            Operation::NativeGate(gate) if gate.qubits.iter().any(|q| q.0 == 3) => {
                q3_operations.push(format!("{:?}", gate.gate_type));
            }
            _ => {}
        }
    }

    // Qubit 3 should have multiple operations
    assert!(
        q3_operations.len() > 5,
        "Qubit 3 should have multiple operations after expansion"
    );

    // Check that the operations include expected gate types
    assert!(
        q3_operations.iter().any(|g| g == "RZ"),
        "Should have RZ gates on qubit 3"
    );
    assert!(
        q3_operations.iter().any(|g| g == "CX" || g == "CNOT"),
        "Should have CX gates on qubit 3"
    );
    assert!(
        q3_operations.iter().any(|g| g == "H" || g == "Hadamard"),
        "Should have H gates from expansions"
    );
}

#[test]
fn test_two_qubit_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[6];
        cx q[0], q[3];
        cz q[0], q[5];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse two-qubit gates");

    // Find all two-qubit gates
    let mut two_qubit_gates = Vec::new();

    for op in &program.operations {
        match op {
            Operation::Gate { name, qubits, .. } if qubits.len() == 2 => {
                two_qubit_gates.push((name.clone(), qubits[0], qubits[1]));
            }
            Operation::NativeGate(gate) if gate.qubits.len() == 2 => {
                two_qubit_gates.push((
                    format!("{:?}", gate.gate_type),
                    gate.qubits[0].0,
                    gate.qubits[1].0,
                ));
            }
            _ => {}
        }
    }

    // We expect:
    // - CX from the cx instruction
    // - CX from the cz expansion (cz -> H-CX-H)
    let cx_gates: Vec<_> = two_qubit_gates
        .iter()
        .filter(|(name, _, _)| name == "CX" || name == "CNOT")
        .collect();

    assert_eq!(cx_gates.len(), 2, "Should have 2 CX gates");

    // Check the connections
    assert!(
        cx_gates.iter().any(|(_, q1, q2)| *q1 == 0 && *q2 == 3),
        "Should have CX between qubits 0 and 3"
    );
    assert!(
        cx_gates.iter().any(|(_, q1, q2)| *q1 == 0 && *q2 == 5),
        "Should have CX between qubits 0 and 5 (from CZ expansion)"
    );
}
