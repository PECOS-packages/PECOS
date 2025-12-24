use pecos_core::prelude::GateType;
use pecos_qasm::{Operation, QASMParser};

// Helper function to check if an operation is a gate with a specific name
fn is_gate_with_name(op: &Operation, gate_name: &str) -> bool {
    match op {
        Operation::Gate { name, .. } => {
            name == gate_name || name.to_uppercase() == gate_name.to_uppercase()
        }
        Operation::NativeGate(gate) => {
            let gate_type_name = format!("{:?}", gate.gate_type).to_lowercase();
            let target_name = gate_name.to_lowercase();
            gate_type_name == target_name
                || (target_name == "cx" && gate_type_name == "cx")
                || (target_name == "cnot" && gate_type_name == "cx")
                || (target_name == "h" && gate_type_name == "h")
                || (target_name == "rz" && gate_type_name == "rz")
        }
        _ => false,
    }
}

// Helper function to extract gate name from operation
fn get_gate_name(op: &Operation) -> Option<String> {
    match op {
        Operation::Gate { name, .. } => Some(name.clone()),
        Operation::NativeGate(gate) => Some(format!("{:?}", gate.gate_type)),
        _ => None,
    }
}

#[test]
fn test_controlled_rotation_gates() {
    // Test controlled rotation gates expansion
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        // Test controlled rotation gates
        qreg q[4];
        crz(0.3 * pi) q[0],q[1];
        crx(0.5 * pi) q[2],q[1];
        cry(0.5 * pi) q[3],q[0];
    "#;

    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(program) => {
            println!("Parsed {} operations", program.operations.len());

            // Count specific gate types
            let cx_count = program
                .operations
                .iter()
                .filter(|op| is_gate_with_name(op, "CX"))
                .count();

            let rz_count = program
                .operations
                .iter()
                .filter(|op| is_gate_with_name(op, "RZ"))
                .count();

            let h_count = program
                .operations
                .iter()
                .filter(|op| is_gate_with_name(op, "H"))
                .count();

            println!("Gate counts - CX: {cx_count}, RZ: {rz_count}, H: {h_count}");

            // Verify the operations expanded correctly
            // Each controlled rotation requires 2 CX gates (3 gates total * 2 = 6)
            assert_eq!(
                cx_count, 6,
                "Expected 6 CX gates from 3 controlled rotations"
            );

            // crz contributes 2 RZ gates, crx uses ry which expands to rx (h-rz-h),
            // and cry uses ry gates
            assert!(
                rz_count > 2,
                "Expected multiple RZ gates from controlled rotations"
            );

            // The rx gates expand to h-rz-h patterns
            assert!(h_count > 0, "Expected H gates from the expansions");
        }
        Err(e) => {
            panic!("Failed to parse controlled rotation gates: {e}");
        }
    }
}

#[test]
fn test_crz_expansion() {
    // Test specific expansion of crz gate
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        crz(pi/2) q[0],q[1];
    "#;

    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(program) => {
            println!(
                "CRZ expansion resulted in {} operations",
                program.operations.len()
            );

            // crz(theta) expands to: rz(theta/2) b; cx a,b; rz(-theta/2) b; cx a,b;
            assert_eq!(
                program.operations.len(),
                4,
                "CRZ should expand to 4 operations"
            );

            // Verify the sequence
            match &program.operations[0] {
                Operation::Gate {
                    name,
                    parameters,
                    qubits,
                } => {
                    assert_eq!(name, "RZ");
                    assert_eq!(qubits, &[1]); // Target qubit
                    assert!(
                        (parameters[0] - std::f64::consts::PI / 4.0).abs() < 1e-10,
                        "First RZ should have angle pi/4"
                    );
                }
                Operation::NativeGate(gate) if matches!(gate.gate_type, GateType::RZ) => {
                    assert_eq!(gate.qubits.len(), 1);
                    assert_eq!(gate.qubits[0].0, 1); // Target qubit
                    // For native gates, the angle is in the angles field as Angle64
                    assert_eq!(gate.angles.len(), 1);
                    assert!(
                        (gate.angles[0].to_radians() - std::f64::consts::PI / 4.0).abs() < 1e-10,
                        "First RZ should have angle pi/4"
                    );
                }
                _ => panic!("Expected RZ gate at position 0"),
            }

            match &program.operations[1] {
                Operation::Gate { name, qubits, .. } => {
                    assert_eq!(name, "CX");
                    assert_eq!(qubits, &[0, 1]); // Control, target
                }
                Operation::NativeGate(gate) if matches!(gate.gate_type, GateType::CX) => {
                    assert_eq!(gate.qubits.len(), 2);
                    assert_eq!(gate.qubits[0].0, 0); // Control
                    assert_eq!(gate.qubits[1].0, 1); // Target
                }
                _ => panic!("Expected CX gate at position 1"),
            }

            match &program.operations[2] {
                Operation::Gate {
                    name,
                    parameters,
                    qubits,
                } => {
                    assert_eq!(name, "RZ");
                    assert_eq!(qubits, &[1]); // Target qubit
                    assert!(
                        (parameters[0] + std::f64::consts::PI / 4.0).abs() < 1e-10,
                        "Second RZ should have angle -pi/4"
                    );
                }
                Operation::NativeGate(gate) if matches!(gate.gate_type, GateType::RZ) => {
                    assert_eq!(gate.qubits.len(), 1);
                    assert_eq!(gate.qubits[0].0, 1); // Target qubit
                    // For native gates, the angle is in the angles field as Angle64
                    // Note: Angle64 normalizes to [0, 2π), so -π/4 becomes 7π/4
                    assert_eq!(gate.angles.len(), 1);
                    let angle = gate.angles[0].to_radians();
                    let expected = 7.0 * std::f64::consts::PI / 4.0; // -pi/4 normalized to 7pi/4
                    assert!(
                        (angle - expected).abs() < 1e-10,
                        "Second RZ should have angle 7pi/4 (normalized from -pi/4), got {angle}"
                    );
                }
                _ => panic!("Expected RZ gate at position 2"),
            }

            match &program.operations[3] {
                Operation::Gate { name, qubits, .. } => {
                    assert_eq!(name, "CX");
                    assert_eq!(qubits, &[0, 1]); // Control, target
                }
                Operation::NativeGate(gate) if matches!(gate.gate_type, GateType::CX) => {
                    assert_eq!(gate.qubits.len(), 2);
                    assert_eq!(gate.qubits[0].0, 0); // Control
                    assert_eq!(gate.qubits[1].0, 1); // Target
                }
                _ => panic!("Expected CX gate at position 3"),
            }
        }
        Err(e) => {
            panic!("Failed to parse crz gate: {e}");
        }
    }
}

#[test]
fn test_crx_expansion() {
    // Test specific expansion of crx gate
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        crx(pi/2) q[0],q[1];
    "#;

    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(program) => {
            println!(
                "CRX expansion resulted in {} operations",
                program.operations.len()
            );

            // crx expands to a controlled version of rx
            // It should include CX gates and rotations
            let cx_count = program
                .operations
                .iter()
                .filter(|op| is_gate_with_name(op, "CX"))
                .count();
            assert_eq!(cx_count, 2, "CRX should include 2 CX gates");

            // Look for the overall pattern of gate types
            let gate_types: Vec<String> = program
                .operations
                .iter()
                .filter_map(get_gate_name)
                .collect();

            println!("CRX gate sequence: {gate_types:?}");

            // crx uses ry gates which expand to rx (h-rz-h) patterns
            assert!(
                gate_types
                    .iter()
                    .any(|name| name.to_uppercase() == "H" || name.to_uppercase() == "HADAMARD"),
                "CRX should contain H gates from RY expansion"
            );
            assert!(
                gate_types.iter().any(|name| name.to_uppercase() == "RZ"),
                "CRX should contain RZ gates from RY expansion"
            );
            assert!(
                gate_types
                    .iter()
                    .any(|name| name.to_uppercase() == "CX" || name.to_uppercase() == "CNOT"),
                "CRX should include CX gates"
            );
        }
        Err(e) => {
            panic!("Failed to parse crx gate: {e}");
        }
    }
}

#[test]
fn test_cry_expansion() {
    // Test specific expansion of cry gate
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        cry(pi/2) q[0],q[1];
    "#;

    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(program) => {
            println!(
                "CRY expansion resulted in {} operations",
                program.operations.len()
            );

            // cry uses ry gates which expand to rx (h-rz-h) patterns
            // Each ry expands to: rx(-pi/2); rz(theta); rx(pi/2)
            // And each rx expands to: h; rz(angle); h
            // So we expect more than 4 operations due to expansions
            assert!(
                program.operations.len() > 4,
                "CRY should expand to more than 4 operations due to ry expansion"
            );

            // Count gate types
            let cx_count = program
                .operations
                .iter()
                .filter(|op| is_gate_with_name(op, "CX"))
                .count();
            let h_count = program
                .operations
                .iter()
                .filter(|op| is_gate_with_name(op, "H"))
                .count();
            let rz_count = program
                .operations
                .iter()
                .filter(|op| is_gate_with_name(op, "RZ"))
                .count();

            println!("CRY gate counts - CX: {cx_count}, H: {h_count}, RZ: {rz_count}");

            // Should have 2 CX gates from the original cry structure
            assert_eq!(cx_count, 2, "CRY should have 2 CX gates");

            // Should have multiple H and RZ gates from ry expansion
            assert!(h_count > 0, "CRY should have H gates from ry expansion");
            assert!(rz_count > 0, "CRY should have RZ gates from ry expansion");
        }
        Err(e) => {
            panic!("Failed to parse cry gate: {e}");
        }
    }
}
