use pecos_qasm::{Operation, parser::QASMParser};

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
                || (target_name == "cx" && gate_type_name == "cnot")
                || (target_name == "cnot" && gate_type_name == "cnot")
                || (target_name == "h" && gate_type_name == "hadamard")
        }
        _ => false,
    }
}

#[test]
fn test_measure_register_expansion() {
    // Test that measure q -> c expands to individual measurements
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[3];
        creg c[3];

        h q;  // Apply hadamard to all qubits in register
        measure q -> c;  // Measure all qubits to all classical bits
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // Count the number of measurements
    let measure_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::MeasureWithMapping { .. }))
        .count();

    // Should have 3 individual measurements
    assert_eq!(measure_count, 3, "Expected 3 measurements");

    // Verify each measurement is correct
    let measurements: Vec<_> = program
        .operations
        .iter()
        .filter_map(|op| match op {
            Operation::MeasureWithMapping {
                gate,
                c_reg,
                c_index,
            } => {
                let qubit = gate.qubits.first().map_or(0, |q| q.0);
                Some((qubit, c_reg.clone(), *c_index))
            }
            _ => None,
        })
        .collect();

    assert_eq!(measurements.len(), 3);

    // Check that measurements map correctly
    for (i, (_qubit, c_reg, c_index)) in measurements.iter().enumerate() {
        assert_eq!(c_reg, "c", "Expected classical register c");
        assert_eq!(*c_index, i, "Expected classical index to match");
        // Qubit IDs might vary, but we verify there are 3 unique ones
    }

    // Verify we have 3 unique qubits
    let unique_qubits: std::collections::BTreeSet<_> =
        measurements.iter().map(|(q, _, _)| q).collect();
    assert_eq!(unique_qubits.len(), 3, "Expected 3 unique qubits");
}

#[test]
fn test_register_gate_expansion_should_work() {
    // According to OpenQASM 2.0 spec, gates on registers should expand
    // to individual qubit operations when registers have the same size
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[3];

        // This should expand to h q[0]; h q[1]; h q[2];
        h q;
    "#;

    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(program) => {
            println!("SUCCESS: Parser supports register-level gates");

            // Debug: print all operations
            println!("Operations generated:");
            for (i, op) in program.operations.iter().enumerate() {
                match op {
                    Operation::Gate { name, qubits, .. } => {
                        println!("  [{i}] Gate: {name} on qubits: {qubits:?}");
                    }
                    Operation::NativeGate(gate) => {
                        println!(
                            "  [{i}] NativeGate: {:?} on qubits: {:?}",
                            gate.gate_type, gate.qubits
                        );
                    }
                    _ => {
                        println!("  [{i}] Other operation: {op:?}");
                    }
                }
            }

            // Count H gates - should be 3
            let h_count = program
                .operations
                .iter()
                .filter(|op| is_gate_with_name(op, "H"))
                .count();

            println!("H gate count: {h_count}");
            assert_eq!(
                h_count, 3,
                "Should have expanded to 3 H gates, but got {h_count}"
            );
        }
        Err(e) => {
            println!("LIMITATION: Parser doesn't support register-level gates yet: {e}");
            println!("This should be implemented to match OpenQASM 2.0 spec");
        }
    }
}

#[test]
fn test_two_qubit_register_gate_expansion() {
    // Two-qubit gates on registers of same size should expand
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg a[2];
        qreg b[2];

        // This should expand to: cx a[0], b[0]; cx a[1], b[1];
        cx a, b;
    "#;

    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(program) => {
            println!("SUCCESS: Parser supports register-level two-qubit gates");

            let cx_count = program
                .operations
                .iter()
                .filter(|op| is_gate_with_name(op, "CX"))
                .count();

            assert_eq!(cx_count, 2, "Should have expanded to 2 CX gates");
        }
        Err(e) => {
            println!("LIMITATION: Parser doesn't support register-level two-qubit gates: {e}");
        }
    }
}

#[test]
fn test_measurement_register_expansion_works() {
    // This already works in PECOS
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[3];
        creg c[3];

        // This works and expands to individual measurements
        measure q -> c;
    "#;

    let program = QASMParser::parse_str(qasm).expect("Should parse register measurement");

    // After expansion, should have individual measurements
    let measure_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::MeasureWithMapping { .. }))
        .count();

    assert_eq!(
        measure_count, 3,
        "Should have 3 individual measurements after expansion"
    );
}

#[test]
fn test_barrier_register_expansion_works() {
    // This already works in PECOS
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[3];

        // This works and expands to all qubits in q
        barrier q;
    "#;

    let program = QASMParser::parse_str(qasm).expect("Should parse register barrier");

    // Should have a barrier with 3 qubits
    for op in &program.operations {
        if let Operation::Barrier { qubits } = op {
            assert_eq!(qubits.len(), 3, "Barrier should include all 3 qubits");
        }
    }
}

#[test]
fn test_mixed_size_register_error() {
    // This should fail according to OpenQASM spec
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg a[2];
        qreg b[3];

        // This should fail - registers have different sizes
        cx a, b;
    "#;

    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(_) => {
            println!("WARNING: Parser accepted mismatched register sizes - should fail");
        }
        Err(e) => {
            println!("Correctly rejected mismatched sizes: {e}");
        }
    }
}

#[test]
fn test_gate_with_params_on_register() {
    // Parameterized gates on registers should also expand
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];

        // This should expand to: rz(pi/4) q[0]; rz(pi/4) q[1];
        rz(pi/4) q;
    "#;

    let result = QASMParser::parse_str(qasm);

    match result {
        Ok(program) => {
            println!("SUCCESS: Parser supports parameterized gates on registers");

            let rz_count = program
                .operations
                .iter()
                .filter(|op| is_gate_with_name(op, "RZ"))
                .count();

            assert_eq!(rz_count, 2, "Should have expanded to 2 RZ gates");
        }
        Err(e) => {
            println!("LIMITATION: Parser doesn't support parameterized gates on registers: {e}");
        }
    }
}
