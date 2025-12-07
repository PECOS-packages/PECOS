use pecos_core::prelude::GateType;
use pecos_engines::ClassicalControlEngineBuilder;
use pecos_qasm::{Operation, parser::QASMParser};

// Helper function to check if an operation is a specific gate
fn is_gate_with_name(op: &Operation, gate_name: &str) -> bool {
    match op {
        Operation::Gate { name, .. } => name.eq_ignore_ascii_case(gate_name),
        Operation::NativeGate(gate) => {
            let gate_type_str = format!("{:?}", gate.gate_type);
            gate_type_str.eq_ignore_ascii_case(gate_name)
                || (gate_name.eq_ignore_ascii_case("cx") && matches!(gate.gate_type, GateType::CX))
                || (gate_name.eq_ignore_ascii_case("cnot")
                    && matches!(gate.gate_type, GateType::CX))
                || (gate_name.eq_ignore_ascii_case("h") && matches!(gate.gate_type, GateType::H))
                || (gate_name.eq_ignore_ascii_case("x") && matches!(gate.gate_type, GateType::X))
        }
        _ => false,
    }
}

use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;

#[test]
fn test_x_gate_and_measure() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[12];
        creg c[12];

        x q[10];
        measure q[10] -> c[10];
    "#;

    // First test parsing
    let program = QASMParser::parse_str(qasm).expect("Failed to parse X gate and measure");

    // Count operations
    let mut operation_types = Vec::new();

    for op in &program.operations {
        match op {
            Operation::Gate { name, qubits, .. } => {
                operation_types.push(("gate", name.clone(), qubits.clone()));
            }
            Operation::NativeGate(gate) => {
                let gate_name = format!("{:?}", gate.gate_type);
                let qubits = gate.qubits.iter().map(|q| q.0).collect();
                operation_types.push(("gate", gate_name, qubits));
            }
            Operation::MeasureWithMapping {
                gate,
                c_reg,
                c_index,
            } => {
                let qubit = gate.qubits.first().map_or(0, |q| q.0);
                operation_types.push(("measure", format!("{c_reg}[{c_index}]"), vec![qubit]));
            }
            _ => {}
        }
    }

    // We should have at least 2 operations (X gate might be expanded)
    assert!(
        operation_types.len() >= 2,
        "Should have at least 2 operations"
    );

    // Check for X gate (or its expansion)
    let has_x = program
        .operations
        .iter()
        .any(|op| is_gate_with_name(op, "X"));
    assert!(has_x, "Should have X gate");

    // Check for measurement
    let has_measure = operation_types
        .iter()
        .any(|(op_type, _, _)| op_type == &"measure");
    assert!(has_measure, "Should have measure operation");

    // Verify the measurement is from q[10] to c[10]
    for (op_type, target, qubits) in &operation_types {
        if op_type == &"measure" {
            assert_eq!(qubits, &vec![10], "Measurement should be on qubit 10");
            assert_eq!(
                target, "c[10]",
                "Measurement should be to classical bit c[10]"
            );
        }
    }

    // Now test actual simulation - X gate should flip the qubit from |0⟩ to |1⟩
    let shot_vec = qasm_engine()
        .program(Qasm::from_string(qasm))
        .to_sim()
        .seed(42)
        .workers(1)
        .run(100)
        .expect("Failed to run simulation");

    // Verify that qubit 10 is always measured as 1 (since X flips it)
    assert_eq!(shot_vec.len(), 100, "Should have 100 shots");

    for shot in &shot_vec.shots {
        let value = shot
            .data
            .get("c")
            .and_then(pecos_engines::prelude::Data::as_u32)
            .expect("c register should be convertible to u32");
        // Extract bit 10 from the result
        let bit_10 = (value >> 10) & 1u32;
        assert_eq!(bit_10, 1u32, "Bit 10 should always be 1 after X gate");
    }
}

#[test]
fn test_multiple_measurements() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        creg c[4];

        h q[0];
        x q[1];
        y q[2];
        z q[3];

        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];
        measure q[3] -> c[3];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse multiple measurements");

    // Count measurements
    let mut measurements = Vec::new();

    for op in &program.operations {
        if let Operation::MeasureWithMapping {
            gate,
            c_reg,
            c_index,
        } = op
        {
            let qubit = gate.qubits.first().map_or(0, |q| q.0);
            measurements.push((qubit, c_reg.clone(), *c_index));
        }
    }

    assert_eq!(measurements.len(), 4, "Should have 4 measurements");

    // Check each measurement
    assert!(measurements.contains(&(0, "c".to_string(), 0)));
    assert!(measurements.contains(&(1, "c".to_string(), 1)));
    assert!(measurements.contains(&(2, "c".to_string(), 2)));
    assert!(measurements.contains(&(3, "c".to_string(), 3)));
}

#[test]
fn test_measure_syntax_variations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        creg d[2];

        // Standard measurement
        measure q[0] -> c[0];

        // Measurement to different register
        measure q[1] -> d[0];

        // Measurement with different indices
        measure q[2] -> c[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse measure syntax variations");

    let mut measurements = Vec::new();

    for op in &program.operations {
        if let Operation::MeasureWithMapping {
            gate,
            c_reg,
            c_index,
        } = op
        {
            let qubit = gate.qubits.first().map_or(0, |q| q.0);
            measurements.push((qubit, c_reg.clone(), *c_index));
        }
    }

    assert_eq!(measurements.len(), 3, "Should have 3 measurements");

    // Verify each measurement
    assert!(
        measurements
            .iter()
            .any(|(q, reg, idx)| *q == 0 && reg == "c" && *idx == 0)
    );
    assert!(
        measurements
            .iter()
            .any(|(q, reg, idx)| *q == 1 && reg == "d" && *idx == 0)
    );
    assert!(
        measurements
            .iter()
            .any(|(q, reg, idx)| *q == 2 && reg == "c" && *idx == 1)
    );
}

#[test]
fn test_measure_after_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];

        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    let program =
        QASMParser::parse_str(qasm).expect("Failed to parse gates followed by measurements");

    // Track the order of operations
    let mut operation_sequence = Vec::new();

    for op in &program.operations {
        match op {
            Operation::Gate { name, .. } => {
                operation_sequence.push(format!("gate:{name}"));
            }
            Operation::NativeGate(gate) => {
                let gate_name = format!("{:?}", gate.gate_type);
                operation_sequence.push(format!("gate:{gate_name}"));
            }
            Operation::MeasureWithMapping { gate, .. } => {
                let qubit = gate.qubits.first().map_or(0, |q| q.0);
                operation_sequence.push(format!("measure:q[{qubit}]"));
            }
            _ => {}
        }
    }

    // Verify that measurements come after gates
    let measure_indices: Vec<_> = operation_sequence
        .iter()
        .enumerate()
        .filter(|(_, op)| op.starts_with("measure:"))
        .map(|(i, _)| i)
        .collect();

    assert_eq!(measure_indices.len(), 2, "Should have 2 measurements");

    // Both measurements should be at the end
    assert!(
        measure_indices[0] > 0,
        "First measurement should not be at the beginning"
    );
}
