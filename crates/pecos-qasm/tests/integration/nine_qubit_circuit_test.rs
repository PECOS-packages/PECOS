use pecos_core::prelude::GateType;
use pecos_qasm::{Operation, parser::QASMParser};

// Helper function to check if an operation is a specific gate type
fn is_gate_type(op: &Operation, gate_name: &str) -> bool {
    match op {
        Operation::Gate { name, .. } => name.eq_ignore_ascii_case(gate_name),
        Operation::NativeGate(gate) => {
            let gate_type_str = format!("{:?}", gate.gate_type);
            gate_type_str.eq_ignore_ascii_case(gate_name)
                || (gate_name.eq_ignore_ascii_case("cx") && matches!(gate.gate_type, GateType::CX))
                || (gate_name.eq_ignore_ascii_case("cnot")
                    && matches!(gate.gate_type, GateType::CX))
                || (gate_name.eq_ignore_ascii_case("h") && matches!(gate.gate_type, GateType::H))
        }
        _ => false,
    }
}

// Helper function to get qubits from an operation
fn get_operation_qubits(op: &Operation) -> Vec<usize> {
    match op {
        Operation::Gate { qubits, .. } => qubits.clone(),
        Operation::NativeGate(gate) => gate.qubits.iter().map(|q| q.0).collect(),
        _ => vec![],
    }
}

#[test]
#[allow(clippy::too_many_lines)]
fn test_nine_qubit_quantum_circuit() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[9];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        cz q[7],q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        cz q[0],q[7];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        cz q[2],q[5];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        cz q[0],q[7];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        cz q[4],q[3];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[8];
        cz q[0],q[7];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        cz q[3],q[6];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        cz q[4],q[3];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[7];
        cz q[2],q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[5];
        cz q[6],q[8];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        cz q[4],q[3];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        cz q[3],q[6];
        cz q[7],q[4];
        rx(0.5*pi) q[5];
        cz q[0],q[7];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[3];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        cz q[6],q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[2];
        cz q[3],q[6];
        cz q[7],q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[5];
        cz q[6],q[8];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        cz q[4],q[3];
        rx(0.5*pi) q[5];
        cz q[6],q[8];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        cz q[7],q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        cz q[4],q[3];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        cz q[2],q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[3];
        cz q[7],q[4];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        cz q[7],q[4];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        cz q[7],q[4];
        cz q[6],q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        cz q[2],q[5];
        rx(0.5*pi) q[3];
        cz q[7],q[4];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        cz q[4],q[3];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        cz q[2],q[0];
        cz q[1],q[3];
        cz q[7],q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        cz q[2],q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        cz q[2],q[5];
        cz q[4],q[3];
        rx(0.5*pi) q[0];
        cz q[7],q[1];
        rx(0.5*pi) q[2];
        cz q[3],q[6];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[0];
        cz q[2],q[5];
        rx(0.5*pi) q[4];
        cz q[6],q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[2];
        cz q[3],q[6];
        cz q[7],q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[7],q[1];
        rx(0.5*pi) q[2];
        cz q[4],q[3];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[7],q[1];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[3];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        cz q[6],q[8];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        cz q[7],q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[8];
        cz q[2],q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        cz q[4],q[3];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        cz q[3],q[6];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[7];
        cz q[2],q[0];
        rx(0.5*pi) q[3];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        cz q[6],q[8];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        cz q[6],q[8];
        cz q[2],q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        cz q[6],q[8];
        cz q[2],q[0];
        rx(0.5*pi) q[1];
        cz q[3],q[6];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        cz q[3],q[6];
        cz q[7],q[4];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[3];
        rx(0.5*pi) q[4];
        cz q[6],q[8];
        rx(0.5*pi) q[2];
        cz q[3],q[6];
        cz q[7],q[4];
        rx(0.5*pi) q[8];
        cz q[7],q[1];
        cz q[2],q[5];
        cz q[4],q[3];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        cz q[6],q[8];
        rx(0.5*pi) q[7];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        cz q[7],q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[1];
        cz q[2],q[5];
        cz q[4],q[3];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        cz q[2],q[5];
        cz q[4],q[3];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        cz q[7],q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        cz q[7],q[1];
        cz q[4],q[3];
        rx(0.5*pi) q[5];
        cz q[6],q[8];
        cz q[1],q[3];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        cz q[0],q[7];
        cz q[1],q[3];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[0];
        cz q[4],q[3];
        cz q[6],q[8];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        cz q[4],q[3];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        cz q[3],q[6];
        cz q[7],q[4];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[1];
        cz q[4],q[3];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[1];
        cz q[3],q[6];
        cz q[7],q[4];
        rx(0.5*pi) q[1];
        cz q[4],q[3];
        cz q[6],q[8];
        rx(0.5*pi) q[7];
        cz q[0],q[7];
        cz q[3],q[6];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[4];
        cz q[6],q[8];
        rx(0.5*pi) q[7];
        cz q[2],q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[3];
        cz q[7],q[4];
        cz q[6],q[8];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[3];
        rx(0.5*pi) q[4];
        cz q[6],q[8];
        rx(0.5*pi) q[7];
        cz q[0],q[7];
        rx(0.5*pi) q[1];
        cz q[2],q[5];
        rx(0.5*pi) q[4];
        cz q[6],q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        cz q[3],q[6];
        cz q[7],q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        cz q[6],q[8];
        rx(0.5*pi) q[7];
        cz q[2],q[0];
        cz q[4],q[3];
        rx(0.5*pi) q[5];
        cz q[6],q[8];
        rx(0.5*pi) q[7];
        cz q[2],q[0];
        cz q[3],q[6];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[7];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        cz q[2],q[5];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        cz q[3],q[6];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[2];
        cz q[4],q[3];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        rx(0.5*pi) q[8];
        rx(0.5*pi) q[0];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
        cz q[7],q[4];
        rx(0.5*pi) q[5];
        rx(0.5*pi) q[6];
        cz q[1],q[3];
        cz q[7],q[4];
        rx(0.5*pi) q[6];
        cz q[0],q[7];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[4];
        rx(0.5*pi) q[6];
        cz q[2],q[0];
        cz q[7],q[4];
        cz q[0],q[7];
        cz q[2],q[5];
        cz q[4],q[3];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse nine-qubit circuit");

    // Count the types of gates after expansion
    let mut h_count = 0;
    let mut cx_count = 0; // CZ expands to H-CX-H
    let mut rxy_count = 0;
    let mut total_operations = 0;

    for op in &program.operations {
        total_operations += 1;
        if is_gate_type(op, "H") {
            h_count += 1;
        } else if is_gate_type(op, "CX") {
            cx_count += 1;
        } else if is_gate_type(op, "RXY1Q") {
            rxy_count += 1;
        }
    }

    // With gate expansions, we expect more operations
    assert!(
        total_operations > 500,
        "Should have more than 500 operations, got {total_operations}"
    );

    // Each CZ expands to 3 gates (H-CX-H)
    assert!(
        h_count > 160,
        "Should have more than 160 H gates, got {h_count}"
    );
    assert!(
        cx_count > 80,
        "Should have more than 80 CX gates, got {cx_count}"
    );

    assert!(rxy_count > 100, "Should have many RXY1Q gates from RX");

    // Check that all operations are on valid qubits
    for op in &program.operations {
        let qubits = get_operation_qubits(op);
        for qubit in qubits {
            assert!(qubit < 9, "Qubit index {qubit} is out of range");
        }
    }
}

#[test]
fn test_cz_gate_connectivity() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[9];
        cz q[1],q[3];
        cz q[7],q[4];
        cz q[0],q[7];
        cz q[2],q[5];
        cz q[4],q[3];
        cz q[3],q[6];
        cz q[6],q[8];
        cz q[7],q[1];
        cz q[2],q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse CZ connectivity");

    // CZ expands to H-CX-H, so we track CX gates to find the connectivity
    let mut cx_pairs = Vec::new();

    for op in &program.operations {
        if is_gate_type(op, "CX") {
            let qubits = get_operation_qubits(op);
            assert_eq!(qubits.len(), 2, "CX gate should have exactly 2 qubits");
            cx_pairs.push((qubits[0], qubits[1]));
        }
    }

    // We expect 9 CX gates (one for each CZ)
    assert_eq!(cx_pairs.len(), 9);

    // Check some specific connections
    assert!(cx_pairs.contains(&(1, 3)));
    assert!(cx_pairs.contains(&(7, 4)));
    assert!(cx_pairs.contains(&(0, 7)));
    assert!(cx_pairs.contains(&(2, 5)));
    assert!(cx_pairs.contains(&(4, 3)));
    assert!(cx_pairs.contains(&(3, 6)));
    assert!(cx_pairs.contains(&(6, 8)));
    assert!(cx_pairs.contains(&(7, 1)));
    assert!(cx_pairs.contains(&(2, 0)));
}

#[test]
fn test_rx_half_pi_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[3];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];
        rx(0.5*pi) q[2];
        rx(pi/2) q[0];
        rx(1.5707963267948966) q[1];  // numerical pi/2
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse RX gates");

    // RX lowers directly to RXY1Q(theta, 0).
    let mut total_ops = 0;
    let mut rxy_count = 0;

    for op in &program.operations {
        total_ops += 1;
        if is_gate_type(op, "RXY1Q") {
            rxy_count += 1;
        }
    }

    assert_eq!(total_ops, 5, "Should have one native operation per RX");
    assert_eq!(rxy_count, 5, "Should have 5 RXY1Q gates");
}

#[test]
fn test_circuit_patterns() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[4];
        // Pattern 1: CZ followed by RX on both qubits
        cz q[0],q[1];
        rx(0.5*pi) q[0];
        rx(0.5*pi) q[1];

        // Pattern 2: Multiple RX then CZ
        rx(0.5*pi) q[2];
        rx(0.5*pi) q[3];
        cz q[2],q[3];

        // Pattern 3: Interleaved CZ and RX
        cz q[0],q[2];
        rx(0.5*pi) q[1];
        cz q[1],q[3];
        rx(0.5*pi) q[2];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse circuit patterns");

    // After expansion, count the gate types
    let mut h_gates = 0;
    let mut cx_gates = 0;
    let mut rxy_gates = 0;

    for op in &program.operations {
        if is_gate_type(op, "H") {
            h_gates += 1;
        } else if is_gate_type(op, "CX") {
            cx_gates += 1;
        } else if is_gate_type(op, "RXY1Q") {
            rxy_gates += 1;
        }
    }

    // Corrected counts based on actual QASM:
    // We have 4 CZ gates (each expands to H-CX-H = 8H + 4CX)
    // We actually have 6 RX gates in the code (not 7):
    //   Pattern 1: rx q[0], rx q[1]
    //   Pattern 2: rx q[2], rx q[3]
    //   Pattern 3: rx q[1], rx q[2]
    // Each RX lowers to one RXY1Q gate.
    assert_eq!(cx_gates, 4, "Should have 4 CX gates from CZ expansions");
    assert_eq!(rxy_gates, 6, "Should have 6 RXY1Q gates from RX expansions");
    assert_eq!(h_gates, 8, "Should have 8 H gates from CZ expansions");
}
