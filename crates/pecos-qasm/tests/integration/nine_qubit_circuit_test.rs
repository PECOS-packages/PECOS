use pecos_qasm::{Operation, parser::QASMParser};

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
    let mut total_operations = 0;

    for op in &program.operations {
        total_operations += 1;
        if let Operation::Gate { name, .. } = op {
            match name.as_str() {
                "H" => h_count += 1,
                "CX" => cx_count += 1,
                _ => {}
            }
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

    // RX gates may also be expanded
    assert!(
        total_operations - h_count - cx_count > 100,
        "Should have many other operations"
    );

    // Check that all operations are on valid qubits
    for op in &program.operations {
        if let Operation::Gate { qubits, .. } = op {
            for &qubit in qubits {
                assert!(qubit < 9, "Qubit index {qubit} is out of range");
            }
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
        if let Operation::Gate { name, qubits, .. } = op {
            if name == "CX" {
                assert_eq!(qubits.len(), 2, "CX gate should have exactly 2 qubits");
                cx_pairs.push((qubits[0], qubits[1]));
            }
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

    // RX expands to H-RZ-H, so we look for the pattern
    let mut total_ops = 0;
    let mut h_count = 0;
    let mut rz_count = 0;

    for op in &program.operations {
        total_ops += 1;
        if let Operation::Gate { name, .. } = op {
            match name.as_str() {
                "H" => h_count += 1,
                "RZ" => rz_count += 1,
                _ => {}
            }
        }
    }

    // Each RX expands to 3 gates (H-RZ-H)
    // We have 5 RX gates, so expect 15 total operations
    assert_eq!(total_ops, 15, "Should have 15 operations after expansion");
    assert_eq!(h_count, 10, "Should have 10 H gates (2 per RX)");
    assert_eq!(rz_count, 5, "Should have 5 RZ gates (1 per RX)");
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
    let mut rz_gates = 0;

    for op in &program.operations {
        if let Operation::Gate { name, .. } = op {
            match name.as_str() {
                "H" => h_gates += 1,
                "CX" => cx_gates += 1,
                "RZ" => rz_gates += 1,
                _ => {}
            }
        }
    }

    // Corrected counts based on actual QASM:
    // We have 4 CZ gates (each expands to H-CX-H = 8H + 4CX)
    // We actually have 6 RX gates in the code (not 7):
    //   Pattern 1: rx q[0], rx q[1]
    //   Pattern 2: rx q[2], rx q[3]
    //   Pattern 3: rx q[1], rx q[2]
    // Each RX expands to H-RZ-H = 12H + 6RZ
    assert_eq!(cx_gates, 4, "Should have 4 CX gates from CZ expansions");
    assert_eq!(rz_gates, 6, "Should have 6 RZ gates from RX expansions");
    assert_eq!(h_gates, 20, "Should have 20 H gates total");
}
