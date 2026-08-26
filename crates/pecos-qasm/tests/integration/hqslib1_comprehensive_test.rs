use pecos_qasm::{Operation, QASMParser};

// Helper function to extract gate names from operations
fn get_gate_names(operations: &[Operation]) -> Vec<String> {
    operations
        .iter()
        .filter_map(|op| match op {
            Operation::Gate { name, .. } => Some(name.clone()),
            Operation::NativeGate(gate) => Some(format!("{:?}", gate.gate_type)),
            _ => None,
        })
        .collect()
}

// Helper function to count specific gate occurrences
fn count_gate(operations: &[Operation], gate_name: &str) -> usize {
    operations
        .iter()
        .filter(|op| match op {
            Operation::Gate { name, .. } => name.eq_ignore_ascii_case(gate_name),
            Operation::NativeGate(gate) => {
                let gate_type_str = format!("{:?}", gate.gate_type);
                gate_type_str.eq_ignore_ascii_case(gate_name)
            }
            _ => false,
        })
        .count()
}

#[test]
fn test_hqslib1_all_basic_single_qubit_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[1];

        // Test all basic single-qubit gates
        x q[0];
        y q[0];
        z q[0];
        h q[0];
        id q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse basic single-qubit gates");

    // Verify all gates are parsed
    assert_eq!(count_gate(&program.operations, "X"), 1);
    assert_eq!(count_gate(&program.operations, "Y"), 1);
    assert_eq!(count_gate(&program.operations, "Z"), 1);
    assert_eq!(count_gate(&program.operations, "H"), 1);
    // id gate should expand to RZ(0)
    assert!(count_gate(&program.operations, "RZ") >= 1);
}

#[test]
fn test_hqslib1_phase_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[1];

        // Test all phase gates
        s q[0];
        sdg q[0];
        t q[0];
        tdg q[0];
        p(pi/3) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse phase gates");

    // Named phase gates stay named; only the arbitrary phase gate is an RZ.
    assert_eq!(count_gate(&program.operations, "SZ"), 1);
    assert_eq!(count_gate(&program.operations, "SZdg"), 1);
    assert_eq!(count_gate(&program.operations, "T"), 1);
    assert_eq!(count_gate(&program.operations, "Tdg"), 1);
    assert_eq!(count_gate(&program.operations, "RZ"), 1);
}

#[test]
fn test_hqslib1_rotation_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[1];

        // Test rotation gates
        rx(pi/2) q[0];
        ry(pi/3) q[0];
        rz(pi/4) q[0];

        // Test uppercase aliases
        RX(pi/2) q[0];
        RY(pi/3) q[0];
        Rz(pi/4) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse rotation gates");

    // rx and ry expand to R1XY, rz is native
    assert!(count_gate(&program.operations, "R1XY") >= 4);
    assert!(count_gate(&program.operations, "RZ") >= 2);
}

#[test]
fn test_hqslib1_universal_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[1];

        // Test universal single-qubit gates
        U(pi/2, pi/4, pi/3) q[0];
        u(pi/2, pi/4, pi/3) q[0];
        U1q(pi/2, pi/4) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse universal gates");

    // U gates decompose to RZ + R1XY + RZ
    // U1q is directly R1XY
    let gate_names = get_gate_names(&program.operations);
    assert!(gate_names.contains(&"RZ".to_string()));
    assert!(gate_names.contains(&"R1XY".to_string()));
}

#[test]
fn test_hqslib1_two_qubit_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[2];

        // Test two-qubit gates
        cx q[0],q[1];
        cy q[0],q[1];
        cz q[0],q[1];
        swap q[0],q[1];

        // Test HQS-specific gates
        ZZ q[0],q[1];

        // Test uppercase aliases
        CNOT q[0],q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse two-qubit gates");

    // Verify gates are present
    assert!(count_gate(&program.operations, "CX") >= 2); // cx and CNOT
    assert!(program.gate_definitions.contains_key("cy"));
    assert!(program.gate_definitions.contains_key("cz"));
    assert!(program.gate_definitions.contains_key("swap"));
    assert!(count_gate(&program.operations, "SZZ") >= 1); // ZZ maps to SZZ
}

#[test]
fn test_hqslib1_controlled_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[2];

        // Test controlled phase gate
        cp(pi/4) q[0],q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse controlled gates");

    // cp gate should be defined
    assert!(program.gate_definitions.contains_key("cp"));
}

#[test]
fn test_hqslib1_three_qubit_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[3];

        // Test Toffoli gate
        ccx q[0],q[1],q[2];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse three-qubit gates");

    // ccx (Toffoli) should be defined
    assert!(program.gate_definitions.contains_key("ccx"));
}

#[test]
fn test_hqslib1_sqrt_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[1];

        // Test sqrt gates
        sx q[0];
        sxdg q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse sqrt gates");

    assert_eq!(count_gate(&program.operations, "SX"), 1);
    assert_eq!(count_gate(&program.operations, "SXdg"), 1);
}

#[test]
fn test_hqslib1_uppercase_compatibility_aliases() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[2];

        // Test uppercase aliases for compatibility
        S q[0];
        Sdg q[0];
        T q[0];
        Tdg q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse uppercase aliases");

    assert_eq!(count_gate(&program.operations, "SZ"), 1);
    assert_eq!(count_gate(&program.operations, "SZdg"), 1);
    assert_eq!(count_gate(&program.operations, "T"), 1);
    assert_eq!(count_gate(&program.operations, "Tdg"), 1);
}

#[test]
fn test_hqslib1_complex_circuit() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[4];
        creg c[4];

        // Initialize with Hadamards
        h q[0];
        h q[1];

        // Create entanglement with HQS-specific gates
        ZZ q[0],q[1];
        U1q(pi/2, 0) q[2];

        // Apply some rotations
        Rz(pi/4) q[0];
        rx(pi/3) q[1];
        ry(pi/6) q[2];

        // More entanglement
        cx q[1],q[2];
        cy q[2],q[3];
        cz q[0],q[3];

        // Three-qubit gate
        ccx q[0],q[1],q[2];

        // Phase gates
        s q[0];
        t q[1];
        p(pi/8) q[2];

        // Swap
        swap q[2],q[3];

        // Measure all
        measure q -> c;
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse complex circuit");

    // Verify measurements
    let measure_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::MeasureWithMapping { .. }))
        .count();
    assert_eq!(measure_count, 4);

    // Verify various gates are present
    assert!(count_gate(&program.operations, "H") >= 2);
    assert!(count_gate(&program.operations, "SZZ") >= 1);
    assert!(count_gate(&program.operations, "R1XY") >= 1);
    assert!(count_gate(&program.operations, "CX") >= 1);
}

#[test]
fn test_hqslib1_gate_parameters() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[2];

        // Test various parameter expressions
        rx(pi) q[0];
        ry(2*pi) q[0];
        rz(-pi/2) q[0];
        U(pi/2, -pi/4, 3*pi/4) q[0];
        U1q(0.5*pi, pi/6) q[0];
        p(-2*pi) q[0];
        cp(pi/8) q[0],q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse gates with parameters");

    // Just verify parsing succeeds with various parameter expressions
    assert!(!program.operations.is_empty());
}

#[test]
fn test_hqslib1_no_rzz_gate_definition() {
    // This test verifies that RZZ works as a native gate, not through hqslib1.inc
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[2];

        // RZZ should work as a native gate
        RZZ(pi/4) q[0],q[1];
        RZZ(-pi/2) q[0],q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse RZZ gates");

    // RZZ should NOT be in gate_definitions (it's native)
    assert!(!program.gate_definitions.contains_key("RZZ"));
    assert!(!program.gate_definitions.contains_key("rzz"));

    // But it should be in operations as native gates
    assert_eq!(count_gate(&program.operations, "RZZ"), 2);
}
