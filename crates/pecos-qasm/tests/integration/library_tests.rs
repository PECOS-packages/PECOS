use pecos_qasm::{Operation, QASMParser};

#[test]
fn test_hqslib1_basic_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[2];

        // Test HQS-specific gates
        U1q(pi/2, 0) q[0];
        Rz(pi) q[1];
        ZZ q[0], q[1];

        // Test basic gates
        x q[0];
        y q[1];
        z q[0];
        h q[1];

        // Test rotation gates
        rx(pi/2) q[0];
        ry(pi/3) q[1];
        rz(pi/4) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // Verify the gates were parsed
    let gate_ops: Vec<_> = program
        .operations
        .iter()
        .filter_map(|op| match op {
            Operation::Gate { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();

    // Check that all operations expanded to native gates
    assert!(gate_ops.contains(&"R1XY")); // U1q expands to R1XY
    assert!(gate_ops.contains(&"RZ")); // Rz expands to RZ
    assert!(gate_ops.contains(&"SZZ")); // ZZ expands to SZZ only
}

#[test]
fn test_hqslib1_cx_gate() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[2];

        // Test CNOT aliases
        cx q[0], q[1];
        CX q[0], q[1];
        CNOT q[0], q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // All should expand to native CX
    let gate_ops: Vec<_> = program
        .operations
        .iter()
        .filter_map(|op| match op {
            Operation::Gate { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(gate_ops.len(), 3);
    assert!(gate_ops.iter().all(|&gate| gate == "CX"));
}

#[test]
fn test_hqslib1_controlled_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[3];

        // Test controlled gates
        cy q[0], q[1];
        cz q[1], q[2];
        ccx q[0], q[1], q[2];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // These should all be present or expanded
    assert!(program.gate_definitions.contains_key("cy"));
    assert!(program.gate_definitions.contains_key("cz"));
    assert!(program.gate_definitions.contains_key("ccx"));
}

#[test]
fn test_hqslib1_phase_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[2];

        // Test phase gates
        s q[0];
        sdg q[0];
        t q[0];
        tdg q[0];
        p(pi/2) q[1];
        cp(pi/4) q[0], q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // Check these gates are available
    assert!(program.gate_definitions.contains_key("s"));
    assert!(program.gate_definitions.contains_key("sdg"));
    assert!(program.gate_definitions.contains_key("t"));
    assert!(program.gate_definitions.contains_key("tdg"));
    assert!(program.gate_definitions.contains_key("p"));
    assert!(program.gate_definitions.contains_key("cp"));
}

#[test]
fn test_hqslib1_universal_gate() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[1];

        // Test the general U gate
        U(pi/2, pi/4, pi/3) q[0];
        u(pi/2, pi/4, pi/3) q[0];  // lowercase alias
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // U gate should expand to RZ + R1XY + RZ
    let gate_ops: Vec<_> = program
        .operations
        .iter()
        .filter_map(|op| match op {
            Operation::Gate { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();

    // Should see RZ and R1XY from the U gate expansion
    assert!(gate_ops.contains(&"RZ"));
    assert!(gate_ops.contains(&"R1XY"));
}

#[test]
fn test_hqslib1_compatibility_uppercase() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[2];

        // Test uppercase aliases for compatibility
        H q[0];      // Native gate
        X q[0];      // Native gate
        Y q[0];      // Native gate
        Z q[0];      // Native gate
        S q[1];      // Alias for s
        Sdg q[1];    // Alias for sdg
        T q[1];      // Alias for t
        Tdg q[1];    // Alias for tdg
        RX(pi/2) q[0];   // Alias for rx
        RY(pi/3) q[1];   // Alias for ry
        RZ(pi/4) q[0];   // Native gate
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // All these should work without errors
    let gate_ops: Vec<_> = program
        .operations
        .iter()
        .filter_map(|op| match op {
            Operation::Gate { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();

    // Should have expanded to native gates
    assert!(gate_ops.contains(&"H".to_string()));
    assert!(gate_ops.contains(&"X".to_string()));
    assert!(gate_ops.contains(&"Y".to_string()));
    assert!(gate_ops.contains(&"Z".to_string()));
    assert!(gate_ops.contains(&"RZ".to_string())); // From S, Sdg, T, Tdg, RZ
    assert!(gate_ops.contains(&"R1XY".to_string())); // From RX, RY
}

#[test]
fn test_hqslib1_swap_and_sx() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[2];

        // Test swap and sqrt(X) gates
        swap q[0], q[1];
        sx q[0];
        sxdg q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // Verify these gates are available
    assert!(program.gate_definitions.contains_key("swap"));
    assert!(program.gate_definitions.contains_key("sx"));
    assert!(program.gate_definitions.contains_key("sxdg"));
}
