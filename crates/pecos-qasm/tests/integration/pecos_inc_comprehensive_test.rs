use pecos_qasm::{Operation, QASMParser};

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
fn test_pecos_inc_all_single_qubit_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "pecos.inc";

        qreg q[1];

        // Test all single-qubit gates in pecos.inc
        h q[0];
        x q[0];
        y q[0];
        z q[0];
    "#;

    let program =
        QASMParser::parse_str(qasm).expect("Failed to parse pecos.inc single-qubit gates");

    // All gates should map directly to native gates
    assert_eq!(count_gate(&program.operations, "H"), 1);
    assert_eq!(count_gate(&program.operations, "X"), 1);
    assert_eq!(count_gate(&program.operations, "Y"), 1);
    assert_eq!(count_gate(&program.operations, "Z"), 1);
}

#[test]
fn test_pecos_inc_rotation_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "pecos.inc";

        qreg q[1];

        // Test rotation gates
        rz(pi/4) q[0];
        rz(-pi/2) q[0];
        rz(2*pi) q[0];

        rxy1q(pi/2, 0) q[0];
        rxy1q(pi/3, pi/2) q[0];
        rxy1q(pi, pi/4) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse pecos.inc rotation gates");

    // All should map to native gates
    assert_eq!(count_gate(&program.operations, "RZ"), 3);
    assert_eq!(count_gate(&program.operations, "RXY1Q"), 3);
}

#[test]
fn test_pecos_inc_two_qubit_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "pecos.inc";

        qreg q[2];

        // Test two-qubit gates
        cx q[0],q[1];
        szz q[0],q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse pecos.inc two-qubit gates");

    // Both should map to native gates
    assert_eq!(count_gate(&program.operations, "CX"), 1);
    assert_eq!(count_gate(&program.operations, "SZZ"), 1);
}

#[test]
fn test_pecos_inc_native_gates_uppercase() {
    // Test that native gates work with uppercase directly
    let qasm = r#"
        OPENQASM 2.0;
        include "pecos.inc";

        qreg q[2];

        // Native gates should work with uppercase
        H q[0];
        X q[0];
        Y q[0];
        Z q[0];
        RZ(pi/2) q[0];
        RXY1Q(pi/2, pi/4) q[0];
        CX q[0],q[1];
        SZZ q[0],q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse uppercase native gates");

    // All should work as native gates
    assert_eq!(count_gate(&program.operations, "H"), 1);
    assert_eq!(count_gate(&program.operations, "X"), 1);
    assert_eq!(count_gate(&program.operations, "Y"), 1);
    assert_eq!(count_gate(&program.operations, "Z"), 1);
    assert_eq!(count_gate(&program.operations, "RZ"), 1);
    assert_eq!(count_gate(&program.operations, "RXY1Q"), 1);
    assert_eq!(count_gate(&program.operations, "CX"), 1);
    assert_eq!(count_gate(&program.operations, "SZZ"), 1);
}

#[test]
fn test_pecos_inc_with_measurements() {
    let qasm = r#"
        OPENQASM 2.0;
        include "pecos.inc";

        qreg q[3];
        creg c[3];

        // Apply some gates
        h q[0];
        cx q[0],q[1];
        szz q[1],q[2];
        rz(pi/4) q[2];

        // Measure all qubits
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse pecos.inc with measurements");

    // Count measurements
    let measure_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::MeasureWithMapping { .. }))
        .count();
    assert_eq!(measure_count, 3);

    // Verify gates
    assert_eq!(count_gate(&program.operations, "H"), 1);
    assert_eq!(count_gate(&program.operations, "CX"), 1);
    assert_eq!(count_gate(&program.operations, "SZZ"), 1);
    assert_eq!(count_gate(&program.operations, "RZ"), 1);
}

#[test]
fn test_pecos_inc_parameter_expressions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "pecos.inc";

        qreg q[1];

        // Test various parameter expressions
        rz(0) q[0];
        rz(pi) q[0];
        rz(2*pi) q[0];
        rz(-pi/2) q[0];
        rz(3*pi/4) q[0];

        rxy1q(pi/2, 0) q[0];
        rxy1q(pi, pi/2) q[0];
        rxy1q(2*pi, -pi/4) q[0];
        rxy1q(0.5*pi, 0.25*pi) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse parameter expressions");

    assert_eq!(count_gate(&program.operations, "RZ"), 5);
    assert_eq!(count_gate(&program.operations, "RXY1Q"), 4);
}

#[test]
fn test_pecos_inc_complex_circuit() {
    let qasm = r#"
        OPENQASM 2.0;
        include "pecos.inc";

        qreg q[5];
        creg c[5];

        // Initialize with Hadamards
        h q[0];
        h q[1];
        h q[2];

        // Create GHZ-like state
        cx q[0],q[1];
        cx q[1],q[2];
        cx q[2],q[3];
        cx q[3],q[4];

        // Apply rotations
        rz(pi/4) q[0];
        rxy1q(pi/2, 0) q[1];
        rz(-pi/2) q[2];
        rxy1q(pi/3, pi/2) q[3];

        // ZZ interactions
        szz q[0],q[1];
        szz q[2],q[3];
        szz q[3],q[4];

        // Apply Pauli gates
        x q[1];
        y q[2];
        z q[3];

        // More rotations
        rz(pi/8) q[0];
        rxy1q(pi/6, pi/4) q[4];

        // Final entanglement
        cx q[0],q[4];

        // Measure all
        measure q -> c;
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse complex circuit");

    // Verify gate counts
    assert_eq!(count_gate(&program.operations, "H"), 3);
    assert_eq!(count_gate(&program.operations, "CX"), 5);
    assert_eq!(count_gate(&program.operations, "RZ"), 3);
    assert_eq!(count_gate(&program.operations, "RXY1Q"), 3);
    assert_eq!(count_gate(&program.operations, "SZZ"), 3);
    assert_eq!(count_gate(&program.operations, "X"), 1);
    assert_eq!(count_gate(&program.operations, "Y"), 1);
    assert_eq!(count_gate(&program.operations, "Z"), 1);

    // Verify measurements
    let measure_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::MeasureWithMapping { .. }))
        .count();
    assert_eq!(measure_count, 5);
}

#[test]
fn test_pecos_inc_minimal_nature() {
    // Test that pecos.inc only provides minimal gates
    let qasm = r#"
        OPENQASM 2.0;
        include "pecos.inc";

        qreg q[2];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse pecos.inc");

    // Check that only the minimal set of gates is defined
    assert!(program.gate_definitions.contains_key("h"));
    assert!(program.gate_definitions.contains_key("x"));
    assert!(program.gate_definitions.contains_key("y"));
    assert!(program.gate_definitions.contains_key("z"));
    assert!(program.gate_definitions.contains_key("rz"));
    assert!(program.gate_definitions.contains_key("rxy1q"));
    assert!(program.gate_definitions.contains_key("cx"));
    assert!(program.gate_definitions.contains_key("szz"));

    // Check that common gates from qelib1 are NOT defined
    assert!(!program.gate_definitions.contains_key("rx"));
    assert!(!program.gate_definitions.contains_key("ry"));
    assert!(!program.gate_definitions.contains_key("u3"));
    assert!(!program.gate_definitions.contains_key("cz"));
    assert!(!program.gate_definitions.contains_key("ccx"));
    assert!(!program.gate_definitions.contains_key("swap"));
}

#[test]
fn test_pecos_inc_with_barriers() {
    let qasm = r#"
        OPENQASM 2.0;
        include "pecos.inc";

        qreg q[3];

        h q[0];
        cx q[0],q[1];

        barrier q;

        rz(pi/4) q[0];
        szz q[1],q[2];

        barrier q[0],q[1];

        x q[2];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse with barriers");

    // Count barriers
    let barrier_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::Barrier { .. }))
        .count();
    assert_eq!(barrier_count, 2);

    // Verify gates still work
    assert_eq!(count_gate(&program.operations, "H"), 1);
    assert_eq!(count_gate(&program.operations, "CX"), 1);
    assert_eq!(count_gate(&program.operations, "RZ"), 1);
    assert_eq!(count_gate(&program.operations, "SZZ"), 1);
    assert_eq!(count_gate(&program.operations, "X"), 1);
}

#[test]
fn test_pecos_inc_native_gate_compatibility() {
    // Test that gates in pecos.inc match native PECOS gates exactly
    let qasm = r#"
        OPENQASM 2.0;
        include "pecos.inc";

        qreg q[2];

        // These should all resolve to native gates with no expansion
        h q[0];      // -> H
        x q[0];      // -> X
        y q[0];      // -> Y
        z q[0];      // -> Z
        rz(pi) q[0]; // -> RZ
        rxy1q(pi/2, 0) q[0]; // -> RXY1Q
        cx q[0],q[1];  // -> CX
        szz q[0],q[1]; // -> SZZ
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse native compatibility test");

    // All operations should be native gates or simple gate calls that map to native
    for op in &program.operations {
        if let Operation::Gate { name, .. } = op {
            // Gate names should match what's defined in pecos.inc
            assert!(
                ["h", "x", "y", "z", "rz", "rxy1q", "cx", "szz"]
                    .contains(&name.to_lowercase().as_str()),
                "Unexpected gate: {name}"
            );
        }
        // NativeGate and other operations are expected and don't need checking
    }
}
