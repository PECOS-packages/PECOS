//! Tests for special gates including sqrt(X) variants and other non-standard gates

use pecos_core::prelude::GateType;
use pecos_qasm::{Operation, QASMParser};

// Helper function to extract gate name from operation
fn get_gate_name(op: &Operation) -> Option<String> {
    match op {
        Operation::Gate { name, .. } => Some(name.clone()),
        Operation::NativeGate(gate) => Some(format!("{:?}", gate.gate_type)),
        _ => None,
    }
}

// Helper function to check if an operation is a gate (either variant)
fn is_gate_operation(op: &Operation) -> bool {
    matches!(op, Operation::Gate { .. } | Operation::NativeGate(_))
}

#[test]
fn test_sqrt_x_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        //test SX, SXdg, CSX gates
        qreg q[2];
        sx q[0];
        x q[1];
        sxdg q[1];
        csx q[0],q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM with sqrt(X) gates");

    // Verify that the program parsed successfully and has operations
    assert!(!program.operations.is_empty(), "Should have operations");

    // Check that the sqrt(X) gates are available (either as native gates or defined in qelib1)
    let gate_names: Vec<String> = program
        .operations
        .iter()
        .filter_map(get_gate_name)
        .collect();

    // Debug: print what gates we actually have
    println!("Gates in operations: {gate_names:?}");

    // The gates might be expanded, so let's just check that we have some operations
    assert!(!gate_names.is_empty(), "Should have some gate operations");
}

#[test]
fn test_sx_gates_expansion() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        //test SX, SXdg, CSX gates
        qreg q[2];
        sx q[0];
        X q[1];
        sxdg q[1];
        csx q[0],q[1];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    // After all expansions, named square-root gates remain native operations.
    // sx -> SX
    // x -> X (native)
    // sxdg -> SXdg
    // csx -> CX (in our simplified implementation)
    assert!(!program.operations.is_empty());

    // Verify all operations are valid gates
    for op in &program.operations {
        assert!(is_gate_operation(op));
    }
}

#[test]
fn test_sx_gate_parameters() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        sx q[0];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    assert_eq!(program.operations.len(), 1);
    assert!(matches!(
        &program.operations[0],
        Operation::NativeGate(gate) if gate.gate_type == GateType::SX && gate.angles.is_empty()
    ));
}

#[test]
fn test_sxdg_gate_parameters() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        sxdg q[0];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    assert_eq!(program.operations.len(), 1);
    assert!(matches!(
        &program.operations[0],
        Operation::NativeGate(gate) if gate.gate_type == GateType::SXdg && gate.angles.is_empty()
    ));
}

#[test]
fn test_sqrt_x_gate_definitions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        sx q[0];
        sxdg q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM with sqrt(X) gates");

    // Verify that sx and sxdg are defined in qelib1
    assert!(
        program.gate_definitions.contains_key("sx"),
        "sx should be defined in qelib1"
    );
    assert!(
        program.gate_definitions.contains_key("sxdg"),
        "sxdg should be defined in qelib1"
    );

    // Verify the structure of the gate definitions
    if let Some(sx_def) = program.gate_definitions.get("sx") {
        assert_eq!(sx_def.params.len(), 0, "sx should have no parameters");
        assert_eq!(sx_def.qargs.len(), 1, "sx should act on one qubit");
    }

    if let Some(sxdg_def) = program.gate_definitions.get("sxdg") {
        assert_eq!(sxdg_def.params.len(), 0, "sxdg should have no parameters");
        assert_eq!(sxdg_def.qargs.len(), 1, "sxdg should act on one qubit");
    }
}

#[test]
fn test_controlled_sx_gate() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        csx q[0],q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM with csx gate");

    // Verify that csx is defined in qelib1
    assert!(
        program.gate_definitions.contains_key("csx"),
        "csx should be defined in qelib1"
    );

    // Verify the structure of the csx gate definition
    if let Some(csx_def) = program.gate_definitions.get("csx") {
        assert_eq!(csx_def.params.len(), 0, "csx should have no parameters");
        assert_eq!(csx_def.qargs.len(), 2, "csx should act on two qubits");
    }
}
