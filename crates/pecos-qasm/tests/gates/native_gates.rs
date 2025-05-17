use pecos_qasm::ast::Operation;
use pecos_qasm::parser::QASMParser;

#[test]
fn test_lowercase_gates_resolve_to_uppercase() {
    let qasm_str = r#"
    OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[2];
    H q[0];   // lowercase h
    X q[1];   // lowercase x
    H q[0];   // uppercase H
    X q[1];   // uppercase X
    "#;

    let program = QASMParser::parse_str(qasm_str).expect("Failed to parse QASM");

    // Check that the operations are expanded correctly
    let gate_ops: Vec<_> = program
        .operations
        .iter()
        .filter_map(|op| match op {
            Operation::Gate { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();

    // After expansion, all should be uppercase native gates
    assert_eq!(gate_ops, vec!["H", "X", "H", "X"]);
}

#[test]
fn test_native_gate_list_has_no_lowercase() {
    // This test verifies that only uppercase gates are native
    // CX is still native in PECOS, so it doesn't need to be defined
    let qasm_str = r"
    OPENQASM 2.0;

    qreg q[2];
    CX q[0], q[1];
    ";

    let program = QASMParser::parse_str(qasm_str).expect("Failed to parse QASM");

    // Check that CX works as a native gate (uppercase)
    let gate_ops: Vec<_> = program
        .operations
        .iter()
        .filter_map(|op| match op {
            Operation::Gate { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(gate_ops, vec!["CX"]);

    // Now test that lowercase gates need to be defined in qelib1
    let qasm_str2 = r#"
    OPENQASM 2.0;
    include "qelib1.inc";

    qreg q[2];
    cx q[0], q[1];  // lowercase cx from qelib1
    "#;

    let program2 = QASMParser::parse_str(qasm_str2).expect("Failed to parse QASM");

    // After expansion, lowercase cx should be expanded to uppercase CX
    let gate_ops2: Vec<_> = program2
        .operations
        .iter()
        .filter_map(|op| match op {
            Operation::Gate { name, .. } => Some(name.as_str()),
            _ => None,
        })
        .collect();

    assert_eq!(gate_ops2, vec!["CX"]);
}

#[test]
fn test_lowercase_undefined_gate_error() {
    // Test that lowercase gates without definitions fail
    let qasm_str = r"
    OPENQASM 2.0;

    qreg q[1];
    h q[0];   // This should fail without qelib1.inc
    ";

    let result = QASMParser::parse_str(qasm_str);
    assert!(result.is_err());
    let err = result.unwrap_err();
    assert!(err.to_string().contains("Undefined gate 'h'"));
}
