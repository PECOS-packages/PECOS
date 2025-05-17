//! Showcase the simplified QASM API

use pecos_engines::ClassicalEngine;
use pecos_qasm::QASMEngine;
use std::str::FromStr;

#[test]
fn test_simple_api() {
    // Simple case - from string
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        H q[0];
    "#;

    let engine = QASMEngine::from_str(qasm).unwrap();
    assert_eq!(engine.num_qubits(), 2);
}

#[test]
fn test_configurable_api() {
    // Complex case - with virtual includes and custom paths
    let qasm = r#"
        OPENQASM 2.0;
        include "custom.inc";
        qreg q[1];
        my_gate q[0];
    "#;

    let engine = QASMEngine::builder()
        .with_virtual_include("custom.inc", "gate my_gate a { H a; }")
        .with_include_path("/custom/path")
        .build_from_str(qasm)
        .unwrap();

    assert!(engine.gate_definitions().unwrap().contains_key("my_gate"));
}
