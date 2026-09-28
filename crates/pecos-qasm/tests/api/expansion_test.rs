use pecos_core::prelude::{Angle64, GateType};
use pecos_qasm::Operation;
use pecos_qasm::parser::QASMParser;

#[test]
fn test_preprocess_and_expand() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];

        gate bell a, b {
            H a;
            CX a, b;
        }

        bell q[0], q[1];
    "#;

    // Test phase 1: Just preprocessing
    let preprocessed = QASMParser::preprocess(qasm).unwrap();
    println!("After Phase 1 (includes resolved):");
    println!("{preprocessed}");
    assert!(preprocessed.contains("gate h")); // Should have qelib1.inc contents
    assert!(preprocessed.contains("gate bell")); // Should still have user gates

    // Test phases 1 and 2: Preprocessing and expansion
    let expanded = QASMParser::preprocess_and_expand(qasm).unwrap();
    println!("\nAfter Phase 2 (gates expanded):");
    println!("{expanded}");
    assert!(!expanded.contains("gate bell")); // User gates should be gone
    assert!(!expanded.contains("bell q")); // Gate calls should be expanded
    assert!(expanded.contains("H q")); // Should have native operations
}

#[test]
fn test_expansion_details() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];

        // This gate uses non-native gates
        gate my_gate a {
            H a;
            s a;
            H a;
        }

        my_gate q[0];
    "#;

    let expanded = QASMParser::preprocess_and_expand(qasm).unwrap();
    println!("Expanded QASM:");
    println!("{expanded}");

    // The user gate is gone and its body is emitted under the canonical native
    // names the parser accepts without an include.
    assert!(!expanded.contains("my_gate q"));
    assert!(expanded.contains("H q"));
    assert!(expanded.contains("SZ q"));
}

#[test]
fn expanded_qasm_keeps_rotation_angles_and_re_parses() {
    // A rotation inside a gate body must come out with its angle, in a form the
    // parser accepts again; the value is compared as an `Angle64`, not as text.
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];
        gate g(t) a { RZ(t) a; }
        g(0.5) q[0];
    ";
    let expanded = QASMParser::preprocess_and_expand(qasm).unwrap();
    assert!(expanded.contains("RZ("), "angle missing from: {expanded}");

    let reparsed = QASMParser::parse_str(&expanded).unwrap();
    assert_eq!(reparsed.operations.len(), 1);
    let Operation::NativeGate(gate) = &reparsed.operations[0] else {
        panic!("expected a native gate, got {:?}", reparsed.operations[0]);
    };
    assert_eq!(gate.gate_type, GateType::RZ);
    assert_eq!(gate.angles.len(), 1);
    assert_eq!(gate.angles[0], Angle64::from_radians(0.5));
}

#[test]
fn expanded_conditional_qasm_keeps_registers_and_angles() {
    // The mapped renderer used to hand conditional bodies to the plain
    // `Display`, which prints `gid[..]` and drops angles.
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];
        creg c[1];
        gate g(t) a { RZ(t) a; }
        if (c == 1) g(0.5) q[0];
    ";
    let expanded = QASMParser::preprocess_and_expand(qasm).unwrap();
    // The condition prints through `Expression`'s own `Display`, so its exact
    // bracketing is not asserted; the operation, its angle and its register are.
    assert!(
        expanded.contains("if (") && expanded.contains("RZ(0.5) q[0]"),
        "conditional lost: {expanded}"
    );
    assert!(!expanded.contains("gid["), "register lost: {expanded}");

    let reparsed = QASMParser::parse_str(&expanded).unwrap();
    assert_eq!(reparsed.operations.len(), 1);
    let Operation::If { operation, .. } = &reparsed.operations[0] else {
        panic!("expected a conditional, got {:?}", reparsed.operations[0]);
    };
    let Operation::NativeGate(gate) = operation.as_ref() else {
        panic!("expected a native gate body, got {operation:?}");
    };
    assert_eq!(gate.gate_type, GateType::RZ);
    assert_eq!(gate.angles[0], Angle64::from_radians(0.5));
}
