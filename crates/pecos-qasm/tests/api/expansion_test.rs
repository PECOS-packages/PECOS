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

    // s gate expands to RZ(pi/2), which is native RZ
    // h gate expands to H (native)
    assert!(expanded.contains("H q"));
    assert!(expanded.contains("RZ("));
}
