use pecos_qasm::{ParseConfig, QASMParser};

#[test]
fn test_simple_unified_includes() {
    // The simple unified system: last write wins

    // Test 1: Default behavior - system includes are pre-loaded
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        h q[0];
    "#;

    let program1 = QASMParser::parse_str(qasm).unwrap();
    assert!(program1.gate_definitions.contains_key("h"));
    assert!(program1.gate_definitions.contains_key("cx")); // System qelib1 has many gates

    // Test 2: User override - last write wins
    let mut config = ParseConfig::default();
    config.includes.push((
        "qelib1.inc".to_string(),
        r"
        // Custom qelib1.inc - only has h gate
        gate h a {
            H a;
        }
        "
        .to_string(),
    ));

    let program2 = QASMParser::parse_with_config(qasm, &config).unwrap();
    assert!(program2.gate_definitions.contains_key("h"));
    assert!(!program2.gate_definitions.contains_key("cx")); // User version only has h

    // Test 3: Mixed sources - user provides custom.inc, system provides qelib1
    let qasm_mixed = r#"
        OPENQASM 2.0;
        include "custom.inc";   // User provided
        include "qelib1.inc";   // Will use system version
        qreg q[1];
        my_gate q[0];
        h q[0];
    "#;

    let mut config = ParseConfig::default();
    config.includes.push((
        "custom.inc".to_string(),
        "gate my_gate a { X a; }".to_string(),
    ));
    // Don't override qelib1 - let system version be used

    let program3 = QASMParser::parse_with_config(qasm_mixed, &config).unwrap();
    assert!(program3.gate_definitions.contains_key("my_gate")); // From user custom.inc
    assert!(program3.gate_definitions.contains_key("h")); // From system qelib1
    assert!(program3.gate_definitions.contains_key("cx")); // System qelib1 has cx
}
