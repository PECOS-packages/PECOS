use pecos_qasm::{ParseConfig, QASMParser};

#[test]
fn debug_include_behavior() {
    // Let's trace exactly what's happening with includes

    // Test case: User overrides qelib1.inc
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        h q[0];
    "#;

    let mut config = ParseConfig::default();
    config.includes.push((
        "qelib1.inc".to_string(),
        r"
        // Minimal custom qelib1 - just lowercase h mapping to native H
        gate h a {
            H a;
        }
        "
        .to_string(),
    ));

    let program = QASMParser::parse_with_config(qasm, &config).unwrap();

    // Debug: print what gates we have
    println!("Gates after parsing with custom qelib1:");
    for name in program.gate_definitions.keys() {
        println!("  - {name}");
    }

    // The issue might be that other includes are bringing in CX
    // Let's check if our minimal qelib1 actually replaced the system one
    assert!(program.gate_definitions.contains_key("h"));

    // This test shows what's actually happening
    if program.gate_definitions.contains_key("CX") {
        println!("UNEXPECTED: CX gate found even though custom qelib1 doesn't have it");
        println!("This means either:");
        println!("  1. System qelib1 is still being used somewhere");
        println!("  2. Another include is defining CX");
        println!("  3. The preprocessor isn't replacing includes as expected");
    } else {
        println!("SUCCESS: Only gates from custom qelib1 are present");
    }
}
