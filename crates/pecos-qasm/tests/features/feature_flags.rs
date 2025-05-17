use pecos_engines::engines::classical::ClassicalEngine;
use pecos_qasm::engine::QASMEngine;
use std::str::FromStr;

#[test]
fn test_openqasm_standard_vs_extended() {
    // This QASM follows standard OpenQASM 2.0 spec
    let standard_qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[4];
        creg d[1];

        // These are all valid in standard OpenQASM 2.0
        c = 2;
        if (c == 2) H q[0];      // Register compared to int
        if (c != 0) X q[1];      // Register compared to int
        if (c > 1) H q[0];       // Register compared to int

        d[0] = 1;
        if (d[0] == 1) X q[1];   // Bit compared to int
        if (c <= 3) H q[0];      // Register compared to int
    "#;

    // This QASM uses extended features
    let extended_qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg a[4];
        creg b[4];
        creg c[4];

        a = 2;
        b = 3;

        // These require the extended feature flag
        if (a < b) H q[0];                  // Register compared to register
        if ((a + b) == 5) X q[1];          // Expression compared to int
        if (a[0] & b[0] == 0) H q[0];      // Bitwise operation in condition
        if ((a * 2) > b) X q[1];           // Complex expression
    "#;

    // Standard QASM should work without any flags
    let mut engine1 = QASMEngine::from_str(standard_qasm).expect("Failed to load program");
    assert!(
        !engine1.complex_conditionals_enabled(),
        "Complex conditionals should be disabled by default"
    );
    engine1
        .generate_commands()
        .expect("Standard QASM should execute without extended features");

    // Extended QASM should fail without the flag
    let mut engine2 = QASMEngine::from_str(extended_qasm).expect("Failed to load program");
    let result = engine2.generate_commands();
    assert!(result.is_err(), "Extended QASM should fail without flag");

    // Extended QASM should work with the flag
    let mut engine3 = QASMEngine::builder()
        .allow_complex_conditionals(true)
        .build_from_str(extended_qasm)
        .expect("Failed to load program");
    assert!(
        engine3.complex_conditionals_enabled(),
        "Complex conditionals should be enabled"
    );
    engine3
        .generate_commands()
        .expect("Extended QASM should execute with flag enabled");

    println!("Feature flag showcase test completed successfully");
}

#[test]
fn test_error_messages_are_helpful() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg a[2];
        creg b[2];

        a = 1;
        b = 2;

        if (a < b) H q[0];  // Should fail without flag
    "#;

    let mut engine = QASMEngine::from_str(qasm).expect("Failed to load program");

    let result = engine.generate_commands();
    assert!(result.is_err());

    if let Err(error) = result {
        let error_msg = error.to_string();
        assert!(error_msg.contains("Complex conditionals are not allowed"));
        assert!(error_msg.contains("register/bit compared to integer"));
        assert!(error_msg.contains("standard OpenQASM 2.0"));
        assert!(error_msg.contains("allow_complex_conditionals"));
        println!("Error message is helpful: {error_msg}");
    }
}

#[test]
fn test_mixed_conditionals() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg a[2];
        creg b[2];
        creg c[4];

        a = 1;
        b = 2;
        c = 3;

        // Standard conditionals should work
        if (c == 3) H q[0];
        if (a[0] == 1) X q[1];

        // This extended conditional should fail without flag
        if (a != b) H q[0];
    "#;

    let mut engine = QASMEngine::from_str(qasm).expect("Failed to load program");

    // Should fail on the extended conditional
    let result = engine.generate_commands();
    assert!(result.is_err(), "Should fail on extended conditional");

    // Now enable the flag and try again
    let mut engine2 = QASMEngine::builder()
        .allow_complex_conditionals(true)
        .build_from_str(qasm)
        .expect("Failed to load program");

    // Should succeed with flag enabled
    let result2 = engine2.generate_commands();
    assert!(result2.is_ok(), "Should succeed with flag enabled");
}
