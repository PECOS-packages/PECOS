use pecos_engines::engines::classical::ClassicalEngine;
use pecos_qasm::{QASMEngine, QASMEngineBuilder};

#[test]
fn test_standard_conditionals_always_work() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg c[4];
        creg d[1];

        c = 2;
        d[0] = 1;
        
        // These should always work (standard OpenQASM 2.0)
        if (c == 2) H q[0];
        if (d[0] == 1) X q[0];
        if (c > 1) H q[0];
        if (c <= 3) X q[0];
    "#;

    let mut engine = QASMEngine::from_str(qasm)
        .expect("Failed to load program");

    // Don't enable complex conditionals
    assert!(!engine.complex_conditionals_enabled());
    let _messages = engine
        .generate_commands()
        .expect("Failed to generate commands");

    println!("Standard conditionals test passed");
}

#[test]
fn test_complex_conditionals_fail_by_default() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg a[2];
        creg b[2];

        a = 1;
        b = 2;
        
        // This should fail (not standard OpenQASM 2.0)
        if (a[0] & b[0] == 1) H q[0];
    "#;

    let mut engine = QASMEngine::from_str(qasm)
        .expect("Failed to load program");

    // Don't enable complex conditionals (should be false by default)
    assert!(!engine.complex_conditionals_enabled());
    let result = engine.generate_commands();

    assert!(
        result.is_err(),
        "Complex conditionals should fail by default"
    );
    if let Err(error) = result {
        let error_msg = error.to_string();
        assert!(
            error_msg.contains("Complex conditionals are not allowed"),
            "Should get proper error message, got: {error_msg}"
        );
    }
}

#[test]
fn test_complex_conditionals_work_with_flag() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg a[2];
        creg b[2];

        a = 1;
        b = 1;
        
        // This should work when flag is enabled
        if ((a[0] & b[0]) == 1) H q[0];
    "#;

    let mut engine = QASMEngineBuilder::new()
        .allow_complex_conditionals(true)
        .build_from_str(qasm)
        .expect("Failed to load program");

    // Enable complex conditionals
    assert!(engine.complex_conditionals_enabled());

    let _messages = engine
        .generate_commands()
        .expect("Failed to generate commands with complex conditionals enabled");

    println!("Complex conditionals with flag test passed");
}

#[test]
fn test_register_to_register_comparison_fails() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg a[2];
        creg b[2];

        a = 1;
        b = 2;
        
        // This should fail (register compared to register, not integer)
        if (a < b) H q[0];
    "#;

    let mut engine = QASMEngine::from_str(qasm)
        .expect("Failed to load program");
    let result = engine.generate_commands();

    assert!(
        result.is_err(),
        "Register to register comparison should fail"
    );
    if let Err(error) = result {
        let error_msg = error.to_string();
        assert!(
            error_msg.contains("Complex conditionals are not allowed"),
            "Should get proper error message, got: {error_msg}"
        );
    }
}

#[test]
fn test_expression_to_expression_fails() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg a[2];
        
        a = 2;
        
        // This should fail (expression compared to expression, not simple register to int)
        if ((a + 1) == 3) H q[0];
    "#;

    let mut engine = QASMEngine::from_str(qasm)
        .expect("Failed to load program");
    let result = engine.generate_commands();

    assert!(
        result.is_err(),
        "Expression to expression comparison should fail"
    );
    if let Err(error) = result {
        let error_msg = error.to_string();
        assert!(
            error_msg.contains("Complex conditionals are not allowed"),
            "Should get proper error message, got: {error_msg}"
        );
    }
}

#[test]
fn test_toggle_feature_flag() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg a[2];
        
        a = 2;
        
        // This should fail or succeed based on flag
        if ((a + 1) == 3) H q[0];
    "#;

    // Test with flag disabled
    let mut engine1 = QASMEngine::from_str(qasm)
        .expect("Failed to load program");
    let result1 = engine1.generate_commands();
    assert!(result1.is_err(), "Should fail without flag");

    // Test with flag enabled
    let mut engine2 = QASMEngineBuilder::new()
        .allow_complex_conditionals(true)
        .build_from_str(qasm)
        .expect("Failed to load program");
    let result2 = engine2.generate_commands();
    assert!(result2.is_ok(), "Should succeed with flag enabled");
}
