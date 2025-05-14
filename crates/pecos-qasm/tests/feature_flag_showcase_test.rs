use pecos_qasm::parser::QASMParser;
use pecos_qasm::engine::QASMEngine;
use pecos_engines::engines::classical::ClassicalEngine;

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
        if (c == 2) h q[0];      // Register compared to int
        if (c != 0) x q[1];      // Register compared to int
        if (c > 1) h q[0];       // Register compared to int
        
        d[0] = 1;
        if (d[0] == 1) x q[1];   // Bit compared to int
        if (c <= 3) h q[0];      // Register compared to int
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
        if (a < b) h q[0];                  // Register compared to register
        if ((a + b) == 5) x q[1];          // Expression compared to int
        if (a[0] & b[0] == 0) h q[0];      // Bitwise operation in condition
        if ((a * 2) > b) x q[1];           // Complex expression
    "#;
    
    // Standard QASM should work without any flags
    let program1 = QASMParser::parse_str(standard_qasm).expect("Standard QASM should parse");
    let mut engine1 = QASMEngine::new().expect("Failed to create engine");
    assert!(!engine1.allow_complex_conditionals(), "Complex conditionals should be disabled by default");
    engine1.load_program(program1).expect("Failed to load program");
    engine1.generate_commands().expect("Standard QASM should execute without extended features");
    
    // Extended QASM should fail without the flag
    let program2 = QASMParser::parse_str(extended_qasm).expect("Extended QASM should parse");
    let mut engine2 = QASMEngine::new().expect("Failed to create engine");
    engine2.load_program(program2.clone()).expect("Failed to load program");
    let result = engine2.generate_commands();
    assert!(result.is_err(), "Extended QASM should fail without flag");
    
    // Extended QASM should work with the flag
    let mut engine3 = QASMEngine::new().expect("Failed to create engine");
    engine3.set_allow_complex_conditionals(true);
    assert!(engine3.allow_complex_conditionals(), "Complex conditionals should be enabled");
    engine3.load_program(program2).expect("Failed to load program");
    engine3.generate_commands().expect("Extended QASM should execute with flag enabled");
    
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
        
        if (a < b) h q[0];  // Should fail without flag
    "#;
    
    let program = QASMParser::parse_str(qasm).expect("Should parse");
    let mut engine = QASMEngine::new().expect("Failed to create engine");
    engine.load_program(program).expect("Failed to load program");
    
    let result = engine.generate_commands();
    assert!(result.is_err());
    
    if let Err(error) = result {
        let error_msg = error.to_string();
        assert!(error_msg.contains("Complex conditionals are not allowed"));
        assert!(error_msg.contains("register/bit compared to integer"));
        assert!(error_msg.contains("standard OpenQASM 2.0"));
        assert!(error_msg.contains("allow_complex_conditionals"));
        println!("Error message is helpful: {}", error_msg);
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
        if (c == 3) h q[0];
        if (a[0] == 1) x q[1];
        
        // This extended conditional should fail without flag
        if (a != b) h q[0];
    "#;
    
    let program = QASMParser::parse_str(qasm).expect("Should parse");
    let mut engine = QASMEngine::new().expect("Failed to create engine");
    engine.load_program(program).expect("Failed to load program");
    
    // Should fail on the extended conditional
    let result = engine.generate_commands();
    assert!(result.is_err(), "Should fail on extended conditional");
    
    // Now enable the flag and try again
    let program2 = QASMParser::parse_str(qasm).expect("Should parse");
    let mut engine2 = QASMEngine::new().expect("Failed to create engine");
    engine2.set_allow_complex_conditionals(true);
    engine2.load_program(program2).expect("Failed to load program");
    
    // Should succeed with flag enabled
    let result2 = engine2.generate_commands();
    assert!(result2.is_ok(), "Should succeed with flag enabled");
}