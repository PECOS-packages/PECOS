use pecos_qasm::parser::QASMParser;
use pecos_qasm::engine::QASMEngine;
use pecos_engines::engines::classical::ClassicalEngine;

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
        if (c == 2) h q[0];
        if (d[0] == 1) x q[0];
        if (c > 1) h q[0];
        if (c <= 3) x q[0];
    "#;
    
    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    let mut engine = QASMEngine::new().expect("Failed to create engine");
    
    // Don't enable complex conditionals
    assert!(!engine.allow_complex_conditionals());
    
    engine.load_program(program).expect("Failed to load program");
    let _messages = engine.generate_commands().expect("Failed to generate commands");
    
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
        if (a[0] & b[0] == 1) h q[0];
    "#;
    
    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    let mut engine = QASMEngine::new().expect("Failed to create engine");
    
    // Don't enable complex conditionals (should be false by default)
    assert!(!engine.allow_complex_conditionals());
    
    engine.load_program(program).expect("Failed to load program");
    let result = engine.generate_commands();
    
    assert!(result.is_err(), "Complex conditionals should fail by default");
    if let Err(error) = result {
        let error_msg = error.to_string();
        assert!(error_msg.contains("Complex conditionals are not allowed"),
                "Should get proper error message, got: {}", error_msg);
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
        if ((a[0] & b[0]) == 1) h q[0];
    "#;
    
    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    let mut engine = QASMEngine::new().expect("Failed to create engine");
    
    // Enable complex conditionals
    engine.set_allow_complex_conditionals(true);
    assert!(engine.allow_complex_conditionals());
    
    engine.load_program(program).expect("Failed to load program");
    let _messages = engine.generate_commands().expect("Failed to generate commands with complex conditionals enabled");
    
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
        if (a < b) h q[0];
    "#;
    
    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    let mut engine = QASMEngine::new().expect("Failed to create engine");
    
    engine.load_program(program).expect("Failed to load program");
    let result = engine.generate_commands();
    
    assert!(result.is_err(), "Register to register comparison should fail");
    if let Err(error) = result {
        let error_msg = error.to_string();
        assert!(error_msg.contains("Complex conditionals are not allowed"),
                "Should get proper error message, got: {}", error_msg);
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
        if ((a + 1) == 3) h q[0];
    "#;
    
    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    let mut engine = QASMEngine::new().expect("Failed to create engine");
    
    engine.load_program(program).expect("Failed to load program");
    let result = engine.generate_commands();
    
    assert!(result.is_err(), "Expression to expression comparison should fail");
    if let Err(error) = result {
        let error_msg = error.to_string();
        assert!(error_msg.contains("Complex conditionals are not allowed"),
                "Should get proper error message, got: {}", error_msg);
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
        if ((a + 1) == 3) h q[0];
    "#;
    
    let program1 = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    let program2 = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    
    // Test with flag disabled
    let mut engine1 = QASMEngine::new().expect("Failed to create engine");
    engine1.load_program(program1).expect("Failed to load program");
    let result1 = engine1.generate_commands();
    assert!(result1.is_err(), "Should fail without flag");
    
    // Test with flag enabled
    let mut engine2 = QASMEngine::new().expect("Failed to create engine");
    engine2.set_allow_complex_conditionals(true);
    engine2.load_program(program2).expect("Failed to load program");
    let result2 = engine2.generate_commands();
    assert!(result2.is_ok(), "Should succeed with flag enabled");
}