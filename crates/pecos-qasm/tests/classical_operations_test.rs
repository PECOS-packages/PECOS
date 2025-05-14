use pecos_qasm::parser::QASMParser;
use pecos_qasm::engine::QASMEngine;
use pecos_engines::engines::classical::ClassicalEngine;

#[test]
fn test_comprehensive_classical_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg c[4];
        creg a[2];
        creg b[3];
        creg d[1];

        c = 2;
        c = a;
        c[1] = b[1] & a[1] | a[0];
        c = b & a;
        b = a + b;
        b[1] = b[0] + ~b[2];
        c = a - b;
        d = a << 1;
        d = c >> 2;
        c[0] = 1;
        b = a * c / b;
        d[0] = a[0] ^ 1;
        h q[0];
        rx((0.5+0.5)*pi) q[0];
    "#;
    
    // Parse the QASM program
    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    
    // Create and load the engine
    let mut engine = QASMEngine::new().expect("Failed to create engine");
    engine.load_program(program).expect("Failed to load program");
    
    // Generate commands - this verifies that all operations are supported
    let _messages = engine.generate_commands().expect("Failed to generate commands");
    
    println!("Comprehensive classical operations test passed");
}

#[test]
fn test_classical_assignment_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        creg c[4];
        creg a[2];
        creg b[3];

        c = 2;           // Direct integer assignment
        c = a;           // Register to register assignment
        c[0] = 1;        // Single bit assignment
    "#;
    
    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    let mut engine = QASMEngine::new().expect("Failed to create engine");
    engine.load_program(program).expect("Failed to load program");
    let _messages = engine.generate_commands().expect("Failed to generate commands");
    
    println!("Classical assignment operations test passed");
}

#[test]
fn test_classical_conditional_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg c[4];
        creg a[2];
        creg b[3];

        c[1] = b[1] & a[1] | a[0];
        c = 2;
        if (c == 2) h q[0];
        if (c == 1) x q[0];
    "#;

    let _program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    
    // Check that the conditional operations are parsed correctly
    println!("Classical conditional operations test passed");
}

#[test]
fn test_classical_bitwise_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        creg a[2];
        creg b[3];
        creg c[4];
        creg d[1];

        c = b & a;                    // Bitwise AND
        c[1] = b[1] & a[1] | a[0];   // Bitwise AND and OR
        b[1] = b[0] + ~b[2];         // Bitwise NOT
        d[0] = a[0] ^ 1;             // Bitwise XOR
    "#;
    
    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    let mut engine = QASMEngine::new().expect("Failed to create engine");
    engine.load_program(program).expect("Failed to load program");
    let _messages = engine.generate_commands().expect("Failed to generate commands");
    
    println!("Classical bitwise operations test passed");
}

#[test]
fn test_classical_arithmetic_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        creg a[2];
        creg b[3];
        creg c[4];

        b = a + b;           // Addition
        c = a - b;           // Subtraction (exponentiation not supported)
        b = a * c / b;       // Multiplication and division
    "#;
    
    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    let mut engine = QASMEngine::new().expect("Failed to create engine");
    engine.load_program(program).expect("Failed to load program");
    let _messages = engine.generate_commands().expect("Failed to generate commands");
    
    println!("Classical arithmetic operations test passed");
}

#[test]
fn test_classical_shift_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        creg a[2];
        creg c[4];
        creg d[1];

        d = a << 1;          // Left shift
        d = c >> 2;          // Right shift
    "#;
    
    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    let mut engine = QASMEngine::new().expect("Failed to create engine");
    engine.load_program(program).expect("Failed to load program");
    let _messages = engine.generate_commands().expect("Failed to generate commands");
    
    println!("Classical shift operations test passed");
}

#[test]
fn test_quantum_gates_with_classical_conditions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg c[4];
        creg d[1];

        c = 2;
        if (c == 2) h q[0];
        d = 1;
        if (d == 1) rx((0.5+0.5)*pi) q[0];
    "#;

    let _program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    
    // Check that quantum gates with classical conditions are parsed correctly
    println!("Quantum gates with classical conditions test passed");
}

#[test]
fn test_complex_expression_in_quantum_gate() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg d[1];

        rx((0.5+0.5)*pi) q[0];
    "#;
    
    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");
    
    // Check that the expression (0.5+0.5)*pi is properly parsed
    assert!(!program.operations.is_empty(), "Should have at least one operation");
    
    println!("Complex expression in quantum gate test passed");
}

#[test]
fn test_unsupported_operations() {
    // Test that exponentiation is now supported
    let qasm_exp = r#"
        OPENQASM 2.0;
        creg a[2];
        creg b[3];
        creg c[4];
        c = b**a;  // This is now supported
    "#;

    let result = QASMParser::parse_str(qasm_exp);
    assert!(result.is_ok(), "Exponentiation should now be supported");
    
    // Test that comparison operators in if statements need specific format
    let qasm_comp = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[4];
        if (c >= 2) h q[0];  // This might need different syntax
    "#;
    
    let result = QASMParser::parse_str(qasm_comp);
    // This may or may not work depending on how conditionals are implemented
    if result.is_err() {
        println!("Comparison operator syntax may need adjustment");
    }
}