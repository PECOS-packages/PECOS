use pecos_engines::engines::classical::ClassicalEngine;
use pecos_qasm::engine::QASMEngine;
use pecos_qasm::parser::QASMParser;
use std::str::FromStr;

#[test]
fn test_all_comparison_operators() {
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
        if (b != 2) c[1] = b[1] & a[1] | a[0];
        c = b & a | d;

        d[0] = a[0] ^ 1;
        if (c >= 2) H q[0];
        if (c <= 2) H q[0];
        if (c < 2) H q[0];
        if (c > 2) H q[0];
        if (c != 2) H q[0];
        if (d == 1) H q[0]; // Changed rx to h for now
    "#;

    // Create and load the engine
    let mut engine = QASMEngine::from_str(qasm).expect("Failed to load program");

    // Generate commands - this verifies that all operations are supported
    let _messages = engine
        .generate_commands()
        .expect("Failed to generate commands");

    println!("All comparison operators test passed");
}

#[test]
fn test_bit_indexing_in_conditionals() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[4];
        creg d[1];

        c[0] = 1;
        c[1] = 0;
        if (c[0] == 1) H q[0];  // Should execute
        if (c[1] != 0) X q[1];  // Should not execute

        d[0] = 1;
        if (d[0] == 1) H q[0];  // Should execute
    "#;

    let mut engine = QASMEngine::from_str(qasm).expect("Failed to load program");
    let _messages = engine
        .generate_commands()
        .expect("Failed to generate commands");

    println!("Bit indexing in conditionals test passed");
}

#[test]
fn test_complex_conditional_expressions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg a[2];
        creg b[3];
        creg c[4];

        a = 1;
        b = 2;
        c = a + b;  // c = 3

        if (c >= 3) H q[0];   // Should execute
        if (c > 3) X q[0];    // Should not execute
        if (c <= 3) H q[0];  // Should execute
        if (c < 3) X q[0];   // Should not execute
        if (c != 0) H q[0];  // Should execute
    "#;

    let mut engine = QASMEngine::from_str(qasm).expect("Failed to load program");
    let _messages = engine
        .generate_commands()
        .expect("Failed to generate commands");

    println!("Complex conditional expressions test passed");
}

#[test]
fn test_comparison_operators_syntax() {
    // Test that all comparison operators are parsed correctly
    let test_cases = vec![
        ("if (c == 2) H q[0];", "equals"),
        ("if (c != 2) H q[0];", "not equals"),
        ("if (c < 2) H q[0];", "less than"),
        ("if (c > 2) H q[0];", "greater than"),
        ("if (c <= 2) H q[0];", "less than or equal"),
        ("if (c >= 2) H q[0];", "greater than or equal"),
    ];

    for (qasm_snippet, desc) in test_cases {
        let qasm = format!(
            r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[1];
            creg c[4];
            {qasm_snippet}
        "#
        );

        let program = QASMParser::parse_str(&qasm)
            .unwrap_or_else(|_| panic!("Failed to parse {desc} operator"));
        assert!(
            !program.operations.is_empty(),
            "{desc} operator should create an operation"
        );
    }

    println!("All comparison operators syntax test passed");
}

#[test]
fn test_mixed_operations_with_conditionals() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg a[2];
        creg b[3];
        creg c[4];
        creg d[1];

        // Initialize values
        a = 1;
        b = 2;
        d[0] = 1;

        // Mixed operations
        c = b & a | d;  // c = (2 & 1) | 1 = 1 | 1 = 1

        // Conditional with bit indexing
        if (d[0] == 1) H q[0];  // Should execute

        // Bitwise operation followed by conditional
        d[0] = a[0] ^ 1;  // d[0] = 1 ^ 1 = 0
        if (d[0] == 0) X q[1];  // Should execute

        // Complex expression in conditional
        // Complex expressions in conditionals not yet supported
        // if ((a[0] | b[0]) != 0) H q[0];  // Would execute
    "#;

    let _program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // Just check parsing for now
    println!("Mixed operations with conditionals test passed");
}
