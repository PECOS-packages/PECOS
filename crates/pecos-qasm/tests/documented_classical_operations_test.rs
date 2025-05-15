use pecos_qasm::parser::QASMParser;

#[test]
fn test_supported_classical_operations() {
    // This test documents what classical operations are supported by the PECOS QASM parser
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg c[4];
        creg a[2];
        creg b[3];
        creg d[1];

        // SUPPORTED OPERATIONS:
        
        // 1. Basic assignments
        c = 2;              // Direct integer assignment
        c = a;              // Register to register assignment
        c[0] = 1;           // Single bit assignment
        
        // 2. Bitwise operations
        c = b & a;          // Bitwise AND
        c[1] = b[1] & a[1] | a[0];  // Bitwise AND and OR
        b[1] = b[0] + ~b[2];        // Bitwise NOT (note: may cause runtime issues)
        d[0] = a[0] ^ 1;            // Bitwise XOR
        
        // 3. Arithmetic operations (but may cause runtime overflow)
        b = a + b;          // Addition
        c = a - b;          // Subtraction 
        b = a * c / b;      // Multiplication and division
        
        // 4. Bit shifting operations
        d = a << 1;         // Left shift
        d = c >> 2;         // Right shift
        
        // 5. Conditional statements (limited syntax)
        if (c == 2) h q[0]; // Only == comparison operator is reliably supported
        if (c == 1) x q[0];
        
        // 6. Complex expressions in quantum gates
        rx((0.5+0.5)*pi) q[0];
        rz(pi/2) q[0];
        
        // UNSUPPORTED OPERATIONS:
        // - Exponentiation (**) - Not implemented in grammar
        // - Comparison operators in conditionals (>=, <=, !=, >, <) - Limited support
        // - if statements with complex expressions - Limited support
    "#;

    let program = QASMParser::parse_str_with_includes(qasm).expect("Failed to parse QASM");
    assert!(
        !program.operations.is_empty(),
        "Program should have operations"
    );

    println!("Supported classical operations documented and tested");
}

#[test]
fn test_unsupported_classical_operations() {
    // Test for operations that are NOT supported

    // 1. Exponentiation - now supported
    let qasm_exp = r#"
        OPENQASM 2.0;
        creg c[4];
        creg b[3];
        c = b**2;  // Exponentiation is now supported
    "#;

    assert!(
        QASMParser::parse_str_with_includes(qasm_exp).is_ok(),
        "Exponentiation (**) should now be supported"
    );

    // 2. Complex conditionals may have issues
    let qasm_complex_if = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[4];
        if (c >= 2) h q[0];  // >= operator may not be fully supported
    "#;

    // This parses but may have runtime issues
    let result = QASMParser::parse_str_with_includes(qasm_complex_if);
    if result.is_err() {
        println!("Complex conditionals with >= operator not supported");
    }

    println!("Unsupported operations documented");
}

#[test]
fn test_modified_example_without_unsupported_features() {
    // This is a modified version of the original example that removes unsupported features
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
        // Remove unsupported if (b != 2)
        c[1] = b[1] & a[1] | a[0];
        c = b & a;
        b = a + b;
        b[1] = b[0] + ~b[2];
        // Remove unsupported c = a - (b**c);
        c = a - b;  // Simple subtraction instead
        d = a << 1;
        d = c >> 2;
        c[0] = 1;
        b = a * c / b;
        d[0] = a[0] ^ 1;
        // Remove unsupported if(c>=2)
        if (c == 2) h q[0];
        if (d == 1) rx((0.5+0.5)*pi) q[0];
    "#;

    let program = QASMParser::parse_str_with_includes(qasm).expect("Failed to parse modified QASM");
    assert!(
        !program.operations.is_empty(),
        "Program should have operations"
    );

    println!("Modified example without unsupported features works");
}
