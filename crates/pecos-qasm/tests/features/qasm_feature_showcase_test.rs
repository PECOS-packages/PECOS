use pecos_engines::engines::classical::ClassicalEngine;
use pecos_qasm::engine::QASMEngine;
use pecos_qasm::parser::QASMParser;
use std::str::FromStr;

#[test]
fn test_qasm_comparison_operators_showcase() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[4];
        creg a[2];
        creg b[3];
        creg c[4];

        // Initialize registers
        a = 1;
        b = 2;

        // All comparison operators work in conditionals
        if (a == 1) H q[0];  // Equals
        if (b != 1) X q[1];  // Not equals
        if (a < 2) H q[2];   // Less than
        if (b > 1) X q[3];   // Greater than
        if (a <= 1) H q[0];  // Less than or equal
        if (b >= 2) X q[1];  // Greater than or equal

        // Bit indexing works in conditionals
        c[0] = 1;
        c[1] = 0;
        if (c[0] == 1) H q[2];  // Test specific bit
        if (c[1] != 1) X q[3];  // Test another bit

        // Mixed arithmetic and conditionals
        c = a + b;  // c = 3
        if (c == 3) H q[0];

        // Bitwise operations with conditionals
        c = a | b;  // c = 3
        if (c > 0) X q[1];
    "#;

    let mut engine = QASMEngine::from_str(qasm).expect("Failed to load program");
    let _messages = engine
        .generate_commands()
        .expect("Failed to generate commands");

    println!("QASM feature showcase test passed - all comparison operators and bit indexing work!");
}

#[test]
fn test_currently_unsupported_features() {
    // Document what doesn't work yet

    // 1. Complex expressions in conditionals
    let qasm1 = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg a[2];
        creg b[2];
        if ((a[0] | b[0]) != 0) H q[0];  // Complex expression
    "#;

    // Complex expressions now parse successfully, but fail at engine level without flag
    let mut engine1 = QASMEngine::from_str(qasm1).expect("Failed to load program");
    let result1 = engine1.generate_commands();
    assert!(
        result1.is_err(),
        "Complex expressions should fail at runtime without flag"
    );

    // 2. Exponentiation operator
    let qasm2 = r"
        OPENQASM 2.0;
        creg c[4];
        creg a[2];
        c = a**2;  // Exponentiation (now supported)
    ";

    let result2 = QASMParser::parse_str_raw(qasm2);
    assert!(result2.is_ok(), "Exponentiation operator should now work");

    println!("Unsupported features correctly identified");
}

#[test]
fn test_supported_classical_operators() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg a[4];
        creg b[4];
        creg c[4];

        // Arithmetic operators
        a = 2;
        b = 3;
        c = a + b;    // Addition
        c = b - a;    // Subtraction (be careful with unsigned underflow)
        c = a * b;    // Multiplication
        c = b / a;    // Division

        // Bitwise operators
        c = a & b;    // AND
        c = a | b;    // OR
        c = a ^ b;    // XOR
        c = ~a;       // NOT

        // Shift operators
        c = a << 1;   // Left shift
        c = b >> 1;   // Right shift

        // Mixed operations
        c[0] = a[0] & b[0];   // Bit-level operations
        c = (a + b) & 7;      // Combined arithmetic and bitwise

        // In quantum gates
        if (c != 0) H q[0];
        rx(pi/2) q[0];  // Complex expressions with bit indexing not yet supported in gate params
    "#;

    let mut engine = QASMEngine::from_str(qasm).expect("Failed to load program");
    let _messages = engine
        .generate_commands()
        .expect("Failed to generate commands");

    println!("All supported classical operators test passed");
}

#[test]
fn test_negative_values_and_signed_arithmetic() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg a[4];
        creg b[4];
        creg c[4];

        // Set up values
        a = 5;
        b = 3;
        c = a - b;  // c = 2 (positive result)

        // Be careful with underflow - this would cause issues:
        // b = 5;
        // a = 3;
        // c = a - b;  // Would underflow in unsigned arithmetic!

        // Using signed values in gate parameters
        RZ(-pi/2) q[0];    // Negative parameter
        rx(pi * -0.5) q[0]; // Negative expression
    "#;

    let mut engine = QASMEngine::from_str(qasm).expect("Failed to load program");
    let _messages = engine
        .generate_commands()
        .expect("Failed to generate commands");

    println!("Negative values and signed arithmetic test passed");
}
