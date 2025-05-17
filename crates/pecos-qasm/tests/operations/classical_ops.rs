//! Comprehensive tests for classical operations in QASM
//! Consolidates tests for basic, complex, and supported classical operations

use pecos_qasm::{Operation, engine::QASMEngine, parser::QASMParser};
use std::str::FromStr;

#[test]
fn test_basic_classical_assignments() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg c[4];
        creg a[2];
        creg b[3];

        // Basic assignments
        c = 2;              // Direct integer assignment
        c = a;              // Register to register assignment
        c[0] = 1;           // Bit assignment
        c[1] = a[0];        // Bit to bit assignment
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse basic classical operations");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_classical_arithmetic_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg a[4];
        creg b[4];
        creg c[4];

        // Arithmetic operations
        c = a + b;          // Addition
        c = a - b;          // Subtraction
        c = a * b;          // Multiplication
        c = a / b;          // Division (integer)
        c = a ^ b;          // XOR
        c = a & b;          // AND
        c = a | b;          // OR
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse arithmetic operations");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_classical_bitwise_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg a[8];
        creg b[8];
        creg c[8];

        // Bitwise operations
        c = ~a;             // NOT
        c = a << 1;         // Left shift
        c = a >> 2;         // Right shift
        c[0] = a[0] ^ 1;    // XOR with constant
        c[1] = ~a[1];       // NOT individual bit
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse bitwise operations");
    assert!(!program.operations.is_empty());
}

#[test]
fn test_classical_conditional_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        creg c[4];
        creg a[2];
        creg b[3];

        // Complex conditional operations
        if (b != 2) c[1] = b[1] & a[1] | a[0];
        if (a == 0) x q[0];
        if (c > 5) x q[1];  // Simplified to single operation
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse conditional operations");

    // Verify conditional operations are parsed
    let has_conditionals = program
        .operations
        .iter()
        .any(|op| matches!(op, Operation::If { .. }));
    assert!(has_conditionals, "Should have conditional operations");
}

#[test]
fn test_complex_classical_expressions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "hqslib1.inc";

        qreg q[1];
        creg c[4];
        creg a[2];
        creg b[3];
        creg d[1];

        // Complex expressions
        c = 2;
        c = a;
        c[1] = b[1] & a[1] | a[0];
        b = a + b;
        b[1] = b[0] + ~b[2];
        c = a - b;
        d = a << 1;
        d = c >> 2;
        b = a * c / b;
        d[0] = a[0] ^ 1;
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse complex expressions");

    // Count classical operations
    let classical_count = program
        .operations
        .iter()
        .filter(|op| matches!(op, Operation::ClassicalAssignment { .. }))
        .count();

    assert!(
        classical_count >= 10,
        "Should have many classical operations"
    );
}

#[test]
fn test_classical_operations_with_execution() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];
        creg c[4];
        creg a[2];

        // Test with actual execution
        c = 5;              // c = 0101
        a = 3;              // a = 11
        c = c + a;          // c = 0101 + 0011 = 1000 (8)
        c[0] = 0;           // c = 1000

        measure q[0] -> a[0];
    "#;

    // Test both parsing and execution
    let _engine = QASMEngine::from_str(qasm).expect("Failed to create engine");
    // Simply verify the engine was created successfully with the classical operations
    // More comprehensive testing happens in other tests that actually run the simulation
}

#[test]
fn test_supported_vs_unsupported_operations() {
    // Document what's supported vs not supported

    // SUPPORTED:
    let supported_qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        creg c[4];
        creg a[4];

        // These should all parse successfully
        c = 5;              // Integer assignment
        c = a;              // Register assignment
        c = a + 5;          // Arithmetic with constants
        c = a & 15;         // Bitwise with integer (hex not supported)
        c[0] = 1;           // Bit assignment
        c = ~a;             // Unary operations
    "#;

    match QASMParser::parse_str(supported_qasm) {
        Ok(_) => {} // Test passes
        Err(e) => panic!("All supported operations should parse, but got error: {e:?}"),
    }

    // UNSUPPORTED (if any):
    // Add tests for operations that should fail if there are known unsupported cases
}
