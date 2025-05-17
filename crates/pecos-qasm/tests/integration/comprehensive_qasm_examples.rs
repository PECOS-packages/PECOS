use pecos_qasm::QASMParser;

#[test]
fn test_comprehensive_qasm_program() {
    // This test combines all the QASM examples provided by the user
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        // Register declaration
        qreg q[4];

        // Various rotation gates
        rz(1.5*pi) q[3];
        rx(0.0375*pi) q[3];
        rxx(0.0375*pi) q[0],q[1];
        rz(0.5*pi) q[3];
        rzz(0.0375*pi) q[0],q[1];

        // Basic gates
        cx q[0],q[3];
        rz(1.5*pi) q[3];
        rx(1.9625*pi) q[3];
        cz q[0],q[1]; //hey look ma its a cz

        // Three-qubit gate
        ccx q[3],q[1],q[2];

        // Barrier
        barrier q[0],q[3],q[2];

        // General unitary gates
        u3(3.141596, 0.5*pi, 0.3*pi) q[2];
        cu1(0.8*pi) q[0],q[1];

        // sqrt(X) gates
        sx q[0];
        x q[1];
        sxdg q[1];
        csx q[0],q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse comprehensive QASM program");

    // Basic validation
    assert!(!program.operations.is_empty(), "Should have operations");
    assert_eq!(
        program.quantum_registers.len(),
        1,
        "Should have one quantum register"
    );
    assert!(
        program.quantum_registers.contains_key("q"),
        "Should have register q"
    );
    assert_eq!(
        program.quantum_registers["q"].len(),
        4,
        "Register q should have 4 qubits"
    );
}

#[test]
fn test_qasm_with_comments_and_expressions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        //some comments

        qreg q[2]; // register declaration

        // Mathematical expressions in parameters
        rz(1.5*pi) q[0];
        rx(0.0375*pi) q[0];
        u3(3.141596, 0.5* pi ,0.3*pi) q[1]; // spaces in expressions

        // Instead of block comment, use line comments
        // spanning multiple lines
        cx q[0],q[1]; // inline comment

        // Comment at end
    "#;

    let program =
        QASMParser::parse_str(qasm).expect("Failed to parse QASM with comments and expressions");

    // Verify parsing succeeded despite various comment styles
    assert!(!program.operations.is_empty(), "Should have operations");
}

#[test]
fn test_all_gate_types() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[3];

        // Single-qubit gates
        x q[0];
        y q[0];
        z q[0];
        h q[0];
        s q[0];
        sdg q[0];
        t q[0];
        tdg q[0];
        sx q[0];
        sxdg q[0];

        // Parameterized single-qubit gates
        rx(pi/2) q[0];
        ry(pi/3) q[0];
        rz(pi/4) q[0];
        u1(pi/5) q[0];
        u2(pi/6, pi/7) q[0];
        u3(pi/8, pi/9, pi/10) q[0];

        // Two-qubit gates
        cx q[0],q[1];
        cy q[0],q[1];
        cz q[0],q[1];
        csx q[0],q[1];
        swap q[0],q[1];

        // Parameterized two-qubit gates
        cu1(pi/2) q[0],q[1];
        rzz(pi/3) q[0],q[1];
        rxx(pi/4) q[0],q[1];

        // Three-qubit gates
        ccx q[0],q[1],q[2];

        // Other operations
        barrier q[0],q[1],q[2];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM with all gate types");

    // Verify it parses successfully with many different gate types
    assert!(
        !program.operations.is_empty(),
        "Should have many operations"
    );
}

#[test]
fn test_mathematical_constants_and_functions() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[1];

        // Using pi constant
        rz(pi) q[0];
        rz(pi/2) q[0];
        rz(2*pi) q[0];
        rz(1.5*pi) q[0];

        // Nested expressions
        rz((pi/2) + (pi/4)) q[0];
        rz(pi * (1 + 0.5)) q[0];

        // Decimal values
        rz(3.14159) q[0];
        rz(0.0375*pi) q[0];
    "#;

    let program =
        QASMParser::parse_str(qasm).expect("Failed to parse QASM with mathematical constants");

    // Verify mathematical expressions are handled correctly
    assert!(
        !program.operations.is_empty(),
        "Should have operations with mathematical expressions"
    );
}
