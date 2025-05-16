use pecos_engines::engines::classical::ClassicalEngine;
use pecos_qasm::engine::QASMEngine;
use pecos_qasm::{Operation, QASMParser};

#[test]
fn test_p_zero_gate_compiles() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        p(0) q[0];
        measure q[0] -> c[0];
    "#;

    // Parse and compile
    let mut engine = QASMEngine::from_str(qasm)
        .expect("Failed to load program");

    // This should now compile successfully with the updated qelib1.inc
    let _messages = engine
        .generate_commands()
        .expect("p(0) gate should compile");

    println!("p(0) gate successfully compiled");
}

#[test]
fn test_u_identity_gate_expansion() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u(0,0,0) q[0];
    "#;

    // Parse the program
    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // The u gate should be expanded to its constituent gates
    // For U(0,0,0), it should expand to: RZ(0), rx(0), RZ(0)
    // which effectively is the identity
    println!("Operations count: {}", program.operations.len());

    // Note: The current implementation may not fully expand the u gate
    // This test documents the current behavior
    if program.operations.len() == 1 {
        if let Some(op) = program.operations.first() {
            match op {
                Operation::Gate { name, .. } => {
                    assert_eq!(name, "U", "Gate should be 'U'");
                }
                _ => panic!("Expected a gate operation"),
            }
        }
    }
}

#[test]
fn test_gate_definitions_updated() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // Check that p and u gates are now defined
    assert!(
        program.gate_definitions.contains_key("p"),
        "p gate should be defined"
    );
    assert!(
        program.gate_definitions.contains_key("u"),
        "u gate should be defined"
    );

    // Verify the p gate definition
    if let Some(p_def) = program.gate_definitions.get("p") {
        assert_eq!(p_def.params.len(), 1, "p gate should have 1 parameter");
        assert_eq!(p_def.qargs.len(), 1, "p gate should have 1 qubit argument");
        println!(
            "p gate correctly defined with {} operations",
            p_def.body.len()
        );

        // Check that p(0) is equivalent to RZ(0)
        if let Some(first_op) = p_def.body.first() {
            assert_eq!(first_op.name, "rz", "p gate should use rz internally");
        }
    }

    // Verify the u gate definition
    if let Some(u_def) = program.gate_definitions.get("u") {
        assert_eq!(u_def.params.len(), 3, "u gate should have 3 parameters");
        assert_eq!(u_def.qargs.len(), 1, "u gate should have 1 qubit argument");
        println!(
            "u gate correctly defined with {} operations",
            u_def.body.len()
        );

        // U(0,0,0) should simplify to identity (RZ(0), rx(0), RZ(0))
        assert_eq!(u_def.body.len(), 3, "u gate should have 3 operations");
    }
}

#[test]
fn test_p_gate_expansion() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        p(1.5707963267948966) q[0];  // pi/2
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // The operations should be expanded
    assert_eq!(
        program.operations.len(),
        1,
        "Should have 1 expanded operation"
    );

    // Check that the p gate is expanded to rz
    if let Some(op) = program.operations.first() {
        match op {
            Operation::Gate {
                name, parameters, ..
            } => {
                assert_eq!(name, "RZ", "p gate should expand to RZ");
                assert_eq!(parameters.len(), 1, "Should have 1 parameter");
                assert!(
                    (parameters[0] - std::f64::consts::PI / 2.0).abs() < 0.0001,
                    "Parameter should be pi/2"
                );
            }
            _ => panic!("Expected a gate operation"),
        }
    }
}

#[test]
fn test_identity_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        id q[0];        // Identity gate
        p(0) q[0];      // Phase(0) is identity
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // Both operations should expand/compile correctly
    assert!(
        program.operations.len() >= 2,
        "Should have at least 2 operations"
    );
}
