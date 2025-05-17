use pecos_engines::engines::classical::ClassicalEngine;
use pecos_qasm::engine::QASMEngine;
use pecos_qasm::{Operation, QASMParser};
use std::str::FromStr;

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
    let mut engine = QASMEngine::from_str(qasm).expect("Failed to load program");

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
                    println!("Gate after expansion: {name}");
                    // u(0,0,0) might remain as U or be expanded
                    // depending on implementation
                }
                _ => panic!("Expected a gate operation"),
            }
        }
    } else {
        // If expanded, check we have the expected operations
        println!(
            "Gate was expanded into {} operations",
            program.operations.len()
        );
    }
}

#[test]
fn test_p_gate_expansion() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        p(0) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse phase gate");

    // p(0) expands to rz(0)
    assert_eq!(program.operations.len(), 1);

    if let Operation::Gate {
        name, parameters, ..
    } = &program.operations[0]
    {
        assert_eq!(name, "RZ");
        assert_eq!(parameters.len(), 1);
        assert!(
            (parameters[0] - 0.0).abs() < f64::EPSILON,
            "RZ angle should be 0"
        );
    } else {
        panic!("Expected RZ gate");
    }
}

#[test]
fn test_u_gate_expansion() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u(0,0,0) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse u gate");

    // u(0,0,0) now maps directly to native U gate
    assert_eq!(program.operations.len(), 1);

    if let Operation::Gate {
        name, parameters, ..
    } = &program.operations[0]
    {
        assert_eq!(name, "U");
        assert_eq!(parameters.len(), 3);
        assert!(
            (parameters[0] - 0.0).abs() < f64::EPSILON,
            "U theta parameter should be 0"
        );
        assert!(
            (parameters[1] - 0.0).abs() < f64::EPSILON,
            "U phi parameter should be 0"
        );
        assert!(
            (parameters[2] - 0.0).abs() < f64::EPSILON,
            "U lambda parameter should be 0"
        );
    } else {
        panic!("Expected U gate");
    }
}

#[test]
fn test_identity_operations() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        u(0,0,0) q[0];  // Identity
        p(0) q[1];      // Phase 0 is also identity
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse identity operations");

    println!("Identity operations parsed: {}", program.operations.len());

    // Both operations are identity operations
    for op in &program.operations {
        if let Operation::Gate {
            name, parameters, ..
        } = op
        {
            match name.as_str() {
                "U" => {
                    assert_eq!(parameters.len(), 3);
                    assert!((parameters[0] - 0.0).abs() < f64::EPSILON);
                    assert!((parameters[1] - 0.0).abs() < f64::EPSILON);
                    assert!((parameters[2] - 0.0).abs() < f64::EPSILON);
                }
                "RZ" => {
                    assert_eq!(parameters.len(), 1);
                    assert!((parameters[0] - 0.0).abs() < f64::EPSILON);
                }
                _ => {}
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

    // Parse to load gate definitions
    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // Check that p gate is defined
    assert!(
        program.gate_definitions.contains_key("p"),
        "p gate should be defined"
    );

    // Check u gate is defined
    assert!(
        program.gate_definitions.contains_key("u"),
        "u gate should be defined"
    );
}

#[test]
fn test_zero_angle_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        p(0) q[0];
        u(0,0,0) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse zero angle gates");

    // p(0) expands to rz(0)
    // u(0,0,0) now maps directly to native U gate
    // So total: rz(0), U(0,0,0)
    assert_eq!(program.operations.len(), 2);

    // Check that we have the expected gates
    for (i, op) in program.operations.iter().enumerate() {
        match op {
            Operation::Gate {
                name, parameters, ..
            } if name == "RZ" => {
                assert_eq!(parameters.len(), 1);
                assert!(
                    (parameters[0] - 0.0).abs() < f64::EPSILON,
                    "RZ angle at operation {i} should be 0"
                );
            }
            Operation::Gate {
                name, parameters, ..
            } if name == "U" => {
                assert_eq!(parameters.len(), 3);
                assert!(
                    (parameters[0] - 0.0).abs() < f64::EPSILON,
                    "U theta parameter should be 0"
                );
                assert!(
                    (parameters[1] - 0.0).abs() < f64::EPSILON,
                    "U phi parameter should be 0"
                );
                assert!(
                    (parameters[2] - 0.0).abs() < f64::EPSILON,
                    "U lambda parameter should be 0"
                );
            }
            _ => {}
        }
    }
}

#[test]
fn test_u_gate_is_native() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u(1.5708, 0, 3.14159) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // U gate should remain as U (not expanded) since it's native
    assert_eq!(program.operations.len(), 1);

    if let Operation::Gate { name, .. } = &program.operations[0] {
        assert_eq!(name, "U");
    } else {
        panic!("Expected U gate operation");
    }
}
