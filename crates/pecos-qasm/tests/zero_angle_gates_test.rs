use pecos_qasm::{Operation, parser::QASMParser};

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
    // u(0,0,0) expands to: rz(0); rx(0); rz(0)
    // and rx(0) expands to: h; rz(0); h
    // So total: rz(0), rz(0), h, rz(0), h, rz(0)
    assert_eq!(program.operations.len(), 6);
    
    // Check that all RZ gates have angle 0
    for (i, op) in program.operations.iter().enumerate() {
        match op {
            Operation::Gate { name, parameters, .. } if name == "RZ" => {
                assert_eq!(parameters.len(), 1);
                assert_eq!(parameters[0], 0.0, "RZ angle at operation {} should be 0", i);
            }
            Operation::Gate { name, parameters, .. } if name == "H" => {
                assert!(parameters.is_empty(), "H gate should have no parameters");
            }
            _ => {}
        }
    }
}

#[test]
fn test_phase_gate_expansion() {
    // Test that p(0) expands to rz(0)
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        p(0) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse phase gate");
    
    // p(0) expands to rz(0)
    assert_eq!(program.operations.len(), 1);
    
    match &program.operations[0] {
        Operation::Gate { name, qubits, parameters } => {
            assert_eq!(name, "RZ");
            assert_eq!(qubits, &[0]);
            assert_eq!(parameters.len(), 1);
            assert_eq!(parameters[0], 0.0);
        }
        _ => panic!("Expected RZ gate"),
    }
}

#[test]
fn test_u_gate_expansion() {
    // Test that u(0,0,0) expands correctly
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        u(0,0,0) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse u gate");
    
    // u(0,0,0) expands to: rz(0); rx(0); rz(0)
    // and rx(0) expands to: h; rz(0); h
    // So final sequence: rz(0), h, rz(0), h, rz(0)
    assert_eq!(program.operations.len(), 5);
    
    let expected_gates = ["RZ", "H", "RZ", "H", "RZ"];
    for (i, op) in program.operations.iter().enumerate() {
        match op {
            Operation::Gate { name, parameters, .. } => {
                assert_eq!(name, expected_gates[i], "Gate at position {} should be {}", i, expected_gates[i]);
                if name == "RZ" {
                    assert_eq!(parameters.len(), 1);
                    assert_eq!(parameters[0], 0.0);
                } else if name == "H" {
                    assert!(parameters.is_empty());
                }
            }
            _ => panic!("Expected gate operation"),
        }
    }
}