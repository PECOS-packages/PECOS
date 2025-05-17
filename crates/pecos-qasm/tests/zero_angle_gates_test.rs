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
    // u(0,0,0) now maps directly to native U gate
    // So total: rz(0), U(0,0,0)
    assert_eq!(program.operations.len(), 2);
    
    // Check that we have the expected gates
    for (i, op) in program.operations.iter().enumerate() {
        match op {
            Operation::Gate { name, parameters, .. } if name == "RZ" => {
                assert_eq!(parameters.len(), 1);
                assert_eq!(parameters[0], 0.0, "RZ angle at operation {} should be 0", i);
            }
            Operation::Gate { name, parameters, .. } if name == "U" => {
                assert_eq!(parameters.len(), 3);
                assert_eq!(parameters[0], 0.0, "U theta parameter should be 0");
                assert_eq!(parameters[1], 0.0, "U phi parameter should be 0");
                assert_eq!(parameters[2], 0.0, "U lambda parameter should be 0");
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
    
    // u(0,0,0) now maps directly to native U gate
    assert_eq!(program.operations.len(), 1);
    
    // Check that the single operation is a U gate
    match &program.operations[0] {
        Operation::Gate { name, parameters, qubits } => {
            assert_eq!(name, "U");
            assert_eq!(parameters.len(), 3);
            assert_eq!(parameters[0], 0.0, "U theta parameter should be 0");
            assert_eq!(parameters[1], 0.0, "U phi parameter should be 0");
            assert_eq!(parameters[2], 0.0, "U lambda parameter should be 0");
            assert_eq!(qubits, &[0]);
        }
        _ => panic!("Expected U gate operation"),
    }
}