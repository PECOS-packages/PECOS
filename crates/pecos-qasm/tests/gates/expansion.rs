use pecos_core::prelude::GateType;
use pecos_qasm::Operation;
use pecos_qasm::parser::QASMParser;

// Helper function to check if an operation is a specific gate
fn is_gate_with_name(op: &Operation, gate_name: &str) -> bool {
    match op {
        Operation::Gate { name, .. } => name == gate_name,
        Operation::NativeGate(gate) => match gate_name {
            "H" => matches!(gate.gate_type, GateType::H),
            "X" => matches!(gate.gate_type, GateType::X),
            "CX" => matches!(gate.gate_type, GateType::CX),
            _ => false,
        },
        _ => false,
    }
}

#[test]
fn test_gate_expansion_basic() {
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];

        gate mygate a { H a; }

        mygate q[0];
    ";

    let program = QASMParser::parse_str_raw(qasm).unwrap();

    // Gate definition should be loaded
    assert!(program.gate_definitions.contains_key("mygate"));

    // The mygate operation should be expanded to H
    assert_eq!(program.operations.len(), 1);

    assert!(
        is_gate_with_name(&program.operations[0], "H"),
        "Expected H gate"
    );
}

#[test]
fn test_gate_expansion_native_gate() {
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];
        H q[0];
    ";

    let program = QASMParser::parse_str_raw(qasm).unwrap();

    // Native gate should not be expanded
    assert_eq!(program.operations.len(), 1);

    assert!(
        is_gate_with_name(&program.operations[0], "H"),
        "Expected H gate"
    );
}

#[test]
fn test_gate_expansion_rx() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(pi/2) q[0];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    // The rx gate should be expanded to h; rz; h
    assert_eq!(program.operations.len(), 3);

    // Check first operation is h
    match &program.operations[0] {
        Operation::Gate { name, qubits, .. } => {
            assert_eq!(name, "H");
            assert_eq!(qubits, &[0]);
        }
        Operation::NativeGate(gate) => {
            assert_eq!(gate.gate_type, pecos_core::gate_type::GateType::H);
            assert_eq!(gate.qubits.len(), 1);
            assert_eq!({ gate.qubits[0].0 }, 0);
        }
        _ => panic!("Expected h gate"),
    }

    // Check second operation is rz
    match &program.operations[1] {
        Operation::Gate {
            name,
            qubits,
            parameters,
            ..
        } => {
            assert_eq!(name, "RZ");
            assert_eq!(qubits, &[0]);
            assert_eq!(parameters.len(), 1);
            assert!(
                (parameters[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-6,
                "Expected parameter PI/2, got {}",
                parameters[0]
            );
        }
        Operation::NativeGate(gate) => {
            assert_eq!(gate.gate_type, pecos_core::gate_type::GateType::RZ);
            assert_eq!(gate.qubits.len(), 1);
            assert_eq!({ gate.qubits[0].0 }, 0);
            // Rotation gate angles are now stored in gate.angles as Angle64
            assert_eq!(gate.angles.len(), 1);
            assert!(
                (gate.angles[0].to_radians() - std::f64::consts::FRAC_PI_2).abs() < 1e-6,
                "Expected angle PI/2, got {}",
                gate.angles[0].to_radians()
            );
        }
        _ => panic!("Expected rz gate"),
    }

    // Check third operation is h
    match &program.operations[2] {
        Operation::Gate { name, qubits, .. } => {
            assert_eq!(name, "H");
            assert_eq!(qubits, &[0]);
        }
        Operation::NativeGate(gate) => {
            assert_eq!(gate.gate_type, pecos_core::gate_type::GateType::H);
            assert_eq!(gate.qubits.len(), 1);
            assert_eq!({ gate.qubits[0].0 }, 0);
        }
        _ => panic!("Expected h gate"),
    }
}

#[test]
fn test_gate_expansion_cz() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        cz q[0], q[1];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    // The cz gate should be expanded to h; cx; h
    assert_eq!(program.operations.len(), 3);

    // Check first operation is h
    match &program.operations[0] {
        Operation::Gate { name, qubits, .. } => {
            assert_eq!(name, "H");
            assert_eq!(qubits, &[1]);
        }
        Operation::NativeGate(gate) => {
            assert_eq!(gate.gate_type, pecos_core::gate_type::GateType::H);
            assert_eq!(gate.qubits.len(), 1);
            assert_eq!({ gate.qubits[0].0 }, 1);
        }
        _ => panic!("Expected h gate"),
    }

    // Check second operation is cx
    match &program.operations[1] {
        Operation::Gate { name, qubits, .. } => {
            assert_eq!(name, "CX");
            assert_eq!(qubits, &[0, 1]);
        }
        Operation::NativeGate(gate) => {
            assert_eq!(gate.gate_type, pecos_core::gate_type::GateType::CX);
            assert_eq!(gate.qubits.len(), 2);
            assert_eq!({ gate.qubits[0].0 }, 0);
            assert_eq!({ gate.qubits[1].0 }, 1);
        }
        _ => panic!("Expected cx gate"),
    }

    // Check third operation is h
    match &program.operations[2] {
        Operation::Gate { name, qubits, .. } => {
            assert_eq!(name, "H");
            assert_eq!(qubits, &[1]);
        }
        Operation::NativeGate(gate) => {
            assert_eq!(gate.gate_type, pecos_core::gate_type::GateType::H);
            assert_eq!(gate.qubits.len(), 1);
            assert_eq!({ gate.qubits[0].0 }, 1);
        }
        _ => panic!("Expected h gate"),
    }
}

#[test]
fn test_gate_definitions_loaded() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    // Check a known qelib1 gate exists in the definitions
    assert!(program.gate_definitions.contains_key("cx"));
    assert!(program.gate_definitions.contains_key("h"));
    assert!(program.gate_definitions.contains_key("x"));
    assert!(program.gate_definitions.contains_key("y"));
    assert!(program.gate_definitions.contains_key("z"));
}
