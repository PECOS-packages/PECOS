use pecos_qasm::Operation;
use pecos_qasm::parser::QASMParser;

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

    if let Operation::Gate { name, .. } = &program.operations[0] {
        assert_eq!(name, "H");
    } else {
        panic!("Expected gate operation");
    }
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

    if let Operation::Gate { name, .. } = &program.operations[0] {
        assert_eq!(name, "H");
    } else {
        panic!("Expected gate operation");
    }
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
    if let Operation::Gate { name, qubits, .. } = &program.operations[0] {
        assert_eq!(name, "H");
        assert_eq!(qubits, &[0]);
    } else {
        panic!("Expected h gate");
    }

    // Check second operation is rz
    if let Operation::Gate {
        name,
        qubits,
        parameters,
        ..
    } = &program.operations[1]
    {
        assert_eq!(name, "RZ");
        assert_eq!(qubits, &[0]);
        assert_eq!(parameters.len(), 1);
        assert!(
            (parameters[0] - std::f64::consts::FRAC_PI_2).abs() < 1e-6,
            "Expected parameter PI/2, got {}",
            parameters[0]
        );
    } else {
        panic!("Expected rz gate");
    }

    // Check third operation is h
    if let Operation::Gate { name, qubits, .. } = &program.operations[2] {
        assert_eq!(name, "H");
        assert_eq!(qubits, &[0]);
    } else {
        panic!("Expected h gate");
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
    if let Operation::Gate { name, qubits, .. } = &program.operations[0] {
        assert_eq!(name, "H");
        assert_eq!(qubits, &[1]);
    } else {
        panic!("Expected h gate");
    }

    // Check second operation is cx
    if let Operation::Gate { name, qubits, .. } = &program.operations[1] {
        assert_eq!(name, "CX");
        assert_eq!(qubits, &[0, 1]);
    } else {
        panic!("Expected cx gate");
    }

    // Check third operation is h
    if let Operation::Gate { name, qubits, .. } = &program.operations[2] {
        assert_eq!(name, "H");
        assert_eq!(qubits, &[1]);
    } else {
        panic!("Expected h gate");
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
