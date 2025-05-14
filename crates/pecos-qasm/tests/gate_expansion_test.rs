use pecos_qasm::parser::QASMParser;
use pecos_qasm::parser::Operation;

#[test]
fn test_gate_expansion_rx() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        rx(1.5708) q[0];
    "#;
    
    let program = QASMParser::parse_str(qasm).unwrap();
    
    // The rx gate should be expanded to h; rz; h
    assert_eq!(program.operations.len(), 3);
    
    // Check first operation is h
    if let Operation::Gate { name, arguments, registers, .. } = &program.operations[0] {
        assert_eq!(name, "H");
        assert_eq!(arguments, &[0]);
        assert_eq!(registers, &["q"]);
    } else {
        panic!("Expected h gate");
    }
    
    // Check second operation is rz
    if let Operation::Gate { name, arguments, registers, parameters } = &program.operations[1] {
        assert_eq!(name, "RZ");
        assert_eq!(arguments, &[0]);
        assert_eq!(registers, &["q"]);
        assert_eq!(parameters.len(), 1);
        assert!((parameters[0] - 1.5708).abs() < 0.0001);
    } else {
        panic!("Expected rz gate");
    }
    
    // Check third operation is h
    if let Operation::Gate { name, arguments, registers, .. } = &program.operations[2] {
        assert_eq!(name, "H");
        assert_eq!(arguments, &[0]);
        assert_eq!(registers, &["q"]);
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
    
    // Check first operation is h on second qubit
    if let Operation::Gate { name, arguments, registers, .. } = &program.operations[0] {
        assert_eq!(name, "H");
        assert_eq!(arguments, &[1]);
        assert_eq!(registers, &["q"]);
    } else {
        panic!("Expected h gate");
    }
    
    // Check second operation is cx
    if let Operation::Gate { name, arguments, registers, .. } = &program.operations[1] {
        assert_eq!(name, "CX");
        assert_eq!(arguments, &[0, 1]);
        assert_eq!(registers, &["q", "q"]);
    } else {
        panic!("Expected cx gate");
    }
    
    // Check third operation is h on second qubit
    if let Operation::Gate { name, arguments, registers, .. } = &program.operations[2] {
        assert_eq!(name, "H");
        assert_eq!(arguments, &[1]);
        assert_eq!(registers, &["q"]);
    } else {
        panic!("Expected h gate");
    }
}

#[test]
fn test_gate_remains_native() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        h q[0];
        cx q[0], q[1];
    "#;
    
    let program = QASMParser::parse_str(qasm).unwrap();
    
    // Native gates should not be expanded
    assert_eq!(program.operations.len(), 2);
    
    // Check operations remain as-is
    if let Operation::Gate { name, .. } = &program.operations[0] {
        assert_eq!(name, "H");
    } else {
        panic!("Expected h gate");
    }
    
    if let Operation::Gate { name, .. } = &program.operations[1] {
        assert_eq!(name, "cx");
    } else {
        panic!("Expected cx gate");
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
    
    // Check that common gates are defined
    assert!(program.gate_definitions.contains_key("rx"));
    assert!(program.gate_definitions.contains_key("cz"));
    assert!(program.gate_definitions.contains_key("s"));
    assert!(program.gate_definitions.contains_key("t"));
    
    // Check a gate definition structure
    let rx_def = &program.gate_definitions["rx"];
    assert_eq!(rx_def.name, "rx");
    assert_eq!(rx_def.params, vec!["theta"]);
    assert_eq!(rx_def.qargs, vec!["a"]);
}