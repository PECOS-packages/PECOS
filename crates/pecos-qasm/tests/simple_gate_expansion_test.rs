use pecos_qasm::parser::Operation;
use pecos_qasm::parser::QASMParser;

#[test]
fn test_simple_gate_definition() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        gate mygate a { h a; }
        
        mygate q[0];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    // Gate definition should be loaded
    assert!(program.gate_definitions.contains_key("mygate"));

    // The mygate operation should be expanded to h
    assert_eq!(program.operations.len(), 1);

    if let Operation::Gate { name, .. } = &program.operations[0] {
        assert_eq!(name, "h");
    } else {
        panic!("Expected gate operation");
    }
}

#[test]
fn test_native_gate_parsing() {
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        gate h a { rz(0) a; }
        
        h q[0];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    // h gate definition should be loaded
    assert!(program.gate_definitions.contains_key("h"));

    // The h operation should be expanded to its definition
    assert_eq!(program.operations.len(), 1);

    if let Operation::Gate { name, .. } = &program.operations[0] {
        assert_eq!(name, "rz");
    } else {
        panic!("Expected gate operation");
    }
}
