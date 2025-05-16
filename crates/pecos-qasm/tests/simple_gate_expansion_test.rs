use pecos_qasm::Operation;
use pecos_qasm::parser::QASMParser;

#[test]
fn test_simple_gate_definition() {
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
fn test_native_gate_parsing() {
    let qasm = r"
        OPENQASM 2.0;
        qreg q[1];
        
        gate H a { RZ(0) a; }
        
        H q[0];
    ";

    let program = QASMParser::parse_str_raw(qasm).unwrap();

    // H gate definition should be loaded
    assert!(program.gate_definitions.contains_key("H"));

    // The H operation should be expanded to its definition
    assert_eq!(program.operations.len(), 1);

    if let Operation::Gate { name, .. } = &program.operations[0] {
        assert_eq!(name, "RZ");
    } else {
        panic!("Expected gate operation");
    }
}
