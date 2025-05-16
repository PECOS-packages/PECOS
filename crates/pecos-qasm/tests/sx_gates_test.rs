use pecos_qasm::Operation;
use pecos_qasm::parser::QASMParser;

#[test]
fn test_sx_gates_expansion() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        //test SX, SXdg, CSX gates
        qreg q[2];
        sx q[0];
        X q[1];
        sxdg q[1];
        csx q[0],q[1];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    // After all expansions, we'll have a specific set of native operations
    // sx -> RZ(-pi/2), H, RZ(-pi/2)
    // x -> X (native)
    // sxdg -> RZ(pi/2), H, RZ(pi/2)
    // csx -> CX (in our simplified implementation)
    // Total operations will be the expanded native gates
    assert!(!program.operations.is_empty());

    // Verify all operations are valid gates
    for op in &program.operations {
        assert!(matches!(op, Operation::Gate { .. }));
    }
}

#[test]
fn test_sx_gate_parameters() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        sx q[0];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    // sx expands to: sdg, h, sdg
    assert_eq!(program.operations.len(), 3);

    // Check first sdg gate has correct parameter
    if let Operation::Gate {
        name, parameters, ..
    } = &program.operations[0]
    {
        assert_eq!(name, "RZ");
        assert_eq!(parameters.len(), 1);
        assert!((parameters[0] + std::f64::consts::PI / 2.0).abs() < 0.0001); // -pi/2
    }

    // Check h gate
    if let Operation::Gate {
        name, parameters, ..
    } = &program.operations[1]
    {
        assert_eq!(name, "H");
        assert!(parameters.is_empty());
    }

    // Check second sdg gate has correct parameter
    if let Operation::Gate {
        name, parameters, ..
    } = &program.operations[2]
    {
        assert_eq!(name, "RZ");
        assert_eq!(parameters.len(), 1);
        assert!((parameters[0] + std::f64::consts::PI / 2.0).abs() < 0.0001); // -pi/2
    }
}

#[test]
fn test_sxdg_gate_parameters() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        sxdg q[0];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    // sxdg expands to: s, h, s
    assert_eq!(program.operations.len(), 3);

    // Check first s gate has correct parameter
    if let Operation::Gate {
        name, parameters, ..
    } = &program.operations[0]
    {
        assert_eq!(name, "RZ");
        assert_eq!(parameters.len(), 1);
        assert!((parameters[0] - std::f64::consts::PI / 2.0).abs() < 0.0001); // pi/2
    }

    // Check h gate
    if let Operation::Gate {
        name, parameters, ..
    } = &program.operations[1]
    {
        assert_eq!(name, "H");
        assert!(parameters.is_empty());
    }

    // Check second s gate has correct parameter
    if let Operation::Gate {
        name, parameters, ..
    } = &program.operations[2]
    {
        assert_eq!(name, "RZ");
        assert_eq!(parameters.len(), 1);
        assert!((parameters[0] - std::f64::consts::PI / 2.0).abs() < 0.0001); // pi/2
    }
}
