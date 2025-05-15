use pecos_qasm::parser::Operation;
use pecos_qasm::parser::QASMParser;

#[test]
fn test_sx_gates_expansion() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        //test SX, SXdg, CSX gates
        qreg q[2];
        sx q[0];
        x q[1];
        sxdg q[1];
        csx q[0],q[1];
    "#;

    let program = QASMParser::parse_str(qasm).unwrap();

    // sx expands to: sdg, h, sdg (3 operations)
    // x is native (1 operation)
    // sxdg expands to: s, h, s (3 operations)
    // csx is not defined in qelib1.inc, so it remains as-is (1 operation)
    // Total: 3 + 1 + 3 + 1 = 8 operations
    assert_eq!(program.operations.len(), 8);

    // Check that sx is expanded to sdg, h, sdg
    if let Operation::Gate { name, .. } = &program.operations[0] {
        assert_eq!(name, "RZ"); // sdg is RZ(-pi/2)
    }
    if let Operation::Gate { name, .. } = &program.operations[1] {
        assert_eq!(name, "H");
    }
    if let Operation::Gate { name, .. } = &program.operations[2] {
        assert_eq!(name, "RZ"); // sdg is RZ(-pi/2)
    }

    // Check x gate
    if let Operation::Gate { name, .. } = &program.operations[3] {
        assert_eq!(name, "X");
    }

    // Check that sxdg is expanded to s, h, s
    if let Operation::Gate { name, .. } = &program.operations[4] {
        assert_eq!(name, "RZ"); // s is RZ(pi/2)
    }
    if let Operation::Gate { name, .. } = &program.operations[5] {
        assert_eq!(name, "H");
    }
    if let Operation::Gate { name, .. } = &program.operations[6] {
        assert_eq!(name, "RZ"); // s is RZ(pi/2)
    }

    // Check csx gate (not expanded)
    if let Operation::Gate { name, .. } = &program.operations[7] {
        assert_eq!(name, "csx");
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
