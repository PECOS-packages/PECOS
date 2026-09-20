use pecos_qasm::QASMParser;

#[test]
fn surplus_gate_parameters_are_rejected() {
    for operation in [
        "rz(0.1,0.2) q[0];",
        "u(0.1,0.2,0.3,0.4) q[0];",
        "u(a,b,c,d) q[0];",
        "gate probe(a,b,c,d) x { u(a,b,c,d) x; } probe(0.1,0.2,0.3,0.4) q[0];",
    ] {
        let source = format!("OPENQASM 2.0; include \"qelib1.inc\"; qreg q[1]; {operation}");
        let error =
            QASMParser::parse_str(&source).expect_err("surplus parameters must not be dropped");
        if operation == "u(a,b,c,d) q[0];" {
            assert!(error.to_string().contains("Cannot evaluate variable 'a'"));
        } else {
            assert!(error.to_string().contains("parameters but got"));
        }
    }
}
