use pecos_qasm::QASMParser;

#[test]
fn test_gate_empty_param_list() {
    // Test that gates can be defined with empty parentheses
    let qasm = r"
        OPENQASM 2.0;

        // Gate without parameters (no parentheses)
        gate mygate1 a {
            X a;
        }

        // Gate with empty parentheses - should also work
        gate mygate2() a {
            Y a;
        }

        // Gate with parameters
        gate mygate3(theta) a {
            RZ(theta) a;
        }

        qreg q[1];
        mygate1 q[0];
        mygate2 q[0];
        mygate3(pi/2) q[0];
    ";

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM with empty param list");

    // Verify all three gate definitions were parsed
    assert!(program.gate_definitions.contains_key("mygate1"));
    assert!(program.gate_definitions.contains_key("mygate2"));
    assert!(program.gate_definitions.contains_key("mygate3"));

    // Verify the gates have the correct parameter counts
    assert_eq!(program.gate_definitions["mygate1"].params.len(), 0);
    assert_eq!(program.gate_definitions["mygate2"].params.len(), 0);
    assert_eq!(program.gate_definitions["mygate3"].params.len(), 1);
}

#[test]
fn test_opaque_empty_param_list() {
    // Test that opaque declarations can also have empty parentheses
    let qasm = r"
        OPENQASM 2.0;

        // Opaque gate without parameters (no parentheses)
        opaque myopaque1 a;

        // Opaque gate with empty parentheses
        opaque myopaque2() a;

        // Opaque gate with parameters
        opaque myopaque3(theta) a;

        qreg q[1];
    ";

    let _program =
        QASMParser::parse_str(qasm).expect("Failed to parse QASM with opaque empty param list");
}
