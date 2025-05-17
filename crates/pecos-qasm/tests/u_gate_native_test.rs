use pecos_qasm::{QASMParser, Operation};

#[test]
fn test_u_gate_is_native() {
    // Test that U gate is treated as a native gate and not expanded
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[2];
        u(0.5*pi, 0.25*pi, 1*pi) q[0];
        U(0.5*pi, 0.25*pi, 1*pi) q[1];  // Test uppercase native U
    "#;

    let result = QASMParser::parse_str(qasm);
    
    match result {
        Ok(program) => {
            println!("Operations:");
            for (i, op) in program.operations.iter().enumerate() {
                match op {
                    Operation::Gate { name, qubits, parameters } => {
                        println!("  [{}] Gate: {} on qubits {:?} with params {:?}", i, name, qubits, parameters);
                    }
                    _ => {}
                }
            }
            
            let u_count = program.operations.iter()
                .filter(|op| matches!(op, Operation::Gate { name, .. } if name == "u" || name == "U"))
                .count();
            
            println!("U count: {}", u_count);
            
            // We expect 2 U gates (one from lowercase u, one from uppercase U)
            assert_eq!(u_count, 2, "Expected 2 U gates");
            
            // Verify no other gates were generated (like rz or rx from expansion)
            let other_gates = program.operations.iter()
                .filter(|op| matches!(op, Operation::Gate { name, .. } if name != "u" && name != "U"))
                .count();
                
            assert_eq!(other_gates, 0, "Expected no other gates from U expansion");
        }
        Err(e) => {
            panic!("Failed to parse circuit: {}", e);
        }
    }
}