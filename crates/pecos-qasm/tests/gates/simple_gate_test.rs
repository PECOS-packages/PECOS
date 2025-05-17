use pecos_qasm::{QASMParser, Operation};

#[test]
fn test_simple_gates() {
    // Test simple circuit with cx and u gates
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";

        qreg q[3];
        cx q[0],q[1];
        u(0, 0, 1*pi) q[0];
        cz q[1],q[2];  // This should expand to h-cx-h
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
            
            let cx_count = program.operations.iter()
                .filter(|op| matches!(op, Operation::Gate { name, .. } if name == "cx" || name == "CX"))
                .count();
            
            let u_count = program.operations.iter()
                .filter(|op| matches!(op, Operation::Gate { name, .. } if name == "u" || name == "U"))
                .count();
            
            println!("CX count: {}, U count: {}", cx_count, u_count);
            
            // We expect 2 cx (1 original + 1 from cz expansion) and 1 u gate
            assert_eq!(cx_count, 2, "Expected 2 CX gates (1 original + 1 from cz expansion)");
            assert_eq!(u_count, 1, "Expected 1 U gate");
        }
        Err(e) => {
            panic!("Failed to parse circuit: {}", e);
        }
    }
}