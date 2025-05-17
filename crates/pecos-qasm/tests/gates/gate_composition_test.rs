use pecos_qasm::QASMParser;

#[test]
fn test_gate_composition() {
    let qasm = r"
        OPENQASM 2.0;
        qreg q[3];
        creg c[3];

        // Define a bell pair gate using basic gates
        gate bell a, b {
            H a;
            CX a, b;
        }

        // Define a more complex gate using the bell gate
        gate bell_with_phase(theta) a, b {
            bell a, b;
            RZ(theta) a;
            RZ(theta) b;
        }

        // Define an even more complex gate using previous definitions
        gate bell_swap c1, c2, target {
            bell c1, target;
            bell_with_phase(pi/2) c2, target;
            CX c1, c2;
            H target;
        }

        // Use the composed gates
        bell_swap q[0], q[1], q[2];

        measure q -> c;
    ";

    let result = QASMParser::parse_str_raw(qasm);

    match result {
        Ok(program) => {
            println!("Successfully parsed program with composed gates");

            // The operations should be fully expanded
            for (i, op) in program.operations.iter().enumerate() {
                println!("Operation {i}: {op:?}");
            }

            // Count the expanded operations
            let gate_count = program
                .operations
                .iter()
                .filter(|op| matches!(op, pecos_qasm::Operation::Gate { .. }))
                .count();

            // bell_swap should expand to many basic gates
            assert!(
                gate_count > 5,
                "Expected many gates after expansion, got {gate_count}"
            );
        }
        Err(e) => {
            panic!("Failed to parse gate composition: {e}");
        }
    }
}

// Circular dependency tests moved to circular_dependency_test.rs
// to better handle stack overflow testing

#[test]
fn test_undefined_gate_in_definition() {
    let qasm = r"
        OPENQASM 2.0;
        qreg q[2];

        // Define a gate using an undefined gate
        gate mygate a {
            undefined_gate a;
        }

        mygate q[0];
    ";

    let result = QASMParser::parse_str_raw(qasm);

    match result {
        Ok(program) => {
            // The undefined gate should remain in the expanded operations
            let has_undefined = program.operations.iter().any(|op| {
                if let pecos_qasm::Operation::Gate { name, .. } = op {
                    name == "undefined_gate"
                } else {
                    false
                }
            });

            assert!(
                has_undefined,
                "Expected undefined_gate to remain in operations"
            );
        }
        Err(e) => {
            println!("Got error: {e}");
        }
    }
}
