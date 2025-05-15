use pecos_qasm::QASMParser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Comprehensive example of gate definitions in QASM
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        
        qreg q[5];
        creg c[5];
        
        // 1. Simple gate definition (no parameters)
        gate bell a, b {
            h a;
            cx a, b;
        }
        
        // 2. Gate with parameters
        gate rot_both(theta) q1, q2 {
            rx(theta) q1;
            ry(theta) q2;
        }
        
        // 3. Gate using previously defined gates
        gate bell_phase(phi) a, b {
            bell a, b;
            cphase(phi) a, b;
        }
        
        // 4. Gate with multiple parameters
        gate custom_u(alpha, beta, gamma) q {
            rz(alpha) q;
            ry(beta) q;
            rz(gamma) q;
        }
        
        // 5. Complex gate with multiple qubits
        gate w_state a, b, c {
            h a;
            // Create equal superposition
            cx a, b;
            cx a, c;
            // Adjust phases
            cphase(2*pi/3) a, b;
            cphase(2*pi/3) b, c;
        }
        
        // 6. Gate that redefines a library gate
        gate my_hadamard q {
            rz(pi) q;
            sx q;
            rz(pi) q;
        }
        
        // Use all our custom gates
        bell q[0], q[1];
        rot_both(pi/4) q[1], q[2];
        bell_phase(pi/3) q[2], q[3];
        custom_u(pi/4, pi/2, 3*pi/4) q[3];
        w_state q[0], q[1], q[2];
        my_hadamard q[4];
        
        // Standard gates still work
        h q[4];
        
        measure q -> c;
    "#;

    let program = QASMParser::parse_str(qasm)?;

    println!("Gate Definition Examples");
    println!("=======================\n");

    // List all custom gate definitions
    println!("Custom gate definitions found:");
    let mut custom_gates: Vec<_> = program
        .gate_definitions
        .keys()
        .filter(|name| {
            ![
                "h", "cx", "rx", "ry", "rz", "cphase", "sx", "x", "y", "z", "s", "t",
            ]
            .contains(&name.as_str())
        })
        .collect();
    custom_gates.sort();

    for gate_name in &custom_gates {
        let gate_def = &program.gate_definitions[*gate_name];
        print!("  - {}", gate_name);
        if !gate_def.params.is_empty() {
            print!("(");
            for (i, param) in gate_def.params.iter().enumerate() {
                if i > 0 {
                    print!(", ");
                }
                print!("{}", param);
            }
            print!(")");
        }
        print!(" ");
        for (i, qarg) in gate_def.qargs.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{}", qarg);
        }
        println!(" {{ ... }}");
    }

    println!(
        "\nExpanded operations ({} total):",
        program.operations.len()
    );
    for (i, op) in program.operations.iter().take(10).enumerate() {
        match op {
            pecos_qasm::parser::Operation::Gate {
                name,
                qubits,
                parameters,
            } => {
                print!("  {}: {} ", i, name);
                if !parameters.is_empty() {
                    print!("(");
                    for (j, p) in parameters.iter().enumerate() {
                        if j > 0 {
                            print!(", ");
                        }
                        print!("{:.4}", p);
                    }
                    print!(") ");
                }
                println!("q{:?}", qubits);
            }
            _ => {}
        }
    }
    if program.operations.len() > 10 {
        println!("  ... ({} more operations)", program.operations.len() - 10);
    }

    Ok(())
}
