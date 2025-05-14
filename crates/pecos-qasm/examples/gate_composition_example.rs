use pecos_qasm::QASMParser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example showing gate composition and how includes work
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        
        qreg q[3];
        creg c[3];
        
        // Define custom gates using gates from qelib1.inc
        
        // A bell state preparation gate
        gate bell a, b {
            h a;
            cx a, b;
        }
        
        // A W-state preparation using composition
        gate w_state a, b, c {
            // First create superposition
            h a;
            
            // Create entanglement
            cx a, b;
            cx a, c;
            
            // Apply phase corrections
            rz(pi/3) a;
            rz(pi/3) b;
            rz(pi/3) c;
        }
        
        // More complex gate using previous definitions
        gate teleport_prep sender, channel, receiver {
            // Create bell pair between channel and receiver
            bell channel, receiver;
            
            // Prepare sender qubit in superposition
            h sender;
            rz(pi/4) sender;
            
            // Entangle sender with channel
            cx sender, channel;
            h sender;
        }
        
        // Use the composed gates
        teleport_prep q[0], q[1], q[2];
        
        // Measure all qubits
        measure q -> c;
    "#;
    
    let program = QASMParser::parse_str(qasm)?;
    
    println!("Gate Composition Example");
    println!("=======================\n");
    
    // Show gate definitions
    println!("Custom gate definitions:");
    for (name, _) in &program.gate_definitions {
        // Skip qelib1 gates
        if !["h", "cx", "rz", "x", "y", "z", "s", "t", "rx", "ry"].contains(&name.as_str()) {
            println!("  - {}", name);
        }
    }
    
    println!("\nExpanded operations:");
    for (i, op) in program.operations.iter().enumerate() {
        match op {
            pecos_qasm::parser::Operation::Gate { name, qubits, parameters } => {
                print!("  {}: {} ", i, name);
                if !parameters.is_empty() {
                    print!("(");
                    for (j, p) in parameters.iter().enumerate() {
                        if j > 0 { print!(", "); }
                        print!("{:.4}", p);
                    }
                    print!(") ");
                }
                print!("q{:?}", qubits);
                println!();
            }
            pecos_qasm::parser::Operation::Measure { .. } => {
                println!("  {}: measure", i);
            }
            _ => {}
        }
    }
    
    println!("\nThe teleport_prep gate was expanded into {} basic operations", 
             program.operations.iter()
                 .filter(|op| matches!(op, pecos_qasm::parser::Operation::Gate { .. }))
                 .count());
    
    Ok(())
}