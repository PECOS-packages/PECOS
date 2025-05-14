use pecos_qasm::QASMParser;

fn main() {
    // Example with circular dependency
    let qasm = r#"OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];

// Define some gates with a circular dependency
gate rotate_x(theta) q {
    rotate_y(theta) q;  // Calls rotate_y
}

gate rotate_y(theta) q {
    rotate_z(theta) q;  // Calls rotate_z
}

gate rotate_z(theta) q {
    rotate_x(theta) q;  // Calls rotate_x - creates cycle!
}

// This will trigger the circular dependency
rotate_x(pi/2) q[0];
"#;
    
    match QASMParser::parse_str(qasm) {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => {
            println!("Error detected: {}\n", e);
            
            // Show the problematic code with context
            let lines: Vec<&str> = qasm.lines().collect();
            
            // Find the cycle in the code
            println!("The circular dependency exists in these gate definitions:");
            println!();
            
            // Show rotate_x definition
            if let Some((idx, _)) = lines.iter().enumerate().find(|(_, line)| line.contains("gate rotate_x")) {
                println!("{}:  {}", idx + 1, lines[idx]);
                if idx + 1 < lines.len() {
                    println!("{}:  {}", idx + 2, lines[idx + 1]);
                    println!("       ^^^^^^^^^ calls rotate_y");
                }
            }
            println!();
            
            // Show rotate_y definition
            if let Some((idx, _)) = lines.iter().enumerate().find(|(_, line)| line.contains("gate rotate_y")) {
                println!("{}:  {}", idx + 1, lines[idx]);
                if idx + 1 < lines.len() {
                    println!("{}:  {}", idx + 2, lines[idx + 1]);
                    println!("       ^^^^^^^^^ calls rotate_z");
                }
            }
            println!();
            
            // Show rotate_z definition
            if let Some((idx, _)) = lines.iter().enumerate().find(|(_, line)| line.contains("gate rotate_z")) {
                println!("{}:  {}", idx + 1, lines[idx]);
                if idx + 1 < lines.len() {
                    println!("{}:  {}", idx + 2, lines[idx + 1]);
                    println!("       ^^^^^^^^^ calls rotate_x (creating the cycle!)");
                }
            }
        }
    }
}