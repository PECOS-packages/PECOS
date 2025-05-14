use pecos_qasm::QASMParser;

fn main() {
    // Example with circular dependency
    let qasm = r#"OPENQASM 2.0;
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
            println!("{}", e);
        }
    }
    
    println!("\n--- Another example ---\n");
    
    // Simpler self-referential example
    let qasm2 = r#"OPENQASM 2.0;
qreg q[1];

gate recursive_gate a {
    recursive_gate a;  // Direct self-reference
}

recursive_gate q[0];
"#;
    
    match QASMParser::parse_str(qasm2) {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => {
            println!("{}", e);
        }
    }
}