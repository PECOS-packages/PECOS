use pecos_qasm::parser::QASMParser;

#[test]
fn test_barrier_mapping_full() -> Result<(), Box<dyn std::error::Error>> {
    // Test the complete barrier example from the test
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[4];
        qreg w[8]; 
        qreg a[1];
        qreg b[5];
        qreg c[3];
        creg a[5];
        
        // Regular barrier with multiple qubits
        barrier q[0],q[3],q[2];

        // All qubits from a register
        barrier c;

        // Mix of different registers
        barrier a[0], b[4], c;
        
        // More combinations
        barrier w[1], w[7];
        
        // Inside a conditional
        if(a>=5) barrier w[1], w[7];
    "#;

    // Let's print the expected mapping
    println!("\n=== Expected Qubit Mappings: ===");
    println!("q[0] -> 0");
    println!("q[1] -> 1"); 
    println!("q[2] -> 2");
    println!("q[3] -> 3");
    println!("w[0] -> 4");
    println!("w[1] -> 5");
    println!("w[2] -> 6");
    println!("w[3] -> 7");
    println!("w[4] -> 8");
    println!("w[5] -> 9");
    println!("w[6] -> 10");
    println!("w[7] -> 11");
    println!("a[0] -> 12");
    println!("b[0] -> 13");
    println!("b[1] -> 14");
    println!("b[2] -> 15");
    println!("b[3] -> 16");
    println!("b[4] -> 17");
    println!("c[0] -> 18");
    println!("c[1] -> 19");
    println!("c[2] -> 20");

    // Parse and see the operations
    let program = QASMParser::parse_str(qasm)?;
    
    // Print actual operations
    println!("\n=== Parsed Operations: ===");
    for (i, op) in program.operations.iter().enumerate() {
        println!("Op {}: {:?}", i, op);
    }

    Ok(())
}