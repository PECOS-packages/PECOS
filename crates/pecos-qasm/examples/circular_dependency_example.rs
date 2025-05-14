use pecos_qasm::QASMParser;

fn main() {
    // Example 1: Direct circular dependency (caught by parser)
    let qasm_with_cycle = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        // This gate references itself
        gate recursive q {
            recursive q;
        }
        
        // Attempt to use the recursive gate
        recursive q[0];
    "#;
    
    match QASMParser::parse_str(qasm_with_cycle) {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => println!("Caught circular dependency: {}", e),
    }
    
    // Example 2: Indirect circular dependency
    let qasm_indirect_cycle = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        gate a q { b q; }
        gate b q { c q; }
        gate c q { a q; }
        
        // This will trigger the cycle detection
        a q[0];
    "#;
    
    match QASMParser::parse_str(qasm_indirect_cycle) {
        Ok(_) => println!("Unexpected success!"),
        Err(e) => println!("Caught circular dependency: {}", e),
    }
    
    // Example 3: Valid deep nesting (no cycle)
    let qasm_valid = r#"
        OPENQASM 2.0;
        qreg q[1];
        
        gate level3 q { h q; }
        gate level2 q { level3 q; x q; }
        gate level1 q { level2 q; y q; }
        gate level0 q { level1 q; z q; }
        
        level0 q[0];
    "#;
    
    match QASMParser::parse_str(qasm_valid) {
        Ok(_) => println!("Valid deep nesting works correctly!"),
        Err(e) => println!("Unexpected error: {}", e),
    }
}