use pecos_qasm::QASMParser;

fn main() {
    // Example demonstrating the error when trying to use opaque gates
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        
        qreg q[2];
        creg c[2];
        
        // Declare an opaque gate
        opaque oracle a;
        
        // Try to use the opaque gate - this will cause an error
        h q[0];
        oracle q[0];  // This line will cause an error
        
        measure q -> c;
    "#;
    
    // Parse the QASM
    match QASMParser::parse_str(qasm) {
        Ok(_) => {
            println!("This shouldn't happen - we expect an error");
        }
        Err(e) => {
            println!("Expected error occurred:");
            println!("{}", e);
            println!("\nThis error is expected because opaque gates are not yet implemented in PECOS.");
            println!("You can declare opaque gates, but cannot use them in circuits.");
        }
    }
}