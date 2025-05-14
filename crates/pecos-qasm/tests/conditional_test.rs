use pecos_engines::Engine;
use pecos_qasm::engine::QASMEngine;
use std::error::Error;

#[test]
fn test_conditional_execution() -> Result<(), Box<dyn Error>> {
    // Create QASM that includes conditional statements
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        
        // Create registers
        qreg q[2];
        creg c[2];
        
        // Initialize qubit 0 in superposition
        h q[0];
        
        // Measure qubit 0 to c[0]
        measure q[0] -> c[0];
        
        // Conditional quantum operation: if c[0]==1, apply X to q[1]
        if(c[0]==1) x q[1];
        
        // Measure q[1] to c[1]
        measure q[1] -> c[1];
    "#;

    // Create and initialize the engine
    let mut engine = QASMEngine::new()?;
    engine.from_str(qasm)?;
    
    // Run multiple shots to see different outcomes
    let total_shots = 10;
    let mut ones_count = 0;
    
    for _ in 0..total_shots {
        // Process the circuit for this shot
        let result = engine.process(())?;
        
        // Check the results
        if let Some(c_value) = result.registers.get("c") {
            // The c register should have the measurement results
            // If c[0] == 1, then c[1] should also be 1 due to the conditional
            // If c[0] == 0, then c[1] should be 0 (no X applied)
            println!("Shot result: c = {:#04b}", c_value);
            
            // Count shots where we got a 1 on the first qubit
            if c_value & 1 == 1 {
                ones_count += 1;
                
                // For these shots, c[1] should also be 1 due to the conditional X
                assert_eq!(c_value & 2, 2, "If c[0]=1, then c[1] should be 1 due to conditional X");
            }
        } else {
            panic!("No 'c' register in results");
        }
    }
    
    // Since h creates a 50/50 superposition, we expect approximately half
    // the shots to have c[0]=1, but allow some statistical variation
    println!("Got {} shots with c[0]=1 out of {}", ones_count, total_shots);
    
    // In all cases, the conditional logic should be correct
    Ok(())
}

#[test]
fn test_conditional_classical_assignment() -> Result<(), Box<dyn Error>> {
    // Create QASM with conditional classical assignments
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        
        // Create registers
        qreg q[1];
        creg c[2];
        
        // Initialize qubit in superposition
        h q[0];
        
        // Measure qubit to c[0]
        measure q[0] -> c[0];
        
        // Conditional classical operation: if c[0]==1, set c[1]=1
        if(c[0]==1) c[1] = 1;
        
        // Conditional classical operation: if c[0]==0, set c[1]=0
        if(c[0]==0) c[1] = 0;
    "#;

    // Create and initialize the engine
    let mut engine = QASMEngine::new()?;
    engine.from_str(qasm)?;
    
    // Run multiple shots
    let total_shots = 10;
    
    for _ in 0..total_shots {
        // Process the circuit
        let result = engine.process(())?;
        
        // Check results
        if let Some(c_value) = result.registers.get("c") {
            let c0 = c_value & 1;
            let c1 = (c_value >> 1) & 1;
            
            println!("Shot result: c[0]={}, c[1]={}", c0, c1);
            
            // c[1] should equal c[0] due to the conditional assignments
            assert_eq!(c0, c1, "c[1] should equal c[0] due to conditional assignment");
        } else {
            panic!("No 'c' register in results");
        }
    }
    
    Ok(())
}