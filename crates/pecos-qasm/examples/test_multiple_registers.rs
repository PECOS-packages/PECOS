use pecos_qasm::QASMEngine;
use pecos_engines::Engine;
use pecos_core::errors::PecosError;

fn main() -> Result<(), PecosError> {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q1[2];
        qreg q2[3];
        creg c[5];
        h q1[0];
        cx q1[0],q2[0];
        h q1[1]; 
        cx q1[1],q2[1];
        h q2[2];
        measure q1[0] -> c[0];
        measure q1[1] -> c[1];
        measure q2[0] -> c[2];
        measure q2[1] -> c[3];
        measure q2[2] -> c[4];
    "#;

    let mut engine = QASMEngine::new()?;
    engine.from_str(qasm)?;
    
    // Test the get_qubit_id method
    println!("Testing get_qubit_id:");
    println!("q1[0] -> {:?}", engine.get_qubit_id("q1", 0));
    println!("q1[1] -> {:?}", engine.get_qubit_id("q1", 1));
    println!("q2[0] -> {:?}", engine.get_qubit_id("q2", 0));
    println!("q2[1] -> {:?}", engine.get_qubit_id("q2", 1));
    println!("q2[2] -> {:?}", engine.get_qubit_id("q2", 2));
    println!("q3[0] -> {:?}", engine.get_qubit_id("q3", 0)); // Should be None
    println!();
    
    // Run the circuit
    let result = engine.process(())?;
    
    println!("Circuit executed successfully!");
    println!("Classical register 'c' value: {:?}", result.registers.get("c"));
    
    Ok(())
}