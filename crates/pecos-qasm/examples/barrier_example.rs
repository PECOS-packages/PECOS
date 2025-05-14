use pecos_qasm::QASMEngine;
use pecos_engines::Engine;
use pecos_core::errors::PecosError;

fn main() -> Result<(), PecosError> {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        qreg r[2];
        creg c[6];
        
        // Apply some gates
        h q[0];
        cx q[0], q[1];
        
        // Barrier with individual qubits
        barrier q[0], q[1];
        
        h q[2];
        cx q[2], q[3];
        
        // Barrier with entire register
        barrier q;
        
        h r[0];
        cx r[0], r[1];
        
        // Mixed barrier with register and individual qubits
        barrier r, q[0], q[3];
        
        // Measure all qubits
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];
        measure q[3] -> c[3];
        measure r[0] -> c[4];
        measure r[1] -> c[5];
    "#;

    let mut engine = QASMEngine::new()?;
    engine.from_str(qasm)?;
    
    // Run the circuit
    let result = engine.process(())?;
    
    println!("Circuit executed successfully!");
    println!("Measurement results: {:?}", result.registers);
    
    Ok(())
}