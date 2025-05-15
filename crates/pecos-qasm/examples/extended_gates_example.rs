use pecos_engines::ClassicalEngine;
use pecos_qasm::QASMEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let qasm_code = r#"
        OPENQASM 2.0;
        
        // Declare quantum and classical registers
        qreg q[3];
        creg c[3];
        
        // Apply single-qubit gates
        h q[0];    // Hadamard
        s q[1];    // S gate
        t q[2];    // T gate
        
        // Apply S-dagger and T-dagger gates
        sdg q[0];
        tdg q[1];
        
        // Apply two-qubit gates
        cz q[0], q[1];   // Controlled-Z
        cy q[1], q[2];   // Controlled-Y
        swap q[0], q[2]; // SWAP
        
        // Apply native PECOS gates
        rz(1.5708) q[0];  // RZ rotation (pi/2)
        cx q[1], q[2];    // CNOT
        
        // Measure all qubits
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];
    "#;

    // Create engine and parse QASM
    let mut engine = QASMEngine::new()?;
    engine.from_str(qasm_code)?;

    // Print the parsed program structure
    println!("Program parsed successfully!");

    // Check program structure (just the public interface)
    // Since program.operations is private, we just verify parsing works

    // Generate commands to verify the circuit compiles
    let _commands = engine.generate_commands()?;
    println!("Circuit compiled successfully!");

    // Note: To actually run the circuit, you would need to use
    // a suitable simulation backend from pecos-engines

    Ok(())
}
