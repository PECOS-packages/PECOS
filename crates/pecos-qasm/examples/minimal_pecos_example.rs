use pecos_engines::ClassicalEngine;
use pecos_qasm::QASMEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let qasm_code = r#"
        OPENQASM 2.0;
        include "pecos.inc";
        
        // Declare quantum and classical registers
        qreg q[3];
        creg c[3];
        
        // Use only native PECOS gates
        h q[0];
        x q[1];
        y q[2];
        
        // Native rotations
        rz(1.5708) q[0];  // π/2 rotation
        r1xy(0.7854, 0.3927) q[1];  // π/4, π/8 rotation
        
        // Native two-qubit gates
        cx q[0], q[1];
        szz q[1], q[2];
        
        // Measure all qubits
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];
    "#;

    // Create engine and parse QASM
    let mut engine = QASMEngine::new()?;
    engine.from_str(qasm_code)?;

    // Print the parsed program structure
    println!("Program using minimal pecos.inc parsed successfully!");

    // Generate commands to verify the circuit compiles
    let _commands = engine.generate_commands()?;
    println!("Circuit with native gates compiled successfully!");

    println!("\nThis example demonstrates using only native PECOS gates");
    println!("via the minimal pecos.inc library.");

    Ok(())
}
