use pecos_qasm::QASMParser;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Example demonstrating opaque gate declarations in QASM
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        
        // Registers
        qreg q[4];
        creg c[4];
        
        // Opaque gate declarations
        // These represent gates implemented at hardware level
        // without decomposition in QASM
        
        // Single-qubit opaque gate without parameters
        opaque oracle_x a;
        
        // Single-qubit opaque gate with parameters
        opaque oracle_phase(theta) a;
        
        // Two-qubit opaque gate
        opaque oracle_cnot a, b;
        
        // Multi-qubit opaque gate with parameters
        opaque oracle_3q(alpha, beta) a, b, c;
        
        // For now, we can only declare opaque gates, not use them
        // Using opaque gates will throw an error
        // oracle_x q[0];           // This would cause an error
        // oracle_phase(pi/4) q[1]; // This would cause an error

        // But we can still use regular gates
        h q[0];
        cx q[0], q[1];

        // Measure qubits
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    // Parse the QASM
    let program = QASMParser::parse_str(qasm)?;

    println!("Parsed QASM program with opaque gates:");
    println!("Version: {}", program.version);
    println!("\nQuantum registers:");
    for (name, qubits) in &program.quantum_registers {
        println!("  {} -> {:?}", name, qubits);
    }

    println!("\nOperations:");
    for (i, op) in program.operations.iter().enumerate() {
        println!("  {}: {:?}", i, op);
    }

    // Count opaque gate declarations vs usage
    let mut opaque_declarations = 0;
    let mut gate_usages = 0;

    for op in &program.operations {
        match op {
            pecos_qasm::parser::Operation::OpaqueGate { .. } => opaque_declarations += 1,
            pecos_qasm::parser::Operation::Gate { .. } => gate_usages += 1,
            _ => {}
        }
    }

    println!("\nOpaque gate declarations: {}", opaque_declarations);
    println!("Gate usages: {}", gate_usages);

    Ok(())
}
