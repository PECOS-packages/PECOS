use pecos_qasm::QASMEngine;
use pecos_engines::ClassicalEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== PECOS QASM Gate Support ===\n");
    
    // Gates that ACTUALLY work
    let supported_qasm = r#"
        OPENQASM 2.0;
        qreg q[4];
        creg c[4];
        
        // These gates are ACTUALLY supported by the engine:
        
        // Native single-qubit gates
        h q[0];
        x q[1];
        y q[2];
        z q[3];
        
        // Phase gates (engine implementation)
        s q[0];
        sdg q[1];
        t q[2];
        tdg q[3];
        
        // Native rotations
        rz(1.5*pi) q[0];
        
        // Native two-qubit gates
        cx q[0],q[1];
        rzz(0.0375*pi) q[2],q[3];
        szz q[0],q[2];
        
        // Engine-implemented two-qubit gates
        cz q[0],q[1];
        cy q[1],q[2];
        swap q[2],q[3];
        
        measure q[0] -> c[0];
        measure q[1] -> c[1];
        measure q[2] -> c[2];
        measure q[3] -> c[3];
    "#;
    
    let mut engine = QASMEngine::new()?;
    engine.from_str(supported_qasm)?;
    println!("[OK] Actually supported gates compiled successfully!");
    
    let _commands = engine.generate_commands()?;
    
    // Gates defined in qelib1.inc but NOT working
    println!("\n=== Gates in qelib1.inc but NOT working ===\n");
    
    let test_cases = vec![
        ("rx(0.1) q[0];", "rx - X-axis rotation (decomposed)"),
        ("crz(0.1) q[0],q[1];", "crz - Controlled RZ (decomposed)"),
        ("cphase(0.1) q[0],q[1];", "cphase - Controlled phase (decomposed)"),
        ("sx q[0];", "sx - Square root of X (decomposed)"),
        ("sxdg q[0];", "sxdg - Inverse square root of X (decomposed)"),
    ];
    
    for (gate, description) in test_cases {
        let test_qasm = format!(r#"
            OPENQASM 2.0;
            include "qelib1.inc";
            qreg q[2];
            {}
        "#, gate);
        
        let mut engine = QASMEngine::new()?;
        match engine.from_str(&test_qasm) {
            Ok(_) => {
                match engine.generate_commands() {
                    Ok(_) => println!("[OK] {} - Unexpectedly works!", description),
                    Err(_) => println!("[FAIL] {} - Defined but not supported", description),
                }
            }
            Err(_) => println!("[FAIL] {} - Parse error", description),
        }
    }
    
    println!("\n=== Summary ===");
    println!("The engine only supports gates with explicit implementations.");
    println!("Gates defined via decomposition in qelib1.inc are NOT automatically expanded.");
    println!("\nTo use the full qelib1.inc, the engine would need to:");
    println!("1. Parse and apply gate decompositions, OR");
    println!("2. Add explicit implementations for these gates");
    
    Ok(())
}