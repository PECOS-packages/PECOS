use pecos_engines::ClassicalEngine;
use pecos_qasm::QASMEngine;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Test gates that are defined in qelib1.inc
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        
        qreg q[2];
        creg c[2];
        
        // Test CRZ gate (defined in qelib1.inc)
        crz(1.5708) q[0], q[1];  // π/2
        
        // Test other gates from qelib1.inc
        cphase(0.7854) q[0], q[1];  // π/4
        phase(0.3927) q[0];
        
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    let mut engine = QASMEngine::new()?;
    match engine.from_str(qasm) {
        Ok(_) => println!("[OK] QASM with qelib1.inc gates parsed successfully!"),
        Err(e) => println!("[FAIL] Parse error: {:?}", e),
    }

    match engine.generate_commands() {
        Ok(_) => println!("[OK] Circuit compiled successfully!"),
        Err(e) => println!("[FAIL] Compilation error: {:?}", e),
    }

    println!("\nTesting unsupported gates:");

    let unsupported_qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        
        qreg q[3];
        
        // This should fail - not in qelib1.inc
        ccx q[0], q[1], q[2];
    "#;

    let mut engine2 = QASMEngine::new()?;
    match engine2.from_str(unsupported_qasm) {
        Ok(_) => println!("[OK] QASM parsed"),
        Err(e) => println!("[FAIL] Parse error: {:?}", e),
    }

    match engine2.generate_commands() {
        Ok(_) => println!("[FAIL] Unexpectedly compiled CCX gate!"),
        Err(e) => println!("[OK] Expected error for unsupported gate: {:?}", e),
    }

    Ok(())
}
