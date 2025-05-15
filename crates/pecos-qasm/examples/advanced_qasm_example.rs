use anyhow::Result;
use pecos_engines::engines::classical::ClassicalEngine;
use pecos_qasm::engine::QASMEngine;
use pecos_qasm::parser::QASMParser;

fn main() -> Result<()> {
    // Example of a supported QASM program
    let supported_qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        
        qreg q[4];
        creg c[4];
        
        // Supported gates and operations
        h q[0];
        rz(1.5*pi) q[1];
        cx q[0],q[1];
        x q[2];
        y q[3];
        rz(0.0375*pi) q[2];
        cz q[2],q[3];
        measure q -> c;
        
        // Conditional operations
        if(c==2) x q[0];
        
        // Mathematical expressions
        rx(0.5*pi) q[1];
        rzz(0.0375*pi) q[0],q[1];
        szz q[2],q[3];
        
        // Supported gate decompositions
        swap q[1],q[3];
        cy q[0],q[2];
        
        // Phase gates
        s q[1];
        sdg q[2];
        t q[3];
        tdg q[0];
        
        // Newer gates from qelib1
        sx q[0];
        sxdg q[1];
        rz(1.9625*pi) q[2];
    "#;

    println!("Parsing supported QASM program...");
    let program = QASMParser::parse_str(supported_qasm)?;
    println!("Parsed successfully!");

    let mut engine = QASMEngine::new()?;
    engine.load_program(program)?;
    let _commands = engine.generate_commands()?;
    println!("Circuit compiled successfully!");

    // Now demonstrate unsupported gates by showing what happens when we try to use them
    println!("\n--- Testing Unsupported Gates ---");

    // Example 1: RXX gate (not supported)
    let unsupported_rxx = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        rxx(0.5*pi) q[0],q[1];  // This should fail during compilation
    "#;

    println!("\n1. Testing RXX gate:");
    match QASMParser::parse_str(unsupported_rxx) {
        Ok(program) => {
            println!("   Parsed successfully");
            let mut engine = QASMEngine::new()?;
            engine.load_program(program)?;
            match engine.generate_commands() {
                Ok(_) => println!("   RXX gate supported (unexpected)"),
                Err(e) => println!("   RXX gate not supported: {}", e),
            }
        }
        Err(e) => println!("   Parse error: {}", e),
    }

    // Example 2: Toffoli gate (check if defined in qelib1.inc)
    let unsupported_ccx = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        ccx q[0],q[1],q[2];  // Toffoli gate
    "#;

    println!("\n2. Testing Toffoli (CCX) gate:");
    match QASMParser::parse_str(unsupported_ccx) {
        Ok(program) => {
            println!("   Parsed successfully");
            let mut engine = QASMEngine::new()?;
            engine.load_program(program)?;
            match engine.generate_commands() {
                Ok(_) => println!("   CCX gate supported (unexpected)"),
                Err(e) => println!("   CCX gate not supported: {}", e),
            }
        }
        Err(e) => println!("   Parse error: {}", e),
    }

    // Example 3: Barrier operation (not supported)
    let unsupported_barrier = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        barrier q[0],q[1];  // Timing barrier
    "#;

    println!("\n3. Testing barrier operation:");
    match QASMParser::parse_str(unsupported_barrier) {
        Ok(program) => {
            // Parser might succeed, but operations might not be supported
            println!("   Parsed successfully");
            let mut engine = QASMEngine::new()?;
            engine.load_program(program)?;
            match engine.generate_commands() {
                Ok(_) => println!("   Barrier supported (unexpected)"),
                Err(e) => println!("   Barrier not supported: {}", e),
            }
        }
        Err(e) => println!("   Parse error: {}", e),
    }

    // Example 4: CSX gate (testing our newly verified gate)
    let csx_test = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        csx q[0],q[1];  // Controlled-SX gate
    "#;

    println!("\n4. Testing CSX gate:");
    match QASMParser::parse_str(csx_test) {
        Ok(program) => {
            println!("   Parsed successfully");
            let mut engine = QASMEngine::new()?;
            engine.load_program(program)?;
            match engine.generate_commands() {
                Ok(_) => println!("   CSX gate compilation attempted"),
                Err(e) => println!("   CSX gate error: {}", e),
            }
        }
        Err(e) => println!("   Parse error: {}", e),
    }

    println!("\nNote: The PECOS QASM engine currently supports:");
    println!("   - Basic single-qubit gates (H, X, Y, Z, S, T)");
    println!("   - Single-qubit rotations (RZ, RX via decomposition)");
    println!("   - Two-qubit gates (CX, CZ, CY, SWAP)");
    println!("   - Parameterized rotations (RZZ, SZZ)");
    println!("   - sqrt(X) gates (SX, SXdg)");

    println!("\nGates that may not be supported in the engine:");
    println!("   - rxx: XX rotation gate");
    println!("   - ccx: Toffoli (controlled-controlled-X) gate  ");
    println!("   - barrier: Timing optimization barrier");
    println!("   - u3: General single-qubit unitary");
    println!("   - cu1: Controlled phase gate");
    println!("   - csx: Controlled-SX gate (not defined in qelib1.inc)");

    Ok(())
}
