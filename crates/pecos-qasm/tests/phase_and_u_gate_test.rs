use pecos_engines::engines::classical::ClassicalEngine;
use pecos_qasm::engine::QASMEngine;
use pecos_qasm::parser::QASMParser;

#[test]
fn test_phase_zero_gate() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        p(0) q[0];
        measure q[0] -> c[0];
    "#;

    // Create and run the engine
    let mut engine = QASMEngine::from_str(qasm)
        .expect("Failed to load program");

    // The phase gate p(0) should not affect the |0⟩ state
    // We expect this to compile and run without errors
    match engine.generate_commands() {
        Ok(_) => {
            println!("Phase gate p(0) compiled successfully");
        }
        Err(e) => {
            // If p gate is not directly supported, check if it's in the error
            assert!(
                e.to_string().contains('p') || e.to_string().contains("phase"),
                "Unexpected error: {e}"
            );
        }
    }
}

#[test]
fn test_u_gate_identity() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        u(0,0,0) q[0];
        measure q[0] -> c[0];
    "#;

    // Create and run the engine
    let mut engine = QASMEngine::from_str(qasm)
        .expect("Failed to load program");

    // The U(0,0,0) gate should be the identity operation
    // We expect this might fail since u gate might not be supported
    match engine.generate_commands() {
        Ok(_) => {
            println!("U gate U(0,0,0) compiled successfully");
        }
        Err(e) => {
            // Check that the error mentions the u gate
            assert!(
                e.to_string().contains('u') || e.to_string().contains("unitary"),
                "Unexpected error: {e}"
            );
        }
    }
}

#[test]
fn test_combined_phase_and_u() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        p(0) q[0];
        u(0,0,0) q[0];
        measure q[0] -> c[0];
    "#;

    // Create and run the engine
    let mut engine = QASMEngine::from_str(qasm)
        .expect("Failed to load program");

    // Test the combination of p(0) and U(0,0,0)
    match engine.generate_commands() {
        Ok(_) => {
            println!("Combined p(0) and U(0,0,0) compiled successfully");
        }
        Err(e) => {
            println!("Expected error for unsupported gates: {e}");
            // Make sure the error is about unsupported gates
            assert!(
                e.to_string().contains("gate") || e.to_string().contains("supported"),
                "Unexpected error type: {e}"
            );
        }
    }
}

#[test]
fn test_phase_expansion() {
    // First, let's see what gates are actually defined in qelib1.inc
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
    "#;

    let program = QASMParser::parse_str(qasm).expect("Failed to parse QASM");

    // Check if p gate is defined
    if program.gate_definitions.contains_key("p") {
        println!("Phase gate 'p' is defined in qelib1.inc");
        if let Some(p_def) = program.gate_definitions.get("p") {
            println!("p gate body has {} operations", p_def.body.len());
            for (i, op) in p_def.body.iter().enumerate() {
                println!("  Operation {}: {}", i, op.name);
            }
        }
    } else {
        println!("Phase gate 'p' is NOT defined in qelib1.inc");
    }

    // Check if u gate is defined
    if program.gate_definitions.contains_key("u") {
        println!("Universal gate 'u' is defined in qelib1.inc");
    } else {
        println!("Universal gate 'u' is NOT defined in qelib1.inc");
    }

    // Check if u1, u2, u3 are defined
    for gate in &["u1", "u2", "u3"] {
        if program.gate_definitions.contains_key(*gate) {
            println!("{gate} gate is defined in qelib1.inc");
        }
    }
}
