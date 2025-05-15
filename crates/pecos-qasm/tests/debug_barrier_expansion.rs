use pecos_qasm::parser::QASMParser;
use pecos_qasm::preprocessor::Preprocessor;

#[test]
fn test_barrier_mapping_debug() -> Result<(), Box<dyn std::error::Error>> {
    // Isolated test for the problematic conditional barrier
    let qasm = r#"
        OPENQASM 2.0;
        qreg q[4];
        qreg w[8];
        creg a[5];

        // This is the line causing issues
        if(a>=5) barrier w[1], w[7];
    "#;

    // First check phase 1 (preprocessing)
    let mut preprocessor = Preprocessor::new();
    let preprocessed = preprocessor.preprocess_str(qasm)?;
    println!("\n=== Phase 1 (after preprocessing): ===");
    println!("{}", preprocessed);

    // Now check phase 2 expansion
    let expanded_phase2 = QASMParser::expand_all_gate_definitions(&preprocessed)?;
    println!("\n=== Phase 2 (after gate expansion): ===");
    println!("{}", expanded_phase2);

    // Finally parse and see what happens
    println!("\n=== Attempting full parse: ===");
    match QASMParser::parse_str(qasm) {
        Ok(program) => {
            println!("Parse succeeded!");
            println!("Operations: {:?}", program.operations);
        }
        Err(e) => {
            println!("Parse failed with error: {}", e);
        }
    }

    Ok(())
}