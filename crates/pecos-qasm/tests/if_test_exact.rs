// Test to verify exact issue with if statement processing  
use pecos_core::errors::PecosError;
use pecos_engines::{MonteCarloEngine, PassThroughNoiseModel};
use pecos_qasm::QASMEngine;
use std::collections::HashMap;

fn run_qasm_sim(qasm: &str,
                shots: usize,
                seed: Option<u64>,) -> Result<HashMap<String, Vec<u32>>, PecosError> {
    let mut engine = QASMEngine::new()?;
    engine.from_str(qasm)?;
    
    let results = MonteCarloEngine::run_with_noise_model(
        Box::new(engine),
        Box::new(PassThroughNoiseModel),
        shots,
        1,
        seed,
    )?.register_shots;
    
    Ok(results)
}

#[test]
fn test_exact_issue() {
    // Test the exact problem from test_cond_bell
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg one_0[2];
        
        h q[0];
        cx q[0], q[1];
        measure q[0] -> one_0[0];  // This will be 0 or 1 due to Bell state
        
        // If one_0[0] is 0, then apply X to q[1]
        // After this, q[1] should be in |1> state when one_0[0] == 0
        if(one_0[0]==0) x q[1];
        
        measure q[1] -> one_0[1];  // Should always be 1
        one_0[0] = 0;             // Reset to 0
    "#;

    // Run just once 
    let results = run_qasm_sim(qasm, 1, Some(42)).unwrap();
    
    println!("Test results: {:?}", results);
    
    // The expected result is one_0 = "10" (binary) = 2 (decimal)
    assert!(results.contains_key("one_0"));
    
    // For testing, let's understand what's happening
    println!("Full result: {:?}", results["one_0"][0]);
    
    // The bits should be: [0, 1] which equals 2 in decimal
    assert_eq!(results["one_0"][0], 2, "Expected result to be 2 (binary 10)");
}

#[test]
fn test_if_with_zero() {
    // Test case where measurement is forced to 0 by preparing |0> state
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        
        // Prepare q[0] in |0> state
        // Don't apply anything - it's already in |0>
        
        // Prepare q[1] in |0> state
        // Don't apply anything - it's already in |0>
        
        measure q[0] -> c[0];      // Will be 0
        
        if(c[0]==0) x q[1];        // Should execute
        
        measure q[1] -> c[1];      // Should be 1
        c[0] = 0;                  // Reset to 0
    "#;

    let results = run_qasm_sim(qasm, 1, Some(42)).unwrap();
    
    println!("If with zero test results: {:?}", results);
    
    assert!(results.contains_key("c"));
    assert_eq!(results["c"][0], 2, "Expected result to be 2 (binary 10)");
}

#[test]
fn test_if_with_one() {
    // Test case where measurement is forced to 1 by applying X
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        
        // Prepare q[0] in |1> state
        x q[0];
        
        // Prepare q[1] in |0> state
        // Don't apply anything - it's already in |0>
        
        measure q[0] -> c[0];      // Will be 1
        
        if(c[0]==0) x q[1];        // Should NOT execute
        
        measure q[1] -> c[1];      // Should be 0
        c[0] = 0;                  // Reset to 0
    "#;

    let results = run_qasm_sim(qasm, 1, Some(42)).unwrap();
    
    println!("If with one test results: {:?}", results);
    
    assert!(results.contains_key("c"));
    assert_eq!(results["c"][0], 0, "Expected result to be 0 (binary 00)");
}