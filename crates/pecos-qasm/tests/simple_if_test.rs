// Test to verify if statement processing
use pecos_core::errors::PecosError;
use pecos_engines::{MonteCarloEngine, PassThroughNoiseModel};
use pecos_qasm::QASMEngine;
use std::collections::HashMap;

fn run_qasm_sim(
    qasm: &str,
    shots: usize,
    seed: Option<u64>,
) -> Result<HashMap<String, Vec<u32>>, PecosError> {
    let engine = QASMEngine::from_str(qasm)?;

    let results = MonteCarloEngine::run_with_noise_model(
        Box::new(engine),
        Box::new(PassThroughNoiseModel),
        shots,
        1,
        seed,
    )?
    .register_shots;

    Ok(results)
}

#[test]
fn test_simple_if() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        
        c[0] = 0;
        if(c[0]==0) X q[0];
        measure q[0] -> c[0];
    "#;

    let results = run_qasm_sim(qasm, 1, Some(42)).unwrap();

    println!("Simple if test results: {results:?}");

    assert!(results.contains_key("c"));
    assert_eq!(results["c"], vec![1]);
}
