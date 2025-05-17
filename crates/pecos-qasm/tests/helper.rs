use pecos_core::errors::PecosError;
use pecos_engines::{MonteCarloEngine, PassThroughNoiseModel};
use pecos_qasm::QASMEngine;
use std::collections::HashMap;
use std::str::FromStr;

pub fn run_qasm_sim(
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
