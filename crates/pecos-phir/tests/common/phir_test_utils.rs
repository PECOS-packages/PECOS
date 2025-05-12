#![allow(dead_code)]

use pecos_core::errors::PecosError;
use pecos_engines::{Engine, MonteCarloEngine, NoiseModel, PassThroughNoiseModel, ShotResults};
use pecos_engines::core::shot_results::ShotResult;
use pecos_phir::v0_1::engine::PHIREngine;
use pecos_phir::setup_phir_engine;
use std::path::PathBuf;

/// Run a PHIR simulation and get the results
/// 
/// # Arguments
/// 
/// * `path` - Path to the PHIR JSON file (relative to CARGO_MANIFEST_DIR)
/// * `shots` - Number of shots to run
/// * `workers` - Number of workers to use
/// * `seed` - Optional seed for reproducibility
/// * `noise_model` - Optional noise model to use (defaults to PassThroughNoiseModel)
/// 
/// # Returns
/// 
/// * `ShotResults` - The results of the simulation
pub fn run_phir_simulation<T: NoiseModel + 'static>(
    path: &str,
    shots: usize,
    workers: usize,
    seed: Option<u64>,
    noise_model: Option<T>,
) -> Result<ShotResults, PecosError> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    // Path to the test file
    let phir_path = manifest_dir.join(path);

    // Set up the PHIR engine
    let classical_engine = setup_phir_engine(&phir_path)
        .map_err(|e| PecosError::with_context(e, format!("Failed to set up PHIR engine from file: {}", path)))?;

    // Use the provided noise model or default to PassThroughNoiseModel
    let noise_model_box: Box<dyn NoiseModel> = match noise_model {
        Some(model) => Box::new(model),
        None => Box::new(PassThroughNoiseModel),
    };

    // Run the Monte Carlo engine
    let results = MonteCarloEngine::run_with_noise_model(
        classical_engine,
        noise_model_box,
        shots,
        workers,
        seed,
    )
    .map_err(|e| PecosError::with_context(e, "Failed to run Monte Carlo engine with noise model"))?;
    
    Ok(results)
}

/// Run a PHIR program directly using the PHIREngine
/// 
/// This is useful for tests that don't need a full simulation
/// but just want to verify the core engine functionality.
pub fn run_phir_engine(path: &str) -> Result<ShotResult, PecosError> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let phir_path = manifest_dir.join(path);
    
    // Create a PHIR engine from the program file
    let mut engine = PHIREngine::new(phir_path)?;
    
    // Execute the program
    let result = engine.process(())?;
    
    Ok(result)
}

/// Helper function to get the simulation results for a PHIR test
/// with default settings (1 shot, 1 worker, no seed, no noise)
pub fn get_phir_results(path: &str) -> Result<ShotResult, PecosError> {
    run_phir_engine(path)
}

/// Assert that a register has an expected value in a ShotResult
/// 
/// # Arguments
/// 
/// * `result` - The ShotResult to check
/// * `register_name` - The name of the register to check
/// * `expected_value` - The expected value of the register
/// 
/// # Panics
/// 
/// * If the register does not exist
/// * If the register value does not match the expected value
pub fn assert_shotresult_value(result: &ShotResult, register_name: &str, expected_value: u32) {
    // Check the register value
    if let Some(&value) = result.registers.get(register_name) {
        assert_eq!(value, expected_value, 
            "Register '{}' has value {} but expected {}", 
            register_name, value, expected_value);
        return;
    }
    
    // Also check the u64 registers
    if let Some(&value) = result.registers_u64.get(register_name) {
        // Convert to u32 and compare
        if value <= u32::MAX as u64 {
            let value_u32 = value as u32;
            assert_eq!(value_u32, expected_value, 
                "Register '{}' has u64 value {} but expected {} as u32", 
                register_name, value, expected_value);
            return;
        } else {
            panic!("Register '{}' has u64 value {} which is too large to convert to u32 for comparison", 
                register_name, value);
        }
    }
    
    // Also check the i64 registers
    if let Some(&value) = result.registers_i64.get(register_name) {
        // Convert to u32 and compare
        if value >= 0 && value <= u32::MAX as i64 {
            let value_u32 = value as u32;
            assert_eq!(value_u32, expected_value, 
                "Register '{}' has i64 value {} but expected {} as u32", 
                register_name, value, expected_value);
            return;
        } else {
            panic!("Register '{}' has i64 value {} which cannot be converted to u32 for comparison", 
                register_name, value);
        }
    }
    
    panic!("Register '{}' not found in result. Available registers: {:?}", 
        register_name, 
        result.registers.keys().collect::<Vec<_>>());
}

/// Assert that multiple registers have expected values in a ShotResult
/// 
/// # Arguments
/// 
/// * `result` - The ShotResult to check
/// * `expected_values` - A vector of (register_name, expected_value) pairs
/// 
/// # Panics
/// 
/// * If any register does not exist
/// * If any register value does not match the expected value
pub fn assert_shotresult_values(result: &ShotResult, expected_values: &[(&str, u32)]) {
    for (register_name, expected_value) in expected_values {
        assert_shotresult_value(result, register_name, *expected_value);
    }
}

/// Assert that a register has an expected value in a ShotResults
/// 
/// # Arguments
/// 
/// * `results` - The simulation results
/// * `register_name` - The name of the register to check
/// * `expected_value` - The expected value of the register
/// 
/// # Panics
/// 
/// * If the register does not exist
/// * If the register value does not match the expected value
pub fn assert_register_value(results: &ShotResults, register_name: &str, expected_value: i64) {
    // First check in i64 registers which is most accurate for our expected values
    if let Some(values) = results.register_shots_i64.get(register_name) {
        assert!(values.len() > 0, "Register '{}' found but has no values", register_name);
        assert_eq!(values[0], expected_value, 
            "Register '{}' has i64 value {} but expected {}", 
            register_name, values[0], expected_value);
        return;
    }
    
    // Then check in the u32 registers
    if let Some(values) = results.register_shots.get(register_name) {
        assert!(values.len() > 0, "Register '{}' found but has no values", register_name);
        // Convert to i64 for comparison
        let value_i64 = values[0] as i64;
        assert_eq!(value_i64, expected_value, 
            "Register '{}' has u32 value {} but expected {} as i64", 
            register_name, values[0], expected_value);
        return;
    }
    
    // Finally check in u64 registers
    if let Some(values) = results.register_shots_u64.get(register_name) {
        assert!(values.len() > 0, "Register '{}' found but has no values", register_name);
        // For large u64 values outside the i64 range, this could fail
        if let Ok(value_i64) = i64::try_from(values[0]) {
            assert_eq!(value_i64, expected_value, 
                "Register '{}' has u64 value {} but expected {} as i64", 
                register_name, values[0], expected_value);
            return;
        } else {
            panic!("Register '{}' has u64 value {} which is too large to convert to i64 for comparison", 
                register_name, values[0]);
        }
    }
    
    panic!("Register '{}' not found in any register types. Available registers: {:?}", 
        register_name, 
        results.register_shots.keys().chain(results.register_shots_u64.keys()).chain(results.register_shots_i64.keys())
            .collect::<std::collections::HashSet<_>>());
}

/// Assert that multiple registers have expected values
/// 
/// # Arguments
/// 
/// * `results` - The simulation results
/// * `expected_values` - A vector of (register_name, expected_value) pairs
/// 
/// # Panics
/// 
/// * If any register does not exist
/// * If any register value does not match the expected value
pub fn assert_register_values(results: &ShotResults, expected_values: &[(&str, i64)]) {
    for (register_name, expected_value) in expected_values {
        assert_register_value(results, register_name, *expected_value);
    }
}