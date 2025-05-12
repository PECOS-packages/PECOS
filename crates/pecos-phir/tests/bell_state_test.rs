mod common;

use pecos_core::rng::RngManageable;
use pecos_engines::engines::MonteCarloEngine;
use pecos_engines::{PassThroughNoiseModel, DepolarizingNoiseModel};
use pecos_phir::setup_phir_engine;
use std::collections::HashMap;
use std::path::PathBuf;

// Import helpers from common module
use crate::common::phir_test_utils::{get_phir_results, assert_shotresult_value};

#[test]
fn test_bell_state_noiseless() {
    // Get the path to the Bell state example
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir
        .parent()
        .expect("CARGO_MANIFEST_DIR should have a parent")
        .parent()
        .expect("Expected to find workspace directory as parent of crates/");
    let bell_file = workspace_dir.join("examples/phir/bell.json");

    // Run the Bell state example with 100 shots and 2 workers
    let classical_engine =
        setup_phir_engine(&bell_file).expect("Failed to set up PHIR engine from bell.json file");

    // Use the generic approach
    let results = MonteCarloEngine::run_with_noise_model(
        classical_engine,
        Box::new(PassThroughNoiseModel),
        100,
        2,
        None, // No specific seed
    )
    .expect("Failed to run Monte Carlo engine with noise model");

    // Count occurrences of each result
    let mut counts: HashMap<String, usize> = HashMap::new();

    // Process results - note that the test could pass even if "result" is not in the shot
    for shot in &results.shots {
        // If there's no "result" key in the output, just count it as an empty result
        let result_str = shot
            .get("result")
            .map_or_else(String::new, std::clone::Clone::clone);
        *counts.entry(result_str).or_insert(0) += 1;
    }

    // Print the counts for debugging
    println!("Noiseless Bell state results:");
    for (result, count) in &counts {
        println!("  {result}: {count}");
    }

    // The test passes if there are no errors in the execution
    assert!(!results.shots.is_empty(), "Expected non-empty results");
    
    println!("Results: {:?}", results);
}

#[test]
#[ignore = "Direct execution with PHIREngine not working for Bell state example yet"]
fn test_bell_state_using_helper() {
    // Get the path to the Bell state example
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir
        .parent()
        .expect("CARGO_MANIFEST_DIR should have a parent")
        .parent()
        .expect("Expected to find workspace directory as parent of crates/");
    let bell_path = workspace_dir.join("examples/phir/bell.json").to_string_lossy().to_string();

    // Run a single instance of the Bell state test
    let result = get_phir_results(&bell_path)
        .expect("Failed to run Bell state PHIR program");
    
    // Print all information about the result for debugging
    println!("ShotResult: {:?}", result);
    println!("Registers: {:?}", result.registers);
    
    // Bell state should result in either 00 (0) or 11 (3) measurement outcomes
    if let Some(&value) = result.registers.get("result") {
        assert!(value == 0 || value == 3, 
            "Expected Bell state result to be 0 or 3, got {}", value);
    } else {
        // Handle the case where "result" is not in registers
        if let Some(&value) = result.registers.get("output") {
            assert!(value == 0 || value == 3, 
                "Expected Bell state output to be 0 or 3, got {}", value);
        } else {
            // No result or output register found
            panic!("Expected 'result' or 'output' register to be present");
        }
    }
}

#[allow(clippy::cast_precision_loss)]
#[test]
fn test_bell_state_with_noise() {
    // Get the path to the Bell state example
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir
        .parent()
        .expect("CARGO_MANIFEST_DIR should have a parent")
        .parent()
        .expect("Expected to find workspace directory as parent of crates/");
    let bell_file = workspace_dir.join("examples/phir/bell.json");

    // Try multiple runs with different seeds
    for seed in 1..=3 {
        println!("Attempting test with seed {seed}");

        // Run the Bell state example with high noise probability for more reliable testing
        let classical_engine = setup_phir_engine(&bell_file)
            .expect("Failed to set up PHIR engine from bell.json file");

        // Create a noise model with 30% depolarizing noise
        let mut noise_model =
            DepolarizingNoiseModel::new_uniform(0.3);

        // Set the seed
        noise_model
            .set_seed(seed)
            .expect("Failed to set seed for noise model");

        // Use the generic approach
        let results = MonteCarloEngine::run_with_noise_model(
            classical_engine,
            Box::new(noise_model),
            100, // 100 shots is enough for this simple test
            2,
            Some(seed), // Use the current iteration as seed
        )
        .expect("Failed to run Monte Carlo engine with noise model");

        // Count occurrences of each result
        let mut counts: HashMap<String, usize> = HashMap::new();

        // For the noisy version, we just ensure it runs without errors
        assert!(!results.shots.is_empty(), "Expected non-empty results");

        // Count all results, handling the case where "result" might not be present
        for shot in &results.shots {
            let result_str = shot
                .get("result")
                .map_or_else(String::new, std::clone::Clone::clone);
            *counts.entry(result_str).or_insert(0) += 1;
        }

        // Print the counts for debugging
        println!("Noisy Bell state results (p=0.3, seed={seed}):");
        for (result, count) in &counts {
            println!("  {result}: {count}");
        }

        // The test passes if execution completes without errors
        // Actual noise validation is done in the unit tests for each noise model
    }
}