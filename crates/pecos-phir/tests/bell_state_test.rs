mod common;

use pecos_core::errors::PecosError;
use pecos_core::rng::RngManageable;
use pecos_engines::{DepolarizingNoiseModel, PassThroughNoiseModel};
use std::collections::HashMap;

// Import helpers from common module
use crate::common::phir_test_utils::run_phir_simulation_from_json;

#[test]
fn test_bell_state_noiseless() -> Result<(), PecosError> {
    // Define the Bell state PHIR program inline
    let bell_json = r#"{
      "format": "PHIR/JSON",
      "version": "0.1.0",
      "metadata": {"description": "Bell state preparation"},
      "ops": [
        {
          "data": "qvar_define",
          "data_type": "qubits",
          "variable": "q",
          "size": 2
        },
        {
          "data": "cvar_define",
          "data_type": "i64",
          "variable": "m",
          "size": 2
        },
        {"qop": "H", "args": [["q", 0]]},
        {"qop": "CX", "args": [["q", 0], ["q", 1]]},
        {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
        {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
        {"cop": "Result", "args": ["m"], "returns": ["v"]}
      ]
    }"#;

    // Run the Bell state example with 100 shots and 2 workers
    let results = run_phir_simulation_from_json(
        bell_json,
        100,
        2,
        None, // No specific seed
        None::<PassThroughNoiseModel>,
        None::<&std::path::Path>,
    )?;

    // Count occurrences of each result
    let mut counts: HashMap<String, usize> = HashMap::new();

    // Process results
    for shot in &results.shots {
        // If there's no "c" key in the output, just count it as an empty result
        let result_str = shot
            .get("v")
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

    println!("Results: {results:?}");

    Ok(())
}

#[test]
fn test_bell_state_using_helper() -> Result<(), PecosError> {
    // Define the Bell state PHIR program inline
    let bell_json = r#"{
      "format": "PHIR/JSON",
      "version": "0.1.0",
      "metadata": {"description": "Bell state preparation"},
      "ops": [
        {
          "data": "qvar_define",
          "data_type": "qubits",
          "variable": "q",
          "size": 2
        },
        {
          "data": "cvar_define",
          "data_type": "i64",
          "variable": "m",
          "size": 2
        },
        {"qop": "H", "args": [["q", 0]]},
        {"qop": "CX", "args": [["q", 0], ["q", 1]]},
        {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
        {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
        {"cop": "Result", "args": ["m"], "returns": ["c"]}
      ]
    }"#;

    // Run a single instance of the Bell state test
    let results = run_phir_simulation_from_json(
        bell_json,
        1,
        1,
        None, // No specific seed
        None::<PassThroughNoiseModel>,
        None::<&std::path::Path>,
    )?;

    // Print all information about the result for debugging
    println!("ShotResults: {results:?}");

    // Bell state should result in either 00 (0) or 11 (3) measurement outcomes
    // The bell.json file maps "m" to "c" in its Result command
    let shot = &results.shots[0];

    // First check for the "c" register which is specified in the Bell state JSON
    if let Some(value) = shot.get("c") {
        assert!(
            value == "0" || value == "3",
            "Expected Bell state result to be 0 or 3, got {value}"
        );
        return Ok(());
    }

    // Try fallback registers as well
    if let Some(value) = shot.get("result") {
        assert!(
            value == "0" || value == "3",
            "Expected Bell state result to be 0 or 3, got {value}"
        );
    } else if let Some(value) = shot.get("output") {
        assert!(
            value == "0" || value == "3",
            "Expected Bell state output to be 0 or 3, got {value}"
        );
    } else if let Some(value) = shot.get("m") {
        // The m register is the measurement register in bell.json
        assert!(
            value == "0" || value == "3",
            "Expected Bell state m register to be 0 or 3, got {value}"
        );
    } else {
        // No known register found - print available registers
        println!(
            "Available registers in shot: {:?}",
            shot.keys().collect::<Vec<_>>()
        );
        panic!("Expected one of 'c', 'result', 'output', or 'm' registers to be present");
    }

    Ok(())
}

#[allow(clippy::cast_precision_loss)]
#[test]
fn test_bell_state_with_noise() -> Result<(), PecosError> {
    // Define the Bell state PHIR program inline
    let bell_json = r#"{
      "format": "PHIR/JSON",
      "version": "0.1.0",
      "metadata": {"description": "Bell state preparation"},
      "ops": [
        {
          "data": "qvar_define",
          "data_type": "qubits",
          "variable": "q",
          "size": 2
        },
        {
          "data": "cvar_define",
          "data_type": "i64",
          "variable": "m",
          "size": 2
        },
        {"qop": "H", "args": [["q", 0]]},
        {"qop": "CX", "args": [["q", 0], ["q", 1]]},
        {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
        {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
        {"cop": "Result", "args": ["m"], "returns": ["c"]}
      ]
    }"#;

    // Try multiple runs with different seeds
    for seed in 1..=3 {
        println!("Attempting test with seed {seed}");

        // Create a noise model with 30% depolarizing noise
        let mut noise_model = DepolarizingNoiseModel::new_uniform(0.3);

        // Set the seed
        noise_model
            .set_seed(seed)
            .expect("Failed to set seed for noise model");

        // Run the Bell state example with high noise probability for more reliable testing
        let results = run_phir_simulation_from_json(
            bell_json,
            100, // 100 shots is enough for this simple test
            2,
            Some(seed), // Use the current iteration as seed
            Some(noise_model),
            None::<&std::path::Path>,
        )?;

        // Count occurrences of each result
        let mut counts: HashMap<String, usize> = HashMap::new();

        // For the noisy version, we just ensure it runs without errors
        assert!(!results.shots.is_empty(), "Expected non-empty results");

        // Count all results, handling the case where "c" might not be present
        for shot in &results.shots {
            let result_str = shot
                .get("c")
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

    Ok(())
}
