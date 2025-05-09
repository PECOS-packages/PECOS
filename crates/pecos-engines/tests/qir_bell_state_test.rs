use std::collections::HashMap;
use std::path::PathBuf;

use pecos_core::rng::RngManageable;
use pecos_engines::engines::MonteCarloEngine;
use pecos_engines::engines::qir::QirEngine;

/// Get the path to the QIR Bell state example
fn get_qir_program_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir.parent().unwrap().parent().unwrap();
    workspace_dir.join("examples/qir/bell.ll")
}

/// Check if LLVM llc tool version 14 is available
fn is_llc_available() -> bool {
    if cfg!(windows) {
        std::env::var("PATH")
            .map(|paths| {
                paths
                    .split(';')
                    .any(|dir| std::path::Path::new(dir).join("llc.exe").exists())
            })
            .unwrap_or(false)
    } else {
        std::env::var("PATH")
            .map(|paths| {
                paths
                    .split(':')
                    .any(|dir| std::path::Path::new(dir).join("llc").exists())
            })
            .unwrap_or(false)
    }
}

/// Skip the test with appropriate message if LLVM is not available
fn skip_if_llc_missing(test_name: &str) -> bool {
    if !is_llc_available() {
        println!("Skipping {test_name}: LLVM 'llc' tool not found");
        println!("To enable QIR tests, install LLVM version 14 (e.g., 'sudo apt install llvm-14')");
        return true;
    }
    false
}

#[test]
fn test_qir_bell_state_noiseless() {
    // Skip if LLVM is not available
    if skip_if_llc_missing("test_qir_bell_state_noiseless") {
        return;
    }

    // Create a QIR engine directly with the file path
    let qir_engine = QirEngine::new(get_qir_program_path());

    // Create a noiseless model
    let noise_model =
        Box::new(pecos_engines::engines::noise::DepolarizingNoiseModel::new_uniform(0.0));

    // Run the Bell state example with 100 shots and 2 workers
    let results = MonteCarloEngine::run_with_noise_model(
        Box::new(qir_engine),
        noise_model,
        100,
        2,
        None, // No specific seed
    )
    .expect("QIR execution should succeed as we already checked for LLVM availability");

    // Count occurrences of each result
    let mut counts: HashMap<String, usize> = HashMap::new();

    // Process results, handling the case where "result" might not be present
    for shot in &results.shots {
        // If there's no "result" key in the output, just count it as an empty result
        let result_str = shot
            .get("result")
            .map_or_else(String::new, std::clone::Clone::clone);
        *counts.entry(result_str).or_insert(0) += 1;
    }

    // Print the counts for debugging
    println!("Noiseless QIR Bell state results:");
    for (result, count) in &counts {
        println!("  {result}: {count}");
    }

    // The test passes if there are no errors in execution
    assert!(!results.shots.is_empty(), "Expected non-empty results");
}

#[test]
#[allow(clippy::missing_panics_doc)]
#[allow(clippy::cast_precision_loss)]
pub fn test_qir_bell_state_with_noise() {
    // Skip if LLVM is not available
    if skip_if_llc_missing("test_qir_bell_state_with_noise") {
        return;
    }

    // Try a few seeds
    for seed in 1..=3 {
        println!("Testing with seed: {seed}");

        let noise_probability = 0.3;
        let shots = 100;

        // Create QirEngine
        let qir_engine = QirEngine::new(get_qir_program_path());

        // Create a noise model with the specified probability
        let mut noise_model =
            pecos_engines::engines::noise::DepolarizingNoiseModel::new_uniform(noise_probability);

        // Set the seed on the noise model
        noise_model.set_seed(seed).unwrap();

        // Run with the MonteCarloEngine directly, specifying the number of shots
        let results = MonteCarloEngine::run_with_noise_model(
            Box::new(qir_engine),
            Box::new(noise_model),
            shots,
            2, // Number of workers
            Some(seed),
        )
        .expect("QIR execution should succeed as we already checked for LLVM availability");

        // Count results
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

        // Print counts for debugging
        println!("Counts with noise (seed {seed}):");
        for (result, count) in &counts {
            println!("  {result}: {count}");
        }

        // The test passes if execution completes without errors
        // Actual noise validation is done in the unit tests for the noise models
    }
}
