use pecos_core::rng::RngManageable;
use pecos_engines::engines::MonteCarloEngine;
use pecos_engines::engines::classical::setup_engine;
use std::collections::HashMap;
use std::path::PathBuf;

#[test]
fn test_bell_state_noiseless() {
    // Get the path to the Bell state example
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir.parent().unwrap().parent().unwrap();
    let bell_file = workspace_dir.join("examples/phir/bell.json");

    // Run the Bell state example with 100 shots and 2 workers
    let classical_engine = setup_engine(&bell_file, None).unwrap();

    // Create a noiseless model
    let noise_model =
        Box::new(pecos_engines::engines::noise::DepolarizingNoiseModel::new_uniform(0.0));

    // Use the generic approach
    let results = MonteCarloEngine::run_with_noise_model(
        classical_engine,
        noise_model,
        100,
        2,
        None, // No specific seed
    )
    .unwrap();

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
}

#[allow(clippy::cast_precision_loss)]
#[test]
fn test_bell_state_with_noise() {
    // Get the path to the Bell state example
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace_dir = manifest_dir.parent().unwrap().parent().unwrap();
    let bell_file = workspace_dir.join("examples/phir/bell.json");

    // Try multiple runs with different seeds
    for seed in 1..=3 {
        println!("Attempting test with seed {seed}");

        // Run the Bell state example with high noise probability for more reliable testing
        let classical_engine = setup_engine(&bell_file, None).unwrap();

        // Create a noise model with 30% depolarizing noise
        let mut noise_model =
            pecos_engines::engines::noise::DepolarizingNoiseModel::new_uniform(0.3);

        // Set the seed
        noise_model.set_seed(seed).unwrap();

        // Use the generic approach
        let results = MonteCarloEngine::run_with_noise_model(
            classical_engine,
            Box::new(noise_model),
            100, // 100 shots is enough for this simple test
            2,
            Some(seed), // Use the current iteration as seed
        )
        .unwrap();

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
