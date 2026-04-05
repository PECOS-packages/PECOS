use pecos::prelude::*;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

/// Helper function to run PECOS CLI with given parameters
fn run_pecos(
    file_path: &PathBuf,
    shots: usize,
    workers: usize,
    noise_model: &str,
    noise_prob: &str,
    seed: u64,
) -> Result<String, Box<dyn std::error::Error>> {
    let output = Command::new(assert_cmd::cargo::cargo_bin!("pecos"))
        .env("RUST_LOG", "info")
        .arg("run")
        .arg(file_path)
        .arg("-s")
        .arg(shots.to_string())
        .arg("-w")
        .arg(workers.to_string())
        .arg("-m")
        .arg(noise_model)
        .arg("-p")
        .arg(noise_prob)
        .arg("-d")
        .arg(seed.to_string())
        .output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);

        // Provide more context about the error
        return Err(Box::new(PecosError::Resource(format!(
            "PECOS run failed for file '{}' with settings (shots={}, workers={}, model={}, noise={}, seed={}): {}",
            file_path.display(),
            shots,
            workers,
            noise_model,
            noise_prob,
            seed,
            stderr
        ))));
    }

    let output_str = String::from_utf8(output.stdout).map_err(|e| {
        Box::new(PecosError::Resource(format!("Failed to parse output: {e}")))
            as Box<dyn std::error::Error>
    })?;

    Ok(output_str)
}

/// Extract measurement results from JSON output
/// Handles the new columnar format: {"c": [3, 0, ...]}
fn get_values(json_output: &str) -> Vec<String> {
    let mut register_values: BTreeMap<String, Vec<String>> = BTreeMap::new();

    // Parse the JSON - expecting an object with register names as keys
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_output)
        && let Some(obj) = json.as_object()
    {
        // For each register, collect its values
        for (reg_name, values) in obj {
            if let Some(arr) = values.as_array() {
                let string_values: Vec<String> =
                    arr.iter().map(|v| v.to_string().replace('"', "")).collect();
                register_values.insert(reg_name.clone(), string_values);
            }
        }
    }

    // Convert to the format expected by tests: comma-separated values per register
    let mut result = Vec::new();
    for (_, values) in register_values {
        let value_str = values.join(", ");
        result.push(value_str);
    }

    result.sort();
    result
}

/// Test that our circuit produces deterministic results with the same seed
#[test]
fn test_circuit_determinism() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let phir_path = manifest_dir.join("../../examples/phir/simple_test.phir.json");

    println!("DETERMINISM TEST: Verifying consistent results with same seed");
    println!("----------------------------------------------------------");

    // Run multiple times with the same seed and verify results are identical
    let output1 = run_pecos(&phir_path, 1, 1, "depolarizing", "0.0", 42)?;
    let output2 = run_pecos(&phir_path, 1, 1, "depolarizing", "0.0", 42)?;
    let output3 = run_pecos(&phir_path, 1, 1, "depolarizing", "0.0", 42)?;

    println!("Run 1 results: {}", output1.trim());
    println!("Run 2 results: {}", output2.trim());
    println!("Run 3 results: {}", output3.trim());

    // Compare outputs
    let values1 = get_values(&output1);
    let values2 = get_values(&output2);
    let values3 = get_values(&output3);

    // All runs should produce identical results with the same seed
    assert_eq!(
        values1, values2,
        "Runs with the same seed should produce identical results"
    );
    assert_eq!(
        values2, values3,
        "Runs with the same seed should produce identical results"
    );

    println!("Circuit produces deterministic results with the same seed");

    // Now verify that different seeds produce different results
    let output_seed1 = run_pecos(&phir_path, 5, 1, "depolarizing", "0.0", 42)?;
    let output_seed2 = run_pecos(&phir_path, 5, 1, "depolarizing", "0.0", 13)?;

    println!("Seed 42 results: {:.60}...", output_seed1.trim());
    println!("Seed 13 results: {:.60}...", output_seed2.trim());

    let values_seed1 = get_values(&output_seed1);
    let values_seed2 = get_values(&output_seed2);

    // With multiple shots, even in a deterministic circuit, different seeds should
    // produce different measurement statistics due to how RNG affects the simulation
    // This is expected even though our circuit is deterministic in a single shot

    // Note: This assertion could sometimes fail if by chance both seeds happen to produce
    // the same results, but with 5 shots it's extremely unlikely
    assert_ne!(
        values_seed1, values_seed2,
        "Different seeds should produce different results with multiple shots"
    );

    println!("Different seeds produce different results as expected");

    Ok(())
}

/// Test that both PHIR and QASM implementations produce the same results
#[test]
fn test_cross_implementation_consistency() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let phir_path = manifest_dir.join("../../examples/phir/simple_test.phir.json");
    let qasm_path = manifest_dir.join("../../examples/qasm/simple_test.qasm");

    println!("CROSS-IMPLEMENTATION TEST: Checking PHIR and QASM produce consistent results");
    println!("---------------------------------------------------------------------");

    // Run both implementations with same seed and settings
    let phir_output = run_pecos(&phir_path, 1, 1, "depolarizing", "0.0", 42)?;
    let qasm_output = run_pecos(&qasm_path, 1, 1, "depolarizing", "0.0", 42)?;

    println!("PHIR results: {}", phir_output.trim());
    println!("QASM results: {}", qasm_output.trim());

    // Extract values
    let phir_values = get_values(&phir_output);
    let qasm_values = get_values(&qasm_output);

    // The implementations should produce identical results
    assert_eq!(
        phir_values, qasm_values,
        "PHIR and QASM implementations should produce identical results"
    );

    println!("PHIR and QASM implementations are consistent");

    Ok(())
}

/// Test how noise affects determinism
#[test]
fn test_noise_impact_on_determinism() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let phir_path = manifest_dir.join("../../examples/phir/simple_test.phir.json");

    println!("NOISE IMPACT TEST: Analyzing how noise affects deterministic behavior");
    println!("----------------------------------------------------------------");

    // Run with 10 shots and no noise, with same seed
    let noiseless_run1 = run_pecos(&phir_path, 10, 1, "depolarizing", "0.0", 42)?;
    let noiseless_run2 = run_pecos(&phir_path, 10, 1, "depolarizing", "0.0", 42)?;

    // Run with 10 shots and noise, with same seed
    let noisy_run1 = run_pecos(&phir_path, 10, 1, "depolarizing", "0.1", 42)?;
    let noisy_run2 = run_pecos(&phir_path, 10, 1, "depolarizing", "0.1", 42)?;

    println!("Noiseless run 1: {:.60}...", noiseless_run1.trim());
    println!("Noiseless run 2: {:.60}...", noiseless_run2.trim());
    println!("Noisy run 1: {:.60}...", noisy_run1.trim());
    println!("Noisy run 2: {:.60}...", noisy_run2.trim());

    // Extract values
    let noiseless_values1 = get_values(&noiseless_run1);
    let noiseless_values2 = get_values(&noiseless_run2);
    let noisy_values1 = get_values(&noisy_run1);
    let noisy_values2 = get_values(&noisy_run2);

    // Noiseless runs should be identical with the same seed
    assert_eq!(
        noiseless_values1, noiseless_values2,
        "Noiseless runs with the same seed should be identical"
    );

    // Noisy runs should also be identical with the same seed
    // This confirms that noise application is deterministic when using the same seed
    assert_eq!(
        noisy_values1, noisy_values2,
        "Noise application should be deterministic with the same seed"
    );

    // Noiseless and noisy runs should differ
    assert_ne!(
        noiseless_values1, noisy_values1,
        "Noiseless and noisy runs should produce different results"
    );

    println!("Noise application is deterministic with fixed seeds");
    println!("Noise changes the output distribution as expected");

    Ok(())
}

/// Test that each worker count produces deterministic results
///
/// NOTE: Different worker counts will produce different (but deterministic) results.
/// This is intentional - each worker gets its own RNG stream for optimal parallelization
#[test]
fn test_worker_count_consistency() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let phir_path = manifest_dir.join("../../examples/phir/simple_test.phir.json");

    println!("WORKER COUNT DETERMINISM TEST");
    println!("----------------------------------------------------------------------");
    println!("Verifying that results are deterministic for each worker count.");
    println!("Note: Different worker counts intentionally produce different results.");

    // Run with different worker counts but the same seed
    let single_worker = run_pecos(&phir_path, 10, 1, "depolarizing", "0.0", 42)?;
    let multi_worker = run_pecos(&phir_path, 10, 4, "depolarizing", "0.0", 42)?;

    println!("Single worker results: {:.60}...", single_worker.trim());
    println!("Multiple worker results: {:.60}...", multi_worker.trim());

    // Extract values
    let single_values = get_values(&single_worker);
    let multi_values = get_values(&multi_worker);

    // Verify each worker count produces deterministic results
    let single_worker_2 = run_pecos(&phir_path, 10, 1, "depolarizing", "0.0", 42)?;
    let multi_worker_2 = run_pecos(&phir_path, 10, 4, "depolarizing", "0.0", 42)?;

    let single_values_2 = get_values(&single_worker_2);
    let multi_values_2 = get_values(&multi_worker_2);

    // These SHOULD be equal - same seed and worker count
    assert_eq!(
        single_values, single_values_2,
        "Single worker should produce deterministic results"
    );
    assert_eq!(
        multi_values, multi_values_2,
        "Multi-worker should produce deterministic results"
    );

    println!("Single worker: deterministic with seed");
    println!("Multi-worker: deterministic with seed");

    // Different worker counts producing different results is EXPECTED
    if single_values != multi_values {
        println!("Different worker counts produce different results (expected behavior)");
    }

    println!("Worker count determinism test passed");

    Ok(())
}
