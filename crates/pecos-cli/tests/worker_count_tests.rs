/// # Worker Count Tests
///
/// This file contains tests that verify deterministic behavior across different
/// worker count configurations in the PECOS CLI. Key aspects tested include:
///
/// 1. Self-Determinism: Each worker count should be deterministic with respect to itself
///    when run with the same seed
///
/// 2. Small Shot Counts: Tests with small shot counts (10) and various worker counts (1, 5, 10)
///    to ensure deterministic behavior even in edge cases
///
/// 3. Worker Count Effects: Analyzing how different worker counts may produce different
///    distributions due to parallelization differences
///
/// These tests help ensure that the PECOS simulator maintains proper deterministic
/// behavior regardless of the parallelization configuration.
use assert_cmd::prelude::*;
use pecos::prelude::*;
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
    let output = Command::cargo_bin("pecos")?
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
        .arg("-f")
        .arg("pretty-compact") // Force consistent format for test
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

/// Extract measurement results as arrays from JSON output
fn get_values(json_output: &str) -> Vec<String> {
    let mut values = Vec::new();

    // Try to parse the JSON using serde_json, which is the most reliable method
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_output) {
        if let Some(obj) = json.as_object() {
            for (_, value) in obj {
                if let Some(array) = value.as_array() {
                    // Convert the array to a string representation
                    let value_str = array
                        .iter()
                        .map(|v| v.to_string().replace('"', ""))
                        .collect::<Vec<_>>()
                        .join(", ");
                    values.push(value_str);
                }
            }
            values.sort();
            return values;
        }
    }

    // Fallback to manual parsing if serde_json fails
    let mut in_array = false;
    let mut current_array = String::new();

    for line in json_output.lines() {
        let trimmed = line.trim();

        // Start of an array
        if trimmed.contains('[') {
            in_array = true;
            current_array = trimmed
                .chars()
                .skip_while(|&c| c != '[')
                .skip(1) // Skip the '['
                .collect();
            // If the array ends on the same line
            if trimmed.contains(']') {
                in_array = false;
                current_array = current_array.chars().take_while(|&c| c != ']').collect();
                values.push(current_array.trim().to_string());
                current_array = String::new();
            }
        }
        // End of an array
        else if in_array && trimmed.contains(']') {
            in_array = false;
            current_array.push_str(
                &trimmed
                    .chars()
                    .take_while(|&c| c != ']')
                    .collect::<String>(),
            );
            values.push(current_array.trim().to_string());
            current_array = String::new();
        }
        // Middle of an array
        else if in_array {
            current_array.push_str(trimmed);
        }
    }

    // Sort for stable comparison
    values.sort();
    values
}

/// Test that each worker count configuration is deterministic with itself
/// (i.e., same seed and workers always produces the same results)
#[test]
fn test_worker_count_self_determinism() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.json");

    println!("WORKER COUNT SELF-DETERMINISM: Testing that each worker count is self-consistent");
    println!("----------------------------------------------------------------------------");

    // Test with 1 worker - with noise
    println!("Testing 1 worker with p=0.1 noise:");
    let single_worker_run1 = run_pecos(&bell_json_path, 100, 1, "depolarizing", "0.1", 42)?;
    let single_worker_run2 = run_pecos(&bell_json_path, 100, 1, "depolarizing", "0.1", 42)?;

    let values_1w_run1 = get_values(&single_worker_run1);
    let values_1w_run2 = get_values(&single_worker_run2);

    assert_eq!(
        values_1w_run1, values_1w_run2,
        "Results should be deterministic for single worker"
    );
    println!("1 worker configuration is deterministic");

    // Test with 2 workers - with noise
    println!("\nTesting 2 workers with p=0.1 noise:");
    let two_workers_run1 = run_pecos(&bell_json_path, 100, 2, "depolarizing", "0.1", 42)?;
    let two_workers_run2 = run_pecos(&bell_json_path, 100, 2, "depolarizing", "0.1", 42)?;

    let values_2w_run1 = get_values(&two_workers_run1);
    let values_2w_run2 = get_values(&two_workers_run2);

    assert_eq!(
        values_2w_run1, values_2w_run2,
        "Results should be deterministic for two workers"
    );
    println!("2 worker configuration is deterministic");

    // Test with 4 workers - with noise
    println!("\nTesting 4 workers with p=0.1 noise:");
    let four_workers_run1 = run_pecos(&bell_json_path, 100, 4, "depolarizing", "0.1", 42)?;
    let four_workers_run2 = run_pecos(&bell_json_path, 100, 4, "depolarizing", "0.1", 42)?;

    let values_4w_run1 = get_values(&four_workers_run1);
    let values_4w_run2 = get_values(&four_workers_run2);

    assert_eq!(
        values_4w_run1, values_4w_run2,
        "Results should be deterministic for four workers"
    );
    println!("4 worker configuration is deterministic");

    Ok(())
}

/// Test with small number of shots (10) and verify behavior with different worker counts,
/// both with and without noise
#[test]
#[allow(clippy::similar_names)]
fn test_small_shots_with_multiple_workers() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.json");

    println!("SMALL SHOT COUNT TEST: Verifying behavior with 10 shots and various worker counts");
    println!("------------------------------------------------------------------------");
    println!("This test verifies that each worker configuration is self-deterministic");
    println!("even with small shot counts, and analyzes the effects of worker parallelization");

    // ------------------------
    // Tests with noise (0.1)
    // ------------------------
    println!("\nTests with noise (p=0.1):");

    // 1 worker, with noise
    let w1_noise_run1 = run_pecos(&bell_json_path, 10, 1, "depolarizing", "0.1", 42)?;
    let w1_noise_run2 = run_pecos(&bell_json_path, 10, 1, "depolarizing", "0.1", 42)?;

    let w1_noise_values1 = get_values(&w1_noise_run1);
    let w1_noise_values2 = get_values(&w1_noise_run2);

    assert_eq!(
        w1_noise_values1, w1_noise_values2,
        "10 shots with 1 worker (with noise) should be deterministic with same seed"
    );
    println!("1 worker with noise: deterministic with same seed");

    // Run with different seed to verify different results
    let w1_noise_diff_seed = run_pecos(&bell_json_path, 10, 1, "depolarizing", "0.1", 43)?;
    let w1_noise_diff_values = get_values(&w1_noise_diff_seed);

    if w1_noise_values1 != w1_noise_diff_values {
        println!("1 worker with noise: different seeds produce different results");
    }

    // 5 workers, with noise
    let w5_noise_run1 = run_pecos(&bell_json_path, 10, 5, "depolarizing", "0.1", 42)?;
    let w5_noise_run2 = run_pecos(&bell_json_path, 10, 5, "depolarizing", "0.1", 42)?;

    let w5_noise_values1 = get_values(&w5_noise_run1);
    let w5_noise_values2 = get_values(&w5_noise_run2);

    assert_eq!(
        w5_noise_values1, w5_noise_values2,
        "10 shots with 5 workers (with noise) should be deterministic with same seed"
    );
    println!("5 workers with noise: deterministic with same seed");

    // 10 workers, with noise (more workers than shots!)
    let w10_noise_run1 = run_pecos(&bell_json_path, 10, 10, "depolarizing", "0.1", 42)?;
    let w10_noise_run2 = run_pecos(&bell_json_path, 10, 10, "depolarizing", "0.1", 42)?;

    let w10_noise_values1 = get_values(&w10_noise_run1);
    let w10_noise_values2 = get_values(&w10_noise_run2);

    assert_eq!(
        w10_noise_values1, w10_noise_values2,
        "10 shots with 10 workers (with noise) should be deterministic with same seed"
    );
    println!("10 workers with noise: deterministic with same seed");

    // ------------------------
    // Tests without noise (0.0)
    // ------------------------
    println!("\nTests without noise (p=0.0):");

    // 1 worker, without noise
    let w1_no_noise_run1 = run_pecos(&bell_json_path, 10, 1, "depolarizing", "0.0", 42)?;
    let w1_no_noise_run2 = run_pecos(&bell_json_path, 10, 1, "depolarizing", "0.0", 42)?;

    let w1_no_noise_values1 = get_values(&w1_no_noise_run1);
    let w1_no_noise_values2 = get_values(&w1_no_noise_run2);

    assert_eq!(
        w1_no_noise_values1, w1_no_noise_values2,
        "10 shots with 1 worker (without noise) should be deterministic with same seed"
    );
    println!("1 worker without noise: deterministic with same seed");

    // Try different seeds without noise
    // Note: While theoretically no-noise should produce identical results regardless of seed,
    // there might still be RNG usage in the codebase (like for initial state prep)
    // that causes different results with different seeds even with 0 noise probability.
    let w1_no_noise_diff_seed = run_pecos(&bell_json_path, 10, 1, "depolarizing", "0.0", 43)?;
    let w1_no_noise_diff_values = get_values(&w1_no_noise_diff_seed);

    if w1_no_noise_values1 == w1_no_noise_diff_values {
        println!("Without noise: different seeds still produce the same results");
    } else {
        println!(
            "Without noise: different seeds produced different results (this may be normal if seed impacts execution beyond noise)"
        );
    }

    // 5 workers, without noise
    let w5_no_noise_run1 = run_pecos(&bell_json_path, 10, 5, "depolarizing", "0.0", 42)?;
    let w5_no_noise_run2 = run_pecos(&bell_json_path, 10, 5, "depolarizing", "0.0", 42)?;

    let w5_no_noise_values1 = get_values(&w5_no_noise_run1);
    let w5_no_noise_values2 = get_values(&w5_no_noise_run2);

    assert_eq!(
        w5_no_noise_values1, w5_no_noise_values2,
        "10 shots with 5 workers (without noise) should be deterministic with same seed"
    );
    println!("5 workers without noise: deterministic with same seed");

    // 10 workers, without noise (more workers than shots!)
    let w10_no_noise_run1 = run_pecos(&bell_json_path, 10, 10, "depolarizing", "0.0", 42)?;
    let w10_no_noise_run2 = run_pecos(&bell_json_path, 10, 10, "depolarizing", "0.0", 42)?;

    let w10_no_noise_values1 = get_values(&w10_no_noise_run1);
    let w10_no_noise_values2 = get_values(&w10_no_noise_run2);

    assert_eq!(
        w10_no_noise_values1, w10_no_noise_values2,
        "10 shots with 10 workers (without noise) should be deterministic with same seed"
    );
    println!("10 workers without noise: deterministic with same seed");

    // Check if different worker counts produce the same results without noise
    // Note: Even without noise, initial random state preparation or how the workload is
    // distributed among workers might cause differences in results
    if w1_no_noise_values1 == w5_no_noise_values1 && w1_no_noise_values1 == w10_no_noise_values1 {
        println!("All worker counts without noise produce identical results");
    } else {
        println!("Different worker counts without noise produced different results");
        println!(
            "  (This is expected if worker count affects random number generation or state preparation)"
        );
        println!("  1 worker: {w1_no_noise_values1:?}");
        println!("  5 workers: {w5_no_noise_values1:?}");
        println!("  10 workers: {w10_no_noise_values1:?}");
    }

    // Check if different worker counts with noise produce different results
    if w1_noise_values1 != w5_noise_values1
        || w1_noise_values1 != w10_noise_values1
        || w5_noise_values1 != w10_noise_values1
    {
        println!(
            "Different worker counts with noise produce different results (expected behavior)"
        );
    } else {
        println!(
            "Note: All worker counts with noise produced identical results (somewhat unexpected)"
        );
    }

    Ok(())
}
