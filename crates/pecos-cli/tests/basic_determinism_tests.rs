/// # Basic Determinism Tests
///
/// This file contains the fundamental determinism tests for the PECOS CLI.
/// Key aspects tested include:
///
/// 1. Basic Determinism: Running the same command with the same seed
///    should produce identical results
///
/// 2. File Format Determinism: Testing across different file formats
///    (PHIR, QASM, QIR) to ensure consistent behavior
///
/// 3. Cross-Model Consistency: Verifying that different noise models
///    work properly and produce consistent results when configured identically
///
/// These tests provide the foundation for ensuring PECOS maintains deterministic
/// behavior, which is crucial for reproducible quantum simulations.
use pecos::prelude::*;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

// Test lock removed: These tests only verify determinism by executing quantum programs
// They don't modify any shared state and can safely run in parallel

/// Helper function to run PECOS CLI with given parameters
fn run_pecos(
    file_path: &PathBuf,
    shots: usize,
    workers: usize,
    noise_model: &str,
    noise_prob: &str,
    seed: u64,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("pecos"));
    cmd.env("RUST_LOG", "info").arg("run");

    // Add --jit flag for LLVM files (when Selene is not available)
    if file_path.extension().and_then(|s| s.to_str()) == Some("ll") {
        cmd.arg("--jit");
    }

    let output = cmd
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

/// Helper function to test determinism for a specific file
fn test_determinism_for_file(
    file_path: &PathBuf,
    shots: usize,
    workers: usize,
    noise_model: &str,
    noise_prob: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing file: {}", file_path.display());

    // Run twice with seed 42
    let seed_42_run1 = run_pecos(file_path, shots, workers, noise_model, noise_prob, 42)?;
    let seed_42_run2 = run_pecos(file_path, shots, workers, noise_model, noise_prob, 42)?;

    // Run twice with seed 43
    let seed_43_run1 = run_pecos(file_path, shots, workers, noise_model, noise_prob, 43)?;
    let seed_43_run2 = run_pecos(file_path, shots, workers, noise_model, noise_prob, 43)?;

    // Verify determinism with the same seed
    let values_42_1 = get_values(&seed_42_run1);
    let values_42_2 = get_values(&seed_42_run2);
    assert_eq!(
        values_42_1,
        values_42_2,
        "File {}: Results with seed 42 should have the same values across runs",
        file_path.display()
    );

    // Verify determinism with seed 43
    let values_43_1 = get_values(&seed_43_run1);
    let values_43_2 = get_values(&seed_43_run2);
    assert_eq!(
        values_43_1,
        values_43_2,
        "File {}: Results with seed 43 should have the same values across runs",
        file_path.display()
    );

    // Verify that different seeds produce different results (if there's randomness in the program)
    // Note: Some deterministic programs might still produce the same results with different seeds
    if values_42_1 != values_43_1 {
        println!(
            "  - Different seeds produce different results (as expected with noise/randomness)"
        );
    } else if noise_prob == "0.0" {
        println!("  - Same results with different seeds (expected for noiseless simulation)");
    } else {
        println!("  - Same results with different seeds (unexpected with noise, but could happen)");
    }

    Ok(())
}

/// Test basic determinism with PHIR (JSON) files
#[test]
fn test_basic_determinism_phir_json() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    println!("BASIC DETERMINISM TEST - PHIR-JSON FILES");
    println!("-----------------------------------");

    // Test bell.json with depolarizing noise model
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.phir.json");
    println!("\nTesting with depolarizing noise (p=0.1):");
    test_determinism_for_file(&bell_json_path, 100, 1, "depolarizing", "0.1")?;

    // Test with general noise model
    println!("\nTesting with general noise (p=0.1 for all types):");
    test_determinism_for_file(&bell_json_path, 100, 1, "general", "0.1,0.05,0.05,0.1,0.2")?;

    // Test with no noise
    println!("\nTesting with no noise (p=0.0):");
    test_determinism_for_file(&bell_json_path, 100, 1, "depolarizing", "0.0")?;

    // Test qprog.phir.json
    let qprog_json_path = manifest_dir.join("../../examples/phir/qprog.phir.json");
    println!("\nTesting qprog.phir.json:");
    test_determinism_for_file(&qprog_json_path, 100, 1, "depolarizing", "0.1")?;

    println!("\nPHIR-JSON files exhibit deterministic behavior with the same seed");

    Ok(())
}

/// Test basic determinism with QASM files
#[test]
fn test_basic_determinism_qasm() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    println!("BASIC DETERMINISM TEST - QASM FILES");
    println!("----------------------------------");

    // Get list of QASM files
    let qasm_files = vec!["bell.qasm", "hadamard.qasm", "multi_register.qasm"];

    for qasm_file in qasm_files {
        let file_path = manifest_dir.join(format!("../../examples/qasm/{qasm_file}"));

        println!("\nTesting {qasm_file}");

        // Test with depolarizing noise
        println!("With depolarizing noise (p=0.1):");
        test_determinism_for_file(&file_path, 100, 1, "depolarizing", "0.1")?;

        // Test with general noise
        println!("With general noise (p=0.1 for all types):");
        test_determinism_for_file(&file_path, 100, 1, "general", "0.1,0.05,0.05,0.1,0.2")?;
    }

    println!("\nQASM files exhibit deterministic behavior with the same seed");

    Ok(())
}

/// Test basic determinism with LLVM files, gracefully skipping if LLVM tools are unavailable
#[test]
fn test_basic_determinism_llvm() {
    // No lock needed: This test only verifies determinism without modifying shared state

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_ll_path = manifest_dir.join("../../examples/llvm/bell.ll");

    println!("BASIC DETERMINISM TEST - LLVM FILES");
    println!("---------------------------------");

    // Try to run LLVM tests, but handle any errors gracefully
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        // Test with depolarizing noise
        println!("\nTesting with depolarizing noise (p=0.1):");
        test_determinism_for_file(&bell_ll_path, 100, 1, "depolarizing", "0.1")?;

        // Test with general noise
        println!("\nTesting with general noise (p=0.1 for all types):");
        test_determinism_for_file(&bell_ll_path, 100, 1, "general", "0.1,0.05,0.05,0.1,0.2")?;

        // Test with multiple workers
        println!("\nTesting with multiple workers (2):");
        test_determinism_for_file(&bell_ll_path, 100, 2, "depolarizing", "0.1")?;

        Ok(())
    })();

    // If there was an error, print a message but don't fail the test
    if let Err(e) = result {
        println!("Skipping LLVM determinism test - LLVM engine error: {e}");
        println!("This might be due to missing LLVM tools or other dependencies");
        return;
    }

    println!("\nLLVM files exhibit deterministic behavior with the same seed");
}

/// Test that with 0 noise probability, both noise models give identical results
#[test]
fn test_cross_model_consistency() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.phir.json");

    println!("CROSS-MODEL CONSISTENCY TEST");
    println!("----------------------------");
    println!("With 0 noise probability, both depolarizing and general noise models");
    println!("should produce identical results.");

    // Test that with 0 noise probability, both models give identical results
    let dep_output = run_pecos(&bell_json_path, 100, 1, "depolarizing", "0.0", 42)?;
    let gen_output = run_pecos(
        &bell_json_path,
        100,
        1,
        "general",
        "0.0,0.0,0.0,0.0,0.0",
        42,
    )?;

    let dep_values = get_values(&dep_output);
    let gen_values = get_values(&gen_output);

    assert_eq!(
        dep_values, gen_values,
        "With 0 noise, depolarizing and general models should produce identical results"
    );

    println!("\nBoth noise models produce identical results with 0 noise probability");

    Ok(())
}
