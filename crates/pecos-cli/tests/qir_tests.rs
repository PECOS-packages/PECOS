/// # QIR Tests
///
/// This file contains comprehensive tests for QIR (Quantum Intermediate Representation)
/// functionality in the PECOS simulator. These tests ensure that QIR programs:
///
/// 1. Produce correct quantum mechanical behavior (e.g., Bell state distributions)
/// 2. Generate deterministic results with the same seed
/// 3. Work correctly with various noise models
/// 4. Produce results consistent with PHIR and QASM implementations
///
/// Note: These tests require QIR compilation capabilities which depend on
/// LLVM toolchain availability. If tests fail due to missing dependencies,
/// ensure that the LLVM toolchain is properly installed.
use assert_cmd::prelude::*;
use pecos::prelude::*;
use std::collections::HashMap;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Mutex;
use std::sync::Once;
use std::time::Duration;

// Create a static mutex to ensure tests run sequentially
// This prevents race conditions when multiple tests try to access shared resources
static TEST_MUTEX: Mutex<()> = Mutex::new(());

// Static variable for test initialization
static INIT: Once = Once::new();

// Setup function for cleaning up any leftover files from previous test runs
fn setup() {
    // Run this initialization only once, for all tests
    INIT.call_once(|| {
        println!("Initializing QIR test environment...");

        // Clean up any temporary directories from previous test runs
        let temp_dir = std::env::temp_dir();
        let entries = match std::fs::read_dir(&temp_dir) {
            Ok(entries) => entries,
            Err(e) => {
                println!("Warning: Could not read temporary directory: {e}");
                return;
            }
        };

        // Use flatten() to simplify the iterator chain and handle Result automatically
        for entry in entries.flatten() {
            let path = entry.path();
            // Use and_then to chain Optional operations cleanly
            if let Some(name) = path.file_name().and_then(|f| f.to_str()) {
                // Only remove directories that match our QIR pattern
                if name.starts_with("qir_") && path.is_dir() {
                    println!("Cleaning up old temporary directory: {}", path.display());
                    let _ = std::fs::remove_dir_all(path);
                }
            }
        }

        // Give file system operations time to complete
        std::thread::sleep(Duration::from_millis(500));
        println!("Test environment initialized");
    });
}

/// Helper function to run PECOS CLI with given parameters
fn run_pecos(
    file_path: &PathBuf,
    shots: usize,
    workers: usize,
    noise_model: &str,
    noise_prob: &str,
    seed: u64,
) -> Result<String, Box<dyn std::error::Error>> {
    // Add a small delay between test executions to prevent potential file system races
    std::thread::sleep(Duration::from_millis(100));
    let mut cmd = Command::cargo_bin("pecos")?;
    cmd.env("RUST_LOG", "info")
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
        .arg(seed.to_string());

    let output = cmd.output()?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(Box::new(PecosError::Resource(format!(
            "PECOS run failed for QIR file '{}' with settings (shots={}, workers={}, model={}, noise={}, seed={}): {}",
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
    let mut register_values: HashMap<String, Vec<String>> = HashMap::new();

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

/// Test that QIR Bell state produces correct 50/50 distribution
#[test]
fn test_qir_bell_state_distribution() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize test environment and acquire lock to ensure sequential execution
    setup();
    let _lock = TEST_MUTEX.lock().unwrap();
    println!("Running QIR Bell state distribution test (sequential execution)...");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_qir_path = manifest_dir.join("../../examples/qir/bell.ll");

    println!("QIR BELL STATE TEST: Verifying correct quantum mechanical behavior");
    println!("-----------------------------------------------------------------");

    // Run QIR Bell state simulation
    let output = run_pecos(&bell_qir_path, 100, 1, "depolarizing", "0.0", 42)?;
    println!("QIR Bell state results: {}", output.trim());

    // Count occurrences of each measurement outcome
    let values = get_values(&output);
    if values.len() != 1 {
        return Err(Box::new(PecosError::Resource(format!(
            "Expected 1 register with values, got {}",
            values.len()
        ))));
    }

    let outcomes = values[0].split(", ").collect::<Vec<_>>();
    let mut counts = HashMap::new();

    for outcome in &outcomes {
        *counts.entry(*outcome).or_insert(0) += 1;
    }

    // Print the distribution of outcomes
    println!("QIR outcome distribution:");
    let mut total_outcomes = 0;
    let mut state_00_count = 0;
    let mut state_11_count = 0;

    for (outcome, count) in &counts {
        println!(
            "  |{:02b}⟩ ({}): {} times ({}%)",
            outcome.parse::<i32>().unwrap_or(0),
            outcome,
            count,
            (count * 100) / outcomes.len()
        );
        total_outcomes += count;

        if outcome == &"0" {
            state_00_count = *count;
        } else if outcome == &"3" {
            state_11_count = *count;
        }
    }

    // Verify Bell state behavior
    let expected_states_count = state_00_count + state_11_count;
    println!(
        "  |00⟩ and |11⟩ states: {} out of {} ({}%)",
        expected_states_count,
        total_outcomes,
        (expected_states_count * 100) / total_outcomes
    );

    // Bell state should have 100% of outcomes being either |00⟩ or |11⟩
    assert_eq!(
        expected_states_count,
        total_outcomes,
        "Expected all outcomes to be |00⟩ or |11⟩, but got {}%",
        (expected_states_count * 100) / total_outcomes
    );

    // Check for balanced distribution
    if state_00_count > 0 && state_11_count > 0 {
        let ratio_00 = (state_00_count * 100) / expected_states_count;
        let ratio_11 = (state_11_count * 100) / expected_states_count;

        println!("  |00⟩ to |11⟩ ratio: {ratio_00}% to {ratio_11}%");

        assert!(
            (40..=60).contains(&ratio_00),
            "Expected |00⟩ probability between 40% and 60%, but got {ratio_00}%"
        );

        println!("QIR Bell state probabilities are correctly balanced");
    } else {
        return Err(Box::new(PecosError::Resource(
            "Missing either |00⟩ or |11⟩ state in QIR Bell state simulation".to_string(),
        )));
    }

    Ok(())
}

/// Test that QIR produces deterministic results with the same seed
#[test]
fn test_qir_determinism() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize test environment and acquire lock to ensure sequential execution
    setup();
    let _lock = TEST_MUTEX.lock().unwrap();
    println!("Running QIR determinism test (sequential execution)...");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_qir_path = manifest_dir.join("../../examples/qir/bell.ll");

    println!("QIR DETERMINISM TEST: Verifying reproducible results with same seed");
    println!("------------------------------------------------------------------");

    // Run QIR program twice with same seed
    let run1 = run_pecos(&bell_qir_path, 50, 1, "depolarizing", "0.0", 42)?;
    let run2 = run_pecos(&bell_qir_path, 50, 1, "depolarizing", "0.0", 42)?;

    let values1 = get_values(&run1);
    let values2 = get_values(&run2);

    assert_eq!(
        values1, values2,
        "QIR should produce identical results with the same seed"
    );

    println!("QIR produces deterministic results with the same seed");

    // Test with different seeds produces different results
    let run3 = run_pecos(&bell_qir_path, 50, 1, "depolarizing", "0.0", 123)?;
    let values3 = get_values(&run3);

    assert_ne!(
        values1, values3,
        "QIR should produce different results with different seeds"
    );

    println!("QIR produces different results with different seeds");

    Ok(())
}

/// Test QIR compilation and execution
#[test]
fn test_qir_compile_and_run() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize test environment and acquire lock to ensure sequential execution
    setup();
    let _lock = TEST_MUTEX.lock().unwrap();
    println!("Running QIR compilation and execution test (sequential execution)...");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_file = manifest_dir.join("../../examples/qir/qprog.ll");

    // Remove the cached library to ensure we see compilation messages
    let build_dir = manifest_dir.join("../../examples/qir/build");
    if build_dir.exists() {
        let _ = std::fs::remove_dir_all(&build_dir);
    }

    // First, test compilation
    let output = Command::cargo_bin("pecos")?
        .env("RUST_LOG", "info")
        .arg("compile")
        .arg(&test_file)
        .output()?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        output.status.success(),
        "Compilation should succeed. Error: {stderr}"
    );

    // Verify compilation worked by checking logs
    assert!(
        stderr.contains("Starting compilation") || stderr.contains("Compilation successful"),
        "Should show compilation activity"
    );

    // Then, test execution
    let output = Command::cargo_bin("pecos")?
        .env("RUST_LOG", "info")
        .arg("run")
        .arg(&test_file)
        .arg("-s")
        .arg("1") // Run just 1 shot for the test
        .output()?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    let stdout = String::from_utf8_lossy(&output.stdout);

    assert!(
        output.status.success(),
        "Execution should succeed. Error: {stderr}"
    );

    // For QIR run, check that it produced output
    assert!(
        stdout.contains('[') && stdout.contains(']'),
        "Should output JSON results"
    );

    Ok(())
}

/// Test QIR with various shot counts
#[test]
fn test_qir_shot_counts() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize test environment and acquire lock to ensure sequential execution
    setup();
    let _lock = TEST_MUTEX.lock().unwrap();
    println!("Running QIR shot counts test (sequential execution)...");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_qir_path = manifest_dir.join("../../examples/qir/bell.ll");

    println!("QIR SHOT COUNT TEST: Testing various numbers of shots");
    println!("---------------------------------------------------");

    // Test different shot counts
    for &shots in &[1, 10, 100, 1000] {
        println!("\nTesting with {shots} shots:");

        let output = run_pecos(&bell_qir_path, shots, 1, "depolarizing", "0.0", 42)?;
        let values = get_values(&output);

        if values.len() != 1 {
            return Err(Box::new(PecosError::Resource(format!(
                "Expected 1 register with values, got {}",
                values.len()
            ))));
        }

        let outcomes = values[0].split(", ").collect::<Vec<_>>();
        assert_eq!(
            outcomes.len(),
            shots,
            "Expected {} measurement outcomes, got {}",
            shots,
            outcomes.len()
        );

        // All outcomes should be either 0 or 3 for a Bell state
        let valid_outcomes = outcomes.iter().all(|&o| o == "0" || o == "3");
        assert!(
            valid_outcomes,
            "All outcomes should be |00⟩ (0) or |11⟩ (3) for a Bell state"
        );

        println!("  Correctly produced {shots} shots with valid Bell state outcomes");
    }

    Ok(())
}

/// Test QIR with multiple workers
#[test]
fn test_qir_multiple_workers() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize test environment and acquire lock to ensure sequential execution
    setup();
    let _lock = TEST_MUTEX.lock().unwrap();
    println!("Running QIR multi-worker test (sequential execution)...");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_qir_path = manifest_dir.join("../../examples/qir/bell.ll");

    println!("QIR MULTI-WORKER TEST: Testing parallel execution");
    println!("-----------------------------------------------");

    // Run with different numbers of workers
    for &workers in &[1, 2, 4] {
        println!("\nTesting with {workers} workers:");

        let output = run_pecos(&bell_qir_path, 100, workers, "depolarizing", "0.0", 42)?;
        let values = get_values(&output);

        if values.len() != 1 {
            return Err(Box::new(PecosError::Resource(format!(
                "Expected 1 register with values, got {}",
                values.len()
            ))));
        }

        let outcomes = values[0].split(", ").collect::<Vec<_>>();
        let state_00_count = outcomes.iter().filter(|&&o| o == "0").count();
        let state_11_count = outcomes.iter().filter(|&&o| o == "3").count();

        // Verify we still get valid Bell state results
        assert_eq!(
            state_00_count + state_11_count,
            100,
            "All outcomes should be |00⟩ or |11⟩"
        );

        // Check for reasonable distribution
        let ratio_00 = state_00_count;
        assert!(
            (35..=65).contains(&ratio_00),
            "Distribution should be roughly balanced even with {workers} workers"
        );

        println!("  {workers} workers: {state_00_count} |00⟩, {state_11_count} |11⟩ states");
    }

    Ok(())
}
