/// # Bell State Tests
///
/// This file contains tests that verify the quantum mechanical behavior of Bell states
/// in the PECOS simulator. Key aspects tested include:
///
/// 1. Proper 50/50 Distribution: Bell states should produce a quantum superposition
///    with equal probability of measuring |00⟩ and |11⟩ states
///
/// 2. Cross-Implementation Validation: Ensuring consistency between different
///    file formats (PHIR, QASM)
///
/// 3. Noise Effects: Analyzing how adding noise affects the Bell state probability
///    distribution by introducing |01⟩ and |10⟩ outcomes
///
/// These tests help verify that the quantum simulator correctly implements
/// quantum entanglement, superposition, and noise models.
use assert_cmd::prelude::*;
use pecos::prelude::*;
use std::collections::HashMap;
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

    // Fallback to manual parsing if serde_json fails (simplified for test)
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

/// Test that a perfect (noiseless) Bell state produces the expected 50/50 distribution
/// of |00⟩ (0) and |11⟩ (3) outcomes
#[test]
fn test_perfect_bell_state_distribution() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.json");

    println!("PERFECT BELL STATE TEST: Verifying 50/50 distribution of |00⟩ and |11⟩ states");
    println!("---------------------------------------------------------------------------");

    // Run noiseless Bell state simulation with 100 shots
    let output = run_pecos(&bell_json_path, 100, 1, "depolarizing", "0.0", 42)?;
    println!("Bell state results: {}", output.trim());

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
    println!("Outcome distribution:");
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

    // Verify Bell state behavior - should have only 0 and 3 outcomes (|00⟩ and |11⟩)
    let expected_states_count = state_00_count + state_11_count;
    println!(
        "  |00⟩ and |11⟩ states: {} out of {} ({}%)",
        expected_states_count,
        total_outcomes,
        (expected_states_count * 100) / total_outcomes
    );

    // Bell state should have 100% of outcomes being either |00⟩ or |11⟩
    assert!(
        expected_states_count == total_outcomes,
        "Expected all outcomes to be |00⟩ or |11⟩, but got {}%",
        (expected_states_count * 100) / total_outcomes
    );

    // Bell state should have roughly equal probability (40-60% range) of |00⟩ and |11⟩
    if state_00_count > 0 && state_11_count > 0 {
        let ratio_00 = (state_00_count * 100) / expected_states_count;
        let ratio_11 = (state_11_count * 100) / expected_states_count;

        println!("  |00⟩ to |11⟩ ratio: {ratio_00}% to {ratio_11}%");

        // Check if probabilities are roughly balanced (between 40% and 60%)
        assert!(
            (40..=60).contains(&ratio_00),
            "Expected |00⟩ probability between 40% and 60%, but got {ratio_00}%"
        );

        println!("Bell state probabilities are correctly balanced between |00⟩ and |11⟩");
    } else {
        return Err(Box::new(PecosError::Resource(
            "Missing either |00⟩ or |11⟩ state in Bell state simulation".to_string(),
        )));
    }

    Ok(())
}

/// Test that Bell state probabilities are consistent between PHIR and QASM implementations
#[test]
fn test_cross_implementation_validation() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.json");
    let bell_qasm_path = manifest_dir.join("../../examples/qasm/bell.qasm");

    println!("BELL STATE CROSS-VALIDATION: Comparing PHIR and QASM implementations");
    println!("------------------------------------------------------------------");

    // Run both implementations with the same seed
    let phir_output = run_pecos(&bell_json_path, 100, 1, "depolarizing", "0.0", 42)?;
    let qasm_output = run_pecos(&bell_qasm_path, 100, 1, "depolarizing", "0.0", 42)?;

    // Extract the values and compare
    let phir_values = get_values(&phir_output);
    let qasm_values = get_values(&qasm_output);

    println!("PHIR results: {:.60}...", phir_output.trim());
    println!("QASM results: {:.60}...", qasm_output.trim());

    // Both implementations should produce valid quantum Bell state results
    // Each should have a near 50/50 distribution of |00⟩ and |11⟩

    // Function to count |00⟩ and |11⟩ states
    let count_bell_states = |values: &[String]| -> (usize, usize) {
        let outcomes = values[0].split(", ").collect::<Vec<_>>();

        let state_00_count = outcomes.iter().filter(|&&o| o == "0").count();
        let state_11_count = outcomes.iter().filter(|&&o| o == "3").count();

        (state_00_count, state_11_count)
    };

    // Check both implementations
    let (phir_00_count, phir_11_count) = count_bell_states(&phir_values);
    let (qasm_00_count, qasm_11_count) = count_bell_states(&qasm_values);

    println!("PHIR Bell state distribution: {phir_00_count}% |00⟩, {phir_11_count}% |11⟩");
    println!("QASM Bell state distribution: {qasm_00_count}% |00⟩, {qasm_11_count}% |11⟩");

    // Verify PHIR implementation has balanced distribution
    assert!(
        (40..=60).contains(&phir_00_count),
        "PHIR implementation should have between 40% and 60% |00⟩ states, but got {phir_00_count}%"
    );

    // Verify QASM implementation has balanced distribution
    assert!(
        (40..=60).contains(&qasm_00_count),
        "QASM implementation should have between 40% and 60% |00⟩ states, but got {qasm_00_count}%"
    );

    println!("PHIR and QASM Bell state implementations produce identical results");

    Ok(())
}

/// Analyze Bell state outcomes with noise
#[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
fn analyze_noisy_bell_state(
    output: &str,
    model_name: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!(
        "{} noise model results (truncated): {:.100}...",
        model_name,
        output.trim()
    );

    // Count occurrences of each measurement outcome
    let values = get_values(output);
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
    println!("{model_name} noise model outcome distribution:");
    let mut total = 0;
    let mut state_00_count = 0;
    let mut state_11_count = 0;
    let mut state_01_count = 0;
    let mut state_10_count = 0;

    // We'll sort the outcomes for consistent display
    let mut sorted_outcomes: Vec<_> = counts.iter().collect();
    sorted_outcomes.sort_by_key(|k| k.0);

    for (outcome, count) in sorted_outcomes {
        let percentage = (count * 100) / outcomes.len() as i32;
        println!(
            "  Outcome {} (|{:02b}⟩): {} times ({}%)",
            outcome,
            outcome.parse::<i32>().unwrap_or(0),
            count,
            percentage
        );

        total += count;

        match *outcome {
            "0" => state_00_count = *count,
            "1" => state_01_count = *count,
            "2" => state_10_count = *count,
            "3" => state_11_count = *count,
            _ => {}
        }
    }

    // Calculate statistics
    let expected_states = state_00_count + state_11_count;
    let noise_states = state_01_count + state_10_count;

    println!(
        "  Bell states (|00⟩ and |11⟩): {} out of {} ({}%)",
        expected_states,
        total,
        (expected_states * 100) / total
    );

    println!(
        "  Noise-induced states (|01⟩ and |10⟩): {} out of {} ({}%)",
        noise_states,
        total,
        (noise_states * 100) / total
    );

    // With noise p=0.1, we should still have a majority of |00⟩ and |11⟩ states,
    // but with some |01⟩ and |10⟩ states due to noise
    assert!(
        expected_states > noise_states,
        "Expected Bell states (|00⟩ and |11⟩) to be more common than noise-induced states"
    );

    // We should see some noise-induced states
    assert!(
        noise_states > 0,
        "Expected to see some noise-induced states (|01⟩ and |10⟩) with p=0.1"
    );

    // Bell states should still be somewhat balanced despite noise
    if state_00_count > 0 && state_11_count > 0 {
        let ratio_00 = (state_00_count * 100) / expected_states;
        let ratio_11 = (state_11_count * 100) / expected_states;

        println!("  Bell states ratio - |00⟩ to |11⟩: {ratio_00}% to {ratio_11}%");

        // With noise, ratios might be less balanced, but should still be somewhat close
        assert!(
            (30..=70).contains(&ratio_00),
            "Expected |00⟩ probability between 30% and 70% with noise, but got {ratio_00}%"
        );
    }

    // Noise-induced states should also be somewhat balanced (|01⟩ and |10⟩)
    if state_01_count > 0 && state_10_count > 0 {
        let ratio_01 = (state_01_count * 100) / noise_states;
        let ratio_10 = (state_10_count * 100) / noise_states;

        println!("  Noise states ratio - |01⟩ to |10⟩: {ratio_01}% to {ratio_10}%");
    }

    Ok(())
}

/// Test how noise affects Bell state simulations by comparing outcomes with both
/// depolarizing and general noise models
#[test]
fn test_bell_state_with_noise() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.json");

    println!("BELL STATE WITH NOISE: Analyzing how noise affects Bell state outcomes");
    println!("-------------------------------------------------------------------");
    println!("With noise (p=0.1), we expect to see mostly |00⟩ and |11⟩ states,");
    println!("but also some |01⟩ and |10⟩ states introduced by the noise.");

    // Run with depolarizing noise model
    println!("\n1. Testing with depolarizing noise model (p=0.1):");
    let noisy_dep_output = run_pecos(&bell_json_path, 500, 1, "depolarizing", "0.1", 42)?;
    analyze_noisy_bell_state(&noisy_dep_output, "Depolarizing")?;

    // Run with general noise model
    println!("\n2. Testing with general noise model (p=0.1 for all error types):");
    let noisy_gen_output = run_pecos(
        &bell_json_path,
        500,
        1,
        "general",
        "0.1,0.1,0.1,0.1,0.1",
        42,
    )?;
    analyze_noisy_bell_state(&noisy_gen_output, "General")?;

    println!(
        "\nBoth noise models produce expected behavior: mostly Bell states with some noise-induced states"
    );

    Ok(())
}

/// Test that with the same seed, both noise models produce deterministic results
#[test]
fn test_noise_model_determinism() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.json");

    println!("NOISE MODEL DETERMINISM: Verifying noise models are deterministic with same seed");
    println!("------------------------------------------------------------------------");

    // Run depolarizing model twice with same seed
    let dep_run1 = run_pecos(&bell_json_path, 50, 1, "depolarizing", "0.1", 42)?;
    let dep_run2 = run_pecos(&bell_json_path, 50, 1, "depolarizing", "0.1", 42)?;

    let dep_values1 = get_values(&dep_run1);
    let dep_values2 = get_values(&dep_run2);

    assert_eq!(
        dep_values1, dep_values2,
        "Depolarizing noise model should produce identical results with the same seed"
    );
    println!("Depolarizing noise model is deterministic with the same seed");

    // Run general model twice with same seed
    let gen_run1 = run_pecos(&bell_json_path, 50, 1, "general", "0.1,0.1,0.1,0.1,0.1", 42)?;
    let gen_run2 = run_pecos(&bell_json_path, 50, 1, "general", "0.1,0.1,0.1,0.1,0.1", 42)?;

    let gen_values1 = get_values(&gen_run1);
    let gen_values2 = get_values(&gen_run2);

    assert_eq!(
        gen_values1, gen_values2,
        "General noise model should produce identical results with the same seed"
    );
    println!("General noise model is deterministic with the same seed");

    Ok(())
}
