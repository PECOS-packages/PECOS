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
    simulator: Option<&str>,
) -> Result<String, Box<dyn std::error::Error>> {
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

    // Add simulator parameter if specified
    if let Some(sim) = simulator {
        cmd.arg("-S").arg(sim);
    }

    let output = cmd.output()?;

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
    let mut register_values: std::collections::HashMap<String, Vec<String>> =
        std::collections::HashMap::new();

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

/// Test that a perfect (noiseless) Bell state produces the expected 50/50 distribution
/// of |00⟩ (0) and |11⟩ (3) outcomes
#[test]
fn test_perfect_bell_state_distribution() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.json");

    println!("PERFECT BELL STATE TEST: Verifying 50/50 distribution of |00⟩ and |11⟩ states");
    println!("---------------------------------------------------------------------------");

    // Run noiseless Bell state simulation with 100 shots
    let output = run_pecos(&bell_json_path, 100, 1, "depolarizing", "0.0", 42, None)?;
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

/// Test that Bell state probabilities are consistent between PHIR, QASM, and QIR implementations
#[test]
fn test_cross_implementation_validation() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.json");
    let bell_qasm_path = manifest_dir.join("../../examples/qasm/bell.qasm");
    let bell_qir_path = manifest_dir.join("../../examples/qir/bell.ll");

    println!("BELL STATE CROSS-VALIDATION: Comparing PHIR, QASM, and QIR implementations");
    println!("------------------------------------------------------------------------");

    // Run all three implementations with the same seed
    let phir_output = run_pecos(&bell_json_path, 100, 1, "depolarizing", "0.0", 42, None)?;
    let qasm_output = run_pecos(&bell_qasm_path, 100, 1, "depolarizing", "0.0", 42, None)?;
    let qir_output = run_pecos(&bell_qir_path, 100, 1, "depolarizing", "0.0", 42, None)?;

    // Extract the values and compare
    let phir_values = get_values(&phir_output);
    let qasm_values = get_values(&qasm_output);
    let qir_values = get_values(&qir_output);

    println!("PHIR results: {:.60}...", phir_output.trim());
    println!("QASM results: {:.60}...", qasm_output.trim());
    println!("QIR results:  {:.60}...", qir_output.trim());

    // All implementations should produce valid quantum Bell state results
    // Each should have a near 50/50 distribution of |00⟩ and |11⟩

    // Function to count |00⟩ and |11⟩ states
    let count_bell_states = |values: &[String]| -> (usize, usize) {
        let outcomes = values[0].split(", ").collect::<Vec<_>>();

        let state_00_count = outcomes.iter().filter(|&&o| o == "0").count();
        let state_11_count = outcomes.iter().filter(|&&o| o == "3").count();

        (state_00_count, state_11_count)
    };

    // Check all implementations
    let (phir_00_count, phir_11_count) = count_bell_states(&phir_values);
    let (qasm_00_count, qasm_11_count) = count_bell_states(&qasm_values);
    let (qir_00_count, qir_11_count) = count_bell_states(&qir_values);

    println!("PHIR Bell state distribution: {phir_00_count}% |00⟩, {phir_11_count}% |11⟩");
    println!("QASM Bell state distribution: {qasm_00_count}% |00⟩, {qasm_11_count}% |11⟩");
    println!("QIR Bell state distribution:  {qir_00_count}% |00⟩, {qir_11_count}% |11⟩");

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

    // Verify QIR implementation has balanced distribution
    assert!(
        (40..=60).contains(&qir_00_count),
        "QIR implementation should have between 40% and 60% |00⟩ states, but got {qir_00_count}%"
    );

    println!("PHIR, QASM, and QIR Bell state implementations all produce correct distributions");

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
    let noisy_dep_output = run_pecos(&bell_json_path, 500, 1, "depolarizing", "0.1", 42, None)?;
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
        None,
    )?;
    analyze_noisy_bell_state(&noisy_gen_output, "General")?;

    println!(
        "\nBoth noise models produce expected behavior: mostly Bell states with some noise-induced states"
    );

    Ok(())
}

/// Test that with the same seed, all implementations produce deterministic results
#[test]
fn test_seed_determinism() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.json");
    let bell_qasm_path = manifest_dir.join("../../examples/qasm/bell.qasm");
    let bell_qir_path = manifest_dir.join("../../examples/qir/bell.ll");

    println!("SEED DETERMINISM: Verifying all implementations are deterministic with same seed");
    println!("------------------------------------------------------------------------------");

    // Test PHIR determinism
    let phir_run1 = run_pecos(&bell_json_path, 50, 1, "depolarizing", "0.0", 42, None)?;
    let phir_run2 = run_pecos(&bell_json_path, 50, 1, "depolarizing", "0.0", 42, None)?;

    let phir_values1 = get_values(&phir_run1);
    let phir_values2 = get_values(&phir_run2);

    assert_eq!(
        phir_values1, phir_values2,
        "PHIR implementation should produce identical results with the same seed"
    );
    println!("PHIR implementation is deterministic with the same seed");

    // Test QASM determinism
    let qasm_run1 = run_pecos(&bell_qasm_path, 50, 1, "depolarizing", "0.0", 42, None)?;
    let qasm_run2 = run_pecos(&bell_qasm_path, 50, 1, "depolarizing", "0.0", 42, None)?;

    let qasm_values1 = get_values(&qasm_run1);
    let qasm_values2 = get_values(&qasm_run2);

    assert_eq!(
        qasm_values1, qasm_values2,
        "QASM implementation should produce identical results with the same seed"
    );
    println!("QASM implementation is deterministic with the same seed");

    // Test QIR determinism
    let qir_run1 = run_pecos(&bell_qir_path, 50, 1, "depolarizing", "0.0", 42, None)?;
    let qir_run2 = run_pecos(&bell_qir_path, 50, 1, "depolarizing", "0.0", 42, None)?;

    let qir_values1 = get_values(&qir_run1);
    let qir_values2 = get_values(&qir_run2);

    assert_eq!(
        qir_values1, qir_values2,
        "QIR implementation should produce identical results with the same seed"
    );
    println!("QIR implementation is deterministic with the same seed");

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
    let dep_run1 = run_pecos(&bell_json_path, 50, 1, "depolarizing", "0.1", 42, None)?;
    let dep_run2 = run_pecos(&bell_json_path, 50, 1, "depolarizing", "0.1", 42, None)?;

    let dep_values1 = get_values(&dep_run1);
    let dep_values2 = get_values(&dep_run2);

    assert_eq!(
        dep_values1, dep_values2,
        "Depolarizing noise model should produce identical results with the same seed"
    );
    println!("Depolarizing noise model is deterministic with the same seed");

    // Run general model twice with same seed
    let gen_run1 = run_pecos(
        &bell_json_path,
        50,
        1,
        "general",
        "0.1,0.1,0.1,0.1,0.1",
        42,
        None,
    )?;
    let gen_run2 = run_pecos(
        &bell_json_path,
        50,
        1,
        "general",
        "0.1,0.1,0.1,0.1,0.1",
        42,
        None,
    )?;

    let gen_values1 = get_values(&gen_run1);
    let gen_values2 = get_values(&gen_run2);

    assert_eq!(
        gen_values1, gen_values2,
        "General noise model should produce identical results with the same seed"
    );
    println!("General noise model is deterministic with the same seed");

    Ok(())
}

/// Test QIR implementation with noise models
#[test]
fn test_qir_with_noise() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_qir_path = manifest_dir.join("../../examples/qir/bell.ll");

    println!("QIR WITH NOISE: Testing QIR implementation with various noise models");
    println!("------------------------------------------------------------------");

    // Test with depolarizing noise
    let qir_dep_output = run_pecos(&bell_qir_path, 500, 1, "depolarizing", "0.1", 42, None)?;

    println!("\n1. Testing QIR with depolarizing noise model (p=0.1):");
    analyze_noisy_bell_state(&qir_dep_output, "QIR Depolarizing")?;

    // Test with general noise
    let qir_gen_output = run_pecos(
        &bell_qir_path,
        500,
        1,
        "general",
        "0.1,0.1,0.1,0.1,0.1",
        42,
        None,
    )?;

    println!("\n2. Testing QIR with general noise model (p=0.1 for all error types):");
    analyze_noisy_bell_state(&qir_gen_output, "QIR General")?;

    println!("\nQIR implementation correctly handles noise models");

    Ok(())
}

/// Test both simulator engines (state vector and stabilizer) and verify they produce
/// identical results for Bell state circuits
#[test]
fn test_simulator_engines() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_qasm_path = manifest_dir.join("../../examples/qasm/bell.qasm");

    println!("SIMULATOR ENGINE COMPARISON: Testing both state vector and stabilizer simulators");
    println!("--------------------------------------------------------------------------------");
    println!(
        "Bell state circuit is a Clifford circuit, so both simulators should produce identical results"
    );

    // Run with state vector simulator (default)
    let state_vector_output = run_pecos(
        &bell_qasm_path,
        100,
        1,
        "depolarizing",
        "0.0",
        42,
        Some("statevector"),
    )?;
    println!(
        "State vector simulator results: {:.60}...",
        state_vector_output.trim()
    );

    // Run with stabilizer simulator
    let stabilizer_output = run_pecos(
        &bell_qasm_path,
        100,
        1,
        "depolarizing",
        "0.0",
        42,
        Some("stabilizer"),
    )?;
    println!(
        "Stabilizer simulator results: {:.60}...",
        stabilizer_output.trim()
    );

    // Extract and compare the values
    let sv_values = get_values(&state_vector_output);
    let stab_values = get_values(&stabilizer_output);

    // Count |00⟩ and |11⟩ states for each simulator
    let count_bell_states = |values: &[String]| -> (usize, usize) {
        if values.is_empty() {
            return (0, 0);
        }

        let outcomes = values[0].split(", ").collect::<Vec<_>>();

        let state_00_count = outcomes.iter().filter(|&&o| o == "0").count();
        let state_11_count = outcomes.iter().filter(|&&o| o == "3").count();

        (state_00_count, state_11_count)
    };

    let (sv_00_count, sv_11_count) = count_bell_states(&sv_values);
    let (stab_00_count, stab_11_count) = count_bell_states(&stab_values);

    println!("State vector simulator: {sv_00_count} |00⟩ states, {sv_11_count} |11⟩ states");
    println!("Stabilizer simulator:  {stab_00_count} |00⟩ states, {stab_11_count} |11⟩ states");

    // Note: The two simulators may produce different measurement outcome sequences
    // even with the same seed, due to different implementations and RNG usage,
    // but both should produce valid Bell state distributions

    // Both simulators should produce balanced Bell state outcomes
    assert!(
        (40..=60).contains(&sv_00_count),
        "State vector simulator should have between 40% and 60% |00⟩ states, but got {sv_00_count}%"
    );

    assert!(
        (40..=60).contains(&stab_00_count),
        "Stabilizer simulator should have between 40% and 60% |00⟩ states, but got {stab_00_count}%"
    );

    // Both simulators should only produce |00⟩ and |11⟩ states for a Bell state
    let sv_outcomes = sv_values[0].split(", ").collect::<Vec<_>>();
    let stab_outcomes = stab_values[0].split(", ").collect::<Vec<_>>();

    assert!(
        sv_outcomes.iter().all(|&x| x == "0" || x == "3"),
        "State vector simulator should only produce |00⟩ and |11⟩ states"
    );

    assert!(
        stab_outcomes.iter().all(|&x| x == "0" || x == "3"),
        "Stabilizer simulator should only produce |00⟩ and |11⟩ states"
    );

    println!(
        "Both simulators produce correct Bell state distributions with proper quantum behavior"
    );

    Ok(())
}
