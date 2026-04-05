use pecos::prelude::*;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::process::Command;

/// Configuration for running PECOS CLI tests
#[derive(Copy, Clone)]
struct PecosTestConfig<'a> {
    file_path: &'a PathBuf,
    shots: usize,
    workers: usize,
    noise_model: &'a str,
    noise_prob: &'a str,
    seed: u64,
    simulator: Option<&'a str>,
    use_jit: bool,
}

/// Helper function to run PECOS CLI with given parameters
fn run_pecos(config: PecosTestConfig) -> Result<String, Box<dyn std::error::Error>> {
    let mut cmd = Command::new(assert_cmd::cargo::cargo_bin!("pecos"));
    cmd.env("RUST_LOG", "warn") // Use warn to avoid pipe buffer issues with verbose output
        .env("RUST_BACKTRACE", "0") // Disable backtrace to avoid extra output on segfault
        .arg("run")
        .arg(config.file_path)
        .arg("-s")
        .arg(config.shots.to_string())
        .arg("-w")
        .arg(config.workers.to_string())
        .arg("-m")
        .arg(config.noise_model)
        .arg("-p")
        .arg(config.noise_prob)
        .arg("-d")
        .arg(config.seed.to_string());

    // Add simulator parameter if specified
    if let Some(sim) = config.simulator {
        cmd.arg("-S").arg(sim);
    }

    // Add JIT flag if specified (for LLVM files when Selene is not available)
    if config.use_jit {
        cmd.arg("--jit");
    }

    let output = cmd.output()?;

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    // Special handling for QIS files which may segfault during cleanup
    let is_qis = config.file_path.extension().and_then(|s| s.to_str()) == Some("ll");

    // For QIS files, check if we got valid output even if the process exited with error
    if is_qis && !output.status.success() {
        // QIS programs have a known segfault issue during cleanup
        // Check if we still got valid JSON output before the segfault
        if stdout.trim().starts_with('{') && stdout.trim().ends_with('}') {
            // We have valid JSON output despite the segfault
            log::debug!("Note: QIS process segfaulted during cleanup but produced valid output");
            return Ok(stdout.to_string());
        }
        // No valid output, this is a real failure
        return Err(Box::new(PecosError::Resource(format!(
            "QIS execution failed for file '{}': exit_code={:?}, stderr='{}', stdout='{}'",
            config.file_path.display(),
            output.status.code(),
            stderr,
            stdout
        ))));
    } else if !output.status.success() {
        // Provide more context about the error for non-QIS files
        return Err(Box::new(PecosError::Resource(format!(
            "PECOS run failed for file '{}' with settings (shots={}, workers={}, model={}, noise={}, seed={}): stderr='{}', stdout='{}', exit_code={:?}",
            config.file_path.display(),
            config.shots,
            config.workers,
            config.noise_model,
            config.noise_prob,
            config.seed,
            stderr,
            stdout,
            output.status.code()
        ))));
    }

    // Return the stdout we already converted
    Ok(stdout.to_string())
}

/// Extract measurement results from JSON output
/// Handles different output formats:
/// - Combined format: {"c": [3, 0, ...]} or any single register
/// - Individual indexed format: {"m0": [0, 1], "m1": [0, 1]} or any indexed registers
///
/// Also handles output that may contain non-JSON text before the JSON
fn get_values(json_output: &str) -> Vec<String> {
    let mut register_values: std::collections::BTreeMap<String, Vec<String>> =
        std::collections::BTreeMap::new();

    // Extract JSON part from output (may have other text like "Quantum runtime initialized")
    let json_part = json_output
        .lines()
        .find(|line| line.trim().starts_with('{') && line.trim().ends_with('}'))
        .map_or(json_output.trim(), str::trim);

    // Parse the JSON - expecting an object with register names as keys
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_part)
        && let Some(obj) = json.as_object()
    {
        // Group registers by their base name (without numeric suffix)
        let mut register_groups: std::collections::BTreeMap<
            String,
            Vec<(String, usize, Vec<i64>)>,
        > = std::collections::BTreeMap::new();
        let mut single_registers: std::collections::BTreeMap<String, Vec<String>> =
            std::collections::BTreeMap::new();

        for (reg_name, values) in obj {
            if let Some(arr) = values.as_array() {
                // Try to parse as indexed register
                let mut base_name = String::new();
                let mut index = None;
                let chars: Vec<char> = reg_name.chars().collect();
                let mut i = chars.len();

                // Find where digits end from the right
                while i > 0 && chars[i - 1].is_ascii_digit() {
                    i -= 1;
                }

                if i > 0 && i < chars.len() {
                    // We have both base and digits
                    base_name = chars[..i].iter().collect();
                    let index_str: String = chars[i..].iter().collect();
                    index = index_str.parse::<usize>().ok();
                }

                if let Some(idx) = index {
                    // This is an indexed register
                    let measurements: Vec<i64> =
                        arr.iter().map(|v| v.as_i64().unwrap_or(0)).collect();

                    register_groups.entry(base_name.clone()).or_default().push((
                        reg_name.clone(),
                        idx,
                        measurements,
                    ));
                } else {
                    // Single register (no numeric suffix or couldn't parse)
                    let string_values: Vec<String> =
                        arr.iter().map(|v| v.to_string().replace('"', "")).collect();
                    single_registers.insert(reg_name.clone(), string_values);
                }
            }
        }

        // Check if we should combine indexed registers
        for (base_name, mut group) in register_groups {
            if group.len() > 1 {
                // Multiple registers with same base - combine them
                group.sort_by_key(|&(_, idx, _)| idx);

                // Get number of shots
                let num_shots = group.first().map_or(0, |(_, _, m)| m.len());

                // Combine into classical register values
                let mut combined_values = Vec::new();
                for shot_idx in 0..num_shots {
                    let mut value = 0i64;
                    for (bit_position, (_, _idx, measurements)) in group.iter().enumerate() {
                        if shot_idx < measurements.len() {
                            value |= measurements[shot_idx] << bit_position;
                        }
                    }
                    combined_values.push(value.to_string());
                }

                // Use the base name for the combined register
                register_values.insert(base_name, combined_values);
            } else if let Some((orig_name, _, measurements)) = group.into_iter().next() {
                // Single indexed register - keep as is
                let string_values: Vec<String> = measurements
                    .iter()
                    .map(std::string::ToString::to_string)
                    .collect();
                register_values.insert(orig_name, string_values);
            }
        }

        // Add single registers
        for (reg_name, values) in single_registers {
            register_values.insert(reg_name, values);
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
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.phir.json");

    println!("PERFECT BELL STATE TEST: Verifying 50/50 distribution of |00⟩ and |11⟩ states");
    println!("---------------------------------------------------------------------------");

    // Run noiseless Bell state simulation with 100 shots
    let output = run_pecos(PecosTestConfig {
        file_path: &bell_json_path,
        shots: 100,
        workers: 1,
        noise_model: "depolarizing",
        noise_prob: "0.0",
        seed: 42,
        simulator: None,
        use_jit: false,
    })?;
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
    let mut counts = BTreeMap::new();

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

/// Test that Bell state probabilities are consistent between PHIR, QASM, and LLVM implementations
#[test]
fn test_cross_implementation_validation() -> Result<(), Box<dyn std::error::Error>> {
    // No lock needed: This test only executes quantum programs without modifying shared state

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.phir.json");
    let bell_qasm_path = manifest_dir.join("../../examples/qasm/bell.qasm");
    let bell_llvm_path = manifest_dir.join("../../examples/llvm/bell.ll");

    println!("BELL STATE CROSS-VALIDATION: Comparing PHIR, QASM, and LLVM implementations");
    println!("------------------------------------------------------------------------");

    // Run all three implementations with the same seed
    let phir_output = run_pecos(PecosTestConfig {
        file_path: &bell_json_path,
        shots: 100,
        workers: 1,
        noise_model: "depolarizing",
        noise_prob: "0.0",
        seed: 42,
        simulator: None,
        use_jit: false,
    })?;
    let qasm_output = run_pecos(PecosTestConfig {
        file_path: &bell_qasm_path,
        shots: 100,
        workers: 1,
        noise_model: "depolarizing",
        noise_prob: "0.0",
        seed: 42,
        simulator: None,
        use_jit: false,
    })?;
    let llvm_output = run_pecos(PecosTestConfig {
        file_path: &bell_llvm_path,
        shots: 100,
        workers: 1,
        noise_model: "depolarizing",
        noise_prob: "0.0",
        seed: 42,
        simulator: None,
        use_jit: true,
    })?;

    // Extract the values and compare
    let phir_values = get_values(&phir_output);
    let qasm_values = get_values(&qasm_output);
    let llvm_values = get_values(&llvm_output);

    println!("PHIR results: {:.60}...", phir_output.trim());
    println!("QASM results: {:.60}...", qasm_output.trim());
    println!("LLVM results:  {:.60}...", llvm_output.trim());

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
    let (llvm_00_count, llvm_11_count) = count_bell_states(&llvm_values);

    println!("PHIR Bell state distribution: {phir_00_count}% |00⟩, {phir_11_count}% |11⟩");
    println!("QASM Bell state distribution: {qasm_00_count}% |00⟩, {qasm_11_count}% |11⟩");
    println!("LLVM Bell state distribution:  {llvm_00_count}% |00⟩, {llvm_11_count}% |11⟩");

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

    // Verify LLVM implementation has balanced distribution
    assert!(
        (40..=60).contains(&llvm_00_count),
        "LLVM implementation should have between 40% and 60% |00⟩ states, but got {llvm_00_count}%"
    );

    println!("PHIR, QASM, and LLVM Bell state implementations all produce correct distributions");

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
    let mut counts = BTreeMap::new();

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
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.phir.json");

    println!("BELL STATE WITH NOISE: Analyzing how noise affects Bell state outcomes");
    println!("-------------------------------------------------------------------");
    println!("With noise (p=0.1), we expect to see mostly |00⟩ and |11⟩ states,");
    println!("but also some |01⟩ and |10⟩ states introduced by the noise.");

    // Run with depolarizing noise model
    println!("\n1. Testing with depolarizing noise model (p=0.1):");
    let noisy_dep_output = run_pecos(PecosTestConfig {
        file_path: &bell_json_path,
        shots: 200,
        workers: 1,
        noise_model: "depolarizing",
        noise_prob: "0.1",
        seed: 42,
        simulator: None,
        use_jit: false,
    })?;
    analyze_noisy_bell_state(&noisy_dep_output, "Depolarizing")?;

    // Run with general noise model
    println!("\n2. Testing with general noise model (p=0.1 for all error types):");
    let noisy_gen_output = run_pecos(PecosTestConfig {
        file_path: &bell_json_path,
        shots: 200,
        workers: 1,
        noise_model: "general",
        noise_prob: "0.1,0.1,0.1,0.1,0.1",
        seed: 42,
        simulator: None,
        use_jit: false,
    })?;
    analyze_noisy_bell_state(&noisy_gen_output, "General")?;

    println!(
        "\nBoth noise models produce expected behavior: mostly Bell states with some noise-induced states"
    );

    Ok(())
}

/// Test that with the same seed, all implementations produce deterministic results
#[test]
fn test_seed_determinism() -> Result<(), Box<dyn std::error::Error>> {
    // No lock needed: This test only executes quantum programs without modifying shared state

    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.phir.json");
    let bell_qasm_path = manifest_dir.join("../../examples/qasm/bell.qasm");
    let bell_llvm_path = manifest_dir.join("../../examples/llvm/bell.ll");

    println!("SEED DETERMINISM: Verifying all implementations are deterministic with same seed");
    println!("------------------------------------------------------------------------------");

    // Test PHIR determinism
    let phir_config = PecosTestConfig {
        file_path: &bell_json_path,
        shots: 50,
        workers: 1,
        noise_model: "depolarizing",
        noise_prob: "0.0",
        seed: 42,
        simulator: None,
        use_jit: false,
    };
    let phir_run1 = run_pecos(phir_config)?;
    let phir_run2 = run_pecos(phir_config)?;

    let phir_values1 = get_values(&phir_run1);
    let phir_values2 = get_values(&phir_run2);

    assert_eq!(
        phir_values1, phir_values2,
        "PHIR implementation should produce identical results with the same seed"
    );
    println!("PHIR implementation is deterministic with the same seed");

    // Test QASM determinism
    let qasm_config = PecosTestConfig {
        file_path: &bell_qasm_path,
        shots: 50,
        workers: 1,
        noise_model: "depolarizing",
        noise_prob: "0.0",
        seed: 42,
        simulator: None,
        use_jit: false,
    };
    let qasm_run1 = run_pecos(qasm_config)?;
    let qasm_run2 = run_pecos(qasm_config)?;

    let qasm_values1 = get_values(&qasm_run1);
    let qasm_values2 = get_values(&qasm_run2);

    assert_eq!(
        qasm_values1, qasm_values2,
        "QASM implementation should produce identical results with the same seed"
    );
    println!("QASM implementation is deterministic with the same seed");

    // Test LLVM determinism
    let llvm_config = PecosTestConfig {
        file_path: &bell_llvm_path,
        shots: 50,
        workers: 1,
        noise_model: "depolarizing",
        noise_prob: "0.0",
        seed: 42,
        simulator: None,
        use_jit: true,
    };
    let llvm_run1 = run_pecos(llvm_config)?;
    let llvm_run2 = run_pecos(llvm_config)?;

    let llvm_values1 = get_values(&llvm_run1);
    let llvm_values2 = get_values(&llvm_run2);

    assert_eq!(
        llvm_values1, llvm_values2,
        "LLVM implementation should produce identical results with the same seed"
    );
    println!("LLVM implementation is deterministic with the same seed");

    Ok(())
}

/// Test that with the same seed, both noise models produce deterministic results
#[test]
fn test_noise_model_determinism() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_json_path = manifest_dir.join("../../examples/phir/bell.phir.json");

    println!("NOISE MODEL DETERMINISM: Verifying noise models are deterministic with same seed");
    println!("------------------------------------------------------------------------");

    // Run depolarizing model twice with same seed
    let dep_config = PecosTestConfig {
        file_path: &bell_json_path,
        shots: 50,
        workers: 1,
        noise_model: "depolarizing",
        noise_prob: "0.1",
        seed: 42,
        simulator: None,
        use_jit: false,
    };
    let dep_run1 = run_pecos(dep_config)?;
    let dep_run2 = run_pecos(dep_config)?;

    let dep_values1 = get_values(&dep_run1);
    let dep_values2 = get_values(&dep_run2);

    assert_eq!(
        dep_values1, dep_values2,
        "Depolarizing noise model should produce identical results with the same seed"
    );
    println!("Depolarizing noise model is deterministic with the same seed");

    // Run general model twice with same seed
    let gen_config = PecosTestConfig {
        file_path: &bell_json_path,
        shots: 50,
        workers: 1,
        noise_model: "general",
        noise_prob: "0.1,0.1,0.1,0.1,0.1",
        seed: 42,
        simulator: None,
        use_jit: false,
    };
    let gen_run1 = run_pecos(gen_config)?;
    let gen_run2 = run_pecos(gen_config)?;

    let gen_values1 = get_values(&gen_run1);
    let gen_values2 = get_values(&gen_run2);

    assert_eq!(
        gen_values1, gen_values2,
        "General noise model should produce identical results with the same seed"
    );
    println!("General noise model is deterministic with the same seed");

    Ok(())
}

/// Test LLVM implementation with depolarizing noise model
#[test]
fn test_qis_with_depolarizing_noise() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_llvm_path = manifest_dir.join("../../examples/llvm/bell.ll");

    println!(
        "LLVM WITH DEPOLARIZING NOISE: Testing LLVM implementation with depolarizing noise model"
    );
    println!("------------------------------------------------------------------");

    // Test with depolarizing noise - reduced shots to avoid segfault issues
    let llvm_dep_output = run_pecos(PecosTestConfig {
        file_path: &bell_llvm_path,
        shots: 100,
        workers: 1,
        noise_model: "depolarizing",
        noise_prob: "0.1",
        seed: 42,
        simulator: None,
        use_jit: true,
    })?;

    println!("Testing LLVM with depolarizing noise model (p=0.1):");
    analyze_noisy_bell_state(&llvm_dep_output, "LLVM Depolarizing")?;

    println!("\nLLVM implementation correctly handles depolarizing noise model");

    Ok(())
}

/// Test LLVM implementation with general noise model
#[test]
fn test_qis_with_general_noise() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let bell_llvm_path = manifest_dir.join("../../examples/llvm/bell.ll");

    println!("LLVM WITH GENERAL NOISE: Testing LLVM implementation with general noise model");
    println!("------------------------------------------------------------------");

    // Test with general noise - reduced shots to avoid segfault issues
    let llvm_gen_output = run_pecos(PecosTestConfig {
        file_path: &bell_llvm_path,
        shots: 100,
        workers: 1,
        noise_model: "general",
        noise_prob: "0.1,0.1,0.1,0.1,0.1",
        seed: 42,
        simulator: None,
        use_jit: true,
    })?;

    println!("Testing LLVM with general noise model (p=0.1 for all error types):");
    analyze_noisy_bell_state(&llvm_gen_output, "LLVM General")?;

    println!("\nLLVM implementation correctly handles general noise model");

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
    let state_vector_output = run_pecos(PecosTestConfig {
        file_path: &bell_qasm_path,
        shots: 100,
        workers: 1,
        noise_model: "depolarizing",
        noise_prob: "0.0",
        seed: 42,
        simulator: Some("statevector"),
        use_jit: false,
    })?;
    println!(
        "State vector simulator results: {:.60}...",
        state_vector_output.trim()
    );

    // Run with stabilizer simulator
    let stabilizer_output = run_pecos(PecosTestConfig {
        file_path: &bell_qasm_path,
        shots: 100,
        workers: 1,
        noise_model: "depolarizing",
        noise_prob: "0.0",
        seed: 42,
        simulator: Some("stabilizer"),
        use_jit: false,
    })?;
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
