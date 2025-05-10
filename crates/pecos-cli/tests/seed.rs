use assert_cmd::prelude::*;
use std::path::PathBuf;
use std::process::Command;

// Helper function to extract keys from JSON output
fn get_keys(json_output: &str) -> Vec<String> {
    let mut keys = Vec::new();

    // Try to parse the JSON using serde_json, which is the most reliable method
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(json_output) {
        if let Some(obj) = json.as_object() {
            for key in obj.keys() {
                keys.push(key.clone());
            }
            keys.sort();
            return keys;
        }
    }

    // Fallback to manual parsing if serde_json fails
    for line in json_output.lines() {
        if let Some(key_part) = line.trim().strip_prefix("\"") {
            if let Some(end_idx) = key_part.find("\": ") {
                keys.push(key_part[..end_idx].to_string());
            }
        }
    }

    // Sort for stable comparison
    keys.sort();
    keys
}

// Helper function to extract values from JSON output
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
    // This is a simplified version that may not handle all JSON formats correctly
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

#[test]
fn test_seed_produces_consistent_results() -> Result<(), Box<dyn std::error::Error>> {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_file = manifest_dir.join("../../examples/phir/bell.json");

    // Run multiple times with seed 42, forcing JSON format
    let seed_42_run1 = Command::cargo_bin("pecos")?
        .env("RUST_LOG", "info")
        .arg("run")
        .arg(&test_file)
        .arg("-s")
        .arg("10") // Fewer shots for faster tests
        .arg("-w")
        .arg("1") // Single worker to avoid thread scheduling differences
        .arg("-p")
        .arg("0.1")
        .arg("-d")
        .arg("42")
        .arg("-f")
        .arg("pretty-compact") // Force consistent format for test
        .output()?;

    let seed_42_run2 = Command::cargo_bin("pecos")?
        .env("RUST_LOG", "info")
        .arg("run")
        .arg(&test_file)
        .arg("-s")
        .arg("10")
        .arg("-w")
        .arg("1")
        .arg("-p")
        .arg("0.1")
        .arg("-d")
        .arg("42")
        .arg("-f")
        .arg("pretty-compact") // Force consistent format for test
        .output()?;

    // Run multiple times with seed 43
    let seed_43_run1 = Command::cargo_bin("pecos")?
        .env("RUST_LOG", "info")
        .arg("run")
        .arg(&test_file)
        .arg("-s")
        .arg("10")
        .arg("-w")
        .arg("1")
        .arg("-p")
        .arg("0.1")
        .arg("-d")
        .arg("43")
        .arg("-f")
        .arg("pretty-compact") // Force consistent format for test
        .output()?;

    let seed_43_run2 = Command::cargo_bin("pecos")?
        .env("RUST_LOG", "info")
        .arg("run")
        .arg(&test_file)
        .arg("-s")
        .arg("10")
        .arg("-w")
        .arg("1")
        .arg("-p")
        .arg("0.1")
        .arg("-d")
        .arg("43")
        .arg("-f")
        .arg("pretty-compact") // Force consistent format for test
        .output()?;

    // Check that all commands ran successfully
    assert!(seed_42_run1.status.success(), "First seed 42 run failed");
    assert!(seed_42_run2.status.success(), "Second seed 42 run failed");
    assert!(seed_43_run1.status.success(), "First seed 43 run failed");
    assert!(seed_43_run2.status.success(), "Second seed 43 run failed");

    // Convert outputs to strings
    let seed_42_output1 = String::from_utf8(seed_42_run1.stdout)?;
    let seed_42_output2 = String::from_utf8(seed_42_run2.stdout)?;
    let seed_43_output1 = String::from_utf8(seed_43_run1.stdout)?;
    let seed_43_output2 = String::from_utf8(seed_43_run2.stdout)?;

    // We need to normalize the JSON by sorting the keys, to ensure a stable order for comparison
    // Since we can't use serde_json without adding a dependency, we'll just print the sorted keys
    // and check for key existence

    // Check that seed 42 runs have the same keys
    let keys_42_1 = get_keys(&seed_42_output1);
    let keys_42_2 = get_keys(&seed_42_output2);
    assert_eq!(
        keys_42_1, keys_42_2,
        "Results with seed 42 should have the same keys across runs"
    );

    // Check that seed 43 runs have the same keys
    let keys_43_1 = get_keys(&seed_43_output1);
    let keys_43_2 = get_keys(&seed_43_output2);
    assert_eq!(
        keys_43_1, keys_43_2,
        "Results with seed 43 should have the same keys across runs"
    );

    // Check that seed 42 runs have values in the same positions
    let values_42_1 = get_values(&seed_42_output1);
    let values_42_2 = get_values(&seed_42_output2);
    assert_eq!(
        values_42_1, values_42_2,
        "Results with seed 42 should have the same values across runs"
    );

    // Check that seed 43 runs have values in the same positions
    let values_43_1 = get_values(&seed_43_output1);
    let values_43_2 = get_values(&seed_43_output2);
    assert_eq!(
        values_43_1, values_43_2,
        "Results with seed 43 should have the same values across runs"
    );

    // Verify that different seeds produce different results by checking value patterns
    assert_ne!(
        values_42_1, values_43_1,
        "Results with different seeds (42 vs 43) should differ"
    );

    Ok(())
}
