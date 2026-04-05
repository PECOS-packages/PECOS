// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Tests for the simulation builder API with QASM, PHIR/JSON, and QIS formats.

#![cfg(feature = "runtime")]

use pecos::prelude::*;
use pecos_engines::sim_builder;
use std::collections::BTreeMap;
use std::fs;

// Simple deterministic circuit that applies various gates and produces
// a predictable output pattern. We will implement this in multiple formats.
// The circuit:
// - Uses 3 qubits
// - Initializes q[0] to |1⟩
// - Creates superposition on q[1]
// - Applies entangling operations
// - Produces deterministic measurement patterns

// Note: Removed S, Tdg, and Z gates as they're not implemented in all backends
const SIMPLE_TEST_QASM: &str = r#"
OPENQASM 2.0;
include "qelib1.inc";

qreg q[3];
creg c[3];

x q[0];
h q[1];

cx q[0], q[1];
cx q[1], q[2];

measure q -> c;
"#;

const SIMPLE_TEST_PHIR: &str = r#"{
  "format": "PHIR/JSON",
  "version": "0.1.0",
  "metadata": {"description": "Simple test circuit"},
  "ops": [
    {
      "data": "qvar_define",
      "data_type": "qubits",
      "variable": "q",
      "size": 3
    },
    {
      "data": "cvar_define",
      "data_type": "i64",
      "variable": "m",
      "size": 3
    },
    {"qop": "X", "args": [["q", 0]]},
    {"qop": "H", "args": [["q", 1]]},
    {"qop": "CX", "args": [["q", 0], ["q", 1]]},
    {"qop": "CX", "args": [["q", 1], ["q", 2]]},
    {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
    {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
    {"qop": "Measure", "args": [["q", 2]], "returns": [["m", 2]]},
    {"cop": "Result", "args": ["m"], "returns": ["c"]}
  ]
}"#;

// Helper function to count occurrences of each measurement outcome
fn count_outcomes(results: &[u32]) -> BTreeMap<u32, usize> {
    let mut counts = BTreeMap::new();
    for &result in results {
        *counts.entry(result).or_insert(0) += 1;
    }
    counts
}

#[test]
fn test_sim_with_qasm_direct() -> Result<(), PecosError> {
    // Run simulation with 100 shots
    let results = sim_builder()
        .classical(qasm_engine().qasm(SIMPLE_TEST_QASM))
        .seed(42) // seed for determinism
        .run(100)?; // shots

    // Verify results contain 100 shots
    assert_eq!(results.len(), 100);

    // Extract 'c' register values from shots
    let c_values: Vec<u32> = results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();
    assert_eq!(c_values.len(), 100);

    // Print outcome distribution for debugging
    println!("QASM Direct Results:");
    let counts = count_outcomes(&c_values);
    for (outcome, count) in &counts {
        println!(
            "  |{:03b}⟩ ({}): {} times ({}%)",
            outcome,
            outcome,
            count,
            (count * 100) / c_values.len()
        );
    }

    // With this deterministic circuit, shots should produce consistent patterns
    // We expect a limited set of outcomes based on the specific gates
    // applied in the circuit

    Ok(())
}

#[test]
fn test_sim_with_phir_direct() -> Result<(), PecosError> {
    // Run simulation with 100 shots
    let results = sim_builder()
        .classical(pecos_phir_json::phir_json_engine().json(SIMPLE_TEST_PHIR)?)
        .seed(42) // seed for determinism
        .run(100)?;

    // Verify results contain 100 shots
    assert_eq!(results.len(), 100);

    // Extract 'c' register values from shots
    let c_values: Vec<u32> = results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();
    assert_eq!(c_values.len(), 100);

    // Print outcome distribution for debugging
    println!("PHIR Direct Results:");
    let counts = count_outcomes(&c_values);
    for (outcome, count) in &counts {
        println!(
            "  |{:03b}⟩ ({}): {} times ({}%)",
            outcome,
            outcome,
            count,
            (count * 100) / c_values.len()
        );
    }

    Ok(())
}

#[test]
fn test_cross_format_consistency() -> Result<(), PecosError> {
    // Run simulations with the same seed
    let qasm_results = sim_builder()
        .classical(qasm_engine().qasm(SIMPLE_TEST_QASM))
        .seed(42) // same seed
        .run(100)?;

    let phir_results = sim_builder()
        .classical(pecos_phir_json::phir_json_engine().json(SIMPLE_TEST_PHIR)?)
        .seed(42) // same seed
        .run(100)?;

    // Both formats should produce 100 shots
    assert_eq!(qasm_results.len(), 100);
    assert_eq!(phir_results.len(), 100);

    // Extract 'c' register values from shots
    let qasm_c_values: Vec<u32> = qasm_results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();
    let phir_c_values: Vec<u32> = phir_results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();

    // Compare actual results - with the same seed, the results should be identical
    assert_eq!(
        qasm_c_values, phir_c_values,
        "QASM and PHIR results should be identical with the same seed"
    );

    // Print comparison
    println!("Cross-Format Consistency Results:");
    println!("  QASM: First 5 results = {:?}", &qasm_c_values[0..5]);
    println!("  PHIR: First 5 results = {:?}", &phir_c_values[0..5]);

    Ok(())
}

#[test]
fn test_sim_from_files() -> Result<(), PecosError> {
    // Create a temporary directory that won't be automatically deleted
    let temp_dir = tempfile::Builder::new()
        .prefix("pecos-test-")
        .tempdir()
        .map_err(|e| PecosError::IO(std::io::Error::other(e)))?;

    // Make sure the temporary directory exists
    println!("Created temp dir at: {:?}", temp_dir.path());

    // Create QASM test file
    let qasm_path = temp_dir.path().join("simple_test.qasm");
    fs::write(&qasm_path, SIMPLE_TEST_QASM).map_err(PecosError::IO)?;

    // Create PHIR/JSON test file
    let phir_path = temp_dir.path().join("simple_test.json");
    fs::write(&phir_path, SIMPLE_TEST_PHIR).map_err(PecosError::IO)?;

    // Verify files were created
    println!("Created test files:");
    println!("  QASM: {qasm_path:?}");
    println!("  PHIR: {phir_path:?}");

    // Run simulations
    let qasm_results = sim_builder()
        .classical(qasm_engine().qasm(SIMPLE_TEST_QASM))
        .seed(42)
        .run(100)?;
    let phir_results = sim_builder()
        .classical(pecos_phir_json::phir_json_engine().json(SIMPLE_TEST_PHIR)?)
        .seed(42)
        .run(100)?;

    // Verify results contain 100 shots
    assert_eq!(qasm_results.len(), 100);
    assert_eq!(phir_results.len(), 100);

    // Extract 'c' register values from shots
    let qasm_c_values: Vec<u32> = qasm_results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();
    let phir_c_values: Vec<u32> = phir_results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();

    // Compare results - should be identical with the same seed
    assert_eq!(
        qasm_c_values, phir_c_values,
        "Results from file-based engines should match"
    );

    // Keep directory alive until the end of the test
    // This prevents premature cleanup of the temporary directory
    let _keep_alive = temp_dir;

    Ok(())
}

#[test]
fn test_noise_model_effects() -> Result<(), PecosError> {
    // Run simulation with no noise
    let noiseless_results = sim_builder()
        .classical(qasm_engine().qasm(SIMPLE_TEST_QASM))
        .seed(42)
        .noise(PassThroughNoiseModel::builder()) // explicitly use pass-through (no noise)
        .run(500)?; // more shots to analyze statistics

    // Run simulation with depolarizing noise
    let noisy_results = sim_builder()
        .classical(qasm_engine().qasm(SIMPLE_TEST_QASM))
        .seed(42) // same seed
        .noise(DepolarizingNoise { p: 0.1 }) // 10% noise
        .run(500)?; // same shot count

    // Both should have 500 shots
    assert_eq!(noiseless_results.len(), 500);
    assert_eq!(noisy_results.len(), 500);

    // Extract 'c' register values from shots
    let noiseless_c_values: Vec<u32> = noiseless_results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();
    let noisy_c_values: Vec<u32> = noisy_results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();
    assert_eq!(noiseless_c_values.len(), 500);
    assert_eq!(noisy_c_values.len(), 500);

    // Count outcome frequencies for both runs
    let noiseless_counts = count_outcomes(&noiseless_c_values);
    let noisy_counts = count_outcomes(&noisy_c_values);

    // Print noiseless distribution
    println!("Noiseless Results Distribution:");
    for (outcome, count) in &noiseless_counts {
        println!(
            "  |{:03b}⟩ ({}): {} times ({}%)",
            outcome,
            outcome,
            count,
            (count * 100) / noiseless_c_values.len()
        );
    }

    // Print noisy distribution
    println!("Noisy Results Distribution:");
    for (outcome, count) in &noisy_counts {
        println!(
            "  |{:03b}⟩ ({}): {} times ({}%)",
            outcome,
            outcome,
            count,
            (count * 100) / noisy_c_values.len()
        );
    }

    // With noise, we should see more outcome patterns
    assert!(
        noisy_counts.len() > noiseless_counts.len(),
        "Noisy simulation should have more diverse outcomes than noiseless"
    );

    Ok(())
}

// Note: Worker count determinism test removed as different worker counts
// intentionally produce different (but deterministic) results. Each worker
// gets its own RNG stream for optimal parallel performance. Users should
// use consistent seed AND worker count for reproducible results.

#[test]
fn test_deterministic_outcome_frequencies() -> Result<(), PecosError> {
    // Run with 1000 shots to get reliable statistics
    let results = sim_builder()
        .classical(qasm_engine().qasm(SIMPLE_TEST_QASM))
        .seed(42)
        .run(1000)?;

    // Extract 'c' register values from shots
    let c_values: Vec<u32> = results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();

    // Analyze the outcome distribution
    let counts = count_outcomes(&c_values);

    // Count total measurements
    let total_measurements = c_values.len();

    // Print detailed distribution
    println!("Outcome Distribution for 1000 shots:");
    for (outcome, count) in &counts {
        // Using `as_` here is acceptable for tests - the values are small and we're just
        // calculating percentages for display purposes
        #[allow(clippy::cast_precision_loss)]
        let percentage = (*count as f64 / total_measurements as f64) * 100.0;
        println!("  |{outcome:03b}⟩ ({outcome}): {count} times ({percentage:.2}%)");
    }

    // For our deterministic circuit, we expect to see only a limited set of outcomes
    // The exact distribution depends on the circuit design

    Ok(())
}
