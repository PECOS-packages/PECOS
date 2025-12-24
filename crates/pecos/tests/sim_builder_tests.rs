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

//! Tests for the simulation builder API in the PECOS crate.

#![cfg(feature = "runtime")]

use pecos::prelude::*;
use pecos_engines::sim_builder;

/// Simple bell state program for testing.
const BELL_STATE_QASM: &str = r#"
OPENQASM 2.0;
include "qelib1.inc";
qreg q[2];
creg c[2];
h q[0];
cx q[0], q[1];
measure q -> c;
"#;

#[test]
fn test_sim_with_qasm_engine() -> Result<(), PecosError> {
    // Run simulation with explicit engine builder
    let results = sim_builder()
        .classical(qasm_engine().qasm(BELL_STATE_QASM))
        .seed(42)
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

    // With the bell state and no noise, we should only see |00⟩ and |11⟩ states
    for result in &c_values {
        // For a 2-qubit register, each shot result should be 0 or 3 (binary 00 or 11)
        assert!(*result == 0 || *result == 3);
    }

    Ok(())
}

#[test]
fn test_sim_with_qasm_program() -> Result<(), PecosError> {
    // Run simulation with QASM program
    let results = sim_builder()
        .classical(qasm_engine().qasm(BELL_STATE_QASM))
        .seed(42)
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

    // With the bell state and no noise, we should only see |00⟩ and |11⟩ states
    for result in &c_values {
        // For a 2-qubit register, each shot result should be 0 or 3 (binary 00 or 11)
        assert!(*result == 0 || *result == 3);
    }

    Ok(())
}

#[test]
fn test_sim_workers_parameter() -> Result<(), PecosError> {
    // Run simulation with 4 workers
    let results = sim_builder()
        .classical(qasm_engine().qasm(BELL_STATE_QASM))
        .seed(42)
        .workers(4)
        .run(100)?;

    // Verify results are correct
    assert_eq!(results.len(), 100);

    // Extract 'c' register values from shots
    let c_values: Vec<u32> = results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();
    assert_eq!(c_values.len(), 100);

    Ok(())
}

#[test]
fn test_sim_with_custom_noise_model() -> Result<(), PecosError> {
    // Create a custom noise model (PassThroughNoiseModel has no effect)
    let noise_model = PassThroughNoiseModel::builder();

    // Run simulation with custom noise model
    let results = sim_builder()
        .classical(qasm_engine().qasm(BELL_STATE_QASM))
        .seed(42)
        .noise(noise_model)
        .run(100)?;

    // Verify results are correct
    assert_eq!(results.len(), 100);

    // Extract 'c' register values from shots
    let c_values: Vec<u32> = results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();
    assert_eq!(c_values.len(), 100);

    Ok(())
}

#[test]
fn test_sim_with_custom_quantum_engine() -> Result<(), PecosError> {
    // Run simulation with custom quantum engine
    let results = sim_builder()
        .classical(qasm_engine().qasm(BELL_STATE_QASM))
        .seed(42)
        .quantum(state_vector().qubits(2))
        .run(100)?;

    // Verify results are correct
    assert_eq!(results.len(), 100);

    // Extract 'c' register values from shots
    let c_values: Vec<u32> = results
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();
    assert_eq!(c_values.len(), 100);

    Ok(())
}

#[test]
fn test_sim_determinism() -> Result<(), PecosError> {
    // Run simulations with the same seed
    let results1 = sim_builder()
        .classical(qasm_engine().qasm(BELL_STATE_QASM))
        .seed(42)
        .run(100)?;

    let results2 = sim_builder()
        .classical(qasm_engine().qasm(BELL_STATE_QASM))
        .seed(42)
        .run(100)?;

    // Results should be identical
    assert_eq!(results1, results2);

    // Now run with a different seed
    let results3 = sim_builder()
        .classical(qasm_engine().qasm(BELL_STATE_QASM))
        .seed(43)
        .run(100)?;

    // Results should be different (this is probabilistic but very likely)
    // We're checking if the measurements are completely identical, which is
    // extremely unlikely with different seeds over 100 shots
    assert!(results1 != results3);

    Ok(())
}

#[test]
fn test_sim_different_shots() -> Result<(), PecosError> {
    // Run with 50 shots
    let results1 = sim_builder()
        .classical(qasm_engine().qasm(BELL_STATE_QASM))
        .seed(42)
        .run(50)?;

    // Run with 200 shots
    let results2 = sim_builder()
        .classical(qasm_engine().qasm(BELL_STATE_QASM))
        .seed(42)
        .run(200)?;

    // Verify shot count matches
    assert_eq!(results1.len(), 50);
    assert_eq!(results2.len(), 200);

    // Extract 'c' register values from shots
    let c_values1: Vec<u32> = results1
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();
    let c_values2: Vec<u32> = results2
        .shots
        .iter()
        .filter_map(|shot| shot.data.get("c").and_then(pecos::prelude::Data::as_u32))
        .collect();
    assert_eq!(c_values1.len(), 50);
    assert_eq!(c_values2.len(), 200);

    Ok(())
}
