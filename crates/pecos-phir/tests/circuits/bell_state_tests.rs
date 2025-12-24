/*!
Consolidated Bell state tests for PHIR

This file consolidates all Bell state tests that were previously scattered
across multiple files. Bell states are fundamental in quantum computing,
so we test them thoroughly but avoid redundancy.
*/

use pecos_core::errors::PecosError;
use pecos_engines::{Engine, ClassicalEngine};
use pecos_engines::hybrid::builder::HybridEngineBuilder;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::shot_results::Data;
use pecos_phir::PhirEngine;
use pecos_phir_json::v0_1::engine::PhirJsonEngine;
use pecos_phir_json::v0_1::ast::PHIRProgram;
use pecos_phir_json::phir_json_to_module;
use std::collections::BTreeMap;

/// Helper function to create Bell state JSON
fn bell_state_json() -> &'static str {
    r#"{
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {"description": "Bell state preparation"},
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "i64", "variable": "m", "size": 2},
            {"qop": "H", "args": [["q", 0]]},
            {"qop": "CX", "args": [["q", 0], ["q", 1]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
            {"cop": "Result", "args": ["m"], "returns": ["result"]}
        ]
    }"#
}

#[test]
fn test_bell_state_phir_engine() -> Result<(), PecosError> {
    // Test Bell state with PhirEngine
    let module = phir_json_to_module(bell_state_json())?;
    let engine = PhirEngine::new(module)
        .map_err(|e| PecosError::Input(format!("Failed to create PhirEngine: {}", e)))?;

    let num_qubits = engine.num_qubits();
    let quantum_engine = Box::new(StateVecEngine::new(num_qubits));

    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(engine))
        .with_quantum_engine(quantum_engine)
        .build();

    // Run a single shot
    let shot = hybrid.run_shot()?;

    // Verify we got a valid Bell state result (0 or 3)
    if let Some(Data::U32(value)) = shot.data.get("result") {
        assert!(
            *value == 0 || *value == 3,
            "Bell state should produce |00⟩ (0) or |11⟩ (3), got {}", value
        );
    } else {
        panic!("Expected 'result' key in output");
    }

    Ok(())
}

#[test]
fn test_bell_state_phir_json_engine() -> Result<(), PecosError> {
    // Test Bell state with PhirJsonEngine
    let program: PHIRProgram = serde_json::from_str(bell_state_json())
        .map_err(|e| PecosError::Input(format!("Failed to parse PHIR JSON: {}", e)))?;

    let mut engine = PhirJsonEngine::from_program(program)?;

    // Execute directly (PhirJsonEngine has built-in quantum backend)
    let shot = engine.process(())?;

    // Check for result in various possible keys
    let value = shot.data.get("result")
        .or(shot.data.get("m"))
        .or(shot.data.get("output"))
        .expect("Should have a result key");

    if let Data::U32(v) = value {
        assert!(
            *v == 0 || *v == 3,
            "Bell state should produce |00⟩ (0) or |11⟩ (3), got {}", v
        );
    } else {
        panic!("Expected U32 data type");
    }

    Ok(())
}

#[test]
fn test_bell_state_engine_comparison() -> Result<(), PecosError> {
    // Verify both engines produce valid Bell state results
    let num_shots = 100;

    // Collect results from PhirEngine
    let module = phir_json_to_module(bell_state_json())?;
    let engine = PhirEngine::new(module.clone())
        .map_err(|e| PecosError::Input(format!("Failed to create PhirEngine: {}", e)))?;

    let quantum_engine = Box::new(StateVecEngine::new(engine.num_qubits()));
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(engine))
        .with_quantum_engine(quantum_engine)
        .build();

    let mut phir_results = BTreeMap::new();
    for _ in 0..num_shots {
        let shot = hybrid.run_shot()?;
        if let Some(Data::U32(value)) = shot.data.get("result") {
            *phir_results.entry(*value).or_insert(0) += 1;
        }
        Engine::reset(&mut hybrid)?;
    }

    // Collect results from PhirJsonEngine
    let program: PHIRProgram = serde_json::from_str(bell_state_json())?;
    let mut json_engine = PhirJsonEngine::from_program(program)?;

    let mut json_results = BTreeMap::new();
    for _ in 0..num_shots {
        let shot = json_engine.process(())?;
        let value = shot.data.get("result")
            .or(shot.data.get("m"))
            .expect("Should have a result");

        if let Data::U32(v) = value {
            *json_results.entry(*v).or_insert(0) += 1;
        }
        Engine::reset(&mut json_engine)?;
    }

    // Both engines should only produce 0 or 3
    assert!(phir_results.keys().all(|&k| k == 0 || k == 3),
            "PhirEngine produced invalid Bell state");
    assert!(json_results.keys().all(|&k| k == 0 || k == 3),
            "PhirJsonEngine produced invalid Bell state");

    // Both should have non-zero counts for both outcomes (with high probability)
    assert!(phir_results.len() == 2 || num_shots < 50,
            "PhirEngine should produce both outcomes with {} shots", num_shots);
    assert!(json_results.len() == 2 || num_shots < 50,
            "PhirJsonEngine should produce both outcomes with {} shots", num_shots);

    Ok(())
}

#[test]
fn test_bell_state_distribution() -> Result<(), PecosError> {
    // Test that Bell state produces roughly 50/50 distribution
    let num_shots = 1000;
    let tolerance = 0.1; // Allow 10% deviation from 50/50

    let module = phir_json_to_module(bell_state_json())?;
    let engine = PhirEngine::new(module)
        .map_err(|e| PecosError::Input(format!("Failed to create PhirEngine: {}", e)))?;

    let quantum_engine = Box::new(StateVecEngine::new(engine.num_qubits()));
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(engine))
        .with_quantum_engine(quantum_engine)
        .build();

    let mut counts = BTreeMap::new();
    for _ in 0..num_shots {
        let shot = hybrid.run_shot()?;
        if let Some(Data::U32(value)) = shot.data.get("result") {
            *counts.entry(*value).or_insert(0) += 1;
        }
        Engine::reset(&mut hybrid)?;
    }

    // Should have exactly 2 outcomes
    assert_eq!(counts.len(), 2, "Bell state should have exactly 2 outcomes");

    // Check distribution is roughly 50/50
    let count_0 = counts.get(&0).unwrap_or(&0);
    let count_3 = counts.get(&3).unwrap_or(&0);

    assert_eq!(count_0 + count_3, num_shots, "Total counts should equal shots");

    let ratio_0 = *count_0 as f64 / num_shots as f64;
    let ratio_3 = *count_3 as f64 / num_shots as f64;

    assert!(
        (ratio_0 - 0.5).abs() < tolerance,
        "|00⟩ probability {:.2} deviates too much from 0.5", ratio_0
    );
    assert!(
        (ratio_3 - 0.5).abs() < tolerance,
        "|11⟩ probability {:.2} deviates too much from 0.5", ratio_3
    );

    Ok(())
}

#[test]
fn test_bell_state_with_custom_output_name() -> Result<(), PecosError> {
    // Test Bell state with different output variable name
    let custom_json = r#"{
        "format": "PHIR/JSON",
        "version": "0.1.0",
        "metadata": {"description": "Bell state with custom output"},
        "ops": [
            {"data": "qvar_define", "data_type": "qubits", "variable": "q", "size": 2},
            {"data": "cvar_define", "data_type": "i64", "variable": "m", "size": 2},
            {"qop": "H", "args": [["q", 0]]},
            {"qop": "CX", "args": [["q", 0], ["q", 1]]},
            {"qop": "Measure", "args": [["q", 0]], "returns": [["m", 0]]},
            {"qop": "Measure", "args": [["q", 1]], "returns": [["m", 1]]},
            {"cop": "Result", "args": ["m"], "returns": ["my_custom_output"]}
        ]
    }"#;

    let module = phir_json_to_module(custom_json)?;
    let engine = PhirEngine::new(module)
        .map_err(|e| PecosError::Input(format!("Failed to create PhirEngine: {}", e)))?;

    let quantum_engine = Box::new(StateVecEngine::new(engine.num_qubits()));
    let mut hybrid = HybridEngineBuilder::new()
        .with_classical_engine(Box::new(engine))
        .with_quantum_engine(quantum_engine)
        .build();

    let shot = hybrid.run_shot()?;

    // Should have the custom output name
    assert!(shot.data.contains_key("my_custom_output"),
            "Should have custom output name");

    if let Some(Data::U32(value)) = shot.data.get("my_custom_output") {
        assert!(*value == 0 || *value == 3,
                "Bell state should produce |00⟩ (0) or |11⟩ (3)");
    }

    Ok(())
}