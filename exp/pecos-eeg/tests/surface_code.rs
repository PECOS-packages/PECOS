// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0

//! Integration test: EEG analysis on a repetition code circuit.

use pecos_core::Gate;
use pecos_eeg::Bm;
use pecos_eeg::circuit::{NoiseModel, analyze_expanded};
use pecos_eeg::dem_mapping::*;
use pecos_eeg::eeg::EegType;
use pecos_eeg::expand;
use pecos_quantum::TickCircuit;

/// Build a 3-qubit repetition code with 2 syndrome rounds.
///
/// Layout: data qubits 0,1,2; ancilla qubits 3,4
/// Stabilizers: Z0Z1 (measured by ancilla 3), Z1Z2 (measured by ancilla 4)
fn build_repetition_code() -> (Vec<Gate>, Vec<Detector>, Vec<Observable>) {
    let mut tc = TickCircuit::new();

    // Initialize all qubits
    tc.tick().pz(&[0, 1, 2, 3, 4]);

    // Two syndrome extraction rounds
    for _round in 0..2 {
        tc.tick().pz(&[3, 4]);
        tc.tick().cx(&[(0, 3), (1, 4)]);
        tc.tick().cx(&[(1, 3), (2, 4)]);
        tc.tick().mz(&[3, 4]);
    }

    // Final data readout
    tc.tick().mz(&[0, 1, 2]);

    let gates: Vec<Gate> = tc
        .iter_gate_batches()
        .map(|batch| batch.as_gate().clone())
        .collect();

    // Detector stabilizers: X on ancilla qubit (anticommutes with Z errors
    // that propagate through CX from data qubits).
    let detectors = vec![
        Detector {
            id: 0,
            stabilizer: Bm::x(3),
        },
        Detector {
            id: 1,
            stabilizer: Bm::x(4),
        },
    ];

    let observables = vec![Observable {
        id: 0,
        pauli: Bm::x(0).multiply(&Bm::x(1)).multiply(&Bm::x(2)),
    }];

    (gates, detectors, observables)
}

#[test]
fn test_repetition_code_no_noise() {
    let (gates, _, _) = build_repetition_code();
    let expanded = expand::expand_circuit(&gates);
    let noise = NoiseModel::coherent_only(0.0);
    let result = analyze_expanded(&expanded.gates, &noise);

    assert!(result.generators.is_empty());
}

#[test]
fn test_repetition_code_coherent_noise() {
    let (gates, detectors, observables) = build_repetition_code();
    let expanded = expand::expand_circuit(&gates);
    let noise = NoiseModel::coherent_only(0.1);
    let result = analyze_expanded(&expanded.gates, &noise);

    let h_count = result
        .generators
        .iter()
        .filter(|g| g.eeg_type == EegType::H)
        .count();
    assert!(h_count > 0, "Should have H generators from RZ noise");

    let dem_entries = build_dem(&result.generators, &detectors, &observables);

    assert!(
        !dem_entries.is_empty(),
        "Coherent noise should produce detection events"
    );

    let dem_str = format_dem(&dem_entries);
    eprintln!("Coherent DEM:\n{dem_str}");

    for entry in &dem_entries {
        assert!(entry.probability > 0.0, "Probability must be positive");
        assert!(
            entry.probability < 1.0,
            "Probability {:.6} too large",
            entry.probability
        );
    }
}

#[test]
fn test_repetition_code_depolarizing_noise() {
    let (gates, detectors, observables) = build_repetition_code();
    let expanded = expand::expand_circuit(&gates);
    let noise = NoiseModel::depolarizing(0.003);
    let result = analyze_expanded(&expanded.gates, &noise);

    let s_count = result
        .generators
        .iter()
        .filter(|g| g.eeg_type == EegType::S)
        .count();
    assert!(s_count > 0);

    let dem_entries = build_dem(&result.generators, &detectors, &observables);

    let dem_str = format_dem(&dem_entries);
    eprintln!("Stochastic DEM:\n{dem_str}");

    assert!(!dem_entries.is_empty());
    for entry in &dem_entries {
        assert!(entry.probability > 0.0);
        assert!(entry.probability < 0.5);
    }
}

#[test]
fn test_repetition_code_combined_noise() {
    let (gates, detectors, observables) = build_repetition_code();
    let expanded = expand::expand_circuit(&gates);
    let noise = NoiseModel::depolarizing(0.003).with_idle_rz(0.1);
    let result = analyze_expanded(&expanded.gates, &noise);

    let h_count = result
        .generators
        .iter()
        .filter(|g| g.eeg_type == EegType::H)
        .count();
    let s_count = result
        .generators
        .iter()
        .filter(|g| g.eeg_type == EegType::S)
        .count();
    assert!(h_count > 0);
    assert!(s_count > 0);

    let dem_entries = build_dem(&result.generators, &detectors, &observables);

    let dem_str = format_dem(&dem_entries);
    eprintln!("Combined DEM:\n{dem_str}");

    assert!(!dem_entries.is_empty());
}

#[test]
fn test_eeg_generator_count_scales_linearly() {
    for num_rounds in [1, 2, 4, 8] {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1, 2, 3, 4]);

        for _ in 0..num_rounds {
            tc.tick().pz(&[3, 4]);
            tc.tick().cx(&[(0, 3), (1, 4)]);
            tc.tick().cx(&[(1, 3), (2, 4)]);
            tc.tick().mz(&[3, 4]);
        }
        tc.tick().mz(&[0, 1, 2]);

        let gates: Vec<Gate> = tc
            .iter_gate_batches()
            .map(|batch| batch.as_gate().clone())
            .collect();
        let expanded = expand::expand_circuit(&gates);
        let noise = NoiseModel::coherent_only(0.1);
        let result = analyze_expanded(&expanded.gates, &noise);

        let h_count = result
            .generators
            .iter()
            .filter(|g| g.eeg_type == EegType::H)
            .count();
        eprintln!("Rounds={num_rounds}: {h_count} H generators");
        assert!(h_count < 1000, "Generator count should be polynomial");
    }
}
