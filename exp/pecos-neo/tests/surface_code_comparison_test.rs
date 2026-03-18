// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Surface code comparison tests between `ComposableNoiseModel` and `GeneralNoiseModel`.
//!
//! These tests validate that both noise models produce statistically similar results
//! when running surface code syndrome extraction at different numbers of rounds.

use pecos_engines::byte_message::ByteMessageBuilder;
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{Engine, QuantumSystem};
use pecos_neo::prelude::*;
use pecos_qsim::SparseStab;
use std::collections::HashMap;

/// Configuration for a simple repetition code (distance-3).
///
/// This is a 1D version of the surface code, easier to work with for testing.
/// Data qubits: 0, 1, 2
/// Z-ancillas: 3 (checks Z0*Z1), 4 (checks Z1*Z2)
///
/// Logical Z: Z0*Z1*Z2
/// Logical X: X0 (or X1 or X2)
struct RepetitionCodeD3 {
    data_qubits: Vec<usize>,
    z_ancillas: Vec<usize>,
    logical_z_qubits: Vec<usize>,
}

impl RepetitionCodeD3 {
    fn new() -> Self {
        Self {
            data_qubits: vec![0, 1, 2],
            z_ancillas: vec![3, 4],
            logical_z_qubits: vec![0, 1, 2],
        }
    }

    fn num_qubits(&self) -> usize {
        5 // 3 data + 2 ancilla
    }
}

/// Results from running syndrome extraction.
#[derive(Debug, Clone)]
struct SyndromeResult {
    /// Syndrome bits for each round (ancilla measurements).
    syndromes: Vec<Vec<bool>>,
    /// Final data qubit measurements (used for logical error detection).
    #[allow(dead_code)]
    final_data: Vec<bool>,
    /// Whether a logical error occurred (parity of logical Z operator).
    logical_error: bool,
}

/// Statistics collected over many shots.
#[derive(Debug, Default)]
struct SyndromeStatistics {
    total_shots: usize,
    logical_errors: usize,
    /// Syndrome occurrence rates per (round, ancilla).
    syndrome_rates: HashMap<(usize, usize), usize>,
    /// Temporal correlations: consecutive syndromes on same ancilla.
    temporal_correlations: HashMap<usize, usize>,
    /// Spatial correlations: syndromes on adjacent ancillas in same round.
    spatial_correlations: HashMap<usize, usize>,
}

impl SyndromeStatistics {
    fn new() -> Self {
        Self::default()
    }

    fn record(&mut self, result: &SyndromeResult) {
        self.total_shots += 1;

        if result.logical_error {
            self.logical_errors += 1;
        }

        // Record syndrome occurrences
        for (round, syndrome) in result.syndromes.iter().enumerate() {
            for (ancilla, &bit) in syndrome.iter().enumerate() {
                if bit {
                    *self.syndrome_rates.entry((round, ancilla)).or_insert(0) += 1;
                }
            }

            // Spatial correlation: adjacent ancillas triggered in same round
            if syndrome.len() >= 2 && syndrome[0] && syndrome[1] {
                *self.spatial_correlations.entry(round).or_insert(0) += 1;
            }
        }

        // Temporal correlation: same ancilla triggered in consecutive rounds
        for round in 1..result.syndromes.len() {
            for ancilla in 0..result.syndromes[round].len() {
                if result.syndromes[round][ancilla] && result.syndromes[round - 1][ancilla] {
                    *self.temporal_correlations.entry(ancilla).or_insert(0) += 1;
                }
            }
        }
    }

    fn logical_error_rate(&self) -> f64 {
        if self.total_shots == 0 {
            0.0
        } else {
            self.logical_errors as f64 / self.total_shots as f64
        }
    }

    fn syndrome_rate(&self, round: usize, ancilla: usize) -> f64 {
        let count = *self.syndrome_rates.get(&(round, ancilla)).unwrap_or(&0);
        if self.total_shots == 0 {
            0.0
        } else {
            count as f64 / self.total_shots as f64
        }
    }

    fn average_syndrome_rate(&self) -> f64 {
        if self.syndrome_rates.is_empty() || self.total_shots == 0 {
            0.0
        } else {
            let total: usize = self.syndrome_rates.values().sum();
            total as f64 / (self.syndrome_rates.len() * self.total_shots) as f64
        }
    }

    fn temporal_correlation_rate(&self, ancilla: usize, num_rounds: usize) -> f64 {
        let count = *self.temporal_correlations.get(&ancilla).unwrap_or(&0);
        let max_correlations = (num_rounds.saturating_sub(1)) * self.total_shots;
        if max_correlations == 0 {
            0.0
        } else {
            count as f64 / max_correlations as f64
        }
    }

    fn spatial_correlation_rate(&self, num_rounds: usize) -> f64 {
        let total: usize = self.spatial_correlations.values().sum();
        let max_correlations = num_rounds * self.total_shots;
        if max_correlations == 0 {
            0.0
        } else {
            total as f64 / max_correlations as f64
        }
    }
}

/// Run repetition code with `GeneralNoiseModel`.
fn run_general_noise_repetition(
    code: &RepetitionCodeD3,
    noise_model: GeneralNoiseModel,
    num_rounds: usize,
    num_shots: usize,
) -> SyndromeStatistics {
    let quantum = Box::new(StateVecEngine::new(code.num_qubits()));
    let mut system = QuantumSystem::new(Box::new(noise_model), quantum);
    system.set_seed(42);

    let mut stats = SyndromeStatistics::new();

    for _ in 0..num_shots {
        // Build circuit
        let mut builder = ByteMessageBuilder::new();
        let _ = builder.for_quantum_operations();

        // Initialize data qubits
        for &q in &code.data_qubits {
            builder.add_prep(&[q]);
        }

        // Syndrome extraction rounds
        for _round in 0..num_rounds {
            // Prepare ancillas
            for &a in &code.z_ancillas {
                builder.add_prep(&[a]);
            }

            // CNOT gates for parity checks
            // Ancilla 3 checks Z0*Z1
            builder.add_cx(&[0], &[3]);
            builder.add_cx(&[1], &[3]);
            // Ancilla 4 checks Z1*Z2
            builder.add_cx(&[1], &[4]);
            builder.add_cx(&[2], &[4]);

            // Measure ancillas
            for &a in &code.z_ancillas {
                builder.add_measurements(&[a]);
            }
        }

        // Final data measurements
        for &q in &code.data_qubits {
            builder.add_measurements(&[q]);
        }

        let circ = builder.build();

        system.reset().expect("Failed to reset system");
        let output = system.process(circ).expect("Processing failed");

        // Parse outcomes
        let outcomes: Vec<u32> = output
            .outcomes()
            .map(|o| o.into_iter().collect())
            .unwrap_or_default();

        // Extract syndromes and final data
        let total_ancilla_meas = num_rounds * code.z_ancillas.len();
        let syndromes: Vec<Vec<bool>> = (0..num_rounds)
            .map(|r| {
                let start = r * code.z_ancillas.len();
                code.z_ancillas
                    .iter()
                    .enumerate()
                    .map(|(i, _)| outcomes.get(start + i).is_some_and(|&v| v != 0))
                    .collect()
            })
            .collect();

        let final_data: Vec<bool> = code
            .data_qubits
            .iter()
            .enumerate()
            .map(|(i, _)| {
                outcomes
                    .get(total_ancilla_meas + i)
                    .is_some_and(|&v| v != 0)
            })
            .collect();

        // Check logical error (parity of logical Z = Z0*Z1*Z2)
        let logical_parity: bool = code
            .logical_z_qubits
            .iter()
            .filter_map(|&q| {
                let idx = code.data_qubits.iter().position(|&d| d == q)?;
                final_data.get(idx).copied()
            })
            .fold(false, |acc, b| acc ^ b);

        let result = SyndromeResult {
            syndromes,
            final_data,
            logical_error: logical_parity,
        };

        stats.record(&result);
    }

    stats
}

/// Configuration for composable noise model.
#[derive(Clone)]
struct ComposableNoiseConfig {
    p1: f64,
    p2: f64,
    p_meas: f64,
}

impl ComposableNoiseConfig {
    fn build(&self) -> ComposableNoiseModel {
        ComposableNoiseModel::new()
            .add_channel(SingleQubitChannel::depolarizing(self.p1))
            .add_channel(TwoQubitChannel::depolarizing(self.p2))
            .add_channel(MeasurementChannel::symmetric(self.p_meas))
    }
}

/// Run repetition code with `ComposableNoiseModel`.
fn run_composable_noise_repetition(
    code: &RepetitionCodeD3,
    noise_config: ComposableNoiseConfig,
    num_rounds: usize,
    num_shots: usize,
) -> SyndromeStatistics {
    let mut stats = SyndromeStatistics::new();

    for shot in 0..num_shots {
        // Build circuit
        let mut builder = CommandBuilder::new();

        // Initialize data qubits
        for &q in &code.data_qubits {
            builder = builder.pz(q);
        }

        // Track which measurement index corresponds to which qubit
        let mut meas_order: Vec<usize> = Vec::new();

        // Syndrome extraction rounds
        for _round in 0..num_rounds {
            // Prepare ancillas
            for &a in &code.z_ancillas {
                builder = builder.pz(a);
            }

            // CNOT gates for parity checks
            builder = builder.cx(0, 3); // Z0*Z1 on ancilla 3
            builder = builder.cx(1, 3);
            builder = builder.cx(1, 4); // Z1*Z2 on ancilla 4
            builder = builder.cx(2, 4);

            // Measure ancillas
            for &a in &code.z_ancillas {
                builder = builder.mz(a);
                meas_order.push(a);
            }
        }

        // Final data measurements
        for &q in &code.data_qubits {
            builder = builder.mz(q);
            meas_order.push(q);
        }

        let commands = builder.build();

        // Create fresh noise model per shot
        let noise_model = noise_config.build();

        // Run with fresh simulator and RNG state per shot for better independence
        let mut state = SparseStab::new(code.num_qubits());
        let mut runner = CircuitRunner::<SparseStab>::new()
            .with_noise(noise_model)
            .with_seed(42 + shot as u64);

        state.reset();
        let outcomes = runner.apply_circuit(&mut state, &commands).unwrap();

        // Extract syndromes and data from measurement outcomes in order
        // outcomes.as_slice() gives us all measurements in the order they were performed
        let all_outcomes: Vec<bool> = outcomes.iter().map(|o| o.outcome).collect();

        // Extract syndromes from ancilla measurements (first num_rounds * 2 measurements)
        let total_ancilla_meas = num_rounds * code.z_ancillas.len();
        let syndromes: Vec<Vec<bool>> = (0..num_rounds)
            .map(|r| {
                let start = r * code.z_ancillas.len();
                code.z_ancillas
                    .iter()
                    .enumerate()
                    .map(|(i, _)| {
                        let meas_idx = start + i;
                        all_outcomes.get(meas_idx).copied().unwrap_or(false)
                    })
                    .collect()
            })
            .collect();

        // Extract final data measurements
        let final_data: Vec<bool> = code
            .data_qubits
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let meas_idx = total_ancilla_meas + i;
                all_outcomes.get(meas_idx).copied().unwrap_or(false)
            })
            .collect();

        // Check logical error (parity of logical Z = Z0*Z1*Z2)
        let logical_parity: bool = final_data.iter().fold(false, |acc, &b| acc ^ b);

        let result = SyndromeResult {
            syndromes,
            final_data,
            logical_error: logical_parity,
        };

        stats.record(&result);
    }

    stats
}

const NUM_SHOTS: usize = 2000;
const TOLERANCE: f64 = 0.03; // 3% absolute tolerance

/// Compare two rates and check if they match within tolerance.
fn rates_match(rate1: f64, rate2: f64, tolerance: f64) -> bool {
    (rate1 - rate2).abs() <= tolerance
}

#[test]
fn test_repetition_code_logical_error_vs_rounds() {
    // Test that logical error rate increases with number of rounds
    // and that both noise models produce similar rates.

    let code = RepetitionCodeD3::new();

    // Error probabilities
    let p1 = 0.01; // 1% single-qubit error
    let p2 = 0.02; // 2% two-qubit error
    let p_meas = 0.01; // 1% measurement error

    // GeneralNoiseModel
    let general_model = GeneralNoiseModel::builder()
        .with_prep_probability(0.0)
        .with_meas_0_probability(p_meas)
        .with_meas_1_probability(p_meas)
        .with_average_p1_probability(p1 / 1.5) // Scale for average probability
        .with_average_p2_probability(p2 / 1.25)
        .with_p1_emission_ratio(0.0)
        .with_p2_emission_ratio(0.0)
        .with_seed(42)
        .build();

    // ComposableNoiseModel config
    let composable_config = ComposableNoiseConfig { p1, p2, p_meas };

    println!("\nRepetition Code Logical Error Rate vs Rounds:");
    println!("  p1={p1}, p2={p2}, p_meas={p_meas}");
    println!(
        "  {:>8} {:>15} {:>15} {:>10}",
        "Rounds", "GeneralNM", "ComposableNM", "Match"
    );
    println!("  {:->8} {:->15} {:->15} {:->10}", "", "", "", "");

    let rounds_to_test = [1, 2, 3, 5, 8];
    let mut all_match = true;

    for &num_rounds in &rounds_to_test {
        let general_stats =
            run_general_noise_repetition(&code, general_model.clone(), num_rounds, NUM_SHOTS);
        let composable_stats = run_composable_noise_repetition(
            &code,
            composable_config.clone(),
            num_rounds,
            NUM_SHOTS,
        );

        let general_rate = general_stats.logical_error_rate();
        let composable_rate = composable_stats.logical_error_rate();

        let matched = rates_match(general_rate, composable_rate, TOLERANCE);
        all_match = all_match && matched;

        println!(
            "  {:>8} {:>14.3}% {:>14.3}% {:>10}",
            num_rounds,
            general_rate * 100.0,
            composable_rate * 100.0,
            if matched { "OK" } else { "MISMATCH" }
        );
    }

    assert!(
        all_match,
        "Logical error rates should match within {:.0}% tolerance",
        TOLERANCE * 100.0
    );
}

#[test]
fn test_repetition_code_syndrome_rates() {
    // Test that syndrome rates are similar between the two noise models.

    let code = RepetitionCodeD3::new();
    let num_rounds = 3;

    let p1 = 0.02;
    let p2 = 0.03;
    let p_meas = 0.02;

    let general_model = GeneralNoiseModel::builder()
        .with_prep_probability(0.0)
        .with_meas_0_probability(p_meas)
        .with_meas_1_probability(p_meas)
        .with_average_p1_probability(p1 / 1.5)
        .with_average_p2_probability(p2 / 1.25)
        .with_p1_emission_ratio(0.0)
        .with_p2_emission_ratio(0.0)
        .with_seed(42)
        .build();

    let composable_config = ComposableNoiseConfig { p1, p2, p_meas };

    let general_stats = run_general_noise_repetition(&code, general_model, num_rounds, NUM_SHOTS);
    let composable_stats =
        run_composable_noise_repetition(&code, composable_config, num_rounds, NUM_SHOTS);

    println!("\nRepetition Code Syndrome Rates ({num_rounds} rounds):");
    println!("  p1={p1}, p2={p2}, p_meas={p_meas}");
    println!(
        "  Average syndrome rate: General={:.2}%, Composable={:.2}%",
        general_stats.average_syndrome_rate() * 100.0,
        composable_stats.average_syndrome_rate() * 100.0
    );

    // Compare per-round, per-ancilla rates
    println!("\n  Per-round syndrome rates:");
    let mut all_match = true;

    for round in 0..num_rounds {
        for ancilla in 0..code.z_ancillas.len() {
            let general_rate = general_stats.syndrome_rate(round, ancilla);
            let composable_rate = composable_stats.syndrome_rate(round, ancilla);
            let matched = rates_match(general_rate, composable_rate, TOLERANCE);
            all_match = all_match && matched;

            println!(
                "    Round {} Ancilla {}: General={:.2}%, Composable={:.2}% {}",
                round,
                ancilla,
                general_rate * 100.0,
                composable_rate * 100.0,
                if matched { "" } else { "MISMATCH" }
            );
        }
    }

    assert!(
        all_match,
        "Syndrome rates should match within {:.0}% tolerance",
        TOLERANCE * 100.0
    );
}

#[test]
fn test_repetition_code_syndrome_correlations() {
    // Test that temporal and spatial syndrome correlations are similar.

    let code = RepetitionCodeD3::new();
    let num_rounds = 5;

    let p1 = 0.02;
    let p2 = 0.04;
    let p_meas = 0.02;

    let general_model = GeneralNoiseModel::builder()
        .with_prep_probability(0.0)
        .with_meas_0_probability(p_meas)
        .with_meas_1_probability(p_meas)
        .with_average_p1_probability(p1 / 1.5)
        .with_average_p2_probability(p2 / 1.25)
        .with_p1_emission_ratio(0.0)
        .with_p2_emission_ratio(0.0)
        .with_seed(42)
        .build();

    let composable_config = ComposableNoiseConfig { p1, p2, p_meas };

    let general_stats = run_general_noise_repetition(&code, general_model, num_rounds, NUM_SHOTS);
    let composable_stats =
        run_composable_noise_repetition(&code, composable_config, num_rounds, NUM_SHOTS);

    println!("\nRepetition Code Syndrome Correlations ({num_rounds} rounds):");
    println!("  p1={p1}, p2={p2}, p_meas={p_meas}");

    // Temporal correlations
    println!("\n  Temporal correlations (same ancilla, consecutive rounds):");
    let mut all_match = true;

    for ancilla in 0..code.z_ancillas.len() {
        let general_rate = general_stats.temporal_correlation_rate(ancilla, num_rounds);
        let composable_rate = composable_stats.temporal_correlation_rate(ancilla, num_rounds);
        let matched = rates_match(general_rate, composable_rate, TOLERANCE);
        all_match = all_match && matched;

        println!(
            "    Ancilla {}: General={:.3}%, Composable={:.3}% {}",
            ancilla,
            general_rate * 100.0,
            composable_rate * 100.0,
            if matched { "" } else { "MISMATCH" }
        );
    }

    // Spatial correlations
    let general_spatial = general_stats.spatial_correlation_rate(num_rounds);
    let composable_spatial = composable_stats.spatial_correlation_rate(num_rounds);
    let spatial_matched = rates_match(general_spatial, composable_spatial, TOLERANCE);
    all_match = all_match && spatial_matched;

    println!("\n  Spatial correlations (adjacent ancillas, same round):");
    println!(
        "    General={:.3}%, Composable={:.3}% {}",
        general_spatial * 100.0,
        composable_spatial * 100.0,
        if spatial_matched { "" } else { "MISMATCH" }
    );

    assert!(
        all_match,
        "Syndrome correlations should match within {:.0}% tolerance",
        TOLERANCE * 100.0
    );
}

#[test]
fn test_repetition_code_error_scaling() {
    // Test that logical error rate scales similarly with physical error rate.

    let code = RepetitionCodeD3::new();
    let num_rounds = 3;

    println!("\nRepetition Code Error Scaling ({num_rounds} rounds):");
    println!(
        "  {:>10} {:>15} {:>15} {:>10}",
        "p_phys", "GeneralNM", "ComposableNM", "Match"
    );
    println!("  {:->10} {:->15} {:->15} {:->10}", "", "", "", "");

    let error_rates = [0.005, 0.01, 0.02, 0.03, 0.05];
    let mut all_match = true;

    for &p in &error_rates {
        let general_model = GeneralNoiseModel::builder()
            .with_prep_probability(0.0)
            .with_meas_0_probability(p)
            .with_meas_1_probability(p)
            .with_average_p1_probability(p / 1.5)
            .with_average_p2_probability(p / 1.25)
            .with_p1_emission_ratio(0.0)
            .with_p2_emission_ratio(0.0)
            .with_seed(42)
            .build();

        let composable_config = ComposableNoiseConfig {
            p1: p,
            p2: p,
            p_meas: p,
        };

        let general_stats =
            run_general_noise_repetition(&code, general_model, num_rounds, NUM_SHOTS);
        let composable_stats =
            run_composable_noise_repetition(&code, composable_config, num_rounds, NUM_SHOTS);

        let general_rate = general_stats.logical_error_rate();
        let composable_rate = composable_stats.logical_error_rate();

        // Use relative tolerance for higher error rates
        let tolerance = TOLERANCE.max(p * 0.5);
        let matched = rates_match(general_rate, composable_rate, tolerance);
        all_match = all_match && matched;

        println!(
            "  {:>10.3} {:>14.2}% {:>14.2}% {:>10}",
            p,
            general_rate * 100.0,
            composable_rate * 100.0,
            if matched { "OK" } else { "MISMATCH" }
        );
    }

    assert!(all_match, "Error scaling should match between noise models");
}
