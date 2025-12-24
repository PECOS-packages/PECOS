#![allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use pecos_engines::Engine;
use pecos_engines::byte_message::ByteMessage;
use pecos_engines::noise::{BiasedDepolarizingNoiseModel, GeneralNoiseModel};
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{EngineSystem, QuantumSystem};
use std::collections::BTreeMap;

fn main() {
    // Create a simple quantum circuit that prepares a superposition and measures it
    let circ = ByteMessage::quantum_operations_builder()
        .add_h(&[0])
        .add_measurements(&[0])
        .build();

    // Create a quantum engine with 1 qubit
    let quantum = Box::new(StateVecEngine::new(1));

    // Compare BiasedDepolarizingNoise with equivalent GeneralNoise
    compare_biased_and_general(&circ, quantum.as_ref());

    // Test Bell state with both noise models
    bell_state_comparison();
}

fn compare_biased_and_general(circ: &ByteMessage, quantum: &StateVecEngine) {
    println!("Comparing BiasedDepolarizingNoise and GeneralNoise");
    println!("{:-^100}", "");
    println!(
        "{:<30} | {:<10} | {:<20} | {:<20}",
        "Configuration", "Expected", "BiasedDepolarizingNoise", "GeneralNoise"
    );
    println!("{:-^100}", "");

    // Configurations to test
    let configs = [
        (0.0, 0.0, "No bias (ideal)"),
        (0.2, 0.0, "20% flip 0->1, never 1->0"),
        (0.0, 0.2, "Never flip 0->1, 20% flip 1->0"),
        (0.3, 0.3, "30% flip both ways"),
        (0.5, 0.0, "50% flip 0->1, never 1->0"),
        (0.0, 0.5, "Never flip 0->1, 50% flip 1->0"),
    ];

    let num_shots = 10000;
    let seed = 42;

    for (p_flip_0, p_flip_1, desc) in configs {
        // Create biased depolarizing noise model with custom settings
        let biased_noise = BiasedDepolarizingNoiseModel::builder()
            .with_prep_probability(0.0)
            .with_meas_0_probability(p_flip_0) // Probability of flipping 0 to 1
            .with_meas_1_probability(p_flip_1) // Probability of flipping 1 to 0
            .with_p1_probability(0.0)
            .with_p2_probability(0.0)
            .with_seed(seed)
            .build();
        let mut biased_system =
            QuantumSystem::new(Box::new(biased_noise), Box::new(quantum.clone()));

        // Create equivalent general noise model (with gate noise set to 0)
        let general_noise = GeneralNoiseModel::builder()
            .with_prep_probability(0.0)
            .with_meas_0_probability(p_flip_0)
            .with_meas_1_probability(p_flip_1)
            .with_p1_probability(0.0)
            .with_p2_probability(0.0)
            .with_seed(seed)
            .build();
        let mut general_system =
            QuantumSystem::new(Box::new(general_noise), Box::new(quantum.clone()));

        // Run simulations with both noise models
        let mut biased_counts = BTreeMap::new();
        let mut general_counts = BTreeMap::new();

        for _ in 0..num_shots {
            // Run with biased noise model
            biased_system
                .reset()
                .expect("Failed to reset biased system");
            let biased_results = biased_system
                .process_as_system(circ.clone())
                .expect("Failed to process circuit with biased noise");
            let biased_measurements = biased_results
                .outcomes()
                .expect("Failed to parse biased measurements");
            let biased_result = biased_measurements
                .first()
                .map_or("?", |&value| if value == 1 { "1" } else { "0" });
            *biased_counts.entry(biased_result.to_string()).or_insert(0) += 1;

            // Run with general noise model
            general_system
                .reset()
                .expect("Failed to reset general system");
            let general_results = general_system
                .process_as_system(circ.clone())
                .expect("Failed to process circuit with general noise");
            let general_measurements = general_results
                .outcomes()
                .expect("Failed to parse general measurements");
            let general_result = general_measurements
                .first()
                .map_or("?", |&value| if value == 1 { "1" } else { "0" });
            *general_counts
                .entry(general_result.to_string())
                .or_insert(0) += 1;
        }

        // Calculate percentages
        let biased_pct_1 = biased_counts.get("1").unwrap_or(&0) * 100 / num_shots;
        let general_pct_1 = general_counts.get("1").unwrap_or(&0) * 100 / num_shots;

        // Calculate expected results based on probabilities
        // For a 50/50 input with H gate:
        // Expected 1s = 50% * p_flip_0 + 50% * (1-p_flip_1)
        let calc_1: f64 = 50.0 * p_flip_0 + 50.0 * (1.0 - p_flip_1);
        let expected_1 = calc_1.round() as usize;

        println!(
            "{:<30} | {:<10} | {:<20} | {:<20}",
            desc,
            format!("{}%", expected_1),
            format!("{}%", biased_pct_1),
            format!("{}%", general_pct_1)
        );
    }
    println!("{:-^100}", "");
    println!();
}

#[allow(clippy::too_many_lines)]
fn bell_state_comparison() {
    println!("Comparing Bell state with both noise models");
    println!("{:-^100}", "");

    // Create a Bell state circuit
    let bell_circ = ByteMessage::quantum_operations_builder()
        .add_h(&[0])
        .add_cx(&[0], &[1])
        .add_measurements(&[0])
        .add_measurements(&[1])
        .build();

    // Parameters for the test
    let p_flip_0 = 0.2;
    let p_flip_1 = 0.3;
    let num_shots = 10000;
    let seed = 42;

    // Create quantum engine with 2 qubits
    let quantum = StateVecEngine::new(2);

    // Create biased depolarizing noise model with custom settings
    let biased_noise = BiasedDepolarizingNoiseModel::builder()
        .with_prep_probability(0.0)
        .with_meas_0_probability(p_flip_0) // Probability of flipping 0 to 1
        .with_meas_1_probability(p_flip_1) // Probability of flipping 1 to 0
        .with_p1_probability(0.0)
        .with_p2_probability(0.0)
        .with_seed(seed)
        .build();
    let mut biased_system = QuantumSystem::new(Box::new(biased_noise), Box::new(quantum.clone()));

    // Create equivalent general noise model
    let general_noise = GeneralNoiseModel::builder()
        .with_prep_probability(0.0)
        .with_meas_0_probability(p_flip_0)
        .with_meas_1_probability(p_flip_1)
        .with_p1_probability(0.0)
        .with_p2_probability(0.0)
        .with_seed(seed)
        .build();
    let mut general_system = QuantumSystem::new(Box::new(general_noise), Box::new(quantum.clone()));

    // Run simulations with both models
    let mut biased_counts = BTreeMap::new();
    let mut general_counts = BTreeMap::new();

    for _ in 0..num_shots {
        // Run with biased noise model
        biased_system
            .reset()
            .expect("Failed to reset biased system");
        let biased_results = biased_system
            .process_as_system(bell_circ.clone())
            .expect("Failed to process bell circuit with biased noise");
        let biased_measurements = biased_results
            .outcomes()
            .expect("Failed to parse biased measurements");

        // Combine the measurement results into a string
        let mut biased_result = String::new();
        for &value in &biased_measurements {
            biased_result.push(if value == 1 { '1' } else { '0' });
        }
        *biased_counts.entry(biased_result).or_insert(0) += 1;

        // Run with general noise model
        general_system
            .reset()
            .expect("Failed to reset general system");
        let general_results = general_system
            .process_as_system(bell_circ.clone())
            .expect("Failed to process bell circuit with general noise");
        let general_measurements = general_results
            .outcomes()
            .expect("Failed to parse general measurements");

        // Combine the measurement results into a string
        let mut general_result = String::new();
        for &value in &general_measurements {
            general_result.push(if value == 1 { '1' } else { '0' });
        }
        *general_counts.entry(general_result).or_insert(0) += 1;
    }

    // Sort both expected and actual by the pattern (00, 01, 10, 11) for easier comparison
    let patterns = ["00", "01", "10", "11"];

    println!("Bell state with biased noise: p_flip_0 = {p_flip_0}, p_flip_1 = {p_flip_1}");
    println!("{:-^80}", "");
    println!(
        "{:<10} | {:<30} | {:<30}",
        "Pattern", "BiasedDepolarizingNoise", "GeneralNoise"
    );
    println!("{:-^80}", "");

    for pattern in &patterns {
        let biased_count = biased_counts.get(*pattern).unwrap_or(&0);
        let biased_pct = biased_count * 100 / num_shots;

        let general_count = general_counts.get(*pattern).unwrap_or(&0);
        let general_pct = general_count * 100 / num_shots;

        println!(
            "{:<10} | {:<30} | {:<30}",
            pattern,
            format!("{} ({}%)", biased_count, biased_pct),
            format!("{} ({}%)", general_count, general_pct)
        );
    }
    println!("{:-^80}", "");
    println!();

    // Calculate theoretical probabilities
    let mut expected_probs = BTreeMap::new();
    expected_probs.insert(
        "00".to_string(),
        ((1.0 - p_flip_0) * (1.0 - p_flip_0) + p_flip_1 * p_flip_1) * 50.0,
    );
    expected_probs.insert(
        "01".to_string(),
        ((1.0 - p_flip_0) * p_flip_0 + p_flip_1 * (1.0 - p_flip_1)) * 50.0,
    );
    expected_probs.insert(
        "10".to_string(),
        (p_flip_0 * (1.0 - p_flip_0) + (1.0 - p_flip_1) * p_flip_1) * 50.0,
    );
    expected_probs.insert(
        "11".to_string(),
        (p_flip_0 * p_flip_0 + (1.0 - p_flip_1) * (1.0 - p_flip_1)) * 50.0,
    );

    println!("Theoretical probabilities:");
    println!("{:-^60}", "");
    println!("{:<10} | {:<20}", "Pattern", "Expected %");
    println!("{:-^60}", "");

    for pattern in &patterns {
        let expected = expected_probs.get(*pattern).unwrap_or(&0.0);
        println!("{:<10} | {:<20}", pattern, format!("{:.2}%", expected));
    }
    println!("{:-^60}", "");
}
