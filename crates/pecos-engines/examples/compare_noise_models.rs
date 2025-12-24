use pecos_engines::byte_message::ByteMessage;
use pecos_engines::noise::{DepolarizingNoiseModel, GeneralNoiseModel};
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{Engine, EngineSystem, QuantumSystem};
use std::collections::BTreeMap;

fn main() {
    // Create the same Bell state circuit as in run_noisy_circ.rs
    let circ = ByteMessage::quantum_operations_builder()
        .add_h(&[0])
        .add_cx(&[0], &[1])
        .add_measurements(&[0])
        .add_measurements(&[1])
        .build();

    // Test that GeneralNoise can reproduce DepolarizingNoise behavior
    compare_depolarizing_with_general(&circ);

    // Also test with a few different noise parameter sets
    println!("\nTesting with asymmetric measurement errors:");
    test_asymmetric_measurements();
}

#[allow(clippy::too_many_lines)]
fn compare_depolarizing_with_general(circ: &ByteMessage) {
    // Same noise parameters as in run_noisy_circ.rs
    let p_noise = 0.1;
    let seed = 42;
    let num_shots = 20;

    // Create quantum engine
    let quantum = StateVecEngine::new(2);

    // Create depolarizing noise model with same seed
    let depolarizing_noise = DepolarizingNoiseModel::builder()
        .with_uniform_probability(p_noise)
        .with_seed(seed)
        .build();
    let mut depolarizing_system =
        QuantumSystem::new(Box::new(depolarizing_noise), Box::new(quantum.clone()));

    // Create equivalent general noise model
    let general_noise = GeneralNoiseModel::builder()
        .with_prep_probability(p_noise)
        .with_meas_0_probability(p_noise)
        .with_meas_1_probability(p_noise)
        .with_p1_probability(p_noise)
        .with_p2_probability(p_noise)
        .with_seed(seed)
        .build();
    let mut general_system = QuantumSystem::new(Box::new(general_noise), Box::new(quantum.clone()));

    println!("Comparing DepolarizingNoise vs GeneralNoise with p = {p_noise}");
    println!("{:-^80}", "");
    println!(
        "{:<10} | {:<32} | {:<32}",
        "Shot", "DepolarizingNoise", "GeneralNoise"
    );
    println!("{:-^80}", "");

    // Collect results from both noise models
    let mut depolarizing_results = Vec::new();
    let mut general_results = Vec::new();

    for i in 0..num_shots {
        // Run with depolarizing noise
        depolarizing_system
            .reset()
            .expect("Failed to reset depolarizing system");
        let results = depolarizing_system
            .process_as_system(circ.clone())
            .expect("Failed to process with depolarizing noise");
        let measurements = results
            .outcomes()
            .expect("Failed to parse depolarizing measurements");

        // Format result string
        let result_str = measurements
            .iter()
            .map(|&value| value.to_string())
            .collect::<String>();

        depolarizing_results.push(result_str);

        // Run with general noise
        general_system
            .reset()
            .expect("Failed to reset general system");
        let results = general_system
            .process_as_system(circ.clone())
            .expect("Failed to process with general noise");
        let measurements = results
            .outcomes()
            .expect("Failed to parse general measurements");

        // Format result string
        let result_str = measurements
            .iter()
            .map(|&value| value.to_string())
            .collect::<String>();

        general_results.push(result_str);

        // Print the results for this shot
        println!(
            "{:<10} | {:<32} | {:<32}",
            i + 1,
            depolarizing_results[i],
            general_results[i]
        );
    }

    println!("{:-^80}", "");

    // Compare how many results match between the two models
    let matching_count = depolarizing_results
        .iter()
        .zip(general_results.iter())
        .filter(|(a, b)| a == b)
        .count();

    println!(
        "\nMatching results: {}/{} ({}%)",
        matching_count,
        num_shots,
        matching_count * 100 / num_shots
    );

    // Count distribution of each outcome
    let mut depolarizing_counts = BTreeMap::new();
    let mut general_counts = BTreeMap::new();

    for result in &depolarizing_results {
        *depolarizing_counts.entry(result.clone()).or_insert(0) += 1;
    }

    for result in &general_results {
        *general_counts.entry(result.clone()).or_insert(0) += 1;
    }

    println!("\nResult distributions:");
    println!("{:-^80}", "");
    println!(
        "{:<10} | {:<32} | {:<32}",
        "Result", "DepolarizingNoise", "GeneralNoise"
    );
    println!("{:-^80}", "");

    let patterns = ["00", "01", "10", "11"];
    for pattern in &patterns {
        let dep_count = depolarizing_counts.get(*pattern).unwrap_or(&0);
        let gen_count = general_counts.get(*pattern).unwrap_or(&0);

        println!(
            "{:<10} | {:<32} | {:<32}",
            pattern,
            format!("{} ({}%)", dep_count, dep_count * 100 / num_shots),
            format!("{} ({}%)", gen_count, gen_count * 100 / num_shots)
        );
    }
    println!("{:-^80}", "");
}

#[allow(clippy::similar_names)]
fn test_asymmetric_measurements() {
    // Create a simple circuit with H-gate and measurement
    let circ = ByteMessage::quantum_operations_builder()
        .add_h(&[0])
        .add_measurements(&[0])
        .build();

    // Create quantum engine
    let quantum = StateVecEngine::new(1);

    // Test parameters
    let num_shots = 10000;
    let seed = 42;

    // Create general noise with asymmetric measurement errors
    let p_prep = 0.05;
    let p_meas_0 = 0.2; // 20% chance to flip 0->1
    let p_meas_1 = 0.1; // 10% chance to flip 1->0
    let p1 = 0.05;

    let general_noise = GeneralNoiseModel::builder()
        .with_prep_probability(p_prep)
        .with_meas_0_probability(p_meas_0)
        .with_meas_1_probability(p_meas_1)
        .with_p1_probability(p1)
        .with_p2_probability(0.0) // Not used in this circuit
        .with_seed(seed)
        .build();
    let mut general_system = QuantumSystem::new(Box::new(general_noise), Box::new(quantum.clone()));

    // For comparison, a depolarizing model with symmetric errors
    let p_depolarizing = f64::midpoint(p_meas_0, p_meas_1); // Average of the asymmetric errors
    let depolarizing_noise = DepolarizingNoiseModel::builder()
        .with_prep_probability(p_prep)
        .with_meas_probability(p_depolarizing)
        .with_single_qubit_probability(p1)
        .with_two_qubit_probability(0.0)
        .with_seed(seed)
        .build();
    let mut depolarizing_system =
        QuantumSystem::new(Box::new(depolarizing_noise), Box::new(quantum.clone()));

    // Run simulations
    let mut general_counts = BTreeMap::new();
    let mut depolarizing_counts = BTreeMap::new();

    for _ in 0..num_shots {
        // Run with general noise
        general_system
            .reset()
            .expect("Failed to reset general system");
        let results = general_system
            .process_as_system(circ.clone())
            .expect("Failed to process with general noise");
        let measurements = results
            .outcomes()
            .expect("Failed to parse general measurements");
        let result = measurements
            .first()
            .map_or("?", |&v| if v == 1 { "1" } else { "0" });
        *general_counts.entry(result.to_string()).or_insert(0) += 1;

        // Run with depolarizing noise
        depolarizing_system
            .reset()
            .expect("Failed to reset depolarizing system");
        let results = depolarizing_system
            .process_as_system(circ.clone())
            .expect("Failed to process with depolarizing noise");
        let measurements = results
            .outcomes()
            .expect("Failed to parse depolarizing measurements");
        let result = measurements
            .first()
            .map_or("?", |&v| if v == 1 { "1" } else { "0" });
        *depolarizing_counts.entry(result.to_string()).or_insert(0) += 1;
    }

    // Calculate percentages
    let general_pct_0 = general_counts.get("0").unwrap_or(&0) * 100 / num_shots;
    let general_pct_1 = general_counts.get("1").unwrap_or(&0) * 100 / num_shots;

    let depol_pct_0 = depolarizing_counts.get("0").unwrap_or(&0) * 100 / num_shots;
    let depol_pct_1 = depolarizing_counts.get("1").unwrap_or(&0) * 100 / num_shots;

    println!("Asymmetric measurement errors (GeneralNoise):");
    println!("  p_prep = {p_prep}, p1 = {p1}, p_meas_0 = {p_meas_0}, p_meas_1 = {p_meas_1}");
    println!("  Results: 0: {general_pct_0}%, 1: {general_pct_1}%");

    println!("\nSymmetric measurement errors (DepolarizingNoise):");
    println!("  p_prep = {p_prep}, p1 = {p1}, p_meas = {p_depolarizing}");
    println!("  Results: 0: {depol_pct_0}%, 1: {depol_pct_1}%");

    // Calculate expected percentages for asymmetric case
    let expected_p0_asymmetric = 0.5 * (1.0 - p_meas_0) + 0.5 * p_meas_1;
    let expected_p1_asymmetric = 0.5 * p_meas_0 + 0.5 * (1.0 - p_meas_1);

    // Calculate expected percentages for symmetric case
    let expected_p0_symmetric = 0.5 * (1.0 - p_depolarizing) + 0.5 * p_depolarizing;
    let expected_p1_symmetric = 0.5 * p_depolarizing + 0.5 * (1.0 - p_depolarizing);

    println!("\nTheoretical expectations (ignoring gate/prep errors):");
    println!(
        "  Asymmetric: 0: {:.1}%, 1: {:.1}%",
        expected_p0_asymmetric * 100.0,
        expected_p1_asymmetric * 100.0
    );
    println!(
        "  Symmetric:  0: {:.1}%, 1: {:.1}%",
        expected_p0_symmetric * 100.0,
        expected_p1_symmetric * 100.0
    );
}
