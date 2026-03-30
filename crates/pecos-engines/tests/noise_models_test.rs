use pecos_engines::byte_message::ByteMessage;
use pecos_engines::noise::{
    BiasedDepolarizingNoiseModel, DepolarizingNoiseModel, NoiseModel, PassThroughNoiseModel,
};
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{Engine, EngineSystem, QuantumSystem};
use std::collections::BTreeMap;

// Helper function to count measurement results from multiple shots
fn count_results(
    noise_model: Box<dyn NoiseModel>,
    circ: &ByteMessage,
    num_shots: usize,
    num_qubits: usize,
) -> BTreeMap<String, usize> {
    let quantum = Box::new(StateVecEngine::new(num_qubits));
    let mut system = QuantumSystem::new(noise_model, quantum);
    system.set_seed(42);

    let mut counts = BTreeMap::new();
    for _ in 0..num_shots {
        system.reset().expect("Failed to reset system");
        let results = system
            .process_as_system(circ.clone())
            .expect("Failed to process circuit");
        let measurements = results.outcomes().expect("Failed to parse measurements");

        let result_str = measurements
            .iter()
            .map(|&value| if value == 1 { '1' } else { '0' })
            .collect::<String>();

        *counts.entry(result_str).or_insert(0) += 1;
    }

    counts
}

#[test]
fn test_biased_depolarizing_noise() {
    // Create a simple H-gate circuit with measurement
    let circ = ByteMessage::quantum_operations_builder()
        .h(&[0])
        .mz(&[0])
        .build();

    // Test with uniform depolarizing probability
    let uniform_noise = BiasedDepolarizingNoiseModel::builder()
        .with_uniform_probability(0.1)
        .with_seed(42)
        .build();

    // Get distribution after 1000 shots
    let counts = count_results(Box::new(uniform_noise), &circ, 1000, 1);

    // With Hadamard, we expect roughly 50% 0s and 50% 1s
    let count_0 = *counts.get("0").unwrap_or(&0);
    let count_1 = *counts.get("1").unwrap_or(&0);

    // Allow for some statistical variance (±10%)
    #[allow(clippy::cast_precision_loss)]
    let diff = (count_0 as f64 - count_1 as f64).abs() / 1000.0;
    assert!(
        diff <= 0.1,
        "Expected approximately even distribution, got {count_0} zeros and {count_1} ones"
    );
}

#[test]
fn test_depolarizing_noise() {
    // Create a Bell state circuit
    let bell_circ = ByteMessage::quantum_operations_builder()
        .h(&[0])
        .cx(&[(0, 1)])
        .mz(&[0])
        .mz(&[1])
        .build();

    // Test with no noise - should get ideal Bell state (00 and 11)
    let no_noise = DepolarizingNoiseModel::builder()
        .with_uniform_probability(0.0)
        .with_seed(42)
        .build();

    let ideal_counts = count_results(Box::new(no_noise), &bell_circ, 1000, 2);

    // In the ideal case, we expect only 00 and 11 with roughly equal probability
    assert!(
        ideal_counts.get("01").unwrap_or(&0) + ideal_counts.get("10").unwrap_or(&0) < 50,
        "Ideal Bell state should have negligible 01 and 10 results"
    );

    let count_00 = *ideal_counts.get("00").unwrap_or(&0);
    let count_11 = *ideal_counts.get("11").unwrap_or(&0);

    // Allow truncation and wrap in test code
    #[allow(clippy::cast_possible_truncation, clippy::cast_possible_wrap)]
    let diff = (count_00 as i32 - count_11 as i32).abs();

    assert!(
        diff < 100,
        "00 and 11 should be approximately equal in ideal Bell state"
    );

    // Test with moderate noise - should see some 01 and 10 results
    let moderate_noise = DepolarizingNoiseModel::builder()
        .with_uniform_probability(0.1)
        .with_seed(42)
        .build();

    let noisy_counts = count_results(Box::new(moderate_noise), &bell_circ, 1000, 2);

    // With noise, we expect to see some 01 and 10 results
    assert!(
        noisy_counts.get("01").unwrap_or(&0) + noisy_counts.get("10").unwrap_or(&0) > 50,
        "Noisy Bell state should have some 01 and 10 results"
    );
}

#[test]
fn test_pass_through_noise() {
    // Create a simple H-gate circuit with measurement
    let circ = ByteMessage::quantum_operations_builder()
        .h(&[0])
        .mz(&[0])
        .build();

    // Create pass-through noise model (no noise)
    let no_noise = Box::new(PassThroughNoiseModel::new());

    // Run with 1000 shots
    let counts = count_results(no_noise, &circ, 1000, 1);

    // With a Hadamard gate, we expect roughly 50% 0s and 50% an 1s
    let count_0 = *counts.get("0").unwrap_or(&0);
    let count_1 = *counts.get("1").unwrap_or(&0);

    // Allow for some statistical variance (±5%)
    #[allow(clippy::cast_precision_loss)]
    let diff_0 = (count_0 as f64 - 500.0).abs();
    #[allow(clippy::cast_precision_loss)]
    let diff_1 = (count_1 as f64 - 500.0).abs();

    assert!(
        diff_0 <= 50.0,
        "Expected approximately 500 zeros, got {count_0}"
    );
    assert!(
        diff_1 <= 50.0,
        "Expected approximately 500 ones, got {count_1}"
    );
}
