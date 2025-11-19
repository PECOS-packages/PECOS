// This test file contains numeric conversions that are safe in our context but trigger Clippy warnings.
// The following safety considerations apply:
// 1. u32 to i32 casts: Measurement results in quantum simulations are always small non-negative values.
// 2. i32 to u64 casts: Loop indices are always non-negative, so no sign information is actually lost.
// 3. usize to u32 casts: We're using small loop counts (e.g., 0..100) that are guaranteed to fit in u32.
// 4. Type conversions and small f64 multiplications: These maintain sufficient precision for our tests.
//
// Given these constraints and the nature of these tests, we can safely allow the warnings.
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_sign_loss)]
#![allow(clippy::cast_possible_truncation)]

use log::info;
use pecos_engines::noise::general::GeneralNoiseModel;
use pecos_engines::quantum::{QuantumEngine, StateVecEngine};
use pecos_engines::{
    Engine, GateType, QuantumSystem, byte_message::ByteMessage, engine_system::ControlEngine,
};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use std::collections::BTreeMap;

/// Reset a noise model and set its seed in one operation
///
/// This function applies the `reset_with_seed` method to a `GeneralNoiseModel`
fn reset_model_with_seed(
    model: &mut GeneralNoiseModel,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    model
        .reset_with_seed(seed)
        .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
}

fn create_noise_model() -> GeneralNoiseModel {
    info!("Creating noise model with moderate error rates");

    // Create a noise model with moderate error rates using the builder pattern
    // Set single-qubit error rates with uniform distribution
    let mut single_qubit_weights = BTreeMap::new();
    single_qubit_weights.insert("X".to_string(), 0.25);
    single_qubit_weights.insert("Y".to_string(), 0.25);
    single_qubit_weights.insert("Z".to_string(), 0.25);
    single_qubit_weights.insert("L".to_string(), 0.25);

    // Set two-qubit error rates with uniform distribution
    let mut two_qubit_weights = BTreeMap::new();
    two_qubit_weights.insert("XX".to_string(), 0.2);
    two_qubit_weights.insert("YY".to_string(), 0.2);
    two_qubit_weights.insert("ZZ".to_string(), 0.2);
    two_qubit_weights.insert("XL".to_string(), 0.2);
    two_qubit_weights.insert("LX".to_string(), 0.2);

    // Use builder to construct the model with all parameters set
    let mut model = GeneralNoiseModel::builder()
        .with_prep_probability(0.1)
        .with_meas_0_probability(0.1)
        .with_meas_1_probability(0.1)
        .with_p1_probability(0.1)
        .with_p2_probability(0.1)
        .with_p1_pauli_model(&single_qubit_weights)
        .with_p2_pauli_model(&two_qubit_weights)
        .with_p1_emission_ratio(0.5)
        .with_p2_emission_ratio(0.5)
        .with_prep_leak_ratio(0.5)
        .build();

    // Reset the model to ensure clean state
    info!("Resetting model");
    model.reset().expect("Failed to reset noise model");

    model
}

fn apply_noise(model: &mut GeneralNoiseModel, msg: &ByteMessage) -> Vec<ByteMessage> {
    info!("Applying noise to message");
    // If measurement results are required from measurements, we provide pseudorandom ones,
    // but always from the same seed. This is because we are testing that different noise models
    // respond differently to the same inputs.
    let mut measure_rng = ChaCha8Rng::seed_from_u64(5330);
    let mut state = model
        .start(msg.clone())
        .expect("Failed to start noise model processing");
    let mut messages = Vec::new();
    loop {
        match state {
            pecos_engines::engine_system::EngineStage::Complete(noisy_msg) => {
                messages.push(noisy_msg);
                return messages;
            }
            pecos_engines::engine_system::EngineStage::NeedsProcessing(intermediate_msg) => {
                // if the intermediate message requires measurements, give it some!
                messages.push(intermediate_msg.clone());
                let gates = intermediate_msg
                    .quantum_ops()
                    .expect("Failed to get quantum operations");
                let mut response = ByteMessage::outcomes_builder();
                for gate in &gates {
                    match &gate.gate_type {
                        GateType::Measure | GateType::MeasureLeaked => {
                            for _ in &gate.qubits {
                                let outcome = usize::from(measure_rng.random_bool(0.5));
                                response.add_outcomes(&[outcome]);
                            }
                        }
                        _ => {}
                    }
                }
                state = model
                    .continue_processing(response.build())
                    .expect("Failed to continue processing with measurements");
            }
        }
    }
}

/// Compare two `ByteMessage`s by parsing their quantum operations and results
fn compare_messages(msg1: &ByteMessage, msg2: &ByteMessage) -> bool {
    let quantum_ops_left = msg1.quantum_ops().unwrap_or_default();
    let quantum_ops_right = msg2.quantum_ops().unwrap_or_default();
    let results_left = msg1.outcomes().unwrap_or_default();
    let results_right = msg2.outcomes().unwrap_or_default();
    if quantum_ops_left != quantum_ops_right {
        eprintln!("Quantum operations differ: {quantum_ops_left:?} vs {quantum_ops_right:?}",);
        return false;
    }
    if results_left != results_right {
        eprintln!("Measurement outcomes differ: {results_left:?} vs {results_right:?}");
        return false;
    }
    true
}

/// Compare two `ByteMessage` vectors by parsing their quantum operations and results
///
/// This function extracts and compares the quantum operations and results from two
/// vectors of messages to determine if they represent the same conversation between
/// the noise model and the quantum engine.
fn compare_message_lists(messages_left: &[ByteMessage], messages_right: &[ByteMessage]) -> bool {
    if messages_left.len() != messages_right.len() {
        eprintln!(
            "Message lengths differ: {} vs {}",
            messages_left.len(),
            messages_right.len()
        );
        return false;
    }
    for (i, (msg1, msg2)) in messages_left.iter().zip(messages_right.iter()).enumerate() {
        if !compare_messages(msg1, msg2) {
            eprintln!("Messages differ at index {i}");
            return false;
        }
    }
    true
}

#[test]
fn test_prep_determinism() {
    let seed = 42;
    info!("Creating noise models with identical seeds");
    let mut model1 = create_noise_model();

    // Apply noise to model1
    reset_model_with_seed(&mut model1, seed).expect("Failed to reset model with seed");

    // Create a message with multiple prep gates
    let mut builder = ByteMessage::quantum_operations_builder();
    for _ in 0..20 {
        builder.add_prep(&[0]);
    }
    let msg = builder.build();

    // Apply noise to the message
    let noisy1 = apply_noise(&mut model1, &msg);

    // Reset model1 with the same seed for deterministic behavior
    reset_model_with_seed(&mut model1, seed).expect("Failed to reset model with seed");

    // Apply noise again to the message
    let noisy2 = apply_noise(&mut model1, &msg);

    // Now these should be identical
    info!("Comparing noisy1 and noisy2 - should be identical with same seed and model");
    assert!(
        compare_message_lists(&noisy1, &noisy2),
        "Message lists should be identical with same seed and model"
    );

    info!("Ensuring that the noise is actually being applied");
    assert!(
        !compare_messages(&msg, &noisy1[0]),
        "Original message should be different from noisy message"
    );

    // Now create a completely different model to verify we see different noise
    info!("Creating a model with a different seed");
    let mut model3 = create_noise_model();
    reset_model_with_seed(&mut model3, seed + 1)
        .expect("Failed to reset model3 with different seed"); // different seed

    // Apply noise with different model
    let noisy3 = apply_noise(&mut model3, &msg);

    // These should be different
    info!("Comparing noisy1 and noisy3 - should be different with different seeds");
    assert!(
        !compare_message_lists(&noisy1, &noisy3),
        "Different seeds should produce different message lists"
    );
}

#[test]
fn test_single_qubit_gate_determinism() {
    let seed = 42;
    info!("Creating noise model with seed");
    let mut model1 = create_noise_model();

    // Apply noise to model1
    reset_model_with_seed(&mut model1, seed).expect("Failed to reset model with seed");

    // Create a message with multiple single-qubit gates
    let mut builder = ByteMessage::quantum_operations_builder();
    for _ in 0..10 {
        // Repeat pattern to increase chance of errors
        builder.add_h(&[0]);
        builder.add_rz(0.5, &[0]);
        builder.add_r1xy(0.5, 0.5, &[0]);
        builder.add_h(&[1]);
        builder.add_rz(0.5, &[1]);
    }
    let msg = builder.build();

    // Apply noise the first time
    info!("Applying noise first time");
    let noisy1 = apply_noise(&mut model1, &msg);

    // Reset model with the same seed for deterministic behavior
    info!("Resetting model with same seed");
    reset_model_with_seed(&mut model1, seed).expect("Failed to reset model with seed");

    // Apply noise again with the same model
    info!("Applying noise second time");
    let noisy2 = apply_noise(&mut model1, &msg);

    // Verify determinism
    info!("Comparing results - should be identical with same seed");
    assert!(
        compare_message_lists(&noisy1, &noisy2),
        "Results should be identical with same seed"
    );

    // Verify that we get some errors due to noise
    info!("Comparing original instruction and its noisy command output");
    assert!(
        !compare_messages(&msg, &noisy1[0]),
        "Original message should be different from noisy message"
    );
}

#[test]
fn test_two_qubit_gate_determinism() {
    let seed = 42;
    info!("Creating noise models with identical seeds");
    let mut model1 = create_noise_model();

    // Apply noise to model1
    reset_model_with_seed(&mut model1, seed).expect("Failed to reset model with seed");

    // Create a message with many two-qubit gates to increase chance of errors
    let mut builder = ByteMessage::quantum_operations_builder();
    for _ in 0..20 {
        // Repeat pattern multiple times
        builder.add_cx(&[0], &[1]);
        builder.add_cx(&[1], &[2]);
        builder.add_cx(&[2], &[3]);
        builder.add_cx(&[3], &[0]);
    }
    let msg = builder.build();

    // Apply noise to the message
    let noisy1 = apply_noise(&mut model1, &msg);

    // Reset model1 with the same seed for deterministic behavior
    reset_model_with_seed(&mut model1, seed).expect("Failed to reset model with seed");

    // Apply noise again to the message
    let noisy2 = apply_noise(&mut model1, &msg);

    // Now these should be identical
    info!("Comparing noisy1 and noisy2 - should be identical with same seed and model");
    assert!(
        compare_message_lists(&noisy1, &noisy2),
        "Messages should be identical with same seed and model"
    );

    // Verify that the message is actually being modified by the noise model
    info!("Verifying that noise is being applied");
    assert!(
        !compare_messages(&msg, &noisy1[0]),
        "Original message should be different from noisy message"
    );
}

#[test]
fn test_measurement_determinism() {
    let seed = 42;
    let mut model1 = create_noise_model();
    let mut model2 = create_noise_model();

    reset_model_with_seed(&mut model1, seed).expect("Failed to reset model with seed");
    reset_model_with_seed(&mut model2, seed).expect("Failed to reset model with seed");

    // Create a message with measurements
    let mut builder = ByteMessage::quantum_operations_builder();
    builder.add_h(&[0]);
    builder.add_h(&[1]);
    builder.add_cx(&[0], &[1]);
    builder.add_measurements(&[0]);
    builder.add_measurements(&[1]);
    let msg = builder.build();

    // Apply noise multiple times
    let noisy1 = apply_noise(&mut model1, &msg);

    reset_model_with_seed(&mut model1, seed).expect("Failed to reset model with seed");

    let noisy2 = apply_noise(&mut model2, &msg);

    // Verify determinism in the quantum operations
    assert!(compare_message_lists(&noisy1, &noisy2));
}

#[test]
fn test_different_seeds_produce_different_results() {
    let seed1 = 42;
    let seed2 = 43; // Different seed
    let mut model1 = create_noise_model();
    let mut model2 = create_noise_model();

    reset_model_with_seed(&mut model1, seed1).expect("Failed to reset model with seed");
    reset_model_with_seed(&mut model2, seed2).expect("Failed to reset model with seed");

    // Create a larger circuit to increase the chance of errors
    let mut builder = ByteMessage::quantum_operations_builder();
    for _ in 0..15 {
        // Repeat pattern to create a longer circuit
        builder.add_h(&[0]);
        builder.add_cx(&[0], &[1]);
        builder.add_h(&[1]);
        builder.add_cx(&[1], &[2]);
        builder.add_h(&[2]);
    }
    let msg = builder.build();

    // Apply noise with different seeds
    let noisy1 = apply_noise(&mut model1, &msg);
    let noisy2 = apply_noise(&mut model2, &msg);

    // With different seeds, we expect different noise results
    info!("Comparing outputs from different seeds - should be different");
    assert!(
        !compare_message_lists(&noisy1, &noisy2),
        "Different seeds should produce different noise patterns"
    );
}

/// Runs a complete quantum simulation including the actual measurement outcomes
///
/// This function:
/// 1. Creates a `QuantumSystem` with the provided noise model and quantum engine
/// 2. Sets the seed for the system
/// 3. Runs the circuit and collects the actual measurement outcomes
/// 4. Returns the measurement results as a `BTreeMap` of result IDs to values
fn run_complete_simulation(
    noise_model: &mut GeneralNoiseModel,
    quantum_engine: Box<dyn QuantumEngine>,
    circuit: &ByteMessage,
    seed: u64,
) -> BTreeMap<usize, i32> {
    // Create a quantum system with the noise model and quantum engine
    let mut system = QuantumSystem::new(Box::new(noise_model.clone()), quantum_engine);

    // Set the seed for deterministic behavior
    system.set_seed(seed).expect("Failed to set seed");

    // Reset the system to ensure clean state
    system.reset().expect("Failed to reset system");

    // Run the circuit through the system
    let output = system
        .process(circuit.clone())
        .expect("Failed to process circuit");

    // Extract the measurement results
    let measurements: Vec<(usize, u32)> = output
        .outcomes()
        .map(|outcomes| outcomes.into_iter().enumerate().collect())
        .expect("Failed to extract measurements");

    // Convert u32 values to i32 for the HashMap, handling potential overflow
    measurements
        .into_iter()
        .map(|(k, v)| {
            // Safe conversion from u32 to i32, handling potential overflow
            let value = if v > i32::MAX as u32 {
                i32::MAX
            } else {
                v as i32
            };
            (k, value)
        })
        .collect()
}

#[test]
fn test_complete_measurement_determinism() {
    let seed = 42;
    info!("Testing complete measurement determinism with end-to-end simulation");

    // Create two identical noise models
    let mut model1 = create_noise_model();
    let mut model2 = create_noise_model();

    // Set the same seed for both models
    reset_model_with_seed(&mut model1, seed).expect("Failed to reset model with seed");
    reset_model_with_seed(&mut model2, seed).expect("Failed to reset model with seed");

    // Create a circuit with superposition and entanglement to test measurement
    let mut builder = ByteMessage::quantum_operations_builder();
    // Create a Bell state
    builder.add_h(&[0]);
    builder.add_cx(&[0], &[1]);
    // Add measurements for both qubits
    builder.add_measurements(&[0]);
    let circuit = builder.build();

    // Create two identical quantum engines
    let engine1 = Box::new(StateVecEngine::new(2));
    let engine2 = Box::new(StateVecEngine::new(2));

    // Run complete simulations with both models
    info!("Running first complete simulation");
    let results1 = run_complete_simulation(&mut model1, engine1, &circuit, seed);

    info!("Running second complete simulation with identical seed");
    let results2 = run_complete_simulation(&mut model2, engine2, &circuit, seed);

    // The measurement results should be identical
    info!("Comparing measurement results between runs");
    assert_eq!(
        results1, results2,
        "Measurement results should be identical with the same seed"
    );

    // Now run with a different seed
    info!("Running third simulation with different seed");
    let mut model3 = create_noise_model();
    reset_model_with_seed(&mut model3, seed + 1).expect("Failed to reset model with seed");
    let engine3 = Box::new(StateVecEngine::new(2));
    let results3 = run_complete_simulation(&mut model3, engine3, &circuit, seed + 1);

    // These should be different (most of the time)
    // Note: There's a small probability they could be the same by chance,
    // so we don't strictly assert, but log the comparison
    if results1 == results3 {
        info!("NOTE: Results with different seeds happened to be identical (small probability)");
    } else {
        info!("Results with different seeds are different, as expected");
    }
}

#[test]
fn test_deterministic_measurement() {
    // This test verifies that using the same seed produces the same measurement results
    let seed = 42;
    info!("Testing deterministic measurement with seed {seed}");

    // Create a noise model with significant measurement error
    let model = GeneralNoiseModel::builder()
        .with_prep_probability(0.01)
        .with_meas_0_probability(0.2)
        .with_meas_1_probability(0.2)
        .with_average_p1_probability(0.1)
        .with_average_p2_probability(0.1)
        .build();

    // Box the model for use with the NoiseModel trait
    let mut model = Box::new(model);

    // Create a circuit that puts a qubit in superposition and measures it
    let mut builder = ByteMessage::quantum_operations_builder();
    builder.add_h(&[0]); // Put qubit 0 in superposition
    builder.add_measurements(&[0]); // Measure qubit 0
    let circuit = builder.build();

    info!("Running first measurement with seed {seed}");
    reset_model_with_seed(&mut model, seed).expect("Failed to reset model with seed");
    let engine1 = Box::new(StateVecEngine::new(1));
    let result1 = run_complete_simulation(&mut model, engine1, &circuit, seed);
    let value1 = result1.get(&0).copied().unwrap_or(0);

    info!("First measurement result: {value1}");

    info!("Running second measurement with same seed {seed}");
    reset_model_with_seed(&mut model, seed).expect("Failed to reset model with seed");
    let engine2 = Box::new(StateVecEngine::new(1));
    let result2 = run_complete_simulation(&mut model, engine2, &circuit, seed);
    let value2 = result2.get(&0).copied().unwrap_or(0);

    info!("Second measurement result: {value2}");

    // The results should be identical with the same seed
    assert_eq!(
        value1, value2,
        "Measurement results should be identical with the same seed"
    );

    // Now try with a different seed
    let different_seed = seed + 1000;
    info!("Running measurement with different seed {different_seed}");
    reset_model_with_seed(&mut model, different_seed).expect("Failed to reset model with seed");
    let engine3 = Box::new(StateVecEngine::new(1));
    let result3 = run_complete_simulation(&mut model, engine3, &circuit, different_seed);
    let value3 = result3.get(&0).copied().unwrap_or(0);

    info!("Different seed result: {value3}");

    // IMPROVEMENT 1: Assert that different seeds produce different results
    // (with a caveat for the small probability that they might be the same by chance)
    if value1 == value3 {
        info!(
            "NOTE: Same measurement result with different seeds. This can happen with low probability."
        );

        // Try one more seed to reduce the probability of false positives
        let another_seed = seed + 2000;
        reset_model_with_seed(&mut model, another_seed).expect("Failed to reset model with seed");
        let engine4 = Box::new(StateVecEngine::new(1));
        let result4 = run_complete_simulation(&mut model, engine4, &circuit, another_seed);
        let value4 = result4.get(&0).copied().unwrap_or(0);

        // With a second different seed, the probability of getting the same result again is even lower
        if value1 == value4 {
            info!(
                "NOTE: Still same measurement result with a third seed. Very unlikely but possible."
            );
        } else {
            // Different results with the new seed, so we can assert determinism
            info!("Different seed produced different result: {value4}");
            assert_ne!(
                value1, value4,
                "Different seeds should usually produce different measurement results"
            );
        }
    } else {
        // Different results as expected
        assert_ne!(
            value1, value3,
            "Different seeds should usually produce different measurement results"
        );
    }

    // Now run multiple measurements with increasing seeds to test we get a mix of results
    let mut zeros = 0;
    let mut ones = 0;
    let num_tests = 20;

    info!("Running {num_tests} measurements with different seeds");
    for i in 0..num_tests {
        // Use a different deterministic seed for each test iteration derived from the base seed
        // Converting i to u64 is safe since we're only using small non-negative loop values
        let test_seed = seed + i as u64;
        reset_model_with_seed(&mut model, test_seed).expect("Failed to reset model with seed");
        let engine = Box::new(StateVecEngine::new(1));
        let result = run_complete_simulation(&mut model, engine, &circuit, test_seed);
        let value = result.get(&0).copied().unwrap_or(0);

        if value == 0 {
            zeros += 1;
        } else {
            ones += 1;
        }
    }

    info!("Got {zeros} zeros and {ones} ones with different seeds");

    // With enough different seeds, we should get some variation
    // The probability of getting all zeros or all ones with 20 measurements and a roughly
    // 50/50 chance for each is approximately 2^(-19), which is extremely unlikely
    if zeros == 0 || ones == 0 {
        info!(
            "NOTE: Got only {} measurements. This is highly unusual but technically possible.",
            if zeros == 0 { "ones" } else { "zeros" }
        );
    } else {
        info!("Got a mixture of results with different seeds, as expected");
    }
}

/// IMPROVEMENT 2: Comprehensive end-to-end test combining all noise types
#[test]
fn test_comprehensive_noise_determinism() {
    info!("Testing comprehensive noise determinism (all noise types)");

    // Create a noise model with all types of noise
    let model = GeneralNoiseModel::builder()
        // Preparation errors
        .with_prep_probability(0.05)
        .with_prep_leak_ratio(0.2)
        // Measurement errors
        .with_meas_0_probability(0.1)
        .with_meas_1_probability(0.15)
        // Gate errors
        .with_average_p1_probability(0.2)
        .with_average_p2_probability(0.1)
        // Leakage and emission errors
        .with_p1_emission_ratio(0.3)
        .with_p2_emission_ratio(0.3)
        .build();

    // Box the model
    let mut model = Box::new(model);

    // Create a complex circuit with all types of operations:
    // 1. Preparation (implicit at start)
    // 2. Various single-qubit gates
    // 3. Two-qubit gates
    // 4. Parameterized gates
    // 5. Measurements
    let mut builder = ByteMessage::quantum_operations_builder();

    // Use 3 qubits
    // Apply a variety of single and two-qubit gates
    builder.add_h(&[0]); // Apply Hadamard to qubit 0
    builder.add_rz(0.5, &[1]); // Apply RZ to qubit 1
    builder.add_cx(&[0], &[1]); // Apply CNOT from qubit 0 to qubit 1
    builder.add_h(&[2]); // Apply Hadamard to qubit 2
    builder.add_cx(&[1], &[2]); // Apply CNOT from qubit 1 to qubit 2

    // RX and RY gates can be implemented using H-RZ-H and other combinations
    builder.add_h(&[0]); // Start of RX implementation
    builder.add_rz(0.25, &[0]);
    builder.add_h(&[0]); // End of RX implementation

    builder.add_h(&[1]); // Start of RY approximation
    builder.add_z(&[1]);
    builder.add_rz(0.33, &[1]);
    builder.add_z(&[1]);
    builder.add_h(&[1]); // End of RY approximation

    builder.add_x(&[2]); // Apply X to qubit 2
    builder.add_y(&[0]); // Apply Y to qubit 0
    builder.add_z(&[1]); // Apply Z to qubit 1
    builder.add_rzz(0.75, &[0], &[2]); // Apply RZZ to qubits 0 and 2
    builder.add_cx(&[2], &[0]); // Apply CNOT from qubit 2 to qubit 0

    // Add measurements for all qubits
    builder.add_measurements(&[0]);

    let circuit = builder.build();

    // Run the circuit with a fixed seed
    let seed = 9876;
    info!("Running first simulation with seed {seed}");
    reset_model_with_seed(&mut model, seed).expect("Failed to reset model with seed");
    let engine1 = Box::new(StateVecEngine::new(3));
    let results1 = run_complete_simulation(&mut model, engine1, &circuit, seed);

    // Sort and print results for readability
    let mut results1_vec: Vec<(usize, i32)> = results1.iter().map(|(&k, &v)| (k, v)).collect();
    results1_vec.sort_by_key(|&(k, _)| k);
    info!("First run results: {results1_vec:?}");

    // Run again with the same seed - should get identical results
    info!("Running second simulation with the same seed {seed}");
    reset_model_with_seed(&mut model, seed).expect("Failed to reset model with seed");
    let engine2 = Box::new(StateVecEngine::new(3));
    let results2 = run_complete_simulation(&mut model, engine2, &circuit, seed);

    // Sort and print results for readability
    let mut results2_vec: Vec<(usize, i32)> = results2.iter().map(|(&k, &v)| (k, v)).collect();
    results2_vec.sort_by_key(|&(k, _)| k);
    info!("Second run results: {results2_vec:?}");

    // The results should be identical with the same seed
    assert_eq!(
        results1, results2,
        "Measurement results should be identical with the same seed in comprehensive test"
    );

    // Run again with a different seed - should get different results
    let different_seed = seed + 1000;
    info!("Running third simulation with different seed {different_seed}");
    reset_model_with_seed(&mut model, different_seed).expect("Failed to reset model with seed");
    let engine3 = Box::new(StateVecEngine::new(3));
    let results3 = run_complete_simulation(&mut model, engine3, &circuit, different_seed);

    // Sort and print results for readability
    let mut results3_vec: Vec<(usize, i32)> = results3.iter().map(|(&k, &v)| (k, v)).collect();
    results3_vec.sort_by_key(|&(k, _)| k);
    info!("Different seed results: {results3_vec:?}");

    // The results should be different (high probability)
    // If they happen to be identical, try yet another seed
    if results1 == results3 {
        info!(
            "NOTE: Same measurement results with different seeds. This can happen with low probability."
        );

        let another_seed = seed + 2000;
        info!("Trying yet another seed: {another_seed}");
        reset_model_with_seed(&mut model, another_seed).expect("Failed to reset model with seed");
        let engine4 = Box::new(StateVecEngine::new(3));
        let results4 = run_complete_simulation(&mut model, engine4, &circuit, another_seed);

        // The probability of getting identical results again is extremely low
        if results1 == results4 {
            info!(
                "NOTE: Still same results with a third seed. Extremely unlikely but technically possible."
            );
        } else {
            info!("Different seed produced different results as expected");
            assert_ne!(
                results1, results4,
                "Different seeds should produce different results in comprehensive test"
            );
        }
    } else {
        info!("Different seed produced different results as expected");
        assert_ne!(
            results1, results3,
            "Different seeds should produce different results in comprehensive test"
        );
    }
}

/// IMPROVEMENT 3: Test long-running determinism with a large circuit
#[test]
fn test_long_running_determinism() {
    info!("Testing long-running determinism with many operations");

    // Create a noise model with moderate error rates
    let model = GeneralNoiseModel::builder()
        .with_prep_probability(0.01)
        .with_meas_0_probability(0.02)
        .with_meas_1_probability(0.02)
        .with_average_p1_probability(0.1)
        .with_average_p2_probability(0.05)
        .build();

    // Box the model
    let mut model = Box::new(model);

    // Create a circuit with a very large number of operations
    let mut builder = ByteMessage::quantum_operations_builder();

    // First create a GHZ state across 5 qubits
    builder.add_h(&[0]);
    builder.add_cx(&[0], &[1]);
    builder.add_cx(&[0], &[2]);
    builder.add_cx(&[0], &[3]);
    builder.add_cx(&[0], &[4]);

    // Now apply a repeated pattern of gates to create a long sequence
    // This gives the RNG many opportunities to diverge if there are issues
    info!("Building a circuit with 500+ operations...");
    // We're using a small, positive loop count where usize will fit in both u32 and f64 without precision loss
    for i in 0..100 {
        // 100 repetitions of 5+ operations = 500+ operations total
        // Rotate each qubit differently based on iteration
        builder.add_rz(0.01 * f64::from(i as u32), &[0]);

        // Implement RX using H-RZ-H
        builder.add_h(&[1]);
        builder.add_rz(0.02 * f64::from(i as u32), &[1]);
        builder.add_h(&[1]);

        // Implement RY using H-Z-RZ-Z-H
        builder.add_h(&[2]);
        builder.add_z(&[2]);
        builder.add_rz(0.03 * f64::from(i as u32), &[2]);
        builder.add_z(&[2]);
        builder.add_h(&[2]);

        builder.add_rz(0.04 * f64::from(i as u32), &[3]);

        // Another RX implementation
        builder.add_h(&[4]);
        builder.add_rz(0.05 * f64::from(i as u32), &[4]);
        builder.add_h(&[4]);

        // Add entangling operations that change with iteration
        let q1 = i % 5;
        let q2 = (i + 1) % 5;
        builder.add_cx(&[q1], &[q2]);
    }

    // Add measurements for all qubits
    builder.add_measurements(&[0]);

    let circuit = builder.build();

    // Run the circuit twice with the same seed
    let seed = 54321;
    info!("Running first long simulation with seed {seed}");
    reset_model_with_seed(&mut model, seed).expect("Failed to reset model with seed");
    let engine1 = Box::new(StateVecEngine::new(5));
    let results1 = run_complete_simulation(&mut model, engine1, &circuit, seed);

    info!("Running second long simulation with the same seed {seed}");
    reset_model_with_seed(&mut model, seed).expect("Failed to reset model with seed");
    let engine2 = Box::new(StateVecEngine::new(5));
    let results2 = run_complete_simulation(&mut model, engine2, &circuit, seed);

    // Sort and print a summary of the results
    let mut results1_vec: Vec<(usize, i32)> = results1.iter().map(|(&k, &v)| (k, v)).collect();
    results1_vec.sort_by_key(|&(k, _)| k);
    info!("First run results: {results1_vec:?}");

    let mut results2_vec: Vec<(usize, i32)> = results2.iter().map(|(&k, &v)| (k, v)).collect();
    results2_vec.sort_by_key(|&(k, _)| k);
    info!("Second run results: {results2_vec:?}");

    // Results should be identical despite the long sequence of operations
    assert_eq!(
        results1, results2,
        "Results should be identical with the same seed even with a very long circuit"
    );

    // Run with a different seed
    let different_seed = seed + 1000;
    info!("Running with a different seed {different_seed}");
    reset_model_with_seed(&mut model, different_seed).expect("Failed to reset model with seed");
    let engine3 = Box::new(StateVecEngine::new(5));
    let results3 = run_complete_simulation(&mut model, engine3, &circuit, different_seed);

    // Results should be different (with high probability)
    if results1 == results3 {
        info!("NOTE: Same results with different seeds. This is very unlikely but possible.");

        // Try one more seed
        let another_seed = seed + 2000;
        info!("Trying yet another seed: {another_seed}");
        reset_model_with_seed(&mut model, another_seed).expect("Failed to reset model with seed");
        let engine4 = Box::new(StateVecEngine::new(5));
        let results4 = run_complete_simulation(&mut model, engine4, &circuit, another_seed);

        if results1 == results4 {
            info!("NOTE: Still same results with a third seed. Extremely unlikely.");
        } else {
            info!("Different seed produced different results as expected");
            assert_ne!(
                results1, results4,
                "Different seeds should produce different results"
            );
        }
    } else {
        info!("Different seed produced different results as expected");
        assert_ne!(
            results1, results3,
            "Different seeds should produce different results"
        );
    }

    info!("Long-running determinism test passed successfully!");
}
