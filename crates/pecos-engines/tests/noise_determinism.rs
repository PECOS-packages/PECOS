use log::info;
use pecos_engines::{
    byte_message::ByteMessage,
    engines::ControlEngine,
    engines::noise::{NoiseModel, general::GeneralNoiseModel},
};
use std::collections::HashMap;

/// Reset a noise model and set its seed in one operation
///
/// This function works with boxed noise models and takes care of
/// downcasting to `GeneralNoiseModel` to use the `reset_with_seed` method.
fn reset_model_with_seed(
    model: &mut Box<dyn NoiseModel>,
    seed: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let general_noise = model
        .as_any_mut()
        .downcast_mut::<GeneralNoiseModel>()
        .unwrap();
    general_noise.reset_with_seed(seed)
}

fn create_noise_model() -> Box<dyn NoiseModel> {
    info!("Creating noise model with moderate error rates");
    // Create a noise model with moderate error rates
    let mut model = GeneralNoiseModel::new(0.1, 0.1, 0.1, 0.1, 0.1);

    // Set single-qubit error rates with uniform distribution
    let mut single_qubit_weights = HashMap::new();
    single_qubit_weights.insert("X".to_string(), 0.25);
    single_qubit_weights.insert("Y".to_string(), 0.25);
    single_qubit_weights.insert("Z".to_string(), 0.25);
    single_qubit_weights.insert("L".to_string(), 0.25);
    info!("Setting single-qubit Pauli model");
    model.set_p1_pauli_model(&single_qubit_weights);

    // Set two-qubit error rates with uniform distribution
    let mut two_qubit_weights = HashMap::new();
    two_qubit_weights.insert("XX".to_string(), 0.2);
    two_qubit_weights.insert("YY".to_string(), 0.2);
    two_qubit_weights.insert("ZZ".to_string(), 0.2);
    two_qubit_weights.insert("XL".to_string(), 0.2);
    two_qubit_weights.insert("LX".to_string(), 0.2);
    info!("Setting two-qubit Pauli model");
    model.set_p2_pauli_model(&two_qubit_weights);

    // Set emission ratios to ensure errors are introduced
    info!("Setting emission ratios");
    model.set_p1_emission_ratio(0.5);
    model.set_p2_emission_ratio(0.5);
    model.set_prep_leak_ratio(0.5);

    // Scale parameters before using the model
    info!("Scaling parameters");
    model.scale_parameters();

    // Reset the model to ensure clean state
    info!("Resetting model");
    model.reset().unwrap();

    Box::new(model)
}

fn apply_noise(model: &mut Box<dyn NoiseModel>, msg: &ByteMessage) -> ByteMessage {
    info!("Applying noise to message");
    match model.start(msg.clone()).unwrap() {
        pecos_engines::engines::EngineStage::NeedsProcessing(noisy_msg) => {
            info!("Processing noisy message");
            match model.continue_processing(noisy_msg).unwrap() {
                pecos_engines::engines::EngineStage::Complete(result) => result,
                pecos_engines::engines::EngineStage::NeedsProcessing(_) => {
                    panic!("Expected Complete stage")
                }
            }
        }
        pecos_engines::engines::EngineStage::Complete(_) => {
            panic!("Expected NeedsProcessing stage")
        }
    }
}

fn compare_messages(msg1: &ByteMessage, msg2: &ByteMessage) -> bool {
    let ops1 = msg1.parse_quantum_operations().unwrap_or_default();
    let ops2 = msg2.parse_quantum_operations().unwrap_or_default();
    ops1 == ops2
}

#[test]
fn test_prep_determinism() {
    let seed = 42;
    info!("Creating noise models with identical seeds");
    let mut model1 = create_noise_model();

    // Apply noise to model1
    reset_model_with_seed(&mut model1, seed).unwrap();

    // Create a message with multiple prep gates
    let mut builder = ByteMessage::quantum_operations_builder();
    for _ in 0..6 {
        builder.add_prep(&[0]);
    }
    let msg = builder.build();

    // Apply noise to the message
    let noisy1 = apply_noise(&mut model1, &msg);

    // Reset model1 with the same seed for deterministic behavior
    reset_model_with_seed(&mut model1, seed).unwrap();

    // Apply noise again to the message
    let noisy2 = apply_noise(&mut model1, &msg);

    // Now these should be identical
    info!("Comparing noisy1 and noisy2 - should be identical with same seed and model");
    assert!(
        compare_messages(&noisy1, &noisy2),
        "Messages should be identical with same seed and model"
    );

    // Now create a completely different model to verify we see different noise
    info!("Creating a model with a different seed");
    let mut model3 = create_noise_model();
    reset_model_with_seed(&mut model3, seed + 1).unwrap(); // different seed

    // Apply noise with different model
    let noisy3 = apply_noise(&mut model3, &msg);

    // These should be different
    info!("Comparing noisy1 and noisy3 - should be different with different seeds");
    assert!(
        !compare_messages(&noisy1, &noisy3),
        "Different seeds should produce different messages"
    );
}

#[test]
fn test_single_qubit_gate_determinism() {
    let seed = 42;
    info!("Creating noise model with seed");
    let mut model1 = create_noise_model();

    // Apply noise to model1
    reset_model_with_seed(&mut model1, seed).unwrap();

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
    reset_model_with_seed(&mut model1, seed).unwrap();

    // Apply noise again with the same model
    info!("Applying noise second time");
    let noisy2 = apply_noise(&mut model1, &msg);

    // Verify determinism
    info!("Comparing results - should be identical with same seed");
    assert!(
        compare_messages(&noisy1, &noisy2),
        "Results should be identical with same seed"
    );

    // Verify that we get some errors due to noise
    info!("Comparing original and noisy messages");
    assert!(
        !compare_messages(&msg, &noisy1),
        "Original message should be different from noisy message"
    );
}

#[test]
fn test_two_qubit_gate_determinism() {
    let seed = 42;
    info!("Creating noise models with identical seeds");
    let mut model1 = create_noise_model();

    // Apply noise to model1
    reset_model_with_seed(&mut model1, seed).unwrap();

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
    reset_model_with_seed(&mut model1, seed).unwrap();

    // Apply noise again to the message
    let noisy2 = apply_noise(&mut model1, &msg);

    // Now these should be identical
    info!("Comparing noisy1 and noisy2 - should be identical with same seed and model");
    assert!(
        compare_messages(&noisy1, &noisy2),
        "Messages should be identical with same seed and model"
    );

    // Verify that the message is actually being modified by the noise model
    info!("Verifying that noise is being applied");
    assert!(
        !compare_messages(&msg, &noisy1),
        "Original message should be different from noisy message"
    );
}

#[test]
fn test_measurement_determinism() {
    let seed = 42;
    let mut model1 = create_noise_model();
    let mut model2 = create_noise_model();

    reset_model_with_seed(&mut model1, seed).unwrap();
    reset_model_with_seed(&mut model2, seed).unwrap();

    // Create a message with measurements
    let mut builder = ByteMessage::quantum_operations_builder();
    builder.add_h(&[0]);
    builder.add_h(&[1]);
    builder.add_cx(&[0], &[1]);
    builder.add_measurements(&[0], &[0]);
    builder.add_measurements(&[1], &[1]);
    let msg = builder.build();

    // Apply noise multiple times
    let noisy1 = apply_noise(&mut model1, &msg);

    reset_model_with_seed(&mut model1, seed).unwrap();

    let noisy2 = apply_noise(&mut model2, &msg);

    // Verify determinism in the quantum operations
    assert!(compare_messages(&noisy1, &noisy2));
}

#[test]
fn test_different_seeds_produce_different_results() {
    let seed1 = 42;
    let seed2 = 43; // Different seed
    let mut model1 = create_noise_model();
    let mut model2 = create_noise_model();

    reset_model_with_seed(&mut model1, seed1).unwrap();
    reset_model_with_seed(&mut model2, seed2).unwrap();

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
        !compare_messages(&noisy1, &noisy2),
        "Different seeds should produce different noise patterns"
    );
}
