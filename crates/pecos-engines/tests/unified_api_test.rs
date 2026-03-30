//! Tests for the unified simulation API
//!
//! These tests demonstrate the consistent API across different engine types.

// Note: These are compile-time tests to verify the API consistency.
// Actual execution tests would require fully implemented engines.

#[test]
fn test_quantum_engine_builders() {
    use pecos_engines::{sparse_stab, state_vector};

    // Test that quantum engine builders can be created and configured
    let _state_vec_builder = state_vector().qubits(4);
    let _sparse_stab_builder = sparse_stab().qubits(4);

    // Test that builders can be created without qubit count (will be set later)
    let _state_vec_no_qubits = state_vector();
    let _sparse_stab_no_qubits = sparse_stab();

    // Test chaining
    let _chained = sparse_stab().qubits(2);
}

#[test]
fn test_noise_conversions() {
    use pecos_engines::{
        BiasedDepolarizingNoise, DepolarizingNoise, PassThroughNoise,
        noise::{GeneralNoiseModelBuilder, NoiseModel},
    };

    // Test that all noise types can be converted
    // Test IntoNoiseModel trait
    use pecos_engines::noise::IntoNoiseModel;

    let _: Box<dyn NoiseModel> = PassThroughNoise.into_noise_model();
    let _: Box<dyn NoiseModel> = DepolarizingNoise { p: 0.01 }.into_noise_model();
    let _: Box<dyn NoiseModel> = BiasedDepolarizingNoise { p: 0.01 }.into_noise_model();
    let _: Box<dyn NoiseModel> = Box::new(GeneralNoiseModelBuilder::new().build());
}

#[test]
fn test_sim_config() {
    use pecos_engines::sim_builder::SimConfig;

    let config = SimConfig::default();
    assert_eq!(config.workers, 1);
    assert!(config.seed.is_none());
    assert!(!config.verbose);
}
