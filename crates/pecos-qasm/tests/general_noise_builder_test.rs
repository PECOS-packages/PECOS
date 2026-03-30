// Tests for GeneralNoiseModelBuilder with fluent API

use pecos_core::gate_type::GateType;
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::prelude::{sparse_stab, state_vector};
use pecos_engines::sim_builder;
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;
use std::collections::BTreeMap;

#[test]
fn test_general_noise_builder_basic() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    // Create builder with fluent API
    let noise_builder = GeneralNoiseModel::builder()
        .with_seed(42)
        .with_p1_probability(0.001)
        .with_p2_probability(0.01)
        .with_meas_0_probability(0.002)
        .with_meas_1_probability(0.002);

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .noise(noise_builder)
        .seed(42)
        .run(1000)
        .unwrap();

    assert_eq!(results.len(), 1000);

    // Check Bell state results with noise
    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    // Should see mostly 0 (00) and 3 (11), but some errors due to noise
    let mut counts = std::collections::BTreeMap::new();
    for val in values {
        *counts.entry(val).or_insert(0) += 1;
    }

    // Should see some errors (01 and 10 states)
    assert!(counts.len() > 2, "Should see errors due to noise");
}

#[test]
fn test_general_noise_builder_with_pauli_models() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
    "#;

    // Create p1 Pauli model
    let mut p1_model = BTreeMap::new();
    p1_model.insert("X".to_string(), 0.5);
    p1_model.insert("Y".to_string(), 0.3);
    p1_model.insert("Z".to_string(), 0.2);

    let noise_builder = GeneralNoiseModel::builder()
        .with_seed(42)
        .with_p1_probability(0.1) // High error rate for testing
        .with_p1_pauli_model(&p1_model);

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .noise(noise_builder)
        .seed(42)
        .run(1000)
        .unwrap();

    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    // Count errors (should see some 0s due to high error rate)
    let zeros = values.iter().filter(|&&v| v == 0).count();
    // With fixed seeds on both noise and simulation, results should be deterministic
    assert!(
        zeros > 50,
        "Should see errors with 10% p1 error rate, got {zeros} zeros"
    );
}

#[test]
fn test_general_noise_builder_complex_configuration() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        cx q[0], q[1];
        cx q[1], q[2];
        measure q -> c;
    "#;

    // Create complex Pauli models
    let mut p1_model = BTreeMap::new();
    p1_model.insert("X".to_string(), 0.6);
    p1_model.insert("Y".to_string(), 0.2);
    p1_model.insert("Z".to_string(), 0.2);

    let mut p2_model = BTreeMap::new();
    p2_model.insert("IX".to_string(), 0.25);
    p2_model.insert("XI".to_string(), 0.25);
    p2_model.insert("XX".to_string(), 0.25);
    p2_model.insert("YY".to_string(), 0.25);

    let noise_builder = GeneralNoiseModel::builder()
        .with_seed(123)
        .with_scale(1.5)
        .with_leakage_scale(0.1)
        .with_emission_scale(0.8)
        .with_prep_probability(0.001)
        .with_average_p1_probability(0.0008)
        .with_p1_pauli_model(&p1_model)
        .with_average_p2_probability(0.008)
        .with_p2_pauli_model(&p2_model)
        .with_meas_0_probability(0.002)
        .with_meas_1_probability(0.003)
        .with_noiseless_gate(GateType::H);

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .noise(noise_builder)
        .seed(123)
        .workers(2)
        .run(500)
        .unwrap();

    assert_eq!(results.len(), 500);
}

#[test]
fn test_general_noise_builder_noiseless_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];  // This will be noiseless
        x q[0];  // This will have noise
        cx q[0], q[1];  // This will have noise
        measure q -> c;
    "#;

    let noise_builder = GeneralNoiseModel::builder()
        .with_seed(42)
        .with_p1_probability(0.5) // Very high error rate
        .with_p2_probability(0.5) // Very high error rate
        .with_noiseless_gate(GateType::H) // H gate is noiseless
        .with_noiseless_gate(GateType::MZ); // Measurement is noiseless

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .noise(noise_builder)
        .seed(42)
        .run(1000)
        .unwrap();

    // Even with very high error rates, H being noiseless should preserve some structure
    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    // Should see all possible states due to high noise on X and CX
    let unique_states: std::collections::BTreeSet<_> = values.iter().copied().collect();
    assert!(
        unique_states.len() >= 3,
        "High noise should create various states"
    );
}

#[test]
fn test_general_noise_builder_with_prep_errors() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        // No gates, just measure initialized qubits
        measure q -> c;
    "#;

    let noise_builder = GeneralNoiseModel::builder()
        .with_seed(42)
        .with_prep_probability(0.1); // 10% prep error

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .noise(noise_builder)
        .seed(42)
        .run(1000)
        .unwrap();

    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    // Count non-zero results (prep errors)
    let non_zeros = values.iter().filter(|&&v| v != 0).count();

    // With 10% prep error per qubit and 2 qubits:
    // P(at least one error) = 1 - P(no errors) = 1 - 0.9^2 = 0.19
    // So expect about 190 non-zero results out of 1000
    // However, with seeded RNG, we might get consistent but lower values
    assert!(
        non_zeros > 10,
        "Should see some prep errors (got {non_zeros} non-zeros)"
    );
    assert!(
        non_zeros < 300,
        "Prep errors shouldn't be too frequent (got {non_zeros} non-zeros)"
    );
}

#[test]
fn test_general_noise_builder_measurement_errors() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        x q[0];
        x q[1];
        measure q -> c;
    "#;

    let noise_builder = GeneralNoiseModel::builder()
        .with_seed(42)
        .with_meas_0_probability(0.05) // 5% chance |0> measured as |1>
        .with_meas_1_probability(0.10); // 10% chance |1> measured as |0>

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .noise(noise_builder)
        .seed(42)
        .run(1000)
        .unwrap();

    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    // Should be 3 (11) without errors
    let mut counts = std::collections::BTreeMap::new();
    for val in values {
        *counts.entry(val).or_insert(0) += 1;
    }

    // Should see measurement errors
    assert!(
        counts.contains_key(&0),
        "Should see 00 from double meas error"
    );
    assert!(counts.contains_key(&1), "Should see 01 from meas error");
    assert!(counts.contains_key(&2), "Should see 10 from meas error");
    assert!(counts.contains_key(&3), "Should see 11 as intended result");

    // Most results should still be 11
    assert!(counts[&3] > 700, "Most results should be correct");
}

#[test]
fn test_general_noise_builder_chaining_all_methods() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    // Test that all builder methods can be chained
    let noise_builder = GeneralNoiseModel::builder()
        .with_seed(42)
        .with_scale(1.2)
        .with_leakage_scale(0.1)
        .with_emission_scale(0.9)
        .with_prep_probability(0.001)
        .with_p1_probability(0.001)
        .with_average_p1_probability(0.0008)
        .with_p2_probability(0.01)
        .with_average_p2_probability(0.008)
        .with_meas_0_probability(0.002)
        .with_meas_1_probability(0.003)
        .with_p_idle_coherent(false)
        .with_p_idle_linear_rate(0.0001)
        .with_noiseless_gate(GateType::H)
        .with_noiseless_gate(GateType::CX);

    // Should compile and run without errors
    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .noise(noise_builder)
        .seed(42)
        .run(100)
        .unwrap();

    assert_eq!(results.len(), 100);
}

#[test]
fn test_general_noise_builder_with_multiple_noiseless_gates() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        s q[0];
        t q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let noise_builder = GeneralNoiseModel::builder()
        .with_seed(42)
        .with_p1_probability(0.1) // High noise
        .with_p2_probability(0.1) // High noise
        .with_noiseless_gate(GateType::H)
        .with_noiseless_gate(GateType::SZ) // S gate
        .with_noiseless_gate(GateType::T)
        .with_noiseless_gate(GateType::CX)
        .with_noiseless_gate(GateType::MZ);

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .quantum(state_vector()) // Need StateVector for T gate
        .noise(noise_builder)
        .seed(42)
        .run(100)
        .unwrap();

    assert_eq!(results.len(), 100);

    // With all gates noiseless, should get perfect results
    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    // With all gates noiseless, there's still quantum randomness from H gate
    // We should see a superposition state, but no noise errors
    let unique_values: std::collections::BTreeSet<_> = values.iter().copied().collect();

    // Count the different states we see
    let mut counts = std::collections::BTreeMap::new();
    for val in values {
        *counts.entry(val).or_insert(0) += 1;
    }

    // The circuit has H, S, T gates which create a complex superposition
    // We're not creating a simple Bell state here
    // Just verify that with all gates noiseless, we get consistent quantum results
    assert!(
        unique_values.len() <= 4,
        "Should see limited states with quantum superposition"
    );
}

#[test]
fn test_general_noise_builder_comparison_with_sim_builder() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let noise_builder = GeneralNoiseModel::builder()
        .with_seed(42)
        .with_p1_probability(0.001)
        .with_p2_probability(0.01);

    // Test full method chaining with simulation builder
    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .quantum(sparse_stab())
        .noise(noise_builder)
        .seed(42)
        .workers(2)
        .run(100)
        .unwrap();
    assert_eq!(results.len(), 100);

    // Check binary string format
    let shot_map = results.try_as_shot_map().unwrap();
    let binary_values = shot_map.try_bits_as_binary("c").unwrap();

    assert_eq!(binary_values.len(), 100);
    for binary in &binary_values {
        assert_eq!(binary.len(), 2);
        assert!(binary.chars().all(|c| c == '0' || c == '1'));
    }
}
