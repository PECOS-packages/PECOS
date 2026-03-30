//! Example of using `GeneralNoiseModelBuilder` with fluent API and the unified simulation API

use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::{GateType, sim_builder, sparse_stab};
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;
use std::collections::BTreeMap;

fn run_basic_noise_example(qasm: &str) {
    println!("Example 1: Basic noise configuration");
    let basic_noise = GeneralNoiseModel::builder()
        .with_seed(42)
        .with_p1_probability(0.001)
        .with_p2_probability(0.01)
        .with_meas_0_probability(0.002)
        .with_meas_1_probability(0.002);

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .seed(42)
        .noise(basic_noise)
        .run(1000)
        .unwrap();

    println!("Ran 1000 shots with basic noise");
    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    // Count unique states
    let mut state_counts = std::collections::BTreeMap::new();
    for val in values {
        *state_counts.entry(val).or_insert(0) += 1;
    }
    println!("State distribution: {state_counts:?}\n");
}

fn main() {
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

    // Example 1: Basic noise configuration
    run_basic_noise_example(qasm);

    // Example 2: Complex noise with Pauli models
    println!("Example 2: Complex noise with Pauli error models");

    // Define single-qubit Pauli error model
    let mut p1_pauli = BTreeMap::new();
    p1_pauli.insert("X".to_string(), 0.6); // 60% X errors
    p1_pauli.insert("Y".to_string(), 0.2); // 20% Y errors
    p1_pauli.insert("Z".to_string(), 0.2); // 20% Z errors

    // Define two-qubit Pauli error model
    let mut p2_pauli = BTreeMap::new();
    p2_pauli.insert("IX".to_string(), 0.25);
    p2_pauli.insert("XI".to_string(), 0.25);
    p2_pauli.insert("XX".to_string(), 0.25);
    p2_pauli.insert("YY".to_string(), 0.25);

    let complex_noise = GeneralNoiseModel::builder()
        .with_seed(123)
        .with_scale(1.5) // Scale all error rates by 1.5x
        .with_average_p1_probability(0.001)
        .with_p1_pauli_model(&p1_pauli)
        .with_average_p2_probability(0.01)
        .with_p2_pauli_model(&p2_pauli)
        .with_prep_probability(0.001)
        .with_leakage_scale(0.1)
        .with_emission_scale(0.8);

    let _results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .seed(123)
        .noise(complex_noise)
        .run(500)
        .unwrap();

    println!("Ran 500 shots with complex Pauli noise models\n");

    // Example 3: Noiseless gates
    println!("Example 3: Selective noiseless gates");

    let selective_noise = GeneralNoiseModel::builder()
        .with_seed(42)
        .with_p1_probability(0.1) // High single-qubit error
        .with_p2_probability(0.1) // High two-qubit error
        .with_noiseless_gate(pecos_core::prelude::GateType::H) // H gates have no noise
        .with_noiseless_gate(pecos_core::prelude::GateType::MZ); // Measurements have no noise

    let _results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .noise(selective_noise)
        .run(100)
        .unwrap();

    println!("Ran 100 shots with selective noiseless gates");
    println!("H and MEASURE gates are noiseless, CX gates have 10% error rate\n");

    // Example 4: Full configuration with all parameters
    println!("Example 4: Full noise configuration");

    let full_noise = GeneralNoiseModel::builder()
        .with_seed(456)
        .with_scale(1.2)
        .with_leakage_scale(0.2)
        .with_emission_scale(0.7)
        .with_prep_probability(0.0005)
        .with_p1_probability(0.001)
        .with_average_p1_probability(0.0008)
        .with_p2_probability(0.01)
        .with_average_p2_probability(0.008)
        .with_meas_0_probability(0.001)
        .with_meas_1_probability(0.003)
        .with_p_idle_coherent(false)
        .with_p_idle_linear_rate(0.0001)
        .with_noiseless_gate(GateType::H)
        .with_noiseless_gate(GateType::CX);

    // Use with full simulation configuration
    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .seed(456)
        .workers(2)
        .noise(full_noise)
        .quantum(sparse_stab().qubits(3))
        .run(50)
        .unwrap();

    println!("Ran 50 shots with full noise configuration");
    let shot_map = results.try_as_shot_map().unwrap();
    let binary_values = shot_map.try_bits_as_binary("c").unwrap();
    println!("Sample results (binary): {:?}", &binary_values[..5]);
}
