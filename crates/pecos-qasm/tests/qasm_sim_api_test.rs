// Tests for the new qasm_sim API

use pecos_qasm::prelude::*;
use std::collections::BTreeMap;

#[test]
fn test_simple_run() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let results = qasm_sim(qasm).run(100).unwrap();
    assert_eq!(results.len(), 100);

    // Check Bell state results
    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    for val in values {
        assert!(val == 0 || val == 3); // |00> or |11>
    }
}

#[test]
fn test_build_once_run_multiple() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
    "#;

    let sim = qasm_sim(qasm).seed(42).build().unwrap();

    // Run multiple times
    let results1 = sim.run(100).unwrap();
    let results2 = sim.run(1000).unwrap();
    let results3 = sim.run(10).unwrap();

    assert_eq!(results1.len(), 100);
    assert_eq!(results2.len(), 1000);
    assert_eq!(results3.len(), 10);
}

#[test]
fn test_with_depolarizing_noise() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
    "#;

    // Use builder for depolarizing noise
    let noise_builder = DepolarizingNoiseModel::builder().with_uniform_probability(0.1);

    let results = qasm_sim(qasm)
        .seed(42)
        .noise(noise_builder)
        .run(1000)
        .unwrap();

    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    // Count errors
    let errors = values.iter().filter(|&&v| v == 0).count();

    // With 10% noise, expect some errors
    assert!(errors > 50);
    assert!(errors < 200);
}

#[test]
fn test_custom_depolarizing_noise() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    // Use builder for custom depolarizing noise
    let noise_builder = DepolarizingNoiseModel::builder()
        .with_prep_probability(0.01)
        .with_meas_probability(0.01)
        .with_p1_probability(0.001)
        .with_p2_probability(0.1); // High two-qubit error

    let results = qasm_sim(qasm)
        .seed(42)
        .noise(noise_builder)
        .run(1000)
        .unwrap();

    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    // Count non-Bell states
    let mut counts = BTreeMap::new();
    for val in values {
        *counts.entry(val).or_insert(0) += 1;
    }

    // Should see some errors (01 and 10 states)
    assert!(counts.contains_key(&1) || counts.contains_key(&2));
}

#[test]
fn test_biased_depolarizing_noise() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
    "#;

    // Use builder for biased depolarizing noise
    let noise_builder = BiasedDepolarizingNoiseModel::builder().with_uniform_probability(0.2);

    let results = qasm_sim(qasm)
        .seed(42)
        .noise(noise_builder)
        .run(1000)
        .unwrap();

    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    // With biased depolarizing noise, we expect some errors
    let ones = values.iter().filter(|&&v| v == 1).count();
    let zeros = values.iter().filter(|&&v| v == 0).count();

    // Should see some error distribution
    assert!(ones > 0, "Should have some 1s");
    assert!(zeros > 0, "Should have some 0s");
}

#[test]
fn test_state_vector_engine() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        rz(0.5) q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    // StateVector can handle non-Clifford gates
    let results = qasm_sim(qasm)
        .seed(42)
        .quantum_engine(QuantumEngineType::StateVector)
        .run(100)
        .unwrap();

    assert_eq!(results.len(), 100);
}

#[test]
fn test_auto_workers() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        h q[1];
        h q[2];
        measure q -> c;
    "#;

    let results = qasm_sim(qasm).seed(42).auto_workers().run(1000).unwrap();

    assert_eq!(results.len(), 1000);
}

#[test]
fn test_deterministic_with_seed() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
    "#;

    // Run twice with same seed
    let sim = qasm_sim(qasm).seed(123).build().unwrap();

    let results1 = sim.run(100).unwrap();
    let results2 = sim.run(100).unwrap();

    // Convert to comparable format
    let map1 = results1.try_as_shot_map().unwrap();
    let map2 = results2.try_as_shot_map().unwrap();

    let values1 = map1.try_bits_as_u64("c").unwrap();
    let values2 = map2.try_bits_as_u64("c").unwrap();

    // Same seed should give same results
    assert_eq!(values1, values2);
}

#[test]
fn test_full_configuration() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let noise_builder = BiasedDepolarizingNoiseModel::builder().with_uniform_probability(0.01);

    let sim = qasm_sim(qasm)
        .seed(42)
        .workers(2)
        .quantum_engine(QuantumEngineType::SparseStabilizer)
        .noise(noise_builder)
        .build()
        .unwrap();

    // Run multiple times
    for shots in [10, 100, 1000] {
        let results = sim.run(shots).unwrap();
        assert_eq!(results.len(), shots);
    }
}

#[test]
fn test_passthrough_noise() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
    "#;

    let results = qasm_sim(qasm)
        .noise(PassThroughNoiseModel::builder())
        .run(100)
        .unwrap();

    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    // No noise, all should be 1
    assert!(values.iter().all(|&v| v == 1));
}

#[test]
fn test_general_noise() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        measure q[0] -> c[0];
    "#;

    // Use GeneralNoiseModelBuilder instead of old GeneralNoise
    let noise_builder = GeneralNoiseModel::builder()
        .with_seed(42)
        .with_p1_probability(0.001)
        .with_meas_0_probability(0.001)
        .with_meas_1_probability(0.001);

    let results = qasm_sim(qasm).noise(noise_builder).run(10).unwrap();

    assert_eq!(results.len(), 10);
}

#[test]
fn test_binary_string_format() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[4];
        creg c[4];
        h q[0];
        cx q[0], q[1];
        h q[2];
        cx q[2], q[3];
        measure q -> c;
    "#;

    // Test default format returns BigUint
    let sim_default = qasm_sim(qasm).seed(42).build().unwrap();
    let results_default = sim_default.run(10).unwrap();
    let map_default = results_default.try_as_shot_map().unwrap();

    // Verify we can get BigUint values
    let biguint_values = map_default.try_bits_as_biguint("c").unwrap();
    assert_eq!(biguint_values.len(), 10);

    // Test binary string format
    let sim_binary = qasm_sim(qasm)
        .seed(42)
        .with_binary_string_format()
        .build()
        .unwrap();

    let results_binary = sim_binary.run(10).unwrap();
    let map_binary = results_binary.try_as_shot_map().unwrap();

    // Should be able to get binary strings
    let binary_values = map_binary.try_bits_as_binary("c").unwrap();
    assert_eq!(binary_values.len(), 10);

    // Check format is correct (4 bits)
    for binary_str in &binary_values {
        assert_eq!(binary_str.len(), 4);
        // Should only contain 0s and 1s
        assert!(binary_str.chars().all(|c| c == '0' || c == '1'));
    }

    // Check expected Bell state patterns (0000, 0011, 1100, 1111)
    for binary_str in &binary_values {
        let valid_states = ["0000", "0011", "1100", "1111"];
        assert!(valid_states.contains(&binary_str.as_str()));
    }
}

#[test]
fn test_binary_string_format_large_register() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[10];
        creg c[10];
        // Create a known pattern
        x q[0];
        x q[2];
        x q[4];
        x q[6];
        x q[8];
        measure q -> c;
    "#;

    let results = qasm_sim(qasm).with_binary_string_format().run(5).unwrap();

    let map = results.try_as_shot_map().unwrap();
    let binary_values = map.try_bits_as_binary("c").unwrap();

    assert_eq!(binary_values.len(), 5);

    // All measurements should be the same: 0101010101
    for binary_str in &binary_values {
        assert_eq!(binary_str, "0101010101");
    }
}
