// Tests for the new run_qasm function

use pecos_qasm::prelude::*;

#[test]
fn test_run_qasm_simple() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        RNGseed(10);
        RNGindex(11);
        // RNGnum();
        measure q -> c;
    "#;

    // Simple usage - ideal simulation
    let results = run_qasm(
        qasm,
        100,
        PassThroughNoiseModelBuilder::new(),
        None,
        None,
        None,
    )
    .unwrap();
    assert_eq!(results.len(), 100);

    // Check Bell state results
    let shot_map = results.try_as_shot_map().unwrap();
    let values = shot_map.try_bits_as_u64("c").unwrap();

    for val in values {
        assert!(val == 0 || val == 3); // |00> or |11>
    }
}

#[test]
fn test_run_qasm_with_noise() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        x q[0];
        measure q[0] -> c[0];
    "#;

    let results = run_qasm(
        qasm,
        1000,
        DepolarizingNoiseModel::builder().with_uniform_probability(0.1),
        None,
        None,
        Some(42),
    )
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
fn test_run_qasm_with_engine() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    // Test with StateVector engine
    let results_sv = run_qasm(
        qasm,
        100,
        PassThroughNoiseModelBuilder::new(),
        Some(QuantumEngineType::StateVector),
        None,
        Some(42),
    )
    .unwrap();
    assert_eq!(results_sv.len(), 100);

    // Test with SparseStabilizer engine
    let results_stab = run_qasm(
        qasm,
        100,
        PassThroughNoiseModelBuilder::new(),
        Some(QuantumEngineType::SparseStabilizer),
        None,
        Some(42),
    )
    .unwrap();
    assert_eq!(results_stab.len(), 100);
}

#[test]
fn test_run_qasm_with_config_structs() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    // Test with config struct converted to enum
    let noise_config = DepolarizingNoiseModel::builder()
        .with_prep_probability(0.01)
        .with_meas_probability(0.01)
        .with_p1_probability(0.001)
        .with_p2_probability(0.1);

    let results = run_qasm(
        qasm,
        1000,
        noise_config,
        None,
        Some(4),  // workers
        Some(42), // seed
    )
    .unwrap();

    assert_eq!(results.len(), 1000);
}

#[test]
fn test_run_qasm_deterministic() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[1];
        creg c[1];
        h q[0];
        measure q[0] -> c[0];
    "#;

    // Run twice with same seed
    let results1 = run_qasm(
        qasm,
        100,
        PassThroughNoiseModelBuilder::new(),
        None,
        None,
        Some(123),
    )
    .unwrap();
    let results2 = run_qasm(
        qasm,
        100,
        PassThroughNoiseModelBuilder::new(),
        None,
        None,
        Some(123),
    )
    .unwrap();

    // Convert to comparable format
    let map1 = results1.try_as_shot_map().unwrap();
    let map2 = results2.try_as_shot_map().unwrap();

    let values1 = map1.try_bits_as_u64("c").unwrap();
    let values2 = map2.try_bits_as_u64("c").unwrap();

    // Same seed should give same results
    assert_eq!(values1, values2);
}
