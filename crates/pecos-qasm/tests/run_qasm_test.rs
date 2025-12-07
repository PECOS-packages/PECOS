// Tests for the new unified QASM API

use pecos_engines::noise::{DepolarizingNoiseModelBuilder, PassThroughNoiseModelBuilder};
use pecos_engines::{sim_builder, sparse_stabilizer, state_vector};
use pecos_programs::Qasm;
use pecos_qasm::qasm_engine;

#[test]
fn test_run_qasm_simple() {
    let qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    // Simple usage - ideal simulation
    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .noise(PassThroughNoiseModelBuilder::new())
        .run(100)
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

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .seed(42)
        .noise(DepolarizingNoiseModelBuilder::new().with_uniform_probability(0.1))
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
    let results_sv = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .seed(42)
        .noise(PassThroughNoiseModelBuilder::new())
        .quantum(state_vector().qubits(2))
        .run(100)
        .unwrap();
    assert_eq!(results_sv.len(), 100);

    // Test with SparseStabilizer engine
    let results_stab = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .seed(42)
        .noise(PassThroughNoiseModelBuilder::new())
        .quantum(sparse_stabilizer().qubits(2))
        .run(100)
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
    let noise_config = DepolarizingNoiseModelBuilder::new()
        .with_prep_probability(0.01)
        .with_meas_probability(0.01)
        .with_p1_probability(0.001)
        .with_p2_probability(0.1);

    let results = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .seed(42)
        .workers(4)
        .noise(noise_config)
        .run(1000)
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
    let results1 = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .seed(123)
        .noise(PassThroughNoiseModelBuilder::new())
        .run(100)
        .unwrap();
    let results2 = sim_builder()
        .classical(qasm_engine().program(Qasm::from_string(qasm)))
        .seed(123)
        .noise(PassThroughNoiseModelBuilder::new())
        .run(100)
        .unwrap();

    // Convert to comparable format
    let map1 = results1.try_as_shot_map().unwrap();
    let map2 = results2.try_as_shot_map().unwrap();

    let values1 = map1.try_bits_as_u64("c").unwrap();
    let values2 = map2.try_bits_as_u64("c").unwrap();

    // Same seed should give same results
    assert_eq!(values1, values2);
}
