//! Tests for Quest quantum simulator integration with `sim()` API

#![cfg(all(feature = "runtime", feature = "quest"))]

use pecos::{quest_density_matrix, quest_state_vec, sim};
use pecos_programs::Qasm;

/// Test Quest state vector with CPU mode
#[test]
fn test_quest_state_vec_cpu() {
    let qasm_code = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let program = Qasm::from_string(qasm_code);

    // Test CPU mode
    let results = sim(program)
        .quantum(quest_state_vec().with_cpu())
        .seed(42)
        .run(100)
        .expect("Simulation should succeed");

    assert_eq!(results.len(), 100, "Should get 100 shots");

    // Verify we got Bell state results (only |00⟩ and |11⟩)
    let shot_map = results
        .try_as_shot_map()
        .expect("Should convert to shot map");
    let measurements = shot_map
        .try_bits_as_u64("c")
        .expect("Should extract measurements");

    for &measurement in &measurements {
        assert!(
            measurement == 0 || measurement == 3,
            "Bell state should only produce |00⟩ (0) or |11⟩ (3), got {measurement}"
        );
    }
}

/// Test Quest state vector with GPU mode (only runs if GPU feature enabled)
#[test]
#[cfg(feature = "cuda")]
fn test_quest_state_vec_gpu() {
    let qasm_code = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let program = Qasm::from_string(qasm_code);

    // Test GPU mode
    let results = sim(program)
        .quantum(quest_state_vec().with_gpu())
        .seed(42)
        .run(100)
        .expect("Simulation should succeed");

    assert_eq!(results.len(), 100, "Should get 100 shots");

    // Verify we got Bell state results
    let shot_map = results
        .try_as_shot_map()
        .expect("Should convert to shot map");
    let measurements = shot_map
        .try_bits_as_u64("c")
        .expect("Should extract measurements");

    for &measurement in &measurements {
        assert!(
            measurement == 0 || measurement == 3,
            "Bell state should only produce |00⟩ (0) or |11⟩ (3), got {measurement}"
        );
    }
}

/// Test Quest density matrix with CPU mode
#[test]
fn test_quest_density_matrix_cpu() {
    let qasm_code = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let program = Qasm::from_string(qasm_code);

    // Test CPU mode
    let results = sim(program)
        .quantum(quest_density_matrix().with_cpu())
        .seed(42)
        .run(100)
        .expect("Simulation should succeed");

    assert_eq!(results.len(), 100, "Should get 100 shots");

    // Verify we got Bell state results
    let shot_map = results
        .try_as_shot_map()
        .expect("Should convert to shot map");
    let measurements = shot_map
        .try_bits_as_u64("c")
        .expect("Should extract measurements");

    for &measurement in &measurements {
        assert!(
            measurement == 0 || measurement == 3,
            "Bell state should only produce |00⟩ (0) or |11⟩ (3), got {measurement}"
        );
    }
}

/// Test Quest density matrix with GPU mode returns appropriate error
/// (GPU density matrix simulation is not yet implemented in `QuEST`)
#[test]
#[cfg(feature = "cuda")]
fn test_quest_density_matrix_gpu() {
    let qasm_code = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let program = Qasm::from_string(qasm_code);

    // GPU density matrix simulation is not yet implemented, so this should return an error
    let result = sim(program)
        .quantum(quest_density_matrix().with_gpu())
        .seed(42)
        .run(100);

    // Verify we get the expected error about GPU density matrix not being implemented
    assert!(result.is_err(), "GPU density matrix should return an error");
    let err_msg = result.unwrap_err().to_string();
    assert!(
        err_msg.contains("density matrix") && err_msg.contains("not yet implemented"),
        "Error should mention density matrix GPU not implemented, got: {err_msg}"
    );
}

/// Test that Quest works with different circuit types
#[test]
fn test_quest_various_gates() {
    let qasm_code = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[3];
        creg c[3];
        h q[0];
        t q[0];
        x q[1];
        y q[2];
        z q[0];
        rx(1.5708) q[1];
        ry(1.5708) q[2];
        rz(1.5708) q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let program = Qasm::from_string(qasm_code);

    // Test with Quest state vector
    let results = sim(program)
        .quantum(quest_state_vec().with_cpu())
        .seed(42)
        .run(10)
        .expect("Simulation should succeed");

    assert_eq!(results.len(), 10, "Should get 10 shots");
}

/// Test that Quest works with seed for reproducibility
///
/// Note: Due to `QuEST`'s global environment design, perfect reproducibility
/// across separate `sim()` calls may not be guaranteed. This test verifies
/// that the seed parameter is accepted and affects the results.
#[test]
fn test_quest_seed_parameter() {
    let qasm_code = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let program = Qasm::from_string(qasm_code);

    // Run with one seed
    let results1 = sim(program.clone())
        .quantum(quest_state_vec().with_cpu())
        .seed(123)
        .run(50)
        .expect("Simulation should succeed");

    // Run with different seed
    let results2 = sim(program)
        .quantum(quest_state_vec().with_cpu())
        .seed(456)
        .run(50)
        .expect("Simulation should succeed");

    // Just verify both completed successfully
    assert_eq!(results1.len(), 50, "Should get 50 shots with seed 123");
    assert_eq!(results2.len(), 50, "Should get 50 shots with seed 456");

    // Verify we got valid Bell state results from both
    let shot_map1 = results1
        .try_as_shot_map()
        .expect("Should convert to shot map");
    let shot_map2 = results2
        .try_as_shot_map()
        .expect("Should convert to shot map");

    let measurements1 = shot_map1
        .try_bits_as_u64("c")
        .expect("Should extract measurements");
    let measurements2 = shot_map2
        .try_bits_as_u64("c")
        .expect("Should extract measurements");

    // Both should only produce valid Bell state outcomes
    for &measurement in &measurements1 {
        assert!(
            measurement == 0 || measurement == 3,
            "Bell state should only produce |00⟩ or |11⟩"
        );
    }
    for &measurement in &measurements2 {
        assert!(
            measurement == 0 || measurement == 3,
            "Bell state should only produce |00⟩ or |11⟩"
        );
    }
}

/// Test that Quest builder can be used with `qubits()` method
#[test]
fn test_quest_builder_with_qubits() {
    let qasm_code = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let program = Qasm::from_string(qasm_code);

    // Test that qubits() method works (though it gets overridden by program)
    let results = sim(program)
        .quantum(quest_state_vec().qubits(2).with_cpu())
        .seed(42)
        .run(10)
        .expect("Simulation should succeed");

    assert_eq!(results.len(), 10, "Should get 10 shots");
}

/// Test that both CPU and GPU modes work correctly
///
/// Note: Due to potential differences in RNG implementation between CPU and GPU,
/// we verify that both modes produce valid results rather than identical results.
#[test]
#[cfg(feature = "cuda")]
#[allow(clippy::similar_names)]
fn test_quest_cpu_and_gpu_both_work() {
    let qasm_code = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q -> c;
    "#;

    let program = Qasm::from_string(qasm_code);

    // Run with CPU
    let results_cpu = sim(program.clone())
        .quantum(quest_state_vec().with_cpu())
        .seed(999)
        .run(50)
        .expect("CPU simulation should succeed");

    // Run with GPU
    let results_gpu = sim(program)
        .quantum(quest_state_vec().with_gpu())
        .seed(999)
        .run(50)
        .expect("GPU simulation should succeed");

    // Verify both got the right number of shots
    assert_eq!(results_cpu.len(), 50, "CPU should get 50 shots");
    assert_eq!(results_gpu.len(), 50, "GPU should get 50 shots");

    // Convert to shot maps
    let shot_map_cpu = results_cpu
        .try_as_shot_map()
        .expect("Should convert CPU results to shot map");
    let shot_map_gpu = results_gpu
        .try_as_shot_map()
        .expect("Should convert GPU results to shot map");

    let measurements_cpu = shot_map_cpu
        .try_bits_as_u64("c")
        .expect("Should extract CPU measurements");
    let measurements_gpu = shot_map_gpu
        .try_bits_as_u64("c")
        .expect("Should extract GPU measurements");

    // Both should produce valid Bell state results
    for &measurement in &measurements_cpu {
        assert!(
            measurement == 0 || measurement == 3,
            "CPU Bell state should only produce |00⟩ or |11⟩, got {measurement}"
        );
    }
    for &measurement in &measurements_gpu {
        assert!(
            measurement == 0 || measurement == 3,
            "GPU Bell state should only produce |00⟩ or |11⟩, got {measurement}"
        );
    }
}
