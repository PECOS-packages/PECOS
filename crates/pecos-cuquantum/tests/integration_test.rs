//! Integration tests for pecos-cuquantum
//!
//! These tests only run when:
//! 1. The `integration-tests` feature is enabled
//! 2. cuQuantum is available at build time
//!
//! Run with: `cargo test -p pecos-cuquantum --features integration-tests`
//!
//! They verify actual GPU simulation functionality.

#![cfg(feature = "integration-tests")]

use pecos_cuquantum::{
    CuDensityMat, CuQuantumError, CuStateVec, CuTensorNet, is_cuquantum_available,
};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};

/// Skip a test if cuQuantum is not available at runtime
macro_rules! skip_if_no_cuquantum {
    () => {
        if !is_cuquantum_available() {
            eprintln!("Skipping test: cuQuantum not available");
            return;
        }
    };
}

// =============================================================================
// cuStateVec tests
// =============================================================================

#[test]
fn test_custatevec_creation() {
    skip_if_no_cuquantum!();

    let result = CuStateVec::new(4);
    assert!(result.is_ok(), "Failed to create CuStateVec");

    let sim = result.unwrap();
    assert_eq!(sim.num_qubits(), 4);
}

#[test]
fn test_custatevec_bell_state() {
    skip_if_no_cuquantum!();

    let mut sim = CuStateVec::new(2).expect("Failed to create CuStateVec");

    // Create Bell state: |00> + |11>
    sim.h(&[pecos_cuquantum::QubitId(0)]);
    sim.cx(&[(pecos_cuquantum::QubitId(0), pecos_cuquantum::QubitId(1))]);

    // Measure - results should be correlated
    let results = sim.mz(&[pecos_cuquantum::QubitId(0), pecos_cuquantum::QubitId(1)]);
    assert_eq!(results.len(), 2);
    // In a Bell state, both qubits should have the same measurement outcome
    assert_eq!(results[0].outcome, results[1].outcome);
}

#[test]
fn test_custatevec_reset() {
    skip_if_no_cuquantum!();

    let mut sim = CuStateVec::new(2).expect("Failed to create CuStateVec");

    // Apply some gates
    sim.h(&[pecos_cuquantum::QubitId(0)]);
    sim.x(&[pecos_cuquantum::QubitId(1)]);

    // Reset to |00>
    sim.reset();

    // After reset, measuring should give deterministic 0
    let results = sim.mz(&[pecos_cuquantum::QubitId(0), pecos_cuquantum::QubitId(1)]);
    assert!(!results[0].outcome);
    assert!(!results[1].outcome);
}

#[test]
fn test_custatevec_clone() {
    skip_if_no_cuquantum!();

    use pecos_cuquantum::QubitId;

    // Create simulator and put it in a specific state
    let mut original = CuStateVec::with_seed(3, 42).expect("Failed to create CuStateVec");

    // Put in |100> state (X on qubit 0)
    original.x(&[QubitId(0)]);

    // Clone the state
    let mut cloned = original.clone();

    // Verify they have the same number of qubits
    assert_eq!(original.num_qubits(), cloned.num_qubits());

    // Verify the cloned state has the same |100> state
    // Measure qubit 0 should give 1
    let original_result = original.mz(&[QubitId(0)]);
    let cloned_result = cloned.mz(&[QubitId(0)]);

    assert!(original_result[0].outcome, "Original qubit 0 should be 1");
    assert!(cloned_result[0].outcome, "Cloned qubit 0 should be 1");

    // Verify they are independent - modify original, check clone unchanged
    // Reset original, clone should still be in post-measurement state
    original.reset();

    // After reset, original's qubit 0 is 0
    let original_after_reset = original.mz(&[QubitId(0)]);
    assert!(
        !original_after_reset[0].outcome,
        "Original after reset should be 0"
    );

    // Note: After the first measurement, both states collapsed,
    // so we can't easily verify independence post-measurement.
    // The main verification is that clone succeeds and has correct initial state.
}

#[test]
fn test_custatevec_try_clone() {
    skip_if_no_cuquantum!();

    use pecos_cuquantum::{QubitId, TryClone};

    // Create simulator and put it in a specific state
    let mut original = CuStateVec::with_seed(3, 42).expect("Failed to create CuStateVec");

    // Put in |100> state (X on qubit 0)
    original.x(&[QubitId(0)]);

    // TryClone the state - should return Result
    let cloned = original.try_clone();
    assert!(cloned.is_ok(), "try_clone should succeed");

    let mut cloned = cloned.unwrap();

    // Verify they have the same number of qubits
    assert_eq!(original.num_qubits(), cloned.num_qubits());

    // Verify the cloned state has the same |100> state
    let original_result = original.mz(&[QubitId(0)]);
    let cloned_result = cloned.mz(&[QubitId(0)]);

    assert!(original_result[0].outcome, "Original qubit 0 should be 1");
    assert!(cloned_result[0].outcome, "Cloned qubit 0 should be 1");
}

// =============================================================================
// cuTensorNet tests
// =============================================================================

#[test]
fn test_cutensornet_creation() {
    skip_if_no_cuquantum!();

    let result = CuTensorNet::new();
    assert!(result.is_ok(), "Failed to create CuTensorNet");
}

#[test]
fn test_cutensornet_version() {
    skip_if_no_cuquantum!();

    let version = CuTensorNet::version();
    // Version should be a reasonable number (e.g., 20000 for v2.0.0)
    assert!(version > 0, "Invalid version: {}", version);
}

// =============================================================================
// cuDensityMat tests
// =============================================================================

#[test]
fn test_cudensitymat_creation() {
    skip_if_no_cuquantum!();

    let result = CuDensityMat::new(4);
    assert!(result.is_ok(), "Failed to create CuDensityMat");

    let sim = result.unwrap();
    assert_eq!(sim.num_qubits(), 4);
}

#[test]
fn test_cudensitymat_version() {
    skip_if_no_cuquantum!();

    let version = CuDensityMat::version();
    // Version should be a reasonable number
    assert!(version > 0, "Invalid version: {}", version);
}

#[test]
fn test_cudensitymat_invalid_args() {
    skip_if_no_cuquantum!();

    // Zero qubits should fail
    let result = CuDensityMat::new(0);
    assert!(result.is_err());
    match result {
        Err(CuQuantumError::InvalidArgument(msg)) => {
            assert!(msg.contains("at least 1"));
        }
        _ => panic!("Expected InvalidArgument error"),
    }
}

// =============================================================================
// Additional CuStateVec tests
// =============================================================================

#[test]
fn test_custatevec_rotation_gates() {
    skip_if_no_cuquantum!();

    use pecos_cuquantum::{ArbitraryRotationGateable, QubitId};
    use std::f64::consts::PI;

    let mut sim = CuStateVec::new(2).expect("Failed to create CuStateVec");

    // RX(pi) should flip |0> to |1>
    sim.rx(PI.into(), &[QubitId(0)]);
    let result = sim.mz(&[QubitId(0)]);
    assert!(result[0].outcome, "RX(pi)|0> should give |1>");

    sim.reset();

    // RZ should not change measurement outcome (only phase)
    sim.rz((PI / 2.0).into(), &[QubitId(0)]);
    let result = sim.mz(&[QubitId(0)]);
    assert!(!result[0].outcome, "RZ|0> should still be |0>");

    sim.reset();

    // RY(pi) should flip |0> to |1>
    sim.ry(PI.into(), &[QubitId(0)]);
    let result = sim.mz(&[QubitId(0)]);
    assert!(result[0].outcome, "RY(pi)|0> should give |1>");
}

#[test]
fn test_custatevec_t_gate() {
    skip_if_no_cuquantum!();

    use pecos_cuquantum::{ArbitraryRotationGateable, QubitId};

    let mut sim = CuStateVec::new(1).expect("Failed to create CuStateVec");

    // T gate on |0> should not change measurement outcome
    sim.t(&[QubitId(0)]);
    let result = sim.mz(&[QubitId(0)]);
    assert!(!result[0].outcome, "T|0> should still be |0>");

    sim.reset();

    // T^4 = Z, T^8 = I
    // Apply T 8 times to |+> state, should return to |+>
    sim.h(&[QubitId(0)]); // |+>
    for _ in 0..8 {
        sim.t(&[QubitId(0)]);
    }
    sim.h(&[QubitId(0)]); // H|+> = |0>
    let result = sim.mz(&[QubitId(0)]);
    assert!(!result[0].outcome, "T^8 H|0> should return to |0>");
}

#[test]
fn test_custatevec_sampling() {
    skip_if_no_cuquantum!();

    use pecos_cuquantum::QubitId;

    let mut sim = CuStateVec::with_seed(2, 12345).expect("Failed to create CuStateVec");

    // Create Bell state
    sim.h(&[QubitId(0)]);
    sim.cx(&[(QubitId(0), QubitId(1))]);

    // Sample multiple times
    let samples = sim.sample(100);
    assert_eq!(samples.len(), 100);

    // All samples should be either 00 (0) or 11 (3)
    for sample in &samples {
        assert!(
            *sample == 0 || *sample == 3,
            "Bell state sample should be 00 or 11, got {}",
            sample
        );
    }

    // Should have both outcomes (with high probability)
    let zeros = samples.iter().filter(|&&s| s == 0).count();
    let threes = samples.iter().filter(|&&s| s == 3).count();
    assert!(zeros > 10, "Should have some |00> outcomes, got {}", zeros);
    assert!(
        threes > 10,
        "Should have some |11> outcomes, got {}",
        threes
    );
}

#[test]
fn test_custatevec_ghz_state() {
    skip_if_no_cuquantum!();

    use pecos_cuquantum::QubitId;

    let mut sim = CuStateVec::with_seed(4, 42).expect("Failed to create CuStateVec");

    // Create 4-qubit GHZ state: |0000> + |1111>
    sim.h(&[QubitId(0)]);
    sim.cx(&[(QubitId(0), QubitId(1))]);
    sim.cx(&[(QubitId(1), QubitId(2))]);
    sim.cx(&[(QubitId(2), QubitId(3))]);

    // Sample and verify all qubits are correlated
    let samples = sim.sample(50);
    for sample in &samples {
        // All bits should be the same (either 0000=0 or 1111=15)
        assert!(
            *sample == 0 || *sample == 15,
            "GHZ state sample should be 0000 or 1111, got {}",
            sample
        );
    }
}

// =============================================================================
// Two-qubit rotation gate tests
// =============================================================================

#[test]
fn test_custatevec_rzz_gate() {
    skip_if_no_cuquantum!();

    use pecos_cuquantum::QubitId;
    use std::f64::consts::PI;

    let mut sim = CuStateVec::with_seed(2, 42).expect("Failed to create CuStateVec");

    // RZZ on |00> should not change measurement outcomes (only adds global phase)
    sim.rzz((PI / 4.0).into(), &[(QubitId(0), QubitId(1))]);
    let result = sim.mz(&[QubitId(0), QubitId(1)]);
    assert!(
        !result[0].outcome,
        "RZZ on |00> should still give |0> for qubit 0"
    );
    assert!(
        !result[1].outcome,
        "RZZ on |00> should still give |0> for qubit 1"
    );

    sim.reset();

    // Create Bell state and apply RZZ
    // |00> + |11> -> e^(-i*theta/2)|00> + e^(-i*theta/2)|11> (same phase)
    sim.h(&[QubitId(0)]);
    sim.cx(&[(QubitId(0), QubitId(1))]);
    sim.rzz(PI.into(), &[(QubitId(0), QubitId(1))]); // RZZ(pi) adds -i to both |00> and |11>

    // Measure - should still be perfectly correlated
    let result = sim.mz(&[QubitId(0), QubitId(1)]);
    assert_eq!(
        result[0].outcome, result[1].outcome,
        "Bell state correlation preserved after RZZ"
    );
}

#[test]
fn test_custatevec_rxx_gate() {
    skip_if_no_cuquantum!();

    use pecos_cuquantum::QubitId;
    use std::f64::consts::PI;

    let mut sim = CuStateVec::with_seed(2, 42).expect("Failed to create CuStateVec");

    // RXX(pi) on |00> should give |11> (up to global phase)
    sim.rxx(PI.into(), &[(QubitId(0), QubitId(1))]);
    let result = sim.mz(&[QubitId(0), QubitId(1)]);
    assert!(
        result[0].outcome,
        "RXX(pi) on |00> should give |1> for qubit 0"
    );
    assert!(
        result[1].outcome,
        "RXX(pi) on |00> should give |1> for qubit 1"
    );

    sim.reset();

    // RXX(pi/2) on |00> creates superposition
    sim.rxx((PI / 2.0).into(), &[(QubitId(0), QubitId(1))]);

    // Sample many times - should get both correlated outcomes
    let samples = sim.sample(100);
    let zeros = samples.iter().filter(|&&s| s == 0).count();
    let threes = samples.iter().filter(|&&s| s == 3).count();
    assert!(zeros > 20, "Should have some |00> outcomes, got {}", zeros);
    assert!(
        threes > 20,
        "Should have some |11> outcomes, got {}",
        threes
    );
}

#[test]
fn test_custatevec_ryy_gate() {
    skip_if_no_cuquantum!();

    use pecos_cuquantum::QubitId;
    use std::f64::consts::PI;

    let mut sim = CuStateVec::with_seed(2, 42).expect("Failed to create CuStateVec");

    // RYY(pi) on |00> should give -|11> (up to global phase, measurement gives |11>)
    sim.ryy(PI.into(), &[(QubitId(0), QubitId(1))]);
    let result = sim.mz(&[QubitId(0), QubitId(1)]);
    assert!(
        result[0].outcome,
        "RYY(pi) on |00> should give |1> for qubit 0"
    );
    assert!(
        result[1].outcome,
        "RYY(pi) on |00> should give |1> for qubit 1"
    );

    sim.reset();

    // RYY(pi/2) on |00> creates superposition
    sim.ryy((PI / 2.0).into(), &[(QubitId(0), QubitId(1))]);

    // Sample many times - should get correlated outcomes
    let samples = sim.sample(100);
    let zeros = samples.iter().filter(|&&s| s == 0).count();
    let threes = samples.iter().filter(|&&s| s == 3).count();
    assert!(zeros > 20, "Should have some |00> outcomes, got {}", zeros);
    assert!(
        threes > 20,
        "Should have some |11> outcomes, got {}",
        threes
    );
}

#[test]
fn test_custatevec_combined_rotations() {
    skip_if_no_cuquantum!();

    use pecos_cuquantum::QubitId;
    use std::f64::consts::PI;

    // Test that RXX(2*theta) = RYY(2*theta) = RZZ(2*theta) for specific angles
    // gives equivalent behavior when combined with appropriate single-qubit gates

    let mut sim = CuStateVec::with_seed(2, 123).expect("Failed to create CuStateVec");

    // Apply sequence: H-RZZ(pi/2)-H on both qubits is equivalent to RXX(pi/2)
    sim.h(&[QubitId(0), QubitId(1)]);
    sim.rzz((PI / 2.0).into(), &[(QubitId(0), QubitId(1))]);
    sim.h(&[QubitId(0), QubitId(1)]);

    // Sample and check correlation
    let samples = sim.sample(100);
    for sample in &samples {
        // Should only get correlated outcomes (00 or 11)
        assert!(
            *sample == 0 || *sample == 3,
            "H-RZZ(pi/2)-H should produce correlated outcomes, got {}",
            sample
        );
    }
}

// =============================================================================
// Error handling tests
// =============================================================================

#[test]
fn test_custatevec_zero_qubits_error() {
    skip_if_no_cuquantum!();

    // Zero qubits should fail
    let result = CuStateVec::new(0);
    assert!(
        result.is_err(),
        "Creating CuStateVec with 0 qubits should fail"
    );
    match result {
        Err(CuQuantumError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("at least 1"),
                "Error should mention qubit requirement"
            );
        }
        _ => panic!("Expected InvalidArgument error for 0 qubits"),
    }
}

#[test]
fn test_custatevec_too_many_qubits_error() {
    skip_if_no_cuquantum!();

    // More than 30 qubits should fail (would require > 16 GB GPU memory)
    let result = CuStateVec::new(31);
    assert!(
        result.is_err(),
        "Creating CuStateVec with 31 qubits should fail"
    );
    match result {
        Err(CuQuantumError::InvalidArgument(msg)) => {
            assert!(
                msg.contains("memory") || msg.contains("30"),
                "Error should mention memory limit"
            );
        }
        _ => panic!("Expected InvalidArgument error for too many qubits"),
    }
}

#[test]
fn test_cudensitymat_too_many_qubits() {
    skip_if_no_cuquantum!();

    // Density matrix with too many qubits should fail (O(4^n) memory)
    // Even 20 qubits would need 4^20 * 16 bytes = 17.6 TB
    let result = CuDensityMat::new(20);
    assert!(
        result.is_err(),
        "Creating CuDensityMat with 20 qubits should fail"
    );
}

#[test]
fn test_custatevec_seed_reproducibility() {
    skip_if_no_cuquantum!();

    use pecos_cuquantum::QubitId;

    // Two simulators with the same seed should give the same results
    let mut sim1 = CuStateVec::with_seed(2, 12345).expect("Failed to create CuStateVec");
    let mut sim2 = CuStateVec::with_seed(2, 12345).expect("Failed to create CuStateVec");

    // Create superposition and measure
    sim1.h(&[QubitId(0), QubitId(1)]);
    sim2.h(&[QubitId(0), QubitId(1)]);

    let samples1 = sim1.sample(10);
    let samples2 = sim2.sample(10);

    assert_eq!(
        samples1, samples2,
        "Same seed should give same sampling results"
    );
}

// =============================================================================
// Edge case tests
// =============================================================================

#[test]
fn test_custatevec_identity_operations() {
    skip_if_no_cuquantum!();

    use pecos_cuquantum::QubitId;

    let mut sim = CuStateVec::new(2).expect("Failed to create CuStateVec");

    // X^2 = I
    sim.x(&[QubitId(0)]);
    sim.x(&[QubitId(0)]);
    let result = sim.mz(&[QubitId(0)]);
    assert!(!result[0].outcome, "X^2 should be identity");

    sim.reset();

    // H^2 = I
    sim.h(&[QubitId(0)]);
    sim.h(&[QubitId(0)]);
    let result = sim.mz(&[QubitId(0)]);
    assert!(!result[0].outcome, "H^2 should be identity");

    sim.reset();

    // S^4 = I
    for _ in 0..4 {
        sim.sz(&[QubitId(0)]);
    }
    sim.h(&[QubitId(0)]); // Put in superposition to check phase
    sim.h(&[QubitId(0)]); // Back to computational basis
    let result = sim.mz(&[QubitId(0)]);
    assert!(!result[0].outcome, "S^4 should be identity");
}

#[test]
fn test_custatevec_multiple_qubit_gate_batches() {
    skip_if_no_cuquantum!();

    use pecos_cuquantum::QubitId;

    let mut sim = CuStateVec::new(4).expect("Failed to create CuStateVec");

    // Apply gates to multiple qubits at once
    sim.h(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    sim.x(&[QubitId(0), QubitId(2)]); // X on qubits 0 and 2
    sim.h(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);

    // Qubits 0 and 2 should be |1>, qubits 1 and 3 should be |0>
    let results = sim.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    assert!(results[0].outcome, "Qubit 0 should be |1>");
    assert!(!results[1].outcome, "Qubit 1 should be |0>");
    assert!(results[2].outcome, "Qubit 2 should be |1>");
    assert!(!results[3].outcome, "Qubit 3 should be |0>");
}
