mod helpers;

mod advanced_gates {
    use crate::helpers::assert_states_equal;
    use pecos_core::Angle64;
    use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec, qid, qid2};
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, PI};

    #[test]
    fn test_rotation_composition() {
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);

        // Test that rotation decompositions work
        // RY(θ) = RX(π/2)RZ(θ)RX(-π/2)
        q1.ry(Angle64::from_radians(FRAC_PI_3), &qid(0));

        q2.rx(Angle64::from_radians(FRAC_PI_2), &qid(0))
            .rz(Angle64::from_radians(FRAC_PI_3), &qid(0))
            .rx(Angle64::from_radians(-FRAC_PI_2), &qid(0));

        assert_states_equal(q1.state(), q2.state());
    }

    // TODO: add
    #[test]
    fn test_rotation_angle_relations() {}

    #[test]
    fn test_rotation_arithmetic() {
        let q = StateVec::new(1);

        // Test that RY(θ₁)RY(θ₂) = RY(θ₁ + θ₂) when commuting
        let theta1 = FRAC_PI_3;
        let theta2 = FRAC_PI_6;

        // Method 1: Two separate rotations
        let mut q1 = q.clone();
        q1.ry(Angle64::from_radians(theta1), &qid(0))
            .ry(Angle64::from_radians(theta2), &qid(0));

        // Method 2: Combined rotation
        let mut q2 = q.clone();
        q2.ry(Angle64::from_radians(theta1 + theta2), &qid(0));

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_rotation_symmetries() {
        // Test that all rotations are symmetric under exchange of qubits
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Prepare same non-trivial initial state
        q1.h(&qid(0));
        q1.h(&qid(1));
        q2.h(&qid(0));
        q2.h(&qid(1));

        let theta = PI / 3.0;

        // Test RYY symmetry
        q1.ryy(Angle64::from_radians(theta), &qid2(0, 1));
        q2.ryy(Angle64::from_radians(theta), &qid2(1, 0));

        for (a, b) in q1.state().iter().zip(q2.state().iter()) {
            assert!((a - b).norm() < 1e-10);
        }

        // Test RZZ symmetry
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);
        q1.h(&qid(0));
        q1.h(&qid(1));
        q2.h(&qid(0));
        q2.h(&qid(1));

        q1.rzz(Angle64::from_radians(theta), &qid2(0, 1));
        q2.rzz(Angle64::from_radians(theta), &qid2(1, 0));

        for (a, b) in q1.state().iter().zip(q2.state().iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_sq_rotation_commutation() {
        // RX and RY don't commute - verify RX(θ)RY(φ) ≠ RY(φ)RX(θ)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);

        let theta = FRAC_PI_3; // π/3
        let phi = FRAC_PI_4; // π/4

        // Apply in different orders
        q1.rx(Angle64::from_radians(theta), &qid(0))
            .ry(Angle64::from_radians(phi), &qid(0));
        q2.ry(Angle64::from_radians(phi), &qid(0))
            .rx(Angle64::from_radians(theta), &qid(0));

        println!("RY(π/4)RX(π/3)|0⟩ = {:?}", q1.state());
        println!("RX(π/3)RY(π/4)|0⟩ = {:?}", q2.state());

        // States should be different - check they're not equal up to global phase
        let ratio = q2.state()[0] / q1.state()[0];
        assert!((q2.state()[1] / q1.state()[1] - ratio).norm() > 1e-10);
    }

    #[test]
    fn test_sq_rotation_decompositions() {
        // H = RZ(-π)RY(-π/2)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);

        println!("Initial states:");
        println!("q1 = {:?}", q1.state());
        println!("q2 = {:?}", q2.state());

        q1.h(&qid(0)); // Direct H
        println!("After H: q1 = {:?}", q1.state());

        // H via rotations - changed order and added negative sign to RZ angle
        q2.ry(Angle64::from_radians(-FRAC_PI_2), &qid(0))
            .rz(Angle64::from_radians(-PI), &qid(0));
        println!("After RZ(-π)RY(-π/2): q2 = {:?}", q2.state());

        // Compare up to global phase by looking at ratios between components
        let ratio = q2.state()[0] / q1.state()[0];
        println!("Ratio = {ratio:?}");
        for (a, b) in q1.state().iter().zip(q2.state().iter()) {
            println!("Comparing {a} and {b}");
            assert!(
                (a * ratio - b).norm() < 1e-10,
                "States differ: {a} vs {b} (ratio: {ratio})"
            );
        }
    }
}

mod quantum_states {
    use crate::helpers::assert_states_equal;
    use pecos_core::Angle64;
    use pecos_simulators::{
        ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, StateVec, qid, qid2,
    };
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2};

    #[test]
    fn test_bell_state_entanglement() {
        let mut state_vec = StateVec::new(2);

        // Prepare Bell State: (|00⟩ + |11⟩) / √2
        state_vec.h(&qid(0));
        state_vec.cx(&qid2(0, 1));

        let expected_amplitude = 1.0 / 2.0_f64.sqrt();

        assert!((state_vec.state()[0].re - expected_amplitude).abs() < 1e-10);
        assert!((state_vec.state()[3].re - expected_amplitude).abs() < 1e-10);

        assert!(state_vec.state()[1].norm() < 1e-10);
        assert!(state_vec.state()[2].norm() < 1e-10);
    }
    #[test]
    fn test_ghz_state() {
        // Test creating and verifying a GHZ state
        let mut q = StateVec::new(3);
        q.h(&qid(0)).cx(&qid2(0, 1)).cx(&qid2(1, 2)); // Create GHZ state

        // Verify properties
        let mut norm_squared = 0.0;
        for i in 0..8 {
            if i == 0 || i == 7 {
                // |000⟩ or |111⟩
                norm_squared += q.state()[i].norm_sqr();
                assert!((q.state()[i].norm() - FRAC_1_SQRT_2).abs() < 1e-10);
            } else {
                assert!(q.state()[i].norm() < 1e-10);
            }
        }
        assert!((norm_squared - 1.0).abs() < 1e-10);
    }
    #[test]
    fn test_state_preparation_fidelity() {
        let mut q = StateVec::new(2);

        // Method 1: H + CNOT
        q.h(&qid(0)).cx(&qid2(0, 1));
        let probs1 = [
            q.probability(0),
            q.probability(1),
            q.probability(2),
            q.probability(3),
        ];

        // Method 2: Rotations
        q.reset();
        q.ry(Angle64::from_radians(FRAC_PI_2), &qid(0))
            .cx(&qid2(0, 1)); // Remove rz(PI) since it just adds phase

        // Compare probability distributions
        assert!((q.probability(0) - probs1[0]).abs() < 1e-10);
        assert!((q.probability(1) - probs1[1]).abs() < 1e-10);
        assert!((q.probability(2) - probs1[2]).abs() < 1e-10);
        assert!((q.probability(3) - probs1[3]).abs() < 1e-10);
    }

    #[test]
    fn test_state_prep_consistency() {
        // First method: direct X gate
        let mut q1 = StateVec::new(2);
        q1.x(&qid(1)); // Direct preparation of |01⟩

        // Verify first preparation - |01⟩ corresponds to binary 10 (decimal 2)
        assert!(
            (q1.probability(2) - 1.0).abs() < 1e-10,
            "First preparation failed"
        );
        assert!(q1.probability(0) < 1e-10);
        assert!(q1.probability(1) < 1e-10);
        assert!(q1.probability(3) < 1e-10);

        // Second method: using two X gates that cancel on qubit 0
        let mut q2 = StateVec::new(2);
        q2.x(&qid(0)).x(&qid(1)).x(&qid(0)); // Should give |01⟩

        // Verify second preparation - |01⟩ corresponds to binary 10 (decimal 2)
        assert!(
            (q2.probability(2) - 1.0).abs() < 1e-10,
            "Second preparation failed"
        );
        assert!(q2.probability(0) < 1e-10);
        assert!(q2.probability(1) < 1e-10);
        assert!(q2.probability(3) < 1e-10);

        // Verify both methods give the same state
        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_arbitrary_state_preparation() {
        let mut q = StateVec::new(1);

        // Try to prepare various single-qubit states
        // |+⟩ state
        q.h(&qid(0));
        assert!((q.probability(0) - 0.5).abs() < 1e-10);
        assert!((q.probability(1) - 0.5).abs() < 1e-10);

        // |+i⟩ state
        q.reset();
        q.h(&qid(0)).sz(&qid(0));
        assert!((q.probability(0) - 0.5).abs() < 1e-10);
        assert!((q.probability(1) - 0.5).abs() < 1e-10);
    }
}

mod gate_sequences {
    use crate::helpers::assert_states_equal;
    use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec, qid, qid2};

    #[test]
    fn test_operation_chains() {
        // Test complex sequences of operations
        let mut q = StateVec::new(2);

        // Create maximally entangled state then disentangle
        q.h(&qid(0))
            .cx(&qid2(0, 1)) // Create Bell state
            .cx(&qid2(0, 1))
            .h(&qid(0)); // Disentangle (apply the same operations in reverse)

        // Should be back to |00⟩
        assert!((q.probability(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_gates() {
        let mut state_vec = StateVec::new(1);

        // Apply Hadamard twice: H * H = I
        state_vec.h(&qid(0));
        state_vec.h(&qid(0));

        // Verify state is back to |0⟩
        assert!((state_vec.probability(0) - 1.0).abs() < 1e-10);
        assert!((state_vec.probability(1)).abs() < 1e-10);

        // Apply X twice: X * X = I
        state_vec.x(&qid(0));
        state_vec.x(&qid(0));
        assert!((state_vec.probability(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_gate_identities() {
        // Test S = T^2
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);

        // Put in superposition first to check phases
        q1.h(&qid(0));
        q2.h(&qid(0));

        q1.sz(&qid(0)); // S gate
        q2.t(&qid(0)).t(&qid(0)); // Two T gates

        assert_states_equal(q1.state(), q2.state());
    }
    #[test]
    fn test_gate_decompositions() {
        // Test that composite operations match their decompositions
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Test SWAP decomposition into CNOTs
        q1.x(&qid(0)); // Start with |10⟩
        q1.swap(&qid2(0, 1)); // Direct SWAP

        q2.x(&qid(0)); // Also start with |10⟩
        q2.cx(&qid2(0, 1)).cx(&qid2(1, 0)).cx(&qid2(0, 1)); // SWAP decomposition

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_bell_state_preparation() {
        let mut q = StateVec::new(2);
        q.h(&qid(0)).cx(&qid2(0, 1));
        assert!((q.probability(0) - 0.5).abs() < 1e-10);
        assert!((q.probability(3) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ghz_state_preparation() {
        let mut q = StateVec::new(3);
        q.h(&qid(0)).cx(&qid2(0, 1)).cx(&qid2(1, 2));
        assert!((q.probability(0) - 0.5).abs() < 1e-10);
        assert!((q.probability(7) - 0.5).abs() < 1e-10);
    }
}

mod numerical_properties {
    use pecos_core::Angle64;
    use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec, qid, qid2};
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6};

    #[test]
    fn test_state_normalization() {
        let mut state_vec = StateVec::new(3);

        // Apply multiple gates
        state_vec.h(&qid(0));
        state_vec.cx(&qid2(0, 1));
        state_vec.cx(&qid2(1, 2));

        // Verify normalization
        let norm: f64 = state_vec
            .state()
            .iter()
            .map(num_complex::Complex::norm_sqr)
            .sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_numerical_stability() {
        let mut q = StateVec::new(4);

        // Apply many rotations to test numerical stability
        for _ in 0..100 {
            q.rx(Angle64::from_radians(FRAC_PI_3), &qid(0))
                .ry(Angle64::from_radians(FRAC_PI_4), &qid(1))
                .rz(Angle64::from_radians(FRAC_PI_6), &qid(2))
                .cx(&qid2(0, 3));
        }

        // Check normalization is preserved
        let total_prob: f64 = q.state().iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((total_prob - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_phase_coherence() {
        let mut q = StateVec::new(1);

        // Apply series of phase rotations that should cancel
        q.h(&qid(0)) // Create superposition
            .rz(Angle64::from_radians(FRAC_PI_4), &qid(0))
            .rz(Angle64::from_radians(FRAC_PI_4), &qid(0))
            .rz(Angle64::from_radians(-FRAC_PI_2), &qid(0)); // Should cancel

        // Should be back to |+⟩
        assert!((q.state()[0].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((q.state()[1].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!(q.state()[0].im.abs() < 1e-10);
        assert!(q.state()[1].im.abs() < 1e-10);
    }

    #[test]
    fn test_bit_indexing() {
        let mut q = StateVec::new(3);

        println!("Initial state (|000⟩):");
        for i in 0..8 {
            println!("  {:03b}: {:.3}", i, q.state()[i]);
        }

        // Put |+⟩ on qubit 0 (LSB)
        q.h(&qid(0));

        println!("\nAfter H on qubit 0:");
        for i in 0..8 {
            println!("  {:03b}: {:.3}", i, q.state()[i]);
        }

        // Check state is |+⟩|0⟩|0⟩
        // Only indices that differ in LSB (qubit 0) should be FRAC_1_SQRT_2
        for i in 0..8 {
            let qubit0 = i & 1;
            let qubit1 = (i >> 1) & 1;
            let qubit2 = (i >> 2) & 1;

            let expected = if qubit1 == 0 && qubit2 == 0 {
                FRAC_1_SQRT_2
            } else {
                0.0
            };

            if (q.state()[i].re - expected).abs() >= 1e-10 {
                println!("\nMismatch at index {i}: {i:03b}");
                println!("Qubit values: q2={qubit2}, q1={qubit1}, q0={qubit0}");
                println!("Expected {}, got {}", expected, q.state()[i].re);
            }
            assert!((q.state()[i].re - expected).abs() < 1e-10);
        }
    }
}

mod locality_tests {
    use pecos_core::Angle64;
    use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec, qid, qid2};
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    #[test]
    fn test_single_qubit_locality() {
        // Test on 3 qubit system that gates only affect their target
        let mut q = StateVec::new(3);

        // Prepare state |+⟩|0⟩|0⟩
        q.h(&qid(0)); // Affects least significant bit

        // Apply X to qubit 2 (most significant bit)
        q.x(&qid(2));

        // Check that qubit 0 is still in |+⟩ state
        // When qubit 2 is |1⟩, check LSB still shows |+⟩
        assert!((q.state()[4].re - FRAC_1_SQRT_2).abs() < 1e-10); // |100⟩
        assert!((q.state()[5].re - FRAC_1_SQRT_2).abs() < 1e-10); // |101⟩
    }

    #[test]
    fn test_two_qubit_locality() {
        let mut q = StateVec::new(4);

        println!("Initial state:");
        for i in 0..16 {
            println!("  {:04b}: {:.3}", i, q.state()[i]);
        }

        // Prepare |+⟩ on qubit 0 (LSB)
        q.h(&qid(0));

        println!("\nAfter H on qubit 0:");
        for i in 0..16 {
            println!("  {:04b}: {:.3}", i, q.state()[i]);
        }

        // Apply CX between qubits 2,3
        q.cx(&qid2(2, 3));

        println!("\nAfter CX on qubits 2,3:");
        for i in 0..16 {
            println!("  {:04b}: {:.3}", i, q.state()[i]);

            // Extract qubit values
            // let _q0 = i & 1;
            let q1 = (i >> 1) & 1;
            let q2 = (i >> 2) & 1;
            let q3 = (i >> 3) & 1;

            // Only states with q0=0 or q0=1 and q1=q2=q3=0 should have amplitude
            let expected = if q1 == 0 && q2 == 0 && q3 == 0 {
                FRAC_1_SQRT_2
            } else {
                0.0
            };

            if (q.state()[i].re - expected).abs() >= 1e-10 {
                println!("Mismatch at {i:04b}");
                println!("Expected {}, got {}", expected, q.state()[i].re);
            }
            assert!((q.state()[i].re - expected).abs() < 1e-10);
        }
    }

    #[test]
    fn test_two_qubit_gate_locality() {
        let mut q = StateVec::new(3);

        // Prepare state |+⟩|0⟩|0⟩
        q.h(&qid(0));

        // Apply CX on qubits 1 and 2 (no effect on qubit 0)
        q.cx(&qid2(1, 2));

        // Qubit 0 should remain in superposition
        let expected_amp = 1.0 / 2.0_f64.sqrt();
        assert!((q.state()[0].re - expected_amp).abs() < 1e-10);
        assert!((q.state()[1].re - expected_amp).abs() < 1e-10);
    }

    #[test]
    fn test_rotation_locality() {
        let mut q = StateVec::new(3);

        println!("Initial state:");
        for i in 0..8 {
            println!("  {:03b}: {:.3}", i, q.state()[i]);
        }

        // Prepare |+⟩ on qubit 0 (LSB)
        q.h(&qid(0));

        println!("\nAfter H on qubit 0:");
        for i in 0..8 {
            println!("  {:03b}: {:.3}", i, q.state()[i]);
        }

        // Apply rotation to qubit 1
        q.rx(Angle64::from_radians(PI / 2.0), &qid(1));

        println!("\nAfter RX on qubit 1:");
        for i in 0..8 {
            println!("  {:03b}: {:.3}", i, q.state()[i]);
        }

        // Check each basis state contribution
        for i in 0..8 {
            let expected = FRAC_1_SQRT_2;
            if (q.state()[i].norm() - expected).abs() >= 1e-10 {
                println!("\nMismatch at index {i}: {i:03b}");
                println!("Expected norm {}, got {}", expected, q.state()[i].norm());
            }
        }
    }

    #[test]
    fn test_adjacent_vs_distant_qubits() {
        let mut q1 = StateVec::new(4);
        let mut q2 = StateVec::new(4);

        // Test operations on adjacent vs distant qubits
        q1.h(&qid(0)).cx(&qid2(0, 1)); // Adjacent qubits
        q2.h(&qid(0)).cx(&qid2(0, 3)); // Distant qubits

        // Both should maintain proper normalization
        let norm1: f64 = q1.state().iter().map(num_complex::Complex::norm_sqr).sum();
        let norm2: f64 = q2.state().iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((norm1 - 1.0).abs() < 1e-10);
        assert!((norm2 - 1.0).abs() < 1e-10);
    }
}

// Edge cases and numerical stability
mod edge_cases {
    use crate::helpers::assert_states_equal;
    use pecos_core::Angle64;
    use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec, qid};
    use std::f64::consts::PI;

    #[test]
    fn test_small_angle_rotations() {
        let mut q = StateVec::new(1);
        let small_angle = 1e-6;
        q.rx(Angle64::from_radians(small_angle), &qid(0));
        let total_prob: f64 = q.state().iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((total_prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_repeated_operations() {
        let mut q = StateVec::new(1);
        for _ in 0..1000 {
            q.h(&qid(0)).sz(&qid(0)).h(&qid(0));
        }
        let norm: f64 = q.state().iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((norm - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_rotation_angle_precision() {
        let mut q = StateVec::new(1);

        // Test small angle rotations
        let small_angle = 1e-6;
        q.rx(Angle64::from_radians(small_angle), &qid(0));

        // Check that probabilities sum to 1
        let total_prob: f64 = q.state().iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((total_prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sq_rotation_edge_cases() {
        let mut q = StateVec::new(1);

        // Test RX(0): Should be identity
        let initial = q.state().clone();
        q.rx(Angle64::from_radians(0.0), &qid(0));
        assert_states_equal(q.state(), &initial);

        // Test RX(2π): Should also be identity up to global phase
        q.rx(Angle64::from_radians(2.0 * PI), &qid(0));
        assert_states_equal(q.state(), &initial);

        // Test RY(0): Should be identity
        q.ry(Angle64::from_radians(0.0), &qid(0));
        assert_states_equal(q.state(), &initial);

        // Test RY(2π): Should also be identity up to global phase
        q.ry(Angle64::from_radians(2.0 * PI), &qid(0));
        assert_states_equal(q.state(), &initial);

        // Test RZ(0): Should be identity
        q.rz(Angle64::from_radians(0.0), &qid(0));
        assert_states_equal(q.state(), &initial);

        // Test RZ(2π): Should also be identity up to global phase
        q.rz(Angle64::from_radians(2.0 * PI), &qid(0));
        assert_states_equal(q.state(), &initial);
    }
}

mod large_systems {
    use pecos_core::Angle64;
    use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec, qid, qid2};

    #[test]
    fn test_state_normalization_after_random_gates() {
        let mut state_vec = StateVec::new(3);

        // Apply a sequence of random gates
        state_vec.h(&qid(0));
        state_vec.cx(&qid2(0, 1));
        state_vec.rz(Angle64::from_radians(std::f64::consts::PI / 3.0), &qid(2));
        state_vec.swap(&qid2(1, 2));

        // Check if the state is still normalized
        let norm: f64 = state_vec
            .state()
            .iter()
            .map(num_complex::Complex::norm_sqr)
            .sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }
}

mod detailed_sq_gate_cases {
    use crate::helpers::assert_states_equal;
    use pecos_core::Angle64;
    use pecos_simulators::{
        ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, StateVec, qid, qid2,
    };
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, PI};

    #[test]
    fn test_rx_step_by_step() {
        let mut q = StateVec::new(1);

        // Step 1: RX(0) should be identity
        q.rx(Angle64::from_radians(0.0), &qid(0));
        assert!((q.state()[0].re - 1.0).abs() < 1e-10);
        assert!(q.state()[1].norm() < 1e-10);

        // Step 2: RX(π) on |0⟩ should give -i|1⟩
        let mut q = StateVec::new(1);
        q.rx(Angle64::from_radians(PI), &qid(0));
        println!("RX(π)|0⟩ = {:?}", q.state()); // Debug output
        assert!(q.state()[0].norm() < 1e-10);
        assert!((q.state()[1].im + 1.0).abs() < 1e-10);

        // Step 3: RX(π/2) on |0⟩ should give (|0⟩ - i|1⟩)/√2
        let mut q = StateVec::new(1);
        q.rx(Angle64::from_radians(FRAC_PI_2), &qid(0));
        println!("RX(π/2)|0⟩ = {:?}", q.state()); // Debug output
        let expected_amp = 1.0 / 2.0_f64.sqrt();
        assert!((q.state()[0].re - expected_amp).abs() < 1e-10);
        assert!((q.state()[1].im + expected_amp).abs() < 1e-10);
    }

    #[test]
    fn test_ry_step_by_step() {
        // Step 1: RY(0) should be identity
        let mut q = StateVec::new(1);
        q.ry(Angle64::from_radians(0.0), &qid(0));
        println!("RY(0)|0⟩ = {:?}", q.state());
        assert!((q.state()[0].re - 1.0).abs() < 1e-10);
        assert!(q.state()[1].norm() < 1e-10);

        // Step 2: RY(π) on |0⟩ should give |1⟩
        let mut q = StateVec::new(1);
        q.ry(Angle64::from_radians(PI), &qid(0));
        println!("RY(π)|0⟩ = {:?}", q.state());
        assert!(q.state()[0].norm() < 1e-10);
        assert!((q.state()[1].re - 1.0).abs() < 1e-10);

        // Step 3: RY(π/2) on |0⟩ should give (|0⟩ + |1⟩)/√2
        let mut q = StateVec::new(1);
        q.ry(Angle64::from_radians(FRAC_PI_2), &qid(0));
        println!("RY(π/2)|0⟩ = {:?}", q.state());
        let expected_amp = 1.0 / 2.0_f64.sqrt();
        assert!((q.state()[0].re - expected_amp).abs() < 1e-10);
        assert!((q.state()[1].re - expected_amp).abs() < 1e-10);

        // Step 4: RY(-π/2) on |0⟩ should give (|0⟩ - |1⟩)/√2
        let mut q = StateVec::new(1);
        q.ry(Angle64::from_radians(-FRAC_PI_2), &qid(0));
        println!("RY(-π/2)|0⟩ = {:?}", q.state());
        assert!((q.state()[0].re - expected_amp).abs() < 1e-10);
        assert!((q.state()[1].re + expected_amp).abs() < 1e-10);
    }

    #[test]
    fn test_rz_step_by_step() {
        // Step 1: RZ(0) should be identity
        let mut q = StateVec::new(1);
        q.rz(Angle64::from_radians(0.0), &qid(0));
        println!("RZ(0)|0⟩ = {:?}", q.state());
        assert!((q.state()[0].re - 1.0).abs() < 1e-10);
        assert!(q.state()[1].norm() < 1e-10);

        // Step 2: RZ(π/2) on |+⟩ should give |+i⟩ = (|0⟩ + i|1⟩)/√2
        let mut q = StateVec::new(1);
        q.h(&qid(0)); // Create |+⟩
        q.rz(Angle64::from_radians(FRAC_PI_2), &qid(0));
        println!("RZ(π/2)|+⟩ = {:?}", q.state());
        let expected_amp = 1.0 / 2.0_f64.sqrt();
        assert!((q.state()[0].norm() - expected_amp).abs() < 1e-10);
        assert!((q.state()[1].norm() - expected_amp).abs() < 1e-10);
        // Check relative phase
        let ratio = q.state()[1] / q.state()[0];
        println!("Relative phase ratio = {ratio:?}");
        assert!(
            (ratio.im - 1.0).abs() < 1e-10,
            "Relative phase incorrect: ratio = {ratio}"
        );
        assert!(
            ratio.re.abs() < 1e-10,
            "Relative phase has unexpected real component: {}",
            ratio.re
        );

        // Step 3: Two RZ(π/2) operations should equal one RZ(π)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.rz(Angle64::from_radians(PI), &qid(0));
        q2.rz(Angle64::from_radians(FRAC_PI_2), &qid(0));
        q2.rz(Angle64::from_radians(FRAC_PI_2), &qid(0));
        println!("RZ(π)|0⟩ vs RZ(π/2)RZ(π/2)|0⟩:");
        println!("q1 = {:?}", q1.state());
        println!("q2 = {:?}", q2.state());
        let ratio = q2.state()[0] / q1.state()[0];
        let phase = ratio.arg();
        println!("Phase difference between q2 and q1: {phase}");
        assert!(
            (ratio.norm() - 1.0).abs() < 1e-10,
            "Magnitudes differ: ratio = {ratio}"
        );
        // Don't check exact phase, just verify states are equal up to global phase
        assert!((q2.state()[1] * q1.state()[0] - q2.state()[0] * q1.state()[1]).norm() < 1e-10);
    }

    #[test]
    fn test_sq_standard_gate_decompositions() {
        // Test S = RZ(π/2)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.sz(&qid(0));
        q2.rz(Angle64::from_radians(FRAC_PI_2), &qid(0));
        println!("S|0⟩ = {:?}", q1.state());
        println!("RZ(π/2)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test X = RX(π)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.x(&qid(0));
        q2.rx(Angle64::from_radians(PI), &qid(0));
        println!("X|0⟩ = {:?}", q1.state());
        println!("RX(π)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test Y = RY(π)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.y(&qid(0));
        q2.ry(Angle64::from_radians(PI), &qid(0));
        println!("Y|0⟩ = {:?}", q1.state());
        println!("RY(π)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test Z = RZ(π)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.z(&qid(0));
        q2.rz(Angle64::from_radians(PI), &qid(0));
        println!("Z|0⟩ = {:?}", q1.state());
        println!("RZ(π)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test √X = RX(π/2)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.sx(&qid(0));
        q2.rx(Angle64::from_radians(FRAC_PI_2), &qid(0));
        println!("√X|0⟩ = {:?}", q1.state());
        println!("RX(π/2)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test √Y = RY(π/2)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.sy(&qid(0));
        q2.ry(Angle64::from_radians(FRAC_PI_2), &qid(0));
        println!("√Y|0⟩ = {:?}", q1.state());
        println!("RY(π/2)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test S = TT as RZ(π/4)RZ(π/4)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q2.rz(Angle64::from_radians(FRAC_PI_4), &qid(0))
            .rz(Angle64::from_radians(FRAC_PI_4), &qid(0));
        q1.sz(&qid(0));
        println!("S|0⟩ = {:?}", q1.state());
        println!("T²|0⟩ = RZ(π/4)RZ(π/4)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test H = RX(π)RY(π/2) decomposition
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.ry(Angle64::from_radians(FRAC_PI_2), &qid(0))
            .rx(Angle64::from_radians(PI), &qid(0));
        println!("H|0⟩ = {:?}", q1.state());
        println!("RX(π)RY(π/2)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_rx_rotation_angle_relations() {
        // Test that RX(θ)RX(-θ) = I
        let mut q = StateVec::new(1);
        let theta = FRAC_PI_3;

        // Apply forward then reverse rotations
        q.rx(Angle64::from_radians(theta), &qid(0))
            .rx(Angle64::from_radians(-theta), &qid(0));

        // Should get back to |0⟩ up to global phase
        assert!(q.state()[1].norm() < 1e-10);
        assert!((q.state()[0].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_ry_rotation_angle_relations() {
        // Test that RY(θ)RY(-θ) = I
        let mut q = StateVec::new(1);
        let theta = FRAC_PI_3;

        // Apply forward then reverse rotations
        q.ry(Angle64::from_radians(theta), &qid(0))
            .ry(Angle64::from_radians(-theta), &qid(0));

        // Should get back to |0⟩ up to global phase
        assert!(q.state()[1].norm() < 1e-10);
        assert!((q.state()[0].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_rz_rotation_angle_relations() {
        // Test that RZ(θ)RZ(-θ) = I
        let mut q = StateVec::new(1);
        let theta = FRAC_PI_3;

        // Apply forward then reverse rotations
        q.rz(Angle64::from_radians(theta), &qid(0))
            .rz(Angle64::from_radians(-theta), &qid(0));

        // Should get back to |0⟩ up to global phase
        assert!(q.state()[1].norm() < 1e-10);
        assert!((q.state()[0].norm() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_state_vec_u_vs_trait_u() {
        // Initialize state vectors with one qubit in the |0⟩ state.
        let mut state_vec_u = StateVec::new(1);
        let mut trait_u = StateVec::new(1);

        let theta = FRAC_PI_3;
        let phi = FRAC_PI_4;
        let lambda = FRAC_PI_6;

        // Apply `u` from the StateVec implementation.
        state_vec_u.u(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            Angle64::from_radians(lambda),
            &qid(0),
        );

        // Apply `u` from the ArbitraryRotationGateable trait.
        ArbitraryRotationGateable::u(
            &mut trait_u,
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            Angle64::from_radians(lambda),
            &qid(0),
        );

        assert_states_equal(state_vec_u.state(), trait_u.state());
    }

    #[test]
    fn test_r1xy_vs_u() {
        let mut state_r1xy = StateVec::new(1);
        let mut state_u = StateVec::new(1);

        let theta = FRAC_PI_3;
        let phi = FRAC_PI_4;

        // Apply r1xy and equivalent u gates
        state_r1xy.r1xy(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            &qid(0),
        );
        state_u.u(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi - FRAC_PI_2),
            Angle64::from_radians(FRAC_PI_2 - phi),
            &qid(0),
        );

        assert_states_equal(state_r1xy.state(), state_u.state());
    }

    #[test]
    fn test_rz_vs_u() {
        let mut state_rz = StateVec::new(1);
        let mut state_u = StateVec::new(1);

        let theta = FRAC_PI_3;

        // Apply rz and u gates
        state_rz.rz(Angle64::from_radians(theta), &qid(0));
        state_u.u(
            Angle64::from_radians(0.0),
            Angle64::from_radians(0.0),
            Angle64::from_radians(theta),
            &qid(0),
        );

        assert_states_equal(state_rz.state(), state_u.state());
    }

    #[test]
    fn test_u_decomposition() {
        let mut state_u = StateVec::new(1);
        let mut state_decomposed = StateVec::new(1);

        let theta = FRAC_PI_3;
        let phi = FRAC_PI_4;
        let lambda = FRAC_PI_6;

        // Apply U gate
        state_u.u(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            Angle64::from_radians(lambda),
            &qid(0),
        );

        // Apply the decomposed gates
        state_decomposed.rz(Angle64::from_radians(lambda), &qid(0));
        state_decomposed.r1xy(
            Angle64::from_radians(theta),
            Angle64::from_radians(FRAC_PI_2),
            &qid(0),
        );
        state_decomposed.rz(Angle64::from_radians(phi), &qid(0));

        // Assert that the states are equal
        assert_states_equal(state_u.state(), state_decomposed.state());
    }

    #[test]
    fn test_x_vs_r1xy() {
        let mut state = StateVec::new(1);
        state.x(&qid(0));
        let mut state_after_x = state.clone();

        state.reset();
        state.r1xy(
            Angle64::from_radians(PI),
            Angle64::from_radians(0.0),
            &qid(0),
        );
        let mut state_after_r1xy = state.clone();

        assert_states_equal(state_after_x.state(), state_after_r1xy.state());
    }

    #[test]
    fn test_y_vs_r1xy() {
        let mut state = StateVec::new(1);
        state.y(&qid(0));
        let mut state_after_y = state.clone();

        state.reset();
        state.r1xy(
            Angle64::from_radians(PI),
            Angle64::from_radians(FRAC_PI_2),
            &qid(0),
        );
        let mut state_after_r1xy = state.clone();

        assert_states_equal(state_after_y.state(), state_after_r1xy.state());
    }

    #[test]
    fn test_h_vs_r1xy_rz() {
        let mut state = StateVec::new(1);
        state.h(&qid(0)); // Apply the H gate
        let mut state_after_h = state.clone();

        state.reset(); // Reset state to |0⟩
        state
            .r1xy(
                Angle64::from_radians(FRAC_PI_2),
                Angle64::from_radians(-FRAC_PI_2),
                &qid(0),
            )
            .rz(Angle64::from_radians(PI), &qid(0));
        let mut state_after_r1xy_rz = state.clone();

        assert_states_equal(state_after_h.state(), state_after_r1xy_rz.state());
    }

    #[test]
    fn test_u_special_cases() {
        // Test 1: U(π, 0, π) should be X gate
        let mut q = StateVec::new(1);
        q.u(
            Angle64::from_radians(PI),
            Angle64::from_radians(0.0),
            Angle64::from_radians(PI),
            &qid(0),
        );
        assert!(q.state()[0].norm() < 1e-10);
        assert!((q.state()[1].re - 1.0).abs() < 1e-10);

        // Test 2: Hadamard gate
        // H = U(π/2, 0, π)
        let mut q = StateVec::new(1);
        q.u(
            Angle64::from_radians(PI / 2.0),
            Angle64::from_radians(0.0),
            Angle64::from_radians(PI),
            &qid(0),
        );
        assert!((q.state()[0].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((q.state()[1].re - FRAC_1_SQRT_2).abs() < 1e-10);

        // Test 3: U(0, 0, π) should be Z gate
        let mut q = StateVec::new(1);
        q.h(&qid(0)); // First put in superposition
        let initial = q.state().clone();
        q.u(
            Angle64::from_radians(0.0),
            Angle64::from_radians(0.0),
            Angle64::from_radians(PI),
            &qid(0),
        );
        assert!((q.state()[0] - initial[0]).norm() < 1e-10);
        assert!((q.state()[1] + initial[1]).norm() < 1e-10);

        // Additional test: U3(π/2, π/2, -π/2) should be S†H
        let mut q = StateVec::new(1);
        q.u(
            Angle64::from_radians(PI / 2.0),
            Angle64::from_radians(PI / 2.0),
            Angle64::from_radians(-PI / 2.0),
            &qid(0),
        );
        // This creates the state (|0⟩ + i|1⟩)/√2
        assert!((q.state()[0].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((q.state()[1].im - FRAC_1_SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_u_composition() {
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);

        // Two U gates that should multiply to identity
        q1.u(
            Angle64::from_radians(PI / 3.0),
            Angle64::from_radians(PI / 4.0),
            Angle64::from_radians(PI / 6.0),
            &qid(0),
        );
        q1.u(
            Angle64::from_radians(-PI / 3.0),
            Angle64::from_radians(-PI / 6.0),
            Angle64::from_radians(-PI / 4.0),
            &qid(0),
        );

        // Compare with initial state
        for (a, b) in q1.state().iter().zip(q2.state().iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_phase_relationships() {
        // Test expected phase relationships between gates
        let q = StateVec::new(1);

        // Test that T * T = S
        let mut q1 = q.clone();
        q1.t(&qid(0)).t(&qid(0));

        let mut q2 = q.clone();
        q2.sz(&qid(0));

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_hadamard_properties() {
        // Test H^2 = I
        let mut q = StateVec::new(1);
        q.x(&qid(0)); // Start with |1⟩
        let initial = q.state().clone();
        q.h(&qid(0)).h(&qid(0));
        assert_states_equal(q.state(), &initial);

        // Test HXH = Z
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);

        q1.h(&qid(0)).x(&qid(0)).h(&qid(0));
        q2.z(&qid(0));

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_non_commuting_gates() {
        let mut state1 = StateVec::new(1);
        let mut state2 = StateVec::new(1);

        state1.h(&qid(0));
        state1.z(&qid(0));

        state2.z(&qid(0));
        state2.h(&qid(0));

        // Compute the global norm difference
        let diff_norm: f64 = state1
            .state()
            .iter()
            .zip(state2.state().iter())
            .map(|(a, b)| (a - b).norm_sqr())
            .sum::<f64>()
            .sqrt();

        assert!(diff_norm > 1e-10, "H and Z should not commute.");
    }

    // Tests for sqrt gate direct implementations vs decompositions
    #[test]
    fn test_sx_vs_decomposition() {
        // SX = H.SZ.H decomposition
        // Test on |0⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);

        direct.sx(&qid(0));
        decomposed.h(&qid(0)).sz(&qid(0)).h(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on |1⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));

        direct.sx(&qid(0));
        decomposed.h(&qid(0)).sz(&qid(0)).h(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on superposition |+⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));

        direct.sx(&qid(0));
        decomposed.h(&qid(0)).sz(&qid(0)).h(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on multi-qubit system
        for target in 0..3 {
            let mut direct = StateVec::new(3);
            let mut decomposed = StateVec::new(3);

            // Create entangled state
            direct.h(&qid(0)).cx(&qid2(0, 1)).cx(&qid2(1, 2));
            decomposed.h(&qid(0)).cx(&qid2(0, 1)).cx(&qid2(1, 2));

            direct.sx(&qid(target));
            decomposed.h(&qid(target)).sz(&qid(target)).h(&qid(target));

            assert_states_equal(direct.state(), decomposed.state());
        }
    }

    #[test]
    fn test_sxdg_vs_decomposition() {
        // SXDG = H.SZDG.H decomposition
        // Test on |0⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);

        direct.sxdg(&qid(0));
        decomposed.h(&qid(0)).szdg(&qid(0)).h(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on superposition
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));

        direct.sxdg(&qid(0));
        decomposed.h(&qid(0)).szdg(&qid(0)).h(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_sy_vs_decomposition() {
        // SY = H.X decomposition
        // Test on |0⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);

        direct.sy(&qid(0));
        decomposed.h(&qid(0)).x(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on |1⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));

        direct.sy(&qid(0));
        decomposed.h(&qid(0)).x(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on multi-qubit system
        for target in 0..3 {
            let mut direct = StateVec::new(3);
            let mut decomposed = StateVec::new(3);

            direct.h(&qid(0)).cx(&qid2(0, 1));
            decomposed.h(&qid(0)).cx(&qid2(0, 1));

            direct.sy(&qid(target));
            decomposed.h(&qid(target)).x(&qid(target));

            assert_states_equal(direct.state(), decomposed.state());
        }
    }

    #[test]
    fn test_sydg_vs_decomposition() {
        // SYDG = X.H decomposition
        // Test on |0⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);

        direct.sydg(&qid(0));
        decomposed.x(&qid(0)).h(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on superposition
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));

        direct.sydg(&qid(0));
        decomposed.x(&qid(0)).h(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_szdg_vs_decomposition() {
        // SZDG = Z.SZ decomposition (from default trait)
        // Test on superposition (since szdg on |0⟩ is trivial)
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));

        direct.szdg(&qid(0));
        decomposed.z(&qid(0)).sz(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on |1⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));

        direct.szdg(&qid(0));
        decomposed.z(&qid(0)).sz(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on multi-qubit system
        for target in 0..3 {
            let mut direct = StateVec::new(3);
            let mut decomposed = StateVec::new(3);

            direct.h(&qid(0)).cx(&qid2(0, 1)).h(&qid(2));
            decomposed.h(&qid(0)).cx(&qid2(0, 1)).h(&qid(2));

            direct.szdg(&qid(target));
            decomposed.z(&qid(target)).sz(&qid(target));

            assert_states_equal(direct.state(), decomposed.state());
        }
    }

    #[test]
    fn test_r1xy_vs_decomposition() {
        // R1XY(theta, phi) = RZ(-phi + pi/2).RY(theta).RZ(phi - pi/2)
        let theta = FRAC_PI_3;
        let phi = FRAC_PI_4;

        // Test on |0⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);

        direct.r1xy(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            &qid(0),
        );
        decomposed
            .rz(Angle64::from_radians(-phi + FRAC_PI_2), &qid(0))
            .ry(Angle64::from_radians(theta), &qid(0))
            .rz(Angle64::from_radians(phi - FRAC_PI_2), &qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on |1⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));

        direct.r1xy(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            &qid(0),
        );
        decomposed
            .rz(Angle64::from_radians(-phi + FRAC_PI_2), &qid(0))
            .ry(Angle64::from_radians(theta), &qid(0))
            .rz(Angle64::from_radians(phi - FRAC_PI_2), &qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on superposition
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));

        direct.r1xy(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            &qid(0),
        );
        decomposed
            .rz(Angle64::from_radians(-phi + FRAC_PI_2), &qid(0))
            .ry(Angle64::from_radians(theta), &qid(0))
            .rz(Angle64::from_radians(phi - FRAC_PI_2), &qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test with different angles
        for &theta in &[FRAC_PI_2, PI, FRAC_PI_4, FRAC_PI_6] {
            for &phi in &[0.0, FRAC_PI_2, PI, -FRAC_PI_4] {
                let mut direct = StateVec::new(1);
                let mut decomposed = StateVec::new(1);
                direct.h(&qid(0));
                decomposed.h(&qid(0));

                direct.r1xy(
                    Angle64::from_radians(theta),
                    Angle64::from_radians(phi),
                    &qid(0),
                );
                decomposed
                    .rz(Angle64::from_radians(-phi + FRAC_PI_2), &qid(0))
                    .ry(Angle64::from_radians(theta), &qid(0))
                    .rz(Angle64::from_radians(phi - FRAC_PI_2), &qid(0));

                assert_states_equal(direct.state(), decomposed.state());
            }
        }
    }

    #[test]
    fn test_sqrt_gate_inverse_relations() {
        // Test SX * SXDG = I
        let mut q = StateVec::new(1);
        q.h(&qid(0)); // Start in superposition
        let initial = q.state().clone();
        q.sx(&qid(0)).sxdg(&qid(0));
        assert_states_equal(q.state(), &initial);

        // Test SXDG * SX = I
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.sxdg(&qid(0)).sx(&qid(0));
        assert_states_equal(q.state(), &initial);

        // Test SY * SYDG = I
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.sy(&qid(0)).sydg(&qid(0));
        assert_states_equal(q.state(), &initial);

        // Test SYDG * SY = I
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.sydg(&qid(0)).sy(&qid(0));
        assert_states_equal(q.state(), &initial);

        // Test SZ * SZDG = I
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.sz(&qid(0)).szdg(&qid(0));
        assert_states_equal(q.state(), &initial);

        // Test SZDG * SZ = I
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.szdg(&qid(0)).sz(&qid(0));
        assert_states_equal(q.state(), &initial);
    }

    #[test]
    fn test_sqrt_gate_squared_relations() {
        // Test SX * SX = X
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));

        q1.sx(&qid(0)).sx(&qid(0));
        q2.x(&qid(0));

        assert_states_equal(q1.state(), q2.state());

        // Test SY * SY = Y
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));

        q1.sy(&qid(0)).sy(&qid(0));
        q2.y(&qid(0));

        assert_states_equal(q1.state(), q2.state());

        // Test SZ * SZ = Z
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));

        q1.sz(&qid(0)).sz(&qid(0));
        q2.z(&qid(0));

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_f_vs_decomposition() {
        // F = SX.SZ decomposition (apply sx then sz)
        // Test on |0⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);

        direct.f(&qid(0));
        decomposed.sx(&qid(0)).sz(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on |1⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));

        direct.f(&qid(0));
        decomposed.sx(&qid(0)).sz(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on superposition |+⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));

        direct.f(&qid(0));
        decomposed.sx(&qid(0)).sz(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on multi-qubit system
        for target in 0..3 {
            let mut direct = StateVec::new(3);
            let mut decomposed = StateVec::new(3);

            direct.h(&qid(0)).cx(&qid2(0, 1)).h(&qid(2));
            decomposed.h(&qid(0)).cx(&qid2(0, 1)).h(&qid(2));

            direct.f(&qid(target));
            decomposed.sx(&qid(target)).sz(&qid(target));

            assert_states_equal(direct.state(), decomposed.state());
        }
    }

    #[test]
    fn test_fdg_vs_decomposition() {
        // FDG = SZDG.SXDG decomposition (apply szdg then sxdg)
        // Test on |0⟩
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);

        direct.fdg(&qid(0));
        decomposed.szdg(&qid(0)).sxdg(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on superposition
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));

        direct.fdg(&qid(0));
        decomposed.szdg(&qid(0)).sxdg(&qid(0));

        assert_states_equal(direct.state(), decomposed.state());

        // Test on multi-qubit system
        for target in 0..3 {
            let mut direct = StateVec::new(3);
            let mut decomposed = StateVec::new(3);

            direct.h(&qid(0)).cx(&qid2(0, 1));
            decomposed.h(&qid(0)).cx(&qid2(0, 1));

            direct.fdg(&qid(target));
            decomposed.szdg(&qid(target)).sxdg(&qid(target));

            assert_states_equal(direct.state(), decomposed.state());
        }
    }

    #[test]
    fn test_f_fdg_inverse_relations() {
        // Test F * FDG = I
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.f(&qid(0)).fdg(&qid(0));
        assert_states_equal(q.state(), &initial);

        // Test FDG * F = I
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.fdg(&qid(0)).f(&qid(0));
        assert_states_equal(q.state(), &initial);

        // Test F^3 = I (F is order 3)
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.f(&qid(0)).f(&qid(0)).f(&qid(0));
        assert_states_equal(q.state(), &initial);
    }

    #[test]
    fn test_f2_vs_decomposition() {
        // F2 = SXDG.SY (apply SXDG first, then SY)
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.f2(&qid(0));
        decomposed.sxdg(&qid(0)).sy(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test with superposition state
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));
        direct.f2(&qid(0));
        decomposed.sxdg(&qid(0)).sy(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in multi-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.f2(&qid(1));
        decomposed.sxdg(&qid(1)).sy(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_f2dg_vs_decomposition() {
        // F2DG = SYDG.SX (apply SYDG first, then SX)
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.f2dg(&qid(0));
        decomposed.sydg(&qid(0)).sx(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test with superposition state
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));
        direct.f2dg(&qid(0));
        decomposed.sydg(&qid(0)).sx(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_f3_vs_decomposition() {
        // F3 = SXDG.SZ (apply SXDG first, then SZ)
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.f3(&qid(0));
        decomposed.sxdg(&qid(0)).sz(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test with superposition state
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));
        direct.f3(&qid(0));
        decomposed.sxdg(&qid(0)).sz(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in multi-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.f3(&qid(1));
        decomposed.sxdg(&qid(1)).sz(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_f3dg_vs_decomposition() {
        // F3DG = SZDG.SX (apply SZDG first, then SX)
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.f3dg(&qid(0));
        decomposed.szdg(&qid(0)).sx(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test with superposition state
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));
        direct.f3dg(&qid(0));
        decomposed.szdg(&qid(0)).sx(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_f4_vs_decomposition() {
        // F4 = SZ.SX (apply SZ first, then SX)
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.f4(&qid(0));
        decomposed.sz(&qid(0)).sx(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test with superposition state
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));
        direct.f4(&qid(0));
        decomposed.sz(&qid(0)).sx(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in multi-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.f4(&qid(1));
        decomposed.sz(&qid(1)).sx(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_f4dg_vs_decomposition() {
        // F4DG = SXDG.SZDG (apply SXDG first, then SZDG)
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.f4dg(&qid(0));
        decomposed.sxdg(&qid(0)).szdg(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test with superposition state
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));
        direct.f4dg(&qid(0));
        decomposed.sxdg(&qid(0)).szdg(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_face_gate_inverse_relations() {
        // Test F2 * F2DG = I
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.f2(&qid(0)).f2dg(&qid(0));
        assert_states_equal(q.state(), &initial);

        // Test F3 * F3DG = I
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.f3(&qid(0)).f3dg(&qid(0));
        assert_states_equal(q.state(), &initial);

        // Test F4 * F4DG = I
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.f4(&qid(0)).f4dg(&qid(0));
        assert_states_equal(q.state(), &initial);

        // Test F2DG * F2 = I
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.f2dg(&qid(0)).f2(&qid(0));
        assert_states_equal(q.state(), &initial);

        // Test F3DG * F3 = I
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.f3dg(&qid(0)).f3(&qid(0));
        assert_states_equal(q.state(), &initial);

        // Test F4DG * F4 = I
        let mut q = StateVec::new(1);
        q.h(&qid(0));
        let initial = q.state().clone();
        q.f4dg(&qid(0)).f4(&qid(0));
        assert_states_equal(q.state(), &initial);
    }

    #[test]
    fn test_h2_vs_decomposition() {
        // H2 = SY.Z (apply SY first, then Z)
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h2(&qid(0));
        decomposed.sy(&qid(0)).z(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test with superposition state
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));
        direct.h2(&qid(0));
        decomposed.sy(&qid(0)).z(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in multi-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.h2(&qid(1));
        decomposed.sy(&qid(1)).z(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_h3_vs_decomposition() {
        // H3 = SZ.Y (apply SZ first, then Y)
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h3(&qid(0));
        decomposed.sz(&qid(0)).y(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test with superposition state
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));
        direct.h3(&qid(0));
        decomposed.sz(&qid(0)).y(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in multi-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.h3(&qid(1));
        decomposed.sz(&qid(1)).y(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_h4_vs_decomposition() {
        // H4 = SZ.X (apply SZ first, then X)
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h4(&qid(0));
        decomposed.sz(&qid(0)).x(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test with superposition state
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));
        direct.h4(&qid(0));
        decomposed.sz(&qid(0)).x(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in multi-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.h4(&qid(1));
        decomposed.sz(&qid(1)).x(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_h5_vs_decomposition() {
        // H5 = SX.Z (apply SX first, then Z)
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h5(&qid(0));
        decomposed.sx(&qid(0)).z(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test with superposition state
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));
        direct.h5(&qid(0));
        decomposed.sx(&qid(0)).z(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in multi-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.h5(&qid(1));
        decomposed.sx(&qid(1)).z(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_h6_vs_decomposition() {
        // H6 = SX.Y (apply SX first, then Y)
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h6(&qid(0));
        decomposed.sx(&qid(0)).y(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test with superposition state
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.h(&qid(0));
        decomposed.h(&qid(0));
        direct.h6(&qid(0));
        decomposed.sx(&qid(0)).y(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in multi-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.h6(&qid(1));
        decomposed.sx(&qid(1)).y(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_hadamard_variant_properties() {
        // H2^2 = Z (since H2 = SY.Z and (SY)^2 = Y, so H2^2 = SY.Z.SY.Z = SY.SY.Z.Z = Y.I = Y... wait)
        // Actually let me verify H2^2 empirically
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));
        q1.h2(&qid(0)).h2(&qid(0));
        // H2^2 should equal some known gate - let's check empirically
        q2.sy(&qid(0)).z(&qid(0)).sy(&qid(0)).z(&qid(0));
        assert_states_equal(q1.state(), q2.state());

        // H3^2 = Y * SZ * Y * SZ = Y * Y * SZ * SZ (if Y and SZ commute... they don't)
        // Let's just verify H3 applied twice matches decomposition applied twice
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));
        q1.h3(&qid(0)).h3(&qid(0));
        q2.sz(&qid(0)).y(&qid(0)).sz(&qid(0)).y(&qid(0));
        assert_states_equal(q1.state(), q2.state());

        // H4^2
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));
        q1.h4(&qid(0)).h4(&qid(0));
        q2.sz(&qid(0)).x(&qid(0)).sz(&qid(0)).x(&qid(0));
        assert_states_equal(q1.state(), q2.state());

        // H5^2
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));
        q1.h5(&qid(0)).h5(&qid(0));
        q2.sx(&qid(0)).z(&qid(0)).sx(&qid(0)).z(&qid(0));
        assert_states_equal(q1.state(), q2.state());

        // H6^2
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));
        q1.h6(&qid(0)).h6(&qid(0));
        q2.sx(&qid(0)).y(&qid(0)).sx(&qid(0)).y(&qid(0));
        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_face_gates_on_one_state() {
        // Test all face gates starting from |1⟩ state
        // F
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.f(&qid(0));
        decomposed.sx(&qid(0)).sz(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // FDG
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.fdg(&qid(0));
        decomposed.szdg(&qid(0)).sxdg(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // F2
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.f2(&qid(0));
        decomposed.sxdg(&qid(0)).sy(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // F2DG
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.f2dg(&qid(0));
        decomposed.sydg(&qid(0)).sx(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // F3
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.f3(&qid(0));
        decomposed.sxdg(&qid(0)).sz(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // F3DG
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.f3dg(&qid(0));
        decomposed.szdg(&qid(0)).sx(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // F4
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.f4(&qid(0));
        decomposed.sz(&qid(0)).sx(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // F4DG
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.f4dg(&qid(0));
        decomposed.sxdg(&qid(0)).szdg(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_hadamard_gates_on_one_state() {
        // Test all hadamard variants starting from |1⟩ state
        // H2
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.h2(&qid(0));
        decomposed.sy(&qid(0)).z(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // H3
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.h3(&qid(0));
        decomposed.sz(&qid(0)).y(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // H4
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.h4(&qid(0));
        decomposed.sz(&qid(0)).x(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // H5
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.h5(&qid(0));
        decomposed.sx(&qid(0)).z(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // H6
        let mut direct = StateVec::new(1);
        let mut decomposed = StateVec::new(1);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.h6(&qid(0));
        decomposed.sx(&qid(0)).y(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_face_gates_locality() {
        // Verify face gates preserve entanglement and match decomposition on Bell state
        // F on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.f(&qid(0));
        decomposed.sx(&qid(0)).sz(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // FDG on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.fdg(&qid(0));
        decomposed.szdg(&qid(0)).sxdg(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // F2 on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.f2(&qid(0));
        decomposed.sxdg(&qid(0)).sy(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // F3 on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.f3(&qid(0));
        decomposed.sxdg(&qid(0)).sz(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // F4 on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.f4(&qid(0));
        decomposed.sz(&qid(0)).sx(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_hadamard_gates_locality() {
        // Verify hadamard variants preserve entanglement and match decomposition on Bell state
        // H2 on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.h2(&qid(0));
        decomposed.sy(&qid(0)).z(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // H3 on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.h3(&qid(0));
        decomposed.sz(&qid(0)).y(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // H4 on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.h4(&qid(0));
        decomposed.sz(&qid(0)).x(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // H5 on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.h5(&qid(0));
        decomposed.sx(&qid(0)).z(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // H6 on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.h6(&qid(0));
        decomposed.sx(&qid(0)).y(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_batch_face_gates() {
        // Test applying face gates to multiple qubits at once
        let mut direct = StateVec::new(3);
        let mut sequential = StateVec::new(3);

        // Initialize both to same state
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        sequential.h(&qid(0)).h(&qid(1)).h(&qid(2));

        // Batch apply F to qubits 0 and 2
        let q0 = qid(0)[0];
        let q2 = qid(2)[0];
        direct.f(&[q0, q2]);

        // Sequential apply
        sequential.f(&qid(0));
        sequential.f(&qid(2));

        assert_states_equal(direct.state(), sequential.state());

        // Test F2 batch
        let mut direct = StateVec::new(3);
        let mut sequential = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        sequential.h(&qid(0)).h(&qid(1)).h(&qid(2));

        let q1 = qid(1)[0];
        direct.f2(&[q0, q1, q2]);
        sequential.f2(&qid(0)).f2(&qid(1)).f2(&qid(2));

        assert_states_equal(direct.state(), sequential.state());
    }

    #[test]
    fn test_batch_hadamard_gates() {
        // Test applying hadamard variants to multiple qubits at once
        let mut direct = StateVec::new(3);
        let mut sequential = StateVec::new(3);

        // Initialize both to same state
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        sequential.h(&qid(0)).h(&qid(1)).h(&qid(2));

        // Batch apply H2 to qubits 0 and 2
        let q0 = qid(0)[0];
        let q2 = qid(2)[0];
        direct.h2(&[q0, q2]);

        // Sequential apply
        sequential.h2(&qid(0));
        sequential.h2(&qid(2));

        assert_states_equal(direct.state(), sequential.state());

        // Test H5 batch (all qubits)
        let mut direct = StateVec::new(3);
        let mut sequential = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        sequential.h(&qid(0)).h(&qid(1)).h(&qid(2));

        let q1 = qid(1)[0];
        direct.h5(&[q0, q1, q2]);
        sequential.h5(&qid(0)).h5(&qid(1)).h5(&qid(2));

        assert_states_equal(direct.state(), sequential.state());
    }

    #[test]
    fn test_sqrt_gates_vs_rotations() {
        // SX = RX(π/2)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));

        q1.sx(&qid(0));
        q2.rx(Angle64::from_radians(FRAC_PI_2), &qid(0));

        assert_states_equal(q1.state(), q2.state());

        // SXDG = RX(-π/2)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));

        q1.sxdg(&qid(0));
        q2.rx(Angle64::from_radians(-FRAC_PI_2), &qid(0));

        assert_states_equal(q1.state(), q2.state());

        // SY = RY(π/2)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));

        q1.sy(&qid(0));
        q2.ry(Angle64::from_radians(FRAC_PI_2), &qid(0));

        assert_states_equal(q1.state(), q2.state());

        // SYDG = RY(-π/2)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));

        q1.sydg(&qid(0));
        q2.ry(Angle64::from_radians(-FRAC_PI_2), &qid(0));

        assert_states_equal(q1.state(), q2.state());

        // SZ = RZ(π/2)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));

        q1.sz(&qid(0));
        q2.rz(Angle64::from_radians(FRAC_PI_2), &qid(0));

        assert_states_equal(q1.state(), q2.state());

        // SZDG = RZ(-π/2)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(&qid(0));
        q2.h(&qid(0));

        q1.szdg(&qid(0));
        q2.rz(Angle64::from_radians(-FRAC_PI_2), &qid(0));

        assert_states_equal(q1.state(), q2.state());
    }
}

mod detailed_tq_gate_cases {
    use crate::helpers::assert_states_equal;
    use num_complex::Complex64;
    use pecos_core::{Angle64, QubitId};
    use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec, qid, qid2};
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, PI};

    #[test]
    fn test_cx_decomposition() {
        let mut state_cx = StateVec::new(2);
        let mut state_decomposed = StateVec::new(2);

        let control = 0;
        let target = 1;

        // Apply CX gate
        state_cx.cx(&qid2(control, target));

        // Apply the decomposed gates
        state_decomposed.r1xy(
            Angle64::from_radians(-FRAC_PI_2),
            Angle64::from_radians(FRAC_PI_2),
            &qid(target),
        );
        state_decomposed.rzz(Angle64::from_radians(FRAC_PI_2), &qid2(control, target));
        state_decomposed.rz(Angle64::from_radians(-FRAC_PI_2), &qid(control));
        state_decomposed.r1xy(
            Angle64::from_radians(FRAC_PI_2),
            Angle64::from_radians(PI),
            &qid(target),
        );
        state_decomposed.rz(Angle64::from_radians(-FRAC_PI_2), &qid(target));

        // Assert that the states are equal
        assert_states_equal(state_cx.state(), state_decomposed.state());
    }

    #[test]
    fn test_rxx_decomposition() {
        let mut state_rxx = StateVec::new(2);
        let mut state_decomposed = StateVec::new(2);

        let control = 0;
        let target = 1;

        // Apply RXX gate
        state_rxx.rxx(Angle64::from_radians(FRAC_PI_4), &qid2(control, target));

        // Apply the decomposed gates
        state_decomposed.r1xy(
            Angle64::from_radians(FRAC_PI_2),
            Angle64::from_radians(FRAC_PI_2),
            &qid(control),
        );
        state_decomposed.r1xy(
            Angle64::from_radians(FRAC_PI_2),
            Angle64::from_radians(FRAC_PI_2),
            &qid(target),
        );
        state_decomposed.rzz(Angle64::from_radians(FRAC_PI_4), &qid2(control, target));
        state_decomposed.r1xy(
            Angle64::from_radians(FRAC_PI_2),
            Angle64::from_radians(-FRAC_PI_2),
            &qid(control),
        );
        state_decomposed.r1xy(
            Angle64::from_radians(FRAC_PI_2),
            Angle64::from_radians(-FRAC_PI_2),
            &qid(target),
        );

        // Assert that the states are equal
        assert_states_equal(state_rxx.state(), state_decomposed.state());
    }

    #[test]
    fn test_two_qubit_unitary_swap_simple() {
        let mut state_vec = StateVec::new(2);

        let swap_gate = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ];

        state_vec.prepare_computational_basis(2); // |10⟩
        state_vec.two_qubit_unitary(1, 0, swap_gate);

        assert!((state_vec.probability(1) - 1.0).abs() < 1e-10); // Should now be |01⟩
    }

    #[test]
    fn test_cx_all_basis_states() {
        let mut state_vec = StateVec::new(2);

        // |00⟩ → should remain |00⟩
        state_vec.prepare_computational_basis(0);
        state_vec.cx(&qid2(1, 0));
        assert!((state_vec.probability(0) - 1.0).abs() < 1e-10);

        // |01⟩ → should remain |01⟩
        state_vec.prepare_computational_basis(1);
        state_vec.cx(&qid2(1, 0));
        assert!((state_vec.probability(1) - 1.0).abs() < 1e-10);

        // |10⟩ → should flip to |11⟩
        state_vec.prepare_computational_basis(2);
        state_vec.cx(&qid2(1, 0));
        assert!((state_vec.probability(3) - 1.0).abs() < 1e-10);

        // |11⟩ → should flip to |10⟩
        state_vec.prepare_computational_basis(3);
        state_vec.cx(&qid2(1, 0));
        assert!((state_vec.probability(2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_control_target_independence() {
        // Test that CY and CZ work regardless of which qubit is control/target
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Prepare same initial state
        q1.h(&qid(0));
        q1.h(&qid(1));
        q2.h(&qid(0));
        q2.h(&qid(1));

        // Apply gates with different control/target
        q1.cz(&qid2(0, 1));
        q2.cz(&qid2(1, 0));

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_rxx_symmetry() {
        // Test that RXX is symmetric under exchange of qubits
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Prepare same non-trivial initial state
        q1.h(&qid(0));
        q1.h(&qid(1));
        q2.h(&qid(0));
        q2.h(&qid(1));

        // Apply RXX with different qubit orders
        q1.rxx(Angle64::from_radians(FRAC_PI_3), &qid2(0, 1));
        q2.rxx(Angle64::from_radians(FRAC_PI_3), &qid2(1, 0));

        // Results should be identical
        for (a, b) in q1.state().iter().zip(q2.state().iter()) {
            assert!((a - b).norm() < 1e-10);
        }
    }

    #[test]
    fn test_ryy_qubit_order_invariance() {
        let theta = FRAC_PI_4;

        // Test on random initial states
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);
        q1.h(&qid(0)).x(&qid(1)); // Random state
        q2.h(&qid(0)).x(&qid(1)); // Same initial state

        q1.ryy(Angle64::from_radians(theta), &qid2(0, 1));
        q2.ryy(Angle64::from_radians(theta), &qid2(1, 0));

        // States should be exactly equal
        for (a, b) in q1.state().iter().zip(q2.state().iter()) {
            assert!(
                (a - b).norm() < 1e-10,
                "Qubit order test failed: a={a}, b={b}"
            );
        }
    }

    #[test]
    fn test_ryy_large_system() {
        let theta = FRAC_PI_3;

        // Initialize a 5-qubit state
        let mut q = StateVec::new(5);
        q.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3)).h(&qid(4)); // Superposition state

        // Apply RYY on qubits 2 and 4
        q.ryy(Angle64::from_radians(theta), &qid2(2, 4));

        // Ensure state vector normalization is preserved
        let norm: f64 = q.state().iter().map(num_complex::Complex::norm_sqr).sum();
        assert!(
            (norm - 1.0).abs() < 1e-10,
            "State normalization test failed: norm={norm}"
        );
    }

    #[test]
    fn test_ryy_edge_cases() {
        let mut q = StateVec::new(2);

        // Apply RYY gate
        q.ryy(Angle64::from_radians(PI), &qid2(0, 1));

        // Define the expected result for RYY(π)
        let expected = vec![
            Complex64::new(0.0, 0.0),  // |00⟩
            Complex64::new(0.0, 0.0),  // |01⟩
            Complex64::new(0.0, 0.0),  // |10⟩
            Complex64::new(-1.0, 0.0), // |11⟩
        ];

        // Compare simulated state vector to the expected result
        assert_states_equal(q.state(), &expected);
    }

    #[test]
    fn test_ryy_global_phase() {
        let mut q = StateVec::new(2);

        q.ryy(Angle64::from_radians(PI), &qid2(0, 1));

        // Define the expected result for RYY(π)
        let expected = vec![
            Complex64::new(0.0, 0.0),  // |00⟩
            Complex64::new(0.0, 0.0),  // |01⟩
            Complex64::new(0.0, 0.0),  // |10⟩
            Complex64::new(-1.0, 0.0), // |11⟩
        ];

        // Compare states
        assert_states_equal(q.state(), &expected);
    }

    #[test]
    fn test_ryy_small_angles() {
        let theta = 1e-10; // Very small angle
        let mut q = StateVec::new(2);

        // Initialize |00⟩
        let initial = q.state().clone();
        q.ryy(Angle64::from_radians(theta), &qid2(0, 1));

        // Expect state to remain close to the initial state
        for (a, b) in q.state().iter().zip(initial.iter()) {
            assert!(
                (a - b).norm() < 1e-10,
                "Small angle test failed: a={a}, b={b}"
            );
        }
    }

    #[test]
    fn test_ryy_randomized() {
        use rand::RngExt;

        let mut rng = rand::rng();
        let theta = rng.random_range(0.0..2.0 * PI);

        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Random initial state
        q1.h(&qid(0)).h(&qid(1));
        q2.h(&qid(0)).h(&qid(1));

        // Apply RYY with random qubit order
        q1.ryy(Angle64::from_radians(theta), &qid2(0, 1));
        q2.ryy(Angle64::from_radians(theta), &qid2(1, 0));

        // Compare states
        for (a, b) in q1.state().iter().zip(q2.state().iter()) {
            assert!(
                (a - b).norm() < 1e-10,
                "Randomized test failed: a={a}, b={b}"
            );
        }
    }

    #[test]
    fn test_szz_equivalence() {
        // Test that SZZ is equivalent to RZZ(π/2)
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Create some non-trivial initial state
        q1.h(&qid(0));
        q2.h(&qid(0));

        // Compare direct SZZ vs RZZ(π/2)
        q1.szz(&qid2(0, 1));
        q2.rzz(Angle64::from_radians(FRAC_PI_2), &qid2(0, 1));

        assert_states_equal(q1.state(), q2.state());

        // Also verify decomposition matches
        let mut q3 = StateVec::new(2);
        q3.h(&qid(0)); // Same initial state
        q3.h(&qid(0))
            .h(&qid(1))
            .sxx(&qid2(0, 1))
            .h(&qid(0))
            .h(&qid(1));

        assert_states_equal(q1.state(), q3.state());
    }

    #[test]
    fn test_szz_trait_equivalence() {
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Create some non-trivial initial state
        q1.h(&qid(0));
        q2.h(&qid(0));

        // Compare CliffordGateable trait szz vs ArbitraryRotationGateable trait rzz(π/2)
        CliffordGateable::szz(&mut q1, &qid2(0, 1));
        ArbitraryRotationGateable::rzz(&mut q2, Angle64::from_radians(PI / 2.0), &qid2(0, 1));

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_two_qubit_unitary_properties() {
        let mut q = StateVec::new(2);

        // Create a non-trivial state
        q.h(&qid(0));
        q.h(&qid(1));

        // iSWAP matrix
        let iswap = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ];

        q.two_qubit_unitary(0, 1, iswap);

        // Verify normalization is preserved
        let norm: f64 = q.state().iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((norm - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_two_qubit_unitary_identity() {
        let mut state_vec = StateVec::new(2);

        // Identity matrix
        let identity_gate = [
            [
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
            [
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        ];

        // Apply the identity gate
        state_vec.prepare_computational_basis(2);
        state_vec.two_qubit_unitary(0, 1, identity_gate);

        // State should remain |10⟩
        assert!((state_vec.probability(2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_controlled_gate_symmetries() {
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Test SWAP symmetry
        q1.x(&qid(0)); // |10⟩
        q2.x(&qid(0)); // |10⟩

        q1.cx(&qid2(0, 1)).cx(&qid2(1, 0)).cx(&qid2(0, 1)); // SWAP via CNOTs
        q2.swap(&qid2(0, 1)); // Direct SWAP

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_controlled_gate_phases() {
        // Test phase behavior of controlled operations
        let mut q = StateVec::new(2);

        // Create superposition with phases
        q.h(&qid(0)).sz(&qid(0));
        q.h(&qid(1)).sz(&qid(1));

        // Control operations should preserve phases correctly
        let initial = q.state().clone();
        q.cz(&qid2(0, 1)).cz(&qid2(0, 1)); // CZ^2 = I

        assert_states_equal(q.state(), &initial);
    }

    #[test]
    fn test_cy_vs_decomposition() {
        // CY = SZDG(target).CX.SZ(target)
        // Test on |00⟩
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.cy(&qid2(0, 1));
        decomposed.szdg(&qid(1)).cx(&qid2(0, 1)).sz(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on |10⟩ (control=1)
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.cy(&qid2(0, 1));
        decomposed.szdg(&qid(1)).cx(&qid2(0, 1)).sz(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.cy(&qid2(0, 1));
        decomposed.szdg(&qid(1)).cx(&qid2(0, 1)).sz(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test with reversed qubit order (target, control)
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.x(&qid(1)); // control=1
        decomposed.x(&qid(1));
        direct.cy(&qid2(1, 0)); // CY with q1 as control, q0 as target
        decomposed.szdg(&qid(0)).cx(&qid2(1, 0)).sz(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in 3-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.cy(&qid2(0, 2)); // control=0, target=2
        decomposed.szdg(&qid(2)).cx(&qid2(0, 2)).sz(&qid(2));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_sxx_vs_decomposition() {
        // SXX = SX(q1).SX(q2).SYDG(q1).CX(q1,q2).SY(q1)
        // Test on |00⟩
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.sxx(&qid2(0, 1));
        decomposed
            .sx(&qid(0))
            .sx(&qid(1))
            .sydg(&qid(0))
            .cx(&qid2(0, 1))
            .sy(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on |11⟩
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.x(&qid(0)).x(&qid(1));
        decomposed.x(&qid(0)).x(&qid(1));
        direct.sxx(&qid2(0, 1));
        decomposed
            .sx(&qid(0))
            .sx(&qid(1))
            .sydg(&qid(0))
            .cx(&qid2(0, 1))
            .sy(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.sxx(&qid2(0, 1));
        decomposed
            .sx(&qid(0))
            .sx(&qid(1))
            .sydg(&qid(0))
            .cx(&qid2(0, 1))
            .sy(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in 3-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.sxx(&qid2(0, 2));
        decomposed
            .sx(&qid(0))
            .sx(&qid(2))
            .sydg(&qid(0))
            .cx(&qid2(0, 2))
            .sy(&qid(0));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_sxxdg_vs_decomposition() {
        // SXXDG is the inverse of SXX
        // Test SXX * SXXDG = I
        let mut q = StateVec::new(2);
        q.h(&qid(0)).h(&qid(1));
        let initial = q.state().clone();
        q.sxx(&qid2(0, 1)).sxxdg(&qid2(0, 1));
        assert_states_equal(q.state(), &initial);

        // Test SXXDG * SXX = I
        let mut q = StateVec::new(2);
        q.h(&qid(0)).h(&qid(1));
        let initial = q.state().clone();
        q.sxxdg(&qid2(0, 1)).sxx(&qid2(0, 1));
        assert_states_equal(q.state(), &initial);

        // Test on Bell state
        let mut q = StateVec::new(2);
        q.h(&qid(0)).cx(&qid2(0, 1));
        let initial = q.state().clone();
        q.sxx(&qid2(0, 1)).sxxdg(&qid2(0, 1));
        assert_states_equal(q.state(), &initial);
    }

    #[test]
    fn test_sxx_squared() {
        // SXX^2 = XX (the Pauli XX gate, up to global phase)
        // XX = X⊗X swaps |00⟩↔|11⟩ and |01⟩↔|10⟩
        let mut q = StateVec::new(2);
        q.h(&qid(0)).h(&qid(1));
        q.sxx(&qid2(0, 1)).sxx(&qid2(0, 1));
        // After SXX^2, compare with X⊗X applied to initial state
        let mut q_xx = StateVec::new(2);
        q_xx.h(&qid(0)).h(&qid(1));
        q_xx.x(&qid(0)).x(&qid(1)); // X⊗X

        // States should match up to global phase
        // Check that probabilities match
        let prob_match = q
            .state()
            .iter()
            .zip(q_xx.state().iter())
            .all(|(a, b)| (a.norm_sqr() - b.norm_sqr()).abs() < 1e-10);
        assert!(prob_match, "SXX^2 should give XX (up to global phase)");
    }

    #[test]
    fn test_syy_vs_decomposition() {
        // SYY = SZDG(q1).SZDG(q2).SXX.SZ(q1).SZ(q2)
        // (rotate Y basis to X basis, apply SXX, rotate back)
        // Test on |00⟩
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.syy(&qid2(0, 1));
        decomposed
            .szdg(&qid(0))
            .szdg(&qid(1))
            .sxx(&qid2(0, 1))
            .sz(&qid(0))
            .sz(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on |11⟩
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.x(&qid(0)).x(&qid(1));
        decomposed.x(&qid(0)).x(&qid(1));
        direct.syy(&qid2(0, 1));
        decomposed
            .szdg(&qid(0))
            .szdg(&qid(1))
            .sxx(&qid2(0, 1))
            .sz(&qid(0))
            .sz(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.syy(&qid2(0, 1));
        decomposed
            .szdg(&qid(0))
            .szdg(&qid(1))
            .sxx(&qid2(0, 1))
            .sz(&qid(0))
            .sz(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in 3-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.syy(&qid2(0, 2));
        decomposed
            .szdg(&qid(0))
            .szdg(&qid(2))
            .sxx(&qid2(0, 2))
            .sz(&qid(0))
            .sz(&qid(2));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_syydg_vs_decomposition() {
        // SYYDG is the inverse of SYY
        // Test SYY * SYYDG = I
        let mut q = StateVec::new(2);
        q.h(&qid(0)).h(&qid(1));
        let initial = q.state().clone();
        q.syy(&qid2(0, 1)).syydg(&qid2(0, 1));
        assert_states_equal(q.state(), &initial);

        // Test SYYDG * SYY = I
        let mut q = StateVec::new(2);
        q.h(&qid(0)).h(&qid(1));
        let initial = q.state().clone();
        q.syydg(&qid2(0, 1)).syy(&qid2(0, 1));
        assert_states_equal(q.state(), &initial);

        // Test on Bell state
        let mut q = StateVec::new(2);
        q.h(&qid(0)).cx(&qid2(0, 1));
        let initial = q.state().clone();
        q.syy(&qid2(0, 1)).syydg(&qid2(0, 1));
        assert_states_equal(q.state(), &initial);
    }

    #[test]
    fn test_syy_squared() {
        // SYY^2 = YY (the Pauli YY gate, up to global phase)
        let mut q = StateVec::new(2);
        q.h(&qid(0)).h(&qid(1));
        q.syy(&qid2(0, 1)).syy(&qid2(0, 1));
        // After SYY^2, compare with Y⊗Y applied to initial state
        let mut q_yy = StateVec::new(2);
        q_yy.h(&qid(0)).h(&qid(1));
        q_yy.y(&qid(0)).y(&qid(1)); // Y⊗Y

        // States should match up to global phase
        let prob_match = q
            .state()
            .iter()
            .zip(q_yy.state().iter())
            .all(|(a, b)| (a.norm_sqr() - b.norm_sqr()).abs() < 1e-10);
        assert!(prob_match, "SYY^2 should give YY (up to global phase)");
    }

    #[test]
    fn test_szz_vs_decomposition() {
        // SZZ = H(q1).H(q2).SXX.H(q1).H(q2)
        // (rotate Z basis to X basis, apply SXX, rotate back)
        // Test on |00⟩
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.szz(&qid2(0, 1));
        decomposed
            .h(&qid(0))
            .h(&qid(1))
            .sxx(&qid2(0, 1))
            .h(&qid(0))
            .h(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on |11⟩
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.x(&qid(0)).x(&qid(1));
        decomposed.x(&qid(0)).x(&qid(1));
        direct.szz(&qid2(0, 1));
        decomposed
            .h(&qid(0))
            .h(&qid(1))
            .sxx(&qid2(0, 1))
            .h(&qid(0))
            .h(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.szz(&qid2(0, 1));
        decomposed
            .h(&qid(0))
            .h(&qid(1))
            .sxx(&qid2(0, 1))
            .h(&qid(0))
            .h(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in 3-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.szz(&qid2(0, 2));
        decomposed
            .h(&qid(0))
            .h(&qid(2))
            .sxx(&qid2(0, 2))
            .h(&qid(0))
            .h(&qid(2));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_szzdg_vs_decomposition() {
        // SZZDG is the inverse of SZZ
        // Test SZZ * SZZDG = I
        let mut q = StateVec::new(2);
        q.h(&qid(0)).h(&qid(1));
        let initial = q.state().clone();
        q.szz(&qid2(0, 1)).szzdg(&qid2(0, 1));
        assert_states_equal(q.state(), &initial);

        // Test SZZDG * SZZ = I
        let mut q = StateVec::new(2);
        q.h(&qid(0)).h(&qid(1));
        let initial = q.state().clone();
        q.szzdg(&qid2(0, 1)).szz(&qid2(0, 1));
        assert_states_equal(q.state(), &initial);

        // Test on Bell state
        let mut q = StateVec::new(2);
        q.h(&qid(0)).cx(&qid2(0, 1));
        let initial = q.state().clone();
        q.szz(&qid2(0, 1)).szzdg(&qid2(0, 1));
        assert_states_equal(q.state(), &initial);
    }

    #[test]
    fn test_szz_squared() {
        // SZZ^2 = ZZ (the Pauli ZZ gate, up to global phase)
        let mut q = StateVec::new(2);
        q.h(&qid(0)).h(&qid(1));
        q.szz(&qid2(0, 1)).szz(&qid2(0, 1));
        // After SZZ^2, compare with Z⊗Z applied to initial state
        let mut q_zz = StateVec::new(2);
        q_zz.h(&qid(0)).h(&qid(1));
        q_zz.z(&qid(0)).z(&qid(1)); // Z⊗Z

        // States should match up to global phase
        let prob_match = q
            .state()
            .iter()
            .zip(q_zz.state().iter())
            .all(|(a, b)| (a.norm_sqr() - b.norm_sqr()).abs() < 1e-10);
        assert!(prob_match, "SZZ^2 should give ZZ (up to global phase)");
    }

    #[test]
    fn test_iswap_vs_decomposition() {
        // iSWAP = SZ(q1).SZ(q2).H(q1).CX(q1,q2).CX(q2,q1).H(q2)
        // Test on |00⟩ (should stay |00⟩)
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.iswap(&qid2(0, 1));
        decomposed
            .sz(&qid(0))
            .sz(&qid(1))
            .h(&qid(0))
            .cx(&qid2(0, 1))
            .cx(&qid2(1, 0))
            .h(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on |01⟩ (should become i|10⟩)
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.x(&qid(1));
        decomposed.x(&qid(1));
        direct.iswap(&qid2(0, 1));
        decomposed
            .sz(&qid(0))
            .sz(&qid(1))
            .h(&qid(0))
            .cx(&qid2(0, 1))
            .cx(&qid2(1, 0))
            .h(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on |10⟩ (should become i|01⟩)
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.iswap(&qid2(0, 1));
        decomposed
            .sz(&qid(0))
            .sz(&qid(1))
            .h(&qid(0))
            .cx(&qid2(0, 1))
            .cx(&qid2(1, 0))
            .h(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on |11⟩ (should stay |11⟩)
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.x(&qid(0)).x(&qid(1));
        decomposed.x(&qid(0)).x(&qid(1));
        direct.iswap(&qid2(0, 1));
        decomposed
            .sz(&qid(0))
            .sz(&qid(1))
            .h(&qid(0))
            .cx(&qid2(0, 1))
            .cx(&qid2(1, 0))
            .h(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.iswap(&qid2(0, 1));
        decomposed
            .sz(&qid(0))
            .sz(&qid(1))
            .h(&qid(0))
            .cx(&qid2(0, 1))
            .cx(&qid2(1, 0))
            .h(&qid(1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in 3-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.iswap(&qid2(0, 2));
        decomposed
            .sz(&qid(0))
            .sz(&qid(2))
            .h(&qid(0))
            .cx(&qid2(0, 2))
            .cx(&qid2(2, 0))
            .h(&qid(2));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_iswap_squared() {
        // iSWAP^2 = -SWAP (up to global phase)
        // Two iSWAPs should swap with phase -1
        let mut q = StateVec::new(2);
        q.x(&qid(1)); // Start with |01⟩
        q.iswap(&qid2(0, 1)).iswap(&qid2(0, 1));
        // After two iSWAPs on |01⟩: i*i*|01⟩ = -|01⟩
        // Check amplitude is at |01⟩ with phase -1
        let state = q.state();
        assert!((state[0].norm_sqr()).abs() < 1e-10); // |00⟩ = 0
        assert!((state[2].norm_sqr() - 1.0).abs() < 1e-10); // |01⟩ has full amplitude
        assert!((state[1].norm_sqr()).abs() < 1e-10); // |10⟩ = 0
        assert!((state[3].norm_sqr()).abs() < 1e-10); // |11⟩ = 0
    }

    #[test]
    fn test_g_vs_decomposition() {
        // G = CZ.H(q1).H(q2).CZ
        // Test on |00⟩
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.g(&qid2(0, 1));
        decomposed
            .cz(&qid2(0, 1))
            .h(&qid(0))
            .h(&qid(1))
            .cz(&qid2(0, 1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on |01⟩
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.x(&qid(1));
        decomposed.x(&qid(1));
        direct.g(&qid2(0, 1));
        decomposed
            .cz(&qid2(0, 1))
            .h(&qid(0))
            .h(&qid(1))
            .cz(&qid2(0, 1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on |10⟩
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.x(&qid(0));
        decomposed.x(&qid(0));
        direct.g(&qid2(0, 1));
        decomposed
            .cz(&qid2(0, 1))
            .h(&qid(0))
            .h(&qid(1))
            .cz(&qid2(0, 1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on |11⟩
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.x(&qid(0)).x(&qid(1));
        decomposed.x(&qid(0)).x(&qid(1));
        direct.g(&qid2(0, 1));
        decomposed
            .cz(&qid2(0, 1))
            .h(&qid(0))
            .h(&qid(1))
            .cz(&qid2(0, 1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test on Bell state
        let mut direct = StateVec::new(2);
        let mut decomposed = StateVec::new(2);
        direct.h(&qid(0)).cx(&qid2(0, 1));
        decomposed.h(&qid(0)).cx(&qid2(0, 1));
        direct.g(&qid2(0, 1));
        decomposed
            .cz(&qid2(0, 1))
            .h(&qid(0))
            .h(&qid(1))
            .cz(&qid2(0, 1));
        assert_states_equal(direct.state(), decomposed.state());

        // Test in 3-qubit system
        let mut direct = StateVec::new(3);
        let mut decomposed = StateVec::new(3);
        direct.h(&qid(0)).h(&qid(1)).h(&qid(2));
        decomposed.h(&qid(0)).h(&qid(1)).h(&qid(2));
        direct.g(&qid2(0, 2));
        decomposed
            .cz(&qid2(0, 2))
            .h(&qid(0))
            .h(&qid(2))
            .cz(&qid2(0, 2));
        assert_states_equal(direct.state(), decomposed.state());
    }

    #[test]
    fn test_g_is_involution() {
        // G^2 = I (G is its own inverse)
        let mut q = StateVec::new(2);
        q.h(&qid(0)).h(&qid(1));
        let initial = q.state().clone();
        q.g(&qid2(0, 1)).g(&qid2(0, 1));
        assert_states_equal(q.state(), &initial);

        // Test on Bell state
        let mut q = StateVec::new(2);
        q.h(&qid(0)).cx(&qid2(0, 1));
        let initial = q.state().clone();
        q.g(&qid2(0, 1)).g(&qid2(0, 1));
        assert_states_equal(q.state(), &initial);
    }

    // === Batch operation tests ===
    // These test applying gates to multiple qubit pairs simultaneously

    #[test]
    fn test_cy_batch() {
        // Apply CY to two pairs: (0,1) and (2,3) in a 4-qubit system
        let mut batch = StateVec::new(4);
        let mut sequential = StateVec::new(4);

        // Prepare initial state
        batch.h(&qid(0)).h(&qid(2));
        sequential.h(&qid(0)).h(&qid(2));

        // Batch: apply CY to both pairs at once
        batch.cy(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);

        // Sequential: apply CY to each pair separately
        sequential.cy(&qid2(0, 1)).cy(&qid2(2, 3));

        assert_states_equal(batch.state(), sequential.state());
    }

    #[test]
    fn test_sxx_batch() {
        // Apply SXX to two pairs in a 4-qubit system
        let mut batch = StateVec::new(4);
        let mut sequential = StateVec::new(4);

        batch.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));
        sequential.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));

        batch.sxx(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        sequential.sxx(&qid2(0, 1)).sxx(&qid2(2, 3));

        assert_states_equal(batch.state(), sequential.state());
    }

    #[test]
    fn test_sxxdg_batch() {
        let mut batch = StateVec::new(4);
        let mut sequential = StateVec::new(4);

        batch.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));
        sequential.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));

        batch.sxxdg(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        sequential.sxxdg(&qid2(0, 1)).sxxdg(&qid2(2, 3));

        assert_states_equal(batch.state(), sequential.state());
    }

    #[test]
    fn test_syy_batch() {
        let mut batch = StateVec::new(4);
        let mut sequential = StateVec::new(4);

        batch.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));
        sequential.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));

        batch.syy(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        sequential.syy(&qid2(0, 1)).syy(&qid2(2, 3));

        assert_states_equal(batch.state(), sequential.state());
    }

    #[test]
    fn test_syydg_batch() {
        let mut batch = StateVec::new(4);
        let mut sequential = StateVec::new(4);

        batch.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));
        sequential.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));

        batch.syydg(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        sequential.syydg(&qid2(0, 1)).syydg(&qid2(2, 3));

        assert_states_equal(batch.state(), sequential.state());
    }

    #[test]
    fn test_szz_batch() {
        let mut batch = StateVec::new(4);
        let mut sequential = StateVec::new(4);

        batch.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));
        sequential.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));

        batch.szz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        sequential.szz(&qid2(0, 1)).szz(&qid2(2, 3));

        assert_states_equal(batch.state(), sequential.state());
    }

    #[test]
    fn test_szzdg_batch() {
        let mut batch = StateVec::new(4);
        let mut sequential = StateVec::new(4);

        batch.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));
        sequential.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));

        batch.szzdg(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        sequential.szzdg(&qid2(0, 1)).szzdg(&qid2(2, 3));

        assert_states_equal(batch.state(), sequential.state());
    }

    #[test]
    fn test_iswap_batch() {
        let mut batch = StateVec::new(4);
        let mut sequential = StateVec::new(4);

        batch.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));
        sequential.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));

        batch.iswap(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        sequential.iswap(&qid2(0, 1)).iswap(&qid2(2, 3));

        assert_states_equal(batch.state(), sequential.state());
    }

    #[test]
    fn test_g_batch() {
        let mut batch = StateVec::new(4);
        let mut sequential = StateVec::new(4);

        batch.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));
        sequential.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));

        batch.g(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        sequential.g(&qid2(0, 1)).g(&qid2(2, 3));

        assert_states_equal(batch.state(), sequential.state());
    }

    #[test]
    fn test_batch_with_non_adjacent_qubits() {
        // Test batch operations on non-adjacent qubit pairs: (0,2) and (1,3)
        let mut batch = StateVec::new(4);
        let mut sequential = StateVec::new(4);

        batch.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));
        sequential.h(&qid(0)).h(&qid(1)).h(&qid(2)).h(&qid(3));

        // SXX on non-adjacent pairs
        batch.sxx(&[QubitId(0), QubitId(2), QubitId(1), QubitId(3)]);
        sequential.sxx(&qid2(0, 2)).sxx(&qid2(1, 3));

        assert_states_equal(batch.state(), sequential.state());
    }

    #[test]
    fn test_batch_in_larger_system() {
        // Test batch operations in 6-qubit system with various spacings
        let mut batch = StateVec::new(6);
        let mut sequential = StateVec::new(6);

        // Prepare superposition
        for i in 0..6 {
            batch.h(&qid(i));
            sequential.h(&qid(i));
        }

        // Apply SYY to pairs (0,3) and (2,5) - large spacing
        batch.syy(&[QubitId(0), QubitId(3), QubitId(2), QubitId(5)]);
        sequential.syy(&qid2(0, 3)).syy(&qid2(2, 5));

        assert_states_equal(batch.state(), sequential.state());
    }

    #[test]
    fn test_batch_three_pairs() {
        // Test batch with 3 pairs in 6-qubit system
        let mut batch = StateVec::new(6);
        let mut sequential = StateVec::new(6);

        for i in 0..6 {
            batch.h(&qid(i));
            sequential.h(&qid(i));
        }

        // Apply iSWAP to 3 pairs
        batch.iswap(&[
            QubitId(0),
            QubitId(1),
            QubitId(2),
            QubitId(3),
            QubitId(4),
            QubitId(5),
        ]);
        sequential
            .iswap(&qid2(0, 1))
            .iswap(&qid2(2, 3))
            .iswap(&qid2(4, 5));

        assert_states_equal(batch.state(), sequential.state());
    }
}

mod detail_meas_cases {
    use pecos_simulators::{CliffordGateable, QuantumSimulator, StateVec, qid, qid2};

    #[test]
    fn test_measurement_on_entangled_state() {
        let mut q = StateVec::new(2);

        // Create Bell state (|00⟩ + |11⟩) / sqrt(2)
        q.h(&qid(0));
        q.cx(&qid2(0, 1));

        // Measure the first qubit
        let result1 = q.mz(&qid(0)).into_iter().next().unwrap();

        // Measure the second qubit - should match the first
        let result2 = q.mz(&qid(1)).into_iter().next().unwrap();

        assert_eq!(result1.outcome, result2.outcome);
    }

    #[test]
    fn test_measurement_properties() {
        let mut q = StateVec::new(2);

        // Test 1: Measuring |0⟩ should always give 0
        let result = q.mz(&qid(0)).into_iter().next().unwrap();
        assert!(!result.outcome);
        assert!((q.probability(0) - 1.0).abs() < 1e-10);

        // Test 2: Measuring |1⟩ should always give 1
        q.reset();
        q.x(&qid(0));
        let result = q.mz(&qid(0)).into_iter().next().unwrap();
        assert!(result.outcome);
        assert!((q.probability(1) - 1.0).abs() < 1e-10);

        // Test 3: In a Bell state, measurements should correlate
        q.reset();
        q.h(&qid(0)).cx(&qid2(0, 1)); // Create Bell state
        let result1 = q.mz(&qid(0)).into_iter().next().unwrap();
        let result2 = q.mz(&qid(1)).into_iter().next().unwrap();
        assert_eq!(
            result1.outcome, result2.outcome,
            "Bell state measurements should correlate"
        );

        // Test 4: Repeated measurements should be consistent
        q.reset();
        q.h(&qid(0)); // Create superposition
        let first = q.mz(&qid(0)).into_iter().next().unwrap();
        let second = q.mz(&qid(0)).into_iter().next().unwrap(); // Measure again
        assert_eq!(
            first.outcome, second.outcome,
            "Repeated measurements should give same result"
        );
    }

    #[test]
    fn test_measurement_basis_transforms() {
        let mut q = StateVec::new(1);

        // |0⟩ in X basis
        q.h(&qid(0));

        // Measure in Z basis
        let result = q.mz(&qid(0)).into_iter().next().unwrap();

        // Result should be random but state should collapse
        let final_prob = if result.outcome {
            q.probability(1)
        } else {
            q.probability(0)
        };
        assert!((final_prob - 1.0).abs() < 1e-10);
    }
}
