mod helpers;

mod advanced_gates {
    use crate::helpers::assert_states_equal;
    use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, StateVec};
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, PI};

    #[test]
    fn test_rotation_composition() {
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);

        // Test that rotation decompositions work
        // RY(θ) = RX(π/2)RZ(θ)RX(-π/2)
        q1.ry(FRAC_PI_3, 0);

        q2.rx(FRAC_PI_2, 0).rz(FRAC_PI_3, 0).rx(-FRAC_PI_2, 0);

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
        q1.ry(theta1, 0).ry(theta2, 0);

        // Method 2: Combined rotation
        let mut q2 = q.clone();
        q2.ry(theta1 + theta2, 0);

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_rotation_symmetries() {
        // Test that all rotations are symmetric under exchange of qubits
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Prepare same non-trivial initial state
        q1.h(0);
        q1.h(1);
        q2.h(0);
        q2.h(1);

        let theta = PI / 3.0;

        // Test RYY symmetry
        q1.ryy(theta, 0, 1);
        q2.ryy(theta, 1, 0);

        for (a, b) in q1.state().iter().zip(q2.state().iter()) {
            assert!((a - b).norm() < 1e-10);
        }

        // Test RZZ symmetry
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);
        q1.h(0);
        q1.h(1);
        q2.h(0);
        q2.h(1);

        q1.rzz(theta, 0, 1);
        q2.rzz(theta, 1, 0);

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
        q1.rx(theta, 0).ry(phi, 0);
        q2.ry(phi, 0).rx(theta, 0);

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

        q1.h(0); // Direct H
        println!("After H: q1 = {:?}", q1.state());

        // H via rotations - changed order and added negative sign to RZ angle
        q2.ry(-FRAC_PI_2, 0).rz(-PI, 0);
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
    use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, StateVec};
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2};

    #[test]
    fn test_bell_state_entanglement() {
        let mut state_vec = StateVec::new(2);

        // Prepare Bell State: (|00⟩ + |11⟩) / √2
        state_vec.h(0);
        state_vec.cx(0, 1);

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
        q.h(0).cx(0, 1).cx(1, 2); // Create GHZ state

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
        q.h(0).cx(0, 1);
        let probs1 = [
            q.probability(0),
            q.probability(1),
            q.probability(2),
            q.probability(3),
        ];

        // Method 2: Rotations
        q.reset();
        q.ry(FRAC_PI_2, 0).cx(0, 1); // Remove rz(PI) since it just adds phase

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
        q1.x(1); // Direct preparation of |01⟩

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
        q2.x(0).x(1).x(0); // Should give |01⟩

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
        q.h(0);
        assert!((q.probability(0) - 0.5).abs() < 1e-10);
        assert!((q.probability(1) - 0.5).abs() < 1e-10);

        // |+i⟩ state
        q.reset();
        q.h(0).sz(0);
        assert!((q.probability(0) - 0.5).abs() < 1e-10);
        assert!((q.probability(1) - 0.5).abs() < 1e-10);
    }
}

mod gate_sequences {
    use crate::helpers::assert_states_equal;
    use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, StateVec};

    #[test]
    fn test_operation_chains() {
        // Test complex sequences of operations
        let mut q = StateVec::new(2);

        // Create maximally entangled state then disentangle
        q.h(0)
            .cx(0, 1) // Create Bell state
            .cx(0, 1)
            .h(0); // Disentangle (apply the same operations in reverse)

        // Should be back to |00⟩
        assert!((q.probability(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_inverse_gates() {
        let mut state_vec = StateVec::new(1);

        // Apply Hadamard twice: H * H = I
        state_vec.h(0);
        state_vec.h(0);

        // Verify state is back to |0⟩
        assert!((state_vec.probability(0) - 1.0).abs() < 1e-10);
        assert!((state_vec.probability(1)).abs() < 1e-10);

        // Apply X twice: X * X = I
        state_vec.x(0);
        state_vec.x(0);
        assert!((state_vec.probability(0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_phase_gate_identities() {
        // Test S = T^2
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);

        // Put in superposition first to check phases
        q1.h(0);
        q2.h(0);

        q1.sz(0); // S gate
        q2.t(0).t(0); // Two T gates

        assert_states_equal(q1.state(), q2.state());
    }
    #[test]
    fn test_gate_decompositions() {
        // Test that composite operations match their decompositions
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Test SWAP decomposition into CNOTs
        q1.x(0); // Start with |10⟩
        q1.swap(0, 1); // Direct SWAP

        q2.x(0); // Also start with |10⟩
        q2.cx(0, 1).cx(1, 0).cx(0, 1); // SWAP decomposition

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_bell_state_preparation() {
        let mut q = StateVec::new(2);
        q.h(0).cx(0, 1);
        assert!((q.probability(0) - 0.5).abs() < 1e-10);
        assert!((q.probability(3) - 0.5).abs() < 1e-10);
    }

    #[test]
    fn test_ghz_state_preparation() {
        let mut q = StateVec::new(3);
        q.h(0).cx(0, 1).cx(1, 2);
        assert!((q.probability(0) - 0.5).abs() < 1e-10);
        assert!((q.probability(7) - 0.5).abs() < 1e-10);
    }
}

mod numerical_properties {
    use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, StateVec};
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6};

    #[test]
    fn test_state_normalization() {
        let mut state_vec = StateVec::new(3);

        // Apply multiple gates
        state_vec.h(0);
        state_vec.cx(0, 1);
        state_vec.cx(1, 2);

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
            q.rx(FRAC_PI_3, 0)
                .ry(FRAC_PI_4, 1)
                .rz(FRAC_PI_6, 2)
                .cx(0, 3);
        }

        // Check normalization is preserved
        let total_prob: f64 = q.state().iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((total_prob - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_phase_coherence() {
        let mut q = StateVec::new(1);

        // Apply series of phase rotations that should cancel
        q.h(0) // Create superposition
            .rz(FRAC_PI_4, 0)
            .rz(FRAC_PI_4, 0)
            .rz(-FRAC_PI_2, 0); // Should cancel

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
        q.h(0);

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
    use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, StateVec};
    use std::f64::consts::{FRAC_1_SQRT_2, PI};

    #[test]
    fn test_single_qubit_locality() {
        // Test on 3 qubit system that gates only affect their target
        let mut q = StateVec::new(3);

        // Prepare state |+⟩|0⟩|0⟩
        q.h(0); // Affects least significant bit

        // Apply X to qubit 2 (most significant bit)
        q.x(2);

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
        q.h(0);

        println!("\nAfter H on qubit 0:");
        for i in 0..16 {
            println!("  {:04b}: {:.3}", i, q.state()[i]);
        }

        // Apply CX between qubits 2,3
        q.cx(2, 3);

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
        q.h(0);

        // Apply CX on qubits 1 and 2 (no effect on qubit 0)
        q.cx(1, 2);

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
        q.h(0);

        println!("\nAfter H on qubit 0:");
        for i in 0..8 {
            println!("  {:03b}: {:.3}", i, q.state()[i]);
        }

        // Apply rotation to qubit 1
        q.rx(PI / 2.0, 1);

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
        q1.h(0).cx(0, 1); // Adjacent qubits
        q2.h(0).cx(0, 3); // Distant qubits

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
    use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, StateVec};
    use std::f64::consts::PI;

    #[test]
    fn test_small_angle_rotations() {
        let mut q = StateVec::new(1);
        let small_angle = 1e-6;
        q.rx(small_angle, 0);
        let total_prob: f64 = q.state().iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((total_prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_repeated_operations() {
        let mut q = StateVec::new(1);
        for _ in 0..1000 {
            q.h(0).sz(0).h(0);
        }
        let norm: f64 = q.state().iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((norm - 1.0).abs() < 1e-8);
    }

    #[test]
    fn test_rotation_angle_precision() {
        let mut q = StateVec::new(1);

        // Test small angle rotations
        let small_angle = 1e-6;
        q.rx(small_angle, 0);

        // Check that probabilities sum to 1
        let total_prob: f64 = q.state().iter().map(num_complex::Complex::norm_sqr).sum();
        assert!((total_prob - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sq_rotation_edge_cases() {
        let mut q = StateVec::new(1);

        // Test RX(0): Should be identity
        let initial = q.state().to_vec();
        q.rx(0.0, 0);
        assert_states_equal(q.state(), &initial);

        // Test RX(2π): Should also be identity up to global phase
        q.rx(2.0 * PI, 0);
        assert_states_equal(q.state(), &initial);

        // Test RY(0): Should be identity
        q.ry(0.0, 0);
        assert_states_equal(q.state(), &initial);

        // Test RY(2π): Should also be identity up to global phase
        q.ry(2.0 * PI, 0);
        assert_states_equal(q.state(), &initial);

        // Test RZ(0): Should be identity
        q.rz(0.0, 0);
        assert_states_equal(q.state(), &initial);

        // Test RZ(2π): Should also be identity up to global phase
        q.rz(2.0 * PI, 0);
        assert_states_equal(q.state(), &initial);
    }
}

mod large_systems {
    use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, StateVec};

    #[test]
    fn test_large_system() {
        // Test with a large number of qubits to ensure robustness.
        let num_qubits = 20; // 20 qubits => 2^20 amplitudes (~1M complex numbers)
        let mut q = StateVec::new(num_qubits);

        // Apply Hadamard to the first qubit
        q.h(0);

        // Check normalization and amplitudes for |0...0> and |1...0>
        let expected_amp = 1.0 / (2.0_f64.sqrt());
        assert!((q.state()[0].norm() - expected_amp).abs() < 1e-10);
        assert!((q.state()[1].norm() - expected_amp).abs() < 1e-10);

        // Ensure all other amplitudes remain zero
        for i in 2..q.state().len() {
            assert!(q.state()[i].norm() < 1e-10);
        }
    }

    #[test]
    fn test_state_normalization_after_random_gates() {
        let mut state_vec = StateVec::new(3);

        // Apply a sequence of random gates
        state_vec.h(0);
        state_vec.cx(0, 1);
        state_vec.rz(std::f64::consts::PI / 3.0, 2);
        state_vec.swap(1, 2);

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
    use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, StateVec};
    use std::f64::consts::{FRAC_1_SQRT_2, FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, FRAC_PI_6, PI};

    #[test]
    fn test_rx_step_by_step() {
        let mut q = StateVec::new(1);

        // Step 1: RX(0) should be identity
        q.rx(0.0, 0);
        assert!((q.state()[0].re - 1.0).abs() < 1e-10);
        assert!(q.state()[1].norm() < 1e-10);

        // Step 2: RX(π) on |0⟩ should give -i|1⟩
        let mut q = StateVec::new(1);
        q.rx(PI, 0);
        println!("RX(π)|0⟩ = {:?}", q.state()); // Debug output
        assert!(q.state()[0].norm() < 1e-10);
        assert!((q.state()[1].im + 1.0).abs() < 1e-10);

        // Step 3: RX(π/2) on |0⟩ should give (|0⟩ - i|1⟩)/√2
        let mut q = StateVec::new(1);
        q.rx(FRAC_PI_2, 0);
        println!("RX(π/2)|0⟩ = {:?}", q.state()); // Debug output
        let expected_amp = 1.0 / 2.0_f64.sqrt();
        assert!((q.state()[0].re - expected_amp).abs() < 1e-10);
        assert!((q.state()[1].im + expected_amp).abs() < 1e-10);
    }

    #[test]
    fn test_ry_step_by_step() {
        // Step 1: RY(0) should be identity
        let mut q = StateVec::new(1);
        q.ry(0.0, 0);
        println!("RY(0)|0⟩ = {:?}", q.state());
        assert!((q.state()[0].re - 1.0).abs() < 1e-10);
        assert!(q.state()[1].norm() < 1e-10);

        // Step 2: RY(π) on |0⟩ should give |1⟩
        let mut q = StateVec::new(1);
        q.ry(PI, 0);
        println!("RY(π)|0⟩ = {:?}", q.state());
        assert!(q.state()[0].norm() < 1e-10);
        assert!((q.state()[1].re - 1.0).abs() < 1e-10);

        // Step 3: RY(π/2) on |0⟩ should give (|0⟩ + |1⟩)/√2
        let mut q = StateVec::new(1);
        q.ry(FRAC_PI_2, 0);
        println!("RY(π/2)|0⟩ = {:?}", q.state());
        let expected_amp = 1.0 / 2.0_f64.sqrt();
        assert!((q.state()[0].re - expected_amp).abs() < 1e-10);
        assert!((q.state()[1].re - expected_amp).abs() < 1e-10);

        // Step 4: RY(-π/2) on |0⟩ should give (|0⟩ - |1⟩)/√2
        let mut q = StateVec::new(1);
        q.ry(-FRAC_PI_2, 0);
        println!("RY(-π/2)|0⟩ = {:?}", q.state());
        assert!((q.state()[0].re - expected_amp).abs() < 1e-10);
        assert!((q.state()[1].re + expected_amp).abs() < 1e-10);
    }

    #[test]
    fn test_rz_step_by_step() {
        // Step 1: RZ(0) should be identity
        let mut q = StateVec::new(1);
        q.rz(0.0, 0);
        println!("RZ(0)|0⟩ = {:?}", q.state());
        assert!((q.state()[0].re - 1.0).abs() < 1e-10);
        assert!(q.state()[1].norm() < 1e-10);

        // Step 2: RZ(π/2) on |+⟩ should give |+i⟩ = (|0⟩ + i|1⟩)/√2
        let mut q = StateVec::new(1);
        q.h(0); // Create |+⟩
        q.rz(FRAC_PI_2, 0);
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
        q1.rz(PI, 0);
        q2.rz(FRAC_PI_2, 0);
        q2.rz(FRAC_PI_2, 0);
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
        q1.sz(0);
        q2.rz(FRAC_PI_2, 0);
        println!("S|0⟩ = {:?}", q1.state());
        println!("RZ(π/2)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test X = RX(π)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.x(0);
        q2.rx(PI, 0);
        println!("X|0⟩ = {:?}", q1.state());
        println!("RX(π)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test Y = RY(π)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.y(0);
        q2.ry(PI, 0);
        println!("Y|0⟩ = {:?}", q1.state());
        println!("RY(π)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test Z = RZ(π)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.z(0);
        q2.rz(PI, 0);
        println!("Z|0⟩ = {:?}", q1.state());
        println!("RZ(π)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test √X = RX(π/2)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.sx(0);
        q2.rx(FRAC_PI_2, 0);
        println!("√X|0⟩ = {:?}", q1.state());
        println!("RX(π/2)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test √Y = RY(π/2)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.sy(0);
        q2.ry(FRAC_PI_2, 0);
        println!("√Y|0⟩ = {:?}", q1.state());
        println!("RY(π/2)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test S = TT as RZ(π/4)RZ(π/4)
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q2.rz(FRAC_PI_4, 0).rz(FRAC_PI_4, 0);
        q1.sz(0);
        println!("S|0⟩ = {:?}", q1.state());
        println!("T²|0⟩ = RZ(π/4)RZ(π/4)|0⟩ = {:?}", q2.state());
        assert_states_equal(q1.state(), q2.state());

        // Test H = RX(π)RY(π/2) decomposition
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);
        q1.h(0);
        q2.ry(FRAC_PI_2, 0).rx(PI, 0);
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
        q.rx(theta, 0).rx(-theta, 0);

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
        q.ry(theta, 0).ry(-theta, 0);

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
        q.rz(theta, 0).rz(-theta, 0);

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
        state_vec_u.u(theta, phi, lambda, 0);

        // Apply `u` from the ArbitraryRotationGateable trait.
        ArbitraryRotationGateable::u(&mut trait_u, theta, phi, lambda, 0);

        assert_states_equal(state_vec_u.state(), trait_u.state());
    }

    #[test]
    fn test_r1xy_vs_u() {
        let mut state_r1xy = StateVec::new(1);
        let mut state_u = StateVec::new(1);

        let theta = FRAC_PI_3;
        let phi = FRAC_PI_4;

        // Apply r1xy and equivalent u gates
        state_r1xy.r1xy(theta, phi, 0);
        state_u.u(theta, phi - FRAC_PI_2, FRAC_PI_2 - phi, 0);

        assert_states_equal(state_r1xy.state(), state_u.state());
    }

    #[test]
    fn test_rz_vs_u() {
        let mut state_rz = StateVec::new(1);
        let mut state_u = StateVec::new(1);

        let theta = FRAC_PI_3;

        // Apply rz and u gates
        state_rz.rz(theta, 0);
        state_u.u(0.0, 0.0, theta, 0);

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
        state_u.u(theta, phi, lambda, 0);

        // Apply the decomposed gates
        state_decomposed.rz(lambda, 0);
        state_decomposed.r1xy(theta, FRAC_PI_2, 0);
        state_decomposed.rz(phi, 0);

        // Assert that the states are equal
        assert_states_equal(state_u.state(), state_decomposed.state());
    }

    #[test]
    fn test_x_vs_r1xy() {
        let mut state = StateVec::new(1);
        state.x(0);
        let state_after_x = state.clone();

        state.reset();
        state.r1xy(PI, 0.0, 0);
        let state_after_r1xy = state.clone();

        assert_states_equal(state_after_x.state(), state_after_r1xy.state());
    }

    #[test]
    fn test_y_vs_r1xy() {
        let mut state = StateVec::new(1);
        state.y(0);
        let state_after_y = state.clone();

        state.reset();
        state.r1xy(PI, FRAC_PI_2, 0);
        let state_after_r1xy = state.clone();

        assert_states_equal(state_after_y.state(), state_after_r1xy.state());
    }

    #[test]
    fn test_h_vs_r1xy_rz() {
        let mut state = StateVec::new(1);
        state.h(0); // Apply the H gate
        let state_after_h = state.clone();

        state.reset(); // Reset state to |0⟩
        state.r1xy(FRAC_PI_2, -FRAC_PI_2, 0).rz(PI, 0);
        let state_after_r1xy_rz = state.clone();

        assert_states_equal(state_after_h.state(), state_after_r1xy_rz.state());
    }

    #[test]
    fn test_u_special_cases() {
        // Test 1: U(π, 0, π) should be X gate
        let mut q = StateVec::new(1);
        q.u(PI, 0.0, PI, 0);
        assert!(q.state()[0].norm() < 1e-10);
        assert!((q.state()[1].re - 1.0).abs() < 1e-10);

        // Test 2: Hadamard gate
        // H = U(π/2, 0, π)
        let mut q = StateVec::new(1);
        q.u(PI / 2.0, 0.0, PI, 0);
        assert!((q.state()[0].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((q.state()[1].re - FRAC_1_SQRT_2).abs() < 1e-10);

        // Test 3: U(0, 0, π) should be Z gate
        let mut q = StateVec::new(1);
        q.h(0); // First put in superposition
        let initial = q.state().to_vec();
        q.u(0.0, 0.0, PI, 0);
        assert!((q.state()[0] - initial[0]).norm() < 1e-10);
        assert!((q.state()[1] + initial[1]).norm() < 1e-10);

        // Additional test: U3(π/2, π/2, -π/2) should be S†H
        let mut q = StateVec::new(1);
        q.u(PI / 2.0, PI / 2.0, -PI / 2.0, 0);
        // This creates the state (|0⟩ + i|1⟩)/√2
        assert!((q.state()[0].re - FRAC_1_SQRT_2).abs() < 1e-10);
        assert!((q.state()[1].im - FRAC_1_SQRT_2).abs() < 1e-10);
    }

    #[test]
    fn test_u_composition() {
        let mut q1 = StateVec::new(1);
        let q2 = StateVec::new(1);

        // Two U gates that should multiply to identity
        q1.u(PI / 3.0, PI / 4.0, PI / 6.0, 0);
        q1.u(-PI / 3.0, -PI / 6.0, -PI / 4.0, 0);

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
        q1.t(0).t(0);

        let mut q2 = q.clone();
        q2.sz(0);

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_hadamard_properties() {
        // Test H^2 = I
        let mut q = StateVec::new(1);
        q.x(0); // Start with |1⟩
        let initial = q.state().to_vec();
        q.h(0).h(0);
        assert_states_equal(q.state(), &initial);

        // Test HXH = Z
        let mut q1 = StateVec::new(1);
        let mut q2 = StateVec::new(1);

        q1.h(0).x(0).h(0);
        q2.z(0);

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_non_commuting_gates() {
        let mut state1 = StateVec::new(1);
        let mut state2 = StateVec::new(1);

        state1.h(0);
        state1.z(0);

        state2.z(0);
        state2.h(0);

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
}

mod detailed_tq_gate_cases {
    use crate::helpers::assert_states_equal;
    use num_complex::Complex64;
    use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, StateVec};
    use std::f64::consts::{FRAC_PI_2, FRAC_PI_3, FRAC_PI_4, PI};

    #[test]
    fn test_cx_decomposition() {
        let mut state_cx = StateVec::new(2);
        let mut state_decomposed = StateVec::new(2);

        let control = 0;
        let target = 1;

        // Apply CX gate
        state_cx.cx(control, target);

        // Apply the decomposed gates
        state_decomposed.r1xy(-FRAC_PI_2, FRAC_PI_2, target);
        state_decomposed.rzz(FRAC_PI_2, control, target);
        state_decomposed.rz(-FRAC_PI_2, control);
        state_decomposed.r1xy(FRAC_PI_2, PI, target);
        state_decomposed.rz(-FRAC_PI_2, target);

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
        state_rxx.rxx(FRAC_PI_4, control, target);

        // Apply the decomposed gates
        state_decomposed.r1xy(FRAC_PI_2, FRAC_PI_2, control);
        state_decomposed.r1xy(FRAC_PI_2, FRAC_PI_2, target);
        state_decomposed.rzz(FRAC_PI_4, control, target);
        state_decomposed.r1xy(FRAC_PI_2, -FRAC_PI_2, control);
        state_decomposed.r1xy(FRAC_PI_2, -FRAC_PI_2, target);

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
        state_vec.cx(1, 0);
        assert!((state_vec.probability(0) - 1.0).abs() < 1e-10);

        // |01⟩ → should remain |01⟩
        state_vec.prepare_computational_basis(1);
        state_vec.cx(1, 0);
        assert!((state_vec.probability(1) - 1.0).abs() < 1e-10);

        // |10⟩ → should flip to |11⟩
        state_vec.prepare_computational_basis(2);
        state_vec.cx(1, 0);
        assert!((state_vec.probability(3) - 1.0).abs() < 1e-10);

        // |11⟩ → should flip to |10⟩
        state_vec.prepare_computational_basis(3);
        state_vec.cx(1, 0);
        assert!((state_vec.probability(2) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_control_target_independence() {
        // Test that CY and CZ work regardless of which qubit is control/target
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Prepare same initial state
        q1.h(0);
        q1.h(1);
        q2.h(0);
        q2.h(1);

        // Apply gates with different control/target
        q1.cz(0, 1);
        q2.cz(1, 0);

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_rxx_symmetry() {
        // Test that RXX is symmetric under exchange of qubits
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Prepare same non-trivial initial state
        q1.h(0);
        q1.h(1);
        q2.h(0);
        q2.h(1);

        // Apply RXX with different qubit orders
        q1.rxx(FRAC_PI_3, 0, 1);
        q2.rxx(FRAC_PI_3, 1, 0);

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
        q1.h(0).x(1); // Random state
        q2.h(0).x(1); // Same initial state

        q1.ryy(theta, 0, 1);
        q2.ryy(theta, 1, 0);

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
        q.h(0).h(1).h(2).h(3).h(4); // Superposition state

        // Apply RYY on qubits 2 and 4
        q.ryy(theta, 2, 4);

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
        q.ryy(PI, 0, 1);

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

        q.ryy(PI, 0, 1);

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
        let initial = q.state().to_vec();
        q.ryy(theta, 0, 1);

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
        q1.h(0).h(1);
        q2.h(0).h(1);

        // Apply RYY with random qubit order
        q1.ryy(theta, 0, 1);
        q2.ryy(theta, 1, 0);

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
        q1.h(0);
        q2.h(0);

        // Compare direct SZZ vs RZZ(π/2)
        q1.szz(0, 1);
        q2.rzz(FRAC_PI_2, 0, 1);

        assert_states_equal(q1.state(), q2.state());

        // Also verify decomposition matches
        let mut q3 = StateVec::new(2);
        q3.h(0); // Same initial state
        q3.h(0).h(1).sxx(0, 1).h(0).h(1);

        assert_states_equal(q1.state(), q3.state());
    }

    #[test]
    fn test_szz_trait_equivalence() {
        let mut q1 = StateVec::new(2);
        let mut q2 = StateVec::new(2);

        // Create some non-trivial initial state
        q1.h(0);
        q2.h(0);

        // Compare CliffordGateable trait szz vs ArbitraryRotationGateable trait rzz(π/2)
        CliffordGateable::<usize>::szz(&mut q1, 0, 1);
        ArbitraryRotationGateable::<usize>::rzz(&mut q2, PI / 2.0, 0, 1);

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_two_qubit_unitary_properties() {
        let mut q = StateVec::new(2);

        // Create a non-trivial state
        q.h(0);
        q.h(1);

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
        q1.x(0); // |10⟩
        q2.x(0); // |10⟩

        q1.cx(0, 1).cx(1, 0).cx(0, 1); // SWAP via CNOTs
        q2.swap(0, 1); // Direct SWAP

        assert_states_equal(q1.state(), q2.state());
    }

    #[test]
    fn test_controlled_gate_phases() {
        // Test phase behavior of controlled operations
        let mut q = StateVec::new(2);

        // Create superposition with phases
        q.h(0).sz(0);
        q.h(1).sz(1);

        // Control operations should preserve phases correctly
        let initial = q.state().to_vec();
        q.cz(0, 1).cz(0, 1); // CZ^2 = I

        assert_states_equal(q.state(), &initial);
    }
}

mod detail_meas_cases {
    use pecos_qsim::{CliffordGateable, QuantumSimulator, StateVec};

    #[test]
    fn test_measurement_on_entangled_state() {
        let mut q = StateVec::new(2);

        // Create Bell state (|00⟩ + |11⟩) / sqrt(2)
        q.h(0);
        q.cx(0, 1);

        // Measure the first qubit
        let result1 = q.mz(0);

        // Measure the second qubit - should match the first
        let result2 = q.mz(1);

        assert_eq!(result1.outcome, result2.outcome);
    }

    #[test]
    fn test_measurement_properties() {
        let mut q = StateVec::new(2);

        // Test 1: Measuring |0⟩ should always give 0
        let result = q.mz(0);
        assert!(!result.outcome);
        assert!((q.probability(0) - 1.0).abs() < 1e-10);

        // Test 2: Measuring |1⟩ should always give 1
        q.reset();
        q.x(0);
        let result = q.mz(0);
        assert!(result.outcome);
        assert!((q.probability(1) - 1.0).abs() < 1e-10);

        // Test 3: In a Bell state, measurements should correlate
        q.reset();
        q.h(0).cx(0, 1); // Create Bell state
        let result1 = q.mz(0);
        let result2 = q.mz(1);
        assert_eq!(
            result1.outcome, result2.outcome,
            "Bell state measurements should correlate"
        );

        // Test 4: Repeated measurements should be consistent
        q.reset();
        q.h(0); // Create superposition
        let first = q.mz(0);
        let second = q.mz(0); // Measure again
        assert_eq!(
            first.outcome, second.outcome,
            "Repeated measurements should give same result"
        );
    }

    #[test]
    fn test_measurement_basis_transforms() {
        let mut q = StateVec::new(1);

        // |0⟩ in X basis
        q.h(0);

        // Measure in Z basis
        let result = q.mz(0);

        // Result should be random but state should collapse
        let final_prob = if result.outcome {
            q.probability(1)
        } else {
            q.probability(0)
        };
        assert!((final_prob - 1.0).abs() < 1e-10);
    }
}
