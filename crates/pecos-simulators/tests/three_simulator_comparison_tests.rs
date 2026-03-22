use pecos_core::Angle64;
use pecos_simulators::ArbitraryRotationGateable;
use pecos_simulators::CliffordGateable;
use pecos_simulators::DensityMatrix;
use pecos_simulators::QuantumSimulator;
use pecos_simulators::StateVec;
use pecos_simulators::{SparseStab, qid, qid2};
use std::f64::consts::PI;

// Helper function to check if two probabilities are close enough
fn assert_probs_equal(p1: f64, p2: f64) {
    assert!(
        (p1 - p2).abs() < 1e-10,
        "Probabilities differ: {p1} vs {p2}"
    );
}

// Helper function to compare probabilities for all three simulators
fn compare_all_probabilities(
    sv: &mut StateVec,
    dm: &mut DensityMatrix,
    stab: &SparseStab,
    num_qubits: usize,
) {
    for i in 0..(1 << num_qubits) {
        let sv_prob = sv.probability(i);
        let dm_prob = dm.probability(i);

        // For stabilizer, calculate probability by measuring qubits
        let mut stab_prob = 1.0;
        let mut probability_is_zero = false;

        // Create a fresh copy for each basis state
        let mut stab_copy = stab.clone();

        for q in 0..num_qubits {
            // Check if we want bit q to be 0 or 1
            let bit_is_one = (i >> q) & 1 == 1;

            // Try to force the measurement to the desired outcome
            let result = stab_copy.mz_forced(q, bit_is_one);

            // If this was a non-deterministic measurement, the probability is 0.5
            if !result.is_deterministic {
                stab_prob *= 0.5;
            } else if result.outcome != bit_is_one {
                // If deterministic but different from what we want, probability is 0
                probability_is_zero = true;
                break;
            }
        }

        let stab_prob = if probability_is_zero { 0.0 } else { stab_prob };

        // Compare all probability pairs
        assert_probs_equal(sv_prob, dm_prob);
        assert_probs_equal(sv_prob, stab_prob);
        assert_probs_equal(dm_prob, stab_prob);
    }
}

// Helper function to compare a clifford circuit among all three simulators
fn compare_clifford_circuit<F>(num_qubits: usize, circuit_fn: F)
where
    F: Fn(&mut StateVec, &mut DensityMatrix, &mut SparseStab),
{
    let seed = 42; // Fixed seed for determinism
    let mut sv = StateVec::with_seed(num_qubits, seed);
    let mut dm = DensityMatrix::with_seed(num_qubits, seed);
    let mut stab = SparseStab::with_seed(num_qubits, seed);

    // Apply the circuit to all three simulators
    circuit_fn(&mut sv, &mut dm, &mut stab);

    // Compare the resulting states
    compare_all_probabilities(&mut sv, &mut dm, &stab, num_qubits);
}

// Helper function to compare a general circuit between StateVec and DensityMatrix
fn compare_general_circuit<F>(num_qubits: usize, circuit_fn: F)
where
    F: Fn(&mut StateVec, &mut DensityMatrix),
{
    let seed = 42; // Fixed seed for determinism
    let mut sv = StateVec::with_seed(num_qubits, seed);
    let mut dm = DensityMatrix::with_seed(num_qubits, seed);

    // Apply the circuit to the two non-Clifford simulators
    circuit_fn(&mut sv, &mut dm);

    // Compare the resulting states
    for i in 0..(1 << num_qubits) {
        let sv_prob = sv.probability(i);
        let dm_prob = dm.probability(i);
        assert_probs_equal(sv_prob, dm_prob);
    }
}

#[test]
fn test_initial_state_consistency() {
    for num_qubits in 1..=5 {
        let mut sv = StateVec::new(num_qubits);
        let mut dm = DensityMatrix::new(num_qubits);
        let stab = SparseStab::new(num_qubits);

        compare_all_probabilities(&mut sv, &mut dm, &stab, num_qubits);
    }
}

#[test]
fn test_basic_gates_consistency() {
    // Test X, H gates on a single qubit across all simulators
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.x(&qid(0));
        dm.x(&qid(0));
        stab.x(&qid(0));
    });

    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h(&qid(0));
        dm.h(&qid(0));
        stab.h(&qid(0));
    });

    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.y(&qid(0));
        dm.y(&qid(0));
        stab.y(&qid(0));
    });

    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.z(&qid(0));
        dm.z(&qid(0));
        stab.z(&qid(0));
    });

    // Test sequence of gates
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h(&qid(0)).z(&qid(0)).h(&qid(0)); // Effective X gate
        dm.h(&qid(0)).z(&qid(0)).h(&qid(0));
        stab.h(&qid(0)).z(&qid(0)).h(&qid(0));
    });
}

#[test]
fn test_phase_gates_consistency() {
    // Test phase gates (S = sqrt of Z)
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.sz(&qid(0));
        dm.sz(&qid(0));
        stab.sz(&qid(0));
    });

    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h(&qid(0)).sz(&qid(0)).h(&qid(0));
        dm.h(&qid(0)).sz(&qid(0)).h(&qid(0));
        stab.h(&qid(0)).sz(&qid(0)).h(&qid(0));
    });

    // Test that S^2 = Z
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.sz(&qid(0)).sz(&qid(0));
        dm.sz(&qid(0)).sz(&qid(0));
        stab.sz(&qid(0)).sz(&qid(0));
    });
}

#[test]
fn test_multi_qubit_gates_consistency() {
    // Test two-qubit CNOT gate
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.cx(&qid2(0, 1));
        dm.cx(&qid2(0, 1));
        stab.cx(&qid2(0, 1));
    });

    // Test CZ gate
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.cz(&qid2(0, 1));
        dm.cz(&qid2(0, 1));
        stab.cz(&qid2(0, 1));
    });

    // Test SWAP gate
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.swap(&qid2(0, 1));
        dm.swap(&qid2(0, 1));
        stab.swap(&qid2(0, 1));
    });
}

#[test]
fn test_bell_state_consistency() {
    // Test creation of Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).cx(&qid2(0, 1));
        dm.h(&qid(0)).cx(&qid2(0, 1));
        stab.h(&qid(0)).cx(&qid2(0, 1));
    });

    // Test creation of Bell state |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).cx(&qid2(0, 1)).z(&qid(1));
        dm.h(&qid(0)).cx(&qid2(0, 1)).z(&qid(1));
        stab.h(&qid(0)).cx(&qid2(0, 1)).z(&qid(1));
    });

    // Test creation of Bell state |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).cx(&qid2(0, 1)).x(&qid(1));
        dm.h(&qid(0)).cx(&qid2(0, 1)).x(&qid(1));
        stab.h(&qid(0)).cx(&qid2(0, 1)).x(&qid(1));
    });

    // Test creation of Bell state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).cx(&qid2(0, 1)).z(&qid(0)).x(&qid(1));
        dm.h(&qid(0)).cx(&qid2(0, 1)).z(&qid(0)).x(&qid(1));
        stab.h(&qid(0)).cx(&qid2(0, 1)).z(&qid(0)).x(&qid(1));
    });
}

#[test]
fn test_ghz_state_consistency() {
    // Test creation of GHZ state (|000⟩ + |111⟩)/√2 with increasing number of qubits
    for num_qubits in 3..=5 {
        compare_clifford_circuit(num_qubits, |sv, dm, stab| {
            sv.h(&qid(0));
            dm.h(&qid(0));
            stab.h(&qid(0));

            // Entangle all qubits
            for i in 0..(num_qubits - 1) {
                sv.cx(&qid2(i, i + 1));
                dm.cx(&qid2(i, i + 1));
                stab.cx(&qid2(i, i + 1));
            }
        });
    }
}

#[test]
fn test_measurement_consistency() {
    let num_qubits = 1;
    let seed = 42; // Fixed seed for deterministic behavior

    // Test Z-basis measurement
    let mut sv = StateVec::with_seed(num_qubits, seed);
    let mut dm = DensityMatrix::with_seed(num_qubits, seed);
    let mut stab = SparseStab::with_seed(num_qubits, seed);

    // Put qubits in superposition
    sv.h(&qid(0));
    dm.h(&qid(0));
    stab.h(&qid(0));

    // With identical seeds, measurements should give identical results
    let sv_result = sv.mz(&qid(0)).into_iter().next().unwrap();
    let dm_result = dm.mz(&qid(0)).into_iter().next().unwrap();
    let stab_result = stab.mz(&qid(0)).into_iter().next().unwrap();

    assert_eq!(sv_result.outcome, dm_result.outcome);
    assert_eq!(sv_result.outcome, stab_result.outcome);
    assert_eq!(dm_result.outcome, stab_result.outcome);

    assert_eq!(sv_result.is_deterministic, dm_result.is_deterministic);
    assert_eq!(sv_result.is_deterministic, stab_result.is_deterministic);
    assert_eq!(dm_result.is_deterministic, stab_result.is_deterministic);

    // After measurement, states should be consistent
    compare_all_probabilities(&mut sv, &mut dm, &stab, num_qubits);

    // Test X-basis measurement (H→Z→H)
    let mut sv = StateVec::with_seed(num_qubits, seed);
    let mut dm = DensityMatrix::with_seed(num_qubits, seed);
    let mut stab = SparseStab::with_seed(num_qubits, seed);

    // Prepare |0⟩, then apply Z to get a deterministic result
    sv.z(&qid(0));
    dm.z(&qid(0));
    stab.z(&qid(0));

    // Measure in X basis
    let sv_result = sv.mx(&qid(0)).into_iter().next().unwrap();
    let dm_result = dm.mx(&qid(0)).into_iter().next().unwrap();
    let stab_result = stab.mx(&qid(0)).into_iter().next().unwrap();

    assert_eq!(sv_result.outcome, dm_result.outcome);
    assert_eq!(sv_result.outcome, stab_result.outcome);
    assert_eq!(dm_result.outcome, stab_result.outcome);

    assert_eq!(sv_result.is_deterministic, dm_result.is_deterministic);
    assert_eq!(sv_result.is_deterministic, stab_result.is_deterministic);
    assert_eq!(dm_result.is_deterministic, stab_result.is_deterministic);

    // After measurement, all states should still be consistent
    compare_all_probabilities(&mut sv, &mut dm, &stab, num_qubits);
}

#[test]
fn test_complex_circuit_consistency() {
    // Test more complex Clifford circuits
    for num_qubits in 3..=4 {
        compare_clifford_circuit(num_qubits, |sv, dm, stab| {
            // Create GHZ state
            sv.h(&qid(0));
            dm.h(&qid(0));
            stab.h(&qid(0));

            for i in 0..(num_qubits - 1) {
                sv.cx(&qid2(i, i + 1));
                dm.cx(&qid2(i, i + 1));
                stab.cx(&qid2(i, i + 1));
            }

            // Apply some additional gates
            sv.h(&qid(1)).sz(&qid(2));
            dm.h(&qid(1)).sz(&qid(2));
            stab.h(&qid(1)).sz(&qid(2));

            if num_qubits > 3 {
                sv.cz(&qid2(0, 3)).swap(&qid2(1, 2));
                dm.cz(&qid2(0, 3)).swap(&qid2(1, 2));
                stab.cz(&qid2(0, 3)).swap(&qid2(1, 2));
            }
        });
    }
}

#[test]
fn test_non_clifford_circuits() {
    // Test rotation gates (only StateVec and DensityMatrix)
    compare_general_circuit(1, |sv, dm| {
        sv.rx(Angle64::from_radians(PI / 4.0), &qid(0));
        dm.rx(Angle64::from_radians(PI / 4.0), &qid(0));
    });

    compare_general_circuit(1, |sv, dm| {
        sv.rz(Angle64::from_radians(PI / 3.0), &qid(0));
        dm.rz(Angle64::from_radians(PI / 3.0), &qid(0));
    });

    // Test two-qubit rotations
    compare_general_circuit(2, |sv, dm| {
        sv.h(&qid(0))
            .h(&qid(1))
            .rzz(Angle64::from_radians(PI / 4.0), &qid2(0, 1));
        dm.h(&qid(0))
            .h(&qid(1))
            .rzz(Angle64::from_radians(PI / 4.0), &qid2(0, 1));
    });

    // Test complex non-Clifford circuit
    compare_general_circuit(3, |sv, dm| {
        // Create GHZ state
        sv.h(&qid(0)).cx(&qid2(0, 1)).cx(&qid2(1, 2));
        dm.h(&qid(0)).cx(&qid2(0, 1)).cx(&qid2(1, 2));

        // Apply non-Clifford rotations
        sv.rx(Angle64::from_radians(PI / 5.0), &qid(0))
            .rz(Angle64::from_radians(PI / 7.0), &qid(1))
            .rzz(Angle64::from_radians(PI / 9.0), &qid2(0, 2));
        dm.rx(Angle64::from_radians(PI / 5.0), &qid(0))
            .rz(Angle64::from_radians(PI / 7.0), &qid(1))
            .rzz(Angle64::from_radians(PI / 9.0), &qid2(0, 2));
    });
}

#[test]
fn test_prepare_computational_basis_consistency() {
    for num_qubits in 1..=3 {
        // Test each computational basis state
        for i in 0..(1 << num_qubits) {
            let mut sv = StateVec::new(num_qubits);
            let mut dm = DensityMatrix::new(num_qubits);
            let mut stab = SparseStab::new(num_qubits);

            // For stabilizer simulator, we need to manually apply X gates
            // based on the bits of i
            for q in 0..num_qubits {
                if (i >> q) & 1 == 1 {
                    stab.x(&qid(q));
                }
            }

            // Use prepare_computational_basis for StateVec and DensityMatrix
            sv.prepare_computational_basis(i);
            dm.prepare_computational_basis(i);

            // Compare the states
            compare_all_probabilities(&mut sv, &mut dm, &stab, num_qubits);
        }
    }
}

#[test]
fn test_prepare_plus_states_consistency() {
    // Test |+⟩ state preparation using H gates
    for num_qubits in 1..=3 {
        let mut sv = StateVec::new(num_qubits);
        let mut dm = DensityMatrix::new(num_qubits);
        let mut stab = SparseStab::new(num_qubits);

        // Apply H to all qubits
        for q in 0..num_qubits {
            sv.h(&qid(q));
            dm.h(&qid(q));
            stab.h(&qid(q));
        }

        // Compare the states
        compare_all_probabilities(&mut sv, &mut dm, &stab, num_qubits);
    }
}

#[test]
fn test_reset_consistency() {
    // Test reset behavior
    for num_qubits in 1..=3 {
        let mut sv = StateVec::new(num_qubits);
        let mut dm = DensityMatrix::new(num_qubits);
        let mut stab = SparseStab::new(num_qubits);

        // Apply some gates to get to a non-trivial state
        sv.h(&qid(0));
        dm.h(&qid(0));
        stab.h(&qid(0));

        if num_qubits > 1 {
            sv.cx(&qid2(0, 1));
            dm.cx(&qid2(0, 1));
            stab.cx(&qid2(0, 1));
        }

        // Reset all simulators
        sv.reset();
        dm.reset();
        stab.reset();

        // Compare the states
        compare_all_probabilities(&mut sv, &mut dm, &stab, num_qubits);
    }
}

// ============================================================================
// H-variant gates (H2-H6): 3-simulator consistency
// ============================================================================

#[test]
fn test_h_variant_gates_consistency() {
    // Each H variant applied to |0>
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h2(&qid(0));
        dm.h2(&qid(0));
        stab.h2(&qid(0));
    });
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h3(&qid(0));
        dm.h3(&qid(0));
        stab.h3(&qid(0));
    });
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h4(&qid(0));
        dm.h4(&qid(0));
        stab.h4(&qid(0));
    });
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h5(&qid(0));
        dm.h5(&qid(0));
        stab.h5(&qid(0));
    });
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h6(&qid(0));
        dm.h6(&qid(0));
        stab.h6(&qid(0));
    });

    // H variants applied to H|0> = |+>
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h(&qid(0)).h2(&qid(0));
        dm.h(&qid(0)).h2(&qid(0));
        stab.h(&qid(0)).h2(&qid(0));
    });
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h(&qid(0)).h3(&qid(0));
        dm.h(&qid(0)).h3(&qid(0));
        stab.h(&qid(0)).h3(&qid(0));
    });

    // All H variants are self-inverse: Hi * Hi = I
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h(&qid(0)).h2(&qid(0)).h2(&qid(0));
        dm.h(&qid(0)).h2(&qid(0)).h2(&qid(0));
        stab.h(&qid(0)).h2(&qid(0)).h2(&qid(0));
    });
}

// ============================================================================
// F-family gates: 3-simulator consistency
// ============================================================================

#[test]
fn test_f_family_gates_consistency() {
    // Each F variant on |0>
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.f(&qid(0));
        dm.f(&qid(0));
        stab.f(&qid(0));
    });
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.fdg(&qid(0));
        dm.fdg(&qid(0));
        stab.fdg(&qid(0));
    });
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.f2(&qid(0));
        dm.f2(&qid(0));
        stab.f2(&qid(0));
    });
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.f2dg(&qid(0));
        dm.f2dg(&qid(0));
        stab.f2dg(&qid(0));
    });
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.f3(&qid(0));
        dm.f3(&qid(0));
        stab.f3(&qid(0));
    });
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.f3dg(&qid(0));
        dm.f3dg(&qid(0));
        stab.f3dg(&qid(0));
    });
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.f4(&qid(0));
        dm.f4(&qid(0));
        stab.f4(&qid(0));
    });
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.f4dg(&qid(0));
        dm.f4dg(&qid(0));
        stab.f4dg(&qid(0));
    });

    // F * Fdg = I
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h(&qid(0)).f(&qid(0)).fdg(&qid(0));
        dm.h(&qid(0)).f(&qid(0)).fdg(&qid(0));
        stab.h(&qid(0)).f(&qid(0)).fdg(&qid(0));
    });

    // F^3 = I (F is order 3)
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h(&qid(0)).f(&qid(0)).f(&qid(0)).f(&qid(0));
        dm.h(&qid(0)).f(&qid(0)).f(&qid(0)).f(&qid(0));
        stab.h(&qid(0)).f(&qid(0)).f(&qid(0)).f(&qid(0));
    });

    // F on |+>
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h(&qid(0)).f(&qid(0));
        dm.h(&qid(0)).f(&qid(0));
        stab.h(&qid(0)).f(&qid(0));
    });
}

// ============================================================================
// All 2q Clifford gates: 3-simulator consistency
// ============================================================================

#[test]
fn test_all_2q_gates_consistency() {
    // SXX family
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).sxx(&qid2(0, 1));
        dm.h(&qid(0)).sxx(&qid2(0, 1));
        stab.h(&qid(0)).sxx(&qid2(0, 1));
    });
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).sxxdg(&qid2(0, 1));
        dm.h(&qid(0)).sxxdg(&qid2(0, 1));
        stab.h(&qid(0)).sxxdg(&qid2(0, 1));
    });

    // SYY family
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).syy(&qid2(0, 1));
        dm.h(&qid(0)).syy(&qid2(0, 1));
        stab.h(&qid(0)).syy(&qid2(0, 1));
    });
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).syydg(&qid2(0, 1));
        dm.h(&qid(0)).syydg(&qid2(0, 1));
        stab.h(&qid(0)).syydg(&qid2(0, 1));
    });

    // SZZ family
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).szz(&qid2(0, 1));
        dm.h(&qid(0)).szz(&qid2(0, 1));
        stab.h(&qid(0)).szz(&qid2(0, 1));
    });
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).szzdg(&qid2(0, 1));
        dm.h(&qid(0)).szzdg(&qid2(0, 1));
        stab.h(&qid(0)).szzdg(&qid2(0, 1));
    });

    // ISWAP family
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).iswap(&qid2(0, 1));
        dm.h(&qid(0)).iswap(&qid2(0, 1));
        stab.h(&qid(0)).iswap(&qid2(0, 1));
    });
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).iswapdg(&qid2(0, 1));
        dm.h(&qid(0)).iswapdg(&qid2(0, 1));
        stab.h(&qid(0)).iswapdg(&qid2(0, 1));
    });

    // G family (G is self-inverse)
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).g(&qid2(0, 1));
        dm.h(&qid(0)).g(&qid2(0, 1));
        stab.h(&qid(0)).g(&qid2(0, 1));
    });
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).gdg(&qid2(0, 1));
        dm.h(&qid(0)).gdg(&qid2(0, 1));
        stab.h(&qid(0)).gdg(&qid2(0, 1));
    });

    // CY
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).cy(&qid2(0, 1));
        dm.h(&qid(0)).cy(&qid2(0, 1));
        stab.h(&qid(0)).cy(&qid2(0, 1));
    });

    // ISWAP * ISWAPdg = I
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).iswap(&qid2(0, 1)).iswapdg(&qid2(0, 1));
        dm.h(&qid(0)).iswap(&qid2(0, 1)).iswapdg(&qid2(0, 1));
        stab.h(&qid(0)).iswap(&qid2(0, 1)).iswapdg(&qid2(0, 1));
    });

    // G * G = I (G is Hermitian)
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(&qid(0)).g(&qid2(0, 1)).g(&qid2(0, 1));
        dm.h(&qid(0)).g(&qid2(0, 1)).g(&qid2(0, 1));
        stab.h(&qid(0)).g(&qid2(0, 1)).g(&qid2(0, 1));
    });
}
