use pecos_qsim::ArbitraryRotationGateable;
use pecos_qsim::CliffordGateable;
use pecos_qsim::DensityMatrix;
use pecos_qsim::QuantumSimulator;
use pecos_qsim::StateVec;
use pecos_qsim::StdSparseStab;
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
    sv: &StateVec,
    dm: &DensityMatrix,
    stab: &StdSparseStab,
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
    F: Fn(&mut StateVec, &mut DensityMatrix, &mut StdSparseStab),
{
    let seed = 42; // Fixed seed for determinism
    let mut sv = StateVec::with_seed(num_qubits, seed);
    let mut dm = DensityMatrix::with_seed(num_qubits, seed);
    let mut stab = StdSparseStab::with_seed(num_qubits, seed);

    // Apply the circuit to all three simulators
    circuit_fn(&mut sv, &mut dm, &mut stab);

    // Compare the resulting states
    compare_all_probabilities(&sv, &dm, &stab, num_qubits);
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
        let sv = StateVec::new(num_qubits);
        let dm = DensityMatrix::new(num_qubits);
        let stab = StdSparseStab::new(num_qubits);

        compare_all_probabilities(&sv, &dm, &stab, num_qubits);
    }
}

#[test]
fn test_basic_gates_consistency() {
    // Test X, H gates on a single qubit across all simulators
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.x(0);
        dm.x(0);
        stab.x(0);
    });

    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h(0);
        dm.h(0);
        stab.h(0);
    });

    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.y(0);
        dm.y(0);
        stab.y(0);
    });

    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.z(0);
        dm.z(0);
        stab.z(0);
    });

    // Test sequence of gates
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h(0).z(0).h(0); // Effective X gate
        dm.h(0).z(0).h(0);
        stab.h(0).z(0).h(0);
    });
}

#[test]
fn test_phase_gates_consistency() {
    // Test phase gates (S = sqrt of Z)
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.sz(0);
        dm.sz(0);
        stab.sz(0);
    });

    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.h(0).sz(0).h(0);
        dm.h(0).sz(0).h(0);
        stab.h(0).sz(0).h(0);
    });

    // Test that S^2 = Z
    compare_clifford_circuit(1, |sv, dm, stab| {
        sv.sz(0).sz(0);
        dm.sz(0).sz(0);
        stab.sz(0).sz(0);
    });
}

#[test]
fn test_multi_qubit_gates_consistency() {
    // Test two-qubit CNOT gate
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.cx(0, 1);
        dm.cx(0, 1);
        stab.cx(0, 1);
    });

    // Test CZ gate
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.cz(0, 1);
        dm.cz(0, 1);
        stab.cz(0, 1);
    });

    // Test SWAP gate
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.swap(0, 1);
        dm.swap(0, 1);
        stab.swap(0, 1);
    });
}

#[test]
fn test_bell_state_consistency() {
    // Test creation of Bell state |Φ⁺⟩ = (|00⟩ + |11⟩)/√2
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(0).cx(0, 1);
        dm.h(0).cx(0, 1);
        stab.h(0).cx(0, 1);
    });

    // Test creation of Bell state |Φ⁻⟩ = (|00⟩ - |11⟩)/√2
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(0).cx(0, 1).z(1);
        dm.h(0).cx(0, 1).z(1);
        stab.h(0).cx(0, 1).z(1);
    });

    // Test creation of Bell state |Ψ⁺⟩ = (|01⟩ + |10⟩)/√2
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(0).cx(0, 1).x(1);
        dm.h(0).cx(0, 1).x(1);
        stab.h(0).cx(0, 1).x(1);
    });

    // Test creation of Bell state |Ψ⁻⟩ = (|01⟩ - |10⟩)/√2
    compare_clifford_circuit(2, |sv, dm, stab| {
        sv.h(0).cx(0, 1).z(0).x(1);
        dm.h(0).cx(0, 1).z(0).x(1);
        stab.h(0).cx(0, 1).z(0).x(1);
    });
}

#[test]
fn test_ghz_state_consistency() {
    // Test creation of GHZ state (|000⟩ + |111⟩)/√2 with increasing number of qubits
    for num_qubits in 3..=5 {
        compare_clifford_circuit(num_qubits, |sv, dm, stab| {
            sv.h(0);
            dm.h(0);
            stab.h(0);

            // Entangle all qubits
            for i in 0..(num_qubits - 1) {
                sv.cx(i, i + 1);
                dm.cx(i, i + 1);
                stab.cx(i, i + 1);
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
    let mut stab = StdSparseStab::with_seed(num_qubits, seed);

    // Put qubits in superposition
    sv.h(0);
    dm.h(0);
    stab.h(0);

    // With identical seeds, measurements should give identical results
    let sv_result = sv.mz(0);
    let dm_result = dm.mz(0);
    let stab_result = stab.mz(0);

    assert_eq!(sv_result.outcome, dm_result.outcome);
    assert_eq!(sv_result.outcome, stab_result.outcome);
    assert_eq!(dm_result.outcome, stab_result.outcome);

    assert_eq!(sv_result.is_deterministic, dm_result.is_deterministic);
    assert_eq!(sv_result.is_deterministic, stab_result.is_deterministic);
    assert_eq!(dm_result.is_deterministic, stab_result.is_deterministic);

    // After measurement, states should be consistent
    compare_all_probabilities(&sv, &dm, &stab, num_qubits);

    // Test X-basis measurement (H→Z→H)
    let mut sv = StateVec::with_seed(num_qubits, seed);
    let mut dm = DensityMatrix::with_seed(num_qubits, seed);
    let mut stab = StdSparseStab::with_seed(num_qubits, seed);

    // Prepare |0⟩, then apply Z to get a deterministic result
    sv.z(0);
    dm.z(0);
    stab.z(0);

    // Measure in X basis
    let sv_result = sv.mx(0);
    let dm_result = dm.mx(0);
    let stab_result = stab.mx(0);

    assert_eq!(sv_result.outcome, dm_result.outcome);
    assert_eq!(sv_result.outcome, stab_result.outcome);
    assert_eq!(dm_result.outcome, stab_result.outcome);

    assert_eq!(sv_result.is_deterministic, dm_result.is_deterministic);
    assert_eq!(sv_result.is_deterministic, stab_result.is_deterministic);
    assert_eq!(dm_result.is_deterministic, stab_result.is_deterministic);

    // After measurement, all states should still be consistent
    compare_all_probabilities(&sv, &dm, &stab, num_qubits);
}

#[test]
fn test_complex_circuit_consistency() {
    // Test more complex Clifford circuits
    for num_qubits in 3..=4 {
        compare_clifford_circuit(num_qubits, |sv, dm, stab| {
            // Create GHZ state
            sv.h(0);
            dm.h(0);
            stab.h(0);

            for i in 0..(num_qubits - 1) {
                sv.cx(i, i + 1);
                dm.cx(i, i + 1);
                stab.cx(i, i + 1);
            }

            // Apply some additional gates
            sv.h(1).sz(2);
            dm.h(1).sz(2);
            stab.h(1).sz(2);

            if num_qubits > 3 {
                sv.cz(0, 3).swap(1, 2);
                dm.cz(0, 3).swap(1, 2);
                stab.cz(0, 3).swap(1, 2);
            }
        });
    }
}

#[test]
fn test_non_clifford_circuits() {
    // Test rotation gates (only StateVec and DensityMatrix)
    compare_general_circuit(1, |sv, dm| {
        sv.rx(PI / 4.0, 0);
        dm.rx(PI / 4.0, 0);
    });

    compare_general_circuit(1, |sv, dm| {
        sv.rz(PI / 3.0, 0);
        dm.rz(PI / 3.0, 0);
    });

    // Test two-qubit rotations
    compare_general_circuit(2, |sv, dm| {
        sv.h(0).h(1).rzz(PI / 4.0, 0, 1);
        dm.h(0).h(1).rzz(PI / 4.0, 0, 1);
    });

    // Test complex non-Clifford circuit
    compare_general_circuit(3, |sv, dm| {
        // Create GHZ state
        sv.h(0).cx(0, 1).cx(1, 2);
        dm.h(0).cx(0, 1).cx(1, 2);

        // Apply non-Clifford rotations
        sv.rx(PI / 5.0, 0).rz(PI / 7.0, 1).rzz(PI / 9.0, 0, 2);
        dm.rx(PI / 5.0, 0).rz(PI / 7.0, 1).rzz(PI / 9.0, 0, 2);
    });
}

#[test]
fn test_prepare_computational_basis_consistency() {
    for num_qubits in 1..=3 {
        // Test each computational basis state
        for i in 0..(1 << num_qubits) {
            let mut sv = StateVec::new(num_qubits);
            let mut dm = DensityMatrix::new(num_qubits);
            let mut stab = StdSparseStab::new(num_qubits);

            // For stabilizer simulator, we need to manually apply X gates
            // based on the bits of i
            for q in 0..num_qubits {
                if (i >> q) & 1 == 1 {
                    stab.x(q);
                }
            }

            // Use prepare_computational_basis for StateVec and DensityMatrix
            sv.prepare_computational_basis(i);
            dm.prepare_computational_basis(i);

            // Compare the states
            compare_all_probabilities(&sv, &dm, &stab, num_qubits);
        }
    }
}

#[test]
fn test_prepare_plus_states_consistency() {
    // Test |+⟩ state preparation using H gates
    for num_qubits in 1..=3 {
        let mut sv = StateVec::new(num_qubits);
        let mut dm = DensityMatrix::new(num_qubits);
        let mut stab = StdSparseStab::new(num_qubits);

        // Apply H to all qubits
        for q in 0..num_qubits {
            sv.h(q);
            dm.h(q);
            stab.h(q);
        }

        // Compare the states
        compare_all_probabilities(&sv, &dm, &stab, num_qubits);
    }
}

#[test]
fn test_reset_consistency() {
    // Test reset behavior
    for num_qubits in 1..=3 {
        let mut sv = StateVec::new(num_qubits);
        let mut dm = DensityMatrix::new(num_qubits);
        let mut stab = StdSparseStab::new(num_qubits);

        // Apply some gates to get to a non-trivial state
        sv.h(0);
        dm.h(0);
        stab.h(0);

        if num_qubits > 1 {
            sv.cx(0, 1);
            dm.cx(0, 1);
            stab.cx(0, 1);
        }

        // Reset all simulators
        sv.reset();
        dm.reset();
        stab.reset();

        // Compare the states
        compare_all_probabilities(&sv, &dm, &stab, num_qubits);
    }
}
