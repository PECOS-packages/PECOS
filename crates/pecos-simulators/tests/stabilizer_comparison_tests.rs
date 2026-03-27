use pecos_core::{QubitId, qid};
use pecos_simulators::CliffordGateable;
use pecos_simulators::DensityMatrix;
use pecos_simulators::QuantumSimulator;
use pecos_simulators::SparseStab;

// Helper function to check if two probabilities are close enough
fn assert_probs_equal(p1: f64, p2: f64) {
    assert!(
        (p1 - p2).abs() < 1e-10,
        "Probabilities differ: {p1} vs {p2}"
    );
}

// Helper function to compare the results of multiple basis states between simulators
fn compare_probabilities(dm: &mut DensityMatrix, stab: &SparseStab, num_qubits: usize) {
    // For SparseStab, we can only compute probabilities for computational basis states
    // by measuring the Z operator on each qubit

    // We'll compare the probability of measuring all combinations of 0s and 1s
    for i in 0..(1 << num_qubits) {
        let dm_prob = dm.probability(i);

        // For the stabilizer simulator, we need to determine the probability by
        // examining the stabilizer generators

        // We'll simulate measuring each qubit in the Z basis with a fixed outcome
        // corresponding to the bits of i
        let mut stab_prob = 1.0; // Start with probability 1.0
        let mut probability_is_zero = false;

        // Create a fresh copy for each basis state
        let mut stab_copy = stab.clone();

        for qubit in 0..num_qubits {
            // Check if we want bit qubit to be 0 or 1
            let bit_is_one = (i >> qubit) & 1 == 1;

            // Try to force the measurement to the desired outcome
            let result = stab_copy.mz_forced(qubit, bit_is_one);

            // If this was a non-deterministic measurement, the probability is 0.5
            if !result.is_deterministic {
                stab_prob *= 0.5;
            } else if result.outcome != bit_is_one {
                // If deterministic but different from what we want, probability is 0
                probability_is_zero = true;
                break;
            }
            // If deterministic and already equal to what we want, probability unchanged
        }

        let stab_prob = if probability_is_zero { 0.0 } else { stab_prob };

        // Compare the probabilities
        assert_probs_equal(dm_prob, stab_prob);
    }
}

#[test]
fn test_compare_initial_state() {
    // Test that both simulators start in the |0...0⟩ state
    let num_qubits = 2;
    let stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_x_gate() {
    // Test X gates give identical results
    let num_qubits = 1;
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply X to qubit 0
    stab.x(&qid(0));
    dm.x(&qid(0));

    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_pauli_gates() {
    // Test all Pauli gates give identical results
    let num_qubits = 1;

    // Test X gate
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.x(&qid(0));
    dm.x(&qid(0));
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Test Y gate
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.y(&qid(0));
    dm.y(&qid(0));
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Test Z gate
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.z(&qid(0));
    dm.z(&qid(0));
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Test combinations of Pauli gates
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.x(&qid(0)).z(&qid(0));
    dm.x(&qid(0)).z(&qid(0));
    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_hadamard() {
    // Test Hadamard gates give identical results
    let num_qubits = 1;
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply H to qubit 0
    stab.h(&qid(0));
    dm.h(&qid(0));

    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_sz_gate() {
    // Test S gate (sqrt of Z) gives identical results
    let num_qubits = 1;
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Put qubit in superposition first
    stab.h(&qid(0));
    dm.h(&qid(0));

    // Apply S gate
    stab.sz(&qid(0));
    dm.sz(&qid(0));

    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_phase_gates() {
    // Test various phase gates
    let num_qubits = 1;

    // Test S gate
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).sz(&qid(0));
    dm.h(&qid(0)).sz(&qid(0));
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Test S† gate
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).szdg(&qid(0));
    dm.h(&qid(0)).szdg(&qid(0));
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Test combined phases
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).sz(&qid(0)).sz(&qid(0)); // S^2 = Z
    dm.h(&qid(0)).sz(&qid(0)).sz(&qid(0));
    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_bell_state() {
    // Test creating a Bell state gives identical results
    let num_qubits = 2;
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply sequence of gates to create a Bell state
    stab.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
    dm.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);

    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_two_qubit_gates() {
    // Test two-qubit gates give identical results
    let num_qubits = 2;

    // Test CNOT (CX) gate
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
    dm.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Test CZ gate
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).h(&qid(1)).cz(&[(QubitId(0), QubitId(1))]);
    dm.h(&qid(0)).h(&qid(1)).cz(&[(QubitId(0), QubitId(1))]);
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Test SWAP gate
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.x(&qid(0)).swap(&[(QubitId(0), QubitId(1))]);
    dm.x(&qid(0)).swap(&[(QubitId(0), QubitId(1))]);
    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_complex_circuit() {
    // Test a more complex Clifford circuit
    let num_qubits = 3;
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Create a GHZ state
    stab.h(&qid(0))
        .cx(&[(QubitId(0), QubitId(1))])
        .cx(&[(QubitId(1), QubitId(2))]);
    dm.h(&qid(0))
        .cx(&[(QubitId(0), QubitId(1))])
        .cx(&[(QubitId(1), QubitId(2))]);

    compare_probabilities(&mut dm, &stab, num_qubits);

    // Apply more Clifford gates
    stab.x(&qid(0)).h(&qid(1)).z(&qid(2));
    dm.x(&qid(0)).h(&qid(1)).z(&qid(2));

    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_measurements() {
    // Test measurement behavior with fixed seed
    let num_qubits = 1;
    let seed = 42; // Fixed seed for deterministic behavior

    let mut stab = SparseStab::with_seed(num_qubits, seed);
    let mut dm = DensityMatrix::with_seed(num_qubits, seed);

    // Put qubits in superposition
    stab.h(&qid(0));
    dm.h(&qid(0));

    // With identical seeds, measurements should give identical results
    let stab_result = stab.mz(&qid(0)).into_iter().next().unwrap();
    let dm_result = dm.mz(&qid(0)).into_iter().next().unwrap();

    assert_eq!(
        stab_result.outcome, dm_result.outcome,
        "Measurement outcomes differ despite using the same seed"
    );
    assert_eq!(
        stab_result.is_deterministic, dm_result.is_deterministic,
        "Determinism flags differ despite using the same seed"
    );
}

#[test]
fn test_compare_prepare_z() {
    // Test computational basis preparation using reset + gates instead of special methods
    let num_qubits = 2;

    // Test |00⟩ state - already the default state
    let stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Test |10⟩ state
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.x(&qid(0));
    dm.x(&qid(0));
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Test |11⟩ state
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.x(&qid(0)).x(&qid(1));
    dm.x(&qid(0)).x(&qid(1));
    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_prepare_x() {
    // Test |+⟩ state preparation using standard gate operations
    let num_qubits = 2;

    // Create |++⟩ state using Hadamard gates instead of direct methods
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).h(&qid(1));
    dm.h(&qid(0)).h(&qid(1));
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Test |--⟩ state
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).h(&qid(1)).z(&qid(0)).z(&qid(1)); // Apply Z after H to get |-⟩
    dm.h(&qid(0)).h(&qid(1)).z(&qid(0)).z(&qid(1));
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Test |+-⟩ state
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).h(&qid(1)).z(&qid(1)); // Apply Z to just qubit 1
    dm.h(&qid(0)).h(&qid(1)).z(&qid(1));
    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_prepare_y() {
    // Test |+i⟩ state preparation using standard gate operations
    let num_qubits = 2;

    // Create |+i,+i⟩ state using S and H gates
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).h(&qid(1)).sz(&qid(0)).sz(&qid(1)); // H followed by S gives |+i⟩
    dm.h(&qid(0)).h(&qid(1)).sz(&qid(0)).sz(&qid(1));
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Test |-i,-i⟩ state
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).h(&qid(1)).szdg(&qid(0)).szdg(&qid(1)); // H followed by S† gives |-i⟩
    dm.h(&qid(0)).h(&qid(1)).szdg(&qid(0)).szdg(&qid(1));
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Test |+i,-i⟩ state
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).h(&qid(1)).sz(&qid(0)).szdg(&qid(1));
    dm.h(&qid(0)).h(&qid(1)).sz(&qid(0)).szdg(&qid(1));
    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_reset() {
    // Test reset behavior
    let num_qubits = 2;
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Apply some gates to get to a non-trivial state
    stab.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
    dm.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);

    // Reset both simulators
    stab.reset();
    dm.reset();

    // Both should be in the |0...0⟩ state
    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_bell_states() {
    let num_qubits = 2;

    // Bell state |Phi+> = (|00> + |11>)/sqrt(2)
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
    dm.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Bell state |Phi-> = (|00> - |11>)/sqrt(2)
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]).z(&qid(1));
    dm.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]).z(&qid(1));
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Bell state |Psi+> = (|01> + |10>)/sqrt(2)
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]).x(&qid(1));
    dm.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]).x(&qid(1));
    compare_probabilities(&mut dm, &stab, num_qubits);

    // Bell state |Psi-> = (|01> - |10>)/sqrt(2)
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);
    stab.h(&qid(0))
        .cx(&[(QubitId(0), QubitId(1))])
        .z(&qid(0))
        .x(&qid(1));
    dm.h(&qid(0))
        .cx(&[(QubitId(0), QubitId(1))])
        .z(&qid(0))
        .x(&qid(1));
    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_ghz_state() {
    // Test GHZ state preparation and operations
    let num_qubits = 3;
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Create a GHZ state |000> + |111>
    stab.h(&qid(0))
        .cx(&[(QubitId(0), QubitId(1))])
        .cx(&[(QubitId(1), QubitId(2))]);
    dm.h(&qid(0))
        .cx(&[(QubitId(0), QubitId(1))])
        .cx(&[(QubitId(1), QubitId(2))]);

    compare_probabilities(&mut dm, &stab, num_qubits);

    // Apply X to all qubits, should get |111> + |000>
    stab.x(&qid(0)).x(&qid(1)).x(&qid(2));
    dm.x(&qid(0)).x(&qid(1)).x(&qid(2));

    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_w_state() {
    // Test W state preparation (|001> + |010> + |100>)
    // This is more complex but still within Clifford operations
    let num_qubits = 3;
    let mut stab = SparseStab::new(num_qubits);
    let mut dm = DensityMatrix::new(num_qubits);

    // Create a W state approximation using only Clifford gates
    // Start with |001>
    stab.x(&qid(2));
    dm.x(&qid(2));

    // Apply H to qubit 0 and 1
    stab.h(&qid(0)).h(&qid(1));
    dm.h(&qid(0)).h(&qid(1));

    // Apply CZ between qubits 0,2 and 1,2
    stab.cz(&[(QubitId(0), QubitId(2))])
        .cz(&[(QubitId(1), QubitId(2))]);
    dm.cz(&[(QubitId(0), QubitId(2))])
        .cz(&[(QubitId(1), QubitId(2))]);

    // Apply H again to qubit 0 and 1
    stab.h(&qid(0)).h(&qid(1));
    dm.h(&qid(0)).h(&qid(1));

    compare_probabilities(&mut dm, &stab, num_qubits);
}

#[test]
fn test_compare_mixed_basis_measurements() {
    // Test measuring in different bases
    let num_qubits = 1;
    let seed = 42;

    // Test X-basis measurements
    let mut stab = SparseStab::with_seed(num_qubits, seed);
    let mut dm = DensityMatrix::with_seed(num_qubits, seed);

    // Prepare |0> state (default)

    // Apply H gate to get |+> state
    stab.h(&qid(0));
    dm.h(&qid(0));

    // Measure in X basis
    let stab_result = stab.mx(&qid(0)).into_iter().next().unwrap();
    let dm_result = dm.mx(&qid(0)).into_iter().next().unwrap();

    assert_eq!(stab_result.outcome, dm_result.outcome);
    assert_eq!(stab_result.is_deterministic, dm_result.is_deterministic);

    // Test Y-basis measurements
    let mut stab = SparseStab::with_seed(num_qubits, seed);
    let mut dm = DensityMatrix::with_seed(num_qubits, seed);

    // Prepare |0> and then apply H and S to get to Y basis state
    stab.h(&qid(0)).sz(&qid(0));
    dm.h(&qid(0)).sz(&qid(0));

    // Measure in Y basis
    let stab_result = stab.my(&qid(0)).into_iter().next().unwrap();
    let dm_result = dm.my(&qid(0)).into_iter().next().unwrap();

    assert_eq!(stab_result.outcome, dm_result.outcome);
    assert_eq!(stab_result.is_deterministic, dm_result.is_deterministic);
}
