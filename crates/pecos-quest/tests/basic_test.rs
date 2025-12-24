//! Basic tests for the `QuEST` wrapper using PECOS-style API

use pecos_num::assert_relative_eq;
use pecos_quest::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, QuestStateVec};
use pecos_rng::PecosRng;

#[test]
fn test_state_creation() {
    let state = QuestStateVec::new(5);
    assert_eq!(state.num_qubits(), 5);

    // Check that initial state is |00000>
    let prob = state.probability(0);
    assert_relative_eq!(prob, 1.0, epsilon = 1e-10);
}

#[test]
fn test_state_with_seed() {
    let state1: QuestStateVec<PecosRng> = QuestStateVec::with_seed(3, 42);
    let state2: QuestStateVec<PecosRng> = QuestStateVec::with_seed(3, 42);

    assert_eq!(state1.num_qubits(), 3);
    assert_eq!(state2.num_qubits(), 3);

    // Both should be in the same initial state
    assert_relative_eq!(
        state1.probability(0),
        state2.probability(0),
        epsilon = 1e-10
    );
}

#[test]
fn test_computational_basis_preparation() {
    let mut state = QuestStateVec::new(2);

    // Prepare |01> (binary 10 = decimal 2)
    state.prepare_computational_basis(0b10);

    assert_relative_eq!(state.probability(0b00), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(0b01), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(0b10), 1.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(0b11), 0.0, epsilon = 1e-10);
}

#[test]
fn test_plus_state_preparation() {
    let mut state = QuestStateVec::new(2);
    state.prepare_plus_state();

    // Each basis state should have probability 1/4
    let expected_prob = 0.25;
    for i in 0..4 {
        assert_relative_eq!(state.probability(i), expected_prob, epsilon = 1e-10);
    }
}

#[test]
fn test_state_access() {
    let state = QuestStateVec::new(2);

    // Initially |00>
    // Check amplitude of |00>
    let amp0 = state.get_amplitude(0);
    assert_relative_eq!(amp0.re, 1.0, epsilon = 1e-10);
    assert_relative_eq!(amp0.im, 0.0, epsilon = 1e-10);

    // Check other amplitudes are zero
    for i in 1..4 {
        let amp = state.get_amplitude(i);
        assert_relative_eq!(amp.re, 0.0, epsilon = 1e-10);
        assert_relative_eq!(amp.im, 0.0, epsilon = 1e-10);
    }
}

#[test]
fn test_reset() {
    let mut state = QuestStateVec::new(2);

    // Change the state
    state.prepare_computational_basis(3);
    assert_relative_eq!(state.probability(3), 1.0, epsilon = 1e-10);

    // Reset should bring back to |00>
    state.reset();
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(3), 0.0, epsilon = 1e-10);
}

#[test]
fn test_pauli_gates() {
    let mut state = QuestStateVec::new(1);

    // Test Pauli-X: |0> -> |1>
    state.reset();
    state.x(0);
    assert_relative_eq!(state.probability(0), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(1), 1.0, epsilon = 1e-10);

    // Test Pauli-Z on |1>: should add phase but not change probabilities
    state.z(0);
    assert_relative_eq!(state.probability(0), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(1), 1.0, epsilon = 1e-10);

    // Test Pauli-Y: X*Z = iY, so after X then Z, we should have i|1>
    // Probability should still be 1 for |1>
    state.reset().x(0).y(0);
    // Y|1> = -i|0>, so we should be in |0>
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(1), 0.0, epsilon = 1e-10);
}

#[test]
fn test_hadamard_gate() {
    let mut state = QuestStateVec::new(1);

    // H|0> = |+> = (|0> + |1>)/sqrt(2)
    state.h(0);

    let expected_prob = 0.5;
    assert_relative_eq!(state.probability(0), expected_prob, epsilon = 1e-10);
    assert_relative_eq!(state.probability(1), expected_prob, epsilon = 1e-10);
}

#[test]
fn test_s_gates() {
    let mut state = QuestStateVec::new(1);

    // S|0> = |0>, probability unchanged
    state.sz(0);
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);

    // S†S = I, so applying S then S† should be identity
    state.szdg(0);
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);
}

#[test]
fn test_cnot_gate() {
    let mut state = QuestStateVec::new(2);

    // CNOT|00> = |00>
    state.cx(0, 1);
    assert_relative_eq!(state.probability(0b00), 1.0, epsilon = 1e-10);

    // In PECOS convention (qubit 0 = LSB):
    // - State 0b01 has qubit 0 = 1 (control set), qubit 1 = 0
    // - State 0b10 has qubit 0 = 0 (control clear), qubit 1 = 1

    // Prepare state with control qubit 0 = 1, apply CNOT(0,1) -> target flips
    state.prepare_computational_basis(0b01); // qubit 0 = 1, qubit 1 = 0
    state.cx(0, 1); // control=0 is set, so target=1 flips: 0->1
    assert_relative_eq!(state.probability(0b11), 1.0, epsilon = 1e-10); // qubit 0 = 1, qubit 1 = 1

    // Prepare state with control qubit 0 = 0, apply CNOT(0,1) -> no change
    state.prepare_computational_basis(0b10); // qubit 0 = 0, qubit 1 = 1
    state.cx(0, 1); // control=0 is clear, target doesn't flip
    assert_relative_eq!(state.probability(0b10), 1.0, epsilon = 1e-10); // unchanged
}

#[test]
fn test_cz_gate() {
    let mut state = QuestStateVec::new(2);

    // CZ|00> = |00>
    state.cz(0, 1);
    assert_relative_eq!(state.probability(0b00), 1.0, epsilon = 1e-10);

    // CZ|11> = -|11> (same probability)
    state.prepare_computational_basis(0b11);
    state.cz(0, 1);
    assert_relative_eq!(state.probability(0b11), 1.0, epsilon = 1e-10);
}

#[test]
fn test_bell_state_creation() {
    let mut state = QuestStateVec::new(2);

    // Create Bell state: H(0) then CNOT(0,1)
    state.h(0).cx(0, 1);

    // Should have equal probability for |00> and |11>
    assert_relative_eq!(state.probability(0b00), 0.5, epsilon = 1e-10);
    assert_relative_eq!(state.probability(0b01), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(0b10), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(0b11), 0.5, epsilon = 1e-10);
}

#[test]
fn test_measurement() {
    let mut state = QuestStateVec::new(1);

    // Measure |0>
    let result = state.mz(0);
    assert!(!result.outcome); // |0> corresponds to false
    assert!(result.is_deterministic);

    // Measure |1>
    state.prepare_computational_basis(1);
    let result = state.mz(0);
    assert!(result.outcome); // |1> corresponds to true
    assert!(result.is_deterministic);

    // Measure superposition state
    state.reset().h(0);
    let result = state.mz(0);
    // Should not be deterministic (though this is probabilistic)
    // For a superposition state, measurement is non-deterministic
    assert!(!result.is_deterministic);
}

#[test]
fn test_rotation_gates() {
    use std::f64::consts::PI;

    let mut state = QuestStateVec::new(1);

    // RX(π) = -iX, so RX(π)|0> should give |1>
    state.rx(PI, 0);
    assert_relative_eq!(state.probability(0), 0.0, epsilon = 1e-10);
    assert_relative_eq!(state.probability(1), 1.0, epsilon = 1e-10);

    // RZ doesn't change computational basis probabilities
    state.reset();
    state.rz(PI / 2.0, 0);
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);

    // RY(π/2) should create superposition
    state.reset();
    state.ry(PI / 2.0, 0);
    assert_relative_eq!(state.probability(0), 0.5, epsilon = 1e-10);
    assert_relative_eq!(state.probability(1), 0.5, epsilon = 1e-10);
}

#[test]
fn test_t_gates() {
    let mut state = QuestStateVec::new(1);

    // T|0> = |0>, probability unchanged
    state.t(0);
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);

    // T†T = I, so applying T then T† should be identity
    state.tdg(0);
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);
}

#[test]
fn test_rzz_gate() {
    use std::f64::consts::PI;

    let mut state = QuestStateVec::new(2);

    // RZZ doesn't change computational basis probabilities
    state.rzz(PI / 2.0, 0, 1);
    assert_relative_eq!(state.probability(0), 1.0, epsilon = 1e-10);

    // Test on |11> state
    state.prepare_computational_basis(0b11);
    state.rzz(PI / 2.0, 0, 1);
    assert_relative_eq!(state.probability(0b11), 1.0, epsilon = 1e-10);
}

#[test]
fn test_method_chaining() {
    let mut state = QuestStateVec::new(2);

    // Test that all methods return &mut Self for chaining
    state
        .reset()
        .h(0)
        .cx(0, 1)
        .z(1)
        .rx(std::f64::consts::PI / 4.0, 0);

    // Just verify it compiles and runs
    assert_eq!(state.num_qubits(), 2);
}

#[test]
fn test_gpu_acceleration_status() {
    let state = QuestStateVec::new(2);
    let qureg_info = state.get_info();
    let env_info = state.get_env_info();

    // Print environment status for visibility
    println!("QuEST Environment Info:");
    println!("  Multithreaded: {}", env_info.is_multithreaded);
    println!("  GPU accelerated: {}", env_info.is_gpu_accelerated);
    println!("  Distributed: {}", env_info.is_distributed);
    println!("  Rank: {}", env_info.rank);
    println!("  Num nodes: {}", env_info.num_nodes);

    println!("\nQureg Info:");
    println!("  Number of qubits: {}", qureg_info.num_qubits);
    println!("  Number of amplitudes: {}", qureg_info.num_amps);
    println!("  Is density matrix: {}", qureg_info.is_density_matrix);

    // The direct QuestStateVec wrapper always uses CPU mode.
    // For GPU acceleration, use the engine builder with .with_gpu().
    // This is because the CUDA backend is loaded at runtime via dlopen,
    // allowing a single binary to work on systems with and without CUDA.
    assert!(
        !env_info.is_gpu_accelerated,
        "QuestStateVec should use CPU mode. GPU acceleration is only available \
         via the engine builder with .with_gpu()."
    );
    println!("\nINFO: QuestStateVec uses CPU mode (as expected)");
    println!("      For GPU acceleration, use quest_state_vec().with_gpu()");
}

/// Test the CUDA engine through the builder interface
#[cfg(feature = "cuda")]
#[test]
fn test_cuda_engine_builder() {
    use pecos_engines::{Engine, QuantumEngineBuilder, byte_message::ByteMessage};
    use pecos_quest::quest_state_vec;

    println!("\n=== Testing CUDA engine builder ===");

    // Test CPU mode first
    let mut cpu_builder = quest_state_vec().qubits(2);
    let mut cpu_engine = cpu_builder.build().expect("Failed to build CPU engine");
    println!("CPU engine created successfully");

    // Create a Bell state circuit: H(0), CNOT(0,1), measure both
    let mut msg_builder = ByteMessage::quantum_operations_builder();
    msg_builder.add_h(&[0]);
    msg_builder.add_cx(&[0], &[1]);
    msg_builder.add_measurements(&[0, 1]);
    let msg = msg_builder.build();

    let result = cpu_engine.process(msg.clone()).expect("CPU process failed");
    let outcomes = result.outcomes().expect("Failed to get outcomes");
    println!("CPU measurement outcomes: {outcomes:?}");

    // Verify Bell state outcomes (both qubits should match)
    assert!(
        outcomes.len() == 2,
        "Expected 2 measurement outcomes, got {}",
        outcomes.len()
    );
    assert_eq!(
        outcomes[0], outcomes[1],
        "Bell state outcomes should match: got {outcomes:?}"
    );

    // Now test GPU mode
    println!("\n=== Testing GPU mode ===");
    let mut gpu_builder = quest_state_vec().qubits(2).with_gpu();
    match gpu_builder.build() {
        Ok(mut gpu_engine) => {
            println!("GPU engine created successfully!");

            // Reset and run the same circuit
            gpu_engine.reset().expect("Reset failed");

            let mut msg_builder = ByteMessage::quantum_operations_builder();
            msg_builder.add_h(&[0]);
            msg_builder.add_cx(&[0], &[1]);
            msg_builder.add_measurements(&[0, 1]);
            let msg = msg_builder.build();

            let result = gpu_engine.process(msg).expect("GPU process failed");
            let outcomes = result.outcomes().expect("Failed to get outcomes");
            println!("GPU measurement outcomes: {outcomes:?}");

            // Verify Bell state outcomes
            assert!(
                outcomes.len() == 2,
                "Expected 2 measurement outcomes, got {}",
                outcomes.len()
            );
            assert_eq!(
                outcomes[0], outcomes[1],
                "Bell state outcomes should match: got {outcomes:?}"
            );

            println!("\nSUCCESS: CUDA engine works correctly!");
        }
        Err(e) => {
            println!("GPU engine build failed (expected if CUDA not available): {e}");
            // Not a failure - CUDA may not be available at runtime
        }
    }
}
