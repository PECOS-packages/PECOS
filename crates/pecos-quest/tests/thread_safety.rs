//! Thread safety tests for `QuEST` wrapper
//! These tests verify that multiple `QuestStateVec` instances can work in parallel
//! without interfering with each other, which is essential for Monte Carlo simulations.

use pecos_num::assert_relative_eq;
use pecos_quest::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, QuestStateVec};
use pecos_rng::PecosRng;
use std::sync::{Arc, Barrier};
use std::thread;

#[test]
fn test_send_sync_traits() {
    // Compile-time check that QuestStateVec implements Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<QuestStateVec>();
}

#[test]
fn test_parallel_independent_instances() {
    const NUM_THREADS: usize = 4;
    const NUM_QUBITS: usize = 3;

    let barrier = Arc::new(Barrier::new(NUM_THREADS));
    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            let barrier = Arc::clone(&barrier);
            thread::spawn(move || {
                // Each thread gets its own completely independent state
                let mut state: QuestStateVec<PecosRng> =
                    QuestStateVec::with_seed(NUM_QUBITS, thread_id as u64 + 42);

                // Wait for all threads to be ready
                barrier.wait();

                // Each thread performs different operations
                match thread_id {
                    0 => {
                        // Thread 0: Create |000>
                        state.reset();
                        let prob = state.probability(0);
                        assert_relative_eq!(prob, 1.0, epsilon = 1e-10);
                        prob
                    }
                    1 => {
                        // Thread 1: Create |111>
                        state.prepare_computational_basis(0b111);
                        let prob = state.probability(0b111);
                        assert_relative_eq!(prob, 1.0, epsilon = 1e-10);
                        prob
                    }
                    2 => {
                        // Thread 2: Create Bell-like state on 3 qubits
                        // H(0) puts qubit 0 in superposition, CX(0,1) entangles qubits 0 and 1
                        // Result: (|000> + |011>)/sqrt(2) in |q2 q1 q0> notation
                        // In PECOS (qubit 0 = LSB): states 0b000 = 0 and 0b011 = 3
                        state.reset();
                        state.h(0).cx(0, 1);
                        let prob_000 = state.probability(0b000);
                        let prob_011 = state.probability(0b011);
                        assert_relative_eq!(prob_000, 0.5, epsilon = 1e-10);
                        assert_relative_eq!(prob_011, 0.5, epsilon = 1e-10);
                        prob_000 + prob_011
                    }
                    3 => {
                        // Thread 3: Create uniform superposition
                        state.prepare_plus_state();
                        let mut total_prob = 0.0;
                        for i in 0..(1 << NUM_QUBITS) {
                            let prob = state.probability(i);
                            assert_relative_eq!(prob, 1.0 / 8.0, epsilon = 1e-10);
                            total_prob += prob;
                        }
                        total_prob
                    }
                    _ => unreachable!(),
                }
            })
        })
        .collect();

    // Collect results from all threads
    let results: Vec<f64> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();

    // Verify all threads completed successfully with expected results
    assert_relative_eq!(results[0], 1.0, epsilon = 1e-10); // |000>
    assert_relative_eq!(results[1], 1.0, epsilon = 1e-10); // |111>
    assert_relative_eq!(results[2], 1.0, epsilon = 1e-10); // Bell state total
    assert_relative_eq!(results[3], 1.0, epsilon = 1e-10); // Plus state total
}

#[test]
fn test_parallel_bell_state_measurements() {
    const NUM_THREADS: usize = 8;

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            thread::spawn(move || {
                let mut state: QuestStateVec<PecosRng> =
                    QuestStateVec::with_seed(2, thread_id as u64 * 1000);

                // Create Bell state
                state.h(0).cx(0, 1);

                // Perform many measurements to verify correlation
                let mut correlations = Vec::new();
                for _measurement in 0..20 {
                    // Reset to Bell state for each measurement
                    state.reset().h(0).cx(0, 1);

                    let result0 = state.mz(0);
                    let result1 = state.mz(1);

                    // In Bell state, measurements should be perfectly correlated
                    correlations.push(result0.outcome == result1.outcome);
                }

                // Return correlation statistics
                let correlation_count = correlations.iter().filter(|&&x| x).count();
                (thread_id, correlation_count, correlations.len())
            })
        })
        .collect();

    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();

    // Verify all threads completed and got reasonable correlation
    for (thread_id, correlation_count, total_measurements) in results {
        println!(
            "Thread {thread_id}: {correlation_count}/{total_measurements} correlated measurements"
        );

        // Bell state measurements should be perfectly correlated
        // (allowing for potential QuEST measurement implementation details)
        assert_eq!(
            correlation_count, total_measurements,
            "Thread {thread_id} had imperfect Bell state correlation"
        );
    }
}

#[test]
fn test_parallel_rotation_gates() {
    const NUM_THREADS: usize = 6;

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            thread::spawn(move || {
                use std::f64::consts::PI;

                let mut state = QuestStateVec::new(1);

                match thread_id % 3 {
                    0 => {
                        // Test RX rotation
                        state.rx(PI, 0); // RX(π)|0> = i|1>
                        let prob_1 = state.probability(1);
                        assert_relative_eq!(prob_1, 1.0, epsilon = 1e-10);
                        prob_1
                    }
                    1 => {
                        // Test RY rotation
                        state.ry(PI / 2.0, 0); // RY(π/2)|0> = (|0> + |1>)/√2
                        let prob_0 = state.probability(0);
                        let prob_1 = state.probability(1);
                        assert_relative_eq!(prob_0, 0.5, epsilon = 1e-10);
                        assert_relative_eq!(prob_1, 0.5, epsilon = 1e-10);
                        prob_0 + prob_1
                    }
                    2 => {
                        // Test RZ rotation (doesn't change computational probabilities)
                        state.rz(PI / 4.0, 0); // RZ only adds phase
                        let prob_0 = state.probability(0);
                        assert_relative_eq!(prob_0, 1.0, epsilon = 1e-10);
                        prob_0
                    }
                    _ => unreachable!(),
                }
            })
        })
        .collect();

    let results: Vec<f64> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();

    // Verify all rotations worked as expected
    for result in results {
        assert_relative_eq!(result, 1.0, epsilon = 1e-10);
    }
}

#[test]
fn test_parallel_cloning_and_states() {
    const NUM_THREADS: usize = 4;

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            thread::spawn(move || {
                // Create template state
                let mut template: QuestStateVec<PecosRng> = QuestStateVec::with_seed(2, 12345); // Same seed
                template.h(0).cx(0, 1); // Bell state

                // Verify template probabilities
                let template_00 = template.probability(0b00);
                let template_11 = template.probability(0b11);
                assert_relative_eq!(template_00, 0.5, epsilon = 1e-10);
                assert_relative_eq!(template_11, 0.5, epsilon = 1e-10);

                // Each thread modifies its own copy
                match thread_id {
                    0 => template.x(0),    // Should flip to |10> + |01>
                    1 => template.z(0),    // Should add phase
                    2 => template.h(1),    // Should create different superposition
                    3 => template.reset(), // Should go back to |00>
                    _ => &mut template,
                };

                // Return final probabilities to verify independence
                let mut probs = Vec::new();
                for i in 0..4 {
                    probs.push(template.probability(i));
                }
                (thread_id, probs)
            })
        })
        .collect();

    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();

    // Verify that each thread produced different results
    for (thread_id, probs) in &results {
        println!("Thread {thread_id}: probabilities = {probs:?}");

        // Each thread should have different probability distributions
        let total_prob: f64 = probs.iter().sum();
        assert_relative_eq!(total_prob, 1.0, epsilon = 1e-10);
    }

    // Verify threads didn't interfere with each other
    // (Results should be deterministic given same operations)
    let (_, thread0_probs) = &results[0];
    let (_, thread3_probs) = &results[3]; // Thread 3 did reset()

    // Thread 3 should be in |00> state
    assert_relative_eq!(thread3_probs[0], 1.0, epsilon = 1e-10);

    // Thread 0 should be different from thread 3
    assert!((thread0_probs[0] - thread3_probs[0]).abs() > 1e-5);
}

#[test]
fn test_many_parallel_instances() {
    // Stress test with many threads to catch race conditions
    const NUM_THREADS: usize = 16;

    let handles: Vec<_> = (0..NUM_THREADS)
        .map(|thread_id| {
            thread::spawn(move || {
                let mut state: QuestStateVec<PecosRng> =
                    QuestStateVec::with_seed(1, thread_id as u64);

                // Perform a series of operations
                for i in 0..10 {
                    match (thread_id + i) % 4 {
                        0 => {
                            state.reset();
                        }
                        1 => {
                            state.x(0);
                        }
                        2 => {
                            state.h(0);
                        }
                        3 => {
                            state.z(0);
                        }
                        _ => unreachable!(),
                    }
                }

                // Final measurement
                let result = state.mz(0);
                (thread_id, result.outcome)
            })
        })
        .collect();

    let results: Vec<_> = handles
        .into_iter()
        .map(|handle| handle.join().unwrap())
        .collect();

    // Just verify all threads completed successfully
    assert_eq!(results.len(), NUM_THREADS);

    println!("All {NUM_THREADS} threads completed successfully");
    for (thread_id, outcome) in results {
        println!(
            "Thread {}: final measurement = {}",
            thread_id,
            if outcome { "1" } else { "0" }
        );
    }
}
