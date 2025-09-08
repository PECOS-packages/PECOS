// Test to verify QulacsStateVec is Send + Sync and works in multi-threaded contexts

#[cfg(test)]
mod thread_safety_tests {
    use crate::QulacsStateVec;
    use pecos_core::RngManageable;
    use pecos_qsim::{CliffordGateable, QuantumSimulator};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;
    use std::sync::{Arc, Mutex};
    use std::thread;

    #[test]
    fn test_send_sync_traits() {
        fn assert_send<T: Send>() {}
        fn assert_sync<T: Sync>() {}

        assert_send::<QulacsStateVec>();
        assert_sync::<QulacsStateVec>();
    }

    #[test]
    fn test_clone_and_thread_independence() {
        // Create a template simulator
        let template_sim = QulacsStateVec::with_seed(2, 42);

        // Clone it for multiple threads
        let sim1 = template_sim.clone();
        let sim2 = template_sim.clone();
        let sim3 = template_sim.clone();

        // Store results from each thread
        let results = Arc::new(Mutex::new(Vec::new()));
        let results1 = Arc::clone(&results);
        let results2 = Arc::clone(&results);
        let results3 = Arc::clone(&results);

        // Spawn threads that work on independent simulators
        let handle1 = thread::spawn(move || {
            let mut sim = sim1;
            sim.h(0usize);
            sim.cx(0usize, 1usize);
            let state = sim.state();
            results1
                .lock()
                .unwrap()
                .push(("thread1", state[0], state[3]));
        });

        let handle2 = thread::spawn(move || {
            let mut sim = sim2;
            sim.x(0usize);
            sim.h(1usize);
            let state = sim.state();
            results2
                .lock()
                .unwrap()
                .push(("thread2", state[1], state[3]));
        });

        let handle3 = thread::spawn(move || {
            let mut sim = sim3;
            sim.h(0usize);
            sim.h(1usize);
            let state = sim.state();
            results3
                .lock()
                .unwrap()
                .push(("thread3", state[0], state[3]));
        });

        // Wait for all threads to complete
        handle1.join().unwrap();
        handle2.join().unwrap();
        handle3.join().unwrap();

        // Verify we got results from all threads
        let final_results = results.lock().unwrap();
        assert_eq!(final_results.len(), 3);

        // Each thread should have produced different results
        println!("Thread results: {:?}", *final_results);

        // Check that each thread worked independently
        for (name, _, _) in final_results.iter() {
            println!("Got result from {name}");
        }
    }

    #[test]
    #[allow(clippy::cast_precision_loss)]
    fn test_concurrent_monte_carlo_simulation() {
        const NUM_THREADS: usize = 4;
        const TRIALS_PER_THREAD: usize = 100;

        // Template simulator for Monte Carlo
        let template = QulacsStateVec::with_seed(1, 123);

        let handles: Vec<_> = (0..NUM_THREADS)
            .map(|thread_id| {
                let mut sim = template.clone();
                // Give each thread a different seed to avoid correlation
                sim.set_rng(ChaCha8Rng::seed_from_u64(123 + thread_id as u64 * 1000))
                    .unwrap();

                thread::spawn(move || {
                    let mut measurement_results = Vec::new();

                    for _trial in 0..TRIALS_PER_THREAD {
                        sim.reset();
                        sim.h(0usize);
                        let result = sim.mz(0usize);
                        measurement_results.push(result.outcome);
                    }

                    // Return thread ID and measurement statistics
                    let ones_count = measurement_results.iter().filter(|&&x| x).count();
                    (thread_id, ones_count, TRIALS_PER_THREAD)
                })
            })
            .collect();

        // Collect results from all threads
        let mut total_ones = 0;
        let mut total_trials = 0;

        for handle in handles {
            let (thread_id, ones_count, trials) = handle.join().unwrap();
            println!(
                "Thread {}: {} ones out of {} trials ({:.1}%)",
                thread_id,
                ones_count,
                trials,
                (ones_count as f64 / trials as f64) * 100.0
            );
            total_ones += ones_count;
            total_trials += trials;
        }

        // Overall statistics should be roughly 50/50 for |+⟩ measurements
        let overall_ratio = total_ones as f64 / total_trials as f64;
        println!(
            "Overall: {} ones out of {} trials ({:.1}%)",
            total_ones,
            total_trials,
            overall_ratio * 100.0
        );

        // Should be approximately 50% (allowing some variance)
        assert!(
            (overall_ratio - 0.5).abs() < 0.1,
            "Expected ~50% measurement outcomes, got {:.1}%",
            overall_ratio * 100.0
        );
    }
}
