use pecos_rng::RngPcg;

use std::sync::Arc;
use std::thread;

#[test]
fn test_pcg_random_generates_values() {
    // Test that random() generates different values
    let mut rng_pcg = RngPcg::new();
    let val1 = rng_pcg.random();
    let val2 = rng_pcg.random();

    // It's extremely unlikely that two consecutive calls return the same value
    // (probability is 1 in 2^32)
    assert_ne!(
        val1, val2,
        "Two consecutive random() calls should generate different values"
    );
}

#[test]
fn test_pcg_bounded_random() {
    // Test bounded random with various bounds
    let bounds = [1, 2, 10, 100, 1000];

    let mut pcg = RngPcg::new();
    for bound in bounds {
        for _ in 0..10 {
            let val = pcg.boundedrand(bound);
            assert!(
                val < bound,
                "boundedrand({bound}) returned {val}, which is >= {bound}"
            );
        }
    }
}

#[test]
fn test_pcg_frandom_range() {
    // Test that frandom returns values in [0.0, 1.0)
    let mut pcg = RngPcg::new();
    for _ in 0..100 {
        let val = pcg.frandom();
        assert!(val >= 0.0, "frandom() returned {val}, which is < 0.0");
        assert!(val < 1.0, "frandom() returned {val}, which is >= 1.0");
    }
}

#[test]
fn test_pcg_seeding() {
    // Test that seeding produces deterministic sequences
    // Note: PCG uses global state, so we test determinism directly

    // Test that the same seed produces the same first few values
    let mut pcg = RngPcg::new();
    pcg.srandom(12345);
    let first_val_1 = pcg.random();
    let second_val_1 = pcg.random();

    pcg.srandom(12345);
    let first_val_2 = pcg.random();
    let second_val_2 = pcg.random();

    assert_eq!(
        first_val_1, first_val_2,
        "First value after seeding should be deterministic"
    );
    assert_eq!(
        second_val_1, second_val_2,
        "Second value after seeding should be deterministic"
    );

    // Test that different seeds produce different values
    pcg.srandom(54321);
    let different_first = pcg.random();

    assert_ne!(
        first_val_1, different_first,
        "Different seeds should produce different values"
    );
}

#[test]
fn test_pcg_deterministic_behavior() {
    // Test that the RNG is deterministic after seeding
    let mut pcg = RngPcg::new();
    pcg.srandom(999);
    let first_value = pcg.random();

    pcg.srandom(999);
    let second_value = pcg.random();

    assert_eq!(
        first_value, second_value,
        "First value after seeding should be deterministic"
    );
}

#[test]
fn test_pcg_shared_state_interference() {
    // This test is more likely to fail when run in parallel with other tests
    // because they all share the same global RNG state

    const ITERATIONS: usize = 100;
    let mut results: Vec<u32> = Vec::new();
    let mut pcg = RngPcg::new();

    for i in 0..ITERATIONS {
        pcg.srandom(42);
        // Add some delay to increase chance of interference
        std::thread::yield_now();

        let val = pcg.random();
        results.push(val);

        assert!(
            !(i > 0 && results[i] != results[0]),
            "Iteration {}: Expected deterministic value {} but got {} (shared state interference detected!)",
            i,
            results[0],
            val
        );
    }
}

#[test]
fn test_pcg_rapid_reseeding() {
    // Rapidly reseed and check values to increase chance of race conditions
    let mut pcg = RngPcg::new();
    let expected_values: Vec<u32> = (0..10)
        .map(|i| {
            pcg.srandom(i);
            pcg.random()
        })
        .collect();

    // Now verify multiple times
    for round in 0..50 {
        for (i, &expected) in expected_values.iter().enumerate() {
            pcg.srandom(i as u64);
            let actual = pcg.random();
            assert_eq!(
                actual, expected,
                "Round {round}, seed {i}: Expected {expected} but got {actual} (state corruption detected!)"
            );
        }
    }
}

#[test]
fn test_pcg_concurrent_access() {
    // This test verifies that threads maintain independent sequences even when running concurrently

    let num_threads = 10;
    let iterations_per_thread = 100;
    let barrier = Arc::new(std::sync::Barrier::new(num_threads));

    let mut pcg = RngPcg::new();
    // First, generate expected sequences for each thread
    let expected_sequences: Vec<Vec<u32>> = (0..num_threads)
        .map(|thread_id| {
            pcg.srandom(thread_id as u64);
            (0..iterations_per_thread).map(|_| pcg.random()).collect()
        })
        .collect();

    let handles: Vec<_> = (0..num_threads)
        .map(|thread_id| {
            let barrier = Arc::clone(&barrier);
            let expected_seq = expected_sequences[thread_id].clone();

            thread::spawn(move || {
                // Wait for all threads to be ready
                barrier.wait();

                // Each thread uses its own seed
                pcg.srandom(thread_id as u64);

                let mut results = Vec::new();
                #[allow(clippy::needless_range_loop)]
                for i in 0..iterations_per_thread {
                    let val = pcg.random();
                    results.push(val);

                    // Verify we're getting the expected value
                    assert_eq!(
                        val, expected_seq[i],
                        "Thread {thread_id} iteration {i}: expected {} but got {val}",
                        expected_seq[i]
                    );

                    // Yield to increase chance of interleaving
                    if i % 10 == 0 {
                        thread::yield_now();
                    }
                }

                results
            })
        })
        .collect();

    // Collect all results and verify
    for (thread_id, handle) in handles.into_iter().enumerate() {
        let results = handle.join().unwrap();
        assert_eq!(
            results, expected_sequences[thread_id],
            "Thread {thread_id} produced unexpected sequence"
        );
    }
}

#[test]
fn test_pcg_thread_independence() {
    // Verify that threads with different seeds maintain independent sequences

    let mut pcg = RngPcg::new();
    // First, get expected sequences
    pcg.srandom(100);
    let expected_seq1: Vec<u32> = (0..5).map(|_| pcg.random()).collect();

    pcg.srandom(200);
    let expected_seq2: Vec<u32> = (0..5).map(|_| pcg.random()).collect();

    // Run threads concurrently
    let handle1 = thread::spawn(move || {
        pcg.srandom(100);
        let mut results = vec![];
        for _ in 0..5 {
            results.push(pcg.random());
            thread::yield_now(); // Encourage interleaving
        }
        results
    });

    let handle2 = thread::spawn(move || {
        pcg.srandom(200);
        let mut results = vec![];
        for _ in 0..5 {
            results.push(pcg.random());
            thread::yield_now(); // Encourage interleaving
        }
        results
    });

    let thread1_results = handle1.join().unwrap();
    let thread2_results = handle2.join().unwrap();

    assert_eq!(
        thread1_results, expected_seq1,
        "Thread 1 maintains independent sequence"
    );
    assert_eq!(
        thread2_results, expected_seq2,
        "Thread 2 maintains independent sequence"
    );
}
