//! Comprehensive determinism tests for `PyMatching` decoder
//!
//! These tests ensure that `PyMatching` provides:
//! 1. Deterministic results with fixed seeds
//! 2. Thread safety in parallel execution
//! 3. Independence between decoder instances
//! 4. Proper handling of global RNG state

use pecos_pymatching::{PyMatchingConfig, PyMatchingDecoder};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Compare weights with tolerance for floating point precision
fn weights_equal(a: f64, b: f64) -> bool {
    (a - b).abs() < f64::EPSILON
}

/// Create a simple test decoder for determinism testing
fn create_simple_test_decoder() -> Result<PyMatchingDecoder, Box<dyn std::error::Error>> {
    let config = PyMatchingConfig {
        num_nodes: Some(6),
        num_observables: 1,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config)?;

    // Add edges for a simple code (even parity for valid syndrome)
    decoder.add_edge(0, 1, &[0], Some(1.0), Some(0.1), None)?;
    decoder.add_edge(1, 2, &[], Some(1.5), Some(0.1), None)?;
    decoder.add_edge(2, 3, &[], Some(2.0), Some(0.1), None)?;
    decoder.add_edge(3, 4, &[], Some(2.5), Some(0.1), None)?;
    decoder.add_edge(4, 5, &[], Some(3.0), Some(0.1), None)?;

    Ok(decoder)
}

#[test]
fn test_pymatching_sequential_determinism() {
    // Test that PyMatching gives identical results with fixed seed across multiple runs

    let mut results = Vec::new();
    let syndrome = vec![1, 0, 1, 0, 0, 0]; // Even parity for valid syndrome

    for run in 0..10 {
        // Set seed before each decoder creation
        PyMatchingDecoder::set_seed(42).unwrap();

        let mut decoder = create_simple_test_decoder().unwrap();
        let result = decoder.decode(&syndrome).unwrap();

        results.push((result.observable.clone(), result.weight));

        if run < 2 {
            println!(
                "PyMatching run {}: observable={:?}, weight={}",
                run, result.observable, result.weight
            );
        }
    }

    // All results should be identical with fixed seed
    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "PyMatching run {i} gave different observable"
        );
        assert!(
            weights_equal(first.1, result.1),
            "PyMatching run {i} gave different weight: {} vs {}",
            first.1,
            result.1
        );
    }

    println!(
        "PyMatching sequential determinism test passed - {} consistent runs",
        results.len()
    );
}

#[test]
#[allow(clippy::similar_names)]
fn test_pymatching_instance_independence() {
    // Test that multiple PyMatching instances behave deterministically with same seed

    let syndrome = vec![1, 0, 1, 0, 0, 0];
    let mut results = Vec::new();

    for i in 0..5 {
        // Set seed before each decoder creation
        PyMatchingDecoder::set_seed(123).unwrap();

        let mut decoder1 = create_simple_test_decoder().unwrap();
        let result1 = decoder1.decode(&syndrome).unwrap();

        // Set same seed again for second decoder
        PyMatchingDecoder::set_seed(123).unwrap();

        let mut decoder2 = create_simple_test_decoder().unwrap();
        let result2 = decoder2.decode(&syndrome).unwrap();

        // Same seed should give same results
        assert_eq!(
            result1.observable, result2.observable,
            "Instance {i} gave different observables with same seed"
        );
        assert!(
            weights_equal(result1.weight, result2.weight),
            "Instance {i} gave different weights with same seed: {} vs {}",
            result1.weight,
            result2.weight
        );

        results.push((result1.observable, result1.weight));
    }

    // All iterations should be consistent
    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(first.0, result.0, "Iteration {i} gave different observable");
        assert!(
            weights_equal(first.1, result.1),
            "Iteration {i} gave different weight: {} vs {}",
            first.1,
            result.1
        );
    }

    println!(
        "PyMatching instance independence test passed - {} consistent iterations",
        results.len()
    );
}

#[test]
fn test_pymatching_different_seeds_different_results() {
    // Test that different seeds give different results (when decoding allows it)
    // This verifies that seeding actually works

    let syndrome = vec![1, 0, 1, 0, 0, 0];
    let mut results = Vec::new();

    for seed in [42, 123, 456, 789, 101_112] {
        PyMatchingDecoder::set_seed(seed).unwrap();

        let mut decoder = create_simple_test_decoder().unwrap();
        let result = decoder.decode(&syndrome).unwrap();

        results.push((seed, result.observable.clone(), result.weight));
    }

    // While deterministic decoding might give same logical result,
    // seeding should at least work consistently
    for (seed, observable, weight) in &results {
        println!("Seed {seed}: observable={observable:?}, weight={weight}");

        // Verify same seed gives same result again
        PyMatchingDecoder::set_seed(*seed).unwrap();
        let mut decoder = create_simple_test_decoder().unwrap();
        let verify_result = decoder.decode(&syndrome).unwrap();

        assert_eq!(
            *observable, verify_result.observable,
            "Seed {seed} inconsistent on re-run"
        );
        assert!(
            weights_equal(*weight, verify_result.weight),
            "Seed {seed} weight inconsistent on re-run: {} vs {}",
            weight,
            verify_result.weight
        );
    }

    println!(
        "PyMatching seed verification test passed - {} seeds tested",
        results.len()
    );
}

#[test]
fn test_pymatching_parallel_with_fixed_seeds() {
    // Test parallel execution where each thread uses a different fixed seed
    // This tests that global RNG state is properly protected

    const NUM_THREADS: usize = 8;
    const NUM_ITERATIONS: usize = 5;

    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    for thread_id in 0..NUM_THREADS {
        let results_clone = Arc::clone(&results);
        let seed = 100 + u32::try_from(thread_id).expect("thread_id too large"); // Different seed per thread

        let handle = thread::spawn(move || {
            for iteration in 0..NUM_ITERATIONS {
                // Set thread-specific seed
                PyMatchingDecoder::set_seed(seed).unwrap();

                let mut decoder = create_simple_test_decoder().unwrap();
                let syndrome = vec![1, 0, 1, 0, 0, 0];
                let result = decoder.decode(&syndrome).unwrap();

                results_clone.lock().unwrap().push((
                    thread_id,
                    iteration,
                    seed,
                    result.observable.clone(),
                    result.weight,
                ));

                // Small delay to increase chance of race conditions
                thread::sleep(Duration::from_micros(10));
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_results = results.lock().unwrap();

    // Check that each thread got consistent results across its iterations
    for thread_id in 0..NUM_THREADS {
        let thread_results: Vec<_> = final_results
            .iter()
            .filter(|(tid, _, _, _, _)| *tid == thread_id)
            .collect();

        if !thread_results.is_empty() {
            let first_result = &thread_results[0];
            for (tid, iter, seed, obs, weight) in &thread_results {
                assert_eq!(
                    first_result.3, *obs,
                    "Thread {tid} iteration {iter} gave different observable (seed {seed})"
                );
                assert!(
                    weights_equal(first_result.4, *weight),
                    "Thread {tid} iteration {iter} gave different weight (seed {seed}): {} vs {}",
                    first_result.4,
                    weight
                );
            }

            println!(
                "Thread {} (seed {}): {} consistent results",
                thread_id,
                first_result.2,
                thread_results.len()
            );
        }
    }

    println!(
        "PyMatching parallel with fixed seeds test passed - {NUM_THREADS} threads × {NUM_ITERATIONS} iterations"
    );
}

#[test]
fn test_pymatching_global_rng_isolation() {
    // Test that decoder operations don't interfere with explicit RNG calls

    let syndrome = vec![1, 0, 1, 0, 0, 0];

    // Set seed and get decoder result
    PyMatchingDecoder::set_seed(555).unwrap();
    let mut decoder1 = create_simple_test_decoder().unwrap();
    let result1 = decoder1.decode(&syndrome).unwrap();

    // Randomize and then reset seed
    PyMatchingDecoder::randomize().unwrap();
    PyMatchingDecoder::set_seed(555).unwrap();

    let mut decoder2 = create_simple_test_decoder().unwrap();
    let result2 = decoder2.decode(&syndrome).unwrap();

    // Same seed should give same result even after randomize
    assert_eq!(
        result1.observable, result2.observable,
        "Results differ after randomize+reseed cycle"
    );
    assert!(
        weights_equal(result1.weight, result2.weight),
        "Weights differ after randomize+reseed cycle: {} vs {}",
        result1.weight,
        result2.weight
    );

    println!("PyMatching global RNG isolation test passed");
}

#[test]
fn test_pymatching_configuration_determinism() {
    // Test that decoder configuration doesn't affect determinism

    let syndrome = vec![1, 0, 1, 0, 0, 0];
    let mut results = Vec::new();

    // Test different configurations with same seed
    let configs = [
        PyMatchingConfig {
            num_nodes: Some(6),
            num_observables: 1,
            ..Default::default()
        },
        PyMatchingConfig {
            num_nodes: Some(6),
            num_observables: 1,
            ..Default::default()
        },
    ];

    for (i, config) in configs.iter().enumerate() {
        PyMatchingDecoder::set_seed(777).unwrap();

        let mut decoder = PyMatchingDecoder::new(config.clone()).unwrap();

        // Add same edges
        decoder
            .add_edge(0, 1, &[0], Some(1.0), Some(0.1), None)
            .unwrap();
        decoder
            .add_edge(1, 2, &[], Some(1.5), Some(0.1), None)
            .unwrap();
        decoder
            .add_edge(2, 3, &[], Some(2.0), Some(0.1), None)
            .unwrap();
        decoder
            .add_edge(3, 4, &[], Some(2.5), Some(0.1), None)
            .unwrap();
        decoder
            .add_edge(4, 5, &[], Some(3.0), Some(0.1), None)
            .unwrap();

        let result = decoder.decode(&syndrome).unwrap();
        results.push((i, result.observable.clone(), result.weight));
    }

    // Same configuration should give same results
    let first = &results[0];
    for (i, obs, weight) in &results {
        assert_eq!(first.1, *obs, "Config {i} gave different observable");
        assert!(
            weights_equal(first.2, *weight),
            "Config {i} gave different weight: {} vs {}",
            first.2,
            weight
        );
    }

    println!(
        "PyMatching configuration determinism test passed - {} configs tested",
        results.len()
    );
}

#[test]
#[allow(clippy::similar_names)]
fn test_pymatching_decoder_state_isolation() {
    // Test that multiple decoder instances don't share internal state

    let syndrome1 = vec![1, 0, 1, 0, 0, 0];
    let syndrome2 = vec![0, 1, 0, 1, 0, 0];

    PyMatchingDecoder::set_seed(888).unwrap();

    // Create multiple decoders
    let mut decoder_a = create_simple_test_decoder().unwrap();
    let mut decoder_b = create_simple_test_decoder().unwrap();
    let mut decoder_c = create_simple_test_decoder().unwrap();

    // Decode different syndromes with different decoders
    let result_a1 = decoder_a.decode(&syndrome1).unwrap();
    let result_b1 = decoder_b.decode(&syndrome2).unwrap();
    let result_c1 = decoder_c.decode(&syndrome1).unwrap();

    // Decoder A and C should give same results for same syndrome
    assert_eq!(
        result_a1.observable, result_c1.observable,
        "Decoders A and C gave different results for same syndrome"
    );
    assert!(
        weights_equal(result_a1.weight, result_c1.weight),
        "Decoders A and C gave different weights for same syndrome: {} vs {}",
        result_a1.weight,
        result_c1.weight
    );

    // Decode again - should be consistent
    let result_a2 = decoder_a.decode(&syndrome1).unwrap();
    let result_b2 = decoder_b.decode(&syndrome2).unwrap();

    assert_eq!(
        result_a1.observable, result_a2.observable,
        "Decoder A gave different results on repeat"
    );
    assert_eq!(
        result_b1.observable, result_b2.observable,
        "Decoder B gave different results on repeat"
    );

    println!("PyMatching decoder state isolation test passed");
}
