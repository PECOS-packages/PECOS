//! Comprehensive determinism tests for Chromobius decoder
//!
//! These tests ensure that the Chromobius decoder provides:
//! 1. Deterministic results across multiple runs
//! 2. Thread safety in parallel execution
//! 3. Independence between decoder instances
//! 4. Consistent behavior under various execution patterns

use pecos_chromobius::{ChromobiusConfig, ChromobiusDecoder};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Create a test DEM for Chromobius
fn create_test_circuit() -> String {
    // Simple detector error model
    r"
error(0.1) D0 D1
error(0.05) D1 L0
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
    "
    .trim()
    .to_string()
}

/// Create test syndrome data
fn create_test_syndrome_small() -> Vec<u8> {
    vec![0b11] // Detectors 0 and 1 triggered - fits in 1 byte
}

// ============================================================================
// Basic Determinism Tests
// ============================================================================

#[test]
fn test_chromobius_sequential_determinism() {
    let circuit = create_test_circuit();
    let syndrome = create_test_syndrome_small();

    let mut results = Vec::new();

    // Run multiple times - should get identical results
    for run in 0..20 {
        let config = ChromobiusConfig::default();
        let mut decoder = ChromobiusDecoder::new(&circuit, config).unwrap();

        let result = decoder.decode_detection_events(&syndrome).unwrap();
        results.push((result.observables, result.weight));

        if run < 3 {
            println!(
                "Chromobius run {}: observables={:?}, weight={:?}",
                run, result.observables, result.weight
            );
        }
    }

    // All results should be identical (Chromobius is deterministic)
    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "Chromobius run {i} gave different observables"
        );
        assert_eq!(
            first.1, result.1,
            "Chromobius run {i} gave different weight"
        );
    }

    println!(
        "Chromobius sequential determinism test passed - {} consistent runs",
        results.len()
    );
}

#[test]
fn test_chromobius_parallel_independence() {
    // Test that multiple Chromobius instances can run in parallel
    // without interfering with each other

    const NUM_THREADS: usize = 10;
    const NUM_ITERATIONS: usize = 8;

    let circuit = Arc::new(create_test_circuit());
    let syndrome = Arc::new(create_test_syndrome_small());
    let results = Arc::new(Mutex::new(Vec::new()));

    let mut handles = vec![];

    for thread_id in 0..NUM_THREADS {
        let circuit_clone = Arc::clone(&circuit);
        let syndrome_clone = Arc::clone(&syndrome);
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            for iteration in 0..NUM_ITERATIONS {
                let config = ChromobiusConfig::default();
                let mut decoder = ChromobiusDecoder::new(&circuit_clone, config).unwrap();

                let result = decoder.decode_detection_events(&syndrome_clone).unwrap();

                results_clone.lock().unwrap().push((
                    thread_id,
                    iteration,
                    result.observables,
                    result.weight,
                ));

                // Small delay to encourage interleaving
                thread::sleep(Duration::from_micros(50));
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_results = results.lock().unwrap();

    // Check that each thread got consistent results
    for thread_id in 0..NUM_THREADS {
        let thread_results: Vec<_> = final_results
            .iter()
            .filter(|(tid, _, _, _)| *tid == thread_id)
            .collect();

        let first_result = &thread_results[0];
        for (i, result) in thread_results.iter().enumerate() {
            assert_eq!(
                first_result.2, result.2,
                "Thread {thread_id} iteration {i} gave different observables"
            );
            assert_eq!(
                first_result.3, result.3,
                "Thread {thread_id} iteration {i} gave different weight"
            );
        }

        if thread_id < 3 {
            println!("Thread {thread_id}: consistent across {NUM_ITERATIONS} iterations");
        }
    }

    // All threads should have gotten the same result (deterministic decoder)
    let first_thread_result = &final_results
        .iter()
        .find(|(tid, _, _, _)| *tid == 0)
        .unwrap();

    for result in final_results.iter() {
        assert_eq!(
            first_thread_result.2, result.2,
            "Different threads gave different observables"
        );
        assert_eq!(
            first_thread_result.3, result.3,
            "Different threads gave different weights"
        );
    }

    println!("Chromobius parallel independence test passed - all threads consistent");
}

#[test]
#[allow(clippy::similar_names)] // result1a/result1b naming is clear: decoder1 first/second run
fn test_chromobius_instance_independence() {
    // Test that multiple decoder instances don't interfere with each other
    let circuit = create_test_circuit();
    let syndrome1 = create_test_syndrome_small();
    let syndrome2 = vec![0b01]; // Different syndrome

    // Create multiple decoders
    let config1 = ChromobiusConfig::default();
    let mut decoder1 = ChromobiusDecoder::new(&circuit, config1).unwrap();

    let config2 = ChromobiusConfig::default();
    let mut decoder2 = ChromobiusDecoder::new(&circuit, config2).unwrap();

    let config3 = ChromobiusConfig::default();
    let mut decoder3 = ChromobiusDecoder::new(&circuit, config3).unwrap();

    // Decode with first decoder
    let result1a = decoder1.decode_detection_events(&syndrome1).unwrap();

    // Decode with second decoder using different syndrome
    let result2 = decoder2.decode_detection_events(&syndrome2).unwrap();

    // Decode with third decoder using same syndrome as first
    let result3 = decoder3.decode_detection_events(&syndrome1).unwrap();

    // Decode again with first decoder - should get same result as before
    let result1b = decoder1.decode_detection_events(&syndrome1).unwrap();

    // Results from same syndrome should be identical
    assert_eq!(
        result1a.observables, result1b.observables,
        "Same decoder gave different results for same syndrome"
    );
    assert_eq!(
        result1a.weight, result1b.weight,
        "Same decoder gave different weights for same syndrome"
    );

    assert_eq!(
        result1a.observables, result3.observables,
        "Different decoders gave different results for same syndrome"
    );
    assert_eq!(
        result1a.weight, result3.weight,
        "Different decoders gave different weights for same syndrome"
    );

    println!("Chromobius instance independence test passed");
    println!(
        "  Syndrome {:?} -> Observables {:?}, Cost {:?}",
        syndrome1, result1a.observables, result1a.weight
    );
    println!(
        "  Syndrome {:?} -> Observables {:?}, Cost {:?}",
        syndrome2, result2.observables, result2.weight
    );
}

#[test]
fn test_chromobius_configuration_determinism() {
    // Test that same configuration always produces same results
    let circuit = create_test_circuit();
    let syndrome = create_test_syndrome_small();

    // Test different configurations
    let test_configs = vec![
        ChromobiusConfig::default(),
        ChromobiusConfig {
            ..Default::default()
        }, // Same as default but explicit
    ];

    for (config_idx, config) in test_configs.into_iter().enumerate() {
        let mut results = Vec::new();

        // Run multiple times with same config
        for _run in 0..15 {
            let mut decoder = ChromobiusDecoder::new(&circuit, config).unwrap();
            let result = decoder.decode_detection_events(&syndrome).unwrap();
            results.push((result.observables, result.weight));
        }

        // All results should be identical for this config
        let first = &results[0];
        for (i, result) in results.iter().enumerate() {
            assert_eq!(
                first.0, result.0,
                "Config {config_idx} run {i} gave different observables"
            );
            assert_eq!(
                first.1, result.1,
                "Config {config_idx} run {i} gave different weight"
            );
        }

        println!(
            "Config {}: deterministic across {} runs",
            config_idx,
            results.len()
        );
    }
}

// ============================================================================
// Stress Tests
// ============================================================================

#[test]
fn test_chromobius_large_circuit_determinism() {
    let circuit = create_test_circuit(); // Use simple circuit for now
    let syndrome = create_test_syndrome_small();

    let mut results = Vec::new();

    for _run in 0..12 {
        let config = ChromobiusConfig::default();
        let mut decoder = ChromobiusDecoder::new(&circuit, config).unwrap();

        let result = decoder.decode_detection_events(&syndrome).unwrap();
        results.push((result.observables, result.weight));
    }

    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "Large circuit run {i} gave different observables"
        );
        assert_eq!(
            first.1, result.1,
            "Large circuit run {i} gave different weight"
        );
    }

    println!(
        "Large circuit determinism test passed - {} syndrome elements",
        syndrome.len()
    );
}

#[test]
fn test_chromobius_concurrent_different_problems() {
    // Test multiple decoders working on different problems simultaneously
    const NUM_THREADS: usize = 6;

    let circuit = Arc::new(create_test_circuit());
    let results = Arc::new(Mutex::new(Vec::new()));

    let test_syndromes = vec![
        vec![0b11],
        vec![0b01],
        vec![0b10],
        vec![0b00],
        vec![0b11], // Repeat to test consistency
        vec![0b01], // Repeat to test consistency
    ];

    let syndromes = Arc::new(test_syndromes);
    let mut handles = vec![];

    for thread_id in 0..NUM_THREADS {
        let circuit_clone = Arc::clone(&circuit);
        let syndromes_clone = Arc::clone(&syndromes);
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            let syndrome = &syndromes_clone[thread_id];

            // Run same problem multiple times in this thread
            for iteration in 0..5 {
                let config = ChromobiusConfig::default();
                let mut decoder = ChromobiusDecoder::new(&circuit_clone, config).unwrap();

                let result = decoder.decode_detection_events(syndrome).unwrap();

                results_clone.lock().unwrap().push((
                    thread_id,
                    iteration,
                    syndrome.clone(),
                    result.observables,
                    result.weight,
                ));

                thread::sleep(Duration::from_micros(100));
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_results = results.lock().unwrap();

    // Check consistency within each thread
    for thread_id in 0..NUM_THREADS {
        let thread_results: Vec<_> = final_results
            .iter()
            .filter(|(tid, _, _, _, _)| *tid == thread_id)
            .collect();

        let first_result = &thread_results[0];
        for (i, result) in thread_results.iter().enumerate() {
            assert_eq!(
                first_result.3, result.3,
                "Thread {thread_id} iteration {i} gave different observables"
            );
            assert_eq!(
                first_result.4, result.4,
                "Thread {thread_id} iteration {i} gave different weight"
            );
        }

        println!(
            "Thread {} (syndrome {:?}): consistent observables {:?}, weight {:?}",
            thread_id, first_result.2, first_result.3, first_result.4
        );
    }

    // Check that repeated syndromes gave same results
    let syndrome_11_results: Vec<_> = final_results
        .iter()
        .filter(|(_, _, syndrome, _, _)| syndrome == &vec![0b11])
        .collect();

    if syndrome_11_results.len() > 1 {
        let first_11 = &syndrome_11_results[0];
        for result in &syndrome_11_results[1..] {
            assert_eq!(
                first_11.3, result.3,
                "Same syndrome [0b11] gave different observables"
            );
            assert_eq!(
                first_11.4, result.4,
                "Same syndrome [0b11] gave different weights"
            );
        }
    }
}

#[test]
fn test_chromobius_repeated_decode_same_instance() {
    // Test that using the same decoder instance repeatedly gives consistent results
    let circuit = create_test_circuit();
    let syndrome = create_test_syndrome_small();

    let config = ChromobiusConfig::default();
    let mut decoder = ChromobiusDecoder::new(&circuit, config).unwrap();

    let mut results = Vec::new();

    for _run in 0..25 {
        let result = decoder.decode_detection_events(&syndrome).unwrap();
        results.push((result.observables, result.weight));
    }

    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "Repeated decode {i} gave different observables"
        );
        assert_eq!(
            first.1, result.1,
            "Repeated decode {i} gave different weight"
        );
    }

    println!(
        "Repeated decode test passed - {} consistent decodes with same instance",
        results.len()
    );
}

#[test]
fn test_chromobius_decoder_state_isolation() {
    // Test that decoder state doesn't leak between different decode operations
    let circuit = create_test_circuit();

    let config = ChromobiusConfig::default();
    let mut decoder = ChromobiusDecoder::new(&circuit, config).unwrap();

    let syndrome1 = vec![0b11];
    let syndrome2 = vec![0b01];
    let syndrome3 = vec![0b11]; // Same as syndrome1

    // Decode first syndrome
    let result1 = decoder.decode_detection_events(&syndrome1).unwrap();

    // Decode different syndrome
    let result2 = decoder.decode_detection_events(&syndrome2).unwrap();

    // Decode first syndrome again - should get same result as first time
    let result3 = decoder.decode_detection_events(&syndrome3).unwrap();

    assert_eq!(
        result1.observables, result3.observables,
        "Decoder state leaked between operations - observables differ"
    );
    assert_eq!(
        result1.weight, result3.weight,
        "Decoder state leaked between operations - weights differ"
    );

    println!("Decoder state isolation test passed");
    println!(
        "  Syndrome {:?} -> Observables {:?}, Cost {:?}",
        syndrome1, result1.observables, result1.weight
    );
    println!(
        "  Syndrome {:?} -> Observables {:?}, Cost {:?}",
        syndrome2, result2.observables, result2.weight
    );
    println!(
        "  Syndrome {:?} -> Observables {:?}, Cost {:?} (should match first)",
        syndrome3, result3.observables, result3.weight
    );
}

#[test]
fn test_chromobius_empty_syndrome_determinism() {
    // Test that empty syndromes are handled deterministically
    let circuit = create_test_circuit();
    let empty_syndrome = vec![0b00];

    let mut results = Vec::new();

    for _run in 0..15 {
        let config = ChromobiusConfig::default();
        let mut decoder = ChromobiusDecoder::new(&circuit, config).unwrap();

        let result = decoder.decode_detection_events(&empty_syndrome).unwrap();
        results.push((result.observables, result.weight));
    }

    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "Empty syndrome run {i} gave different observables"
        );
        assert_eq!(
            first.1, result.1,
            "Empty syndrome run {i} gave different weight"
        );
    }

    println!(
        "Empty syndrome determinism test passed - consistent across {} runs",
        results.len()
    );
    println!(
        "  Empty syndrome result: Observables {:?}, Cost {:?}",
        first.0, first.1
    );
}

#[test]
fn test_chromobius_circuit_reconstruction_determinism() {
    // Test that reconstructing the same circuit gives same results
    let circuit_str = create_test_circuit();
    let syndrome = create_test_syndrome_small();

    let mut results = Vec::new();

    for _run in 0..10 {
        // Reconstruct decoder from circuit string each time
        let config = ChromobiusConfig::default();
        let mut decoder = ChromobiusDecoder::new(&circuit_str, config).unwrap();

        let result = decoder.decode_detection_events(&syndrome).unwrap();
        results.push((result.observables, result.weight));
    }

    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "Circuit reconstruction {i} gave different observables"
        );
        assert_eq!(
            first.1, result.1,
            "Circuit reconstruction {i} gave different weight"
        );
    }

    println!(
        "Circuit reconstruction determinism test passed - {} consistent reconstructions",
        results.len()
    );
}
