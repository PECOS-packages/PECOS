//! Comprehensive determinism tests for Tesseract decoder
//!
//! These tests ensure that the Tesseract decoder provides:
//! 1. Deterministic results across multiple runs
//! 2. Thread safety in parallel execution
//! 3. Independence between decoder instances
//! 4. Consistent behavior under various execution patterns

use ndarray::arr1;
use pecos_decoder_core::Decoder;
use pecos_tesseract::{TesseractConfig, TesseractDecoder};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Create a test syndrome for a small graph
fn create_test_syndrome_small() -> ndarray::Array1<u8> {
    arr1(&[1, 0, 1, 0]) // Simple test pattern matching 4 detectors
}

/// Create a larger test syndrome
fn create_test_syndrome_large() -> ndarray::Array1<u8> {
    arr1(&[1, 0, 1, 0]) // Use same valid pattern as small test - DEM only has 4 detectors
}

/// Create a test DEM string for Tesseract
fn create_test_dem() -> String {
    // Simple repetition code DEM
    r"
error(0.1) D0 D1
error(0.05) D1 D2
error(0.02) D2 D3 L0
    "
    .to_string()
}

// ============================================================================
// Basic Determinism Tests
// ============================================================================

#[test]
fn test_tesseract_sequential_determinism() {
    let dem = create_test_dem();
    let syndrome = create_test_syndrome_small();

    let mut results = Vec::new();

    // Run multiple times - should get identical results
    for run in 0..20 {
        let config = TesseractConfig::default();
        let mut decoder = TesseractDecoder::new(&dem, config).unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();

        results.push((result.predicted_errors.clone(), result.cost));

        if run < 3 {
            println!(
                "Tesseract run {}: predicted_errors={:?}, cost={}",
                run, result.predicted_errors, result.cost
            );
        }
    }

    // All results should be identical (Tesseract is deterministic)
    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "Tesseract run {i} gave different predicted_errors"
        );
        assert!(
            (first.1 - result.1).abs() < 1e-10,
            "Tesseract run {i} gave different cost: expected {}, got {}",
            first.1,
            result.1
        );
    }

    println!(
        "Tesseract sequential determinism test passed - {} consistent runs",
        results.len()
    );
}

#[test]
fn test_tesseract_parallel_independence() {
    // Test that multiple Tesseract instances can run in parallel
    // without interfering with each other

    const NUM_THREADS: usize = 10;
    const NUM_ITERATIONS: usize = 8;

    let dem = Arc::new(create_test_dem());
    let syndrome = Arc::new(create_test_syndrome_small());
    let results = Arc::new(Mutex::new(Vec::new()));

    let mut handles = vec![];

    for thread_id in 0..NUM_THREADS {
        let dem_clone = Arc::clone(&dem);
        let syndrome_clone = Arc::clone(&syndrome);
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            for iteration in 0..NUM_ITERATIONS {
                let config = TesseractConfig::default();
                let mut decoder = TesseractDecoder::new(&dem_clone, config).unwrap();

                let result = decoder.decode(&syndrome_clone.view()).unwrap();

                results_clone.lock().unwrap().push((
                    thread_id,
                    iteration,
                    result.predicted_errors.clone(),
                    result.cost,
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
                "Thread {thread_id} iteration {i} gave different predicted_errors"
            );
            assert!(
                (first_result.3 - result.3).abs() < 1e-10,
                "Thread {thread_id} iteration {i} gave different cost: expected {}, got {}",
                first_result.3,
                result.3
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
            "Different threads gave different predicted_errors"
        );
        assert!(
            (first_thread_result.3 - result.3).abs() < 1e-10,
            "Different threads gave different costs: expected {}, got {}",
            first_thread_result.3,
            result.3
        );
    }

    println!("Tesseract parallel independence test passed - all threads consistent");
}

#[test]
fn test_tesseract_instance_independence() {
    // Test that multiple decoder instances don't interfere with each other
    let dem = create_test_dem();
    let syndrome1 = create_test_syndrome_small();
    let syndrome2 = arr1(&[0, 1, 0, 1]); // Different syndrome

    // Create multiple decoders
    let config1 = TesseractConfig::default();
    let mut decoder1 = TesseractDecoder::new(&dem, config1).unwrap();

    let config2 = TesseractConfig::default();
    let mut decoder2 = TesseractDecoder::new(&dem, config2).unwrap();

    let config3 = TesseractConfig::default();
    let mut decoder3 = TesseractDecoder::new(&dem, config3).unwrap();

    // Decode with first decoder
    let result1a = decoder1.decode(&syndrome1.view()).unwrap();

    // Decode with second decoder using different syndrome
    let result2 = decoder2.decode(&syndrome2.view()).unwrap();

    // Decode with third decoder using same syndrome as first
    let result3 = decoder3.decode(&syndrome1.view()).unwrap();

    // Decode again with first decoder - should get same result as before
    let result1_repeat = decoder1.decode(&syndrome1.view()).unwrap();

    // Results from same syndrome should be identical
    assert_eq!(
        result1a.predicted_errors, result1_repeat.predicted_errors,
        "Same decoder gave different results for same syndrome"
    );
    assert!(
        (result1a.cost - result1_repeat.cost).abs() < 1e-10,
        "Same decoder gave different costs for same syndrome: expected {}, got {}",
        result1a.cost,
        result1_repeat.cost
    );

    assert_eq!(
        result1a.predicted_errors, result3.predicted_errors,
        "Different decoders gave different results for same syndrome"
    );
    assert!(
        (result1a.cost - result3.cost).abs() < 1e-10,
        "Different decoders gave different costs for same syndrome: expected {}, got {}",
        result1a.cost,
        result3.cost
    );

    println!("Tesseract instance independence test passed");
    println!(
        "  Syndrome {:?} -> Predicted_errors {:?}, Cost {}",
        syndrome1, result1a.predicted_errors, result1a.cost
    );
    println!(
        "  Syndrome {:?} -> Predicted_errors {:?}, Cost {}",
        syndrome2, result2.predicted_errors, result2.cost
    );
}

#[test]
fn test_tesseract_configuration_determinism() {
    // Test that same configuration always produces same results
    let dem = create_test_dem();
    let syndrome = create_test_syndrome_small();

    let test_configs = vec![
        TesseractConfig::default(),
        TesseractConfig::fast(),
        TesseractConfig::accurate(),
    ];

    for (config_idx, config) in test_configs.into_iter().enumerate() {
        let mut results = Vec::new();

        // Run multiple times with same config
        for _run in 0..15 {
            let mut decoder = TesseractDecoder::new(&dem, config.clone()).unwrap();
            let result = decoder.decode(&syndrome.view()).unwrap();
            results.push((result.predicted_errors.clone(), result.cost));
        }

        // All results should be identical for this config
        let first = &results[0];
        for (i, result) in results.iter().enumerate() {
            assert_eq!(
                first.0, result.0,
                "Config {config_idx} run {i} gave different predicted_errors"
            );
            assert!(
                (first.1 - result.1).abs() < 1e-10,
                "Config {config_idx} run {i} gave different cost: expected {}, got {}",
                first.1,
                result.1
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
fn test_tesseract_large_syndrome_determinism() {
    let dem = create_test_dem();
    let syndrome = create_test_syndrome_large();

    let mut results = Vec::new();

    for _run in 0..12 {
        let config = TesseractConfig::default();
        let mut decoder = TesseractDecoder::new(&dem, config).unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();

        results.push((result.predicted_errors.clone(), result.cost));
    }

    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "Large syndrome run {i} gave different predicted_errors"
        );
        assert!(
            (first.1 - result.1).abs() < 1e-10,
            "Large syndrome run {i} gave different cost: expected {}, got {}",
            first.1,
            result.1
        );
    }

    println!(
        "Large syndrome determinism test passed - {} syndrome elements",
        syndrome.len()
    );
}

#[test]
fn test_tesseract_concurrent_different_problems() {
    // Test multiple decoders working on different problems simultaneously
    const NUM_THREADS: usize = 6;

    let dem = Arc::new(create_test_dem());
    let results = Arc::new(Mutex::new(Vec::new()));

    let test_syndromes = vec![
        arr1(&[1, 0, 0, 0]),
        arr1(&[0, 1, 0, 0]),
        arr1(&[0, 0, 1, 0]),
        arr1(&[0, 0, 0, 1]),
        arr1(&[1, 1, 0, 0]),
        arr1(&[1, 0, 1, 1]),
    ];

    let syndromes = Arc::new(test_syndromes);
    let mut handles = vec![];

    for thread_id in 0..NUM_THREADS {
        let dem_clone = Arc::clone(&dem);
        let syndromes_clone = Arc::clone(&syndromes);
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            let syndrome = &syndromes_clone[thread_id];

            // Run same problem multiple times in this thread
            for iteration in 0..5 {
                let config = TesseractConfig::default();
                let mut decoder = TesseractDecoder::new(&dem_clone, config).unwrap();

                let result = decoder.decode(&syndrome.view()).unwrap();

                results_clone.lock().unwrap().push((
                    thread_id,
                    iteration,
                    syndrome.clone(),
                    result.predicted_errors.clone(),
                    result.cost,
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
                "Thread {thread_id} iteration {i} gave different predicted_errors"
            );
            assert!(
                (first_result.4 - result.4).abs() < 1e-10,
                "Thread {thread_id} iteration {i} gave different cost: expected {}, got {}",
                first_result.4,
                result.4
            );
        }

        println!(
            "Thread {} (syndrome {:?}): consistent predicted_errors {:?}, cost {}",
            thread_id, first_result.2, first_result.3, first_result.4
        );
    }
}

#[test]
fn test_tesseract_repeated_decode_same_instance() {
    // Test that using the same decoder instance repeatedly gives consistent results
    let dem = create_test_dem();
    let syndrome = create_test_syndrome_small();

    let config = TesseractConfig::default();
    let mut decoder = TesseractDecoder::new(&dem, config).unwrap();

    let mut results = Vec::new();

    for _run in 0..25 {
        let result = decoder.decode(&syndrome.view()).unwrap();
        results.push((result.predicted_errors.clone(), result.cost));
    }

    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "Repeated decode {i} gave different predicted_errors"
        );
        assert!(
            (first.1 - result.1).abs() < 1e-10,
            "Repeated decode {i} gave different cost: expected {}, got {}",
            first.1,
            result.1
        );
    }

    println!(
        "Repeated decode test passed - {} consistent decodes with same instance",
        results.len()
    );
}

#[test]
fn test_tesseract_decoder_state_isolation() {
    // Test that decoder state doesn't leak between different decode operations
    let dem = create_test_dem();

    let config = TesseractConfig::default();
    let mut decoder = TesseractDecoder::new(&dem, config).unwrap();

    let syndrome1 = arr1(&[1, 0, 0, 0]);
    let syndrome2 = arr1(&[0, 1, 1, 0]);
    let syndrome3 = arr1(&[1, 0, 0, 0]); // Same as syndrome1

    // Decode first syndrome
    let result1 = decoder.decode(&syndrome1.view()).unwrap();

    // Decode different syndrome
    let result2 = decoder.decode(&syndrome2.view()).unwrap();

    // Decode first syndrome again - should get same result as first time
    let result3 = decoder.decode(&syndrome3.view()).unwrap();

    assert_eq!(
        result1.predicted_errors, result3.predicted_errors,
        "Decoder state leaked between operations - predicted_errors differ"
    );
    assert!(
        (result1.cost - result3.cost).abs() < 1e-10,
        "Decoder state leaked between operations - costs differ: expected {}, got {}",
        result1.cost,
        result3.cost
    );

    // Result 2 should be different (different syndrome)
    // (We don't assert this as it depends on the specific DEM and syndromes)

    println!("Decoder state isolation test passed");
    println!(
        "  Syndrome {:?} -> Predicted_errors {:?}, Cost {}",
        syndrome1, result1.predicted_errors, result1.cost
    );
    println!(
        "  Syndrome {:?} -> Predicted_errors {:?}, Cost {}",
        syndrome2, result2.predicted_errors, result2.cost
    );
    println!(
        "  Syndrome {:?} -> Predicted_errors {:?}, Cost {} (should match first)",
        syndrome3, result3.predicted_errors, result3.cost
    );
}
