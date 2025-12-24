//! Comprehensive determinism tests for Fusion Blossom decoder
//!
//! These tests ensure that Fusion Blossom provides:
//! 1. Deterministic results across multiple runs
//! 2. Thread safety in parallel execution
//! 3. Independence between decoder instances
//! 4. Consistent behavior under various execution patterns

use ndarray::arr1;
use pecos_fusion_blossom::{FusionBlossomConfig, FusionBlossomDecoder, SolverType};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::Duration;

/// Create a simple test decoder for determinism testing
fn create_simple_test_decoder() -> Result<FusionBlossomDecoder, Box<dyn std::error::Error>> {
    let config = FusionBlossomConfig {
        num_nodes: Some(4),
        num_observables: 1,
        solver_type: SolverType::Serial,
        max_tree_size: None,
    };

    let mut decoder = FusionBlossomDecoder::new(config)?;

    // Add edges for a simple graph
    decoder.add_edge(0, 1, &[0], Some(1.0))?;
    decoder.add_edge(1, 2, &[], Some(1.5))?;
    decoder.add_edge(2, 3, &[], Some(2.0))?;
    decoder.add_edge(0, 3, &[], Some(3.0))?; // Alternative path

    Ok(decoder)
}

/// Create a larger test decoder for stress testing
fn create_large_test_decoder() -> Result<FusionBlossomDecoder, Box<dyn std::error::Error>> {
    let config = FusionBlossomConfig {
        num_nodes: Some(16), // 4x4 grid
        num_observables: 1,
        solver_type: SolverType::Serial,
        max_tree_size: None,
    };

    let mut decoder = FusionBlossomDecoder::new(config)?;

    // Add horizontal edges
    for i in 0..4 {
        for j in 0..3 {
            let node1 = i * 4 + j;
            let node2 = i * 4 + j + 1;
            decoder.add_edge(node1, node2, &[], Some(1.0))?;
        }
    }

    // Add vertical edges
    for i in 0..3 {
        for j in 0..4 {
            let node1 = i * 4 + j;
            let node2 = (i + 1) * 4 + j;
            decoder.add_edge(node1, node2, &[], Some(1.0))?;
        }
    }

    Ok(decoder)
}

#[test]
fn test_fusion_blossom_sequential_determinism() {
    // Test that Fusion Blossom gives identical results across multiple runs

    let mut results = Vec::new();

    for run in 0..10 {
        let mut decoder = create_simple_test_decoder().unwrap();
        let syndrome = arr1(&[1, 0, 1, 0]); // Defects at nodes 0 and 2
        let result = decoder.decode(&syndrome.view()).unwrap();

        results.push((result.observable.clone(), result.weight));

        if run < 2 {
            println!(
                "FusionBlossom run {}: observable={:?}, weight={}",
                run, result.observable, result.weight
            );
        }
    }

    // All results should be identical (FusionBlossom is deterministic)
    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "FusionBlossom run {i} gave different observable"
        );
        assert!(
            (first.1 - result.1).abs() < 1e-10,
            "FusionBlossom run {i} gave different weight: expected {}, got {}",
            first.1,
            result.1
        );
    }

    println!(
        "FusionBlossom sequential determinism test passed - {} consistent runs",
        results.len()
    );
}

#[test]
fn test_fusion_blossom_parallel_independence() {
    // Test that multiple FusionBlossom instances can run in parallel
    // without interfering with each other

    const NUM_THREADS: usize = 10;
    const NUM_ITERATIONS: usize = 8;

    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    for thread_id in 0..NUM_THREADS {
        let results_clone = Arc::clone(&results);

        let handle = thread::spawn(move || {
            for iteration in 0..NUM_ITERATIONS {
                let mut decoder = create_simple_test_decoder().unwrap();
                let syndrome = arr1(&[1, 0, 1, 0]);
                let result = decoder.decode(&syndrome.view()).unwrap();

                results_clone.lock().unwrap().push((
                    thread_id,
                    iteration,
                    result.observable.clone(),
                    result.weight,
                ));

                // Small delay to increase chance of race conditions
                thread::sleep(Duration::from_micros(50));
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_results = results.lock().unwrap();

    // Check that all results are identical (FusionBlossom is deterministic)
    if !final_results.is_empty() {
        let first_result = &final_results[0];
        for (tid, iter, obs, weight) in final_results.iter() {
            assert_eq!(
                first_result.2, *obs,
                "Thread {tid} iteration {iter} gave different observable"
            );
            assert!(
                (first_result.3 - *weight).abs() < 1e-10,
                "Thread {tid} iteration {iter} gave different weight: expected {}, got {}",
                first_result.3,
                *weight
            );
        }

        println!(
            "FusionBlossom parallel test passed - {} threads × {} iterations = {} consistent results",
            NUM_THREADS,
            NUM_ITERATIONS,
            final_results.len()
        );
    }
}

#[test]
fn test_fusion_blossom_instance_independence() {
    // Test that different FusionBlossom instances don't interfere with each other

    let syndrome1 = arr1(&[1, 0, 1, 0]);
    let syndrome2 = arr1(&[0, 1, 0, 1]);

    let mut results = Vec::new();

    for i in 0..5 {
        // Create multiple decoders for same problem
        let mut decoder_a = create_simple_test_decoder().unwrap();
        let mut decoder_b = create_simple_test_decoder().unwrap();

        // Decode same syndrome with both
        let result_a = decoder_a.decode(&syndrome1.view()).unwrap();
        let result_b = decoder_b.decode(&syndrome1.view()).unwrap();

        // Should get identical results
        assert_eq!(
            result_a.observable, result_b.observable,
            "Instance {i} decoders gave different observables for same syndrome"
        );
        assert!(
            (result_a.weight - result_b.weight).abs() < 1e-10,
            "Instance {i} decoders gave different weights for same syndrome: expected {}, got {}",
            result_a.weight,
            result_b.weight
        );

        // Try different syndrome with one decoder
        decoder_a.clear(); // Clear state before decoding different syndrome
        let _result_a2 = decoder_a.decode(&syndrome2.view()).unwrap();

        // Original result should be consistent if we decode again
        decoder_b.clear(); // Clear state before second decode
        let result_b2 = decoder_b.decode(&syndrome1.view()).unwrap();
        assert_eq!(
            result_b.observable, result_b2.observable,
            "Decoder B gave different result on second decode"
        );

        results.push((result_a.observable.clone(), result_a.weight));
    }

    // All iterations should be consistent
    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(first.0, result.0, "Iteration {i} gave different observable");
        assert!(
            (first.1 - result.1).abs() < 1e-10,
            "Iteration {i} gave different weight: expected {}, got {}",
            first.1,
            result.1
        );
    }

    println!(
        "FusionBlossom instance independence test passed - {} iterations",
        results.len()
    );
}

#[test]
fn test_fusion_blossom_configuration_determinism() {
    // Test that identical configurations give identical results

    let syndrome = arr1(&[1, 0, 1, 0]);
    let mut results = Vec::new();

    for _i in 0..5 {
        let config = FusionBlossomConfig {
            num_nodes: Some(4),
            num_observables: 1,
            solver_type: SolverType::Serial,
            max_tree_size: None,
        };

        let mut decoder = FusionBlossomDecoder::new(config).unwrap();

        // Add identical edge structure
        decoder.add_edge(0, 1, &[0], Some(1.0)).unwrap();
        decoder.add_edge(1, 2, &[], Some(1.5)).unwrap();
        decoder.add_edge(2, 3, &[], Some(2.0)).unwrap();
        decoder.add_edge(0, 3, &[], Some(3.0)).unwrap();

        let result = decoder.decode(&syndrome.view()).unwrap();
        results.push((result.observable.clone(), result.weight));
    }

    // All should give identical results
    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(first.0, result.0, "Config {i} gave different observable");
        assert!(
            (first.1 - result.1).abs() < 1e-10,
            "Config {i} gave different weight: expected {}, got {}",
            first.1,
            result.1
        );
    }

    println!(
        "FusionBlossom configuration determinism test passed - {} configs",
        results.len()
    );
}

#[test]
fn test_fusion_blossom_large_graph_determinism() {
    // Test determinism on larger graphs

    let syndrome = arr1(&[1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]);
    let mut results = Vec::new();

    for run in 0..5 {
        let mut decoder = create_large_test_decoder().unwrap();
        let result = decoder.decode(&syndrome.view()).unwrap();

        results.push((result.observable.clone(), result.weight));

        if run == 0 {
            println!(
                "Large graph result: observable={:?}, weight={}",
                result.observable, result.weight
            );
        }
    }

    // All results should be identical
    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "Large graph run {i} gave different observable"
        );
        assert!(
            (first.1 - result.1).abs() < 1e-10,
            "Large graph run {i} gave different weight: expected {}, got {}",
            first.1,
            result.1
        );
    }

    println!(
        "FusionBlossom large graph determinism test passed - {} runs",
        results.len()
    );
}

#[test]
fn test_fusion_blossom_concurrent_different_problems() {
    // Test that solving different problems concurrently doesn't interfere

    let problems = [
        arr1(&[1, 0, 1, 0]),
        arr1(&[0, 1, 0, 1]),
        arr1(&[1, 1, 0, 0]),
        arr1(&[0, 0, 1, 1]),
    ];

    let results = Arc::new(Mutex::new(Vec::new()));
    let mut handles = vec![];

    for (problem_id, syndrome) in problems.iter().enumerate() {
        let results_clone = Arc::clone(&results);
        let syndrome_clone = syndrome.clone();

        let handle = thread::spawn(move || {
            for iteration in 0..3 {
                let mut decoder = create_simple_test_decoder().unwrap();
                let result = decoder.decode(&syndrome_clone.view()).unwrap();

                results_clone.lock().unwrap().push((
                    problem_id,
                    iteration,
                    result.observable.clone(),
                    result.weight,
                ));
            }
        });

        handles.push(handle);
    }

    for handle in handles {
        handle.join().unwrap();
    }

    let final_results = results.lock().unwrap();

    // Check that each problem type got consistent results across iterations
    for problem_id in 0..problems.len() {
        let problem_results: Vec<_> = final_results
            .iter()
            .filter(|(pid, _, _, _)| *pid == problem_id)
            .collect();

        if !problem_results.is_empty() {
            let first_result = &problem_results[0];
            for (pid, iter, obs, weight) in &problem_results {
                assert_eq!(
                    first_result.2, *obs,
                    "Problem {pid} iteration {iter} gave different observable"
                );
                assert!(
                    (first_result.3 - *weight).abs() < 1e-10,
                    "Problem {pid} iteration {iter} gave different weight: expected {}, got {}",
                    first_result.3,
                    *weight
                );
            }

            println!(
                "Problem {}: {} consistent results",
                problem_id,
                problem_results.len()
            );
        }
    }

    println!("FusionBlossom concurrent different problems test passed");
}

#[test]
fn test_fusion_blossom_repeated_decode_same_instance() {
    // Test that repeatedly decoding with same instance gives consistent results

    let syndrome1 = arr1(&[1, 0, 1, 0]);
    let syndrome2 = arr1(&[0, 1, 0, 1]);

    let mut decoder = create_simple_test_decoder().unwrap();

    // Decode syndrome1 multiple times
    let mut results1 = Vec::new();
    for _i in 0..5 {
        decoder.clear(); // Clear state before each decode
        let result = decoder.decode(&syndrome1.view()).unwrap();
        results1.push((result.observable.clone(), result.weight));
    }

    // Decode syndrome2 multiple times
    let mut results2 = Vec::new();
    for _i in 0..5 {
        decoder.clear(); // Clear state before each decode
        let result = decoder.decode(&syndrome2.view()).unwrap();
        results2.push((result.observable.clone(), result.weight));
    }

    // Decode syndrome1 again - should still be consistent
    let mut results1_again = Vec::new();
    for _i in 0..5 {
        decoder.clear(); // Clear state before each decode
        let result = decoder.decode(&syndrome1.view()).unwrap();
        results1_again.push((result.observable.clone(), result.weight));
    }

    // Check consistency within each syndrome
    let first1 = &results1[0];
    for (i, result) in results1.iter().enumerate() {
        assert_eq!(
            first1.0, result.0,
            "Syndrome1 decode {i} gave different observable"
        );
        assert!(
            (first1.1 - result.1).abs() < 1e-10,
            "Syndrome1 decode {i} gave different weight: expected {}, got {}",
            first1.1,
            result.1
        );
    }

    let first2 = &results2[0];
    for (i, result) in results2.iter().enumerate() {
        assert_eq!(
            first2.0, result.0,
            "Syndrome2 decode {i} gave different observable"
        );
        assert!(
            (first2.1 - result.1).abs() < 1e-10,
            "Syndrome2 decode {i} gave different weight: expected {}, got {}",
            first2.1,
            result.1
        );
    }

    // Check that syndrome1 results are consistent across sessions
    let first1_again = &results1_again[0];
    assert_eq!(
        first1.0, first1_again.0,
        "Syndrome1 results changed between sessions"
    );
    assert!(
        (first1.1 - first1_again.1).abs() < 1e-10,
        "Syndrome1 weights changed between sessions: expected {}, got {}",
        first1.1,
        first1_again.1
    );

    println!("FusionBlossom repeated decode test passed - same instance used for multiple decodes");
}

#[test]
#[allow(clippy::similar_names)] // result_a1/b1/c1/a2/b2 naming is clear: decoder + run number
fn test_fusion_blossom_decoder_state_isolation() {
    // Test that multiple decoders don't share internal state

    let syndrome1 = arr1(&[1, 0, 1, 0]);
    let syndrome2 = arr1(&[0, 1, 0, 1]);

    // Create multiple decoders
    let mut decoder_a = create_simple_test_decoder().unwrap();
    let mut decoder_b = create_simple_test_decoder().unwrap();
    let mut decoder_c = create_simple_test_decoder().unwrap();

    // Decode different syndromes with different decoders
    let result_a1 = decoder_a.decode(&syndrome1.view()).unwrap();
    let result_b1 = decoder_b.decode(&syndrome2.view()).unwrap();
    let result_c1 = decoder_c.decode(&syndrome1.view()).unwrap();

    // Decoder A and C should give same results for same syndrome
    assert_eq!(
        result_a1.observable, result_c1.observable,
        "Decoders A and C gave different results for same syndrome"
    );
    assert!(
        (result_a1.weight - result_c1.weight).abs() < 1e-10,
        "Decoders A and C gave different weights for same syndrome: expected {}, got {}",
        result_a1.weight,
        result_c1.weight
    );

    // Clear state before second decode
    decoder_a.clear();
    decoder_b.clear();

    // Decode again - should be consistent
    let result_a2 = decoder_a.decode(&syndrome1.view()).unwrap();
    let result_b2 = decoder_b.decode(&syndrome2.view()).unwrap();

    assert_eq!(
        result_a1.observable, result_a2.observable,
        "Decoder A gave different results on repeat"
    );
    assert_eq!(
        result_b1.observable, result_b2.observable,
        "Decoder B gave different results on repeat"
    );

    println!("FusionBlossom decoder state isolation test passed");
}

#[test]
fn test_fusion_blossom_empty_syndrome_determinism() {
    // Test determinism with empty syndrome (no defects)

    let syndrome = arr1(&[0, 0, 0, 0]);
    let mut results = Vec::new();

    for _run in 0..10 {
        let mut decoder = create_simple_test_decoder().unwrap();
        let result = decoder.decode(&syndrome.view()).unwrap();

        results.push((result.observable.clone(), result.weight));
    }

    // All results should be identical
    let first = &results[0];
    for (i, result) in results.iter().enumerate() {
        assert_eq!(
            first.0, result.0,
            "Empty syndrome run {i} gave different observable"
        );
        assert!(
            (first.1 - result.1).abs() < 1e-10,
            "Empty syndrome run {i} gave different weight: expected {}, got {}",
            first.1,
            result.1
        );
    }

    println!(
        "FusionBlossom empty syndrome determinism test passed - {} runs",
        results.len()
    );
}
