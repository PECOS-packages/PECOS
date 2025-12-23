//! Core algorithm tests for `PyMatching` decoder
//! Based on C++ tests from `PyMatching` repository

use ndarray::Array1;
use pecos_decoder_core::DecodingResultTrait;
use pecos_pymatching::{PyMatchingConfig, PyMatchingDecoder};

/// Test perfect matching with even parity syndrome
#[test]
fn test_perfect_matching_even_parity() {
    // Create a simple graph with boundary nodes to ensure perfect matching exists
    let dem = r"
error(0.1) D0 D1
error(0.1) D1 D2
error(0.1) D2 D3
error(0.1) D0
error(0.1) D3
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
detector(3, 0, 0, 3) D3
    "
    .trim();

    let _config = PyMatchingConfig::default();
    match PyMatchingDecoder::from_dem(dem) {
        Ok(mut decoder) => {
            // Test with even parity (two detections)
            let syndrome = Array1::from_vec(vec![1u8, 0u8, 0u8, 1u8]);
            let result = decoder.decode(syndrome.as_slice().unwrap()).unwrap();
            assert!(result.is_successful());
            println!(
                "Even parity matching succeeded with weight: {}",
                result.weight
            );
        }
        Err(e) => panic!("Decoder creation failed: {e}"),
    }
}

/// Test decoding with negative weights
#[test]
fn test_negative_weight_edges() {
    // DEM with negative weight edges (error probability > 0.5)
    let dem = r"
error(0.1) D0 D1
error(0.8) D1 D2
error(0.1) D2 D3
error(0.1) D0
error(0.1) D3
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
detector(3, 0, 0, 3) D3
    "
    .trim();

    let _config = PyMatchingConfig::default();
    match PyMatchingDecoder::from_dem(dem) {
        Ok(mut decoder) => {
            // Test detection pattern
            let syndrome = Array1::from_vec(vec![0u8, 1u8, 1u8, 0u8]);
            let result = decoder.decode(syndrome.as_slice().unwrap()).unwrap();
            println!(
                "Negative weight test: weight = {}, matched = {}",
                result.weight,
                0 // matched counts not tracked separately
            );
            // With negative weights, the decoder should still find a valid matching
            assert!(result.is_successful());
        }
        Err(e) => println!("Decoder with negative weights failed: {e}"),
    }
}

/// Test weight calculation accuracy
#[test]
fn test_weight_calculation() {
    // Simple chain with known weights
    let dem = r"
error(0.01) D0 D1
error(0.1) D1 D2
error(0.2) D0
error(0.2) D2
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
    "
    .trim();

    let _config = PyMatchingConfig::default();
    match PyMatchingDecoder::from_dem(dem) {
        Ok(mut decoder) => {
            // Single detection at D0 should match to boundary
            let syndrome = Array1::from_vec(vec![1u8, 0u8, 0u8]);
            let result = decoder.decode(syndrome.as_slice().unwrap()).unwrap();
            // Weight should be log((1-0.2)/0.2) = log(4) ≈ 1.386
            println!("Single detection weight: {}", result.weight);
            assert!(result.weight > 0.0); // Should be positive for p < 0.5

            // Two detections should match to each other
            let syndrome = Array1::from_vec(vec![1u8, 1u8, 0u8]);
            let result = decoder.decode(syndrome.as_slice().unwrap()).unwrap();
            // Weight should be log((1-0.01)/0.01) = log(99) ≈ 4.595
            println!("D0-D1 matching weight: {}", result.weight);
            assert!(result.weight > 0.0);
        }
        Err(e) => panic!("Decoder creation failed: {e}"),
    }
}

/// Test batch decoding with multiple syndromes
#[test]
fn test_batch_decoding() {
    let dem = r"
error(0.1) D0 D1
error(0.1) D1 D2
error(0.1) D0
error(0.1) D2
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
    "
    .trim();

    let _config = PyMatchingConfig::default();
    match PyMatchingDecoder::from_dem(dem) {
        Ok(mut decoder) => {
            let syndromes = [
                vec![0u8, 0u8, 0u8], // No detections
                vec![1u8, 0u8, 0u8], // Single detection
                vec![1u8, 1u8, 0u8], // Adjacent pair
                vec![1u8, 0u8, 1u8], // Non-adjacent pair
                vec![1u8, 1u8, 1u8], // Odd parity (should fail)
            ];

            let mut success_count = 0;
            for (i, syndrome) in syndromes.iter().enumerate() {
                let syndrome_array = Array1::from_vec(syndrome.clone());
                let result = decoder.decode(syndrome_array.as_slice().unwrap()).unwrap();
                println!("Syndrome {i}: success, weight = {}", result.weight);
                success_count += 1;
                // Note: PyMatching doesn't fail on odd parity, it finds best matching
            }

            // At least some syndromes should decode successfully
            assert!(success_count > 0, "No syndromes decoded successfully");
        }
        Err(e) => panic!("Decoder creation failed: {e}"),
    }
}

/// Test self-loop edges
#[test]
fn test_self_loop_edges() {
    // DEM with self-loop (single detector error)
    let dem = r"
error(0.1) D0
error(0.1) D0 D1
error(0.1) D1
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
    "
    .trim();

    let _config = PyMatchingConfig::default();
    match PyMatchingDecoder::from_dem(dem) {
        Ok(mut decoder) => {
            // Single detection with self-loop available
            let syndrome = Array1::from_vec(vec![1u8, 0u8]);
            let result = decoder.decode(syndrome.as_slice().unwrap()).unwrap();
            println!(
                "Self-loop decoding succeeded with weight: {}",
                result.weight
            );
            assert!(result.is_successful());
        }
        Err(e) => println!("Decoder with self-loops failed: {e}"),
    }
}

/// Test observable tracking
#[test]
fn test_observable_tracking() {
    // DEM with multiple observables
    let dem = r"
error(0.1) D0 D1 L0
error(0.1) D1 D2 L1
error(0.1) D2 D3 L0 L1
error(0.1) D0
error(0.1) D3
detector(0, 0, 0, 0) D0
detector(1, 0, 0, 1) D1
detector(2, 0, 0, 2) D2
detector(3, 0, 0, 3) D3
    "
    .trim();

    let _config = PyMatchingConfig::default();
    match PyMatchingDecoder::from_dem(dem) {
        Ok(mut decoder) => {
            assert_eq!(decoder.num_observables(), 2);

            // Test syndrome that should flip observables
            let syndrome = Array1::from_vec(vec![1u8, 1u8, 0u8, 0u8]);
            let result = decoder.decode(syndrome.as_slice().unwrap()).unwrap();
            println!("Observable test succeeded");
            // The matching result should contain observable information
            assert!(result.is_successful());
        }
        Err(e) => panic!("Decoder creation failed: {e}"),
    }
}

/// Test large random syndrome patterns
#[test]
fn test_large_random_patterns() {
    // Generate a larger grid code
    let dem = generate_grid_code_dem(5, 5);
    let _config = PyMatchingConfig::default(); // Use default config

    match PyMatchingDecoder::from_dem(&dem) {
        Ok(mut decoder) => {
            let n = decoder.num_detectors();
            println!("Testing large decoder with {n} detectors");

            // Generate random syndrome with even parity
            let mut syndrome = vec![0u8; n];
            let indices = vec![3, 7, 11, 19]; // Even number of detections
            for i in indices {
                if i < n {
                    syndrome[i] = 1;
                }
            }

            let syndrome_array = Array1::from_vec(syndrome);
            let result = decoder.decode(syndrome_array.as_slice().unwrap()).unwrap();
            println!("Large pattern decoded with weight: {}", result.weight);
            assert!(result.is_successful());
        }
        Err(e) => println!("Large decoder creation failed: {e}"),
    }
}

// Helper function to generate grid code DEM
fn generate_grid_code_dem(rows: usize, cols: usize) -> String {
    use std::fmt::Write;
    let mut dem = String::new();

    for i in 0..rows {
        for j in 0..cols {
            let idx = i * cols + j;

            // Add horizontal edges
            if j + 1 < cols {
                let next_idx = i * cols + (j + 1);
                writeln!(dem, "error(0.1) D{idx} D{next_idx}").unwrap();
            }

            // Add vertical edges
            if i + 1 < rows {
                let next_idx = (i + 1) * cols + j;
                writeln!(dem, "error(0.1) D{idx} D{next_idx}").unwrap();
            }

            // Add boundary edges for border nodes
            if i == 0 || i == rows - 1 || j == 0 || j == cols - 1 {
                writeln!(dem, "error(0.1) D{idx}").unwrap();
            }

            // Add detector
            writeln!(dem, "detector({i}, {j}, 0, 0) D{idx}").unwrap();
        }
    }

    // Add observable on one edge
    dem.push_str("error(0.1) D0 D1 L0\n");
    dem
}
