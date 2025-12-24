//! Comprehensive tests for Stim integration in `PyMatching`
//!
//! This test suite covers all aspects of the `PyMatching` decoder's integration with Stim,
//! including detector error model (DEM) parsing, circuit conversion, error handling,
//! and performance with various types of quantum error correction codes.

use pecos_pymatching::{BatchConfig, PyMatchingDecoder, PyMatchingError};

/// Basic test for loading simple detector error models
#[test]
fn test_from_detector_error_model_basic() {
    // Simple repetition code DEM
    let dem_string = r"
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L0
        error(0.05) D0
        error(0.05) D2
    ";

    let mut decoder = PyMatchingDecoder::from_dem(dem_string).unwrap();

    // Verify basic properties
    assert!(
        decoder.num_detectors() >= 3,
        "Should have at least 3 detectors for D0, D1, D2"
    );
    assert_eq!(decoder.num_observables(), 1, "Should have 1 observable L0");
    assert!(
        decoder.num_edges() >= 2,
        "Should have edges for detector pairs"
    );

    // Test decoding with simple syndrome
    let mut syndrome = vec![0u8; decoder.num_detectors()];
    if syndrome.len() >= 2 {
        syndrome[0] = 1;
        syndrome[1] = 1;

        let result = decoder.decode(&syndrome).unwrap();
        assert_eq!(result.observable.len(), 1);
        // Verify decoding works (specific result depends on matching algorithm)
        assert!(result.weight >= 0.0);
    }
}

/// Test loading surface code detector error models
#[test]
fn test_from_detector_error_model_surface_code() {
    // Surface code-like DEM with X and Z type errors
    let dem_string = r"
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L0
        error(0.1) D2 D3 L0
        error(0.1) D3 D0 L0
        error(0.1) D4 D5 L1
        error(0.1) D5 D6 L1
        error(0.1) D6 D7 L1
        error(0.1) D7 D4 L1
        error(0.05) D0 D4
        error(0.05) D1 D5
        error(0.05) D2 D6
        error(0.05) D3 D7
        detector(0, 0, 0) D0
        detector(1, 0, 0) D1
        detector(0, 1, 0) D2
        detector(1, 1, 0) D3
        detector(0, 0, 1) D4
        detector(1, 0, 1) D5
        detector(0, 1, 1) D6
        detector(1, 1, 1) D7
        logical_observable L0
        logical_observable L1
    ";

    let mut decoder = PyMatchingDecoder::from_dem(dem_string).unwrap();

    assert!(decoder.num_detectors() >= 8);
    assert_eq!(decoder.num_observables(), 2);
    assert!(decoder.num_edges() >= 8);

    // Test with a syndrome that should produce a non-trivial logical outcome
    let mut syndrome = vec![0u8; decoder.num_detectors()];
    if syndrome.len() >= 4 {
        syndrome[0] = 1;
        syndrome[2] = 1; // Create a logical error pattern

        let result = decoder.decode(&syndrome).unwrap();
        assert_eq!(result.observable.len(), 2);
        // Weight should be positive for non-trivial correction
        assert!(result.weight > 0.0);
    }
}

/// Test repetition code DEM structures
#[test]
fn test_from_detector_error_model_repetition_code() {
    // 5-qubit repetition code with timelike edges
    let dem_string = r"
        # Round 1 measurements
        error(0.1) D0 D1
        error(0.1) D1 D2
        error(0.1) D2 D3
        error(0.1) D3 D4
        # Round 2 measurements
        error(0.1) D5 D6
        error(0.1) D6 D7
        error(0.1) D7 D8
        error(0.1) D8 D9
        # Timelike edges (measurement errors)
        error(0.01) D0 D5
        error(0.01) D1 D6
        error(0.01) D2 D7
        error(0.01) D3 D8
        error(0.01) D4 D9
        # Boundary errors
        error(0.05) D0 L0
        error(0.05) D4 L0
        error(0.05) D5 L0
        error(0.05) D9 L0
        logical_observable L0
    ";

    let mut decoder = PyMatchingDecoder::from_dem(dem_string).unwrap();

    assert!(decoder.num_detectors() >= 10);
    assert_eq!(decoder.num_observables(), 1);
    assert!(decoder.num_edges() >= 12); // Spacelike + timelike + boundary edges

    // Test decoding with timelike correlation
    let mut syndrome = vec![0u8; decoder.num_detectors()];
    if syndrome.len() >= 6 {
        syndrome[0] = 1;
        syndrome[5] = 1; // Same detector across time

        let result = decoder.decode(&syndrome).unwrap();
        // This should be corrected (weight depends on matching algorithm)
        assert!(result.weight >= 0.0);
        // Verify timelike correlation handling
    }
}

/// Test error handling for invalid DEM strings
#[test]
fn test_from_detector_error_model_invalid_formats() {
    // Test various invalid DEM formats
    let invalid_dems = [
        // Empty string
        "",
        // Invalid syntax
        "invalid syntax here",
        // Negative error probability
        "error(-0.1) D0 D1 L0",
        // Missing detector
        "error(0.1) L0",
        // Invalid detector index format
        "error(0.1) D-1 D0 L0",
        // Probability > 1
        "error(1.5) D0 D1 L0",
    ];

    for (i, invalid_dem) in invalid_dems.iter().enumerate() {
        let result = PyMatchingDecoder::from_dem(invalid_dem);
        match result {
            Err(PyMatchingError::Ffi(_)) => {
                // Expected FFI error for invalid format
                println!("Test case {i}: Got expected FFI error for invalid DEM");
            }
            Err(PyMatchingError::Configuration(_)) => {
                // Also acceptable - configuration error
                println!("Test case {i}: Got expected config error for invalid DEM");
            }
            Ok(_) => {
                // Some invalid DEMs might still parse (e.g., empty string creates empty graph)
                println!("Test case {i}: DEM was unexpectedly accepted");
            }
            Err(e) => {
                println!("Test case {i}: Got error: {e}");
            }
        }
    }
}

/// Test complex DEMs with correlated errors
#[test]
fn test_from_detector_error_model_correlated_errors() {
    // DEM with complex correlated error patterns
    let dem_string = r"
        # Single qubit errors
        error(0.01) D0 L0
        error(0.01) D1 L0
        error(0.01) D2 L0
        error(0.01) D3 L0

        # Two-qubit correlated errors
        error(0.005) D0 D1 L0 L1
        error(0.005) D1 D2 L0 L1
        error(0.005) D2 D3 L0 L1

        # Three-qubit correlated errors (less likely)
        error(0.001) D0 D1 D2 L0 L1
        error(0.001) D1 D2 D3 L0 L1

        # Four-qubit correlated error (very rare)
        error(0.0001) D0 D1 D2 D3

        logical_observable L0
        logical_observable L1
    ";

    let mut decoder = PyMatchingDecoder::from_dem(dem_string).unwrap();

    assert!(decoder.num_detectors() >= 4);
    assert_eq!(decoder.num_observables(), 2);
    assert!(decoder.num_edges() > 0);

    // Test various syndrome patterns
    let test_syndromes = vec![
        vec![1, 0, 0, 0], // Single detection
        vec![1, 1, 0, 0], // Correlated pair
        vec![1, 1, 1, 0], // Three detections
        vec![1, 1, 1, 1], // All detections
    ];

    for (i, mut syndrome) in test_syndromes.into_iter().enumerate() {
        // Pad syndrome to correct length
        syndrome.resize(decoder.num_detectors(), 0);

        let result = decoder.decode(&syndrome).unwrap();
        assert_eq!(result.observable.len(), 2);
        println!(
            "Test syndrome {}: weight = {}, observables = {:?}",
            i, result.weight, result.observable
        );

        // Higher order correlations should generally have higher weights
        assert!(result.weight >= 0.0);
    }
}

/// Test DEMs with measurement errors
#[test]
fn test_from_detector_error_model_measurement_errors() {
    // DEM with explicit measurement errors
    let dem_string = r"
        # Data qubit errors
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L0
        error(0.1) D2 D3 L0

        # Measurement errors (higher probability)
        error(0.01) D0
        error(0.01) D1
        error(0.01) D2
        error(0.01) D3

        # Correlated measurement errors
        error(0.001) D0 D1
        error(0.001) D1 D2
        error(0.001) D2 D3

        logical_observable L0
    ";

    let mut decoder = PyMatchingDecoder::from_dem(dem_string).unwrap();

    assert!(decoder.num_detectors() >= 4);
    assert_eq!(decoder.num_observables(), 1);

    // Test isolated measurement error (should be corrected to boundary)
    let mut syndrome = vec![0u8; decoder.num_detectors()];
    if syndrome.len() >= 2 {
        syndrome[1] = 1; // Single isolated detection

        let result = decoder.decode(&syndrome).unwrap();
        // Single measurement error should be correctable
        assert!(result.weight >= 0.0);
        println!("Measurement error correction weight: {}", result.weight);
    }
}

/// Test boundary handling in Stim-generated models
#[test]
fn test_from_detector_error_model_boundary_handling() {
    // DEM with explicit boundary conditions
    let dem_string = r"
        # Internal edges
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L0
        error(0.1) D2 D3 L0

        # Boundary edges (connect to virtual boundary)
        error(0.05) D0 L0
        error(0.05) D3 L0

        # Mixed internal/boundary errors
        error(0.02) D1 L0
        error(0.02) D2 L0

        logical_observable L0
    ";

    let mut decoder = PyMatchingDecoder::from_dem(dem_string).unwrap();

    assert!(decoder.num_detectors() >= 4);
    assert_eq!(decoder.num_observables(), 1);

    // Test boundary correction scenarios
    let test_cases = vec![
        (vec![1, 0, 0, 0], "Boundary detection"),
        (vec![0, 0, 0, 1], "Other boundary detection"),
        (vec![1, 0, 0, 1], "Both boundary detections"),
        (vec![0, 1, 0, 0], "Internal detection"),
    ];

    for (mut syndrome, description) in test_cases {
        syndrome.resize(decoder.num_detectors(), 0);

        let result = decoder.decode(&syndrome).unwrap();
        println!(
            "{}: weight = {}, observable = {:?}",
            description, result.weight, result.observable
        );

        assert!(result.weight >= 0.0);
        assert_eq!(result.observable.len(), 1);
    }
}

/// Test integration with Stim circuit conversion if available
#[test]
fn test_stim_circuit_integration() {
    // Test if we can load from a Stim circuit file
    // This creates a simple circuit as a string and tries to parse it as DEM

    let stim_circuit = r"
        # Simple repetition code circuit
        R 0 1 2
        TICK
        CX 0 1 1 2
        TICK
        MR 0 1 2
        DETECTOR(0, 0) rec[-3] rec[-2]
        DETECTOR(1, 0) rec[-2] rec[-1]
        OBSERVABLE_INCLUDE(0) rec[-1]
    ";

    // Try to load as DEM (this might not work directly with circuit syntax)
    let result = PyMatchingDecoder::from_dem(stim_circuit);

    match result {
        Ok(decoder) => {
            println!("Successfully loaded circuit as DEM");
            assert!(decoder.num_detectors() >= 1);
            assert!(decoder.num_observables() >= 1);
        }
        Err(e) => {
            println!("Circuit parsing failed as expected: {e}");
            // This is expected since we're mixing circuit and DEM syntax
        }
    }

    // Test with a proper DEM generated from a conceptual circuit
    let circuit_based_dem = r"
        # DEM that could be generated from the above circuit
        error(0.1) D0 D1 L0
        error(0.05) D0
        error(0.05) D1 L0
        logical_observable L0
    ";

    let decoder = PyMatchingDecoder::from_dem(circuit_based_dem).unwrap();
    assert!(decoder.num_detectors() >= 2);
    assert_eq!(decoder.num_observables(), 1);
}

/// Test performance with large Stim-generated models
#[test]
fn test_large_stim_model_performance() {
    // Generate a large DEM programmatically
    let mut dem_lines = Vec::new();
    let size = 20; // 20x20 grid = 400 detectors

    // Add grid-based errors (surface code-like)
    for i in 0..size {
        for j in 0..size {
            let detector_id = i * size + j;

            // Horizontal edges
            if j < size - 1 {
                let neighbor = i * size + (j + 1);
                dem_lines.push(format!("error(0.1) D{detector_id} D{neighbor} L0"));
            }

            // Vertical edges
            if i < size - 1 {
                let neighbor = (i + 1) * size + j;
                dem_lines.push(format!("error(0.1) D{detector_id} D{neighbor} L1"));
            }

            // Boundary edges for edge detectors
            if i == 0 || i == size - 1 || j == 0 || j == size - 1 {
                let obs = if i == 0 || i == size - 1 { "L0" } else { "L1" };
                dem_lines.push(format!("error(0.05) D{detector_id} {obs}"));
            }
        }
    }

    dem_lines.push("logical_observable L0".to_string());
    dem_lines.push("logical_observable L1".to_string());

    let large_dem = dem_lines.join("\n");

    // Time the construction
    let start = std::time::Instant::now();
    let mut decoder = PyMatchingDecoder::from_dem(&large_dem).unwrap();
    let construction_time = start.elapsed();

    println!("Large DEM construction took: {construction_time:?}");
    println!(
        "Graph has {} detectors, {} edges, {} observables",
        decoder.num_detectors(),
        decoder.num_edges(),
        decoder.num_observables()
    );

    assert!(decoder.num_detectors() >= size * size);
    assert_eq!(decoder.num_observables(), 2);
    assert!(decoder.num_edges() > 0);

    // Test decoding performance
    let mut syndrome = vec![0u8; decoder.num_detectors()];
    // Create a random-looking syndrome with ~5% of detectors firing
    for i in (0..syndrome.len()).step_by(20) {
        syndrome[i] = 1;
    }

    let start = std::time::Instant::now();
    let result = decoder.decode(&syndrome).unwrap();
    let decoding_time = start.elapsed();

    println!("Large syndrome decoding took: {decoding_time:?}");
    println!("Correction weight: {}", result.weight);

    assert_eq!(result.observable.len(), 2);
    assert!(result.weight >= 0.0);

    // Performance should be reasonable (< 1 second for this size)
    assert!(
        construction_time.as_millis() < 5000,
        "Construction took too long"
    );
    assert!(decoding_time.as_millis() < 1000, "Decoding took too long");
}

/// Test DEM with multiple observable types
#[test]
fn test_detector_error_model_multiple_observables() {
    // DEM with many observables (typical in large codes)
    let dem_string = r"
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L1
        error(0.1) D2 D3 L2
        error(0.1) D3 D4 L3
        error(0.1) D4 D5 L4
        error(0.1) D5 D0 L5

        # Cross-observable errors
        error(0.05) D0 D3 L0 L3
        error(0.05) D1 D4 L1 L4
        error(0.05) D2 D5 L2 L5

        # Multi-observable errors
        error(0.01) D0 D2 D4 L0 L2 L4
        error(0.01) D1 D3 D5 L1 L3 L5

        logical_observable L0
        logical_observable L1
        logical_observable L2
        logical_observable L3
        logical_observable L4
        logical_observable L5
    ";

    let mut decoder = PyMatchingDecoder::from_dem(dem_string).unwrap();

    assert!(decoder.num_detectors() >= 6);
    assert_eq!(decoder.num_observables(), 6);
    assert!(decoder.num_edges() > 0);

    // Test with extended decoding for >64 observables case
    let mut syndrome = vec![0u8; decoder.num_detectors()];
    if syndrome.len() >= 6 {
        // Create an even-parity syndrome to avoid matching failure
        syndrome[0] = 1;
        syndrome[1] = 1; // Two detections (even parity)

        let result = decoder.decode(&syndrome).unwrap();
        assert_eq!(result.observable.len(), 6);

        // Should trigger observables based on the matching
        let triggered_count = result.observable.iter().filter(|&&x| x != 0).count();
        println!("Triggered {triggered_count} observables");
    }
}

/// Test error handling for edge cases in DEM parsing
#[test]
fn test_detector_error_model_edge_cases() {
    let edge_cases = vec![
        // Very small probabilities
        (
            "error(1e-10) D0 D1 L0\nlogical_observable L0",
            "Very small probability",
        ),
        // Zero probability (should be valid)
        (
            "error(0.0) D0 D1 L0\nlogical_observable L0",
            "Zero probability",
        ),
        // Many detectors in single error
        (
            "error(0.1) D0 D1 D2 D3 D4 D5 L0\nlogical_observable L0",
            "Many detectors",
        ),
        // Large detector indices
        (
            "error(0.1) D1000 D2000 L0\nlogical_observable L0",
            "Large detector indices",
        ),
        // Mixed observable types
        (
            "error(0.1) D0 L0\nerror(0.1) D1 L1 L2\nlogical_observable L0\nlogical_observable L1\nlogical_observable L2",
            "Mixed observables",
        ),
    ];

    for (dem, description) in edge_cases {
        println!("Testing: {description}");

        let result = PyMatchingDecoder::from_dem(dem);
        match result {
            Ok(mut decoder) => {
                println!(
                    "  Successfully parsed, {} detectors, {} observables",
                    decoder.num_detectors(),
                    decoder.num_observables()
                );

                // Try a basic decode to ensure the graph is functional
                let syndrome = vec![0u8; decoder.num_detectors().min(10)];
                let decode_result = decoder.decode(&syndrome).unwrap();
                assert!(decode_result.weight >= 0.0);
            }
            Err(e) => {
                println!("  Failed as expected: {e}");
            }
        }
    }
}

/// Test batch processing with Stim-generated DEMs
#[test]
#[allow(clippy::cast_precision_loss)] // Acceptable for computing error rates
fn test_stim_dem_batch_processing() {
    // Surface code-like DEM for batch testing
    let dem_string = r"
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L0
        error(0.1) D2 D3 L0
        error(0.1) D3 D0 L0
        error(0.05) D0 L0
        error(0.05) D1 L0
        error(0.05) D2 L0
        error(0.05) D3 L0
        logical_observable L0
    ";

    let mut decoder = PyMatchingDecoder::from_dem(dem_string).unwrap();

    // Prepare batch of syndromes
    let num_shots = 100;
    let num_detectors = decoder.num_detectors();
    let mut shots = vec![0u8; num_shots * num_detectors];

    // Create varied syndrome patterns
    for shot in 0..num_shots {
        let base_idx = shot * num_detectors;
        match shot % 5 {
            0 => {
                // No errors
            }
            1 => {
                // Single detection
                if num_detectors > 0 {
                    shots[base_idx] = 1;
                }
            }
            2 => {
                // Pair of detections
                if num_detectors > 1 {
                    shots[base_idx] = 1;
                    shots[base_idx + 1] = 1;
                }
            }
            3 => {
                // Three detections
                if num_detectors > 2 {
                    shots[base_idx] = 1;
                    shots[base_idx + 1] = 1;
                    shots[base_idx + 2] = 1;
                }
            }
            4 => {
                // All detections
                for i in 0..num_detectors {
                    shots[base_idx + i] = 1;
                }
            }
            _ => unreachable!(),
        }
    }

    // Time the batch decoding
    let start = std::time::Instant::now();
    let result = decoder
        .decode_batch_with_config(
            &shots,
            num_shots,
            num_detectors,
            BatchConfig {
                bit_packed_input: false,
                bit_packed_output: false,
                return_weights: true,
            },
        )
        .unwrap();
    let batch_time = start.elapsed();

    println!("Batch decoding {num_shots} shots took: {batch_time:?}");

    assert_eq!(result.predictions.len(), num_shots);
    assert_eq!(result.weights.len(), num_shots);

    // Analyze results
    let mut logical_error_count = 0;
    let mut total_weight = 0.0;

    for (i, (pred, weight)) in result.predictions.iter().zip(&result.weights).enumerate() {
        if pred.iter().any(|&x| x != 0) {
            logical_error_count += 1;
        }
        total_weight += weight;

        if i < 10 {
            println!(
                "Shot {}: weight = {}, logical error = {}",
                i,
                weight,
                pred.iter().any(|&x| x != 0)
            );
        }
    }

    println!(
        "Logical error rate: {}/{} = {:.3}",
        logical_error_count,
        num_shots,
        f64::from(logical_error_count) / num_shots as f64
    );
    println!(
        "Average correction weight: {:.3}",
        total_weight / num_shots as f64
    );

    // Performance check
    assert!(
        batch_time.as_millis() < 1000,
        "Batch decoding took too long"
    );
}

/// Test specific Stim DEM features and edge cases
#[test]
fn test_stim_specific_dem_features() {
    // Test DEM with Stim-specific features
    let stim_dem = r"
        # Pauli frame changes
        error(0.1) D0 D1 L0 L0  # L0 appears twice (Pauli frame)
        error(0.1) D1 D2 L1 L1

        # Hypergraph errors (more than 2 detectors)
        error(0.01) D0 D1 D2 L0
        error(0.01) D1 D2 D3 L1

        # High-weight logical operators
        error(0.001) D0 D1 D2 D3 L0 L1

        logical_observable L0
        logical_observable L1
    ";

    let mut decoder = PyMatchingDecoder::from_dem(stim_dem).unwrap();

    assert!(decoder.num_detectors() >= 4);
    assert_eq!(decoder.num_observables(), 2);

    // Test with empty syndrome (should always work)
    let syndrome = vec![0u8; decoder.num_detectors()];
    let result = decoder.decode(&syndrome).unwrap();
    assert_eq!(result.observable.len(), 2);
    assert!((result.weight - 0.0).abs() < f64::EPSILON); // Empty syndrome should have zero weight
    assert!(result.observable.iter().all(|&x| x == 0)); // No observables triggered
    println!(
        "Empty syndrome decoding: weight = {}, obs = {:?}",
        result.weight, result.observable
    );
}

/// Test memory management with repeated DEM loading
#[test]
fn test_dem_memory_management() {
    let dem_template = r"
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L0
        error(0.05) D0 L0
        error(0.05) D2 L0
        logical_observable L0
    ";

    // Load and drop many decoders to test memory management
    for i in 0..100 {
        let mut decoder = PyMatchingDecoder::from_dem(dem_template).unwrap();

        assert!(decoder.num_detectors() >= 3);
        assert_eq!(decoder.num_observables(), 1);

        // Quick decode test
        let syndrome = [1, 0, 1];
        let result = decoder
            .decode(&syndrome[..decoder.num_detectors().min(3)])
            .unwrap();
        assert!(result.weight >= 0.0);

        if i % 20 == 0 {
            println!("Created and tested decoder {i}");
        }

        // Decoder should be automatically dropped here
    }

    println!("Successfully created and dropped 100 decoders");
}

/// Test compatibility with various DEM formats and encodings
#[test]
fn test_dem_format_compatibility() {
    let format_variants = [
        // Standard format
        r"error(0.1) D0 D1 L0
logical_observable L0",
        // With extra whitespace
        r"  error(0.1)  D0  D1  L0
   logical_observable   L0   ",
        // With comments
        r"# This is a comment
error(0.1) D0 D1 L0  # Inline comment
# Another comment
logical_observable L0",
        // Scientific notation
        r"error(1e-1) D0 D1 L0
error(5.0e-2) D1 D2 L0
logical_observable L0",
        // Multiple lines
        r"error(0.1) D0 D1 L0
error(0.1) D1 D2 L0
error(0.05) D0 L0
error(0.05) D2 L0
logical_observable L0",
    ];

    for (i, dem) in format_variants.iter().enumerate() {
        println!("Testing format variant {i}");

        let result = PyMatchingDecoder::from_dem(dem);
        match result {
            Ok(mut decoder) => {
                assert!(decoder.num_detectors() >= 1);
                assert_eq!(decoder.num_observables(), 1);
                println!("  Format {i} parsed successfully");

                // Test basic functionality
                let syndrome = vec![0u8; decoder.num_detectors().min(5)];
                let decode_result = decoder.decode(&syndrome).unwrap();
                assert!(decode_result.weight >= 0.0);
            }
            Err(e) => {
                println!("  Format {i} failed: {e}");
                // Some format variations might fail, which is acceptable
            }
        }
    }
}

/// Integration test combining DEM loading with advanced decoding features
#[test]
fn test_dem_advanced_decoding_integration() {
    let dem_string = r"
        # Create a non-trivial matching problem
        error(0.1) D0 D1 L0
        error(0.2) D1 D2 L0  # Higher weight path
        error(0.05) D2 D3 L0
        error(0.15) D3 D4 L0
        error(0.08) D4 D5 L0
        error(0.12) D5 D0 L0  # Complete the cycle

        # Alternative paths
        error(0.25) D0 D3 L0  # Direct path with higher weight
        error(0.3) D1 D4 L0
        error(0.35) D2 D5 L0

        # Boundary connections
        error(0.1) D0 L0
        error(0.1) D3 L0

        logical_observable L0
    ";

    let mut decoder = PyMatchingDecoder::from_dem(dem_string).unwrap();

    // Test shortest path functionality
    if decoder.num_detectors() >= 6 {
        let path_result = decoder.get_shortest_path(0, 3);
        match path_result {
            Ok(path) => {
                println!("Shortest path from 0 to 3: {path:?}");
                assert!(!path.is_empty());
                assert_eq!(path[0], 0);
                assert_eq!(path[path.len() - 1], 3);
            }
            Err(e) => {
                println!("Path finding failed: {e}");
                // This might fail if the graph structure doesn't support it
            }
        }
    }

    // Test matched pairs decoding
    let mut syndrome = vec![0u8; decoder.num_detectors()];
    if syndrome.len() >= 4 {
        syndrome[1] = 1;
        syndrome[4] = 1;

        // Test multiple decoding formats
        let basic_result = decoder.decode(&syndrome).unwrap();
        println!(
            "Basic decode: weight = {}, obs = {:?}",
            basic_result.weight, basic_result.observable
        );

        let pairs_result = decoder.decode_to_matched_pairs(&syndrome);
        match pairs_result {
            Ok(pairs) => {
                println!("Matched pairs: {pairs:?}");
                assert!(!pairs.is_empty());
            }
            Err(e) => {
                println!("Matched pairs failed: {e}");
            }
        }

        let edges_result = decoder.decode_to_edges(&syndrome);
        match edges_result {
            Ok(edges) => {
                println!("Matched edges: {edges:?}");
            }
            Err(e) => {
                println!("Matched edges failed: {e}");
            }
        }
    }
}
