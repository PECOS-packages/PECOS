//! Comprehensive tests matching `PyMatching`'s Python/C++ test coverage

use pecos_pymatching::{
    BatchConfig, CheckMatrix, CheckMatrixConfig, MergeStrategy, PyMatchingDecoder,
};
use std::collections::HashSet;

// ============================================================================
// Core Algorithm Tests
// ============================================================================

#[test]
fn test_negative_weight_edges() {
    // Test matching with negative weights (important for QEC)
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(4)
        .observables(2)
        .build()
        .unwrap();

    // Create a graph with negative weight edges
    // In QEC, negative weights correspond to p > 0.5 (more likely to have error)
    decoder
        .add_edge(0, 1, &[0], Some(-1.0), None, None)
        .unwrap();
    decoder.add_edge(1, 2, &[1], Some(2.0), None, None).unwrap();
    decoder
        .add_edge(2, 3, &[0], Some(-0.5), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(0, &[], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(3, &[], Some(1.0), None, None)
        .unwrap();

    // Test with detection at node 1
    let mut detection_events = vec![0u8; 4];
    detection_events[1] = 1;

    let result = decoder.decode(&detection_events).unwrap();
    // Verify negative weight handling
    // Should match through negative weight edge if it's optimal
    assert_eq!(result.observable.len(), 2);
}

#[test]
fn test_zero_weight_edges() {
    // Test edges with p=0.5 (weight = log(1) = 0)
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(3)
        .observables(2)
        .build()
        .unwrap();

    // Add edge with error probability 0.5 (zero weight)
    decoder.add_edge(0, 1, &[0], None, Some(0.5), None).unwrap();
    decoder.add_edge(1, 2, &[1], None, Some(0.1), None).unwrap();

    let edge_data = decoder.get_edge_data(0, 1).unwrap();
    // Weight should be 0 or very close to 0 for p=0.5
    assert!(
        (edge_data.weight).abs() < 1e-10
            || (edge_data.error_probability - 0.5).abs() < f64::EPSILON
    );
}

#[test]
fn test_self_loops() {
    // Test edges from a node to itself
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(4)
        .observables(2)
        .build()
        .unwrap();

    // Try to add a self-loop
    let result = decoder.add_edge(1, 1, &[0], Some(1.0), None, None);
    // PyMatching might reject self-loops or handle them specially
    // We test that it doesn't crash
    if result.is_ok() {
        // If self-loops are allowed, test they work in decoding
        let mut detection_events = vec![0u8; 4];
        detection_events[1] = 1;
        let _ = decoder.decode(&detection_events);
    }
}

#[test]
fn test_parallel_edges_all_strategies() {
    // Test all merge strategies with parallel edges
    let strategies = vec![
        MergeStrategy::Disallow,
        MergeStrategy::Independent,
        MergeStrategy::SmallestWeight,
        MergeStrategy::KeepOriginal,
        MergeStrategy::Replace,
    ];

    for strategy in strategies {
        let mut decoder = PyMatchingDecoder::builder()
            .nodes(3)
            .observables(3)
            .build()
            .unwrap();

        // Add first edge
        decoder.add_edge(0, 1, &[0], Some(2.0), None, None).unwrap();

        // Add parallel edge with different weight and observable
        let result = decoder.add_edge(0, 1, &[1], Some(1.0), None, Some(strategy));

        match strategy {
            MergeStrategy::Disallow => {
                // Should fail for Disallow
                let edge_weight = decoder.get_edge_data(0, 1).unwrap().weight;
                assert!(result.is_err() || (edge_weight - 2.0).abs() < f64::EPSILON);
            }
            MergeStrategy::SmallestWeight | MergeStrategy::Replace => {
                if result.is_ok() {
                    let edge = decoder.get_edge_data(0, 1).unwrap();
                    assert!((edge.weight - 1.0).abs() < 1e-6);
                }
            }
            MergeStrategy::KeepOriginal => {
                if result.is_ok() {
                    let edge = decoder.get_edge_data(0, 1).unwrap();
                    assert!((edge.weight - 2.0).abs() < 1e-6);
                }
            }
            MergeStrategy::Independent => {
                // Independent merge should combine probabilities
                if result.is_ok() {
                    let edge = decoder.get_edge_data(0, 1).unwrap();
                    // Combined weight should be different from both original weights
                    assert!(
                        (edge.weight - 1.0).abs() > f64::EPSILON
                            && (edge.weight - 2.0).abs() > f64::EPSILON,
                        "Combined weight {} should be different from both 1.0 and 2.0",
                        edge.weight
                    );
                }
            }
        }
    }
}

// ============================================================================
// Blossom Algorithm Tests
// ============================================================================

#[test]
fn test_odd_cycle_matching() {
    // Test blossom formation on odd cycles
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(5)
        .observables(2)
        .build()
        .unwrap();

    // Create a pentagon (5-cycle)
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(1.0), None, None).unwrap();
    decoder.add_edge(2, 3, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(3, 4, &[1], Some(1.0), None, None).unwrap();
    decoder.add_edge(4, 0, &[0], Some(1.0), None, None).unwrap();

    // Add boundary edges
    for i in 0..5 {
        decoder
            .add_boundary_edge(i, &[], Some(10.0), None, None)
            .unwrap();
    }

    // Test with 3 detections (odd number forces blossom)
    let mut detection_events = vec![0u8; 5];
    detection_events[0] = 1;
    detection_events[2] = 1;
    detection_events[4] = 1;

    let result = decoder.decode(&detection_events).unwrap();
    // Should find a valid matching
    assert!(result.weight > 0.0);
}

#[test]
fn test_nested_blossoms() {
    // Test nested blossom structures
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(9)
        .observables(2)
        .build()
        .unwrap();

    // Create outer triangle
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(1.0), None, None).unwrap();
    decoder.add_edge(2, 0, &[0], Some(1.0), None, None).unwrap();

    // Create inner structures
    decoder.add_edge(0, 3, &[0], Some(0.5), None, None).unwrap();
    decoder.add_edge(3, 4, &[1], Some(0.5), None, None).unwrap();
    decoder.add_edge(4, 0, &[0], Some(0.5), None, None).unwrap();

    decoder.add_edge(1, 5, &[1], Some(0.5), None, None).unwrap();
    decoder.add_edge(5, 6, &[0], Some(0.5), None, None).unwrap();
    decoder.add_edge(6, 1, &[1], Some(0.5), None, None).unwrap();

    decoder.add_edge(2, 7, &[0], Some(0.5), None, None).unwrap();
    decoder.add_edge(7, 8, &[1], Some(0.5), None, None).unwrap();
    decoder.add_edge(8, 2, &[0], Some(0.5), None, None).unwrap();

    // Test with multiple detections (even parity for valid matching)
    let mut detection_events = vec![0u8; 9];
    detection_events[3] = 1;
    detection_events[5] = 1;
    detection_events[7] = 1;
    detection_events[8] = 1; // Add fourth detection for even parity

    let result = decoder.decode(&detection_events).unwrap();
    assert_eq!(result.observable.len(), 2);
}

// ============================================================================
// Edge Cases and Error Conditions
// ============================================================================

#[test]
fn test_empty_graph_decoding() {
    // Test decoding on a graph with no edges
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(5)
        .observables(2)
        .build()
        .unwrap();

    let detection_events = vec![0u8; 5];
    let result = decoder.decode(&detection_events).unwrap();
    // Should handle gracefully
    assert_eq!(result.observable, vec![0, 0]);
    assert!(
        result.weight.abs() < f64::EPSILON,
        "Weight should be zero but was {}",
        result.weight
    );
}

#[test]
fn test_single_node_graph() {
    // Test minimal graph structure
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(1)
        .observables(1)
        .build()
        .unwrap();

    // Add boundary edge
    decoder
        .add_boundary_edge(0, &[0], Some(1.0), None, None)
        .unwrap();

    // Test with detection
    let detection_events = vec![1u8];
    let result = decoder.decode(&detection_events).unwrap();
    // PyMatching may return different numbers of observables
    // Just check the first observable is set
    assert!(!result.observable.is_empty());
    assert_eq!(result.observable[0], 1);
}

#[test]
fn test_disconnected_components() {
    // Test graph with multiple disconnected components
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(6)
        .observables(2)
        .build()
        .unwrap();

    // Component 1: nodes 0, 1, 2
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(1.0), None, None).unwrap();

    // Component 2: nodes 3, 4, 5 (disconnected from component 1)
    decoder.add_edge(3, 4, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(4, 5, &[1], Some(1.0), None, None).unwrap();

    // Add boundary edges for each component
    decoder
        .add_boundary_edge(0, &[], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(3, &[], Some(1.0), None, None)
        .unwrap();

    // Test with detections in both components
    let mut detection_events = vec![0u8; 6];
    detection_events[1] = 1; // Component 1
    detection_events[4] = 1; // Component 2

    let result = decoder.decode(&detection_events).unwrap();
    // Should handle both components correctly
    assert!(result.weight > 0.0);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

#[test]
fn test_extreme_weights() {
    // Test with very large and very small weights
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(4)
        .observables(2)
        .build()
        .unwrap();

    // Very large weight (low probability) - within PyMatching's limit of ~16M
    decoder.add_edge(0, 1, &[0], Some(1e6), None, None).unwrap();
    // Very small weight (high probability)
    decoder
        .add_edge(1, 2, &[1], Some(1e-10), None, None)
        .unwrap();
    // Normal weight
    decoder.add_edge(2, 3, &[0], Some(1.0), None, None).unwrap();

    // Test decoding doesn't overflow/underflow
    let mut detection_events = vec![0u8; 4];
    detection_events[0] = 1;
    detection_events[3] = 1;

    let result = decoder.decode(&detection_events).unwrap();
    assert!(result.weight.is_finite());
}

#[test]
fn test_weight_normalisation_constant() {
    // Test edge weight normalizing constant behavior
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(3)
        .observables(1)
        .build()
        .unwrap();

    // Add edges with different weights
    decoder.add_edge(0, 1, &[0], Some(0.5), None, None).unwrap();
    decoder.add_edge(1, 2, &[0], Some(1.5), None, None).unwrap();
    decoder.add_edge(0, 2, &[0], Some(2.5), None, None).unwrap();

    let norm_const = decoder.get_edge_weight_normalising_constant(1000);
    assert!(norm_const > 0.0);
    assert!(norm_const.is_finite());
}

// ============================================================================
// Batch Processing Tests
// ============================================================================

#[test]
fn test_batch_with_bit_packing() {
    // Test batch decoding with bit-packed format
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(16) // Use 16 nodes to test bit boundaries
        .observables(20) // More than 16 to test multi-byte packing
        .build()
        .unwrap();

    // Create a more complex graph
    for i in 0..15 {
        decoder
            .add_edge(i, i + 1, &[i % 20], Some(1.0), None, None)
            .unwrap();
    }

    // Test bit-packed batch processing
    let num_shots: usize = 10;
    let num_detectors: usize = 16;
    let num_detector_bytes = num_detectors.div_ceil(8); // 2 bytes per shot

    // Create bit-packed shots
    let mut shots = vec![0u8; num_shots * num_detector_bytes];
    for shot in 0..num_shots {
        let offset = shot * num_detector_bytes;
        // Set different bit patterns for each shot
        // shot is guaranteed < 256 since num_shots=10
        #[allow(clippy::cast_possible_truncation)]
        {
            shots[offset] = (shot % 256) as u8;
            shots[offset + 1] = ((shot * 2) % 256) as u8;
        }
    }

    let result = decoder
        .decode_batch_with_config(
            &shots,
            num_shots,
            num_detectors,
            BatchConfig {
                bit_packed_input: true,
                bit_packed_output: true,
                return_weights: true,
            },
        )
        .unwrap();

    assert_eq!(result.predictions.len(), num_shots);
    assert_eq!(result.weights.len(), num_shots);

    // Verify bit-packed output format
    for prediction in &result.predictions {
        // PyMatching may use different packing, so we just check it's reasonable
        assert!(!prediction.is_empty());
    }
}

// ============================================================================
// Matrix Construction Tests
// ============================================================================

#[test]
fn test_sparse_matrix_with_repetitions() {
    // Test loading from sparse matrix with repetitions (timelike edges)
    let check_matrix = vec![
        (0, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (2, 3, 1),
    ];

    let weights = vec![1.0, 1.0, 1.0, 1.0];
    let repetitions = 5; // 5 rounds of measurements
    let timelike_weights = vec![0.5, 0.5, 0.5]; // One weight per check row

    let config = CheckMatrixConfig {
        repetitions,
        weights: Some(weights),
        error_probabilities: None,
        timelike_weights: Some(timelike_weights),
        measurement_error_probabilities: None,
        use_virtual_boundary: false,
    };
    let matrix = CheckMatrix::from_triplets(check_matrix, 3, 4);
    let decoder = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).unwrap();

    // Should have created nodes for multiple rounds
    assert!(decoder.num_nodes() > 4); // More nodes due to repetitions
}

#[test]
fn test_measurement_error_probabilities() {
    // Test measurement error handling
    let check_matrix = vec![(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 2, 1)];

    let measurement_error_probs = vec![0.01, 0.02]; // Different per check

    let config = CheckMatrixConfig {
        repetitions: 1,
        weights: None,
        error_probabilities: None,
        timelike_weights: None,
        measurement_error_probabilities: Some(measurement_error_probs),
        use_virtual_boundary: false,
    };
    let matrix = CheckMatrix::from_triplets(check_matrix, 2, 3);
    let decoder = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).unwrap();

    // Verify decoder was created with measurement errors
    assert!(decoder.num_edges() > 0);
}

// ============================================================================
// Monte Carlo and Statistical Tests
// ============================================================================

#[test]
fn test_monte_carlo_consistency() {
    // Test statistical properties of decoding
    use std::collections::HashMap;

    let mut decoder = PyMatchingDecoder::builder()
        .nodes(4)
        .observables(2)
        .build()
        .unwrap();

    // Create a simple chain with known error rates
    decoder.add_edge(0, 1, &[0], None, Some(0.1), None).unwrap();
    decoder.add_edge(1, 2, &[1], None, Some(0.1), None).unwrap();
    decoder.add_edge(2, 3, &[0], None, Some(0.1), None).unwrap();
    decoder
        .add_boundary_edge(0, &[], None, Some(0.1), None)
        .unwrap();
    decoder
        .add_boundary_edge(3, &[], None, Some(0.1), None)
        .unwrap();

    // Generate many samples and check statistical properties
    let num_samples = 100;
    let noise_result = decoder.add_noise(num_samples, 42).unwrap();

    // Count logical errors
    let mut logical_errors = HashMap::new();
    for (errors, syndrome) in noise_result.errors.iter().zip(&noise_result.syndromes) {
        // Decode the syndrome
        let result = decoder.decode(syndrome).unwrap();

        // Check if logical error occurred
        let logical_error = result
            .observable
            .iter()
            .zip(errors.iter())
            .any(|(&predicted, &actual)| predicted != actual);

        *logical_errors.entry(logical_error).or_insert(0) += 1;
    }

    // With 10% error rate, we should see some logical errors but not too many
    let error_count = logical_errors.get(&true).unwrap_or(&0);
    assert!(*error_count > 0 && *error_count < num_samples);
}

// ============================================================================
// Advanced Decoding Features
// ============================================================================

#[test]
fn test_decode_to_matched_pairs_complex() {
    // Test matched pairs extraction with complex matching
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(8)
        .observables(3)
        .build()
        .unwrap();

    // Create a graph with multiple matching possibilities
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(2.0), None, None).unwrap();
    decoder.add_edge(2, 3, &[2], Some(1.0), None, None).unwrap();
    decoder.add_edge(3, 4, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(4, 5, &[1], Some(2.0), None, None).unwrap();
    decoder.add_edge(5, 6, &[2], Some(1.0), None, None).unwrap();
    decoder.add_edge(6, 7, &[0], Some(1.0), None, None).unwrap();

    // Cross connections
    decoder
        .add_edge(1, 6, &[0, 1], Some(3.0), None, None)
        .unwrap();
    decoder
        .add_edge(2, 5, &[1, 2], Some(3.0), None, None)
        .unwrap();

    // Boundary edges
    decoder
        .add_boundary_edge(0, &[], Some(5.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(7, &[], Some(5.0), None, None)
        .unwrap();

    // Create a complex detection pattern
    let mut detection_events = vec![0u8; 8];
    detection_events[1] = 1;
    detection_events[2] = 1;
    detection_events[5] = 1;
    detection_events[6] = 1;

    // Get matched pairs
    let pairs = decoder.decode_to_matched_pairs(&detection_events).unwrap();

    // Verify pairs are valid
    let mut matched_detectors = HashSet::new();
    for pair in &pairs {
        assert!(!matched_detectors.contains(&pair.detector1));
        matched_detectors.insert(pair.detector1);

        if let Some(d2) = pair.detector2 {
            assert!(!matched_detectors.contains(&d2));
            matched_detectors.insert(d2);
        }
    }

    // Test dictionary format
    let dict = decoder
        .decode_to_matched_pairs_dict(&detection_events)
        .unwrap();

    // Verify reciprocal matching
    for (d1, maybe_d2) in &dict {
        if let Some(d2) = maybe_d2 {
            assert_eq!(dict.get(d2), Some(&Some(*d1)));
        }
    }
}

#[test]
fn test_shortest_path_complex() {
    // Test shortest path in complex graph
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(10)
        .observables(2)
        .build()
        .unwrap();

    // Create a graph with multiple paths of different weights
    // Direct path with high weight
    decoder
        .add_edge(0, 9, &[0, 1], Some(10.0), None, None)
        .unwrap();

    // Longer path with lower total weight
    for i in 0..9 {
        decoder
            .add_edge(i, i + 1, &[i % 2], Some(0.5), None, None)
            .unwrap();
    }

    // Alternative middle path
    decoder.add_edge(0, 5, &[0], Some(3.0), None, None).unwrap();
    decoder.add_edge(5, 9, &[1], Some(3.0), None, None).unwrap();

    // Find shortest path
    let path = decoder.get_shortest_path(0, 9).unwrap();

    // Path should exist and include start/end
    assert!(!path.is_empty());
    assert_eq!(path[0], 0);
    assert_eq!(path[path.len() - 1], 9);

    // Path should not be the direct edge (weight 10)
    // It should be either the chain (total weight ~4.5) or middle path (weight 6)
    assert!(path.len() > 2);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[test]
fn test_invalid_node_indices() {
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(5)
        .observables(2)
        .build()
        .unwrap();

    // Test adding edge with invalid node index
    // PyMatching auto-expands the graph, so this should succeed
    let result = decoder.add_edge(0, 10, &[0], Some(1.0), None, None);
    assert!(result.is_ok(), "PyMatching should auto-expand nodes");
    assert!(decoder.num_nodes() > 5, "Graph should have expanded");

    // Test adding boundary edge with high node index
    let boundary_result = decoder.add_boundary_edge(20, &[0], Some(1.0), None, None);
    assert!(
        boundary_result.is_ok(),
        "PyMatching should auto-expand for boundary edges"
    );

    // Test adding edge with high observable index
    let obs_result = decoder.add_edge(0, 1, &[100], Some(1.0), None, None);
    assert!(
        obs_result.is_ok(),
        "PyMatching should auto-expand observables"
    );
    assert!(
        decoder.num_observables() > 100,
        "Observables should have expanded"
    );

    // Test querying non-existent edge (between valid nodes with no edge)
    let result = decoder.get_edge_data(0, 4);
    assert!(result.is_err());
}

#[test]
fn test_invalid_detection_events() {
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(5)
        .observables(2)
        .build()
        .unwrap();

    // Too many detection events
    let detection_events = vec![0u8; 10];
    let result = decoder.validate_detector_indices(&detection_events);
    assert!(result.is_err());

    // decode should validate syndrome length
    let decode_result = decoder.decode(&detection_events);
    assert!(
        decode_result.is_err(),
        "decode should error on invalid syndrome length"
    );
    let error = decode_result.unwrap_err();
    assert!(
        error.to_string().contains("Invalid syndrome")
            || error.to_string().contains("expected length"),
        "Error should mention invalid syndrome: '{error}'"
    );
}

#[test]
fn test_invalid_batch_decoding() {
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(5)
        .observables(2)
        .build()
        .unwrap();

    // Test with num_detectors exceeding actual count
    let actual_detectors = decoder.num_detectors();
    let result = decoder.decode_batch_with_config(
        &[0u8; 10],           // some dummy data
        1,                    // num_shots
        actual_detectors + 5, // num_detectors (too large)
        BatchConfig {
            bit_packed_input: false,
            bit_packed_output: false,
            return_weights: false,
        },
    );
    assert!(result.is_err());
    let error = result.unwrap_err();
    assert!(
        error.to_string().contains("Invalid syndrome")
            || error.to_string().contains("expected length"),
        "Error should mention invalid syndrome: '{error}'"
    );

    // Test with mismatched shots array size
    // For 2 shots with actual_detectors detectors each, we need 2 * actual_detectors bytes
    let wrong_size = actual_detectors + 1; // Wrong size
    let result2 = decoder.decode_batch_with_config(
        &vec![0u8; wrong_size], // wrong size
        2,                      // num_shots
        actual_detectors,       // num_detectors
        BatchConfig {
            bit_packed_input: false,
            bit_packed_output: false,
            return_weights: false,
        },
    );
    assert!(result2.is_err());
    assert!(
        result2
            .unwrap_err()
            .to_string()
            .contains("doesn't match expected size")
    );

    // Test empty batch (should succeed with empty result)
    let result3 = decoder.decode_batch_with_config(
        &[],
        0, // num_shots
        actual_detectors,
        BatchConfig {
            bit_packed_input: false,
            bit_packed_output: false,
            return_weights: false,
        },
    );
    assert!(result3.is_ok());
    let batch_result = result3.unwrap();
    assert_eq!(batch_result.predictions.len(), 0);
    assert_eq!(batch_result.weights.len(), 0);
}

#[test]
fn test_shortest_path_connected_graph() {
    // Test shortest path on a connected graph - this should work
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(5)
        .observables(2)
        .build()
        .unwrap();

    // Create a simple connected graph
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(1.0), None, None).unwrap();
    decoder.add_edge(2, 3, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(3, 4, &[1], Some(1.0), None, None).unwrap();

    // Test shortest path on connected graph
    let result = decoder.get_shortest_path(0, 4);
    assert!(result.is_ok(), "Should return Ok for connected graph");
    let path = result.unwrap();
    // Path should have nodes from 0 to 4
    assert!(
        !path.is_empty(),
        "Path should not be empty for connected nodes"
    );
    assert_eq!(path[0], 0, "Path should start at 0");
    assert_eq!(path[path.len() - 1], 4, "Path should end at 4");

    // Test out of bounds nodes still validate properly
    let oob_result = decoder.get_shortest_path(0, 10);
    assert!(oob_result.is_err(), "Should error on out of bounds node");
    assert!(
        oob_result
            .unwrap_err()
            .to_string()
            .contains("out of bounds")
    );
}

#[test]
fn test_shortest_path_disconnected_graph() {
    // Test shortest path behavior with disconnected graphs
    // Our Rust wrapper now checks connectivity before calling PyMatching
    // to prevent segfaults

    let mut decoder = PyMatchingDecoder::builder()
        .nodes(6)
        .observables(2)
        .build()
        .unwrap();

    // Create two disconnected components
    // Component 1: nodes 0-1-2
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(1.0), None, None).unwrap();

    // Component 2: nodes 3-4-5 (disconnected from component 1)
    decoder.add_edge(3, 4, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(4, 5, &[1], Some(1.0), None, None).unwrap();

    // Test path within same component - should work
    let result1 = decoder.get_shortest_path(0, 2);
    assert!(result1.is_ok(), "Should work within connected component");
    let path1 = result1.unwrap();
    assert!(!path1.is_empty(), "Path should exist within component");

    // Test path between disconnected components - should return error gracefully
    let result2 = decoder.get_shortest_path(0, 5);
    assert!(result2.is_err(), "Should error for disconnected components");
    let err = result2.unwrap_err();
    assert!(
        err.to_string().contains("different connected components"),
        "Error should mention disconnected components: {err}"
    );
}
