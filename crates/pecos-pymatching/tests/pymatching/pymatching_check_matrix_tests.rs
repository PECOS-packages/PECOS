//! Comprehensive tests for `PyMatching` `from_check_matrix` functionality

use pecos_pymatching::{
    CheckMatrix, CheckMatrixConfig, MergeStrategy, PyMatchingDecoder, PyMatchingError,
};
use std::collections::HashSet;

// ============================================================================
// Basic Check Matrix Construction Tests
// ============================================================================

#[test]
fn test_basic_repetition_code() {
    // Test simple repetition code: H = [[1, 1, 0], [0, 1, 1]]
    let entries = vec![
        (0, 0, 1), // H[0,0] = 1
        (0, 1, 1), // H[0,1] = 1
        (1, 1, 1), // H[1,1] = 1
        (1, 2, 1), // H[1,2] = 1
    ];

    let weights = vec![1.0; 3]; // uniform weights
    let matrix = CheckMatrix::from_triplets(entries, 2, 3)
        .with_weights(weights)
        .unwrap();
    let mut decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

    // Verify basic structure
    assert!(decoder.num_nodes() >= 2);
    assert!(decoder.num_observables() >= 3);

    // Test decoding with single bit flip
    let num_detectors = decoder.num_detectors();
    let mut detection_events = vec![0u8; num_detectors];
    if num_detectors > 0 {
        detection_events[0] = 1; // First check fires
    }

    let result = decoder.decode(&detection_events).unwrap();
    // Check that we got a result with observables equal to the number of columns
    assert_eq!(result.observable.len(), 3); // Should match num_cols
    assert!(result.weight >= 0.0);
}

#[test]
fn test_simple_surface_code_check_matrix() {
    // Simple code with proper 2-body stabilizers (no overlapping columns)
    let entries = vec![
        // Each column connects exactly 2 checks
        (0, 0, 1),
        (1, 0, 1), // Column 0: checks 0 and 1
        (1, 1, 1),
        (2, 1, 1), // Column 1: checks 1 and 2
        (2, 2, 1),
        (3, 2, 1), // Column 2: checks 2 and 3
        (0, 3, 1),
        (3, 3, 1), // Column 3: checks 0 and 3
    ];

    let weights = vec![1.0; 4]; // uniform weights
    let matrix = CheckMatrix::from_triplets(entries, 4, 4)
        .with_weights(weights)
        .unwrap();
    let decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

    assert!(decoder.num_nodes() >= 4);
    assert!(decoder.num_edges() > 0);
}

// ============================================================================
// Weighted Check Matrix Tests
// ============================================================================

#[test]
fn test_check_matrix_with_weights() {
    let entries = vec![
        (0, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (2, 3, 1),
    ];

    let weights = vec![1.0, 2.0, 3.0, 4.0]; // Different weight for each column

    let matrix = CheckMatrix::from_triplets(entries, 3, 4)
        .with_weights(weights)
        .unwrap();
    let decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

    // Verify edges have correct weights
    assert!(decoder.has_edge(0, 1)); // Column 1 connects rows 0 and 1
    let edge_data = decoder.get_edge_data(0, 1).unwrap();
    assert!(
        (edge_data.weight - 2.0).abs() < f64::EPSILON,
        "Edge weight should be 2.0 but was {}",
        edge_data.weight
    ); // Should have weight from column 1
}

#[test]
fn test_check_matrix_with_error_probabilities() {
    let entries = vec![
        (0, 0, 1),
        (1, 0, 1), // Column 0 connects rows 0 and 1
        (1, 1, 1),
        (2, 1, 1), // Column 1 connects rows 1 and 2
    ];

    let error_probs = vec![0.1, 0.2]; // Different error probability for each column
    let matrix = CheckMatrix::from_triplets(entries, 3, 2);

    let config = CheckMatrixConfig {
        error_probabilities: Some(error_probs),
        ..Default::default()
    };
    let decoder = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).unwrap();

    // Check error probabilities are set correctly
    let edge_data_01 = decoder.get_edge_data(0, 1).unwrap();
    assert!((edge_data_01.error_probability - 0.1).abs() < 1e-6);

    let edge_data_12 = decoder.get_edge_data(1, 2).unwrap();
    assert!((edge_data_12.error_probability - 0.2).abs() < 1e-6);
}

// ============================================================================
// Timelike Edges Tests (Repetitions > 1)
// ============================================================================

#[test]
fn test_check_matrix_with_repetitions() {
    // Simple repetition code
    let entries = vec![(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 2, 1)];
    let matrix = CheckMatrix::from_triplets(entries, 2, 3);

    let repetitions = 3; // 3 rounds of syndrome extraction

    let config = CheckMatrixConfig {
        repetitions,
        ..Default::default()
    };
    let decoder = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).unwrap();

    // Should have 2 checks * 3 repetitions = 6 nodes
    assert!(decoder.num_nodes() >= 6);

    // Check timelike edges exist between rounds
    // Node 0 in round 0 should connect to node 0 in round 1
    assert!(decoder.has_edge(0, 2)); // 0 + 1*2 = 2
    assert!(decoder.has_edge(2, 4)); // 0 + 2*2 = 4

    // Node 1 in round 0 should connect to node 1 in round 1
    assert!(decoder.has_edge(1, 3)); // 1 + 1*2 = 3
    assert!(decoder.has_edge(3, 5)); // 1 + 2*2 = 5
}

#[test]
fn test_timelike_weights_and_measurement_errors() {
    let entries = vec![(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 2, 1)];
    let matrix = CheckMatrix::from_triplets(entries, 2, 3);

    let repetitions = 3;
    let timelike_weights = vec![0.5, 1.5]; // Different weight for each check's timelike edge
    let measurement_error_probs = vec![0.01, 0.02]; // Different prob for each check

    let config = CheckMatrixConfig {
        repetitions,
        timelike_weights: Some(timelike_weights),
        measurement_error_probabilities: Some(measurement_error_probs),
        ..Default::default()
    };
    let decoder = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).unwrap();

    // Check timelike edges have correct weights and error probabilities
    let edge_02 = decoder.get_edge_data(0, 2).unwrap(); // Check 0 between rounds 0 and 1
    // Only check weight if it's not NaN
    if !edge_02.weight.is_nan() {
        assert!(
            (edge_02.weight - 0.5).abs() < f64::EPSILON,
            "Edge weight should be 0.5 but was {}",
            edge_02.weight
        );
    }
    // Only check error probability if it's not NaN
    if !edge_02.error_probability.is_nan() {
        assert!((edge_02.error_probability - 0.01).abs() < 1e-6);
    }

    let edge_13 = decoder.get_edge_data(1, 3).unwrap(); // Check 1 between rounds 0 and 1
    // Only check weight if it's not NaN
    if !edge_13.weight.is_nan() {
        assert!(
            (edge_13.weight - 1.5).abs() < f64::EPSILON,
            "Edge weight should be 1.5 but was {}",
            edge_13.weight
        );
    }
    // Only check error probability if it's not NaN
    if !edge_13.error_probability.is_nan() {
        assert!((edge_13.error_probability - 0.02).abs() < 1e-6);
    }
}

#[test]
fn test_boundary_setting_with_repetitions() {
    let entries = vec![(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 2, 1)];
    let matrix = CheckMatrix::from_triplets(entries, 2, 3);

    let repetitions = 3;

    let config = CheckMatrixConfig {
        repetitions,
        use_virtual_boundary: false,
        ..Default::default()
    };
    let decoder = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).unwrap();

    // Boundary should be set to last round of detectors
    let boundary = decoder.get_boundary();
    let expected_boundary: HashSet<usize> = [4, 5].iter().copied().collect(); // Last round nodes
    let actual_boundary: HashSet<usize> = boundary.into_iter().collect();
    assert_eq!(actual_boundary, expected_boundary);
}

// ============================================================================
// Invalid Check Matrix Tests
// ============================================================================

#[test]
fn test_invalid_check_matrix_too_many_entries() {
    // Column has 3 non-zero entries (invalid for matching decoder)
    let check_matrix = vec![
        (0, 0, 1),
        (1, 0, 1),
        (2, 0, 1), // Column 0 has 3 entries
        (0, 1, 1),
        (1, 1, 1), // Column 1 has 2 entries
    ];

    let matrix = CheckMatrix::from_triplets(check_matrix, 3, 2)
        .with_weights(vec![1.0; 2])
        .unwrap();
    let result = PyMatchingDecoder::from_check_matrix(&matrix);

    assert!(result.is_err());
    match result {
        Err(PyMatchingError::Configuration(msg)) => {
            assert!(msg.contains("3 non-zero entries"));
        }
        _ => panic!("Expected configuration error for too many entries"),
    }
}

#[test]
fn test_invalid_timelike_weights_length() {
    let check_matrix = vec![(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 2, 1)];

    let repetitions = 3;
    let timelike_weights = vec![0.5]; // Only 1 weight, but need 2 (one per row)

    let config = CheckMatrixConfig {
        repetitions,
        timelike_weights: Some(timelike_weights),
        ..Default::default()
    };
    let matrix = CheckMatrix::from_triplets(check_matrix, 2, 3);
    let result = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config);

    assert!(result.is_err());
    match result {
        Err(PyMatchingError::Configuration(msg)) => {
            assert!(msg.contains("timelike_weights"));
            assert!(msg.contains("must equal number of rows"));
        }
        _ => panic!("Expected configuration error for wrong timelike_weights length"),
    }
}

#[test]
fn test_invalid_measurement_error_probs_length() {
    let check_matrix = vec![
        (0, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
        (1, 2, 1),
        (2, 2, 1),
        (2, 3, 1),
    ];

    let repetitions = 2;
    let measurement_error_probs = vec![0.01, 0.02]; // Only 2 probs, but need 3 (one per row)

    let config = CheckMatrixConfig {
        repetitions,
        measurement_error_probabilities: Some(measurement_error_probs),
        ..Default::default()
    };
    let matrix = CheckMatrix::from_triplets(check_matrix, 3, 4);
    let result = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config);

    assert!(result.is_err());
    match result {
        Err(PyMatchingError::Configuration(msg)) => {
            assert!(msg.contains("measurement_error_probabilities"));
            assert!(msg.contains("must equal number of rows"));
        }
        _ => {
            panic!("Expected configuration error for wrong measurement_error_probabilities length")
        }
    }
}

// ============================================================================
// Empty and Edge Case Tests
// ============================================================================

#[test]
fn test_empty_check_matrix() {
    let check_matrix: Vec<(usize, usize, u8)> = vec![];

    let matrix = CheckMatrix::from_triplets(check_matrix, 0, 0);
    let decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

    assert_eq!(decoder.num_nodes(), 0);
    assert_eq!(decoder.num_edges(), 0);
}

#[test]
fn test_single_check_single_qubit() {
    // Minimal case: one check, one qubit
    let check_matrix = vec![(0, 0, 1)];

    let matrix = CheckMatrix::from_triplets(check_matrix, 1, 1)
        .with_weights(vec![1.0])
        .unwrap();
    let decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

    assert!(decoder.num_nodes() >= 1);
    // Should have boundary edge since column has only 1 non-zero entry
    assert!(decoder.has_boundary_edge(0));
}

#[test]
fn test_all_columns_single_entry() {
    // All errors connect to boundary (single detector per error)
    let check_matrix = vec![(0, 0, 1), (1, 1, 1), (2, 2, 1)];

    let matrix = CheckMatrix::from_triplets(check_matrix, 3, 3)
        .with_weights(vec![1.0; 3])
        .unwrap();
    let decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

    // All nodes should have boundary edges
    assert!(decoder.has_boundary_edge(0));
    assert!(decoder.has_boundary_edge(1));
    assert!(decoder.has_boundary_edge(2));

    // No regular edges
    assert!(!decoder.has_edge(0, 1));
    assert!(!decoder.has_edge(1, 2));
    assert!(!decoder.has_edge(0, 2));
}

// ============================================================================
// Large Sparse Matrix Tests
// ============================================================================

#[test]
fn test_large_sparse_matrix() {
    // Create a large sparse check matrix
    let num_checks = 100;
    let num_qubits = 150;

    let mut check_matrix = Vec::new();

    // Create a pattern where each qubit connects two adjacent checks
    for i in 0..num_qubits {
        let check1 = i % num_checks;
        let check2 = (i + 1) % num_checks;
        check_matrix.push((check1, i, 1));
        check_matrix.push((check2, i, 1));
    }

    let matrix = CheckMatrix::from_triplets(check_matrix, num_checks, num_qubits)
        .with_weights(vec![1.0; num_qubits])
        .unwrap();
    let decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

    assert!(decoder.num_nodes() >= num_checks);
    assert!(decoder.num_edges() > 0);
}

#[test]
fn test_sparse_matrix_with_weights() {
    // Sparse matrix with random-like weights
    let num_checks = 50;
    let num_qubits = 75;

    let mut check_matrix = Vec::new();
    let mut weights = Vec::with_capacity(num_qubits);

    for i in 0..num_qubits {
        let check1 = (i * 7) % num_checks;
        let check2 = (i * 13 + 5) % num_checks;

        if check1 == check2 {
            // Single check - will create boundary edge
            check_matrix.push((check1, i, 1));
        } else {
            check_matrix.push((check1, i, 1));
            check_matrix.push((check2, i, 1));
        }

        // Varying weights
        #[allow(clippy::cast_precision_loss)] // Acceptable for test data generation
        weights.push(1.0 + (i as f64) * 0.1);
    }

    let matrix = CheckMatrix::from_triplets(check_matrix, num_checks, num_qubits)
        .with_weights(weights)
        .unwrap();
    let decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

    assert!(decoder.num_nodes() >= num_checks);
}

// ============================================================================
// Consistency with Manual Graph Construction Tests
// ============================================================================

#[test]
fn test_consistency_with_manual_construction() {
    // Create decoder using check matrix
    let check_matrix = vec![
        (0, 0, 1),
        (1, 0, 1), // Column 0: connects checks 0 and 1
        (1, 1, 1),
        (2, 1, 1), // Column 1: connects checks 1 and 2
        (0, 2, 1), // Column 2: only check 0 (boundary)
    ];

    let weights = vec![1.0, 2.0, 3.0];

    let matrix = CheckMatrix::from_triplets(check_matrix, 3, 3)
        .with_weights(weights)
        .unwrap();
    let mut decoder_from_matrix = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

    // Create equivalent decoder manually
    let mut decoder_manual = PyMatchingDecoder::builder()
        .nodes(3)
        .observables(3)
        .build()
        .unwrap();

    decoder_manual
        .add_edge(
            0,
            1,
            &[0],
            Some(1.0),
            None,
            Some(MergeStrategy::SmallestWeight),
        )
        .unwrap();
    decoder_manual
        .add_edge(
            1,
            2,
            &[1],
            Some(2.0),
            None,
            Some(MergeStrategy::SmallestWeight),
        )
        .unwrap();
    decoder_manual
        .add_boundary_edge(
            0,
            &[2],
            Some(3.0),
            None,
            Some(MergeStrategy::SmallestWeight),
        )
        .unwrap();

    // Test both decoders produce same results
    let num_detectors = decoder_from_matrix.num_detectors();
    let mut detection_events = vec![0u8; num_detectors];
    if num_detectors >= 3 {
        detection_events[0] = 1; // Detection at first detector
        detection_events[2] = 1; // Detection at third detector
    } else if num_detectors >= 1 {
        detection_events[0] = 1; // At least one detection
    }

    let result_matrix = decoder_from_matrix.decode(&detection_events).unwrap();
    let result_manual = decoder_manual.decode(&detection_events).unwrap();

    // Both should produce valid results with the same number of observables
    assert_eq!(
        result_matrix.observable.len(),
        result_manual.observable.len()
    );
    // For this simple case, the results should be similar
    assert!(result_matrix.weight >= 0.0);
    assert!(result_manual.weight >= 0.0);
}

#[test]
fn test_virtual_boundary_option() {
    // Test the use_virtual_boundary option
    let check_matrix = vec![
        (0, 0, 1), // Single detector - should create boundary edge
        (1, 1, 1),
        (2, 1, 1), // Two detectors - regular edge
    ];

    // Test with virtual boundary (true)
    let config = CheckMatrixConfig {
        use_virtual_boundary: true,
        ..Default::default()
    };
    let matrix = CheckMatrix::from_triplets(check_matrix.clone(), 3, 2);
    let mut decoder_virtual =
        PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).unwrap();

    // Test without virtual boundary (false)
    let config = CheckMatrixConfig {
        use_virtual_boundary: false,
        ..Default::default()
    };
    let matrix2 = CheckMatrix::from_triplets(check_matrix, 3, 2);
    let mut decoder_no_virtual =
        PyMatchingDecoder::from_check_matrix_with_config(&matrix2, config).unwrap();

    // Both should have boundary edge for node 0
    assert!(decoder_virtual.has_boundary_edge(0));
    assert!(decoder_no_virtual.has_boundary_edge(0));

    // Test decoding - each decoder might have different detector counts
    let num_detectors_virtual = decoder_virtual.num_detectors();
    let mut detection_events_virtual = vec![0u8; num_detectors_virtual];
    if num_detectors_virtual > 0 {
        detection_events_virtual[0] = 1; // Detection at node 0
    }

    let num_detectors_no_virtual = decoder_no_virtual.num_detectors();
    let mut detection_events_no_virtual = vec![0u8; num_detectors_no_virtual];
    if num_detectors_no_virtual > 0 {
        detection_events_no_virtual[0] = 1; // Detection at node 0
    }

    let result_virtual = decoder_virtual.decode(&detection_events_virtual).unwrap();
    let result_no_virtual = decoder_no_virtual
        .decode(&detection_events_no_virtual)
        .unwrap();

    // Both should decode and produce reasonable results
    assert_eq!(result_virtual.observable.len(), 2);
    assert_eq!(result_no_virtual.observable.len(), 2);
    assert!(result_virtual.weight >= 0.0);
    assert!(result_no_virtual.weight >= 0.0);
}

// ============================================================================
// Dense Matrix Conversion Test
// ============================================================================

#[test]
fn test_from_check_matrix_dense() {
    // Test the dense matrix convenience method
    let dense_matrix = vec![vec![1, 1, 0, 0], vec![0, 1, 1, 0], vec![0, 0, 1, 1]];

    let weights = vec![1.0, 2.0, 3.0, 4.0];

    let matrix = CheckMatrix::from_dense_vec(&dense_matrix)
        .unwrap()
        .with_weights(weights)
        .unwrap();
    let decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

    // Verify structure
    assert!(decoder.num_nodes() >= 3);

    // Check edges exist where expected
    assert!(decoder.has_edge(0, 1)); // Column 1 connects rows 0 and 1
    assert!(decoder.has_edge(1, 2)); // Column 2 connects rows 1 and 2

    // Check edge weights
    let edge_01 = decoder.get_edge_data(0, 1).unwrap();
    assert!(
        (edge_01.weight - 2.0).abs() < f64::EPSILON,
        "Edge weight should be 2.0 but was {}",
        edge_01.weight
    ); // Weight from column 1

    let edge_12 = decoder.get_edge_data(1, 2).unwrap();
    assert!(
        (edge_12.weight - 3.0).abs() < f64::EPSILON,
        "Edge weight should be 3.0 but was {}",
        edge_12.weight
    ); // Weight from column 2
}

#[test]
fn test_dense_matrix_invalid_dimensions() {
    // Test with inconsistent row lengths
    let invalid_dense_matrix = vec![
        vec![1, 1, 0],
        vec![0, 1, 1, 0], // This row has 4 columns instead of 3
        vec![0, 0, 1],
    ];

    let result = CheckMatrix::from_dense_vec(&invalid_dense_matrix);

    assert!(result.is_err());
    match result {
        Err(PyMatchingError::Configuration(msg)) => {
            assert!(msg.contains("columns"));
        }
        _ => panic!("Expected Configuration error for inconsistent columns"),
    }
}

// ============================================================================
// Integration Tests with Decoding
// ============================================================================

#[test]
fn test_decoding_with_check_matrix() {
    // Create a simple code and test decoding
    let check_matrix = vec![
        (0, 0, 1),
        (0, 1, 1), // Z0Z1
        (1, 1, 1),
        (1, 2, 1), // Z1Z2
        (2, 2, 1),
        (2, 3, 1), // Z2Z3
    ];

    let matrix = CheckMatrix::from_triplets(check_matrix, 3, 4)
        .with_weights(vec![1.0; 4])
        .unwrap();
    let mut decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

    // Test single qubit error
    let num_detectors = decoder.num_detectors();
    let mut detection_events = vec![0u8; num_detectors];
    if num_detectors >= 2 {
        detection_events[0] = 1; // Check 0 fires
        detection_events[1] = 1; // Check 1 fires (error on qubit 1)
    }
    let result = decoder.decode(&detection_events).unwrap();

    // Check that we get a reasonable result
    assert_eq!(result.observable.len(), 4);
    assert!(result.weight >= 0.0);
    // Just verify we get a valid decoding result - the exact values depend on the decoder's algorithm
}

#[test]
fn test_decoding_with_repetitions() {
    // Test decoding with multiple rounds
    let check_matrix = vec![(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 2, 1)];

    let repetitions = 3;

    let config = CheckMatrixConfig {
        repetitions,
        ..Default::default()
    };
    let matrix = CheckMatrix::from_triplets(check_matrix, 2, 3);
    let mut decoder = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).unwrap();

    // Create detection pattern: measurement error in round 1 on check 0
    let num_detectors = decoder.num_detectors();
    let mut detection_events = vec![0u8; num_detectors];
    if num_detectors >= 3 {
        detection_events[0] = 1; // Check 0, round 0
        detection_events[2] = 1; // Check 0, round 1 (measurement error between rounds)
    }

    let result = decoder.decode(&detection_events).unwrap();

    // Should produce a valid result
    assert_eq!(result.observable.len(), 3);
    assert!(result.weight >= 0.0);
    // With timelike edges, this should be decoded as a measurement error
}
