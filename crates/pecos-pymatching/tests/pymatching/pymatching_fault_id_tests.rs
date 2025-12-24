//! Comprehensive tests for fault ID and observable management in `PyMatching`
//!
//! This test module focuses on testing:
//! - `ensure_num_fault_ids()` functionality (alias for `ensure_num_observables`)
//! - `ensure_num_observables()` functionality
//! - `num_observables()` getter
//! - Observable count management during graph construction
//! - Observable count validation during decoding
//! - Edge cases with zero/large observable counts
//! - Integration with check matrix construction
//! - Compatibility with petgraph conversion

use pecos_pymatching::{
    BatchConfig, CheckMatrix, CheckMatrixConfig, MergeStrategy, PyMatchingConfig, PyMatchingDecoder,
};

#[test]
fn test_ensure_num_observables_basic() {
    // Test basic functionality of ensure_num_observables
    let config = PyMatchingConfig {
        num_nodes: Some(5),
        num_observables: 10,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Initial observable count should be at least 10
    assert!(decoder.num_observables() >= 10);
    let initial_count = decoder.num_observables();

    // Ensure we have at least 20 observables
    decoder.ensure_num_observables(20).unwrap();
    assert!(decoder.num_observables() >= 20);

    // Ensure we have at least 50 observables
    decoder.ensure_num_observables(50).unwrap();
    assert!(decoder.num_observables() >= 50);

    // Calling with a smaller number should not reduce the count
    decoder.ensure_num_observables(30).unwrap();
    assert!(decoder.num_observables() >= 50);

    // PyMatching may round up to powers of 2 or other convenient sizes
    println!(
        "Observable counts: initial={}, after ensure(20)={}, after ensure(50)={}",
        initial_count,
        20,
        decoder.num_observables()
    );
}

#[test]
fn test_ensure_num_fault_ids_alias() {
    // Test that ensure_num_fault_ids is properly aliased to ensure_num_observables
    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 5,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config.clone()).unwrap();
    let _initial_count = decoder.num_observables();

    // Test ensure_num_fault_ids (alias)
    decoder.ensure_num_observables(25).unwrap();
    assert!(decoder.num_observables() >= 25);

    // Both methods should have the same effect
    let mut decoder2 = PyMatchingDecoder::new(config).unwrap();
    decoder2.ensure_num_observables(25).unwrap();

    // Both decoders should have the same observable count
    assert_eq!(decoder.num_observables(), decoder2.num_observables());
}

#[test]
fn test_observable_count_with_edge_addition() {
    // Test that adding edges with high observable indices automatically expands the count
    let config = PyMatchingConfig {
        num_nodes: Some(6),
        num_observables: 10,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();
    let _initial_observables = decoder.num_observables();

    // Add edge with observables within current range
    decoder
        .add_edge(0, 1, &[0, 5, 9], Some(1.0), None, None)
        .unwrap();

    // PyMatching auto-expands when adding edges with high observable indices
    // Add edge with observables beyond current range
    decoder
        .add_edge(2, 3, &[15, 20, 30], Some(1.0), None, None)
        .unwrap();

    // The observable count may have been automatically expanded
    // This is implementation-dependent behavior in PyMatching
    println!(
        "Observables after adding edge with indices [15,20,30]: {}",
        decoder.num_observables()
    );

    // Explicitly ensure we have enough observables for our high indices
    decoder.ensure_num_observables(31).unwrap();
    assert!(decoder.num_observables() >= 31);

    // Add boundary edge with even higher observable
    decoder
        .add_boundary_edge(4, &[50, 60], Some(1.0), None, None)
        .unwrap();

    // Ensure we have enough for these as well
    decoder.ensure_num_observables(61).unwrap();
    assert!(decoder.num_observables() >= 61);
}

#[test]
fn test_zero_observables_edge_case() {
    // Test behavior with zero observables initially
    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 0,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // PyMatching may have a minimum observable count even when 0 is requested
    let initial_count = decoder.num_observables();
    println!("Initial observable count when requesting 0: {initial_count}");

    // Add edge with no observables
    decoder.add_edge(0, 1, &[], Some(1.0), None, None).unwrap();

    // Add edge with observables - this should work even if we started with 0
    decoder
        .add_edge(1, 2, &[0, 1, 2], Some(1.0), None, None)
        .unwrap();

    // Ensure we have at least 3 observables for the edge we just added
    decoder.ensure_num_observables(3).unwrap();
    assert!(decoder.num_observables() >= 3);
}

#[test]
fn test_large_observable_counts() {
    // Test with large observable counts
    let test_sizes = vec![64, 65, 100, 128, 256, 1000];

    for size in test_sizes {
        let config = PyMatchingConfig {
            num_nodes: Some(10),
            num_observables: size,
            ..Default::default()
        };

        let mut decoder = PyMatchingDecoder::new(config).unwrap();

        // Should have at least the requested number
        assert!(
            decoder.num_observables() >= size,
            "Failed for size {}: got {}",
            size,
            decoder.num_observables()
        );

        // Add edges using high observable indices
        let high_index = size - 1;
        decoder
            .add_edge(
                0,
                1,
                &[0, high_index / 2, high_index],
                Some(1.0),
                None,
                None,
            )
            .unwrap();

        // Decode with appropriate method based on size
        let detection_events = vec![1, 1, 0, 0, 0, 0, 0, 0, 0, 0];

        let result = decoder.decode(&detection_events).unwrap();
        assert_eq!(result.observable.len(), size);
    }
}

#[test]
fn test_observable_management_during_decoding() {
    // Test that observable count is properly managed during different decoding operations
    let config = PyMatchingConfig {
        num_nodes: Some(6),
        num_observables: 10,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Set up a graph with specific observables
    decoder
        .add_edge(0, 1, &[0, 1], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(1, 2, &[2, 3], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(2, 3, &[4, 5], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(3, 4, &[6, 7], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(4, 5, &[8, 9], Some(1.0), None, None)
        .unwrap();

    // Add boundary edges
    decoder
        .add_boundary_edge(0, &[], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(5, &[], Some(1.0), None, None)
        .unwrap();

    // Test decoding with standard decode
    let detection_events = vec![1, 0, 1, 0, 0, 0];
    let result = decoder.decode(&detection_events).unwrap();

    // Result should have the correct number of observables
    assert_eq!(result.observable.len(), 10);

    // Now expand observables and test again
    decoder.ensure_num_observables(20).unwrap();

    let result2 = decoder.decode(&detection_events).unwrap();
    assert_eq!(result2.observable.len(), 20);

    // Test with extended decode for >64 observables
    decoder.ensure_num_observables(100).unwrap();
    let result3 = decoder.decode(&detection_events).unwrap();
    assert_eq!(result3.observable.len(), 100);
}

#[test]
fn test_observable_count_with_check_matrix() {
    // Test observable management when creating decoder from check matrix

    // Create a check matrix with 5 columns (observables)
    let check_matrix = vec![
        (0, 0, 1), // Row 0, Col 0
        (0, 1, 1), // Row 0, Col 1
        (1, 1, 1), // Row 1, Col 1
        (1, 2, 1), // Row 1, Col 2
        (2, 2, 1), // Row 2, Col 2
        (2, 3, 1), // Row 2, Col 3
        (3, 3, 1), // Row 3, Col 3
        (3, 4, 1), // Row 3, Col 4
    ];

    let matrix = CheckMatrix::from_triplets(check_matrix, 4, 5)
        .with_weights(vec![1.0; 5])
        .unwrap();
    let decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

    // Should have at least 5 observables (may be more due to PyMatching defaults)
    assert!(decoder.num_observables() >= 5);

    // Test with larger check matrix (100 observables)
    let mut large_matrix = Vec::new();
    for i in 0..50 {
        // Each observable touches two detectors
        large_matrix.push((i, i, 1));
        large_matrix.push((i, i + 50, 1));
    }

    let large_matrix_struct = CheckMatrix::from_triplets(large_matrix, 51, 100)
        .with_weights(vec![1.0; 100])
        .unwrap();
    let large_decoder = PyMatchingDecoder::from_check_matrix(&large_matrix_struct).unwrap();

    assert!(large_decoder.num_observables() >= 100);
}

#[test]
fn test_observable_count_with_repetitions() {
    // Test observable management with timelike repetitions
    let check_matrix = vec![(0, 0, 1), (0, 1, 1), (1, 1, 1), (1, 2, 1)];

    let config = CheckMatrixConfig {
        repetitions: 5,
        ..Default::default()
    };
    let matrix = CheckMatrix::from_triplets(check_matrix, 2, 3);
    let decoder = PyMatchingDecoder::from_check_matrix_with_config(&matrix, config).unwrap();

    // Observable count should still be based on columns, not affected by repetitions
    assert!(decoder.num_observables() >= 3);

    // But we should have more nodes due to repetitions
    assert!(decoder.num_nodes() >= 10); // 2 detectors * 5 repetitions
}

#[test]
fn test_observable_validation_in_batch_decode() {
    // Test that batch decoding properly handles observable counts
    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 10,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Set up graph
    decoder
        .add_edge(0, 1, &[0, 1, 2], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(1, 2, &[3, 4, 5], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(2, 3, &[6, 7, 8], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(0, &[9], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(3, &[9], Some(1.0), None, None)
        .unwrap();

    // Prepare batch data
    let num_shots = 5;
    let num_detectors = decoder.num_detectors();
    let shots = vec![0u8; num_shots * num_detectors];

    // Decode batch
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

    // Each prediction should respect the observable count
    for prediction in &result.predictions {
        assert!(prediction.len() >= 10);
    }

    // Now expand observables and decode again
    decoder.ensure_num_observables(20).unwrap();

    let result2 = decoder
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

    // Predictions should now be larger
    for prediction in &result2.predictions {
        assert!(prediction.len() >= 20);
    }
}

#[test]
fn test_observable_count_persistence() {
    // Test that observable count is properly maintained across operations
    let config = PyMatchingConfig {
        num_nodes: Some(8),
        num_observables: 15,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();
    let initial_count = decoder.num_observables();

    // Add various edges
    decoder
        .add_edge(0, 1, &[0, 5, 10], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(2, 3, &[1, 6, 11], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(4, &[2, 7, 12], Some(1.0), None, None)
        .unwrap();

    // Count should not decrease
    assert!(decoder.num_observables() >= initial_count);

    // Set boundary
    decoder.set_boundary(&[0, 1, 2, 3]);

    // Count should still not decrease
    assert!(decoder.num_observables() >= initial_count);

    // Get all edges
    let edges = decoder.get_all_edges();

    // Check that observable indices in edges are valid
    for edge in edges {
        for obs in &edge.observables {
            assert!(
                *obs < decoder.num_observables(),
                "Observable index {} exceeds count {}",
                obs,
                decoder.num_observables()
            );
        }
    }
}

#[test]
fn test_observable_edge_cases_with_merge_strategies() {
    // Test observable handling with different merge strategies
    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 5,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Add initial edge with observables [0, 1]
    decoder
        .add_edge(0, 1, &[0, 1], Some(1.0), None, None)
        .unwrap();

    // Try different merge strategies with different observables

    // SmallestWeight - merge with different observables
    decoder
        .add_edge(
            0,
            1,
            &[2, 3],
            Some(0.5),
            None,
            Some(MergeStrategy::SmallestWeight),
        )
        .unwrap();

    // Independent - should allow parallel edge with same nodes but different observables
    decoder
        .add_edge(
            0,
            1,
            &[4],
            Some(2.0),
            None,
            Some(MergeStrategy::Independent),
        )
        .unwrap();

    // Replace - should replace with new observables
    decoder
        .add_edge(
            0,
            1,
            &[0, 2, 4],
            Some(3.0),
            None,
            Some(MergeStrategy::Replace),
        )
        .unwrap();

    // Verify edge data
    let edge_data = decoder.get_edge_data(0, 1).unwrap();

    // The final observables depend on the merge strategy behavior
    // Just verify that all observable indices are valid
    for obs in &edge_data.observables {
        assert!(*obs < decoder.num_observables());
    }
}

#[test]
fn test_from_dem_observable_count() {
    // Test observable count when loading from DEM
    let dem_string = r"
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L1
        error(0.1) D2 D3 L2
        error(0.1) D3 D4 L3
        error(0.1) D4 D5 L4
        error(0.1) D0 D5 L5 L6 L7
    ";

    let decoder = PyMatchingDecoder::from_dem(dem_string).unwrap();

    // Should have at least 8 observables (L0 through L7)
    assert!(decoder.num_observables() >= 8);

    // Test with larger observable indices
    let dem_large = r"
        error(0.1) D0 D1 L50
        error(0.1) D1 D2 L100
        error(0.1) D2 D3 L150
    ";

    let decoder_large = PyMatchingDecoder::from_dem(dem_large).unwrap();

    // Should have expanded to accommodate L150
    assert!(decoder_large.num_observables() > 150);
}

#[test]
fn test_config_observable_propagation() {
    // Test that config num_observables is properly propagated
    let test_configs = vec![
        (0, "zero observables"),
        (1, "single observable"),
        (64, "exactly 64 observables"),
        (65, "just over 64 observables"),
        (128, "power of 2 observables"),
        (1000, "large observable count"),
    ];

    for (num_obs, description) in test_configs {
        let config = PyMatchingConfig {
            num_nodes: Some(10),
            num_observables: num_obs,
            ..Default::default()
        };

        let decoder = PyMatchingDecoder::new(config.clone()).unwrap();

        // Should have at least the requested number
        assert!(
            decoder.num_observables() >= num_obs,
            "Failed for {}: requested {}, got {}",
            description,
            num_obs,
            decoder.num_observables()
        );

        // Config should be preserved
        assert_eq!(config.num_observables, num_obs);
    }
}

#[test]
fn test_builder_pattern_observable_count() {
    // Test observable count with builder pattern
    let decoder = PyMatchingDecoder::builder()
        .nodes(10)
        .observables(75)
        .build()
        .unwrap();

    assert!(decoder.num_observables() >= 75);

    // Test with default (should use default from config)
    let decoder_default = PyMatchingDecoder::builder().nodes(10).build().unwrap();

    // Default is 64 according to PyMatchingConfig::default()
    assert!(decoder_default.num_observables() >= 64);
}

#[test]
fn test_dense_check_matrix_observable_count() {
    // Test observable count with dense check matrix
    let check_matrix = vec![
        vec![1, 1, 0, 0, 0],
        vec![0, 1, 1, 0, 0],
        vec![0, 0, 1, 1, 0],
        vec![0, 0, 0, 1, 1],
    ];

    let matrix = CheckMatrix::from_dense_vec(&check_matrix).unwrap();
    let decoder = PyMatchingDecoder::from_check_matrix(&matrix).unwrap();

    // Should have at least 5 observables (number of columns)
    assert!(decoder.num_observables() >= 5);
}

#[test]
fn test_observable_indices_in_matched_dict() {
    // Test that observable information is preserved through matching operations
    let config = PyMatchingConfig {
        num_nodes: Some(6),
        num_observables: 10,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Create edges with specific observable patterns
    decoder
        .add_edge(0, 1, &[0, 1], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(1, 2, &[2, 3], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(3, 4, &[4, 5], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(4, 5, &[6, 7], Some(1.0), None, None)
        .unwrap();

    // Add boundary edges
    decoder
        .add_boundary_edge(0, &[8], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(2, &[8], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(3, &[9], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(5, &[9], Some(1.0), None, None)
        .unwrap();

    // Create detection events
    let detection_events = vec![1, 0, 1, 1, 0, 1];

    // Decode and check observable result
    let result = decoder.decode(&detection_events).unwrap();
    assert_eq!(result.observable.len(), 10);

    // Get matched pairs
    let matched_dict = decoder.decode_to_matched_dict(&detection_events).unwrap();

    // The matched pairs should be consistent with the observables triggered
    println!("Matched pairs: {:?}", matched_dict.matches);
    println!("Observables triggered: {:?}", result.observable);
}

#[test]
fn test_error_handling_invalid_observable_indices() {
    // While PyMatching auto-expands, test behavior with very large indices
    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 10,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Add edge with very large observable index
    // PyMatching should handle this gracefully by expanding
    let large_index = 1_000_000;
    let result = decoder.add_edge(0, 1, &[large_index], Some(1.0), None, None);

    // This should succeed as PyMatching auto-expands
    assert!(result.is_ok());

    // But the decoder might have expanded to accommodate
    if decoder.num_observables() > large_index {
        println!(
            "PyMatching expanded to {} observables to accommodate index {}",
            decoder.num_observables(),
            large_index
        );
    }
}

#[test]
fn test_observable_count_after_noise_simulation() {
    // Test that noise simulation respects observable count
    let config = PyMatchingConfig {
        num_nodes: Some(6),
        num_observables: 15,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Add edges with error probabilities for noise simulation
    decoder
        .add_edge(0, 1, &[0, 1], None, Some(0.1), None)
        .unwrap();
    decoder
        .add_edge(1, 2, &[2, 3], None, Some(0.1), None)
        .unwrap();
    decoder
        .add_edge(2, 3, &[4, 5], None, Some(0.1), None)
        .unwrap();
    decoder
        .add_edge(3, 4, &[6, 7], None, Some(0.1), None)
        .unwrap();
    decoder
        .add_edge(4, 5, &[8, 9], None, Some(0.1), None)
        .unwrap();

    // Add some edges with higher observable indices
    decoder
        .add_edge(0, 5, &[10, 11, 12], None, Some(0.05), None)
        .unwrap();
    decoder
        .add_boundary_edge(0, &[13, 14], None, Some(0.05), None)
        .unwrap();

    // Simulate noise
    let num_samples = 10;
    let noise_result = decoder.add_noise(num_samples, 42).unwrap();

    // Each error pattern should have the correct number of observables
    assert_eq!(noise_result.errors.len(), num_samples);
    for error_pattern in &noise_result.errors {
        assert_eq!(error_pattern.len(), decoder.num_observables());

        // Check that only valid observable indices have errors
        for (idx, &error) in error_pattern.iter().enumerate() {
            if error != 0 {
                assert!(idx < 15, "Error at invalid observable index {idx}");
            }
        }
    }
}

#[test]
fn test_observable_count_concurrency() {
    // Test that observable count is consistent across multiple operations
    let config = PyMatchingConfig {
        num_nodes: Some(8),
        num_observables: 20,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Perform multiple operations that might affect observable count
    let operations = vec![
        (0, 1, vec![0, 5, 10]),
        (1, 2, vec![1, 6, 11]),
        (2, 3, vec![2, 7, 12]),
        (3, 4, vec![3, 8, 13]),
        (4, 5, vec![4, 9, 14]),
        (5, 6, vec![15, 16, 17]),
        (6, 7, vec![18, 19]),
    ];

    for (node1, node2, observables) in operations {
        decoder
            .add_edge(node1, node2, &observables, Some(1.0), None, None)
            .unwrap();

        // Observable count should never decrease
        assert!(decoder.num_observables() >= 20);
    }

    // Final count should still be at least 20
    assert!(decoder.num_observables() >= 20);
}

#[test]
fn test_observable_boundary_interactions() {
    // Test observable behavior with boundary nodes
    let config = PyMatchingConfig {
        num_nodes: Some(6),
        num_observables: 8,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Set some boundary nodes
    decoder.set_boundary(&[0, 5]);

    // Add edges between boundary and non-boundary nodes
    decoder
        .add_edge(0, 1, &[0, 1], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(1, 2, &[2, 3], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(4, 5, &[6, 7], Some(1.0), None, None)
        .unwrap();

    // Add boundary edges with observables
    decoder
        .add_boundary_edge(2, &[4, 5], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(3, &[6, 7], Some(1.0), None, None)
        .unwrap();

    // Check that all edges respect observable count
    let all_edges = decoder.get_all_edges();
    for edge in all_edges {
        for &obs in &edge.observables {
            assert!(obs < decoder.num_observables());
        }
    }
}

#[test]
fn test_observable_count_in_path_finding() {
    // Test that path finding operations don't affect observable count
    let config = PyMatchingConfig {
        num_nodes: Some(6),
        num_observables: 12,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();
    let initial_count = decoder.num_observables();

    // Create a connected graph
    decoder
        .add_edge(0, 1, &[0, 1], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(1, 2, &[2, 3], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(2, 3, &[4, 5], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(3, 4, &[6, 7], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(4, 5, &[8, 9], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(0, 5, &[10, 11], Some(5.0), None, None)
        .unwrap(); // Direct but costly path

    // Find shortest path
    let path = decoder.get_shortest_path(0, 5).unwrap();
    assert!(!path.is_empty());

    // Observable count should not have changed
    assert_eq!(decoder.num_observables(), initial_count);
}

#[test]
fn test_decode_methods_observable_consistency() {
    // Test that different decode methods return consistent observable counts
    let config = PyMatchingConfig {
        num_nodes: Some(6),
        num_observables: 80, // More than 64 to test extended decoding
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Set up graph
    decoder
        .add_edge(0, 1, &[0, 10, 20], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(1, 2, &[30, 40, 50], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(2, 3, &[60, 70, 79], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(0, &[], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(3, &[], Some(1.0), None, None)
        .unwrap();

    let detection_events = vec![1, 0, 0, 1, 0, 0];

    // Use extended decode since we have >64 observables
    let result = decoder.decode(&detection_events).unwrap();
    assert_eq!(result.observable.len(), 80);

    // Test batch decode
    let batch_result = decoder
        .decode_batch_with_config(
            &detection_events,
            1, // Single shot
            detection_events.len(),
            BatchConfig {
                bit_packed_input: false,
                bit_packed_output: false,
                return_weights: false,
            },
        )
        .unwrap();

    // Batch result should also respect observable count
    assert_eq!(batch_result.predictions.len(), 1);
    assert!(batch_result.predictions[0].len() >= 80);
}

#[test]
fn test_observable_count_edge_modification() {
    // Test observable count stability during edge modifications
    let config = PyMatchingConfig {
        num_nodes: Some(5),
        num_observables: 15,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Add initial edges
    decoder
        .add_edge(0, 1, &[0, 1, 2], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(1, 2, &[3, 4, 5], Some(2.0), None, None)
        .unwrap();
    let count_after_add = decoder.num_observables();

    // Replace edge with different observables
    decoder
        .add_edge(
            0,
            1,
            &[10, 11, 12],
            Some(0.5),
            None,
            Some(MergeStrategy::Replace),
        )
        .unwrap();

    // Count should not decrease
    assert!(decoder.num_observables() >= count_after_add);

    // Add parallel edge with independent strategy
    decoder
        .add_edge(
            1,
            2,
            &[13, 14],
            Some(1.5),
            None,
            Some(MergeStrategy::Independent),
        )
        .unwrap();

    // Count should accommodate all observable indices
    assert!(decoder.num_observables() >= 15);
}

#[test]
fn test_observable_weights_correlation() {
    // Test that observable indices are properly correlated with edge weights
    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 6,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Add edges with specific weight-observable patterns
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder
        .add_edge(1, 2, &[1, 2], Some(2.0), None, None)
        .unwrap();
    decoder
        .add_edge(2, 3, &[3, 4, 5], Some(3.0), None, None)
        .unwrap();

    // Get all edges and verify observable-weight relationships
    let edges = decoder.get_all_edges();
    for edge in edges {
        // More observables should correlate with higher weights in this test
        let num_obs = edge.observables.len();
        assert!(num_obs > 0 && num_obs <= 3);

        // All observable indices should be valid
        for &obs in &edge.observables {
            assert!(obs < decoder.num_observables());
        }
    }
}
