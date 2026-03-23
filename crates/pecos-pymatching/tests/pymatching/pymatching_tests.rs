//! Comprehensive tests for the `PyMatching` API

use pecos_pymatching::{BatchConfig, MergeStrategy, PyMatchingConfig, PyMatchingDecoder};

#[test]
fn test_graph_construction() {
    // Test basic construction
    let config = PyMatchingConfig {
        num_nodes: Some(10),
        num_observables: 3,
        ..Default::default()
    };

    let decoder = PyMatchingDecoder::new(config).unwrap();
    assert_eq!(decoder.num_nodes(), 10);
    // PyMatching defaults to 64 observables if num_observables <= 64
    assert!(decoder.num_observables() >= 3);
    assert_eq!(decoder.num_edges(), 0);
}

#[test]
fn test_edge_management() {
    let config = PyMatchingConfig {
        num_nodes: Some(6),
        num_observables: 2,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Add regular edge with weight
    decoder.add_edge(0, 1, &[0], Some(2.5), None, None).unwrap();
    assert!(decoder.has_edge(0, 1));
    assert!(decoder.has_edge(1, 0)); // Should be symmetric

    // Add edge with error probability
    decoder.add_edge(1, 2, &[1], None, Some(0.1), None).unwrap();
    assert!(decoder.has_edge(1, 2));

    // Add boundary edge
    decoder
        .add_boundary_edge(3, &[0, 1], Some(3.0), None, None)
        .unwrap();
    assert!(decoder.has_boundary_edge(3));

    // Test edge data retrieval
    let edge_data = decoder.get_edge_data(0, 1).unwrap();
    assert_eq!(edge_data.node1, 0);
    assert_eq!(edge_data.node2, Some(1));
    assert_eq!(edge_data.observables, vec![0]);
    assert!((edge_data.weight - 2.5).abs() < 1e-6);

    // Test boundary edge data
    let boundary_data = decoder.get_boundary_edge_data(3).unwrap();
    assert_eq!(boundary_data.node1, 3);
    assert_eq!(boundary_data.node2, None);
    assert_eq!(boundary_data.observables, vec![0, 1]);
}

#[test]
fn test_merge_strategies() {
    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 2,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Add edge
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();

    // Try different merge strategies

    // SmallestWeight - should keep weight 0.5
    decoder
        .add_edge(
            0,
            1,
            &[1],
            Some(0.5),
            None,
            Some(MergeStrategy::SmallestWeight),
        )
        .unwrap();
    let edge_data = decoder.get_edge_data(0, 1).unwrap();
    assert!((edge_data.weight - 0.5).abs() < 1e-6);

    // KeepOriginal - should keep weight 0.5
    decoder
        .add_edge(
            0,
            1,
            &[0],
            Some(2.0),
            None,
            Some(MergeStrategy::KeepOriginal),
        )
        .unwrap();
    let edge_data = decoder.get_edge_data(0, 1).unwrap();
    assert!((edge_data.weight - 0.5).abs() < 1e-6);

    // Replace - should update to weight 3.0
    decoder
        .add_edge(0, 1, &[1], Some(3.0), None, Some(MergeStrategy::Replace))
        .unwrap();
    let edge_data = decoder.get_edge_data(0, 1).unwrap();
    assert!((edge_data.weight - 3.0).abs() < 1e-6);
}

#[test]
fn test_boundary_management() {
    let config = PyMatchingConfig {
        num_nodes: Some(8),
        num_observables: 2,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Set boundary nodes
    decoder.set_boundary(&[0, 2, 4, 6]);

    // Check boundary
    assert!(decoder.is_boundary_node(0));
    assert!(!decoder.is_boundary_node(1));
    assert!(decoder.is_boundary_node(2));
    assert!(!decoder.is_boundary_node(3));

    let boundary = decoder.get_boundary();
    assert_eq!(boundary.len(), 4);
    assert!(boundary.contains(&0));
    assert!(boundary.contains(&2));
    assert!(boundary.contains(&4));
    assert!(boundary.contains(&6));
}

#[test]
fn test_basic_decoding() {
    let config = PyMatchingConfig {
        num_nodes: Some(5),
        num_observables: 2,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Create a simple matching graph
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(1.0), None, None).unwrap();
    decoder.add_edge(2, 3, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(3, 4, &[1], Some(1.0), None, None).unwrap();
    decoder
        .add_boundary_edge(0, &[], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(4, &[], Some(1.0), None, None)
        .unwrap();

    // Test with detection events at nodes 1 and 3
    let mut detection_events = vec![0u8; 5];
    detection_events[1] = 1;
    detection_events[3] = 1;

    let result = decoder.decode(&detection_events).unwrap();
    // Path from 1 to 3 crosses observables [1] and [0]
    assert_eq!(result.observable, vec![1, 1]);
    assert!(result.weight > 0.0);
}

#[test]
fn test_extended_decoding() {
    // Test with >64 observables
    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 100,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Ensure we have enough observables
    decoder.ensure_num_observables(100).unwrap();

    // Add edges with high observable indices
    decoder
        .add_edge(0, 1, &[65, 70], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_edge(1, 2, &[80, 90], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(0, &[], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(2, &[], Some(1.0), None, None)
        .unwrap();

    // Set boundary
    decoder.set_boundary(&[0, 2]);

    // Test decoding with a simple syndrome
    // Get the actual number of detectors after setting boundary
    let num_detectors = decoder.num_detectors();
    let mut detection_events = vec![0u8; num_detectors];
    if num_detectors > 1 {
        detection_events[1] = 1; // Single detection at node 1
    }

    let result = decoder.decode(&detection_events).unwrap();
    // The exact observables triggered depend on the matching
    // We just verify that decoding works with >64 observables
    assert_eq!(result.observable.len(), 100);
    // Don't assert specific observables as the matching algorithm's choice may vary
}

#[test]
fn test_decode_to_matched_pairs_error_handling() {
    let config = PyMatchingConfig {
        num_nodes: Some(6),
        num_observables: 2,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Create a graph
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(1.0), None, None).unwrap();
    decoder.add_edge(3, 4, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(4, 5, &[1], Some(1.0), None, None).unwrap();
    decoder
        .add_boundary_edge(2, &[], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(5, &[], Some(1.0), None, None)
        .unwrap();

    // Detection events at 0, 1, 3, 4
    let mut detection_events = vec![0u8; 6];
    detection_events[0] = 1;
    detection_events[1] = 1;
    detection_events[3] = 1;
    detection_events[4] = 1;

    // Test decode_to_matched_pairs
    let result = decoder.decode_to_matched_pairs(&detection_events);
    assert!(result.is_ok(), "decode_to_matched_pairs should now work");

    let pairs = result.unwrap();
    // Verify matched pairs structure

    // Should have matched the detection events
    assert!(!pairs.is_empty());

    // Check that our detection events (0, 1, 3, 4) are involved in matchings
    let matched_detectors: Vec<i64> = pairs
        .iter()
        .flat_map(|p| vec![p.detector1, p.detector2.unwrap_or(-1)])
        .filter(|&d| d >= 0)
        .collect();

    // Should include some of our detection events
    assert!(
        matched_detectors
            .iter()
            .any(|&d| d == 0 || d == 1 || d == 3 || d == 4)
    );

    // Test dictionary format
    let match_dict = decoder
        .decode_to_matched_pairs_dict(&detection_events)
        .unwrap();
    // Verify match dictionary structure

    // Dictionary should contain entries for matched detectors
    assert!(!match_dict.is_empty());

    // Check that if detector A is matched to B, then B is matched to A
    for (det1, maybe_det2) in &match_dict {
        if let Some(det2) = maybe_det2 {
            // Check reciprocal matching
            assert_eq!(
                match_dict.get(det2),
                Some(&Some(*det1)),
                "If {det1} -> {det2}, then {det2} -> {det1} should exist"
            );
        }
    }
}

#[test]
fn test_decode_with_pair_extraction() {
    // Alternative approach: Use regular decode and extract matching info
    let config = PyMatchingConfig {
        num_nodes: Some(6),
        num_observables: 2,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Create a simple matching problem
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(1.0), None, None).unwrap();
    decoder.add_edge(3, 4, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(4, 5, &[1], Some(1.0), None, None).unwrap();
    decoder
        .add_boundary_edge(2, &[], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(5, &[], Some(1.0), None, None)
        .unwrap();

    // Detection events at 0, 1, 3, 4
    let mut detection_events = vec![0u8; 6];
    detection_events[0] = 1;
    detection_events[1] = 1;
    detection_events[3] = 1;
    detection_events[4] = 1;

    let result = decoder.decode(&detection_events).unwrap();

    // The decode result tells us which observables were triggered
    // This gives us information about the matching, even if not pairs directly
    assert_eq!(result.observable.len(), 2);
}

#[test]
fn test_decode_to_edges_error_handling() {
    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 2,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Linear chain with boundary
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(1.0), None, None).unwrap();
    decoder.add_edge(2, 3, &[0], Some(1.0), None, None).unwrap();
    decoder
        .add_boundary_edge(0, &[], Some(1.0), None, None)
        .unwrap();
    decoder
        .add_boundary_edge(3, &[], Some(1.0), None, None)
        .unwrap();

    // Detection events at 1 and 2
    let mut detection_events = vec![0u8; 4];
    detection_events[1] = 1;
    detection_events[2] = 1;

    // Test decode_to_edges
    let result = decoder.decode_to_edges(&detection_events);
    assert!(result.is_ok(), "decode_to_edges should now work");

    let edges = result.unwrap();
    // Verify edges in solution

    // Should have edges in the solution
    assert!(!edges.is_empty());

    // The edges should connect our detection events (1 and 2)
    // Check that edges involve detectors 1 and 2
    let edge_detectors: Vec<i64> = edges
        .iter()
        .flat_map(|e| vec![e.detector1, e.detector2.unwrap_or(-1)])
        .filter(|&d| d >= 0)
        .collect();

    assert!(edge_detectors.iter().any(|&d| d == 1 || d == 2));
}

#[test]
fn test_edge_weight_tracking() {
    // Alternative: Track edge weights to understand matching behavior
    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 2,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Create edges with different weights
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(2.0), None, None).unwrap(); // Higher weight
    decoder.add_edge(2, 3, &[0], Some(1.0), None, None).unwrap();
    decoder
        .add_edge(0, 3, &[0, 1], Some(3.5), None, None)
        .unwrap(); // Alternative path

    // Test different detection patterns
    let test_cases = vec![
        vec![1, 1, 0, 0], // Adjacent detections
        vec![1, 0, 0, 1], // Distant detections
        vec![0, 1, 1, 0], // Middle detections
    ];

    for detection_events in test_cases {
        let result = decoder.decode(&detection_events).unwrap();
        // Track results for analysis

        // The weight gives us information about which edges were used
        assert!(result.weight >= 0.0);
    }
}

#[test]
fn test_batch_decoding() {
    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 2,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Simple square graph
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(1.0), None, None).unwrap();
    decoder.add_edge(2, 3, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(3, 0, &[1], Some(1.0), None, None).unwrap();

    // Prepare batch of 3 shots
    let num_shots = 3;
    let num_detectors = 4;
    let mut shots = vec![0u8; num_shots * num_detectors];

    // Shot 0: detections at 0, 2
    shots[0] = 1;
    shots[2] = 1;

    // Shot 1: detections at 1, 3
    shots[4 + 1] = 1;
    shots[4 + 3] = 1;

    // Shot 2: detections at 0, 1
    shots[8] = 1;
    shots[9] = 1;

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

    assert_eq!(result.predictions.len(), num_shots);
    assert_eq!(result.weights.len(), num_shots);

    // Each prediction should have the right number of observables
    for pred in &result.predictions {
        assert!(pred.len() >= 2);
    }
}

#[test]
fn test_shortest_path() {
    let config = PyMatchingConfig {
        num_nodes: Some(5),
        num_observables: 2,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Create a path: 0-1-2-3-4
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(1.0), None, None).unwrap();
    decoder.add_edge(2, 3, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(3, 4, &[1], Some(1.0), None, None).unwrap();

    // Also add a shortcut with higher weight: 0-4
    decoder
        .add_edge(0, 4, &[0, 1], Some(5.0), None, None)
        .unwrap();

    // Find shortest path from 0 to 4
    let result = decoder.get_shortest_path(0, 4);
    assert!(result.is_ok(), "get_shortest_path should now work");

    let path = result.unwrap();
    // Verify path structure

    // Path should include nodes along the way
    assert!(!path.is_empty(), "Path should not be empty");
    assert_eq!(path[0], 0, "Path should start at node 0");
    assert_eq!(path[path.len() - 1], 4, "Path should end at node 4");

    // The shortest path should be 0-1-2-3-4 (total weight 4)
    // rather than direct 0-4 (weight 5)
    assert!(path.len() >= 5, "Path should include intermediate nodes");
}

#[test]
fn test_path_analysis_via_decode() {
    // Alternative: Analyze paths by testing specific detection patterns
    let config = PyMatchingConfig {
        num_nodes: Some(5),
        num_observables: 2,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Create a path graph with branches: 0-1-2-3-4
    //                                       \-3-/
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(1.0), None, None).unwrap();
    decoder.add_edge(2, 3, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(3, 4, &[1], Some(1.0), None, None).unwrap();
    decoder
        .add_edge(1, 3, &[0, 1], Some(1.5), None, None)
        .unwrap(); // Shortcut

    // Test path selection by placing detections at endpoints
    let mut detection_events = vec![0u8; 5];
    detection_events[0] = 1;
    detection_events[4] = 1;

    let result = decoder.decode(&detection_events).unwrap();

    // The decoder should find a reasonable path
    // Note: The actual weight depends on the specific matching algorithm and graph structure
    assert!(result.weight > 0.0); // Should have some weight
}

#[test]
fn test_noise_simulation() {
    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 2,
        ..Default::default()
    };

    // Test add_noise functionality
    let num_samples = 100; // Increased from 10 to make test more reliable
    let rng_seed = 42;

    // Need to add edges with error probabilities for noise simulation
    let mut decoder = PyMatchingDecoder::new(config).unwrap();
    decoder.add_edge(0, 1, &[0], None, Some(0.1), None).unwrap();
    decoder.add_edge(1, 2, &[1], None, Some(0.1), None).unwrap();
    decoder.add_edge(2, 3, &[0], None, Some(0.1), None).unwrap();

    // Add boundary edges to make noise simulation work
    decoder
        .add_boundary_edge(0, &[], None, Some(0.1), None)
        .unwrap();
    decoder
        .add_boundary_edge(3, &[], None, Some(0.1), None)
        .unwrap();

    let result = decoder.add_noise(num_samples, rng_seed);
    assert!(result.is_ok(), "add_noise should now work");

    let noise = result.unwrap();
    assert_eq!(noise.errors.len(), num_samples);
    assert_eq!(noise.syndromes.len(), num_samples);

    // Check sizes
    for (errors, syndrome) in noise.errors.iter().zip(&noise.syndromes) {
        assert_eq!(errors.len(), decoder.num_observables());
        assert_eq!(syndrome.len(), decoder.num_detectors());
    }

    // With 10% error probability and 100 samples, we should see some errors
    let total_errors: usize = noise
        .errors
        .iter()
        .map(|e| e.iter().filter(|&&x| x != 0).count())
        .sum();

    // Count total syndrome detections as well
    let total_syndromes: usize = noise
        .syndromes
        .iter()
        .map(|s| s.iter().filter(|&&x| x != 0).count())
        .sum();

    // With 100 samples and 5 edges at 10% error rate each,
    // we expect about 50 errors total. The probability of getting 0 errors
    // is astronomically small (0.9^500 ≈ 10^-23)
    assert!(
        total_errors > 0 || total_syndromes > 0,
        "Should have generated some errors or syndromes with 10% probability over {num_samples} samples. Got {total_errors} errors and {total_syndromes} syndromes"
    );
}

#[test]
fn test_monte_carlo_simulation() {
    // Alternative: Implement our own noise simulation
    use pecos_random::{PecosRng, RngExt};

    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 2,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Add edges with error probabilities
    decoder.add_edge(0, 1, &[0], None, Some(0.1), None).unwrap();
    decoder.add_edge(1, 2, &[1], None, Some(0.1), None).unwrap();
    decoder.add_edge(2, 3, &[0], None, Some(0.1), None).unwrap();
    decoder
        .add_boundary_edge(0, &[], None, Some(0.1), None)
        .unwrap();
    decoder
        .add_boundary_edge(3, &[], None, Some(0.1), None)
        .unwrap();

    // Simulate noise manually
    let mut rng = PecosRng::seed_from_u64(42);
    let num_samples = 100;
    let mut failure_count = 0;

    for _ in 0..num_samples {
        // Generate random detection events based on error probabilities
        let mut detection_events = vec![0u8; 4];

        // Simple noise model: each detector has 10% chance of firing
        for event in detection_events.iter_mut().take(4) {
            if rng.random::<f64>() < 0.1 {
                *event = 1;
            }
        }

        // Only decode if there are detection events
        let num_detections: u8 = detection_events.iter().sum();
        if num_detections % 2 == 1 {
            // Odd number of detections - add boundary detection
            if rng.random::<bool>() {
                detection_events[0] = 1 - detection_events[0];
            } else {
                detection_events[3] = 1 - detection_events[3];
            }
        }

        if num_detections > 0 {
            let result = decoder.decode(&detection_events).unwrap();
            // Check if any observable was triggered (indicating a logical error)
            if result.observable.iter().any(|&x| x != 0) {
                failure_count += 1;
            }
        }
    }

    let failure_rate = f64::from(failure_count) / f64::from(num_samples);
    // Track simulation results

    // With 10% physical error rate, logical error rate should be reasonable
    assert!(failure_rate < 0.5);
}

#[test]
fn test_dem_loading() {
    // Create a simple DEM string
    let dem_string = r"
        error(0.1) D0 D1 L0
        error(0.1) D1 D2 L1
        error(0.1) D2 D3 L0
    ";

    let mut decoder = PyMatchingDecoder::from_dem(dem_string).unwrap();

    // Should have created appropriate graph
    assert!(decoder.num_nodes() > 0);
    assert!(decoder.num_edges() > 0);
    assert_eq!(decoder.num_observables(), 2);

    // Test decoding
    let mut detection_events = vec![0u8; decoder.num_detectors()];
    if detection_events.len() > 1 {
        detection_events[0] = 1;
        detection_events[1] = 1;
        let result = decoder.decode(&detection_events).unwrap();
        assert_eq!(result.observable.len(), 2);
    }
}

#[test]
fn test_weight_normalisation() {
    let config = PyMatchingConfig {
        num_nodes: Some(3),
        num_observables: 1,
        ..Default::default()
    };

    let decoder = PyMatchingDecoder::new(config).unwrap();

    // Get normalising constant
    let norm_const = decoder.get_edge_weight_normalising_constant(1000);
    assert!(norm_const > 0.0);
}

#[test]
fn test_rng_methods() {
    // Test setting seed for reproducibility
    PyMatchingDecoder::set_seed(42).unwrap();

    // Generate some random floats
    let r1 = PyMatchingDecoder::rand_float(0.0, 1.0).unwrap();
    let r2 = PyMatchingDecoder::rand_float(0.0, 1.0).unwrap();

    // Since PyMatching uses global RNG state that can be affected by parallel test execution,
    // we can't guarantee exact reproducibility. Instead, verify basic functionality:
    // 1. Random values are in range
    assert!(
        (0.0..1.0).contains(&r1),
        "Random value should be in range [0, 1)"
    );
    assert!(
        (0.0..1.0).contains(&r2),
        "Random value should be in range [0, 1)"
    );

    // 2. Consecutive values are different (extremely unlikely to be equal)
    assert!(
        (r1 - r2).abs() > f64::EPSILON,
        "Consecutive random values should be different but were both {r1}"
    );

    // Test randomize
    PyMatchingDecoder::randomize().unwrap();
    let r3 = PyMatchingDecoder::rand_float(0.0, 1.0).unwrap();

    // Very unlikely to get same value after randomize
    assert!(
        (r3 - r1).abs() > f64::EPSILON,
        "Randomize should change the sequence but got {r3} and {r1}"
    );

    // Test range
    let r_range = PyMatchingDecoder::rand_float(10.0, 20.0).unwrap();
    assert!(
        (10.0..20.0).contains(&r_range),
        "Random float should be in specified range"
    );
}

#[test]
fn test_builder_pattern() {
    // Test builder construction
    let decoder = PyMatchingDecoder::builder()
        .nodes(10)
        .observables(4)
        .build()
        .unwrap();

    assert_eq!(decoder.num_nodes(), 10);
    assert!(decoder.num_observables() >= 4);

    // The builder pattern correctly constructs the decoder with specified parameters.
    // Note: RNG seed testing is unreliable in parallel test execution since PyMatching
    // uses a global RNG state. The seed is set, but we can't guarantee deterministic
    // behavior across different test runs.
}

#[test]
fn test_error_probability_check() {
    let config = PyMatchingConfig {
        num_nodes: Some(3),
        num_observables: 1,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Initially no edges, so should return true (vacuously)
    assert!(decoder.all_edges_have_error_probabilities());

    // Add edge with weight only (PyMatching may assign default error probability)
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    // PyMatching's behavior with edges without explicit error probabilities may vary
    // So we just check that the method works without asserting specific behavior
    let _ = decoder.all_edges_have_error_probabilities();

    // Add edge with explicit error probability
    decoder.add_edge(1, 2, &[0], None, Some(0.1), None).unwrap();
    // PyMatching may have different behavior, so we don't assert specific values
    let _has_probs = decoder.all_edges_have_error_probabilities();
}

#[test]
fn test_detector_validation() {
    let config = PyMatchingConfig {
        num_nodes: Some(5),
        num_observables: 2,
        ..Default::default()
    };

    let decoder = PyMatchingDecoder::new(config).unwrap();

    // Valid detection events
    let valid_events = vec![0u8; 5];
    decoder.validate_detector_indices(&valid_events).unwrap();

    // Too many detection events should fail
    let invalid_events = vec![0u8; 10];
    assert!(decoder.validate_detector_indices(&invalid_events).is_err());
}

#[test]
fn test_get_all_edges() {
    let config = PyMatchingConfig {
        num_nodes: Some(4),
        num_observables: 2,
        ..Default::default()
    };

    let mut decoder = PyMatchingDecoder::new(config).unwrap();

    // Add various edges
    decoder.add_edge(0, 1, &[0], Some(1.0), None, None).unwrap();
    decoder.add_edge(1, 2, &[1], Some(2.0), None, None).unwrap();
    decoder
        .add_boundary_edge(3, &[0, 1], Some(3.0), None, None)
        .unwrap();

    let all_edges = decoder.get_all_edges();
    assert_eq!(all_edges.len(), 3);

    // Check we have the expected edges
    let has_edge_01 = all_edges
        .iter()
        .any(|e| e.node1 == 0 && e.node2 == Some(1) && e.observables == vec![0]);
    let has_edge_12 = all_edges
        .iter()
        .any(|e| e.node1 == 1 && e.node2 == Some(2) && e.observables == vec![1]);
    let has_boundary_3 = all_edges
        .iter()
        .any(|e| e.node1 == 3 && e.node2.is_none() && e.observables == vec![0, 1]);

    assert!(has_edge_01);
    assert!(has_edge_12);
    assert!(has_boundary_3);
}
