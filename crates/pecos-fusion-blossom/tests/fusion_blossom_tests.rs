//! Tests for Fusion Blossom decoder integration

use ndarray::{Array2, array};
use pecos_fusion_blossom::{FusionBlossomConfig, FusionBlossomDecoder, SolverType};

#[test]
fn test_create_decoder() {
    let config = FusionBlossomConfig {
        num_nodes: Some(4),
        num_observables: 1,
        solver_type: SolverType::Serial,
        max_tree_size: None,
    };

    let decoder = FusionBlossomDecoder::new(config);
    assert!(decoder.is_ok());
}

#[test]
fn test_add_edges() {
    let config = FusionBlossomConfig {
        num_nodes: Some(4),
        num_observables: 2,
        solver_type: SolverType::Serial,
        max_tree_size: None,
    };

    let mut decoder = FusionBlossomDecoder::new(config).unwrap();

    // Add regular edge
    let result = decoder.add_edge(0, 1, &[0], Some(1.5));
    assert!(result.is_ok());

    // Add boundary edge
    let result = decoder.add_boundary_edge(2, &[1], Some(2.0));
    assert!(result.is_ok());
}

#[test]
fn test_decode_empty_syndrome() {
    let config = FusionBlossomConfig {
        num_nodes: Some(4),
        num_observables: 1,
        solver_type: SolverType::Serial,
        max_tree_size: None,
    };

    let mut decoder = FusionBlossomDecoder::new(config).unwrap();

    // Add some edges
    decoder.add_edge(0, 1, &[0], Some(1.0)).unwrap();
    decoder.add_edge(1, 2, &[0], Some(1.0)).unwrap();
    decoder.add_edge(2, 3, &[0], Some(1.0)).unwrap();

    // Empty syndrome
    let syndrome = array![0, 0, 0, 0];
    let result = decoder.decode(&syndrome.view());

    assert!(result.is_ok());
    let decoding = result.unwrap();
    assert_eq!(decoding.observable, vec![0]);
    assert!(
        decoding.weight.abs() < f64::EPSILON,
        "Weight should be zero but was {}",
        decoding.weight
    );
    assert!(decoding.matched_edges.is_empty());
}

#[test]
fn test_decode_simple_syndrome() {
    let config = FusionBlossomConfig {
        num_nodes: Some(4),
        num_observables: 1,
        solver_type: SolverType::Serial,
        max_tree_size: None,
    };

    let mut decoder = FusionBlossomDecoder::new(config).unwrap();

    // Create a simple chain: 0 -- 1 -- 2 -- 3
    decoder.add_edge(0, 1, &[0], Some(1.0)).unwrap();
    decoder.add_edge(1, 2, &[0], Some(1.0)).unwrap();
    decoder.add_edge(2, 3, &[0], Some(1.0)).unwrap();

    // Syndrome with defects at nodes 0 and 3
    let syndrome = array![1, 0, 0, 1];
    let result = decoder.decode(&syndrome.view());

    assert!(result.is_ok());
    let decoding = result.unwrap();
    // Should match path 0-1-2-3, flipping observable 3 times
    assert_eq!(decoding.observable, vec![1]);
    assert!(
        (decoding.weight - 3.0).abs() < f64::EPSILON,
        "Weight should be 3.0 but was {}",
        decoding.weight
    );
}

#[test]
fn test_from_check_matrix() {
    // Simple repetition code check matrix
    let check_matrix: Array2<u8> = array![[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1],];

    let weights = vec![1.0, 1.0, 1.0, 1.0];

    let config = FusionBlossomConfig {
        num_nodes: None, // Will be inferred from check matrix
        num_observables: 4,
        solver_type: SolverType::Serial,
        max_tree_size: None,
    };

    let decoder = FusionBlossomDecoder::from_check_matrix(&check_matrix, Some(&weights), config);

    assert!(decoder.is_ok());
    let mut decoder = decoder.unwrap();

    // Test decoding
    let syndrome = array![1, 0, 1]; // Errors on first and third checks
    let result = decoder.decode(&syndrome.view());

    assert!(result.is_ok());
}

#[test]
fn test_multiple_observables() {
    let config = FusionBlossomConfig {
        num_nodes: Some(4),
        num_observables: 3,
        solver_type: SolverType::Serial,
        max_tree_size: None,
    };

    let mut decoder = FusionBlossomDecoder::new(config).unwrap();

    // Add edges with different observable masks
    decoder.add_edge(0, 1, &[0, 2], Some(1.0)).unwrap();
    decoder.add_edge(1, 2, &[1], Some(1.0)).unwrap();
    decoder.add_edge(2, 3, &[0, 1], Some(1.0)).unwrap();

    // Syndrome with defects at nodes 0 and 3
    let syndrome = array![1, 0, 0, 1];
    let result = decoder.decode(&syndrome.view());

    assert!(result.is_ok());
    let decoding = result.unwrap();
    assert_eq!(decoding.observable.len(), 3);
}

#[test]
fn test_error_cases() {
    let config = FusionBlossomConfig {
        num_nodes: Some(4),
        num_observables: 1,
        solver_type: SolverType::Serial,
        max_tree_size: None,
    };

    let mut decoder = FusionBlossomDecoder::new(config).unwrap();

    // Test invalid node index
    let result = decoder.add_edge(0, 5, &[0], Some(1.0));
    assert!(result.is_err());

    // Test negative weight
    let result = decoder.add_edge(0, 1, &[0], Some(-1.0));
    assert!(result.is_err());

    // Test wrong syndrome size
    let syndrome = array![1, 0]; // Too short
    let result = decoder.decode(&syndrome.view());
    assert!(result.is_err());
}
