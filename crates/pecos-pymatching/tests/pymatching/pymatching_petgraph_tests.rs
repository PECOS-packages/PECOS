//! Tests for `PyMatching` petgraph integration

use ::petgraph::graph::{NodeIndex, UnGraph};
use pecos_pymatching::*;
use std::collections::HashSet;

#[test]
fn test_basic_petgraph_conversion() {
    // Create a simple PyMatching decoder
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(5)
        .observables(3)
        .build()
        .unwrap();

    // Add edges to form a chain
    decoder
        .add_edge(0, 1, &[0], Some(1.0), Some(0.1), None)
        .unwrap();
    decoder
        .add_edge(1, 2, &[1], Some(2.0), Some(0.2), None)
        .unwrap();
    decoder.add_edge(2, 3, &[2], Some(1.5), None, None).unwrap();
    decoder
        .add_edge(3, 4, &[0, 1], Some(3.0), Some(0.3), None)
        .unwrap();

    // Convert to petgraph
    let (graph, node_map) = pymatching_to_petgraph(&decoder);

    // Verify all nodes are present
    assert_eq!(graph.node_count(), 5);
    assert_eq!(graph.edge_count(), 4);

    // Verify node data
    for i in 0..5 {
        let idx = node_map[&i];
        assert_eq!(graph[idx].id, i);
        assert!(!graph[idx].is_boundary); // No boundary nodes set
    }

    // Verify edge data
    let edge_01 = graph
        .edges_connecting(node_map[&0], node_map[&1])
        .next()
        .unwrap();
    let edge_weight = edge_01.weight();
    assert_eq!(edge_weight.observables, vec![0]);
    // When error probability is provided, weight is calculated as -ln((1-p)/p)
    let expected_weight = -((1.0 - 0.1) / 0.1_f64).ln();
    assert!((edge_weight.weight - expected_weight).abs() < 1e-10);
    assert_eq!(edge_weight.error_probability, Some(0.1));
}

#[test]
fn test_petgraph_with_boundary_nodes() {
    // Create a petgraph with specific structure
    let mut graph = UnGraph::new_undirected();

    // Add nodes
    let n0 = graph.add_node(PyMatchingNode {
        id: 0,
        is_boundary: false,
    });
    let n1 = graph.add_node(PyMatchingNode {
        id: 1,
        is_boundary: false,
    });
    let n2 = graph.add_node(PyMatchingNode {
        id: 2,
        is_boundary: true,
    });
    let n3 = graph.add_node(PyMatchingNode {
        id: 3,
        is_boundary: false,
    });

    // Add edges
    graph.add_edge(
        n0,
        n1,
        PyMatchingEdge {
            observables: vec![0],
            weight: 1.0,
            error_probability: Some(0.05),
        },
    );
    graph.add_edge(
        n1,
        n2,
        PyMatchingEdge {
            observables: vec![1],
            weight: 2.0,
            error_probability: None,
        },
    );
    graph.add_edge(
        n2,
        n3,
        PyMatchingEdge {
            observables: vec![0, 1],
            weight: 1.5,
            error_probability: Some(0.1),
        },
    );

    // Mark n2 as boundary
    let mut boundary_nodes = HashSet::new();
    boundary_nodes.insert(n2);

    // Convert to PyMatching
    let decoder = pymatching_from_petgraph(&graph, &boundary_nodes, 2).unwrap();

    // Verify structure
    assert_eq!(decoder.num_nodes(), 4);
    // PyMatching uses a minimum of 64 observables by default
    assert!(decoder.num_observables() >= 2);

    // Verify edges
    assert!(decoder.has_edge(0, 1));
    assert!(decoder.has_edge(1, 2));
    assert!(decoder.has_edge(2, 3));

    // Verify boundary
    let boundary = decoder.get_boundary();
    assert!(boundary.contains(&2));
}

#[test]
fn test_weighted_graph_conversion() {
    // Create a simple triangle graph with weights
    let mut graph = UnGraph::new_undirected();

    let n0 = graph.add_node(());
    let n1 = graph.add_node(());
    let n2 = graph.add_node(());

    graph.add_edge(n0, n1, 1.0);
    graph.add_edge(n1, n2, 2.0);
    graph.add_edge(n2, n0, 3.0);

    // Convert to PyMatching
    let mut decoder = pymatching_from_petgraph_weighted(&graph, Some(3)).unwrap();

    // Verify structure
    assert_eq!(decoder.num_nodes(), 3);
    assert_eq!(decoder.num_edges(), 3);
    // PyMatching uses a minimum of 64 observables by default
    assert!(decoder.num_observables() >= 3);

    // Test decoding
    let mut syndrome = vec![0u8; 3];
    syndrome[0] = 1;
    syndrome[1] = 1;

    let result = decoder.decode(&syndrome).unwrap();
    // Should find some matching
    assert!(result.weight > 0.0);
}

#[test]
fn test_round_trip_preservation() {
    // Create a more complex decoder
    let mut decoder1 = PyMatchingDecoder::builder()
        .nodes(6)
        .observables(4)
        .build()
        .unwrap();

    // Add various edges
    decoder1
        .add_edge(0, 1, &[0], Some(1.0), Some(0.1), None)
        .unwrap();
    decoder1
        .add_edge(1, 2, &[1], Some(2.0), None, None)
        .unwrap();
    decoder1
        .add_edge(2, 3, &[2], Some(1.5), Some(0.15), None)
        .unwrap();
    decoder1
        .add_edge(3, 4, &[3], Some(2.5), None, None)
        .unwrap();
    decoder1
        .add_edge(4, 5, &[0, 2], Some(3.0), Some(0.2), None)
        .unwrap();
    decoder1
        .add_edge(5, 0, &[1, 3], Some(1.8), None, None)
        .unwrap();

    // Set some boundary nodes
    decoder1.set_boundary(&[0, 3]);

    // Convert to petgraph and back
    let (graph, _node_map) = pymatching_to_petgraph(&decoder1);

    let decoder2 = pymatching_from_petgraph(&graph, &HashSet::new(), 4).unwrap();

    // Verify all edges are preserved
    assert!(decoder2.has_edge(0, 1));
    assert!(decoder2.has_edge(1, 2));
    assert!(decoder2.has_edge(2, 3));
    assert!(decoder2.has_edge(3, 4));
    assert!(decoder2.has_edge(4, 5));
    assert!(decoder2.has_edge(5, 0));

    // Note: Boundary information is stored in node data but not automatically
    // restored without explicit boundary_nodes parameter
}

#[test]
fn test_decoding_after_conversion() {
    // Create a surface code-like structure in petgraph
    let mut graph = UnGraph::new_undirected();

    // Create a 2x2 grid of data qubits (4 nodes)
    // 0---1
    // |   |
    // 2---3
    let nodes: Vec<_> = (0..4)
        .map(|i| {
            graph.add_node(PyMatchingNode {
                id: i,
                is_boundary: false,
            })
        })
        .collect();

    // Add edges with observables
    graph.add_edge(
        nodes[0],
        nodes[1],
        PyMatchingEdge {
            observables: vec![0],
            weight: 1.0,
            error_probability: Some(0.01),
        },
    );
    graph.add_edge(
        nodes[0],
        nodes[2],
        PyMatchingEdge {
            observables: vec![1],
            weight: 1.0,
            error_probability: Some(0.01),
        },
    );
    graph.add_edge(
        nodes[1],
        nodes[3],
        PyMatchingEdge {
            observables: vec![1],
            weight: 1.0,
            error_probability: Some(0.01),
        },
    );
    graph.add_edge(
        nodes[2],
        nodes[3],
        PyMatchingEdge {
            observables: vec![0],
            weight: 1.0,
            error_probability: Some(0.01),
        },
    );

    // Convert to PyMatching
    let mut decoder = pymatching_from_petgraph(&graph, &HashSet::new(), 2).unwrap();

    // Test decoding various syndromes
    let test_cases = vec![
        (vec![1, 1, 0, 0], vec![1, 0]), // Nodes 0,1 active -> observable 0
        (vec![1, 0, 1, 0], vec![0, 1]), // Nodes 0,2 active -> observable 1
        (vec![0, 0, 0, 0], vec![0, 0]), // No syndrome -> no correction
    ];

    for (syndrome, expected) in test_cases {
        let result = decoder.decode(&syndrome).unwrap();
        assert_eq!(result.observable, expected);
    }
}

#[test]
fn test_large_graph_performance() {
    use std::time::Instant;

    // Create a larger graph (10x10 grid)
    let size = 10;
    let mut graph = UnGraph::new_undirected();

    // Add nodes
    let mut node_grid = vec![vec![NodeIndex::default(); size]; size];
    for (i, row) in node_grid.iter_mut().enumerate() {
        for (j, cell) in row.iter_mut().enumerate() {
            let id = i * size + j;
            *cell = graph.add_node(PyMatchingNode {
                id,
                is_boundary: false,
            });
        }
    }

    // Add edges (grid connectivity)
    let mut obs_idx = 0;
    for i in 0..size {
        for j in 0..size {
            // Right edge
            if j < size - 1 {
                graph.add_edge(
                    node_grid[i][j],
                    node_grid[i][j + 1],
                    PyMatchingEdge {
                        observables: vec![obs_idx % 10],
                        weight: 1.0,
                        error_probability: Some(0.01),
                    },
                );
                obs_idx += 1;
            }
            // Down edge
            if i < size - 1 {
                graph.add_edge(
                    node_grid[i][j],
                    node_grid[i + 1][j],
                    PyMatchingEdge {
                        observables: vec![obs_idx % 10],
                        weight: 1.0,
                        error_probability: Some(0.01),
                    },
                );
                obs_idx += 1;
            }
        }
    }

    // Time the conversion
    let start = Instant::now();
    let mut decoder = pymatching_from_petgraph(&graph, &HashSet::new(), 10).unwrap();
    let conversion_time = start.elapsed();

    println!("Conversion time for {size}x{size} grid: {conversion_time:?}");

    // Verify structure
    assert_eq!(decoder.num_nodes(), size * size);
    assert!(decoder.num_edges() > 0);

    // Test that decoding works
    let syndrome = vec![0u8; size * size];
    let result = decoder.decode(&syndrome).unwrap();
    assert!(
        result.weight.abs() < f64::EPSILON,
        "Weight should be zero but was {}",
        result.weight
    ); // Zero syndrome should give zero weight
}
