//! Example demonstrating `PyMatching`'s petgraph integration
//!
//! This example shows how to:
//! 1. Create a graph using petgraph
//! 2. Convert it to `PyMatching`
//! 3. Use it for decoding
//! 4. Convert back to petgraph for further analysis

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use ::petgraph::graph::UnGraph;
    use pecos_pymatching::{
        PyMatchingEdge, PyMatchingNode, pymatching_from_petgraph,
        pymatching_from_petgraph_weighted, pymatching_to_petgraph,
    };
    use std::collections::HashSet;

    println!("=== PyMatching Petgraph Integration Example ===\n");

    // Create a surface code-like graph using petgraph
    let mut graph = UnGraph::new_undirected();

    // Create a 3x3 grid of nodes
    let mut nodes = Vec::new();
    for i in 0..9 {
        let node = graph.add_node(PyMatchingNode {
            id: i,
            is_boundary: false,
        });
        nodes.push(node);
    }

    // Add horizontal edges
    for row in 0..3 {
        for col in 0..2 {
            let idx = row * 3 + col;
            graph.add_edge(
                nodes[idx],
                nodes[idx + 1],
                PyMatchingEdge {
                    observables: vec![idx % 2],
                    weight: 1.0,
                    error_probability: Some(0.01),
                },
            );
        }
    }

    // Add vertical edges
    for row in 0..2 {
        for col in 0..3 {
            let idx = row * 3 + col;
            graph.add_edge(
                nodes[idx],
                nodes[idx + 3],
                PyMatchingEdge {
                    observables: vec![(idx + 1) % 2],
                    weight: 1.0,
                    error_probability: Some(0.01),
                },
            );
        }
    }

    println!(
        "Created petgraph with {} nodes and {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // Convert to PyMatching
    let mut decoder = pymatching_from_petgraph(&graph, &HashSet::new(), 2)?;

    println!("Converted to PyMatching decoder:");
    println!("  Nodes: {}", decoder.num_nodes());
    println!("  Edges: {}", decoder.num_edges());
    println!("  Observables: {}", decoder.num_observables());

    // Test decoding with a simple syndrome
    let syndrome = vec![1, 1, 0, 0, 0, 0, 0, 0, 0]; // Nodes 0 and 1 active
    let result = decoder.decode(&syndrome).unwrap();

    println!("\nDecoding result:");
    println!("  Syndrome: {syndrome:?}");
    println!("  Correction: {:?}", result.observable);
    println!("  Weight: {}", result.weight);

    // Convert back to petgraph for analysis
    let (result_graph, node_map) = pymatching_to_petgraph(&decoder);

    println!("\nConverted back to petgraph:");
    println!("  Nodes: {}", result_graph.node_count());
    println!("  Edges: {}", result_graph.edge_count());

    // Example: Find neighbors of node 0
    if let Some(&node_idx) = node_map.get(&0) {
        let neighbors: Vec<_> = result_graph
            .neighbors(node_idx)
            .map(|n| result_graph[n].id)
            .collect();
        println!("  Neighbors of node 0: {neighbors:?}");
    }

    // Example: Create from weighted petgraph
    println!("\n=== Creating from Weighted Graph ===");

    let mut weighted_graph = UnGraph::new_undirected();
    let n0 = weighted_graph.add_node(());
    let n1 = weighted_graph.add_node(());
    let n2 = weighted_graph.add_node(());

    weighted_graph.add_edge(n0, n1, 1.5);
    weighted_graph.add_edge(n1, n2, 2.0);
    weighted_graph.add_edge(n2, n0, 2.5);

    let weighted_decoder = pymatching_from_petgraph_weighted(&weighted_graph, Some(3))?;

    println!("Created decoder from weighted graph:");
    println!("  Nodes: {}", weighted_decoder.num_nodes());
    println!("  Edges: {}", weighted_decoder.num_edges());

    Ok(())
}
