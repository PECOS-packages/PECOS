//! Petgraph integration for `PyMatching` decoder
//!
//! This module provides conversion between `PyMatching` decoders and petgraph graphs,
//! enabling interoperability with the Rust graph ecosystem.

use super::{PyMatchingDecoder, PyMatchingError};
use petgraph::graph::{NodeIndex, UnGraph};
use petgraph::visit::EdgeRef;
use std::collections::HashMap;

/// Node data for petgraph representation
#[derive(Debug, Clone)]
pub struct PyMatchingNode {
    /// Original node ID in `PyMatching`
    pub id: usize,
    /// Whether this is a boundary node
    pub is_boundary: bool,
}

/// Edge data for petgraph representation
#[derive(Debug, Clone)]
pub struct PyMatchingEdge {
    /// Observable indices crossed by this edge
    pub observables: Vec<usize>,
    /// Edge weight (log likelihood)
    pub weight: f64,
    /// Error probability (if available)
    pub error_probability: Option<f64>,
}

/// Create a `PyMatching` decoder from a petgraph undirected graph
///
/// # Arguments
/// * `graph` - The petgraph to convert
/// * `boundary_nodes` - Set of node indices that should be boundary nodes
/// * `num_observables` - Number of observables in the system
///
/// # Example
/// ```
/// # use pecos_pymatching::*;
/// # fn main() -> Result<(), PyMatchingError> {
/// use ::petgraph::graph::UnGraph;
/// use std::collections::HashSet;
///
/// let mut graph = UnGraph::new_undirected();
/// let n0 = graph.add_node(PyMatchingNode { id: 0, is_boundary: false });
/// let n1 = graph.add_node(PyMatchingNode { id: 1, is_boundary: false });
/// graph.add_edge(n0, n1, PyMatchingEdge {
///     observables: vec![0],
///     weight: 1.0,
///     error_probability: Some(0.1),
/// });
///
/// let decoder = pymatching_from_petgraph(&graph, &HashSet::new(), 1)?;
/// assert_eq!(decoder.num_nodes(), 2);
/// assert!(decoder.num_observables() >= 1);
/// # Ok(())
/// # }
/// ```
/// # Errors
///
/// Returns a [`PyMatchingError`] if:
/// - The decoder builder fails
/// - Setting the number of observables fails
/// - Adding an edge fails
pub fn pymatching_from_petgraph<S: std::hash::BuildHasher>(
    graph: &UnGraph<PyMatchingNode, PyMatchingEdge>,
    boundary_nodes: &std::collections::HashSet<NodeIndex, S>,
    num_observables: usize,
) -> Result<PyMatchingDecoder, PyMatchingError> {
    // Find the maximum node ID to determine graph size
    let max_node_id = graph.node_weights().map(|n| n.id).max().unwrap_or(0);

    // Create decoder with appropriate size
    let mut decoder = PyMatchingDecoder::builder()
        .nodes(max_node_id + 1)
        .observables(num_observables)
        .build()?;

    // Ensure the decoder has at least the requested number of observables
    // Note: PyMatching defaults to 64 observables minimum
    decoder.ensure_num_observables(num_observables)?;

    // Add all edges
    for edge in graph.edge_references() {
        let source_node = &graph[edge.source()];
        let target_node = &graph[edge.target()];
        let edge_data = edge.weight();

        // All edges in the petgraph are regular edges between nodes
        // Boundary nodes are just marked as such, but edges to them are still regular edges
        decoder.add_edge(
            source_node.id,
            target_node.id,
            &edge_data.observables,
            Some(edge_data.weight),
            edge_data.error_probability,
            None,
        )?;
    }

    // Set boundary nodes based on both explicit boundary_nodes set and node is_boundary flag
    let mut all_boundary_ids = Vec::new();

    // Add explicitly specified boundary nodes
    for &idx in boundary_nodes {
        all_boundary_ids.push(graph[idx].id);
    }

    // Add nodes marked as boundary in their data
    for node_idx in graph.node_indices() {
        if graph[node_idx].is_boundary && !boundary_nodes.contains(&node_idx) {
            all_boundary_ids.push(graph[node_idx].id);
        }
    }

    if !all_boundary_ids.is_empty() {
        decoder.set_boundary(&all_boundary_ids);
    }

    Ok(decoder)
}

/// Convert a `PyMatching` decoder to a petgraph undirected graph
///
/// # Returns
/// A tuple of (graph, `node_map`) where:
/// - graph is the petgraph representation
/// - `node_map` maps `PyMatching` node IDs to petgraph `NodeIndex`
///
/// # Example
/// ```
/// # use pecos_pymatching::*;
/// # fn main() -> Result<(), PyMatchingError> {
/// // Create a decoder
/// let mut decoder = PyMatchingDecoder::builder()
///     .nodes(3)
///     .observables(2)
///     .build()?;
///
/// // Add some edges
/// decoder.add_edge(0, 1, &[0], Some(1.0), None, None)?;
/// decoder.add_edge(1, 2, &[1], Some(2.0), None, None)?;
///
/// let (graph, node_map) = pymatching_to_petgraph(&decoder);
///
/// // Access nodes by their original PyMatching ID
/// let node_0_index = node_map[&0];
/// let node_data = &graph[node_0_index];
/// assert_eq!(node_data.id, 0);
/// assert_eq!(graph.node_count(), 3);
/// assert_eq!(graph.edge_count(), 2);
/// # Ok(())
/// # }
/// ```
#[must_use]
pub fn pymatching_to_petgraph(
    decoder: &PyMatchingDecoder,
) -> (
    UnGraph<PyMatchingNode, PyMatchingEdge>,
    HashMap<usize, NodeIndex>,
) {
    let mut graph = UnGraph::new_undirected();
    let mut node_map = HashMap::new();

    // Get boundary nodes
    let boundary_nodes = decoder.get_boundary();
    let boundary_set: std::collections::HashSet<_> = boundary_nodes.into_iter().collect();

    // Add all nodes
    let num_nodes = decoder.num_nodes();
    for node_id in 0..num_nodes {
        let node_data = PyMatchingNode {
            id: node_id,
            is_boundary: boundary_set.contains(&node_id),
        };
        let idx = graph.add_node(node_data);
        node_map.insert(node_id, idx);
    }

    // Add all edges
    let edges = decoder.get_all_edges();
    for edge in edges {
        // Skip boundary edges for now (they're represented differently in petgraph)
        if let Some(node2) = edge.node2
            && node2 < num_nodes
        {
            // Calculate weight from error probability if weight is NaN
            let weight = if edge.weight.is_nan()
                && edge.error_probability > 0.0
                && edge.error_probability < 1.0
            {
                // Weight = -log((1-p)/p) where p is error probability
                -((1.0 - edge.error_probability) / edge.error_probability).ln()
            } else {
                edge.weight
            };

            let edge_data = PyMatchingEdge {
                observables: edge.observables.clone(),
                weight,
                error_probability: if edge.error_probability.is_finite()
                    && edge.error_probability > 0.0
                {
                    Some(edge.error_probability)
                } else {
                    None
                },
            };

            if let (Some(&idx1), Some(&idx2)) = (node_map.get(&edge.node1), node_map.get(&node2)) {
                graph.add_edge(idx1, idx2, edge_data);
            }
        }
    }

    (graph, node_map)
}

/// Create a `PyMatching` decoder from a simple petgraph with just weights
///
/// This is a convenience method for graphs where edges only have weights,
/// not full `PyMatchingEdge` data.
///
/// # Arguments
/// * `graph` - The petgraph with f64 edge weights
/// * `num_observables` - Number of observables (defaults to 1 per edge)
///
/// # Errors
///
/// Returns a [`PyMatchingError`] if:
/// - The decoder builder fails
/// - Adding an edge fails
pub fn pymatching_from_petgraph_weighted(
    graph: &UnGraph<(), f64>,
    num_observables: Option<usize>,
) -> Result<PyMatchingDecoder, PyMatchingError> {
    let num_nodes = graph.node_count();
    let num_obs = num_observables.unwrap_or_else(|| graph.edge_count());

    let mut decoder = PyMatchingDecoder::builder()
        .nodes(num_nodes)
        .observables(num_obs)
        .build()?;

    // Add edges with sequential observable assignment
    for (obs_idx, edge) in graph.edge_references().enumerate() {
        let weight = *edge.weight();
        let observables = if obs_idx < num_obs {
            vec![obs_idx]
        } else {
            vec![]
        };

        decoder.add_edge(
            edge.source().index(),
            edge.target().index(),
            &observables,
            Some(weight),
            None,
            None,
        )?;
    }

    Ok(decoder)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn test_petgraph_round_trip() {
        // Create a simple decoder
        let mut decoder = PyMatchingDecoder::builder()
            .nodes(4)
            .observables(2)
            .build()
            .unwrap();

        // Add some edges
        decoder
            .add_edge(0, 1, &[0], Some(1.0), Some(0.1), None)
            .unwrap();
        decoder.add_edge(1, 2, &[1], Some(2.0), None, None).unwrap();
        decoder
            .add_edge(2, 3, &[0, 1], Some(1.5), Some(0.2), None)
            .unwrap();
        decoder
            .add_boundary_edge(0, &[], Some(3.0), None, None)
            .unwrap();

        // Convert to petgraph
        let (graph, node_map) = pymatching_to_petgraph(&decoder);

        // Verify structure
        assert_eq!(graph.node_count(), 4);
        assert_eq!(graph.edge_count(), 3); // Boundary edges not included

        // Verify node mapping
        for i in 0..4 {
            assert!(node_map.contains_key(&i));
            assert_eq!(graph[node_map[&i]].id, i);
        }

        // Create new decoder from petgraph
        let decoder2 = pymatching_from_petgraph(&graph, &HashSet::new(), 2).unwrap();

        // Verify edges exist
        assert!(decoder2.has_edge(0, 1));
        assert!(decoder2.has_edge(1, 2));
        assert!(decoder2.has_edge(2, 3));
    }

    #[test]
    fn test_from_weighted_graph() {
        use petgraph::graph::UnGraph;

        // Create a simple weighted graph
        let mut graph = UnGraph::new_undirected();
        let n0 = graph.add_node(());
        let n1 = graph.add_node(());
        let n2 = graph.add_node(());

        graph.add_edge(n0, n1, 1.0);
        graph.add_edge(n1, n2, 2.0);
        graph.add_edge(n2, n0, 3.0);

        // Convert to decoder
        let decoder = pymatching_from_petgraph_weighted(&graph, Some(3)).unwrap();

        // Verify structure
        assert_eq!(decoder.num_nodes(), 3);
        assert_eq!(decoder.num_edges(), 3);
        // PyMatching uses a minimum of 64 observables by default
        assert!(decoder.num_observables() >= 3);

        // Verify edges
        let edge_01 = decoder.get_edge_data(0, 1).unwrap();
        assert!(
            (edge_01.weight - 1.0).abs() < f64::EPSILON,
            "Expected weight 1.0, got {}",
            edge_01.weight
        );
        assert_eq!(edge_01.observables[0], 0);
    }
}
