// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Graph algorithms for PECOS quantum error correction.
//!
//! This module provides graph data structures and algorithms needed for quantum error
//! correction, particularly for the MWPM (Minimum Weight Perfect Matching) decoder.
//!
//! Built on top of rustworkx-core and petgraph, providing both Rust and Python APIs.

// Re-export petgraph from rustworkx-core to ensure version consistency
pub use rustworkx_core::petgraph;

use rustworkx_core::max_weight_matching::max_weight_matching;
use rustworkx_core::petgraph::algo::dijkstra;
use rustworkx_core::petgraph::graph::{NodeIndex, UnGraph};
use rustworkx_core::petgraph::visit::EdgeRef;
use std::collections::BTreeMap;

/// Edge attributes for graph edges.
///
/// This stores arbitrary key-value pairs for edge attributes, similar to `NetworkX`'s
/// edge data dictionaries. Common attributes include 'weight', '`syn_path`', '`data_path`'.
/// Uses `BTreeMap` for deterministic ordering.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeData {
    /// Map of attribute names to their values (`BTreeMap` ensures deterministic ordering)
    attributes: BTreeMap<String, EdgeAttribute>,
}

/// Values that can be stored as edge attributes.
#[derive(Debug, Clone, PartialEq)]
pub enum EdgeAttribute {
    /// Floating point number (commonly used for weight)
    Float(f64),
    /// Integer
    Int(i64),
    /// String
    String(String),
    /// Boolean
    Bool(bool),
    /// List of integers (e.g., for paths)
    IntList(Vec<i64>),
}

impl EdgeData {
    /// Creates a new empty `EdgeData`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            attributes: BTreeMap::new(),
        }
    }

    /// Creates `EdgeData` from a weight value only.
    #[must_use]
    pub fn from_weight(weight: f64) -> Self {
        let mut data = Self::new();
        data.set("weight", EdgeAttribute::Float(weight));
        data
    }

    /// Sets an attribute.
    pub fn set(&mut self, key: &str, value: EdgeAttribute) {
        self.attributes.insert(key.to_string(), value);
    }

    /// Gets an attribute.
    #[must_use]
    pub fn get(&self, key: &str) -> Option<&EdgeAttribute> {
        self.attributes.get(key)
    }

    /// Gets the weight attribute as f64, or returns 1.0 if not set.
    #[must_use]
    #[allow(clippy::cast_precision_loss)] // i64 to f64 conversion is acceptable for graph weights
    pub fn weight(&self) -> f64 {
        match self.get("weight") {
            Some(EdgeAttribute::Float(w)) => *w,
            Some(EdgeAttribute::Int(i)) => *i as f64,
            _ => 1.0, // Default weight
        }
    }

    /// Returns all attributes.
    #[must_use]
    pub fn attributes(&self) -> &BTreeMap<String, EdgeAttribute> {
        &self.attributes
    }
}

impl Default for EdgeData {
    fn default() -> Self {
        Self::new()
    }
}

/// A graph data structure for quantum error correction applications.
///
/// This is a thin wrapper around petgraph's `UnGraph` (undirected graph) that provides
/// a convenient API for PECOS use cases, particularly MWPM decoding.
///
/// # Examples
///
/// ```
/// use pecos_num::graph::Graph;
///
/// let mut graph = Graph::new();
/// let n0 = graph.add_node();
/// let n1 = graph.add_node();
/// graph.add_edge(n0, n1, 1.0);
/// ```
#[derive(Debug, Clone)]
pub struct Graph {
    /// The underlying petgraph graph structure.
    /// Uses () for node weights and `EdgeData` for edge attributes.
    graph: UnGraph<(), EdgeData>,
}

impl Graph {
    /// Creates a new empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: UnGraph::new_undirected(),
        }
    }

    /// Creates a new graph with pre-allocated capacity for nodes and edges.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Expected number of nodes
    /// * `edges` - Expected number of edges
    #[must_use]
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            graph: UnGraph::with_capacity(nodes, edges),
        }
    }

    /// Adds a new node to the graph.
    ///
    /// Returns the index of the newly created node.
    pub fn add_node(&mut self) -> usize {
        self.graph.add_node(()).index()
    }

    /// Adds an edge between two nodes with the specified weight.
    ///
    /// # Arguments
    ///
    /// * `a` - Index of the first node
    /// * `b` - Index of the second node
    /// * `weight` - Weight of the edge
    ///
    /// # Panics
    ///
    /// Panics if either node index is invalid.
    pub fn add_edge(&mut self, a: usize, b: usize, weight: f64) {
        let node_a = NodeIndex::new(a);
        let node_b = NodeIndex::new(b);
        let edge_data = EdgeData::from_weight(weight);
        self.graph.add_edge(node_a, node_b, edge_data);
    }

    /// Adds an edge between two nodes with full edge data attributes.
    ///
    /// # Arguments
    ///
    /// * `a` - Index of the first node
    /// * `b` - Index of the second node
    /// * `data` - `EdgeData` containing all attributes
    ///
    /// # Panics
    ///
    /// Panics if either node index is invalid.
    pub fn add_edge_with_data(&mut self, a: usize, b: usize, data: EdgeData) {
        let node_a = NodeIndex::new(a);
        let node_b = NodeIndex::new(b);
        self.graph.add_edge(node_a, node_b, data);
    }

    /// Returns the number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns the number of edges in the graph.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Returns a vector of all node indices in the graph.
    ///
    /// This is equivalent to `NetworkX`'s `graph.nodes()`.
    ///
    /// # Returns
    ///
    /// A vector containing all node indices (0 to node_count-1).
    #[must_use]
    pub fn nodes(&self) -> Vec<usize> {
        (0..self.graph.node_count()).collect()
    }

    /// Computes the maximum weight matching of the graph.
    ///
    /// This function finds a matching (set of edges with no common vertices) that
    /// maximizes the sum of edge weights. This is used in MWPM decoders for quantum
    /// error correction.
    ///
    /// # Arguments
    ///
    /// * `max_cardinality` - If true, prioritize maximum cardinality over maximum weight
    ///
    /// # Returns
    ///
    /// A `BTreeMap` mapping node indices to their matched partners. Each matched pair
    /// appears twice (once for each direction). `BTreeMap` ensures deterministic ordering.
    ///
    /// # Panics
    ///
    /// Should never panic as the weight conversion is infallible.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// let n1 = graph.add_node();
    /// let n2 = graph.add_node();
    /// let n3 = graph.add_node();
    ///
    /// graph.add_edge(n0, n1, 10.0);
    /// graph.add_edge(n2, n3, 20.0);
    ///
    /// let matching = graph.max_weight_matching(false);
    /// assert_eq!(matching.len(), 4);  // Two pairs, each appearing twice
    /// ```
    #[must_use]
    pub fn max_weight_matching(&self, max_cardinality: bool) -> BTreeMap<usize, usize> {
        self.max_weight_matching_with_precision(max_cardinality, 1000.0)
    }

    /// Compute maximum weight perfect matching with configurable weight precision.
    ///
    /// This is the same as `max_weight_matching` but allows you to control the
    /// float-to-integer conversion multiplier.
    ///
    /// # Arguments
    ///
    /// * `max_cardinality` - If true, compute maximum cardinality matching with maximum weight
    /// * `weight_multiplier` - Multiplier for converting float weights to integers
    ///
    /// # Returns
    ///
    /// A `BTreeMap` mapping node indices to their matched partners.
    ///
    /// # Weight Multiplier Guidelines
    ///
    /// The matching algorithm internally uses integer weights. Floating-point weights are
    /// converted by multiplying by `weight_multiplier` and casting to `i128`.
    ///
    /// **Common values:**
    /// - `1000.0` (default): Preserves 3 decimal places, good for most use cases
    /// - `1.0`: Use when weights are already integers to avoid unnecessary scaling
    /// - `10000.0` or higher: Use when you need to preserve more decimal precision
    ///
    /// **When to adjust:**
    /// - If weights are integers (e.g., -5, -10, -15), use `1.0`
    /// - If weights have many decimal places (e.g., 0.0001 differences), increase multiplier
    /// - If weights span a large range, ensure `weight * multiplier` fits in `i128`
    ///
    /// # Panics
    ///
    /// Should never panic as the weight conversion is infallible.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// let n1 = graph.add_node();
    /// let n2 = graph.add_node();
    /// let n3 = graph.add_node();
    ///
    /// // Integer weights - use multiplier of 1.0
    /// graph.add_edge(n0, n1, -5.0);
    /// graph.add_edge(n2, n3, -10.0);
    ///
    /// let matching = graph.max_weight_matching_with_precision(false, 1.0);
    /// assert_eq!(matching.len(), 4);
    /// ```
    #[must_use]
    pub fn max_weight_matching_with_precision(
        &self,
        max_cardinality: bool,
        weight_multiplier: f64,
    ) -> BTreeMap<usize, usize> {
        // Convert f64 weights to i128 by scaling with the provided multiplier
        // The algorithm expects i128 weights and returns Result<i128, E>
        let matching = max_weight_matching(
            &self.graph,
            max_cardinality,
            |e| {
                let weight = e.weight().weight(); // Get weight from EdgeData
                #[allow(clippy::cast_possible_truncation)]
                // Truncation is acceptable for graph weights
                Ok::<i128, std::convert::Infallible>((weight * weight_multiplier) as i128)
            },
            false, // verify_optimum_flag - set to false for performance
        )
        .expect("Infallible conversion should never fail");

        // Convert HashSet<(usize, usize)> to BTreeMap<usize, usize>
        // The matching set contains pairs (a, b) where a < b
        // We return a BTreeMap with both (a, b) and (b, a) for convenience
        // BTreeMap ensures deterministic ordering (important for PECOS)
        matching
            .iter()
            .flat_map(|&(a, b)| [(a, b), (b, a)])
            .collect()
    }

    /// Returns a list of all edges as (source, target, weight) tuples.
    ///
    /// Useful for inspecting the graph structure or converting to other formats.
    #[must_use]
    pub fn edges(&self) -> Vec<(usize, usize, f64)> {
        self.graph
            .edge_references()
            .map(|e| {
                let source = e.source().index();
                let target = e.target().index();
                let weight = e.weight().weight();
                (source, target, weight)
            })
            .collect()
    }

    /// Gets the edge data between two nodes.
    ///
    /// # Arguments
    ///
    /// * `a` - Index of the first node
    /// * `b` - Index of the second node
    ///
    /// # Returns
    ///
    /// A reference to the `EdgeData` if an edge exists, None otherwise.
    #[must_use]
    pub fn get_edge_data(&self, a: usize, b: usize) -> Option<&EdgeData> {
        let node_a = NodeIndex::new(a);
        let node_b = NodeIndex::new(b);

        // Find the edge between the two nodes
        self.graph
            .find_edge(node_a, node_b)
            .and_then(|edge_idx| self.graph.edge_weight(edge_idx))
    }

    /// Creates a subgraph containing only the specified nodes.
    ///
    /// # Arguments
    ///
    /// * `nodes` - A slice of node indices to include in the subgraph
    ///
    /// # Returns
    ///
    /// A new Graph containing only the specified nodes and edges between them.
    #[must_use]
    pub fn subgraph(&self, nodes: &[usize]) -> Self {
        let mut new_graph = Graph::new();

        // Map old node indices to new node indices (BTreeMap for deterministic ordering)
        let mut node_map = BTreeMap::new();
        for &old_idx in nodes {
            let new_idx = new_graph.add_node();
            node_map.insert(old_idx, new_idx);
        }

        // Add edges between nodes that are both in the subgraph
        for edge in self.graph.edge_references() {
            let source = edge.source().index();
            let target = edge.target().index();

            if let (Some(&new_source), Some(&new_target)) =
                (node_map.get(&source), node_map.get(&target))
            {
                let edge_data = edge.weight().clone();
                new_graph.add_edge_with_data(new_source, new_target, edge_data);
            }
        }

        new_graph
    }

    /// Computes single-source shortest paths using Dijkstra's algorithm.
    ///
    /// # Arguments
    ///
    /// * `source` - The source node index
    ///
    /// # Returns
    ///
    /// A `BTreeMap` mapping each reachable node to a vector of node indices representing
    /// the shortest path from the source to that node.
    ///
    /// # Panics
    ///
    /// Panics if the source node does not exist in the graph.
    #[must_use]
    pub fn single_source_shortest_path(&self, source: usize) -> BTreeMap<usize, Vec<usize>> {
        use std::collections::BTreeSet;

        let source_node = NodeIndex::new(source);

        // Use Dijkstra to get distances
        let distances = dijkstra(&self.graph, source_node, None, |e| e.weight().weight());

        // Now reconstruct paths using BFS-like approach
        let mut paths: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        paths.insert(source, vec![source]);

        // Build paths iteratively (BTreeSet for deterministic ordering)
        let mut to_visit: Vec<usize> = vec![source];
        let mut visited: BTreeSet<usize> = BTreeSet::new();
        visited.insert(source);

        while let Some(current) = to_visit.pop() {
            let current_node = NodeIndex::new(current);
            let current_path = paths
                .get(&current)
                .expect("Path for current node must exist")
                .clone();
            let current_dist = distances
                .get(&current_node)
                .copied()
                .unwrap_or(f64::INFINITY);

            // Check all neighbors
            for edge in self.graph.edges(current_node) {
                let neighbor = edge.target().index();

                if !visited.contains(&neighbor) {
                    let edge_weight = edge.weight().weight();
                    let neighbor_dist = distances
                        .get(&NodeIndex::new(neighbor))
                        .copied()
                        .unwrap_or(f64::INFINITY);

                    // Check if this edge is on a shortest path
                    if (current_dist + edge_weight - neighbor_dist).abs() < 1e-10 {
                        let mut new_path = current_path.clone();
                        new_path.push(neighbor);
                        paths.insert(neighbor, new_path);
                        to_visit.push(neighbor);
                        visited.insert(neighbor);
                    }
                }
            }
        }

        paths
    }

    /// Provides direct access to the underlying petgraph for advanced operations.
    ///
    /// This allows users to leverage the full petgraph API when needed.
    #[must_use]
    pub fn as_petgraph(&self) -> &UnGraph<(), EdgeData> {
        &self.graph
    }

    /// Provides mutable access to the underlying petgraph for advanced operations.
    pub fn as_petgraph_mut(&mut self) -> &mut UnGraph<(), EdgeData> {
        &mut self.graph
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

/// A graph with arbitrary node identifiers mapped to internal integer indices.
///
/// This wrapper around `Graph` provides NetworkX-style functionality where nodes
/// can be identified by any hashable type (strings, integers, etc.) rather than
/// just `usize` indices.
///
/// # Type Parameters
///
/// * `K` - The node identifier type (must be `Hash + Eq + Ord + Clone`)
///
/// # Examples
///
/// ```
/// use pecos_num::graph::MappedGraph;
///
/// let mut graph = MappedGraph::<String>::new();
/// graph.add_edge("v1".to_string(), "v2".to_string(), 1.0);
/// graph.add_edge("v2".to_string(), "v3".to_string(), 2.0);
/// ```
#[derive(Debug, Clone)]
pub struct MappedGraph<K: std::hash::Hash + Eq + Ord + Clone> {
    /// The underlying integer-indexed graph
    graph: Graph,
    /// Mapping from user node IDs to internal indices
    node_to_index: BTreeMap<K, usize>,
    /// Mapping from internal indices to user node IDs
    index_to_node: BTreeMap<usize, K>,
}

impl<K: std::hash::Hash + Eq + Ord + Clone> MappedGraph<K> {
    /// Creates a new empty mapped graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: Graph::new(),
            node_to_index: BTreeMap::new(),
            index_to_node: BTreeMap::new(),
        }
    }

    /// Creates a new mapped graph with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            graph: Graph::with_capacity(nodes, edges),
            node_to_index: BTreeMap::new(),
            index_to_node: BTreeMap::new(),
        }
    }

    /// Gets or creates an internal index for a node ID.
    fn get_or_create_index(&mut self, node: K) -> usize {
        if let Some(&idx) = self.node_to_index.get(&node) {
            idx
        } else {
            let idx = self.graph.add_node();
            self.node_to_index.insert(node.clone(), idx);
            self.index_to_node.insert(idx, node);
            idx
        }
    }

    /// Adds an edge between two nodes with the specified weight.
    ///
    /// If either node doesn't exist, it will be created automatically.
    pub fn add_edge(&mut self, a: K, b: K, weight: f64) {
        let idx_a = self.get_or_create_index(a);
        let idx_b = self.get_or_create_index(b);
        self.graph.add_edge(idx_a, idx_b, weight);
    }

    /// Adds an edge between two nodes with full edge data.
    pub fn add_edge_with_data(&mut self, a: K, b: K, data: EdgeData) {
        let idx_a = self.get_or_create_index(a);
        let idx_b = self.get_or_create_index(b);
        self.graph.add_edge_with_data(idx_a, idx_b, data);
    }

    /// Returns the number of nodes in the graph.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns the number of edges in the graph.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Returns a vector of all node IDs in the graph.
    #[must_use]
    pub fn nodes(&self) -> Vec<K> {
        self.index_to_node.values().cloned().collect()
    }

    /// Computes the maximum weight matching of the graph.
    ///
    /// Returns a map from node IDs to their matched partners.
    #[must_use]
    pub fn max_weight_matching(&self, max_cardinality: bool) -> BTreeMap<K, K> {
        self.max_weight_matching_with_precision(max_cardinality, 1000.0)
    }

    /// Compute maximum weight perfect matching with configurable weight precision.
    ///
    /// This is the same as `max_weight_matching` but allows you to control the
    /// float-to-integer conversion multiplier. See `Graph::max_weight_matching_with_precision`
    /// for detailed documentation on the `weight_multiplier` parameter.
    ///
    /// # Arguments
    ///
    /// * `max_cardinality` - If true, compute maximum cardinality matching with maximum weight
    /// * `weight_multiplier` - Multiplier for converting float weights to integers (default: 1000.0)
    ///
    /// # Returns
    ///
    /// A `BTreeMap` mapping node IDs to their matched partners.
    #[must_use]
    pub fn max_weight_matching_with_precision(
        &self,
        max_cardinality: bool,
        weight_multiplier: f64,
    ) -> BTreeMap<K, K> {
        let index_matching = self
            .graph
            .max_weight_matching_with_precision(max_cardinality, weight_multiplier);

        index_matching
            .iter()
            .filter_map(|(&idx_a, &idx_b)| {
                let node_a = self.index_to_node.get(&idx_a)?;
                let node_b = self.index_to_node.get(&idx_b)?;
                Some((node_a.clone(), node_b.clone()))
            })
            .collect()
    }

    /// Returns a list of all edges as (source, target, weight) tuples.
    #[must_use]
    pub fn edges(&self) -> Vec<(K, K, f64)> {
        self.graph
            .edges()
            .into_iter()
            .filter_map(|(idx_a, idx_b, weight)| {
                let node_a = self.index_to_node.get(&idx_a)?;
                let node_b = self.index_to_node.get(&idx_b)?;
                Some((node_a.clone(), node_b.clone(), weight))
            })
            .collect()
    }

    /// Gets the edge data between two nodes.
    #[must_use]
    pub fn get_edge_data(&self, a: &K, b: &K) -> Option<&EdgeData> {
        let idx_a = self.node_to_index.get(a)?;
        let idx_b = self.node_to_index.get(b)?;
        self.graph.get_edge_data(*idx_a, *idx_b)
    }

    /// Creates a subgraph containing only the specified nodes.
    #[must_use]
    pub fn subgraph(&self, nodes: &[K]) -> Self {
        // Get internal indices for requested nodes
        let indices: Vec<usize> = nodes
            .iter()
            .filter_map(|node| self.node_to_index.get(node).copied())
            .collect();

        // Create subgraph of internal graph
        let sub_graph = self.graph.subgraph(&indices);

        // Build new mappings for subgraph nodes
        let mut new_node_to_index = BTreeMap::new();
        let mut new_index_to_node = BTreeMap::new();

        for (new_idx, &old_idx) in indices.iter().enumerate() {
            if let Some(node) = self.index_to_node.get(&old_idx) {
                new_node_to_index.insert(node.clone(), new_idx);
                new_index_to_node.insert(new_idx, node.clone());
            }
        }

        Self {
            graph: sub_graph,
            node_to_index: new_node_to_index,
            index_to_node: new_index_to_node,
        }
    }

    /// Computes single-source shortest paths using Dijkstra's algorithm.
    #[must_use]
    pub fn single_source_shortest_path(&self, source: &K) -> BTreeMap<K, Vec<K>> {
        let Some(&source_idx) = self.node_to_index.get(source) else {
            return BTreeMap::new();
        };

        let index_paths = self.graph.single_source_shortest_path(source_idx);

        index_paths
            .into_iter()
            .filter_map(|(target_idx, path_indices)| {
                let target = self.index_to_node.get(&target_idx)?;
                let path: Vec<K> = path_indices
                    .iter()
                    .filter_map(|&idx| self.index_to_node.get(&idx).cloned())
                    .collect();
                Some((target.clone(), path))
            })
            .collect()
    }

    /// Provides access to the underlying integer-indexed graph.
    #[must_use]
    pub fn as_graph(&self) -> &Graph {
        &self.graph
    }

    /// Provides mutable access to the underlying graph.
    ///
    /// # Safety
    ///
    /// Modifying the underlying graph directly can invalidate the node mappings.
    /// Use with caution.
    pub fn as_graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }
}

impl<K: std::hash::Hash + Eq + Ord + Clone> Default for MappedGraph<K> {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_creation() {
        let graph = Graph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_add_nodes() {
        let mut graph = Graph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        assert_eq!(n0, 0);
        assert_eq!(n1, 1);
        assert_eq!(n2, 2);
        assert_eq!(graph.node_count(), 3);
    }

    #[test]
    fn test_add_edges() {
        let mut graph = Graph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph.add_edge(n0, n1, 1.0);
        graph.add_edge(n1, n2, 2.0);

        assert_eq!(graph.edge_count(), 2);

        let edges = graph.edges();
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn test_max_weight_matching_simple() {
        let mut graph = Graph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let n3 = graph.add_node();

        // Two separate edges with different weights
        graph.add_edge(n0, n1, 10.0);
        graph.add_edge(n2, n3, 20.0);

        let matching = graph.max_weight_matching(false);

        // Both edges should be in the matching
        assert_eq!(matching.len(), 4); // Each pair appears twice
        assert_eq!(matching.get(&n0), Some(&n1));
        assert_eq!(matching.get(&n1), Some(&n0));
        assert_eq!(matching.get(&n2), Some(&n3));
        assert_eq!(matching.get(&n3), Some(&n2));
    }

    #[test]
    fn test_max_weight_matching_choice() {
        let mut graph = Graph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        // Triangle: algorithm should choose the heaviest edge
        graph.add_edge(n0, n1, 1.0);
        graph.add_edge(n1, n2, 10.0);
        graph.add_edge(n0, n2, 2.0);

        let matching = graph.max_weight_matching(false);

        // Should match n1-n2 (weight 10) and leave n0 unmatched
        assert_eq!(matching.len(), 2);
        assert_eq!(matching.get(&n1), Some(&n2));
        assert_eq!(matching.get(&n2), Some(&n1));
    }

    #[test]
    fn test_with_capacity() {
        let graph = Graph::with_capacity(10, 20);
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_edges_list() {
        let mut graph = Graph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();

        graph.add_edge(n0, n1, 5.5);

        let edges = graph.edges();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], (n0, n1, 5.5));
    }

    #[test]
    fn test_as_petgraph() {
        let mut graph = Graph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        graph.add_edge(n0, n1, 1.0);

        let pg = graph.as_petgraph();
        assert_eq!(pg.node_count(), 2);
        assert_eq!(pg.edge_count(), 1);
    }
}
