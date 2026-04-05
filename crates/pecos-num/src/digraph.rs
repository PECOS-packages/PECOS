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

//! Directed graph implementation for PECOS.
//!
//! This module provides a directed graph data structure built on top of petgraph's
//! `StableGraph`. It follows the same API patterns as the undirected `Graph` type
//! but adds directed-specific operations like predecessors, successors, and
//! in/out degree queries.
//!
//! For directed acyclic graphs (DAGs) with cycle checking, see the [`crate::dag`] module.

use rustworkx_core::petgraph::Directed;
use rustworkx_core::petgraph::algo;
use rustworkx_core::petgraph::stable_graph::StableGraph;
use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences};

use std::collections::BTreeMap;

// Re-use attribute types from graph module
use crate::graph::{Attribute, EdgeAttrs, EdgeData, GraphAttrs, NodeAttrs};

/// A directed graph data structure for PECOS.
///
/// This is a wrapper around petgraph's `StableGraph<Directed>` that provides
/// a convenient API for PECOS use cases. Unlike the undirected `Graph` type,
/// `DiGraph` has distinct notions of predecessors/successors and in/out edges.
///
/// The graph uses `StableGraph` internally, which means node and edge indices
/// remain stable after removals (indices are not reused).
///
/// # Examples
///
/// ```
/// use pecos_num::digraph::DiGraph;
///
/// let mut graph = DiGraph::new();
/// let a = graph.add_node();
/// let b = graph.add_node();
/// let c = graph.add_node();
///
/// graph.add_edge(a, b).weight(1.0);
/// graph.add_edge(b, c).weight(2.0);
///
/// assert_eq!(graph.successors(a), vec![b]);
/// assert_eq!(graph.predecessors(c), vec![b]);
/// assert_eq!(graph.out_degree(a), 1);
/// assert_eq!(graph.in_degree(c), 1);
/// ```
#[derive(Debug, Clone)]
pub struct DiGraph {
    /// The underlying petgraph stable directed graph.
    graph: StableGraph<NodeAttrs, EdgeData, Directed>,
    /// Graph-level metadata and attributes.
    graph_data: GraphAttrs,
}

/// Builder for configuring directed edge attributes using a fluent interface.
///
/// This builder is returned by `DiGraph::add_edge()` and allows setting edge weight
/// and attributes via method chaining. The edge is automatically added to the graph
/// when the builder is dropped.
pub struct DiEdgeBuilder<'a> {
    graph: &'a mut DiGraph,
    source: usize,
    target: usize,
    weight: f64,
    attrs: BTreeMap<String, Attribute>,
}

#[allow(clippy::return_self_not_must_use, clippy::must_use_candidate)] // builder uses Drop to commit
impl DiEdgeBuilder<'_> {
    /// Sets the weight of the edge.
    /// The edge is added to the graph when the builder is dropped.
    pub fn weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Adds a single attribute to the edge.
    /// The edge is added to the graph when the builder is dropped.
    pub fn add_attr(mut self, key: impl Into<String>, value: Attribute) -> Self {
        self.attrs.insert(key.into(), value);
        self
    }

    /// Adds multiple attributes to the edge at once.
    /// The edge is added to the graph when the builder is dropped.
    pub fn add_attrs(mut self, attrs: BTreeMap<String, Attribute>) -> Self {
        self.attrs.extend(attrs);
        self
    }
}

impl Drop for DiEdgeBuilder<'_> {
    fn drop(&mut self) {
        use rustworkx_core::petgraph::graph::NodeIndex;
        let source = NodeIndex::new(self.source);
        let target = NodeIndex::new(self.target);
        let edge_data = (self.weight, self.attrs.clone());
        self.graph.graph.add_edge(source, target, edge_data);
    }
}

impl DiGraph {
    /// Creates a new empty directed graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: StableGraph::new(),
            graph_data: GraphAttrs::new(),
        }
    }

    /// Creates a new directed graph with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Expected number of nodes
    /// * `edges` - Expected number of edges
    #[must_use]
    pub fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            graph: StableGraph::with_capacity(nodes, edges),
            graph_data: GraphAttrs::new(),
        }
    }

    /// Adds a new node to the graph with empty data.
    ///
    /// Returns the index of the newly created node.
    pub fn add_node(&mut self) -> usize {
        self.graph.add_node(NodeAttrs::new()).index()
    }

    /// Adds a node with pre-built `NodeAttrs`.
    ///
    /// # Arguments
    ///
    /// * `data` - Pre-built `NodeAttrs` with attributes
    ///
    /// # Returns
    ///
    /// The index of the newly created node.
    pub fn add_node_with_data(&mut self, data: NodeAttrs) -> usize {
        self.graph.add_node(data).index()
    }

    /// Removes a node from the graph and all edges connected to it.
    ///
    /// # Arguments
    ///
    /// * `node` - The index of the node to remove
    ///
    /// # Returns
    ///
    /// The node's data if the node existed, or `None` if not found.
    ///
    /// # Note
    ///
    /// Unlike petgraph's `Graph`, `StableGraph` does not invalidate other node
    /// indices when a node is removed.
    pub fn remove_node(&mut self, node: usize) -> Option<NodeAttrs> {
        use rustworkx_core::petgraph::graph::NodeIndex;
        self.graph.remove_node(NodeIndex::new(node))
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
    #[must_use]
    pub fn nodes(&self) -> Vec<usize> {
        self.graph
            .node_indices()
            .map(rustworkx_core::petgraph::prelude::NodeIndex::index)
            .collect()
    }

    /// Adds an edge between two nodes, returning a builder to configure attributes.
    ///
    /// # Arguments
    ///
    /// * `source` - Index of the source node
    /// * `target` - Index of the target node
    ///
    /// # Returns
    ///
    /// A `DiEdgeBuilder` for configuring edge attributes via method chaining.
    ///
    /// # Panics
    ///
    /// Panics if either node index is invalid.
    pub fn add_edge(&mut self, source: usize, target: usize) -> DiEdgeBuilder<'_> {
        DiEdgeBuilder {
            graph: self,
            source,
            target,
            weight: 1.0,
            attrs: BTreeMap::new(),
        }
    }

    /// Adds an edge with full edge data (weight and attributes).
    ///
    /// # Arguments
    ///
    /// * `source` - Index of the source node
    /// * `target` - Index of the target node
    /// * `data` - `EdgeAttrs` containing weight and attributes
    pub fn add_edge_with_data(&mut self, source: usize, target: usize, data: EdgeAttrs) {
        use rustworkx_core::petgraph::graph::NodeIndex;
        let source_node = NodeIndex::new(source);
        let target_node = NodeIndex::new(target);
        self.graph
            .add_edge(source_node, target_node, data.into_edge_data());
    }

    /// Removes an edge by its edge ID.
    ///
    /// # Returns
    ///
    /// The edge data if it existed, or `None` otherwise.
    pub fn remove_edge(&mut self, edge_id: usize) -> Option<EdgeAttrs> {
        use rustworkx_core::petgraph::graph::EdgeIndex;
        self.graph
            .remove_edge(EdgeIndex::new(edge_id))
            .map(EdgeAttrs::from)
    }

    /// Returns a list of all edges as (source, target, weight) tuples.
    #[must_use]
    pub fn edges(&self) -> Vec<(usize, usize, f64)> {
        self.graph
            .edge_references()
            .map(|e| {
                let source = e.source().index();
                let target = e.target().index();
                let weight = e.weight().0;
                (source, target, weight)
            })
            .collect()
    }

    // ==================== Directed-specific queries ====================

    /// Returns the predecessor nodes of a given node.
    ///
    /// Predecessors are nodes that have edges pointing TO this node.
    ///
    /// # Arguments
    ///
    /// * `node` - The node index
    ///
    /// # Returns
    ///
    /// A vector of predecessor node indices.
    #[must_use]
    pub fn predecessors(&self, node: usize) -> Vec<usize> {
        use rustworkx_core::petgraph::Direction;
        use rustworkx_core::petgraph::graph::NodeIndex;

        self.graph
            .neighbors_directed(NodeIndex::new(node), Direction::Incoming)
            .map(rustworkx_core::petgraph::prelude::NodeIndex::index)
            .collect()
    }

    /// Returns the successor nodes of a given node.
    ///
    /// Successors are nodes that this node has edges pointing TO.
    ///
    /// # Arguments
    ///
    /// * `node` - The node index
    ///
    /// # Returns
    ///
    /// A vector of successor node indices.
    #[must_use]
    pub fn successors(&self, node: usize) -> Vec<usize> {
        use rustworkx_core::petgraph::Direction;
        use rustworkx_core::petgraph::graph::NodeIndex;

        self.graph
            .neighbors_directed(NodeIndex::new(node), Direction::Outgoing)
            .map(rustworkx_core::petgraph::prelude::NodeIndex::index)
            .collect()
    }

    /// Returns the in-degree of a node (number of incoming edges).
    ///
    /// # Arguments
    ///
    /// * `node` - The node index
    #[must_use]
    pub fn in_degree(&self, node: usize) -> usize {
        use rustworkx_core::petgraph::Direction;
        use rustworkx_core::petgraph::graph::NodeIndex;

        self.graph
            .edges_directed(NodeIndex::new(node), Direction::Incoming)
            .count()
    }

    /// Returns the out-degree of a node (number of outgoing edges).
    ///
    /// # Arguments
    ///
    /// * `node` - The node index
    #[must_use]
    pub fn out_degree(&self, node: usize) -> usize {
        use rustworkx_core::petgraph::Direction;
        use rustworkx_core::petgraph::graph::NodeIndex;

        self.graph
            .edges_directed(NodeIndex::new(node), Direction::Outgoing)
            .count()
    }

    /// Returns the edge IDs of incoming edges to a node.
    ///
    /// # Arguments
    ///
    /// * `node` - The node index
    #[must_use]
    pub fn in_edges(&self, node: usize) -> Vec<usize> {
        use rustworkx_core::petgraph::Direction;
        use rustworkx_core::petgraph::graph::NodeIndex;

        self.graph
            .edges_directed(NodeIndex::new(node), Direction::Incoming)
            .map(|e| e.id().index())
            .collect()
    }

    /// Returns the edge IDs of outgoing edges from a node.
    ///
    /// # Arguments
    ///
    /// * `node` - The node index
    #[must_use]
    pub fn out_edges(&self, node: usize) -> Vec<usize> {
        use rustworkx_core::petgraph::Direction;
        use rustworkx_core::petgraph::graph::NodeIndex;

        self.graph
            .edges_directed(NodeIndex::new(node), Direction::Outgoing)
            .map(|e| e.id().index())
            .collect()
    }

    // ==================== Attribute access ====================

    /// Gets a reference to graph-level attributes.
    #[must_use]
    pub fn graph_data(&self) -> &GraphAttrs {
        &self.graph_data
    }

    /// Gets a mutable reference to graph-level attributes.
    pub fn graph_data_mut(&mut self) -> &mut GraphAttrs {
        &mut self.graph_data
    }

    /// Gets a reference to graph-level attributes as a `BTreeMap`.
    #[must_use]
    pub fn attrs(&self) -> &BTreeMap<String, Attribute> {
        &self.graph_data
    }

    /// Gets a mutable reference to graph-level attributes.
    pub fn attrs_mut(&mut self) -> &mut BTreeMap<String, Attribute> {
        &mut self.graph_data
    }

    /// Gets a reference to a node's attributes.
    #[must_use]
    pub fn node_attrs(&self, node: usize) -> Option<&BTreeMap<String, Attribute>> {
        use rustworkx_core::petgraph::graph::NodeIndex;
        self.graph
            .node_weight(NodeIndex::new(node))
            .map(|attrs| &**attrs)
    }

    /// Gets a mutable reference to a node's attributes.
    pub fn node_attrs_mut(&mut self, node: usize) -> Option<&mut BTreeMap<String, Attribute>> {
        use rustworkx_core::petgraph::graph::NodeIndex;
        self.graph
            .node_weight_mut(NodeIndex::new(node))
            .map(|attrs| &mut **attrs)
    }

    /// Gets a reference to an edge's attributes (not including weight).
    #[must_use]
    pub fn edge_attrs(&self, source: usize, target: usize) -> Option<&BTreeMap<String, Attribute>> {
        let edge_id = self.find_edge(source, target)?;
        self.edge_attrs_by_id(edge_id)
    }

    /// Gets a mutable reference to an edge's attributes.
    pub fn edge_attrs_mut(
        &mut self,
        source: usize,
        target: usize,
    ) -> Option<&mut BTreeMap<String, Attribute>> {
        let edge_id = self.find_edge(source, target)?;
        self.edge_attrs_by_id_mut(edge_id)
    }

    /// Gets a reference to edge attributes by edge ID.
    #[must_use]
    pub fn edge_attrs_by_id(&self, edge_id: usize) -> Option<&BTreeMap<String, Attribute>> {
        use rustworkx_core::petgraph::graph::EdgeIndex;
        self.graph
            .edge_weight(EdgeIndex::new(edge_id))
            .map(|(_, attrs)| attrs)
    }

    /// Gets a mutable reference to edge attributes by edge ID.
    pub fn edge_attrs_by_id_mut(
        &mut self,
        edge_id: usize,
    ) -> Option<&mut BTreeMap<String, Attribute>> {
        use rustworkx_core::petgraph::graph::EdgeIndex;
        self.graph
            .edge_weight_mut(EdgeIndex::new(edge_id))
            .map(|(_, attrs)| attrs)
    }

    // ==================== Edge weight access ====================

    /// Finds the edge ID between two nodes.
    ///
    /// # Note
    ///
    /// For directed graphs, this finds an edge from `source` to `target`.
    /// The reverse edge (if it exists) would have a different ID.
    #[must_use]
    pub fn find_edge(&self, source: usize, target: usize) -> Option<usize> {
        use rustworkx_core::petgraph::graph::NodeIndex;
        self.graph
            .find_edge(NodeIndex::new(source), NodeIndex::new(target))
            .map(rustworkx_core::petgraph::prelude::EdgeIndex::index)
    }

    /// Gets the endpoints of an edge by its edge ID.
    #[must_use]
    pub fn edge_endpoints(&self, edge_id: usize) -> Option<(usize, usize)> {
        use rustworkx_core::petgraph::graph::EdgeIndex;
        self.graph
            .edge_endpoints(EdgeIndex::new(edge_id))
            .map(|(s, t)| (s.index(), t.index()))
    }

    /// Gets the weight of an edge by its edge ID.
    ///
    /// # Panics
    ///
    /// Panics if the edge ID is invalid.
    #[must_use]
    pub fn edge_weight(&self, edge_id: usize) -> f64 {
        use rustworkx_core::petgraph::graph::EdgeIndex;
        self.graph
            .edge_weight(EdgeIndex::new(edge_id))
            .expect("Invalid edge ID")
            .0
    }

    /// Sets the weight of an edge by its edge ID.
    ///
    /// # Panics
    ///
    /// Panics if the edge ID is invalid.
    pub fn set_edge_weight(&mut self, edge_id: usize, weight: f64) {
        use rustworkx_core::petgraph::graph::EdgeIndex;
        self.graph
            .edge_weight_mut(EdgeIndex::new(edge_id))
            .expect("Invalid edge ID")
            .0 = weight;
    }

    /// Gets the weight of an edge between two nodes.
    #[must_use]
    pub fn get_weight(&self, source: usize, target: usize) -> Option<f64> {
        self.find_edge(source, target)
            .map(|edge_id| self.edge_weight(edge_id))
    }

    /// Sets the weight of an edge between two nodes.
    ///
    /// # Panics
    ///
    /// Panics if the edge doesn't exist.
    pub fn set_weight(&mut self, source: usize, target: usize, weight: f64) {
        let edge_id = self.find_edge(source, target).expect("Edge not found");
        self.set_edge_weight(edge_id, weight);
    }

    /// Gets the edge data between two nodes as `EdgeAttrs`.
    #[must_use]
    pub fn get_edge_data(&self, source: usize, target: usize) -> Option<EdgeAttrs> {
        use rustworkx_core::petgraph::graph::NodeIndex;
        self.graph
            .find_edge(NodeIndex::new(source), NodeIndex::new(target))
            .and_then(|edge_idx| self.graph.edge_weight(edge_idx))
            .map(EdgeAttrs::from_edge_data)
    }

    // ==================== Graph algorithms ====================

    /// Performs a topological sort of the graph.
    ///
    /// Returns `None` if the graph contains a cycle.
    ///
    /// # Returns
    ///
    /// A vector of node indices in topological order, or `None` if cyclic.
    #[must_use]
    pub fn topological_sort(&self) -> Option<Vec<usize>> {
        algo::toposort(&self.graph, None).ok().map(|nodes| {
            nodes
                .into_iter()
                .map(rustworkx_core::petgraph::prelude::NodeIndex::index)
                .collect()
        })
    }

    /// Checks if the graph is acyclic (contains no cycles).
    #[must_use]
    pub fn is_acyclic(&self) -> bool {
        !algo::is_cyclic_directed(&self.graph)
    }

    /// Checks if there is a path from `source` to `target`.
    #[must_use]
    pub fn has_path(&self, source: usize, target: usize) -> bool {
        use rustworkx_core::petgraph::graph::NodeIndex;
        algo::has_path_connecting(
            &self.graph,
            NodeIndex::new(source),
            NodeIndex::new(target),
            None,
        )
    }

    // ==================== Subgraph operations ====================

    /// Creates a subgraph containing only the specified nodes.
    ///
    /// Edges between nodes in the subgraph are preserved.
    #[must_use]
    pub fn subgraph(&self, nodes: &[usize]) -> Self {
        let mut new_graph = DiGraph::new();

        // Map old node indices to new node indices
        let mut node_map = BTreeMap::new();
        for &old_idx in nodes {
            if let Some(node_data) = self.node_attrs(old_idx) {
                let mut new_attrs = NodeAttrs::new();
                new_attrs.extend(node_data.clone());
                let new_idx = new_graph.add_node_with_data(new_attrs);
                node_map.insert(old_idx, new_idx);
            }
        }

        // Add edges between nodes in the subgraph
        for edge in self.graph.edge_references() {
            let source = edge.source().index();
            let target = edge.target().index();

            if let (Some(&new_source), Some(&new_target)) =
                (node_map.get(&source), node_map.get(&target))
            {
                let edge_data = EdgeAttrs::from_edge_data(edge.weight());
                new_graph.add_edge_with_data(new_source, new_target, edge_data);
            }
        }

        new_graph
    }

    // ==================== petgraph access ====================

    /// Provides direct access to the underlying petgraph.
    #[must_use]
    pub fn as_petgraph(&self) -> &StableGraph<NodeAttrs, EdgeData, Directed> {
        &self.graph
    }

    /// Provides mutable access to the underlying petgraph.
    pub fn as_petgraph_mut(&mut self) -> &mut StableGraph<NodeAttrs, EdgeData, Directed> {
        &mut self.graph
    }
}

impl Default for DiGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_digraph_creation() {
        let graph = DiGraph::new();
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_add_nodes() {
        let mut graph = DiGraph::new();
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
        let mut graph = DiGraph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph.add_edge(n0, n1).weight(1.0);
        graph.add_edge(n1, n2).weight(2.0);

        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_predecessors_successors() {
        let mut graph = DiGraph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph.add_edge(n0, n1);
        graph.add_edge(n0, n2);
        graph.add_edge(n1, n2);

        // n0 has no predecessors, two successors
        assert!(graph.predecessors(n0).is_empty());
        assert_eq!(graph.successors(n0).len(), 2);

        // n1 has one predecessor (n0), one successor (n2)
        assert_eq!(graph.predecessors(n1), vec![n0]);
        assert_eq!(graph.successors(n1), vec![n2]);

        // n2 has two predecessors, no successors
        assert_eq!(graph.predecessors(n2).len(), 2);
        assert!(graph.successors(n2).is_empty());
    }

    #[test]
    fn test_in_out_degree() {
        let mut graph = DiGraph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph.add_edge(n0, n1);
        graph.add_edge(n0, n2);
        graph.add_edge(n1, n2);

        assert_eq!(graph.in_degree(n0), 0);
        assert_eq!(graph.out_degree(n0), 2);

        assert_eq!(graph.in_degree(n1), 1);
        assert_eq!(graph.out_degree(n1), 1);

        assert_eq!(graph.in_degree(n2), 2);
        assert_eq!(graph.out_degree(n2), 0);
    }

    #[test]
    fn test_topological_sort_acyclic() {
        let mut graph = DiGraph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph.add_edge(n0, n1);
        graph.add_edge(n1, n2);

        let sorted = graph.topological_sort();
        assert!(sorted.is_some());

        let sorted = sorted.unwrap();
        assert_eq!(sorted.len(), 3);
        // n0 must come before n1, n1 before n2
        assert!(sorted.iter().position(|&x| x == n0) < sorted.iter().position(|&x| x == n1));
        assert!(sorted.iter().position(|&x| x == n1) < sorted.iter().position(|&x| x == n2));
    }

    #[test]
    fn test_topological_sort_cyclic() {
        let mut graph = DiGraph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph.add_edge(n0, n1);
        graph.add_edge(n1, n2);
        graph.add_edge(n2, n0); // Creates cycle

        assert!(graph.topological_sort().is_none());
        assert!(!graph.is_acyclic());
    }

    #[test]
    fn test_is_acyclic() {
        let mut graph = DiGraph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();

        graph.add_edge(n0, n1);
        assert!(graph.is_acyclic());

        graph.add_edge(n1, n0);
        assert!(!graph.is_acyclic());
    }

    #[test]
    fn test_has_path() {
        let mut graph = DiGraph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let n3 = graph.add_node(); // Disconnected

        graph.add_edge(n0, n1);
        graph.add_edge(n1, n2);

        assert!(graph.has_path(n0, n2));
        assert!(!graph.has_path(n2, n0)); // Directed: no reverse path
        assert!(!graph.has_path(n0, n3)); // Disconnected
    }

    #[test]
    fn test_edge_weight() {
        let mut graph = DiGraph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();

        graph.add_edge(n0, n1).weight(5.0);

        assert_eq!(graph.get_weight(n0, n1), Some(5.0));
        assert_eq!(graph.get_weight(n1, n0), None); // Directed edge

        graph.set_weight(n0, n1, 10.0);
        assert_eq!(graph.get_weight(n0, n1), Some(10.0));
    }

    #[test]
    fn test_node_attrs() {
        let mut graph = DiGraph::new();
        let n0 = graph.add_node();

        graph
            .node_attrs_mut(n0)
            .unwrap()
            .insert("label".to_string(), Attribute::String("start".into()));

        assert_eq!(
            graph.node_attrs(n0).unwrap().get("label"),
            Some(&Attribute::String("start".into()))
        );
    }

    #[test]
    fn test_subgraph() {
        let mut graph = DiGraph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        let n3 = graph.add_node();

        graph.add_edge(n0, n1);
        graph.add_edge(n1, n2);
        graph.add_edge(n2, n3);
        graph.add_edge(n0, n3);

        let sub = graph.subgraph(&[n1, n2]);
        assert_eq!(sub.node_count(), 2);
        assert_eq!(sub.edge_count(), 1); // Only n1 -> n2 edge
    }

    #[test]
    fn test_stable_indices_after_removal() {
        let mut graph = DiGraph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();

        graph.add_edge(n0, n1);
        graph.add_edge(n1, n2);

        // Remove middle node
        graph.remove_node(n1);

        // n0 and n2 should still be accessible by their original indices
        assert!(graph.node_attrs(n0).is_some());
        assert!(graph.node_attrs(n1).is_none()); // Removed
        assert!(graph.node_attrs(n2).is_some());

        // Node count reflects removal
        assert_eq!(graph.node_count(), 2);
    }
}
