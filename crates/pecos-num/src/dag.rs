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

//! Directed Acyclic Graph (DAG) implementation for PECOS.
//!
//! DAG data structure with runtime cycle detection.
//! Unlike [`DiGraph`](crate::digraph::DiGraph), the `DAG` type guarantees
//! that no cycles can be introduced when adding edges.
//!
//! The DAG is particularly useful for:
//! - Quantum circuit representation
//! - Dependency graphs
//! - Task scheduling
//!
//! # Example
//!
//! ```
//! use pecos_num::dag::DAG;
//!
//! let mut dag = DAG::new();
//! let a = dag.add_node();
//! let b = dag.add_node();
//! let c = dag.add_node();
//!
//! // These edges form a valid DAG
//! dag.add_edge(a, b).unwrap();
//! dag.add_edge(b, c).unwrap();
//!
//! // This would create a cycle, so it fails
//! assert!(dag.add_edge(c, a).is_err());
//!
//! // Topological sort is guaranteed to succeed
//! let sorted = dag.topological_sort();
//! assert_eq!(sorted, vec![a, b, c]);
//! ```

use rustworkx_core::petgraph;
use rustworkx_core::petgraph::Directed;
use rustworkx_core::petgraph::algo;
use rustworkx_core::petgraph::graph::NodeIndex;
use rustworkx_core::petgraph::stable_graph::StableGraph;
use rustworkx_core::petgraph::visit::{EdgeRef, IntoEdgeReferences, Visitable};

use std::collections::BTreeMap;
use std::error::Error;
use std::fmt;

// Re-use attribute types from graph module
use crate::graph::{Attribute, EdgeAttrs, EdgeData, GraphAttrs, NodeAttrs};

/// Error returned when trying to create a DAG from a graph that contains cycles.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DAGHasCycleError;

impl fmt::Display for DAGHasCycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Graph contains a cycle and cannot be converted to a DAG")
    }
}

impl Error for DAGHasCycleError {}

/// Error returned when adding an edge would create a cycle in the DAG.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DagWouldCycleError;

impl fmt::Display for DagWouldCycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Adding this edge would create a cycle in the DAG")
    }
}

impl Error for DagWouldCycleError {}

/// A Directed Acyclic Graph (DAG) with runtime cycle detection.
///
/// This type guarantees that the graph remains acyclic at all times.
/// Any attempt to add an edge that would create a cycle will fail with
/// a [`DagWouldCycleError`].
///
/// The DAG provides efficient methods for:
/// - Adding nodes and edges (with cycle checking)
/// - Topological sorting (guaranteed to succeed)
/// - Finding ancestors and descendants
/// - DAG-specific algorithms like longest path and layers
///
/// # Performance
///
/// Cycle detection uses a DFS-based approach. The `add_child` and `add_parent`
/// methods bypass cycle checking since adding a new node cannot create a cycle,
/// making them more efficient for building DAGs incrementally.
#[derive(Debug, Clone)]
pub struct DAG {
    /// The underlying petgraph stable directed graph.
    graph: StableGraph<NodeAttrs, EdgeData, Directed>,
    /// Graph-level metadata and attributes.
    graph_data: GraphAttrs,
    /// Cached DFS space for cycle detection (reused across calls).
    cycle_state:
        algo::DfsSpace<NodeIndex, <StableGraph<NodeAttrs, EdgeData, Directed> as Visitable>::Map>,
}

impl DAG {
    /// Creates a new empty DAG.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: StableGraph::new(),
            graph_data: GraphAttrs::new(),
            cycle_state: algo::DfsSpace::new(&StableGraph::<NodeAttrs, EdgeData, Directed>::new()),
        }
    }

    /// Creates a new DAG with pre-allocated capacity.
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
            cycle_state: algo::DfsSpace::new(&StableGraph::<NodeAttrs, EdgeData, Directed>::new()),
        }
    }

    /// Tries to create a DAG from a `DiGraph`.
    ///
    /// Returns an error if the graph contains cycles.
    ///
    /// # Arguments
    ///
    /// * `digraph` - The directed graph to convert
    ///
    /// # Returns
    ///
    /// `Ok(DAG)` if the graph is acyclic, `Err(DAGHasCycleError)` otherwise.
    ///
    /// # Errors
    ///
    /// Returns [`DAGHasCycleError`] if the input graph contains cycles.
    #[allow(clippy::needless_pass_by_value)]
    pub fn try_from_digraph(digraph: crate::digraph::DiGraph) -> Result<Self, DAGHasCycleError> {
        let graph = digraph.as_petgraph().clone();
        if algo::is_cyclic_directed(&graph) {
            return Err(DAGHasCycleError);
        }
        Ok(Self {
            graph,
            graph_data: digraph.graph_data().clone(),
            cycle_state: algo::DfsSpace::new(&StableGraph::<NodeAttrs, EdgeData, Directed>::new()),
        })
    }

    // ==================== Node operations ====================

    /// Adds a new node to the DAG with empty data.
    ///
    /// Returns the index of the newly created node.
    pub fn add_node(&mut self) -> usize {
        self.graph.add_node(NodeAttrs::new()).index()
    }

    /// Adds a node with pre-built `NodeAttrs`.
    pub fn add_node_with_data(&mut self, data: NodeAttrs) -> usize {
        self.graph.add_node(data).index()
    }

    /// Removes a node from the DAG and all edges connected to it.
    ///
    /// # Returns
    ///
    /// The node's data if it existed, or `None` otherwise.
    pub fn remove_node(&mut self, node: usize) -> Option<NodeAttrs> {
        self.graph.remove_node(NodeIndex::new(node))
    }

    /// Returns the number of nodes in the DAG.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Returns the number of edges in the DAG.
    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Returns a vector of all node indices in the DAG.
    #[must_use]
    pub fn nodes(&self) -> Vec<usize> {
        self.graph
            .node_indices()
            .map(rustworkx_core::petgraph::prelude::NodeIndex::index)
            .collect()
    }

    // ==================== Edge operations with cycle checking ====================

    /// Adds an edge between two nodes, checking for cycles.
    ///
    /// # Arguments
    ///
    /// * `source` - Index of the source node
    /// * `target` - Index of the target node
    ///
    /// # Returns
    ///
    /// `Ok(edge_id)` if the edge was added successfully, or
    /// `Err(DagWouldCycleError)` if adding the edge would create a cycle.
    ///
    /// # Errors
    ///
    /// Returns [`DagWouldCycleError`] if the edge would create a cycle.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_num::dag::DAG;
    ///
    /// let mut dag = DAG::new();
    /// let a = dag.add_node();
    /// let b = dag.add_node();
    ///
    /// assert!(dag.add_edge(a, b).is_ok());
    /// assert!(dag.add_edge(b, a).is_err()); // Would create cycle
    /// ```
    pub fn add_edge(&mut self, source: usize, target: usize) -> Result<usize, DagWouldCycleError> {
        self.add_edge_with_weight(source, target, 1.0)
    }

    /// Adds an edge with a specific weight, checking for cycles.
    ///
    /// # Returns
    ///
    /// `Ok(edge_id)` if successful, `Err(DagWouldCycleError)` if it would create a cycle.
    ///
    /// # Errors
    ///
    /// Returns [`DagWouldCycleError`] if the edge would create a cycle.
    pub fn add_edge_with_weight(
        &mut self,
        source: usize,
        target: usize,
        weight: f64,
    ) -> Result<usize, DagWouldCycleError> {
        let data = EdgeAttrs::with_weight(weight);
        self.add_edge_with_data(source, target, data)
    }

    /// Adds an edge with full edge data, checking for cycles.
    ///
    /// # Returns
    ///
    /// `Ok(edge_id)` if successful, `Err(DagWouldCycleError)` if it would create a cycle.
    ///
    /// # Errors
    ///
    /// Returns [`DagWouldCycleError`] if the edge would create a cycle.
    pub fn add_edge_with_data(
        &mut self,
        source: usize,
        target: usize,
        data: EdgeAttrs,
    ) -> Result<usize, DagWouldCycleError> {
        let source_node = NodeIndex::new(source);
        let target_node = NodeIndex::new(target);

        // Check if adding this edge would create a cycle.
        // A cycle would be created if there's already a path from target to source.
        if algo::has_path_connecting(
            &self.graph,
            target_node,
            source_node,
            Some(&mut self.cycle_state),
        ) {
            return Err(DagWouldCycleError);
        }

        // Also check for self-loops
        if source == target {
            return Err(DagWouldCycleError);
        }

        let edge_idx = self
            .graph
            .add_edge(source_node, target_node, data.into_edge_data());
        Ok(edge_idx.index())
    }

    /// Adds a child node with an edge from the parent, bypassing cycle check.
    ///
    /// This is more efficient than `add_node` + `add_edge` because adding a new
    /// node cannot create a cycle.
    ///
    /// # Arguments
    ///
    /// * `parent` - The parent node index
    /// * `edge_data` - Edge attributes for the new edge
    /// * `node_data` - Node attributes for the new child node
    ///
    /// # Returns
    ///
    /// A tuple of `(child_node_id, edge_id)`.
    pub fn add_child(
        &mut self,
        parent: usize,
        edge_data: EdgeAttrs,
        node_data: NodeAttrs,
    ) -> (usize, usize) {
        let child = self.graph.add_node(node_data);
        let edge = self
            .graph
            .add_edge(NodeIndex::new(parent), child, edge_data.into_edge_data());
        (child.index(), edge.index())
    }

    /// Adds a parent node with an edge to the child, bypassing cycle check.
    ///
    /// This is more efficient than `add_node` + `add_edge` because adding a new
    /// node cannot create a cycle.
    ///
    /// # Arguments
    ///
    /// * `child` - The child node index
    /// * `edge_data` - Edge attributes for the new edge
    /// * `node_data` - Node attributes for the new parent node
    ///
    /// # Returns
    ///
    /// A tuple of `(parent_node_id, edge_id)`.
    pub fn add_parent(
        &mut self,
        child: usize,
        edge_data: EdgeAttrs,
        node_data: NodeAttrs,
    ) -> (usize, usize) {
        let parent = self.graph.add_node(node_data);
        let edge = self
            .graph
            .add_edge(parent, NodeIndex::new(child), edge_data.into_edge_data());
        (parent.index(), edge.index())
    }

    /// Removes an edge by its edge ID.
    pub fn remove_edge(&mut self, edge_id: usize) -> Option<EdgeAttrs> {
        use rustworkx_core::petgraph::graph::EdgeIndex;
        self.graph
            .remove_edge(EdgeIndex::new(edge_id))
            .map(EdgeAttrs::from)
    }

    /// Returns all edges as (source, target, weight) tuples.
    #[must_use]
    pub fn edges(&self) -> Vec<(usize, usize, f64)> {
        self.graph
            .edge_references()
            .map(|e| (e.source().index(), e.target().index(), e.weight().0))
            .collect()
    }

    // ==================== Directed graph queries ====================

    /// Returns the predecessor nodes of a given node.
    #[must_use]
    pub fn predecessors(&self, node: usize) -> Vec<usize> {
        use rustworkx_core::petgraph::Direction;
        self.graph
            .neighbors_directed(NodeIndex::new(node), Direction::Incoming)
            .map(rustworkx_core::petgraph::prelude::NodeIndex::index)
            .collect()
    }

    /// Returns the successor nodes of a given node.
    #[must_use]
    pub fn successors(&self, node: usize) -> Vec<usize> {
        use rustworkx_core::petgraph::Direction;
        self.graph
            .neighbors_directed(NodeIndex::new(node), Direction::Outgoing)
            .map(rustworkx_core::petgraph::prelude::NodeIndex::index)
            .collect()
    }

    /// Returns the in-degree of a node.
    #[must_use]
    pub fn in_degree(&self, node: usize) -> usize {
        use rustworkx_core::petgraph::Direction;
        self.graph
            .edges_directed(NodeIndex::new(node), Direction::Incoming)
            .count()
    }

    /// Returns the out-degree of a node.
    #[must_use]
    pub fn out_degree(&self, node: usize) -> usize {
        use rustworkx_core::petgraph::Direction;
        self.graph
            .edges_directed(NodeIndex::new(node), Direction::Outgoing)
            .count()
    }

    /// Returns the edge IDs of incoming edges to a node.
    #[must_use]
    pub fn in_edges(&self, node: usize) -> Vec<usize> {
        use rustworkx_core::petgraph::Direction;
        self.graph
            .edges_directed(NodeIndex::new(node), Direction::Incoming)
            .map(|e| e.id().index())
            .collect()
    }

    /// Returns the edge IDs of outgoing edges from a node.
    #[must_use]
    pub fn out_edges(&self, node: usize) -> Vec<usize> {
        use rustworkx_core::petgraph::Direction;
        self.graph
            .edges_directed(NodeIndex::new(node), Direction::Outgoing)
            .map(|e| e.id().index())
            .collect()
    }

    // ==================== DAG-specific algorithms ====================

    /// Performs a topological sort of the DAG.
    ///
    /// Unlike `DiGraph::topological_sort()`, this is guaranteed to succeed
    /// since the DAG cannot contain cycles.
    ///
    /// # Returns
    ///
    /// A vector of node indices in topological order.
    ///
    /// # Panics
    ///
    /// This method will not panic because the DAG invariant guarantees no cycles.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn topological_sort(&self) -> Vec<usize> {
        algo::toposort(&self.graph, None)
            .expect("DAG should never have cycles")
            .into_iter()
            .map(rustworkx_core::petgraph::prelude::NodeIndex::index)
            .collect()
    }

    /// Performs a lexicographical topological sort using a key function.
    ///
    /// When multiple nodes have no remaining dependencies, they are ordered
    /// by the key function.
    ///
    /// # Arguments
    ///
    /// * `key` - A function that returns a sort key for each node index
    ///
    /// # Returns
    ///
    /// A vector of node indices in lexicographical topological order.
    ///
    /// # Panics
    ///
    /// This method will not panic because the DAG invariant guarantees no cycles.
    #[allow(clippy::missing_panics_doc)]
    pub fn lexicographical_topological_sort<F, K>(&self, mut key: F) -> Vec<usize>
    where
        F: FnMut(usize) -> K,
        K: Ord,
    {
        use rustworkx_core::dag_algo::lexicographical_topological_sort;
        use std::convert::Infallible;

        let key_fn = |node: NodeIndex| -> Result<K, Infallible> { Ok(key(node.index())) };

        lexicographical_topological_sort(&self.graph, key_fn, false, None)
            .expect("DAG should never have cycles")
            .into_iter()
            .map(rustworkx_core::petgraph::prelude::NodeIndex::index)
            .collect()
    }

    /// Finds the longest path in the DAG by edge weight.
    ///
    /// # Returns
    ///
    /// A tuple of `(path, total_weight)` where `path` is the sequence of node
    /// indices and `total_weight` is the sum of edge weights along the path.
    /// Returns `(vec![], 0.0)` for an empty graph.
    #[must_use]
    pub fn longest_path(&self) -> (Vec<usize>, f64) {
        use rustworkx_core::dag_algo::longest_path;

        let weight_fn = |edge: petgraph::stable_graph::EdgeReference<'_, EdgeData>| {
            Ok::<f64, std::convert::Infallible>(edge.weight().0)
        };

        match longest_path(&self.graph, weight_fn) {
            Ok(Some((path, weight))) => (
                path.into_iter()
                    .map(rustworkx_core::petgraph::prelude::NodeIndex::index)
                    .collect(),
                weight,
            ),
            Ok(None) | Err(_) => (vec![], 0.0),
        }
    }

    /// Returns an iterator over the layers of the DAG.
    ///
    /// A layer is a set of nodes that can be processed in parallel (all their
    /// dependencies are in previous layers).
    ///
    /// # Arguments
    ///
    /// * `first_layer` - The starting nodes (typically nodes with in-degree 0)
    ///
    /// # Returns
    ///
    /// An iterator yielding vectors of node indices for each layer.
    pub fn layers(&self, first_layer: Vec<usize>) -> impl Iterator<Item = Vec<usize>> + '_ {
        use rustworkx_core::dag_algo::layers;

        let first: Vec<NodeIndex> = first_layer.into_iter().map(NodeIndex::new).collect();

        layers(&self.graph, first)
            .filter_map(std::result::Result::ok)
            .map(|layer| {
                layer
                    .into_iter()
                    .map(rustworkx_core::petgraph::prelude::NodeIndex::index)
                    .collect()
            })
    }

    /// Returns nodes with in-degree 0 (source nodes / roots).
    #[must_use]
    pub fn roots(&self) -> Vec<usize> {
        self.nodes()
            .into_iter()
            .filter(|&n| self.in_degree(n) == 0)
            .collect()
    }

    /// Returns nodes with out-degree 0 (sink nodes / leaves).
    #[must_use]
    pub fn leaves(&self) -> Vec<usize> {
        self.nodes()
            .into_iter()
            .filter(|&n| self.out_degree(n) == 0)
            .collect()
    }

    /// Returns all ancestors of a node (transitive predecessors).
    ///
    /// The result does not include the node itself.
    #[must_use]
    pub fn ancestors(&self, node: usize) -> Vec<usize> {
        use rustworkx_core::petgraph::visit::Bfs;

        let mut ancestors = Vec::new();
        let mut bfs = Bfs::new(petgraph::visit::Reversed(&self.graph), NodeIndex::new(node));

        // Skip the starting node
        bfs.next(petgraph::visit::Reversed(&self.graph));

        while let Some(n) = bfs.next(petgraph::visit::Reversed(&self.graph)) {
            ancestors.push(n.index());
        }

        ancestors
    }

    /// Returns all descendants of a node (transitive successors).
    ///
    /// The result does not include the node itself.
    #[must_use]
    pub fn descendants(&self, node: usize) -> Vec<usize> {
        use rustworkx_core::petgraph::visit::Bfs;

        let mut descendants = Vec::new();
        let mut bfs = Bfs::new(&self.graph, NodeIndex::new(node));

        // Skip the starting node
        bfs.next(&self.graph);

        while let Some(n) = bfs.next(&self.graph) {
            descendants.push(n.index());
        }

        descendants
    }

    /// Returns the depth of the DAG (length of longest path from any root to any leaf).
    #[must_use]
    pub fn depth(&self) -> usize {
        let (path, _) = self.longest_path();
        if path.is_empty() {
            0
        } else {
            path.len() - 1 // Depth is number of edges, not nodes
        }
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
        self.graph
            .node_weight(NodeIndex::new(node))
            .map(|attrs| &**attrs)
    }

    /// Gets a mutable reference to a node's attributes.
    pub fn node_attrs_mut(&mut self, node: usize) -> Option<&mut BTreeMap<String, Attribute>> {
        self.graph
            .node_weight_mut(NodeIndex::new(node))
            .map(|attrs| &mut **attrs)
    }

    /// Gets a reference to an edge's attributes.
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

    /// Gets the edge data (weight and attributes) between two nodes.
    ///
    /// Returns `None` if no edge exists between the nodes.
    #[must_use]
    pub fn get_edge_data(&self, source: usize, target: usize) -> Option<EdgeAttrs> {
        self.graph
            .find_edge(NodeIndex::new(source), NodeIndex::new(target))
            .and_then(|edge_idx| self.graph.edge_weight(edge_idx))
            .map(EdgeAttrs::from_edge_data)
    }

    // ==================== Edge weight access ====================

    /// Finds the edge ID between two nodes.
    #[must_use]
    pub fn find_edge(&self, source: usize, target: usize) -> Option<usize> {
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
    /// Panics if no edge exists between the given nodes.
    pub fn set_weight(&mut self, source: usize, target: usize, weight: f64) {
        let edge_id = self.find_edge(source, target).expect("Edge not found");
        self.set_edge_weight(edge_id, weight);
    }

    // ==================== Path queries ====================

    /// Returns true if there is a path from source to target.
    #[must_use]
    pub fn has_path(&self, source: usize, target: usize) -> bool {
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
    /// The resulting subgraph is still a valid DAG.
    #[must_use]
    #[allow(clippy::missing_panics_doc)]
    pub fn subgraph(&self, nodes: &[usize]) -> Self {
        use rustworkx_core::petgraph::visit::EdgeRef;
        let mut new_dag = DAG::new();

        // Map old node indices to new node indices
        let mut node_map = BTreeMap::new();
        for &old_idx in nodes {
            if let Some(node_data) = self.node_attrs(old_idx) {
                let mut new_attrs = NodeAttrs::new();
                new_attrs.extend(node_data.clone());
                let new_idx = new_dag.add_node_with_data(new_attrs);
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
                let (weight, attrs) = edge.weight();
                let edge_attrs = EdgeAttrs::from_edge_data(&(*weight, attrs.clone()));
                new_dag
                    .add_edge_with_data(new_source, new_target, edge_attrs)
                    .expect("subgraph edges cannot create cycles");
            }
        }

        new_dag
    }

    // ==================== petgraph access ====================

    /// Provides direct access to the underlying petgraph.
    #[must_use]
    pub fn as_petgraph(&self) -> &StableGraph<NodeAttrs, EdgeData, Directed> {
        &self.graph
    }

    /// Provides mutable access to the underlying petgraph.
    ///
    /// # Warning
    ///
    /// Modifying the graph directly can introduce cycles, breaking the DAG invariant.
    /// Use with caution.
    pub fn as_petgraph_mut(&mut self) -> &mut StableGraph<NodeAttrs, EdgeData, Directed> {
        &mut self.graph
    }
}

impl Default for DAG {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    #[test]
    fn test_dag_creation() {
        let dag = DAG::new();
        assert_eq!(dag.node_count(), 0);
        assert_eq!(dag.edge_count(), 0);
    }

    #[test]
    fn test_add_valid_edges() {
        let mut dag = DAG::new();
        let a = dag.add_node();
        let b = dag.add_node();
        let c = dag.add_node();

        assert!(dag.add_edge(a, b).is_ok());
        assert!(dag.add_edge(b, c).is_ok());
        assert!(dag.add_edge(a, c).is_ok()); // Skip edge, still valid

        assert_eq!(dag.edge_count(), 3);
    }

    #[test]
    fn test_reject_cycle() {
        let mut dag = DAG::new();
        let a = dag.add_node();
        let b = dag.add_node();
        let c = dag.add_node();

        dag.add_edge(a, b).unwrap();
        dag.add_edge(b, c).unwrap();

        // This would create a cycle: c -> a -> b -> c
        assert!(dag.add_edge(c, a).is_err());
        assert_eq!(dag.edge_count(), 2); // Edge was not added
    }

    #[test]
    fn test_reject_self_loop() {
        let mut dag = DAG::new();
        let a = dag.add_node();

        assert!(dag.add_edge(a, a).is_err());
    }

    #[test]
    fn test_topological_sort() {
        let mut dag = DAG::new();
        let a = dag.add_node();
        let b = dag.add_node();
        let c = dag.add_node();

        dag.add_edge(a, b).unwrap();
        dag.add_edge(b, c).unwrap();

        let sorted = dag.topological_sort();
        assert_eq!(sorted, vec![a, b, c]);
    }

    #[test]
    fn test_add_child() {
        let mut dag = DAG::new();
        let parent = dag.add_node();

        let (child, edge) = dag.add_child(parent, EdgeAttrs::with_weight(5.0), NodeAttrs::new());

        assert_eq!(dag.node_count(), 2);
        assert_eq!(dag.edge_count(), 1);
        assert_eq!(dag.edge_weight(edge), 5.0);
        assert_eq!(dag.successors(parent), vec![child]);
    }

    #[test]
    fn test_add_parent() {
        let mut dag = DAG::new();
        let child = dag.add_node();

        let (parent, edge) = dag.add_parent(child, EdgeAttrs::with_weight(3.0), NodeAttrs::new());

        assert_eq!(dag.node_count(), 2);
        assert_eq!(dag.edge_count(), 1);
        assert_eq!(dag.edge_weight(edge), 3.0);
        assert_eq!(dag.predecessors(child), vec![parent]);
    }

    #[test]
    fn test_roots_and_leaves() {
        let mut dag = DAG::new();
        let a = dag.add_node();
        let b = dag.add_node();
        let c = dag.add_node();
        let d = dag.add_node();

        dag.add_edge(a, c).unwrap();
        dag.add_edge(b, c).unwrap();
        dag.add_edge(c, d).unwrap();

        let mut roots = dag.roots();
        roots.sort_unstable();
        assert_eq!(roots, vec![a, b]);

        assert_eq!(dag.leaves(), vec![d]);
    }

    #[test]
    fn test_ancestors_descendants() {
        let mut dag = DAG::new();
        let a = dag.add_node();
        let b = dag.add_node();
        let c = dag.add_node();
        let d = dag.add_node();

        dag.add_edge(a, b).unwrap();
        dag.add_edge(b, c).unwrap();
        dag.add_edge(c, d).unwrap();

        // Ancestors of d: c, b, a
        let ancestors = dag.ancestors(d);
        assert_eq!(ancestors.len(), 3);
        assert!(ancestors.contains(&a));
        assert!(ancestors.contains(&b));
        assert!(ancestors.contains(&c));

        // Descendants of a: b, c, d
        let descendants = dag.descendants(a);
        assert_eq!(descendants.len(), 3);
        assert!(descendants.contains(&b));
        assert!(descendants.contains(&c));
        assert!(descendants.contains(&d));
    }

    #[test]
    fn test_longest_path() {
        let mut dag = DAG::new();
        let a = dag.add_node();
        let b = dag.add_node();
        let c = dag.add_node();
        let d = dag.add_node();

        dag.add_edge_with_weight(a, b, 1.0).unwrap();
        dag.add_edge_with_weight(b, c, 2.0).unwrap();
        dag.add_edge_with_weight(a, d, 10.0).unwrap(); // Shorter path, higher weight

        let (path, weight) = dag.longest_path();
        assert_eq!(path, vec![a, d]);
        assert_eq!(weight, 10.0);
    }

    #[test]
    fn test_layers() {
        let mut dag = DAG::new();
        let a = dag.add_node();
        let b = dag.add_node();
        let c = dag.add_node();
        let d = dag.add_node();

        dag.add_edge(a, c).unwrap();
        dag.add_edge(b, c).unwrap();
        dag.add_edge(c, d).unwrap();

        let roots = dag.roots();
        let layers: Vec<Vec<usize>> = dag.layers(roots).collect();

        assert_eq!(layers.len(), 3);
        // First layer: roots (a, b)
        assert!(layers[0].contains(&a));
        assert!(layers[0].contains(&b));
        // Second layer: c
        assert_eq!(layers[1], vec![c]);
        // Third layer: d
        assert_eq!(layers[2], vec![d]);
    }

    #[test]
    fn test_depth() {
        let mut dag = DAG::new();
        let a = dag.add_node();
        let b = dag.add_node();
        let c = dag.add_node();

        dag.add_edge(a, b).unwrap();
        dag.add_edge(b, c).unwrap();

        assert_eq!(dag.depth(), 2); // Two edges: a->b, b->c
    }

    #[test]
    fn test_lexicographical_topological_sort() {
        let mut dag = DAG::new();
        let a = dag.add_node();
        let b = dag.add_node();
        let c = dag.add_node();

        // a and b have no dependencies, c depends on both
        dag.add_edge(a, c).unwrap();
        dag.add_edge(b, c).unwrap();

        // With key that prefers lower indices
        let sorted = dag.lexicographical_topological_sort(|n| n);
        assert_eq!(sorted[0], a); // a comes before b
        assert_eq!(sorted[1], b);
        assert_eq!(sorted[2], c);

        // With key that prefers higher indices
        let sorted = dag.lexicographical_topological_sort(std::cmp::Reverse);
        assert_eq!(sorted[0], b); // b comes before a
        assert_eq!(sorted[1], a);
        assert_eq!(sorted[2], c);
    }

    #[test]
    fn test_try_from_digraph() {
        use crate::digraph::DiGraph;

        // Acyclic graph should succeed
        let mut digraph = DiGraph::new();
        let a = digraph.add_node();
        let b = digraph.add_node();
        digraph.add_edge(a, b);

        let dag = DAG::try_from_digraph(digraph);
        assert!(dag.is_ok());

        // Cyclic graph should fail
        let mut cyclic = DiGraph::new();
        let x = cyclic.add_node();
        let y = cyclic.add_node();
        cyclic.add_edge(x, y);
        cyclic.add_edge(y, x);

        let result = DAG::try_from_digraph(cyclic);
        assert!(result.is_err());
    }
}
