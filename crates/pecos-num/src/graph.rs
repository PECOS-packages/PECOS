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
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Attribute value type used for both nodes and edges.
///
/// This enum provides two paths for attribute storage:
/// 1. **Fast path**: Native types (Float, Int, String, Bool, `IntList`, `StringList`) for common use cases
/// 2. **Flexible path**: Json variant for arbitrary/heterogeneous data structures
///
/// The Json variant enables Python-like flexibility for complex attributes like `[1, "v10"]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Attribute {
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
    /// List of strings (e.g., for `syn_path` with string node IDs)
    StringList(Vec<String>),
    /// Arbitrary JSON value (for heterogeneous lists, nested structures, etc.)
    ///
    /// This variant stores any JSON-compatible data via `serde_json::Value`.
    /// Use this for:
    /// - Heterogeneous lists like `[1, "v10"]`
    /// - Nested structures like `{"foo": [1, 2, 3]}`
    /// - Arbitrary Python objects (converted via pythonize)
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::Attribute;
    /// use serde_json::json;
    ///
    /// let attr = Attribute::Json(json!([1, "v10"]));  // Heterogeneous list
    /// let attr2 = Attribute::Json(json!({"foo": "bar"}));  // Object
    /// ```
    Json(serde_json::Value),
}

// Type alias for backward compatibility
pub type EdgeAttribute = Attribute;

/// Core attribute storage for graphs, nodes, and edges.
///
/// This is the base type that stores arbitrary key-value pairs as a `BTreeMap`.
/// Wrapped by `GraphAttrs` and `NodeAttrs` for type safety.
/// These types Deref to `BTreeMap<String, Attribute>` for direct map operations.
///
/// # Examples
///
/// ```
/// use pecos_num::graph::{Attrs, Attribute};
///
/// let mut attrs = Attrs::new();
/// attrs.insert("x".to_string(), Attribute::Float(1.0));
/// attrs.insert("y".to_string(), Attribute::Float(2.0));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Attrs {
    /// Map of attribute names to their values (`BTreeMap` ensures deterministic ordering)
    map: BTreeMap<String, Attribute>,
}

impl Attrs {
    /// Creates a new empty `Attrs`.
    #[must_use]
    pub fn new() -> Self {
        Self {
            map: BTreeMap::new(),
        }
    }
}

impl Default for Attrs {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for Attrs {
    type Target = BTreeMap<String, Attribute>;

    fn deref(&self) -> &Self::Target {
        &self.map
    }
}

impl std::ops::DerefMut for Attrs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.map
    }
}

/// Node attributes - thin wrapper around `Attrs` for type safety.
///
/// Stores arbitrary key-value pairs for node attributes, similar to `NetworkX`'s node data.
/// Derefs to `BTreeMap<String, Attribute>` for direct map operations.
///
/// # Examples
///
/// ```
/// use pecos_num::graph::{NodeAttrs, Attribute};
///
/// let mut attrs = NodeAttrs::new();
/// attrs.insert("x".to_string(), Attribute::Float(1.0));
/// attrs.insert("y".to_string(), Attribute::Float(2.0));
/// attrs.insert("qubit_type".to_string(), Attribute::String("data".into()));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct NodeAttrs(Attrs);

impl NodeAttrs {
    /// Creates a new empty `NodeAttrs`.
    #[must_use]
    pub fn new() -> Self {
        Self(Attrs::new())
    }
}

impl Default for NodeAttrs {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for NodeAttrs {
    type Target = BTreeMap<String, Attribute>;

    fn deref(&self) -> &Self::Target {
        &self.0.map
    }
}

impl std::ops::DerefMut for NodeAttrs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.map
    }
}

/// Edge data combining weight and attributes.
///
/// In petgraph/rustworkx-core, the edge weight type parameter represents the primary edge data.
/// We use a tuple `(f64, BTreeMap<String, Attribute>)` where:
/// - `.0` is the edge weight (default 1.0)
/// - `.1` is a map of arbitrary attributes
///
/// This design treats weight as a first-class value separate from other attributes,
/// aligning with rustworkx-core conventions while maintaining rich attribute support.
pub type EdgeData = (f64, BTreeMap<String, Attribute>);

/// Edge attributes - wrapper providing convenient access to edge weight and attributes.
///
/// This type wraps `EdgeData` to provide builder-style methods for constructing edges.
/// Similar to `NetworkX`'s edge data dictionaries.
///
/// # Examples
///
/// ```
/// use pecos_num::graph::{EdgeAttrs, Attribute};
///
/// let attrs = EdgeAttrs::with_weight(5.0)
///     .attr("label", Attribute::String("boundary".into()))
///     .attr("syn_path", Attribute::IntList(vec![1, 2, 3]));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EdgeAttrs {
    weight: f64,
    attrs: BTreeMap<String, Attribute>,
}

impl EdgeAttrs {
    /// Creates a new `EdgeAttrs` with default weight (1.0) and empty attributes.
    #[must_use]
    pub fn new() -> Self {
        Self {
            weight: 1.0,
            attrs: BTreeMap::new(),
        }
    }

    /// Creates `EdgeAttrs` with specified weight and empty attributes.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::EdgeAttrs;
    ///
    /// let attrs = EdgeAttrs::with_weight(5.0);
    /// assert_eq!(attrs.weight(), 5.0);
    /// ```
    #[must_use]
    pub fn with_weight(weight: f64) -> Self {
        Self {
            weight,
            attrs: BTreeMap::new(),
        }
    }

    /// Creates `EdgeAttrs` from weight and attribute map.
    #[must_use]
    pub fn from_parts(weight: f64, attrs: BTreeMap<String, Attribute>) -> Self {
        Self { weight, attrs }
    }

    /// Gets the edge weight.
    #[must_use]
    pub fn weight(&self) -> f64 {
        self.weight
    }

    /// Sets the edge weight.
    pub fn set_weight(&mut self, weight: f64) {
        self.weight = weight;
    }

    /// Builder method to set weight (chainable).
    #[must_use]
    pub fn weight_builder(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Builder method to add an attribute (chainable).
    #[must_use]
    pub fn attr(mut self, key: impl Into<String>, value: Attribute) -> Self {
        self.attrs.insert(key.into(), value);
        self
    }

    /// Gets a reference to the attributes map.
    #[must_use]
    pub fn attrs(&self) -> &BTreeMap<String, Attribute> {
        &self.attrs
    }

    /// Gets a mutable reference to the attributes map.
    pub fn attrs_mut(&mut self) -> &mut BTreeMap<String, Attribute> {
        &mut self.attrs
    }

    /// Converts to `EdgeData` tuple.
    #[must_use]
    pub fn into_edge_data(self) -> EdgeData {
        (self.weight, self.attrs)
    }

    /// Creates `EdgeAttrs` from `EdgeData` tuple.
    #[must_use]
    pub fn from_edge_data(data: &EdgeData) -> Self {
        Self {
            weight: data.0,
            attrs: data.1.clone(),
        }
    }
}

impl Default for EdgeAttrs {
    fn default() -> Self {
        Self::new()
    }
}

impl From<EdgeData> for EdgeAttrs {
    fn from((weight, attrs): EdgeData) -> Self {
        Self { weight, attrs }
    }
}

impl From<EdgeAttrs> for EdgeData {
    fn from(attrs: EdgeAttrs) -> Self {
        (attrs.weight, attrs.attrs)
    }
}

impl From<f64> for EdgeAttrs {
    fn from(weight: f64) -> Self {
        Self::with_weight(weight)
    }
}

/// Graph-level attributes - thin wrapper around `Attrs` for type safety.
///
/// Stores arbitrary key-value pairs for graph-level metadata (e.g., graph type, distance).
/// Derefs to `BTreeMap<String, Attribute>` for direct map operations.
///
/// # Examples
///
/// ```
/// use pecos_num::graph::{GraphAttrs, Attribute};
///
/// let mut attrs = GraphAttrs::new();
/// attrs.insert("graph_type".to_string(), Attribute::String("surface_code".into()));
/// attrs.insert("distance".to_string(), Attribute::Int(5));
/// attrs.insert("noise_model".to_string(), Attribute::String("depolarizing".into()));
/// ```
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GraphAttrs(Attrs);

impl GraphAttrs {
    /// Creates a new empty `GraphAttrs`.
    #[must_use]
    pub fn new() -> Self {
        Self(Attrs::new())
    }
}

impl Default for GraphAttrs {
    fn default() -> Self {
        Self::new()
    }
}

impl std::ops::Deref for GraphAttrs {
    type Target = BTreeMap<String, Attribute>;

    fn deref(&self) -> &Self::Target {
        &self.0.map
    }
}

impl std::ops::DerefMut for GraphAttrs {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0.map
    }
}

/// A graph data structure for quantum error correction applications.
///
/// This is a thin wrapper around petgraph's `UnGraph` (undirected graph) that provides
/// a convenient API for PECOS use cases, particularly MWPM decoding.
///
/// The graph uses `EdgeData` (tuple of weight and attributes) for edges, treating
/// weight as a first-class citizen separate from other attributes.
///
/// # Examples
///
/// ```
/// use pecos_num::graph::Graph;
///
/// let mut graph = Graph::new();
/// let n0 = graph.add_node();
/// let n1 = graph.add_node();
/// graph.add_edge(n0, n1).weight(1.0);
/// ```
#[derive(Debug, Clone)]
pub struct Graph {
    /// The underlying petgraph graph structure.
    /// Uses `NodeAttrs` for node weight and `EdgeData` (f64, attrs map) for edge weight.
    graph: UnGraph<NodeAttrs, EdgeData>,
    /// Graph-level metadata and attributes
    graph_data: GraphAttrs,
}

/// Builder for configuring edge attributes using a fluent interface.
///
/// This builder is returned by `Graph::add_edge()` and allows setting edge weight
/// and attributes via method chaining. The edge is automatically added to the graph
/// when the builder is dropped.
///
/// # Examples
///
/// ```
/// use pecos_num::graph::{Graph, Attribute};
///
/// let mut graph = Graph::new();
/// let n0 = graph.add_node();
/// let n1 = graph.add_node();
///
/// // Simple weight
/// graph.add_edge(n0, n1).weight(5.0);
///
/// // Multiple attributes
/// // graph.add_edge(n0, n1)
/// //     .weight(10.0)
/// //     .add_attr("label", Attribute::String("boundary".into()))
/// //     .add_attr("custom", Attribute::Bool(true));
/// ```
pub struct EdgeBuilder<'a> {
    graph: &'a mut Graph,
    node_a: usize,
    node_b: usize,
    weight: f64,
    attrs: BTreeMap<String, Attribute>,
}

impl EdgeBuilder<'_> {
    /// Sets the weight of the edge.
    #[must_use]
    pub fn weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Adds a single attribute to the edge.
    ///
    /// # Arguments
    ///
    /// * `key` - The attribute name
    /// * `value` - The attribute value
    #[must_use]
    pub fn add_attr(mut self, key: impl Into<String>, value: Attribute) -> Self {
        self.attrs.insert(key.into(), value);
        self
    }

    /// Adds multiple attributes to the edge at once.
    ///
    /// # Arguments
    ///
    /// * `attrs` - A map of attribute names to values
    #[must_use]
    pub fn add_attrs(mut self, attrs: BTreeMap<String, Attribute>) -> Self {
        self.attrs.extend(attrs);
        self
    }
}

impl Drop for EdgeBuilder<'_> {
    /// Automatically adds the edge to the graph when the builder is dropped.
    fn drop(&mut self) {
        let node_a = NodeIndex::new(self.node_a);
        let node_b = NodeIndex::new(self.node_b);
        let edge_data = (self.weight, self.attrs.clone());
        self.graph.graph.add_edge(node_a, node_b, edge_data);
    }
}

impl Graph {
    /// Creates a new empty graph.
    #[must_use]
    pub fn new() -> Self {
        Self {
            graph: UnGraph::new_undirected(),
            graph_data: GraphAttrs::new(),
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
            graph_data: GraphAttrs::new(),
        }
    }

    /// Adds a new node to the graph with empty data.
    ///
    /// Returns the index of the newly created node.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// let n1 = graph.add_node();
    /// ```
    pub fn add_node(&mut self) -> usize {
        self.graph.add_node(NodeAttrs::new()).index()
    }

    /// Adds a node to the graph with pre-built `NodeAttrs`.
    ///
    /// This allows you to attach attributes to the node at creation time.
    ///
    /// # Arguments
    ///
    /// * `data` - Pre-built `NodeAttrs` with attributes
    ///
    /// # Returns
    ///
    /// The index of the newly created node.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::{Graph, NodeAttrs, Attribute};
    ///
    /// let mut graph = Graph::new();
    ///
    /// // Create node with attributes
    /// let mut data = NodeAttrs::new();
    /// data.insert("x".to_string(), Attribute::Float(1.0));
    /// data.insert("y".to_string(), Attribute::Float(2.0));
    /// data.insert("qubit_type".to_string(), Attribute::String("data".into()));
    ///
    /// let n0 = graph.add_node_with_data(data);
    /// ```
    pub fn add_node_with_data(&mut self, data: NodeAttrs) -> usize {
        self.graph.add_node(data).index()
    }

    /// Gets a reference to all graph-level attributes.
    ///
    /// # Returns
    ///
    /// Reference to the graph's `GraphAttrs` containing all attributes.
    #[must_use]
    pub fn graph_data(&self) -> &GraphAttrs {
        &self.graph_data
    }

    /// Gets a mutable reference to all graph-level attributes.
    ///
    /// # Returns
    ///
    /// Mutable reference to the graph's `GraphAttrs` containing all attributes.
    pub fn graph_data_mut(&mut self) -> &mut GraphAttrs {
        &mut self.graph_data
    }

    /// Gets a reference to graph-level attributes as a `BTreeMap`.
    ///
    /// This is a convenience method that returns `&BTreeMap<String, Attribute>` via Deref.
    /// Prefer this over `graph_data()` for direct map access.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::{Graph, Attribute};
    ///
    /// let mut graph = Graph::new();
    /// graph.attrs_mut().insert("distance".to_string(), Attribute::Int(5));
    /// assert_eq!(graph.attrs().get("distance"), Some(&Attribute::Int(5)));
    /// ```
    #[must_use]
    pub fn attrs(&self) -> &BTreeMap<String, Attribute> {
        &self.graph_data
    }

    /// Gets a mutable reference to graph-level attributes as a `BTreeMap`.
    ///
    /// This is a convenience method that returns `&mut BTreeMap<String, Attribute>` via `DerefMut`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::{Graph, Attribute};
    ///
    /// let mut graph = Graph::new();
    /// graph.attrs_mut().insert("distance".to_string(), Attribute::Int(5));
    /// graph.attrs_mut().insert("type".to_string(), Attribute::String("surface_code".into()));
    /// ```
    pub fn attrs_mut(&mut self) -> &mut BTreeMap<String, Attribute> {
        &mut self.graph_data
    }

    /// Gets a reference to a node's attributes as a `BTreeMap`.
    ///
    /// # Arguments
    ///
    /// * `node` - The node index
    ///
    /// # Returns
    ///
    /// Reference to the node's attributes, or None if the node doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::{Graph, Attribute};
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// graph.node_attrs_mut(n0).unwrap().insert("x".to_string(), Attribute::Float(1.0));
    /// assert_eq!(graph.node_attrs(n0).unwrap().get("x"), Some(&Attribute::Float(1.0)));
    /// ```
    #[must_use]
    pub fn node_attrs(&self, node: usize) -> Option<&BTreeMap<String, Attribute>> {
        self.graph
            .node_weight(NodeIndex::new(node))
            .map(|attrs| &**attrs)
    }

    /// Gets a mutable reference to a node's attributes as a `BTreeMap`.
    ///
    /// # Arguments
    ///
    /// * `node` - The node index
    ///
    /// # Returns
    ///
    /// Mutable reference to the node's attributes, or None if the node doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::{Graph, Attribute};
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// graph.node_attrs_mut(n0).unwrap().insert("x".to_string(), Attribute::Float(1.0));
    /// graph.node_attrs_mut(n0).unwrap().insert("y".to_string(), Attribute::Float(2.0));
    /// ```
    pub fn node_attrs_mut(&mut self, node: usize) -> Option<&mut BTreeMap<String, Attribute>> {
        self.graph
            .node_weight_mut(NodeIndex::new(node))
            .map(|attrs| &mut **attrs)
    }

    /// Gets a reference to an edge's attributes as a `BTreeMap`.
    ///
    /// # Arguments
    ///
    /// * `a` - First node index
    /// * `b` - Second node index
    ///
    /// # Returns
    ///
    /// Reference to the edge's attributes, or None if the edge doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::{Graph, Attribute};
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// let n1 = graph.add_node();
    /// graph.add_edge(n0, n1).weight(5.0);
    ///
    /// if let Some(attrs) = graph.edge_attrs(n0, n1) {
    ///     // attrs is the BTreeMap of custom attributes (not including weight)
    ///     // Use get_weight() to access the edge weight
    /// }
    /// ```
    #[must_use]
    pub fn edge_attrs(&self, a: usize, b: usize) -> Option<&BTreeMap<String, Attribute>> {
        let edge_id = self.find_edge(a, b)?;
        self.graph
            .edge_weight(petgraph::graph::EdgeIndex::new(edge_id))
            .map(|(_, attrs)| attrs)
    }

    /// Gets a mutable reference to an edge's attributes as a `BTreeMap`.
    ///
    /// # Arguments
    ///
    /// * `a` - First node index
    /// * `b` - Second node index
    ///
    /// # Returns
    ///
    /// Mutable reference to the edge's attributes, or None if the edge doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::{Graph, Attribute};
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// let n1 = graph.add_node();
    /// graph.add_edge(n0, n1);
    ///
    /// if let Some(attrs) = graph.edge_attrs_mut(n0, n1) {
    ///     attrs.insert("label".to_string(), Attribute::String("boundary".into()));
    /// }
    /// ```
    pub fn edge_attrs_mut(
        &mut self,
        a: usize,
        b: usize,
    ) -> Option<&mut BTreeMap<String, Attribute>> {
        let edge_id = self.find_edge(a, b)?;
        self.graph
            .edge_weight_mut(petgraph::graph::EdgeIndex::new(edge_id))
            .map(|(_, attrs)| attrs)
    }

    /// Gets a reference to edge attributes by edge ID.
    ///
    /// Returns a reference to the `BTreeMap` of edge attributes for direct access.
    ///
    /// # Arguments
    ///
    /// * `edge_id` - The edge index
    ///
    /// # Returns
    ///
    /// Reference to the `BTreeMap` of edge attributes, or None if edge doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// let n1 = graph.add_node();
    /// graph.add_edge(n0, n1).weight(5.0);
    ///
    /// if let Some(edge_id) = graph.find_edge(n0, n1) {
    ///     if let Some(attrs) = graph.edge_attrs_by_id(edge_id) {
    ///         // Access custom attributes
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn edge_attrs_by_id(&self, edge_id: usize) -> Option<&BTreeMap<String, Attribute>> {
        self.graph
            .edge_weight(petgraph::graph::EdgeIndex::new(edge_id))
            .map(|(_, attrs)| attrs)
    }

    /// Gets a mutable reference to edge attributes by edge ID.
    ///
    /// Returns a mutable reference to the `BTreeMap` of edge attributes for direct modification.
    ///
    /// # Arguments
    ///
    /// * `edge_id` - The edge index
    ///
    /// # Returns
    ///
    /// Mutable reference to the `BTreeMap` of edge attributes, or None if edge doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::{Graph, Attribute};
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// let n1 = graph.add_node();
    /// graph.add_edge(n0, n1).weight(5.0);
    ///
    /// if let Some(edge_id) = graph.find_edge(n0, n1) {
    ///     if let Some(attrs) = graph.edge_attrs_by_id_mut(edge_id) {
    ///         attrs.insert("label".to_string(), Attribute::String("boundary".into()));
    ///     }
    /// }
    /// ```
    pub fn edge_attrs_by_id_mut(
        &mut self,
        edge_id: usize,
    ) -> Option<&mut BTreeMap<String, Attribute>> {
        self.graph
            .edge_weight_mut(petgraph::graph::EdgeIndex::new(edge_id))
            .map(|(_, attrs)| attrs)
    }

    /// Adds an edge between two nodes, returning a builder to configure attributes.
    ///
    /// This method returns an `EdgeBuilder` that allows configuring edge attributes
    /// via method chaining. The edge is automatically added to the graph when the
    /// builder is dropped or when an explicit commit method is called.
    ///
    /// # Arguments
    ///
    /// * `a` - Index of the first node
    /// * `b` - Index of the second node
    ///
    /// # Returns
    ///
    /// An `EdgeBuilder` for configuring edge attributes via method chaining.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// let n1 = graph.add_node();
    ///
    /// // Simple edge with just weight
    /// graph.add_edge(n0, n1).weight(5.0);
    ///
    /// // Edge with multiple attributes
    /// // use pecos_num::graph::EdgeAttribute;
    /// // graph.add_edge(n0, n1)
    /// //     .weight(5.0)
    /// //     .label("boundary")
    /// //     .add_attr("custom", EdgeAttribute::Int(42));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if either node index is invalid.
    pub fn add_edge(&mut self, a: usize, b: usize) -> EdgeBuilder<'_> {
        EdgeBuilder {
            graph: self,
            node_a: a,
            node_b: b,
            weight: 1.0, // Default weight
            attrs: BTreeMap::new(),
        }
    }

    /// Adds an edge between two nodes with full edge data (weight and attributes).
    ///
    /// # Arguments
    ///
    /// * `a` - Index of the first node
    /// * `b` - Index of the second node
    /// * `data` - `EdgeAttrs` containing weight and attributes
    ///
    /// # Panics
    ///
    /// Panics if either node index is invalid.
    pub fn add_edge_with_data(&mut self, a: usize, b: usize, data: EdgeAttrs) {
        let node_a = NodeIndex::new(a);
        let node_b = NodeIndex::new(b);
        self.graph.add_edge(node_a, node_b, data.into_edge_data());
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
    /// graph.add_edge(n0, n1).weight(10.0);
    /// graph.add_edge(n2, n3).weight(20.0);
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
    /// // With negative weights, use max_cardinality=true to force matching
    /// graph.add_edge(n0, n1).weight(-5.0);
    /// graph.add_edge(n2, n3).weight(-10.0);
    ///
    /// let matching = graph.max_weight_matching_with_precision(true, 1.0);
    /// assert_eq!(matching.len(), 4);  // Two pairs, each appearing twice
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
                let weight = e.weight().0; // Direct weight access from EdgeData tuple
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
                let weight = e.weight().0; // Direct weight access
                (source, target, weight)
            })
            .collect()
    }

    /// Gets the edge data between two nodes as `EdgeAttrs`.
    ///
    /// # Arguments
    ///
    /// * `a` - Index of the first node
    /// * `b` - Index of the second node
    ///
    /// # Returns
    ///
    /// `EdgeAttrs` containing weight and attributes if edge exists, None otherwise.
    #[must_use]
    pub fn get_edge_data(&self, a: usize, b: usize) -> Option<EdgeAttrs> {
        let node_a = NodeIndex::new(a);
        let node_b = NodeIndex::new(b);

        // Find the edge between the two nodes
        self.graph
            .find_edge(node_a, node_b)
            .and_then(|edge_idx| self.graph.edge_weight(edge_idx))
            .map(EdgeAttrs::from_edge_data)
    }

    /// Finds the edge ID between two nodes.
    ///
    /// # Arguments
    ///
    /// * `a` - Index of the first node
    /// * `b` - Index of the second node
    ///
    /// # Returns
    ///
    /// The edge index if an edge exists between the nodes, None otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// let n1 = graph.add_node();
    /// graph.add_edge(n0, n1).weight(5.0);
    ///
    /// let edge_id = graph.find_edge(n0, n1).unwrap();
    /// assert_eq!(graph.edge_weight(edge_id), 5.0);
    /// ```
    #[must_use]
    pub fn find_edge(&self, a: usize, b: usize) -> Option<usize> {
        let node_a = NodeIndex::new(a);
        let node_b = NodeIndex::new(b);
        self.graph
            .find_edge(node_a, node_b)
            .map(rustworkx_core::petgraph::prelude::EdgeIndex::index)
    }

    /// Gets the endpoints (node pair) of an edge by its edge ID.
    ///
    /// # Arguments
    ///
    /// * `edge_id` - The edge index
    ///
    /// # Returns
    ///
    /// A tuple `(source, target)` with the node indices, or None if the edge doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// let n1 = graph.add_node();
    /// graph.add_edge(n0, n1);
    ///
    /// let edge_id = graph.find_edge(n0, n1).unwrap();
    /// let (a, b) = graph.edge_endpoints(edge_id).unwrap();
    /// assert_eq!((a, b), (n0, n1));
    /// ```
    #[must_use]
    pub fn edge_endpoints(&self, edge_id: usize) -> Option<(usize, usize)> {
        use rustworkx_core::petgraph::graph::EdgeIndex;
        let edge_idx = EdgeIndex::new(edge_id);
        self.graph
            .edge_endpoints(edge_idx)
            .map(|(a, b)| (a.index(), b.index()))
    }

    /// Gets the weight of an edge by its edge ID.
    ///
    /// # Arguments
    ///
    /// * `edge_id` - The edge index
    ///
    /// # Returns
    ///
    /// The weight of the edge.
    ///
    /// # Panics
    ///
    /// Panics if the `edge_id` is invalid.
    #[must_use]
    pub fn edge_weight(&self, edge_id: usize) -> f64 {
        use rustworkx_core::petgraph::graph::EdgeIndex;
        let edge_idx = EdgeIndex::new(edge_id);
        self.graph.edge_weight(edge_idx).expect("Invalid edge ID").0 // Direct weight access
    }

    /// Sets the weight of an edge by its edge ID.
    ///
    /// # Arguments
    ///
    /// * `edge_id` - The edge index
    /// * `weight` - The new weight value
    ///
    /// # Panics
    ///
    /// Panics if the `edge_id` is invalid.
    pub fn set_edge_weight(&mut self, edge_id: usize, weight: f64) {
        use rustworkx_core::petgraph::graph::EdgeIndex;
        let edge_idx = EdgeIndex::new(edge_id);
        self.graph
            .edge_weight_mut(edge_idx)
            .expect("Invalid edge ID")
            .0 = weight; // Direct weight modification
    }

    /// Sets the weight of an edge between two nodes (NetworkX-style).
    ///
    /// This is a convenience method that finds the edge and sets its weight.
    ///
    /// # Arguments
    ///
    /// * `a` - Index of the first node
    /// * `b` - Index of the second node
    /// * `weight` - The new weight value
    ///
    /// # Panics
    ///
    /// Panics if the edge doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// let n1 = graph.add_node();
    /// graph.add_edge(n0, n1).weight(1.0);
    ///
    /// // Update weight using node pair
    /// graph.set_weight(n0, n1, 5.0);
    /// assert_eq!(graph.get_weight(n0, n1), Some(5.0));
    /// ```
    pub fn set_weight(&mut self, a: usize, b: usize, weight: f64) {
        let edge_id = self.find_edge(a, b).expect("Edge not found");
        self.set_edge_weight(edge_id, weight);
    }

    /// Gets the weight of an edge between two nodes (NetworkX-style).
    ///
    /// # Arguments
    ///
    /// * `a` - Index of the first node
    /// * `b` - Index of the second node
    ///
    /// # Returns
    ///
    /// The weight of the edge, or None if the edge doesn't exist.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_num::graph::Graph;
    ///
    /// let mut graph = Graph::new();
    /// let n0 = graph.add_node();
    /// let n1 = graph.add_node();
    /// graph.add_edge(n0, n1).weight(5.0);
    ///
    /// assert_eq!(graph.get_weight(n0, n1), Some(5.0));
    /// assert_eq!(graph.get_weight(n0, 999), None);
    /// ```
    #[must_use]
    pub fn get_weight(&self, a: usize, b: usize) -> Option<f64> {
        self.find_edge(a, b)
            .map(|edge_id| self.edge_weight(edge_id))
    }

    /// Removes an edge by its edge ID.
    ///
    /// # Arguments
    ///
    /// * `edge_id` - The edge index to remove
    ///
    /// # Returns
    ///
    /// The edge data of the removed edge if it existed, None otherwise.
    pub fn remove_edge(&mut self, edge_id: usize) -> Option<EdgeAttrs> {
        use rustworkx_core::petgraph::graph::EdgeIndex;
        let edge_idx = EdgeIndex::new(edge_id);
        self.graph.remove_edge(edge_idx).map(EdgeAttrs::from)
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
                let edge_data = EdgeAttrs::from_edge_data(edge.weight());
                new_graph.add_edge_with_data(new_source, new_target, edge_data);
            }
        }

        new_graph
    }

    /// Computes shortest path distances from a source node using Dijkstra's algorithm.
    ///
    /// This method only computes distances, not the actual paths. It's more efficient than
    /// `single_source_shortest_path()` if you don't need to reconstruct the paths.
    ///
    /// # Arguments
    ///
    /// * `source` - The source node index
    ///
    /// # Returns
    ///
    /// A `BTreeMap` mapping each reachable node to its distance from the source.
    ///
    /// # Panics
    ///
    /// Panics if the source node does not exist in the graph.
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
    ///
    /// graph.add_edge(n0, n1).weight(1.0);
    /// graph.add_edge(n1, n2).weight(2.0);
    ///
    /// let distances = graph.shortest_path_distances(n0);
    /// assert_eq!(distances.get(&n0), Some(&0.0));
    /// assert_eq!(distances.get(&n1), Some(&1.0));
    /// assert_eq!(distances.get(&n2), Some(&3.0));
    /// ```
    #[must_use]
    pub fn shortest_path_distances(&self, source: usize) -> BTreeMap<usize, f64> {
        let source_node = NodeIndex::new(source);

        // Use Dijkstra to get distances (direct weight access from EdgeData)
        dijkstra(&self.graph, source_node, None, |e| e.weight().0)
            .into_iter()
            .map(|(node, dist)| (node.index(), dist))
            .collect()
    }

    /// Computes single-source shortest paths using Dijkstra's algorithm.
    ///
    /// This method computes both distances and reconstructs the actual paths.
    /// If you only need distances, use `shortest_path_distances()` for better performance.
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
    ///
    /// graph.add_edge(n0, n1).weight(1.0);
    /// graph.add_edge(n1, n2).weight(2.0);
    ///
    /// let paths = graph.single_source_shortest_path(n0);
    /// assert_eq!(paths.get(&n0), Some(&vec![n0]));
    /// assert_eq!(paths.get(&n1), Some(&vec![n0, n1]));
    /// assert_eq!(paths.get(&n2), Some(&vec![n0, n1, n2]));
    /// ```
    #[must_use]
    pub fn single_source_shortest_path(&self, source: usize) -> BTreeMap<usize, Vec<usize>> {
        use std::collections::BTreeSet;

        let source_node = NodeIndex::new(source);

        // Use Dijkstra to get distances (direct weight access from EdgeData)
        let distances = dijkstra(&self.graph, source_node, None, |e| e.weight().0);

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
                    let edge_weight = edge.weight().0; // Direct weight access
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
    /// Edge weight is `EdgeData` (tuple of f64 weight and attribute map).
    #[must_use]
    pub fn as_petgraph(&self) -> &UnGraph<NodeAttrs, EdgeData> {
        &self.graph
    }

    /// Provides mutable access to the underlying petgraph for advanced operations.
    ///
    /// Edge weight is `EdgeData` (tuple of f64 weight and attribute map).
    pub fn as_petgraph_mut(&mut self) -> &mut UnGraph<NodeAttrs, EdgeData> {
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
/// graph.add_edge("v1".to_string(), "v2".to_string()).weight(1.0);
/// graph.add_edge("v2".to_string(), "v3".to_string()).weight(2.0);
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

    /// Adds an edge between two nodes, returning a builder to configure attributes.
    ///
    /// If either node doesn't exist, it will be created automatically.
    ///
    /// This method returns an `EdgeBuilder` that allows configuring edge attributes
    /// via method chaining.
    pub fn add_edge(&mut self, a: K, b: K) -> EdgeBuilder<'_> {
        let idx_a = self.get_or_create_index(a);
        let idx_b = self.get_or_create_index(b);
        self.graph.add_edge(idx_a, idx_b)
    }

    /// Adds an edge between two nodes with full edge data.
    pub fn add_edge_with_data(&mut self, a: K, b: K, data: EdgeAttrs) {
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
    pub fn get_edge_data(&self, a: &K, b: &K) -> Option<EdgeAttrs> {
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

    /// Computes shortest path distances from a source node using Dijkstra's algorithm.
    ///
    /// This method only computes distances, not the actual paths.
    #[must_use]
    pub fn shortest_path_distances(&self, source: &K) -> BTreeMap<K, f64> {
        let Some(&source_idx) = self.node_to_index.get(source) else {
            return BTreeMap::new();
        };

        let index_distances = self.graph.shortest_path_distances(source_idx);

        index_distances
            .into_iter()
            .filter_map(|(target_idx, dist)| {
                let target = self.index_to_node.get(&target_idx)?;
                Some((target.clone(), dist))
            })
            .collect()
    }

    /// Computes single-source shortest paths using Dijkstra's algorithm.
    ///
    /// This method computes both distances and reconstructs the actual paths.
    /// If you only need distances, use `shortest_path_distances()` for better performance.
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
#[allow(clippy::float_cmp)] // Tests use exact float literals for storage/retrieval validation
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

        let _ = graph.add_edge(n0, n1).weight(1.0);
        let _ = graph.add_edge(n1, n2).weight(2.0);

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
        let _ = graph.add_edge(n0, n1).weight(10.0);
        let _ = graph.add_edge(n2, n3).weight(20.0);

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
        let _ = graph.add_edge(n0, n1).weight(1.0);
        let _ = graph.add_edge(n1, n2).weight(10.0);
        let _ = graph.add_edge(n0, n2).weight(2.0);

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

        let _ = graph.add_edge(n0, n1).weight(5.5);

        let edges = graph.edges();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0], (n0, n1, 5.5));
    }

    #[test]
    fn test_as_petgraph() {
        let mut graph = Graph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let _ = graph.add_edge(n0, n1).weight(1.0);

        let pg = graph.as_petgraph();
        assert_eq!(pg.node_count(), 2);
        assert_eq!(pg.edge_count(), 1);
    }

    #[test]
    fn test_node_attrs_builder() {
        // Test BTreeMap-style direct access
        let mut attrs = NodeAttrs::new();
        attrs.insert("x".to_string(), Attribute::Float(1.0));
        attrs.insert("y".to_string(), Attribute::Float(2.0));
        attrs.insert("type".to_string(), Attribute::String("data".into()));

        assert_eq!(attrs.get("x"), Some(&Attribute::Float(1.0)));
        assert_eq!(attrs.get("y"), Some(&Attribute::Float(2.0)));
        assert_eq!(attrs.get("type"), Some(&Attribute::String("data".into())));

        // Test Deref to BTreeMap
        assert_eq!(attrs.len(), 3);
        assert!(attrs.contains_key("x"));
        assert!(attrs.contains_key("y"));
        assert!(attrs.contains_key("type"));
    }

    #[test]
    fn test_node_attrs_mutable() {
        let mut attrs = NodeAttrs::new();
        attrs.insert("foo".to_string(), Attribute::Int(42));
        attrs.insert("bar".to_string(), Attribute::Bool(true));

        assert_eq!(attrs.get("foo"), Some(&Attribute::Int(42)));
        assert_eq!(attrs.get("bar"), Some(&Attribute::Bool(true)));

        // Test remove
        let removed = attrs.remove("foo");
        assert_eq!(removed, Some(Attribute::Int(42)));
        assert_eq!(attrs.get("foo"), None);
    }

    #[test]
    fn test_edge_attrs_builder() {
        // Test EdgeAttrs with weight and attributes
        let attrs = EdgeAttrs::with_weight(5.0)
            .attr("label", Attribute::String("boundary".into()))
            .attr("path", Attribute::IntList(vec![1, 2, 3]));

        assert_eq!(attrs.weight(), 5.0);
        assert_eq!(
            attrs.attrs().get("label"),
            Some(&Attribute::String("boundary".into()))
        );
        assert_eq!(
            attrs.attrs().get("path"),
            Some(&Attribute::IntList(vec![1, 2, 3]))
        );
    }

    #[test]
    fn test_graph_attrs_builder() {
        // Test GraphAttrs with BTreeMap-style access
        let mut attrs = GraphAttrs::new();
        attrs.insert("distance".to_string(), Attribute::Int(5));
        attrs.insert("type".to_string(), Attribute::String("surface_code".into()));

        assert_eq!(attrs.get("distance"), Some(&Attribute::Int(5)));
        assert_eq!(
            attrs.get("type"),
            Some(&Attribute::String("surface_code".into()))
        );
    }

    #[test]
    fn test_attrs_extend() {
        use std::collections::BTreeMap;

        let mut map = BTreeMap::new();
        map.insert("a".to_string(), Attribute::Int(1));
        map.insert("b".to_string(), Attribute::String("test".into()));

        let mut attrs = NodeAttrs::new();
        attrs.extend(map);

        assert_eq!(attrs.get("a"), Some(&Attribute::Int(1)));
        assert_eq!(attrs.get("b"), Some(&Attribute::String("test".into())));
    }
}
