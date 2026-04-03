// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
//     Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Python bindings for the pecos graph module.
//!
//! This module provides Python bindings for graph data structures and algorithms,
//! particularly for MWPM (Minimum Weight Perfect Matching) used in quantum error correction.

use pecos_num::dag::DAG as RustDAG;
use pecos_num::digraph::DiGraph as RustDiGraph;
use pecos_num::graph::{Attribute, Attribute as RustAttribute, EdgeAttrs, Graph as RustGraph};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::{BTreeMap, BTreeSet};

/// Helper function to convert Python values to Attribute enum.
fn python_value_to_attribute(value: &Bound<'_, PyAny>, key: &str) -> PyResult<RustAttribute> {
    if let Ok(b) = value.extract::<bool>() {
        Ok(RustAttribute::Bool(b))
    } else if let Ok(i) = value.extract::<i64>() {
        Ok(RustAttribute::Int(i))
    } else if let Ok(f) = value.extract::<f64>() {
        Ok(RustAttribute::Float(f))
    } else if let Ok(v) = value.extract::<Vec<i64>>() {
        Ok(RustAttribute::IntList(v))
    } else if let Ok(v) = value.extract::<Vec<String>>() {
        Ok(RustAttribute::StringList(v))
    } else if let Ok(s) = value.extract::<String>() {
        Ok(RustAttribute::String(s))
    } else {
        // Fallback to JSON
        let py = value.py();
        let json_module = py.import("json")?;
        let json_str: String = json_module.getattr("dumps")?.call1((value,))?.extract()?;

        match serde_json::from_str(&json_str) {
            Ok(json_value) => Ok(RustAttribute::Json(json_value)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Failed to convert edge attribute '{key}' to JSON: {e}"
            ))),
        }
    }
}

/// Helper function to convert Attribute to Python values.
fn attribute_to_python(py: Python<'_>, attr: &RustAttribute) -> PyResult<Py<PyAny>> {
    Ok(match attr {
        RustAttribute::Float(f) => f.into_pyobject(py)?.into_any().unbind(),
        RustAttribute::Int(i) => i.into_pyobject(py)?.into_any().unbind(),
        RustAttribute::String(s) => s.into_pyobject(py)?.into_any().unbind(),
        RustAttribute::Bool(b) => b.into_pyobject(py)?.as_any().clone().unbind(),
        RustAttribute::IntList(v) => v.into_pyobject(py)?.into_any().unbind(),
        RustAttribute::StringList(v) => v.into_pyobject(py)?.into_any().unbind(),
        RustAttribute::Json(json_value) => {
            let json_str = serde_json::to_string(json_value).unwrap();
            let json_module = py.import("json")?;
            json_module
                .getattr("loads")?
                .call1((json_str,))?
                .into_any()
                .unbind()
        }
    })
}

/// Python wrapper for the Rust Graph type.
///
/// This class provides an interface to graph algorithms for quantum error correction,
/// particularly the MWPM decoder. It wraps the Rust `pecos_num::graph::Graph` type.
///
/// # Examples (Python)
///
/// ```python
/// import pecos_rslib
///
/// # Create a new graph
/// graph = pecos_rslib.graph.Graph()
///
/// # Add nodes
/// n0 = graph.add_node()
/// n1 = graph.add_node()
/// n2 = graph.add_node()
/// n3 = graph.add_node()
///
/// # Add edges with weights
/// graph.add_edge(n0, n1, 10.0)
/// graph.add_edge(n2, n3, 20.0)
///
/// # Compute maximum weight matching
/// matching = graph.max_weight_matching()
/// ```
#[pyclass(name = "Graph", module = "pecos_rslib.graph", from_py_object)]
#[derive(Clone)]
pub struct PyGraph {
    /// The underlying Rust graph
    inner: RustGraph,
}

#[pymethods]
impl PyGraph {
    /// Creates a new empty graph.
    ///
    /// # Returns
    ///
    /// A new empty Graph instance.
    #[new]
    fn new() -> Self {
        Self {
            inner: RustGraph::new(),
        }
    }

    /// Helper method to resolve and validate a node index.
    ///
    /// # Arguments
    ///
    /// * `node` - Integer node ID
    ///
    /// # Returns
    ///
    /// The node index
    ///
    /// # Errors
    ///
    /// Returns an error if the node index is out of bounds or not an integer
    fn resolve_node_id(&self, node: &Bound<'_, PyAny>) -> PyResult<usize> {
        let idx = node.extract::<usize>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Node identifier must be an integer (node ID)",
            )
        })?;

        // Validate node exists
        if idx >= self.inner.node_count() {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Node index {idx} out of bounds (graph has {} nodes)",
                self.inner.node_count()
            )));
        }

        Ok(idx)
    }

    /// Creates a new graph with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Expected number of nodes
    /// * `edges` - Expected number of edges
    ///
    /// # Returns
    ///
    /// A new Graph instance with pre-allocated capacity.
    #[staticmethod]
    fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            inner: RustGraph::with_capacity(nodes, edges),
        }
    }

    /// Adds a new node to the graph.
    ///
    /// # Returns
    ///
    /// The index of the newly created node.
    fn add_node(&mut self) -> usize {
        self.inner.add_node()
    }

    /// Adds an edge between two nodes with default weight of 1.0.
    ///
    /// Use `set_weight()` and `edge_attrs()` to configure the edge after creation.
    ///
    /// # Examples
    ///
    /// ```python
    /// graph.add_edge(0, 1)
    /// graph.set_weight(0, 1, 5.0)
    /// graph.edge_attrs(0, 1)["data_path"] = [1, 2, 3]
    /// ```
    fn add_edge(&mut self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<()> {
        // Use helper to resolve node IDs
        let node_a = self.resolve_node_id(a)?;
        let node_b = self.resolve_node_id(b)?;

        // Create edge data with default weight (1.0 is the default)
        let edge_data = pecos_num::graph::EdgeAttrs::new();

        self.inner.add_edge_with_data(node_a, node_b, edge_data);
        Ok(())
    }

    /// Returns the number of nodes in the graph.
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Returns the number of edges in the graph.
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Returns a list of all node indices in the graph.
    ///
    /// # Returns
    ///
    /// A list containing all node indices (0 to node_count-1).
    fn nodes(&self) -> Vec<usize> {
        self.inner.nodes()
    }

    /// Check if a node exists in the graph.
    ///
    /// # Arguments
    ///
    /// * `node` - The node index to check
    ///
    /// # Returns
    ///
    /// True if the node exists, False otherwise.
    ///
    /// # Examples
    ///
    /// ```python
    /// g = Graph()
    /// n0 = g.add_node()
    /// assert g.has_node(n0)
    /// assert not g.has_node(999)
    /// ```
    fn has_node(&self, node: usize) -> bool {
        node < self.inner.node_count()
    }

    /// Remove a node and all its connected edges from the graph.
    ///
    /// # Arguments
    ///
    /// * `node` - The index of the node to remove
    ///
    /// # Returns
    ///
    /// True if the node existed and was removed, False otherwise.
    ///
    /// # Important
    ///
    /// After removing a node, the indices of other nodes may change.
    /// The last node in the graph will be moved to fill the gap left
    /// by the removed node. This means node indices should not be
    /// cached across remove operations.
    ///
    /// # Examples
    ///
    /// ```python
    /// g = Graph()
    /// n0 = g.add_node()
    /// n1 = g.add_node()
    /// n2 = g.add_node()
    /// g.add_edge(n0, n1, weight=1.0)
    ///
    /// # Remove n0 - this also removes the edge to n1
    /// removed = g.remove_node(n0)
    /// assert removed
    /// assert g.node_count() == 2
    /// ```
    fn remove_node(&mut self, node: usize) -> bool {
        self.inner.remove_node(node).is_some()
    }

    /// Computes the maximum weight matching of the graph.
    ///
    /// This function finds a matching (set of edges with no common vertices) that
    /// maximizes the sum of edge weights. This is used in MWPM decoders for quantum
    /// error correction.
    ///
    /// # Arguments
    ///
    /// * `max_cardinality` - If True, prioritize maximum cardinality over maximum weight
    ///
    /// # Returns
    ///
    /// A dictionary mapping node indices to their matched partners.
    fn max_weight_matching(&self, max_cardinality: bool) -> BTreeMap<usize, usize> {
        self.inner.max_weight_matching(max_cardinality)
    }

    /// Compute maximum weight perfect matching with configurable weight precision.
    ///
    /// This is the same as `max_weight_matching` but allows you to control the
    /// float-to-integer conversion multiplier.
    ///
    /// # Arguments
    ///
    /// * `max_cardinality` - If True, prioritize maximum cardinality over maximum weight
    /// * `weight_multiplier` - Multiplier for converting float weights to integers.
    ///                         Default is 1000.0 (preserves 3 decimal places).
    ///                         Use 1.0 if weights are already integers.
    ///                         Use higher values (10000.0+) for more decimal precision.
    ///
    /// # Returns
    ///
    /// A dictionary mapping node indices to their matched partners.
    ///
    /// # Examples
    ///
    /// ```python
    /// # For integer weights, use weight_multiplier=1.0
    /// g = Graph()
    /// n0, n1, n2, n3 = [g.add_node() for _ in range(4)]
    /// g.add_edge(n0, n1)
    /// e1 = g.find_edge(n0, n1)
    /// g.set_edge_weight(e1, -5.0)
    /// g.add_edge(n2, n3)
    /// e2 = g.find_edge(n2, n3)
    /// g.set_edge_weight(e2, -10.0)
    /// matching = g.max_weight_matching_with_precision(True, 1.0)
    /// ```
    #[pyo3(signature = (max_cardinality, weight_multiplier = 1000.0))]
    fn max_weight_matching_with_precision(
        &self,
        max_cardinality: bool,
        weight_multiplier: f64,
    ) -> BTreeMap<usize, usize> {
        self.inner
            .max_weight_matching_with_precision(max_cardinality, weight_multiplier)
    }

    /// Returns a list of all edges as (source, target, weight) tuples.
    ///
    /// # Returns
    ///
    /// A list of tuples (source, target, weight) for all edges in the graph.
    fn edges(&self) -> Vec<(usize, usize, f64)> {
        self.inner.edges()
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
    /// A dictionary with edge weight and attributes if an edge exists, None otherwise.
    /// The dictionary includes "weight" as a key with the edge weight value.
    fn get_edge_data(&self, py: Python<'_>, a: usize, b: usize) -> Option<Py<PyAny>> {
        self.inner.get_edge_data(a, b).map(|edge_attrs| {
            let dict = PyDict::new(py);

            // Add weight as a first-class dictionary item
            dict.set_item("weight", edge_attrs.weight()).unwrap();

            // Add all other attributes
            for (key, value) in edge_attrs.attrs() {
                match value {
                    RustAttribute::Float(f) => {
                        dict.set_item(key, f).unwrap();
                    }
                    RustAttribute::Int(i) => {
                        dict.set_item(key, i).unwrap();
                    }
                    RustAttribute::String(s) => {
                        dict.set_item(key, s.as_str()).unwrap();
                    }
                    RustAttribute::Bool(b) => {
                        dict.set_item(key, b).unwrap();
                    }
                    RustAttribute::IntList(v) => {
                        dict.set_item(key, v.clone()).unwrap();
                    }
                    RustAttribute::StringList(v) => {
                        dict.set_item(key, v.clone()).unwrap();
                    }
                    RustAttribute::Json(json_value) => {
                        // Convert JSON back to Python using json.loads()
                        let json_str = serde_json::to_string(json_value).unwrap();
                        let json_module = py.import("json").unwrap();
                        let py_obj = json_module
                            .getattr("loads")
                            .unwrap()
                            .call1((json_str,))
                            .unwrap();
                        dict.set_item(key, py_obj).unwrap();
                    }
                }
            }
            dict.into()
        })
    }

    /// Gets a mutable view of edge attributes between two nodes.
    ///
    /// Returns an `EdgeAttrsView` that provides dict-like access to edge attributes,
    /// allowing you to read and write attributes directly.
    ///
    /// # Arguments
    ///
    /// * `a` - Index of the first node
    /// * `b` - Index of the second node
    ///
    /// # Returns
    ///
    /// An `EdgeAttrsView` object with dict-like interface.
    ///
    /// # Examples
    ///
    /// ```python
    /// graph = Graph()
    /// n0 = graph.add_node()
    /// n1 = graph.add_node()
    /// graph.add_edge(n0, n1)
    ///
    /// # Get mutable view and set attributes
    /// attrs = graph.edge_attrs(n0, n1)
    /// attrs['label'] = 'boundary'
    /// attrs['data_path'] = [1, 2, 3]
    ///
    /// # Read attributes
    /// label = attrs['label']
    /// ```
    fn edge_attrs(slf: Py<Self>, a: usize, b: usize) -> PyEdgeAttrsView {
        PyEdgeAttrsView {
            graph: slf,
            node_a: a,
            node_b: b,
        }
    }

    /// Returns a `NodeAttrsView` for accessing node attributes.
    ///
    /// Returns a mutable view into the node's attributes that provides dict-like
    /// access similar to Python dicts.
    ///
    /// # Arguments
    ///
    /// * `node` - The node index
    ///
    /// # Returns
    ///
    /// A `NodeAttrsView` object with dict-like interface.
    ///
    /// # Examples
    ///
    /// ```python
    /// graph = Graph()
    /// n0 = graph.add_node()
    ///
    /// # Get mutable view and set attributes
    /// attrs = graph.node_attrs(n0)
    /// attrs["x"] = 1.0
    /// attrs["y"] = 2.0
    /// attrs["type"] = "data"
    ///
    /// # Read attributes
    /// x_val = attrs["x"]
    /// ```
    fn node_attrs(slf: Py<Self>, node: usize) -> PyNodeAttrsView {
        PyNodeAttrsView { graph: slf, node }
    }

    /// Returns a `GraphAttrsView` for accessing graph-level attributes.
    ///
    /// Returns a mutable view into the graph's global attributes that provides
    /// dict-like access similar to Python dicts.
    ///
    /// # Returns
    ///
    /// A `GraphAttrsView` object with dict-like interface.
    ///
    /// # Examples
    ///
    /// ```python
    /// graph = Graph()
    ///
    /// # Get mutable view and set attributes
    /// attrs = graph.attrs()
    /// attrs["distance"] = 5
    /// attrs["code_type"] = "surface_code"
    ///
    /// # Read attributes
    /// distance = attrs["distance"]
    /// ```
    fn attrs(slf: Py<Self>) -> PyGraphAttrsView {
        PyGraphAttrsView { graph: slf }
    }

    /// Creates a subgraph containing only the specified nodes.
    ///
    /// # Arguments
    ///
    /// * `nodes` - A list of node indices to include in the subgraph
    ///
    /// # Returns
    ///
    /// A new Graph containing only the specified nodes and edges between them.
    #[allow(clippy::needless_pass_by_value)] // PyO3 requires ownership for internal graph operations
    fn subgraph(&self, nodes: Vec<usize>) -> Self {
        Self {
            inner: self.inner.subgraph(&nodes),
        }
    }

    /// Computes single-source shortest paths using Dijkstra's algorithm.
    ///
    /// # Arguments
    ///
    /// * `source` - The source node index
    ///
    /// # Returns
    ///
    /// A dictionary mapping each reachable node to a list of node indices representing
    /// the shortest path from the source to that node.
    fn single_source_shortest_path(&self, source: usize) -> BTreeMap<usize, Vec<usize>> {
        self.inner.single_source_shortest_path(source)
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
    /// A dictionary mapping each reachable node to its distance from the source.
    ///
    /// # Examples
    ///
    /// ```python
    /// graph = Graph()
    /// n0 = graph.add_node()
    /// n1 = graph.add_node()
    /// n2 = graph.add_node()
    /// graph.add_edge(n0, n1)
    /// graph.set_weight(n0, n1, 1.0)
    /// graph.add_edge(n1, n2)
    /// graph.set_weight(n1, n2, 2.0)
    ///
    /// distances = graph.shortest_path_distances(n0)
    /// assert distances[n0] == 0.0
    /// assert distances[n1] == 1.0
    /// assert distances[n2] == 3.0
    /// ```
    fn shortest_path_distances(&self, source: usize) -> BTreeMap<usize, f64> {
        self.inner.shortest_path_distances(source)
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
    fn find_edge(&self, a: usize, b: usize) -> Option<usize> {
        self.inner.find_edge(a, b)
    }

    /// Gets the endpoints (node pair) of an edge by its edge ID.
    ///
    /// # Arguments
    ///
    /// * `edge_id` - The edge index
    ///
    /// # Returns
    ///
    /// A tuple (source, target) with the node indices, or None if the edge doesn't exist.
    fn edge_endpoints(&self, edge_id: usize) -> Option<(usize, usize)> {
        self.inner.edge_endpoints(edge_id)
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
    fn edge_weight(&self, edge_id: usize) -> f64 {
        self.inner.edge_weight(edge_id)
    }

    /// Sets the weight of an edge by its edge ID.
    ///
    /// # Arguments
    ///
    /// * `edge_id` - The edge index
    /// * `weight` - The new weight value
    fn set_edge_weight(&mut self, edge_id: usize, weight: f64) {
        self.inner.set_edge_weight(edge_id, weight);
    }

    /// Sets the weight of an edge between two nodes (NetworkX-style).
    ///
    /// This is a convenience method that finds the edge and sets its weight.
    ///
    /// # Arguments
    ///
    /// * `a` - First node (integer ID)
    /// * `b` - Second node (integer ID)
    /// * `weight` - The new weight value
    ///
    /// # Examples
    ///
    /// ```python
    /// graph.add_edge(n0, n1)
    /// graph.set_weight(n0, n1, 5.0)  # No need to find edge ID!
    ///
    /// # Works with labels too
    /// graph.set_weight("v1", "v2", 3.0)
    /// ```
    fn set_weight(
        &mut self,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        weight: f64,
    ) -> PyResult<()> {
        let node_a = self.resolve_node_id(a)?;
        let node_b = self.resolve_node_id(b)?;
        self.inner.set_weight(node_a, node_b, weight);
        Ok(())
    }

    /// Gets the weight of an edge between two nodes (NetworkX-style).
    ///
    /// # Arguments
    ///
    /// * `a` - First node (integer ID)
    /// * `b` - Second node (integer ID)
    ///
    /// # Returns
    ///
    /// The weight of the edge, or None if the edge doesn't exist.
    ///
    /// # Examples
    ///
    /// ```python
    /// graph.add_edge(n0, n1)
    /// graph.set_weight(n0, n1, 5.0)
    /// weight = graph.get_weight(n0, n1)  # Returns 5.0
    /// ```
    fn get_weight(&self, a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> PyResult<Option<f64>> {
        let node_a = self.resolve_node_id(a)?;
        let node_b = self.resolve_node_id(b)?;
        Ok(self.inner.get_weight(node_a, node_b))
    }

    /// Removes an edge by its edge ID.
    ///
    /// # Arguments
    ///
    /// * `edge_id` - The edge index to remove
    ///
    /// # Returns
    ///
    /// True if the edge was removed, False otherwise (edge didn't exist).
    fn remove_edge(&mut self, edge_id: usize) -> bool {
        self.inner.remove_edge(edge_id).is_some()
    }

    /// Returns a string representation of the graph.
    fn __repr__(&self) -> String {
        format!(
            "Graph(nodes={}, edges={})",
            self.inner.node_count(),
            self.inner.edge_count()
        )
    }
}

/// Mutable view into edge attributes that provides dict-like access.
///
/// This class holds a reference to the graph and edge endpoints, allowing
/// mutations to be written back to the graph.
#[pyclass(name = "EdgeAttrsView", module = "pecos_rslib.graph")]
pub struct PyEdgeAttrsView {
    graph: Py<PyGraph>,
    node_a: usize,
    node_b: usize,
}

#[pymethods]
impl PyEdgeAttrsView {
    fn __setitem__(&self, py: Python<'_>, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);

        // Convert Python value to Attribute
        let attr = python_value_to_attribute(value, &key)?;

        // Get mutable access to edge attributes
        if let Some(attrs) = graph.inner.edge_attrs_mut(self.node_a, self.node_b) {
            attrs.insert(key, attr);
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                "Edge does not exist",
            ))
        }
    }

    fn __getitem__(&self, py: Python<'_>, key: String) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);

        if let Some(attrs) = graph.inner.edge_attrs(self.node_a, self.node_b) {
            if let Some(attr) = attrs.get(&key) {
                attribute_to_python(py, attr)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(key))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                "Edge does not exist",
            ))
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get(
        &self,
        py: Python<'_>,
        key: &str,
        default: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);

        if let Some(attrs) = graph.inner.edge_attrs(self.node_a, self.node_b) {
            if let Some(attr) = attrs.get(key) {
                attribute_to_python(py, attr)
            } else if let Some(def) = default {
                Ok(def.clone().unbind())
            } else {
                Ok(py.None())
            }
        } else if let Some(def) = default {
            Ok(def.clone().unbind())
        } else {
            Ok(py.None())
        }
    }

    /// Check if an attribute exists (dict-like interface).
    fn __contains__(&self, py: Python<'_>, key: &str) -> bool {
        let graph = self.graph.borrow(py);

        if let Some(attrs) = graph.inner.edge_attrs(self.node_a, self.node_b) {
            attrs.contains_key(key)
        } else {
            false
        }
    }

    /// Insert a key-value pair into edge attributes (chainable).
    ///
    /// This method allows for method chaining, similar to Rust's `BTreeMap` insert.
    ///
    /// # Arguments
    ///
    /// * `key` - The attribute name
    /// * `value` - The attribute value
    ///
    /// # Returns
    ///
    /// Returns self for chaining.
    ///
    /// # Examples
    ///
    /// ```python
    /// # Chainable style
    /// attrs = graph.edge_attrs(n0, n1)
    /// attrs.insert("weight", 5.0).insert("label", "boundary").insert("path", [1, 2, 3])
    ///
    /// # Or dict-like style
    /// attrs["weight"] = 5.0
    /// ```
    fn insert(
        slf: Py<Self>,
        py: Python<'_>,
        key: String,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        // Extract needed data before moving slf
        let (graph_ref, node_a, node_b) = {
            let view = slf.borrow(py);
            (view.graph.clone_ref(py), view.node_a, view.node_b)
        };

        let mut graph = graph_ref.borrow_mut(py);

        // Convert Python value to Attribute
        let attr = python_value_to_attribute(value, &key)?;

        // Get mutable access to edge attributes
        if let Some(attrs) = graph.inner.edge_attrs_mut(node_a, node_b) {
            attrs.insert(key, attr);
            drop(graph); // Release the borrow before returning
            Ok(slf)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                "Edge does not exist",
            ))
        }
    }

    /// Update multiple attributes from a dict (dict-like interface).
    ///
    /// This method updates the edge attributes with key-value pairs from the provided dict,
    /// similar to Python's `dict.update()` method.
    ///
    /// # Arguments
    ///
    /// * `items` - A dictionary or iterable of key-value pairs
    ///
    /// # Examples
    ///
    /// ```python
    /// # From a dict
    /// attrs = graph.edge_attrs(n0, n1)
    /// attrs.update({"weight": 5.0, "label": "boundary", "path": [1, 2, 3]})
    ///
    /// # Can also update from another EdgeAttrsView or any dict-like object
    /// other_attrs = graph.edge_attrs(n2, n3)
    /// attrs.update(other_attrs)
    /// ```
    fn update(&self, py: Python<'_>, items: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);

        if let Some(attrs) = graph.inner.edge_attrs_mut(self.node_a, self.node_b) {
            // Try to iterate over items
            // First try treating it as a dict with .items()
            if let Ok(dict_items) = items.call_method0("items") {
                for item in dict_items.try_iter()? {
                    let pair = item?;
                    let tuple: pyo3::Bound<pyo3::types::PyTuple> = pair.cast_into()?;
                    if tuple.len() != 2 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Expected key-value pairs",
                        ));
                    }
                    let key: String = tuple.get_item(0)?.extract()?;
                    let value = tuple.get_item(1)?;
                    let attr = python_value_to_attribute(&value, &key)?;
                    attrs.insert(key, attr);
                }
            } else {
                // Otherwise try iterating directly (for sequences of tuples)
                for item in items.try_iter()? {
                    let pair = item?;
                    let tuple: pyo3::Bound<pyo3::types::PyTuple> = pair.cast_into()?;
                    if tuple.len() != 2 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Expected key-value pairs",
                        ));
                    }
                    let key: String = tuple.get_item(0)?.extract()?;
                    let value = tuple.get_item(1)?;
                    let attr = python_value_to_attribute(&value, &key)?;
                    attrs.insert(key, attr);
                }
            }
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                "Edge does not exist",
            ))
        }
    }
}

/// Mutable view into node attributes that provides dict-like access.
///
/// This is returned by `Graph.node_attrs(node)` and provides a Python dict-like interface
/// for accessing and modifying attributes of a specific node.
#[pyclass(name = "NodeAttrsView", module = "pecos_rslib.graph")]
pub struct PyNodeAttrsView {
    graph: Py<PyGraph>,
    node: usize,
}

#[pymethods]
impl PyNodeAttrsView {
    /// Set an attribute value (dict-like interface).
    fn __setitem__(&self, py: Python<'_>, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);

        if let Some(attrs) = graph.inner.node_attrs_mut(self.node) {
            let attr = python_value_to_attribute(value, &key)?;
            attrs.insert(key, attr);
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                "Node does not exist",
            ))
        }
    }

    /// Get an attribute value (dict-like interface).
    fn __getitem__(&self, py: Python<'_>, key: String) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);

        if let Some(attrs) = graph.inner.node_attrs(self.node) {
            if let Some(attr) = attrs.get(&key) {
                attribute_to_python(py, attr)
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(key))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                "Node does not exist",
            ))
        }
    }

    /// Delete an attribute (dict-like interface).
    fn __delitem__(&self, py: Python<'_>, key: String) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);

        if let Some(attrs) = graph.inner.node_attrs_mut(self.node) {
            if attrs.remove(&key).is_some() {
                Ok(())
            } else {
                Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(key))
            }
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                "Node does not exist",
            ))
        }
    }

    /// Check if an attribute exists (dict-like interface).
    fn __contains__(&self, py: Python<'_>, key: &str) -> bool {
        let graph = self.graph.borrow(py);

        if let Some(attrs) = graph.inner.node_attrs(self.node) {
            attrs.contains_key(key)
        } else {
            false
        }
    }

    /// Get an attribute with an optional default value.
    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);

        if let Some(attrs) = graph.inner.node_attrs(self.node) {
            if let Some(attr) = attrs.get(key) {
                attribute_to_python(py, attr)
            } else {
                Ok(default.unwrap_or_else(|| py.None()))
            }
        } else {
            Ok(default.unwrap_or_else(|| py.None()))
        }
    }

    /// Insert an attribute and return self for chaining.
    fn insert(
        slf: Py<Self>,
        py: Python<'_>,
        key: String,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        let node = {
            let view = slf.borrow(py);
            view.node
        };

        {
            let view = slf.borrow(py);
            let mut graph = view.graph.borrow_mut(py);

            if let Some(attrs) = graph.inner.node_attrs_mut(node) {
                let attr = python_value_to_attribute(value, &key)?;
                attrs.insert(key, attr);
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    "Node does not exist",
                ));
            }
        }

        Ok(slf)
    }

    /// Update multiple attributes from a dict or iterable of key-value pairs.
    fn update(&self, py: Python<'_>, items: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);

        if let Some(attrs) = graph.inner.node_attrs_mut(self.node) {
            // Try to iterate over items
            // First try treating it as a dict with .items()
            if let Ok(dict_items) = items.call_method0("items") {
                for item in dict_items.try_iter()? {
                    let pair = item?;
                    let tuple: pyo3::Bound<pyo3::types::PyTuple> = pair.cast_into()?;
                    if tuple.len() != 2 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Expected key-value pairs",
                        ));
                    }
                    let key: String = tuple.get_item(0)?.extract()?;
                    let value = tuple.get_item(1)?;
                    let attr = python_value_to_attribute(&value, &key)?;
                    attrs.insert(key, attr);
                }
            } else {
                // Otherwise try iterating directly (for sequences of tuples)
                for item in items.try_iter()? {
                    let pair = item?;
                    let tuple: pyo3::Bound<pyo3::types::PyTuple> = pair.cast_into()?;
                    if tuple.len() != 2 {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                            "Expected key-value pairs",
                        ));
                    }
                    let key: String = tuple.get_item(0)?.extract()?;
                    let value = tuple.get_item(1)?;
                    let attr = python_value_to_attribute(&value, &key)?;
                    attrs.insert(key, attr);
                }
            }
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                "Node does not exist",
            ))
        }
    }
}

/// Mutable view into graph-level attributes that provides dict-like access.
///
/// This is returned by `Graph.attrs()` and provides a Python dict-like interface
/// for accessing and modifying graph-level attributes.
#[pyclass(name = "GraphAttrsView", module = "pecos_rslib.graph")]
pub struct PyGraphAttrsView {
    graph: Py<PyGraph>,
}

#[pymethods]
impl PyGraphAttrsView {
    /// Set an attribute value (dict-like interface).
    fn __setitem__(&self, py: Python<'_>, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        let attrs = graph.inner.attrs_mut();
        let attr = python_value_to_attribute(value, &key)?;
        attrs.insert(key, attr);
        Ok(())
    }

    /// Get an attribute value (dict-like interface).
    fn __getitem__(&self, py: Python<'_>, key: String) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs = graph.inner.attrs();

        if let Some(attr) = attrs.get(&key) {
            attribute_to_python(py, attr)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(key))
        }
    }

    /// Delete an attribute (dict-like interface).
    fn __delitem__(&self, py: Python<'_>, key: String) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        let attrs = graph.inner.attrs_mut();

        if attrs.remove(&key).is_some() {
            Ok(())
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(key))
        }
    }

    /// Check if an attribute exists (dict-like interface).
    fn __contains__(&self, py: Python<'_>, key: &str) -> bool {
        let graph = self.graph.borrow(py);
        let attrs = graph.inner.attrs();
        attrs.contains_key(key)
    }

    /// Get an attribute with an optional default value.
    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs = graph.inner.attrs();

        if let Some(attr) = attrs.get(key) {
            attribute_to_python(py, attr)
        } else {
            Ok(default.unwrap_or_else(|| py.None()))
        }
    }

    /// Insert an attribute and return self for chaining.
    fn insert(
        slf: Py<Self>,
        py: Python<'_>,
        key: String,
        value: &Bound<'_, PyAny>,
    ) -> PyResult<Py<Self>> {
        {
            let view = slf.borrow(py);
            let mut graph = view.graph.borrow_mut(py);
            let attrs = graph.inner.attrs_mut();
            let attr = python_value_to_attribute(value, &key)?;
            attrs.insert(key, attr);
        }
        Ok(slf)
    }

    /// Update multiple attributes from a dict or iterable of key-value pairs.
    fn update(&self, py: Python<'_>, items: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        let attrs = graph.inner.attrs_mut();

        // Try to iterate over items
        // First try treating it as a dict with .items()
        if let Ok(dict_items) = items.call_method0("items") {
            for item in dict_items.try_iter()? {
                let pair = item?;
                let tuple: pyo3::Bound<pyo3::types::PyTuple> = pair.cast_into()?;
                if tuple.len() != 2 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Expected key-value pairs",
                    ));
                }
                let key: String = tuple.get_item(0)?.extract()?;
                let value = tuple.get_item(1)?;
                let attr = python_value_to_attribute(&value, &key)?;
                attrs.insert(key, attr);
            }
        } else {
            // Otherwise try iterating directly (for sequences of tuples)
            for item in items.try_iter()? {
                let pair = item?;
                let tuple: pyo3::Bound<pyo3::types::PyTuple> = pair.cast_into()?;
                if tuple.len() != 2 {
                    return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                        "Expected key-value pairs",
                    ));
                }
                let key: String = tuple.get_item(0)?.extract()?;
                let value = tuple.get_item(1)?;
                let attr = python_value_to_attribute(&value, &key)?;
                attrs.insert(key, attr);
            }
        }
        Ok(())
    }
}

// =============================================================================
// PyDiGraph - Directed Graph
// =============================================================================

/// Python wrapper for the Rust `DiGraph` type (directed graph).
///
/// This class provides an interface to directed graph operations. It wraps
/// the Rust `pecos_num::digraph::DiGraph` type.
///
/// # Examples (Python)
///
/// ```python
/// import pecos_rslib
///
/// # Create a new directed graph
/// g = pecos_rslib.graph.DiGraph()
///
/// # Add nodes
/// n0 = g.add_node()
/// n1 = g.add_node()
/// n2 = g.add_node()
///
/// # Add directed edges
/// g.add_edge(n0, n1)  # n0 -> n1
/// g.add_edge(n1, n2)  # n1 -> n2
///
/// # Query directed relationships
/// assert g.successors(n0) == [n1]
/// assert g.predecessors(n2) == [n1]
///
/// # Topological sort
/// order = g.topological_sort()  # Returns [n0, n1, n2] or None if cyclic
/// ```
#[pyclass(name = "DiGraph", module = "pecos_rslib.graph", from_py_object)]
#[derive(Clone)]
pub struct PyDiGraph {
    inner: RustDiGraph,
}

#[pymethods]
impl PyDiGraph {
    /// Creates a new empty directed graph.
    #[new]
    fn new() -> Self {
        Self {
            inner: RustDiGraph::new(),
        }
    }

    /// Helper method to resolve and validate a node index.
    fn resolve_node_id(&self, node: &Bound<'_, PyAny>) -> PyResult<usize> {
        let idx = node.extract::<usize>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Node identifier must be an integer (node ID)",
            )
        })?;

        if !self.inner.nodes().contains(&idx) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Node index {idx} does not exist in graph"
            )));
        }

        Ok(idx)
    }

    /// Creates a new directed graph with pre-allocated capacity.
    #[staticmethod]
    fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            inner: RustDiGraph::with_capacity(nodes, edges),
        }
    }

    /// Adds a new node to the graph.
    fn add_node(&mut self) -> usize {
        self.inner.add_node()
    }

    /// Adds a directed edge from source to target with default weight of 1.0.
    fn add_edge(
        &mut self,
        source: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let src = self.resolve_node_id(source)?;
        let tgt = self.resolve_node_id(target)?;

        let edge_data = EdgeAttrs::new();
        self.inner.add_edge_with_data(src, tgt, edge_data);

        // Return the edge ID
        Ok(self.inner.find_edge(src, tgt).unwrap_or(0))
    }

    /// Returns the number of nodes in the graph.
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Returns the number of edges in the graph.
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Returns a list of all node indices in the graph.
    fn nodes(&self) -> Vec<usize> {
        self.inner.nodes()
    }

    /// Check if a node exists in the graph.
    fn has_node(&self, node: usize) -> bool {
        self.inner.nodes().contains(&node)
    }

    /// Remove a node and all its connected edges from the graph.
    fn remove_node(&mut self, node: usize) -> bool {
        self.inner.remove_node(node).is_some()
    }

    /// Returns a list of all edges as (source, target, weight) tuples.
    fn edges(&self) -> Vec<(usize, usize, f64)> {
        self.inner.edges()
    }

    /// Returns the predecessors of a node (nodes with edges pointing to this node).
    fn predecessors(&self, node: usize) -> Vec<usize> {
        self.inner.predecessors(node)
    }

    /// Returns the successors of a node (nodes this node points to).
    fn successors(&self, node: usize) -> Vec<usize> {
        self.inner.successors(node)
    }

    /// Returns the in-degree of a node (number of incoming edges).
    fn in_degree(&self, node: usize) -> usize {
        self.inner.in_degree(node)
    }

    /// Returns the out-degree of a node (number of outgoing edges).
    fn out_degree(&self, node: usize) -> usize {
        self.inner.out_degree(node)
    }

    /// Returns edge IDs of incoming edges to a node.
    fn in_edges(&self, node: usize) -> Vec<usize> {
        self.inner.in_edges(node)
    }

    /// Returns edge IDs of outgoing edges from a node.
    fn out_edges(&self, node: usize) -> Vec<usize> {
        self.inner.out_edges(node)
    }

    /// Returns a topological ordering of the graph, or None if the graph has a cycle.
    fn topological_sort(&self) -> Option<Vec<usize>> {
        self.inner.topological_sort()
    }

    /// Returns True if the graph has no cycles.
    fn is_acyclic(&self) -> bool {
        self.inner.is_acyclic()
    }

    /// Returns True if there is a path from source to target.
    fn has_path(&self, source: usize, target: usize) -> bool {
        self.inner.has_path(source, target)
    }

    /// Creates a subgraph containing only the specified nodes.
    #[allow(clippy::needless_pass_by_value)]
    fn subgraph(&self, nodes: Vec<usize>) -> Self {
        Self {
            inner: self.inner.subgraph(&nodes),
        }
    }

    /// Finds the edge ID between two nodes.
    fn find_edge(&self, source: usize, target: usize) -> Option<usize> {
        self.inner.find_edge(source, target)
    }

    /// Gets the endpoints of an edge by its edge ID.
    fn edge_endpoints(&self, edge_id: usize) -> Option<(usize, usize)> {
        self.inner.edge_endpoints(edge_id)
    }

    /// Gets the weight of an edge by its edge ID.
    fn edge_weight(&self, edge_id: usize) -> f64 {
        self.inner.edge_weight(edge_id)
    }

    /// Sets the weight of an edge by its edge ID.
    fn set_edge_weight(&mut self, edge_id: usize, weight: f64) {
        self.inner.set_edge_weight(edge_id, weight);
    }

    /// Gets the weight of an edge between two nodes.
    fn get_weight(
        &self,
        source: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
    ) -> PyResult<Option<f64>> {
        let src = self.resolve_node_id(source)?;
        let tgt = self.resolve_node_id(target)?;
        Ok(self.inner.get_weight(src, tgt))
    }

    /// Sets the weight of an edge between two nodes.
    fn set_weight(
        &mut self,
        source: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
        weight: f64,
    ) -> PyResult<()> {
        let src = self.resolve_node_id(source)?;
        let tgt = self.resolve_node_id(target)?;
        self.inner.set_weight(src, tgt, weight);
        Ok(())
    }

    /// Removes an edge by its edge ID.
    fn remove_edge(&mut self, edge_id: usize) -> bool {
        self.inner.remove_edge(edge_id).is_some()
    }

    /// Gets edge data between two nodes.
    fn get_edge_data(&self, py: Python<'_>, source: usize, target: usize) -> Option<Py<PyAny>> {
        self.inner
            .get_edge_data(source, target)
            .map(|edge_attrs: EdgeAttrs| {
                let dict = PyDict::new(py);
                dict.set_item("weight", edge_attrs.weight()).unwrap();
                for (key, value) in edge_attrs.attrs() {
                    let py_value = attribute_to_python(py, value).unwrap();
                    dict.set_item(key, py_value).unwrap();
                }
                dict.into()
            })
    }

    /// Returns a mutable view of edge attributes.
    fn edge_attrs(slf: Py<Self>, source: usize, target: usize) -> PyDiGraphEdgeAttrsView {
        PyDiGraphEdgeAttrsView {
            graph: slf,
            source,
            target,
        }
    }

    /// Returns a mutable view of node attributes.
    fn node_attrs(slf: Py<Self>, node: usize) -> PyDiGraphNodeAttrsView {
        PyDiGraphNodeAttrsView { graph: slf, node }
    }

    /// Returns a mutable view of graph-level attributes.
    fn attrs(slf: Py<Self>) -> PyDiGraphAttrsView {
        PyDiGraphAttrsView { graph: slf }
    }

    fn __repr__(&self) -> String {
        format!(
            "DiGraph(nodes={}, edges={})",
            self.inner.node_count(),
            self.inner.edge_count()
        )
    }
}

/// Mutable view into `DiGraph` edge attributes.
#[pyclass(name = "DiGraphEdgeAttrsView", module = "pecos_rslib.graph")]
pub struct PyDiGraphEdgeAttrsView {
    graph: Py<PyDiGraph>,
    source: usize,
    target: usize,
}

#[pymethods]
impl PyDiGraphEdgeAttrsView {
    fn __setitem__(&self, py: Python<'_>, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        let attr = python_value_to_attribute(value, &key)?;
        let attrs: &mut BTreeMap<String, Attribute> =
            match graph.inner.edge_attrs_mut(self.source, self.target) {
                Some(a) => a,
                None => {
                    return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Edge does not exist",
                    ));
                }
            };
        attrs.insert(key, attr);
        Ok(())
    }

    fn __getitem__(&self, py: Python<'_>, key: String) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs: &BTreeMap<String, Attribute> =
            match graph.inner.edge_attrs(self.source, self.target) {
                Some(a) => a,
                None => {
                    return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Edge does not exist",
                    ));
                }
            };
        if let Some(attr) = attrs.get(&key) {
            attribute_to_python(py, attr)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(key))
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get(
        &self,
        py: Python<'_>,
        key: &str,
        default: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs: Option<&BTreeMap<String, Attribute>> =
            graph.inner.edge_attrs(self.source, self.target);
        if let Some(attrs) = attrs {
            if let Some(attr) = attrs.get(key) {
                attribute_to_python(py, attr)
            } else if let Some(def) = default {
                Ok(def.clone().unbind())
            } else {
                Ok(py.None())
            }
        } else if let Some(def) = default {
            Ok(def.clone().unbind())
        } else {
            Ok(py.None())
        }
    }
}

/// Mutable view into `DiGraph` node attributes.
#[pyclass(name = "DiGraphNodeAttrsView", module = "pecos_rslib.graph")]
pub struct PyDiGraphNodeAttrsView {
    graph: Py<PyDiGraph>,
    node: usize,
}

#[pymethods]
impl PyDiGraphNodeAttrsView {
    fn __setitem__(&self, py: Python<'_>, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        let attrs: &mut BTreeMap<String, Attribute> = match graph.inner.node_attrs_mut(self.node) {
            Some(a) => a,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    "Node does not exist",
                ));
            }
        };
        let attr = python_value_to_attribute(value, &key)?;
        attrs.insert(key, attr);
        Ok(())
    }

    fn __getitem__(&self, py: Python<'_>, key: String) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs: &BTreeMap<String, Attribute> = match graph.inner.node_attrs(self.node) {
            Some(a) => a,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    "Node does not exist",
                ));
            }
        };
        if let Some(attr) = attrs.get(&key) {
            attribute_to_python(py, attr)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(key))
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs: Option<&BTreeMap<String, Attribute>> = graph.inner.node_attrs(self.node);
        if let Some(attrs) = attrs {
            if let Some(attr) = attrs.get(key) {
                attribute_to_python(py, attr)
            } else {
                Ok(default.unwrap_or_else(|| py.None()))
            }
        } else {
            Ok(default.unwrap_or_else(|| py.None()))
        }
    }
}

/// Mutable view into DiGraph-level attributes.
#[pyclass(name = "DiGraphAttrsView", module = "pecos_rslib.graph")]
pub struct PyDiGraphAttrsView {
    graph: Py<PyDiGraph>,
}

#[pymethods]
impl PyDiGraphAttrsView {
    fn __setitem__(&self, py: Python<'_>, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        let attr = python_value_to_attribute(value, &key)?;
        let attrs: &mut BTreeMap<String, Attribute> = graph.inner.attrs_mut();
        attrs.insert(key, attr);
        Ok(())
    }

    fn __getitem__(&self, py: Python<'_>, key: String) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs: &BTreeMap<String, Attribute> = graph.inner.attrs();
        if let Some(attr) = attrs.get(&key) {
            attribute_to_python(py, attr)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(key))
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs: &BTreeMap<String, Attribute> = graph.inner.attrs();
        if let Some(attr) = attrs.get(key) {
            attribute_to_python(py, attr)
        } else {
            Ok(default.unwrap_or_else(|| py.None()))
        }
    }
}

// =============================================================================
// PyDAG - Directed Acyclic Graph
// =============================================================================

/// Python wrapper for the Rust DAG type (directed acyclic graph).
///
/// This class provides an interface to DAG operations with cycle checking.
/// Adding an edge that would create a cycle raises an error.
///
/// # Examples (Python)
///
/// ```python
/// import pecos_rslib
///
/// # Create a new DAG
/// g = pecos_rslib.graph.DAG()
///
/// # Add nodes
/// n0 = g.add_node()
/// n1 = g.add_node()
/// n2 = g.add_node()
///
/// # Add directed edges (automatically checked for cycles)
/// g.add_edge(n0, n1)  # n0 -> n1
/// g.add_edge(n1, n2)  # n1 -> n2
///
/// # This would raise an error (creates cycle):
/// # g.add_edge(n2, n0)  # Raises DagWouldCycleError
///
/// # Topological sort always succeeds for DAGs
/// order = g.topological_sort()
///
/// # DAG-specific operations
/// roots = g.roots()      # Nodes with no predecessors
/// leaves = g.leaves()    # Nodes with no successors
/// ```
#[pyclass(name = "DAG", module = "pecos_rslib.graph", from_py_object)]
#[derive(Clone)]
pub struct PyDAG {
    inner: RustDAG,
}

// Exception raised when adding an edge would create a cycle in a DAG.
pyo3::create_exception!(
    pecos_rslib,
    DagWouldCycleError,
    pyo3::exceptions::PyException
);

// Exception raised when trying to create a DAG from a cyclic graph.
pyo3::create_exception!(pecos_rslib, DagHasCycleError, pyo3::exceptions::PyException);

#[pymethods]
impl PyDAG {
    /// Creates a new empty DAG.
    #[new]
    fn new() -> Self {
        Self {
            inner: RustDAG::new(),
        }
    }

    /// Helper method to resolve and validate a node index.
    fn resolve_node_id(&self, node: &Bound<'_, PyAny>) -> PyResult<usize> {
        let idx = node.extract::<usize>().map_err(|_| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(
                "Node identifier must be an integer (node ID)",
            )
        })?;

        if !self.inner.nodes().contains(&idx) {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "Node index {idx} does not exist in graph"
            )));
        }

        Ok(idx)
    }

    /// Creates a new DAG with pre-allocated capacity.
    #[staticmethod]
    fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            inner: RustDAG::with_capacity(nodes, edges),
        }
    }

    /// Creates a DAG from a `DiGraph`.
    ///
    /// Raises `DagHasCycleError` if the `DiGraph` contains a cycle.
    #[staticmethod]
    fn from_digraph(digraph: &PyDiGraph) -> PyResult<Self> {
        match RustDAG::try_from_digraph(digraph.inner.clone()) {
            Ok(dag) => Ok(Self { inner: dag }),
            Err(_) => Err(PyErr::new::<DagHasCycleError, _>(
                "DiGraph contains a cycle",
            )),
        }
    }

    /// Adds a new node to the DAG.
    fn add_node(&mut self) -> usize {
        self.inner.add_node()
    }

    /// Adds a directed edge from source to target.
    ///
    /// Raises `DagWouldCycleError` if the edge would create a cycle.
    fn add_edge(
        &mut self,
        source: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
    ) -> PyResult<usize> {
        let src = self.resolve_node_id(source)?;
        let tgt = self.resolve_node_id(target)?;

        let edge_data = EdgeAttrs::new();
        match self.inner.add_edge_with_data(src, tgt, edge_data) {
            Ok(edge_id) => Ok(edge_id),
            Err(_) => Err(PyErr::new::<DagWouldCycleError, _>(
                "Adding this edge would create a cycle",
            )),
        }
    }

    /// Adds a child node connected from the parent.
    ///
    /// This is more efficient than `add_node` + `add_edge` since no cycle check is needed
    /// (a new node cannot create a cycle).
    ///
    /// Returns (`node_id`, `edge_id`).
    fn add_child(&mut self, parent: usize) -> (usize, usize) {
        use pecos_num::graph::NodeAttrs;
        self.inner
            .add_child(parent, EdgeAttrs::new(), NodeAttrs::default())
    }

    /// Adds a parent node connected to the child.
    ///
    /// This is more efficient than `add_node` + `add_edge` since no cycle check is needed.
    ///
    /// Returns (`node_id`, `edge_id`).
    fn add_parent(&mut self, child: usize) -> (usize, usize) {
        use pecos_num::graph::NodeAttrs;
        self.inner
            .add_parent(child, EdgeAttrs::new(), NodeAttrs::default())
    }

    /// Returns the number of nodes in the DAG.
    fn node_count(&self) -> usize {
        self.inner.node_count()
    }

    /// Returns the number of edges in the DAG.
    fn edge_count(&self) -> usize {
        self.inner.edge_count()
    }

    /// Returns a list of all node indices.
    fn nodes(&self) -> Vec<usize> {
        self.inner.nodes()
    }

    /// Check if a node exists.
    fn has_node(&self, node: usize) -> bool {
        self.inner.nodes().contains(&node)
    }

    /// Remove a node and all its edges.
    fn remove_node(&mut self, node: usize) -> bool {
        self.inner.remove_node(node).is_some()
    }

    /// Returns a list of all edges as (source, target, weight) tuples.
    fn edges(&self) -> Vec<(usize, usize, f64)> {
        self.inner.edges()
    }

    /// Returns the predecessors of a node.
    fn predecessors(&self, node: usize) -> Vec<usize> {
        self.inner.predecessors(node)
    }

    /// Returns the successors of a node.
    fn successors(&self, node: usize) -> Vec<usize> {
        self.inner.successors(node)
    }

    /// Returns the in-degree of a node.
    fn in_degree(&self, node: usize) -> usize {
        self.inner.in_degree(node)
    }

    /// Returns the out-degree of a node.
    fn out_degree(&self, node: usize) -> usize {
        self.inner.out_degree(node)
    }

    /// Returns edge IDs of incoming edges.
    fn in_edges(&self, node: usize) -> Vec<usize> {
        self.inner.in_edges(node)
    }

    /// Returns edge IDs of outgoing edges.
    fn out_edges(&self, node: usize) -> Vec<usize> {
        self.inner.out_edges(node)
    }

    /// Returns a topological ordering of the DAG.
    ///
    /// This always succeeds for a DAG (unlike `DiGraph.topological_sort()`).
    fn topological_sort(&self) -> Vec<usize> {
        self.inner.topological_sort()
    }

    /// Returns True if there is a path from source to target.
    fn has_path(&self, source: usize, target: usize) -> bool {
        self.inner.has_path(source, target)
    }

    /// Returns the root nodes (nodes with no predecessors).
    fn roots(&self) -> Vec<usize> {
        self.inner.roots()
    }

    /// Returns the leaf nodes (nodes with no successors).
    fn leaves(&self) -> Vec<usize> {
        self.inner.leaves()
    }

    /// Returns all ancestors of a node (nodes that can reach this node).
    fn ancestors(&self, node: usize) -> BTreeSet<usize> {
        self.inner.ancestors(node).into_iter().collect()
    }

    /// Returns all descendants of a node (nodes reachable from this node).
    fn descendants(&self, node: usize) -> BTreeSet<usize> {
        self.inner.descendants(node).into_iter().collect()
    }

    /// Returns the depth of the DAG (length of the longest path).
    fn depth(&self) -> usize {
        self.inner.depth()
    }

    /// Returns an iterator over the layers of the DAG.
    ///
    /// A layer is a set of nodes that can be processed in parallel (all their
    /// dependencies are in previous layers).
    ///
    /// # Arguments
    ///
    /// * `first_layer` - The starting nodes (typically `dag.roots()`)
    ///
    /// # Returns
    ///
    /// A list of layers, where each layer is a list of node indices.
    ///
    /// # Examples
    ///
    /// ```python
    /// dag = DAG()
    /// n0 = dag.add_node()
    /// n1 = dag.add_node()
    /// n2 = dag.add_node()
    /// dag.add_edge(n0, n1)
    /// dag.add_edge(n0, n2)
    ///
    /// for layer in dag.layers(dag.roots()):
    ///     print(f"Layer: {layer}")
    /// # Output:
    /// # Layer: [0]
    /// # Layer: [1, 2]
    /// ```
    fn layers(&self, first_layer: Vec<usize>) -> Vec<Vec<usize>> {
        self.inner.layers(first_layer).collect()
    }

    /// Returns the longest path in the DAG as (path, `total_weight`).
    fn longest_path(&self) -> (Vec<usize>, f64) {
        self.inner.longest_path()
    }

    /// Creates a subgraph containing only the specified nodes.
    #[allow(clippy::needless_pass_by_value)]
    fn subgraph(&self, nodes: Vec<usize>) -> Self {
        let subgraph = self.inner.subgraph(&nodes);
        Self { inner: subgraph }
    }

    /// Finds the edge ID between two nodes.
    fn find_edge(&self, source: usize, target: usize) -> Option<usize> {
        self.inner.find_edge(source, target)
    }

    /// Gets the endpoints of an edge by its edge ID.
    fn edge_endpoints(&self, edge_id: usize) -> Option<(usize, usize)> {
        self.inner.edge_endpoints(edge_id)
    }

    /// Gets the weight of an edge by its edge ID.
    fn edge_weight(&self, edge_id: usize) -> f64 {
        self.inner.edge_weight(edge_id)
    }

    /// Sets the weight of an edge by its edge ID.
    fn set_edge_weight(&mut self, edge_id: usize, weight: f64) {
        self.inner.set_edge_weight(edge_id, weight);
    }

    /// Gets the weight of an edge between two nodes.
    fn get_weight(
        &self,
        source: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
    ) -> PyResult<Option<f64>> {
        let src = self.resolve_node_id(source)?;
        let tgt = self.resolve_node_id(target)?;
        Ok(self.inner.get_weight(src, tgt))
    }

    /// Sets the weight of an edge between two nodes.
    fn set_weight(
        &mut self,
        source: &Bound<'_, PyAny>,
        target: &Bound<'_, PyAny>,
        weight: f64,
    ) -> PyResult<()> {
        let src = self.resolve_node_id(source)?;
        let tgt = self.resolve_node_id(target)?;
        self.inner.set_weight(src, tgt, weight);
        Ok(())
    }

    /// Removes an edge by its edge ID.
    fn remove_edge(&mut self, edge_id: usize) -> bool {
        self.inner.remove_edge(edge_id).is_some()
    }

    /// Gets edge data between two nodes.
    fn get_edge_data(&self, py: Python<'_>, source: usize, target: usize) -> Option<Py<PyAny>> {
        self.inner
            .get_edge_data(source, target)
            .map(|edge_attrs: EdgeAttrs| {
                let dict = PyDict::new(py);
                dict.set_item("weight", edge_attrs.weight()).unwrap();
                for (key, value) in edge_attrs.attrs() {
                    let py_value = attribute_to_python(py, value).unwrap();
                    dict.set_item(key, py_value).unwrap();
                }
                dict.into()
            })
    }

    /// Returns a mutable view of edge attributes.
    fn edge_attrs(slf: Py<Self>, source: usize, target: usize) -> PyDagEdgeAttrsView {
        PyDagEdgeAttrsView {
            graph: slf,
            source,
            target,
        }
    }

    /// Returns a mutable view of node attributes.
    fn node_attrs(slf: Py<Self>, node: usize) -> PyDagNodeAttrsView {
        PyDagNodeAttrsView { graph: slf, node }
    }

    /// Returns a mutable view of graph-level attributes.
    fn attrs(slf: Py<Self>) -> PyDagGraphAttrsView {
        PyDagGraphAttrsView { graph: slf }
    }

    fn __repr__(&self) -> String {
        format!(
            "DAG(nodes={}, edges={})",
            self.inner.node_count(),
            self.inner.edge_count()
        )
    }
}

/// Mutable view into DAG edge attributes.
#[pyclass(name = "DagEdgeAttrsView", module = "pecos_rslib.graph")]
pub struct PyDagEdgeAttrsView {
    graph: Py<PyDAG>,
    source: usize,
    target: usize,
}

#[pymethods]
impl PyDagEdgeAttrsView {
    fn __setitem__(&self, py: Python<'_>, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        let attr = python_value_to_attribute(value, &key)?;
        let attrs: &mut BTreeMap<String, Attribute> =
            match graph.inner.edge_attrs_mut(self.source, self.target) {
                Some(a) => a,
                None => {
                    return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Edge does not exist",
                    ));
                }
            };
        attrs.insert(key, attr);
        Ok(())
    }

    fn __getitem__(&self, py: Python<'_>, key: String) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs: &BTreeMap<String, Attribute> =
            match graph.inner.edge_attrs(self.source, self.target) {
                Some(a) => a,
                None => {
                    return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                        "Edge does not exist",
                    ));
                }
            };
        if let Some(attr) = attrs.get(&key) {
            attribute_to_python(py, attr)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(key))
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get(
        &self,
        py: Python<'_>,
        key: &str,
        default: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs: Option<&BTreeMap<String, Attribute>> =
            graph.inner.edge_attrs(self.source, self.target);
        if let Some(attrs) = attrs {
            if let Some(attr) = attrs.get(key) {
                attribute_to_python(py, attr)
            } else if let Some(def) = default {
                Ok(def.clone().unbind())
            } else {
                Ok(py.None())
            }
        } else if let Some(def) = default {
            Ok(def.clone().unbind())
        } else {
            Ok(py.None())
        }
    }
}

/// Mutable view into DAG node attributes.
#[pyclass(name = "DagNodeAttrsView", module = "pecos_rslib.graph")]
pub struct PyDagNodeAttrsView {
    graph: Py<PyDAG>,
    node: usize,
}

#[pymethods]
impl PyDagNodeAttrsView {
    fn __setitem__(&self, py: Python<'_>, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        let attrs: &mut BTreeMap<String, Attribute> = match graph.inner.node_attrs_mut(self.node) {
            Some(a) => a,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    "Node does not exist",
                ));
            }
        };
        let attr = python_value_to_attribute(value, &key)?;
        attrs.insert(key, attr);
        Ok(())
    }

    fn __getitem__(&self, py: Python<'_>, key: String) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs: &BTreeMap<String, Attribute> = match graph.inner.node_attrs(self.node) {
            Some(a) => a,
            None => {
                return Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(
                    "Node does not exist",
                ));
            }
        };
        if let Some(attr) = attrs.get(&key) {
            attribute_to_python(py, attr)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(key))
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs: Option<&BTreeMap<String, Attribute>> = graph.inner.node_attrs(self.node);
        if let Some(attrs) = attrs {
            if let Some(attr) = attrs.get(key) {
                attribute_to_python(py, attr)
            } else {
                Ok(default.unwrap_or_else(|| py.None()))
            }
        } else {
            Ok(default.unwrap_or_else(|| py.None()))
        }
    }
}

/// Mutable view into DAG-level attributes.
#[pyclass(name = "DagGraphAttrsView", module = "pecos_rslib.graph")]
pub struct PyDagGraphAttrsView {
    graph: Py<PyDAG>,
}

#[pymethods]
impl PyDagGraphAttrsView {
    fn __setitem__(&self, py: Python<'_>, key: String, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let mut graph = self.graph.borrow_mut(py);
        let attr = python_value_to_attribute(value, &key)?;
        let attrs: &mut BTreeMap<String, Attribute> = graph.inner.attrs_mut();
        attrs.insert(key, attr);
        Ok(())
    }

    fn __getitem__(&self, py: Python<'_>, key: String) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs: &BTreeMap<String, Attribute> = graph.inner.attrs();
        if let Some(attr) = attrs.get(&key) {
            attribute_to_python(py, attr)
        } else {
            Err(PyErr::new::<pyo3::exceptions::PyKeyError, _>(key))
        }
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<Py<PyAny>>) -> PyResult<Py<PyAny>> {
        let graph = self.graph.borrow(py);
        let attrs: &BTreeMap<String, Attribute> = graph.inner.attrs();
        if let Some(attr) = attrs.get(key) {
            attribute_to_python(py, attr)
        } else {
            Ok(default.unwrap_or_else(|| py.None()))
        }
    }
}

/// Register the graph module with Python.
///
/// This function is called from the main module registration to expose the graph
/// functionality to Python. This creates a `graph` submodule accessible as `pecos_rslib.graph`.
pub fn register_graph_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create a graph submodule
    let py = parent_module.py();
    let graph_module = PyModule::new(py, "graph")?;

    // Add undirected Graph classes
    graph_module.add_class::<PyEdgeAttrsView>()?;
    graph_module.add_class::<PyNodeAttrsView>()?;
    graph_module.add_class::<PyGraphAttrsView>()?;
    graph_module.add_class::<PyGraph>()?;

    // Add DiGraph classes
    graph_module.add_class::<PyDiGraph>()?;
    graph_module.add_class::<PyDiGraphEdgeAttrsView>()?;
    graph_module.add_class::<PyDiGraphNodeAttrsView>()?;
    graph_module.add_class::<PyDiGraphAttrsView>()?;

    // Add DAG classes
    graph_module.add_class::<PyDAG>()?;
    graph_module.add_class::<PyDagEdgeAttrsView>()?;
    graph_module.add_class::<PyDagNodeAttrsView>()?;
    graph_module.add_class::<PyDagGraphAttrsView>()?;

    // Add DAG exceptions to graph module
    graph_module.add("DagWouldCycleError", py.get_type::<DagWouldCycleError>())?;
    graph_module.add("DagHasCycleError", py.get_type::<DagHasCycleError>())?;

    // Add the submodule to the parent module
    parent_module.add_submodule(&graph_module)?;

    // Register in sys.modules for `import pecos_rslib.graph` support
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.graph", &graph_module)?;

    // Also add classes to parent module for direct import (e.g., from pecos_rslib import Graph)
    parent_module.add_class::<PyEdgeAttrsView>()?;
    parent_module.add_class::<PyNodeAttrsView>()?;
    parent_module.add_class::<PyGraphAttrsView>()?;
    parent_module.add_class::<PyGraph>()?;
    parent_module.add_class::<PyDiGraph>()?;
    parent_module.add_class::<PyDAG>()?;

    // Add DAG exceptions to parent module for direct import
    parent_module.add("DagWouldCycleError", py.get_type::<DagWouldCycleError>())?;
    parent_module.add("DagHasCycleError", py.get_type::<DagHasCycleError>())?;

    Ok(())
}
