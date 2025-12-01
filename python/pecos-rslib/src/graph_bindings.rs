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

use pecos::graph::{Attribute as RustAttribute, Graph as RustGraph};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;

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
/// import _pecos_rslib
///
/// # Create a new graph
/// graph = _pecos_rslib.graph.Graph()
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
#[pyclass(name = "Graph", module = "_pecos_rslib.graph")]
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
        let edge_data = pecos::graph::EdgeAttrs::new();

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

    // TODO: Add remove_node to Rust Graph API
    // /// Remove a node and all its connected edges from the graph.
    // fn remove_node(&mut self, node: usize) {
    //     self.inner.remove_node(node);
    // }

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
#[pyclass(name = "EdgeAttrsView", module = "_pecos_rslib.graph")]
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
#[pyclass(name = "NodeAttrsView", module = "_pecos_rslib.graph")]
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
#[pyclass(name = "GraphAttrsView", module = "_pecos_rslib.graph")]
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

/// Register the graph module with Python.
///
/// This function is called from the main module registration to expose the graph
/// functionality to Python. This creates a `graph` submodule accessible as `_pecos_rslib.graph`.
pub fn register_graph_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create a graph submodule
    let py = parent_module.py();
    let graph_module = PyModule::new(py, "graph")?;

    // Add classes to the graph submodule
    graph_module.add_class::<PyEdgeAttrsView>()?;
    graph_module.add_class::<PyNodeAttrsView>()?;
    graph_module.add_class::<PyGraphAttrsView>()?;
    graph_module.add_class::<PyGraph>()?;

    // Add the submodule to the parent module
    parent_module.add_submodule(&graph_module)?;

    // Register in sys.modules for `import __pecos_rslib.graph` support
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("_pecos_rslib.graph", &graph_module)?;

    // Also add classes to parent module for direct import (e.g., from _pecos_rslib import Graph)
    parent_module.add_class::<PyEdgeAttrsView>()?;
    parent_module.add_class::<PyNodeAttrsView>()?;
    parent_module.add_class::<PyGraphAttrsView>()?;
    parent_module.add_class::<PyGraph>()?;

    Ok(())
}
