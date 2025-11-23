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

use pecos::graph::{
    EdgeAttribute as RustEdgeAttribute, Graph as RustGraph, MappedGraph as RustMappedGraph,
};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::collections::BTreeMap;

/// Type alias for edge list return type to reduce complexity
type PyEdgeList = Vec<(Py<PyAny>, Py<PyAny>, f64)>;

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
#[pyclass(name = "Graph", module = "pecos_rslib.graph")]
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

    /// Adds an edge between two nodes with optional weight and attributes.
    ///
    /// This method supports NetworkX-style edge addition with keyword arguments.
    ///
    /// # Arguments
    ///
    /// * `a` - Index of the first node
    /// * `b` - Index of the second node
    /// * `weight` - Optional weight of the edge (defaults to 1.0)
    /// * `**kwargs` - Additional edge attributes
    ///
    /// # Examples
    ///
    /// ```python
    /// graph.add_edge(0, 1, weight=5.0)
    /// graph.add_edge(0, 1, weight=5.0, data_path=[1, 2, 3])
    /// ```
    #[pyo3(signature = (a, b, weight=None, **kwargs))]
    fn add_edge(
        &mut self,
        a: usize,
        b: usize,
        weight: Option<f64>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        use pecos::graph::EdgeAttribute;

        // Create edge data from kwargs
        let mut edge_data = pecos::graph::EdgeData::new();

        // Add weight (use provided weight, or from kwargs, or default to 1.0)
        let final_weight = if let Some(w) = weight {
            w
        } else if let Some(kw) = kwargs {
            if let Some(w) = kw.get_item("weight")? {
                w.extract::<f64>().unwrap_or(1.0)
            } else {
                1.0
            }
        } else {
            1.0
        };
        edge_data.set("weight", EdgeAttribute::Float(final_weight));

        // Add other attributes from kwargs
        if let Some(kw) = kwargs {
            for (key, value) in kw.iter() {
                let key_str: String = key.extract()?;

                // Skip "weight" as we already handled it
                if key_str == "weight" {
                    continue;
                }

                // Try to convert value to EdgeAttribute
                // Check bool first because bool can be extracted as float (True -> 1.0)
                if let Ok(b) = value.extract::<bool>() {
                    edge_data.set(&key_str, EdgeAttribute::Bool(b));
                } else if let Ok(i) = value.extract::<i64>() {
                    edge_data.set(&key_str, EdgeAttribute::Int(i));
                } else if let Ok(f) = value.extract::<f64>() {
                    edge_data.set(&key_str, EdgeAttribute::Float(f));
                } else if let Ok(v) = value.extract::<Vec<i64>>() {
                    edge_data.set(&key_str, EdgeAttribute::IntList(v));
                } else if let Ok(s) = value.extract::<String>() {
                    edge_data.set(&key_str, EdgeAttribute::String(s));
                } else {
                    // Unsupported attribute type - provide helpful error
                    let type_name = value
                        .get_type()
                        .name()
                        .ok()
                        .and_then(|n| n.to_str().ok().map(std::string::ToString::to_string))
                        .unwrap_or_else(|| "<unknown>".to_string());
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Unsupported edge attribute type for key '{key_str}'. \
                         Supported types: bool, int, float, str, list[int]. \
                         Got type: {type_name}"
                    )));
                }
            }
        }

        self.inner.add_edge_with_data(a, b, edge_data);
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
    /// A dictionary with edge attributes if an edge exists, None otherwise.
    fn get_edge_data(&self, py: Python<'_>, a: usize, b: usize) -> Option<Py<PyAny>> {
        self.inner.get_edge_data(a, b).map(|edge_data| {
            let dict = PyDict::new(py);
            for (key, value) in edge_data.attributes() {
                match value {
                    RustEdgeAttribute::Float(f) => {
                        dict.set_item(key, f).unwrap();
                    }
                    RustEdgeAttribute::Int(i) => {
                        dict.set_item(key, i).unwrap();
                    }
                    RustEdgeAttribute::String(s) => {
                        dict.set_item(key, s.as_str()).unwrap();
                    }
                    RustEdgeAttribute::Bool(b) => {
                        dict.set_item(key, b).unwrap();
                    }
                    RustEdgeAttribute::IntList(v) => {
                        dict.set_item(key, v.clone()).unwrap();
                    }
                }
            }
            dict.into()
        })
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

    /// Returns a string representation of the graph.
    fn __repr__(&self) -> String {
        format!(
            "Graph(nodes={}, edges={})",
            self.inner.node_count(),
            self.inner.edge_count()
        )
    }
}

/// Node ID type that supports both integers and strings.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Ord, PartialOrd)]
enum NodeId {
    Int(i64),
    Str(String),
}

impl NodeId {
    /// Converts a Python object to a `NodeId`.
    fn from_py(obj: &Bound<'_, PyAny>) -> PyResult<Self> {
        use pyo3::types::PyAnyMethods;

        if let Ok(i) = obj.extract::<i64>() {
            Ok(NodeId::Int(i))
        } else if let Ok(s) = obj.extract::<String>() {
            Ok(NodeId::Str(s))
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "Node ID must be an integer or string",
            ))
        }
    }

    /// Converts a `NodeId` to a Python object.
    fn to_py(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        match self {
            NodeId::Int(i) => Ok((*i).into_pyobject(py)?.into_any().unbind()),
            NodeId::Str(s) => Ok(s.as_str().into_pyobject(py)?.into_any().unbind()),
        }
    }
}

/// Python wrapper for `MappedGraph` with NetworkX-style node IDs.
///
/// This class provides NetworkX-compatible graph functionality where nodes
/// can be identified by either integers or strings, unlike Graph which only
/// supports integer indices.
///
/// # Examples (Python)
///
/// ```python
/// import pecos_rslib
///
/// # Create a graph with string node IDs
/// graph = pecos_rslib.graph.MappedGraph()
///
/// # Add edges with string nodes
/// graph.add_edge('v1', 'v2', weight=5.0, data_path=[1, 2, 3])
/// graph.add_edge('v2', 'v3', weight=3.0)
/// graph.add_edge(0, 'v1', weight=2.0)  # Mixed types work too
///
/// # Compute maximum weight matching
/// matching = graph.max_weight_matching()
/// ```
#[pyclass(name = "MappedGraph", module = "pecos_rslib.graph")]
#[derive(Clone)]
pub struct PyMappedGraph {
    /// The underlying Rust mapped graph
    inner: RustMappedGraph<NodeId>,
}

#[pymethods]
impl PyMappedGraph {
    /// Creates a new empty mapped graph.
    ///
    /// # Returns
    ///
    /// A new empty `MappedGraph` instance.
    #[new]
    fn new() -> Self {
        Self {
            inner: RustMappedGraph::new(),
        }
    }

    /// Creates a new mapped graph with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `nodes` - Expected number of nodes
    /// * `edges` - Expected number of edges
    ///
    /// # Returns
    ///
    /// A new `MappedGraph` instance with pre-allocated capacity.
    #[staticmethod]
    fn with_capacity(nodes: usize, edges: usize) -> Self {
        Self {
            inner: RustMappedGraph::with_capacity(nodes, edges),
        }
    }

    /// Adds an edge between two nodes with optional weight and attributes.
    ///
    /// Nodes can be integers or strings and will be added automatically if they don't exist.
    ///
    /// # Arguments
    ///
    /// * `a` - Node ID (integer or string) for the first node
    /// * `b` - Node ID (integer or string) for the second node
    /// * `weight` - Optional weight of the edge (defaults to 1.0)
    /// * `**kwargs` - Additional edge attributes
    ///
    /// # Examples
    ///
    /// ```python
    /// graph.add_edge('v1', 'v2', weight=5.0)
    /// graph.add_edge(0, 1, weight=5.0, data_path=[1, 2, 3])
    /// graph.add_edge('v1', 0, data_path=[1, 2, 3])  # Mixed types
    /// ```
    #[pyo3(signature = (a, b, weight=None, **kwargs))]
    fn add_edge(
        &mut self,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
        weight: Option<f64>,
        kwargs: Option<&Bound<'_, PyDict>>,
    ) -> PyResult<()> {
        use pecos::graph::EdgeAttribute;

        // Convert Python objects to NodeIds
        let node_a = NodeId::from_py(a)?;
        let node_b = NodeId::from_py(b)?;

        // Create edge data from kwargs
        let mut edge_data = pecos::graph::EdgeData::new();

        // Add weight (use provided weight, or from kwargs, or default to 1.0)
        let final_weight = if let Some(w) = weight {
            w
        } else if let Some(kw) = kwargs {
            if let Some(w) = kw.get_item("weight")? {
                w.extract::<f64>().unwrap_or(1.0)
            } else {
                1.0
            }
        } else {
            1.0
        };
        edge_data.set("weight", EdgeAttribute::Float(final_weight));

        // Add other attributes from kwargs
        if let Some(kw) = kwargs {
            for (key, value) in kw.iter() {
                let key_str: String = key.extract()?;

                // Skip "weight" as we already handled it
                if key_str == "weight" {
                    continue;
                }

                // Try to convert value to EdgeAttribute
                // Check bool first because bool can be extracted as float (True -> 1.0)
                if let Ok(b) = value.extract::<bool>() {
                    edge_data.set(&key_str, EdgeAttribute::Bool(b));
                } else if let Ok(i) = value.extract::<i64>() {
                    edge_data.set(&key_str, EdgeAttribute::Int(i));
                } else if let Ok(f) = value.extract::<f64>() {
                    edge_data.set(&key_str, EdgeAttribute::Float(f));
                } else if let Ok(v) = value.extract::<Vec<i64>>() {
                    edge_data.set(&key_str, EdgeAttribute::IntList(v));
                } else if let Ok(s) = value.extract::<String>() {
                    edge_data.set(&key_str, EdgeAttribute::String(s));
                } else {
                    // Unsupported attribute type - provide helpful error
                    let type_name = value
                        .get_type()
                        .name()
                        .ok()
                        .and_then(|n| n.to_str().ok().map(std::string::ToString::to_string))
                        .unwrap_or_else(|| "<unknown>".to_string());
                    return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                        "Unsupported edge attribute type for key '{key_str}'. \
                         Supported types: bool, int, float, str, list[int]. \
                         Got type: {type_name}"
                    )));
                }
            }
        }

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

    /// Returns a list of all node IDs in the graph.
    ///
    /// # Returns
    ///
    /// A list containing all node IDs (integers or strings).
    fn nodes(&self, py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
        self.inner.nodes().iter().map(|n| n.to_py(py)).collect()
    }

    /// Computes the maximum weight matching of the graph.
    ///
    /// # Arguments
    ///
    /// * `max_cardinality` - If True, prioritize maximum cardinality over maximum weight
    /// * `weight_multiplier` - Optional multiplier for float-to-integer weight conversion (default: 1000.0)
    ///
    /// # Returns
    ///
    /// A dictionary mapping node IDs to their matched partners.
    #[pyo3(signature = (max_cardinality=false, weight_multiplier=1000.0))]
    fn max_weight_matching(
        &self,
        py: Python<'_>,
        max_cardinality: bool,
        weight_multiplier: f64,
    ) -> PyResult<Py<PyDict>> {
        let matching = self
            .inner
            .max_weight_matching_with_precision(max_cardinality, weight_multiplier);
        let dict = PyDict::new(py);

        for (k, v) in matching {
            dict.set_item(k.to_py(py)?, v.to_py(py)?)?;
        }

        Ok(dict.unbind())
    }

    /// Returns a list of all edges as (source, target, weight) tuples.
    ///
    /// # Returns
    ///
    /// A list of tuples (source, target, weight) for all edges in the graph.
    fn edges(&self, py: Python<'_>) -> PyResult<PyEdgeList> {
        self.inner
            .edges()
            .into_iter()
            .map(|(a, b, w)| Ok((a.to_py(py)?, b.to_py(py)?, w)))
            .collect()
    }

    /// Gets the edge data between two nodes.
    ///
    /// # Arguments
    ///
    /// * `a` - ID of the first node
    /// * `b` - ID of the second node
    ///
    /// # Returns
    ///
    /// A dictionary with edge attributes if an edge exists, None otherwise.
    fn get_edge_data(
        &self,
        py: Python<'_>,
        a: &Bound<'_, PyAny>,
        b: &Bound<'_, PyAny>,
    ) -> PyResult<Option<Py<PyAny>>> {
        let node_a = NodeId::from_py(a)?;
        let node_b = NodeId::from_py(b)?;

        Ok(self.inner.get_edge_data(&node_a, &node_b).map(|edge_data| {
            let dict = PyDict::new(py);
            for (key, value) in edge_data.attributes() {
                match value {
                    RustEdgeAttribute::Float(f) => {
                        dict.set_item(key, f).unwrap();
                    }
                    RustEdgeAttribute::Int(i) => {
                        dict.set_item(key, i).unwrap();
                    }
                    RustEdgeAttribute::String(s) => {
                        dict.set_item(key, s.as_str()).unwrap();
                    }
                    RustEdgeAttribute::Bool(b) => {
                        dict.set_item(key, b).unwrap();
                    }
                    RustEdgeAttribute::IntList(v) => {
                        dict.set_item(key, v.clone()).unwrap();
                    }
                }
            }
            dict.into()
        }))
    }

    /// Creates a subgraph containing only the specified nodes.
    ///
    /// # Arguments
    ///
    /// * `nodes` - A list of node IDs to include in the subgraph
    ///
    /// # Returns
    ///
    /// A new `MappedGraph` containing only the specified nodes and edges between them.
    fn subgraph(&self, nodes: Bound<'_, PyAny>) -> PyResult<Self> {
        use pyo3::types::PyList;

        // Use downcast to get the type name before attempting cast
        let type_name = nodes
            .get_type()
            .name()
            .ok()
            .and_then(|n| n.to_str().ok().map(std::string::ToString::to_string))
            .unwrap_or_else(|| "<unknown>".to_string());

        // Use cast_into to convert Bound<PyAny> to Bound<PyList>
        let py_list = nodes.cast_into::<PyList>().map_err(|e| {
            PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Expected a list of nodes, got {type_name}: {e}"
            ))
        })?;

        // Convert each element to NodeId with helpful error messages
        let node_ids: Result<Vec<NodeId>, PyErr> = py_list
            .iter()
            .enumerate()
            .map(|(idx, item)| {
                NodeId::from_py(&item).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Failed to convert node at index {idx} to a valid node identifier: {e}"
                    ))
                })
            })
            .collect();
        let node_ids = node_ids?;

        Ok(Self {
            inner: self.inner.subgraph(&node_ids),
        })
    }

    /// Computes single-source shortest paths using Dijkstra's algorithm.
    ///
    /// # Arguments
    ///
    /// * `source` - The source node ID
    ///
    /// # Returns
    ///
    /// A dictionary mapping each reachable node to a list of node IDs representing
    /// the shortest path from the source to that node.
    fn single_source_shortest_path(
        &self,
        py: Python<'_>,
        source: &Bound<'_, PyAny>,
    ) -> PyResult<Py<PyDict>> {
        let source_id = NodeId::from_py(source)?;
        let paths = self.inner.single_source_shortest_path(&source_id);

        let dict = PyDict::new(py);
        for (target, path) in paths {
            let path_list: Result<Vec<Py<PyAny>>, _> = path.iter().map(|n| n.to_py(py)).collect();
            dict.set_item(target.to_py(py)?, path_list?)?;
        }

        Ok(dict.unbind())
    }

    /// Returns a string representation of the graph.
    fn __repr__(&self) -> String {
        format!(
            "MappedGraph(nodes={}, edges={})",
            self.inner.node_count(),
            self.inner.edge_count()
        )
    }
}

/// Register the graph module with Python.
///
/// This function is called from the main module registration to expose the graph
/// functionality to Python. This creates a `graph` submodule accessible as `pecos_rslib.graph`.
pub fn register_graph_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    // Create graph submodule
    let graph_module = PyModule::new(parent_module.py(), "graph")?;

    // Add the Graph and MappedGraph classes to the graph submodule
    graph_module.add_class::<PyGraph>()?;
    graph_module.add_class::<PyMappedGraph>()?;

    // Add the graph module to the parent module
    parent_module.add_submodule(&graph_module)?;

    Ok(())
}
