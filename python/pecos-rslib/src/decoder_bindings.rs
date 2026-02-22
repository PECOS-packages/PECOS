// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Python bindings for PECOS decoders.
//!
//! This module provides Python bindings for quantum error correction decoders,
//! including `PyMatching`, Fusion Blossom, LDPC decoders, and more.
//!
//! # API Design
//!
//! The API is designed to be:
//! - **Consistent**: All decoders have similar construction patterns and decode methods
//! - **Familiar**: Inspired by original library APIs (`PyMatching`, ldpc, fusion-blossom)
//! - **Unified**: Common result types where appropriate
//!
//! # Decoder Categories
//!
//! ## MWPM Decoders (Minimum Weight Perfect Matching)
//! - `PyMatchingDecoder` - Fast MWPM using `PyMatching` library
//! - `FusionBlossomDecoder` - Pure Rust MWPM implementation
//!
//! ## LDPC Decoders (Low-Density Parity Check)
//! - `BpOsdDecoder` - Belief Propagation + Ordered Statistics Decoding
//! - `BpLsdDecoder` - Belief Propagation + Localized Statistics Decoding
//! - `UnionFindDecoder` - Union-Find based decoder

use ndarray::{Array1, Array2};
use pyo3::prelude::*;

// =============================================================================
// Common Result Types
// =============================================================================

/// Result from MWPM (Minimum Weight Perfect Matching) decoders.
///
/// This unified result type is returned by both `PyMatching` and Fusion Blossom decoders.
///
/// # Attributes
///
/// * `correction` - The decoded correction/observable flip (list of 0/1 for each observable)
/// * `weight` - Total weight of the matching (lower is better)
///
/// # Example
///
/// ```python
/// result = decoder.decode(syndrome)
/// if result.weight < threshold:
///     apply_correction(result.correction)
/// ```
#[pyclass(
    name = "MwpmResult",
    module = "pecos_rslib.decoders",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyMwpmResult {
    /// The decoded correction (observable flips)
    correction_data: Vec<u8>,
    /// Total weight of the matching
    #[pyo3(get)]
    weight: f64,
}

#[pymethods]
impl PyMwpmResult {
    /// The decoded correction (observable flips) as a Python list.
    #[getter]
    fn correction(&self) -> Vec<i32> {
        self.correction_data.iter().map(|&x| i32::from(x)).collect()
    }

    /// Get the correction as a list (alias for correction attribute).
    ///
    /// This mirrors `PyMatching`'s `decode()` return value.
    fn to_list(&self) -> Vec<i32> {
        self.correction()
    }

    fn __repr__(&self) -> String {
        format!(
            "MwpmResult(correction={:?}, weight={:.4})",
            self.correction_data, self.weight
        )
    }

    fn __len__(&self) -> usize {
        self.correction_data.len()
    }

    fn __getitem__(&self, idx: usize) -> PyResult<i32> {
        self.correction_data
            .get(idx)
            .map(|&x| i32::from(x))
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("index out of range"))
    }
}

/// Result from LDPC (Belief Propagation) decoders.
///
/// # Attributes
///
/// * `decoding` - The decoded error vector
/// * `converged` - Whether BP converged before max iterations
/// * `iterations` - Number of BP iterations performed
///
/// # Example
///
/// ```python
/// result = decoder.decode(syndrome)
/// if result.converged:
///     error_estimate = result.decoding
/// ```
#[pyclass(
    name = "BpResult",
    module = "pecos_rslib.decoders",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyBpResult {
    /// The decoded error vector
    decoding_data: Vec<u8>,
    /// Whether the decoder converged
    #[pyo3(get)]
    converged: bool,
    /// Number of iterations performed
    #[pyo3(get)]
    iterations: usize,
}

#[pymethods]
impl PyBpResult {
    /// The decoded error vector as a Python list.
    #[getter]
    fn decoding(&self) -> Vec<i32> {
        self.decoding_data.iter().map(|&x| i32::from(x)).collect()
    }

    /// Get the decoding as a list.
    fn to_list(&self) -> Vec<i32> {
        self.decoding()
    }

    fn __repr__(&self) -> String {
        format!(
            "BpResult(converged={}, iterations={}, decoding_len={})",
            self.converged,
            self.iterations,
            self.decoding_data.len()
        )
    }

    fn __len__(&self) -> usize {
        self.decoding_data.len()
    }

    fn __getitem__(&self, idx: usize) -> PyResult<i32> {
        self.decoding_data
            .get(idx)
            .map(|&x| i32::from(x))
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyIndexError, _>("index out of range"))
    }
}

// =============================================================================
// PyMatching Decoder
// =============================================================================

use pecos::decoders::{
    CheckMatrix as RustCheckMatrix, CheckMatrixConfig as RustCheckMatrixConfig,
    PyMatchingConfig as RustPyMatchingConfig, PyMatchingDecoder as RustPyMatchingDecoder,
};

/// Sparse check matrix for MWPM decoders.
///
/// Represents a parity check matrix H where each column corresponds to an error
/// and each row corresponds to a check/detector. For MWPM decoders, each column
/// should have at most 2 non-zero entries.
///
/// # Construction
///
/// ```python
/// from pecos_rslib.decoders import CheckMatrix
///
/// # From dense matrix (like PyMatching)
/// H = [[1, 1, 0], [0, 1, 1]]
/// matrix = CheckMatrix.from_dense(H)
///
/// # From COO format
/// matrix = CheckMatrix(rows=2, cols=3,
///                      row_indices=[0, 0, 1, 1],
///                      col_indices=[0, 1, 1, 2])
///
/// # With weights (like PyMatching's weights parameter)
/// matrix = CheckMatrix.from_dense(H).with_weights([1.0, 2.0, 1.0])
/// ```
#[pyclass(
    name = "CheckMatrix",
    module = "pecos_rslib.decoders",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyCheckMatrix {
    inner: RustCheckMatrix,
}

#[pymethods]
impl PyCheckMatrix {
    /// Create a check matrix from COO (Coordinate) format.
    ///
    /// # Arguments
    ///
    /// * `rows` - Number of rows (checks/detectors)
    /// * `cols` - Number of columns (errors/qubits)
    /// * `row_indices` - Row indices of non-zero entries
    /// * `col_indices` - Column indices of non-zero entries
    #[new]
    #[pyo3(signature = (rows, cols, row_indices, col_indices))]
    fn new(rows: usize, cols: usize, row_indices: Vec<usize>, col_indices: Vec<usize>) -> Self {
        Self {
            inner: RustCheckMatrix::new(rows, cols, row_indices, col_indices),
        }
    }

    /// Create from a dense 2D matrix.
    ///
    /// This mirrors `PyMatching`'s Matching(H) constructor.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Dense matrix as list of lists (rows x cols)
    ///
    /// # Example
    ///
    /// ```python
    /// # Repetition code check matrix
    /// H = [[1, 1, 0], [0, 1, 1]]
    /// matrix = CheckMatrix.from_dense(H)
    /// ```
    #[staticmethod]
    fn from_dense(matrix: Vec<Vec<u8>>) -> PyResult<Self> {
        RustCheckMatrix::from_dense_vec(&matrix)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Set weights for each column (error).
    ///
    /// This mirrors `PyMatching`'s weights parameter.
    ///
    /// # Arguments
    ///
    /// * `weights` - Weight for each column (length must equal cols)
    ///
    /// # Returns
    ///
    /// A new `CheckMatrix` with weights set.
    fn with_weights(&self, weights: Vec<f64>) -> PyResult<Self> {
        self.inner
            .clone()
            .with_weights(weights)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))
    }

    /// Number of rows (checks/detectors).
    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows()
    }

    /// Number of columns (errors/qubits).
    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols()
    }

    /// Number of non-zero entries.
    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    /// Get weights if set, None otherwise.
    fn weights(&self) -> Option<Vec<f64>> {
        self.inner.weights().map(<[f64]>::to_vec)
    }

    fn __repr__(&self) -> String {
        format!(
            "CheckMatrix(rows={}, cols={}, nnz={})",
            self.inner.rows(),
            self.inner.cols(),
            self.inner.nnz()
        )
    }
}

/// `PyMatching` MWPM decoder.
///
/// Fast minimum-weight perfect matching decoder using the `PyMatching` library.
/// This is the recommended MWPM decoder for most use cases.
///
/// # Construction
///
/// ```python
/// from pecos_rslib.decoders import PyMatchingDecoder, CheckMatrix
///
/// # From check matrix (like PyMatching's Matching(H))
/// H = [[1, 1, 0], [0, 1, 1]]
/// decoder = PyMatchingDecoder.from_check_matrix(CheckMatrix.from_dense(H))
///
/// # From Stim detector error model
/// decoder = PyMatchingDecoder.from_dem(dem_string)
///
/// # Manual graph construction (like PyMatching's add_edge)
/// decoder = PyMatchingDecoder(num_nodes=4)
/// decoder.add_edge(0, 1, observables=[0], weight=1.0)
/// decoder.add_boundary_edge(0, observables=[0])
/// ```
///
/// # Decoding
///
/// ```python
/// syndrome = [1, 0]  # Detection events
/// result = decoder.decode(syndrome)
/// print(f"Correction: {result.correction}, Weight: {result.weight}")
/// ```
// Note: unsendable because contains FFI pointers (cxx UniquePtr)
#[pyclass(
    name = "PyMatchingDecoder",
    module = "pecos_rslib.decoders",
    unsendable
)]
pub struct PyPyMatchingDecoder {
    inner: RustPyMatchingDecoder,
}

#[pymethods]
impl PyPyMatchingDecoder {
    /// Create decoder for manual graph construction.
    ///
    /// Use `add_edge()` and `add_boundary_edge()` to build the matching graph.
    ///
    /// # Arguments
    ///
    /// * `num_nodes` - Number of detector nodes
    /// * `num_observables` - Number of logical observables (default: 64)
    #[new]
    #[pyo3(signature = (num_nodes, num_observables=64))]
    fn new(num_nodes: usize, num_observables: usize) -> PyResult<Self> {
        let config = RustPyMatchingConfig {
            num_nodes: Some(num_nodes),
            num_observables,
            num_neighbours: None,
        };

        RustPyMatchingDecoder::new(config)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Create decoder from a check matrix.
    ///
    /// This mirrors `PyMatching`'s `Matching(H)` constructor.
    ///
    /// # Arguments
    ///
    /// * `check_matrix` - The parity check matrix
    ///
    /// # Example
    ///
    /// ```python
    /// H = CheckMatrix.from_dense([[1, 1, 0], [0, 1, 1]])
    /// decoder = PyMatchingDecoder.from_check_matrix(H)
    /// ```
    #[staticmethod]
    fn from_check_matrix(check_matrix: &PyCheckMatrix) -> PyResult<Self> {
        RustPyMatchingDecoder::from_check_matrix(&check_matrix.inner)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Create decoder from check matrix with additional configuration.
    ///
    /// # Arguments
    ///
    /// * `check_matrix` - The parity check matrix
    /// * `repetitions` - Number of syndrome measurement rounds (for 3D matching)
    /// * `timelike_weights` - Weights for timelike edges between rounds
    /// * `use_virtual_boundary` - Whether to use virtual boundary nodes
    #[staticmethod]
    #[pyo3(signature = (check_matrix, repetitions=1, timelike_weights=None, use_virtual_boundary=true))]
    fn from_check_matrix_with_repetitions(
        check_matrix: &PyCheckMatrix,
        repetitions: usize,
        timelike_weights: Option<Vec<f64>>,
        use_virtual_boundary: bool,
    ) -> PyResult<Self> {
        let config = RustCheckMatrixConfig {
            repetitions,
            timelike_weights,
            use_virtual_boundary,
            ..Default::default()
        };

        RustPyMatchingDecoder::from_check_matrix_with_config(&check_matrix.inner, config)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Create decoder from a Stim Detector Error Model.
    ///
    /// This mirrors `PyMatching`'s `Matching.from_detector_error_model()`.
    ///
    /// # Arguments
    ///
    /// * `dem` - Detector error model string in Stim format
    ///
    /// # Example
    ///
    /// ```python
    /// dem = circuit.detector_error_model().to_string()
    /// decoder = PyMatchingDecoder.from_dem(dem)
    /// ```
    #[staticmethod]
    fn from_dem(dem: &str) -> PyResult<Self> {
        RustPyMatchingDecoder::from_dem(dem)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Add an edge between two detector nodes.
    ///
    /// This mirrors `PyMatching`'s `Matching.add_edge()`.
    ///
    /// # Arguments
    ///
    /// * `node1` - First detector node index
    /// * `node2` - Second detector node index
    /// * `observables` - List of observable indices this edge affects when flipped
    /// * `weight` - Edge weight (default: computed from `error_probability`)
    /// * `error_probability` - Error probability for this edge
    #[pyo3(signature = (node1, node2, observables, weight=None, error_probability=None))]
    fn add_edge(
        &mut self,
        node1: usize,
        node2: usize,
        observables: Vec<usize>,
        weight: Option<f64>,
        error_probability: Option<f64>,
    ) -> PyResult<()> {
        self.inner
            .add_edge(node1, node2, &observables, weight, error_probability, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Add a boundary edge from a detector node.
    ///
    /// Boundary edges connect a detector to the boundary (virtual node).
    /// This mirrors `PyMatching`'s `Matching.add_boundary_edge()`.
    ///
    /// # Arguments
    ///
    /// * `node` - Detector node index
    /// * `observables` - Observable indices affected by this edge
    /// * `weight` - Edge weight
    /// * `error_probability` - Error probability
    #[pyo3(signature = (node, observables, weight=None, error_probability=None))]
    fn add_boundary_edge(
        &mut self,
        node: usize,
        observables: Vec<usize>,
        weight: Option<f64>,
        error_probability: Option<f64>,
    ) -> PyResult<()> {
        self.inner
            .add_boundary_edge(node, &observables, weight, error_probability, None)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode a syndrome to find the most likely error.
    ///
    /// This mirrors `PyMatching`'s `Matching.decode()`.
    ///
    /// # Arguments
    ///
    /// * `syndrome` - Detection events (0 or 1 for each detector)
    ///
    /// # Returns
    ///
    /// `MwpmResult` with correction vector and matching weight.
    ///
    /// # Example
    ///
    /// ```python
    /// syndrome = [1, 0, 1, 0]
    /// result = decoder.decode(syndrome)
    /// correction = result.correction  # Observable flips to apply
    /// ```
    fn decode(&mut self, syndrome: Vec<u8>) -> PyResult<PyMwpmResult> {
        self.inner
            .decode(&syndrome)
            .map(|result| PyMwpmResult {
                correction_data: result.observable,
                weight: result.weight,
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Number of detector nodes in the matching graph.
    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }

    /// Number of nodes (detectors + boundary) in the matching graph.
    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    /// Number of edges in the matching graph.
    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    /// Number of logical observables.
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    fn __repr__(&self) -> String {
        format!(
            "PyMatchingDecoder(detectors={}, edges={}, observables={})",
            self.inner.num_detectors(),
            self.inner.num_edges(),
            self.inner.num_observables()
        )
    }
}

// =============================================================================
// Fusion Blossom Decoder
// =============================================================================

use pecos::decoders::{
    FusionBlossomConfig as RustFusionBlossomConfig,
    FusionBlossomDecoder as RustFusionBlossomDecoder, SolverType as RustSolverType,
    StandardCode as RustStandardCode, SyndromeData as RustSyndromeData,
};

/// Fusion Blossom MWPM decoder.
///
/// Pure Rust implementation of minimum-weight perfect matching.
/// Supports parallel decoding and visualization for debugging.
///
/// # Construction
///
/// ```python
/// from pecos_rslib.decoders import FusionBlossomDecoder
///
/// # From check matrix
/// H = [[1, 1, 0], [0, 1, 1]]
/// decoder = FusionBlossomDecoder.from_check_matrix(H)
///
/// # From standard code (like fusion-blossom's CodeCapacityPlanarCode)
/// decoder = FusionBlossomDecoder.from_standard_code(
///     "code_capacity_rotated", distance=5, error_rate=0.01
/// )
///
/// # Manual construction
/// decoder = FusionBlossomDecoder(num_nodes=4)
/// decoder.add_edge(0, 1, observables=[0], weight=1.0)
/// ```
///
/// # Decoding
///
/// ```python
/// result = decoder.decode(syndrome)
/// decoder.clear()  # Reset for next shot (efficient reuse)
/// ```
#[pyclass(name = "FusionBlossomDecoder", module = "pecos_rslib.decoders")]
pub struct PyFusionBlossomDecoder {
    inner: RustFusionBlossomDecoder,
}

#[pymethods]
impl PyFusionBlossomDecoder {
    /// Create decoder for manual graph construction.
    ///
    /// # Arguments
    ///
    /// * `num_nodes` - Number of detector nodes
    /// * `num_observables` - Number of logical observables (default: 1)
    /// * `solver` - Solver type: "serial" or "parallel" (default: "serial")
    #[new]
    #[pyo3(signature = (num_nodes, num_observables=1, solver="serial"))]
    fn new(num_nodes: usize, num_observables: usize, solver: &str) -> PyResult<Self> {
        let solver_type = match solver {
            "serial" => RustSolverType::Serial,
            "parallel" | "legacy" => RustSolverType::Legacy,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "solver must be 'serial' or 'parallel'",
                ));
            }
        };

        let config = RustFusionBlossomConfig {
            num_nodes: Some(num_nodes),
            num_observables,
            solver_type,
            max_tree_size: None,
        };

        RustFusionBlossomDecoder::new(config)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Create decoder from a check matrix.
    ///
    /// # Arguments
    ///
    /// * `check_matrix` - Dense 2D matrix (list of lists) or `CheckMatrix`
    /// * `weights` - Optional weights for each column
    /// * `num_observables` - Number of observables (default: num columns)
    ///
    /// # Example
    ///
    /// ```python
    /// H = [[1, 1, 0], [0, 1, 1]]
    /// decoder = FusionBlossomDecoder.from_check_matrix(H)
    /// ```
    #[staticmethod]
    #[pyo3(signature = (check_matrix, weights=None, num_observables=None))]
    fn from_check_matrix(
        check_matrix: Vec<Vec<u8>>,
        weights: Option<Vec<f64>>,
        num_observables: Option<usize>,
    ) -> PyResult<Self> {
        let rows = check_matrix.len();
        let cols = if rows > 0 { check_matrix[0].len() } else { 0 };

        let mut arr = Array2::<u8>::zeros((rows, cols));
        for (i, row) in check_matrix.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                arr[[i, j]] = val;
            }
        }

        let config = RustFusionBlossomConfig {
            num_nodes: Some(rows),
            num_observables: num_observables.unwrap_or(cols),
            solver_type: RustSolverType::Serial,
            max_tree_size: None,
        };

        RustFusionBlossomDecoder::from_check_matrix(&arr, weights.as_deref(), config)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Create decoder for a standard QEC code.
    ///
    /// This mirrors fusion-blossom's `CodeCapacityPlanarCode`, etc.
    ///
    /// # Arguments
    ///
    /// * `code_type` - Code type string:
    ///   - "`code_capacity_planar`" / "`code_capacity_rotated`"
    ///   - "`phenomenological_planar`" / "`phenomenological_rotated`"
    ///   - "`circuit_level_planar`"
    /// * `distance` - Code distance
    /// * `error_rate` - Physical error rate
    /// * `max_half_weight` - Maximum half-weight for discretization (default: 500)
    ///
    /// # Example
    ///
    /// ```python
    /// # Like fusion-blossom's CodeCapacityPlanarCode(d=11, p=0.05)
    /// decoder = FusionBlossomDecoder.from_standard_code(
    ///     "code_capacity_planar", distance=11, error_rate=0.05
    /// )
    /// ```
    #[staticmethod]
    #[pyo3(signature = (code_type, distance, error_rate, max_half_weight=500))]
    fn from_standard_code(
        code_type: &str,
        distance: usize,
        error_rate: f64,
        max_half_weight: i32,
    ) -> PyResult<Self> {
        let code = match code_type {
            "code_capacity_planar" => RustStandardCode::CodeCapacityPlanar {
                d: distance,
                p: error_rate,
                max_half_weight,
            },
            "code_capacity_rotated" => RustStandardCode::CodeCapacityRotated {
                d: distance,
                p: error_rate,
                max_half_weight,
            },
            "phenomenological_planar" => RustStandardCode::PhenomenologicalPlanar {
                d: distance,
                p: error_rate,
                p_measurement: error_rate,
                max_half_weight,
            },
            "phenomenological_rotated" => RustStandardCode::PhenomenologicalRotated {
                d: distance,
                p: error_rate,
                p_measurement: error_rate,
                max_half_weight,
            },
            "circuit_level_planar" => RustStandardCode::CircuitLevelPlanar {
                d: distance,
                p: error_rate,
                max_half_weight,
            },
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown code_type: '{code_type}'. Valid: code_capacity_planar, \
                     code_capacity_rotated, phenomenological_planar, phenomenological_rotated, \
                     circuit_level_planar"
                )));
            }
        };

        let config = RustFusionBlossomConfig::default();
        RustFusionBlossomDecoder::from_standard_code(code, config)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Add an edge between two nodes.
    #[pyo3(signature = (node1, node2, observables, weight=None))]
    fn add_edge(
        &mut self,
        node1: usize,
        node2: usize,
        observables: Vec<usize>,
        weight: Option<f64>,
    ) -> PyResult<()> {
        self.inner
            .add_edge(node1, node2, &observables, weight)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Add a boundary edge from a node.
    #[pyo3(signature = (node, observables, weight=None))]
    fn add_boundary_edge(
        &mut self,
        node: usize,
        observables: Vec<usize>,
        weight: Option<f64>,
    ) -> PyResult<()> {
        self.inner
            .add_boundary_edge(node, &observables, weight)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode a syndrome.
    ///
    /// # Arguments
    ///
    /// * `syndrome` - Detection events (0 or 1 for each detector)
    ///
    /// # Returns
    ///
    /// `MwpmResult` with correction and weight.
    fn decode(&mut self, syndrome: Vec<u8>) -> PyResult<PyMwpmResult> {
        let arr = Array1::from_vec(syndrome);
        self.inner
            .decode(&arr.view())
            .map(|result| PyMwpmResult {
                correction_data: result.observable,
                weight: result.weight,
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode from defect vertex indices (sparse syndrome representation).
    ///
    /// More efficient when syndrome is sparse (few defects).
    ///
    /// # Arguments
    ///
    /// * `defects` - List of detector indices with detection events
    /// * `erasures` - Optional list of erasure edge indices
    #[pyo3(signature = (defects, erasures=None))]
    fn decode_from_defects(
        &mut self,
        defects: Vec<usize>,
        erasures: Option<Vec<usize>>,
    ) -> PyResult<PyMwpmResult> {
        let syndrome_data = if let Some(erasure_list) = erasures {
            RustSyndromeData::with_erasures(defects, erasure_list)
        } else {
            RustSyndromeData::from_defects(defects)
        };

        self.inner
            .decode_advanced(syndrome_data)
            .map(|result| PyMwpmResult {
                correction_data: result.observable,
                weight: result.weight,
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Clear decoder state for efficient reuse.
    ///
    /// Call this between decoding shots instead of creating a new decoder.
    fn clear(&mut self) {
        self.inner.clear();
    }

    #[getter]
    fn num_nodes(&self) -> usize {
        self.inner.num_nodes()
    }

    #[getter]
    fn num_edges(&self) -> usize {
        self.inner.num_edges()
    }

    fn __repr__(&self) -> String {
        format!(
            "FusionBlossomDecoder(nodes={}, edges={})",
            self.inner.num_nodes(),
            self.inner.num_edges()
        )
    }
}

// =============================================================================
// LDPC Decoders
// =============================================================================

use pecos::decoders::{
    BpLsdDecoder as RustBpLsdDecoder, BpMethod as RustBpMethod, BpOsdDecoder as RustBpOsdDecoder,
    BpSchedule as RustBpSchedule, InputVectorType as RustInputVectorType,
    OsdMethod as RustOsdMethod, SparseMatrix as RustSparseMatrix, UfMethod as RustUfMethod,
    UnionFindDecoder as RustUnionFindDecoder,
};

/// Sparse parity check matrix for LDPC decoders.
///
/// # Construction
///
/// ```python
/// from pecos_rslib.decoders import SparseMatrix
///
/// # From dense matrix
/// H = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
/// matrix = SparseMatrix(H)
///
/// # From COO format
/// matrix = SparseMatrix.from_coo(
///     rows=3, cols=4,
///     row_indices=[0, 0, 1, 1, 2, 2],
///     col_indices=[0, 1, 1, 2, 2, 3]
/// )
/// ```
#[pyclass(
    name = "SparseMatrix",
    module = "pecos_rslib.decoders",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PySparseMatrix {
    inner: RustSparseMatrix,
}

#[pymethods]
impl PySparseMatrix {
    /// Create from a dense 2D matrix.
    ///
    /// # Arguments
    ///
    /// * `matrix` - Dense matrix as list of lists
    #[new]
    fn new(matrix: Vec<Vec<u8>>) -> Self {
        let rows = matrix.len();
        let cols = if rows > 0 { matrix[0].len() } else { 0 };

        let mut arr = Array2::<u8>::zeros((rows, cols));
        for (i, row) in matrix.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                arr[[i, j]] = val;
            }
        }

        Self {
            inner: RustSparseMatrix::from_dense(&arr.view()),
        }
    }

    /// Create from COO (Coordinate) format.
    #[staticmethod]
    fn from_coo(
        rows: usize,
        cols: usize,
        row_indices: Vec<u32>,
        col_indices: Vec<u32>,
    ) -> PyResult<Self> {
        RustSparseMatrix::from_coo(rows, cols, row_indices, col_indices)
            .map(|inner| Self { inner })
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.rows
    }

    #[getter]
    fn cols(&self) -> usize {
        self.inner.cols
    }

    fn nnz(&self) -> usize {
        self.inner.nnz()
    }

    fn __repr__(&self) -> String {
        format!(
            "SparseMatrix(rows={}, cols={}, nnz={})",
            self.inner.rows,
            self.inner.cols,
            self.inner.nnz()
        )
    }
}

/// BP+OSD decoder for LDPC codes.
///
/// Belief Propagation with Ordered Statistics Decoding post-processing.
/// This mirrors the ldpc library's `BpOsdDecoder`.
///
/// # Construction
///
/// ```python
/// from pecos_rslib.decoders import BpOsdDecoder, SparseMatrix
///
/// H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
///
/// # Like ldpc's BpOsdDecoder(H, error_rate=0.1, bp_method='product_sum', ...)
/// decoder = BpOsdDecoder(
///     H,
///     error_rate=0.1,
///     bp_method="product_sum",
///     max_iter=100,
///     osd_method="osd0",
///     osd_order=0
/// )
/// ```
///
/// # Decoding
///
/// ```python
/// result = decoder.decode(syndrome)
/// if result.converged:
///     error_estimate = result.decoding
/// ```
// Note: unsendable because contains FFI pointers
#[pyclass(name = "BpOsdDecoder", module = "pecos_rslib.decoders", unsendable)]
pub struct PyBpOsdDecoder {
    inner: RustBpOsdDecoder,
}

#[pymethods]
impl PyBpOsdDecoder {
    /// Create a BP+OSD decoder.
    ///
    /// # Arguments
    ///
    /// * `pcm` - Parity check matrix
    /// * `error_rate` - Channel error probability (or use `channel_probs` for per-qubit rates)
    /// * `max_iter` - Maximum BP iterations (default: 100, 0 = use n)
    /// * `bp_method` - BP algorithm: "`product_sum`" or "`minimum_sum`" (default: "`product_sum`")
    /// * `schedule` - Update schedule: "parallel" or "serial" (default: "parallel")
    /// * `osd_method` - OSD variant: "off", "osd0", "`osd_e`", "`osd_cs`" (default: "osd0")
    /// * `osd_order` - OSD order parameter (default: 0)
    #[new]
    #[pyo3(signature = (pcm, error_rate, max_iter=100, bp_method="product_sum", schedule="parallel", osd_method="osd0", osd_order=0))]
    fn new(
        pcm: &PySparseMatrix,
        error_rate: f64,
        max_iter: usize,
        bp_method: &str,
        schedule: &str,
        osd_method: &str,
        osd_order: usize,
    ) -> PyResult<Self> {
        let bp = match bp_method {
            "product_sum" | "ps" => RustBpMethod::ProductSum,
            "minimum_sum" | "ms" => RustBpMethod::MinimumSum,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "bp_method must be 'product_sum' or 'minimum_sum'",
                ));
            }
        };

        let bp_schedule = match schedule {
            "parallel" => RustBpSchedule::Parallel,
            "serial" => RustBpSchedule::Serial,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "schedule must be 'parallel' or 'serial'",
                ));
            }
        };

        let osd = match osd_method {
            "off" => RustOsdMethod::Off,
            "osd0" => RustOsdMethod::Osd0,
            "osd_e" => RustOsdMethod::OsdE,
            "osd_cs" => RustOsdMethod::OsdCs,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "osd_method must be 'off', 'osd0', 'osd_e', or 'osd_cs'",
                ));
            }
        };

        RustBpOsdDecoder::new(
            &pcm.inner,
            Some(error_rate),
            None,
            max_iter,
            bp,
            bp_schedule,
            1.0,
            osd,
            osd_order,
            RustInputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .map(|inner| Self { inner })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode a syndrome.
    ///
    /// # Arguments
    ///
    /// * `syndrome` - Syndrome vector (length = number of checks)
    ///
    /// # Returns
    ///
    /// `BpResult` with decoding, convergence status, and iteration count.
    fn decode(&mut self, syndrome: Vec<u8>) -> PyResult<PyBpResult> {
        let arr = Array1::from_vec(syndrome);
        self.inner
            .decode(&arr.view())
            .map(|result| PyBpResult {
                decoding_data: result.decoding.to_vec(),
                converged: result.converged,
                iterations: result.iterations,
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "BpOsdDecoder(...)".to_string()
    }
}

/// BP+LSD decoder for LDPC codes.
///
/// Belief Propagation with Localized Statistics Decoding.
/// Often faster than OSD for similar accuracy.
///
/// # Construction
///
/// ```python
/// from pecos_rslib.decoders import BpLsdDecoder, SparseMatrix
///
/// H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
///
/// # Like ldpc's BpLsdDecoder
/// decoder = BpLsdDecoder(
///     H,
///     error_rate=0.1,
///     bp_method="product_sum",
///     max_iter=100,
///     lsd_order=0
/// )
/// ```
// Note: unsendable because contains FFI pointers
#[pyclass(name = "BpLsdDecoder", module = "pecos_rslib.decoders", unsendable)]
pub struct PyBpLsdDecoder {
    inner: RustBpLsdDecoder,
}

#[pymethods]
impl PyBpLsdDecoder {
    /// Create a BP+LSD decoder.
    ///
    /// # Arguments
    ///
    /// * `pcm` - Parity check matrix
    /// * `error_rate` - Channel error probability
    /// * `max_iter` - Maximum BP iterations (default: 100)
    /// * `bp_method` - "`product_sum`" or "`minimum_sum`" (default: "`product_sum`")
    /// * `schedule` - "parallel" or "serial" (default: "parallel")
    /// * `lsd_order` - LSD order parameter (default: 0, recommended starting point)
    #[new]
    #[pyo3(signature = (pcm, error_rate, max_iter=100, bp_method="product_sum", schedule="parallel", lsd_order=0))]
    fn new(
        pcm: &PySparseMatrix,
        error_rate: f64,
        max_iter: usize,
        bp_method: &str,
        schedule: &str,
        lsd_order: usize,
    ) -> PyResult<Self> {
        let bp = match bp_method {
            "product_sum" | "ps" => RustBpMethod::ProductSum,
            "minimum_sum" | "ms" => RustBpMethod::MinimumSum,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "bp_method must be 'product_sum' or 'minimum_sum'",
                ));
            }
        };

        let bp_schedule = match schedule {
            "parallel" => RustBpSchedule::Parallel,
            "serial" => RustBpSchedule::Serial,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "schedule must be 'parallel' or 'serial'",
                ));
            }
        };

        RustBpLsdDecoder::new(
            &pcm.inner,
            Some(error_rate),
            None,
            max_iter,
            bp,
            bp_schedule,
            1.0,
            RustOsdMethod::Osd0,
            lsd_order,
            0,
            RustInputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .map(|inner| Self { inner })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode a syndrome.
    fn decode(&mut self, syndrome: Vec<u8>) -> PyResult<PyBpResult> {
        let arr = Array1::from_vec(syndrome);
        self.inner
            .decode(&arr.view())
            .map(|result| PyBpResult {
                decoding_data: result.decoding.to_vec(),
                converged: result.converged,
                iterations: result.iterations,
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "BpLsdDecoder(...)".to_string()
    }
}

/// Union-Find decoder for LDPC codes.
///
/// Cluster-based decoder using the Union-Find data structure.
/// Fast O(n * alpha(n)) complexity per syndrome.
///
/// # Construction
///
/// ```python
/// from pecos_rslib.decoders import UnionFindDecoder, SparseMatrix
///
/// H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
/// decoder = UnionFindDecoder(H, method="inversion")
/// ```
// Note: unsendable because contains FFI pointers
#[pyclass(name = "UnionFindDecoder", module = "pecos_rslib.decoders", unsendable)]
pub struct PyUnionFindDecoder {
    inner: RustUnionFindDecoder,
}

#[pymethods]
impl PyUnionFindDecoder {
    /// Create a Union-Find decoder.
    ///
    /// # Arguments
    ///
    /// * `pcm` - Parity check matrix
    /// * `method` - Decoding method: "inversion" (general) or "peeling" (LDPC only)
    #[new]
    #[pyo3(signature = (pcm, method="inversion"))]
    fn new(pcm: &PySparseMatrix, method: &str) -> PyResult<Self> {
        let uf_method = match method {
            "inversion" => RustUfMethod::Inversion,
            "peeling" => RustUfMethod::Peeling,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "method must be 'inversion' or 'peeling'",
                ));
            }
        };

        RustUnionFindDecoder::new(&pcm.inner, uf_method)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode a syndrome.
    ///
    /// # Arguments
    ///
    /// * `syndrome` - Syndrome vector
    /// * `llrs` - Optional log-likelihood ratios for soft information
    /// * `bits_per_step` - Bits to grow per step (0 = all at once)
    #[pyo3(signature = (syndrome, llrs=None, bits_per_step=0))]
    fn decode(
        &mut self,
        syndrome: Vec<u8>,
        llrs: Option<Vec<f64>>,
        bits_per_step: usize,
    ) -> PyResult<PyBpResult> {
        let arr = Array1::from_vec(syndrome);
        let llrs_slice = llrs.as_deref().unwrap_or(&[]);

        self.inner
            .decode(&arr.view(), llrs_slice, bits_per_step)
            .map(|result| PyBpResult {
                decoding_data: result.decoding.to_vec(),
                converged: result.converged,
                iterations: result.iterations,
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        "UnionFindDecoder(...)".to_string()
    }
}

// =============================================================================
// Tesseract Decoder
// =============================================================================

use pecos::decoders::{
    TesseractConfig as RustTesseractConfig, TesseractDecoder as RustTesseractDecoder,
};

/// Result from Tesseract decoder.
///
/// # Attributes
///
/// * `observables_mask` - Bitwise XOR of observables affected by predicted errors
/// * `cost` - Total cost of the solution
/// * `low_confidence` - Whether this is a low-confidence prediction
#[pyclass(
    name = "TesseractResult",
    module = "pecos_rslib.decoders",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyTesseractResult {
    #[pyo3(get)]
    observables_mask: u64,
    #[pyo3(get)]
    cost: f64,
    #[pyo3(get)]
    low_confidence: bool,
}

#[pymethods]
impl PyTesseractResult {
    /// Get the observable predictions as a list of bits.
    fn observable_bits(&self, num_observables: usize) -> Vec<i32> {
        (0..num_observables)
            .map(|i| ((self.observables_mask >> i) & 1) as i32)
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "TesseractResult(observables_mask={}, cost={:.4}, low_confidence={})",
            self.observables_mask, self.cost, self.low_confidence
        )
    }
}

/// Tesseract search-based decoder for quantum error correction.
///
/// Uses A* search with pruning heuristics to find the most likely error
/// configuration consistent with observed syndromes. Particularly effective
/// for LDPC quantum codes.
///
/// # Construction
///
/// ```python
/// from pecos_rslib.decoders import TesseractDecoder
///
/// # From Stim Detector Error Model string
/// dem = '''
/// error(0.1) D0 D1
/// error(0.05) D1 D2 L0
/// '''
/// decoder = TesseractDecoder.from_dem(dem)
///
/// # With configuration
/// decoder = TesseractDecoder.from_dem(dem, preset="fast")
/// ```
///
/// # Decoding
///
/// ```python
/// # Detection events as list of detector indices that fired
/// detection_indices = [0, 2]
/// result = decoder.decode(detection_indices)
/// print(f"Observable mask: {result.observables_mask}, Cost: {result.cost}")
/// ```
#[pyclass(name = "TesseractDecoder", module = "pecos_rslib.decoders", unsendable)]
pub struct PyTesseractDecoder {
    inner: RustTesseractDecoder,
}

#[pymethods]
impl PyTesseractDecoder {
    /// Create Tesseract decoder from a Detector Error Model string.
    ///
    /// # Arguments
    ///
    /// * `dem` - Detector error model in Stim format
    /// * `preset` - Configuration preset: "default", "fast", or "accurate"
    /// * `det_beam` - Detector beam size (default: `u16::MAX` for infinite)
    /// * `beam_climbing` - Enable beam climbing heuristic
    /// * `verbose` - Enable verbose output
    ///
    /// # Example
    ///
    /// ```python
    /// dem = "error(0.1) D0 D1\\nerror(0.05) D1 D2 L0"
    /// decoder = TesseractDecoder.from_dem(dem)
    /// # Or with fast preset
    /// decoder = TesseractDecoder.from_dem(dem, preset="fast")
    /// ```
    #[staticmethod]
    #[pyo3(signature = (dem, preset="default", det_beam=None, beam_climbing=None, verbose=false))]
    fn from_dem(
        dem: &str,
        preset: &str,
        det_beam: Option<u16>,
        beam_climbing: Option<bool>,
        verbose: bool,
    ) -> PyResult<Self> {
        let mut config = match preset {
            "fast" => RustTesseractConfig::fast(),
            "accurate" => RustTesseractConfig::accurate(),
            _ => RustTesseractConfig::default(),
        };

        // Override with explicit parameters
        if let Some(beam) = det_beam {
            config.det_beam = beam;
        }
        if let Some(climbing) = beam_climbing {
            config.beam_climbing = climbing;
        }
        config.verbose = verbose;

        RustTesseractDecoder::new(dem, config)
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode detection events to find the most likely error configuration.
    ///
    /// # Arguments
    ///
    /// * `detections` - List of detector indices that fired (sparse representation)
    ///
    /// # Returns
    ///
    /// `TesseractResult` with observables mask, cost, and confidence info.
    ///
    /// # Example
    ///
    /// ```python
    /// # Detectors 0 and 2 fired
    /// result = decoder.decode([0, 2])
    /// print(f"Observable prediction: {result.observable_bits(1)}")
    /// ```
    fn decode(&mut self, detections: Vec<u64>) -> PyResult<PyTesseractResult> {
        let detections_arr = ndarray::Array1::from_vec(detections);

        self.inner
            .decode_detections(&detections_arr.view())
            .map(|result| PyTesseractResult {
                observables_mask: result.observables_mask,
                cost: result.cost,
                low_confidence: result.low_confidence,
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode a dense syndrome vector.
    ///
    /// # Arguments
    ///
    /// * `syndrome` - Dense syndrome vector (0 or 1 for each detector)
    ///
    /// # Returns
    ///
    /// `TesseractResult` with observables mask and cost.
    fn decode_syndrome(&mut self, syndrome: Vec<u8>) -> PyResult<PyTesseractResult> {
        // Convert dense syndrome to sparse detection indices
        let detections: Vec<u64> = syndrome
            .iter()
            .enumerate()
            .filter_map(|(i, &val)| if val != 0 { Some(i as u64) } else { None })
            .collect();

        self.decode(detections)
    }

    /// Number of detectors in the error model.
    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }

    /// Number of errors in the error model.
    #[getter]
    fn num_errors(&self) -> usize {
        self.inner.num_errors()
    }

    /// Number of observables in the error model.
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    fn __repr__(&self) -> String {
        format!(
            "TesseractDecoder(detectors={}, errors={}, observables={})",
            self.inner.num_detectors(),
            self.inner.num_errors(),
            self.inner.num_observables()
        )
    }
}

// =============================================================================
// Module Registration
// =============================================================================

/// Register the decoders module with Python.
pub fn register_decoders_module(parent_module: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = parent_module.py();
    let decoders_module = PyModule::new(py, "decoders")?;

    // Common result types
    decoders_module.add_class::<PyMwpmResult>()?;
    decoders_module.add_class::<PyBpResult>()?;

    // Matrix types
    decoders_module.add_class::<PyCheckMatrix>()?;
    decoders_module.add_class::<PySparseMatrix>()?;

    // MWPM decoders
    decoders_module.add_class::<PyPyMatchingDecoder>()?;
    decoders_module.add_class::<PyFusionBlossomDecoder>()?;

    // LDPC decoders
    decoders_module.add_class::<PyBpOsdDecoder>()?;
    decoders_module.add_class::<PyBpLsdDecoder>()?;
    decoders_module.add_class::<PyUnionFindDecoder>()?;

    // Search-based decoders
    decoders_module.add_class::<PyTesseractResult>()?;
    decoders_module.add_class::<PyTesseractDecoder>()?;

    // Add submodule to parent
    parent_module.add_submodule(&decoders_module)?;

    // Register in sys.modules for proper import
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.decoders", &decoders_module)?;

    Ok(())
}
