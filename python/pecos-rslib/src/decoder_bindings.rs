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
//!
//! ## Relay BP Decoders (qLDPC Belief Propagation)
//! - `RelayBpDecoder` - Relay BP ensemble decoder
//! - `MinSumBpDecoder` - Plain min-sum BP decoder

use ndarray::{Array1, Array2};
use pyo3::prelude::*;

use crate::observable_flips_bindings::PyObservableFlips;

fn explicit_decode_attribute_error(class_name: &str, name: &str) -> PyErr {
    if name == "decode" {
        pyo3::exceptions::PyAttributeError::new_err(format!(
            "{class_name} has no attribute 'decode'; use decode_syndrome(...) for a dense vector \
             or decode_from_defects(...) for sparse detector indices"
        ))
    } else {
        pyo3::exceptions::PyAttributeError::new_err(format!(
            "'{class_name}' object has no attribute '{name}'"
        ))
    }
}

// =============================================================================
// Common Result Types
// =============================================================================

/// Result from MWPM (Minimum Weight Perfect Matching) decoders.
///
/// This unified result type is returned by both `PyMatching` and Fusion Blossom decoders.
///
/// # Attributes
///
/// * `observable_flips` - The decoded observable flips
/// * `weight` - Total weight of the matching (lower is better)
///
/// # Example
///
/// ```python
/// result = decoder.decode_syndrome(syndrome)
/// if result.weight < threshold:
///     apply_correction(result.observable_flips)
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
    /// The decoded observable flips with their intrinsic observable count.
    #[getter]
    fn observable_flips(&self) -> PyObservableFlips {
        PyObservableFlips::from_u8_bits(&self.correction_data)
    }

    fn __repr__(&self) -> String {
        format!(
            "MwpmResult(observable_flips={:?}, weight={:.4})",
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
/// * `decoding` - The decoded error vector, indexed by error mechanism, not observable
/// * `converged` - Whether BP converged before max iterations
/// * `iterations` - Number of BP iterations performed
///
/// Per-observable results come from the decoders' `from_dem` constructors,
/// which return `DemAwareResult` from `decode_syndrome`.
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
    ///
    /// This vector is indexed by error mechanism, not by observable.
    /// Per-observable results come from the `from_dem` constructors, whose
    /// `decode_syndrome` method returns `DemAwareResult`.
    #[getter]
    fn decoding(&self) -> Vec<i32> {
        self.decoding_data.iter().map(|&x| i32::from(x)).collect()
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

use pecos_decoders::{
    CheckMatrix as RustCheckMatrix, CheckMatrixConfig as RustCheckMatrixConfig,
    PyMatchingConfig as RustPyMatchingConfig, PyMatchingDecoder as RustPyMatchingDecoder,
};

#[derive(Debug, Clone, Copy, Default, PartialEq)]
struct PyMatchingDemConfig {
    error_probability: Option<f64>,
}

fn pymatching_config(error_probability: Option<f64>) -> PyMatchingDemConfig {
    let mut config = PyMatchingDemConfig::default();
    if let Some(error_probability) = error_probability {
        config.error_probability = Some(error_probability);
    }
    config
}

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
/// # From detector error model
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
/// result = decoder.decode_syndrome(syndrome)
/// print(f"Observable flips: {list(result.observable_flips)}, Weight: {result.weight}")
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

    /// Create decoder from a Detector Error Model.
    ///
    /// This mirrors `PyMatching`'s `Matching.from_detector_error_model()`.
    ///
    /// # Arguments
    ///
    /// * `dem` - Detector error model string in Stim format
    /// * `error_probability` - Replace every edge probability and its derived matching weight;
    ///   closer calibration can improve decoding accuracy without changing asymptotic runtime
    ///
    /// # Example
    ///
    /// ```python
    /// dem = circuit.detector_error_model().to_string()
    /// decoder = PyMatchingDecoder.from_dem(dem)
    /// ```
    #[staticmethod]
    #[pyo3(signature = (dem, error_probability=None))]
    fn from_dem(dem: &str, error_probability: Option<f64>) -> PyResult<Self> {
        let config = pymatching_config(error_probability);
        let inner = if let Some(error_probability) = config.error_probability {
            RustPyMatchingDecoder::from_dem_with_error_probability(dem, error_probability)
        } else {
            RustPyMatchingDecoder::from_dem(dem)
        };
        inner
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Create decoder from a Detector Error Model with correlation support.
    ///
    /// When enabled, PyMatching preserves DEM decomposition correlations while
    /// constructing and decoding the matching graph.
    #[staticmethod]
    #[pyo3(signature = (dem, enable_correlations=true))]
    fn from_dem_with_correlations(dem: &str, enable_correlations: bool) -> PyResult<Self> {
        RustPyMatchingDecoder::from_dem_with_correlations(dem, enable_correlations)
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
    /// This mirrors `PyMatching`'s `Matching.decode()` with an explicit dense encoding name.
    ///
    /// # Arguments
    ///
    /// * `syndrome` - Detection events (0 or 1 for each detector)
    ///
    /// # Returns
    ///
    /// `MwpmResult` with observable flips and matching weight.
    ///
    /// # Example
    ///
    /// ```python
    /// syndrome = [1, 0, 1, 0]
    /// result = decoder.decode_syndrome(syndrome)
    /// observable_flips = result.observable_flips
    /// ```
    fn decode_syndrome(&mut self, syndrome: Vec<u8>) -> PyResult<PyMwpmResult> {
        self.inner
            .decode(&syndrome)
            .map(|result| PyMwpmResult {
                correction_data: result.observable,
                weight: result.weight,
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode a batch of syndromes at once.
    ///
    /// Much faster than calling `decode_syndrome()` in a Python loop -- the entire batch
    /// is processed in Rust with no per-shot Python overhead.
    ///
    /// # Arguments
    ///
    /// * `detection_events` - Flattened detection events array (`num_shots` * `num_detectors` bytes)
    /// * `num_shots` - Number of shots in the batch
    ///
    /// # Returns
    ///
    /// List of observable predictions (one per shot), where each prediction
    /// is a list of 0/1 values (one per observable). Check index 0 for
    /// single-observable codes.
    ///
    /// # Example
    ///
    /// ```python
    /// # detection_events is shape (num_shots, num_detectors), flattened
    /// flat = detection_events.flatten().tolist()
    /// predictions = decoder.decode_batch(flat, num_shots=len(detection_events))
    /// num_errors = sum(p[0] != t for p, t in zip(predictions, true_flips))
    /// ```
    fn decode_batch(
        &mut self,
        detection_events: Vec<u8>,
        num_shots: usize,
    ) -> PyResult<Vec<Vec<u8>>> {
        use pecos_decoders::BatchConfig as RustBatchConfig;

        let num_detectors = self.inner.num_detectors();
        let config = RustBatchConfig {
            bit_packed_input: false,
            bit_packed_output: false,
            return_weights: false,
        };

        self.inner
            .decode_batch_with_config(&detection_events, num_shots, num_detectors, config)
            .map(|result| result.predictions)
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

    fn __getattr__(&self, name: &str) -> PyResult<()> {
        Err(explicit_decode_attribute_error("PyMatchingDecoder", name))
    }
}

// =============================================================================
// Fusion Blossom Decoder
// =============================================================================

use pecos_decoders::{
    FusionBlossomConfig as RustFusionBlossomConfig,
    FusionBlossomDecoder as RustFusionBlossomDecoder, SolverType as RustSolverType,
    StandardCode as RustStandardCode, SyndromeData as RustSyndromeData,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FusionBlossomDemConfig {
    correlated: bool,
    solver_type: RustSolverType,
}

fn fusion_blossom_config(
    correlated: bool,
    solver_type: Option<&str>,
) -> Result<FusionBlossomDemConfig, String> {
    let mut config = FusionBlossomDemConfig {
        correlated,
        solver_type: RustSolverType::Serial,
    };
    match solver_type.unwrap_or("serial") {
        "legacy" => config.solver_type = RustSolverType::Legacy,
        "serial" => {}
        "parallel" => {
            return Err(
                "solver_type 'parallel' requires a partition configuration, which from_dem does not accept"
                    .to_string(),
            );
        }
        _ => return Err("solver_type must be 'legacy' or 'serial'".to_string()),
    }
    Ok(config)
}

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
/// result = decoder.decode_syndrome(syndrome)
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
    /// Create a decoder from a Detector Error Model.
    ///
    /// # Arguments
    ///
    /// * `dem` - Detector error model string in Stim format
    /// * `correlated` - Exploit X-Z correlations from decomposed mechanisms
    /// * `solver_type` - "serial" is usually faster while "legacy" can handle
    ///   more graph shapes; neither changes the DEM-derived memory footprint
    ///
    /// # Example
    ///
    /// ```python
    /// decoder = FusionBlossomDecoder.from_dem(dem_string)
    /// ```
    #[staticmethod]
    #[pyo3(signature = (dem, correlated=false, solver_type=None))]
    fn from_dem(dem: &str, correlated: bool, solver_type: Option<&str>) -> PyResult<Self> {
        let config = fusion_blossom_config(correlated, solver_type)
            .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        let inner = if config.correlated {
            RustFusionBlossomDecoder::from_dem_correlated_with_solver_type(dem, config.solver_type)
        } else {
            RustFusionBlossomDecoder::from_dem_with_solver_type(dem, config.solver_type)
        };
        inner
            .map(|inner| Self { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

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
    /// `MwpmResult` with observable flips and weight.
    fn decode_syndrome(&mut self, syndrome: Vec<u8>) -> PyResult<PyMwpmResult> {
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

    fn __getattr__(&self, name: &str) -> PyResult<()> {
        Err(explicit_decode_attribute_error(
            "FusionBlossomDecoder",
            name,
        ))
    }
}

// =============================================================================
// LDPC Decoders
// =============================================================================

use pecos_decoders::{
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

/// Parse a BP method string into the Rust enum.
fn parse_bp_method(s: &str) -> PyResult<RustBpMethod> {
    match s {
        "product_sum" | "ps" => Ok(RustBpMethod::ProductSum),
        "minimum_sum" | "ms" => Ok(RustBpMethod::MinimumSum),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "bp_method must be 'product_sum' or 'minimum_sum'",
        )),
    }
}

/// Parse a BP schedule string into the Rust enum.
fn parse_bp_schedule(s: &str) -> PyResult<RustBpSchedule> {
    bp_schedule(s).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)
}

fn bp_schedule(s: &str) -> Result<RustBpSchedule, String> {
    match s {
        "parallel" => Ok(RustBpSchedule::Parallel),
        "serial" => Ok(RustBpSchedule::Serial),
        "serial_relative" => Ok(RustBpSchedule::SerialRelative),
        _ => Err("bp_schedule must be 'parallel', 'serial', or 'serial_relative'".to_string()),
    }
}

fn uf_method(s: &str) -> Result<RustUfMethod, String> {
    match s {
        "inversion" => Ok(RustUfMethod::Inversion),
        "peeling" => Ok(RustUfMethod::Peeling),
        _ => Err("method must be 'inversion' or 'peeling'".to_string()),
    }
}

fn optional_usize(value: Option<i128>, parameter: &str) -> PyResult<Option<usize>> {
    value
        .map(|value| {
            usize::try_from(value).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "{parameter} must be a non-negative integer no greater than {}",
                    usize::MAX
                ))
            })
        })
        .transpose()
}

fn optional_u16(value: Option<i64>, parameter: &str) -> PyResult<Option<u16>> {
    value
        .map(|value| {
            u16::try_from(value).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "{parameter} must be an integer between 0 and {}",
                    u16::MAX
                ))
            })
        })
        .transpose()
}

fn optional_i32(value: Option<i64>, parameter: &str) -> PyResult<Option<i32>> {
    value
        .map(|value| {
            i32::try_from(value).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "{parameter} must be an integer between {} and {}",
                    i32::MIN,
                    i32::MAX
                ))
            })
        })
        .transpose()
}

fn optional_u64(value: Option<i128>, parameter: &str) -> PyResult<Option<u64>> {
    value
        .map(|value| {
            u64::try_from(value).map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "{parameter} must be an integer between 0 and {}",
                    u64::MAX
                ))
            })
        })
        .transpose()
}

const DEFAULT_DEM_MAX_ITER: usize = 100;

#[derive(Debug, Clone, Copy, PartialEq)]
struct BpOsdDemConfig {
    error_rate: Option<f64>,
    max_iter: usize,
    bp_method: RustBpMethod,
    bp_schedule: RustBpSchedule,
    ms_scaling_factor: f64,
    osd_method: RustOsdMethod,
    osd_order: usize,
    random_schedule_seed: Option<i32>,
}

impl Default for BpOsdDemConfig {
    fn default() -> Self {
        Self {
            error_rate: None,
            max_iter: DEFAULT_DEM_MAX_ITER,
            bp_method: RustBpMethod::ProductSum,
            bp_schedule: RustBpSchedule::Parallel,
            ms_scaling_factor: 1.0,
            osd_method: RustOsdMethod::Osd0,
            osd_order: 0,
            random_schedule_seed: None,
        }
    }
}

fn bp_osd_config(
    error_rate: Option<f64>,
    max_iter: Option<usize>,
    bp_schedule: Option<&str>,
    ms_scaling_factor: Option<f64>,
    osd_order: Option<usize>,
    random_schedule_seed: Option<i32>,
) -> Result<BpOsdDemConfig, String> {
    let mut config = BpOsdDemConfig::default();
    if let Some(error_rate) = error_rate {
        config.error_rate = Some(error_rate);
    }
    if let Some(max_iter) = max_iter {
        config.max_iter = max_iter;
    }
    if let Some(bp_schedule) = bp_schedule {
        config.bp_schedule = self::bp_schedule(bp_schedule)?;
    }
    if let Some(ms_scaling_factor) = ms_scaling_factor {
        config.bp_method = RustBpMethod::MinimumSum;
        config.ms_scaling_factor = ms_scaling_factor;
    }
    if let Some(osd_order) = osd_order {
        if osd_order > 0 {
            config.osd_method = RustOsdMethod::OsdCs;
        }
        config.osd_order = osd_order;
    }
    if let Some(random_schedule_seed) = random_schedule_seed {
        config.random_schedule_seed = Some(random_schedule_seed);
    }
    Ok(config)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BpLsdDemConfig {
    error_rate: Option<f64>,
    max_iter: usize,
    bp_method: RustBpMethod,
    bp_schedule: RustBpSchedule,
    ms_scaling_factor: f64,
    random_schedule_seed: Option<i32>,
}

impl Default for BpLsdDemConfig {
    fn default() -> Self {
        Self {
            error_rate: None,
            max_iter: DEFAULT_DEM_MAX_ITER,
            bp_method: RustBpMethod::ProductSum,
            bp_schedule: RustBpSchedule::Parallel,
            ms_scaling_factor: 1.0,
            random_schedule_seed: None,
        }
    }
}

fn bp_lsd_config(
    error_rate: Option<f64>,
    max_iter: Option<usize>,
    bp_schedule: Option<&str>,
    ms_scaling_factor: Option<f64>,
    random_schedule_seed: Option<i32>,
) -> Result<BpLsdDemConfig, String> {
    let mut config = BpLsdDemConfig::default();
    if let Some(error_rate) = error_rate {
        config.error_rate = Some(error_rate);
    }
    if let Some(max_iter) = max_iter {
        config.max_iter = max_iter;
    }
    if let Some(bp_schedule) = bp_schedule {
        config.bp_schedule = self::bp_schedule(bp_schedule)?;
    }
    if let Some(ms_scaling_factor) = ms_scaling_factor {
        config.bp_method = RustBpMethod::MinimumSum;
        config.ms_scaling_factor = ms_scaling_factor;
    }
    if let Some(random_schedule_seed) = random_schedule_seed {
        config.random_schedule_seed = Some(random_schedule_seed);
    }
    Ok(config)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct UnionFindDemConfig {
    method: RustUfMethod,
}

impl Default for UnionFindDemConfig {
    fn default() -> Self {
        Self {
            method: RustUfMethod::Inversion,
        }
    }
}

fn union_find_config(method: Option<&str>) -> Result<UnionFindDemConfig, String> {
    let mut config = UnionFindDemConfig::default();
    if let Some(method) = method {
        config.method = uf_method(method)?;
    }
    Ok(config)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct RelayBpDemConfig {
    error_rate: Option<f64>,
    max_iter: usize,
    alpha: Option<f64>,
    seed: u64,
}

impl Default for RelayBpDemConfig {
    fn default() -> Self {
        Self {
            error_rate: None,
            max_iter: DEFAULT_DEM_MAX_ITER,
            alpha: None,
            seed: 0,
        }
    }
}

fn relay_bp_config(
    error_rate: Option<f64>,
    max_iter: Option<usize>,
    alpha: Option<f64>,
    seed: Option<u64>,
) -> RelayBpDemConfig {
    let mut config = RelayBpDemConfig::default();
    if let Some(error_rate) = error_rate {
        config.error_rate = Some(error_rate);
    }
    if let Some(max_iter) = max_iter {
        config.max_iter = max_iter;
    }
    if let Some(alpha) = alpha {
        config.alpha = Some(alpha);
    }
    if let Some(seed) = seed {
        config.seed = seed;
    }
    config
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct MinSumBpDemConfig {
    error_rate: Option<f64>,
    max_iter: usize,
    alpha: Option<f64>,
}

impl Default for MinSumBpDemConfig {
    fn default() -> Self {
        Self {
            error_rate: None,
            max_iter: DEFAULT_DEM_MAX_ITER,
            alpha: None,
        }
    }
}

fn min_sum_bp_config(
    error_rate: Option<f64>,
    max_iter: Option<usize>,
    alpha: Option<f64>,
) -> MinSumBpDemConfig {
    let mut config = MinSumBpDemConfig::default();
    if let Some(error_rate) = error_rate {
        config.error_rate = Some(error_rate);
    }
    if let Some(max_iter) = max_iter {
        config.max_iter = max_iter;
    }
    if let Some(alpha) = alpha {
        config.alpha = Some(alpha);
    }
    config
}

/// Parse an OSD method string into the Rust enum.
fn parse_osd_method(s: &str) -> PyResult<RustOsdMethod> {
    match s {
        "off" => Ok(RustOsdMethod::Off),
        "osd0" => Ok(RustOsdMethod::Osd0),
        "osd_e" => Ok(RustOsdMethod::OsdE),
        "osd_cs" => Ok(RustOsdMethod::OsdCs),
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "osd_method must be 'off', 'osd0', 'osd_e', or 'osd_cs'",
        )),
    }
}

/// Builder for BP+OSD decoder.
///
/// Belief Propagation with Ordered Statistics Decoding post-processing.
///
/// # Example
///
/// ```python
/// from pecos_rslib.decoders import BpOsdBuilder, SparseMatrix
///
/// H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
/// decoder = BpOsdBuilder(H, error_rate=0.1).osd_method("osd_cs").osd_order(7).build()
/// result = decoder.decode_syndrome(syndrome)
/// ```
#[pyclass(name = "BpOsdBuilder", module = "pecos_rslib.decoders")]
pub struct PyBpOsdBuilder {
    pcm: RustSparseMatrix,
    error_rate: f64,
    max_iter: usize,
    bp_method: String,
    schedule: String,
    osd_method: String,
    osd_order: usize,
}

#[pymethods]
impl PyBpOsdBuilder {
    /// Create a new BP+OSD builder.
    ///
    /// # Arguments
    ///
    /// * `pcm` - Parity check matrix
    /// * `error_rate` - Channel error probability
    #[new]
    fn new(pcm: &PySparseMatrix, error_rate: f64) -> Self {
        Self {
            pcm: pcm.inner.clone(),
            error_rate,
            max_iter: 100,
            bp_method: "product_sum".to_string(),
            schedule: "parallel".to_string(),
            osd_method: "osd0".to_string(),
            osd_order: 0,
        }
    }

    /// Set maximum BP iterations (default: 100).
    fn max_iter(mut slf: PyRefMut<'_, Self>, val: usize) -> PyRefMut<'_, Self> {
        slf.max_iter = val;
        slf
    }

    /// Set BP algorithm: "`product_sum`" or "`minimum_sum`" (default: "`product_sum`").
    fn bp_method(mut slf: PyRefMut<'_, Self>, val: String) -> PyRefMut<'_, Self> {
        slf.bp_method = val;
        slf
    }

    /// Set update schedule: "parallel" or "serial" (default: "parallel").
    fn schedule(mut slf: PyRefMut<'_, Self>, val: String) -> PyRefMut<'_, Self> {
        slf.schedule = val;
        slf
    }

    /// Set OSD variant: "off", "osd0", "`osd_e`", "`osd_cs`" (default: "osd0").
    fn osd_method(mut slf: PyRefMut<'_, Self>, val: String) -> PyRefMut<'_, Self> {
        slf.osd_method = val;
        slf
    }

    /// Set OSD order parameter (default: 0).
    fn osd_order(mut slf: PyRefMut<'_, Self>, val: usize) -> PyRefMut<'_, Self> {
        slf.osd_order = val;
        slf
    }

    /// Build the BP+OSD decoder.
    fn build(&self) -> PyResult<PyBpOsdDecoder> {
        let bp = parse_bp_method(&self.bp_method)?;
        let bp_schedule = parse_bp_schedule(&self.schedule)?;
        let osd = parse_osd_method(&self.osd_method)?;

        RustBpOsdDecoder::new(
            &self.pcm,
            Some(self.error_rate),
            None,
            self.max_iter,
            bp,
            bp_schedule,
            1.0,
            osd,
            self.osd_order,
            RustInputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .map(|inner| PyBpOsdDecoder { inner })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "BpOsdBuilder(rows={}, cols={})",
            self.pcm.rows, self.pcm.cols
        )
    }
}

/// BP+OSD decoder for LDPC codes.
///
/// Created via `BpOsdBuilder(...).build()`.
// Note: unsendable because contains FFI pointers
#[pyclass(name = "BpOsdDecoder", module = "pecos_rslib.decoders", unsendable)]
pub struct PyBpOsdDecoder {
    inner: RustBpOsdDecoder,
}

#[pymethods]
impl PyBpOsdDecoder {
    /// Create a DEM-aware BP+OSD decoder from a Detector Error Model.
    ///
    /// * `error_rate` - Uniform prior override; model mismatch can reduce accuracy, with little runtime effect
    /// * `max_iter` - BP iteration cap; larger values can improve convergence but increase runtime
    /// * `bp_schedule` - Update ordering; serial may converge sooner while parallel favors throughput
    /// * `ms_scaling_factor` - Select minimum-sum BP and set its correction factor;
    ///   tuning can improve accuracy with negligible runtime cost
    /// * `osd_order` - Combination-sweep order; larger values can improve accuracy at steep runtime cost
    /// * `random_schedule_seed` - Reproducible randomized scheduling; changes exploration, not its runtime bound
    #[staticmethod]
    #[pyo3(signature = (dem, error_rate=None, max_iter=None, bp_schedule=None, ms_scaling_factor=None, osd_order=None, random_schedule_seed=None))]
    fn from_dem(
        dem: &str,
        error_rate: Option<f64>,
        max_iter: Option<i128>,
        bp_schedule: Option<&str>,
        ms_scaling_factor: Option<f64>,
        osd_order: Option<i128>,
        random_schedule_seed: Option<i64>,
    ) -> PyResult<PyDemAwareDecoder> {
        let config = bp_osd_config(
            error_rate,
            optional_usize(max_iter, "max_iter")?,
            bp_schedule,
            ms_scaling_factor,
            optional_usize(osd_order, "osd_order")?,
            optional_i32(random_schedule_seed, "random_schedule_seed")?,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        PyDemAwareDecoder::from_dem_with_config(dem, DemDecoderConfig::BpOsd(config))
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
    fn decode_syndrome(&mut self, syndrome: Vec<u8>) -> PyResult<PyBpResult> {
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

    #[allow(clippy::unused_self)] // Python instance method
    fn __repr__(&self) -> String {
        "BpOsdDecoder(...)".to_string()
    }

    fn __getattr__(&self, name: &str) -> PyResult<()> {
        Err(explicit_decode_attribute_error("BpOsdDecoder", name))
    }
}

/// Builder for BP+LSD decoder.
///
/// Belief Propagation with Localized Statistics Decoding.
/// Often faster than OSD for similar accuracy.
///
/// # Example
///
/// ```python
/// from pecos_rslib.decoders import BpLsdBuilder, SparseMatrix
///
/// H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
/// decoder = BpLsdBuilder(H, error_rate=0.1).lsd_order(2).build()
/// result = decoder.decode(syndrome)
/// ```
#[pyclass(name = "BpLsdBuilder", module = "pecos_rslib.decoders")]
pub struct PyBpLsdBuilder {
    pcm: RustSparseMatrix,
    error_rate: f64,
    max_iter: usize,
    bp_method: String,
    schedule: String,
    lsd_order: usize,
}

#[pymethods]
impl PyBpLsdBuilder {
    /// Create a new BP+LSD builder.
    ///
    /// # Arguments
    ///
    /// * `pcm` - Parity check matrix
    /// * `error_rate` - Channel error probability
    #[new]
    fn new(pcm: &PySparseMatrix, error_rate: f64) -> Self {
        Self {
            pcm: pcm.inner.clone(),
            error_rate,
            max_iter: 100,
            bp_method: "product_sum".to_string(),
            schedule: "parallel".to_string(),
            lsd_order: 0,
        }
    }

    /// Set maximum BP iterations (default: 100).
    fn max_iter(mut slf: PyRefMut<'_, Self>, val: usize) -> PyRefMut<'_, Self> {
        slf.max_iter = val;
        slf
    }

    /// Set BP algorithm: "`product_sum`" or "`minimum_sum`" (default: "`product_sum`").
    fn bp_method(mut slf: PyRefMut<'_, Self>, val: String) -> PyRefMut<'_, Self> {
        slf.bp_method = val;
        slf
    }

    /// Set update schedule: "parallel" or "serial" (default: "parallel").
    fn schedule(mut slf: PyRefMut<'_, Self>, val: String) -> PyRefMut<'_, Self> {
        slf.schedule = val;
        slf
    }

    /// Set LSD order parameter (default: 0).
    fn lsd_order(mut slf: PyRefMut<'_, Self>, val: usize) -> PyRefMut<'_, Self> {
        slf.lsd_order = val;
        slf
    }

    /// Build the BP+LSD decoder.
    fn build(&self) -> PyResult<PyBpLsdDecoder> {
        let bp = parse_bp_method(&self.bp_method)?;
        let bp_schedule = parse_bp_schedule(&self.schedule)?;

        RustBpLsdDecoder::new(
            &self.pcm,
            Some(self.error_rate),
            None,
            self.max_iter,
            bp,
            bp_schedule,
            1.0,
            RustOsdMethod::Osd0,
            self.lsd_order,
            0,
            RustInputVectorType::Syndrome,
            None,
            None,
            None,
        )
        .map(|inner| PyBpLsdDecoder { inner })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "BpLsdBuilder(rows={}, cols={})",
            self.pcm.rows, self.pcm.cols
        )
    }
}

/// BP+LSD decoder for LDPC codes.
///
/// Created via `BpLsdBuilder(...).build()`.
// Note: unsendable because contains FFI pointers
#[pyclass(name = "BpLsdDecoder", module = "pecos_rslib.decoders", unsendable)]
pub struct PyBpLsdDecoder {
    inner: RustBpLsdDecoder,
}

#[pymethods]
impl PyBpLsdDecoder {
    /// Create a DEM-aware BP+LSD decoder from a Detector Error Model.
    ///
    /// * `error_rate` - Uniform prior override; model mismatch can reduce accuracy, with little runtime effect
    /// * `max_iter` - BP iteration cap; larger values can improve convergence but increase runtime
    /// * `bp_schedule` - Update ordering; serial may converge sooner while parallel favors throughput
    /// * `ms_scaling_factor` - Select minimum-sum BP and set its correction factor;
    ///   tuning can improve accuracy with negligible runtime cost
    /// * `random_schedule_seed` - Reproducible randomized scheduling; changes exploration, not its runtime bound
    #[staticmethod]
    #[pyo3(signature = (dem, error_rate=None, max_iter=None, bp_schedule=None, ms_scaling_factor=None, random_schedule_seed=None))]
    fn from_dem(
        dem: &str,
        error_rate: Option<f64>,
        max_iter: Option<i128>,
        bp_schedule: Option<&str>,
        ms_scaling_factor: Option<f64>,
        random_schedule_seed: Option<i64>,
    ) -> PyResult<PyDemAwareDecoder> {
        let config = bp_lsd_config(
            error_rate,
            optional_usize(max_iter, "max_iter")?,
            bp_schedule,
            ms_scaling_factor,
            optional_i32(random_schedule_seed, "random_schedule_seed")?,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        PyDemAwareDecoder::from_dem_with_config(dem, DemDecoderConfig::BpLsd(config))
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

    #[allow(clippy::unused_self)] // Python instance method
    fn __repr__(&self) -> String {
        "BpLsdDecoder(...)".to_string()
    }
}

/// Builder for Union-Find decoder.
///
/// Cluster-based decoder using the Union-Find data structure.
/// Fast O(n * alpha(n)) complexity per syndrome.
///
/// # Example
///
/// ```python
/// from pecos_rslib.decoders import UnionFindBuilder, SparseMatrix
///
/// H = SparseMatrix([[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]])
/// decoder = UnionFindBuilder(H).method("peeling").build()
/// result = decoder.decode_syndrome(syndrome)
/// ```
#[pyclass(name = "UnionFindBuilder", module = "pecos_rslib.decoders")]
pub struct PyUnionFindBuilder {
    pcm: RustSparseMatrix,
    method: String,
}

#[pymethods]
impl PyUnionFindBuilder {
    /// Create a new Union-Find builder.
    ///
    /// # Arguments
    ///
    /// * `pcm` - Parity check matrix
    #[new]
    fn new(pcm: &PySparseMatrix) -> Self {
        Self {
            pcm: pcm.inner.clone(),
            method: "inversion".to_string(),
        }
    }

    /// Set decoding method: "inversion" (general) or "peeling" (LDPC only).
    fn method(mut slf: PyRefMut<'_, Self>, val: String) -> PyRefMut<'_, Self> {
        slf.method = val;
        slf
    }

    /// Build the Union-Find decoder.
    fn build(&self) -> PyResult<PyUnionFindDecoder> {
        let uf_method = match self.method.as_str() {
            "inversion" => RustUfMethod::Inversion,
            "peeling" => RustUfMethod::Peeling,
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "method must be 'inversion' or 'peeling'",
                ));
            }
        };

        RustUnionFindDecoder::new(&self.pcm, uf_method)
            .map(|inner| PyUnionFindDecoder { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        format!(
            "UnionFindBuilder(rows={}, cols={})",
            self.pcm.rows, self.pcm.cols
        )
    }
}

/// Union-Find decoder for LDPC codes.
///
/// Created via `UnionFindBuilder(...).build()`.
// Note: unsendable because contains FFI pointers
#[pyclass(name = "UnionFindDecoder", module = "pecos_rslib.decoders", unsendable)]
pub struct PyUnionFindDecoder {
    inner: RustUnionFindDecoder,
}

#[pymethods]
impl PyUnionFindDecoder {
    /// Create a DEM-aware Union-Find decoder from a Detector Error Model.
    ///
    /// * `method` - "peeling" is faster on compatible LDPC matrices; "inversion" is more general
    #[staticmethod]
    #[pyo3(signature = (dem, method=None))]
    fn from_dem(dem: &str, method: Option<&str>) -> PyResult<PyDemAwareDecoder> {
        let config =
            union_find_config(method).map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;
        PyDemAwareDecoder::from_dem_with_config(dem, DemDecoderConfig::UnionFind(config))
    }

    /// Decode a syndrome.
    ///
    /// # Arguments
    ///
    /// * `syndrome` - Syndrome vector
    /// * `llrs` - Optional log-likelihood ratios for soft information
    /// * `bits_per_step` - Bits to grow per step (0 = all at once)
    #[pyo3(signature = (syndrome, llrs=None, bits_per_step=0))]
    fn decode_syndrome(
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

    #[allow(clippy::unused_self)] // Python instance method
    fn __repr__(&self) -> String {
        "UnionFindDecoder(...)".to_string()
    }

    fn __getattr__(&self, name: &str) -> PyResult<()> {
        Err(explicit_decode_attribute_error("UnionFindDecoder", name))
    }
}

// =============================================================================
// Tesseract Decoder
// =============================================================================

use pecos_decoders::{
    TesseractConfig as RustTesseractConfig, TesseractDecoder as RustTesseractDecoder,
};

fn tesseract_config(
    preset: &str,
    det_beam: Option<u16>,
    beam_climbing: Option<bool>,
    verbose: Option<bool>,
    no_revisit_dets: Option<bool>,
    pqlimit: Option<usize>,
    det_penalty: Option<f64>,
) -> Result<RustTesseractConfig, String> {
    let mut config = match preset {
        "fast" => RustTesseractConfig::fast(),
        "accurate" => RustTesseractConfig::accurate(),
        "default" => RustTesseractConfig::default(),
        _ => return Err("preset must be 'default', 'fast', or 'accurate'".to_string()),
    };

    if let Some(det_beam) = det_beam {
        config.det_beam = det_beam;
    }
    if let Some(beam_climbing) = beam_climbing {
        config.beam_climbing = beam_climbing;
    }
    if let Some(verbose) = verbose {
        config.verbose = verbose;
    }
    if let Some(no_revisit_dets) = no_revisit_dets {
        config.no_revisit_dets = no_revisit_dets;
    }
    if let Some(pqlimit) = pqlimit {
        config.pqlimit = pqlimit;
    }
    if let Some(det_penalty) = det_penalty {
        config.det_penalty = det_penalty;
    }

    Ok(config)
}

/// Result from Tesseract decoder.
///
/// # Attributes
///
/// * `observable_flips` - Observables affected by predicted errors
/// * `cost` - Total cost of the solution
/// * `low_confidence` - Whether this is a low-confidence prediction
#[pyclass(
    name = "TesseractResult",
    module = "pecos_rslib.decoders",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyTesseractResult {
    observables_mask: u64,
    #[pyo3(get)]
    cost: f64,
    #[pyo3(get)]
    low_confidence: bool,
    num_observables: usize,
}

#[pymethods]
impl PyTesseractResult {
    /// The decoded observable flips with the decoder's observable count.
    #[getter]
    fn observable_flips(&self) -> PyObservableFlips {
        PyObservableFlips::from_mask_value(
            pecos_decoder_core::obs_mask::ObsMask::from_u64(self.observables_mask),
            self.num_observables,
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "TesseractResult(observable_flips=ObservableFlips(num_observables={}, mask={}), cost={:.4}, low_confidence={})",
            self.num_observables, self.observables_mask, self.cost, self.low_confidence
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
/// result = decoder.decode_from_defects(detection_indices)
/// print(f"Observable mask: {result.observable_flips.mask}, Cost: {result.cost}")
/// ```
#[pyclass(name = "TesseractDecoder", module = "pecos_rslib.decoders", unsendable)]
pub struct PyTesseractDecoder {
    inner: RustTesseractDecoder,
    dem_string: String,
    config: RustTesseractConfig,
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
    /// * `verbose` - Enable verbose output; no accuracy/runtime tradeoff when disabled
    /// * `no_revisit_dets` - Avoid revisiting detectors, reducing runtime at possible accuracy cost
    /// * `pqlimit` - Priority queue entry cap; smaller values bound memory at possible accuracy cost
    /// * `det_penalty` - Search penalty for adding detectors; larger values prune more aggressively
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
    #[pyo3(signature = (dem, preset="default", det_beam=None, beam_climbing=None, verbose=None, no_revisit_dets=None, pqlimit=None, det_penalty=None))]
    fn from_dem(
        dem: &str,
        preset: &str,
        det_beam: Option<i64>,
        beam_climbing: Option<bool>,
        verbose: Option<bool>,
        no_revisit_dets: Option<bool>,
        pqlimit: Option<i128>,
        det_penalty: Option<f64>,
    ) -> PyResult<Self> {
        let det_beam = optional_u16(det_beam, "det_beam")?;
        let pqlimit = optional_usize(pqlimit, "pqlimit")?;
        let config = tesseract_config(
            preset,
            det_beam,
            beam_climbing,
            verbose,
            no_revisit_dets,
            pqlimit,
            det_penalty,
        )
        .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?;

        let dem_string = dem.to_string();
        RustTesseractDecoder::new(dem, config.clone())
            .map(|inner| Self {
                inner,
                dem_string,
                config,
            })
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
    /// `TesseractResult` with observable flips, cost, and confidence info.
    ///
    /// # Example
    ///
    /// ```python
    /// # Detectors 0 and 2 fired
    /// result = decoder.decode_from_defects([0, 2])
    /// print(f"Observable prediction: {list(result.observable_flips)}")
    /// ```
    fn decode_from_defects(&mut self, detections: Vec<u64>) -> PyResult<PyTesseractResult> {
        let detections_arr = ndarray::Array1::from_vec(detections);
        let num_observables = self.inner.num_observables();

        self.inner
            .decode_detections(&detections_arr.view())
            .map(|result| PyTesseractResult {
                observables_mask: result.observables_mask,
                cost: result.cost,
                low_confidence: result.low_confidence,
                num_observables,
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

        self.decode_from_defects(detections)
    }

    /// Decode a batch of syndromes in parallel using multiple decoder instances.
    ///
    /// Creates worker decoders on background threads and distributes shots
    /// across them. Much faster than sequential decoding for large batches.
    ///
    /// # Arguments
    ///
    /// * `syndromes` - List of dense syndrome vectors
    /// * `num_workers` - Number of parallel workers (default: number of CPUs)
    ///
    /// # Returns
    ///
    /// List of `TesseractResult` in the same order as inputs.
    #[pyo3(signature = (syndromes, num_workers=None))]
    fn decode_batch(
        &self,
        syndromes: Vec<Vec<u8>>,
        num_workers: Option<usize>,
    ) -> PyResult<Vec<PyTesseractResult>> {
        use rayon::prelude::*;

        let n_workers = num_workers.unwrap_or_else(rayon::current_num_threads);

        // Build a thread pool with the requested size
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let dem_str = &self.dem_string;
        let config = &self.config;
        let num_observables = self.inner.num_observables();

        let results: Result<Vec<_>, _> = pool.install(|| {
            syndromes
                .par_iter()
                .map(|syndrome| {
                    // Each rayon task gets its own thread-local decoder
                    thread_local! {
                        static DECODER: std::cell::RefCell<Option<RustTesseractDecoder>> =
                            const { std::cell::RefCell::new(None) };
                    }

                    DECODER.with(|cell| {
                        let mut decoder_ref = cell.borrow_mut();
                        if decoder_ref.is_none() {
                            *decoder_ref = Some(
                                RustTesseractDecoder::new(dem_str, config.clone())
                                    .map_err(|e| e.to_string())?,
                            );
                        }
                        let decoder = decoder_ref.as_mut().unwrap();

                        // Convert dense to sparse
                        let detections: Vec<u64> = syndrome
                            .iter()
                            .enumerate()
                            .filter_map(|(i, &val)| if val != 0 { Some(i as u64) } else { None })
                            .collect();

                        let detections_arr = ndarray::Array1::from_vec(detections);
                        decoder
                            .decode_detections(&detections_arr.view())
                            .map(|r| PyTesseractResult {
                                observables_mask: r.observables_mask,
                                cost: r.cost,
                                low_confidence: r.low_confidence,
                                num_observables,
                            })
                            .map_err(|e| e.to_string())
                    })
                })
                .collect()
        });

        results.map_err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>)
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

    fn __getattr__(&self, name: &str) -> PyResult<()> {
        Err(explicit_decode_attribute_error("TesseractDecoder", name))
    }
}

// =============================================================================
// Relay BP Decoders
// =============================================================================

use pecos_decoders::{
    MinSumBpBuilder as RustMinSumBpBuilder, MinSumBpDecoder as RustMinSumBpDecoder,
    RelayBpBuilder as RustRelayBpBuilder, RelayBpDecoder as RustRelayBpDecoder,
    StoppingCriterion as RustStoppingCriterion,
};

/// Parse a stopping criterion string into the Rust enum.
///
/// Supported values:
/// - `"pre_iter"` -> `StoppingCriterion::PreIter`
/// - `"all"` -> `StoppingCriterion::All`
/// - `"n_conv_1"` -> `StoppingCriterion::NConv { stop_after: 1 }` (default)
/// - `"n_conv_N"` (e.g., `"n_conv_5"`) -> `StoppingCriterion::NConv { stop_after: N }`
fn parse_stopping_criterion(s: &str) -> PyResult<RustStoppingCriterion> {
    match s {
        "pre_iter" => Ok(RustStoppingCriterion::PreIter),
        "all" => Ok(RustStoppingCriterion::All),
        _ if s.starts_with("n_conv_") => {
            let n_str = &s["n_conv_".len()..];
            let n: usize = n_str.parse().map_err(|_| {
                PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Invalid stopping criterion '{s}': expected 'n_conv_N' where N is a positive integer"
                ))
            })?;
            Ok(RustStoppingCriterion::NConv { stop_after: n })
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unknown stopping criterion '{s}'. Valid: 'pre_iter', 'all', 'n_conv_1', 'n_conv_N'"
        ))),
    }
}

/// Convert a dense check matrix from Python lists to an ndarray `Array2`.
///
/// # Errors
///
/// Returns `PyValueError` if the rows have inconsistent lengths.
fn dense_check_matrix_to_array2(check_matrix: &[Vec<u8>]) -> PyResult<Array2<u8>> {
    let rows = check_matrix.len();
    let cols = if rows > 0 { check_matrix[0].len() } else { 0 };
    for (i, row) in check_matrix.iter().enumerate() {
        if row.len() != cols {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                "check_matrix row {i} has length {} but row 0 has length {cols}",
                row.len()
            )));
        }
    }
    let mut arr = Array2::<u8>::zeros((rows, cols));
    for (i, row) in check_matrix.iter().enumerate() {
        for (j, &val) in row.iter().enumerate() {
            arr[[i, j]] = val;
        }
    }
    Ok(arr)
}

/// Builder for Relay BP ensemble decoder.
///
/// Configures and constructs a `RelayBpDecoder` for qLDPC codes. Uses an
/// ensemble of min-sum BP decoders with randomized damping parameters (relay
/// strategy) to improve convergence on codes where standard BP fails.
///
/// # Example
///
/// ```python
/// from pecos_rslib.decoders import RelayBpBuilder
///
/// H = [[1, 1, 0], [0, 1, 1]]
/// decoder = (
///     RelayBpBuilder(H, [0.003, 0.003, 0.003])
///     .seed(42)
///     .num_sets(100)
///     .build()
/// )
/// result = decoder.decode([1, 0])
/// ```
#[pyclass(name = "RelayBpBuilder", module = "pecos_rslib.decoders")]
pub struct PyRelayBpBuilder {
    check_matrix: Vec<Vec<u8>>,
    error_priors: Vec<f64>,
    max_iter: usize,
    alpha: Option<f64>,
    gamma0: Option<f64>,
    pre_iter: usize,
    num_sets: usize,
    set_max_iter: usize,
    seed: u64,
    stopping: String,
}

#[pymethods]
impl PyRelayBpBuilder {
    /// Create a new Relay BP builder.
    ///
    /// # Arguments
    ///
    /// * `check_matrix` - Parity check matrix as list of lists
    /// * `error_priors` - Prior error probabilities for each bit
    #[new]
    fn new(check_matrix: Vec<Vec<u8>>, error_priors: Vec<f64>) -> Self {
        Self {
            check_matrix,
            error_priors,
            max_iter: 200,
            alpha: None,
            gamma0: None,
            pre_iter: 80,
            num_sets: 300,
            set_max_iter: 60,
            seed: 0,
            stopping: "n_conv_1".to_string(),
        }
    }

    /// Set maximum BP iterations (default: 200).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn max_iter(mut slf: PyRefMut<'_, Self>, val: usize) -> PyRefMut<'_, Self> {
        slf.max_iter = val;
        slf
    }

    /// Set min-sum scaling factor (None = no scaling).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn alpha(mut slf: PyRefMut<'_, Self>, val: Option<f64>) -> PyRefMut<'_, Self> {
        slf.alpha = val;
        slf
    }

    /// Set initial damping factor (None = disabled).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn gamma0(mut slf: PyRefMut<'_, Self>, val: Option<f64>) -> PyRefMut<'_, Self> {
        slf.gamma0 = val;
        slf
    }

    /// Set number of pre-relay BP iterations (default: 80).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn pre_iter(mut slf: PyRefMut<'_, Self>, val: usize) -> PyRefMut<'_, Self> {
        slf.pre_iter = val;
        slf
    }

    /// Set number of relay sets/legs (default: 300).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn num_sets(mut slf: PyRefMut<'_, Self>, val: usize) -> PyRefMut<'_, Self> {
        slf.num_sets = val;
        slf
    }

    /// Set max iterations per relay set (default: 60).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn set_max_iter(mut slf: PyRefMut<'_, Self>, val: usize) -> PyRefMut<'_, Self> {
        slf.set_max_iter = val;
        slf
    }

    /// Set random seed for relay parameter sampling (default: 0).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn seed(mut slf: PyRefMut<'_, Self>, val: u64) -> PyRefMut<'_, Self> {
        slf.seed = val;
        slf
    }

    /// Set stopping criterion (default: `"n_conv_1"`).
    ///
    /// Valid values: `"pre_iter"`, `"all"`, `"n_conv_1"`, `"n_conv_N"`.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn stopping(mut slf: PyRefMut<'_, Self>, val: String) -> PyRefMut<'_, Self> {
        slf.stopping = val;
        slf
    }

    /// Build the Relay BP decoder.
    ///
    /// Returns:
    ///     A `RelayBpDecoder` ready for decoding.
    ///
    /// Raises:
    ///     `RuntimeError`: If the configuration is invalid.
    fn build(&self) -> PyResult<PyRelayBpDecoder> {
        let stopping_criterion = parse_stopping_criterion(&self.stopping)?;
        let arr = dense_check_matrix_to_array2(&self.check_matrix)?;

        RustRelayBpBuilder::new(&arr.view())
            .error_priors(&self.error_priors)
            .max_iter(self.max_iter)
            .alpha(self.alpha)
            .gamma0(self.gamma0)
            .pre_iter(self.pre_iter)
            .num_sets(self.num_sets)
            .set_max_iter(self.set_max_iter)
            .seed(self.seed)
            .stopping_criterion(stopping_criterion)
            .build()
            .map(|inner| PyRelayBpDecoder { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        let rows = self.check_matrix.len();
        let cols = if rows > 0 {
            self.check_matrix[0].len()
        } else {
            0
        };
        format!("RelayBpBuilder(checks={rows}, bits={cols})")
    }
}

/// Relay BP ensemble decoder for qLDPC codes.
///
/// Created via `RelayBpBuilder(...).build()`.
///
/// # Example
///
/// ```python
/// from pecos_rslib.decoders import RelayBpBuilder
///
/// decoder = RelayBpBuilder([[1,1,0],[0,1,1]], [0.003]*3).seed(42).build()
/// result = decoder.decode([1, 0])
/// assert result.converged
/// ```
#[pyclass(name = "RelayBpDecoder", module = "pecos_rslib.decoders")]
pub struct PyRelayBpDecoder {
    inner: RustRelayBpDecoder,
}

#[pymethods]
impl PyRelayBpDecoder {
    /// Create a DEM-aware Relay BP decoder from a Detector Error Model.
    ///
    /// * `error_rate` - Uniform prior override; model mismatch can reduce accuracy, with little runtime effect
    /// * `max_iter` - BP iteration cap; larger values can improve convergence but increase runtime
    /// * `alpha` - Min-sum scaling factor; tuning can improve accuracy with negligible runtime cost
    /// * `seed` - Reproducible relay sampling; changes exploration without increasing its runtime bound
    #[staticmethod]
    #[pyo3(signature = (dem, error_rate=None, max_iter=None, alpha=None, seed=None))]
    fn from_dem(
        dem: &str,
        error_rate: Option<f64>,
        max_iter: Option<i128>,
        alpha: Option<f64>,
        seed: Option<i128>,
    ) -> PyResult<PyDemAwareDecoder> {
        let config = relay_bp_config(
            error_rate,
            optional_usize(max_iter, "max_iter")?,
            alpha,
            optional_u64(seed, "seed")?,
        );
        PyDemAwareDecoder::from_dem_with_config(dem, DemDecoderConfig::RelayBp(config))
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

    /// Number of checks (rows in check matrix).
    #[getter]
    fn check_count(&self) -> usize {
        self.inner.check_count()
    }

    /// Number of bits (columns in check matrix).
    #[getter]
    fn bit_count(&self) -> usize {
        self.inner.bit_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "RelayBpDecoder(checks={}, bits={})",
            self.inner.check_count(),
            self.inner.bit_count()
        )
    }
}

/// Builder for min-sum BP decoder.
///
/// Configures and constructs a `MinSumBpDecoder` for qLDPC codes. Standard
/// min-sum belief propagation -- simpler and faster than `RelayBpDecoder`
/// for codes where plain BP converges.
///
/// # Example
///
/// ```python
/// from pecos_rslib.decoders import MinSumBpBuilder
///
/// H = [[1, 1, 0], [0, 1, 1]]
/// decoder = MinSumBpBuilder(H, [0.003, 0.003, 0.003]).max_iter(100).build()
/// result = decoder.decode([1, 0])
/// ```
#[pyclass(name = "MinSumBpBuilder", module = "pecos_rslib.decoders")]
pub struct PyMinSumBpBuilder {
    check_matrix: Vec<Vec<u8>>,
    error_priors: Vec<f64>,
    max_iter: usize,
    alpha: Option<f64>,
    gamma0: Option<f64>,
}

#[pymethods]
impl PyMinSumBpBuilder {
    /// Create a new min-sum BP builder.
    ///
    /// # Arguments
    ///
    /// * `check_matrix` - Parity check matrix as list of lists
    /// * `error_priors` - Prior error probabilities for each bit
    #[new]
    fn new(check_matrix: Vec<Vec<u8>>, error_priors: Vec<f64>) -> Self {
        Self {
            check_matrix,
            error_priors,
            max_iter: 200,
            alpha: None,
            gamma0: None,
        }
    }

    /// Set maximum BP iterations (default: 200).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn max_iter(mut slf: PyRefMut<'_, Self>, val: usize) -> PyRefMut<'_, Self> {
        slf.max_iter = val;
        slf
    }

    /// Set min-sum scaling factor (None = no scaling).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn alpha(mut slf: PyRefMut<'_, Self>, val: Option<f64>) -> PyRefMut<'_, Self> {
        slf.alpha = val;
        slf
    }

    /// Set initial damping factor (None = disabled).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn gamma0(mut slf: PyRefMut<'_, Self>, val: Option<f64>) -> PyRefMut<'_, Self> {
        slf.gamma0 = val;
        slf
    }

    /// Build the min-sum BP decoder.
    ///
    /// Returns:
    ///     A `MinSumBpDecoder` ready for decoding.
    ///
    /// Raises:
    ///     `RuntimeError`: If the configuration is invalid.
    fn build(&self) -> PyResult<PyMinSumBpDecoder> {
        let arr = dense_check_matrix_to_array2(&self.check_matrix)?;

        RustMinSumBpBuilder::new(&arr.view())
            .error_priors(&self.error_priors)
            .max_iter(self.max_iter)
            .alpha(self.alpha)
            .gamma0(self.gamma0)
            .build()
            .map(|inner| PyMinSumBpDecoder { inner })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn __repr__(&self) -> String {
        let rows = self.check_matrix.len();
        let cols = if rows > 0 {
            self.check_matrix[0].len()
        } else {
            0
        };
        format!("MinSumBpBuilder(checks={rows}, bits={cols})")
    }
}

/// Min-sum BP decoder for qLDPC codes.
///
/// Created via `MinSumBpBuilder(...).build()`.
///
/// # Example
///
/// ```python
/// from pecos_rslib.decoders import MinSumBpBuilder
///
/// decoder = MinSumBpBuilder([[1,1,0],[0,1,1]], [0.003]*3).build()
/// result = decoder.decode([1, 0])
/// assert result.converged
/// ```
#[pyclass(name = "MinSumBpDecoder", module = "pecos_rslib.decoders")]
pub struct PyMinSumBpDecoder {
    inner: RustMinSumBpDecoder,
}

#[pymethods]
impl PyMinSumBpDecoder {
    /// Create a DEM-aware min-sum BP decoder from a Detector Error Model.
    ///
    /// * `error_rate` - Uniform prior override; model mismatch can reduce accuracy, with little runtime effect
    /// * `max_iter` - BP iteration cap; larger values can improve convergence but increase runtime
    /// * `alpha` - Min-sum scaling factor; tuning can improve accuracy with negligible runtime cost
    #[staticmethod]
    #[pyo3(signature = (dem, error_rate=None, max_iter=None, alpha=None))]
    fn from_dem(
        dem: &str,
        error_rate: Option<f64>,
        max_iter: Option<i128>,
        alpha: Option<f64>,
    ) -> PyResult<PyDemAwareDecoder> {
        let config = min_sum_bp_config(error_rate, optional_usize(max_iter, "max_iter")?, alpha);
        PyDemAwareDecoder::from_dem_with_config(dem, DemDecoderConfig::MinSumBp(config))
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

    /// Number of checks (rows in check matrix).
    #[getter]
    fn check_count(&self) -> usize {
        self.inner.check_count()
    }

    /// Number of bits (columns in check matrix).
    #[getter]
    fn bit_count(&self) -> usize {
        self.inner.bit_count()
    }

    fn __repr__(&self) -> String {
        format!(
            "MinSumBpDecoder(checks={}, bits={})",
            self.inner.check_count(),
            self.inner.bit_count()
        )
    }
}

// =============================================================================
// DEM-Aware Decoder (wraps check-matrix decoders for DEM-level decoding)
// =============================================================================

use pecos_decoder_core::DemCheckMatrix;

/// Decoder type for the DEM-aware wrapper.
enum InnerDecoder {
    BpOsd(RustBpOsdDecoder),
    BpLsd(RustBpLsdDecoder),
    UnionFind(RustUnionFindDecoder),
    RelayBp(Box<RustRelayBpDecoder>),
    MinSumBp(Box<RustMinSumBpDecoder>),
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum DemDecoderConfig {
    BpOsd(BpOsdDemConfig),
    BpLsd(BpLsdDemConfig),
    UnionFind(UnionFindDemConfig),
    RelayBp(RelayBpDemConfig),
    MinSumBp(MinSumBpDemConfig),
}

impl DemDecoderConfig {
    fn error_rate(self) -> Option<f64> {
        match self {
            Self::BpOsd(config) => config.error_rate,
            Self::BpLsd(config) => config.error_rate,
            Self::UnionFind(_) => None,
            Self::RelayBp(config) => config.error_rate,
            Self::MinSumBp(config) => config.error_rate,
        }
    }
}

/// DEM-aware decoder that wraps a check-matrix decoder.
///
/// Parses a DEM string, extracts the check matrix and observable matrix,
/// creates the inner decoder, and provides `decode_syndrome()` that returns
/// `observable_flips` -- the same interface as `PyMatching` and Tesseract.
///
/// # Example
///
/// ```python
/// from pecos_rslib.decoders import DemAwareDecoder
///
/// decoder = DemAwareDecoder.from_dem(dem_string, decoder_type="bp_osd")
/// result = decoder.decode_syndrome([0, 1, 1, 0])
/// print(f"Observable prediction: {result.observable_flips}")
/// ```
#[pyclass(name = "DemAwareDecoder", module = "pecos_rslib.decoders", unsendable)]
pub struct PyDemAwareDecoder {
    inner: InnerDecoder,
    dem_check_matrix: DemCheckMatrix,
}

impl PyDemAwareDecoder {
    fn parse_dem(dem: &str) -> PyResult<DemCheckMatrix> {
        let dcm = DemCheckMatrix::from_dem_str(dem)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyValueError, _>(e.to_string()))?;

        if dcm.num_mechanisms == 0 {
            return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "DEM contains no error mechanisms",
            ));
        }

        Ok(dcm)
    }

    fn from_dem_with_config(dem: &str, config: DemDecoderConfig) -> PyResult<Self> {
        let dcm = Self::parse_dem(dem)?;
        Self::from_dem_check_matrix_with_config(dcm, config)
    }

    fn from_dem_check_matrix_with_config(
        dcm: DemCheckMatrix,
        config: DemDecoderConfig,
    ) -> PyResult<Self> {
        // Error priors: use per-mechanism probabilities from DEM, or uniform override.
        let priors: Vec<f64> = if let Some(p) = config.error_rate() {
            vec![p; dcm.num_mechanisms]
        } else {
            dcm.error_priors.clone()
        };

        // The check matrix shape and observable map are structural properties of
        // the DEM and are deliberately never accepted as caller overrides.
        let sparse_h = RustSparseMatrix::from_dense(&dcm.check_matrix.view());

        let inner = match config {
            DemDecoderConfig::BpOsd(config) => {
                let decoder = RustBpOsdDecoder::new(
                    &sparse_h,
                    None,
                    Some(&priors),
                    config.max_iter,
                    config.bp_method,
                    config.bp_schedule,
                    config.ms_scaling_factor,
                    config.osd_method,
                    config.osd_order,
                    RustInputVectorType::Syndrome,
                    None,
                    None,
                    config.random_schedule_seed,
                )
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                InnerDecoder::BpOsd(decoder)
            }
            DemDecoderConfig::BpLsd(config) => {
                let decoder = RustBpLsdDecoder::new(
                    &sparse_h,
                    None,
                    Some(&priors),
                    config.max_iter,
                    config.bp_method,
                    config.bp_schedule,
                    config.ms_scaling_factor,
                    RustOsdMethod::Off,
                    0,
                    0,
                    RustInputVectorType::Syndrome,
                    None,
                    None,
                    config.random_schedule_seed,
                )
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                InnerDecoder::BpLsd(decoder)
            }
            DemDecoderConfig::UnionFind(config) => {
                let decoder = RustUnionFindDecoder::new(&sparse_h, config.method).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                InnerDecoder::UnionFind(decoder)
            }
            DemDecoderConfig::RelayBp(config) => {
                use pecos_decoders::RelayBpBuilder as RustRelayBpBuilderT;
                let h_view = dcm.check_matrix.view();
                let decoder = RustRelayBpBuilderT::new(&h_view)
                    .error_priors(&priors)
                    .max_iter(config.max_iter)
                    .alpha(config.alpha)
                    .seed(config.seed)
                    .build()
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                InnerDecoder::RelayBp(Box::new(decoder))
            }
            DemDecoderConfig::MinSumBp(config) => {
                use pecos_decoders::MinSumBpBuilder as RustMinSumBpBuilderT;
                let h_view = dcm.check_matrix.view();
                let decoder = RustMinSumBpBuilderT::new(&h_view)
                    .error_priors(&priors)
                    .max_iter(config.max_iter)
                    .alpha(config.alpha)
                    .build()
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                InnerDecoder::MinSumBp(Box::new(decoder))
            }
        };

        Ok(Self {
            inner,
            dem_check_matrix: dcm,
        })
    }
}

/// Result from a DEM-aware decoder.
#[pyclass(
    name = "DemAwareResult",
    module = "pecos_rslib.decoders",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyDemAwareResult {
    /// Predicted observable flips, wide enough for more than 64 observables.
    pub observables: pecos_decoder_core::obs_mask::ObsMask,
    /// Whether the BP decoder converged.
    #[pyo3(get)]
    pub converged: bool,
    /// Number of BP iterations used.
    #[pyo3(get)]
    pub iterations: usize,
    num_observables: usize,
}

impl PyDemAwareResult {
    /// Render the mask for `__repr__`: the plain integer when it fits in 64
    /// bits, otherwise the set observable indices.
    fn mask_display(&self) -> String {
        self.observables.to_u64().map_or_else(
            || {
                let bits: Vec<String> = self
                    .observables
                    .iter_set_bits()
                    .map(|bit| bit.to_string())
                    .collect();
                format!("<observables {}>", bits.join(","))
            },
            |value| value.to_string(),
        )
    }
}

#[pymethods]
impl PyDemAwareResult {
    /// The decoded observable flips with the decoder's observable count.
    #[getter]
    fn observable_flips(&self) -> PyObservableFlips {
        PyObservableFlips::from_mask_value(self.observables.clone(), self.num_observables)
    }

    fn __repr__(&self) -> String {
        format!(
            "DemAwareResult(observable_flips=ObservableFlips(num_observables={}, mask={}), converged={}, iterations={})",
            self.num_observables,
            self.mask_display(),
            self.converged,
            self.iterations
        )
    }
}

#[pymethods]
impl PyDemAwareDecoder {
    /// Create a DEM-aware decoder from a DEM string.
    ///
    /// # Arguments
    ///
    /// * `dem` - DEM string in Stim format
    /// * `decoder_type` - One of "`bp_osd`", "`bp_lsd`", "`union_find`", "`relay_bp`", "`min_sum_bp`"
    /// * `error_rate` - Override error rate for BP priors (default: use DEM probabilities)
    /// * `max_iter` - Maximum BP iterations (default: 100)
    ///
    /// # Example
    ///
    /// ```python
    /// decoder = DemAwareDecoder.from_dem(dem, decoder_type="bp_osd")
    /// ```
    #[staticmethod]
    #[pyo3(signature = (dem, decoder_type="bp_osd", error_rate=None, max_iter=100))]
    fn from_dem(
        dem: &str,
        decoder_type: &str,
        error_rate: Option<f64>,
        max_iter: usize,
    ) -> PyResult<Self> {
        let dcm = Self::parse_dem(dem)?;
        let config = match decoder_type {
            "bp_osd" => DemDecoderConfig::BpOsd(
                bp_osd_config(error_rate, Some(max_iter), None, None, None, None)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?,
            ),
            "bp_lsd" => DemDecoderConfig::BpLsd(
                bp_lsd_config(error_rate, Some(max_iter), None, None, None)
                    .map_err(PyErr::new::<pyo3::exceptions::PyValueError, _>)?,
            ),
            "union_find" => DemDecoderConfig::UnionFind(UnionFindDemConfig::default()),
            "relay_bp" => {
                DemDecoderConfig::RelayBp(relay_bp_config(error_rate, Some(max_iter), None, None))
            }
            "min_sum_bp" => {
                DemDecoderConfig::MinSumBp(min_sum_bp_config(error_rate, Some(max_iter), None))
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown decoder type: {decoder_type}. Supported: bp_osd, bp_lsd, union_find, relay_bp, min_sum_bp"
                )));
            }
        };
        Self::from_dem_check_matrix_with_config(dcm, config)
    }

    /// Decode a dense syndrome vector.
    ///
    /// # Arguments
    ///
    /// * `syndrome` - Dense syndrome vector (0 or 1 for each detector)
    ///
    /// # Returns
    ///
    /// `DemAwareResult` with `observable_flips`, `converged`, and `iterations`.
    fn decode_syndrome(&mut self, syndrome: Vec<u8>) -> PyResult<PyDemAwareResult> {
        let arr = Array1::from_vec(syndrome);
        let (decoding, converged, iterations) = match &mut self.inner {
            InnerDecoder::BpOsd(d) => {
                let r = d.decode(&arr.view()).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                (r.decoding.to_vec(), r.converged, r.iterations)
            }
            InnerDecoder::BpLsd(d) => {
                let r = d.decode(&arr.view()).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                (r.decoding.to_vec(), r.converged, r.iterations)
            }
            InnerDecoder::UnionFind(d) => {
                let r = d.decode(&arr.view(), &[], 0).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                (r.decoding.to_vec(), r.converged, r.iterations)
            }
            InnerDecoder::RelayBp(d) => {
                let r = d.decode(&arr.view()).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                (r.decoding.to_vec(), r.converged, r.iterations)
            }
            InnerDecoder::MinSumBp(d) => {
                let r = d.decode(&arr.view()).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?;
                (r.decoding.to_vec(), r.converged, r.iterations)
            }
        };

        let correction: Vec<u8> = decoding.iter().map(|&v| v & 1).collect();
        // Wide packing: the u64 variant silently wraps observable bits at 64.
        let observables = self
            .dem_check_matrix
            .observables_obsmask_from_correction(&correction);

        Ok(PyDemAwareResult {
            observables,
            converged,
            iterations,
            num_observables: self.dem_check_matrix.num_observables,
        })
    }

    /// Number of detectors in the model.
    #[getter]
    fn num_detectors(&self) -> usize {
        self.dem_check_matrix.num_detectors
    }

    /// Number of observables in the model.
    #[getter]
    fn num_observables(&self) -> usize {
        self.dem_check_matrix.num_observables
    }

    /// Number of error mechanisms in the model.
    #[getter]
    fn num_mechanisms(&self) -> usize {
        self.dem_check_matrix.num_mechanisms
    }

    fn __repr__(&self) -> String {
        let decoder_name = match &self.inner {
            InnerDecoder::BpOsd(_) => "bp_osd",
            InnerDecoder::BpLsd(_) => "bp_lsd",
            InnerDecoder::UnionFind(_) => "union_find",
            InnerDecoder::RelayBp(_) => "relay_bp",
            InnerDecoder::MinSumBp(_) => "min_sum_bp",
        };
        format!(
            "DemAwareDecoder(type={}, detectors={}, mechanisms={}, observables={})",
            decoder_name,
            self.dem_check_matrix.num_detectors,
            self.dem_check_matrix.num_mechanisms,
            self.dem_check_matrix.num_observables,
        )
    }

    fn __getattr__(&self, name: &str) -> PyResult<()> {
        Err(explicit_decode_attribute_error("DemAwareDecoder", name))
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
    decoders_module.add_class::<PyObservableFlips>()?;
    decoders_module.add_class::<PyMwpmResult>()?;
    decoders_module.add_class::<PyBpResult>()?;

    // Matrix types
    decoders_module.add_class::<PyCheckMatrix>()?;
    decoders_module.add_class::<PySparseMatrix>()?;

    // MWPM decoders
    decoders_module.add_class::<PyPyMatchingDecoder>()?;
    decoders_module.add_class::<PyFusionBlossomDecoder>()?;

    // LDPC decoders
    decoders_module.add_class::<PyBpOsdBuilder>()?;
    decoders_module.add_class::<PyBpOsdDecoder>()?;
    decoders_module.add_class::<PyBpLsdBuilder>()?;
    decoders_module.add_class::<PyBpLsdDecoder>()?;
    decoders_module.add_class::<PyUnionFindBuilder>()?;
    decoders_module.add_class::<PyUnionFindDecoder>()?;

    // Search-based decoders
    decoders_module.add_class::<PyTesseractResult>()?;
    decoders_module.add_class::<PyTesseractDecoder>()?;

    // Relay BP decoders
    decoders_module.add_class::<PyRelayBpBuilder>()?;
    decoders_module.add_class::<PyRelayBpDecoder>()?;
    decoders_module.add_class::<PyMinSumBpBuilder>()?;
    decoders_module.add_class::<PyMinSumBpDecoder>()?;

    // DEM-aware decoder wrapper
    decoders_module.add_class::<PyDemAwareResult>()?;
    decoders_module.add_class::<PyDemAwareDecoder>()?;

    // Add submodule to parent
    parent_module.add_submodule(&decoders_module)?;

    // Register in sys.modules for proper import
    let sys = py.import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.decoders", &decoders_module)?;

    Ok(())
}

#[cfg(test)]
mod dem_tuning_tests {
    use super::*;

    const DEM: &str =
        "detector D0\ndetector D1\nlogical_observable L0\nerror(0.1) D0\nerror(0.1) D1 L0\n";

    #[test]
    fn pymatching_error_probability_override_reaches_config() {
        let config = pymatching_config(Some(0.123));

        assert_eq!(config.error_probability, Some(0.123));
    }

    #[test]
    fn pymatching_omitted_override_preserves_default() {
        let config = pymatching_config(None);

        assert_eq!(config.error_probability, None);
    }

    #[test]
    fn fusion_blossom_solver_type_override_reaches_config() {
        let config = fusion_blossom_config(true, Some("legacy")).unwrap();

        assert!(config.correlated);
        assert_eq!(config.solver_type, RustSolverType::Legacy);
    }

    #[test]
    fn fusion_blossom_omitted_override_preserves_default() {
        let config = fusion_blossom_config(false, None).unwrap();

        assert!(!config.correlated);
        assert_eq!(config.solver_type, RustSolverType::Serial);
    }

    #[test]
    fn fusion_blossom_parallel_solver_names_parameter() {
        let error = fusion_blossom_config(false, Some("parallel")).unwrap_err();

        assert!(error.contains("solver_type"));
        assert!(error.contains("partition configuration"));
    }

    #[test]
    fn fusion_blossom_unknown_solver_names_parameter() {
        let error = fusion_blossom_config(false, Some("fast")).unwrap_err();

        assert!(error.contains("solver_type"));
    }

    #[test]
    fn tesseract_default_preset_has_documented_fields() {
        let config = tesseract_config("default", None, None, None, None, None, None).unwrap();

        assert_eq!(config.det_beam, u16::MAX);
        assert!(!config.beam_climbing);
        assert!(config.no_revisit_dets);
        assert!(!config.verbose);
        assert_eq!(config.pqlimit, 200_000);
        assert_eq!(config.det_penalty.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn tesseract_fast_preset_has_documented_fields() {
        let config = tesseract_config("fast", None, None, None, None, None, None).unwrap();

        assert_eq!(config.det_beam, 5);
        assert!(config.beam_climbing);
        assert!(config.no_revisit_dets);
        assert!(!config.verbose);
        assert_eq!(config.pqlimit, 200_000);
        assert_eq!(config.det_penalty.to_bits(), 0.1_f64.to_bits());
    }

    #[test]
    fn tesseract_accurate_preset_has_documented_fields() {
        let config = tesseract_config("accurate", None, None, None, None, None, None).unwrap();

        assert_eq!(config.det_beam, u16::MAX);
        assert!(!config.beam_climbing);
        assert!(!config.no_revisit_dets);
        assert!(!config.verbose);
        assert_eq!(config.pqlimit, 1_000_000);
        assert_eq!(config.det_penalty.to_bits(), 0.0_f64.to_bits());
    }

    #[test]
    fn tesseract_det_beam_override_reaches_config() {
        let config = tesseract_config("default", Some(17), None, None, None, None, None).unwrap();

        assert_eq!(config.det_beam, 17);
    }

    #[test]
    fn tesseract_beam_climbing_override_reaches_config() {
        let config =
            tesseract_config("accurate", None, Some(true), None, None, None, None).unwrap();

        assert!(config.beam_climbing);
    }

    #[test]
    fn tesseract_verbose_override_reaches_config() {
        let config = tesseract_config("default", None, None, Some(true), None, None, None).unwrap();

        assert!(config.verbose);
    }

    #[test]
    fn tesseract_no_revisit_dets_override_reaches_config() {
        let config = tesseract_config("fast", None, None, None, Some(false), None, None).unwrap();

        assert!(!config.no_revisit_dets);
    }

    #[test]
    fn tesseract_pqlimit_override_reaches_config() {
        let config = tesseract_config("fast", None, None, None, None, Some(345_678), None).unwrap();

        assert_eq!(config.pqlimit, 345_678);
    }

    #[test]
    fn tesseract_det_penalty_override_reaches_config() {
        let config = tesseract_config("fast", None, None, None, None, None, Some(0.25)).unwrap();

        assert_eq!(config.det_penalty.to_bits(), 0.25_f64.to_bits());
    }

    #[test]
    fn tesseract_override_wins_over_preset() {
        let config = tesseract_config("fast", Some(19), None, None, None, None, None).unwrap();

        assert_eq!(config.det_beam, 19);
        assert!(config.beam_climbing);
    }

    #[test]
    fn tesseract_omitted_override_preserves_preset() {
        let config = tesseract_config("fast", None, None, None, None, None, None).unwrap();

        assert_eq!(config.det_beam, 5);
        assert!(config.beam_climbing);
        assert!(config.no_revisit_dets);
        assert!(!config.verbose);
        assert_eq!(config.pqlimit, 200_000);
        assert_eq!(config.det_penalty.to_bits(), 0.1_f64.to_bits());
    }

    #[test]
    fn tesseract_unknown_preset_names_parameter() {
        let error = tesseract_config("quick", None, None, None, None, None, None).unwrap_err();

        assert!(error.contains("preset"));
    }

    #[test]
    fn bp_osd_error_rate_override_reaches_config() {
        let config = bp_osd_config(Some(0.123), None, None, None, None, None).unwrap();

        assert_eq!(config.error_rate, Some(0.123));
    }

    #[test]
    fn bp_osd_max_iter_override_reaches_config() {
        let config = bp_osd_config(None, Some(17), None, None, None, None).unwrap();

        assert_eq!(config.max_iter, 17);
    }

    #[test]
    fn bp_osd_bp_schedule_override_reaches_config() {
        let config = bp_osd_config(None, None, Some("serial_relative"), None, None, None).unwrap();

        assert_eq!(config.bp_schedule, RustBpSchedule::SerialRelative);
    }

    #[test]
    fn bp_osd_ms_scaling_factor_override_reaches_config() {
        let config = bp_osd_config(None, None, None, Some(0.625), None, None).unwrap();

        assert_eq!(config.bp_method, RustBpMethod::MinimumSum);
        assert_eq!(config.ms_scaling_factor.to_bits(), 0.625_f64.to_bits());
    }

    #[test]
    fn bp_osd_osd_order_override_reaches_config() {
        let config = bp_osd_config(None, None, None, None, Some(2), None).unwrap();

        assert_eq!(config.osd_method, RustOsdMethod::OsdCs);
        assert_eq!(config.osd_order, 2);
    }

    #[test]
    fn bp_osd_random_schedule_seed_override_reaches_config() {
        let config = bp_osd_config(None, None, None, None, None, Some(42)).unwrap();

        assert_eq!(config.random_schedule_seed, Some(42));
    }

    #[test]
    fn bp_osd_omitted_overrides_preserve_defaults() {
        let config = bp_osd_config(None, None, None, None, None, None).unwrap();

        assert_eq!(config.error_rate, None);
        assert_eq!(config.max_iter, 100);
        assert_eq!(config.bp_method, RustBpMethod::ProductSum);
        assert_eq!(config.bp_schedule, RustBpSchedule::Parallel);
        assert_eq!(config.ms_scaling_factor.to_bits(), 1.0_f64.to_bits());
        assert_eq!(config.osd_method, RustOsdMethod::Osd0);
        assert_eq!(config.osd_order, 0);
        assert_eq!(config.random_schedule_seed, None);
    }

    #[test]
    fn bp_osd_unknown_schedule_names_parameter() {
        let error = bp_osd_config(None, None, Some("random"), None, None, None).unwrap_err();

        assert!(error.contains("bp_schedule"));
    }

    #[test]
    fn bp_lsd_error_rate_override_reaches_config() {
        let config = bp_lsd_config(Some(0.234), None, None, None, None).unwrap();

        assert_eq!(config.error_rate, Some(0.234));
    }

    #[test]
    fn bp_lsd_max_iter_override_reaches_config() {
        let config = bp_lsd_config(None, Some(19), None, None, None).unwrap();

        assert_eq!(config.max_iter, 19);
    }

    #[test]
    fn bp_lsd_bp_schedule_override_reaches_config() {
        let config = bp_lsd_config(None, None, Some("serial_relative"), None, None).unwrap();

        assert_eq!(config.bp_schedule, RustBpSchedule::SerialRelative);
    }

    #[test]
    fn bp_lsd_ms_scaling_factor_override_reaches_config() {
        let config = bp_lsd_config(None, None, None, Some(0.75), None).unwrap();

        assert_eq!(config.bp_method, RustBpMethod::MinimumSum);
        assert_eq!(config.ms_scaling_factor.to_bits(), 0.75_f64.to_bits());
    }

    #[test]
    fn bp_lsd_random_schedule_seed_override_reaches_config() {
        let config = bp_lsd_config(None, None, None, None, Some(24)).unwrap();

        assert_eq!(config.random_schedule_seed, Some(24));
    }

    #[test]
    fn bp_lsd_omitted_overrides_preserve_defaults() {
        let config = bp_lsd_config(None, None, None, None, None).unwrap();

        assert_eq!(config.error_rate, None);
        assert_eq!(config.max_iter, 100);
        assert_eq!(config.bp_method, RustBpMethod::ProductSum);
        assert_eq!(config.bp_schedule, RustBpSchedule::Parallel);
        assert_eq!(config.ms_scaling_factor.to_bits(), 1.0_f64.to_bits());
        assert_eq!(config.random_schedule_seed, None);
    }

    #[test]
    fn bp_lsd_unknown_schedule_names_parameter() {
        let error = bp_lsd_config(None, None, Some("random"), None, None).unwrap_err();

        assert!(error.contains("bp_schedule"));
    }

    #[test]
    fn union_find_method_override_reaches_config() {
        let config = union_find_config(Some("peeling")).unwrap();

        assert_eq!(config.method, RustUfMethod::Peeling);
    }

    #[test]
    fn union_find_omitted_override_preserves_default() {
        let config = union_find_config(None).unwrap();

        assert_eq!(config.method, RustUfMethod::Inversion);
    }

    #[test]
    fn union_find_unknown_method_names_parameter() {
        let error = union_find_config(Some("fast")).unwrap_err();

        assert!(error.contains("method"));
    }

    #[test]
    fn relay_bp_error_rate_override_reaches_config() {
        let config = relay_bp_config(Some(0.345), None, None, None);

        assert_eq!(config.error_rate, Some(0.345));
    }

    #[test]
    fn relay_bp_max_iter_override_reaches_config() {
        let config = relay_bp_config(None, Some(23), None, None);

        assert_eq!(config.max_iter, 23);
    }

    #[test]
    fn relay_bp_alpha_override_reaches_config() {
        let config = relay_bp_config(None, None, Some(0.8), None);

        assert_eq!(config.alpha, Some(0.8));
    }

    #[test]
    fn relay_bp_seed_override_reaches_config() {
        let config = relay_bp_config(None, None, None, Some(91));

        assert_eq!(config.seed, 91);
    }

    #[test]
    fn relay_bp_omitted_overrides_preserve_defaults() {
        let config = relay_bp_config(None, None, None, None);

        assert_eq!(config.error_rate, None);
        assert_eq!(config.max_iter, 100);
        assert_eq!(config.alpha, None);
        assert_eq!(config.seed, 0);
    }

    #[test]
    fn min_sum_bp_error_rate_override_reaches_config() {
        let config = min_sum_bp_config(Some(0.456), None, None);

        assert_eq!(config.error_rate, Some(0.456));
    }

    #[test]
    fn min_sum_bp_max_iter_override_reaches_config() {
        let config = min_sum_bp_config(None, Some(29), None);

        assert_eq!(config.max_iter, 29);
    }

    #[test]
    fn min_sum_bp_alpha_override_reaches_config() {
        let config = min_sum_bp_config(None, None, Some(0.7));

        assert_eq!(config.alpha, Some(0.7));
    }

    #[test]
    fn min_sum_bp_omitted_overrides_preserve_defaults() {
        let config = min_sum_bp_config(None, None, None);

        assert_eq!(config.error_rate, None);
        assert_eq!(config.max_iter, 100);
        assert_eq!(config.alpha, None);
    }

    #[test]
    fn tesseract_overrides_reach_config_and_win_over_preset() {
        let decoder = PyTesseractDecoder::from_dem(
            DEM,
            "fast",
            None,
            None,
            None,
            Some(false),
            Some(12_345),
            Some(0.25),
        )
        .unwrap();

        assert!(!decoder.config.no_revisit_dets);
        assert_eq!(decoder.config.pqlimit, 12_345);
        assert!((decoder.config.det_penalty - 0.25).abs() < f64::EPSILON);
        assert_eq!(decoder.config.det_beam, 5);
    }

    #[test]
    fn bp_osd_overrides_reach_inner_decoder() {
        let decoder = PyBpOsdDecoder::from_dem(
            DEM,
            None,
            Some(17),
            Some("serial"),
            Some(0.75),
            Some(2),
            Some(42),
        )
        .unwrap();
        let InnerDecoder::BpOsd(inner) = decoder.inner else {
            panic!("expected BP+OSD inner decoder");
        };

        assert_eq!(inner.max_iter(), 17);
        assert_eq!(inner.bp_method(), RustBpMethod::MinimumSum);
        assert_eq!(inner.bp_schedule(), RustBpSchedule::Serial);
        assert!((inner.ms_scaling_factor() - 0.75).abs() < f64::EPSILON);
        assert_eq!(inner.osd_order(), 2);
        assert_eq!(inner.osd_method(), RustOsdMethod::OsdCs);
        assert_eq!(inner.random_schedule_seed(), 42);
    }

    #[test]
    fn bp_lsd_overrides_reach_inner_decoder() {
        let decoder = PyBpLsdDecoder::from_dem(
            DEM,
            None,
            Some(19),
            Some("serial_relative"),
            Some(0.625),
            Some(24),
        )
        .unwrap();
        let InnerDecoder::BpLsd(inner) = decoder.inner else {
            panic!("expected BP+LSD inner decoder");
        };

        assert_eq!(inner.max_iter(), 19);
        assert_eq!(inner.bp_method(), RustBpMethod::MinimumSum);
        assert_eq!(inner.bp_schedule(), RustBpSchedule::SerialRelative);
        assert!((inner.ms_scaling_factor() - 0.625).abs() < f64::EPSILON);
        assert_eq!(inner.random_schedule_seed(), 24);
    }

    #[test]
    fn union_find_override_reaches_inner_decoder() {
        let decoder = PyUnionFindDecoder::from_dem(DEM, Some("peeling")).unwrap();
        let InnerDecoder::UnionFind(inner) = decoder.inner else {
            panic!("expected Union-Find inner decoder");
        };

        assert_eq!(inner.method(), RustUfMethod::Peeling);
    }

    #[test]
    fn relay_bp_overrides_reach_inner_decoder() {
        let decoder = PyRelayBpDecoder::from_dem(DEM, None, Some(23), Some(0.8), Some(91)).unwrap();
        let InnerDecoder::RelayBp(inner) = decoder.inner else {
            panic!("expected Relay BP inner decoder");
        };

        assert_eq!(inner.max_iter(), 23);
        assert_eq!(inner.alpha(), Some(0.8));
        assert_eq!(inner.seed(), 91);
    }

    #[test]
    fn min_sum_bp_overrides_reach_inner_decoder() {
        let decoder = PyMinSumBpDecoder::from_dem(DEM, None, Some(29), Some(0.7)).unwrap();
        let InnerDecoder::MinSumBp(inner) = decoder.inner else {
            panic!("expected min-sum BP inner decoder");
        };

        assert_eq!(inner.max_iter(), 29);
        assert_eq!(inner.alpha(), Some(0.7));
    }

    #[test]
    fn bp_family_none_overrides_preserve_dem_defaults() {
        let bp_osd = PyBpOsdDecoder::from_dem(DEM, None, None, None, None, None, None).unwrap();
        let InnerDecoder::BpOsd(bp_osd) = bp_osd.inner else {
            panic!("expected BP+OSD inner decoder");
        };
        assert_eq!(bp_osd.max_iter(), 100);
        assert_eq!(bp_osd.bp_method(), RustBpMethod::ProductSum);
        assert_eq!(bp_osd.bp_schedule(), RustBpSchedule::Parallel);
        assert!((bp_osd.ms_scaling_factor() - 1.0).abs() < f64::EPSILON);
        assert_eq!(bp_osd.osd_order(), 0);
        assert_eq!(bp_osd.random_schedule_seed(), -1);

        let relay = PyRelayBpDecoder::from_dem(DEM, None, None, None, None).unwrap();
        let InnerDecoder::RelayBp(relay) = relay.inner else {
            panic!("expected Relay BP inner decoder");
        };
        assert_eq!(relay.max_iter(), 100);
        assert_eq!(relay.alpha(), None);
        assert_eq!(relay.seed(), 0);
    }

    #[test]
    fn textual_guards_name_the_parameter() {
        pyo3::Python::initialize();

        let preset_error =
            PyTesseractDecoder::from_dem(DEM, "quick", None, None, None, None, None, None)
                .err()
                .unwrap();
        assert!(preset_error.to_string().contains("preset"));

        let schedule_error = PyBpLsdDecoder::from_dem(DEM, None, None, Some("random"), None, None)
            .err()
            .unwrap();
        assert!(schedule_error.to_string().contains("bp_schedule"));

        let method_error = PyUnionFindDecoder::from_dem(DEM, Some("fast"))
            .err()
            .unwrap();
        assert!(method_error.to_string().contains("method"));

        let solver_error = PyFusionBlossomDecoder::from_dem(DEM, false, Some("parallel"))
            .err()
            .unwrap();
        let message = solver_error.to_string();
        assert!(message.contains("solver_type"));
        assert!(message.contains("partition configuration"));

        let solver_error = PyFusionBlossomDecoder::from_dem(DEM, false, Some("fast"))
            .err()
            .unwrap();
        assert!(solver_error.to_string().contains("solver_type"));

        for (error, parameter) in [
            (
                optional_usize(Some(-1), "max_iter").unwrap_err(),
                "max_iter",
            ),
            (
                optional_u16(Some(65_536), "det_beam").unwrap_err(),
                "det_beam",
            ),
            (
                optional_i32(Some(i64::from(i32::MAX) + 1), "random_schedule_seed").unwrap_err(),
                "random_schedule_seed",
            ),
            (optional_u64(Some(-1), "seed").unwrap_err(), "seed"),
        ] {
            assert!(error.to_string().contains(parameter));
        }
    }
}
