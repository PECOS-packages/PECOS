// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Python bindings for PECOS fault tolerance analysis.
//!
//! This module provides Python bindings for the fault tolerance infrastructure,
//! enabling PECOS-native DEM (Detector Error Model) generation from quantum circuits.
//!
//! # Main Types
//!
//! - `DagFaultAnalyzer` - Builds fault influence maps from DAG circuits
//! - `DagFaultInfluenceMap` - CSR-optimized influence map (equivalent to DEM)
//! - `FaultLocation` - Represents a fault location in spacetime
//!
//! # Example
//!
//! ```python
//! from pecos_rslib import DagCircuit
//! from pecos_rslib.qec import DagFaultAnalyzer
//!
//! # Build a syndrome extraction circuit
//! dag = DagCircuit()
//! dag.pz(2)      # Prep ancilla
//! dag.cx(0, 2)   # CNOT data -> ancilla
//! dag.cx(1, 2)   # CNOT data -> ancilla
//! dag.mz(2)      # Measure ancilla
//!
//! # Build fault influence map
//! analyzer = DagFaultAnalyzer(dag)
//! influence_map = analyzer.build_influence_map()
//!
//! # Query fault influence (O(1) lookup)
//! has_syndrome, causes_logical = influence_map.classify_fault(0, 1)  # loc 0, X fault
//! ```

use pecos::qec::fault_tolerance::dem_builder::{
    ComparisonMethod as RustComparisonMethod, DemBuilder as RustDemBuilder,
    DetectorErrorModel as RustDetectorErrorModel, EquivalenceResult as RustEquivalenceResult,
    MeasurementNoiseModel as RustMeasurementNoiseModel, MemBuilder as RustMemBuilder,
    ParsedDem as RustParsedDem, compare_dems_exact as rust_compare_dems_exact,
    compare_dems_statistical as rust_compare_dems_statistical,
    verify_dem_equivalence as rust_verify_dem_equivalence,
};
use pecos::qec::fault_tolerance::influence_builder::InfluenceBuilder as RustInfluenceBuilder;
use pecos::qec::fault_tolerance::propagator::{
    DagFaultAnalyzer as RustDagFaultAnalyzer, DagFaultInfluenceMap as RustDagFaultInfluenceMap,
    DagSpacetimeLocation, Pauli,
};
use pecos::quantum::DagCircuit;
use pyo3::Py;
use pyo3::prelude::*;

/// Type alias for batch sampling results: (`detection_events_per_shot`, `observable_flips_per_shot`)
type BatchSampleResult = (Vec<Vec<bool>>, Vec<Vec<bool>>);

// =============================================================================
// Fault Location Types
// =============================================================================

/// A spacetime location for a fault in a DAG circuit.
///
/// Identifies where a fault can occur: the DAG node, qubits involved,
/// whether it's before or after the gate, and the gate type.
///
/// # Attributes
///
/// * `node` - DAG node index
/// * `qubits` - List of qubit indices involved
/// * `before` - Whether fault occurs before (True) or after (False) the gate
/// * `gate_type` - Name of the gate type
#[pyclass(
    name = "FaultLocation",
    module = "pecos_rslib.qec",
    skip_from_py_object
)]
#[derive(Clone)]
pub struct PyFaultLocation {
    node: usize,
    qubits: Vec<usize>,
    before: bool,
    gate_type: String,
}

#[pymethods]
impl PyFaultLocation {
    /// DAG node index.
    #[getter]
    fn node(&self) -> usize {
        self.node
    }

    /// Qubit indices involved in this fault location.
    #[getter]
    fn qubits(&self) -> Vec<usize> {
        self.qubits.clone()
    }

    /// Whether the fault occurs before the gate (True) or after (False).
    #[getter]
    fn before(&self) -> bool {
        self.before
    }

    /// Gate type name.
    #[getter]
    fn gate_type(&self) -> String {
        self.gate_type.clone()
    }

    fn __repr__(&self) -> String {
        let timing = if self.before { "before" } else { "after" };
        format!(
            "FaultLocation(node={}, qubits={:?}, {}, gate={})",
            self.node, self.qubits, timing, self.gate_type
        )
    }
}

impl From<&DagSpacetimeLocation> for PyFaultLocation {
    fn from(loc: &DagSpacetimeLocation) -> Self {
        Self {
            node: loc.node,
            qubits: loc.qubits.iter().map(pecos::QubitId::index).collect(),
            before: loc.before,
            gate_type: format!("{:?}", loc.gate_type),
        }
    }
}

// =============================================================================
// Fault Influence Map
// =============================================================================

/// A fault influence map built from a DAG circuit.
///
/// Maps fault locations to their effects on detectors and logical observables.
/// Uses CSR (Compressed Sparse Row) layout for cache-efficient storage.
///
/// This is functionally equivalent to a Detector Error Model (DEM) but stored
/// in a format optimized for fast querying during sampling.
///
/// # Example
///
/// ```python
/// # Build influence map from analyzer
/// influence_map = analyzer.build_influence_map()
///
/// # Query fault influence
/// has_syndrome, causes_logical = influence_map.classify_fault(loc_idx=0, pauli=1)
///
/// # Get detector indices flipped by this fault
/// detector_indices = influence_map.get_detector_indices(loc_idx=0, pauli=1)
/// ```
#[pyclass(name = "DagFaultInfluenceMap", module = "pecos_rslib.qec")]
pub struct PyDagFaultInfluenceMap {
    inner: RustDagFaultInfluenceMap,
}

#[pymethods]
impl PyDagFaultInfluenceMap {
    /// Number of fault locations in the map.
    #[getter]
    fn num_locations(&self) -> usize {
        self.inner.locations.len()
    }

    /// Number of detectors (measurement-based).
    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.detectors.len()
    }

    /// Number of logical observables tracked.
    #[getter]
    fn num_logicals(&self) -> usize {
        self.inner
            .influences
            .max_logical_index()
            .map_or(0, |i| i + 1)
    }

    /// Get all fault locations.
    ///
    /// Returns:
    ///     List of `FaultLocation` objects.
    fn get_locations(&self) -> Vec<PyFaultLocation> {
        self.inner
            .locations
            .iter()
            .map(PyFaultLocation::from)
            .collect()
    }

    /// Get a specific fault location by index.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///
    /// Returns:
    ///     `FaultLocation` object or None if index is out of range.
    fn get_location(&self, loc_idx: usize) -> Option<PyFaultLocation> {
        self.inner.get_location(loc_idx).map(PyFaultLocation::from)
    }

    /// Classify a fault at the given location.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///     pauli: Pauli type (1=X, 2=Y, 3=Z).
    ///
    /// Returns:
    ///     Tuple (`has_syndrome`, `causes_logical_error`).
    ///     - `has_syndrome`: True if the fault flips at least one detector.
    ///     - `causes_logical_error`: True if the fault flips the logical observable.
    fn classify_fault(&self, loc_idx: usize, pauli: u8) -> (bool, bool) {
        self.inner.classify_fault(loc_idx, pauli)
    }

    /// Get detector indices flipped by a fault.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///     pauli: Pauli type (1=X, 2=Y, 3=Z).
    ///
    /// Returns:
    ///     List of detector indices that are flipped by this fault.
    fn get_detector_indices(&self, loc_idx: usize, pauli: u8) -> Vec<u32> {
        self.inner.get_detector_indices(loc_idx, pauli).to_vec()
    }

    /// Get logical indices flipped by a fault.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///     pauli: Pauli type (1=X, 2=Y, 3=Z).
    ///
    /// Returns:
    ///     List of logical indices that are flipped by this fault.
    fn get_logical_indices(&self, loc_idx: usize, pauli: u8) -> Vec<u32> {
        self.inner.get_logical_indices(loc_idx, pauli).to_vec()
    }

    /// Check if a fault at the given location flips any detector.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///     pauli: Pauli type (1=X, 2=Y, 3=Z).
    ///
    /// Returns:
    ///     True if the fault flips at least one detector.
    fn has_detector_flips(&self, loc_idx: usize, pauli: u8) -> bool {
        self.inner
            .influences
            .has_detector_flips(loc_idx, Pauli::from_u8(pauli))
    }

    /// Check if a fault at the given location flips a logical observable.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///     pauli: Pauli type (1=X, 2=Y, 3=Z).
    ///
    /// Returns:
    ///     True if the fault flips the logical observable.
    fn has_logical_flips(&self, loc_idx: usize, pauli: u8) -> bool {
        self.inner
            .influences
            .has_logical_flips(loc_idx, Pauli::from_u8(pauli))
    }

    /// Get memory statistics for this influence map.
    ///
    /// Returns:
    ///     Dictionary with memory usage statistics.
    fn memory_stats(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let stats = self.inner.memory_stats();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("num_locations", stats.num_locations)?;
        dict.set_item("total_detector_entries", stats.total_detector_entries)?;
        dict.set_item("total_logical_entries", stats.total_logical_entries)?;
        dict.set_item("offset_bytes", stats.offset_bytes)?;
        dict.set_item("data_bytes", stats.data_bytes)?;
        dict.set_item("total_bytes", stats.total_bytes)?;
        Ok(dict.unbind())
    }

    /// Export CSR data for external use (e.g., GPU sampling).
    ///
    /// Returns:
    ///     Dictionary containing all CSR arrays:
    ///     - `num_locations`, `num_detectors`, `num_logicals`
    ///     - `detector_offsets_x`, `detector_data_x`
    ///     - `detector_offsets_y`, `detector_data_y`
    ///     - `detector_offsets_z`, `detector_data_z`
    ///     - `logical_offsets_x`, `logical_data_x`
    ///     - `logical_offsets_y`, `logical_data_y`
    ///     - `logical_offsets_z`, `logical_data_z`
    fn export_csr(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let (
            num_locations,
            num_detectors,
            num_logicals,
            det_off_x,
            det_data_x,
            det_off_y,
            det_data_y,
            det_off_z,
            det_data_z,
            log_off_x,
            log_data_x,
            log_off_y,
            log_data_y,
            log_off_z,
            log_data_z,
        ) = self.inner.export_csr();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("num_locations", num_locations)?;
        dict.set_item("num_detectors", num_detectors)?;
        dict.set_item("num_logicals", num_logicals)?;
        dict.set_item("detector_offsets_x", det_off_x)?;
        dict.set_item("detector_data_x", det_data_x)?;
        dict.set_item("detector_offsets_y", det_off_y)?;
        dict.set_item("detector_data_y", det_data_y)?;
        dict.set_item("detector_offsets_z", det_off_z)?;
        dict.set_item("detector_data_z", det_data_z)?;
        dict.set_item("logical_offsets_x", log_off_x)?;
        dict.set_item("logical_data_x", log_data_x)?;
        dict.set_item("logical_offsets_y", log_off_y)?;
        dict.set_item("logical_data_y", log_data_y)?;
        dict.set_item("logical_offsets_z", log_off_z)?;
        dict.set_item("logical_data_z", log_data_z)?;
        Ok(dict.unbind())
    }

    /// Get the measurements in order (node, qubit, basis).
    ///
    /// Returns:
    ///     List of (`node_id`, qubit, basis) tuples representing measurements
    ///     in the order used by the influence map.
    fn measurements(&self) -> Vec<(usize, usize, u8)> {
        self.inner
            .measurements
            .iter()
            .map(|&(node, qubit, basis)| (node, qubit, basis))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "DagFaultInfluenceMap(locations={}, detectors={}, logicals={})",
            self.num_locations(),
            self.num_detectors(),
            self.num_logicals()
        )
    }

    fn __len__(&self) -> usize {
        self.inner.locations.len()
    }
}

// =============================================================================
// DAG Fault Analyzer
// =============================================================================

/// Analyzes fault tolerance properties of a DAG circuit.
///
/// Builds fault influence maps by backward propagation from measurements.
/// Uses sparse traversal that only visits gates touching qubits with
/// non-trivial Paulis, providing 5-50x speedup over tick-based analysis.
///
/// # Performance
///
/// | Circuit Size | Tick-based | DAG-based | Speedup |
/// |--------------|------------|-----------|---------|
/// | d=3 (17 qubits) | 64 us | 16 us | 4x |
/// | d=5 (49 qubits) | 205 us | 38 us | 5x |
/// | d=7 (97 qubits) | 569 us | 49 us | 11x |
/// | d=11 (241 qubits) | 6529 us | 125 us | 52x |
///
/// # Example
///
/// ```python
/// from pecos_rslib import DagCircuit
/// from pecos_rslib.qec import DagFaultAnalyzer
///
/// dag = DagCircuit()
/// dag.pz(2)
/// dag.cx(0, 2)
/// dag.cx(1, 2)
/// dag.mz(2)
///
/// analyzer = DagFaultAnalyzer(dag)
/// influence_map = analyzer.build_influence_map()
/// ```
#[pyclass(name = "DagFaultAnalyzer", module = "pecos_rslib.qec")]
pub struct PyDagFaultAnalyzer {
    // We need to own the DagCircuit since RustDagFaultAnalyzer borrows it
    dag: DagCircuit,
}

#[pymethods]
impl PyDagFaultAnalyzer {
    /// Create a new DAG fault analyzer.
    ///
    /// Args:
    ///     dag: A `DagCircuit` to analyze.
    #[new]
    fn new(dag: &crate::dag_circuit_bindings::PyDagCircuit) -> Self {
        Self {
            dag: dag.inner.clone(),
        }
    }

    /// Build the complete fault influence map.
    ///
    /// Performs backward propagation from all measurements and creates a
    /// lookup table for fault classification.
    ///
    /// Returns:
    ///     `DagFaultInfluenceMap` with O(1) fault classification.
    fn build_influence_map(&self) -> PyDagFaultInfluenceMap {
        let analyzer = RustDagFaultAnalyzer::new(&self.dag);
        let inner = analyzer.build_influence_map();
        PyDagFaultInfluenceMap { inner }
    }

    /// Maximum node index in the DAG.
    #[getter]
    fn max_node(&self) -> usize {
        let analyzer = RustDagFaultAnalyzer::new(&self.dag);
        analyzer.max_node()
    }

    /// Maximum qubit index in the DAG.
    #[getter]
    fn max_qubit(&self) -> usize {
        let analyzer = RustDagFaultAnalyzer::new(&self.dag);
        analyzer.max_qubit()
    }

    fn __repr__(&self) -> String {
        let analyzer = RustDagFaultAnalyzer::new(&self.dag);
        format!(
            "DagFaultAnalyzer(max_node={}, max_qubit={})",
            analyzer.max_node(),
            analyzer.max_qubit()
        )
    }
}

// =============================================================================
// Influence Builder
// =============================================================================

/// Builder for fault influence maps with proper detector definitions.
///
/// This integrates forward symbolic simulation with backward propagation
/// to create complete influence maps (DEM equivalents) suitable for noisy sampling.
///
/// Unlike `DagFaultAnalyzer` which treats each measurement as a detector,
/// `InfluenceBuilder` uses symbolic simulation to identify which measurements
/// are deterministic (and thus define proper detectors).
///
/// # Example
///
/// ```python
/// from pecos_rslib import DagCircuit
/// from pecos_rslib.qec import InfluenceBuilder
///
/// dag = DagCircuit()
/// # ... build circuit ...
///
/// # Build influence map with logical operator tracking
/// builder = InfluenceBuilder(dag)
/// builder.with_logical_z([0, 1, 2])  # Top row qubits for d=3 surface code
/// influence_map = builder.build()
/// ```
#[pyclass(name = "InfluenceBuilder", module = "pecos_rslib.qec")]
pub struct PyInfluenceBuilder {
    dag: DagCircuit,
    logical_x_qubits: Vec<usize>,
    logical_z_qubits: Vec<usize>,
}

#[pymethods]
impl PyInfluenceBuilder {
    /// Create a new influence builder for the given circuit.
    ///
    /// Args:
    ///     dag: A `DagCircuit` to analyze.
    #[new]
    fn new(dag: &crate::dag_circuit_bindings::PyDagCircuit) -> Self {
        Self {
            dag: dag.inner.clone(),
            logical_x_qubits: Vec::new(),
            logical_z_qubits: Vec::new(),
        }
    }

    /// Add a logical X operator to track.
    ///
    /// The logical X is defined as X on all specified qubits.
    /// This logical is sensitive to Z errors.
    ///
    /// Args:
    ///     qubits: List of qubit indices for the logical X operator.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_logical_x(mut slf: PyRefMut<'_, Self>, qubits: Vec<usize>) -> PyRefMut<'_, Self> {
        slf.logical_x_qubits = qubits;
        slf
    }

    /// Add a logical Z operator to track.
    ///
    /// The logical Z is defined as Z on all specified qubits.
    /// This logical is sensitive to X errors.
    ///
    /// Args:
    ///     qubits: List of qubit indices for the logical Z operator.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_logical_z(mut slf: PyRefMut<'_, Self>, qubits: Vec<usize>) -> PyRefMut<'_, Self> {
        slf.logical_z_qubits = qubits;
        slf
    }

    /// Build the fault influence map.
    ///
    /// This performs:
    /// 1. Forward symbolic simulation to identify deterministic measurements
    /// 2. Detector extraction from deterministic measurement correlations
    /// 3. Backward propagation to build the influence map
    ///
    /// Returns:
    ///     `DagFaultInfluenceMap` with proper detector definitions and logical tracking.
    fn build(&self) -> PyDagFaultInfluenceMap {
        let builder = RustInfluenceBuilder::new(&self.dag)
            .with_logical_x(self.logical_x_qubits.clone())
            .with_logical_z(self.logical_z_qubits.clone());

        let inner = builder.build();
        PyDagFaultInfluenceMap { inner }
    }

    fn __repr__(&self) -> String {
        format!(
            "InfluenceBuilder(logical_x={:?}, logical_z={:?})",
            self.logical_x_qubits, self.logical_z_qubits
        )
    }
}

// =============================================================================
// Detector Error Model
// =============================================================================

/// A Detector Error Model (DEM) in Stim-compatible format.
///
/// This represents the error model of a quantum circuit, mapping error
/// mechanisms to their probabilities. It can be converted to Stim format
/// for use with Stim-based decoders.
///
/// # Example
///
/// ```python
/// from pecos_rslib.qec import DemBuilder
///
/// # Build DEM from influence map
/// builder = DemBuilder(influence_map)
/// builder.with_noise(0.01, 0.01, 0.01, 0.01)
/// builder.with_detectors_json(detectors_json)
/// dem = builder.build()
///
/// # Output in DEM format
/// print(dem.to_string())
/// ```
#[pyclass(name = "DetectorErrorModel", module = "pecos_rslib.qec")]
pub struct PyDetectorErrorModel {
    inner: RustDetectorErrorModel,
}

#[pymethods]
impl PyDetectorErrorModel {
    /// Number of detectors in the model.
    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }

    /// Number of logical observables in the model.
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Convert the DEM to a string in standard DEM format.
    ///
    /// Each error mechanism is output with its total probability, with no
    /// splitting into decomposed forms. This matches Stim's
    /// `detector_error_model(decompose_errors=False)` output.
    ///
    /// Returns:
    ///     A string in DEM format with one entry per mechanism.
    #[allow(clippy::inherent_to_string)] // PyO3 binding - two string formats
    fn to_string(&self) -> String {
        self.inner.to_string()
    }

    /// Convert the DEM to a string with decomposed representations.
    ///
    /// For 2-detector mechanisms, outputs multiple equivalent representations
    /// including L0 cancellation forms where available. Hyperedge errors
    /// (affecting 3+ detectors) are decomposed into graphlike components.
    ///
    /// This matches Stim's `detector_error_model(decompose_errors=True)` output.
    ///
    /// Returns:
    ///     A string in DEM format with decomposed representations.
    fn to_string_decomposed(&self) -> String {
        self.inner.to_string_decomposed()
    }

    /// Number of tracked error contributions.
    #[getter]
    fn num_contributions(&self) -> usize {
        self.inner.num_contributions()
    }

    /// Returns debug info about contributions for a specific mechanism.
    ///
    /// Args:
    ///     detectors: List of detector IDs that define the mechanism.
    ///
    /// Returns:
    ///     Debug string showing source types and probabilities for matching contributions.
    fn contributions_for_mechanism(&self, detectors: Vec<u32>) -> String {
        self.inner.contributions_for_mechanism(&detectors)
    }

    /// Returns debug info about all unique contribution effects.
    ///
    /// Shows each unique detector/logical pattern and how many contributions
    /// target it with their total probability.
    fn all_contribution_effects(&self) -> String {
        self.inner.all_contribution_effects()
    }

    fn __repr__(&self) -> String {
        format!(
            "DetectorErrorModel(detectors={}, observables={}, contributions={})",
            self.num_detectors(),
            self.num_observables(),
            self.num_contributions()
        )
    }

    fn __str__(&self) -> String {
        self.to_string()
    }
}

// =============================================================================
// DEM Builder
// =============================================================================

/// Builder for Detector Error Models (DEMs).
///
/// Constructs a DEM from a fault influence map and detector/observable metadata.
/// Uses the per-qubit fault model for accurate depolarizing noise analysis.
///
/// # Example
///
/// ```python
/// from pecos_rslib.qec import DagFaultAnalyzer, DemBuilder
///
/// # Build influence map
/// analyzer = DagFaultAnalyzer(dag)
/// influence_map = analyzer.build_influence_map()
///
/// # Build DEM
/// builder = DemBuilder(influence_map)
/// builder.with_noise(0.01, 0.01, 0.01, 0.01)
/// builder.with_detectors_json('[{"id": 0, "coords": [0, 0, 0], "records": [-1]}]')
/// dem = builder.build()
///
/// print(dem.to_string())
/// ```
#[pyclass(name = "DemBuilder", module = "pecos_rslib.qec")]
pub struct PyDemBuilder {
    influence_map: RustDagFaultInfluenceMap,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_init: f64,
    detectors_json: Option<String>,
    observables_json: Option<String>,
    num_measurements: Option<usize>,
    /// Measurement order: list of qubits in `TickCircuit` measurement execution order.
    /// This allows proper mapping between record offsets and influence map indices.
    measurement_order: Option<Vec<usize>>,
}

#[pymethods]
impl PyDemBuilder {
    /// Create a new DEM builder from a fault influence map.
    ///
    /// Args:
    ///     `influence_map`: A `DagFaultInfluenceMap` from `DagFaultAnalyzer`.
    #[new]
    fn new(influence_map: &PyDagFaultInfluenceMap) -> Self {
        Self {
            influence_map: influence_map.inner.clone(),
            p1: 0.01,
            p2: 0.01,
            p_meas: 0.01,
            p_init: 0.01,
            detectors_json: None,
            observables_json: None,
            num_measurements: None,
            measurement_order: None,
        }
    }

    /// Set the noise parameters.
    ///
    /// Args:
    ///     p1: Single-qubit depolarizing error rate.
    ///     p2: Two-qubit depolarizing error rate.
    ///     `p_meas`: Measurement error rate.
    ///     `p_init`: Initialization (prep) error rate.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_noise(
        mut slf: PyRefMut<'_, Self>,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_init: f64,
    ) -> PyRefMut<'_, Self> {
        slf.p1 = p1;
        slf.p2 = p2;
        slf.p_meas = p_meas;
        slf.p_init = p_init;
        slf
    }

    /// Set the detector definitions from JSON.
    ///
    /// Args:
    ///     json: JSON string with detector definitions.
    ///           Format: [{"id": 0, "coords": [x, y, t], "records": [-1, -5]}, ...]
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_detectors_json(mut slf: PyRefMut<'_, Self>, json: String) -> PyRefMut<'_, Self> {
        slf.detectors_json = Some(json);
        slf
    }

    /// Set the observable definitions from JSON.
    ///
    /// Args:
    ///     json: JSON string with observable definitions.
    ///           Format: [{"id": 0, "records": [-1, -3, -5]}, ...]
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_observables_json(mut slf: PyRefMut<'_, Self>, json: String) -> PyRefMut<'_, Self> {
        slf.observables_json = Some(json);
        slf
    }

    /// Set the number of measurements (for record offset calculation).
    ///
    /// Args:
    ///     num: Total number of measurements in the circuit.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_num_measurements(mut slf: PyRefMut<'_, Self>, num: usize) -> PyRefMut<'_, Self> {
        slf.num_measurements = Some(num);
        slf
    }

    /// Set the measurement order from the original circuit.
    ///
    /// The measurement order is a list of qubits in the order they were measured
    /// in the original circuit (e.g., `TickCircuit`). This allows proper mapping
    /// between record offsets (which use `TickCircuit` order) and influence map
    /// indices (which may use a different order based on DAG topology).
    ///
    /// Args:
    ///     order: List of qubit indices in measurement execution order.
    ///            order[i] is the qubit measured at `TickCircuit` measurement index i.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_measurement_order(
        mut slf: PyRefMut<'_, Self>,
        order: Vec<usize>,
    ) -> PyRefMut<'_, Self> {
        slf.measurement_order = Some(order);
        slf
    }

    /// Build the Detector Error Model.
    ///
    /// Returns:
    ///     A `DetectorErrorModel` that can be converted to string format.
    ///
    /// Raises:
    ///     `ValueError`: If the detector or observable JSON is malformed.
    fn build(&self) -> PyResult<PyDetectorErrorModel> {
        let mut builder = RustDemBuilder::new(&self.influence_map).with_noise(
            self.p1,
            self.p2,
            self.p_meas,
            self.p_init,
        );

        if let Some(num) = self.num_measurements {
            builder = builder.with_num_measurements(num);
        }

        if let Some(ref order) = self.measurement_order {
            builder = builder.with_measurement_order(order.clone());
        }

        if let Some(ref json) = self.detectors_json {
            builder = builder
                .with_detectors_json(json)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }

        if let Some(ref json) = self.observables_json {
            builder = builder
                .with_observables_json(json)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        }

        let inner = builder.build();
        Ok(PyDetectorErrorModel { inner })
    }

    /// Alias for `build()` - provided for backward compatibility.
    fn build_with_source_tracking(&self) -> PyResult<PyDetectorErrorModel> {
        self.build()
    }

    fn __repr__(&self) -> String {
        format!(
            "DemBuilder(p1={}, p2={}, p_meas={}, p_init={})",
            self.p1, self.p2, self.p_meas, self.p_init
        )
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// Parse detector records from JSON string.
///
/// Extracts the "records" arrays from detector definitions.
fn parse_detector_records(detectors_json: &str) -> PyResult<Vec<Vec<i32>>> {
    if detectors_json.is_empty() {
        return Ok(Vec::new());
    }

    let detectors: Vec<serde_json::Value> = serde_json::from_str(detectors_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {e}")))?;

    let mut detector_records = Vec::with_capacity(detectors.len());

    for det in &detectors {
        let records = det
            .get("records")
            .and_then(|r| r.as_array())
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Detector missing 'records' array")
            })?;

        let offsets: Vec<i32> = records
            .iter()
            .map(|r| {
                r.as_i64().map(|v| v as i32).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Record offset must be integer")
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        detector_records.push(offsets);
    }

    Ok(detector_records)
}

/// Parse observable records from JSON string.
///
/// Extracts the "records" arrays from observable definitions.
fn parse_observable_records(observables_json: &str) -> PyResult<Vec<Vec<i32>>> {
    if observables_json.is_empty() {
        return Ok(Vec::new());
    }

    let observables: Vec<serde_json::Value> = serde_json::from_str(observables_json)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("Invalid JSON: {e}")))?;

    let mut observable_records = Vec::with_capacity(observables.len());

    for obs in &observables {
        let records = obs
            .get("records")
            .and_then(|r| r.as_array())
            .ok_or_else(|| {
                pyo3::exceptions::PyValueError::new_err("Observable missing 'records' array")
            })?;

        let offsets: Vec<i32> = records
            .iter()
            .map(|r| {
                r.as_i64().map(|v| v as i32).ok_or_else(|| {
                    pyo3::exceptions::PyValueError::new_err("Record offset must be integer")
                })
            })
            .collect::<PyResult<Vec<_>>>()?;

        observable_records.push(offsets);
    }

    Ok(observable_records)
}

// =============================================================================
// Measurement Noise Model
// =============================================================================

/// A Measurement Noise Model (MNM) for fast approximate sampling.
///
/// Unlike a DEM which maps error mechanisms to detector effects, the MNM maps
/// directly to measurement effects. This allows sampling raw measurement outcomes
/// without needing detector definitions.
///
/// # Sampling Modes
///
/// - **Per-fault-location** (accurate): Sample each (location, Pauli) independently
/// - **Per-mechanism** (fast, approximate): Sample each unique measurement effect once
///
/// The MNM enables the fast per-mechanism mode while still producing raw measurement
/// outcomes that can be converted to detection events using any detector definition.
///
/// # Example
///
/// ```python
/// from pecos_rslib.qec import MemBuilder
///
/// # Build MNM from influence map
/// mnm = MemBuilder(influence_map).with_noise(0.01, 0.01, 0.01, 0.01).build()
///
/// # Sample measurement outcomes
/// outcomes = mnm.sample()
/// ```
#[pyclass(name = "MeasurementNoiseModel", module = "pecos_rslib.qec")]
pub struct PyMeasurementNoiseModel {
    inner: RustMeasurementNoiseModel,
}

#[pymethods]
impl PyMeasurementNoiseModel {
    /// Number of distinct mechanisms in the model.
    #[getter]
    fn num_mechanisms(&self) -> usize {
        self.inner.num_mechanisms()
    }

    /// Number of measurements in the circuit.
    #[getter]
    fn num_measurements(&self) -> usize {
        self.inner.num_measurements
    }

    /// Sample measurement outcomes.
    ///
    /// Each noise mechanism is sampled once according to its probability.
    /// When a mechanism fires, its measurements are XOR'd into the outcomes.
    ///
    /// Args:
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     List of boolean measurement outcomes.
    #[pyo3(signature = (seed=None))]
    fn sample(&self, seed: Option<u64>) -> Vec<bool> {
        use pecos_rng::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        self.inner.sample(&mut rng)
    }

    /// Sample multiple shots of measurement outcomes.
    ///
    /// Args:
    ///     `num_shots`: Number of shots to sample.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     List of lists, where each inner list contains boolean measurement outcomes.
    #[pyo3(signature = (num_shots, seed=None))]
    fn sample_batch(&self, num_shots: usize, seed: Option<u64>) -> Vec<Vec<bool>> {
        use pecos_rng::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        (0..num_shots)
            .map(|_| self.inner.sample(&mut rng))
            .collect()
    }

    /// Get all mechanisms and their probabilities.
    ///
    /// Returns:
    ///     List of (measurements, probability) tuples.
    fn get_mechanisms(&self) -> Vec<(Vec<u32>, f64)> {
        self.inner
            .iter()
            .map(|(m, &p)| (m.measurements.to_vec(), p))
            .collect()
    }

    /// Convert measurement outcomes to detection events.
    ///
    /// Given raw measurement outcomes and detector definitions, computes which
    /// detectors fire by XOR'ing the specified measurement records for each detector.
    ///
    /// If measurement order was provided when building the MNM, outcomes are first
    /// reordered from influence map order to `TickCircuit` order before applying
    /// detector records.
    ///
    /// Args:
    ///     outcomes: List of boolean measurement outcomes (from `sample()`).
    ///     `detectors_json`: JSON string with detector definitions.
    ///         Format: [{"id": 0, "records": [-1, -5]}, ...]
    ///         Records are negative offsets from end of measurement list.
    ///
    /// Returns:
    ///     List of boolean detection events (True = detector fired).
    ///
    /// Example:
    ///     >>> outcomes = `mnm.sample()`
    ///     >>> `detection_events` = `mnm.to_detection_events(outcomes`, `detectors_json`)
    fn to_detection_events(
        &self,
        outcomes: Vec<bool>,
        detectors_json: &str,
    ) -> PyResult<Vec<bool>> {
        let detector_records = parse_detector_records(detectors_json)?;
        // Use instance method which applies im_to_tc reordering if set
        Ok(self
            .inner
            .compute_detection_events(&outcomes, &detector_records))
    }

    /// Sample and convert to detection events in one step.
    ///
    /// This is a convenience method that combines `sample()` and `to_detection_events()`.
    ///
    /// Args:
    ///     `detectors_json`: JSON string with detector definitions.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Tuple of (`measurement_outcomes`, `detection_events`).
    #[pyo3(signature = (detectors_json, seed=None))]
    fn sample_with_detectors(
        &self,
        detectors_json: &str,
        seed: Option<u64>,
    ) -> PyResult<(Vec<bool>, Vec<bool>)> {
        use pecos_rng::PecosRng;
        use rand::RngExt;

        let detector_records = parse_detector_records(detectors_json)?;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        let (outcomes, detection_events) = self
            .inner
            .sample_with_detectors(&detector_records, &mut rng);
        Ok((outcomes, detection_events))
    }

    /// Sample multiple shots and convert to detection events.
    ///
    /// Args:
    ///     `num_shots`: Number of shots to sample.
    ///     `detectors_json`: JSON string with detector definitions.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     List of (`measurement_outcomes`, `detection_events`) tuples.
    #[pyo3(signature = (num_shots, detectors_json, seed=None))]
    fn sample_batch_with_detectors(
        &self,
        num_shots: usize,
        detectors_json: &str,
        seed: Option<u64>,
    ) -> PyResult<Vec<(Vec<bool>, Vec<bool>)>> {
        use pecos_rng::PecosRng;
        use rand::RngExt;

        let detector_records = parse_detector_records(detectors_json)?;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        let results: Vec<_> = (0..num_shots)
            .map(|_| {
                self.inner
                    .sample_with_detectors(&detector_records, &mut rng)
            })
            .collect();

        Ok(results)
    }

    /// Sample for threshold estimation with both detection events and observable flips.
    ///
    /// This matches Stim's DEM sampler output format, returning the information
    /// needed for decoding and logical error rate computation.
    ///
    /// Args:
    ///     `detectors_json`: JSON string with detector definitions.
    ///     `observables_json`: JSON string with observable definitions.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Tuple of (`detection_events`, `observable_flips`).
    #[pyo3(signature = (detectors_json, observables_json, seed=None))]
    fn sample_for_decoding(
        &self,
        detectors_json: &str,
        observables_json: &str,
        seed: Option<u64>,
    ) -> PyResult<(Vec<bool>, Vec<bool>)> {
        use pecos_rng::PecosRng;
        use rand::RngExt;

        let detector_records = parse_detector_records(detectors_json)?;
        let observable_records = parse_observable_records(observables_json)?;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        let (detection_events, observable_flips) =
            self.inner
                .sample_for_decoding(&detector_records, &observable_records, &mut rng);
        Ok((detection_events, observable_flips))
    }

    /// Batch sample for threshold estimation.
    ///
    /// Efficiently samples multiple shots, returning detection events and observable
    /// flips for each shot. This is the PECOS native alternative to Stim's DEM sampler.
    ///
    /// Args:
    ///     `num_shots`: Number of shots to sample.
    ///     `detectors_json`: JSON string with detector definitions.
    ///     `observables_json`: JSON string with observable definitions.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Tuple of (`detection_events_per_shot`, `observable_flips_per_shot`) as numpy-compatible lists.
    #[pyo3(signature = (num_shots, detectors_json, observables_json, seed=None))]
    fn sample_batch_for_decoding(
        &self,
        num_shots: usize,
        detectors_json: &str,
        observables_json: &str,
        seed: Option<u64>,
    ) -> PyResult<BatchSampleResult> {
        use pecos_rng::PecosRng;
        use rand::RngExt;

        let detector_records = parse_detector_records(detectors_json)?;
        let observable_records = parse_observable_records(observables_json)?;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        let (all_detection_events, all_observable_flips) = self.inner.sample_batch_for_decoding(
            num_shots,
            &detector_records,
            &observable_records,
            &mut rng,
        );

        Ok((all_detection_events, all_observable_flips))
    }

    fn __repr__(&self) -> String {
        format!(
            "MeasurementNoiseModel(mechanisms={}, measurements={})",
            self.num_mechanisms(),
            self.num_measurements()
        )
    }
}

// =============================================================================
// Noisy Sampler (DEM-style sampling)
// =============================================================================

/// Fast noisy sampler for threshold estimation.
///
/// This is essentially a DEM sampler - it samples fault locations and uses
/// the influence map to determine detector and logical effects. This is the
/// recommended approach for threshold estimation as it directly samples
/// detector flips and observable flips without intermediate steps.
///
/// Two modes are supported:
/// - Uniform noise: Same error probability at all locations (fast)
/// - Circuit-level noise: Per-gate-type probabilities (p1, p2, `p_meas`, `p_init`)
///
/// # Example
///
/// ```python
/// from pecos_rslib.qec import DagFaultAnalyzer, NoisySampler
///
/// # Build influence map
/// analyzer = DagFaultAnalyzer(dag)
/// influence_map = analyzer.build_influence_map()
///
/// # Uniform noise (simple)
/// sampler = NoisySampler(influence_map, p_error=0.01, seed=42)
///
/// # Circuit-level noise (accurate)
/// sampler = NoisySampler.with_circuit_noise(
///     influence_map, p1=0.001, p2=0.01, p_meas=0.01, p_init=0.001, seed=42
/// )
///
/// # Sample for threshold estimation
/// det_events, obs_flips = sampler.sample_batch(num_shots=10000)
/// ```
#[pyclass(name = "NoisySampler", module = "pecos_rslib.qec")]
pub struct PyNoisySampler {
    /// Owned influence map (cloned from input).
    influence_map: RustDagFaultInfluenceMap,
    /// Per-location error probabilities (for circuit-level noise).
    per_location_probs: Vec<f64>,
    /// RNG seed.
    seed: u64,
}

#[pymethods]
impl PyNoisySampler {
    /// Create a new noisy sampler with uniform error probability.
    ///
    /// Args:
    ///     `influence_map`: A `DagFaultInfluenceMap` from `DagFaultAnalyzer` or `InfluenceBuilder`.
    ///     `p_error`: Uniform depolarizing error probability per fault location.
    ///     seed: Random seed for reproducibility.
    #[new]
    #[pyo3(signature = (influence_map, p_error, seed=None))]
    fn new(influence_map: &PyDagFaultInfluenceMap, p_error: f64, seed: Option<u64>) -> Self {
        use rand::RngExt;
        let actual_seed = seed.unwrap_or_else(|| rand::rng().random());
        let num_locations = influence_map.inner.locations.len();
        Self {
            influence_map: influence_map.inner.clone(),
            per_location_probs: vec![p_error; num_locations],
            seed: actual_seed,
        }
    }

    /// Create a sampler with circuit-level noise (different rates per gate type).
    ///
    /// This matches the noise model used by `DemBuilder` and Stim, with different
    /// error probabilities for different gate types.
    ///
    /// Args:
    ///     `influence_map`: A `DagFaultInfluenceMap` from `DagFaultAnalyzer` or `InfluenceBuilder`.
    ///     p1: Single-qubit gate error probability.
    ///     p2: Two-qubit gate error probability.
    ///     `p_meas`: Measurement error probability.
    ///     `p_init`: Initialization/prep error probability.
    ///     seed: Random seed for reproducibility.
    #[staticmethod]
    #[pyo3(signature = (influence_map, p1, p2, p_meas, p_init, seed=None))]
    fn with_circuit_noise(
        influence_map: &PyDagFaultInfluenceMap,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_init: f64,
        seed: Option<u64>,
    ) -> Self {
        use pecos::quantum::GateType;
        use rand::RngExt;

        let actual_seed = seed.unwrap_or_else(|| rand::rng().random());

        // Build per-location probabilities based on gate type
        let per_location_probs: Vec<f64> = influence_map
            .inner
            .locations
            .iter()
            .map(|loc| {
                #[allow(clippy::match_same_arms)] // Explicitly list known single-qubit gates
                match loc.gate_type {
                    GateType::PZ | GateType::QAlloc => p_init,
                    GateType::MZ | GateType::MeasureFree => p_meas,
                    GateType::CX | GateType::CZ | GateType::CY | GateType::SWAP => p2,
                    GateType::H
                    | GateType::SZ
                    | GateType::SZdg
                    | GateType::SX
                    | GateType::SXdg
                    | GateType::SY
                    | GateType::SYdg
                    | GateType::X
                    | GateType::Y
                    | GateType::Z
                    | GateType::T
                    | GateType::Tdg => p1,
                    _ => p1, // Default to p1 for unknown gates
                }
            })
            .collect();

        Self {
            influence_map: influence_map.inner.clone(),
            per_location_probs,
            seed: actual_seed,
        }
    }

    /// Sample a single shot.
    ///
    /// Returns:
    ///     Tuple of (`detector_flips`, `logical_flips`) where each is a list of
    ///     indices that flipped.
    fn sample_one(&mut self) -> (Vec<u32>, Vec<u32>) {
        use pecos::qec::fault_tolerance::noisy_sampler::{NoisySampler, PerLocationNoiseModel};

        let noise_model = PerLocationNoiseModel::new(self.per_location_probs.clone());
        let mut sampler = NoisySampler::new(&self.influence_map, noise_model, self.seed);
        let result = sampler.sample_one();
        // Update seed for next call
        self.seed = self.seed.wrapping_add(1);
        (result.detector_flips, result.logical_flips)
    }

    /// Sample multiple shots and return as arrays suitable for decoding.
    ///
    /// This is the main method for threshold estimation. Returns detection
    /// events and observable flips in the same format as Stim's DEM sampler.
    ///
    /// Args:
    ///     `num_shots`: Number of shots to sample.
    ///
    /// Returns:
    ///     Tuple of (`detection_events`, `observable_flips`) where:
    ///     - `detection_events`: List of lists, each inner list contains bool per detector
    ///     - `observable_flips`: List of lists, each inner list contains bool per observable
    fn sample_batch(&mut self, num_shots: usize) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        use pecos::qec::fault_tolerance::noisy_sampler::{NoisySampler, PerLocationNoiseModel};

        let noise_model = PerLocationNoiseModel::new(self.per_location_probs.clone());
        let mut sampler = NoisySampler::new(&self.influence_map, noise_model, self.seed);
        let num_detectors = self.influence_map.detectors.len();
        let num_logicals = self
            .influence_map
            .influences
            .max_logical_index()
            .map_or(1, |i| i + 1);

        let mut all_det_events = Vec::with_capacity(num_shots);
        let mut all_obs_flips = Vec::with_capacity(num_shots);

        for _ in 0..num_shots {
            let result = sampler.sample_one();

            // Convert sparse detector flips to dense bool array
            let mut det_events = vec![false; num_detectors];
            for &idx in &result.detector_flips {
                if (idx as usize) < num_detectors {
                    det_events[idx as usize] = true;
                }
            }

            // Convert sparse logical flips to dense bool array
            let mut obs_flips = vec![false; num_logicals];
            for &idx in &result.logical_flips {
                if (idx as usize) < num_logicals {
                    obs_flips[idx as usize] = true;
                }
            }

            all_det_events.push(det_events);
            all_obs_flips.push(obs_flips);
        }

        // Update seed for reproducibility of subsequent calls
        self.seed = self.seed.wrapping_add(num_shots as u64);

        (all_det_events, all_obs_flips)
    }

    /// Sample and compute statistics directly in Rust.
    ///
    /// This is more efficient than sampling and processing in Python
    /// when you only need aggregate statistics.
    ///
    /// Args:
    ///     `num_shots`: Number of shots to sample.
    ///
    /// Returns:
    ///     Dictionary with statistics:
    ///     - `total_shots`: Number of shots
    ///     - `logical_error_count`: Shots with logical errors
    ///     - `syndrome_count`: Shots with non-trivial syndrome
    ///     - `undetectable_count`: Shots with undetectable logical errors
    ///     - `logical_error_rate`: Fraction with logical errors
    ///     - `syndrome_rate`: Fraction with syndromes
    fn sample_statistics(
        &mut self,
        num_shots: usize,
        py: Python<'_>,
    ) -> PyResult<Py<pyo3::types::PyDict>> {
        use pecos::qec::fault_tolerance::noisy_sampler::{NoisySampler, PerLocationNoiseModel};

        let noise_model = PerLocationNoiseModel::new(self.per_location_probs.clone());
        let mut sampler = NoisySampler::new(&self.influence_map, noise_model, self.seed);
        let stats = sampler.sample_statistics(num_shots);

        // Update seed
        self.seed = self.seed.wrapping_add(num_shots as u64);

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("total_shots", stats.total_shots)?;
        dict.set_item("logical_error_count", stats.logical_error_count)?;
        dict.set_item("syndrome_count", stats.syndrome_count)?;
        dict.set_item("undetectable_count", stats.undetectable_count)?;
        dict.set_item("logical_error_rate", stats.logical_error_rate())?;
        dict.set_item("syndrome_rate", stats.syndrome_rate())?;
        dict.set_item("undetectable_rate", stats.undetectable_rate())?;
        dict.set_item("average_faults", stats.average_faults())?;
        Ok(dict.unbind())
    }

    /// Number of fault locations.
    #[getter]
    fn num_locations(&self) -> usize {
        self.influence_map.locations.len()
    }

    /// Number of detectors.
    #[getter]
    fn num_detectors(&self) -> usize {
        self.influence_map.detectors.len()
    }

    /// Number of logical observables.
    #[getter]
    fn num_logicals(&self) -> usize {
        self.influence_map
            .influences
            .max_logical_index()
            .map_or(1, |i| i + 1)
    }

    /// Sample with explicit detector and observable definitions.
    ///
    /// This combines `NoisySampler`'s fast per-location sampling with explicit
    /// detector/observable definitions (like MNM uses). This gives the best of
    /// both worlds: fast Rust-side sampling with Stim-compatible output.
    ///
    /// Args:
    ///     `num_shots`: Number of shots to sample.
    ///     `detectors_json`: JSON string with detector definitions.
    ///     `observables_json`: JSON string with observable definitions.
    ///     `measurement_order`: Optional list of qubit indices in `TickCircuit` measurement
    ///         execution order. Required when detector definitions use `TickCircuit`
    ///         measurement indices but the influence map uses a different ordering.
    ///         measurement_order[i] is the qubit measured at `TickCircuit` index i.
    ///
    /// Returns:
    ///     Tuple of (`detection_events`, `observable_flips`) matching Stim's format.
    #[pyo3(signature = (num_shots, detectors_json, observables_json, measurement_order=None))]
    fn sample_with_definitions(
        &mut self,
        num_shots: usize,
        detectors_json: &str,
        observables_json: &str,
        measurement_order: Option<Vec<usize>>,
    ) -> PyResult<BatchSampleResult> {
        use pecos::qec::fault_tolerance::noisy_sampler::{NoisySampler, PerLocationNoiseModel};
        use std::collections::HashMap;

        let detector_records = parse_detector_records(detectors_json)?;
        let observable_records = parse_observable_records(observables_json)?;

        let noise_model = PerLocationNoiseModel::new(self.per_location_probs.clone());
        let mut sampler = NoisySampler::new(&self.influence_map, noise_model, self.seed);
        let num_im_measurements = self.influence_map.detectors.len();

        // Build mapping from influence map indices to TickCircuit indices if measurement_order provided
        let im_to_tc: Option<Vec<usize>> = measurement_order.as_ref().map(|tc_order| {
            // Build (qubit, occurrence) -> TC index mapping
            let mut qubit_occurrences: HashMap<usize, Vec<usize>> = HashMap::new();
            for (tc_idx, &qubit) in tc_order.iter().enumerate() {
                qubit_occurrences.entry(qubit).or_default().push(tc_idx);
            }

            // Track how many times we've seen each qubit in the IM
            let mut qubit_seen_count: HashMap<usize, usize> = HashMap::new();

            // For each IM measurement, find corresponding TC index
            self.influence_map
                .measurements
                .iter()
                .map(|&(_node, qubit, _basis)| {
                    let occurrence = *qubit_seen_count.entry(qubit).or_insert(0);
                    qubit_seen_count.insert(qubit, occurrence + 1);

                    // Get the TC index for this qubit's nth occurrence
                    qubit_occurrences
                        .get(&qubit)
                        .and_then(|indices| indices.get(occurrence).copied())
                        .unwrap_or(usize::MAX)
                })
                .collect()
        });

        let num_tc_measurements = measurement_order
            .as_ref()
            .map_or(num_im_measurements, std::vec::Vec::len);

        let mut all_det_events = Vec::with_capacity(num_shots);
        let mut all_obs_flips = Vec::with_capacity(num_shots);

        for _ in 0..num_shots {
            let result = sampler.sample_one();

            // Convert sparse IM measurement flips to dense TC measurement array
            let mut meas_outcomes = vec![false; num_tc_measurements];

            if let Some(ref mapping) = im_to_tc {
                // Reorder from IM order to TC order
                for &im_idx in &result.detector_flips {
                    let im_idx = im_idx as usize;
                    if im_idx < mapping.len() {
                        let tc_idx = mapping[im_idx];
                        if tc_idx < num_tc_measurements {
                            meas_outcomes[tc_idx] = !meas_outcomes[tc_idx];
                        }
                    }
                }
            } else {
                // No reordering needed
                for &idx in &result.detector_flips {
                    if (idx as usize) < num_tc_measurements {
                        meas_outcomes[idx as usize] = true;
                    }
                }
            }

            // Apply detector definitions (XOR of measurement outcomes)
            let det_events: Vec<bool> = detector_records
                .iter()
                .map(|records| {
                    let mut fired = false;
                    for &offset in records {
                        let abs_idx = if offset < 0 {
                            (num_tc_measurements as i32 + offset) as usize
                        } else {
                            offset as usize
                        };
                        if abs_idx < num_tc_measurements && meas_outcomes[abs_idx] {
                            fired = !fired;
                        }
                    }
                    fired
                })
                .collect();

            // Apply observable definitions
            let obs_flips: Vec<bool> = observable_records
                .iter()
                .map(|records| {
                    let mut flipped = false;
                    for &offset in records {
                        let abs_idx = if offset < 0 {
                            (num_tc_measurements as i32 + offset) as usize
                        } else {
                            offset as usize
                        };
                        if abs_idx < num_tc_measurements && meas_outcomes[abs_idx] {
                            flipped = !flipped;
                        }
                    }
                    flipped
                })
                .collect();

            all_det_events.push(det_events);
            all_obs_flips.push(obs_flips);
        }

        self.seed = self.seed.wrapping_add(num_shots as u64);
        Ok((all_det_events, all_obs_flips))
    }

    fn __repr__(&self) -> String {
        format!(
            "NoisySampler(locations={}, detectors={}, logicals={})",
            self.num_locations(),
            self.num_detectors(),
            self.num_logicals(),
        )
    }
}

// =============================================================================
// MNM Builder
// =============================================================================

/// Builder for Measurement Noise Models (MNMs).
///
/// Constructs a MNM from a fault influence map. The MNM aggregates fault locations
/// by their measurement effects (which measurements flip), enabling fast approximate
/// sampling.
///
/// # Comparison with DEM
///
/// | Aspect | DEM | MNM |
/// |--------|-----|-----|
/// | Maps to | Detectors | Measurements |
/// | Use case | Decoding | Sampling |
/// | Aggregates by | Detector signature | Measurement signature |
/// | Output | Stim-compatible DEM | Raw measurement outcomes |
///
/// # Example
///
/// ```python
/// from pecos_rslib.qec import DagFaultAnalyzer, MemBuilder
///
/// # Build influence map
/// analyzer = DagFaultAnalyzer(dag)
/// influence_map = analyzer.build_influence_map()
///
/// # Build MNM for fast sampling
/// mnm = MemBuilder(influence_map).with_noise(0.01, 0.01, 0.01, 0.01).build()
///
/// # Sample many shots quickly
/// for _ in range(10000):
///     outcomes = mnm.sample()
/// ```
#[pyclass(name = "MemBuilder", module = "pecos_rslib.qec")]
pub struct PyMemBuilder {
    influence_map: RustDagFaultInfluenceMap,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_init: f64,
    /// Measurement order from original circuit (list of qubits in measurement order).
    measurement_order: Option<Vec<usize>>,
}

#[pymethods]
impl PyMemBuilder {
    /// Create a new MNM builder from a fault influence map.
    ///
    /// Args:
    ///     `influence_map`: A `DagFaultInfluenceMap` from `DagFaultAnalyzer`.
    #[new]
    fn new(influence_map: &PyDagFaultInfluenceMap) -> Self {
        Self {
            influence_map: influence_map.inner.clone(),
            p1: 0.01,
            p2: 0.01,
            p_meas: 0.01,
            p_init: 0.01,
            measurement_order: None,
        }
    }

    /// Set the noise parameters.
    ///
    /// Args:
    ///     p1: Single-qubit depolarizing error rate.
    ///     p2: Two-qubit depolarizing error rate.
    ///     `p_meas`: Measurement error rate.
    ///     `p_init`: Initialization (prep) error rate.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_noise(
        mut slf: PyRefMut<'_, Self>,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_init: f64,
    ) -> PyRefMut<'_, Self> {
        slf.p1 = p1;
        slf.p2 = p2;
        slf.p_meas = p_meas;
        slf.p_init = p_init;
        slf
    }

    /// Set the measurement order from the original circuit (e.g., `TickCircuit`).
    ///
    /// This is needed when detector definitions use `TickCircuit` measurement indices
    /// but the influence map uses a different ordering based on DAG topology.
    ///
    /// Args:
    ///     order: List of qubit indices in measurement execution order.
    ///            order[i] is the qubit measured at `TickCircuit` measurement index i.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_measurement_order(
        mut slf: PyRefMut<'_, Self>,
        order: Vec<usize>,
    ) -> PyRefMut<'_, Self> {
        slf.measurement_order = Some(order);
        slf
    }

    /// Build the Measurement Noise Model.
    ///
    /// This aggregates all fault locations by their measurement effects.
    /// Locations that produce the same measurement signature have their
    /// probabilities combined using the independent error formula.
    ///
    /// Returns:
    ///     A `MeasurementNoiseModel` for fast approximate sampling.
    fn build(&self) -> PyMeasurementNoiseModel {
        let mut builder = RustMemBuilder::new(&self.influence_map).with_noise(
            self.p1,
            self.p2,
            self.p_meas,
            self.p_init,
        );

        if let Some(ref order) = self.measurement_order {
            builder = builder.with_measurement_order(order.clone());
        }

        let inner = builder.build();
        PyMeasurementNoiseModel { inner }
    }

    fn __repr__(&self) -> String {
        format!(
            "MemBuilder(p1={}, p2={}, p_meas={}, p_init={})",
            self.p1, self.p2, self.p_meas, self.p_init
        )
    }
}

// =============================================================================
// DEM Sampler (Fast DEM-style sampling)
// =============================================================================

/// Fast DEM-style sampler for threshold estimation.
///
/// This sampler aggregates fault effects directly into detector/observable signatures,
/// matching Stim's DEM sampler semantics. It uses data-oriented design for optimal
/// cache performance:
///
/// - Precomputed u64 thresholds (no f64 comparison during sampling)
/// - CSR layout for detector/observable indices
/// - Bit-packed outcomes for compact storage and fast XOR
///
/// # Example
///
/// ```python
/// from pecos_rslib.qec import DagFaultAnalyzer, DemSamplerBuilder
///
/// # Build influence map
/// analyzer = DagFaultAnalyzer(dag)
/// influence_map = analyzer.build_influence_map()
///
/// # Build sampler with explicit detector/observable definitions
/// sampler = DemSamplerBuilder(influence_map) \
///     .with_noise(0.01, 0.01, 0.01, 0.01) \
///     .with_detectors_json(detectors_json) \
///     .with_observables_json(observables_json) \
///     .with_measurement_order(measurement_order) \
///     .build()
///
/// # Fast batch sampling for threshold estimation
/// det_events, obs_flips = sampler.sample_batch(10000)
/// ```
#[pyclass(name = "DemSampler", module = "pecos_rslib.qec")]
pub struct PyDemSampler {
    inner: pecos::qec::fault_tolerance::dem_builder::DemSampler,
}

#[pymethods]
impl PyDemSampler {
    /// Number of mechanisms in the sampler.
    #[getter]
    fn num_mechanisms(&self) -> usize {
        self.inner.num_mechanisms()
    }

    /// Number of detectors.
    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }

    /// Number of observables.
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Sample a single shot.
    ///
    /// Args:
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Tuple of (`detection_events`, `observable_flips`) as boolean lists.
    #[pyo3(signature = (seed=None))]
    fn sample(&self, seed: Option<u64>) -> (Vec<bool>, Vec<bool>) {
        use pecos_rng::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        self.inner.sample(&mut rng)
    }

    /// Sample multiple shots.
    ///
    /// Args:
    ///     `num_shots`: Number of shots to sample.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Tuple of (`all_detection_events`, `all_observable_flips`).
    #[pyo3(signature = (num_shots, seed=None))]
    fn sample_batch(
        &self,
        num_shots: usize,
        seed: Option<u64>,
    ) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        use pecos_rng::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        self.inner.sample_batch(num_shots, &mut rng)
    }

    /// Compute statistics without storing individual shots.
    ///
    /// This is the most efficient method for threshold estimation when you
    /// only need aggregate statistics (logical error rate, syndrome rate).
    ///
    /// Args:
    ///     `num_shots`: Number of shots to sample.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Dictionary with statistics:
    ///     - `total_shots`: Number of shots
    ///     - `logical_error_count`: Shots with logical errors
    ///     - `syndrome_count`: Shots with non-trivial syndrome
    ///     - `undetectable_count`: Shots with undetectable logical errors
    ///     - `logical_error_rate`: Fraction with logical errors
    ///     - `syndrome_rate`: Fraction with syndromes
    ///     - `undetectable_rate`: Fraction with undetectable errors
    #[pyo3(signature = (num_shots, seed=None))]
    fn sample_statistics(
        &self,
        num_shots: usize,
        seed: Option<u64>,
        py: Python<'_>,
    ) -> PyResult<Py<pyo3::types::PyDict>> {
        use rand::RngExt;

        let actual_seed = seed.unwrap_or_else(|| rand::rng().random());
        let stats = self.inner.sample_statistics(num_shots, actual_seed);

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("total_shots", stats.total_shots)?;
        dict.set_item("logical_error_count", stats.logical_error_count)?;
        dict.set_item("syndrome_count", stats.syndrome_count)?;
        dict.set_item("undetectable_count", stats.undetectable_count)?;
        dict.set_item("logical_error_rate", stats.logical_error_rate())?;
        dict.set_item("syndrome_rate", stats.syndrome_rate())?;
        dict.set_item("undetectable_rate", stats.undetectable_rate())?;
        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "DemSampler(mechanisms={}, detectors={}, observables={})",
            self.num_mechanisms(),
            self.num_detectors(),
            self.num_observables(),
        )
    }
}

/// Builder for `DemSampler`.
///
/// Constructs a `DemSampler` from a fault influence map, noise parameters,
/// and explicit detector/observable definitions.
#[pyclass(name = "DemSamplerBuilder", module = "pecos_rslib.qec")]
pub struct PyDemSamplerBuilder {
    influence_map: RustDagFaultInfluenceMap,
    p1: f64,
    p2: f64,
    p_meas: f64,
    p_init: f64,
    detectors_json: Option<String>,
    observables_json: Option<String>,
    measurement_order: Option<Vec<usize>>,
}

#[pymethods]
impl PyDemSamplerBuilder {
    /// Create a new builder from a fault influence map.
    #[new]
    fn new(influence_map: &PyDagFaultInfluenceMap) -> Self {
        Self {
            influence_map: influence_map.inner.clone(),
            p1: 0.01,
            p2: 0.01,
            p_meas: 0.01,
            p_init: 0.01,
            detectors_json: None,
            observables_json: None,
            measurement_order: None,
        }
    }

    /// Set noise parameters.
    fn with_noise(
        mut slf: PyRefMut<'_, Self>,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_init: f64,
    ) -> PyRefMut<'_, Self> {
        slf.p1 = p1;
        slf.p2 = p2;
        slf.p_meas = p_meas;
        slf.p_init = p_init;
        slf
    }

    /// Set detector definitions from JSON.
    fn with_detectors_json(mut slf: PyRefMut<'_, Self>, json: String) -> PyRefMut<'_, Self> {
        slf.detectors_json = Some(json);
        slf
    }

    /// Set observable definitions from JSON.
    fn with_observables_json(mut slf: PyRefMut<'_, Self>, json: String) -> PyRefMut<'_, Self> {
        slf.observables_json = Some(json);
        slf
    }

    /// Set the measurement order mapping from `TickCircuit`.
    fn with_measurement_order(
        mut slf: PyRefMut<'_, Self>,
        order: Vec<usize>,
    ) -> PyRefMut<'_, Self> {
        slf.measurement_order = Some(order);
        slf
    }

    /// Build the `DemSampler`.
    fn build(&self) -> PyResult<PyDemSampler> {
        use pecos::qec::fault_tolerance::dem_builder::DemSamplerBuilder;

        let mut builder = DemSamplerBuilder::new(&self.influence_map).with_noise(
            self.p1,
            self.p2,
            self.p_meas,
            self.p_init,
        );

        if let Some(ref json) = self.detectors_json {
            builder = builder
                .with_detectors_json(json)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
        }

        if let Some(ref json) = self.observables_json {
            builder = builder
                .with_observables_json(json)
                .map_err(pyo3::exceptions::PyValueError::new_err)?;
        }

        if let Some(ref order) = self.measurement_order {
            builder = builder.with_measurement_order(order.clone());
        }

        let inner = builder.build();
        Ok(PyDemSampler { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "DemSamplerBuilder(p1={}, p2={}, p_meas={}, p_init={})",
            self.p1, self.p2, self.p_meas, self.p_init
        )
    }
}

// =============================================================================
// DEM Equivalence Validation
// =============================================================================

/// Result of DEM equivalence comparison.
///
/// Contains detailed information about whether two DEMs are equivalent
/// and what differences were found.
///
/// # Example
///
/// ```python
/// from pecos_rslib.qec import compare_dems_exact
///
/// result = compare_dems_exact(dem1_str, dem2_str, prob_tolerance=0.001)
/// if result.equivalent:
///     print("DEMs are equivalent")
/// else:
///     print(f"Max rate difference: {result.max_rate_difference}")
///     for mech in result.only_in_dem1:
///         print(f"Only in DEM1: {mech}")
/// ```
#[pyclass(name = "EquivalenceResult", module = "pecos_rslib.qec")]
pub struct PyEquivalenceResult {
    inner: RustEquivalenceResult,
}

#[pymethods]
impl PyEquivalenceResult {
    /// Whether the DEMs are equivalent within tolerance.
    #[getter]
    fn equivalent(&self) -> bool {
        self.inner.equivalent
    }

    /// Maximum absolute difference in rates/probabilities.
    #[getter]
    fn max_rate_difference(&self) -> f64 {
        self.inner.max_rate_difference
    }

    /// Maximum relative difference in rates/probabilities.
    #[getter]
    fn max_relative_difference(&self) -> f64 {
        self.inner.max_relative_difference
    }

    /// Correlation of detector rates (statistical comparison).
    #[getter]
    fn correlation(&self) -> f64 {
        self.inner.correlation
    }

    /// Alias for correlation (matches Python API).
    #[getter]
    fn syndrome_rate_correlation(&self) -> f64 {
        self.inner.correlation
    }

    /// Per-detector rate differences (statistical comparison).
    #[getter]
    fn detector_rate_differences(&self) -> Vec<f64> {
        self.inner.detector_rate_differences.clone()
    }

    /// Per-observable rate differences (statistical comparison).
    #[getter]
    fn observable_rate_differences(&self) -> Vec<f64> {
        self.inner.observable_rate_differences.clone()
    }

    /// Number of mechanisms in first DEM.
    #[getter]
    fn dem1_mechanism_count(&self) -> usize {
        self.inner.details.dem1_mechanism_count
    }

    /// Number of mechanisms in second DEM.
    #[getter]
    fn dem2_mechanism_count(&self) -> usize {
        self.inner.details.dem2_mechanism_count
    }

    /// Mechanisms only in first DEM.
    #[getter]
    fn only_in_dem1(&self) -> Vec<String> {
        self.inner.details.only_in_dem1.clone()
    }

    /// Mechanisms only in second DEM.
    #[getter]
    fn only_in_dem2(&self) -> Vec<String> {
        self.inner.details.only_in_dem2.clone()
    }

    /// Get comparison details as a dictionary.
    fn details(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item(
            "dem1_mechanism_count",
            self.inner.details.dem1_mechanism_count,
        )?;
        dict.set_item(
            "dem2_mechanism_count",
            self.inner.details.dem2_mechanism_count,
        )?;
        dict.set_item("only_in_dem1", self.inner.details.only_in_dem1.clone())?;
        dict.set_item("only_in_dem2", self.inner.details.only_in_dem2.clone())?;

        let mismatches: Vec<_> = self
            .inner
            .details
            .prob_mismatches
            .iter()
            .map(|m| (m.target.clone(), m.dem1_prob, m.dem2_prob, m.difference))
            .collect();
        dict.set_item("prob_mismatches", mismatches)?;

        Ok(dict.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "EquivalenceResult(equivalent={}, max_rate_diff={:.6})",
            self.inner.equivalent, self.inner.max_rate_difference
        )
    }
}

/// A parsed Detector Error Model.
///
/// Parses DEM strings in Stim/PECOS format and provides methods for
/// aggregation and sampling.
///
/// # Example
///
/// ```python
/// from pecos_rslib.qec import ParsedDem
///
/// dem = ParsedDem.from_string("error(0.01) D0 D1\\nerror(0.02) D1 D2")
/// print(f"Mechanisms: {dem.num_mechanisms}")
/// print(f"Detectors: {dem.num_detectors}")
/// ```
#[pyclass(name = "ParsedDem", module = "pecos_rslib.qec")]
pub struct PyParsedDem {
    inner: RustParsedDem,
}

#[pymethods]
impl PyParsedDem {
    /// Parse a DEM from a string.
    ///
    /// Args:
    ///     `dem_str`: DEM string in Stim/PECOS format.
    ///
    /// Returns:
    ///     `ParsedDem` object.
    ///
    /// Raises:
    ///     `ValueError`: If the DEM string is malformed.
    #[staticmethod]
    fn from_string(dem_str: &str) -> PyResult<Self> {
        let inner = dem_str
            .parse::<RustParsedDem>()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of mechanisms in the DEM.
    #[getter]
    fn num_mechanisms(&self) -> usize {
        self.inner.mechanisms.len()
    }

    /// Number of detectors (max ID + 1).
    #[getter]
    fn num_detectors(&self) -> u32 {
        self.inner.num_detectors
    }

    /// Number of observables (max ID + 1).
    #[getter]
    fn num_observables(&self) -> u32 {
        self.inner.num_observables
    }

    /// Aggregate mechanisms by their effect.
    ///
    /// Returns a dictionary mapping (`detector_tuple`, `observable_tuple`) to
    /// combined probability. Probabilities are combined using the independent
    /// error formula: p1*(1-p2) + p2*(1-p1).
    ///
    /// Returns:
    ///     Dictionary of {(detectors, observables): probability}.
    fn aggregate(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let agg = self.inner.aggregate();
        let dict = pyo3::types::PyDict::new(py);

        for (key, prob) in agg {
            let det_tuple = pyo3::types::PyTuple::new(py, key.detectors.iter())?;
            let obs_tuple = pyo3::types::PyTuple::new(py, key.observables.iter())?;
            let key_tuple =
                pyo3::types::PyTuple::new(py, [det_tuple.as_any(), obs_tuple.as_any()])?;
            dict.set_item(key_tuple, prob)?;
        }

        Ok(dict.unbind())
    }

    /// Sample from this DEM.
    ///
    /// Args:
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Tuple of (`detector_events`, `observable_flips`) as boolean lists.
    #[pyo3(signature = (seed=None))]
    fn sample(&self, seed: Option<u64>) -> (Vec<bool>, Vec<bool>) {
        use pecos_rng::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        self.inner.sample(&mut rng)
    }

    /// Sample multiple shots from this DEM.
    ///
    /// Args:
    ///     `num_shots`: Number of shots to sample.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Tuple of (`all_detector_events`, `all_observable_flips`).
    #[pyo3(signature = (num_shots, seed=None))]
    fn sample_batch(
        &self,
        num_shots: usize,
        seed: Option<u64>,
    ) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        use pecos_rng::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        self.inner.sample_batch(num_shots, &mut rng)
    }

    /// Convert to an optimized `DemSampler` for fast batch sampling.
    ///
    /// The `DemSampler` uses geometric skip sampling and parallel chunked
    /// processing, which is significantly faster than `sample_batch` for
    /// large shot counts and low error rates.
    ///
    /// Returns:
    ///     `DemSampler`: Optimized sampler for this DEM.
    ///
    /// Example:
    ///     >>> dem = `ParsedDem.from_string("error(0.01)` D0 D1")
    ///     >>> sampler = `dem.to_dem_sampler()`
    ///     >>> stats = `sampler.sample_statistics(100000`, seed=42)
    fn to_dem_sampler(&self) -> PyDemSampler {
        PyDemSampler {
            inner: self.inner.to_dem_sampler(),
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "ParsedDem(mechanisms={}, detectors={}, observables={})",
            self.inner.mechanisms.len(),
            self.inner.num_detectors,
            self.inner.num_observables
        )
    }
}

/// Compare two DEMs for exact mechanism match.
///
/// This comparison aggregates mechanisms by effect and compares probabilities.
/// Appropriate for non-decomposed DEMs or when exact match is required.
///
/// Args:
///     dem1: First DEM string or `ParsedDem`.
///     dem2: Second DEM string or `ParsedDem`.
///     `prob_tolerance`: Relative tolerance for probability comparison (default 1e-6).
///
/// Returns:
///     `EquivalenceResult` with comparison statistics.
///
/// Example:
///     >>> result = `compare_dems_exact(dem1_str`, `dem2_str`, `prob_tolerance=0.001`)
///     >>> if result.equivalent:
///     ...     print("DEMs are equivalent")
#[pyfunction]
#[pyo3(signature = (dem1, dem2, prob_tolerance=1e-6))]
fn compare_dems_exact(
    dem1: &str,
    dem2: &str,
    prob_tolerance: f64,
) -> PyResult<PyEquivalenceResult> {
    let parsed1 = dem1
        .parse::<RustParsedDem>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("DEM1 parse error: {e}")))?;
    let parsed2 = dem2
        .parse::<RustParsedDem>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("DEM2 parse error: {e}")))?;

    let inner = rust_compare_dems_exact(&parsed1, &parsed2, prob_tolerance);
    Ok(PyEquivalenceResult { inner })
}

/// Compare two DEMs statistically by sampling.
///
/// This is the most robust comparison method as it accounts for all
/// decomposition strategies and probability combinations. It compares
/// the joint distribution of syndrome patterns, not just marginal rates.
///
/// Args:
///     dem1: First DEM string or `ParsedDem`.
///     dem2: Second DEM string or `ParsedDem`.
///     `num_shots`: Number of shots for sampling (default 100,000).
///     seed: Random seed (default 42).
///     tolerance: Maximum relative difference to consider equivalent (default 0.05).
///
/// Returns:
///     `EquivalenceResult` with comparison statistics.
///
/// Example:
///     >>> result = `compare_dems_statistical(dem1_str`, `dem2_str`, `num_shots=50000`)
///     >>> print(f"Correlation: {result.correlation}")
#[pyfunction]
#[pyo3(signature = (dem1, dem2, num_shots=100_000, seed=42, tolerance=0.05))]
fn compare_dems_statistical(
    dem1: &str,
    dem2: &str,
    num_shots: usize,
    seed: u64,
    tolerance: f64,
) -> PyResult<PyEquivalenceResult> {
    let parsed1 = dem1
        .parse::<RustParsedDem>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("DEM1 parse error: {e}")))?;
    let parsed2 = dem2
        .parse::<RustParsedDem>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("DEM2 parse error: {e}")))?;

    let inner = rust_compare_dems_statistical(&parsed1, &parsed2, num_shots, seed, tolerance);
    Ok(PyEquivalenceResult { inner })
}

/// Convenience function to verify DEM equivalence.
///
/// Args:
///     dem1: First DEM string.
///     dem2: Second DEM string.
///     method: Comparison method - "exact" or "statistical" (default "exact").
///     `prob_tolerance`: For exact: probability tolerance (default 1e-6).
///     `num_shots`: For statistical: number of shots (default 100,000).
///     tolerance: For statistical: rate tolerance (default 0.05).
///     seed: For statistical: random seed (default 42).
///
/// Returns:
///     True if DEMs are equivalent within tolerance.
///
/// Example:
///     >>> if `verify_dem_equivalence(dem1`, dem2, method="exact"):
///     ...     print("DEMs match exactly")
#[pyfunction]
#[pyo3(signature = (dem1, dem2, method="exact", prob_tolerance=1e-6, num_shots=100_000, tolerance=0.05, seed=42))]
fn verify_dem_equivalence(
    dem1: &str,
    dem2: &str,
    method: &str,
    prob_tolerance: f64,
    num_shots: usize,
    tolerance: f64,
    seed: u64,
) -> PyResult<bool> {
    let comparison_method = match method {
        "exact" => RustComparisonMethod::Exact { prob_tolerance },
        "statistical" => RustComparisonMethod::Statistical {
            num_shots,
            seed,
            tolerance,
        },
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "method must be 'exact' or 'statistical'",
            ));
        }
    };

    rust_verify_dem_equivalence(dem1, dem2, comparison_method)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
}

/// Assert that two DEMs are equivalent, raising an error if not.
///
/// This is a convenience function for testing that raises `AssertionError`
/// if the DEMs are not equivalent.
///
/// Args:
///     dem1: First DEM string.
///     dem2: Second DEM string.
///     method: Comparison method - "exact" or "statistical" (default "exact").
///     `prob_tolerance`: For exact: probability tolerance (default 1e-6).
///     `num_shots`: For statistical: number of shots (default 100,000).
///     tolerance: For statistical: rate tolerance (default 0.05).
///     seed: For statistical: random seed (default 42).
///
/// Raises:
///     `AssertionError`: If DEMs are not equivalent.
///
/// Example:
///     >>> `assert_dems_equivalent(dem1`, dem2, method="exact")  # Raises if not equivalent
#[pyfunction]
#[pyo3(signature = (dem1, dem2, method="exact", prob_tolerance=1e-6, num_shots=100_000, tolerance=0.05, seed=42))]
fn assert_dems_equivalent(
    dem1: &str,
    dem2: &str,
    method: &str,
    prob_tolerance: f64,
    num_shots: usize,
    tolerance: f64,
    seed: u64,
) -> PyResult<()> {
    let parsed1 = dem1
        .parse::<RustParsedDem>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("DEM1 parse error: {e}")))?;
    let parsed2 = dem2
        .parse::<RustParsedDem>()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("DEM2 parse error: {e}")))?;

    let result = match method {
        "exact" => rust_compare_dems_exact(&parsed1, &parsed2, prob_tolerance),
        "statistical" => {
            rust_compare_dems_statistical(&parsed1, &parsed2, num_shots, seed, tolerance)
        }
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "method must be 'exact' or 'statistical'",
            ));
        }
    };

    if result.equivalent {
        Ok(())
    } else {
        let msg = format!(
            "DEMs are not equivalent: max_rate_diff={:.6}, only_in_dem1={:?}, only_in_dem2={:?}",
            result.max_rate_difference, result.details.only_in_dem1, result.details.only_in_dem2
        );
        Err(pyo3::exceptions::PyAssertionError::new_err(msg))
    }
}

// =============================================================================
// Module Registration
// =============================================================================

/// Register the QEC fault tolerance module.
pub fn register_qec_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let qec = PyModule::new(m.py(), "qec")?;

    qec.add_class::<PyFaultLocation>()?;
    qec.add_class::<PyDagFaultInfluenceMap>()?;
    qec.add_class::<PyDagFaultAnalyzer>()?;
    qec.add_class::<PyInfluenceBuilder>()?;
    qec.add_class::<PyDetectorErrorModel>()?;
    qec.add_class::<PyDemBuilder>()?;
    qec.add_class::<PyMeasurementNoiseModel>()?;
    qec.add_class::<PyMemBuilder>()?;
    qec.add_class::<PyNoisySampler>()?;
    qec.add_class::<PyDemSampler>()?;
    qec.add_class::<PyDemSamplerBuilder>()?;
    qec.add_class::<PyEquivalenceResult>()?;
    qec.add_class::<PyParsedDem>()?;

    // Add DEM equivalence functions
    qec.add_function(wrap_pyfunction!(compare_dems_exact, &qec)?)?;
    qec.add_function(wrap_pyfunction!(compare_dems_statistical, &qec)?)?;
    qec.add_function(wrap_pyfunction!(verify_dem_equivalence, &qec)?)?;
    qec.add_function(wrap_pyfunction!(assert_dems_equivalent, &qec)?)?;

    // Add Pauli constants
    qec.add("PAULI_I", 0u8)?;
    qec.add("PAULI_X", 1u8)?;
    qec.add("PAULI_Y", 2u8)?;
    qec.add("PAULI_Z", 3u8)?;

    m.add_submodule(&qec)?;

    // Register in sys.modules so 'from pecos_rslib.qec import ...' works
    let sys = m.py().import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.qec", &qec)?;

    Ok(())
}
