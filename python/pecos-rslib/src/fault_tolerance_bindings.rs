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

use pecos_qec::fault_tolerance::dem_builder::{
    ComparisonMethod as RustComparisonMethod,
    ContributionEffectSummary as RustContributionEffectSummary,
    ContributionRenderRecord as RustContributionRenderRecord,
    ContributionRenderStrategy as RustContributionRenderStrategy,
    ContributionRenderSummary as RustContributionRenderSummary, DemBuilder as RustDemBuilder,
    DemSampler as RustNewDemSampler, DemSamplerBuilder as RustNewDemSamplerBuilder,
    DetectorErrorModel as RustDetectorErrorModel, DirectSourceFamily as RustDirectSourceFamily,
    EquivalenceResult as RustEquivalenceResult, FaultContribution as RustFaultContribution,
    FaultSourceType as RustFaultSourceType, NoiseConfig, ParsedDem as RustParsedDem,
    TwoDetectorDirectRenderPolicy as RustTwoDetectorDirectRenderPolicy,
    compare_dems_exact as rust_compare_dems_exact,
    compare_dems_statistical as rust_compare_dems_statistical,
    verify_dem_equivalence as rust_verify_dem_equivalence,
};
use pecos_qec::fault_tolerance::influence_builder::InfluenceBuilder as RustInfluenceBuilder;
use pecos_qec::fault_tolerance::propagator::{
    DagFaultAnalyzer as RustDagFaultAnalyzer, DagFaultInfluenceMap as RustDagFaultInfluenceMap,
    DagSpacetimeLocation, Pauli,
};
use pecos_quantum::DagCircuit;
use pecos_quantum::QubitId;
use pyo3::Py;
use pyo3::prelude::*;

type PyDemMechanismTuple = (f64, Vec<u32>, Vec<u32>);
type PyDemFitResult = (Vec<PyDemMechanismTuple>, Vec<f64>);

// Adapter for decoder factories that require `Send + Sync` trait objects.
// Decoder implementations own their state; Python access remains GIL-mediated.
struct SendWrapper(Box<dyn pecos_decoders::ObservableDecoder>);
unsafe impl Send for SendWrapper {}
unsafe impl Sync for SendWrapper {}
impl pecos_decoders::ObservableDecoder for SendWrapper {
    fn decode_to_observables(
        &mut self,
        syndrome: &[u8],
    ) -> Result<u64, pecos_decoders::DecoderError> {
        self.0.decode_to_observables(syndrome)
    }
}

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
            qubits: loc.qubits.iter().map(QubitId::index).collect(),
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
/// Maps fault locations to their effects on detectors and DEM outputs.
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
/// has_syndrome, flips_dem_output = influence_map.classify_fault(loc_idx=0, pauli=1)
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

    /// Total number of outputs in the DEM `L<n>` namespace.
    #[getter]
    fn num_dem_outputs(&self) -> usize {
        self.inner.num_dem_outputs()
    }

    /// Number of observable DEM outputs.
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Number of tracked Paulis.
    #[getter]
    fn num_tracked_paulis(&self) -> usize {
        self.inner.num_tracked_paulis()
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
    ///     Tuple (`has_syndrome`, `flips_dem_output`).
    ///     - `has_syndrome`: True if the fault flips at least one detector.
    ///     - `flips_dem_output`: True if the fault flips at least one standard observable DEM output.
    fn classify_fault(&self, loc_idx: usize, pauli: u8) -> (bool, bool) {
        (
            self.inner
                .influences
                .has_detector_flips(loc_idx, Pauli::from_u8(pauli)),
            self.inner.has_observable_flips(loc_idx, pauli),
        )
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

    /// Get standard DEM `L<n>` observable indices flipped by a fault.
    fn get_dem_output_indices(&self, loc_idx: usize, pauli: u8) -> Vec<u32> {
        self.inner.get_observable_indices(loc_idx, pauli)
    }

    /// Get raw internal non-detector influence indices flipped by a fault.
    ///
    /// These are implementation indices used to propagate both observables and
    /// tracked Paulis. Prefer `get_dem_output_indices`,
    /// `get_observable_indices`, or `get_tracked_pauli_indices` for public DEM
    /// semantics.
    fn get_internal_dem_output_indices(&self, loc_idx: usize, pauli: u8) -> Vec<u32> {
        self.inner.get_dem_output_indices(loc_idx, pauli).to_vec()
    }

    /// Get tracked-Pauli indices flipped by a fault.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///     pauli: Pauli type (1=X, 2=Y, 3=Z).
    ///
    /// Returns:
    ///     List of tracked-Pauli indices that are flipped by this fault.
    fn get_tracked_pauli_indices(&self, loc_idx: usize, pauli: u8) -> Vec<u32> {
        self.inner.get_tracked_pauli_indices(loc_idx, pauli)
    }

    /// Get observable indices flipped by a fault.
    fn get_observable_indices(&self, loc_idx: usize, pauli: u8) -> Vec<u32> {
        self.inner.get_observable_indices(loc_idx, pauli)
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

    /// Check if a fault at the given location flips any standard DEM output.
    fn has_dem_output_flips(&self, loc_idx: usize, pauli: u8) -> bool {
        self.inner.has_observable_flips(loc_idx, pauli)
    }

    /// Check if a fault at the given location flips any observable.
    fn has_observable_flips(&self, loc_idx: usize, pauli: u8) -> bool {
        self.inner.has_observable_flips(loc_idx, pauli)
    }

    /// Check if a fault at the given location flips any tracked Pauli.
    ///
    /// Args:
    ///     `loc_idx`: Location index.
    ///     pauli: Pauli type (1=X, 2=Y, 3=Z).
    ///
    /// Returns:
    ///     True if the fault flips at least one tracked Pauli.
    fn has_tracked_pauli_flips(&self, loc_idx: usize, pauli: u8) -> bool {
        self.inner.has_tracked_pauli_flips(loc_idx, pauli)
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
        dict.set_item("total_dem_output_entries", stats.total_dem_output_entries)?;
        dict.set_item("offset_bytes", stats.offset_bytes)?;
        dict.set_item("data_bytes", stats.data_bytes)?;
        dict.set_item("total_bytes", stats.total_bytes)?;
        Ok(dict.unbind())
    }

    /// Export CSR data for external use (e.g., GPU sampling).
    ///
    /// Returns:
    ///     Dictionary containing all CSR arrays:
    ///     - `num_locations`, `num_detectors`, `num_dem_outputs`
    ///     - `num_internal_dem_outputs` for the raw CSR bit-plane width
    ///     - `detector_offsets_x`, `detector_data_x`
    ///     - `detector_offsets_y`, `detector_data_y`
    ///     - `detector_offsets_z`, `detector_data_z`
    ///     - `dem_output_offsets_x`, `dem_output_data_x`
    ///     - `dem_output_offsets_y`, `dem_output_data_y`
    ///     - `dem_output_offsets_z`, `dem_output_data_z`
    fn export_csr(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let num_internal_dem_outputs = self
            .inner
            .influences
            .max_dem_output_index()
            .map_or(0, |idx| idx + 1);
        let (
            num_locations,
            num_detectors,
            num_dem_outputs,
            det_off_x,
            det_data_x,
            det_off_y,
            det_data_y,
            det_off_z,
            det_data_z,
            dem_output_offsets_x,
            dem_output_data_x,
            dem_output_offsets_y,
            dem_output_data_y,
            dem_output_offsets_z,
            dem_output_data_z,
        ) = self.inner.export_csr();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("num_locations", num_locations)?;
        dict.set_item("num_detectors", num_detectors)?;
        dict.set_item("num_dem_outputs", num_dem_outputs)?;
        dict.set_item("num_internal_dem_outputs", num_internal_dem_outputs)?;
        dict.set_item("num_observables", self.num_observables())?;
        dict.set_item("num_tracked_paulis", self.num_tracked_paulis())?;
        dict.set_item("detector_offsets_x", det_off_x)?;
        dict.set_item("detector_data_x", det_data_x)?;
        dict.set_item("detector_offsets_y", det_off_y)?;
        dict.set_item("detector_data_y", det_data_y)?;
        dict.set_item("detector_offsets_z", det_off_z)?;
        dict.set_item("detector_data_z", det_data_z)?;
        dict.set_item("dem_output_offsets_x", &dem_output_offsets_x)?;
        dict.set_item("dem_output_data_x", &dem_output_data_x)?;
        dict.set_item("dem_output_offsets_y", &dem_output_offsets_y)?;
        dict.set_item("dem_output_data_y", &dem_output_data_y)?;
        dict.set_item("dem_output_offsets_z", &dem_output_offsets_z)?;
        dict.set_item("dem_output_data_z", &dem_output_data_z)?;
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
            "DagFaultInfluenceMap(locations={}, detectors={}, tracked_paulis={})",
            self.num_locations(),
            self.num_detectors(),
            self.num_tracked_paulis()
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
/// # Build influence map with tracked Paulis
/// builder = InfluenceBuilder(dag)
/// builder.with_tracked_z([0, 1, 2])  # Track a Z string on these qubits
/// influence_map = builder.build()
/// ```
#[pyclass(name = "InfluenceBuilder", module = "pecos_rslib.qec")]
pub struct PyInfluenceBuilder {
    dag: DagCircuit,
    tracked_x_qubits: Vec<usize>,
    tracked_z_qubits: Vec<usize>,
    tracked_paulis: Vec<pecos_core::PauliString>,
    use_circuit_tracked_paulis: bool,
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
            tracked_x_qubits: Vec::new(),
            tracked_z_qubits: Vec::new(),
            tracked_paulis: Vec::new(),
            use_circuit_tracked_paulis: false,
        }
    }

    /// Add an X-string tracked Pauli.
    ///
    /// The tracked Pauli is X on all specified qubits and is sensitive to Z errors.
    ///
    /// Args:
    ///     qubits: List of qubit indices for the tracked X Pauli.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_tracked_x(mut slf: PyRefMut<'_, Self>, qubits: Vec<usize>) -> PyRefMut<'_, Self> {
        slf.tracked_x_qubits = qubits;
        slf
    }

    /// Add a Z-string tracked Pauli.
    ///
    /// The tracked Pauli is Z on all specified qubits and is sensitive to X errors.
    ///
    /// Args:
    ///     qubits: List of qubit indices for the tracked Z Pauli.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_tracked_z(mut slf: PyRefMut<'_, Self>, qubits: Vec<usize>) -> PyRefMut<'_, Self> {
        slf.tracked_z_qubits = qubits;
        slf
    }

    /// Add a tracked Pauli.
    ///
    /// Each entry is a `(qubit, pauli)` tuple where pauli is "X", "Y", or "Z".
    ///
    /// Args:
    ///     entries: List of (`qubit_index`, `pauli_str`) tuples.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_tracked_pauli(
        mut slf: PyRefMut<'_, Self>,
        entries: Vec<(usize, String)>,
    ) -> PyResult<PyRefMut<'_, Self>> {
        let paulis: Vec<(pecos_core::Pauli, pecos_core::QubitId)> = entries
            .iter()
            .map(|(qubit, p)| {
                let pauli = match p.to_uppercase().as_str() {
                    "X" => Ok(pecos_core::Pauli::X),
                    "Y" => Ok(pecos_core::Pauli::Y),
                    "Z" => Ok(pecos_core::Pauli::Z),
                    _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid Pauli type: {p}. Expected 'X', 'Y', or 'Z'."
                    ))),
                }?;
                Ok((pauli, pecos_core::QubitId::from(*qubit)))
            })
            .collect::<PyResult<_>>()?;
        slf.tracked_paulis
            .push(pecos_core::PauliString::with_phase_and_paulis(
                pecos_core::QuarterPhase::PlusOne,
                paulis,
            ));
        Ok(slf)
    }

    /// Use annotations from the circuit (observables and tracked Paulis).
    ///
    /// Extracts observable and `tracked_pauli()` annotations from the
    /// circuit. Tracked Paulis are propagated with positional awareness
    /// (only faults before each annotation's position affect it).
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_circuit_annotations(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.use_circuit_tracked_paulis = true;
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
    ///     `DagFaultInfluenceMap` with proper detector definitions and tracked Paulis.
    fn build(&self) -> PyDagFaultInfluenceMap {
        let mut builder = RustInfluenceBuilder::new(&self.dag);

        if !self.tracked_x_qubits.is_empty() {
            builder = builder.with_x(&self.tracked_x_qubits);
        }
        if !self.tracked_z_qubits.is_empty() {
            builder = builder.with_z(&self.tracked_z_qubits);
        }

        if self.use_circuit_tracked_paulis {
            builder = builder.with_circuit_annotations(&self.dag);
        }
        for pauli in &self.tracked_paulis {
            builder = builder.with_tracked_pauli(pauli.clone());
        }

        let inner = builder.build();
        PyDagFaultInfluenceMap { inner }
    }

    fn __repr__(&self) -> String {
        format!(
            "InfluenceBuilder(tracked_x={:?}, tracked_z={:?}, tracked_paulis={}, circuit_annotations={})",
            self.tracked_x_qubits,
            self.tracked_z_qubits,
            self.tracked_paulis.len(),
            self.use_circuit_tracked_paulis,
        )
    }
}

// =============================================================================
// Detector Error Model
// =============================================================================

/// A Detector Error Model (DEM) in standard DEM text format.
///
/// This represents the error model of a quantum circuit, mapping error
/// mechanisms to their probabilities. It can be exported as DEM text for use
/// with compatible decoders.
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
#[pyclass(subclass, name = "DetectorErrorModel", module = "pecos_rslib.qec")]
pub struct PyDetectorErrorModel {
    inner: RustDetectorErrorModel,
}

fn split_dem_outputs_for_dem(
    dem_outputs: &[u32],
    dem: &RustDetectorErrorModel,
) -> (Vec<u32>, Vec<u32>) {
    if dem
        .dem_outputs()
        .iter()
        .all(|output| output.kind.is_none() && output.records.is_empty() && output.pauli.is_none())
    {
        return (dem_outputs.to_vec(), Vec::new());
    }

    let mut observables = Vec::new();
    let mut tracked_paulis = Vec::new();
    for &output_id in dem_outputs {
        if let Some(output) = dem.dem_outputs().get(output_id as usize) {
            if output.is_observable() {
                observables.push(output_id);
            }
            if output.is_tracked_pauli() {
                tracked_paulis.push(output_id);
            }
        }
    }
    (observables, tracked_paulis)
}

fn contribution_summary_to_pydict(
    py: Python<'_>,
    summary: RustContributionEffectSummary,
    dem: &RustDetectorErrorModel,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("detectors", summary.effect.detectors.to_vec())?;
    let dem_outputs = summary.effect.dem_outputs.to_vec();
    let (observables, tracked_paulis) = split_dem_outputs_for_dem(&dem_outputs, dem);
    dict.set_item("dem_outputs", &dem_outputs)?;
    dict.set_item("observables", observables)?;
    dict.set_item("tracked_paulis", tracked_paulis)?;
    dict.set_item("num_contributions", summary.num_contributions)?;
    dict.set_item("total_probability", summary.total_probability)?;
    dict.set_item("direct_count", summary.direct_count)?;
    dict.set_item("direct_probability", summary.direct_probability)?;
    dict.set_item("y_decomposed_count", summary.y_decomposed_count)?;
    dict.set_item("y_decomposed_probability", summary.y_decomposed_probability)?;
    dict.set_item(
        "graphlike_decomposable_count",
        summary.graphlike_decomposable_count,
    )?;
    Ok(dict.unbind())
}

fn contribution_render_summary_to_pydict(
    py: Python<'_>,
    summary: RustContributionRenderSummary,
    dem: &RustDetectorErrorModel,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("detectors", summary.effect.detectors.to_vec())?;
    let dem_outputs = summary.effect.dem_outputs.to_vec();
    let (observables, tracked_paulis) = split_dem_outputs_for_dem(&dem_outputs, dem);
    dict.set_item("dem_outputs", &dem_outputs)?;
    dict.set_item("observables", observables)?;
    dict.set_item("tracked_paulis", tracked_paulis)?;
    dict.set_item("rendered_targets", summary.rendered_targets)?;
    dict.set_item("num_contributions", summary.num_contributions)?;
    dict.set_item("total_probability", summary.total_probability)?;
    dict.set_item("combined_probability", summary.combined_probability)?;
    dict.set_item("source_type_counts", summary.source_type_counts)?;
    dict.set_item(
        "source_type_probabilities",
        summary.source_type_probabilities,
    )?;
    dict.set_item(
        "direct_source_family_counts",
        summary.direct_source_family_counts,
    )?;
    dict.set_item(
        "direct_source_family_probabilities",
        summary.direct_source_family_probabilities,
    )?;
    Ok(dict.unbind())
}

fn contribution_render_record_to_pydict(
    py: Python<'_>,
    record: RustContributionRenderRecord,
    dem: &RustDetectorErrorModel,
) -> PyResult<Py<pyo3::types::PyDict>> {
    let dict = contribution_record_to_pydict(py, record.contribution, dem)?;
    let render_strategy = match record.render_strategy {
        RustContributionRenderStrategy::SourceComponents => "SourceComponents",
        RustContributionRenderStrategy::RecordedComponents => "RecordedComponents",
        RustContributionRenderStrategy::TwoDetectorDirect => "TwoDetectorDirect",
        RustContributionRenderStrategy::HyperedgeGraphlike => "HyperedgeGraphlike",
        RustContributionRenderStrategy::EffectDirect => "EffectDirect",
    };
    dict.bind(py)
        .set_item("rendered_targets", record.rendered_targets)?;
    dict.bind(py).set_item("render_strategy", render_strategy)?;
    if let Some(targets) = record.recorded_component_targets {
        dict.bind(py)
            .set_item("recorded_component_targets", targets)?;
    }
    Ok(dict)
}

fn parse_two_detector_direct_render_policy(
    policy: &str,
) -> PyResult<RustTwoDetectorDirectRenderPolicy> {
    match policy {
        "KeepDirect" => Ok(RustTwoDetectorDirectRenderPolicy::KeepDirect),
        "PreferRecordedComponents" => {
            Ok(RustTwoDetectorDirectRenderPolicy::PreferRecordedComponents)
        }
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown two-detector direct render policy: {policy}"
        ))),
    }
}

fn contribution_record_to_pydict(
    py: Python<'_>,
    contribution: RustFaultContribution,
    dem: &RustDetectorErrorModel,
) -> PyResult<Py<pyo3::types::PyDict>> {
    fn pauli_label(pauli: Pauli) -> &'static str {
        match pauli {
            Pauli::I => "I",
            Pauli::X => "X",
            Pauli::Y => "Y",
            Pauli::Z => "Z",
        }
    }

    let dict = pyo3::types::PyDict::new(py);
    dict.set_item("detectors", contribution.effect.detectors.to_vec())?;
    let dem_outputs = contribution.effect.dem_outputs.to_vec();
    let (observables, tracked_paulis) = split_dem_outputs_for_dem(&dem_outputs, dem);
    dict.set_item("dem_outputs", &dem_outputs)?;
    dict.set_item("observables", observables)?;
    dict.set_item("tracked_paulis", tracked_paulis)?;
    dict.set_item("probability", contribution.probability)?;
    dict.set_item("location_indices", contribution.location_indices.to_vec())?;
    dict.set_item(
        "pauli_labels",
        contribution
            .paulis
            .iter()
            .map(|pauli| pauli_label(*pauli))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item(
        "gate_type_labels",
        contribution
            .source_gate_types
            .iter()
            .map(|gate_type| format!("{gate_type:?}"))
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("before_flags", contribution.source_before_flags.to_vec())?;
    if let Some(family) = contribution.direct_source_family {
        let family_label = match family {
            RustDirectSourceFamily::SingleLocation => "SingleLocation",
            RustDirectSourceFamily::SingleLocationY => "SingleLocationY",
            RustDirectSourceFamily::TwoLocationPlainY => "TwoLocationPlainY",
            RustDirectSourceFamily::TwoLocationComponent => "TwoLocationComponent",
            RustDirectSourceFamily::TwoLocationOneSidedComponent => "TwoLocationOneSidedComponent",
            RustDirectSourceFamily::Other => "Other",
        };
        dict.set_item("direct_source_family", family_label)?;
    }

    match contribution.source_type {
        RustFaultSourceType::Direct => {
            dict.set_item("source_type", "Direct")?;
            if let Some((first, second)) = contribution.direct_component_effects {
                dict.set_item("component_1_detectors", first.detectors.to_vec())?;
                dict.set_item("component_1_dem_outputs", first.dem_outputs.to_vec())?;
                dict.set_item("component_2_detectors", second.detectors.to_vec())?;
                dict.set_item("component_2_dem_outputs", second.dem_outputs.to_vec())?;
            }
        }
        RustFaultSourceType::DirectOneSidedComponent => {
            dict.set_item("source_type", "DirectOneSidedComponent")?;
            if let Some((first, second)) = contribution.direct_component_effects {
                dict.set_item("component_1_detectors", first.detectors.to_vec())?;
                dict.set_item("component_1_dem_outputs", first.dem_outputs.to_vec())?;
                dict.set_item("component_2_detectors", second.detectors.to_vec())?;
                dict.set_item("component_2_dem_outputs", second.dem_outputs.to_vec())?;
            }
        }
        RustFaultSourceType::YDecomposed {
            x_detectors,
            x_dem_outputs,
            z_detectors,
            z_dem_outputs,
        } => {
            dict.set_item("source_type", "YDecomposed")?;
            dict.set_item("x_detectors", x_detectors.to_vec())?;
            dict.set_item("x_dem_outputs", x_dem_outputs.to_vec())?;
            dict.set_item("z_detectors", z_detectors.to_vec())?;
            dict.set_item("z_dem_outputs", z_dem_outputs.to_vec())?;
        }
    }

    Ok(dict.unbind())
}

#[pymethods]
impl PyDetectorErrorModel {
    /// Build a DetectorErrorModel directly from a circuit and noise.
    ///
    /// Accepts both `TickCircuit` and `DagCircuit`. Reads detector/tracked-Pauli
    /// definitions from circuit metadata.
    ///
    /// Example:
    ///     >>> dem = DetectorErrorModel.from_circuit(tc, p2=0.01)
    ///     >>> print(dem.to_string())
    ///     >>> sampler = dem.to_sampler()
    #[staticmethod]
    #[pyo3(signature = (circuit, p1=0.001, p2=0.01, p_meas=0.001, p_prep=0.001))]
    fn from_circuit(
        circuit: &pyo3::Bound<'_, pyo3::PyAny>,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
    ) -> PyResult<Self> {
        use pecos_qec::fault_tolerance::dem_builder::DemBuilder;

        if let Ok(dag) =
            circuit.extract::<pyo3::PyRef<'_, crate::dag_circuit_bindings::PyDagCircuit>>()
        {
            let inner = DemBuilder::try_from_circuit(&dag.inner, p1, p2, p_meas, p_prep)
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
            Ok(Self { inner })
        } else if let Ok(tc) =
            circuit.extract::<pyo3::PyRef<'_, crate::dag_circuit_bindings::PyTickCircuit>>()
        {
            let inner = DemBuilder::try_from_tick_circuit(&tc.inner, p1, p2, p_meas, p_prep)
                .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
            Ok(Self { inner })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "from_circuit() expects a DagCircuit or TickCircuit",
            ))
        }
    }

    /// Build a DetectorErrorModel from PECOS DEM metadata JSON.
    ///
    /// This imports observable and tracked-Pauli metadata only; mechanism
    /// errors must be provided through DEM text or built from a circuit.
    ///
    /// Raises:
    ///     `ValueError`: If the metadata JSON is malformed or uses unsupported fields.
    #[staticmethod]
    fn from_pecos_metadata_json(json: &str) -> PyResult<Self> {
        let inner = RustDetectorErrorModel::new()
            .with_pecos_metadata_json(json)
            .map_err(|err| pyo3::exceptions::PyValueError::new_err(err.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of detectors in the model.
    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }

    /// Number of observables in the model.
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Total number of outputs in the DEM `L<n>` namespace.
    #[getter]
    fn num_dem_outputs(&self) -> usize {
        self.inner.num_dem_outputs()
    }

    /// Number of tracked Paulis in the model.
    #[getter]
    fn num_tracked_paulis(&self) -> usize {
        self.inner.num_tracked_paulis()
    }

    /// Convert the DEM to a string in standard DEM format.
    ///
    /// Each error mechanism is output with its total probability, with no
    /// splitting into decomposed forms.
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
    /// Returns:
    ///     A string in DEM format with decomposed representations.
    fn to_string_decomposed(&self) -> String {
        self.inner.to_string_decomposed()
    }

    /// Convert the DEM to a string with an explicit direct-2det render policy.
    fn to_string_decomposed_with_two_detector_direct_policy(
        &self,
        policy: &str,
    ) -> PyResult<String> {
        let policy = parse_two_detector_direct_render_policy(policy)?;
        Ok(self
            .inner
            .to_string_decomposed_with_two_detector_direct_policy(policy))
    }

    /// Convert the DEM to a maximally decomposed graphlike representation.
    ///
    /// When possible, graphlike 2-detector mechanisms are further rewritten
    /// into XORs of standalone singleton detector effects.
    fn to_string_decomposed_maximally(&self) -> String {
        self.inner.to_string_decomposed_maximally()
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
    /// Shows each unique detector/DEM-output pattern and how many contributions
    /// target it with their total probability.
    fn all_contribution_effects(&self) -> String {
        self.inner.all_contribution_effects()
    }

    /// Build a `DemSampler` directly from this DEM — no string round-trip.
    fn to_sampler(&self) -> PyResult<PyDemSampler> {
        use pecos_qec::fault_tolerance::dem_builder::DemSampler;

        let inner = DemSampler::from_detector_error_model(&self.inner);
        Ok(PyDemSampler { inner })
    }

    /// Returns structured summaries for all unique contribution effects.
    fn contribution_effect_summaries(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        self.inner
            .contribution_effect_summaries()
            .into_iter()
            .map(|summary| contribution_summary_to_pydict(py, summary, &self.inner))
            .collect()
    }

    /// Returns structured summaries for render buckets before final regrouping.
    fn contribution_render_summaries(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        self.inner
            .contribution_render_summaries()
            .into_iter()
            .map(|summary| contribution_render_summary_to_pydict(py, summary, &self.inner))
            .collect()
    }

    /// Returns structured summaries for render buckets under an explicit
    /// direct-2det render policy.
    fn contribution_render_summaries_with_two_detector_direct_policy(
        &self,
        py: Python<'_>,
        policy: &str,
    ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        let policy = parse_two_detector_direct_render_policy(policy)?;
        self.inner
            .contribution_render_summaries_with_two_detector_direct_policy(policy)
            .into_iter()
            .map(|summary| contribution_render_summary_to_pydict(py, summary, &self.inner))
            .collect()
    }

    /// Returns per-contribution render records before final regrouping.
    fn contribution_render_records(
        &self,
        py: Python<'_>,
    ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        self.inner
            .contribution_render_records()
            .into_iter()
            .map(|record| contribution_render_record_to_pydict(py, record, &self.inner))
            .collect()
    }

    /// Returns per-contribution render records under an explicit direct-2det
    /// render policy.
    fn contribution_render_records_with_two_detector_direct_policy(
        &self,
        py: Python<'_>,
        policy: &str,
    ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        let policy = parse_two_detector_direct_render_policy(policy)?;
        self.inner
            .contribution_render_records_with_two_detector_direct_policy(policy)
            .into_iter()
            .map(|record| contribution_render_record_to_pydict(py, record, &self.inner))
            .collect()
    }

    /// Returns source-tracked contributions for a full detector/DEM-output effect.
    fn contributions_for_effect(
        &self,
        py: Python<'_>,
        detectors: Vec<u32>,
        dem_outputs: Vec<u32>,
    ) -> PyResult<Vec<Py<pyo3::types::PyDict>>> {
        self.inner
            .contributions_for_effect(&detectors, &dem_outputs)
            .into_iter()
            .map(|contribution| contribution_record_to_pydict(py, contribution, &self.inner))
            .collect()
    }

    fn __repr__(&self) -> String {
        format!(
            "DetectorErrorModel(detectors={}, dem_outputs={}, observables={}, tracked_paulis={}, contributions={})",
            self.num_detectors(),
            self.num_dem_outputs(),
            self.num_observables(),
            self.num_tracked_paulis(),
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

/// Advanced builder for Detector Error Models (DEMs).
///
/// For most use cases, prefer `DetectorErrorModel.from_circuit()` or
/// `DemSampler.from_circuit()` which handle everything automatically.
///
/// Use `DemBuilder` directly when you need:
/// - A custom fault influence map
/// - Non-standard noise configuration
/// - Manual detector and observable definitions
///
/// # Example (advanced)
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
/// builder.with_detectors_json(
///     '[{"id": 0, "coords": [0, 0, 0], "records": [-1]}, '
///     '{"detector_id": 1, "coords": [1, 0, 0], "records": [-2]}]'
/// )
/// builder.with_observables_json(
///     '[{"id": 0, "records": [-1]}, {"observable_id": 1, "records": [-2]}]'
/// )
/// dem = builder.build()
///
/// print(dem.to_string())
/// ```
#[pyclass(name = "DemBuilder", module = "pecos_rslib.qec")]
pub struct PyDemBuilder {
    influence_map: RustDagFaultInfluenceMap,
    noise: NoiseConfig,
    detectors_json: Option<String>,
    observables_json: Option<String>,
    num_measurements: Option<usize>,
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
            noise: NoiseConfig::default(),
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
    ///     `p_prep`: Initialization (prep) error rate.
    ///     `p_idle`: Optional idle noise rate per time unit.
    ///     t1: Optional T1 relaxation time.
    ///     t2: Optional T2 dephasing time.
    ///
    /// Returns:
    ///     Self for method chaining.
    #[pyo3(signature = (p1, p2, p_meas, p_prep, p_idle=None, t1=None, t2=None, idle_rz=None))]
    #[allow(clippy::too_many_arguments)]
    fn with_noise(
        mut slf: PyRefMut<'_, Self>,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
        p_idle: Option<f64>,
        t1: Option<f64>,
        t2: Option<f64>,
        idle_rz: Option<f64>,
    ) -> PyRefMut<'_, Self> {
        let mut noise = NoiseConfig::new(p1, p2, p_meas, p_prep);
        noise.p_idle = p_idle.unwrap_or(0.0);
        if let (Some(t1_val), Some(t2_val)) = (t1, t2) {
            noise = noise.set_t1_t2(t1_val, t2_val);
        }
        if let Some(rz) = idle_rz {
            noise = noise.set_idle_rz(rz);
        }
        slf.noise = noise;
        slf
    }

    /// Set the detector definitions from JSON.
    ///
    /// Args:
    ///     json: JSON string with detector definitions.
    ///           Format: [{"id": 0, "coords": [x, y, t], "records": [-1, -5]}, ...]
    ///           Public surface descriptors using "`detector_id`" are also accepted.
    ///
    /// Returns:
    ///     Self for method chaining.
    fn with_detectors_json(mut slf: PyRefMut<'_, Self>, json: String) -> PyRefMut<'_, Self> {
        slf.detectors_json = Some(json);
        slf
    }

    /// Set the observable definitions from JSON.
    ///
    /// Tracked Paulis are carried by the influence map; this helper is for
    /// observable metadata.
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
    ///     `ValueError`: If the detector or observable JSON is malformed, or
    ///         a used record offset / `meas_id` is out of range for the
    ///         configured measurement count.
    fn build(&self) -> PyResult<PyDetectorErrorModel> {
        let mut builder =
            RustDemBuilder::new(&self.influence_map).with_noise_config(self.noise.clone());

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

        let inner = builder
            .try_build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyDetectorErrorModel { inner })
    }

    /// Alias for `build()` - provided for backward compatibility.
    fn build_with_source_tracking(&self) -> PyResult<PyDetectorErrorModel> {
        self.build()
    }

    fn __repr__(&self) -> String {
        format!(
            "DemBuilder(p1={}, p2={}, p_meas={}, p_prep={}, p_idle={:?})",
            self.noise.p1, self.noise.p2, self.noise.p_meas, self.noise.p_prep, self.noise.p_idle
        )
    }
}

// =============================================================================
// Helper Functions
// =============================================================================

/// `UnionFind` decoder that passes LLRs (from DEM error priors) for weighted decoding.
///
/// The generic `CheckMatrixObservableDecoder` calls `Decoder::decode` which
/// passes empty LLRs. This wrapper stores the LLRs and passes them through
/// to the C++ UF decoder each shot, giving it edge-weight information.
struct WeightedUfObservableDecoder {
    decoder: pecos_decoders::UnionFindDecoder,
    dcm: pecos_decoder_core::dem::DemCheckMatrix,
    llrs: Vec<f64>,
}

impl pecos_decoders::ObservableDecoder for WeightedUfObservableDecoder {
    fn decode_to_observables(
        &mut self,
        syndrome: &[u8],
    ) -> Result<u64, pecos_decoder_core::DecoderError> {
        let arr = ndarray::Array1::from_vec(syndrome.to_vec());
        // bits_per_step=1: grow one bit at a time, sorted by LLR weight.
        // bits_per_step=0 with non-empty LLRs causes the C++ UF decoder to
        // add zero bits per step, looping forever.
        let result = self
            .decoder
            .decode(&arr.view(), &self.llrs, 1)
            .map_err(|e| pecos_decoder_core::DecoderError::DecodingFailed(e.to_string()))?;
        Ok(self
            .dcm
            .observables_mask_from_correction(result.decoding.as_slice().unwrap_or(&[])))
    }
}

/// Wrapper that relabels syndromes before passing to an inner decoder.
///
/// Used for Fusion Blossom parallel where detector IDs need to be
/// round-contiguous for partitioning.
struct RelabeledObservableDecoder {
    decoder: pecos_decoders::FusionBlossomDecoder,
    old_to_new: Vec<usize>,
}

impl pecos_decoders::ObservableDecoder for RelabeledObservableDecoder {
    fn decode_to_observables(
        &mut self,
        syndrome: &[u8],
    ) -> Result<u64, pecos_decoder_core::DecoderError> {
        // Relabel syndrome into the expanded vertex space (detectors + virtual + gap)
        let expected = self.decoder.num_nodes();
        let mut relabeled = vec![0u8; expected];
        for (old_id, &val) in syndrome.iter().enumerate() {
            if old_id < self.old_to_new.len() {
                let new_id = self.old_to_new[old_id];
                if new_id < expected {
                    relabeled[new_id] = val;
                }
            }
        }
        let arr = ndarray::Array1::from_vec(relabeled);
        let result = self
            .decoder
            .decode(&arr.view())
            .map_err(|e| pecos_decoder_core::DecoderError::DecodingFailed(e.to_string()))?;
        let mut mask = 0u64;
        for (i, &v) in result.observable.iter().enumerate() {
            if v != 0 {
                mask |= 1 << i;
            }
        }
        Ok(mask)
    }
}

/// Convert a `DemMatchingGraph` to a DEM string for inner decoder construction.
fn subgraph_to_dem_string(graph: &pecos_decoder_core::DemMatchingGraph) -> String {
    let mut lines = Vec::new();
    for edge in &graph.edges {
        let p = edge.probability;
        let mut targets = Vec::new();
        targets.push(format!("D{}", edge.node1));
        if let Some(n2) = edge.node2 {
            targets.push(format!("D{n2}"));
        }
        for &obs in &edge.observables {
            targets.push(format!("L{obs}"));
        }
        lines.push(format!("error({p}) {}", targets.join(" ")));
    }
    lines.join("\n")
}

/// Create an `ObservableDecoder` from a DEM string and decoder type name.
///
/// This is the shared factory used by `SampleBatch.decode_count`,
/// `DemSampler.sample_decode_count`, and the parallel variants.
fn create_observable_decoder(
    dem: &str,
    decoder_type: &str,
) -> PyResult<Box<dyn pecos_decoders::ObservableDecoder>> {
    use pecos_decoder_core::{CheckMatrixObservableDecoder, DemCheckMatrix};
    use pecos_decoders::{
        BeliefFindDecoder, BpLsdDecoder, BpMethod, BpOsdDecoder, BpSchedule, InputVectorType,
        MinSumBpBuilder, OsdMethod, PyMatchingDecoder, RelayBpBuilder, SparseMatrix,
        TesseractConfig, TesseractDecoder, UfMethod, UnionFindDecoder,
    };

    match decoder_type {
        "pymatching" => {
            // Default: correlated matching enabled (exploits X-Z correlations
            // from depolarizing noise for ~20% fewer errors at d>=5).
            let d = PyMatchingDecoder::from_dem_with_correlations(dem, true)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(Box::new(d))
        }
        "pymatching_uncorrelated" => {
            let d = PyMatchingDecoder::from_dem(dem)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(Box::new(d))
        }
        "tesseract" => {
            let d = TesseractDecoder::new(dem, TesseractConfig::fast())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(Box::new(d))
        }
        s if s.starts_with("k_mwpm") => {
            // K-MWPM: enumerate K matchings via decoding tree, majority vote.
            // Uses UF as the inner MWPM solver (supports decode_with_weights).
            use pecos_decoder_core::k_mwpm::{KMwpmConfig, KMwpmDecoder};
            let mut k: usize = 10;
            if let Some(params) = s.strip_prefix("k_mwpm:") {
                for kv in params.split(',') {
                    let parts: Vec<&str> = kv.splitn(2, '=').collect();
                    if parts.len() == 2 && (parts[0] == "K" || parts[0] == "k") {
                        k = parts[1].parse().unwrap_or(10);
                    }
                }
            }
            // Use FB (standard, non-correlated) as the inner MWPM.
            // K-MWPM captures correlation benefit by exploring multiple matchings.
            let fb = pecos_decoders::FusionBlossomDecoder::from_dem(dem)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(Box::new(KMwpmDecoder::new(fb, KMwpmConfig { k })))
        }
        "astar" => {
            let d =
                pecos_decoders::AStarDecoder::from_dem(dem, pecos_decoders::AStarConfig::default())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
            Ok(Box::new(d))
        }
        "astar_full" => {
            // A* on non-decomposed DEM (preserves hyperedges for Y-error correlations).
            let d = pecos_decoders::AStarDecoder::from_dem_full(
                dem,
                pecos_decoders::AStarConfig::default(),
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(Box::new(d))
        }
        "fusion_blossom" => {
            // Auto: use parallel for large problems (500+ detectors), serial otherwise
            let graph = pecos_decoder_core::DemMatchingGraph::from_dem_str(dem)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let has_coords = graph
                .detector_coords
                .iter()
                .any(std::option::Option::is_some);
            if graph.num_detectors >= 500 && has_coords {
                return create_observable_decoder(dem, "fusion_blossom_parallel");
            }
            create_observable_decoder(dem, "fusion_blossom_serial")
        }
        "fusion_blossom_serial" => {
            use pecos_decoder_core::DemMatchingGraph;
            use pecos_decoders::{FusionBlossomConfig, FusionBlossomDecoder};
            let graph = DemMatchingGraph::from_dem_str(dem)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            // Use absolute weight scaling. Fusion Blossom uses integer weights;
            // we multiply by 1000 for precision (matching the internal 1000x
            // scaling in add_edge). The upstream tutorial uses relative scaling
            // but that loses weight ordering when the range is narrow.

            let config = FusionBlossomConfig {
                num_nodes: Some(graph.num_detectors),
                num_observables: graph.num_observables,
                ..Default::default()
            };
            let mut decoder = FusionBlossomDecoder::new(config)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            for edge in &graph.edges {
                let obs: Vec<usize> = edge.observables.iter().map(|&o| o as usize).collect();
                let scaled_weight = edge.weight;
                match edge.node2 {
                    Some(n2) => {
                        decoder
                            .add_edge(edge.node1 as usize, n2 as usize, &obs, Some(scaled_weight))
                            .map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                            })?;
                    }
                    None => {
                        decoder
                            .add_boundary_edge(edge.node1 as usize, &obs, Some(scaled_weight))
                            .map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                            })?;
                    }
                }
            }
            Ok(Box::new(decoder))
        }
        "fusion_blossom_correlated" => {
            use pecos_decoder_core::DemMatchingGraph;
            use pecos_decoder_core::correlation_table::CorrelationTable;
            use pecos_decoder_core::two_pass_decoder::TwoPassDecoder;
            use pecos_decoders::{FusionBlossomConfig, FusionBlossomDecoder};

            let graph = DemMatchingGraph::from_dem_str(dem)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let config = FusionBlossomConfig {
                num_nodes: Some(graph.num_detectors),
                num_observables: graph.num_observables,
                ..Default::default()
            };
            let mut decoder = FusionBlossomDecoder::new(config)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            // Build edge index map and weights
            let mut edge_index_map = std::collections::BTreeMap::new();
            let mut base_weights = Vec::new();
            for (idx, edge) in graph.edges.iter().enumerate() {
                let obs: Vec<usize> = edge.observables.iter().map(|&o| o as usize).collect();
                base_weights.push(edge.weight);
                let key = if let Some(n2) = edge.node2 {
                    decoder
                        .add_edge(edge.node1 as usize, n2 as usize, &obs, Some(edge.weight))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    if edge.node1 <= n2 {
                        (edge.node1, n2)
                    } else {
                        (n2, edge.node1)
                    }
                } else {
                    decoder
                        .add_boundary_edge(edge.node1 as usize, &obs, Some(edge.weight))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    (edge.node1, u32::MAX)
                };
                edge_index_map.insert(key, idx);
            }

            // Build correlation table from DEM decomposition
            let corr_table =
                CorrelationTable::from_dem_str(dem, &edge_index_map, graph.edges.len()).map_err(
                    |e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()),
                )?;

            let two_pass = TwoPassDecoder::new(decoder, base_weights, corr_table);
            Ok(Box::new(two_pass))
        }
        s if s.starts_with("perturbed_fb_corr") => {
            // Fast perturbed correlated FB ensemble.
            // Parses DemCheckMatrix once, builds K members with perturbed weights
            // via from_check_matrix_correlated (skips DEM text re-parsing).

            use pecos_decoder_core::ensemble::EnsembleDecoder;
            use pecos_decoders::FusionBlossomDecoder;

            let mut k: usize = 5;
            let mut sigma: f64 = 0.5;
            let mut seed: u64 = 42;
            if let Some(params) = s.strip_prefix("perturbed_fb_corr:") {
                for kv in params.split(',') {
                    let parts: Vec<&str> = kv.splitn(2, '=').collect();
                    if parts.len() == 2 {
                        match parts[0] {
                            "K" | "k" => k = parts[1].parse().unwrap_or(5),
                            "sigma" | "s" => sigma = parts[1].parse().unwrap_or(0.5),
                            "seed" => seed = parts[1].parse().unwrap_or(42),
                            _ => {}
                        }
                    }
                }
            }

            // Build K members using from_dem_correlated on perturbed DEM text.
            // This is faster than the generic perturbed: path because it reuses
            // the same DEM parsing approach that FB_corr uses (which handles
            // duplicate edges correctly).
            let mut members: Vec<Box<dyn pecos_decoders::ObservableDecoder>> =
                Vec::with_capacity(k);

            // Unperturbed anchor.
            members.push(Box::new(
                FusionBlossomDecoder::from_dem_correlated(dem).map_err(|e| {
                    PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                })?,
            ));

            // K-1 perturbed members.
            let mut rng = pecos_random::PecosRng::seed_from_u64(seed);
            let mut next_f64 = || rng.next_f64();
            for _ in 1..k {
                let perturbed =
                    pecos_decoder_core::perturbed::perturb_dem(dem, sigma, &mut next_f64);
                if let Ok(dec) = FusionBlossomDecoder::from_dem_correlated(&perturbed) {
                    members.push(Box::new(dec));
                }
            }

            Ok(Box::new(EnsembleDecoder::new(members)))
        }
        "fusion_blossom_parallel" => {
            use pecos_decoder_core::DemMatchingGraph;
            use pecos_decoders::{FusionBlossomConfig, FusionBlossomDecoder, PartitionConfig};

            let graph = DemMatchingGraph::from_dem_str(dem)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            // Group detectors by time coordinate for round-contiguous relabeling.
            let mut round_groups: std::collections::BTreeMap<i64, Vec<u32>> =
                std::collections::BTreeMap::new();
            #[allow(clippy::cast_possible_truncation)] // time coords and detector IDs are small
            for (id, coord) in graph.detector_coords.iter().enumerate() {
                let t = coord
                    .as_ref()
                    .and_then(|c| c.get(2))
                    .copied()
                    .unwrap_or(0.0);
                round_groups
                    .entry((t * 1000.0) as i64)
                    .or_default()
                    .push(id as u32);
            }
            let num_rounds = round_groups.len();
            if num_rounds < 2 {
                // Not enough rounds to partition -- fall back to serial
                return create_observable_decoder(dem, "fusion_blossom");
            }

            // Relabel: each round gets [detectors] [boundary_virtual] contiguously.
            // This ensures boundary edges stay within the same vertex range as
            // the round's detectors.
            let num_dets = graph.num_detectors;
            let mut old_to_new = vec![0usize; num_dets];
            let mut det_to_round = vec![0usize; num_dets];
            let mut new_id = 0usize;
            let mut round_starts = Vec::new();
            let mut round_ends = Vec::new(); // end of each round (after boundary vertex)
            let mut partition_boundary = Vec::new();
            for (round_idx, (_round, ids)) in round_groups.iter().enumerate() {
                round_starts.push(new_id);
                for &old_id in ids {
                    old_to_new[old_id as usize] = new_id;
                    det_to_round[old_id as usize] = round_idx;
                    new_id += 1;
                }
                // Virtual boundary vertex for this round, right after detectors
                partition_boundary.push(new_id);
                new_id += 1;
                round_ends.push(new_id);
            }
            let total_vertex_num = new_id;

            let config = FusionBlossomConfig {
                num_nodes: Some(total_vertex_num),
                num_observables: graph.num_observables,
                ..Default::default()
            };
            let mut decoder = FusionBlossomDecoder::new(config)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            // Mark per-partition boundary vertices as virtual
            for &bnd in &partition_boundary {
                decoder.virtual_vertices.push(bnd);
            }

            for edge in &graph.edges {
                let obs: Vec<usize> = edge.observables.iter().map(|&o| o as usize).collect();
                let n1 = old_to_new[edge.node1 as usize];
                if let Some(n2) = edge.node2 {
                    let n2 = old_to_new[n2 as usize];
                    decoder
                        .add_edge(n1, n2, &obs, Some(edge.weight))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                } else {
                    // Route boundary edge to this detector's round boundary vertex
                    let round_idx = det_to_round[edge.node1 as usize];
                    let bnd = partition_boundary[round_idx];
                    decoder
                        .add_edge(n1, bnd, &obs, Some(edge.weight))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                }
            }

            // Build partition config matching the upstream time-partition pattern.
            // Each partition covers multiple rounds. The interface between
            // adjacent partitions is the first round of the later partition's
            // rounds (which is skipped from its range, creating the gap).

            // Partition ranges: each partition covers multiple rounds of detectors
            // plus that partition's boundary vertices.
            // Boundary vertices are at indices [num_dets, num_dets + num_rounds).
            // We assign boundary vertex for round R to the partition that contains round R.
            let partition_num = num_rounds.clamp(2, 4);

            // Build partition config. Each partition covers multiple rounds.
            // Partition boundaries fall between rounds. The first round of
            // each non-first partition is the interface gap (its vertices
            // are excluded from the partition range).
            let mut part_config = PartitionConfig::new(total_vertex_num);
            part_config.partitions.clear();

            for p_idx in 0..partition_num {
                let start_round = p_idx * num_rounds / partition_num;
                let end_round = (p_idx + 1) * num_rounds / partition_num;
                // First partition starts at its first round.
                // Subsequent partitions skip their first round (interface gap).
                let start_vertex = if p_idx == 0 {
                    round_starts[start_round]
                } else {
                    round_starts[(start_round + 1).min(num_rounds - 1)]
                };
                let end_vertex = round_ends[end_round - 1];
                if start_vertex < end_vertex {
                    part_config
                        .partitions
                        .push(pecos_decoders::VertexRange::new(start_vertex, end_vertex));
                }
            }

            // Linear fusion chain: merge adjacent partitions left to right
            let n_parts = part_config.partitions.len();
            part_config.fusions.clear();
            if n_parts > 1 {
                let mut active: Vec<usize> = (0..n_parts).collect();
                while active.len() > 1 {
                    let mut next_active = Vec::new();
                    let mut i = 0;
                    while i + 1 < active.len() {
                        part_config.fusions.push((active[i], active[i + 1]));
                        next_active.push(n_parts + part_config.fusions.len() - 1);
                        i += 2;
                    }
                    if i < active.len() {
                        next_active.push(active[i]);
                    }
                    active = next_active;
                }
            }

            decoder.set_partition_config(part_config);

            Ok(Box::new(RelabeledObservableDecoder {
                decoder,
                old_to_new,
            }))
        }
        "bp_osd" | "bp_lsd" | "belief_find" | "union_find" | "relay_bp" | "min_sum_bp" => {
            let dcm = DemCheckMatrix::from_dem_str(dem)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let sparse_h = SparseMatrix::from_dense(&dcm.check_matrix.view());
            match decoder_type {
                "bp_osd" => {
                    let d = BpOsdDecoder::new(
                        &sparse_h,
                        None,
                        Some(&dcm.error_priors),
                        100,
                        BpMethod::ProductSum,
                        BpSchedule::Parallel,
                        1.0,
                        OsdMethod::Osd0,
                        0,
                        InputVectorType::Syndrome,
                        None,
                        None,
                        None,
                    )
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    Ok(Box::new(CheckMatrixObservableDecoder::new(d, dcm)))
                }
                "bp_lsd" => {
                    let d = BpLsdDecoder::new(
                        &sparse_h,
                        None,
                        Some(&dcm.error_priors),
                        100,
                        BpMethod::ProductSum,
                        BpSchedule::Parallel,
                        1.0,
                        OsdMethod::Off,
                        0,
                        0,
                        InputVectorType::Syndrome,
                        None,
                        None,
                        None,
                    )
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    Ok(Box::new(CheckMatrixObservableDecoder::new(d, dcm)))
                }
                "belief_find" => {
                    let d = BeliefFindDecoder::new(
                        &sparse_h,
                        None,
                        Some(&dcm.error_priors),
                        100,
                        BpMethod::ProductSum,
                        1.0,
                        BpSchedule::Parallel,
                        None,
                        None,
                        None,
                        UfMethod::Inversion,
                        0,
                    )
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    Ok(Box::new(CheckMatrixObservableDecoder::new(d, dcm)))
                }
                "union_find" => {
                    let d = UnionFindDecoder::new(&sparse_h, UfMethod::Inversion).map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
                    let llrs: Vec<f64> = dcm
                        .error_priors
                        .iter()
                        .map(|&p| {
                            if p > 0.0 && p < 1.0 {
                                ((1.0 - p) / p).ln()
                            } else {
                                0.0
                            }
                        })
                        .collect();
                    Ok(Box::new(WeightedUfObservableDecoder {
                        decoder: d,
                        dcm,
                        llrs,
                    }))
                }
                "relay_bp" => {
                    let h_view = dcm.check_matrix.view();
                    let d = RelayBpBuilder::new(&h_view)
                        .error_priors(&dcm.error_priors)
                        .build()
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    Ok(Box::new(CheckMatrixObservableDecoder::new(d, dcm)))
                }
                "min_sum_bp" => {
                    let h_view = dcm.check_matrix.view();
                    let d = MinSumBpBuilder::new(&h_view)
                        .error_priors(&dcm.error_priors)
                        .build()
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    Ok(Box::new(CheckMatrixObservableDecoder::new(d, dcm)))
                }
                _ => unreachable!(),
            }
        }
        // UF decoder: "pecos_uf" (fast), "pecos_uf:balanced", "pecos_uf:accurate"
        // Also accepts legacy "pecos_uf_correlated" as alias for balanced.
        "pecos_uf" | "pecos_uf:fast" => {
            let d =
                pecos_decoders::UfDecoder::from_dem(dem, pecos_decoders::UfDecoderConfig::fast())
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(Box::new(d))
        }
        "pecos_uf:balanced" | "pecos_uf_correlated" => {
            // Two-pass correlated UF: first pass identifies matched edges,
            // correlation table adjusts weights, second pass re-decodes.
            use pecos_decoder_core::DemMatchingGraph;
            use pecos_decoder_core::correlation_table::CorrelationTable;
            use pecos_decoder_core::two_pass_decoder::TwoPassDecoder;

            let graph = DemMatchingGraph::from_dem_str(dem)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let mut edge_index_map = std::collections::BTreeMap::new();
            let mut base_weights = Vec::with_capacity(graph.edges.len());
            for (idx, edge) in graph.edges.iter().enumerate() {
                base_weights.push(edge.weight);
                let key = match edge.node2 {
                    Some(n2) => {
                        if edge.node1 <= n2 {
                            (edge.node1, n2)
                        } else {
                            (n2, edge.node1)
                        }
                    }
                    None => (edge.node1, u32::MAX),
                };
                edge_index_map.insert(key, idx);
            }

            let corr_table =
                CorrelationTable::from_dem_str(dem, &edge_index_map, graph.edges.len()).map_err(
                    |e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()),
                )?;

            let uf = pecos_decoders::UfDecoder::from_matching_graph(
                &graph,
                pecos_decoders::UfDecoderConfig::balanced(),
            );
            let two_pass = TwoPassDecoder::new(uf, base_weights, corr_table);
            Ok(Box::new(two_pass))
        }
        "pecos_uf:bp" => {
            // BP+UF hybrid: flooding BP (fast, good for d<=7).
            let d =
                pecos_decoders::BpUfDecoder::from_dem(dem, pecos_decoders::BpUfConfig::balanced())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
            Ok(Box::new(d))
        }
        "belief_matching_mgbp" => {
            // Belief-matching with matching-graph BP (Hack et al. 2026 style).
            // BP runs on the matching graph (simpler, better convergence)
            // instead of the Tanner graph.
            use pecos_decoder_core::DemMatchingGraph;
            use pecos_decoder_core::bp_matching::BpMatchingDecoder;
            use pecos_decoder_core::correlation_table::CorrelationTable;
            use pecos_decoders::{FusionBlossomConfig, FusionBlossomDecoder};

            let bp = pecos_decoders::BpUfDecoder::from_dem(
                dem,
                pecos_decoders::BpUfConfig::matching_bp(),
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let graph = DemMatchingGraph::from_dem_str(dem)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let config = FusionBlossomConfig {
                num_nodes: Some(graph.num_detectors),
                num_observables: graph.num_observables,
                ..Default::default()
            };
            let mut fb = FusionBlossomDecoder::new(config)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let mut edge_index_map = std::collections::BTreeMap::new();
            for (idx, edge) in graph.edges.iter().enumerate() {
                let obs: Vec<usize> = edge.observables.iter().map(|&o| o as usize).collect();
                let key = if let Some(n2) = edge.node2 {
                    fb.add_edge(edge.node1 as usize, n2 as usize, &obs, Some(edge.weight))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    if edge.node1 <= n2 {
                        (edge.node1, n2)
                    } else {
                        (n2, edge.node1)
                    }
                } else {
                    fb.add_boundary_edge(edge.node1 as usize, &obs, Some(edge.weight))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    (edge.node1, u32::MAX)
                };
                edge_index_map.insert(key, idx);
            }
            let corr_table =
                CorrelationTable::from_dem_str(dem, &edge_index_map, graph.edges.len()).map_err(
                    |e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()),
                )?;
            Ok(Box::new(BpMatchingDecoder::with_correlations(
                fb, bp, corr_table,
            )))
        }
        "pecos_uf:bp_serial" => {
            // BP+UF hybrid: serial BP (slower, maintains threshold at d=7-11+).
            let d =
                pecos_decoders::BpUfDecoder::from_dem(dem, pecos_decoders::BpUfConfig::accurate())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;
            Ok(Box::new(d))
        }
        "belief_matching" => {
            // Belief-matching: BP soft info → Fusion Blossom MWPM with dynamic weights.
            // Achieves ~0.94% circuit-level threshold (Higgott 2022).
            use pecos_decoder_core::DemMatchingGraph;
            use pecos_decoder_core::bp_matching::BpMatchingDecoder;
            use pecos_decoders::{FusionBlossomConfig, FusionBlossomDecoder};

            // Build BP weight provider.
            let bp =
                pecos_decoders::BpUfDecoder::from_dem(dem, pecos_decoders::BpUfConfig::balanced())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;

            // Build Fusion Blossom as the matching backend.
            let graph = DemMatchingGraph::from_dem_str(dem)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let config = FusionBlossomConfig {
                num_nodes: Some(graph.num_detectors),
                num_observables: graph.num_observables,
                ..Default::default()
            };
            let mut fb = FusionBlossomDecoder::new(config)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            for edge in &graph.edges {
                let obs: Vec<usize> = edge.observables.iter().map(|&o| o as usize).collect();
                match edge.node2 {
                    Some(n2) => {
                        fb.add_edge(edge.node1 as usize, n2 as usize, &obs, Some(edge.weight))
                            .map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                            })?;
                    }
                    None => {
                        fb.add_boundary_edge(edge.node1 as usize, &obs, Some(edge.weight))
                            .map_err(|e| {
                                PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                            })?;
                    }
                }
            }

            Ok(Box::new(BpMatchingDecoder::new(fb, bp)))
        }
        "belief_matching_correlated" => {
            // Correlated belief-matching: BP + correlation table + Fusion Blossom MWPM.
            // Two-pass: BP weights → MWPM → correlation adjustment → MWPM.
            // Combines BP soft info with X-Z cross-lattice correlations.
            use pecos_decoder_core::DemMatchingGraph;
            use pecos_decoder_core::bp_matching::BpMatchingDecoder;
            use pecos_decoder_core::correlation_table::CorrelationTable;
            use pecos_decoders::{FusionBlossomConfig, FusionBlossomDecoder};

            let bp =
                pecos_decoders::BpUfDecoder::from_dem(dem, pecos_decoders::BpUfConfig::balanced())
                    .map_err(|e| {
                        PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                    })?;

            let graph = DemMatchingGraph::from_dem_str(dem)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            // Build Fusion Blossom.
            let config = FusionBlossomConfig {
                num_nodes: Some(graph.num_detectors),
                num_observables: graph.num_observables,
                ..Default::default()
            };
            let mut fb = FusionBlossomDecoder::new(config)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            let mut edge_index_map = std::collections::BTreeMap::new();
            for (idx, edge) in graph.edges.iter().enumerate() {
                let obs: Vec<usize> = edge.observables.iter().map(|&o| o as usize).collect();
                let key = if let Some(n2) = edge.node2 {
                    fb.add_edge(edge.node1 as usize, n2 as usize, &obs, Some(edge.weight))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    if edge.node1 <= n2 {
                        (edge.node1, n2)
                    } else {
                        (n2, edge.node1)
                    }
                } else {
                    fb.add_boundary_edge(edge.node1 as usize, &obs, Some(edge.weight))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    (edge.node1, u32::MAX)
                };
                edge_index_map.insert(key, idx);
            }

            // Build correlation table from decomposed DEM.
            let corr_table =
                CorrelationTable::from_dem_str(dem, &edge_index_map, graph.edges.len()).map_err(
                    |e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()),
                )?;

            Ok(Box::new(BpMatchingDecoder::with_correlations(
                fb, bp, corr_table,
            )))
        }
        s if s.starts_with("belief_matching_hybrid:") => {
            // Hybrid correlated belief-matching: non-decomposed DEM for BP,
            // decomposed DEM for matching graph + correlations.
            // Format: "belief_matching_hybrid:<full_dem_string>"
            // The main `dem` param is the decomposed DEM.
            use pecos_decoder_core::DemMatchingGraph;
            use pecos_decoder_core::bp_matching::BpMatchingDecoder;
            use pecos_decoder_core::correlation_table::CorrelationTable;
            use pecos_decoders::{FusionBlossomConfig, FusionBlossomDecoder};

            let full_dem = &s["belief_matching_hybrid:".len()..];

            // BP uses non-decomposed DEM, matching uses decomposed.
            let bp = pecos_decoders::BpUfDecoder::from_dual_dem(
                full_dem,
                dem,
                pecos_decoders::BpUfConfig::balanced(),
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            // Build Fusion Blossom from decomposed DEM.
            let graph = DemMatchingGraph::from_dem_str(dem)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let config = FusionBlossomConfig {
                num_nodes: Some(graph.num_detectors),
                num_observables: graph.num_observables,
                ..Default::default()
            };
            let mut fb = FusionBlossomDecoder::new(config)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let mut edge_index_map = std::collections::BTreeMap::new();
            for (idx, edge) in graph.edges.iter().enumerate() {
                let obs: Vec<usize> = edge.observables.iter().map(|&o| o as usize).collect();
                let key = if let Some(n2) = edge.node2 {
                    fb.add_edge(edge.node1 as usize, n2 as usize, &obs, Some(edge.weight))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    if edge.node1 <= n2 {
                        (edge.node1, n2)
                    } else {
                        (n2, edge.node1)
                    }
                } else {
                    fb.add_boundary_edge(edge.node1 as usize, &obs, Some(edge.weight))
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                    (edge.node1, u32::MAX)
                };
                edge_index_map.insert(key, idx);
            }
            let corr_table =
                CorrelationTable::from_dem_str(dem, &edge_index_map, graph.edges.len()).map_err(
                    |e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()),
                )?;

            Ok(Box::new(BpMatchingDecoder::with_correlations(
                fb, bp, corr_table,
            )))
        }
        s if s.starts_with("windowed") => {
            // Windowed decoder: "windowed" or "windowed:step=N,buf=M,inner=TYPE,mode=MODE"
            // inner= takes the REST of the string (supports nested specs with commas).
            let mut config = pecos_decoders::WindowedConfig::default();
            let mut inner_type = "pecos_uf".to_string();
            let mut mode = String::new();
            if let Some(params) = s.strip_prefix("windowed:") {
                // Split inner= from the rest: "step=5,buf=5,inner=perturbed:K=7,sigma=0.5"
                // → params before inner, inner spec
                let (own_params, inner_spec) = if let Some(idx) = params.find(",inner=") {
                    (&params[..idx], Some(&params[idx + 7..]))
                } else if let Some(idx) = params.find("inner=") {
                    (&params[..idx.saturating_sub(1)], Some(&params[idx + 6..]))
                } else {
                    (params, None)
                };
                if let Some(spec) = inner_spec {
                    inner_type = spec.to_string();
                }
                for kv in own_params.split(',') {
                    let parts: Vec<&str> = kv.splitn(2, '=').collect();
                    if parts.len() == 2 {
                        match parts[0] {
                            "step" => config.step_size = parts[1].parse().unwrap_or(0),
                            "buf" | "buffer" => config.buffer_size = parts[1].parse().unwrap_or(0),
                            "mode" => mode = parts[1].to_string(),
                            "seam" => config.seam_half_width = parts[1].parse().unwrap_or(0),
                            "ext" | "core_extend" => {
                                config.core_extend = parts[1].parse().unwrap_or(0);
                            }
                            "wmax" | "commit_weight_max" => {
                                config.commit_weight_max = parts[1].parse().unwrap_or(0.0);
                            }
                            _ => {}
                        }
                    }
                }
            }

            if mode == "sandwich" || (mode.is_empty() && config.buffer_size > 0) {
                // Sandwich decoder (two-phase): best accuracy with buf > 0.
                // Default: buf=step, wmax=2.5, PM residual decoder.
                if config.buffer_size == 0 {
                    config.buffer_size = config.step_size;
                }
                if config.commit_weight_max == 0.0 {
                    config.commit_weight_max = 2.5;
                }
                let phase2_type = if inner_type == "pecos_uf" {
                    "pymatching".to_string()
                } else {
                    inner_type.clone()
                };
                let phase1_factory = |sub_dem: &str| -> Result<
                    pecos_decoders::UfDecoder,
                    pecos_decoders::DecoderError,
                > {
                    pecos_decoders::UfDecoder::from_dem(
                        sub_dem,
                        pecos_decoders::UfDecoderConfig::windowed(),
                    )
                };
                let phase2_factory = |sub_dem: &str| -> Result<
                    Box<dyn pecos_decoders::ObservableDecoder>,
                    pecos_decoders::DecoderError,
                > {
                    create_observable_decoder(sub_dem, &phase2_type)
                        .map_err(|e| pecos_decoders::DecoderError::InternalError(e.to_string()))
                };
                let dec = pecos_decoders::SandwichWindowedDecoder::from_dem(
                    dem,
                    config,
                    phase1_factory,
                    phase2_factory,
                )
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
                Ok(Box::new(dec))
            } else if mode == "overlap" {
                // Single-phase overlapping (UF by default).
                let factory = |sub_dem: &str| -> Result<
                    pecos_decoders::UfDecoder,
                    pecos_decoders::DecoderError,
                > {
                    pecos_decoders::UfDecoder::from_dem(
                        sub_dem,
                        pecos_decoders::UfDecoderConfig::windowed(),
                    )
                };
                let dec =
                    pecos_decoders::OverlappingWindowedDecoder::from_dem(dem, config, factory)
                        .map_err(|e| {
                            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string())
                        })?;
                Ok(Box::new(dec))
            } else {
                // Non-overlapping with pluggable inner decoder.
                let factory = |sub_dem: &str| -> Result<
                    Box<dyn pecos_decoders::ObservableDecoder>,
                    pecos_decoders::DecoderError,
                > {
                    create_observable_decoder(sub_dem, &inner_type)
                        .map_err(|e| pecos_decoders::DecoderError::InternalError(e.to_string()))
                };
                let dec = pecos_decoders::WindowedDecoder::from_dem(dem, config, factory).map_err(
                    |e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()),
                )?;
                Ok(Box::new(dec))
            }
        }
        "pecos_uf:accurate" => {
            // UIUF CSS-aware mode. Single-DEM path falls back to balanced.
            // For proper UIUF, use CssUfDecoder directly with separate X/Z DEMs
            // via the PyCssUfDecoder Python class.
            create_observable_decoder(dem, "pecos_uf:balanced")
        }
        #[cfg(feature = "mwpf")]
        "mwpf" => {
            let d =
                pecos_decoders::MwpfDecoder::from_dem(dem, pecos_decoders::MwpfConfig::default())
                    .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(Box::new(d))
        }
        #[cfg(feature = "mwpf")]
        s if s.starts_with("mwpf:") => {
            // Parse "mwpf:key=val,key=val" config overrides.
            // Keys: c/cluster_node_limit, t/timeout, once/only_solve_primal_once, solver
            let mut config = pecos_decoders::MwpfConfig::default();
            for kv in s[5..].split(',') {
                let parts: Vec<&str> = kv.splitn(2, '=').collect();
                if parts.len() != 2 {
                    continue;
                }
                match parts[0] {
                    "c" | "cluster_node_limit" => {
                        config.cluster_node_limit = parts[1].parse().unwrap_or(50);
                    }
                    "t" | "timeout" => {
                        config.timeout = parts[1].parse().ok();
                    }
                    "once" | "only_solve_primal_once" => {
                        config.only_solve_primal_once = parts[1] == "true" || parts[1] == "1";
                    }
                    "solver" => {
                        config.solver_type = match parts[1] {
                            "uf" | "union_find" => pecos_decoders::MwpfSolverType::UnionFind,
                            "sh" | "single_hair" => pecos_decoders::MwpfSolverType::SingleHair,
                            "bp" | "bp_hybrid" => pecos_decoders::MwpfSolverType::BpHybrid,
                            _ => pecos_decoders::MwpfSolverType::JointSingleHair,
                        };
                    }
                    _ => {}
                }
            }
            let d = pecos_decoders::MwpfDecoder::from_dem(dem, config)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(Box::new(d))
        }
        #[cfg(not(feature = "mwpf"))]
        s if s == "mwpf" || s.starts_with("mwpf:") => {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "MWPF decoder is not available in this build. \
                 Install cmake (run `pecos setup`) and rebuild. \
                 See: https://github.com/PECOS-packages/PECOS/blob/dev/docs/user-guide/cmake-setup.md",
            ))
        }
        s if s.starts_with("perturbed") => {
            // Perturbed-weight ensemble: "perturbed" or "perturbed:K=15,sigma=0.7,inner=TYPE"
            // inner= takes the REST of the string (supports nested decoder specs).
            use pecos_decoder_core::perturbed::{PerturbedConfig, build_perturbed_ensemble};

            let mut config = PerturbedConfig::default();
            let mut inner_type = "pymatching".to_string();
            if let Some(params) = s.strip_prefix("perturbed:") {
                // Extract inner= (takes rest of string for nesting support)
                let (own_params, inner_spec) = if let Some(idx) = params.find(",inner=") {
                    (&params[..idx], Some(&params[idx + 7..]))
                } else if let Some(idx) = params.find("inner=") {
                    (&params[..idx.saturating_sub(1)], Some(&params[idx + 6..]))
                } else {
                    (params, None)
                };
                if let Some(spec) = inner_spec {
                    inner_type = spec.to_string();
                }
                for kv in own_params.split(',') {
                    let parts: Vec<&str> = kv.splitn(2, '=').collect();
                    if parts.len() == 2 {
                        match parts[0] {
                            "K" | "k" => config.k = parts[1].parse().unwrap_or(15),
                            "sigma" | "s" => config.sigma = parts[1].parse().unwrap_or(0.7),
                            "seed" => config.seed = parts[1].parse().unwrap_or(42),
                            _ => {}
                        }
                    }
                }
            }

            let ensemble = build_perturbed_ensemble(dem, &config, |sub_dem| {
                create_observable_decoder(sub_dem, &inner_type)
                    .map_err(|e| pecos_decoders::DecoderError::InternalError(e.to_string()))
            })
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

            Ok(Box::new(ensemble))
        }
        s if s.starts_with("beamsearch") => {
            // Beam search windowed decoder: "beamsearch" or "beamsearch:K=5,sigma=0.5,buf=5"
            let mut config = pecos_decoders::BeamSearchConfig::default();
            if let Some(params) = s.strip_prefix("beamsearch:") {
                for kv in params.split(',') {
                    let parts: Vec<&str> = kv.splitn(2, '=').collect();
                    if parts.len() == 2 {
                        match parts[0] {
                            "K" | "k" => config.beam_width = parts[1].parse().unwrap_or(5),
                            "sigma" | "s" => {
                                config.perturbation_sigma = parts[1].parse().unwrap_or(0.5);
                            }
                            "seed" => config.seed = parts[1].parse().unwrap_or(42),
                            "step" => config.window.step_size = parts[1].parse().unwrap_or(0),
                            "buf" | "buffer" => {
                                config.window.buffer_size = parts[1].parse().unwrap_or(0);
                            }
                            "wmax" => {
                                config.window.commit_weight_max = parts[1].parse().unwrap_or(0.0);
                            }
                            _ => {}
                        }
                    }
                }
            }
            // Match sandwich defaults: buf=step, wmax=2.5.
            // When step=0 (auto), buf also needs to be auto. Set buf=5 as a
            // reasonable default (will be auto-tuned by parse_dem_params).
            if config.window.buffer_size == 0 {
                if config.window.step_size > 0 {
                    config.window.buffer_size = config.window.step_size;
                } else {
                    config.window.buffer_size = 5; // auto: will be refined by d_est
                }
            }
            if config.window.commit_weight_max == 0.0 {
                config.window.commit_weight_max = 2.5;
            }

            let phase1_factory = |sub_dem: &str| -> Result<
                pecos_decoders::UfDecoder,
                pecos_decoders::DecoderError,
            > {
                pecos_decoders::UfDecoder::from_dem(
                    sub_dem,
                    pecos_decoders::UfDecoderConfig::windowed(),
                )
            };
            let phase2_factory = |sub_dem: &str| -> Result<
                Box<dyn pecos_decoders::ObservableDecoder>,
                pecos_decoders::DecoderError,
            > {
                create_observable_decoder(sub_dem, "pymatching")
                    .map_err(|e| pecos_decoders::DecoderError::InternalError(e.to_string()))
            };
            let dec = pecos_decoders::BeamSearchWindowedDecoder::from_dem(
                dem,
                config,
                phase1_factory,
                Some(phase2_factory),
            )
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            Ok(Box::new(dec))
        }
        s if s.starts_with("ensemble:") => {
            // Parse "ensemble:dec1,dec2,dec3" -- create multiple decoders and vote.
            use pecos_decoder_core::ensemble::EnsembleDecoder;
            let members_str = &s[9..];
            let mut members: Vec<Box<dyn pecos_decoders::ObservableDecoder>> = Vec::new();
            for spec in members_str.split(',') {
                let spec = spec.trim();
                if spec.is_empty() {
                    continue;
                }
                members.push(create_observable_decoder(dem, spec)?);
            }
            if members.is_empty() {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                    "ensemble: needs at least one decoder",
                ));
            }
            Ok(Box::new(EnsembleDecoder::new(members)))
        }
        // Per-observable subgraph decoder: requires stab_coords from Python.
        // This is NOT callable from the string-based create_observable_decoder API.
        // Use the Python ObservableSubgraphDecoder class directly instead.
        s if s == "observable_subgraph" || s.starts_with("observable_subgraph:") => {
            Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
                "observable_subgraph decoder requires stab_coords. \
                 Use pecos_rslib.qec.ObservableSubgraphDecoder class directly.",
            ))
        }
        _ => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
            "Unsupported decoder_type: {decoder_type}. \
             Supported: pymatching, tesseract, mwpf, pecos_uf (or pecos_uf:fast/balanced/accurate), \
             observable_subgraph, ensemble:d1,d2,..., bp_osd, bp_lsd, union_find, relay_bp, min_sum_bp."
        ))),
    }
}

/// Pre-generated sample batch held in Rust memory.
///
/// Created by `DemSampler.generate_samples()`. Can be decoded by multiple
/// decoders without re-sampling, and without crossing the Rust/Python boundary
/// per shot.
///
/// # Example
///
/// ```python
/// samples = sampler.generate_samples(10000, seed=42)
/// pm_errors = samples.decode_count(dem, "pymatching")
/// ts_errors = samples.decode_count(dem, "tesseract")
/// # Both decoders ran on the exact same samples.
/// ```
#[pyclass(name = "SampleBatch", module = "pecos_rslib.qec")]
pub struct PySampleBatch {
    /// Columnar bit-packed detector columns: det_columns[det_idx][word_idx]
    det_columns: Vec<Vec<u64>>,
    /// Columnar bit-packed observable columns: obs_columns[obs_idx][word_idx]
    obs_columns: Vec<Vec<u64>>,
    num_detectors: usize,
    num_shots: usize,
}

impl PySampleBatch {
    /// Extract syndrome for one shot into a pre-allocated buffer.
    fn extract_syndrome(&self, shot: usize, buf: &mut [u8]) {
        buf.fill(0);
        let word_idx = shot / 64;
        let bit_mask = 1u64 << (shot % 64);
        for (det_idx, col) in self.det_columns.iter().enumerate() {
            if col[word_idx] & bit_mask != 0 {
                buf[det_idx] = 1;
            }
        }
    }

    /// Extract observable mask for one shot.
    fn extract_obs_mask(&self, shot: usize) -> u64 {
        let word_idx = shot / 64;
        let bit_mask = 1u64 << (shot % 64);
        let mut mask = 0u64;
        for (obs_idx, col) in self.obs_columns.iter().enumerate() {
            if col[word_idx] & bit_mask != 0 {
                mask |= 1u64 << obs_idx;
            }
        }
        mask
    }

    /// Build from columnar data (from generate_samples).
    fn from_columnar(
        det_columns: Vec<Vec<u64>>,
        obs_columns: Vec<Vec<u64>>,
        num_shots: usize,
    ) -> Self {
        let num_detectors = det_columns.len();
        Self {
            det_columns,
            obs_columns,
            num_detectors,
            num_shots,
        }
    }

    /// Build from row-major data (from Python constructor).
    fn from_row_major(detection_events: Vec<Vec<u8>>, observable_masks: Vec<u64>) -> Self {
        let num_shots = detection_events.len();
        let num_detectors = detection_events.first().map_or(0, Vec::len);
        let num_words = num_shots.div_ceil(64);

        // Convert row-major → columnar
        let mut det_columns = vec![vec![0u64; num_words]; num_detectors];
        for (shot, row) in detection_events.iter().enumerate() {
            let word_idx = shot / 64;
            let bit_mask = 1u64 << (shot % 64);
            for (det_idx, &val) in row.iter().enumerate() {
                if val != 0 {
                    det_columns[det_idx][word_idx] |= bit_mask;
                }
            }
        }

        // Find max observable index
        let max_obs = observable_masks
            .iter()
            .map(|m| 64 - m.leading_zeros() as usize)
            .max()
            .unwrap_or(0);
        let mut obs_columns = vec![vec![0u64; num_words]; max_obs];
        for (shot, &mask) in observable_masks.iter().enumerate() {
            let word_idx = shot / 64;
            let bit_mask = 1u64 << (shot % 64);
            for (obs_idx, obs_column) in obs_columns.iter_mut().enumerate().take(max_obs) {
                if mask & (1u64 << obs_idx) != 0 {
                    obs_column[word_idx] |= bit_mask;
                }
            }
        }

        Self {
            det_columns,
            obs_columns,
            num_detectors,
            num_shots,
        }
    }
}

#[pymethods]
impl PySampleBatch {
    /// Build a SampleBatch from detection event arrays and observable masks.
    ///
    /// Args:
    ///     detection_events: List of syndromes, each a list of u8 (0/1).
    ///     observable_masks: List of u64 true observable flip masks.
    #[new]
    #[pyo3(signature = (detection_events, observable_masks))]
    fn new(detection_events: Vec<Vec<u8>>, observable_masks: Vec<u64>) -> PyResult<Self> {
        if detection_events.len() != observable_masks.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "detection_events ({}) and observable_masks ({}) must have same length",
                detection_events.len(),
                observable_masks.len(),
            )));
        }
        let expected_len = detection_events.first().map_or(0, Vec::len);
        for (i, row) in detection_events.iter().enumerate() {
            if row.len() != expected_len {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "detection_events row {i} has length {} but expected {expected_len} \
                     (matching row 0)",
                    row.len()
                )));
            }
        }
        Ok(Self::from_row_major(detection_events, observable_masks))
    }

    /// Number of shots in this batch.
    #[getter]
    fn num_shots(&self) -> usize {
        self.num_shots
    }

    /// Get the syndrome for shot `i` as a list of u8 values.
    fn get_syndrome(&self, i: usize) -> PyResult<Vec<u8>> {
        if i >= self.num_shots {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Shot index {i} out of range (num_shots={})",
                self.num_shots
            )));
        }
        let mut buf = vec![0u8; self.num_detectors];
        self.extract_syndrome(i, &mut buf);
        Ok(buf)
    }

    /// Get the expected observable mask for shot `i`.
    fn get_observable_mask(&self, i: usize) -> PyResult<u64> {
        if i >= self.num_shots {
            return Err(PyErr::new::<pyo3::exceptions::PyIndexError, _>(format!(
                "Shot index {i} out of range (num_shots={})",
                self.num_shots
            )));
        }
        Ok(self.extract_obs_mask(i))
    }

    /// Decode all samples with the given decoder type and return the error count.
    ///
    /// This runs entirely in Rust -- no per-shot Python crossing.
    ///
    /// Args:
    ///     dem: DEM string in standard DEM text format for the decoder.
    ///     `decoder_type`: "pymatching", "tesseract", "`bp_osd`", "`bp_lsd`", "`union_find`",
    ///                   "`relay_bp`", or "`min_sum_bp`".
    ///
    /// Returns:
    ///     Number of logical errors.
    #[pyo3(signature = (dem, decoder_type="pymatching"))]
    fn decode_count(&self, dem: &str, decoder_type: &str) -> PyResult<usize> {
        let mut decoder = create_observable_decoder(dem, decoder_type)?;
        let mut errors = 0usize;
        let mut syndrome = vec![0u8; self.num_detectors];
        for i in 0..self.num_shots {
            self.extract_syndrome(i, &mut syndrome);
            let predicted = decoder.decode_to_observables(&syndrome).unwrap_or(u64::MAX);
            if predicted != self.extract_obs_mask(i) {
                errors += 1;
            }
        }
        Ok(errors)
    }

    /// Parallel decode: distributes samples across rayon workers.
    ///
    /// Each worker creates its own decoder instance. Faster for slow decoders.
    ///
    /// Args:
    ///     dem: DEM string for the decoder.
    ///     `decoder_type`: Decoder type string.
    ///     `num_workers`: Number of parallel workers (default: number of CPUs).
    ///
    /// Returns:
    ///     Number of logical errors.
    #[pyo3(signature = (dem, decoder_type="pymatching", num_workers=None))]
    fn decode_count_parallel(
        &self,
        dem: &str,
        decoder_type: &str,
        num_workers: Option<usize>,
    ) -> PyResult<usize> {
        use rayon::prelude::*;

        let n_workers = num_workers.unwrap_or_else(rayon::current_num_threads);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let dem_str = dem.to_string();
        let dt = decoder_type.to_string();
        let n = self.num_shots;
        let num_dets = self.num_detectors;

        // Materialize row-major data for parallel decode.
        let detection_events: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                let mut s = vec![0u8; num_dets];
                self.extract_syndrome(i, &mut s);
                s
            })
            .collect();
        let observable_masks: Vec<u64> = (0..n).map(|i| self.extract_obs_mask(i)).collect();

        let total_errors: usize = pool.install(|| {
            (0..n)
                .into_par_iter()
                .map_init(
                    || create_observable_decoder(&dem_str, &dt).unwrap(),
                    |decoder, i| {
                        let predicted = decoder
                            .decode_to_observables(&detection_events[i])
                            .unwrap_or(u64::MAX);
                        usize::from(predicted != observable_masks[i])
                    },
                )
                .sum()
        });

        Ok(total_errors)
    }

    /// Batch decode all samples at once using `PyMatching`'s batch API.
    ///
    /// Sends all detection events in a single flat array to the decoder,
    /// which can vectorize across shots. Faster than per-shot decode for
    /// `PyMatching`. Only supports pymatching decoder.
    ///
    /// Returns:
    ///     Number of logical errors.
    #[pyo3(signature = (dem))]
    fn decode_count_batch(&self, dem: &str) -> PyResult<usize> {
        use pecos_decoders::{BatchConfig, PyMatchingDecoder};

        let mut decoder = PyMatchingDecoder::from_dem(dem)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let num_detectors = decoder.num_detectors();

        // Flatten all detection events into a single contiguous array
        let mut flat = Vec::with_capacity(self.num_shots * num_detectors);
        let mut syndrome = vec![0u8; self.num_detectors];
        for i in 0..self.num_shots {
            self.extract_syndrome(i, &mut syndrome);
            // Pad or truncate to decoder's num_detectors
            let take = syndrome.len().min(num_detectors);
            flat.extend_from_slice(&syndrome[..take]);
            flat.extend(std::iter::repeat_n(0, num_detectors - take));
        }

        let config = BatchConfig {
            bit_packed_input: false,
            bit_packed_output: false,
            return_weights: false,
        };

        let result = decoder
            .decode_batch_with_config(&flat, self.num_shots, num_detectors, config)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Count errors by comparing predictions to true observable masks
        let num_observables = decoder.num_observables();
        let mut num_errors = 0usize;
        for (i, prediction) in result.predictions.iter().enumerate() {
            let mut predicted_mask = 0u64;
            for (j, &v) in prediction.iter().enumerate() {
                if v != 0 && j < num_observables {
                    predicted_mask |= 1 << j;
                }
            }
            if predicted_mask != self.extract_obs_mask(i) {
                num_errors += 1;
            }
        }

        Ok(num_errors)
    }

    /// Decode all samples and collect per-shot timing statistics.
    ///
    /// Returns a `DecodeStats` with error count, total time, median, and
    /// percentile per-shot decode times. Useful for understanding decoder
    /// performance characteristics (heavy tails, etc.).
    ///
    /// Args:
    ///     dem: DEM string for the decoder.
    ///     `decoder_type`: Decoder type string.
    ///
    /// Returns:
    ///     `DecodeStats` with timing breakdown.
    #[pyo3(signature = (dem, decoder_type="pymatching"))]
    fn decode_stats(&self, dem: &str, decoder_type: &str) -> PyResult<PyDecodeStats> {
        use std::time::Instant;

        let mut decoder = create_observable_decoder(dem, decoder_type)?;
        let mut num_errors = 0usize;
        let mut per_shot_seconds: Vec<f64> = Vec::with_capacity(self.num_shots);
        let mut syndrome = vec![0u8; self.num_detectors];

        for i in 0..self.num_shots {
            self.extract_syndrome(i, &mut syndrome);
            let t0 = Instant::now();
            let predicted = decoder.decode_to_observables(&syndrome).unwrap_or(u64::MAX);
            let elapsed = t0.elapsed().as_secs_f64();
            per_shot_seconds.push(elapsed);
            if predicted != self.extract_obs_mask(i) {
                num_errors += 1;
            }
        }

        Ok(PyDecodeStats::from_times(
            self.num_shots,
            num_errors,
            per_shot_seconds,
        ))
    }

    /// Decode all shots with per-shot timing, using parallel workers.
    ///
    /// Like `decode_stats` but distributes shots across rayon threads.
    /// Useful for slow decoders (MWPF, Tesseract, BP+OSD) where a single
    /// shot can take seconds.
    ///
    /// Per-shot timing is still collected (each worker times its own shots).
    /// The total wall-clock time is approximately `serial_total / num_workers`.
    ///
    /// Args:
    ///     dem: DEM string for the decoder.
    ///     `decoder_type`: Decoder type string.
    ///     `num_workers`: Number of parallel workers (default: number of CPUs).
    #[pyo3(signature = (dem, decoder_type="mwpf", num_workers=None))]
    fn decode_stats_parallel(
        &self,
        dem: &str,
        decoder_type: &str,
        num_workers: Option<usize>,
    ) -> PyResult<PyDecodeStats> {
        use rayon::prelude::*;

        let n_workers = num_workers.unwrap_or_else(rayon::current_num_threads);

        // Validate decoder type early.
        create_observable_decoder(dem, decoder_type)?;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let dem_str = dem.to_string();
        let dt = decoder_type.to_string();
        let num_dets = self.num_detectors;

        // Materialize row-major data for parallel decode.
        let detection_events: Vec<Vec<u8>> = (0..self.num_shots)
            .map(|i| {
                let mut s = vec![0u8; num_dets];
                self.extract_syndrome(i, &mut s);
                s
            })
            .collect();
        let observable_masks: Vec<u64> = (0..self.num_shots)
            .map(|i| self.extract_obs_mask(i))
            .collect();

        // Each worker decodes a slice of shots and returns (errors, per_shot_times).
        let results: Vec<(usize, Vec<f64>)> = pool.install(|| {
            let chunk_size = self.num_shots.div_ceil(n_workers);
            (0..n_workers)
                .into_par_iter()
                .map(|worker_id| {
                    let start = worker_id * chunk_size;
                    let end = (start + chunk_size).min(self.num_shots);
                    if start >= end {
                        return (0, Vec::new());
                    }

                    let mut decoder = create_observable_decoder(&dem_str, &dt).unwrap();
                    let mut errors = 0usize;
                    let mut times = Vec::with_capacity(end - start);

                    for i in start..end {
                        let t0 = std::time::Instant::now();
                        let predicted = decoder
                            .decode_to_observables(&detection_events[i])
                            .unwrap_or(u64::MAX);
                        times.push(t0.elapsed().as_secs_f64());
                        if predicted != observable_masks[i] {
                            errors += 1;
                        }
                    }
                    (errors, times)
                })
                .collect()
        });

        let mut total_errors = 0usize;
        let mut all_times = Vec::with_capacity(self.num_shots);
        for (errs, times) in results {
            total_errors += errs;
            all_times.extend(times);
        }

        Ok(PyDecodeStats::from_times(
            self.num_shots,
            total_errors,
            all_times,
        ))
    }

    fn __repr__(&self) -> String {
        format!("SampleBatch(num_shots={})", self.num_shots)
    }
}

/// Per-shot decode timing statistics.
#[pyclass(name = "DecodeStats", module = "pecos_rslib.qec", skip_from_py_object)]
#[derive(Clone)]
pub struct PyDecodeStats {
    #[pyo3(get)]
    pub num_shots: usize,
    #[pyo3(get)]
    pub num_errors: usize,
    #[pyo3(get)]
    pub logical_error_rate: f64,
    #[pyo3(get)]
    pub total_seconds: f64,
    #[pyo3(get)]
    pub per_shot_mean: f64,
    #[pyo3(get)]
    pub per_shot_median: f64,
    #[pyo3(get)]
    pub per_shot_p99: f64,
    #[pyo3(get)]
    pub per_shot_min: f64,
    #[pyo3(get)]
    pub per_shot_max: f64,
    /// Quantile summary for distribution visualization (violin plots).
    /// 21 values at percentiles [0, 5, 10, 15, ..., 90, 95, 100].
    #[pyo3(get)]
    pub quantiles: Vec<f64>,
}

impl PyDecodeStats {
    // Shot counts and error counts are well within f64 mantissa range (2^52).
    // Percentile index computation is bounded by array length.
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss
    )]
    fn from_times(num_shots: usize, num_errors: usize, mut times: Vec<f64>) -> Self {
        let total_seconds: f64 = times.iter().sum();
        let per_shot_mean = if num_shots > 0 {
            total_seconds / num_shots as f64
        } else {
            0.0
        };

        times.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let percentile = |p: f64| -> f64 {
            if times.is_empty() {
                return 0.0;
            }
            let idx = (p / 100.0 * (times.len() - 1) as f64).round() as usize;
            times[idx.min(times.len() - 1)]
        };

        // 21 quantiles at [0, 5, 10, ..., 95, 100] for violin plots
        let quantiles: Vec<f64> = (0..=20).map(|i| percentile(f64::from(i) * 5.0)).collect();

        Self {
            num_shots,
            num_errors,
            logical_error_rate: if num_shots > 0 {
                num_errors as f64 / num_shots as f64
            } else {
                0.0
            },
            total_seconds,
            per_shot_mean,
            per_shot_median: percentile(50.0),
            per_shot_p99: percentile(99.0),
            per_shot_min: times.first().copied().unwrap_or(0.0),
            per_shot_max: times.last().copied().unwrap_or(0.0),
            quantiles,
        }
    }
}

#[pymethods]
impl PyDecodeStats {
    fn __repr__(&self) -> String {
        format!(
            "DecodeStats(shots={}, errors={}, LER={:.4}, median={:.2e}s, p99={:.2e}s, max={:.2e}s)",
            self.num_shots,
            self.num_errors,
            self.logical_error_rate,
            self.per_shot_median,
            self.per_shot_p99,
            self.per_shot_max,
        )
    }
}

#[pyclass(name = "DemSampler", module = "pecos_rslib.qec")]
pub struct PyDemSampler {
    inner: RustNewDemSampler,
}

#[pymethods]
impl PyDemSampler {
    /// Build a sampler directly from a circuit and noise parameters.
    ///
    /// This is the simplest path: builds the influence map, extracts
    /// annotations, and configures the sampler in one step.
    ///
    /// Args:
    ///     circuit: A `DagCircuit` with gates and annotations.
    ///     p1: Single-qubit depolarizing error rate.
    ///     p2: Two-qubit depolarizing error rate.
    ///     `p_meas`: Measurement error rate.
    ///     `p_prep`: Initialization error rate.
    ///     `p_idle`: Optional idle noise rate per time unit.
    ///
    /// Example:
    ///     >>> sampler = DemSampler.from_circuit(dag, p1=0.001, p2=0.01)
    ///     >>> sampler = DemSampler.from_circuit(tc, p2=0.01)  # TickCircuit also works
    #[staticmethod]
    #[pyo3(signature = (circuit, p1=0.001, p2=0.01, p_meas=0.001, p_prep=0.001, p_idle=None, idle_rz=None))]
    fn from_circuit(
        circuit: &Bound<'_, pyo3::PyAny>,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
        p_idle: Option<f64>,
        idle_rz: Option<f64>,
    ) -> PyResult<Self> {
        let mut noise = NoiseConfig::new(p1, p2, p_meas, p_prep);
        noise.p_idle = p_idle.unwrap_or(0.0);
        if let Some(rz) = idle_rz {
            noise = noise.set_idle_rz(rz);
        }

        // Accept both DagCircuit and TickCircuit
        if let Ok(dag) =
            circuit.extract::<pyo3::PyRef<'_, crate::dag_circuit_bindings::PyDagCircuit>>()
        {
            let inner = RustNewDemSampler::from_circuit(&dag.inner, &noise)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(Self { inner })
        } else if let Ok(tc) =
            circuit.extract::<pyo3::PyRef<'_, crate::dag_circuit_bindings::PyTickCircuit>>()
        {
            let inner = RustNewDemSampler::from_tick_circuit(&tc.inner, &noise)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(Self { inner })
        } else {
            Err(pyo3::exceptions::PyTypeError::new_err(
                "from_circuit() expects a DagCircuit or TickCircuit",
            ))
        }
    }

    /// Create a sampler from a standard DEM-format string.
    ///
    /// Parses `error(p) D0 D3 L0` lines and builds a sampling engine.
    /// Useful for sampling from DEMs produced by EEG analysis.
    ///
    /// Example:
    ///     >>> from pecos_rslib_exp import eeg_heisenberg_dem
    ///     >>> dem_str = eeg_heisenberg_dem(tc, idle_rz=0.05)
    ///     >>> sampler = DemSampler.from_dem_string(dem_str)
    ///     >>> results = sampler.sample_batch(shots=1000000)
    #[staticmethod]
    #[pyo3(signature = (dem_string))]
    fn from_dem_string(dem_string: &str) -> PyResult<Self> {
        use pecos_qec::fault_tolerance::dem_builder::SamplingEngine;

        let mut mechanisms = Vec::new();
        let mut max_det = 0u32;
        let mut max_obs = 0u32;

        for line in dem_string.lines() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            // Parse: error(prob) D0 D3 L0
            let Some(rest) = line.strip_prefix("error(") else {
                continue;
            };
            let Some(paren_end) = rest.find(')') else {
                continue;
            };
            let prob: f64 = rest[..paren_end].parse().map_err(|e| {
                pyo3::exceptions::PyValueError::new_err(format!("bad probability: {e}"))
            })?;
            let tokens = rest[paren_end + 1..].split_whitespace();
            let mut dets = Vec::new();
            let mut obs = Vec::new();
            for tok in tokens {
                if let Some(d) = tok.strip_prefix('D') {
                    let id: u32 = d.parse().map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("bad detector: {e}"))
                    })?;
                    dets.push(id);
                    max_det = max_det.max(id + 1);
                } else if let Some(l) = tok.strip_prefix('L') {
                    let id: u32 = l.parse().map_err(|e| {
                        pyo3::exceptions::PyValueError::new_err(format!("bad observable: {e}"))
                    })?;
                    obs.push(id);
                    max_obs = max_obs.max(id + 1);
                }
            }
            if prob > 0.0 {
                mechanisms.push((prob, dets, obs));
            }
        }

        let engine =
            SamplingEngine::from_mechanisms(mechanisms, max_det as usize, max_obs as usize);
        let inner = RustNewDemSampler::from_engine(engine);
        Ok(Self { inner })
    }

    /// Create a sampler in raw measurement mode with uniform noise.
    #[staticmethod]
    #[pyo3(signature = (influence_map, p_error))]
    fn raw_uniform(influence_map: &PyDagFaultInfluenceMap, p_error: f64) -> PyResult<Self> {
        Self::from_influence_map(influence_map, p_error)
    }

    /// Create a sampler in raw measurement mode with circuit-level noise.
    #[staticmethod]
    #[pyo3(signature = (influence_map, p1, p2, p_meas, p_prep))]
    fn raw(
        influence_map: &PyDagFaultInfluenceMap,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
    ) -> PyResult<Self> {
        Self::from_influence_map_circuit_noise(influence_map, p1, p2, p_meas, p_prep)
    }

    /// Create a sampler in detector-event mode.
    ///
    /// The `observables` argument defines observables.
    #[staticmethod]
    #[pyo3(signature = (influence_map, detectors, observables, p1, p2, p_meas, p_prep, p_idle=None, t1=None, t2=None))]
    #[allow(clippy::too_many_arguments)]
    fn with_detectors(
        influence_map: &PyDagFaultInfluenceMap,
        detectors: Vec<Vec<i32>>,
        observables: Vec<Vec<i32>>,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
        p_idle: Option<f64>,
        t1: Option<f64>,
        t2: Option<f64>,
    ) -> PyResult<Self> {
        let mut noise = NoiseConfig::new(p1, p2, p_meas, p_prep);
        noise.p_idle = p_idle.unwrap_or(0.0);
        if let (Some(t1_val), Some(t2_val)) = (t1, t2) {
            noise = noise.set_t1_t2(t1_val, t2_val);
        }
        let inner = RustNewDemSamplerBuilder::new(&influence_map.inner)
            .with_noise_config(noise)
            .with_detectors(detectors, observables)
            .build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a sampler directly from an influence map with uniform noise.
    ///
    /// Args:
    ///     `influence_map`: A `DagFaultInfluenceMap` from `DagFaultAnalyzer` or `InfluenceBuilder`.
    ///     `p_error`: Uniform depolarizing error probability per fault location.
    #[staticmethod]
    fn from_influence_map(influence_map: &PyDagFaultInfluenceMap, p_error: f64) -> PyResult<Self> {
        let inner = RustNewDemSamplerBuilder::new(&influence_map.inner)
            .with_uniform_noise(p_error)
            .raw_measurements()
            .build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Create a sampler from an influence map with circuit-level noise.
    #[staticmethod]
    fn from_influence_map_circuit_noise(
        influence_map: &PyDagFaultInfluenceMap,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
    ) -> PyResult<Self> {
        let inner = RustNewDemSamplerBuilder::new(&influence_map.inner)
            .with_noise(p1, p2, p_meas, p_prep)
            .raw_measurements()
            .build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Number of mechanisms in the sampler.
    #[getter]
    fn num_mechanisms(&self) -> usize {
        self.inner.num_mechanisms()
    }

    /// Number of output channels (detectors or measurements).
    #[getter]
    fn num_outputs(&self) -> usize {
        self.inner.num_outputs()
    }

    /// Number of detectors (alias for `num_outputs`).
    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_outputs()
    }

    /// Number of observables when sampler metadata is known.
    #[getter]
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Total number of outputs in the DEM `L<n>` namespace.
    #[getter]
    fn num_dem_outputs(&self) -> usize {
        self.inner.num_dem_outputs()
    }

    /// Number of tracked Paulis.
    #[getter]
    fn num_tracked_paulis(&self) -> usize {
        self.inner.num_tracked_paulis()
    }

    /// Sample a single shot.
    ///
    /// Args:
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Tuple of (`detection_events`, `dem_output_flips`) as boolean lists.
    #[pyo3(signature = (seed=None))]
    fn sample(&self, seed: Option<u64>) -> (Vec<bool>, Vec<bool>) {
        use pecos_random::PecosRng;
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
    ///     Tuple of (`all_detection_events`, `all_dem_output_flips`).
    #[pyo3(signature = (num_shots, seed=None))]
    fn sample_batch(
        &self,
        num_shots: usize,
        seed: Option<u64>,
    ) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        use pecos_random::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        self.inner.sample_batch(num_shots, &mut rng)
    }

    /// Sample direct tracked-Pauli flips.
    ///
    /// Raises:
    ///     RuntimeError: If this sampler carries tracked Paulis but the
    ///         backend cannot evaluate tracked-Pauli flips directly.
    #[pyo3(signature = (seed=None))]
    fn sample_tracked_paulis(&self, seed: Option<u64>) -> PyResult<Vec<bool>> {
        use pecos_random::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        self.inner
            .sample_tracked_pauli_flips(&mut rng)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Sample direct tracked-Pauli flips for multiple shots.
    ///
    /// Raises:
    ///     RuntimeError: If this sampler carries tracked Paulis but the
    ///         backend cannot evaluate tracked-Pauli flips directly.
    #[pyo3(signature = (num_shots, seed=None))]
    fn sample_tracked_pauli_batch(
        &self,
        num_shots: usize,
        seed: Option<u64>,
    ) -> PyResult<Vec<Vec<bool>>> {
        use pecos_random::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        self.inner
            .sample_tracked_pauli_batch(num_shots, &mut rng)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Generate samples and store them in Rust memory as a `SampleBatch`.
    ///
    /// The batch can then be decoded by multiple decoders without re-sampling.
    /// This is the proper way to compare decoders: same samples, different decoders.
    ///
    /// Args:
    ///     `num_shots`: Number of shots to sample.
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     `SampleBatch` object with samples held in Rust memory.
    #[pyo3(signature = (num_shots, seed=None))]
    fn generate_samples(&self, num_shots: usize, seed: Option<u64>) -> PySampleBatch {
        use pecos_random::PecosRng;
        use rand::RngExt;

        let mut rng = match seed {
            Some(s) => PecosRng::seed_from_u64(s),
            None => PecosRng::seed_from_u64(rand::rng().random()),
        };

        // Use geometric columnar sampler via DemSampler.
        let (det_columns, obs_columns) = self.inner.sample_batch_geometric(num_shots, &mut rng);

        PySampleBatch::from_columnar(det_columns, obs_columns, num_shots)
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
    ///     - `logical_error_count`: Shots with selected observable flips
    ///     - `syndrome_count`: Shots with non-trivial syndrome
    ///     - `undetectable_count`: Shots with observable flips and no syndrome
    ///     - `logical_error_rate`: Fraction with selected observable flips
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
        let observable_indices = self.inner.observable_ids();
        let tracked_pauli_result = self.inner.tracked_pauli_ids();
        let tracked_pauli_statistics_error =
            tracked_pauli_result.as_ref().err().map(ToString::to_string);
        let tracked_pauli_indices = tracked_pauli_result.unwrap_or_default();
        let per_observable = stats.observable_counts(&observable_indices);
        let per_tracked_pauli: Vec<usize> = tracked_pauli_indices
            .iter()
            .filter_map(|&idx| stats.dem_output_counts().get(idx).copied())
            .collect();
        let logical_rates = stats.logical_rates(&observable_indices);
        #[allow(clippy::cast_precision_loss)] // Counts are converted to rates for Python reporting.
        let n = stats.total_shots as f64;
        #[allow(clippy::cast_precision_loss)] // Counts are converted to rates for Python reporting.
        let tracked_pauli_rates: Vec<f64> = per_tracked_pauli
            .iter()
            .map(|&count| count as f64 / n)
            .collect();

        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("total_shots", stats.total_shots)?;
        dict.set_item("logical_error_count", stats.logical_error_count)?;
        dict.set_item("syndrome_count", stats.syndrome_count)?;
        dict.set_item("undetectable_count", stats.undetectable_count)?;
        dict.set_item("logical_error_rate", stats.logical_error_rate())?;
        dict.set_item("syndrome_rate", stats.syndrome_rate())?;
        dict.set_item("undetectable_rate", stats.undetectable_rate())?;
        dict.set_item("per_detector", &stats.per_detector)?;
        dict.set_item("per_observable", per_observable)?;
        dict.set_item("per_tracked_pauli", per_tracked_pauli)?;
        dict.set_item("per_dem_output", stats.dem_output_counts())?;
        dict.set_item("detector_rates", stats.detector_rates())?;
        dict.set_item("logical_rates", logical_rates)?;
        dict.set_item("tracked_pauli_rates", tracked_pauli_rates)?;
        dict.set_item("dem_output_rates", stats.dem_output_rates())?;
        dict.set_item(
            "tracked_pauli_statistics_supported",
            tracked_pauli_statistics_error.is_none(),
        )?;
        if let Some(error) = tracked_pauli_statistics_error {
            dict.set_item("tracked_pauli_statistics_error", error)?;
        }
        Ok(dict.unbind())
    }

    /// Get labels for the sampler's output channels.
    ///
    /// Returns a dict with:
    ///     - `outputs`: labels for output channels (raw measurements or detectors)
    ///     - `dem_outputs`: labels for all DEM `L<n>` targets
    ///     - `observables`: labels for observables
    ///     - `tracked_paulis`: labels for tracked Paulis
    ///     - `dual_detectors`: labels for dual-output detector channels
    fn labels(&self, py: Python<'_>) -> PyResult<Py<pyo3::types::PyDict>> {
        let labels = self.inner.labels();
        let dict = pyo3::types::PyDict::new(py);
        dict.set_item("outputs", &labels.outputs)?;
        dict.set_item("dem_outputs", &labels.dem_output_labels)?;
        dict.set_item("observables", &labels.dem_output_labels)?;
        dict.set_item("tracked_paulis", &labels.tracked_pauli_labels)?;
        dict.set_item("dual_detectors", &labels.dual_detectors)?;
        Ok(dict.unbind())
    }

    /// Sample and decode in a tight Rust loop, returning only the error count.
    ///
    /// This is the fastest path for threshold estimation -- no per-shot data
    /// crosses the Rust/Python boundary. The sampler produces detection events,
    /// the decoder decodes them via the `ObservableDecoder` trait, and errors
    /// are counted, all in Rust.
    ///
    /// Args:
    ///     dem: DEM string in standard DEM text format for the decoder.
    ///     `num_shots`: Number of shots to sample and decode.
    ///     `decoder_type`: "pymatching" or "tesseract".
    ///     seed: Optional random seed for reproducibility.
    ///
    /// Returns:
    ///     Number of logical errors (mismatches between decoder prediction and true flip).
    #[pyo3(signature = (dem, num_shots, decoder_type="pymatching", seed=None))]
    fn sample_decode_count(
        &self,
        dem: &str,
        num_shots: usize,
        decoder_type: &str,
        seed: Option<u64>,
    ) -> PyResult<usize> {
        use pecos_random::PecosRng;
        use rand::RngExt;

        let actual_seed = seed.unwrap_or_else(|| rand::rng().random());
        let mut rng = PecosRng::seed_from_u64(actual_seed);

        let mut decoder = create_observable_decoder(dem, decoder_type)?;
        let observable_mask = self.inner.observable_dem_output_mask();

        // Tight sample+decode loop -- no Python involvement.
        // Single-threaded: sample and decode sequentially.
        let mut errors = 0usize;
        for _ in 0..num_shots {
            let (det_events, obs_flips) = self.inner.sample(&mut rng);
            let syndrome: Vec<u8> = det_events.iter().map(|&b| u8::from(b)).collect();
            let predicted_mask = decoder.decode_to_observables(&syndrome).unwrap_or(u64::MAX);
            let true_mask = self.inner.observable_mask_from_dem_output_flips(&obs_flips);
            if (predicted_mask & observable_mask) != true_mask {
                errors += 1;
            }
        }
        Ok(errors)
    }

    /// Parallel sample+decode: distributes shots across threads.
    ///
    /// Each thread gets its own sampler clone and decoder instance.
    /// Much faster for slow decoders (Tesseract) where decode time dominates.
    ///
    /// Args:
    ///     dem: DEM string in standard DEM text format for the decoder.
    ///     `num_shots`: Number of shots to sample and decode.
    ///     `decoder_type`: "pymatching", "tesseract", "`bp_osd`", "`bp_lsd`", or "`union_find`".
    ///     seed: Optional base random seed. Each thread gets seed + `thread_id`.
    ///     `num_workers`: Number of parallel workers (default: number of CPUs).
    ///
    /// Returns:
    ///     Number of logical errors.
    #[pyo3(signature = (dem, num_shots, decoder_type="pymatching", seed=None, num_workers=None))]
    fn sample_decode_count_parallel(
        &self,
        dem: &str,
        num_shots: usize,
        decoder_type: &str,
        seed: Option<u64>,
        num_workers: Option<usize>,
    ) -> PyResult<usize> {
        use rayon::prelude::*;

        let actual_seed = seed.unwrap_or(0);
        let n_workers = num_workers.unwrap_or_else(rayon::current_num_threads);

        // Validate decoder type early
        create_observable_decoder(dem, decoder_type)?;

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(n_workers)
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let shots_per_worker = num_shots / n_workers;
        let remainder = num_shots % n_workers;

        let sampler = &self.inner;
        let observable_mask = sampler.observable_dem_output_mask();
        let dem_str = dem.to_string();
        let dt = decoder_type.to_string();

        let total_errors: usize = pool.install(|| {
            (0..n_workers)
                .into_par_iter()
                .map(|worker_id| {
                    use pecos_random::PecosRng;

                    let my_shots = shots_per_worker + usize::from(worker_id < remainder);
                    if my_shots == 0 {
                        return 0;
                    }

                    let my_sampler = sampler.clone();
                    let mut my_rng =
                        PecosRng::seed_from_u64(actual_seed.wrapping_add(worker_id as u64));
                    // unwrap is safe: we validated above
                    let mut decoder = create_observable_decoder(&dem_str, &dt).unwrap();

                    let mut errors = 0usize;
                    for _ in 0..my_shots {
                        let (det_events, obs_flips) = my_sampler.sample(&mut my_rng);
                        let syndrome: Vec<u8> = det_events.iter().map(|&b| u8::from(b)).collect();
                        let predicted =
                            decoder.decode_to_observables(&syndrome).unwrap_or(u64::MAX);
                        let truth = my_sampler.observable_mask_from_dem_output_flips(&obs_flips);
                        if (predicted & observable_mask) != truth {
                            errors += 1;
                        }
                    }
                    errors
                })
                .sum()
        });

        Ok(total_errors)
    }

    fn __repr__(&self) -> String {
        format!(
            "DemSampler(mechanisms={}, outputs={}, dem_outputs={}, observables={}, tracked_paulis={})",
            self.num_mechanisms(),
            self.num_outputs(),
            self.num_dem_outputs(),
            self.num_observables(),
            self.num_tracked_paulis(),
        )
    }
}

/// Builder for `DemSampler`.
///
/// Constructs a `DemSampler` from a fault influence map, noise parameters,
/// and explicit detector / observable definitions.
#[pyclass(name = "DemSamplerBuilder", module = "pecos_rslib.qec")]
pub struct PyDemSamplerBuilder {
    influence_map: RustDagFaultInfluenceMap,
    noise: NoiseConfig,
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
            noise: NoiseConfig::default(),
            detectors_json: None,
            observables_json: None,
            measurement_order: None,
        }
    }

    /// Set noise parameters.
    #[pyo3(signature = (p1, p2, p_meas, p_prep, p_idle=None, t1=None, t2=None, idle_rz=None))]
    #[allow(clippy::too_many_arguments)]
    fn with_noise(
        mut slf: PyRefMut<'_, Self>,
        p1: f64,
        p2: f64,
        p_meas: f64,
        p_prep: f64,
        p_idle: Option<f64>,
        t1: Option<f64>,
        t2: Option<f64>,
        idle_rz: Option<f64>,
    ) -> PyRefMut<'_, Self> {
        let mut noise = NoiseConfig::new(p1, p2, p_meas, p_prep);
        noise.p_idle = p_idle.unwrap_or(0.0);
        if let (Some(t1_val), Some(t2_val)) = (t1, t2) {
            noise = noise.set_t1_t2(t1_val, t2_val);
        }
        if let Some(rz) = idle_rz {
            noise = noise.set_idle_rz(rz);
        }
        slf.noise = noise;
        slf
    }

    /// Set detector definitions from JSON.
    ///
    /// Accepts either legacy detector rows with an `"id"` key or public surface
    /// descriptor rows with a `"detector_id"` key.
    fn with_detectors_json(mut slf: PyRefMut<'_, Self>, json: String) -> PyRefMut<'_, Self> {
        slf.detectors_json = Some(json);
        slf
    }

    /// Set observable definitions from JSON.
    ///
    /// Tracked Paulis are carried by the influence map; this helper is for
    /// observable metadata.
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
        let mut builder = RustNewDemSamplerBuilder::new(&self.influence_map)
            .with_noise_config(self.noise.clone());

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

        let inner = builder
            .build()
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        Ok(PyDemSampler { inner })
    }

    fn __repr__(&self) -> String {
        format!(
            "DemSamplerBuilder(p1={}, p2={}, p_meas={}, p_prep={}, p_idle={:?})",
            self.noise.p1, self.noise.p2, self.noise.p_meas, self.noise.p_prep, self.noise.p_idle
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
/// Parses standard and PECOS DEM strings and provides methods for
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
    ///     `dem_str`: DEM string in standard or PECOS DEM text format.
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

    /// Number of observables.
    #[getter]
    fn num_observables(&self) -> u32 {
        self.inner.num_observables()
    }

    /// Total number of outputs in the DEM `L<n>` namespace.
    #[getter]
    fn num_dem_outputs(&self) -> u32 {
        self.inner.num_dem_outputs()
    }

    /// Number of tracked Paulis.
    #[getter]
    fn num_tracked_paulis(&self) -> u32 {
        self.inner.num_tracked_paulis()
    }

    /// Convert to a decomposed (graphlike) DEM string.
    ///
    /// Mechanisms with <= 2 detectors pass through unchanged.
    /// Hyperedges (3+ detectors) cannot be decomposed without Pauli
    /// provenance and will raise an error.
    ///
    /// For proper decomposition, use ``coherent_dem_decomposed()``
    /// or ``noise_characterization()`` which track X/Z components.
    fn to_string_decomposed(&self) -> PyResult<String> {
        self.inner
            .to_string_decomposed()
            .map_err(pyo3::exceptions::PyValueError::new_err)
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
    ///     Tuple of (`detector_events`, `dem_output_flips`) as boolean lists.
    #[pyo3(signature = (seed=None))]
    fn sample(&self, seed: Option<u64>) -> (Vec<bool>, Vec<bool>) {
        use pecos_random::PecosRng;
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
    ///     Tuple of (`all_detector_events`, `all_dem_output_flips`).
    #[pyo3(signature = (num_shots, seed=None))]
    fn sample_batch(
        &self,
        num_shots: usize,
        seed: Option<u64>,
    ) -> (Vec<Vec<bool>>, Vec<Vec<bool>>) {
        use pecos_random::PecosRng;
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
            "ParsedDem(mechanisms={}, detectors={}, dem_outputs={}, observables={}, tracked_paulis={})",
            self.inner.mechanisms.len(),
            self.inner.num_detectors,
            self.inner.num_dem_outputs(),
            self.inner.num_observables(),
            self.inner.num_tracked_paulis()
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

    rust_verify_dem_equivalence(dem1, dem2, &comparison_method)
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
// CSS UF Decoder (UIUF)
// =============================================================================

/// CSS-aware Union-Find decoder using the UIUF algorithm.
///
/// Takes separate X and Z DEM strings and decodes them jointly, exploiting
/// Y-error identification through cluster intersection.
///
/// `Example::`
///
///     decoder = CssUfDecoder(x_dem_str, z_dem_str)
///     x_obs, z_obs = decoder.decode_css(x_syndrome, z_syndrome)
///
#[pyclass(name = "CssUfDecoder", module = "pecos_rslib.qec")]
pub struct PyCssUfDecoder {
    inner: pecos_decoders::CssUfDecoder,
}

#[pymethods]
impl PyCssUfDecoder {
    /// Create a CSS UF decoder from X and Z DEM strings.
    ///
    /// The qubit-edge mapping is auto-detected from detector coordinates.
    /// If coordinates are missing, falls back to independent X/Z decoding.
    #[new]
    fn new(x_dem: &str, z_dem: &str) -> PyResult<Self> {
        let inner = pecos_decoders::CssUfDecoder::from_dems(
            x_dem,
            z_dem,
            pecos_decoders::UfDecoderConfig::accurate(),
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { inner })
    }

    /// Decode X and Z syndromes jointly using UIUF.
    ///
    /// Args:
    ///     `x_syndrome`: X-basis detection events (bytes).
    ///     `z_syndrome`: Z-basis detection events (bytes).
    ///
    /// Returns:
    ///     Tuple of (`x_observable_mask`, `z_observable_mask`).
    fn decode_css(&mut self, x_syndrome: &[u8], z_syndrome: &[u8]) -> PyResult<(u64, u64)> {
        self.inner
            .decode_css(x_syndrome, z_syndrome)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Number of matched qubit pairs between X and Z graphs.
    /// 0 means no mapping was found (falls back to independent decode).
    #[getter]
    fn num_qubit_pairs(&self) -> usize {
        self.inner.num_qubit_pairs()
    }

    /// Count erasures the intersection would produce for given syndromes.
    fn count_erasures(&mut self, x_syndrome: &[u8], z_syndrome: &[u8]) -> usize {
        self.inner
            .count_intersection_erasures(x_syndrome, z_syndrome)
    }

    /// Decode a batch of syndromes and return the error count.
    ///
    /// Each shot has concatenated `[x_syndrome | z_syndrome]`.
    /// The `x_syndrome` length is specified by `x_num_detectors`.
    ///
    /// Args:
    ///     syndromes: List of concatenated syndrome byte arrays.
    ///     `true_obs_masks`: True observable masks for each shot.
    ///     `x_num_detectors`: Length of the X syndrome prefix.
    ///
    /// Returns:
    ///     Number of logical errors.
    fn decode_count_batch(
        &mut self,
        syndromes: Vec<Vec<u8>>,
        true_obs_masks: Vec<u64>,
        x_num_detectors: usize,
    ) -> PyResult<usize> {
        let mut errors = 0;
        for (syn, &true_obs) in syndromes.iter().zip(true_obs_masks.iter()) {
            let x_syn = &syn[..x_num_detectors.min(syn.len())];
            let z_syn = &syn[x_num_detectors.min(syn.len())..];
            let (x_obs, z_obs) = self
                .inner
                .decode_css(x_syn, z_syn)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let predicted = x_obs ^ z_obs;
            if predicted != true_obs {
                errors += 1;
            }
        }
        Ok(errors)
    }
}

// =============================================================================
// Observable Subgraph Decoder (Python class)
// =============================================================================

/// Per-observable subgraph decoder for transversal gates.
///
/// Partitions a DEM into per-observable graphlike subgraphs using
/// stabilizer coordinate information, then decodes each independently.
///
/// Args:
///     dem: DEM string with detector coordinate declarations.
///     `stab_coords`: List of dicts, one per logical qubit. Each dict has
///         keys "X" and "Z" mapping to lists of (x, y) ancilla coordinates.
///     `inner_decoder`: Inner decoder type string (default "`pecos_uf:fast`").
///
/// Example:
///     >>> decoder = `ObservableSubgraphDecoder`(
///     ...     `dem_str`,
///     ...     [{"X": [(1,0), (3,1)], "Z": [(0,3), (1,1)]}],
///     ...     "`pecos_uf:fast`",
///     ... )
///     >>> obs = decoder.decode(syndrome)
#[pyclass(name = "ObservableSubgraphDecoder", module = "pecos_rslib.qec")]
pub struct PyObservableSubgraphDecoder {
    inner: pecos_decoder_core::observable_subgraph::ObservableSubgraphDecoder,
}

#[pymethods]
impl PyObservableSubgraphDecoder {
    #[new]
    #[pyo3(signature = (dem, stab_coords, inner_decoder="pecos_uf:fast", max_time_radius=None))]
    fn new(
        dem: &str,
        stab_coords: Vec<pyo3::Bound<'_, pyo3::types::PyDict>>,
        inner_decoder: &str,
        max_time_radius: Option<i64>,
    ) -> PyResult<Self> {
        use pecos_decoder_core::observable_subgraph::{ObservableSubgraphDecoder, QubitStabCoords};

        // Parse stab_coords from Python dicts
        let mut rust_stab_coords = Vec::with_capacity(stab_coords.len());
        for dict in &stab_coords {
            let x_list: Vec<(f64, f64)> = dict
                .get_item("X")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'X' key"))?
                .extract()?;
            let z_list: Vec<(f64, f64)> = dict
                .get_item("Z")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Missing 'Z' key"))?
                .extract()?;
            rust_stab_coords.push(QubitStabCoords {
                x_positions: x_list,
                z_positions: z_list,
            });
        }

        let inner = ObservableSubgraphDecoder::from_dem_windowed(
            dem,
            &rust_stab_coords,
            max_time_radius,
            |subgraph| {
                let sub_dem = subgraph_to_dem_string(subgraph);
                let decoder = create_observable_decoder(&sub_dem, inner_decoder)
                    .map_err(|e| pecos_decoders::DecoderError::InternalError(e.to_string()))?;
                Ok(Box::new(SendWrapper(decoder))
                    as Box<dyn pecos_decoders::ObservableDecoder + Send + Sync>)
            },
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { inner })
    }

    /// Decode a syndrome and return observable flip predictions.
    fn decode(&mut self, syndrome: Vec<u8>) -> PyResult<u64> {
        use pecos_decoder_core::ObservableDecoder;
        self.inner
            .decode_to_observables(&syndrome)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Number of observables this decoder handles.
    fn num_observables(&self) -> usize {
        self.inner.num_observables()
    }

    /// Decode a batch of syndromes and return observable predictions.
    ///
    /// Args:
    ///     syndromes: 2D numpy array of shape (`num_shots`, `num_detectors`).
    ///
    /// Returns:
    ///     List of observable flip masks (one per shot).
    fn decode_batch(&mut self, syndromes: Vec<Vec<u8>>) -> PyResult<Vec<u64>> {
        use pecos_decoder_core::ObservableDecoder;
        let mut results = Vec::with_capacity(syndromes.len());
        for syn in &syndromes {
            let obs = self
                .inner
                .decode_to_observables(syn)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            results.push(obs);
        }
        Ok(results)
    }

    /// Decode a `SampleBatch` and return the number of logical errors.
    ///
    /// This runs entirely in Rust — no Python per-shot overhead.
    ///
    /// Args:
    ///     batch: A `SampleBatch` from `DemSampler.generate_samples()`.
    ///
    /// Returns:
    ///     Number of logical errors.
    fn decode_count(&mut self, batch: &PySampleBatch) -> PyResult<usize> {
        let detection_events: Vec<Vec<u8>> = (0..batch.num_shots)
            .map(|i| {
                let mut s = vec![0u8; batch.num_detectors];
                batch.extract_syndrome(i, &mut s);
                s
            })
            .collect();
        let observable_masks: Vec<u64> = (0..batch.num_shots)
            .map(|i| batch.extract_obs_mask(i))
            .collect();
        self.inner
            .decode_count_batched(&detection_events, &observable_masks)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode a `SampleBatch` in parallel using rayon.
    ///
    /// Creates per-worker decoder instances to avoid lock contention.
    /// Requires the DEM string and inner decoder type for reconstruction.
    #[pyo3(signature = (batch, dem, stab_coords, inner_decoder="pymatching", num_workers=None, max_time_radius=None))]
    fn decode_count_parallel(
        &self,
        batch: &PySampleBatch,
        dem: &str,
        stab_coords: Vec<pyo3::Bound<'_, pyo3::types::PyDict>>,
        inner_decoder: &str,
        num_workers: Option<usize>,
        max_time_radius: Option<i64>,
    ) -> PyResult<usize> {
        use pecos_decoder_core::observable_subgraph::{ObservableSubgraphDecoder, QubitStabCoords};
        use rayon::prelude::*;

        // Parse stab_coords
        let mut sc = Vec::with_capacity(stab_coords.len());
        for dict in &stab_coords {
            let x: Vec<(f64, f64)> = dict
                .get_item("X")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("X"))?
                .extract()?;
            let z: Vec<(f64, f64)> = dict
                .get_item("Z")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Z"))?
                .extract()?;
            sc.push(QubitStabCoords {
                x_positions: x,
                z_positions: z,
            });
        }

        let dem_str = dem.to_string();
        let inner_str = inner_decoder.to_string();
        let n = batch.num_shots;

        // Materialize row-major data for parallel decode.
        let events: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                let mut s = vec![0u8; batch.num_detectors];
                batch.extract_syndrome(i, &mut s);
                s
            })
            .collect();
        let masks: Vec<u64> = (0..n).map(|i| batch.extract_obs_mask(i)).collect();

        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_workers.unwrap_or(0))
            .build()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        let errors: usize = pool.install(|| {
            // Split into chunks, each chunk gets its own decoder + batch decode
            let chunk_size = n.div_ceil(rayon::current_num_threads());
            (0..n)
                .collect::<Vec<_>>()
                .par_chunks(chunk_size.max(1))
                .map(|chunk| {
                    // Build a fresh decoder for this worker
                    let mut dec = ObservableSubgraphDecoder::from_dem_windowed(
                        &dem_str,
                        &sc,
                        max_time_radius,
                        |subgraph| {
                            let sub_dem = subgraph_to_dem_string(subgraph);
                            let d =
                                create_observable_decoder(&sub_dem, &inner_str).map_err(|e| {
                                    pecos_decoders::DecoderError::InternalError(e.to_string())
                                })?;
                            Ok(Box::new(SendWrapper(d))
                                as Box<dyn pecos_decoders::ObservableDecoder + Send + Sync>)
                        },
                    )
                    .unwrap();

                    // Collect chunk syndromes and masks for batch decode
                    let chunk_syns: Vec<Vec<u8>> =
                        chunk.iter().map(|&i| events[i].clone()).collect();
                    let chunk_masks: Vec<u64> = chunk.iter().map(|&i| masks[i]).collect();
                    dec.decode_count_batched(&chunk_syns, &chunk_masks)
                        .unwrap_or(chunk.len())
                })
                .sum()
        });

        Ok(errors)
    }

    /// Number of detectors in each subgraph.
    fn subgraph_sizes(&self) -> Vec<usize> {
        (0..self.inner.num_observables())
            .map(|i| self.inner.subgraph(i).map_or(0, |sg| sg.detector_map.len()))
            .collect()
    }

    /// Diagnostics: (`num_edges`, `skipped_hyperedges`) for each subgraph.
    fn subgraph_diagnostics(&self) -> Vec<(usize, usize)> {
        (0..self.inner.num_observables())
            .map(|i| {
                self.inner.subgraph(i).map_or((0, 0), |sg| {
                    (sg.graph.edges.len(), sg.graph.skipped_hyperedges)
                })
            })
            .collect()
    }

    /// Count ghost edges (3-detector cross-qubit hyperedges) in the DEM.
    ///
    /// These are the hyperedges that the ghost protocol decomposes for
    /// modular per-qubit decoding. Returns (`total_ghost_edges`, `num_qubits`).
    #[staticmethod]
    fn count_ghost_edges(
        dem: &str,
        stab_coords: Vec<pyo3::Bound<'_, pyo3::types::PyDict>>,
    ) -> PyResult<(usize, usize)> {
        use pecos_decoder_core::ghost_protocol::extract_ghost_edges_from_dem;
        use pecos_decoder_core::observable_subgraph::QubitStabCoords;

        let mut sc = Vec::with_capacity(stab_coords.len());
        for dict in &stab_coords {
            let x: Vec<(f64, f64)> = dict
                .get_item("X")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("X"))?
                .extract()?;
            let z: Vec<(f64, f64)> = dict
                .get_item("Z")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Z"))?
                .extract()?;
            sc.push(QubitStabCoords {
                x_positions: x,
                z_positions: z,
            });
        }

        let edges = extract_ghost_edges_from_dem(dem, &sc);
        let num_qubits = sc.len();
        Ok((edges.len(), num_qubits))
    }

    /// Get the per-subgraph DEM strings (graphlike, suitable for windowed decoding).
    ///
    /// Each string is a DEM with local detector IDs (0..N) that can be
    /// passed to windowed or sandwich decoders.
    fn subgraph_dems(&self) -> Vec<String> {
        (0..self.inner.num_observables())
            .map(|i| {
                self.inner
                    .subgraph(i)
                    .map_or(String::new(), |sg| subgraph_to_dem_string(&sg.graph))
            })
            .collect()
    }

    /// Get the detector map for each subgraph (local → global index mapping).
    fn subgraph_detector_maps(&self) -> Vec<Vec<usize>> {
        (0..self.inner.num_observables())
            .map(|i| {
                self.inner
                    .subgraph(i)
                    .map_or(Vec::new(), |sg| sg.detector_map.clone())
            })
            .collect()
    }
}

// =============================================================================
// Windowed OSD Decoder (Python class)
// =============================================================================

/// Windowed observable subgraph decoder for deep circuits.
///
/// Splits the DEM into time windows, runs OSD within each window.
/// Prevents the observing region from spanning the full circuit.
///
/// Args:
///     dem: DEM string.
///     `stab_coords`: Stabilizer coordinates per logical qubit.
///     `inner_decoder`: Inner MWPM decoder type.
///     step: Core window size in time steps.
///     buffer: Buffer size on each side (0 = non-overlapping).
#[pyclass(name = "WindowedOsdDecoder", module = "pecos_rslib.qec")]
pub struct PyWindowedOsdDecoder {
    inner: pecos_decoder_core::windowed_osd::WindowedOsdDecoder,
}

#[pymethods]
impl PyWindowedOsdDecoder {
    #[new]
    #[pyo3(signature = (dem, stab_coords, inner_decoder="pymatching", step=8, buffer=4))]
    fn new(
        dem: &str,
        stab_coords: Vec<pyo3::Bound<'_, pyo3::types::PyDict>>,
        inner_decoder: &str,
        step: usize,
        buffer: usize,
    ) -> PyResult<Self> {
        use pecos_decoder_core::observable_subgraph::QubitStabCoords;
        use pecos_decoder_core::windowed_osd::{WindowedOsdConfig, WindowedOsdDecoder};

        let mut sc = Vec::with_capacity(stab_coords.len());
        for dict in &stab_coords {
            let x: Vec<(f64, f64)> = dict
                .get_item("X")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("X"))?
                .extract()?;
            let z: Vec<(f64, f64)> = dict
                .get_item("Z")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Z"))?
                .extract()?;
            sc.push(QubitStabCoords {
                x_positions: x,
                z_positions: z,
            });
        }

        let config = WindowedOsdConfig { step, buffer };

        let inner = WindowedOsdDecoder::from_dem(dem, &sc, &config, |subgraph| {
            let sub_dem = subgraph_to_dem_string(subgraph);
            let d = create_observable_decoder(&sub_dem, inner_decoder)
                .map_err(|e| pecos_decoders::DecoderError::InternalError(e.to_string()))?;
            Ok(Box::new(SendWrapper(d))
                as Box<dyn pecos_decoders::ObservableDecoder + Send + Sync>)
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        Ok(Self { inner })
    }

    fn decode(&mut self, syndrome: Vec<u8>) -> PyResult<u64> {
        use pecos_decoder_core::ObservableDecoder;
        self.inner
            .decode_to_observables(&syndrome)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    fn decode_count(&mut self, batch: &PySampleBatch) -> PyResult<usize> {
        use pecos_decoder_core::ObservableDecoder;
        let mut errors = 0usize;
        let mut syndrome = vec![0u8; batch.num_detectors];
        for i in 0..batch.num_shots {
            batch.extract_syndrome(i, &mut syndrome);
            let predicted = self
                .inner
                .decode_to_observables(&syndrome)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            if predicted != batch.extract_obs_mask(i) {
                errors += 1;
            }
        }
        Ok(errors)
    }

    fn num_windows(&self) -> usize {
        self.inner.windows.len()
    }
}

// =============================================================================
// Logical Algorithm Decoder (Python class)
// =============================================================================

/// Decoder for logical quantum algorithms with per-segment OSD and
/// Pauli frame propagation at transversal gate boundaries.
///
/// Built from a descriptor dict produced by
/// ``LogicalCircuitBuilder.build_algorithm_descriptor()``.
///
/// Supports both batch mode (``decode``, ``decode_count``) and
/// streaming mode (``feed_sparse``, ``flush``, ``reset``).
#[pyclass(name = "LogicalAlgorithmDecoder", module = "pecos_rslib.qec")]
pub struct PyLogicalAlgorithmDecoder {
    inner: pecos_decoder_core::logical_algorithm::StreamingLogicalDecoder,
}

#[pymethods]
impl PyLogicalAlgorithmDecoder {
    /// Build from a descriptor dict and inner decoder type.
    ///
    /// Args:
    ///     descriptor: Dict from ``LogicalCircuitBuilder.build_algorithm_descriptor()``.
    ///     `inner_decoder`: Decoder type string for each segment's OSD inner decoder.
    #[new]
    #[pyo3(signature = (descriptor, inner_decoder="pymatching"))]
    fn new(
        descriptor: &pyo3::Bound<'_, pyo3::types::PyDict>,
        inner_decoder: &str,
    ) -> PyResult<Self> {
        use pecos_decoder_core::logical_algorithm::{
            AlgorithmDescriptor, BoundaryGate, LogicalAlgorithmDecoder, SegmentDescriptor,
        };
        use pecos_decoder_core::observable_subgraph::{ObservableSubgraphDecoder, QubitStabCoords};

        // Parse full DEM and stab_coords for full-circuit OSD
        let full_dem: String = descriptor
            .get_item("full_dem")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("full_dem"))?
            .extract()?;

        // Use first segment's stab_coords as the base (they have the
        // original X/Z assignment; the full-circuit DEM uses original coords).
        let seg_list: Vec<pyo3::Bound<'_, pyo3::types::PyDict>> = descriptor
            .get_item("segments")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("segments"))?
            .extract()?;

        let num_obs: usize = descriptor
            .get_item("num_observables")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("num_observables"))?
            .extract()?;

        // Parse stab_coords from the first segment (original orientation)
        let first_seg = &seg_list[0];
        let sc_list: Vec<pyo3::Bound<'_, pyo3::types::PyDict>> = first_seg
            .get_item("stab_coords")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("stab_coords"))?
            .extract()?;
        let mut rust_sc = Vec::with_capacity(sc_list.len());
        for sc_dict in &sc_list {
            let x: Vec<(f64, f64)> = sc_dict
                .get_item("X")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("X"))?
                .extract()?;
            let z: Vec<(f64, f64)> = sc_dict
                .get_item("Z")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Z"))?
                .extract()?;
            rust_sc.push(QubitStabCoords {
                x_positions: x,
                z_positions: z,
            });
        }

        let inner_str = inner_decoder.to_string();

        // Build full-circuit OSD from the full DEM
        let full_osd = ObservableSubgraphDecoder::from_dem(&full_dem, &rust_sc, |subgraph| {
            let sub_dem = subgraph_to_dem_string(subgraph);
            let d = create_observable_decoder(&sub_dem, &inner_str)
                .map_err(|e| pecos_decoders::DecoderError::InternalError(e.to_string()))?;
            Ok(Box::new(SendWrapper(d))
                as Box<dyn pecos_decoders::ObservableDecoder + Send + Sync>)
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Parse segment descriptors (for metadata)
        let mut seg_descs = Vec::with_capacity(seg_list.len());
        for seg_dict in &seg_list {
            let n_det: usize = seg_dict
                .get_item("num_detectors")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("num_detectors"))?
                .extract()?;
            seg_descs.push(SegmentDescriptor {
                num_detectors: n_det,
                num_observables: num_obs,
            });
        }

        // Parse boundary gates
        let bg_list: Vec<Vec<pyo3::Bound<'_, pyo3::types::PyDict>>> = descriptor
            .get_item("boundary_gates")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("boundary_gates"))?
            .extract()?;

        let mut boundary_gates = Vec::with_capacity(bg_list.len());
        for gates in &bg_list {
            let mut bg_vec = Vec::new();
            for gate_dict in gates {
                let gate_type: String = gate_dict
                    .get_item("type")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("type"))?
                    .extract()?;
                match gate_type.as_str() {
                    "Hadamard" => {
                        let x: u32 = gate_dict.get_item("x_obs_bit")?.unwrap().extract()?;
                        let z: u32 = gate_dict.get_item("z_obs_bit")?.unwrap().extract()?;
                        bg_vec.push(BoundaryGate::Hadamard {
                            x_obs_bit: x,
                            z_obs_bit: z,
                        });
                    }
                    "Cnot" => {
                        bg_vec.push(BoundaryGate::Cnot {
                            ctrl_x_bit: gate_dict.get_item("ctrl_x_bit")?.unwrap().extract()?,
                            ctrl_z_bit: gate_dict.get_item("ctrl_z_bit")?.unwrap().extract()?,
                            tgt_x_bit: gate_dict.get_item("tgt_x_bit")?.unwrap().extract()?,
                            tgt_z_bit: gate_dict.get_item("tgt_z_bit")?.unwrap().extract()?,
                        });
                    }
                    "SGate" => {
                        let x: u32 = gate_dict.get_item("x_obs_bit")?.unwrap().extract()?;
                        let z: u32 = gate_dict.get_item("z_obs_bit")?.unwrap().extract()?;
                        bg_vec.push(BoundaryGate::SGate {
                            x_obs_bit: x,
                            z_obs_bit: z,
                        });
                    }
                    "TGateInjection" => {
                        let z: u32 = gate_dict.get_item("z_obs_bit")?.unwrap().extract()?;
                        let a: u32 = gate_dict.get_item("ancilla_z_bit")?.unwrap().extract()?;
                        bg_vec.push(BoundaryGate::TGateInjection {
                            z_obs_bit: z,
                            ancilla_z_bit: a,
                        });
                    }
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Unknown gate type: {gate_type}"
                        )));
                    }
                }
            }
            boundary_gates.push(bg_vec);
        }

        let algo_desc = AlgorithmDescriptor {
            segments: seg_descs,
            boundary_gates,
            num_observables: num_obs,
        };

        let algo_dec = LogicalAlgorithmDecoder::new(Box::new(full_osd), algo_desc);
        let inner = pecos_decoder_core::logical_algorithm::StreamingLogicalDecoder::new(algo_dec);
        Ok(Self { inner })
    }

    // -- Batch mode --

    /// Decode a single syndrome and return observable flip mask.
    fn decode(&mut self, syndrome: Vec<u8>) -> PyResult<u64> {
        self.inner.reset();
        self.inner
            .decode_shot(&syndrome)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode a batch of samples and count logical errors.
    fn decode_count(&mut self, batch: &PySampleBatch) -> PyResult<usize> {
        let detection_events: Vec<Vec<u8>> = (0..batch.num_shots)
            .map(|i| {
                let mut s = vec![0u8; batch.num_detectors];
                batch.extract_syndrome(i, &mut s);
                s
            })
            .collect();
        let observable_masks: Vec<u64> = (0..batch.num_shots)
            .map(|i| batch.extract_obs_mask(i))
            .collect();
        pecos_decoder_core::logical_algorithm::streaming_decode_count(
            &mut self.inner,
            &detection_events,
            &observable_masks,
        )
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    // -- Streaming mode --

    /// Feed sparse detection events: list of (`detector_index`, value) pairs.
    fn feed_sparse(&mut self, detectors: Vec<(u32, u8)>) {
        self.inner.feed_sparse(&detectors);
    }

    /// Feed a dense syndrome (all detectors in order).
    fn feed_dense(&mut self, syndrome: Vec<u8>) {
        self.inner.feed_dense(&syndrome);
    }

    /// Decode the accumulated syndrome. Call at segment boundaries or end.
    fn flush(&mut self) -> PyResult<u64> {
        self.inner
            .flush()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Reset syndrome buffer for the next shot.
    fn reset(&mut self) {
        self.inner.reset();
    }

    /// Current accumulated observable correction.
    fn accumulated_obs(&self) -> u64 {
        self.inner.accumulated_obs()
    }

    // -- Metadata --

    /// Number of segments.
    fn num_segments(&self) -> usize {
        self.inner.num_segments()
    }

    /// Rounds fed so far.
    fn rounds_fed(&self) -> usize {
        self.inner.rounds_fed()
    }
}

// =============================================================================
// Logical Circuit Decoder with Budget (Python class)
// =============================================================================

/// Budget-aware decoder for logical quantum circuits.
///
/// Selects decode strategy based on available reaction time:
/// - ``"unlimited"``: full-circuit OSD (Clifford circuits, offline)
/// - ``"windowed"``: default windowed OSD (~1ms reaction time)
/// - ``"10ms"``, ``"1000us"``, etc.: explicit reaction time budget
///
/// The reaction time is the time available at feed-forward decision
/// points (T gates, magic state injection). For Clifford-only circuits,
/// use ``"unlimited"`` since there are no mid-circuit decisions.
///
/// `Example::`
///
///     desc = builder.build_algorithm_descriptor(p1=0.001, p2=0.001)
///     decoder = LogicalCircuitDecoder(desc, budget="unlimited")
///     errors = decoder.decode_count(batch)
#[pyclass(name = "LogicalCircuitDecoder", module = "pecos_rslib.qec")]
pub struct PyLogicalCircuitDecoder {
    inner: pecos_decoder_core::logical_algorithm::LogicalCircuitDecoder,
}

#[pymethods]
impl PyLogicalCircuitDecoder {
    #[new]
    #[pyo3(signature = (descriptor, budget="unlimited", inner_decoder="pymatching"))]
    fn new(
        descriptor: &pyo3::Bound<'_, pyo3::types::PyDict>,
        budget: &str,
        inner_decoder: &str,
    ) -> PyResult<Self> {
        use pecos_decoder_core::decode_budget::DecodeBudget;
        use pecos_decoder_core::logical_algorithm::{
            AlgorithmDescriptor, BoundaryGate, FullCircuitStrategy, LogicalCircuitDecoder,
            SegmentDescriptor,
        };
        use pecos_decoder_core::observable_subgraph::{ObservableSubgraphDecoder, QubitStabCoords};

        // Parse full DEM
        let full_dem: String = descriptor
            .get_item("full_dem")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("full_dem"))?
            .extract()?;

        let seg_list: Vec<pyo3::Bound<'_, pyo3::types::PyDict>> = descriptor
            .get_item("segments")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("segments"))?
            .extract()?;

        let num_obs: usize = descriptor
            .get_item("num_observables")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("num_observables"))?
            .extract()?;

        // Parse stab_coords from first segment
        let first_seg = &seg_list[0];
        let sc_list: Vec<pyo3::Bound<'_, pyo3::types::PyDict>> = first_seg
            .get_item("stab_coords")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("stab_coords"))?
            .extract()?;
        let mut rust_sc = Vec::with_capacity(sc_list.len());
        for sc_dict in &sc_list {
            let x: Vec<(f64, f64)> = sc_dict
                .get_item("X")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("X"))?
                .extract()?;
            let z: Vec<(f64, f64)> = sc_dict
                .get_item("Z")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("Z"))?
                .extract()?;
            rust_sc.push(QubitStabCoords {
                x_positions: x,
                z_positions: z,
            });
        }
        let num_qubits = rust_sc.len();

        let inner_str = inner_decoder.to_string();
        let full_osd = ObservableSubgraphDecoder::from_dem(&full_dem, &rust_sc, |subgraph| {
            let sub_dem = subgraph_to_dem_string(subgraph);
            let d = create_observable_decoder(&sub_dem, &inner_str)
                .map_err(|e| pecos_decoders::DecoderError::InternalError(e.to_string()))?;
            Ok(Box::new(SendWrapper(d))
                as Box<dyn pecos_decoders::ObservableDecoder + Send + Sync>)
        })
        .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        // Parse segments
        let mut seg_descs = Vec::with_capacity(seg_list.len());
        for seg_dict in &seg_list {
            let n_det: usize = seg_dict
                .get_item("num_detectors")?
                .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("num_detectors"))?
                .extract()?;
            seg_descs.push(SegmentDescriptor {
                num_detectors: n_det,
                num_observables: num_obs,
            });
        }

        // Parse boundary gates
        let bg_list: Vec<Vec<pyo3::Bound<'_, pyo3::types::PyDict>>> = descriptor
            .get_item("boundary_gates")?
            .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("boundary_gates"))?
            .extract()?;

        let mut boundary_gates = Vec::with_capacity(bg_list.len());
        for gates in &bg_list {
            let mut bg_vec = Vec::new();
            for gate_dict in gates {
                let gate_type: String = gate_dict
                    .get_item("type")?
                    .ok_or_else(|| PyErr::new::<pyo3::exceptions::PyKeyError, _>("type"))?
                    .extract()?;
                match gate_type.as_str() {
                    "Hadamard" => {
                        bg_vec.push(BoundaryGate::Hadamard {
                            x_obs_bit: gate_dict.get_item("x_obs_bit")?.unwrap().extract()?,
                            z_obs_bit: gate_dict.get_item("z_obs_bit")?.unwrap().extract()?,
                        });
                    }
                    "Cnot" => {
                        bg_vec.push(BoundaryGate::Cnot {
                            ctrl_x_bit: gate_dict.get_item("ctrl_x_bit")?.unwrap().extract()?,
                            ctrl_z_bit: gate_dict.get_item("ctrl_z_bit")?.unwrap().extract()?,
                            tgt_x_bit: gate_dict.get_item("tgt_x_bit")?.unwrap().extract()?,
                            tgt_z_bit: gate_dict.get_item("tgt_z_bit")?.unwrap().extract()?,
                        });
                    }
                    "SGate" => {
                        bg_vec.push(BoundaryGate::SGate {
                            x_obs_bit: gate_dict.get_item("x_obs_bit")?.unwrap().extract()?,
                            z_obs_bit: gate_dict.get_item("z_obs_bit")?.unwrap().extract()?,
                        });
                    }
                    "TGateInjection" => {
                        let z: u32 = gate_dict.get_item("z_obs_bit")?.unwrap().extract()?;
                        let a: u32 = gate_dict.get_item("ancilla_z_bit")?.unwrap().extract()?;
                        bg_vec.push(BoundaryGate::TGateInjection {
                            z_obs_bit: z,
                            ancilla_z_bit: a,
                        });
                    }
                    _ => {
                        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                            "Unknown gate type: {gate_type}"
                        )));
                    }
                }
            }
            boundary_gates.push(bg_vec);
        }

        let algo_desc = AlgorithmDescriptor {
            segments: seg_descs,
            boundary_gates,
            num_observables: num_obs,
        };

        // Select budget: "unlimited" for full-circuit, "windowed" for
        // bounded-latency, or a cycle time in microseconds like "1000us".
        let mut distance = 0usize;
        while distance.saturating_mul(distance) < num_qubits {
            distance += 1;
        }
        let decode_budget = match budget {
            "unlimited" | "offline" => DecodeBudget::unlimited(),
            "windowed" => {
                DecodeBudget::from_reaction_time(std::time::Duration::from_millis(1), distance)
            }
            s if s.ends_with("us") => {
                let us: u64 = s[..s.len() - 2].parse().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid cycle time: {s}"
                    ))
                })?;
                DecodeBudget::from_reaction_time(std::time::Duration::from_micros(us), distance)
            }
            s if s.ends_with("ms") => {
                let ms: u64 = s[..s.len() - 2].parse().map_err(|_| {
                    PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                        "Invalid cycle time: {s}"
                    ))
                })?;
                DecodeBudget::from_reaction_time(std::time::Duration::from_millis(ms), distance)
            }
            _ => {
                return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(format!(
                    "Unknown budget: {budget}. Use: unlimited, windowed, or a cycle time like 1000us, 10ms"
                )));
            }
        };

        // Select strategy based on budget.
        let strategy: Box<dyn pecos_decoder_core::decode_budget::DecodeStrategy + Send + Sync> =
            if decode_budget.is_unlimited() {
                // Unlimited: full-circuit OSD (maximum accuracy)
                Box::new(FullCircuitStrategy::new(Box::new(full_osd)))
            } else {
                // Windowed: per-subgraph sandwich decoding.
                // Extract per-subgraph DEMs and detector maps from the full OSD.
                use pecos_decoder_core::logical_algorithm::WindowedOsdStrategy;

                let mut sub_dems = Vec::new();
                let mut det_maps = Vec::new();
                for i in 0..full_osd.num_observables() {
                    if let Some(sg) = full_osd.subgraph(i) {
                        sub_dems.push(subgraph_to_dem_string(&sg.graph));
                        det_maps.push(sg.detector_map.clone());
                    }
                }

                let d = decode_budget.code_distance;
                let buf = decode_budget.overlap_rounds.min(d * 2); // cap at 2d
                let windowed_str = if buf > 0 {
                    format!("windowed:step={d},buf={buf},wmax=2.5")
                } else {
                    // No overlap: use plain PM (faster, but accuracy limited
                    // to non-overlapping windowed matching)
                    format!("windowed:step={d},buf=0")
                };

                let wosd = WindowedOsdStrategy::new(sub_dems, det_maps, |dem_str| {
                    let dec = create_observable_decoder(dem_str, &windowed_str)
                        .map_err(|e| pecos_decoders::DecoderError::InternalError(e.to_string()))?;
                    Ok(Box::new(SendWrapper(dec))
                        as Box<dyn pecos_decoders::ObservableDecoder + Send + Sync>)
                })
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

                Box::new(wosd)
            };

        let inner = LogicalCircuitDecoder::new(algo_desc, strategy, decode_budget, num_qubits);
        Ok(Self { inner })
    }

    /// Decode a single syndrome.
    fn decode(&mut self, syndrome: Vec<u8>) -> PyResult<u64> {
        use pecos_decoder_core::ObservableDecoder;
        self.inner
            .decode_to_observables(&syndrome)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Decode a batch and count errors.
    fn decode_count(&mut self, batch: &PySampleBatch) -> PyResult<usize> {
        let detection_events: Vec<Vec<u8>> = (0..batch.num_shots)
            .map(|i| {
                let mut s = vec![0u8; batch.num_detectors];
                batch.extract_syndrome(i, &mut s);
                s
            })
            .collect();
        let observable_masks: Vec<u64> = (0..batch.num_shots)
            .map(|i| batch.extract_obs_mask(i))
            .collect();
        self.inner
            .decode_count(&detection_events, &observable_masks)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))
    }

    /// Number of segments.
    fn num_segments(&self) -> usize {
        self.inner.num_segments()
    }

    /// Total detectors.
    fn total_detectors(&self) -> usize {
        self.inner.total_detectors()
    }

    /// Whether the circuit has feed-forward decision points (T gates).
    /// If False, the reaction time budget doesn't matter — Clifford only.
    fn has_decision_points(&self) -> bool {
        self.inner.has_decision_points()
    }

    /// Number of decision points.
    fn num_decision_points(&self) -> usize {
        self.inner.num_decision_points()
    }

    /// Reset for next shot.
    fn reset(&mut self) {
        self.inner.reset();
    }
}

// =============================================================================
// Correlation Analysis Functions
// =============================================================================

/// Compute a detector flip frequency matrix from fired-detector lists.
///
/// Args:
///     fired_per_shot: List of lists, each inner list contains the detector
///         indices that fired in that shot (sorted ascending).
///     num_detectors: Total number of detectors.
///
/// Returns:
///     Flat list of length ``num_detectors^2`` (row-major). Diagonal entries
///     are marginal rates; off-diagonal ``M[i*n+j]`` = 0.5 * P(i AND j fire).
#[pyfunction]
#[pyo3(signature = (fired_per_shot, num_detectors))]
fn detector_flip_matrix(fired_per_shot: Vec<Vec<u32>>, num_detectors: usize) -> Vec<f64> {
    pecos_qec::fault_tolerance::correlation::flip_matrix_from_fired(&fired_per_shot, num_detectors)
}

/// Compute per-round detector flip frequency matrices.
///
/// Returns a list of flat matrices, one per round.
#[pyfunction]
#[pyo3(signature = (fired_per_shot, num_detectors, dets_per_round))]
fn detector_flip_matrices_by_round(
    fired_per_shot: Vec<Vec<u32>>,
    num_detectors: usize,
    dets_per_round: usize,
) -> Vec<Vec<f64>> {
    pecos_qec::fault_tolerance::correlation::flip_matrices_by_round(
        &fired_per_shot,
        num_detectors,
        dets_per_round,
    )
}

/// Compute k-body detector firing rates up to a given order.
///
/// Returns a list of ``(detector_indices, rate)`` pairs where
/// ``detector_indices`` is a tuple of sorted detector indices.
#[pyfunction]
#[pyo3(signature = (fired_per_shot, num_detectors, max_order=3))]
fn detector_k_body_rates(
    fired_per_shot: Vec<Vec<u32>>,
    num_detectors: usize,
    max_order: usize,
) -> Vec<(Vec<u32>, f64)> {
    pecos_qec::fault_tolerance::correlation::k_body_rates(&fired_per_shot, num_detectors, max_order)
        .into_iter()
        .collect()
}

/// Compute per-round k-body detector firing rates.
///
/// Returns a list (one per round) of lists of ``(local_indices, rate)`` pairs.
#[pyfunction]
#[pyo3(signature = (fired_per_shot, num_detectors, dets_per_round, max_order=3))]
fn detector_k_body_rates_by_round(
    fired_per_shot: Vec<Vec<u32>>,
    num_detectors: usize,
    dets_per_round: usize,
    max_order: usize,
) -> Vec<Vec<(Vec<u32>, f64)>> {
    pecos_qec::fault_tolerance::correlation::k_body_rates_by_round(
        &fired_per_shot,
        num_detectors,
        dets_per_round,
        max_order,
    )
    .into_iter()
    .map(|m| m.into_iter().collect())
    .collect()
}

/// Compare two flat flip matrices. Returns (max_rel_err, frob_rel_err, worst_i, worst_j).
#[pyfunction]
#[pyo3(signature = (sim, dem, num_detectors, min_rate=0.0005))]
fn compare_flip_matrices_rs(
    sim: Vec<f64>,
    dem: Vec<f64>,
    num_detectors: usize,
    min_rate: f64,
) -> (f64, f64, usize, usize) {
    pecos_qec::fault_tolerance::correlation::compare_flip_matrices(
        &sim,
        &dem,
        num_detectors,
        min_rate,
    )
}

/// Compare k-body rates grouped by order.
///
/// Args:
///     sim: List of ``(detector_indices, rate)`` from simulation.
///     dem: List of ``(detector_indices, rate)`` from DEM.
///     min_rate: Minimum rate to consider.
///
/// Returns:
///     List of ``(order, max_rel_err, rms_rel_err, worst_event)`` tuples.
#[pyfunction]
#[pyo3(signature = (sim, dem, min_rate=0.0005))]
fn compare_k_body_rates_rs(
    sim: Vec<(Vec<u32>, f64)>,
    dem: Vec<(Vec<u32>, f64)>,
    min_rate: f64,
) -> Vec<(usize, f64, f64, Vec<u32>)> {
    let sim_map: std::collections::BTreeMap<Vec<u32>, f64> = sim.into_iter().collect();
    let dem_map: std::collections::BTreeMap<Vec<u32>, f64> = dem.into_iter().collect();
    pecos_qec::fault_tolerance::correlation::compare_k_body(&sim_map, &dem_map, min_rate)
        .into_iter()
        .map(|(order, (me, rms, worst))| (order, me, rms, worst))
        .collect()
}

/// Fit DEM mechanism probabilities to match target detector marginals.
///
/// Takes the mechanism structure (from a stochastic DEM) and exact
/// per-detector marginals (from Heisenberg EEG), and adjusts mechanism
/// probabilities so the DEM reproduces those marginals.
///
/// Args:
///     mechanisms: List of ``(probability, detector_indices, observable_indices)``
///         from the stochastic DEM.
///     target_marginals: Exact per-detector rates from Heisenberg EEG.
///     max_iterations: Maximum fitting iterations (default 200).
///     tolerance: Convergence threshold (default 1e-12).
///
/// Returns:
///     Tuple of ``(fitted_mechanisms, residuals)`` where
///     ``fitted_mechanisms`` has the same format as input but with
///     adjusted probabilities, and ``residuals`` is the per-detector
///     absolute error after fitting.
#[pyfunction]
#[pyo3(signature = (mechanisms, target_marginals, max_iterations=200, tolerance=1e-12))]
fn fit_dem_to_marginals(
    mechanisms: Vec<PyDemMechanismTuple>,
    target_marginals: Vec<f64>,
    max_iterations: usize,
    tolerance: f64,
) -> PyDemFitResult {
    use pecos_qec::fault_tolerance::correlation::{
        DemMechanism, fit_dem_to_marginals as fit_inner,
    };

    let mechs: Vec<DemMechanism> = mechanisms
        .iter()
        .map(|(p, d, o)| DemMechanism {
            probability: *p,
            detectors: d.clone(),
            observables: o.clone(),
        })
        .collect();

    let (fitted, residuals) = fit_inner(&mechs, &target_marginals, max_iterations, tolerance);

    let result: Vec<(f64, Vec<u32>, Vec<u32>)> = fitted
        .iter()
        .map(|m| (m.probability, m.detectors.clone(), m.observables.clone()))
        .collect();

    (result, residuals)
}

/// Format DEM mechanisms as a standard DEM string.
#[pyfunction]
fn mechanisms_to_dem_string(mechanisms: Vec<(f64, Vec<u32>, Vec<u32>)>) -> String {
    use pecos_qec::fault_tolerance::correlation::{
        DemMechanism, mechanisms_to_dem_string as fmt_inner,
    };

    let mechs: Vec<DemMechanism> = mechanisms
        .iter()
        .map(|(p, d, o)| DemMechanism {
            probability: *p,
            detectors: d.clone(),
            observables: o.clone(),
        })
        .collect();

    fmt_inner(&mechs)
}

/// Query whether a decoder type requires decomposed (graphlike) DEMs.
///
/// Returns ``"graphlike"`` for MWPM decoders that need decomposed DEMs
/// (hyperedges cause errors), ``"any"`` for decoders that handle both
/// raw and decomposed DEMs.
///
/// Raises ``ValueError`` for unknown decoder types.
#[pyfunction]
fn decoder_dem_requirement(decoder_type: &str) -> PyResult<String> {
    let base = decoder_type.split(':').next().unwrap_or(decoder_type);
    match base {
        "pymatching"
        | "pymatching_uncorrelated"
        | "fusion_blossom"
        | "fusion_blossom_serial"
        | "fusion_blossom_parallel"
        | "fusion_blossom_correlated"
        | "pecos_uf"
        | "pecos_uf_correlated"
        | "windowed"
        | "k_mwpm"
        | "perturbed_fb_corr"
        | "perturbed_fb"
        | "ensemble" => Ok("graphlike".to_string()),
        "tesseract" | "astar" | "astar_full" | "bp_osd" | "bp_lsd" | "union_find"
        | "min_sum_bp" | "relay_bp" | "mwpf" | "chromobius" => Ok("any".to_string()),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown decoder type: {decoder_type:?}",
        ))),
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
    qec.add_class::<PySampleBatch>()?;
    qec.add_class::<PyCssUfDecoder>()?;
    qec.add_class::<PyObservableSubgraphDecoder>()?;
    qec.add_class::<PyWindowedOsdDecoder>()?;
    qec.add_class::<PyLogicalAlgorithmDecoder>()?;
    qec.add_class::<PyLogicalCircuitDecoder>()?;
    qec.add_class::<PyDecodeStats>()?;
    qec.add_class::<PyDemSampler>()?;
    qec.add_class::<PyDemSamplerBuilder>()?;
    qec.add_class::<PyEquivalenceResult>()?;
    qec.add_class::<PyParsedDem>()?;

    // Add DEM equivalence functions
    qec.add_function(wrap_pyfunction!(compare_dems_exact, &qec)?)?;
    qec.add_function(wrap_pyfunction!(compare_dems_statistical, &qec)?)?;
    qec.add_function(wrap_pyfunction!(verify_dem_equivalence, &qec)?)?;
    qec.add_function(wrap_pyfunction!(assert_dems_equivalent, &qec)?)?;

    // Correlation analysis
    qec.add_function(wrap_pyfunction!(detector_flip_matrix, &qec)?)?;
    qec.add_function(wrap_pyfunction!(detector_flip_matrices_by_round, &qec)?)?;
    qec.add_function(wrap_pyfunction!(detector_k_body_rates, &qec)?)?;
    qec.add_function(wrap_pyfunction!(detector_k_body_rates_by_round, &qec)?)?;
    qec.add_function(wrap_pyfunction!(compare_flip_matrices_rs, &qec)?)?;
    qec.add_function(wrap_pyfunction!(compare_k_body_rates_rs, &qec)?)?;
    qec.add_function(wrap_pyfunction!(fit_dem_to_marginals, &qec)?)?;
    qec.add_function(wrap_pyfunction!(mechanisms_to_dem_string, &qec)?)?;
    qec.add_function(wrap_pyfunction!(decoder_dem_requirement, &qec)?)?;

    // Add Pauli constants
    qec.add("PAULI_I", 0u8)?;
    qec.add("PAULI_X", 1u8)?;
    qec.add("PAULI_Y", 2u8)?;
    qec.add("PAULI_Z", 3u8)?;

    m.add_submodule(&qec)?;

    // Keep the common DEM sampler import available at the package root for
    // scripts that use `from pecos_rslib import DemSampler`.
    m.add("DemSampler", qec.getattr("DemSampler")?)?;

    // Register in sys.modules so 'from pecos_rslib.qec import ...' works
    let sys = m.py().import("sys")?;
    let modules = sys.getattr("modules")?;
    modules.set_item("pecos_rslib.qec", &qec)?;

    Ok(())
}
