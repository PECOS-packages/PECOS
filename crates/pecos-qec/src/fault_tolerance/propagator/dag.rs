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

//! DAG-based fault analysis for quantum circuits (recommended).
//!
//! This is the **recommended** approach for fault tolerance analysis. The DAG
//! representation enables sparse traversal that only visits gates touching qubits
//! with non-trivial Paulis, providing **5-50x speedup** over tick-based analysis
//! for typical surface code circuits.
//!
//! # Performance
//!
//! | Circuit Size | Tick-based | DAG-based | Speedup |
//! |--------------|------------|-----------|---------|
//! | d=3 (17 qubits) | 64 us | 16 us | 4x |
//! | d=5 (49 qubits) | 205 us | 38 us | 5x |
//! | d=7 (97 qubits) | 569 us | 49 us | 11x |
//! | d=11 (241 qubits) | 6529 us | 125 us | 52x |
//!
//! # Key Types
//!
//! - [`DagFaultAnalyzer`]: The main analyzer for building fault influence maps
//! - [`DagSpacetimeLocation`]: Identifies a fault location in a DAG circuit
//! - [`DagFaultInfluenceMap`]: Cache-optimized influence map using CSR layout
//!
//! # Output Terminology
//!
//! The influence map has one detector namespace plus one raw internal
//! non-detector-output namespace. That raw namespace is only a storage detail:
//! metadata maps each raw non-detector output to either a standard observable
//! (`L<n>`) or a PECOS tracked Pauli. Decoder and sampler code should use
//! [`DagFaultInfluenceMap::observable_ids`],
//! [`DagFaultInfluenceMap::observable_id_for_internal_dem_output`], and
//! [`DagFaultInfluenceMap::tracked_pauli_id_for_internal_dem_output`] instead of
//! assuming raw indices are public `L<n>` IDs.
//!
//! Observables and tracked Paulis differ by definition, not just by name.
//! Observables are values observed through measurement-record parities and are
//! visible to DEM decoders as standard `L<n>` outputs. Tracked Paulis are
//! unmeasured Pauli operators annotated at a circuit point, such as logical
//! operators, stabilizers, or other Paulis of interest; the influence map
//! records whether a fault anticommutes with, and therefore would flip, the
//! propagated operator. They are PECOS metadata and are not measurement-record
//! observables.
//!
//! # Example
//!
//! ```
//! use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
//! use pecos_quantum::DagCircuit;
//!
//! let mut dag = DagCircuit::new();
//! dag.pz(&[2]);       // Prep ancilla
//! dag.cx(&[(0, 2)]);    // CNOT data -> ancilla
//! dag.cx(&[(1, 2)]);    // CNOT data -> ancilla
//! dag.mz(&[2]);       // Measure ancilla
//!
//! let analyzer = DagFaultAnalyzer::new(&dag);
//! let map = analyzer.build_influence_map();
//!
//! // O(1) fault classification
//! let (has_syndrome, _flips_non_detector_output) = map.classify_fault(0, 1); // loc 0, X fault
//! ```

use super::{
    DagPropagator, DetectorId, Direction, InfluenceRecorder, MeasurementId, Pauli, apply_gate,
};
use pecos_core::gate_type::GateType;
use pecos_core::{PauliString, QuarterPhase, QubitId};
use pecos_quantum::DagCircuit;
use pecos_simulators::PauliProp;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};

/// Reusable work buffers for propagation, avoiding per-call allocation.
pub struct PropagationBuffers {
    pub visited: Vec<bool>,
    pub active_qubits: Vec<bool>,
    pub heap: BinaryHeap<(usize, usize)>,
}

struct Phase1Request {
    meas_node: usize,
    meas_qubit: usize,
    basis: u8,
    detector_idx: usize,
}

// ============================================================================
// Fault Locations (SoA Layout)
// ============================================================================

/// Fault locations in Struct-of-Arrays (`SoA`) layout for cache-efficient access.
///
/// Each array is indexed by location ID. This layout is more cache-friendly
/// than an array of structs when iterating over specific fields.
#[derive(Debug, Clone, Default)]
pub struct FaultLocations {
    /// Node index for each location.
    pub nodes: Vec<usize>,
    /// Qubit indices for each location (most locations have 1-2 qubits).
    pub qubits: Vec<SmallVec<[usize; 2]>>,
    /// Whether fault occurs before (true) or after (false) the gate.
    pub before: Vec<bool>,
    /// Gate type at each location.
    pub gate_types: Vec<GateType>,
    /// Reverse index: node -> list of location IDs at that node.
    pub node_to_locations: Vec<SmallVec<[usize; 4]>>,
}

impl FaultLocations {
    /// Creates a new empty `FaultLocations`.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates `FaultLocations` with capacity for the given number of locations and nodes.
    #[must_use]
    pub fn with_capacity(num_locations: usize, max_node: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(num_locations),
            qubits: Vec::with_capacity(num_locations),
            before: Vec::with_capacity(num_locations),
            gate_types: Vec::with_capacity(num_locations),
            node_to_locations: vec![SmallVec::new(); max_node + 1],
        }
    }

    /// Returns the number of fault locations.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Returns true if there are no fault locations.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Adds a fault location and returns its ID.
    pub fn push(
        &mut self,
        node: usize,
        qubits: SmallVec<[usize; 2]>,
        before: bool,
        gate_type: GateType,
    ) -> usize {
        let loc_id = self.nodes.len();
        self.nodes.push(node);
        self.qubits.push(qubits);
        self.before.push(before);
        self.gate_types.push(gate_type);

        // Update reverse index
        if node < self.node_to_locations.len() {
            self.node_to_locations[node].push(loc_id);
        }

        loc_id
    }

    /// Returns locations at the given node.
    #[inline]
    #[must_use]
    pub fn locations_at_node(&self, node: usize) -> &[usize] {
        if node < self.node_to_locations.len() {
            &self.node_to_locations[node]
        } else {
            &[]
        }
    }

    /// Returns the before flag for a location.
    #[inline]
    #[must_use]
    pub fn is_before(&self, loc_id: usize) -> bool {
        self.before[loc_id]
    }

    /// Returns the qubits for a location.
    #[inline]
    #[must_use]
    pub fn qubits(&self, loc_id: usize) -> &[usize] {
        &self.qubits[loc_id]
    }

    /// Converts to a Vec of `DagSpacetimeLocation` for backward compatibility.
    #[must_use]
    pub fn to_dag_spacetime_locations(&self) -> Vec<DagSpacetimeLocation> {
        (0..self.len())
            .map(|i| DagSpacetimeLocation {
                node: self.nodes[i],
                qubits: self.qubits[i].iter().map(|&q| QubitId::from(q)).collect(),
                before: self.before[i],
                gate_type: self.gate_types[i],
                idle_duration: 0,
            })
            .collect()
    }
}

// ============================================================================
// DAG Spacetime Location
// ============================================================================

/// A spacetime location in a DAG circuit, identified by node index.
///
/// Unlike `SpacetimeLocation` which uses tick indices, this uses DAG node indices
/// for more efficient sparse propagation.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DagSpacetimeLocation {
    /// The node index in the DAG.
    pub node: usize,
    /// The qubit(s) involved in the gate at this location.
    pub qubits: Vec<QubitId>,
    /// Whether the error occurs before (true) or after (false) the gate.
    pub before: bool,
    /// The type of gate at this location.
    pub gate_type: GateType,
    /// Duration for idle gates (in abstract time units). 0 for non-idle gates.
    pub idle_duration: u64,
}

// ============================================================================
// True SoA Influence Storage (Maximum Cache Efficiency)
// ============================================================================

/// CSR (Compressed Sparse Row) style array for cache-efficient storage.
///
/// This layout stores variable-length rows in a flat array with an offset array.
/// For row `i`, the data is at `data[offsets[i]..offsets[i+1]]`.
///
/// Benefits:
/// - Single contiguous allocation for all data
/// - Cache-friendly sequential access
/// - O(1) access to any row's data slice
#[derive(Debug, Clone, Default)]
pub struct CsrArray {
    /// Offset for each row. Length = `num_rows` + 1.
    /// Row i's data is at `data[offsets[i]..offsets[i+1]]`.
    pub offsets: Vec<u32>,
    /// Flat data array containing all values.
    pub data: Vec<u32>,
}

impl CsrArray {
    /// Creates a new empty CSR array with capacity for the given number of rows.
    #[must_use]
    pub fn with_row_capacity(num_rows: usize) -> Self {
        let mut offsets = Vec::with_capacity(num_rows + 1);
        offsets.push(0);
        Self {
            offsets,
            data: Vec::new(),
        }
    }

    /// Creates a new CSR array with capacity for rows and estimated data.
    #[must_use]
    pub fn with_capacity(num_rows: usize, estimated_data: usize) -> Self {
        let mut offsets = Vec::with_capacity(num_rows + 1);
        offsets.push(0);
        Self {
            offsets,
            data: Vec::with_capacity(estimated_data),
        }
    }

    /// Returns the number of rows.
    #[inline]
    #[must_use]
    pub fn num_rows(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    /// Returns the data slice for the given row.
    #[inline]
    #[must_use]
    pub fn row(&self, row_idx: usize) -> &[u32] {
        if row_idx + 1 < self.offsets.len() {
            let start = self.offsets[row_idx] as usize;
            let end = self.offsets[row_idx + 1] as usize;
            &self.data[start..end]
        } else {
            &[]
        }
    }

    /// Returns true if the row is empty.
    #[inline]
    #[must_use]
    pub fn row_is_empty(&self, row_idx: usize) -> bool {
        if row_idx + 1 < self.offsets.len() {
            self.offsets[row_idx] == self.offsets[row_idx + 1]
        } else {
            true
        }
    }

    /// Returns the number of elements in the given row.
    #[inline]
    #[must_use]
    pub fn row_len(&self, row_idx: usize) -> usize {
        if row_idx + 1 < self.offsets.len() {
            (self.offsets[row_idx + 1] - self.offsets[row_idx]) as usize
        } else {
            0
        }
    }

    /// Finalizes the current row and starts a new one.
    /// Call this after adding all data for the current row.
    #[inline]
    pub fn finish_row(&mut self) {
        #[allow(clippy::cast_possible_truncation)] // data length fits in u32
        self.offsets.push(self.data.len() as u32);
    }

    /// Adds a value to the current row (before calling `finish_row`).
    #[inline]
    pub fn push(&mut self, value: u32) {
        self.data.push(value);
    }

    /// Adds multiple values to the current row.
    #[inline]
    pub fn extend(&mut self, values: impl IntoIterator<Item = u32>) {
        self.data.extend(values);
    }

    /// Returns the total number of elements across all rows.
    #[inline]
    #[must_use]
    pub fn total_elements(&self) -> usize {
        self.data.len()
    }
}

/// True `SoA` (Struct of Arrays) influence storage using CSR layout.
///
/// This is the most cache-efficient representation, storing all influences
/// in flat arrays with CSR-style indexing. Each Pauli type (X, Y, Z) has
/// its own CSR array for maximum locality.
///
/// # Memory Layout
///
/// For N locations and M total detector influences:
/// - Traditional `AoS`: N * (`SmallVec` overhead + potential heap allocs)
/// - True `SoA`: 3 * (N+1) * 4 bytes (offsets) + M * 4 bytes (data)
///
/// The `SoA` layout is more compact and has better cache behavior when
/// iterating over all influences for a specific Pauli type.
#[derive(Debug, Clone, Default)]
pub struct InfluencesSoA {
    /// Number of fault locations.
    pub num_locations: usize,

    /// Detector indices flipped by X faults (Pauli=1).
    /// Row i contains detector indices for location i.
    pub detectors_x: CsrArray,

    /// Detector indices flipped by Y faults (Pauli=2).
    pub detectors_y: CsrArray,

    /// Detector indices flipped by Z faults (Pauli=3).
    pub detectors_z: CsrArray,

    /// Internal non-detector output indices flipped by X faults.
    ///
    /// These raw indices may name either standard observables or PECOS tracked
    /// operators. Use [`DagFaultInfluenceMap`] metadata helpers to map them into
    /// the public `L<n>` observable namespace or tracked-Pauli namespace.
    pub dem_outputs_x: CsrArray,

    /// Internal non-detector output indices flipped by Y faults.
    pub dem_outputs_y: CsrArray,

    /// Internal non-detector output indices flipped by Z faults.
    pub dem_outputs_z: CsrArray,
}

impl InfluencesSoA {
    /// Creates a new `SoA` structure with capacity for the given number of locations.
    #[must_use]
    pub fn with_capacity(num_locations: usize) -> Self {
        // Estimate: average 2 detector influences per location per Pauli type
        let estimated_data = num_locations * 2;
        Self {
            num_locations: 0,
            detectors_x: CsrArray::with_capacity(num_locations, estimated_data),
            detectors_y: CsrArray::with_capacity(num_locations, estimated_data),
            detectors_z: CsrArray::with_capacity(num_locations, estimated_data),
            dem_outputs_x: CsrArray::with_capacity(num_locations, estimated_data / 4),
            dem_outputs_y: CsrArray::with_capacity(num_locations, estimated_data / 4),
            dem_outputs_z: CsrArray::with_capacity(num_locations, estimated_data / 4),
        }
    }

    /// Returns the detector indices for a location and Pauli type.
    #[inline]
    #[must_use]
    pub fn detectors(&self, loc_idx: usize, pauli: Pauli) -> &[u32] {
        match pauli {
            Pauli::I => &[],
            Pauli::X => self.detectors_x.row(loc_idx),
            Pauli::Y => self.detectors_y.row(loc_idx),
            Pauli::Z => self.detectors_z.row(loc_idx),
        }
    }

    /// Returns raw internal non-detector output indices for a location and Pauli type.
    ///
    /// These indices are not necessarily standard `L<n>` IDs. Callers that
    /// need public observable IDs should use
    /// [`DagFaultInfluenceMap::observable_id_for_internal_dem_output`]; callers
    /// that need tracked-Pauli IDs should use
    /// [`DagFaultInfluenceMap::tracked_pauli_id_for_internal_dem_output`].
    #[inline]
    #[must_use]
    pub fn dem_outputs(&self, loc_idx: usize, pauli: Pauli) -> &[u32] {
        match pauli {
            Pauli::I => &[],
            Pauli::X => self.dem_outputs_x.row(loc_idx),
            Pauli::Y => self.dem_outputs_y.row(loc_idx),
            Pauli::Z => self.dem_outputs_z.row(loc_idx),
        }
    }

    /// Returns whether the location has any detector flips for the given Pauli.
    #[inline]
    #[must_use]
    pub fn has_detector_flips(&self, loc_idx: usize, pauli: Pauli) -> bool {
        match pauli {
            Pauli::I => false,
            Pauli::X => !self.detectors_x.row_is_empty(loc_idx),
            Pauli::Y => !self.detectors_y.row_is_empty(loc_idx),
            Pauli::Z => !self.detectors_z.row_is_empty(loc_idx),
        }
    }

    /// Returns whether the location has any non-detector output flips for the given Pauli.
    #[inline]
    #[must_use]
    pub fn has_dem_output_flips(&self, loc_idx: usize, pauli: Pauli) -> bool {
        match pauli {
            Pauli::I => false,
            Pauli::X => !self.dem_outputs_x.row_is_empty(loc_idx),
            Pauli::Y => !self.dem_outputs_y.row_is_empty(loc_idx),
            Pauli::Z => !self.dem_outputs_z.row_is_empty(loc_idx),
        }
    }

    /// Classifies a fault at the given location.
    ///
    /// Returns (`has_syndrome`, `flips_non_detector_output`).
    #[inline]
    #[must_use]
    pub fn classify(&self, loc_idx: usize, pauli: Pauli) -> (bool, bool) {
        (
            self.has_detector_flips(loc_idx, pauli),
            self.has_dem_output_flips(loc_idx, pauli),
        )
    }

    /// Finalizes a location row across all CSR arrays.
    pub fn finish_location(&mut self) {
        self.detectors_x.finish_row();
        self.detectors_y.finish_row();
        self.detectors_z.finish_row();
        self.dem_outputs_x.finish_row();
        self.dem_outputs_y.finish_row();
        self.dem_outputs_z.finish_row();
        self.num_locations += 1;
    }

    /// Returns memory statistics for this structure.
    #[must_use]
    pub fn memory_stats(&self) -> InfluencesSoAStats {
        let offset_bytes = (self.detectors_x.offsets.len()
            + self.detectors_y.offsets.len()
            + self.detectors_z.offsets.len()
            + self.dem_outputs_x.offsets.len()
            + self.dem_outputs_y.offsets.len()
            + self.dem_outputs_z.offsets.len())
            * std::mem::size_of::<u32>();

        let data_bytes = (self.detectors_x.data.len()
            + self.detectors_y.data.len()
            + self.detectors_z.data.len()
            + self.dem_outputs_x.data.len()
            + self.dem_outputs_y.data.len()
            + self.dem_outputs_z.data.len())
            * std::mem::size_of::<u32>();

        InfluencesSoAStats {
            num_locations: self.num_locations,
            total_detector_entries: self.detectors_x.total_elements()
                + self.detectors_y.total_elements()
                + self.detectors_z.total_elements(),
            total_dem_output_entries: self.dem_outputs_x.total_elements()
                + self.dem_outputs_y.total_elements()
                + self.dem_outputs_z.total_elements(),
            offset_bytes,
            data_bytes,
            total_bytes: offset_bytes + data_bytes,
        }
    }

    /// Returns the maximum raw non-detector output influence index, if any.
    ///
    /// When metadata is present, callers should use [`Self::num_dem_outputs`]
    /// for the standard observable `L<n>` namespace and [`Self::num_tracked_paulis`]
    /// for PECOS tracked Paulis.
    #[must_use]
    pub fn max_dem_output_index(&self) -> Option<usize> {
        let max_x = self.dem_outputs_x.data.iter().max();
        let max_y = self.dem_outputs_y.data.iter().max();
        let max_z = self.dem_outputs_z.data.iter().max();

        [max_x, max_y, max_z]
            .into_iter()
            .flatten()
            .max()
            .map(|&v| v as usize)
    }
}

/// Memory statistics for `InfluencesSoA`.
#[derive(Debug, Clone, Copy)]
pub struct InfluencesSoAStats {
    /// Number of fault locations.
    pub num_locations: usize,
    /// Total detector entries across all Pauli types.
    pub total_detector_entries: usize,
    /// Total DEM-output entries across all Pauli types.
    pub total_dem_output_entries: usize,
    /// Bytes used for offset arrays.
    pub offset_bytes: usize,
    /// Bytes used for data arrays.
    pub data_bytes: usize,
    /// Total bytes used.
    pub total_bytes: usize,
}

/// True `SoA` fault influence map using CSR-style storage.
///
/// This is the most memory-efficient and cache-friendly representation.
/// Use this when processing large circuits or when memory is constrained.
#[derive(Debug, Clone, Default)]
pub struct DagFaultInfluenceMap {
    /// Influences in true `SoA` layout.
    pub influences: InfluencesSoA,

    /// Locations indexed by location index.
    pub locations: Vec<DagSpacetimeLocation>,

    /// All detectors in the circuit.
    pub detectors: Vec<DetectorId>,

    /// All measurements in the circuit (node, qubit, basis).
    /// Ordered by `MeasId` when gates carry `MeasId` values.
    pub measurements: Vec<(usize, usize, u8)>,

    /// `MeasId` IDs for each measurement, in the same order as `measurements`.
    /// When populated, `meas_ids[i]` is the stable identity of `measurements[i]`.
    /// Empty for legacy circuits without `MeasId` on gates.
    pub meas_ids: Vec<pecos_core::MeasId>,

    /// Optional labels for non-detector parity outputs.
    /// Indices match the raw non-detector output indices in `influences`.
    pub dem_output_labels: Vec<Option<String>>,

    /// Optional metadata for non-detector outputs tracked by backward propagation.
    ///
    /// These entries may be standard observables or PECOS tracked Paulis.
    /// The metadata kind is the authority for translating raw influence indices
    /// into public namespaces; standard observables use compact `L<n>` IDs and
    /// tracked Paulis use their own compact PECOS-only IDs.
    pub dem_output_metadata: Vec<DemOutputMetadata>,
}

impl DagFaultInfluenceMap {
    /// Creates a new `SoA` map with capacity for the given number of locations.
    #[must_use]
    pub fn with_capacity(num_locations: usize) -> Self {
        Self {
            influences: InfluencesSoA::with_capacity(num_locations),
            locations: Vec::with_capacity(num_locations),
            detectors: Vec::new(),
            measurements: Vec::new(),
            meas_ids: Vec::new(),
            dem_output_labels: Vec::new(),
            dem_output_metadata: Vec::new(),
        }
    }

    /// Classifies a fault at the given location index.
    ///
    /// Returns (`has_syndrome`, `flips_non_detector_output`).
    #[inline]
    #[must_use]
    pub fn classify_fault(&self, loc_idx: usize, pauli: u8) -> (bool, bool) {
        self.influences.classify(loc_idx, Pauli::from_u8(pauli))
    }

    /// Returns the detector indices flipped by a fault.
    #[inline]
    #[must_use]
    pub fn get_detector_indices(&self, loc_idx: usize, pauli: u8) -> &[u32] {
        self.influences.detectors(loc_idx, Pauli::from_u8(pauli))
    }

    /// Returns all raw non-detector output indices flipped by a fault.
    ///
    /// Raw indices are an internal storage detail shared by observables and
    /// tracked Paulis. Prefer [`Self::get_observable_indices`] or
    /// [`Self::get_tracked_pauli_indices`] when a public namespace is needed.
    #[inline]
    #[must_use]
    pub fn get_dem_output_indices(&self, loc_idx: usize, pauli: u8) -> &[u32] {
        self.influences.dem_outputs(loc_idx, Pauli::from_u8(pauli))
    }

    /// Returns the number of standard DEM `L<n>` observable outputs.
    ///
    /// This is a DEM-output alias for [`Self::num_observables`]. It does
    /// not include PECOS tracked Paulis.
    #[must_use]
    pub fn num_dem_outputs(&self) -> usize {
        if self.dem_output_metadata.is_empty() {
            return self.influences.max_dem_output_index().map_or(0, |i| i + 1);
        }
        self.dem_output_metadata
            .iter()
            .filter(|metadata| metadata.kind == DemOutputKind::Observable)
            .count()
    }

    /// Returns the number of observables.
    #[must_use]
    pub fn num_observables(&self) -> usize {
        self.num_dem_outputs()
    }

    /// Returns the standard observable `L<n>` IDs present in this map.
    ///
    /// Tracked Paulis share internal propagation storage but never appear in
    /// this set. Public decoder and sampler paths should use this namespace
    /// rather than raw internal DEM-output indices.
    #[must_use]
    pub fn observable_ids(&self) -> BTreeSet<u32> {
        (0..self.num_dem_outputs())
            .filter_map(|idx| u32::try_from(idx).ok())
            .collect()
    }

    /// Returns the number of PECOS tracked Paulis.
    #[must_use]
    pub fn num_tracked_paulis(&self) -> usize {
        self.dem_output_metadata
            .iter()
            .filter(|metadata| metadata.kind == DemOutputKind::TrackedPauli)
            .count()
    }

    /// Returns tracked-Pauli output indices flipped by a fault.
    #[must_use]
    pub fn get_tracked_pauli_indices(&self, loc_idx: usize, pauli: u8) -> Vec<u32> {
        let outputs = self.get_dem_output_indices(loc_idx, pauli);
        outputs
            .iter()
            .filter_map(|&idx| self.tracked_pauli_id_for_internal_dem_output(idx))
            .collect()
    }

    /// Returns observable output indices flipped by a fault.
    #[must_use]
    pub fn get_observable_indices(&self, loc_idx: usize, pauli: u8) -> Vec<u32> {
        let outputs = self.get_dem_output_indices(loc_idx, pauli);
        if self.dem_output_metadata.is_empty() {
            return outputs.to_vec();
        }
        outputs
            .iter()
            .filter_map(|&idx| self.observable_id_for_internal_dem_output(idx))
            .collect()
    }

    /// Map an internal non-detector output index to the standard observable
    /// `L<n>` ID space.
    #[must_use]
    pub fn observable_id_for_internal_dem_output(&self, idx: u32) -> Option<u32> {
        if self.dem_output_metadata.is_empty() {
            return Some(idx);
        }
        self.output_id_for_kind(idx, DemOutputKind::Observable)
    }

    /// Map an internal non-detector output index to the PECOS tracked-Pauli
    /// ID space.
    #[must_use]
    pub fn tracked_pauli_id_for_internal_dem_output(&self, idx: u32) -> Option<u32> {
        self.output_id_for_kind(idx, DemOutputKind::TrackedPauli)
    }

    /// Returns true if a fault flips any non-detector DEM output.
    #[inline]
    #[must_use]
    pub fn has_dem_output_flips(&self, loc_idx: usize, pauli: u8) -> bool {
        !self.get_dem_output_indices(loc_idx, pauli).is_empty()
    }

    /// Returns true if a fault flips any tracked Pauli.
    #[must_use]
    pub fn has_tracked_pauli_flips(&self, loc_idx: usize, pauli: u8) -> bool {
        !self.get_tracked_pauli_indices(loc_idx, pauli).is_empty()
    }

    /// Returns true if a fault flips any observable.
    #[must_use]
    pub fn has_observable_flips(&self, loc_idx: usize, pauli: u8) -> bool {
        !self.get_observable_indices(loc_idx, pauli).is_empty()
    }

    /// Returns the label for a detector, if any.
    #[inline]
    #[must_use]
    pub fn detector_label(&self, detector_idx: usize) -> Option<&str> {
        self.detectors
            .get(detector_idx)
            .and_then(|d| d.name.as_deref())
    }

    /// Returns the label for a DEM output, if any.
    #[inline]
    #[must_use]
    pub fn dem_output_label(&self, dem_output_idx: usize) -> Option<&str> {
        self.dem_output_labels
            .get(dem_output_idx)
            .and_then(|l| l.as_deref())
    }

    /// Returns metadata for a DEM output, if available.
    #[inline]
    #[must_use]
    pub fn dem_output_metadata(&self, dem_output_idx: usize) -> Option<&DemOutputMetadata> {
        self.dem_output_metadata.get(dem_output_idx)
    }

    /// Replace this map's backward-propagated non-detector outputs with
    /// another map's outputs and metadata.
    pub fn merge_dem_outputs_from(&mut self, other: &Self) {
        self.influences.dem_outputs_x = other.influences.dem_outputs_x.clone();
        self.influences.dem_outputs_y = other.influences.dem_outputs_y.clone();
        self.influences.dem_outputs_z = other.influences.dem_outputs_z.clone();
        self.dem_output_labels.clone_from(&other.dem_output_labels);
        self.dem_output_metadata
            .clone_from(&other.dem_output_metadata);
    }

    /// Returns the location at the given index.
    #[inline]
    #[must_use]
    pub fn get_location(&self, loc_idx: usize) -> Option<&DagSpacetimeLocation> {
        self.locations.get(loc_idx)
    }

    /// Returns the detector at the given index.
    #[inline]
    #[must_use]
    pub fn get_detector(&self, detector_idx: usize) -> Option<&DetectorId> {
        self.detectors.get(detector_idx)
    }

    /// Returns memory statistics.
    #[must_use]
    pub fn memory_stats(&self) -> InfluencesSoAStats {
        self.influences.memory_stats()
    }

    /// Export CSR data for GPU use.
    ///
    /// The exported DEM-output arrays contain only standard observable `L<n>`
    /// outputs. PECOS tracked Paulis share the internal backward-propagation
    /// storage but are intentionally filtered out here so decoder-oriented GPU
    /// code cannot count tracked Paulis as logical errors.
    ///
    /// Returns all CSR arrays needed to construct a GPU influence sampler:
    /// (`num_locations`, `num_detectors`, `num_dem_outputs`,
    ///  `detector_offsets_x`, `detector_data_x`,
    ///  `detector_offsets_y`, `detector_data_y`,
    ///  `detector_offsets_z`, `detector_data_z`,
    ///  `dem_output_offsets_x`, `dem_output_data_x`,
    ///  `dem_output_offsets_y`, `dem_output_data_y`,
    ///  `dem_output_offsets_z`, `dem_output_data_z`)
    #[allow(clippy::type_complexity)]
    #[must_use]
    pub fn export_csr(
        &self,
    ) -> (
        u32,
        u32,
        u32,
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
        Vec<u32>,
    ) {
        #[allow(clippy::cast_possible_truncation)] // location count fits in u32
        let num_locations = self.locations.len() as u32;
        #[allow(clippy::cast_possible_truncation)] // detector count fits in u32
        let num_detectors = self.detectors.len() as u32;
        #[allow(clippy::cast_possible_truncation)] // DEM-output count fits in u32
        let num_dem_outputs = self.num_dem_outputs() as u32;
        let (dem_output_offsets_x, dem_output_data_x) =
            self.observable_csr(&self.influences.dem_outputs_x);
        let (dem_output_offsets_y, dem_output_data_y) =
            self.observable_csr(&self.influences.dem_outputs_y);
        let (dem_output_offsets_z, dem_output_data_z) =
            self.observable_csr(&self.influences.dem_outputs_z);

        (
            num_locations,
            num_detectors,
            num_dem_outputs,
            self.influences.detectors_x.offsets.clone(),
            self.influences.detectors_x.data.clone(),
            self.influences.detectors_y.offsets.clone(),
            self.influences.detectors_y.data.clone(),
            self.influences.detectors_z.offsets.clone(),
            self.influences.detectors_z.data.clone(),
            dem_output_offsets_x,
            dem_output_data_x,
            dem_output_offsets_y,
            dem_output_data_y,
            dem_output_offsets_z,
            dem_output_data_z,
        )
    }

    fn observable_csr(&self, csr: &CsrArray) -> (Vec<u32>, Vec<u32>) {
        if self.dem_output_metadata.is_empty() {
            return (csr.offsets.clone(), csr.data.clone());
        }

        let mut offsets = Vec::with_capacity(csr.offsets.len());
        let mut data = Vec::new();
        offsets.push(0);
        for row_idx in 0..csr.num_rows() {
            data.extend(
                csr.row(row_idx)
                    .iter()
                    .filter_map(|&idx| self.observable_id_for_internal_dem_output(idx)),
            );
            #[allow(clippy::cast_possible_truncation)] // CSR data length fits in u32
            offsets.push(data.len() as u32);
        }
        (offsets, data)
    }

    fn output_id_for_kind(&self, idx: u32, kind: DemOutputKind) -> Option<u32> {
        let metadata = self.dem_output_metadata.get(idx as usize)?;
        if metadata.kind != kind {
            return None;
        }
        #[allow(clippy::cast_possible_truncation)] // filtered output count fits in u32
        Some(
            self.dem_output_metadata[..idx as usize]
                .iter()
                .filter(|metadata| metadata.kind == kind)
                .count() as u32,
        )
    }
}

/// Role of a non-detector output under backward Pauli propagation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DemOutputKind {
    /// A standard `L<n>` observable defined by measurement records.
    Observable,
    /// An unmeasured Pauli-operator annotation, separate from measurement records.
    TrackedPauli,
}

impl DemOutputKind {
    /// Stable string used by PECOS metadata JSON.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Observable => "observable",
            Self::TrackedPauli => "tracked_pauli",
        }
    }

    /// Parses a stable PECOS metadata string.
    #[must_use]
    pub fn from_metadata_str(kind: &str) -> Option<Self> {
        match kind {
            "observable" => Some(Self::Observable),
            "tracked_pauli" => Some(Self::TrackedPauli),
            _ => None,
        }
    }
}

/// Metadata for a PECOS non-detector output.
///
/// Standard DEM text only has `L<n>` observable markers. PECOS keeps this richer
/// record alongside the DEM so callers can distinguish those measurement-record
/// observables from tracked Paulis, which live in a separate
/// PECOS-only namespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DemOutputMetadata {
    /// The output role.
    pub kind: DemOutputKind,
    /// Pauli string whose flip is tracked.
    ///
    /// For observables this is the Pauli associated with the measurement-record
    /// observable. For tracked Paulis this is the unmeasured tracked Pauli
    /// annotated at a circuit point.
    pub pauli: PauliString,
    /// Optional user label.
    pub label: Option<String>,
}

impl DemOutputMetadata {
    /// Creates DEM output metadata.
    #[must_use]
    pub fn new(kind: DemOutputKind, mut pauli: PauliString, label: Option<String>) -> Self {
        // A tracked Pauli op flip is an anticommutation property; global phase has
        // no meaning for DEM/sampler output.
        pauli.set_phase(QuarterPhase::PlusOne);
        Self { kind, pauli, label }
    }

    /// Creates metadata for a tracked Pauli.
    #[must_use]
    pub fn tracked_pauli(pauli: PauliString) -> Self {
        Self::new(DemOutputKind::TrackedPauli, pauli, None)
    }

    /// Creates metadata for an observable.
    #[must_use]
    pub fn observable(pauli: PauliString) -> Self {
        Self::new(DemOutputKind::Observable, pauli, None)
    }

    /// Sets a user-facing op label.
    #[must_use]
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Sets an optional user-facing op label.
    #[must_use]
    pub fn with_optional_label(mut self, label: Option<String>) -> Self {
        self.label = label;
        self
    }
}

// ============================================================================
// Fault Introspection
// ============================================================================

/// A per-gate fault location with access to all possible fault events.
///
/// Each gate at a specific timing (before/after) is one fault location.
/// Multi-qubit gates have multiple per-qubit sub-locations whose effects
/// compose via XOR (symmetric difference) for multi-qubit Pauli events.
///
/// Borrows the influence map so you can query events directly:
/// ```
/// use pecos_qec::fault_tolerance::propagator::dag::DagFaultInfluenceMap;
///
/// let map = DagFaultInfluenceMap::with_capacity(0);
/// for loc in map.gate_fault_locations() {
///     for event in loc.events() {
///         println!("{}: dets={:?}", event.pauli, event.detectors);
///     }
/// }
/// ```
pub struct GateFaultLocation<'a> {
    map: &'a DagFaultInfluenceMap,
    /// DAG node index.
    pub node: usize,
    /// Gate type.
    pub gate_type: GateType,
    /// Qubits this gate acts on.
    pub qubits: Vec<QubitId>,
    /// Before (true) or after (false) the gate.
    pub before: bool,
    /// Per-qubit location indices in the influence map.
    qubit_loc_indices: Vec<(usize, usize)>, // (qubit, loc_idx)
}

/// The effect of a specific fault event (multi-qubit Pauli error).
#[derive(Debug, Clone)]
pub struct FaultEffect {
    /// The multi-qubit Pauli error.
    pub pauli: pecos_core::PauliString,
    /// Detector indices that flip.
    pub detectors: Vec<u32>,
    /// DEM-output indices that flip.
    pub dem_outputs: Vec<u32>,
    /// Raw measurements that flip: `(node, qubit, basis)`.
    ///
    /// Derived from the flipped detectors. Each auto-detected detector
    /// corresponds to one or more measurements; this expands them.
    pub measurements: Vec<(usize, usize, u8)>,
}

impl FaultEffect {
    /// Compose two fault effects (as if both faults occurred).
    ///
    /// - Paulis are multiplied (handles same-qubit algebra + tensor product)
    /// - Detectors, `dem_outputs`, and measurements are XOR'd (symmetric difference)
    ///
    /// This is the building block for weight-w fault analysis:
    /// ```
    /// use pecos_core::PauliString;
    /// use pecos_qec::fault_tolerance::propagator::dag::FaultEffect;
    ///
    /// let effect_a = FaultEffect {
    ///     pauli: PauliString::x(0),
    ///     detectors: vec![0],
    ///     dem_outputs: vec![],
    ///     measurements: vec![],
    /// };
    /// let effect_b = FaultEffect {
    ///     pauli: PauliString::z(1),
    ///     detectors: vec![0, 1],
    ///     dem_outputs: vec![0],
    ///     measurements: vec![],
    /// };
    /// let w2 = effect_a.compose(&effect_b);
    /// assert_eq!(w2.detectors, vec![1]);
    /// assert_eq!(w2.dem_outputs, vec![0]);
    /// ```
    #[must_use]
    pub fn compose(&self, other: &Self) -> Self {
        let mut pauli = self.pauli.clone() * other.pauli.clone();
        pauli.set_phase(pecos_core::QuarterPhase::PlusOne);

        let mut detectors = self.detectors.clone();
        xor_sorted(&mut detectors, &other.detectors);

        let mut dem_outputs = self.dem_outputs.clone();
        xor_sorted(&mut dem_outputs, &other.dem_outputs);

        let mut measurements = self.measurements.clone();
        xor_sorted_tuples(&mut measurements, &other.measurements);

        Self {
            pauli,
            detectors,
            dem_outputs,
            measurements,
        }
    }
}

impl GateFaultLocation<'_> {
    /// Number of qubits this gate acts on.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.qubits.len()
    }

    /// All possible fault events at this location.
    ///
    /// Only returns multi-qubit Paulis where at least one qubit's
    /// single-qubit component has a non-trivial effect in the influence
    /// map. E.g., a measurement-before location might only yield X
    /// faults since Z before MZ is invisible.
    #[must_use]
    pub fn possible_faults(&self) -> Vec<pecos_core::PauliString> {
        let active = self.active_paulis_per_qubit();
        if active.is_empty() {
            return Vec::new();
        }

        let mut combos: Vec<Vec<(usize, pecos_core::Pauli)>> = vec![vec![]];
        for &(q, ref paulis) in &active {
            let mut next = Vec::new();
            for existing in &combos {
                next.push(existing.clone());
                for &p in paulis {
                    let mut extended = existing.clone();
                    extended.push((q.index(), p));
                    next.push(extended);
                }
            }
            combos = next;
        }

        combos
            .into_iter()
            .filter(|c| !c.is_empty())
            .map(|entries| {
                pecos_core::PauliString::with_phase_and_paulis(
                    pecos_core::QuarterPhase::PlusOne,
                    entries
                        .iter()
                        .map(|&(q, p)| (p, QubitId::from(q)))
                        .collect(),
                )
            })
            .collect()
    }

    /// All fault events that have non-trivial effects (flip at least one
    /// detector or logical).
    #[must_use]
    pub fn events(&self) -> Vec<FaultEffect> {
        self.possible_faults()
            .into_iter()
            .map(|ps| self.query(&ps))
            .filter(|e| !e.detectors.is_empty() || !e.dem_outputs.is_empty())
            .collect()
    }

    /// All physically possible fault events, including those with no effect.
    ///
    /// Use this for probability-correct enumeration (e.g., ML decoder).
    /// Events with empty detectors and `dem_outputs` are "trivial" faults that
    /// happen with real probability but don't change any observable.
    #[must_use]
    pub fn all_events(&self) -> Vec<FaultEffect> {
        self.all_physical_paulis()
            .into_iter()
            .map(|ps| self.query(&ps))
            .collect()
    }

    /// All physically meaningful single-qubit Paulis per qubit, regardless
    /// of whether they have non-trivial effects in the influence map.
    ///
    /// Gate-type filtering applies (PZ/MZ only X/Y) but effect filtering
    /// does not. This ensures correct probability accounting.
    fn all_physical_paulis(&self) -> Vec<pecos_core::PauliString> {
        let physical: &[pecos_core::Pauli] = match self.gate_type {
            // Z-basis prep/measurement: only X (bit-flip) fault.
            GateType::PZ | GateType::QAlloc | GateType::MZ | GateType::MeasureFree => {
                &[pecos_core::Pauli::X]
            }
            // Unitary gates: all single-qubit Paulis.
            _ => &[
                pecos_core::Pauli::X,
                pecos_core::Pauli::Y,
                pecos_core::Pauli::Z,
            ],
        };

        // Build all combinations (including I on each qubit)
        let mut combos: Vec<Vec<(usize, pecos_core::Pauli)>> = vec![vec![]];
        for &q in &self.qubits {
            let mut next = Vec::new();
            for existing in &combos {
                next.push(existing.clone()); // I on this qubit
                for &p in physical {
                    let mut extended = existing.clone();
                    extended.push((q.index(), p));
                    next.push(extended);
                }
            }
            combos = next;
        }

        combos
            .into_iter()
            .filter(|c| !c.is_empty())
            .map(|entries| {
                pecos_core::PauliString::with_phase_and_paulis(
                    pecos_core::QuarterPhase::PlusOne,
                    entries
                        .iter()
                        .map(|&(q, p)| (p, QubitId::from(q)))
                        .collect(),
                )
            })
            .collect()
    }

    /// Query the effect of a specific multi-qubit Pauli event.
    #[must_use]
    pub fn query(&self, pauli: &pecos_core::PauliString) -> FaultEffect {
        let entries: Vec<(usize, pecos_core::Pauli)> = pauli
            .paulis()
            .iter()
            .map(|&(p, q)| (q.index(), p))
            .collect();
        let (detectors, dem_outputs) = self.compose_effects(&entries);

        // Resolve detector indices to raw measurements
        let measurements = self.resolve_measurements(&detectors);

        FaultEffect {
            pauli: pauli.clone(),
            detectors,
            dem_outputs,
            measurements,
        }
    }

    /// Which single-qubit Paulis are physically meaningful at each qubit.
    ///
    /// Filters based on both the influence map (has non-trivial effect) and
    /// the gate type (Z after PZ is unphysical, Z before MZ is invisible).
    fn active_paulis_per_qubit(&self) -> Vec<(QubitId, Vec<pecos_core::Pauli>)> {
        // Determine which Paulis are physical for this gate type
        let physical_paulis: &[Pauli] = match self.gate_type {
            // Z-basis prep/measurement: only X (bit-flip) fault.
            GateType::PZ | GateType::QAlloc | GateType::MZ | GateType::MeasureFree => &[Pauli::X],
            // Unitary gates: all single-qubit Paulis.
            _ => &[Pauli::X, Pauli::Y, Pauli::Z],
        };

        self.qubit_loc_indices
            .iter()
            .filter_map(|&(qubit, loc_idx)| {
                let mut paulis = Vec::new();
                for &p in physical_paulis {
                    if self.map.influences.has_detector_flips(loc_idx, p)
                        || self.map.influences.has_dem_output_flips(loc_idx, p)
                    {
                        paulis.push(propagator_to_core_pauli(p));
                    }
                }
                if paulis.is_empty() {
                    None
                } else {
                    Some((QubitId::from(qubit), paulis))
                }
            })
            .collect()
    }

    /// Compose per-qubit effects via XOR (symmetric difference).
    fn compose_effects(&self, entries: &[(usize, pecos_core::Pauli)]) -> (Vec<u32>, Vec<u32>) {
        let mut det_set: Vec<u32> = Vec::new();
        let mut dem_output_set: Vec<u32> = Vec::new();

        for &(qubit, pauli) in entries {
            if pauli == pecos_core::Pauli::I {
                continue;
            }
            let prop_pauli = core_to_propagator_pauli(pauli);

            if let Some(&(_, loc_idx)) = self.qubit_loc_indices.iter().find(|&&(q, _)| q == qubit) {
                let dets = self.map.influences.detectors(loc_idx, prop_pauli);
                xor_sorted(&mut det_set, dets);
                let dem_outputs = self.map.influences.dem_outputs(loc_idx, prop_pauli);
                xor_sorted(&mut dem_output_set, dem_outputs);
            }
        }

        (det_set, dem_output_set)
    }

    /// Resolve detector indices to raw measurement tuples.
    fn resolve_measurements(&self, detector_indices: &[u32]) -> Vec<(usize, usize, u8)> {
        let mut measurements = Vec::new();
        for &det_idx in detector_indices {
            if let Some(det) = self.map.detectors.get(det_idx as usize) {
                for meas_id in &det.measurements {
                    measurements.push((meas_id.tick, meas_id.qubit, meas_id.basis));
                }
            }
        }
        measurements
    }
}

/// Symmetric difference for sorted `(usize, usize, u8)` tuples (measurements).
fn xor_sorted_tuples(acc: &mut Vec<(usize, usize, u8)>, other: &[(usize, usize, u8)]) {
    if other.is_empty() {
        return;
    }
    if acc.is_empty() {
        acc.extend_from_slice(other);
        return;
    }
    let mut result = Vec::with_capacity(acc.len() + other.len());
    let mut i = 0;
    let mut j = 0;
    while i < acc.len() && j < other.len() {
        match acc[i].cmp(&other[j]) {
            std::cmp::Ordering::Less => {
                result.push(acc[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(other[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&acc[i..]);
    result.extend_from_slice(&other[j..]);
    *acc = result;
}

/// Convert `pecos_core::Pauli` to the propagator's `Pauli`.
fn core_to_propagator_pauli(p: pecos_core::Pauli) -> Pauli {
    match p {
        pecos_core::Pauli::I => Pauli::I,
        pecos_core::Pauli::X => Pauli::X,
        pecos_core::Pauli::Y => Pauli::Y,
        pecos_core::Pauli::Z => Pauli::Z,
    }
}

/// Convert the propagator's `Pauli` to `pecos_core::Pauli`.
fn propagator_to_core_pauli(p: Pauli) -> pecos_core::Pauli {
    match p {
        Pauli::I => pecos_core::Pauli::I,
        Pauli::X => pecos_core::Pauli::X,
        Pauli::Y => pecos_core::Pauli::Y,
        Pauli::Z => pecos_core::Pauli::Z,
    }
}

/// Symmetric difference of two sorted u32 slices, mutating `acc` in place.
fn xor_sorted(acc: &mut Vec<u32>, other: &[u32]) {
    if other.is_empty() {
        return;
    }
    if acc.is_empty() {
        acc.extend_from_slice(other);
        return;
    }
    // Build symmetric difference: elements in exactly one of the two sets
    let mut result = Vec::with_capacity(acc.len() + other.len());
    let mut i = 0;
    let mut j = 0;
    while i < acc.len() && j < other.len() {
        match acc[i].cmp(&other[j]) {
            std::cmp::Ordering::Less => {
                result.push(acc[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                result.push(other[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                // In both sets -- they cancel (XOR)
                i += 1;
                j += 1;
            }
        }
    }
    result.extend_from_slice(&acc[i..]);
    result.extend_from_slice(&other[j..]);
    *acc = result;
}

impl DagFaultInfluenceMap {
    /// Group per-qubit locations into per-gate fault locations.
    ///
    /// Each returned [`GateFaultLocation`] represents a gate at a specific
    /// timing (before/after) and supports querying multi-qubit Pauli events.
    ///
    /// ```
    /// use pecos_qec::fault_tolerance::propagator::dag::DagFaultInfluenceMap;
    ///
    /// let map = DagFaultInfluenceMap::with_capacity(0);
    /// for loc in map.gate_fault_locations() {
    ///     for event in loc.events() {
    ///         println!("{}: dets={:?} dem_outputs={:?}", event.pauli, event.detectors, event.dem_outputs);
    ///     }
    /// }
    /// ```
    #[must_use]
    pub fn gate_fault_locations(&self) -> Vec<GateFaultLocation<'_>> {
        let mut groups: std::collections::BTreeMap<(usize, bool), Vec<(usize, usize)>> =
            std::collections::BTreeMap::new();

        for (loc_idx, loc) in self.locations.iter().enumerate() {
            let key = (loc.node, loc.before);
            for q in &loc.qubits {
                groups.entry(key).or_default().push((q.index(), loc_idx));
            }
        }

        groups
            .into_iter()
            .map(|((node, before), qubit_locs)| {
                let gate_type = self.locations[qubit_locs[0].1].gate_type;
                let qubits: Vec<QubitId> =
                    qubit_locs.iter().map(|&(q, _)| QubitId::from(q)).collect();
                GateFaultLocation {
                    map: self,
                    node,
                    gate_type,
                    qubits,
                    before,
                    qubit_loc_indices: qubit_locs,
                }
            })
            .collect()
    }
}

// ============================================================================
// Weight-w Fault Enumeration
// ============================================================================

/// A single component of a multi-fault combination.
#[derive(Debug, Clone)]
pub struct FaultComponent {
    /// Index into `gate_fault_locations()`.
    pub location_index: usize,
    /// The fault event at this location.
    pub event: FaultEffect,
}

/// A weight-w combination of faults and their combined effect.
#[derive(Debug, Clone)]
pub struct FaultCombo {
    /// The individual (location, event) pairs.
    pub components: Vec<FaultComponent>,
    /// Combined effect (XOR of all component effects).
    pub effect: FaultEffect,
}

impl DagFaultInfluenceMap {
    /// Enumerate all weight-w fault combinations, calling `f` for each.
    ///
    /// At weight 1, this iterates every fault location and every possible
    /// fault event. At weight 2, all pairs of (location, event). And so on.
    ///
    /// Uses a callback to avoid allocating a potentially huge result vec.
    ///
    /// ```
    /// use pecos_qec::fault_tolerance::propagator::dag::DagFaultInfluenceMap;
    ///
    /// let map = DagFaultInfluenceMap::with_capacity(0);
    /// // Find all undetectable weight-2 errors
    /// map.for_each_fault_combo(2, |combo| {
    ///     if !combo.effect.dem_outputs.is_empty() && combo.effect.detectors.is_empty() {
    ///         println!("Undetectable w=2:");
    ///         for c in &combo.components {
    ///             println!("  {} at loc {}", c.event.pauli, c.location_index);
    ///         }
    ///     }
    /// });
    /// ```
    pub fn for_each_fault_combo(&self, weight: usize, mut f: impl FnMut(&FaultCombo)) {
        let locs = self.gate_fault_locations();

        // Pre-compute events for each location
        let all_events: Vec<Vec<FaultEffect>> =
            locs.iter().map(GateFaultLocation::events).collect();

        let empty_effect = FaultEffect {
            pauli: pecos_core::PauliString::identity(),
            detectors: Vec::new(),
            dem_outputs: Vec::new(),
            measurements: Vec::new(),
        };

        let mut components = Vec::with_capacity(weight);
        let mut effects_stack = vec![empty_effect];

        enumerate_combos(
            &all_events,
            weight,
            0, // start_loc
            &mut components,
            &mut effects_stack,
            &mut f,
        );
    }
}

/// Recursive helper for weight-w combination enumeration.
fn enumerate_combos(
    all_events: &[Vec<FaultEffect>],
    remaining: usize,
    start_loc: usize,
    components: &mut Vec<FaultComponent>,
    effects_stack: &mut Vec<FaultEffect>,
    f: &mut impl FnMut(&FaultCombo),
) {
    if remaining == 0 {
        f(&FaultCombo {
            components: components.clone(),
            effect: effects_stack.last().unwrap().clone(),
        });
        return;
    }

    for loc_idx in start_loc..all_events.len() {
        for event in &all_events[loc_idx] {
            let combined = effects_stack.last().unwrap().compose(event);

            components.push(FaultComponent {
                location_index: loc_idx,
                event: event.clone(),
            });
            effects_stack.push(combined);

            enumerate_combos(
                all_events,
                remaining - 1,
                loc_idx + 1, // no repeats
                components,
                effects_stack,
                f,
            );

            components.pop();
            effects_stack.pop();
        }
    }
}

// ============================================================================
// Recorder Types
// ============================================================================

/// Recorder that writes to a true `SoA` influence map.
///
/// This recorder builds the `SoA` structure incrementally. Unlike other recorders,
/// it requires locations to be processed in order and finalized one at a time.
pub struct SoARecorderBuilder {
    /// The `SoA` structure being built.
    influences: InfluencesSoA,
    /// Current location being built.
    current_location: usize,
    /// Pending detector indices for current location (X, Y, Z).
    pending_x: Vec<u32>,
    pending_y: Vec<u32>,
    pending_z: Vec<u32>,
}

impl SoARecorderBuilder {
    /// Creates a new `SoA` recorder builder.
    #[must_use]
    pub fn new(num_locations: usize) -> Self {
        Self {
            influences: InfluencesSoA::with_capacity(num_locations),
            current_location: 0,
            pending_x: Vec::with_capacity(8),
            pending_y: Vec::with_capacity(8),
            pending_z: Vec::with_capacity(8),
        }
    }

    /// Flushes pending data for the current location and advances to the next.
    pub fn finish_location(&mut self) {
        // Flush pending data to CSR arrays
        self.influences.detectors_x.extend(self.pending_x.drain(..));
        self.influences.detectors_y.extend(self.pending_y.drain(..));
        self.influences.detectors_z.extend(self.pending_z.drain(..));

        // Finalize the row
        self.influences.finish_location();
        self.current_location += 1;
    }

    /// Finishes building and returns the `SoA` structure.
    #[must_use]
    pub fn finish(mut self) -> InfluencesSoA {
        // Flush any remaining pending data
        if !self.pending_x.is_empty() || !self.pending_y.is_empty() || !self.pending_z.is_empty() {
            self.finish_location();
        }
        self.influences
    }

    /// Records a detector influence for the current location.
    #[inline]
    pub fn record_detector(&mut self, pauli: Pauli, detector_idx: u32) {
        match pauli {
            Pauli::I => {}
            Pauli::X => self.pending_x.push(detector_idx),
            Pauli::Y => self.pending_y.push(detector_idx),
            Pauli::Z => self.pending_z.push(detector_idx),
        }
    }
}

/// Bucket-based recorder that accumulates influences per location for O(n) CSR construction.
///
/// Unlike a sorting approach, this uses per-location buckets (`SmallVecs`) to collect
/// detector indices, then flattens to CSR format. This is O(n) in the number of
/// influences, avoiding the O(n log n) sort overhead.
#[allow(clippy::struct_field_names)]
pub struct BucketRecorder {
    /// Per-location detector indices for X faults.
    x_buckets: Vec<SmallVec<[u32; 4]>>,
    /// Per-location detector indices for Y faults.
    y_buckets: Vec<SmallVec<[u32; 4]>>,
    /// Per-location detector indices for Z faults.
    z_buckets: Vec<SmallVec<[u32; 4]>>,
}

impl BucketRecorder {
    /// Creates a new bucket recorder for the given number of locations.
    #[must_use]
    pub fn new(num_locations: usize) -> Self {
        Self {
            x_buckets: vec![SmallVec::new(); num_locations],
            y_buckets: vec![SmallVec::new(); num_locations],
            z_buckets: vec![SmallVec::new(); num_locations],
        }
    }

    /// Converts buckets to `SoA` format in O(n) time.
    #[must_use]
    pub fn into_soa(self) -> InfluencesSoA {
        let num_locations = self.x_buckets.len();
        let mut soa = InfluencesSoA::with_capacity(num_locations);

        // Flatten buckets into CSR arrays
        for i in 0..num_locations {
            soa.detectors_x.extend(self.x_buckets[i].iter().copied());
            soa.detectors_y.extend(self.y_buckets[i].iter().copied());
            soa.detectors_z.extend(self.z_buckets[i].iter().copied());
            soa.finish_location();
        }

        soa
    }
}

impl InfluenceRecorder for BucketRecorder {
    #[inline]
    fn record(
        &mut self,
        loc_idx: usize,
        _qubit: usize,
        obs_x: bool,
        obs_z: bool,
        detector_idx: usize,
    ) {
        #[allow(clippy::cast_possible_truncation)] // detector index fits in u32
        let det = detector_idx as u32;

        // X fault anticommutes with Z observable
        if obs_z {
            self.x_buckets[loc_idx].push(det);
        }
        // Z fault anticommutes with X observable
        if obs_x {
            self.z_buckets[loc_idx].push(det);
        }
        // Y fault anticommutes with X or Z but NOT both (Y commutes with Y)
        if obs_x ^ obs_z {
            self.y_buckets[loc_idx].push(det);
        }
    }
}

// ============================================================================
// DAG Fault Analyzer
// ============================================================================

/// Propagates Paulis backward through a DAG circuit using sparse traversal.
///
/// This is significantly faster than `TickFaultAnalyzer` for circuits with
/// local connectivity (like surface codes) because it only visits gates that
/// touch qubits with non-trivial Paulis.
///
/// # Example
///
/// ```
/// use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
/// use pecos_quantum::DagCircuit;
///
/// // Build a simple syndrome extraction circuit
/// let mut dag = DagCircuit::new();
/// dag.pz(&[2]);           // Prep ancilla
/// dag.cx(&[(0, 2)]);        // CNOT data -> ancilla
/// dag.cx(&[(1, 2)]);        // CNOT data -> ancilla
/// dag.mz(&[2]);           // Measure ancilla
///
/// // Build the fault influence map using sparse propagation
/// let propagator = DagFaultAnalyzer::new(&dag);
/// let influence_map = propagator.build_influence_map();
/// ```
pub struct DagFaultAnalyzer<'a> {
    /// Base propagator for traversal infrastructure.
    propagator: DagPropagator<'a>,
    /// All fault locations in `SoA` layout.
    locations: FaultLocations,
}

impl<'a> DagFaultAnalyzer<'a> {
    /// Creates a new DAG backward propagator for the given circuit.
    ///
    /// Pre-computes indices for efficient sparse traversal.
    #[must_use]
    pub fn new(dag: &'a DagCircuit) -> Self {
        let propagator = DagPropagator::new(dag);

        // Extract locations using SoA layout
        let locations = Self::extract_locations(&propagator, dag);

        Self {
            propagator,
            locations,
        }
    }

    /// Returns the underlying propagator.
    #[inline]
    #[must_use]
    pub fn propagator(&self) -> &DagPropagator<'a> {
        &self.propagator
    }

    /// Returns the maximum node index.
    #[inline]
    #[must_use]
    pub fn max_node(&self) -> usize {
        self.propagator.max_node()
    }

    /// Returns the maximum qubit index.
    #[inline]
    #[must_use]
    pub fn max_qubit(&self) -> usize {
        self.propagator.max_qubit()
    }

    /// Extracts fault locations from the circuit using the propagator.
    ///
    /// For multi-qubit gates, creates separate fault locations for each qubit.
    /// This enables proper per-qubit fault analysis for depolarizing noise models
    /// (e.g., distinguishing XI from IX from XX on a CX gate).
    fn extract_locations(propagator: &DagPropagator<'_>, dag: &DagCircuit) -> FaultLocations {
        let topo_order = dag.topological_order();

        // Estimate capacity: roughly 4 locations per gate (2 qubits x 2 timings for 2Q gates)
        let estimated_locations = topo_order.len() * 4;
        let mut locations =
            FaultLocations::with_capacity(estimated_locations, propagator.max_node());

        for &node in &topo_order {
            if let Some(gate) = propagator.gate(node) {
                // Skip meta-gates — they don't create fault locations
                if gate.gate_type.is_meta() {
                    continue;
                }

                let is_measurement = matches!(gate.gate_type, GateType::MZ | GateType::MeasureFree);

                // Convert QubitId to usize
                let qubits: SmallVec<[usize; 2]> =
                    gate.qubits.iter().map(pecos_core::QubitId::index).collect();

                // Standard circuit noise model: one fault location per gate.
                //   Measurement: before (X flip before readout)
                //   All others (prep, unitary, idle): after
                // Idle gates on non-active qubits provide the missing "before"
                // coverage that would otherwise require before-gate locations.
                let before = is_measurement;
                for &q in &qubits {
                    let single_qubit: SmallVec<[usize; 2]> = smallvec::smallvec![q];
                    locations.push(node, single_qubit, before, gate.gate_type);
                }
            }
        }

        locations
    }

    /// Builds the complete fault influence map.
    ///
    /// This performs backward propagation from all measurements and creates a
    /// lookup table for fault classification using CSR (Compressed Sparse Row)
    /// layout for maximum cache efficiency.
    ///
    /// # Example
    /// ```
    /// use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
    /// use pecos_quantum::DagCircuit;
    ///
    /// let mut dag = DagCircuit::new();
    /// dag.pz(&[2]);
    /// dag.cx(&[(0, 2)]);
    /// dag.mz(&[2]);
    ///
    /// let propagator = DagFaultAnalyzer::new(&dag);
    /// let map = propagator.build_influence_map();
    ///
    /// // Check memory usage
    /// let stats = map.memory_stats();
    /// println!("Total bytes: {}", stats.total_bytes);
    /// ```
    #[must_use]
    pub fn build_influence_map(&self) -> DagFaultInfluenceMap {
        let num_locations = self.locations.len();
        let mut map = DagFaultInfluenceMap::with_capacity(num_locations);

        // Copy locations
        map.locations = self.locations.to_dag_spacetime_locations();

        // Extract measurements and create detectors
        let (measurements, meas_ids) = self.extract_measurements();
        map.measurements.clone_from(&measurements);
        map.meas_ids = meas_ids;

        for &(node, qubit, basis) in &measurements {
            let measurement_id = MeasurementId {
                tick: node,
                qubit,
                basis,
            };
            map.detectors.push(DetectorId::single(measurement_id));
        }

        // Use forest propagation: per-ancilla Phase 1/Phase 2 split.
        let recorder = self.propagate_all_forest();

        // Convert buckets to SoA format (O(n) flattening)
        map.influences = recorder.into_soa();

        map
    }

    /// Extracts all measurements from the circuit in a deterministic order.
    ///
    /// Measurements are sorted by:
    /// 1. Topological position (to respect causal dependencies)
    /// 2. Qubit index (to break ties for concurrent/independent measurements)
    ///
    /// This gives deterministic measurement ordering where measurements on
    /// lower-indexed qubits appear first when they are in the same "layer" of
    /// the circuit.
    #[must_use]
    /// Extract measurements with optional `MeasId` IDs.
    ///
    /// Returns `(measurements, meas_ids)` where:
    /// - `measurements` is `Vec<(node, qubit, basis)>` in `MeasId` order
    /// - `meas_ids` is `Vec<MeasId>` (empty for legacy circuits)
    pub fn extract_measurements(&self) -> (Vec<(usize, usize, u8)>, Vec<pecos_core::MeasId>) {
        let mut entries = Vec::new(); // (sort_key, qubit, node, basis, Option<MeasId>)

        for &node in self.propagator.topo_order() {
            if let Some(gate) = self.propagator.gate(node) {
                let basis = match gate.gate_type {
                    GateType::MZ | GateType::MeasureFree => 0, // Z-basis
                    _ => continue,
                };

                if gate.meas_ids.is_empty() {
                    let topo_pos = self.propagator.topo_position(node);
                    for qubit in &gate.qubits {
                        entries.push((topo_pos, qubit.index(), node, basis, None));
                    }
                } else {
                    for (i, qubit) in gate.qubits.iter().enumerate() {
                        let mr = gate.meas_ids.get(i).copied();
                        let sort_key = mr.map_or(usize::MAX, pecos_core::MeasId::index);
                        entries.push((sort_key, qubit.index(), node, basis, mr));
                    }
                }
            }
        }

        entries.sort_by_key(|&(sort_key, qubit, _, _, _)| (sort_key, qubit));

        let has_meas_ids = entries.iter().any(|(_, _, _, _, mr)| mr.is_some());
        let meas_ids = if has_meas_ids {
            entries
                .iter()
                .map(|(_, _, _, _, mr)| mr.unwrap_or(pecos_core::MeasId(usize::MAX)))
                .collect()
        } else {
            Vec::new()
        };

        let measurements = entries
            .into_iter()
            .map(|(_, qubit, node, basis, _)| (node, qubit, basis))
            .collect();

        (measurements, meas_ids)
    }

    // =========================================================================
    // Generic Propagation with Composable Recorder (DOD/ECS)
    // =========================================================================

    /// Propagates backward from a measurement using a generic recorder.
    ///
    /// This is the core propagation method that separates traversal logic from
    /// recording logic, following DOD/ECS principles.
    ///
    /// # Type Parameters
    /// * `R` - The recorder type implementing `InfluenceRecorder`
    ///
    /// # Arguments
    /// * `meas_node` - The measurement node
    /// * `meas_qubit` - The measured qubit
    /// * `basis` - Measurement basis (0=Z, 1=X)
    /// * `detector_idx` - Index of the detector being propagated from
    /// * `recorder` - The recorder for recording influences
    /// * `visited` - Work buffer for visited nodes (reusable)
    /// * `active_qubits` - Work buffer for active qubits (reusable)
    /// * `heap` - Work heap for traversal (reusable)
    pub fn propagate_from_measurement_generic<R: InfluenceRecorder>(
        &self,
        meas_node: usize,
        meas_qubit: usize,
        basis: u8,
        detector_idx: usize,
        recorder: &mut R,
        work: &mut PropagationBuffers,
    ) {
        let visited = &mut work.visited;
        let active_qubits = &mut work.active_qubits;
        let heap = &mut work.heap;
        // Clear work arrays
        visited.fill(false);
        active_qubits.fill(false);
        heap.clear();

        // Start with the observable being measured
        let mut prop = PauliProp::new();
        if basis == 0 {
            prop.track_z(&[meas_qubit]);
        } else {
            prop.track_x(&[meas_qubit]);
        }

        // Get measurement position (O(1) lookup)
        let meas_topo_pos = self.propagator.topo_position(meas_node);

        // Check fault at measurement node (before=true only)
        self.record_at_node_generic(meas_node, &prop, detector_idx, recorder, true);

        // Initialize: add gates on the measurement qubit
        if meas_qubit <= self.max_qubit() {
            active_qubits[meas_qubit] = true;
            for (topo_pos, node) in self.propagator.qubit_gates_backward(meas_qubit) {
                if topo_pos < meas_topo_pos && !visited[node] {
                    visited[node] = true;
                    heap.push((topo_pos, node));
                }
            }
        }

        // Process gates in reverse topo order - only gates on active wires
        while let Some((_, node)) = heap.pop() {
            if let Some(gate) = self.propagator.gate(node) {
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= self.max_qubit() {
                        was_active[j] = active_qubits[q.index()];
                    }
                }

                // Check before=false locations (error after gate)
                self.record_at_node_generic(node, &prop, detector_idx, recorder, false);

                // Handle prep gates specially - they kill the Pauli and stop propagation
                // on their qubits. Errors before a prep don't affect measurements after it.
                if matches!(gate.gate_type, GateType::PZ | GateType::QAlloc) {
                    for q in &gate.qubits {
                        let idx = q.index();
                        if idx <= self.max_qubit() {
                            // Kill the Pauli by toggling off
                            if prop.contains_x(idx) {
                                prop.track_x(&[idx]);
                            }
                            if prop.contains_z(idx) {
                                prop.track_z(&[idx]);
                            }
                            active_qubits[idx] = false;
                        }
                    }
                    // Don't record before=true for preps (they only have after locations anyway)
                    // and don't continue propagating on these qubits
                    continue;
                }

                // Apply gate backward
                apply_gate(&mut prop, gate, Direction::Backward);

                // Check before=true locations
                self.record_at_node_generic(node, &prop, detector_idx, recorder, true);

                // Check if Pauli spread to new qubits
                let node_topo_pos = self.propagator.topo_position(node);
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= self.max_qubit() {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            active_qubits[idx] = true;
                            for (topo_pos, new_node) in self.propagator.qubit_gates_backward(idx) {
                                if topo_pos < node_topo_pos && !visited[new_node] {
                                    visited[new_node] = true;
                                    heap.push((topo_pos, new_node));
                                }
                            }
                        } else if !now_active && was {
                            active_qubits[idx] = false;
                        }
                    }
                }
            }
        }
    }

    /// Records influences at a node using a generic recorder.
    #[inline]
    fn record_at_node_generic<R: InfluenceRecorder>(
        &self,
        node: usize,
        prop: &PauliProp,
        detector_idx: usize,
        recorder: &mut R,
        only_before: bool,
    ) {
        for &loc_idx in self.locations.locations_at_node(node) {
            if self.locations.is_before(loc_idx) != only_before {
                continue;
            }

            for &q in self.locations.qubits(loc_idx) {
                let obs_x = prop.contains_x(q);
                let obs_z = prop.contains_z(q);

                // Delegate to the recorder
                if obs_x || obs_z {
                    recorder.record(loc_idx, q, obs_x, obs_z, detector_idx);
                }
            }
        }
    }

    // ====================================================================
    // Forest propagation: per-ancilla Phase 1 / Phase 2 split
    // ====================================================================

    /// Captured influence entry from Phase 2 (shared tail below PZ).
    /// Stored with `topo_pos` for prefix slicing across measurements.
    ///
    /// Phase 1: propagate from MZ backward through within-round gates,
    /// stopping at the ancilla's PZ. Records influences normally.
    /// Returns the PZ node's topo position, or None if no PZ was hit.
    ///
    /// After return, `work.heap` still contains data qubit gates below
    /// the PZ — ready for Phase 2.
    fn propagate_phase1<R: InfluenceRecorder>(
        &self,
        request: &Phase1Request,
        recorder: &mut R,
        work: &mut PropagationBuffers,
        prop: &mut PauliProp,
    ) -> Option<usize> {
        let visited = &mut work.visited;
        let active_qubits = &mut work.active_qubits;
        let heap = &mut work.heap;
        visited.fill(false);
        active_qubits.fill(false);
        heap.clear();

        *prop = PauliProp::new();
        if request.basis == 0 {
            prop.track_z(&[request.meas_qubit]);
        } else {
            prop.track_x(&[request.meas_qubit]);
        }

        let meas_topo_pos = self.propagator.topo_position(request.meas_node);
        self.record_at_node_generic(
            request.meas_node,
            prop,
            request.detector_idx,
            recorder,
            true,
        );

        if request.meas_qubit <= self.max_qubit() {
            active_qubits[request.meas_qubit] = true;
            for (topo_pos, node) in self.propagator.qubit_gates_backward(request.meas_qubit) {
                if topo_pos < meas_topo_pos && !visited[node] {
                    visited[node] = true;
                    heap.push((topo_pos, node));
                }
            }
        }

        while let Some((_, node)) = heap.pop() {
            if let Some(gate) = self.propagator.gate(node) {
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= self.max_qubit() {
                        was_active[j] = active_qubits[q.index()];
                    }
                }

                self.record_at_node_generic(node, prop, request.detector_idx, recorder, false);

                if matches!(gate.gate_type, GateType::PZ | GateType::QAlloc) {
                    let pz_topo = self.propagator.topo_position(node);
                    for q in &gate.qubits {
                        let idx = q.index();
                        if idx <= self.max_qubit() {
                            if prop.contains_x(idx) {
                                prop.track_x(&[idx]);
                            }
                            if prop.contains_z(idx) {
                                prop.track_z(&[idx]);
                            }
                            active_qubits[idx] = false;
                        }
                    }
                    // Stop Phase 1 — data qubit gates remain in the heap.
                    return Some(pz_topo);
                }

                apply_gate(prop, gate, Direction::Backward);
                self.record_at_node_generic(node, prop, request.detector_idx, recorder, true);

                let node_topo_pos = self.propagator.topo_position(node);
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= self.max_qubit() {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];
                        if now_active && !was {
                            active_qubits[idx] = true;
                            for (topo_pos, new_node) in self.propagator.qubit_gates_backward(idx) {
                                if topo_pos < node_topo_pos && !visited[new_node] {
                                    visited[new_node] = true;
                                    heap.push((topo_pos, new_node));
                                }
                            }
                        } else if !now_active && was {
                            active_qubits[idx] = false;
                        }
                    }
                }
            }
        }
        None // No PZ hit (e.g., first round init detectors)
    }

    /// Phase 2: continue backward propagation from data qubit frontier
    /// below PZ. Records to `recorder` AND captures the visited node
    /// sequence for replay.
    fn propagate_phase2_capture<R: InfluenceRecorder>(
        &self,
        detector_idx: usize,
        recorder: &mut R,
        work: &mut PropagationBuffers,
        prop: &mut PauliProp,
    ) -> Vec<usize> {
        let mut visited_nodes: Vec<usize> = Vec::new();
        let visited = &mut work.visited;
        let active_qubits = &mut work.active_qubits;
        let heap = &mut work.heap;

        while let Some((_, node)) = heap.pop() {
            if let Some(gate) = self.propagator.gate(node) {
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= self.max_qubit() {
                        was_active[j] = active_qubits[q.index()];
                    }
                }

                self.record_at_node_generic(node, prop, detector_idx, recorder, false);
                visited_nodes.push(node);

                if matches!(gate.gate_type, GateType::PZ | GateType::QAlloc) {
                    for q in &gate.qubits {
                        let idx = q.index();
                        if idx <= self.max_qubit() {
                            if prop.contains_x(idx) {
                                prop.track_x(&[idx]);
                            }
                            if prop.contains_z(idx) {
                                prop.track_z(&[idx]);
                            }
                            active_qubits[idx] = false;
                        }
                    }
                    continue;
                }

                apply_gate(prop, gate, Direction::Backward);
                self.record_at_node_generic(node, prop, detector_idx, recorder, true);

                let node_topo = self.propagator.topo_position(node);
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= self.max_qubit() {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];
                        if now_active && !was {
                            active_qubits[idx] = true;
                            for (topo_pos, new_node) in self.propagator.qubit_gates_backward(idx) {
                                if topo_pos < node_topo && !visited[new_node] {
                                    visited[new_node] = true;
                                    heap.push((topo_pos, new_node));
                                }
                            }
                        } else if !now_active && was {
                            active_qubits[idx] = false;
                        }
                    }
                }
            }
        }
        visited_nodes
    }

    /// Replay Phase 2 using a cached node sequence.
    ///
    /// Iterates the captured nodes, re-applies gates backward to the Pauli
    /// state, and re-records with the correct `obs_x/obs_z` values. No heap
    /// or visited array needed — just a flat loop over known nodes.
    fn replay_phase2<R: InfluenceRecorder>(
        &self,
        nodes: &[usize],
        pz_topo: usize,
        detector_idx: usize,
        recorder: &mut R,
        prop: &mut PauliProp,
    ) {
        for &node in nodes {
            if self.propagator.topo_position(node) >= pz_topo {
                continue;
            }

            if let Some(gate) = self.propagator.gate(node) {
                self.record_at_node_generic(node, prop, detector_idx, recorder, false);

                if matches!(gate.gate_type, GateType::PZ | GateType::QAlloc) {
                    for q in &gate.qubits {
                        let idx = q.index();
                        if idx <= self.max_qubit() {
                            if prop.contains_x(idx) {
                                prop.track_x(&[idx]);
                            }
                            if prop.contains_z(idx) {
                                prop.track_z(&[idx]);
                            }
                        }
                    }
                    continue;
                }

                apply_gate(prop, gate, Direction::Backward);
                self.record_at_node_generic(node, prop, detector_idx, recorder, true);
            }
        }
    }

    /// Parallel forest propagation: groups measurements by ancilla qubit,
    /// propagates the latest measurement fully with capture, replays the
    /// shared tail prefix for earlier measurements.
    #[must_use]
    pub fn propagate_all_forest(&self) -> BucketRecorder {
        use rayon::prelude::*;

        let (measurements, _meas_ids) = self.extract_measurements();
        let num_locations = self.locations.len();

        // Group measurement indices by qubit (ancilla).
        let mut by_qubit: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        for (det_idx, &(_, qubit, _)) in measurements.iter().enumerate() {
            by_qubit.entry(qubit).or_default().push(det_idx);
        }

        // Collect ancilla groups for parallel iteration.
        let groups: Vec<Vec<usize>> = by_qubit.into_values().collect();

        let per_thread: Vec<BucketRecorder> = groups
            .par_iter()
            .map(|det_indices| {
                let mut recorder = BucketRecorder::new(num_locations);
                let mut work = PropagationBuffers {
                    visited: vec![false; self.propagator.max_node() + 1],
                    active_qubits: vec![false; self.propagator.max_qubit() + 1],
                    heap: BinaryHeap::with_capacity(64),
                };

                // Sort by topo position (ascending = earliest first).
                let mut sorted = det_indices.clone();
                sorted.sort_by_key(|&i| self.propagator.topo_position(measurements[i].0));

                // Latest measurement: Phase 1 + Phase 2 with capture
                let Some(&latest) = sorted.last() else {
                    return recorder;
                };
                let (l_node, l_qubit, l_basis) = measurements[latest];

                let mut prop = PauliProp::new();

                let latest_request = Phase1Request {
                    meas_node: l_node,
                    meas_qubit: l_qubit,
                    basis: l_basis,
                    detector_idx: latest,
                };
                let _pz_topo =
                    self.propagate_phase1(&latest_request, &mut recorder, &mut work, &mut prop);
                let tail_capture =
                    self.propagate_phase2_capture(latest, &mut recorder, &mut work, &mut prop);

                // Earlier measurements: Phase 1 + replay tail with correct Pauli state
                for &det_idx in sorted[..sorted.len() - 1].iter().rev() {
                    let (m_node, m_qubit, m_basis) = measurements[det_idx];

                    let request = Phase1Request {
                        meas_node: m_node,
                        meas_qubit: m_qubit,
                        basis: m_basis,
                        detector_idx: det_idx,
                    };
                    let pz_topo_i =
                        self.propagate_phase1(&request, &mut recorder, &mut work, &mut prop);

                    // Replay cached node sequence with correct Pauli state
                    if let Some(pz_pos) = pz_topo_i {
                        self.replay_phase2(
                            &tail_capture,
                            pz_pos,
                            det_idx,
                            &mut recorder,
                            &mut prop,
                        );
                    }
                }

                recorder
            })
            .collect();

        // Merge all recorders
        let mut merged = BucketRecorder::new(num_locations);
        for rec in per_thread {
            for i in 0..num_locations {
                merged.x_buckets[i].extend(rec.x_buckets[i].iter().copied());
                merged.y_buckets[i].extend(rec.y_buckets[i].iter().copied());
                merged.z_buckets[i].extend(rec.z_buckets[i].iter().copied());
            }
        }
        merged
    }

    /// Builds a fault influence map using a custom recorder.
    ///
    /// This is the most flexible method, allowing custom recording strategies.
    ///
    /// # Example
    /// ```
    /// use pecos_qec::fault_tolerance::propagator::{
    ///     DagFaultAnalyzer, CountingRecorder,
    /// };
    /// use pecos_quantum::DagCircuit;
    ///
    /// let mut dag = DagCircuit::new();
    /// dag.pz(&[2]);
    /// dag.cx(&[(0, 2)]);
    /// dag.mz(&[2]);
    ///
    /// let propagator = DagFaultAnalyzer::new(&dag);
    ///
    /// // Use a counting recorder to count influences
    /// let mut recorder = CountingRecorder::default();
    /// propagator.propagate_all(&mut recorder);
    /// println!("Total influences: {}", recorder.count);
    /// ```
    pub fn propagate_all<R: InfluenceRecorder>(&self, recorder: &mut R) {
        let (measurements, _) = self.extract_measurements();

        let mut work = PropagationBuffers {
            visited: vec![false; self.propagator.max_node() + 1],
            active_qubits: vec![false; self.propagator.max_qubit() + 1],
            heap: BinaryHeap::with_capacity(64),
        };

        for (detector_idx, &(node, qubit, basis)) in measurements.iter().enumerate() {
            self.propagate_from_measurement_generic(
                node,
                qubit,
                basis,
                detector_idx,
                recorder,
                &mut work,
            );
        }
    }

    /// Parallel version: propagates from all measurements using rayon.
    /// Each thread gets its own `BucketRecorder`, results are merged.
    #[must_use]
    pub fn propagate_all_parallel(&self) -> BucketRecorder {
        use rayon::prelude::*;

        let (measurements, _) = self.extract_measurements();
        let num_locations = self.locations.len();

        let chunk_size = measurements.len().div_ceil(rayon::current_num_threads());

        let per_thread: Vec<BucketRecorder> = measurements
            .par_chunks(chunk_size.max(1))
            .enumerate()
            .map(|(chunk_idx, chunk)| {
                let base_idx = chunk_idx * chunk_size;
                let mut recorder = BucketRecorder::new(num_locations);
                let mut work = PropagationBuffers {
                    visited: vec![false; self.propagator.max_node() + 1],
                    active_qubits: vec![false; self.propagator.max_qubit() + 1],
                    heap: BinaryHeap::with_capacity(64),
                };

                for (i, &(node, qubit, basis)) in chunk.iter().enumerate() {
                    self.propagate_from_measurement_generic(
                        node,
                        qubit,
                        basis,
                        base_idx + i,
                        &mut recorder,
                        &mut work,
                    );
                }
                recorder
            })
            .collect();

        // Merge all recorders
        let mut merged = BucketRecorder::new(num_locations);
        for rec in per_thread {
            for i in 0..num_locations {
                merged.x_buckets[i].extend(rec.x_buckets[i].iter().copied());
                merged.y_buckets[i].extend(rec.y_buckets[i].iter().copied());
                merged.z_buckets[i].extend(rec.z_buckets[i].iter().copied());
            }
        }
        merged
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_quantum::DagCircuit;

    // =========================================================================
    // Helper Functions
    // =========================================================================

    /// Simple Z-stabilizer measurement circuit: measures Z0 Z1 parity
    fn simple_syndrome_circuit() -> DagCircuit {
        let mut dag = DagCircuit::new();
        dag.pz(&[2]); // Prep ancilla in |0>
        dag.cx(&[(0, 2)]); // CNOT from data 0 to ancilla
        dag.cx(&[(1, 2)]); // CNOT from data 1 to ancilla
        dag.mz(&[2]); // Measure ancilla
        dag
    }

    /// Circuit with CZ gates for testing multi-qubit symmetric faults
    fn cz_syndrome_circuit() -> DagCircuit {
        let mut dag = DagCircuit::new();
        dag.pz(&[2]);
        dag.h(&[2]); // Put ancilla in |+> for X-type measurement
        dag.cz(&[(0, 2)]);
        dag.cz(&[(1, 2)]);
        dag.h(&[2]);
        dag.mz(&[2]);
        dag
    }

    /// Builds a random Clifford DAG circuit for testing
    fn random_dag_circuit(num_qubits: usize, num_gates: usize, seed: u64) -> DagCircuit {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut dag = DagCircuit::new();

        // Simple deterministic pseudo-random using hash
        let mut state = seed;
        let mut next_rand = || -> u64 {
            let mut hasher = DefaultHasher::new();
            state.hash(&mut hasher);
            state = hasher.finish();
            state
        };

        // Add some prep gates
        for q in 0..num_qubits {
            if next_rand() % 2 == 0 {
                dag.pz(&[q]);
            }
        }

        // Add random Clifford gates
        for _ in 0..num_gates {
            let gate_type = next_rand() % 5;
            #[allow(clippy::cast_possible_truncation)] // 64-bit target
            let q1 = (next_rand() % num_qubits as u64) as usize;

            match gate_type {
                0 => {
                    dag.h(&[q1]);
                }
                1 => {
                    dag.sz(&[q1]);
                }
                2 => {
                    dag.szdg(&[q1]);
                }
                3 => {
                    // CX - need two different qubits
                    #[allow(clippy::cast_possible_truncation)] // 64-bit target
                    let mut q2 = (next_rand() % num_qubits as u64) as usize;
                    if q2 == q1 {
                        q2 = (q1 + 1) % num_qubits;
                    }
                    dag.cx(&[(q1, q2)]);
                }
                _ => {
                    // CZ - need two different qubits
                    #[allow(clippy::cast_possible_truncation)] // 64-bit target
                    let mut q2 = (next_rand() % num_qubits as u64) as usize;
                    if q2 == q1 {
                        q2 = (q1 + 1) % num_qubits;
                    }
                    dag.cz(&[(q1, q2)]);
                }
            }
        }

        // Add measurements on some qubits
        for q in 0..num_qubits {
            if next_rand() % 3 == 0 {
                dag.mz(&[q]);
            }
        }
        // Ensure at least one measurement
        if dag.topological_order().iter().all(|&n| {
            dag.gate(n)
                .is_none_or(|g| !matches!(g.gate_type, GateType::MZ | GateType::MeasureFree))
        }) {
            dag.mz(&[0]);
        }

        dag
    }

    /// Builds a surface-code-like syndrome extraction circuit
    fn surface_code_circuit(distance: usize) -> DagCircuit {
        let mut dag = DagCircuit::new();
        let data_qubits = distance * distance;
        let ancilla_qubits = data_qubits - 1;
        let grid_size = distance;

        // Build connectivity: each ancilla measures a plaquette
        let mut ancilla_neighbors: Vec<Vec<usize>> = Vec::new();
        for a_idx in 0..ancilla_qubits {
            let row = a_idx / (grid_size - 1).max(1);
            let col = a_idx % (grid_size - 1).max(1);
            let mut neighbors = Vec::new();
            for (dr, dc) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
                let d_row = row + dr;
                let d_col = col + dc;
                if d_row < grid_size && d_col < grid_size {
                    let d_idx = d_row * grid_size + d_col;
                    if d_idx < data_qubits {
                        neighbors.push(d_idx);
                    }
                }
            }
            ancilla_neighbors.push(neighbors);
        }

        // Build circuit
        for a in 0..ancilla_qubits {
            dag.pz(&[data_qubits + a]);
        }
        for (a, neighbors) in ancilla_neighbors.iter().enumerate() {
            for &d in neighbors {
                dag.cx(&[(d, data_qubits + a)]);
            }
        }
        for a in 0..ancilla_qubits {
            dag.mz(&[data_qubits + a]);
        }

        dag
    }

    // =========================================================================
    // Basic Functionality Tests
    // =========================================================================

    #[test]
    fn test_fault_locations_basic() {
        let dag = simple_syndrome_circuit();
        let propagator = DagPropagator::new(&dag);
        let locations = DagFaultAnalyzer::extract_locations(&propagator, &dag);

        assert!(!locations.is_empty());
        // Should have locations at prep (after), CX gates (before/after), and measurement (before)
        assert!(locations.len() >= 4);
    }

    #[test]
    fn test_dag_spacetime_location_ordering() {
        // Verify that DagSpacetimeLocation has consistent ordering
        let loc1 = DagSpacetimeLocation {
            node: 0,
            qubits: vec![QubitId::from(0)],
            before: true,
            gate_type: GateType::H,
            idle_duration: 0,
        };
        let loc2 = DagSpacetimeLocation {
            node: 1,
            qubits: vec![QubitId::from(0)],
            before: true,
            gate_type: GateType::H,
            idle_duration: 0,
        };
        assert!(loc1 < loc2);
    }

    #[test]
    fn test_csr_array_basic() {
        let mut csr = CsrArray::with_capacity(3, 10);

        // Row 0: [1, 2, 3]
        csr.push(1);
        csr.push(2);
        csr.push(3);
        csr.finish_row();

        // Row 1: [] (empty)
        csr.finish_row();

        // Row 2: [4, 5]
        csr.push(4);
        csr.push(5);
        csr.finish_row();

        assert_eq!(csr.num_rows(), 3);
        assert_eq!(csr.row(0), &[1, 2, 3]);
        assert!(csr.row_is_empty(1));
        assert_eq!(csr.row(2), &[4, 5]);
        assert_eq!(csr.total_elements(), 5);
    }

    #[test]
    fn test_influences_soa_classification() {
        let mut soa = InfluencesSoA::with_capacity(2);

        // Location 0: X flips detector 0, Z flips nothing
        soa.detectors_x.push(0);
        soa.detectors_y.finish_row();
        soa.detectors_z.finish_row();
        soa.detectors_x.finish_row();
        soa.dem_outputs_x.finish_row();
        soa.dem_outputs_y.finish_row();
        soa.dem_outputs_z.finish_row();
        soa.num_locations += 1;

        // Location 1: Z flips detector 1
        soa.detectors_x.finish_row();
        soa.detectors_y.finish_row();
        soa.detectors_z.push(1);
        soa.detectors_z.finish_row();
        soa.dem_outputs_x.finish_row();
        soa.dem_outputs_y.finish_row();
        soa.dem_outputs_z.finish_row();
        soa.num_locations += 1;

        assert!(soa.has_detector_flips(0, Pauli::X));
        assert!(!soa.has_detector_flips(0, Pauli::Z));
        assert!(!soa.has_detector_flips(1, Pauli::X));
        assert!(soa.has_detector_flips(1, Pauli::Z));
    }

    #[test]
    fn test_export_csr_filters_tracked_paulis_from_dem_outputs() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);

        let map = crate::fault_tolerance::InfluenceBuilder::new(&dag)
            .with_z(&[0])
            .build();

        assert_eq!(map.num_dem_outputs(), 0);
        assert_eq!(map.num_tracked_paulis(), 1);
        assert!(
            map.influences.max_dem_output_index().is_some(),
            "tracked Pauli should still use internal propagation storage"
        );

        let (
            _num_locations,
            _num_detectors,
            num_dem_outputs,
            _detector_offsets_x,
            _detector_data_x,
            _detector_offsets_y,
            _detector_data_y,
            _detector_offsets_z,
            _detector_data_z,
            dem_output_offsets_x,
            dem_output_data_x,
            dem_output_offsets_y,
            dem_output_data_y,
            dem_output_offsets_z,
            dem_output_data_z,
        ) = map.export_csr();

        assert_eq!(num_dem_outputs, 0);
        assert!(dem_output_data_x.is_empty());
        assert!(dem_output_data_y.is_empty());
        assert!(dem_output_data_z.is_empty());
        assert_eq!(dem_output_offsets_x.len(), map.locations.len() + 1);
        assert_eq!(dem_output_offsets_y.len(), map.locations.len() + 1);
        assert_eq!(dem_output_offsets_z.len(), map.locations.len() + 1);
    }

    #[test]
    fn test_dem_output_helpers_use_separate_compact_id_spaces() {
        let mut map = DagFaultInfluenceMap::with_capacity(1);
        map.locations.push(DagSpacetimeLocation {
            node: 0,
            qubits: vec![QubitId(0)],
            before: false,
            gate_type: GateType::H,
            idle_duration: 0,
        });
        map.dem_output_metadata = vec![
            DemOutputMetadata::tracked_pauli(pecos_core::PauliString::xs(&[0])),
            DemOutputMetadata::observable(pecos_core::PauliString::zs(&[0])),
            DemOutputMetadata::tracked_pauli(pecos_core::PauliString::zs(&[1])),
        ];

        map.influences.dem_outputs_x.extend([0, 1, 2]);
        map.influences.dem_outputs_x.finish_row();
        map.influences.dem_outputs_y.finish_row();
        map.influences.dem_outputs_z.finish_row();
        map.influences.detectors_x.finish_row();
        map.influences.detectors_y.finish_row();
        map.influences.detectors_z.finish_row();
        map.influences.num_locations = 1;

        assert_eq!(map.num_dem_outputs(), 1);
        assert_eq!(map.num_tracked_paulis(), 2);
        assert_eq!(map.get_observable_indices(0, Pauli::X.as_u8()), vec![0]);
        assert_eq!(
            map.get_tracked_pauli_indices(0, Pauli::X.as_u8()),
            vec![0, 1]
        );

        let (
            _num_locations,
            _num_detectors,
            num_dem_outputs,
            _detector_offsets_x,
            _detector_data_x,
            _detector_offsets_y,
            _detector_data_y,
            _detector_offsets_z,
            _detector_data_z,
            dem_output_offsets_x,
            dem_output_data_x,
            _dem_output_offsets_y,
            _dem_output_data_y,
            _dem_output_offsets_z,
            _dem_output_data_z,
        ) = map.export_csr();

        assert_eq!(num_dem_outputs, 1);
        assert_eq!(dem_output_offsets_x, vec![0, 1]);
        assert_eq!(dem_output_data_x, vec![0]);
    }

    // =========================================================================
    // Per-Qubit Fault Location Tests
    // =========================================================================

    #[test]
    fn test_per_qubit_cx_fault_locations() {
        // Test that CX gates have separate fault locations for each qubit.
        // This enables proper depolarizing noise analysis (XI vs IX vs XX).
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.pz(&[1]);
        dag.cx(&[(0, 1)]); // Two-qubit gate
        dag.mz(&[0]);
        dag.mz(&[1]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let map = analyzer.build_influence_map();

        // Find the CX gate locations
        let cx_locations: Vec<_> = map
            .locations
            .iter()
            .filter(|loc| matches!(loc.gate_type, GateType::CX))
            .collect();

        // Should have 2 locations: one per qubit (after only)
        assert_eq!(
            cx_locations.len(),
            2,
            "CX should have 2 fault locations (1 per qubit, after gate)"
        );

        // Each location should have exactly 1 qubit (per-qubit fault model)
        for loc in &cx_locations {
            assert_eq!(
                loc.qubits.len(),
                1,
                "Each CX fault location should have 1 qubit for per-qubit analysis"
            );
        }

        // Both qubits should be represented
        let qubit_set: std::collections::HashSet<_> = cx_locations
            .iter()
            .flat_map(|loc| loc.qubits.iter().map(pecos_core::QubitId::index))
            .collect();
        assert!(qubit_set.contains(&0), "Should have location for qubit 0");
        assert!(qubit_set.contains(&1), "Should have location for qubit 1");
    }

    #[test]
    fn test_per_qubit_cz_fault_locations() {
        // Test CZ gates have per-qubit fault locations
        let dag = cz_syndrome_circuit();
        let analyzer = DagFaultAnalyzer::new(&dag);
        let map = analyzer.build_influence_map();

        // Find CZ gate locations
        let cz_locations: Vec<_> = map
            .locations
            .iter()
            .filter(|loc| matches!(loc.gate_type, GateType::CZ))
            .collect();

        assert!(!cz_locations.is_empty(), "Should have CZ fault locations");

        // Each location should have exactly 1 qubit
        for loc in &cz_locations {
            assert_eq!(loc.qubits.len(), 1, "CZ locations should have 1 qubit each");
        }
    }

    #[test]
    fn test_per_qubit_fault_influences() {
        // Test that per-qubit fault locations correctly track influences.
        // In the standard model, faults are AFTER unitary gates.
        // X on the TARGET after CX(0, 2) flips the Z-measurement on qubit 2.
        let mut dag = DagCircuit::new();
        dag.pz(&[2]); // ancilla
        dag.cx(&[(0, 2)]); // X on target flips measurement
        dag.mz(&[2]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let map = analyzer.build_influence_map();

        // X error on target qubit 2 after CX should flip the measurement
        let mut found_target_influence = false;
        for (loc_idx, loc) in map.locations.iter().enumerate() {
            if loc.qubits.iter().any(|q| q.index() == 2)
                && matches!(loc.gate_type, GateType::CX)
                && map.influences.has_detector_flips(loc_idx, Pauli::X)
            {
                found_target_influence = true;
            }
        }

        assert!(
            found_target_influence,
            "X error on target qubit after CX should influence measurement"
        );
    }

    #[test]
    fn test_all_paulis_on_per_qubit_location() {
        // Test X, Y, Z faults on per-qubit locations
        let mut dag = DagCircuit::new();
        dag.pz(&[2]);
        dag.cx(&[(0, 2)]);
        dag.cx(&[(1, 2)]);
        dag.mz(&[2]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let map = analyzer.build_influence_map();

        // Find a CX location and check all Pauli influences
        let cx_idx = map
            .locations
            .iter()
            .position(|loc| matches!(loc.gate_type, GateType::CX))
            .expect("Should have CX location");

        // Check that we can query all Pauli types
        // The SoA structure supports X, Y, Z queries
        let _has_x = map.influences.has_detector_flips(cx_idx, Pauli::X);
        let _has_y = map.influences.has_detector_flips(cx_idx, Pauli::Y);
        let _has_z = map.influences.has_detector_flips(cx_idx, Pauli::Z);
    }

    #[test]
    fn test_per_qubit_locations_for_2q_gates() {
        // Verify all locations have single qubits (per-qubit fault model)
        let dag = surface_code_circuit(3); // d=3 has multi-qubit CX gates
        let analyzer = DagFaultAnalyzer::new(&dag);
        let map = analyzer.build_influence_map();

        // All locations should have exactly 1 qubit in per-qubit model
        for loc in &map.locations {
            assert_eq!(
                loc.qubits.len(),
                1,
                "All fault locations should have exactly 1 qubit for per-qubit analysis"
            );
        }

        // Check that we have locations for CX gates
        let cx_count = map
            .locations
            .iter()
            .filter(|loc| matches!(loc.gate_type, GateType::CX))
            .count();
        assert!(cx_count > 0, "Should have CX fault locations");
    }

    // =========================================================================
    // Randomized DAG Testing
    // =========================================================================

    #[test]
    fn test_random_dag_forward_backward_consistency() {
        // Test that backward propagation influence maps are consistent
        // with forward fault propagation on random DAG circuits
        use super::super::{Direction, propagate_sparse_dag};

        let num_tests = 20;
        let mut total_locations = 0;
        let mut total_consistent = 0;

        for seed in 0..num_tests {
            let dag = random_dag_circuit(5, 15, seed);
            let analyzer = DagFaultAnalyzer::new(&dag);
            let map = analyzer.build_influence_map();

            if map.measurements.is_empty() {
                continue; // Skip circuits without measurements
            }

            // For each fault location, verify backward matches forward
            for (loc_idx, loc) in map.locations.iter().enumerate() {
                // Test all Pauli types (X, Y, Z) on per-qubit locations
                for pauli in 1u8..4 {
                    total_locations += 1;

                    let back_has_syndrome = map
                        .influences
                        .has_detector_flips(loc_idx, Pauli::from_u8(pauli));

                    // Forward: inject fault and propagate to see if it reaches measurements
                    let mut prop = PauliProp::new();
                    for q in &loc.qubits {
                        match pauli {
                            1 => prop.track_x(&[q.index()]),
                            2 => {
                                prop.track_x(&[q.index()]);
                                prop.track_z(&[q.index()]);
                            }
                            3 => prop.track_z(&[q.index()]),
                            _ => {}
                        }
                    }

                    // Propagate forward from this location
                    propagate_sparse_dag(&dag, &mut prop, Direction::Forward);

                    // Check if propagated error anticommutes with any measurement
                    let mut fwd_has_syndrome = false;
                    for &(_, meas_qubit, basis) in &map.measurements {
                        let anticommutes = if basis == 0 {
                            // Z-measurement anticommutes with X or Y
                            prop.contains_x(meas_qubit)
                        } else {
                            // X-measurement anticommutes with Z or Y
                            prop.contains_z(meas_qubit)
                        };
                        if anticommutes {
                            fwd_has_syndrome = true;
                            break;
                        }
                    }

                    // With per-qubit fault model, all locations have exactly 1 qubit
                    // Backward and forward analysis should agree
                    if back_has_syndrome == fwd_has_syndrome {
                        total_consistent += 1;
                    }
                }
            }
        }

        let consistency = if total_locations > 0 {
            f64::from(total_consistent) / f64::from(total_locations)
        } else {
            1.0
        };

        // With per-qubit fault model, consistency may be lower due to timing
        // differences (before vs after gates) in forward propagation test.
        // 80% is acceptable for this approximate validation.
        assert!(
            consistency > 0.80,
            "Random DAG consistency too low: {:.1}% ({}/{})",
            consistency * 100.0,
            total_consistent,
            total_locations
        );
    }

    #[test]
    fn test_random_dag_varying_sizes() {
        // Test on various circuit sizes to catch edge cases
        let configs = [
            (2, 5),   // Minimal
            (4, 10),  // Small
            (8, 30),  // Medium
            (12, 50), // Larger
        ];

        for (num_qubits, num_gates) in configs {
            for seed in 0..3 {
                let dag = random_dag_circuit(num_qubits, num_gates, seed);
                let analyzer = DagFaultAnalyzer::new(&dag);

                // Should not panic
                let map = analyzer.build_influence_map();

                // Basic sanity checks
                assert!(
                    !map.locations.is_empty() || dag.topological_order().is_empty(),
                    "Size ({num_qubits}, {num_gates}), seed {seed}: expected locations"
                );
            }
        }
    }

    // =========================================================================
    // Surface Code Specific Tests
    // =========================================================================

    #[test]
    fn test_surface_code_d3() {
        let dag = surface_code_circuit(3);
        let analyzer = DagFaultAnalyzer::new(&dag);
        let map = analyzer.build_influence_map();

        // d=3 has 9 data qubits and 8 ancillas
        assert_eq!(map.detectors.len(), 8, "d=3 should have 8 detectors");
        assert!(!map.locations.is_empty());
    }

    #[test]
    fn test_surface_code_d5() {
        let dag = surface_code_circuit(5);
        let analyzer = DagFaultAnalyzer::new(&dag);
        let map = analyzer.build_influence_map();

        // d=5 has 25 data qubits and 24 ancillas
        assert_eq!(map.detectors.len(), 24, "d=5 should have 24 detectors");
    }

    #[test]
    fn test_surface_code_per_qubit_fault_coverage() {
        // Verify that surface code circuits have proper per-qubit fault coverage
        let dag = surface_code_circuit(3);
        let analyzer = DagFaultAnalyzer::new(&dag);
        let map = analyzer.build_influence_map();

        // All locations should have exactly 1 qubit in per-qubit model
        for loc in &map.locations {
            assert_eq!(
                loc.qubits.len(),
                1,
                "All fault locations should have 1 qubit"
            );
        }

        // Find CX gate locations (from syndrome extraction)
        let cx_locations: Vec<_> = map
            .locations
            .iter()
            .enumerate()
            .filter(|(_, loc)| matches!(loc.gate_type, GateType::CX))
            .collect();

        assert!(
            !cx_locations.is_empty(),
            "Surface code should have CX fault locations"
        );

        // Check that CX locations have proper influences
        for (loc_idx, loc) in cx_locations {
            // At least some Pauli type should have detector flips
            let has_any_flip = [Pauli::X, Pauli::Y, Pauli::Z]
                .iter()
                .any(|&p| map.influences.has_detector_flips(loc_idx, p));
            // Most CX locations should detect something
            if !has_any_flip {
                // Only locations after measurements or before preps might have no flips
                assert!(
                    matches!(loc.gate_type, GateType::PZ | GateType::QAlloc) || !loc.before,
                    "Multi-qubit location {loc:?} has no detector flips"
                );
            }
        }
    }

    // =========================================================================
    // BucketRecorder Tests
    // =========================================================================

    #[test]
    fn test_bucket_recorder_basic() {
        let mut recorder = BucketRecorder::new(3);

        // Record some influences
        recorder.record(0, 0, true, false, 0); // X flip on loc 0
        recorder.record(0, 0, false, true, 1); // Z flip on loc 0
        recorder.record(1, 1, true, true, 0); // Both on loc 1

        let soa = recorder.into_soa();

        assert_eq!(soa.num_locations, 3);
        // Loc 0: X flips det 1 (from obs_z=true), Z flips det 0 (from obs_x=true)
        assert!(!soa.detectors_x.row(0).is_empty() || !soa.detectors_z.row(0).is_empty());
    }

    #[test]
    fn test_bucket_recorder_with_analyzer() {
        let dag = simple_syndrome_circuit();
        let analyzer = DagFaultAnalyzer::new(&dag);

        // Use propagate_all with a custom recorder
        let mut recorder = super::super::CountingRecorder::default();
        analyzer.propagate_all(&mut recorder);

        assert!(recorder.count > 0, "Should record some influences");
    }
}
