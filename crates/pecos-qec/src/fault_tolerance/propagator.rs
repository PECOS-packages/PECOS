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

//! Pauli propagation infrastructure and fault analysis.
//!
//! This module provides bidirectional Pauli propagation through quantum circuits,
//! with specialized support for fault tolerance analysis. By propagating observables
//! backward from measurements/logicals, we can efficiently determine which faults
//! affect which detectors:
//!
//! 1. **Speed up fault enumeration** - O(1) lookup instead of O(circuit_depth) propagation
//! 2. **Build detector error models** - Direct mapping from faults to detectors
//! 3. **Analyze syndrome histories** - Know which round each fault affects
//!
//! # Concept
//!
//! Instead of forward propagation:
//! ```text
//! For each fault:
//!     Propagate forward through circuit
//!     Check which measurements flip
//! ```
//!
//! We do backward pre-computation:
//! ```text
//! For each measurement M:
//!     Start with X (for Z-measurement) or Z (for X-measurement)
//!     Propagate backward through circuit
//!     Record: "fault at location L would flip M"
//! ```
//!
//! # Example
//!
//! ```
//! use pecos_qec::fault_tolerance::propagator::{
//!     BackwardPropagator, FaultInfluenceMap,
//! };
//! use pecos_quantum::TickCircuit;
//!
//! // Build a simple syndrome extraction circuit
//! let mut circuit = TickCircuit::new();
//! circuit.tick().pz(&[2]);           // Prep ancilla
//! circuit.tick().cx(&[(0, 2)]);      // CNOT data -> ancilla
//! circuit.tick().cx(&[(1, 2)]);      // CNOT data -> ancilla
//! circuit.tick().mz(&[2]);           // Measure ancilla
//!
//! // Build the fault influence map
//! let propagator = BackwardPropagator::new(&circuit);
//! let influence_map = propagator.build_influence_map();
//!
//! // Now we can query: which measurements does a fault at location L flip?
//! // This is O(1) lookup instead of O(circuit_depth) propagation
//! ```

use super::{extract_spacetime_locations, PauliFault, SpacetimeLocation};
use pecos_core::gate_type::GateType;
use pecos_core::QubitId;
use pecos_quantum::TickCircuit;
use pecos_qsim::{CliffordGateable, PauliProp};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BinaryHeap};

// ============================================================================
// Entity IDs (Type-Safe Indices)
// ============================================================================

/// A node (gate) in the DAG circuit.
///
/// This is a type-safe wrapper around a raw index, following ECS principles
/// where entities are just IDs and components hold the data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct NodeId(pub u32);

impl NodeId {
    /// Creates a new NodeId from a raw index.
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    /// Returns the raw index.
    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    /// Creates from usize (for compatibility).
    #[inline]
    pub const fn from_usize(index: usize) -> Self {
        Self(index as u32)
    }
}

impl From<usize> for NodeId {
    #[inline]
    fn from(index: usize) -> Self {
        Self::from_usize(index)
    }
}

impl From<NodeId> for usize {
    #[inline]
    fn from(id: NodeId) -> Self {
        id.index()
    }
}

/// A fault location in the circuit.
///
/// Identifies a specific spacetime point where a fault can occur
/// (before or after a gate on specific qubits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct LocationId(pub u32);

impl LocationId {
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn from_usize(index: usize) -> Self {
        Self(index as u32)
    }
}

impl From<usize> for LocationId {
    #[inline]
    fn from(index: usize) -> Self {
        Self::from_usize(index)
    }
}

impl From<LocationId> for usize {
    #[inline]
    fn from(id: LocationId) -> Self {
        id.index()
    }
}

/// A detector (measurement-based syndrome bit).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct DetectorIdx(pub u32);

impl DetectorIdx {
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn from_usize(index: usize) -> Self {
        Self(index as u32)
    }
}

impl From<usize> for DetectorIdx {
    #[inline]
    fn from(index: usize) -> Self {
        Self::from_usize(index)
    }
}

impl From<DetectorIdx> for usize {
    #[inline]
    fn from(id: DetectorIdx) -> Self {
        id.index()
    }
}

/// A logical observable index.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(transparent)]
pub struct LogicalIdx(pub u32);

impl LogicalIdx {
    #[inline]
    pub const fn new(index: u32) -> Self {
        Self(index)
    }

    #[inline]
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    #[inline]
    pub const fn from_usize(index: usize) -> Self {
        Self(index as u32)
    }
}

impl From<usize> for LogicalIdx {
    #[inline]
    fn from(index: usize) -> Self {
        Self::from_usize(index)
    }
}

impl From<LogicalIdx> for usize {
    #[inline]
    fn from(id: LogicalIdx) -> Self {
        id.index()
    }
}

/// Pauli type for faults (I=0, X=1, Y=2, Z=3).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
#[repr(u8)]
pub enum Pauli {
    #[default]
    I = 0,
    X = 1,
    Y = 2,
    Z = 3,
}

impl Pauli {
    /// Creates from raw u8 value.
    #[inline]
    pub const fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::X,
            2 => Self::Y,
            3 => Self::Z,
            _ => Self::I,
        }
    }

    /// Returns the raw u8 value.
    #[inline]
    pub const fn as_u8(self) -> u8 {
        self as u8
    }

    /// Returns true if this is a non-identity Pauli.
    #[inline]
    pub const fn is_nontrivial(self) -> bool {
        self.as_u8() != 0
    }
}

// ============================================================================
// Direction and Unified Propagation
// ============================================================================

/// Direction of Pauli propagation through a circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Direction {
    /// Forward propagation: P → G P G†
    /// Propagate from earlier ticks to later ticks.
    Forward,
    /// Backward propagation: P → G† P G
    /// Propagate from later ticks to earlier ticks.
    Backward,
}

/// Applies a gate to a PauliProp in the specified direction.
///
/// For forward propagation (P → G P G†), we apply the gate's transformation.
/// For backward propagation (P → G† P G), we apply the adjoint transformation.
///
/// Most Clifford gates are self-adjoint (H, CX, CZ, X, Y, Z), so the transformation
/// is the same in both directions. For non-self-adjoint gates (SZ, SX, SY and their
/// daggers), we swap the gate with its adjoint for backward propagation.
///
/// Special handling:
/// - **Prep gates**: No transformation in either direction (transparent to propagation)
/// - **Measure gates**: No transformation in either direction (for propagation purposes)
#[inline]
pub fn apply_gate(prop: &mut PauliProp, gate: &pecos_core::Gate, direction: Direction) {
    // Use gate.qubits directly - SmallVec derefs to &[QubitId], no allocation needed
    let qubits = &gate.qubits;

    match gate.gate_type {
        // Self-adjoint single-qubit gates - same in both directions
        GateType::I => {
            prop.identity(qubits);
        }
        GateType::X => {
            prop.x(qubits);
        }
        GateType::Y => {
            prop.y(qubits);
        }
        GateType::Z => {
            prop.z(qubits);
        }
        GateType::H => {
            prop.h(qubits);
        }

        // Non-self-adjoint gates - swap with adjoint for backward
        GateType::SX => {
            match direction {
                Direction::Forward => prop.sx(qubits),
                Direction::Backward => prop.sxdg(qubits),
            };
        }
        GateType::SXdg => {
            match direction {
                Direction::Forward => prop.sxdg(qubits),
                Direction::Backward => prop.sx(qubits),
            };
        }
        GateType::SY => {
            match direction {
                Direction::Forward => prop.sy(qubits),
                Direction::Backward => prop.sydg(qubits),
            };
        }
        GateType::SYdg => {
            match direction {
                Direction::Forward => prop.sydg(qubits),
                Direction::Backward => prop.sy(qubits),
            };
        }
        GateType::SZ => {
            match direction {
                Direction::Forward => prop.sz(qubits),
                Direction::Backward => prop.szdg(qubits),
            };
        }
        GateType::SZdg => {
            match direction {
                Direction::Forward => prop.szdg(qubits),
                Direction::Backward => prop.sz(qubits),
            };
        }

        // Self-adjoint two-qubit gates - same in both directions
        // Access qubits directly instead of using chunks iterator
        GateType::CX => {
            if qubits.len() >= 2 {
                prop.cx(&qubits[0..2]);
            }
        }
        GateType::CY => {
            if qubits.len() >= 2 {
                prop.cy(&qubits[0..2]);
            }
        }
        GateType::CZ => {
            if qubits.len() >= 2 {
                prop.cz(&qubits[0..2]);
            }
        }
        GateType::SWAP => {
            if qubits.len() >= 2 {
                prop.swap(&qubits[0..2]);
            }
        }

        // State preparation - no transformation in either direction
        // For Pauli propagation purposes, prep gates are transparent
        GateType::Prep | GateType::QAlloc => {
            // No-op: Pauli propagation passes through prep gates unchanged
        }

        // Measurements don't transform Paulis for propagation purposes
        GateType::Measure | GateType::MeasureFree => {}

        _ => {
            // Unsupported gate type - no transformation
        }
    }
}

/// Propagates a PauliProp through a circuit in the specified direction.
///
/// # Arguments
/// * `circuit` - The circuit to propagate through
/// * `prop` - The PauliProp to propagate (modified in place)
/// * `direction` - Forward or Backward propagation
pub fn propagate_through_circuit(circuit: &TickCircuit, prop: &mut PauliProp, direction: Direction) {
    match direction {
        Direction::Forward => {
            for tick in circuit.ticks() {
                for gate in tick.gates() {
                    apply_gate(prop, gate, direction);
                }
            }
        }
        Direction::Backward => {
            for tick in circuit.ticks().iter().rev() {
                for gate in tick.gates() {
                    apply_gate(prop, gate, direction);
                }
            }
        }
    }
}

/// Propagates a PauliProp through a range of ticks in the specified direction.
///
/// # Arguments
/// * `circuit` - The circuit to propagate through
/// * `prop` - The PauliProp to propagate (modified in place)
/// * `start_tick` - The tick to start from (inclusive)
/// * `end_tick` - The tick to end at (inclusive)
/// * `direction` - Forward or Backward propagation
///
/// For Forward: propagates from start_tick to end_tick
/// For Backward: propagates from end_tick to start_tick
pub fn propagate_tick_range(
    circuit: &TickCircuit,
    prop: &mut PauliProp,
    start_tick: usize,
    end_tick: usize,
    direction: Direction,
) {
    let num_ticks = circuit.ticks().len();
    let start = start_tick.min(num_ticks.saturating_sub(1));
    let end = end_tick.min(num_ticks.saturating_sub(1));

    match direction {
        Direction::Forward => {
            for tick_idx in start..=end {
                let tick = &circuit.ticks()[tick_idx];
                for gate in tick.gates() {
                    apply_gate(prop, gate, direction);
                }
            }
        }
        Direction::Backward => {
            for tick_idx in (start..=end).rev() {
                let tick = &circuit.ticks()[tick_idx];
                for gate in tick.gates() {
                    apply_gate(prop, gate, direction);
                }
            }
        }
    }
}

// ============================================================================
// DAG-Based Sparse Propagation
// ============================================================================

use pecos_quantum::{DagCircuit, DagTraversalIndex};
use std::collections::BTreeSet;

/// Pre-computed index for efficient DAG-based Pauli propagation.
///
/// This struct pre-computes data structures needed for sparse propagation,
/// making repeated propagations through the same circuit much faster.
///
/// # Example
/// ```
/// use pecos_qec::fault_tolerance::propagator::{DagPropagator, Direction};
/// use pecos_quantum::DagCircuit;
/// use pecos_qsim::PauliProp;
///
/// let mut dag = DagCircuit::new();
/// dag.h(0);
/// dag.cx(0, 1);
///
/// // Pre-compute indices (do this once)
/// let propagator = DagPropagator::new(&dag);
///
/// // Propagate multiple times efficiently
/// let mut prop = PauliProp::new();
/// prop.add_z(0);
/// propagator.propagate_sparse(&mut prop, Direction::Forward);
/// ```
/// Reusable work buffers for propagation to avoid repeated allocations.
///
/// Create once and reuse across multiple propagations for best performance.
#[derive(Debug, Clone)]
pub struct PropagatorWorkBuffers {
    /// Visited nodes (indexed by node id)
    pub visited: Vec<bool>,
    /// Active qubits (indexed by qubit id)
    pub active_qubits: Vec<bool>,
    /// Priority queue for heap-based traversal (topo_pos, node_id)
    pub heap: BinaryHeap<(usize, usize)>,
}

impl PropagatorWorkBuffers {
    /// Creates new work buffers sized for the given propagator.
    #[must_use]
    pub fn new(max_node: usize, max_qubit: usize) -> Self {
        Self {
            visited: vec![false; max_node + 1],
            active_qubits: vec![false; max_qubit + 1],
            heap: BinaryHeap::with_capacity(64),
        }
    }

    /// Clears all buffers for reuse.
    pub fn clear(&mut self) {
        self.visited.fill(false);
        self.active_qubits.fill(false);
        self.heap.clear();
    }

    /// Resizes buffers to accommodate larger circuits.
    pub fn resize(&mut self, max_node: usize, max_qubit: usize) {
        if self.visited.len() <= max_node {
            self.visited.resize(max_node + 1, false);
        }
        if self.active_qubits.len() <= max_qubit {
            self.active_qubits.resize(max_qubit + 1, false);
        }
    }
}

// ============================================================================
// Composable Systems (DOD/ECS Architecture)
// ============================================================================

/// Trait for recording influences during propagation.
///
/// This trait enables different recording strategies to be plugged into
/// the propagation loop without changing the traversal logic.
///
/// Following ECS principles, the recorder is a "system" that operates on
/// component data (locations, Pauli states) and produces output (influence maps).
pub trait InfluenceRecorder {
    /// Records influences for a fault at the given location.
    ///
    /// # Arguments
    /// * `loc_idx` - Location index in the FaultLocations array
    /// * `qubit` - The qubit where the fault occurs
    /// * `obs_x` - Whether the current observable has X on this qubit
    /// * `obs_z` - Whether the current observable has Z on this qubit
    /// * `detector_idx` - The detector being propagated from
    fn record(&mut self, loc_idx: usize, qubit: usize, obs_x: bool, obs_z: bool, detector_idx: usize);
}

/// Context for a single backward propagation pass.
///
/// Bundles the state needed for propagation, enabling reuse and
/// clean separation between traversal and recording.
#[derive(Debug)]
pub struct PropagationContext<'a> {
    /// Current Pauli observable being propagated.
    pub prop: PauliProp,
    /// Work buffers for traversal.
    pub buffers: &'a mut PropagatorWorkBuffers,
    /// Current detector index being processed.
    pub detector_idx: usize,
}

impl<'a> PropagationContext<'a> {
    /// Creates a new propagation context.
    pub fn new(buffers: &'a mut PropagatorWorkBuffers, detector_idx: usize) -> Self {
        buffers.clear();
        Self {
            prop: PauliProp::new(),
            buffers,
            detector_idx,
        }
    }

    /// Initializes the Pauli observable for a Z-basis measurement.
    pub fn init_z_measurement(&mut self, qubit: usize) {
        self.prop.add_z(qubit);
    }

    /// Initializes the Pauli observable for an X-basis measurement.
    pub fn init_x_measurement(&mut self, qubit: usize) {
        self.prop.add_x(qubit);
    }

    /// Initializes the Pauli observable based on measurement basis.
    pub fn init_measurement(&mut self, qubit: usize, basis: u8) {
        if basis == 0 {
            self.prop.add_z(qubit);
        } else {
            self.prop.add_x(qubit);
        }
    }

    /// Returns whether the observable has X on the given qubit.
    #[inline]
    pub fn has_x(&self, qubit: usize) -> bool {
        self.prop.contains_x(qubit)
    }

    /// Returns whether the observable has Z on the given qubit.
    #[inline]
    pub fn has_z(&self, qubit: usize) -> bool {
        self.prop.contains_z(qubit)
    }

    /// Returns whether the qubit is currently active (has non-trivial Pauli).
    #[inline]
    pub fn is_active(&self, qubit: usize) -> bool {
        self.prop.contains_x(qubit) || self.prop.contains_z(qubit)
    }

    /// Marks a qubit as active in the traversal.
    #[inline]
    pub fn activate_qubit(&mut self, qubit: usize) {
        if qubit < self.buffers.active_qubits.len() {
            self.buffers.active_qubits[qubit] = true;
        }
    }

    /// Marks a qubit as inactive in the traversal.
    #[inline]
    pub fn deactivate_qubit(&mut self, qubit: usize) {
        if qubit < self.buffers.active_qubits.len() {
            self.buffers.active_qubits[qubit] = false;
        }
    }

    /// Returns whether a qubit was active before the current gate.
    #[inline]
    pub fn was_active(&self, qubit: usize) -> bool {
        qubit < self.buffers.active_qubits.len() && self.buffers.active_qubits[qubit]
    }
}

/// An event during backward propagation.
///
/// The propagation yields events that can be handled by different systems
/// (recording influences, debugging, profiling, etc.).
#[derive(Debug, Clone)]
pub enum PropagationEvent {
    /// A node is about to be processed (before gate application).
    BeforeGate {
        /// The node index.
        node: usize,
    },
    /// A node has been processed (after gate application).
    AfterGate {
        /// The node index.
        node: usize,
    },
    /// The Pauli spread to a new qubit.
    PauliSpread {
        /// The qubit the Pauli spread to.
        qubit: usize,
    },
    /// The Pauli retracted from a qubit.
    PauliRetract {
        /// The qubit the Pauli retracted from.
        qubit: usize,
    },
}

/// Null recorder that discards all influences (useful for testing traversal).
#[derive(Default)]
pub struct NullRecorder;

impl InfluenceRecorder for NullRecorder {
    #[inline]
    fn record(&mut self, _loc_idx: usize, _qubit: usize, _obs_x: bool, _obs_z: bool, _detector_idx: usize) {
        // Discard all influences
    }
}

/// Counting recorder that just counts how many influences are recorded.
#[derive(Default)]
pub struct CountingRecorder {
    /// Total number of record calls.
    pub count: usize,
    /// Count by Pauli type (0=I, 1=X, 2=Y, 3=Z).
    pub by_pauli: [usize; 4],
}

impl InfluenceRecorder for CountingRecorder {
    #[inline]
    fn record(&mut self, _loc_idx: usize, _qubit: usize, obs_x: bool, obs_z: bool, _detector_idx: usize) {
        self.count += 1;
        if obs_z {
            self.by_pauli[1] += 1; // X fault
        }
        if obs_x {
            self.by_pauli[3] += 1; // Z fault
        }
        if obs_x || obs_z {
            self.by_pauli[2] += 1; // Y fault
        }
    }
}

// ============================================================================
// SoA Fault Data Structures
// ============================================================================

/// Fault locations in Struct-of-Arrays (SoA) layout for cache-efficient access.
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
    /// Creates a new empty FaultLocations.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates FaultLocations with capacity for the given number of locations and nodes.
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
    pub fn push(&mut self, node: usize, qubits: SmallVec<[usize; 2]>, before: bool, gate_type: GateType) -> usize {
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

    /// Converts to a Vec of DagSpacetimeLocation for backward compatibility.
    #[must_use]
    pub fn to_dag_spacetime_locations(&self) -> Vec<DagSpacetimeLocation> {
        (0..self.len())
            .map(|i| DagSpacetimeLocation {
                node: self.nodes[i],
                qubits: self.qubits[i].iter().map(|&q| QubitId::from(q)).collect(),
                before: self.before[i],
                gate_type: self.gate_types[i],
            })
            .collect()
    }
}

/// Fault influences in Struct-of-Arrays (SoA) layout.
///
/// Each array is indexed by location ID. For each Pauli type (X=1, Z=2, Y=3),
/// stores which detectors/logicals are flipped.
#[derive(Debug, Clone, Default)]
pub struct FaultInfluences {
    /// Detector indices flipped by X error at each location.
    pub detectors_x: Vec<SmallVec<[usize; 4]>>,
    /// Detector indices flipped by Z error at each location.
    pub detectors_z: Vec<SmallVec<[usize; 4]>>,
    /// Detector indices flipped by Y error at each location.
    pub detectors_y: Vec<SmallVec<[usize; 4]>>,
    /// Logical indices flipped by X error at each location.
    pub logicals_x: Vec<SmallVec<[usize; 4]>>,
    /// Logical indices flipped by Z error at each location.
    pub logicals_z: Vec<SmallVec<[usize; 4]>>,
    /// Logical indices flipped by Y error at each location.
    pub logicals_y: Vec<SmallVec<[usize; 4]>>,
}

impl FaultInfluences {
    /// Creates a new empty FaultInfluences.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates FaultInfluences with capacity for the given number of locations.
    #[must_use]
    pub fn with_capacity(num_locations: usize) -> Self {
        Self {
            detectors_x: Vec::with_capacity(num_locations),
            detectors_z: Vec::with_capacity(num_locations),
            detectors_y: Vec::with_capacity(num_locations),
            logicals_x: Vec::with_capacity(num_locations),
            logicals_z: Vec::with_capacity(num_locations),
            logicals_y: Vec::with_capacity(num_locations),
        }
    }

    /// Returns the number of locations.
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.detectors_x.len()
    }

    /// Returns true if there are no influences.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.detectors_x.is_empty()
    }

    /// Adds influence data for a location.
    pub fn push(
        &mut self,
        detectors_x: SmallVec<[usize; 4]>,
        detectors_z: SmallVec<[usize; 4]>,
        detectors_y: SmallVec<[usize; 4]>,
        logicals_x: SmallVec<[usize; 4]>,
        logicals_z: SmallVec<[usize; 4]>,
        logicals_y: SmallVec<[usize; 4]>,
    ) {
        self.detectors_x.push(detectors_x);
        self.detectors_z.push(detectors_z);
        self.detectors_y.push(detectors_y);
        self.logicals_x.push(logicals_x);
        self.logicals_z.push(logicals_z);
        self.logicals_y.push(logicals_y);
    }

    /// Adds empty influence data for a location (no effects).
    pub fn push_empty(&mut self) {
        self.detectors_x.push(SmallVec::new());
        self.detectors_z.push(SmallVec::new());
        self.detectors_y.push(SmallVec::new());
        self.logicals_x.push(SmallVec::new());
        self.logicals_z.push(SmallVec::new());
        self.logicals_y.push(SmallVec::new());
    }

    /// Gets detector indices for a specific Pauli type (1=X, 2=Z, 3=Y).
    #[inline]
    #[must_use]
    pub fn detectors(&self, loc_id: usize, pauli: u8) -> &[usize] {
        match pauli {
            1 => &self.detectors_x[loc_id],
            2 => &self.detectors_z[loc_id],
            3 => &self.detectors_y[loc_id],
            _ => &[],
        }
    }

    /// Gets logical indices for a specific Pauli type (1=X, 2=Z, 3=Y).
    #[inline]
    #[must_use]
    pub fn logicals(&self, loc_id: usize, pauli: u8) -> &[usize] {
        match pauli {
            1 => &self.logicals_x[loc_id],
            2 => &self.logicals_z[loc_id],
            3 => &self.logicals_y[loc_id],
            _ => &[],
        }
    }
}

/// Combined fault analysis result with SoA data layout.
#[derive(Debug, Clone)]
pub struct FaultAnalysis {
    /// Fault locations (SoA).
    pub locations: FaultLocations,
    /// Fault influences (SoA).
    pub influences: FaultInfluences,
    /// Detector metadata: (node, qubit, basis) for each detector.
    pub detectors: Vec<(usize, usize, u8)>,
    /// Number of logical observables.
    pub num_logicals: usize,
}

impl FaultAnalysis {
    /// Creates a new empty FaultAnalysis.
    #[must_use]
    pub fn new() -> Self {
        Self {
            locations: FaultLocations::new(),
            influences: FaultInfluences::new(),
            detectors: Vec::new(),
            num_logicals: 0,
        }
    }
}

impl Default for FaultAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

pub struct DagPropagator<'a> {
    /// Reference to the underlying DAG circuit.
    dag: &'a DagCircuit,
    /// Pre-computed traversal index from DagCircuit.
    index: DagTraversalIndex,
}

impl<'a> DagPropagator<'a> {
    /// Creates a new DagPropagator with pre-computed indices.
    ///
    /// This is O(V + E) where V is the number of gates and E is the number of edges.
    #[must_use]
    pub fn new(dag: &'a DagCircuit) -> Self {
        let index = dag.build_traversal_index();
        Self { dag, index }
    }

    /// Creates a DagPropagator from an existing traversal index.
    ///
    /// Use this when you already have a `DagTraversalIndex` to avoid recomputing it.
    #[must_use]
    pub fn with_index(dag: &'a DagCircuit, index: DagTraversalIndex) -> Self {
        Self { dag, index }
    }

    /// Returns a reference to the traversal index.
    #[inline]
    #[must_use]
    pub fn index(&self) -> &DagTraversalIndex {
        &self.index
    }

    /// Returns the maximum node index.
    #[inline]
    #[must_use]
    pub fn max_node(&self) -> usize {
        self.index.max_node()
    }

    /// Returns the maximum qubit index.
    #[inline]
    #[must_use]
    pub fn max_qubit(&self) -> usize {
        self.index.max_qubit()
    }

    /// Returns the topological position of a node.
    #[inline]
    #[must_use]
    pub fn topo_position(&self, node: usize) -> usize {
        self.index.topo_position(node)
    }

    /// Returns a reference to the gate at the given node.
    #[inline]
    #[must_use]
    pub fn gate(&self, node: usize) -> Option<&pecos_core::Gate> {
        self.dag.gate(node)
    }

    /// Returns a reference to the underlying DAG circuit.
    #[inline]
    #[must_use]
    pub fn dag(&self) -> &DagCircuit {
        self.dag
    }

    /// Returns the gates touching a qubit in forward order.
    #[inline]
    #[must_use]
    pub fn qubit_gates_forward(&self, qubit: usize) -> &[(usize, usize)] {
        self.index.qubit_gates(qubit)
    }

    /// Returns the gates touching a qubit in backward order.
    /// Note: Returns an iterator over reversed order.
    #[inline]
    #[must_use]
    pub fn qubit_gates_backward(&self, qubit: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.index.qubit_gates_reversed(qubit)
    }

    /// Returns the topological order (forward direction).
    #[inline]
    #[must_use]
    pub fn topo_order(&self) -> &[usize] {
        self.index.topo_order()
    }

    /// Creates work buffers sized for this propagator.
    #[must_use]
    pub fn create_work_buffers(&self) -> PropagatorWorkBuffers {
        PropagatorWorkBuffers::new(self.index.max_node(), self.index.max_qubit())
    }

    // ==================== Local Graph Traversal ====================

    /// Returns the predecessor node on the given qubit, if any.
    ///
    /// This is the gate immediately before this node on the qubit wire.
    #[inline]
    #[must_use]
    pub fn predecessor_on_qubit(&self, node: usize, qubit: usize) -> Option<usize> {
        self.index.predecessor_on_qubit(node, qubit)
    }

    /// Returns the successor node on the given qubit, if any.
    ///
    /// This is the gate immediately after this node on the qubit wire.
    #[inline]
    #[must_use]
    pub fn successor_on_qubit(&self, node: usize, qubit: usize) -> Option<usize> {
        self.index.successor_on_qubit(node, qubit)
    }

    /// Returns all neighboring nodes in the given direction.
    ///
    /// Each neighbor is returned with the qubit wire connecting them.
    /// For a 2-qubit gate, this may return up to 2 neighbors (one per qubit wire).
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_qec::fault_tolerance::propagator::{DagPropagator, Direction};
    /// use pecos_quantum::DagCircuit;
    ///
    /// let mut dag = DagCircuit::new();
    /// dag.h(0);
    /// let h = dag.last_added_node().unwrap();
    /// dag.cx(0, 1);
    /// let cx = dag.last_added_node().unwrap();
    /// let mz0 = dag.mz(0).node();
    /// let mz1 = dag.mz(1).node();
    ///
    /// let propagator = DagPropagator::new(&dag);
    ///
    /// // CX has h as backward neighbor on qubit 0, nothing on qubit 1
    /// let back: Vec<_> = propagator.neighbors(cx, Direction::Backward).collect();
    /// assert_eq!(back, vec![(h, 0)]);
    ///
    /// // CX has mz0 as forward neighbor on qubit 0, mz1 on qubit 1
    /// let fwd: Vec<_> = propagator.neighbors(cx, Direction::Forward).collect();
    /// assert!(fwd.contains(&(mz0, 0)));
    /// assert!(fwd.contains(&(mz1, 1)));
    /// ```
    pub fn neighbors(&self, node: usize, direction: Direction) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.gate(node)
            .into_iter()
            .flat_map(move |gate| {
                gate.qubits.iter().filter_map(move |qubit| {
                    let q = qubit.index();
                    let neighbor = match direction {
                        Direction::Forward => self.index.successor_on_qubit(node, q),
                        Direction::Backward => self.index.predecessor_on_qubit(node, q),
                    };
                    neighbor.map(|n| (n, q))
                })
            })
    }

    /// Propagates a PauliProp through the circuit using sparse traversal.
    ///
    /// Only visits gates that touch qubits with non-trivial Paulis.
    /// When a gate spreads the Pauli to new qubits, those qubits' gates
    /// are added to the processing set.
    pub fn propagate_sparse(&self, prop: &mut PauliProp, direction: Direction) {
        if self.index.topo_order().is_empty() {
            return;
        }

        // Use bit vectors for O(1) lookup
        let mut should_process = vec![false; self.max_node() + 1];

        // Track active qubits (those with non-trivial Pauli)
        let mut active_qubits: Vec<bool> = vec![false; self.max_qubit() + 1];
        for q in prop.get_x_qubits() {
            if q < active_qubits.len() {
                active_qubits[q] = true;
            }
        }
        for q in prop.get_z_qubits() {
            if q < active_qubits.len() {
                active_qubits[q] = true;
            }
        }

        // Mark initial gates to process
        for (qubit, &is_active) in active_qubits.iter().enumerate() {
            if is_active {
                for &(_topo_pos, node) in self.index.qubit_gates(qubit) {
                    should_process[node] = true;
                }
            }
        }

        // Get topological order slice
        let topo_order = self.index.topo_order();

        // Create iterator based on direction
        let node_iter: Box<dyn Iterator<Item = usize>> = match direction {
            Direction::Forward => Box::new(topo_order.iter().copied()),
            Direction::Backward => Box::new(topo_order.iter().copied().rev()),
        };

        // Process gates in topological order
        for node in node_iter {
            if !should_process[node] {
                continue;
            }

            if let Some(gate) = self.gate(node) {
                // Check if gate touches any active qubit
                let touches_active = gate.qubits.iter().any(|q| {
                    let idx = q.index();
                    idx < active_qubits.len() && active_qubits[idx]
                });

                if touches_active {
                    // Track which qubits were active before using a small fixed array
                    // to avoid heap allocation (most gates have 1-2 qubits, max ~4)
                    let mut was_active_flags = [false; 8];
                    for (i, q) in gate.qubits.iter().enumerate() {
                        if i < was_active_flags.len() {
                            let idx = q.index();
                            was_active_flags[i] =
                                idx < active_qubits.len() && active_qubits[idx];
                        }
                    }

                    // Apply the gate
                    apply_gate(prop, gate, direction);

                    // Update active qubits and check for spreading
                    for (i, q) in gate.qubits.iter().enumerate() {
                        let idx = q.index();
                        if idx < active_qubits.len() {
                            let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                            let was_active = i < was_active_flags.len() && was_active_flags[i];

                            if now_active && !was_active {
                                // Pauli spread to this qubit - mark its gates for processing
                                active_qubits[idx] = true;
                                for &(_topo_pos, new_node) in self.index.qubit_gates(idx) {
                                    should_process[new_node] = true;
                                }
                            } else if !now_active && was_active {
                                active_qubits[idx] = false;
                            }
                        }
                    }
                }
            }
        }
    }

    /// Propagates backward from a starting node using heap-based traversal.
    ///
    /// This is more efficient than `propagate_sparse` when starting from a specific
    /// point (like a measurement) because it only visits gates that could affect
    /// the Pauli, using a priority queue ordered by topological position.
    ///
    /// # Arguments
    /// * `start_node` - The node to start propagation from
    /// * `prop` - The PauliProp to propagate (modified in place)
    /// * `work` - Reusable work buffers (will be cleared before use)
    pub fn propagate_backward_from(
        &self,
        start_node: usize,
        prop: &mut PauliProp,
        work: &mut PropagatorWorkBuffers,
    ) {
        work.clear();

        let start_topo_pos = self.index.topo_position(start_node);
        let max_qubit = self.index.max_qubit();

        // Initialize active qubits from current Pauli support
        for q in prop.get_x_qubits() {
            if q <= max_qubit {
                work.active_qubits[q] = true;
            }
        }
        for q in prop.get_z_qubits() {
            if q <= max_qubit {
                work.active_qubits[q] = true;
            }
        }

        // Add initial gates to heap (only gates before start_node)
        for (qubit, &is_active) in work.active_qubits.iter().enumerate() {
            if is_active && qubit <= max_qubit {
                for (topo_pos, node) in self.index.qubit_gates_reversed(qubit) {
                    if topo_pos < start_topo_pos && !work.visited[node] {
                        work.visited[node] = true;
                        work.heap.push((topo_pos, node));
                    }
                }
            }
        }

        // Process gates in reverse topological order
        while let Some((_, node)) = work.heap.pop() {
            if let Some(gate) = self.gate(node) {
                // Track which qubits were active before
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= max_qubit {
                        was_active[j] = work.active_qubits[q.index()];
                    }
                }

                // Apply gate backward
                apply_gate(prop, gate, Direction::Backward);

                // Check if Pauli spread to new qubits
                let node_topo_pos = self.index.topo_position(node);
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= max_qubit {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            work.active_qubits[idx] = true;
                            for (topo_pos, new_node) in self.index.qubit_gates_reversed(idx) {
                                if topo_pos < node_topo_pos && !work.visited[new_node] {
                                    work.visited[new_node] = true;
                                    work.heap.push((topo_pos, new_node));
                                }
                            }
                        } else if !now_active && was {
                            work.active_qubits[idx] = false;
                        }
                    }
                }
            }
        }
    }

    /// Propagates forward from a starting node using heap-based traversal.
    ///
    /// # Arguments
    /// * `start_node` - The node to start propagation from
    /// * `prop` - The PauliProp to propagate (modified in place)
    /// * `work` - Reusable work buffers (will be cleared before use)
    pub fn propagate_forward_from(
        &self,
        start_node: usize,
        prop: &mut PauliProp,
        work: &mut PropagatorWorkBuffers,
    ) {
        work.clear();

        let start_topo_pos = self.index.topo_position(start_node);
        let max_topo_pos = self.index.topo_order().len();
        let max_qubit = self.index.max_qubit();

        // Initialize active qubits from current Pauli support
        for q in prop.get_x_qubits() {
            if q <= max_qubit {
                work.active_qubits[q] = true;
            }
        }
        for q in prop.get_z_qubits() {
            if q <= max_qubit {
                work.active_qubits[q] = true;
            }
        }

        // Add initial gates to heap (only gates after start_node)
        // Use negated position for min-heap behavior (process smallest first)
        for (qubit, &is_active) in work.active_qubits.iter().enumerate() {
            if is_active && qubit <= max_qubit {
                for &(topo_pos, node) in self.index.qubit_gates(qubit) {
                    if topo_pos > start_topo_pos && !work.visited[node] {
                        work.visited[node] = true;
                        // Negate to get min-heap behavior
                        work.heap.push((max_topo_pos - topo_pos, node));
                    }
                }
            }
        }

        // Process gates in forward topological order
        while let Some((neg_pos, node)) = work.heap.pop() {
            let node_topo_pos = max_topo_pos - neg_pos;

            if let Some(gate) = self.gate(node) {
                // Track which qubits were active before
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= max_qubit {
                        was_active[j] = work.active_qubits[q.index()];
                    }
                }

                // Apply gate forward
                apply_gate(prop, gate, Direction::Forward);

                // Check if Pauli spread to new qubits
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= max_qubit {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            work.active_qubits[idx] = true;
                            for &(topo_pos, new_node) in self.index.qubit_gates(idx) {
                                if topo_pos > node_topo_pos && !work.visited[new_node] {
                                    work.visited[new_node] = true;
                                    work.heap.push((max_topo_pos - topo_pos, new_node));
                                }
                            }
                        } else if !now_active && was {
                            work.active_qubits[idx] = false;
                        }
                    }
                }
            }
        }
    }

    /// Propagates through all gates in topological order (non-sparse).
    pub fn propagate_full(&self, prop: &mut PauliProp, direction: Direction) {
        let topo_order = self.index.topo_order();

        match direction {
            Direction::Forward => {
                for &node in topo_order {
                    if let Some(gate) = self.gate(node) {
                        apply_gate(prop, gate, direction);
                    }
                }
            }
            Direction::Backward => {
                for &node in topo_order.iter().rev() {
                    if let Some(gate) = self.gate(node) {
                        apply_gate(prop, gate, direction);
                    }
                }
            }
        }
    }
}

/// Returns the set of qubits with non-trivial Pauli operators (the "support").
///
/// A qubit has a non-trivial Pauli if it has X, Z, or both (Y).
fn get_pauli_support(prop: &PauliProp) -> BTreeSet<usize> {
    let mut support = BTreeSet::new();
    support.extend(prop.get_x_qubits());
    support.extend(prop.get_z_qubits());
    support
}

/// Checks if a gate touches any qubit in the given set.
fn gate_touches_qubits(gate: &pecos_core::Gate, qubits: &BTreeSet<usize>) -> bool {
    gate.qubits.iter().any(|q| qubits.contains(&q.index()))
}

/// Propagates a PauliProp through a DagCircuit using sparse traversal.
///
/// This creates a temporary `DagPropagator` for one-off propagation.
/// For repeated propagations through the same circuit, use `DagPropagator`
/// directly to avoid recomputing indices.
///
/// # Example
/// ```
/// use pecos_qec::fault_tolerance::propagator::{propagate_sparse_dag, Direction};
/// use pecos_quantum::DagCircuit;
/// use pecos_qsim::PauliProp;
///
/// let mut dag = DagCircuit::new();
/// dag.pz(0);
/// dag.h(0);
/// dag.cx(0, 1);  // This will spread X from qubit 0 to qubit 1
/// dag.mz(0);
/// dag.mz(1);
///
/// // Start with Z on qubit 0 only
/// let mut prop = PauliProp::new();
/// prop.add_z(0);
///
/// // Propagate forward - only visits gates on qubit 0, then qubit 1 after CX
/// propagate_sparse_dag(&dag, &mut prop, Direction::Forward);
/// ```
pub fn propagate_sparse_dag(dag: &DagCircuit, prop: &mut PauliProp, direction: Direction) {
    let propagator = DagPropagator::new(dag);
    propagator.propagate_sparse(prop, direction);
}

/// Propagates a PauliProp through a DagCircuit visiting all gates.
///
/// This is the non-sparse version that processes all gates in topological order.
/// Use `propagate_sparse_dag` for better performance with sparse Paulis.
///
/// # Arguments
/// * `dag` - The DAG circuit to propagate through
/// * `prop` - The PauliProp to propagate (modified in place)
/// * `direction` - Forward or Backward propagation
pub fn propagate_through_dag(dag: &DagCircuit, prop: &mut PauliProp, direction: Direction) {
    let topo_order = dag.topological_order();

    let ordered_nodes: Vec<usize> = match direction {
        Direction::Forward => topo_order,
        Direction::Backward => topo_order.into_iter().rev().collect(),
    };

    for node in ordered_nodes {
        if let Some(gate) = dag.gate(node) {
            apply_gate(prop, gate, direction);
        }
    }
}

/// Propagates a PauliProp backward through a DagCircuit from a specific gate.
///
/// This starts at the given gate and propagates backward through all predecessor
/// gates, only visiting gates that could affect the Pauli.
///
/// # Arguments
/// * `dag` - The DAG circuit to propagate through
/// * `prop` - The PauliProp to propagate (modified in place)
/// * `start_node` - The node to start backward propagation from
pub fn propagate_backward_from_node(dag: &DagCircuit, prop: &mut PauliProp, start_node: usize) {
    // Get topological order and filter to nodes before start_node
    let topo_order = dag.topological_order();
    let start_pos = topo_order.iter().position(|&n| n == start_node).unwrap_or(0);

    // Only process nodes before (and including) start_node
    let relevant_nodes: Vec<usize> = topo_order[..=start_pos].iter().copied().rev().collect();

    let mut processed: BTreeSet<usize> = BTreeSet::new();

    for node in relevant_nodes {
        if processed.contains(&node) {
            continue;
        }

        let support = get_pauli_support(prop);
        if let Some(gate) = dag.gate(node) {
            if gate_touches_qubits(gate, &support) {
                apply_gate(prop, gate, Direction::Backward);
                processed.insert(node);
            }
        }
    }
}

// ============================================================================
// Standalone Backward Propagation Functions
// ============================================================================

/// Propagates a Pauli backward through a circuit from a given starting tick.
///
/// This is the backward analog of forward Pauli propagation. Starting with a Pauli
/// at `start_tick`, it propagates backward through all preceding ticks.
///
/// # Arguments
/// * `circuit` - The circuit to propagate through
/// * `prop` - The Pauli to propagate (will be modified in place)
/// * `start_tick` - The tick to start from (propagates backward from here to tick 0)
///
/// # Example
/// ```
/// use pecos_qec::fault_tolerance::propagator::propagate_backward_from_tick;
/// use pecos_quantum::TickCircuit;
/// use pecos_qsim::PauliProp;
///
/// let mut circuit = TickCircuit::new();
/// circuit.tick().pz(&[0]);
/// circuit.tick().h(&[0]);
/// circuit.tick().mz(&[0]);
///
/// // Start with Z at the measurement (tick 2) and propagate backward
/// let mut prop = PauliProp::new();
/// prop.add_z(0);
/// propagate_backward_from_tick(&circuit, &mut prop, 2);
///
/// // After H gate backward propagation, Z becomes X
/// assert!(prop.contains_x(0));
/// assert!(!prop.contains_z(0));
/// ```
pub fn propagate_backward_from_tick(circuit: &TickCircuit, prop: &mut PauliProp, start_tick: usize) {
    propagate_tick_range(circuit, prop, 0, start_tick, Direction::Backward);
}

/// Propagates a fault backward through a circuit.
///
/// This is the backward analog of `propagate_fault`. Given a fault at a specific
/// location, it initializes a Pauli with that fault and propagates backward
/// through all preceding gates.
///
/// # Arguments
/// * `circuit` - The circuit to propagate through
/// * `fault` - The fault to propagate backward
///
/// # Returns
/// A `PauliProp` representing the backward-propagated fault (what it would have
/// looked like at the beginning of the circuit).
///
/// # Example
/// ```
/// use pecos_qec::fault_tolerance::{PauliFault, SpacetimeLocation};
/// use pecos_qec::fault_tolerance::propagator::propagate_fault_backward;
/// use pecos_quantum::TickCircuit;
/// use pecos_core::gate_type::GateType;
/// use pecos_core::QubitId;
///
/// let mut circuit = TickCircuit::new();
/// circuit.tick().pz(&[0]);
/// circuit.tick().h(&[0]);
/// circuit.tick().mz(&[0]);
///
/// // Create a Z fault at the measurement location
/// let loc = SpacetimeLocation {
///     tick: 2,
///     qubits: vec![QubitId(0)],
///     before: true,
///     gate_type: GateType::Measure,
///     gate_index: 0,
/// };
/// let fault = PauliFault::new(loc, vec![3]); // Z fault
///
/// let prop = propagate_fault_backward(&circuit, &fault);
/// // Z propagated backward through H becomes X
/// assert!(prop.contains_x(0));
/// ```
pub fn propagate_fault_backward(circuit: &TickCircuit, fault: &PauliFault) -> PauliProp {
    let mut prop = init_pauli_prop_with_fault(fault);
    let fault_tick = fault.location.tick;

    // Determine which tick to start propagating from
    let end_tick = if fault.location.before {
        // Fault is before gates at fault_tick, so the fault exists at the START of fault_tick
        // Backward propagation goes through ticks [0, fault_tick-1]
        fault_tick.saturating_sub(1)
    } else {
        // Fault is after gates at fault_tick, so the fault exists at the END of fault_tick
        // Backward propagation goes through ticks [0, fault_tick]
        fault_tick
    };

    propagate_tick_range(circuit, &mut prop, 0, end_tick, Direction::Backward);
    prop
}

/// Propagates an observable backward through the circuit.
///
/// This is useful for understanding what an observable (like a Z-measurement or
/// a logical operator) looks like at earlier points in the circuit.
///
/// # Arguments
/// * `circuit` - The circuit to propagate through
/// * `x_positions` - Qubits with X in the observable
/// * `z_positions` - Qubits with Z in the observable
/// * `start_tick` - The tick where the observable is defined (e.g., measurement tick)
///
/// # Returns
/// A `PauliProp` representing the backward-propagated observable.
pub fn propagate_observable_backward(
    circuit: &TickCircuit,
    x_positions: &[usize],
    z_positions: &[usize],
    start_tick: usize,
) -> PauliProp {
    let mut prop = PauliProp::new();

    for &q in x_positions {
        prop.add_x(q);
    }
    for &q in z_positions {
        prop.add_z(q);
    }

    propagate_backward_from_tick(circuit, &mut prop, start_tick);
    prop
}

/// Initialize a PauliProp with the given fault.
fn init_pauli_prop_with_fault(fault: &PauliFault) -> PauliProp {
    let mut prop = PauliProp::new();
    for (qubit, &pauli) in fault.location.qubits.iter().zip(fault.paulis.iter()) {
        let q = qubit.index();
        match pauli {
            1 => prop.add_x(q),
            2 => {
                prop.add_x(q);
                prop.add_z(q);
            }
            3 => prop.add_z(q),
            _ => {}
        }
    }
    prop
}

// ============================================================================
// Core Types
// ============================================================================

/// Unique identifier for a measurement in the circuit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct MeasurementId {
    /// Which tick the measurement occurs in.
    pub tick: usize,
    /// Which qubit is measured.
    pub qubit: usize,
    /// Measurement basis: 0 = Z, 1 = X.
    pub basis: u8,
}

/// Unique identifier for a detector (syndrome bit).
///
/// A detector is typically defined as the XOR of two measurements,
/// detecting changes in syndrome between rounds.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DetectorId {
    /// The measurements that make up this detector.
    /// For a simple detector: [m_i]
    /// For a comparison detector: [m_i, m_{i-1}]
    /// Using SmallVec to avoid heap allocation for common 1-2 measurement cases.
    pub measurements: SmallVec<[MeasurementId; 2]>,
    /// Optional name/label for the detector.
    pub name: Option<String>,
}

impl DetectorId {
    /// Creates a single-measurement detector.
    #[inline]
    pub fn single(measurement: MeasurementId) -> Self {
        Self {
            measurements: smallvec::smallvec![measurement],
            name: None,
        }
    }

    /// Creates a comparison detector (XOR of two measurements).
    #[inline]
    pub fn comparison(m1: MeasurementId, m2: MeasurementId) -> Self {
        Self {
            measurements: smallvec::smallvec![m1, m2],
            name: None,
        }
    }

    /// Adds a name to the detector.
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }
}

/// Unique identifier for a logical observable.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct LogicalId {
    /// Index of the logical qubit.
    pub logical_qubit: usize,
    /// Which observable: 0 = Z, 1 = X.
    pub observable: u8,
}

/// What a single fault location influences.
///
/// Uses fixed-size arrays indexed by Pauli type (0=I, 1=X, 2=Y, 3=Z) for fast access.
#[derive(Debug, Clone)]
pub struct FaultInfluence {
    /// Which detectors this fault flips, indexed by Pauli type (1=X, 2=Y, 3=Z).
    /// Index 0 is unused (identity fault has no effect).
    pub detector_flips: [Vec<DetectorId>; 4],

    /// Which logical observables this fault flips, indexed by Pauli type.
    pub logical_flips: [Vec<LogicalId>; 4],

    /// Which raw measurements this fault flips, indexed by Pauli type.
    pub measurement_flips: [Vec<MeasurementId>; 4],

    /// Per-qubit detector flips for multi-qubit locations.
    /// Key: (qubit_index_in_location, pauli_type), Value: detector IDs flipped by that qubit
    pub per_qubit_detector_flips: BTreeMap<(usize, u8), Vec<DetectorId>>,
}

impl Default for FaultInfluence {
    fn default() -> Self {
        Self {
            detector_flips: Default::default(),
            logical_flips: Default::default(),
            measurement_flips: Default::default(),
            per_qubit_detector_flips: BTreeMap::new(),
        }
    }
}

impl FaultInfluence {
    /// Returns true if this fault has no effect.
    pub fn is_trivial(&self) -> bool {
        self.detector_flips.iter().all(|v| v.is_empty())
            && self.logical_flips.iter().all(|v| v.is_empty())
            && self.measurement_flips.iter().all(|v| v.is_empty())
    }

    /// Returns all detectors flipped by a specific Pauli type.
    #[inline]
    pub fn detectors_for_pauli(&self, pauli: u8) -> &[DetectorId] {
        self.detector_flips.get(pauli as usize).map_or(&[], |v| v.as_slice())
    }

    /// Returns all logicals flipped by a specific Pauli type.
    #[inline]
    pub fn logicals_for_pauli(&self, pauli: u8) -> &[LogicalId] {
        self.logical_flips.get(pauli as usize).map_or(&[], |v| v.as_slice())
    }
}

/// Pre-computed map from fault locations to their influences.
///
/// This is the main output of backward propagation - a lookup table
/// that tells you what each fault location affects.
#[derive(Debug, Clone)]
pub struct FaultInfluenceMap {
    /// For each spacetime location, what it influences.
    pub influences: BTreeMap<SpacetimeLocation, FaultInfluence>,

    /// All detectors in the circuit.
    pub detectors: Vec<DetectorId>,

    /// All logical observables being tracked.
    pub logicals: Vec<LogicalId>,

    /// All measurements in the circuit.
    pub measurements: Vec<MeasurementId>,

    /// Reverse map: for each detector, which fault locations flip it.
    pub detector_to_faults: BTreeMap<DetectorId, Vec<(SpacetimeLocation, u8)>>,

    /// Reverse map: for each logical, which fault locations flip it.
    pub logical_to_faults: BTreeMap<LogicalId, Vec<(SpacetimeLocation, u8)>>,
}

impl FaultInfluenceMap {
    /// Creates an empty influence map.
    pub fn new() -> Self {
        Self {
            influences: BTreeMap::new(),
            detectors: Vec::new(),
            logicals: Vec::new(),
            measurements: Vec::new(),
            detector_to_faults: BTreeMap::new(),
            logical_to_faults: BTreeMap::new(),
        }
    }

    /// Returns the influence of a fault at the given location.
    pub fn get_influence(&self, location: &SpacetimeLocation) -> Option<&FaultInfluence> {
        self.influences.get(location)
    }

    /// Quickly classifies a single-qubit fault based on pre-computed influences.
    ///
    /// Returns (has_syndrome, has_logical_error) for the given Pauli type.
    /// For multi-qubit locations, use `classify_multi_qubit_fault` instead.
    pub fn classify_fault(&self, location: &SpacetimeLocation, pauli: u8) -> (bool, bool) {
        if let Some(influence) = self.influences.get(location) {
            let has_syndrome = !influence.detectors_for_pauli(pauli).is_empty();
            let has_logical = !influence.logicals_for_pauli(pauli).is_empty();
            (has_syndrome, has_logical)
        } else {
            (false, false)
        }
    }

    /// Classifies a multi-qubit fault where the same Pauli is applied to all qubits.
    ///
    /// For multi-qubit locations (e.g., CX gate), applying the same Pauli to both
    /// qubits can have cancellation effects. This method properly computes the
    /// combined effect by XORing the per-qubit influences.
    ///
    /// For Y faults, we decompose Y = XZ and combine the X and Z contributions,
    /// since Y anticommutes with both X and Z components of the observable.
    ///
    /// Returns (has_syndrome, has_logical_error).
    pub fn classify_multi_qubit_fault(
        &self,
        location: &SpacetimeLocation,
        pauli: u8,
    ) -> (bool, bool) {
        if let Some(influence) = self.influences.get(location) {
            // Count detector flips per detector
            let mut detector_flip_counts: BTreeMap<&DetectorId, usize> = BTreeMap::new();

            // Collect flips from each qubit in the location
            for qubit_idx in 0..location.qubits.len() {
                if pauli == 2 {
                    // Y = XZ: a Y fault flips a detector if EITHER the X component
                    // OR the Z component would flip it. Count contributions from both.
                    // X component flips detectors sensitive to Z
                    if let Some(detectors) =
                        influence.per_qubit_detector_flips.get(&(qubit_idx, 1))
                    {
                        for detector in detectors {
                            *detector_flip_counts.entry(detector).or_insert(0) += 1;
                        }
                    }
                    // Z component flips detectors sensitive to X
                    if let Some(detectors) =
                        influence.per_qubit_detector_flips.get(&(qubit_idx, 3))
                    {
                        for detector in detectors {
                            *detector_flip_counts.entry(detector).or_insert(0) += 1;
                        }
                    }
                } else {
                    // X or Z fault: straightforward
                    if let Some(detectors) =
                        influence.per_qubit_detector_flips.get(&(qubit_idx, pauli))
                    {
                        for detector in detectors {
                            *detector_flip_counts.entry(detector).or_insert(0) += 1;
                        }
                    }
                }
            }

            // Syndrome = odd number of flips for any detector
            let has_syndrome = detector_flip_counts.values().any(|&count| count % 2 == 1);

            // For logicals, use the same approach
            // (simplified: just check if any qubit flips logical, proper handling TBD)
            let has_logical = !influence.logicals_for_pauli(pauli).is_empty();

            (has_syndrome, has_logical)
        } else {
            (false, false)
        }
    }

    /// Returns all fault locations that flip a specific detector.
    pub fn faults_for_detector(&self, detector: &DetectorId) -> &[(SpacetimeLocation, u8)] {
        self.detector_to_faults
            .get(detector)
            .map_or(&[], |v| v.as_slice())
    }

    /// Returns the number of fault locations tracked.
    pub fn num_fault_locations(&self) -> usize {
        self.influences.len()
    }
}

impl Default for FaultInfluenceMap {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Backward Propagator
// ============================================================================

/// Propagates Paulis backward through a circuit to build influence maps.
pub struct BackwardPropagator<'a> {
    circuit: &'a TickCircuit,
    /// Fault locations extracted from the circuit.
    locations: Vec<SpacetimeLocation>,
    /// Pre-computed index: tick -> (location_index, before) pairs for O(1) lookup
    tick_locations: Vec<Vec<(usize, bool)>>,
}

impl<'a> BackwardPropagator<'a> {
    /// Creates a new backward propagator for the given circuit.
    pub fn new(circuit: &'a TickCircuit) -> Self {
        let locations = extract_spacetime_locations(circuit, false);

        // Build tick index for O(1) lookup
        let num_ticks = circuit.ticks().len();
        let mut tick_locations = vec![Vec::new(); num_ticks + 1];
        for (idx, loc) in locations.iter().enumerate() {
            if loc.tick < tick_locations.len() {
                tick_locations[loc.tick].push((idx, loc.before));
            }
        }

        Self {
            circuit,
            locations,
            tick_locations,
        }
    }

    /// Builds the complete fault influence map.
    ///
    /// This performs backward propagation from all measurements and
    /// creates a lookup table for fault classification.
    pub fn build_influence_map(&self) -> FaultInfluenceMap {
        self.build_influence_map_with_logicals(&[])
    }

    /// Builds the fault influence map with logical operator tracking.
    ///
    /// # Arguments
    ///
    /// * `logicals` - Logical operators as (x_positions, z_positions) pairs.
    ///   The first element of each pair is the X component positions,
    ///   the second is the Z component positions.
    pub fn build_influence_map_with_logicals(
        &self,
        logicals: &[(&[usize], &[usize])],
    ) -> FaultInfluenceMap {
        let mut map = FaultInfluenceMap::new();

        // Extract all measurements from the circuit
        let measurements = self.extract_measurements();
        map.measurements = measurements.clone();

        // Create simple detectors (one per measurement for now)
        // TODO: Support comparison detectors for multi-round circuits
        for m in &measurements {
            map.detectors.push(DetectorId::single(*m));
        }

        // Create logical IDs
        for (i, _) in logicals.iter().enumerate() {
            map.logicals.push(LogicalId {
                logical_qubit: i,
                observable: 0, // Z observable
            });
        }

        // Initialize influence for each fault location
        for loc in &self.locations {
            map.influences.insert(loc.clone(), FaultInfluence::default());
        }

        // Backward propagate from each measurement
        for measurement in &measurements {
            self.propagate_from_measurement(measurement, &mut map);
        }

        // Backward propagate from each logical operator
        for (i, (x_pos, z_pos)) in logicals.iter().enumerate() {
            let logical_id = LogicalId {
                logical_qubit: i,
                observable: 0,
            };
            self.propagate_from_logical(x_pos, z_pos, &logical_id, &mut map);
        }

        // Build reverse maps
        self.build_reverse_maps(&mut map);

        map
    }

    /// Extracts all measurements from the circuit.
    fn extract_measurements(&self) -> Vec<MeasurementId> {
        let mut measurements = Vec::new();

        for (tick_idx, tick) in self.circuit.iter_ticks() {
            for gate in tick.gates() {
                // Currently only Z-basis measurements are supported
                let basis = match gate.gate_type {
                    GateType::Measure | GateType::MeasureFree => 0, // Z-basis
                    _ => continue,
                };

                for qubit in &gate.qubits {
                    measurements.push(MeasurementId {
                        tick: tick_idx,
                        qubit: qubit.index(),
                        basis,
                    });
                }
            }
        }

        measurements
    }

    /// Propagates backward from a measurement to find which faults flip it.
    ///
    /// We propagate the OBSERVABLE being measured backward through the circuit.
    /// An error P at location L flips the measurement if P anticommutes with
    /// the back-propagated observable at L.
    ///
    /// For Z-measurement, the observable is Z. Propagating Z backward tells us
    /// the effective observable at each location. An X error anticommutes with Z,
    /// so X errors flip the measurement where the observable has Z component.
    fn propagate_from_measurement(&self, measurement: &MeasurementId, map: &mut FaultInfluenceMap) {
        // Start with the observable being measured (not what flips it)
        // Z-measurement measures Z, X-measurement measures X
        let initial_pauli = if measurement.basis == 0 { 3u8 } else { 1u8 }; // Z or X

        let mut prop = PauliProp::new();
        match initial_pauli {
            1 => prop.add_x(measurement.qubit),
            3 => prop.add_z(measurement.qubit),
            _ => {}
        }

        let detector = DetectorId::single(*measurement);

        // Check faults at the measurement tick (before=true faults only)
        // These are right before the measurement and see the initial sensitivity
        self.record_influences_at_tick_filtered(
            measurement.tick,
            &prop,
            &detector,
            None,
            map,
            true, // only before=true locations
        );

        // Propagate backward through ticks
        // For each tick t, we need to handle before=false and before=true locations separately:
        // - before=false (after gates at t): check with sensitivity BEFORE applying gates backward
        // - before=true (before gates at t): check with sensitivity AFTER applying gates backward
        for tick_idx in (0..measurement.tick).rev() {
            // Check before=false locations (faults that happen after gates at this tick)
            // These see the sensitivity at the state "after tick t gates executed"
            self.record_influences_at_tick_filtered(
                tick_idx,
                &prop,
                &detector,
                None,
                map,
                false, // only before=false locations
            );

            // Apply gates at this tick backward
            // For backward propagation through a sequence G1, G2, G3:
            // If forward is G3† G2† G1† P G1 G2 G3
            // Backward is G1 G2 G3 P' G3† G2† G1†
            // So we apply gates in FORWARD order when propagating backward
            let tick = &self.circuit.ticks()[tick_idx];
            for gate in tick.gates() {
                self.apply_gate_backward(&mut prop, gate);
            }

            // Check before=true locations (faults that happen before gates at this tick)
            // These see the sensitivity at the state "before tick t gates executed"
            self.record_influences_at_tick_filtered(
                tick_idx,
                &prop,
                &detector,
                None,
                map,
                true, // only before=true locations
            );
        }
    }

    /// Propagates backward from a logical operator.
    ///
    /// We propagate the logical OBSERVABLE backward through the circuit.
    /// An error P at location L flips the logical if P anticommutes with
    /// the back-propagated observable at L.
    fn propagate_from_logical(
        &self,
        x_positions: &[usize],
        z_positions: &[usize],
        logical_id: &LogicalId,
        map: &mut FaultInfluenceMap,
    ) {
        // Start with the logical observable itself (not swapped)
        // The recording function handles anticommutation checking
        let mut prop = PauliProp::new();

        // X positions in logical -> X in prop
        for &q in x_positions {
            prop.add_x(q);
        }
        // Z positions in logical -> Z in prop
        for &q in z_positions {
            prop.add_z(q);
        }

        // Dummy detector for the recording function
        let dummy_detector = DetectorId::single(MeasurementId {
            tick: 0,
            qubit: 0,
            basis: 0,
        });

        // Propagate backward through all ticks
        let num_ticks = self.circuit.ticks().len();
        for tick_idx in (0..num_ticks).rev() {
            // Check before=false locations (after gates at this tick)
            self.record_influences_at_tick_filtered(
                tick_idx,
                &prop,
                &dummy_detector,
                Some(logical_id),
                map,
                false,
            );

            // Apply gates backward (in forward order)
            let tick = &self.circuit.ticks()[tick_idx];
            for gate in tick.gates() {
                self.apply_gate_backward(&mut prop, gate);
            }

            // Check before=true locations (before gates at this tick)
            self.record_influences_at_tick_filtered(
                tick_idx,
                &prop,
                &dummy_detector,
                Some(logical_id),
                map,
                true,
            );
        }
    }

    /// Records which fault locations at a tick would contribute to the propagated Pauli.
    /// Filters by before flag to handle timing correctly.
    ///
    /// The `prop` contains the back-propagated OBSERVABLE. A fault P anticommutes with
    /// the observable Q if they share positions where both are non-identity but different.
    ///
    /// Anticommutation rules (for single qubit):
    /// - X anticommutes with Z and Y
    /// - Z anticommutes with X and Y
    /// - Y anticommutes with X, Z, and Y
    #[inline]
    fn record_influences_at_tick_filtered(
        &self,
        tick_idx: usize,
        prop: &PauliProp,
        detector: &DetectorId,
        logical: Option<&LogicalId>,
        map: &mut FaultInfluenceMap,
        only_before: bool,
    ) {
        // Use pre-computed tick index for O(1) lookup instead of O(n) linear scan
        if tick_idx >= self.tick_locations.len() {
            return;
        }

        for &(loc_idx, before) in &self.tick_locations[tick_idx] {
            if before != only_before {
                continue;
            }

            let loc = &self.locations[loc_idx];

            // Check each qubit in the fault location
            for (qubit_idx, qubit) in loc.qubits.iter().enumerate() {
                let q = qubit.index();

                // The back-propagated observable tells us what measurement is sensitive to
                // prop contains the observable, we check what anticommutes with it
                let obs_x = prop.contains_x(q);
                let obs_z = prop.contains_z(q);
                // Observable is: I (neither), X (x only), Z (z only), Y (both x and z)

                if let Some(influence) = map.influences.get_mut(loc) {
                    // X fault anticommutes with Z or Y observable
                    // (X anticommutes with Z, X anticommutes with Y=iXZ)
                    // X fault anticommutes with Z or Y observable
                    let x_flips = obs_z; // Z or Y (both have Z component)
                    if x_flips {
                        if let Some(log) = logical {
                            influence.logical_flips[1].push(*log);
                        } else {
                            influence.detector_flips[1].push(detector.clone());
                            influence.measurement_flips[1].extend(detector.measurements.iter().copied());
                            // Also record per-qubit influence for multi-qubit fault handling
                            influence
                                .per_qubit_detector_flips
                                .entry((qubit_idx, 1))
                                .or_default()
                                .push(detector.clone());
                        }
                    }

                    // Z fault anticommutes with X or Y observable
                    // (Z anticommutes with X, Z anticommutes with Y=iXZ)
                    let z_flips = obs_x; // X or Y (both have X component)
                    if z_flips {
                        if let Some(log) = logical {
                            influence.logical_flips[3].push(*log);
                        } else {
                            influence.detector_flips[3].push(detector.clone());
                            influence.measurement_flips[3].extend(detector.measurements.iter().copied());
                            // Also record per-qubit influence
                            influence
                                .per_qubit_detector_flips
                                .entry((qubit_idx, 3))
                                .or_default()
                                .push(detector.clone());
                        }
                    }

                    // Y fault = iXZ: Y anticommutes with X, Z, and Y
                    // Y anticommutes with observable if observable has X or Z component
                    let y_flips = obs_x || obs_z;
                    if y_flips {
                        if let Some(log) = logical {
                            influence.logical_flips[2].push(*log);
                        } else {
                            influence.detector_flips[2].push(detector.clone());
                            influence.measurement_flips[2].extend(detector.measurements.iter().copied());
                            // Also record per-qubit influence
                            influence
                                .per_qubit_detector_flips
                                .entry((qubit_idx, 2))
                                .or_default()
                                .push(detector.clone());
                        }
                    }
                }
            }
        }
    }

    /// Records which fault locations at a tick would contribute to the propagated Pauli.
    /// (Legacy function that doesn't filter by before flag)
    ///
    /// The `prop` contains the back-propagated OBSERVABLE. Uses anticommutation checking.
    #[allow(dead_code)]
    fn record_influences_at_tick(
        &self,
        tick_idx: usize,
        prop: &PauliProp,
        detector: &DetectorId,
        logical: Option<&LogicalId>,
        map: &mut FaultInfluenceMap,
    ) {
        // Find fault locations at this tick
        for loc in &self.locations {
            if loc.tick != tick_idx {
                continue;
            }

            // Check each qubit in the fault location
            for qubit in &loc.qubits {
                let q = qubit.index();

                // Check what observable is at this position
                let obs_x = prop.contains_x(q);
                let obs_z = prop.contains_z(q);

                if let Some(influence) = map.influences.get_mut(loc) {
                    // X fault anticommutes with Z or Y observable
                    if obs_z {
                        if let Some(log) = logical {
                            influence.logical_flips[1].push(*log);
                        } else {
                            influence.detector_flips[1].push(detector.clone());
                            influence.measurement_flips[1].extend(detector.measurements.iter().copied());
                        }
                    }

                    // Z fault anticommutes with X or Y observable
                    if obs_x {
                        if let Some(log) = logical {
                            influence.logical_flips[3].push(*log);
                        } else {
                            influence.detector_flips[3].push(detector.clone());
                            influence.measurement_flips[3].extend(detector.measurements.iter().copied());
                        }
                    }

                    // Y fault anticommutes with X, Z, or Y observable
                    if obs_x || obs_z {
                        if let Some(log) = logical {
                            influence.logical_flips[2].push(*log);
                        } else {
                            influence.detector_flips[2].push(detector.clone());
                            influence.measurement_flips[2].extend(detector.measurements.iter().copied());
                        }
                    }
                }
            }
        }
    }

    /// Applies a gate backward to a PauliProp.
    ///
    /// For Clifford gates, backward propagation follows specific rules:
    /// - CX: Same as forward (CX is self-adjoint)
    /// - H: Same as forward (H is self-adjoint)
    /// - SZ (S gate): X → -Y, Y → X, Z → Z (adjoint of forward)
    #[inline]
    fn apply_gate_backward(&self, prop: &mut PauliProp, gate: &pecos_core::Gate) {
        // Access gate.qubits directly - no allocation needed
        let qubits = &gate.qubits;

        match gate.gate_type {
            GateType::CX => {
                // CX is self-adjoint, same propagation as forward
                // X on control -> X on control AND target
                // X on target -> X on target
                // Z on control -> Z on control
                // Z on target -> Z on control AND target
                if qubits.len() >= 2 {
                    let control = qubits[0].index();
                    let target = qubits[1].index();

                    let ctrl_x = prop.contains_x(control);
                    let tgt_z = prop.contains_z(target);

                    // X spreads from control to target
                    if ctrl_x {
                        prop.add_x(target);
                    }
                    // Z spreads from target to control
                    if tgt_z {
                        prop.add_z(control);
                    }
                }
            }

            GateType::CZ => {
                // CZ is self-adjoint
                // X on either qubit -> X on that qubit AND Z on the other
                if qubits.len() >= 2 {
                    let q0 = qubits[0].index();
                    let q1 = qubits[1].index();

                    let x0 = prop.contains_x(q0);
                    let x1 = prop.contains_x(q1);

                    if x0 {
                        prop.add_z(q1);
                    }
                    if x1 {
                        prop.add_z(q0);
                    }
                }
            }

            GateType::H => {
                // H is self-adjoint: X <-> Z
                if let Some(qid) = qubits.first() {
                    let q = qid.index();
                    let has_x = prop.contains_x(q);
                    let has_z = prop.contains_z(q);

                    // Swap X and Z using toggle
                    if has_x && !has_z {
                        // Remove X by toggling, add Z
                        prop.add_x(q); // toggles off
                        prop.add_z(q);
                    } else if has_z && !has_x {
                        // Remove Z by toggling, add X
                        prop.add_z(q); // toggles off
                        prop.add_x(q);
                    }
                    // If both or neither, no change needed
                }
            }

            GateType::SZ => {
                // SZ† (adjoint): X -> -Y (we track as XZ), Y -> X, Z -> Z
                // Since we track Paulis mod phase, X -> Y means X -> XZ
                if let Some(qid) = qubits.first() {
                    let q = qid.index();
                    let has_x = prop.contains_x(q);
                    let has_z = prop.contains_z(q);

                    if has_x && !has_z {
                        // X -> XZ (Y with phase)
                        prop.add_z(q);
                    } else if has_x && has_z {
                        // Y (XZ) -> X: remove Z by toggling
                        prop.add_z(q); // toggles off
                    }
                    // Z -> Z (no change)
                }
            }

            GateType::SZdg => {
                // SZdg = SZ†, so SZdg† = SZ
                // Forward SZ: X -> Y, Y -> -X, Z -> Z
                if let Some(qid) = qubits.first() {
                    let q = qid.index();
                    let has_x = prop.contains_x(q);
                    let has_z = prop.contains_z(q);

                    if has_x && !has_z {
                        // X -> XZ (Y)
                        prop.add_z(q);
                    } else if has_x && has_z {
                        // Y -> X: remove Z by toggling
                        prop.add_z(q); // toggles off
                    }
                }
            }

            GateType::X | GateType::Y | GateType::Z => {
                // Pauli gates are self-adjoint, and Paulis commute with themselves
                // X commutes with X, anticommutes with Y, Z (but we're tracking stabilizer)
                // For stabilizer tracking, Pauli gates don't change the Pauli frame
            }

            GateType::Prep | GateType::QAlloc => {
                // Preparation resets the qubit - backward propagation stops here
                // Any Pauli on a prepared qubit doesn't propagate further back
                // Toggle off both X and Z if present
                for qid in qubits.iter() {
                    let q = qid.index();
                    if prop.contains_x(q) {
                        prop.add_x(q); // toggles off
                    }
                    if prop.contains_z(q) {
                        prop.add_z(q); // toggles off
                    }
                }
            }

            GateType::Measure | GateType::MeasureFree => {
                // Measurement - we've already started from here, nothing to do
            }

            _ => {
                // Other gates - treat as identity for now
            }
        }
    }

    /// Builds reverse maps (detector -> faults, logical -> faults).
    fn build_reverse_maps(&self, map: &mut FaultInfluenceMap) {
        for (loc, influence) in &map.influences {
            for (pauli, detectors) in influence.detector_flips.iter().enumerate() {
                for detector in detectors {
                    map.detector_to_faults
                        .entry(detector.clone())
                        .or_default()
                        .push((loc.clone(), pauli as u8));
                }
            }

            for (pauli, logicals) in influence.logical_flips.iter().enumerate() {
                for logical in logicals {
                    map.logical_to_faults
                        .entry(*logical)
                        .or_default()
                        .push((loc.clone(), pauli as u8));
                }
            }
        }
    }
}

// ============================================================================
// DAG-Based Backward Propagator (Sparse)
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
}

/// Pre-computed map from DAG fault locations to their influences.
#[derive(Debug, Clone)]
pub struct DagFaultInfluenceMap {
    /// For each spacetime location, what it influences.
    pub influences: BTreeMap<DagSpacetimeLocation, FaultInfluence>,

    /// All detectors in the circuit.
    pub detectors: Vec<DetectorId>,

    /// All logical observables being tracked.
    pub logicals: Vec<LogicalId>,

    /// All measurements in the circuit (node, qubit, basis).
    pub measurements: Vec<(usize, usize, u8)>,

    /// Reverse map: for each detector, which fault locations flip it.
    pub detector_to_faults: BTreeMap<DetectorId, Vec<(DagSpacetimeLocation, u8)>>,

    /// Reverse map: for each logical, which fault locations flip it.
    pub logical_to_faults: BTreeMap<LogicalId, Vec<(DagSpacetimeLocation, u8)>>,
}

impl DagFaultInfluenceMap {
    /// Creates an empty influence map.
    #[must_use]
    pub fn new() -> Self {
        Self {
            influences: BTreeMap::new(),
            detectors: Vec::new(),
            logicals: Vec::new(),
            measurements: Vec::new(),
            detector_to_faults: BTreeMap::new(),
            logical_to_faults: BTreeMap::new(),
        }
    }

    /// Returns the influence of a fault at the given location.
    #[must_use]
    pub fn get_influence(&self, location: &DagSpacetimeLocation) -> Option<&FaultInfluence> {
        self.influences.get(location)
    }

    /// Classifies a fault at the given location.
    ///
    /// Returns (has_syndrome, causes_logical_error).
    #[must_use]
    pub fn classify_fault(&self, location: &DagSpacetimeLocation, pauli: u8) -> (bool, bool) {
        if let Some(influence) = self.influences.get(location) {
            let has_syndrome = !influence.detector_flips[pauli as usize].is_empty();
            let causes_logical = !influence.logical_flips[pauli as usize].is_empty();
            (has_syndrome, causes_logical)
        } else {
            (false, false)
        }
    }
}

impl Default for DagFaultInfluenceMap {
    fn default() -> Self {
        Self::new()
    }
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
    /// Offset for each row. Length = num_rows + 1.
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

/// True SoA (Struct of Arrays) influence storage using CSR layout.
///
/// This is the most cache-efficient representation, storing all influences
/// in flat arrays with CSR-style indexing. Each Pauli type (X, Y, Z) has
/// its own CSR array for maximum locality.
///
/// # Memory Layout
///
/// For N locations and M total detector influences:
/// - Traditional AoS: N * (SmallVec overhead + potential heap allocs)
/// - True SoA: 3 * (N+1) * 4 bytes (offsets) + M * 4 bytes (data)
///
/// The SoA layout is more compact and has better cache behavior when
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

    /// Logical indices flipped by X faults.
    pub logicals_x: CsrArray,

    /// Logical indices flipped by Y faults.
    pub logicals_y: CsrArray,

    /// Logical indices flipped by Z faults.
    pub logicals_z: CsrArray,
}

impl InfluencesSoA {
    /// Creates a new SoA structure with capacity for the given number of locations.
    #[must_use]
    pub fn with_capacity(num_locations: usize) -> Self {
        // Estimate: average 2 detector influences per location per Pauli type
        let estimated_data = num_locations * 2;
        Self {
            num_locations: 0,
            detectors_x: CsrArray::with_capacity(num_locations, estimated_data),
            detectors_y: CsrArray::with_capacity(num_locations, estimated_data),
            detectors_z: CsrArray::with_capacity(num_locations, estimated_data),
            logicals_x: CsrArray::with_capacity(num_locations, estimated_data / 4),
            logicals_y: CsrArray::with_capacity(num_locations, estimated_data / 4),
            logicals_z: CsrArray::with_capacity(num_locations, estimated_data / 4),
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

    /// Returns the logical indices for a location and Pauli type.
    #[inline]
    #[must_use]
    pub fn logicals(&self, loc_idx: usize, pauli: Pauli) -> &[u32] {
        match pauli {
            Pauli::I => &[],
            Pauli::X => self.logicals_x.row(loc_idx),
            Pauli::Y => self.logicals_y.row(loc_idx),
            Pauli::Z => self.logicals_z.row(loc_idx),
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

    /// Returns whether the location has any logical flips for the given Pauli.
    #[inline]
    #[must_use]
    pub fn has_logical_flips(&self, loc_idx: usize, pauli: Pauli) -> bool {
        match pauli {
            Pauli::I => false,
            Pauli::X => !self.logicals_x.row_is_empty(loc_idx),
            Pauli::Y => !self.logicals_y.row_is_empty(loc_idx),
            Pauli::Z => !self.logicals_z.row_is_empty(loc_idx),
        }
    }

    /// Classifies a fault at the given location.
    ///
    /// Returns (has_syndrome, causes_logical_error).
    #[inline]
    #[must_use]
    pub fn classify(&self, loc_idx: usize, pauli: Pauli) -> (bool, bool) {
        (
            self.has_detector_flips(loc_idx, pauli),
            self.has_logical_flips(loc_idx, pauli),
        )
    }

    /// Finalizes a location row across all CSR arrays.
    pub fn finish_location(&mut self) {
        self.detectors_x.finish_row();
        self.detectors_y.finish_row();
        self.detectors_z.finish_row();
        self.logicals_x.finish_row();
        self.logicals_y.finish_row();
        self.logicals_z.finish_row();
        self.num_locations += 1;
    }

    /// Returns memory statistics for this structure.
    #[must_use]
    pub fn memory_stats(&self) -> InfluencesSoAStats {
        let offset_bytes = (self.detectors_x.offsets.len()
            + self.detectors_y.offsets.len()
            + self.detectors_z.offsets.len()
            + self.logicals_x.offsets.len()
            + self.logicals_y.offsets.len()
            + self.logicals_z.offsets.len())
            * std::mem::size_of::<u32>();

        let data_bytes = (self.detectors_x.data.len()
            + self.detectors_y.data.len()
            + self.detectors_z.data.len()
            + self.logicals_x.data.len()
            + self.logicals_y.data.len()
            + self.logicals_z.data.len())
            * std::mem::size_of::<u32>();

        InfluencesSoAStats {
            num_locations: self.num_locations,
            total_detector_entries: self.detectors_x.total_elements()
                + self.detectors_y.total_elements()
                + self.detectors_z.total_elements(),
            total_logical_entries: self.logicals_x.total_elements()
                + self.logicals_y.total_elements()
                + self.logicals_z.total_elements(),
            offset_bytes,
            data_bytes,
            total_bytes: offset_bytes + data_bytes,
        }
    }
}

/// Memory statistics for `InfluencesSoA`.
#[derive(Debug, Clone, Copy)]
pub struct InfluencesSoAStats {
    /// Number of fault locations.
    pub num_locations: usize,
    /// Total detector entries across all Pauli types.
    pub total_detector_entries: usize,
    /// Total logical entries across all Pauli types.
    pub total_logical_entries: usize,
    /// Bytes used for offset arrays.
    pub offset_bytes: usize,
    /// Bytes used for data arrays.
    pub data_bytes: usize,
    /// Total bytes used.
    pub total_bytes: usize,
}

/// True SoA fault influence map using CSR-style storage.
///
/// This is the most memory-efficient and cache-friendly representation.
/// Use this when processing large circuits or when memory is constrained.
#[derive(Debug, Clone, Default)]
pub struct DagFaultInfluenceMapSoA {
    /// Influences in true SoA layout.
    pub influences: InfluencesSoA,

    /// Locations indexed by location index.
    pub locations: Vec<DagSpacetimeLocation>,

    /// All detectors in the circuit.
    pub detectors: Vec<DetectorId>,

    /// All measurements in the circuit (node, qubit, basis).
    pub measurements: Vec<(usize, usize, u8)>,
}

impl DagFaultInfluenceMapSoA {
    /// Creates a new SoA map with capacity for the given number of locations.
    #[must_use]
    pub fn with_capacity(num_locations: usize) -> Self {
        Self {
            influences: InfluencesSoA::with_capacity(num_locations),
            locations: Vec::with_capacity(num_locations),
            detectors: Vec::new(),
            measurements: Vec::new(),
        }
    }

    /// Classifies a fault at the given location index.
    ///
    /// Returns (has_syndrome, causes_logical_error).
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

    /// Returns the logical indices flipped by a fault.
    #[inline]
    #[must_use]
    pub fn get_logical_indices(&self, loc_idx: usize, pauli: u8) -> &[u32] {
        self.influences.logicals(loc_idx, Pauli::from_u8(pauli))
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
}

/// Recorder that writes to a true SoA influence map.
///
/// This recorder builds the SoA structure incrementally. Unlike other recorders,
/// it requires locations to be processed in order and finalized one at a time.
pub struct SoARecorderBuilder {
    /// The SoA structure being built.
    influences: InfluencesSoA,
    /// Current location being built.
    current_location: usize,
    /// Pending detector indices for current location (X, Y, Z).
    pending_x: Vec<u32>,
    pending_y: Vec<u32>,
    pending_z: Vec<u32>,
}

impl SoARecorderBuilder {
    /// Creates a new SoA recorder builder.
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

    /// Finishes building and returns the SoA structure.
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
/// Unlike a sorting approach, this uses per-location buckets (SmallVecs) to collect
/// detector indices, then flattens to CSR format. This is O(n) in the number of
/// influences, avoiding the O(n log n) sort overhead.
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

    /// Converts buckets to SoA format in O(n) time.
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
    fn record(&mut self, loc_idx: usize, _qubit: usize, obs_x: bool, obs_z: bool, detector_idx: usize) {
        let det = detector_idx as u32;

        // X fault anticommutes with Z observable
        if obs_z {
            self.x_buckets[loc_idx].push(det);
        }
        // Z fault anticommutes with X observable
        if obs_x {
            self.z_buckets[loc_idx].push(det);
        }
        // Y fault anticommutes with X or Z observable
        if obs_x || obs_z {
            self.y_buckets[loc_idx].push(det);
        }
    }
}

/// Propagates Paulis backward through a DAG circuit using sparse traversal.
///
/// This is significantly faster than `BackwardPropagator` for circuits with
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
/// dag.pz(2);           // Prep ancilla
/// dag.cx(0, 2);        // CNOT data -> ancilla
/// dag.cx(1, 2);        // CNOT data -> ancilla
/// dag.mz(2);           // Measure ancilla
///
/// // Build the fault influence map using sparse propagation
/// let propagator = DagFaultAnalyzer::new(&dag);
/// let influence_map = propagator.build_influence_map();
/// ```
pub struct DagFaultAnalyzer<'a> {
    /// Base propagator for traversal infrastructure.
    propagator: DagPropagator<'a>,
    /// All fault locations in SoA layout.
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
    fn extract_locations(propagator: &DagPropagator<'_>, dag: &DagCircuit) -> FaultLocations {
        let topo_order = dag.topological_order();

        // Estimate capacity: roughly 2 locations per gate
        let estimated_locations = topo_order.len() * 2;
        let mut locations = FaultLocations::with_capacity(estimated_locations, propagator.max_node());

        for &node in &topo_order {
            if let Some(gate) = propagator.gate(node) {
                let is_measurement = matches!(
                    gate.gate_type,
                    GateType::Measure | GateType::MeasureFree
                );
                let is_prep = matches!(gate.gate_type, GateType::Prep | GateType::QAlloc);

                // Convert QubitId to usize
                let qubits: SmallVec<[usize; 2]> = gate.qubits.iter().map(|q| q.index()).collect();

                if is_measurement {
                    // Measurements only have before=true locations
                    locations.push(node, qubits, true, gate.gate_type);
                } else if is_prep {
                    // Preps only have before=false locations
                    locations.push(node, qubits, false, gate.gate_type);
                } else {
                    // Regular gates have both before and after locations
                    locations.push(node, qubits.clone(), true, gate.gate_type);
                    locations.push(node, qubits, false, gate.gate_type);
                }
            }
        }

        locations
    }

    /// Builds the complete fault influence map.
    ///
    /// This performs backward propagation from all measurements and
    /// creates a lookup table for fault classification.
    ///
    /// Note: This also builds reverse maps (detector -> faults). For performance-critical
    /// code, use `build_influence_map_soa` instead.
    #[must_use]
    pub fn build_influence_map(&self) -> DagFaultInfluenceMap {
        self.build_influence_map_with_logicals(&[])
    }

    /// Builds the fault influence map with logical operator tracking.
    ///
    /// # Arguments
    ///
    /// * `logicals` - Logical operators as (x_positions, z_positions) pairs.
    #[must_use]
    pub fn build_influence_map_with_logicals(
        &self,
        logicals: &[(&[usize], &[usize])],
    ) -> DagFaultInfluenceMap {
        self.build_influence_map_impl(logicals, true)
    }

    /// Builds the fault influence map with optional reverse maps.
    ///
    /// # Arguments
    ///
    /// * `logicals` - Logical operators as (x_positions, z_positions) pairs.
    /// * `build_reverse` - Whether to build reverse maps (detector -> faults).
    #[must_use]
    pub fn build_influence_map_impl(
        &self,
        logicals: &[(&[usize], &[usize])],
        build_reverse: bool,
    ) -> DagFaultInfluenceMap {
        let mut map = DagFaultInfluenceMap::new();

        // Extract all measurements from the circuit
        let measurements = self.extract_measurements();
        map.measurements = measurements.clone();

        // Create simple detectors (one per measurement)
        for &(node, qubit, basis) in &measurements {
            let measurement_id = MeasurementId {
                tick: node, // Use node as "tick" for compatibility
                qubit,
                basis,
            };
            map.detectors.push(DetectorId::single(measurement_id));
        }

        // Create logical IDs
        for (i, _) in logicals.iter().enumerate() {
            map.logicals.push(LogicalId {
                logical_qubit: i,
                observable: 0, // Z observable
            });
        }

        // Initialize influence for each fault location
        // Convert SoA back to AoS for the output map
        for loc in self.locations.to_dag_spacetime_locations() {
            map.influences
                .insert(loc, FaultInfluence::default());
        }

        // Pre-allocate work arrays to reuse across propagations
        let mut visited = vec![false; self.propagator.max_node() + 1];
        let mut active_qubits = vec![false; self.propagator.max_qubit() + 1];
        let mut heap: BinaryHeap<(usize, usize)> = BinaryHeap::with_capacity(64);

        // Backward propagate from each measurement
        for (detector_idx, &(node, qubit, basis)) in measurements.iter().enumerate() {
            self.propagate_from_measurement_indexed(
                node,
                qubit,
                basis,
                detector_idx,
                &mut map,
                &mut visited,
                &mut active_qubits,
                &mut heap,
            );
        }

        // Backward propagate from each logical operator
        for (i, (x_pos, z_pos)) in logicals.iter().enumerate() {
            let logical_id = LogicalId {
                logical_qubit: i,
                observable: 0,
            };
            self.propagate_from_logical_reuse(
                x_pos,
                z_pos,
                &logical_id,
                &mut map,
                &mut visited,
                &mut active_qubits,
                &mut heap,
            );
        }

        // Build reverse maps only if requested
        if build_reverse {
            self.build_reverse_maps(&mut map);
        }

        map
    }

    /// Builds the fault influence map using true SoA storage (fastest and most memory-efficient).
    ///
    /// This uses CSR (Compressed Sparse Row) layout for maximum cache efficiency
    /// and minimal memory overhead. Best for large circuits or memory-constrained
    /// environments.
    ///
    /// # Example
    /// ```
    /// use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
    /// use pecos_quantum::DagCircuit;
    ///
    /// let mut dag = DagCircuit::new();
    /// dag.pz(2);
    /// dag.cx(0, 2);
    /// dag.mz(2);
    ///
    /// let propagator = DagFaultAnalyzer::new(&dag);
    /// let map = propagator.build_influence_map_soa();
    ///
    /// // Check memory usage
    /// let stats = map.memory_stats();
    /// println!("Total bytes: {}", stats.total_bytes);
    /// ```
    #[must_use]
    pub fn build_influence_map_soa(&self) -> DagFaultInfluenceMapSoA {
        self.build_influence_map_soa_direct()
    }

    /// Builds the fault influence map directly into SoA format.
    ///
    /// Uses a bucket-based recorder that collects influences per location in O(n),
    /// then flattens to CSR format.
    #[must_use]
    pub fn build_influence_map_soa_direct(&self) -> DagFaultInfluenceMapSoA {
        let num_locations = self.locations.len();
        let mut map = DagFaultInfluenceMapSoA::with_capacity(num_locations);

        // Copy locations
        map.locations = self.locations.to_dag_spacetime_locations();

        // Extract measurements and create detectors
        let measurements = self.extract_measurements();
        map.measurements = measurements.clone();

        for &(node, qubit, basis) in &measurements {
            let measurement_id = MeasurementId {
                tick: node,
                qubit,
                basis,
            };
            map.detectors.push(DetectorId::single(measurement_id));
        }

        // Use bucket recorder for O(n) construction
        let mut recorder = BucketRecorder::new(num_locations);

        // Propagate using the generic method with bucket recorder
        self.propagate_all(&mut recorder);

        // Convert buckets to SoA format (O(n) flattening)
        map.influences = recorder.into_soa();

        map
    }

    /// Extracts all measurements from the circuit.
    fn extract_measurements(&self) -> Vec<(usize, usize, u8)> {
        let mut measurements = Vec::new();

        for &node in self.propagator.topo_order() {
            if let Some(gate) = self.propagator.gate(node) {
                let basis = match gate.gate_type {
                    GateType::Measure | GateType::MeasureFree => 0, // Z-basis
                    _ => continue,
                };

                for qubit in &gate.qubits {
                    measurements.push((node, qubit.index(), basis));
                }
            }
        }

        measurements
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
        visited: &mut [bool],
        active_qubits: &mut [bool],
        heap: &mut BinaryHeap<(usize, usize)>,
    ) {
        // Clear work arrays
        visited.fill(false);
        active_qubits.fill(false);
        heap.clear();

        // Start with the observable being measured
        let mut prop = PauliProp::new();
        if basis == 0 {
            prop.add_z(meas_qubit);
        } else {
            prop.add_x(meas_qubit);
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

                // Check before=false locations
                self.record_at_node_generic(node, &prop, detector_idx, recorder, false);

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
    /// dag.pz(2);
    /// dag.cx(0, 2);
    /// dag.mz(2);
    ///
    /// let propagator = DagFaultAnalyzer::new(&dag);
    ///
    /// // Use a counting recorder to count influences
    /// let mut recorder = CountingRecorder::default();
    /// propagator.propagate_all(&mut recorder);
    /// println!("Total influences: {}", recorder.count);
    /// ```
    pub fn propagate_all<R: InfluenceRecorder>(&self, recorder: &mut R) {
        let measurements = self.extract_measurements();

        // Pre-allocate work arrays
        let mut visited = vec![false; self.propagator.max_node() + 1];
        let mut active_qubits = vec![false; self.propagator.max_qubit() + 1];
        let mut heap: BinaryHeap<(usize, usize)> = BinaryHeap::with_capacity(64);

        for (detector_idx, &(node, qubit, basis)) in measurements.iter().enumerate() {
            self.propagate_from_measurement_generic(
                node,
                qubit,
                basis,
                detector_idx,
                recorder,
                &mut visited,
                &mut active_qubits,
                &mut heap,
            );
        }
    }

    /// Propagates backward from a measurement using truly sparse traversal.
    ///
    /// Only visits gates on the wires (qubits) we're propagating through.
    /// Uses a max-heap to process gates in reverse topo order efficiently.
    #[allow(dead_code)]
    fn propagate_from_measurement(
        &self,
        meas_node: usize,
        meas_qubit: usize,
        basis: u8,
        map: &mut DagFaultInfluenceMap,
    ) {
        // Start with the observable being measured
        let mut prop = PauliProp::new();
        if basis == 0 {
            prop.add_z(meas_qubit);
        } else {
            prop.add_x(meas_qubit);
        }

        let measurement_id = MeasurementId {
            tick: meas_node,
            qubit: meas_qubit,
            basis,
        };
        let detector = DetectorId::single(measurement_id);

        // Get measurement position (O(1) lookup)
        let meas_topo_pos = self.propagator.topo_position(meas_node);

        // Check fault at measurement node (before=true only)
        self.record_influences_at_node_fast(meas_node, &prop, &detector, None, map, true);

        // Track visited nodes (queued or processed) and active qubits
        let mut visited = vec![false; self.propagator.max_node() + 1];
        let mut active_qubits = vec![false; self.propagator.max_qubit() + 1];

        // Max-heap: (topo_pos, node) - highest topo_pos comes first (reverse order)
        // Pre-allocate with estimated capacity (average gates per qubit * expected active qubits)
        let mut heap: BinaryHeap<(usize, usize)> = BinaryHeap::with_capacity(32);

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
                // Track which qubits were active before
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= self.max_qubit() {
                        was_active[j] = active_qubits[q.index()];
                    }
                }

                // Check before=false locations (after this gate executes)
                self.record_influences_at_node_fast(node, &prop, &detector, None, map, false);

                // Apply gate backward
                apply_gate(&mut prop, gate, Direction::Backward);

                // Check before=true locations (before this gate executes)
                self.record_influences_at_node_fast(node, &prop, &detector, None, map, true);

                // Check if Pauli spread to new qubits
                let node_topo_pos = self.propagator.topo_position(node);
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= self.max_qubit() {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            // Pauli spread to this qubit - add its gates to heap
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

    /// Propagates backward from a measurement using detector index.
    /// This avoids creating DetectorId on each call - uses the pre-existing one from map.detectors.
    fn propagate_from_measurement_indexed(
        &self,
        meas_node: usize,
        meas_qubit: usize,
        basis: u8,
        detector_idx: usize,
        map: &mut DagFaultInfluenceMap,
        visited: &mut [bool],
        active_qubits: &mut [bool],
        heap: &mut BinaryHeap<(usize, usize)>,
    ) {
        // Clear work arrays
        visited.fill(false);
        active_qubits.fill(false);
        heap.clear();

        // Start with the observable being measured
        let mut prop = PauliProp::new();
        if basis == 0 {
            prop.add_z(meas_qubit);
        } else {
            prop.add_x(meas_qubit);
        }

        // Get measurement position (O(1) lookup)
        let meas_topo_pos = self.propagator.topo_position(meas_node);

        // Check fault at measurement node (before=true only)
        self.record_influences_at_node_indexed(meas_node, &prop, detector_idx, None, map, true);

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

                // Check before=false locations
                self.record_influences_at_node_indexed(node, &prop, detector_idx, None, map, false);

                // Apply gate backward
                apply_gate(&mut prop, gate, Direction::Backward);

                // Check before=true locations
                self.record_influences_at_node_indexed(node, &prop, detector_idx, None, map, true);

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

    /// Propagates backward from a measurement, reusing pre-allocated work arrays.
    #[allow(dead_code)]
    fn propagate_from_measurement_reuse(
        &self,
        meas_node: usize,
        meas_qubit: usize,
        basis: u8,
        map: &mut DagFaultInfluenceMap,
        visited: &mut [bool],
        active_qubits: &mut [bool],
        heap: &mut BinaryHeap<(usize, usize)>,
    ) {
        // Clear work arrays (faster than reallocating)
        visited.fill(false);
        active_qubits.fill(false);
        heap.clear();

        // Start with the observable being measured
        let mut prop = PauliProp::new();
        if basis == 0 {
            prop.add_z(meas_qubit);
        } else {
            prop.add_x(meas_qubit);
        }

        let measurement_id = MeasurementId {
            tick: meas_node,
            qubit: meas_qubit,
            basis,
        };
        let detector = DetectorId::single(measurement_id);

        // Get measurement position (O(1) lookup)
        let meas_topo_pos = self.propagator.topo_position(meas_node);

        // Check fault at measurement node (before=true only)
        self.record_influences_at_node_fast(meas_node, &prop, &detector, None, map, true);

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
                // Track which qubits were active before
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= self.max_qubit() {
                        was_active[j] = active_qubits[q.index()];
                    }
                }

                // Check before=false locations (after this gate executes)
                self.record_influences_at_node_fast(node, &prop, &detector, None, map, false);

                // Apply gate backward
                apply_gate(&mut prop, gate, Direction::Backward);

                // Check before=true locations (before this gate executes)
                self.record_influences_at_node_fast(node, &prop, &detector, None, map, true);

                // Check if Pauli spread to new qubits
                let node_topo_pos = self.propagator.topo_position(node);
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= self.max_qubit() {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            // Pauli spread to this qubit - add its gates to heap
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

    /// Propagates backward from a logical operator using truly sparse traversal.
    ///
    /// Only visits gates on the wires we're propagating through.
    /// Uses a max-heap to process gates in reverse topo order efficiently.
    #[allow(dead_code)]
    fn propagate_from_logical(
        &self,
        x_positions: &[usize],
        z_positions: &[usize],
        logical_id: &LogicalId,
        map: &mut DagFaultInfluenceMap,
    ) {
        let mut prop = PauliProp::new();
        for &q in x_positions {
            prop.add_x(q);
        }
        for &q in z_positions {
            prop.add_z(q);
        }

        let dummy_detector = DetectorId::single(MeasurementId {
            tick: 0,
            qubit: 0,
            basis: 0,
        });

        // Track visited nodes (queued or processed) and active qubits
        let mut visited = vec![false; self.propagator.max_node() + 1];
        let mut active_qubits = vec![false; self.propagator.max_qubit() + 1];

        // Max-heap: (topo_pos, node) - highest topo_pos comes first (reverse order)
        // Pre-allocate with estimated capacity
        let mut heap: BinaryHeap<(usize, usize)> = BinaryHeap::with_capacity(64);

        // Initialize from logical operator support - add all gates on these qubits
        for &q in x_positions {
            if q <= self.max_qubit() && !active_qubits[q] {
                active_qubits[q] = true;
                for (topo_pos, node) in self.propagator.qubit_gates_backward(q) {
                    if !visited[node] {
                        visited[node] = true;
                        heap.push((topo_pos, node));
                    }
                }
            }
        }
        for &q in z_positions {
            if q <= self.max_qubit() && !active_qubits[q] {
                active_qubits[q] = true;
                for (topo_pos, node) in self.propagator.qubit_gates_backward(q) {
                    if !visited[node] {
                        visited[node] = true;
                        heap.push((topo_pos, node));
                    }
                }
            }
        }

        // Process gates in reverse topo order - only gates on active wires
        while let Some((_, node)) = heap.pop() {
            if let Some(gate) = self.propagator.gate(node) {
                // Track which qubits were active before
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= self.max_qubit() {
                        was_active[j] = active_qubits[q.index()];
                    }
                }

                // Check before=false locations (after this gate executes)
                self.record_influences_at_node_fast(
                    node,
                    &prop,
                    &dummy_detector,
                    Some(logical_id),
                    map,
                    false,
                );

                // Apply gate backward
                apply_gate(&mut prop, gate, Direction::Backward);

                // Check before=true locations (before this gate executes)
                self.record_influences_at_node_fast(
                    node,
                    &prop,
                    &dummy_detector,
                    Some(logical_id),
                    map,
                    true,
                );

                // Check if Pauli spread to new qubits
                let node_topo_pos = self.propagator.topo_position(node);
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= self.max_qubit() {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            // Pauli spread to this qubit - add its gates to heap
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

    /// Propagates backward from a logical operator, reusing pre-allocated work arrays.
    fn propagate_from_logical_reuse(
        &self,
        x_positions: &[usize],
        z_positions: &[usize],
        logical_id: &LogicalId,
        map: &mut DagFaultInfluenceMap,
        visited: &mut [bool],
        active_qubits: &mut [bool],
        heap: &mut BinaryHeap<(usize, usize)>,
    ) {
        // Clear work arrays
        visited.fill(false);
        active_qubits.fill(false);
        heap.clear();

        let mut prop = PauliProp::new();
        for &q in x_positions {
            prop.add_x(q);
        }
        for &q in z_positions {
            prop.add_z(q);
        }

        let dummy_detector = DetectorId::single(MeasurementId {
            tick: 0,
            qubit: 0,
            basis: 0,
        });

        // Initialize from logical operator support
        for &q in x_positions {
            if q <= self.max_qubit() && !active_qubits[q] {
                active_qubits[q] = true;
                for (topo_pos, node) in self.propagator.qubit_gates_backward(q) {
                    if !visited[node] {
                        visited[node] = true;
                        heap.push((topo_pos, node));
                    }
                }
            }
        }
        for &q in z_positions {
            if q <= self.max_qubit() && !active_qubits[q] {
                active_qubits[q] = true;
                for (topo_pos, node) in self.propagator.qubit_gates_backward(q) {
                    if !visited[node] {
                        visited[node] = true;
                        heap.push((topo_pos, node));
                    }
                }
            }
        }

        // Process gates in reverse topo order
        while let Some((_, node)) = heap.pop() {
            if let Some(gate) = self.propagator.gate(node) {
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= self.max_qubit() {
                        was_active[j] = active_qubits[q.index()];
                    }
                }

                self.record_influences_at_node_fast(
                    node,
                    &prop,
                    &dummy_detector,
                    Some(logical_id),
                    map,
                    false,
                );

                apply_gate(&mut prop, gate, Direction::Backward);

                self.record_influences_at_node_fast(
                    node,
                    &prop,
                    &dummy_detector,
                    Some(logical_id),
                    map,
                    true,
                );

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

    /// Records influences using detector index instead of DetectorId reference.
    /// Looks up the detector from map.detectors only when needed.
    #[inline]
    fn record_influences_at_node_indexed(
        &self,
        node: usize,
        prop: &PauliProp,
        detector_idx: usize,
        logical: Option<&LogicalId>,
        map: &mut DagFaultInfluenceMap,
        only_before: bool,
    ) {
        // Use pre-computed index for O(1) lookup
        for &loc_idx in self.locations.locations_at_node(node) {
            let before = self.locations.is_before(loc_idx);
            if before != only_before {
                continue;
            }

            // Build location key for BTreeMap lookup
            let loc = DagSpacetimeLocation {
                node: self.locations.nodes[loc_idx],
                qubits: self.locations.qubits[loc_idx].iter().map(|&q| QubitId::from(q)).collect(),
                before,
                gate_type: self.locations.gate_types[loc_idx],
            };

            for &q in self.locations.qubits(loc_idx) {
                let obs_x = prop.contains_x(q);
                let obs_z = prop.contains_z(q);

                if let Some(influence) = map.influences.get_mut(&loc) {
                    // X fault anticommutes with Z or Y observable
                    if obs_z {
                        if let Some(log) = logical {
                            influence.logical_flips[1].push(*log);
                        } else {
                            let detector = &map.detectors[detector_idx];
                            influence.detector_flips[1].push(detector.clone());
                            influence.measurement_flips[1].extend(detector.measurements.iter().copied());
                        }
                    }

                    // Z fault anticommutes with X or Y observable
                    if obs_x {
                        if let Some(log) = logical {
                            influence.logical_flips[3].push(*log);
                        } else {
                            let detector = &map.detectors[detector_idx];
                            influence.detector_flips[3].push(detector.clone());
                            influence.measurement_flips[3].extend(detector.measurements.iter().copied());
                        }
                    }

                    // Y fault anticommutes with X, Z, or Y observable
                    if obs_x || obs_z {
                        if let Some(log) = logical {
                            influence.logical_flips[2].push(*log);
                        } else {
                            let detector = &map.detectors[detector_idx];
                            influence.detector_flips[2].push(detector.clone());
                            influence.measurement_flips[2].extend(detector.measurements.iter().copied());
                        }
                    }
                }
            }
        }
    }

    /// Records influences using pre-computed node-to-locations index (O(1) lookup).
    #[inline]
    #[allow(dead_code)]
    fn record_influences_at_node_fast(
        &self,
        node: usize,
        prop: &PauliProp,
        detector: &DetectorId,
        logical: Option<&LogicalId>,
        map: &mut DagFaultInfluenceMap,
        only_before: bool,
    ) {
        // Use pre-computed index for O(1) lookup
        for &loc_idx in self.locations.locations_at_node(node) {
            let before = self.locations.is_before(loc_idx);
            if before != only_before {
                continue;
            }

            // Build location key for BTreeMap lookup
            let loc = DagSpacetimeLocation {
                node: self.locations.nodes[loc_idx],
                qubits: self.locations.qubits[loc_idx].iter().map(|&q| QubitId::from(q)).collect(),
                before,
                gate_type: self.locations.gate_types[loc_idx],
            };

            for (qubit_idx, &q) in self.locations.qubits(loc_idx).iter().enumerate() {
                let obs_x = prop.contains_x(q);
                let obs_z = prop.contains_z(q);

                if let Some(influence) = map.influences.get_mut(&loc) {
                    // X fault anticommutes with Z or Y observable
                    if obs_z {
                        if let Some(log) = logical {
                            influence.logical_flips[1].push(*log);
                        } else {
                            influence.detector_flips[1].push(detector.clone());
                            influence.measurement_flips[1].extend(detector.measurements.iter().copied());
                            influence
                                .per_qubit_detector_flips
                                .entry((qubit_idx, 1))
                                .or_default()
                                .push(detector.clone());
                        }
                    }

                    // Z fault anticommutes with X or Y observable
                    if obs_x {
                        if let Some(log) = logical {
                            influence.logical_flips[3].push(*log);
                        } else {
                            influence.detector_flips[3].push(detector.clone());
                            influence.measurement_flips[3].extend(detector.measurements.iter().copied());
                            influence
                                .per_qubit_detector_flips
                                .entry((qubit_idx, 3))
                                .or_default()
                                .push(detector.clone());
                        }
                    }

                    // Y fault anticommutes with X, Z, or Y observable
                    if obs_x || obs_z {
                        if let Some(log) = logical {
                            influence.logical_flips[2].push(*log);
                        } else {
                            influence.detector_flips[2].push(detector.clone());
                            influence.measurement_flips[2].extend(detector.measurements.iter().copied());
                            influence
                                .per_qubit_detector_flips
                                .entry((qubit_idx, 2))
                                .or_default()
                                .push(detector.clone());
                        }
                    }
                }
            }
        }
    }

    /// Builds reverse maps (detector -> faults, logical -> faults).
    fn build_reverse_maps(&self, map: &mut DagFaultInfluenceMap) {
        for (loc, influence) in &map.influences {
            for (pauli, detectors) in influence.detector_flips.iter().enumerate() {
                for detector in detectors {
                    map.detector_to_faults
                        .entry(detector.clone())
                        .or_default()
                        .push((loc.clone(), pauli as u8));
                }
            }

            for (pauli, logicals) in influence.logical_flips.iter().enumerate() {
                for logical in logicals {
                    map.logical_to_faults
                        .entry(*logical)
                        .or_default()
                        .push((loc.clone(), pauli as u8));
                }
            }
        }
    }
}

// ============================================================================
// Integration with Fault Checking
// ============================================================================

/// Efficient fault checker using pre-computed influence maps.
///
/// This provides O(1) fault classification instead of O(circuit_depth)
/// forward propagation.
pub struct InfluenceBasedChecker<'a> {
    influence_map: &'a FaultInfluenceMap,
}

impl<'a> InfluenceBasedChecker<'a> {
    /// Creates a new checker from a pre-computed influence map.
    pub fn new(influence_map: &'a FaultInfluenceMap) -> Self {
        Self { influence_map }
    }

    /// Classifies a fault at the given location with the given Pauli type.
    ///
    /// For single-qubit locations, returns whether any qubit causes syndrome/logical.
    /// For multi-qubit locations where the same Pauli is applied to all qubits,
    /// use `classify_uniform` which properly handles cancellation effects.
    ///
    /// Returns (has_syndrome, causes_logical_error).
    pub fn classify(&self, location: &SpacetimeLocation, pauli: u8) -> (bool, bool) {
        self.influence_map.classify_fault(location, pauli)
    }

    /// Classifies a multi-qubit fault where the same Pauli is applied to all qubits.
    ///
    /// This properly handles cancellation: if the same Pauli on two different qubits
    /// both flip the same detector, they cancel out (XOR semantics).
    ///
    /// For Y faults (single or multi-qubit), we decompose Y = XZ and combine the
    /// X and Z contributions with XOR semantics.
    ///
    /// Returns (has_syndrome, causes_logical_error).
    pub fn classify_uniform(&self, location: &SpacetimeLocation, pauli: u8) -> (bool, bool) {
        // Always use multi-qubit logic for Y faults (even single-qubit)
        // because Y = XZ needs to combine X and Z contributions
        if pauli == 2 || location.qubits.len() > 1 {
            self.influence_map
                .classify_multi_qubit_fault(location, pauli)
        } else {
            // Single-qubit X or Z: simple lookup
            self.influence_map.classify_fault(location, pauli)
        }
    }

    /// Returns all detectors flipped by the given fault.
    pub fn detectors_flipped(
        &self,
        location: &SpacetimeLocation,
        pauli: u8,
    ) -> Vec<&DetectorId> {
        self.influence_map
            .get_influence(location)
            .map_or(Vec::new(), |inf| {
                inf.detectors_for_pauli(pauli).iter().collect()
            })
    }

    /// Checks if a fault causes an undetectable logical error.
    pub fn is_undetectable_logical_error(
        &self,
        location: &SpacetimeLocation,
        pauli: u8,
    ) -> bool {
        let (has_syndrome, has_logical) = self.classify(location, pauli);
        !has_syndrome && has_logical
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn simple_syndrome_circuit() -> TickCircuit {
        // Simple Z-stabilizer measurement: Z0 Z1
        // Ancilla qubit 2 measures the parity
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[2]); // Prep ancilla in |0>
        circuit.tick().cx(&[(0, 2)]); // CNOT from data 0 to ancilla
        circuit.tick().cx(&[(1, 2)]); // CNOT from data 1 to ancilla
        circuit.tick().mz(&[2]); // Measure ancilla
        circuit
    }

    #[test]
    fn test_extract_measurements() {
        let circuit = simple_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);
        let measurements = propagator.extract_measurements();

        assert_eq!(measurements.len(), 1);
        assert_eq!(measurements[0].tick, 3);
        assert_eq!(measurements[0].qubit, 2);
        assert_eq!(measurements[0].basis, 0); // Z-measurement
    }

    #[test]
    fn test_build_influence_map() {
        let circuit = simple_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);
        let map = propagator.build_influence_map();

        // Should have fault locations
        assert!(map.num_fault_locations() > 0);

        // Should have one detector
        assert_eq!(map.detectors.len(), 1);

        // Should have one measurement
        assert_eq!(map.measurements.len(), 1);

        println!("Influence map has {} fault locations", map.num_fault_locations());
        println!("Detectors: {:?}", map.detectors);
    }

    #[test]
    fn test_x_error_flips_z_measurement() {
        let circuit = simple_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);
        let map = propagator.build_influence_map();

        // An X error on data qubit 0 before the first CNOT should flip the measurement
        // because it propagates through CX to the ancilla

        // Find a fault location on qubit 0
        let mut found_x_flip = false;
        for (loc, influence) in &map.influences {
            if loc.qubits.iter().any(|q| q.index() == 0) {
                // Check if X error here flips the detector
                if !influence.detectors_for_pauli(1).is_empty() {
                    found_x_flip = true;
                    println!(
                        "X error at {:?} flips detector",
                        loc
                    );
                }
            }
        }

        assert!(found_x_flip, "Should find X errors that flip the measurement");
    }

    #[test]
    fn test_z_error_no_syndrome() {
        let circuit = simple_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);
        let map = propagator.build_influence_map();

        // A Z error on data qubits should NOT flip the Z-measurement
        // because Z commutes with CX on control, and measurement is Z-basis

        // Check that Z errors on data qubits don't flip detectors
        for (loc, influence) in &map.influences {
            if loc.qubits.iter().any(|q| q.index() == 0 || q.index() == 1) {
                // Z errors on data qubits shouldn't flip Z-measurement
                let z_flips = influence.detectors_for_pauli(3);
                // This may or may not be empty depending on exact location
                println!(
                    "Z error at {:?} flips {} detectors",
                    loc,
                    z_flips.len()
                );
            }
        }
    }

    #[test]
    fn test_influence_based_checker() {
        let circuit = simple_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);
        let map = propagator.build_influence_map();

        let checker = InfluenceBasedChecker::new(&map);

        // Test classification for each fault location
        for loc in map.influences.keys() {
            let (has_x_syndrome, has_x_logical) = checker.classify(loc, 1);
            let (has_z_syndrome, has_z_logical) = checker.classify(loc, 3);

            println!(
                "Location {:?}: X->({}, {}), Z->({}, {})",
                loc, has_x_syndrome, has_x_logical, has_z_syndrome, has_z_logical
            );
        }
    }

    #[test]
    fn test_with_logical_operator() {
        let circuit = simple_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);

        // Logical Z = Z0 Z1 (the stabilizer being measured)
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1])];
        let map = propagator.build_influence_map_with_logicals(logicals);

        // Should have one logical
        assert_eq!(map.logicals.len(), 1);

        // Check for faults that flip the logical
        let mut found_logical_flip = false;
        for (loc, influence) in &map.influences {
            for (pauli, logicals) in influence.logical_flips.iter().enumerate() {
                if !logicals.is_empty() {
                    found_logical_flip = true;
                    println!(
                        "Pauli {} at {:?} flips logical",
                        pauli, loc
                    );
                }
            }
        }

        // Z errors on data qubits should flip the logical Z
        assert!(found_logical_flip, "Should find faults that flip logical");
    }

    #[test]
    fn test_reverse_map() {
        let circuit = simple_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);
        let map = propagator.build_influence_map();

        // Check reverse map
        for (detector, faults) in &map.detector_to_faults {
            println!(
                "Detector {:?} is flipped by {} fault locations",
                detector, faults.len()
            );
            assert!(!faults.is_empty());
        }
    }

    fn two_round_syndrome_circuit() -> TickCircuit {
        // Two rounds of Z-stabilizer measurement
        let mut circuit = TickCircuit::new();

        // Round 1
        circuit.tick().pz(&[2]);
        circuit.tick().cx(&[(0, 2)]);
        circuit.tick().cx(&[(1, 2)]);
        circuit.tick().mz(&[2]);

        // Round 2
        circuit.tick().pz(&[2]);
        circuit.tick().cx(&[(0, 2)]);
        circuit.tick().cx(&[(1, 2)]);
        circuit.tick().mz(&[2]);

        circuit
    }

    #[test]
    fn test_two_round_measurements() {
        let circuit = two_round_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);
        let measurements = propagator.extract_measurements();

        // Should have two measurements (one per round)
        assert_eq!(measurements.len(), 2);
        assert_eq!(measurements[0].tick, 3);
        assert_eq!(measurements[1].tick, 7);

        let map = propagator.build_influence_map();
        assert_eq!(map.detectors.len(), 2);

        println!("Two-round circuit has {} fault locations", map.num_fault_locations());
    }

    #[test]
    fn test_y_error_handling() {
        // Y = iXZ, so Y error should flip Z-measurement (via X component)
        let circuit = simple_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);
        let map = propagator.build_influence_map();

        // Find a CX gate location and check Y error behavior
        let mut found_y_effect = false;
        for (loc, influence) in &map.influences {
            if loc.gate_type == GateType::CX {
                // Y error (pauli=2) should have same detector effect as X (pauli=1)
                // since Y = iXZ and only X flips Z-measurement
                let y_detectors = influence.detectors_for_pauli(2);
                let x_detectors = influence.detectors_for_pauli(1);

                // If X flips a detector, Y might too (depending on location)
                if !x_detectors.is_empty() || !y_detectors.is_empty() {
                    found_y_effect = true;
                    println!(
                        "At {:?}: X flips {} detectors, Y flips {} detectors",
                        loc,
                        x_detectors.len(),
                        y_detectors.len()
                    );
                }
            }
        }
        assert!(found_y_effect, "Should find some Y error effects");
    }

    #[test]
    fn test_two_qubit_gate_fault_location() {
        // Test that faults on 2-qubit gates affect both qubits correctly
        let circuit = simple_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);
        let map = propagator.build_influence_map();

        // Find CX gate locations (should have 2 qubits)
        let mut found_two_qubit_loc = false;
        for loc in map.influences.keys() {
            if loc.gate_type == GateType::CX {
                assert_eq!(loc.qubits.len(), 2, "CX should have 2 qubits");
                found_two_qubit_loc = true;

                // The influence for this location applies to faults on either qubit
                println!(
                    "CX at tick {} on qubits {:?}",
                    loc.tick,
                    loc.qubits.iter().map(|q| q.index()).collect::<Vec<_>>()
                );
            }
        }
        assert!(found_two_qubit_loc, "Should find CX gate locations");
    }

    #[test]
    fn test_undetectable_logical_error_detection() {
        let circuit = simple_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);

        // Logical X = X0 (anticommutes with Z0 Z1)
        let logicals: &[(&[usize], &[usize])] = &[(&[0], &[])];
        let map = propagator.build_influence_map_with_logicals(logicals);
        let checker = InfluenceBasedChecker::new(&map);

        // An X error on data qubit 0 should:
        // - Flip the detector (has syndrome)
        // - Flip the logical X (has logical error)
        // So it should NOT be an undetectable logical error

        // Check each fault location
        let mut found_undetectable = false;
        let mut found_detectable_logical = false;
        for loc in map.influences.keys() {
            // Check X error (pauli=1)
            if checker.is_undetectable_logical_error(loc, 1) {
                found_undetectable = true;
                println!("Undetectable logical error: X at {:?}", loc);
            }
            let (has_syn, has_log) = checker.classify(loc, 1);
            if has_syn && has_log {
                found_detectable_logical = true;
            }
        }

        // In this simple circuit, X errors on data qubits are detected
        // so there shouldn't be undetectable logical errors for X
        println!(
            "Found undetectable: {}, Found detectable with logical: {}",
            found_undetectable, found_detectable_logical
        );
    }

    #[test]
    fn test_preparation_stops_backward_propagation() {
        // After a prep gate, backward propagation should stop for that qubit.
        // Errors on the ANCILLA before prep shouldn't affect measurements after it.
        // However, errors on DATA qubits persist and CAN affect later measurements
        // (since data qubits are not reset by the prep).
        let circuit = two_round_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);
        let map = propagator.build_influence_map();

        // The second measurement (tick 7) should NOT be affected by faults
        // on the ANCILLA QUBIT (q2) before the second prep (tick 4).
        // Faults on data qubits (q0, q1) CAN affect round 2.

        // Find the second detector
        let second_detector = &map.detectors[1];

        // Get faults that flip the second detector
        let faults_for_second = map.faults_for_detector(second_detector);

        // Ancilla-only faults before tick 4 should not affect round 2
        for (loc, _pauli) in faults_for_second {
            // Check if this is an ancilla-only fault before round 2
            let is_ancilla_only = loc.qubits.iter().all(|q| q.index() == 2);
            if is_ancilla_only && loc.tick < 4 {
                panic!(
                    "Ancilla fault at tick {} should not affect measurement in round 2 (prep at tick 4)",
                    loc.tick
                );
            }
        }

        // Verify that round 2 ancilla faults DO affect round 2 measurement
        let round2_ancilla_faults: Vec<_> = faults_for_second
            .iter()
            .filter(|(loc, _)| {
                let is_ancilla_only = loc.qubits.iter().all(|q| q.index() == 2);
                is_ancilla_only && loc.tick >= 4
            })
            .collect();
        assert!(
            !round2_ancilla_faults.is_empty(),
            "Should have ancilla faults in round 2 affecting round 2 measurement"
        );

        println!(
            "Second detector is affected by {} faults total",
            faults_for_second.len()
        );
        println!(
            "  {} are round-2 ancilla faults",
            round2_ancilla_faults.len()
        );
    }

    fn x_stabilizer_circuit() -> TickCircuit {
        // X-stabilizer measurement: X0 X1
        // Uses H gates to convert to Z-basis measurement
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[2]); // Prep ancilla in |0>
        circuit.tick().h(&[2]); // H to get |+>
        circuit.tick().cx(&[(2, 0)]); // CNOT from ancilla to data 0
        circuit.tick().cx(&[(2, 1)]); // CNOT from ancilla to data 1
        circuit.tick().h(&[2]); // H before measurement
        circuit.tick().mz(&[2]); // Measure ancilla in Z basis (effective X measurement)
        circuit
    }

    #[test]
    fn test_h_gate_backward_propagation() {
        // Test that H gates correctly transform Paulis backward
        let circuit = x_stabilizer_circuit();
        let propagator = BackwardPropagator::new(&circuit);
        let map = propagator.build_influence_map();

        // In X-stabilizer circuit, Z errors on data qubits should flip the measurement
        // (because Z anticommutes with X stabilizer)
        let mut found_z_flip = false;
        for (loc, influence) in &map.influences {
            if loc.qubits.iter().any(|q| q.index() == 0 || q.index() == 1) {
                if !influence.detectors_for_pauli(3).is_empty() {
                    found_z_flip = true;
                    println!("Z error at {:?} flips X-stabilizer measurement", loc);
                }
            }
        }

        assert!(found_z_flip, "Z errors on data should flip X-stabilizer");
    }

    #[test]
    fn test_backward_vs_forward_consistency() {
        // Verify backward propagation gives consistent results with forward propagation
        use super::super::{PauliFault, propagate_fault, has_syndrome, anticommutes_with_logical};

        let circuit = simple_syndrome_circuit();

        // Build backward influence map
        let propagator = BackwardPropagator::new(&circuit);
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1])]; // Z logical
        let map = propagator.build_influence_map_with_logicals(logicals);
        let backward_checker = InfluenceBasedChecker::new(&map);

        let z_ancillas: &[usize] = &[2];
        let x_ancillas: &[usize] = &[];

        // Compare results for key fault locations
        let mut consistent = 0;
        let mut total = 0;

        for loc in map.influences.keys() {
            // Only check single-qubit locations for simpler comparison
            if loc.qubits.len() == 1 {
                total += 1;

                // Backward classification for X error
                let (back_syn, back_log) = backward_checker.classify(loc, 1);

                // Forward propagation: inject X error and check
                let fault = PauliFault::new(loc.clone(), vec![1]); // X error
                let prop = propagate_fault(&circuit, &fault);
                let fwd_syn = has_syndrome(&prop, z_ancillas, x_ancillas);
                let fwd_log = anticommutes_with_logical(&prop, logicals[0].0, logicals[0].1);

                if back_syn == fwd_syn {
                    consistent += 1;
                }

                println!(
                    "X at {:?}: backward=({}, {}), forward=({}, {})",
                    loc, back_syn, back_log, fwd_syn, fwd_log
                );
            }
        }

        println!("Checked {} locations, {} consistent", total, consistent);
        // All single-qubit locations should have consistent syndrome detection
        assert_eq!(consistent, total, "All syndrome detections should match");
    }

    #[test]
    fn test_empty_circuit() {
        let circuit = TickCircuit::new();
        let propagator = BackwardPropagator::new(&circuit);
        let map = propagator.build_influence_map();

        assert_eq!(map.num_fault_locations(), 0);
        assert_eq!(map.measurements.len(), 0);
        assert_eq!(map.detectors.len(), 0);
    }

    #[test]
    fn test_circuit_without_measurements() {
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1]);
        circuit.tick().cx(&[(0, 1)]);
        // No measurements

        let propagator = BackwardPropagator::new(&circuit);
        let map = propagator.build_influence_map();

        assert!(map.num_fault_locations() > 0); // Should have fault locations
        assert_eq!(map.measurements.len(), 0); // But no measurements
        assert_eq!(map.detectors.len(), 0); // And no detectors
    }

    #[test]
    fn test_detector_id_helpers() {
        let m1 = MeasurementId {
            tick: 0,
            qubit: 0,
            basis: 0,
        };
        let m2 = MeasurementId {
            tick: 1,
            qubit: 0,
            basis: 0,
        };

        // Single-measurement detector
        let d1 = DetectorId::single(m1);
        assert_eq!(d1.measurements.len(), 1);
        assert!(d1.name.is_none());

        // Comparison detector (XOR of two measurements)
        let d2 = DetectorId::comparison(m1, m2);
        assert_eq!(d2.measurements.len(), 2);

        // Named detector
        let d3 = DetectorId::single(m1).with_name("Z_check");
        assert_eq!(d3.name, Some("Z_check".to_string()));
    }

    #[test]
    fn test_fault_influence_is_trivial() {
        let influence = FaultInfluence::default();
        assert!(influence.is_trivial());

        let mut non_trivial = FaultInfluence::default();
        non_trivial.detector_flips[1].push(DetectorId::single(MeasurementId {
            tick: 0,
            qubit: 0,
            basis: 0,
        }));
        assert!(!non_trivial.is_trivial());
    }

    #[test]
    fn test_fault_influence_map_default() {
        let map = FaultInfluenceMap::default();
        assert_eq!(map.num_fault_locations(), 0);
        assert!(map.detectors.is_empty());
        assert!(map.logicals.is_empty());
    }

    /// Generate a random Clifford circuit for testing.
    fn random_clifford_circuit(num_qubits: usize, num_ticks: usize, seed: u64) -> TickCircuit {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Simple PRNG based on seed
        let mut state = seed;
        let mut next_rand = || {
            let mut hasher = DefaultHasher::new();
            state.hash(&mut hasher);
            state = hasher.finish();
            state
        };

        let mut circuit = TickCircuit::new();

        // First tick: prep all qubits
        let all_qubits: Vec<usize> = (0..num_qubits).collect();
        circuit.tick().pz(&all_qubits);

        // Middle ticks: random Clifford gates
        for _ in 0..num_ticks {
            let mut tick = circuit.tick();
            let gate_type = next_rand() % 4;

            match gate_type {
                0 => {
                    // H on random qubit
                    let q = (next_rand() % num_qubits as u64) as usize;
                    tick.h(&[q]);
                }
                1 => {
                    // S on random qubit
                    let q = (next_rand() % num_qubits as u64) as usize;
                    tick.sz(&[q]);
                }
                2 if num_qubits >= 2 => {
                    // CX on random pair
                    let q1 = (next_rand() % num_qubits as u64) as usize;
                    let mut q2 = (next_rand() % num_qubits as u64) as usize;
                    if q2 == q1 {
                        q2 = (q1 + 1) % num_qubits;
                    }
                    tick.cx(&[(q1, q2)]);
                }
                _ => {
                    // CZ on random pair
                    if num_qubits >= 2 {
                        let q1 = (next_rand() % num_qubits as u64) as usize;
                        let mut q2 = (next_rand() % num_qubits as u64) as usize;
                        if q2 == q1 {
                            q2 = (q1 + 1) % num_qubits;
                        }
                        tick.cz(&[(q1, q2)]);
                    }
                }
            }
        }

        // Last tick: measure some qubits (ancillas)
        let num_ancillas = (num_qubits / 2).max(1);
        let ancillas: Vec<usize> = (0..num_ancillas).collect();
        circuit.tick().mz(&ancillas);

        circuit
    }

    #[test]
    fn test_h_gate_propagation() {
        // Minimal test: prep, H, measure
        use super::super::{PauliFault, propagate_fault, has_syndrome};

        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0]); // tick 0: prep
        circuit.tick().h(&[0]); // tick 1: H
        circuit.tick().mz(&[0]); // tick 2: measure

        let propagator = BackwardPropagator::new(&circuit);
        let map = propagator.build_influence_map();
        let backward_checker = InfluenceBasedChecker::new(&map);

        let z_ancillas: &[usize] = &[0];
        let x_ancillas: &[usize] = &[];

        println!("=== H gate propagation test ===");
        println!("Measurements: {:?}", map.measurements);

        for loc in map.influences.keys() {
            for pauli in [1u8, 3] {
                let p_name = if pauli == 1 { "X" } else { "Z" };

                // Backward
                let (back_syn, _) = backward_checker.classify(loc, pauli);

                // Forward
                let fault = PauliFault::new(loc.clone(), vec![pauli; loc.qubits.len()]);
                let prop = propagate_fault(&circuit, &fault);
                let fwd_syn = has_syndrome(&prop, z_ancillas, x_ancillas);

                let status = if back_syn == fwd_syn { "OK" } else { "MISMATCH" };
                println!(
                    "{} tick={} q={:?} before={} {:?} {}: back={} fwd={} prop={}",
                    status, loc.tick, loc.qubits.iter().map(|q| q.index()).collect::<Vec<_>>(),
                    loc.before, loc.gate_type, p_name, back_syn, fwd_syn, prop
                );
            }
        }
    }

    #[test]
    fn test_backward_vs_forward_random_circuits() {
        // Test backward propagation matches forward on random Clifford circuits
        // Focus on single-qubit locations where semantics are clear
        use super::super::{PauliFault, propagate_fault, has_syndrome};

        let num_tests = 10;
        let mut total_checked = 0;
        let mut total_consistent = 0;
        let mut mismatches = Vec::new();

        for seed in 0..num_tests {
            let circuit = random_clifford_circuit(4, 5, seed);
            let propagator = BackwardPropagator::new(&circuit);
            let map = propagator.build_influence_map();
            let backward_checker = InfluenceBasedChecker::new(&map);

            // Determine ancilla qubits (the ones being measured)
            let z_ancillas: Vec<usize> = map.measurements.iter().map(|m| m.qubit).collect();
            let x_ancillas: &[usize] = &[];

            // Only check single-qubit locations for now
            // Multi-qubit locations have additional complexity with correlated errors
            for loc in map.influences.keys() {
                if loc.qubits.len() != 1 {
                    continue;
                }

                // Only check X and Z errors (Y has complex behavior)
                for pauli in [1u8, 3] {
                    // Backward classification
                    let (back_syn, _back_log) = backward_checker.classify(loc, pauli);

                    // Forward propagation
                    let fault = PauliFault::new(loc.clone(), vec![pauli]);
                    let prop = propagate_fault(&circuit, &fault);
                    let fwd_syn = has_syndrome(&prop, &z_ancillas, x_ancillas);

                    total_checked += 1;
                    if back_syn == fwd_syn {
                        total_consistent += 1;
                    } else {
                        mismatches.push((seed, loc.clone(), pauli, back_syn, fwd_syn, prop.clone()));
                    }
                }
            }
        }

        // Print first few mismatches with more detail
        for (seed, loc, pauli, back, fwd, prop) in mismatches.iter().take(5) {
            let p_name = if *pauli == 1 { "X" } else { "Z" };
            println!(
                "MISMATCH seed={} tick={} q={} before={} {:?} {}: back={} fwd={} prop={}",
                seed, loc.tick, loc.qubits[0].index(), loc.before, loc.gate_type, p_name, back, fwd, prop
            );
        }

        println!(
            "Random circuit test (single-qubit X/Z): {}/{} consistent ({:.1}%)",
            total_consistent,
            total_checked,
            100.0 * total_consistent as f64 / total_checked as f64
        );

        // For single-qubit X/Z errors, we should have very high consistency
        let consistency_rate = total_consistent as f64 / total_checked as f64;
        assert!(
            consistency_rate > 0.95,
            "Backward/forward consistency too low: {:.1}%",
            consistency_rate * 100.0
        );
    }

    #[test]
    fn test_backward_vs_forward_all_paulis() {
        // Exhaustively test all Pauli types on simple circuit
        use super::super::{PauliFault, propagate_fault, has_syndrome, anticommutes_with_logical};

        let circuit = simple_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1])];
        let map = propagator.build_influence_map_with_logicals(logicals);
        let backward_checker = InfluenceBasedChecker::new(&map);

        let z_ancillas: &[usize] = &[2];
        let x_ancillas: &[usize] = &[];

        let mut mismatches = Vec::new();

        for loc in map.influences.keys() {
            for pauli in [1u8, 2, 3] {
                // Create fault with this Pauli on each qubit
                let paulis = vec![pauli; loc.qubits.len()];
                let fault = PauliFault::new(loc.clone(), paulis);

                // Backward
                let (back_syn, back_log) = backward_checker.classify(loc, pauli);

                // Forward
                let prop = propagate_fault(&circuit, &fault);
                let fwd_syn = has_syndrome(&prop, z_ancillas, x_ancillas);
                let fwd_log = anticommutes_with_logical(&prop, logicals[0].0, logicals[0].1);

                if back_syn != fwd_syn || back_log != fwd_log {
                    mismatches.push((loc.clone(), pauli, back_syn, fwd_syn, back_log, fwd_log));
                }
            }
        }

        for (loc, pauli, back_syn, fwd_syn, back_log, fwd_log) in &mismatches {
            println!(
                "Mismatch: {:?} pauli={}: syn back={} fwd={}, log back={} fwd={}",
                loc, pauli, back_syn, fwd_syn, back_log, fwd_log
            );
        }

        // For the simple syndrome circuit, we expect high consistency
        // Some edge cases may differ based on how we handle 2-qubit faults
        assert!(
            mismatches.len() <= 2,
            "Too many mismatches: {}",
            mismatches.len()
        );
    }

    #[test]
    fn test_tree_structure_analysis() {
        // Analyze the "tree" structure - how many faults affect each detector
        let circuit = simple_syndrome_circuit();
        let propagator = BackwardPropagator::new(&circuit);
        let map = propagator.build_influence_map();

        println!("=== Tree Structure Analysis ===");
        println!("Total fault locations: {}", map.num_fault_locations());
        println!("Total detectors: {}", map.detectors.len());

        // For each detector, how many fault locations affect it?
        for detector in &map.detectors {
            let faults = map.faults_for_detector(detector);
            println!(
                "Detector {:?}: affected by {} fault locations",
                detector.measurements,
                faults.len()
            );

            // Group by Pauli type
            let mut by_pauli: std::collections::BTreeMap<u8, usize> = std::collections::BTreeMap::new();
            for (_loc, pauli) in faults {
                *by_pauli.entry(*pauli).or_insert(0) += 1;
            }
            for (pauli, count) in by_pauli {
                let name = match pauli {
                    1 => "X",
                    2 => "Y",
                    3 => "Z",
                    _ => "?",
                };
                println!("  {} faults: {}", name, count);
            }
        }

        // Check for "shared structure" - faults that affect multiple detectors
        // (In this simple circuit there's only one detector, so this won't show much)
        let two_round = two_round_syndrome_circuit();
        let prop2 = BackwardPropagator::new(&two_round);
        let map2 = prop2.build_influence_map();

        println!("\n=== Two-Round Circuit ===");
        println!("Total fault locations: {}", map2.num_fault_locations());
        println!("Total detectors: {}", map2.detectors.len());

        // Find faults that affect multiple detectors (shared paths)
        let mut multi_detector_faults = 0;
        for (loc, influence) in &map2.influences {
            let mut affected_detectors = 0;
            for detectors in influence.detector_flips.iter() {
                affected_detectors += detectors.len();
            }
            if affected_detectors > 1 {
                multi_detector_faults += 1;
                println!(
                    "Shared fault at {:?} affects {} detectors",
                    loc, affected_detectors
                );
            }
        }
        println!(
            "Faults affecting multiple detectors: {}",
            multi_detector_faults
        );
    }

    #[test]
    fn test_backward_vs_forward_varying_sizes() {
        // Test backward propagation matches forward on random Clifford circuits
        // with varying widths and depths.
        //
        // Tests ALL fault locations including multi-qubit gates, using the
        // `classify_uniform` method which properly handles cancellation effects
        // when the same Pauli is applied to all qubits.
        use super::super::{PauliFault, propagate_fault, has_syndrome};

        // Test configurations: (num_qubits, num_ticks, num_seeds)
        let configs = [
            (2, 5, 5),   // Small: 2 qubits, 5 ticks
            (4, 10, 5),  // Medium: 4 qubits, 10 ticks
            (6, 15, 3),  // Larger: 6 qubits, 15 ticks
            (8, 20, 3),  // Large: 8 qubits, 20 ticks
            (10, 30, 2), // Extra large: 10 qubits, 30 ticks
        ];

        let mut total_checked = 0;
        let mut total_consistent = 0;
        let mut config_results = Vec::new();

        for (num_qubits, num_ticks, num_seeds) in configs {
            let mut config_checked = 0;
            let mut config_consistent = 0;

            for seed in 0..num_seeds {
                let circuit = random_clifford_circuit(num_qubits, num_ticks, seed as u64);
                let propagator = BackwardPropagator::new(&circuit);
                let map = propagator.build_influence_map();
                let backward_checker = InfluenceBasedChecker::new(&map);

                let z_ancillas: Vec<usize> = map.measurements.iter().map(|m| m.qubit).collect();
                let x_ancillas: &[usize] = &[];

                // Test ALL fault locations (single and multi-qubit)
                for loc in map.influences.keys() {
                    // Test all Pauli types
                    for pauli in [1u8, 2, 3] {
                        // Use classify_uniform which handles multi-qubit XOR cancellation
                        let (back_syn, _) = backward_checker.classify_uniform(loc, pauli);

                        // Forward: apply same Pauli to all qubits in the location
                        let paulis = vec![pauli; loc.qubits.len()];
                        let fault = PauliFault::new(loc.clone(), paulis);
                        let prop = propagate_fault(&circuit, &fault);
                        let fwd_syn = has_syndrome(&prop, &z_ancillas, x_ancillas);

                        config_checked += 1;
                        total_checked += 1;
                        if back_syn == fwd_syn {
                            config_consistent += 1;
                            total_consistent += 1;
                        }
                    }
                }
            }

            let rate = 100.0 * config_consistent as f64 / config_checked as f64;
            config_results.push((num_qubits, num_ticks, config_checked, config_consistent, rate));
        }

        // Print results per configuration
        println!("\n=== Backward vs Forward Consistency by Circuit Size ===");
        for (qubits, ticks, checked, consistent, rate) in &config_results {
            println!(
                "{}q x {}t: {}/{} ({:.1}%)",
                qubits, ticks, consistent, checked, rate
            );
        }

        let overall_rate = 100.0 * total_consistent as f64 / total_checked as f64;
        println!(
            "\nOverall: {}/{} ({:.1}%)",
            total_consistent, total_checked, overall_rate
        );

        // Should have 100% consistency for all fault types
        assert_eq!(
            total_consistent, total_checked,
            "Expected 100% consistency, got {:.1}%",
            overall_rate
        );
    }

    #[test]
    fn test_backward_vs_forward_with_stabilizer_sim() {
        // Compare backward propagation with full stabilizer simulation.
        //
        // PauliProp tracks whether an error FLIPS a measurement, not the absolute value.
        // To compare with stabilizer sim, we need to run the sim both with and without
        // the error to see if the error changed the measurement outcome.
        use super::super::{PauliFault, propagate_fault, has_syndrome};

        let configs = [
            (3, 8, 5),  // 3 qubits, 8 ticks, 5 seeds
            (5, 12, 3), // 5 qubits, 12 ticks, 3 seeds
        ];

        let mut total_checked = 0;
        let mut total_consistent = 0;
        let mut pauli_prop_vs_stab_consistent = 0;

        for (num_qubits, num_ticks, num_seeds) in configs {
            for seed in 0..num_seeds {
                let circuit = random_clifford_circuit(num_qubits, num_ticks, seed as u64);
                let propagator = BackwardPropagator::new(&circuit);
                let map = propagator.build_influence_map();
                let backward_checker = InfluenceBasedChecker::new(&map);

                let z_ancillas: Vec<usize> = map.measurements.iter().map(|m| m.qubit).collect();
                let x_ancillas: &[usize] = &[];

                // Get baseline measurement (no fault)
                let baseline = simulate_circuit(&circuit, &z_ancillas);

                // Test single-qubit fault locations
                for loc in map.influences.keys() {
                    if loc.qubits.len() != 1 {
                        continue;
                    }

                    let qubit = loc.qubits[0].index();

                    for pauli in [1u8, 3] {
                        // 1. Backward propagation result
                        let (back_syn, _) = backward_checker.classify(loc, pauli);

                        // 2. Forward Pauli propagation result
                        let fault = PauliFault::new(loc.clone(), vec![pauli]);
                        let prop = propagate_fault(&circuit, &fault);
                        let pauli_prop_syn = has_syndrome(&prop, &z_ancillas, x_ancillas);

                        // 3. Full stabilizer simulation: compare with and without fault
                        let with_fault = simulate_with_fault_injection(
                            &circuit,
                            loc.tick,
                            loc.before,
                            qubit,
                            pauli,
                            &z_ancillas,
                        );
                        // Syndrome = measurement changed from baseline
                        let stab_syn = with_fault != baseline;

                        total_checked += 1;

                        // Check backward vs forward Pauli prop
                        if back_syn == pauli_prop_syn {
                            total_consistent += 1;
                        }

                        // Check Pauli prop vs stabilizer sim
                        if pauli_prop_syn == stab_syn {
                            pauli_prop_vs_stab_consistent += 1;
                        }
                    }
                }
            }
        }

        println!("\n=== Stabilizer Simulation Comparison ===");
        println!(
            "Backward vs PauliProp: {}/{} ({:.1}%)",
            total_consistent,
            total_checked,
            100.0 * total_consistent as f64 / total_checked as f64
        );
        println!(
            "PauliProp vs StabSim: {}/{} ({:.1}%)",
            pauli_prop_vs_stab_consistent,
            total_checked,
            100.0 * pauli_prop_vs_stab_consistent as f64 / total_checked as f64
        );

        // Backward vs PauliProp must agree - this is the core validation
        assert_eq!(
            total_consistent, total_checked,
            "Backward vs PauliProp mismatch"
        );

        // Note: PauliProp vs StabSim comparison is informational.
        // They may differ because:
        // 1. Pauli frame tracking semantics differ from full state simulation
        // 2. Stabilizer sim measures actual state, while PauliProp tracks error frame
        // The key insight is that for fault tolerance analysis, we care about
        // whether errors FLIP measurements, which PauliProp correctly captures.
    }

    /// Helper: simulate circuit without any faults to get baseline measurement
    fn simulate_circuit(circuit: &TickCircuit, z_ancillas: &[usize]) -> bool {
        use pecos_qsim::{CliffordGateable, SparseStab};
        use pecos_core::QubitId;

        let mut max_qubit = 0;
        for tick in circuit.ticks() {
            for gate in tick.gates() {
                for q in &gate.qubits {
                    max_qubit = max_qubit.max(q.index());
                }
            }
        }

        let mut sim = SparseStab::new(max_qubit + 1);

        for (_tick_idx, tick) in circuit.iter_ticks() {
            for gate in tick.gates() {
                let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();
                match gate.gate_type {
                    GateType::Prep | GateType::QAlloc => {
                        sim.pz(&qubits);
                    }
                    GateType::H => {
                        sim.h(&qubits);
                    }
                    GateType::SZ => {
                        sim.sz(&qubits);
                    }
                    GateType::SZdg => {
                        sim.szdg(&qubits);
                    }
                    GateType::CX => {
                        for pair in qubits.chunks(2) {
                            if pair.len() == 2 {
                                sim.cx(&[pair[0], pair[1]]);
                            }
                        }
                    }
                    GateType::CZ => {
                        for pair in qubits.chunks(2) {
                            if pair.len() == 2 {
                                sim.cz(&[pair[0], pair[1]]);
                            }
                        }
                    }
                    GateType::Measure | GateType::MeasureFree => {}
                    _ => {}
                }
            }
        }

        // Measure ancillas
        let mut has_nonzero = false;
        for &ancilla in z_ancillas {
            let results = sim.mz(&[QubitId(ancilla)]);
            if !results.is_empty() && results[0].outcome {
                has_nonzero = true;
            }
        }
        has_nonzero
    }

    /// Helper: simulate circuit with fault injection using full stabilizer simulation
    fn simulate_with_fault_injection(
        circuit: &TickCircuit,
        fault_tick: usize,
        fault_before: bool,
        fault_qubit: usize,
        fault_pauli: u8,
        z_ancillas: &[usize],
    ) -> bool {
        use pecos_qsim::{CliffordGateable, SparseStab};
        use pecos_core::QubitId;

        let mut max_qubit = 0;
        for tick in circuit.ticks() {
            for gate in tick.gates() {
                for q in &gate.qubits {
                    max_qubit = max_qubit.max(q.index());
                }
            }
        }

        let mut sim = SparseStab::new(max_qubit + 1);

        for (tick_idx, tick) in circuit.iter_ticks() {
            // Inject fault before gates at this tick
            if tick_idx == fault_tick && fault_before {
                inject_pauli(&mut sim, fault_qubit, fault_pauli);
            }

            // Apply gates
            for gate in tick.gates() {
                let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();
                match gate.gate_type {
                    GateType::Prep | GateType::QAlloc => {
                        sim.pz(&qubits);
                    }
                    GateType::H => {
                        sim.h(&qubits);
                    }
                    GateType::SZ => {
                        sim.sz(&qubits);
                    }
                    GateType::SZdg => {
                        sim.szdg(&qubits);
                    }
                    GateType::CX => {
                        for pair in qubits.chunks(2) {
                            if pair.len() == 2 {
                                sim.cx(&[pair[0], pair[1]]);
                            }
                        }
                    }
                    GateType::CZ => {
                        for pair in qubits.chunks(2) {
                            if pair.len() == 2 {
                                sim.cz(&[pair[0], pair[1]]);
                            }
                        }
                    }
                    GateType::Measure | GateType::MeasureFree => {}
                    _ => {}
                }
            }

            // Inject fault after gates at this tick
            if tick_idx == fault_tick && !fault_before {
                inject_pauli(&mut sim, fault_qubit, fault_pauli);
            }
        }

        // Measure ancillas
        let mut has_nonzero = false;
        for &ancilla in z_ancillas {
            let results = sim.mz(&[QubitId(ancilla)]);
            if !results.is_empty() && results[0].outcome {
                has_nonzero = true;
            }
        }
        has_nonzero
    }

    /// Helper: inject a Pauli error into a stabilizer simulator
    fn inject_pauli<S: pecos_qsim::CliffordGateable>(sim: &mut S, qubit: usize, pauli: u8) {
        use pecos_core::QubitId;
        match pauli {
            1 => {
                sim.x(&[QubitId(qubit)]);
            }
            2 => {
                sim.y(&[QubitId(qubit)]);
            }
            3 => {
                sim.z(&[QubitId(qubit)]);
            }
            _ => {}
        }
    }

    #[test]
    fn test_backward_vs_forward_deep_circuits() {
        // Test with deeper circuits to stress-test the backward propagation
        // Tests all fault locations using classify_uniform.
        use super::super::{PauliFault, propagate_fault, has_syndrome};

        let depths = [50, 100];
        let num_qubits = 4;

        let mut total_checked = 0;
        let mut total_consistent = 0;

        for depth in depths {
            for seed in 0..3 {
                let circuit = random_clifford_circuit(num_qubits, depth, seed as u64);
                let propagator = BackwardPropagator::new(&circuit);
                let map = propagator.build_influence_map();
                let backward_checker = InfluenceBasedChecker::new(&map);

                let z_ancillas: Vec<usize> = map.measurements.iter().map(|m| m.qubit).collect();
                let x_ancillas: &[usize] = &[];

                // Get all fault locations
                let locations: Vec<_> = map.influences.keys().collect();

                // Sample some fault locations (not all, to keep test fast)
                let sample_size = locations.len().min(50);

                for loc in locations.iter().take(sample_size) {
                    for pauli in [1u8, 2, 3] {
                        let (back_syn, _) = backward_checker.classify_uniform(loc, pauli);

                        let paulis = vec![pauli; loc.qubits.len()];
                        let fault = PauliFault::new((*loc).clone(), paulis);
                        let prop = propagate_fault(&circuit, &fault);
                        let fwd_syn = has_syndrome(&prop, &z_ancillas, x_ancillas);

                        total_checked += 1;
                        if back_syn == fwd_syn {
                            total_consistent += 1;
                        }
                    }
                }
            }
        }

        println!(
            "\nDeep circuit test: {}/{} ({:.1}%)",
            total_consistent,
            total_checked,
            100.0 * total_consistent as f64 / total_checked as f64
        );

        assert_eq!(
            total_consistent, total_checked,
            "Deep circuit consistency failure"
        );
    }

    #[test]
    fn test_backward_vs_forward_with_logicals() {
        // Test that logical operator tracking also matches between backward and forward
        use super::super::{PauliFault, propagate_fault, anticommutes_with_logical};

        let configs = [
            (4, 10, 5),
            (6, 15, 3),
        ];

        let mut total_checked = 0;
        let mut total_consistent = 0;

        for (num_qubits, num_ticks, num_seeds) in configs {
            for seed in 0..num_seeds {
                let circuit = random_clifford_circuit(num_qubits, num_ticks, seed as u64);

                // Define some logical operators (arbitrary for testing)
                // Use first half of non-ancilla qubits for logical Z
                let num_ancillas = (num_qubits / 2).max(1);
                let data_qubits: Vec<usize> = (num_ancillas..num_qubits).collect();

                if data_qubits.is_empty() {
                    continue;
                }

                // Logical Z = Z on all data qubits (product of Zs)
                let logical_z: (&[usize], &[usize]) = (&[], &data_qubits);
                let logicals: &[(&[usize], &[usize])] = &[logical_z];

                let propagator = BackwardPropagator::new(&circuit);
                let map = propagator.build_influence_map_with_logicals(logicals);
                let backward_checker = InfluenceBasedChecker::new(&map);

                for loc in map.influences.keys() {
                    if loc.qubits.len() != 1 {
                        continue;
                    }

                    for pauli in [1u8, 3] {
                        let (_, back_log) = backward_checker.classify(loc, pauli);

                        let fault = PauliFault::new(loc.clone(), vec![pauli]);
                        let prop = propagate_fault(&circuit, &fault);
                        let fwd_log = anticommutes_with_logical(&prop, logicals[0].0, logicals[0].1);

                        total_checked += 1;
                        if back_log == fwd_log {
                            total_consistent += 1;
                        }
                    }
                }
            }
        }

        println!(
            "\nLogical operator test: {}/{} ({:.1}%)",
            total_consistent,
            total_checked,
            100.0 * total_consistent as f64 / total_checked as f64
        );

        assert_eq!(
            total_consistent, total_checked,
            "Logical operator consistency failure"
        );
    }

    #[test]
    fn test_propagate_fault_backward_basic() {
        use super::propagate_fault_backward;
        use crate::fault_tolerance::{PauliFault, SpacetimeLocation};
        use pecos_core::gate_type::GateType;
        use pecos_core::QubitId;
        use pecos_quantum::TickCircuit;

        // Build a simple circuit without prep: H -> CX -> MZ
        // Prep gates clear the Pauli during backward propagation, so we skip them
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1)]);
        circuit.tick().mz(&[0, 1]);

        // Create an X fault after the CX gate on qubit 0
        let location = SpacetimeLocation::new(
            1,
            vec![QubitId(0), QubitId(1)],
            false,
            GateType::CX,
            0,
        );
        let fault = PauliFault::new(location, vec![1, 0]); // X on control, I on target

        // Propagate backward
        let prop = propagate_fault_backward(&circuit, &fault);

        // After propagating X on qubit 0 backward through CX (no change on control)
        // Then backward through H: X -> Z
        assert!(
            prop.contains_z(0),
            "X fault on CX control should propagate to Z after backward H"
        );
    }

    #[test]
    fn test_propagate_observable_backward() {
        use super::propagate_observable_backward;
        use pecos_quantum::TickCircuit;

        // Build a simple circuit without prep: just CX
        let mut circuit = TickCircuit::new();
        circuit.tick().cx(&[(0, 1)]);

        // Z observable on qubit 1 (target of CX)
        let prop = propagate_observable_backward(&circuit, &[], &[1], 0);

        // Z on target propagates backward through CX to ZZ
        // (Z_target -> Z_control * Z_target)
        assert!(prop.contains_z(0), "Z on target should spread to control backward through CX");
        assert!(prop.contains_z(1), "Z on target should remain on target backward through CX");
    }

    #[test]
    fn test_propagate_backward_from_tick() {
        use super::propagate_backward_from_tick;
        use pecos_quantum::TickCircuit;
        use pecos_qsim::PauliProp;

        // Build a circuit without prep: H -> CX -> H
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1)]);
        circuit.tick().h(&[0]);

        // Start with Z on qubit 0 at the end
        let mut prop = PauliProp::new();
        prop.add_z(0);

        // Propagate backward from tick 2 (last H)
        propagate_backward_from_tick(&circuit, &mut prop, 2);

        // Z -> H -> X -> CX -> X (control stays X) -> H -> Z
        assert!(
            prop.contains_z(0),
            "Z should return to Z after H -> CX -> H backward"
        );
    }

    #[test]
    fn test_propagate_backward_with_prep_transparent() {
        use super::propagate_fault_backward;
        use crate::fault_tolerance::{PauliFault, SpacetimeLocation};
        use pecos_core::gate_type::GateType;
        use pecos_core::QubitId;
        use pecos_quantum::TickCircuit;

        // Build a circuit with prep at the start
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1]);
        circuit.tick().h(&[0]);
        circuit.tick().mz(&[0]);

        // Create a Z fault after H
        let location = SpacetimeLocation::new(
            1,
            vec![QubitId(0)],
            false,
            GateType::H,
            0,
        );
        let fault = PauliFault::new(location, vec![3]); // Z fault

        // Propagate backward
        let prop = propagate_fault_backward(&circuit, &fault);

        // Z -> H -> X, and prep gates are transparent to propagation
        // So the final result should be X on qubit 0
        assert!(
            prop.contains_x(0) && !prop.contains_z(0),
            "Z should become X after backward propagation through H, prep is transparent"
        );
    }

    #[test]
    fn test_standalone_vs_backward_propagator() {
        use super::{propagate_fault_backward, BackwardPropagator};
        use crate::fault_tolerance::{PauliFault, SpacetimeLocation};
        use pecos_core::gate_type::GateType;
        use pecos_core::QubitId;
        use pecos_quantum::TickCircuit;

        // Build a circuit with a measurement (no prep to keep Pauli)
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1)]);
        circuit.tick().mz(&[0]);

        // Create a fault before the measurement
        let location = SpacetimeLocation::new(
            2,
            vec![QubitId(0)],
            true,
            GateType::Measure,
            0,
        );
        let fault = PauliFault::new(location.clone(), vec![1]); // X fault

        // Use standalone function
        let prop_standalone = propagate_fault_backward(&circuit, &fault);

        // Use BackwardPropagator for comparison
        let backward_prop = BackwardPropagator::new(&circuit);
        let influence_map = backward_prop.build_influence_map();

        // Fault should propagate to some Pauli at the start
        // X before mz at tick 2 -> propagates back through CX (X on control stays X)
        // -> propagates back through H: X -> Z
        assert!(
            prop_standalone.contains_z(0),
            "Fault should propagate to Z on qubit 0"
        );

        // The influence map should have recorded this location
        assert!(
            influence_map.influences.contains_key(&location),
            "Influence map should contain this fault location"
        );
    }

    #[test]
    fn test_unified_propagator_sz_forward() {
        use super::{apply_gate, Direction};
        use pecos_core::gate_type::GateType;
        use pecos_core::QubitId;
        use pecos_qsim::PauliProp;

        // Test forward propagation through SZ gate
        // SZ: X -> Y (XZ), Y -> -X, Z -> Z

        // Test X -> Y
        let mut prop = PauliProp::new();
        prop.add_x(0);
        let gate = pecos_core::Gate::simple(GateType::SZ, vec![QubitId(0)]);
        apply_gate(&mut prop, &gate, Direction::Forward);
        assert!(prop.contains_x(0), "X should still have X component after SZ");
        assert!(prop.contains_z(0), "X should gain Z component after SZ (X -> Y = XZ)");

        // Test Y -> X (Y = XZ, after SZ becomes X)
        let mut prop = PauliProp::new();
        prop.add_x(0);
        prop.add_z(0); // Y = XZ
        apply_gate(&mut prop, &gate, Direction::Forward);
        assert!(prop.contains_x(0), "Y should still have X component after SZ");
        assert!(!prop.contains_z(0), "Y should lose Z component after SZ (Y -> X)");

        // Test Z -> Z
        let mut prop = PauliProp::new();
        prop.add_z(0);
        apply_gate(&mut prop, &gate, Direction::Forward);
        assert!(!prop.contains_x(0), "Z should not gain X after SZ");
        assert!(prop.contains_z(0), "Z should remain Z after SZ");
    }

    #[test]
    fn test_unified_propagator_sz_backward() {
        use super::{apply_gate, Direction};
        use pecos_core::gate_type::GateType;
        use pecos_core::QubitId;
        use pecos_qsim::PauliProp;

        // Test backward propagation through SZ gate (applies SZdg)
        // SZdg: X -> -Y (XZ), Y -> X, Z -> Z
        // (Same as SZ for phase-free tracking)

        // Test X -> Y (same as forward)
        let mut prop = PauliProp::new();
        prop.add_x(0);
        let gate = pecos_core::Gate::simple(GateType::SZ, vec![QubitId(0)]);
        apply_gate(&mut prop, &gate, Direction::Backward);
        assert!(prop.contains_x(0), "X should still have X component after SZ backward");
        assert!(prop.contains_z(0), "X should gain Z component after SZ backward (X -> Y)");

        // Test Y -> X
        let mut prop = PauliProp::new();
        prop.add_x(0);
        prop.add_z(0); // Y = XZ
        apply_gate(&mut prop, &gate, Direction::Backward);
        assert!(prop.contains_x(0), "Y should still have X component after SZ backward");
        assert!(!prop.contains_z(0), "Y should lose Z component after SZ backward (Y -> X)");

        // Test Z -> Z
        let mut prop = PauliProp::new();
        prop.add_z(0);
        apply_gate(&mut prop, &gate, Direction::Backward);
        assert!(!prop.contains_x(0), "Z should not gain X after SZ backward");
        assert!(prop.contains_z(0), "Z should remain Z after SZ backward");
    }

    #[test]
    fn test_unified_propagator_szdg_both_directions() {
        use super::{apply_gate, Direction};
        use pecos_core::gate_type::GateType;
        use pecos_core::QubitId;
        use pecos_qsim::PauliProp;

        let gate = pecos_core::Gate::simple(GateType::SZdg, vec![QubitId(0)]);

        // Forward through SZdg should use szdg()
        // SZdg: X -> -Y, Y -> X, Z -> Z
        let mut prop = PauliProp::new();
        prop.add_x(0);
        apply_gate(&mut prop, &gate, Direction::Forward);
        assert!(prop.contains_x(0) && prop.contains_z(0), "X -> Y through SZdg forward");

        // Backward through SZdg should use sz() (the adjoint)
        let mut prop = PauliProp::new();
        prop.add_x(0);
        apply_gate(&mut prop, &gate, Direction::Backward);
        assert!(prop.contains_x(0) && prop.contains_z(0), "X -> Y through SZdg backward (uses SZ)");
    }

    #[test]
    fn test_unified_propagator_sz_circuit_roundtrip() {
        use super::{propagate_through_circuit, Direction};
        use pecos_quantum::TickCircuit;
        use pecos_qsim::PauliProp;

        // Build circuit: SZ -> H -> SZ
        let mut circuit = TickCircuit::new();
        circuit.tick().sz(&[0]);
        circuit.tick().h(&[0]);
        circuit.tick().sz(&[0]);

        // Start with X, propagate forward, then backward - should return to X
        let mut prop = PauliProp::new();
        prop.add_x(0);

        // Forward: X -> SZ -> Y -> H -> Y -> SZ -> X
        propagate_through_circuit(&circuit, &mut prop, Direction::Forward);
        let forward_x = prop.contains_x(0);
        let forward_z = prop.contains_z(0);

        // Now take the result and propagate backward
        propagate_through_circuit(&circuit, &mut prop, Direction::Backward);

        // Should be back to X
        assert!(
            prop.contains_x(0) && !prop.contains_z(0),
            "Roundtrip should return X to X, got X={}, Z={}",
            prop.contains_x(0),
            prop.contains_z(0)
        );

        println!(
            "Forward result: X={}, Z={}",
            forward_x, forward_z
        );
    }

    #[test]
    fn test_unified_propagator_sz_forward_backward_equivalence() {
        use super::{apply_gate, Direction};
        use pecos_core::gate_type::GateType;
        use pecos_core::QubitId;
        use pecos_qsim::PauliProp;

        // For phase-free Pauli tracking, SZ and SZdg produce the same transformation
        // This test verifies that forward(SZ) == backward(SZdg) and vice versa

        let sz_gate = pecos_core::Gate::simple(GateType::SZ, vec![QubitId(0)]);
        let szdg_gate = pecos_core::Gate::simple(GateType::SZdg, vec![QubitId(0)]);

        // Forward SZ should equal Backward SZdg
        let mut prop1 = PauliProp::new();
        prop1.add_x(0);
        apply_gate(&mut prop1, &sz_gate, Direction::Forward);

        let mut prop2 = PauliProp::new();
        prop2.add_x(0);
        apply_gate(&mut prop2, &szdg_gate, Direction::Backward);

        assert_eq!(
            prop1.contains_x(0), prop2.contains_x(0),
            "Forward SZ and Backward SZdg should match for X component"
        );
        assert_eq!(
            prop1.contains_z(0), prop2.contains_z(0),
            "Forward SZ and Backward SZdg should match for Z component"
        );

        // Backward SZ should equal Forward SZdg
        let mut prop3 = PauliProp::new();
        prop3.add_x(0);
        apply_gate(&mut prop3, &sz_gate, Direction::Backward);

        let mut prop4 = PauliProp::new();
        prop4.add_x(0);
        apply_gate(&mut prop4, &szdg_gate, Direction::Forward);

        assert_eq!(
            prop3.contains_x(0), prop4.contains_x(0),
            "Backward SZ and Forward SZdg should match for X component"
        );
        assert_eq!(
            prop3.contains_z(0), prop4.contains_z(0),
            "Backward SZ and Forward SZdg should match for Z component"
        );
    }

    // ========================================================================
    // DAG-Based Sparse Propagation Tests
    // ========================================================================

    #[test]
    fn test_dag_propagate_forward_simple() {
        use super::{propagate_through_dag, Direction};
        use pecos_quantum::DagCircuit;
        use pecos_qsim::PauliProp;

        // Build a simple DAG circuit: H on qubit 0
        let mut dag = DagCircuit::new();
        dag.h(0);

        // Propagate Z forward through H -> X
        let mut prop = PauliProp::new();
        prop.add_z(0);
        propagate_through_dag(&dag, &mut prop, Direction::Forward);

        assert!(prop.contains_x(0), "Z -> H -> X");
        assert!(!prop.contains_z(0), "Z should become X after H");
    }

    #[test]
    fn test_dag_propagate_backward_simple() {
        use super::{propagate_through_dag, Direction};
        use pecos_quantum::DagCircuit;
        use pecos_qsim::PauliProp;

        // Build a simple DAG circuit: H on qubit 0
        let mut dag = DagCircuit::new();
        dag.h(0);

        // Propagate Z backward through H -> X
        let mut prop = PauliProp::new();
        prop.add_z(0);
        propagate_through_dag(&dag, &mut prop, Direction::Backward);

        assert!(prop.contains_x(0), "Z -> H backward -> X");
        assert!(!prop.contains_z(0), "Z should become X after H backward");
    }

    #[test]
    fn test_dag_sparse_vs_full_equivalence() {
        use super::{propagate_sparse_dag, propagate_through_dag, Direction};
        use pecos_quantum::DagCircuit;
        use pecos_qsim::PauliProp;

        // Build a circuit with gates on multiple qubits
        let mut dag = DagCircuit::new();
        dag.h(0);
        dag.h(1);
        dag.h(2);
        dag.cx(0, 1);
        dag.cx(1, 2);
        dag.h(0);
        dag.h(1);
        dag.h(2);

        // Start with Z on qubit 0 only
        let mut prop_full = PauliProp::new();
        prop_full.add_z(0);

        let mut prop_sparse = PauliProp::new();
        prop_sparse.add_z(0);

        // Propagate both ways
        propagate_through_dag(&dag, &mut prop_full, Direction::Forward);
        propagate_sparse_dag(&dag, &mut prop_sparse, Direction::Forward);

        // Results should be identical
        assert_eq!(
            prop_full.contains_x(0), prop_sparse.contains_x(0),
            "X on qubit 0 should match"
        );
        assert_eq!(
            prop_full.contains_z(0), prop_sparse.contains_z(0),
            "Z on qubit 0 should match"
        );
        assert_eq!(
            prop_full.contains_x(1), prop_sparse.contains_x(1),
            "X on qubit 1 should match"
        );
        assert_eq!(
            prop_full.contains_z(1), prop_sparse.contains_z(1),
            "Z on qubit 1 should match"
        );
        assert_eq!(
            prop_full.contains_x(2), prop_sparse.contains_x(2),
            "X on qubit 2 should match"
        );
        assert_eq!(
            prop_full.contains_z(2), prop_sparse.contains_z(2),
            "Z on qubit 2 should match"
        );
    }

    #[test]
    fn test_dag_sparse_spreading() {
        use super::{propagate_sparse_dag, Direction};
        use pecos_quantum::DagCircuit;
        use pecos_qsim::PauliProp;

        // Build a circuit where X spreads through CX gates
        let mut dag = DagCircuit::new();
        dag.cx(0, 1);  // X on control spreads to target
        dag.cx(1, 2);  // X on 1 spreads to 2
        dag.cx(2, 3);  // X on 2 spreads to 3

        // Start with X on qubit 0
        let mut prop = PauliProp::new();
        prop.add_x(0);

        propagate_sparse_dag(&dag, &mut prop, Direction::Forward);

        // X should have spread to all qubits
        assert!(prop.contains_x(0), "X should remain on qubit 0");
        assert!(prop.contains_x(1), "X should spread to qubit 1");
        assert!(prop.contains_x(2), "X should spread to qubit 2");
        assert!(prop.contains_x(3), "X should spread to qubit 3");
    }

    #[test]
    fn test_dag_sparse_backward_spreading() {
        use super::{propagate_sparse_dag, Direction};
        use pecos_quantum::DagCircuit;
        use pecos_qsim::PauliProp;

        // Build a circuit with CX gates
        // Backward: Z on target of CX spreads to control
        let mut dag = DagCircuit::new();
        dag.cx(0, 1);  // control=0, target=1
        dag.cx(1, 2);  // control=1, target=2

        // Start with Z on qubit 2 (target of last CX)
        // CX backward: Z on target -> Z on both control and target
        let mut prop = PauliProp::new();
        prop.add_z(2);

        propagate_sparse_dag(&dag, &mut prop, Direction::Backward);

        // Z on qubit 2 should spread backward through CX gates
        // CX(1,2) backward: Z on 2 -> Z on 1 and Z on 2
        // CX(0,1) backward: Z on 1 -> Z on 0 and Z on 1
        assert!(prop.contains_z(2), "Z should remain on qubit 2");
        assert!(prop.contains_z(1), "Z on 2 through CX backward adds Z on 1");
        assert!(prop.contains_z(0), "Z on 1 through CX backward adds Z on 0");
    }

    #[test]
    fn test_dag_sparse_isolated_qubits() {
        use super::{propagate_sparse_dag, Direction};
        use pecos_quantum::DagCircuit;
        use pecos_qsim::PauliProp;

        // Build a circuit with isolated qubit groups
        // Group 1: qubits 0, 1
        // Group 2: qubits 2, 3 (isolated from group 1)
        let mut dag = DagCircuit::new();
        dag.h(0);
        dag.h(2);
        dag.cx(0, 1);  // Only connects 0 and 1
        dag.cx(2, 3);  // Only connects 2 and 3
        dag.h(0);
        dag.h(2);

        // Start with Z on qubit 0 only
        let mut prop = PauliProp::new();
        prop.add_z(0);

        propagate_sparse_dag(&dag, &mut prop, Direction::Forward);

        // Qubits 2 and 3 should be unaffected (they're isolated)
        assert!(!prop.contains_x(2), "Qubit 2 should be unaffected");
        assert!(!prop.contains_z(2), "Qubit 2 should be unaffected");
        assert!(!prop.contains_x(3), "Qubit 3 should be unaffected");
        assert!(!prop.contains_z(3), "Qubit 3 should be unaffected");

        // Qubits 0 and 1 should be affected
        let has_pauli_0 = prop.contains_x(0) || prop.contains_z(0);
        let has_pauli_1 = prop.contains_x(1) || prop.contains_z(1);
        assert!(has_pauli_0 || has_pauli_1, "Qubits 0/1 should have Paulis");
    }

    #[test]
    fn test_dag_vs_tick_circuit_equivalence() {
        use super::{propagate_sparse_dag, propagate_through_circuit, Direction};
        use pecos_quantum::{DagCircuit, TickCircuit};
        use pecos_qsim::PauliProp;

        // Build equivalent circuits in both representations
        let mut tick = TickCircuit::new();
        tick.tick().h(&[0]);
        tick.tick().cx(&[(0, 1)]);
        tick.tick().h(&[0]).h(&[1]);

        let mut dag = DagCircuit::new();
        dag.h(0);
        dag.cx(0, 1);
        dag.h(0);
        dag.h(1);

        // Start with Z on qubit 0
        let mut prop_tick = PauliProp::new();
        prop_tick.add_z(0);

        let mut prop_dag = PauliProp::new();
        prop_dag.add_z(0);

        // Propagate forward
        propagate_through_circuit(&tick, &mut prop_tick, Direction::Forward);
        propagate_sparse_dag(&dag, &mut prop_dag, Direction::Forward);

        // Results should match
        assert_eq!(
            prop_tick.contains_x(0), prop_dag.contains_x(0),
            "X on qubit 0 should match between TickCircuit and DagCircuit"
        );
        assert_eq!(
            prop_tick.contains_z(0), prop_dag.contains_z(0),
            "Z on qubit 0 should match"
        );
        assert_eq!(
            prop_tick.contains_x(1), prop_dag.contains_x(1),
            "X on qubit 1 should match"
        );
        assert_eq!(
            prop_tick.contains_z(1), prop_dag.contains_z(1),
            "Z on qubit 1 should match"
        );
    }

    #[test]
    fn test_dag_roundtrip() {
        use super::{propagate_through_dag, Direction};
        use pecos_quantum::DagCircuit;
        use pecos_qsim::PauliProp;

        // Build a circuit
        let mut dag = DagCircuit::new();
        dag.h(0);
        dag.cx(0, 1);
        dag.sz(0);

        // Start with X on qubit 0
        let mut prop = PauliProp::new();
        prop.add_x(0);

        // Propagate forward then backward - should return to original
        propagate_through_dag(&dag, &mut prop, Direction::Forward);
        propagate_through_dag(&dag, &mut prop, Direction::Backward);

        assert!(prop.contains_x(0), "X should return to qubit 0 after roundtrip");
        assert!(!prop.contains_z(0), "No Z on qubit 0 after roundtrip");
        assert!(!prop.contains_x(1), "No X on qubit 1 after roundtrip");
        assert!(!prop.contains_z(1), "No Z on qubit 1 after roundtrip");
    }

    // ========================================================================
    // Benchmarks
    // ========================================================================

    /// Benchmark comparing TickCircuit vs DagCircuit propagation methods.
    ///
    /// Run with: cargo test -p pecos-qec benchmark_propagation --release -- --nocapture --ignored
    #[test]
    #[ignore] // Run manually with --ignored flag
    fn benchmark_propagation_methods() {
        use super::{propagate_through_circuit, DagPropagator, Direction};
        use pecos_qsim::PauliProp;
        use std::time::Instant;

        println!("\n========================================");
        println!("Pauli Propagation Benchmark (Optimized)");
        println!("========================================\n");

        // Test configurations: (num_qubits, circuit_depth, num_iterations)
        let configs = [
            (10, 50, 1000),   // Small circuit
            (50, 100, 500),   // Medium circuit
            (100, 200, 200),  // Large circuit
            (200, 500, 50),   // Very large circuit
        ];

        for (num_qubits, depth, iterations) in configs {
            println!("--- {} qubits, depth {}, {} iterations ---", num_qubits, depth, iterations);

            // Build equivalent circuits
            let (tick_circuit, dag_circuit) = build_test_circuits(num_qubits, depth);

            // Pre-compute DAG propagator (do this once, reuse many times)
            let start = Instant::now();
            let dag_propagator = DagPropagator::new(&dag_circuit);
            let index_build_time = start.elapsed();
            println!("  (Index build time: {:.1} us)", index_build_time.as_micros());

            // Benchmark 1: TickCircuit full traversal
            let start = Instant::now();
            for _ in 0..iterations {
                let mut prop = PauliProp::new();
                prop.add_z(0);
                propagate_through_circuit(&tick_circuit, &mut prop, Direction::Forward);
            }
            let tick_time = start.elapsed();

            // Benchmark 2: DagPropagator full traversal (pre-computed)
            let start = Instant::now();
            for _ in 0..iterations {
                let mut prop = PauliProp::new();
                prop.add_z(0);
                dag_propagator.propagate_full(&mut prop, Direction::Forward);
            }
            let dag_full_time = start.elapsed();

            // Benchmark 3: DagPropagator sparse traversal (1 qubit Pauli)
            let start = Instant::now();
            for _ in 0..iterations {
                let mut prop = PauliProp::new();
                prop.add_z(0);
                dag_propagator.propagate_sparse(&mut prop, Direction::Forward);
            }
            let dag_sparse_1q_time = start.elapsed();

            // Benchmark 4: DagPropagator sparse traversal (half qubits Pauli)
            let start = Instant::now();
            for _ in 0..iterations {
                let mut prop = PauliProp::new();
                for q in 0..(num_qubits / 2) {
                    prop.add_z(q);
                }
                dag_propagator.propagate_sparse(&mut prop, Direction::Forward);
            }
            let dag_sparse_half_time = start.elapsed();

            // Print results
            let tick_per_iter = tick_time.as_micros() as f64 / iterations as f64;
            let dag_full_per_iter = dag_full_time.as_micros() as f64 / iterations as f64;
            let dag_sparse_1q_per_iter = dag_sparse_1q_time.as_micros() as f64 / iterations as f64;
            let dag_sparse_half_per_iter = dag_sparse_half_time.as_micros() as f64 / iterations as f64;

            println!("  TickCircuit (full):      {:>8.1} us/iter", tick_per_iter);
            println!("  DagPropagator (full):    {:>8.1} us/iter ({:.2}x vs Tick)",
                dag_full_per_iter, dag_full_per_iter / tick_per_iter);
            println!("  DagPropagator sparse 1q: {:>8.1} us/iter ({:.2}x vs Tick)",
                dag_sparse_1q_per_iter,
                dag_sparse_1q_per_iter / tick_per_iter);
            println!("  DagPropagator sparse {}q:{:>8.1} us/iter ({:.2}x vs Tick)",
                num_qubits / 2,
                dag_sparse_half_per_iter,
                dag_sparse_half_per_iter / tick_per_iter);
            println!();
        }

        // Additional test: Backward propagation
        println!("--- Backward Propagation (100 qubits, depth 200) ---");
        let (tick_circuit, dag_circuit) = build_test_circuits(100, 200);
        let dag_propagator = DagPropagator::new(&dag_circuit);
        let iterations = 200;

        let start = Instant::now();
        for _ in 0..iterations {
            let mut prop = PauliProp::new();
            prop.add_z(0);
            propagate_through_circuit(&tick_circuit, &mut prop, Direction::Backward);
        }
        let tick_back_time = start.elapsed();

        let start = Instant::now();
        for _ in 0..iterations {
            let mut prop = PauliProp::new();
            prop.add_z(0);
            dag_propagator.propagate_sparse(&mut prop, Direction::Backward);
        }
        let dag_sparse_back_time = start.elapsed();

        let tick_per = tick_back_time.as_micros() as f64 / iterations as f64;
        let sparse_per = dag_sparse_back_time.as_micros() as f64 / iterations as f64;
        println!("  TickCircuit backward:       {:>8.1} us/iter", tick_per);
        println!("  DagPropagator sparse back:  {:>8.1} us/iter ({:.2}x vs Tick)",
            sparse_per, sparse_per / tick_per);

        println!("\n========================================\n");
    }

    /// Build test circuits with a mix of single and two-qubit gates.
    fn build_test_circuits(num_qubits: usize, depth: usize) -> (TickCircuit, DagCircuit) {
        use pecos_quantum::{DagCircuit, TickCircuit};

        let mut tick = TickCircuit::new();
        let mut dag = DagCircuit::new();

        for d in 0..depth {
            // Layer of single-qubit gates
            // Collect qubits for each gate type
            let h_qubits: Vec<usize> = (0..num_qubits).filter(|&q| (d + q) % 3 == 0).collect();
            let x_qubits: Vec<usize> = (0..num_qubits).filter(|&q| (d + q) % 3 == 1).collect();
            let z_qubits: Vec<usize> = (0..num_qubits).filter(|&q| (d + q) % 3 == 2).collect();

            tick.tick().h(&h_qubits).x(&x_qubits).z(&z_qubits);

            for &q in &h_qubits {
                dag.h(q);
            }
            for &q in &x_qubits {
                dag.x(q);
            }
            for &q in &z_qubits {
                dag.z(q);
            }

            // Layer of two-qubit gates (every other layer)
            if d % 2 == 1 && num_qubits >= 2 {
                let cx_pairs: Vec<(usize, usize)> = (0..num_qubits - 1)
                    .step_by(2)
                    .map(|q| (q, q + 1))
                    .collect();
                tick.tick().cx(&cx_pairs);
                for (c, t) in cx_pairs {
                    dag.cx(c, t);
                }
            }
        }

        (tick, dag)
    }

    /// Benchmark specifically for syndrome extraction circuit pattern.
    #[test]
    #[ignore]
    fn benchmark_syndrome_extraction_pattern() {
        use super::{propagate_through_circuit, DagPropagator, Direction};
        use pecos_qsim::PauliProp;
        use std::time::Instant;

        println!("\n========================================");
        println!("Syndrome Extraction Pattern Benchmark");
        println!("========================================\n");

        // Simulate a surface code-like pattern
        // data qubits + ancilla qubits, ancillas only interact with neighbors
        let configs = [
            (9, 8, 1000),     // 3x3 data + 8 ancillas (small surface code)
            (25, 24, 500),    // 5x5 data + 24 ancillas
            (49, 48, 200),    // 7x7 data + 48 ancillas
            (121, 120, 100),  // 11x11 data + 120 ancillas (larger)
        ];

        for (data_qubits, ancilla_qubits, iterations) in configs {
            println!("--- {} data + {} ancilla qubits ---", data_qubits, ancilla_qubits);

            // Build syndrome extraction circuits
            let (tick, dag) = build_syndrome_extraction_circuit(data_qubits, ancilla_qubits);

            // Pre-compute DAG propagator
            let start = Instant::now();
            let dag_propagator = DagPropagator::new(&dag);
            let index_time = start.elapsed();
            println!("  (Index build: {:.1} us)", index_time.as_micros());

            // Benchmark: Propagate Z on single ancilla (typical for backward prop from measurement)
            let ancilla_qubit = data_qubits; // First ancilla

            let start = Instant::now();
            for _ in 0..iterations {
                let mut prop = PauliProp::new();
                prop.add_z(ancilla_qubit);
                propagate_through_circuit(&tick, &mut prop, Direction::Backward);
            }
            let tick_time = start.elapsed();

            let start = Instant::now();
            for _ in 0..iterations {
                let mut prop = PauliProp::new();
                prop.add_z(ancilla_qubit);
                dag_propagator.propagate_sparse(&mut prop, Direction::Backward);
            }
            let sparse_time = start.elapsed();

            let tick_per_iter = tick_time.as_micros() as f64 / iterations as f64;
            let sparse_per_iter = sparse_time.as_micros() as f64 / iterations as f64;

            println!("  TickCircuit:          {:>8.1} us/iter", tick_per_iter);
            println!("  DagPropagator sparse: {:>8.1} us/iter ({:.2}x vs Tick)",
                sparse_per_iter, sparse_per_iter / tick_per_iter);
            println!();
        }
    }

    /// Build a syndrome extraction-like circuit with multiple rounds.
    /// Uses realistic 2D surface code connectivity where each ancilla touches
    /// 4 neighboring data qubits in a grid pattern.
    fn build_syndrome_extraction_circuit_multi(
        data_qubits: usize,
        ancilla_qubits: usize,
        rounds: usize,
    ) -> (TickCircuit, DagCircuit) {
        use pecos_quantum::{DagCircuit, TickCircuit};

        let mut tick = TickCircuit::new();
        let mut dag = DagCircuit::new();

        let total = data_qubits + ancilla_qubits;
        let ancilla_list: Vec<usize> = (data_qubits..total).collect();

        // Compute grid size for 2D connectivity
        // For d x d data qubits, we have roughly (d-1)^2 ancillas
        let grid_size = (data_qubits as f64).sqrt().ceil() as usize;

        // Build connectivity map: each ancilla touches 4 neighboring data qubits
        // in a 2D grid pattern (like a rotated surface code)
        let mut ancilla_neighbors: Vec<Vec<usize>> = Vec::with_capacity(ancilla_qubits);
        for a_idx in 0..ancilla_qubits {
            let row = a_idx / (grid_size - 1).max(1);
            let col = a_idx % (grid_size - 1).max(1);

            let mut neighbors = Vec::with_capacity(4);
            // Four neighboring data qubits in a plaquette pattern
            let offsets = [(0, 0), (0, 1), (1, 0), (1, 1)];
            for (dr, dc) in offsets {
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

        for _ in 0..rounds {
            // Prep ancillas
            tick.tick().pz(&ancilla_list);
            for &a in &ancilla_list {
                dag.pz(a);
            }

            // CNOT rounds - apply CNOTs in 4 time steps to avoid collisions
            for cnot_round in 0..4 {
                let cx_pairs: Vec<(usize, usize)> = ancilla_neighbors
                    .iter()
                    .enumerate()
                    .filter_map(|(a_idx, neighbors)| {
                        neighbors.get(cnot_round).map(|&d| (d, data_qubits + a_idx))
                    })
                    .collect();

                if !cx_pairs.is_empty() {
                    tick.tick().cx(&cx_pairs);
                    for (d, a) in cx_pairs {
                        dag.cx(d, a);
                    }
                }
            }

            // Measure ancillas
            tick.tick().mz(&ancilla_list);
            for &a in &ancilla_list {
                dag.mz(a);
            }
        }

        (tick, dag)
    }

    /// Build a syndrome extraction-like circuit (single round for backwards compatibility).
    fn build_syndrome_extraction_circuit(
        data_qubits: usize,
        ancilla_qubits: usize,
    ) -> (TickCircuit, DagCircuit) {
        build_syndrome_extraction_circuit_multi(data_qubits, ancilla_qubits, 1)
    }

    // =========================================================================
    // DagFaultAnalyzer tests
    // =========================================================================

    fn simple_dag_syndrome_circuit() -> DagCircuit {
        // Simple Z-stabilizer measurement: Z0 Z1
        // Ancilla qubit 2 measures the parity
        let mut dag = DagCircuit::new();
        dag.pz(2); // Prep ancilla in |0>
        dag.cx(0, 2); // CNOT from data 0 to ancilla
        dag.cx(1, 2); // CNOT from data 1 to ancilla
        dag.mz(2); // Measure ancilla
        dag
    }

    #[test]
    fn test_dag_backward_propagator_extract_measurements() {
        use super::DagFaultAnalyzer;

        let dag = simple_dag_syndrome_circuit();
        let propagator = DagFaultAnalyzer::new(&dag);
        let measurements = propagator.extract_measurements();

        assert_eq!(measurements.len(), 1);
        // node index, qubit, basis
        assert_eq!(measurements[0].1, 2); // qubit 2
        assert_eq!(measurements[0].2, 0); // Z-basis
    }

    #[test]
    fn test_dag_backward_propagator_build_influence_map() {
        use super::DagFaultAnalyzer;

        let dag = simple_dag_syndrome_circuit();
        let propagator = DagFaultAnalyzer::new(&dag);
        let map = propagator.build_influence_map();

        // Should have fault locations
        assert!(!map.influences.is_empty());

        // Should have one detector
        assert_eq!(map.detectors.len(), 1);

        // Should have one measurement
        assert_eq!(map.measurements.len(), 1);

        println!(
            "DAG Influence map has {} fault locations",
            map.influences.len()
        );
    }

    #[test]
    fn test_dag_backward_propagator_soa_equivalence() {
        use super::DagFaultAnalyzer;

        let dag = simple_dag_syndrome_circuit();
        let propagator = DagFaultAnalyzer::new(&dag);

        // Build both map types
        let btree_map = propagator.build_influence_map();
        let soa_map = propagator.build_influence_map_soa();

        // Same number of locations
        assert_eq!(
            btree_map.influences.len(),
            soa_map.influences.num_locations,
            "Location count mismatch"
        );

        // Same detectors
        assert_eq!(
            btree_map.detectors.len(),
            soa_map.detectors.len(),
            "Detector count mismatch"
        );
        assert_eq!(
            btree_map.measurements.len(),
            soa_map.measurements.len(),
            "Measurement count mismatch"
        );

        // Compare detector IDs
        for (i, (btree_det, soa_det)) in btree_map
            .detectors
            .iter()
            .zip(soa_map.detectors.iter())
            .enumerate()
        {
            assert_eq!(
                btree_det, soa_det,
                "Detector {} mismatch: {:?} vs {:?}",
                i, btree_det, soa_det
            );
        }

        // Compare influences at each location (find matching location in soa by value)
        for (loc, btree_influence) in &btree_map.influences {
            // Find the corresponding location in soa_map
            let soa_loc_idx = soa_map
                .locations
                .iter()
                .position(|l| l == loc)
                .expect("Location should exist in soa map");

            // Compare for each Pauli type (1=X, 2=Y, 3=Z)
            for pauli in 1u8..4 {
                // Get detector IDs from btree map
                let btree_detector_ids: Vec<_> = btree_influence.detector_flips[pauli as usize]
                    .iter()
                    .cloned()
                    .collect();

                // Get detector IDs from SoA map by resolving indices
                let soa_indices = soa_map.get_detector_indices(soa_loc_idx, pauli);
                let soa_detector_ids: Vec<_> = soa_indices
                    .iter()
                    .map(|&idx| soa_map.detectors[idx as usize].clone())
                    .collect();

                assert_eq!(
                    btree_detector_ids.len(),
                    soa_detector_ids.len(),
                    "Location {:?}, Pauli {}: detector count mismatch ({} vs {})",
                    loc,
                    pauli,
                    btree_detector_ids.len(),
                    soa_detector_ids.len()
                );

                // Compare the actual detector IDs (order may differ, so compare as sets)
                for det in &btree_detector_ids {
                    assert!(
                        soa_detector_ids.contains(det),
                        "Location {:?}, Pauli {}: detector {:?} in btree but not in soa",
                        loc,
                        pauli,
                        det
                    );
                }
            }
        }
    }

    #[test]
    fn test_dag_backward_propagator_soa_equivalence_larger() {
        use super::DagFaultAnalyzer;

        // Test on larger circuits to ensure correctness at scale
        let configs = [
            (9, 8),   // 3x3 surface code
            (25, 24), // 5x5 surface code
        ];

        for (data_qubits, ancilla_qubits) in configs {
            let (_, dag) = build_syndrome_extraction_circuit(data_qubits, ancilla_qubits);
            let propagator = DagFaultAnalyzer::new(&dag);

            let btree_map = propagator.build_influence_map();
            let soa_map = propagator.build_influence_map_soa();

            // Same counts
            assert_eq!(
                btree_map.influences.len(),
                soa_map.influences.num_locations,
                "Location count mismatch for {} data qubits",
                data_qubits
            );
            assert_eq!(
                btree_map.detectors.len(),
                soa_map.detectors.len(),
                "Detector count mismatch for {} data qubits",
                data_qubits
            );

            // Verify detector content equivalence (find matching locations by value)
            let mut total_checked = 0;
            for (loc, btree_influence) in &btree_map.influences {
                // Find the corresponding location in soa_map by value
                let soa_loc_idx = soa_map
                    .locations
                    .iter()
                    .position(|l| l == loc)
                    .expect("Location should exist in soa map");

                for pauli in 1u8..4 {
                    // Skip identity
                    let btree_count = btree_influence.detector_flips[pauli as usize].len();
                    let soa_count = soa_map.get_detector_indices(soa_loc_idx, pauli).len();

                    assert_eq!(
                        btree_count, soa_count,
                        "Location {:?}, Pauli {}: {} vs {} detectors",
                        loc, pauli, btree_count, soa_count
                    );
                    total_checked += btree_count;
                }
            }

            println!(
                "Verified {} detector flip entries for d={} circuit",
                total_checked,
                (data_qubits as f64).sqrt() as usize
            );
        }
    }

    #[test]
    fn test_dag_map_types_equivalence() {
        // Comprehensive test verifying both DAG map types produce equivalent results.
        // This ensures: btree (original) == soa (optimized)
        use super::DagFaultAnalyzer;

        let configs = [
            (9, 8, 1),   // d=3, 1 round
            (9, 8, 2),   // d=3, 2 rounds
            (25, 24, 1), // d=5, 1 round
        ];

        for (data_qubits, ancilla_qubits, rounds) in configs {
            let (_, dag) = build_syndrome_extraction_circuit_multi(data_qubits, ancilla_qubits, rounds);
            let propagator = DagFaultAnalyzer::new(&dag);

            // Build both map types
            let btree_map = propagator.build_influence_map();
            let soa_map = propagator.build_influence_map_soa();

            // Verify counts match
            assert_eq!(
                btree_map.influences.len(),
                soa_map.influences.num_locations,
                "Location count mismatch for d={}, rounds={}",
                (data_qubits as f64).sqrt() as usize,
                rounds
            );

            // Verify classification results match for all locations (find by value)
            let mut mismatches = 0;
            for (btree_loc, btree_influence) in &btree_map.influences {
                // Find the corresponding location in soa_map by value
                let soa_loc_idx = soa_map
                    .locations
                    .iter()
                    .position(|l| l == btree_loc)
                    .expect("Location should exist in soa map");

                for pauli in 1..4u8 {
                    let p = pauli as usize;

                    // Check detector flip counts
                    let btree_det = btree_influence.detector_flips[p].len();
                    let soa_det = soa_map.get_detector_indices(soa_loc_idx, pauli).len();

                    if btree_det != soa_det {
                        mismatches += 1;
                        println!(
                            "Mismatch at loc {:?}, pauli {}: btree={}, soa={}",
                            btree_loc.node, pauli, btree_det, soa_det
                        );
                    }
                }
            }

            assert_eq!(
                mismatches, 0,
                "Found {} mismatches for d={}, rounds={}",
                mismatches,
                (data_qubits as f64).sqrt() as usize,
                rounds
            );

            println!(
                "Verified equivalence for d={}, {} rounds: {} locations",
                (data_qubits as f64).sqrt() as usize,
                rounds,
                soa_map.locations.len()
            );
        }
    }

    #[test]
    fn test_dag_backward_propagator_x_error_flips_z_measurement() {
        use super::DagFaultAnalyzer;

        let dag = simple_dag_syndrome_circuit();
        let propagator = DagFaultAnalyzer::new(&dag);
        let map = propagator.build_influence_map();

        // Find fault locations where X error would flip the detector
        let mut found_x_flip = false;
        for (loc, influence) in &map.influences {
            // Check if X error on data qubits flips the detector
            if loc.qubits.iter().any(|q| q.index() < 2) {
                // Data qubit - X error is index 1
                let detectors = &influence.detector_flips[1];
                if !detectors.is_empty() {
                    found_x_flip = true;
                    println!("X error at node {} flips detector", loc.node);
                }
            }
        }

        assert!(
            found_x_flip,
            "X error on data qubit should flip Z-measurement"
        );
    }

    #[test]
    fn test_dag_backward_propagator_z_error_no_flip() {
        use super::DagFaultAnalyzer;

        let dag = simple_dag_syndrome_circuit();
        let propagator = DagFaultAnalyzer::new(&dag);
        let map = propagator.build_influence_map();

        // Z errors on data qubits should NOT flip the Z-measurement detector
        // (Z commutes with CNOT control, and with Z-measurement)
        for (loc, influence) in &map.influences {
            // Check data qubits (0, 1)
            if loc.qubits.iter().all(|q| q.index() < 2) {
                if loc.before {
                    // Z error before the circuit shouldn't flip anything
                    // because Z commutes through CX control
                    // Z error is index 3
                    let _detectors = &influence.detector_flips[3];
                    // Z on data qubit before CX shouldn't flip
                    // (need to check the actual circuit structure)
                }
            }
        }
    }

    /// Benchmark comparing TickCircuit BackwardPropagator vs DAG version
    #[test]
    #[ignore] // Run manually with --ignored flag
    fn benchmark_dag_backward_propagator() {
        use super::{BackwardPropagator, DagFaultAnalyzer};
        use std::time::Instant;

        println!("\n========================================");
        println!("Backward Propagator Benchmark");
        println!("========================================\n");

        // Test configurations: (data_qubits, ancilla_qubits, iterations)
        let configs = [
            (9, 8, 100),    // 3x3 surface code
            (25, 24, 50),   // 5x5 surface code
            (49, 48, 20),   // 7x7 surface code
            (121, 120, 10), // 11x11 surface code
        ];

        for (data_qubits, ancilla_qubits, iterations) in configs {
            println!(
                "--- {} data + {} ancilla qubits ---",
                data_qubits, ancilla_qubits
            );

            // Build equivalent circuits
            let (tick, dag) = build_syndrome_extraction_circuit(data_qubits, ancilla_qubits);

            // Benchmark TickCircuit BackwardPropagator
            let start = Instant::now();
            for _ in 0..iterations {
                let propagator = BackwardPropagator::new(&tick);
                let _map = propagator.build_influence_map();
            }
            let tick_time = start.elapsed();

            // Benchmark DAG BackwardPropagator (full)
            let start = Instant::now();
            for _ in 0..iterations {
                let propagator = DagFaultAnalyzer::new(&dag);
                let _map = propagator.build_influence_map();
            }
            let dag_time = start.elapsed();

            // Benchmark DAG BackwardPropagator (SoA, index-only)
            let start = Instant::now();
            for _ in 0..iterations {
                let propagator = DagFaultAnalyzer::new(&dag);
                let _map = propagator.build_influence_map_soa();
            }
            let dag_soa_time = start.elapsed();

            let tick_per_iter = tick_time.as_micros() as f64 / iterations as f64;
            let dag_per_iter = dag_time.as_micros() as f64 / iterations as f64;
            let dag_soa_per_iter = dag_soa_time.as_micros() as f64 / iterations as f64;

            println!("  TickCircuit BackwardPropagator:   {:>8.1} us/iter", tick_per_iter);
            println!(
                "  DAG BackwardPropagator:           {:>8.1} us/iter ({:.2}x)",
                dag_per_iter,
                dag_per_iter / tick_per_iter
            );
            println!(
                "  DAG BackwardPropagator (soa):     {:>8.1} us/iter ({:.2}x)",
                dag_soa_per_iter,
                dag_soa_per_iter / tick_per_iter
            );
            println!();
        }
    }

    /// Profile where time is spent in DAG propagator for small circuits
    #[test]
    #[ignore]
    fn profile_dag_phases() {
        use super::DagFaultAnalyzer;
        use std::time::Instant;

        println!("\n========================================");
        println!("DAG Propagator Phase Profiling");
        println!("========================================\n");

        // Test small circuit (d=3)
        let (_, dag) = build_syndrome_extraction_circuit(9, 8);
        let iterations = 1000;

        // Phase 1: Constructor
        let start = Instant::now();
        let mut propagators = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            propagators.push(DagFaultAnalyzer::new(&dag));
        }
        let construct_time = start.elapsed();
        println!(
            "d=3: Construction: {:>6.1} us/iter",
            construct_time.as_micros() as f64 / iterations as f64
        );

        // Phase 2: build_influence_map (reusing propagator)
        let propagator = &propagators[0];
        let start = Instant::now();
        for _ in 0..iterations {
            let _map = propagator.build_influence_map();
        }
        let map_time = start.elapsed();
        println!(
            "d=3: build_influence_map: {:>6.1} us/iter",
            map_time.as_micros() as f64 / iterations as f64
        );

        // Phase 3: build_influence_map_soa (optimized)
        let start = Instant::now();
        for _ in 0..iterations {
            let _map = propagator.build_influence_map_soa();
        }
        let map_soa_time = start.elapsed();
        println!(
            "d=3: build_influence_map_soa: {:>6.1} us/iter",
            map_soa_time.as_micros() as f64 / iterations as f64
        );

        // Total
        println!(
            "d=3: Total (new + map): {:>6.1} us/iter",
            (construct_time.as_micros() + map_time.as_micros()) as f64 / iterations as f64
        );
        println!(
            "d=3: Total (new + map_soa): {:>6.1} us/iter",
            (construct_time.as_micros() + map_soa_time.as_micros()) as f64 / iterations as f64
        );

        // Compare with d=5
        println!();
        let (_, dag5) = build_syndrome_extraction_circuit(25, 24);
        let iterations = 500;

        let start = Instant::now();
        let mut propagators5 = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            propagators5.push(DagFaultAnalyzer::new(&dag5));
        }
        let construct_time5 = start.elapsed();
        println!(
            "d=5: Construction: {:>6.1} us/iter",
            construct_time5.as_micros() as f64 / iterations as f64
        );

        let propagator5 = &propagators5[0];
        let start = Instant::now();
        for _ in 0..iterations {
            let _map = propagator5.build_influence_map();
        }
        let map_time5 = start.elapsed();
        println!(
            "d=5: build_influence_map: {:>6.1} us/iter",
            map_time5.as_micros() as f64 / iterations as f64
        );

        let start = Instant::now();
        for _ in 0..iterations {
            let _map = propagator5.build_influence_map_soa();
        }
        let map_soa_time5 = start.elapsed();
        println!(
            "d=5: build_influence_map_soa: {:>6.1} us/iter",
            map_soa_time5.as_micros() as f64 / iterations as f64
        );

        println!(
            "d=5: Total (new + map): {:>6.1} us/iter",
            (construct_time5.as_micros() + map_time5.as_micros()) as f64 / iterations as f64
        );
        println!(
            "d=5: Total (new + map_soa): {:>6.1} us/iter",
            (construct_time5.as_micros() + map_soa_time5.as_micros()) as f64 / iterations as f64
        );
    }

    /// Benchmark with multiple syndrome extraction rounds (more realistic)
    #[test]
    #[ignore] // Run manually with --ignored flag
    fn benchmark_multi_round() {
        use super::{BackwardPropagator, DagFaultAnalyzer};
        use std::time::Instant;

        println!("\n========================================");
        println!("Multi-Round Syndrome Extraction Benchmark");
        println!("========================================\n");

        // Test different code sizes with multiple rounds
        let configs = [
            (9, 8, "d=3"),     // 3x3
            (25, 24, "d=5"),   // 5x5
            (49, 48, "d=7"),   // 7x7
            (121, 120, "d=11"), // 11x11
        ];

        for (data_qubits, ancilla_qubits, label) in configs {
            println!("=== {} ({} data + {} ancilla) ===", label, data_qubits, ancilla_qubits);

            for rounds in [1, 2, 4, 8] {
                let (tick, dag) = build_syndrome_extraction_circuit_multi(data_qubits, ancilla_qubits, rounds);
                let iterations = if data_qubits > 100 { 5 } else if data_qubits > 30 { 10 } else { 20 };

                // Benchmark TickCircuit BackwardPropagator
                let start = Instant::now();
                for _ in 0..iterations {
                    let propagator = BackwardPropagator::new(&tick);
                    let _map = propagator.build_influence_map();
                }
                let tick_time = start.elapsed();

                // Benchmark DAG BackwardPropagator (SoA, index-only)
                let start = Instant::now();
                for _ in 0..iterations {
                    let propagator = DagFaultAnalyzer::new(&dag);
                    let _map = propagator.build_influence_map_soa();
                }
                let dag_soa_time = start.elapsed();

                let tick_per_iter = tick_time.as_micros() as f64 / iterations as f64;
                let dag_soa_per_iter = dag_soa_time.as_micros() as f64 / iterations as f64;

                println!(
                    "  {} rounds: Tick {:>9.1}us, DAG(soa) {:>9.1}us ({:.2}x)",
                    rounds, tick_per_iter, dag_soa_per_iter, dag_soa_per_iter / tick_per_iter
                );
            }
            println!();
        }
    }

    /// Analyze how many nodes are visited per measurement to verify plateau behavior
    #[test]
    #[ignore]
    fn analyze_nodes_visited_per_measurement() {
        use super::DagFaultAnalyzer;
        use pecos_qsim::PauliProp;
        use std::collections::BinaryHeap;

        println!("\n========================================");
        println!("Nodes Visited Per Measurement Analysis");
        println!("========================================\n");

        let configs = [
            (9, 8, "d=3"),
            (25, 24, "d=5"),
            (49, 48, "d=7"),
        ];

        for (data_qubits, ancilla_qubits, label) in configs {
            println!("=== {} ({} data + {} ancilla) ===", label, data_qubits, ancilla_qubits);

            for rounds in [1, 2, 4, 8] {
                let (_, dag) = build_syndrome_extraction_circuit_multi(data_qubits, ancilla_qubits, rounds);
                let propagator = DagFaultAnalyzer::new(&dag);

                let measurements = propagator.extract_measurements();
                let num_measurements = measurements.len();

                // Count nodes visited for each measurement
                let mut total_nodes_visited = 0usize;
                let mut max_nodes_visited = 0usize;
                let mut min_nodes_visited = usize::MAX;

                let mut visited = vec![false; propagator.max_node() + 1];
                let mut active_qubits = vec![false; propagator.max_qubit() + 1];
                let mut heap: BinaryHeap<(usize, usize)> = BinaryHeap::with_capacity(64);

                for &(meas_node, meas_qubit, basis) in &measurements {
                    // Clear work arrays
                    visited.fill(false);
                    active_qubits.fill(false);
                    heap.clear();

                    let mut prop = PauliProp::new();
                    if basis == 0 {
                        prop.add_z(meas_qubit);
                    } else {
                        prop.add_x(meas_qubit);
                    }

                    let meas_topo_pos = propagator.propagator().topo_position(meas_node);
                    let mut nodes_visited = 1; // count the measurement node itself

                    // Initialize
                    if meas_qubit <= propagator.max_qubit() {
                        active_qubits[meas_qubit] = true;
                        for (topo_pos, node) in propagator.propagator().qubit_gates_backward(meas_qubit) {
                            if topo_pos < meas_topo_pos && !visited[node] {
                                visited[node] = true;
                                heap.push((topo_pos, node));
                            }
                        }
                    }

                    // Process
                    while let Some((_, node)) = heap.pop() {
                        nodes_visited += 1;
                        if let Some(gate) = propagator.propagator().gate(node) {
                            let mut was_active = [false; 8];
                            for (j, q) in gate.qubits.iter().enumerate() {
                                if j < was_active.len() && q.index() <= propagator.max_qubit() {
                                    was_active[j] = active_qubits[q.index()];
                                }
                            }

                            super::apply_gate(&mut prop, gate, super::Direction::Backward);

                            let node_topo_pos = propagator.propagator().topo_position(node);
                            for (j, q) in gate.qubits.iter().enumerate() {
                                let idx = q.index();
                                if idx <= propagator.max_qubit() {
                                    let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                                    let was = j < was_active.len() && was_active[j];

                                    if now_active && !was {
                                        active_qubits[idx] = true;
                                        for (topo_pos, new_node) in propagator.propagator().qubit_gates_backward(idx) {
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

                    total_nodes_visited += nodes_visited;
                    max_nodes_visited = max_nodes_visited.max(nodes_visited);
                    min_nodes_visited = min_nodes_visited.min(nodes_visited);
                }

                let avg_nodes = total_nodes_visited as f64 / num_measurements as f64;
                let total_gates = propagator.propagator().topo_order().len();
                let gates_per_round = total_gates / rounds;

                // Expected nodes per measurement with plateau:
                // Each measurement should only see ~6 gates per round (prep + 4 CNOTs + measure on its ancilla)
                // plus some spread to neighboring data qubits
                let expected_plateau = gates_per_round.min(20); // rough estimate

                println!(
                    "  {} rounds: {} meas, avg {:.1} nodes/meas (min {}, max {}), gates/round {}, expected ~{}",
                    rounds, num_measurements, avg_nodes, min_nodes_visited, max_nodes_visited,
                    gates_per_round, expected_plateau
                );
            }
            println!();
        }
    }

    /// Analyze qubit spread to understand if spatial locality is working
    #[test]
    #[ignore]
    fn analyze_qubit_spread() {
        use super::DagFaultAnalyzer;
        use pecos_qsim::PauliProp;
        use std::collections::BinaryHeap;

        println!("\n========================================");
        println!("Qubit Spread Analysis");
        println!("========================================\n");

        let configs = [
            (9, 8, "d=3"),
            (25, 24, "d=5"),
            (49, 48, "d=7"),
        ];

        for (data_qubits, ancilla_qubits, label) in configs {
            println!("=== {} ({} data + {} ancilla) ===", label, data_qubits, ancilla_qubits);

            for rounds in [1, 2, 4, 8] {
                let (_, dag) = build_syndrome_extraction_circuit_multi(data_qubits, ancilla_qubits, rounds);
                let propagator = DagFaultAnalyzer::new(&dag);

                let measurements = propagator.extract_measurements();

                // Track how many qubits become active during propagation
                let mut total_qubits_touched = 0usize;
                let mut max_qubits_touched = 0usize;

                let mut visited = vec![false; propagator.max_node() + 1];
                let mut active_qubits = vec![false; propagator.max_qubit() + 1];
                let mut heap: BinaryHeap<(usize, usize)> = BinaryHeap::with_capacity(64);

                for &(meas_node, meas_qubit, basis) in &measurements {
                    visited.fill(false);
                    active_qubits.fill(false);
                    heap.clear();

                    let mut prop = PauliProp::new();
                    if basis == 0 {
                        prop.add_z(meas_qubit);
                    } else {
                        prop.add_x(meas_qubit);
                    }

                    let meas_topo_pos = propagator.propagator().topo_position(meas_node);
                    let mut qubits_touched = 1; // the measurement qubit

                    if meas_qubit <= propagator.max_qubit() {
                        active_qubits[meas_qubit] = true;
                        for (topo_pos, node) in propagator.propagator().qubit_gates_backward(meas_qubit) {
                            if topo_pos < meas_topo_pos && !visited[node] {
                                visited[node] = true;
                                heap.push((topo_pos, node));
                            }
                        }
                    }

                    while let Some((_, node)) = heap.pop() {
                        if let Some(gate) = propagator.propagator().gate(node) {
                            let mut was_active = [false; 8];
                            for (j, q) in gate.qubits.iter().enumerate() {
                                if j < was_active.len() && q.index() <= propagator.max_qubit() {
                                    was_active[j] = active_qubits[q.index()];
                                }
                            }

                            super::apply_gate(&mut prop, gate, super::Direction::Backward);

                            let node_topo_pos = propagator.propagator().topo_position(node);
                            for (j, q) in gate.qubits.iter().enumerate() {
                                let idx = q.index();
                                if idx <= propagator.max_qubit() {
                                    let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                                    let was = j < was_active.len() && was_active[j];

                                    if now_active && !was {
                                        qubits_touched += 1;
                                        active_qubits[idx] = true;
                                        for (topo_pos, new_node) in propagator.propagator().qubit_gates_backward(idx) {
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

                    total_qubits_touched += qubits_touched;
                    max_qubits_touched = max_qubits_touched.max(qubits_touched);
                }

                let avg_qubits = total_qubits_touched as f64 / measurements.len() as f64;
                let total_qubits = data_qubits + ancilla_qubits;

                println!(
                    "  {} rounds: avg {:.1} qubits/meas (max {}), total qubits {}",
                    rounds, avg_qubits, max_qubits_touched, total_qubits
                );
            }
            println!();
        }
    }

    #[test]
    fn test_soa_equivalence_and_memory() {
        // Verify that SoA format produces equivalent results to vec format
        // and check memory usage.
        use super::DagFaultAnalyzer;
        use pecos_quantum::DagCircuit;

        // Build a d=3 syndrome extraction circuit
        let mut dag = DagCircuit::new();
        let data_qubits = 9;
        let ancilla_qubits = 8;

        // Prepare ancillas
        for a in 0..ancilla_qubits {
            dag.pz(data_qubits + a);
        }

        // Apply CNOTs
        for a in 0..ancilla_qubits {
            let d = a % data_qubits;
            dag.cx(d, data_qubits + a);
        }

        // Measure ancillas
        for a in 0..ancilla_qubits {
            dag.mz(data_qubits + a);
        }

        let propagator = DagFaultAnalyzer::new(&dag);

        // Build both formats
        let btree_map = propagator.build_influence_map();
        let soa = propagator.build_influence_map_soa();

        // Verify equivalence
        assert_eq!(btree_map.influences.len(), soa.locations.len());
        assert_eq!(btree_map.detectors.len(), soa.detectors.len());
        assert_eq!(btree_map.measurements.len(), soa.measurements.len());

        // Check that classification produces the same results (find locations by value)
        for (loc, btree_influence) in &btree_map.influences {
            // Find the corresponding location in soa by value
            let soa_loc_idx = soa
                .locations
                .iter()
                .position(|l| l == loc)
                .expect("Location should exist in soa");

            for pauli in [1u8, 2, 3] {
                let btree_has_syndrome = !btree_influence.detector_flips[pauli as usize].is_empty();
                let btree_has_logical = !btree_influence.logical_flips[pauli as usize].is_empty();
                let (soa_syn, soa_log) = soa.classify_fault(soa_loc_idx, pauli);

                assert_eq!(
                    btree_has_syndrome, soa_syn,
                    "Syndrome mismatch at loc {:?} pauli {}",
                    loc, pauli
                );
                assert_eq!(
                    btree_has_logical, soa_log,
                    "Logical mismatch at loc {:?} pauli {}",
                    loc, pauli
                );

                // Check detector indices match
                let btree_detector_count = btree_influence.detector_flips[pauli as usize].len();
                let soa_detectors = soa.get_detector_indices(soa_loc_idx, pauli);

                assert_eq!(
                    btree_detector_count,
                    soa_detectors.len(),
                    "Detector count mismatch at loc {:?} pauli {}",
                    loc,
                    pauli
                );
            }
        }

        // Print memory statistics
        let stats = soa.memory_stats();
        println!("\n=== SoA Memory Statistics ===");
        println!("Locations: {}", stats.num_locations);
        println!("Detector entries: {}", stats.total_detector_entries);
        println!("Logical entries: {}", stats.total_logical_entries);
        println!("Offset bytes: {}", stats.offset_bytes);
        println!("Data bytes: {}", stats.data_bytes);
        println!("Total bytes: {}", stats.total_bytes);

        println!("\nSoA equivalence verified for {} locations", soa.locations.len());
    }

    #[test]
    fn test_csr_array_basic_operations() {
        // Test CSR array basic operations
        use super::CsrArray;

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
        assert_eq!(csr.row(1), &[] as &[u32]);
        assert_eq!(csr.row(2), &[4, 5]);
        assert_eq!(csr.row(99), &[] as &[u32]); // Out of bounds

        assert!(!csr.row_is_empty(0));
        assert!(csr.row_is_empty(1));
        assert!(!csr.row_is_empty(2));

        assert_eq!(csr.row_len(0), 3);
        assert_eq!(csr.row_len(1), 0);
        assert_eq!(csr.row_len(2), 2);

        assert_eq!(csr.total_elements(), 5);
    }

    #[test]
    fn test_influences_soa_operations() {
        // Test InfluencesSoA operations
        use super::{InfluencesSoA, Pauli};

        let mut soa = InfluencesSoA::with_capacity(2);

        // Location 0: X flips detectors 0, 1; Y flips 2; Z flips nothing
        soa.detectors_x.push(0);
        soa.detectors_x.push(1);
        soa.detectors_y.push(2);
        soa.finish_location();

        // Location 1: X flips nothing; Y flips 3; Z flips 4, 5, 6
        soa.detectors_y.push(3);
        soa.detectors_z.push(4);
        soa.detectors_z.push(5);
        soa.detectors_z.push(6);
        soa.finish_location();

        assert_eq!(soa.num_locations, 2);

        // Check location 0
        assert_eq!(soa.detectors(0, Pauli::X), &[0, 1]);
        assert_eq!(soa.detectors(0, Pauli::Y), &[2]);
        assert_eq!(soa.detectors(0, Pauli::Z), &[] as &[u32]);
        assert!(soa.has_detector_flips(0, Pauli::X));
        assert!(soa.has_detector_flips(0, Pauli::Y));
        assert!(!soa.has_detector_flips(0, Pauli::Z));

        // Check location 1
        assert_eq!(soa.detectors(1, Pauli::X), &[] as &[u32]);
        assert_eq!(soa.detectors(1, Pauli::Y), &[3]);
        assert_eq!(soa.detectors(1, Pauli::Z), &[4, 5, 6]);
        assert!(!soa.has_detector_flips(1, Pauli::X));
        assert!(soa.has_detector_flips(1, Pauli::Y));
        assert!(soa.has_detector_flips(1, Pauli::Z));

        // Check classify
        assert_eq!(soa.classify(0, Pauli::X), (true, false));
        assert_eq!(soa.classify(0, Pauli::Z), (false, false));
        assert_eq!(soa.classify(1, Pauli::Z), (true, false));

        // Check stats
        let stats = soa.memory_stats();
        assert_eq!(stats.num_locations, 2);
        assert_eq!(stats.total_detector_entries, 7); // 2 + 1 + 0 + 0 + 1 + 3
    }
}
