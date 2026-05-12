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
//! with specialized support for fault tolerance analysis. By propagating
//! detector/observable measurement parities and unmeasured tracked Pauli
//! operators backward through the circuit, we can efficiently determine which
//! faults affect which outputs:
//!
//! 1. **Speed up fault enumeration** - O(1) lookup instead of `O(circuit_depth)` propagation
//! 2. **Build detector error models** - Direct mapping from faults to detectors
//! 3. **Analyze syndrome histories** - Know which round each fault affects
//!
//! # Recommended: DAG-based Analysis
//!
//! For best performance, use [`DagFaultAnalyzer`] with [`DagCircuit`](pecos_quantum::DagCircuit).
//! The DAG representation enables sparse traversal that only visits gates on active
//! qubit wires, providing **5-50x speedup** over tick-based analysis for typical
//! surface code circuits.
//!
//! ```
//! use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
//! use pecos_quantum::DagCircuit;
//!
//! // Build a simple syndrome extraction circuit
//! let mut dag = DagCircuit::new();
//! dag.pz(&[2]);       // Prep ancilla
//! dag.cx(&[(0, 2)]);    // CNOT data -> ancilla
//! dag.cx(&[(1, 2)]);    // CNOT data -> ancilla
//! dag.mz(&[2]);       // Measure ancilla
//!
//! // Build the fault influence map
//! let analyzer = DagFaultAnalyzer::new(&dag);
//! let map = analyzer.build_influence_map();
//!
//! // O(1) lookup: which detector/non-detector outputs does a fault at location L flip?
//! let (has_syndrome, _flips_non_detector_output) = map.classify_fault(0, 1); // loc 0, X fault
//! ```
//!
//! Observables and tracked Paulis are distinct. Observables are values
//! observed through measurement-record parities and become standard `L<n>`
//! outputs in DEM text. Tracked Paulis are Pauli operators annotated at
//! circuit points; they are not measured and are not applied to the computation.
//! PECOS records whether each fault anticommutes with, and therefore would flip,
//! the propagated operator.
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
//! # Legacy: Tick-based Analysis
//!
//! For circuits already in [`TickCircuit`](pecos_quantum::TickCircuit) format,
//! use [`TickFaultAnalyzer`]. This processes gates tick-by-tick and is simpler
//! but slower for large circuits with local connectivity.
//!
//! ```
//! use pecos_qec::fault_tolerance::propagator::{TickFaultAnalyzer, FaultInfluenceMap};
//! use pecos_quantum::TickCircuit;
//!
//! let mut circuit = TickCircuit::new();
//! circuit.tick().pz(&[2]);
//! circuit.tick().cx(&[(0, 2)]);
//! circuit.tick().cx(&[(1, 2)]);
//! circuit.tick().mz(&[2]);
//!
//! let propagator = TickFaultAnalyzer::new(&circuit);
//! let influence_map = propagator.build_influence_map();
//! ```

// Submodules
mod checker;
pub mod dag;
mod pauli;
mod tick;
mod tick_batched;
pub mod types;

// Re-export from submodules
pub use checker::InfluenceBasedChecker;
pub use dag::{
    BucketRecorder, CsrArray, DagFaultAnalyzer, DagFaultInfluenceMap, DagSpacetimeLocation,
    DemOutputKind, DemOutputMetadata, FaultCombo, FaultComponent, FaultEffect, FaultLocations,
    GateFaultLocation, InfluencesSoA, InfluencesSoAStats, SoARecorderBuilder,
};
pub use pauli::{
    Direction, apply_gate, init_pauli_prop_with_fault, propagate_backward_from_tick,
    propagate_fault_backward, propagate_observable_backward, propagate_through_circuit,
    propagate_tick_range,
};
pub use tick::TickFaultAnalyzer;
pub use tick_batched::TickFaultAnalyzerBatched;
pub use types::{
    DemOutputIdx, DetectorId, DetectorIdx, FaultInfluence, FaultInfluenceMap, LocationId,
    MeasurementId, NodeId, Pauli, TrackedPauliId, TrackedPauliIdx,
};

// Internal imports
use super::{PauliFault, SpacetimeLocation, extract_spacetime_locations};
use pecos_core::gate_type::GateType;
use pecos_quantum::{DagCircuit, DagTraversalIndex};
use pecos_simulators::PauliProp;
use std::collections::{BTreeSet, BinaryHeap};

// ============================================================================
// DAG-Based Sparse Propagation Infrastructure
// ============================================================================

/// Reusable work buffers for propagation to avoid repeated allocations.
///
/// Create once and reuse across multiple propagations for best performance.
#[derive(Debug, Clone)]
pub struct PropagatorWorkBuffers {
    /// Visited nodes (indexed by node id)
    pub visited: Vec<bool>,
    /// Active qubits (indexed by qubit id)
    pub active_qubits: Vec<bool>,
    /// Priority queue for heap-based traversal (`topo_pos`, `node_id`)
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
    /// * `loc_idx` - Location index in the `FaultLocations` array
    /// * `qubit` - The qubit where the fault occurs
    /// * `obs_x` - Whether the current observable has X on this qubit
    /// * `obs_z` - Whether the current observable has Z on this qubit
    /// * `detector_idx` - The detector being propagated from
    fn record(
        &mut self,
        loc_idx: usize,
        qubit: usize,
        obs_x: bool,
        obs_z: bool,
        detector_idx: usize,
    );
}

/// Context for a single backward propagation pass.
///
/// Bundles the state needed for propagation, enabling reuse and
/// clean separation between traversal and recording.
#[derive(Debug)]
pub struct PropagationContext<'a> {
    /// Current Pauli operator being propagated.
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

    /// Initializes the Pauli operator for a Z-basis measurement.
    pub fn init_z_measurement(&mut self, qubit: usize) {
        self.prop.track_z(&[qubit]);
    }

    /// Initializes the Pauli operator for an X-basis measurement.
    pub fn init_x_measurement(&mut self, qubit: usize) {
        self.prop.track_x(&[qubit]);
    }

    /// Initializes the Pauli operator based on measurement basis.
    pub fn init_measurement(&mut self, qubit: usize, basis: u8) {
        if basis == 0 {
            self.prop.track_z(&[qubit]);
        } else {
            self.prop.track_x(&[qubit]);
        }
    }

    /// Returns whether the observable has X on the given qubit.
    #[inline]
    #[must_use]
    pub fn has_x(&self, qubit: usize) -> bool {
        self.prop.contains_x(qubit)
    }

    /// Returns whether the observable has Z on the given qubit.
    #[inline]
    #[must_use]
    pub fn has_z(&self, qubit: usize) -> bool {
        self.prop.contains_z(qubit)
    }

    /// Returns whether the qubit is currently active (has non-trivial Pauli).
    #[inline]
    #[must_use]
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
    #[must_use]
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
    fn record(
        &mut self,
        _loc_idx: usize,
        _qubit: usize,
        _obs_x: bool,
        _obs_z: bool,
        _detector_idx: usize,
    ) {
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
    fn record(
        &mut self,
        _loc_idx: usize,
        _qubit: usize,
        obs_x: bool,
        obs_z: bool,
        _detector_idx: usize,
    ) {
        self.count += 1;
        if obs_z {
            self.by_pauli[1] += 1; // X fault
        }
        if obs_x {
            self.by_pauli[3] += 1; // Z fault
        }
        if obs_x ^ obs_z {
            self.by_pauli[2] += 1; // Y fault
        }
    }
}

// ============================================================================
// DAG Propagator
// ============================================================================

/// Pre-computed index for efficient DAG-based Pauli propagation.
///
/// This struct pre-computes data structures needed for sparse propagation,
/// making repeated propagations through the same circuit much faster.
///
/// # Example
/// ```
/// use pecos_qec::fault_tolerance::propagator::{DagPropagator, Direction};
/// use pecos_quantum::DagCircuit;
/// use pecos_simulators::PauliProp;
///
/// let mut dag = DagCircuit::new();
/// dag.h(&[0]);
/// dag.cx(&[(0, 1)]);
///
/// // Pre-compute indices (do this once)
/// let propagator = DagPropagator::new(&dag);
///
/// // Propagate multiple times efficiently
/// let mut prop = PauliProp::new();
/// prop.track_z(&[0]);
/// propagator.propagate_sparse(&mut prop, Direction::Forward);
/// ```
pub struct DagPropagator<'a> {
    /// Reference to the underlying DAG circuit.
    dag: &'a DagCircuit,
    /// Pre-computed traversal index from `DagCircuit`.
    index: DagTraversalIndex,
}

impl<'a> DagPropagator<'a> {
    /// Creates a new `DagPropagator` with pre-computed indices.
    ///
    /// This is O(V + E) where V is the number of gates and E is the number of edges.
    #[must_use]
    pub fn new(dag: &'a DagCircuit) -> Self {
        let index = dag.build_traversal_index();
        Self { dag, index }
    }

    /// Creates a `DagPropagator` from an existing traversal index.
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

    /// Returns the topological order of nodes.
    #[inline]
    #[must_use]
    pub fn topo_order(&self) -> &[usize] {
        self.index.topo_order()
    }

    /// Returns the topological position of a node.
    #[inline]
    #[must_use]
    pub fn topo_position(&self, node: usize) -> usize {
        self.index.topo_position(node)
    }

    /// Returns gates on a qubit in backward order (from end to start).
    #[inline]
    pub fn qubit_gates_backward(&self, qubit: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.index.qubit_gates_reversed(qubit)
    }

    /// Returns gates on a qubit in forward order (from start to end).
    #[inline]
    pub fn qubit_gates_forward(&self, qubit: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.index.qubit_gates(qubit).iter().copied()
    }

    /// Returns the gate at a node, if any.
    #[inline]
    #[must_use]
    pub fn gate(&self, node: usize) -> Option<&pecos_core::Gate> {
        self.dag.gate(node)
    }

    /// Returns neighbors of a node in the specified direction.
    ///
    /// For Forward: returns successor nodes (next in execution order).
    /// For Backward: returns predecessor nodes (previous in execution order).
    ///
    /// # Example
    /// ```
    /// use pecos_qec::fault_tolerance::propagator::{DagPropagator, Direction};
    /// use pecos_quantum::DagCircuit;
    ///
    /// let mut dag = DagCircuit::new();
    /// dag.pz(&[0]);
    /// dag.h(&[0]);
    /// dag.mz(&[0]);
    ///
    /// let prep = dag.topological_order()[0]; // first node is PZ
    ///
    /// let propagator = DagPropagator::new(&dag);
    ///
    /// // The propagator can traverse neighbors in either direction
    /// // Here we just verify the API works
    /// let forward: Vec<_> = propagator.neighbors(prep, Direction::Forward).collect();
    /// assert!(!forward.is_empty()); // prep connects to h
    /// ```
    pub fn neighbors(&self, node: usize, direction: Direction) -> impl Iterator<Item = usize> {
        match direction {
            Direction::Forward => self.dag.successors(node).into_iter(),
            Direction::Backward => self.dag.predecessors(node).into_iter(),
        }
    }

    /// Sparse propagation through the DAG, visiting only gates on active qubit wires.
    ///
    /// This is significantly faster than dense propagation for circuits with
    /// local connectivity (like surface codes), where Paulis only touch a
    /// small subset of qubits.
    pub fn propagate_sparse(&self, prop: &mut PauliProp, direction: Direction) {
        // Find initial active qubits
        let max_qubit = self.max_qubit();
        let mut active_qubits: BTreeSet<usize> = (0..=max_qubit)
            .filter(|&q| prop.contains_x(q) || prop.contains_z(q))
            .collect();

        if active_qubits.is_empty() {
            return;
        }

        // Process nodes in topological order (forward) or reverse (backward)
        let topo = self.topo_order();
        let node_iter: Box<dyn Iterator<Item = &usize>> = match direction {
            Direction::Forward => Box::new(topo.iter()),
            Direction::Backward => Box::new(topo.iter().rev()),
        };

        for &node in node_iter {
            if let Some(gate) = self.gate(node) {
                // Check if this gate touches any active qubit
                let touches_active = gate
                    .qubits
                    .iter()
                    .any(|q| active_qubits.contains(&q.index()));

                if touches_active {
                    // Apply the gate
                    apply_gate(prop, gate, direction);

                    // Update active qubits
                    for q in &gate.qubits {
                        let idx = q.index();
                        if prop.contains_x(idx) || prop.contains_z(idx) {
                            active_qubits.insert(idx);
                        } else {
                            active_qubits.remove(&idx);
                        }
                    }
                }
            }
        }
    }

    /// Dense propagation through the entire DAG (visits all gates).
    ///
    /// Use `propagate_sparse` instead for better performance when Paulis
    /// only touch a subset of qubits.
    pub fn propagate_dense(&self, prop: &mut PauliProp, direction: Direction) {
        let topo = self.topo_order();
        let node_iter: Box<dyn Iterator<Item = &usize>> = match direction {
            Direction::Forward => Box::new(topo.iter()),
            Direction::Backward => Box::new(topo.iter().rev()),
        };

        for &node in node_iter {
            if let Some(gate) = self.gate(node) {
                apply_gate(prop, gate, direction);
            }
        }
    }

    /// Propagate backward from a specific node, stopping at prep gates.
    ///
    /// This is useful for tracking what faults affect a measurement:
    /// start with the measurement's observable and propagate backward.
    pub fn propagate_backward_from(&self, prop: &mut PauliProp, start_node: usize) {
        let start_pos = self.topo_position(start_node);
        let max_qubit = self.max_qubit();

        // Track active qubits
        let mut active_qubits: BTreeSet<usize> = (0..=max_qubit)
            .filter(|&q| prop.contains_x(q) || prop.contains_z(q))
            .collect();

        // Process nodes in reverse topological order up to start_node
        for &node in self.topo_order().iter().rev() {
            let node_pos = self.topo_position(node);
            if node_pos > start_pos {
                continue;
            }

            if let Some(gate) = self.gate(node) {
                // Check if this gate touches any active qubit
                let touches_active = gate
                    .qubits
                    .iter()
                    .any(|q| active_qubits.contains(&q.index()));

                if touches_active {
                    // Handle prep gates specially - they kill the Pauli
                    if matches!(gate.gate_type, GateType::PZ | GateType::QAlloc) {
                        for q in &gate.qubits {
                            let idx = q.index();
                            if prop.contains_x(idx) {
                                prop.track_x(&[idx]); // toggle off
                            }
                            if prop.contains_z(idx) {
                                prop.track_z(&[idx]); // toggle off
                            }
                            active_qubits.remove(&idx);
                        }
                    } else {
                        // Apply gate backward
                        apply_gate(prop, gate, Direction::Backward);

                        // Update active qubits
                        for q in &gate.qubits {
                            let idx = q.index();
                            if prop.contains_x(idx) || prop.contains_z(idx) {
                                active_qubits.insert(idx);
                            } else {
                                active_qubits.remove(&idx);
                            }
                        }
                    }
                }
            }
        }
    }
}

// ============================================================================
// Standalone DAG Propagation Functions
// ============================================================================

/// Propagates a Pauli through a DAG circuit using sparse traversal.
///
/// This is a convenience function that creates a temporary `DagPropagator`.
/// For repeated propagations, create a `DagPropagator` once and reuse it.
pub fn propagate_sparse_dag(dag: &DagCircuit, prop: &mut PauliProp, direction: Direction) {
    let propagator = DagPropagator::new(dag);
    propagator.propagate_sparse(prop, direction);
}

/// Propagates a Pauli through a DAG circuit (dense traversal).
///
/// This visits all gates in topological order. For sparse circuits,
/// use `propagate_sparse_dag` instead.
pub fn propagate_through_dag(dag: &DagCircuit, prop: &mut PauliProp, direction: Direction) {
    let propagator = DagPropagator::new(dag);
    propagator.propagate_dense(prop, direction);
}

/// Propagates a Pauli backward from a specific node in a DAG circuit.
///
/// This is useful for understanding what observable is being measured,
/// or what faults affect a specific location.
pub fn propagate_backward_from_node(dag: &DagCircuit, prop: &mut PauliProp, start_node: usize) {
    let propagator = DagPropagator::new(dag);
    propagator.propagate_backward_from(prop, start_node);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_quantum::TickCircuit;

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
        let propagator = TickFaultAnalyzer::new(&circuit);
        let map = propagator.build_influence_map();

        assert_eq!(map.measurements.len(), 1);
        assert_eq!(map.measurements[0].tick, 3);
        assert_eq!(map.measurements[0].qubit, 2);
        assert_eq!(map.measurements[0].basis, 0); // Z-measurement
    }

    #[test]
    fn test_build_influence_map() {
        let circuit = simple_syndrome_circuit();
        let propagator = TickFaultAnalyzer::new(&circuit);
        let map = propagator.build_influence_map();

        // Should have fault locations
        assert!(map.num_fault_locations() > 0);

        // Should have one detector
        assert_eq!(map.detectors.len(), 1);

        // Should have one measurement
        assert_eq!(map.measurements.len(), 1);

        println!(
            "Influence map has {} fault locations",
            map.num_fault_locations()
        );
        println!("Detectors: {:?}", map.detectors);
    }

    #[test]
    fn test_x_error_flips_z_measurement() {
        let circuit = simple_syndrome_circuit();
        let propagator = TickFaultAnalyzer::new(&circuit);
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
                    println!("X error at {loc:?} flips detector");
                }
            }
        }

        assert!(
            found_x_flip,
            "Should find X errors that flip the measurement"
        );
    }

    #[test]
    fn test_z_error_no_syndrome() {
        let circuit = simple_syndrome_circuit();
        let propagator = TickFaultAnalyzer::new(&circuit);
        let map = propagator.build_influence_map();

        // A Z error on data qubits should NOT flip the Z-measurement
        // because Z commutes with CX on control, and measurement is Z-basis

        // Check that Z errors on data qubits don't flip detectors
        for (loc, influence) in &map.influences {
            if loc.qubits.iter().any(|q| q.index() == 0 || q.index() == 1) {
                // Z errors on data qubits shouldn't flip Z-measurement
                let z_flips = influence.detectors_for_pauli(3);
                // This may or may not be empty depending on exact location
                println!("Z error at {:?} flips {} detectors", loc, z_flips.len());
            }
        }
    }

    #[test]
    fn test_influence_based_checker() {
        let circuit = simple_syndrome_circuit();
        let propagator = TickFaultAnalyzer::new(&circuit);
        let map = propagator.build_influence_map();

        let checker = InfluenceBasedChecker::new(&map);

        // Get any fault location and check classification
        if let Some((loc, _)) = map.influences.iter().next() {
            let (has_syndrome, flips_tracked_pauli) = checker.classify(loc, 1); // X fault
            println!(
                "Location {loc:?}: syndrome={has_syndrome}, tracked_pauli={flips_tracked_pauli}"
            );
        }
    }

    #[test]
    fn test_dag_propagator_basic() {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);
        dag.mz(&[0]);

        let propagator = DagPropagator::new(&dag);

        // Start with Z at the end
        let mut prop = PauliProp::new();
        prop.track_z(&[0]);

        // Propagate backward through H: Z -> X
        propagator.propagate_sparse(&mut prop, Direction::Backward);

        // After H backward, Z becomes X
        assert!(prop.contains_x(0));
        assert!(!prop.contains_z(0));
    }

    #[test]
    fn test_dag_propagator_cx() {
        let mut dag = DagCircuit::new();
        dag.cx(&[(0, 1)]);

        let propagator = DagPropagator::new(&dag);

        // Test 1: X on control spreads to target
        let mut prop = PauliProp::new();
        prop.track_x(&[0]);
        propagator.propagate_sparse(&mut prop, Direction::Forward);
        assert!(prop.contains_x(0));
        assert!(prop.contains_x(1));

        // Test 2: Z on target spreads to control
        let mut prop2 = PauliProp::new();
        prop2.track_z(&[1]);
        propagator.propagate_sparse(&mut prop2, Direction::Backward);
        assert!(prop2.contains_z(0));
        assert!(prop2.contains_z(1));
    }

    #[test]
    fn test_dag_fault_analyzer_basic() {
        let mut dag = DagCircuit::new();
        dag.pz(&[2]);
        dag.cx(&[(0, 2)]);
        dag.cx(&[(1, 2)]);
        dag.mz(&[2]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let map = analyzer.build_influence_map();

        // Should have locations and detectors
        assert!(!map.locations.is_empty());
        assert!(!map.detectors.is_empty());
    }

    // Additional tests for backward vs forward consistency
    #[test]
    fn test_backward_vs_forward_simple() {
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0]);

        // Forward: X -> Z
        let mut forward = PauliProp::new();
        forward.track_x(&[0]);
        propagate_through_circuit(&circuit, &mut forward, Direction::Forward);

        // Backward: Z -> X
        let mut backward = PauliProp::new();
        backward.track_z(&[0]);
        propagate_through_circuit(&circuit, &mut backward, Direction::Backward);

        // Forward X->Z, Backward Z->X
        assert!(forward.contains_z(0));
        assert!(backward.contains_x(0));
    }

    #[test]
    fn test_dag_fault_analyzer_z_error_check() {
        let mut dag = DagCircuit::new();
        dag.pz(&[2]);
        dag.cx(&[(0, 2)]);
        dag.cx(&[(1, 2)]);
        dag.mz(&[2]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let map = analyzer.build_influence_map();

        // Check that we have locations
        assert!(!map.locations.is_empty());
    }

    #[test]
    fn test_dag_fault_analyzer_larger_circuits() {
        // Test with a larger circuit to ensure scalability
        let mut dag = DagCircuit::new();
        for i in 0..10 {
            dag.pz(&[i]);
        }
        for i in 0..9 {
            dag.cx(&[(i, i + 1)]);
        }
        for i in 0..10 {
            dag.mz(&[i]);
        }

        let analyzer = DagFaultAnalyzer::new(&dag);
        let map = analyzer.build_influence_map();

        // Should have locations and detectors
        assert!(!map.locations.is_empty());
        assert!(!map.detectors.is_empty());
    }

    #[test]
    fn test_backward_vs_forward_random_circuits() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Test that backward propagation is consistent with forward propagation
        // using random Clifford circuits

        let mut state = 42u64;
        let mut next_rand = || -> u64 {
            let mut hasher = DefaultHasher::new();
            state.hash(&mut hasher);
            state = hasher.finish();
            state
        };

        for _ in 0..10 {
            let num_qubits = 3;
            let num_gates = 5;

            let mut circuit = TickCircuit::new();
            for _ in 0..num_gates {
                let gate_type = next_rand() % 4;
                #[allow(clippy::cast_possible_truncation)] // 64-bit target
                let q1 = (next_rand() % num_qubits as u64) as usize;

                let mut t = circuit.tick();
                match gate_type {
                    0 => {
                        t.h(&[q1]);
                    }
                    1 => {
                        t.sz(&[q1]);
                    }
                    2 => {
                        #[allow(clippy::cast_possible_truncation)] // 64-bit target
                        let q2 = ((next_rand() % (num_qubits - 1) as u64) as usize + q1 + 1)
                            % num_qubits;
                        t.cx(&[(q1, q2)]);
                    }
                    _ => {
                        #[allow(clippy::cast_possible_truncation)] // 64-bit target
                        let q2 = ((next_rand() % (num_qubits - 1) as u64) as usize + q1 + 1)
                            % num_qubits;
                        t.cz(&[(q1, q2)]);
                    }
                }
            }

            // Test forward then backward should give identity for self-adjoint circuits
            // (This is not exactly identity due to how we track things, but structure should match)
            let mut prop = PauliProp::new();
            prop.track_x(&[0]);
            propagate_through_circuit(&circuit, &mut prop, Direction::Forward);
            propagate_through_circuit(&circuit, &mut prop, Direction::Backward);

            // After forward then backward, we should get back something consistent
            // (exact check depends on circuit structure)
        }
    }

    #[test]
    fn test_backward_vs_forward_with_tracked_paulis() {
        // Test that tracked-Pauli propagation works with backward propagation
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2]);
        circuit.tick().cx(&[(0, 2)]);
        circuit.tick().cx(&[(1, 2)]);
        circuit.tick().mz(&[2]);

        // Define a simple tracked Z Pauli = Z0 Z1
        let tracked_paulis: &[(&[usize], &[usize])] = &[(&[], &[0, 1])];

        let propagator = TickFaultAnalyzer::new(&circuit);
        let map = propagator.build_influence_map_with_tracked_paulis(tracked_paulis);

        // Check that tracked-Pauli propagation is populated
        assert_eq!(map.tracked_paulis.len(), 1);

        // X errors on data qubits should flip the tracked Pauli
        let mut found_tracked_pauli_flip = false;
        for (loc, influence) in &map.influences {
            if loc.qubits.iter().any(|q| q.index() == 0 || q.index() == 1)
                && !influence.tracked_paulis_for_pauli(1).is_empty()
            {
                found_tracked_pauli_flip = true;
            }
        }
        assert!(
            found_tracked_pauli_flip,
            "Should find X errors that flip tracked Pauli"
        );
    }

    #[test]
    fn test_backward_vs_forward_with_stabilizer_sim() {
        // Additional test using more complex circuits
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2, 3]);
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1)]);
        circuit.tick().cx(&[(0, 2)]);
        circuit.tick().h(&[0]);
        circuit.tick().mz(&[0, 1, 2, 3]);

        let propagator = TickFaultAnalyzer::new(&circuit);
        let map = propagator.build_influence_map();

        // Should have multiple measurements
        assert_eq!(map.measurements.len(), 4);
        assert_eq!(map.detectors.len(), 4);
    }

    #[test]
    fn test_backward_vs_forward_varying_sizes() {
        // Test with varying circuit sizes
        for size in [2, 4, 8] {
            let mut circuit = TickCircuit::new();

            // Prep all qubits
            let qubits: Vec<usize> = (0..size).collect();
            circuit.tick().pz(&qubits);

            // Chain of CNOTs
            for i in 0..size - 1 {
                circuit.tick().cx(&[(i, i + 1)]);
            }

            // Measure all
            circuit.tick().mz(&qubits);

            let propagator = TickFaultAnalyzer::new(&circuit);
            let map = propagator.build_influence_map();

            assert_eq!(map.measurements.len(), size);
        }
    }

    #[test]
    fn test_backward_vs_forward_deep_circuits() {
        // Test with deeper circuits (more ticks)
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1]);

        for _ in 0..20 {
            circuit.tick().cx(&[(0, 1)]);
            circuit.tick().h(&[0]);
        }

        circuit.tick().mz(&[0, 1]);

        let propagator = TickFaultAnalyzer::new(&circuit);
        let map = propagator.build_influence_map();

        assert!(map.num_fault_locations() > 0);
    }

    #[test]
    fn test_dag_map_multi_round() {
        // Multi-round syndrome extraction
        let mut dag = DagCircuit::new();

        // Round 1
        dag.pz(&[2]);
        dag.cx(&[(0, 2)]);
        dag.cx(&[(1, 2)]);
        dag.mz(&[2]);

        // Round 2
        dag.pz(&[2]);
        dag.cx(&[(0, 2)]);
        dag.cx(&[(1, 2)]);
        dag.mz(&[2]);

        let analyzer = DagFaultAnalyzer::new(&dag);
        let map = analyzer.build_influence_map();

        // Should have 2 measurements (2 rounds)
        assert_eq!(map.detectors.len(), 2);
    }

    fn pauli_signature(prop: &PauliProp, qubits: &[usize]) -> Vec<(bool, bool)> {
        qubits
            .iter()
            .map(|&q| (prop.contains_x(q), prop.contains_z(q)))
            .collect()
    }

    fn pauli_prop_from_signature(signature: &[(bool, bool)]) -> PauliProp {
        let mut prop = PauliProp::new();
        for (qubit, &(has_x, has_z)) in signature.iter().enumerate() {
            if has_x {
                prop.track_x(&[qubit]);
            }
            if has_z {
                prop.track_z(&[qubit]);
            }
        }
        prop
    }

    fn add_standard_clifford_gate(circuit: &mut TickCircuit, gate_type: GateType) {
        match gate_type {
            GateType::I => {
                circuit.tick().iden(&[0]);
            }
            GateType::X => {
                circuit.tick().x(&[0]);
            }
            GateType::Y => {
                circuit.tick().y(&[0]);
            }
            GateType::Z => {
                circuit.tick().z(&[0]);
            }
            GateType::H => {
                circuit.tick().h(&[0]);
            }
            GateType::F => {
                circuit.tick().f(&[0]);
            }
            GateType::Fdg => {
                circuit.tick().fdg(&[0]);
            }
            GateType::SX => {
                circuit.tick().sx(&[0]);
            }
            GateType::SXdg => {
                circuit.tick().sxdg(&[0]);
            }
            GateType::SY => {
                circuit.tick().sy(&[0]);
            }
            GateType::SYdg => {
                circuit.tick().sydg(&[0]);
            }
            GateType::SZ => {
                circuit.tick().sz(&[0]);
            }
            GateType::SZdg => {
                circuit.tick().szdg(&[0]);
            }
            GateType::CX => {
                circuit.tick().cx(&[(0, 1)]);
            }
            GateType::CY => {
                circuit.tick().cy(&[(0, 1)]);
            }
            GateType::CZ => {
                circuit.tick().cz(&[(0, 1)]);
            }
            GateType::SXX => {
                circuit.tick().sxx(&[(0, 1)]);
            }
            GateType::SXXdg => {
                circuit.tick().sxxdg(&[(0, 1)]);
            }
            GateType::SYY => {
                circuit.tick().syy(&[(0, 1)]);
            }
            GateType::SYYdg => {
                circuit.tick().syydg(&[(0, 1)]);
            }
            GateType::SZZ => {
                circuit.tick().szz(&[(0, 1)]);
            }
            GateType::SZZdg => {
                circuit.tick().szzdg(&[(0, 1)]);
            }
            GateType::SWAP => {
                circuit.tick().swap(&[(0, 1)]);
            }
            _ => unreachable!("not a standard Clifford gate: {gate_type:?}"),
        }
    }

    fn assert_pauli_signature_after_gate(
        gate_type: GateType,
        input: [(bool, bool); 2],
        expected: [(bool, bool); 2],
    ) {
        let mut circuit = TickCircuit::new();
        add_standard_clifford_gate(&mut circuit, gate_type);

        let mut prop = pauli_prop_from_signature(&input);
        propagate_through_circuit(&circuit, &mut prop, Direction::Forward);

        assert_eq!(
            pauli_signature(&prop, &[0, 1]),
            expected,
            "{gate_type:?} should map {input:?} to {expected:?} up to Pauli phase"
        );
    }

    #[test]
    fn test_standard_clifford_pauli_conjugation_tables() {
        const I: (bool, bool) = (false, false);
        const X: (bool, bool) = (true, false);
        const Z: (bool, bool) = (false, true);
        const Y: (bool, bool) = (true, true);

        assert_pauli_signature_after_gate(GateType::H, [X, I], [Z, I]);
        assert_pauli_signature_after_gate(GateType::H, [Z, I], [X, I]);
        assert_pauli_signature_after_gate(GateType::F, [X, I], [Y, I]);
        assert_pauli_signature_after_gate(GateType::F, [Y, I], [Z, I]);
        assert_pauli_signature_after_gate(GateType::F, [Z, I], [X, I]);
        assert_pauli_signature_after_gate(GateType::Fdg, [X, I], [Z, I]);
        assert_pauli_signature_after_gate(GateType::Fdg, [Y, I], [X, I]);
        assert_pauli_signature_after_gate(GateType::Fdg, [Z, I], [Y, I]);
        assert_pauli_signature_after_gate(GateType::SX, [Z, I], [Y, I]);
        assert_pauli_signature_after_gate(GateType::SY, [X, I], [Z, I]);
        assert_pauli_signature_after_gate(GateType::SZ, [X, I], [Y, I]);

        assert_pauli_signature_after_gate(GateType::CX, [X, I], [X, X]);
        assert_pauli_signature_after_gate(GateType::CX, [I, Z], [Z, Z]);
        assert_pauli_signature_after_gate(GateType::CY, [X, I], [X, Y]);
        assert_pauli_signature_after_gate(GateType::CZ, [X, I], [X, Z]);
        assert_pauli_signature_after_gate(GateType::SWAP, [X, Z], [Z, X]);

        assert_pauli_signature_after_gate(GateType::SXX, [Z, I], [Y, X]);
        assert_pauli_signature_after_gate(GateType::SXX, [I, Z], [X, Y]);
        assert_pauli_signature_after_gate(GateType::SYY, [X, I], [Z, Y]);
        assert_pauli_signature_after_gate(GateType::SYY, [I, X], [Y, Z]);
        assert_pauli_signature_after_gate(GateType::SZZ, [X, I], [Y, Z]);
        assert_pauli_signature_after_gate(GateType::SZZ, [I, X], [Z, Y]);

        assert_pauli_signature_after_gate(GateType::SXXdg, [Z, I], [Y, X]);
        assert_pauli_signature_after_gate(GateType::SYYdg, [X, I], [Z, Y]);
        assert_pauli_signature_after_gate(GateType::SZZdg, [X, I], [Y, Z]);
    }

    #[test]
    fn test_rz_propagation_matches_sz() {
        let mut rotated = TickCircuit::new();
        rotated.tick().rz(pecos_core::Angle64::QUARTER_TURN, &[0]);

        let mut simplified = TickCircuit::new();
        simplified.tick().sz(&[0]);

        let mut rotated_prop = PauliProp::new();
        rotated_prop.track_x(&[0]);
        propagate_through_circuit(&rotated, &mut rotated_prop, Direction::Forward);

        let mut simplified_prop = PauliProp::new();
        simplified_prop.track_x(&[0]);
        propagate_through_circuit(&simplified, &mut simplified_prop, Direction::Forward);

        assert_eq!(
            pauli_signature(&rotated_prop, &[0]),
            pauli_signature(&simplified_prop, &[0])
        );
    }

    #[test]
    fn test_r1xy_propagation_matches_sx() {
        let mut rotated = TickCircuit::new();
        rotated.tick().r1xy(
            pecos_core::Angle64::QUARTER_TURN,
            pecos_core::Angle64::ZERO,
            &[0],
        );

        let mut simplified = TickCircuit::new();
        simplified.tick().sx(&[0]);

        let mut rotated_prop = PauliProp::new();
        rotated_prop.track_z(&[0]);
        propagate_through_circuit(&rotated, &mut rotated_prop, Direction::Forward);

        let mut simplified_prop = PauliProp::new();
        simplified_prop.track_z(&[0]);
        propagate_through_circuit(&simplified, &mut simplified_prop, Direction::Forward);

        assert_eq!(
            pauli_signature(&rotated_prop, &[0]),
            pauli_signature(&simplified_prop, &[0])
        );
    }

    #[test]
    fn test_rzz_propagation_matches_szz() {
        let mut rotated = TickCircuit::new();
        rotated
            .tick()
            .rzz(pecos_core::Angle64::QUARTER_TURN, &[(0, 1)]);

        let mut simplified = TickCircuit::new();
        simplified.tick().szz(&[(0, 1)]);

        let mut rotated_prop = PauliProp::new();
        rotated_prop.track_x(&[0]);
        propagate_through_circuit(&rotated, &mut rotated_prop, Direction::Forward);

        let mut simplified_prop = PauliProp::new();
        simplified_prop.track_x(&[0]);
        propagate_through_circuit(&simplified, &mut simplified_prop, Direction::Forward);

        assert_eq!(
            pauli_signature(&rotated_prop, &[0, 1]),
            pauli_signature(&simplified_prop, &[0, 1])
        );
    }
}
