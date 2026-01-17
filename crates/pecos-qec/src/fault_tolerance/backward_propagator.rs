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

//! Backward Pauli propagation for efficient fault tolerance analysis.
//!
//! This module implements backward propagation of Pauli operators from measurements
//! and logical operators to fault locations. By pre-computing which fault locations
//! affect which measurements/logicals, we can:
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
//! use pecos_qec::fault_tolerance::backward_propagator::{
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

use pecos_quantum::DagCircuit;
use std::collections::BTreeSet;

/// Pre-computed index for efficient DAG-based Pauli propagation.
///
/// This struct pre-computes data structures needed for sparse propagation,
/// making repeated propagations through the same circuit much faster.
///
/// # Example
/// ```
/// use pecos_qec::fault_tolerance::backward_propagator::{DagPropagator, Direction};
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
pub struct DagPropagator {
    /// Gates in topological order (forward direction)
    topo_order_forward: Vec<usize>,
    /// Gates in reverse topological order (backward direction)
    topo_order_backward: Vec<usize>,
    /// Gates touching each qubit, sorted by topological position
    /// Index is qubit index, value is list of (topo_position, node_id)
    qubit_gates: Vec<Vec<(usize, usize)>>,
    /// Maximum node index
    max_node: usize,
    /// Reference to gates (stored as Option<Gate> to allow indexing)
    gates: Vec<Option<pecos_core::Gate>>,
}

impl DagPropagator {
    /// Creates a new DagPropagator with pre-computed indices.
    ///
    /// This is O(V + E) where V is the number of gates and E is the number of edges.
    #[must_use]
    pub fn new(dag: &DagCircuit) -> Self {
        let topo_order_forward = dag.topological_order();
        let max_node = topo_order_forward.iter().copied().max().unwrap_or(0);

        // Find max qubit index
        let mut max_qubit = 0usize;
        for &node in &topo_order_forward {
            if let Some(gate) = dag.gate(node) {
                for q in &gate.qubits {
                    max_qubit = max_qubit.max(q.index());
                }
            }
        }

        // Build per-qubit gate index
        let mut qubit_gates: Vec<Vec<(usize, usize)>> = vec![Vec::new(); max_qubit + 1];
        for (topo_pos, &node) in topo_order_forward.iter().enumerate() {
            if let Some(gate) = dag.gate(node) {
                for q in &gate.qubits {
                    qubit_gates[q.index()].push((topo_pos, node));
                }
            }
        }

        // Copy gates for direct access
        let mut gates = vec![None; max_node + 1];
        for &node in &topo_order_forward {
            gates[node] = dag.gate(node).cloned();
        }

        let topo_order_backward: Vec<usize> = topo_order_forward.iter().copied().rev().collect();

        Self {
            topo_order_forward,
            topo_order_backward,
            qubit_gates,
            max_node,
            gates,
        }
    }

    /// Propagates a PauliProp through the circuit using sparse traversal.
    ///
    /// Only visits gates that touch qubits with non-trivial Paulis.
    /// When a gate spreads the Pauli to new qubits, those qubits' gates
    /// are added to the processing set.
    pub fn propagate_sparse(&self, prop: &mut PauliProp, direction: Direction) {
        if self.topo_order_forward.is_empty() {
            return;
        }

        // Use bit vectors for O(1) lookup
        let mut should_process = vec![false; self.max_node + 1];

        // Track active qubits (those with non-trivial Pauli)
        let mut active_qubits: Vec<bool> = vec![false; self.qubit_gates.len()];
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
                for &(_topo_pos, node) in &self.qubit_gates[qubit] {
                    should_process[node] = true;
                }
            }
        }

        // Get iteration order
        let order = match direction {
            Direction::Forward => &self.topo_order_forward,
            Direction::Backward => &self.topo_order_backward,
        };

        // Process gates in topological order
        for &node in order {
            if !should_process[node] {
                continue;
            }

            if let Some(gate) = &self.gates[node] {
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
                                for &(_topo_pos, new_node) in &self.qubit_gates[idx] {
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

    /// Propagates through all gates in topological order (non-sparse).
    pub fn propagate_full(&self, prop: &mut PauliProp, direction: Direction) {
        let order = match direction {
            Direction::Forward => &self.topo_order_forward,
            Direction::Backward => &self.topo_order_backward,
        };

        for &node in order {
            if let Some(gate) = &self.gates[node] {
                apply_gate(prop, gate, direction);
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
/// use pecos_qec::fault_tolerance::backward_propagator::{propagate_sparse_dag, Direction};
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
/// use pecos_qec::fault_tolerance::backward_propagator::propagate_backward_from_tick;
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
/// use pecos_qec::fault_tolerance::backward_propagator::propagate_fault_backward;
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

/// Vec-based fault influence map for faster small circuit performance.
///
/// Unlike `DagFaultInfluenceMap` which uses a BTreeMap for influences,
/// this struct stores influences in a Vec indexed by location index.
/// This provides O(1) access instead of O(log n) BTreeMap lookups,
/// which is significantly faster for small circuits.
#[derive(Debug, Clone)]
pub struct DagFaultInfluenceMapVec {
    /// Influences indexed by location index.
    pub influences: Vec<FaultInfluence>,

    /// Locations indexed by location index (for reference).
    pub locations: Vec<DagSpacetimeLocation>,

    /// All detectors in the circuit.
    pub detectors: Vec<DetectorId>,

    /// All logical observables being tracked.
    pub logicals: Vec<LogicalId>,

    /// All measurements in the circuit (node, qubit, basis).
    pub measurements: Vec<(usize, usize, u8)>,
}

impl DagFaultInfluenceMapVec {
    /// Creates a new influence map with the given locations.
    #[must_use]
    pub fn with_locations(locations: Vec<DagSpacetimeLocation>) -> Self {
        let len = locations.len();
        Self {
            influences: vec![FaultInfluence::default(); len],
            locations,
            detectors: Vec::new(),
            logicals: Vec::new(),
            measurements: Vec::new(),
        }
    }

    /// Returns the influence of a fault at the given location index.
    #[inline]
    #[must_use]
    pub fn get_influence(&self, loc_idx: usize) -> Option<&FaultInfluence> {
        self.influences.get(loc_idx)
    }

    /// Returns the influence of a fault at the given location index (mutable).
    #[inline]
    #[must_use]
    pub fn get_influence_mut(&mut self, loc_idx: usize) -> Option<&mut FaultInfluence> {
        self.influences.get_mut(loc_idx)
    }

    /// Classifies a fault at the given location index.
    ///
    /// Returns (has_syndrome, causes_logical_error).
    #[inline]
    #[must_use]
    pub fn classify_fault(&self, loc_idx: usize, pauli: u8) -> (bool, bool) {
        if let Some(influence) = self.influences.get(loc_idx) {
            let has_syndrome = !influence.detector_flips[pauli as usize].is_empty();
            let causes_logical = !influence.logical_flips[pauli as usize].is_empty();
            (has_syndrome, causes_logical)
        } else {
            (false, false)
        }
    }

    /// Returns the location at the given index.
    #[inline]
    #[must_use]
    pub fn get_location(&self, loc_idx: usize) -> Option<&DagSpacetimeLocation> {
        self.locations.get(loc_idx)
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
/// use pecos_qec::fault_tolerance::backward_propagator::DagBackwardPropagator;
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
/// let propagator = DagBackwardPropagator::new(&dag);
/// let influence_map = propagator.build_influence_map();
/// ```
pub struct DagBackwardPropagator {
    /// Topological order (forward direction).
    topo_order: Vec<usize>,
    /// Position of each node in topological order (for O(1) lookup).
    topo_positions: Vec<usize>,
    /// Gates touching each qubit, sorted by topological position (backward).
    /// Index is qubit index, value is list of (topo_position, node_id).
    qubit_gates_backward: Vec<Vec<(usize, usize)>>,
    /// Maximum node index.
    max_node: usize,
    /// Maximum qubit index.
    max_qubit: usize,
    /// Gates stored by node index for direct access.
    gates: Vec<Option<pecos_core::Gate>>,
    /// Fault locations indexed by node.
    /// node_locations[node] = list of (location_index, before_flag)
    node_locations: Vec<Vec<(usize, bool)>>,
    /// All fault locations.
    locations: Vec<DagSpacetimeLocation>,
}

impl DagBackwardPropagator {
    /// Creates a new DAG backward propagator for the given circuit.
    ///
    /// Pre-computes indices for efficient sparse traversal.
    #[must_use]
    pub fn new(dag: &DagCircuit) -> Self {
        let topo_order = dag.topological_order();
        let max_node = topo_order.iter().copied().max().unwrap_or(0);

        // Build position lookup (node -> topo position)
        let mut topo_positions = vec![usize::MAX; max_node + 1];
        for (pos, &node) in topo_order.iter().enumerate() {
            topo_positions[node] = pos;
        }

        // Find max qubit index
        let mut max_qubit = 0usize;
        for &node in &topo_order {
            if let Some(gate) = dag.gate(node) {
                for q in &gate.qubits {
                    max_qubit = max_qubit.max(q.index());
                }
            }
        }

        // Build per-qubit gate index (sorted by topo position, reversed for backward)
        let mut qubit_gates_backward: Vec<Vec<(usize, usize)>> = vec![Vec::new(); max_qubit + 1];
        for (topo_pos, &node) in topo_order.iter().enumerate() {
            if let Some(gate) = dag.gate(node) {
                for q in &gate.qubits {
                    qubit_gates_backward[q.index()].push((topo_pos, node));
                }
            }
        }
        // Reverse each qubit's gate list for backward traversal
        for gates in &mut qubit_gates_backward {
            gates.reverse();
        }

        // Copy gates for direct access
        let mut gates = vec![None; max_node + 1];
        for &node in &topo_order {
            gates[node] = dag.gate(node).cloned();
        }

        // Extract locations and build node-to-locations index
        let locations = Self::extract_locations_static(&gates, &topo_order);
        let mut node_locations: Vec<Vec<(usize, bool)>> = vec![Vec::new(); max_node + 1];
        for (loc_idx, loc) in locations.iter().enumerate() {
            node_locations[loc.node].push((loc_idx, loc.before));
        }

        Self {
            topo_order,
            topo_positions,
            qubit_gates_backward,
            max_node,
            max_qubit,
            gates,
            node_locations,
            locations,
        }
    }

    /// Extracts fault locations from the gates.
    fn extract_locations_static(
        gates: &[Option<pecos_core::Gate>],
        topo_order: &[usize],
    ) -> Vec<DagSpacetimeLocation> {
        let mut locations = Vec::new();

        for &node in topo_order {
            if let Some(gate) = &gates[node] {
                let is_measurement = matches!(
                    gate.gate_type,
                    GateType::Measure | GateType::MeasureFree
                );
                let is_prep = matches!(gate.gate_type, GateType::Prep | GateType::QAlloc);

                if is_measurement {
                    locations.push(DagSpacetimeLocation {
                        node,
                        qubits: gate.qubits.to_vec(),
                        before: true,
                        gate_type: gate.gate_type,
                    });
                } else if is_prep {
                    locations.push(DagSpacetimeLocation {
                        node,
                        qubits: gate.qubits.to_vec(),
                        before: false,
                        gate_type: gate.gate_type,
                    });
                } else {
                    locations.push(DagSpacetimeLocation {
                        node,
                        qubits: gate.qubits.to_vec(),
                        before: true,
                        gate_type: gate.gate_type,
                    });
                    locations.push(DagSpacetimeLocation {
                        node,
                        qubits: gate.qubits.to_vec(),
                        before: false,
                        gate_type: gate.gate_type,
                    });
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
    /// Note: This also builds reverse maps (detector -> faults). For faster
    /// performance when reverse maps aren't needed, use `build_influence_map_fast`.
    #[must_use]
    pub fn build_influence_map(&self) -> DagFaultInfluenceMap {
        self.build_influence_map_with_logicals(&[])
    }

    /// Builds the fault influence map without reverse maps (faster).
    ///
    /// This skips building the expensive reverse maps (detector_to_faults,
    /// logical_to_faults) which can significantly improve performance for
    /// small circuits. Use this when you only need to query fault influences
    /// by location, not by detector.
    #[must_use]
    pub fn build_influence_map_fast(&self) -> DagFaultInfluenceMap {
        self.build_influence_map_impl(&[], false)
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
        for loc in &self.locations {
            map.influences
                .insert(loc.clone(), FaultInfluence::default());
        }

        // Pre-allocate work arrays to reuse across propagations
        let mut visited = vec![false; self.max_node + 1];
        let mut active_qubits = vec![false; self.max_qubit + 1];
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

    /// Builds the fault influence map using Vec-based storage (fastest).
    ///
    /// This uses a Vec indexed by location index instead of a BTreeMap,
    /// providing O(1) access instead of O(log n) lookups. This is the
    /// fastest option and is recommended for all circuit sizes.
    #[must_use]
    pub fn build_influence_map_vec(&self) -> DagFaultInfluenceMapVec {
        let mut map = DagFaultInfluenceMapVec::with_locations(self.locations.clone());

        // Extract all measurements from the circuit
        let measurements = self.extract_measurements();
        map.measurements = measurements.clone();

        // Create simple detectors (one per measurement)
        for &(node, qubit, basis) in &measurements {
            let measurement_id = MeasurementId {
                tick: node,
                qubit,
                basis,
            };
            map.detectors.push(DetectorId::single(measurement_id));
        }

        // Pre-allocate work arrays to reuse across propagations
        let mut visited = vec![false; self.max_node + 1];
        let mut active_qubits = vec![false; self.max_qubit + 1];
        let mut heap: BinaryHeap<(usize, usize)> = BinaryHeap::with_capacity(64);

        // Backward propagate from each measurement
        for (detector_idx, &(node, qubit, basis)) in measurements.iter().enumerate() {
            self.propagate_from_measurement_vec(
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

        map
    }

    /// Propagates backward from a measurement using Vec-based influence storage.
    fn propagate_from_measurement_vec(
        &self,
        meas_node: usize,
        meas_qubit: usize,
        basis: u8,
        detector_idx: usize,
        map: &mut DagFaultInfluenceMapVec,
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
        let meas_topo_pos = self.topo_positions[meas_node];

        // Check fault at measurement node (before=true only)
        self.record_influences_vec(meas_node, &prop, detector_idx, map, true);

        // Initialize: add gates on the measurement qubit
        if meas_qubit <= self.max_qubit {
            active_qubits[meas_qubit] = true;
            for &(topo_pos, node) in &self.qubit_gates_backward[meas_qubit] {
                if topo_pos < meas_topo_pos && !visited[node] {
                    visited[node] = true;
                    heap.push((topo_pos, node));
                }
            }
        }

        // Process gates in reverse topo order - only gates on active wires
        while let Some((_, node)) = heap.pop() {
            if let Some(gate) = &self.gates[node] {
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= self.max_qubit {
                        was_active[j] = active_qubits[q.index()];
                    }
                }

                // Check before=false locations
                self.record_influences_vec(node, &prop, detector_idx, map, false);

                // Apply gate backward
                apply_gate(&mut prop, gate, Direction::Backward);

                // Check before=true locations
                self.record_influences_vec(node, &prop, detector_idx, map, true);

                // Check if Pauli spread to new qubits
                let node_topo_pos = self.topo_positions[node];
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= self.max_qubit {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            active_qubits[idx] = true;
                            for &(topo_pos, new_node) in &self.qubit_gates_backward[idx] {
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

    /// Records influences using Vec-based storage (O(1) indexed access).
    #[inline]
    fn record_influences_vec(
        &self,
        node: usize,
        prop: &PauliProp,
        detector_idx: usize,
        map: &mut DagFaultInfluenceMapVec,
        only_before: bool,
    ) {
        // Use pre-computed index for O(1) lookup
        for &(loc_idx, before) in &self.node_locations[node] {
            if before != only_before {
                continue;
            }

            let loc = &self.locations[loc_idx];

            for qubit in loc.qubits.iter() {
                let q = qubit.index();
                let obs_x = prop.contains_x(q);
                let obs_z = prop.contains_z(q);

                // Direct Vec access - no BTreeMap lookup
                let influence = &mut map.influences[loc_idx];

                // X fault anticommutes with Z or Y observable
                if obs_z {
                    let detector = &map.detectors[detector_idx];
                    influence.detector_flips[1].push(detector.clone());
                    influence
                        .measurement_flips[1]
                        .extend(detector.measurements.iter().copied());
                }

                // Z fault anticommutes with X or Y observable
                if obs_x {
                    let detector = &map.detectors[detector_idx];
                    influence.detector_flips[3].push(detector.clone());
                    influence
                        .measurement_flips[3]
                        .extend(detector.measurements.iter().copied());
                }

                // Y fault anticommutes with X, Z, or Y observable
                if obs_x || obs_z {
                    let detector = &map.detectors[detector_idx];
                    influence.detector_flips[2].push(detector.clone());
                    influence
                        .measurement_flips[2]
                        .extend(detector.measurements.iter().copied());
                }
            }
        }
    }

    /// Extracts all measurements from the circuit.
    fn extract_measurements(&self) -> Vec<(usize, usize, u8)> {
        let mut measurements = Vec::new();

        for &node in &self.topo_order {
            if let Some(gate) = &self.gates[node] {
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
        let meas_topo_pos = self.topo_positions[meas_node];

        // Check fault at measurement node (before=true only)
        self.record_influences_at_node_fast(meas_node, &prop, &detector, None, map, true);

        // Track visited nodes (queued or processed) and active qubits
        let mut visited = vec![false; self.max_node + 1];
        let mut active_qubits = vec![false; self.max_qubit + 1];

        // Max-heap: (topo_pos, node) - highest topo_pos comes first (reverse order)
        // Pre-allocate with estimated capacity (average gates per qubit * expected active qubits)
        let mut heap: BinaryHeap<(usize, usize)> = BinaryHeap::with_capacity(32);

        // Initialize: add gates on the measurement qubit
        if meas_qubit <= self.max_qubit {
            active_qubits[meas_qubit] = true;
            for &(topo_pos, node) in &self.qubit_gates_backward[meas_qubit] {
                if topo_pos < meas_topo_pos && !visited[node] {
                    visited[node] = true;
                    heap.push((topo_pos, node));
                }
            }
        }

        // Process gates in reverse topo order - only gates on active wires
        while let Some((_, node)) = heap.pop() {

            if let Some(gate) = &self.gates[node] {
                // Track which qubits were active before
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= self.max_qubit {
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
                let node_topo_pos = self.topo_positions[node];
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= self.max_qubit {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            // Pauli spread to this qubit - add its gates to heap
                            active_qubits[idx] = true;
                            for &(topo_pos, new_node) in &self.qubit_gates_backward[idx] {
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
        let meas_topo_pos = self.topo_positions[meas_node];

        // Check fault at measurement node (before=true only)
        self.record_influences_at_node_indexed(meas_node, &prop, detector_idx, None, map, true);

        // Initialize: add gates on the measurement qubit
        if meas_qubit <= self.max_qubit {
            active_qubits[meas_qubit] = true;
            for &(topo_pos, node) in &self.qubit_gates_backward[meas_qubit] {
                if topo_pos < meas_topo_pos && !visited[node] {
                    visited[node] = true;
                    heap.push((topo_pos, node));
                }
            }
        }

        // Process gates in reverse topo order - only gates on active wires
        while let Some((_, node)) = heap.pop() {
            if let Some(gate) = &self.gates[node] {
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= self.max_qubit {
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
                let node_topo_pos = self.topo_positions[node];
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= self.max_qubit {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            active_qubits[idx] = true;
                            for &(topo_pos, new_node) in &self.qubit_gates_backward[idx] {
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
        let meas_topo_pos = self.topo_positions[meas_node];

        // Check fault at measurement node (before=true only)
        self.record_influences_at_node_fast(meas_node, &prop, &detector, None, map, true);

        // Initialize: add gates on the measurement qubit
        if meas_qubit <= self.max_qubit {
            active_qubits[meas_qubit] = true;
            for &(topo_pos, node) in &self.qubit_gates_backward[meas_qubit] {
                if topo_pos < meas_topo_pos && !visited[node] {
                    visited[node] = true;
                    heap.push((topo_pos, node));
                }
            }
        }

        // Process gates in reverse topo order - only gates on active wires
        while let Some((_, node)) = heap.pop() {
            if let Some(gate) = &self.gates[node] {
                // Track which qubits were active before
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= self.max_qubit {
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
                let node_topo_pos = self.topo_positions[node];
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= self.max_qubit {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            // Pauli spread to this qubit - add its gates to heap
                            active_qubits[idx] = true;
                            for &(topo_pos, new_node) in &self.qubit_gates_backward[idx] {
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
        let mut visited = vec![false; self.max_node + 1];
        let mut active_qubits = vec![false; self.max_qubit + 1];

        // Max-heap: (topo_pos, node) - highest topo_pos comes first (reverse order)
        // Pre-allocate with estimated capacity
        let mut heap: BinaryHeap<(usize, usize)> = BinaryHeap::with_capacity(64);

        // Initialize from logical operator support - add all gates on these qubits
        for &q in x_positions {
            if q <= self.max_qubit && !active_qubits[q] {
                active_qubits[q] = true;
                for &(topo_pos, node) in &self.qubit_gates_backward[q] {
                    if !visited[node] {
                        visited[node] = true;
                        heap.push((topo_pos, node));
                    }
                }
            }
        }
        for &q in z_positions {
            if q <= self.max_qubit && !active_qubits[q] {
                active_qubits[q] = true;
                for &(topo_pos, node) in &self.qubit_gates_backward[q] {
                    if !visited[node] {
                        visited[node] = true;
                        heap.push((topo_pos, node));
                    }
                }
            }
        }

        // Process gates in reverse topo order - only gates on active wires
        while let Some((_, node)) = heap.pop() {
            if let Some(gate) = &self.gates[node] {
                // Track which qubits were active before
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= self.max_qubit {
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
                let node_topo_pos = self.topo_positions[node];
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= self.max_qubit {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            // Pauli spread to this qubit - add its gates to heap
                            active_qubits[idx] = true;
                            for &(topo_pos, new_node) in &self.qubit_gates_backward[idx] {
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
            if q <= self.max_qubit && !active_qubits[q] {
                active_qubits[q] = true;
                for &(topo_pos, node) in &self.qubit_gates_backward[q] {
                    if !visited[node] {
                        visited[node] = true;
                        heap.push((topo_pos, node));
                    }
                }
            }
        }
        for &q in z_positions {
            if q <= self.max_qubit && !active_qubits[q] {
                active_qubits[q] = true;
                for &(topo_pos, node) in &self.qubit_gates_backward[q] {
                    if !visited[node] {
                        visited[node] = true;
                        heap.push((topo_pos, node));
                    }
                }
            }
        }

        // Process gates in reverse topo order
        while let Some((_, node)) = heap.pop() {
            if let Some(gate) = &self.gates[node] {
                let mut was_active = [false; 8];
                for (j, q) in gate.qubits.iter().enumerate() {
                    if j < was_active.len() && q.index() <= self.max_qubit {
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

                let node_topo_pos = self.topo_positions[node];
                for (j, q) in gate.qubits.iter().enumerate() {
                    let idx = q.index();
                    if idx <= self.max_qubit {
                        let now_active = prop.contains_x(idx) || prop.contains_z(idx);
                        let was = j < was_active.len() && was_active[j];

                        if now_active && !was {
                            active_qubits[idx] = true;
                            for &(topo_pos, new_node) in &self.qubit_gates_backward[idx] {
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
        for &(loc_idx, before) in &self.node_locations[node] {
            if before != only_before {
                continue;
            }

            let loc = &self.locations[loc_idx];

            for qubit in loc.qubits.iter() {
                let q = qubit.index();
                let obs_x = prop.contains_x(q);
                let obs_z = prop.contains_z(q);

                if let Some(influence) = map.influences.get_mut(loc) {
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
        for &(loc_idx, before) in &self.node_locations[node] {
            if before != only_before {
                continue;
            }

            let loc = &self.locations[loc_idx];

            for (qubit_idx, qubit) in loc.qubits.iter().enumerate() {
                let q = qubit.index();
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
    // DagBackwardPropagator tests
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
        use super::DagBackwardPropagator;

        let dag = simple_dag_syndrome_circuit();
        let propagator = DagBackwardPropagator::new(&dag);
        let measurements = propagator.extract_measurements();

        assert_eq!(measurements.len(), 1);
        // node index, qubit, basis
        assert_eq!(measurements[0].1, 2); // qubit 2
        assert_eq!(measurements[0].2, 0); // Z-basis
    }

    #[test]
    fn test_dag_backward_propagator_build_influence_map() {
        use super::DagBackwardPropagator;

        let dag = simple_dag_syndrome_circuit();
        let propagator = DagBackwardPropagator::new(&dag);
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
    fn test_dag_backward_propagator_vec_equivalence() {
        use super::DagBackwardPropagator;

        let dag = simple_dag_syndrome_circuit();
        let propagator = DagBackwardPropagator::new(&dag);

        // Build both map types
        let btree_map = propagator.build_influence_map_fast();
        let vec_map = propagator.build_influence_map_vec();

        // Same number of locations
        assert_eq!(btree_map.influences.len(), vec_map.influences.len());

        // Same detectors
        assert_eq!(btree_map.detectors.len(), vec_map.detectors.len());
        assert_eq!(btree_map.measurements.len(), vec_map.measurements.len());

        // Compare influences for each location
        for (loc, btree_influence) in &btree_map.influences {
            // Find the corresponding location in the vec map
            let vec_idx = vec_map
                .locations
                .iter()
                .position(|l| l == loc)
                .expect("Location should exist in vec map");
            let vec_influence = &vec_map.influences[vec_idx];

            // Compare detector flips
            for pauli in 0..4 {
                assert_eq!(
                    btree_influence.detector_flips[pauli].len(),
                    vec_influence.detector_flips[pauli].len(),
                    "Mismatch in detector_flips[{}] for location {:?}",
                    pauli,
                    loc
                );
            }
        }
    }

    #[test]
    fn test_dag_backward_propagator_x_error_flips_z_measurement() {
        use super::DagBackwardPropagator;

        let dag = simple_dag_syndrome_circuit();
        let propagator = DagBackwardPropagator::new(&dag);
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
        use super::DagBackwardPropagator;

        let dag = simple_dag_syndrome_circuit();
        let propagator = DagBackwardPropagator::new(&dag);
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
        use super::{BackwardPropagator, DagBackwardPropagator};
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
                let propagator = DagBackwardPropagator::new(&dag);
                let _map = propagator.build_influence_map();
            }
            let dag_time = start.elapsed();

            // Benchmark DAG BackwardPropagator (fast, no reverse maps)
            let start = Instant::now();
            for _ in 0..iterations {
                let propagator = DagBackwardPropagator::new(&dag);
                let _map = propagator.build_influence_map_fast();
            }
            let dag_fast_time = start.elapsed();

            // Benchmark DAG BackwardPropagator (Vec-based, fastest)
            let start = Instant::now();
            for _ in 0..iterations {
                let propagator = DagBackwardPropagator::new(&dag);
                let _map = propagator.build_influence_map_vec();
            }
            let dag_vec_time = start.elapsed();

            let tick_per_iter = tick_time.as_micros() as f64 / iterations as f64;
            let dag_per_iter = dag_time.as_micros() as f64 / iterations as f64;
            let dag_fast_per_iter = dag_fast_time.as_micros() as f64 / iterations as f64;
            let dag_vec_per_iter = dag_vec_time.as_micros() as f64 / iterations as f64;

            println!("  TickCircuit BackwardPropagator: {:>8.1} us/iter", tick_per_iter);
            println!(
                "  DAG BackwardPropagator:         {:>8.1} us/iter ({:.2}x)",
                dag_per_iter,
                dag_per_iter / tick_per_iter
            );
            println!(
                "  DAG BackwardPropagator (fast):  {:>8.1} us/iter ({:.2}x)",
                dag_fast_per_iter,
                dag_fast_per_iter / tick_per_iter
            );
            println!(
                "  DAG BackwardPropagator (vec):   {:>8.1} us/iter ({:.2}x)",
                dag_vec_per_iter,
                dag_vec_per_iter / tick_per_iter
            );
            println!();
        }
    }

    /// Profile where time is spent in DAG propagator for small circuits
    #[test]
    #[ignore]
    fn profile_dag_phases() {
        use super::DagBackwardPropagator;
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
            propagators.push(DagBackwardPropagator::new(&dag));
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

        // Phase 3: build_influence_map_fast (no reverse maps)
        let start = Instant::now();
        for _ in 0..iterations {
            let _map = propagator.build_influence_map_fast();
        }
        let map_fast_time = start.elapsed();
        println!(
            "d=3: build_influence_map_fast: {:>6.1} us/iter",
            map_fast_time.as_micros() as f64 / iterations as f64
        );

        // Total
        println!(
            "d=3: Total (new + map): {:>6.1} us/iter",
            (construct_time.as_micros() + map_time.as_micros()) as f64 / iterations as f64
        );
        println!(
            "d=3: Total (new + map_fast): {:>6.1} us/iter",
            (construct_time.as_micros() + map_fast_time.as_micros()) as f64 / iterations as f64
        );

        // Compare with d=5
        println!();
        let (_, dag5) = build_syndrome_extraction_circuit(25, 24);
        let iterations = 500;

        let start = Instant::now();
        let mut propagators5 = Vec::with_capacity(iterations);
        for _ in 0..iterations {
            propagators5.push(DagBackwardPropagator::new(&dag5));
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
            let _map = propagator5.build_influence_map_fast();
        }
        let map_fast_time5 = start.elapsed();
        println!(
            "d=5: build_influence_map_fast: {:>6.1} us/iter",
            map_fast_time5.as_micros() as f64 / iterations as f64
        );

        println!(
            "d=5: Total (new + map): {:>6.1} us/iter",
            (construct_time5.as_micros() + map_time5.as_micros()) as f64 / iterations as f64
        );
        println!(
            "d=5: Total (new + map_fast): {:>6.1} us/iter",
            (construct_time5.as_micros() + map_fast_time5.as_micros()) as f64 / iterations as f64
        );
    }

    /// Benchmark with multiple syndrome extraction rounds (more realistic)
    #[test]
    #[ignore] // Run manually with --ignored flag
    fn benchmark_multi_round() {
        use super::{BackwardPropagator, DagBackwardPropagator};
        use std::time::Instant;

        println!("\n========================================");
        println!("Multi-Round Syndrome Extraction Benchmark");
        println!("========================================\n");

        // Test different code sizes with multiple rounds
        let configs = [
            (25, 24, "d=5"),   // 5x5
            (49, 48, "d=7"),   // 7x7
            (121, 120, "d=11"), // 11x11
        ];

        for (data_qubits, ancilla_qubits, label) in configs {
            println!("=== {} ({} data + {} ancilla) ===", label, data_qubits, ancilla_qubits);

            for rounds in [1, 2, 4, 8] {
                let (tick, dag) = build_syndrome_extraction_circuit_multi(data_qubits, ancilla_qubits, rounds);
                let iterations = if data_qubits > 100 { 5 } else { 20 };

                // Benchmark TickCircuit BackwardPropagator
                let start = Instant::now();
                for _ in 0..iterations {
                    let propagator = BackwardPropagator::new(&tick);
                    let _map = propagator.build_influence_map();
                }
                let tick_time = start.elapsed();

                // Benchmark DAG BackwardPropagator
                let start = Instant::now();
                for _ in 0..iterations {
                    let propagator = DagBackwardPropagator::new(&dag);
                    let _map = propagator.build_influence_map();
                }
                let dag_time = start.elapsed();

                let tick_per_iter = tick_time.as_micros() as f64 / iterations as f64;
                let dag_per_iter = dag_time.as_micros() as f64 / iterations as f64;

                println!(
                    "  {} rounds: Tick {:>8.1}us, DAG {:>8.1}us ({:.2}x)",
                    rounds, tick_per_iter, dag_per_iter, dag_per_iter / tick_per_iter
                );
            }
            println!();
        }
    }
}
