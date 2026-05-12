// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! DAG circuit implementation.
//!
//! This module provides [`DagCircuit`], a directed acyclic graph representation
//! of quantum circuits where nodes are gates and edges are qubit wires.
//!
//! The design follows a wire-edge DAG model: edges represent qubit wires
//! flowing between gates, not just abstract dependencies.

use std::collections::{BTreeMap, BTreeSet};

use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, Gate, QubitId, TimeUnits};
use pecos_num::dag::{DAG, DagWouldCycleError};

use crate::circuit::{Circuit, CircuitMut, GateHandle, GateView};

// Re-export attribute type for use with DagCircuit
pub use pecos_num::graph::Attribute;

// ==================== Traversal Index ====================

/// Pre-computed traversal indices for efficient DAG iteration.
///
/// This struct contains cached data structures that enable O(1) lookups
/// for common traversal patterns. It's designed for data-oriented access
/// with flat arrays that are cache-friendly.
///
/// Build once from a [`DagCircuit`], then use for multiple traversal operations.
///
/// # Example
///
/// ```
/// use pecos_quantum::DagCircuit;
///
/// let mut dag = DagCircuit::new();
/// dag.h(&[0]);
/// dag.cx(&[(0, 1)]);
/// dag.mz(&[0]);
/// dag.mz(&[1]);
///
/// let index = dag.build_traversal_index();
/// assert_eq!(index.max_qubit(), 1);
/// assert_eq!(index.topo_order().len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct DagTraversalIndex {
    /// Gates in topological order (forward direction).
    topo_order: Vec<usize>,
    /// Position of each node in topological order (node -> position).
    /// Enables O(1) lookup of relative ordering.
    topo_positions: Vec<usize>,
    /// Gates touching each qubit, sorted by topological position.
    /// `qubit_gates[qubit]` = list of `(topo_position, node_id)`.
    qubit_gates: Vec<Vec<(usize, usize)>>,
    /// Maximum node index in the circuit.
    max_node: usize,
    /// Maximum qubit index in the circuit.
    max_qubit: usize,
}

impl DagTraversalIndex {
    /// Returns the topological order of nodes.
    #[inline]
    #[must_use]
    pub fn topo_order(&self) -> &[usize] {
        &self.topo_order
    }

    /// Returns the topological order reversed (for backward traversal).
    pub fn topo_order_reversed(&self) -> impl Iterator<Item = usize> + '_ {
        self.topo_order.iter().copied().rev()
    }

    /// Returns the topological position of a node (O(1) lookup).
    #[inline]
    #[must_use]
    pub fn topo_position(&self, node: usize) -> usize {
        self.topo_positions[node]
    }

    /// Returns gates on a qubit in forward topological order.
    /// Each entry is `(topo_position, node_id)`.
    #[inline]
    #[must_use]
    pub fn qubit_gates(&self, qubit: usize) -> &[(usize, usize)] {
        &self.qubit_gates[qubit]
    }

    /// Returns gates on a qubit in reverse topological order (for backward traversal).
    pub fn qubit_gates_reversed(&self, qubit: usize) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.qubit_gates[qubit].iter().copied().rev()
    }

    /// Returns the maximum node index.
    #[inline]
    #[must_use]
    pub fn max_node(&self) -> usize {
        self.max_node
    }

    /// Returns the maximum qubit index.
    #[inline]
    #[must_use]
    pub fn max_qubit(&self) -> usize {
        self.max_qubit
    }

    /// Returns the number of qubits (`max_qubit` + 1).
    #[inline]
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.max_qubit + 1
    }

    /// Returns the number of gate nodes in the traversal index.
    ///
    /// A batched DAG node still contributes one node here. Use
    /// [`DagCircuit::gate_count`] when you need the number of individual gates
    /// represented by batched gate nodes.
    #[inline]
    #[must_use]
    pub fn num_gate_nodes(&self) -> usize {
        self.topo_order.len()
    }

    /// Creates reusable work buffers sized for this circuit.
    #[must_use]
    pub fn create_work_buffers(&self) -> TraversalWorkBuffers {
        TraversalWorkBuffers::new(self.max_node, self.max_qubit)
    }

    // ==================== Local Graph Traversal ====================

    /// Returns the predecessor node on the given qubit, if any.
    ///
    /// This is the gate immediately before this node on the qubit wire.
    /// Returns `None` if this is the first gate on the qubit or if the
    /// node doesn't touch this qubit.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::{DagCircuit, Gate, QubitId};
    ///
    /// let mut dag = DagCircuit::new();
    /// let h = dag.add_gate(Gate::h(&[0]));
    /// let cx = dag.add_gate(Gate::cx(&[(0, 1)]));
    /// dag.connect(h, cx, QubitId::from(0)).unwrap();
    /// let mz = dag.add_gate(Gate::mz(&[QubitId::from(0)]));
    /// dag.connect(cx, mz, QubitId::from(0)).unwrap();
    ///
    /// let index = dag.build_traversal_index();
    ///
    /// // mz's predecessor on qubit 0 is cx
    /// assert_eq!(index.predecessor_on_qubit(mz, 0), Some(cx));
    /// // cx's predecessor on qubit 0 is h
    /// assert_eq!(index.predecessor_on_qubit(cx, 0), Some(h));
    /// // h has no predecessor on qubit 0
    /// assert_eq!(index.predecessor_on_qubit(h, 0), None);
    /// // cx's predecessor on qubit 1 is None (first gate on qubit 1)
    /// assert_eq!(index.predecessor_on_qubit(cx, 1), None);
    /// ```
    #[must_use]
    pub fn predecessor_on_qubit(&self, node: usize, qubit: usize) -> Option<usize> {
        if qubit >= self.qubit_gates.len() {
            return None;
        }

        let topo_pos = self.topo_position(node);
        let gates = &self.qubit_gates[qubit];

        // Binary search for this node's position in the qubit's gate list
        let idx = gates.binary_search_by_key(&topo_pos, |&(tp, _)| tp).ok()?;

        // Return predecessor if it exists
        if idx > 0 {
            Some(gates[idx - 1].1)
        } else {
            None
        }
    }

    /// Returns the successor node on the given qubit, if any.
    ///
    /// This is the gate immediately after this node on the qubit wire.
    /// Returns `None` if this is the last gate on the qubit or if the
    /// node doesn't touch this qubit.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::{DagCircuit, Gate, QubitId};
    ///
    /// let mut dag = DagCircuit::new();
    /// let h = dag.add_gate(Gate::h(&[0]));
    /// let cx = dag.add_gate(Gate::cx(&[(0, 1)]));
    /// dag.connect(h, cx, QubitId::from(0)).unwrap();
    /// let mz = dag.add_gate(Gate::mz(&[QubitId::from(0)]));
    /// dag.connect(cx, mz, QubitId::from(0)).unwrap();
    ///
    /// let index = dag.build_traversal_index();
    ///
    /// // h's successor on qubit 0 is cx
    /// assert_eq!(index.successor_on_qubit(h, 0), Some(cx));
    /// // cx's successor on qubit 0 is mz
    /// assert_eq!(index.successor_on_qubit(cx, 0), Some(mz));
    /// // mz has no successor on qubit 0
    /// assert_eq!(index.successor_on_qubit(mz, 0), None);
    /// ```
    #[must_use]
    pub fn successor_on_qubit(&self, node: usize, qubit: usize) -> Option<usize> {
        if qubit >= self.qubit_gates.len() {
            return None;
        }

        let topo_pos = self.topo_position(node);
        let gates = &self.qubit_gates[qubit];

        // Binary search for this node's position in the qubit's gate list
        let idx = gates.binary_search_by_key(&topo_pos, |&(tp, _)| tp).ok()?;

        // Return successor if it exists
        if idx + 1 < gates.len() {
            Some(gates[idx + 1].1)
        } else {
            None
        }
    }

    /// Returns all predecessor nodes (nodes immediately before on each qubit wire).
    ///
    /// Given the qubits this node touches, returns the immediate predecessor
    /// on each qubit wire. This is useful for backward traversal from a node.
    ///
    /// # Arguments
    /// * `node` - The node to find predecessors for
    /// * `qubits` - The qubits this node touches
    ///
    /// # Returns
    /// Iterator of `(predecessor_node, qubit)` pairs
    pub fn predecessors<'a>(
        &'a self,
        node: usize,
        qubits: &'a [usize],
    ) -> impl Iterator<Item = (usize, usize)> + 'a {
        qubits
            .iter()
            .filter_map(move |&q| self.predecessor_on_qubit(node, q).map(|pred| (pred, q)))
    }

    /// Returns all successor nodes (nodes immediately after on each qubit wire).
    ///
    /// Given the qubits this node touches, returns the immediate successor
    /// on each qubit wire. This is useful for forward traversal from a node.
    ///
    /// # Arguments
    /// * `node` - The node to find successors for
    /// * `qubits` - The qubits this node touches
    ///
    /// # Returns
    /// Iterator of `(successor_node, qubit)` pairs
    pub fn successors<'a>(
        &'a self,
        node: usize,
        qubits: &'a [usize],
    ) -> impl Iterator<Item = (usize, usize)> + 'a {
        qubits
            .iter()
            .filter_map(move |&q| self.successor_on_qubit(node, q).map(|succ| (succ, q)))
    }
}

/// Reusable work buffers for traversal algorithms.
///
/// Pre-allocate once and reuse across multiple traversals to avoid
/// repeated allocations.
#[derive(Debug, Clone)]
pub struct TraversalWorkBuffers {
    /// Visited flags indexed by node.
    pub visited: Vec<bool>,
    /// Active qubit flags.
    pub active_qubits: Vec<bool>,
    /// Priority queue for heap-based traversal.
    pub heap: std::collections::BinaryHeap<(usize, usize)>,
}

impl TraversalWorkBuffers {
    /// Creates new work buffers sized for the given circuit dimensions.
    #[must_use]
    pub fn new(max_node: usize, max_qubit: usize) -> Self {
        Self {
            visited: vec![false; max_node + 1],
            active_qubits: vec![false; max_qubit + 1],
            heap: std::collections::BinaryHeap::with_capacity(64),
        }
    }

    /// Clears all buffers for reuse.
    pub fn clear(&mut self) {
        self.visited.fill(false);
        self.active_qubits.fill(false);
        self.heap.clear();
    }
}

/// A directed acyclic graph representation of a quantum circuit.
///
/// Each node in the DAG represents a quantum gate. Edges represent qubit wires
/// flowing between gates - each edge is labeled with the [`QubitId`] it carries.
/// This design follows a wire-edge DAG model.
///
/// For a two-qubit gate like CX, there are two incoming edges (one per qubit)
/// and two outgoing edges.
///
/// This representation is useful for:
///
/// - Circuit optimization
/// - Resource estimation
/// - Noise model application (walk each qubit wire)
/// - Generating matching graphs or detector error models
///
/// # Example
///
/// ```
/// use pecos_quantum::DagCircuit;
/// use pecos_core::{Gate, QubitId};
///
/// let mut circuit = DagCircuit::new();
///
/// // Build a Bell state circuit
/// let h = circuit.add_gate(Gate::h(&[0]));
/// let cx = circuit.add_gate(Gate::cx(&[(0, 1)]));
///
/// // Connect H to CX on qubit 0
/// circuit.connect(h, cx, QubitId::from(0)).unwrap();
///
/// // Query circuit properties
/// assert_eq!(circuit.gate_count(), 2);
/// assert_eq!(circuit.wire_count(), 1);
/// ```
/// A measurement reference returned by [`DagCircuit::mz`].
///
/// Carries both the DAG node index and the qubit that was measured.
/// Dereferences to `usize` (the node index) for use in detector/observable
/// definitions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeasRef {
    /// DAG node index.
    pub node: usize,
    /// Qubit that was measured.
    pub qubit: QubitId,
}

impl std::ops::Deref for MeasRef {
    type Target = usize;
    fn deref(&self) -> &usize {
        &self.node
    }
}

impl From<MeasRef> for usize {
    fn from(m: MeasRef) -> usize {
        m.node
    }
}

/// The role of a Pauli annotation in the circuit.
///
/// All three kinds track the same thing -- whether a Pauli string flips due to
/// faults. The difference is how the answer is read out and what it means.
#[derive(Debug, Clone)]
pub enum AnnotationKind {
    /// Stabilizer check: the Pauli should be deterministic (flip = error detected).
    /// Stores measurement node indices for classical readout via XOR, plus
    /// optional coordinates for visualization/matching.
    Detector {
        measurement_nodes: Vec<usize>,
        coords: Vec<f64>,
    },
    /// Logical observable: the Pauli's flip determines a logical outcome.
    /// Stores measurement node indices for classical readout via XOR.
    Observable { measurement_nodes: Vec<usize> },
    /// Tracked Pauli: no measurement readout.
    /// Position is determined by a `TrackedPauliMeta` node in the DAG.
    TrackedPauli,
}

/// A unified Pauli annotation: detectors, observables, and tracked Paulis
/// are all Pauli strings tracked for flipping via backward propagation.
///
/// - **Detectors** are stabilizer checks that should be +1 (noiseless).
///   Their Pauli is Z on the measured qubits.
/// - **Observables** are logical operators read out via measurements.
///   Their Pauli is Z on the measured qubits.
/// - **Tracked Paulis** are arbitrary Pauli strings with no measurement readout.
///   Their Pauli is user-specified and their position comes from a meta-gate node.
#[derive(Debug, Clone)]
pub struct PauliAnnotation {
    /// The Pauli string being tracked.
    pub pauli: pecos_core::PauliString,
    /// What role this annotation plays.
    pub kind: AnnotationKind,
    /// Optional label.
    pub label: Option<String>,
}

#[derive(Debug, Clone)]
pub struct DagCircuit {
    /// The underlying DAG structure.
    dag: DAG,
    /// Gates stored by node index.
    gates: Vec<Option<Gate>>,
    /// Qubit labels for each edge, indexed by edge ID.
    edge_qubits: BTreeMap<usize, QubitId>,
    /// Tracks the most recent gate on each qubit for auto-wiring in builder mode.
    qubit_heads: BTreeMap<QubitId, usize>,
    /// Tracks the last added node for `.meta()` calls.
    last_node: Option<usize>,
    /// Maximum qubit index seen so far (updated incrementally on gate addition).
    max_qubit: usize,
    /// Unified Pauli annotations (detectors, observables, and tracked Paulis).
    annotations: Vec<PauliAnnotation>,
    /// Measurement labels (`node_index` → label).
    measurement_labels: BTreeMap<usize, String>,
}

impl DagCircuit {
    /// Creates a new empty circuit DAG.
    #[must_use]
    pub fn new() -> Self {
        Self {
            dag: DAG::new(),
            gates: Vec::new(),
            edge_qubits: BTreeMap::new(),
            qubit_heads: BTreeMap::new(),
            last_node: None,
            max_qubit: 0,
            annotations: Vec::new(),
            measurement_labels: BTreeMap::new(),
        }
    }

    /// Creates a new circuit DAG with pre-allocated capacity.
    ///
    /// # Arguments
    ///
    /// * `gates` - Expected number of gates
    /// * `wires` - Expected number of qubit wires (edges)
    #[must_use]
    pub fn with_capacity(gates: usize, wires: usize) -> Self {
        Self {
            dag: DAG::with_capacity(gates, wires),
            gates: Vec::with_capacity(gates),
            // BTreeMap doesn't support with_capacity - it allocates as needed
            edge_qubits: BTreeMap::new(),
            qubit_heads: BTreeMap::new(),
            last_node: None,
            max_qubit: 0,
            annotations: Vec::new(),
            measurement_labels: BTreeMap::new(),
        }
    }

    // ==================== Gate operations ====================

    /// Adds a validated gate to the circuit.
    ///
    /// Returns the node index of the newly added gate.
    /// The gate is not connected to any other gates yet - use [`connect`](Self::connect)
    /// to add qubit wires.
    ///
    /// # Arguments
    ///
    /// * `gate` - The gate to add
    ///
    /// # Panics
    ///
    /// Panics if [`Gate::validate`] rejects the gate payload. Use
    /// [`try_add_gate`](Self::try_add_gate) for fallible insertion.
    pub fn add_gate(&mut self, gate: Gate) -> usize {
        self.try_add_gate(gate)
            .unwrap_or_else(|err| panic!("Invalid gate: {err}"))
    }

    /// Try to add a validated gate to the circuit.
    ///
    /// # Errors
    ///
    /// Returns an error if [`Gate::validate`] rejects the gate payload.
    pub fn try_add_gate(&mut self, gate: Gate) -> Result<usize, String> {
        gate.validate()?;
        Ok(self.add_gate_unchecked(gate))
    }

    fn add_gate_unchecked(&mut self, gate: Gate) -> usize {
        let node_idx = self.dag.add_node();
        // Ensure gates vector is large enough
        if node_idx >= self.gates.len() {
            self.gates.resize(node_idx + 1, None);
        }
        // Update max_qubit tracking
        for q in &gate.qubits {
            self.max_qubit = self.max_qubit.max(q.index());
        }
        self.gates[node_idx] = Some(gate);
        node_idx
    }

    /// Removes a gate from the circuit.
    ///
    /// Also removes all qubit wires connected to this gate.
    ///
    /// # Returns
    ///
    /// The removed gate if it existed, or `None` otherwise.
    pub fn remove_gate(&mut self, node: usize) -> Option<Gate> {
        // Remove edge qubit mappings for edges connected to this node
        let in_edges = self.dag.in_edges(node);
        let out_edges = self.dag.out_edges(node);
        for edge_id in in_edges.iter().chain(out_edges.iter()) {
            self.edge_qubits.remove(edge_id);
        }

        self.dag.remove_node(node);
        if node < self.gates.len() {
            self.gates[node].take()
        } else {
            None
        }
    }

    /// Gets a reference to the gate at the given node index.
    #[must_use]
    pub fn gate(&self, node: usize) -> Option<&Gate> {
        self.gates.get(node).and_then(|g| g.as_ref())
    }

    /// Gets a mutable reference to the gate at the given node index.
    pub fn gate_mut(&mut self, node: usize) -> Option<&mut Gate> {
        self.gates.get_mut(node).and_then(|g| g.as_mut())
    }

    /// Returns the number of gates in the circuit.
    ///
    /// Batched gate nodes count by individual gate. For example, a node carrying
    /// `Gate::cx(&[(0, 1), (2, 3)])` contributes two gates.
    #[must_use]
    pub fn gate_count(&self) -> usize {
        self.gates.iter().flatten().map(Gate::num_gates).sum()
    }

    /// Returns the number of gate nodes stored in the DAG.
    ///
    /// A batched node carrying `Gate::cx(&[(0, 1), (2, 3)])` contributes one
    /// gate node and two gates.
    #[must_use]
    pub fn gate_node_count(&self) -> usize {
        self.dag.node_count()
    }

    /// Returns all node indices in the circuit.
    #[must_use]
    pub fn nodes(&self) -> Vec<usize> {
        self.dag.nodes()
    }

    /// Returns the maximum qubit index used in this circuit.
    ///
    /// This is tracked incrementally as gates are added, providing O(1) access.
    /// Returns 0 for empty circuits.
    #[must_use]
    #[inline]
    pub fn max_qubit(&self) -> usize {
        self.max_qubit
    }

    // ==================== Wire (edge) operations ====================

    /// Connects two gates with a qubit wire.
    ///
    /// Creates an edge from `from` to `to` representing the given qubit
    /// flowing between the gates.
    ///
    /// # Arguments
    ///
    /// * `from` - The source gate node index
    /// * `to` - The target gate node index
    /// * `qubit` - The qubit being passed along this wire
    ///
    /// # Returns
    ///
    /// The edge ID of the new wire, or an error if it would create a cycle.
    ///
    /// # Errors
    ///
    /// Returns [`DagWouldCycleError`] if adding this wire would create a cycle.
    pub fn connect(
        &mut self,
        from: usize,
        to: usize,
        qubit: QubitId,
    ) -> Result<usize, DagWouldCycleError> {
        let edge_id = self.dag.add_edge(from, to)?;
        self.edge_qubits.insert(edge_id, qubit);
        Ok(edge_id)
    }

    /// Connects two gates on all shared qubits.
    ///
    /// For each qubit that both gates act on, creates an edge from `from` to `to`.
    ///
    /// # Returns
    ///
    /// A vector of `(qubit, edge_id)` pairs for each connection made.
    ///
    /// # Errors
    ///
    /// Returns [`DagWouldCycleError`] if any connection would create a cycle.
    /// In case of error, no connections are made.
    pub fn connect_all(
        &mut self,
        from: usize,
        to: usize,
    ) -> Result<Vec<(QubitId, usize)>, DagWouldCycleError> {
        let from_qubits: BTreeSet<QubitId> = self
            .gate(from)
            .map(|g| g.qubits.iter().copied().collect())
            .unwrap_or_default();

        let to_qubits: BTreeSet<QubitId> = self
            .gate(to)
            .map(|g| g.qubits.iter().copied().collect())
            .unwrap_or_default();

        let shared: Vec<QubitId> = from_qubits.intersection(&to_qubits).copied().collect();

        let mut results = Vec::with_capacity(shared.len());
        for qubit in shared {
            let edge_id = self.connect(from, to, qubit)?;
            results.push((qubit, edge_id));
        }
        Ok(results)
    }

    /// Removes a wire (edge) by its edge ID.
    ///
    /// # Returns
    ///
    /// The qubit that was carried by this wire, or `None` if the edge didn't exist.
    pub fn remove_wire(&mut self, edge_id: usize) -> Option<QubitId> {
        self.dag.remove_edge(edge_id);
        self.edge_qubits.remove(&edge_id)
    }

    /// Returns the number of wires (edges) in the circuit.
    #[must_use]
    pub fn wire_count(&self) -> usize {
        self.dag.edge_count()
    }

    /// Returns the qubit carried by a wire.
    #[must_use]
    pub fn wire_qubit(&self, edge_id: usize) -> Option<QubitId> {
        self.edge_qubits.get(&edge_id).copied()
    }

    /// Returns all wires as (from, to, qubit) tuples.
    #[must_use]
    pub fn wires(&self) -> Vec<(usize, usize, QubitId)> {
        self.dag
            .edges()
            .into_iter()
            .filter_map(|(from, to, _weight)| {
                let edge_id = self.dag.find_edge(from, to)?;
                let qubit = self.edge_qubits.get(&edge_id)?;
                Some((from, to, *qubit))
            })
            .collect()
    }

    /// Returns incoming wires to a gate as `(edge_id, qubit)` pairs.
    #[must_use]
    pub fn incoming_wires(&self, node: usize) -> Vec<(usize, QubitId)> {
        self.dag
            .in_edges(node)
            .into_iter()
            .filter_map(|edge_id| {
                let qubit = self.edge_qubits.get(&edge_id)?;
                Some((edge_id, *qubit))
            })
            .collect()
    }

    /// Returns outgoing wires from a gate as `(edge_id, qubit)` pairs.
    #[must_use]
    pub fn outgoing_wires(&self, node: usize) -> Vec<(usize, QubitId)> {
        self.dag
            .out_edges(node)
            .into_iter()
            .filter_map(|edge_id| {
                let qubit = self.edge_qubits.get(&edge_id)?;
                Some((edge_id, *qubit))
            })
            .collect()
    }

    /// Returns the predecessor gate for a specific qubit input.
    #[must_use]
    pub fn predecessor_on_qubit(&self, node: usize, qubit: QubitId) -> Option<usize> {
        for edge_id in self.dag.in_edges(node) {
            if self.edge_qubits.get(&edge_id) == Some(&qubit) {
                return self.dag.edge_endpoints(edge_id).map(|(src, _)| src);
            }
        }
        None
    }

    /// Returns the successor gate for a specific qubit output.
    #[must_use]
    pub fn successor_on_qubit(&self, node: usize, qubit: QubitId) -> Option<usize> {
        for edge_id in self.dag.out_edges(node) {
            if self.edge_qubits.get(&edge_id) == Some(&qubit) {
                return self.dag.edge_endpoints(edge_id).map(|(_, tgt)| tgt);
            }
        }
        None
    }

    /// Returns all predecessor gates (gates with wires into this gate).
    #[must_use]
    pub fn predecessors(&self, node: usize) -> Vec<usize> {
        self.dag.predecessors(node)
    }

    /// Returns all successor gates (gates with wires from this gate).
    #[must_use]
    pub fn successors(&self, node: usize) -> Vec<usize> {
        self.dag.successors(node)
    }

    // ==================== Circuit properties ====================

    /// Returns the circuit depth (longest path from any root to any leaf).
    ///
    /// This represents the minimum number of time steps needed to execute
    /// the circuit, assuming gates on independent qubits can execute in parallel.
    #[must_use]
    pub fn depth(&self) -> usize {
        self.dag.depth()
    }

    /// Returns the circuit width (number of unique qubits used).
    #[must_use]
    pub fn width(&self) -> usize {
        self.qubits().len()
    }

    /// Returns all unique qubits used in the circuit.
    ///
    /// This includes qubits from both gates and wires.
    #[must_use]
    pub fn qubits(&self) -> Vec<QubitId> {
        let mut qubits: BTreeSet<QubitId> = self
            .gates
            .iter()
            .flatten()
            .flat_map(|g| g.qubits.iter().copied())
            .collect();

        // Also include qubits from wires
        qubits.extend(self.edge_qubits.values().copied());

        // BTreeSet is already sorted, so just collect to Vec
        qubits.into_iter().collect()
    }

    /// Returns the count of single-qubit gates.
    #[must_use]
    pub fn single_qubit_gate_count(&self) -> usize {
        self.gates
            .iter()
            .flatten()
            .filter(|g| g.is_single_qubit())
            .map(Gate::num_gates)
            .sum()
    }

    /// Returns the count of two-qubit gates.
    #[must_use]
    pub fn two_qubit_gate_count(&self) -> usize {
        self.gates
            .iter()
            .flatten()
            .filter(|g| g.is_two_qubit())
            .map(Gate::num_gates)
            .sum()
    }

    /// Returns the count of gates of a specific type.
    #[must_use]
    pub fn gate_type_count(&self, gate_type: GateType) -> usize {
        self.gates
            .iter()
            .flatten()
            .filter(|g| g.gate_type == gate_type)
            .map(Gate::num_gates)
            .sum()
    }

    // ==================== Topological operations ====================

    /// Returns gates in topological order (valid execution order).
    ///
    /// This is guaranteed to succeed since the circuit is a DAG.
    #[must_use]
    pub fn topological_order(&self) -> Vec<usize> {
        self.dag.topological_sort()
    }

    /// Returns an iterator over circuit layers.
    ///
    /// Each layer contains gates that can execute in parallel
    /// (all their dependencies are in previous layers).
    pub fn layers(&self) -> impl Iterator<Item = Vec<usize>> + '_ {
        let roots = self.dag.roots();
        self.dag.layers(roots)
    }

    /// Export as a plain ASCII circuit diagram.
    ///
    /// Uses [`layers`](Self::layers) to determine column layout.
    /// Horizontal qubit wires with gate symbols placed at each layer column.
    #[must_use]
    pub fn to_ascii(&self) -> String {
        self.render_with(&pecos_core::circuit_diagram::DiagramStyle::default())
            .ascii()
    }

    /// ASCII circuit diagram with ANSI color codes.
    ///
    /// Same layout as [`to_ascii`](Self::to_ascii) with color-coded gate
    /// categories: blue for single-qubit, green for two-qubit, yellow for
    /// measurements, cyan for preparations.
    #[must_use]
    pub fn to_color_ascii(&self) -> String {
        self.render_with(
            &pecos_core::circuit_diagram::DiagramStyle::builder()
                .ansi_color(true)
                .build(),
        )
        .ascii()
    }

    /// Unicode circuit diagram with box-drawing characters.
    #[must_use]
    pub fn to_unicode(&self) -> String {
        self.render_with(
            &pecos_core::circuit_diagram::DiagramStyle::builder()
                .symbols(pecos_core::circuit_diagram::SymbolSet::Unicode)
                .build(),
        )
        .unicode()
    }

    /// Unicode circuit diagram with ANSI color codes.
    #[must_use]
    pub fn to_color_unicode(&self) -> String {
        self.render_with(
            &pecos_core::circuit_diagram::DiagramStyle::builder()
                .symbols(pecos_core::circuit_diagram::SymbolSet::Unicode)
                .ansi_color(true)
                .build(),
        )
        .unicode()
    }

    /// Export as an SVG circuit diagram.
    #[must_use]
    pub fn to_svg(&self) -> String {
        self.render_with(&pecos_core::circuit_diagram::DiagramStyle::default())
            .svg()
    }

    /// Export as a `TikZ` `tikzpicture`.
    #[must_use]
    pub fn to_tikz(&self) -> String {
        self.render_with(&pecos_core::circuit_diagram::DiagramStyle::default())
            .tikz()
    }

    /// Export as a Graphviz DOT digraph.
    #[must_use]
    pub fn to_dot(&self) -> String {
        self.render_with(&pecos_core::circuit_diagram::DiagramStyle::default())
            .dot()
    }

    /// Create a [`DiagramRenderer`](pecos_core::circuit_diagram::DiagramRenderer)
    /// bound to a custom [`DiagramStyle`](pecos_core::circuit_diagram::DiagramStyle).
    #[must_use]
    pub fn render_with<'a>(
        &self,
        style: &'a pecos_core::circuit_diagram::DiagramStyle,
    ) -> pecos_core::circuit_diagram::DiagramRenderer<'a> {
        let (header, layers) = self.diagram_parts();
        let diagram = crate::circuit_display::build_diagram_or_empty(&layers, style.angle_unit);
        pecos_core::circuit_diagram::DiagramRenderer::new(diagram, header, style)
    }

    fn diagram_parts(&self) -> (String, Vec<Vec<&Gate>>) {
        let layers: Vec<Vec<&Gate>> = self
            .layers()
            .map(|node_ids| node_ids.iter().filter_map(|&id| self.gate(id)).collect())
            .collect();
        let num_qubits = self.qubits().len();
        let num_layers = layers.len();
        let header = format!(
            "DagCircuit: {} qubit{}, {} layer{}",
            num_qubits,
            if num_qubits == 1 { "" } else { "s" },
            num_layers,
            if num_layers == 1 { "" } else { "s" },
        );
        (header, layers)
    }

    /// Returns the root gates (gates with no incoming wires).
    #[must_use]
    pub fn roots(&self) -> Vec<usize> {
        self.dag.roots()
    }

    /// Returns the leaf gates (gates with no outgoing wires).
    #[must_use]
    pub fn leaves(&self) -> Vec<usize> {
        self.dag.leaves()
    }

    /// Builds a pre-computed traversal index for efficient iteration.
    ///
    /// This creates cached data structures that enable O(1) lookups for
    /// topological positions and per-qubit gate lists. Build once and
    /// reuse for multiple traversal operations.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::DagCircuit;
    ///
    /// let mut dag = DagCircuit::new();
    /// dag.h(&[0]);
    /// dag.cx(&[(0, 1)]);
    ///
    /// let index = dag.build_traversal_index();
    /// // O(1) position lookup
    /// let h_node = index.topo_order()[0];
    /// assert_eq!(index.topo_position(h_node), 0);
    /// ```
    #[must_use]
    pub fn build_traversal_index(&self) -> DagTraversalIndex {
        let topo_order = self.topological_order();
        let max_node = topo_order.iter().copied().max().unwrap_or(0);
        let max_qubit = self.max_qubit;

        // Build position lookup (node -> topo position) for O(1) access
        let mut topo_positions = vec![usize::MAX; max_node + 1];
        for (pos, &node) in topo_order.iter().enumerate() {
            topo_positions[node] = pos;
        }

        // Build per-qubit gate index
        let mut qubit_gates: Vec<Vec<(usize, usize)>> = vec![Vec::new(); max_qubit + 1];
        for (topo_pos, &node) in topo_order.iter().enumerate() {
            if let Some(gate) = self.gate(node) {
                for q in &gate.qubits {
                    qubit_gates[q.index()].push((topo_pos, node));
                }
            }
        }

        DagTraversalIndex {
            topo_order,
            topo_positions,
            qubit_gates,
            max_node,
            max_qubit,
        }
    }

    // ==================== Qubit-based queries ====================

    /// Returns all gates acting on a specific qubit.
    #[must_use]
    pub fn gates_on_qubit(&self, qubit: QubitId) -> Vec<usize> {
        self.dag
            .nodes()
            .into_iter()
            .filter(|&node| self.gate(node).is_some_and(|g| g.qubits.contains(&qubit)))
            .collect()
    }

    /// Returns gates acting on a specific qubit in topological order.
    ///
    /// This follows the qubit wire through the circuit.
    #[must_use]
    pub fn qubit_timeline(&self, qubit: QubitId) -> Vec<usize> {
        let mut gates = self.gates_on_qubit(qubit);
        let topo_order = self.topological_order();

        // Create position map for sorting
        let mut positions = vec![usize::MAX; self.gates.len()];
        for (pos, &node) in topo_order.iter().enumerate() {
            if node < positions.len() {
                positions[node] = pos;
            }
        }

        gates.sort_by_key(|&node| positions.get(node).copied().unwrap_or(usize::MAX));
        gates
    }

    /// Returns all wires carrying a specific qubit.
    #[must_use]
    pub fn wires_for_qubit(&self, qubit: QubitId) -> Vec<usize> {
        self.edge_qubits
            .iter()
            .filter_map(|(&edge_id, &q)| if q == qubit { Some(edge_id) } else { None })
            .collect()
    }

    // ==================== Iteration ====================

    /// Returns an iterator over all gates in the circuit.
    pub fn iter_gates(&self) -> impl Iterator<Item = (usize, &Gate)> {
        self.dag
            .nodes()
            .into_iter()
            .filter_map(|node| self.gate(node).map(|g| (node, g)))
    }

    /// Returns an iterator over gates in topological order.
    pub fn iter_gates_topo(&self) -> impl Iterator<Item = (usize, &Gate)> {
        self.topological_order()
            .into_iter()
            .filter_map(|node| self.gate(node).map(|g| (node, g)))
    }

    // ==================== Builder-style gate methods ====================
    //
    // These methods provide a simulator-like API for building circuits.
    // They automatically wire gates together based on qubit identity and
    // support method chaining.
    //
    // The API follows the same conventions as the simulator traits
    // (CliffordGateable, ArbitraryRotationGateable):
    // - Rotation gates take angle first, then qubit: `rx(theta, q)`
    // - Two-qubit rotations: `rzz(theta, q1, q2)`
    // - All methods return `&mut Self` for chaining

    /// Adds a gate and auto-wires it to previous gates on the same qubits.
    pub fn add_gate_auto_wire(&mut self, gate: Gate) -> usize {
        let qubits = gate.qubits.clone();
        let node = self.add_gate(gate);

        // Connect to previous gates on each qubit
        for qubit in &qubits {
            if let Some(&prev_node) = self.qubit_heads.get(qubit) {
                // Connect previous gate to this gate on this qubit
                let _ = self.connect(prev_node, node, *qubit);
            }
            // Update the head for this qubit
            self.qubit_heads.insert(*qubit, node);
        }

        // Track last added node for .meta() calls
        self.last_node = Some(node);

        node
    }

    /// Add metadata to the last added gate.
    ///
    /// This allows attaching attributes to gates in a chainable way:
    /// ```
    /// use pecos_quantum::{DagCircuit, Attribute};
    ///
    /// let mut circuit = DagCircuit::new();
    /// circuit.h(&[0]).meta("error_rate", Attribute::Float(0.01)).cx(&[(0, 1)]);
    /// circuit.mz(&[0]);
    /// circuit.meta("basis", Attribute::String("Z".into()));
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if called before any gate has been added.
    pub fn meta(&mut self, key: &str, value: impl Into<Attribute>) -> &mut Self {
        let node = self
            .last_node
            .expect("meta() called before any gate was added");
        self.set_gate_attr(node, key, value.into());
        self
    }

    /// Add multiple metadata attributes to the last added gate.
    ///
    /// This allows attaching multiple attributes at once in a chainable way:
    /// ```
    /// use pecos_quantum::{DagCircuit, Attribute};
    /// use std::collections::BTreeMap;
    ///
    /// let mut circuit = DagCircuit::new();
    /// let attrs = BTreeMap::from([
    ///     ("duration".to_string(), Attribute::Float(50.0)),
    ///     ("error_rate".to_string(), Attribute::Float(0.001)),
    /// ]);
    /// circuit.h(&[0]).metas(attrs).cx(&[(0, 1)]);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics if called before any gate has been added.
    pub fn metas(&mut self, attrs: BTreeMap<String, Attribute>) -> &mut Self {
        let node = self
            .last_node
            .expect("metas() called before any gate was added");
        self.set_gate_attrs(node, attrs);
        self
    }

    /// Returns the node index of the last added gate, if any.
    #[must_use]
    pub fn last_added_node(&self) -> Option<usize> {
        self.last_node
    }

    // -------------------- Single-qubit Clifford gates --------------------

    /// Apply identity gate(s).
    pub fn identity(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::simple(GateType::I, vec![q.into()]));
        }
        self
    }

    /// Alias for `identity`.
    pub fn iden(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.identity(qubits)
    }

    /// Apply X (Pauli-X) gate(s).
    pub fn x(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::x(&[q]));
        }
        self
    }

    /// Apply Y (Pauli-Y) gate(s).
    pub fn y(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::y(&[q]));
        }
        self
    }

    /// Apply Z (Pauli-Z) gate(s).
    pub fn z(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::z(&[q]));
        }
        self
    }

    /// Apply Hadamard gate(s).
    pub fn h(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::h(&[q]));
        }
        self
    }

    /// Apply SZ (sqrt(Z), S gate) gate(s).
    pub fn sz(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::simple(GateType::SZ, vec![q.into()]));
        }
        self
    }

    /// Apply SZ-dagger (S-dagger) gate(s).
    pub fn szdg(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::simple(GateType::SZdg, vec![q.into()]));
        }
        self
    }

    /// Apply SX (sqrt(X)) gate(s).
    pub fn sx(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::simple(GateType::SX, vec![q.into()]));
        }
        self
    }

    /// Apply SX-dagger (sqrt(X) inverse) gate(s).
    pub fn sxdg(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::simple(GateType::SXdg, vec![q.into()]));
        }
        self
    }

    /// Apply SY (sqrt(Y)) gate(s).
    pub fn sy(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::simple(GateType::SY, vec![q.into()]));
        }
        self
    }

    /// Apply SY-dagger (sqrt(Y) inverse) gate(s).
    pub fn sydg(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::simple(GateType::SYdg, vec![q.into()]));
        }
        self
    }

    /// Apply T gate(s).
    pub fn t(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::simple(GateType::T, vec![q.into()]));
        }
        self
    }

    /// Apply T-dagger gate(s).
    pub fn tdg(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::simple(GateType::Tdg, vec![q.into()]));
        }
        self
    }

    // -------------------- Single-qubit rotation gates --------------------

    /// Apply RX (rotation about X) gate(s).
    pub fn rx(
        &mut self,
        theta: impl Into<Angle64>,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> &mut Self {
        let angle = theta.into();
        for &q in qubits {
            self.add_gate_auto_wire(Gate::rx(angle, &[q]));
        }
        self
    }

    /// Apply RY (rotation about Y) gate(s).
    pub fn ry(
        &mut self,
        theta: impl Into<Angle64>,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> &mut Self {
        let angle = theta.into();
        for &q in qubits {
            self.add_gate_auto_wire(Gate::ry(angle, &[q]));
        }
        self
    }

    /// Apply RZ (rotation about Z) gate(s).
    pub fn rz(
        &mut self,
        theta: impl Into<Angle64>,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> &mut Self {
        let angle = theta.into();
        for &q in qubits {
            self.add_gate_auto_wire(Gate::rz(angle, &[q]));
        }
        self
    }

    /// Apply general single-qubit unitary U(theta, phi, lambda) gate(s).
    pub fn u(
        &mut self,
        theta: impl Into<Angle64>,
        phi: impl Into<Angle64>,
        lambda: impl Into<Angle64>,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> &mut Self {
        let t = theta.into();
        let p = phi.into();
        let l = lambda.into();
        for &q in qubits {
            self.add_gate_auto_wire(Gate::with_angles(
                GateType::U,
                vec![t, p, l],
                vec![q.into()],
            ));
        }
        self
    }

    /// Apply R1XY (X-Y plane rotation) gate(s).
    pub fn r1xy(
        &mut self,
        theta: impl Into<Angle64>,
        phi: impl Into<Angle64>,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> &mut Self {
        let t = theta.into();
        let p = phi.into();
        for &q in qubits {
            self.add_gate_auto_wire(Gate::with_angles(
                GateType::R1XY,
                vec![t, p],
                vec![q.into()],
            ));
        }
        self
    }

    // -------------------- Two-qubit gates --------------------

    /// Apply a CX (CNOT) gate.
    ///
    /// The first qubit is the control, the second is the target.
    /// Flips the target qubit if the control is |1>.
    pub fn cx(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        for &(c, t) in pairs {
            self.add_gate_auto_wire(Gate::cx(&[(c, t)]));
        }
        self
    }

    /// Apply CY (controlled-Y) gate(s).
    ///
    /// The first element of each pair is the control, the second is the target.
    pub fn cy(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        for &(c, t) in pairs {
            self.add_gate_auto_wire(Gate::cy(&[(c, t)]));
        }
        self
    }

    /// Apply CZ (controlled-Z) gate(s).
    ///
    /// Applies a phase flip when both qubits are |1>. This gate is symmetric.
    pub fn cz(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        for &(q1, q2) in pairs {
            self.add_gate_auto_wire(Gate::cz(&[(q1, q2)]));
        }
        self
    }

    /// Apply SZZ (sqrt(ZZ)) gate(s).
    ///
    /// Native entangling gate on some trapped-ion systems.
    pub fn szz(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        for &(q1, q2) in pairs {
            self.add_gate_auto_wire(Gate::szz(&[(q1, q2)]));
        }
        self
    }

    /// Apply SZZ-dagger (sqrt(ZZ) inverse) gate(s).
    pub fn szzdg(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        for &(q1, q2) in pairs {
            self.add_gate_auto_wire(Gate::szzdg(&[(q1, q2)]));
        }
        self
    }

    /// Apply SXX (sqrt(XX)) gate(s).
    pub fn sxx(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        for &(q1, q2) in pairs {
            self.add_gate_auto_wire(Gate::sxx(&[(q1, q2)]));
        }
        self
    }

    /// Apply SXX-dagger (sqrt(XX) inverse) gate(s).
    pub fn sxxdg(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        for &(q1, q2) in pairs {
            self.add_gate_auto_wire(Gate::sxxdg(&[(q1, q2)]));
        }
        self
    }

    /// Apply SYY (sqrt(YY)) gate(s).
    pub fn syy(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        for &(q1, q2) in pairs {
            self.add_gate_auto_wire(Gate::syy(&[(q1, q2)]));
        }
        self
    }

    /// Apply SYY-dagger (sqrt(YY) inverse) gate(s).
    pub fn syydg(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        for &(q1, q2) in pairs {
            self.add_gate_auto_wire(Gate::syydg(&[(q1, q2)]));
        }
        self
    }

    /// Apply RZZ (ZZ rotation) gate(s).
    ///
    /// Implements exp(-i * theta/2 * Z*Z). The angle can be `Angle64` or `f64` (radians).
    pub fn rzz(
        &mut self,
        theta: impl Into<Angle64>,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        let angle = theta.into();
        for &(q1, q2) in pairs {
            self.add_gate_auto_wire(Gate::rzz(angle, &[(q1, q2)]));
        }
        self
    }

    /// Apply RXX (XX rotation) gate(s).
    ///
    /// Implements exp(-i * theta/2 * X*X). Native gate on trapped-ion systems.
    pub fn rxx(
        &mut self,
        theta: impl Into<Angle64>,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        let angle = theta.into();
        for &(q1, q2) in pairs {
            self.add_gate_auto_wire(Gate::rxx(angle, &[(q1, q2)]));
        }
        self
    }

    /// Apply RYY (YY rotation) gate(s).
    ///
    /// Implements exp(-i * theta/2 * Y*Y).
    pub fn ryy(
        &mut self,
        theta: impl Into<Angle64>,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        let angle = theta.into();
        for &(q1, q2) in pairs {
            self.add_gate_auto_wire(Gate::ryy(angle, &[(q1, q2)]));
        }
        self
    }

    /// Apply SWAP gate(s).
    ///
    /// Exchanges the states of two qubits.
    pub fn swap(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        for &(q1, q2) in pairs {
            self.add_gate_auto_wire(Gate::swap(&[(q1, q2)]));
        }
        self
    }

    /// Apply CRZ (controlled-RZ) gate(s).
    ///
    /// The angle can be an `Angle64` or an `f64` (interpreted as radians).
    pub fn crz(
        &mut self,
        theta: impl Into<Angle64>,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        let angle = theta.into();
        for &(c, t) in pairs {
            self.add_gate_auto_wire(Gate::with_angles(
                GateType::CRZ,
                vec![angle],
                vec![c.into(), t.into()],
            ));
        }
        self
    }

    // -------------------- Three-qubit gates --------------------

    /// Apply a CCX (Toffoli) gate.
    ///
    /// The first two qubits are controls, the third is the target.
    pub fn ccx(
        &mut self,
        c1: impl Into<QubitId>,
        c2: impl Into<QubitId>,
        target: impl Into<QubitId>,
    ) -> &mut Self {
        self.add_gate_auto_wire(Gate::simple(
            GateType::CCX,
            vec![c1.into(), c2.into(), target.into()],
        ));
        self
    }

    // -------------------- Idle --------------------

    /// Apply an idle gate with a specified duration in abstract time units.
    ///
    /// Idle gates represent waiting time on a qubit, useful for noise modeling.
    /// The interpretation of time units (nanoseconds, clock cycles, etc.) is
    /// defined by your noise model or timing configuration.
    ///
    /// Accepts `TimeUnits` or `u64`.
    ///
    /// # Example
    /// ```
    /// use pecos_quantum::DagCircuit;
    /// use pecos_core::TimeUnits;
    ///
    /// let mut circuit = DagCircuit::new();
    /// circuit.idle(TimeUnits::new(100), &[0]);
    /// circuit.idle(100u64, &[0, 1]);  // idle on two qubits
    /// ```
    pub fn idle(
        &mut self,
        duration: impl Into<TimeUnits>,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> &mut Self {
        let units: TimeUnits = duration.into();
        for &q in qubits {
            self.add_gate_auto_wire(Gate::idle(units.as_f64(), vec![q.into()]));
        }
        self
    }

    // -------------------- Measurement and preparation --------------------
    //
    // Measurements return Vec<MeasRef> (lightweight Copy handles).
    // Preparations return &mut Self and are chainable.

    /// Measure qubit(s) in the Z basis.
    ///
    /// Each qubit becomes a separate measurement node in the DAG.
    ///
    /// # Example
    /// ```
    /// use pecos_quantum::DagCircuit;
    ///
    /// let mut circuit = DagCircuit::new();
    /// circuit.h(&[0]);
    /// let nodes = circuit.mz(&[0, 1]);
    /// assert_eq!(nodes.len(), 2);
    /// ```
    pub fn mz(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> Vec<MeasRef> {
        qubits
            .iter()
            .map(|&q| {
                let qubit = q.into();
                let node = self.add_gate_auto_wire(Gate::mz(&[qubit]));
                MeasRef { node, qubit }
            })
            .collect()
    }

    /// Measure qubits and label them.
    ///
    /// Labels are stored on the circuit and flow through to the sampler output.
    pub fn mz_labeled(&mut self, entries: &[(impl Into<QubitId> + Copy, &str)]) -> Vec<MeasRef> {
        entries
            .iter()
            .map(|&(q, label)| {
                let qubit = q.into();
                let node = self.add_gate_auto_wire(Gate::mz(&[qubit]));
                let mref = MeasRef { node, qubit };
                self.set_measurement_label(node, label);
                mref
            })
            .collect()
    }

    /// Set a label on a measurement node.
    pub fn set_measurement_label(&mut self, node: usize, label: &str) {
        self.measurement_labels.insert(node, label.to_string());
    }

    /// Get the label of a measurement node, if any.
    #[must_use]
    pub fn measurement_label(&self, node: usize) -> Option<&str> {
        self.measurement_labels.get(&node).map(String::as_str)
    }

    /// Measure and free qubit(s) (destructive measurement).
    pub fn mz_free(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::mz_free(&[q]));
        }
        self
    }

    // ========================================================================
    // Detector and Observable Annotations
    // ========================================================================

    /// Annotate a detector: a set of measurements whose XOR should be
    /// deterministic in the noiseless case.
    ///
    /// The Pauli string is automatically Z on the measured qubits.
    ///
    /// Returns the annotation index.
    pub fn detector(&mut self, measurements: &[impl Into<usize> + Copy]) -> usize {
        let meas_nodes: Vec<usize> = measurements.iter().map(|&m| m.into()).collect();
        let pauli = self.pauli_from_measurement_nodes(&meas_nodes);
        let idx = self.annotations.len();
        self.annotations.push(PauliAnnotation {
            pauli,
            kind: AnnotationKind::Detector {
                measurement_nodes: meas_nodes,
                coords: Vec::new(),
            },
            label: None,
        });
        idx
    }

    /// Annotate a labeled detector.
    pub fn detector_labeled(
        &mut self,
        label: &str,
        measurements: &[impl Into<usize> + Copy],
    ) -> usize {
        let meas_nodes: Vec<usize> = measurements.iter().map(|&m| m.into()).collect();
        let pauli = self.pauli_from_measurement_nodes(&meas_nodes);
        let idx = self.annotations.len();
        self.annotations.push(PauliAnnotation {
            pauli,
            kind: AnnotationKind::Detector {
                measurement_nodes: meas_nodes,
                coords: Vec::new(),
            },
            label: Some(label.to_string()),
        });
        idx
    }

    /// Annotate a detector with coordinates.
    pub fn detector_with_coords(
        &mut self,
        measurements: &[impl Into<usize> + Copy],
        coords: &[f64],
    ) -> usize {
        let meas_nodes: Vec<usize> = measurements.iter().map(|&m| m.into()).collect();
        let pauli = self.pauli_from_measurement_nodes(&meas_nodes);
        let idx = self.annotations.len();
        self.annotations.push(PauliAnnotation {
            pauli,
            kind: AnnotationKind::Detector {
                measurement_nodes: meas_nodes,
                coords: coords.to_vec(),
            },
            label: None,
        });
        idx
    }

    /// Annotate a logical observable: a set of measurements whose XOR
    /// defines whether a logical operator flipped.
    ///
    /// The Pauli string is automatically Z on the measured qubits.
    ///
    /// Returns the annotation index.
    pub fn observable(&mut self, measurements: &[impl Into<usize> + Copy]) -> usize {
        let meas_nodes: Vec<usize> = measurements.iter().map(|&m| m.into()).collect();
        let pauli = self.pauli_from_measurement_nodes(&meas_nodes);
        let idx = self.annotations.len();
        self.annotations.push(PauliAnnotation {
            pauli,
            kind: AnnotationKind::Observable {
                measurement_nodes: meas_nodes,
            },
            label: None,
        });
        idx
    }

    /// Annotate a labeled observable.
    pub fn observable_labeled(
        &mut self,
        label: &str,
        measurements: &[impl Into<usize> + Copy],
    ) -> usize {
        let meas_nodes: Vec<usize> = measurements.iter().map(|&m| m.into()).collect();
        let pauli = self.pauli_from_measurement_nodes(&meas_nodes);
        let idx = self.annotations.len();
        self.annotations.push(PauliAnnotation {
            pauli,
            kind: AnnotationKind::Observable {
                measurement_nodes: meas_nodes,
            },
            label: Some(label.to_string()),
        });
        idx
    }

    /// Derive a `PauliString` from measurement nodes.
    /// Z-basis measurements → Z on the measured qubit.
    fn pauli_from_measurement_nodes(&self, nodes: &[usize]) -> pecos_core::PauliString {
        let qubits: Vec<usize> = nodes
            .iter()
            .filter_map(|&node| {
                let gate = self.gate(node)?;
                Some(gate.qubits.iter().map(pecos_core::QubitId::index))
            })
            .flatten()
            .collect();
        pecos_core::PauliString::zs(&qubits)
    }

    /// Place a tracked-Pauli meta-gate at this point in the circuit.
    ///
    /// This is a **positional** annotation: only faults BEFORE this node
    /// can flip the tracked Pauli. The meta-gate does not affect quantum state
    /// -- simulators ignore it.
    ///
    /// Accepts a [`PauliString`](pecos_core::PauliString), which supports
    /// the `X(q) & Y(q) & Z(q)` composition syntax.
    ///
    /// Returns the annotation index.
    ///
    /// # Example
    /// ```
    /// use pecos_quantum::DagCircuit;
    /// use pecos_core::pauli::{X, Z};
    ///
    /// let mut c = DagCircuit::new();
    /// c.pz(&[0, 1, 2]);
    /// c.cx(&[(0, 1)]);
    /// // Place X_0 & Z_1 & Z_2 check HERE -- only faults above can flip it
    /// c.tracked_pauli(X(0) & Z(1) & Z(2));
    /// c.cx(&[(1, 2)]);  // faults here don't affect the check
    /// ```
    pub fn tracked_pauli(&mut self, mut pauli: pecos_core::PauliString) -> usize {
        // Phase is irrelevant for flip tracking -- normalize to +1
        pauli.set_phase(pecos_core::QuarterPhase::PlusOne);
        let idx = self.annotations.len();
        self.insert_pauli_meta_gate(&pauli);
        self.annotations.push(PauliAnnotation {
            pauli,
            kind: AnnotationKind::TrackedPauli,
            label: None,
        });
        idx
    }

    /// Place a labeled tracked-Pauli meta-gate.
    pub fn tracked_pauli_labeled(
        &mut self,
        label: &str,
        mut pauli: pecos_core::PauliString,
    ) -> usize {
        pauli.set_phase(pecos_core::QuarterPhase::PlusOne);
        let idx = self.annotations.len();
        self.insert_pauli_meta_gate(&pauli);
        self.annotations.push(PauliAnnotation {
            pauli,
            kind: AnnotationKind::TrackedPauli,
            label: Some(label.to_string()),
        });
        idx
    }

    /// Insert a `TrackedPauliMeta` gate node into the DAG.
    fn insert_pauli_meta_gate(&mut self, pauli: &pecos_core::PauliString) {
        let qubits: Vec<QubitId> = pauli.qubits().into_iter().map(QubitId::from).collect();
        let gate = Gate::simple(GateType::TrackedPauliMeta, qubits);
        self.add_gate_auto_wire(gate);
    }

    /// Get all annotations.
    #[must_use]
    pub fn annotations(&self) -> &[PauliAnnotation] {
        &self.annotations
    }

    /// Add a pre-built annotation (used for conversion from `TickCircuit`).
    pub fn add_annotation(&mut self, ann: PauliAnnotation) {
        // For tracked-Pauli annotations, insert the meta-gate node.
        if matches!(ann.kind, AnnotationKind::TrackedPauli) {
            self.insert_pauli_meta_gate(&ann.pauli);
        }
        self.annotations.push(ann);
    }

    /// Get detector annotations.
    pub fn detectors(&self) -> impl Iterator<Item = &PauliAnnotation> {
        self.annotations
            .iter()
            .filter(|a| matches!(a.kind, AnnotationKind::Detector { .. }))
    }

    /// Get observable annotations.
    pub fn observables(&self) -> impl Iterator<Item = &PauliAnnotation> {
        self.annotations
            .iter()
            .filter(|a| matches!(a.kind, AnnotationKind::Observable { .. }))
    }

    /// Get tracked-Pauli annotations.
    pub fn tracked_paulis(&self) -> impl Iterator<Item = &PauliAnnotation> {
        self.annotations
            .iter()
            .filter(|a| matches!(a.kind, AnnotationKind::TrackedPauli))
    }

    // ========================================================================
    // Preparation Gates
    // ========================================================================

    /// Prepare qubit(s) in the |0> state (Z-basis preparation).
    pub fn pz(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::simple(GateType::PZ, vec![q.into()]));
        }
        self
    }

    /// Allocate qubit(s) in the |0> state.
    pub fn qalloc(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::qalloc(&[q]));
        }
        self
    }

    /// Free/deallocate qubit(s).
    pub fn qfree(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        for &q in qubits {
            self.add_gate_auto_wire(Gate::qfree(&[q]));
        }
        self
    }

    // ==================== DAG access ====================

    /// Provides direct access to the underlying DAG.
    #[must_use]
    pub fn as_dag(&self) -> &DAG {
        &self.dag
    }

    /// Provides mutable access to the underlying DAG.
    ///
    /// # Warning
    ///
    /// Modifying the DAG directly can break invariants if gates and
    /// `edge_qubits` are not kept in sync. Use with caution.
    pub fn as_dag_mut(&mut self) -> &mut DAG {
        &mut self.dag
    }

    // ==================== Attributes ====================

    /// Returns a reference to the circuit-level (graph-level) attributes.
    ///
    /// These are attributes that apply to the circuit as a whole,
    /// such as metadata about the source program or compilation options.
    #[must_use]
    pub fn attrs(&self) -> &BTreeMap<String, Attribute> {
        self.dag.attrs()
    }

    /// Returns a mutable reference to the circuit-level attributes.
    pub fn attrs_mut(&mut self) -> &mut BTreeMap<String, Attribute> {
        self.dag.attrs_mut()
    }

    /// Returns a reference to attributes on a specific gate (node).
    ///
    /// Gate attributes can store per-gate metadata like rotation angles,
    /// error rates, or timing information.
    #[must_use]
    pub fn gate_attrs(&self, node: usize) -> Option<&BTreeMap<String, Attribute>> {
        self.dag.node_attrs(node)
    }

    /// Returns a mutable reference to attributes on a specific gate.
    pub fn gate_attrs_mut(&mut self, node: usize) -> Option<&mut BTreeMap<String, Attribute>> {
        self.dag.node_attrs_mut(node)
    }

    /// Returns a reference to attributes on a specific wire (edge) by edge ID.
    ///
    /// Wire attributes can store per-wire metadata like error channels
    /// or timing constraints.
    #[must_use]
    pub fn wire_attrs(&self, edge_id: usize) -> Option<&BTreeMap<String, Attribute>> {
        self.dag.edge_attrs_by_id(edge_id)
    }

    /// Returns a mutable reference to attributes on a specific wire by edge ID.
    pub fn wire_attrs_mut(&mut self, edge_id: usize) -> Option<&mut BTreeMap<String, Attribute>> {
        self.dag.edge_attrs_by_id_mut(edge_id)
    }

    /// Returns a reference to attributes on a wire between two gates.
    #[must_use]
    pub fn wire_attrs_between(
        &self,
        from: usize,
        to: usize,
    ) -> Option<&BTreeMap<String, Attribute>> {
        self.dag.edge_attrs(from, to)
    }

    /// Returns a mutable reference to attributes on a wire between two gates.
    pub fn wire_attrs_between_mut(
        &mut self,
        from: usize,
        to: usize,
    ) -> Option<&mut BTreeMap<String, Attribute>> {
        self.dag.edge_attrs_mut(from, to)
    }

    /// Sets a circuit-level attribute.
    pub fn set_attr(&mut self, key: impl Into<String>, value: Attribute) {
        self.attrs_mut().insert(key.into(), value);
    }

    /// Sets multiple circuit-level attributes at once.
    pub fn set_attrs(&mut self, attrs: BTreeMap<String, Attribute>) {
        self.attrs_mut().extend(attrs);
    }

    /// Gets a circuit-level attribute by key.
    #[must_use]
    pub fn get_attr(&self, key: &str) -> Option<&Attribute> {
        self.attrs().get(key)
    }

    /// Sets an attribute on a specific gate.
    ///
    /// Returns `true` if the gate exists, `false` otherwise.
    pub fn set_gate_attr(&mut self, node: usize, key: impl Into<String>, value: Attribute) -> bool {
        if let Some(attrs) = self.gate_attrs_mut(node) {
            attrs.insert(key.into(), value);
            true
        } else {
            false
        }
    }

    /// Sets multiple attributes on a specific gate at once.
    ///
    /// Returns `true` if the gate exists, `false` otherwise.
    pub fn set_gate_attrs(&mut self, node: usize, attrs: BTreeMap<String, Attribute>) -> bool {
        if let Some(gate_attrs) = self.gate_attrs_mut(node) {
            gate_attrs.extend(attrs);
            true
        } else {
            false
        }
    }

    /// Gets an attribute from a specific gate.
    #[must_use]
    pub fn get_gate_attr(&self, node: usize, key: &str) -> Option<&Attribute> {
        self.gate_attrs(node).and_then(|attrs| attrs.get(key))
    }

    /// Sets an attribute on a specific wire.
    ///
    /// Returns `true` if the wire exists, `false` otherwise.
    pub fn set_wire_attr(
        &mut self,
        edge_id: usize,
        key: impl Into<String>,
        value: Attribute,
    ) -> bool {
        if let Some(attrs) = self.wire_attrs_mut(edge_id) {
            attrs.insert(key.into(), value);
            true
        } else {
            false
        }
    }

    /// Gets an attribute from a specific wire.
    #[must_use]
    pub fn get_wire_attr(&self, edge_id: usize, key: &str) -> Option<&Attribute> {
        self.wire_attrs(edge_id).and_then(|attrs| attrs.get(key))
    }
}

impl Default for DagCircuit {
    fn default() -> Self {
        Self::new()
    }
}

// ==================== Circuit trait implementation ====================

impl Circuit for DagCircuit {
    fn gate_count(&self) -> usize {
        self.gate_count()
    }

    fn wire_count(&self) -> usize {
        self.wire_count()
    }

    fn qubits(&self) -> Vec<QubitId> {
        self.qubits()
    }

    fn depth(&self) -> usize {
        self.depth()
    }

    fn gate(&self, index: GateHandle) -> Option<&Gate> {
        self.gate(index)
    }

    fn nodes(&self) -> Vec<GateHandle> {
        self.nodes()
    }

    fn iter_gates(&self) -> Box<dyn Iterator<Item = GateView<'_>> + '_> {
        Box::new(self.dag.nodes().into_iter().filter_map(|node| {
            self.gate(node).map(|g| GateView {
                gate: g,
                index: node,
            })
        }))
    }

    fn topological_order(&self) -> Vec<GateHandle> {
        self.topological_order()
    }

    fn iter_gates_topo(&self) -> Box<dyn Iterator<Item = GateView<'_>> + '_> {
        Box::new(self.topological_order().into_iter().filter_map(|node| {
            self.gate(node).map(|g| GateView {
                gate: g,
                index: node,
            })
        }))
    }

    fn predecessors(&self, gate: GateHandle) -> Vec<GateHandle> {
        self.predecessors(gate)
    }

    fn successors(&self, gate: GateHandle) -> Vec<GateHandle> {
        self.successors(gate)
    }

    fn roots(&self) -> Vec<GateHandle> {
        self.roots()
    }

    fn leaves(&self) -> Vec<GateHandle> {
        self.leaves()
    }

    fn gates_on_qubit(&self, qubit: QubitId) -> Vec<GateHandle> {
        self.gates_on_qubit(qubit)
    }

    fn qubit_timeline(&self, qubit: QubitId) -> Vec<GateHandle> {
        self.qubit_timeline(qubit)
    }

    fn circuit_attrs(&self) -> &BTreeMap<String, Attribute> {
        self.attrs()
    }

    fn gate_attrs(&self, gate: GateHandle) -> Option<&BTreeMap<String, Attribute>> {
        self.gate_attrs(gate)
    }
}

impl CircuitMut for DagCircuit {
    fn add_gate(&mut self, gate: Gate) -> GateHandle {
        DagCircuit::add_gate(self, gate)
    }

    fn remove_gate(&mut self, gate: GateHandle) -> Option<Gate> {
        DagCircuit::remove_gate(self, gate)
    }

    fn set_circuit_attr(&mut self, key: impl Into<String>, value: Attribute) {
        self.set_attr(key, value);
    }

    fn set_circuit_attrs(&mut self, attrs: BTreeMap<String, Attribute>) {
        self.set_attrs(attrs);
    }

    fn set_gate_attr(
        &mut self,
        gate: GateHandle,
        key: impl Into<String>,
        value: Attribute,
    ) -> bool {
        DagCircuit::set_gate_attr(self, gate, key.into(), value)
    }

    fn set_gate_attrs(&mut self, gate: GateHandle, attrs: BTreeMap<String, Attribute>) -> bool {
        DagCircuit::set_gate_attrs(self, gate, attrs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::Angle64;

    #[test]
    fn test_empty_circuit() {
        let circuit = DagCircuit::new();
        assert_eq!(circuit.gate_count(), 0);
        assert_eq!(circuit.wire_count(), 0);
        assert_eq!(circuit.depth(), 0);
        assert_eq!(circuit.width(), 0);
    }

    #[test]
    fn test_add_single_gate() {
        let mut circuit = DagCircuit::new();
        let h = circuit.add_gate(Gate::h(&[0]));

        assert_eq!(circuit.gate_count(), 1);
        assert!(circuit.gate(h).is_some());
        assert_eq!(circuit.gate(h).unwrap().gate_type, GateType::H);
        assert_eq!(circuit.wire_count(), 0); // No connections yet
    }

    #[test]
    fn test_connect_gates() {
        let mut circuit = DagCircuit::new();

        let h = circuit.add_gate(Gate::h(&[0]));
        let t = circuit.add_gate(Gate::new(
            GateType::T,
            vec![],
            vec![],
            vec![QubitId::from(0)],
        ));

        let edge_id = circuit.connect(h, t, QubitId::from(0)).unwrap();

        assert_eq!(circuit.wire_count(), 1);
        assert_eq!(circuit.wire_qubit(edge_id), Some(QubitId::from(0)));
    }

    #[test]
    fn test_bell_state_circuit() {
        let mut circuit = DagCircuit::new();

        let h = circuit.add_gate(Gate::h(&[0]));
        let cx = circuit.add_gate(Gate::cx(&[(0, 1)]));

        // Connect H to CX on qubit 0
        circuit.connect(h, cx, QubitId::from(0)).unwrap();

        assert_eq!(circuit.gate_count(), 2);
        assert_eq!(circuit.wire_count(), 1);
        assert_eq!(circuit.depth(), 1);
        assert_eq!(circuit.width(), 2);
        assert_eq!(circuit.single_qubit_gate_count(), 1);
        assert_eq!(circuit.two_qubit_gate_count(), 1);
    }

    #[test]
    fn test_batched_gate_nodes_count_gates() {
        let mut circuit = DagCircuit::new();

        circuit.add_gate(Gate::h(&[0, 1, 2, 3]));
        circuit.add_gate(Gate::cx(&[(0, 1), (2, 3)]));

        assert_eq!(circuit.nodes().len(), 2);
        assert_eq!(circuit.gate_node_count(), 2);
        assert_eq!(circuit.gate_count(), 6);
        assert_eq!(circuit.build_traversal_index().num_gate_nodes(), 2);
        assert_eq!(circuit.single_qubit_gate_count(), 4);
        assert_eq!(circuit.two_qubit_gate_count(), 2);
        assert_eq!(circuit.gate_type_count(GateType::H), 4);
        assert_eq!(circuit.gate_type_count(GateType::CX), 2);
    }

    #[test]
    fn test_separate_compatible_nodes_remain_separate_nodes() {
        let mut circuit = DagCircuit::new();

        circuit.add_gate(Gate::h(&[0]));
        circuit.add_gate(Gate::h(&[1]));
        circuit.add_gate(Gate::cx(&[(2, 3)]));
        circuit.add_gate(Gate::cx(&[(4, 5)]));

        assert_eq!(circuit.gate_node_count(), 4);
        assert_eq!(circuit.gate_count(), 4);
        assert_eq!(circuit.build_traversal_index().num_gate_nodes(), 4);
        assert_eq!(circuit.gate_type_count(GateType::H), 2);
        assert_eq!(circuit.gate_type_count(GateType::CX), 2);

        let ticks = crate::TickCircuit::from(&circuit);
        assert_eq!(ticks.gate_count(), 4);
        assert_eq!(ticks.gate_batch_count(), 2);
        assert_eq!(ticks.get_tick(0).unwrap().gate_batch_count(), 2);
    }

    #[test]
    fn test_two_qubit_gate_multiple_wires() {
        let mut circuit = DagCircuit::new();

        let h0 = circuit.add_gate(Gate::h(&[0]));
        let h1 = circuit.add_gate(Gate::h(&[1]));
        let cx = circuit.add_gate(Gate::cx(&[(0, 1)]));

        // CX has two incoming wires
        circuit.connect(h0, cx, QubitId::from(0)).unwrap();
        circuit.connect(h1, cx, QubitId::from(1)).unwrap();

        assert_eq!(circuit.wire_count(), 2);

        let incoming = circuit.incoming_wires(cx);
        assert_eq!(incoming.len(), 2);

        let qubits: BTreeSet<QubitId> = incoming.iter().map(|(_, q)| *q).collect();
        assert!(qubits.contains(&QubitId::from(0)));
        assert!(qubits.contains(&QubitId::from(1)));
    }

    #[test]
    fn test_connect_all() {
        let mut circuit = DagCircuit::new();

        let cx1 = circuit.add_gate(Gate::cx(&[(0, 1)]));
        let cx2 = circuit.add_gate(Gate::cx(&[(0, 1)]));

        // Both gates share qubits 0 and 1
        let connections = circuit.connect_all(cx1, cx2).unwrap();

        assert_eq!(connections.len(), 2);
        assert_eq!(circuit.wire_count(), 2);
    }

    #[test]
    fn test_predecessor_successor_on_qubit() {
        let mut circuit = DagCircuit::new();

        let h = circuit.add_gate(Gate::h(&[0]));
        let cx = circuit.add_gate(Gate::cx(&[(0, 1)]));
        let t = circuit.add_gate(Gate::new(
            GateType::T,
            vec![],
            vec![],
            vec![QubitId::from(0)],
        ));

        circuit.connect(h, cx, QubitId::from(0)).unwrap();
        circuit.connect(cx, t, QubitId::from(0)).unwrap();

        // CX's predecessor on qubit 0 is H
        assert_eq!(circuit.predecessor_on_qubit(cx, QubitId::from(0)), Some(h));
        // CX has no predecessor on qubit 1
        assert_eq!(circuit.predecessor_on_qubit(cx, QubitId::from(1)), None);

        // CX's successor on qubit 0 is T
        assert_eq!(circuit.successor_on_qubit(cx, QubitId::from(0)), Some(t));
    }

    #[test]
    fn test_topological_order() {
        let mut circuit = DagCircuit::new();

        let h = circuit.add_gate(Gate::h(&[0]));
        let t = circuit.add_gate(Gate::new(
            GateType::T,
            vec![],
            vec![],
            vec![QubitId::from(0)],
        ));
        let cx = circuit.add_gate(Gate::cx(&[(0, 1)]));

        circuit.connect(h, t, QubitId::from(0)).unwrap();
        circuit.connect(t, cx, QubitId::from(0)).unwrap();

        let order = circuit.topological_order();
        assert_eq!(order, vec![h, t, cx]);
    }

    #[test]
    fn test_reject_cycle() {
        let mut circuit = DagCircuit::new();

        let a = circuit.add_gate(Gate::h(&[0]));
        let b = circuit.add_gate(Gate::h(&[0]));
        let c = circuit.add_gate(Gate::h(&[0]));

        circuit.connect(a, b, QubitId::from(0)).unwrap();
        circuit.connect(b, c, QubitId::from(0)).unwrap();

        // This would create a cycle
        assert!(circuit.connect(c, a, QubitId::from(0)).is_err());
    }

    #[test]
    fn test_wires() {
        let mut circuit = DagCircuit::new();

        let h = circuit.add_gate(Gate::h(&[0]));
        let t = circuit.add_gate(Gate::new(
            GateType::T,
            vec![],
            vec![],
            vec![QubitId::from(0)],
        ));

        circuit.connect(h, t, QubitId::from(0)).unwrap();

        let wires = circuit.wires();
        assert_eq!(wires.len(), 1);
        assert_eq!(wires[0], (h, t, QubitId::from(0)));
    }

    #[test]
    fn test_wires_for_qubit() {
        let mut circuit = DagCircuit::new();

        let h0 = circuit.add_gate(Gate::h(&[0]));
        let h1 = circuit.add_gate(Gate::h(&[1]));
        let cx = circuit.add_gate(Gate::cx(&[(0, 1)]));

        let e0 = circuit.connect(h0, cx, QubitId::from(0)).unwrap();
        let e1 = circuit.connect(h1, cx, QubitId::from(1)).unwrap();

        let wires_q0 = circuit.wires_for_qubit(QubitId::from(0));
        assert_eq!(wires_q0, vec![e0]);

        let wires_q1 = circuit.wires_for_qubit(QubitId::from(1));
        assert_eq!(wires_q1, vec![e1]);
    }

    #[test]
    fn test_qubit_timeline() {
        let mut circuit = DagCircuit::new();

        let h = circuit.add_gate(Gate::h(&[0]));
        let t = circuit.add_gate(Gate::new(
            GateType::T,
            vec![],
            vec![],
            vec![QubitId::from(0)],
        ));
        let rz = circuit.add_gate(Gate::rz(Angle64::from_turns(0.5), &[0]));

        circuit.connect(h, t, QubitId::from(0)).unwrap();
        circuit.connect(t, rz, QubitId::from(0)).unwrap();

        let timeline = circuit.qubit_timeline(QubitId::from(0));
        assert_eq!(timeline, vec![h, t, rz]);
    }

    #[test]
    fn test_layers() {
        let mut circuit = DagCircuit::new();

        // Layer 0: h0, h1 (parallel)
        let h0 = circuit.add_gate(Gate::h(&[0]));
        let h1 = circuit.add_gate(Gate::h(&[1]));

        // Layer 1: cx
        let cx = circuit.add_gate(Gate::cx(&[(0, 1)]));
        circuit.connect(h0, cx, QubitId::from(0)).unwrap();
        circuit.connect(h1, cx, QubitId::from(1)).unwrap();

        let layers: Vec<Vec<usize>> = circuit.layers().collect();
        assert_eq!(layers.len(), 2);

        // First layer contains both H gates
        assert!(layers[0].contains(&h0));
        assert!(layers[0].contains(&h1));

        // Second layer contains CX
        assert_eq!(layers[1], vec![cx]);
    }

    #[test]
    fn test_gate_type_count() {
        let mut circuit = DagCircuit::new();

        circuit.add_gate(Gate::h(&[0]));
        circuit.add_gate(Gate::h(&[1]));
        circuit.add_gate(Gate::cx(&[(0, 1)]));
        circuit.add_gate(Gate::rz(Angle64::from_turns(0.5), &[0]));

        assert_eq!(circuit.gate_type_count(GateType::H), 2);
        assert_eq!(circuit.gate_type_count(GateType::CX), 1);
        assert_eq!(circuit.gate_type_count(GateType::RZ), 1);
        assert_eq!(circuit.gate_type_count(GateType::X), 0);
    }

    #[test]
    fn test_remove_gate() {
        let mut circuit = DagCircuit::new();

        let h = circuit.add_gate(Gate::h(&[0]));
        let cx = circuit.add_gate(Gate::cx(&[(0, 1)]));
        circuit.connect(h, cx, QubitId::from(0)).unwrap();

        assert_eq!(circuit.gate_count(), 2);
        assert_eq!(circuit.wire_count(), 1);

        let removed = circuit.remove_gate(h);
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().gate_type, GateType::H);
        assert_eq!(circuit.gate_count(), 1);
        assert_eq!(circuit.wire_count(), 0); // Wire was also removed
    }

    #[test]
    fn test_qubits() {
        let mut circuit = DagCircuit::new();

        circuit.add_gate(Gate::h(&[0]));
        circuit.add_gate(Gate::h(&[2]));
        circuit.add_gate(Gate::cx(&[(0, 1)]));

        let qubits = circuit.qubits();
        assert_eq!(qubits.len(), 3);
        assert!(qubits.contains(&QubitId::from(0)));
        assert!(qubits.contains(&QubitId::from(1)));
        assert!(qubits.contains(&QubitId::from(2)));
    }

    #[test]
    fn test_circuit_attributes() {
        let mut circuit = DagCircuit::new();

        // Set circuit-level attributes
        circuit.set_attr("name", Attribute::String("test_circuit".to_string()));
        circuit.set_attr("version", Attribute::Int(1));
        circuit.set_attr("optimized", Attribute::Bool(true));

        // Get them back
        assert_eq!(
            circuit.get_attr("name"),
            Some(&Attribute::String("test_circuit".to_string()))
        );
        assert_eq!(circuit.get_attr("version"), Some(&Attribute::Int(1)));
        assert_eq!(circuit.get_attr("optimized"), Some(&Attribute::Bool(true)));
        assert_eq!(circuit.get_attr("nonexistent"), None);
    }

    #[test]
    fn test_gate_attributes() {
        let mut circuit = DagCircuit::new();

        let h = circuit.add_gate(Gate::h(&[0]));
        let rz = circuit.add_gate(Gate::rz(
            Angle64::from_radians(std::f64::consts::PI / 4.0),
            &[0],
        ));

        // Set gate-level attributes
        assert!(circuit.set_gate_attr(h, "error_rate", Attribute::Float(0.001)));
        assert!(circuit.set_gate_attr(rz, "angle", Attribute::Float(std::f64::consts::PI / 4.0)));
        assert!(circuit.set_gate_attr(rz, "source", Attribute::String("optimization".to_string())));

        // Get them back
        assert_eq!(
            circuit.get_gate_attr(h, "error_rate"),
            Some(&Attribute::Float(0.001))
        );
        assert_eq!(
            circuit.get_gate_attr(rz, "angle"),
            Some(&Attribute::Float(std::f64::consts::PI / 4.0))
        );
        assert_eq!(circuit.get_gate_attr(h, "angle"), None);

        // Non-existent gate
        assert!(!circuit.set_gate_attr(999, "key", Attribute::Bool(true)));
        assert_eq!(circuit.get_gate_attr(999, "key"), None);
    }

    #[test]
    fn test_wire_attributes() {
        let mut circuit = DagCircuit::new();

        let h = circuit.add_gate(Gate::h(&[0]));
        let t = circuit.add_gate(Gate::new(
            GateType::T,
            vec![],
            vec![],
            vec![QubitId::from(0)],
        ));
        let edge_id = circuit.connect(h, t, QubitId::from(0)).unwrap();

        // Set wire-level attributes
        assert!(circuit.set_wire_attr(edge_id, "delay_ns", Attribute::Float(10.5)));
        assert!(circuit.set_wire_attr(edge_id, "channel", Attribute::String("q0".to_string())));

        // Get them back
        assert_eq!(
            circuit.get_wire_attr(edge_id, "delay_ns"),
            Some(&Attribute::Float(10.5))
        );
        assert_eq!(
            circuit.get_wire_attr(edge_id, "channel"),
            Some(&Attribute::String("q0".to_string()))
        );
        assert_eq!(circuit.get_wire_attr(edge_id, "nonexistent"), None);

        // Non-existent wire
        assert!(!circuit.set_wire_attr(999, "key", Attribute::Bool(true)));
        assert_eq!(circuit.get_wire_attr(999, "key"), None);
    }

    #[test]
    fn test_wire_attrs_between() {
        let mut circuit = DagCircuit::new();

        let h = circuit.add_gate(Gate::h(&[0]));
        let t = circuit.add_gate(Gate::new(
            GateType::T,
            vec![],
            vec![],
            vec![QubitId::from(0)],
        ));
        circuit.connect(h, t, QubitId::from(0)).unwrap();

        // Access wire attributes by endpoints
        let attrs = circuit.wire_attrs_between_mut(h, t);
        assert!(attrs.is_some());
        attrs
            .unwrap()
            .insert("test".to_string(), Attribute::Bool(true));

        // Read it back
        let attrs = circuit.wire_attrs_between(h, t);
        assert!(attrs.is_some());
        assert_eq!(attrs.unwrap().get("test"), Some(&Attribute::Bool(true)));

        // Non-existent edge
        assert!(circuit.wire_attrs_between(h, h).is_none());
    }

    #[test]
    fn test_builder_methods_chaining() {
        // Test that builder methods allow chaining and auto-wire correctly
        let mut circuit = DagCircuit::new();

        // Build a Bell state using method chaining
        circuit.h(&[0]).cx(&[(0, 1)]);

        assert_eq!(circuit.gate_count(), 2);
        assert_eq!(circuit.wire_count(), 1); // H -> CX on qubit 0
        assert_eq!(circuit.width(), 2);

        // Check topological order is correct
        let order = circuit.topological_order();
        assert_eq!(order.len(), 2);

        // First gate should be H
        let first_gate = circuit.gate(order[0]).unwrap();
        assert_eq!(first_gate.gate_type, GateType::H);

        // Second gate should be CX
        let second_gate = circuit.gate(order[1]).unwrap();
        assert_eq!(second_gate.gate_type, GateType::CX);
    }

    #[test]
    fn test_builder_with_rotation_gates() {
        use std::f64::consts::PI;

        let mut circuit = DagCircuit::new();

        // Test that rotation gates accept both Angle64 and f64
        // API follows simulator convention: rx(theta, q)
        circuit
            .h(&[0])
            .rz(PI / 4.0, &[0]) // f64 in radians, then qubit
            .rx(Angle64::QUARTER_TURN, &[0]); // Angle64, then qubit
        circuit.mz(&[0]);

        assert_eq!(circuit.gate_count(), 4);
        assert_eq!(circuit.wire_count(), 3); // h -> rz -> rx -> mz

        // Check the RZ angle was stored correctly
        let gates: Vec<_> = circuit.iter_gates_topo().collect();
        let rz_gate = gates
            .iter()
            .find(|(_, g)| g.gate_type == GateType::RZ)
            .unwrap()
            .1;
        assert!((rz_gate.angles[0].to_radians() - PI / 4.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder_idle() {
        use pecos_core::TimeUnits;

        let mut circuit = DagCircuit::new();

        // Idle gates represent waiting time in abstract time units
        circuit.h(&[0]).idle(TimeUnits::new(100), &[0]).h(&[0]);
        circuit.mz(&[0]);

        assert_eq!(circuit.gate_count(), 4);

        // Check the idle duration was stored correctly
        let gates: Vec<_> = circuit.iter_gates_topo().collect();
        let idle_gate = gates
            .iter()
            .find(|(_, g)| g.gate_type == GateType::Idle)
            .unwrap()
            .1;
        assert!((idle_gate.idle_duration() - 100.0).abs() < 1e-10);

        // Test with different time units
        let mut circuit2 = DagCircuit::new();
        circuit2.idle(TimeUnits::new(1000), &[0]);
        let gate = circuit2.gate(0).unwrap();
        assert!((gate.idle_duration() - 1000.0).abs() < 1e-10);

        // Test with u64
        let mut circuit3 = DagCircuit::new();
        circuit3.idle(200u64, &[0]);
        let gate = circuit3.gate(0).unwrap();
        assert!((gate.idle_duration() - 200.0).abs() < 1e-10);
    }

    #[test]
    fn test_builder_two_qubit_parallel_paths() {
        let mut circuit = DagCircuit::new();

        // Two parallel qubit paths
        // Measurements are not chainable (matches simulator API)
        circuit.h(&[0]).h(&[1]).cx(&[(0, 1)]);
        circuit.mz(&[0]);
        circuit.mz(&[1]);

        assert_eq!(circuit.gate_count(), 5);
        // Wires: h0->cx, h1->cx, cx->m0, cx->m1
        assert_eq!(circuit.wire_count(), 4);
        assert_eq!(circuit.width(), 2);

        // Check qubit timelines
        let q0_timeline = circuit.qubit_timeline(QubitId::from(0));
        assert_eq!(q0_timeline.len(), 3); // h, cx, mz

        let q1_timeline = circuit.qubit_timeline(QubitId::from(1));
        assert_eq!(q1_timeline.len(), 3); // h, cx, mz
    }

    #[test]
    fn test_builder_simulator_api_compatibility() {
        use std::f64::consts::FRAC_PI_4;

        // Test that the API matches simulator conventions
        let mut circuit = DagCircuit::new();

        // Matches CliffordGateable API
        circuit
            .h(&[0])
            .sz(&[0]) // sqrt(Z), same as simulator
            .szdg(&[0]) // sqrt(Z) dagger
            .cx(&[(0, 1)])
            .szz(&[(0, 1)]) // sqrt(ZZ)
            .szzdg(&[(0, 1)]); // sqrt(ZZ) dagger

        assert_eq!(circuit.gate_count(), 6);

        // Matches ArbitraryRotationGateable API: rx(theta, q)
        let mut circuit2 = DagCircuit::new();
        circuit2
            .rx(FRAC_PI_4, &[0])
            .ry(FRAC_PI_4, &[0])
            .rz(FRAC_PI_4, &[0])
            .rzz(FRAC_PI_4, &[(0, 1)]); // rzz(theta, &[(q1, q2)])

        assert_eq!(circuit2.gate_count(), 4);
    }

    #[test]
    fn test_builder_meta() {
        let mut circuit = DagCircuit::new();

        // Chain meta with gates (meta returns &mut Self for gates)
        circuit
            .h(&[0])
            .meta("error_rate", Attribute::Float(0.001))
            .cx(&[(0, 1)])
            .meta("fidelity", Attribute::Float(0.99));

        // Measurement returns node indices; use last_node for meta
        circuit.mz(&[0]);
        circuit.meta("basis", Attribute::String("Z".to_string()));

        assert_eq!(circuit.gate_count(), 3);

        // Verify attributes were set correctly
        let gates: Vec<_> = circuit.iter_gates_topo().collect();

        // H gate should have error_rate
        let h_node = gates
            .iter()
            .find(|(_, g)| g.gate_type == GateType::H)
            .unwrap()
            .0;
        assert_eq!(
            circuit.get_gate_attr(h_node, "error_rate"),
            Some(&Attribute::Float(0.001))
        );

        // CX gate should have fidelity
        let cx_node = gates
            .iter()
            .find(|(_, g)| g.gate_type == GateType::CX)
            .unwrap()
            .0;
        assert_eq!(
            circuit.get_gate_attr(cx_node, "fidelity"),
            Some(&Attribute::Float(0.99))
        );

        // Measure gate should have basis
        let mz_node = gates
            .iter()
            .find(|(_, g)| g.gate_type == GateType::MZ)
            .unwrap()
            .0;
        assert_eq!(
            circuit.get_gate_attr(mz_node, "basis"),
            Some(&Attribute::String("Z".to_string()))
        );
    }

    #[test]
    fn test_mz_returns_refs() {
        let mut circuit = DagCircuit::new();

        circuit.h(&[0]);
        let refs = circuit.mz(&[0]);
        assert_eq!(refs.len(), 1);
        circuit.h(&[1]); // continue building

        assert_eq!(circuit.gate_count(), 3);
    }

    #[test]
    fn test_pz_chainable() {
        let mut circuit = DagCircuit::new();

        circuit.pz(&[0]);
        circuit.h(&[0]); // continue building
        circuit.mz(&[0]);

        assert_eq!(circuit.gate_count(), 3);
    }

    #[test]
    fn test_prep_handle_with_meta() {
        let mut circuit = DagCircuit::new();

        circuit
            .pz(&[0])
            .meta("reason", Attribute::String("reset".to_string()));
        circuit.h(&[0]);

        assert_eq!(circuit.gate_count(), 2);

        // Verify the metadata was attached
        let attrs = circuit.gate_attrs(0).expect("gate should have attributes");
        assert_eq!(
            attrs.get("reason"),
            Some(&Attribute::String("reset".to_string()))
        );
    }

    #[test]
    fn test_last_added_node() {
        let mut circuit = DagCircuit::new();

        assert!(circuit.last_added_node().is_none());

        circuit.h(&[0]);
        let h_node = circuit.last_added_node().unwrap();

        circuit.cx(&[(0, 1)]);
        let cx_node = circuit.last_added_node().unwrap();

        assert_ne!(h_node, cx_node);

        // Verify the nodes have the right gates
        assert_eq!(circuit.gate(h_node).unwrap().gate_type, GateType::H);
        assert_eq!(circuit.gate(cx_node).unwrap().gate_type, GateType::CX);
    }

    #[test]
    fn test_metas_on_gate() {
        let mut circuit = DagCircuit::new();

        let attrs = BTreeMap::from([
            ("duration".to_string(), Attribute::Float(50.0)),
            ("error_rate".to_string(), Attribute::Float(0.001)),
        ]);

        circuit.h(&[0]).metas(attrs).cx(&[(0, 1)]);

        let gate_attrs = circuit.gate_attrs(0).expect("gate should have attributes");
        assert_eq!(gate_attrs.get("duration"), Some(&Attribute::Float(50.0)));
        assert_eq!(gate_attrs.get("error_rate"), Some(&Attribute::Float(0.001)));
    }

    #[test]
    fn test_set_attrs_on_circuit() {
        let mut circuit = DagCircuit::new();

        let attrs = BTreeMap::from([
            ("name".to_string(), Attribute::String("bell".to_string())),
            ("version".to_string(), Attribute::Int(1)),
        ]);

        circuit.set_attrs(attrs);
        circuit.h(&[0]);

        assert_eq!(
            circuit.get_attr("name"),
            Some(&Attribute::String("bell".to_string()))
        );
        assert_eq!(circuit.get_attr("version"), Some(&Attribute::Int(1)));
    }

    #[test]
    fn test_set_gate_attrs() {
        let mut circuit = DagCircuit::new();

        let node = circuit.add_gate(Gate::h(&[0]));

        let attrs = BTreeMap::from([
            ("duration".to_string(), Attribute::Float(50.0)),
            ("fidelity".to_string(), Attribute::Float(0.999)),
        ]);

        assert!(circuit.set_gate_attrs(node, attrs));

        let gate_attrs = circuit
            .gate_attrs(node)
            .expect("gate should have attributes");
        assert_eq!(gate_attrs.get("duration"), Some(&Attribute::Float(50.0)));
        assert_eq!(gate_attrs.get("fidelity"), Some(&Attribute::Float(0.999)));
    }

    #[test]
    fn test_try_add_gate_rejects_invalid_gate_payload() {
        let mut circuit = DagCircuit::new();
        let err = circuit
            .try_add_gate(Gate::cx(&[(0, 0)]))
            .expect_err("DAG should reject invalid gate payloads");

        assert!(err.contains("requires distinct qubits"));
        assert!(circuit.nodes().is_empty());
    }

    #[test]
    #[should_panic(expected = "Invalid gate")]
    fn test_add_gate_panics_on_invalid_gate_payload() {
        let mut circuit = DagCircuit::new();
        circuit.add_gate(Gate::cx(&[(0, 0)]));
    }
}
