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

//! Tick-based quantum circuit representation.
//!
//! A [`TickCircuit`] represents a quantum circuit as a sequence of discrete time slices
//! called "ticks". Each tick contains gates that execute in parallel on non-overlapping
//! qubits.
//!
//! This representation is natural for:
//! - QEC syndrome extraction rounds
//! - Hardware with discrete clock cycles
//! - Protocols with explicit parallelism
//!
//! # Example
//!
//! ```
//! use pecos_quantum::TickCircuit;
//!
//! let mut circuit = TickCircuit::new();
//!
//! // Each tick() returns a handle - regular gates chain on the handle
//! circuit.tick().pz(0);               // Tick 0: Prepare q0 (breaks chain)
//! circuit.tick().pz(1);               // Tick 1: Prepare q1 (breaks chain)
//! circuit.tick().h(0).x(1);           // Tick 2: H on q0, X on q1 (chains!)
//! circuit.tick().cx(0, 1);            // Tick 3: CNOT
//! circuit.tick().mz(0);               // Tick 4: Measure q0 (breaks chain)
//! circuit.tick().mz(1);               // Tick 5: Measure q1 (breaks chain)
//!
//! assert_eq!(circuit.num_ticks(), 6);
//!
//! // Preps and measurements break the chain but allow .meta():
//! circuit.tick().pz(0).meta("reason", pecos_quantum::Attribute::String("init".into()));
//! circuit.tick().mz(0).meta("basis", pecos_quantum::Attribute::String("Z".into()));
//!
//! // Tick-level metadata: call meta() before adding gates
//! use pecos_quantum::Attribute;
//! let mut circuit2 = TickCircuit::new();
//! circuit2.tick()
//!     .meta("round", Attribute::Int(0))    // Tick metadata (no gates added yet)
//!     .h(0)
//!     .meta("duration", Attribute::Float(50.0)); // Gate metadata (after a gate)
//! ```

use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, Gate, Nanoseconds, QubitId};
use std::collections::{BTreeMap, BTreeSet};

use crate::Attribute;
use crate::dag_circuit::DagCircuit;
use std::fmt;

/// Error when trying to add a gate that uses a qubit already in use in this tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct QubitConflictError {
    /// The qubit(s) that are already in use.
    pub conflicting_qubits: Vec<QubitId>,
    /// The tick index where the conflict occurred.
    pub tick_idx: Option<usize>,
}

impl fmt::Display for QubitConflictError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let qubits: Vec<String> = self
            .conflicting_qubits
            .iter()
            .map(std::string::ToString::to_string)
            .collect();
        if let Some(idx) = self.tick_idx {
            write!(
                f,
                "Qubit(s) {} already in use in tick {}",
                qubits.join(", "),
                idx
            )
        } else {
            write!(
                f,
                "Qubit(s) {} already in use in this tick",
                qubits.join(", ")
            )
        }
    }
}

impl std::error::Error for QubitConflictError {}

/// A single time slice containing gates that execute in parallel.
#[derive(Debug, Clone, Default)]
pub struct Tick {
    /// Gates in this tick (all act on disjoint qubits).
    gates: Vec<Gate>,
    /// Metadata for each gate, indexed by position in `gates`.
    gate_attrs: BTreeMap<usize, BTreeMap<String, Attribute>>,
    /// Tick-level metadata.
    attrs: BTreeMap<String, Attribute>,
}

impl Tick {
    /// Create a new empty tick.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the number of gates in this tick.
    #[must_use]
    pub fn len(&self) -> usize {
        self.gates.len()
    }

    /// Check if the tick is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gates.is_empty()
    }

    /// Get the gates in this tick.
    #[must_use]
    pub fn gates(&self) -> &[Gate] {
        &self.gates
    }

    /// Add a gate to this tick.
    pub fn add_gate(&mut self, gate: Gate) -> usize {
        let idx = self.gates.len();
        self.gates.push(gate);
        idx
    }

    /// Set metadata on a gate.
    pub fn set_gate_attr(&mut self, gate_idx: usize, key: &str, value: Attribute) {
        self.gate_attrs
            .entry(gate_idx)
            .or_default()
            .insert(key.to_string(), value);
    }

    /// Set multiple metadata attributes on a gate at once.
    pub fn set_gate_attrs(&mut self, gate_idx: usize, attrs: BTreeMap<String, Attribute>) {
        self.gate_attrs.entry(gate_idx).or_default().extend(attrs);
    }

    /// Get metadata from a gate.
    #[must_use]
    pub fn get_gate_attr(&self, gate_idx: usize, key: &str) -> Option<&Attribute> {
        self.gate_attrs.get(&gate_idx).and_then(|m| m.get(key))
    }

    /// Set tick-level metadata.
    pub fn set_attr(&mut self, key: &str, value: Attribute) {
        self.attrs.insert(key.to_string(), value);
    }

    /// Set multiple tick-level metadata attributes at once.
    pub fn set_attrs(&mut self, attrs: BTreeMap<String, Attribute>) {
        self.attrs.extend(attrs);
    }

    /// Get tick-level metadata.
    #[must_use]
    pub fn get_attr(&self, key: &str) -> Option<&Attribute> {
        self.attrs.get(key)
    }

    /// Get all attributes for a gate.
    pub fn gate_attrs(&self, gate_idx: usize) -> impl Iterator<Item = (&String, &Attribute)> {
        self.gate_attrs
            .get(&gate_idx)
            .into_iter()
            .flat_map(|m| m.iter())
    }

    /// Get all tick-level attributes.
    pub fn tick_attrs(&self) -> impl Iterator<Item = (&String, &Attribute)> {
        self.attrs.iter()
    }

    /// Get the set of qubits used in this tick.
    ///
    /// This is computed lazily by iterating through all gates.
    #[must_use]
    pub fn active_qubits(&self) -> BTreeSet<QubitId> {
        self.gates
            .iter()
            .flat_map(|gate| gate.qubits.iter().copied())
            .collect()
    }

    /// Check if a specific qubit is already in use in this tick.
    #[must_use]
    pub fn uses_qubit(&self, qubit: QubitId) -> bool {
        self.gates.iter().any(|gate| gate.qubits.contains(&qubit))
    }

    /// Check if any of the given qubits are already in use in this tick.
    ///
    /// Returns the conflicting qubits if any.
    #[must_use]
    pub fn find_conflicts(&self, qubits: &[QubitId]) -> Vec<QubitId> {
        let active = self.active_qubits();
        qubits
            .iter()
            .filter(|q| active.contains(q))
            .copied()
            .collect()
    }

    /// Try to add a gate to this tick, returning an error if any qubit is already in use.
    ///
    /// # Errors
    ///
    /// Returns `QubitConflictError` if any qubit in the gate is already used by another gate in this tick.
    pub fn try_add_gate(&mut self, gate: Gate) -> Result<usize, QubitConflictError> {
        let conflicts = self.find_conflicts(&gate.qubits);
        if !conflicts.is_empty() {
            return Err(QubitConflictError {
                conflicting_qubits: conflicts,
                tick_idx: None,
            });
        }
        Ok(self.add_gate(gate))
    }
}

/// A quantum circuit represented as a sequence of parallel time slices (ticks).
///
/// Each tick contains gates that can execute simultaneously on non-overlapping qubits.
/// This representation is simpler than a DAG and natural for clocked/synchronized
/// quantum operations.
///
/// # Example
///
/// ```
/// use pecos_quantum::TickCircuit;
///
/// let mut circuit = TickCircuit::new();
///
/// // Each tick() returns a TickHandle for adding gates
/// // Regular gates chain, but preps/measurements break the chain
/// circuit.tick().pz(0);                  // Tick 0: Prepare q0 (breaks chain)
/// circuit.tick().pz(1);                  // Tick 1: Prepare q1 (breaks chain)
/// circuit.tick().h(0).x(1);              // Tick 2: H and X (chains!)
/// circuit.tick().cx(0, 1);               // Tick 3: CNOT
/// circuit.tick().mz(0);                  // Tick 4: Measure q0 (breaks chain)
/// circuit.tick().mz(1);                  // Tick 5: Measure q1 (breaks chain)
///
/// assert_eq!(circuit.num_ticks(), 6);
/// ```
#[derive(Debug, Clone, Default)]
pub struct TickCircuit {
    /// The sequence of ticks.
    ticks: Vec<Tick>,
    /// Next tick index to allocate.
    next_tick: usize,
    /// Circuit-level metadata.
    circuit_attrs: BTreeMap<String, Attribute>,
}

/// Handle to a specific tick for adding gates.
///
/// Gates added through the handle are placed in the associated tick.
/// The handle chains for fluent API usage.
pub struct TickHandle<'a> {
    circuit: &'a mut TickCircuit,
    tick_idx: usize,
    last_gate_idx: Option<usize>,
}

/// Handle returned by preparation operations on a tick.
///
/// This handle breaks the method chain (unlike regular gates),
/// but still allows attaching metadata via `.meta()`.
pub struct TickPrepHandle<'a> {
    circuit: &'a mut TickCircuit,
    tick_idx: usize,
    gate_idx: usize,
}

impl TickPrepHandle<'_> {
    /// Add metadata to this preparation.
    ///
    /// Returns `()` to break the chain.
    pub fn meta(self, key: &str, value: impl Into<Attribute>) {
        if let Some(tick) = self.circuit.get_tick_mut(self.tick_idx) {
            tick.set_gate_attr(self.gate_idx, key, value.into());
        }
    }

    /// Add multiple metadata attributes to this preparation.
    ///
    /// Returns `()` to break the chain.
    pub fn metas(self, attrs: BTreeMap<String, Attribute>) {
        if let Some(tick) = self.circuit.get_tick_mut(self.tick_idx) {
            tick.set_gate_attrs(self.gate_idx, attrs);
        }
    }
}

/// Handle returned by measurement operations on a tick.
///
/// This handle breaks the method chain (unlike regular gates),
/// but still allows attaching metadata via `.meta()`.
pub struct TickMeasureHandle<'a> {
    circuit: &'a mut TickCircuit,
    tick_idx: usize,
    gate_idx: usize,
}

impl TickMeasureHandle<'_> {
    /// Add metadata to this measurement.
    ///
    /// Returns `()` to break the chain.
    pub fn meta(self, key: &str, value: impl Into<Attribute>) {
        if let Some(tick) = self.circuit.get_tick_mut(self.tick_idx) {
            tick.set_gate_attr(self.gate_idx, key, value.into());
        }
    }

    /// Add multiple metadata attributes to this measurement.
    ///
    /// Returns `()` to break the chain.
    pub fn metas(self, attrs: BTreeMap<String, Attribute>) {
        if let Some(tick) = self.circuit.get_tick_mut(self.tick_idx) {
            tick.set_gate_attrs(self.gate_idx, attrs);
        }
    }
}

impl TickCircuit {
    /// Create a new empty tick circuit.
    #[must_use]
    pub fn new() -> Self {
        Self {
            ticks: Vec::new(),
            next_tick: 0,
            circuit_attrs: BTreeMap::new(),
        }
    }

    /// Get the number of ticks (excluding trailing empty ticks).
    #[must_use]
    pub fn num_ticks(&self) -> usize {
        let mut count = self.ticks.len();
        while count > 0 && self.ticks[count - 1].is_empty() {
            count -= 1;
        }
        count
    }

    /// Get the total number of gates across all ticks.
    #[must_use]
    pub fn gate_count(&self) -> usize {
        self.ticks.iter().map(Tick::len).sum()
    }

    /// Get a tick by index.
    #[must_use]
    pub fn get_tick(&self, idx: usize) -> Option<&Tick> {
        self.ticks.get(idx)
    }

    /// Get a mutable tick by index.
    pub fn get_tick_mut(&mut self, idx: usize) -> Option<&mut Tick> {
        self.ticks.get_mut(idx)
    }

    /// Get all ticks.
    #[must_use]
    pub fn ticks(&self) -> &[Tick] {
        &self.ticks
    }

    /// Get the next tick index that will be allocated.
    #[must_use]
    pub fn next_tick_index(&self) -> usize {
        self.next_tick
    }

    /// Create a new tick and return a handle for adding gates to it.
    ///
    /// The tick acts as a mini-circuit where gates can be chained.
    pub fn tick(&mut self) -> TickHandle<'_> {
        let tick_idx = self.next_tick;
        self.next_tick += 1;

        // Ensure tick exists
        while tick_idx >= self.ticks.len() {
            self.ticks.push(Tick::new());
        }

        TickHandle {
            circuit: self,
            tick_idx,
            last_gate_idx: None,
        }
    }

    /// Set circuit-level metadata.
    pub fn set_meta(&mut self, key: &str, value: impl Into<Attribute>) {
        self.circuit_attrs.insert(key.to_string(), value.into());
    }

    /// Set multiple circuit-level metadata attributes at once.
    pub fn set_metas(&mut self, attrs: BTreeMap<String, Attribute>) {
        self.circuit_attrs.extend(attrs);
    }

    /// Get circuit-level metadata.
    #[must_use]
    pub fn get_meta(&self, key: &str) -> Option<&Attribute> {
        self.circuit_attrs.get(key)
    }

    /// Get all circuit-level attributes.
    pub fn circuit_attrs(&self) -> impl Iterator<Item = (&String, &Attribute)> {
        self.circuit_attrs.iter()
    }
}

// ============================================================================
// TickHandle - handle for adding gates to a specific tick
// ============================================================================

impl<'a> TickHandle<'a> {
    /// Get the tick index this handle refers to.
    #[must_use]
    pub fn index(&self) -> usize {
        self.tick_idx
    }

    /// Add a gate to this tick.
    ///
    /// # Panics
    ///
    /// Panics if any qubit in the gate is already used by another gate in this tick.
    /// Use `try_add_gate` for fallible gate addition.
    fn add_gate(&mut self, gate: Gate) -> &mut Self {
        match self.circuit.ticks[self.tick_idx].try_add_gate(gate) {
            Ok(idx) => {
                self.last_gate_idx = Some(idx);
                self
            }
            Err(mut err) => {
                err.tick_idx = Some(self.tick_idx);
                panic!("{}", err);
            }
        }
    }

    /// Try to add a gate to this tick, returning an error if any qubit is already in use.
    ///
    /// # Errors
    ///
    /// Returns `QubitConflictError` if any qubit in the gate is already used by another gate in this tick.
    pub fn try_add_gate(&mut self, gate: Gate) -> Result<&mut Self, QubitConflictError> {
        match self.circuit.ticks[self.tick_idx].try_add_gate(gate) {
            Ok(idx) => {
                self.last_gate_idx = Some(idx);
                Ok(self)
            }
            Err(mut err) => {
                err.tick_idx = Some(self.tick_idx);
                Err(err)
            }
        }
    }

    /// Add a gate and return the gate index.
    ///
    /// # Panics
    ///
    /// Panics if any qubit in the gate is already used by another gate in this tick.
    fn add_gate_get_idx(&mut self, gate: Gate) -> usize {
        match self.circuit.ticks[self.tick_idx].try_add_gate(gate) {
            Ok(idx) => {
                self.last_gate_idx = Some(idx);
                idx
            }
            Err(mut err) => {
                err.tick_idx = Some(self.tick_idx);
                panic!("{}", err);
            }
        }
    }

    /// Set metadata on the last added gate.
    ///
    /// If no gate has been added yet, sets tick-level metadata instead.
    pub fn meta(&mut self, key: &str, value: impl Into<Attribute>) -> &mut Self {
        if let Some(gate_idx) = self.last_gate_idx {
            self.circuit.ticks[self.tick_idx].set_gate_attr(gate_idx, key, value.into());
        } else {
            // No gate yet - set tick-level metadata
            self.circuit.ticks[self.tick_idx].set_attr(key, value.into());
        }
        self
    }

    /// Set multiple metadata attributes on the last added gate.
    ///
    /// If no gate has been added yet, sets tick-level metadata instead.
    pub fn metas(&mut self, attrs: BTreeMap<String, Attribute>) -> &mut Self {
        if let Some(gate_idx) = self.last_gate_idx {
            self.circuit.ticks[self.tick_idx].set_gate_attrs(gate_idx, attrs);
        } else {
            // No gate yet - set tick-level metadata
            self.circuit.ticks[self.tick_idx].set_attrs(attrs);
        }
        self
    }

    // =========================================================================
    // Single-qubit gates
    // =========================================================================

    /// Apply a Hadamard gate.
    pub fn h(&mut self, q: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::h(&[q.into()]))
    }

    /// Apply a Pauli-X gate.
    pub fn x(&mut self, q: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::x(&[q.into()]))
    }

    /// Apply a Pauli-Y gate.
    pub fn y(&mut self, q: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::y(&[q.into()]))
    }

    /// Apply a Pauli-Z gate.
    pub fn z(&mut self, q: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::z(&[q.into()]))
    }

    /// Apply an S gate (sqrt-Z).
    pub fn sz(&mut self, q: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::simple(GateType::SZ, vec![q.into()]))
    }

    /// Apply an S-dagger gate.
    pub fn szdg(&mut self, q: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::simple(GateType::SZdg, vec![q.into()]))
    }

    /// Apply a T gate.
    pub fn t(&mut self, q: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::simple(GateType::T, vec![q.into()]))
    }

    /// Apply a T-dagger gate.
    pub fn tdg(&mut self, q: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::simple(GateType::Tdg, vec![q.into()]))
    }

    /// Apply an RX rotation.
    pub fn rx(&mut self, theta: impl Into<Angle64>, q: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::rx(theta.into(), &[q.into()]))
    }

    /// Apply an RY rotation.
    pub fn ry(&mut self, theta: impl Into<Angle64>, q: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::ry(theta.into(), &[q.into()]))
    }

    /// Apply an RZ rotation.
    pub fn rz(&mut self, theta: impl Into<Angle64>, q: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::rz(theta.into(), &[q.into()]))
    }

    // =========================================================================
    // Two-qubit gates
    // =========================================================================

    /// Apply a CNOT (CX) gate.
    pub fn cx(&mut self, ctrl: impl Into<QubitId>, tgt: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::cx(&[(ctrl.into(), tgt.into())]))
    }

    /// Apply an SZZ gate (sqrt-ZZ).
    pub fn szz(&mut self, q1: impl Into<QubitId>, q2: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::szz(&[(q1.into(), q2.into())]))
    }

    /// Apply an SZZ-dagger gate.
    pub fn szzdg(&mut self, q1: impl Into<QubitId>, q2: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::szzdg(&[(q1.into(), q2.into())]))
    }

    /// Apply an RZZ rotation.
    pub fn rzz(
        &mut self,
        theta: impl Into<Angle64>,
        q1: impl Into<QubitId>,
        q2: impl Into<QubitId>,
    ) -> &mut Self {
        self.add_gate(Gate::rzz(theta.into(), &[(q1.into(), q2.into())]))
    }

    // =========================================================================
    // State preparation and measurement
    // =========================================================================

    /// Prepare a qubit in the |0⟩ state.
    ///
    /// Returns a [`TickPrepHandle`] that allows attaching metadata via `.meta()`.
    /// This breaks the chain - only `.meta()` can be called on the result.
    pub fn pz(mut self, q: impl Into<QubitId>) -> TickPrepHandle<'a> {
        let gate_idx = self.add_gate_get_idx(Gate::prep(&[q.into()]));
        TickPrepHandle {
            circuit: self.circuit,
            tick_idx: self.tick_idx,
            gate_idx,
        }
    }

    /// Prepare a qubit (alias for pz).
    ///
    /// Returns a [`TickPrepHandle`] that allows attaching metadata via `.meta()`.
    pub fn prep(self, q: impl Into<QubitId>) -> TickPrepHandle<'a> {
        self.pz(q)
    }

    /// Measure a qubit in the Z basis.
    ///
    /// Returns a [`TickMeasureHandle`] that allows attaching metadata via `.meta()`.
    /// This breaks the chain - only `.meta()` can be called on the result.
    pub fn mz(mut self, q: impl Into<QubitId>) -> TickMeasureHandle<'a> {
        let gate_idx = self.add_gate_get_idx(Gate::measure(&[q.into()]));
        TickMeasureHandle {
            circuit: self.circuit,
            tick_idx: self.tick_idx,
            gate_idx,
        }
    }

    /// Measure and free a qubit (destructive measurement).
    ///
    /// Returns a [`TickMeasureHandle`] that allows attaching metadata via `.meta()`.
    pub fn measure_free(mut self, q: impl Into<QubitId>) -> TickMeasureHandle<'a> {
        let gate_idx = self.add_gate_get_idx(Gate::simple(GateType::MeasureFree, vec![q.into()]));
        TickMeasureHandle {
            circuit: self.circuit,
            tick_idx: self.tick_idx,
            gate_idx,
        }
    }

    // =========================================================================
    // Resource management
    // =========================================================================

    /// Allocate a qubit.
    pub fn qalloc(&mut self, q: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::qalloc(&[q.into()]))
    }

    /// Free a qubit.
    pub fn qfree(&mut self, q: impl Into<QubitId>) -> &mut Self {
        self.add_gate(Gate::qfree(&[q.into()]))
    }

    // =========================================================================
    // Timing
    // =========================================================================

    /// Insert an idle (wait) operation.
    pub fn idle(&mut self, duration: impl Into<Nanoseconds>, q: impl Into<QubitId>) -> &mut Self {
        let ns: Nanoseconds = duration.into();
        self.add_gate(Gate::idle(ns.as_f64(), vec![q.into()]))
    }
}

// ============================================================================
// Conversions between TickCircuit and DagCircuit
// ============================================================================

impl From<&DagCircuit> for TickCircuit {
    /// Convert a `DagCircuit` to a `TickCircuit`.
    ///
    /// Each layer of parallel gates in the DAG becomes a tick.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::{DagCircuit, TickCircuit, Gate, QubitId};
    ///
    /// let mut dag = DagCircuit::new();
    /// let h = dag.add_gate(Gate::h(&[0]));
    /// let cx = dag.add_gate(Gate::cx(&[(0, 1)]));
    /// dag.connect(h, cx, QubitId::from(0)).unwrap();
    ///
    /// let tc = TickCircuit::from(&dag);
    /// assert_eq!(tc.num_ticks(), 2); // H in tick 0, CX in tick 1
    /// ```
    fn from(dag: &DagCircuit) -> Self {
        let mut tc = TickCircuit::new();

        for layer in dag.layers() {
            // Allocate a new tick for this layer
            let tick_idx = {
                let handle = tc.tick();
                handle.index()
            };

            // Add all gates in this layer to the tick
            if let Some(tick) = tc.get_tick_mut(tick_idx) {
                for node_id in layer {
                    if let Some(gate) = dag.gate(node_id) {
                        let gate_idx = tick.add_gate(gate.clone());

                        // Copy gate attributes
                        if let Some(attrs) = dag.gate_attrs(node_id) {
                            for (key, value) in attrs {
                                tick.set_gate_attr(gate_idx, key, value.clone());
                            }
                        }
                    }
                }
            }
        }

        // Copy circuit-level attributes, restoring tick-level attrs from prefixed keys
        let tick_attr_prefix = "tick[";
        for (key, value) in dag.attrs() {
            if key.starts_with(tick_attr_prefix) {
                // Parse tick[N].attr_name format
                if let Some(rest) = key.strip_prefix(tick_attr_prefix)
                    && let Some(bracket_pos) = rest.find(']')
                    && let Ok(tick_idx) = rest[..bracket_pos].parse::<usize>()
                    && rest.len() > bracket_pos + 1
                    && rest.as_bytes()[bracket_pos + 1] == b'.'
                {
                    let attr_name = &rest[bracket_pos + 2..];
                    if let Some(tick) = tc.get_tick_mut(tick_idx) {
                        tick.set_attr(attr_name, value.clone());
                    }
                }
            } else {
                tc.set_meta(key, value.clone());
            }
        }

        tc
    }
}

impl From<DagCircuit> for TickCircuit {
    fn from(dag: DagCircuit) -> Self {
        TickCircuit::from(&dag)
    }
}

impl From<&TickCircuit> for DagCircuit {
    /// Convert a `TickCircuit` to a `DagCircuit`.
    ///
    /// Gates are added in tick order, with qubit wires connecting
    /// consecutive gates on the same qubit.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::{DagCircuit, TickCircuit};
    ///
    /// let mut tc = TickCircuit::new();
    /// tc.tick().h(0);
    /// tc.tick().cx(0, 1);
    ///
    /// let dag = DagCircuit::from(&tc);
    /// assert_eq!(dag.gate_count(), 2);
    /// assert_eq!(dag.wire_count(), 1); // H->CX on qubit 0
    /// ```
    fn from(tc: &TickCircuit) -> Self {
        let mut dag = DagCircuit::new();

        // Track the last node for each qubit to connect wires
        let mut last_node: BTreeMap<QubitId, usize> = BTreeMap::new();

        for (tick_idx, tick) in tc.ticks().iter().enumerate() {
            for (gate_idx, gate) in tick.gates().iter().enumerate() {
                let node = dag.add_gate(gate.clone());

                // Connect wires from previous gates on the same qubits
                for qubit in &gate.qubits {
                    if let Some(&prev_node) = last_node.get(qubit) {
                        // Connect previous node to this one on this qubit
                        let _ = dag.connect(prev_node, node, *qubit);
                    }
                    last_node.insert(*qubit, node);
                }

                // Copy gate attributes
                for (key, value) in tick.gate_attrs(gate_idx) {
                    dag.set_gate_attr(node, key, value.clone());
                }
            }

            // Store tick-level attributes as circuit-level with tick[N].key prefix
            for (key, value) in tick.tick_attrs() {
                let prefixed_key = format!("tick[{tick_idx}].{key}");
                dag.set_attr(prefixed_key, value.clone());
            }
        }

        // Copy circuit-level attributes
        for (key, value) in tc.circuit_attrs() {
            dag.set_attr(key.clone(), value.clone());
        }

        dag
    }
}

impl From<TickCircuit> for DagCircuit {
    fn from(tc: TickCircuit) -> Self {
        DagCircuit::from(&tc)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_tick_circuit() {
        let mut tc = TickCircuit::new();

        // Preps and measurements break the chain (return handles with only .meta())
        // For multiple preps in the same tick, add gates directly to the tick
        tc.tick().pz(0); // tick 0: first prep
        // If we want both preps in tick 0, we'd use the Tick API directly.
        // Here we use separate ticks for clarity:
        tc.tick().pz(1); // tick 1: second prep
        tc.tick().h(0); // tick 2
        tc.tick().cx(0, 1); // tick 3
        tc.tick().mz(0); // tick 4
        tc.tick().mz(1); // tick 5

        assert_eq!(tc.num_ticks(), 6);
        assert_eq!(tc.gate_count(), 6);
    }

    #[test]
    fn test_multiple_preps_same_tick() {
        let mut tc = TickCircuit::new();

        // To add multiple preps to the same tick, allocate the tick first
        // then add gates directly
        let tick_idx = tc.tick().index();
        // tick() consumed the handle by calling index(), so get the tick
        tc.get_tick_mut(tick_idx)
            .unwrap()
            .add_gate(Gate::prep(&[0]));
        tc.get_tick_mut(tick_idx)
            .unwrap()
            .add_gate(Gate::prep(&[1]));

        tc.tick().h(0);
        tc.tick().cx(0, 1);

        // Multiple measurements in same tick
        let meas_tick = tc.tick().index();
        tc.get_tick_mut(meas_tick)
            .unwrap()
            .add_gate(Gate::measure(&[0]));
        tc.get_tick_mut(meas_tick)
            .unwrap()
            .add_gate(Gate::measure(&[1]));

        assert_eq!(tc.num_ticks(), 4);
        assert_eq!(tc.gate_count(), 6);

        // Check tick contents
        assert_eq!(tc.get_tick(0).unwrap().len(), 2); // Two preps
        assert_eq!(tc.get_tick(1).unwrap().len(), 1); // One H
        assert_eq!(tc.get_tick(2).unwrap().len(), 1); // One CX
        assert_eq!(tc.get_tick(3).unwrap().len(), 2); // Two measurements
    }

    #[test]
    fn test_meta_on_gates() {
        let mut tc = TickCircuit::new();

        tc.tick()
            .h(0)
            .meta("duration", Attribute::Float(50.0))
            .meta("error_rate", Attribute::Float(0.001))
            .x(1)
            .meta("duration", Attribute::Float(50.0));

        let tick = tc.get_tick(0).unwrap();
        assert_eq!(
            tick.get_gate_attr(0, "duration"),
            Some(&Attribute::Float(50.0))
        );
        assert_eq!(
            tick.get_gate_attr(0, "error_rate"),
            Some(&Attribute::Float(0.001))
        );
        assert_eq!(
            tick.get_gate_attr(1, "duration"),
            Some(&Attribute::Float(50.0))
        );
    }

    #[test]
    fn test_tick_meta() {
        let mut tc = TickCircuit::new();

        // meta() before any gates = tick-level metadata
        tc.tick().meta("round", Attribute::Int(0)).h(0);
        tc.tick().meta("round", Attribute::Int(1)).cx(0, 1);

        assert_eq!(
            tc.get_tick(0).unwrap().get_attr("round"),
            Some(&Attribute::Int(0))
        );
        assert_eq!(
            tc.get_tick(1).unwrap().get_attr("round"),
            Some(&Attribute::Int(1))
        );
    }

    #[test]
    fn test_tick_index() {
        let mut tc = TickCircuit::new();

        let t0 = tc.tick();
        assert_eq!(t0.index(), 0);

        let t1 = tc.tick();
        assert_eq!(t1.index(), 1);

        assert_eq!(tc.next_tick_index(), 2);
    }

    #[test]
    fn test_gates_chain_but_preps_and_meas_break() {
        let mut tc = TickCircuit::new();

        // Regular gates chain within a tick
        tc.tick().h(0).x(1).y(2).z(3);
        tc.tick().cx(0, 1).szz(2, 3);

        // But preps and measurements break the chain
        tc.tick().pz(0); // breaks chain
        tc.tick().mz(0); // breaks chain

        assert_eq!(tc.num_ticks(), 4);
        assert_eq!(tc.gate_count(), 8);
    }

    #[test]
    fn test_prep_and_meas_with_meta() {
        let mut tc = TickCircuit::new();

        // Preps and measurements allow .meta() before breaking
        tc.tick()
            .pz(0)
            .meta("reason", Attribute::String("init".into()));
        tc.tick().h(0);
        tc.tick().mz(0).meta("basis", Attribute::String("Z".into()));

        assert_eq!(tc.num_ticks(), 3);

        // Check that metadata was attached
        assert_eq!(
            tc.get_tick(0).unwrap().get_gate_attr(0, "reason"),
            Some(&Attribute::String("init".into()))
        );
        assert_eq!(
            tc.get_tick(2).unwrap().get_gate_attr(0, "basis"),
            Some(&Attribute::String("Z".into()))
        );
    }

    #[test]
    fn test_circuit_meta() {
        let mut tc = TickCircuit::new();
        tc.set_meta("name", Attribute::String("bell_state".to_string()));
        tc.tick().h(0);

        assert_eq!(
            tc.get_meta("name"),
            Some(&Attribute::String("bell_state".to_string()))
        );
    }

    #[test]
    fn test_tick_circuit_to_dag_circuit() {
        let mut tc = TickCircuit::new();
        tc.set_meta("circuit_name", Attribute::String("test".to_string()));

        // Build a small circuit
        tc.tick().h(0).x(1); // Tick 0: parallel H and X
        tc.tick().cx(0, 1); // Tick 1: CX
        tc.tick().h(0); // Tick 2: H

        let dag = DagCircuit::from(&tc);

        // Check gate counts
        assert_eq!(dag.gate_count(), 4);

        // Check wires: H(0)->CX->H(0), X(1)->CX
        // So we have 3 wires: H(0)->CX, CX->H(0), X(1)->CX
        assert_eq!(dag.wire_count(), 3);

        // Check circuit attributes
        assert_eq!(
            dag.get_attr("circuit_name"),
            Some(&Attribute::String("test".to_string()))
        );
    }

    #[test]
    fn test_dag_circuit_to_tick_circuit() {
        let mut dag = DagCircuit::new();
        dag.set_attr("version".to_string(), Attribute::Int(1));

        // Build: H(0) -> CX(0,1)
        //        X(1) ----^
        let h = dag.add_gate(Gate::h(&[0]));
        let x = dag.add_gate(Gate::x(&[1]));
        let cx = dag.add_gate(Gate::cx(&[(0, 1)]));

        dag.connect(h, cx, QubitId::from(0)).unwrap();
        dag.connect(x, cx, QubitId::from(1)).unwrap();

        let tc = TickCircuit::from(&dag);

        // H and X are parallel (layer 0), CX depends on both (layer 1)
        assert_eq!(tc.num_ticks(), 2);

        // First tick should have H and X (order may vary)
        let tick0 = tc.get_tick(0).unwrap();
        assert_eq!(tick0.gates().len(), 2);

        // Second tick should have CX
        let tick1 = tc.get_tick(1).unwrap();
        assert_eq!(tick1.gates().len(), 1);

        // Check circuit attribute
        assert_eq!(tc.get_meta("version"), Some(&Attribute::Int(1)));
    }

    #[test]
    fn test_round_trip_tick_to_dag_to_tick() {
        let mut tc1 = TickCircuit::new();
        tc1.tick().h(0);
        tc1.tick().cx(0, 1);
        tc1.tick().h(1);

        // Convert to DAG and back
        let dag = DagCircuit::from(&tc1);
        let tc2 = TickCircuit::from(&dag);

        // Should have same structure
        assert_eq!(tc1.num_ticks(), tc2.num_ticks());
        for i in 0..tc1.num_ticks() {
            assert_eq!(
                tc1.get_tick(i).unwrap().gates().len(),
                tc2.get_tick(i).unwrap().gates().len()
            );
        }
    }

    #[test]
    fn test_tick_attrs_preserved_in_conversion() {
        let mut tc1 = TickCircuit::new();
        tc1.set_meta("circuit_name", Attribute::String("test".to_string()));

        // Add tick-level metadata
        tc1.tick()
            .meta("round", Attribute::Int(0))
            .meta("syndrome_type", Attribute::String("X".to_string()))
            .h(0);
        tc1.tick().meta("round", Attribute::Int(1)).cx(0, 1);

        // Convert to DAG
        let dag = DagCircuit::from(&tc1);

        // Check tick-level attrs are stored with prefix
        assert_eq!(dag.get_attr("tick[0].round"), Some(&Attribute::Int(0)));
        assert_eq!(
            dag.get_attr("tick[0].syndrome_type"),
            Some(&Attribute::String("X".to_string()))
        );
        assert_eq!(dag.get_attr("tick[1].round"), Some(&Attribute::Int(1)));
        // Circuit-level attr preserved without prefix
        assert_eq!(
            dag.get_attr("circuit_name"),
            Some(&Attribute::String("test".to_string()))
        );

        // Convert back to TickCircuit
        let tc2 = TickCircuit::from(&dag);

        // Check tick-level attrs are restored
        assert_eq!(
            tc2.get_tick(0).unwrap().get_attr("round"),
            Some(&Attribute::Int(0))
        );
        assert_eq!(
            tc2.get_tick(0).unwrap().get_attr("syndrome_type"),
            Some(&Attribute::String("X".to_string()))
        );
        assert_eq!(
            tc2.get_tick(1).unwrap().get_attr("round"),
            Some(&Attribute::Int(1))
        );
        // Circuit-level attr preserved
        assert_eq!(
            tc2.get_meta("circuit_name"),
            Some(&Attribute::String("test".to_string()))
        );
    }

    #[test]
    fn test_active_qubits() {
        let mut tc = TickCircuit::new();
        tc.tick().h(0).x(1).cx(2, 3);

        let tick = tc.get_tick(0).unwrap();
        let active = tick.active_qubits();

        assert_eq!(active.len(), 4);
        assert!(active.contains(&QubitId::from(0)));
        assert!(active.contains(&QubitId::from(1)));
        assert!(active.contains(&QubitId::from(2)));
        assert!(active.contains(&QubitId::from(3)));
        assert!(!active.contains(&QubitId::from(4)));
    }

    #[test]
    fn test_uses_qubit() {
        let mut tc = TickCircuit::new();
        tc.tick().h(0).cx(1, 2);

        let tick = tc.get_tick(0).unwrap();

        assert!(tick.uses_qubit(QubitId::from(0)));
        assert!(tick.uses_qubit(QubitId::from(1)));
        assert!(tick.uses_qubit(QubitId::from(2)));
        assert!(!tick.uses_qubit(QubitId::from(3)));
    }

    #[test]
    fn test_find_conflicts() {
        let mut tc = TickCircuit::new();
        tc.tick().h(0).cx(1, 2);

        let tick = tc.get_tick(0).unwrap();

        // No conflict
        assert!(
            tick.find_conflicts(&[QubitId::from(3), QubitId::from(4)])
                .is_empty()
        );

        // Conflict with qubit 0
        let conflicts = tick.find_conflicts(&[QubitId::from(0), QubitId::from(5)]);
        assert_eq!(conflicts, vec![QubitId::from(0)]);

        // Multiple conflicts
        let conflicts = tick.find_conflicts(&[QubitId::from(0), QubitId::from(2)]);
        assert_eq!(conflicts.len(), 2);
    }

    #[test]
    fn test_try_add_gate_success() {
        let mut tc = TickCircuit::new();
        let mut handle = tc.tick();

        // Add first gate
        assert!(handle.try_add_gate(Gate::h(&[0])).is_ok());

        // Add another gate on different qubit - should succeed
        assert!(handle.try_add_gate(Gate::x(&[1])).is_ok());
    }

    #[test]
    fn test_try_add_gate_conflict() {
        let mut tc = TickCircuit::new();
        let mut handle = tc.tick();
        handle.h(0);

        // Try to add another gate on the same qubit - should fail
        let result = handle.try_add_gate(Gate::x(&[0]));

        match result {
            Err(err) => {
                assert_eq!(err.conflicting_qubits, vec![QubitId::from(0)]);
                assert_eq!(err.tick_idx, Some(0));
            }
            Ok(_) => panic!("Expected conflict error"),
        }
    }

    #[test]
    #[should_panic(expected = "Qubit(s) 0 already in use in tick 0")]
    fn test_qubit_conflict_panics() {
        let mut tc = TickCircuit::new();
        // This should panic because qubit 0 is used twice in the same tick
        tc.tick().h(0).x(0);
    }

    #[test]
    fn test_two_qubit_gate_conflict() {
        let mut tc = TickCircuit::new();
        let mut handle = tc.tick();
        handle.cx(0, 1);

        // Both qubits of CX should be marked as in use
        let result = handle.try_add_gate(Gate::h(&[0]));
        assert!(result.is_err());

        let mut handle2 = tc.tick();
        handle2.cx(2, 3);
        let result = handle2.try_add_gate(Gate::h(&[3]));
        assert!(result.is_err());
    }

    #[test]
    fn test_metas_on_gates() {
        let mut tc = TickCircuit::new();

        let attrs = BTreeMap::from([
            ("duration".to_string(), Attribute::Float(50.0)),
            ("error_rate".to_string(), Attribute::Float(0.001)),
        ]);

        tc.tick().h(0).metas(attrs).x(1);

        let tick = tc.get_tick(0).unwrap();
        assert_eq!(
            tick.get_gate_attr(0, "duration"),
            Some(&Attribute::Float(50.0))
        );
        assert_eq!(
            tick.get_gate_attr(0, "error_rate"),
            Some(&Attribute::Float(0.001))
        );
    }

    #[test]
    fn test_metas_on_tick() {
        let mut tc = TickCircuit::new();

        let attrs = BTreeMap::from([
            ("round".to_string(), Attribute::Int(0)),
            ("syndrome".to_string(), Attribute::String("X".to_string())),
        ]);

        // metas() before any gates = tick-level metadata
        tc.tick().metas(attrs).h(0);

        let tick = tc.get_tick(0).unwrap();
        assert_eq!(tick.get_attr("round"), Some(&Attribute::Int(0)));
        assert_eq!(
            tick.get_attr("syndrome"),
            Some(&Attribute::String("X".to_string()))
        );
    }

    #[test]
    fn test_set_metas_on_circuit() {
        let mut tc = TickCircuit::new();

        let attrs = BTreeMap::from([
            ("name".to_string(), Attribute::String("bell".to_string())),
            ("version".to_string(), Attribute::Int(1)),
        ]);

        tc.set_metas(attrs);
        tc.tick().h(0);

        assert_eq!(
            tc.get_meta("name"),
            Some(&Attribute::String("bell".to_string()))
        );
        assert_eq!(tc.get_meta("version"), Some(&Attribute::Int(1)));
    }
}
