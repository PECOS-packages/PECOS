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
//! circuit.tick().pz(&[0]);              // Tick 0: Prepare q0 (breaks chain)
//! circuit.tick().pz(&[1]);              // Tick 1: Prepare q1 (breaks chain)
//! circuit.tick().h(&[0]).x(&[1]);       // Tick 2: H on q0, X on q1 (chains!)
//! circuit.tick().cx(&[(0, 1)]);         // Tick 3: CNOT
//! circuit.tick().mz(&[0]);              // Tick 4: Measure q0 (breaks chain)
//! circuit.tick().mz(&[1]);              // Tick 5: Measure q1 (breaks chain)
//!
//! assert_eq!(circuit.num_ticks(), 6);
//!
//! // Preps and measurements break the chain but allow .meta():
//! circuit.tick().pz(&[0]).meta("reason", pecos_quantum::Attribute::String("init".into()));
//! circuit.tick().mz(&[0]).meta("basis", pecos_quantum::Attribute::String("Z".into()));
//!
//! // Tick-level metadata: call meta() before adding gates
//! use pecos_quantum::Attribute;
//! let mut circuit2 = TickCircuit::new();
//! circuit2.tick()
//!     .meta("round", Attribute::Int(0))    // Tick metadata (no gates added yet)
//!     .h(&[0])
//!     .meta("duration", Attribute::Float(50.0)); // Gate metadata (after a gate)
//!
//! // Bulk operations - apply gates to multiple qubits at once:
//! let mut circuit3 = TickCircuit::new();
//! circuit3.tick().pz(&[0, 1, 2, 3]);      // Prep 4 qubits at once
//! circuit3.tick().h(&[0, 1, 2, 3]);       // H on 4 qubits at once
//! circuit3.tick().cx(&[(0, 1), (2, 3)]);  // 2 CX gates in parallel
//! circuit3.tick().mz(&[0, 1, 2, 3]);      // Measure all 4 qubits
//! ```

use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, Gate, GateQubits, GateSignature, Nanoseconds, QubitId};
use std::collections::{BTreeMap, BTreeSet, HashMap};

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

/// Error when a custom gate is used with a different signature than previously established.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GateSignatureMismatchError {
    pub name: String,
    pub expected_quantum_arity: usize,
    pub actual_quantum_arity: usize,
    pub expected_angle_arity: usize,
    pub actual_angle_arity: usize,
}

impl fmt::Display for GateSignatureMismatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Gate '{}' signature mismatch: expected ({} qubits, {} angles), got ({} qubits, {} angles)",
            self.name,
            self.expected_quantum_arity,
            self.expected_angle_arity,
            self.actual_quantum_arity,
            self.actual_angle_arity,
        )
    }
}

impl std::error::Error for GateSignatureMismatchError {}

/// Error when adding a custom gate to a tick.
#[derive(Debug, Clone)]
pub enum CustomGateError {
    SignatureMismatch(GateSignatureMismatchError),
    QubitConflict(QubitConflictError),
}

impl fmt::Display for CustomGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SignatureMismatch(e) => write!(f, "{e}"),
            Self::QubitConflict(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for CustomGateError {}

impl From<GateSignatureMismatchError> for CustomGateError {
    fn from(e: GateSignatureMismatchError) -> Self {
        Self::SignatureMismatch(e)
    }
}

impl From<QubitConflictError> for CustomGateError {
    fn from(e: QubitConflictError) -> Self {
        Self::QubitConflict(e)
    }
}

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

    /// Get mutable access to the gates in this tick.
    pub fn gates_mut(&mut self) -> &mut [Gate] {
        &mut self.gates
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

    /// Remove all gates that use any of the specified qubits.
    ///
    /// Returns the number of gates removed.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::{TickCircuit, QubitId};
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0]).x(&[1]).cx(&[(2, 3)]);
    ///
    /// let tick = circuit.get_tick_mut(0).unwrap();
    /// let removed = tick.discard(&[QubitId::from(0), QubitId::from(2)]);
    ///
    /// assert_eq!(removed, 2);  // H on q0 and CX on q2,q3 removed
    /// assert_eq!(tick.len(), 1);  // Only X on q1 remains
    /// ```
    pub fn discard(&mut self, qubits: &[QubitId]) -> usize {
        let qubits_set: BTreeSet<_> = qubits.iter().copied().collect();

        // Find indices of gates to remove (those using any of the specified qubits)
        let indices_to_remove: Vec<usize> = self
            .gates
            .iter()
            .enumerate()
            .filter(|(_, gate)| gate.qubits.iter().any(|q| qubits_set.contains(q)))
            .map(|(idx, _)| idx)
            .collect();

        let removed_count = indices_to_remove.len();

        if removed_count == 0 {
            return 0;
        }

        // Remove gates in reverse order to preserve indices
        for &idx in indices_to_remove.iter().rev() {
            self.gates.remove(idx);
        }

        // Rebuild gate_attrs with updated indices
        let old_attrs = std::mem::take(&mut self.gate_attrs);
        for (old_idx, attrs) in old_attrs {
            // Count how many removed indices are before this one
            let shift = indices_to_remove.iter().filter(|&&i| i < old_idx).count();
            if !indices_to_remove.contains(&old_idx) {
                self.gate_attrs.insert(old_idx - shift, attrs);
            }
        }

        removed_count
    }

    /// Remove a specific gate by index.
    ///
    /// Returns the removed gate, or `None` if the index is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0]).x(&[1]).z(&[2]);
    ///
    /// let tick = circuit.get_tick_mut(0).unwrap();
    /// let removed = tick.remove_gate(1);  // Remove X gate
    ///
    /// assert!(removed.is_some());
    /// assert_eq!(tick.len(), 2);  // H and Z remain
    /// ```
    pub fn remove_gate(&mut self, idx: usize) -> Option<Gate> {
        if idx >= self.gates.len() {
            return None;
        }

        let gate = self.gates.remove(idx);

        // Rebuild gate_attrs with updated indices
        let old_attrs = std::mem::take(&mut self.gate_attrs);
        for (old_idx, attrs) in old_attrs {
            if old_idx < idx {
                self.gate_attrs.insert(old_idx, attrs);
            } else if old_idx > idx {
                self.gate_attrs.insert(old_idx - 1, attrs);
            }
            // Skip old_idx == idx (removed gate's attrs)
        }

        Some(gate)
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
/// circuit.tick().pz(&[0]);                   // Tick 0: Prepare q0 (breaks chain)
/// circuit.tick().pz(&[1]);                   // Tick 1: Prepare q1 (breaks chain)
/// circuit.tick().h(&[0]).x(&[1]);            // Tick 2: H and X (chains!)
/// circuit.tick().cx(&[(0, 1)]);              // Tick 3: CNOT
/// circuit.tick().mz(&[0]);                   // Tick 4: Measure q0 (breaks chain)
/// circuit.tick().mz(&[1]);                   // Tick 5: Measure q1 (breaks chain)
///
/// assert_eq!(circuit.num_ticks(), 6);
///
/// // Bulk operations - all methods accept slices:
/// circuit.tick().pz(&[0, 1, 2, 3]);          // Prep multiple qubits
/// circuit.tick().h(&[0, 1, 2, 3]);           // H on multiple qubits
/// circuit.tick().cx(&[(0, 1), (2, 3)]);      // Multiple CX gates
/// circuit.tick().mz(&[0, 1, 2, 3]);          // Measure multiple qubits
/// ```
#[derive(Debug, Clone, Default)]
pub struct TickCircuit {
    /// The sequence of ticks.
    ticks: Vec<Tick>,
    /// Next tick index to allocate.
    next_tick: usize,
    /// Circuit-level metadata.
    circuit_attrs: BTreeMap<String, Attribute>,
    /// Gate signatures for custom gate validation (JIT + AOT).
    gate_signatures: HashMap<String, GateSignature>,
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
            gate_signatures: HashMap::new(),
        }
    }

    /// Get the number of ticks in the circuit.
    #[must_use]
    pub fn num_ticks(&self) -> usize {
        self.ticks.len()
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

    /// Get mutable access to all ticks.
    pub fn ticks_mut(&mut self) -> &mut [Tick] {
        &mut self.ticks
    }

    /// Export as a plain ASCII circuit diagram.
    ///
    /// Produces horizontal qubit-wire lines with gate symbols placed at each
    /// tick column. Two-qubit gates show `.`/`[X]` with `|` connectors.
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

    fn diagram_parts(&self) -> (String, Vec<Vec<&pecos_core::Gate>>) {
        let layers: Vec<Vec<&pecos_core::Gate>> = self
            .ticks
            .iter()
            .map(|t| t.gates().iter().collect())
            .collect();
        let num_qubits = self.all_qubits().len();
        let header = format!(
            "TickCircuit: {} qubit{}, {} tick{}",
            num_qubits,
            if num_qubits == 1 { "" } else { "s" },
            self.ticks.len(),
            if self.ticks.len() == 1 { "" } else { "s" },
        );
        (header, layers)
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

    // =========================================================================
    // Circuit manipulation
    // =========================================================================

    /// Clear the circuit and start fresh.
    ///
    /// This completely replaces the circuit with a new empty instance,
    /// releasing any allocated memory. Use this when memory usage is a concern
    /// or when you want absolute certainty of a fresh state.
    ///
    /// For performance-critical code or when creating many circuits in sequence,
    /// consider using [`reset()`](Self::reset) instead, which preserves memory allocation.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0]);
    /// assert_eq!(circuit.num_ticks(), 1);
    ///
    /// circuit.clear();
    /// assert_eq!(circuit.num_ticks(), 0);
    /// ```
    pub fn clear(&mut self) {
        *self = Self::new();
    }

    /// Reset the circuit state while preserving allocated memory.
    ///
    /// Unlike [`clear()`](Self::clear), this method preserves the allocated memory
    /// for better performance when reusing the same circuit multiple times.
    /// This is the recommended method for performance-critical code,
    /// especially when creating many circuits in sequence.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    ///
    /// // Build first circuit
    /// circuit.tick().h(&[0]);
    /// circuit.tick().cx(&[(0, 1)]);
    /// assert_eq!(circuit.num_ticks(), 2);
    ///
    /// // Reset and build another circuit (memory preserved)
    /// circuit.reset();
    /// assert_eq!(circuit.num_ticks(), 0);
    ///
    /// circuit.tick().x(&[0]);
    /// assert_eq!(circuit.num_ticks(), 1);
    /// ```
    pub fn reset(&mut self) {
        self.ticks.clear();
        self.next_tick = 0;
        self.circuit_attrs.clear();
        self.gate_signatures.clear();
    }

    /// Reserve empty ticks in advance.
    ///
    /// This preallocates `n` empty ticks, which can be useful when you know
    /// the circuit structure ahead of time.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.reserve_ticks(4);
    ///
    /// // Ticks 0-3 are now available (though empty)
    /// assert_eq!(circuit.ticks().len(), 4);
    ///
    /// // tick() will start from tick 4
    /// circuit.tick().h(&[0]);
    /// assert_eq!(circuit.num_ticks(), 5);
    /// ```
    pub fn reserve_ticks(&mut self, n: usize) {
        let target_len = self.ticks.len() + n;
        self.ticks.reserve(n);
        while self.ticks.len() < target_len {
            self.ticks.push(Tick::new());
        }
        self.next_tick = self.ticks.len();
    }

    /// Insert an empty tick at a specific position.
    ///
    /// All ticks at or after `idx` are shifted to the right.
    /// Returns a [`TickHandle`] to the newly inserted tick.
    ///
    /// # Panics
    ///
    /// Panics if `idx > self.ticks().len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0]);      // Tick 0
    /// circuit.tick().cx(&[(0, 1)]); // Tick 1
    ///
    /// // Insert a new tick between them
    /// circuit.insert_tick(1).x(&[1]);
    ///
    /// // Now: H at tick 0, X at tick 1, CX at tick 2
    /// assert_eq!(circuit.num_ticks(), 3);
    /// ```
    pub fn insert_tick(&mut self, idx: usize) -> TickHandle<'_> {
        assert!(
            idx <= self.ticks.len(),
            "insert_tick index {} out of bounds for circuit with {} ticks",
            idx,
            self.ticks.len()
        );

        self.ticks.insert(idx, Tick::new());
        self.next_tick = self.ticks.len();

        TickHandle {
            circuit: self,
            tick_idx: idx,
            last_gate_idx: None,
        }
    }

    /// Get a handle to an existing tick for adding more gates.
    ///
    /// This allows adding gates to a tick that was previously created,
    /// which is useful when building circuits non-sequentially.
    ///
    /// # Panics
    ///
    /// Panics if `idx >= self.ticks().len()`.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.reserve_ticks(3);
    ///
    /// // Add gates to specific ticks
    /// circuit.tick_at(0).h(&[0]);
    /// circuit.tick_at(2).cx(&[(0, 1)]);
    /// circuit.tick_at(1).x(&[1]);  // Fill in the middle later
    ///
    /// assert_eq!(circuit.num_ticks(), 3);
    /// ```
    pub fn tick_at(&mut self, idx: usize) -> TickHandle<'_> {
        assert!(
            idx < self.ticks.len(),
            "tick_at index {} out of bounds for circuit with {} ticks",
            idx,
            self.ticks.len()
        );

        TickHandle {
            circuit: self,
            tick_idx: idx,
            last_gate_idx: None,
        }
    }

    /// Remove all gates that use any of the specified qubits from a tick.
    ///
    /// Returns the number of gates removed, or `None` if the tick index is out of bounds.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0]).x(&[1]).cx(&[(2, 3)]);
    ///
    /// let removed = circuit.discard(&[0, 2], 0);
    /// assert_eq!(removed, Some(2));  // H on q0 and CX on q2,q3 removed
    /// assert_eq!(circuit.get_tick(0).unwrap().len(), 1);  // Only X remains
    /// ```
    pub fn discard(
        &mut self,
        qubits: &[impl Into<QubitId> + Copy],
        tick_idx: usize,
    ) -> Option<usize> {
        let qubit_ids: Vec<QubitId> = qubits.iter().map(|&q| q.into()).collect();
        self.get_tick_mut(tick_idx)
            .map(|tick| tick.discard(&qubit_ids))
    }

    // =========================================================================
    // Gate signature validation
    // =========================================================================

    /// Import gate signatures in bulk (e.g., from a `GateRegistry`).
    pub fn import_signatures(&mut self, sigs: &HashMap<String, GateSignature>) {
        self.gate_signatures
            .extend(sigs.iter().map(|(name, sig)| (name.clone(), sig.clone())));
    }

    /// Get read access to the gate signatures.
    #[must_use]
    pub fn gate_signatures(&self) -> &HashMap<String, GateSignature> {
        &self.gate_signatures
    }

    /// Validate a custom gate against its previously established signature,
    /// or register it if this is the first use.
    ///
    /// # Errors
    ///
    /// Returns `GateSignatureMismatchError` if the gate has been seen before
    /// with a different quantum or angle arity.
    pub fn validate_or_register_gate(
        &mut self,
        name: &str,
        quantum_arity: usize,
        angle_arity: usize,
    ) -> Result<(), GateSignatureMismatchError> {
        if let Some(existing) = self.gate_signatures.get(name) {
            if existing.quantum_arity != quantum_arity || existing.angle_arity != angle_arity {
                return Err(GateSignatureMismatchError {
                    name: name.to_string(),
                    expected_quantum_arity: existing.quantum_arity,
                    actual_quantum_arity: quantum_arity,
                    expected_angle_arity: existing.angle_arity,
                    actual_angle_arity: angle_arity,
                });
            }
        } else {
            self.gate_signatures.insert(
                name.to_string(),
                GateSignature {
                    quantum_arity,
                    angle_arity,
                },
            );
        }
        Ok(())
    }

    // =========================================================================
    // Iteration helpers
    // =========================================================================

    /// Iterate over all gates in the circuit, across all ticks.
    ///
    /// Gates are yielded in tick order, then in order within each tick.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0, 1]);
    /// circuit.tick().cx(&[(0, 1)]);
    ///
    /// for gate in circuit.iter_gates() {
    ///     println!("{:?} on {:?}", gate.gate_type, gate.qubits);
    /// }
    /// ```
    pub fn iter_gates(&self) -> impl Iterator<Item = &Gate> {
        self.ticks.iter().flat_map(Tick::gates)
    }

    /// Iterate over all gates with their tick index.
    ///
    /// Yields `(tick_index, gate)` pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0]);
    /// circuit.tick().x(&[0]);
    ///
    /// for (tick_idx, gate) in circuit.iter_gates_with_tick() {
    ///     println!("Tick {}: {:?}", tick_idx, gate.gate_type);
    /// }
    /// ```
    pub fn iter_gates_with_tick(&self) -> impl Iterator<Item = (usize, &Gate)> {
        self.ticks
            .iter()
            .enumerate()
            .flat_map(|(tick_idx, tick)| tick.gates().iter().map(move |gate| (tick_idx, gate)))
    }

    /// Iterate over ticks with their index.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0, 1, 2]);
    /// circuit.tick().cx(&[(0, 1), (1, 2)]);
    ///
    /// for (tick_idx, tick) in circuit.iter_ticks() {
    ///     println!("Tick {} has {} gates", tick_idx, tick.len());
    /// }
    /// ```
    pub fn iter_ticks(&self) -> impl Iterator<Item = (usize, &Tick)> {
        self.ticks.iter().enumerate()
    }

    /// Iterate over gates filtered by gate type.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    /// use pecos_core::gate_type::GateType;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0, 1, 2]);
    /// circuit.tick().x(&[0]).cx(&[(1, 2)]);
    ///
    /// // Get all H gates
    /// let h_gates: Vec<_> = circuit.iter_gates_by_type(GateType::H).collect();
    /// assert_eq!(h_gates.len(), 1);  // One Gate object with 3 qubits
    /// ```
    pub fn iter_gates_by_type(&self, gate_type: GateType) -> impl Iterator<Item = &Gate> {
        self.iter_gates().filter(move |g| g.gate_type == gate_type)
    }

    /// Get all qubits used in the circuit.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0, 1, 2]);
    /// circuit.tick().cx(&[(0, 1)]);
    ///
    /// let qubits = circuit.all_qubits();
    /// assert_eq!(qubits.len(), 3);
    /// ```
    #[must_use]
    pub fn all_qubits(&self) -> BTreeSet<QubitId> {
        self.iter_gates()
            .flat_map(|gate| gate.qubits.iter().copied())
            .collect()
    }

    /// Count gates by type across the entire circuit.
    ///
    /// Returns a map from `GateType` to count.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    /// use pecos_core::gate_type::GateType;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0, 1, 2]);
    /// circuit.tick().cx(&[(0, 1), (1, 2)]);
    ///
    /// let counts = circuit.gate_counts_by_type();
    /// assert_eq!(counts.get(&GateType::H), Some(&1));  // 1 H gate object (with 3 qubits)
    /// assert_eq!(counts.get(&GateType::CX), Some(&1)); // 1 CX gate object (with 2 pairs)
    /// ```
    #[must_use]
    pub fn gate_counts_by_type(&self) -> BTreeMap<GateType, usize> {
        let mut counts = BTreeMap::new();
        for gate in self.iter_gates() {
            *counts.entry(gate.gate_type).or_insert(0) += 1;
        }
        counts
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

    /// Apply Hadamard gate(s) to one or more qubits.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// // Single qubit
    /// circuit.tick().h(&[0]);
    /// // Multiple qubits in one call
    /// circuit.tick().h(&[1, 2, 3]);
    /// ```
    pub fn h(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::h(qubits))
    }

    /// Apply Pauli-X gate(s) to one or more qubits.
    pub fn x(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::x(qubits))
    }

    /// Apply Pauli-Y gate(s) to one or more qubits.
    pub fn y(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::y(qubits))
    }

    /// Apply Pauli-Z gate(s) to one or more qubits.
    pub fn z(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::z(qubits))
    }

    /// Apply identity gate(s) to one or more qubits.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().iden(&[0]);           // Single qubit
    /// circuit.tick().iden(&[1, 2, 3]);     // Multiple qubits
    /// ```
    pub fn iden(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::i(qubits))
    }

    /// Apply SX gate(s) (sqrt-X) to one or more qubits.
    pub fn sx(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::sx(qubits))
    }

    /// Apply SX-dagger gate(s) to one or more qubits.
    pub fn sxdg(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::sxdg(qubits))
    }

    /// Apply SY gate(s) (sqrt-Y) to one or more qubits.
    pub fn sy(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::sy(qubits))
    }

    /// Apply SY-dagger gate(s) to one or more qubits.
    pub fn sydg(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::sydg(qubits))
    }

    /// Apply SZ gate(s) (sqrt-Z) to one or more qubits.
    pub fn sz(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::sz(qubits))
    }

    /// Apply SZ-dagger gate(s) to one or more qubits.
    pub fn szdg(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::szdg(qubits))
    }

    /// Apply F gate(s) to one or more qubits.
    pub fn f(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::f(qubits))
    }

    /// Apply F-dagger gate(s) to one or more qubits.
    pub fn fdg(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::fdg(qubits))
    }

    /// Apply T gate(s) to one or more qubits.
    pub fn t(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::t(qubits))
    }

    /// Apply T-dagger gate(s) to one or more qubits.
    pub fn tdg(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::tdg(qubits))
    }

    /// Apply RX rotation(s) to one or more qubits.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = TickCircuit::new();
    /// // Single qubit
    /// circuit.tick().rx(PI / 2.0, &[0]);
    /// // Multiple qubits with same angle
    /// circuit.tick().rx(PI / 4.0, &[1, 2, 3]);
    /// ```
    pub fn rx(
        &mut self,
        theta: impl Into<Angle64>,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> &mut Self {
        self.add_gate(Gate::rx(theta.into(), qubits))
    }

    /// Apply RY rotation(s) to one or more qubits.
    pub fn ry(
        &mut self,
        theta: impl Into<Angle64>,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> &mut Self {
        self.add_gate(Gate::ry(theta.into(), qubits))
    }

    /// Apply RZ rotation(s) to one or more qubits.
    pub fn rz(
        &mut self,
        theta: impl Into<Angle64>,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> &mut Self {
        self.add_gate(Gate::rz(theta.into(), qubits))
    }

    /// Apply R1XY rotation(s) to one or more qubits.
    ///
    /// This is a single-qubit rotation parameterized by two angles (theta, phi).
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().r1xy(PI / 2.0, PI / 4.0, &[0]);
    /// ```
    pub fn r1xy(
        &mut self,
        theta: impl Into<Angle64>,
        phi: impl Into<Angle64>,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> &mut Self {
        self.add_gate(Gate::r1xy(theta.into(), phi.into(), qubits))
    }

    /// Apply U gate(s) (general single-qubit unitary) to one or more qubits.
    ///
    /// The U gate is parameterized by three angles (theta, phi, lambda).
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().u(PI / 2.0, 0.0, PI, &[0]);
    /// ```
    pub fn u(
        &mut self,
        theta: impl Into<Angle64>,
        phi: impl Into<Angle64>,
        lambda: impl Into<Angle64>,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> &mut Self {
        self.add_gate(Gate::u(theta.into(), phi.into(), lambda.into(), qubits))
    }

    // =========================================================================
    // Two-qubit gates
    // =========================================================================

    /// Apply CNOT (CX) gate(s) to one or more qubit pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// // Single pair
    /// circuit.tick().cx(&[(0, 1)]);
    /// // Multiple pairs in one call
    /// circuit.tick().cx(&[(2, 3), (4, 5), (6, 7)]);
    /// ```
    pub fn cx(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::cx(pairs))
    }

    /// Apply CY gate(s) to one or more qubit pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().cy(&[(0, 1)]);
    /// circuit.tick().cy(&[(2, 3), (4, 5)]);
    /// ```
    pub fn cy(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::cy(pairs))
    }

    /// Apply CZ gate(s) to one or more qubit pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().cz(&[(0, 1)]);
    /// circuit.tick().cz(&[(2, 3), (4, 5)]);
    /// ```
    pub fn cz(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::cz(pairs))
    }

    /// Apply SZZ gate(s) (sqrt-ZZ) to one or more qubit pairs.
    pub fn szz(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::szz(pairs))
    }

    /// Apply SZZ-dagger gate(s) to one or more qubit pairs.
    pub fn szzdg(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::szzdg(pairs))
    }

    /// Apply SXX gate(s) (sqrt-XX) to one or more qubit pairs.
    pub fn sxx(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::sxx(pairs))
    }

    /// Apply SXX-dagger gate(s) to one or more qubit pairs.
    pub fn sxxdg(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::sxxdg(pairs))
    }

    /// Apply SYY gate(s) (sqrt-YY) to one or more qubit pairs.
    pub fn syy(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::syy(pairs))
    }

    /// Apply SYY-dagger gate(s) to one or more qubit pairs.
    pub fn syydg(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::syydg(pairs))
    }

    /// Apply SWAP gate(s) to one or more qubit pairs.
    pub fn swap(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::swap(pairs))
    }

    /// Apply CH (controlled-Hadamard) gate(s) to one or more qubit pairs.
    pub fn ch(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::ch(pairs))
    }

    /// Apply RXX rotation(s) to one or more qubit pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().rxx(PI / 4.0, &[(0, 1)]);
    /// ```
    pub fn rxx(
        &mut self,
        theta: impl Into<Angle64>,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::rxx(theta.into(), pairs))
    }

    /// Apply RYY rotation(s) to one or more qubit pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().ryy(PI / 4.0, &[(0, 1)]);
    /// ```
    pub fn ryy(
        &mut self,
        theta: impl Into<Angle64>,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::ryy(theta.into(), pairs))
    }

    /// Apply RZZ rotation(s) to one or more qubit pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    /// use std::f64::consts::PI;
    ///
    /// let mut circuit = TickCircuit::new();
    /// // Single pair
    /// circuit.tick().rzz(PI / 4.0, &[(0, 1)]);
    /// // Multiple pairs with same angle
    /// circuit.tick().rzz(PI / 2.0, &[(2, 3), (4, 5)]);
    /// ```
    pub fn rzz(
        &mut self,
        theta: impl Into<Angle64>,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        self.add_gate(Gate::rzz(theta.into(), pairs))
    }

    /// Apply CRZ (controlled-RZ) gate(s) to one or more qubit pairs.
    ///
    /// The first qubit in each pair is the control, the second is the target.
    pub fn crz(
        &mut self,
        theta: impl Into<Angle64>,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        let angle = theta.into();
        for &(c, t) in pairs {
            self.add_gate(Gate::with_angles(
                GateType::CRZ,
                vec![angle],
                vec![c.into(), t.into()],
            ));
        }
        self
    }

    // =========================================================================
    // Three-qubit gates
    // =========================================================================

    /// Apply CCX (Toffoli) gate(s).
    ///
    /// Each triple is (control1, control2, target).
    pub fn ccx(
        &mut self,
        triples: &[(
            impl Into<QubitId> + Copy,
            impl Into<QubitId> + Copy,
            impl Into<QubitId> + Copy,
        )],
    ) -> &mut Self {
        for &(c1, c2, t) in triples {
            self.add_gate(Gate::simple(
                GateType::CCX,
                vec![c1.into(), c2.into(), t.into()],
            ));
        }
        self
    }

    // =========================================================================
    // State preparation and measurement
    // =========================================================================

    /// Prepare qubit(s) in the |0⟩ state.
    ///
    /// Returns a [`TickPrepHandle`] that allows attaching metadata via `.meta()`.
    /// This breaks the chain - only `.meta()` can be called on the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().pz(&[0]);           // Single qubit
    /// circuit.tick().pz(&[1, 2, 3]);     // Multiple qubits
    /// ```
    pub fn pz(mut self, qubits: &[impl Into<QubitId> + Copy]) -> TickPrepHandle<'a> {
        let gate_idx = self.add_gate_get_idx(Gate::prep(qubits));
        TickPrepHandle {
            circuit: self.circuit,
            tick_idx: self.tick_idx,
            gate_idx,
        }
    }

    /// Prepare qubit(s) (alias for pz).
    ///
    /// Returns a [`TickPrepHandle`] that allows attaching metadata via `.meta()`.
    pub fn prep(self, qubits: &[impl Into<QubitId> + Copy]) -> TickPrepHandle<'a> {
        self.pz(qubits)
    }

    /// Measure qubit(s) in the Z basis.
    ///
    /// Returns a [`TickMeasureHandle`] that allows attaching metadata via `.meta()`.
    /// This breaks the chain - only `.meta()` can be called on the result.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().mz(&[0]);           // Single qubit
    /// circuit.tick().mz(&[1, 2, 3]);     // Multiple qubits
    /// ```
    pub fn mz(mut self, qubits: &[impl Into<QubitId> + Copy]) -> TickMeasureHandle<'a> {
        let gate_idx = self.add_gate_get_idx(Gate::measure(qubits));
        TickMeasureHandle {
            circuit: self.circuit,
            tick_idx: self.tick_idx,
            gate_idx,
        }
    }

    /// Measure and free qubit(s) (destructive measurement).
    ///
    /// Returns a [`TickMeasureHandle`] that allows attaching metadata via `.meta()`.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().measure_free(&[0, 1]);
    /// ```
    pub fn measure_free(mut self, qubits: &[impl Into<QubitId> + Copy]) -> TickMeasureHandle<'a> {
        let gate_idx = self.add_gate_get_idx(Gate::measure_free(qubits));
        TickMeasureHandle {
            circuit: self.circuit,
            tick_idx: self.tick_idx,
            gate_idx,
        }
    }

    // =========================================================================
    // Resource management
    // =========================================================================

    /// Allocate one or more qubits.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().qalloc(&[0, 1, 2, 3]);
    /// ```
    pub fn qalloc(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::qalloc(qubits))
    }

    /// Free one or more qubits.
    pub fn qfree(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        self.add_gate(Gate::qfree(qubits))
    }

    // =========================================================================
    // Timing
    // =========================================================================

    /// Insert an idle (wait) operation for one or more qubits.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// // Idle for 100 nanoseconds
    /// circuit.tick().idle(100, &[0, 1, 2]);
    /// ```
    pub fn idle(
        &mut self,
        duration: impl Into<Nanoseconds>,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> &mut Self {
        let ns: Nanoseconds = duration.into();
        self.add_gate(Gate::idle(
            ns.as_f64(),
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        ))
    }

    // =========================================================================
    // Custom gates with signature validation
    // =========================================================================

    /// Add a custom gate with signature validation.
    ///
    /// On first use, the gate name's signature (quantum arity, angle arity)
    /// is recorded. Subsequent uses are validated against this signature.
    ///
    /// The `_symbol` metadata is automatically set to the gate name.
    ///
    /// # Errors
    ///
    /// Returns `CustomGateError::SignatureMismatch` if the arity does not match
    /// a previous use, or `CustomGateError::QubitConflict` if a qubit is already
    /// in use in this tick.
    pub fn custom_gate(
        &mut self,
        name: &str,
        qubits: &[usize],
        angles: &[Angle64],
    ) -> Result<&mut Self, CustomGateError> {
        self.circuit
            .validate_or_register_gate(name, qubits.len(), angles.len())?;

        let qubit_ids: GateQubits = qubits.iter().map(|&q| QubitId::from(q)).collect();
        let gate = Gate::new(GateType::Custom, angles.to_vec(), vec![], qubit_ids);

        match self.circuit.ticks[self.tick_idx].try_add_gate(gate) {
            Ok(idx) => {
                self.last_gate_idx = Some(idx);
                // Auto-store _symbol metadata
                self.circuit.ticks[self.tick_idx].set_gate_attr(
                    idx,
                    "_symbol",
                    Attribute::String(name.to_string()),
                );
                Ok(self)
            }
            Err(mut err) => {
                err.tick_idx = Some(self.tick_idx);
                Err(CustomGateError::QubitConflict(err))
            }
        }
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
    /// tc.tick().h(&[0]);
    /// tc.tick().cx(&[(0, 1)]);
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
        tc.tick().pz(&[0]); // tick 0: first prep
        // If we want both preps in tick 0, we'd use the Tick API directly.
        // Here we use separate ticks for clarity:
        tc.tick().pz(&[1]); // tick 1: second prep
        tc.tick().h(&[0]); // tick 2
        tc.tick().cx(&[(0, 1)]); // tick 3
        tc.tick().mz(&[0]); // tick 4
        tc.tick().mz(&[1]); // tick 5

        assert_eq!(tc.num_ticks(), 6);
        assert_eq!(tc.gate_count(), 6);
    }

    #[test]
    fn test_multiple_preps_same_tick() {
        let mut tc = TickCircuit::new();

        // To add multiple preps to the same tick, use bulk prep
        tc.tick().pz(&[0, 1]); // Both preps in tick 0

        tc.tick().h(&[0]);
        tc.tick().cx(&[(0, 1)]);

        // Multiple measurements in same tick using bulk measurement
        tc.tick().mz(&[0, 1]);

        assert_eq!(tc.num_ticks(), 4);
        assert_eq!(tc.gate_count(), 4); // 1 bulk prep, 1 H, 1 CX, 1 bulk measure

        // Check tick contents
        assert_eq!(tc.get_tick(0).unwrap().len(), 1); // One bulk prep gate
        assert_eq!(tc.get_tick(1).unwrap().len(), 1); // One H
        assert_eq!(tc.get_tick(2).unwrap().len(), 1); // One CX
        assert_eq!(tc.get_tick(3).unwrap().len(), 1); // One bulk measurement
    }

    #[test]
    fn test_meta_on_gates() {
        let mut tc = TickCircuit::new();

        tc.tick()
            .h(&[0])
            .meta("duration", Attribute::Float(50.0))
            .meta("error_rate", Attribute::Float(0.001))
            .x(&[1])
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
        tc.tick().meta("round", Attribute::Int(0)).h(&[0]);
        tc.tick().meta("round", Attribute::Int(1)).cx(&[(0, 1)]);

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
        tc.tick().h(&[0]).x(&[1]).y(&[2]).z(&[3]);
        tc.tick().cx(&[(0, 1)]).szz(&[(2, 3)]);

        // But preps and measurements break the chain
        tc.tick().pz(&[0]); // breaks chain
        tc.tick().mz(&[0]); // breaks chain

        assert_eq!(tc.num_ticks(), 4);
        assert_eq!(tc.gate_count(), 8);
    }

    #[test]
    fn test_prep_and_meas_with_meta() {
        let mut tc = TickCircuit::new();

        // Preps and measurements allow .meta() before breaking
        tc.tick()
            .pz(&[0])
            .meta("reason", Attribute::String("init".into()));
        tc.tick().h(&[0]);
        tc.tick()
            .mz(&[0])
            .meta("basis", Attribute::String("Z".into()));

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
        tc.tick().h(&[0]);

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
        tc.tick().h(&[0]).x(&[1]); // Tick 0: parallel H and X
        tc.tick().cx(&[(0, 1)]); // Tick 1: CX
        tc.tick().h(&[0]); // Tick 2: H

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
        tc1.tick().h(&[0]);
        tc1.tick().cx(&[(0, 1)]);
        tc1.tick().h(&[1]);

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
            .h(&[0]);
        tc1.tick().meta("round", Attribute::Int(1)).cx(&[(0, 1)]);

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
        tc.tick().h(&[0]).x(&[1]).cx(&[(2, 3)]);

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
        tc.tick().h(&[0]).cx(&[(1, 2)]);

        let tick = tc.get_tick(0).unwrap();

        assert!(tick.uses_qubit(QubitId::from(0)));
        assert!(tick.uses_qubit(QubitId::from(1)));
        assert!(tick.uses_qubit(QubitId::from(2)));
        assert!(!tick.uses_qubit(QubitId::from(3)));
    }

    #[test]
    fn test_find_conflicts() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]).cx(&[(1, 2)]);

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
        handle.h(&[0]);

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
        tc.tick().h(&[0]).x(&[0]);
    }

    #[test]
    fn test_two_qubit_gate_conflict() {
        let mut tc = TickCircuit::new();
        let mut handle = tc.tick();
        handle.cx(&[(0, 1)]);

        // Both qubits of CX should be marked as in use
        let result = handle.try_add_gate(Gate::h(&[0]));
        assert!(result.is_err());

        let mut handle2 = tc.tick();
        handle2.cx(&[(2, 3)]);
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

        tc.tick().h(&[0]).metas(attrs).x(&[1]);

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
        tc.tick().metas(attrs).h(&[0]);

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
        tc.tick().h(&[0]);

        assert_eq!(
            tc.get_meta("name"),
            Some(&Attribute::String("bell".to_string()))
        );
        assert_eq!(tc.get_meta("version"), Some(&Attribute::Int(1)));
    }

    #[test]
    fn test_bulk_operations() {
        let mut tc = TickCircuit::new();

        // Test bulk single-qubit gates
        tc.tick().h(&[0, 1, 2, 3]);
        assert_eq!(tc.get_tick(0).unwrap().len(), 1); // One gate with 4 qubits

        // Test bulk two-qubit gates
        tc.tick().cx(&[(0, 1), (2, 3)]);
        assert_eq!(tc.get_tick(1).unwrap().len(), 1); // One gate with 2 pairs

        // Test bulk prep and measure
        tc.tick().pz(&[0, 1, 2, 3]);
        tc.tick().mz(&[0, 1, 2, 3]);
        assert_eq!(tc.get_tick(2).unwrap().len(), 1);
        assert_eq!(tc.get_tick(3).unwrap().len(), 1);

        // Test bulk qalloc/qfree
        tc.tick().qalloc(&[4, 5, 6]);
        tc.tick().qfree(&[4, 5, 6]);
        assert_eq!(tc.get_tick(4).unwrap().len(), 1);
        assert_eq!(tc.get_tick(5).unwrap().len(), 1);
    }

    #[test]
    fn test_iteration_helpers() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0, 1]);
        tc.tick().cx(&[(0, 1)]);
        tc.tick().mz(&[0, 1]);

        // Test iter_gates
        let gates: Vec<_> = tc.iter_gates().collect();
        assert_eq!(gates.len(), 3);

        // Test iter_gates_with_tick
        let gates_with_tick: Vec<_> = tc.iter_gates_with_tick().collect();
        assert_eq!(gates_with_tick.len(), 3);
        assert_eq!(gates_with_tick[0].0, 0); // First gate is in tick 0
        assert_eq!(gates_with_tick[1].0, 1); // Second gate is in tick 1

        // Test iter_ticks
        let ticks: Vec<_> = tc.iter_ticks().collect();
        assert_eq!(ticks.len(), 3);

        // Test iter_gates_by_type
        let h_gates: Vec<_> = tc.iter_gates_by_type(GateType::H).collect();
        assert_eq!(h_gates.len(), 1);

        // Test all_qubits
        let qubits = tc.all_qubits();
        assert_eq!(qubits.len(), 2);
        assert!(qubits.contains(&QubitId::from(0)));
        assert!(qubits.contains(&QubitId::from(1)));

        // Test gate_counts_by_type
        let counts = tc.gate_counts_by_type();
        assert_eq!(counts.get(&GateType::H), Some(&1));
        assert_eq!(counts.get(&GateType::CX), Some(&1));
        assert_eq!(counts.get(&GateType::Measure), Some(&1));
    }

    #[test]
    fn test_clear() {
        let mut tc = TickCircuit::new();
        tc.set_meta("name", Attribute::String("test".to_string()));
        tc.tick().h(&[0]);
        tc.tick().cx(&[(0, 1)]);

        assert_eq!(tc.num_ticks(), 2);
        assert!(tc.get_meta("name").is_some());

        tc.clear();

        assert_eq!(tc.num_ticks(), 0);
        assert_eq!(tc.gate_count(), 0);
        assert!(tc.get_meta("name").is_none());
        assert_eq!(tc.next_tick_index(), 0);
    }

    #[test]
    fn test_reset() {
        let mut tc = TickCircuit::new();
        tc.set_meta("name", Attribute::String("test".to_string()));
        tc.tick().h(&[0]);
        tc.tick().cx(&[(0, 1)]);

        assert_eq!(tc.num_ticks(), 2);

        tc.reset();

        assert_eq!(tc.num_ticks(), 0);
        assert_eq!(tc.gate_count(), 0);
        assert!(tc.get_meta("name").is_none());
        assert_eq!(tc.next_tick_index(), 0);

        // Can reuse the circuit
        tc.tick().x(&[0]);
        assert_eq!(tc.num_ticks(), 1);
    }

    #[test]
    fn test_reserve_ticks() {
        let mut tc = TickCircuit::new();
        tc.reserve_ticks(4);

        assert_eq!(tc.ticks().len(), 4);
        assert_eq!(tc.next_tick_index(), 4);

        // All ticks are empty
        for tick in tc.ticks() {
            assert!(tick.is_empty());
        }

        // New tick() starts after reserved ticks
        tc.tick().h(&[0]);
        assert_eq!(tc.ticks().len(), 5);
        assert_eq!(tc.next_tick_index(), 5);
    }

    #[test]
    fn test_insert_tick() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]); // Tick 0
        tc.tick().cx(&[(0, 1)]); // Tick 1

        // Insert between them
        tc.insert_tick(1).x(&[1]);

        assert_eq!(tc.num_ticks(), 3);

        // Check order: H at 0, X at 1, CX at 2
        let tick0 = tc.get_tick(0).unwrap();
        assert_eq!(tick0.gates()[0].gate_type, GateType::H);

        let tick1 = tc.get_tick(1).unwrap();
        assert_eq!(tick1.gates()[0].gate_type, GateType::X);

        let tick2 = tc.get_tick(2).unwrap();
        assert_eq!(tick2.gates()[0].gate_type, GateType::CX);
    }

    #[test]
    fn test_insert_tick_at_beginning() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick().x(&[1]);

        tc.insert_tick(0).z(&[2]);

        assert_eq!(tc.num_ticks(), 3);

        // Z should now be at tick 0
        let tick0 = tc.get_tick(0).unwrap();
        assert_eq!(tick0.gates()[0].gate_type, GateType::Z);
    }

    #[test]
    fn test_insert_tick_at_end() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);

        // Insert at the end (same as tick())
        tc.insert_tick(1).x(&[1]);

        assert_eq!(tc.num_ticks(), 2);

        let tick1 = tc.get_tick(1).unwrap();
        assert_eq!(tick1.gates()[0].gate_type, GateType::X);
    }

    #[test]
    fn test_tick_at() {
        let mut tc = TickCircuit::new();
        tc.reserve_ticks(3);

        // Add gates to ticks out of order
        tc.tick_at(2).cx(&[(0, 1)]);
        tc.tick_at(0).h(&[0]);
        tc.tick_at(1).x(&[1]);

        assert_eq!(tc.num_ticks(), 3);

        // Check each tick has the right gate
        assert_eq!(tc.get_tick(0).unwrap().gates()[0].gate_type, GateType::H);
        assert_eq!(tc.get_tick(1).unwrap().gates()[0].gate_type, GateType::X);
        assert_eq!(tc.get_tick(2).unwrap().gates()[0].gate_type, GateType::CX);
    }

    #[test]
    fn test_tick_at_add_more_gates() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]); // Tick 0 with H

        // Add more gates to tick 0
        tc.tick_at(0).x(&[1]);

        assert_eq!(tc.num_ticks(), 1);
        assert_eq!(tc.get_tick(0).unwrap().len(), 2);
    }

    #[test]
    #[should_panic(expected = "tick_at index 5 out of bounds")]
    fn test_tick_at_out_of_bounds() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick_at(5); // Should panic
    }

    #[test]
    #[should_panic(expected = "insert_tick index 5 out of bounds")]
    fn test_insert_tick_out_of_bounds() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.insert_tick(5); // Should panic
    }

    #[test]
    fn test_tick_discard() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]).x(&[1]).cx(&[(2, 3)]);

        let tick = tc.get_tick_mut(0).unwrap();
        assert_eq!(tick.len(), 3);

        // Discard gates using qubit 0 and qubit 2
        let removed = tick.discard(&[QubitId::from(0), QubitId::from(2)]);

        assert_eq!(removed, 2); // H on q0 and CX on q2,q3
        assert_eq!(tick.len(), 1); // Only X on q1 remains
        assert_eq!(tick.gates()[0].gate_type, GateType::X);
    }

    #[test]
    fn test_tick_discard_no_match() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]).x(&[1]);

        let tick = tc.get_tick_mut(0).unwrap();
        let removed = tick.discard(&[QubitId::from(5), QubitId::from(6)]);

        assert_eq!(removed, 0);
        assert_eq!(tick.len(), 2);
    }

    #[test]
    fn test_tick_discard_preserves_attrs() {
        let mut tc = TickCircuit::new();
        tc.tick()
            .h(&[0])
            .meta("h_attr", Attribute::Int(1))
            .x(&[1])
            .meta("x_attr", Attribute::Int(2))
            .z(&[2])
            .meta("z_attr", Attribute::Int(3));

        let tick = tc.get_tick_mut(0).unwrap();

        // Remove the X gate (index 1)
        let removed = tick.discard(&[QubitId::from(1)]);
        assert_eq!(removed, 1);
        assert_eq!(tick.len(), 2);

        // H attr should still be at index 0
        assert_eq!(tick.get_gate_attr(0, "h_attr"), Some(&Attribute::Int(1)));
        // Z attr should now be at index 1 (shifted from 2)
        assert_eq!(tick.get_gate_attr(1, "z_attr"), Some(&Attribute::Int(3)));
        // X attr should be gone
        assert!(tick.get_gate_attr(1, "x_attr").is_none());
    }

    #[test]
    fn test_tick_remove_gate() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]).x(&[1]).z(&[2]);

        let tick = tc.get_tick_mut(0).unwrap();
        assert_eq!(tick.len(), 3);

        let removed = tick.remove_gate(1); // Remove X
        assert!(removed.is_some());
        assert_eq!(removed.unwrap().gate_type, GateType::X);
        assert_eq!(tick.len(), 2);

        // Check remaining gates
        assert_eq!(tick.gates()[0].gate_type, GateType::H);
        assert_eq!(tick.gates()[1].gate_type, GateType::Z);
    }

    #[test]
    fn test_tick_remove_gate_out_of_bounds() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);

        let tick = tc.get_tick_mut(0).unwrap();
        let removed = tick.remove_gate(5);
        assert!(removed.is_none());
        assert_eq!(tick.len(), 1);
    }

    #[test]
    fn test_circuit_discard() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]).x(&[1]).cx(&[(2, 3)]);

        let removed = tc.discard(&[0, 2], 0);
        assert_eq!(removed, Some(2));
        assert_eq!(tc.get_tick(0).unwrap().len(), 1);
    }

    #[test]
    fn test_circuit_discard_invalid_tick() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);

        let removed = tc.discard(&[0], 5);
        assert_eq!(removed, None);
    }

    // =========================================================================
    // Gate signature validation tests
    // =========================================================================

    #[test]
    fn test_custom_gate_jit_registration() {
        let mut tc = TickCircuit::new();
        tc.tick()
            .custom_gate("MY_GATE", &[0, 1], &[])
            .expect("first use should succeed");

        assert!(tc.gate_signatures().contains_key("MY_GATE"));
        let sig = &tc.gate_signatures()["MY_GATE"];
        assert_eq!(sig.quantum_arity, 2);
        assert_eq!(sig.angle_arity, 0);
    }

    #[test]
    fn test_custom_gate_consistent_use_ok() {
        let mut tc = TickCircuit::new();
        tc.tick()
            .custom_gate("MY_GATE", &[0, 1], &[])
            .expect("first use");
        tc.tick()
            .custom_gate("MY_GATE", &[2, 3], &[])
            .expect("consistent use should succeed");
    }

    #[test]
    fn test_custom_gate_mismatch_quantum_arity() {
        let mut tc = TickCircuit::new();
        tc.tick()
            .custom_gate("MY_GATE", &[0, 1], &[])
            .expect("first use");
        let mut handle = tc.tick();
        let result = handle.custom_gate("MY_GATE", &[0, 1, 2], &[]);
        if let Err(CustomGateError::SignatureMismatch(e)) = result {
            assert_eq!(e.expected_quantum_arity, 2);
            assert_eq!(e.actual_quantum_arity, 3);
        } else {
            panic!("expected SignatureMismatch error");
        }
    }

    #[test]
    fn test_custom_gate_mismatch_angle_arity() {
        let mut tc = TickCircuit::new();
        let angle = Angle64::from_radians(1.0);
        tc.tick()
            .custom_gate("PARAM_GATE", &[0], &[angle])
            .expect("first use");
        let mut handle = tc.tick();
        let result = handle.custom_gate("PARAM_GATE", &[0], &[]);
        if let Err(CustomGateError::SignatureMismatch(e)) = result {
            assert_eq!(e.expected_angle_arity, 1);
            assert_eq!(e.actual_angle_arity, 0);
        } else {
            panic!("expected SignatureMismatch error");
        }
    }

    #[test]
    fn test_custom_gate_stores_symbol_metadata() {
        let mut tc = TickCircuit::new();
        tc.tick()
            .custom_gate("FOOBAR", &[0], &[])
            .expect("should succeed");

        let tick = tc.get_tick(0).unwrap();
        let symbol = tick.get_gate_attr(0, "_symbol");
        assert_eq!(symbol, Some(&Attribute::String("FOOBAR".to_string())));
    }

    #[test]
    fn test_custom_gate_with_angles() {
        let mut tc = TickCircuit::new();
        let a1 = Angle64::from_radians(0.5);
        let a2 = Angle64::from_radians(1.0);
        tc.tick()
            .custom_gate("PARAM2", &[0], &[a1, a2])
            .expect("should succeed");

        let tick = tc.get_tick(0).unwrap();
        let gate = &tick.gates()[0];
        assert_eq!(gate.gate_type, GateType::Custom);
        assert_eq!(gate.angles.len(), 2);
        assert_eq!(gate.angles[0], a1);
        assert_eq!(gate.angles[1], a2);
    }

    #[test]
    fn test_custom_gate_qubit_conflict() {
        let mut tc = TickCircuit::new();
        let mut handle = tc.tick();
        handle.h(&[0]);
        let result = handle.custom_gate("MY_GATE", &[0], &[]);
        assert!(matches!(result, Err(CustomGateError::QubitConflict(_))));
    }

    #[test]
    fn test_import_signatures() {
        let mut tc = TickCircuit::new();
        let mut sigs = HashMap::new();
        sigs.insert(
            "AOT_GATE".to_string(),
            GateSignature {
                quantum_arity: 2,
                angle_arity: 1,
            },
        );
        tc.import_signatures(&sigs);

        // Now using AOT_GATE with correct arity succeeds
        let angle = Angle64::from_radians(0.5);
        tc.tick()
            .custom_gate("AOT_GATE", &[0, 1], &[angle])
            .expect("correct arity");

        // Wrong arity fails
        let mut handle = tc.tick();
        let result = handle.custom_gate("AOT_GATE", &[0], &[angle]);
        assert!(matches!(result, Err(CustomGateError::SignatureMismatch(_))));
    }

    #[test]
    fn test_reset_clears_signatures() {
        let mut tc = TickCircuit::new();
        tc.tick()
            .custom_gate("MY_GATE", &[0, 1], &[])
            .expect("first use");
        assert!(!tc.gate_signatures().is_empty());

        tc.reset();
        assert!(tc.gate_signatures().is_empty());
    }
}
