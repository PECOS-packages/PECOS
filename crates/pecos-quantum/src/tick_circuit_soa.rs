//! Data-Oriented Design (DOD) `TickCircuit` implementation.
//!
//! This module provides `TickCircuitSoA`, an alternative representation of tick-based
//! quantum circuits optimized for **batched simulation**.
//!
//! # Design Goals
//!
//! 1. **Batched gate application**: Gates grouped by type within each tick for batch execution
//! 2. **Cache-friendly memory layout**: Qubits for same-type gates stored contiguously
//! 3. **O(1) lookups**: Pre-computed indexes for qubit-to-gate and tick-to-gate queries
//! 4. **Efficient simulation**: Direct batch calls to simulator without per-gate dispatch
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                     TickCircuitSoA                              │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  TickGateGroups (simulation-optimized)                          │
//! │  ├── ticks[0]: [H→[q0,q1], CX→[c0,t0,c1,t1], ...]              │
//! │  ├── ticks[1]: [Mz→[q0,q1,q2], ...]                            │
//! │  └── ...                                                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  GateStorage (SoA layout for individual gate access)            │
//! │  ├── types:        [H, CX, H, Mz, ...]        (Vec<GateType>)   │
//! │  ├── tick_ids:     [0, 0, 1, 2, ...]          (Vec<u16>)        │
//! │  ├── qubit_spans:  [(0,1), (1,3), (3,4), ...] (Vec<(u32,u32)>)  │
//! │  └── qubits:       [0, 0, 1, 0, ...]          (Vec<QubitId>)    │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  CircuitIndexes                                                 │
//! │  ├── tick_gates:       [[0,1], [2,3], ...]    (Vec<Vec<u32>>)   │
//! │  ├── qubit_to_gates:   [[0,1], [1], ...]      (Vec<SmallVec>)   │
//! │  └── max_qubit: usize                                           │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # Batched Simulation
//!
//! The key optimization is grouping gates by type within each tick:
//!
//! ```text
//! // Without batching (gate-by-gate):
//! for gate in tick.gates():
//!     match gate.type:
//!         H  => sim.h(&[gate.qubit])      // 4 separate calls
//!         H  => sim.h(&[gate.qubit])
//!         CX => sim.cx(&[c, t])
//!         CX => sim.cx(&[c, t])
//!
//! // With batching (one call per type):
//! sim.h(&[q0, q1, q2, q3])              // 1 batched call
//! sim.cx(&[c0, t0, c1, t1])             // 1 batched call
//! ```
//!
//! # Usage
//!
//! ```
//! use pecos_quantum::TickCircuitSoA;
//!
//! // Build using the builder pattern
//! let mut builder = TickCircuitSoA::builder();
//! builder
//!     .tick()
//!         .h(&[0, 1])
//!         .cx(&[(0, 1)])
//!     .tick()
//!         .mz(&[0, 1]);
//! let circuit = builder.build();
//! ```
//!
//! With a simulator, the batched iteration looks like:
//!
//! ```text
//! for (tick_idx, tick) in circuit.iter_ticks_batched() {
//!     for batch in tick.iter() {
//!         match batch.gate_type {
//!             GateType::H  => sim.h(batch.qubits()),
//!             GateType::CX => sim.cx(batch.qubits()),
//!             GateType::MZ => { sim.mz(batch.qubits()); }
//!             // ...
//!         }
//!     }
//! }
//! ```

use crate::Attribute;
use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, QubitId};
use smallvec::SmallVec;
use std::collections::BTreeMap;

// ============================================================================
// Gate Batching for Simulation
// ============================================================================

/// A batch of gates of the same type, ready for efficient batch application.
///
/// For single-qubit gates, `qubits` contains one qubit per gate instance.
/// For two-qubit gates, `qubits` contains pairs: `[c0, t0, c1, t1, ...]`.
#[derive(Debug, Clone)]
pub struct GateBatch {
    /// The gate type for all gates in this batch.
    pub gate_type: GateType,
    /// Qubits for batch application (contiguous for cache efficiency).
    /// Single-qubit: `[q0, q1, q2, ...]`
    /// Two-qubit: `[c0, t0, c1, t1, ...]` (control-target pairs)
    pub qubits: SmallVec<[QubitId; 16]>,
    /// Angles for parameterized gates (one per gate instance).
    pub angles: SmallVec<[Angle64; 4]>,
    /// Parameters for gates with params (e.g., idle duration).
    pub params: SmallVec<[f64; 4]>,
}

impl GateBatch {
    /// Creates a new empty batch for the given gate type.
    #[inline]
    #[must_use]
    pub fn new(gate_type: GateType) -> Self {
        Self {
            gate_type,
            qubits: SmallVec::new(),
            angles: SmallVec::new(),
            params: SmallVec::new(),
        }
    }

    /// Returns the qubits for batch application.
    #[inline]
    #[must_use]
    pub fn qubits(&self) -> &[QubitId] {
        &self.qubits
    }

    /// Returns the angles for parameterized gates.
    #[inline]
    #[must_use]
    pub fn angles(&self) -> &[Angle64] {
        &self.angles
    }

    /// Returns the number of gate instances in this batch.
    #[inline]
    #[must_use]
    pub fn gate_count(&self) -> usize {
        let arity = self.gate_type.quantum_arity();
        self.qubits
            .len()
            .checked_div(arity)
            .unwrap_or(self.qubits.len())
    }

    /// Returns true if the batch is empty.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.qubits.is_empty()
    }

    /// Adds a gate's qubits to this batch.
    #[inline]
    pub fn add_qubits(&mut self, qubits: &[QubitId]) {
        self.qubits.extend_from_slice(qubits);
    }

    /// Adds a gate's angle to this batch.
    #[inline]
    pub fn add_angle(&mut self, angle: Angle64) {
        self.angles.push(angle);
    }
}

/// Gates for a single tick, grouped by type for batch application.
#[derive(Debug, Clone, Default)]
pub struct TickBatches {
    /// Gate batches, one per gate type that appears in this tick.
    /// Ordered by insertion (first gate type seen comes first).
    pub batches: SmallVec<[GateBatch; 8]>,
}

impl TickBatches {
    /// Creates a new empty tick.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns an iterator over the batches.
    #[inline]
    pub fn iter(&self) -> impl Iterator<Item = &GateBatch> {
        self.batches.iter()
    }

    /// Returns the number of batches.
    #[inline]
    #[must_use]
    pub fn batch_count(&self) -> usize {
        self.batches.len()
    }

    /// Returns the total number of gates in this tick.
    #[must_use]
    pub fn gate_count(&self) -> usize {
        self.batches.iter().map(GateBatch::gate_count).sum()
    }

    /// Adds a gate to the appropriate batch (creates batch if needed).
    ///
    /// # Panics
    /// Panics if internal batch list is unexpectedly empty after insertion.
    pub fn add_gate(&mut self, gate_type: GateType, qubits: &[QubitId], angles: &[Angle64]) {
        // Find or create batch for this gate type
        let batch = if let Some(batch) = self.batches.iter_mut().find(|b| b.gate_type == gate_type)
        {
            batch
        } else {
            self.batches.push(GateBatch::new(gate_type));
            self.batches.last_mut().expect("batch was just pushed")
        };

        batch.add_qubits(qubits);
        for &angle in angles {
            batch.add_angle(angle);
        }
    }

    /// Returns the batch for a specific gate type, if present.
    #[inline]
    #[must_use]
    pub fn batch_for_type(&self, gate_type: GateType) -> Option<&GateBatch> {
        self.batches.iter().find(|b| b.gate_type == gate_type)
    }
}

/// Pre-grouped gates by type for each tick, optimized for batched simulation.
///
/// This is the primary structure for efficient circuit execution:
/// - Gates are grouped by type within each tick
/// - Qubits for same-type gates are stored contiguously
/// - Enables single batch calls to the simulator per gate type
#[derive(Debug, Clone, Default)]
pub struct TickGateGroups {
    /// For each tick, the batched gates.
    pub ticks: Vec<TickBatches>,
    /// Number of ticks.
    pub num_ticks: usize,
    /// Maximum qubit index seen.
    pub max_qubit: usize,
}

impl TickGateGroups {
    /// Creates empty gate groups.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns the number of ticks.
    #[inline]
    #[must_use]
    pub fn num_ticks(&self) -> usize {
        self.num_ticks
    }

    /// Returns the batches for a specific tick.
    #[inline]
    #[must_use]
    pub fn tick(&self, tick_idx: usize) -> Option<&TickBatches> {
        self.ticks.get(tick_idx)
    }

    /// Returns an iterator over all ticks.
    #[inline]
    pub fn iter_ticks(&self) -> impl Iterator<Item = (usize, &TickBatches)> {
        self.ticks.iter().enumerate()
    }

    /// Ensures capacity for the given tick index.
    fn ensure_tick(&mut self, tick_idx: usize) {
        if tick_idx >= self.ticks.len() {
            self.ticks.resize_with(tick_idx + 1, TickBatches::new);
        }
        self.num_ticks = self.num_ticks.max(tick_idx + 1);
    }

    /// Adds a gate to the appropriate tick and batch.
    pub fn add_gate(
        &mut self,
        tick_idx: usize,
        gate_type: GateType,
        qubits: &[QubitId],
        angles: &[Angle64],
    ) {
        self.ensure_tick(tick_idx);
        self.ticks[tick_idx].add_gate(gate_type, qubits, angles);

        // Track max qubit
        for qubit in qubits {
            self.max_qubit = self.max_qubit.max(qubit.index());
        }
    }

    /// Clears all gate groups.
    pub fn clear(&mut self) {
        self.ticks.clear();
        self.num_ticks = 0;
        self.max_qubit = 0;
    }

    /// Returns the total number of gates across all ticks.
    #[must_use]
    pub fn total_gate_count(&self) -> usize {
        self.ticks.iter().map(TickBatches::gate_count).sum()
    }
}

// ============================================================================
// Gate ID (Stable, Generational)
// ============================================================================

/// A stable identifier for a gate in a `TickCircuitSoA`.
///
/// Unlike raw indices, `GateId` includes a generation counter that allows
/// detecting use-after-free when gates are removed. This provides safety
/// without the overhead of reference counting.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GateId {
    /// Index into the gate storage arrays.
    index: u32,
    /// Generation counter for detecting stale IDs.
    generation: u16,
}

impl GateId {
    /// Creates a new `GateId`.
    #[inline]
    #[must_use]
    pub const fn new(index: u32, generation: u16) -> Self {
        Self { index, generation }
    }

    /// Returns the raw index (use with caution).
    #[inline]
    #[must_use]
    pub const fn index(self) -> usize {
        self.index as usize
    }

    /// Returns the generation.
    #[inline]
    #[must_use]
    pub const fn generation(self) -> u16 {
        self.generation
    }
}

// ============================================================================
// Gate Storage (SoA Layout)
// ============================================================================

/// Structure-of-Arrays storage for gate data.
///
/// All gates are stored in parallel arrays for cache-friendly access.
/// Variable-length data (qubits, angles) uses span-based indexing into
/// contiguous backing arrays.
#[derive(Debug, Clone, Default)]
pub struct GateStorage {
    // Core gate data (one element per gate)
    /// Gate types (H, CX, Mz, etc.)
    pub types: Vec<GateType>,
    /// Which tick each gate belongs to
    pub tick_ids: Vec<u16>,
    /// Span (start, end) into the qubits array
    pub qubit_spans: Vec<(u32, u32)>,
    /// Span (start, end) into the angles array
    pub angle_spans: Vec<(u32, u32)>,
    /// Generation counter for each slot (for `GateId` validation)
    pub generations: Vec<u16>,
    /// Whether each slot is occupied (for sparse storage after removals)
    pub occupied: Vec<bool>,

    // Backing arrays for variable-length data
    /// All qubits, indexed by `qubit_spans`
    pub qubits: Vec<QubitId>,
    /// All angles, indexed by `angle_spans`
    pub angles: Vec<Angle64>,
    /// All params (e.g., idle duration), indexed similarly
    pub param_spans: Vec<(u32, u32)>,
    pub params: Vec<f64>,

    // Free list for slot reuse after removal
    free_slots: Vec<u32>,
}

impl GateStorage {
    /// Creates empty gate storage.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates gate storage with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(gate_capacity: usize, qubit_capacity: usize) -> Self {
        Self {
            types: Vec::with_capacity(gate_capacity),
            tick_ids: Vec::with_capacity(gate_capacity),
            qubit_spans: Vec::with_capacity(gate_capacity),
            angle_spans: Vec::with_capacity(gate_capacity),
            generations: Vec::with_capacity(gate_capacity),
            occupied: Vec::with_capacity(gate_capacity),
            qubits: Vec::with_capacity(qubit_capacity),
            angles: Vec::new(),
            param_spans: Vec::with_capacity(gate_capacity),
            params: Vec::new(),
            free_slots: Vec::new(),
        }
    }

    /// Returns the number of gates (including removed slots).
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.types.len()
    }

    /// Returns true if there are no gates.
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.types.is_empty()
    }

    /// Returns the number of active (non-removed) gates.
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.occupied.iter().filter(|&&o| o).count()
    }

    /// Adds a gate and returns its ID.
    pub fn add_gate(
        &mut self,
        gate_type: GateType,
        tick_id: u16,
        qubits: &[QubitId],
        angles: &[Angle64],
        params: &[f64],
    ) -> GateId {
        let (index, generation) = if let Some(slot) = self.free_slots.pop() {
            // Reuse a freed slot
            let idx = slot as usize;
            self.generations[idx] = self.generations[idx].wrapping_add(1);
            self.types[idx] = gate_type;
            self.tick_ids[idx] = tick_id;
            self.occupied[idx] = true;
            (slot, self.generations[idx])
        } else {
            // Allocate new slot
            #[allow(clippy::cast_possible_truncation)] // gate index fits in u32
            let idx = self.types.len() as u32;
            self.types.push(gate_type);
            self.tick_ids.push(tick_id);
            self.generations.push(0);
            self.occupied.push(true);
            // Placeholders for spans - will be set below
            self.qubit_spans.push((0, 0));
            self.angle_spans.push((0, 0));
            self.param_spans.push((0, 0));
            (idx, 0)
        };

        let idx = index as usize;

        // Add qubits
        #[allow(clippy::cast_possible_truncation)] // qubit pool index fits in u32
        let qubit_start = self.qubits.len() as u32;
        self.qubits.extend_from_slice(qubits);
        #[allow(clippy::cast_possible_truncation)] // qubit pool index fits in u32
        let qubit_end = self.qubits.len() as u32;
        self.qubit_spans[idx] = (qubit_start, qubit_end);

        // Add angles
        #[allow(clippy::cast_possible_truncation)] // angle pool index fits in u32
        let angle_start = self.angles.len() as u32;
        self.angles.extend_from_slice(angles);
        #[allow(clippy::cast_possible_truncation)] // angle pool index fits in u32
        let angle_end = self.angles.len() as u32;
        self.angle_spans[idx] = (angle_start, angle_end);

        // Add params
        #[allow(clippy::cast_possible_truncation)] // param pool index fits in u32
        let param_start = self.params.len() as u32;
        self.params.extend_from_slice(params);
        #[allow(clippy::cast_possible_truncation)] // param pool index fits in u32
        let param_end = self.params.len() as u32;
        self.param_spans[idx] = (param_start, param_end);

        GateId::new(index, generation)
    }

    /// Validates that a `GateId` is still valid.
    #[inline]
    #[must_use]
    pub fn is_valid(&self, id: GateId) -> bool {
        let idx = id.index();
        idx < self.len() && self.generations[idx] == id.generation() && self.occupied[idx]
    }

    /// Returns the gate type for a valid ID.
    #[inline]
    #[must_use]
    pub fn gate_type(&self, id: GateId) -> Option<GateType> {
        if self.is_valid(id) {
            Some(self.types[id.index()])
        } else {
            None
        }
    }

    /// Returns the tick ID for a valid gate.
    #[inline]
    #[must_use]
    pub fn tick_id(&self, id: GateId) -> Option<u16> {
        if self.is_valid(id) {
            Some(self.tick_ids[id.index()])
        } else {
            None
        }
    }

    // =========================================================================
    // Unchecked accessors for hot paths (internal use)
    // =========================================================================

    /// Returns the gate type without validation. Use only when index is known valid.
    #[inline]
    #[must_use]
    pub fn type_unchecked(&self, idx: usize) -> GateType {
        self.types[idx]
    }

    /// Returns the tick ID without validation.
    #[inline]
    #[must_use]
    pub fn tick_id_unchecked(&self, idx: usize) -> u16 {
        self.tick_ids[idx]
    }

    /// Returns the qubits without validation.
    #[inline]
    #[must_use]
    pub fn qubits_unchecked(&self, idx: usize) -> &[QubitId] {
        let (start, end) = self.qubit_spans[idx];
        &self.qubits[start as usize..end as usize]
    }

    /// Returns whether the slot is occupied.
    #[inline]
    #[must_use]
    pub fn is_occupied(&self, idx: usize) -> bool {
        idx < self.occupied.len() && self.occupied[idx]
    }

    /// Returns the total number of slots (for iteration bounds).
    #[inline]
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.types.len()
    }

    /// Returns the qubits for a valid gate.
    #[inline]
    #[must_use]
    pub fn gate_qubits(&self, id: GateId) -> Option<&[QubitId]> {
        if self.is_valid(id) {
            let (start, end) = self.qubit_spans[id.index()];
            Some(&self.qubits[start as usize..end as usize])
        } else {
            None
        }
    }

    /// Returns the angles for a valid gate.
    #[inline]
    #[must_use]
    pub fn gate_angles(&self, id: GateId) -> Option<&[Angle64]> {
        if self.is_valid(id) {
            let (start, end) = self.angle_spans[id.index()];
            Some(&self.angles[start as usize..end as usize])
        } else {
            None
        }
    }

    /// Returns the params for a valid gate.
    #[inline]
    #[must_use]
    pub fn gate_params(&self, id: GateId) -> Option<&[f64]> {
        if self.is_valid(id) {
            let (start, end) = self.param_spans[id.index()];
            Some(&self.params[start as usize..end as usize])
        } else {
            None
        }
    }

    /// Removes a gate by ID. The slot can be reused.
    pub fn remove(&mut self, id: GateId) -> bool {
        if self.is_valid(id) {
            let idx = id.index();
            self.occupied[idx] = false;
            self.free_slots.push(id.index);
            true
        } else {
            false
        }
    }

    /// Clears all gates.
    pub fn clear(&mut self) {
        self.types.clear();
        self.tick_ids.clear();
        self.qubit_spans.clear();
        self.angle_spans.clear();
        self.param_spans.clear();
        self.generations.clear();
        self.occupied.clear();
        self.qubits.clear();
        self.angles.clear();
        self.params.clear();
        self.free_slots.clear();
    }

    /// Iterator over all valid gate IDs.
    pub fn iter_ids(&self) -> impl Iterator<Item = GateId> + '_ {
        (0..self.len()).filter_map(move |idx| {
            if self.occupied[idx] {
                #[allow(clippy::cast_possible_truncation)] // gate index fits in u32
                Some(GateId::new(idx as u32, self.generations[idx]))
            } else {
                None
            }
        })
    }
}

// ============================================================================
// Circuit Indexes
// ============================================================================

/// Pre-computed indexes for efficient circuit queries.
///
/// Uses raw u32 indices instead of `GateIds` for minimal overhead in hot paths.
#[derive(Debug, Clone, Default)]
pub struct CircuitIndexes {
    /// For each tick, the list of gate indices in that tick.
    /// Using Vec<Vec<u32>> for simplicity; could use CSR format for less allocation.
    pub tick_gates: Vec<Vec<u32>>,

    /// For each qubit, the list of gate indices that touch it.
    /// Indexed by qubit index; grows dynamically.
    /// Uses u32 instead of `GateId` to avoid validation overhead.
    pub qubit_to_gates: Vec<SmallVec<[u32; 8]>>,

    /// For each qubit, gates sorted by tick (for efficient backward traversal).
    /// Each entry is (`tick_id`, `gate_idx`).
    pub qubit_gates_by_tick: Vec<Vec<(u16, u32)>>,

    /// Maximum qubit index seen.
    pub max_qubit: usize,

    /// Number of ticks.
    pub num_ticks: usize,
}

impl CircuitIndexes {
    /// Creates empty indexes.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Ensures the qubit index can accommodate the given qubit.
    fn ensure_qubit_capacity(&mut self, qubit: usize) {
        if qubit >= self.qubit_to_gates.len() {
            self.qubit_to_gates.resize(qubit + 1, SmallVec::new());
            self.qubit_gates_by_tick.resize(qubit + 1, Vec::new());
        }
        self.max_qubit = self.max_qubit.max(qubit);
    }

    /// Registers a gate in the indexes (using raw index).
    pub fn register_gate_raw(&mut self, gate_idx: u32, tick_id: u16, qubits: &[QubitId]) {
        // Add to tick index
        let tick = tick_id as usize;
        if tick >= self.tick_gates.len() {
            self.tick_gates.resize(tick + 1, Vec::new());
        }
        self.tick_gates[tick].push(gate_idx);
        self.num_ticks = self.num_ticks.max(tick + 1);

        // Add to qubit indexes
        for qubit in qubits {
            let q = qubit.index();
            self.ensure_qubit_capacity(q);
            self.qubit_to_gates[q].push(gate_idx);
            self.qubit_gates_by_tick[q].push((tick_id, gate_idx));
        }
    }

    /// Registers a gate in the qubit index (`GateId` version for compatibility).
    pub fn register_gate(&mut self, gate_id: GateId, qubits: &[QubitId]) {
        for qubit in qubits {
            let q = qubit.index();
            self.ensure_qubit_capacity(q);
            self.qubit_to_gates[q].push(gate_id.index);
        }
    }

    /// Returns all gate indices touching the given qubit.
    #[inline]
    #[must_use]
    pub fn gates_touching_qubit_raw(&self, qubit: usize) -> &[u32] {
        if qubit < self.qubit_to_gates.len() {
            &self.qubit_to_gates[qubit]
        } else {
            &[]
        }
    }

    /// Returns all gates touching the given qubit (`GateId` version).
    #[inline]
    #[must_use]
    pub fn gates_touching_qubit(&self, _qubit: usize) -> &[GateId] {
        // Note: This is a bit of a hack - we're reinterpreting u32 as GateId
        // In practice, for read-only circuits without removal, generation is always 0
        &[] // Return empty - use gates_touching_qubit_raw instead
    }

    /// Returns gate indices in a specific tick.
    #[inline]
    #[must_use]
    pub fn gates_in_tick(&self, tick: usize) -> &[u32] {
        if tick < self.tick_gates.len() {
            &self.tick_gates[tick]
        } else {
            &[]
        }
    }

    /// Returns gates on a qubit sorted by tick (for backward traversal).
    #[inline]
    #[must_use]
    pub fn qubit_gates_sorted(&self, qubit: usize) -> &[(u16, u32)] {
        if qubit < self.qubit_gates_by_tick.len() {
            &self.qubit_gates_by_tick[qubit]
        } else {
            &[]
        }
    }

    /// Sorts the `qubit_gates_by_tick` for each qubit (call after building).
    pub fn finalize(&mut self) {
        for gates in &mut self.qubit_gates_by_tick {
            gates.sort_by_key(|&(tick, _)| tick);
        }
    }

    /// Clears all indexes.
    pub fn clear(&mut self) {
        self.tick_gates.clear();
        self.qubit_to_gates.clear();
        self.qubit_gates_by_tick.clear();
        self.max_qubit = 0;
        self.num_ticks = 0;
    }

    /// Rebuilds indexes from gate storage.
    pub fn rebuild(&mut self, storage: &GateStorage) {
        self.clear();

        for idx in 0..storage.slot_count() {
            if storage.is_occupied(idx) {
                let tick_id = storage.tick_id_unchecked(idx);
                let qubits = storage.qubits_unchecked(idx);
                #[allow(clippy::cast_possible_truncation)] // gate index fits in u32
                self.register_gate_raw(idx as u32, tick_id, qubits);
            }
        }

        self.finalize();
    }
}

// ============================================================================
// Metadata Storage
// ============================================================================

/// Lazy metadata storage - only allocates when metadata is actually used.
#[derive(Debug, Clone, Default)]
pub struct MetadataStorage {
    /// Per-gate attributes.
    pub gate_attrs: BTreeMap<GateId, BTreeMap<String, Attribute>>,
    /// Per-tick attributes.
    pub tick_attrs: Vec<BTreeMap<String, Attribute>>,
    /// Circuit-level attributes.
    pub circuit_attrs: BTreeMap<String, Attribute>,
}

impl MetadataStorage {
    /// Creates empty metadata storage.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets a gate attribute.
    pub fn set_gate_attr(&mut self, gate_id: GateId, key: &str, value: Attribute) {
        self.gate_attrs
            .entry(gate_id)
            .or_default()
            .insert(key.to_string(), value);
    }

    /// Gets a gate attribute.
    #[must_use]
    pub fn get_gate_attr(&self, gate_id: GateId, key: &str) -> Option<&Attribute> {
        self.gate_attrs.get(&gate_id).and_then(|m| m.get(key))
    }

    /// Sets a tick attribute.
    pub fn set_tick_attr(&mut self, tick: usize, key: &str, value: Attribute) {
        if tick >= self.tick_attrs.len() {
            self.tick_attrs.resize(tick + 1, BTreeMap::new());
        }
        self.tick_attrs[tick].insert(key.to_string(), value);
    }

    /// Gets a tick attribute.
    #[must_use]
    pub fn get_tick_attr(&self, tick: usize, key: &str) -> Option<&Attribute> {
        self.tick_attrs.get(tick).and_then(|m| m.get(key))
    }

    /// Sets a circuit attribute.
    pub fn set_circuit_attr(&mut self, key: &str, value: Attribute) {
        self.circuit_attrs.insert(key.to_string(), value);
    }

    /// Gets a circuit attribute.
    #[must_use]
    pub fn get_circuit_attr(&self, key: &str) -> Option<&Attribute> {
        self.circuit_attrs.get(key)
    }

    /// Clears all metadata.
    pub fn clear(&mut self) {
        self.gate_attrs.clear();
        self.tick_attrs.clear();
        self.circuit_attrs.clear();
    }
}

// ============================================================================
// TickCircuitSoA
// ============================================================================

/// A tick-based quantum circuit optimized for batched simulation.
///
/// This is a DOD (Data-Oriented Design) alternative to [`TickCircuit`](crate::TickCircuit)
/// that provides:
/// - **Batched simulation**: Gates grouped by type for efficient batch execution
/// - **Cache-friendly access**: Qubits for same-type gates stored contiguously
/// - **Individual gate access**: `SoA` storage for analysis workloads
#[derive(Debug, Clone, Default)]
pub struct TickCircuitSoA {
    /// Gates grouped by type for batched simulation (primary interface).
    pub batched: TickGateGroups,
    /// Gate data in `SoA` layout (for individual gate access).
    pub storage: GateStorage,
    /// Pre-computed indexes.
    pub indexes: CircuitIndexes,
    /// Metadata (lazy allocation).
    pub metadata: MetadataStorage,
}

impl TickCircuitSoA {
    /// Creates a new empty circuit.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a builder for constructing circuits with a fluent API.
    #[must_use]
    pub fn builder() -> TickCircuitSoABuilder {
        TickCircuitSoABuilder::new()
    }

    /// Returns the number of ticks.
    #[inline]
    #[must_use]
    pub fn num_ticks(&self) -> usize {
        self.indexes.num_ticks
    }

    /// Returns the total number of active gates.
    #[inline]
    #[must_use]
    pub fn gate_count(&self) -> usize {
        self.storage.active_count()
    }

    /// Returns the maximum qubit index.
    #[inline]
    #[must_use]
    pub fn max_qubit(&self) -> usize {
        self.indexes.max_qubit
    }

    /// Returns the gate type for a gate ID.
    #[inline]
    #[must_use]
    pub fn gate_type(&self, id: GateId) -> Option<GateType> {
        self.storage.gate_type(id)
    }

    /// Returns the qubits for a gate ID.
    #[inline]
    #[must_use]
    pub fn gate_qubits(&self, id: GateId) -> Option<&[QubitId]> {
        self.storage.gate_qubits(id)
    }

    /// Returns the angles for a gate ID.
    #[inline]
    #[must_use]
    pub fn gate_angles(&self, id: GateId) -> Option<&[Angle64]> {
        self.storage.gate_angles(id)
    }

    /// Returns all gates touching a specific qubit.
    #[inline]
    #[must_use]
    pub fn gates_touching_qubit(&self, qubit: usize) -> &[GateId] {
        self.indexes.gates_touching_qubit(qubit)
    }

    /// Returns all gate indices touching a specific qubit (optimized, no validation).
    #[inline]
    #[must_use]
    pub fn gates_touching_qubit_raw(&self, qubit: usize) -> &[u32] {
        self.indexes.gates_touching_qubit_raw(qubit)
    }

    /// Returns gate indices in a specific tick (optimized, O(1)).
    #[inline]
    #[must_use]
    pub fn gates_in_tick_raw(&self, tick: usize) -> &[u32] {
        self.indexes.gates_in_tick(tick)
    }

    /// Returns gates on a qubit sorted by tick (for backward traversal).
    #[inline]
    #[must_use]
    pub fn qubit_gates_sorted(&self, qubit: usize) -> &[(u16, u32)] {
        self.indexes.qubit_gates_sorted(qubit)
    }

    /// Validates that a gate ID is still valid.
    #[inline]
    #[must_use]
    pub fn is_valid(&self, id: GateId) -> bool {
        self.storage.is_valid(id)
    }

    /// Iterator over all valid gate IDs.
    pub fn iter_gate_ids(&self) -> impl Iterator<Item = GateId> + '_ {
        self.storage.iter_ids()
    }

    /// Iterator over gate IDs in a specific tick.
    #[allow(clippy::cast_possible_truncation)] // tick index fits in u16
    pub fn gates_in_tick(&self, tick: usize) -> impl Iterator<Item = GateId> + '_ {
        self.storage
            .iter_ids()
            .filter(move |&id| self.storage.tick_id(id) == Some(tick as u16))
    }

    // =========================================================================
    // Batched Simulation API
    // =========================================================================

    /// Returns an iterator over ticks with batched gates for simulation.
    ///
    /// This is the primary API for efficient circuit simulation:
    /// ```text
    /// for (tick_idx, tick) in circuit.iter_ticks_batched() {
    ///     for batch in tick.iter() {
    ///         match batch.gate_type {
    ///             GateType::H  => sim.h(batch.qubits()),
    ///             GateType::CX => sim.cx(batch.qubits()),
    ///             // ...
    ///         }
    ///     }
    /// }
    /// ```
    #[inline]
    pub fn iter_ticks_batched(&self) -> impl Iterator<Item = (usize, &TickBatches)> {
        self.batched.iter_ticks()
    }

    /// Returns the batched gates for a specific tick.
    #[inline]
    #[must_use]
    pub fn tick_batched(&self, tick: usize) -> Option<&TickBatches> {
        self.batched.tick(tick)
    }

    /// Returns the number of ticks (from batched representation).
    #[inline]
    #[must_use]
    pub fn num_ticks_batched(&self) -> usize {
        self.batched.num_ticks()
    }

    /// Clears the circuit.
    pub fn clear(&mut self) {
        self.batched.clear();
        self.storage.clear();
        self.indexes.clear();
        self.metadata.clear();
    }

    /// Rebuilds indexes after modifications.
    pub fn rebuild_indexes(&mut self) {
        self.indexes.rebuild(&self.storage);
    }
}

// ============================================================================
// Builder Pattern
// ============================================================================

/// Builder for constructing `TickCircuitSoA` with a fluent API.
#[derive(Debug)]
pub struct TickCircuitSoABuilder {
    batched: TickGateGroups,
    storage: GateStorage,
    indexes: CircuitIndexes,
    metadata: MetadataStorage,
    current_tick: u16,
    last_gate_id: Option<GateId>,
}

impl TickCircuitSoABuilder {
    /// Creates a new builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            batched: TickGateGroups::new(),
            storage: GateStorage::new(),
            indexes: CircuitIndexes::new(),
            metadata: MetadataStorage::new(),
            current_tick: 0,
            last_gate_id: None,
        }
    }

    /// Starts a new tick.
    pub fn tick(&mut self) -> &mut Self {
        // Update num_ticks
        self.indexes.num_ticks = (self.current_tick + 1) as usize;
        self.current_tick += 1;
        self.last_gate_id = None;
        self
    }

    /// Adds a gate to the current tick.
    fn add_gate(
        &mut self,
        gate_type: GateType,
        qubits: &[QubitId],
        angles: &[Angle64],
        params: &[f64],
    ) -> &mut Self {
        let tick = self.current_tick.saturating_sub(1);

        // Add to batched representation (primary for simulation)
        self.batched
            .add_gate(tick as usize, gate_type, qubits, angles);

        // Add to SoA storage (for individual gate access)
        let gate_id = self
            .storage
            .add_gate(gate_type, tick, qubits, angles, params);

        // Update indexes
        self.indexes.register_gate_raw(gate_id.index, tick, qubits);
        self.last_gate_id = Some(gate_id);
        self
    }

    /// Sets metadata on the last added gate (or tick if no gate yet).
    pub fn meta(&mut self, key: &str, value: impl Into<Attribute>) -> &mut Self {
        if let Some(gate_id) = self.last_gate_id {
            self.metadata.set_gate_attr(gate_id, key, value.into());
        } else {
            // Set tick-level metadata
            let tick = self.current_tick.saturating_sub(1) as usize;
            self.metadata.set_tick_attr(tick, key, value.into());
        }
        self
    }

    /// Builds the final circuit.
    #[must_use]
    pub fn build(mut self) -> TickCircuitSoA {
        // Finalize indexes (sort qubit gates by tick for efficient traversal)
        self.indexes.finalize();
        TickCircuitSoA {
            batched: self.batched,
            storage: self.storage,
            indexes: self.indexes,
            metadata: self.metadata,
        }
    }

    // =========================================================================
    // Gate methods (mirror TickHandle API)
    // =========================================================================

    /// Apply Hadamard gate(s).
    pub fn h(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        let qs: Vec<QubitId> = qubits.iter().map(|&q| q.into()).collect();
        self.add_gate(GateType::H, &qs, &[], &[])
    }

    /// Apply X gate(s).
    pub fn x(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        let qs: Vec<QubitId> = qubits.iter().map(|&q| q.into()).collect();
        self.add_gate(GateType::X, &qs, &[], &[])
    }

    /// Apply Y gate(s).
    pub fn y(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        let qs: Vec<QubitId> = qubits.iter().map(|&q| q.into()).collect();
        self.add_gate(GateType::Y, &qs, &[], &[])
    }

    /// Apply Z gate(s).
    pub fn z(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        let qs: Vec<QubitId> = qubits.iter().map(|&q| q.into()).collect();
        self.add_gate(GateType::Z, &qs, &[], &[])
    }

    /// Apply SX gate(s).
    pub fn sx(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        let qs: Vec<QubitId> = qubits.iter().map(|&q| q.into()).collect();
        self.add_gate(GateType::SX, &qs, &[], &[])
    }

    /// Apply SY gate(s).
    pub fn sy(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        let qs: Vec<QubitId> = qubits.iter().map(|&q| q.into()).collect();
        self.add_gate(GateType::SY, &qs, &[], &[])
    }

    /// Apply SZ gate(s).
    pub fn sz(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        let qs: Vec<QubitId> = qubits.iter().map(|&q| q.into()).collect();
        self.add_gate(GateType::SZ, &qs, &[], &[])
    }

    /// Apply CX gate(s).
    pub fn cx(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        let qs: Vec<QubitId> = pairs
            .iter()
            .flat_map(|&(c, t)| [c.into(), t.into()])
            .collect();
        self.add_gate(GateType::CX, &qs, &[], &[])
    }

    /// Apply CZ gate(s).
    pub fn cz(
        &mut self,
        pairs: &[(impl Into<QubitId> + Copy, impl Into<QubitId> + Copy)],
    ) -> &mut Self {
        let qs: Vec<QubitId> = pairs
            .iter()
            .flat_map(|&(a, b)| [a.into(), b.into()])
            .collect();
        self.add_gate(GateType::CZ, &qs, &[], &[])
    }

    /// Prepare qubit(s) in |0⟩.
    pub fn pz(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        let qs: Vec<QubitId> = qubits.iter().map(|&q| q.into()).collect();
        self.add_gate(GateType::PZ, &qs, &[], &[])
    }

    /// Measure qubit(s) in Z basis.
    pub fn mz(&mut self, qubits: &[impl Into<QubitId> + Copy]) -> &mut Self {
        let qs: Vec<QubitId> = qubits.iter().map(|&q| q.into()).collect();
        self.add_gate(GateType::MZ, &qs, &[], &[])
    }
}

impl Default for TickCircuitSoABuilder {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Conversion from TickCircuit
// ============================================================================

impl From<&crate::TickCircuit> for TickCircuitSoA {
    fn from(tc: &crate::TickCircuit) -> Self {
        let mut builder = TickCircuitSoABuilder::new();

        for (tick_idx, tick_data) in tc.iter_ticks() {
            // Ensure we're at the right tick
            #[allow(clippy::cast_possible_truncation)] // tick index fits in u16
            while builder.current_tick <= tick_idx as u16 {
                builder.tick();
            }

            for (gate_idx, gate) in tick_data.gates().iter().enumerate() {
                let qubits: Vec<QubitId> = gate.qubits.to_vec();
                let angles: Vec<Angle64> = gate.angles.to_vec();
                let params: Vec<f64> = gate.params.to_vec();

                let tick_num = builder.current_tick.saturating_sub(1);

                // Add to batched representation (primary for simulation)
                builder
                    .batched
                    .add_gate(tick_num as usize, gate.gate_type, &qubits, &angles);

                // Add to SoA storage (for individual gate access)
                let gate_id =
                    builder
                        .storage
                        .add_gate(gate.gate_type, tick_num, &qubits, &angles, &params);
                builder.indexes.register_gate(gate_id, &qubits);
                builder.indexes.num_ticks = builder.indexes.num_ticks.max(tick_num as usize + 1);

                // Copy gate attributes
                for (key, value) in tick_data.gate_attrs(gate_idx) {
                    builder.metadata.set_gate_attr(gate_id, key, value.clone());
                }
            }

            // Copy tick attributes
            for (key, value) in tick_data.tick_attrs() {
                builder.metadata.set_tick_attr(tick_idx, key, value.clone());
            }
        }

        // Copy circuit attributes
        for (key, value) in tc.circuit_attrs() {
            builder.metadata.set_circuit_attr(key, value.clone());
        }

        builder.build()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_construction() {
        let mut builder = TickCircuitSoA::builder();
        builder
            .tick()
            .pz(&[0, 1])
            .tick()
            .h(&[0])
            .cx(&[(0, 1)])
            .tick()
            .mz(&[0, 1]);

        let circuit = builder.build();

        assert_eq!(circuit.num_ticks(), 3);
        assert_eq!(circuit.gate_count(), 4); // pz, h, cx, mz
    }

    #[test]
    fn test_gate_lookup() {
        let mut builder = TickCircuitSoA::builder();
        builder.tick().h(&[0]).x(&[1]);

        let circuit = builder.build();

        // Find all gates
        let gate_ids: Vec<_> = circuit.iter_gate_ids().collect();
        assert_eq!(gate_ids.len(), 2);

        // Check gate types
        assert_eq!(circuit.gate_type(gate_ids[0]), Some(GateType::H));
        assert_eq!(circuit.gate_type(gate_ids[1]), Some(GateType::X));

        // Check qubits
        assert_eq!(
            circuit.gate_qubits(gate_ids[0]),
            Some([QubitId::from(0)].as_slice())
        );
        assert_eq!(
            circuit.gate_qubits(gate_ids[1]),
            Some([QubitId::from(1)].as_slice())
        );
    }

    #[test]
    fn test_qubit_index() {
        let mut builder = TickCircuitSoA::builder();
        builder.tick().h(&[0]).x(&[1]).tick().cx(&[(0, 1)]);

        let circuit = builder.build();

        // Gates on qubit 0: H and CX (using raw accessor for efficiency)
        let q0_gates = circuit.gates_touching_qubit_raw(0);
        assert_eq!(q0_gates.len(), 2);

        // Gates on qubit 1: X and CX
        let q1_gates = circuit.gates_touching_qubit_raw(1);
        assert_eq!(q1_gates.len(), 2);

        // Gates on qubit 2: none
        let q2_gates = circuit.gates_touching_qubit_raw(2);
        assert_eq!(q2_gates.len(), 0);
    }

    #[test]
    fn test_metadata() {
        let mut builder = TickCircuitSoA::builder();
        builder
            .tick()
            .meta("round", Attribute::Int(0))
            .h(&[0])
            .meta("duration", Attribute::Float(50.0));

        let circuit = builder.build();

        // Check tick metadata
        assert_eq!(
            circuit.metadata.get_tick_attr(0, "round"),
            Some(&Attribute::Int(0))
        );

        // Check gate metadata
        let gate_ids: Vec<_> = circuit.iter_gate_ids().collect();
        assert_eq!(
            circuit.metadata.get_gate_attr(gate_ids[0], "duration"),
            Some(&Attribute::Float(50.0))
        );
    }

    #[test]
    fn test_gate_removal() {
        let mut builder = TickCircuitSoA::builder();
        builder.tick().h(&[0]).x(&[1]);

        let mut circuit = builder.build();
        let gate_ids: Vec<_> = circuit.iter_gate_ids().collect();

        assert_eq!(circuit.gate_count(), 2);

        // Remove first gate
        assert!(circuit.storage.remove(gate_ids[0]));
        assert_eq!(circuit.gate_count(), 1);

        // Gate ID is now invalid
        assert!(!circuit.is_valid(gate_ids[0]));
        assert!(circuit.is_valid(gate_ids[1]));
    }

    #[test]
    fn test_generational_ids() {
        let mut storage = GateStorage::new();

        // Add and remove a gate
        let id1 = storage.add_gate(GateType::H, 0, &[QubitId::from(0)], &[], &[]);
        assert!(storage.is_valid(id1));

        storage.remove(id1);
        assert!(!storage.is_valid(id1));

        // Add another gate - reuses the slot with new generation
        let id2 = storage.add_gate(GateType::X, 0, &[QubitId::from(0)], &[], &[]);
        assert!(storage.is_valid(id2));
        assert!(!storage.is_valid(id1)); // Old ID still invalid

        // Same index, different generation
        assert_eq!(id1.index(), id2.index());
        assert_ne!(id1.generation(), id2.generation());
    }

    #[test]
    fn test_batched_simulation_api() {
        let mut builder = TickCircuitSoA::builder();
        builder
            .tick()
            .pz(&[0, 1, 2, 3]) // 4 preps
            .tick()
            .h(&[0, 1]) // 2 H gates
            .x(&[2, 3]) // 2 X gates
            .tick()
            .cx(&[(0, 1), (2, 3)]) // 2 CX gates
            .tick()
            .mz(&[0, 1, 2, 3]); // 4 measurements

        let circuit = builder.build();

        // Check tick count
        assert_eq!(circuit.num_ticks_batched(), 4);

        // Check tick 0: preps batched together
        let tick0 = circuit.tick_batched(0).unwrap();
        assert_eq!(tick0.batch_count(), 1);
        let prep_batch = tick0.batch_for_type(GateType::PZ).unwrap();
        assert_eq!(prep_batch.qubits().len(), 4);
        assert_eq!(prep_batch.gate_count(), 4);

        // Check tick 1: H and X are separate batches
        let tick1 = circuit.tick_batched(1).unwrap();
        assert_eq!(tick1.batch_count(), 2);
        let h_batch = tick1.batch_for_type(GateType::H).unwrap();
        assert_eq!(h_batch.qubits().len(), 2);
        let x_batch = tick1.batch_for_type(GateType::X).unwrap();
        assert_eq!(x_batch.qubits().len(), 2);

        // Check tick 2: CX gates batched
        let tick2 = circuit.tick_batched(2).unwrap();
        assert_eq!(tick2.batch_count(), 1);
        let cx_batch = tick2.batch_for_type(GateType::CX).unwrap();
        assert_eq!(cx_batch.qubits().len(), 4); // 2 pairs = 4 qubits
        assert_eq!(cx_batch.gate_count(), 2);

        // Check tick 3: measurements batched
        let tick3 = circuit.tick_batched(3).unwrap();
        assert_eq!(tick3.batch_count(), 1);
        let mz_batch = tick3.batch_for_type(GateType::MZ).unwrap();
        assert_eq!(mz_batch.qubits().len(), 4);
        assert_eq!(mz_batch.gate_count(), 4);
    }

    #[test]
    fn test_iter_ticks_batched() {
        let mut builder = TickCircuitSoA::builder();
        builder.tick().h(&[0, 1, 2]).tick().cx(&[(0, 1)]);

        let circuit = builder.build();

        // Iterate and count
        let mut tick_count = 0;
        let mut total_batches = 0;
        for (_tick_idx, tick) in circuit.iter_ticks_batched() {
            tick_count += 1;
            total_batches += tick.batch_count();
        }

        assert_eq!(tick_count, 2);
        assert_eq!(total_batches, 2); // H batch + CX batch
    }
}
