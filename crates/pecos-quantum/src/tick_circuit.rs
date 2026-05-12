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
//! // Preps break the chain but allow .meta():
//! circuit.tick().pz(&[0]).meta("reason", pecos_quantum::Attribute::String("init".into()));
//! // Measurements return refs for annotations:
//! let ms = circuit.tick().mz(&[0]);
//! circuit.detector(&ms);
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
//!
//! assert_eq!(circuit3.gate_count(), 14);       // individual gate applications
//! assert_eq!(circuit3.gate_batch_count(), 4);  // stored same-type batches
//! ```

use pecos_core::gate_type::GateType;
use pecos_core::{
    Angle64, ChannelExpr, Gate, GateMeasIds, GateQubits, GateSignature, MeasId, QubitId, TimeUnits,
};
use std::collections::{BTreeMap, BTreeSet};

use crate::Attribute;
use crate::dag_circuit::{AnnotationKind, DagCircuit, PauliAnnotation};
use std::fmt;
use std::ops::{Deref, Index};

fn meta_json_array(circuit: &TickCircuit, key: &str) -> Result<Vec<serde_json::Value>, String> {
    let Some(attr) = circuit.get_meta(key) else {
        return Ok(Vec::new());
    };
    match attr {
        Attribute::String(s) => {
            if s.trim().is_empty() {
                return Ok(Vec::new());
            }
            serde_json::from_str::<Vec<serde_json::Value>>(s)
                .map_err(|e| format!("metadata {key:?} must be a JSON array: {e}"))
        }
        Attribute::Json(serde_json::Value::Array(values)) => Ok(values.clone()),
        _ => Err(format!(
            "metadata {key:?} must be a JSON array string or JSON array"
        )),
    }
}

fn set_meta_json_array(
    circuit: &mut TickCircuit,
    key: &str,
    values: &[serde_json::Value],
) -> Result<(), String> {
    let json =
        serde_json::to_string(values).map_err(|e| format!("could not serialize {key:?}: {e}"))?;
    circuit.set_meta(key, Attribute::String(json));
    Ok(())
}

fn json_metadata_id(key: &str, id: u64) -> Result<usize, String> {
    usize::try_from(id).map_err(|_| format!("metadata {key:?} id {id} does not fit usize"))
}

fn next_metadata_id(values: &[serde_json::Value], key: &str) -> Result<usize, String> {
    let mut max_id = None;
    for value in values {
        if let Some(id) = value.get("id").and_then(serde_json::Value::as_u64) {
            let id = json_metadata_id(key, id)?;
            max_id = Some(max_id.map_or(id, |max_id: usize| max_id.max(id)));
        }
    }
    match max_id {
        Some(max_id) => max_id
            .checked_add(1)
            .ok_or_else(|| format!("metadata {key:?} id counter overflow")),
        None => Ok(values.len()),
    }
}

fn metadata_count(values: &[serde_json::Value], key: &str) -> Result<usize, String> {
    let mut count = values.len();
    for value in values {
        if let Some(id) = value.get("id").and_then(serde_json::Value::as_u64) {
            let next = json_metadata_id(key, id)?
                .checked_add(1)
                .ok_or_else(|| format!("metadata {key:?} count overflow"))?;
            count = count.max(next);
        }
    }
    Ok(count)
}

fn metadata_count_attr(values: &[serde_json::Value], key: &str) -> Result<Attribute, String> {
    let count = metadata_count(values, key)?;
    let count = i64::try_from(count)
        .map_err(|_| format!("metadata {key:?} count {count} does not fit i64"))?;
    Ok(Attribute::Int(count))
}

fn ensure_unique_metadata_id(
    values: &[serde_json::Value],
    key: &str,
    id_name: &str,
    id: usize,
) -> Result<(), String> {
    for value in values {
        if let Some(existing) = value.get("id").and_then(serde_json::Value::as_u64) {
            let existing = json_metadata_id(key, existing)?;
            if existing == id {
                return Err(format!("{key} metadata already contains {id_name} {id}"));
            }
        }
    }
    Ok(())
}

fn observable_id_from_label(label: Option<&str>) -> Result<Option<usize>, String> {
    let Some(label) = label else {
        return Ok(None);
    };
    let Some(rest) = label.strip_prefix('L') else {
        return Ok(None);
    };
    if rest.is_empty() {
        return Ok(None);
    }
    rest.parse::<usize>().map(Some).map_err(|_| {
        format!(
            "observable label {label:?} starts with 'L' but does not contain a valid integer id"
        )
    })
}

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

/// Error when trying to add a gate to a tick.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TickGateError {
    /// The gate payload itself is invalid.
    InvalidGate {
        /// Validation error from [`Gate::validate`].
        message: String,
        /// The tick index where the invalid gate was being inserted.
        tick_idx: Option<usize>,
    },
    /// The gate is valid, but overlaps a qubit already used in this tick.
    QubitConflict(QubitConflictError),
}

impl TickGateError {
    fn set_tick_idx(&mut self, tick_idx: usize) {
        match self {
            Self::InvalidGate { tick_idx: idx, .. } => *idx = Some(tick_idx),
            Self::QubitConflict(err) => err.tick_idx = Some(tick_idx),
        }
    }
}

impl fmt::Display for TickGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGate {
                message,
                tick_idx: Some(tick_idx),
            } => write!(f, "Invalid gate in tick {tick_idx}: {message}"),
            Self::InvalidGate {
                message,
                tick_idx: None,
            } => write!(f, "Invalid gate: {message}"),
            Self::QubitConflict(err) => write!(f, "{err}"),
        }
    }
}

impl std::error::Error for TickGateError {}

impl From<QubitConflictError> for TickGateError {
    fn from(e: QubitConflictError) -> Self {
        Self::QubitConflict(e)
    }
}

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
    InvalidGate(String),
    QubitConflict(QubitConflictError),
}

impl fmt::Display for CustomGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SignatureMismatch(e) => write!(f, "{e}"),
            Self::InvalidGate(e) => write!(f, "{e}"),
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

impl From<TickGateError> for CustomGateError {
    fn from(e: TickGateError) -> Self {
        match e {
            TickGateError::InvalidGate { message, .. } => Self::InvalidGate(message),
            TickGateError::QubitConflict(err) => Self::QubitConflict(err),
        }
    }
}

#[derive(Debug, Clone, Default)]
struct TickGateStorage {
    commands: Vec<Gate>,
}

impl TickGateStorage {
    fn len(&self) -> usize {
        self.commands.len()
    }

    fn is_empty(&self) -> bool {
        self.commands.is_empty()
    }

    fn as_slice(&self) -> &[Gate] {
        &self.commands
    }

    fn iter(&self) -> std::slice::Iter<'_, Gate> {
        self.commands.iter()
    }

    fn get(&self, idx: usize) -> Option<&Gate> {
        self.commands.get(idx)
    }

    fn push(&mut self, gate: Gate) {
        self.commands.push(gate);
    }

    fn set(&mut self, idx: usize, gate: Gate) {
        self.commands[idx] = gate;
    }

    fn remove(&mut self, idx: usize) -> Gate {
        self.commands.remove(idx)
    }

    fn append_batch(&mut self, idx: usize, gate: Gate) {
        assert!(
            self.commands[idx].can_batch_with(&gate),
            "cannot merge incompatible gate batches"
        );
        self.commands[idx].append_batch(gate);
    }

    fn truncate_payload(&mut self, idx: usize, qubit_len: usize, meas_id_len: usize) {
        self.commands[idx].qubits.truncate(qubit_len);
        self.commands[idx].meas_ids.truncate(meas_id_len);
    }
}

impl Deref for TickGateStorage {
    type Target = [Gate];

    fn deref(&self) -> &Self::Target {
        self.as_slice()
    }
}

impl Index<usize> for TickGateStorage {
    type Output = Gate;

    fn index(&self, index: usize) -> &Self::Output {
        &self.commands[index]
    }
}

/// A single time slice containing gates that execute in parallel.
#[derive(Debug, Clone, Default)]
pub struct Tick {
    /// Gate batches in this tick (all act on disjoint qubits).
    gate_batches: TickGateStorage,
    /// Metadata for each gate batch, indexed by position in `gate_batches`.
    batch_attrs: Vec<Option<BTreeMap<String, Attribute>>>,
    /// Tick-level metadata.
    attrs: BTreeMap<String, Attribute>,
}

#[derive(Debug, Clone, Copy)]
struct GateBatchPiece {
    gate_idx: usize,
    qubit_start: usize,
    qubit_len: usize,
    meas_id_start: usize,
    meas_id_len: usize,
}

/// Borrowed view of one stored same-type gate batch in a [`Tick`].
#[derive(Debug, Clone, Copy)]
pub struct GateBatchRef<'a> {
    batch_index: usize,
    gate: &'a Gate,
    attrs: Option<&'a BTreeMap<String, Attribute>>,
}

impl<'a> GateBatchRef<'a> {
    fn new(
        batch_index: usize,
        gate: &'a Gate,
        attrs: Option<&'a BTreeMap<String, Attribute>>,
    ) -> Self {
        Self {
            batch_index,
            gate,
            attrs,
        }
    }

    /// Return this batch's index within its tick.
    #[must_use]
    pub fn batch_index(self) -> usize {
        self.batch_index
    }

    /// Return the underlying stored [`Gate`] batch.
    #[must_use]
    pub fn as_gate(self) -> &'a Gate {
        self.gate
    }

    /// Return the number of individual gates represented by this batch.
    #[must_use]
    pub fn gate_count(self) -> usize {
        self.gate.num_gates()
    }

    /// Return the metadata attribute with the given key, if present.
    #[must_use]
    pub fn get_attr(self, key: &str) -> Option<&'a Attribute> {
        self.attrs.and_then(|attrs| attrs.get(key))
    }

    /// Iterate over metadata attributes attached to this batch.
    pub fn attrs(self) -> impl Iterator<Item = (&'a String, &'a Attribute)> {
        self.attrs.into_iter().flat_map(|attrs| attrs.iter())
    }

    /// Return one individual gate from this batch.
    #[must_use]
    pub fn instance(self, instance_index: usize) -> Option<GateInstanceRef<'a>> {
        let gate_count = self.gate_count();
        if instance_index >= gate_count {
            return None;
        }

        let qubits = if gate_count == 1 {
            self.gate.qubits.as_slice()
        } else {
            let arity = self.gate.quantum_arity();
            let start = instance_index * arity;
            let end = start + arity;
            self.gate.qubits.get(start..end)?
        };

        let meas_ids = if self.gate.meas_ids.is_empty() {
            &self.gate.meas_ids[0..0]
        } else if gate_count == 1 {
            self.gate.meas_ids.as_slice()
        } else {
            if !self.gate.meas_ids.len().is_multiple_of(gate_count) {
                return None;
            }
            let arity = self.gate.meas_ids.len() / gate_count;
            let start = instance_index * arity;
            let end = start + arity;
            self.gate.meas_ids.get(start..end)?
        };

        Some(GateInstanceRef {
            batch: self,
            instance_index,
            qubits,
            meas_ids,
        })
    }

    /// Iterate over individual gates represented by this batch.
    pub fn iter_gate_instances(self) -> impl Iterator<Item = GateInstanceRef<'a>> {
        (0..self.gate_count()).filter_map(move |idx| self.instance(idx))
    }
}

impl Deref for GateBatchRef<'_> {
    type Target = Gate;

    fn deref(&self) -> &Self::Target {
        self.gate
    }
}

/// Borrowed view of one individual gate inside a [`GateBatchRef`].
#[derive(Debug, Clone, Copy)]
pub struct GateInstanceRef<'a> {
    batch: GateBatchRef<'a>,
    instance_index: usize,
    qubits: &'a [QubitId],
    meas_ids: &'a [MeasId],
}

impl<'a> GateInstanceRef<'a> {
    /// Return the stored batch this individual gate came from.
    #[must_use]
    pub fn batch(self) -> GateBatchRef<'a> {
        self.batch
    }

    /// Return the parent batch's index within its tick.
    #[must_use]
    pub fn batch_index(self) -> usize {
        self.batch.batch_index()
    }

    /// Return this gate's position within its stored batch.
    #[must_use]
    pub fn instance_index(self) -> usize {
        self.instance_index
    }

    /// Return this individual gate's type.
    #[must_use]
    pub fn gate_type(self) -> GateType {
        self.batch.gate_type
    }

    /// Return this individual gate's qubit support.
    #[must_use]
    pub fn qubits(self) -> &'a [QubitId] {
        self.qubits
    }

    /// Return this individual gate's measurement ids, if any.
    #[must_use]
    pub fn meas_ids(self) -> &'a [MeasId] {
        self.meas_ids
    }

    /// Return this individual gate's rotation angles.
    #[must_use]
    pub fn angles(self) -> &'a [Angle64] {
        self.batch.gate.angles.as_slice()
    }

    /// Return this individual gate's non-angle parameters.
    #[must_use]
    pub fn params(self) -> &'a [f64] {
        self.batch.gate.params.as_slice()
    }

    /// Return this individual gate's channel payload, if this is a channel.
    #[must_use]
    pub fn channel(self) -> Option<&'a ChannelExpr> {
        self.batch.gate.channel.as_ref()
    }

    /// Return the metadata attribute with the given key, if present.
    #[must_use]
    pub fn get_attr(self, key: &str) -> Option<&'a Attribute> {
        self.batch.get_attr(key)
    }

    /// Iterate over metadata attributes attached to the parent batch.
    pub fn attrs(self) -> impl Iterator<Item = (&'a String, &'a Attribute)> {
        self.batch.attrs()
    }

    /// Materialize this individual gate as an owned [`Gate`].
    ///
    /// The returned gate carries this instance's sliced qubits and measurement
    /// ids, plus the parent batch's gate type, angles, parameters, and channel
    /// payload. Batch metadata is intentionally not copied into the `Gate`;
    /// use [`attrs`](Self::attrs) when metadata needs to travel alongside the
    /// materialized operation.
    #[must_use]
    pub fn to_gate(self) -> Gate {
        Gate {
            gate_type: self.gate_type(),
            qubits: self.qubits.iter().copied().collect::<GateQubits>(),
            angles: self.batch.gate.angles.clone(),
            params: self.batch.gate.params.clone(),
            meas_ids: self.meas_ids.iter().copied().collect::<GateMeasIds>(),
            channel: self.batch.gate.channel.clone(),
        }
    }
}

impl Tick {
    /// Create a new empty tick.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the number of stored gate batches in this tick.
    ///
    /// This is the number of stored gate batches. Batched commands such as
    /// `cx(&[(0, 1), (2, 3)])` count as one stored batch.
    #[must_use]
    pub fn len(&self) -> usize {
        self.gate_batches.len()
    }

    /// Get the number of individual gate applications in this tick.
    ///
    /// Batched commands count by the number of gates they represent:
    /// `h(&[0, 1, 2])` counts as three H gates, and
    /// `cx(&[(0, 1), (2, 3)])` counts as two CX gates.
    #[must_use]
    pub fn gate_count(&self) -> usize {
        self.gate_batches.iter().map(Gate::num_gates).sum()
    }

    /// Get the number of compatible gate batches in this tick.
    ///
    /// Gates with the same type, parameters, payload shape, and metadata can
    /// execute as one batch when they differ only by disjoint qubits.
    #[must_use]
    pub fn gate_batch_count(&self) -> usize {
        let mut representative_indices: Vec<usize> = Vec::new();
        'gate: for (idx, gate) in self.gate_batches.iter().enumerate() {
            for &rep_idx in &representative_indices {
                if self.gate_attrs_equivalent(rep_idx, idx)
                    && self.gate_batches[rep_idx].can_batch_with(gate)
                {
                    continue 'gate;
                }
            }
            representative_indices.push(idx);
        }
        representative_indices.len()
    }

    /// Check if the tick is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.gate_batches.is_empty()
    }

    /// Get the raw stored gate batches in this tick.
    ///
    /// In `TickCircuit`, a stored [`Gate`] command may represent one or more
    /// individual gates on disjoint qubits. For example,
    /// `cx(&[(0, 1), (2, 3)])` is one batch containing two gates.
    ///
    /// The batches preserve the complete [`Gate`] payload, including
    /// measurement IDs and typed channel payloads.
    ///
    /// Prefer [`iter_gate_batches`](Self::iter_gate_batches) for new read-only
    /// consumers; it returns [`GateBatchRef`] values that carry the batch index
    /// and metadata alongside the underlying `Gate`. This raw slice accessor is
    /// kept for compatibility, storage inspection, and code that intentionally
    /// needs stable batch indices.
    #[must_use]
    pub fn gate_batches(&self) -> &[Gate] {
        self.gate_batches.as_slice()
    }

    /// Iterate over full-fidelity borrowed gate-batch views in this tick.
    pub fn iter_gate_batches(&self) -> impl Iterator<Item = GateBatchRef<'_>> {
        self.gate_batches
            .iter()
            .enumerate()
            .map(|(idx, gate)| GateBatchRef::new(idx, gate, self.normalized_gate_attrs(idx)))
    }

    /// Iterate over individual gates expanded from this tick's stored batches.
    pub fn iter_gate_instances(&self) -> impl Iterator<Item = GateInstanceRef<'_>> {
        self.iter_gate_batches()
            .flat_map(GateBatchRef::iter_gate_instances)
    }

    /// Add a gate to this tick.
    ///
    /// # Panics
    ///
    /// Panics if [`Gate::validate`] rejects the gate payload or if the gate
    /// conflicts with an existing gate in this tick. Use
    /// [`try_add_gate`](Self::try_add_gate) for fallible insertion.
    pub fn add_gate(&mut self, gate: Gate) -> usize {
        self.try_add_gate(gate)
            .unwrap_or_else(|err| panic!("{err}"))
    }

    fn push_gate_unchecked(&mut self, gate: Gate) -> usize {
        let idx = self.gate_batches.len();
        self.gate_batches.push(gate);
        self.batch_attrs.push(None);
        idx
    }

    fn push_gate_unchecked_piece(&mut self, gate: Gate) -> GateBatchPiece {
        let qubit_len = gate.qubits.len();
        let meas_id_len = gate.meas_ids.len();
        let gate_idx = self.push_gate_unchecked(gate);
        GateBatchPiece {
            gate_idx,
            qubit_start: 0,
            qubit_len,
            meas_id_start: 0,
            meas_id_len,
        }
    }

    fn normalized_gate_attrs(&self, gate_idx: usize) -> Option<&BTreeMap<String, Attribute>> {
        self.batch_attrs
            .get(gate_idx)
            .and_then(Option::as_ref)
            .filter(|attrs| !attrs.is_empty())
    }

    fn gate_attrs_equivalent(&self, a: usize, b: usize) -> bool {
        self.normalized_gate_attrs(a) == self.normalized_gate_attrs(b)
    }

    fn gate_has_no_attrs(&self, gate_idx: usize) -> bool {
        self.normalized_gate_attrs(gate_idx).is_none()
    }

    fn compatible_empty_attr_batch(&self, gate: &Gate) -> Option<usize> {
        self.gate_batches
            .iter()
            .enumerate()
            .find(|(idx, existing)| self.gate_has_no_attrs(*idx) && existing.can_batch_with(gate))
            .map(|(idx, _)| idx)
    }

    fn whole_gate_piece(&self, gate_idx: usize) -> GateBatchPiece {
        let gate = &self.gate_batches[gate_idx];
        GateBatchPiece {
            gate_idx,
            qubit_start: 0,
            qubit_len: gate.qubits.len(),
            meas_id_start: 0,
            meas_id_len: gate.meas_ids.len(),
        }
    }

    fn merge_compatible_piece_at(&mut self, piece: GateBatchPiece) -> GateBatchPiece {
        if piece.gate_idx >= self.gate_batches.len() {
            return piece;
        }

        let Some(target_idx) = (0..piece.gate_idx).find(|&idx| {
            self.gate_attrs_equivalent(idx, piece.gate_idx)
                && self.gate_batches[idx].can_batch_with(&self.gate_batches[piece.gate_idx])
        }) else {
            return piece;
        };

        let qubit_start = self.gate_batches[target_idx].qubits.len();
        let meas_id_start = self.gate_batches[target_idx].meas_ids.len();
        let gate = self.gate_batches[piece.gate_idx].clone();
        self.gate_batches.append_batch(target_idx, gate);
        self.remove_gate(piece.gate_idx);
        GateBatchPiece {
            gate_idx: target_idx,
            qubit_start,
            qubit_len: piece.qubit_len,
            meas_id_start,
            meas_id_len: piece.meas_id_len,
        }
    }

    fn merge_compatible_gate_at(&mut self, gate_idx: usize) -> usize {
        if gate_idx >= self.gate_batches.len() {
            return gate_idx;
        }
        self.merge_compatible_piece_at(self.whole_gate_piece(gate_idx))
            .gate_idx
    }

    fn isolate_batch_piece(&mut self, piece: GateBatchPiece) -> usize {
        if piece.gate_idx >= self.gate_batches.len() {
            return piece.gate_idx;
        }

        let gate_qubit_len = self.gate_batches[piece.gate_idx].qubits.len();
        let gate_meas_id_len = self.gate_batches[piece.gate_idx].meas_ids.len();
        if piece.qubit_start == 0
            && piece.qubit_len == gate_qubit_len
            && piece.meas_id_start == 0
            && piece.meas_id_len == gate_meas_id_len
        {
            return piece.gate_idx;
        }

        assert_eq!(
            piece.qubit_start + piece.qubit_len,
            gate_qubit_len,
            "batched gate metadata can only split the appended suffix"
        );
        assert_eq!(
            piece.meas_id_start + piece.meas_id_len,
            gate_meas_id_len,
            "batched gate metadata can only split the appended measurement-id suffix"
        );

        let mut split_gate = self.gate_batches[piece.gate_idx].clone();
        split_gate.qubits = self.gate_batches[piece.gate_idx].qubits[piece.qubit_start..].into();
        split_gate.meas_ids =
            self.gate_batches[piece.gate_idx].meas_ids[piece.meas_id_start..].into();

        self.gate_batches
            .truncate_payload(piece.gate_idx, piece.qubit_start, piece.meas_id_start);

        let split_idx = self.push_gate_unchecked(split_gate);
        if let Some(attrs) = self.normalized_gate_attrs(piece.gate_idx).cloned() {
            self.batch_attrs[split_idx] = Some(attrs);
        }
        split_idx
    }

    fn set_gate_attr_for_piece(
        &mut self,
        piece: GateBatchPiece,
        key: &str,
        value: Attribute,
    ) -> GateBatchPiece {
        let gate_idx = self.isolate_batch_piece(piece);
        self.set_gate_attr(gate_idx, key, value);
        self.merge_compatible_piece_at(self.whole_gate_piece(gate_idx))
    }

    fn set_gate_attrs_for_piece(
        &mut self,
        piece: GateBatchPiece,
        attrs: BTreeMap<String, Attribute>,
    ) -> GateBatchPiece {
        let gate_idx = self.isolate_batch_piece(piece);
        self.set_gate_attrs(gate_idx, attrs);
        self.merge_compatible_piece_at(self.whole_gate_piece(gate_idx))
    }

    /// Set metadata on a gate.
    ///
    /// Returns the gate index.
    ///
    /// # Panics
    ///
    /// Panics if `gate_idx` is not a valid stored batch index in this tick.
    pub fn set_gate_attr(&mut self, gate_idx: usize, key: &str, value: Attribute) -> usize {
        assert!(
            gate_idx < self.gate_batches.len(),
            "gate index {gate_idx} out of bounds"
        );
        self.batch_attrs[gate_idx]
            .get_or_insert_with(BTreeMap::new)
            .insert(key.to_string(), value);
        gate_idx
    }

    /// Set multiple metadata attributes on a gate at once.
    ///
    /// Returns the gate index.
    ///
    /// # Panics
    ///
    /// Panics if `gate_idx` is not a valid stored batch index in this tick.
    pub fn set_gate_attrs(&mut self, gate_idx: usize, attrs: BTreeMap<String, Attribute>) -> usize {
        assert!(
            gate_idx < self.gate_batches.len(),
            "gate index {gate_idx} out of bounds"
        );
        if !attrs.is_empty() {
            self.batch_attrs[gate_idx]
                .get_or_insert_with(BTreeMap::new)
                .extend(attrs);
        }
        gate_idx
    }

    /// Get metadata from a gate.
    #[must_use]
    pub fn get_gate_attr(&self, gate_idx: usize, key: &str) -> Option<&Attribute> {
        self.batch_attrs
            .get(gate_idx)
            .and_then(Option::as_ref)
            .and_then(|m| m.get(key))
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
        self.batch_attrs
            .get(gate_idx)
            .and_then(Option::as_ref)
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
        self.gate_batches
            .iter()
            .flat_map(|gate| gate.qubits.iter().copied())
            .collect()
    }

    /// Check if a specific qubit is already in use in this tick.
    #[must_use]
    pub fn uses_qubit(&self, qubit: QubitId) -> bool {
        self.gate_batches
            .iter()
            .any(|gate| gate.qubits.contains(&qubit))
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

    /// Try to add a gate to this tick.
    ///
    /// # Errors
    ///
    /// Returns [`TickGateError::InvalidGate`] if the gate payload is invalid, or
    /// [`TickGateError::QubitConflict`] if any qubit in the gate is already used
    /// by another gate in this tick.
    pub(crate) fn try_add_gate_preserving_command(
        &mut self,
        gate: Gate,
    ) -> Result<usize, TickGateError> {
        gate.validate()
            .map_err(|message| TickGateError::InvalidGate {
                message,
                tick_idx: None,
            })?;
        let conflicts = self.find_conflicts(&gate.qubits);
        if !conflicts.is_empty() {
            return Err(TickGateError::QubitConflict(QubitConflictError {
                conflicting_qubits: conflicts,
                tick_idx: None,
            }));
        }
        Ok(self.push_gate_unchecked(gate))
    }

    /// Try to add a gate to this tick.
    ///
    /// # Errors
    ///
    /// Returns [`TickGateError::InvalidGate`] if the gate payload is invalid, or
    /// [`TickGateError::QubitConflict`] if any qubit in the gate is already used
    /// by another gate in this tick.
    pub fn try_add_gate(&mut self, gate: Gate) -> Result<usize, TickGateError> {
        self.try_add_gate_piece(gate).map(|piece| piece.gate_idx)
    }

    fn try_add_gate_piece(&mut self, gate: Gate) -> Result<GateBatchPiece, TickGateError> {
        gate.validate()
            .map_err(|message| TickGateError::InvalidGate {
                message,
                tick_idx: None,
            })?;
        let conflicts = self.find_conflicts(&gate.qubits);
        if !conflicts.is_empty() {
            return Err(TickGateError::QubitConflict(QubitConflictError {
                conflicting_qubits: conflicts,
                tick_idx: None,
            }));
        }
        if let Some(gate_idx) = self.compatible_empty_attr_batch(&gate) {
            let piece = GateBatchPiece {
                gate_idx,
                qubit_start: self.gate_batches[gate_idx].qubits.len(),
                qubit_len: gate.qubits.len(),
                meas_id_start: self.gate_batches[gate_idx].meas_ids.len(),
                meas_id_len: gate.meas_ids.len(),
            };
            self.gate_batches.append_batch(gate_idx, gate);
            return Ok(piece);
        }
        Ok(self.push_gate_unchecked_piece(gate))
    }

    /// Replace a stored gate batch while preserving storage invariants.
    ///
    /// # Errors
    ///
    /// Returns [`TickGateError::InvalidGate`] if `gate_idx` is out of bounds or
    /// if the replacement gate payload is invalid. Returns
    /// [`TickGateError::QubitConflict`] if the replacement overlaps another
    /// command in this tick.
    pub fn replace_gate_batch(&mut self, gate_idx: usize, gate: Gate) -> Result<(), TickGateError> {
        if gate_idx >= self.gate_batches.len() {
            return Err(TickGateError::InvalidGate {
                message: format!("gate index {gate_idx} out of bounds"),
                tick_idx: None,
            });
        }
        gate.validate()
            .map_err(|message| TickGateError::InvalidGate {
                message,
                tick_idx: None,
            })?;

        let mut active = BTreeSet::new();
        for (idx, existing) in self.gate_batches.iter().enumerate() {
            if idx == gate_idx {
                continue;
            }
            active.extend(existing.qubits.iter().copied());
        }
        let conflicts: Vec<QubitId> = gate
            .qubits
            .iter()
            .filter(|q| active.contains(q))
            .copied()
            .collect();
        if !conflicts.is_empty() {
            return Err(TickGateError::QubitConflict(QubitConflictError {
                conflicting_qubits: conflicts,
                tick_idx: None,
            }));
        }

        self.gate_batches.set(gate_idx, gate);
        Ok(())
    }

    /// Mutate a stored gate batch through a temporary [`Gate`] value.
    ///
    /// The updated gate batch is validated with the same conflict checks as a
    /// full replacement before it is written back.
    ///
    /// # Errors
    ///
    /// Propagates the same errors as [`replace_gate_batch`](Self::replace_gate_batch).
    pub fn update_gate_batch(
        &mut self,
        gate_idx: usize,
        update: impl FnOnce(&mut Gate),
    ) -> Result<(), TickGateError> {
        let Some(existing) = self.gate_batches.get(gate_idx) else {
            return Err(TickGateError::InvalidGate {
                message: format!("gate index {gate_idx} out of bounds"),
                tick_idx: None,
            });
        };
        let mut gate = existing.clone();
        update(&mut gate);
        self.replace_gate_batch(gate_idx, gate)
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
            .gate_batches
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
            let _ = self.remove_gate(idx);
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
        if idx >= self.gate_batches.len() {
            return None;
        }

        let gate = self.gate_batches.remove(idx);
        self.batch_attrs.remove(idx);

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
/// Measurement reference in a tick circuit: (`tick_index`, `gate_index`, qubit).
///
/// Returned by `mz()` for use in `detector()` and `observable()` annotations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TickMeasRef {
    /// Tick index.
    pub tick: usize,
    /// Gate index within the tick.
    pub gate_idx: usize,
    /// Qubit that was measured.
    pub qubit: QubitId,
    /// Measurement record index (cumulative count of MZ qubits in circuit order).
    pub record_idx: usize,
    /// Stable measurement result identity (SSA value).
    pub meas_id: MeasId,
}

#[derive(Debug, Clone, Default)]
pub struct TickCircuit {
    /// The sequence of ticks.
    ticks: Vec<Tick>,
    /// Next tick index to allocate.
    next_tick: usize,
    /// Circuit-level metadata.
    circuit_attrs: BTreeMap<String, Attribute>,
    /// Gate signatures for custom gate validation (JIT + AOT).
    gate_signatures: BTreeMap<String, GateSignature>,
    /// Unified Pauli annotations (detectors, observables, operators).
    annotations: Vec<PauliAnnotation>,
    /// Running count of measurement records (incremented by each MZ qubit).
    next_meas_record: usize,
}

/// Maps an ideal circuit gate to zero or more channel annotations.
///
/// [`TickCircuit::with_noise`] uses this trait to compile an ideal circuit into
/// an annotated circuit with explicit channel operations interleaved after the
/// ideal gates that triggered them.
pub trait GateNoiseModel {
    /// Returns channel operations that should be placed after `gate`.
    fn channels_after(&self, gate: &Gate) -> Vec<ChannelExpr>;
}

impl<F> GateNoiseModel for F
where
    F: Fn(&Gate) -> Vec<ChannelExpr>,
{
    fn channels_after(&self, gate: &Gate) -> Vec<ChannelExpr> {
        self(gate)
    }
}

fn schedule_channel_gate(noise_ticks: &mut Vec<Tick>, gate: Gate) {
    let mut pending = Some(gate);
    for tick in noise_ticks.iter_mut() {
        let gate_ref = pending.as_ref().expect("pending gate is present");
        if tick.find_conflicts(&gate_ref.qubits).is_empty() {
            tick.add_gate(pending.take().expect("pending gate is present"));
            return;
        }
    }

    let mut tick = Tick::new();
    tick.add_gate(pending.expect("pending gate is present"));
    noise_ticks.push(tick);
}

/// Handle to a specific tick for adding gates.
///
/// Gates added through the handle are placed in the associated tick.
/// The handle chains for fluent API usage.
pub struct TickHandle<'a> {
    circuit: &'a mut TickCircuit,
    tick_idx: usize,
    last_gate_idx: Option<usize>,
    last_gate_piece: Option<GateBatchPiece>,
}

/// Handle returned by preparation operations on a tick.
///
/// This handle breaks the method chain (unlike regular gates),
/// but still allows attaching metadata via `.meta()`.
pub struct TickPrepHandle<'a> {
    circuit: &'a mut TickCircuit,
    tick_idx: usize,
    gate_piece: GateBatchPiece,
}

impl TickPrepHandle<'_> {
    /// Add metadata to this preparation.
    ///
    /// Returns `()` to break the chain.
    pub fn meta(self, key: &str, value: impl Into<Attribute>) {
        if let Some(tick) = self.circuit.get_tick_mut(self.tick_idx) {
            tick.set_gate_attr_for_piece(self.gate_piece, key, value.into());
        }
    }

    /// Add multiple metadata attributes to this preparation.
    ///
    /// Returns `()` to break the chain.
    pub fn metas(self, attrs: BTreeMap<String, Attribute>) {
        if let Some(tick) = self.circuit.get_tick_mut(self.tick_idx) {
            tick.set_gate_attrs_for_piece(self.gate_piece, attrs);
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
    gate_piece: GateBatchPiece,
}

impl TickMeasureHandle<'_> {
    /// Add metadata to this measurement.
    ///
    /// Returns `()` to break the chain.
    pub fn meta(self, key: &str, value: impl Into<Attribute>) {
        if let Some(tick) = self.circuit.get_tick_mut(self.tick_idx) {
            tick.set_gate_attr_for_piece(self.gate_piece, key, value.into());
        }
    }

    /// Add multiple metadata attributes to this measurement.
    ///
    /// Returns `()` to break the chain.
    pub fn metas(self, attrs: BTreeMap<String, Attribute>) {
        if let Some(tick) = self.circuit.get_tick_mut(self.tick_idx) {
            tick.set_gate_attrs_for_piece(self.gate_piece, attrs);
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
            gate_signatures: BTreeMap::new(),
            annotations: Vec::new(),
            next_meas_record: 0,
        }
    }

    /// Get the number of ticks in the circuit.
    #[must_use]
    pub fn num_ticks(&self) -> usize {
        self.ticks.len()
    }

    /// Total number of measurement results produced so far.
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.next_meas_record
    }

    /// Advance the measurement counter by `n` (for external MZ gate construction).
    pub fn advance_meas_counter(&mut self, n: usize) {
        self.next_meas_record += n;
    }

    /// Get the total number of individual gate applications across all ticks.
    ///
    /// Batched commands count by individual gate. For example,
    /// `cx(&[(0, 1), (2, 3)])` contributes two gates.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0, 1, 2]);
    /// circuit.tick().cx(&[(0, 1), (2, 3)]);
    ///
    /// assert_eq!(circuit.gate_count(), 5);
    /// assert_eq!(circuit.gate_batch_count(), 2);
    /// ```
    #[must_use]
    pub fn gate_count(&self) -> usize {
        self.ticks.iter().map(Tick::gate_count).sum()
    }

    /// Get the total number of compatible gate batches across all ticks.
    ///
    /// A batch is a stored command group that can execute together because the
    /// gates are identical except for disjoint qubit support and compatible
    /// metadata.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0]).h(&[1]).cx(&[(2, 3), (4, 5)]);
    ///
    /// assert_eq!(circuit.gate_count(), 4);
    /// assert_eq!(circuit.gate_batch_count(), 2); // one H batch, one CX batch
    /// ```
    #[must_use]
    pub fn gate_batch_count(&self) -> usize {
        self.ticks.iter().map(Tick::gate_batch_count).sum()
    }

    /// Convert a per-tick gate index to a global gate index.
    ///
    /// Global index = sum of stored gate batches for all ticks before
    /// `tick_idx` + `gate_idx`.
    #[must_use]
    pub fn global_gate_index(&self, tick_idx: usize, gate_idx: usize) -> usize {
        self.ticks[..tick_idx].iter().map(Tick::len).sum::<usize>() + gate_idx
    }

    /// Get a tick by index.
    #[must_use]
    pub fn get_tick(&self, idx: usize) -> Option<&Tick> {
        self.ticks.get(idx)
    }

    /// Get a mutable tick by index.
    ///
    /// Mutating a tick through this handle must preserve the usual
    /// `TickCircuit` invariants: each stored [`Gate`] must validate, qubits may
    /// not overlap within a tick, and gate metadata must stay aligned with the
    /// stored batches. Prefer `Tick` methods such as
    /// [`Tick::add_gate`], [`Tick::remove_gate`], and
    /// [`Tick::replace_gate_batch`] over direct structural rewrites.
    pub fn get_tick_mut(&mut self, idx: usize) -> Option<&mut Tick> {
        self.ticks.get_mut(idx)
    }

    /// Get all ticks.
    #[must_use]
    pub fn ticks(&self) -> &[Tick] {
        &self.ticks
    }

    /// Get mutable access to all ticks as a slice.
    ///
    /// This is an escape hatch for passes that need to mutate existing ticks in
    /// place. Do not reorder, insert, or remove ticks through this slice. Keep
    /// each tick's gate/metadata invariants intact by using `Tick` mutation
    /// methods rather than editing stored batches directly.
    pub fn ticks_mut(&mut self) -> &mut [Tick] {
        &mut self.ticks
    }

    /// Remove all ticks from this circuit for an internal structural rewrite.
    ///
    /// This is crate-private because a partially drained circuit has
    /// temporarily invalid tick structure. Pair it with
    /// [`replace_ticks`](Self::replace_ticks) in the same transformation.
    pub(crate) fn take_ticks(&mut self) -> Vec<Tick> {
        let ticks = std::mem::take(&mut self.ticks);
        self.next_tick = 0;
        ticks
    }

    /// Replace all ticks after an internal structural rewrite.
    ///
    /// Updates `next_tick` to match the replacement length. The caller remains
    /// responsible for preserving measurement record numbering, annotation
    /// references, tick ordering, and each tick's gate/metadata alignment.
    pub(crate) fn replace_ticks(&mut self, ticks: Vec<Tick>) {
        self.ticks = ticks;
        self.next_tick = self.ticks.len();
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
            .map(|t| t.gate_batches().iter().collect())
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
            last_gate_piece: None,
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

    /// Add detector metadata defined by measurement-record offsets.
    ///
    /// This appends one entry to the circuit-level `"detectors"` JSON metadata
    /// list and updates `"num_detectors"`. It is intended for circuits whose
    /// detector definitions are stored in metadata rather than as direct
    /// [`TickMeasRef`] annotations.
    ///
    /// # Errors
    ///
    /// Returns an error if existing detector metadata is not a JSON array, if
    /// JSON serialization fails, or if `detector_id` duplicates an existing
    /// explicit detector id.
    pub fn add_detector_metadata(
        &mut self,
        records: &[i64],
        coords: Option<&[f64]>,
        label: Option<&str>,
        detector_id: Option<usize>,
    ) -> Result<usize, String> {
        let mut detectors = meta_json_array(self, "detectors")?;
        let id = match detector_id {
            Some(id) => id,
            None => next_metadata_id(&detectors, "detectors")?,
        };
        ensure_unique_metadata_id(&detectors, "detectors", "detector_id", id)?;

        let mut detector = serde_json::Map::new();
        detector.insert("id".to_string(), serde_json::json!(id));
        detector.insert("records".to_string(), serde_json::json!(records));
        if let Some(coords) = coords {
            detector.insert("coords".to_string(), serde_json::json!(coords));
        }
        if let Some(label) = label {
            detector.insert("label".to_string(), serde_json::json!(label));
        }
        detectors.push(serde_json::Value::Object(detector));
        set_meta_json_array(self, "detectors", &detectors)?;
        self.set_meta(
            "num_detectors",
            metadata_count_attr(&detectors, "detectors")?,
        );
        Ok(id)
    }

    /// Add observable metadata defined by measurement-record offsets.
    ///
    /// Standard observables live in the decoder `L<n>` id space. A label of
    /// `"L3"` therefore selects observable id 3 unless `observable_id` is
    /// provided, in which case the two must agree.
    ///
    /// # Errors
    ///
    /// Returns an error if existing observable metadata is not a JSON array, if
    /// JSON serialization fails, if the label/id conflict, or if the selected
    /// id duplicates an existing explicit observable id.
    pub fn add_observable_metadata(
        &mut self,
        records: &[i64],
        observable_id: Option<usize>,
        label: Option<&str>,
    ) -> Result<usize, String> {
        let mut observables = meta_json_array(self, "observables")?;
        let label_id = observable_id_from_label(label)?;
        if let (Some(observable_id), Some(label_id)) = (observable_id, label_id)
            && observable_id != label_id
        {
            return Err(format!(
                "observable_id={observable_id} conflicts with label id L{label_id}"
            ));
        }
        let id = observable_id
            .or(label_id)
            .map_or_else(|| next_metadata_id(&observables, "observables"), Ok)?;
        ensure_unique_metadata_id(&observables, "observables", "observable_id", id)?;

        let mut observable = serde_json::Map::new();
        observable.insert("id".to_string(), serde_json::json!(id));
        observable.insert("records".to_string(), serde_json::json!(records));
        if let Some(label) = label {
            observable.insert("label".to_string(), serde_json::json!(label));
        }
        observables.push(serde_json::Value::Object(observable));
        set_meta_json_array(self, "observables", &observables)?;
        self.set_meta(
            "num_observables",
            metadata_count_attr(&observables, "observables")?,
        );
        Ok(id)
    }

    /// Get all circuit-level attributes.
    pub fn circuit_attrs(&self) -> impl Iterator<Item = (&String, &Attribute)> {
        self.circuit_attrs.iter()
    }

    // --- Circuit manipulation ---

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
        self.annotations.clear();
        self.next_meas_record = 0;
    }

    /// Try to compile an ideal circuit plus a gate-triggered noise model into
    /// an annotated circuit containing explicit channel operations.
    ///
    /// The original gates are preserved. For each source tick, channel
    /// operations returned by `noise.channels_after(gate)` are scheduled into
    /// one or more immediately following ticks while respecting qubit
    /// conflicts. This produces a concrete inline representation useful for
    /// inspection, visualization, and simulators that consume interleaved
    /// channel operations directly.
    ///
    /// For measurements, `channels_after` is literal: returned channels are
    /// placed after the measurement operation. Physical pre-measurement noise
    /// and classical readout flips should use explicit APIs for those concepts
    /// instead of being hidden in this post-gate hook.
    ///
    /// # Errors
    ///
    /// Returns an error if the source circuit already contains channel
    /// operations. Apply either inline channels or a noise model, not both.
    pub fn try_with_noise<N: GateNoiseModel>(&self, noise: &N) -> Result<Self, String> {
        for (tick_idx, tick) in self.iter_ticks() {
            for gate in tick.iter_gate_batches() {
                if gate.is_channel() {
                    let gate_idx = gate.batch_index();
                    return Err(format!(
                        "with_noise cannot apply a noise model to a circuit that already contains channel operations (first channel at tick {tick_idx} gate {gate_idx})"
                    ));
                }
            }
        }

        let mut out = Self::new();
        out.circuit_attrs.clone_from(&self.circuit_attrs);
        out.gate_signatures.clone_from(&self.gate_signatures);
        out.annotations.clone_from(&self.annotations);
        out.next_meas_record = self.next_meas_record;

        for tick in &self.ticks {
            out.ticks.push(tick.clone());

            let mut noise_ticks = Vec::new();
            for gate in tick.iter_gate_batches() {
                for channel in noise.channels_after(gate.as_gate()) {
                    schedule_channel_gate(&mut noise_ticks, Gate::channel(channel));
                }
            }
            out.ticks.extend(noise_ticks);
        }

        out.next_tick = out.ticks.len();
        Ok(out)
    }

    /// Compile an ideal circuit plus a gate-triggered noise model into an
    /// annotated circuit containing explicit channel operations.
    ///
    /// This is the convenience form of [`try_with_noise`](Self::try_with_noise).
    ///
    /// # Panics
    ///
    /// Panics if the source circuit already contains channel operations.
    #[must_use]
    pub fn with_noise<N: GateNoiseModel>(&self, noise: &N) -> Self {
        self.try_with_noise(noise)
            .expect("with_noise requires an ideal circuit without existing channel operations")
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
            last_gate_piece: None,
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
            last_gate_piece: None,
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

    // --- Gate signature validation ---

    /// Import gate signatures in bulk (e.g., from a `GateRegistry`).
    pub fn import_signatures(&mut self, sigs: &BTreeMap<String, GateSignature>) {
        self.gate_signatures
            .extend(sigs.iter().map(|(name, sig)| (name.clone(), sig.clone())));
    }

    /// Get read access to the gate signatures.
    #[must_use]
    pub fn gate_signatures(&self) -> &BTreeMap<String, GateSignature> {
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

    // --- Iteration helpers ---

    /// Iterate over full-fidelity gate batches in the circuit.
    ///
    /// This is the preferred API for consumers that execute or analyze batched
    /// commands. Each yielded [`Gate`] may represent multiple individual gates
    /// on disjoint qubits, and carries the full gate payload.
    pub fn iter_gate_batches(&self) -> impl Iterator<Item = GateBatchRef<'_>> {
        self.ticks.iter().flat_map(Tick::iter_gate_batches)
    }

    /// Returns true if any tick contains an explicit channel operation.
    #[must_use]
    pub fn has_channel_operations(&self) -> bool {
        self.iter_gate_batches().any(|gate| gate.is_channel())
    }

    /// Iterate over full-fidelity gate batches with their tick index.
    pub fn iter_gate_batches_with_tick(&self) -> impl Iterator<Item = (usize, GateBatchRef<'_>)> {
        self.ticks
            .iter()
            .enumerate()
            .flat_map(|(tick_idx, tick)| tick.iter_gate_batches().map(move |gate| (tick_idx, gate)))
    }

    /// Iterate over individual gates expanded from stored batches.
    pub fn iter_gate_instances(&self) -> impl Iterator<Item = GateInstanceRef<'_>> {
        self.ticks.iter().flat_map(Tick::iter_gate_instances)
    }

    /// Iterate over individual gates with their tick index.
    pub fn iter_gate_instances_with_tick(
        &self,
    ) -> impl Iterator<Item = (usize, GateInstanceRef<'_>)> {
        self.ticks.iter().enumerate().flat_map(|(tick_idx, tick)| {
            tick.iter_gate_instances().map(move |gate| (tick_idx, gate))
        })
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
    /// circuit.tick().cx(&[(0, 1)]);
    /// circuit.tick().cx(&[(1, 2)]);
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
    /// assert_eq!(h_gates.len(), 1);  // One H batch with 3 qubits
    /// ```
    pub fn iter_gates_by_type(
        &self,
        gate_type: GateType,
    ) -> impl Iterator<Item = GateBatchRef<'_>> {
        self.iter_gate_batches()
            .filter(move |g| g.gate_type == gate_type)
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
        self.iter_gate_batches()
            .flat_map(|gate| gate.as_gate().qubits.iter().copied())
            .collect()
    }

    /// Count gates by type across the entire circuit.
    ///
    /// Returns a map from `GateType` to gate count. Batched commands
    /// such as `cx(&[(0, 1), (2, 3)])` count as two CX gates even though they
    /// are stored as one [`Gate`] carrying two disjoint pairs.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    /// use pecos_core::gate_type::GateType;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0, 1, 2, 3]);
    /// circuit.tick().cx(&[(0, 1), (2, 3)]);
    ///
    /// let counts = circuit.gate_counts_by_type();
    /// assert_eq!(counts.get(&GateType::H), Some(&4));  // 4 H gates
    /// assert_eq!(counts.get(&GateType::CX), Some(&2)); // 2 CX gates
    /// ```
    #[must_use]
    pub fn gate_counts_by_type(&self) -> BTreeMap<GateType, usize> {
        let mut counts = BTreeMap::new();
        for gate in self.iter_gate_batches() {
            *counts.entry(gate.gate_type).or_insert(0) += gate.num_gates();
        }
        counts
    }
    // ==================== Annotations ====================

    /// Annotate a detector: measurements whose XOR should be deterministic.
    pub fn detector(&mut self, measurements: &[TickMeasRef]) -> usize {
        let meas_nodes: Vec<usize> = measurements.iter().map(|m| m.record_idx).collect();
        let pauli = pecos_core::PauliString::zs(
            &measurements
                .iter()
                .map(|m| m.qubit.index())
                .collect::<Vec<_>>(),
        );
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
    pub fn detector_labeled(&mut self, label: &str, measurements: &[TickMeasRef]) -> usize {
        let idx = self.detector(measurements);
        self.annotations[idx].label = Some(label.to_string());
        idx
    }

    /// Annotate a logical observable.
    pub fn observable(&mut self, measurements: &[TickMeasRef]) -> usize {
        let meas_nodes: Vec<usize> = measurements.iter().map(|m| m.record_idx).collect();
        let pauli = pecos_core::PauliString::zs(
            &measurements
                .iter()
                .map(|m| m.qubit.index())
                .collect::<Vec<_>>(),
        );
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
    pub fn observable_labeled(&mut self, label: &str, measurements: &[TickMeasRef]) -> usize {
        let idx = self.observable(measurements);
        self.annotations[idx].label = Some(label.to_string());
        idx
    }

    /// Place a tracked-Pauli annotation.
    pub fn tracked_pauli(&mut self, mut pauli: pecos_core::PauliString) -> usize {
        pauli.set_phase(pecos_core::QuarterPhase::PlusOne);
        let idx = self.annotations.len();
        self.annotations.push(PauliAnnotation {
            pauli,
            kind: AnnotationKind::TrackedPauli,
            label: None,
        });
        idx
    }

    /// Place a labeled tracked-Pauli annotation.
    pub fn tracked_pauli_labeled(&mut self, label: &str, pauli: pecos_core::PauliString) -> usize {
        let idx = self.tracked_pauli(pauli);
        self.annotations[idx].label = Some(label.to_string());
        idx
    }

    /// Get all annotations.
    #[must_use]
    pub fn annotations(&self) -> &[PauliAnnotation] {
        &self.annotations
    }

    // ==================== Idle ====================

    /// Insert identity gates for qubits not operated on during each tick.
    ///
    /// For each tick, finds qubits that are in the circuit's qubit set but
    /// not actively operated on, and inserts an identity (I) gate. These
    /// gates receive `p1` noise from the noise model, matching Stim's
    /// convention of `DEPOLARIZE1` on idle qubits between ticks.
    ///
    /// This is separate from `GateType::Idle` which represents explicit
    /// wait operations with duration-dependent `p_idle` noise.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// circuit.tick().h(&[0]);
    /// circuit.tick().cx(&[(0, 1)]);
    /// circuit.tick().mz(&[0, 1]);
    ///
    /// circuit.fill_idle_gates();
    /// ```
    /// Insert Idle gates after each two-qubit gate on both of its qubits.
    ///
    /// Delegates to `InsertIdleAfterTwoQubitGates` pass. See [`crate::pass`].
    pub fn insert_idle_after_two_qubit_gates(&mut self, duration: f64) {
        use crate::pass::{CircuitPass, InsertIdleAfterTwoQubitGates};
        InsertIdleAfterTwoQubitGates(duration).apply_tick(self);
    }

    pub fn fill_idle_gates(&mut self) {
        let all_qubits = self.all_qubits();
        if all_qubits.is_empty() {
            return;
        }

        for tick in &mut self.ticks {
            let active = tick.active_qubits();
            for &q in &all_qubits {
                if !active.contains(&q) {
                    // Duration 1 = one tick of idling
                    let _ = tick.try_add_gate(Gate::idle(1.0, vec![q]));
                }
            }
        }
    }

    /// Compact ticks by merging gates into earlier ticks when possible.
    ///
    /// ASAP scheduling: walk ticks in order, try to merge each tick's gates
    /// into the latest tick where all qubits are free. Produces the minimum
    /// number of ticks for the same gate dependency structure.
    ///
    /// This is useful after replaying a serialized trace (e.g., from QIR)
    /// where each gate gets its own tick even if they could run in parallel.
    ///
    /// Gate metadata and tick-level metadata are preserved. Tick-level
    /// metadata from merged ticks is dropped (the target tick's metadata
    /// wins).
    ///
    /// # Panics
    ///
    /// Panics if an existing gate in the circuit fails validation while being
    /// moved into its compacted tick. Circuits built through `TickCircuit`
    /// constructors already validate gates at insertion time.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// // Serialized: each gate in its own tick
    /// circuit.tick().h(&[0]);
    /// circuit.tick().h(&[1]);
    /// circuit.tick().cx(&[(0, 1)]);
    /// assert_eq!(circuit.num_ticks(), 3);
    ///
    /// circuit.compact_ticks();
    /// // H(0) and H(1) merged into one tick; CX(0,1) stays separate
    /// assert_eq!(circuit.num_ticks(), 2);
    /// ```
    pub fn compact_ticks(&mut self) {
        if self.ticks.len() <= 1 {
            return;
        }

        let old_ticks: Vec<Tick> = self.ticks.drain(..).collect();
        let mut compacted: Vec<Tick> = Vec::new();

        for tick in old_ticks {
            let mut placed = false;

            // Try to merge into the latest existing tick where all qubits are free.
            // Walk backwards to find the latest valid target (ASAP scheduling).
            for target_idx in (0..compacted.len()).rev() {
                let can_merge = tick.gate_batches.iter().all(|gate| {
                    gate.qubits
                        .iter()
                        .all(|q| !compacted[target_idx].uses_qubit(*q))
                });

                if can_merge {
                    // Check that no tick between target+1..end uses any of these qubits
                    // (would violate ordering).
                    let all_clear = (target_idx + 1..compacted.len()).all(|between| {
                        tick.gate_batches.iter().all(|gate| {
                            gate.qubits
                                .iter()
                                .all(|q| !compacted[between].uses_qubit(*q))
                        })
                    });

                    if all_clear {
                        // Move gates and their per-gate metadata into the target tick.
                        for (gi, gate) in tick.gate_batches.iter().enumerate() {
                            if let Some(attrs) = tick.normalized_gate_attrs(gi) {
                                let new_idx = compacted[target_idx]
                                    .try_add_gate_preserving_command(gate.clone())
                                    .unwrap_or_else(|err| panic!("{err}"));
                                compacted[target_idx].set_gate_attrs(new_idx, attrs.clone());
                                compacted[target_idx].merge_compatible_gate_at(new_idx);
                            } else {
                                compacted[target_idx].add_gate(gate.clone());
                            }
                        }
                        placed = true;
                        break;
                    }
                }
            }

            if !placed {
                compacted.push(tick);
            }
        }

        self.ticks = compacted;
        self.next_tick = self.ticks.len();
    }
}

// --- TickHandle - handle for adding gates to a specific tick ---

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
    /// Panics if the gate payload is invalid or any qubit in the gate is
    /// already used by another gate in this tick.
    /// Use `try_add_gate` for fallible gate addition.
    fn add_gate(&mut self, gate: Gate) -> &mut Self {
        match self.circuit.ticks[self.tick_idx].try_add_gate_piece(gate) {
            Ok(piece) => {
                self.last_gate_idx = Some(piece.gate_idx);
                self.last_gate_piece = Some(piece);
                self
            }
            Err(mut err) => {
                err.set_tick_idx(self.tick_idx);
                panic!("{}", err);
            }
        }
    }

    /// Try to add a gate to this tick.
    ///
    /// # Errors
    ///
    /// Returns [`TickGateError::InvalidGate`] if the gate payload is invalid, or
    /// [`TickGateError::QubitConflict`] if any qubit in the gate is already used
    /// by another gate in this tick.
    pub fn try_add_gate(&mut self, gate: Gate) -> Result<&mut Self, TickGateError> {
        match self.circuit.ticks[self.tick_idx].try_add_gate_piece(gate) {
            Ok(piece) => {
                self.last_gate_idx = Some(piece.gate_idx);
                self.last_gate_piece = Some(piece);
                Ok(self)
            }
            Err(mut err) => {
                err.set_tick_idx(self.tick_idx);
                Err(err)
            }
        }
    }

    /// Add a gate and return the gate index.
    ///
    /// # Panics
    ///
    /// Panics if the gate payload is invalid or any qubit in the gate is
    /// already used by another gate in this tick.
    fn add_gate_get_piece(&mut self, gate: Gate) -> GateBatchPiece {
        match self.circuit.ticks[self.tick_idx].try_add_gate_piece(gate) {
            Ok(piece) => {
                self.last_gate_idx = Some(piece.gate_idx);
                self.last_gate_piece = Some(piece);
                piece
            }
            Err(mut err) => {
                err.set_tick_idx(self.tick_idx);
                panic!("{}", err);
            }
        }
    }

    fn add_gate_get_idx(&mut self, gate: Gate) -> usize {
        self.add_gate_get_piece(gate).gate_idx
    }

    /// Set metadata on the last added gate.
    ///
    /// If no gate has been added yet, sets tick-level metadata instead.
    pub fn meta(&mut self, key: &str, value: impl Into<Attribute>) -> &mut Self {
        if let Some(piece) = self.last_gate_piece {
            let piece =
                self.circuit.ticks[self.tick_idx].set_gate_attr_for_piece(piece, key, value.into());
            self.last_gate_idx = Some(piece.gate_idx);
            self.last_gate_piece = Some(piece);
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
        if let Some(piece) = self.last_gate_piece {
            let piece = self.circuit.ticks[self.tick_idx].set_gate_attrs_for_piece(piece, attrs);
            self.last_gate_idx = Some(piece.gate_idx);
            self.last_gate_piece = Some(piece);
        } else {
            // No gate yet - set tick-level metadata
            self.circuit.ticks[self.tick_idx].set_attrs(attrs);
        }
        self
    }

    // --- Single-qubit gates ---

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

    /// Place a typed channel operation in this tick.
    ///
    /// This is for annotated/noisy circuits. It does not use custom-gate
    /// metadata; the channel payload is stored directly on the gate.
    pub fn channel(&mut self, channel: ChannelExpr) -> &mut Self {
        self.add_gate(Gate::channel(channel))
    }

    // --- Two-qubit gates ---

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

    // --- Three-qubit gates ---

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

    // --- State preparation and measurement ---

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
        let gate_piece = self.add_gate_get_piece(Gate::pz(qubits));
        self.last_gate_idx = None;
        self.last_gate_piece = None;
        TickPrepHandle {
            circuit: self.circuit,
            tick_idx: self.tick_idx,
            gate_piece,
        }
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
    pub fn mz(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Vec<TickMeasRef> {
        let mut gate = Gate::mz(qubits);
        let mut refs = Vec::with_capacity(qubits.len());
        for &q in qubits {
            let tick_idx = self.tick_idx;
            let record_idx = self.circuit.next_meas_record;
            self.circuit.next_meas_record += 1;
            let mr = MeasId(record_idx);
            gate.meas_ids.push(mr);
            refs.push(TickMeasRef {
                tick: tick_idx,
                gate_idx: 0, // placeholder, updated below
                qubit: q.into(),
                record_idx,
                meas_id: mr,
            });
        }
        let gate_idx = self.add_gate_get_idx(gate);
        self.last_gate_idx = None;
        self.last_gate_piece = None;
        // Fix up gate_idx in refs (needed because we had to build gate before adding)
        refs.into_iter()
            .map(|mut r| {
                r.gate_idx = gate_idx;
                r
            })
            .collect()
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
    /// circuit.tick().mz_free(&[0, 1]);
    /// ```
    pub fn mz_free(mut self, qubits: &[impl Into<QubitId> + Copy]) -> Vec<TickMeasRef> {
        let mut gate = Gate::mz_free(qubits);
        let mut refs = Vec::with_capacity(qubits.len());
        for &q in qubits {
            let tick_idx = self.tick_idx;
            let record_idx = self.circuit.next_meas_record;
            self.circuit.next_meas_record += 1;
            let mr = MeasId(record_idx);
            gate.meas_ids.push(mr);
            refs.push(TickMeasRef {
                tick: tick_idx,
                gate_idx: 0,
                qubit: q.into(),
                record_idx,
                meas_id: mr,
            });
        }
        let gate_idx = self.add_gate_get_idx(gate);
        self.last_gate_idx = None;
        self.last_gate_piece = None;
        refs.into_iter()
            .map(|mut r| {
                r.gate_idx = gate_idx;
                r
            })
            .collect()
    }

    // --- Resource management ---

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

    // --- Timing ---

    /// Insert an idle (wait) operation for one or more qubits.
    ///
    /// Duration is in abstract time units. The interpretation (nanoseconds,
    /// clock cycles, etc.) is defined by your noise model or timing configuration.
    ///
    /// # Examples
    ///
    /// ```
    /// use pecos_quantum::TickCircuit;
    ///
    /// let mut circuit = TickCircuit::new();
    /// // Idle for 100 time units
    /// circuit.tick().idle(100, &[0, 1, 2]);
    /// ```
    pub fn idle(
        &mut self,
        duration: impl Into<TimeUnits>,
        qubits: &[impl Into<QubitId> + Copy],
    ) -> &mut Self {
        let units: TimeUnits = duration.into();
        self.add_gate(Gate::idle(
            units.as_f64(),
            qubits.iter().map(|&q| q.into()).collect::<GateQubits>(),
        ))
    }

    // --- Custom gates with signature validation ---

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

        match self.circuit.ticks[self.tick_idx].try_add_gate_piece(gate) {
            Ok(piece) => {
                // Auto-store _symbol metadata
                let piece = self.circuit.ticks[self.tick_idx].set_gate_attr_for_piece(
                    piece,
                    "_symbol",
                    Attribute::String(name.to_string()),
                );
                self.last_gate_idx = Some(piece.gate_idx);
                self.last_gate_piece = Some(piece);
                Ok(self)
            }
            Err(mut err) => {
                err.set_tick_idx(self.tick_idx);
                Err(err.into())
            }
        }
    }
}

// --- Conversions between TickCircuit and DagCircuit ---

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
        let mut dag_node_to_record_indices: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
        let mut next_meas_record = 0usize;

        for layer in dag.layers() {
            let mut layer_ticks = vec![Tick::new()];

            for node_id in layer {
                if let Some(gate) = dag.gate(node_id) {
                    let mut gate = gate.clone();
                    if matches!(
                        gate.gate_type,
                        GateType::MZ | GateType::MeasureFree | GateType::MeasureLeaked
                    ) {
                        if gate.meas_ids.is_empty() {
                            let mut records = Vec::with_capacity(gate.qubits.len());
                            for _ in &gate.qubits {
                                let record_idx = next_meas_record;
                                next_meas_record += 1;
                                gate.meas_ids.push(MeasId(record_idx));
                                records.push(record_idx);
                            }
                            dag_node_to_record_indices.insert(node_id, records);
                        } else {
                            let records: Vec<usize> =
                                gate.meas_ids.iter().map(|meas_id| meas_id.0).collect();
                            if let Some(next) = records.iter().max().map(|record| record + 1) {
                                next_meas_record = next_meas_record.max(next);
                            }
                            dag_node_to_record_indices.insert(node_id, records);
                        }
                    }

                    let target_idx = layer_ticks
                        .iter()
                        .position(|tick| tick.find_conflicts(&gate.qubits).is_empty())
                        .unwrap_or_else(|| {
                            layer_ticks.push(Tick::new());
                            layer_ticks.len() - 1
                        });
                    let tick = &mut layer_ticks[target_idx];
                    // Copy gate attributes
                    if let Some(attrs) = dag.gate_attrs(node_id) {
                        let gate_idx = tick
                            .try_add_gate_preserving_command(gate)
                            .unwrap_or_else(|err| panic!("{err}"));
                        tick.set_gate_attrs(gate_idx, attrs.clone());
                        tick.merge_compatible_gate_at(gate_idx);
                    } else {
                        tick.add_gate(gate);
                    }
                }
            }

            tc.ticks.extend(layer_ticks);
        }
        tc.next_tick = tc.ticks.len();
        tc.next_meas_record = next_meas_record;

        // Copy circuit-level attributes, restoring tick-level attrs from prefixed keys
        let tick_attr_prefix = "tick[";
        for (key, value) in dag.attrs() {
            if key.starts_with(tick_attr_prefix) {
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

        // Transfer annotations, remapping DAG measurement nodes to TickCircuit
        // measurement record indices. Tracked Paulis have no measurement
        // readout and keep their Pauli role unchanged.
        tc.annotations = dag
            .annotations()
            .iter()
            .map(|ann| {
                let kind = match &ann.kind {
                    AnnotationKind::Detector {
                        measurement_nodes,
                        coords,
                    } => AnnotationKind::Detector {
                        measurement_nodes: remap_dag_measurement_nodes(
                            &dag_node_to_record_indices,
                            measurement_nodes,
                        ),
                        coords: coords.clone(),
                    },
                    AnnotationKind::Observable { measurement_nodes } => {
                        AnnotationKind::Observable {
                            measurement_nodes: remap_dag_measurement_nodes(
                                &dag_node_to_record_indices,
                                measurement_nodes,
                            ),
                        }
                    }
                    AnnotationKind::TrackedPauli => AnnotationKind::TrackedPauli,
                };
                PauliAnnotation {
                    pauli: ann.pauli.clone(),
                    kind,
                    label: ann.label.clone(),
                }
            })
            .collect();

        tc
    }
}

fn remap_dag_measurement_nodes(
    dag_node_to_record_indices: &BTreeMap<usize, Vec<usize>>,
    measurement_nodes: &[usize],
) -> Vec<usize> {
    measurement_nodes
        .iter()
        .flat_map(|node| {
            dag_node_to_record_indices
                .get(node)
                .unwrap_or_else(|| panic!("annotation references non-measurement DAG node {node}"))
                .iter()
                .copied()
        })
        .collect()
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

        // Map measurement_record_index -> dag node for annotation transfer
        let mut meas_record_to_node: BTreeMap<usize, usize> = BTreeMap::new();
        let mut meas_record_idx = 0usize;

        for (tick_idx, tick) in tc.iter_ticks() {
            for batch in tick.iter_gate_batches() {
                // DagCircuit stores individual gate applications. Use the
                // TickCircuit instance view so qubit/meas-id slicing has one
                // implementation. Zero-gate metadata batches do not have gate
                // instances, so keep their stored command as a single DAG node.
                let split_gates: Vec<Gate> = if batch.gate_count() == 0 {
                    vec![batch.as_gate().clone()]
                } else {
                    batch
                        .iter_gate_instances()
                        .map(GateInstanceRef::to_gate)
                        .collect()
                };

                let mut split_nodes = Vec::with_capacity(split_gates.len());
                for split_gate in &split_gates {
                    let node = dag.add_gate(split_gate.clone());
                    split_nodes.push(node);

                    // For MZ gates, map each qubit's record to this node
                    if split_gate.gate_type == GateType::MZ {
                        for _q in &split_gate.qubits {
                            meas_record_to_node.insert(meas_record_idx, node);
                            meas_record_idx += 1;
                        }
                    }

                    // Connect wires from previous gates on the same qubits
                    for qubit in &split_gate.qubits {
                        if let Some(&prev_node) = last_node.get(qubit) {
                            let _ = dag.connect(prev_node, node, *qubit);
                        }
                        last_node.insert(*qubit, node);
                    }
                }

                // Copy batch-level gate attributes to every split gate.
                for (key, value) in batch.attrs() {
                    for &node in &split_nodes {
                        dag.set_gate_attr(node, key, value.clone());
                    }
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

        // Transfer annotations, remapping measurement record indices to DAG node indices
        for ann in &tc.annotations {
            let remapped_kind = match &ann.kind {
                AnnotationKind::Detector {
                    measurement_nodes,
                    coords,
                } => {
                    let dag_nodes: Vec<usize> = measurement_nodes
                        .iter()
                        .filter_map(|&rec| meas_record_to_node.get(&rec).copied())
                        .collect();
                    AnnotationKind::Detector {
                        measurement_nodes: dag_nodes,
                        coords: coords.clone(),
                    }
                }
                AnnotationKind::Observable { measurement_nodes } => {
                    let dag_nodes: Vec<usize> = measurement_nodes
                        .iter()
                        .filter_map(|&rec| meas_record_to_node.get(&rec).copied())
                        .collect();
                    AnnotationKind::Observable {
                        measurement_nodes: dag_nodes,
                    }
                }
                AnnotationKind::TrackedPauli => AnnotationKind::TrackedPauli,
            };
            dag.add_annotation(PauliAnnotation {
                pauli: ann.pauli.clone(),
                kind: remapped_kind,
                label: ann.label.clone(),
            });
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
        assert_eq!(tc.gate_count(), 6); // 2 preps, 1 H, 1 CX, 2 measurements

        // Check tick contents
        assert_eq!(tc.get_tick(0).unwrap().len(), 1); // One bulk prep gate
        assert_eq!(tc.get_tick(1).unwrap().len(), 1); // One H
        assert_eq!(tc.get_tick(2).unwrap().len(), 1); // One CX
        assert_eq!(tc.get_tick(3).unwrap().len(), 1); // One bulk measurement
    }

    #[test]
    fn test_tick_construction_merges_compatible_gate_batches() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]).h(&[1]).cx(&[(2, 3)]).cx(&[(4, 5)]);

        let tick = tc.get_tick(0).unwrap();
        assert_eq!(tick.len(), 2); // one H batch, one CX batch
        assert_eq!(tick.gate_count(), 4);
        assert_eq!(tick.gate_batch_count(), 2);
        assert_eq!(tc.gate_count(), 4);
        assert_eq!(tc.gate_batch_count(), 2);

        assert_eq!(tick.gate_batches()[0].gate_type, GateType::H);
        assert_eq!(
            tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(0), QubitId::from(1)]
        );
        assert_eq!(tick.gate_batches()[1].gate_type, GateType::CX);
        assert_eq!(
            tick.gate_batches()[1].qubits.as_slice(),
            &[
                QubitId::from(2),
                QubitId::from(3),
                QubitId::from(4),
                QubitId::from(5)
            ]
        );
    }

    #[test]
    fn test_tick_replace_gate_batch_updates_stored_views() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]).h(&[1]);

        let tick = tc.get_tick_mut(0).unwrap();
        tick.replace_gate_batch(0, Gate::x(&[0, 1]))
            .expect("same support replacement should be valid");

        assert_eq!(tick.len(), 1);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batches()[0].gate_type, GateType::X);
        assert_eq!(
            tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(0), QubitId::from(1)]
        );
    }

    #[test]
    fn test_tick_replace_gate_batch_preserves_aligned_attrs() {
        let mut tc = TickCircuit::new();
        tc.tick()
            .h(&[0, 1])
            .meta("calibration", Attribute::String("old".into()));

        let tick = tc.get_tick_mut(0).unwrap();
        tick.replace_gate_batch(0, Gate::x(&[0, 1]))
            .expect("same support replacement should be valid");

        assert_eq!(tick.len(), 1);
        assert_eq!(tick.gate_batches()[0].gate_type, GateType::X);
        assert_eq!(
            tick.get_gate_attr(0, "calibration"),
            Some(&Attribute::String("old".into()))
        );
    }

    #[test]
    fn test_tick_replace_gate_batch_rejects_overlapping_qubits() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]).x(&[1]);

        let tick = tc.get_tick_mut(0).unwrap();
        let err = tick
            .replace_gate_batch(0, Gate::z(&[1]))
            .expect_err("replacement overlaps the X command on q1");

        assert!(matches!(err, TickGateError::QubitConflict(_)));
        assert_eq!(tick.gate_batches()[0].gate_type, GateType::H);
        assert_eq!(tick.gate_batches()[1].gate_type, GateType::X);
    }

    #[test]
    fn test_tick_update_gate_batch_keeps_measurement_ids_in_sync() {
        let mut tc = TickCircuit::new();
        tc.tick().mz(&[0, 1]);

        let tick = tc.get_tick_mut(0).unwrap();
        tick.update_gate_batch(0, |gate| {
            gate.meas_ids[0] = MeasId(10);
            gate.meas_ids[1] = MeasId(11);
        })
        .expect("measurement id update should be valid");

        assert_eq!(
            tick.gate_batches()[0].meas_ids.as_slice(),
            &[MeasId(10), MeasId(11)]
        );
    }

    #[test]
    fn test_tick_construction_batches_only_same_metadata() {
        let mut same = TickCircuit::new();
        {
            let mut tick = same.tick();
            tick.h(&[0])
                .meta("calibration", Attribute::String("a".into()));
            tick.h(&[1])
                .meta("calibration", Attribute::String("a".into()));
        }

        let tick = same.get_tick(0).unwrap();
        assert_eq!(tick.len(), 1);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batch_count(), 1);
        assert_eq!(
            tick.get_gate_attr(0, "calibration"),
            Some(&Attribute::String("a".into()))
        );

        let mut different = TickCircuit::new();
        {
            let mut tick = different.tick();
            tick.h(&[0])
                .meta("calibration", Attribute::String("a".into()));
            tick.h(&[1])
                .meta("calibration", Attribute::String("b".into()));
        }

        let tick = different.get_tick(0).unwrap();
        assert_eq!(tick.len(), 2);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batch_count(), 2);
    }

    #[test]
    fn test_tick_construction_metadata_applies_to_last_gate_before_batching() {
        let mut tc = TickCircuit::new();
        {
            let mut tick = tc.tick();
            tick.h(&[0])
                .h(&[1])
                .meta("calibration", Attribute::String("second".into()));
        }

        let tick = tc.get_tick(0).unwrap();
        assert_eq!(tick.len(), 2);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batch_count(), 2);
        assert_eq!(
            tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(0)]
        );
        assert_eq!(
            tick.gate_batches()[1].qubits.as_slice(),
            &[QubitId::from(1)]
        );
        assert_eq!(tick.get_gate_attr(0, "calibration"), None);
        assert_eq!(
            tick.get_gate_attr(1, "calibration"),
            Some(&Attribute::String("second".into()))
        );
    }

    #[test]
    fn test_tick_construction_multiple_meta_calls_batch_after_completion() {
        let mut tc = TickCircuit::new();
        {
            let mut tick = tc.tick();
            tick.h(&[0])
                .meta("calibration", Attribute::String("a".into()))
                .meta("role", Attribute::String("drive".into()));
            tick.h(&[1])
                .meta("calibration", Attribute::String("a".into()))
                .meta("role", Attribute::String("drive".into()));
        }

        let tick = tc.get_tick(0).unwrap();
        assert_eq!(tick.len(), 1);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batch_count(), 1);
        assert_eq!(
            tick.get_gate_attr(0, "calibration"),
            Some(&Attribute::String("a".into()))
        );
        assert_eq!(
            tick.get_gate_attr(0, "role"),
            Some(&Attribute::String("drive".into()))
        );
    }

    #[test]
    fn test_prep_metadata_applies_before_batching() {
        let mut tc = TickCircuit::new();
        tc.reserve_ticks(1);
        tc.tick_at(0).pz(&[0]);
        tc.tick_at(0)
            .pz(&[1])
            .meta("calibration", Attribute::String("second".into()));

        let tick = tc.get_tick(0).unwrap();
        assert_eq!(tick.len(), 2);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batch_count(), 2);
        assert_eq!(tick.get_gate_attr(0, "calibration"), None);
        assert_eq!(
            tick.get_gate_attr(1, "calibration"),
            Some(&Attribute::String("second".into()))
        );
    }

    #[test]
    fn test_tick_construction_preserves_measurement_ids_when_batching() {
        let mut tc = TickCircuit::new();
        tc.reserve_ticks(1);

        let refs0 = tc.tick_at(0).mz(&[0]);
        let refs1 = tc.tick_at(0).mz(&[1]);

        let tick = tc.get_tick(0).unwrap();
        assert_eq!(tick.len(), 1);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batch_count(), 1);
        assert_eq!(refs0[0].gate_idx, 0);
        assert_eq!(refs1[0].gate_idx, 0);
        assert_eq!(refs0[0].meas_id, MeasId(0));
        assert_eq!(refs1[0].meas_id, MeasId(1));
        assert_eq!(
            tick.gate_batches()[0].meas_ids.as_slice(),
            &[MeasId(0), MeasId(1)]
        );
    }

    #[test]
    fn test_measurement_batching_respects_gate_metadata() {
        let mut same = TickCircuit::new();
        same.reserve_ticks(1);
        let refs0 = same.tick_at(0).mz(&[0]);
        same.get_tick_mut(0).unwrap().set_gate_attr(
            refs0[0].gate_idx,
            "basis",
            Attribute::String("Z".into()),
        );
        let refs1 = same.tick_at(0).mz(&[1]);
        let tick = same.get_tick_mut(0).unwrap();
        tick.set_gate_attr(refs1[0].gate_idx, "basis", Attribute::String("Z".into()));
        tick.merge_compatible_gate_at(refs1[0].gate_idx);

        let tick = same.get_tick(0).unwrap();
        assert_eq!(tick.len(), 1);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batch_count(), 1);
        assert_eq!(
            tick.gate_batches()[0].meas_ids.as_slice(),
            &[MeasId(0), MeasId(1)]
        );
        assert_eq!(
            tick.get_gate_attr(0, "basis"),
            Some(&Attribute::String("Z".into()))
        );

        let mut different = TickCircuit::new();
        different.reserve_ticks(1);
        let refs0 = different.tick_at(0).mz(&[0]);
        different.get_tick_mut(0).unwrap().set_gate_attr(
            refs0[0].gate_idx,
            "basis",
            Attribute::String("Z".into()),
        );
        let refs1 = different.tick_at(0).mz(&[1]);
        let tick = different.get_tick_mut(0).unwrap();
        tick.set_gate_attr(refs1[0].gate_idx, "basis", Attribute::String("X".into()));
        tick.merge_compatible_gate_at(refs1[0].gate_idx);

        let tick = different.get_tick(0).unwrap();
        assert_eq!(tick.len(), 2);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batch_count(), 2);
        assert_eq!(tick.gate_batches()[0].meas_ids.as_slice(), &[MeasId(0)]);
        assert_eq!(tick.gate_batches()[1].meas_ids.as_slice(), &[MeasId(1)]);
        assert_eq!(
            tick.get_gate_attr(0, "basis"),
            Some(&Attribute::String("Z".into()))
        );
        assert_eq!(
            tick.get_gate_attr(1, "basis"),
            Some(&Attribute::String("X".into()))
        );
    }

    #[test]
    fn test_tick_construction_keeps_different_parameters_in_separate_batches() {
        let mut tc = TickCircuit::new();
        tc.tick()
            .rz(Angle64::from_turns(0.25), &[0])
            .rz(Angle64::from_turns(0.5), &[1]);

        let tick = tc.get_tick(0).unwrap();
        assert_eq!(tick.len(), 2);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batch_count(), 2);
    }

    #[test]
    fn test_parameterized_gate_batching_counts_and_round_trip() {
        let mut tc1 = TickCircuit::new();
        tc1.tick()
            .rz(Angle64::from_turns(0.25), &[0])
            .rz(Angle64::from_turns(0.25), &[1])
            .rz(Angle64::from_turns(0.5), &[2]);

        let tick = tc1.get_tick(0).unwrap();
        assert_eq!(tick.len(), 2);
        assert_eq!(tick.gate_count(), 3);
        assert_eq!(tick.gate_batch_count(), 2);
        assert_eq!(tc1.gate_count(), 3);
        assert_eq!(tc1.gate_batch_count(), 2);

        let dag = DagCircuit::from(&tc1);
        assert_eq!(dag.gate_count(), 3);
        assert_eq!(dag.gate_node_count(), 3);
        assert_eq!(dag.gate_type_count(GateType::RZ), 3);

        let tc2 = TickCircuit::from(&dag);
        let tick = tc2.get_tick(0).unwrap();
        assert_eq!(tc2.gate_count(), 3);
        assert_eq!(tc2.gate_batch_count(), 2);
        assert_eq!(tick.len(), 2);
        assert_eq!(tick.gate_count(), 3);
        assert_eq!(tick.gate_batch_count(), 2);
        assert_eq!(
            tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(0), QubitId::from(1)]
        );
        assert_eq!(
            tick.gate_batches()[0].angles,
            Gate::rz(Angle64::from_turns(0.25), &[0]).angles
        );
        assert_eq!(
            tick.gate_batches()[1].qubits.as_slice(),
            &[QubitId::from(2)]
        );
        assert_eq!(
            tick.gate_batches()[1].angles,
            Gate::rz(Angle64::from_turns(0.5), &[2]).angles
        );
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
        tc.tick().mz(&[0]);
        // Attach metadata to the measurement gate directly
        tc.get_tick_mut(2)
            .unwrap()
            .set_gate_attr(0, "basis", Attribute::String("Z".into()));

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
    fn test_tick_to_dag_splits_batched_standard_two_qubit_cliffords_as_pairs() {
        fn add_gate(tc: &mut TickCircuit, gate_type: GateType, pairs: &[(usize, usize)]) {
            match gate_type {
                GateType::CX => {
                    tc.tick().cx(pairs);
                }
                GateType::CY => {
                    tc.tick().cy(pairs);
                }
                GateType::CZ => {
                    tc.tick().cz(pairs);
                }
                GateType::SXX => {
                    tc.tick().sxx(pairs);
                }
                GateType::SXXdg => {
                    tc.tick().sxxdg(pairs);
                }
                GateType::SYY => {
                    tc.tick().syy(pairs);
                }
                GateType::SYYdg => {
                    tc.tick().syydg(pairs);
                }
                GateType::SZZ => {
                    tc.tick().szz(pairs);
                }
                GateType::SZZdg => {
                    tc.tick().szzdg(pairs);
                }
                GateType::SWAP => {
                    tc.tick().swap(pairs);
                }
                _ => unreachable!(),
            }
        }

        let pair_sets = [
            [(0usize, 1usize), (2usize, 3usize)],
            [(4usize, 1usize), (9usize, 2usize)],
        ];

        for gate_type in [
            GateType::CX,
            GateType::CY,
            GateType::CZ,
            GateType::SXX,
            GateType::SXXdg,
            GateType::SYY,
            GateType::SYYdg,
            GateType::SZZ,
            GateType::SZZdg,
            GateType::SWAP,
        ] {
            for pairs in pair_sets {
                let mut tc = TickCircuit::new();
                add_gate(&mut tc, gate_type, &pairs);

                let dag = DagCircuit::from(&tc);
                let gates: Vec<_> = dag.iter_gates().map(|(_, gate)| gate).collect();
                assert_eq!(gates.len(), 2, "{gate_type:?} {pairs:?}");
                assert!(
                    gates.iter().all(|gate| gate.gate_type == gate_type),
                    "{gate_type:?} {pairs:?}"
                );
                assert!(
                    gates.iter().all(|gate| gate.qubits.len() == 2),
                    "{gate_type:?} should remain pairwise in the DAG for {pairs:?}"
                );
                for (q0, q1) in pairs {
                    assert!(
                        gates.iter().any(|gate| gate
                            .qubits
                            .iter()
                            .copied()
                            .eq([QubitId(q0), QubitId(q1)])),
                        "{gate_type:?} should preserve pair ({q0}, {q1})"
                    );
                }
            }
        }
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
        assert_eq!(tick0.gate_batches().len(), 2);

        // Second tick should have CX
        let tick1 = tc.get_tick(1).unwrap();
        assert_eq!(tick1.gate_batches().len(), 1);

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
                tc1.get_tick(i).unwrap().gate_batches().len(),
                tc2.get_tick(i).unwrap().gate_batches().len()
            );
        }
    }

    #[test]
    fn test_tick_dag_round_trip_preserves_counts_and_metadata_batches() {
        let mut tc1 = TickCircuit::new();
        tc1.set_meta("name", Attribute::String("metadata-heavy".into()));
        tc1.tick()
            .meta("round", Attribute::Int(0))
            .h(&[0, 1])
            .meta("calibration", Attribute::String("h-cal".into()));
        tc1.tick()
            .meta("round", Attribute::Int(1))
            .cx(&[(0, 2), (1, 3)])
            .meta("calibration", Attribute::String("cx-cal".into()));
        let ms = tc1.tick().mz(&[0, 1]);

        assert_eq!(tc1.num_ticks(), 3);
        assert_eq!(tc1.gate_count(), 6);
        assert_eq!(tc1.gate_batch_count(), 3);
        assert_eq!(ms[0].meas_id, MeasId(0));
        assert_eq!(ms[1].meas_id, MeasId(1));

        let dag = DagCircuit::from(&tc1);
        assert_eq!(dag.gate_count(), 6);
        assert_eq!(dag.gate_node_count(), 6);
        assert_eq!(dag.gate_type_count(GateType::H), 2);
        assert_eq!(dag.gate_type_count(GateType::CX), 2);
        assert_eq!(dag.gate_type_count(GateType::MZ), 2);

        for (node, gate) in dag.iter_gates() {
            match gate.gate_type {
                GateType::H => assert_eq!(
                    dag.get_gate_attr(node, "calibration"),
                    Some(&Attribute::String("h-cal".into()))
                ),
                GateType::CX => assert_eq!(
                    dag.get_gate_attr(node, "calibration"),
                    Some(&Attribute::String("cx-cal".into()))
                ),
                _ => {}
            }
        }

        let tc2 = TickCircuit::from(&dag);
        assert_eq!(tc2.num_ticks(), 3);
        assert_eq!(tc2.gate_count(), 6);
        assert_eq!(tc2.gate_batch_count(), 3);
        assert_eq!(
            tc2.get_meta("name"),
            Some(&Attribute::String("metadata-heavy".into()))
        );
        assert_eq!(
            tc2.get_tick(0).unwrap().get_attr("round"),
            Some(&Attribute::Int(0))
        );
        assert_eq!(
            tc2.get_tick(1).unwrap().get_attr("round"),
            Some(&Attribute::Int(1))
        );

        let tick0 = tc2.get_tick(0).unwrap();
        assert_eq!(tick0.len(), 1);
        assert_eq!(tick0.gate_count(), 2);
        assert_eq!(tick0.gate_batch_count(), 1);
        assert_eq!(tick0.gate_batches()[0].gate_type, GateType::H);
        assert_eq!(
            tick0.get_gate_attr(0, "calibration"),
            Some(&Attribute::String("h-cal".into()))
        );

        let tick1 = tc2.get_tick(1).unwrap();
        assert_eq!(tick1.len(), 1);
        assert_eq!(tick1.gate_count(), 2);
        assert_eq!(tick1.gate_batch_count(), 1);
        assert_eq!(tick1.gate_batches()[0].gate_type, GateType::CX);
        assert_eq!(
            tick1.get_gate_attr(0, "calibration"),
            Some(&Attribute::String("cx-cal".into()))
        );

        let tick2 = tc2.get_tick(2).unwrap();
        assert_eq!(tick2.len(), 1);
        assert_eq!(tick2.gate_count(), 2);
        assert_eq!(tick2.gate_batch_count(), 1);
        assert_eq!(tick2.gate_batches()[0].gate_type, GateType::MZ);
        assert_eq!(
            tick2.gate_batches()[0].meas_ids.as_slice(),
            &[MeasId(0), MeasId(1)]
        );
    }

    #[test]
    fn test_dag_to_tick_preserves_distinct_metadata_batches() {
        let mut dag = DagCircuit::new();
        let h0 = dag.add_gate(Gate::h(&[0]));
        dag.set_gate_attr(h0, "calibration", Attribute::String("a".into()));
        let h1 = dag.add_gate(Gate::h(&[1]));
        dag.set_gate_attr(h1, "calibration", Attribute::String("b".into()));
        let h2 = dag.add_gate(Gate::h(&[2]));
        dag.set_gate_attr(h2, "calibration", Attribute::String("a".into()));

        let tc = TickCircuit::from(&dag);
        assert_eq!(tc.gate_count(), 3);
        assert_eq!(tc.gate_batch_count(), 2);

        let tick = tc.get_tick(0).unwrap();
        assert_eq!(tick.len(), 2);
        assert_eq!(tick.gate_count(), 3);
        assert_eq!(tick.gate_batch_count(), 2);
        assert_eq!(
            tick.get_gate_attr(0, "calibration"),
            Some(&Attribute::String("a".into()))
        );
        assert_eq!(
            tick.get_gate_attr(1, "calibration"),
            Some(&Attribute::String("b".into()))
        );
        assert_eq!(
            tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(0), QubitId::from(2)]
        );
        assert_eq!(
            tick.gate_batches()[1].qubits.as_slice(),
            &[QubitId::from(1)]
        );
    }

    #[test]
    fn test_batched_measurement_metadata_round_trip() {
        let mut tc1 = TickCircuit::new();
        let ms = tc1.tick().mz(&[0, 1]);
        tc1.get_tick_mut(0).unwrap().set_gate_attr(
            ms[0].gate_idx,
            "basis",
            Attribute::String("Z".into()),
        );

        let dag = DagCircuit::from(&tc1);
        assert_eq!(dag.gate_count(), 2);
        assert_eq!(dag.gate_node_count(), 2);
        for (node, gate) in dag.iter_gates() {
            assert_eq!(gate.gate_type, GateType::MZ);
            assert_eq!(
                dag.get_gate_attr(node, "basis"),
                Some(&Attribute::String("Z".into()))
            );
        }

        let tc2 = TickCircuit::from(&dag);
        assert_eq!(tc2.gate_count(), 2);
        assert_eq!(tc2.gate_batch_count(), 1);

        let tick = tc2.get_tick(0).unwrap();
        assert_eq!(tick.len(), 1);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batch_count(), 1);
        assert_eq!(tick.gate_batches()[0].gate_type, GateType::MZ);
        assert_eq!(
            tick.gate_batches()[0].meas_ids.as_slice(),
            &[MeasId(0), MeasId(1)]
        );
        assert_eq!(
            tick.get_gate_attr(0, "basis"),
            Some(&Attribute::String("Z".into()))
        );
    }

    #[test]
    fn test_channel_gate_counts_as_single_operation_through_round_trip() {
        let mut tc1 = TickCircuit::new();
        tc1.tick().channel(
            pecos_core::channel::Depolarizing(0.1, 0) & pecos_core::channel::Dephasing(0.2, 1),
        );

        let tick = tc1.get_tick(0).unwrap();
        assert_eq!(tick.len(), 1);
        assert_eq!(tick.gate_count(), 1);
        assert_eq!(tick.gate_batch_count(), 1);
        assert_eq!(tc1.gate_count(), 1);
        assert_eq!(tc1.gate_batch_count(), 1);
        assert_eq!(
            tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(0), QubitId::from(1)]
        );

        let dag = DagCircuit::from(&tc1);
        assert_eq!(dag.gate_count(), 1);
        assert_eq!(dag.gate_node_count(), 1);
        let (_, gate) = dag.iter_gates().next().unwrap();
        assert!(gate.is_channel());
        assert_eq!(
            gate.qubits.as_slice(),
            &[QubitId::from(0), QubitId::from(1)]
        );

        let tc2 = TickCircuit::from(&dag);
        assert_eq!(tc2.gate_count(), 1);
        assert_eq!(tc2.gate_batch_count(), 1);
        let gate = &tc2.get_tick(0).unwrap().gate_batches()[0];
        assert!(gate.is_channel());
        assert_eq!(
            gate.qubits.as_slice(),
            &[QubitId::from(0), QubitId::from(1)]
        );
    }

    #[test]
    fn test_batched_measurement_annotation_records_round_trip() {
        let mut tc1 = TickCircuit::new();
        let ms = tc1.tick().mz(&[0, 1]);
        tc1.detector_labeled("det01", &ms);
        tc1.observable_labeled("obs1", &[ms[1]]);

        let dag = DagCircuit::from(&tc1);
        let tc2 = TickCircuit::from(&dag);

        assert_eq!(tc2.num_measurements(), 2);
        assert_eq!(tc2.annotations().len(), 2);
        match &tc2.annotations()[0].kind {
            AnnotationKind::Detector {
                measurement_nodes, ..
            } => assert_eq!(measurement_nodes.as_slice(), &[0, 1]),
            other => panic!("expected detector annotation, got {other:?}"),
        }
        match &tc2.annotations()[1].kind {
            AnnotationKind::Observable { measurement_nodes } => {
                assert_eq!(measurement_nodes.as_slice(), &[1]);
            }
            other => panic!("expected observable annotation, got {other:?}"),
        }

        let tick = tc2.get_tick(0).unwrap();
        assert_eq!(
            tick.gate_batches()[0].meas_ids.as_slice(),
            &[MeasId(0), MeasId(1)]
        );
    }

    #[test]
    fn test_dag_batched_measurement_node_annotation_expands_to_tick_records() {
        let mut dag = DagCircuit::new();
        let node = dag.add_gate(Gate::mz(&[0, 1]));
        dag.detector_labeled("batched-detector", &[node]);

        let tc = TickCircuit::from(&dag);
        assert_eq!(tc.num_measurements(), 2);
        assert_eq!(tc.annotations().len(), 1);
        match &tc.annotations()[0].kind {
            AnnotationKind::Detector {
                measurement_nodes, ..
            } => assert_eq!(measurement_nodes.as_slice(), &[0, 1]),
            other => panic!("expected detector annotation, got {other:?}"),
        }

        let tick = tc.get_tick(0).unwrap();
        assert_eq!(tick.len(), 1);
        assert_eq!(tick.gate_batches()[0].gate_type, GateType::MZ);
        assert_eq!(
            tick.gate_batches()[0].meas_ids.as_slice(),
            &[MeasId(0), MeasId(1)]
        );
    }

    #[test]
    fn test_dag_to_tick_preserves_existing_measurement_ids_and_advances_counter() {
        let mut dag = DagCircuit::new();
        let mut gate = Gate::mz(&[0]);
        gate.meas_ids.push(MeasId(5));
        let node = dag.add_gate(gate);
        dag.observable_labeled("obs5", &[node]);

        let mut tc = TickCircuit::from(&dag);
        assert_eq!(tc.num_measurements(), 6);
        let tick = tc.get_tick(0).unwrap();
        assert_eq!(tick.gate_batches()[0].meas_ids.as_slice(), &[MeasId(5)]);
        match &tc.annotations()[0].kind {
            AnnotationKind::Observable { measurement_nodes } => {
                assert_eq!(measurement_nodes.as_slice(), &[5]);
            }
            other => panic!("expected observable annotation, got {other:?}"),
        }

        let next = tc.tick().mz(&[1]);
        assert_eq!(next[0].record_idx, 6);
        assert_eq!(next[0].meas_id, MeasId(6));
        assert_eq!(tc.num_measurements(), 7);
    }

    #[test]
    #[should_panic(expected = "annotation references non-measurement DAG node")]
    fn test_dag_to_tick_rejects_annotation_referencing_non_measurement_node() {
        let mut dag = DagCircuit::new();
        let h = dag.add_gate(Gate::h(&[0]));
        dag.detector_labeled("not-a-measurement", &[h]);

        let _ = TickCircuit::from(&dag);
    }

    #[test]
    fn test_detector_observable_and_tracked_pauli_remain_distinct_after_round_trip() {
        use pecos_core::pauli::{X, Z};

        let mut tc1 = TickCircuit::new();
        tc1.tick().pz(&[0, 1, 2]);
        let ms = tc1.tick().mz(&[0, 1]);
        tc1.detector_labeled("detector", &[ms[0]]);
        tc1.observable_labeled("observable", &[ms[1]]);
        tc1.tracked_pauli_labeled("tracked", X(0) & Z(2));

        let tc2 = TickCircuit::from(&DagCircuit::from(&tc1));
        assert_eq!(tc2.annotations().len(), 3);

        assert_eq!(tc2.annotations()[0].label.as_deref(), Some("detector"));
        match &tc2.annotations()[0].kind {
            AnnotationKind::Detector {
                measurement_nodes, ..
            } => assert_eq!(measurement_nodes.as_slice(), &[0]),
            other => panic!("expected detector annotation, got {other:?}"),
        }

        assert_eq!(tc2.annotations()[1].label.as_deref(), Some("observable"));
        match &tc2.annotations()[1].kind {
            AnnotationKind::Observable { measurement_nodes } => {
                assert_eq!(measurement_nodes.as_slice(), &[1]);
            }
            other => panic!("expected observable annotation, got {other:?}"),
        }

        assert_eq!(tc2.annotations()[2].label.as_deref(), Some("tracked"));
        assert!(matches!(
            tc2.annotations()[2].kind,
            AnnotationKind::TrackedPauli
        ));
        assert_eq!(tc2.annotations()[2].pauli, X(0) & Z(2));
    }

    #[test]
    fn test_small_pseudorandom_tick_dag_round_trip_invariants() {
        fn assert_no_tick_overlaps(circuit: &TickCircuit) {
            for (tick_idx, tick) in circuit.iter_ticks() {
                let mut active = BTreeSet::new();
                for gate in tick.gate_batches() {
                    gate.validate()
                        .unwrap_or_else(|err| panic!("invalid gate in tick {tick_idx}: {err}"));
                    for &qubit in &gate.qubits {
                        assert!(
                            active.insert(qubit),
                            "qubit {qubit:?} appears more than once in tick {tick_idx}"
                        );
                    }
                }
            }
        }

        fn measurement_ids(circuit: &TickCircuit) -> Vec<MeasId> {
            circuit
                .iter_gate_batches()
                .flat_map(|gate| gate.as_gate().meas_ids.iter().copied())
                .collect()
        }

        let mut state = 0x5eed_u64;
        for case_idx in 0..16 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1);
            let base = ((state >> 32) as usize % 4) * 10;

            let mut tc1 = TickCircuit::new();
            tc1.tick()
                .meta("case", Attribute::Int(case_idx))
                .h(&[base, base + 1])
                .meta("role", Attribute::String("prepare".into()));

            if state & 1 == 0 {
                tc1.tick()
                    .cx(&[(base, base + 2), (base + 1, base + 3)])
                    .meta("role", Attribute::String("entangle".into()));
            } else {
                tc1.tick()
                    .rz(Angle64::from_turns(0.25), &[base])
                    .rz(Angle64::from_turns(0.25), &[base + 1])
                    .rz(Angle64::from_turns(0.5), &[base + 2]);
            }

            let ms = tc1.tick().mz(&[base, base + 1]);
            if case_idx % 2 == 0 {
                tc1.detector_labeled("det", &ms);
            } else {
                tc1.observable_labeled("obs", &ms);
            }

            let tc2 = TickCircuit::from(&DagCircuit::from(&tc1));
            assert_eq!(tc2.gate_count(), tc1.gate_count());
            assert_eq!(tc2.num_measurements(), tc1.num_measurements());
            assert_eq!(tc2.gate_counts_by_type(), tc1.gate_counts_by_type());
            assert_eq!(measurement_ids(&tc2), measurement_ids(&tc1));
            assert_eq!(tc2.annotations().len(), tc1.annotations().len());
            assert_no_tick_overlaps(&tc2);
        }
    }

    #[test]
    fn test_pseudorandom_round_trip_preserves_measurement_annotation_details() {
        use pecos_core::pauli::{X, Y, Z};

        fn annotation_by_label<'a>(circuit: &'a TickCircuit, label: &str) -> &'a PauliAnnotation {
            circuit
                .annotations()
                .iter()
                .find(|ann| ann.label.as_deref() == Some(label))
                .unwrap_or_else(|| panic!("missing annotation {label}"))
        }

        let mut state = 0xaced_u64;
        for case_idx in 0..12 {
            state = state
                .wrapping_mul(2_862_933_555_777_941_757)
                .wrapping_add(3_037_000_493);
            let base = 20 * case_idx;

            let mut tc1 = TickCircuit::new();
            tc1.tick().h(&[base, base + 1]).cx(&[(base + 2, base + 3)]);
            if state & 1 == 0 {
                tc1.tick().cx(&[(base, base + 2), (base + 1, base + 3)]);
            } else {
                tc1.tick()
                    .rz(Angle64::from_turns(0.25), &[base])
                    .rz(Angle64::from_turns(0.25), &[base + 1]);
            }

            let measurements = tc1.tick().mz(&[base, base + 1, base + 2]);
            let detector_records = if state & 2 == 0 {
                vec![measurements[0], measurements[2]]
            } else {
                vec![measurements[1]]
            };
            let observable_records = if state & 4 == 0 {
                vec![measurements[1]]
            } else {
                vec![measurements[0], measurements[2]]
            };
            let tracked = if state & 8 == 0 {
                X(base) & Z(base + 3)
            } else {
                Y(base + 3)
            };

            tc1.detector_labeled(&format!("det-{case_idx}"), &detector_records);
            tc1.observable_labeled(&format!("obs-{case_idx}"), &observable_records);
            tc1.tracked_pauli_labeled(&format!("track-{case_idx}"), tracked.clone());

            let tc2 = TickCircuit::from(&DagCircuit::from(&tc1));
            assert_eq!(tc2.gate_count(), tc1.gate_count(), "case {case_idx}");
            assert_eq!(
                tc2.num_measurements(),
                tc1.num_measurements(),
                "case {case_idx}"
            );
            assert_eq!(tc2.annotations().len(), 3, "case {case_idx}");

            let det = annotation_by_label(&tc2, &format!("det-{case_idx}"));
            match &det.kind {
                AnnotationKind::Detector {
                    measurement_nodes, ..
                } => assert_eq!(
                    measurement_nodes,
                    &detector_records
                        .iter()
                        .map(|m| m.record_idx)
                        .collect::<Vec<_>>(),
                    "case {case_idx}"
                ),
                other => panic!("expected detector annotation, got {other:?}"),
            }

            let obs = annotation_by_label(&tc2, &format!("obs-{case_idx}"));
            match &obs.kind {
                AnnotationKind::Observable { measurement_nodes } => assert_eq!(
                    measurement_nodes,
                    &observable_records
                        .iter()
                        .map(|m| m.record_idx)
                        .collect::<Vec<_>>(),
                    "case {case_idx}"
                ),
                other => panic!("expected observable annotation, got {other:?}"),
            }

            let track = annotation_by_label(&tc2, &format!("track-{case_idx}"));
            assert!(matches!(track.kind, AnnotationKind::TrackedPauli));
            assert_eq!(track.pauli, tracked, "case {case_idx}");
        }
    }

    #[test]
    fn test_all_standard_gate_families_round_trip_through_dag() {
        use pecos_core::pauli::{X, Z};

        fn channel_payloads(circuit: &TickCircuit) -> Vec<pecos_core::ChannelExpr> {
            circuit
                .iter_gate_batches()
                .filter_map(|gate| gate.channel.clone())
                .collect()
        }

        fn nonzero_gate_counts(
            circuit: &TickCircuit,
        ) -> std::collections::BTreeMap<GateType, usize> {
            circuit
                .gate_counts_by_type()
                .into_iter()
                .filter(|(_, count)| *count > 0)
                .collect()
        }

        let mut tc1 = TickCircuit::new();
        tc1.tick()
            .x(&[0])
            .y(&[1])
            .z(&[2])
            .h(&[3])
            .sx(&[4])
            .sxdg(&[5])
            .sy(&[6])
            .sydg(&[7])
            .sz(&[8])
            .szdg(&[9])
            .f(&[10])
            .fdg(&[11])
            .iden(&[12]);
        tc1.tick()
            .cx(&[(20, 21)])
            .cy(&[(22, 23)])
            .cz(&[(24, 25)])
            .sxx(&[(26, 27)])
            .sxxdg(&[(28, 29)])
            .syy(&[(30, 31)])
            .syydg(&[(32, 33)])
            .szz(&[(34, 35)])
            .szzdg(&[(36, 37)])
            .swap(&[(38, 39)]);
        tc1.tick()
            .rx(Angle64::from_turns(0.125), &[40])
            .ry(Angle64::from_turns(0.25), &[41])
            .rz(Angle64::from_turns(0.375), &[42])
            .r1xy(Angle64::from_turns(0.125), Angle64::from_turns(0.25), &[43])
            .u(
                Angle64::from_turns(0.125),
                Angle64::from_turns(0.25),
                Angle64::from_turns(0.375),
                &[44],
            );
        tc1.tick()
            .channel(pecos_core::channel::Depolarizing(0.01, 50));
        tc1.tick()
            .channel(pecos_core::channel::Depolarizing2(0.02, 51, 52));
        tc1.tick().idle(3u64, &[60]);
        tc1.tick().pz(&[70, 71]);
        let ms = tc1.tick().mz(&[70, 71]);
        tc1.detector_labeled("det-all-gates", &[ms[0]]);
        tc1.observable_labeled("obs-all-gates", &[ms[1]]);
        tc1.tracked_pauli_labeled("tracked-all-gates", X(70) & Z(71));

        let tc2 = TickCircuit::from(&DagCircuit::from(&tc1));

        assert_eq!(tc2.gate_count(), tc1.gate_count());
        assert_eq!(tc2.num_measurements(), tc1.num_measurements());
        assert_eq!(nonzero_gate_counts(&tc2), nonzero_gate_counts(&tc1));
        assert_eq!(channel_payloads(&tc2), channel_payloads(&tc1));
        assert_eq!(tc2.annotations().len(), tc1.annotations().len());
        assert_eq!(
            tc2.annotations()
                .iter()
                .map(|ann| ann.label.as_deref())
                .collect::<Vec<_>>(),
            tc1.annotations()
                .iter()
                .map(|ann| ann.label.as_deref())
                .collect::<Vec<_>>()
        );
        assert!(matches!(
            tc2.annotations()[0].kind,
            AnnotationKind::Detector { .. }
        ));
        assert!(matches!(
            tc2.annotations()[1].kind,
            AnnotationKind::Observable { .. }
        ));
        assert!(matches!(
            tc2.annotations()[2].kind,
            AnnotationKind::TrackedPauli
        ));
        assert!(tc2.has_channel_operations());
    }

    #[test]
    fn test_seeded_mixed_standard_gate_round_trip_preserves_metadata_annotations_and_batches() {
        use pecos_core::pauli::{X, Y, Z};

        fn apply_single(tick: &mut TickHandle<'_>, gate_type: GateType, qubit: usize) {
            match gate_type {
                GateType::X => {
                    tick.x(&[qubit]);
                }
                GateType::Y => {
                    tick.y(&[qubit]);
                }
                GateType::Z => {
                    tick.z(&[qubit]);
                }
                GateType::H => {
                    tick.h(&[qubit]);
                }
                GateType::SZ => {
                    tick.sz(&[qubit]);
                }
                GateType::SZdg => {
                    tick.szdg(&[qubit]);
                }
                GateType::SX => {
                    tick.sx(&[qubit]);
                }
                GateType::SXdg => {
                    tick.sxdg(&[qubit]);
                }
                GateType::SY => {
                    tick.sy(&[qubit]);
                }
                GateType::SYdg => {
                    tick.sydg(&[qubit]);
                }
                GateType::F => {
                    tick.f(&[qubit]);
                }
                GateType::Fdg => {
                    tick.fdg(&[qubit]);
                }
                other => panic!("unexpected single-qubit gate {other:?}"),
            }
        }

        fn apply_pair(tick: &mut TickHandle<'_>, gate_type: GateType, a: usize, b: usize) {
            match gate_type {
                GateType::CX => {
                    tick.cx(&[(a, b)]);
                }
                GateType::CY => {
                    tick.cy(&[(a, b)]);
                }
                GateType::CZ => {
                    tick.cz(&[(a, b)]);
                }
                GateType::SXX => {
                    tick.sxx(&[(a, b)]);
                }
                GateType::SXXdg => {
                    tick.sxxdg(&[(a, b)]);
                }
                GateType::SYY => {
                    tick.syy(&[(a, b)]);
                }
                GateType::SYYdg => {
                    tick.syydg(&[(a, b)]);
                }
                GateType::SZZ => {
                    tick.szz(&[(a, b)]);
                }
                GateType::SZZdg => {
                    tick.szzdg(&[(a, b)]);
                }
                GateType::SWAP => {
                    tick.swap(&[(a, b)]);
                }
                other => panic!("unexpected two-qubit gate {other:?}"),
            }
        }

        fn nonzero_gate_counts(
            circuit: &TickCircuit,
        ) -> std::collections::BTreeMap<GateType, usize> {
            circuit
                .gate_counts_by_type()
                .into_iter()
                .filter(|(_, count)| *count > 0)
                .collect()
        }

        fn choose_gate(gates: &[GateType], state: u64, shift: u32) -> GateType {
            let len = u64::try_from(gates.len()).unwrap();
            let idx = usize::try_from((state >> shift) % len).unwrap();
            gates[idx]
        }

        const ONE_Q: &[GateType] = &[
            GateType::X,
            GateType::Y,
            GateType::Z,
            GateType::H,
            GateType::SZ,
            GateType::SZdg,
            GateType::SX,
            GateType::SXdg,
            GateType::SY,
            GateType::SYdg,
            GateType::F,
            GateType::Fdg,
        ];
        const TWO_Q: &[GateType] = &[
            GateType::CX,
            GateType::CY,
            GateType::CZ,
            GateType::SXX,
            GateType::SXXdg,
            GateType::SYY,
            GateType::SYYdg,
            GateType::SZZ,
            GateType::SZZdg,
            GateType::SWAP,
        ];

        let mut state = 0xdecaf_bad5eed_u64;
        for case_idx in 0..10usize {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let base = 100 * case_idx;
            let one_a = choose_gate(ONE_Q, state, 0);
            let one_b = choose_gate(ONE_Q, state, 8);
            let two = choose_gate(TWO_Q, state, 16);
            let rz_bucket = u8::try_from((state >> 24) & 3).unwrap();
            let rz_turns = f64::from(rz_bucket + 1) / 8.0;

            let mut tc1 = TickCircuit::new();
            tc1.set_meta("case", Attribute::Int(i64::try_from(case_idx).unwrap()));
            {
                let mut tick = tc1.tick();
                tick.meta("round", Attribute::Int(0))
                    .meta("kind", Attribute::String("single".into()));
                apply_single(&mut tick, one_a, base);
                apply_single(&mut tick, one_b, base + 1);
            }
            {
                let mut tick = tc1.tick();
                tick.meta("round", Attribute::Int(1))
                    .meta("kind", Attribute::String("mixed".into()));
                apply_pair(&mut tick, two, base + 2, base + 3);
                tick.rz(Angle64::from_turns(rz_turns), &[base + 4])
                    .rx(Angle64::from_turns(0.25), &[base + 5]);
            }
            tc1.tick()
                .meta("round", Attribute::Int(2))
                .channel(pecos_core::channel::Depolarizing(0.01, base + 6));
            tc1.tick().pz(&[base, base + 1, base + 2]);
            let measurements = tc1.tick().mz(&[base, base + 1, base + 2]);
            tc1.detector_labeled(
                &format!("det-{case_idx}"),
                &[measurements[0], measurements[2]],
            );
            tc1.observable_labeled(&format!("obs-{case_idx}"), &[measurements[1]]);
            let tracked = if state & (1 << 32) == 0 {
                X(base) & Z(base + 3)
            } else {
                Y(base + 2)
            };
            tc1.tracked_pauli_labeled(&format!("tracked-{case_idx}"), tracked.clone());

            let tc2 = TickCircuit::from(&DagCircuit::from(&tc1));

            assert_eq!(
                tc2.get_meta("case"),
                tc1.get_meta("case"),
                "case {case_idx}"
            );
            assert_eq!(tc2.gate_count(), tc1.gate_count(), "case {case_idx}");
            assert_eq!(
                tc2.num_measurements(),
                tc1.num_measurements(),
                "case {case_idx}"
            );
            assert_eq!(
                nonzero_gate_counts(&tc2),
                nonzero_gate_counts(&tc1),
                "case {case_idx}"
            );
            assert_eq!(tc2.annotations().len(), 3, "case {case_idx}");
            assert_eq!(
                tc2.annotations()
                    .iter()
                    .map(|ann| ann.label.as_deref())
                    .collect::<Vec<_>>(),
                tc1.annotations()
                    .iter()
                    .map(|ann| ann.label.as_deref())
                    .collect::<Vec<_>>(),
                "case {case_idx}"
            );
            assert!(matches!(
                tc2.annotations()[0].kind,
                AnnotationKind::Detector { .. }
            ));
            assert!(matches!(
                tc2.annotations()[1].kind,
                AnnotationKind::Observable { .. }
            ));
            assert!(matches!(
                tc2.annotations()[2].kind,
                AnnotationKind::TrackedPauli
            ));
            assert_eq!(tc2.annotations()[2].pauli, tracked, "case {case_idx}");
            assert!(tc2.has_channel_operations(), "case {case_idx}");
        }
    }

    #[test]
    fn test_batching_invariants_cover_parameters_channels_and_measurements() {
        let same_rz = Gate::rz(Angle64::from_turns(0.25), &[0]);
        let same_rz_disjoint = Gate::rz(Angle64::from_turns(0.25), &[1]);
        let different_rz = Gate::rz(Angle64::from_turns(0.5), &[2]);
        assert!(same_rz.can_batch_with(&same_rz_disjoint));
        assert!(!same_rz.can_batch_with(&different_rz));

        let channel0 = Gate::channel(pecos_core::channel::Depolarizing(0.01, 10));
        let channel1 = Gate::channel(pecos_core::channel::Depolarizing(0.01, 11));
        assert!(!channel0.can_batch_with(&channel1));

        let mut tick = Tick::new();
        tick.add_gate(same_rz);
        tick.add_gate(same_rz_disjoint);
        tick.merge_compatible_gate_at(1);
        tick.add_gate(different_rz);
        tick.add_gate(channel0);
        tick.add_gate(channel1);

        assert_eq!(tick.gate_count(), 5);
        assert_eq!(tick.gate_batch_count(), 4);
        assert_eq!(
            tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(0), QubitId::from(1)]
        );
        assert!(tick.gate_batches()[2].is_channel());
        assert!(tick.gate_batches()[3].is_channel());

        let mut meas = TickCircuit::new();
        meas.reserve_ticks(1);
        let refs0 = meas.tick_at(0).mz(&[0, 1]);
        meas.get_tick_mut(0).unwrap().set_gate_attr(
            refs0[0].gate_idx,
            "readout_family",
            Attribute::String("fast".into()),
        );
        let refs2 = meas.tick_at(0).mz(&[2]);
        let tick = meas.get_tick_mut(0).unwrap();
        tick.set_gate_attr(
            refs2[0].gate_idx,
            "readout_family",
            Attribute::String("slow".into()),
        );
        tick.merge_compatible_gate_at(refs2[0].gate_idx);

        let tick = meas.get_tick(0).unwrap();
        assert_eq!(tick.len(), 2);
        assert_eq!(tick.gate_count(), 3);
        assert_eq!(tick.gate_batch_count(), 2);
        assert_eq!(
            tick.gate_batches()[0].meas_ids.as_slice(),
            &[MeasId(0), MeasId(1)]
        );
        assert_eq!(tick.gate_batches()[1].meas_ids.as_slice(), &[MeasId(2)]);
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
    fn test_batched_two_qubit_gate_accepts_disjoint_pairs_and_rejects_overlap() {
        let mut tick = Tick::new();
        assert!(tick.try_add_gate(Gate::cx(&[(0, 1), (2, 3)])).is_ok());
        assert_eq!(tick.len(), 1);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batch_count(), 1);

        let err = Tick::new()
            .try_add_gate(Gate::cx(&[(0, 1), (1, 2)]))
            .unwrap_err();
        match err {
            TickGateError::InvalidGate { message, .. } => {
                assert!(message.contains("requires distinct qubits"));
                assert!(message.contains('1'));
            }
            TickGateError::QubitConflict(_) => panic!("overlap within one gate command is invalid"),
        }
    }

    #[test]
    fn test_try_add_gate_conflict() {
        let mut tc = TickCircuit::new();
        let mut handle = tc.tick();
        handle.h(&[0]);

        // Try to add another gate on the same qubit - should fail
        let result = handle.try_add_gate(Gate::x(&[0]));

        match result {
            Err(TickGateError::QubitConflict(err)) => {
                assert_eq!(err.conflicting_qubits, vec![QubitId::from(0)]);
                assert_eq!(err.tick_idx, Some(0));
            }
            Ok(_) => panic!("Expected conflict error"),
            Err(err) => panic!("Expected conflict error, got {err}"),
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
    fn test_try_add_gate_rejects_invalid_gate_payload_before_storage() {
        let mut tc = TickCircuit::new();
        let mut handle = tc.tick();
        let invalid = Gate::cx(&[(0, 0)]);

        let result = handle.try_add_gate(invalid);

        match result {
            Err(TickGateError::InvalidGate {
                message, tick_idx, ..
            }) => {
                assert_eq!(tick_idx, Some(0));
                assert!(message.contains("requires distinct qubits"));
            }
            Ok(_) => panic!("Expected invalid-gate error"),
            Err(err) => panic!("Expected invalid-gate error, got {err}"),
        }
        assert!(tc.get_tick(0).unwrap().is_empty());
    }

    #[test]
    #[should_panic(expected = "Invalid gate in tick 0")]
    fn test_tick_handle_panics_on_invalid_gate_payload() {
        let mut tc = TickCircuit::new();
        tc.tick().cx(&[(0, 0)]);
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
        tc.tick()
            .h(&[0, 1, 2, 3])
            .meta("calibration", Attribute::String("h-cal".into()));
        tc.tick().cx(&[(0, 1), (2, 3)]);
        tc.tick().mz(&[0, 1, 2, 3]);

        // Test explicit batched views. These preserve full Gate payloads.
        let batches: Vec<_> = tc.iter_gate_batches().collect();
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0].batch_index(), 0);
        assert_eq!(batches[0].gate_type, GateType::H);
        assert_eq!(batches[0].num_gates(), 4);
        assert_eq!(batches[0].gate_count(), 4);
        assert_eq!(
            batches[0].get_attr("calibration"),
            Some(&Attribute::String("h-cal".into()))
        );
        assert_eq!(tc.get_tick(0).unwrap().gate_batches()[0].num_gates(), 4);

        let batches_with_tick: Vec<_> = tc.iter_gate_batches_with_tick().collect();
        assert_eq!(batches_with_tick.len(), 3);
        assert_eq!(batches_with_tick[2].0, 2); // Third batch is in tick 2
        assert_eq!(batches_with_tick[2].1.gate_type, GateType::MZ);

        let instances_with_tick: Vec<_> = tc.iter_gate_instances_with_tick().collect();
        assert_eq!(instances_with_tick.len(), 10);
        assert_eq!(instances_with_tick[0].0, 0);
        assert_eq!(instances_with_tick[0].1.batch_index(), 0);
        assert_eq!(instances_with_tick[0].1.instance_index(), 0);
        assert_eq!(instances_with_tick[0].1.gate_type(), GateType::H);
        assert_eq!(instances_with_tick[0].1.qubits(), &[QubitId::from(0)]);
        assert_eq!(
            instances_with_tick[0].1.to_gate().qubits.as_slice(),
            &[QubitId::from(0)]
        );
        assert_eq!(
            instances_with_tick[0].1.get_attr("calibration"),
            Some(&Attribute::String("h-cal".into()))
        );
        assert_eq!(instances_with_tick[3].1.qubits(), &[QubitId::from(3)]);
        assert_eq!(instances_with_tick[4].0, 1);
        assert_eq!(instances_with_tick[4].1.instance_index(), 0);
        assert_eq!(instances_with_tick[4].1.gate_type(), GateType::CX);
        assert_eq!(
            instances_with_tick[4].1.qubits(),
            &[QubitId::from(0), QubitId::from(1)]
        );
        assert_eq!(
            instances_with_tick[5].1.qubits(),
            &[QubitId::from(2), QubitId::from(3)]
        );
        assert_eq!(instances_with_tick[5].1.instance_index(), 1);
        assert_eq!(instances_with_tick[6].0, 2);
        assert_eq!(instances_with_tick[6].1.gate_type(), GateType::MZ);
        assert_eq!(instances_with_tick[6].1.qubits(), &[QubitId::from(0)]);
        assert_eq!(instances_with_tick[6].1.meas_ids(), &[MeasId(0)]);
        assert_eq!(
            instances_with_tick[6].1.to_gate().meas_ids.as_slice(),
            &[MeasId(0)]
        );
        assert_eq!(instances_with_tick[9].1.meas_ids(), &[MeasId(3)]);

        // Test iter_ticks
        let ticks: Vec<_> = tc.iter_ticks().collect();
        assert_eq!(ticks.len(), 3);

        // Test iter_gates_by_type
        let h_gates: Vec<_> = tc.iter_gates_by_type(GateType::H).collect();
        assert_eq!(h_gates.len(), 1);

        // Test all_qubits
        let qubits = tc.all_qubits();
        assert_eq!(qubits.len(), 4);
        assert!(qubits.contains(&QubitId::from(0)));
        assert!(qubits.contains(&QubitId::from(1)));
        assert!(qubits.contains(&QubitId::from(2)));
        assert!(qubits.contains(&QubitId::from(3)));

        // Test gate_counts_by_type
        let counts = tc.gate_counts_by_type();
        assert_eq!(counts.get(&GateType::H), Some(&4));
        assert_eq!(counts.get(&GateType::CX), Some(&2));
        assert_eq!(counts.get(&GateType::MZ), Some(&4));
    }

    #[test]
    fn test_gate_instance_to_gate_preserves_payloads_without_attrs() {
        let angle = Angle64::from_turn_ratio(1, 8);
        let mut rzz_tick = Tick::new();
        rzz_tick.add_gate(Gate::rzz(angle, &[(0, 1), (2, 3)]));
        rzz_tick.set_gate_attr(0, "calibration", Attribute::String("rzz-cal".into()));

        let rzz_instances: Vec<_> = rzz_tick.iter_gate_instances().collect();
        assert_eq!(rzz_instances.len(), 2);
        assert_eq!(
            rzz_instances[0].get_attr("calibration"),
            Some(&Attribute::String("rzz-cal".into()))
        );
        assert_eq!(
            rzz_instances[0].attrs().count(),
            1,
            "batch metadata remains available through the instance view"
        );
        assert_eq!(rzz_instances[0].angles(), &[angle]);
        assert_eq!(
            rzz_instances[0].to_gate(),
            Gate::rzz(angle, &[(0usize, 1usize)]),
            "materialized gates carry sliced support and payload, not attrs"
        );
        assert_eq!(
            rzz_instances[1].to_gate(),
            Gate::rzz(angle, &[(2usize, 3usize)])
        );

        let duration = 8.0_f64;
        let mut idle_tick = Tick::new();
        idle_tick.add_gate(Gate::idle(
            duration,
            vec![QubitId::from(4), QubitId::from(5)],
        ));
        let idle_instances: Vec<_> = idle_tick.iter_gate_instances().collect();
        assert_eq!(idle_instances.len(), 2);
        let idle_gate = idle_instances[1].to_gate();
        assert_eq!(idle_gate.gate_type, GateType::Idle);
        assert_eq!(idle_gate.qubits.as_slice(), &[QubitId::from(5)]);
        assert_eq!(idle_gate.params.len(), 1);
        assert_eq!(idle_gate.params[0].to_bits(), duration.to_bits());

        let mut meas_tc = TickCircuit::new();
        meas_tc.tick().mz(&[8, 9]);
        let meas_instances: Vec<_> = meas_tc.get_tick(0).unwrap().iter_gate_instances().collect();
        assert_eq!(meas_instances.len(), 2);
        assert_eq!(
            meas_instances[0].to_gate().meas_ids.as_slice(),
            &[MeasId(0)]
        );
        assert_eq!(
            meas_instances[1].to_gate().meas_ids.as_slice(),
            &[MeasId(1)]
        );

        let channel = pecos_core::channel::Depolarizing(0.125, 6);
        let mut channel_tick = Tick::new();
        channel_tick.add_gate(Gate::channel(channel.clone()));
        let channel_instances: Vec<_> = channel_tick.iter_gate_instances().collect();
        assert_eq!(channel_instances.len(), 1);
        let channel_gate = channel_instances[0].to_gate();
        assert_eq!(channel_gate.gate_type, GateType::Channel);
        assert_eq!(channel_gate.qubits.as_slice(), &[QubitId::from(6)]);
        assert_eq!(channel_gate.channel.as_ref(), Some(&channel));
    }

    #[test]
    fn test_gate_instance_iteration_skips_annotation_batches() {
        let mut tick = Tick::new();
        tick.add_gate(Gate::simple(
            GateType::TrackedPauliMeta,
            vec![QubitId::from(0), QubitId::from(1)],
        ));

        let batches: Vec<_> = tick.iter_gate_batches().collect();
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].gate_count(), 0);
        assert_eq!(tick.iter_gate_instances().count(), 0);
    }

    #[test]
    fn test_tick_to_dag_keeps_zero_gate_metadata_nodes_from_gate_count() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);
        tc.tick();
        tc.get_tick_mut(1).unwrap().add_gate(Gate::simple(
            GateType::TrackedPauliMeta,
            vec![QubitId::from(1), QubitId::from(2)],
        ));
        tc.tick().mz(&[0]);

        let dag = DagCircuit::from(&tc);

        assert_eq!(
            dag.gate_count(),
            2,
            "metadata nodes are not gate applications"
        );
        let metadata_nodes: Vec<_> = dag
            .nodes()
            .into_iter()
            .filter(|&node| {
                dag.gate(node)
                    .is_some_and(|gate| gate.gate_type == GateType::TrackedPauliMeta)
            })
            .collect();
        assert_eq!(metadata_nodes.len(), 1);
        assert_eq!(dag.gate(metadata_nodes[0]).unwrap().num_gates(), 0);
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
    fn test_reset_clears_annotations_and_measurement_counter() {
        let mut tc = TickCircuit::new();
        let first_measurement = tc.tick().mz(&[0]);
        tc.detector(&first_measurement);
        tc.observable(&first_measurement);
        tc.tracked_pauli(pecos_core::pauli::Z(0));

        assert_eq!(tc.num_measurements(), 1);
        assert_eq!(tc.annotations().len(), 3);

        tc.reset();

        assert_eq!(tc.num_ticks(), 0);
        assert_eq!(tc.num_measurements(), 0);
        assert!(tc.annotations().is_empty());

        let reused_measurement = tc.tick().mz(&[1]);
        assert_eq!(reused_measurement[0].record_idx, 0);
    }

    #[test]
    fn test_metadata_helpers_build_detector_and_observable_json() {
        let mut tc = TickCircuit::new();

        let detector_id = tc
            .add_detector_metadata(&[-1], Some(&[0.0, 1.0, 2.0]), Some("d0"), None)
            .unwrap();
        let observable_id = tc
            .add_observable_metadata(&[-1, -2], None, Some("L2"))
            .unwrap();

        assert_eq!(detector_id, 0);
        assert_eq!(observable_id, 2);
        assert_eq!(tc.get_meta("num_detectors"), Some(&Attribute::Int(1)));
        assert_eq!(tc.get_meta("num_observables"), Some(&Attribute::Int(3)));

        let detectors = match tc.get_meta("detectors").unwrap() {
            Attribute::String(value) => value,
            other => panic!("expected detectors JSON string, got {other:?}"),
        };
        assert_eq!(
            detectors,
            r#"[{"coords":[0.0,1.0,2.0],"id":0,"label":"d0","records":[-1]}]"#
        );

        let observables = match tc.get_meta("observables").unwrap() {
            Attribute::String(value) => value,
            other => panic!("expected observables JSON string, got {other:?}"),
        };
        assert_eq!(observables, r#"[{"id":2,"label":"L2","records":[-1,-2]}]"#);
    }

    #[test]
    fn test_metadata_helpers_reject_conflicts_and_duplicates() {
        let mut tc = TickCircuit::new();

        let err = tc
            .add_observable_metadata(&[-1], Some(1), Some("L2"))
            .unwrap_err();
        assert!(err.contains("conflicts"));

        tc.add_detector_metadata(&[-1], None, None, Some(7))
            .unwrap();
        let err = tc
            .add_detector_metadata(&[-2], None, None, Some(7))
            .unwrap_err();
        assert!(err.contains("already contains detector_id 7"));

        tc.add_observable_metadata(&[-1], Some(3), None).unwrap();
        let err = tc
            .add_observable_metadata(&[-2], Some(3), None)
            .unwrap_err();
        assert!(err.contains("already contains observable_id 3"));
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
        assert_eq!(tick0.gate_batches()[0].gate_type, GateType::H);

        let tick1 = tc.get_tick(1).unwrap();
        assert_eq!(tick1.gate_batches()[0].gate_type, GateType::X);

        let tick2 = tc.get_tick(2).unwrap();
        assert_eq!(tick2.gate_batches()[0].gate_type, GateType::CX);
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
        assert_eq!(tick0.gate_batches()[0].gate_type, GateType::Z);
    }

    #[test]
    fn test_insert_tick_at_end() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);

        // Insert at the end (same as tick())
        tc.insert_tick(1).x(&[1]);

        assert_eq!(tc.num_ticks(), 2);

        let tick1 = tc.get_tick(1).unwrap();
        assert_eq!(tick1.gate_batches()[0].gate_type, GateType::X);
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
        assert_eq!(
            tc.get_tick(0).unwrap().gate_batches()[0].gate_type,
            GateType::H
        );
        assert_eq!(
            tc.get_tick(1).unwrap().gate_batches()[0].gate_type,
            GateType::X
        );
        assert_eq!(
            tc.get_tick(2).unwrap().gate_batches()[0].gate_type,
            GateType::CX
        );
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
        assert_eq!(tick.gate_batches()[0].gate_type, GateType::X);
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
    fn test_tick_remove_gate_preserves_aligned_attrs() {
        let mut tc = TickCircuit::new();
        tc.tick()
            .h(&[0])
            .meta("h_attr", Attribute::Int(1))
            .x(&[1])
            .meta("x_attr", Attribute::Int(2))
            .z(&[2])
            .meta("z_attr", Attribute::Int(3));

        let tick = tc.get_tick_mut(0).unwrap();
        let removed = tick.remove_gate(0).expect("H batch should exist");

        assert_eq!(removed.gate_type, GateType::H);
        assert_eq!(tick.len(), 2);
        assert_eq!(tick.gate_batches()[0].gate_type, GateType::X);
        assert_eq!(tick.gate_batches()[1].gate_type, GateType::Z);
        assert_eq!(tick.get_gate_attr(0, "x_attr"), Some(&Attribute::Int(2)));
        assert_eq!(tick.get_gate_attr(1, "z_attr"), Some(&Attribute::Int(3)));
        assert!(tick.get_gate_attr(0, "h_attr").is_none());
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
        assert_eq!(tick.gate_batches()[0].gate_type, GateType::H);
        assert_eq!(tick.gate_batches()[1].gate_type, GateType::Z);
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
    fn test_compact_ticks_preserves_and_merges_aligned_batch_attrs() {
        let mut tc = TickCircuit::new();
        tc.tick()
            .h(&[0])
            .meta("calibration", Attribute::String("shared".into()));
        tc.tick()
            .h(&[1])
            .meta("calibration", Attribute::String("shared".into()));

        tc.compact_ticks();

        assert_eq!(tc.num_ticks(), 1);
        let tick = tc.get_tick(0).unwrap();
        assert_eq!(tick.len(), 1);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batch_count(), 1);
        assert_eq!(
            tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(0), QubitId::from(1)]
        );
        assert_eq!(
            tick.get_gate_attr(0, "calibration"),
            Some(&Attribute::String("shared".into()))
        );
    }

    #[test]
    fn test_compact_ticks_keeps_different_batch_attrs_separate() {
        let mut tc = TickCircuit::new();
        tc.tick()
            .h(&[0])
            .meta("calibration", Attribute::String("a".into()));
        tc.tick()
            .h(&[1])
            .meta("calibration", Attribute::String("b".into()));

        tc.compact_ticks();

        assert_eq!(tc.num_ticks(), 1);
        let tick = tc.get_tick(0).unwrap();
        assert_eq!(tick.len(), 2);
        assert_eq!(tick.gate_count(), 2);
        assert_eq!(tick.gate_batch_count(), 2);
        assert_eq!(
            tick.get_gate_attr(0, "calibration"),
            Some(&Attribute::String("a".into()))
        );
        assert_eq!(
            tick.get_gate_attr(1, "calibration"),
            Some(&Attribute::String("b".into()))
        );
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

    // --- Gate signature validation tests ---

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
        let gate = &tick.gate_batches()[0];
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
        let mut sigs = BTreeMap::new();
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

    #[test]
    fn test_tick_circuit_annotations() {
        use pecos_core::pauli::X;

        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1, 2]);
        tc.tick().cx(&[(0, 2)]);
        tc.tick().cx(&[(1, 2)]);
        let ms = tc.tick().mz(&[2]);

        assert_eq!(ms.len(), 1);
        assert_eq!(ms[0].qubit, QubitId::from(2));

        tc.detector_labeled("Z_check", &ms);
        tc.observable_labeled("logical_Z", &ms);
        tc.tracked_pauli_labeled("logical_X", X(0) & X(1));

        assert_eq!(tc.annotations().len(), 3);
        assert_eq!(tc.annotations()[0].label.as_deref(), Some("Z_check"));
        assert_eq!(tc.annotations()[1].label.as_deref(), Some("logical_Z"));
        assert_eq!(tc.annotations()[2].label.as_deref(), Some("logical_X"));
    }

    #[test]
    fn test_tick_to_dag_annotation_transfer() {
        use pecos_core::pauli::Z;

        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1, 2]);
        tc.tick().cx(&[(0, 2)]);
        tc.tick().cx(&[(1, 2)]);
        let ms = tc.tick().mz(&[2]);
        tc.detector_labeled("det0", &ms);
        tc.observable_labeled("obs0", &ms);
        tc.tracked_pauli_labeled("op0", Z(0) & Z(1));

        let dag = DagCircuit::from(&tc);

        // Annotations should transfer
        assert_eq!(dag.annotations().len(), 3);
        assert_eq!(dag.annotations()[0].label.as_deref(), Some("det0"));
        assert_eq!(dag.annotations()[1].label.as_deref(), Some("obs0"));
        assert_eq!(dag.annotations()[2].label.as_deref(), Some("op0"));

        // Kinds preserved
        assert!(matches!(
            dag.annotations()[0].kind,
            crate::dag_circuit::AnnotationKind::Detector { .. }
        ));
        assert!(matches!(
            dag.annotations()[1].kind,
            crate::dag_circuit::AnnotationKind::Observable { .. }
        ));
        assert!(matches!(
            dag.annotations()[2].kind,
            crate::dag_circuit::AnnotationKind::TrackedPauli
        ));
    }

    #[test]
    fn test_dag_to_tick_annotation_transfer() {
        use pecos_core::pauli::X;

        let mut dag = DagCircuit::new();
        dag.pz(&[0, 1]);
        dag.cx(&[(0, 1)]);
        let ms = dag.mz(&[0, 1]);
        dag.detector_labeled("d0", &[ms[0]]);
        dag.observable_labeled("o0", &[ms[0], ms[1]]);
        dag.tracked_pauli_labeled("p0", X(0) & X(1));

        let tc = TickCircuit::from(&dag);

        assert_eq!(tc.annotations().len(), 3);
        assert_eq!(tc.annotations()[0].label.as_deref(), Some("d0"));
        assert_eq!(tc.annotations()[1].label.as_deref(), Some("o0"));
        assert_eq!(tc.annotations()[2].label.as_deref(), Some("p0"));
    }

    #[test]
    fn test_annotation_round_trip() {
        use pecos_core::pauli::X;

        // Build TickCircuit with annotations
        let mut tc1 = TickCircuit::new();
        tc1.tick().pz(&[0, 1, 2]);
        tc1.tick().cx(&[(0, 2)]);
        tc1.tick().cx(&[(1, 2)]);
        let ms = tc1.tick().mz(&[2]);
        tc1.detector_labeled("syndr", &ms);
        let ms_data = tc1.tick().mz(&[0, 1]);
        tc1.observable_labeled("log_Z", &ms_data);
        tc1.tracked_pauli_labeled("log_X", X(0) & X(1));

        // TickCircuit -> DagCircuit -> TickCircuit
        let dag = DagCircuit::from(&tc1);
        let tc2 = TickCircuit::from(&dag);

        // Annotation count and labels preserved
        assert_eq!(tc2.annotations().len(), tc1.annotations().len());
        for (a1, a2) in tc1.annotations().iter().zip(tc2.annotations()) {
            assert_eq!(a1.label, a2.label);
            assert_eq!(a1.pauli, a2.pauli);
        }
    }

    #[test]
    fn test_mz_returns_refs() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        let ms = tc.tick().mz(&[0, 1]);

        assert_eq!(ms.len(), 2);
        assert_eq!(ms[0].qubit, QubitId::from(0));
        assert_eq!(ms[1].qubit, QubitId::from(1));
        // Both from same tick and gate
        assert_eq!(ms[0].tick, ms[1].tick);
        assert_eq!(ms[0].gate_idx, ms[1].gate_idx);
    }

    #[test]
    fn test_fill_idle_gates() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]); // qubit 1 idle
        tc.tick().cx(&[(0, 1)]); // none idle

        let count_before = tc.gate_count();
        tc.fill_idle_gates();
        let count_after = tc.gate_count();

        // Tick 0: qubit 1 was idle, should get an idle gate
        assert!(count_after > count_before, "Should have added idle gates");
    }

    #[test]
    fn test_channel_gate_is_first_class_tick_operation() {
        let mut tc = TickCircuit::new();
        tc.tick()
            .channel(pecos_core::channel::Depolarizing(0.25, 0));

        let gate = &tc.get_tick(0).unwrap().gate_batches()[0];
        assert_eq!(gate.gate_type, GateType::Channel);
        assert_eq!(gate.qubits.as_slice(), &[QubitId::from(0)]);
        assert!(gate.channel_expr().is_some());
        assert!(tc.has_channel_operations());
        assert!(gate.validate().is_ok());
    }

    #[test]
    fn test_with_noise_inserts_channel_ticks() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]).x(&[1]);
        tc.tick().cx(&[(0, 1)]);

        let noisy = tc.with_noise(&|gate: &Gate| {
            gate.qubits
                .iter()
                .map(|q| pecos_core::channel::Depolarizing(0.01, q.index()))
                .collect()
        });

        assert_eq!(noisy.num_ticks(), 4);
        assert_eq!(
            noisy.get_tick(0).unwrap().gate_batches()[0].gate_type,
            GateType::H
        );
        assert!(
            noisy
                .get_tick(1)
                .unwrap()
                .gate_batches()
                .iter()
                .all(Gate::is_channel)
        );
        assert_eq!(
            noisy.get_tick(2).unwrap().gate_batches()[0].gate_type,
            GateType::CX
        );
        assert!(
            noisy
                .get_tick(3)
                .unwrap()
                .gate_batches()
                .iter()
                .all(Gate::is_channel)
        );
    }

    #[test]
    fn test_with_noise_on_batched_source_gate_emits_per_qubit_channels() {
        use std::cell::{Cell, RefCell};

        let mut tc = TickCircuit::new();
        tc.tick().h(&[0, 1]);

        let calls = Cell::new(0);
        let seen_qubits = RefCell::new(Vec::new());
        let noisy = tc.with_noise(&|gate: &Gate| {
            calls.set(calls.get() + 1);
            seen_qubits.borrow_mut().push(gate.qubits.clone());
            gate.qubits
                .iter()
                .map(|qubit| pecos_core::channel::Depolarizing(0.01, qubit.index()))
                .collect()
        });

        assert_eq!(calls.get(), 1);
        assert_eq!(
            seen_qubits.borrow()[0].as_slice(),
            &[QubitId::from(0), QubitId::from(1)]
        );
        assert_eq!(noisy.num_ticks(), 2);
        assert_eq!(noisy.get_tick(0).unwrap().gate_count(), 2);

        let noise_tick = noisy.get_tick(1).unwrap();
        assert_eq!(noise_tick.len(), 2);
        assert_eq!(noise_tick.gate_count(), 2);
        assert!(noise_tick.gate_batches().iter().all(Gate::is_channel));
        assert_eq!(
            noise_tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(0)]
        );
        assert_eq!(
            noise_tick.gate_batches()[1].qubits.as_slice(),
            &[QubitId::from(1)]
        );
    }

    #[test]
    fn test_with_noise_on_batched_measurement_places_channels_after_measurement_tick() {
        use std::cell::{Cell, RefCell};

        let mut tc = TickCircuit::new();
        tc.tick().mz(&[0, 1]);

        let calls = Cell::new(0);
        let seen_measurement_ids = RefCell::new(Vec::new());
        let noisy = tc.with_noise(&|gate: &Gate| {
            assert_eq!(gate.gate_type, GateType::MZ);
            calls.set(calls.get() + 1);
            seen_measurement_ids
                .borrow_mut()
                .extend(gate.meas_ids.iter().copied());
            gate.qubits
                .iter()
                .map(|qubit| pecos_core::channel::Dephasing(0.02, qubit.index()))
                .collect()
        });

        assert_eq!(calls.get(), 1);
        assert_eq!(
            seen_measurement_ids.borrow().as_slice(),
            &[MeasId(0), MeasId(1)]
        );
        assert_eq!(noisy.num_ticks(), 2);

        let meas_tick = noisy.get_tick(0).unwrap();
        assert_eq!(meas_tick.gate_batches()[0].gate_type, GateType::MZ);
        assert_eq!(
            meas_tick.gate_batches()[0].meas_ids.as_slice(),
            &[MeasId(0), MeasId(1)]
        );

        let noise_tick = noisy.get_tick(1).unwrap();
        assert_eq!(noise_tick.len(), 2);
        assert!(noise_tick.gate_batches().iter().all(Gate::is_channel));
        assert_eq!(
            noise_tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(0)]
        );
        assert_eq!(
            noise_tick.gate_batches()[1].qubits.as_slice(),
            &[QubitId::from(1)]
        );
    }

    #[test]
    fn test_with_noise_empty_channels_preserves_tick_structure() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]).x(&[1]);
        tc.tick().cx(&[(0, 1)]);
        tc.tick().mz(&[0, 1]);

        let noisy = tc.with_noise(&|_: &Gate| Vec::new());

        assert_eq!(noisy.num_ticks(), tc.num_ticks());
        for tick_idx in 0..tc.num_ticks() {
            let original = tc.get_tick(tick_idx).unwrap();
            let copied = noisy.get_tick(tick_idx).unwrap();
            assert_eq!(copied.gate_batches().len(), original.gate_batches().len());
            for (copied_gate, original_gate) in
                copied.gate_batches().iter().zip(original.gate_batches())
            {
                assert_eq!(copied_gate.gate_type, original_gate.gate_type);
                assert_eq!(copied_gate.qubits, original_gate.qubits);
                assert!(!copied_gate.is_channel());
            }
        }
    }

    #[test]
    fn test_with_noise_splits_conflicting_channel_ticks() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]);

        let noisy = tc.with_noise(&|_: &Gate| {
            vec![
                pecos_core::channel::Depolarizing(0.01, 0),
                pecos_core::channel::Dephasing(0.02, 0),
            ]
        });

        assert_eq!(noisy.num_ticks(), 3);
        assert_eq!(
            noisy.get_tick(0).unwrap().gate_batches()[0].gate_type,
            GateType::H
        );

        let first_noise_tick = noisy.get_tick(1).unwrap();
        assert_eq!(first_noise_tick.gate_batches().len(), 1);
        assert!(first_noise_tick.gate_batches()[0].is_channel());
        assert_eq!(
            first_noise_tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(0)]
        );

        let second_noise_tick = noisy.get_tick(2).unwrap();
        assert_eq!(second_noise_tick.gate_batches().len(), 1);
        assert!(second_noise_tick.gate_batches()[0].is_channel());
        assert_eq!(
            second_noise_tick.gate_batches()[0].qubits.as_slice(),
            &[QubitId::from(0)]
        );
    }

    #[test]
    fn test_with_noise_places_measurement_channels_after_measurement_tick() {
        let mut tc = TickCircuit::new();
        tc.tick().mz(&[0]);

        let noisy = tc.with_noise(&|gate: &Gate| {
            if gate.gate_type == GateType::MZ {
                vec![pecos_core::channel::Dephasing(0.25, 0)]
            } else {
                Vec::new()
            }
        });

        assert_eq!(noisy.num_ticks(), 2);
        assert_eq!(
            noisy.get_tick(0).unwrap().gate_batches()[0].gate_type,
            GateType::MZ
        );
        let channel = &noisy.get_tick(1).unwrap().gate_batches()[0];
        assert!(channel.is_channel());
        assert_eq!(channel.qubits.as_slice(), &[QubitId::from(0)]);
    }

    #[test]
    fn test_with_noise_packs_disjoint_channels_and_splits_conflicts() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[0]).x(&[1]);

        let noisy = tc.with_noise(&|gate: &Gate| {
            gate.qubits
                .iter()
                .flat_map(|q| {
                    [
                        pecos_core::channel::Depolarizing(0.01, q.index()),
                        pecos_core::channel::Dephasing(0.02, q.index()),
                    ]
                })
                .collect()
        });

        assert_eq!(noisy.num_ticks(), 3);
        assert_eq!(noisy.get_tick(1).unwrap().gate_batches().len(), 2);
        assert_eq!(noisy.get_tick(2).unwrap().gate_batches().len(), 2);
        assert!(
            noisy
                .get_tick(1)
                .unwrap()
                .gate_batches()
                .iter()
                .all(Gate::is_channel)
        );
        assert!(
            noisy
                .get_tick(2)
                .unwrap()
                .gate_batches()
                .iter()
                .all(Gate::is_channel)
        );
    }

    #[test]
    fn test_with_noise_rejects_existing_channel_operations() {
        let mut tc = TickCircuit::new();
        tc.tick()
            .channel(pecos_core::channel::Depolarizing(0.25, 0));

        let err = tc
            .try_with_noise(&|_: &Gate| vec![pecos_core::channel::Depolarizing(0.01, 0)])
            .unwrap_err();

        assert!(err.contains("already contains channel operations"));
        assert!(err.contains("tick 0"));
        assert!(err.contains("gate 0"));
    }

    #[test]
    fn test_meas_record_idx_single_qubit() {
        let mut tc = TickCircuit::new();
        let m0 = tc.tick().mz(&[0]);
        let m1 = tc.tick().mz(&[1]);
        assert_eq!(m0[0].record_idx, 0);
        assert_eq!(m1[0].record_idx, 1);
    }

    #[test]
    fn test_meas_record_idx_multi_qubit() {
        let mut tc = TickCircuit::new();
        let ms = tc.tick().mz(&[0, 1, 2]);
        assert_eq!(ms[0].record_idx, 0);
        assert_eq!(ms[1].record_idx, 1);
        assert_eq!(ms[2].record_idx, 2);
        // Next measurement continues the count
        let m2 = tc.tick().mz(&[3]);
        assert_eq!(m2[0].record_idx, 3);
    }

    #[test]
    fn test_detector_uses_record_idx() {
        // Two qubits measured in one gate: detector referencing each
        // should get DIFFERENT record indices (not the same gate index).
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        let ms = tc.tick().mz(&[0, 1]);

        // Detector on qubit 0's measurement
        tc.detector(&[ms[0]]);
        // Detector on qubit 1's measurement
        tc.detector(&[ms[1]]);

        let anns = tc.annotations();
        match &anns[0].kind {
            AnnotationKind::Detector {
                measurement_nodes, ..
            } => {
                assert_eq!(measurement_nodes, &[0], "D0 should reference record 0 (q0)");
            }
            _ => panic!("Expected detector"),
        }
        match &anns[1].kind {
            AnnotationKind::Detector {
                measurement_nodes, ..
            } => {
                assert_eq!(measurement_nodes, &[1], "D1 should reference record 1 (q1)");
            }
            _ => panic!("Expected detector"),
        }
    }

    #[test]
    fn test_detector_multi_qubit_mz_no_xor_cancel() {
        // Bug regression: two refs from same multi-qubit MZ gate
        // used to have the same gate_idx, causing XOR cancellation.
        // With record_idx they should be distinct.
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[0, 1]);
        let ms = tc.tick().mz(&[0, 1]);

        // Detector comparing both measurements (XOR of records 0 and 1)
        tc.detector(&[ms[0], ms[1]]);

        let anns = tc.annotations();
        match &anns[0].kind {
            AnnotationKind::Detector {
                measurement_nodes, ..
            } => {
                assert_eq!(measurement_nodes.len(), 2);
                assert_ne!(
                    measurement_nodes[0], measurement_nodes[1],
                    "Two qubits from same MZ must have different record indices"
                );
            }
            _ => panic!("Expected detector"),
        }
    }
}
