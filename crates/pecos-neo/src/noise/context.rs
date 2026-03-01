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

//! Shared context for noise channels.
//!
//! The noise context maintains state that needs to be shared across
//! multiple noise channels, such as which qubits are currently leaked.
//!
//! This implementation uses bit vectors for O(1) state lookups, enabling
//! efficient simulation of large qubit counts (tested up to 1M+ qubits).
//!
//! ## Qubit Lifecycle
//!
//! A qubit can be in one of these states:
//! - **Unknown**: Never been prepared, doesn't exist in the system
//! - **Active**: Has been prepared and not yet measured out (or re-prepared after measurement)
//! - **Inactive**: Was prepared but has been measured (can be re-prepared)
//! - **Leaked**: Outside the computational subspace (preparation will unleak)
//!
//! ## Crosstalk Support
//!
//! The context supports two types of crosstalk targeting:
//! - **Local**: Affects only qubits near the gated qubits (specified by the device)
//! - **Global**: Affects all other active qubits in the system
//!
//! ## Performance
//!
//! All single-qubit operations (`is_leaked`, `mark_leaked`, etc.) are O(1).
//! Batch operations like `crosstalk_targets` scale linearly with qubit count
//! but use efficient bitwise operations.

use crate::command::GateType;
use crate::extensible::{GateCategory, GateDefinitions, GateId, GateSpec};
use pecos_core::{Angle64, QubitId, TimeUnits};
use smallvec::SmallVec;
use std::collections::BTreeSet;

// ============================================================================
// BitVec - Efficient bit vector implementation
// ============================================================================

/// Simple bit vector implementation using u64 words.
///
/// Provides O(1) get/set operations with excellent cache locality.
#[derive(Debug, Clone, Default)]
pub struct BitVec {
    words: Vec<u64>,
}

impl BitVec {
    /// Create a new bit vector with the given capacity (in bits).
    #[must_use]
    pub fn with_capacity(bits: usize) -> Self {
        let words = bits.div_ceil(64);
        Self {
            words: vec![0; words],
        }
    }

    /// Get bit at index. O(1).
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> bool {
        let word = index / 64;
        let bit = index % 64;
        if word >= self.words.len() {
            return false;
        }
        (self.words[word] >> bit) & 1 != 0
    }

    /// Set bit at index to true. O(1).
    #[inline]
    pub fn set(&mut self, index: usize) {
        let word = index / 64;
        let bit = index % 64;
        if word >= self.words.len() {
            self.words.resize(word + 1, 0);
        }
        self.words[word] |= 1 << bit;
    }

    /// Set bit at index to false. O(1).
    #[inline]
    pub fn clear(&mut self, index: usize) {
        let word = index / 64;
        let bit = index % 64;
        if word < self.words.len() {
            self.words[word] &= !(1 << bit);
        }
    }

    /// Clear all bits.
    pub fn clear_all(&mut self) {
        for word in &mut self.words {
            *word = 0;
        }
    }

    /// Count set bits.
    #[must_use]
    pub fn count_ones(&self) -> usize {
        self.words.iter().map(|w| w.count_ones() as usize).sum()
    }

    /// Iterate over indices of set bits.
    pub fn iter_ones(&self) -> impl Iterator<Item = usize> + '_ {
        self.words.iter().enumerate().flat_map(|(word_idx, &word)| {
            (0..64).filter_map(move |bit| {
                if (word >> bit) & 1 != 0 {
                    Some(word_idx * 64 + bit)
                } else {
                    None
                }
            })
        })
    }

    /// AND-NOT: self AND (NOT other), useful for "active but not leaked".
    pub fn and_not_iter<'a>(&'a self, other: &'a BitVec) -> impl Iterator<Item = usize> + 'a {
        self.words
            .iter()
            .enumerate()
            .flat_map(move |(word_idx, &word)| {
                let other_word = other.words.get(word_idx).copied().unwrap_or(0);
                let combined = word & !other_word;
                (0..64).filter_map(move |bit| {
                    if (combined >> bit) & 1 != 0 {
                        Some(word_idx * 64 + bit)
                    } else {
                        None
                    }
                })
            })
    }
}

// ============================================================================
// QubitState Enum
// ============================================================================

/// Activity state of a qubit in the noise system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QubitState {
    /// Qubit has been prepared and is available for gates.
    Active,
    /// Qubit was measured and is now inactive (can be re-prepared).
    Inactive,
    /// Qubit is outside the computational subspace.
    Leaked,
}

// ============================================================================
// GateInfo - Gate context for dynamic noise parameters
// ============================================================================

/// Information about the current gate being processed.
///
/// This struct is used to pass gate context to composite primitives that need
/// dynamic parameters (e.g., angle-dependent error rates).
#[derive(Debug, Clone)]
pub struct GateInfo {
    /// The type of gate being executed.
    pub gate_type: GateType,
    /// Angle parameters (if any).
    pub angles: SmallVec<[Angle64; 2]>,
    /// Number of qubits involved.
    pub num_qubits: usize,
}

impl GateInfo {
    /// Create new gate info.
    #[must_use]
    pub fn new(gate_type: GateType, angles: SmallVec<[Angle64; 2]>, num_qubits: usize) -> Self {
        Self {
            gate_type,
            angles,
            num_qubits,
        }
    }

    /// Get the first angle if present.
    #[must_use]
    pub fn angle(&self) -> Option<Angle64> {
        self.angles.first().copied()
    }

    /// Check if this is a two-qubit gate.
    #[must_use]
    pub fn is_two_qubit(&self) -> bool {
        self.num_qubits == 2
    }
}

// ============================================================================
// IdleInfo - Idle event context for time-dependent noise
// ============================================================================

/// Information about the current idle event being processed.
///
/// This struct is used to pass idle duration to composite primitives that need
/// time-dependent parameters (e.g., T1/T2 decay rates).
#[derive(Debug, Clone, Copy)]
pub struct IdleInfo {
    /// Duration of the idle period in abstract time units.
    pub duration: TimeUnits,
}

impl IdleInfo {
    /// Create new idle info.
    #[must_use]
    pub fn new(duration: TimeUnits) -> Self {
        Self { duration }
    }

    /// Get the duration as a raw f64 value.
    #[must_use]
    pub fn duration_f64(&self) -> f64 {
        self.duration.as_f64()
    }
}

// ============================================================================
// NoiseContext - Main context using BitVec
// ============================================================================

/// Shared mutable state for noise channels.
///
/// This context is passed to each noise channel when processing events,
/// allowing channels to share state like leakage tracking and qubit activity.
///
/// The context tracks several pieces of information:
/// 1. Which qubits exist (have ever been prepared)
/// 2. Which qubits are currently active (prepared but not measured)
/// 3. Which qubits are leaked (outside computational subspace)
/// 4. Which gate types are noiseless (no noise applied)
///
/// ## Performance
///
/// Uses bit vectors for O(1) single-qubit operations, enabling efficient
/// simulation of 1M+ qubits. See the `large_scale` example for benchmarks.
#[derive(Debug, Clone)]
pub struct NoiseContext {
    /// Bit vector: bit i is set if qubit i is active.
    active: BitVec,
    /// Bit vector: bit i is set if qubit i is leaked.
    leaked: BitVec,
    /// Bit vector: bit i is set if qubit i has ever been prepared.
    exists: BitVec,
    /// Cached count of active qubits.
    active_count: usize,
    /// Cached count of leaked qubits.
    leaked_count: usize,
    /// Gate types that should not have noise applied.
    /// Uses `BTreeSet` since this is cold data (queried rarely, small set).
    pub noiseless_gates: BTreeSet<GateType>,
    /// Current measurement outcome for the qubit being processed.
    /// Set temporarily during measurement noise processing.
    current_outcome: Option<bool>,
    /// Current gate information for dynamic noise parameters.
    /// Set temporarily during gate noise processing.
    current_gate: Option<GateInfo>,
    /// Current idle information for time-dependent noise parameters.
    /// Set temporarily during idle noise processing.
    current_idle: Option<IdleInfo>,
    /// Current qubit index within a multi-qubit gate (0, 1, ...).
    /// Used for correlated noise actions.
    current_qubit_index: usize,
    /// Sampled correlated value for two-qubit actions.
    /// Actions can store a sampled value here on the first qubit
    /// and retrieve it on the second qubit.
    sampled_correlation: Option<usize>,
    /// The qubits involved in the current gate.
    current_gate_qubits: SmallVec<[QubitId; 2]>,
    /// Per-qubit "fired" flags for two-stage processing.
    /// Index corresponds to qubit position in gate (0, 1, ...).
    fired_flags: [bool; 4],
    /// Optional gate definitions for category/spec lookups.
    /// When set, channels can query gate metadata via `category()`, `spec()`, etc.
    gate_definitions: Option<GateDefinitions>,
}

impl Default for NoiseContext {
    fn default() -> Self {
        Self::new()
    }
}

impl NoiseContext {
    /// Default initial capacity (can grow dynamically).
    const DEFAULT_CAPACITY: usize = 1024;

    /// Create a new empty noise context.
    #[must_use]
    pub fn new() -> Self {
        Self::with_capacity(Self::DEFAULT_CAPACITY)
    }

    /// Create a new noise context with the given initial capacity.
    ///
    /// The context can grow beyond this capacity, but pre-allocating
    /// for large simulations avoids reallocation during execution.
    #[must_use]
    pub fn with_capacity(num_qubits: usize) -> Self {
        Self {
            active: BitVec::with_capacity(num_qubits),
            leaked: BitVec::with_capacity(num_qubits),
            exists: BitVec::with_capacity(num_qubits),
            active_count: 0,
            leaked_count: 0,
            noiseless_gates: BTreeSet::new(),
            current_outcome: None,
            current_gate: None,
            current_idle: None,
            current_qubit_index: 0,
            sampled_correlation: None,
            current_gate_qubits: SmallVec::new(),
            fired_flags: [false; 4],
            gate_definitions: None,
        }
    }

    // ========================================================================
    // Leakage Tracking
    // ========================================================================

    /// Mark a qubit as leaked.
    pub fn mark_leaked(&mut self, qubit: QubitId) {
        if !self.leaked.get(qubit.0) {
            self.leaked.set(qubit.0);
            self.leaked_count += 1;
        }
    }

    /// Mark a qubit as no longer leaked (seepage).
    pub fn mark_unleaked(&mut self, qubit: QubitId) {
        if self.leaked.get(qubit.0) {
            self.leaked.clear(qubit.0);
            self.leaked_count -= 1;
        }
    }

    /// Check if a qubit is currently leaked. O(1).
    #[inline]
    #[must_use]
    pub fn is_leaked(&self, qubit: QubitId) -> bool {
        self.leaked.get(qubit.0)
    }

    /// Get the number of leaked qubits. O(1).
    #[must_use]
    pub fn leaked_count(&self) -> usize {
        self.leaked_count
    }

    /// Check if any qubit in the slice is leaked.
    ///
    /// This is optimized for the common case where no qubits are leaked:
    /// - If `leaked_count == 0`, returns `false` immediately (O(1))
    /// - Otherwise, checks each qubit in the slice
    ///
    /// For gate operations, this is typically faster than calling `is_leaked()`
    /// on each qubit individually since most simulations have few leaked qubits.
    #[inline]
    #[must_use]
    pub fn any_leaked(&self, qubits: &[QubitId]) -> bool {
        // Fast path: if no qubits are leaked at all, return immediately
        if self.leaked_count == 0 {
            return false;
        }
        // Slow path: check each qubit
        qubits.iter().any(|q| self.leaked.get(q.0))
    }

    // ========================================================================
    // Measurement Outcome Tracking (for composite-based noise)
    // ========================================================================

    /// Set the current measurement outcome for noise processing.
    ///
    /// This is used by the composite noise system to pass outcome information
    /// to primitives during measurement noise application.
    pub fn set_current_outcome(&mut self, outcome: bool) {
        self.current_outcome = Some(outcome);
    }

    /// Clear the current measurement outcome.
    pub fn clear_current_outcome(&mut self) {
        self.current_outcome = None;
    }

    /// Get the current measurement outcome.
    ///
    /// Returns `None` if not currently processing a measurement event.
    #[must_use]
    pub fn current_outcome(&self) -> Option<bool> {
        self.current_outcome
    }

    // ========================================================================
    // Gate Context Tracking (for composite-based noise with dynamic parameters)
    // ========================================================================

    /// Set the current gate information for noise processing.
    ///
    /// This is used by the composite noise system to pass gate information
    /// to primitives that need dynamic parameters (e.g., angle-dependent error rates).
    pub fn set_current_gate(&mut self, gate_type: GateType, angles: &[Angle64], num_qubits: usize) {
        self.current_gate = Some(GateInfo::new(
            gate_type,
            angles.iter().copied().collect(),
            num_qubits,
        ));
    }

    /// Clear the current gate information.
    pub fn clear_current_gate(&mut self) {
        self.current_gate = None;
    }

    /// Get the current gate information.
    ///
    /// Returns `None` if not currently processing a gate event.
    #[must_use]
    pub fn current_gate(&self) -> Option<&GateInfo> {
        self.current_gate.as_ref()
    }

    // ========================================================================
    // Idle Context Tracking (for composite-based noise with time-dependent parameters)
    // ========================================================================

    /// Set the current idle information for noise processing.
    ///
    /// This is used by the composite noise system to pass duration information
    /// to primitives that need time-dependent parameters (e.g., T1/T2 decay).
    pub fn set_current_idle(&mut self, duration: TimeUnits) {
        self.current_idle = Some(IdleInfo::new(duration));
    }

    /// Clear the current idle information.
    pub fn clear_current_idle(&mut self) {
        self.current_idle = None;
    }

    /// Get the current idle information.
    ///
    /// Returns `None` if not currently processing an idle event.
    #[must_use]
    pub fn current_idle(&self) -> Option<&IdleInfo> {
        self.current_idle.as_ref()
    }

    // ========================================================================
    // Correlated Noise Support (for two-qubit gates)
    // ========================================================================

    /// Set the current qubit index and gate qubits for correlated noise.
    ///
    /// This is called by `CompositeChannel` when processing multi-qubit gates,
    /// allowing actions to know which qubit they're processing and access
    /// the other qubits involved.
    pub fn set_current_qubit_index(&mut self, index: usize, gate_qubits: &[QubitId]) {
        self.current_qubit_index = index;
        self.current_gate_qubits = gate_qubits.iter().copied().collect();
    }

    /// Get the current qubit index within a multi-qubit gate.
    #[must_use]
    pub fn current_qubit_index(&self) -> usize {
        self.current_qubit_index
    }

    /// Get the qubits involved in the current gate.
    #[must_use]
    pub fn current_gate_qubits(&self) -> &[QubitId] {
        &self.current_gate_qubits
    }

    /// Get the other qubit in a two-qubit gate.
    ///
    /// Returns `None` if not a two-qubit gate or if index is invalid.
    #[must_use]
    pub fn other_qubit(&self) -> Option<QubitId> {
        if self.current_gate_qubits.len() != 2 {
            return None;
        }
        let other_index = 1 - self.current_qubit_index;
        self.current_gate_qubits.get(other_index).copied()
    }

    /// Store a sampled correlation value.
    ///
    /// Used by correlated actions to sample once on the first qubit
    /// and retrieve on the second qubit.
    pub fn set_sampled_correlation(&mut self, value: usize) {
        self.sampled_correlation = Some(value);
    }

    /// Get the stored sampled correlation value.
    #[must_use]
    pub fn sampled_correlation(&self) -> Option<usize> {
        self.sampled_correlation
    }

    /// Clear correlated noise state.
    pub fn clear_correlation(&mut self) {
        self.current_qubit_index = 0;
        self.sampled_correlation = None;
        self.current_gate_qubits.clear();
        self.fired_flags = [false; 4];
    }

    // ========================================================================
    // Two-Stage Fired Flags (for correlated effects)
    // ========================================================================

    /// Set the fired flag for the current qubit index.
    ///
    /// This is used in two-stage processing where stage 1 samples whether
    /// each qubit "fires" (e.g., emits/leaks), and stage 2 applies effects
    /// based on the cross-conditions (e.g., partner depolarizing).
    pub fn set_fired(&mut self, index: usize, fired: bool) {
        if index < self.fired_flags.len() {
            self.fired_flags[index] = fired;
        }
    }

    /// Check if the qubit at the given index fired.
    #[must_use]
    pub fn is_fired(&self, index: usize) -> bool {
        index < self.fired_flags.len() && self.fired_flags[index]
    }

    /// Check if the current qubit fired.
    #[must_use]
    pub fn current_qubit_fired(&self) -> bool {
        self.is_fired(self.current_qubit_index)
    }

    /// Check if the partner qubit fired (in a two-qubit gate).
    ///
    /// Returns `false` if not a two-qubit gate.
    #[must_use]
    pub fn partner_fired(&self) -> bool {
        if self.current_gate_qubits.len() != 2 {
            return false;
        }
        let partner_index = 1 - self.current_qubit_index;
        self.is_fired(partner_index)
    }

    /// Clear all fired flags.
    pub fn clear_fired_flags(&mut self) {
        self.fired_flags = [false; 4];
    }

    // ========================================================================
    // Qubit Activity Tracking
    // ========================================================================

    /// Mark a qubit as prepared (exists and is active in the system).
    ///
    /// This also clears any leakage state for the qubit.
    pub fn mark_prepared(&mut self, qubit: QubitId) {
        if !self.exists.get(qubit.0) {
            self.exists.set(qubit.0);
        }
        if !self.active.get(qubit.0) {
            self.active.set(qubit.0);
            self.active_count += 1;
        }
        if self.leaked.get(qubit.0) {
            self.leaked.clear(qubit.0);
            self.leaked_count -= 1;
        }
    }

    /// Mark a qubit as measured (becomes inactive but still exists).
    ///
    /// Note: This does NOT clear leakage. Measuring a leaked qubit
    /// returns a special value but the qubit remains leaked until re-prepared.
    pub fn mark_measured(&mut self, qubit: QubitId) {
        if self.active.get(qubit.0) {
            self.active.clear(qubit.0);
            self.active_count -= 1;
        }
    }

    /// Check if a qubit is currently active (prepared but not measured). O(1).
    #[inline]
    #[must_use]
    pub fn is_active(&self, qubit: QubitId) -> bool {
        self.active.get(qubit.0)
    }

    /// Check if a qubit exists (has ever been prepared). O(1).
    #[inline]
    #[must_use]
    pub fn exists(&self, qubit: QubitId) -> bool {
        self.exists.get(qubit.0)
    }

    /// Get the number of active qubits. O(1).
    #[must_use]
    pub fn active_count(&self) -> usize {
        self.active_count
    }

    /// Get the state of a qubit.
    #[must_use]
    pub fn qubit_state(&self, qubit: QubitId) -> Option<QubitState> {
        if !self.exists(qubit) {
            return None;
        }
        if self.is_leaked(qubit) {
            Some(QubitState::Leaked)
        } else if self.is_active(qubit) {
            Some(QubitState::Active)
        } else {
            Some(QubitState::Inactive)
        }
    }

    // ========================================================================
    // Crosstalk Support
    // ========================================================================

    /// Get all prepared qubits except those in the given set.
    ///
    /// Useful for global crosstalk calculations.
    #[must_use]
    pub fn other_qubits(&self, exclude: &[QubitId]) -> Vec<QubitId> {
        self.exists
            .iter_ones()
            .filter(|&idx| !exclude.iter().any(|q| q.0 == idx))
            .map(QubitId)
            .collect()
    }

    /// Get all active qubits except those in the given set.
    ///
    /// Useful for crosstalk - only active qubits should receive crosstalk errors.
    #[must_use]
    pub fn other_active_qubits(&self, exclude: &[QubitId]) -> Vec<QubitId> {
        self.active
            .iter_ones()
            .filter(|&idx| !exclude.iter().any(|q| q.0 == idx))
            .map(QubitId)
            .collect()
    }

    /// Get all active, non-leaked qubits except those in the given set.
    ///
    /// This is the typical set of qubits that should receive crosstalk errors.
    #[must_use]
    pub fn crosstalk_targets(&self, exclude: &[QubitId]) -> Vec<QubitId> {
        self.active
            .and_not_iter(&self.leaked)
            .filter(|&idx| !exclude.iter().any(|q| q.0 == idx))
            .map(QubitId)
            .collect()
    }

    /// Get crosstalk targets for global crosstalk (all active qubits not in gate).
    #[must_use]
    pub fn global_crosstalk_targets(&self, gated_qubits: &[QubitId]) -> Vec<QubitId> {
        self.crosstalk_targets(gated_qubits)
    }

    /// Get crosstalk targets for local crosstalk (only specified neighbors).
    ///
    /// The `neighbors` are typically provided by the device topology.
    #[must_use]
    pub fn local_crosstalk_targets(
        &self,
        gated_qubits: &[QubitId],
        neighbors: &[QubitId],
    ) -> Vec<QubitId> {
        neighbors
            .iter()
            .filter(|q| !gated_qubits.contains(q) && self.is_active(**q) && !self.is_leaked(**q))
            .copied()
            .collect()
    }

    // ========================================================================
    // Noiseless Gates
    // ========================================================================

    /// Add a gate type to the noiseless set.
    ///
    /// Gates in this set will not have noise applied to them.
    pub fn add_noiseless_gate(&mut self, gate_type: GateType) {
        self.noiseless_gates.insert(gate_type);
    }

    /// Remove a gate type from the noiseless set.
    pub fn remove_noiseless_gate(&mut self, gate_type: GateType) {
        self.noiseless_gates.remove(&gate_type);
    }

    /// Check if a gate type is noiseless.
    #[must_use]
    pub fn is_noiseless(&self, gate_type: GateType) -> bool {
        self.noiseless_gates.contains(&gate_type)
    }

    /// Clear all noiseless gates.
    pub fn clear_noiseless_gates(&mut self) {
        self.noiseless_gates.clear();
    }

    // ========================================================================
    // Reset
    // ========================================================================

    /// Reset the context for a new shot.
    ///
    /// This clears per-shot state (leaked qubits, prepared qubits, active qubits)
    /// but preserves configuration (noiseless gates, gate definitions).
    pub fn reset(&mut self) {
        self.active.clear_all();
        self.leaked.clear_all();
        self.exists.clear_all();
        self.active_count = 0;
        self.leaked_count = 0;
        self.current_outcome = None;
        self.current_gate = None;
        self.current_idle = None;
        self.fired_flags = [false; 4];
        // Note: noiseless_gates and gate_definitions are configuration, not per-shot state
    }

    // ========================================================================
    // Gate Definitions
    // ========================================================================

    /// Set gate definitions for this context.
    ///
    /// When set, channels can query gate metadata via `category()`, `spec()`, etc.
    /// This enables category-based noise filtering and uniform treatment of
    /// core and custom gates.
    pub fn set_gate_definitions(&mut self, defs: GateDefinitions) {
        self.gate_definitions = Some(defs);
    }

    /// Get gate definitions if set.
    #[must_use]
    pub fn gate_definitions(&self) -> Option<&GateDefinitions> {
        self.gate_definitions.as_ref()
    }

    /// Get the category of a gate by its ID. O(1).
    ///
    /// Returns `None` if gate definitions are not set or the gate is unknown.
    #[must_use]
    pub fn category(&self, gate_id: GateId) -> Option<GateCategory> {
        self.gate_definitions
            .as_ref()
            .and_then(|d| d.category(gate_id))
    }

    /// Get the spec of a gate by its ID. O(1).
    ///
    /// Returns `None` if gate definitions are not set or the gate is unknown.
    #[must_use]
    pub fn gate_spec(&self, gate_id: GateId) -> Option<&GateSpec> {
        self.gate_definitions.as_ref().and_then(|d| d.spec(gate_id))
    }

    /// Get the quantum arity of a gate by its ID. O(1).
    ///
    /// Returns `None` if gate definitions are not set or the gate is unknown.
    #[must_use]
    pub fn quantum_arity(&self, gate_id: GateId) -> Option<u8> {
        self.gate_definitions
            .as_ref()
            .and_then(|d| d.quantum_arity(gate_id))
    }

    /// Check if a gate is single-qubit by its ID. O(1).
    #[must_use]
    pub fn is_single_qubit_gate(&self, gate_id: GateId) -> bool {
        self.quantum_arity(gate_id) == Some(1)
    }

    /// Check if a gate is two-qubit by its ID. O(1).
    #[must_use]
    pub fn is_two_qubit_gate(&self, gate_id: GateId) -> bool {
        self.quantum_arity(gate_id) == Some(2)
    }

    /// Get the error probability for a gate from definitions. O(1).
    ///
    /// Returns 0.0 if gate definitions are not set or the gate has no noise config.
    #[must_use]
    pub fn gate_error_probability(&self, gate_id: GateId) -> f64 {
        self.gate_definitions
            .as_ref()
            .map_or(0.0, |d| d.error_probability(gate_id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bitvec_basic() {
        let mut bv = BitVec::with_capacity(100);

        assert!(!bv.get(0));
        assert!(!bv.get(50));
        assert!(!bv.get(99));

        bv.set(50);
        assert!(bv.get(50));
        assert!(!bv.get(49));
        assert!(!bv.get(51));

        bv.clear(50);
        assert!(!bv.get(50));
    }

    #[test]
    fn test_bitvec_count() {
        let mut bv = BitVec::with_capacity(1000);

        bv.set(10);
        bv.set(100);
        bv.set(500);
        bv.set(999);

        assert_eq!(bv.count_ones(), 4);
    }

    #[test]
    fn test_bitvec_iter_ones() {
        let mut bv = BitVec::with_capacity(200);

        bv.set(5);
        bv.set(64);
        bv.set(65);
        bv.set(128);

        let ones: Vec<_> = bv.iter_ones().collect();
        assert_eq!(ones, vec![5, 64, 65, 128]);
    }

    #[test]
    fn test_leakage_tracking() {
        let mut ctx = NoiseContext::new();

        assert!(!ctx.is_leaked(QubitId(0)));
        assert_eq!(ctx.leaked_count(), 0);

        ctx.mark_leaked(QubitId(0));
        assert!(ctx.is_leaked(QubitId(0)));
        assert_eq!(ctx.leaked_count(), 1);

        ctx.mark_unleaked(QubitId(0));
        assert!(!ctx.is_leaked(QubitId(0)));
        assert_eq!(ctx.leaked_count(), 0);
    }

    #[test]
    fn test_any_leaked() {
        let mut ctx = NoiseContext::new();

        // No leakage - any_leaked should return false immediately (fast path)
        let qubits = [QubitId(0), QubitId(1), QubitId(2)];
        assert!(!ctx.any_leaked(&qubits));
        assert!(!ctx.any_leaked(&[]));

        // Leak one qubit
        ctx.mark_leaked(QubitId(1));

        // Now any_leaked should detect it
        assert!(ctx.any_leaked(&qubits));
        assert!(ctx.any_leaked(&[QubitId(1)]));
        assert!(!ctx.any_leaked(&[QubitId(0), QubitId(2)]));

        // Unleak the qubit - back to fast path
        ctx.mark_unleaked(QubitId(1));
        assert!(!ctx.any_leaked(&qubits));
    }

    #[test]
    fn test_preparation_clears_leakage() {
        let mut ctx = NoiseContext::new();

        ctx.mark_leaked(QubitId(0));
        assert!(ctx.is_leaked(QubitId(0)));

        ctx.mark_prepared(QubitId(0));
        assert!(!ctx.is_leaked(QubitId(0)));
        assert!(ctx.exists(QubitId(0)));
    }

    #[test]
    fn test_other_qubits() {
        let mut ctx = NoiseContext::new();
        ctx.mark_prepared(QubitId(0));
        ctx.mark_prepared(QubitId(1));
        ctx.mark_prepared(QubitId(2));

        let others = ctx.other_qubits(&[QubitId(1)]);
        assert!(others.contains(&QubitId(0)));
        assert!(!others.contains(&QubitId(1)));
        assert!(others.contains(&QubitId(2)));
    }

    #[test]
    fn test_qubit_lifecycle() {
        let mut ctx = NoiseContext::new();

        // Initially, qubit doesn't exist
        assert!(!ctx.exists(QubitId(0)));
        assert!(ctx.qubit_state(QubitId(0)).is_none());

        // After preparation, qubit is active
        ctx.mark_prepared(QubitId(0));
        assert!(ctx.exists(QubitId(0)));
        assert!(ctx.is_active(QubitId(0)));
        assert_eq!(ctx.qubit_state(QubitId(0)), Some(QubitState::Active));

        // After measurement, qubit is inactive
        ctx.mark_measured(QubitId(0));
        assert!(ctx.exists(QubitId(0)));
        assert!(!ctx.is_active(QubitId(0)));
        assert_eq!(ctx.qubit_state(QubitId(0)), Some(QubitState::Inactive));

        // Re-preparation makes it active again
        ctx.mark_prepared(QubitId(0));
        assert!(ctx.is_active(QubitId(0)));
        assert_eq!(ctx.qubit_state(QubitId(0)), Some(QubitState::Active));

        // Leakage changes state
        ctx.mark_leaked(QubitId(0));
        assert_eq!(ctx.qubit_state(QubitId(0)), Some(QubitState::Leaked));
    }

    #[test]
    fn test_crosstalk_targets() {
        let mut ctx = NoiseContext::new();

        // Prepare some qubits
        ctx.mark_prepared(QubitId(0));
        ctx.mark_prepared(QubitId(1));
        ctx.mark_prepared(QubitId(2));
        ctx.mark_prepared(QubitId(3));

        // All are active initially
        let targets = ctx.global_crosstalk_targets(&[QubitId(0)]);
        assert_eq!(targets.len(), 3);
        assert!(!targets.contains(&QubitId(0)));

        // Measure one qubit - no longer a crosstalk target
        ctx.mark_measured(QubitId(1));
        let targets = ctx.global_crosstalk_targets(&[QubitId(0)]);
        assert_eq!(targets.len(), 2);
        assert!(!targets.contains(&QubitId(1)));

        // Leak one qubit - no longer a crosstalk target
        ctx.mark_leaked(QubitId(2));
        let targets = ctx.global_crosstalk_targets(&[QubitId(0)]);
        assert_eq!(targets.len(), 1);
        assert!(!targets.contains(&QubitId(2)));
        assert!(targets.contains(&QubitId(3)));
    }

    #[test]
    fn test_local_crosstalk_targets() {
        let mut ctx = NoiseContext::new();

        ctx.mark_prepared(QubitId(0));
        ctx.mark_prepared(QubitId(1));
        ctx.mark_prepared(QubitId(2));
        ctx.mark_prepared(QubitId(3));

        // Only neighbors 1 and 2 are considered for local crosstalk
        let neighbors = &[QubitId(1), QubitId(2)];
        let gated = &[QubitId(0)];

        let targets = ctx.local_crosstalk_targets(gated, neighbors);
        assert_eq!(targets.len(), 2);
        assert!(targets.contains(&QubitId(1)));
        assert!(targets.contains(&QubitId(2)));
        assert!(!targets.contains(&QubitId(3))); // Not a neighbor

        // Measure qubit 1 - no longer a target
        ctx.mark_measured(QubitId(1));
        let targets = ctx.local_crosstalk_targets(gated, neighbors);
        assert_eq!(targets.len(), 1);
        assert!(!targets.contains(&QubitId(1)));
    }

    #[test]
    fn test_noiseless_gates() {
        let mut ctx = NoiseContext::new();

        // Initially no noiseless gates
        assert!(!ctx.is_noiseless(GateType::H));
        assert!(!ctx.is_noiseless(GateType::CX));

        // Add H as noiseless
        ctx.add_noiseless_gate(GateType::H);
        assert!(ctx.is_noiseless(GateType::H));
        assert!(!ctx.is_noiseless(GateType::CX));

        // Remove H
        ctx.remove_noiseless_gate(GateType::H);
        assert!(!ctx.is_noiseless(GateType::H));

        // Add multiple, then clear
        ctx.add_noiseless_gate(GateType::H);
        ctx.add_noiseless_gate(GateType::CX);
        ctx.clear_noiseless_gates();
        assert!(!ctx.is_noiseless(GateType::H));
        assert!(!ctx.is_noiseless(GateType::CX));
    }

    #[test]
    fn test_reset_preserves_noiseless_gates() {
        let mut ctx = NoiseContext::new();

        // Add noiseless gate and some state
        ctx.add_noiseless_gate(GateType::H);
        ctx.mark_prepared(QubitId(0));
        ctx.mark_leaked(QubitId(1));

        // Reset should clear state but keep noiseless gates
        ctx.reset();

        assert!(ctx.is_noiseless(GateType::H)); // Preserved
        assert!(!ctx.exists(QubitId(0))); // Cleared
        assert!(!ctx.is_leaked(QubitId(1))); // Cleared
    }

    #[test]
    fn test_large_scale() {
        // Test with 10000 qubits
        let mut ctx = NoiseContext::with_capacity(10_000);

        // Prepare all
        for i in 0..10_000 {
            ctx.mark_prepared(QubitId(i));
        }
        assert_eq!(ctx.active_count(), 10_000);

        // Leak every 100th
        for i in (0..10_000).step_by(100) {
            ctx.mark_leaked(QubitId(i));
        }
        assert_eq!(ctx.leaked_count(), 100);

        // Crosstalk targets should exclude leaked
        let targets = ctx.crosstalk_targets(&[QubitId(50)]);
        assert_eq!(targets.len(), 10_000 - 100 - 1); // -100 leaked, -1 excluded
    }
}
