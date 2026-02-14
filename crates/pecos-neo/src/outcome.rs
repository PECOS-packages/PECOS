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

//! Measurement outcome types.
//!
//! This module provides typed representations of measurement outcomes,
//! replacing the generic result handling in `ByteMessage`.

use pecos_core::QubitId;
use std::collections::BTreeMap;

/// A single measurement outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct MeasurementOutcome {
    /// The qubit that was measured.
    pub qubit: QubitId,

    /// The measurement result (true = 1, false = 0).
    ///
    /// For leaked qubits measured with `MeasureLeaked`, this is meaningless
    /// and `is_leaked` will be true.
    pub outcome: bool,

    /// Whether the outcome was deterministic (state was in an eigenstate).
    pub is_deterministic: bool,

    /// Whether the qubit was in a leaked state when measured.
    ///
    /// For `MeasureLeaked` gate type, leaked qubits return value 2.
    /// For regular `MZ`, leaked qubits return value 1.
    pub is_leaked: bool,
}

impl MeasurementOutcome {
    /// Create a new measurement outcome.
    #[must_use]
    pub fn new(qubit: QubitId, outcome: bool, is_deterministic: bool) -> Self {
        Self {
            qubit,
            outcome,
            is_deterministic,
            is_leaked: false,
        }
    }

    /// Create a random (non-deterministic) measurement outcome.
    #[must_use]
    pub fn random(qubit: QubitId, outcome: bool) -> Self {
        Self::new(qubit, outcome, false)
    }

    /// Create a deterministic measurement outcome.
    #[must_use]
    pub fn deterministic(qubit: QubitId, outcome: bool) -> Self {
        Self::new(qubit, outcome, true)
    }

    /// Create an outcome for a leaked qubit.
    #[must_use]
    pub fn leaked(qubit: QubitId) -> Self {
        Self {
            qubit,
            outcome: true, // Leaked qubits return 1 for regular Measure
            is_deterministic: true,
            is_leaked: true,
        }
    }

    /// Get the outcome as an integer (0, 1, or 2 for leaked).
    ///
    /// For regular measurements: 0 or 1
    /// For `MeasureLeaked` on a leaked qubit: 2
    #[must_use]
    pub fn as_int(&self) -> u8 {
        u8::from(self.outcome)
    }

    /// Get the outcome as an integer for `MeasureLeaked` gate type.
    ///
    /// Returns 2 if the qubit was leaked, otherwise 0 or 1.
    #[must_use]
    pub fn as_int_leaked(&self) -> u8 {
        if self.is_leaked {
            2
        } else {
            u8::from(self.outcome)
        }
    }
}

/// Collection of measurement outcomes from a circuit execution.
///
/// Outcomes are stored in order of measurement and can also be accessed by qubit ID.
#[derive(Debug, Clone, Default)]
pub struct MeasurementOutcomes {
    /// All outcomes in order of measurement.
    outcomes: Vec<MeasurementOutcome>,

    /// Map from qubit ID to the index of its most recent outcome.
    /// This allows fast lookup of the last measurement result for a qubit.
    qubit_to_last_outcome: BTreeMap<QubitId, usize>,
}

impl MeasurementOutcomes {
    /// Create an empty outcome collection.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an outcome collection with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            outcomes: Vec::with_capacity(capacity),
            qubit_to_last_outcome: BTreeMap::new(),
        }
    }

    /// Record a measurement outcome.
    pub fn record(&mut self, outcome: MeasurementOutcome) {
        let index = self.outcomes.len();
        self.qubit_to_last_outcome.insert(outcome.qubit, index);
        self.outcomes.push(outcome);
    }

    /// Record a measurement outcome with the given values.
    pub fn record_outcome(&mut self, qubit: QubitId, outcome: bool, is_deterministic: bool) {
        self.record(MeasurementOutcome::new(qubit, outcome, is_deterministic));
    }

    /// Get the number of recorded outcomes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    /// Check if there are no recorded outcomes.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }

    /// Get all outcomes in order.
    #[must_use]
    pub fn as_slice(&self) -> &[MeasurementOutcome] {
        &self.outcomes
    }

    /// Iterate over all outcomes in order.
    pub fn iter(&self) -> impl Iterator<Item = &MeasurementOutcome> {
        self.outcomes.iter()
    }

    /// Get the most recent outcome for a specific qubit.
    #[must_use]
    pub fn get(&self, qubit: QubitId) -> Option<&MeasurementOutcome> {
        self.qubit_to_last_outcome
            .get(&qubit)
            .map(|&idx| &self.outcomes[idx])
    }

    /// Get all outcomes for a specific qubit (in case of multiple measurements).
    #[must_use]
    pub fn get_all(&self, qubit: QubitId) -> Vec<&MeasurementOutcome> {
        self.outcomes.iter().filter(|o| o.qubit == qubit).collect()
    }

    /// Get the outcome bit for a qubit, or None if not measured.
    #[must_use]
    pub fn get_bit(&self, qubit: QubitId) -> Option<bool> {
        self.get(qubit).map(|o| o.outcome)
    }

    /// Clear all recorded outcomes.
    pub fn clear(&mut self) {
        self.outcomes.clear();
        self.qubit_to_last_outcome.clear();
    }

    /// Get a bitstring representation of outcomes for the given qubits.
    ///
    /// Returns None if any qubit has not been measured.
    #[must_use]
    pub fn bitstring(&self, qubits: &[QubitId]) -> Option<Vec<bool>> {
        qubits.iter().map(|&q| self.get_bit(q)).collect()
    }

    /// Get outcomes as a map from qubit ID to outcome bit.
    #[must_use]
    pub fn as_map(&self) -> BTreeMap<QubitId, bool> {
        self.qubit_to_last_outcome
            .iter()
            .map(|(&q, &idx)| (q, self.outcomes[idx].outcome))
            .collect()
    }

    /// Flip the outcome for a specific qubit (for noise simulation).
    ///
    /// Returns true if the qubit was found and flipped, false otherwise.
    pub fn flip(&mut self, qubit: QubitId) -> bool {
        if let Some(&idx) = self.qubit_to_last_outcome.get(&qubit) {
            self.outcomes[idx].outcome = !self.outcomes[idx].outcome;
            true
        } else {
            false
        }
    }

    /// Mark the most recent outcome for a qubit as leaked (for noise simulation).
    ///
    /// This is used by `MeasureLeaked` to indicate that the qubit was in a leaked state.
    /// Returns true if the qubit was found, false otherwise.
    pub fn mark_leaked(&mut self, qubit: QubitId) -> bool {
        if let Some(&idx) = self.qubit_to_last_outcome.get(&qubit) {
            self.outcomes[idx].is_leaked = true;
            true
        } else {
            false
        }
    }

    /// Force the outcome for a specific qubit to a given value (for noise simulation).
    ///
    /// This sets the outcome to the specified value regardless of the actual measurement.
    /// Returns true if the qubit was found and set, false otherwise.
    pub fn set_outcome(&mut self, qubit: QubitId, value: bool) -> bool {
        if let Some(&idx) = self.qubit_to_last_outcome.get(&qubit) {
            self.outcomes[idx].outcome = value;
            true
        } else {
            false
        }
    }
}

impl FromIterator<MeasurementOutcome> for MeasurementOutcomes {
    fn from_iter<I: IntoIterator<Item = MeasurementOutcome>>(iter: I) -> Self {
        let mut outcomes = Self::new();
        for outcome in iter {
            outcomes.record(outcome);
        }
        outcomes
    }
}

// ============================================================================
// RegisterMap
// ============================================================================

/// Mapping from register names to qubit ranges.
///
/// This bridges qubit-indexed measurement outcomes to named registers,
/// as used in QASM-style programs where results are accessed by register name
/// (e.g., `c[0]`, `c[1]`).
///
/// # Example
///
/// ```
/// use pecos_neo::outcome::RegisterMap;
/// use pecos_core::QubitId;
///
/// let mut reg = RegisterMap::new();
/// reg.add_register("data", &[QubitId(0), QubitId(1)]);
/// reg.add_register("syndrome", &[QubitId(2), QubitId(3), QubitId(4)]);
///
/// assert_eq!(reg.get("data"), Some(&[QubitId(0), QubitId(1)][..]));
/// assert_eq!(reg.register_names().count(), 2);
/// ```
#[derive(Debug, Clone, Default)]
pub struct RegisterMap {
    /// Register name -> ordered list of `QubitId`s in that register.
    registers: BTreeMap<String, Vec<QubitId>>,
}

impl RegisterMap {
    /// Create an empty register map.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a register with the given name and qubit IDs.
    pub fn add_register(&mut self, name: impl Into<String>, qubits: &[QubitId]) {
        self.registers.insert(name.into(), qubits.to_vec());
    }

    /// Create from an iterator of (name, qubits) pairs.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::outcome::RegisterMap;
    /// use pecos_core::QubitId;
    ///
    /// let reg = RegisterMap::from_pairs([
    ///     ("q".to_string(), vec![QubitId(0), QubitId(1)]),
    ///     ("c".to_string(), vec![QubitId(2)]),
    /// ]);
    /// assert_eq!(reg.register_names().count(), 2);
    /// ```
    #[must_use]
    pub fn from_pairs(pairs: impl IntoIterator<Item = (String, Vec<QubitId>)>) -> Self {
        Self {
            registers: pairs.into_iter().collect(),
        }
    }

    /// Get the qubits for a named register.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&[QubitId]> {
        self.registers.get(name).map(Vec::as_slice)
    }

    /// Iterate over register names (sorted).
    pub fn register_names(&self) -> impl Iterator<Item = &str> {
        self.registers.keys().map(String::as_str)
    }

    /// Number of registers.
    #[must_use]
    pub fn len(&self) -> usize {
        self.registers.len()
    }

    /// Check if empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.registers.is_empty()
    }
}

impl MeasurementOutcomes {
    /// Get outcomes for a named register as a bitstring.
    ///
    /// Returns `None` if the register doesn't exist or any qubit hasn't been measured.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_neo::outcome::{MeasurementOutcomes, RegisterMap};
    /// use pecos_core::QubitId;
    ///
    /// let mut outcomes = MeasurementOutcomes::new();
    /// outcomes.record_outcome(QubitId(0), true, false);
    /// outcomes.record_outcome(QubitId(1), false, false);
    ///
    /// let mut reg = RegisterMap::new();
    /// reg.add_register("c", &[QubitId(0), QubitId(1)]);
    ///
    /// assert_eq!(outcomes.register_bitstring(&reg, "c"), Some(vec![true, false]));
    /// assert_eq!(outcomes.register_bitstring(&reg, "missing"), None);
    /// ```
    #[must_use]
    pub fn register_bitstring(&self, register: &RegisterMap, name: &str) -> Option<Vec<bool>> {
        let qubits = register.get(name)?;
        self.bitstring(qubits)
    }

    /// Get all registers as a map of name -> bitstring.
    ///
    /// Registers where any qubit hasn't been measured are omitted.
    #[must_use]
    pub fn as_register_map(&self, register: &RegisterMap) -> BTreeMap<String, Vec<bool>> {
        let mut result = BTreeMap::new();
        for name in register.register_names() {
            if let Some(bits) = self.register_bitstring(register, name) {
                result.insert(name.to_string(), bits);
            }
        }
        result
    }
}

impl<'a> IntoIterator for &'a MeasurementOutcomes {
    type Item = &'a MeasurementOutcome;
    type IntoIter = std::slice::Iter<'a, MeasurementOutcome>;

    fn into_iter(self) -> Self::IntoIter {
        self.outcomes.iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_measurement_outcome() {
        let outcome = MeasurementOutcome::new(QubitId(0), true, false);
        assert_eq!(outcome.qubit, QubitId(0));
        assert!(outcome.outcome);
        assert!(!outcome.is_deterministic);
        assert_eq!(outcome.as_int(), 1);

        let det = MeasurementOutcome::deterministic(QubitId(1), false);
        assert!(det.is_deterministic);
        assert!(!det.outcome);
        assert_eq!(det.as_int(), 0);
    }

    #[test]
    fn test_measurement_outcomes_collection() {
        let mut outcomes = MeasurementOutcomes::new();
        assert!(outcomes.is_empty());

        outcomes.record_outcome(QubitId(0), true, false);
        outcomes.record_outcome(QubitId(1), false, true);

        assert_eq!(outcomes.len(), 2);
        assert!(!outcomes.is_empty());

        assert_eq!(outcomes.get_bit(QubitId(0)), Some(true));
        assert_eq!(outcomes.get_bit(QubitId(1)), Some(false));
        assert_eq!(outcomes.get_bit(QubitId(2)), None);
    }

    #[test]
    fn test_multiple_measurements_same_qubit() {
        let mut outcomes = MeasurementOutcomes::new();

        outcomes.record_outcome(QubitId(0), false, false);
        outcomes.record_outcome(QubitId(0), true, false);

        // get() returns the most recent
        assert_eq!(outcomes.get_bit(QubitId(0)), Some(true));

        // get_all() returns all measurements
        let all = outcomes.get_all(QubitId(0));
        assert_eq!(all.len(), 2);
        assert!(!all[0].outcome);
        assert!(all[1].outcome);
    }

    #[test]
    fn test_bitstring() {
        let mut outcomes = MeasurementOutcomes::new();
        outcomes.record_outcome(QubitId(0), true, false);
        outcomes.record_outcome(QubitId(1), false, false);
        outcomes.record_outcome(QubitId(2), true, false);

        let bs = outcomes.bitstring(&[QubitId(0), QubitId(1), QubitId(2)]);
        assert_eq!(bs, Some(vec![true, false, true]));

        // Missing qubit returns None
        let bs = outcomes.bitstring(&[QubitId(0), QubitId(3)]);
        assert_eq!(bs, None);
    }

    #[test]
    fn test_flip() {
        let mut outcomes = MeasurementOutcomes::new();
        outcomes.record_outcome(QubitId(0), true, false);

        assert!(outcomes.flip(QubitId(0)));
        assert_eq!(outcomes.get_bit(QubitId(0)), Some(false));

        assert!(!outcomes.flip(QubitId(1))); // Not measured
    }

    #[test]
    fn test_set_outcome() {
        let mut outcomes = MeasurementOutcomes::new();
        outcomes.record_outcome(QubitId(0), true, false);
        outcomes.record_outcome(QubitId(1), false, false);

        // Force qubit 0 to false (was true)
        assert!(outcomes.set_outcome(QubitId(0), false));
        assert_eq!(outcomes.get_bit(QubitId(0)), Some(false));

        // Force qubit 1 to true (was false)
        assert!(outcomes.set_outcome(QubitId(1), true));
        assert_eq!(outcomes.get_bit(QubitId(1)), Some(true));

        // Setting to same value should work
        assert!(outcomes.set_outcome(QubitId(0), false));
        assert_eq!(outcomes.get_bit(QubitId(0)), Some(false));

        // Non-existent qubit returns false
        assert!(!outcomes.set_outcome(QubitId(2), true));
    }

    // ========================================================================
    // RegisterMap tests
    // ========================================================================

    #[test]
    fn test_register_map_basic() {
        let mut reg = RegisterMap::new();
        assert!(reg.is_empty());

        reg.add_register("data", &[QubitId(0), QubitId(1)]);
        reg.add_register("syndrome", &[QubitId(2), QubitId(3)]);

        assert_eq!(reg.len(), 2);
        assert!(!reg.is_empty());
        assert_eq!(reg.get("data"), Some(&[QubitId(0), QubitId(1)][..]));
        assert_eq!(reg.get("missing"), None);
    }

    #[test]
    fn test_register_map_from_pairs() {
        let reg = RegisterMap::from_pairs([
            ("a".to_string(), vec![QubitId(0)]),
            ("b".to_string(), vec![QubitId(1), QubitId(2)]),
        ]);

        assert_eq!(reg.len(), 2);
        assert_eq!(reg.get("a"), Some(&[QubitId(0)][..]));
        assert_eq!(reg.get("b"), Some(&[QubitId(1), QubitId(2)][..]));
    }

    #[test]
    fn test_register_bitstring() {
        let mut outcomes = MeasurementOutcomes::new();
        outcomes.record_outcome(QubitId(0), true, false);
        outcomes.record_outcome(QubitId(1), false, false);
        outcomes.record_outcome(QubitId(2), true, false);

        let mut reg = RegisterMap::new();
        reg.add_register("c", &[QubitId(0), QubitId(1)]);
        reg.add_register("d", &[QubitId(2)]);

        assert_eq!(
            outcomes.register_bitstring(&reg, "c"),
            Some(vec![true, false])
        );
        assert_eq!(outcomes.register_bitstring(&reg, "d"), Some(vec![true]));
        assert_eq!(outcomes.register_bitstring(&reg, "missing"), None);
    }

    #[test]
    fn test_as_register_map() {
        let mut outcomes = MeasurementOutcomes::new();
        outcomes.record_outcome(QubitId(0), true, false);
        outcomes.record_outcome(QubitId(1), false, false);

        let mut reg = RegisterMap::new();
        reg.add_register("c", &[QubitId(0), QubitId(1)]);
        reg.add_register("unmeasured", &[QubitId(5)]); // Not measured

        let map = outcomes.as_register_map(&reg);
        assert_eq!(map.len(), 1); // Only "c" should be present
        assert_eq!(map["c"], vec![true, false]);
    }
}
