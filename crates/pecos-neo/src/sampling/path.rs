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

//! Path-based exploration for programs with measurement-dependent branching.
//!
//! This module provides techniques for systematically exploring different
//! execution paths through quantum programs with classical control flow.
//!
//! ## Techniques
//!
//! - **Path Recording**: Record which measurement outcomes occurred during a run
//! - **Path Replay**: Re-run a program forcing specific measurement outcomes
//! - **Path Enumeration**: Systematically enumerate all paths up to a depth
//! - **Path-Weighted Analysis**: Compute statistics weighted by path probabilities
//!
//! ## When to Use
//!
//! These techniques are useful for:
//! - QEC circuits with syndrome measurement and feedback
//! - Programs where rare branches lead to interesting behavior
//! - Systematic analysis of all possible execution paths
//! - Debugging by replaying specific execution traces
//!
//! ## Example
//!
//! ```no_run
//! use pecos_neo::sampling::path::{MeasurementPath, PathExplorer, PathEnumerator};
//! use pecos_neo::prelude::*;
//! use pecos_qsim::SparseStab;
//!
//! let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();
//! let mut explorer = PathExplorer::new(SparseStab::new(1));
//!
//! // Record a path during execution
//! let result = explorer.run_and_record(&commands);
//!
//! // Enumerate all paths up to 3 measurements
//! for path in PathEnumerator::new(3) {
//!     let result = explorer.run_with_path(&commands, &path);
//!     let weight = path.probability();
//!     // ... accumulate weighted statistics
//! }
//! ```

use crate::command::{CommandQueue, GateCommand, GateType};
use crate::outcome::{MeasurementOutcome, MeasurementOutcomes};
use crate::sampling::weight::SampleWeight;
use pecos_core::QubitId;
use pecos_qsim::{CliffordGateable, ForcedMeasurement};
use pecos_rng::PecosRng;
use smallvec::SmallVec;

/// A single measurement outcome in a path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PathOutcome {
    /// The qubit that was measured.
    pub qubit: QubitId,
    /// The measurement outcome (true = 1, false = 0).
    pub outcome: bool,
    /// Whether this measurement was deterministic (eigenstate).
    pub is_deterministic: bool,
}

impl PathOutcome {
    /// Create a new path outcome.
    #[must_use]
    pub fn new(qubit: QubitId, outcome: bool, is_deterministic: bool) -> Self {
        Self {
            qubit,
            outcome,
            is_deterministic,
        }
    }
}

/// A sequence of measurement outcomes representing an execution path.
///
/// For stabilizer simulation:
/// - Deterministic measurements have probability 1 (eigenstate)
/// - Non-deterministic measurements have probability 0.5 each way
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct MeasurementPath {
    /// The sequence of measurement outcomes.
    outcomes: SmallVec<[PathOutcome; 16]>,
}

impl MeasurementPath {
    /// Create an empty path.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a path with pre-allocated capacity.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            outcomes: SmallVec::with_capacity(capacity),
        }
    }

    /// Add a measurement outcome to the path.
    pub fn push(&mut self, outcome: PathOutcome) {
        self.outcomes.push(outcome);
    }

    /// Add a measurement outcome (convenience method).
    pub fn record(&mut self, qubit: QubitId, outcome: bool, is_deterministic: bool) {
        self.push(PathOutcome::new(qubit, outcome, is_deterministic));
    }

    /// Get the number of measurements in this path.
    #[must_use]
    pub fn len(&self) -> usize {
        self.outcomes.len()
    }

    /// Check if the path is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.outcomes.is_empty()
    }

    /// Get the measurement outcome at a specific index.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&PathOutcome> {
        self.outcomes.get(index)
    }

    /// Iterate over the outcomes.
    pub fn iter(&self) -> impl Iterator<Item = &PathOutcome> {
        self.outcomes.iter()
    }

    /// Get the number of non-deterministic measurements.
    #[must_use]
    pub fn num_random_measurements(&self) -> usize {
        self.outcomes.iter().filter(|o| !o.is_deterministic).count()
    }

    /// Compute the probability of this path (for stabilizer simulation).
    ///
    /// For stabilizer states:
    /// - Deterministic measurements contribute factor 1
    /// - Non-deterministic measurements contribute factor 0.5
    ///
    /// Returns the probability as a `SampleWeight` for numerical stability.
    #[must_use]
    pub fn probability(&self) -> SampleWeight {
        let num_random = self.num_random_measurements();
        if num_random == 0 {
            SampleWeight::one()
        } else {
            // P = 0.5^num_random = 2^(-num_random)
            // log(P) = -num_random * log(2)
            let log_prob = -(num_random as f64) * std::f64::consts::LN_2;
            SampleWeight::from_log(log_prob)
        }
    }

    /// Get the probability as a f64 (may underflow for long paths).
    #[must_use]
    pub fn probability_f64(&self) -> f64 {
        self.probability().weight()
    }

    /// Create a path signature for hashing/comparison.
    ///
    /// This only includes non-deterministic outcomes since deterministic
    /// outcomes are fixed by the circuit structure.
    #[must_use]
    pub fn signature(&self) -> PathSignature {
        let bits: Vec<bool> = self
            .outcomes
            .iter()
            .filter(|o| !o.is_deterministic)
            .map(|o| o.outcome)
            .collect();
        PathSignature { bits }
    }

    /// Clear the path for reuse.
    pub fn clear(&mut self) {
        self.outcomes.clear();
    }
}

/// A compact signature of a path (only non-deterministic outcomes).
///
/// Two paths with the same signature took the same "random" branches,
/// even if deterministic measurements differed (which shouldn't happen
/// for the same circuit).
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PathSignature {
    bits: Vec<bool>,
}

impl PathSignature {
    /// Get the number of random decisions in this path.
    #[must_use]
    pub fn len(&self) -> usize {
        self.bits.len()
    }

    /// Check if empty (all measurements were deterministic).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.bits.is_empty()
    }

    /// Convert to a binary string for display.
    #[must_use]
    pub fn to_binary_string(&self) -> String {
        self.bits
            .iter()
            .map(|&b| if b { '1' } else { '0' })
            .collect()
    }
}

/// Iterator that enumerates all possible paths up to a given number of measurements.
///
/// This is useful for systematic exploration of all branches in a bounded program.
///
/// # Warning
///
/// The number of paths grows exponentially: 2^n for n non-deterministic measurements.
/// Use with caution for programs with many measurements.
pub struct PathEnumerator {
    max_measurements: usize,
    current: u64,
    max_value: u64,
}

impl PathEnumerator {
    /// Create a new enumerator for paths up to `max_measurements` non-deterministic outcomes.
    #[must_use]
    pub fn new(max_measurements: usize) -> Self {
        let max_value = if max_measurements >= 64 {
            u64::MAX
        } else {
            (1u64 << max_measurements) - 1
        };
        Self {
            max_measurements,
            current: 0,
            max_value,
        }
    }

    /// Get the total number of paths that will be enumerated.
    #[must_use]
    pub fn total_paths(&self) -> u64 {
        self.max_value + 1
    }
}

impl Iterator for PathEnumerator {
    type Item = EnumeratedPath;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current > self.max_value {
            return None;
        }

        let path = EnumeratedPath {
            bits: self.current,
            len: self.max_measurements,
        };
        self.current += 1;
        Some(path)
    }
}

/// A path from enumeration, represented compactly as bits.
#[derive(Debug, Clone, Copy)]
pub struct EnumeratedPath {
    bits: u64,
    len: usize,
}

impl EnumeratedPath {
    /// Create a new enumerated path with the given bits and length.
    ///
    /// The bits represent the outcomes of non-deterministic measurements:
    /// - bit 0 = first non-deterministic measurement
    /// - bit 1 = second non-deterministic measurement
    /// - etc.
    #[must_use]
    pub fn new(bits: u64, len: usize) -> Self {
        Self { bits, len }
    }

    /// Get the bit pattern as a u64 (for display/indexing).
    #[must_use]
    pub fn index(&self) -> u64 {
        self.bits
    }

    /// Get the length (number of non-deterministic measurements).
    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    /// Check if the path has no measurements.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Get the outcome for the i-th non-deterministic measurement.
    #[must_use]
    pub fn outcome(&self, index: usize) -> bool {
        (self.bits >> index) & 1 == 1
    }

    /// Get the probability of this path (0.5^len).
    #[must_use]
    pub fn probability(&self) -> f64 {
        0.5_f64.powi(self.len as i32)
    }

    /// Get the probability as a `SampleWeight`.
    #[must_use]
    pub fn probability_weight(&self) -> SampleWeight {
        let log_prob = -(self.len as f64) * std::f64::consts::LN_2;
        SampleWeight::from_log(log_prob)
    }

    /// Convert to binary string for display.
    #[must_use]
    pub fn to_binary_string(&self) -> String {
        (0..self.len)
            .map(|i| if self.outcome(i) { '1' } else { '0' })
            .collect()
    }
}

/// Result of running a program with path recording.
#[derive(Debug, Clone)]
pub struct PathRecordedResult {
    /// The measurement outcomes from the run.
    pub outcomes: MeasurementOutcomes,
    /// The path taken (sequence of measurement outcomes).
    pub path: MeasurementPath,
}

/// A runner that can record and replay execution paths.
///
/// This enables:
/// - Recording which path was taken during a run
/// - Replaying a specific path by forcing measurement outcomes
/// - Systematic enumeration of all paths
pub struct PathExplorer<S: CliffordGateable + ForcedMeasurement> {
    simulator: S,
    rng: PecosRng,
}

impl<S: CliffordGateable + ForcedMeasurement> PathExplorer<S> {
    /// Create a new path explorer with the given simulator.
    pub fn new(simulator: S) -> Self {
        Self {
            simulator,
            rng: PecosRng::seed_from_u64(0),
        }
    }

    /// Set the RNG seed.
    #[must_use]
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.rng = PecosRng::seed_from_u64(seed);
        self
    }

    /// Run a program and record the path taken.
    ///
    /// This executes the program normally (with random measurement outcomes)
    /// while recording which outcomes occurred.
    pub fn run_and_record(&mut self, commands: &CommandQueue) -> PathRecordedResult {
        self.simulator.reset();
        let mut outcomes = MeasurementOutcomes::new();
        let mut path = MeasurementPath::new();

        for command in commands {
            self.execute_command_recording(command, &mut outcomes, &mut path);
        }

        PathRecordedResult { outcomes, path }
    }

    /// Run a program following a specific path.
    ///
    /// This forces measurement outcomes to match the given path.
    /// If the program has more measurements than the path, additional
    /// measurements use the default outcome (false/0).
    ///
    /// Returns the outcomes and the actual path taken (which may differ
    /// from the input if some measurements were deterministic).
    pub fn run_with_path(
        &mut self,
        commands: &CommandQueue,
        forced_path: &EnumeratedPath,
    ) -> PathRecordedResult {
        self.simulator.reset();
        let mut outcomes = MeasurementOutcomes::new();
        let mut path = MeasurementPath::new();
        let mut path_index = 0;

        for command in commands {
            self.execute_command_with_path(
                command,
                &mut outcomes,
                &mut path,
                forced_path,
                &mut path_index,
            );
        }

        PathRecordedResult { outcomes, path }
    }

    /// Execute a command while recording the path.
    fn execute_command_recording(
        &mut self,
        command: &GateCommand,
        outcomes: &mut MeasurementOutcomes,
        path: &mut MeasurementPath,
    ) {
        match command.gate_type {
            GateType::PZ | GateType::QAlloc => {
                let qubits: Vec<QubitId> = command.qubits.iter().copied().collect();
                self.simulator.pz(&qubits);
            }

            GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree => {
                for &qubit in &command.qubits {
                    let result = self.simulator.mz(&[qubit]);
                    let r = &result[0];
                    outcomes.record(MeasurementOutcome::new(
                        qubit,
                        r.outcome,
                        r.is_deterministic,
                    ));
                    path.record(qubit, r.outcome, r.is_deterministic);
                }
            }

            _ => {
                self.execute_gate(command);
            }
        }
    }

    /// Execute a command with forced path outcomes.
    fn execute_command_with_path(
        &mut self,
        command: &GateCommand,
        outcomes: &mut MeasurementOutcomes,
        path: &mut MeasurementPath,
        forced_path: &EnumeratedPath,
        path_index: &mut usize,
    ) {
        match command.gate_type {
            GateType::PZ | GateType::QAlloc => {
                let qubits: Vec<QubitId> = command.qubits.iter().copied().collect();
                self.simulator.pz(&qubits);
            }

            GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree => {
                for &qubit in &command.qubits {
                    // Get the forced outcome for this measurement
                    let forced_outcome = if *path_index < forced_path.len {
                        forced_path.outcome(*path_index)
                    } else {
                        false // Default to 0 if path is exhausted
                    };

                    // Force the measurement to the desired outcome
                    // mz_forced handles both deterministic and non-deterministic cases:
                    // - If deterministic: returns the fixed outcome (ignores forced_outcome)
                    // - If non-deterministic: forces to forced_outcome
                    let result = self.simulator.mz_forced(qubit.index(), forced_outcome);

                    if !result.is_deterministic {
                        // Only consume a path bit for non-deterministic measurements
                        *path_index += 1;
                    }

                    outcomes.record(MeasurementOutcome::new(
                        qubit,
                        result.outcome,
                        result.is_deterministic,
                    ));
                    path.record(qubit, result.outcome, result.is_deterministic);
                }
            }

            _ => {
                self.execute_gate(command);
            }
        }
    }

    /// Execute a gate command.
    fn execute_gate(&mut self, command: &GateCommand) {
        let qubits: Vec<QubitId> = command.qubits.iter().copied().collect();

        match command.gate_type {
            GateType::I => {
                self.simulator.identity(&qubits);
            }
            GateType::X => {
                self.simulator.x(&qubits);
            }
            GateType::Y => {
                self.simulator.y(&qubits);
            }
            GateType::Z => {
                self.simulator.z(&qubits);
            }
            GateType::H => {
                self.simulator.h(&qubits);
            }
            GateType::SX => {
                self.simulator.sx(&qubits);
            }
            GateType::SXdg => {
                self.simulator.sxdg(&qubits);
            }
            GateType::SY => {
                self.simulator.sy(&qubits);
            }
            GateType::SYdg => {
                self.simulator.sydg(&qubits);
            }
            GateType::SZ => {
                self.simulator.sz(&qubits);
            }
            GateType::SZdg => {
                self.simulator.szdg(&qubits);
            }
            GateType::CX => {
                self.simulator.cx(&qubits);
            }
            GateType::CY => {
                self.simulator.cy(&qubits);
            }
            GateType::CZ => {
                self.simulator.cz(&qubits);
            }
            GateType::SZZ => {
                self.simulator.szz(&qubits);
            }
            GateType::SZZdg => {
                self.simulator.szzdg(&qubits);
            }
            GateType::SWAP => {
                self.simulator.swap(&qubits);
            }
            _ => {}
        }
    }
}

/// Statistics accumulated across multiple paths with weights.
#[derive(Debug, Clone, Default)]
pub struct PathStatistics {
    /// Sum of weighted values.
    weighted_sum: f64,
    /// Sum of weights.
    total_weight: f64,
    /// Number of paths explored.
    num_paths: usize,
    /// Number of paths that matched a predicate.
    num_matching: usize,
}

impl PathStatistics {
    /// Create new empty statistics.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a path result with its probability weight.
    pub fn add(&mut self, value: f64, weight: f64) {
        self.weighted_sum += value * weight;
        self.total_weight += weight;
        self.num_paths += 1;
        if value > 0.0 {
            self.num_matching += 1;
        }
    }

    /// Add a path result using `SampleWeight`.
    pub fn add_weighted(&mut self, value: f64, weight: &SampleWeight) {
        self.add(value, weight.weight());
    }

    /// Get the weighted mean.
    #[must_use]
    pub fn mean(&self) -> f64 {
        if self.total_weight > 0.0 {
            self.weighted_sum / self.total_weight
        } else {
            0.0
        }
    }

    /// Get the number of paths explored.
    #[must_use]
    pub fn num_paths(&self) -> usize {
        self.num_paths
    }

    /// Get the number of matching paths (value > 0).
    #[must_use]
    pub fn num_matching(&self) -> usize {
        self.num_matching
    }

    /// Get the total weight (should be ~1.0 if all paths enumerated).
    #[must_use]
    pub fn total_weight(&self) -> f64 {
        self.total_weight
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::command::CommandBuilder;
    use pecos_qsim::SparseStab;

    #[test]
    fn test_measurement_path_basic() {
        let mut path = MeasurementPath::new();
        assert!(path.is_empty());

        path.record(QubitId(0), false, true); // deterministic
        path.record(QubitId(1), true, false); // random
        path.record(QubitId(2), false, false); // random

        assert_eq!(path.len(), 3);
        assert_eq!(path.num_random_measurements(), 2);

        // Probability = 0.5^2 = 0.25
        assert!((path.probability_f64() - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_path_signature() {
        let mut path1 = MeasurementPath::new();
        path1.record(QubitId(0), false, true); // deterministic - ignored
        path1.record(QubitId(1), true, false); // random
        path1.record(QubitId(2), false, false); // random

        let mut path2 = MeasurementPath::new();
        path2.record(QubitId(0), true, true); // different deterministic - ignored
        path2.record(QubitId(1), true, false); // same random
        path2.record(QubitId(2), false, false); // same random

        // Signatures should match (only random outcomes matter)
        assert_eq!(path1.signature(), path2.signature());
        assert_eq!(path1.signature().to_binary_string(), "10");
    }

    #[test]
    fn test_path_enumerator() {
        let enumerator = PathEnumerator::new(3);
        assert_eq!(enumerator.total_paths(), 8);

        let paths: Vec<_> = PathEnumerator::new(3).collect();
        assert_eq!(paths.len(), 8);

        // Check all combinations are present
        let strings: Vec<_> = paths
            .iter()
            .map(super::EnumeratedPath::to_binary_string)
            .collect();
        assert!(strings.contains(&"000".to_string()));
        assert!(strings.contains(&"001".to_string()));
        assert!(strings.contains(&"010".to_string()));
        assert!(strings.contains(&"011".to_string()));
        assert!(strings.contains(&"100".to_string()));
        assert!(strings.contains(&"101".to_string()));
        assert!(strings.contains(&"110".to_string()));
        assert!(strings.contains(&"111".to_string()));
    }

    #[test]
    fn test_enumerated_path_probability() {
        let path = EnumeratedPath {
            bits: 0b101,
            len: 3,
        };
        assert!((path.probability() - 0.125).abs() < 1e-10); // 0.5^3

        assert!(path.outcome(0)); // bit 0 = 1
        assert!(!path.outcome(1)); // bit 1 = 0
        assert!(path.outcome(2)); // bit 2 = 1
    }

    #[test]
    fn test_path_explorer_record() {
        let commands = CommandBuilder::new()
            .pz(0)
            .h(0) // Creates superposition
            .mz(0)
            .build();

        let mut explorer = PathExplorer::new(SparseStab::new(1)).with_seed(42);
        let result = explorer.run_and_record(&commands);

        assert_eq!(result.path.len(), 1);
        assert!(!result.path.get(0).unwrap().is_deterministic);
    }

    #[test]
    fn test_path_explorer_replay() {
        let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

        let mut explorer = PathExplorer::new(SparseStab::new(1));

        // Force outcome 0
        let path0 = EnumeratedPath { bits: 0, len: 1 };
        let result0 = explorer.run_with_path(&commands, &path0);
        assert!(!result0.outcomes.get_bit(QubitId(0)).unwrap());

        // Force outcome 1
        let path1 = EnumeratedPath { bits: 1, len: 1 };
        let result1 = explorer.run_with_path(&commands, &path1);
        assert!(result1.outcomes.get_bit(QubitId(0)).unwrap());
    }

    #[test]
    fn test_path_enumeration_statistics() {
        // Simple circuit: H then measure
        // Should have 50% probability of each outcome
        let commands = CommandBuilder::new().pz(0).h(0).mz(0).build();

        let mut explorer = PathExplorer::new(SparseStab::new(1));
        let mut stats = PathStatistics::new();

        // Enumerate all paths (just 2 for 1 measurement)
        for path in PathEnumerator::new(1) {
            let result = explorer.run_with_path(&commands, &path);
            let outcome = result.outcomes.get_bit(QubitId(0)).unwrap_or(false);
            let value = if outcome { 1.0 } else { 0.0 };
            stats.add(value, path.probability());
        }

        // Mean should be 0.5 (50% chance of 1)
        assert!(
            (stats.mean() - 0.5).abs() < 1e-10,
            "Expected mean 0.5, got {}",
            stats.mean()
        );

        // Total weight should be 1.0 (complete enumeration)
        assert!(
            (stats.total_weight() - 1.0).abs() < 1e-10,
            "Expected total weight 1.0, got {}",
            stats.total_weight()
        );
    }

    #[test]
    fn test_path_enumeration_bell_state() {
        // Bell state: H on q0, CX, measure both
        // Should always get correlated outcomes (00 or 11)
        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .h(0)
            .cx(0, 1)
            .mz(0)
            .mz(1)
            .build();

        let mut explorer = PathExplorer::new(SparseStab::new(2));

        // The first measurement is non-deterministic (50/50)
        // The second is deterministic (correlated with first)
        // So we only have 2 paths, not 4

        let path0 = EnumeratedPath { bits: 0, len: 1 }; // Force first to 0
        let result0 = explorer.run_with_path(&commands, &path0);
        let q0_0 = result0.outcomes.get_bit(QubitId(0)).unwrap();
        let q1_0 = result0.outcomes.get_bit(QubitId(1)).unwrap();
        assert_eq!(q0_0, q1_0, "Bell state should be correlated");

        let path1 = EnumeratedPath { bits: 1, len: 1 }; // Force first to 1
        let result1 = explorer.run_with_path(&commands, &path1);
        let q0_1 = result1.outcomes.get_bit(QubitId(0)).unwrap();
        let q1_1 = result1.outcomes.get_bit(QubitId(1)).unwrap();
        assert_eq!(q0_1, q1_1, "Bell state should be correlated");
    }
}
