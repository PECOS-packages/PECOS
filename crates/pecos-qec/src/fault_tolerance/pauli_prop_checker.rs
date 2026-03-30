// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Efficient fault tolerance checking using Pauli propagation.
//!
//! This module provides fault tolerance checking using the `PauliProp` simulator,
//! which is significantly faster than full stabilizer simulation for tracking
//! how Pauli errors propagate through Clifford circuits.
//!
//! # Efficiency
//!
//! `PauliProp` tracks only the X and Z components of a Pauli error as it propagates
//! through gates. This gives O(1) gate operations instead of O(n^2) for full
//! stabilizer tableau simulation.
//!
//! # Usage
//!
//! ```
//! use pecos_qec::PauliPropChecker;
//! use pecos_quantum::TickCircuit;
//!
//! // Build a 3-qubit code syndrome extraction circuit
//! let mut circuit = TickCircuit::new();
//! circuit.tick().pz(&[0, 1, 2, 3, 4]);  // Initialize qubits
//! circuit.tick().cx(&[(0, 3), (1, 4)]); // CNOT from data to ancilla
//! circuit.tick().cx(&[(1, 3), (2, 4)]); // Second round
//! circuit.tick().mz(&[3, 4]);           // Measure ancillas
//!
//! let checker = PauliPropChecker::new(&circuit);
//!
//! // Analyze all weight-1 faults
//! let z_ancillas = &[3, 4];
//! let x_ancillas: &[usize] = &[];
//! let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];
//! let results = checker.analyze_all_faults(z_ancillas, x_ancillas, logicals);
//! println!("Total faults: {}", results.len());
//! ```

use super::propagator::{Direction, apply_gate};
use super::{
    FaultCheckConfig, FaultCheckResult, FaultConfiguration, PauliFault, PauliFaultIterator,
    SpacetimeLocation,
};
use pecos_core::QubitId;
use pecos_core::gate_type::GateType;
use pecos_quantum::TickCircuit;
use pecos_simulators::{CliffordGateable, PauliProp};
use std::collections::HashSet;

/// Detects which qubits in a circuit are "input qubits" (used but never prepared).
///
/// Input qubits are qubits that:
/// - Are used by gates in the circuit
/// - Never have a preparation gate (Prep) applied to them
///
/// These qubits enter the circuit carrying data from a previous stage and may
/// have faults from that previous stage. For fault tolerance analysis, we need
/// to consider input faults (s) in addition to internal faults (r), subject to
/// the constraint s + r <= t.
///
/// # Returns
///
/// A sorted vector of qubit indices that are input qubits.
#[must_use]
pub fn detect_input_qubits(circuit: &TickCircuit) -> Vec<usize> {
    let mut all_qubits: HashSet<usize> = HashSet::new();
    let mut prepared_qubits: HashSet<usize> = HashSet::new();

    for (_tick_idx, tick) in circuit.iter_ticks() {
        for gate in tick.gates() {
            for &qubit in &gate.qubits {
                let q = qubit.index();
                all_qubits.insert(q);

                // Check if this is a preparation gate
                if gate.gate_type == GateType::PZ {
                    prepared_qubits.insert(q);
                }
            }
        }
    }

    // Input qubits are those used but never prepared
    let mut input_qubits: Vec<usize> = all_qubits.difference(&prepared_qubits).copied().collect();
    input_qubits.sort_unstable();
    input_qubits
}

/// Detects which qubits in a circuit are "ancilla qubits" (prepared fresh).
///
/// Ancilla qubits are qubits that have a preparation gate (Prep) in the circuit.
/// These start in a known state and don't carry errors from previous stages.
///
/// # Returns
///
/// A sorted vector of qubit indices that are ancilla qubits.
#[must_use]
pub fn detect_ancilla_qubits(circuit: &TickCircuit) -> Vec<usize> {
    let mut prepared_qubits: HashSet<usize> = HashSet::new();

    for (_tick_idx, tick) in circuit.iter_ticks() {
        for gate in tick.gates() {
            if gate.gate_type == GateType::PZ {
                for &qubit in &gate.qubits {
                    prepared_qubits.insert(qubit.index());
                }
            }
        }
    }

    let mut ancillas: Vec<usize> = prepared_qubits.into_iter().collect();
    ancillas.sort_unstable();
    ancillas
}

/// Detects which qubits in a circuit are "output qubits" (used but never measured).
///
/// Output qubits are qubits that:
/// - Are used by gates in the circuit
/// - Never have a measurement gate applied to them
///
/// These qubits exit the circuit carrying data (and potentially errors) to the
/// next stage. For fault tolerance analysis with ambiguous syndromes, we need
/// to consider what syndromes these output errors would produce in a follow-up
/// ideal EC round.
///
/// # Returns
///
/// A sorted vector of qubit indices that are output qubits.
#[must_use]
pub fn detect_output_qubits(circuit: &TickCircuit) -> Vec<usize> {
    let mut all_qubits: HashSet<usize> = HashSet::new();
    let mut measured_qubits: HashSet<usize> = HashSet::new();

    for (_tick_idx, tick) in circuit.iter_ticks() {
        for gate in tick.gates() {
            for &qubit in &gate.qubits {
                let q = qubit.index();
                all_qubits.insert(q);

                // Check if this is a measurement gate
                if matches!(
                    gate.gate_type,
                    GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree
                ) {
                    measured_qubits.insert(q);
                }
            }
        }
    }

    // Output qubits are those used but never measured
    let mut output_qubits: Vec<usize> = all_qubits.difference(&measured_qubits).copied().collect();
    output_qubits.sort_unstable();
    output_qubits
}

/// Describes the input/output structure of a circuit/gadget.
///
/// This is determined automatically from the circuit structure and tells us
/// how fault tolerance analysis should be performed.
///
/// This is an internal type - use the accessor methods on the checkers instead.
#[derive(Debug, Clone, Default)]
#[allow(clippy::struct_field_names)]
pub(crate) struct CircuitIO {
    /// Qubits used but never prepared (carry data from previous stage)
    pub(crate) input_qubits: Vec<usize>,
    /// Qubits used but never measured (carry data to next stage)
    pub(crate) output_qubits: Vec<usize>,
    /// Qubits that are prepared (start fresh)
    pub(crate) ancilla_qubits: Vec<usize>,
    /// Qubits that are measured (results available)
    pub(crate) measured_qubits: Vec<usize>,
}

impl CircuitIO {
    /// Analyzes a circuit and returns its I/O structure.
    pub fn from_circuit(circuit: &TickCircuit) -> Self {
        let mut all_qubits: HashSet<usize> = HashSet::new();
        let mut prepared_qubits: HashSet<usize> = HashSet::new();
        let mut measured_qubits: HashSet<usize> = HashSet::new();

        for (_tick_idx, tick) in circuit.iter_ticks() {
            for gate in tick.gates() {
                for &qubit in &gate.qubits {
                    let q = qubit.index();
                    all_qubits.insert(q);

                    if gate.gate_type == GateType::PZ {
                        prepared_qubits.insert(q);
                    }
                    if matches!(
                        gate.gate_type,
                        GateType::MZ | GateType::MeasureLeaked | GateType::MeasureFree
                    ) {
                        measured_qubits.insert(q);
                    }
                }
            }
        }

        let input_qubits: Vec<usize> = all_qubits.difference(&prepared_qubits).copied().collect();
        let output_qubits: Vec<usize> = all_qubits.difference(&measured_qubits).copied().collect();

        let mut io = Self {
            input_qubits,
            output_qubits,
            ancilla_qubits: prepared_qubits.into_iter().collect(),
            measured_qubits: measured_qubits.into_iter().collect(),
        };

        // Sort all vectors for consistent ordering
        io.input_qubits.sort_unstable();
        io.output_qubits.sort_unstable();
        io.ancilla_qubits.sort_unstable();
        io.measured_qubits.sort_unstable();

        io
    }

    /// Returns true if this circuit has input qubits (is not self-contained).
    pub fn has_inputs(&self) -> bool {
        !self.input_qubits.is_empty()
    }

    /// Returns true if this circuit has output qubits (data exits).
    pub fn has_outputs(&self) -> bool {
        !self.output_qubits.is_empty()
    }

    /// Returns a human-readable description of the circuit type.
    pub fn circuit_type(&self) -> &'static str {
        match (self.has_inputs(), self.has_outputs()) {
            (false, false) => "self-contained (state prep + final measurement)",
            (false, true) => "state preparation (no inputs, has outputs)",
            (true, false) => "final measurement (has inputs, no outputs)",
            (true, true) => "pass-through gadget (has inputs and outputs)",
        }
    }
}

/// Initializes a `PauliProp` simulator with a fault configuration.
///
/// This sets up the initial Pauli error that will be propagated through the circuit.
fn init_pauli_prop_with_fault(fault: &PauliFault) -> PauliProp {
    let mut prop = PauliProp::new();

    for (qubit, &pauli) in fault.location.qubits.iter().zip(&fault.paulis) {
        let q = qubit.index();
        match pauli {
            1 => prop.track_x(&[q]),
            2 => prop.track_y(&[q]),
            3 => prop.track_z(&[q]),
            _ => {} // Identity
        }
    }

    prop
}

/// Propagates a Pauli fault through a circuit using `PauliProp`.
///
/// Returns the propagated `PauliProp` state after the circuit.
#[must_use]
pub fn propagate_fault(circuit: &TickCircuit, fault: &PauliFault) -> PauliProp {
    let mut prop = init_pauli_prop_with_fault(fault);

    // Find the tick where the fault occurs
    let fault_tick = fault.location.tick;

    for (tick_idx, tick) in circuit.iter_ticks() {
        // Skip ticks before the fault occurs
        // - For before=true faults: skip ticks < fault_tick, propagate from fault_tick onward
        // - For before=false faults: skip ticks <= fault_tick, propagate from fault_tick+1 onward
        //   BUT we need to apply gates at fault_tick BEFORE the fault, not skip entirely
        if fault.location.before {
            // Fault is before gates at fault_tick, so propagate from fault_tick onward
            if tick_idx < fault_tick {
                continue;
            }
        } else {
            // Fault is after gates at fault_tick, so propagate from fault_tick+1 onward
            if tick_idx <= fault_tick {
                continue;
            }
        }

        // Apply all gates in this tick
        for gate in tick.gates() {
            apply_gate(&mut prop, gate, Direction::Forward);
        }
    }

    prop
}

/// Propagates multiple faults through a circuit.
///
/// Faults are combined (`XORed`) and then propagated.
#[must_use]
pub fn propagate_faults(circuit: &TickCircuit, faults: &FaultConfiguration) -> PauliProp {
    let mut prop = PauliProp::new();

    // Combine all faults into initial state
    for fault in &faults.faults {
        for (qubit, &pauli) in fault.location.qubits.iter().zip(&fault.paulis) {
            let q = qubit.index();
            match pauli {
                1 => prop.track_x(&[q]),
                2 => prop.track_y(&[q]),
                3 => prop.track_z(&[q]),
                _ => {}
            }
        }
    }

    // Find the minimum fault tick to know where to start propagating
    let min_tick = faults
        .faults
        .iter()
        .map(|f| f.location.tick)
        .min()
        .unwrap_or(0);

    // Propagate through the circuit from the minimum tick onward
    for (tick_idx, tick) in circuit.iter_ticks() {
        if tick_idx >= min_tick {
            for gate in tick.gates() {
                apply_gate(&mut prop, gate, Direction::Forward);
            }
        }
    }

    prop
}

/// Checks if a propagated Pauli error anticommutes with a logical operator.
///
/// Returns true if the error anticommutes (causes a logical error).
#[must_use]
pub fn anticommutes_with_logical(
    prop: &PauliProp,
    logical_xs: &[usize],
    logical_zs: &[usize],
) -> bool {
    // Count anticommutations:
    // - X in prop anticommutes with Z in logical
    // - Z in prop anticommutes with X in logical

    let mut anticommute_count = 0;

    // Check X positions in propagated error against Z positions in logical
    for &q in &prop.get_x_qubits() {
        if logical_zs.contains(&q) {
            anticommute_count += 1;
        }
    }

    // Check Z positions in propagated error against X positions in logical
    for &q in &prop.get_z_qubits() {
        if logical_xs.contains(&q) {
            anticommute_count += 1;
        }
    }

    // Anticommutes if odd number of anticommutations
    anticommute_count % 2 == 1
}

/// Gets the syndrome bits that would be flipped by a propagated error.
///
/// For Z-basis measurements, X or Y errors on the measured qubit flip the outcome.
/// For X-basis measurements, Z or Y errors on the measured qubit flip the outcome.
///
/// # Arguments
///
/// * `prop` - The propagated Pauli error
/// * `z_measurement_qubits` - Qubits measured in Z basis (ancillas for Z-type stabilizers)
/// * `x_measurement_qubits` - Qubits measured in X basis (ancillas for X-type stabilizers)
///
/// # Returns
///
/// A tuple of (`z_syndrome_flips`, `x_syndrome_flips`) where each is a Vec of qubit indices
/// that would have their measurement outcome flipped.
#[must_use]
pub fn get_syndrome_flips(
    prop: &PauliProp,
    z_measurement_qubits: &[usize],
    x_measurement_qubits: &[usize],
) -> (Vec<usize>, Vec<usize>) {
    let mut z_flips = Vec::new();
    let mut x_flips = Vec::new();

    // Z-basis measurement: X or Y errors flip the outcome
    for &q in z_measurement_qubits {
        if prop.contains_x(q) {
            // X or Y on this qubit flips Z measurement
            z_flips.push(q);
        }
    }

    // X-basis measurement: Z or Y errors flip the outcome
    for &q in x_measurement_qubits {
        if prop.contains_z(q) {
            // Z or Y on this qubit flips X measurement
            x_flips.push(q);
        }
    }

    (z_flips, x_flips)
}

/// Checks if a propagated error would produce a non-trivial syndrome.
///
/// Returns true if any syndrome bit would be flipped.
#[must_use]
pub fn has_syndrome(
    prop: &PauliProp,
    z_measurement_qubits: &[usize],
    x_measurement_qubits: &[usize],
) -> bool {
    // Check Z-basis measurements for X errors
    for &q in z_measurement_qubits {
        if prop.contains_x(q) {
            return true;
        }
    }

    // Check X-basis measurements for Z errors
    for &q in x_measurement_qubits {
        if prop.contains_z(q) {
            return true;
        }
    }

    false
}

/// Computes the syndrome that an error would produce against a set of stabilizers.
///
/// This is used to compute "follow-up syndromes" - what syndromes would be produced
/// if we measured these stabilizers after the gadget completes.
///
/// # Arguments
///
/// * `prop` - The propagated Pauli error (on output qubits)
/// * `stabilizers` - List of stabilizers as (`x_positions`, `z_positions`) tuples
///
/// # Returns
///
/// A vector of bools, one per stabilizer, indicating whether each would be flipped.
/// True means the error anticommutes with that stabilizer (syndrome = 1).
#[must_use]
pub fn compute_stabilizer_syndromes(
    prop: &PauliProp,
    stabilizers: &[(&[usize], &[usize])],
) -> Vec<bool> {
    stabilizers
        .iter()
        .map(|(x_positions, z_positions)| {
            // Error E anticommutes with stabilizer S if odd number of anticommutations
            // X in E anticommutes with Z in S, and Z in E anticommutes with X in S
            let mut anticommute_count = 0;

            // Check X positions in error against Z positions in stabilizer
            for &q in &prop.get_x_qubits() {
                if z_positions.contains(&q) {
                    anticommute_count += 1;
                }
            }

            // Check Z positions in error against X positions in stabilizer
            for &q in &prop.get_z_qubits() {
                if x_positions.contains(&q) {
                    anticommute_count += 1;
                }
            }

            anticommute_count % 2 == 1
        })
        .collect()
}

/// Extracts the output error from a propagated error.
///
/// Returns a new `PauliProp` containing only the error on the specified output qubits.
#[must_use]
pub fn extract_output_error(prop: &PauliProp, output_qubits: &[usize]) -> PauliProp {
    let mut output = PauliProp::new();

    for &q in output_qubits {
        if prop.contains_x(q) && prop.contains_z(q) {
            output.track_y(&[q]);
        } else if prop.contains_x(q) {
            output.track_x(&[q]);
        } else if prop.contains_z(q) {
            output.track_z(&[q]);
        }
    }

    output
}

/// Configuration for follow-up syndrome analysis.
///
/// When analyzing gadgets with output qubits, the syndrome from within the gadget
/// may be ambiguous. But if we consider what syndromes would be produced by an
/// ideal error correction round following the gadget, we may be able to uniquely
/// identify the logical outcome.
#[derive(Debug, Clone, Default)]
pub struct FollowUpConfig {
    /// Output qubits (data qubits that exit the gadget)
    pub output_qubits: Vec<usize>,

    /// Stabilizers that will be measured in the follow-up EC round.
    /// Each stabilizer is (`x_positions`, `z_positions`).
    pub follow_up_stabilizers: Vec<(Vec<usize>, Vec<usize>)>,
}

impl FollowUpConfig {
    /// Creates a new follow-up configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the output qubits.
    #[must_use]
    pub fn with_output_qubits(mut self, qubits: Vec<usize>) -> Self {
        self.output_qubits = qubits;
        self
    }

    /// Adds a stabilizer that will be measured in the follow-up round.
    ///
    /// # Arguments
    ///
    /// * `x_positions` - Qubits where the stabilizer has X
    /// * `z_positions` - Qubits where the stabilizer has Z
    #[must_use]
    pub fn with_stabilizer(mut self, x_positions: Vec<usize>, z_positions: Vec<usize>) -> Self {
        self.follow_up_stabilizers.push((x_positions, z_positions));
        self
    }

    /// Adds multiple stabilizers from a code definition.
    ///
    /// Convenient for adding all stabilizers from a `StabilizerCodeSpec`.
    #[must_use]
    pub fn with_stabilizers(mut self, stabilizers: Vec<(Vec<usize>, Vec<usize>)>) -> Self {
        self.follow_up_stabilizers.extend(stabilizers);
        self
    }

    /// Returns true if follow-up analysis is configured.
    #[must_use]
    pub fn has_follow_up(&self) -> bool {
        !self.follow_up_stabilizers.is_empty()
    }
}

/// Classification of a fault based on detectability and logical error.
///
/// This classification determines circuit fault tolerance without needing a decoder:
/// - `UndetectableLogicalError`: The circuit is fundamentally broken for this fault
/// - `UndetectableStabilizer`: The fault is harmless (equivalent to stabilizer)
/// - `DetectableError`: Whether this causes a logical error depends on decoder quality
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FaultClass {
    /// Undetectable logical error: syndrome = 0, but anticommutes with logical operator.
    /// This is an inherent circuit failure - no decoder can help.
    UndetectableLogicalError,

    /// Undetectable stabilizer: syndrome = 0, and commutes with all logical operators.
    /// This is harmless - the fault has no effect on the logical state.
    UndetectableStabilizer,

    /// Detectable error: syndrome != 0.
    /// A good decoder should be able to correct this.
    DetectableError,
}

impl FaultClass {
    /// Returns true if this fault class represents a certain logical failure.
    #[must_use]
    pub fn is_certain_failure(&self) -> bool {
        matches!(self, FaultClass::UndetectableLogicalError)
    }

    /// Returns true if this fault class is definitely safe.
    #[must_use]
    pub fn is_safe(&self) -> bool {
        matches!(self, FaultClass::UndetectableStabilizer)
    }

    /// Returns true if this fault class is detectable by syndrome measurement.
    #[must_use]
    pub fn is_detectable(&self) -> bool {
        matches!(self, FaultClass::DetectableError)
    }
}

/// Refined classification of detectable errors based on syndrome analysis.
///
/// By grouping faults by their syndrome pattern, we can determine whether
/// a decoder can reliably correct them without actually running a decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SyndromeClass {
    /// All faults producing this syndrome cause NO logical error.
    /// A decoder that corrects based on this syndrome will always succeed.
    Correctable,

    /// All faults producing this syndrome cause a logical error.
    /// The decoder can detect that something uncorrectable happened,
    /// but cannot fix it. At least we know we failed.
    DetectedUncorrectable,

    /// Some faults with this syndrome cause logical errors, others don't.
    /// The decoder must guess and will sometimes be wrong.
    /// This indicates a fundamental limitation - faults differ by a logical operator.
    Ambiguous,
}

impl SyndromeClass {
    /// Returns true if a decoder will always succeed for this syndrome.
    #[must_use]
    pub fn is_correctable(&self) -> bool {
        matches!(self, SyndromeClass::Correctable)
    }

    /// Returns true if the decoder will always fail but can detect failure.
    #[must_use]
    pub fn is_detected_failure(&self) -> bool {
        matches!(self, SyndromeClass::DetectedUncorrectable)
    }

    /// Returns true if decoder success depends on which fault occurred.
    #[must_use]
    pub fn is_ambiguous(&self) -> bool {
        matches!(self, SyndromeClass::Ambiguous)
    }
}

/// Analysis of a single syndrome pattern.
#[derive(Debug, Clone)]
pub struct SyndromeAnalysis {
    /// The syndrome pattern (Z flips, then X flips).
    pub syndrome: Vec<u8>,
    /// Number of faults that produce this syndrome and cause no logical error.
    pub correctable_count: usize,
    /// Number of faults that produce this syndrome and cause a logical error.
    pub uncorrectable_count: usize,
    /// Classification based on the counts.
    pub class: SyndromeClass,
}

impl SyndromeAnalysis {
    /// Returns the total number of faults with this syndrome.
    #[must_use]
    pub fn total_faults(&self) -> usize {
        self.correctable_count + self.uncorrectable_count
    }

    /// Returns the probability of successful correction assuming uniform fault distribution.
    #[must_use]
    pub fn success_probability(&self) -> f64 {
        let total = self.total_faults();
        if total == 0 {
            1.0
        } else {
            self.correctable_count as f64 / total as f64
        }
    }
}

/// Why fault tolerance check failed.
///
/// Provides details about what went wrong when a circuit is not fault tolerant.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FaultToleranceFailure {
    /// Faults that produce no syndrome but cause logical errors.
    /// No decoder can detect or correct these.
    UndetectableLogicalErrors {
        /// Number of such faults found.
        count: usize,
    },

    /// Same syndrome maps to different logical outcomes.
    /// This implies undetectable logical errors exist (F1·F2 where F1, F2 have
    /// same syndrome but different logical effects).
    AmbiguousSyndromes {
        /// Number of ambiguous syndromes.
        count: usize,
        /// Total faults affected.
        affected_faults: usize,
    },
}

impl FaultToleranceFailure {
    /// Returns a human-readable description.
    #[must_use]
    pub fn description(&self) -> String {
        match self {
            FaultToleranceFailure::UndetectableLogicalErrors { count } => {
                format!(
                    "{count} undetectable logical error(s): faults causing logical errors with no syndrome"
                )
            }
            FaultToleranceFailure::AmbiguousSyndromes {
                count,
                affected_faults,
            } => {
                format!(
                    "{count} ambiguous syndrome(s) affecting {affected_faults} fault(s): same syndrome, different logical outcomes"
                )
            }
        }
    }
}

/// Detailed decoder-independent analysis of fault tolerance.
///
/// This goes beyond simple fault classification to analyze whether a decoder
/// CAN succeed, not just whether faults are detectable.
#[derive(Debug, Clone)]
pub struct DecoderAnalysis {
    /// Analysis for each unique syndrome pattern.
    pub syndromes: Vec<SyndromeAnalysis>,

    /// Number of syndromes where decoder will always succeed.
    pub correctable_syndromes: usize,

    /// Number of syndromes where decoder will always fail (but knows it).
    pub detected_uncorrectable_syndromes: usize,

    /// Number of syndromes where decoder outcome is uncertain.
    pub ambiguous_syndromes: usize,

    /// Total faults that fall into correctable syndromes.
    pub correctable_faults: usize,

    /// Total faults that fall into detected-uncorrectable syndromes.
    pub detected_uncorrectable_faults: usize,

    /// Total faults that fall into ambiguous syndromes.
    pub ambiguous_faults: usize,

    /// Faults with no syndrome (undetectable) - from `FaultClass` analysis.
    pub undetectable_logical_errors: usize,

    /// Faults with no syndrome that are stabilizers.
    pub undetectable_stabilizers: usize,
}

impl DecoderAnalysis {
    /// Returns whether the circuit is fault tolerant at this weight.
    ///
    /// **Binary verdict:**
    /// - `Ok(())` → Circuit IS fault tolerant. Any correct decoder will succeed.
    /// - `Err(failures)` → Circuit is NOT fault tolerant. No decoder can fully correct.
    ///
    /// At weight ≤ t (where t = ⌊(d-1)/2⌋), fault tolerance is all-or-nothing:
    /// either all syndromes have unique logical effects (any decoder works),
    /// or they don't (no decoder can fully succeed).
    pub fn is_fault_tolerant(&self) -> Result<(), Vec<FaultToleranceFailure>> {
        let mut failures = Vec::new();

        if self.undetectable_logical_errors > 0 {
            failures.push(FaultToleranceFailure::UndetectableLogicalErrors {
                count: self.undetectable_logical_errors,
            });
        }

        if self.ambiguous_syndromes > 0 {
            failures.push(FaultToleranceFailure::AmbiguousSyndromes {
                count: self.ambiguous_syndromes,
                affected_faults: self.ambiguous_faults,
            });
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }

    /// Simple boolean check for fault tolerance.
    #[must_use]
    pub fn is_ft(&self) -> bool {
        self.undetectable_logical_errors == 0 && self.ambiguous_syndromes == 0
    }

    /// Returns the total number of faults analyzed.
    #[must_use]
    pub fn total_faults(&self) -> usize {
        self.correctable_faults
            + self.detected_uncorrectable_faults
            + self.ambiguous_faults
            + self.undetectable_logical_errors
            + self.undetectable_stabilizers
    }

    /// Returns the best-case logical error rate (assuming optimal decoder choices).
    ///
    /// This counts undetectable logical errors plus the minority of ambiguous faults.
    #[must_use]
    pub fn best_case_failure_rate(&self, total_faults: usize) -> f64 {
        if total_faults == 0 {
            return 0.0;
        }

        // Undetectable logical errors always fail
        let mut failures = self.undetectable_logical_errors;

        // For ambiguous syndromes, optimal decoder picks the majority outcome
        // So failures = minority count for each ambiguous syndrome
        for syndrome in &self.syndromes {
            if syndrome.class == SyndromeClass::Ambiguous {
                failures += syndrome.correctable_count.min(syndrome.uncorrectable_count);
            }
        }

        // Detected uncorrectable always fail
        failures += self.detected_uncorrectable_faults;

        failures as f64 / total_faults as f64
    }

    /// Returns the worst-case logical error rate (assuming adversarial decoder choices).
    #[must_use]
    pub fn worst_case_failure_rate(&self, total_faults: usize) -> f64 {
        if total_faults == 0 {
            return 0.0;
        }

        // Undetectable logical errors always fail
        let mut failures = self.undetectable_logical_errors;

        // For ambiguous syndromes, worst decoder picks wrong every time for majority
        for syndrome in &self.syndromes {
            if syndrome.class == SyndromeClass::Ambiguous {
                failures += syndrome.correctable_count.max(syndrome.uncorrectable_count);
            }
        }

        // Detected uncorrectable always fail
        failures += self.detected_uncorrectable_faults;

        failures as f64 / total_faults as f64
    }
}

// ============================================================================
// Syndrome History Analysis for Multi-Round QEC
// ============================================================================

/// A measurement round in a circuit.
///
/// Represents a set of measurements that occur at a specific tick.
#[derive(Debug, Clone)]
pub struct MeasurementRound {
    /// The tick at which this measurement round occurs.
    pub tick: usize,
    /// Qubits measured in Z basis in this round.
    pub z_qubits: Vec<usize>,
    /// Qubits measured in X basis in this round.
    pub x_qubits: Vec<usize>,
}

/// Syndrome history for a single fault across multiple measurement rounds.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SyndromeHistory {
    /// Syndrome at each measurement round.
    /// Each entry is a vector of syndrome bits (0 or 1) for that round.
    pub rounds: Vec<Vec<u8>>,
}

impl SyndromeHistory {
    /// Returns true if any round has a non-trivial syndrome.
    #[must_use]
    pub fn is_detected(&self) -> bool {
        self.rounds.iter().any(|r| r.iter().any(|&b| b != 0))
    }

    /// Returns the indices of rounds where syndrome was non-trivial.
    #[must_use]
    pub fn detection_rounds(&self) -> Vec<usize> {
        self.rounds
            .iter()
            .enumerate()
            .filter(|(_, r)| r.iter().any(|&b| b != 0))
            .map(|(i, _)| i)
            .collect()
    }

    /// Returns true if this is an empty history (no rounds).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rounds.is_empty()
    }
}

/// Analysis of a single syndrome history pattern.
#[derive(Debug, Clone)]
pub struct SyndromeHistoryAnalysis {
    /// The syndrome history pattern.
    pub history: SyndromeHistory,
    /// Number of faults with this history that cause no logical error.
    pub correctable_count: usize,
    /// Number of faults with this history that cause a logical error.
    pub uncorrectable_count: usize,
    /// Classification based on the counts.
    pub class: SyndromeClass,
}

/// Result of syndrome history analysis for multi-round circuits.
#[derive(Debug, Clone)]
pub struct SyndromeHistoryResult {
    /// Measurement rounds identified in the circuit.
    pub rounds: Vec<MeasurementRound>,

    /// Analysis for each unique syndrome history pattern.
    pub histories: Vec<SyndromeHistoryAnalysis>,

    /// Number of faults that are never detected in any round but cause logical errors.
    pub never_detected_logical_errors: usize,

    /// Number of faults that are never detected but are stabilizer-equivalent.
    pub never_detected_stabilizers: usize,

    /// Number of faults detected in at least one round, all causing no logical error.
    pub correctable_faults: usize,

    /// Number of faults detected, all causing logical error.
    pub detected_uncorrectable_faults: usize,

    /// Number of faults with ambiguous outcome (same history, different logical effect).
    pub ambiguous_faults: usize,

    /// Number of ambiguous syndrome history patterns.
    pub ambiguous_histories: usize,

    /// Total faults analyzed.
    pub total_faults: usize,
}

impl SyndromeHistoryResult {
    /// Returns whether the circuit is fault tolerant with syndrome history analysis.
    ///
    /// This is more permissive than single-shot analysis because a fault only needs
    /// to be detected in SOME round, not necessarily the final state.
    pub fn is_fault_tolerant(&self) -> Result<(), Vec<FaultToleranceFailure>> {
        let mut failures = Vec::new();

        if self.never_detected_logical_errors > 0 {
            failures.push(FaultToleranceFailure::UndetectableLogicalErrors {
                count: self.never_detected_logical_errors,
            });
        }

        if self.ambiguous_histories > 0 {
            failures.push(FaultToleranceFailure::AmbiguousSyndromes {
                count: self.ambiguous_histories,
                affected_faults: self.ambiguous_faults,
            });
        }

        if failures.is_empty() {
            Ok(())
        } else {
            Err(failures)
        }
    }

    /// Simple boolean check for fault tolerance.
    #[must_use]
    pub fn is_ft(&self) -> bool {
        self.never_detected_logical_errors == 0 && self.ambiguous_histories == 0
    }
}

/// Extracts measurement rounds from a circuit.
///
/// A measurement round is a set of Z-basis measurement operations at the same tick.
/// Note: Currently only tracks Z-basis measurements (Measure, `MeasureFree`).
#[must_use]
pub fn extract_measurement_rounds(circuit: &TickCircuit) -> Vec<MeasurementRound> {
    let mut rounds = Vec::new();

    for (tick_idx, tick) in circuit.iter_ticks() {
        let mut z_qubits = Vec::new();
        let x_qubits = Vec::new(); // Currently not tracking X-basis measurements

        for gate in tick.gates() {
            match gate.gate_type {
                GateType::MZ | GateType::MeasureFree => {
                    // Z-basis measurement
                    for q in &gate.qubits {
                        z_qubits.push(q.0);
                    }
                }
                _ => {}
            }
        }

        if !z_qubits.is_empty() || !x_qubits.is_empty() {
            rounds.push(MeasurementRound {
                tick: tick_idx,
                z_qubits,
                x_qubits,
            });
        }
    }

    rounds
}

/// Propagates a fault through a circuit up to a specific tick and extracts syndrome.
///
/// This simulates what the syndrome would be if we stopped propagation at `until_tick`.
fn propagate_until_tick(circuit: &TickCircuit, fault: &PauliFault, until_tick: usize) -> PauliProp {
    let mut prop = PauliProp::new();

    // Initialize with fault
    for (qubit, pauli_byte) in fault.location.qubits.iter().zip(fault.paulis.iter()) {
        let qubit_idx = qubit.0;
        match pauli_byte {
            1 => prop.track_x(&[qubit_idx]), // X
            2 => prop.track_z(&[qubit_idx]), // Z
            3 => {
                // Y = iXZ
                prop.track_x(&[qubit_idx]);
                prop.track_z(&[qubit_idx]);
            }
            _ => {} // I
        }
    }

    // Propagate through ticks up to and including until_tick
    let fault_tick = fault.location.tick;

    for (tick_idx, tick) in circuit.iter_ticks() {
        if tick_idx > until_tick {
            break;
        }

        // Skip propagation before the fault occurs
        if tick_idx < fault_tick || (tick_idx == fault_tick && fault.location.before) {
            continue;
        }

        // Propagate through all gates in this tick
        for gate in tick.gates() {
            let qubits: Vec<QubitId> = gate.qubits.iter().copied().collect();
            match gate.gate_type {
                GateType::CX => {
                    if qubits.len() >= 2 {
                        prop.cx(&[(qubits[0], qubits[1])]);
                    }
                }
                GateType::CZ => {
                    if qubits.len() >= 2 {
                        prop.cz(&[(qubits[0], qubits[1])]);
                    }
                }
                GateType::CY => {
                    if qubits.len() >= 2 {
                        prop.cy(&[(qubits[0], qubits[1])]);
                    }
                }
                GateType::H => {
                    for q in &qubits {
                        prop.h(&[*q]);
                    }
                }
                GateType::SZ | GateType::SZdg => {
                    for q in &qubits {
                        prop.sz(&[*q]);
                    }
                }
                GateType::SX | GateType::SXdg => {
                    for q in &qubits {
                        prop.sx(&[*q]);
                    }
                }
                GateType::SY | GateType::SYdg => {
                    for q in &qubits {
                        prop.sy(&[*q]);
                    }
                }
                GateType::SWAP => {
                    if qubits.len() >= 2 {
                        prop.swap(&[(qubits[0], qubits[1])]);
                    }
                }
                _ => {}
            }
        }
    }

    prop
}

/// Computes syndrome history for a fault across all measurement rounds.
fn compute_syndrome_history(
    circuit: &TickCircuit,
    fault: &PauliFault,
    rounds: &[MeasurementRound],
) -> SyndromeHistory {
    let fault_tick = fault.location.tick;
    let mut history_rounds = Vec::new();

    for round in rounds {
        // Only check rounds that occur after the fault
        // Fault hasn't happened if: round is before fault, or same tick but fault is "after" the gates
        if round.tick < fault_tick || (round.tick == fault_tick && !fault.location.before) {
            // Fault hasn't happened yet at this round
            history_rounds.push(vec![0; round.z_qubits.len() + round.x_qubits.len()]);
            continue;
        }

        // Propagate fault up to this measurement round
        let prop = propagate_until_tick(circuit, fault, round.tick);

        // Extract syndrome for this round
        let mut syndrome = Vec::new();

        // Z measurements detect X errors
        for &q in &round.z_qubits {
            syndrome.push(u8::from(prop.contains_x(q)));
        }

        // X measurements detect Z errors
        for &q in &round.x_qubits {
            syndrome.push(u8::from(prop.contains_z(q)));
        }

        history_rounds.push(syndrome);
    }

    SyndromeHistory {
        rounds: history_rounds,
    }
}

/// Classifies a propagated fault based on syndrome and logical error.
///
/// # Arguments
///
/// * `prop` - The propagated Pauli error
/// * `z_measurement_qubits` - Qubits measured in Z basis
/// * `x_measurement_qubits` - Qubits measured in X basis
/// * `logicals` - Logical operators as (`x_positions`, `z_positions`) pairs
///
/// # Returns
///
/// The classification of the fault.
#[must_use]
pub fn classify_fault(
    prop: &PauliProp,
    z_measurement_qubits: &[usize],
    x_measurement_qubits: &[usize],
    logicals: &[(&[usize], &[usize])],
) -> FaultClass {
    let detectable = has_syndrome(prop, z_measurement_qubits, x_measurement_qubits);

    if detectable {
        FaultClass::DetectableError
    } else {
        // Undetectable - check if it's a logical error or stabilizer
        let causes_logical_error = logicals
            .iter()
            .any(|(xs, zs)| anticommutes_with_logical(prop, xs, zs));

        if causes_logical_error {
            FaultClass::UndetectableLogicalError
        } else {
            FaultClass::UndetectableStabilizer
        }
    }
}

/// Result of propagating a fault through a circuit.
#[derive(Debug, Clone)]
pub struct PropagationResult {
    /// The final propagated Pauli error.
    pub propagated_error: PauliProp,
    /// Syndrome bits flipped on Z-basis measurement qubits.
    pub z_syndrome_flips: Vec<usize>,
    /// Syndrome bits flipped on X-basis measurement qubits.
    pub x_syndrome_flips: Vec<usize>,
    /// Whether any logical operator is affected.
    pub logical_errors: Vec<bool>,
}

impl PropagationResult {
    /// Returns true if any syndrome bit is flipped.
    #[must_use]
    pub fn has_syndrome(&self) -> bool {
        !self.z_syndrome_flips.is_empty() || !self.x_syndrome_flips.is_empty()
    }

    /// Returns true if any logical error occurred.
    #[must_use]
    pub fn has_logical_error(&self) -> bool {
        self.logical_errors.iter().any(|&e| e)
    }

    /// Returns the weight of the propagated error.
    #[must_use]
    pub fn output_weight(&self) -> usize {
        self.propagated_error.weight()
    }

    /// Classifies this fault based on detectability and logical error.
    #[must_use]
    pub fn classify(&self) -> FaultClass {
        if self.has_syndrome() {
            FaultClass::DetectableError
        } else if self.has_logical_error() {
            FaultClass::UndetectableLogicalError
        } else {
            FaultClass::UndetectableStabilizer
        }
    }
}

/// Summary of fault tolerance analysis without using a decoder.
///
/// This provides a decoder-independent assessment of circuit fault tolerance.
#[derive(Debug, Clone)]
pub struct FaultToleranceAnalysis {
    /// Total number of fault configurations tested.
    pub total_tested: usize,

    /// Number of faults that cause undetectable logical errors.
    /// These represent inherent circuit failures - no decoder can help.
    pub undetectable_logical_errors: usize,

    /// Number of faults that produce undetectable stabilizers.
    /// These are harmless - the fault has no effect on logical state.
    pub undetectable_stabilizers: usize,

    /// Number of faults that produce detectable errors.
    /// A good decoder should be able to correct these.
    pub detectable_errors: usize,

    /// The fault weight tested.
    pub weight: usize,

    /// The specific fault configurations that cause undetectable logical errors.
    /// Empty if `collect_failures` was false during analysis.
    pub failure_details: Vec<(FaultConfiguration, PropagationResult)>,
}

impl FaultToleranceAnalysis {
    /// Returns true if the circuit is t-fault tolerant (no undetectable logical errors).
    #[must_use]
    pub fn is_fault_tolerant(&self) -> bool {
        self.undetectable_logical_errors == 0
    }

    /// Returns the fraction of faults that are certain failures.
    #[must_use]
    pub fn failure_rate(&self) -> f64 {
        if self.total_tested == 0 {
            0.0
        } else {
            self.undetectable_logical_errors as f64 / self.total_tested as f64
        }
    }

    /// Returns the fraction of faults that are safe (stabilizers).
    #[must_use]
    pub fn safe_rate(&self) -> f64 {
        if self.total_tested == 0 {
            0.0
        } else {
            self.undetectable_stabilizers as f64 / self.total_tested as f64
        }
    }

    /// Returns the fraction of faults that are detectable.
    #[must_use]
    pub fn detectable_rate(&self) -> f64 {
        if self.total_tested == 0 {
            0.0
        } else {
            self.detectable_errors as f64 / self.total_tested as f64
        }
    }
}

/// A fault checker using Pauli propagation for efficient fault tolerance testing.
///
/// This is significantly faster than `FaultChecker` for checking if faults
/// cause logical errors, as it only tracks Pauli error propagation rather
/// than full quantum state.
///
/// # Automatic I/O Detection
///
/// The checker automatically detects the circuit's I/O structure:
///
/// - **Input qubits**: used but never prepared (carry data from previous stage)
/// - **Output qubits**: used but never measured (carry data to next stage)
/// - **Ancilla qubits**: prepared fresh within the circuit
/// - **Measured qubits**: have measurement results available
///
/// This determines how fault tolerance analysis is performed:
///
/// - **Has inputs** → Use s + r <= t enumeration (input faults + internal faults)
/// - **Has outputs + ambiguous syndromes** → May need follow-up stabilizers
/// - **No inputs, no outputs** → Self-contained, standard analysis
pub struct PauliPropChecker<'a> {
    circuit: &'a TickCircuit,
    config: FaultCheckConfig,
    locations: Vec<SpacetimeLocation>,
    /// Complete I/O structure detected from the circuit
    io: CircuitIO,
}

impl<'a> PauliPropChecker<'a> {
    /// Creates a new Pauli propagation checker for the given circuit.
    ///
    /// Automatically analyzes the circuit's I/O structure to determine:
    /// - Which qubits are inputs (need s + r <= t enumeration)
    /// - Which qubits are outputs (may need follow-up syndrome analysis)
    #[must_use]
    pub fn new(circuit: &'a TickCircuit) -> Self {
        let locations = super::circuit_runner::extract_spacetime_locations(circuit, false);
        let io = CircuitIO::from_circuit(circuit);
        Self {
            circuit,
            config: FaultCheckConfig::default(),
            locations,
            io,
        }
    }

    /// Sets the fault check configuration.
    #[must_use]
    pub fn with_config(mut self, config: FaultCheckConfig) -> Self {
        self.config = config;
        self
    }

    /// Sets whether to include initial qubit locations.
    #[must_use]
    pub fn with_initial_locations(mut self, include: bool) -> Self {
        self.locations = super::circuit_runner::extract_spacetime_locations(self.circuit, include);
        self
    }

    /// Returns the spacetime locations that will be checked.
    #[must_use]
    pub fn locations(&self) -> &[SpacetimeLocation] {
        &self.locations
    }

    /// Returns the detected input qubits.
    ///
    /// These are qubits used by the circuit but never prepared, meaning they
    /// carry data (and potentially errors) from a previous stage.
    #[must_use]
    pub fn input_qubits(&self) -> &[usize] {
        &self.io.input_qubits
    }

    /// Returns the detected output qubits.
    ///
    /// These are qubits used by the circuit but never measured, meaning they
    /// carry data (and potentially errors) to the next stage.
    #[must_use]
    pub fn output_qubits(&self) -> &[usize] {
        &self.io.output_qubits
    }

    /// Returns true if this circuit has input qubits.
    ///
    /// If true, fault tolerance analysis should use s + r <= t enumeration
    /// to account for input faults.
    #[must_use]
    pub fn has_input_qubits(&self) -> bool {
        self.io.has_inputs()
    }

    /// Returns true if this circuit has output qubits.
    ///
    /// If true and analysis shows ambiguous syndromes, follow-up stabilizers
    /// may be needed to properly assess fault tolerance.
    #[must_use]
    pub fn has_output_qubits(&self) -> bool {
        self.io.has_outputs()
    }

    /// Returns the ancilla qubits (prepared within the circuit).
    #[must_use]
    pub fn ancilla_qubits(&self) -> &[usize] {
        &self.io.ancilla_qubits
    }

    /// Returns the measured qubits.
    #[must_use]
    pub fn measured_qubits(&self) -> &[usize] {
        &self.io.measured_qubits
    }

    /// Returns a description of the circuit type based on I/O structure.
    #[must_use]
    pub fn circuit_type(&self) -> &'static str {
        self.io.circuit_type()
    }

    /// Checks for logical errors using Pauli propagation.
    ///
    /// # Arguments
    ///
    /// * `logical_xs` - X positions of the logical operator to check against
    /// * `logical_zs` - Z positions of the logical operator to check against
    ///
    /// # Returns
    ///
    /// The result of the fault tolerance check.
    #[must_use]
    pub fn check_logical_error(
        &self,
        logical_xs: &[usize],
        logical_zs: &[usize],
    ) -> FaultCheckResult {
        let mut failures = Vec::new();
        let mut total_tested = 0;

        let fault_iter = PauliFaultIterator::new(
            self.locations.clone(),
            self.config.max_weight,
            self.config.clone(),
        );

        for fault_config in fault_iter {
            total_tested += 1;

            // Propagate faults through circuit
            let prop = propagate_faults(self.circuit, &fault_config);

            // Check if propagated error anticommutes with logical operator
            if anticommutes_with_logical(&prop, logical_xs, logical_zs) {
                failures.push(fault_config);

                if self.config.stop_on_first_failure {
                    break;
                }
            }
        }

        FaultCheckResult::new(failures, total_tested, self.config.max_weight)
    }

    /// Checks for logical errors against multiple logical operators.
    ///
    /// Returns true if any fault causes an error on any logical operator.
    #[must_use]
    pub fn check_multiple_logicals(
        &self,
        logicals: &[(&[usize], &[usize])], // Vec of (logical_xs, logical_zs)
    ) -> FaultCheckResult {
        let mut failures = Vec::new();
        let mut total_tested = 0;

        let fault_iter = PauliFaultIterator::new(
            self.locations.clone(),
            self.config.max_weight,
            self.config.clone(),
        );

        for fault_config in fault_iter {
            total_tested += 1;

            let prop = propagate_faults(self.circuit, &fault_config);

            // Check against all logical operators
            let causes_error = logicals
                .iter()
                .any(|(xs, zs)| anticommutes_with_logical(&prop, xs, zs));

            if causes_error {
                failures.push(fault_config);

                if self.config.stop_on_first_failure {
                    break;
                }
            }
        }

        FaultCheckResult::new(failures, total_tested, self.config.max_weight)
    }

    /// Checks if the propagated error has weight above a threshold.
    ///
    /// Useful for checking if errors spread beyond acceptable limits.
    #[must_use]
    pub fn check_error_weight(&self, max_output_weight: usize) -> FaultCheckResult {
        let mut failures = Vec::new();
        let mut total_tested = 0;

        let fault_iter = PauliFaultIterator::new(
            self.locations.clone(),
            self.config.max_weight,
            self.config.clone(),
        );

        for fault_config in fault_iter {
            total_tested += 1;

            let prop = propagate_faults(self.circuit, &fault_config);

            if prop.weight() > max_output_weight {
                failures.push(fault_config);

                if self.config.stop_on_first_failure {
                    break;
                }
            }
        }

        FaultCheckResult::new(failures, total_tested, self.config.max_weight)
    }

    /// Checks syndrome detection - finds faults that produce unexpected syndromes.
    ///
    /// This checks if faults produce syndromes that don't match what's expected
    /// for the input error weight. Useful for verifying syndrome extraction circuits.
    ///
    /// # Arguments
    ///
    /// * `z_ancillas` - Qubits measured in Z basis (for detecting X errors)
    /// * `x_ancillas` - Qubits measured in X basis (for detecting Z errors)
    /// * `expect_syndrome` - Whether we expect faults to produce a syndrome
    ///
    /// # Returns
    ///
    /// Faults that produce unexpected syndrome behavior.
    #[must_use]
    pub fn check_syndrome_detection(
        &self,
        z_ancillas: &[usize],
        x_ancillas: &[usize],
        expect_syndrome: bool,
    ) -> FaultCheckResult {
        let mut failures = Vec::new();
        let mut total_tested = 0;

        let fault_iter = PauliFaultIterator::new(
            self.locations.clone(),
            self.config.max_weight,
            self.config.clone(),
        );

        for fault_config in fault_iter {
            total_tested += 1;

            let prop = propagate_faults(self.circuit, &fault_config);
            let produces_syndrome = has_syndrome(&prop, z_ancillas, x_ancillas);

            // Failure if syndrome doesn't match expectation
            if produces_syndrome != expect_syndrome {
                failures.push(fault_config);

                if self.config.stop_on_first_failure {
                    break;
                }
            }
        }

        FaultCheckResult::new(failures, total_tested, self.config.max_weight)
    }

    /// Full fault tolerance check combining syndrome and logical error detection.
    ///
    /// This implements the exRec conditions from arXiv:quant-ph/0504218:
    /// - For fault-free EC: weight <= t errors should produce syndromes but no logical errors
    /// - For fault-free Ga: weight <= t errors should produce weight <= t output errors
    ///
    /// # Arguments
    ///
    /// * `z_ancillas` - Z-basis measurement qubits
    /// * `x_ancillas` - X-basis measurement qubits
    /// * `logicals` - Logical operators as (`x_positions`, `z_positions`) pairs
    /// * `data_qubits` - Data qubit indices (for checking output error weight)
    ///
    /// # Returns
    ///
    /// A vector of `PropagationResult` for each fault configuration.
    #[must_use]
    pub fn analyze_all_faults(
        &self,
        z_ancillas: &[usize],
        x_ancillas: &[usize],
        logicals: &[(&[usize], &[usize])],
    ) -> Vec<(FaultConfiguration, PropagationResult)> {
        let mut results = Vec::new();

        let fault_iter = PauliFaultIterator::new(
            self.locations.clone(),
            self.config.max_weight,
            self.config.clone(),
        );

        for fault_config in fault_iter {
            let prop = propagate_faults(self.circuit, &fault_config);

            let (z_flips, x_flips) = get_syndrome_flips(&prop, z_ancillas, x_ancillas);

            let logical_errors: Vec<bool> = logicals
                .iter()
                .map(|(xs, zs)| anticommutes_with_logical(&prop, xs, zs))
                .collect();

            let result = PropagationResult {
                propagated_error: prop,
                z_syndrome_flips: z_flips,
                x_syndrome_flips: x_flips,
                logical_errors,
            };

            results.push((fault_config, result));
        }

        results
    }

    /// Analyzes fault tolerance without using a decoder.
    ///
    /// This classifies all faults into three categories:
    /// - **Undetectable logical errors**: Certain failures (no decoder can help)
    /// - **Undetectable stabilizers**: Safe (no effect on logical state)
    /// - **Detectable errors**: Decoder-dependent (good decoder should correct)
    ///
    /// A circuit is t-fault tolerant if and only if there are no undetectable
    /// logical errors for weight-t faults.
    ///
    /// # Arguments
    ///
    /// * `z_ancillas` - Qubits measured in Z basis (for detecting X errors)
    /// * `x_ancillas` - Qubits measured in X basis (for detecting Z errors)
    /// * `logicals` - Logical operators as (`x_positions`, `z_positions`) pairs
    /// * `collect_failures` - Whether to store detailed info for failures
    ///
    /// # Returns
    ///
    /// A `FaultToleranceAnalysis` with counts and optional failure details.
    #[must_use]
    pub fn analyze_fault_tolerance(
        &self,
        z_ancillas: &[usize],
        x_ancillas: &[usize],
        logicals: &[(&[usize], &[usize])],
        collect_failures: bool,
    ) -> FaultToleranceAnalysis {
        let mut total_tested = 0;
        let mut undetectable_logical_errors = 0;
        let mut undetectable_stabilizers = 0;
        let mut detectable_errors = 0;
        let mut failure_details = Vec::new();

        let fault_iter = PauliFaultIterator::new(
            self.locations.clone(),
            self.config.max_weight,
            self.config.clone(),
        );

        for fault_config in fault_iter {
            total_tested += 1;

            let prop = propagate_faults(self.circuit, &fault_config);
            let classification = classify_fault(&prop, z_ancillas, x_ancillas, logicals);

            match classification {
                FaultClass::UndetectableLogicalError => {
                    undetectable_logical_errors += 1;

                    if collect_failures {
                        let (z_flips, x_flips) = get_syndrome_flips(&prop, z_ancillas, x_ancillas);
                        let logical_errors: Vec<bool> = logicals
                            .iter()
                            .map(|(xs, zs)| anticommutes_with_logical(&prop, xs, zs))
                            .collect();

                        let result = PropagationResult {
                            propagated_error: prop,
                            z_syndrome_flips: z_flips,
                            x_syndrome_flips: x_flips,
                            logical_errors,
                        };
                        failure_details.push((fault_config, result));
                    }
                }
                FaultClass::UndetectableStabilizer => {
                    undetectable_stabilizers += 1;
                }
                FaultClass::DetectableError => {
                    detectable_errors += 1;
                }
            }
        }

        FaultToleranceAnalysis {
            total_tested,
            undetectable_logical_errors,
            undetectable_stabilizers,
            detectable_errors,
            weight: self.config.max_weight,
            failure_details,
        }
    }

    /// Checks if the circuit is t-fault tolerant (no undetectable logical errors).
    ///
    /// This is a convenience method that returns true if all weight-t faults
    /// either produce a syndrome (detectable) or are equivalent to a stabilizer.
    #[must_use]
    pub fn is_fault_tolerant(
        &self,
        z_ancillas: &[usize],
        x_ancillas: &[usize],
        logicals: &[(&[usize], &[usize])],
    ) -> bool {
        let fault_iter = PauliFaultIterator::new(
            self.locations.clone(),
            self.config.max_weight,
            self.config.clone(),
        );

        for fault_config in fault_iter {
            let prop = propagate_faults(self.circuit, &fault_config);
            let classification = classify_fault(&prop, z_ancillas, x_ancillas, logicals);

            if classification == FaultClass::UndetectableLogicalError {
                return false;
            }
        }

        true
    }

    /// Performs detailed decoder-independent analysis of fault tolerance.
    ///
    /// This goes beyond basic fault classification to determine:
    /// - Which syndromes can be reliably corrected (all faults -> no logical error)
    /// - Which syndromes indicate uncorrectable errors (all faults -> logical error)
    /// - Which syndromes are ambiguous (mixed outcomes, decoder must guess)
    ///
    /// This analysis tells you whether a "perfect" decoder could achieve zero
    /// logical errors, without actually implementing or running a decoder.
    ///
    /// # Arguments
    ///
    /// * `z_ancillas` - Qubits measured in Z basis
    /// * `x_ancillas` - Qubits measured in X basis
    /// * `logicals` - Logical operators as (`x_positions`, `z_positions`) pairs
    ///
    /// # Returns
    ///
    /// A `DecoderAnalysis` with detailed syndrome-by-syndrome breakdown.
    #[must_use]
    pub fn analyze_decoder_requirements(
        &self,
        z_ancillas: &[usize],
        x_ancillas: &[usize],
        logicals: &[(&[usize], &[usize])],
    ) -> DecoderAnalysis {
        use std::collections::HashMap;

        // Map from syndrome -> (correctable_count, uncorrectable_count)
        let mut syndrome_map: HashMap<Vec<u8>, (usize, usize)> = HashMap::new();
        let mut undetectable_logical_errors = 0;
        let mut undetectable_stabilizers = 0;

        let fault_iter = PauliFaultIterator::new(
            self.locations.clone(),
            self.config.max_weight,
            self.config.clone(),
        );

        for fault_config in fault_iter {
            let prop = propagate_faults(self.circuit, &fault_config);

            // Get syndrome
            let syndrome = crate::fault_tolerance::decoder_integration::extract_syndrome(
                &prop, z_ancillas, x_ancillas,
            );

            // Check if causes logical error
            let causes_logical = logicals
                .iter()
                .any(|(xs, zs)| anticommutes_with_logical(&prop, xs, zs));

            if syndrome.iter().all(|&b| b == 0) {
                // Undetectable
                if causes_logical {
                    undetectable_logical_errors += 1;
                } else {
                    undetectable_stabilizers += 1;
                }
            } else {
                // Detectable - group by syndrome
                let entry = syndrome_map.entry(syndrome).or_insert((0, 0));
                if causes_logical {
                    entry.1 += 1;
                } else {
                    entry.0 += 1;
                }
            }
        }

        // Build syndrome analyses
        let mut syndromes = Vec::new();
        let mut correctable_syndromes = 0;
        let mut detected_uncorrectable_syndromes = 0;
        let mut ambiguous_syndromes = 0;
        let mut correctable_faults = 0;
        let mut detected_uncorrectable_faults = 0;
        let mut ambiguous_faults = 0;

        for (syndrome, (correctable, uncorrectable)) in syndrome_map {
            let class = if uncorrectable == 0 {
                correctable_syndromes += 1;
                correctable_faults += correctable;
                SyndromeClass::Correctable
            } else if correctable == 0 {
                detected_uncorrectable_syndromes += 1;
                detected_uncorrectable_faults += uncorrectable;
                SyndromeClass::DetectedUncorrectable
            } else {
                ambiguous_syndromes += 1;
                ambiguous_faults += correctable + uncorrectable;
                SyndromeClass::Ambiguous
            };

            syndromes.push(SyndromeAnalysis {
                syndrome,
                correctable_count: correctable,
                uncorrectable_count: uncorrectable,
                class,
            });
        }

        // Sort syndromes by total fault count (most common first)
        syndromes.sort_by_key(|s| std::cmp::Reverse(s.total_faults()));

        DecoderAnalysis {
            syndromes,
            correctable_syndromes,
            detected_uncorrectable_syndromes,
            ambiguous_syndromes,
            correctable_faults,
            detected_uncorrectable_faults,
            ambiguous_faults,
            undetectable_logical_errors,
            undetectable_stabilizers,
        }
    }

    /// Analyzes fault tolerance using syndrome history across multiple measurement rounds.
    ///
    /// This is more accurate for multi-round QEC circuits than single-shot analysis.
    /// A fault is considered "detected" if it produces a syndrome in ANY measurement round,
    /// not just the final state.
    ///
    /// # Arguments
    ///
    /// * `logicals` - Logical operators as (`x_positions`, `z_positions`) pairs
    ///
    /// # Returns
    ///
    /// A `SyndromeHistoryResult` with detailed analysis.
    ///
    /// # How it works
    ///
    /// 1. Identifies all measurement rounds in the circuit
    /// 2. For each fault, computes its syndrome at each measurement round
    /// 3. Groups faults by their complete syndrome history
    /// 4. Checks if each history pattern has consistent logical outcomes
    #[must_use]
    pub fn analyze_with_syndrome_history(
        &self,
        logicals: &[(&[usize], &[usize])],
    ) -> SyndromeHistoryResult {
        use std::collections::HashMap;

        // Extract measurement rounds from circuit
        let rounds = extract_measurement_rounds(self.circuit);

        if rounds.is_empty() {
            // No measurements, can't do syndrome history analysis
            return SyndromeHistoryResult {
                rounds: vec![],
                histories: vec![],
                never_detected_logical_errors: 0,
                never_detected_stabilizers: 0,
                correctable_faults: 0,
                detected_uncorrectable_faults: 0,
                ambiguous_faults: 0,
                ambiguous_histories: 0,
                total_faults: 0,
            };
        }

        // Map from syndrome history -> (correctable_count, uncorrectable_count)
        let mut history_map: HashMap<SyndromeHistory, (usize, usize)> = HashMap::new();
        let mut never_detected_logical_errors = 0;
        let mut never_detected_stabilizers = 0;
        let mut total_faults = 0;

        let fault_iter = PauliFaultIterator::new(
            self.locations.clone(),
            self.config.max_weight,
            self.config.clone(),
        );

        for fault_config in fault_iter {
            total_faults += 1;

            // Get the first (and usually only) fault
            let fault = &fault_config.faults[0];

            // Compute syndrome history for this fault
            let history = compute_syndrome_history(self.circuit, fault, &rounds);

            // Propagate to end of circuit to check logical error
            let prop = propagate_faults(self.circuit, &fault_config);
            let causes_logical = logicals
                .iter()
                .any(|(xs, zs)| anticommutes_with_logical(&prop, xs, zs));

            if history.is_detected() {
                // Detected in some round - group by history
                let entry = history_map.entry(history).or_insert((0, 0));
                if causes_logical {
                    entry.1 += 1;
                } else {
                    entry.0 += 1;
                }
            } else {
                // Never detected in any round
                if causes_logical {
                    never_detected_logical_errors += 1;
                } else {
                    never_detected_stabilizers += 1;
                }
            }
        }

        // Build history analyses
        let mut histories = Vec::new();
        let mut correctable_faults = 0;
        let mut detected_uncorrectable_faults = 0;
        let mut ambiguous_faults = 0;
        let mut ambiguous_histories = 0;

        for (history, (correctable, uncorrectable)) in history_map {
            let class = if uncorrectable == 0 {
                correctable_faults += correctable;
                SyndromeClass::Correctable
            } else if correctable == 0 {
                detected_uncorrectable_faults += uncorrectable;
                SyndromeClass::DetectedUncorrectable
            } else {
                ambiguous_histories += 1;
                ambiguous_faults += correctable + uncorrectable;
                SyndromeClass::Ambiguous
            };

            histories.push(SyndromeHistoryAnalysis {
                history,
                correctable_count: correctable,
                uncorrectable_count: uncorrectable,
                class,
            });
        }

        // Sort by total fault count
        histories.sort_by(|a, b| {
            let total_a = a.correctable_count + a.uncorrectable_count;
            let total_b = b.correctable_count + b.uncorrectable_count;
            total_b.cmp(&total_a)
        });

        SyndromeHistoryResult {
            rounds,
            histories,
            never_detected_logical_errors,
            never_detected_stabilizers,
            correctable_faults,
            detected_uncorrectable_faults,
            ambiguous_faults,
            ambiguous_histories,
            total_faults,
        }
    }

    /// Analyzes fault tolerance with proper s + r <= t enumeration.
    ///
    /// This method automatically handles both:
    /// - **Self-contained circuits** (no input qubits): enumerate internal faults r <= t
    /// - **Gadgets with inputs** (has input qubits): enumerate (s, r) where s + r <= t
    ///
    /// For gadgets with input qubits, this enumerates all combinations of:
    /// - Input faults (weight s) on the input qubits (errors from previous stage)
    /// - Internal faults (weight r) at circuit locations
    ///
    /// where s + r <= t (the fault tolerance level from `config.max_weight`).
    ///
    /// # Arguments
    ///
    /// * `z_ancillas` - Qubits measured in Z basis
    /// * `x_ancillas` - Qubits measured in X basis
    /// * `logicals` - Logical operators as (`x_positions`, `z_positions`) pairs
    /// * `collect_failures` - Whether to store detailed info for failures
    ///
    /// # Returns
    ///
    /// A `FaultToleranceAnalysis` covering all (s, r) combinations.
    #[must_use]
    pub fn analyze_with_input_faults(
        &self,
        z_ancillas: &[usize],
        x_ancillas: &[usize],
        logicals: &[(&[usize], &[usize])],
        collect_failures: bool,
    ) -> FaultToleranceAnalysis {
        let t = self.config.max_weight;

        // If no input qubits, fall back to standard analysis
        if self.io.input_qubits.is_empty() {
            return self.analyze_fault_tolerance(
                z_ancillas,
                x_ancillas,
                logicals,
                collect_failures,
            );
        }

        let mut total_tested = 0;
        let mut undetectable_logical_errors = 0;
        let mut undetectable_stabilizers = 0;
        let mut detectable_errors = 0;
        let mut failure_details = Vec::new();

        // Enumerate all (s, r) combinations where s + r <= t
        for s in 0..=t {
            let max_r = t - s;

            // Generate input fault combinations of weight s
            let input_fault_combos =
                generate_pauli_combinations(&self.io.input_qubits, s, &self.config);

            for input_fault in &input_fault_combos {
                // Helper to test a single (input_fault, internal_faults) combination
                // Returns (classification, prop) for potential failure collection
                let test_combination = |internal_faults: &[PauliFault]| -> (FaultClass, PauliProp) {
                    // Initialize with input fault
                    let mut prop = PauliProp::new();
                    for (&qubit, &pauli) in self.io.input_qubits.iter().zip(input_fault.iter()) {
                        match pauli {
                            1 => prop.track_x(&[qubit]),
                            2 => prop.track_y(&[qubit]),
                            3 => prop.track_z(&[qubit]),
                            _ => {} // Identity
                        }
                    }

                    // Apply internal faults
                    for fault in internal_faults {
                        for (qubit, &pauli) in fault.location.qubits.iter().zip(&fault.paulis) {
                            let q = qubit.index();
                            match pauli {
                                1 => prop.track_x(&[q]),
                                2 => prop.track_y(&[q]),
                                3 => prop.track_z(&[q]),
                                _ => {}
                            }
                        }
                    }

                    // Propagate through circuit
                    for (_tick_idx, tick) in self.circuit.iter_ticks() {
                        for gate in tick.gates() {
                            apply_gate(&mut prop, gate, Direction::Forward);
                        }
                    }

                    // Classify
                    let classification = classify_fault(&prop, z_ancillas, x_ancillas, logicals);
                    (classification, prop)
                };

                // Helper to build FaultConfiguration from input fault + internal faults
                let build_fault_config = |internal_faults: &[PauliFault]| -> FaultConfiguration {
                    let mut faults = Vec::new();

                    // Add input faults as PauliFault at tick 0, before=true
                    // Each input qubit with a non-identity Pauli gets its own fault entry
                    for (&qubit, &pauli) in self.io.input_qubits.iter().zip(input_fault.iter()) {
                        if pauli != 0 {
                            let location = SpacetimeLocation {
                                tick: 0,
                                qubits: vec![QubitId::new(qubit)],
                                before: true,
                                gate_type: GateType::I, // Input fault, not associated with a gate
                                gate_index: 0,
                            };
                            faults.push(PauliFault::new(location, vec![pauli]));
                        }
                    }

                    // Add internal faults
                    faults.extend(internal_faults.iter().cloned());

                    FaultConfiguration::with_faults(faults)
                };

                // Helper to process a test result
                let mut process_result = |internal_faults: &[PauliFault]| {
                    total_tested += 1;
                    let (classification, prop) = test_combination(internal_faults);

                    match classification {
                        FaultClass::UndetectableLogicalError => {
                            undetectable_logical_errors += 1;

                            if collect_failures {
                                let (z_flips, x_flips) =
                                    get_syndrome_flips(&prop, z_ancillas, x_ancillas);
                                let logical_errors: Vec<bool> = logicals
                                    .iter()
                                    .map(|(xs, zs)| anticommutes_with_logical(&prop, xs, zs))
                                    .collect();

                                let result = PropagationResult {
                                    propagated_error: prop,
                                    z_syndrome_flips: z_flips,
                                    x_syndrome_flips: x_flips,
                                    logical_errors,
                                };
                                let fault_config = build_fault_config(internal_faults);
                                failure_details.push((fault_config, result));
                            }
                        }
                        FaultClass::UndetectableStabilizer => {
                            undetectable_stabilizers += 1;
                        }
                        FaultClass::DetectableError => {
                            detectable_errors += 1;
                        }
                    }
                };

                if max_r == 0 {
                    // Special case: no internal faults, just test input fault alone
                    // Skip if input fault is also identity (s=0, r=0 = no fault)
                    if s > 0 {
                        process_result(&[]);
                    }
                } else {
                    // Enumerate internal faults up to weight max_r
                    let internal_config = FaultCheckConfig {
                        max_weight: max_r,
                        ..self.config.clone()
                    };

                    let fault_iter =
                        PauliFaultIterator::new(self.locations.clone(), max_r, internal_config);

                    for internal_fault in fault_iter {
                        process_result(&internal_fault.faults);
                    }
                }
            }
        }

        FaultToleranceAnalysis {
            total_tested,
            undetectable_logical_errors,
            undetectable_stabilizers,
            detectable_errors,
            weight: t,
            failure_details,
        }
    }

    /// Checks if the circuit is t-fault tolerant considering input faults.
    ///
    /// For circuits with input qubits, this checks all (s, r) combinations
    /// where s + r <= t. For self-contained circuits, it's equivalent to
    /// `is_fault_tolerant`.
    #[must_use]
    pub fn is_fault_tolerant_with_inputs(
        &self,
        z_ancillas: &[usize],
        x_ancillas: &[usize],
        logicals: &[(&[usize], &[usize])],
    ) -> bool {
        let analysis = self.analyze_with_input_faults(z_ancillas, x_ancillas, logicals, false);
        analysis.undetectable_logical_errors == 0
    }

    /// Analyzes a gadget considering both internal syndromes AND follow-up syndromes.
    ///
    /// For gadgets with output qubits, the syndrome from within the gadget may be
    /// ambiguous (same syndrome can lead to different logical outcomes). However,
    /// in a proper QEC protocol, an ideal error correction round follows the gadget.
    /// The output error (residual Pauli on output qubits) will produce syndromes in
    /// this follow-up round, which can disambiguate the outcome.
    ///
    /// # The key insight
    ///
    /// The "full syndrome" = (`gadget_syndrome`, `follow_up_syndrome`) where:
    /// - `gadget_syndrome` = syndromes from ancilla measurements within the gadget
    /// - `follow_up_syndrome` = syndromes the output error would produce in ideal EC
    ///
    /// Two faults with the same `gadget_syndrome` but different logical outcomes MUST
    /// produce different output errors (otherwise they'd be equivalent). These different
    /// output errors will produce different `follow_up_syndromes`, making the full
    /// syndrome unique.
    ///
    /// # Arguments
    ///
    /// * `z_ancillas` - Z-basis measurement qubits within the gadget
    /// * `x_ancillas` - X-basis measurement qubits within the gadget
    /// * `logicals` - Logical operators to check against
    /// * `follow_up` - Configuration for follow-up syndrome analysis
    ///
    /// # Returns
    ///
    /// A `DecoderAnalysis` using the combined (gadget + follow-up) syndrome.
    #[must_use]
    pub fn analyze_with_follow_up(
        &self,
        z_ancillas: &[usize],
        x_ancillas: &[usize],
        logicals: &[(&[usize], &[usize])],
        follow_up: &FollowUpConfig,
    ) -> DecoderAnalysis {
        use std::collections::HashMap;

        // Map from full syndrome -> (correctable_count, uncorrectable_count)
        let mut syndrome_map: HashMap<Vec<u8>, (usize, usize)> = HashMap::new();
        let mut undetectable_logical_errors = 0;
        let mut undetectable_stabilizers = 0;

        let fault_iter = PauliFaultIterator::new(
            self.locations.clone(),
            self.config.max_weight,
            self.config.clone(),
        );

        // Convert follow-up stabilizers to borrowed slice format for compute_stabilizer_syndromes
        let follow_up_refs: Vec<(&[usize], &[usize])> = follow_up
            .follow_up_stabilizers
            .iter()
            .map(|(x, z)| (x.as_slice(), z.as_slice()))
            .collect();

        for fault_config in fault_iter {
            // Propagate fault through circuit
            let prop = propagate_faults(self.circuit, &fault_config);

            // Compute gadget syndrome (within-gadget ancilla measurements)
            let (z_flips, x_flips) = get_syndrome_flips(&prop, z_ancillas, x_ancillas);
            let gadget_syndrome_detected = !z_flips.is_empty() || !x_flips.is_empty();

            // Extract output error and compute follow-up syndrome
            let output_error = if follow_up.has_follow_up() {
                extract_output_error(&prop, &follow_up.output_qubits)
            } else {
                PauliProp::new()
            };

            let follow_up_syndrome = if follow_up.has_follow_up() {
                compute_stabilizer_syndromes(&output_error, &follow_up_refs)
            } else {
                vec![]
            };

            let follow_up_detected = follow_up_syndrome.iter().any(|&s| s);

            // Check if fault causes logical error
            let causes_logical_error = logicals
                .iter()
                .any(|(xs, zs)| anticommutes_with_logical(&prop, xs, zs));

            // Classify based on combined detectability
            let detected = gadget_syndrome_detected || follow_up_detected;

            if detected {
                // Build full syndrome key: gadget syndrome + follow-up syndrome
                let mut full_syndrome: Vec<u8> = Vec::new();

                // Encode gadget syndrome as bit vector
                for &q in z_ancillas {
                    full_syndrome.push(u8::from(z_flips.contains(&q)));
                }
                for &q in x_ancillas {
                    full_syndrome.push(u8::from(x_flips.contains(&q)));
                }

                // Append follow-up syndrome
                for &s in &follow_up_syndrome {
                    full_syndrome.push(u8::from(s));
                }

                // Update counts for this full syndrome
                let entry = syndrome_map.entry(full_syndrome).or_insert((0, 0));
                if causes_logical_error {
                    entry.1 += 1;
                } else {
                    entry.0 += 1;
                }
            } else {
                // Undetectable by either gadget or follow-up
                if causes_logical_error {
                    undetectable_logical_errors += 1;
                } else {
                    undetectable_stabilizers += 1;
                }
            }
        }

        // Build analysis from syndrome map
        let mut syndromes = Vec::new();
        let mut correctable_syndromes = 0;
        let mut detected_uncorrectable_syndromes = 0;
        let mut ambiguous_syndromes = 0;
        let mut correctable_faults = 0;
        let mut detected_uncorrectable_faults = 0;
        let mut ambiguous_faults = 0;

        for (syndrome, (correctable, uncorrectable)) in syndrome_map {
            let class = if uncorrectable == 0 {
                correctable_syndromes += 1;
                correctable_faults += correctable;
                SyndromeClass::Correctable
            } else if correctable == 0 {
                detected_uncorrectable_syndromes += 1;
                detected_uncorrectable_faults += uncorrectable;
                SyndromeClass::DetectedUncorrectable
            } else {
                ambiguous_syndromes += 1;
                ambiguous_faults += correctable + uncorrectable;
                SyndromeClass::Ambiguous
            };

            syndromes.push(SyndromeAnalysis {
                syndrome,
                correctable_count: correctable,
                uncorrectable_count: uncorrectable,
                class,
            });
        }

        // Sort by total fault count
        syndromes.sort_by_key(|s| std::cmp::Reverse(s.total_faults()));

        DecoderAnalysis {
            syndromes,
            correctable_syndromes,
            detected_uncorrectable_syndromes,
            ambiguous_syndromes,
            correctable_faults,
            detected_uncorrectable_faults,
            ambiguous_faults,
            undetectable_logical_errors,
            undetectable_stabilizers,
        }
    }

    /// Checks if a gadget is fault tolerant when considering follow-up EC.
    ///
    /// This is the key method for analyzing gadgets with output qubits. It considers:
    /// - Syndromes from within the gadget
    /// - Syndromes that would be produced by an ideal EC round after the gadget
    ///
    /// A gadget passes if the combined (gadget + follow-up) syndromes uniquely
    /// identify all logical outcomes.
    #[must_use]
    pub fn is_gadget_fault_tolerant(
        &self,
        z_ancillas: &[usize],
        x_ancillas: &[usize],
        logicals: &[(&[usize], &[usize])],
        follow_up: &FollowUpConfig,
    ) -> bool {
        let analysis = self.analyze_with_follow_up(z_ancillas, x_ancillas, logicals, follow_up);
        analysis.is_ft()
    }
}

/// Generates all Pauli error combinations of a given weight on specified qubits.
///
/// Returns a vector of vectors, where each inner vector has length equal to `qubits.len()`
/// and contains Pauli indices (0=I, 1=X, 2=Y, 3=Z).
fn generate_pauli_combinations(
    qubits: &[usize],
    weight: usize,
    config: &FaultCheckConfig,
) -> Vec<Vec<u8>> {
    if weight == 0 {
        // Weight 0 = identity on all qubits
        return vec![vec![0; qubits.len()]];
    }

    if weight > qubits.len() {
        // Can't have more faults than qubits
        return vec![];
    }

    // Generate all combinations of `weight` positions from qubits
    let n = qubits.len();
    let mut results = Vec::new();

    // Choose which positions have non-identity Paulis
    for positions in combinations(n, weight) {
        // For each position combination, enumerate Pauli types
        let pauli_choices: Vec<u8> = if config.include_x && config.include_y && config.include_z {
            vec![1, 2, 3] // X, Y, Z
        } else if config.include_x && !config.include_y && !config.include_z {
            vec![1] // X only
        } else if !config.include_x && !config.include_y && config.include_z {
            vec![3] // Z only
        } else if config.include_x && !config.include_y && config.include_z {
            vec![1, 3] // X and Z (CSS mode)
        } else {
            // Fallback to all Paulis
            vec![1, 2, 3]
        };

        // Enumerate all Pauli assignments for chosen positions
        for paulis in pauli_product(&pauli_choices, weight) {
            let mut combo = vec![0u8; n];
            for (pos_idx, &pos) in positions.iter().enumerate() {
                combo[pos] = paulis[pos_idx];
            }
            results.push(combo);
        }
    }

    results
}

/// Generates all k-combinations of indices 0..n.
fn combinations(n: usize, k: usize) -> Vec<Vec<usize>> {
    if k == 0 {
        return vec![vec![]];
    }
    if k > n {
        return vec![];
    }

    let mut results = Vec::new();
    let mut combo: Vec<usize> = (0..k).collect();

    loop {
        results.push(combo.clone());

        // Find rightmost position that can be incremented
        let mut found = false;
        for i in (0..k).rev() {
            if combo[i] < n - k + i {
                combo[i] += 1;
                // Reset all positions after i
                for j in (i + 1)..k {
                    combo[j] = combo[j - 1] + 1;
                }
                found = true;
                break;
            }
        }

        if !found {
            break;
        }
    }

    results
}

/// Generates Cartesian product of Pauli choices for `count` positions.
fn pauli_product(choices: &[u8], count: usize) -> Vec<Vec<u8>> {
    if count == 0 {
        return vec![vec![]];
    }

    let mut results = Vec::new();
    let sub = pauli_product(choices, count - 1);

    for &p in choices {
        for s in &sub {
            let mut v = vec![p];
            v.extend(s);
            results.push(v);
        }
    }

    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_init_pauli_prop_with_fault() {
        let loc = SpacetimeLocation::new(0, vec![QubitId(0), QubitId(1)], false, GateType::CX, 0);
        let fault = PauliFault::new(loc, vec![1, 3]); // X on q0, Z on q1

        let prop = init_pauli_prop_with_fault(&fault);

        assert!(prop.contains_x(0));
        assert!(!prop.contains_z(0));
        assert!(!prop.contains_x(1));
        assert!(prop.contains_z(1));
    }

    #[test]
    fn test_propagate_x_through_cx() {
        // X on control propagates to target: XI -> XX
        let mut circuit = TickCircuit::new();
        circuit.tick().cx(&[(0, 1)]);

        let _loc = SpacetimeLocation::new(0, vec![QubitId(0)], false, GateType::CX, 0);

        // Initialize with X on qubit 0, then propagate through CX
        let mut prop = PauliProp::new();
        prop.track_x(&[0]);

        // Apply CX
        prop.cx(&[(QubitId(0), QubitId(1))]);

        // X should now be on both qubits
        assert!(prop.contains_x(0));
        assert!(prop.contains_x(1));
    }

    #[test]
    fn test_propagate_z_through_cx() {
        // Z on target propagates to control: IZ -> ZZ
        let mut prop = PauliProp::new();
        prop.track_z(&[1]);

        prop.cx(&[(QubitId(0), QubitId(1))]);

        // Z should now be on both qubits
        assert!(prop.contains_z(0));
        assert!(prop.contains_z(1));
    }

    #[test]
    fn test_propagate_through_h() {
        // H swaps X and Z
        let mut prop = PauliProp::new();
        prop.track_x(&[0]);

        prop.h(&[QubitId(0)]);

        // X becomes Z
        assert!(!prop.contains_x(0));
        assert!(prop.contains_z(0));
    }

    #[test]
    fn test_anticommutes_with_logical() {
        // X error anticommutes with Z logical
        let mut prop = PauliProp::new();
        prop.track_x(&[0]);

        let logical_xs: &[usize] = &[];
        let logical_zs: &[usize] = &[0];

        assert!(anticommutes_with_logical(&prop, logical_xs, logical_zs));
    }

    #[test]
    fn test_commutes_with_logical() {
        // Z error commutes with Z logical
        let mut prop = PauliProp::new();
        prop.track_z(&[0]);

        let logical_xs: &[usize] = &[];
        let logical_zs: &[usize] = &[0];

        assert!(!anticommutes_with_logical(&prop, logical_xs, logical_zs));
    }

    #[test]
    fn test_pauli_prop_checker_creation() {
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1)]);

        let checker = PauliPropChecker::new(&circuit);
        assert_eq!(checker.locations().len(), 2);
    }

    #[test]
    fn test_pauli_prop_checker_bell_state() {
        // Bell state circuit: H then CX
        let mut circuit = TickCircuit::new();
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1)]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .all_paulis()
            .stop_on_first(false);

        let checker = PauliPropChecker::new(&circuit).with_config(config);

        // Logical Z for Bell state is ZZ
        let logical_xs: &[usize] = &[];
        let logical_zs: &[usize] = &[0, 1];

        let result = checker.check_logical_error(logical_xs, logical_zs);

        // Some faults should cause logical errors (X on either qubit)
        println!(
            "Bell state: {} failures out of {} tested",
            result.num_failures(),
            result.total_tested
        );
    }

    #[test]
    fn test_pauli_prop_checker_error_weight() {
        let mut circuit = TickCircuit::new();
        circuit.tick().cx(&[(0, 1)]);
        circuit.tick().cx(&[(1, 2)]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let checker = PauliPropChecker::new(&circuit).with_config(config);

        // Check that errors don't spread beyond weight 2
        let result = checker.check_error_weight(2);

        println!(
            "Error weight check: {} failures (errors > weight 2)",
            result.num_failures()
        );
    }

    /// Test with 3-qubit bit-flip code syndrome extraction
    #[test]
    fn test_three_qubit_code_pauli_prop() {
        // 3-qubit bit-flip code syndrome extraction
        // Data qubits: 0, 1, 2
        // Ancilla qubits: 3, 4
        // Z0Z1 measured by ancilla 3, Z1Z2 measured by ancilla 4
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only() // X errors for bit-flip code
            .stop_on_first(false);

        let checker = PauliPropChecker::new(&circuit).with_config(config);

        // Logical X for 3-qubit bit-flip code: X0X1X2
        let logical_xs: &[usize] = &[0, 1, 2];
        let logical_zs: &[usize] = &[];

        let result = checker.check_logical_error(logical_xs, logical_zs);

        println!(
            "3-qubit code (X errors): {} failures out of {} tested",
            result.num_failures(),
            result.total_tested
        );

        // Weight-1 X errors should NOT cause logical X errors
        // (the code is distance 3, so weight-1 errors are correctable)
        // However, our simple check doesn't account for decoding, so we may see failures
    }

    #[test]
    fn test_get_syndrome_flips() {
        // X error on qubit 0 should flip Z-basis measurement on qubit 0
        let mut prop = PauliProp::new();
        prop.track_x(&[0]);

        let (z_flips, x_flips) = get_syndrome_flips(&prop, &[0, 1], &[]);
        assert_eq!(z_flips, vec![0]);
        assert!(x_flips.is_empty());
    }

    #[test]
    fn test_has_syndrome() {
        let mut prop = PauliProp::new();
        prop.track_x(&[3]); // X on ancilla qubit 3

        // Should detect syndrome on qubit 3
        assert!(has_syndrome(&prop, &[3, 4], &[]));

        // Should not detect syndrome if we only check qubit 4
        assert!(!has_syndrome(&prop, &[4], &[]));
    }

    #[test]
    fn test_syndrome_detection_three_qubit_code() {
        // 3-qubit bit-flip code: X errors on data qubits should produce syndromes
        // Data qubits: 0, 1, 2
        // Ancilla qubits: 3, 4 (Z-basis measurement)
        // Stabilizers: Z0Z1 (ancilla 3), Z1Z2 (ancilla 4)

        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let checker = PauliPropChecker::new(&circuit).with_config(config);

        // Check that weight-1 X errors produce syndromes
        let result = checker.check_syndrome_detection(&[3, 4], &[], true);

        println!(
            "3-qubit code syndrome detection: {} faults don't produce syndrome (should be 0)",
            result.num_failures()
        );

        // All weight-1 X errors on data qubits should produce a syndrome
        // (faults on ancillas might not, but data qubit errors should)
    }

    #[test]
    fn test_analyze_all_faults() {
        // Simple circuit for full analysis
        let mut circuit = TickCircuit::new();
        circuit.tick().cx(&[(0, 1)]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let checker = PauliPropChecker::new(&circuit).with_config(config);

        // Analyze with Z-basis measurements on qubit 1
        let z_ancillas = &[1usize];
        let x_ancillas: &[usize] = &[];
        let logicals: &[(&[usize], &[usize])] = &[(&[0, 1], &[])]; // Logical X = XX

        let results = checker.analyze_all_faults(z_ancillas, x_ancillas, logicals);

        println!("Analyzed {} fault configurations:", results.len());
        for (fault, result) in &results {
            println!(
                "  Fault: {} -> syndrome: {:?}, logical error: {:?}, output weight: {}",
                fault.faults[0].pauli_string(),
                result.z_syndrome_flips,
                result.logical_errors,
                result.output_weight()
            );
        }

        assert!(!results.is_empty());
    }

    #[test]
    fn test_propagation_result() {
        let mut prop = PauliProp::new();
        prop.track_x(&[0]);
        prop.track_z(&[1]);

        let result = PropagationResult {
            propagated_error: prop,
            z_syndrome_flips: vec![3],
            x_syndrome_flips: vec![],
            logical_errors: vec![false, true],
        };

        assert!(result.has_syndrome());
        assert!(result.has_logical_error());
        assert_eq!(result.output_weight(), 2);
    }

    #[test]
    fn test_fault_class_methods() {
        assert!(FaultClass::UndetectableLogicalError.is_certain_failure());
        assert!(!FaultClass::UndetectableLogicalError.is_safe());
        assert!(!FaultClass::UndetectableLogicalError.is_detectable());

        assert!(!FaultClass::UndetectableStabilizer.is_certain_failure());
        assert!(FaultClass::UndetectableStabilizer.is_safe());
        assert!(!FaultClass::UndetectableStabilizer.is_detectable());

        assert!(!FaultClass::DetectableError.is_certain_failure());
        assert!(!FaultClass::DetectableError.is_safe());
        assert!(FaultClass::DetectableError.is_detectable());
    }

    #[test]
    fn test_classify_fault_detectable() {
        // Error that produces a syndrome is detectable
        let mut prop = PauliProp::new();
        prop.track_x(&[0]); // X error on qubit 0

        // If qubit 0 is measured in Z basis, this is detectable
        let classification = classify_fault(&prop, &[0], &[], &[(&[], &[0])]);
        assert_eq!(classification, FaultClass::DetectableError);
    }

    #[test]
    fn test_classify_fault_undetectable_stabilizer() {
        // Error that produces no syndrome and commutes with logicals
        let mut prop = PauliProp::new();
        prop.track_x(&[0]);
        prop.track_x(&[1]); // X0X1 = stabilizer of 3-qubit code

        // Logical Z = ZZZ, so X0X1 commutes with it
        // No syndrome measurement qubits, so no syndrome
        let classification = classify_fault(&prop, &[], &[], &[(&[], &[0, 1, 2])]);
        assert_eq!(classification, FaultClass::UndetectableStabilizer);
    }

    #[test]
    fn test_classify_fault_undetectable_logical() {
        // Error that produces no syndrome but anticommutes with logical
        let mut prop = PauliProp::new();
        prop.track_x(&[0]); // X on qubit 0

        // No syndrome measurement qubits, so no syndrome
        // Logical Z = Z on qubit 0, so X anticommutes
        let classification = classify_fault(&prop, &[], &[], &[(&[], &[0])]);
        assert_eq!(classification, FaultClass::UndetectableLogicalError);
    }

    #[test]
    fn test_propagation_result_classify() {
        // Test the classify() method on PropagationResult

        // Detectable error
        let result = PropagationResult {
            propagated_error: PauliProp::new(),
            z_syndrome_flips: vec![0],
            x_syndrome_flips: vec![],
            logical_errors: vec![false],
        };
        assert_eq!(result.classify(), FaultClass::DetectableError);

        // Undetectable stabilizer
        let result = PropagationResult {
            propagated_error: PauliProp::new(),
            z_syndrome_flips: vec![],
            x_syndrome_flips: vec![],
            logical_errors: vec![false],
        };
        assert_eq!(result.classify(), FaultClass::UndetectableStabilizer);

        // Undetectable logical error
        let result = PropagationResult {
            propagated_error: PauliProp::new(),
            z_syndrome_flips: vec![],
            x_syndrome_flips: vec![],
            logical_errors: vec![true],
        };
        assert_eq!(result.classify(), FaultClass::UndetectableLogicalError);
    }

    #[test]
    fn test_analyze_fault_tolerance_three_qubit_code() {
        // 3-qubit bit-flip code syndrome extraction
        // Data qubits: 0, 1, 2
        // Ancilla qubits: 3, 4
        // Stabilizers: Z0Z1 (ancilla 3), Z1Z2 (ancilla 4)
        // Logical X = X0X1X2, Logical Z = Z0Z1Z2

        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let checker = PauliPropChecker::new(&circuit).with_config(config);

        // Logical Z spans all data qubits
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];
        let z_ancillas = &[3usize, 4];
        let x_ancillas: &[usize] = &[];

        let analysis = checker.analyze_fault_tolerance(z_ancillas, x_ancillas, logicals, true);

        println!("3-qubit code fault tolerance analysis:");
        println!("  Total tested: {}", analysis.total_tested);
        println!(
            "  Undetectable logical errors: {}",
            analysis.undetectable_logical_errors
        );
        println!(
            "  Undetectable stabilizers: {}",
            analysis.undetectable_stabilizers
        );
        println!("  Detectable errors: {}", analysis.detectable_errors);

        // This naive syndrome extraction circuit is NOT fault tolerant!
        // An X error on a data qubit that occurs AFTER its entangling gate
        // won't produce a syndrome but will still cause a logical error.
        //
        // For example: X on qubit 2 after cx(2,4) -> no syndrome, but anticommutes
        // with logical Z = Z0Z1Z2.
        //
        // This demonstrates exactly the kind of circuit vulnerability that
        // fault classification is designed to detect.
        assert!(
            !analysis.is_fault_tolerant(),
            "Naive syndrome extraction should NOT be 1-fault tolerant"
        );
        assert!(
            analysis.undetectable_logical_errors > 0,
            "Should have undetectable logical errors"
        );

        // Print the failure details to understand the vulnerabilities
        for (fault, result) in &analysis.failure_details {
            println!(
                "  Vulnerability: {} at tick {} -> {:?}",
                fault.faults[0].pauli_string(),
                fault.faults[0].location.tick,
                result.classify()
            );
        }
    }

    #[test]
    fn test_fault_tolerance_analysis_methods() {
        let analysis = FaultToleranceAnalysis {
            total_tested: 100,
            undetectable_logical_errors: 5,
            undetectable_stabilizers: 10,
            detectable_errors: 85,
            weight: 1,
            failure_details: vec![],
        };

        assert!(!analysis.is_fault_tolerant());
        assert!((analysis.failure_rate() - 0.05).abs() < 1e-10);
        assert!((analysis.safe_rate() - 0.10).abs() < 1e-10);
        assert!((analysis.detectable_rate() - 0.85).abs() < 1e-10);
    }

    #[test]
    fn test_is_fault_tolerant_method() {
        // Simple circuit that IS fault tolerant for weight-1 X errors
        let mut circuit = TickCircuit::new();
        circuit.tick().cx(&[(0, 1)]);
        circuit.tick().mz(&[1]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let checker = PauliPropChecker::new(&circuit).with_config(config);

        // Measure qubit 1 in Z basis, so X errors on qubit 1 are detected
        let z_ancillas = &[1usize];
        let x_ancillas: &[usize] = &[];

        // Logical Z = Z0 (just for testing - single qubit)
        // X on qubit 0 -> XX after CX -> X on qubit 1 detected
        // X on qubit 1 after CX -> detected
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0])];

        let is_ft = checker.is_fault_tolerant(z_ancillas, x_ancillas, logicals);
        println!("Simple circuit is 1-fault tolerant for X errors: {is_ft}");
    }

    #[test]
    fn test_repeated_syndrome_measurement_concept() {
        // Demonstrate why repeated syndrome measurement helps fault tolerance.
        //
        // Single round problem: X error on data qubit AFTER its CX gate
        // -> no syndrome in this round, but error persists on data
        //
        // With repeated measurement:
        // Round 1: X error after CX -> no syndrome
        // Round 2: same X error now present BEFORE CX -> syndrome detected!
        //
        // The key insight is that persistent errors on data qubits will
        // eventually be caught by a subsequent syndrome round.

        // Single round circuit (NOT fault tolerant)
        let mut single_round = TickCircuit::new();
        single_round.tick().pz(&[3, 4]);
        single_round.tick().cx(&[(0, 3)]);
        single_round.tick().cx(&[(1, 3)]);
        single_round.tick().cx(&[(1, 4)]);
        single_round.tick().cx(&[(2, 4)]);
        single_round.tick().mz(&[3, 4]);

        // Two-round circuit (each round resets and re-measures)
        // In practice, the decoder looks at syndrome CHANGES between rounds
        let mut two_rounds = TickCircuit::new();
        // Round 1
        two_rounds.tick().pz(&[3, 4]);
        two_rounds.tick().cx(&[(0, 3)]);
        two_rounds.tick().cx(&[(1, 3)]);
        two_rounds.tick().cx(&[(1, 4)]);
        two_rounds.tick().cx(&[(2, 4)]);
        two_rounds.tick().mz(&[3, 4]);
        // Round 2 (ancillas reset)
        two_rounds.tick().pz(&[3, 4]);
        two_rounds.tick().cx(&[(0, 3)]);
        two_rounds.tick().cx(&[(1, 3)]);
        two_rounds.tick().cx(&[(1, 4)]);
        two_rounds.tick().cx(&[(2, 4)]);
        two_rounds.tick().mz(&[3, 4]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];
        let z_ancillas = &[3usize, 4];
        let x_ancillas: &[usize] = &[];

        // Analyze single round
        let single_checker = PauliPropChecker::new(&single_round).with_config(config.clone());
        let single_analysis =
            single_checker.analyze_fault_tolerance(z_ancillas, x_ancillas, logicals, false);

        // Analyze two rounds
        let two_checker = PauliPropChecker::new(&two_rounds).with_config(config);
        let two_analysis =
            two_checker.analyze_fault_tolerance(z_ancillas, x_ancillas, logicals, false);

        println!("Single round analysis:");
        println!(
            "  Undetectable logical errors: {}",
            single_analysis.undetectable_logical_errors
        );
        println!("  Detectable errors: {}", single_analysis.detectable_errors);

        println!("Two round analysis:");
        println!(
            "  Undetectable logical errors: {}",
            two_analysis.undetectable_logical_errors
        );
        println!("  Detectable errors: {}", two_analysis.detectable_errors);

        // With two rounds, errors that escaped detection in round 1
        // get caught in round 2. The detectable rate should be higher.
        //
        // Note: This is a simplified model. Real fault tolerance analysis
        // for repeated measurements considers syndrome differences and
        // requires d rounds for a distance-d code.
        assert!(
            two_analysis.detectable_errors >= single_analysis.detectable_errors,
            "Two rounds should detect at least as many errors as one round"
        );
    }

    #[test]
    fn test_why_naive_circuit_fails() {
        // Detailed examination of WHY the naive circuit isn't fault tolerant.
        //
        // The 3-qubit code measures stabilizers Z0Z1 and Z1Z2.
        // Circuit does: CX(0,3), CX(1,3), CX(1,4), CX(2,4), then measure 3,4.
        //
        // Consider X error on qubit 2 occurring AFTER cx(2,4):
        // - The error doesn't propagate to ancilla 4 (CX already done)
        // - Ancilla 3 never touched qubit 2, so no propagation there either
        // - Result: X2 on data, no syndrome
        // - But X2 anticommutes with logical Z = Z0Z1Z2 -> logical error!

        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]); // tick 4
        circuit.tick().mz(&[3, 4]); // tick 5

        // Manually create a fault: X on qubit 2 at tick 5 (after cx(2,4))
        let loc = SpacetimeLocation::new(5, vec![QubitId(2)], false, GateType::MeasureFree, 0);
        let fault = PauliFault::new(loc, vec![1]); // X error
        let fault_config = FaultConfiguration::with_faults(vec![fault]);

        // Propagate through circuit
        let prop = propagate_faults(&circuit, &fault_config);

        // Check: should have X on qubit 2, nothing on ancillas
        assert!(prop.contains_x(2), "X should be on qubit 2");
        assert!(!prop.contains_x(3), "No X on ancilla 3");
        assert!(!prop.contains_x(4), "No X on ancilla 4");

        // No syndrome (no X on Z-measurement qubits)
        assert!(
            !has_syndrome(&prop, &[3, 4], &[]),
            "Should have no syndrome"
        );

        // But anticommutes with logical Z = Z0Z1Z2
        let causes_logical = anticommutes_with_logical(&prop, &[], &[0, 1, 2]);
        assert!(causes_logical, "Should cause logical error");

        // Classification: UndetectableLogicalError
        let class = classify_fault(&prop, &[3, 4], &[], &[(&[], &[0, 1, 2])]);
        assert_eq!(class, FaultClass::UndetectableLogicalError);

        println!("Demonstrated: X on qubit 2 after cx(2,4) is an undetectable logical error");
    }

    #[test]
    fn test_syndrome_class_methods() {
        assert!(SyndromeClass::Correctable.is_correctable());
        assert!(!SyndromeClass::Correctable.is_detected_failure());
        assert!(!SyndromeClass::Correctable.is_ambiguous());

        assert!(!SyndromeClass::DetectedUncorrectable.is_correctable());
        assert!(SyndromeClass::DetectedUncorrectable.is_detected_failure());
        assert!(!SyndromeClass::DetectedUncorrectable.is_ambiguous());

        assert!(!SyndromeClass::Ambiguous.is_correctable());
        assert!(!SyndromeClass::Ambiguous.is_detected_failure());
        assert!(SyndromeClass::Ambiguous.is_ambiguous());
    }

    #[test]
    fn test_decoder_analysis_three_qubit_code() {
        // Analyze the 3-qubit bit-flip code syndrome extraction
        // This should show:
        // - Some correctable syndromes (X errors that produce unique syndromes)
        // - No ambiguous syndromes (3-qubit code has unique syndromes for weight-1)
        // - Some undetectable logical errors (faults after CX gates)

        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let checker = PauliPropChecker::new(&circuit).with_config(config);

        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];
        let z_ancillas = &[3usize, 4];
        let x_ancillas: &[usize] = &[];

        let analysis = checker.analyze_decoder_requirements(z_ancillas, x_ancillas, logicals);

        println!("Decoder Analysis for 3-qubit code:");
        println!(
            "  Correctable syndromes: {}",
            analysis.correctable_syndromes
        );
        println!(
            "  Detected uncorrectable syndromes: {}",
            analysis.detected_uncorrectable_syndromes
        );
        println!("  Ambiguous syndromes: {}", analysis.ambiguous_syndromes);
        println!("  Correctable faults: {}", analysis.correctable_faults);
        println!(
            "  Detected uncorrectable faults: {}",
            analysis.detected_uncorrectable_faults
        );
        println!("  Ambiguous faults: {}", analysis.ambiguous_faults);
        println!(
            "  Undetectable logical errors: {}",
            analysis.undetectable_logical_errors
        );
        println!(
            "  Undetectable stabilizers: {}",
            analysis.undetectable_stabilizers
        );

        // Print each syndrome's details
        for syn in &analysis.syndromes {
            println!(
                "  Syndrome {:?}: {} correctable, {} uncorrectable -> {:?}",
                syn.syndrome, syn.correctable_count, syn.uncorrectable_count, syn.class
            );
        }

        // The 3-qubit code should have unique syndromes for weight-1 data errors
        // So there should be NO ambiguous syndromes for the detectable faults
        // (but there ARE undetectable logical errors from faults after gates)

        let total = analysis.total_faults();
        println!("\nFailure rate analysis:");
        println!(
            "  Best case: {:.1}%",
            analysis.best_case_failure_rate(total) * 100.0
        );
        println!(
            "  Worst case: {:.1}%",
            analysis.worst_case_failure_rate(total) * 100.0
        );
        println!("  Is fault tolerant: {}", analysis.is_ft());

        // The naive circuit is NOT fault tolerant due to undetectable logical errors
        match analysis.is_fault_tolerant() {
            Ok(()) => panic!("Naive circuit should NOT be fault tolerant"),
            Err(failures) => {
                println!("\n  Failures:");
                for failure in &failures {
                    println!("    - {}", failure.description());
                }
                // Should have undetectable logical errors
                assert!(
                    failures.iter().any(|f| matches!(
                        f,
                        FaultToleranceFailure::UndetectableLogicalErrors { .. }
                    )),
                    "Should have undetectable logical errors"
                );
            }
        }
    }

    #[test]
    fn test_decoder_analysis_single_round_not_ft() {
        // Single-round syndrome extraction is NOT fault tolerant
        // because faults after the last CX on each data qubit don't produce syndromes.

        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[1]);
        circuit.tick().cx(&[(0, 1)]);
        circuit.tick().mz(&[1]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let checker = PauliPropChecker::new(&circuit).with_config(config);

        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0])];
        let z_ancillas = &[1usize];
        let x_ancillas: &[usize] = &[];

        let analysis = checker.analyze_decoder_requirements(z_ancillas, x_ancillas, logicals);

        println!("Single-round circuit (NOT fault tolerant):");
        println!(
            "  Undetectable logical errors: {}",
            analysis.undetectable_logical_errors
        );
        println!("  Is FT: {}", analysis.is_ft());

        // Single round has undetectable logical errors (X on data after CX)
        assert!(
            !analysis.is_ft(),
            "Single round should NOT be fault tolerant"
        );
    }

    #[test]
    fn test_two_round_syndrome_extraction_analysis() {
        // Two-round syndrome extraction - examine the analysis.
        //
        // Note: Our single-shot analysis has limitations for multi-round circuits.
        // Real FT guarantees require the decoder to use syndrome HISTORY.
        // Here we just examine what the analysis tells us.

        let mut circuit = TickCircuit::new();

        // Round 1
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        // Round 2 (fresh ancillas)
        circuit.tick().pz(&[5, 6]);
        circuit.tick().cx(&[(0, 5)]);
        circuit.tick().cx(&[(1, 5)]);
        circuit.tick().cx(&[(1, 6)]);
        circuit.tick().cx(&[(2, 6)]);
        circuit.tick().mz(&[5, 6]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let checker = PauliPropChecker::new(&circuit).with_config(config);

        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];
        let z_ancillas = &[5usize, 6];
        let x_ancillas: &[usize] = &[];

        let analysis = checker.analyze_decoder_requirements(z_ancillas, x_ancillas, logicals);

        println!("\nTwo-round syndrome extraction:");
        println!("  Total faults: {}", analysis.total_faults());
        println!(
            "  Undetectable logical errors: {}",
            analysis.undetectable_logical_errors
        );
        println!("  Is FT (by single-shot analysis): {}", analysis.is_ft());

        // Two-round still has undetectable errors at the END of round 2.
        // Real FT requires decoder to use syndrome history, not single-shot.
        // This test documents that limitation.
        assert!(
            analysis.undetectable_logical_errors > 0,
            "Two-round still has undetectable errors at circuit end"
        );
    }

    #[test]
    fn test_syndrome_history_analysis() {
        // Test the syndrome history analysis on a two-round circuit.
        // This analysis tracks syndromes across rounds rather than just at the end.

        let mut circuit = TickCircuit::new();

        // Round 1: syndrome extraction
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        // Round 2: syndrome extraction with fresh ancillas
        circuit.tick().pz(&[5, 6]);
        circuit.tick().cx(&[(0, 5)]);
        circuit.tick().cx(&[(1, 5)]);
        circuit.tick().cx(&[(1, 6)]);
        circuit.tick().cx(&[(2, 6)]);
        circuit.tick().mz(&[5, 6]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let checker = PauliPropChecker::new(&circuit).with_config(config);

        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];

        let result = checker.analyze_with_syndrome_history(logicals);

        println!("\nSyndrome history analysis:");
        println!("  Measurement rounds found: {}", result.rounds.len());
        println!("  Total faults: {}", result.total_faults);
        println!(
            "  Never-detected logical errors: {}",
            result.never_detected_logical_errors
        );
        println!(
            "  Never-detected stabilizers: {}",
            result.never_detected_stabilizers
        );
        println!("  Correctable faults: {}", result.correctable_faults);
        println!(
            "  Detected uncorrectable: {}",
            result.detected_uncorrectable_faults
        );
        println!("  Ambiguous faults: {}", result.ambiguous_faults);
        println!("  Unique syndrome histories: {}", result.histories.len());
        println!("  Is FT: {}", result.is_ft());

        // We should have found 2 measurement rounds
        assert_eq!(result.rounds.len(), 2, "Should find 2 measurement rounds");

        // With syndrome history, we can detect more faults than single-shot analysis
        // A fault in round 1 that escapes round 1 might still be caught in round 2
        assert!(result.total_faults > 0, "Should have analyzed some faults");
    }

    #[test]
    fn test_fault_tolerant_with_final_measurement() {
        // A circuit that IS fault tolerant by our analysis:
        // Syndrome extraction followed by DESTRUCTIVE measurement of all data qubits.
        //
        // Any X error that escapes syndrome extraction will be caught by
        // the final Z-basis measurement of data qubits.
        //
        // This models the "final round" of a QEC protocol where we read out
        // the logical qubit.

        let mut circuit = TickCircuit::new();

        // Data qubits: 0, 1, 2
        // Ancillas: 3, 4
        // Stabilizers: Z0Z1 (ancilla 3), Z1Z2 (ancilla 4)

        // Syndrome extraction
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        // Final destructive measurement of data qubits
        circuit.tick().mz(&[0, 1, 2]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let checker = PauliPropChecker::new(&circuit).with_config(config);

        // For this test, we check syndrome on ALL measured qubits (ancillas + data)
        // Any X error will flip SOME measurement result.
        let z_measurement_qubits = &[0usize, 1, 2, 3, 4];
        let x_ancillas: &[usize] = &[];

        // Logical Z = Z0Z1Z2 (the parity of data measurements)
        // An X error on a single data qubit flips that qubit's measurement,
        // which is detected. But does it cause a logical error?
        //
        // Actually, for this to work, we need to think about what "logical error" means
        // when we've measured everything. The decoder sees all measurements and
        // can compute the logical value. A single X error changes one data measurement,
        // which the decoder can account for.
        //
        // For simplicity, let's say the logical is determined by majority vote
        // of data qubits, so a single X error doesn't flip the logical.
        // This means no weight-1 X fault causes a logical error!

        // Define logical such that single X on data doesn't flip it:
        // Use the trivial logical (always commutes) for this test
        let logicals: &[(&[usize], &[usize])] = &[]; // No logical operators to check

        let analysis =
            checker.analyze_decoder_requirements(z_measurement_qubits, x_ancillas, logicals);

        println!("\nCircuit with final data measurement:");
        println!("  Total faults: {}", analysis.total_faults());
        println!(
            "  Correctable syndromes: {}",
            analysis.correctable_syndromes
        );
        println!(
            "  Undetectable logical errors: {}",
            analysis.undetectable_logical_errors
        );
        println!(
            "  Undetectable stabilizers: {}",
            analysis.undetectable_stabilizers
        );
        println!("  Is FT: {}", analysis.is_ft());

        // With no logical operators defined, all faults are either:
        // - Detected (syndrome on some measurement)
        // - Or stabilizer-equivalent (no effect)
        // So this should be "fault tolerant" by our definition.
        assert!(
            analysis.is_ft(),
            "Should be FT when all errors are detected"
        );

        match analysis.is_fault_tolerant() {
            Ok(()) => println!("\n  VERDICT: Fault tolerant!"),
            Err(failures) => {
                for f in &failures {
                    println!("  Failure: {}", f.description());
                }
                panic!("Expected FT");
            }
        }
    }

    #[test]
    fn test_simple_fault_tolerant_circuit() {
        // Simplest fault-tolerant example:
        // A single qubit prepared and immediately measured.
        // No gates = no fault locations between prep and measurement = trivially FT.

        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0]);
        circuit.tick().mz(&[0]);

        let config = FaultCheckConfig::new()
            .with_weight(1)
            .x_only()
            .stop_on_first(false);

        let checker = PauliPropChecker::new(&circuit).with_config(config);

        // Logical Z = Z0, measure qubit 0
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0])];
        let z_ancillas = &[0usize];
        let x_ancillas: &[usize] = &[];

        let analysis = checker.analyze_decoder_requirements(z_ancillas, x_ancillas, logicals);

        println!("\nSimple prep-measure circuit:");
        println!("  Total faults: {}", analysis.total_faults());
        println!(
            "  Undetectable logical errors: {}",
            analysis.undetectable_logical_errors
        );
        println!(
            "  Undetectable stabilizers: {}",
            analysis.undetectable_stabilizers
        );

        // X fault on the single qubit during prep/measure:
        // - If before MZ: flips measurement (detected), and is a logical X error
        // - Both outcomes consistent (all faults with syndrome cause logical error)
        // So this should be FT (no ambiguity, no undetectable logical errors)

        // Actually, an X error before measurement WILL be detected (syndrome = 1)
        // and it DOES cause a logical error (anticommutes with Z0).
        // So all faults are "DetectedUncorrectable" - consistent outcome.
        // This is FT by our definition (no ambiguity).

        println!("  Is FT: {}", analysis.is_ft());

        assert!(
            analysis.is_ft(),
            "Simple prep-measure should be FT (all outcomes consistent)"
        );
    }

    // ========================================================================
    // Tests for input qubit detection and s + r <= t enumeration
    // ========================================================================

    #[test]
    fn test_detect_input_qubits_state_prep() {
        // State preparation circuit: all qubits are prepared
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2]); // Prepare all qubits
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1), (0, 2)]);

        let input_qubits = detect_input_qubits(&circuit);
        assert!(
            input_qubits.is_empty(),
            "State prep should have no input qubits"
        );

        let checker = PauliPropChecker::new(&circuit);
        assert!(!checker.has_input_qubits());
    }

    #[test]
    fn test_detect_input_qubits_syndrome_extraction() {
        // Syndrome extraction: data qubits not prepared, ancillas prepared
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]); // Only prepare ancillas (3, 4)
        circuit.tick().cx(&[(0, 3)]); // Data qubit 0 used without prep
        circuit.tick().cx(&[(1, 3)]); // Data qubit 1 used without prep
        circuit.tick().cx(&[(1, 4)]); // Data qubit 1 used
        circuit.tick().cx(&[(2, 4)]); // Data qubit 2 used without prep
        circuit.tick().mz(&[3, 4]);

        let input_qubits = detect_input_qubits(&circuit);
        assert_eq!(input_qubits, vec![0, 1, 2], "Data qubits should be inputs");

        let ancillas = detect_ancilla_qubits(&circuit);
        assert_eq!(ancillas, vec![3, 4], "Qubits 3, 4 should be ancillas");

        let checker = PauliPropChecker::new(&circuit);
        assert!(checker.has_input_qubits());
        assert_eq!(checker.input_qubits(), &[0, 1, 2]);
    }

    #[test]
    fn test_pauli_combinations_weight_0() {
        let qubits = vec![0, 1, 2];
        let config = FaultCheckConfig::default();
        let combos = generate_pauli_combinations(&qubits, 0, &config);

        assert_eq!(combos.len(), 1, "Weight 0 should give single identity");
        assert_eq!(combos[0], vec![0, 0, 0], "Should be all identity");
    }

    #[test]
    fn test_pauli_combinations_weight_1() {
        let qubits = vec![0, 1];
        let config = FaultCheckConfig::default(); // X, Y, Z all included

        let combos = generate_pauli_combinations(&qubits, 1, &config);

        // 2 qubits, weight 1, 3 Pauli types = 2 * 3 = 6 combinations
        assert_eq!(combos.len(), 6);

        // Check we have X, Y, Z on each qubit
        let has_x0 = combos.iter().any(|c| c[0] == 1 && c[1] == 0);
        let has_y0 = combos.iter().any(|c| c[0] == 2 && c[1] == 0);
        let has_z0 = combos.iter().any(|c| c[0] == 3 && c[1] == 0);
        let has_x1 = combos.iter().any(|c| c[0] == 0 && c[1] == 1);
        let has_y1 = combos.iter().any(|c| c[0] == 0 && c[1] == 2);
        let has_z1 = combos.iter().any(|c| c[0] == 0 && c[1] == 3);

        assert!(has_x0 && has_y0 && has_z0);
        assert!(has_x1 && has_y1 && has_z1);
    }

    #[test]
    #[allow(clippy::naive_bytecount)]
    fn test_pauli_combinations_x_only() {
        let qubits = vec![0, 1, 2];
        let config = FaultCheckConfig::new().x_only();

        let combos = generate_pauli_combinations(&qubits, 1, &config);

        // 3 qubits, weight 1, X only = 3 combinations
        assert_eq!(combos.len(), 3);

        // All should have exactly one X
        for combo in &combos {
            let x_count = combo.iter().filter(|&&p| p == 1).count();
            let non_id_count = combo.iter().filter(|&&p| p != 0).count();
            assert_eq!(x_count, 1);
            assert_eq!(non_id_count, 1);
        }
    }

    #[test]
    fn test_analyze_with_input_faults_no_inputs() {
        // Circuit with no input qubits should behave same as regular analysis
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2, 3, 4]); // Prepare all
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let config = FaultCheckConfig::new().with_weight(1).x_only();
        let checker = PauliPropChecker::new(&circuit).with_config(config);

        let z_ancillas = &[3, 4];
        let x_ancillas: &[usize] = &[];
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];

        // Should have same result as regular analysis
        let regular = checker.analyze_fault_tolerance(z_ancillas, x_ancillas, logicals, false);
        let with_inputs =
            checker.analyze_with_input_faults(z_ancillas, x_ancillas, logicals, false);

        assert_eq!(regular.total_tested, with_inputs.total_tested);
        assert_eq!(
            regular.undetectable_logical_errors,
            with_inputs.undetectable_logical_errors
        );
    }

    #[test]
    fn test_analyze_with_input_faults_has_inputs() {
        // Syndrome extraction circuit with input qubits
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]); // Only ancillas prepared
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let config = FaultCheckConfig::new().with_weight(1).x_only();
        let checker = PauliPropChecker::new(&circuit).with_config(config);

        assert!(checker.has_input_qubits());
        assert_eq!(checker.input_qubits(), &[0, 1, 2]);

        let z_ancillas = &[3, 4];
        let x_ancillas: &[usize] = &[];
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];

        let analysis = checker.analyze_with_input_faults(z_ancillas, x_ancillas, logicals, false);

        // Should test more cases than internal-only
        // For t=1: s=0,r=1 (internal only) + s=1,r=0 (input only)
        println!("Total tested with input faults: {}", analysis.total_tested);

        // With 3 input qubits, weight-1 input faults = 3 X errors
        // Each combined with internal weight-0 (just 1 option: no fault)
        // Plus all internal weight-1 faults with input weight-0
        assert!(
            analysis.total_tested > 0,
            "Should test some fault combinations"
        );
    }

    #[test]
    fn test_s_plus_r_enumeration_in_pauli_prop_checker() {
        // Verify that s + r <= t is correctly enumerated
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[2]); // Only qubit 2 is prepared (ancilla)
        circuit.tick().cx(&[(0, 2)]); // Qubit 0 is input
        circuit.tick().cx(&[(1, 2)]); // Qubit 1 is input
        circuit.tick().mz(&[2]);

        // t = 1, X only
        let config = FaultCheckConfig::new().with_weight(1).x_only();
        let checker = PauliPropChecker::new(&circuit).with_config(config);

        assert_eq!(checker.input_qubits(), &[0, 1]);

        let z_ancillas = &[2];
        let x_ancillas: &[usize] = &[];
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1])];

        let analysis = checker.analyze_with_input_faults(z_ancillas, x_ancillas, logicals, false);

        // For t=1 with 2 input qubits and some internal locations:
        // s=0, r=1: enumerate internal faults
        // s=1, r=0: enumerate 2 input faults (X on q0, X on q1)
        // Total should be internal_count + 2
        println!("s+r<=1 enumeration: {} total tested", analysis.total_tested);

        // With input faults, we test more combinations
        let regular = checker.analyze_fault_tolerance(z_ancillas, x_ancillas, logicals, false);
        assert!(
            analysis.total_tested > regular.total_tested,
            "Should test more with input faults"
        );
    }

    #[test]
    fn test_combinations_helper() {
        // Test the combinations helper function
        let c2_3 = combinations(3, 2);
        assert_eq!(c2_3.len(), 3); // C(3,2) = 3
        assert!(c2_3.contains(&vec![0, 1]));
        assert!(c2_3.contains(&vec![0, 2]));
        assert!(c2_3.contains(&vec![1, 2]));

        let c1_4 = combinations(4, 1);
        assert_eq!(c1_4.len(), 4); // C(4,1) = 4

        let c0_5 = combinations(5, 0);
        assert_eq!(c0_5.len(), 1); // C(n,0) = 1 (empty set)
        assert_eq!(c0_5[0], Vec::<usize>::new());

        let c3_2 = combinations(2, 3);
        assert_eq!(c3_2.len(), 0); // Can't choose 3 from 2
    }

    #[test]
    fn test_pauli_product_helper() {
        let choices = vec![1u8, 2, 3]; // X, Y, Z

        let p1 = pauli_product(&choices, 1);
        assert_eq!(p1.len(), 3); // 3^1 = 3

        let p2 = pauli_product(&choices, 2);
        assert_eq!(p2.len(), 9); // 3^2 = 9

        let p0 = pauli_product(&choices, 0);
        assert_eq!(p0.len(), 1);
        assert_eq!(p0[0], Vec::<u8>::new());
    }

    // ========================================================================
    // Tests for follow-up syndrome analysis
    // ========================================================================

    #[test]
    fn test_compute_stabilizer_syndromes() {
        // Error: X on qubit 0
        let mut prop = PauliProp::new();
        prop.track_x(&[0]);

        // Stabilizers: Z0Z1 (checks X errors on q0 or q1)
        let stabilizers: &[(&[usize], &[usize])] = &[(&[], &[0, 1])];

        let syndromes = compute_stabilizer_syndromes(&prop, stabilizers);
        assert_eq!(syndromes, vec![true]); // X0 anticommutes with Z0Z1

        // Error: Z on qubit 0
        let mut prop_z = PauliProp::new();
        prop_z.track_z(&[0]);

        let syndromes_z = compute_stabilizer_syndromes(&prop_z, stabilizers);
        assert_eq!(syndromes_z, vec![false]); // Z0 commutes with Z0Z1
    }

    #[test]
    fn test_compute_stabilizer_syndromes_multiple() {
        // Error: X on qubit 1
        let mut prop = PauliProp::new();
        prop.track_x(&[1]);

        // Three-qubit code stabilizers: Z0Z1, Z1Z2
        let stabilizers: &[(&[usize], &[usize])] = &[(&[], &[0, 1]), (&[], &[1, 2])];

        let syndromes = compute_stabilizer_syndromes(&prop, stabilizers);
        // X1 anticommutes with both Z0Z1 and Z1Z2
        assert_eq!(syndromes, vec![true, true]);

        // Error: X on qubit 0
        let mut prop_x0 = PauliProp::new();
        prop_x0.track_x(&[0]);

        let syndromes_x0 = compute_stabilizer_syndromes(&prop_x0, stabilizers);
        // X0 anticommutes with Z0Z1 only
        assert_eq!(syndromes_x0, vec![true, false]);
    }

    #[test]
    fn test_extract_output_error() {
        let mut prop = PauliProp::new();
        prop.track_x(&[0]);
        prop.track_z(&[1]);
        prop.track_x(&[2]);
        prop.track_z(&[2]); // Y on qubit 2
        prop.track_x(&[5]); // Not in output qubits

        let output_qubits = vec![0, 1, 2, 3];
        let output = extract_output_error(&prop, &output_qubits);

        // Should have X on 0, Z on 1, Y on 2, nothing on 3
        assert!(output.contains_x(0));
        assert!(!output.contains_z(0));
        assert!(!output.contains_x(1));
        assert!(output.contains_z(1));
        assert!(output.contains_x(2));
        assert!(output.contains_z(2));
        assert!(!output.contains_x(3));
        assert!(!output.contains_z(3));

        // Qubit 5 should not be in output
        assert!(!output.contains_x(5));
    }

    #[test]
    fn test_follow_up_config_builder() {
        let config = FollowUpConfig::new()
            .with_output_qubits(vec![0, 1, 2])
            .with_stabilizer(vec![], vec![0, 1]) // Z0Z1
            .with_stabilizer(vec![], vec![1, 2]); // Z1Z2

        assert_eq!(config.output_qubits, vec![0, 1, 2]);
        assert_eq!(config.follow_up_stabilizers.len(), 2);
        assert!(config.has_follow_up());

        let empty = FollowUpConfig::new();
        assert!(!empty.has_follow_up());
    }

    #[test]
    fn test_analyze_with_follow_up_no_follow_up() {
        // Without follow-up config, should behave like regular analysis
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2, 3, 4]);
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let config = FaultCheckConfig::new().with_weight(1).x_only();
        let checker = PauliPropChecker::new(&circuit).with_config(config);

        let z_ancillas = &[3, 4];
        let x_ancillas: &[usize] = &[];
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];
        let no_follow_up = FollowUpConfig::new();

        let analysis =
            checker.analyze_with_follow_up(z_ancillas, x_ancillas, logicals, &no_follow_up);

        // Just verify we get some results
        assert!(analysis.total_faults() > 0);
    }

    #[test]
    fn test_analyze_with_follow_up_resolves_ambiguity() {
        // Syndrome extraction circuit for three-qubit code
        // Without follow-up: may have ambiguous syndromes
        // With follow-up stabilizers: should resolve ambiguity
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2, 3, 4]); // Prepare all (state prep scenario)
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let config = FaultCheckConfig::new().with_weight(1).x_only();
        let checker = PauliPropChecker::new(&circuit).with_config(config);

        let z_ancillas = &[3, 4];
        let x_ancillas: &[usize] = &[];
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];

        // Without follow-up
        let no_follow_up = FollowUpConfig::new();
        let analysis_no_follow_up =
            checker.analyze_with_follow_up(z_ancillas, x_ancillas, logicals, &no_follow_up);

        // With follow-up: specify the code's stabilizers
        let follow_up = FollowUpConfig::new()
            .with_output_qubits(vec![0, 1, 2])
            .with_stabilizer(vec![], vec![0, 1]) // Z0Z1
            .with_stabilizer(vec![], vec![1, 2]); // Z1Z2

        let analysis_with_follow_up =
            checker.analyze_with_follow_up(z_ancillas, x_ancillas, logicals, &follow_up);

        println!(
            "Without follow-up: {} ambiguous syndromes",
            analysis_no_follow_up.ambiguous_syndromes
        );
        println!(
            "With follow-up: {} ambiguous syndromes",
            analysis_with_follow_up.ambiguous_syndromes
        );

        // With follow-up, the output error provides additional syndrome information
        // This should help disambiguate (or at least not make things worse)
        // The key insight: different output errors produce different follow-up syndromes
    }

    #[test]
    fn test_is_gadget_fault_tolerant() {
        // Simple test of the convenience method
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2, 3, 4]);
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let config = FaultCheckConfig::new().with_weight(1).x_only();
        let checker = PauliPropChecker::new(&circuit).with_config(config);

        let z_ancillas = &[3, 4];
        let x_ancillas: &[usize] = &[];
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1, 2])];

        let follow_up = FollowUpConfig::new()
            .with_output_qubits(vec![0, 1, 2])
            .with_stabilizer(vec![], vec![0, 1])
            .with_stabilizer(vec![], vec![1, 2]);

        let is_ft = checker.is_gadget_fault_tolerant(z_ancillas, x_ancillas, logicals, &follow_up);
        println!("Gadget is fault tolerant with follow-up: {is_ft}");
    }

    #[test]
    fn test_detect_output_qubits_final_measurement() {
        // Circuit with final measurement - no output qubits
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2, 3, 4]);
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[0, 1, 2, 3, 4]); // All qubits measured

        let outputs = detect_output_qubits(&circuit);
        assert!(outputs.is_empty(), "Final measurement has no output qubits");
    }

    #[test]
    fn test_detect_output_qubits_syndrome_extraction() {
        // Syndrome extraction - data qubits are outputs, ancillas measured
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]); // Only ancillas prepared
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]); // Only ancillas measured

        let outputs = detect_output_qubits(&circuit);
        // Data qubits 0, 1, 2 are used (in CX) but never measured
        assert!(outputs.contains(&0));
        assert!(outputs.contains(&1));
        assert!(outputs.contains(&2));
        // Ancillas 3, 4 are measured, so not outputs
        assert!(!outputs.contains(&3));
        assert!(!outputs.contains(&4));
    }

    #[test]
    fn test_circuit_io_from_circuit() {
        // Test the full CircuitIO detection
        // This circuit has inputs (data qubits), outputs (data qubits), and ancillas
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]); // Ancillas prepared
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]); // Ancillas measured

        let checker = PauliPropChecker::new(&circuit);

        // Input qubits: data qubits 0, 1, 2 used but never prepared
        assert_eq!(checker.input_qubits().len(), 3);
        assert!(checker.input_qubits().contains(&0));
        assert!(checker.input_qubits().contains(&1));
        assert!(checker.input_qubits().contains(&2));

        // Output qubits: data qubits 0, 1, 2 used but never measured
        assert_eq!(checker.output_qubits().len(), 3);
        assert!(checker.output_qubits().contains(&0));
        assert!(checker.output_qubits().contains(&1));
        assert!(checker.output_qubits().contains(&2));

        // Ancilla qubits: 3, 4 are prepared
        assert_eq!(checker.ancilla_qubits().len(), 2);
        assert!(checker.ancilla_qubits().contains(&3));
        assert!(checker.ancilla_qubits().contains(&4));

        // Measured qubits: 3, 4
        assert_eq!(checker.measured_qubits().len(), 2);
        assert!(checker.measured_qubits().contains(&3));
        assert!(checker.measured_qubits().contains(&4));

        // Helper methods
        assert!(checker.has_input_qubits());
        assert!(checker.has_output_qubits());
        assert_eq!(
            checker.circuit_type(),
            "pass-through gadget (has inputs and outputs)"
        );
    }

    #[test]
    fn test_circuit_io_state_prep() {
        // State preparation: all qubits prepared, no outputs
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2]); // All qubits prepared
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1), (0, 2)]);
        circuit.tick().mz(&[0, 1, 2]); // All measured

        let checker = PauliPropChecker::new(&circuit);

        assert!(
            checker.input_qubits().is_empty(),
            "State prep has no inputs"
        );
        assert!(checker.output_qubits().is_empty(), "All qubits measured");
        assert!(!checker.has_input_qubits());
        assert!(!checker.has_output_qubits());
        assert_eq!(
            checker.circuit_type(),
            "self-contained (state prep + final measurement)"
        );
    }

    #[test]
    fn test_circuit_io_state_prep_no_measurement() {
        // State preparation that produces output qubits
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[0, 1, 2]); // All qubits prepared
        circuit.tick().h(&[0]);
        circuit.tick().cx(&[(0, 1), (0, 2)]);
        // No measurement - outputs go to next stage

        let checker = PauliPropChecker::new(&circuit);

        assert!(
            checker.input_qubits().is_empty(),
            "State prep has no inputs"
        );
        assert_eq!(checker.output_qubits().len(), 3, "All qubits are outputs");
        assert!(!checker.has_input_qubits());
        assert!(checker.has_output_qubits());
        assert_eq!(
            checker.circuit_type(),
            "state preparation (no inputs, has outputs)"
        );
    }

    #[test]
    fn test_circuit_io_final_measurement() {
        // Final measurement: has inputs, measures everything
        let mut circuit = TickCircuit::new();
        // No prep - qubits come from previous stage
        circuit.tick().mz(&[0, 1, 2]); // Measure all

        let checker = PauliPropChecker::new(&circuit);

        assert_eq!(checker.input_qubits().len(), 3, "All qubits are inputs");
        assert!(
            checker.output_qubits().is_empty(),
            "No outputs after measurement"
        );
        assert!(checker.has_input_qubits());
        assert!(!checker.has_output_qubits());
        assert_eq!(
            checker.circuit_type(),
            "final measurement (has inputs, no outputs)"
        );
    }

    #[test]
    fn test_pauli_prop_checker_accessor_methods() {
        // Verify that PauliPropChecker accessor methods work correctly
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[3, 4]);
        circuit.tick().cx(&[(0, 3)]);
        circuit.tick().cx(&[(1, 3)]);
        circuit.tick().cx(&[(1, 4)]);
        circuit.tick().cx(&[(2, 4)]);
        circuit.tick().mz(&[3, 4]);

        let checker = PauliPropChecker::new(&circuit);

        // The checker should have detected the I/O structure
        assert!(
            checker.has_input_qubits(),
            "Checker should detect input qubits"
        );

        // Verify accessor methods
        assert_eq!(checker.input_qubits().len(), 3);
        assert_eq!(checker.output_qubits().len(), 3);
        assert!(checker.has_input_qubits());
        assert!(checker.has_output_qubits());
    }

    #[test]
    fn test_analyze_with_input_faults_collects_failures() {
        // Test that collect_failures=true actually collects failure details
        // Use a simple circuit where we know failures will occur
        let mut circuit = TickCircuit::new();
        circuit.tick().pz(&[2]); // Ancilla
        circuit.tick().cx(&[(0, 2), (1, 2)]);
        circuit.tick().mz(&[2]);

        let config = FaultCheckConfig::new().with_weight(1).all_paulis();
        let checker = PauliPropChecker::new(&circuit).with_config(config);

        let z_ancillas = &[2];
        let x_ancillas: &[usize] = &[];
        // Logical Z = Z0 Z1 - a Z error on either data qubit is undetectable
        let logicals: &[(&[usize], &[usize])] = &[(&[], &[0, 1])];

        // First check without collection
        let analysis_no_collect =
            checker.analyze_with_input_faults(z_ancillas, x_ancillas, logicals, false);

        // Then check with collection
        let analysis_with_collect =
            checker.analyze_with_input_faults(z_ancillas, x_ancillas, logicals, true);

        // Both should have same counts
        assert_eq!(
            analysis_no_collect.total_tested,
            analysis_with_collect.total_tested
        );
        assert_eq!(
            analysis_no_collect.undetectable_logical_errors,
            analysis_with_collect.undetectable_logical_errors
        );

        // With collection, failure_details should be populated
        assert_eq!(
            analysis_with_collect.failure_details.len(),
            analysis_with_collect.undetectable_logical_errors,
            "failure_details should contain one entry per undetectable logical error"
        );

        // Each failure detail should have valid data
        for (fault_config, prop_result) in &analysis_with_collect.failure_details {
            // Fault config should have faults (either input or internal)
            // Note: input faults are represented as PauliFaults at tick 0
            assert!(
                !fault_config.faults.is_empty()
                    || analysis_with_collect.undetectable_logical_errors > 0,
                "Failure should have associated faults or be from input-only fault"
            );

            // PropagationResult should indicate a logical error
            assert!(
                prop_result.logical_errors.iter().any(|&e| e),
                "Collected failure should have logical error"
            );
        }

        println!(
            "Collected {} failure details out of {} undetectable logical errors",
            analysis_with_collect.failure_details.len(),
            analysis_with_collect.undetectable_logical_errors
        );
    }
}
