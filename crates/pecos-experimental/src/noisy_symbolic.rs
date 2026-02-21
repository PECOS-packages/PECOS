// Copyright 2025 The PECOS Developers
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

//! Noisy measurement history for efficient sampling with depolarizing noise.
//!
//! **⚠️ EXPERIMENTAL: This API is unstable and may change without notice.**
//!
//! This module extends the symbolic measurement history to include fault events
//! from a depolarizing noise model. Faults are treated as "hidden random bits"
//! with their own probabilities that get XOR'd into measurement outcomes.
//!
//! # Overview
//!
//! The key insight is that faults can be precompiled: for each possible fault
//! location and type, we use the Pauli propagator to determine which measurements
//! get flipped. This creates a dependency structure:
//!
//! ```text
//! Random bits:  r1 (50%), r2 (50%), ...     # from non-deterministic measurements
//! Fault bits:   f1 (p1), f2 (p2), ...       # from noise events
//!
//! Measurements:
//!   m1 = r1 ^ flip1
//!   m2 = r1 ^ flip2                         # correlated with m1
//!   m3 = m1 ^ m2 ^ f1 ^ f3 ^ flip3          # depends on measurements AND faults
//!   m4 = m1 ^ m3 ^ f1 ^ f6 ^ flip4
//! ```
//!
//! During sampling, we:
//! 1. Sample fault bits (Bernoulli with fault probabilities)
//! 2. Sample random bits (50/50, same as before)
//! 3. Compute all measurements via XOR chains
//! 4. Return only the measurement outcomes (fault bits are hidden)

use pecos_core::BitSet;
use pecos_core::QubitId;
use pecos_core::gate_type::GateType;
use pecos_qsim::CliffordGateable;
use pecos_qsim::measurement_sampler::SampleResult;
use pecos_qsim::pauli_prop::PauliProp;
use pecos_qsim::symbolic_sparse_stab::MeasurementHistory;
use pecos_quantum::Circuit;
use pecos_rng::{PecosRng, Rng, RngBulkExt, RngExt};
use std::collections::BTreeSet;
use wide::u64x4;

/// A single fault event in the noise model.
///
/// Each fault event has a probability of occurring and affects a set of
/// measurements when it does occur.
#[derive(Clone, Debug, PartialEq)]
pub struct FaultEvent {
    /// Probability that this fault occurs (0.0 to 1.0).
    pub probability: f64,

    /// Human-readable description of the fault (e.g., "X on qubit 2 after gate 5").
    pub description: String,
}

/// A measurement result with both measurement and fault dependencies.
///
/// Extends `SymbolicMeasurementResult` by adding fault dependencies.
/// The final outcome is: `flip ^ XOR(measurement_deps) ^ XOR(fault_deps)`
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NoisyMeasurementResult {
    /// Indices of other measurements whose outcomes XOR together.
    /// Empty set means no measurement dependency.
    pub measurement_deps: BitSet,

    /// Indices of fault events whose outcomes XOR into this measurement.
    /// Empty set means this measurement is not affected by any faults.
    pub fault_deps: BitSet,

    /// Whether to flip the XOR result (from unitary gate phases).
    pub flip: bool,

    /// Whether this measurement was deterministic in the noiseless case.
    pub is_deterministic: bool,

    /// The unique index of this measurement.
    pub index: usize,
}

impl std::fmt::Display for NoisyMeasurementResult {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "m{}=", self.index)?;

        if self.is_deterministic {
            let mut terms = Vec::new();

            // Add measurement dependencies
            for dep in &self.measurement_deps {
                terms.push(format!("m{dep}"));
            }

            // Add fault dependencies
            for dep in &self.fault_deps {
                terms.push(format!("f{dep}"));
            }

            if terms.is_empty() {
                // No dependencies, just the flip value
                write!(f, "{}", u8::from(self.flip))
            } else {
                // Write XOR expression
                write!(f, "{}", terms.join("^"))?;
                if self.flip {
                    write!(f, "^1")?;
                }
                Ok(())
            }
        } else {
            // Non-deterministic measurement
            if self.fault_deps.is_empty() {
                write!(f, "?")
            } else {
                // Random but with fault dependencies
                let fault_terms: Vec<String> =
                    self.fault_deps.iter().map(|d| format!("f{d}")).collect();
                write!(f, "?^{}", fault_terms.join("^"))
            }
        }
    }
}

/// A measurement with flattened dependencies for vectorized sampling.
///
/// Instead of depending on previous measurements, all dependencies are
/// resolved to ultimate random bits and fault bits. This enables
/// processing 64 shots at a time without sequential dependencies.
#[derive(Clone, Debug)]
struct FlattenedMeasurement {
    /// Indices of random bits that XOR into this measurement.
    /// Each non-deterministic measurement has its own random bit (indexed by measurement index).
    random_bit_deps: BitSet,

    /// Indices of fault events that XOR into this measurement.
    fault_deps: BitSet,

    /// The accumulated flip value after resolving all dependencies.
    flip: bool,
}

/// Classification of a flattened measurement for efficient SIMD sampling.
///
/// Similar to `MeasurementKind` in the noiseless sampler, this classifies
/// measurements into special cases that can be handled more efficiently.
#[derive(Clone, Debug)]
enum NoisyMeasurementKind {
    /// Deterministic constant value (no random deps, no fault deps)
    Fixed(bool),
    /// Pure random 50/50 (single random dep, no fault deps, no flip)
    PureRandom(usize),
    /// Pure random 50/50 flipped (single random dep, no fault deps, flip=true)
    PureRandomFlipped(usize),
    /// Copy of a random measurement column (we'll reuse the column)
    Copy(usize),
    /// Flipped copy of a random measurement column
    CopyFlipped(usize),
    /// General case: XOR of random bits and fault bits
    Computed {
        random_deps: Vec<usize>,
        fault_deps: Vec<usize>,
        flip: bool,
    },
}

impl NoisyMeasurementKind {
    /// Classify a flattened measurement for efficient processing.
    fn from_flattened(flat: &FlattenedMeasurement) -> Self {
        let num_random = flat.random_bit_deps.len();
        let num_faults = flat.fault_deps.len();

        // Fixed value: no dependencies at all
        if num_random == 0 && num_faults == 0 {
            return NoisyMeasurementKind::Fixed(flat.flip);
        }

        // Pure random: single random dep, no faults
        if num_random == 1 && num_faults == 0 {
            let r_idx = flat.random_bit_deps.iter().next().unwrap();
            if flat.flip {
                return NoisyMeasurementKind::PureRandomFlipped(r_idx);
            }
            return NoisyMeasurementKind::PureRandom(r_idx);
        }

        // General computed case
        NoisyMeasurementKind::Computed {
            random_deps: flat.random_bit_deps.iter().collect(),
            fault_deps: flat.fault_deps.iter().collect(),
            flip: flat.flip,
        }
    }

    /// Convert classified measurements to detect Copy patterns.
    ///
    /// After classifying all measurements, we can detect when a measurement
    /// is just a copy (or flipped copy) of a random bit column that was
    /// generated earlier. This lets us clone columns instead of recomputing.
    fn detect_copies(kinds: &mut [Self]) {
        // Track which random indices have pure random columns we can copy
        let mut random_column_sources: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();

        for (meas_idx, kind) in kinds.iter_mut().enumerate() {
            match kind {
                NoisyMeasurementKind::PureRandom(r_idx) => {
                    // This measurement generates a new random column
                    random_column_sources.insert(*r_idx, meas_idx);
                }
                NoisyMeasurementKind::PureRandomFlipped(r_idx) => {
                    // Check if we already have this random column
                    if let Some(&src_meas) = random_column_sources.get(r_idx) {
                        // We can just flip-copy from the earlier measurement
                        *kind = NoisyMeasurementKind::CopyFlipped(src_meas);
                    }
                }
                NoisyMeasurementKind::Computed {
                    random_deps,
                    fault_deps,
                    flip,
                } => {
                    // Check if this is effectively a copy/flipped-copy
                    if fault_deps.is_empty() && random_deps.len() == 1 {
                        let r_idx = random_deps[0];
                        if let Some(&src_meas) = random_column_sources.get(&r_idx) {
                            if *flip {
                                *kind = NoisyMeasurementKind::CopyFlipped(src_meas);
                            } else {
                                *kind = NoisyMeasurementKind::Copy(src_meas);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

/// Depolarizing noise model parameters.
///
/// Specifies error rates for different operation types.
#[derive(Clone, Debug, PartialEq)]
pub struct DepolarizingNoiseModel {
    /// Error probability for single-qubit gates.
    /// Each of X, Y, Z occurs with probability p1/3.
    pub p1: f64,

    /// Error probability for two-qubit gates.
    /// Each of the 15 non-identity Pauli pairs occurs with probability p2/15.
    pub p2: f64,

    /// Measurement fault probability.
    /// The measurement outcome is flipped with this probability.
    pub p_meas: f64,

    /// State preparation fault probability.
    /// The prepared state has an X error with this probability.
    pub p_prep: f64,
}

impl Default for DepolarizingNoiseModel {
    fn default() -> Self {
        Self {
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.0,
        }
    }
}

impl DepolarizingNoiseModel {
    /// Create a new noise model with the given parameters.
    #[must_use]
    pub fn new(p1: f64, p2: f64, p_meas: f64, p_prep: f64) -> Self {
        Self {
            p1,
            p2,
            p_meas,
            p_prep,
        }
    }

    /// Create a uniform noise model where all error types have the same rate.
    #[must_use]
    pub fn uniform(p: f64) -> Self {
        Self {
            p1: p,
            p2: p,
            p_meas: p,
            p_prep: p,
        }
    }

    /// Check if the noise model is noiseless (all probabilities are zero).
    #[must_use]
    pub fn is_noiseless(&self) -> bool {
        self.p1 == 0.0 && self.p2 == 0.0 && self.p_meas == 0.0 && self.p_prep == 0.0
    }
}

/// Noisy measurement history containing measurements and fault events.
///
/// This structure enables efficient sampling of noisy measurement outcomes
/// by precomputing how each fault affects the measurements.
#[derive(Clone, Debug)]
pub struct NoisyMeasurementHistory {
    /// The measurement results with their dependencies.
    measurements: Vec<NoisyMeasurementResult>,

    /// The fault events with their probabilities.
    faults: Vec<FaultEvent>,
}

impl NoisyMeasurementHistory {
    /// Create a new empty noisy measurement history.
    #[must_use]
    pub fn new() -> Self {
        Self {
            measurements: Vec::new(),
            faults: Vec::new(),
        }
    }

    /// Create a noisy measurement history from a noiseless one.
    ///
    /// This creates a history with no fault events - useful as a starting point
    /// before adding faults via `add_fault`.
    #[must_use]
    pub fn from_noiseless(history: &MeasurementHistory) -> Self {
        let measurements = history
            .iter()
            .map(|m| NoisyMeasurementResult {
                measurement_deps: m.outcome.clone(),
                fault_deps: BitSet::new(),
                flip: m.flip,
                is_deterministic: m.is_deterministic,
                index: m.index,
            })
            .collect();

        Self {
            measurements,
            faults: Vec::new(),
        }
    }

    /// Add a fault event that affects the specified measurements.
    ///
    /// Returns the index of the new fault event.
    pub fn add_fault(
        &mut self,
        probability: f64,
        affected_measurements: &BTreeSet<usize>,
        description: String,
    ) -> usize {
        let fault_index = self.faults.len();

        // Add the fault event
        self.faults.push(FaultEvent {
            probability,
            description,
        });

        // Add this fault as a dependency to all affected measurements
        for &meas_idx in affected_measurements {
            if meas_idx < self.measurements.len() {
                self.measurements[meas_idx].fault_deps.insert(fault_index);
            }
        }

        fault_index
    }

    /// Get the number of measurements.
    #[inline]
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.measurements.len()
    }

    /// Get the number of fault events.
    #[inline]
    #[must_use]
    pub fn num_faults(&self) -> usize {
        self.faults.len()
    }

    /// Get a reference to the measurements.
    #[inline]
    #[must_use]
    pub fn measurements(&self) -> &[NoisyMeasurementResult] {
        &self.measurements
    }

    /// Get a reference to the fault events.
    #[inline]
    #[must_use]
    pub fn faults(&self) -> &[FaultEvent] {
        &self.faults
    }

    /// Get a specific measurement by index.
    #[inline]
    #[must_use]
    pub fn get(&self, index: usize) -> Option<&NoisyMeasurementResult> {
        self.measurements.get(index)
    }

    /// Get a specific fault event by index.
    #[inline]
    #[must_use]
    pub fn get_fault(&self, index: usize) -> Option<&FaultEvent> {
        self.faults.get(index)
    }

    /// Iterate over all measurements.
    pub fn iter(&self) -> impl Iterator<Item = &NoisyMeasurementResult> {
        self.measurements.iter()
    }

    /// Iterate over all fault events.
    pub fn iter_faults(&self) -> impl Iterator<Item = &FaultEvent> {
        self.faults.iter()
    }

    /// Check if this history has any fault events.
    #[inline]
    #[must_use]
    pub fn has_faults(&self) -> bool {
        !self.faults.is_empty()
    }

    /// Flatten all measurement dependencies for vectorized sampling.
    ///
    /// This resolves all measurement-to-measurement dependencies so each
    /// measurement depends only on random bits and fault bits. This enables
    /// computing all measurements independently, allowing 64 shots to be
    /// processed at once using u64 bit operations.
    fn flatten_dependencies(&self) -> Vec<FlattenedMeasurement> {
        let n = self.measurements.len();
        let mut flattened = Vec::with_capacity(n);

        for meas_idx in 0..n {
            let meas = &self.measurements[meas_idx];
            let mut random_deps = BitSet::new();
            let mut fault_deps = meas.fault_deps.clone();
            let mut flip = meas.flip;

            // If non-deterministic, this measurement has its own random bit
            if !meas.is_deterministic {
                random_deps.insert(meas_idx);
            }

            // Resolve measurement dependencies by XORing in their flattened deps
            // Note: measurement_deps might include the current measurement's own index
            // (for non-deterministic measurements), which we skip since it's handled
            // by the random bit above. We also skip any forward references (dep_idx >= meas_idx)
            // as these would indicate a circular dependency which shouldn't occur in a
            // properly constructed measurement history.
            for dep_idx in &meas.measurement_deps {
                if dep_idx >= meas_idx {
                    // Skip self-reference and forward references
                    continue;
                }
                let dep: &FlattenedMeasurement = &flattened[dep_idx];
                // XOR = symmetric difference for sets
                random_deps.symmetric_difference_update(&dep.random_bit_deps);
                fault_deps.symmetric_difference_update(&dep.fault_deps);
                flip ^= dep.flip;
            }

            flattened.push(FlattenedMeasurement {
                random_bit_deps: random_deps,
                fault_deps,
                flip,
            });
        }

        flattened
    }

    /// Flatten and classify measurements for optimal SIMD sampling.
    ///
    /// This combines `flatten_dependencies` with `NoisyMeasurementKind` classification
    /// to produce an optimized representation for sampling.
    fn classify_measurements(&self) -> Vec<NoisyMeasurementKind> {
        let flattened = self.flatten_dependencies();
        let mut kinds: Vec<NoisyMeasurementKind> = flattened
            .iter()
            .map(NoisyMeasurementKind::from_flattened)
            .collect();
        NoisyMeasurementKind::detect_copies(&mut kinds);
        kinds
    }
}

impl Default for NoisyMeasurementHistory {
    fn default() -> Self {
        Self::new()
    }
}

/// Pauli type for fault injection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pauli {
    X,
    Y,
    Z,
}

impl std::fmt::Display for Pauli {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Pauli::X => write!(f, "X"),
            Pauli::Y => write!(f, "Y"),
            Pauli::Z => write!(f, "Z"),
        }
    }
}

/// A gate location in the circuit, used for fault injection.
#[derive(Clone, Debug)]
struct GateLocation {
    /// The gate type
    gate_type: GateType,
    /// The qubits involved
    qubits: Vec<usize>,
}

/// Builder for creating noisy measurement histories from circuits.
///
/// This builder walks a circuit and noise model to determine:
/// 1. What faults can occur (based on noise model probabilities)
/// 2. Which measurements each fault affects (via Pauli propagation)
///
/// # Example
///
/// ```rust
/// use pecos_experimental::{
///     NoisyMeasurementHistoryBuilder, DepolarizingNoiseModel, execute_hugr,
/// };
/// use pecos_qsim::SymbolicSparseStab;
/// use pecos_quantum::{DagCircuit, Gate};
///
/// // Create a circuit with gates that can have faults
/// let mut circuit = DagCircuit::new();
/// circuit.add_gate(Gate::h(&[0]));
/// circuit.add_gate(Gate::cx(&[(0, 1)]));
/// circuit.add_gate(Gate::measure(&[0]));
/// circuit.add_gate(Gate::measure(&[1]));
///
/// // Run symbolic simulation to get noiseless measurement history
/// let mut sim = SymbolicSparseStab::new(2);
/// execute_hugr(&mut sim, &circuit).unwrap();
///
/// // Build noisy measurement history
/// let noisy_history = NoisyMeasurementHistoryBuilder::new()
///     .with_noise_model(DepolarizingNoiseModel::uniform(0.001))
///     .build_from_circuit(&circuit, sim.measurement_history());
///
/// assert_eq!(noisy_history.num_measurements(), 2);
/// ```
pub struct NoisyMeasurementHistoryBuilder {
    noise_model: DepolarizingNoiseModel,
}

impl Default for NoisyMeasurementHistoryBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl NoisyMeasurementHistoryBuilder {
    /// Create a new builder with default (noiseless) noise model.
    #[must_use]
    pub fn new() -> Self {
        Self {
            noise_model: DepolarizingNoiseModel::default(),
        }
    }

    /// Set the noise model.
    #[must_use]
    pub fn with_noise_model(mut self, noise_model: DepolarizingNoiseModel) -> Self {
        self.noise_model = noise_model;
        self
    }

    /// Build a noisy measurement history from a circuit.
    ///
    /// This method:
    /// 1. Converts the noiseless measurement history to noisy format
    /// 2. Walks the circuit to identify fault locations
    /// 3. For each fault, propagates it forward to determine affected measurements
    /// 4. Adds fault events with appropriate probabilities
    ///
    /// # Arguments
    /// * `circuit` - The circuit to analyze
    /// * `noiseless_history` - The measurement history from noiseless symbolic execution
    ///
    /// # Returns
    /// A `NoisyMeasurementHistory` with fault events based on the noise model.
    #[must_use]
    pub fn build_from_circuit<C: Circuit>(
        &self,
        circuit: &C,
        noiseless_history: &MeasurementHistory,
    ) -> NoisyMeasurementHistory {
        // Start with the noiseless history
        let mut history = NoisyMeasurementHistory::from_noiseless(noiseless_history);

        // If noiseless, nothing to do
        if self.noise_model.is_noiseless() {
            return history;
        }

        // Collect gate locations and identify measurement positions
        let (gate_locations, measurement_positions) = Self::collect_gate_info(circuit);

        // Process each gate for potential faults
        for (loc_idx, location) in gate_locations.iter().enumerate() {
            self.add_faults_for_gate(
                &mut history,
                location,
                loc_idx,
                &gate_locations,
                &measurement_positions,
            );
        }

        history
    }

    /// Collect gate information from the circuit.
    ///
    /// Returns (`gate_locations`, `measurement_positions`) where:
    /// - `gate_locations`: All gates in topological order
    /// - `measurement_positions`: Map from gate index to measurement index
    fn collect_gate_info<C: Circuit>(
        circuit: &C,
    ) -> (Vec<GateLocation>, std::collections::HashMap<usize, usize>) {
        let mut gate_locations = Vec::new();
        let mut measurement_positions = std::collections::HashMap::new();
        let mut measurement_count = 0;

        for gate_view in circuit.iter_gates_topo() {
            let gate = gate_view.gate;
            let qubits: Vec<usize> = gate.qubits.iter().map(pecos_core::QubitId::index).collect();

            // Track measurement positions
            if matches!(
                gate.gate_type,
                GateType::Measure | GateType::MeasureFree | GateType::MeasureLeaked
            ) {
                measurement_positions.insert(gate_locations.len(), measurement_count);
                measurement_count += 1;
            }

            gate_locations.push(GateLocation {
                gate_type: gate.gate_type,
                qubits,
            });
        }

        (gate_locations, measurement_positions)
    }

    /// Add fault events for a specific gate.
    #[allow(clippy::too_many_lines)]
    fn add_faults_for_gate(
        &self,
        history: &mut NoisyMeasurementHistory,
        location: &GateLocation,
        loc_idx: usize,
        all_gates: &[GateLocation],
        measurement_positions: &std::collections::HashMap<usize, usize>,
    ) {
        match location.gate_type {
            // Single-qubit Clifford gates: depolarizing noise applies X, Y, or Z
            GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::H
            | GateType::SZ
            | GateType::SZdg => {
                if self.noise_model.p1 > 0.0 {
                    let q = location.qubits[0];
                    let p_each = self.noise_model.p1 / 3.0;

                    for pauli in [Pauli::X, Pauli::Y, Pauli::Z] {
                        let affected = Self::propagate_fault(
                            pauli,
                            q,
                            loc_idx + 1, // Fault occurs AFTER the gate
                            all_gates,
                            measurement_positions,
                        );

                        if !affected.is_empty() {
                            history.add_fault(
                                p_each,
                                &affected,
                                format!("{pauli} after {:?} on q{q}", location.gate_type),
                            );
                        }
                    }
                }
            }

            // Two-qubit Clifford gates: depolarizing noise applies one of 15 Pauli pairs
            GateType::CX | GateType::CY | GateType::CZ => {
                if self.noise_model.p2 > 0.0 {
                    let q1 = location.qubits[0];
                    let q2 = location.qubits[1];
                    let p_each = self.noise_model.p2 / 15.0;

                    // All 15 non-identity Pauli pairs
                    let paulis = [Pauli::X, Pauli::Y, Pauli::Z];
                    for &p1 in &paulis {
                        for &p2 in &paulis {
                            // Both qubits get a Pauli (IX, IY, IZ already covered, skip II)
                            let affected = Self::propagate_two_qubit_fault(
                                p1,
                                q1,
                                p2,
                                q2,
                                loc_idx + 1,
                                all_gates,
                                measurement_positions,
                            );

                            if !affected.is_empty() {
                                history.add_fault(
                                    p_each,
                                    &affected,
                                    format!(
                                        "{p1}{p2} after {:?} on q{q1},q{q2}",
                                        location.gate_type
                                    ),
                                );
                            }
                        }
                    }

                    // Also handle single-qubit Paulis on each qubit (XI, YI, ZI, IX, IY, IZ)
                    // These are the remaining 6 of the 15 non-identity pairs
                    for &p in &paulis {
                        // Pauli on q1 only
                        let affected = Self::propagate_fault(
                            p,
                            q1,
                            loc_idx + 1,
                            all_gates,
                            measurement_positions,
                        );
                        if !affected.is_empty() {
                            history.add_fault(
                                p_each,
                                &affected,
                                format!("{p}I after {:?} on q{q1},q{q2}", location.gate_type),
                            );
                        }

                        // Pauli on q2 only
                        let affected = Self::propagate_fault(
                            p,
                            q2,
                            loc_idx + 1,
                            all_gates,
                            measurement_positions,
                        );
                        if !affected.is_empty() {
                            history.add_fault(
                                p_each,
                                &affected,
                                format!("I{p} after {:?} on q{q1},q{q2}", location.gate_type),
                            );
                        }
                    }
                }
            }

            // State preparation: X error with probability p_prep
            GateType::Prep | GateType::QAlloc => {
                if self.noise_model.p_prep > 0.0 && !location.qubits.is_empty() {
                    let q = location.qubits[0];

                    // X error flips the prepared |0⟩ to |1⟩
                    let affected = Self::propagate_fault(
                        Pauli::X,
                        q,
                        loc_idx + 1,
                        all_gates,
                        measurement_positions,
                    );

                    if !affected.is_empty() {
                        history.add_fault(
                            self.noise_model.p_prep,
                            &affected,
                            format!("X prep error on q{q}"),
                        );
                    }
                }
            }

            // Measurement: flip the measurement outcome with probability p_meas
            GateType::Measure | GateType::MeasureFree | GateType::MeasureLeaked => {
                if self.noise_model.p_meas > 0.0 {
                    // Measurement fault directly flips this measurement
                    if let Some(&meas_idx) = measurement_positions.get(&loc_idx) {
                        let mut affected = BTreeSet::new();
                        affected.insert(meas_idx);

                        history.add_fault(
                            self.noise_model.p_meas,
                            &affected,
                            format!("Meas fault on m{meas_idx}"),
                        );
                    }
                }
            }

            // Other gates: no noise applied
            _ => {}
        }
    }

    /// Propagate a single-qubit Pauli fault forward through the circuit.
    ///
    /// Returns the set of measurement indices that would be flipped by this fault.
    fn propagate_fault(
        pauli: Pauli,
        qubit: usize,
        start_loc: usize,
        all_gates: &[GateLocation],
        measurement_positions: &std::collections::HashMap<usize, usize>,
    ) -> BTreeSet<usize> {
        let mut prop = PauliProp::new();

        // Add the initial Pauli
        match pauli {
            Pauli::X => prop.add_x(qubit),
            Pauli::Y => prop.add_y(qubit),
            Pauli::Z => prop.add_z(qubit),
        }

        let mut affected_measurements = BTreeSet::new();

        // Propagate through subsequent gates
        for (loc_idx, location) in all_gates.iter().enumerate().skip(start_loc) {
            match location.gate_type {
                // Single-qubit Clifford gates
                // Note: X, Y, Z gates don't change the X/Z basis of Paulis for propagation purposes
                // (sign changes don't affect measurement flips), so they fall through to _ => {}
                GateType::H => {
                    if !location.qubits.is_empty() {
                        prop.h(&[QubitId(location.qubits[0])]);
                    }
                }
                GateType::SZ => {
                    if !location.qubits.is_empty() {
                        prop.sz(&[QubitId(location.qubits[0])]);
                    }
                }
                GateType::SZdg => {
                    if !location.qubits.is_empty() {
                        // S† = S³
                        let q = QubitId(location.qubits[0]);
                        prop.sz(&[q]).sz(&[q]).sz(&[q]);
                    }
                }

                // Two-qubit Clifford gates
                GateType::CX => {
                    if location.qubits.len() >= 2 {
                        prop.cx(&[QubitId(location.qubits[0]), QubitId(location.qubits[1])]);
                    }
                }
                GateType::CY => {
                    if location.qubits.len() >= 2 {
                        let (q1, q2) = (QubitId(location.qubits[0]), QubitId(location.qubits[1]));
                        // CY = (I ⊗ S†) CX (I ⊗ S)
                        prop.sz(&[q2]).cx(&[q1, q2]).sz(&[q2]).sz(&[q2]).sz(&[q2]);
                    }
                }
                GateType::CZ => {
                    if location.qubits.len() >= 2 {
                        let (q1, q2) = (QubitId(location.qubits[0]), QubitId(location.qubits[1]));
                        // CZ = (I ⊗ H) CX (I ⊗ H)
                        prop.h(&[q2]).cx(&[q1, q2]).h(&[q2]);
                    }
                }

                // Measurements
                GateType::Measure | GateType::MeasureFree | GateType::MeasureLeaked => {
                    if !location.qubits.is_empty() {
                        let q = location.qubits[0];
                        // Check if this fault would flip the measurement
                        // A Z-basis measurement is flipped iff there's an X component
                        if prop.contains_x(q)
                            && let Some(&meas_idx) = measurement_positions.get(&loc_idx)
                        {
                            affected_measurements.insert(meas_idx);
                        }
                    }
                }

                // Other gates: pass through unchanged
                _ => {}
            }
        }

        affected_measurements
    }

    /// Propagate a two-qubit Pauli fault forward through the circuit.
    fn propagate_two_qubit_fault(
        pauli1: Pauli,
        qubit1: usize,
        pauli2: Pauli,
        qubit2: usize,
        start_loc: usize,
        all_gates: &[GateLocation],
        measurement_positions: &std::collections::HashMap<usize, usize>,
    ) -> BTreeSet<usize> {
        let mut prop = PauliProp::new();

        // Add both Paulis
        match pauli1 {
            Pauli::X => prop.add_x(qubit1),
            Pauli::Y => prop.add_y(qubit1),
            Pauli::Z => prop.add_z(qubit1),
        }
        match pauli2 {
            Pauli::X => prop.add_x(qubit2),
            Pauli::Y => prop.add_y(qubit2),
            Pauli::Z => prop.add_z(qubit2),
        }

        let mut affected_measurements = BTreeSet::new();

        // Propagate through subsequent gates (same logic as single-qubit)
        for (loc_idx, location) in all_gates.iter().enumerate().skip(start_loc) {
            match location.gate_type {
                GateType::H => {
                    if !location.qubits.is_empty() {
                        prop.h(&[QubitId(location.qubits[0])]);
                    }
                }
                GateType::SZ => {
                    if !location.qubits.is_empty() {
                        prop.sz(&[QubitId(location.qubits[0])]);
                    }
                }
                GateType::SZdg => {
                    if !location.qubits.is_empty() {
                        let q = QubitId(location.qubits[0]);
                        prop.sz(&[q]).sz(&[q]).sz(&[q]);
                    }
                }
                GateType::CX => {
                    if location.qubits.len() >= 2 {
                        prop.cx(&[QubitId(location.qubits[0]), QubitId(location.qubits[1])]);
                    }
                }
                GateType::CY => {
                    if location.qubits.len() >= 2 {
                        let (q1, q2) = (QubitId(location.qubits[0]), QubitId(location.qubits[1]));
                        prop.sz(&[q2]).cx(&[q1, q2]).sz(&[q2]).sz(&[q2]).sz(&[q2]);
                    }
                }
                GateType::CZ => {
                    if location.qubits.len() >= 2 {
                        let (q1, q2) = (QubitId(location.qubits[0]), QubitId(location.qubits[1]));
                        prop.h(&[q2]).cx(&[q1, q2]).h(&[q2]);
                    }
                }
                GateType::Measure | GateType::MeasureFree | GateType::MeasureLeaked => {
                    if !location.qubits.is_empty() {
                        let q = location.qubits[0];
                        if prop.contains_x(q)
                            && let Some(&meas_idx) = measurement_positions.get(&loc_idx)
                        {
                            affected_measurements.insert(meas_idx);
                        }
                    }
                }
                _ => {}
            }
        }

        affected_measurements
    }
}

/// Sampler for noisy measurement histories.
///
/// This sampler extends the regular measurement sampler to handle fault events.
/// For each sample:
/// 1. Fault bits are sampled with Bernoulli(probability) for each fault event
/// 2. Random bits are sampled 50/50 for non-deterministic measurements
/// 3. Each measurement is computed as: flip ^ `XOR(measurement_deps)` ^ `XOR(fault_deps)`
///    where for non-deterministic measurements, the XOR includes a random bit
///
/// The fault bits are "hidden" - they affect measurement outcomes but are not
/// directly returned. Only measurement outcomes are returned.
#[derive(Clone, Debug)]
pub struct NoisyMeasurementSampler<'a> {
    /// Reference to the noisy measurement history
    history: &'a NoisyMeasurementHistory,
}

impl<'a> NoisyMeasurementSampler<'a> {
    /// Create a new sampler from a noisy measurement history.
    #[must_use]
    pub fn new(history: &'a NoisyMeasurementHistory) -> Self {
        Self { history }
    }

    /// Returns the number of measurements per shot.
    #[inline]
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.history.num_measurements()
    }

    /// Returns the number of fault events.
    #[inline]
    #[must_use]
    pub fn num_faults(&self) -> usize {
        self.history.num_faults()
    }

    /// Sample measurement outcomes.
    ///
    /// Uses [`PecosRng`] for high performance random number generation.
    ///
    /// # Arguments
    /// * `shots` - Number of measurement shots to generate
    ///
    /// # Returns
    /// A [`SampleResult`] containing the sampled measurement outcomes.
    #[inline]
    #[must_use]
    pub fn sample(&self, shots: usize) -> SampleResult {
        let mut rng: PecosRng = rand::make_rng();
        self.sample_with_rng(shots, &mut rng)
    }

    /// Sample measurement outcomes with a specific seed for reproducibility.
    ///
    /// # Arguments
    /// * `shots` - Number of measurement shots to generate
    /// * `seed` - Seed for the random number generator
    ///
    /// # Returns
    /// A [`SampleResult`] containing the sampled measurement outcomes.
    #[inline]
    #[must_use]
    pub fn sample_with_seed(&self, shots: usize, seed: u64) -> SampleResult {
        let mut rng = PecosRng::seed_from_u64(seed);
        self.sample_with_rng(shots, &mut rng)
    }

    /// Sample measurement outcomes with a custom random number generator.
    ///
    /// This is the core sampling method, optimized using SIMD (u64x4) to process
    /// 256 shots at a time. It uses classified measurements to enable special-case
    /// handling for Fixed, Copy, and other patterns.
    ///
    /// # Arguments
    /// * `shots` - Number of measurement shots to generate
    /// * `rng` - Random number generator to use
    ///
    /// # Returns
    /// A [`SampleResult`] containing the sampled measurement outcomes.
    #[must_use]
    pub fn sample_with_rng<R: Rng + RngBulkExt>(&self, shots: usize, rng: &mut R) -> SampleResult {
        let num_measurements = self.history.num_measurements();

        if num_measurements == 0 || shots == 0 {
            return SampleResult::new(vec![Vec::new(); num_measurements], shots);
        }

        let num_words = shots.div_ceil(64);
        let num_simd_words = num_words.div_ceil(4);
        let num_faults = self.history.num_faults();

        // Classify measurements for optimized processing
        let kinds = self.history.classify_measurements();

        // Get fault probabilities
        let fault_probs: Vec<f64> = self
            .history
            .faults()
            .iter()
            .map(|f| f.probability)
            .collect();

        // Pre-generate all random columns using bulk fill
        // We only need columns for measurements that need fresh random bits
        let mut random_columns: Vec<Vec<u64x4>> = Vec::with_capacity(num_measurements);
        for kind in &kinds {
            if matches!(
                kind,
                NoisyMeasurementKind::PureRandom(_) | NoisyMeasurementKind::PureRandomFlipped(_)
            ) {
                random_columns.push(generate_random_column_simd(num_simd_words, rng));
            } else if let NoisyMeasurementKind::Computed { random_deps, .. } = kind {
                // For computed, check if we need any random columns that aren't generated yet
                for &r_idx in random_deps {
                    while random_columns.len() <= r_idx {
                        random_columns.push(generate_random_column_simd(num_simd_words, rng));
                    }
                }
            }
        }
        // Ensure we have random columns for all potential indices
        while random_columns.len() < num_measurements {
            random_columns.push(generate_random_column_simd(num_simd_words, rng));
        }

        // Pre-generate all fault columns using SIMD Bernoulli sampling
        let fault_columns: Vec<Vec<u64x4>> = fault_probs
            .iter()
            .map(|&p| sample_bernoulli_column_simd(rng, p, shots, num_simd_words))
            .collect();

        // Build measurement columns using classification
        let mut columns: Vec<Vec<u64x4>> = Vec::with_capacity(num_measurements);

        for kind in &kinds {
            let col = match kind {
                NoisyMeasurementKind::Fixed(value) => {
                    let fill = if *value {
                        u64x4::splat(!0u64)
                    } else {
                        u64x4::splat(0u64)
                    };
                    vec![fill; num_simd_words]
                }
                NoisyMeasurementKind::PureRandom(r_idx) => random_columns[*r_idx].clone(),
                NoisyMeasurementKind::PureRandomFlipped(r_idx) => {
                    random_columns[*r_idx].iter().map(|v| !*v).collect()
                }
                NoisyMeasurementKind::Copy(src) => columns[*src].clone(),
                NoisyMeasurementKind::CopyFlipped(src) => {
                    columns[*src].iter().map(|v| !*v).collect()
                }
                NoisyMeasurementKind::Computed {
                    random_deps,
                    fault_deps,
                    flip,
                } => {
                    let init = if *flip {
                        u64x4::splat(!0u64)
                    } else {
                        u64x4::splat(0u64)
                    };
                    let mut result = vec![init; num_simd_words];

                    // XOR with random dependencies
                    for &r_idx in random_deps {
                        let src_col = &random_columns[r_idx];
                        for (r, s) in result.iter_mut().zip(src_col.iter()) {
                            *r ^= *s;
                        }
                    }

                    // XOR with fault dependencies
                    for &f_idx in fault_deps {
                        if f_idx < num_faults {
                            let src_col = &fault_columns[f_idx];
                            for (r, s) in result.iter_mut().zip(src_col.iter()) {
                                *r ^= *s;
                            }
                        }
                    }

                    result
                }
            };
            columns.push(col);
        }

        // Convert SIMD columns to u64 columns
        let u64_columns: Vec<Vec<u64>> = columns
            .into_iter()
            .map(|col| simd_column_to_u64_vec(col, num_words))
            .collect();

        SampleResult::new(u64_columns, shots)
    }
}

// ============================================================================
// SIMD Helper Functions
// ============================================================================

/// Convert a SIMD column to a u64 column.
#[inline]
fn simd_column_to_u64_vec(simd_col: Vec<u64x4>, num_words: usize) -> Vec<u64> {
    let mut result = Vec::with_capacity(num_words);
    for simd_val in simd_col {
        let arr: [u64; 4] = simd_val.into();
        for val in arr {
            if result.len() >= num_words {
                return result;
            }
            result.push(val);
        }
    }
    result
}

/// Generate a SIMD column of random bits using bulk fill.
#[inline]
fn generate_random_column_simd<R: Rng + RngBulkExt>(
    num_simd_words: usize,
    rng: &mut R,
) -> Vec<u64x4> {
    let mut column: Vec<u64x4> = vec![u64x4::splat(0); num_simd_words];

    // Safety: u64x4 is repr(C) containing 4 u64s
    let u64_slice: &mut [u64] = unsafe {
        std::slice::from_raw_parts_mut(column.as_mut_ptr().cast::<u64>(), num_simd_words * 4)
    };

    rng.fill_u64_bulk(u64_slice);
    column
}

/// Sample a SIMD column of Bernoulli random bits.
#[inline]
fn sample_bernoulli_column_simd<R: Rng>(
    rng: &mut R,
    p: f64,
    shots: usize,
    num_simd_words: usize,
) -> Vec<u64x4> {
    if p <= 0.0 {
        return vec![u64x4::splat(0); num_simd_words];
    }
    if p >= 1.0 {
        return vec![u64x4::splat(!0u64); num_simd_words];
    }

    let num_words = shots.div_ceil(64);
    let mut column: Vec<u64x4> = vec![u64x4::splat(0); num_simd_words];

    // Sample each u64 word
    for word_idx in 0..num_words {
        let simd_idx = word_idx / 4;
        let lane_idx = word_idx % 4;

        let shots_in_word = if word_idx == num_words - 1 {
            let remaining = shots % 64;
            if remaining == 0 { 64 } else { remaining }
        } else {
            64
        };

        let word = sample_bernoulli_word(rng, p, shots_in_word);

        // Insert into the correct lane
        let mut arr: [u64; 4] = column[simd_idx].into();
        arr[lane_idx] = word;
        column[simd_idx] = u64x4::from(arr);
    }

    column
}

/// Sample 64 Bernoulli random bits at once.
///
/// Uses a geometric distribution approach for efficiency when p is small.
#[inline]
fn sample_bernoulli_word<R: Rng>(rng: &mut R, p: f64, num_bits: usize) -> u64 {
    if p <= 0.0 {
        return 0;
    }
    if p >= 1.0 {
        return !0u64;
    }

    // For very small probabilities, use geometric distribution
    if p < 0.1 {
        sample_bernoulli_geometric(rng, p, num_bits)
    } else {
        // For larger probabilities, direct sampling is fine
        sample_bernoulli_direct(rng, p, num_bits)
    }
}

/// Direct Bernoulli sampling: compare each random float to probability.
#[inline]
fn sample_bernoulli_direct<R: Rng>(rng: &mut R, p: f64, num_bits: usize) -> u64 {
    let mut result = 0u64;
    for i in 0..num_bits {
        if rng.random::<f64>() < p {
            result |= 1u64 << i;
        }
    }
    result
}

/// Geometric distribution approach: sample gaps between 1-bits.
/// More efficient when p is small (few 1s expected).
#[inline]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
fn sample_bernoulli_geometric<R: Rng>(rng: &mut R, p: f64, num_bits: usize) -> u64 {
    let mut result = 0u64;
    let log_1_minus_p = (1.0 - p).ln();

    let mut pos = 0usize;
    loop {
        // Sample from geometric distribution: number of 0s before next 1
        let u: f64 = rng.random();
        if u <= 0.0 {
            continue; // Avoid log(0)
        }
        let skip = (u.ln() / log_1_minus_p).floor() as usize;
        pos += skip;

        if pos >= num_bits {
            break;
        }

        result |= 1u64 << pos;
        pos += 1;

        if pos >= num_bits {
            break;
        }
    }

    result
}

impl std::fmt::Display for NoisyMeasurementHistory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Format faults
        if !self.faults.is_empty() {
            writeln!(f, "Faults:")?;
            for (i, fault) in self.faults.iter().enumerate() {
                writeln!(
                    f,
                    "  f{i}: p={:.6} ({})",
                    fault.probability, fault.description
                )?;
            }
            writeln!(f)?;
        }

        // Format measurements
        writeln!(f, "Measurements:")?;
        for m in &self.measurements {
            writeln!(f, "  {m}")?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_noise_model_default() {
        let model = DepolarizingNoiseModel::default();
        assert!(model.is_noiseless());
    }

    #[test]
    fn test_noise_model_uniform() {
        let model = DepolarizingNoiseModel::uniform(0.01);
        assert!((model.p1 - 0.01).abs() < f64::EPSILON);
        assert!((model.p2 - 0.01).abs() < f64::EPSILON);
        assert!((model.p_meas - 0.01).abs() < f64::EPSILON);
        assert!((model.p_prep - 0.01).abs() < f64::EPSILON);
        assert!(!model.is_noiseless());
    }

    #[test]
    fn test_noisy_measurement_result_display() {
        // Deterministic with no dependencies
        let m = NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps: BitSet::new(),
            flip: false,
            is_deterministic: true,
            index: 0,
        };
        assert_eq!(format!("{m}"), "m0=0");

        // Deterministic with flip
        let m = NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps: BitSet::new(),
            flip: true,
            is_deterministic: true,
            index: 1,
        };
        assert_eq!(format!("{m}"), "m1=1");

        // With measurement dependency
        let mut deps = BitSet::new();
        deps.insert(0);
        let m = NoisyMeasurementResult {
            measurement_deps: deps,
            fault_deps: BitSet::new(),
            flip: false,
            is_deterministic: true,
            index: 2,
        };
        assert_eq!(format!("{m}"), "m2=m0");

        // With fault dependency
        let mut fault_deps = BitSet::new();
        fault_deps.insert(0);
        let m = NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps,
            flip: false,
            is_deterministic: true,
            index: 3,
        };
        assert_eq!(format!("{m}"), "m3=f0");

        // With both dependencies and flip
        let mut meas_deps = BitSet::new();
        meas_deps.insert(0);
        meas_deps.insert(1);
        let mut fault_deps = BitSet::new();
        fault_deps.insert(2);
        let m = NoisyMeasurementResult {
            measurement_deps: meas_deps,
            fault_deps,
            flip: true,
            is_deterministic: true,
            index: 4,
        };
        assert_eq!(format!("{m}"), "m4=m0^m1^f2^1");
    }

    #[test]
    fn test_add_fault() {
        let mut history = NoisyMeasurementHistory::new();

        // Add some measurements first
        history.measurements.push(NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps: BitSet::new(),
            flip: false,
            is_deterministic: true,
            index: 0,
        });
        history.measurements.push(NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps: BitSet::new(),
            flip: false,
            is_deterministic: true,
            index: 1,
        });
        history.measurements.push(NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps: BitSet::new(),
            flip: false,
            is_deterministic: true,
            index: 2,
        });

        // Add a fault that affects measurements 0 and 2
        let mut affected = BTreeSet::new();
        affected.insert(0);
        affected.insert(2);
        let fault_idx = history.add_fault(0.01, &affected, "X on q0".to_string());

        assert_eq!(fault_idx, 0);
        assert_eq!(history.num_faults(), 1);
        assert!(history.measurements[0].fault_deps.contains(0));
        assert!(!history.measurements[1].fault_deps.contains(0));
        assert!(history.measurements[2].fault_deps.contains(0));
    }

    #[test]
    fn test_noisy_sampler_no_faults() {
        // Create a simple history with deterministic measurements
        let mut history = NoisyMeasurementHistory::new();
        history.measurements.push(NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps: BitSet::new(),
            flip: false, // Should be 0
            is_deterministic: true,
            index: 0,
        });
        history.measurements.push(NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps: BitSet::new(),
            flip: true, // Should be 1
            is_deterministic: true,
            index: 1,
        });

        let sampler = NoisyMeasurementSampler::new(&history);
        let result = sampler.sample_with_seed(100, 42);

        assert_eq!(result.num_measurements(), 2);
        assert_eq!(result.shots(), 100);

        // All shots should have same deterministic outcomes
        for shot in 0..100 {
            assert!(!*result.get(shot, 0), "m0 should always be false");
            assert!(*result.get(shot, 1), "m1 should always be true");
        }
    }

    #[test]
    fn test_noisy_sampler_with_certain_fault() {
        // Create a history where a fault with probability 1.0 flips a measurement
        let mut history = NoisyMeasurementHistory::new();

        // m0 = 0 (deterministic, no fault)
        history.measurements.push(NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps: BitSet::new(),
            flip: false,
            is_deterministic: true,
            index: 0,
        });

        // m1 = f0 (deterministic, but depends on fault 0)
        let mut fault_deps = BitSet::new();
        fault_deps.insert(0);
        history.measurements.push(NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps,
            flip: false,
            is_deterministic: true,
            index: 1,
        });

        // Add a fault with probability 1.0 - it always occurs
        let mut affected = BTreeSet::new();
        affected.insert(1);
        history.add_fault(1.0, &affected, "Always fault".to_string());

        let sampler = NoisyMeasurementSampler::new(&history);
        let result = sampler.sample_with_seed(100, 42);

        // m0 should always be 0, m1 should always be 1 (flipped by fault)
        for shot in 0..100 {
            assert!(!*result.get(shot, 0), "m0 should be false");
            assert!(
                *result.get(shot, 1),
                "m1 should be true (flipped by certain fault)"
            );
        }
    }

    #[test]
    fn test_noisy_sampler_with_zero_probability_fault() {
        // Create a history where a fault with probability 0.0 should never occur
        let mut history = NoisyMeasurementHistory::new();

        // m0 = 0 ^ f0, but f0 never occurs
        let mut fault_deps = BitSet::new();
        fault_deps.insert(0);
        history.measurements.push(NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps,
            flip: false,
            is_deterministic: true,
            index: 0,
        });

        // Add a fault with probability 0.0 - it never occurs
        let mut affected = BTreeSet::new();
        affected.insert(0);
        history.add_fault(0.0, &affected, "Never fault".to_string());

        let sampler = NoisyMeasurementSampler::new(&history);
        let result = sampler.sample_with_seed(100, 42);

        // m0 should always be 0 (fault never flips it)
        for shot in 0..100 {
            assert!(!*result.get(shot, 0), "m0 should always be false");
        }
    }

    #[test]
    fn test_noisy_sampler_statistical_fault_rate() {
        // Create a history with a fault at 50% probability
        let mut history = NoisyMeasurementHistory::new();

        // m0 = f0 (depends only on fault 0)
        let mut fault_deps = BitSet::new();
        fault_deps.insert(0);
        history.measurements.push(NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps,
            flip: false,
            is_deterministic: true,
            index: 0,
        });

        // Add a fault with 50% probability
        let mut affected = BTreeSet::new();
        affected.insert(0);
        history.add_fault(0.5, &affected, "50% fault".to_string());

        let sampler = NoisyMeasurementSampler::new(&history);
        let result = sampler.sample_with_seed(10000, 42);

        // Count how many times m0 is 1 (which happens when fault occurs)
        let count = result.count_ones(0);

        // With 10000 shots and p=0.5, expect ~5000 ± 200 (within 2%)
        assert!(
            (4800..=5200).contains(&count),
            "Expected ~5000 faults, got {count}"
        );
    }

    #[test]
    fn test_noisy_sampler_with_measurement_and_fault_deps() {
        // m0 = random (50/50)
        // m1 = m0 ^ f0 (depends on m0 and fault 0)
        // If f0 = 0: m1 = m0 (correlated like Bell state)
        // If f0 = 1: m1 = !m0 (anti-correlated)

        let mut history = NoisyMeasurementHistory::new();

        // m0 = random
        history.measurements.push(NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps: BitSet::new(),
            flip: false,
            is_deterministic: false, // Random
            index: 0,
        });

        // m1 = m0 ^ f0
        let mut meas_deps = BitSet::new();
        meas_deps.insert(0);
        let mut fault_deps = BitSet::new();
        fault_deps.insert(0);
        history.measurements.push(NoisyMeasurementResult {
            measurement_deps: meas_deps,
            fault_deps,
            flip: false,
            is_deterministic: true,
            index: 1,
        });

        // Add fault with 50% probability
        let mut affected = BTreeSet::new();
        affected.insert(1);
        history.add_fault(0.5, &affected, "Decorrelation fault".to_string());

        let sampler = NoisyMeasurementSampler::new(&history);
        let result = sampler.sample_with_seed(10000, 42);

        // Count outcomes
        let mut count_00 = 0;
        let mut count_01 = 0;
        let mut count_10 = 0;
        let mut count_11 = 0;

        for shot in 0..10000 {
            let m0 = *result.get(shot, 0);
            let m1 = *result.get(shot, 1);
            match (m0, m1) {
                (false, false) => count_00 += 1,
                (false, true) => count_01 += 1,
                (true, false) => count_10 += 1,
                (true, true) => count_11 += 1,
            }
        }

        // With p_fault = 0.5 and random m0:
        // - P(00) = P(m0=0) * P(f0=0) = 0.5 * 0.5 = 0.25
        // - P(01) = P(m0=0) * P(f0=1) = 0.5 * 0.5 = 0.25
        // - P(10) = P(m0=1) * P(f0=1) = 0.5 * 0.5 = 0.25
        // - P(11) = P(m0=1) * P(f0=0) = 0.5 * 0.5 = 0.25
        // All four outcomes should be ~25% (2500 ± 200)
        assert!(
            (2300..=2700).contains(&count_00),
            "Expected ~2500 for 00, got {count_00}"
        );
        assert!(
            (2300..=2700).contains(&count_01),
            "Expected ~2500 for 01, got {count_01}"
        );
        assert!(
            (2300..=2700).contains(&count_10),
            "Expected ~2500 for 10, got {count_10}"
        );
        assert!(
            (2300..=2700).contains(&count_11),
            "Expected ~2500 for 11, got {count_11}"
        );
    }

    #[test]
    fn test_builder_noiseless_same_as_from_noiseless() {
        // Create a simple measurement history
        let mut history = NoisyMeasurementHistory::new();
        history.measurements.push(NoisyMeasurementResult {
            measurement_deps: BitSet::new(),
            fault_deps: BitSet::new(),
            flip: false,
            is_deterministic: true,
            index: 0,
        });

        // Use from_noiseless (noiseless) - should have no faults
        assert_eq!(history.num_faults(), 0);
        assert_eq!(history.num_measurements(), 1);
    }

    #[test]
    fn test_propagate_x_fault_direct_to_measurement() {
        // Test that an X fault on qubit 0 right before measurement 0 affects m0
        use pecos_qsim::PauliProp;

        let mut prop = PauliProp::new();
        prop.add_x(0);

        // X on qubit 0 should flip Z-basis measurement on qubit 0
        assert!(prop.contains_x(0));
    }

    #[test]
    fn test_propagate_z_fault_no_flip() {
        // Test that a Z fault doesn't flip Z-basis measurements directly
        use pecos_qsim::PauliProp;

        let mut prop = PauliProp::new();
        prop.add_z(0);

        // Z on qubit 0 should NOT flip Z-basis measurement on qubit 0
        assert!(!prop.contains_x(0));
    }

    #[test]
    fn test_propagate_x_through_h_becomes_z() {
        // Test that X -> H -> Z (doesn't flip)
        use pecos_core::QubitId;
        use pecos_qsim::{CliffordGateable, PauliProp};

        let mut prop = PauliProp::new();
        prop.add_x(0);
        prop.h(&[QubitId(0)]);

        // After H, X becomes Z, which doesn't flip Z-basis measurement
        assert!(!prop.contains_x(0));
        assert!(prop.contains_z(0));
    }

    #[test]
    fn test_propagate_z_through_h_becomes_x() {
        // Test that Z -> H -> X (does flip)
        use pecos_qsim::{CliffordGateable, PauliProp};

        let mut prop = PauliProp::new();
        prop.add_z(0);
        prop.h(&[QubitId(0)]);

        // After H, Z becomes X, which does flip Z-basis measurement
        assert!(prop.contains_x(0));
        assert!(!prop.contains_z(0));
    }

    #[test]
    fn test_propagate_x_through_cx_spreads() {
        // Test that X on control of CX spreads to target
        use pecos_qsim::{CliffordGateable, PauliProp};

        let mut prop = PauliProp::new();
        prop.add_x(0); // X on control
        prop.cx(&[QubitId(0), QubitId(1)]);

        // X on control propagates to target: XI -> XX
        assert!(prop.contains_x(0));
        assert!(prop.contains_x(1));
    }

    #[test]
    fn test_propagate_z_through_cx_spreads() {
        // Test that Z on target of CX spreads to control
        use pecos_qsim::{CliffordGateable, PauliProp};

        let mut prop = PauliProp::new();
        prop.add_z(1); // Z on target
        prop.cx(&[QubitId(0), QubitId(1)]);

        // Z on target propagates to control: IZ -> ZZ
        assert!(prop.contains_z(0));
        assert!(prop.contains_z(1));
    }
}
