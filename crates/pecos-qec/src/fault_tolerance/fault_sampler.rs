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

//! Stochastic raw-measurement sampling via fault table overlay.
//!
//! # Architecture
//!
//! Raw measurement output = ideal measurement values XOR sampled physical faults.
//!
//! These are computed independently:
//! - **Ideal values** from [`MeasurementSampler`](pecos_simulators::measurement_sampler::MeasurementSampler),
//!   which respects the Copy/Computed dependency graph from symbolic simulation.
//!   Non-deterministic measurements share latent random variables through the
//!   stabilizer eigenvalue structure.
//! - **Physical faults** from a fault table where each entry has a probability
//!   and a set of affected measurements. Faults are sampled independently per
//!   shot (Bernoulli) and XOR'd onto the ideal values.
//!
//! This separation is critical: the dependency graph captures *ideal* measurement
//! correlations (same stabilizer across resets), while fault events represent
//! *physical* noise processes (gate errors, measurement flips, prep errors).
//! Mixing them — e.g., flattening fault deps through Copy chains — incorrectly
//! cancels faults that affect only one measurement in a correlated pair.

use pecos_core::gate_type::GateType;
use pecos_core::pauli::pauli_string::PauliString;
use pecos_core::{Pauli, QubitId};
use pecos_quantum::{AnnotationKind, TickCircuit};
use pecos_random::{PecosRng, RngExt};
use pecos_simulators::measurement_sampler::{MeasurementKind, SampleResult};
use pecos_simulators::symbolic_sparse_stab::MeasurementHistory;
use pecos_simulators::{BitmaskPauliProp, CliffordGateable};
use std::collections::{BTreeSet, HashMap};
use std::fmt;

/// Error returned when `build_fault_table` encounters an unsupported gate.
#[derive(Clone, Debug)]
pub struct UnsupportedGateError {
    pub gate_type: GateType,
    pub tick: usize,
    pub gate_in_tick: usize,
    pub qubits: Vec<usize>,
}

impl fmt::Display for UnsupportedGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Unsupported gate {:?} at tick {} gate {} on qubits {:?}. \
             Supported: H, X, Y, Z, SZ, SZdg, SX, SXdg, SY, SYdg, F, Fdg, \
             CX, CY, CZ, SXX, SXXdg, SYY, SYYdg, SZZ, SZZdg, SWAP, \
             MZ/MeasureFree/MeasureLeaked, PZ, QAlloc, QFree, I, Idle, \
             plus metadata (MeasCrosstalk*, TrackedPauliMeta).",
            self.gate_type, self.tick, self.gate_in_tick, self.qubits
        )
    }
}

impl std::error::Error for UnsupportedGateError {}

/// Standard single-qubit Clifford gates supported by `CliffordGateable`.
pub const STANDARD_1Q_CLIFFORD_GATES: &[GateType] = &[
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

/// Standard two-qubit Clifford gates supported by `CliffordGateable`.
pub const STANDARD_2Q_CLIFFORD_GATES: &[GateType] = &[
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

#[inline]
fn is_standard_1q_clifford_gate(gate_type: GateType) -> bool {
    STANDARD_1Q_CLIFFORD_GATES.contains(&gate_type)
}

#[inline]
fn is_standard_2q_clifford_gate(gate_type: GateType) -> bool {
    STANDARD_2Q_CLIFFORD_GATES.contains(&gate_type)
}

#[inline]
fn is_supported_measurement_gate(gate_type: GateType) -> bool {
    matches!(
        gate_type,
        GateType::MZ | GateType::MeasureFree | GateType::MeasureLeaked
    )
}

#[inline]
fn is_supported_prep_gate(gate_type: GateType) -> bool {
    matches!(gate_type, GateType::PZ | GateType::QAlloc)
}

#[inline]
fn is_supported_noop_or_metadata_gate(gate_type: GateType) -> bool {
    matches!(
        gate_type,
        GateType::QFree
            | GateType::I
            | GateType::Idle
            | GateType::MeasCrosstalkGlobalPayload
            | GateType::MeasCrosstalkLocalPayload
            | GateType::TrackedPauliMeta
    )
}

/// A fault mechanism: fires with probability `p`, then uniformly selects one
/// of its alternatives to determine which measurements are flipped.
///
/// For a depolarizing channel with k non-identity Paulis and total error
/// probability p: the mechanism fires with probability p, then each of the
/// k alternatives is chosen with probability 1/k. This matches the stabilizer
/// sim's "exactly one Pauli error per gate event" semantics.
#[derive(Clone, Debug, PartialEq)]
pub struct FaultMechanism {
    /// Total probability that this mechanism fires (one Bernoulli per shot).
    pub probability: f64,
    /// Each alternative is a set of measurements that get flipped if that
    /// alternative is selected. Empty alternatives (no measurements flipped)
    /// are preserved — they represent Pauli errors that commute with all
    /// subsequent measurements (e.g., Z after MZ). Keeping them maintains
    /// the correct 1/k uniform denominator for the depolarizing channel.
    pub alternatives: Vec<Vec<usize>>,
}

/// Noise parameters for depolarizing fault injection.
#[derive(Clone, Debug)]
pub struct StochasticNoiseParams {
    pub p1: f64,
    pub p2: f64,
    pub p_meas: f64,
    pub p_prep: f64,
}

/// A gate in the flattened gate list (one entry per qubit-pair or single qubit).
#[derive(Clone, Debug)]
pub(crate) struct GateLoc {
    pub(crate) tick: usize,
    pub(crate) gate_index: usize,
    pub(crate) gate_type: GateType,
    pub(crate) qubits: Vec<usize>,
}

/// Single-qubit Pauli type for fault injection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) enum PauliType {
    X,
    Y,
    Z,
}

/// Build a fault table from a `TickCircuit` and noise parameters.
///
/// Each entry describes one possible fault mechanism: its probability and
/// which measurements it would flip if it occurs. The table is used for
/// independent per-shot Bernoulli sampling.
///
/// Gate ordering follows the `TickCircuit` tick-by-tick structure, which must
/// match the measurement numbering used by detector/DEM-output record indices.
///
/// # Supported gates
///
/// **Fault injection** (noise applied after these gates):
/// - Single-qubit Clifford: `H`, `X`, `Y`, `Z`, `SZ`, `SZdg`, `SX`, `SXdg`,
///   `SY`, `SYdg`, `F`, `Fdg` → `p=p1`, 3 alternatives
/// - Two-qubit Clifford: `CX`, `CY`, `CZ`, `SXX`, `SXXdg`, `SYY`, `SYYdg`,
///   `SZZ`, `SZZdg`, `SWAP` → `p=p2`, 15 alternatives
/// - State preparation: `PZ`, `QAlloc` → mechanism with `p=p_prep`, 1 alternative (`X`)
/// - Measurement: `MZ`, `MeasureFree`, `MeasureLeaked` → mechanism with
///   `p=p_meas`, 1 alternative (flip)
///
/// Each mechanism fires at most once per shot (Bernoulli with total probability p).
/// When it fires, exactly one alternative is chosen uniformly at random. This
/// matches the depolarizing channel semantics: "with probability p, apply one
/// of the k non-identity Paulis, each equally likely."
///
/// **Propagation** (gates that transform a propagating Pauli):
/// - All single-qubit Cliffords: Clifford conjugation via direct Pauli-basis updates
/// - All two-qubit Cliffords: Clifford conjugation via direct Pauli-basis updates
/// - `PZ`, `QAlloc`: absorbs all Pauli components on the reset qubit
/// - `MZ`: records `X`-component flip, then absorbs all components (state collapse)
///
/// **No-op** (pass through without noise or transformation):
/// - `I`, `Idle`, `QFree`, `MeasCrosstalkGlobalPayload`,
///   `MeasCrosstalkLocalPayload`, `TrackedPauliMeta`
///
/// Any gate not in the above lists returns [`UnsupportedGateError`].
///
/// # Errors
///
/// Returns [`UnsupportedGateError`] when the circuit contains a gate outside
/// the supported Clifford/prep/measurement/metadata set.
pub fn build_fault_table(
    tc: &TickCircuit,
    noise: &StochasticNoiseParams,
) -> Result<Vec<FaultMechanism>, UnsupportedGateError> {
    let mut catalog = FaultCatalog::from_circuit(tc)?;
    catalog.with_noise(noise);
    Ok(catalog.to_mechanisms())
}

/// Validate that all gates in the `TickCircuit` are supported (before flattening).
fn validate_tick_circuit(tc: &TickCircuit) -> Result<(), UnsupportedGateError> {
    for (tick_idx, tick) in tc.iter_ticks() {
        for gate in tick.iter_gate_batches() {
            if is_standard_1q_clifford_gate(gate.gate_type)
                || is_standard_2q_clifford_gate(gate.gate_type)
                || is_supported_measurement_gate(gate.gate_type)
                || is_supported_prep_gate(gate.gate_type)
                || is_supported_noop_or_metadata_gate(gate.gate_type)
            {
                continue;
            }
            return Err(UnsupportedGateError {
                gate_type: gate.gate_type,
                tick: tick_idx,
                gate_in_tick: gate.batch_index(),
                qubits: gate.qubits.iter().map(pecos_core::QubitId::index).collect(),
            });
        }
    }
    Ok(())
}

/// Flatten a `TickCircuit` into individual gate applications with measurement
/// position tracking.
///
/// Stored batches are expanded through `TickCircuit`'s `GateInstanceRef`
/// iterator so the qubit/measurement-id slicing semantics are shared with
/// other consumers. Each measurement and each multi-qubit pair gets its own
/// position for fault injection. Returns the gate list and a map from gate-list
/// index to measurement index.
pub(crate) fn flatten_tick_circuit(tc: &TickCircuit) -> (Vec<GateLoc>, HashMap<usize, usize>) {
    let mut gates = Vec::new();
    let mut meas_positions = HashMap::new();
    let mut meas_count = 0usize;

    for (tick_idx, tick) in tc.iter_ticks() {
        for gate in tick.iter_gate_instances() {
            let qs: Vec<usize> = gate
                .qubits()
                .iter()
                .map(pecos_core::QubitId::index)
                .collect();
            if is_supported_measurement_gate(gate.gate_type()) {
                meas_positions.insert(gates.len(), meas_count);
                meas_count += 1;
            }
            gates.push(GateLoc {
                tick: tick_idx,
                gate_index: gate.batch_index(),
                gate_type: gate.gate_type(),
                qubits: qs,
            });
        }
    }

    (gates, meas_positions)
}

/// Propagate a single-qubit Pauli fault forward through the gate list.
///
/// Returns the set of measurement indices whose outcomes would be flipped
/// by this Pauli error at this position.
#[cfg(test)]
pub(crate) fn propagate_single(
    pauli: PauliType,
    qubit: usize,
    start: usize,
    gates: &[GateLoc],
    meas_positions: &HashMap<usize, usize>,
) -> BTreeSet<usize> {
    let mut prop = BitmaskPauliProp::new();
    match pauli {
        PauliType::X => prop.track_x(&[qubit]),
        PauliType::Y => prop.track_y(&[qubit]),
        PauliType::Z => prop.track_z(&[qubit]),
    }

    propagate_forward(&mut prop, start, gates, meas_positions)
}

fn propagate_single_effect(
    pauli: PauliType,
    qubit: usize,
    start: usize,
    gates: &[GateLoc],
    meas_positions: &HashMap<usize, usize>,
    tracked_paulis: &[PauliString],
) -> PropagatedFaultEffect {
    let mut prop = BitmaskPauliProp::new();
    match pauli {
        PauliType::X => prop.track_x(&[qubit]),
        PauliType::Y => prop.track_y(&[qubit]),
        PauliType::Z => prop.track_z(&[qubit]),
    }

    let affected_measurements = propagate_forward(&mut prop, start, gates, meas_positions);
    let affected_tracked_paulis = tracked_paulis_flipped_by(&prop, tracked_paulis);
    PropagatedFaultEffect {
        affected_measurements,
        affected_tracked_paulis,
    }
}

#[cfg(test)]
fn propagate_pair_effect(
    faults: [(PauliType, usize); 2],
    start: usize,
    gates: &[GateLoc],
    meas_positions: &HashMap<usize, usize>,
    tracked_paulis: &[PauliString],
) -> PropagatedFaultEffect {
    let mut prop = BitmaskPauliProp::new();
    for (pauli, qubit) in faults {
        match pauli {
            PauliType::X => prop.track_x(&[qubit]),
            PauliType::Y => prop.track_y(&[qubit]),
            PauliType::Z => prop.track_z(&[qubit]),
        }
    }

    let affected_measurements = propagate_forward(&mut prop, start, gates, meas_positions);
    let affected_tracked_paulis = tracked_paulis_flipped_by(&prop, tracked_paulis);
    PropagatedFaultEffect {
        affected_measurements,
        affected_tracked_paulis,
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct PropagatedFaultEffect {
    affected_measurements: BTreeSet<usize>,
    affected_tracked_paulis: Vec<usize>,
}

#[derive(Default)]
struct PropagatedEffectCache {
    singles: HashMap<(usize, PauliType, usize), PropagatedFaultEffect>,
}

impl PropagatedEffectCache {
    #[cfg(test)]
    fn len(&self) -> usize {
        self.singles.len()
    }

    fn single(
        &mut self,
        pauli: PauliType,
        qubit: usize,
        start: usize,
        gates: &[GateLoc],
        meas_positions: &HashMap<usize, usize>,
        tracked_paulis: &[PauliString],
    ) -> PropagatedFaultEffect {
        self.singles
            .entry((start, pauli, qubit))
            .or_insert_with(|| {
                propagate_single_effect(pauli, qubit, start, gates, meas_positions, tracked_paulis)
            })
            .clone()
    }
}

fn xor_fault_effects(
    left: &PropagatedFaultEffect,
    right: &PropagatedFaultEffect,
) -> PropagatedFaultEffect {
    let mut affected_measurements = left.affected_measurements.clone();
    for &measurement in &right.affected_measurements {
        if !affected_measurements.remove(&measurement) {
            affected_measurements.insert(measurement);
        }
    }

    PropagatedFaultEffect {
        affected_measurements,
        affected_tracked_paulis: xor_sorted_unique_indices(
            &left.affected_tracked_paulis,
            &right.affected_tracked_paulis,
        ),
    }
}

fn xor_sorted_unique_indices(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut out = Vec::with_capacity(left.len() + right.len());
    let mut i = 0usize;
    let mut j = 0usize;
    while i < left.len() && j < right.len() {
        match left[i].cmp(&right[j]) {
            std::cmp::Ordering::Less => {
                out.push(left[i]);
                i += 1;
            }
            std::cmp::Ordering::Greater => {
                out.push(right[j]);
                j += 1;
            }
            std::cmp::Ordering::Equal => {
                i += 1;
                j += 1;
            }
        }
    }
    out.extend_from_slice(&left[i..]);
    out.extend_from_slice(&right[j..]);
    out
}

/// Core forward propagation: evolve a Pauli through gates, collecting affected measurements.
fn propagate_forward(
    prop: &mut BitmaskPauliProp,
    start: usize,
    gates: &[GateLoc],
    meas_positions: &HashMap<usize, usize>,
) -> BTreeSet<usize> {
    let mut affected = BTreeSet::new();

    for (loc_idx, loc) in gates.iter().enumerate().skip(start) {
        match loc.gate_type {
            GateType::H if !loc.qubits.is_empty() => {
                prop.h(&[QubitId(loc.qubits[0])]);
            }
            GateType::SZ if !loc.qubits.is_empty() => {
                prop.sz(&[QubitId(loc.qubits[0])]);
            }
            GateType::SZdg if !loc.qubits.is_empty() => {
                let q = QubitId(loc.qubits[0]);
                prop.szdg(&[q]);
            }
            GateType::SX if !loc.qubits.is_empty() => {
                prop.sx(&[QubitId(loc.qubits[0])]);
            }
            GateType::SXdg if !loc.qubits.is_empty() => {
                prop.sxdg(&[QubitId(loc.qubits[0])]);
            }
            GateType::SY if !loc.qubits.is_empty() => {
                prop.sy(&[QubitId(loc.qubits[0])]);
            }
            GateType::SYdg if !loc.qubits.is_empty() => {
                prop.sydg(&[QubitId(loc.qubits[0])]);
            }
            GateType::F if !loc.qubits.is_empty() => {
                prop.f(&[QubitId(loc.qubits[0])]);
            }
            GateType::Fdg if !loc.qubits.is_empty() => {
                prop.fdg(&[QubitId(loc.qubits[0])]);
            }
            GateType::CX if loc.qubits.len() >= 2 => {
                prop.cx(&[(QubitId(loc.qubits[0]), QubitId(loc.qubits[1]))]);
            }
            GateType::CY if loc.qubits.len() >= 2 => {
                let (q1, q2) = (QubitId(loc.qubits[0]), QubitId(loc.qubits[1]));
                prop.cy(&[(q1, q2)]);
            }
            GateType::CZ if loc.qubits.len() >= 2 => {
                let (q1, q2) = (QubitId(loc.qubits[0]), QubitId(loc.qubits[1]));
                prop.cz(&[(q1, q2)]);
            }
            GateType::SXX if loc.qubits.len() >= 2 => {
                let pair = [(QubitId(loc.qubits[0]), QubitId(loc.qubits[1]))];
                prop.sxx(&pair);
            }
            GateType::SXXdg if loc.qubits.len() >= 2 => {
                let pair = [(QubitId(loc.qubits[0]), QubitId(loc.qubits[1]))];
                prop.sxxdg(&pair);
            }
            GateType::SYY if loc.qubits.len() >= 2 => {
                let pair = [(QubitId(loc.qubits[0]), QubitId(loc.qubits[1]))];
                prop.syy(&pair);
            }
            GateType::SYYdg if loc.qubits.len() >= 2 => {
                let pair = [(QubitId(loc.qubits[0]), QubitId(loc.qubits[1]))];
                prop.syydg(&pair);
            }
            GateType::SZZ if loc.qubits.len() >= 2 => {
                let pair = [(QubitId(loc.qubits[0]), QubitId(loc.qubits[1]))];
                prop.szz(&pair);
            }
            GateType::SZZdg if loc.qubits.len() >= 2 => {
                let pair = [(QubitId(loc.qubits[0]), QubitId(loc.qubits[1]))];
                prop.szzdg(&pair);
            }
            GateType::SWAP if loc.qubits.len() >= 2 => {
                let pair = [(QubitId(loc.qubits[0]), QubitId(loc.qubits[1]))];
                prop.swap(&pair);
            }
            // PZ/QAlloc absorbs propagating errors on the reset qubit
            GateType::PZ | GateType::QAlloc if !loc.qubits.is_empty() => {
                prop.clear_qubit(loc.qubits[0]);
            }
            // MZ: X component flips the measurement, then qubit state collapses
            GateType::MZ | GateType::MeasureFree | GateType::MeasureLeaked
                if !loc.qubits.is_empty() =>
            {
                let q = loc.qubits[0];
                if prop.contains_x(q)
                    && let Some(&meas_idx) = meas_positions.get(&loc_idx)
                {
                    affected.insert(meas_idx);
                }
                prop.clear_qubit(q);
            }
            _ => {}
        }

        if prop.is_identity() {
            break;
        }
    }

    affected
}

// ============================================================================
// Fault Catalog: per-location, per-alternative lookup table
// ============================================================================

/// The kind of physical fault mechanism.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FaultKind {
    /// A Pauli error injected after a gate.
    Pauli,
    /// A measurement outcome flip.
    MeasurementFlip,
    /// A preparation error (X on |0⟩).
    PrepFlip,
}

/// Which noise channel produced this fault location.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FaultChannel {
    /// Single-qubit depolarizing (`p1`).
    P1,
    /// Two-qubit depolarizing (`p2`).
    P2,
    /// Measurement flip (`p_meas`).
    PMeas,
    /// State preparation flip (`p_prep`).
    PPrep,
}

/// One alternative within a physical fault location.
#[derive(Clone, Debug)]
pub struct FaultAlternative {
    /// Kind of fault.
    pub kind: FaultKind,
    /// The Pauli error for this alternative (None for measurement/prep faults).
    pub pauli: Option<PauliString>,
    /// Raw measurement indices flipped by this fault.
    pub affected_measurements: Vec<usize>,
    /// Detector indices flipped (computed from measurement effects + detector records).
    pub affected_detectors: Vec<usize>,
    /// Observable indices flipped.
    pub affected_observables: Vec<usize>,
    /// Tracked-Pauli indices flipped.
    pub affected_tracked_paulis: Vec<usize>,
    /// Probability of this alternative conditioned on the mechanism firing (`1/k`).
    pub conditional_probability: f64,
    /// Marginal probability of this specific alternative at this location: `p_i / k_i`.
    ///
    /// This is NOT "probability of this fault and no others." A full-circuit
    /// configuration probability requires multiplying by `(1 - p_j)` for all
    /// other locations `j`.
    pub absolute_probability: f64,
}

/// A physical fault location in the circuit.
#[derive(Clone, Debug)]
pub struct FaultLocation {
    /// Tick index in the `TickCircuit`.
    pub tick: usize,
    /// Gate index within the tick.
    pub gate_index: usize,
    /// Gate type at this location.
    pub gate_type: GateType,
    /// Qubits involved.
    pub qubits: Vec<usize>,
    /// Which noise channel this location belongs to.
    pub channel: FaultChannel,
    /// Total probability that this mechanism fires: `p_i`.
    pub channel_probability: f64,
    /// Probability that no fault occurs at this location: `1 - p_i`.
    pub no_fault_probability: f64,
    /// Number of fault alternatives at this location: `k_i`.
    pub num_alternatives: usize,
    /// All fault alternatives at this location.
    pub faults: Vec<FaultAlternative>,
}

/// Complete fault catalog for a circuit + noise model.
///
/// Each location is an independent physical fault mechanism.
/// Each alternative within a location is one possible Pauli error
/// (for depolarizing) or outcome flip (for measurement/prep).
///
/// Probability model (independent mechanisms):
///
/// For location `i` with `k_i` alternatives:
/// - `channel_probability` = `p_i` (total probability mechanism fires)
/// - `no_fault_probability` = `1 - p_i`
/// - `conditional_probability` = `1/k_i` (uniform alternative choice)
/// - `absolute_probability` = `p_i / k_i` (marginal alternative probability)
///
/// Full-circuit configuration probability for "alternative j at location i,
/// no fault at all other locations":
/// ```text
/// P = (p_i / k_i) * product_{m != i} (1 - p_m)
/// ```
#[derive(Clone, Debug)]
pub struct FaultCatalog {
    pub locations: Vec<FaultLocation>,
}

/// One yielded configuration from `fault_configurations(k)`.
#[derive(Clone, Debug)]
pub struct FaultConfiguration {
    /// Indices into `catalog.locations` for the k selected locations.
    pub location_indices: Vec<usize>,
    /// Alternative index chosen within each selected location.
    pub alternative_indices: Vec<usize>,
    /// Combined measurement indices (XOR parity across selected alternatives).
    pub affected_measurements: Vec<usize>,
    /// Combined detector indices (XOR parity).
    pub affected_detectors: Vec<usize>,
    /// Combined observable indices (XOR parity).
    pub affected_observables: Vec<usize>,
    /// Combined tracked-Pauli indices (XOR parity).
    pub affected_tracked_paulis: Vec<usize>,
    /// Product of selected alternatives' `absolute_probability`.
    pub selected_probability: f64,
    /// `selected_probability * product(unselected no_fault_probability)`.
    pub configuration_probability: f64,
}

impl FaultCatalog {
    /// Build a structural fault catalog from a circuit.
    ///
    /// The returned catalog includes all structurally supported noisy locations,
    /// independent of any concrete noise point. All channel and alternative
    /// probabilities are initialized to zero except `no_fault_probability`, which
    /// is initialized to one.
    ///
    /// # Errors
    ///
    /// Returns [`UnsupportedGateError`] when the circuit contains a gate outside
    /// the supported Clifford/prep/measurement/metadata set.
    pub fn from_circuit(tc: &TickCircuit) -> Result<Self, UnsupportedGateError> {
        build_structural_fault_catalog(tc)
    }

    /// Recompute noise-dependent probability fields for this catalog.
    ///
    /// This updates only `channel_probability`, `no_fault_probability`, and
    /// `absolute_probability`. Structural fields such as `num_alternatives`,
    /// `conditional_probability`, Pauli labels, and effect lists are unchanged.
    ///
    /// # Panics
    ///
    /// Panics if a malformed catalog contains more than `u32::MAX` alternatives
    /// at one location. Catalogs produced by [`FaultCatalog::from_circuit`] have
    /// at most 15 alternatives per location.
    pub fn with_noise(&mut self, noise: &StochasticNoiseParams) -> &mut Self {
        for loc in &mut self.locations {
            let p = match loc.channel {
                FaultChannel::P1 => noise.p1,
                FaultChannel::P2 => noise.p2,
                FaultChannel::PMeas => noise.p_meas,
                FaultChannel::PPrep => noise.p_prep,
            };
            let k = loc.num_alternatives;
            debug_assert!(k > 0, "fault location has no alternatives");
            debug_assert_eq!(k, loc.faults.len(), "num_alternatives out of sync");
            let k_f64 = f64::from(u32::try_from(k).expect("fault alternative count exceeds u32"));

            loc.channel_probability = p;
            loc.no_fault_probability = 1.0 - p;
            for alt in &mut loc.faults {
                alt.absolute_probability = p / k_f64;
            }
        }
        self
    }

    /// Clone this catalog and apply a concrete noise point to the clone.
    #[must_use]
    pub fn parameterized(&self, noise: &StochasticNoiseParams) -> Self {
        let mut copy = self.clone();
        copy.with_noise(noise);
        copy
    }

    /// Convert this catalog into raw-measurement sampling mechanisms.
    ///
    /// This is a materialization step for raw measurement sampling only. It
    /// drops zero-probability locations and locations where every alternative has
    /// empty `affected_measurements`, while preserving empty alternatives inside
    /// any kept mechanism to maintain the correct uniform denominator.
    #[must_use]
    pub fn to_mechanisms(&self) -> Vec<FaultMechanism> {
        self.locations
            .iter()
            .filter(|loc| loc.channel_probability > 0.0)
            .filter(|loc| {
                loc.faults
                    .iter()
                    .any(|alt| !alt.affected_measurements.is_empty())
            })
            .map(|loc| FaultMechanism {
                probability: loc.channel_probability,
                alternatives: loc
                    .faults
                    .iter()
                    .map(|alt| alt.affected_measurements.clone())
                    .collect(),
            })
            .collect()
    }

    /// Lazily iterate all k-fault configurations.
    ///
    /// Each yielded `FaultConfiguration` represents exactly k distinct locations
    /// firing, with one alternative chosen per location. Effects are combined by
    /// XOR parity. Probabilities follow the independent-mechanism model.
    ///
    /// Zero-probability alternatives are skipped. A structural location with
    /// `channel_probability == 0` remains in [`FaultCatalog::locations`] but is
    /// not yielded as a selected fault configuration.
    ///
    /// For k=0: yields one no-fault event.
    #[must_use]
    pub fn fault_configurations(&self, k: usize) -> FaultConfigurationIter<'_> {
        FaultConfigurationIter::new(self, k)
    }
}

/// Internal cursor for k-fault configuration iteration.
///
/// Holds the combination/alternative state machine. Shared by both
/// `FaultConfigurationIter` (borrowed) and `OwnedFaultConfigIter` (owned).
/// Combinations range over nonzero-probability fault alternatives only; the
/// full structural catalog remains available through `FaultCatalog::locations`.
struct FaultConfigCursor {
    k: usize,
    location_indices: Vec<usize>,
    alternative_indices: Vec<Vec<usize>>,
    combo: Vec<usize>,
    alt_indices: Vec<usize>,
    alt_counts: Vec<usize>,
    started: bool,
    done: bool,
}

impl FaultConfigCursor {
    fn new(catalog: &FaultCatalog, k: usize) -> Self {
        let mut location_indices = Vec::new();
        let mut alternative_indices = Vec::new();
        for (loc_idx, loc) in catalog.locations.iter().enumerate() {
            let alts: Vec<usize> = loc
                .faults
                .iter()
                .enumerate()
                .filter_map(|(alt_idx, alt)| (alt.absolute_probability > 0.0).then_some(alt_idx))
                .collect();
            if !alts.is_empty() {
                location_indices.push(loc_idx);
                alternative_indices.push(alts);
            }
        }

        let num_active_locations = location_indices.len();
        if k == 0 || k > num_active_locations {
            return Self {
                k,
                location_indices,
                alternative_indices,
                combo: Vec::new(),
                alt_indices: Vec::new(),
                alt_counts: Vec::new(),
                started: false,
                done: k > num_active_locations && k > 0,
            };
        }
        let combo: Vec<usize> = (0..k).collect();
        let alt_counts: Vec<usize> = combo
            .iter()
            .map(|&i| alternative_indices[i].len())
            .collect();
        let alt_indices = vec![0usize; k];
        Self {
            k,
            location_indices,
            alternative_indices,
            combo,
            alt_indices,
            alt_counts,
            started: false,
            done: false,
        }
    }

    /// Advance to the next state. Returns true if a new valid state exists.
    fn advance(&mut self) -> bool {
        // Try advancing alternatives (mixed-radix counter)
        for i in (0..self.k).rev() {
            self.alt_indices[i] += 1;
            if self.alt_indices[i] < self.alt_counts[i] {
                return true;
            }
            self.alt_indices[i] = 0;
        }
        // Try advancing combination
        let mut i = self.k;
        while i > 0 {
            i -= 1;
            self.combo[i] += 1;
            if self.combo[i] <= self.location_indices.len() - self.k + i {
                for j in (i + 1)..self.k {
                    self.combo[j] = self.combo[j - 1] + 1;
                }
                for j in 0..self.k {
                    self.alt_counts[j] = self.alternative_indices[self.combo[j]].len();
                    self.alt_indices[j] = 0;
                }
                return true;
            }
        }
        false
    }

    /// Build a `FaultConfiguration` from the current cursor state + catalog data.
    fn build(&self, catalog: &FaultCatalog) -> FaultConfiguration {
        if self.k == 0 {
            let no_fault_prob: f64 = catalog
                .locations
                .iter()
                .map(|l| l.no_fault_probability)
                .product();
            return FaultConfiguration {
                location_indices: Vec::new(),
                alternative_indices: Vec::new(),
                affected_measurements: Vec::new(),
                affected_detectors: Vec::new(),
                affected_observables: Vec::new(),
                affected_tracked_paulis: Vec::new(),
                selected_probability: 1.0,
                configuration_probability: no_fault_prob,
            };
        }

        let mut meas_set = std::collections::BTreeSet::new();
        let mut det_set = std::collections::BTreeSet::new();
        let mut obs_set = std::collections::BTreeSet::new();
        let mut tracked_pauli_set = std::collections::BTreeSet::new();
        let mut selected_prob = 1.0;

        for i in 0..self.k {
            let location_index = self.location_indices[self.combo[i]];
            let alternative_index = self.alternative_indices[self.combo[i]][self.alt_indices[i]];
            let loc = &catalog.locations[location_index];
            let alt = &loc.faults[alternative_index];
            selected_prob *= alt.absolute_probability;
            for &m in &alt.affected_measurements {
                if !meas_set.remove(&m) {
                    meas_set.insert(m);
                }
            }
            for &d in &alt.affected_detectors {
                if !det_set.remove(&d) {
                    det_set.insert(d);
                }
            }
            for &o in &alt.affected_observables {
                if !obs_set.remove(&o) {
                    obs_set.insert(o);
                }
            }
            for &op in &alt.affected_tracked_paulis {
                if !tracked_pauli_set.remove(&op) {
                    tracked_pauli_set.insert(op);
                }
            }
        }

        let selected_set: std::collections::BTreeSet<usize> = self
            .combo
            .iter()
            .map(|&i| self.location_indices[i])
            .collect();
        let unselected_no_fault: f64 = catalog
            .locations
            .iter()
            .enumerate()
            .filter(|(i, _)| !selected_set.contains(i))
            .map(|(_, loc)| loc.no_fault_probability)
            .product();

        FaultConfiguration {
            location_indices: self
                .combo
                .iter()
                .map(|&i| self.location_indices[i])
                .collect(),
            alternative_indices: self
                .combo
                .iter()
                .zip(self.alt_indices.iter())
                .map(|(&loc_pos, &alt_pos)| self.alternative_indices[loc_pos][alt_pos])
                .collect(),
            affected_measurements: meas_set.into_iter().collect(),
            affected_detectors: det_set.into_iter().collect(),
            affected_observables: obs_set.into_iter().collect(),
            affected_tracked_paulis: tracked_pauli_set.into_iter().collect(),
            selected_probability: selected_prob,
            configuration_probability: selected_prob * unselected_no_fault,
        }
    }

    /// Drive the iterator: yield next configuration or None.
    fn next_config(&mut self, catalog: &FaultCatalog) -> Option<FaultConfiguration> {
        if self.done {
            return None;
        }
        if self.k == 0 {
            self.done = true;
            return Some(self.build(catalog));
        }
        if !self.started {
            self.started = true;
            return Some(self.build(catalog));
        }
        if self.advance() {
            Some(self.build(catalog))
        } else {
            self.done = true;
            None
        }
    }
}

/// Lazy iterator over k-fault configurations (borrows catalog).
pub struct FaultConfigurationIter<'a> {
    catalog: &'a FaultCatalog,
    cursor: FaultConfigCursor,
}

impl<'a> FaultConfigurationIter<'a> {
    fn new(catalog: &'a FaultCatalog, k: usize) -> Self {
        let cursor = FaultConfigCursor::new(catalog, k);
        Self { catalog, cursor }
    }
}

impl Iterator for FaultConfigurationIter<'_> {
    type Item = FaultConfiguration;
    fn next(&mut self) -> Option<Self::Item> {
        self.cursor.next_config(self.catalog)
    }
}

/// Owned k-fault configuration iterator (no lifetime borrows).
/// Suitable for FFI / `PyO3` where lifetimes are not expressible.
pub struct OwnedFaultConfigIter {
    catalog: FaultCatalog,
    cursor: FaultConfigCursor,
}

impl OwnedFaultConfigIter {
    /// Create from an owned catalog clone.
    #[must_use]
    pub fn new(catalog: FaultCatalog, k: usize) -> Self {
        let cursor = FaultConfigCursor::new(&catalog, k);
        Self { catalog, cursor }
    }
}

impl Iterator for OwnedFaultConfigIter {
    type Item = FaultConfiguration;
    fn next(&mut self) -> Option<Self::Item> {
        self.cursor.next_config(&self.catalog)
    }
}

/// Build a fault catalog from a `TickCircuit` and noise parameters.
///
/// Returns per-location, per-alternative fault data including Pauli labels,
/// affected detectors, observables, tracked Paulis, and probability fields.
///
/// Reads detector/observable metadata and tracked-Pauli annotations
/// from the circuit when present.
///
/// # Errors
///
/// Returns [`UnsupportedGateError`] when the circuit contains a gate outside
/// the supported Clifford/prep/measurement/metadata set.
pub fn build_fault_catalog(
    tc: &TickCircuit,
    noise: &StochasticNoiseParams,
) -> Result<FaultCatalog, UnsupportedGateError> {
    let mut catalog = FaultCatalog::from_circuit(tc)?;
    catalog.with_noise(noise);
    Ok(catalog)
}

fn build_structural_fault_catalog(tc: &TickCircuit) -> Result<FaultCatalog, UnsupportedGateError> {
    validate_tick_circuit(tc)?;
    let (gates, meas_positions) = flatten_tick_circuit(tc);

    // Parse detector/DEM-output records for measurement→detector/op mapping
    let det_records = parse_detector_records(tc);
    let obs_records = parse_observable_records(tc);
    let tracked_pauli_annotations = parse_tracked_pauli_annotations(tc);
    let num_meas = tc
        .get_meta("num_measurements")
        .and_then(|a| {
            if let pecos_quantum::Attribute::String(s) = a {
                s.parse::<usize>().ok()
            } else {
                None
            }
        })
        .unwrap_or(meas_positions.len());
    let record_effect_index = RecordEffectIndex::new(&det_records, &obs_records, num_meas);

    let mut locations = Vec::new();

    let pauli_types = [PauliType::X, PauliType::Y, PauliType::Z];
    let mut effect_cache = PropagatedEffectCache::default();

    for (loc_idx, loc) in gates.iter().enumerate() {
        let tick_idx = loc.tick;
        let gate_idx = loc.gate_index;
        let gate_type = loc.gate_type;
        let qubits = &loc.qubits;

        match gate_type {
            gate_type if is_standard_1q_clifford_gate(gate_type) && !loc.qubits.is_empty() => {
                let q = loc.qubits[0];
                let num_alts = 3;
                let conditional_probability = 1.0 / 3.0;
                let mut faults = Vec::with_capacity(num_alts);
                for &pt in &pauli_types {
                    let effect = effect_cache.single(
                        pt,
                        q,
                        loc_idx + 1,
                        &gates,
                        &meas_positions,
                        &tracked_pauli_annotations,
                    );
                    let pauli = pauli_type_to_string(pt, q);
                    let (affected, dets, obs, tracked) =
                        catalog_effect_parts(effect, &record_effect_index);
                    faults.push(FaultAlternative {
                        kind: FaultKind::Pauli,
                        pauli: Some(pauli),
                        affected_measurements: affected,
                        affected_detectors: dets,
                        affected_observables: obs,
                        affected_tracked_paulis: tracked,
                        conditional_probability,
                        absolute_probability: 0.0,
                    });
                }
                let num_alts = faults.len();
                locations.push(FaultLocation {
                    tick: tick_idx,
                    gate_index: gate_idx,
                    gate_type,
                    qubits: qubits.clone(),
                    channel: FaultChannel::P1,
                    channel_probability: 0.0,
                    no_fault_probability: 1.0,
                    num_alternatives: num_alts,
                    faults,
                });
            }

            gate_type if is_standard_2q_clifford_gate(gate_type) && loc.qubits.len() >= 2 => {
                let (q1, q2) = (loc.qubits[0], loc.qubits[1]);
                let num_alts = 15;
                let conditional_probability = 1.0 / 15.0;
                let mut faults = Vec::with_capacity(num_alts);

                // 9 two-qubit pairs
                for &p1 in &pauli_types {
                    for &p2 in &pauli_types {
                        let left = effect_cache.single(
                            p1,
                            q1,
                            loc_idx + 1,
                            &gates,
                            &meas_positions,
                            &tracked_pauli_annotations,
                        );
                        let right = effect_cache.single(
                            p2,
                            q2,
                            loc_idx + 1,
                            &gates,
                            &meas_positions,
                            &tracked_pauli_annotations,
                        );
                        let effect = xor_fault_effects(&left, &right);
                        let pauli = pauli_pair_to_string(p1, q1, p2, q2);
                        let (affected, dets, obs, tracked) =
                            catalog_effect_parts(effect, &record_effect_index);
                        faults.push(FaultAlternative {
                            kind: FaultKind::Pauli,
                            pauli: Some(pauli),
                            affected_measurements: affected,
                            affected_detectors: dets,
                            affected_observables: obs,
                            affected_tracked_paulis: tracked,
                            conditional_probability,
                            absolute_probability: 0.0,
                        });
                    }
                }
                // 6 single-qubit (PI and IP)
                for &p in &pauli_types {
                    let effect = effect_cache.single(
                        p,
                        q1,
                        loc_idx + 1,
                        &gates,
                        &meas_positions,
                        &tracked_pauli_annotations,
                    );
                    let pauli = pauli_type_to_string(p, q1);
                    let (affected, dets, obs, tracked) =
                        catalog_effect_parts(effect, &record_effect_index);
                    faults.push(FaultAlternative {
                        kind: FaultKind::Pauli,
                        pauli: Some(pauli),
                        affected_measurements: affected,
                        affected_detectors: dets,
                        affected_observables: obs,
                        affected_tracked_paulis: tracked,
                        conditional_probability,
                        absolute_probability: 0.0,
                    });

                    let effect = effect_cache.single(
                        p,
                        q2,
                        loc_idx + 1,
                        &gates,
                        &meas_positions,
                        &tracked_pauli_annotations,
                    );
                    let pauli = pauli_type_to_string(p, q2);
                    let (affected, dets, obs, tracked) =
                        catalog_effect_parts(effect, &record_effect_index);
                    faults.push(FaultAlternative {
                        kind: FaultKind::Pauli,
                        pauli: Some(pauli),
                        affected_measurements: affected,
                        affected_detectors: dets,
                        affected_observables: obs,
                        affected_tracked_paulis: tracked,
                        conditional_probability,
                        absolute_probability: 0.0,
                    });
                }
                let n_alts = faults.len();
                locations.push(FaultLocation {
                    tick: tick_idx,
                    gate_index: gate_idx,
                    gate_type,
                    qubits: qubits.clone(),
                    channel: FaultChannel::P2,
                    channel_probability: 0.0,
                    no_fault_probability: 1.0,
                    num_alternatives: n_alts,
                    faults,
                });
            }

            GateType::PZ | GateType::QAlloc if !loc.qubits.is_empty() => {
                let q = loc.qubits[0];
                let effect = effect_cache.single(
                    PauliType::X,
                    q,
                    loc_idx + 1,
                    &gates,
                    &meas_positions,
                    &tracked_pauli_annotations,
                );
                let (affected, dets, obs, tracked) =
                    catalog_effect_parts(effect, &record_effect_index);
                locations.push(FaultLocation {
                    tick: tick_idx,
                    gate_index: gate_idx,
                    gate_type,
                    qubits: qubits.clone(),
                    channel: FaultChannel::PPrep,
                    channel_probability: 0.0,
                    no_fault_probability: 1.0,
                    num_alternatives: 1,
                    faults: vec![FaultAlternative {
                        kind: FaultKind::PrepFlip,
                        pauli: None,
                        affected_measurements: affected,
                        affected_detectors: dets,
                        affected_observables: obs,
                        affected_tracked_paulis: tracked,
                        conditional_probability: 1.0,
                        absolute_probability: 0.0,
                    }],
                });
            }

            GateType::MZ | GateType::MeasureFree | GateType::MeasureLeaked => {
                if let Some(&meas_idx) = meas_positions.get(&loc_idx) {
                    let affected = vec![meas_idx];
                    let dets = record_effect_index.detectors_for_measurements(&affected);
                    let obs = record_effect_index.observables_for_measurements(&affected);
                    locations.push(FaultLocation {
                        tick: tick_idx,
                        gate_index: gate_idx,
                        gate_type,
                        qubits: qubits.clone(),
                        channel: FaultChannel::PMeas,
                        channel_probability: 0.0,
                        no_fault_probability: 1.0,
                        num_alternatives: 1,
                        faults: vec![FaultAlternative {
                            kind: FaultKind::MeasurementFlip,
                            pauli: None,
                            affected_measurements: affected,
                            affected_detectors: dets,
                            affected_observables: obs,
                            affected_tracked_paulis: Vec::new(),
                            conditional_probability: 1.0,
                            absolute_probability: 0.0,
                        }],
                    });
                }
            }

            _ => {}
        }
    }

    Ok(FaultCatalog { locations })
}

// ---- Helpers for fault catalog ----

fn pauli_type_to_pauli(pt: PauliType) -> Pauli {
    match pt {
        PauliType::X => Pauli::X,
        PauliType::Y => Pauli::Y,
        PauliType::Z => Pauli::Z,
    }
}

fn pauli_type_to_string(pt: PauliType, qubit: usize) -> PauliString {
    PauliString::with_phase_and_paulis(
        pecos_core::QuarterPhase::PlusOne,
        vec![(pauli_type_to_pauli(pt), QubitId(qubit))],
    )
}

fn pauli_pair_to_string(p1: PauliType, q1: usize, p2: PauliType, q2: usize) -> PauliString {
    PauliString::with_phase_and_paulis(
        pecos_core::QuarterPhase::PlusOne,
        vec![
            (pauli_type_to_pauli(p1), QubitId(q1)),
            (pauli_type_to_pauli(p2), QubitId(q2)),
        ],
    )
}

fn parse_records_from_meta(tc: &TickCircuit, key: &str) -> Vec<Vec<i32>> {
    let Some(pecos_quantum::Attribute::String(json)) = tc.get_meta(key) else {
        return Vec::new();
    };
    parse_records_array_list(json)
}

fn parse_detector_records(tc: &TickCircuit) -> Vec<Vec<i32>> {
    parse_records_from_meta(tc, "detectors")
}

fn parse_observable_records(tc: &TickCircuit) -> Vec<Vec<i32>> {
    parse_records_from_meta(tc, "observables")
}

fn parse_tracked_pauli_annotations(tc: &TickCircuit) -> Vec<PauliString> {
    tc.annotations()
        .iter()
        .filter(|ann| matches!(ann.kind, AnnotationKind::TrackedPauli))
        .map(|ann| {
            let mut pauli = ann.pauli.clone();
            pauli.set_phase(pecos_core::QuarterPhase::PlusOne);
            pauli
        })
        .collect()
}

fn tracked_paulis_flipped_by(
    prop: &BitmaskPauliProp,
    tracked_paulis: &[PauliString],
) -> Vec<usize> {
    tracked_paulis
        .iter()
        .enumerate()
        .filter_map(|(idx, tracked_pauli)| {
            let mut parity = false;
            for &(pauli, qubit) in tracked_pauli.paulis() {
                let q = qubit.index();
                match pauli {
                    Pauli::X => parity ^= prop.contains_z(q),
                    Pauli::Y => parity ^= prop.contains_x(q) ^ prop.contains_z(q),
                    Pauli::Z => parity ^= prop.contains_x(q),
                    Pauli::I => {}
                }
            }
            parity.then_some(idx)
        })
        .collect()
}

/// Simple parser for `[{"records": [...]}, ...]` JSON without `serde_json`.
fn parse_records_array_list(json: &str) -> Vec<Vec<i32>> {
    let json = json.trim();
    if json.is_empty() || json == "[]" {
        return Vec::new();
    }
    let mut results = Vec::new();
    // Find each "records": [...] within the JSON
    let mut search_from = 0;
    while let Some(pos) = json[search_from..].find("\"records\"") {
        let pos = search_from + pos;
        let rest = &json[pos..];
        if let Some(arr_start) = rest.find('[') {
            if let Some(arr_end) = rest[arr_start..].find(']') {
                let arr_str = &rest[arr_start + 1..arr_start + arr_end];
                let nums: Vec<i32> = arr_str
                    .split(',')
                    .filter_map(|s| s.trim().parse().ok())
                    .collect();
                results.push(nums);
                search_from = pos + arr_start + arr_end + 1;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    results
}

fn record_absolute_index(num_meas: usize, rec: i32) -> Option<usize> {
    let base = i64::try_from(num_meas).ok()?;
    let abs_idx = base.checked_add(i64::from(rec))?;
    usize::try_from(abs_idx).ok()
}

struct RecordEffectIndex {
    detectors_by_measurement: Vec<Vec<usize>>,
    observables_by_measurement: Vec<Vec<usize>>,
}

impl RecordEffectIndex {
    fn new(det_records: &[Vec<i32>], obs_records: &[Vec<i32>], num_meas: usize) -> Self {
        Self {
            detectors_by_measurement: records_by_measurement(det_records, num_meas),
            observables_by_measurement: records_by_measurement(obs_records, num_meas),
        }
    }

    /// Map measurement effects to detector effects via record XOR.
    fn detectors_for_measurements(&self, affected_meas: &[usize]) -> Vec<usize> {
        measurements_to_record_effects(affected_meas, &self.detectors_by_measurement)
    }

    /// Map measurement effects to observable effects via record XOR.
    fn observables_for_measurements(&self, affected_meas: &[usize]) -> Vec<usize> {
        measurements_to_record_effects(affected_meas, &self.observables_by_measurement)
    }
}

fn catalog_effect_parts(
    effect: PropagatedFaultEffect,
    record_effect_index: &RecordEffectIndex,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let affected: Vec<usize> = effect.affected_measurements.into_iter().collect();
    let dets = record_effect_index.detectors_for_measurements(&affected);
    let obs = record_effect_index.observables_for_measurements(&affected);
    (affected, dets, obs, effect.affected_tracked_paulis)
}

fn records_by_measurement(records_by_output: &[Vec<i32>], num_meas: usize) -> Vec<Vec<usize>> {
    let mut by_measurement = vec![Vec::new(); num_meas];
    for (output_idx, records) in records_by_output.iter().enumerate() {
        for &rec in records {
            if let Some(meas_idx) = record_absolute_index(num_meas, rec)
                && meas_idx < num_meas
            {
                by_measurement[meas_idx].push(output_idx);
            }
        }
    }
    by_measurement
}

fn measurements_to_record_effects(
    affected_meas: &[usize],
    outputs_by_measurement: &[Vec<usize>],
) -> Vec<usize> {
    let mut fired = Vec::new();
    for &meas_idx in affected_meas {
        if let Some(outputs) = outputs_by_measurement.get(meas_idx) {
            for &output_idx in outputs {
                toggle_sorted(&mut fired, output_idx);
            }
        }
    }
    fired
}

fn toggle_sorted(values: &mut Vec<usize>, value: usize) {
    match values.binary_search(&value) {
        Ok(pos) => {
            values.remove(pos);
        }
        Err(pos) => {
            values.insert(pos, value);
        }
    }
}

// ============================================================================
// Shared symbolic simulation helper
// ============================================================================

/// Run `SymbolicSparseStab` through a `TickCircuit` with proper PZ (reset)
/// semantics, returning the `MeasurementHistory` with correct cross-reset
/// correlations.
///
/// Iterates tick-by-tick to match the `TickCircuit`'s measurement numbering
/// (which detector/DEM-output record indices reference).
///
/// Errors on unsupported gates with tick/gate/qubit context (same gate set
/// as [`build_fault_table`]).
///
/// # Errors
///
/// Returns [`UnsupportedGateError`] when the circuit contains a gate outside
/// the supported Clifford/prep/measurement/metadata set.
pub fn symbolic_measurement_history(
    tc: &TickCircuit,
) -> Result<MeasurementHistory, UnsupportedGateError> {
    use pecos_simulators::SymbolicSparseStab;

    let num_qubits = tc
        .iter_gate_batches()
        .flat_map(|g| g.as_gate().qubits.iter())
        .map(|q| q.index() + 1)
        .max()
        .unwrap_or(0);

    let mut sim = SymbolicSparseStab::new(num_qubits);

    for (tick_idx, tick) in tc.iter_ticks() {
        for gate in tick.iter_gate_batches() {
            let gate_idx = gate.batch_index();
            let qs: Vec<usize> = gate.qubits.iter().map(pecos_core::QubitId::index).collect();

            match gate.gate_type {
                GateType::PZ | GateType::QAlloc => {
                    for &q in &qs {
                        sim.pz(q);
                    }
                }
                GateType::H => {
                    sim.h(&qs);
                }
                GateType::X => {
                    sim.x(&qs);
                }
                GateType::Y => {
                    sim.y(&qs);
                }
                GateType::Z => {
                    sim.z(&qs);
                }
                GateType::SZ => {
                    sim.sz(&qs);
                }
                GateType::SZdg => {
                    sim.szdg(&qs);
                }
                GateType::SX => {
                    sim.sx(&qs);
                }
                GateType::SXdg => {
                    sim.sxdg(&qs);
                }
                GateType::SY => {
                    sim.sy(&qs);
                }
                GateType::SYdg => {
                    sim.sydg(&qs);
                }
                GateType::F => {
                    sim.sx(&qs);
                    sim.sz(&qs);
                }
                GateType::Fdg => {
                    sim.szdg(&qs);
                    sim.sxdg(&qs);
                }
                GateType::CX => {
                    let pairs = symbolic_pairs(&qs);
                    sim.cx(&pairs);
                }
                GateType::CY => {
                    sim.cy(&symbolic_pairs(&qs));
                }
                GateType::CZ => {
                    sim.cz(&symbolic_pairs(&qs));
                }
                GateType::SXX => {
                    sim.sxx(&symbolic_pairs(&qs));
                }
                GateType::SXXdg => {
                    sim.sxxdg(&symbolic_pairs(&qs));
                }
                GateType::SYY => {
                    sim.syy(&symbolic_pairs(&qs));
                }
                GateType::SYYdg => {
                    sim.syydg(&symbolic_pairs(&qs));
                }
                GateType::SZZ => {
                    sim.szz(&symbolic_pairs(&qs));
                }
                GateType::SZZdg => {
                    sim.szzdg(&symbolic_pairs(&qs));
                }
                GateType::SWAP => {
                    sim.swap(&symbolic_pairs(&qs));
                }
                GateType::MZ | GateType::MeasureFree | GateType::MeasureLeaked => {
                    sim.mz(&qs);
                }
                GateType::I
                | GateType::Idle
                | GateType::QFree
                | GateType::MeasCrosstalkGlobalPayload
                | GateType::MeasCrosstalkLocalPayload
                | GateType::TrackedPauliMeta => {}
                other => {
                    return Err(UnsupportedGateError {
                        gate_type: other,
                        tick: tick_idx,
                        gate_in_tick: gate_idx,
                        qubits: qs,
                    });
                }
            }
        }
    }

    Ok(sim.measurement_history().clone())
}

fn symbolic_pairs(qs: &[usize]) -> Vec<(usize, usize)> {
    qs.chunks(2)
        .filter(|c| c.len() == 2)
        .map(|c| (c[0], c[1]))
        .collect()
}

// ============================================================================
// Raw Measurement Plan: geometric/O(fired) columnar sampling
// ============================================================================

/// Zero out bits beyond `shots` in the final word of each column.
fn mask_partial_final_word(columns: &mut [Vec<u64>], shots: usize) {
    let remainder = shots % 64;
    if remainder == 0 {
        return;
    }
    let mask = (1u64 << remainder) - 1;
    for col in columns.iter_mut() {
        if let Some(last) = col.last_mut() {
            *last &= mask;
        }
    }
}

/// Columnar raw-measurement result with r-source access.
///
/// The measurement columns are the final output (base XOR faults).
/// The `r_columns` field holds the latent random source columns that feed
/// into the ideal measurement dependency graph.
pub struct RawSampleResult {
    /// Final measurement columns: `columns[meas_idx][word_idx]`, bit i = shot word*64+i.
    /// Bits beyond `shots` in the final word are always zero.
    pub columns: Vec<Vec<u64>>,
    /// Latent r-source columns (one per Random measurement kind).
    /// Bits beyond `shots` in the final word are always zero.
    pub r_columns: Vec<Vec<u64>>,
    /// Measurement index that introduced each r-source.
    /// `r_source_measurements[k]` is the measurement index for `r_columns[k]`.
    pub r_source_measurements: Vec<usize>,
    pub shots: usize,
}

/// A compiled plan for sampling raw measurements from a stochastic circuit.
///
/// Combines:
/// - **r-sources** (p=0.5): non-deterministic measurement variables from the
///   ideal dependency graph. These fan out through Copy/Computed relationships.
/// - **Physical mechanisms**: depolarizing gate faults, prep faults,
///   measurement flips. These do NOT fan out through ideal dependencies.
///
/// Physical mechanisms are sampled using geometric skip (O(fired events) per
/// mechanism), matching the DEM sampler's performance characteristics.
pub struct RawMeasurementPlan {
    pub num_measurements: usize,
    kinds: Vec<MeasurementKind>,
    pub mechanisms: Vec<FaultMechanism>,
    /// Precomputed 1/ln(1-p) for geometric skip sampling, one per mechanism.
    inv_log_1_minus_p: Vec<f64>,
}

impl RawMeasurementPlan {
    /// Build a plan from a measurement history and fault mechanisms.
    #[must_use]
    pub fn new(history: &MeasurementHistory, mechanisms: Vec<FaultMechanism>) -> Self {
        let kinds = MeasurementKind::from_history(history);
        let inv_log_1_minus_p = mechanisms
            .iter()
            .map(|m| {
                let log_1_minus_p = (1.0 - m.probability).ln();
                if log_1_minus_p.abs() < f64::EPSILON {
                    0.0
                } else {
                    1.0 / log_1_minus_p
                }
            })
            .collect();
        Self {
            num_measurements: kinds.len(),
            kinds,
            mechanisms,
            inv_log_1_minus_p,
        }
    }

    /// Sample raw measurements using geometric skip for physical faults.
    ///
    /// Returns a `SampleResult` for compatibility with existing code.
    /// For r-event access, use [`sample_raw`].
    #[must_use]
    pub fn sample(&self, shots: usize, seed: u64) -> SampleResult {
        let raw = self.sample_raw(shots, seed);
        SampleResult::new(raw.columns, shots)
    }

    /// Sample raw measurements with r-source column access.
    ///
    /// Physical mechanisms use geometric skip: O(p * shots) RNG calls per
    /// mechanism, not O(shots). For typical QEC noise (p ~ 0.005, 20k shots),
    /// this is ~100 firings per mechanism vs 20000 iterations.
    #[must_use]
    pub fn sample_raw(&self, shots: usize, seed: u64) -> RawSampleResult {
        if shots == 0 {
            let r_source_measurements = self.r_source_indices();
            return RawSampleResult {
                columns: vec![Vec::new(); self.num_measurements],
                r_columns: vec![Vec::new(); r_source_measurements.len()],
                r_source_measurements,
                shots: 0,
            };
        }

        let num_words = shots.div_ceil(64);

        // 1. Sample base values (r-sources + constants) and capture r columns
        let mut rng_base = PecosRng::seed_from_u64(seed);
        let (mut columns, mut r_columns) = self.sample_base(num_words, &mut rng_base);

        // 2. Overlay physical faults using geometric skip
        if !self.mechanisms.is_empty() {
            let mut rng_fault = PecosRng::seed_from_u64(seed.wrapping_add(1));
            self.overlay_faults_geometric(shots, &mut columns, &mut rng_fault);
        }

        // 3. Mask partial final word so bits beyond `shots` are always zero
        mask_partial_final_word(&mut columns, shots);
        mask_partial_final_word(&mut r_columns, shots);

        RawSampleResult {
            columns,
            r_columns,
            r_source_measurements: self.r_source_indices(),
            shots,
        }
    }

    /// Returns the measurement indices that correspond to r-sources (Random kinds).
    fn r_source_indices(&self) -> Vec<usize> {
        self.kinds
            .iter()
            .enumerate()
            .filter_map(|(i, k)| {
                if matches!(k, MeasurementKind::Random) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Sample base measurement values from r-sources and constants.
    /// Returns (`measurement_columns`, `r_source_columns`).
    fn sample_base(&self, num_words: usize, rng: &mut PecosRng) -> (Vec<Vec<u64>>, Vec<Vec<u64>>) {
        let mut columns: Vec<Vec<u64>> = Vec::with_capacity(self.num_measurements);
        let mut r_columns: Vec<Vec<u64>> = Vec::new();

        for kind in &self.kinds {
            match kind {
                MeasurementKind::Fixed(value) => {
                    let fill = if *value { !0u64 } else { 0u64 };
                    columns.push(vec![fill; num_words]);
                }
                MeasurementKind::Random => {
                    let mut col = vec![0u64; num_words];
                    for word in &mut col {
                        *word = rng.next_u64();
                    }
                    r_columns.push(col.clone());
                    columns.push(col);
                }
                MeasurementKind::Copy(src) => {
                    columns.push(columns[*src].clone());
                }
                MeasurementKind::CopyFlipped(src) => {
                    let flipped: Vec<u64> = columns[*src].iter().map(|w| !w).collect();
                    columns.push(flipped);
                }
                MeasurementKind::Computed { deps, flip } => {
                    let init = if *flip { !0u64 } else { 0u64 };
                    let mut col = vec![init; num_words];
                    for &dep in deps {
                        for (w, &d) in col.iter_mut().zip(columns[dep].iter()) {
                            *w ^= d;
                        }
                    }
                    columns.push(col);
                }
            }
        }

        (columns, r_columns)
    }

    /// Overlay physical faults using geometric skip sampling.
    ///
    /// For each mechanism with probability p:
    /// - Precomputed `inv_log = 1/ln(1-p)`
    /// - Sample `skip = floor(ln(U) * inv_log)` to jump to next fired shot
    /// - At fired shot: choose uniform alternative, XOR affected measurements
    ///
    /// Complexity: O(p * shots) per mechanism (geometric = O(fired events)).
    fn overlay_faults_geometric(&self, shots: usize, columns: &mut [Vec<u64>], rng: &mut PecosRng) {
        let num_words = columns.first().map_or(0, Vec::len);
        for (mech_idx, mechanism) in self.mechanisms.iter().enumerate() {
            let inv_log = self.inv_log_1_minus_p[mech_idx];
            let p = mechanism.probability;
            let num_alts = mechanism.alternatives.len();
            if num_alts == 0 {
                continue;
            }

            // p=1: every shot fires (handle before inv_log check since inv_log=0 for p=1)
            if p >= 1.0 {
                if num_alts == 1 {
                    let word_masks = full_shot_word_masks(shots, num_words);
                    apply_word_masks(columns, &mechanism.alternatives[0], &word_masks);
                } else {
                    let mut alt_word_masks = vec![vec![0u64; num_words]; num_alts];
                    for shot in 0..shots {
                        let word_idx = shot / 64;
                        let bit_idx = shot % 64;
                        let alt_idx = rng.random_range(0..num_alts);
                        alt_word_masks[alt_idx][word_idx] ^= 1u64 << bit_idx;
                    }
                    apply_alternative_word_masks(columns, mechanism, &alt_word_masks);
                }
                continue;
            }

            // Skip p=0 mechanisms (inv_log=0 means p≈0 or exactly 0)
            if p == 0.0 || inv_log == 0.0 {
                continue;
            }

            // Geometric skip sampling: O(fired events)
            let mut shot: usize = 0;
            let mut alt_word_masks: Option<Vec<Vec<u64>>> = None;
            while shot < shots {
                // Sample skip distance
                #[allow(clippy::cast_precision_loss)]
                let u = (rng.next_u64() as f64) / (u64::MAX as f64);
                let u = if u == 0.0 { f64::MIN_POSITIVE } else { u };
                #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
                let skip = (u.ln() * inv_log).floor() as usize;

                shot += skip;
                if shot >= shots {
                    break;
                }

                // This shot fires — choose alternative and XOR
                let word_idx = shot / 64;
                let bit_idx = shot % 64;
                let mask = 1u64 << bit_idx;

                let alt_idx = if num_alts == 1 {
                    0
                } else {
                    rng.random_range(0..num_alts)
                };
                alt_word_masks.get_or_insert_with(|| vec![vec![0u64; num_words]; num_alts])
                    [alt_idx][word_idx] ^= mask;

                shot += 1;
            }
            if let Some(alt_word_masks) = alt_word_masks {
                apply_alternative_word_masks(columns, mechanism, &alt_word_masks);
            }
        }
    }
}

fn full_shot_word_masks(shots: usize, num_words: usize) -> Vec<u64> {
    let mut masks = vec![!0u64; num_words];
    mask_partial_final_word(std::slice::from_mut(&mut masks), shots);
    masks
}

fn apply_alternative_word_masks(
    columns: &mut [Vec<u64>],
    mechanism: &FaultMechanism,
    alt_word_masks: &[Vec<u64>],
) {
    for (measurements, word_masks) in mechanism.alternatives.iter().zip(alt_word_masks) {
        apply_word_masks(columns, measurements, word_masks);
    }
}

fn apply_word_masks(columns: &mut [Vec<u64>], measurements: &[usize], word_masks: &[u64]) {
    if word_masks.iter().all(|&mask| mask == 0) {
        return;
    }
    for &meas_idx in measurements {
        if let Some(column) = columns.get_mut(meas_idx) {
            for (word, &mask) in column.iter_mut().zip(word_masks) {
                *word ^= mask;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!(
            (actual - expected).abs() < 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    #[test]
    fn test_record_effect_index_maps_measurement_effects_by_xor() {
        let det_records = vec![vec![-1], vec![-2, -1], vec![-1, -1], vec![-4], vec![1]];
        let obs_records = vec![vec![-2], vec![-1, -2]];
        let index = RecordEffectIndex::new(&det_records, &obs_records, 3);

        assert_eq!(index.detectors_for_measurements(&[2]), vec![0, 1]);
        assert_eq!(index.detectors_for_measurements(&[1, 2]), vec![0]);
        assert_eq!(index.observables_for_measurements(&[1]), vec![0, 1]);
        assert_eq!(index.observables_for_measurements(&[1, 2]), vec![0]);
    }

    /// Build a minimal `TickCircuit`: PZ(0) H(0) CX(0,1) H(0) MZ(0) PZ(0) H(0) CX(0,1) H(0) MZ(0)
    fn two_round_x_check() -> TickCircuit {
        let mut tc = TickCircuit::new();
        // Round 1
        tc.tick().h(&[QubitId(0)]);
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.tick().pz(&[QubitId(0)]);
        // Round 2
        tc.tick().h(&[QubitId(0)]);
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc
    }

    #[test]
    fn test_meas_fault_affects_single_measurement() {
        let tc = two_round_x_check();
        let noise = StochasticNoiseParams {
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.01,
            p_prep: 0.0,
        };
        let mechanisms = build_fault_table(&tc, &noise).unwrap();

        // Should have exactly 2 measurement mechanisms (one per MZ),
        // each with 1 alternative that flips that measurement.
        assert_eq!(mechanisms.len(), 2);
        assert_eq!(mechanisms[0].alternatives, vec![vec![0]]);
        assert_eq!(mechanisms[1].alternatives, vec![vec![1]]);
        assert!((mechanisms[0].probability - 0.01).abs() < 1e-10);
    }

    #[test]
    fn test_prep_fault_reaches_next_measurement_only() {
        let tc = two_round_x_check();
        let noise = StochasticNoiseParams {
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.01,
        };
        let mechanisms = build_fault_table(&tc, &noise).unwrap();

        // PZ(0) before round 2: single alternative affecting only m1
        let round2_prep = mechanisms.iter().find(|m| m.alternatives == vec![vec![1]]);
        assert!(
            round2_prep.is_some(),
            "PZ before round 2 should produce mechanism affecting m1"
        );
    }

    #[test]
    fn test_prep_fault_does_not_cross_pz() {
        let tc = two_round_x_check();
        let noise = StochasticNoiseParams {
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.01,
        };
        let mechanisms = build_fault_table(&tc, &noise).unwrap();

        // No alternative should affect BOTH m0 and m1 (PZ between rounds absorbs)
        for m in &mechanisms {
            for alt in &m.alternatives {
                assert!(
                    !(alt.contains(&0) && alt.contains(&1)),
                    "Fault alternative crosses PZ boundary: {alt:?}"
                );
            }
        }
    }

    #[test]
    fn test_flatten_tick_circuit_preserves_source_metadata() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0), QubitId(1)]);
        tc.tick()
            .cx(&[(QubitId(0), QubitId(1)), (QubitId(2), QubitId(3))]);
        tc.tick().mz(&[QubitId(0), QubitId(1)]);

        let (gates, meas_positions) = flatten_tick_circuit(&tc);

        assert_eq!(gates.len(), 6);
        assert_eq!(meas_positions.get(&4), Some(&0));
        assert_eq!(meas_positions.get(&5), Some(&1));

        assert_eq!(gates[0].tick, 0);
        assert_eq!(gates[0].gate_index, 0);
        assert_eq!(gates[0].gate_type, GateType::H);
        assert_eq!(gates[0].qubits, vec![0]);

        assert_eq!(gates[1].tick, 0);
        assert_eq!(gates[1].gate_index, 0);
        assert_eq!(gates[1].gate_type, GateType::H);
        assert_eq!(gates[1].qubits, vec![1]);

        assert_eq!(gates[2].tick, 1);
        assert_eq!(gates[2].gate_index, 0);
        assert_eq!(gates[2].gate_type, GateType::CX);
        assert_eq!(gates[2].qubits, vec![0, 1]);

        assert_eq!(gates[3].tick, 1);
        assert_eq!(gates[3].gate_index, 0);
        assert_eq!(gates[3].gate_type, GateType::CX);
        assert_eq!(gates[3].qubits, vec![2, 3]);

        assert_eq!(gates[4].tick, 2);
        assert_eq!(gates[4].gate_index, 0);
        assert_eq!(gates[4].gate_type, GateType::MZ);
        assert_eq!(gates[4].qubits, vec![0]);

        assert_eq!(gates[5].tick, 2);
        assert_eq!(gates[5].gate_index, 0);
        assert_eq!(gates[5].gate_type, GateType::MZ);
        assert_eq!(gates[5].qubits, vec![1]);
    }

    #[test]
    fn test_flatten_tick_circuit_skips_zero_gate_metadata_batches() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.insert_tick(1);
        tc.get_tick_mut(1)
            .unwrap()
            .add_gate(pecos_core::Gate::simple(
                GateType::TrackedPauliMeta,
                vec![QubitId(1), QubitId(2)],
            ));
        tc.tick().mz(&[QubitId(0)]);

        let (gates, meas_positions) = flatten_tick_circuit(&tc);

        assert_eq!(gates.len(), 2);
        assert_eq!(gates[0].gate_type, GateType::H);
        assert_eq!(gates[1].gate_type, GateType::MZ);
        assert_eq!(meas_positions.get(&1), Some(&0));
    }

    // ---- Direct propagation tests using propagate_single ----

    #[test]
    fn test_propagate_x_before_cx_reaches_target_mz() {
        // Circuit: CX(0,1) MZ(1)
        // X on q0 before CX: CX maps XI → XX → MZ(q1) sees X → flips
        let mut tc = TickCircuit::new();
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().mz(&[QubitId(1)]);

        let (gates, meas_pos) = flatten_tick_circuit(&tc);
        let affected = propagate_single(PauliType::X, 0, 0, &gates, &meas_pos);
        assert_eq!(
            affected,
            BTreeSet::from([0]),
            "X on q0 before CX(0,1) MZ(1) should flip m0"
        );
    }

    #[test]
    fn test_propagate_z_before_cx_stays_on_control() {
        // Circuit: CX(0,1) MZ(1)
        // Z on q0 before CX: CX maps ZI → ZI → MZ(q1) sees I → no flip
        let mut tc = TickCircuit::new();
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().mz(&[QubitId(1)]);

        let (gates, meas_pos) = flatten_tick_circuit(&tc);
        let affected = propagate_single(PauliType::Z, 0, 0, &gates, &meas_pos);
        assert!(
            affected.is_empty(),
            "Z on q0 before CX(0,1) should not reach MZ(q1)"
        );
    }

    #[test]
    fn test_propagate_x_on_target_unchanged_by_cx() {
        // Circuit: CX(0,1) MZ(1)
        // X on q1 before CX: CX maps IX → IX → MZ(q1) sees X → flips
        let mut tc = TickCircuit::new();
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().mz(&[QubitId(1)]);

        let (gates, meas_pos) = flatten_tick_circuit(&tc);
        let affected = propagate_single(PauliType::X, 1, 0, &gates, &meas_pos);
        assert_eq!(affected, BTreeSet::from([0]));
    }

    #[test]
    fn test_propagate_z_on_target_spreads_to_control_via_cx() {
        // Circuit: CX(0,1) MZ(0) MZ(1)
        // Z on q1 before CX: CX maps IZ → ZZ → MZ(q0) sees Z (no flip), MZ(q1) sees Z (no flip)
        let mut tc = TickCircuit::new();
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().mz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(1)]);

        let (gates, meas_pos) = flatten_tick_circuit(&tc);
        let affected = propagate_single(PauliType::Z, 1, 0, &gates, &meas_pos);
        assert!(
            affected.is_empty(),
            "Z errors don't flip Z-basis measurements"
        );
    }

    #[test]
    fn test_propagate_x_through_h_becomes_z() {
        // Circuit: H(0) MZ(0)
        // X on q0 at position 0: H maps X→Z → MZ sees Z → no flip
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);

        let (gates, meas_pos) = flatten_tick_circuit(&tc);
        let affected = propagate_single(PauliType::X, 0, 0, &gates, &meas_pos);
        assert!(
            affected.is_empty(),
            "X through H becomes Z, should not flip MZ"
        );
    }

    #[test]
    fn test_propagate_z_through_h_becomes_x() {
        // Circuit: H(0) MZ(0)
        // Z on q0 at position 0: H maps Z→X → MZ sees X → flips
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);

        let (gates, meas_pos) = flatten_tick_circuit(&tc);
        let affected = propagate_single(PauliType::Z, 0, 0, &gates, &meas_pos);
        assert_eq!(
            affected,
            BTreeSet::from([0]),
            "Z through H becomes X, should flip MZ"
        );
    }

    #[test]
    fn test_propagate_x_absorbed_by_pz() {
        // Circuit: PZ(0) MZ(0)
        // X on q0 at position 0: PZ absorbs it → MZ sees I → no flip
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);

        let (gates, meas_pos) = flatten_tick_circuit(&tc);
        let affected = propagate_single(PauliType::X, 0, 0, &gates, &meas_pos);
        assert!(affected.is_empty(), "X should be absorbed by PZ");
    }

    #[test]
    fn test_pz_absorbs_all_pauli_components_before_reset() {
        // Circuit: PZ(0) H(0) MZ(0)
        // Any fault before the reset is absorbed. Faults after the reset still
        // propagate through the H according to normal Clifford conjugation.
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[QubitId(0)]);
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);

        let (gates, meas_pos) = flatten_tick_circuit(&tc);
        for pauli in [PauliType::X, PauliType::Y, PauliType::Z] {
            let affected = propagate_single(pauli, 0, 0, &gates, &meas_pos);
            assert!(
                affected.is_empty(),
                "{pauli:?} before PZ should be absorbed by the reset"
            );
        }

        assert!(
            propagate_single(PauliType::X, 0, 1, &gates, &meas_pos).is_empty(),
            "X after PZ becomes Z through H and should not flip MZ"
        );
        assert_eq!(
            propagate_single(PauliType::Y, 0, 1, &gates, &meas_pos),
            BTreeSet::from([0]),
            "Y after PZ keeps an X component through H and should flip MZ"
        );
        assert_eq!(
            propagate_single(PauliType::Z, 0, 1, &gates, &meas_pos),
            BTreeSet::from([0]),
            "Z after PZ becomes X through H and should flip MZ"
        );
    }

    #[test]
    fn test_propagate_x_absorbed_by_mz() {
        // Circuit: MZ(0) MZ(0) — X on q0 should flip first MZ only
        // (MZ collapses qubit, absorbing the error)
        let mut tc = TickCircuit::new();
        tc.tick().mz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);

        let (gates, meas_pos) = flatten_tick_circuit(&tc);
        let affected = propagate_single(PauliType::X, 0, 0, &gates, &meas_pos);
        assert_eq!(
            affected,
            BTreeSet::from([0]),
            "X should flip first MZ only, not second"
        );
    }

    #[test]
    fn test_xor_combined_single_effects_match_pair_propagation() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().h(&[QubitId(1)]);
        tc.tick().mz(&[QubitId(0), QubitId(1)]);
        tc.tracked_pauli_labeled("tracked_z0", PauliString::z(0));
        tc.tracked_pauli_labeled("tracked_z1", PauliString::z(1));

        let (gates, meas_pos) = flatten_tick_circuit(&tc);
        let tracked_paulis = parse_tracked_pauli_annotations(&tc);
        let start = 1;
        let left =
            propagate_single_effect(PauliType::X, 0, start, &gates, &meas_pos, &tracked_paulis);
        let right =
            propagate_single_effect(PauliType::Z, 1, start, &gates, &meas_pos, &tracked_paulis);
        let combined = xor_fault_effects(&left, &right);
        let direct = propagate_pair_effect(
            [(PauliType::X, 0), (PauliType::Z, 1)],
            start,
            &gates,
            &meas_pos,
            &tracked_paulis,
        );

        assert_eq!(combined, direct);
    }

    #[test]
    fn test_propagated_effect_cache_matches_fresh_propagation() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.tracked_pauli_labeled("tracked_x0", PauliString::x(0));

        let (gates, meas_pos) = flatten_tick_circuit(&tc);
        let tracked_paulis = parse_tracked_pauli_annotations(&tc);
        let fresh = propagate_single_effect(PauliType::Z, 0, 0, &gates, &meas_pos, &tracked_paulis);

        let mut cache = PropagatedEffectCache::default();
        let first = cache.single(PauliType::Z, 0, 0, &gates, &meas_pos, &tracked_paulis);
        assert_eq!(first, fresh);
        assert_eq!(cache.len(), 1);

        let mut mutated_clone = first.clone();
        mutated_clone.affected_measurements.clear();
        mutated_clone.affected_tracked_paulis.clear();

        let second = cache.single(PauliType::Z, 0, 0, &gates, &meas_pos, &tracked_paulis);
        assert_eq!(second, fresh);
        assert_ne!(second, mutated_clone);
        assert_eq!(
            cache.len(),
            1,
            "repeating the same propagation key should reuse the cached entry"
        );

        let other = cache.single(PauliType::X, 0, 0, &gates, &meas_pos, &tracked_paulis);
        let other_fresh =
            propagate_single_effect(PauliType::X, 0, 0, &gates, &meas_pos, &tracked_paulis);
        assert_eq!(other, other_fresh);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_propagate_x_check_round_reaches_ancilla_only() {
        // X-check pattern: H(0) CX(0,1) CX(0,2) H(0) MZ(0)
        // X on q1 (data) at start: CX maps IX→IX on q1 (target stays).
        // After H-CX-CX-H, X on q1 doesn't propagate to ancilla.
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().cx(&[(QubitId(0), QubitId(2))]);
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);

        let (gates, meas_pos) = flatten_tick_circuit(&tc);

        // X on data q1: CX(ctrl=0, tgt=1) doesn't spread X from target to control.
        // So X stays on q1, never reaches MZ(q0).
        let affected = propagate_single(PauliType::X, 1, 0, &gates, &meas_pos);
        assert!(
            affected.is_empty(),
            "X on data qubit should not reach ancilla MZ in X-check"
        );

        // Z on data q1: CX maps IZ → ZZ (spreads to control q0).
        // Then H(q0) maps Z→X on ancilla. MZ(q0) sees X → flips.
        let affected = propagate_single(PauliType::Z, 1, 0, &gates, &meas_pos);
        assert_eq!(
            affected,
            BTreeSet::from([0]),
            "Z on data should reach ancilla MZ in X-check"
        );
    }

    #[test]
    fn test_empty_alternative_preserved_for_correct_denominator() {
        // H(0); MZ(0): p1 faults are injected AFTER H, directly before MZ.
        // The 3 alternatives (X, Y, Z injected between H and MZ):
        //   X: has X component → flips MZ
        //   Y: has X component → flips MZ
        //   Z: commutes with MZ → no flip (empty alternative)
        // All 3 must be present so each is chosen with probability 1/3.
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);

        let noise = StochasticNoiseParams {
            p1: 0.01,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.0,
        };
        let mechanisms = build_fault_table(&tc, &noise).unwrap();

        assert_eq!(mechanisms.len(), 1, "one mechanism for the H gate");
        let m = &mechanisms[0];
        assert_eq!(
            m.alternatives.len(),
            3,
            "all 3 Pauli alternatives must be present"
        );
        // Exactly one alternative should be empty (Z between H and MZ commutes)
        let empty_count = m.alternatives.iter().filter(|a| a.is_empty()).count();
        assert_eq!(
            empty_count, 1,
            "Z injected after H commutes with MZ — should be empty no-op alternative"
        );
    }

    #[test]
    fn test_zero_noise_produces_no_faults() {
        let tc = two_round_x_check();
        let noise = StochasticNoiseParams {
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.0,
        };
        let faults = build_fault_table(&tc, &noise).unwrap();
        assert!(faults.is_empty());
    }

    #[test]
    fn test_unsupported_gate_rejected_even_with_zero_noise() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().t(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);

        // Zero noise — validation runs on raw TickCircuit before anything else
        let noise = StochasticNoiseParams {
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.0,
        };
        let result = build_fault_table(&tc, &noise);
        assert!(result.is_err(), "T should be rejected");
        let err = result.unwrap_err();
        assert_eq!(err.gate_type, GateType::T);
        assert_eq!(err.tick, 1, "T is in tick 1");
        assert_eq!(err.gate_in_tick, 0, "T is gate 0 within that tick");
        assert_eq!(err.qubits, vec![0], "full original qubit list");
    }

    // ---- symbolic_measurement_history tests ----

    #[test]
    fn test_symbolic_history_rejects_unsupported_gate() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().t(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);

        let result = symbolic_measurement_history(&tc);
        assert!(result.is_err(), "T should be rejected");
        let err = result.unwrap_err();
        assert_eq!(err.gate_type, GateType::T);
        assert_eq!(err.tick, 1);
        assert_eq!(err.qubits, vec![0]);
    }

    #[test]
    fn test_symbolic_history_cy_circuit_succeeds() {
        // CY(0,1) MZ(1): should not error; CY is a valid Clifford gate
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        let pairs = [(QubitId(0), QubitId(1))];
        tc.tick().cy(&pairs);
        tc.tick().mz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(1)]);

        let history = symbolic_measurement_history(&tc);
        assert!(history.is_ok(), "CY should be supported");
        assert_eq!(history.unwrap().len(), 2);
    }

    #[test]
    fn test_symbolic_history_bell_produces_correct_kinds() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().mz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(1)]);

        let history = symbolic_measurement_history(&tc).unwrap();
        let kinds = MeasurementKind::from_history(&history);
        assert_eq!(kinds.len(), 2);
        assert!(matches!(kinds[0], MeasurementKind::Random));
        assert!(matches!(kinds[1], MeasurementKind::Copy(0)));
    }

    #[test]
    fn test_symbolic_history_reset_breaks_copy_chain_between_rounds() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().mz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(1)]);
        tc.tick().pz(&[QubitId(0)]);
        tc.tick().pz(&[QubitId(1)]);
        tc.tick().h(&[QubitId(0)]);
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().mz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(1)]);

        let history = symbolic_measurement_history(&tc).unwrap();
        let kinds = MeasurementKind::from_history(&history);
        assert_eq!(kinds.len(), 4);
        assert!(matches!(kinds[0], MeasurementKind::Random));
        assert!(matches!(kinds[1], MeasurementKind::Copy(0)));
        assert!(
            matches!(kinds[2], MeasurementKind::Random),
            "measurement after reset should introduce a fresh random source"
        );
        assert!(
            !matches!(kinds[2], MeasurementKind::Copy(0)),
            "reset must break the copy chain from the first round"
        );
        assert!(matches!(kinds[3], MeasurementKind::Copy(2)));
    }

    // ---- FaultCatalog tests ----

    #[test]
    fn test_catalog_single_qubit_depolarizing() {
        // H(0) MZ(0): p1 fault after H has 3 alternatives
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".to_string()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String("[]".to_string()),
        );
        tc.set_meta(
            "observables",
            pecos_quantum::Attribute::String("[]".to_string()),
        );

        let noise = StochasticNoiseParams {
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();

        // Should have exactly 1 location (H gate) with 3 alternatives
        let h_locs: Vec<_> = catalog
            .locations
            .iter()
            .filter(|l| l.gate_type == GateType::H)
            .collect();
        assert_eq!(h_locs.len(), 1);
        let loc = &h_locs[0];
        assert_eq!(loc.faults.len(), 3);
        assert_eq!(loc.channel, FaultChannel::P1);
        assert!((loc.channel_probability - 0.03).abs() < 1e-10);
        assert!((loc.no_fault_probability - 0.97).abs() < 1e-10);
        assert_eq!(loc.num_alternatives, 3);

        for fault in &loc.faults {
            assert_eq!(fault.kind, FaultKind::Pauli);
            assert!(fault.pauli.is_some());
            assert!((fault.conditional_probability - 1.0 / 3.0).abs() < 1e-10);
            assert!((fault.absolute_probability - 0.01).abs() < 1e-10);
        }
    }

    #[test]
    fn test_catalog_two_qubit_depolarizing() {
        // CX(0,1) MZ(0) MZ(1): p2 fault has 15 alternatives
        let mut tc = TickCircuit::new();
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().mz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(1)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("2".to_string()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String("[]".to_string()),
        );
        tc.set_meta(
            "observables",
            pecos_quantum::Attribute::String("[]".to_string()),
        );

        let noise = StochasticNoiseParams {
            p1: 0.0,
            p2: 0.15,
            p_meas: 0.0,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();

        let cx_locs: Vec<_> = catalog
            .locations
            .iter()
            .filter(|l| l.gate_type == GateType::CX)
            .collect();
        assert_eq!(cx_locs.len(), 1);
        let loc = &cx_locs[0];
        assert_eq!(loc.faults.len(), 15);
        assert_eq!(loc.num_alternatives, 15);

        for fault in &loc.faults {
            assert_eq!(fault.kind, FaultKind::Pauli);
            assert!(fault.pauli.is_some());
            assert!((fault.conditional_probability - 1.0 / 15.0).abs() < 1e-10);
            assert!((fault.absolute_probability - 0.01).abs() < 1e-10);
        }

        // Verify 9 two-qubit PauliStrings and 6 single-qubit PauliStrings
        let two_term: usize = loc
            .faults
            .iter()
            .filter(|f| f.pauli.as_ref().unwrap().iter_pairs().count() == 2)
            .count();
        let one_term: usize = loc
            .faults
            .iter()
            .filter(|f| f.pauli.as_ref().unwrap().iter_pairs().count() == 1)
            .count();
        assert_eq!(two_term, 9, "Should have 9 two-qubit Pauli alternatives");
        assert_eq!(one_term, 6, "Should have 6 single-qubit Pauli alternatives");
    }

    #[test]
    fn test_catalog_supports_all_traced_qis_clifford_gates() {
        let mut tc = TickCircuit::new();
        tc.tick().szdg(&[QubitId(0)]);
        tc.tick().sx(&[QubitId(0)]);
        tc.tick().sxdg(&[QubitId(1)]);
        tc.tick().sy(&[QubitId(0)]);
        tc.tick().sydg(&[QubitId(1)]);
        tc.tick().f(&[QubitId(0)]);
        tc.tick().fdg(&[QubitId(1)]);
        tc.tick().cy(&[(QubitId(0), QubitId(1))]);
        tc.tick().cz(&[(QubitId(0), QubitId(1))]);
        tc.tick().sxx(&[(QubitId(0), QubitId(1))]);
        tc.tick().sxxdg(&[(QubitId(0), QubitId(1))]);
        tc.tick().syy(&[(QubitId(0), QubitId(1))]);
        tc.tick().syydg(&[(QubitId(0), QubitId(1))]);
        tc.tick().szz(&[(QubitId(0), QubitId(1))]);
        tc.tick().szzdg(&[(QubitId(0), QubitId(1))]);
        tc.tick().swap(&[(QubitId(0), QubitId(1))]);
        tc.tick().mz(&[QubitId(0), QubitId(1)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("2".to_string()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String("[]".to_string()),
        );
        tc.set_meta(
            "observables",
            pecos_quantum::Attribute::String("[]".to_string()),
        );

        let noise = StochasticNoiseParams {
            p1: 0.03,
            p2: 0.15,
            p_meas: 0.0,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();

        for (gate_type, expected_alternatives) in [
            (GateType::SZdg, 3),
            (GateType::SX, 3),
            (GateType::SXdg, 3),
            (GateType::SY, 3),
            (GateType::SYdg, 3),
            (GateType::F, 3),
            (GateType::Fdg, 3),
            (GateType::CY, 15),
            (GateType::CZ, 15),
            (GateType::SXX, 15),
            (GateType::SXXdg, 15),
            (GateType::SYY, 15),
            (GateType::SYYdg, 15),
            (GateType::SZZ, 15),
            (GateType::SZZdg, 15),
            (GateType::SWAP, 15),
        ] {
            let locations: Vec<_> = catalog
                .locations
                .iter()
                .filter(|loc| loc.gate_type == gate_type)
                .collect();
            assert_eq!(locations.len(), 1, "{gate_type:?}");
            assert_eq!(
                locations[0].faults.len(),
                expected_alternatives,
                "{gate_type:?}"
            );
        }
    }

    #[test]
    fn test_catalog_fault_effects_through_new_clifford_gates() {
        fn fault_for_pauli<'a>(
            loc: &'a FaultLocation,
            pauli: &PauliString,
        ) -> &'a FaultAlternative {
            loc.faults
                .iter()
                .find(|fault| fault.pauli.as_ref() == Some(pauli))
                .expect("missing expected Pauli fault")
        }

        let mut single = TickCircuit::new();
        single.tick().h(&[QubitId(0)]);
        single.tick().sy(&[QubitId(0)]);
        single.tick().mz(&[QubitId(0)]);
        single.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".to_string()),
        );
        single.set_meta(
            "detectors",
            pecos_quantum::Attribute::String("[]".to_string()),
        );
        single.set_meta(
            "observables",
            pecos_quantum::Attribute::String("[]".to_string()),
        );

        let single_catalog = build_fault_catalog(
            &single,
            &StochasticNoiseParams {
                p1: 0.03,
                p2: 0.0,
                p_meas: 0.0,
                p_prep: 0.0,
            },
        )
        .unwrap();
        let h_loc = single_catalog
            .locations
            .iter()
            .find(|loc| loc.gate_type == GateType::H)
            .unwrap();
        assert_eq!(
            fault_for_pauli(h_loc, &pauli_type_to_string(PauliType::X, 0)).affected_measurements,
            Vec::<usize>::new(),
            "SY maps X to Z, so it should not flip MZ"
        );
        assert_eq!(
            fault_for_pauli(h_loc, &pauli_type_to_string(PauliType::Y, 0)).affected_measurements,
            vec![0],
            "SY maps Y to Y, so it should flip MZ"
        );
        assert_eq!(
            fault_for_pauli(h_loc, &pauli_type_to_string(PauliType::Z, 0)).affected_measurements,
            vec![0],
            "SY maps Z to X, so it should flip MZ"
        );

        let mut face = TickCircuit::new();
        face.tick().h(&[QubitId(0)]);
        face.tick().f(&[QubitId(0)]);
        face.tick().mz(&[QubitId(0)]);
        face.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".to_string()),
        );
        face.set_meta(
            "detectors",
            pecos_quantum::Attribute::String("[]".to_string()),
        );
        face.set_meta(
            "observables",
            pecos_quantum::Attribute::String("[]".to_string()),
        );

        let face_catalog = build_fault_catalog(
            &face,
            &StochasticNoiseParams {
                p1: 0.03,
                p2: 0.0,
                p_meas: 0.0,
                p_prep: 0.0,
            },
        )
        .unwrap();
        let h_loc = face_catalog
            .locations
            .iter()
            .find(|loc| loc.gate_type == GateType::H)
            .unwrap();
        assert_eq!(
            fault_for_pauli(h_loc, &pauli_type_to_string(PauliType::X, 0)).affected_measurements,
            vec![0],
            "F maps X to Y, so it should flip MZ"
        );
        assert_eq!(
            fault_for_pauli(h_loc, &pauli_type_to_string(PauliType::Y, 0)).affected_measurements,
            Vec::<usize>::new(),
            "F maps Y to Z, so it should not flip MZ"
        );
        assert_eq!(
            fault_for_pauli(h_loc, &pauli_type_to_string(PauliType::Z, 0)).affected_measurements,
            vec![0],
            "F maps Z to X, so it should flip MZ"
        );

        let mut face_dagger = TickCircuit::new();
        face_dagger.tick().h(&[QubitId(0)]);
        face_dagger.tick().fdg(&[QubitId(0)]);
        face_dagger.tick().mz(&[QubitId(0)]);
        face_dagger.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".to_string()),
        );
        face_dagger.set_meta(
            "detectors",
            pecos_quantum::Attribute::String("[]".to_string()),
        );
        face_dagger.set_meta(
            "observables",
            pecos_quantum::Attribute::String("[]".to_string()),
        );

        let face_dagger_catalog = build_fault_catalog(
            &face_dagger,
            &StochasticNoiseParams {
                p1: 0.03,
                p2: 0.0,
                p_meas: 0.0,
                p_prep: 0.0,
            },
        )
        .unwrap();
        let h_loc = face_dagger_catalog
            .locations
            .iter()
            .find(|loc| loc.gate_type == GateType::H)
            .unwrap();
        assert_eq!(
            fault_for_pauli(h_loc, &pauli_type_to_string(PauliType::X, 0)).affected_measurements,
            Vec::<usize>::new(),
            "Fdg maps X to Z, so it should not flip MZ"
        );
        assert_eq!(
            fault_for_pauli(h_loc, &pauli_type_to_string(PauliType::Y, 0)).affected_measurements,
            vec![0],
            "Fdg maps Y to X, so it should flip MZ"
        );
        assert_eq!(
            fault_for_pauli(h_loc, &pauli_type_to_string(PauliType::Z, 0)).affected_measurements,
            vec![0],
            "Fdg maps Z to Y, so it should flip MZ"
        );

        let mut two_qubit = TickCircuit::new();
        two_qubit.tick().cx(&[(QubitId(0), QubitId(1))]);
        two_qubit.tick().sxx(&[(QubitId(0), QubitId(1))]);
        two_qubit.tick().mz(&[QubitId(0), QubitId(1)]);
        two_qubit.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("2".to_string()),
        );
        two_qubit.set_meta(
            "detectors",
            pecos_quantum::Attribute::String("[]".to_string()),
        );
        two_qubit.set_meta(
            "observables",
            pecos_quantum::Attribute::String("[]".to_string()),
        );

        let two_catalog = build_fault_catalog(
            &two_qubit,
            &StochasticNoiseParams {
                p1: 0.0,
                p2: 0.15,
                p_meas: 0.0,
                p_prep: 0.0,
            },
        )
        .unwrap();
        let cx_loc = two_catalog
            .locations
            .iter()
            .find(|loc| loc.gate_type == GateType::CX)
            .unwrap();
        assert_eq!(
            fault_for_pauli(cx_loc, &pauli_type_to_string(PauliType::X, 0)).affected_measurements,
            vec![0],
            "SXX leaves XI as XI"
        );
        assert_eq!(
            fault_for_pauli(cx_loc, &pauli_type_to_string(PauliType::X, 1)).affected_measurements,
            vec![1],
            "SXX leaves IX as IX"
        );
        assert_eq!(
            fault_for_pauli(cx_loc, &pauli_type_to_string(PauliType::Z, 0)).affected_measurements,
            vec![0, 1],
            "SXX maps ZI to YX"
        );
        assert_eq!(
            fault_for_pauli(cx_loc, &pauli_type_to_string(PauliType::Z, 1)).affected_measurements,
            vec![0, 1],
            "SXX maps IZ to XY"
        );
    }

    #[test]
    fn test_catalog_keeps_observables_and_tracked_paulis_distinct() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tracked_pauli_labeled("tracked_z0", PauliString::z(0));
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String("[]".to_string()),
        );
        tc.set_meta(
            "observables",
            pecos_quantum::Attribute::String("[]".to_string()),
        );

        let catalog = build_fault_catalog(
            &tc,
            &StochasticNoiseParams {
                p1: 0.03,
                p2: 0.0,
                p_meas: 0.0,
                p_prep: 0.0,
            },
        )
        .unwrap();

        let h_loc = catalog
            .locations
            .iter()
            .find(|loc| loc.gate_type == GateType::H)
            .unwrap();
        let x_fault = h_loc
            .faults
            .iter()
            .find(|fault| fault.pauli.as_ref() == Some(&PauliString::x(0)))
            .unwrap();
        let y_fault = h_loc
            .faults
            .iter()
            .find(|fault| fault.pauli.as_ref() == Some(&PauliString::y(0)))
            .unwrap();
        let z_fault = h_loc
            .faults
            .iter()
            .find(|fault| fault.pauli.as_ref() == Some(&PauliString::z(0)))
            .unwrap();

        assert_eq!(x_fault.affected_observables, Vec::<usize>::new());
        assert_eq!(x_fault.affected_tracked_paulis, vec![0]);
        assert_eq!(y_fault.affected_tracked_paulis, vec![0]);
        assert_eq!(z_fault.affected_tracked_paulis, Vec::<usize>::new());

        let configs: Vec<_> = catalog.fault_configurations(1).collect();
        assert!(
            configs
                .iter()
                .any(|config| config.affected_tracked_paulis.as_slice() == [0]
                    && config.affected_observables.is_empty())
        );
    }

    #[test]
    fn test_catalog_after_tick_dag_round_trip_keeps_outputs_and_tracked_paulis_separate() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0), QubitId(1)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String(tc.num_measurements().to_string()),
        );
        tc.add_detector_metadata(&[-1], None, Some("D0"), Some(0))
            .unwrap();
        tc.add_observable_metadata(&[-1], Some(0), Some("L0"))
            .unwrap();
        tc.tracked_pauli_labeled("tracked_z1", PauliString::z(1));

        let round_tripped = TickCircuit::from(&pecos_quantum::DagCircuit::from(&tc));
        assert_eq!(round_tripped.annotations().len(), 1);
        assert!(matches!(
            round_tripped.annotations()[0].kind,
            AnnotationKind::TrackedPauli
        ));

        let catalog = build_fault_catalog(
            &round_tripped,
            &StochasticNoiseParams {
                p1: 0.03,
                p2: 0.0,
                p_meas: 0.01,
                p_prep: 0.0,
            },
        )
        .unwrap();

        let h_loc = catalog
            .locations
            .iter()
            .find(|loc| loc.gate_type == GateType::H && loc.qubits.as_slice() == [0])
            .unwrap();
        let x_fault = h_loc
            .faults
            .iter()
            .find(|fault| fault.pauli.as_ref() == Some(&PauliString::x(0)))
            .unwrap();
        assert_eq!(x_fault.affected_measurements, vec![0]);
        assert_eq!(x_fault.affected_detectors, vec![0]);
        assert_eq!(x_fault.affected_observables, vec![0]);
        assert!(x_fault.affected_tracked_paulis.is_empty());

        let tracked_h_loc = catalog
            .locations
            .iter()
            .find(|loc| loc.gate_type == GateType::H && loc.qubits.as_slice() == [1])
            .unwrap();
        let tracked_x_fault = tracked_h_loc
            .faults
            .iter()
            .find(|fault| fault.pauli.as_ref() == Some(&PauliString::x(1)))
            .unwrap();
        assert!(tracked_x_fault.affected_measurements.is_empty());
        assert!(tracked_x_fault.affected_detectors.is_empty());
        assert!(tracked_x_fault.affected_observables.is_empty());
        assert_eq!(tracked_x_fault.affected_tracked_paulis, vec![0]);

        let meas_fault = catalog
            .locations
            .iter()
            .find(|loc| loc.channel == FaultChannel::PMeas)
            .and_then(|loc| loc.faults.first())
            .unwrap();
        assert_eq!(meas_fault.affected_measurements, vec![0]);
        assert_eq!(meas_fault.affected_detectors, vec![0]);
        assert_eq!(meas_fault.affected_observables, vec![0]);
        assert!(meas_fault.affected_tracked_paulis.is_empty());

        assert!(catalog.to_mechanisms().iter().any(|mechanism| {
            mechanism
                .alternatives
                .iter()
                .any(|alternative| alternative.as_slice() == [0])
        }));
    }

    #[test]
    fn test_catalog_two_qubit_propagation_keeps_output_kinds_distinct() {
        fn assert_case(gate_type: GateType, tracked_pauli: PauliString) {
            let mut tc = TickCircuit::new();
            tc.tick().h(&[QubitId(0)]);
            match gate_type {
                GateType::CX => {
                    tc.tick().cx(&[(QubitId(0), QubitId(1))]);
                    tc.tick().mz(&[QubitId(0)]);
                }
                GateType::CZ => {
                    tc.tick().cz(&[(QubitId(0), QubitId(1))]);
                    tc.tick().mz(&[QubitId(0)]);
                }
                GateType::SWAP => {
                    tc.tick().swap(&[(QubitId(0), QubitId(1))]);
                    tc.tick().cx(&[(QubitId(1), QubitId(2))]);
                    tc.tick().mz(&[QubitId(2)]);
                }
                other => panic!("unexpected gate type {other:?}"),
            }
            tc.set_meta(
                "num_measurements",
                pecos_quantum::Attribute::String(tc.num_measurements().to_string()),
            );
            tc.add_detector_metadata(&[-1], None, Some("D0"), Some(0))
                .unwrap();
            tc.add_observable_metadata(&[-1], Some(0), Some("L0"))
                .unwrap();
            tc.tracked_pauli_labeled("tracked", tracked_pauli);

            let catalog = build_fault_catalog(
                &tc,
                &StochasticNoiseParams {
                    p1: 0.03,
                    p2: 0.0,
                    p_meas: 0.0,
                    p_prep: 0.0,
                },
            )
            .unwrap();

            let h_loc = catalog
                .locations
                .iter()
                .find(|loc| loc.gate_type == GateType::H && loc.qubits.as_slice() == [0])
                .unwrap();
            let x_fault = h_loc
                .faults
                .iter()
                .find(|fault| fault.pauli.as_ref() == Some(&PauliString::x(0)))
                .unwrap();

            assert_eq!(x_fault.affected_measurements, vec![0], "{gate_type:?}");
            assert_eq!(x_fault.affected_detectors, vec![0], "{gate_type:?}");
            assert_eq!(x_fault.affected_observables, vec![0], "{gate_type:?}");
            assert_eq!(x_fault.affected_tracked_paulis, vec![0], "{gate_type:?}");
        }

        // X0 before CX becomes X0 X1.
        assert_case(GateType::CX, PauliString::z(1));
        // X0 before CZ becomes X0 Z1.
        assert_case(GateType::CZ, PauliString::x(1));
        // X0 before SWAP becomes X1, then the extra CX maps it to X1 X2.
        assert_case(GateType::SWAP, PauliString::z(1));
    }

    #[test]
    fn test_catalog_all_two_qubit_cliffords_propagate_x_fault_measurement_support() {
        fn apply_gate(tc: &mut TickCircuit, gate_type: GateType) {
            match gate_type {
                GateType::CX => {
                    tc.tick().cx(&[(QubitId(0), QubitId(1))]);
                }
                GateType::CY => {
                    tc.tick().cy(&[(QubitId(0), QubitId(1))]);
                }
                GateType::CZ => {
                    tc.tick().cz(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SXX => {
                    tc.tick().sxx(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SXXdg => {
                    tc.tick().sxxdg(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SYY => {
                    tc.tick().syy(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SYYdg => {
                    tc.tick().syydg(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SZZ => {
                    tc.tick().szz(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SZZdg => {
                    tc.tick().szzdg(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SWAP => {
                    tc.tick().swap(&[(QubitId(0), QubitId(1))]);
                }
                other => panic!("unexpected gate type {other:?}"),
            }
        }

        for (gate_type, expected_measurements) in [
            (GateType::CX, &[0usize, 1][..]),
            (GateType::CY, &[0usize, 1][..]),
            (GateType::CZ, &[0usize][..]),
            (GateType::SXX, &[0usize][..]),
            (GateType::SXXdg, &[0usize][..]),
            (GateType::SYY, &[1usize][..]),
            (GateType::SYYdg, &[1usize][..]),
            (GateType::SZZ, &[0usize][..]),
            (GateType::SZZdg, &[0usize][..]),
            (GateType::SWAP, &[1usize][..]),
        ] {
            let mut tc = TickCircuit::new();
            tc.tick().h(&[QubitId(0)]);
            apply_gate(&mut tc, gate_type);
            tc.tick().mz(&[QubitId(0), QubitId(1)]);

            let catalog = build_fault_catalog(
                &tc,
                &StochasticNoiseParams {
                    p1: 0.03,
                    p2: 0.0,
                    p_meas: 0.0,
                    p_prep: 0.0,
                },
            )
            .unwrap();

            let h_loc = catalog
                .locations
                .iter()
                .find(|loc| loc.gate_type == GateType::H && loc.qubits.as_slice() == [0])
                .unwrap();
            let x_fault = h_loc
                .faults
                .iter()
                .find(|fault| fault.pauli.as_ref() == Some(&PauliString::x(0)))
                .unwrap();

            assert_eq!(
                x_fault.affected_measurements.as_slice(),
                expected_measurements,
                "{gate_type:?}"
            );
        }
    }

    #[test]
    fn test_catalog_standard_cliffords_match_forward_pauli_oracle_for_all_alternatives() {
        fn apply_single_gate(tc: &mut TickCircuit, gate_type: GateType) {
            match gate_type {
                GateType::X => {
                    tc.tick().x(&[QubitId(0)]);
                }
                GateType::Y => {
                    tc.tick().y(&[QubitId(0)]);
                }
                GateType::Z => {
                    tc.tick().z(&[QubitId(0)]);
                }
                GateType::H => {
                    tc.tick().h(&[QubitId(0)]);
                }
                GateType::SZ => {
                    tc.tick().sz(&[QubitId(0)]);
                }
                GateType::SZdg => {
                    tc.tick().szdg(&[QubitId(0)]);
                }
                GateType::SX => {
                    tc.tick().sx(&[QubitId(0)]);
                }
                GateType::SXdg => {
                    tc.tick().sxdg(&[QubitId(0)]);
                }
                GateType::SY => {
                    tc.tick().sy(&[QubitId(0)]);
                }
                GateType::SYdg => {
                    tc.tick().sydg(&[QubitId(0)]);
                }
                GateType::F => {
                    tc.tick().f(&[QubitId(0)]);
                }
                GateType::Fdg => {
                    tc.tick().fdg(&[QubitId(0)]);
                }
                other => panic!("unexpected single-qubit gate {other:?}"),
            }
        }

        fn apply_pair_gate(tc: &mut TickCircuit, gate_type: GateType) {
            match gate_type {
                GateType::CX => {
                    tc.tick().cx(&[(QubitId(0), QubitId(1))]);
                }
                GateType::CY => {
                    tc.tick().cy(&[(QubitId(0), QubitId(1))]);
                }
                GateType::CZ => {
                    tc.tick().cz(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SXX => {
                    tc.tick().sxx(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SXXdg => {
                    tc.tick().sxxdg(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SYY => {
                    tc.tick().syy(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SYYdg => {
                    tc.tick().syydg(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SZZ => {
                    tc.tick().szz(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SZZdg => {
                    tc.tick().szzdg(&[(QubitId(0), QubitId(1))]);
                }
                GateType::SWAP => {
                    tc.tick().swap(&[(QubitId(0), QubitId(1))]);
                }
                other => panic!("unexpected two-qubit gate {other:?}"),
            }
        }

        fn pauli_type(pauli: Pauli) -> PauliType {
            match pauli {
                Pauli::X => PauliType::X,
                Pauli::Y => PauliType::Y,
                Pauli::Z => PauliType::Z,
                Pauli::I => panic!("identity is not a fault alternative"),
            }
        }

        fn expected_effect(
            pauli: &PauliString,
            start: usize,
            gates: &[GateLoc],
            meas_positions: &HashMap<usize, usize>,
            tracked_paulis: &[PauliString],
        ) -> PropagatedFaultEffect {
            let terms: Vec<_> = pauli
                .iter_pairs()
                .map(|(p, q)| (pauli_type(p), q.index()))
                .collect();
            match terms.as_slice() {
                [(p, q)] => {
                    propagate_single_effect(*p, *q, start, gates, meas_positions, tracked_paulis)
                }
                [(p0, q0), (p1, q1)] => propagate_pair_effect(
                    [(*p0, *q0), (*p1, *q1)],
                    start,
                    gates,
                    meas_positions,
                    tracked_paulis,
                ),
                other => panic!("expected one- or two-qubit Pauli alternative, got {other:?}"),
            }
        }

        for gate_type in STANDARD_1Q_CLIFFORD_GATES {
            let mut tc = TickCircuit::new();
            tc.tick().h(&[QubitId(0)]);
            apply_single_gate(&mut tc, *gate_type);
            tc.tick().cx(&[(QubitId(0), QubitId(1))]);
            tc.tick().mz(&[QubitId(0)]);
            tc.set_meta(
                "num_measurements",
                pecos_quantum::Attribute::String(tc.num_measurements().to_string()),
            );
            tc.add_detector_metadata(&[-1], None, Some("D0"), Some(0))
                .unwrap();
            tc.add_observable_metadata(&[-1], Some(0), Some("L0"))
                .unwrap();
            tc.tracked_pauli_labeled("tracked_x1", PauliString::x(1));
            tc.tracked_pauli_labeled("tracked_y1", PauliString::y(1));
            tc.tracked_pauli_labeled("tracked_z1", PauliString::z(1));

            let (gates, meas_positions) = flatten_tick_circuit(&tc);
            let tracked_paulis = parse_tracked_pauli_annotations(&tc);
            let catalog = build_fault_catalog(
                &tc,
                &StochasticNoiseParams {
                    p1: 0.03,
                    p2: 0.0,
                    p_meas: 0.0,
                    p_prep: 0.0,
                },
            )
            .unwrap();
            let source_loc_idx = gates
                .iter()
                .position(|loc| loc.gate_type == GateType::H && loc.qubits.as_slice() == [0])
                .unwrap();
            let source_loc = catalog
                .locations
                .iter()
                .find(|loc| {
                    loc.tick == gates[source_loc_idx].tick
                        && loc.gate_index == gates[source_loc_idx].gate_index
                })
                .unwrap();

            for fault in &source_loc.faults {
                let pauli = fault.pauli.as_ref().unwrap();
                let effect = expected_effect(
                    pauli,
                    source_loc_idx + 1,
                    &gates,
                    &meas_positions,
                    &tracked_paulis,
                );
                let measurements: Vec<_> = effect.affected_measurements.iter().copied().collect();
                assert_eq!(
                    fault.affected_measurements, measurements,
                    "{gate_type:?} {pauli:?}"
                );
                assert_eq!(
                    fault.affected_detectors, measurements,
                    "{gate_type:?} {pauli:?}"
                );
                assert_eq!(
                    fault.affected_observables, measurements,
                    "{gate_type:?} {pauli:?}"
                );
                assert_eq!(
                    fault.affected_tracked_paulis, effect.affected_tracked_paulis,
                    "{gate_type:?} {pauli:?}"
                );
            }
        }

        for gate_type in STANDARD_2Q_CLIFFORD_GATES {
            let mut tc = TickCircuit::new();
            tc.tick().cx(&[(QubitId(0), QubitId(1))]);
            apply_pair_gate(&mut tc, *gate_type);
            tc.tick()
                .cx(&[(QubitId(0), QubitId(2)), (QubitId(1), QubitId(3))]);
            tc.tick().mz(&[QubitId(0), QubitId(1)]);
            tc.set_meta(
                "num_measurements",
                pecos_quantum::Attribute::String(tc.num_measurements().to_string()),
            );
            tc.add_detector_metadata(&[-2], None, Some("D0"), Some(0))
                .unwrap();
            tc.add_detector_metadata(&[-1], None, Some("D1"), Some(1))
                .unwrap();
            tc.add_observable_metadata(&[-2], Some(0), Some("L0"))
                .unwrap();
            tc.add_observable_metadata(&[-1], Some(1), Some("L1"))
                .unwrap();
            tc.tracked_pauli_labeled("tracked_x2", PauliString::x(2));
            tc.tracked_pauli_labeled("tracked_y2", PauliString::y(2));
            tc.tracked_pauli_labeled("tracked_z2", PauliString::z(2));
            tc.tracked_pauli_labeled("tracked_x3", PauliString::x(3));
            tc.tracked_pauli_labeled("tracked_y3", PauliString::y(3));
            tc.tracked_pauli_labeled("tracked_z3", PauliString::z(3));

            let (gates, meas_positions) = flatten_tick_circuit(&tc);
            let tracked_paulis = parse_tracked_pauli_annotations(&tc);
            let catalog = build_fault_catalog(
                &tc,
                &StochasticNoiseParams {
                    p1: 0.0,
                    p2: 0.03,
                    p_meas: 0.0,
                    p_prep: 0.0,
                },
            )
            .unwrap();
            let source_loc_idx = gates
                .iter()
                .position(|loc| loc.gate_type == GateType::CX && loc.qubits.as_slice() == [0, 1])
                .unwrap();
            let source_loc = catalog
                .locations
                .iter()
                .find(|loc| {
                    loc.tick == gates[source_loc_idx].tick
                        && loc.gate_index == gates[source_loc_idx].gate_index
                })
                .unwrap();

            for fault in &source_loc.faults {
                let pauli = fault.pauli.as_ref().unwrap();
                let effect = expected_effect(
                    pauli,
                    source_loc_idx + 1,
                    &gates,
                    &meas_positions,
                    &tracked_paulis,
                );
                let measurements: Vec<_> = effect.affected_measurements.iter().copied().collect();
                assert_eq!(
                    fault.affected_measurements, measurements,
                    "{gate_type:?} {pauli:?}"
                );
                assert_eq!(
                    fault.affected_detectors, measurements,
                    "{gate_type:?} {pauli:?}"
                );
                assert_eq!(
                    fault.affected_observables, measurements,
                    "{gate_type:?} {pauli:?}"
                );
                assert_eq!(
                    fault.affected_tracked_paulis, effect.affected_tracked_paulis,
                    "{gate_type:?} {pauli:?}"
                );
            }
        }
    }

    #[test]
    fn test_fault_configurations_xor_detectors_observables_and_tracked_paulis_separately() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0), QubitId(1)]);
        tc.tick().cx(&[(QubitId(0), QubitId(2))]);
        tc.tick().mz(&[QubitId(0), QubitId(1)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String(tc.num_measurements().to_string()),
        );
        tc.add_detector_metadata(&[-2, -1], None, Some("D0"), Some(0))
            .unwrap();
        tc.add_observable_metadata(&[-2], Some(0), Some("L0"))
            .unwrap();
        tc.tracked_pauli_labeled("tracked_z2", PauliString::z(2));

        let catalog = build_fault_catalog(
            &tc,
            &StochasticNoiseParams {
                p1: 1.0,
                p2: 0.0,
                p_meas: 0.0,
                p_prep: 0.0,
            },
        )
        .unwrap();
        let h0 = catalog
            .locations
            .iter()
            .position(|loc| loc.gate_type == GateType::H && loc.qubits.as_slice() == [0])
            .unwrap();
        let h1 = catalog
            .locations
            .iter()
            .position(|loc| loc.gate_type == GateType::H && loc.qubits.as_slice() == [1])
            .unwrap();
        let x0 = catalog.locations[h0]
            .faults
            .iter()
            .position(|fault| fault.pauli.as_ref() == Some(&PauliString::x(0)))
            .unwrap();
        let x1 = catalog.locations[h1]
            .faults
            .iter()
            .position(|fault| fault.pauli.as_ref() == Some(&PauliString::x(1)))
            .unwrap();

        let config = catalog
            .fault_configurations(2)
            .find(|config| {
                config.location_indices == [h0, h1] && config.alternative_indices == [x0, x1]
            })
            .unwrap();

        assert_eq!(catalog.locations[h0].faults[x0].affected_detectors, [0]);
        assert_eq!(catalog.locations[h0].faults[x0].affected_observables, [0]);
        assert_eq!(
            catalog.locations[h0].faults[x0].affected_tracked_paulis,
            [0]
        );
        assert_eq!(catalog.locations[h1].faults[x1].affected_detectors, [0]);
        assert!(
            catalog.locations[h1].faults[x1]
                .affected_observables
                .is_empty()
        );
        assert!(
            catalog.locations[h1].faults[x1]
                .affected_tracked_paulis
                .is_empty()
        );

        assert_eq!(config.affected_measurements, [0, 1]);
        assert!(config.affected_detectors.is_empty());
        assert_eq!(config.affected_observables, [0]);
        assert_eq!(config.affected_tracked_paulis, [0]);
    }

    #[test]
    fn test_tracked_pauli_phase_is_ignored_for_flip_tracking() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tracked_pauli_labeled("plus_z0", PauliString::z(0));
        tc.tracked_pauli_labeled(
            "minus_z0",
            PauliString::with_phase_and_paulis(
                pecos_core::QuarterPhase::MinusOne,
                vec![(Pauli::Z, QubitId(0))],
            ),
        );

        let tracked_paulis = parse_tracked_pauli_annotations(&tc);
        assert_eq!(tracked_paulis.len(), 2);
        assert!(
            tracked_paulis
                .iter()
                .all(|op| op.phase() == pecos_core::QuarterPhase::PlusOne)
        );
        assert_eq!(tracked_paulis[0], tracked_paulis[1]);

        let catalog = build_fault_catalog(
            &tc,
            &StochasticNoiseParams {
                p1: 0.03,
                p2: 0.0,
                p_meas: 0.0,
                p_prep: 0.0,
            },
        )
        .unwrap();

        let h_loc = catalog
            .locations
            .iter()
            .find(|loc| loc.gate_type == GateType::H)
            .unwrap();
        let x_fault = h_loc
            .faults
            .iter()
            .find(|fault| fault.pauli.as_ref() == Some(&PauliString::x(0)))
            .unwrap();
        let z_fault = h_loc
            .faults
            .iter()
            .find(|fault| fault.pauli.as_ref() == Some(&PauliString::z(0)))
            .unwrap();

        assert_eq!(x_fault.affected_tracked_paulis, vec![0, 1]);
        assert_eq!(z_fault.affected_tracked_paulis, Vec::<usize>::new());
    }

    #[test]
    fn test_structural_catalog_includes_zero_probability_locations() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".to_string()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String("[]".to_string()),
        );
        tc.set_meta(
            "observables",
            pecos_quantum::Attribute::String("[]".to_string()),
        );

        let catalog = FaultCatalog::from_circuit(&tc).unwrap();
        assert_eq!(catalog.locations.len(), 2);
        assert!(
            catalog
                .locations
                .iter()
                .all(|loc| loc.channel_probability.abs() < 1e-12)
        );
        assert!(
            catalog
                .locations
                .iter()
                .all(|loc| (loc.no_fault_probability - 1.0).abs() < 1e-12)
        );
        assert!(
            catalog
                .locations
                .iter()
                .flat_map(|loc| &loc.faults)
                .all(|fault| fault.absolute_probability.abs() < 1e-12)
        );
        assert!(
            catalog
                .locations
                .iter()
                .any(|loc| loc.channel == FaultChannel::P1)
        );
        assert!(
            catalog
                .locations
                .iter()
                .any(|loc| loc.channel == FaultChannel::PMeas)
        );
    }

    #[test]
    fn test_parameterized_matches_direct_for_fully_nonzero_noise() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[QubitId(0)]);
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".to_string()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String(r#"[{"records":[-1]}]"#.to_string()),
        );
        tc.set_meta(
            "observables",
            pecos_quantum::Attribute::String(r#"[{"records":[-1]}]"#.to_string()),
        );

        let noise = StochasticNoiseParams {
            p1: 0.03,
            p2: 0.02,
            p_meas: 0.01,
            p_prep: 0.004,
        };
        let direct = build_fault_catalog(&tc, &noise).unwrap();
        let mut split = FaultCatalog::from_circuit(&tc).unwrap();
        split.with_noise(&noise);

        assert_eq!(direct.locations.len(), split.locations.len());
        for (a, b) in direct.locations.iter().zip(&split.locations) {
            assert_eq!(a.tick, b.tick);
            assert_eq!(a.gate_index, b.gate_index);
            assert_eq!(a.gate_type, b.gate_type);
            assert_eq!(a.qubits, b.qubits);
            assert_eq!(a.channel, b.channel);
            assert_close(a.channel_probability, b.channel_probability);
            assert_close(a.no_fault_probability, b.no_fault_probability);
            assert_eq!(a.num_alternatives, b.num_alternatives);
            assert_eq!(a.faults.len(), b.faults.len());
            for (af, bf) in a.faults.iter().zip(&b.faults) {
                assert_eq!(af.kind, bf.kind);
                assert_eq!(af.pauli, bf.pauli);
                assert_eq!(af.affected_measurements, bf.affected_measurements);
                assert_eq!(af.affected_detectors, bf.affected_detectors);
                assert_eq!(af.affected_observables, bf.affected_observables);
                assert_eq!(af.affected_tracked_paulis, bf.affected_tracked_paulis);
                assert_close(af.conditional_probability, bf.conditional_probability);
                assert_close(af.absolute_probability, bf.absolute_probability);
            }
        }
    }

    #[test]
    fn test_with_noise_overwrites_previous_probabilities() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".to_string()),
        );

        let mut catalog = FaultCatalog::from_circuit(&tc).unwrap();
        catalog.with_noise(&StochasticNoiseParams {
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.01,
            p_prep: 0.0,
        });
        catalog.with_noise(&StochasticNoiseParams {
            p1: 0.09,
            p2: 0.0,
            p_meas: 0.02,
            p_prep: 0.0,
        });

        let h_loc = catalog
            .locations
            .iter()
            .find(|loc| loc.channel == FaultChannel::P1)
            .unwrap();
        assert_close(h_loc.channel_probability, 0.09);
        assert_close(h_loc.no_fault_probability, 0.91);
        assert!(
            h_loc
                .faults
                .iter()
                .all(|fault| (fault.absolute_probability - 0.03).abs() < 1e-12)
        );

        let meas_loc = catalog
            .locations
            .iter()
            .find(|loc| loc.channel == FaultChannel::PMeas)
            .unwrap();
        assert_close(meas_loc.channel_probability, 0.02);
        assert_close(meas_loc.faults[0].absolute_probability, 0.02);
    }

    #[test]
    fn test_sparse_channel_keeps_structure_but_filters_raw_mechanisms() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".to_string()),
        );

        let noise = StochasticNoiseParams {
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.02,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();
        assert_eq!(catalog.locations.len(), 2);
        assert!(
            catalog
                .locations
                .iter()
                .any(|loc| loc.channel == FaultChannel::P1 && loc.channel_probability.abs() < 1e-12)
        );

        let mechanisms = catalog.to_mechanisms();
        assert_eq!(mechanisms.len(), 1);
        assert_close(mechanisms[0].probability, 0.02);
        assert_eq!(mechanisms[0].alternatives, vec![vec![0]]);
        assert_eq!(mechanisms, build_fault_table(&tc, &noise).unwrap());
    }

    #[test]
    fn test_fault_configurations_skip_zero_probability_fault_events() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".to_string()),
        );

        let catalog = FaultCatalog::from_circuit(&tc).unwrap();
        assert_eq!(catalog.fault_configurations(0).count(), 1);
        assert_eq!(
            catalog.fault_configurations(1).count(),
            0,
            "unparameterized structural catalogs should not yield zero-probability selected faults"
        );

        let mut parameterized = catalog.clone();
        parameterized.with_noise(&StochasticNoiseParams {
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.02,
            p_prep: 0.0,
        });
        assert_eq!(
            parameterized.fault_configurations(1).count(),
            1,
            "only the nonzero measurement fault location should be yielded"
        );
        assert_eq!(
            parameterized
                .fault_configurations(1)
                .next()
                .unwrap()
                .location_indices,
            vec![1]
        );
    }

    #[test]
    fn test_tracked_only_effect_stays_in_catalog_but_not_raw_mechanisms() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tracked_pauli_labeled("tracked_z0", PauliString::z(0));

        let mut catalog = FaultCatalog::from_circuit(&tc).unwrap();
        catalog.with_noise(&StochasticNoiseParams {
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.0,
        });

        let h_loc = catalog
            .locations
            .iter()
            .find(|loc| loc.channel == FaultChannel::P1)
            .unwrap();
        assert!(h_loc.faults.iter().any(|fault| {
            fault.affected_measurements.is_empty() && !fault.affected_tracked_paulis.is_empty()
        }));
        assert!(catalog.to_mechanisms().is_empty());
    }

    #[test]
    fn test_to_mechanisms_matches_old_build_fault_table() {
        // The key invariant: catalog.to_mechanisms() must produce the same
        // mechanisms as the old build_fault_table path for nonzero noise.
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[QubitId(0), QubitId(1)]);
        tc.tick().h(&[QubitId(0)]);
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().mz(&[QubitId(0), QubitId(1)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("2".to_string()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String("[]".to_string()),
        );
        tc.set_meta(
            "observables",
            pecos_quantum::Attribute::String("[]".to_string()),
        );

        let noise = StochasticNoiseParams {
            p1: 0.01,
            p2: 0.05,
            p_meas: 0.02,
            p_prep: 0.01,
        };

        // Old path (now a wrapper, but the output must match):
        let old_mechanisms = build_fault_table(&tc, &noise).unwrap();

        // New path:
        let mut catalog = FaultCatalog::from_circuit(&tc).unwrap();
        catalog.with_noise(&noise);
        let new_mechanisms = catalog.to_mechanisms();

        assert_eq!(
            old_mechanisms.len(),
            new_mechanisms.len(),
            "mechanism count must match"
        );
        for (old, new) in old_mechanisms.iter().zip(&new_mechanisms) {
            assert_close(old.probability, new.probability);
            assert_eq!(
                old.alternatives.len(),
                new.alternatives.len(),
                "alternative count must match"
            );
            for (old_alt, new_alt) in old.alternatives.iter().zip(&new.alternatives) {
                assert_eq!(old_alt, new_alt, "measurement effects must match");
            }
        }
    }

    #[test]
    fn test_catalog_meas_prep_probabilities() {
        // PZ(0) MZ(0): prep X fault goes directly to MZ (flips it)
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".to_string()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String("[]".to_string()),
        );
        tc.set_meta(
            "observables",
            pecos_quantum::Attribute::String("[]".to_string()),
        );

        let noise = StochasticNoiseParams {
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.007,
            p_prep: 0.003,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();

        let prep = catalog
            .locations
            .iter()
            .find(|l| l.faults.iter().any(|f| f.kind == FaultKind::PrepFlip));
        assert!(prep.is_some(), "Should have a prep fault location");
        let prep = prep.unwrap();
        assert!((prep.channel_probability - 0.003).abs() < 1e-10);
        assert!(prep.faults[0].pauli.is_none());

        let meas = catalog.locations.iter().find(|l| {
            l.faults
                .iter()
                .any(|f| f.kind == FaultKind::MeasurementFlip)
        });
        assert!(meas.is_some(), "Should have a measurement fault location");
        let meas = meas.unwrap();
        assert!((meas.channel_probability - 0.007).abs() < 1e-10);
        assert!(meas.faults[0].pauli.is_none());
    }

    #[test]
    fn test_catalog_separate_locations_same_detector_effect() {
        // Two H gates on same qubit → two separate locations
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".to_string()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String(r#"[{"records": [-1]}]"#.to_string()),
        );
        tc.set_meta(
            "observables",
            pecos_quantum::Attribute::String("[]".to_string()),
        );

        let noise = StochasticNoiseParams {
            p1: 0.01,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();

        // Both H gates → separate locations even if they have the same detector effect
        let h_locs: Vec<_> = catalog
            .locations
            .iter()
            .filter(|l| l.gate_type == GateType::H)
            .collect();
        assert_eq!(
            h_locs.len(),
            2,
            "Two H gates should produce two separate locations"
        );
    }

    #[test]
    fn test_catalog_full_configuration_probability() {
        // H(0) MZ(0) with p1=0.03, p_meas=0.01.
        // Two locations: H (3 alts) and MZ (1 alt).
        // Pick alt 0 at H, no fault at MZ:
        //   P = (0.03/3) * (1 - 0.01) = 0.01 * 0.99 = 0.0099
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".to_string()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String("[]".to_string()),
        );
        tc.set_meta(
            "observables",
            pecos_quantum::Attribute::String("[]".to_string()),
        );

        let noise = StochasticNoiseParams {
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.01,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();
        assert_eq!(catalog.locations.len(), 2); // H + MZ

        let h_loc = &catalog.locations[0]; // H
        let mz_loc = &catalog.locations[1]; // MZ

        // Pick first H alternative, no fault at MZ
        let alt_prob = h_loc.faults[0].absolute_probability; // 0.03/3 = 0.01
        let no_mz_prob = mz_loc.no_fault_probability; // 1 - 0.01 = 0.99
        let config_prob = alt_prob * no_mz_prob;

        assert!((config_prob - 0.0099).abs() < 1e-10);
    }

    // ---- fault_configurations iterator tests ----

    #[test]
    fn test_configurations_k0_one_no_fault_event() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".into()),
        );
        tc.set_meta("detectors", pecos_quantum::Attribute::String("[]".into()));
        tc.set_meta("observables", pecos_quantum::Attribute::String("[]".into()));

        let noise = StochasticNoiseParams {
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.01,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();

        let configs: Vec<_> = catalog.fault_configurations(0).collect();
        assert_eq!(configs.len(), 1);
        let c = &configs[0];
        assert!(c.location_indices.is_empty());
        assert!(c.alternative_indices.is_empty());
        assert!(c.affected_measurements.is_empty());
        assert!(c.affected_detectors.is_empty());
        assert_close(c.selected_probability, 1.0);
        // config_prob = product of all no_fault_probability
        let expected: f64 = catalog
            .locations
            .iter()
            .map(|l| l.no_fault_probability)
            .product();
        assert!((c.configuration_probability - expected).abs() < 1e-12);
    }

    #[test]
    fn test_configurations_k1_matches_single_fault() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".into()),
        );
        tc.set_meta("detectors", pecos_quantum::Attribute::String("[]".into()));
        tc.set_meta("observables", pecos_quantum::Attribute::String("[]".into()));

        let noise = StochasticNoiseParams {
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.01,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();

        let configs: Vec<_> = catalog.fault_configurations(1).collect();
        // Total k=1 configs = sum of num_alternatives across all locations
        let expected_count: usize = catalog.locations.iter().map(|l| l.num_alternatives).sum();
        assert_eq!(configs.len(), expected_count);

        // First config should match first location, first alternative
        let c = &configs[0];
        assert_eq!(c.location_indices, vec![0]);
        assert_eq!(c.alternative_indices, vec![0]);
        let alt = &catalog.locations[0].faults[0];
        assert_eq!(c.affected_measurements, alt.affected_measurements);
        assert!((c.selected_probability - alt.absolute_probability).abs() < 1e-12);
    }

    #[test]
    fn test_configurations_skip_zero_probability_structural_locations() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".into()),
        );
        tc.set_meta("detectors", pecos_quantum::Attribute::String("[]".into()));
        tc.set_meta("observables", pecos_quantum::Attribute::String("[]".into()));

        let noise = StochasticNoiseParams {
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();
        assert_eq!(catalog.locations.len(), 2);

        let h_idx = catalog
            .locations
            .iter()
            .position(|loc| loc.gate_type == GateType::H)
            .unwrap();
        let mz_idx = catalog
            .locations
            .iter()
            .position(|loc| loc.gate_type == GateType::MZ)
            .unwrap();
        assert_close(catalog.locations[mz_idx].channel_probability, 0.0);

        let configs: Vec<_> = catalog.fault_configurations(1).collect();
        assert_eq!(configs.len(), 3);
        assert!(configs.iter().all(|c| c.location_indices == vec![h_idx]));
        assert!(configs.iter().all(|c| c.selected_probability > 0.0));
        assert!(
            configs
                .iter()
                .all(|c| !c.location_indices.contains(&mz_idx))
        );
        assert_eq!(catalog.fault_configurations(2).count(), 0);
    }

    #[test]
    fn test_configurations_all_zero_noise_only_yields_k0() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".into()),
        );
        tc.set_meta("detectors", pecos_quantum::Attribute::String("[]".into()));
        tc.set_meta("observables", pecos_quantum::Attribute::String("[]".into()));

        let catalog = build_fault_catalog(
            &tc,
            &StochasticNoiseParams {
                p1: 0.0,
                p2: 0.0,
                p_meas: 0.0,
                p_prep: 0.0,
            },
        )
        .unwrap();

        let k0: Vec<_> = catalog.fault_configurations(0).collect();
        assert_eq!(k0.len(), 1);
        assert_close(k0[0].configuration_probability, 1.0);
        assert_eq!(catalog.fault_configurations(1).count(), 0);
        assert_eq!(catalog.fault_configurations(2).count(), 0);
    }

    #[test]
    fn test_configurations_include_nonzero_silent_faults() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("0".into()),
        );
        tc.set_meta("detectors", pecos_quantum::Attribute::String("[]".into()));
        tc.set_meta("observables", pecos_quantum::Attribute::String("[]".into()));

        let catalog = build_fault_catalog(
            &tc,
            &StochasticNoiseParams {
                p1: 0.03,
                p2: 0.0,
                p_meas: 0.0,
                p_prep: 0.0,
            },
        )
        .unwrap();

        let configs: Vec<_> = catalog.fault_configurations(1).collect();
        assert_eq!(configs.len(), 3);
        assert!(configs.iter().all(|c| c.affected_measurements.is_empty()));
        assert!(configs.iter().all(|c| c.affected_detectors.is_empty()));
        assert!(configs.iter().all(|c| c.affected_observables.is_empty()));
        assert!(configs.iter().all(|c| c.selected_probability > 0.0));
    }

    #[test]
    fn test_configurations_with_noise_zeroes_previous_channel() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".into()),
        );
        tc.set_meta("detectors", pecos_quantum::Attribute::String("[]".into()));
        tc.set_meta("observables", pecos_quantum::Attribute::String("[]".into()));

        let mut catalog = FaultCatalog::from_circuit(&tc).unwrap();
        catalog.with_noise(&StochasticNoiseParams {
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.01,
            p_prep: 0.0,
        });
        catalog.with_noise(&StochasticNoiseParams {
            p1: 0.0,
            p2: 0.0,
            p_meas: 0.02,
            p_prep: 0.0,
        });

        let mz_idx = catalog
            .locations
            .iter()
            .position(|loc| loc.gate_type == GateType::MZ)
            .unwrap();
        let configs: Vec<_> = catalog.fault_configurations(1).collect();
        assert_eq!(configs.len(), 1);
        assert_eq!(configs[0].location_indices, vec![mz_idx]);
        assert_close(configs[0].selected_probability, 0.02);
    }

    #[test]
    fn test_configurations_k2_xor_cancels_duplicate_effects() {
        // Two H gates both flipping measurement 0 → XOR cancels
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".into()),
        );
        tc.set_meta(
            "detectors",
            pecos_quantum::Attribute::String(r#"[{"records":[-1]}]"#.into()),
        );
        tc.set_meta("observables", pecos_quantum::Attribute::String("[]".into()));

        let noise = StochasticNoiseParams {
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.0,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();
        assert_eq!(catalog.locations.len(), 3); // two H locations + structural MZ location

        // Find a k=2 config where both locations fire with Z alternative (flips MZ)
        // Z after first H → X at second H → X at MZ → flips meas 0
        // Z after second H → Z at MZ → doesn't flip
        // So to get XOR cancel: need two alternatives that BOTH flip meas 0
        let configs: Vec<_> = catalog.fault_configurations(2).collect();
        // Check that some configs have empty affected_measurements (XOR cancel)
        let cancelled: Vec<_> = configs
            .iter()
            .filter(|c| c.affected_measurements.is_empty())
            .collect();
        assert!(!cancelled.is_empty(), "Some k=2 configs should XOR-cancel");
    }

    #[test]
    fn test_configurations_k2_probability_hand_calc() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".into()),
        );
        tc.set_meta("detectors", pecos_quantum::Attribute::String("[]".into()));
        tc.set_meta("observables", pecos_quantum::Attribute::String("[]".into()));

        let noise = StochasticNoiseParams {
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.01,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();
        // 2 locations: H (3 alts, p=0.03) and MZ (1 alt, p=0.01)

        let configs: Vec<_> = catalog.fault_configurations(2).collect();
        // k=2 means both locations fire
        // selected_probability = (0.03/3) * (0.01/1) = 0.01 * 0.01 = 0.0001
        // configuration_probability = selected * (no unselected) = 0.0001
        assert_eq!(configs.len(), 3); // 3 alternatives at H × 1 at MZ
        for c in &configs {
            assert!((c.selected_probability - 0.0001).abs() < 1e-12);
            assert!((c.configuration_probability - 0.0001).abs() < 1e-12);
        }
    }

    #[test]
    fn test_configurations_all_fault_weights_sum_to_one() {
        let mut tc = TickCircuit::new();
        tc.tick().pz(&[QubitId(0)]);
        tc.tick().h(&[QubitId(0)]);
        tc.tick().cx(&[(QubitId(0), QubitId(1))]);
        tc.tick().mz(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(1)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("2".into()),
        );
        tc.set_meta("detectors", pecos_quantum::Attribute::String("[]".into()));
        tc.set_meta("observables", pecos_quantum::Attribute::String("[]".into()));

        let noise = StochasticNoiseParams {
            p1: 0.01,
            p2: 0.05,
            p_meas: 0.02,
            p_prep: 0.01,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();

        let total: f64 = (0..=catalog.locations.len())
            .flat_map(|k| catalog.fault_configurations(k))
            .map(|c| c.configuration_probability)
            .sum();

        assert!(
            (total - 1.0).abs() < 1e-12,
            "all truncated-by-k configurations across k=0..N should sum to 1, got {total}"
        );
    }

    #[test]
    fn test_configurations_iterator_is_lazy() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.tick().mz(&[QubitId(0)]);
        tc.set_meta(
            "num_measurements",
            pecos_quantum::Attribute::String("1".into()),
        );
        tc.set_meta("detectors", pecos_quantum::Attribute::String("[]".into()));
        tc.set_meta("observables", pecos_quantum::Attribute::String("[]".into()));

        let noise = StochasticNoiseParams {
            p1: 0.03,
            p2: 0.0,
            p_meas: 0.01,
            p_prep: 0.0,
        };
        let catalog = build_fault_catalog(&tc, &noise).unwrap();

        // Take only first 2 items from k=1 iterator (doesn't allocate all)
        let first_two: Vec<_> = catalog.fault_configurations(1).take(2).collect();
        assert_eq!(first_two.len(), 2);
    }

    // ---- RawMeasurementPlan tests ----

    #[test]
    fn test_plan_bell_r_source_shared_by_copy() {
        // Bell: H(0) CX(0,1) MZ(0) MZ(1)
        // m0 = Random, m1 = Copy(m0). Both share the same r-source.
        // With zero noise, m0 == m1 for all shots.
        use pecos_simulators::SymbolicSparseStab;

        let mut sim = SymbolicSparseStab::new(2);
        sim.h(&[0]).cx(&[(0, 1)]);
        sim.mz(&[0]);
        sim.mz(&[1]);

        let plan = RawMeasurementPlan::new(sim.measurement_history(), vec![]);
        let result = plan.sample(1000, 42);

        for shot in 0..1000 {
            let m0 = result.get(shot, 0).0;
            let m1 = result.get(shot, 1).0;
            assert_eq!(m0, m1, "Bell pair: m0 must equal m1 (shot {shot})");
        }
    }

    #[test]
    fn test_plan_physical_fault_does_not_inherit_copy() {
        // Bell: m0 = Random, m1 = Copy(m0).
        // Add a physical fault that flips ONLY m0 with p=1.
        // Result: m0 is flipped, m1 is NOT — the fault does not propagate
        // through the ideal Copy dependency.
        use pecos_simulators::SymbolicSparseStab;

        let mut sim = SymbolicSparseStab::new(2);
        sim.h(&[0]).cx(&[(0, 1)]);
        sim.mz(&[0]);
        sim.mz(&[1]);

        // Fault that always fires, flipping only m0
        let mechanisms = vec![FaultMechanism {
            probability: 1.0,
            alternatives: vec![vec![0]],
        }];
        let plan = RawMeasurementPlan::new(sim.measurement_history(), mechanisms);
        let result = plan.sample(1000, 42);

        for shot in 0..1000 {
            let m0 = result.get(shot, 0).0;
            let m1 = result.get(shot, 1).0;
            // m0 = base XOR 1 (always flipped), m1 = base (not flipped)
            // Since base m0 == base m1, after flip: m0 != m1
            assert_ne!(m0, m1, "Fault on m0 must not inherit to m1 (shot {shot})");
        }
    }

    #[test]
    fn test_plan_grouped_alternatives_preserve_empty() {
        // Deterministic base (m0 = Fixed(false) = always 0) with a p=1 mechanism
        // having 3 alternatives: [flip m0, flip m0, no-op].
        // Each shot fires and picks one uniformly → 2/3 get flipped.
        use pecos_simulators::SymbolicSparseStab;

        let mut sim = SymbolicSparseStab::new(1);
        sim.mz(&[0]); // m0 = Fixed(false)

        let mechanisms = vec![FaultMechanism {
            probability: 1.0,
            alternatives: vec![vec![0], vec![0], vec![]],
        }];
        let plan = RawMeasurementPlan::new(sim.measurement_history(), mechanisms);
        let result = plan.sample(9000, 42);

        // base=0, fault flips with prob 2/3 → mean should be ~2/3.
        let ones: usize = (0..9000).filter(|&s| result.get(s, 0).0).count();
        let mean = f64::from(u32::try_from(ones).expect("sample count fits in u32")) / 9000.0;
        assert!(
            (mean - 2.0 / 3.0).abs() < 0.03,
            "Expected ~2/3 flip rate from grouped alternatives, got {mean:.4}"
        );
    }

    #[test]
    fn test_plan_geometric_sampling_firing_rates() {
        use pecos_simulators::SymbolicSparseStab;

        let mut sim = SymbolicSparseStab::new(1);
        sim.mz(&[0]); // deterministic base measurement: m0 = 0

        let shots = 200_000usize;
        for (p, low, high) in [
            (0.001, 120, 280),
            (0.05, 9400, 10600),
            (0.5, 99_000, 101_000),
        ] {
            let mechanisms = vec![FaultMechanism {
                probability: p,
                alternatives: vec![vec![0]],
            }];
            let plan = RawMeasurementPlan::new(sim.measurement_history(), mechanisms);
            let result = plan.sample(shots, 42);

            let firing_count = (0..shots).filter(|&shot| result.get(shot, 0).0).count();
            assert!(
                (low..=high).contains(&firing_count),
                "p={p} firing count {firing_count} outside expected range [{low}, {high}]"
            );
        }
    }

    #[test]
    fn test_sample_raw_word_boundaries_are_masked() {
        use pecos_simulators::SymbolicSparseStab;

        let mut sim = SymbolicSparseStab::new(1);
        sim.mz(&[0]); // deterministic base measurement: m0 = 0

        let mechanisms = vec![FaultMechanism {
            probability: 1.0,
            alternatives: vec![vec![0]],
        }];
        let plan = RawMeasurementPlan::new(sim.measurement_history(), mechanisms);

        for shots in [63usize, 64, 65, 128, 129] {
            let raw = plan.sample_raw(shots, 42);
            let expected_words = shots.div_ceil(64);
            assert_eq!(raw.columns[0].len(), expected_words);
            for shot in 0..shots {
                let word_idx = shot / 64;
                let bit_idx = shot % 64;
                assert_ne!(
                    raw.columns[0][word_idx] & (1u64 << bit_idx),
                    0,
                    "shot {shot} should be flipped for p=1"
                );
            }

            let remainder = shots % 64;
            if remainder != 0 {
                let tail_mask = !((1u64 << remainder) - 1);
                assert_eq!(
                    raw.columns[0].last().copied().unwrap() & tail_mask,
                    0,
                    "bits beyond {shots} shots should be masked off"
                );
            }
        }
    }

    #[test]
    fn test_sample_raw_masks_final_word_no_mechanisms() {
        // 100 shots (not a multiple of 64): final word should have bits 100..128 = 0
        use pecos_simulators::SymbolicSparseStab;

        let mut sim = SymbolicSparseStab::new(1);
        sim.h(&[0]);
        sim.mz(&[0]); // Random

        let plan = RawMeasurementPlan::new(sim.measurement_history(), vec![]);
        let raw = plan.sample_raw(100, 42);

        // 100 shots → 2 words. Last word should have bits 36..63 = 0 (100 - 64 = 36 valid bits)
        assert_eq!(raw.columns[0].len(), 2);
        let last_word = raw.columns[0][1];
        let valid_bits = 100 - 64;
        let tail_mask = !((1u64 << valid_bits) - 1);
        assert_eq!(
            last_word & tail_mask,
            0,
            "Bits beyond shots should be zero in measurement columns"
        );
    }

    #[test]
    fn test_sample_raw_r_columns_masked() {
        use pecos_simulators::SymbolicSparseStab;

        let mut sim = SymbolicSparseStab::new(1);
        sim.h(&[0]);
        sim.mz(&[0]); // Random

        let plan = RawMeasurementPlan::new(sim.measurement_history(), vec![]);
        let raw = plan.sample_raw(100, 42);

        assert_eq!(raw.r_columns.len(), 1);
        assert_eq!(raw.r_columns[0].len(), 2);
        let last_word = raw.r_columns[0][1];
        let valid_bits = 100 - 64;
        let tail_mask = !((1u64 << valid_bits) - 1);
        assert_eq!(
            last_word & tail_mask,
            0,
            "Bits beyond shots should be zero in r_columns"
        );
    }

    #[test]
    fn test_sample_raw_bell_r_source_mapping() {
        // Bell: H(0) CX(0,1) MZ(0) MZ(1)
        // m0=Random, m1=Copy(m0) → exactly one r-source at measurement 0
        use pecos_simulators::SymbolicSparseStab;

        let mut sim = SymbolicSparseStab::new(2);
        sim.h(&[0]).cx(&[(0, 1)]);
        sim.mz(&[0]);
        sim.mz(&[1]);

        let plan = RawMeasurementPlan::new(sim.measurement_history(), vec![]);
        let raw = plan.sample_raw(64, 42);

        assert_eq!(raw.r_columns.len(), 1, "Bell pair has one r-source");
        assert_eq!(
            raw.r_source_measurements,
            vec![0],
            "r-source introduced at measurement 0"
        );
        // The r column should equal the m0 column (since m0 = Random = r0 directly)
        assert_eq!(raw.r_columns[0], raw.columns[0]);
        // And m1 = Copy(m0), so columns[1] == columns[0]
        assert_eq!(raw.columns[0], raw.columns[1]);
    }

    #[test]
    fn test_sample_raw_zero_shots_invariant() {
        // Bell circuit with zero shots: r_columns length must match r_source_measurements
        use pecos_simulators::SymbolicSparseStab;

        let mut sim = SymbolicSparseStab::new(2);
        sim.h(&[0]).cx(&[(0, 1)]);
        sim.mz(&[0]);
        sim.mz(&[1]);

        let plan = RawMeasurementPlan::new(sim.measurement_history(), vec![]);
        let raw = plan.sample_raw(0, 42);

        assert_eq!(raw.columns.len(), 2);
        assert!(raw.columns[0].is_empty());
        assert!(raw.columns[1].is_empty());
        assert_eq!(raw.r_source_measurements, vec![0]);
        assert_eq!(raw.r_columns.len(), 1);
        assert!(raw.r_columns[0].is_empty());
        assert_eq!(raw.shots, 0);
    }
}
