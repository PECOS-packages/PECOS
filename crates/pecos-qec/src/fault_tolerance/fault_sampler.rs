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
use pecos_simulators::CliffordGateable;
use pecos_simulators::measurement_sampler::{MeasurementKind, SampleResult};
use pecos_simulators::pauli_prop::PauliProp;
use pecos_simulators::symbolic_sparse_stab::MeasurementHistory;
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
             plus metadata (MeasCrosstalk*, PauliOperatorMeta).",
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
            | GateType::PauliOperatorMeta
    )
}

/// A fault mechanism: fires with probability `p`, then uniformly selects one
/// of its alternatives to determine which measurements are flipped.
///
/// For a depolarizing channel with k non-identity Paulis and total error
/// probability p: the mechanism fires with probability p, then each of the
/// k alternatives is chosen with probability 1/k. This matches the stabilizer
/// sim's "exactly one Pauli error per gate event" semantics.
#[derive(Clone, Debug)]
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
    pub(crate) gate_type: GateType,
    pub(crate) qubits: Vec<usize>,
}

/// Single-qubit Pauli type for fault injection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PauliType {
    X,
    Y,
    Z,
}

/// Build a fault table from a TickCircuit and noise parameters.
///
/// Each entry describes one possible fault mechanism: its probability and
/// which measurements it would flip if it occurs. The table is used for
/// independent per-shot Bernoulli sampling.
///
/// Gate ordering follows the TickCircuit tick-by-tick structure, which must
/// match the measurement numbering used by detector/DEM-output record indices.
///
/// # Supported gates
///
/// **Fault injection** (noise applied after these gates):
/// - Single-qubit Clifford: H, X, Y, Z, SZ, SZdg, SX, SXdg, SY, SYdg, F, Fdg → p=p1, 3 alternatives
/// - Two-qubit Clifford: CX, CY, CZ, SXX, SXXdg, SYY, SYYdg, SZZ, SZZdg, SWAP → p=p2, 15 alts
/// - State preparation: PZ, QAlloc → mechanism with p=p_prep, 1 alternative (X)
/// - Measurement: MZ, MeasureFree, MeasureLeaked → mechanism with p=p_meas, 1 alternative (flip)
///
/// Each mechanism fires at most once per shot (Bernoulli with total probability p).
/// When it fires, exactly one alternative is chosen uniformly at random. This
/// matches the depolarizing channel semantics: "with probability p, apply one
/// of the k non-identity Paulis, each equally likely."
///
/// **Propagation** (gates that transform a propagating Pauli):
/// - All single-qubit Cliffords: Clifford conjugation via direct Pauli-basis updates
/// - All two-qubit Cliffords: Clifford conjugation via direct Pauli-basis updates
/// - PZ, QAlloc: absorbs all Pauli components on the reset qubit
/// - MZ: records X-component flip, then absorbs all components (state collapse)
///
/// **No-op** (pass through without noise or transformation):
/// - I, Idle, QFree, MeasCrosstalkGlobalPayload, MeasCrosstalkLocalPayload, PauliOperatorMeta
///
/// Any gate not in the above lists returns [`UnsupportedGateError`].
///
pub fn build_fault_table(
    tc: &TickCircuit,
    noise: &StochasticNoiseParams,
) -> Result<Vec<FaultMechanism>, UnsupportedGateError> {
    validate_tick_circuit(tc)?;
    let (gates, meas_positions) = flatten_tick_circuit(tc);

    if noise.p1 == 0.0 && noise.p2 == 0.0 && noise.p_meas == 0.0 && noise.p_prep == 0.0 {
        return Ok(Vec::new());
    }
    let mut mechanisms = Vec::new();

    for (loc_idx, loc) in gates.iter().enumerate() {
        match loc.gate_type {
            // Single-qubit Clifford: one mechanism with 3 alternatives (X/Y/Z)
            gate_type
                if is_standard_1q_clifford_gate(gate_type)
                    && noise.p1 > 0.0
                    && !loc.qubits.is_empty() =>
            {
                let q = loc.qubits[0];
                let alts: Vec<Vec<usize>> = [PauliType::X, PauliType::Y, PauliType::Z]
                    .iter()
                    .map(|&p| {
                        propagate_single(p, q, loc_idx + 1, &gates, &meas_positions)
                            .into_iter()
                            .collect()
                    })
                    .collect();
                // Only include if at least one alternative has an effect
                if alts.iter().any(|a| !a.is_empty()) {
                    mechanisms.push(FaultMechanism {
                        probability: noise.p1,
                        alternatives: alts,
                    });
                }
            }

            // Two-qubit Clifford: one mechanism with 15 alternatives
            gate_type
                if is_standard_2q_clifford_gate(gate_type)
                    && noise.p2 > 0.0
                    && loc.qubits.len() >= 2 =>
            {
                let (q1, q2) = (loc.qubits[0], loc.qubits[1]);
                let paulis = [PauliType::X, PauliType::Y, PauliType::Z];
                let mut alts: Vec<Vec<usize>> = Vec::new();

                // 9 two-qubit pairs
                for &p1 in &paulis {
                    for &p2 in &paulis {
                        let a: Vec<usize> =
                            propagate_pair(p1, q1, p2, q2, loc_idx + 1, &gates, &meas_positions)
                                .into_iter()
                                .collect();
                        alts.push(a);
                    }
                }
                // 6 single-qubit (PI and IP)
                for &p in &paulis {
                    let a: Vec<usize> =
                        propagate_single(p, q1, loc_idx + 1, &gates, &meas_positions)
                            .into_iter()
                            .collect();
                    alts.push(a);
                    let a: Vec<usize> =
                        propagate_single(p, q2, loc_idx + 1, &gates, &meas_positions)
                            .into_iter()
                            .collect();
                    alts.push(a);
                }
                if alts.iter().any(|a| !a.is_empty()) {
                    mechanisms.push(FaultMechanism {
                        probability: noise.p2,
                        alternatives: alts,
                    });
                }
            }

            // State preparation: single alternative (X error)
            GateType::PZ | GateType::QAlloc if noise.p_prep > 0.0 && !loc.qubits.is_empty() => {
                let q = loc.qubits[0];
                let a: Vec<usize> =
                    propagate_single(PauliType::X, q, loc_idx + 1, &gates, &meas_positions)
                        .into_iter()
                        .collect();
                if !a.is_empty() {
                    mechanisms.push(FaultMechanism {
                        probability: noise.p_prep,
                        alternatives: vec![a],
                    });
                }
            }

            // Measurement fault: single alternative (flip this measurement)
            GateType::MZ | GateType::MeasureFree | GateType::MeasureLeaked
                if noise.p_meas > 0.0 =>
            {
                if let Some(&meas_idx) = meas_positions.get(&loc_idx) {
                    mechanisms.push(FaultMechanism {
                        probability: noise.p_meas,
                        alternatives: vec![vec![meas_idx]],
                    });
                }
            }

            _ => {}
        }
    }

    Ok(mechanisms)
}

/// Validate that all gates in the TickCircuit are supported (before flattening).
fn validate_tick_circuit(tc: &TickCircuit) -> Result<(), UnsupportedGateError> {
    for (tick_idx, tick) in tc.ticks().iter().enumerate() {
        for (gate_idx, gate) in tick.gates().iter().enumerate() {
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
                gate_in_tick: gate_idx,
                qubits: gate.qubits.iter().map(|q| q.index()).collect(),
            });
        }
    }
    Ok(())
}

/// Flatten a TickCircuit into a gate list with measurement position tracking.
///
/// Multi-qubit gates are split into individual entries so each measurement/pair
/// gets its own position for fault injection. Returns the gate list and a map
/// from gate-list index to measurement index.
pub(crate) fn flatten_tick_circuit(tc: &TickCircuit) -> (Vec<GateLoc>, HashMap<usize, usize>) {
    let mut gates = Vec::new();
    let mut meas_positions = HashMap::new();
    let mut meas_count = 0usize;

    for tick in tc.ticks() {
        for gate in tick.gates() {
            let qs: Vec<usize> = gate.qubits.iter().map(|q| q.index()).collect();
            let is_mz = is_supported_measurement_gate(gate.gate_type);
            let is_2q = is_standard_2q_clifford_gate(gate.gate_type);

            if is_mz && qs.len() > 1 {
                for &q in &qs {
                    meas_positions.insert(gates.len(), meas_count);
                    meas_count += 1;
                    gates.push(GateLoc {
                        gate_type: gate.gate_type,
                        qubits: vec![q],
                    });
                }
            } else if is_2q && qs.len() > 2 {
                for pair in qs.chunks(2).filter(|c| c.len() == 2) {
                    gates.push(GateLoc {
                        gate_type: gate.gate_type,
                        qubits: vec![pair[0], pair[1]],
                    });
                }
            } else if qs.len() > 1 && !is_2q && !is_mz {
                for &q in &qs {
                    gates.push(GateLoc {
                        gate_type: gate.gate_type,
                        qubits: vec![q],
                    });
                }
            } else {
                if is_mz {
                    meas_positions.insert(gates.len(), meas_count);
                    meas_count += 1;
                }
                gates.push(GateLoc {
                    gate_type: gate.gate_type,
                    qubits: qs,
                });
            }
        }
    }

    (gates, meas_positions)
}

/// Propagate a single-qubit Pauli fault forward through the gate list.
///
/// Returns the set of measurement indices whose outcomes would be flipped
/// by this Pauli error at this position.
pub(crate) fn propagate_single(
    pauli: PauliType,
    qubit: usize,
    start: usize,
    gates: &[GateLoc],
    meas_positions: &HashMap<usize, usize>,
) -> BTreeSet<usize> {
    let mut prop = PauliProp::new();
    match pauli {
        PauliType::X => prop.track_x(&[qubit]),
        PauliType::Y => prop.track_y(&[qubit]),
        PauliType::Z => prop.track_z(&[qubit]),
    };

    propagate_forward(&mut prop, start, gates, meas_positions)
}

fn propagate_single_effect(
    pauli: PauliType,
    qubit: usize,
    start: usize,
    gates: &[GateLoc],
    meas_positions: &HashMap<usize, usize>,
    tracked_ops: &[PauliString],
) -> PropagatedFaultEffect {
    let mut prop = PauliProp::new();
    match pauli {
        PauliType::X => prop.track_x(&[qubit]),
        PauliType::Y => prop.track_y(&[qubit]),
        PauliType::Z => prop.track_z(&[qubit]),
    };

    let affected_measurements = propagate_forward(&mut prop, start, gates, meas_positions);
    let affected_tracked_ops = tracked_ops_flipped_by(&prop, tracked_ops);
    PropagatedFaultEffect {
        affected_measurements,
        affected_tracked_ops,
    }
}

/// Propagate a two-qubit Pauli fault forward through the gate list.
fn propagate_pair(
    p1: PauliType,
    q1: usize,
    p2: PauliType,
    q2: usize,
    start: usize,
    gates: &[GateLoc],
    meas_positions: &HashMap<usize, usize>,
) -> BTreeSet<usize> {
    let mut prop = PauliProp::new();
    match p1 {
        PauliType::X => prop.track_x(&[q1]),
        PauliType::Y => prop.track_y(&[q1]),
        PauliType::Z => prop.track_z(&[q1]),
    };
    match p2 {
        PauliType::X => prop.track_x(&[q2]),
        PauliType::Y => prop.track_y(&[q2]),
        PauliType::Z => prop.track_z(&[q2]),
    };

    propagate_forward(&mut prop, start, gates, meas_positions)
}

fn propagate_pair_effect(
    p1: PauliType,
    q1: usize,
    p2: PauliType,
    q2: usize,
    start: usize,
    gates: &[GateLoc],
    meas_positions: &HashMap<usize, usize>,
    tracked_ops: &[PauliString],
) -> PropagatedFaultEffect {
    let mut prop = PauliProp::new();
    match p1 {
        PauliType::X => prop.track_x(&[q1]),
        PauliType::Y => prop.track_y(&[q1]),
        PauliType::Z => prop.track_z(&[q1]),
    };
    match p2 {
        PauliType::X => prop.track_x(&[q2]),
        PauliType::Y => prop.track_y(&[q2]),
        PauliType::Z => prop.track_z(&[q2]),
    };

    let affected_measurements = propagate_forward(&mut prop, start, gates, meas_positions);
    let affected_tracked_ops = tracked_ops_flipped_by(&prop, tracked_ops);
    PropagatedFaultEffect {
        affected_measurements,
        affected_tracked_ops,
    }
}

struct PropagatedFaultEffect {
    affected_measurements: BTreeSet<usize>,
    affected_tracked_ops: Vec<usize>,
}

/// Core forward propagation: evolve a Pauli through gates, collecting affected measurements.
fn propagate_forward(
    prop: &mut PauliProp,
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
                if prop.contains_x(q) {
                    if let Some(&meas_idx) = meas_positions.get(&loc_idx) {
                        affected.insert(meas_idx);
                    }
                }
                prop.clear_qubit(q);
            }
            _ => {}
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
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum FaultChannel {
    /// Single-qubit depolarizing (p1).
    P1,
    /// Two-qubit depolarizing (p2).
    P2,
    /// Measurement flip (p_meas).
    PMeas,
    /// State preparation flip (p_prep).
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
    /// Tracked-operator indices flipped.
    pub affected_tracked_ops: Vec<usize>,
    /// Probability of this alternative conditioned on the mechanism firing (1/k).
    pub conditional_probability: f64,
    /// Marginal probability of this specific alternative at this location: p_i / k_i.
    ///
    /// This is NOT "probability of this fault and no others." A full-circuit
    /// configuration probability requires multiplying by (1 - p_j) for all
    /// other locations j.
    pub absolute_probability: f64,
}

/// A physical fault location in the circuit.
#[derive(Clone, Debug)]
pub struct FaultLocation {
    /// Tick index in the TickCircuit.
    pub tick: usize,
    /// Gate index within the tick.
    pub gate_index: usize,
    /// Gate type at this location.
    pub gate_type: GateType,
    /// Qubits involved.
    pub qubits: Vec<usize>,
    /// Which noise channel this location belongs to.
    pub channel: FaultChannel,
    /// Total probability that this mechanism fires: p_i.
    pub channel_probability: f64,
    /// Probability that no fault occurs at this location: 1 - p_i.
    pub no_fault_probability: f64,
    /// Number of fault alternatives at this location: k_i.
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
/// For location i with k_i alternatives:
/// - `channel_probability` = p_i (total probability mechanism fires)
/// - `no_fault_probability` = 1 - p_i
/// - `conditional_probability` = 1/k_i (uniform alternative choice)
/// - `absolute_probability` = p_i / k_i (marginal alternative probability)
///
/// Full-circuit configuration probability for "alternative j at location i,
/// no fault at all other locations":
///   P = (p_i / k_i) * product_{m != i} (1 - p_m)
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
    /// Combined tracked-operator indices (XOR parity).
    pub affected_tracked_ops: Vec<usize>,
    /// Product of selected alternatives' absolute_probability.
    pub selected_probability: f64,
    /// selected_probability * product of unselected locations' no_fault_probability.
    pub configuration_probability: f64,
}

impl FaultCatalog {
    /// Lazily iterate all k-fault configurations.
    ///
    /// Each yielded `FaultConfiguration` represents exactly k distinct locations
    /// firing, with one alternative chosen per location. Effects are combined by
    /// XOR parity. Probabilities follow the independent-mechanism model.
    ///
    /// For k=0: yields one no-fault event.
    pub fn fault_configurations(&self, k: usize) -> FaultConfigurationIter<'_> {
        FaultConfigurationIter::new(self, k)
    }
}

/// Internal cursor for k-fault configuration iteration.
///
/// Holds the combination/alternative state machine. Shared by both
/// `FaultConfigurationIter` (borrowed) and `OwnedFaultConfigIter` (owned).
struct FaultConfigCursor {
    k: usize,
    combo: Vec<usize>,
    alt_indices: Vec<usize>,
    alt_counts: Vec<usize>,
    started: bool,
    done: bool,
}

impl FaultConfigCursor {
    fn new(num_locations: usize, k: usize, alt_count_fn: impl Fn(usize) -> usize) -> Self {
        if k == 0 || k > num_locations {
            return Self {
                k,
                combo: Vec::new(),
                alt_indices: Vec::new(),
                alt_counts: Vec::new(),
                started: false,
                done: k > num_locations && k > 0,
            };
        }
        let combo: Vec<usize> = (0..k).collect();
        let alt_counts: Vec<usize> = combo.iter().map(|&i| alt_count_fn(i)).collect();
        let alt_indices = vec![0usize; k];
        Self {
            k,
            combo,
            alt_indices,
            alt_counts,
            started: false,
            done: false,
        }
    }

    /// Advance to the next state. Returns true if a new valid state exists.
    fn advance(&mut self, num_locations: usize, alt_count_fn: impl Fn(usize) -> usize) -> bool {
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
            if self.combo[i] <= num_locations - self.k + i {
                for j in (i + 1)..self.k {
                    self.combo[j] = self.combo[j - 1] + 1;
                }
                for j in 0..self.k {
                    self.alt_counts[j] = alt_count_fn(self.combo[j]);
                    self.alt_indices[j] = 0;
                }
                return true;
            }
        }
        false
    }

    /// Build a FaultConfiguration from the current cursor state + catalog data.
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
                affected_tracked_ops: Vec::new(),
                selected_probability: 1.0,
                configuration_probability: no_fault_prob,
            };
        }

        let mut meas_set = std::collections::BTreeSet::new();
        let mut det_set = std::collections::BTreeSet::new();
        let mut obs_set = std::collections::BTreeSet::new();
        let mut tracked_op_set = std::collections::BTreeSet::new();
        let mut selected_prob = 1.0;

        for i in 0..self.k {
            let loc = &catalog.locations[self.combo[i]];
            let alt = &loc.faults[self.alt_indices[i]];
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
            for &op in &alt.affected_tracked_ops {
                if !tracked_op_set.remove(&op) {
                    tracked_op_set.insert(op);
                }
            }
        }

        let selected_set: std::collections::BTreeSet<usize> = self.combo.iter().copied().collect();
        let unselected_no_fault: f64 = catalog
            .locations
            .iter()
            .enumerate()
            .filter(|(i, _)| !selected_set.contains(i))
            .map(|(_, loc)| loc.no_fault_probability)
            .product();

        FaultConfiguration {
            location_indices: self.combo.clone(),
            alternative_indices: self.alt_indices.clone(),
            affected_measurements: meas_set.into_iter().collect(),
            affected_detectors: det_set.into_iter().collect(),
            affected_observables: obs_set.into_iter().collect(),
            affected_tracked_ops: tracked_op_set.into_iter().collect(),
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
        let n = catalog.locations.len();
        if self.advance(n, |i| catalog.locations[i].faults.len()) {
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
        let cursor = FaultConfigCursor::new(catalog.locations.len(), k, |i| {
            catalog.locations[i].faults.len()
        });
        Self { catalog, cursor }
    }
}

impl<'a> Iterator for FaultConfigurationIter<'a> {
    type Item = FaultConfiguration;
    fn next(&mut self) -> Option<Self::Item> {
        self.cursor.next_config(self.catalog)
    }
}

/// Owned k-fault configuration iterator (no lifetime borrows).
/// Suitable for FFI / PyO3 where lifetimes are not expressible.
pub struct OwnedFaultConfigIter {
    catalog: FaultCatalog,
    cursor: FaultConfigCursor,
}

impl OwnedFaultConfigIter {
    /// Create from an owned catalog clone.
    pub fn new(catalog: FaultCatalog, k: usize) -> Self {
        let cursor = FaultConfigCursor::new(catalog.locations.len(), k, |i| {
            catalog.locations[i].faults.len()
        });
        Self { catalog, cursor }
    }
}

impl Iterator for OwnedFaultConfigIter {
    type Item = FaultConfiguration;
    fn next(&mut self) -> Option<Self::Item> {
        self.cursor.next_config(&self.catalog)
    }
}

/// Build a fault catalog from a TickCircuit and noise parameters.
///
/// Returns per-location, per-alternative fault data including Pauli labels,
/// affected detectors, observables, tracked operators, and probability fields.
///
/// Reads detector/observable metadata and tracked-operator annotations
/// from the circuit when present.
pub fn build_fault_catalog(
    tc: &TickCircuit,
    noise: &StochasticNoiseParams,
) -> Result<FaultCatalog, UnsupportedGateError> {
    validate_tick_circuit(tc)?;
    let (gates, meas_positions) = flatten_tick_circuit(tc);

    // Parse detector/DEM-output records for measurement→detector/op mapping
    let det_records = parse_detector_records(tc);
    let obs_records = parse_observable_records(tc);
    let tracked_op_annotations = parse_tracked_operator_annotations(tc);
    let num_meas = tc
        .get_meta("num_measurements")
        .and_then(|a| {
            if let pecos_quantum::Attribute::String(s) = a {
                s.parse::<usize>().ok()
            } else {
                None
            }
        })
        .unwrap_or_else(|| meas_positions.len());

    let mut locations = Vec::new();

    // Track original tick/gate indices through the flattened gate list
    let mut tick_gate_map: Vec<(usize, usize)> = Vec::new();
    for (tick_idx, tick) in tc.ticks().iter().enumerate() {
        for (gate_idx, _gate) in tick.gates().iter().enumerate() {
            tick_gate_map.push((tick_idx, gate_idx));
        }
    }

    // Re-walk the flattened gate list (same order as build_fault_table)
    // but record location metadata and Pauli labels
    let mut flat_idx_to_tick_gate: Vec<(usize, usize, GateType, Vec<usize>)> = Vec::new();
    {
        let mut orig_idx = 0;
        for tick in tc.ticks() {
            for gate in tick.gates() {
                let qs: Vec<usize> = gate.qubits.iter().map(|q| q.index()).collect();
                let is_mz = is_supported_measurement_gate(gate.gate_type);
                let is_2q = is_standard_2q_clifford_gate(gate.gate_type);
                let (tick_idx, gate_idx) = tick_gate_map[orig_idx];

                if is_mz && qs.len() > 1 {
                    for &q in &qs {
                        flat_idx_to_tick_gate.push((tick_idx, gate_idx, gate.gate_type, vec![q]));
                    }
                } else if is_2q && qs.len() > 2 {
                    for pair in qs.chunks(2).filter(|c| c.len() == 2) {
                        flat_idx_to_tick_gate.push((
                            tick_idx,
                            gate_idx,
                            gate.gate_type,
                            vec![pair[0], pair[1]],
                        ));
                    }
                } else if qs.len() > 1 && !is_2q && !is_mz {
                    for &q in &qs {
                        flat_idx_to_tick_gate.push((tick_idx, gate_idx, gate.gate_type, vec![q]));
                    }
                } else {
                    flat_idx_to_tick_gate.push((tick_idx, gate_idx, gate.gate_type, qs));
                }
                orig_idx += 1;
            }
        }
    }

    let pauli_types = [PauliType::X, PauliType::Y, PauliType::Z];

    for (loc_idx, loc) in gates.iter().enumerate() {
        let (tick_idx, gate_idx, gate_type, ref qubits) = flat_idx_to_tick_gate[loc_idx];

        match loc.gate_type {
            gate_type
                if is_standard_1q_clifford_gate(gate_type)
                    && noise.p1 > 0.0
                    && !loc.qubits.is_empty() =>
            {
                let q = loc.qubits[0];
                let num_alts = 3;
                let mut faults = Vec::with_capacity(num_alts);
                for &pt in &pauli_types {
                    let effect = propagate_single_effect(
                        pt,
                        q,
                        loc_idx + 1,
                        &gates,
                        &meas_positions,
                        &tracked_op_annotations,
                    );
                    let pauli = pauli_type_to_string(pt, q);
                    let (affected, dets, obs, tracked) =
                        catalog_effect_parts(effect, &det_records, &obs_records, num_meas);
                    faults.push(FaultAlternative {
                        kind: FaultKind::Pauli,
                        pauli: Some(pauli),
                        affected_measurements: affected,
                        affected_detectors: dets,
                        affected_observables: obs,
                        affected_tracked_ops: tracked,
                        conditional_probability: 1.0 / num_alts as f64,
                        absolute_probability: noise.p1 / num_alts as f64,
                    });
                }
                // Include all locations with nonzero channel probability (even no-effect ones)
                let num_alts = faults.len();
                locations.push(FaultLocation {
                    tick: tick_idx,
                    gate_index: gate_idx,
                    gate_type,
                    qubits: qubits.clone(),
                    channel: FaultChannel::P1,
                    channel_probability: noise.p1,
                    no_fault_probability: 1.0 - noise.p1,
                    num_alternatives: num_alts,
                    faults,
                });
            }

            gate_type
                if is_standard_2q_clifford_gate(gate_type)
                    && noise.p2 > 0.0
                    && loc.qubits.len() >= 2 =>
            {
                let (q1, q2) = (loc.qubits[0], loc.qubits[1]);
                let num_alts = 15;
                let mut faults = Vec::with_capacity(num_alts);

                // 9 two-qubit pairs
                for &p1 in &pauli_types {
                    for &p2 in &pauli_types {
                        let effect = propagate_pair_effect(
                            p1,
                            q1,
                            p2,
                            q2,
                            loc_idx + 1,
                            &gates,
                            &meas_positions,
                            &tracked_op_annotations,
                        );
                        let pauli = pauli_pair_to_string(p1, q1, p2, q2);
                        let (affected, dets, obs, tracked) =
                            catalog_effect_parts(effect, &det_records, &obs_records, num_meas);
                        faults.push(FaultAlternative {
                            kind: FaultKind::Pauli,
                            pauli: Some(pauli),
                            affected_measurements: affected,
                            affected_detectors: dets,
                            affected_observables: obs,
                            affected_tracked_ops: tracked,
                            conditional_probability: 1.0 / num_alts as f64,
                            absolute_probability: noise.p2 / num_alts as f64,
                        });
                    }
                }
                // 6 single-qubit (PI and IP)
                for &p in &pauli_types {
                    let effect = propagate_single_effect(
                        p,
                        q1,
                        loc_idx + 1,
                        &gates,
                        &meas_positions,
                        &tracked_op_annotations,
                    );
                    let pauli = pauli_type_to_string(p, q1);
                    let (affected, dets, obs, tracked) =
                        catalog_effect_parts(effect, &det_records, &obs_records, num_meas);
                    faults.push(FaultAlternative {
                        kind: FaultKind::Pauli,
                        pauli: Some(pauli),
                        affected_measurements: affected,
                        affected_detectors: dets,
                        affected_observables: obs,
                        affected_tracked_ops: tracked,
                        conditional_probability: 1.0 / num_alts as f64,
                        absolute_probability: noise.p2 / num_alts as f64,
                    });

                    let effect = propagate_single_effect(
                        p,
                        q2,
                        loc_idx + 1,
                        &gates,
                        &meas_positions,
                        &tracked_op_annotations,
                    );
                    let pauli = pauli_type_to_string(p, q2);
                    let (affected, dets, obs, tracked) =
                        catalog_effect_parts(effect, &det_records, &obs_records, num_meas);
                    faults.push(FaultAlternative {
                        kind: FaultKind::Pauli,
                        pauli: Some(pauli),
                        affected_measurements: affected,
                        affected_detectors: dets,
                        affected_observables: obs,
                        affected_tracked_ops: tracked,
                        conditional_probability: 1.0 / num_alts as f64,
                        absolute_probability: noise.p2 / num_alts as f64,
                    });
                }
                let n_alts = faults.len();
                locations.push(FaultLocation {
                    tick: tick_idx,
                    gate_index: gate_idx,
                    gate_type,
                    qubits: qubits.clone(),
                    channel: FaultChannel::P2,
                    channel_probability: noise.p2,
                    no_fault_probability: 1.0 - noise.p2,
                    num_alternatives: n_alts,
                    faults,
                });
            }

            GateType::PZ | GateType::QAlloc if noise.p_prep > 0.0 && !loc.qubits.is_empty() => {
                let q = loc.qubits[0];
                let effect = propagate_single_effect(
                    PauliType::X,
                    q,
                    loc_idx + 1,
                    &gates,
                    &meas_positions,
                    &tracked_op_annotations,
                );
                let (affected, dets, obs, tracked) =
                    catalog_effect_parts(effect, &det_records, &obs_records, num_meas);
                locations.push(FaultLocation {
                    tick: tick_idx,
                    gate_index: gate_idx,
                    gate_type,
                    qubits: qubits.clone(),
                    channel: FaultChannel::PPrep,
                    channel_probability: noise.p_prep,
                    no_fault_probability: 1.0 - noise.p_prep,
                    num_alternatives: 1,
                    faults: vec![FaultAlternative {
                        kind: FaultKind::PrepFlip,
                        pauli: None,
                        affected_measurements: affected,
                        affected_detectors: dets,
                        affected_observables: obs,
                        affected_tracked_ops: tracked,
                        conditional_probability: 1.0,
                        absolute_probability: noise.p_prep,
                    }],
                });
            }

            GateType::MZ | GateType::MeasureFree | GateType::MeasureLeaked
                if noise.p_meas > 0.0 =>
            {
                if let Some(&meas_idx) = meas_positions.get(&loc_idx) {
                    let affected = vec![meas_idx];
                    let dets = measurements_to_detectors(&affected, &det_records, num_meas);
                    let obs = measurements_to_observables(&affected, &obs_records, num_meas);
                    locations.push(FaultLocation {
                        tick: tick_idx,
                        gate_index: gate_idx,
                        gate_type,
                        qubits: qubits.clone(),
                        channel: FaultChannel::PMeas,
                        channel_probability: noise.p_meas,
                        no_fault_probability: 1.0 - noise.p_meas,
                        num_alternatives: 1,
                        faults: vec![FaultAlternative {
                            kind: FaultKind::MeasurementFlip,
                            pauli: None,
                            affected_measurements: affected,
                            affected_detectors: dets,
                            affected_observables: obs,
                            affected_tracked_ops: Vec::new(),
                            conditional_probability: 1.0,
                            absolute_probability: noise.p_meas,
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
    let json = match tc.get_meta(key) {
        Some(pecos_quantum::Attribute::String(s)) => s,
        _ => return Vec::new(),
    };
    parse_records_array_list(&json)
}

fn parse_detector_records(tc: &TickCircuit) -> Vec<Vec<i32>> {
    parse_records_from_meta(tc, "detectors")
}

fn parse_observable_records(tc: &TickCircuit) -> Vec<Vec<i32>> {
    parse_records_from_meta(tc, "observables")
}

fn parse_tracked_operator_annotations(tc: &TickCircuit) -> Vec<PauliString> {
    tc.annotations()
        .iter()
        .filter(|ann| matches!(ann.kind, AnnotationKind::Operator))
        .map(|ann| {
            let mut pauli = ann.pauli.clone();
            pauli.set_phase(pecos_core::QuarterPhase::PlusOne);
            pauli
        })
        .collect()
}

fn tracked_ops_flipped_by(prop: &PauliProp, tracked_ops: &[PauliString]) -> Vec<usize> {
    tracked_ops
        .iter()
        .enumerate()
        .filter_map(|(idx, tracked_op)| {
            let mut parity = false;
            for &(pauli, qubit) in tracked_op.paulis() {
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

/// Simple parser for `[{"records": [...]}, ...]` JSON without serde_json.
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

/// Map measurement effects to detector effects via record XOR.
fn measurements_to_detectors(
    affected_meas: &[usize],
    det_records: &[Vec<i32>],
    num_meas: usize,
) -> Vec<usize> {
    let mut fired = Vec::new();
    for (det_idx, records) in det_records.iter().enumerate() {
        let mut parity = 0u8;
        for &rec in records {
            let abs_idx = (num_meas as i32 + rec) as usize;
            if affected_meas.contains(&abs_idx) {
                parity ^= 1;
            }
        }
        if parity != 0 {
            fired.push(det_idx);
        }
    }
    fired
}

/// Map measurement effects to observable effects via record XOR.
fn measurements_to_observables(
    affected_meas: &[usize],
    obs_records: &[Vec<i32>],
    num_meas: usize,
) -> Vec<usize> {
    let mut fired = Vec::new();
    for (obs_idx, records) in obs_records.iter().enumerate() {
        let mut parity = 0u8;
        for &rec in records {
            let abs_idx = (num_meas as i32 + rec) as usize;
            if affected_meas.contains(&abs_idx) {
                parity ^= 1;
            }
        }
        if parity != 0 {
            fired.push(obs_idx);
        }
    }
    fired
}

fn catalog_effect_parts(
    effect: PropagatedFaultEffect,
    det_records: &[Vec<i32>],
    obs_records: &[Vec<i32>],
    num_meas: usize,
) -> (Vec<usize>, Vec<usize>, Vec<usize>, Vec<usize>) {
    let affected: Vec<usize> = effect.affected_measurements.into_iter().collect();
    let dets = measurements_to_detectors(&affected, det_records, num_meas);
    let obs = measurements_to_observables(&affected, obs_records, num_meas);
    (affected, dets, obs, effect.affected_tracked_ops)
}

// ============================================================================
// Shared symbolic simulation helper
// ============================================================================

/// Run `SymbolicSparseStab` through a `TickCircuit` with proper PZ (reset)
/// semantics, returning the `MeasurementHistory` with correct cross-reset
/// correlations.
///
/// Iterates tick-by-tick to match the TickCircuit's measurement numbering
/// (which detector/DEM-output record indices reference).
///
/// Errors on unsupported gates with tick/gate/qubit context (same gate set
/// as [`build_fault_table`]).
pub fn symbolic_measurement_history(
    tc: &TickCircuit,
) -> Result<MeasurementHistory, UnsupportedGateError> {
    use pecos_simulators::SymbolicSparseStab;

    let num_qubits = tc
        .ticks()
        .iter()
        .flat_map(|t| t.gates().iter())
        .flat_map(|g| g.qubits.iter())
        .map(|q| q.index() + 1)
        .max()
        .unwrap_or(0);

    let mut sim = SymbolicSparseStab::new(num_qubits);

    for (tick_idx, tick) in tc.ticks().iter().enumerate() {
        for (gate_idx, gate) in tick.gates().iter().enumerate() {
            let qs: Vec<usize> = gate.qubits.iter().map(|q| q.index()).collect();

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
                | GateType::PauliOperatorMeta => {}
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
    pub fn sample(&self, shots: usize, seed: u64) -> SampleResult {
        let raw = self.sample_raw(shots, seed);
        SampleResult::new(raw.columns, shots)
    }

    /// Sample raw measurements with r-source column access.
    ///
    /// Physical mechanisms use geometric skip: O(p * shots) RNG calls per
    /// mechanism, not O(shots). For typical QEC noise (p ~ 0.005, 20k shots),
    /// this is ~100 firings per mechanism vs 20000 iterations.
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
    /// Returns (measurement_columns, r_source_columns).
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
        for (mech_idx, mechanism) in self.mechanisms.iter().enumerate() {
            let inv_log = self.inv_log_1_minus_p[mech_idx];
            let p = mechanism.probability;
            let num_alts = mechanism.alternatives.len();

            // p=1: every shot fires (handle before inv_log check since inv_log=0 for p=1)
            if p >= 1.0 {
                for shot in 0..shots {
                    let word_idx = shot / 64;
                    let bit_idx = shot % 64;
                    let mask = 1u64 << bit_idx;
                    let alt_idx = if num_alts == 1 {
                        0
                    } else {
                        rng.random_range(0..num_alts)
                    };
                    for &meas_idx in &mechanism.alternatives[alt_idx] {
                        columns[meas_idx][word_idx] ^= mask;
                    }
                }
                continue;
            }

            // Skip p=0 mechanisms (inv_log=0 means p≈0 or exactly 0)
            if p == 0.0 || inv_log == 0.0 {
                continue;
            }

            // Geometric skip sampling: O(fired events)
            let mut shot: usize = 0;
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
                for &meas_idx in &mechanism.alternatives[alt_idx] {
                    if meas_idx < columns.len() {
                        columns[meas_idx][word_idx] ^= mask;
                    }
                }

                shot += 1;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a minimal TickCircuit: PZ(0) H(0) CX(0,1) H(0) MZ(0) PZ(0) H(0) CX(0,1) H(0) MZ(0)
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
                    "Fault alternative crosses PZ boundary: {:?}",
                    alt
                );
            }
        }
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
    fn test_catalog_keeps_observables_and_tracked_ops_distinct() {
        let mut tc = TickCircuit::new();
        tc.tick().h(&[QubitId(0)]);
        tc.pauli_operator_labeled("tracked_z0", PauliString::z(0));
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
        assert_eq!(x_fault.affected_tracked_ops, vec![0]);
        assert_eq!(y_fault.affected_tracked_ops, vec![0]);
        assert_eq!(z_fault.affected_tracked_ops, Vec::<usize>::new());

        let configs: Vec<_> = catalog.fault_configurations(1).collect();
        assert!(
            configs
                .iter()
                .any(|config| config.affected_tracked_ops.as_slice() == [0]
                    && config.affected_observables.is_empty())
        );
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
        assert_eq!(c.selected_probability, 1.0);
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
        assert_eq!(catalog.locations.len(), 2);

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
        let mean = ones as f64 / 9000.0;
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
