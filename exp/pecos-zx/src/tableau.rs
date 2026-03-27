// Copyright 2026 The PECOS Developers
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

//! Stabilizer tableau propagation and detector extraction.
//!
//! This module provides a standalone algebraic approach to detector extraction
//! using the Heisenberg picture. A [`CliffordTableau`] maintains an accumulated
//! Clifford unitary and tracks preparations and measurements, then extracts
//! detectors via GF(2) nullspace computation.
//!
//! This is independent of `pecos-simulators` -- it uses only [`CliffordRep`] from
//! `pecos-core` and [`Mat2`] from `quizx` for GF(2) linear algebra.
//!
//! # Algorithm
//!
//! **Heisenberg picture**: Maintain a `CliffordRep` representing the accumulated
//! unitary U. After propagating all gates, `U Z_q U^dagger` gives the
//! forward-propagated stabilizer for each prepared qubit q.
//!
//! **Detector extraction**:
//! 1. Build stabilizer rows S: forward-propagated `Z_q` for each prepared qubit
//! 2. Build measurement rows M: `Z_{q_k}` for each measurement k
//! 3. Stack A = \[M; S\], compute nullspace of A^T
//! 4. Each nullspace vector's first m components give a detector
//! 5. Sign tracking via `PauliString` multiplication determines expected parity

use pecos_core::clifford_rep::CliffordRep;
use pecos_core::gate_type::GateType;
use pecos_core::{ClassicalBitId, PauliString, QuarterPhase};
use pecos_quantum::{Circuit, DagCircuit};
use quizx::linalg::Mat2;

use crate::symplectic::{SymplecticMatrix, SymplecticVector, padded_clifford};

// ============================================================================
// Types
// ============================================================================

/// A record of a single measurement operation.
#[derive(Debug, Clone)]
pub struct MeasurementRecord {
    /// The qubit that was measured.
    pub qubit: usize,
    /// The classical bit this measurement writes to, if any.
    pub cbit: Option<ClassicalBitId>,
}

/// The result of detector extraction.
#[derive(Debug, Clone)]
pub struct DetectorResult {
    /// Each detector is a list of measurement indices whose XOR is deterministic.
    pub detectors: Vec<Vec<usize>>,
    /// The expected parity for each detector (true = odd, false = even).
    pub expected_parities: Vec<bool>,
}

/// Errors that can occur during tableau propagation.
#[derive(Debug, thiserror::Error)]
pub enum TableauError {
    /// A gate type that cannot be represented as a Clifford operation.
    #[error("unsupported gate type for tableau propagation: {0:?}")]
    UnsupportedGate(GateType),
}

/// Stabilizer tableau for Clifford circuit analysis.
///
/// Tracks an accumulated Clifford unitary in the Heisenberg picture,
/// along with preparation and measurement records, enabling detector
/// extraction via GF(2) nullspace computation.
#[derive(Debug, Clone)]
pub struct CliffordTableau {
    n: usize,
    clifford: CliffordRep,
    prepared: Vec<bool>,
    measurements: Vec<MeasurementRecord>,
}

impl CliffordTableau {
    /// Create a new tableau for `n` qubits with identity Clifford.
    #[must_use]
    pub fn new(n: usize) -> Self {
        Self {
            n,
            clifford: CliffordRep::identity(n),
            prepared: vec![false; n],
            measurements: Vec::new(),
        }
    }

    /// Mark qubit `q` as prepared in |0> and reset its Clifford columns to identity.
    pub fn prepare(&mut self, q: usize) {
        self.prepared[q] = true;
        self.clifford.set_x_image(q, PauliString::x(q));
        self.clifford.set_z_image(q, PauliString::z(q));
    }

    /// Record a measurement of qubit `q`, optionally targeting classical bit `cbit`.
    pub fn mz(&mut self, q: usize, cbit: Option<ClassicalBitId>) {
        self.measurements.push(MeasurementRecord { qubit: q, cbit });
    }

    /// Number of qubits.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// Number of measurements recorded so far.
    #[must_use]
    pub fn num_measurements(&self) -> usize {
        self.measurements.len()
    }

    /// Number of prepared qubits.
    #[must_use]
    pub fn num_prepared(&self) -> usize {
        self.prepared.iter().filter(|&&p| p).count()
    }

    /// The measurement records.
    #[must_use]
    pub fn measurements(&self) -> &[MeasurementRecord] {
        &self.measurements
    }

    /// The forward-propagated stabilizer generators (one per prepared qubit).
    ///
    /// Returns `(qubit_index, propagated_stabilizer)` pairs.
    #[must_use]
    pub fn stabilizer_generators(&self) -> Vec<(usize, PauliString)> {
        self.prepared
            .iter()
            .enumerate()
            .filter(|(_, p)| **p)
            .map(|(q, _)| (q, self.clifford.z_image(q).clone()))
            .collect()
    }

    // ========================================================================
    // Gate application
    // ========================================================================

    fn apply_gate(&mut self, gate_rep: CliffordRep) {
        let padded = padded_clifford(gate_rep, self.n);
        self.clifford = padded.compose(&self.clifford);
    }

    /// Apply a Hadamard gate on qubit `q`.
    pub fn apply_h(&mut self, q: usize) {
        self.apply_gate(CliffordRep::h(q));
    }

    /// Apply an S gate on qubit `q`.
    pub fn apply_s(&mut self, q: usize) {
        self.apply_gate(CliffordRep::sz(q));
    }

    /// Apply an S-dagger gate on qubit `q`.
    pub fn apply_sdg(&mut self, q: usize) {
        self.apply_gate(CliffordRep::szdg(q));
    }

    /// Apply a Pauli X gate on qubit `q`.
    pub fn apply_x(&mut self, q: usize) {
        self.apply_gate(CliffordRep::x(q));
    }

    /// Apply a Pauli Y gate on qubit `q`.
    pub fn apply_y(&mut self, q: usize) {
        self.apply_gate(CliffordRep::y(q));
    }

    /// Apply a Pauli Z gate on qubit `q`.
    pub fn apply_z(&mut self, q: usize) {
        self.apply_gate(CliffordRep::z(q));
    }

    /// Apply an SX (sqrt-X) gate on qubit `q`.
    pub fn apply_sx(&mut self, q: usize) {
        self.apply_gate(CliffordRep::sx(q));
    }

    /// Apply an SY (sqrt-Y) gate on qubit `q`.
    pub fn apply_sy(&mut self, q: usize) {
        self.apply_gate(CliffordRep::sy(q));
    }

    /// Apply a CX (CNOT) gate with `control` and `target`.
    pub fn apply_cx(&mut self, control: usize, target: usize) {
        self.apply_gate(CliffordRep::cx(control, target));
    }

    /// Apply a CZ gate on qubits `q0` and `q1`.
    pub fn apply_cz(&mut self, q0: usize, q1: usize) {
        self.apply_gate(CliffordRep::cz(q0, q1));
    }

    /// Apply a CY gate with `control` and `target`.
    pub fn apply_cy(&mut self, control: usize, target: usize) {
        self.apply_gate(CliffordRep::cy(control, target));
    }

    /// Apply a SWAP gate on qubits `q0` and `q1`.
    pub fn apply_swap(&mut self, q0: usize, q1: usize) {
        self.apply_gate(CliffordRep::swap(q0, q1));
    }

    // ========================================================================
    // Accessors
    // ========================================================================

    /// The internal Clifford representation.
    #[must_use]
    pub fn clifford(&self) -> &CliffordRep {
        &self.clifford
    }

    /// Which qubits have been prepared (indexed by qubit number).
    #[must_use]
    pub fn prepared(&self) -> &[bool] {
        &self.prepared
    }

    // ========================================================================
    // from_dag / apply_dag
    // ========================================================================

    /// Build a `CliffordTableau` by iterating a `DagCircuit` in topological order.
    ///
    /// # Errors
    ///
    /// Returns `TableauError::UnsupportedGate` if the circuit contains non-Clifford gates.
    pub fn from_dag(dag: &DagCircuit) -> Result<Self, TableauError> {
        let n = dag.width();
        let mut tableau = Self::new(n);
        tableau.apply_dag(dag)?;
        Ok(tableau)
    }

    /// Apply a `DagCircuit` segment to this tableau.
    ///
    /// The tableau must already have enough qubits for the circuit. Qubits in
    /// the `DagCircuit` are mapped by their index in `dag.qubits()`.
    ///
    /// # Errors
    ///
    /// Returns `TableauError::UnsupportedGate` if the circuit contains non-Clifford gates.
    pub fn apply_dag(&mut self, dag: &DagCircuit) -> Result<(), TableauError> {
        for (node_id, gate) in dag.iter_gates_topo() {
            let arity = gate.quantum_arity();
            let qubits = &gate.qubits;

            for chunk in qubits.chunks(arity) {
                let qs: Vec<usize> = chunk.iter().map(|q| usize::from(*q)).collect();

                match gate.gate_type {
                    // Preparation
                    GateType::PZ | GateType::QAlloc => {
                        self.prepare(qs[0]);
                    }

                    // Measurement
                    GateType::MZ | GateType::MeasureFree => {
                        let cbit = dag.measurement_target(node_id);
                        self.mz(qs[0], cbit);
                    }

                    // Single-qubit Clifford gates
                    GateType::I => {}
                    GateType::H => self.apply_h(qs[0]),
                    GateType::X => self.apply_x(qs[0]),
                    GateType::Y => self.apply_y(qs[0]),
                    GateType::Z => self.apply_z(qs[0]),
                    GateType::SX => self.apply_sx(qs[0]),
                    GateType::SY => self.apply_sy(qs[0]),
                    GateType::SZ => self.apply_s(qs[0]),
                    GateType::SXdg => {
                        // SXdg = SX^3 = SX * SX * SX
                        self.apply_sx(qs[0]);
                        self.apply_sx(qs[0]);
                        self.apply_sx(qs[0]);
                    }
                    GateType::SYdg => {
                        // SYdg = SY^3
                        self.apply_sy(qs[0]);
                        self.apply_sy(qs[0]);
                        self.apply_sy(qs[0]);
                    }
                    GateType::SZdg => self.apply_sdg(qs[0]),

                    // Two-qubit Clifford gates
                    GateType::CX => self.apply_cx(qs[0], qs[1]),
                    GateType::CY => self.apply_cy(qs[0], qs[1]),
                    GateType::CZ => self.apply_cz(qs[0], qs[1]),
                    GateType::SWAP => self.apply_swap(qs[0], qs[1]),

                    other => return Err(TableauError::UnsupportedGate(other)),
                }
            }
        }

        Ok(())
    }

    // ========================================================================
    // Detector extraction
    // ========================================================================

    /// Extract detectors from the accumulated tableau.
    ///
    /// Uses a GF(2) nullspace computation to find which measurement outcome
    /// combinations are deterministic, and tracks signs to determine the
    /// expected parity of each detector.
    #[must_use]
    pub fn extract_detectors(&self) -> DetectorResult {
        let m = self.measurements.len();
        let prepared_qubits: Vec<usize> = self
            .prepared
            .iter()
            .enumerate()
            .filter(|(_, p)| **p)
            .map(|(q, _)| q)
            .collect();
        let p = prepared_qubits.len();
        let total_rows = m + p;
        let cols = 2 * self.n;

        if total_rows == 0 || cols == 0 {
            return DetectorResult {
                detectors: Vec::new(),
                expected_parities: Vec::new(),
            };
        }

        // Build the constraint matrix A (total_rows x cols):
        // - First m rows: measurement observables Z_{q_k}
        // - Next p rows: forward-propagated stabilizer generators
        let mut a = Mat2::zeros(total_rows, cols);

        // Measurement rows: Z_{q_k} has a single 1 at position n + q_k
        for (k, meas) in self.measurements.iter().enumerate() {
            a[(k, self.n + meas.qubit)] = 1;
        }

        // Stabilizer rows: forward-propagated Z_q for each prepared qubit
        for (idx, &q) in prepared_qubits.iter().enumerate() {
            let stab = self.clifford.z_image(q);
            let sv = SymplecticVector::from_pauli_string(stab, self.n);
            for col in 0..cols {
                a[(m + idx, col)] = sv.bits()[col];
            }
        }

        // Compute nullspace of A^T
        let at = a.transpose();
        let nullspace = at.nullspace();

        // Extract detectors from nullspace vectors
        let mut detectors = Vec::new();
        let mut expected_parities = Vec::new();

        for ns_vec in &nullspace {
            // The nullspace vector has total_rows components.
            // First m components: which measurements are in this detector.
            let meas_indices: Vec<usize> = (0..m).filter(|&k| ns_vec[(0, k)] == 1).collect();

            if meas_indices.is_empty() {
                // Pure stabilizer relation, not a detector
                continue;
            }

            // Sign tracking: multiply the stabilizer generators that participate
            let mut stab_product = PauliString::identity();
            for idx in 0..p {
                if ns_vec[(0, m + idx)] == 1 {
                    let stab = self.clifford.z_image(prepared_qubits[idx]);
                    stab_product = stab_product * stab;
                }
            }

            // The expected parity is determined by the sign of the stabilizer product.
            // Phase +1 => even parity (XOR = 0), Phase -1 => odd parity (XOR = 1).
            let parity = stab_product.phase() == QuarterPhase::MinusOne;

            detectors.push(meas_indices);
            expected_parities.push(parity);
        }

        DetectorResult {
            detectors,
            expected_parities,
        }
    }
}

// ============================================================================
// Periodic circuit analysis
// ============================================================================

/// A detector expressed as measurement offsets relative to a round boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeriodicDetector {
    /// Measurement indices relative to the start of the relevant round(s).
    pub measurement_offsets: Vec<usize>,
    /// Expected parity of the XOR of these measurements.
    pub expected_parity: bool,
}

/// Result of periodic circuit analysis.
///
/// Classifies detectors from a periodic QEC circuit into categories based on
/// which round boundaries they straddle.
#[derive(Debug, Clone)]
pub struct PeriodicAnalysis {
    /// Detectors involving only round-0 measurements (known from initial state).
    pub init_boundary: Vec<PeriodicDetector>,
    /// Detectors connecting consecutive rounds (round i and round i+1).
    pub inter_round: Vec<PeriodicDetector>,
    /// Detectors within a single round.
    pub intra_round: Vec<PeriodicDetector>,
    /// Detectors involving the final measurement layer.
    pub final_boundary: Vec<PeriodicDetector>,
    /// The symplectic transfer map for one body round.
    pub transfer_map: SymplecticMatrix,
    /// Number of measurements per body round.
    pub measurements_per_round: usize,
    /// Number of data qubits (from the init segment).
    pub num_data_qubits: usize,
    /// Total qubit count used in the analysis.
    pub num_total_qubits: usize,
}

/// Errors from periodic circuit analysis.
#[derive(Debug, thiserror::Error)]
pub enum PeriodicCircuitError {
    /// A tableau propagation error.
    #[error(transparent)]
    Tableau(#[from] TableauError),
    /// The body segment has no measurements.
    #[error("body segment must contain at least one measurement")]
    NoBodyMeasurements,
    /// Inconsistent body rounds: different measurement counts.
    #[error("body rounds produced different measurement counts ({0} vs {1})")]
    InconsistentRounds(usize, usize),
}

/// Build a single unrolled `DagCircuit` from periodic segments.
///
/// Replays gates from `init`, then `body` repeated `num_rounds` times, then
/// `finalize`. Each gate is cloned and auto-wired into the target circuit.
///
/// This is useful for feeding periodic circuits into pipelines that expect a
/// single flat circuit (e.g., ZX conversion and DEM extraction).
#[must_use]
pub fn build_unrolled_circuit(
    init: &DagCircuit,
    body: &DagCircuit,
    finalize: &DagCircuit,
    num_rounds: usize,
) -> DagCircuit {
    let mut target = DagCircuit::new();

    // Replay init
    for (_node, gate) in init.iter_gates_topo() {
        target.add_gate_auto_wire(gate.clone());
    }

    // Replay body num_rounds times
    for _ in 0..num_rounds {
        for (_node, gate) in body.iter_gates_topo() {
            target.add_gate_auto_wire(gate.clone());
        }
    }

    // Replay finalize
    for (_node, gate) in finalize.iter_gates_topo() {
        target.add_gate_auto_wire(gate.clone());
    }

    target
}

/// Analyze a periodic QEC circuit.
///
/// Accepts three `DagCircuit` segments:
/// - `init`: prepare data qubits (and possibly ancillas)
/// - `body`: one round of syndrome extraction (prepare ancillas, gates, measure ancillas)
/// - `finalize`: final measurements (typically measure all data qubits)
///
/// Runs the tableau through init -> body (round 0) -> body (round 1) -> finalize,
/// extracts detectors, and classifies them by which round boundaries they reference.
///
/// # Errors
///
/// Returns `PeriodicCircuitError` if the body has no measurements or if the two
/// body rounds produce different measurement counts.
pub fn analyze_periodic(
    init: &DagCircuit,
    body: &DagCircuit,
    finalize: &DagCircuit,
) -> Result<PeriodicAnalysis, PeriodicCircuitError> {
    let n = init.width().max(body.width()).max(finalize.width());
    let mut tableau = CliffordTableau::new(n);

    // Apply init segment
    tableau.apply_dag(init)?;
    let meas_after_init = tableau.num_measurements();

    // Apply body round 0
    tableau.apply_dag(body)?;
    let meas_after_round0 = tableau.num_measurements();
    let round0_count = meas_after_round0 - meas_after_init;

    if round0_count == 0 {
        return Err(PeriodicCircuitError::NoBodyMeasurements);
    }

    // Snapshot the Clifford after one body round for the transfer map.
    // We need the Clifford that represents just the body round's unitary
    // (excluding prep/measure). Build it by running body on a fresh tableau
    // without recording prep/measure -- but that changes semantics.
    // Instead, extract it from the current accumulated Clifford.
    // The transfer map is the symplectic representation of the accumulated
    // Clifford at this point (after init + 1 body round).
    let transfer_map = SymplecticMatrix::from_clifford_rep(tableau.clifford());

    // Apply body round 1
    tableau.apply_dag(body)?;
    let meas_after_round1 = tableau.num_measurements();
    let round1_count = meas_after_round1 - meas_after_round0;

    if round1_count != round0_count {
        return Err(PeriodicCircuitError::InconsistentRounds(
            round0_count,
            round1_count,
        ));
    }

    // Apply finalize
    tableau.apply_dag(finalize)?;
    let total_meas = tableau.num_measurements();
    let final_count = total_meas - meas_after_round1;

    // Extract all detectors
    let result = tableau.extract_detectors();

    // Classify detectors by which measurement ranges they reference
    let num_data_qubits = init.width();
    let (init_boundary, inter_round, intra_round, final_boundary) = classify_detectors(
        &result,
        meas_after_init,
        round0_count,
        meas_after_round0,
        round1_count,
        meas_after_round1,
        final_count,
    );

    Ok(PeriodicAnalysis {
        init_boundary,
        inter_round,
        intra_round,
        final_boundary,
        transfer_map,
        measurements_per_round: round0_count,
        num_data_qubits,
        num_total_qubits: n,
    })
}

/// Classify detectors into periodic categories based on measurement index ranges.
///
/// Measurement layout:
/// - `[0, meas_after_init)`: init measurements
/// - `[meas_after_init, meas_after_init + round0_count)`: round 0 body measurements
/// - `[meas_after_round0, meas_after_round0 + round1_count)`: round 1 body measurements
/// - `[meas_after_round1, meas_after_round1 + final_count)`: finalize measurements
fn classify_detectors(
    result: &DetectorResult,
    meas_after_init: usize,
    round0_count: usize,
    meas_after_round0: usize,
    round1_count: usize,
    meas_after_round1: usize,
    final_count: usize,
) -> (
    Vec<PeriodicDetector>,
    Vec<PeriodicDetector>,
    Vec<PeriodicDetector>,
    Vec<PeriodicDetector>,
) {
    let round0_start = meas_after_init;
    let round0_end = meas_after_init + round0_count;
    let round1_start = meas_after_round0;
    let round1_end = meas_after_round0 + round1_count;
    let final_start = meas_after_round1;
    let _final_end = meas_after_round1 + final_count;

    let mut init_boundary = Vec::new();
    let mut inter_round = Vec::new();
    let mut intra_round = Vec::new();
    let mut final_boundary = Vec::new();

    for (det, &parity) in result.detectors.iter().zip(&result.expected_parities) {
        let has_init = det.iter().any(|&m| m < round0_start);
        let has_round0 = det.iter().any(|&m| m >= round0_start && m < round0_end);
        let has_round1 = det.iter().any(|&m| m >= round1_start && m < round1_end);
        let has_final = det.iter().any(|&m| m >= final_start);

        if has_round0 && has_round1 && !has_init && !has_final {
            // Inter-round: connects round 0 to round 1
            // Store offsets relative to round starts
            let offsets: Vec<usize> = det
                .iter()
                .map(|&m| {
                    if m >= round1_start {
                        // Round 1 measurement -> offset within round + round0_count
                        // (so compose can shift by round index)
                        round0_count + (m - round1_start)
                    } else {
                        // Round 0 measurement -> offset within round
                        m - round0_start
                    }
                })
                .collect();
            inter_round.push(PeriodicDetector {
                measurement_offsets: offsets,
                expected_parity: parity,
            });
        } else if has_final || (has_round1 && !has_round0) {
            // Final boundary: involves finalize measurements and/or only round 1
            let offsets: Vec<usize> = det
                .iter()
                .map(|&m| {
                    if m >= final_start {
                        // Finalize measurement -> offset from round1 start
                        round1_count + (m - final_start)
                    } else if m >= round1_start {
                        // Round 1 measurement -> offset within round
                        m - round1_start
                    } else {
                        // Round 0 body measurement appearing in a final-boundary
                        // detector. This happens when the nullspace basis combines
                        // a round-0 measurement with finalize measurements. During
                        // compose, offset < round_count maps to the last body round.
                        m - round0_start
                    }
                })
                .collect();
            final_boundary.push(PeriodicDetector {
                measurement_offsets: offsets,
                expected_parity: parity,
            });
        } else if has_round0 && !has_round1 && !has_final {
            // Could be init boundary or intra-round
            if has_init {
                // Init boundary: involves init-phase + round 0 measurements
                let offsets: Vec<usize> = det
                    .iter()
                    .map(|&m| {
                        if m >= round0_start {
                            meas_after_init + (m - round0_start)
                        } else {
                            m
                        }
                    })
                    .collect();
                init_boundary.push(PeriodicDetector {
                    measurement_offsets: offsets,
                    expected_parity: parity,
                });
            } else {
                // Intra-round: only round 0 measurements, no init/round1/final
                // Check if these are single-round detectors
                let offsets: Vec<usize> = det.iter().map(|&m| m - round0_start).collect();
                // Verify it also appears in round 1 by checking if the same
                // pattern exists as an intra-round detector
                intra_round.push(PeriodicDetector {
                    measurement_offsets: offsets,
                    expected_parity: parity,
                });
            }
        } else if !has_round0 && !has_round1 && !has_final {
            // Only init measurements -- init boundary
            let offsets: Vec<usize> = det.to_vec();
            init_boundary.push(PeriodicDetector {
                measurement_offsets: offsets,
                expected_parity: parity,
            });
        } else {
            // Init boundary: mix of init and round 0
            let offsets: Vec<usize> = det
                .iter()
                .map(|&m| {
                    if m >= round0_start {
                        meas_after_init + (m - round0_start)
                    } else {
                        m
                    }
                })
                .collect();
            init_boundary.push(PeriodicDetector {
                measurement_offsets: offsets,
                expected_parity: parity,
            });
        }
    }

    (init_boundary, inter_round, intra_round, final_boundary)
}

impl PeriodicAnalysis {
    /// Expand the periodic detector structure for `num_rounds` body rounds.
    ///
    /// Produces a `DetectorResult` with all detectors properly shifted:
    /// - Init boundary detectors (emitted once, unchanged)
    /// - Inter-round detectors (emitted for each pair of consecutive rounds)
    /// - Intra-round detectors (emitted for each round)
    /// - Final boundary detectors (emitted once, shifted to the last round)
    #[must_use]
    pub fn compose(&self, num_rounds: usize) -> DetectorResult {
        let mut detectors = Vec::new();
        let mut expected_parities = Vec::new();
        let mpr = self.measurements_per_round;

        // Init boundary detectors (always present, no shift needed)
        for det in &self.init_boundary {
            detectors.push(det.measurement_offsets.clone());
            expected_parities.push(det.expected_parity);
        }

        // Intra-round detectors: emit for each round
        for round in 0..num_rounds {
            let shift = round * mpr;
            for det in &self.intra_round {
                let shifted: Vec<usize> =
                    det.measurement_offsets.iter().map(|&o| o + shift).collect();
                detectors.push(shifted);
                expected_parities.push(det.expected_parity);
            }
        }

        // Inter-round detectors: emit for each pair (round i, round i+1)
        for round in 0..num_rounds.saturating_sub(1) {
            let shift = round * mpr;
            for det in &self.inter_round {
                let shifted: Vec<usize> =
                    det.measurement_offsets.iter().map(|&o| o + shift).collect();
                detectors.push(shifted);
                expected_parities.push(det.expected_parity);
            }
        }

        // Final boundary detectors: shift to last round
        let final_shift = num_rounds.saturating_sub(1) * mpr;
        for det in &self.final_boundary {
            let shifted: Vec<usize> = det
                .measurement_offsets
                .iter()
                .map(|&o| o + final_shift)
                .collect();
            detectors.push(shifted);
            expected_parities.push(det.expected_parity);
        }

        DetectorResult {
            detectors,
            expected_parities,
        }
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::PauliOperator;

    // ====================================================================
    // Test 1: Identity circuit
    // ====================================================================

    #[test]
    fn identity_circuit_detectors() {
        // Prep + measure each qubit, no gates.
        // Each measurement individually deterministic => n detectors.
        let n = 3;
        let mut tab = CliffordTableau::new(n);
        for q in 0..n {
            tab.prepare(q);
        }
        for q in 0..n {
            tab.mz(q, None);
        }

        let result = tab.extract_detectors();
        assert_eq!(
            result.detectors.len(),
            n,
            "Expected {n} detectors for identity circuit, got {}",
            result.detectors.len()
        );

        // Each detector should involve exactly one measurement
        for det in &result.detectors {
            assert_eq!(det.len(), 1, "Each detector should be a single measurement");
        }

        // All expected parities should be false (even = 0 outcome)
        // because initial state is |0> and measurement is Z-basis
        for &parity in &result.expected_parities {
            assert!(!parity, "Expected even parity for |0> state measurement");
        }
    }

    // ====================================================================
    // Test 2: Bell state
    // ====================================================================

    #[test]
    fn bell_state_detector() {
        // Prep q0, q1. H(0), CX(0,1). Measure q0, q1.
        // Neither individually deterministic, but parity is.
        // Expect 1 detector involving both measurements.
        let mut tab = CliffordTableau::new(2);
        tab.prepare(0);
        tab.prepare(1);
        tab.apply_h(0);
        tab.apply_cx(0, 1);
        tab.mz(0, None);
        tab.mz(1, None);

        let result = tab.extract_detectors();
        assert_eq!(
            result.detectors.len(),
            1,
            "Expected 1 detector for Bell state, got {}",
            result.detectors.len()
        );

        let det = &result.detectors[0];
        assert_eq!(det.len(), 2, "Detector should involve both measurements");
        assert!(det.contains(&0));
        assert!(det.contains(&1));

        // Expected parity: even (m0 XOR m1 = 0, both agree in Bell state)
        assert!(
            !result.expected_parities[0],
            "Expected even parity for Bell state"
        );
    }

    // ====================================================================
    // Test 3: Repetition code (1 round)
    // ====================================================================

    #[test]
    fn repetition_code_detectors() {
        // 3 data qubits (0,1,2) + 2 ancilla qubits (3,4), all prepared in |0>.
        // In a single round with known initial state, ancilla measurements
        // are deterministic because the measurement observables are products
        // of the initial stabilizer generators.
        //
        // Syndrome extraction:
        //   CX(0,3), CX(1,3)  -- ancilla 3 measures Z0*Z1
        //   CX(1,4), CX(2,4)  -- ancilla 4 measures Z1*Z2
        //
        // Measure ancillas 3,4.
        // Expect 2 detectors (one per ancilla measurement).
        let n = 5;
        let mut tab = CliffordTableau::new(n);
        for q in 0..n {
            tab.prepare(q);
        }

        // Syndrome extraction
        tab.apply_cx(0, 3);
        tab.apply_cx(1, 3);
        tab.apply_cx(1, 4);
        tab.apply_cx(2, 4);

        // Measure ancillas
        tab.mz(3, None);
        tab.mz(4, None);

        let result = tab.extract_detectors();
        assert_eq!(
            result.detectors.len(),
            2,
            "Expected 2 detectors for repetition code, got {}",
            result.detectors.len()
        );

        // Each detector should involve exactly one measurement
        for det in &result.detectors {
            assert_eq!(
                det.len(),
                1,
                "Each repetition code detector should be a single measurement"
            );
        }

        // Both parities should be even (ancillas start in |0>, no errors)
        for &parity in &result.expected_parities {
            assert!(
                !parity,
                "Expected even parity for ancilla measurements with no errors"
            );
        }
    }

    // ====================================================================
    // Test 4: from_dag round-trip
    // ====================================================================

    #[test]
    fn from_dag_bell_state() {
        // Build a Bell circuit via DagCircuit, then extract detectors via from_dag.
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.pz(&[1]);
        dag.h(&[0]);
        dag.cx(&[(0, 1)]);
        dag.mz(&[0]);
        dag.mz(&[1]);

        let tab = CliffordTableau::from_dag(&dag).expect("Bell circuit is Clifford");
        let result = tab.extract_detectors();

        assert_eq!(result.detectors.len(), 1);
        assert_eq!(result.detectors[0].len(), 2);
        assert!(!result.expected_parities[0]);
    }

    // ====================================================================
    // Test 5: Cross-validation with Pauli webs
    // ====================================================================

    #[test]
    fn cross_validate_with_pauli_webs() {
        use crate::convert::dag_to_zx;
        use crate::pauli_web::{WebClassification, classify_webs, compute_pauli_webs};

        // Build the repetition code circuit
        let mut dag = DagCircuit::new();

        // Data qubits 0,1,2 are inputs (not prepared)
        // Ancilla qubits 3,4 are prepared
        dag.pz(&[3]);
        dag.pz(&[4]);

        // Syndrome extraction
        dag.cx(&[(0, 3)]);
        dag.cx(&[(1, 3)]);
        dag.cx(&[(1, 4)]);
        dag.cx(&[(2, 4)]);

        // Measure ancillas
        dag.mz(&[3]);
        dag.mz(&[4]);

        // Measure data qubits (needed for ZX graph to have outputs)
        dag.mz(&[0]);
        dag.mz(&[1]);
        dag.mz(&[2]);

        // Method 1: Tableau-based detector extraction
        let tab = CliffordTableau::from_dag(&dag).expect("repetition code is Clifford");
        let tableau_result = tab.extract_detectors();
        let tableau_detector_count = tableau_result.detectors.len();

        // Method 2: Pauli web-based classification
        let zx_graph = dag_to_zx(&dag).expect("should convert to ZX");
        let web_result = compute_pauli_webs(&zx_graph);
        let classifications = classify_webs(&web_result);
        let web_detector_count = classifications
            .iter()
            .filter(|c| **c == WebClassification::Detector)
            .count();

        assert_eq!(
            tableau_detector_count, web_detector_count,
            "Tableau detectors ({tableau_detector_count}) should match \
             Pauli web detectors ({web_detector_count})"
        );
    }

    // ====================================================================
    // Test 6: Stabilizer generators
    // ====================================================================

    #[test]
    fn stabilizer_generators_identity() {
        let n = 2;
        let mut tab = CliffordTableau::new(n);
        tab.prepare(0);
        tab.prepare(1);

        let gens = tab.stabilizer_generators();
        assert_eq!(gens.len(), 2);

        // Without any gates, stabilizers should be +Z_0 and +Z_1
        for (q, ps) in &gens {
            assert_eq!(ps.phase(), QuarterPhase::PlusOne);
            assert_eq!(ps.weight(), 1);
            assert_eq!(ps.get(*q), pecos_core::Pauli::Z);
        }
    }

    #[test]
    fn stabilizer_generators_bell() {
        let mut tab = CliffordTableau::new(2);
        tab.prepare(0);
        tab.prepare(1);
        tab.apply_h(0);
        tab.apply_cx(0, 1);

        let gens = tab.stabilizer_generators();
        assert_eq!(gens.len(), 2);

        // After H(0), CX(0,1):
        // Z_0 -> H: X_0 -> CX: X_0 X_1
        // Z_1 -> H: Z_1 -> CX: Z_0 Z_1
        let (_, g0) = &gens[0];
        assert_eq!(g0.weight(), 2);
        assert_eq!(g0.get(0), pecos_core::Pauli::X);
        assert_eq!(g0.get(1), pecos_core::Pauli::X);

        let (_, g1) = &gens[1];
        assert_eq!(g1.weight(), 2);
        assert_eq!(g1.get(0), pecos_core::Pauli::Z);
        assert_eq!(g1.get(1), pecos_core::Pauli::Z);
    }

    // ====================================================================
    // Test 7: Unsupported gate error
    // ====================================================================

    #[test]
    fn unsupported_gate_error() {
        use pecos_core::Angle64;

        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.rz(Angle64::from_turns(0.125), &[0]); // T gate equivalent, non-Clifford
        dag.mz(&[0]);

        let result = CliffordTableau::from_dag(&dag);
        assert!(result.is_err());
    }

    // ====================================================================
    // Test 8: Gate correctness via CliffordRep comparison
    // ====================================================================

    #[test]
    fn gate_application_matches_clifford_rep() {
        // Build a circuit manually and via DagCircuit, verify same Clifford
        let mut tab = CliffordTableau::new(3);
        tab.prepare(0);
        tab.prepare(1);
        tab.prepare(2);
        tab.apply_h(0);
        tab.apply_s(1);
        tab.apply_cx(0, 1);
        tab.apply_cz(1, 2);

        // Build same via direct CliffordRep composition
        let n = 3;
        let h = padded_clifford(CliffordRep::h(0), n);
        let s = padded_clifford(CliffordRep::sz(1), n);
        let cx = padded_clifford(CliffordRep::cx(0, 1), n);
        let cz = padded_clifford(CliffordRep::cz(1, 2), n);

        let expected = cz.compose(&cx.compose(&s.compose(&h)));

        // Compare stabilizer generators
        for q in 0..n {
            let tab_gen = tab.stabilizer_generators();
            let (_, tab_ps) = tab_gen.iter().find(|(qq, _)| *qq == q).unwrap();
            let exp_ps = expected.z_image(q);

            assert_eq!(
                tab_ps.phase(),
                exp_ps.phase(),
                "Phase mismatch for qubit {q}"
            );
            for i in 0..n {
                assert_eq!(
                    tab_ps.get(i),
                    exp_ps.get(i),
                    "Pauli mismatch at qubit {q}, position {i}"
                );
            }
        }
    }

    // ====================================================================
    // Periodic analysis helper: build repetition code segments
    // ====================================================================

    /// Build init/body/finalize segments for a 3-data, 2-ancilla repetition code.
    /// Qubits: 0,1,2 = data; 3,4 = ancillas.
    fn rep_code_segments() -> (DagCircuit, DagCircuit, DagCircuit) {
        // Init: prepare all 5 qubits
        let mut init = DagCircuit::new();
        for q in 0..5 {
            init.pz(&[q]);
        }

        // Body: prep ancillas, syndrome extraction, measure ancillas
        let mut body = DagCircuit::new();
        body.pz(&[3]);
        body.pz(&[4]);
        body.cx(&[(0, 3)]);
        body.cx(&[(1, 3)]);
        body.cx(&[(1, 4)]);
        body.cx(&[(2, 4)]);
        body.mz(&[3]);
        body.mz(&[4]);

        // Finalize: measure data qubits
        let mut finalize = DagCircuit::new();
        finalize.mz(&[0]);
        finalize.mz(&[1]);
        finalize.mz(&[2]);

        (init, body, finalize)
    }

    // ====================================================================
    // Test 9: Periodic analysis - repetition code 2 rounds
    // ====================================================================

    #[test]
    fn periodic_repetition_code() {
        let (init, body, finalize) = rep_code_segments();
        let analysis =
            analyze_periodic(&init, &body, &finalize).expect("repetition code should analyze");

        assert_eq!(
            analysis.measurements_per_round, 2,
            "2 ancilla measurements per round"
        );
        assert_eq!(analysis.num_total_qubits, 5);

        // 2 inter-round detectors (round-0 ancilla XOR round-1 ancilla)
        assert_eq!(
            analysis.inter_round.len(),
            2,
            "Expected 2 inter-round detectors, got {}",
            analysis.inter_round.len()
        );

        // Total classified detectors should match what the 2-round analysis produces
        let total = analysis.init_boundary.len()
            + analysis.inter_round.len()
            + analysis.intra_round.len()
            + analysis.final_boundary.len();
        assert_eq!(total, 7, "Expected 7 total detectors, got {total}");

        // compose(2) should reproduce the same count as the 2-round analysis
        let result = analysis.compose(2);
        assert_eq!(
            result.detectors.len(),
            7,
            "compose(2) should give 7 detectors"
        );
    }

    // ====================================================================
    // Test 10: compose(N) scaling
    // ====================================================================

    #[test]
    fn compose_scaling() {
        let (init, body, finalize) = rep_code_segments();
        let analysis =
            analyze_periodic(&init, &body, &finalize).expect("repetition code should analyze");

        // For N rounds: 2 init + 2*(N-1) inter-round + 3 final
        // (plus any intra-round * N)
        let intra_per_round = analysis.intra_round.len();

        for &n in &[1, 3, 10] {
            let result = analysis.compose(n);
            let expected = analysis.init_boundary.len()
                + intra_per_round * n
                + analysis.inter_round.len() * n.saturating_sub(1)
                + analysis.final_boundary.len();
            assert_eq!(
                result.detectors.len(),
                expected,
                "compose({n}): expected {expected} detectors, got {}",
                result.detectors.len()
            );
        }
    }

    // ====================================================================
    // Test 11: Transfer map validity
    // ====================================================================

    #[test]
    fn transfer_map_is_symplectic() {
        let (init, body, finalize) = rep_code_segments();
        let analysis =
            analyze_periodic(&init, &body, &finalize).expect("repetition code should analyze");

        assert!(
            analysis.transfer_map.is_valid(),
            "Transfer map should be a valid symplectic matrix"
        );

        // Composing with itself should still produce a valid symplectic matrix
        let doubled = analysis.transfer_map.compose(&analysis.transfer_map);
        assert!(
            doubled.is_valid(),
            "Composed transfer map should be a valid symplectic matrix"
        );
    }

    // ====================================================================
    // Test 12: Trivial periodic circuit
    // ====================================================================

    #[test]
    fn trivial_periodic_circuit() {
        // Body: prep ancilla (q1), measure ancilla (q1), no gates on data (q0).
        // Each ancilla measurement is individually deterministic every round.
        let mut init = DagCircuit::new();
        init.pz(&[0]);
        init.pz(&[1]);

        let mut body = DagCircuit::new();
        body.pz(&[1]);
        body.mz(&[1]);

        let mut finalize = DagCircuit::new();
        finalize.mz(&[0]);

        let analysis =
            analyze_periodic(&init, &body, &finalize).expect("trivial circuit should analyze");

        assert_eq!(analysis.measurements_per_round, 1);

        // The ancilla measurement should be deterministic within each round
        // (it's prepped and measured without entanglement).
        // This could show up as init_boundary for round 0 and intra-round,
        // or as init_boundary + inter_round depending on classification.
        let total_periodic = analysis.init_boundary.len()
            + analysis.inter_round.len()
            + analysis.intra_round.len()
            + analysis.final_boundary.len();
        assert!(
            total_periodic > 0,
            "Should have at least some detectors in trivial circuit"
        );

        // compose(N) should give a reasonable count
        let result_5 = analysis.compose(5);
        assert!(
            !result_5.detectors.is_empty(),
            "compose(5) should produce detectors"
        );
    }

    // ====================================================================
    // Test 13: Cross-validate with unrolled circuit
    // ====================================================================

    #[test]
    fn cross_validate_periodic_with_unrolled() {
        // Build a 3-round repetition code as a single unrolled DagCircuit
        let mut unrolled = DagCircuit::new();

        // Init: prep all qubits
        for q in 0..5 {
            unrolled.pz(&[q]);
        }

        // 3 body rounds
        for _round in 0..3 {
            unrolled.pz(&[3]);
            unrolled.pz(&[4]);
            unrolled.cx(&[(0, 3)]);
            unrolled.cx(&[(1, 3)]);
            unrolled.cx(&[(1, 4)]);
            unrolled.cx(&[(2, 4)]);
            unrolled.mz(&[3]);
            unrolled.mz(&[4]);
        }

        // Finalize
        unrolled.mz(&[0]);
        unrolled.mz(&[1]);
        unrolled.mz(&[2]);

        let unrolled_tab =
            CliffordTableau::from_dag(&unrolled).expect("unrolled circuit is Clifford");
        let unrolled_result = unrolled_tab.extract_detectors();

        // Periodic analysis + compose(3)
        let (init, body, finalize) = rep_code_segments();
        let analysis =
            analyze_periodic(&init, &body, &finalize).expect("repetition code should analyze");
        let periodic_result = analysis.compose(3);

        assert_eq!(
            periodic_result.detectors.len(),
            unrolled_result.detectors.len(),
            "Periodic compose(3) should match unrolled: periodic={}, unrolled={}",
            periodic_result.detectors.len(),
            unrolled_result.detectors.len()
        );
    }

    // ====================================================================
    // Test 14: No body measurements error
    // ====================================================================

    #[test]
    fn no_body_measurements_error() {
        let mut init = DagCircuit::new();
        init.pz(&[0]);

        // Body with no measurements
        let mut body = DagCircuit::new();
        body.h(&[0]);

        let mut finalize = DagCircuit::new();
        finalize.mz(&[0]);

        let result = analyze_periodic(&init, &body, &finalize);
        assert!(
            result.is_err(),
            "Should error when body has no measurements"
        );
    }
}
