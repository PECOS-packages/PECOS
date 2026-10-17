// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

//! Noisy single-shot BT CCZ preparation with syndrome repair in the loop.
//!
//! Implements arXiv:2510.07269, appendix "Details for simulations", Fig.
//! (`fig:simulation_circuits`)(d) and Table II, with two explicit extensions. Repair rejection
//! (corrected syndrome outside im `H_Z`) is not specified by the paper
//! (source line 1161); it is counted separately from X-detector rejection. We use
//! any solution of `H_Z` x = s', instead of integer programming, as permitted
//! for logical-plus preparation by source line 1150.
//!
//! X-check detectors plus inverse-target readout use dephasing, ideal inverse
//! encoding, X-check pivot detectors, and logical-pivot observables. In exact
//! mode the logical CCZ tensor is undone before scoring the target. This is an
//! intentional extension: the paper measures logical X directly. Non-pivot Z
//! readouts diagnose residual X errors but never reject a shot; the paper assumes
//! light Z postselection without simulating it (source line 1161).
//!
//! A residual physical X error on a logical pivot can pass through inverse
//! encoding as X on that pivot without being a logical X operator; exact-mode
//! tensor undo then propagates it into Z errors on other logical pivots, causing
//! observable flips without an X-detector event, while the surrogate has no
//! tensor undo and is blind to this mechanism. The `accepted_z_clean` and
//! `logical_failures_z_clean` pair, restricted to zero Z diagnostics, is the pair
//! comparable with the paper's postselected logical error rate; the unfiltered
//! `accepted` and `logical_failures` pair includes X errors the paper assumes
//! removed by light Z-detector postselection (arXiv:2510.07269, "Details for
//! simulations", source line 1161).
//!
//! The Z-clean filter also removes accepted shots with harmless residual X
//! (for example, X parts of post-CCZ depolarizing noise), so its acceptance is
//! filtered differently between models and is not directly comparable with the
//! paper's acceptance; only the Z-clean logical-failure ratio is comparable.
//!
//! For the three benchmark codes, dim(ker `M_Z` / im `H_Z`) = 3 = k: the boundary-map
//! meta checks of arXiv:2510.07269, equation (`eq:3D_check_mats`), source line 637,
//! have three fewer independent rows than the left kernel of `H_Z`. Thus a zero
//! meta syndrome does not guarantee image membership; repair rejection tests the
//! corrected syndrome against im `H_Z`. This module keeps the paper's boundary
//! map; complete meta checks would let the decoder repair such syndrome errors
//! instead of rejecting them.
//!
//! Compatibility follows arXiv:2610.06811, Section 6: all noiseless Z measurements
//! are compatible; syndrome correction prepares an exact code state. Inverse
//! encoding makes the X-check pivot measurements deterministic, and undoing the
//! logical tensor makes the logical X measurements deterministic too. Code-space
//! membership alone does not suffice for the latter. The theorem then covers
//! Pauli noise and Pauli corrections. Prefix and suffix run on one simulator,
//! with a classical repair step between them and no intervening reset.

use std::time::{Duration, Instant};

use ndarray::{Array1, Array2};
use pecos_core::{ChannelExpr, QubitId, channel, unitary_rep};
use pecos_num::graph::bipartite_edge_coloring;
use pecos_qec::BivariateTricycleCode;
use pecos_quantum::{F2Matrix, TickCircuit};
use pecos_random::PecosRng;
use pecos_relay_bp::{RelayBpDecoder, StoppingCriterion};

use crate::PhasePoly;
use crate::distillation::validate_probability;
use crate::runner::{Program, ShotResult};
use crate::tricycle_ccz::{Encoder, stage};

#[cfg(test)]
mod tests;

/// Physical CCZ or the paper's Pauli approximation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CczModel {
    /// Two cup-product layers, each gate followed by three-qubit depolarization.
    Exact,
    /// Interpretation of arXiv:2510.07269, "Details for simulations", lines
    /// 1157-1159: the correlated-Z channel replaces both propagated CZ errors
    /// and new three-qubit depolarizing errors ("the above errors"). No CCZ or
    /// additional depolarizing channel is applied in this mode.
    PaperSurrogate,
}

/// Counts over a complete run; logical failures count accepted shots only.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Counts {
    /// Number of requested shots executed.
    pub attempted: usize,
    /// Rejected because syndrome repair failed, before running the suffix.
    pub repair_failures: usize,
    /// Shots with successful repair and all X detectors zero.
    pub accepted: usize,
    /// Accepted shots with any nonzero logical observable.
    pub logical_failures: usize,
    /// Accepted shots with any nonzero non-pivot Z diagnostic.
    pub accepted_with_residual_x: usize,
    /// Accepted shots with every non-pivot Z diagnostic zero.
    pub accepted_z_clean: usize,
    /// Z-clean accepted shots with any nonzero logical observable.
    pub logical_failures_z_clean: usize,
}

/// Recorded syndrome and readout for one attempt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Outcome {
    /// Raw Z syndromes, in block order then `H_Z` row order.
    pub syndromes: Vec<Vec<u8>>,
    /// Number of nonzero meta syndromes sent to relay-BP.
    pub decoder_calls: usize,
    /// True when the corrected syndrome is outside the image of `H_Z`.
    pub repair_failure: bool,
    /// Suffix measurements and annotations; absent on repair failure.
    pub readout: Option<ShotResult>,
    /// Non-pivot Z measurements, in block order; empty on repair failure.
    pub diagnostics: Vec<bool>,
}

impl Outcome {
    /// Successful repair and no X-detector event.
    #[must_use]
    pub fn accepted(&self) -> bool {
        !self.repair_failure
            && self
                .readout
                .as_ref()
                .is_some_and(|r| r.detectors.iter().all(|b| !b))
    }
}

/// Counts and per-shot records from one reproducible run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunResult {
    /// Aggregate protocol outcomes.
    pub counts: Counts,
    /// Outcomes in attempt order, including repair rejections.
    pub outcomes: Vec<Outcome>,
}

/// Precompiled noisy preparation and readout, with a reusable GF(2) solver.
/// Decoder matrix and uniform priors are stored once; a builder creates fresh
/// memory-BP state per run, using explicitly pinned relay settings.
pub struct NoisyCcz {
    n: usize,
    m_z: usize,
    prefix: Program,
    suffix: Program,
    meta: Array2<u8>,
    priors: Vec<f64>,
    solver: SyndromeSolver,
    diagnostics_start: usize,
    construction_times: (Duration, Duration),
}

impl NoisyCcz {
    /// Compile both circuit segments and precompute syndrome repair.
    ///
    /// `q` is the explicit syndrome-error prior, independent of physical `p`.
    /// No idle noise is included. Each data and ancilla qubit is prepared once.
    ///
    /// # Errors
    /// Rejects invalid p, nonfinite q or q outside (0, 0.5), size overflow,
    /// edge-coloring errors, or invalid generated circuits.
    pub fn new(
        code: &BivariateTricycleCode,
        p: f64,
        q: f64,
        model: CczModel,
    ) -> Result<Self, String> {
        Self::build(
            code,
            p,
            q,
            model,
            #[cfg(test)]
            &[],
        )
    }

    fn build(
        code: &BivariateTricycleCode,
        p: f64,
        q: f64,
        model: CczModel,
        #[cfg(test)] faults: &[Fault],
    ) -> Result<Self, String> {
        let started = Instant::now();
        validate_probability(p)?;
        if !q.is_finite() || q <= 0.0 || q >= 0.5 {
            return Err("q must be finite and in (0,0.5)".into());
        }
        let n = code.num_qubits();
        let m_z = code.hz().num_checks();
        let total = n
            .checked_add(m_z)
            .and_then(|v| v.checked_mul(3))
            .ok_or("three-block circuit size overflows usize")?;
        let mut prefix = preparation(code, p, total)?;
        #[cfg(test)]
        append_faults(
            &mut prefix,
            faults,
            FaultPoint::BeforeAncillaMeasurement,
            3 * n,
            total,
        )?;
        prefix.tick().mz(&(3 * n..total).collect::<Vec<_>>());
        let mut suffix = TickCircuit::new();
        #[cfg(test)]
        append_faults(&mut suffix, faults, FaultPoint::BeforeCcz, 0, 3 * n)?;
        ccz_stage(&mut suffix, code, p, model);
        let diagnostics_start = readout(&mut suffix, code, p, model)?;
        let meta_rows = code.meta_z().rows();
        let meta = Array2::from_shape_vec((meta_rows.len(), m_z), meta_rows.concat())
            .map_err(|e| e.to_string())?;
        let priors = vec![q; m_z];
        let solver = SyndromeSolver::new(&F2Matrix::from_rows(code.hz().rows()));
        let build_time = started.elapsed();
        let started = Instant::now();
        let prefix = Program::compile(&prefix).map_err(|e| e.to_string())?;
        let suffix = Program::compile(&suffix).map_err(|e| e.to_string())?;
        Ok(Self {
            n,
            m_z,
            prefix,
            suffix,
            meta,
            priors,
            solver,
            diagnostics_start,
            construction_times: (build_time, started.elapsed()),
        })
    }

    /// Wall times for circuit/solver construction and program compilation, respectively.
    #[must_use]
    pub fn construction_times(&self) -> (Duration, Duration) {
        self.construction_times
    }

    /// Prefix program, including the one Z-check round.
    #[must_use]
    pub fn prefix(&self) -> &Program {
        &self.prefix
    }

    /// CCZ/noise and readout program, touching only data qubits.
    #[must_use]
    pub fn suffix(&self) -> &Program {
        &self.suffix
    }

    /// Run shots with fresh noise, measurement, and relay-BP streams.
    ///
    /// Three seeds are drawn in that order from `seed`. Decoder state is rebuilt
    /// per call and advances across the run, so equal (shots, seed) reproduce all
    /// counts and records. Zero meta syndromes bypass decoding, but not the image
    /// check. Repair failure stops the shot before any CCZ or readout operations.
    ///
    /// # Errors
    /// Returns decoder construction/execution errors or incompatible measurements.
    pub fn run(&self, shots: usize, seed: u64) -> Result<RunResult, String> {
        let mut outcomes = Vec::with_capacity(shots);
        let counts = self.run_inner(shots, seed, |outcome| outcomes.push(outcome))?;
        Ok(RunResult { counts, outcomes })
    }

    /// Run with the same streams as [`Self::run`], retaining only aggregate counts.
    /// Memory use is independent of the number of shots.
    ///
    /// # Errors
    /// Returns decoder construction/execution errors or incompatible measurements.
    pub fn run_counts(&self, shots: usize, seed: u64) -> Result<Counts, String> {
        self.run_inner(shots, seed, drop)
    }

    fn repair_block(
        &self,
        syndrome: &[u8],
        block: usize,
        decoder: &mut RelayBpDecoder,
        decoder_calls: &mut usize,
    ) -> Result<Option<Vec<QubitId>>, String> {
        let mut corrected = Array1::from_vec(syndrome.to_vec());
        let sigma = self.meta.dot(&corrected).mapv(|v| v % 2);
        if sigma.iter().any(|&b| b != 0) {
            *decoder_calls += 1;
            let repair = decoder.decode(&sigma.view()).map_err(|e| e.to_string())?;
            // The image check subsumes both decoder.converged and meta validity:
            // im H_Z is contained in ker M_Z.
            for (bit, error) in corrected.iter_mut().zip(repair.decoding) {
                *bit ^= error;
            }
        }
        Ok(self.solver.solve(&corrected.to_vec()).map(|x| {
            x.iter()
                .enumerate()
                .filter_map(|(q, &b)| (b != 0).then_some(QubitId(block * self.n + q)))
                .collect()
        }))
    }

    fn run_inner(
        &self,
        shots: usize,
        seed: u64,
        mut sink: impl FnMut(Outcome),
    ) -> Result<Counts, String> {
        let mut seeds = PecosRng::seed_from_u64(seed);
        let mut noise = PecosRng::seed_from_u64(seeds.next_u64());
        let mut sim = PhasePoly::with_seed(self.prefix.num_qubits(), seeds.next_u64());
        // Every numeric relay-BP setting is pinned to the current builder default, so a
        // library default change cannot silently change results. Relay gammas are sampled
        // from `gamma_dist_interval`.
        let mut decoder = RelayBpDecoder::builder(&self.meta.view())
            .error_priors(&self.priors)
            .max_iter(200)
            .alpha(None)
            .alpha_iteration_scaling_factor(1.0)
            .gamma0(Some(0.65))
            .pre_iter(80)
            .num_sets(300)
            .set_max_iter(60)
            .gamma_dist_interval((-0.24, 0.66))
            .stopping_criterion(StoppingCriterion::NConv { stop_after: 1 })
            .seed(seeds.next_u64())
            .build()
            .map_err(|e| e.to_string())?;
        let mut counts = Counts::default();
        for _ in 0..shots {
            counts.attempted += 1;
            let prefix = self
                .prefix
                .run_shot(&mut sim, &mut noise)
                .map_err(|e| e.to_string())?;
            let mut outcome = Outcome {
                syndromes: prefix
                    .records
                    .chunks(self.m_z)
                    .map(|s| s.iter().map(|&b| u8::from(b)).collect())
                    .collect(),
                decoder_calls: 0,
                repair_failure: false,
                readout: None,
                diagnostics: Vec::new(),
            };
            for (block, syndrome) in outcome.syndromes.iter().enumerate() {
                let Some(support) =
                    self.repair_block(syndrome, block, &mut decoder, &mut outcome.decoder_calls)?
                else {
                    outcome.repair_failure = true;
                    break;
                };
                sim.x(&support);
            }
            if outcome.repair_failure {
                counts.repair_failures += 1;
            } else {
                let readout = self
                    .suffix
                    .run_segment(&mut sim, &mut noise)
                    .map_err(|e| e.to_string())?;
                outcome.diagnostics = readout.records[self.diagnostics_start..].to_vec();
                let logical_failure = readout.observables.iter().any(|&b| b);
                outcome.readout = Some(readout);
                if outcome.accepted() {
                    counts.accepted += 1;
                    let residual_x = outcome.diagnostics.iter().any(|&b| b);
                    counts.logical_failures += usize::from(logical_failure);
                    counts.accepted_with_residual_x += usize::from(residual_x);
                    if !residual_x {
                        counts.accepted_z_clean += 1;
                        counts.logical_failures_z_clean += usize::from(logical_failure);
                    }
                }
            }
            sink(outcome);
        }
        Ok(counts)
    }
}

// Reduce [H_Z | I] once. The right half transforms each syndrome; rows below
// rank(H_Z) test image membership. Set all free variables to zero to select one solution.
struct SyndromeSolver {
    transform: Vec<Vec<u8>>,
    pivots: Vec<usize>,
    width: usize,
}

impl SyndromeSolver {
    fn new(matrix: &F2Matrix) -> Self {
        let width = matrix.num_cols();
        let rows = matrix
            .rows()
            .into_iter()
            .enumerate()
            .map(|(i, mut row)| {
                row.extend((0..matrix.num_rows()).map(|j| u8::from(i == j)));
                row
            })
            .collect();
        let (reduced, pivots) = F2Matrix::from_rows(rows).row_reduce();
        Self {
            transform: reduced.rows().iter().map(|r| r[width..].to_vec()).collect(),
            pivots: pivots.into_iter().take_while(|&p| p < width).collect(),
            width,
        }
    }

    fn solve(&self, syndrome: &[u8]) -> Option<Vec<u8>> {
        let rhs = self
            .transform
            .iter()
            .map(|row| row.iter().zip(syndrome).fold(0, |a, (&b, &c)| a ^ (b & c)))
            .collect::<Vec<_>>();
        if rhs[self.pivots.len()..].iter().any(|&b| b != 0) {
            return None;
        }
        let mut x = vec![0; self.width];
        for (&pivot, &bit) in self.pivots.iter().zip(&rhs) {
            x[pivot] = bit;
        }
        Some(x)
    }
}

fn preparation(code: &BivariateTricycleCode, p: f64, total: usize) -> Result<TickCircuit, String> {
    let n = code.num_qubits();
    let m_z = code.hz().num_checks();
    let mut circuit = TickCircuit::new();
    circuit.tick().px(&(0..3 * n).collect::<Vec<_>>());
    circuit.tick_at(0).pz(&(3 * n..total).collect::<Vec<_>>());
    let mut noise = circuit.tick();
    for q in 0..3 * n {
        noise.channel(channel::Dephasing(p, q));
    }
    for q in 3 * n..total {
        noise.channel(channel::BitFlip(p, q));
    }
    let edges = code
        .hz()
        .rows()
        .iter()
        .enumerate()
        .flat_map(|(check, row)| {
            row.iter()
                .enumerate()
                .filter_map(move |(data, &bit)| (bit != 0).then_some((data, check)))
        })
        .collect::<Vec<_>>();
    let coloring = bipartite_edge_coloring(n, m_z, &edges).map_err(|e| e.to_string())?;
    for color in 0..coloring.num_colors() {
        let pairs = (0..3)
            .flat_map(|block| {
                edges
                    .iter()
                    .zip(coloring.colors())
                    .filter(move |&(_, &c)| c == color)
                    .map(move |(&(data, check), _)| (block * n + data, 3 * n + block * m_z + check))
            })
            .collect::<Vec<_>>();
        circuit.tick().cx(&pairs);
        let mut noise = circuit.tick();
        for (data, ancilla) in pairs {
            noise.channel(channel::Depolarizing2(p, data, ancilla));
        }
    }
    let mut noise = circuit.tick();
    for q in 3 * n..total {
        noise.channel(channel::BitFlip(p, q));
    }
    Ok(circuit)
}

fn depolarizing3(p: f64, qubits: [usize; 3]) -> ChannelExpr {
    let paulis = [
        unitary_rep::I,
        unitary_rep::X,
        unitary_rep::Y,
        unitary_rep::Z,
    ];
    ChannelExpr::MixedUnitary(
        (0..64)
            .map(|digits| {
                let unitary = paulis[digits % 4](QubitId(qubits[0]))
                    & paulis[(digits / 4) % 4](QubitId(qubits[1]))
                    & paulis[digits / 16](QubitId(qubits[2]));
                (if digits == 0 { 1.0 - p } else { p / 63.0 }, unitary)
            })
            .collect(),
    )
}

fn correlated_z(p: f64, support: &[usize]) -> ChannelExpr {
    let identity = support
        .iter()
        .map(|&q| unitary_rep::I(q))
        .reduce(|a, b| a & b)
        .expect("nonempty CCZ support");
    let error = support
        .iter()
        .map(|&q| unitary_rep::Z(q))
        .reduce(|a, b| a & b)
        .expect("nonempty CCZ support");
    ChannelExpr::MixedUnitary(vec![(1.0 - p, identity), (p, error)])
}

fn surrogate_supports(code: &BivariateTricycleCode) -> Vec<Vec<usize>> {
    let n = code.num_qubits();
    let mut supports = (0..3 * n).map(|q| vec![q]).collect::<Vec<_>>();
    for [a, b, c] in code.cup_product_ccz().concat() {
        let triple = [a, n + b, 2 * n + c];
        for q in triple {
            supports[q].extend(triple);
        }
    }
    for support in &mut supports {
        support.sort_unstable();
        support.dedup();
    }
    supports
}

fn ccz_stage(circuit: &mut TickCircuit, code: &BivariateTricycleCode, p: f64, model: CczModel) {
    let n = code.num_qubits();
    match model {
        CczModel::Exact => {
            for layer in code.cup_product_ccz() {
                let triples = layer
                    .iter()
                    .map(|&[a, b, c]| (a, n + b, 2 * n + c))
                    .collect::<Vec<_>>();
                circuit.tick().ccz(&triples);
                let mut noise = circuit.tick();
                for (a, b, c) in triples {
                    noise.channel(depolarizing3(p, [a, b, c]));
                }
            }
        }
        CczModel::PaperSurrogate => {
            for support in surrogate_supports(code) {
                circuit.tick().channel(correlated_z(p, &support));
            }
        }
    }
}

#[cfg(test)]
#[derive(Clone, Copy, PartialEq, Eq)]
enum FaultPoint {
    BeforeAncillaMeasurement,
    BeforeCcz,
}

#[cfg(test)]
struct Fault {
    point: FaultPoint,
    qubit: usize,
    kind: pecos_core::gate_type::GateType,
}

#[cfg(test)]
fn append_faults(
    circuit: &mut TickCircuit,
    faults: &[Fault],
    point: FaultPoint,
    start: usize,
    end: usize,
) -> Result<(), String> {
    use pecos_core::gate_type::GateType;
    for fault in faults.iter().filter(|f| f.point == point) {
        if !(start..end).contains(&fault.qubit) {
            return Err("forced fault qubit outside the named region".into());
        }
        match fault.kind {
            GateType::X => {
                circuit.tick().x(&[fault.qubit]);
            }
            GateType::Y => {
                circuit.tick().y(&[fault.qubit]);
            }
            GateType::Z => {
                circuit.tick().z(&[fault.qubit]);
            }
            _ => return Err("forced fault must be X, Y, or Z".into()),
        }
    }
    Ok(())
}

fn readout(
    suffix: &mut TickCircuit,
    code: &BivariateTricycleCode,
    p: f64,
    model: CczModel,
) -> Result<usize, String> {
    let n = code.num_qubits();
    let encoder = Encoder::new(code);
    let mut noise = suffix.tick();
    for data in 0..3 * n {
        noise.channel(channel::Dephasing(p, data));
    }
    stage(suffix, n, &encoder.stabilizers, &encoder.stabilizer_pivots);
    stage(suffix, n, &encoder.logicals, &encoder.logical_pivots);
    // Only X-check pivots define detectors; the ideal-encoder benchmark in
    // tricycle_ccz also adds Z diagnostics as detectors.
    for block in 0..3 {
        let qubits = encoder
            .stabilizer_pivots
            .iter()
            .map(|q| block * n + q)
            .collect::<Vec<_>>();
        for measurement in suffix.tick().mx(&qubits) {
            suffix.detector(&[measurement]).map_err(|e| e.to_string())?;
        }
    }
    let layers = code.cup_product_ccz();
    let tensor = if model == CczModel::Exact {
        layers.concat()
    } else {
        Vec::new()
    };
    encoder.score(suffix, n, &tensor)?;
    let diagnostics_start = 3 * (encoder.stabilizer_pivots.len() + encoder.logical_pivots.len());
    for block in 0..3 {
        let qubits = encoder
            .non_pivots
            .iter()
            .map(|q| block * n + q)
            .collect::<Vec<_>>();
        suffix.tick().mz(&qubits);
    }
    Ok(diagnostics_start)
}
