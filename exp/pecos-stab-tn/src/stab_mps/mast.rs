// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! MAST: Magic state injection Augmented Stabilizer Tensor network.
//!
//! Instead of applying non-Clifford gates directly (which increases MPS bond
//! dimension), each non-Clifford gate is replaced by:
//!
//! 1. Prepare a magic state |+_T> on a fresh ancilla
//! 2. CNOT between ancilla and target (Clifford -- only touches tableau)
//! 3. Predetermine the uniformly random ancilla outcome and immediately apply
//!    its data-qubit correction
//! 4. Defer the ancilla projection until the end
//!
//! At the end of the circuit, all deferred measurements are performed.
//! The predetermined half-probability gadget outcomes are exact for the
//! untruncated state. Under MPS truncation they can differ from the truncated
//! representation's own outcome distribution; the forced outcome continues to
//! implement the exact, untruncated injection gadget.
//! For random circuits with t <= N, most projections are non-entangling,
//! keeping the MPS bond dimension bounded by ~3 on average.
//!
//! # References
//!
//! Nakhl et al., "Stabilizer Tensor Networks with Magic State Injection,"
//! PRL 134, 190602 (2025). arXiv:2411.12482.

use crate::errors::MpsError;
use crate::mps::{Mps, MpsConfig};
use num_complex::Complex64;
use pecos_core::{Angle64, QubitId};
use pecos_random::PecosRng;
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator, SparseStabY,
};

use super::non_clifford;

/// Order used to collapse deferred magic-state ancillas.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ProjectionOrder {
    /// Collapse in reverse injection order.
    Input,
    /// Recompute MPS-frame locality before every collapse and choose the
    /// smallest `(span, support size, injection index)` tuple. With `k`
    /// deferred injections, repeatedly recomputing locality costs O(k^2).
    #[default]
    MinSpan,
}

/// Diagnostics for one deferred magic-state ancilla projection.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ProjectionRecord {
    /// Ancilla qubit index in the expanded MAST system.
    pub ancilla: usize,
    /// Number of MPS sites in the conjugated observable's support.
    pub support_size: usize,
    /// Distance between the first and last supported MPS sites.
    pub mps_span: usize,
    /// Maximum MPS bond dimension immediately before projection.
    pub bond_before: usize,
    /// Maximum MPS bond dimension after projection and any correction.
    pub bond_after: usize,
}

/// A deferred ancilla measurement.
#[derive(Clone, Copy)]
struct DeferredMeasurement {
    /// The ancilla qubit index (in the expanded system).
    ancilla: usize,
    /// Uniform gadget outcome chosen when the magic state was injected.
    predetermined_outcome: bool,
    /// Zero-based position in the original injection sequence.
    injection_index: usize,
}

/// Magic-state-injection augmented stabilizer tensor-network simulator.
///
/// `max_non_clifford` preallocates one fresh ancilla slot per deferred
/// non-Clifford RZ. Exceeding that capacity panics. Replay the circuit through
/// [`super::compile::StabMpsCompile`] and call
/// [`super::compile::StabMpsCompile::advise`] to compute
/// `deferred_ancillas_required`; during execution,
/// [`Self::remaining_injections`] reports unused slots.
///
/// Non-Clifford ancilla projections remain deferred until completion. Their
/// uniformly random outcomes and corresponding corrections are handled at the
/// injection points. Call [`Self::project_all`] explicitly to project every
/// magic-state ancilla. Alternatively, measuring data through
/// [`CliffordGateable::mz`] calls `project_all()` first and then performs the
/// requested Z measurements. [`Self::flush`] only materializes pending merged
/// RZ rotations; it does not project already deferred injections.
///
/// The injection gadget's predetermined outcomes have exact probability 1/2
/// for the untruncated state. If truncation erases that deferred branch,
/// `project_all` continues on the surviving complement. That continuation is a
/// normalized approximate state whose injection-time correction does not match
/// the projected outcome: it is not a valid gadget trajectory for either
/// outcome. This additional bias is not represented by `truncation_error`;
/// `deferred_branch_lost_count` is its only witness. An untruncated
/// configuration never reaches this policy.
///
/// Data-qubit measurements use an exact sample-then-force route: evaluate the
/// physical Z expectation, sample that probability, then apply the normalized
/// forced projector for the drawn outcome. This deliberately avoids the
/// `StabMps` pragmatic measurement path, whose uncompensated tableau
/// pre-reduction is not composable with MAST's exact forced ancilla projections.
/// Exact compensation can apply long-range CNOTs to the MPS and therefore cost
/// more time and bond dimension. Lazy virtual-frame operations are materialized
/// before this exact route.
///
/// Prefer `Mast` for T-like, Clifford-correction workloads when sufficient
/// ancilla capacity is available and deferred projection keeps the coefficient
/// MPS small. Prefer [`super::StabMps`] for direct arbitrary rotations, limited
/// ancillary memory, or its amplitude, probability, and sampling read APIs.
pub struct Mast {
    /// Number of data qubits.
    num_data_qubits: usize,
    /// Seed supplied at construction, or `None` for entropy-backed resets.
    construction_seed: Option<u64>,
    /// Maximum number of non-Clifford gates (= number of ancilla slots).
    max_non_clifford: usize,
    /// Total qubits = data + ancillas.
    total_qubits: usize,
    /// The underlying stabilizer tableau for all qubits.
    tableau: SparseStabY,
    /// The MPS over all qubits.
    mps: Mps,
    config: MpsConfig,
    /// Next available ancilla index.
    next_ancilla: usize,
    /// Deferred measurements to perform at the end.
    deferred: Vec<DeferredMeasurement>,
    /// Policy for selecting the next deferred ancilla to project.
    projection_order: ProjectionOrder,
    /// Per-projection locality and bond-dimension diagnostics since reset.
    projection_records: Vec<ProjectionRecord>,
    /// Peak bond dimension observed before or after a deferred projection.
    projection_peak_bond: usize,
    global_phase: Complex64,
    disent_flags: Vec<Option<super::SiteEigenstate>>,
    numerical_flag_redetection: bool,
    gf2_matrix: super::ofd::Gf2FlipMatrix,
    rng: PecosRng,
    /// Runtime counters for non-Clifford decomposition paths.
    pub stats: super::StabMpsStats,
    /// Pending non-Clifford RZ angle per qubit when `merge_rz` is on.
    /// Flushed when any other gate touches the qubit (except RZ-same-qubit
    /// merges, Z/S/Sdg/CZ commutes). Mirror of `StabMps`'s field.
    pending_rz: Vec<Option<Angle64>>,
    /// When `true`, consecutive `rz(θ, q)` on same qubit merge before
    /// invoking magic-state injection. Big win for ion-trap RZ noise.
    merge_rz: bool,
}

impl Mast {
    /// Create a MAST simulator with `num_qubits` data qubits and room for
    /// `max_non_clifford` non-Clifford gates.
    ///
    /// # Panics
    ///
    /// Applying more injections than `max_non_clifford` panics. Use
    /// [`Self::remaining_injections`] to inspect capacity; compile-only
    /// [`super::compile::StabMpsCompile::advise`] reports the required deferred
    /// ancilla capacity for an analyzed circuit.
    #[must_use]
    pub fn new(num_qubits: usize, max_non_clifford: usize) -> Self {
        let total = num_qubits + max_non_clifford;
        let (tableau, rng) = super::initial_tableau_and_rng(total, None);
        Self {
            num_data_qubits: num_qubits,
            construction_seed: None,
            max_non_clifford,
            total_qubits: total,
            tableau,
            mps: Mps::new(total, MpsConfig::default()),
            config: MpsConfig::default(),
            next_ancilla: num_qubits,
            deferred: Vec::new(),
            projection_order: ProjectionOrder::default(),
            projection_records: Vec::new(),
            projection_peak_bond: 0,
            global_phase: Complex64::new(1.0, 0.0),
            disent_flags: vec![Some(super::SiteEigenstate::Z(false)); total],
            numerical_flag_redetection: false,
            gf2_matrix: super::ofd::Gf2FlipMatrix::new(total),
            rng,
            stats: super::StabMpsStats::default(),
            pending_rz: vec![None; total],
            merge_rz: false,
        }
    }

    /// Create with a specific seed for reproducible stochastic operations.
    ///
    /// Seeds both the simulator's [`pecos_random::PecosRng`] and its tableau.
    /// Identically configured fresh instances reproduce an identical sequence
    /// of predetermined gadget outcomes and measurements. On reset, both
    /// rebuilt RNGs are seeded from the current simulator stream, giving
    /// deterministic continuation rather than replaying this seed.
    ///
    /// # Panics
    ///
    /// Applying more injections than `max_non_clifford` panics. See
    /// [`Self::new`] for capacity-planning details.
    #[must_use]
    pub fn with_seed(num_qubits: usize, max_non_clifford: usize, seed: u64) -> Self {
        let total = num_qubits + max_non_clifford;
        let (tableau, rng) = super::initial_tableau_and_rng(total, Some(seed));
        Self {
            num_data_qubits: num_qubits,
            construction_seed: Some(seed),
            max_non_clifford,
            total_qubits: total,
            tableau,
            mps: Mps::new(total, MpsConfig::default()),
            config: MpsConfig::default(),
            next_ancilla: num_qubits,
            deferred: Vec::new(),
            projection_order: ProjectionOrder::default(),
            projection_records: Vec::new(),
            projection_peak_bond: 0,
            global_phase: Complex64::new(1.0, 0.0),
            disent_flags: vec![Some(super::SiteEigenstate::Z(false)); total],
            numerical_flag_redetection: false,
            gf2_matrix: super::ofd::Gf2FlipMatrix::new(total),
            rng,
            stats: super::StabMpsStats::default(),
            pending_rz: vec![None; total],
            merge_rz: false,
        }
    }

    /// Replace the coefficient-MPS configuration on a freshly constructed
    /// simulator.
    ///
    /// This fluent construction option reinitializes the coefficient MPS to
    /// `|0...0>` and is therefore intended to be called immediately after
    /// [`Self::new`] or [`Self::with_seed`], before applying gates. It is useful
    /// for correctness studies that disable truncation independently of the
    /// production defaults.
    #[must_use]
    pub fn with_mps_config(mut self, config: MpsConfig) -> Self {
        self.mps = Mps::new(self.total_qubits, config.clone());
        self.config = config;
        self
    }

    /// Enable RZ batching on the same qubit. See `StabMpsBuilder::merge_rz`
    /// for semantics. MAST defaults this to false so each RZ immediately makes
    /// its injection and ancilla-capacity cost visible; `StabMps` defaults it
    /// to true for throughput. Fluent-style setter on MAST.
    #[must_use]
    pub fn with_merge_rz(mut self, merge: bool) -> Self {
        self.merge_rz = merge;
        self
    }

    /// Numerically recover missing exact-disentangling |0> flags at product
    /// sites. Default: false.
    #[must_use]
    pub fn with_numerical_flag_redetection(mut self, enable: bool) -> Self {
        self.numerical_flag_redetection = enable;
        self
    }

    /// Select the ordering policy for deferred ancilla projections.
    ///
    /// The default is [`ProjectionOrder::MinSpan`]. Select
    /// [`ProjectionOrder::Input`] to preserve the legacy reverse-injection
    /// projection sequence.
    #[must_use]
    pub fn projection_order(mut self, projection_order: ProjectionOrder) -> Self {
        self.projection_order = projection_order;
        self
    }

    /// Flush any pending merged RZ on qubit `q` via magic-state injection.
    /// No-op when `merge_rz` is off or the slot is empty.
    fn flush_pending_rz(&mut self, q: usize) {
        if !self.merge_rz {
            return;
        }
        if let Some(theta) = self.pending_rz[q].take() {
            self.rz_apply_direct(theta, q);
        }
    }

    /// Apply `rz(theta)` on qubit `q` directly (without the merge buffer).
    /// Handles Clifford-angle shortcuts and MAST magic-state injection.
    fn rz_apply_direct(&mut self, theta: Angle64, q: usize) {
        if theta == Angle64::ZERO {
            return;
        }
        let qid = QubitId(q);
        if theta == Angle64::HALF_TURN {
            self.global_phase *= Complex64::new(0.0, -1.0);
            self.tableau.z(&[qid]);
            return;
        }
        if theta == Angle64::QUARTER_TURN {
            let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
            self.global_phase *= Complex64::new(inv_sqrt2, -inv_sqrt2);
            self.tableau.sz(&[qid]);
            return;
        }
        if theta == Angle64::THREE_QUARTERS_TURN {
            let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
            self.global_phase *= Complex64::new(inv_sqrt2, inv_sqrt2);
            self.tableau.szdg(&[qid]);
            return;
        }
        self.inject_magic_state(theta, q);
    }

    /// Materialize all pending merged RZ rotations. Public; useful before read
    /// operations.
    pub fn flush(&mut self) {
        if !self.merge_rz {
            return;
        }
        for q in 0..self.total_qubits {
            self.flush_pending_rz(q);
        }
    }

    #[must_use]
    /// Return the number of data qubits, excluding preallocated ancillas.
    pub fn num_data_qubits(&self) -> usize {
        self.num_data_qubits
    }

    #[must_use]
    /// Return the number of ancilla slots consumed by materialized injections.
    /// Pending merged rotations are not reflected and may consume slots later.
    pub fn num_ancillas_used(&self) -> usize {
        self.next_ancilla - self.num_data_qubits
    }

    /// Number of additional magic-state injections available before the
    /// configured `max_non_clifford` capacity is exhausted. Pending merged
    /// rotations are not reflected and may consume this capacity later.
    #[must_use]
    pub fn remaining_injections(&self) -> usize {
        self.max_non_clifford - self.num_ancillas_used()
    }

    #[must_use]
    /// Return the largest bond dimension currently stored in the coefficient MPS.
    /// Pending operations are not materialized by this diagnostic.
    pub fn max_bond_dim(&self) -> usize {
        self.mps.max_bond_dim()
    }

    /// Accumulated approximate infidelity from SVD truncation.
    /// Pending operations are not materialized by this diagnostic.
    #[must_use]
    pub fn truncation_error(&self) -> f64 {
        self.mps.truncation_error()
    }

    /// True sum of all relative discarded SVD weights over this run.
    #[must_use]
    pub fn summed_discarded_weight(&self) -> f64 {
        self.mps.summed_discarded_weight()
    }

    /// Largest coefficient-MPS bond dimension observed over this run.
    #[must_use]
    pub fn lifetime_peak_bond(&self) -> usize {
        self.mps.lifetime_peak_bond()
    }

    /// Number of sampled data projections retried after branch vanish.
    #[must_use]
    pub fn branch_vanish_retry_count(&self) -> u64 {
        self.mps.branch_vanish_retry_count()
    }

    /// Number of deferred gadget branches replaced by their complement.
    #[must_use]
    pub fn deferred_branch_lost_count(&self) -> u64 {
        self.mps.deferred_branch_lost_count()
    }

    /// Number of SVDs where `max_bond_dim` was the binding cap.
    /// Pending operations are not materialized by this diagnostic.
    #[must_use]
    pub fn bond_cap_hits(&self) -> u64 {
        self.mps.bond_cap_hits()
    }

    #[must_use]
    /// Borrow the coefficient MPS over data qubits and preallocated ancillas.
    ///
    /// Call [`Self::flush`] first if pending merged rotations must be included,
    /// and [`Self::project_all`] first if deferred injections must be completed.
    pub fn mps(&self) -> &Mps {
        &self.mps
    }

    /// Return stored diagnostics for deferred projections performed since reset.
    /// Pending operations are not materialized by this diagnostic.
    #[must_use]
    pub fn projection_records(&self) -> &[ProjectionRecord] {
        &self.projection_records
    }

    /// Return the peak MPS bond dimension observed during deferred projection.
    ///
    /// Returns zero when no deferred projection has run since reset. Pending
    /// operations are not materialized by this diagnostic.
    #[must_use]
    pub fn projection_peak_bond(&self) -> usize {
        self.projection_peak_bond
    }

    /// Inject a magic state for RZ(theta) on the target qubit.
    ///
    /// Magic state teleportation protocol:
    /// 1. Prepare ancilla in |+>: H on ancilla
    /// 2. Apply RZ(theta) on ancilla (local, single-site MPS gate)
    /// 3. CNOT(target, ancilla) -- **target controls, ancilla is CX target**
    /// 4. Predetermine the uniformly random ancilla outcome
    /// 5. If it is 1, immediately apply RZ(2*theta) to the data qubit
    /// 6. Defer projection of the ancilla onto the predetermined outcome
    ///
    /// When the ancilla is later measured:
    /// - Outcome 0: data qubit has RZ(theta) applied. Done.
    /// - Outcome 1: data qubit has RZ(-theta). Correction: RZ(2*theta) on data.
    ///   For T gate (theta=pi/4): correction = S = RZ(pi/2), which is Clifford.
    fn inject_magic_state(&mut self, theta: Angle64, target: usize) {
        assert!(
            self.next_ancilla < self.total_qubits,
            "exceeded max_non_clifford ancilla slots"
        );

        // The injection-gadget measurement is exactly uniform and independent
        // of the data state, so its outcome may be chosen at the injection
        // point while the physical ancilla projection remains deferred.
        let predetermined_outcome = self.rng.random_bool(0.5);

        let ancilla = self.next_ancilla;
        self.next_ancilla += 1;

        let anc_qid = QubitId(ancilla);
        let tgt_qid = QubitId(target);

        // Step 1: Prepare ancilla in |+>
        self.tableau.h(&[anc_qid]);

        // Step 2: Apply RZ(theta) on the ancilla.
        // Ancilla is in |+> (product state), so Z_anc is a destabilizer flip
        // at the ancilla site -- single-site gate, no bond dim growth.
        let half_rad = theta.to_radians_signed() / 2.0;
        let cos_half = half_rad.cos();
        let sin_half = half_rad.sin();
        super::expect_mps_operation(
            non_clifford::apply_rz_stab_mps(
                &mut self.tableau,
                &mut self.mps,
                cos_half,
                sin_half,
                ancilla,
                true,
                &mut non_clifford::RzContext {
                    disent_flags: &mut self.disent_flags,
                    deferred_ops: &[],
                    numerical_flag_redetection: self.numerical_flag_redetection,
                    gf2_matrix: &mut self.gf2_matrix,
                    stats: &mut self.stats,
                    saturation_telemetry: None,
                },
            ),
            "Mast::inject_magic_state RZ update",
        );

        // Step 3: CNOT(target, ancilla) -- target controls, ancilla is CX target
        // This is the key: data qubit controls, ancilla flips.
        self.tableau.cx(&[(tgt_qid, anc_qid)]);

        // Step 4: Apply the branch correction at the injection point. Delaying
        // this gate would require conjugating it through every later gate on
        // the data qubit.
        if predetermined_outcome {
            self.apply_injection_correction(theta + theta, target);
        }

        // Step 5: Record only the predetermined ancilla projection.
        self.deferred.push(DeferredMeasurement {
            ancilla,
            predetermined_outcome,
            injection_index: ancilla - self.num_data_qubits,
        });
    }

    /// Apply a predetermined magic-state-injection correction immediately.
    fn apply_injection_correction(&mut self, correction_angle: Angle64, target: usize) {
        let tgt = QubitId(target);

        if correction_angle == Angle64::ZERO {
            // No correction needed.
        } else if correction_angle == Angle64::HALF_TURN {
            // RZ(pi) = -iZ.
            self.global_phase *= Complex64::new(0.0, -1.0);
            self.tableau.z(&[tgt]);
        } else if correction_angle == Angle64::QUARTER_TURN {
            // RZ(pi/2) = e^{-i*pi/4} S -- the T-gate correction.
            let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
            self.global_phase *= Complex64::new(inv_sqrt2, -inv_sqrt2);
            self.tableau.sz(&[tgt]);
        } else if correction_angle == Angle64::THREE_QUARTERS_TURN {
            let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
            self.global_phase *= Complex64::new(inv_sqrt2, inv_sqrt2);
            self.tableau.szdg(&[tgt]);
        } else {
            // Non-Clifford correction: apply via the STN protocol.
            let (sin_half, cos_half) = correction_angle.half_angle_sin_cos();
            super::expect_mps_operation(
                non_clifford::apply_rz_stab_mps(
                    &mut self.tableau,
                    &mut self.mps,
                    cos_half,
                    sin_half,
                    target,
                    true,
                    &mut non_clifford::RzContext {
                        disent_flags: &mut self.disent_flags,
                        deferred_ops: &[],
                        numerical_flag_redetection: self.numerical_flag_redetection,
                        gf2_matrix: &mut self.gf2_matrix,
                        stats: &mut self.stats,
                        saturation_telemetry: None,
                    },
                ),
                "Mast::apply_injection_correction RZ update",
            );
        }
    }

    /// Project all deferred ancillas onto their predetermined outcomes.
    ///
    /// For each deferred ancilla:
    /// 1. Force-project the ancilla in Z using the shared normalized STN
    ///    projection protocol
    /// 2. Apply no correction; it was already applied at injection time
    ///
    /// Calling `mz` on data qubits performs this completion step automatically.
    pub fn project_all(&mut self) {
        let result = (|| -> Result<(), MpsError> {
            match self.projection_order {
                ProjectionOrder::Input => {
                    // Preserve the original drain and reverse-iteration path.
                    let deferred: Vec<DeferredMeasurement> =
                        self.deferred.drain(..).rev().collect();
                    for dm in deferred {
                        let (support_size, mps_span) = self.projection_locality(dm.ancilla);
                        self.project_deferred(dm, support_size, mps_span)?;
                    }
                }
                ProjectionOrder::MinSpan => {
                    while !self.deferred.is_empty() {
                        let mut selected = (0, usize::MAX, usize::MAX, usize::MAX);
                        for (position, dm) in self.deferred.iter().enumerate() {
                            let (support_size, mps_span) = self.projection_locality(dm.ancilla);
                            let candidate = (position, mps_span, support_size, dm.injection_index);
                            if (candidate.1, candidate.2, candidate.3)
                                < (selected.1, selected.2, selected.3)
                            {
                                selected = candidate;
                            }
                        }
                        let dm = self.deferred.remove(selected.0);
                        self.project_deferred(dm, selected.2, selected.1)?;
                    }
                }
            }
            Ok(())
        })();
        super::expect_mps_operation(result, "Mast::project_all deferred projection");
    }

    fn projection_locality(&self, ancilla: usize) -> (usize, usize) {
        let support = super::measure::conjugated_z_support(&self.tableau, ancilla, &[]);
        let span = support
            .first()
            .zip(support.last())
            .map_or(0, |(first, last)| last - first);
        (support.len(), span)
    }

    fn project_deferred(
        &mut self,
        dm: DeferredMeasurement,
        support_size: usize,
        mps_span: usize,
    ) -> Result<(), MpsError> {
        let bond_before = self.mps.max_bond_dim();
        self.projection_peak_bond = self.projection_peak_bond.max(bond_before);

        // The branch correction was applied at injection time. Try the
        // predetermined branch transactionally so a vanished attempt cannot
        // mutate the live tableau/MPS pair.
        let mut candidate_tableau = self.tableau.clone();
        let mut candidate_mps = self.mps.clone();
        let mut projection = super::measure::project_forced_z_with_update(
            &mut candidate_tableau,
            &mut candidate_mps,
            dm.ancilla,
            dm.predetermined_outcome,
        )?;
        let branch_lost = projection.snapped_probability == 0.0
            || projection.survival_ratio < super::measure::BRANCH_VANISH_SURVIVAL_THRESHOLD;
        debug_assert!(
            !branch_lost,
            "Mast::project_all predetermined deferred branch was lost"
        );
        if branch_lost {
            candidate_tableau = self.tableau.clone();
            candidate_mps = self.mps.clone();
            let original_config = self.mps.config().clone();
            let mut retry_config = original_config.clone();
            retry_config.max_bond_dim = candidate_mps.physical_rank_ceiling();
            retry_config.svd_cutoff = 0.0;
            retry_config.max_truncation_error = Some(0.0);
            candidate_mps.set_config(retry_config);
            projection = super::measure::project_forced_z_with_update(
                &mut candidate_tableau,
                &mut candidate_mps,
                dm.ancilla,
                !dm.predetermined_outcome,
            )?;
            let complement_lost = projection.snapped_probability == 0.0
                || projection.survival_ratio < super::measure::BRANCH_VANISH_SURVIVAL_THRESHOLD;
            debug_assert!(
                !complement_lost,
                "Mast::project_all complement deferred branch was also lost"
            );
            assert!(
                !complement_lost,
                "Mast::project_all complement deferred branch was also lost"
            );
            candidate_mps.set_config(original_config);
            candidate_mps.record_deferred_branch_lost();
        }
        self.tableau = candidate_tableau;
        self.mps = candidate_mps;
        super::repair_disent_flags(&self.mps, &mut self.disent_flags, &projection.update);

        let bond_after = self.mps.max_bond_dim();
        self.projection_peak_bond = self.projection_peak_bond.max(bond_after);
        self.projection_records.push(ProjectionRecord {
            ancilla: dm.ancilla,
            support_size,
            mps_span,
            bond_before,
            bond_after,
        });
        Ok(())
    }

    /// Evaluate a physical data-qubit Z probability, then apply the normalized
    /// forced projector to the live state for the sampled outcome.
    fn measure_data_qubit_exact(&mut self, q_idx: usize) -> Result<MeasurementResult, MpsError> {
        let live = super::measure_qubit_exact_transactional(
            &mut self.tableau,
            &mut self.mps,
            &mut self.rng,
            q_idx,
            "Mast::mz data-qubit projection",
        )?;
        super::repair_disent_flags(&self.mps, &mut self.disent_flags, &live.update);
        Ok(live.measurement)
    }
}

impl QuantumSimulator for Mast {
    /// Reset data, ancillas, capacity use, and diagnostics as if newly
    /// constructed with the retained configuration.
    ///
    /// For a seeded simulator, the rebuilt tableau seed and continuing
    /// simulator-RNG seed are drawn from the current simulator stream. This is
    /// deterministic continuation, not replay of the construction stream. An
    /// unseeded simulator obtains fresh entropy.
    fn reset(&mut self) -> &mut Self {
        (self.tableau, self.rng) = super::reset_tableau_and_rng(
            self.total_qubits,
            self.construction_seed.is_some(),
            &mut self.rng,
        );
        self.mps = Mps::new(self.total_qubits, self.config.clone());
        self.next_ancilla = self.num_data_qubits;
        self.deferred.clear();
        self.projection_records.clear();
        self.projection_peak_bond = 0;
        self.global_phase = Complex64::new(1.0, 0.0);
        self.disent_flags = vec![Some(super::SiteEigenstate::Z(false)); self.total_qubits];
        self.gf2_matrix.reset();
        self.stats = super::StabMpsStats::default();
        for slot in &mut self.pending_rz {
            *slot = None;
        }
        self
    }

    fn num_qubits(&self) -> usize {
        self.num_data_qubits
    }
}

impl CliffordGateable for Mast {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.tableau.sz(qubits);
        self
    }

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        // H does not commute with RZ: flush pending merged RZ first.
        for &q in qubits {
            self.flush_pending_rz(q.index());
        }
        self.tableau.h(qubits);
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // CX doesn't commute with RZ on arbitrary qubits: flush both.
        for &(c, t) in pairs {
            self.flush_pending_rz(c.index());
            self.flush_pending_rz(t.index());
        }
        self.tableau.cx(pairs);
        self
    }

    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // CZ is diagonal, commutes with RZ on either qubit — no flush needed.
        self.tableau.cz(pairs);
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        // Flush any pending merged RZ on measured qubits before measuring.
        for &q in qubits {
            self.flush_pending_rz(q.index());
        }
        // Project all deferred measurements first
        self.project_all();
        // Then sample and force-project each data qubit through the exact route.
        let mut measurements = Vec::with_capacity(qubits.len());
        for &q in qubits {
            measurements.push(super::expect_mps_operation(
                self.measure_data_qubit_exact(q.index()),
                "Mast::mz data-qubit projection",
            ));
        }
        debug_assert_eq!(measurements.len(), qubits.len());
        measurements
    }
}

impl ArbitraryRotationGateable for Mast {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        self.h(qubits);
        self.rz(theta, qubits);
        self.h(qubits);
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let q_idx = q.index();
            if !self.merge_rz {
                self.rz_apply_direct(theta, q_idx);
                continue;
            }
            let is_clifford_angle = theta == Angle64::ZERO
                || theta == Angle64::HALF_TURN
                || theta == Angle64::QUARTER_TURN
                || theta == Angle64::THREE_QUARTERS_TURN;
            if is_clifford_angle {
                // Clifford-angle RZ commutes with pending non-Clifford RZ;
                // no flush needed, apply directly.
                self.rz_apply_direct(theta, q_idx);
            } else {
                let prev = self.pending_rz[q_idx].unwrap_or(Angle64::ZERO);
                let merged = prev + theta;
                if merged == Angle64::ZERO
                    || merged == Angle64::HALF_TURN
                    || merged == Angle64::QUARTER_TURN
                    || merged == Angle64::THREE_QUARTERS_TURN
                {
                    self.pending_rz[q_idx] = None;
                    self.rz_apply_direct(merged, q_idx);
                } else {
                    self.pending_rz[q_idx] = Some(merged);
                }
            }
        }
        self
    }

    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            self.cx(&[(q0, q1)]);
            self.rz(theta, &[q1]);
            self.cx(&[(q0, q1)]);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn assert_mast_disent_flags_sound(mast: &Mast, context: &str) {
        super::super::assert_disent_flags_match_stored_mps(&mast.mps, &mast.disent_flags, context);
    }

    fn project_next_and_assert(mast: &mut Mast, context: &str) {
        let (position, support_size, mps_span) = match mast.projection_order {
            ProjectionOrder::Input => {
                let position = mast.deferred.len() - 1;
                let (support_size, mps_span) =
                    mast.projection_locality(mast.deferred[position].ancilla);
                (position, support_size, mps_span)
            }
            ProjectionOrder::MinSpan => {
                let mut selected = (0, usize::MAX, usize::MAX, usize::MAX);
                for (position, dm) in mast.deferred.iter().enumerate() {
                    let (support_size, mps_span) = mast.projection_locality(dm.ancilla);
                    let candidate = (position, mps_span, support_size, dm.injection_index);
                    if (candidate.1, candidate.2, candidate.3)
                        < (selected.1, selected.2, selected.3)
                    {
                        selected = candidate;
                    }
                }
                (selected.0, selected.2, selected.1)
            }
        };
        let dm = mast.deferred.remove(position);
        mast.project_deferred(dm, support_size, mps_span).unwrap();
        assert_mast_disent_flags_sound(mast, context);
    }

    #[test]
    fn test_mast_mz_returns_one_result_per_requested_qubit() {
        let mut mast = Mast::with_seed(3, 0, 0x000A_11CE);
        mast.h(&[QubitId(0)]);
        mast.cx(&[(QubitId(0), QubitId(1))]);
        let qubits = [QubitId(0), QubitId(1), QubitId(2)];

        let measurements = mast.mz(&qubits);

        assert_eq!(measurements.len(), qubits.len());
    }

    #[test]
    fn test_mast_projection_and_measurement_disent_flags_match_marginals() {
        let t = Angle64::QUARTER_TURN / 2u64;
        for projection_order in [ProjectionOrder::Input, ProjectionOrder::MinSpan] {
            for numerical_flag_redetection in [false, true] {
                let mut mast = Mast::with_seed(4, 4, 0x7000_0000)
                    .with_merge_rz(false)
                    .with_numerical_flag_redetection(numerical_flag_redetection)
                    .projection_order(projection_order);
                mast.h(&[QubitId(0), QubitId(2)]);
                mast.cx(&[(QubitId(0), QubitId(1)), (QubitId(2), QubitId(3))]);
                mast.rz(t, &[QubitId(0)]);
                mast.cz(&[(QubitId(1), QubitId(2))]);
                mast.rz(t, &[QubitId(2)]);
                mast.h(&[QubitId(1)]);
                mast.rz(t, &[QubitId(1)]);

                let mut projection = 0;
                while !mast.deferred.is_empty() {
                    project_next_and_assert(
                        &mut mast,
                        &format!(
                            "MAST deferred projection {projection}; order={projection_order:?} redetect={numerical_flag_redetection}"
                        ),
                    );
                    projection += 1;
                }

                let _ = mast.mz(&[QubitId(3)]);
                assert_mast_disent_flags_sound(
                    &mast,
                    &format!(
                        "MAST data measurement; order={projection_order:?} redetect={numerical_flag_redetection}"
                    ),
                );
            }
        }
    }

    #[test]
    #[ignore = "dense 8-site MAST reconstruction regression; run explicitly with --release"]
    fn test_mast_disent_flag_projection_continuation_matches_dense() {
        use crate::stab_mps::StabMps;
        use pecos_simulators::DenseStateVec;

        #[derive(Clone, Copy, Debug)]
        enum Op {
            H(usize),
            Sz(usize),
            Cx(usize, usize),
            Cz(usize, usize),
            Rz(usize, f64),
        }

        fn next(state: &mut u64) -> u64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        }

        fn random_distinct_pair(state: &mut u64, n: usize) -> (usize, usize) {
            let first = (next(state) % n as u64) as usize;
            let mut second = (next(state) % (n - 1) as u64) as usize;
            if second >= first {
                second += 1;
            }
            (first, second)
        }

        fn apply(op: Op, mast: &mut Mast, dense: &mut DenseStateVec) {
            match op {
                Op::H(q) => {
                    mast.h(&[QubitId(q)]);
                    dense.h(&[QubitId(q)]);
                }
                Op::Sz(q) => {
                    mast.sz(&[QubitId(q)]);
                    dense.sz(&[QubitId(q)]);
                }
                Op::Cx(control, target) => {
                    let pair = [(QubitId(control), QubitId(target))];
                    mast.cx(&pair);
                    dense.cx(&pair);
                }
                Op::Cz(first, second) => {
                    let pair = [(QubitId(first), QubitId(second))];
                    mast.cz(&pair);
                    dense.cz(&pair);
                }
                Op::Rz(q, radians) => {
                    let angle = Angle64::from_radians(radians);
                    mast.rz(angle, &[QubitId(q)]);
                    dense.rz(angle, &[QubitId(q)]);
                }
            }
        }

        fn fidelity(first: &[Complex64], second: &[Complex64]) -> f64 {
            first
                .iter()
                .zip(second)
                .map(|(a, b)| a.conj() * b)
                .sum::<Complex64>()
                .norm_sqr()
        }

        fn project_z(state: &[Complex64], qubit: usize, outcome: bool) -> Vec<Complex64> {
            let probability = state
                .iter()
                .enumerate()
                .filter(|(index, _)| ((*index >> qubit) & 1 != 0) == outcome)
                .map(|(_, amplitude)| amplitude.norm_sqr())
                .sum::<f64>();
            assert!(probability > 1e-14);
            let scale = probability.sqrt().recip();
            state
                .iter()
                .enumerate()
                .map(|(index, &amplitude)| {
                    if ((index >> qubit) & 1 != 0) == outcome {
                        amplitude * scale
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                })
                .collect()
        }

        fn mast_data_state_vector(mast: &Mast) -> Vec<Complex64> {
            let mut view = StabMps::builder(mast.total_qubits).merge_rz(false).build();
            view.tableau = mast.tableau.clone();
            view.mps = mast.mps.clone();
            view.global_phase = mast.global_phase;
            let full = view.state_vector();
            let data_dimension = 1_usize << mast.num_data_qubits;
            let mut best = Vec::new();
            let mut best_norm = 0.0;
            let mut total_norm = 0.0;
            for block in full.chunks_exact(data_dimension) {
                let norm = block
                    .iter()
                    .map(num_complex::Complex::norm_sqr)
                    .sum::<f64>();
                total_norm += norm;
                if norm > best_norm {
                    best_norm = norm;
                    best = block.to_vec();
                }
            }
            // The dense tableau projector used only by this test becomes
            // mildly ill-conditioned on the exact-mz continuation circuit.
            assert!(best_norm > 0.99, "ancillas did not factor: {best_norm}");
            assert!((total_norm - 1.0).abs() < 1e-8);
            let scale = best_norm.sqrt().recip();
            for amplitude in &mut best {
                *amplitude *= scale;
            }
            best
        }

        fn exact_config() -> MpsConfig {
            MpsConfig {
                max_bond_dim: 64,
                svd_cutoff: 0.0,
                max_truncation_error: Some(0.0),
                parallel: false,
                direction_alternating_compression: false,
            }
        }

        const N: usize = 4;
        for exact_data_measurement in [false, true] {
            for numerical_flag_redetection in [false, true] {
                let (circuit_seed, measured_qubit, continuation_seed) = if exact_data_measurement {
                    (9_u64, 3_usize, 5_u64)
                } else {
                    (11, 0, 4)
                };
                let mut random = circuit_seed + 1;
                let mut mast = Mast::with_seed(N, 4, 0x6000_0000 + circuit_seed)
                    .with_mps_config(exact_config())
                    .with_merge_rz(false)
                    .with_numerical_flag_redetection(numerical_flag_redetection);
                let mut dense = DenseStateVec::new(N);
                let mut injections = 0;
                for step in 0..18 {
                    let choice = next(&mut random) % 8;
                    let q = (next(&mut random) % N as u64) as usize;
                    let op = match choice {
                        0 | 1 => Op::H(q),
                        2 => Op::Sz(q),
                        3 | 4 => {
                            let (control, target) = random_distinct_pair(&mut random, N);
                            Op::Cx(control, target)
                        }
                        5 => {
                            let (first, second) = random_distinct_pair(&mut random, N);
                            Op::Cz(first, second)
                        }
                        _ if injections < 3 => {
                            injections += 1;
                            Op::Rz(q, if step & 1 == 0 { 0.37 } else { -0.61 })
                        }
                        _ => Op::H(q),
                    };
                    apply(op, &mut mast, &mut dense);
                }
                assert_eq!(injections, 3);
                let dense_before_projection = dense.state();

                let expected = if exact_data_measurement {
                    let result = mast
                        .mz(&[QubitId(measured_qubit)])
                        .into_iter()
                        .next()
                        .expect("one measurement result");
                    project_z(&dense_before_projection, measured_qubit, result.outcome)
                } else {
                    mast.project_all();
                    dense_before_projection
                };
                let post_projection_fidelity = fidelity(&mast_data_state_vector(&mast), &expected);
                assert!(post_projection_fidelity > 1.0 - 1e-9);

                let mut oracle = DenseStateVec::from_state(
                    &expected,
                    PecosRng::seed_from_u64(continuation_seed),
                );
                let mut continuation_random = continuation_seed + 1;
                for _ in 0..4 {
                    let choice = next(&mut continuation_random) % 3;
                    let q = (next(&mut continuation_random) % N as u64) as usize;
                    let op = match choice {
                        0 => Op::H(q),
                        1 => Op::Sz(q),
                        _ => {
                            let (control, target) =
                                random_distinct_pair(&mut continuation_random, N);
                            Op::Cx(control, target)
                        }
                    };
                    apply(op, &mut mast, &mut oracle);
                }
                let target = (next(&mut continuation_random) % N as u64) as usize;
                apply(Op::Rz(target, 0.37), &mut mast, &mut oracle);
                mast.project_all();

                let actual = mast_data_state_vector(&mast);
                let expected_continued = oracle.state();
                let continued_fidelity = fidelity(&actual, &expected_continued);
                let max_probability_error = actual
                    .iter()
                    .zip(&expected_continued)
                    .map(|(a, b)| (a.norm_sqr() - b.norm_sqr()).abs())
                    .fold(0.0_f64, f64::max);
                eprintln!(
                    "MAST continuation: exact_mz={exact_data_measurement} \
                     redetect={numerical_flag_redetection} seed={circuit_seed} \
                     fidelity={continued_fidelity:.16} \
                     max_probability_error={max_probability_error:.3e} stats={:?}",
                    mast.stats
                );
                let (minimum_fidelity, maximum_probability_error) = if exact_data_measurement {
                    (1.0 - 1e-9, 1e-9)
                } else {
                    (1.0 - 1e-9, 1e-6)
                };
                assert!(continued_fidelity > minimum_fidelity);
                assert!(max_probability_error < maximum_probability_error);
            }
        }
    }

    #[test]
    fn test_mast_pure_clifford() {
        // Pure Clifford circuit should work like STN
        let mut mast = Mast::new(2, 4);
        mast.h(&[QubitId(0)]);
        mast.cx(&[(QubitId(0), QubitId(1))]);
        assert_eq!(mast.num_ancillas_used(), 0);
        assert_eq!(mast.max_bond_dim(), 1);
    }

    #[test]
    fn test_mast_single_t_gate() {
        // T gate uses magic state injection
        let mut mast = Mast::new(1, 4);
        assert_eq!(mast.remaining_injections(), 4);
        mast.h(&[QubitId(0)]);
        mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        assert_eq!(mast.num_ancillas_used(), 1);
        assert_eq!(mast.remaining_injections(), 3);
        // Bond dim should be low -- the RZ on the ancilla is a single-site gate
        assert!(
            mast.max_bond_dim() <= 2,
            "bond dim should be low, got {}",
            mast.max_bond_dim()
        );
        mast.reset();
        assert_eq!(mast.remaining_injections(), 4);
    }

    #[test]
    fn test_mast_norm_preserved() {
        let mut mast = Mast::new(2, 4);
        mast.h(&[QubitId(0)]);
        mast.cx(&[(QubitId(0), QubitId(1))]);
        mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(1)]);

        assert_relative_eq!(mast.mps().norm_squared(), 1.0, epsilon = 1e-8);
    }

    fn apply_seeded_projection_regression_circuit(mast: &mut Mast) {
        let t = Angle64::QUARTER_TURN / 2u64;
        mast.h(&[QubitId(0), QubitId(2)]);
        mast.cx(&[(QubitId(0), QubitId(1))]);
        mast.rz(t, &[QubitId(0)]);
        mast.h(&[QubitId(1)]);
        mast.rz(t, &[QubitId(2)]);
        mast.cx(&[(QubitId(2), QubitId(0))]);
        mast.rz(t, &[QubitId(1)]);
        mast.rz(t, &[QubitId(0)]);
    }

    fn project_all_legacy(mast: &mut Mast) {
        let deferred: Vec<DeferredMeasurement> = mast.deferred.drain(..).rev().collect();
        for dm in deferred {
            let (support_size, mps_span) = mast.projection_locality(dm.ancilla);
            mast.project_deferred(dm, support_size, mps_span).unwrap();
        }
    }

    #[test]
    fn test_mast_default_is_min_span_and_input_preserves_legacy_path() {
        let mut default_path = Mast::with_seed(3, 4, 0x51_7a);
        assert_eq!(default_path.projection_order, ProjectionOrder::MinSpan);
        assert_eq!(default_path.mps().config().max_bond_dim, 128);
        assert_eq!(default_path.mps().config().max_truncation_error, Some(1e-8));
        let mut explicit_min_span =
            Mast::with_seed(3, 4, 0x51_7a).projection_order(ProjectionOrder::MinSpan);
        let mut explicit_input =
            Mast::with_seed(3, 4, 0x51_7a).projection_order(ProjectionOrder::Input);
        let mut legacy_path = Mast::with_seed(3, 4, 0x51_7a);
        apply_seeded_projection_regression_circuit(&mut default_path);
        apply_seeded_projection_regression_circuit(&mut explicit_min_span);
        apply_seeded_projection_regression_circuit(&mut explicit_input);
        apply_seeded_projection_regression_circuit(&mut legacy_path);

        default_path.project_all();
        explicit_min_span.project_all();
        explicit_input.project_all();
        project_all_legacy(&mut legacy_path);

        assert_eq!(
            explicit_input
                .projection_records()
                .iter()
                .map(|record| record.ancilla)
                .collect::<Vec<_>>(),
            vec![6, 5, 4, 3]
        );
        assert_eq!(
            default_path.mps().state_vector(),
            explicit_min_span.mps().state_vector()
        );
        assert_eq!(
            explicit_input.projection_records(),
            legacy_path.projection_records()
        );
        assert_eq!(
            explicit_input.mps().state_vector(),
            legacy_path.mps().state_vector()
        );

        let default_outcomes: Vec<bool> = default_path
            .mz(&[QubitId(0), QubitId(1), QubitId(2)])
            .into_iter()
            .map(|result| result.outcome)
            .collect();
        let min_span_outcomes: Vec<bool> = explicit_min_span
            .mz(&[QubitId(0), QubitId(1), QubitId(2)])
            .into_iter()
            .map(|result| result.outcome)
            .collect();
        let input_outcomes: Vec<bool> = explicit_input
            .mz(&[QubitId(0), QubitId(1), QubitId(2)])
            .into_iter()
            .map(|result| result.outcome)
            .collect();
        let legacy_outcomes: Vec<bool> = legacy_path
            .mz(&[QubitId(0), QubitId(1), QubitId(2)])
            .into_iter()
            .map(|result| result.outcome)
            .collect();
        assert_eq!(default_outcomes, min_span_outcomes);
        assert_eq!(input_outcomes, legacy_outcomes);
    }

    #[test]
    fn test_mast_mps_config_and_truncation_telemetry() {
        let config = MpsConfig {
            max_bond_dim: 7,
            max_truncation_error: Some(0.0),
            ..MpsConfig::default()
        };
        let mut mast = Mast::new(2, 1).with_mps_config(config);

        assert_eq!(mast.mps().config().max_bond_dim, 7);
        assert_eq!(mast.mps().config().max_truncation_error, Some(0.0));
        assert_relative_eq!(mast.truncation_error(), 0.0, epsilon = f64::EPSILON);
        assert_eq!(mast.bond_cap_hits(), 0);

        mast.mps.record_truncation(0.25, true);
        assert_relative_eq!(mast.truncation_error(), 0.25, epsilon = f64::EPSILON);
        assert_eq!(mast.bond_cap_hits(), 1);

        mast.reset();
        assert_eq!(mast.mps().config().max_bond_dim, 7);
        assert_eq!(mast.mps().config().max_truncation_error, Some(0.0));
        assert_relative_eq!(mast.truncation_error(), 0.0, epsilon = f64::EPSILON);
        assert_eq!(mast.bond_cap_hits(), 0);
    }

    #[test]
    fn test_mast_projection_diagnostics_populated_and_reset() {
        for order in [ProjectionOrder::Input, ProjectionOrder::MinSpan] {
            let mut mast = Mast::with_seed(3, 4, 91).projection_order(order);
            apply_seeded_projection_regression_circuit(&mut mast);
            mast.project_all();

            assert_eq!(mast.projection_records().len(), 4);
            assert!(
                mast.projection_records()
                    .iter()
                    .all(|record| record.bond_before > 0 && record.bond_after > 0)
            );
            let recorded_peak = mast
                .projection_records()
                .iter()
                .map(|record| record.bond_before.max(record.bond_after))
                .max()
                .expect("four projection records");
            assert_eq!(mast.projection_peak_bond(), recorded_peak);

            mast.reset();
            assert!(mast.projection_records().is_empty());
            assert_eq!(mast.projection_peak_bond(), 0);
        }
    }

    #[test]
    fn test_mast_seeded_reset_continues_measurements_and_clears_diagnostics() {
        let run = |mast: &mut Mast| {
            mast.h(&[QubitId(0)]);
            mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
            mast.mz(&[QubitId(0)])[0].outcome
        };

        let collect = || {
            let mut mast = Mast::with_seed(1, 2, 0x5eed);
            let mut outcomes = Vec::with_capacity(200);
            for _ in 0..200 {
                mast.reset();
                assert!(mast.projection_records().is_empty());
                assert_eq!(mast.projection_peak_bond(), 0);
                assert_eq!(mast.stats.total_nonclifford, 0);
                outcomes.push(run(&mut mast));
                assert_eq!(mast.projection_records().len(), 1);
            }
            outcomes
        };

        let first = collect();
        let second = collect();
        assert_eq!(first, second, "seeded reset continuation must reproduce");
        let ones = first.iter().filter(|&&outcome| outcome).count();
        eprintln!("Mast seeded reset loop: zeros={}, ones={ones}", 200 - ones);
        assert!(
            ones > 0 && ones < 200,
            "reset loop must produce both outcomes"
        );
    }

    #[test]
    fn test_mast_unseeded_reset_smoke() {
        let mut mast = Mast::new(1, 1);
        mast.h(&[QubitId(0)]);
        let _ = mast.mz(&[QubitId(0)]);
        mast.reset();
        mast.h(&[QubitId(0)]);
        let _ = mast.mz(&[QubitId(0)]);
    }

    #[test]
    fn test_mast_t_on_zero_deterministic() {
        // T|0> via MAST: data stays in |0>, measurement should be deterministic
        for trial in 0..20 {
            let mut mast = Mast::with_seed(1, 4, 7000 + trial);
            mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
            let r = mast.mz(&[QubitId(0)]);
            assert!(!r[0].outcome, "trial {trial}: T|0> should measure as 0");
        }
    }

    #[test]
    fn test_mast_t_on_plus_statistics() {
        // H then T via MAST, then measure: should get 50/50 (T only changes phase)
        let num_trials = 200;
        let mut count_0 = 0;
        for trial in 0..num_trials {
            let mut mast = Mast::with_seed(1, 4, 8000 + trial);
            mast.h(&[QubitId(0)]);
            mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
            let r = mast.mz(&[QubitId(0)]);
            if !r[0].outcome {
                count_0 += 1;
            }
        }
        let p0 = f64::from(count_0) / num_trials as f64;
        assert!((p0 - 0.5).abs() < 0.1, "p(0) = {p0:.2}, expected ~0.5");
    }

    /// Multi-qubit MAST vs STN: sample measurement distributions on a
    /// Clifford+T circuit. Each of the 2^n outcomes should have matching
    /// probabilities between MAST and STN.
    #[test]
    fn test_mast_vs_stn_multi_qubit() {
        use crate::stab_mps::StabMps;
        let num_trials = 1000;
        let n = 4;
        // Circuit: H on all, CX(0,1), T(0), CX(1,2), T(1), CX(2,3), T(2)
        let apply = |s: &mut dyn FnMut(&[QubitId])| {
            let _ = s;
        };
        let _ = apply;

        let mut stn_counts = vec![0u32; 1 << n];
        let mut mast_counts = vec![0u32; 1 << n];
        for trial in 0..num_trials {
            // STN
            let mut s = StabMps::with_seed(n, 10_000 + trial);
            s.h(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
            s.cx(&[(QubitId(0), QubitId(1))]);
            s.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
            s.cx(&[(QubitId(1), QubitId(2))]);
            s.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(1)]);
            s.cx(&[(QubitId(2), QubitId(3))]);
            s.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(2)]);
            let mut idx = 0usize;
            for q in 0..n {
                if s.mz(&[QubitId(q)])[0].outcome {
                    idx |= 1 << q;
                }
            }
            stn_counts[idx] += 1;

            // MAST
            let mut m = Mast::with_seed(n, 10, 10_000 + trial);
            m.h(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
            m.cx(&[(QubitId(0), QubitId(1))]);
            m.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
            m.cx(&[(QubitId(1), QubitId(2))]);
            m.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(1)]);
            m.cx(&[(QubitId(2), QubitId(3))]);
            m.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(2)]);
            let mut idx = 0usize;
            for q in 0..n {
                if m.mz(&[QubitId(q)])[0].outcome {
                    idx |= 1 << q;
                }
            }
            mast_counts[idx] += 1;
        }

        // Chi-squared-like check: each outcome should have close probabilities.
        let mut max_diff: f64 = 0.0;
        for i in 0..(1 << n) {
            let p_stn = f64::from(stn_counts[i]) / num_trials as f64;
            let p_mast = f64::from(mast_counts[i]) / num_trials as f64;
            let diff = (p_stn - p_mast).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            eprintln!("outcome {i:04b}: STN={p_stn:.3}, MAST={p_mast:.3}");
        }
        eprintln!("max |p_STN - p_MAST| = {max_diff:.3}");
        // Statistical tolerance for 1000 trials ~= 3 sigma on p=0.5 is 0.047.
        // Use 0.08 to allow for multiple-outcome max.
        assert!(
            max_diff < 0.08,
            "MAST and STN distributions diverge: max diff {max_diff:.3}"
        );
    }

    #[test]
    fn test_mast_vs_stn_single_qubit() {
        // Compare MAST and STN state vectors for H, T on single qubit
        use crate::stab_mps::StabMps;

        let mut stn = StabMps::new(1);
        stn.h(&[QubitId(0)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        let _stn_sv = stn.state_vector();

        // MAST: the state vector includes ancilla qubits, so we can't
        // directly compare. But the data qubit probabilities should match.
        // Use measurement statistics instead.
        let num_trials = 500;
        let mut stn_count = 0;
        let mut mast_count = 0;
        for trial in 0..num_trials {
            let mut s = StabMps::with_seed(1, 9000 + trial);
            s.h(&[QubitId(0)]);
            s.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
            if !s.mz(&[QubitId(0)])[0].outcome {
                stn_count += 1;
            }

            let mut m = Mast::with_seed(1, 4, 9000 + trial);
            m.h(&[QubitId(0)]);
            m.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
            if !m.mz(&[QubitId(0)])[0].outcome {
                mast_count += 1;
            }
        }
        let stn_p0 = f64::from(stn_count) / num_trials as f64;
        let mast_p0 = f64::from(mast_count) / num_trials as f64;
        eprintln!("STN p(0) = {stn_p0:.3}, MAST p(0) = {mast_p0:.3}");
        // Both should be ~0.5 (T only changes phase, not Z-basis probabilities)
        assert!(
            (stn_p0 - mast_p0).abs() < 0.1,
            "STN p(0)={stn_p0:.3} vs MAST p(0)={mast_p0:.3} should be similar"
        );
    }

    #[test]
    fn test_stn_3qubit_measurement_correlation() {
        // Test that STN gives same results as plain SparseStabY for pure Clifford.
        use crate::stab_mps::StabMps;

        let mut stn_corr = 0;
        let mut tab_corr = 0;
        let num_trials = 50;
        for trial in 0..num_trials {
            // STN version
            let mut stn = StabMps::with_seed(3, 6000 + trial);
            stn.h(&[QubitId(0)]);
            stn.cx(&[(QubitId(0), QubitId(1))]);
            stn.h(&[QubitId(2)]);
            stn.cx(&[(QubitId(0), QubitId(2))]);
            let r2_stn = stn.mz(&[QubitId(2)])[0].outcome;
            let r0_stn = stn.mz(&[QubitId(0)])[0].outcome;
            if r0_stn == r2_stn {
                stn_corr += 1;
            }

            // Plain SparseStabY version (same seed)
            let mut tab = SparseStabY::with_seed(3, 6000 + trial);
            tab.h(&[QubitId(0)]);
            tab.cx(&[(QubitId(0), QubitId(1))]);
            tab.h(&[QubitId(2)]);
            tab.cx(&[(QubitId(0), QubitId(2))]);
            let r2_tab = tab.mz(&[QubitId(2)])[0].outcome;
            let r0_tab = tab.mz(&[QubitId(0)])[0].outcome;
            if r0_tab == r2_tab {
                tab_corr += 1;
            }
        }
        let stn_rate = f64::from(stn_corr) / num_trials as f64;
        let tab_rate = f64::from(tab_corr) / num_trials as f64;
        eprintln!("STN correlation: {stn_rate:.2}, SparseStabY correlation: {tab_rate:.2}");
        // Both should match
        assert!(
            (stn_rate - tab_rate).abs() < 0.2,
            "STN {stn_rate:.2} should match SparseStabY {tab_rate:.2}"
        );
    }

    #[test]
    fn test_manual_mast_with_sparse_stab() {
        // Verify the magic state teleportation protocol using plain SparseStabY.
        // This tests the PROTOCOL, not the STN implementation.
        let mut correlated = 0;
        let num_trials = 100;
        for trial in 0..num_trials {
            let mut tab = SparseStabY::with_seed(3, 7000 + trial);
            // Bell state on q0, q1
            tab.h(&[QubitId(0)]);
            tab.cx(&[(QubitId(0), QubitId(1))]);
            // Magic state injection for T on q0:
            tab.h(&[QubitId(2)]); // ancilla in |+>
            tab.sz(&[QubitId(2)]); // S on ancilla (half of T = S*T^{1/2}... wait, we need T)
            // Actually, SparseStabY can't do T. Let me use T = RZ(pi/4) via the Clifford S.
            // T|+> via Clifford: not possible. T is non-Clifford.
            // In the SparseStabY world, we can test the protocol with S instead of T.
            // S|+> = (|0> + i|1>)/sqrt(2)
            // Protocol: prepare S|+>, CNOT(data, anc), measure anc, correct.
            // For S: correction if outcome=1 is RZ(2*pi/2)=RZ(pi)=-iZ (Clifford).
            // S gate on q0 of Bell state: (|00> + i|11>)/sqrt(2)
            // CNOT(q0, q2):
            tab.cx(&[(QubitId(0), QubitId(2))]);
            let anc_result = tab.mz(&[QubitId(2)])[0].outcome;
            if anc_result {
                // Correction: RZ(pi) = -iZ on q0
                tab.z(&[QubitId(0)]);
            }
            let r0 = tab.mz(&[QubitId(0)])[0].outcome;
            let r1 = tab.mz(&[QubitId(1)])[0].outcome;
            if r0 == r1 {
                correlated += 1;
            }
        }
        let rate = f64::from(correlated) / num_trials as f64;
        eprintln!("SparseStabY manual S-injection correlation: {rate:.2}");
        assert!(rate > 0.90, "correlation {rate:.2} should be > 0.90");
    }

    #[test]
    fn test_manual_mast_with_stn_clifford() {
        // Manual MAST with S (Clifford) instead of T.
        // This should work because the MPS stays trivial.
        use crate::stab_mps::StabMps;

        let mut correlated = 0;
        let num_trials = 100;
        for trial in 0..num_trials {
            let mut stn = StabMps::with_seed(3, 5000 + trial);
            stn.h(&[QubitId(0)]);
            stn.cx(&[(QubitId(0), QubitId(1))]);

            // S-injection (Clifford, MPS stays trivial):
            stn.h(&[QubitId(2)]);
            stn.sz(&[QubitId(2)]); // S instead of T
            stn.cx(&[(QubitId(0), QubitId(2))]);
            let anc_result = stn.mz(&[QubitId(2)])[0].outcome;
            if anc_result {
                stn.z(&[QubitId(0)]); // RZ(pi) correction for S
            }

            let r0 = stn.mz(&[QubitId(0)])[0].outcome;
            let r1 = stn.mz(&[QubitId(1)])[0].outcome;
            if r0 == r1 {
                correlated += 1;
            }
        }
        let rate = f64::from(correlated) / num_trials as f64;
        eprintln!("STN Clifford injection correlation: {rate:.2}");
        assert!(rate > 0.90, "correlation {rate:.2} should be > 0.90");
    }

    #[test]
    fn test_z2_expectation_value() {
        // Verify the Z_2 expectation value matches between STN and direct computation.
        use crate::stab_mps::StabMps;
        use nalgebra::DMatrix;
        use pecos_simulators::StabVec;

        let mut stn = StabMps::new(3);
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.h(&[QubitId(2)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(2)]);
        stn.cx(&[(QubitId(0), QubitId(2))]);

        // Compute <Z_2> from state vector
        let mut crz = StabVec::builder(3).seed(42).build();
        crz.h(&[QubitId(0)]);
        crz.cx(&[(QubitId(0), QubitId(1))]);
        crz.h(&[QubitId(2)]);
        crz.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(2)]);
        crz.cx(&[(QubitId(0), QubitId(2))]);
        let crz_sv = crz.state_vector();

        // <Z_2> from state vector: sum |a_i|^2 * (-1)^{bit 2 of i}
        let mut z2_ev_direct = 0.0;
        for (i, a) in crz_sv.iter().enumerate() {
            let bit2 = (i >> 2) & 1; // qubit 2 in LSB convention
            let sign = if bit2 == 1 { -1.0 } else { 1.0 };
            z2_ev_direct += a.norm_sqr() * sign;
        }

        // <Z_2> from STN decomposition
        let decomp = crate::stab_mps::pauli_decomp::decompose_z(
            stn.tableau().stabs(),
            stn.tableau().destabs(),
            2,
        );
        eprintln!("Z_2 decomp: {decomp:?}");

        let z_gate = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        );
        let x_gate = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );

        if let crate::stab_mps::pauli_decomp::ZDecomposition::DestabilizerFlip {
            flip_sites,
            phase,
            sign_sites,
        } = decomp
        {
            let mut ops: Vec<(usize, DMatrix<Complex64>)> = Vec::new();
            for j in &flip_sites {
                ops.push((*j, x_gate.clone()));
            }
            for k in &sign_sites {
                ops.push((*k, z_gate.clone()));
            }
            let raw_ev = stn.mps().expectation_product(&ops);
            let z2_ev_stn = (phase * raw_ev).re;
            eprintln!("Z_2 EV: direct={z2_ev_direct:.6}, STN={z2_ev_stn:.6}, phase={phase:.4}");
            approx::assert_relative_eq!(z2_ev_stn, z2_ev_direct, epsilon = 1e-6);
        }
    }

    #[test]
    fn test_stn_state_before_ancilla_measurement() {
        // Check that the STN state vector before ancilla measurement is correct.
        use crate::stab_mps::StabMps;
        use pecos_simulators::StabVec;

        let mut stn = StabMps::new(3);
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.h(&[QubitId(2)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(2)]);
        stn.cx(&[(QubitId(0), QubitId(2))]);

        let mut crz = StabVec::builder(3).seed(42).build();
        crz.h(&[QubitId(0)]);
        crz.cx(&[(QubitId(0), QubitId(1))]);
        crz.h(&[QubitId(2)]);
        crz.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(2)]);
        crz.cx(&[(QubitId(0), QubitId(2))]);

        let stn_sv = stn.state_vector();
        let crz_sv = crz.state_vector();

        // Check overlap
        let norm_stn: f64 = stn_sv.iter().map(nalgebra::Complex::norm_sqr).sum();
        let norm_crz: f64 = crz_sv.iter().map(nalgebra::Complex::norm_sqr).sum();
        let overlap: Complex64 = stn_sv
            .iter()
            .zip(crz_sv.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();

        eprintln!(
            "State before ancilla meas: norm_stn={norm_stn:.4}, norm_crz={norm_crz:.4}, overlap={:.4}",
            overlap.norm_sqr()
        );
        assert!(
            (overlap.norm_sqr() - 1.0).abs() < 0.01,
            "states should match (overlap = {:.4})",
            overlap.norm_sqr()
        );
    }

    #[test]
    fn test_manual_mast_with_stn_nonclifford() {
        // Manual MAST with T (non-Clifford).
        // This tests whether the STN measurement handles the ancilla correctly.
        use crate::stab_mps::StabMps;

        let mut correlated = 0;
        let num_trials = 100;
        for trial in 0..num_trials {
            let mut stn = StabMps::with_seed(3, 5000 + trial);
            stn.h(&[QubitId(0)]);
            stn.cx(&[(QubitId(0), QubitId(1))]);

            // T-injection (non-Clifford):
            stn.h(&[QubitId(2)]);
            stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(2)]);
            stn.cx(&[(QubitId(0), QubitId(2))]);
            let anc_result = stn.mz(&[QubitId(2)])[0].outcome;
            if anc_result {
                stn.sz(&[QubitId(0)]); // S correction
            }

            let r0 = stn.mz(&[QubitId(0)])[0].outcome;
            let r1 = stn.mz(&[QubitId(1)])[0].outcome;
            if r0 == r1 {
                correlated += 1;
            }
        }
        let rate = f64::from(correlated) / num_trials as f64;
        eprintln!("STN T-injection correlation: {rate:.2}");
        assert!(rate > 0.90, "correlation {rate:.2} should be > 0.90");
    }

    #[test]
    fn test_mast_measurement() {
        // Bell state + T via MAST: after ancilla projection, data qubits
        // should be in Bell+T state with correlated measurements.
        //
        // Diagnose: check MPS norm and bond dims after each step.
        let mut mast = Mast::with_seed(2, 4, 42);
        mast.h(&[QubitId(0)]);
        mast.cx(&[(QubitId(0), QubitId(1))]);

        eprintln!(
            "After Bell: norm={:.4}, bonds={:?}",
            mast.mps().norm_squared(),
            mast.mps().bond_dims()
        );

        mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);

        eprintln!(
            "After T inject: norm={:.4}, bonds={:?}, ancillas={}",
            mast.mps().norm_squared(),
            mast.mps().bond_dims(),
            mast.num_ancillas_used()
        );

        // Project deferred measurements
        mast.project_all();

        eprintln!(
            "After project: norm={:.4}, bonds={:?}",
            mast.mps().norm_squared(),
            mast.mps().bond_dims()
        );

        // Check MPS state
        let mps_sv = mast.mps().state_vector();
        eprintln!("MPS SV after project:");
        for (i, a) in mps_sv.iter().enumerate() {
            if a.norm() > 1e-12 {
                eprintln!("  [{i:06b}] = {:.4} + {:.4}i", a.re, a.im);
            }
        }

        // Now measure both data qubits
        let mut correlated = 0;
        let num_trials = 100;
        for trial in 0..num_trials {
            let mut m = Mast::with_seed(2, 4, 5000 + trial);
            m.h(&[QubitId(0)]);
            m.cx(&[(QubitId(0), QubitId(1))]);
            m.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);

            let r0 = m.mz(&[QubitId(0)])[0].outcome;
            let r1 = m.mz(&[QubitId(1)])[0].outcome;
            if r0 == r1 {
                correlated += 1;
            }
        }
        let correlation_rate = f64::from(correlated) / num_trials as f64;
        eprintln!("Correlation rate: {correlation_rate:.2}");
        assert!(
            correlation_rate > 0.90,
            "correlation rate {correlation_rate:.2} should be > 0.90"
        );
    }

    #[test]
    fn test_mast_merge_rz_two_t_gates_merge() {
        // Two T on same qubit with merge_rz should produce a single
        // non-Clifford (merged to S = Clifford fast-path). Eager path
        // would do two MAST injections.
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut m = Mast::with_seed(2, 4, 7).with_merge_rz(true);
        m.h(&[QubitId(0)]);
        m.rz(t, &[QubitId(0)]);
        m.rz(t, &[QubitId(0)]);
        m.flush();
        // T+T = S (Clifford). No ancillas used.
        assert_eq!(
            m.num_ancillas_used(),
            0,
            "T+T should merge to S (Clifford), no MAST ancillas used"
        );
    }

    #[test]
    fn test_mast_merge_rz_intervening_cz_still_merges() {
        // CZ on different qubits doesn't flush pending_rz on q0. Merge.
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut m = Mast::with_seed(2, 4, 9).with_merge_rz(true);
        m.h(&[QubitId(0), QubitId(1)]);
        m.rz(t, &[QubitId(0)]);
        m.cz(&[(QubitId(0), QubitId(1))]); // CZ commutes with RZ
        m.rz(t, &[QubitId(0)]);
        m.flush();
        // Merged T+T = S. No MAST ancilla used.
        assert_eq!(
            m.num_ancillas_used(),
            0,
            "CZ should not flush pending_rz, merge persists"
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "Mast::project_all predetermined deferred branch was lost")]
    fn mast_deferred_branch_loss_keeps_debug_assertion() {
        let mut mast = Mast::with_seed(1, 1, 17);
        mast.h(&[QubitId(0)]);
        mast.rz(Angle64::QUARTER_TURN / 2_u64, &[QubitId(0)]);
        crate::stab_mps::measure::inject_projection_vanishes(1);
        mast.project_all();
    }

    #[cfg(not(debug_assertions))]
    fn assert_deferred_ancilla_outcome(mast: &Mast, ancilla: usize, outcome: bool) {
        let norm_squared = mast.mps.norm_squared();
        let expectation =
            crate::stab_mps::measure::z_expectation_value(&mast.tableau, &mast.mps, ancilla).re
                / norm_squared;
        let expected = if outcome { -1.0 } else { 1.0 };
        assert!(
            (expectation - expected).abs() < 1e-10,
            "deferred ancilla {ancilla} projected outcome mismatch: expected {outcome}, Z={expectation:.16e}"
        );
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn mast_deferred_branch_loss_uses_untruncated_complement_in_release() {
        let configured = MpsConfig {
            max_bond_dim: 1,
            svd_cutoff: 1e-7,
            max_truncation_error: Some(1e-4),
            parallel: false,
            direction_alternating_compression: false,
        };
        let mut mast = Mast::with_seed(1, 1, 17).with_mps_config(configured.clone());
        mast.h(&[QubitId(0)]);
        mast.rz(Angle64::QUARTER_TURN / 2_u64, &[QubitId(0)]);
        let deferred = mast.deferred[0];
        crate::stab_mps::measure::inject_projection_vanishes(1);
        mast.project_all();
        assert_eq!(mast.deferred_branch_lost_count(), 1);
        assert_deferred_ancilla_outcome(&mast, deferred.ancilla, !deferred.predetermined_outcome);
        assert!((mast.mps.norm_squared() - 1.0).abs() < 1e-12);
        assert_eq!(mast.mps.config().max_bond_dim, configured.max_bond_dim);
        assert_eq!(
            mast.mps.config().svd_cutoff.to_bits(),
            configured.svd_cutoff.to_bits()
        );
        assert_eq!(
            mast.mps.config().max_truncation_error,
            configured.max_truncation_error
        );
    }

    #[cfg(not(debug_assertions))]
    #[test]
    fn mast_zero_probability_deferred_loss_projects_complement_in_release() {
        let configured = MpsConfig {
            max_bond_dim: 1,
            svd_cutoff: 1e-7,
            max_truncation_error: Some(1e-4),
            parallel: false,
            direction_alternating_compression: false,
        };
        let mut mast = Mast::with_seed(1, 1, 23).with_mps_config(configured);
        mast.h(&[QubitId(0)]);
        mast.rz(Angle64::QUARTER_TURN / 2_u64, &[QubitId(0)]);
        let deferred = mast.deferred[0];
        // Preserve a healthy survival ratio while emulating the probability
        // zero returned when prior configured truncation erased this branch.
        crate::stab_mps::measure::inject_zero_projection_probabilities(1);
        mast.project_all();
        assert_eq!(mast.deferred_branch_lost_count(), 1);
        assert_deferred_ancilla_outcome(&mast, deferred.ancilla, !deferred.predetermined_outcome);
    }

    #[cfg(not(debug_assertions))]
    #[test]
    #[should_panic(expected = "Mast::project_all complement deferred branch was also lost")]
    fn mast_deferred_double_loss_panics_in_release() {
        let mut mast = Mast::with_seed(1, 1, 29);
        mast.h(&[QubitId(0)]);
        mast.rz(Angle64::QUARTER_TURN / 2_u64, &[QubitId(0)]);
        crate::stab_mps::measure::inject_projection_vanishes(2);
        mast.project_all();
    }
}
