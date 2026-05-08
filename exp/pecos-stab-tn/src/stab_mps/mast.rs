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
//! 3. Defer the ancilla measurement until the end
//!
//! At the end of the circuit, all deferred measurements are performed.
//! For random circuits with t <= N, most projections are non-entangling,
//! keeping the MPS bond dimension bounded by ~3 on average.
//!
//! # References
//!
//! Nakhl et al., "Stabilizer Tensor Networks with Magic State Injection,"
//! PRL 134, 190602 (2025). arXiv:2411.12482.

use crate::mps::{Mps, MpsConfig};
use num_complex::Complex64;
use pecos_core::{Angle64, QubitId};
use pecos_random::PecosRng;
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator, SparseStabY,
};

use super::non_clifford;

/// A deferred ancilla measurement.
struct DeferredMeasurement {
    /// The ancilla qubit index (in the expanded system).
    ancilla: usize,
    /// The target data qubit that needs correction if ancilla outcome = 1.
    target: usize,
    /// The correction angle: RZ(2*theta) applied to target if ancilla = 1.
    /// For T gates: correction = RZ(pi/2) = S (Clifford).
    correction_angle: Angle64,
}

/// MAST simulator: Magic state injection Augmented STN.
///
/// Wraps the STN approach with magic state injection for non-Clifford gates.
/// Pre-allocates ancilla qubits for up to `max_non_clifford` T/RZ gates.
pub struct Mast {
    /// Number of data qubits.
    num_data_qubits: usize,
    /// Maximum number of non-Clifford gates (= number of ancilla slots).
    _max_non_clifford: usize,
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
    global_phase: Complex64,
    disent_flags: Vec<Option<super::SiteEigenstate>>,
    gf2_matrix: super::ofd::Gf2FlipMatrix,
    rng: PecosRng,
    pub stats: super::StabMpsStats,
    /// Deferred virtual-frame Clifford V for lazy measurement
    /// (see `super::measure::DeferredOp`).
    deferred_ops: Vec<super::measure::DeferredOp>,
    /// When `true`, measurement uses the lazy virtual-frame path:
    /// accumulates `pre_reduce` CNOTs and post-projection basis-rotation
    /// Cliffords into a deferred queue rather than applying them eagerly
    /// to the MPS. Set via `with_lazy_measure(true)`.
    lazy_measure: bool,
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
    #[must_use]
    pub fn new(num_qubits: usize, max_non_clifford: usize) -> Self {
        let total = num_qubits + max_non_clifford;
        Self {
            num_data_qubits: num_qubits,
            _max_non_clifford: max_non_clifford,
            total_qubits: total,
            tableau: SparseStabY::new(total).with_destab_sign_tracking(),
            mps: Mps::new(total, MpsConfig::default()),
            config: MpsConfig::default(),
            next_ancilla: num_qubits,
            deferred: Vec::new(),
            global_phase: Complex64::new(1.0, 0.0),
            disent_flags: vec![Some(super::SiteEigenstate::Z(false)); total],
            gf2_matrix: super::ofd::Gf2FlipMatrix::new(total),
            rng: PecosRng::seed_from_u64(0),
            stats: super::StabMpsStats::default(),
            deferred_ops: Vec::new(),
            lazy_measure: false,
            pending_rz: vec![None; total],
            merge_rz: false,
        }
    }

    /// Create with a specific seed.
    #[must_use]
    pub fn with_seed(num_qubits: usize, max_non_clifford: usize, seed: u64) -> Self {
        let total = num_qubits + max_non_clifford;
        Self {
            num_data_qubits: num_qubits,
            _max_non_clifford: max_non_clifford,
            total_qubits: total,
            tableau: SparseStabY::with_seed(total, seed).with_destab_sign_tracking(),
            mps: Mps::new(total, MpsConfig::default()),
            config: MpsConfig::default(),
            next_ancilla: num_qubits,
            deferred: Vec::new(),
            global_phase: Complex64::new(1.0, 0.0),
            disent_flags: vec![Some(super::SiteEigenstate::Z(false)); total],
            gf2_matrix: super::ofd::Gf2FlipMatrix::new(total),
            rng: PecosRng::seed_from_u64(seed),
            stats: super::StabMpsStats::default(),
            deferred_ops: Vec::new(),
            lazy_measure: false,
            pending_rz: vec![None; total],
            merge_rz: false,
        }
    }

    /// Enable lazy virtual-frame measurement. Fluent-style setter; returns
    /// `self` for chaining after `new`/`with_seed`. See
    /// `StabMpsBuilder::lazy_measure` for semantics.
    #[must_use]
    pub fn with_lazy_measure(mut self, lazy: bool) -> Self {
        self.lazy_measure = lazy;
        self
    }

    /// Enable RZ batching on same qubit. See `StabMpsBuilder::merge_rz` for
    /// semantics. Fluent-style setter on MAST.
    #[must_use]
    pub fn with_merge_rz(mut self, merge: bool) -> Self {
        self.merge_rz = merge;
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

    /// Flush all pending merged RZ. Public; useful before read operations
    /// when `merge_rz` is on.
    pub fn flush(&mut self) {
        if !self.merge_rz {
            return;
        }
        for q in 0..self.total_qubits {
            self.flush_pending_rz(q);
        }
    }

    #[must_use]
    pub fn num_data_qubits(&self) -> usize {
        self.num_data_qubits
    }

    #[must_use]
    pub fn num_ancillas_used(&self) -> usize {
        self.next_ancilla - self.num_data_qubits
    }

    #[must_use]
    pub fn max_bond_dim(&self) -> usize {
        self.mps.max_bond_dim()
    }

    #[must_use]
    pub fn mps(&self) -> &Mps {
        &self.mps
    }

    /// Inject a magic state for RZ(theta) on the target qubit.
    ///
    /// Magic state teleportation protocol:
    /// 1. Prepare ancilla in |+>: H on ancilla
    /// 2. Apply RZ(theta) on ancilla (local, single-site MPS gate)
    /// 3. CNOT(target, ancilla) -- **target controls, ancilla is CX target**
    /// 4. Defer measurement of ancilla
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
        non_clifford::apply_rz_stab_mps(
            &mut self.tableau,
            &mut self.mps,
            cos_half,
            sin_half,
            ancilla,
            true,
            &mut non_clifford::RzContext {
                disent_flags: &mut self.disent_flags,
                gf2_matrix: &mut self.gf2_matrix,
                stats: &mut self.stats,
            },
        );

        // Step 3: CNOT(target, ancilla) -- target controls, ancilla is CX target
        // This is the key: data qubit controls, ancilla flips.
        self.tableau.cx(&[(tgt_qid, anc_qid)]);

        // Step 4: Record deferred measurement with correction angle
        self.deferred.push(DeferredMeasurement {
            ancilla,
            target,
            correction_angle: theta + theta, // RZ(2*theta) correction if outcome=1
        });
    }

    /// Project all deferred ancilla measurements.
    ///
    /// For each deferred ancilla:
    /// 1. Measure ancilla in Z basis (using shared STN measurement protocol)
    /// 2. If outcome = 1: apply RZ(2*theta) correction to the target data qubit
    ///    (For T gates, this is S = RZ(pi/2), which is Clifford)
    pub fn project_all(&mut self) {
        let deferred: Vec<DeferredMeasurement> = self.deferred.drain(..).rev().collect();
        for dm in deferred {
            // Measure the ancilla using the shared STN measurement protocol
            let result = if self.lazy_measure {
                super::measure::measure_qubit_stab_mps_lazy(
                    &mut self.tableau,
                    &mut self.mps,
                    &mut self.rng,
                    dm.ancilla,
                    &mut self.deferred_ops,
                )
            } else {
                super::measure::measure_qubit_stab_mps(
                    &mut self.tableau,
                    &mut self.mps,
                    &mut self.rng,
                    dm.ancilla,
                )
            };

            // If outcome = 1 (true in PECOS convention): apply correction
            if result.outcome {
                let corr = dm.correction_angle;
                let tgt = QubitId(dm.target);

                // Check if correction is a Clifford angle
                if corr == Angle64::ZERO {
                    // No correction needed
                } else if corr == Angle64::HALF_TURN {
                    // RZ(pi) = -iZ
                    self.global_phase *= Complex64::new(0.0, -1.0);
                    self.tableau.z(&[tgt]);
                } else if corr == Angle64::QUARTER_TURN {
                    // RZ(pi/2) = e^{-i*pi/4} S -- this is the T gate correction
                    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
                    self.global_phase *= Complex64::new(inv_sqrt2, -inv_sqrt2);
                    self.tableau.sz(&[tgt]);
                } else if corr == Angle64::THREE_QUARTERS_TURN {
                    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
                    self.global_phase *= Complex64::new(inv_sqrt2, inv_sqrt2);
                    self.tableau.szdg(&[tgt]);
                } else {
                    // Non-Clifford correction: apply via STN protocol
                    let (sin_half, cos_half) = corr.half_angle_sin_cos();
                    non_clifford::apply_rz_stab_mps(
                        &mut self.tableau,
                        &mut self.mps,
                        cos_half,
                        sin_half,
                        dm.target,
                        true,
                        &mut non_clifford::RzContext {
                            disent_flags: &mut self.disent_flags,
                            gf2_matrix: &mut self.gf2_matrix,
                            stats: &mut self.stats,
                        },
                    );
                }
            }
        }
    }
}

impl QuantumSimulator for Mast {
    fn reset(&mut self) -> &mut Self {
        self.tableau = SparseStabY::new(self.total_qubits).with_destab_sign_tracking();
        self.mps = Mps::new(self.total_qubits, self.config.clone());
        self.next_ancilla = self.num_data_qubits;
        self.deferred.clear();
        self.global_phase = Complex64::new(1.0, 0.0);
        self.disent_flags = vec![Some(super::SiteEigenstate::Z(false)); self.total_qubits];
        self.gf2_matrix.reset();
        self.deferred_ops.clear();
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
        // Then measure data qubits using the full STN measurement protocol
        qubits
            .iter()
            .map(|&q| {
                if self.lazy_measure {
                    super::measure::measure_qubit_stab_mps_lazy(
                        &mut self.tableau,
                        &mut self.mps,
                        &mut self.rng,
                        q.index(),
                        &mut self.deferred_ops,
                    )
                } else {
                    super::measure::measure_qubit_stab_mps(
                        &mut self.tableau,
                        &mut self.mps,
                        &mut self.rng,
                        q.index(),
                    )
                }
            })
            .collect()
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
        mast.h(&[QubitId(0)]);
        mast.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        assert_eq!(mast.num_ancillas_used(), 1);
        // Bond dim should be low -- the RZ on the ancilla is a single-site gate
        assert!(
            mast.max_bond_dim() <= 2,
            "bond dim should be low, got {}",
            mast.max_bond_dim()
        );
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

    #[test]
    fn test_mast_with_lazy_measure_bell_correlation() {
        // Fluent setter on MAST: measurements via lazy path.
        for trial in 0..10 {
            let mut m = Mast::with_seed(2, 4, 5000 + trial).with_lazy_measure(true);
            m.h(&[QubitId(0)]);
            m.cx(&[(QubitId(0), QubitId(1))]);
            m.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
            let r0 = m.mz(&[QubitId(0)])[0].outcome;
            let r1 = m.mz(&[QubitId(1)])[0].outcome;
            assert_eq!(r0, r1, "lazy MAST Bell+T trial {trial}");
        }
    }
}
