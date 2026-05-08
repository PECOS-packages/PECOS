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

//! Clifford+RZ simulator using the sum-over-Cliffords decomposition.
//!
//! Represents a quantum state as a weighted sum of stabilizer states (CH-form):
//!
//! ```text
//! |psi> = sum_k alpha_k |phi_k>
//! ```
//!
//! Clifford gates are applied to all terms. Non-Clifford RZ gates decompose each term:
//!
//! ```text
//! RZ(theta) |phi> = cos(theta/2) |phi> - i*sin(theta/2) Z|phi>
//! ```
//!
//! doubling the number of terms per RZ gate. The cost is exponential in the number
//! of non-Clifford gates, but polynomial in the number of qubits and Clifford gates.
//!
//! # References
//!
//! - Bravyi, Browne, Calpin, Campbell, Gosset, Howard.
//!   "Simulation of quantum circuits by low-rank stabilizer decompositions."
//!   arXiv:1808.00128 (2019).

pub mod ch_form;
pub mod exact_scalar;
pub mod quadratic_form;
pub mod sparse_binary_matrix;

use crate::{ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator};
use ch_form::CHFormGeneric;
use core::fmt::Debug;
use num_complex::Complex64;
use pecos_core::{Angle64, BitSet, IndexSet, QubitId};
use pecos_random::{PecosRng, Rng, RngExt, SeedableRng};

/// Clifford+RZ simulator using sum-over-Cliffords decomposition.
///
/// Each term is a (coefficient, CH-form state) pair. Clifford gates are free
/// (applied to all terms). Each RZ gate doubles the number of terms.
///
/// RZ gates on the same qubit are automatically fused: `RZ(a) RZ(b) = RZ(a+b)`.
/// Pending RZ angles are flushed when a non-commuting gate or measurement is applied.
///
/// # Pruning
///
/// Terms with negligible coefficients are pruned before each RZ decomposition.
/// The pruning threshold can be configured via the builder:
///
/// ```
/// use pecos_simulators::StabVec;
///
/// let num_qubits = 4;
/// let sim = StabVec::builder(num_qubits)
///     .pruning_threshold(1e-6)
///     .seed(42)
///     .build();
/// ```
///
use crate::clifford_frame::{CliffordFrame, GEN_LENS, GENERATORS, PHASE_COCYCLE};

#[derive(Clone, Debug)]
pub struct StabVecGeneric<S: IndexSet = BitSet, R: SeedableRng + Rng + Debug = PecosRng> {
    num_qubits: usize,
    terms: Vec<(Complex64, CHFormGeneric<S, R>)>,
    /// Pending RZ angles per qubit.
    pending_rz: Vec<Angle64>,
    /// Single-qubit Clifford frame per qubit. All 24 Clifford elements tracked.
    /// State = frame * `pending_rz` * |`stored_state`⟩.
    /// Single-qubit Cliffords compose into the frame in O(1).
    /// Flushed via H+S generator sequence when a two-qubit gate or measurement arrives.
    cliff_frame: Vec<CliffordFrame>,
    /// Global phase from frame compositions: e^{i*`frame_phase`*pi/4}, mod 8.
    frame_phase: u8,
    gamma_diff_qubits: Vec<usize>,
    rel_pruning_threshold: f64,
    /// Monte Carlo measurement threshold. When `Some(n)`, uses MC term sampling
    /// for measurement if T > n (O(T) instead of O(T*pairs)). `None` = exact only.
    /// Default: `Some(2048)`.
    mc_threshold: Option<usize>,
    rng: R,
}

/// Default Clifford+RZ simulator using `BitSet` and `PecosRng`.
pub type StabVec<R = PecosRng> = StabVecGeneric<BitSet, R>;

/// Builder for configuring a `StabVec` simulator.
pub struct StabVecBuilder {
    num_qubits: usize,
    seed: Option<u64>,
    rel_pruning_threshold: f64,
    mc_threshold: Option<usize>,
}

impl StabVecBuilder {
    /// Set the pruning threshold. Terms with |c|^2 < threshold * max(|c|^2) are pruned.
    ///
    /// - Default: 1e-8 (conservative, safe for precision work like QEC)
    /// - 0.0: exact simulation (no pruning, exponential cost)
    /// - 1e-4 to 1e-6: aggressive (faster sampling, lower precision)
    /// - 1e-12 or less: for studying effects at logical error rates ~1e-10
    #[must_use]
    pub fn pruning_threshold(mut self, threshold: f64) -> Self {
        self.rel_pruning_threshold = threshold;
        self
    }

    /// Set the Monte Carlo measurement threshold.
    ///
    /// - `Some(n)`: Use MC term sampling when T > n (default: `Some(2048)`)
    /// - `None`: Always use exact measurement (slower for large T)
    #[must_use]
    pub fn mc_threshold(mut self, threshold: Option<usize>) -> Self {
        self.mc_threshold = threshold;
        self
    }

    /// Set the RNG seed for reproducible measurements.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the simulator.
    #[must_use]
    pub fn build(self) -> StabVec {
        let rng = if let Some(seed) = self.seed {
            PecosRng::seed_from_u64(seed)
        } else {
            rand::make_rng()
        };
        let ch = CHFormGeneric::with_rng(self.num_qubits, rng.clone());
        StabVecGeneric {
            num_qubits: self.num_qubits,
            terms: vec![(Complex64::new(1.0, 0.0), ch)],
            pending_rz: vec![Angle64::default(); self.num_qubits],
            cliff_frame: vec![CliffordFrame::IDENTITY; self.num_qubits],
            frame_phase: 0,
            gamma_diff_qubits: Vec::new(),
            rel_pruning_threshold: self.rel_pruning_threshold,
            mc_threshold: self.mc_threshold,
            rng,
        }
    }
}

impl<S: IndexSet, R: SeedableRng + Rng + Debug + Clone> StabVecGeneric<S, R> {
    /// Recompute `gamma_diff_qubits` from the actual surviving terms.
    /// Only keeps qubits where gamma genuinely differs across at least one pair.
    fn recompute_gamma_diff(&mut self) {
        self.gamma_diff_qubits.clear();
        if self.terms.len() <= 1 {
            return;
        }
        let g0 = self.terms[0].1.gamma();
        for (p, &gp) in g0.iter().enumerate() {
            if self.terms[1..].iter().any(|(_, ch)| ch.gamma()[p] != gp) {
                self.gamma_diff_qubits.push(p);
            }
        }
    }

    /// Merge terms with identical gamma and omega. This is exact (no approximation).
    /// Terms with the same gamma and omega produce identical amplitudes, so their
    /// coefficients can be summed. Reduces T without loss of precision.
    /// Only worth calling when duplicates are likely (e.g., after measurement projection).
    fn merge_identical_terms(&mut self) {
        if self.terms.len() <= 4 {
            return;
        }

        let diff = &self.gamma_diff_qubits;

        // Omega key: compact u64 encoding (fixed-size, no overflow concern).
        let omega_keys: Vec<u64> = (0..self.terms.len())
            .map(|idx| {
                let omega = self.terms[idx].1.omega_exact();
                if omega.is_zero() {
                    0u64
                } else {
                    // sqrt2_pow may be negative; we intentionally reinterpret the bits for a sort key
                    #[allow(clippy::cast_sign_loss)]
                    let sqrt2_bits = (omega.sqrt2_pow() as u64) & 0xFFFF;
                    1 | (u64::from(omega.sign()) << 1)
                        | (u64::from(omega.phase8()) << 2)
                        | (sqrt2_bits << 5)
                }
            })
            .collect();

        // Sort by (gamma on diff qubits, omega) via direct comparison.
        // Avoids packed key overflow for large diff sets.
        let mut sorted: Vec<usize> = (0..self.terms.len()).collect();
        sorted.sort_unstable_by(|&a, &b| {
            let ga = self.terms[a].1.gamma();
            let gb = self.terms[b].1.gamma();
            for &p in diff {
                match (ga[p] & 3).cmp(&(gb[p] & 3)) {
                    std::cmp::Ordering::Equal => {}
                    ord => return ord,
                }
            }
            omega_keys[a].cmp(&omega_keys[b])
        });

        // Detect identical groups by comparing adjacent sorted elements directly.
        let same_key = |a: usize, b: usize| -> bool {
            if omega_keys[a] != omega_keys[b] {
                return false;
            }
            let ga = self.terms[a].1.gamma();
            let gb = self.terms[b].1.gamma();
            diff.iter().all(|&p| (ga[p] & 3) == (gb[p] & 3))
        };

        // Merge adjacent groups
        let mut merged: Vec<(Complex64, usize)> = Vec::new(); // (summed coeff, representative idx)
        let mut gs = 0;
        while gs < sorted.len() {
            let mut ge = gs + 1;
            while ge < sorted.len() && same_key(sorted[ge], sorted[gs]) {
                ge += 1;
            }
            let rep = sorted[gs];
            let mut sum_coeff = self.terms[rep].0;
            for &idx in &sorted[gs + 1..ge] {
                sum_coeff += self.terms[idx].0;
            }
            merged.push((sum_coeff, rep));
            gs = ge;
        }

        if merged.len() < self.terms.len() {
            // Keep only representative terms with summed coefficients.
            // Mark representatives and set their new coefficients.
            let mut keep = vec![false; self.terms.len()];
            let mut new_coeffs = vec![Complex64::new(0.0, 0.0); self.terms.len()];
            for &(c, idx) in &merged {
                if c.norm_sqr() > 1e-30 {
                    keep[idx] = true;
                    new_coeffs[idx] = c;
                }
            }
            let mut i = 0;
            let mut write = 0;
            while i < self.terms.len() {
                if keep[i] {
                    self.terms[i].0 = new_coeffs[i];
                    if write != i {
                        self.terms.swap(write, i);
                        keep.swap(write, i);
                        new_coeffs.swap(write, i);
                    }
                    write += 1;
                }
                i += 1;
            }
            self.terms.truncate(write);
            if self.terms.is_empty() {
                let ch = CHFormGeneric::with_rng(self.num_qubits, self.rng.clone());
                self.terms.push((Complex64::new(0.0, 0.0), ch));
            }
            // Recompute gamma_diff_qubits from actual surviving terms.
            self.recompute_gamma_diff();
        }
    }

    /// Create with a specific RNG and default pruning threshold.
    pub fn with_rng(num_qubits: usize, rng: R) -> Self {
        let ch = CHFormGeneric::with_rng(num_qubits, rng.clone());
        Self {
            num_qubits,
            terms: vec![(Complex64::new(1.0, 0.0), ch)],
            pending_rz: vec![Angle64::default(); num_qubits],
            cliff_frame: vec![CliffordFrame::IDENTITY; num_qubits],
            frame_phase: 0,
            gamma_diff_qubits: Vec::new(),
            rel_pruning_threshold: 1e-8,
            mc_threshold: Some(2048),
            rng,
        }
    }

    /// Create with a specific seed.
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_rng(num_qubits, R::seed_from_u64(seed))
    }

    /// Number of qubits.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Number of terms in the decomposition.
    #[must_use]
    pub fn num_terms(&self) -> usize {
        self.terms.len()
    }

    /// Compute the full state vector by summing all terms.
    ///
    /// O(2^n * `num_terms`) -- only use for small systems and testing.
    #[must_use]
    pub fn state_vector(&mut self) -> Vec<Complex64> {
        self.flush_all_cliff_frames();
        self.flush_all_pending_rz();
        self.state_vector_no_flush()
    }

    /// Compute state vector without flushing pending RZ (internal use).
    fn state_vector_no_flush(&self) -> Vec<Complex64> {
        let dim = 1 << self.num_qubits;
        let mut sv = vec![Complex64::new(0.0, 0.0); dim];
        for (coeff, ch) in &self.terms {
            for (x, sv_x) in sv.iter_mut().enumerate() {
                *sv_x += coeff * ch.amplitude(x);
            }
        }
        sv
    }

    /// Remove terms with negligible coefficients.
    ///
    /// Uses both absolute threshold (1e-14) and the configurable relative
    /// threshold. The relative threshold aggressively prunes small-angle
    /// rotation terms where many coefficients are tiny relative to the
    /// dominant terms.
    #[allow(dead_code)]
    fn prune_terms(&mut self) {
        if self.rel_pruning_threshold <= 0.0 {
            return; // exact mode: no pruning
        }
        let abs_threshold: f64 = 1e-14;
        let max_coeff_sq = self
            .terms
            .iter()
            .map(|(c, _)| c.norm_sqr())
            .fold(0.0f64, f64::max);
        let rel_threshold = max_coeff_sq * self.rel_pruning_threshold;
        let threshold = abs_threshold.max(rel_threshold);
        self.terms.retain(|(coeff, _)| coeff.norm_sqr() > threshold);
        // Always keep at least one term
        if self.terms.is_empty() {
            let ch = CHFormGeneric::with_rng(self.num_qubits, self.rng.clone());
            self.terms.push((Complex64::new(0.0, 0.0), ch));
        }
    }

    /// Flush all pending RZ gates (apply them to the state).
    pub fn flush_all_pending_rz(&mut self) {
        for q in 0..self.num_qubits {
            self.flush_pending_rz(q);
        }
    }

    /// Flush pending RZ on a specific qubit.
    fn flush_pending_rz(&mut self, q: usize) {
        let angle = self.pending_rz[q];
        if angle == Angle64::default() {
            return;
        }
        self.pending_rz[q] = Angle64::default();
        self.apply_rz_immediate(angle, q);
    }

    /// Flush the Clifford frame on qubit q by applying its H+S generator sequence.
    fn flush_cliff_frame(&mut self, q: usize) {
        let cf = self.cliff_frame[q];
        if cf.is_identity() {
            return;
        }
        self.cliff_frame[q] = CliffordFrame::IDENTITY;

        // Fast paths for common frames (avoid GENERATORS lookup overhead).
        let qid = QubitId(q);
        if cf.is_pauli() {
            // Paulis: diagonal part is cheap, non-diagonal part uses X/Y gate.
            match cf.index() {
                1 => {
                    // X: must flush pending_rz (X anticommutes with RZ)
                    self.pending_rz[q] = -self.pending_rz[q];
                    self.flush_pending_rz(q);
                    self.apply_clifford(|ch| {
                        ch.x(&[qid]);
                    });
                }
                2 => {
                    // Y: anticommutes with RZ
                    self.pending_rz[q] = -self.pending_rz[q];
                    self.flush_pending_rz(q);
                    self.apply_clifford(|ch| {
                        ch.y(&[qid]);
                    });
                }
                3 => {
                    // Z: diagonal, commutes with RZ
                    for (_, ch) in &mut self.terms {
                        ch.z(&[qid]);
                    }
                }
                _ => {}
            }
            return;
        }

        // Flush pending RZ first (non-diagonal Cliffords don't commute with RZ).
        self.flush_pending_rz(q);

        // General path: apply via H+S generator decomposition.
        let idx = cf.index() as usize;
        let len = GEN_LENS[idx] as usize;
        let seq = &GENERATORS[idx];
        for &g in seq.iter().take(len) {
            match g {
                0 => self.apply_clifford(|ch| {
                    ch.h(&[qid]);
                }),
                1 => self.apply_clifford(|ch| {
                    ch.sz(&[qid]);
                }),
                _ => {}
            }
        }
    }

    /// Flush all Clifford frames and apply accumulated phase.
    fn flush_all_cliff_frames(&mut self) {
        for q in 0..self.num_qubits {
            self.flush_cliff_frame(q);
        }
        if self.frame_phase != 0 {
            use crate::clifford_frame::PHASE_ROOTS;
            let [re, im] = PHASE_ROOTS[(self.frame_phase & 7) as usize];
            let phase = Complex64::new(re, im);
            for (coeff, _) in &mut self.terms {
                *coeff *= phase;
            }
            self.frame_phase = 0;
        }
    }

    fn apply_clifford(&mut self, f: impl Fn(&mut CHFormGeneric<S, R>)) {
        for (_, ch) in &mut self.terms {
            f(ch);
        }
    }

    /// Apply a Clifford gate that produces identical structural changes (F,G,M,v,s)
    /// for all terms. Apply to term[0], share Arcs, compute gamma delta.
    fn apply_clifford_structural(&mut self, f: impl Fn(&mut CHFormGeneric<S, R>)) {
        if self.terms.len() <= 1 {
            for (_, ch) in &mut self.terms {
                f(ch);
            }
            return;
        }
        // Structural optimization is only valid when all terms share the same
        // F, G, M, v, s matrices (differ only in gamma/omega/coefficient).
        // After H is applied to terms with different gammas, the structural
        // matrices can diverge. Check Arc pointer equality as a fast guard.
        let structurally_uniform =
            std::sync::Arc::ptr_eq(&self.terms[0].1.arc_f(), &self.terms[1].1.arc_f());
        if !structurally_uniform {
            for (_, ch) in &mut self.terms {
                f(ch);
            }
            return;
        }
        let n = self.num_qubits;
        let gamma_before = self.terms[0].1.gamma().to_vec();
        f(&mut self.terms[0].1);
        // Compute gamma delta
        let mut delta = vec![0u8; n];
        let gamma_after = self.terms[0].1.gamma();
        for p in 0..n {
            delta[p] = (gamma_after[p] + 4 - gamma_before[p]) & 3;
        }
        // Share Arcs
        let shared_f = self.terms[0].1.arc_f();
        let shared_g = self.terms[0].1.arc_g();
        let shared_m = self.terms[0].1.arc_m();
        let shared_v = self.terms[0].1.arc_v();
        let shared_s = self.terms[0].1.arc_s();
        for (_, ch) in &mut self.terms[1..] {
            ch.apply_gamma_delta(&delta);
            ch.set_arcs(
                shared_f.clone(),
                shared_g.clone(),
                shared_m.clone(),
                shared_v.clone(),
                shared_s.clone(),
            );
        }
    }

    /// Buffer an RZ gate. Fuses with any pending RZ on the same qubit.
    /// Uses Angle64 fixed-point addition for exact fusion (T+T=S, 4T=Z, 8T=I).
    fn apply_rz(&mut self, theta: Angle64, q: usize) {
        self.pending_rz[q] += theta;
    }

    /// Apply RZ(theta) immediately (decompose into terms).
    fn apply_rz_immediate(&mut self, theta: Angle64, q: usize) {
        // Detect Clifford angles using exact Angle64 fixed-point comparison.
        // No float conversion needed for detection -- only for the decomposition coefficients.

        // RZ(0) = I (identity, no terms added)
        if theta == Angle64::ZERO {
            return;
        }

        // RZ(pi) = -iZ (Clifford)
        if theta == Angle64::HALF_TURN {
            let phase = Complex64::new(0.0, -1.0); // -i
            for (coeff, ch) in &mut self.terms {
                *coeff *= phase;
                ch.z(&[QubitId(q)]);
            }
            return;
        }

        // RZ(pi/2) = e^{-i*pi/4} * S (Clifford)
        if theta == Angle64::QUARTER_TURN {
            let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
            let phase = Complex64::new(inv_sqrt2, -inv_sqrt2); // e^{-i*pi/4}
            for (coeff, ch) in &mut self.terms {
                *coeff *= phase;
                ch.sz(&[QubitId(q)]);
            }
            return;
        }

        // RZ(3pi/2) = RZ(-pi/2) = e^{i*pi/4} * Sdg (Clifford)
        if theta == Angle64::THREE_QUARTERS_TURN {
            let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
            let phase = Complex64::new(inv_sqrt2, inv_sqrt2); // e^{i*pi/4}
            for (coeff, ch) in &mut self.terms {
                *coeff *= phase;
                ch.szdg(&[QubitId(q)]);
            }
            return;
        }

        // Non-Clifford angle: decompose into two terms.
        // Use Angle64's built-in half-angle trig (exact halving in fixed-point,
        // optimized minimax polynomials -- no radians conversion needed).
        let (sin_half, cos_half_val) = theta.half_angle_sin_cos();

        // Prune negligible terms before doubling to limit growth.
        // Collect gamma Vecs from pruned terms to reuse (avoids malloc churn).
        let mut gamma_pool: Vec<Vec<u8>> = Vec::new();
        if self.rel_pruning_threshold > 0.0 {
            let abs_threshold: f64 = 1e-14;
            let max_coeff_sq = self
                .terms
                .iter()
                .map(|(c, _)| c.norm_sqr())
                .fold(0.0f64, f64::max);
            let threshold = abs_threshold.max(max_coeff_sq * self.rel_pruning_threshold);
            let mut i = 0;
            while i < self.terms.len() {
                if self.terms[i].0.norm_sqr() <= threshold {
                    let (_, mut ch) = self.terms.swap_remove(i);
                    gamma_pool.push(ch.take_gamma());
                } else {
                    i += 1;
                }
            }
            if self.terms.is_empty() {
                let ch = CHFormGeneric::with_rng(self.num_qubits, self.rng.clone());
                self.terms.push((Complex64::new(0.0, 0.0), ch));
            }
        }

        // Track that qubit q now has gamma divergence between terms.
        if !self.gamma_diff_qubits.contains(&q) {
            self.gamma_diff_qubits.push(q);
            self.gamma_diff_qubits.sort_unstable();
        }

        // Modify existing terms in-place (cos terms), push new Z terms.
        let orig_len = self.terms.len();
        self.terms.reserve(orig_len);
        for i in 0..orig_len {
            // Create Z term: reuse pooled gamma Vec if available, else clone.
            let mut ch_z = if let Some(mut reused_gamma) = gamma_pool.pop() {
                reused_gamma.copy_from_slice(self.terms[i].1.gamma());
                self.terms[i].1.clone_with_gamma(reused_gamma)
            } else {
                self.terms[i].1.clone()
            };
            ch_z.z(&[QubitId(q)]);
            let c = self.terms[i].0;
            let z_coeff = Complex64::new(c.im * sin_half, -c.re * sin_half);
            self.terms.push((z_coeff, ch_z));
            self.terms[i].0 *= cos_half_val;
        }
    }

    /// Apply RX(theta) on a qubit.
    ///
    /// RX(theta) = H * RZ(theta) * H
    #[allow(dead_code)]
    fn apply_rx(&mut self, theta: Angle64, q: usize) {
        self.h(&[QubitId(q)]);
        self.apply_rz(theta, q);
        self.h(&[QubitId(q)]);
    }

    #[allow(dead_code)]
    fn apply_rzz(&mut self, theta: Angle64, q0: usize, q1: usize) {
        self.cx(&[(QubitId(q0), QubitId(q1))]);
        self.apply_rz(theta, q1);
        self.cx(&[(QubitId(q0), QubitId(q1))]);
    }

    /// Measure a qubit. Returns the measurement result and projects the state.
    ///
    /// For a single term, uses O(n) probability computation.
    /// For multiple terms, computes the combined state vector O(T * 2^n * n)
    /// and sums probabilities. Future optimization: O(T^2 * n^3) pairwise
    /// inner products using `ExponentialSum` would avoid the 2^n factor.
    fn measure_qubit(&mut self, q: usize, forced: Option<bool>) -> MeasurementResult {
        // Z-basis measurement on qubit q.
        // Frames and pending_rz on OTHER qubits commute with Z_q -- no flush needed.
        // Only qubit q's frame matters:
        // - Diagonal frame (Z→±Z): just flips the outcome. Discard frame.
        // - Non-diagonal frame: must flush (changes measurement basis).
        // Pending_rz on q is diagonal: doesn't affect Z measurement. Discard after.
        // Pending_rz on q is diagonal: doesn't affect Z measurement. Discard after.
        let mut flip_outcome = false;
        let cf_q = self.cliff_frame[q];
        if !cf_q.is_identity() {
            if cf_q.is_diagonal() {
                flip_outcome = !cf_q.z_image().positive; // Z->-Z flips outcome
                self.cliff_frame[q] = CliffordFrame::IDENTITY;
            } else {
                // Non-diagonal: flush this qubit's frame (needs pending_rz flushed first).
                self.flush_cliff_frame(q);
            }
        }
        // Pending RZ on q doesn't affect Z measurement. Discard it.
        // After measurement, qubit is in Z eigenstate; pending phase is irrelevant.
        self.pending_rz[q] = Angle64::default();

        // Compute probability of measuring 0
        let prob0 = if self.terms.len() == 1 {
            // Single term: O(n) using CH-form structure directly
            let (coeff, ch) = &self.terms[0];
            coeff.norm_sqr() * ch.prob_z_zero(q)
        } else if self.num_qubits <= 6 {
            // For small qubit counts, state vector is fast enough.
            let sv = self.state_vector();
            let mut p = 0.0;
            for (x, sv_x) in sv.iter().enumerate() {
                if (x >> q) & 1 == 0 {
                    p += sv_x.norm_sqr();
                }
            }
            p
        } else if self.terms.len() <= 8 {
            // expectation_value_zq depends only on shared structure (G/v/s), same for all terms.
            let ez0 = self.terms[0].1.expectation_value_zq(q);
            if ez0 == 0.0 {
                // Full pairwise computation. Use gamma_diff_qubits for O(|diff|) early-skip.
                let sc = self.terms[0].1.precompute_shared_constraints();
                let t = self.terms.len();
                let omegas: Vec<_> = self
                    .terms
                    .iter()
                    .map(|(_, ch)| ch.omega_complex())
                    .collect();
                let diff = &self.gamma_diff_qubits;
                let ez = self.terms[0].1.expectation_value_zq(q);
                let one_plus_ez = 1.0 + ez;
                let mut prob = 0.0;
                for j in 0..t {
                    prob += self.terms[j].0.norm_sqr() * one_plus_ez;
                }
                for j in 0..t {
                    for k in (j + 1)..t {
                        // Fast early-skip using diff qubits only.
                        // If any diff qubit (other than z_qubit q) has l=2, skip.
                        let g1 = self.terms[j].1.gamma();
                        let g2 = self.terms[k].1.gamma();
                        let mut skip = false;
                        for &p in diff {
                            if p == q {
                                continue;
                            }
                            if (g1[p] ^ g2[p]) == 2 {
                                skip = true;
                                break;
                            }
                        }
                        if skip {
                            continue;
                        }
                        let cjk = self.terms[j].0.conj() * self.terms[k].0;
                        let (ip, ip_z) = self.terms[j].1.inner_product_pair_precomputed(
                            &self.terms[k].1,
                            q,
                            &sc,
                            omegas[j],
                            omegas[k],
                            Some(diff),
                        );
                        prob += 2.0 * (cjk * (ip + ip_z)).re;
                    }
                }
                0.5 * prob
            } else {
                // Deterministic: all terms have the same Z_q expectation.
                let norm: f64 = self.terms.iter().map(|(c, _)| c.norm_sqr()).sum();
                0.5 * norm * (1.0 + ez0)
            }
        } else {
            // Large T: first check if measurement is deterministic from structure.
            let ez = self.terms[0].1.expectation_value_zq(q);
            if ez != 0.0 {
                // Deterministic: all terms have the same Z_q expectation.
                let norm: f64 = self.terms.iter().map(|(c, _)| c.norm_sqr()).sum();
                0.5 * norm * (1.0 + ez)
            } else if self.mc_threshold.is_some_and(|t| self.terms.len() > t) {
                // Very large T: Monte Carlo term sampling. Pick a term proportional
                // to |c_j|², use its single-term probability as Pr(0).
                // This approximation drops cross-term interference but is good when
                // terms are nearly orthogonal (most cross-terms are zero from gamma
                // bucketing). O(T) instead of O(T * pairs_per_bucket).
                let norm_sq: f64 = self.terms.iter().map(|(c, _)| c.norm_sqr()).sum();
                let r: f64 = self.rng.random::<f64>() * norm_sq;
                let mut cumulative = 0.0;
                let mut chosen = 0;
                for (j, (c, _)) in self.terms.iter().enumerate() {
                    cumulative += c.norm_sqr();
                    if cumulative >= r {
                        chosen = j;
                        break;
                    }
                }
                self.terms[chosen].1.prob_z_zero(q) * norm_sq
            } else {
                // Non-deterministic: sort-based bucketing.
                let sc = self.terms[0].1.precompute_shared_constraints();
                let t = self.terms.len();
                let omegas: Vec<_> = self
                    .terms
                    .iter()
                    .map(|(_, ch)| ch.omega_complex())
                    .collect();
                let diff = &self.gamma_diff_qubits;

                // Sort by (gamma on diff qubits excluding q, then q) via direct
                // comparison. This groups terms that differ only on qubit q adjacently,
                // enabling efficient cross-term bucketing. Avoids packed-key overflow
                // for large diff sets.
                let mut sorted_indices: Vec<usize> = (0..t).collect();
                sorted_indices.sort_unstable_by(|&a, &b| {
                    let ga = self.terms[a].1.gamma();
                    let gb = self.terms[b].1.gamma();
                    // Primary: all diff qubits except q
                    for &p in diff {
                        if p == q {
                            continue;
                        }
                        match (ga[p] & 3).cmp(&(gb[p] & 3)) {
                            std::cmp::Ordering::Equal => {}
                            ord => return ord,
                        }
                    }
                    // Secondary: q itself (refines within masked groups)
                    (ga[q] & 3).cmp(&(gb[q] & 3))
                });

                // Group detection: same gamma on all diff qubits except q.
                let same_masked = |a: usize, b: usize| -> bool {
                    let ga = self.terms[a].1.gamma();
                    let gb = self.terms[b].1.gamma();
                    diff.iter().all(|&p| p == q || (ga[p] & 3) == (gb[p] & 3))
                };

                let mut prob = 0.0;
                let ez = self.terms[0].1.expectation_value_zq(q);
                let one_plus_ez = 1.0 + ez;
                for j in 0..t {
                    prob += self.terms[j].0.norm_sqr() * one_plus_ez;
                }
                // Cross terms: only within groups of matching masked keys
                let mut group_start = 0;
                while group_start < t {
                    let mut group_end = group_start + 1;
                    while group_end < t
                        && same_masked(sorted_indices[group_end], sorted_indices[group_start])
                    {
                        group_end += 1;
                    }
                    for a in group_start..group_end {
                        let j = sorted_indices[a];
                        for &k in &sorted_indices[(a + 1)..group_end] {
                            let cjk = self.terms[j].0.conj() * self.terms[k].0;
                            let (ip, ip_z) = self.terms[j].1.inner_product_pair_precomputed(
                                &self.terms[k].1,
                                q,
                                &sc,
                                omegas[j],
                                omegas[k],
                                Some(&self.gamma_diff_qubits),
                            );
                            prob += 2.0 * (cjk * (ip + ip_z)).re;
                        }
                    }
                    group_start = group_end;
                }
                0.5 * prob
            } // end non-deterministic
        };

        // Adjust probability for frame flip (Z→-Z swaps |0⟩ and |1⟩ probabilities).
        let actual_prob0 = if flip_outcome { 1.0 - prob0 } else { prob0 };

        // Determine outcome from user's perspective (actual state).
        let outcome = if let Some(forced_val) = forced {
            forced_val
        } else if (actual_prob0 - 1.0).abs() < 1e-10 {
            false // deterministic |0>
        } else if actual_prob0 < 1e-10 {
            true // deterministic |1>
        } else {
            let r: f64 = self.rng.random();
            r >= actual_prob0
        };

        let is_deterministic = (actual_prob0 - 1.0).abs() < 1e-10 || actual_prob0 < 1e-10;

        // Projection uses stored-state outcome (flip back if frame flipped).
        let stored_outcome = outcome ^ flip_outcome;

        // Project: measure each CH-form term, keep only compatible terms.
        // After measurement, the state should be projected onto the outcome subspace.
        // For each term, force the measurement outcome and adjust coefficients.
        //
        // The simplest correct approach: reconstruct from the projected state vector.
        // But that loses the stabilizer structure.
        //
        // Better: measure each term independently with the forced outcome.
        // The CH-form measurement correctly projects each term.
        // The coefficients stay the same. Then renormalize.

        // Project each term. The structural changes (F,G,M,v,s) are identical
        // for all terms. Gamma deltas from right_cz/right_s are also identical
        // (constant +2 or +3 independent of starting gamma). Omega changes are
        // the same (depend only on shared state). Apply mz_forced to term[0],
        // compute deltas, propagate to others.
        // Project each term. When gamma[q] is the same for all terms, delta is
        // identical and all terms take the same structural path -- we can apply
        // mz_forced once and share Arcs. Otherwise, apply individually.
        // gamma[q] is uniform if q is not in the diff set (diff tracks all divergent qubits).
        let gamma_q_uniform = self.terms.len() <= 1 || !self.gamma_diff_qubits.contains(&q);
        if gamma_q_uniform && self.terms.len() > 1 {
            // All terms have the same gamma[q], so delta is identical.
            // Structural changes and omega transform are the same for all terms.
            // Apply mz_forced once, compute deltas, propagate to others.
            let gamma_before = self.terms[0].1.gamma().to_vec();
            let omega_before = self.terms[0].1.omega_exact();
            self.terms[0].1.mz_forced(q, stored_outcome);
            let omega_after = self.terms[0].1.omega_exact();
            let mut gamma_delta = vec![0u8; self.num_qubits];
            for p in 0..self.num_qubits {
                gamma_delta[p] = (self.terms[0].1.gamma()[p] + 4 - gamma_before[p]) & 3;
            }
            let shared_f = self.terms[0].1.arc_f();
            let shared_g = self.terms[0].1.arc_g();
            let shared_m = self.terms[0].1.arc_m();
            let shared_v = self.terms[0].1.arc_v();
            let shared_s = self.terms[0].1.arc_s();
            for (_, ch) in &mut self.terms[1..] {
                ch.apply_gamma_delta(&gamma_delta);
                ch.apply_omega_transform(omega_before, omega_after);
                ch.set_arcs(
                    shared_f.clone(),
                    shared_g.clone(),
                    shared_m.clone(),
                    shared_v.clone(),
                    shared_s.clone(),
                );
            }
        } else {
            for (_coeff, ch) in &mut self.terms {
                ch.mz_forced(q, stored_outcome);
            }
        }

        // Merge terms with identical gamma+omega (exact, reduces T).
        // Skip merge when diff_qubits is large relative to T (no collisions possible).
        // With D diff qubits, there are up to 4^D unique gamma keys.
        // If 4^D >> T, no two terms share a key, so merge is a no-op.
        let diff_capacity = if self.gamma_diff_qubits.len() <= 10 {
            1usize << (2 * self.gamma_diff_qubits.len()) // 4^D
        } else {
            usize::MAX
        };
        if diff_capacity <= 4 * self.terms.len() {
            self.merge_identical_terms();
        }

        // Renormalize.
        // After merging, all terms have distinct gamma+omega, so all cross-term
        // inner products are zero. Norm is simply sum of |c_j|^2.
        if self.terms.len() > 1 {
            let norm_sq: f64 = self.terms.iter().map(|(c, _)| c.norm_sqr()).sum();
            if norm_sq > 1e-15 && (norm_sq - 1.0).abs() > 1e-10 {
                let inv_norm = 1.0 / norm_sq.sqrt();
                for (coeff, _) in &mut self.terms {
                    *coeff *= inv_norm;
                }
            }
        }

        MeasurementResult {
            outcome,
            is_deterministic,
        }
    }
}

// ============================================================================
// Constructors for default types
// ============================================================================

impl StabVecGeneric<BitSet, PecosRng> {
    /// Create a builder for configuring the simulator.
    #[must_use]
    pub fn builder(num_qubits: usize) -> StabVecBuilder {
        StabVecBuilder {
            num_qubits,
            seed: None,
            rel_pruning_threshold: 1e-8,
            mc_threshold: Some(2048),
        }
    }

    /// Create a new Clifford+RZ simulator with default RNG.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        let rng: PecosRng = rand::make_rng();
        Self::with_rng(num_qubits, rng)
    }

    /// Create with a specific seed.
    #[must_use]
    pub fn new_with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::with_seed(num_qubits, seed)
    }
}

// ============================================================================
// Trait implementations
// ============================================================================

impl<S: IndexSet, R: SeedableRng + Rng + Debug + Clone> QuantumSimulator for StabVecGeneric<S, R> {
    fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    fn reset(&mut self) -> &mut Self {
        let rng = self.rng.clone();
        let ch = CHFormGeneric::with_rng(self.num_qubits, rng);
        self.terms = vec![(Complex64::new(1.0, 0.0), ch)];
        self.pending_rz.fill(Angle64::default());
        // rel_pruning_threshold preserved across reset
        self
    }
}

impl<S: IndexSet, R: SeedableRng + Rng + Debug + Clone> CliffordGateable for StabVecGeneric<S, R> {
    // === Single-qubit Cliffords: all compose into the frame in O(1) ===
    // Diagonal gates (Z, S, Sdg) commute with pending_rz.
    // Non-diagonal gates (H, X, Y, SX, etc.) negate pending_rz if they
    // anticommute with Z, or flush pending_rz if they don't simply negate.

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qi = q.index();
            self.pending_rz[qi] = -self.pending_rz[qi]; // X anticommutes with RZ
            let old = self.cliff_frame[qi];
            self.frame_phase = (self.frame_phase
                + PHASE_COCYCLE[CliffordFrame::X.index() as usize][old.index() as usize])
                & 7;
            self.cliff_frame[qi] = CliffordFrame::X.compose(old);
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qi = q.index();
            self.pending_rz[qi] = -self.pending_rz[qi]; // Y anticommutes with RZ
            let old = self.cliff_frame[qi];
            self.frame_phase = (self.frame_phase
                + PHASE_COCYCLE[CliffordFrame::Y.index() as usize][old.index() as usize])
                & 7;
            self.cliff_frame[qi] = CliffordFrame::Y.compose(old);
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qi = q.index();
            // Z commutes with RZ, no negation needed.
            let old = self.cliff_frame[qi];
            self.frame_phase = (self.frame_phase
                + PHASE_COCYCLE[CliffordFrame::Z.index() as usize][old.index() as usize])
                & 7;
            self.cliff_frame[qi] = CliffordFrame::Z.compose(old);
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qi = q.index();
            // S is diagonal, commutes with RZ.
            let old = self.cliff_frame[qi];
            self.frame_phase = (self.frame_phase
                + PHASE_COCYCLE[CliffordFrame::SZ.index() as usize][old.index() as usize])
                & 7;
            self.cliff_frame[qi] = CliffordFrame::SZ.compose(old);
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qi = q.index();
            let old = self.cliff_frame[qi];
            self.frame_phase = (self.frame_phase
                + PHASE_COCYCLE[CliffordFrame::SZDG.index() as usize][old.index() as usize])
                & 7;
            self.cliff_frame[qi] = CliffordFrame::SZDG.compose(old);
        }
        self
    }

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        // H maps Z->X. If there's pending_rz, must flush everything first.
        // If pending_rz is zero, H can be composed into the Clifford frame!
        for &q in qubits {
            let qi = q.index();
            if self.pending_rz[qi] == Angle64::default() {
                // No pending RZ: safe to compose H into frame.
                let old = self.cliff_frame[qi];
                self.frame_phase = (self.frame_phase
                    + PHASE_COCYCLE[CliffordFrame::H.index() as usize][old.index() as usize])
                    & 7;
                self.cliff_frame[qi] = CliffordFrame::H.compose(old);
            } else {
                // Pending RZ exists: must flush frame and RZ, then apply H.
                self.flush_cliff_frame(qi);
                self.flush_pending_rz(qi);
                self.apply_clifford(|ch| {
                    ch.h(&[q]);
                });
            }
        }
        self
    }

    // === Two-qubit gates ===
    // Pauli frames propagate through CX/CZ in O(1) with phase correction.
    // Non-Pauli frames must be flushed.

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let c = q0.index();
            let t = q1.index();
            let fc = self.cliff_frame[c];
            let ft = self.cliff_frame[t];
            if fc.is_pauli() && ft.is_pauli() {
                let (new_c, new_t, phase) = CliffordFrame::push_through_cx(fc, ft);
                self.cliff_frame[c] = new_c;
                self.cliff_frame[t] = new_t;
                self.frame_phase = (self.frame_phase + phase) & 7;
            } else {
                self.flush_cliff_frame(c);
                self.flush_cliff_frame(t);
            }
            self.flush_pending_rz(t);
        }
        self.apply_clifford(|ch| {
            ch.cx(pairs);
        });
        self
    }

    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let q = q0.index();
            let r = q1.index();
            let fq = self.cliff_frame[q];
            let fr = self.cliff_frame[r];
            if fq.is_pauli() && fr.is_pauli() {
                let (new_q, new_r, phase) = CliffordFrame::push_through_cz(fq, fr);
                self.cliff_frame[q] = new_q;
                self.cliff_frame[r] = new_r;
                self.frame_phase = (self.frame_phase + phase) & 7;
            } else {
                self.flush_cliff_frame(q);
                self.flush_cliff_frame(r);
            }
        }
        self.apply_clifford_structural(|ch| {
            ch.cz(pairs);
        });
        self
    }

    fn szz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let q = q0.index();
            let r = q1.index();
            let fq = self.cliff_frame[q];
            let fr = self.cliff_frame[r];
            if fq.is_pauli() && fr.is_pauli() {
                let (new_q, new_r, phase) = CliffordFrame::push_through_szz(fq, fr);
                self.cliff_frame[q] = new_q;
                self.cliff_frame[r] = new_r;
                self.frame_phase = (self.frame_phase + phase) & 7;
            } else {
                self.flush_cliff_frame(q);
                self.flush_cliff_frame(r);
            }
        }
        self.apply_clifford_structural(|ch| {
            ch.szz(pairs);
        });
        self
    }

    fn szzdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // SZZdg = SZZ^{-1}. Pauli propagation same as SZZ (inverse has same symplectic).
        for &(q0, q1) in pairs {
            let q = q0.index();
            let r = q1.index();
            let fq = self.cliff_frame[q];
            let fr = self.cliff_frame[r];
            if fq.is_pauli() && fr.is_pauli() {
                let (new_q, new_r, phase) = CliffordFrame::push_through_szz(fq, fr);
                self.cliff_frame[q] = new_q;
                self.cliff_frame[r] = new_r;
                // SZZdg has opposite phase from SZZ propagation
                self.frame_phase = (self.frame_phase + (8 - phase) % 8) & 7;
            } else {
                self.flush_cliff_frame(q);
                self.flush_cliff_frame(r);
            }
        }
        self.apply_clifford_structural(|ch| {
            ch.szzdg(pairs);
        });
        self
    }

    fn sxx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let q = q0.index();
            let r = q1.index();
            let fq = self.cliff_frame[q];
            let fr = self.cliff_frame[r];
            if fq.is_pauli() && fr.is_pauli() {
                let (new_q, new_r, phase) = CliffordFrame::push_through_sxx(fq, fr);
                self.cliff_frame[q] = new_q;
                self.cliff_frame[r] = new_r;
                self.frame_phase = (self.frame_phase + phase) & 7;
            } else {
                self.flush_cliff_frame(q);
                self.flush_cliff_frame(r);
            }
        }
        // SXX = H*H * SZZ * H*H
        let q0s: Vec<QubitId> = pairs.iter().map(|p| p.0).collect();
        let q1s: Vec<QubitId> = pairs.iter().map(|p| p.1).collect();
        self.apply_clifford(|ch| {
            ch.h(&q0s);
            ch.h(&q1s);
        });
        self.apply_clifford_structural(|ch| {
            ch.szz(pairs);
        });
        self.apply_clifford(|ch| {
            ch.h(&q0s);
            ch.h(&q1s);
        });
        self
    }

    fn sxxdg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let q = q0.index();
            let r = q1.index();
            let fq = self.cliff_frame[q];
            let fr = self.cliff_frame[r];
            if fq.is_pauli() && fr.is_pauli() {
                let (new_q, new_r, phase) = CliffordFrame::push_through_sxx(fq, fr);
                self.cliff_frame[q] = new_q;
                self.cliff_frame[r] = new_r;
                self.frame_phase = (self.frame_phase + (8 - phase) % 8) & 7;
            } else {
                self.flush_cliff_frame(q);
                self.flush_cliff_frame(r);
            }
        }
        let q0s: Vec<QubitId> = pairs.iter().map(|p| p.0).collect();
        let q1s: Vec<QubitId> = pairs.iter().map(|p| p.1).collect();
        self.apply_clifford(|ch| {
            ch.h(&q0s);
            ch.h(&q1s);
        });
        self.apply_clifford_structural(|ch| {
            ch.szzdg(pairs);
        });
        self.apply_clifford(|ch| {
            ch.h(&q0s);
            ch.h(&q1s);
        });
        self
    }

    fn syy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let q = q0.index();
            let r = q1.index();
            let fq = self.cliff_frame[q];
            let fr = self.cliff_frame[r];
            if fq.is_pauli() && fr.is_pauli() {
                let (new_q, new_r, phase) = CliffordFrame::push_through_syy(fq, fr);
                self.cliff_frame[q] = new_q;
                self.cliff_frame[r] = new_r;
                self.frame_phase = (self.frame_phase + phase) & 7;
            } else {
                self.flush_cliff_frame(q);
                self.flush_cliff_frame(r);
            }
        }
        // SYY = S*S * SXX * Sdg*Sdg
        let all_qubits: Vec<QubitId> = pairs.iter().flat_map(|&(q0, q1)| [q0, q1]).collect();
        self.apply_clifford_structural(|ch| {
            ch.sz(&all_qubits);
        });
        self.sxx(pairs);
        self.apply_clifford_structural(|ch| {
            ch.szdg(&all_qubits);
        });
        self
    }

    fn syydg(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let q = q0.index();
            let r = q1.index();
            let fq = self.cliff_frame[q];
            let fr = self.cliff_frame[r];
            if fq.is_pauli() && fr.is_pauli() {
                let (new_q, new_r, phase) = CliffordFrame::push_through_syy(fq, fr);
                self.cliff_frame[q] = new_q;
                self.cliff_frame[r] = new_r;
                self.frame_phase = (self.frame_phase + (8 - phase) % 8) & 7;
            } else {
                self.flush_cliff_frame(q);
                self.flush_cliff_frame(r);
            }
        }
        let all_qubits: Vec<QubitId> = pairs.iter().flat_map(|&(q0, q1)| [q0, q1]).collect();
        self.apply_clifford_structural(|ch| {
            ch.sz(&all_qubits);
        });
        self.sxxdg(pairs);
        self.apply_clifford_structural(|ch| {
            ch.szdg(&all_qubits);
        });
        self
    }

    fn cy(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            self.flush_cliff_frame(q0.index());
            self.flush_cliff_frame(q1.index());
        }
        self.apply_clifford(|ch| {
            ch.cy(pairs);
        });
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits
            .iter()
            .map(|&q| self.measure_qubit(q.index(), None))
            .collect()
    }

    fn mnz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        // Measure -Z: flip outcome. If frame is diagonal, compose Z into frame.
        for &q in qubits {
            let qi = q.index();
            let old = self.cliff_frame[qi];
            self.frame_phase = (self.frame_phase
                + PHASE_COCYCLE[CliffordFrame::Z.index() as usize][old.index() as usize])
                & 7;
            self.cliff_frame[qi] = CliffordFrame::Z.compose(old);
        }
        self.mz(qubits)
    }

    fn pz(&mut self, qubits: &[QubitId]) -> &mut Self {
        // Prep |0⟩: discard diagonal frame and pending_rz (they don't survive reset).
        for &q in qubits {
            let qi = q.index();
            if self.cliff_frame[qi].is_diagonal() {
                // Diagonal frame: outcome might flip but we force |0⟩ anyway. Discard.
                self.cliff_frame[qi] = CliffordFrame::IDENTITY;
                self.pending_rz[qi] = Angle64::default();
            }
            // Non-diagonal: default mpz handles it (flushes everything).
        }
        self.mpz(qubits);
        self
    }

    fn pnz(&mut self, qubits: &[QubitId]) -> &mut Self {
        // Prep |1⟩: same as pz but flip.
        for &q in qubits {
            let qi = q.index();
            if self.cliff_frame[qi].is_diagonal() {
                self.cliff_frame[qi] = CliffordFrame::IDENTITY;
                self.pending_rz[qi] = Angle64::default();
            }
        }
        self.mpnz(qubits);
        self
    }
}

impl<S: IndexSet, R: SeedableRng + Rng + Debug + Clone> ArbitraryRotationGateable
    for StabVecGeneric<S, R>
{
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        // RX = H * RZ * H. Use frame-aware H and RZ.
        self.h(qubits);
        self.rz(theta, qubits);
        self.h(qubits);
        self
    }

    fn ry(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        // RY = Sdg * H * RZ * H * S. Use frame-aware gates.
        self.szdg(qubits);
        self.h(qubits);
        self.rz(theta, qubits);
        self.h(qubits);
        self.sz(qubits);
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        // RZ: flush frame if non-diagonal (it doesn't commute with RZ).
        // Diagonal frames (Pauli Z, S, Sdg) commute with RZ.
        for &q in qubits {
            let qi = q.index();
            let cf = self.cliff_frame[qi];
            if !cf.is_identity() && !cf.is_diagonal() {
                // Non-diagonal frame doesn't commute with RZ. Flush.
                self.flush_cliff_frame(qi);
            }
            // If frame anticommutes with Z (X or Y component), negate the angle.
            // Frame C: C†ZC = ±Z. If -Z, then C*RZ(θ) = RZ(-θ)*C.
            if !cf.is_identity() && cf.is_diagonal() && !cf.z_image().positive {
                self.apply_rz(-theta, qi);
            } else {
                self.apply_rz(theta, qi);
            }
        }
        self
    }

    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // RZZ = CX * RZ_tgt * CX. Use frame-aware CX and RZ.
        self.cx(pairs);
        let targets: Vec<QubitId> = pairs.iter().map(|p| p.1).collect();
        self.rz(theta, &targets);
        self.cx(pairs);
        self
    }

    fn rxx(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // RXX = H*H * RZZ * H*H. Use frame-aware gates.
        let q0s: Vec<QubitId> = pairs.iter().map(|p| p.0).collect();
        let q1s: Vec<QubitId> = pairs.iter().map(|p| p.1).collect();
        let both: Vec<QubitId> = q0s.iter().chain(q1s.iter()).copied().collect();
        self.h(&both);
        self.rzz(theta, pairs);
        self.h(&both);
        self
    }

    fn ryy(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // RYY = S*S * RXX * Sdg*Sdg. Use frame-aware gates.
        let q0s: Vec<QubitId> = pairs.iter().map(|p| p.0).collect();
        let q1s: Vec<QubitId> = pairs.iter().map(|p| p.1).collect();
        let both: Vec<QubitId> = q0s.iter().chain(q1s.iter()).copied().collect();
        self.sz(&both);
        self.rxx(theta, pairs);
        self.szdg(&both);
        self
    }
}

impl<S: IndexSet, R: SeedableRng + Rng + Debug + Clone> pecos_core::RngManageable
    for StabVecGeneric<S, R>
{
    type Rng = R;

    fn set_rng(&mut self, rng: Self::Rng) {
        self.rng = rng;
    }

    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)] // statistical tests use count as f64
mod tests {
    use super::*;
    use crate::StateVec;
    use pecos_core::qid;

    const EPS: f64 = 1e-8;

    /// Compare state vectors up to global phase.
    fn states_match_up_to_phase(a: &[Complex64], b: &[Complex64], label: &str) {
        // Find global phase ratio from first non-zero pair
        let mut ratio = None;
        for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
            if ai.norm() > EPS && bi.norm() > EPS {
                ratio = Some(bi / ai);
                break;
            }
            // Both should be zero or both non-zero
            assert!(
                (ai.norm() > EPS) == (bi.norm() > EPS),
                "{label}: amplitude[{i}] zero mismatch: a={ai:.6}, b={bi:.6}"
            );
        }

        if let Some(r) = ratio {
            for (i, (ai, bi)) in a.iter().zip(b.iter()).enumerate() {
                let diff = (ai * r - bi).norm();
                assert!(
                    diff < EPS,
                    "{label}: amplitude[{i}] mismatch after phase correction: \
                     a={ai:.6}, b={bi:.6}, ratio={r:.6}, diff={diff:.2e}"
                );
            }
        }
    }

    #[test]
    fn test_stab_vec_initial_state() {
        let mut sim = StabVec::new(2);
        assert_eq!(sim.num_terms(), 1);
        let sv = sim.state_vector();
        assert!((sv[0] - Complex64::new(1.0, 0.0)).norm() < EPS);
        assert!(sv[1].norm() < EPS);
    }

    #[test]
    fn test_clifford_only_matches_statevec() {
        let mut crz = StabVec::new(2);
        let mut sv = StateVec::new(2);

        // Apply Clifford circuit
        crz.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]).sz(&qid(1));
        sv.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]).sz(&qid(1));

        assert_eq!(crz.num_terms(), 1); // Still one term (no RZ)
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "clifford_only");
    }

    #[test]
    fn test_single_rz_doubles_terms() {
        let mut crz = StabVec::new(1);
        crz.h(&qid(0));
        assert_eq!(crz.num_terms(), 1);

        let theta = Angle64::from_radians(0.3);
        crz.rz(theta, &qid(0));
        // RZ is buffered; terms double on flush (e.g., before measurement)
        assert_eq!(crz.num_terms(), 1); // still 1 until flushed
        crz.flush_all_pending_rz();
        assert_eq!(crz.num_terms(), 2); // now doubled
    }

    #[test]
    fn test_rz_matches_statevec() {
        let mut crz = StabVec::new(1);
        let mut sv = StateVec::new(1);

        let theta = Angle64::from_radians(0.7);
        crz.h(&qid(0)).rz(theta, &qid(0));
        sv.h(&qid(0)).rz(theta, &qid(0));

        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "single_rz");
    }

    #[test]
    fn test_t_gate_matches_statevec() {
        let mut crz = StabVec::new(1);
        let mut sv = StateVec::new(1);

        // T gate = RZ(pi/4)
        let theta = Angle64::from_radians(std::f64::consts::FRAC_PI_4);
        crz.h(&qid(0)).rz(theta, &qid(0));
        sv.h(&qid(0)).rz(theta, &qid(0));

        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "t_gate");
    }

    #[test]
    fn test_multiple_rz_matches_statevec() {
        let mut crz = StabVec::new(2);
        let mut sv = StateVec::new(2);

        let theta1 = Angle64::from_radians(0.5);
        let theta2 = Angle64::from_radians(1.2);

        crz.h(&qid(0))
            .h(&qid(1))
            .rz(theta1, &qid(0))
            .rz(theta2, &qid(1));
        sv.h(&qid(0))
            .h(&qid(1))
            .rz(theta1, &qid(0))
            .rz(theta2, &qid(1));

        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "multiple_rz");
        assert_eq!(crz.num_terms(), 4); // 2 RZ on different qubits -> 4 terms after flush
    }

    #[test]
    fn test_rx_matches_statevec() {
        let mut crz = StabVec::new(1);
        let mut sv = StateVec::new(1);

        let theta = Angle64::from_radians(0.9);
        crz.rx(theta, &qid(0));
        sv.rx(theta, &qid(0));

        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "rx");
    }

    #[test]
    fn test_rzz_matches_statevec() {
        let mut crz = StabVec::new(2);
        let mut sv = StateVec::new(2);

        let theta = Angle64::from_radians(0.6);
        crz.h(&qid(0))
            .h(&qid(1))
            .rzz(theta, &[(QubitId(0), QubitId(1))]);
        sv.h(&qid(0))
            .h(&qid(1))
            .rzz(theta, &[(QubitId(0), QubitId(1))]);

        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "rzz");
    }

    #[test]
    fn test_mixed_stab_vec_circuit() {
        let mut crz = StabVec::new(2);
        let mut sv = StateVec::new(2);

        let theta = Angle64::from_radians(0.4);

        // H - CX - RZ - H - measure-like comparison
        crz.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .rz(theta, &qid(0))
            .h(&qid(0));
        sv.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .rz(theta, &qid(0))
            .h(&qid(0));

        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "mixed_circuit");
    }

    #[test]
    fn test_rz_clifford_angle_stays_one_term() {
        // RZ(0) = I: no term growth
        let mut crz = StabVec::new(1);
        crz.h(&qid(0));
        crz.rz(Angle64::from_radians(0.0), &qid(0));
        assert_eq!(crz.num_terms(), 1);

        // RZ(pi) = -iZ: no term growth
        let mut crz2 = StabVec::new(1);
        crz2.h(&qid(0));
        crz2.rz(Angle64::from_radians(std::f64::consts::PI), &qid(0));
        assert_eq!(crz2.num_terms(), 1, "RZ(pi) should not add terms");

        // RZ(pi/2) = e^{-i*pi/4} S: no term growth
        let mut crz3 = StabVec::new(1);
        crz3.h(&qid(0));
        crz3.rz(Angle64::from_radians(std::f64::consts::FRAC_PI_2), &qid(0));
        assert_eq!(crz3.num_terms(), 1, "RZ(pi/2) should not add terms");

        // RZ(-pi/2) = e^{i*pi/4} Sdg: no term growth
        let mut crz4 = StabVec::new(1);
        crz4.h(&qid(0));
        crz4.rz(Angle64::from_radians(-std::f64::consts::FRAC_PI_2), &qid(0));
        assert_eq!(crz4.num_terms(), 1, "RZ(-pi/2) should not add terms");
    }

    // ========================================================================
    // Measurement tests
    // ========================================================================

    #[test]
    fn test_measurement_deterministic_zero_state() {
        let mut crz = StabVec::new_with_seed(1, 42);
        let results = crz.mz(&qid(0));
        assert!(results[0].is_deterministic);
        assert!(!results[0].outcome); // |0>
    }

    #[test]
    fn test_measurement_after_rz() {
        // RZ(theta) on |0> gives e^{-i*theta/2}|0> -- still deterministic |0>
        let mut crz = StabVec::new_with_seed(1, 42);
        let theta = Angle64::from_radians(0.7);
        crz.rz(theta, &qid(0));
        let results = crz.mz(&qid(0));
        assert!(results[0].is_deterministic);
        assert!(!results[0].outcome);
    }

    #[test]
    fn test_measurement_after_h_rz() {
        // H|0> then RZ should give non-deterministic measurement
        let mut crz = StabVec::new_with_seed(1, 42);
        let theta = Angle64::from_radians(0.5);
        crz.h(&qid(0)).rz(theta, &qid(0));
        let results = crz.mz(&qid(0));
        assert!(!results[0].is_deterministic);
    }

    #[test]
    fn test_measurement_statistics() {
        // H|0> then RZ(theta): Pr(0) = cos^2(theta/2), Pr(1) = sin^2(theta/2)
        // Wait, that's wrong -- H*RZ*|0> = cos(t/2)|+> - i*sin(t/2)|->
        // Pr(0) = |<0|psi>|^2 = |cos(t/2)/sqrt(2) - i*sin(t/2)/sqrt(2)|^2
        //       = (cos^2+sin^2)/2 = 1/2
        // So Pr(0) = 1/2 regardless of theta! That's because RZ is diagonal and
        // H|0>=|+> has equal amplitudes. Let me use a circuit that gives unequal probs.
        //
        // Better: |0> -> RX(theta) -> MZ
        // RX(theta)|0> = cos(t/2)|0> - i*sin(t/2)|1>
        // Pr(0) = cos^2(t/2), Pr(1) = sin^2(t/2)

        let theta = Angle64::from_radians(1.0); // ~cos^2(0.5) ≈ 0.7702
        let expected_p0 = (0.5f64).cos().powi(2);
        let num_shots = 10000;
        let mut count0 = 0;

        for seed in 0..num_shots {
            let mut crz = StabVec::new_with_seed(1, seed);
            crz.rx(theta, &qid(0));
            let results = crz.mz(&qid(0));
            if !results[0].outcome {
                count0 += 1;
            }
        }

        let observed_p0 = f64::from(count0) / num_shots as f64;
        let tolerance = 3.0 / (num_shots as f64).sqrt(); // ~3 sigma
        assert!(
            (observed_p0 - expected_p0).abs() < tolerance,
            "Measurement statistics: expected p0={expected_p0:.4}, observed={observed_p0:.4}, \
             tolerance={tolerance:.4}"
        );
    }

    #[test]
    fn test_measurement_bell_state_with_rz() {
        // Create Bell state, apply RZ on q0, measure both.
        // After measuring q0, q1 outcome should be correlated.
        let theta = Angle64::from_radians(0.6);
        let mut crz = StabVec::new_with_seed(2, 42);
        crz.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .rz(theta, &qid(0));

        // Compute state vector to verify it's correct
        let mut sv = StateVec::new(2);
        sv.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .rz(theta, &qid(0));
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "bell_rz_before_meas");

        // Measure q0 -- state should still have correlation structure
        let r0 = crz.mz(&qid(0));
        let r1 = crz.mz(&qid(1));
        // After Bell+RZ, the state is (cos|00> - i*sin|00> + cos|11> + i*sin|11>)/sqrt(2)
        // Wait, let me think... RZ on q0 of Bell:
        // RZ(t)|Bell> = (e^{-it/2}|00> + e^{it/2}|11>)/sqrt(2)
        // So Pr(00) = 1/2, Pr(11) = 1/2. Outcomes are always correlated!
        assert!(
            r1[0].is_deterministic,
            "q1 should be deterministic after q0 measurement"
        );
        assert_eq!(
            r0[0].outcome, r1[0].outcome,
            "Bell+RZ: q0 and q1 should be correlated"
        );
    }

    #[test]
    fn test_mid_circuit_measurement() {
        // Measure, then apply more gates.
        // |0> -> H -> RZ(0.5) -> MZ(force 0) -> H -> MZ
        // After measuring 0, state is |0>. After H, state is |+>.
        // Second measurement should be non-deterministic (50/50).
        let theta = Angle64::from_radians(0.5);

        let mut crz = StabVec::new_with_seed(1, 42);
        crz.h(&qid(0)).rz(theta, &qid(0));

        // Force measurement outcome to 0
        let result = crz.measure_qubit(0, Some(false));
        assert!(!result.outcome);

        // After measuring |0>, apply H -> should give |+> (non-deterministic).
        crz.h(&qid(0));

        // Check the state vector is normalized
        let sv = crz.state_vector();
        let norm: f64 = sv.iter().map(num_complex::Complex::norm_sqr).sum();
        assert!(
            (norm - 1.0).abs() < 0.01,
            "State should be normalized after mid-circuit meas + H, got norm={norm:.4}"
        );

        // Both amplitudes should have equal magnitude (|+> up to phase)
        assert!(
            (sv[0].norm() - sv[1].norm()).abs() < 0.01,
            "After mid-circuit meas + H: |amp[0]|={:.4} should equal |amp[1]|={:.4}",
            sv[0].norm(),
            sv[1].norm()
        );
    }

    #[test]
    fn test_three_qubit_circuit() {
        let mut crz = StabVec::new(3);
        let mut sv = StateVec::new(3);

        let theta = Angle64::from_radians(0.8);

        // GHZ-like circuit with RZ
        crz.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .cx(&[(QubitId(1), QubitId(2))])
            .rz(theta, &qid(1));
        sv.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .cx(&[(QubitId(1), QubitId(2))])
            .rz(theta, &qid(1));

        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "3qubit_ghz_rz");
    }

    #[test]
    fn test_reset() {
        let mut crz = StabVec::new(2);
        let theta = Angle64::from_radians(0.5);
        crz.h(&qid(0)).rz(theta, &qid(0));
        crz.flush_all_pending_rz();
        assert_eq!(crz.num_terms(), 2);

        crz.reset();
        assert_eq!(crz.num_terms(), 1);
        let sv = crz.state_vector();
        assert!((sv[0] - Complex64::new(1.0, 0.0)).norm() < EPS);
    }

    #[test]
    fn test_rz_at_clifford_angles_vs_statevec() {
        // RZ(pi/2) should be equivalent to S (up to global phase)
        let mut crz = StabVec::new(1);
        let mut sv = StateVec::new(1);
        let half_pi = Angle64::from_radians(std::f64::consts::FRAC_PI_2);
        crz.h(&qid(0)).rz(half_pi, &qid(0));
        sv.h(&qid(0)).rz(half_pi, &qid(0));
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "rz_pi_2");

        // RZ(pi) should be equivalent to Z (up to global phase)
        let mut crz2 = StabVec::new(1);
        let mut sv2 = StateVec::new(1);
        let pi = Angle64::from_radians(std::f64::consts::PI);
        crz2.h(&qid(0)).rz(pi, &qid(0));
        sv2.h(&qid(0)).rz(pi, &qid(0));
        states_match_up_to_phase(&crz2.state_vector(), &sv2.state(), "rz_pi");
    }

    #[test]
    fn test_many_rz_gates() {
        // 5 RZ gates -> 32 terms. Verify state still matches StateVec.
        let mut crz = StabVec::new(2);
        let mut sv = StateVec::new(2);

        let angles: Vec<Angle64> = [0.3, 0.7, 1.1, 0.5, 0.9]
            .iter()
            .map(|&a| Angle64::from_radians(a))
            .collect();

        // Interleave Clifford and RZ gates
        crz.h(&qid(0)).h(&qid(1));
        sv.h(&qid(0)).h(&qid(1));

        crz.rz(angles[0], &qid(0));
        sv.rz(angles[0], &qid(0));

        crz.cx(&[(QubitId(0), QubitId(1))]);
        sv.cx(&[(QubitId(0), QubitId(1))]);

        crz.rz(angles[1], &qid(1));
        sv.rz(angles[1], &qid(1));

        crz.rz(angles[2], &qid(0));
        sv.rz(angles[2], &qid(0));

        crz.h(&qid(0));
        sv.h(&qid(0));

        crz.rz(angles[3], &qid(0));
        sv.rz(angles[3], &qid(0));

        crz.rz(angles[4], &qid(1));
        sv.rz(angles[4], &qid(1));

        // With RZ fusion + commutation, same-qubit rotations merge even through
        // commuting Cliffords. a0+a2 fuse (through CX control), a1+a4 fuse.
        // Result: 3 independent RZ -> 2^3 = 8 terms.
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "5_rz_gates");
        assert_eq!(crz.num_terms(), 8);
    }

    #[test]
    fn test_measurement_probability_matches_statevec() {
        // Compare exact measurement probabilities between StabVec and StateVec.
        // Circuit: H(0) - CX(0,1) - RZ(0.8, q0) - H(1)
        // Then compute Pr(q0=0) and Pr(q1=0) from both simulators.
        let theta = Angle64::from_radians(0.8);

        let mut crz = StabVec::new(2);
        let mut sv = StateVec::new(2);

        crz.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .rz(theta, &qid(0))
            .h(&qid(1));
        sv.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .rz(theta, &qid(0))
            .h(&qid(1));

        let crz_sv = crz.state_vector();
        let sv_sv = sv.state();

        // Pr(q0=0) = sum |amp[x]|^2 for x with bit 0 = 0
        for q in 0..2 {
            let crz_p0: f64 = crz_sv
                .iter()
                .enumerate()
                .filter(|(x, _)| (x >> q) & 1 == 0)
                .map(|(_, a)| a.norm_sqr())
                .sum();
            let sv_p0: f64 = sv_sv
                .iter()
                .enumerate()
                .filter(|(x, _)| (x >> q) & 1 == 0)
                .map(|(_, a)| a.norm_sqr())
                .sum();
            assert!(
                (crz_p0 - sv_p0).abs() < EPS,
                "Pr(q{q}=0): crz={crz_p0:.6}, sv={sv_p0:.6}"
            );
        }
    }

    #[test]
    fn test_post_measurement_state_matches_statevec() {
        // After forced measurement, compare the projected state vectors.
        let theta = Angle64::from_radians(0.6);

        let mut crz = StabVec::new(2);
        let mut sv = StateVec::new(2);

        crz.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .rz(theta, &qid(0));
        sv.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .rz(theta, &qid(0));

        // Force q0 = 0 on StabVec
        crz.measure_qubit(0, Some(false));

        // For StateVec, project manually: zero out amplitudes where q0=1, renormalize
        let mut sv_state = sv.state();
        for (x, amp) in sv_state.iter_mut().enumerate() {
            if x & 1 == 1 {
                *amp = Complex64::new(0.0, 0.0);
            }
        }
        let norm_sq: f64 = sv_state.iter().map(num_complex::Complex::norm_sqr).sum();
        let inv_norm = 1.0 / norm_sq.sqrt();
        for a in &mut sv_state {
            *a *= inv_norm;
        }

        // Compare post-measurement state vectors (up to global phase)
        states_match_up_to_phase(&crz.state_vector(), &sv_state, "post_measurement");
    }

    #[test]
    fn test_measurement_does_not_corrupt_other_qubits() {
        // 3-qubit circuit: measure q0, verify q1 and q2 state is correct.
        let theta = Angle64::from_radians(0.5);

        let mut crz = StabVec::new(3);
        let mut sv = StateVec::new(3);

        // Prepare: H(0) CX(0,1) RZ(q2) -- q2 is independent
        crz.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .h(&qid(2))
            .rz(theta, &qid(2));
        sv.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .h(&qid(2))
            .rz(theta, &qid(2));

        // Force q0 = 0 on StabVec
        crz.measure_qubit(0, Some(false));

        // Project StateVec manually: zero amplitudes where q0=1, renormalize
        let mut sv_state = sv.state();
        for (x, amp) in sv_state.iter_mut().enumerate() {
            if x & 1 == 1 {
                *amp = Complex64::new(0.0, 0.0);
            }
        }
        let norm_sq: f64 = sv_state.iter().map(num_complex::Complex::norm_sqr).sum();
        let inv_norm = 1.0 / norm_sq.sqrt();
        for a in &mut sv_state {
            *a *= inv_norm;
        }

        // Post-measurement states should match (up to global phase)
        states_match_up_to_phase(&crz.state_vector(), &sv_state, "no_corruption");
    }

    #[test]
    fn test_measurement_statistics_2qubit() {
        // Verify measurement distribution on an entangled+rotated 2-qubit state.
        // Circuit: H(0) - RZ(theta, q0) - CX(0,1)
        // This creates a state where Pr(00) != Pr(11) (not a standard Bell state).
        let theta = Angle64::from_radians(1.0);
        let num_shots = 5000;

        // Compute expected probabilities from state vector
        let mut sv = StateVec::new(2);
        sv.h(&qid(0))
            .rz(theta, &qid(0))
            .cx(&[(QubitId(0), QubitId(1))]);
        let sv_state = sv.state();
        let expected_probs: Vec<f64> = sv_state
            .iter()
            .map(num_complex::Complex::norm_sqr)
            .collect();

        // Sample from StabVec
        let mut counts = [0u32; 4];
        for seed in 0..num_shots {
            let mut crz = StabVec::new_with_seed(2, seed);
            crz.h(&qid(0))
                .rz(theta, &qid(0))
                .cx(&[(QubitId(0), QubitId(1))]);
            let r0 = crz.mz(&qid(0));
            let r1 = crz.mz(&qid(1));
            let outcome = usize::from(r0[0].outcome) | (usize::from(r1[0].outcome) << 1);
            counts[outcome] += 1;
        }

        let tolerance = 4.0 / (num_shots as f64).sqrt(); // ~4 sigma
        for (i, (&count, &expected)) in counts.iter().zip(expected_probs.iter()).enumerate() {
            let observed = f64::from(count) / num_shots as f64;
            assert!(
                (observed - expected).abs() < tolerance,
                "2qubit stats: Pr({i:02b}) expected={expected:.4}, observed={observed:.4}, tol={tolerance:.4}"
            );
        }
    }

    #[test]
    fn test_ry_gate() {
        // RY uses default decomposition: Sdg RX Sz.
        // RX uses our H RZ H. So this tests the full chain.
        let mut crz = StabVec::new(1);
        let mut sv = StateVec::new(1);

        let theta = Angle64::from_radians(1.2);
        crz.ry(theta, &qid(0));
        sv.ry(theta, &qid(0));

        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "ry_gate");
    }

    #[test]
    fn test_rz_after_measurement() {
        // Measure, then apply RZ, then compare state vector with manual computation.
        // |0> -> H -> MZ(force 0) -> RZ(theta) -> compare
        // After measuring 0, state is |0>. RZ(theta)|0> = e^{-i*theta/2}|0>.
        // State vector should have amp[0] = e^{-i*theta/2}, amp[1] = 0.
        let theta = Angle64::from_radians(0.8);

        let mut crz = StabVec::new_with_seed(1, 42);
        crz.h(&qid(0));
        crz.measure_qubit(0, Some(false));
        crz.rz(theta, &qid(0));

        let sv = crz.state_vector();
        // Should be normalized
        let norm: f64 = sv.iter().map(num_complex::Complex::norm_sqr).sum();
        assert!(
            (norm - 1.0).abs() < EPS,
            "norm after meas+RZ should be 1, got {norm}"
        );
        // amp[1] should be 0 (deterministic |0> rotated stays in |0>)
        assert!(sv[1].norm() < EPS, "amp[1] should be 0 after meas(0)+RZ");
        // amp[0] should have magnitude 1
        assert!((sv[0].norm() - 1.0).abs() < EPS, "|amp[0]| should be 1");
    }

    #[test]
    fn test_rz_after_measurement_nondeterministic() {
        // H -> RZ -> MZ(force 0) -> H -> RZ -> compare with projected state
        let theta1 = Angle64::from_radians(0.5);
        let theta2 = Angle64::from_radians(0.9);

        let mut crz = StabVec::new_with_seed(1, 42);
        crz.h(&qid(0)).rz(theta1, &qid(0));
        crz.measure_qubit(0, Some(false));
        // After projecting to |0>, apply H -> RZ
        crz.h(&qid(0)).rz(theta2, &qid(0));

        // Build reference: |0> -> H -> RZ(theta2)
        let mut sv = StateVec::new(1);
        sv.h(&qid(0)).rz(theta2, &qid(0));

        // States should match up to global phase
        states_match_up_to_phase(
            &crz.state_vector(),
            &sv.state(),
            "rz_after_nondeterministic_meas",
        );
    }

    #[test]
    fn test_rzz_then_measurement() {
        // Verify measurement after RZZ gives correct statistics.
        // H(0) H(1) - RZZ(theta) - MZ(0) MZ(1)
        let theta = Angle64::from_radians(0.7);

        // Compute expected probabilities from StateVec
        let mut sv = StateVec::new(2);
        sv.h(&qid(0))
            .h(&qid(1))
            .rzz(theta, &[(QubitId(0), QubitId(1))]);
        let sv_state = sv.state();
        let expected_probs: Vec<f64> = sv_state
            .iter()
            .map(num_complex::Complex::norm_sqr)
            .collect();

        // Verify StabVec state matches before measurement
        let mut crz = StabVec::new(2);
        crz.h(&qid(0))
            .h(&qid(1))
            .rzz(theta, &[(QubitId(0), QubitId(1))]);
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "rzz_before_meas");

        // Sample and check statistics
        let num_shots = 5000;
        let mut counts = [0u32; 4];
        for seed in 0..num_shots {
            let mut crz = StabVec::new_with_seed(2, seed);
            crz.h(&qid(0))
                .h(&qid(1))
                .rzz(theta, &[(QubitId(0), QubitId(1))]);
            let r0 = crz.mz(&qid(0));
            let r1 = crz.mz(&qid(1));
            let outcome = usize::from(r0[0].outcome) | (usize::from(r1[0].outcome) << 1);
            counts[outcome] += 1;
        }

        let tolerance = 4.0 / (num_shots as f64).sqrt();
        for (i, (&count, &expected)) in counts.iter().zip(expected_probs.iter()).enumerate() {
            let observed = f64::from(count) / num_shots as f64;
            assert!(
                (observed - expected).abs() < tolerance,
                "RZZ stats: Pr({i:02b}) expected={expected:.4}, observed={observed:.4}"
            );
        }
    }

    #[test]
    fn test_5_qubit_circuit() {
        // Verify StabVec works at 5 qubits with entanglement and RZ gates.
        let mut crz = StabVec::new(5);
        let mut sv = StateVec::new(5);

        let theta1 = Angle64::from_radians(0.4);
        let theta2 = Angle64::from_radians(1.1);

        // Build an entangled 5-qubit state with RZ gates
        for q in 0..5 {
            crz.h(&[QubitId(q)]);
            sv.h(&[QubitId(q)]);
        }
        for q in 0..4 {
            crz.cx(&[(QubitId(q), QubitId(q + 1))]);
            sv.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
        crz.rz(theta1, &[QubitId(1)]);
        sv.rz(theta1, &[QubitId(1)]);
        crz.rz(theta2, &[QubitId(3)]);
        sv.rz(theta2, &[QubitId(3)]);

        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "5_qubit_circuit");
    }

    #[test]
    fn test_5_qubit_measurement() {
        // Measure all 5 qubits after Clifford+RZ circuit, verify normalization.
        let theta = Angle64::from_radians(0.6);

        let mut crz = StabVec::new_with_seed(5, 42);
        crz.h(&[QubitId(0)])
            .cx(&[(QubitId(0), QubitId(1))])
            .cx(&[(QubitId(1), QubitId(2))])
            .rz(theta, &[QubitId(0)])
            .h(&[QubitId(3)])
            .cx(&[(QubitId(3), QubitId(4))]);

        // Measure all qubits
        let results = crz.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3), QubitId(4)]);
        assert_eq!(results.len(), 5);

        // After measuring all qubits, state should be a computational basis state (normalized)
        let sv = crz.state_vector();
        let norm: f64 = sv.iter().map(num_complex::Complex::norm_sqr).sum();
        assert!(
            (norm - 1.0).abs() < EPS,
            "5-qubit post-measurement norm = {norm}"
        );

        // Exactly one amplitude should be non-zero
        let nonzero_count = sv.iter().filter(|a| a.norm() > EPS).count();
        assert_eq!(
            nonzero_count, 1,
            "After measuring all qubits, should have exactly 1 nonzero amplitude"
        );
    }

    #[test]
    fn test_builder_default() {
        let mut sim = StabVec::builder(2).build();
        sim.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
        let sv = sim.state_vector();
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        assert!((sv[0].norm() - inv_sqrt2).abs() < EPS);
    }

    #[test]
    fn test_builder_with_seed() {
        let mut sim1 = StabVec::builder(1).seed(42).build();
        let mut sim2 = StabVec::builder(1).seed(42).build();
        sim1.h(&qid(0));
        sim2.h(&qid(0));
        let r1 = sim1.mz(&qid(0));
        let r2 = sim2.mz(&qid(0));
        assert_eq!(
            r1[0].outcome, r2[0].outcome,
            "Same seed should give same outcome"
        );
    }

    #[test]
    fn test_builder_exact_mode() {
        // With threshold=0 (exact mode), no terms are pruned even for small angles.
        let theta = Angle64::from_radians(0.001);
        let mut sim = StabVec::builder(1).pruning_threshold(0.0).seed(42).build();
        sim.h(&qid(0));
        for _ in 0..8 {
            sim.rz(theta, &qid(0));
        }
        sim.flush_all_pending_rz();
        // With exact mode, all terms survive (no pruning). Each RZ gives 2 terms.
        // But same-qubit fusion reduces 8 RZ to 1 RZ -> 2 terms.
        assert_eq!(sim.num_terms(), 2);
    }

    #[test]
    fn test_builder_aggressive_pruning() {
        // With aggressive pruning, small-angle terms are removed faster.
        let theta = Angle64::from_radians(5.0f64.to_radians());
        let mut sim = StabVec::builder(4).pruning_threshold(1e-4).seed(42).build();
        for q in 0..4 {
            sim.h(&[QubitId(q)]);
        }
        // Apply 4 small RZ on different qubits
        for q in 0..4 {
            sim.rz(theta, &[QubitId(q)]);
        }
        sim.flush_all_pending_rz();
        // With aggressive pruning, many of the 16 terms get pruned
        assert!(
            sim.num_terms() < 16,
            "Aggressive pruning should reduce term count"
        );
    }

    #[test]
    fn test_pz_prep() {
        // X|0> = |1>, then PZ resets to |0>
        let mut crz = StabVec::new(1);
        crz.x(&qid(0));
        crz.pz(&qid(0));
        let results = crz.mz(&qid(0));
        assert!(results[0].is_deterministic);
        assert!(!results[0].outcome, "PZ should reset to |0>");
    }

    #[test]
    fn test_rxx_matches_statevec() {
        let mut crz = StabVec::new(2);
        let mut sv = StateVec::new(2);
        let theta = Angle64::from_radians(0.7);
        crz.h(&qid(0))
            .h(&qid(1))
            .rxx(theta, &[(QubitId(0), QubitId(1))]);
        sv.h(&qid(0))
            .h(&qid(1))
            .rxx(theta, &[(QubitId(0), QubitId(1))]);
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "rxx");
    }

    #[test]
    fn test_ryy_matches_statevec() {
        let mut crz = StabVec::new(2);
        let mut sv = StateVec::new(2);
        let theta = Angle64::from_radians(0.9);
        crz.h(&qid(0))
            .h(&qid(1))
            .ryy(theta, &[(QubitId(0), QubitId(1))]);
        sv.h(&qid(0))
            .h(&qid(1))
            .ryy(theta, &[(QubitId(0), QubitId(1))]);
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "ryy");
    }

    #[test]
    fn test_exact_mode_matches_statevec() {
        // With pruning_threshold=0, results should match StateVec exactly (up to phase).
        let mut crz = StabVec::builder(2).pruning_threshold(0.0).seed(42).build();
        let mut sv = StateVec::new(2);
        let theta = Angle64::from_radians(0.3);
        crz.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .rz(theta, &qid(0))
            .h(&qid(1))
            .rz(theta, &qid(1));
        sv.h(&qid(0))
            .cx(&[(QubitId(0), QubitId(1))])
            .rz(theta, &qid(0))
            .h(&qid(1))
            .rz(theta, &qid(1));
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "exact_mode");
    }

    // ========================================================================
    // Qubit range coverage tests
    // ========================================================================

    /// Test `StabVec` at qubit counts that exercise the pairwise inner product
    /// measurement path (n>6) and various `ExponentialSum` tiers.
    #[test]
    fn test_stab_vec_medium_qubit_counts() {
        // These exercise: n>6 pairwise measurement, ExponentialSum d>3 path
        for nq in [8, 10, 14, 20] {
            let mut crz = StabVec::new_with_seed(nq, 42);
            let mut sv = StateVec::new(nq);
            let theta = Angle64::from_radians(0.5);

            // H on all, CX chain, RZ on q0
            for q in 0..nq {
                crz.h(&[QubitId(q)]);
                sv.h(&[QubitId(q)]);
            }
            if nq > 1 {
                crz.cx(&[(QubitId(0), QubitId(1))]);
                sv.cx(&[(QubitId(0), QubitId(1))]);
            }
            crz.rz(theta, &[QubitId(0)]);
            sv.rz(theta, &[QubitId(0)]);

            states_match_up_to_phase(&crz.state_vector(), &sv.state(), &format!("{nq}q"));
        }
    }

    #[test]
    fn test_high_depth_measurement_matches_statevec() {
        // Verify measurement statistics with many RZ gates (4-8 terms) match StateVec.
        // This exercises the early-skip optimization and precomputed constraints.
        for nrz in [2, 3, 4] {
            let nq = 8; // pairwise path (n > 6)
            let theta = Angle64::from_radians(0.3);
            let mut crz_p0_sum = 0.0;
            let nshots = 5000;
            for seed in 0..nshots {
                let mut crz = StabVec::new_with_seed(nq, seed);
                for q in 0..nq {
                    crz.h(&[QubitId(q)]);
                }
                if nq > 1 {
                    crz.cx(&[(QubitId(0), QubitId(1))]);
                }
                for r in 0..nrz {
                    crz.rz(theta, &[QubitId(r % nq)]);
                }
                let results = crz.mz(&[QubitId(0)]);
                if !results[0].outcome {
                    crz_p0_sum += 1.0;
                }
            }
            // Compare to StateVec probability
            let mut sv = StateVec::new(nq);
            for q in 0..nq {
                sv.h(&[QubitId(q)]);
            }
            if nq > 1 {
                sv.cx(&[(QubitId(0), QubitId(1))]);
            }
            for r in 0..nrz {
                sv.rz(theta, &[QubitId(r % nq)]);
            }
            let sv_p0: f64 = sv
                .state()
                .iter()
                .enumerate()
                .filter(|(x, _)| x & 1 == 0)
                .map(|(_, a)| a.norm_sqr())
                .sum();
            let crz_p0 = crz_p0_sum / nshots as f64;
            assert!(
                (crz_p0 - sv_p0).abs() < 0.05,
                "nrz={nrz}: Pr(q0=0) StabVec={crz_p0:.3} vs StateVec={sv_p0:.3}"
            );
        }
    }

    #[test]
    fn test_high_depth_renormalization() {
        // After measurement with many terms, verify the post-measurement state
        // is correctly normalized by checking that subsequent measurements work.
        let nq = 8;
        let theta = Angle64::from_radians(0.4);
        for nrz in [3, 4, 5] {
            let mut crz = StabVec::new_with_seed(nq, 42);
            for q in 0..nq {
                crz.h(&[QubitId(q)]);
            }
            for r in 0..nrz {
                crz.rz(theta, &[QubitId(r)]);
            }
            // First measurement
            let _ = crz.mz(&[QubitId(0)]);
            // State should still be valid -- second measurement should work
            let results = crz.mz(&[QubitId(1)]);
            assert!(
                results.len() == 1,
                "nrz={nrz}: second measurement should succeed"
            );
        }
    }

    #[test]
    fn test_stab_vec_measurement_at_pairwise_threshold() {
        // n=7 (state vector path) and n=8 (pairwise path) should both work
        for nq in [6, 7, 8] {
            let theta = Angle64::from_radians(0.5);
            let mut crz = StabVec::new_with_seed(nq, 42);
            for q in 0..nq {
                crz.h(&[QubitId(q)]);
            }
            crz.rz(theta, &[QubitId(0)]);
            let results = crz.mz(&[QubitId(0)]);
            let _ = results[0].outcome; // just verify measurement completes
        }
    }

    #[test]
    fn test_stab_vec_at_u64_boundary() {
        // n=62 (last u64 ExponentialSum) -- verify measurement works
        let nq = 62;
        let mut crz = StabVec::new_with_seed(nq, 42);
        for q in 0..nq {
            crz.h(&[QubitId(q)]);
        }
        crz.rz(Angle64::from_radians(0.3), &[QubitId(0)]);
        let results = crz.mz(&[QubitId(0)]);
        assert!(results.len() == 1);
    }

    #[test]
    fn test_stab_vec_at_u128_boundary() {
        // n=63 (first u128 ExponentialSum) -- verify measurement works
        let nq = 63;
        let mut crz = StabVec::new_with_seed(nq, 42);
        for q in 0..nq {
            crz.h(&[QubitId(q)]);
        }
        crz.rz(Angle64::from_radians(0.3), &[QubitId(0)]);
        let results = crz.mz(&[QubitId(0)]);
        assert!(results.len() == 1);
    }

    #[test]
    fn test_ry_simple() {
        // Just H then RY on 1 qubit
        let mut crz = StabVec::new(1);
        let mut sv = StateVec::new(1);
        crz.h(&qid(0));
        sv.h(&qid(0));
        crz.ry(Angle64::from_radians(0.3), &qid(0));
        sv.ry(Angle64::from_radians(0.3), &qid(0));
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "h_ry");
    }

    #[test]
    fn test_ry_on_zero() {
        // RY on |0> should match statevec
        let mut crz = StabVec::new(1);
        let mut sv = StateVec::new(1);
        crz.ry(Angle64::from_radians(0.3), &qid(0));
        sv.ry(Angle64::from_radians(0.3), &qid(0));
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "ry_on_zero");
    }

    #[test]
    fn test_engine_circuit_statevec_match() {
        // Reproduce the engine round-trip circuit: H, CX, RZ, H, RY, CZ, RX
        let mut crz = StabVec::new(3);
        let mut sv = StateVec::new(3);

        crz.h(&[QubitId(0), QubitId(1), QubitId(2)]);
        sv.h(&[QubitId(0), QubitId(1), QubitId(2)]);
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "after H");

        crz.cx(&[(QubitId(0), QubitId(1))]);
        sv.cx(&[(QubitId(0), QubitId(1))]);
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "after CX01");

        crz.cx(&[(QubitId(1), QubitId(2))]);
        sv.cx(&[(QubitId(1), QubitId(2))]);
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "after CX12");

        crz.rz(Angle64::from_radians(0.5), &[QubitId(0)]);
        sv.rz(Angle64::from_radians(0.5), &[QubitId(0)]);
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "after RZ0");

        crz.rz(Angle64::from_radians(0.8), &[QubitId(2)]);
        sv.rz(Angle64::from_radians(0.8), &[QubitId(2)]);
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "after RZ2");

        crz.h(&[QubitId(1)]);
        sv.h(&[QubitId(1)]);
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "after H1");

        crz.ry(Angle64::from_radians(0.3), &[QubitId(1)]);
        sv.ry(Angle64::from_radians(0.3), &[QubitId(1)]);
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "after RY1");

        crz.cz(&[(QubitId(0), QubitId(2))]);
        sv.cz(&[(QubitId(0), QubitId(2))]);
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "after CZ02");

        crz.rx(Angle64::from_radians(0.6), &[QubitId(0)]);
        sv.rx(Angle64::from_radians(0.6), &[QubitId(0)]);
        states_match_up_to_phase(&crz.state_vector(), &sv.state(), "after RX0");
    }
}
