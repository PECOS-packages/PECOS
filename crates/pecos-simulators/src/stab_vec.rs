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
use crate::clifford_frame::{CliffordFrame, GATE_PHASE_DELTA, GEN_LENS, GENERATORS, PHASE_COCYCLE};

#[derive(Clone, Debug)]
pub struct StabVecGeneric<S: IndexSet = BitSet, R: SeedableRng + Rng + Debug = PecosRng> {
    num_qubits: usize,
    terms: Vec<(Complex64, CHFormGeneric<S, R>)>,
    /// Pending RZ angles per qubit.
    pending_rz: Vec<Angle64>,
    /// Single-qubit Clifford frame per qubit. All 24 Clifford elements tracked.
    /// State = `pending_rz` * frame * |`stored_state`⟩.
    /// Single-qubit Cliffords compose into the frame in O(1).
    /// Flushed via H+S generator sequence when a two-qubit gate or measurement arrives.
    cliff_frame: Vec<CliffordFrame>,
    /// Global phase from frame compositions: e^{i*`frame_phase`*pi/4}, mod 8.
    frame_phase: u8,
    /// Other global phase accumulated by phase-exact default decompositions.
    global_phase: Angle64,
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
            global_phase: Angle64::ZERO,
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

    /// Whether every term currently shares the structural inputs used by Z projection.
    ///
    /// This is exposed for correctness tests that must prove both projection
    /// dispatch paths are represented in their corpus.
    #[doc(hidden)]
    #[must_use]
    pub fn has_shared_projection_structure(&self) -> bool {
        self.terms.first().is_none_or(|(_, first)| {
            self.terms[1..]
                .iter()
                .all(|(_, ch)| ch.shares_projection_structure(first))
        })
    }

    /// Compute the exact norm and Z=0 probability of the represented state.
    ///
    /// Both quantities use the same pairwise CH-form overlaps. The shared-
    /// structure implementation is selected only when pointer equality proves
    /// its precondition; structurally divergent terms use the general overlap.
    fn exact_norm_and_prob0(&self, q: usize) -> (f64, f64) {
        let shared_structure = self.has_shared_projection_structure();
        let shared_constraints =
            shared_structure.then(|| self.terms[0].1.precompute_shared_constraints());
        let omegas: Vec<_> = if shared_structure {
            self.terms
                .iter()
                .map(|(_, ch)| ch.omega_complex())
                .collect()
        } else {
            Vec::new()
        };

        let mut norm_sq = 0.0;
        let mut twice_prob0 = 0.0;
        for (coefficient, ch) in &self.terms {
            let weight = coefficient.norm_sqr();
            norm_sq += weight;
            twice_prob0 += weight * (1.0 + ch.expectation_value_zq(q));
        }
        for j in 0..self.terms.len() {
            for k in (j + 1)..self.terms.len() {
                let (inner, inner_z) = if let Some(constraints) = &shared_constraints {
                    self.terms[j].1.inner_product_pair_precomputed(
                        &self.terms[k].1,
                        q,
                        constraints,
                        omegas[j],
                        omegas[k],
                        Some(&self.gamma_diff_qubits),
                    )
                } else {
                    self.terms[j].1.inner_product_pair(&self.terms[k].1, q)
                };
                let coefficient_product = self.terms[j].0.conj() * self.terms[k].0;
                norm_sq += 2.0 * (coefficient_product * inner).re;
                twice_prob0 += 2.0 * (coefficient_product * (inner + inner_z)).re;
            }
        }
        (norm_sq, 0.5 * twice_prob0)
    }

    fn projection_coefficient_scale(ch: &CHFormGeneric<S, R>, q: usize) -> f64 {
        if ch.expectation_value_zq(q) == 0.0 {
            std::f64::consts::FRAC_1_SQRT_2
        } else {
            1.0
        }
    }

    /// Merge shared-structure terms with identical gamma and omega.
    ///
    /// The shared F/G/M/v/s precondition makes matching gamma and omega
    /// sufficient to prove identical amplitudes. Only worth calling when
    /// duplicates are likely (e.g., after measurement projection).
    fn merge_identical_terms(&mut self) {
        if self.terms.len() <= 4 || !self.has_shared_projection_structure() {
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
            global_phase: Angle64::ZERO,
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

    /// Materialize a pending RZ together with its current Clifford frame before
    /// applying a gate that does not commute with Z on this qubit.
    fn flush_noncommuting_pending_rz(&mut self, q: usize) {
        if self.pending_rz[q] != Angle64::ZERO {
            self.flush_cliff_frame(q);
            self.flush_pending_rz(q);
        }
    }

    /// Compose a named gate into the deferred Clifford frame, including the
    /// phase conversion between the gate and element-matrix conventions.
    fn compose_cliff_frame(&mut self, q: usize, gate: CliffordFrame) {
        let old = self.cliff_frame[q];
        self.frame_phase = (self.frame_phase
            + (8 - GATE_PHASE_DELTA[gate.index() as usize])
            + PHASE_COCYCLE[gate.index() as usize][old.index() as usize])
            & 7;
        self.cliff_frame[q] = gate.compose(old);
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
                    // The frame phase is relative to ELEMENT_MATRIX[2], while
                    // CH-form's named Y emits the standard Y matrix.
                    self.frame_phase = (self.frame_phase
                        + GATE_PHASE_DELTA[CliffordFrame::Y.index() as usize])
                        & 7;
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

        // A pending RZ can coexist only with a frame that preserves the Z axis.
        // Anti-diagonal frames carry the sign negation performed by X/Y
        // composition; restore it before materializing the frame to the right
        // of the RZ. The Pauli X/Y fast paths above do the same thing.
        if self.pending_rz[q] != Angle64::ZERO {
            let z_image = cf.z_image();
            debug_assert_eq!(
                z_image.axis,
                crate::clifford_frame::PauliAxis::Z,
                "a pending RZ requires a frame that preserves the Z axis"
            );
            if !z_image.positive {
                self.pending_rz[q] = -self.pending_rz[q];
            }
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
        if self.frame_phase != 0 || self.global_phase != Angle64::ZERO {
            use crate::clifford_frame::PHASE_ROOTS;
            let [re, im] = PHASE_ROOTS[(self.frame_phase & 7) as usize];
            let frame_phase = Complex64::new(re, im);
            let global_phase = Complex64::from_polar(1.0, self.global_phase.to_radians_signed());
            let phase = frame_phase * global_phase;
            for (coeff, _) in &mut self.terms {
                *coeff *= phase;
            }
            self.frame_phase = 0;
            self.global_phase = Angle64::ZERO;
        }
    }

    fn apply_clifford(&mut self, f: impl Fn(&mut CHFormGeneric<S, R>)) {
        for (_, ch) in &mut self.terms {
            f(ch);
        }
    }

    fn apply_c_type_checked(
        ch: &mut CHFormGeneric<S, R>,
        operation: &impl Fn(&mut CHFormGeneric<S, R>),
    ) {
        let f_before = ch.arc_f();
        let g_before = ch.arc_g();
        let v_before = ch.arc_v();
        let s_before = ch.arc_s();
        operation(ch);
        debug_assert!(
            std::sync::Arc::ptr_eq(&f_before, &ch.arc_f()),
            "C-type operation changed F"
        );
        debug_assert!(
            std::sync::Arc::ptr_eq(&g_before, &ch.arc_g()),
            "C-type operation changed G"
        );
        debug_assert!(
            std::sync::Arc::ptr_eq(&v_before, &ch.arc_v()),
            "C-type operation changed v"
        );
        debug_assert!(
            std::sync::Arc::ptr_eq(&s_before, &ch.arc_s()),
            "C-type operation changed s"
        );
    }

    /// Apply a C-type Clifford whose M and gamma transforms are identical for
    /// terms sharing G and M. F, G, v, and s must be left unchanged.
    fn apply_c_type_clifford(&mut self, operation: impl Fn(&mut CHFormGeneric<S, R>)) {
        if self.terms.len() <= 1 {
            for (_, ch) in &mut self.terms {
                Self::apply_c_type_checked(ch, &operation);
            }
            return;
        }

        // Pointer equality is a conservative proof that every term has the
        // same G/M inputs. Checking every term is essential: H can make later
        // terms structurally diverge while an earlier pair remains shared.
        let structurally_uniform = self
            .terms
            .iter()
            .enumerate()
            .all(|(index, (_, ch))| index == 0 || ch.shares_c_type_structure(&self.terms[0].1));
        if !structurally_uniform {
            for (_, ch) in &mut self.terms {
                Self::apply_c_type_checked(ch, &operation);
            }
            return;
        }

        let n = self.num_qubits;
        let gamma_before = self.terms[0].1.gamma().to_vec();
        Self::apply_c_type_checked(&mut self.terms[0].1, &operation);

        // C-type gates apply a term-independent additive gamma delta.
        let mut delta = vec![0u8; n];
        let gamma_after = self.terms[0].1.gamma();
        for p in 0..n {
            delta[p] = (gamma_after[p] + 4 - gamma_before[p]) & 3;
        }

        // Only M changes structurally. Preserve each term's F/G/v/s and
        // propagate the exact gamma transform.
        let shared_m = self.terms[0].1.arc_m();
        for (_, ch) in &mut self.terms[1..] {
            ch.apply_gamma_delta(&delta);
            ch.set_shared_m(shared_m.clone());
        }
    }

    /// Buffer an RZ gate. Fuses with any pending RZ on the same qubit.
    /// Uses Angle64 fixed-point addition for exact fusion of the rotation angle.
    /// The stored angle is only defined mod 2pi while RZ has period 4pi, so the
    /// scalar -1 lost when a signed sum wraps is tracked in `global_phase`
    /// (e.g. 8T = RZ(2pi) = -I).
    fn apply_rz(&mut self, theta: Angle64, q: usize) {
        const HALF: i128 = 1_i128 << 63;
        const FULL: i128 = 1_i128 << 64;

        let signed_fraction = |angle: Angle64| {
            let fraction = i128::from(angle.fraction());
            if fraction > HALF {
                fraction - FULL
            } else {
                fraction
            }
        };
        let previous = self.pending_rz[q];
        let combined = previous + theta;
        if signed_fraction(previous) + signed_fraction(theta) != signed_fraction(combined) {
            // Replacing a signed sum that crossed the principal-value boundary
            // by its stored representative changes RZ by a scalar -1.
            self.global_phase += Angle64::HALF_TURN;
        }
        self.pending_rz[q] = combined;
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

        // Non-Clifford angle: decompose into two terms. The shared helper uses
        // the signed fixed-point representative before taking the half-angle.
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
    /// For a single term, uses O(n) probability computation. Small systems use
    /// their state vector; larger shared-structure decompositions use optimized
    /// pairwise overlaps. Structurally divergent decompositions use the general
    /// pairwise CH-form overlap.
    fn measure_qubit(&mut self, q: usize, forced: Option<bool>) -> MeasurementResult {
        // Z-basis measurement on qubit q.
        // Frames and pending_rz on OTHER qubits commute with Z_q -- no flush needed.
        // Only qubit q's frame matters:
        // - Diagonal frame (Z→+Z): retain its selected branch's phase.
        // - Non-diagonal frame: must flush (changes measurement basis).
        // Pending_rz on q is diagonal: retain its selected branch's phase too.
        let cf_q = self.cliff_frame[q];
        let discarded_diagonal_frame = if cf_q.is_diagonal() {
            // A diagonal frame cannot flip the outcome, but its eigenvalue on
            // the selected basis state remains as a branch-global phase.
            self.cliff_frame[q] = CliffordFrame::IDENTITY;
            Some(cf_q)
        } else {
            // Non-diagonal: flush this qubit's frame (needs pending_rz flushed first).
            self.flush_cliff_frame(q);
            None
        };
        // A non-diagonal frame flush materialized its pending RZ. Otherwise,
        // defer the diagonal RZ eigenvalue until the outcome has been selected.
        let discarded_pending_rz = self.pending_rz[q];
        self.pending_rz[q] = Angle64::ZERO;
        if discarded_diagonal_frame.is_none() {
            debug_assert_eq!(discarded_pending_rz, Angle64::ZERO);
        }

        // Compute probability of measuring 0. Optimized multi-term paths are
        // valid only while all terms share the CH structure they precompute.
        // Exact branches carry the input norm they already compute; optimized
        // shared-structure branches use StabVec's normalized-state invariant.
        let structure_uniform = self.has_shared_projection_structure();
        let (state_norm_sq, prob0) = if self.terms.len() == 1 {
            // Single term: O(n) using CH-form structure directly
            let (coeff, ch) = &self.terms[0];
            let norm_sq = coeff.norm_sqr();
            (norm_sq, norm_sq * ch.prob_z_zero(q))
        } else if self.num_qubits <= 6 {
            // For small qubit counts, state vector is fast enough.
            let sv = self.state_vector();
            let mut norm_sq = 0.0;
            let mut p = 0.0;
            for (x, sv_x) in sv.iter().enumerate() {
                norm_sq += sv_x.norm_sqr();
                if (x >> q) & 1 == 0 {
                    p += sv_x.norm_sqr();
                }
            }
            (norm_sq, p)
        } else if !structure_uniform {
            self.exact_norm_and_prob0(q)
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
                (1.0, 0.5 * prob)
            } else {
                // Deterministic: all terms have the same Z_q expectation.
                (1.0, 0.5 * (1.0 + ez0))
            }
        } else {
            // Large T: first check if measurement is deterministic from structure.
            let ez = self.terms[0].1.expectation_value_zq(q);
            if ez != 0.0 {
                // Deterministic: all terms have the same Z_q expectation.
                (1.0, 0.5 * (1.0 + ez))
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
                (1.0, self.terms[chosen].1.prob_z_zero(q))
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
                (1.0, 0.5 * prob)
            } // end non-deterministic
        };

        // Determine the outcome.
        let outcome = if let Some(forced_val) = forced {
            forced_val
        } else if (prob0 - 1.0).abs() < 1e-10 {
            false // deterministic |0>
        } else if prob0 < 1e-10 {
            true // deterministic |1>
        } else {
            let r: f64 = self.rng.random();
            r >= prob0
        };

        let is_deterministic = (prob0 - 1.0).abs() < 1e-10 || prob0 < 1e-10;

        // Project: measure each CH-form term, keep only compatible terms.
        // After measurement, the state should be projected onto the outcome subspace.
        // For each term, force the measurement outcome and adjust coefficients.
        //
        // The simplest correct approach: reconstruct from the projected state vector.
        // But that loses the stabilizer structure.
        //
        // Better: measure each term independently with the forced outcome.
        // CH-form keeps a nondeterministic stabilizer post-state normalized,
        // so its corresponding coefficient carries the projector's 1/sqrt(2).

        // Apply the projector once only when every structural input is shared
        // and gamma[q] is uniform; otherwise project each term independently.
        // The diff set tracks every qubit whose gamma varies between terms.
        let gamma_q_uniform = self.terms.len() <= 1 || !self.gamma_diff_qubits.contains(&q);
        let structure_uniform = self.has_shared_projection_structure();
        if gamma_q_uniform && structure_uniform && self.terms.len() > 1 {
            debug_assert!(
                self.terms.iter().all(|(_, ch)| !ch.omega_exact().is_zero()),
                "zero-omega terms must be removed after projection"
            );
            // All terms have the same gamma[q], so delta is identical.
            // Structural changes and omega transform are the same for all terms.
            // Apply the projector once, compute deltas, propagate to others.
            let projection_scale = Self::projection_coefficient_scale(&self.terms[0].1, q);
            let gamma_before = self.terms[0].1.gamma().to_vec();
            let omega_before = self.terms[0].1.omega_exact();
            self.terms[0].1.project_z(q, outcome);
            self.terms[0].0 *= projection_scale;
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
            for (coefficient, ch) in &mut self.terms[1..] {
                ch.apply_gamma_delta(&gamma_delta);
                ch.apply_omega_transform(omega_before, omega_after);
                *coefficient *= projection_scale;
                ch.set_arcs(
                    shared_f.clone(),
                    shared_g.clone(),
                    shared_m.clone(),
                    shared_v.clone(),
                    shared_s.clone(),
                );
            }
        } else {
            for (coefficient, ch) in &mut self.terms {
                let projection_scale = Self::projection_coefficient_scale(ch, q);
                ch.project_z(q, outcome);
                *coefficient *= projection_scale;
            }
        }

        // Incompatible stabilizer terms project to the zero state. Remove them
        // before merging and normalization so they cannot act as structural
        // representatives or contribute their coefficients to the norm.
        self.terms.retain(|(_, ch)| !ch.omega_exact().is_zero());
        if self.terms.is_empty() {
            let ch = CHFormGeneric::with_rng(self.num_qubits, self.rng.clone());
            self.terms.push((Complex64::new(0.0, 0.0), ch));
        }
        self.recompute_gamma_diff();

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

        // P0 and P1 are complementary orthogonal projectors, so the squared
        // norm after projection is the probability weight already computed.
        let projected_norm_sq = if outcome {
            state_norm_sq - prob0
        } else {
            prob0
        };
        if projected_norm_sq > 0.0 {
            let inv_norm = 1.0 / projected_norm_sq.sqrt();
            for (coeff, _) in &mut self.terms {
                *coeff *= inv_norm;
            }
        }

        // RZ(theta) = diag(exp(-i*theta/2), exp(i*theta/2)). The discarded
        // diagonal Clifford likewise acts on the surviving basis state by a
        // scalar eigenvalue. Preserve both through the simulator's global-phase hook.
        let rz_branch_phase = if outcome {
            discarded_pending_rz.signed_half()
        } else {
            -discarded_pending_rz.signed_half()
        };
        let frame_branch_phase = discarded_diagonal_frame.map_or(Angle64::ZERO, |frame| {
            frame
                .computational_basis_phase(outcome)
                .expect("a diagonal frame must have a basis phase")
        });
        self.apply_global_phase(rz_branch_phase + frame_branch_phase, &[QubitId(q)]);

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
        self.cliff_frame.fill(CliffordFrame::IDENTITY);
        self.frame_phase = 0;
        self.global_phase = Angle64::ZERO;
        self.gamma_diff_qubits.clear();
        // rel_pruning_threshold preserved across reset
        self
    }
}

impl<S: IndexSet, R: SeedableRng + Rng + Debug + Clone> CliffordGateable for StabVecGeneric<S, R> {
    fn apply_global_phase(&mut self, phase: Angle64, qubits: &[QubitId]) -> &mut Self {
        for _ in qubits {
            self.global_phase += phase;
        }
        self
    }

    // === Single-qubit Cliffords: all compose into the frame in O(1) ===
    // Diagonal gates (Z, S, Sdg) commute with pending_rz.
    // Non-diagonal gates (H, X, Y, SX, etc.) negate pending_rz if they
    // anticommute with Z, or flush pending_rz if they don't simply negate.

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qi = q.index();
            self.pending_rz[qi] = -self.pending_rz[qi]; // X anticommutes with RZ
            self.compose_cliff_frame(qi, CliffordFrame::X);
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qi = q.index();
            self.pending_rz[qi] = -self.pending_rz[qi]; // Y anticommutes with RZ
            self.compose_cliff_frame(qi, CliffordFrame::Y);
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qi = q.index();
            // Z commutes with RZ, no negation needed.
            self.compose_cliff_frame(qi, CliffordFrame::Z);
        }
        self
    }

    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qi = q.index();
            // S is diagonal, commutes with RZ.
            self.compose_cliff_frame(qi, CliffordFrame::SZ);
        }
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let qi = q.index();
            self.compose_cliff_frame(qi, CliffordFrame::SZDG);
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
                self.compose_cliff_frame(qi, CliffordFrame::H);
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

    fn h3(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.flush_noncommuting_pending_rz(q.index());
        }
        self.sz(qubits).y(qubits);
        // H3 = exp(-i*pi/4) * Y * SZ.
        for _ in qubits {
            self.frame_phase = (self.frame_phase + 7) & 7;
        }
        self
    }

    fn h4(&mut self, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            self.flush_noncommuting_pending_rz(q.index());
        }
        self.sz(qubits).x(qubits);
        // Correct the projective frame composition to canonical H4 phase.
        for _ in qubits {
            self.frame_phase = (self.frame_phase + 7) & 7;
        }
        self
    }

    fn h6(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.sx(qubits)
            .y(qubits)
            .apply_global_phase(-(Angle64::QUARTER_TURN / 2u64), qubits);
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
        self.apply_c_type_clifford(|ch| {
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
        self.apply_c_type_clifford(|ch| {
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
        self.apply_c_type_clifford(|ch| {
            ch.szzdg(pairs);
        });
        self
    }

    fn sxx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        for &(q0, q1) in pairs {
            let q = q0.index();
            let r = q1.index();
            for target in [q, r] {
                self.flush_noncommuting_pending_rz(target);
            }
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
        self.apply_c_type_clifford(|ch| {
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
            for target in [q, r] {
                self.flush_noncommuting_pending_rz(target);
            }
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
        self.apply_c_type_clifford(|ch| {
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
        self.apply_c_type_clifford(|ch| {
            ch.sz(&all_qubits);
        });
        self.sxx(pairs);
        self.apply_c_type_clifford(|ch| {
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
        self.apply_c_type_clifford(|ch| {
            ch.sz(&all_qubits);
        });
        self.sxxdg(pairs);
        self.apply_c_type_clifford(|ch| {
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
        // Measure -Z via the trait's reference decomposition (X; MZ; X). The
        // old shortcut composed a Z frame, which commutes with a Z readout
        // and could never flip the outcome.
        self.x(qubits);
        let results = self.mz(qubits);
        self.x(qubits);
        results
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
            // is_diagonal() guarantees Z→+Z, so a retained frame cannot negate theta.
            self.apply_rz(theta, qi);
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
    use crate::{StateVec, StateVecSoA};
    use pecos_core::gate_type::{GateType, NAMED_TWO_QUBIT_ROOT_GATES};
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
    fn test_inherited_defaults_preserve_many_terms_and_exact_state() {
        // h2 and h4 are inherited defaults that deliver their residual phase
        // through apply_global_phase. h2's residues cancel to zero; h4 carries
        // a net -pi/4, so a dropped accumulator shows up on the state.
        let mut sim = StabVec::new(1);
        let term = sim.terms[0].1.clone();
        let coefficient = Complex64::new(1.0 / 4094.0, 0.0);
        sim.terms = (0..4094).map(|_| (coefficient, term.clone())).collect();
        let mut expected = StateVec::new(1);

        sim.h2(&[QubitId(0)]).h4(&[QubitId(0)]);
        expected.h2(&[QubitId(0)]).h4(&[QubitId(0)]);
        assert_eq!(sim.num_terms(), 4094);
        for (actual, expected) in sim.state_vector().iter().zip(expected.state()) {
            assert!((actual - expected).norm() < 1e-12);
        }
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

        // Exercise the T-equivalent RZ(pi/4), which differs by a global phase.
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
    fn stab_vec_reset_clears_frame_state() {
        let mut reset_sim = StabVec::new_with_seed(2, 17);
        let mut fresh_sim = StabVec::new_with_seed(2, 17);
        let q0 = qid(0);
        let q1 = qid(1);

        // Leave both non-zero rotations pending while accumulating X/Y frames.
        // Y's standard-gate phase leaves a non-trivial global frame phase.
        reset_sim
            .rz(Angle64::from_radians(0.37), &q0)
            .x(&q0)
            .rz(Angle64::from_radians(-0.91), &q1)
            .y(&q1);
        assert_eq!(reset_sim.cliff_frame[0], CliffordFrame::X);
        assert_eq!(reset_sim.cliff_frame[1], CliffordFrame::Y);
        assert_ne!(reset_sim.pending_rz[0], Angle64::ZERO);
        assert_ne!(reset_sim.pending_rz[1], Angle64::ZERO);
        assert_ne!(reset_sim.frame_phase, 0);

        reset_sim.reset();
        reset_sim.h(&[QubitId(0), QubitId(1)]);
        fresh_sim.h(&[QubitId(0), QubitId(1)]);

        assert_eq!(reset_sim.state_vector(), fresh_sim.state_vector());
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
    fn t_only_flushes_pending_rotation_on_its_target() {
        let epsilon = Angle64::from_radians(1e-5);
        let q0 = qid(0);
        let q1 = qid(1);

        let mut interleaved = StabVec::new(2);
        interleaved.h(&q0).h(&q1).rz(epsilon, &q1);
        assert_eq!(interleaved.pending_rz[1], epsilon);
        interleaved.t(&q0);
        assert_eq!(
            interleaved.pending_rz[1], epsilon,
            "T on q0 must not materialize and prune q1's pending rotation"
        );
        interleaved.rz(-epsilon, &q1);
        assert_eq!(interleaved.pending_rz[1], Angle64::ZERO);

        let mut reference = StabVec::new(2);
        reference.h(&q0).h(&q1).t(&q0);

        let actual = interleaved.state_vector();
        let expected = reference.state_vector();
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).norm() < 1e-10,
                "basis {index}: actual={actual}, expected={expected}"
            );
        }
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
        assert_eq!(results.len(), 1);
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
        assert_eq!(results.len(), 1);
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

    fn apply_two_qubit_root<S: CliffordGateable>(sim: &mut S, gate: GateType) {
        let pair = [(QubitId(0), QubitId(1))];
        match gate {
            GateType::SXX => {
                sim.sxx(&pair);
            }
            GateType::SXXdg => {
                sim.sxxdg(&pair);
            }
            GateType::SYY => {
                sim.syy(&pair);
            }
            GateType::SYYdg => {
                sim.syydg(&pair);
            }
            GateType::SZZ => {
                sim.szz(&pair);
            }
            GateType::SZZdg => {
                sim.szzdg(&pair);
            }
            other => panic!("unsupported two-qubit root gate {other:?}"),
        }
    }

    fn prepare_nonuniform_terms(
        stab_vec: &mut StabVec,
        state_vec: &mut StateVecSoA,
        term_count: usize,
    ) {
        let q0 = qid(0);
        stab_vec.h(&q0);
        state_vec.h(&q0);
        for step in 0..term_count.ilog2() {
            let angle = Angle64::from_radians(0.37 + 0.11 * f64::from(step));
            stab_vec.rz(angle, &q0).h(&q0);
            state_vec.rz(angle, &q0).h(&q0);
        }
        assert_eq!(stab_vec.num_terms(), term_count);
    }

    fn prepare_uniform_terms(
        stab_vec: &mut StabVec,
        state_vec: &mut StateVecSoA,
        term_count: usize,
    ) {
        let q0 = qid(0);
        stab_vec.h(&q0);
        state_vec.h(&q0);
        stab_vec.flush_all_cliff_frames();
        let _ = state_vec.state();
        for step in 0..term_count.ilog2() {
            let angle = Angle64::from_radians(0.37 + 0.11 * f64::from(step));
            stab_vec.rz(angle, &q0);
            stab_vec.flush_all_pending_rz();
            state_vec.rz(angle, &q0);
        }
        assert_eq!(stab_vec.num_terms(), term_count);
    }

    fn assert_phase_exact_state_matches(actual: &[Complex64], expected: &[Complex64], label: &str) {
        let actual_norm: f64 = actual.iter().map(Complex64::norm_sqr).sum();
        let expected_norm: f64 = expected.iter().map(Complex64::norm_sqr).sum();
        assert!(
            (actual_norm - 1.0).abs() < EPS,
            "{label}: StabVec norm is {actual_norm:.12}, expected 1"
        );
        assert!(
            (actual_norm - expected_norm).abs() < EPS,
            "{label}: norm mismatch: StabVec={actual_norm:.12}, StateVecSoA={expected_norm:.12}"
        );
        for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (actual - expected).norm() < EPS,
                "{label}: amplitude[{index}] mismatch: StabVec={actual:.12}, \
                 StateVecSoA={expected:.12}"
            );
        }
    }

    fn normalized_z_projection(
        input: &[Complex64],
        measured_qubit: usize,
        outcome: bool,
        label: &str,
    ) -> Vec<Complex64> {
        let mut projected = input.to_vec();
        for (basis, amplitude) in projected.iter_mut().enumerate() {
            if (((basis >> measured_qubit) & 1) != 0) != outcome {
                *amplitude = Complex64::new(0.0, 0.0);
            }
        }
        let norm = projected
            .iter()
            .map(Complex64::norm_sqr)
            .sum::<f64>()
            .sqrt();
        assert!(
            norm > EPS,
            "{label}: outcome {outcome} has zero probability"
        );
        for amplitude in &mut projected {
            *amplitude /= norm;
        }
        projected
    }

    fn apply_deferred_test_gate<S: CliffordGateable>(
        simulator: &mut S,
        gate: &str,
        qubits: &[QubitId],
    ) {
        match gate {
            "X" => {
                simulator.x(qubits);
            }
            "Y" => {
                simulator.y(qubits);
            }
            _ => unreachable!("test gate must be X or Y"),
        }
    }

    #[test]
    fn x_and_y_preserve_phase_on_every_clifford_input_frame() {
        let q0 = qid(0);
        for input_frame in 0..24 {
            for pauli in ["X", "Y"] {
                let mut stab_vec = StabVec::builder(1).seed(0x715).build();
                let mut state_vec = StateVecSoA::with_seed(1, 0x715);
                stab_vec
                    .ry(Angle64::from_radians(0.731), &q0)
                    .rz(Angle64::from_radians(-0.417), &q0);
                state_vec
                    .ry(Angle64::from_radians(0.731), &q0)
                    .rz(Angle64::from_radians(-0.417), &q0);
                let _ = stab_vec.state_vector();
                let _ = state_vec.state();

                for &generator in &GENERATORS[input_frame][..GEN_LENS[input_frame] as usize] {
                    match generator {
                        0 => {
                            stab_vec.h(&q0);
                            state_vec.h(&q0);
                        }
                        1 => {
                            stab_vec.sz(&q0);
                            state_vec.sz(&q0);
                        }
                        _ => unreachable!("generator sequence contains only H and S"),
                    }
                }
                assert_eq!(usize::from(stab_vec.cliff_frame[0].index()), input_frame);
                apply_deferred_test_gate(&mut stab_vec, pauli, &q0);
                apply_deferred_test_gate(&mut state_vec, pauli, &q0);
                assert_phase_exact_state_matches(
                    &stab_vec.state_vector(),
                    &state_vec.state(),
                    &format!("frame {input_frame}; {pauli} on a generic state"),
                );
            }
        }

        for input_frame in [0, 3, 4, 5] {
            for pauli in ["X", "Y"] {
                let mut stab_vec = StabVec::builder(1).seed(0x715).build();
                let mut state_vec = StateVecSoA::with_seed(1, 0x715);
                for &generator in &GENERATORS[input_frame][..GEN_LENS[input_frame] as usize] {
                    match generator {
                        0 => {
                            stab_vec.h(&q0);
                            state_vec.h(&q0);
                        }
                        1 => {
                            stab_vec.sz(&q0);
                            state_vec.sz(&q0);
                        }
                        _ => unreachable!("generator sequence contains only H and S"),
                    }
                }
                let angle = Angle64::from_radians(0.37);
                stab_vec.rz(angle, &q0);
                state_vec.rz(angle, &q0);
                assert_eq!(stab_vec.pending_rz[0], angle);

                apply_deferred_test_gate(&mut stab_vec, pauli, &q0);
                apply_deferred_test_gate(&mut state_vec, pauli, &q0);
                assert_phase_exact_state_matches(
                    &stab_vec.state_vector(),
                    &state_vec.state(),
                    &format!("diagonal frame {input_frame}; RZ; {pauli}"),
                );
            }
        }
    }

    #[test]
    fn pauli_gates_preserve_phase_when_flushing_an_existing_frame() {
        let q0 = qid(0);

        let mut stab_x = StabVec::builder(1).seed(0x715).build();
        let mut state_x = StateVecSoA::with_seed(1, 0x715);
        stab_x.z(&q0).x(&q0);
        state_x.z(&q0).x(&q0);
        assert_phase_exact_state_matches(&stab_x.state_vector(), &state_x.state(), "Z; X");

        let angle = Angle64::from_radians(0.37);
        let mut stab_rz_x = StabVec::builder(1).seed(0x715).build();
        let mut state_rz_x = StateVecSoA::with_seed(1, 0x715);
        stab_rz_x.sz(&q0).rz(angle, &q0).x(&q0);
        state_rz_x.sz(&q0).rz(angle, &q0).x(&q0);
        assert_phase_exact_state_matches(
            &stab_rz_x.state_vector(),
            &state_rz_x.state(),
            "S; RZ(0.37); X",
        );

        let mut stab_y = StabVec::builder(1).seed(0x715).build();
        let mut state_y = StateVecSoA::with_seed(1, 0x715);
        stab_y.z(&q0).y(&q0);
        state_y.z(&q0).y(&q0);
        assert_phase_exact_state_matches(&stab_y.state_vector(), &state_y.state(), "Z; Y");

        let mut stab_rz_y = StabVec::builder(1).seed(0x715).build();
        let mut state_rz_y = StateVecSoA::with_seed(1, 0x715);
        stab_rz_y.sz(&q0).rz(angle, &q0).y(&q0);
        state_rz_y.sz(&q0).rz(angle, &q0).y(&q0);
        assert_phase_exact_state_matches(
            &stab_rz_y.state_vector(),
            &state_rz_y.state(),
            "S; RZ(0.37); Y",
        );
    }

    #[test]
    fn negative_z_measurements_preserve_phase_when_flushing_an_existing_frame() {
        let q0 = qid(0);
        let mut stab_mnz = StabVec::builder(1).seed(0x715).build();
        let mut state_mnz = StateVecSoA::with_seed(1, 0x715);
        stab_mnz.z(&q0);
        state_mnz.z(&q0);
        let stab_result = stab_mnz.mnz(&q0);
        let state_result = state_mnz.mnz(&q0);
        assert_eq!(stab_result[0].outcome, state_result[0].outcome);
        assert_eq!(
            stab_result[0].is_deterministic,
            state_result[0].is_deterministic
        );
        assert_phase_exact_state_matches(&stab_mnz.state_vector(), &state_mnz.state(), "Z; MNZ");

        let mut stab_mpnz = StabVec::builder(1).seed(0x715).build();
        let mut state_mpnz = StateVecSoA::with_seed(1, 0x715);
        stab_mpnz.z(&q0);
        state_mpnz.z(&q0);
        let stab_result = stab_mpnz.mpnz(&q0);
        let state_result = state_mpnz.mpnz(&q0);
        assert_eq!(stab_result[0].outcome, state_result[0].outcome);
        assert_eq!(
            stab_result[0].is_deterministic,
            state_result[0].is_deterministic
        );
        assert_phase_exact_state_matches(&stab_mpnz.state_vector(), &state_mpnz.state(), "Z; MPNZ");
    }

    #[test]
    fn measurement_preserves_pending_rz_branch_phase() {
        let q0 = qid(0);
        let ordinary = Angle64::from_radians(0.37);
        for (label, angles, expected_pending, expected_global_phase) in [
            (
                "ordinary pending RZ",
                vec![ordinary],
                ordinary,
                Angle64::ZERO,
            ),
            (
                "pending RZ above a half turn",
                vec![Angle64::THREE_QUARTERS_TURN],
                Angle64::THREE_QUARTERS_TURN,
                Angle64::ZERO,
            ),
            (
                "pending RZ at pi",
                vec![Angle64::HALF_TURN],
                Angle64::HALF_TURN,
                Angle64::ZERO,
            ),
            (
                "pending RZ accumulated across the 2pi wrap",
                vec![Angle64::THREE_QUARTERS_TURN, Angle64::THREE_QUARTERS_TURN],
                Angle64::HALF_TURN,
                Angle64::HALF_TURN,
            ),
        ] {
            let mut stab_vec = StabVec::builder(1).seed(0x714).build();
            let mut state_vec = StateVecSoA::with_seed(1, 0x714);

            stab_vec.h(&q0);
            state_vec.h(&q0);
            for angle in angles {
                stab_vec.rz(angle, &q0);
                state_vec.rz(angle, &q0);
            }
            assert_eq!(stab_vec.cliff_frame[0], CliffordFrame::IDENTITY);
            assert_eq!(stab_vec.pending_rz[0], expected_pending);
            assert_eq!(stab_vec.global_phase, expected_global_phase);

            let input = state_vec.state();
            for outcome in [false, true] {
                let expected = normalized_z_projection(&input, 0, outcome, label);
                let mut measured = stab_vec.clone();
                let result = measured.measure_qubit(0, Some(outcome));
                assert_eq!(result.outcome, outcome);
                assert_phase_exact_state_matches(
                    &measured.state_vector(),
                    &expected,
                    &format!("{label}, outcome {outcome}"),
                );
            }
        }
    }

    #[test]
    fn measurement_preserves_every_diagonal_clifford_branch_phase() {
        let q0 = qid(0);
        for (name, frame) in [
            ("I", CliffordFrame::IDENTITY),
            ("Z", CliffordFrame::Z),
            ("S", CliffordFrame::SZ),
            ("Sdg", CliffordFrame::SZDG),
        ] {
            let mut stab_vec = StabVec::builder(1).seed(0x714).build();
            let mut state_vec = StateVecSoA::with_seed(1, 0x714);

            stab_vec.h(&q0);
            stab_vec.flush_all_cliff_frames();
            state_vec.h(&q0);
            match frame.index() {
                0 => {}
                3 => {
                    stab_vec.z(&q0);
                    state_vec.z(&q0);
                }
                4 => {
                    stab_vec.sz(&q0);
                    state_vec.sz(&q0);
                }
                5 => {
                    stab_vec.szdg(&q0);
                    state_vec.szdg(&q0);
                }
                _ => unreachable!("test table contains only diagonal frames"),
            }
            assert_eq!(stab_vec.cliff_frame[0], frame);
            assert_eq!(stab_vec.pending_rz[0], Angle64::ZERO);

            let label = format!("diagonal {name} frame");
            let input = state_vec.state();
            for outcome in [false, true] {
                let expected = normalized_z_projection(&input, 0, outcome, &label);
                let mut measured = stab_vec.clone();
                let result = measured.measure_qubit(0, Some(outcome));
                assert_eq!(result.outcome, outcome);
                assert_phase_exact_state_matches(
                    &measured.state_vector(),
                    &expected,
                    &format!("{label}, outcome {outcome}"),
                );
            }
        }
    }

    #[test]
    fn diagonal_phase_emission_does_not_change_seeded_outcome_streams() {
        let q0 = qid(0);
        let angle = Angle64::from_radians(0.37);
        let mut with_phase = Vec::with_capacity(16_750);
        let mut without_phase = Vec::with_capacity(16_750);

        for seed in 0..67 {
            let mut phase_sim = StabVec::builder(1).seed(seed).build();
            let mut reference_sim = StabVec::builder(1).seed(seed).build();
            for _ in 0..250 {
                phase_sim.h(&q0);
                phase_sim.flush_all_cliff_frames();
                phase_sim.sz(&q0).rz(angle, &q0);
                with_phase.push(u8::from(phase_sim.mz(&q0)[0].outcome));

                reference_sim.h(&q0);
                without_phase.push(u8::from(reference_sim.mz(&q0)[0].outcome));

                phase_sim.reset();
                reference_sim.reset();
            }
        }

        assert_eq!(with_phase.len(), 16_750);
        assert_eq!(with_phase, without_phase);
    }

    #[test]
    fn measurement_preserves_combined_diagonal_frame_and_rz_branch_phase() {
        let q0 = qid(0);
        let angle = Angle64::from_radians(-0.43);
        let mut stab_vec = StabVec::builder(1).seed(0x714).build();
        let mut state_vec = StateVecSoA::with_seed(1, 0x714);

        stab_vec.h(&q0);
        stab_vec.flush_all_cliff_frames();
        stab_vec.sz(&q0).rz(angle, &q0);
        state_vec.h(&q0).sz(&q0).rz(angle, &q0);
        assert_eq!(stab_vec.cliff_frame[0], CliffordFrame::SZ);
        assert_eq!(stab_vec.pending_rz[0], angle);

        let label = "diagonal S frame with pending negative RZ";
        let input = state_vec.state();
        for outcome in [false, true] {
            let expected = normalized_z_projection(&input, 0, outcome, label);
            let mut measured = stab_vec.clone();
            let result = measured.measure_qubit(0, Some(outcome));
            assert_eq!(result.outcome, outcome);
            assert_phase_exact_state_matches(
                &measured.state_vector(),
                &expected,
                &format!("{label}, outcome {outcome}"),
            );
        }
    }

    #[test]
    fn entangled_measurement_preserves_branch_phase_with_other_amplitudes() {
        let q0 = qid(0);
        let q1 = qid(1);
        let measured_angle = Angle64::from_radians(0.37);
        let other_angle = Angle64::from_radians(0.61);
        let mut stab_vec = StabVec::builder(3).seed(0x714).build();
        let mut state_vec = StateVecSoA::with_seed(3, 0x714);

        stab_vec
            .h(&q0)
            .h(&q1)
            .cx(&[(QubitId(0), QubitId(2))])
            .rz(other_angle, &q1)
            .sz(&q0)
            .rz(measured_angle, &q0);
        state_vec
            .h(&q0)
            .h(&q1)
            .cx(&[(QubitId(0), QubitId(2))])
            .rz(other_angle, &q1)
            .sz(&q0)
            .rz(measured_angle, &q0);
        assert_eq!(stab_vec.cliff_frame[0], CliffordFrame::SZ);
        assert_eq!(stab_vec.pending_rz[0], measured_angle);
        assert_eq!(stab_vec.pending_rz[1], other_angle);

        let input = state_vec.state();
        for outcome in [false, true] {
            let expected = normalized_z_projection(&input, 0, outcome, "entangled measured qubit");
            let mut measured = stab_vec.clone();
            let result = measured.measure_qubit(0, Some(outcome));
            assert_eq!(result.outcome, outcome);
            assert_phase_exact_state_matches(
                &measured.state_vector(),
                &expected,
                &format!("entangled measured qubit, outcome {outcome}"),
            );
        }
    }

    #[test]
    fn z_preparations_preserve_both_sampled_branches_and_reference_trajectories() {
        let q0 = qid(0);
        let q1 = qid(1);
        let angle = Angle64::from_radians(0.37);
        let preparation_angle = Angle64::from_radians(0.8);
        let mut workaround_complement_count = 0;

        for prepare_negative_z in [false, true] {
            let name = if prepare_negative_z { "PNZ" } else { "PZ" };
            let mut saw_partner_branch = [false; 2];

            for seed in 0..40 {
                let mut stab_vec = StabVec::builder(2).seed(seed).build();
                let mut state_vec = StateVecSoA::with_seed(2, seed);
                stab_vec
                    .ry(preparation_angle, &q0)
                    .cx(&[(QubitId(0), QubitId(1))])
                    .sz(&q0)
                    .rz(angle, &q0);
                state_vec
                    .ry(preparation_angle, &q0)
                    .cx(&[(QubitId(0), QubitId(1))])
                    .sz(&q0)
                    .rz(angle, &q0);

                let mut old_pnz_workaround = prepare_negative_z.then(|| stab_vec.clone());
                let mut explicit_reference = stab_vec.clone();
                if prepare_negative_z {
                    stab_vec.pnz(&q0);
                    explicit_reference.mpnz(&q0);
                } else {
                    stab_vec.pz(&q0);
                    explicit_reference.mpz(&q0);
                }

                let actual = stab_vec.state_vector();
                let partner_is_one = actual
                    .iter()
                    .enumerate()
                    .filter(|(basis, _)| basis & (1 << q1[0].index()) != 0)
                    .map(|(_, amplitude)| amplitude.norm_sqr())
                    .sum::<f64>()
                    > 0.5;
                saw_partner_branch[usize::from(partner_is_one)] = true;

                let label = format!("{name} sampled trajectory for seed {seed}");
                let mut expected =
                    normalized_z_projection(&state_vec.state(), 0, partner_is_one, &label);
                if partner_is_one != prepare_negative_z {
                    for amplitudes in expected.as_chunks_mut::<2>().0 {
                        amplitudes.swap(0, 1);
                    }
                }
                assert_phase_exact_state_matches(&actual, &expected, &label);
                assert_phase_exact_state_matches(
                    &actual,
                    &explicit_reference.state_vector(),
                    &format!("{name} reference decomposition for seed {seed}"),
                );

                if let Some(workaround) = &mut old_pnz_workaround {
                    workaround.mpz(&q0);
                    workaround.x(&q0);
                    let workaround_state = workaround.state_vector();
                    let workaround_partner_is_one = workaround_state
                        .iter()
                        .enumerate()
                        .filter(|(basis, _)| basis & (1 << q1[0].index()) != 0)
                        .map(|(_, amplitude)| amplitude.norm_sqr())
                        .sum::<f64>()
                        > 0.5;
                    workaround_complement_count +=
                        usize::from(workaround_partner_is_one != partner_is_one);
                }
            }

            assert!(
                saw_partner_branch.into_iter().all(std::convert::identity),
                "{name} did not cover both nondeterministic input branches"
            );
        }

        assert_eq!(
            workaround_complement_count, 9,
            "the removed MPZ; X workaround no longer witnesses the reviewed trajectory divergence"
        );
    }

    #[test]
    fn two_qubit_roots_match_state_vec_soa_across_term_counts() {
        for gate in NAMED_TWO_QUBIT_ROOT_GATES {
            for term_count in [2, 4, 8, 16] {
                for (structure, prepare) in [
                    (
                        "shared structure",
                        prepare_uniform_terms as fn(&mut StabVec, &mut StateVecSoA, usize),
                    ),
                    ("divergent structure", prepare_nonuniform_terms),
                ] {
                    let mut stab_vec = StabVec::builder(2).pruning_threshold(0.0).build();
                    let mut state_vec = StateVecSoA::new(2);
                    prepare(&mut stab_vec, &mut state_vec, term_count);
                    assert_phase_exact_state_matches(
                        &stab_vec.state_vector(),
                        &state_vec.state(),
                        &format!("input with {term_count} terms and {structure}"),
                    );

                    apply_two_qubit_root(&mut stab_vec, gate);
                    apply_two_qubit_root(&mut state_vec, gate);

                    let actual = stab_vec.state_vector();
                    let expected = state_vec.state();
                    assert_phase_exact_state_matches(
                        &actual,
                        &expected,
                        &format!("{gate:?} with {term_count} terms and {structure}"),
                    );
                }
            }
        }
    }

    #[test]
    fn two_qubit_roots_match_state_vec_soa_with_pending_rz_across_term_counts() {
        for gate in NAMED_TWO_QUBIT_ROOT_GATES {
            for term_count in [2, 4, 8, 16] {
                let mut stab_vec = StabVec::builder(2).pruning_threshold(0.0).build();
                let mut state_vec = StateVecSoA::new(2);
                prepare_nonuniform_terms(&mut stab_vec, &mut state_vec, term_count / 2);
                let q0 = qid(0);
                let angle = Angle64::from_radians(0.37);
                stab_vec.rz(angle, &q0);
                state_vec.rz(angle, &q0);

                apply_two_qubit_root(&mut stab_vec, gate);
                apply_two_qubit_root(&mut state_vec, gate);

                let actual = stab_vec.state_vector();
                let expected = state_vec.state();
                assert_eq!(stab_vec.num_terms(), term_count);
                assert_phase_exact_state_matches(
                    &actual,
                    &expected,
                    &format!("{gate:?} with pending RZ and {term_count} terms"),
                );
            }
        }
    }

    #[test]
    fn c_type_fast_path_checks_every_term_structure() {
        let mut stab_vec = StabVec::builder(2).pruning_threshold(0.0).build();
        let mut discarded_reference = StateVecSoA::new(2);
        prepare_uniform_terms(&mut stab_vec, &mut discarded_reference, 4);

        // Keep terms 0 and 1 shared while making only a later term's M distinct.
        stab_vec.terms[2].1.sz(&qid(0));
        assert!(
            stab_vec.terms[0]
                .1
                .shares_c_type_structure(&stab_vec.terms[1].1)
        );
        assert!(
            !stab_vec.terms[0]
                .1
                .shares_c_type_structure(&stab_vec.terms[2].1)
        );

        let norm = stab_vec
            .state_vector()
            .iter()
            .map(Complex64::norm_sqr)
            .sum::<f64>()
            .sqrt();
        for (coefficient, _) in &mut stab_vec.terms {
            *coefficient /= norm;
        }
        let input = stab_vec.state_vector();
        let mut state_vec = StateVecSoA::from_state(&input, PecosRng::seed_from_u64(42));

        stab_vec.szz(&[(QubitId(0), QubitId(1))]);
        state_vec.szz(&[(QubitId(0), QubitId(1))]);
        assert_phase_exact_state_matches(
            &stab_vec.state_vector(),
            &state_vec.state(),
            "SZZ with divergence after the first two terms",
        );
    }

    #[test]
    fn inherited_h3_h4_match_state_vec_soa_with_decomposed_terms_and_pending_rz() {
        for gate in ["H3", "H4"] {
            for term_count in [2, 4, 8, 16] {
                let mut stab_vec = StabVec::builder(2).pruning_threshold(0.0).build();
                let mut state_vec = StateVecSoA::new(2);
                prepare_nonuniform_terms(&mut stab_vec, &mut state_vec, term_count);
                match gate {
                    "H3" => {
                        stab_vec.h3(&qid(0));
                        state_vec.h3(&qid(0));
                    }
                    "H4" => {
                        stab_vec.h4(&qid(0));
                        state_vec.h4(&qid(0));
                    }
                    _ => unreachable!(),
                }
                assert_phase_exact_state_matches(
                    &stab_vec.state_vector(),
                    &state_vec.state(),
                    &format!("{gate} with {term_count} terms"),
                );
            }

            let mut stab_vec = StabVec::builder(2).pruning_threshold(0.0).build();
            let mut state_vec = StateVecSoA::new(2);
            let angle = Angle64::from_radians(0.37);
            stab_vec.h(&qid(0)).rz(angle, &qid(0));
            state_vec.h(&qid(0)).rz(angle, &qid(0));
            match gate {
                "H3" => {
                    stab_vec.h3(&qid(0));
                    state_vec.h3(&qid(0));
                }
                "H4" => {
                    stab_vec.h4(&qid(0));
                    state_vec.h4(&qid(0));
                }
                _ => unreachable!(),
            }
            assert_phase_exact_state_matches(
                &stab_vec.state_vector(),
                &state_vec.state(),
                &format!("{gate} with pending RZ"),
            );
        }
    }

    #[test]
    fn measurement_projection_removes_zero_omega_terms_before_uniform_gamma_path() {
        let mut stab_vec = StabVec::builder(3).pruning_threshold(0.0).build();
        let mut state_vec = StateVecSoA::new(3);
        // Each H RZ H decomposes its qubit into terms with opposite
        // deterministic Z support. Projecting q0 onto |1> makes term 0 a
        // zero-omega term while two compatible q1 terms remain live.
        for (qubit, angle) in [(QubitId(0), 0.37), (QubitId(1), 0.53)] {
            let angle = Angle64::from_radians(angle);
            stab_vec.h(&[qubit]).rz(angle, &[qubit]).h(&[qubit]);
            state_vec.h(&[qubit]).rz(angle, &[qubit]).h(&[qubit]);
        }
        assert_eq!(stab_vec.num_terms(), 4);
        assert!((stab_vec.terms[0].1.prob_z_zero(0) - 1.0).abs() < EPS);
        stab_vec.measure_qubit(0, Some(true));

        let mut expected = state_vec.state();
        for (index, amplitude) in expected.iter_mut().enumerate() {
            if index & 1 == 0 {
                *amplitude = Complex64::new(0.0, 0.0);
            }
        }
        let norm = expected.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        for amplitude in &mut expected {
            *amplitude /= norm;
        }
        assert_phase_exact_state_matches(
            &stab_vec.state_vector(),
            &expected,
            "first forced projection",
        );
        assert_eq!(stab_vec.num_terms(), 2);
        assert!(
            stab_vec
                .terms
                .iter()
                .all(|(_, ch)| !ch.omega_exact().is_zero())
        );
        assert!(!stab_vec.has_shared_projection_structure());

        // q2 has uniform gamma and multiple live terms. The surviving terms'
        // other structure differs, so projection must process them separately.
        stab_vec.measure_qubit(2, Some(false));
        assert_phase_exact_state_matches(
            &stab_vec.state_vector(),
            &expected,
            "second forced projection after zero-omega cleanup",
        );
    }

    fn prepare_divergent_probability_counterexample() -> (StabVec, Vec<Complex64>) {
        let mut stab = StabVec::builder(7).pruning_threshold(0.0).seed(694).build();
        let mut dense = StateVecSoA::with_seed(7, 694);
        for (q, angle) in [(0, 0.37), (1, 0.53), (2, 0.71)] {
            let qubit = QubitId(q);
            let angle = Angle64::from_radians(angle);
            stab.h(&[qubit]).rz(angle, &[qubit]).h(&[qubit]);
            dense.h(&[qubit]).rz(angle, &[qubit]).h(&[qubit]);
        }
        stab.measure_qubit(0, Some(true));

        let mut projected = dense.state();
        for (basis, amplitude) in projected.iter_mut().enumerate() {
            if basis & 1 == 0 {
                *amplitude = Complex64::new(0.0, 0.0);
            }
        }
        let inv_norm = 1.0
            / projected
                .iter()
                .map(Complex64::norm_sqr)
                .sum::<f64>()
                .sqrt();
        for amplitude in &mut projected {
            *amplitude *= inv_norm;
        }
        let mut dense = StateVecSoA::from_state(&projected, PecosRng::seed_from_u64(694));

        stab.sz(&[QubitId(2)])
            .h(&[QubitId(5)])
            .cx(&[(QubitId(4), QubitId(5))])
            .h(&[QubitId(1)])
            .sz(&[QubitId(0), QubitId(0), QubitId(4)])
            .h(&[QubitId(2)]);
        dense
            .sz(&[QubitId(2)])
            .h(&[QubitId(5)])
            .cx(&[(QubitId(4), QubitId(5))])
            .h(&[QubitId(1)])
            .sz(&[QubitId(0), QubitId(0), QubitId(4)])
            .h(&[QubitId(2)]);
        let _ = stab.state_vector();
        (stab, dense.state())
    }

    #[test]
    fn divergent_structure_measurement_probability_matches_state_vec_soa() {
        let (stab, expected_state) = prepare_divergent_probability_counterexample();
        assert_eq!(stab.num_terms(), 4);
        assert!(!stab.has_shared_projection_structure());
        let expected_prob0 = expected_state
            .iter()
            .enumerate()
            .filter(|(basis, _)| (basis >> 2) & 1 == 0)
            .map(|(_, amplitude)| amplitude.norm_sqr())
            .sum::<f64>();
        assert!((expected_prob0 - 0.825_916_885_511).abs() < EPS);

        let samples = 20_000_u32;
        let mut zero_count = 0_u32;
        for seed in 0..samples {
            let mut sample = stab.clone();
            sample.rng = PecosRng::seed_from_u64(u64::from(seed));
            if !sample.mz(&[QubitId(2)])[0].outcome {
                zero_count += 1;
            }
        }
        let observed_prob0 = f64::from(zero_count) / f64::from(samples);
        assert!(
            (observed_prob0 - expected_prob0).abs() < 0.015,
            "observed Pr(0)={observed_prob0}, expected {expected_prob0}"
        );
    }

    #[test]
    fn divergent_projection_uses_exact_norm_and_preserves_state() {
        let (stab, state) = prepare_divergent_probability_counterexample();
        for outcome in [false, true] {
            let mut expected = state.clone();
            for (basis, amplitude) in expected.iter_mut().enumerate() {
                if (((basis >> 2) & 1) != 0) != outcome {
                    *amplitude = Complex64::new(0.0, 0.0);
                }
            }
            let inv_norm = 1.0 / expected.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
            for amplitude in &mut expected {
                *amplitude *= inv_norm;
            }
            let mut measured = stab.clone();
            measured.measure_qubit(2, Some(outcome));
            assert_phase_exact_state_matches(
                &measured.state_vector(),
                &expected,
                &format!("divergent projection outcome {outcome}"),
            );
        }
    }

    #[test]
    fn divergent_projection_does_not_merge_by_gamma_and_omega_alone() {
        let mut stab = StabVec::builder(3).pruning_threshold(0.0).build();
        let mut dense = StateVecSoA::new(3);
        for (q, radians) in [(0, 0.339_52), (1, 0.721_75), (2, 0.090_18)] {
            let qubit = QubitId(q);
            let angle = Angle64::from_radians(radians);
            stab.h(&[qubit]).rz(angle, &[qubit]).h(&[qubit]);
            dense.h(&[qubit]).rz(angle, &[qubit]).h(&[qubit]);
        }
        stab.rz(Angle64::from_radians(0.409_39), &qid(0))
            .cx(&[(QubitId(2), QubitId(0))])
            .sz(&qid(2))
            .rz(Angle64::from_radians(0.661_47), &qid(0))
            .cx(&[(QubitId(2), QubitId(0))])
            .sz(&qid(0))
            .rz(Angle64::from_radians(0.562_83), &qid(0))
            .cx(&[(QubitId(1), QubitId(2))])
            .sz(&qid(1))
            .rz(Angle64::from_radians(0.606_67), &qid(2));
        dense
            .rz(Angle64::from_radians(0.409_39), &qid(0))
            .cx(&[(QubitId(2), QubitId(0))])
            .sz(&qid(2))
            .rz(Angle64::from_radians(0.661_47), &qid(0))
            .cx(&[(QubitId(2), QubitId(0))])
            .sz(&qid(0))
            .rz(Angle64::from_radians(0.562_83), &qid(0))
            .cx(&[(QubitId(1), QubitId(2))])
            .sz(&qid(1))
            .rz(Angle64::from_radians(0.606_67), &qid(2));

        let mut expected = dense.state();
        let _ = stab.state_vector();
        assert!(!stab.has_shared_projection_structure());
        stab.measure_qubit(1, Some(false));
        for (basis, amplitude) in expected.iter_mut().enumerate() {
            if (basis >> 1) & 1 != 0 {
                *amplitude = Complex64::new(0.0, 0.0);
            }
        }
        let inv_norm = 1.0 / expected.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
        for amplitude in &mut expected {
            *amplitude *= inv_norm;
        }
        assert_phase_exact_state_matches(
            &stab.state_vector(),
            &expected,
            "divergent terms with matching gamma and omega",
        );
    }

    #[test]
    fn single_surviving_term_is_renormalized() {
        let mut stab = StabVec::builder(1).pruning_threshold(0.0).build();
        let angle = Angle64::from_radians(0.37);
        stab.h(&qid(0)).rz(angle, &qid(0)).h(&qid(0));
        assert_eq!(stab.num_terms(), 2);

        stab.measure_qubit(0, Some(false));
        assert_eq!(stab.num_terms(), 1);
        assert_phase_exact_state_matches(
            &stab.state_vector(),
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            "single surviving term",
        );
    }

    #[test]
    fn nondeterministic_single_term_projection_preserves_norm() {
        let mut stab = StabVec::builder(1).pruning_threshold(0.0).build();
        stab.h(&qid(0));
        let _ = stab.state_vector();
        assert_eq!(stab.num_terms(), 1);

        stab.measure_qubit(0, Some(false));
        assert_phase_exact_state_matches(
            &stab.state_vector(),
            &[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            "nondeterministic single-term projection",
        );
    }

    #[test]
    fn impossible_forced_measurement_retains_zero_term() {
        let mut stab = StabVec::builder(7).pruning_threshold(0.0).build();
        stab.measure_qubit(0, Some(true));
        assert_eq!(stab.num_terms(), 1);
        assert_eq!(stab.terms[0].0, Complex64::new(0.0, 0.0));

        stab.measure_qubit(1, Some(false));
        assert!(
            stab.state_vector()
                .iter()
                .all(|amplitude| *amplitude == Complex64::new(0.0, 0.0))
        );
    }

    #[cfg(debug_assertions)]
    #[test]
    fn c_type_contract_asserts_unchanged_structure() {
        for changed in ["F", "G", "v", "s"] {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let mut stab = StabVec::new(2);
                stab.apply_c_type_clifford(|ch| {
                    let f = ch.arc_f();
                    let g = ch.arc_g();
                    let m = ch.arc_m();
                    let v = ch.arc_v();
                    let s = ch.arc_s();
                    ch.set_arcs(
                        if changed == "F" {
                            std::sync::Arc::new((*f).clone())
                        } else {
                            f
                        },
                        if changed == "G" {
                            std::sync::Arc::new((*g).clone())
                        } else {
                            g
                        },
                        m,
                        if changed == "v" {
                            std::sync::Arc::new((*v).clone())
                        } else {
                            v
                        },
                        if changed == "s" {
                            std::sync::Arc::new((*s).clone())
                        } else {
                            s
                        },
                    );
                });
            }));
            assert!(result.is_err(), "missing {changed} contract assertion");
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    fn exact_overlap_contract_asserts_complete_gamma_diff_set() {
        let mut stab = StabVec::builder(2).pruning_threshold(0.0).build();
        let mut unused_reference = StateVecSoA::new(2);
        prepare_uniform_terms(&mut stab, &mut unused_reference, 2);
        assert!(stab.has_shared_projection_structure());
        assert_eq!(stab.gamma_diff_qubits, vec![0]);

        stab.gamma_diff_qubits.clear();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            stab.exact_norm_and_prob0(1);
        }));
        assert!(result.is_err(), "incomplete gamma diff set was accepted");
    }
}
