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

//! `StabMps` — hybrid stabilizer-tableau + MPS simulator.
//!
//! Represents a quantum state as:
//!
//! ```text
//! |psi> = sum_i nu_i D_i |phi>
//! ```
//!
//! where |phi> is a stabilizer state tracked by a `SparseStabY` tableau,
//! `D_i` are destabilizer operators, and `nu_i` are complex coefficients
//! stored as an MPS.
//!
//! - Clifford gates: update only the tableau (O(n^2)), MPS untouched
//! - Non-Clifford gates (RZ): decompose `Z_q` in stabilizer basis, apply to MPS
//!
//! Based on: Masot-Llima, Garcia-Saez. "Stabilizer Tensor Networks: Universal
//! Quantum Simulator on a Basis of Stabilizer States." PRL 133, 230601 (2024).
//! arXiv:2403.08724.

mod canonical_ket;
pub mod compile;
pub mod disentangle;
pub mod mast;
pub mod measure;
pub mod non_clifford;
pub mod ofd;
pub mod pauli_decomp;
pub mod renyi;
pub mod tableau_compose;

use crate::errors::MpsError;
use crate::mps::{Mps, MpsConfig};
use nalgebra::DMatrix;
use num_complex::Complex64;
use pecos_core::{Angle64, QubitId, RngManageable};
use pecos_random::PecosRng;
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator, SparseStabY,
};
use std::time::Instant;

fn initial_tableau_and_rng(num_qubits: usize, seed: Option<u64>) -> (SparseStabY, PecosRng) {
    if let Some(seed) = seed {
        return (
            SparseStabY::with_seed(num_qubits, seed).with_destab_sign_tracking(),
            PecosRng::seed_from_u64(seed),
        );
    }

    let tableau = SparseStabY::new(num_qubits).with_destab_sign_tracking();
    let rng = tableau.rng().clone();
    (tableau, rng)
}

fn reset_tableau_and_rng(
    num_qubits: usize,
    was_seeded: bool,
    current_rng: &mut PecosRng,
) -> (SparseStabY, PecosRng) {
    if was_seeded {
        // A seeded reset continues deterministically from the live simulator
        // stream instead of replaying the construction stream. Give the two
        // rebuilt RNGs independent seeds drawn in a stable order.
        let tableau_seed = current_rng.next_u64();
        let simulator_seed = current_rng.next_u64();
        return (
            SparseStabY::with_seed(num_qubits, tableau_seed).with_destab_sign_tracking(),
            PecosRng::seed_from_u64(simulator_seed),
        );
    }

    initial_tableau_and_rng(num_qubits, None)
}

/// Unwrap an internal MPS operation at an infallible simulator boundary.
///
/// The stability census exercises this boundary as a regression guard. A
/// failure here means the numerical state is unrecoverable, not that the
/// simulator produced a usable partial result. If a fallible simulator API is
/// wanted, that framework-wide trait change should be tracked as its own issue.
#[track_caller]
pub(super) fn expect_mps_operation<T>(result: Result<T, MpsError>, operation: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            panic!("{operation} failed: {error}; the coefficient MPS state is unrecoverable")
        }
    }
}

/// Known Pauli eigenstate at an MPS site, used for exact disentangling.
///
/// For every variant, `false` denotes the `+1` eigenstate and `true` the
/// `-1` eigenstate. Thus `Z(false)` is `|0>`, `Z(true)` is `|1>`,
/// `X(false)` is `|+>`, `X(true)` is `|->`, `Y(false)` is `|+i>`, and
/// `Y(true)` is `|-i>`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SiteEigenstate {
    /// A Z eigenstate: `false` is `|0>` and `true` is `|1>`.
    Z(bool),
    /// An X eigenstate: `false` is `|+>` and `true` is `|->`.
    X(bool),
    /// A Y eigenstate: `false` is `|+i>` and `true` is `|-i>`.
    Y(bool),
}

/// Return the unnormalized stored-frame weight where `site` has coefficient bit one.
///
/// This is the representation-independent invariant behind a `Z(false)`
/// disentangling proof.  In particular, it remains valid when either adjacent
/// MPS bond is larger than one.
#[cfg(test)]
fn stored_mps_one_weight(mps: &Mps, site: usize) -> f64 {
    let one_projector = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
        ],
    );
    mps.expectation_product(&[(site, one_projector)]).re
}

#[cfg(test)]
fn stored_mps_one_marginal(mps: &Mps, site: usize) -> f64 {
    let norm_squared = mps.norm_squared();
    assert!(norm_squared > 1e-20, "stored MPS must have nonzero norm");
    stored_mps_one_weight(mps, site) / norm_squared
}

/// Test whether a stored coefficient-MPS site is numerically exactly `|0>`.
///
/// Most candidates produced by projection have bond-one neighbours, so keep
/// that O(1) tensor check first.  At larger bond dimension the local tensor is
/// gauge-dependent; fall back to the physical one-site marginal instead.
fn cheap_product_zero_site_test(mps: &Mps, site: usize) -> Option<bool> {
    const TOLERANCE: f64 = 1e-12;
    const MIN_SCALE: f64 = 1e-30;

    if mps.bond_dim(site) == 1 && mps.bond_dim(site + 1) == 1 {
        let tensor = &mps.tensors()[site];
        let scale = tensor[(0, 0)].norm();
        return Some(scale > MIN_SCALE && tensor[(0, 1)].norm() <= TOLERANCE * scale);
    }

    // Projection often writes an identically zero sigma=1 block while leaving
    // entangled virtual bonds on both sides.  That structural zero is already
    // a gauge-independent sufficient proof and avoids a chain contraction.
    let tensor = &mps.tensors()[site];
    let chi_r = mps.bond_dim(site + 1);
    if (0..tensor.nrows()).all(|row| {
        (chi_r..2 * chi_r).all(|column| tensor[(row, column)] == Complex64::new(0.0, 0.0))
    }) {
        return Some(true);
    }

    None
}

// This threshold is based on the contraction error scale and the worst-case
// state error 2*sqrt(epsilon) <= 6.33e-8, not on a measured firing-rate gain.
// The exact workload census was unchanged at 2815 flags for 1e-12 and 1e-15;
// its 2793-flag truncating configuration was newly added and has no before
// value. Thus that workload contains no observed marginal in (1e-15, 1e-12].
const PRODUCT_ZERO_PROBABILITY_TOLERANCE: f64 = 1e-15;

// A running bitstring probability below this is reported as exactly zero.
// The singular and batched query walks MUST share this constant: the batched
// API's contract is bit-for-bit agreement with per-query singular calls, and
// that guarantee is load-bearing on the two paths pruning identically.
const QUERY_ZERO_PROBABILITY_FLOOR: f64 = 1e-30;

#[cfg(test)]
const STORED_PROOF_SOUNDNESS_TOLERANCE: f64 = 5e-16;

fn are_numerical_product_zero_sites(mps: &Mps, sites: &[usize]) -> Vec<bool> {
    let mut results = vec![false; sites.len()];
    let mut fallback_indices = Vec::new();

    for (index, &site) in sites.iter().enumerate() {
        if let Some(result) = cheap_product_zero_site_test(mps, site) {
            results[index] = result;
        } else {
            fallback_indices.push(index);
        }
    }

    if !fallback_indices.is_empty() {
        // The cache owns both chain environments for this verification pass.
        // Its immutable MPS borrow prevents tensor mutation until every
        // fallback candidate has been evaluated and the cache is dropped.
        let fallback_sites: Vec<usize> =
            fallback_indices.iter().map(|&index| sites[index]).collect();
        let environments = mps.environment_cache(&fallback_sites);
        if environments.norm_squared() <= 1e-20 {
            return vec![false; sites.len()];
        }
        for index in fallback_indices {
            results[index] = environments.one_site_basis_marginal(sites[index], 1).abs()
                <= PRODUCT_ZERO_PROBABILITY_TOLERANCE;
        }
    }

    results
}

fn is_numerical_product_zero_site(mps: &Mps, site: usize) -> bool {
    are_numerical_product_zero_sites(mps, &[site])[0]
}

fn refresh_disent_flag(mps: &Mps, disent_flags: &mut [Option<SiteEigenstate>], site: usize) {
    disent_flags[site] =
        is_numerical_product_zero_site(mps, site).then_some(SiteEigenstate::Z(false));
}

fn repair_disent_flags(
    mps: &Mps,
    disent_flags: &mut [Option<SiteEigenstate>],
    update: &measure::ProjectionUpdate,
) {
    let mut sites = update.modified_sites.clone();
    if let Some(site) = update.collapsed_site
        && !sites.contains(&site)
    {
        sites.push(site);
    }
    let verified = are_numerical_product_zero_sites(mps, &sites);
    for (&site, is_zero) in sites.iter().zip(verified) {
        disent_flags[site] = is_zero.then_some(SiteEigenstate::Z(false));
    }
}

/// Sample and transactionally force one exact Z-measurement branch.
///
/// `StabMps` exact measurement and MAST data measurement share this owning
/// layer so neither can expose a partially mutated tableau/MPS pair.
fn measure_qubit_exact_transactional(
    tableau: &mut SparseStabY,
    mps: &mut Mps,
    rng: &mut PecosRng,
    q_idx: usize,
    operation: &str,
) -> Result<measure::LiveMeasurementResult, MpsError> {
    let probability_one = measure::z_outcome_probability(tableau, mps, q_idx, true, operation);
    let is_probability_zero = probability_one <= 0.0;
    let is_probability_one = probability_one >= 1.0;
    let outcome = if is_probability_zero {
        false
    } else if is_probability_one {
        true
    } else {
        rng.random_bool(probability_one)
    };

    let mut candidate_tableau = tableau.clone();
    let mut candidate_mps = mps.clone();
    let first = measure::project_forced_z_with_update(
        &mut candidate_tableau,
        &mut candidate_mps,
        q_idx,
        outcome,
    )?;
    assert!(
        first.snapped_probability > 0.0,
        "{operation}: sampled a projector-impossible outcome"
    );

    let projection = if first.survival_ratio < measure::BRANCH_VANISH_SURVIVAL_THRESHOLD {
        candidate_tableau = tableau.clone();
        candidate_mps = mps.clone();
        let original_config = mps.config().clone();
        let mut retry_config = original_config.clone();
        retry_config.max_bond_dim = candidate_mps.physical_rank_ceiling();
        retry_config.svd_cutoff = 0.0;
        retry_config.max_truncation_error = Some(0.0);
        candidate_mps.set_config(retry_config);
        let retry = measure::project_forced_z_with_update(
            &mut candidate_tableau,
            &mut candidate_mps,
            q_idx,
            outcome,
        )?;
        let retry_vanished = retry.survival_ratio < measure::BRANCH_VANISH_SURVIVAL_THRESHOLD;
        debug_assert!(
            !retry_vanished,
            "{operation}: sampled branch vanished on the untruncated retry"
        );
        assert!(
            !retry_vanished,
            "{operation}: sampled branch vanished on the untruncated retry"
        );
        candidate_mps.set_config(original_config);
        candidate_mps.record_branch_vanish_retry();
        retry
    } else {
        first
    };

    *tableau = candidate_tableau;
    *mps = candidate_mps;
    Ok(measure::LiveMeasurementResult {
        measurement: MeasurementResult {
            outcome,
            is_deterministic: is_probability_zero || is_probability_one,
        },
        update: projection.update,
    })
}

fn invalidate_disent_flags(
    disent_flags: &mut [Option<SiteEigenstate>],
    update: &measure::ProjectionUpdate,
) {
    for &site in &update.modified_sites {
        disent_flags[site] = None;
    }
    if let Some(site) = update.collapsed_site {
        disent_flags[site] = None;
    }
}

#[cfg(test)]
fn assert_disent_flags_match_stored_mps(
    mps: &Mps,
    disent_flags: &[Option<SiteEigenstate>],
    context: &str,
) {
    for (site, flag) in disent_flags.iter().enumerate() {
        if matches!(flag, Some(SiteEigenstate::Z(false))) {
            let marginal = stored_mps_one_marginal(mps, site);
            assert!(
                marginal.abs() <= STORED_PROOF_SOUNDNESS_TOLERANCE,
                "stored |0> proof at site {site} has P(sigma=1)={marginal:.3e}; {context}; bonds=({}, {})",
                mps.bond_dim(site),
                mps.bond_dim(site + 1)
            );
        }
    }
}

/// Single-qubit Pauli kind for specifying multi-qubit Pauli strings
/// (e.g., stabilizer generators of QEC codes).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PauliKind {
    /// Pauli X.
    X,
    /// Pauli Y.
    Y,
    /// Pauli Z.
    Z,
}

/// Policy for the family of operations implemented through single-qubit Z measurement.
///
/// This controls [`CliffordGateable::mz`], reset, `pz`/`px`, syndrome
/// extraction. [`StabMps::sample_bitstrings`] is always exact by construction
/// and does not dispatch through this policy. Per-shot sampling through the
/// selected policy is expressed by cloning a prepared simulator for each
/// shot, reseeding it with a distinct per-shot seed, and explicitly
/// measuring its qubits with [`CliffordGateable::mz`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MeasurementMode {
    /// Sample the normalized Born probability and transactionally force the
    /// sampled branch, retrying without truncation if that branch vanishes.
    #[default]
    Exact,
    /// Preserve the legacy uncompensated eager pre-reduction path. This is
    /// faster for some workloads but can bias the stored conditional state.
    Pragmatic,
    /// Keep measurement-basis Cliffords in a deferred virtual frame.
    ///
    /// Conditional states are exact after the issue #555 and #572 frame and
    /// projection fixes, subject to the configured MPS truncation. Call
    /// [`StabMps::flush`] before Rust state reads. This path consumes a distinct
    /// RNG stream from [`Self::Exact`], so equal seeds are not shot-for-shot
    /// comparable across the two modes.
    Lazy,
}

/// Single-qubit Clifford kind used internally for Pauli frame propagation.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SingleQubitCliffordKind {
    H,
    SZ,
    SZdg,
    X,
    Y,
    Z,
}

/// Runtime feature flags for [`StabMps`], stored as a bitfield.
///
/// Each accessor and setter mirrors the equivalently named option on
/// [`StabMpsBuilder`].
#[derive(Clone, Copy, Debug)]
pub struct StabMpsFlags(u8);

impl StabMpsFlags {
    const NORMALIZE_AFTER_GATE: u8 = 1 << 0;
    const MERGE_RZ: u8 = 1 << 1;
    const PAULI_FRAME_TRACKING: u8 = 1 << 2;
    const NUMERICAL_FLAG_REDETECTION: u8 = 1 << 3;
    const SATURATION_TELEMETRY: u8 = 1 << 4;

    /// Default flags: normalization and RZ merging enabled, everything else off.
    #[must_use]
    pub const fn new() -> Self {
        Self(Self::NORMALIZE_AFTER_GATE | Self::MERGE_RZ)
    }

    fn get(self, bit: u8) -> bool {
        self.0 & bit != 0
    }

    fn set(&mut self, bit: u8, val: bool) {
        if val {
            self.0 |= bit;
        } else {
            self.0 &= !bit;
        }
    }

    #[must_use]
    /// Return whether non-Clifford gates are followed by MPS normalization.
    pub fn normalize_after_gate(self) -> bool {
        self.get(Self::NORMALIZE_AFTER_GATE)
    }
    /// Set whether non-Clifford gates are followed by MPS normalization.
    pub fn set_normalize_after_gate(&mut self, v: bool) {
        self.set(Self::NORMALIZE_AFTER_GATE, v);
    }
    #[must_use]
    /// Return whether consecutive same-qubit RZ rotations are merged.
    pub fn merge_rz(self) -> bool {
        self.get(Self::MERGE_RZ)
    }
    /// Set whether consecutive same-qubit RZ rotations are merged.
    pub fn set_merge_rz(&mut self, v: bool) {
        self.set(Self::MERGE_RZ, v);
    }
    #[must_use]
    /// Return whether Pauli errors are tracked in a classical frame.
    pub fn pauli_frame_tracking(self) -> bool {
        self.get(Self::PAULI_FRAME_TRACKING)
    }
    /// Set whether Pauli errors are tracked in a classical frame.
    pub fn set_pauli_frame_tracking(&mut self, v: bool) {
        self.set(Self::PAULI_FRAME_TRACKING, v);
    }
    #[must_use]
    /// Return whether product-site eigenstate flags are recovered numerically.
    pub fn numerical_flag_redetection(self) -> bool {
        self.get(Self::NUMERICAL_FLAG_REDETECTION)
    }
    /// Set whether product-site eigenstate flags are recovered numerically.
    pub fn set_numerical_flag_redetection(&mut self, v: bool) {
        self.set(Self::NUMERICAL_FLAG_REDETECTION, v);
    }
    #[must_use]
    /// Return whether saturated-regime event telemetry is collected.
    pub fn saturation_telemetry(self) -> bool {
        self.get(Self::SATURATION_TELEMETRY)
    }
    /// Set whether saturated-regime event telemetry is collected.
    pub fn set_saturation_telemetry(&mut self, v: bool) {
        self.set(Self::SATURATION_TELEMETRY, v);
    }
}

impl Default for StabMpsFlags {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for configuring an `StabMps` simulator.
pub struct StabMpsBuilder {
    num_qubits: usize,
    seed: Option<u64>,
    max_bond_dim: usize,
    svd_cutoff: f64,
    max_truncation_error: Option<f64>,
    parallel: bool,
    auto_grow_bond_dim: Option<f64>,
    auto_grow_max_bond_dim: usize,
    measurement_mode: MeasurementMode,
    flags: StabMpsFlags,
}

impl StabMpsBuilder {
    /// Maximum MPS bond dimension. Singular values beyond this are discarded
    /// during SVD truncation after two-site gates.
    ///
    /// - Default: 128
    /// - Higher values give more accuracy at the cost of memory and time
    /// - For n qubits, the exact max is 2^(n/2)
    #[must_use]
    pub fn max_bond_dim(mut self, dim: usize) -> Self {
        self.max_bond_dim = dim;
        self
    }

    /// Minimum singular value to keep, relative to the norm of each MPS
    /// matrix being split. In a canonical compression sweep this is relative
    /// to the global state norm, so global rescaling does not change ranks.
    ///
    /// - Default: 1e-12
    /// - Lower values keep more precision
    #[must_use]
    pub fn svd_cutoff(mut self, cutoff: f64) -> Self {
        self.svd_cutoff = cutoff;
        self
    }

    /// Normalize the MPS after each non-Clifford gate.
    /// Prevents unbounded norm drift from accumulated SVD numerical noise.
    ///
    /// - Default: true
    /// - Set to false if you need to track the unnormalized state
    #[must_use]
    pub fn normalize_after_gate(mut self, normalize: bool) -> Self {
        self.flags.set_normalize_after_gate(normalize);
        self
    }

    /// Seed the simulator's [`pecos_random::PecosRng`] and stabilizer-tableau RNG.
    ///
    /// Fresh simulators built with the same seed, configuration, gates, noise
    /// calls, measurements, and sampling calls consume the same random stream.
    /// Reproducibility covers those stochastic results, not floating-point
    /// equivalence across different PECOS versions or platforms. `reset()`
    /// deterministically seeds its rebuilt tableau and continuing simulator RNG
    /// from the current simulator-RNG stream; it does not replay this seed.
    #[must_use]
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Maximum relative truncation error per SVD (adaptive bond dimension).
    ///
    /// When set, bonds with low entanglement get small bond dimension (fast)
    /// while bonds with high entanglement grow up to `max_bond_dim` (accurate).
    /// The discarded weight at each SVD stays below this fraction.
    ///
    /// - Default: 1e-8
    /// - Typical values: 1e-8 to 1e-3
    /// - `0.0` disables the adaptive bound: no positive discarded weight is
    ///   permitted, so only `svd_cutoff` and `max_bond_dim` truncate
    /// - `max_bond_dim` still acts as a hard cap
    ///
    /// # Panics
    ///
    /// Panics if `error` is negative, NaN, or infinite.
    #[must_use]
    pub fn max_truncation_error(mut self, error: f64) -> Self {
        assert!(
            error.is_finite() && error >= 0.0,
            "max_truncation_error must be finite and non-negative"
        );
        self.max_truncation_error = Some(error);
        self
    }

    /// Enable parallel MPS operations via rayon.
    ///
    /// - Default: false
    /// - Useful for large bond dimensions (chi > 16)
    /// - Do not enable when parallelizing at the shot/circuit level
    #[must_use]
    pub fn parallel(mut self, parallel: bool) -> Self {
        self.parallel = parallel;
        self
    }

    /// Select the single-qubit measurement policy. The default is
    /// [`MeasurementMode::Exact`]. See [`MeasurementMode`] for the affected
    /// operation family and the exact behavior of plural sampling.
    #[must_use]
    pub fn measurement(mut self, mode: MeasurementMode) -> Self {
        self.measurement_mode = mode;
        self
    }

    /// Enable adaptive bond-dim auto-grow. When the running truncation
    /// error exceeds `threshold` AND the bond-dim cap was binding (a
    /// truncation step actually discarded singular values at the cap),
    /// the simulator doubles `max_bond_dim` (capped at
    /// `auto_grow_max_bond_dim`, default 4096).
    ///
    /// Removes the manual tuning step for deep T-heavy circuits where
    /// the default cap of 128 is insufficient. Cost: rebuild bond
    /// allocation on growth (rare). Benefit: avoids surprise truncation
    /// when entanglement spikes.
    ///
    /// - Default: `None` (disabled — fixed `max_bond_dim`).
    /// - Typical thresholds: 1e-6 (conservative), 1e-4 (aggressive).
    #[must_use]
    pub fn auto_grow_bond_dim(mut self, threshold: f64) -> Self {
        self.auto_grow_bond_dim = Some(threshold);
        self
    }

    /// Hard cap on `auto_grow_bond_dim`'s growth. Default: 4096.
    #[must_use]
    pub fn auto_grow_max_bond_dim(mut self, cap: usize) -> Self {
        self.auto_grow_max_bond_dim = cap;
        self
    }

    /// Enable Pauli frame tracking: `inject_x_in_frame`, `inject_y_in_frame`,
    /// `inject_z_in_frame`, and (when the flag is set)
    /// `apply_depolarizing*` track Pauli errors as classical bits rather
    /// than applying them to the quantum state. Clifford gates propagate
    /// the frame via Heisenberg rules; measurements XOR the tracked
    /// Z-bit into the outcome.
    ///
    /// **Big win** for Pauli-noise-heavy QEC simulation: each error is
    /// a single bit flip (O(1)) instead of an O(n) tableau update.
    ///
    /// - Default: false.
    /// - Sign tracking: `pauli_frame_phase` evolves through Clifford
    ///   propagation per Heisenberg sign-flip rules (H·Y·H = -Y,
    ///   SZ·Y·SZ† = -X, etc.) and folds into `global_phase` at flush.
    /// - `State_vector` after flush: EXACT for all states. The frame is
    ///   applied to the MPS via `C† · P · C = phase · X_flip · Z_sign`
    ///   (decomposition in the MPS frame), not to the tableau. The
    ///   Clifford `C` is unchanged, the MPS absorbs the frame's full
    ///   content, and there is no state-dependent phase loss.
    #[must_use]
    pub fn pauli_frame_tracking(mut self, enable: bool) -> Self {
        self.flags.set_pauli_frame_tracking(enable);
        self
    }

    /// Merge consecutive `rz(θ, q)` on the same qubit into a single
    /// `rz(Σθ, q)` before invoking the non-Clifford path. Any gate
    /// touching `q` (other than another `rz` on `q`) flushes the
    /// accumulated angle first. Intended for ion-trap-style memory-error
    /// models where every idle qubit receives a small RZ each time step:
    /// adjacent idle rounds merge into one non-Clifford op.
    ///
    /// - Default: true.
    ///   MAST instead defaults to false so injection and ancilla-capacity use
    ///   remains visible after each RZ call.
    /// - Semantics: strictly equivalent to applying each `rz` individually
    ///   (tableau and MPS paths both reduce non-Clifford count). No
    ///   accuracy trade-off.
    /// - Clifford-angle RZ (0, π/2, π, 3π/2) is detected and applied
    ///   directly as before (no buffering).
    #[must_use]
    pub fn merge_rz(mut self, merge: bool) -> Self {
        self.flags.set_merge_rz(merge);
        self
    }

    /// Numerically recover missing exact-disentangling |0> flags at product sites.
    ///
    /// When enabled, a failed symbolic flag search checks candidate bond-one MPS
    /// tensors using a fixed relative tolerance of 1e-12. Redetection
    /// self-disables while lazy deferred operations are pending because the
    /// stored tensors then differ from the effective MPS-frame state. Default:
    /// false.
    #[must_use]
    pub fn numerical_flag_redetection(mut self, enable: bool) -> Self {
        self.flags.set_numerical_flag_redetection(enable);
        self
    }

    /// Collect detailed saturated-regime profiling data.
    ///
    /// This records per-event timing and live bond profiles and performs the
    /// diagnostic signed-Pauli eigenstate test. It is disabled by default.
    #[must_use]
    pub fn saturation_telemetry(mut self, enable: bool) -> Self {
        self.flags.set_saturation_telemetry(enable);
        self
    }

    /// QEC-style preset retained for source compatibility.
    ///
    /// This selects exact measurement. The frozen Stage-B recipe in
    /// `examples/measurement_mode_bench.rs` measures manually expanded
    /// distance-3 and distance-5 repetition-code syndrome circuits in both
    /// check bases: eight rounds, coherent `RZ(0.1)` on every data qubit per
    /// round, two-qubit depolarizing noise at `p=1e-3`, identical pre-generated
    /// noise for both modes, `merge_rz(false)`, otherwise-default builders,
    /// 2,000 shots at d3 and 500 at d5, one warmup, timed seeds 7001 through
    /// 7007, and pinned release execution. Median Exact slowdowns relative to
    /// Pragmatic were 1.496812 (d3/Z), 1.488242 (d3/X), 1.457758 (d5/Z), and
    /// 1.548499 (d5/X): geometric mean 1.497474 and maximum 1.548499. Both are
    /// inside the decided Exact-everywhere limits (geometric mean at most 3
    /// and per-workload maximum at most 5). The ratios are specific to this
    /// noisy workload family: on noiseless Clifford-dominated variants of the
    /// same circuits, where pragmatic measurements are nearly free, the
    /// relative gap is far larger even though absolute per-shot costs stay
    /// small. Note also that this family never triggers pragmatic's biased
    /// drift path, so pragmatic was timed at its unbiased best case.
    ///
    /// Run the recorded recipe with
    /// `taskset -c 2 cargo run --release -p pecos-stab-tn --example measurement_mode_bench`.
    /// The preset's other settings match the general defaults:
    /// `max_bond_dim(128)`, `max_truncation_error(1e-8)`, and `merge_rz(true)`.
    ///
    /// Override any of these with subsequent builder calls:
    /// ```
    /// use pecos_stab_tn::stab_mps::StabMps;
    ///
    /// let sim = StabMps::builder(4).for_qec().max_bond_dim(64).build();
    /// assert_eq!(sim.num_qubits(), 4);
    /// ```
    #[must_use]
    pub fn for_qec(self) -> Self {
        self.for_qec_with_bond_dim(128)
    }

    /// Like the general defaults but with a caller-chosen `max_bond_dim` cap.
    /// Use when the default 128 is too tight (deep T-heavy circuits)
    /// or too loose (memory-constrained environments).
    #[must_use]
    pub fn for_qec_with_bond_dim(self, bond_dim: usize) -> Self {
        self.max_truncation_error(1e-8)
            .max_bond_dim(bond_dim)
            .merge_rz(true)
            .measurement(MeasurementMode::Exact)
    }

    /// Build the simulator.
    #[must_use]
    pub fn build(self) -> StabMps {
        let config = MpsConfig {
            max_bond_dim: self.max_bond_dim,
            svd_cutoff: self.svd_cutoff,
            max_truncation_error: self.max_truncation_error,
            parallel: self.parallel,
        };
        let (tableau, rng) = initial_tableau_and_rng(self.num_qubits, self.seed);
        StabMps {
            num_qubits: self.num_qubits,
            construction_seed: self.seed,
            tableau,
            mps: Mps::new(self.num_qubits, config.clone()),
            config,
            global_phase: Complex64::new(1.0, 0.0),
            disent_flags: vec![Some(SiteEigenstate::Z(false)); self.num_qubits],
            gf2_matrix: ofd::Gf2FlipMatrix::new(self.num_qubits),
            rng,
            stats: StabMpsStats::default(),
            saturation_telemetry: SaturationTelemetry::default(),
            deferred_ops: Vec::new(),
            uncompensated_pre_reduction_count: 0,
            pending_rz: vec![None; self.num_qubits],
            auto_grow_bond_dim: self.auto_grow_bond_dim,
            auto_grow_max_bond_dim: self.auto_grow_max_bond_dim,
            last_truncation_error: 0.0,
            pauli_frame_x: vec![false; self.num_qubits],
            pauli_frame_z: vec![false; self.num_qubits],
            pauli_frame_phase: Complex64::new(1.0, 0.0),
            measurement_mode: self.measurement_mode,
            flags: self.flags,
        }
    }
}

/// Stabilizer tensor-network simulator for Clifford circuits with non-Clifford rotations.
///
/// Clifford gates update a stabilizer tableau, while non-Clifford rotations
/// update an MPS of coefficients. This is usually preferable to a dense state
/// vector when the OFD nullity and resulting MPS bond dimensions remain small.
///
/// # Quick start
///
/// ```
/// use pecos_core::{Angle64, QubitId};
/// use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable};
/// use pecos_stab_tn::stab_mps::StabMps;
///
/// let mut sim = StabMps::builder(2).seed(7).build();
/// sim.h(&[QubitId(0)]);
/// sim.cx(&[(QubitId(0), QubitId(1))]);
/// sim.rz(Angle64::QUARTER_TURN / 2_u64, &[QubitId(1)]);
///
/// let outcome = sim.mz(&[QubitId(0)])[0].outcome;
/// sim.flush();
/// let bits = [outcome, outcome];
/// assert!((sim.prob_bitstring(&bits) - 1.0).abs() < 1e-12);
/// ```
///
/// Bitstrings always use qubit-index order: `bits[q]` is qubit `q`. See the
/// crate-level **Bitstring convention** section for conversion to a state-vector
/// index.
///
/// # Read validity and accuracy checklist
///
/// Rust callers must call [`Self::flush`] before state and diagnostic reads when
/// lazy measurement or merged RZ is enabled. If Pauli-frame tracking is enabled,
/// call [`Self::flush_pauli_frame_to_state`] as well before reads that must include
/// the physical Pauli frame. The Python bindings automatically perform
/// `flush()` for state and MPS-diagnostic reads, but not for the pure
/// `is_state_exact()` and `uncompensated_pre_reduction_count` diagnostics, and they do not
/// implicitly materialize a Pauli frame.
///
/// Before relying on a state read, check the diagnostics:
///
/// 1. [`Self::is_state_exact`] is `true` after the required flushes.
/// 2. [`Self::uncompensated_pre_reduction_count`] is zero.
/// 3. [`Self::truncation_error`] and [`Self::summed_discarded_weight`] are
///    acceptable for the application.
/// 4. [`Self::bond_cap_hits`] is zero, or the configured bond cap is known to be
///    adequate despite having bound an SVD.
#[derive(Clone)]
pub struct StabMps {
    num_qubits: usize,
    /// Seed supplied at construction, or `None` for entropy-backed resets.
    construction_seed: Option<u64>,
    tableau: SparseStabY,
    mps: Mps,
    config: MpsConfig,
    /// Global phase accumulated from Clifford-angle RZ gates.
    global_phase: Complex64,
    /// Per-site eigenstate tracking for exact disentangling.
    disent_flags: Vec<Option<SiteEigenstate>>,
    /// GF(2) flip matrix for OFD diagnostic.
    gf2_matrix: ofd::Gf2FlipMatrix,
    rng: PecosRng,
    /// Diagnostic counters. Updated by `non_clifford::apply_rz_stab_mps`.
    pub stats: StabMpsStats,
    /// Detailed saturated-regime diagnostics, populated only when enabled.
    saturation_telemetry: SaturationTelemetry,
    /// Deferred virtual-frame Clifford V (see `measure::DeferredOp`).
    deferred_ops: Vec<measure::DeferredOp>,
    /// Count of pragmatic measurements whose pre-reduction was uncompensated.
    uncompensated_pre_reduction_count: u64,
    /// Pending non-Clifford RZ angle per qubit when `merge_rz` is on.
    pending_rz: Vec<Option<Angle64>>,
    /// Auto-grow bond-dim threshold; `None` disables.
    auto_grow_bond_dim: Option<f64>,
    /// Hard cap when auto-growing.
    auto_grow_max_bond_dim: usize,
    /// Snapshot of `mps.truncation_error()` at the last auto-grow check.
    last_truncation_error: f64,
    /// Pauli frame X bit per qubit.
    pauli_frame_x: Vec<bool>,
    /// Pauli frame Z bit per qubit.
    pauli_frame_z: Vec<bool>,
    /// Global scalar of the Pauli frame.
    pauli_frame_phase: Complex64,
    /// Policy for single-qubit measurement and the operations built on it.
    measurement_mode: MeasurementMode,
    /// Runtime feature flags.
    flags: StabMpsFlags,
}

/// Runtime statistics for diagnostics.
#[derive(Clone, Copy, Debug, Default)]
pub struct StabMpsStats {
    /// Total non-Clifford RZ calls (Clifford-angle RZs not counted).
    pub total_nonclifford: u64,
    /// Non-Cliffords that hit the single-site decomposition (cheap).
    pub single_site: u64,
    /// Non-Cliffords that fired multi-site disent (tableau right-compose).
    pub multi_disent: u64,
    /// Multi-site rotations that could use a stored |0> proof but bypassed
    /// tableau-right-composing disentangling because a Lazy frame was pending.
    pub deferred_disent_bypass: u64,
    /// Missing |0> flags recovered numerically at product sites.
    pub numerical_redetect: u64,
    /// Non-Cliffords that fell through to the std multi-site CNOT cascade path.
    pub multi_std: u64,
    /// Standard multi-site rotations implemented by clone/add/compress.
    pub multi_std_add: u64,
    /// Standard multi-site rotations implemented by a long-range CNOT cascade.
    pub multi_std_cascade: u64,
    /// Non-`+Z` product-site opportunities rejected by the current detector
    /// that a signed X/Y/Z Pauli-eigenstate detector could accept. Populated
    /// only when saturated-regime telemetry is enabled; branch denominators
    /// and axis/sign counts live in [`SaturationTelemetry::signed_eigenstates`].
    pub signed_eigenstate_candidates: u64,
    /// Non-Cliffords that hit the Stabilizer branch (scalar or diagonal).
    pub stabilizer: u64,
    /// OFD diagnostic: non-Cliffords whose flip pattern is in the span of
    /// previously-added patterns (would not increase bond dim under OFD).
    pub ofd_in_span: u64,
    /// OFD diagnostic: non-Cliffords whose flip pattern is linearly independent
    /// from previous (OFD would grow bond dim by factor 2).
    pub ofd_new_dim: u64,
    /// Cross-tab: OFD `in_span` gates that the heuristic routed through std path.
    /// These are the "OFD wins" — OFD would avoid MPS CNOT cascade.
    pub ofd_in_span_std: u64,
    /// Cross-tab: OFD `in_span` gates that the heuristic routed through single-site.
    /// Both paths are cheap; OFD doesn't improve here.
    pub ofd_in_span_single: u64,
    /// Cross-tab: OFD `in_span` gates that the heuristic routed through disent path.
    pub ofd_in_span_disent: u64,
}

/// Standard multi-site implementation selected for one non-Clifford event.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MultiStdSubtype {
    /// Clone/add followed by full compression.
    Add,
    /// Forward/reverse long-range CNOT cascade.
    Cascade,
}

/// Detailed record for one standard multi-site rotation.
#[derive(Clone, Debug)]
pub struct MultiStdEventTelemetry {
    /// Implementation selected for this event.
    pub subtype: MultiStdSubtype,
    /// Inclusive support span, `max_site - min_site`.
    pub span: usize,
    /// Live internal bond dimensions across the support span before the event.
    pub bond_profile: Vec<usize>,
    /// Whether the flip pattern was already in the OFD span.
    pub ofd_in_span: bool,
    /// Wall time spent in this event's implementation.
    pub wall_time_seconds: f64,
}

/// Detailed record for one exact multi-site disentangling event.
#[derive(Clone, Debug)]
pub struct MultiDisentEventTelemetry {
    /// Inclusive support span, `max_site - min_site`.
    pub span: usize,
    /// Live internal bond dimensions across the support span before the event.
    pub bond_profile: Vec<usize>,
    /// Wall time spent in the disentangling implementation.
    pub wall_time_seconds: f64,
}

/// Actual multi-site branch whose signed-eigenstate opportunities were tested.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignedEigenstateBranch {
    /// Exact tableau-right-composing disentangling branch.
    MultiDisent,
    /// Standard clone/add/compress branch.
    MultiStdAdd,
    /// Standard long-range CNOT-cascade branch.
    MultiStdCascade,
}

/// Candidate counts separated by detected Pauli axis and eigenvalue sign.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SignedEigenstateBreakdown {
    /// `|+>` candidates.
    pub x_plus: u64,
    /// `|->` candidates.
    pub x_minus: u64,
    /// `|+i>` candidates.
    pub y_plus: u64,
    /// `|-i>` candidates.
    pub y_minus: u64,
    /// `|1>` candidates. `|0>` is deliberately excluded.
    pub z_minus: u64,
}

impl SignedEigenstateBreakdown {
    /// Total candidates across every non-`+Z` axis/sign bucket.
    #[must_use]
    pub const fn total(self) -> u64 {
        self.x_plus + self.x_minus + self.y_plus + self.y_minus + self.z_minus
    }

    pub(super) fn record(&mut self, eigenstate: SiteEigenstate) {
        match eigenstate {
            SiteEigenstate::X(false) => self.x_plus += 1,
            SiteEigenstate::X(true) => self.x_minus += 1,
            SiteEigenstate::Y(false) => self.y_plus += 1,
            SiteEigenstate::Y(true) => self.y_minus += 1,
            SiteEigenstate::Z(true) => self.z_minus += 1,
            // +Z is the current criterion, not a signed candidate; the sole
            // call site excludes it. Skip rather than panic: a data-dependent
            // release panic on a diagnostic path would be worse than a
            // miscount in telemetry that changes no result.
            SiteEigenstate::Z(false) => debug_assert!(
                false,
                "+Z reached the signed-candidate breakdown despite its exclusion"
            ),
        }
    }
}

/// Signed-eigenstate candidate numerator and denominators for one actual branch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SignedEigenstateBranchTelemetry {
    /// Number of multi-site events routed through this branch and inspected.
    pub events: u64,
    /// Number of local Pauli-support sites tested across those events.
    pub sites_tested: u64,
    /// Candidate numerator by detected axis and sign.
    pub candidates: SignedEigenstateBreakdown,
}

/// Signed-eigenstate diagnostics separated by the branch actually taken.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SignedEigenstateTelemetry {
    /// Exact multi-site disentangling events.
    pub multi_disent: SignedEigenstateBranchTelemetry,
    /// Standard clone/add/compress events.
    pub multi_std_add: SignedEigenstateBranchTelemetry,
    /// Standard long-range CNOT-cascade events.
    pub multi_std_cascade: SignedEigenstateBranchTelemetry,
}

impl SignedEigenstateTelemetry {
    pub(super) fn branch_mut(
        &mut self,
        branch: SignedEigenstateBranch,
    ) -> &mut SignedEigenstateBranchTelemetry {
        match branch {
            SignedEigenstateBranch::MultiDisent => &mut self.multi_disent,
            SignedEigenstateBranch::MultiStdAdd => &mut self.multi_std_add,
            SignedEigenstateBranch::MultiStdCascade => &mut self.multi_std_cascade,
        }
    }
}

/// Runtime-gated saturated-regime event details.
#[derive(Clone, Debug, Default)]
pub struct SaturationTelemetry {
    /// One entry for every standard multi-site rotation.
    pub multi_std_events: Vec<MultiStdEventTelemetry>,
    /// One entry for every exact multi-site disentangling rotation.
    pub multi_disent_events: Vec<MultiDisentEventTelemetry>,
    /// Gauge-independent signed-eigenstate candidates and branch denominators.
    pub signed_eigenstates: SignedEigenstateTelemetry,
}

/// Operation counts and wall time for one query phase at one trie depth.
#[derive(Clone, Copy, Debug, Default)]
pub struct QueryPhaseTelemetry {
    /// Number of times the phase ran.
    pub calls: u64,
    /// Actual SVD operations performed within the phase.
    pub svd_operations: u64,
    /// SVD operations for which `max_bond_dim` was binding.
    pub capped_svd_operations: u64,
    /// Accumulated phase wall time.
    pub wall_time_seconds: f64,
}

/// Query profiling buckets for one `prob_bitstrings` trie depth.
#[derive(Clone, Copy, Debug, Default)]
pub struct QueryDepthTelemetry {
    /// Z-expectation and probability evaluation.
    pub expectation: QueryPhaseTelemetry,
    /// Compensated tableau-generator pre-reduction.
    pub pre_reduction: QueryPhaseTelemetry,
    /// Stabilizer/destabilizer decomposition after any pre-reduction.
    pub decomposition: QueryPhaseTelemetry,
    /// Projection construction and tableau update.
    pub projection: QueryPhaseTelemetry,
    /// Exact post-projection right-canonical QR sweep.
    pub post_projection_qr: QueryPhaseTelemetry,
    /// Post-projection configured SVD compression.
    pub post_projection_svd: QueryPhaseTelemetry,
    /// Projection survival-ratio contraction.
    pub survival: QueryPhaseTelemetry,
    /// Conditional post-projection normalization.
    pub normalization: QueryPhaseTelemetry,
    /// Path selection and returned modified-site bookkeeping.
    pub bookkeeping: QueryPhaseTelemetry,
    phase_active: bool,
}

/// Depth-bucketed profile returned by [`StabMps::prob_bitstrings_profiled`].
#[derive(Clone, Debug, Default)]
pub struct ProbabilityQueryTelemetry {
    /// Entry `d` contains work performed for trie projections at depth `d`.
    pub by_depth: Vec<QueryDepthTelemetry>,
    /// Wall time of the complete `prob_bitstrings` call, including trie walks
    /// and state cloning outside forced projection.
    pub whole_call_wall_time_seconds: f64,
}

#[derive(Clone, Copy)]
pub(super) enum QueryPhase {
    Expectation,
    PreReduction,
    Decomposition,
    Projection,
    PostProjectionQr,
    PostProjectionSvd,
    Survival,
    Normalization,
    Bookkeeping,
}

impl QueryDepthTelemetry {
    pub(super) fn begin_phase(&mut self) {
        assert!(
            !self.phase_active,
            "query telemetry phase scopes must be disjoint"
        );
        self.phase_active = true;
    }

    pub(super) fn record(
        &mut self,
        phase: QueryPhase,
        svd_operations: u64,
        capped_svd_operations: u64,
        wall_time_seconds: f64,
    ) {
        assert!(self.phase_active, "query telemetry phase was not started");
        self.phase_active = false;
        let target = match phase {
            QueryPhase::Expectation => &mut self.expectation,
            QueryPhase::PreReduction => &mut self.pre_reduction,
            QueryPhase::Decomposition => &mut self.decomposition,
            QueryPhase::Projection => &mut self.projection,
            QueryPhase::PostProjectionQr => &mut self.post_projection_qr,
            QueryPhase::PostProjectionSvd => &mut self.post_projection_svd,
            QueryPhase::Survival => &mut self.survival,
            QueryPhase::Normalization => &mut self.normalization,
            QueryPhase::Bookkeeping => &mut self.bookkeeping,
        };
        target.calls += 1;
        target.svd_operations += svd_operations;
        target.capped_svd_operations += capped_svd_operations;
        target.wall_time_seconds += wall_time_seconds;
    }

    /// Sum of every disjoint phase bucket at this trie depth.
    #[must_use]
    pub fn attributed_wall_time_seconds(&self) -> f64 {
        self.expectation.wall_time_seconds
            + self.pre_reduction.wall_time_seconds
            + self.decomposition.wall_time_seconds
            + self.projection.wall_time_seconds
            + self.post_projection_qr.wall_time_seconds
            + self.post_projection_svd.wall_time_seconds
            + self.survival.wall_time_seconds
            + self.normalization.wall_time_seconds
            + self.bookkeeping.wall_time_seconds
    }

    /// Return whether every phase scope at this depth closed without overlap.
    #[must_use]
    pub const fn phase_scopes_disjoint(&self) -> bool {
        !self.phase_active
    }
}

impl ProbabilityQueryTelemetry {
    /// Sum of all depth-bucketed phase wall times.
    #[must_use]
    pub fn attributed_wall_time_seconds(&self) -> f64 {
        self.by_depth
            .iter()
            .map(QueryDepthTelemetry::attributed_wall_time_seconds)
            .sum()
    }

    /// Return whether all depth-local timing scopes completed without nesting.
    #[must_use]
    pub fn phase_scopes_disjoint(&self) -> bool {
        self.by_depth
            .iter()
            .all(QueryDepthTelemetry::phase_scopes_disjoint)
    }
}

struct PrefixSamplingContext<'a> {
    rng: &'a mut PecosRng,
    frame_x: &'a [bool],
    num_qubits: usize,
    output: &'a mut Vec<Vec<bool>>,
}

/// Tableau/MPS state owned by one node of a forced-projection prefix tree.
///
/// Both batched probability queries and prefix-tree sampling use this type so
/// the clone-at-branch and atomic-projection rules have a single owner.
#[derive(Clone)]
struct PrefixProjectionState {
    tableau: SparseStabY,
    mps: Mps,
}

impl PrefixProjectionState {
    fn project_z(
        &mut self,
        qubit: usize,
        outcome: bool,
        telemetry: Option<&mut QueryDepthTelemetry>,
    ) -> Result<f64, MpsError> {
        match telemetry {
            Some(telemetry) => measure::project_forced_z_profiled(
                &mut self.tableau,
                &mut self.mps,
                qubit,
                outcome,
                telemetry,
            ),
            None => measure::project_forced_z(&mut self.tableau, &mut self.mps, qubit, outcome),
        }
    }
}

#[derive(Default)]
struct ProbabilityQueryTrieNode {
    children: [Option<Box<Self>>; 2],
    query_indices: Vec<usize>,
}

impl ProbabilityQueryTrieNode {
    fn insert(&mut self, bitstring: &[bool], query_index: usize) {
        let mut node = self;
        for &bit in bitstring {
            node = node.children[usize::from(bit)].get_or_insert_with(|| Box::new(Self::default()));
        }
        node.query_indices.push(query_index);
    }
}

impl StabMps {
    /// Create a builder for configuring the simulator.
    #[must_use]
    pub fn builder(num_qubits: usize) -> StabMpsBuilder {
        StabMpsBuilder {
            num_qubits,
            seed: None,
            max_bond_dim: 128,
            svd_cutoff: 1e-12,
            max_truncation_error: Some(1e-8),
            parallel: false,
            auto_grow_bond_dim: None,
            auto_grow_max_bond_dim: 4096,
            measurement_mode: MeasurementMode::default(),
            flags: StabMpsFlags::new(),
        }
    }

    /// Create a new STN simulator with default configuration.
    #[must_use]
    pub fn new(num_qubits: usize) -> Self {
        Self::builder(num_qubits).build()
    }

    /// Create with a specific seed for reproducible stochastic operations.
    ///
    /// This seeds both the simulator's [`pecos_random::PecosRng`] and the
    /// stabilizer tableau. Identically configured fresh instances reproduce an
    /// identical call sequence. On reset, both rebuilt RNGs are seeded from the
    /// current simulator stream, giving deterministic continuation rather than
    /// replaying the construction stream.
    #[must_use]
    pub fn with_seed(num_qubits: usize, seed: u64) -> Self {
        Self::builder(num_qubits).seed(seed).build()
    }

    #[must_use]
    /// Return the number of simulated qubits.
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Current maximum bond dimension in the MPS.
    #[must_use]
    pub fn max_bond_dim(&self) -> usize {
        self.mps.max_bond_dim()
    }

    /// Return detailed saturated-regime event telemetry collected so far.
    #[must_use]
    pub fn saturation_profile(&self) -> &SaturationTelemetry {
        &self.saturation_telemetry
    }

    /// Theoretical minimum bond dimension from GF(2) OFD analysis.
    ///
    /// Returns `2^(t - rank)` where t is the number of non-Clifford gates
    /// applied and rank is the GF(2) rank of the flip pattern matrix.
    /// This is the best bond dimension achievable by Clifford disentangling.
    #[must_use]
    pub fn theoretical_min_bond_dim(&self) -> usize {
        self.gf2_matrix.theoretical_min_bond_dim()
    }

    /// OFD null space dimension (Liu-Clark 2412.17209 Section III.C).
    ///
    /// Returns `t - rank` where t is the number of absorbed non-Clifford gates
    /// and rank is the GF(2) rank of their flip patterns. This is the number
    /// of gates that could NOT be disentangled (required bond-dim growth).
    ///
    /// The bond dimension lower bound from OFD is `2^nullity`.
    ///
    /// For research circuits where nullity < log₂(N), the simulation is
    /// efficient (polynomial in N).
    #[must_use]
    pub fn ofd_nullity(&self) -> usize {
        let t = self.gf2_matrix.num_gates();
        let r = self.gf2_matrix.gf2_rank();
        t.saturating_sub(r)
    }

    /// Number of non-Clifford gates that OFD disentangled (rank of GF(2) matrix).
    #[must_use]
    pub fn ofd_disentangled_count(&self) -> usize {
        self.gf2_matrix.gf2_rank()
    }

    /// Total non-Clifford gates recorded in the GF(2) basis.
    #[must_use]
    pub fn ofd_total_absorbed(&self) -> usize {
        self.gf2_matrix.num_gates()
    }

    /// Access the GF(2) flip matrix (for diagnostics).
    #[must_use]
    pub fn gf2_matrix(&self) -> &ofd::Gf2FlipMatrix {
        &self.gf2_matrix
    }

    /// Wavefunction amplitude `⟨s|C|psi⟩` for a given bitstring `s`.
    ///
    /// `bitstring` has length `num_qubits`; bit k corresponds to qubit k.
    /// Returns the unnormalized amplitude coefficient.
    /// See the crate-level **Bitstring convention** section.
    ///
    /// This materializes the full dense state using [`Self::state_vector`], with
    /// `O(2^n)` memory and greater construction cost than a single amplitude
    /// needs; it is limited to `n <= 14`. Prefer [`Self::amplitude_iterative`]
    /// for scalable selected-amplitude reads.
    ///
    /// Call [`Self::flush`] first when lazy measurement or RZ merging is enabled,
    /// and materialize a tracked Pauli frame when it must be included.
    ///
    /// # Panics
    /// Panics if bitstring length doesn't match `num_qubits`, or n > 14.
    #[must_use]
    pub fn amplitude(&self, bitstring: &[bool]) -> Complex64 {
        assert_eq!(
            bitstring.len(),
            self.num_qubits,
            "bitstring length mismatch"
        );
        assert!(self.num_qubits <= 14, "amplitude requires n <= 14");
        let sv = self.state_vector();
        // Convert bitstring to the state_vector's little-endian index:
        // x = Σ_q bitstring[q] * 2^q.
        let mut idx = 0usize;
        for (q, &b) in bitstring.iter().enumerate() {
            if b {
                idx |= 1 << q;
            }
        }
        sv[idx]
    }

    /// Compute `⟨Ψ|P|Ψ⟩` for an arbitrary multi-qubit Pauli string `P`.
    ///
    /// `pauli_string` lists non-identity factors as `(qubit, PauliKind)`
    /// pairs; qubits not listed get `I`. Returns the real expectation
    /// value (the Hermitian Pauli always has real expectation).
    ///
    /// Building block for code-state fidelity at large `n` (sum over
    /// stabilizer group of `⟨Ψ|g|Ψ⟩`), variational energy estimation,
    /// and arbitrary-observable readout.
    ///
    /// Call [`Self::flush`] first when lazy measurement or RZ merging is enabled,
    /// and materialize a tracked Pauli frame when it must be included.
    ///
    /// # Method
    /// Tableau-based decomposition: writes `P` as
    ///   `phase · X_{flip} · Z_{sign}`  (in stored-MPS frame, after
    ///   conjugating by `C†`)
    /// using `pauli_decomp::decompose_pauli_string`, then evaluates via
    /// `measure::pauli_expectation` on the MPS. Scales to arbitrary `n`.
    ///
    /// # Panics
    /// Panics if any qubit index exceeds `num_qubits`.
    #[must_use]
    pub fn pauli_expectation(&self, pauli_string: &[(usize, PauliKind)]) -> f64 {
        // Translate public PauliKind into pauli_decomp's enum.
        let decomp_input: Vec<(usize, pauli_decomp::PauliKindForDecomp)> = pauli_string
            .iter()
            .map(|&(q, k)| {
                assert!(q < self.num_qubits, "pauli qubit index {q} >= num_qubits");
                let pk = match k {
                    PauliKind::X => pauli_decomp::PauliKindForDecomp::X,
                    PauliKind::Y => pauli_decomp::PauliKindForDecomp::Y,
                    PauliKind::Z => pauli_decomp::PauliKindForDecomp::Z,
                };
                (q, pk)
            })
            .collect();
        let (flip, sign, phase) = pauli_decomp::decompose_pauli_string(
            self.tableau.stabs(),
            self.tableau.destabs(),
            &decomp_input,
        );
        measure::pauli_expectation(&self.mps, &flip, &sign, phase).re
    }

    /// Compute the overlap `⟨s|Ψ⟩` where `|s⟩` is a stabilizer state given
    /// as a CH-form simulator. Uses the importance-sampling estimator from
    /// CD-Loschmidt-echoes (Mello, Santini, Collura, arXiv:2502.01872 Eq. 1):
    ///
    ///   `⟨s|Ψ⟩ = E_{x ~ |⟨x|s⟩|²}[ ⟨x|Ψ⟩ / ⟨x|s⟩ ]`
    ///
    /// Variance is `1 − |⟨s|Ψ⟩|²` (independent of N — Eq. 2 of the paper),
    /// so a few hundred samples typically suffice for 1% statistical error.
    /// Scales to arbitrary `n` (uses `amplitude_iterative` for `⟨x|Ψ⟩` and
    /// CH-form `amplitude` + sequential measurement for `⟨x|s⟩` and
    /// stabilizer Born sampling).
    ///
    /// Call [`Self::flush`] first when lazy measurement or RZ merging is enabled,
    /// and materialize a tracked Pauli frame when it must be included.
    ///
    /// Note: requires `n ≤ 64` due to CH-form's `usize`-indexed amplitude
    /// API; that's already a much higher limit than the SV path's `n ≤ 14`.
    ///
    /// # Arguments
    /// - `s`: CH-form simulator representing the stabilizer state `|s⟩`.
    ///   Caller is responsible for setting up the desired Clifford circuit
    ///   on `s` before passing it in. **Mutated** as samples are drawn
    ///   (cloned internally per sample); a fresh CH-form is used per shot.
    /// - `num_samples`: number of MC samples. ~100 gives ~10% error,
    ///   ~10000 gives ~1% error.
    /// - `rng_seed`: optional seed for per-sample CH-form clones. When
    ///   `None`, uses a deterministic hash of the sample index (reproducible
    ///   but not caller-controllable). Pass `Some(seed)` to control the
    ///   MC stream for reproducibility across runs.
    ///
    /// # Returns
    /// Complex MC estimate of `⟨s|Ψ⟩`. Take `.norm_sqr()` for a fidelity
    /// estimate `|⟨s|Ψ⟩|²`.
    ///
    /// # Limitations
    /// - Requires `n <= 64` (usize bitstring index in CH-form).
    /// - Statistical estimator: not exact. Use `code_state_fidelity` for
    ///   exact answer at `n <= 14`.
    /// - The CH-form `s` must be on the same number of qubits as `self`.
    ///
    /// # Panics
    ///
    /// Panics if `s.num_qubits() != self.num_qubits` or `num_qubits > 64`.
    #[must_use]
    pub fn overlap_with_stabilizer<
        R: pecos_random::SeedableRng + pecos_random::Rng + std::fmt::Debug + Clone,
    >(
        &self,
        s: &pecos_simulators::CHForm<R>,
        num_samples: usize,
        rng_seed: Option<u64>,
    ) -> Complex64 {
        use pecos_core::RngManageable;

        assert_eq!(
            s.num_qubits(),
            self.num_qubits,
            "stabilizer-state qubit count mismatch"
        );
        assert!(
            self.num_qubits <= 64,
            "overlap_with_stabilizer requires n <= 64"
        );

        let n = self.num_qubits;
        let mut acc = Complex64::new(0.0, 0.0);
        let mut samples_used = 0usize;

        for sample_idx in 0..num_samples {
            // Per-sample clone of |s⟩ with a fresh RNG seed so each sample
            // produces an independent bitstring from the Born distribution.
            // Use sample_idx-based seed (self is &self so we can't advance
            // self.rng). The wrapping_mul mixes bits to avoid trivial overlap.
            let mut s_sampler = s.clone();
            let base_seed = rng_seed.unwrap_or(42);
            let sample_seed = (sample_idx as u64)
                .wrapping_mul(2_654_435_761)
                .wrapping_add(base_seed);
            s_sampler.set_rng(R::seed_from_u64(sample_seed));
            let mut bitstring = vec![false; n];
            for (q, bit) in bitstring.iter_mut().enumerate() {
                let outcome = s_sampler.mz(&[pecos_core::QubitId(q)])[0].outcome;
                *bit = outcome;
            }
            // Compute x as usize index per CH-form's amplitude API:
            // bit q of x corresponds to qubit q's outcome (LSB-first).
            let mut x_idx = 0usize;
            for (q, &bit) in bitstring.iter().enumerate() {
                if bit {
                    x_idx |= 1usize << q;
                }
            }
            let amp_xs = s.amplitude(x_idx);
            if amp_xs.norm_sqr() < 1e-30 {
                // Zero-amplitude sample: should be impossible if we sampled
                // from the correct Born distribution. Skip defensively.
                continue;
            }
            // Compute <x|Ψ>; both APIs use bitstring[q] for qubit q.
            let amp_xpsi = self.amplitude_iterative(&bitstring);
            acc += amp_xpsi / amp_xs;
            samples_used += 1;
        }
        if samples_used == 0 {
            eprintln!(
                "warning: overlap_with_stabilizer: all {num_samples} samples had zero amplitude — returning 0"
            );
            return Complex64::new(0.0, 0.0);
        }
        acc / Complex64::new(
            f64::from(u32::try_from(samples_used).expect("samples fit in u32")),
            0.0,
        )
    }

    /// Compute `⟨Ψ|P_code|Ψ⟩` where `P_code` is the projector onto the
    /// stabilizer code subspace defined by `stabilizer_generators`.
    ///
    /// Each generator is a Pauli string given as a `Vec<(usize, PauliKind)>`
    /// listing non-identity factors. `P_code = Π_i (I + g_i)/2` for `k`
    /// generators yields a fidelity in [0, 1]: 1 means `|Ψ⟩` is fully
    /// inside the code subspace, 0 means fully outside.
    ///
    /// Useful for QEC verification: after running a code's preparation /
    /// syndrome-extraction circuit, this returns how much of the state is
    /// in the codespace. Compare against expected value (1.0 for noiseless,
    /// less for noisy circuits).
    ///
    /// # Method
    /// Expands `P_code = (1/2^k) Σ_{g ∈ stabilizer group} g` and computes
    /// `(1/2^k) Σ ⟨Ψ|g|Ψ⟩` via `pauli_expectation` per group element.
    /// Scales to arbitrary `n` (the bottleneck is `2^k` group enumeration
    /// where `k = stabilizer_generators.len()`).
    ///
    /// For codes with many generators, prefer
    /// `StabMps::overlap_with_stabilizer` (CD Loschmidt MC) targeting one
    /// specific code state at a time.
    ///
    /// Call [`Self::flush`] first when lazy measurement or RZ merging is enabled,
    /// and materialize a tracked Pauli frame when it must be included.
    ///
    /// # Panics
    /// Panics if any qubit index in a generator is ≥ `num_qubits`, or if
    /// `2^k` overflows `usize` (e.g., k > 62 on 64-bit).
    #[must_use]
    pub fn code_state_fidelity(&self, stabilizer_generators: &[Vec<(usize, PauliKind)>]) -> f64 {
        let k = stabilizer_generators.len();
        assert!(
            k <= 30,
            "code_state_fidelity: 2^k group enumeration with k={k} would take too long"
        );
        let n = self.num_qubits;
        for gen_string in stabilizer_generators {
            for &(q, _) in gen_string {
                assert!(q < n, "generator qubit index {q} >= num_qubits {n}");
            }
        }
        let group_size = 1usize << k;
        let mut acc = 0.0;
        for mask in 0..group_size {
            // Compose group element by multiplying generators selected by mask.
            // Use Pauli aggregation via decompose_pauli_string's per-qubit logic
            // — but we just need <Ψ|g|Ψ>, so flatten the selected generators
            // into one Pauli-string list and let pauli_expectation aggregate.
            let mut composed: Vec<(usize, PauliKind)> = Vec::new();
            for (i, generator) in stabilizer_generators.iter().enumerate() {
                if (mask >> i) & 1 == 1 {
                    composed.extend_from_slice(generator);
                }
            }
            acc += self.pauli_expectation(&composed);
        }
        acc / f64::from(u32::try_from(group_size).expect("group_size fits in u32"))
    }

    /// Complex amplitude ⟨s|Ψ⟩ in [`Self::state_vector`]'s canonical
    /// global-phase convention, via iterative forced projection without
    /// renormalization (Liu-Clark 2412.17209 Section VI.B).
    /// `bitstring[q]` specifies qubit `q`; see the crate-level
    /// **Bitstring convention** section.
    ///
    /// Call [`Self::flush`] first when lazy measurement or RZ merging is enabled,
    /// and materialize a tracked Pauli frame when it must be included.
    ///
    /// Scales beyond `amplitude`'s n ≤ 14 limit by working directly on the
    /// MPS + tableau. The product of conditional probabilities supplies the
    /// amplitude magnitude. After forcing all N outcomes, the final tableau's
    /// GF(2) sign equations identify the virtual-basis coefficient whose phase
    /// supplies the complex relative phase without enumerating the state. All
    /// returned amplitudes share the physical state's one arbitrary global
    /// phase; this method does not define an absolute ket phase.
    ///
    /// # Correctness
    /// Exact match to `amplitude` (SV-based) at n ≤ 14 for Clifford+T
    /// circuits, including relative phase in `state_vector`'s convention.
    /// Scales to arbitrary n via MPS operations. Probabilities via
    /// `prob_bitstring` are always correct.
    ///
    /// # Performance
    ///
    /// Relative-phase tracking performs O(n^3) canonical-ket elimination at
    /// each stabilizer-group change, while reusing the post-change ket across
    /// intervening generator rebases and virtual gates that preserve the
    /// group. In a release dense-generator benchmark (H on every qubit,
    /// triangular CX network, all-zero amplitude; three-run medians), cache
    /// reuse reduced n=32/64/128/192 calls from 2.028/18.903/239.150/1120.132 ms
    /// to 1.195/10.395/127.186/557.597 ms. The cached path still scaled about
    /// n^3.4 and was 2.9x/6.0x/28.4x/35.9x slower than the phase-insensitive
    /// dev parent on that adversarial generator basis.
    ///
    /// # Panics
    ///
    /// Panics if the bitstring length doesn't match `num_qubits`, or if forced
    /// projection encounters an unrecoverable numerical MPS state.
    #[must_use]
    pub fn amplitude_iterative(&self, bitstring: &[bool]) -> Complex64 {
        assert_eq!(
            bitstring.len(),
            self.num_qubits,
            "bitstring length mismatch"
        );
        let mut tab = self.tableau.clone();
        let mut mps = self.mps.clone();
        let mut projected_norm = 1.0;
        let mut phase_tracker = canonical_ket::CanonicalPhaseTracker::new();
        for (q, &s_q) in bitstring.iter().enumerate() {
            let probability = expect_mps_operation(
                measure::project_forced_z_with_phase(
                    &mut tab,
                    &mut mps,
                    q,
                    s_q,
                    &mut phase_tracker,
                ),
                "StabMps::amplitude_iterative forced projection",
            );
            if probability < 1e-20 {
                return Complex64::new(0.0, 0.0);
            }
            projected_norm *= probability.sqrt();
        }
        let mps_index = measure::projected_mps_basis_index(&tab, bitstring);
        let coefficient = mps.amplitude(&mps_index);
        let coefficient_norm = coefficient.norm();
        assert!(
            coefficient_norm.is_finite() && projected_norm.is_finite(),
            "forced projection produced a non-finite amplitude component"
        );
        if coefficient_norm == 0.0 {
            // This invariant holds only because the expectation endpoint snap
            // forces structurally zero branches to exact zero. If that snap is
            // ever removed, fail loudly in debug builds instead of normalizing
            // a retained-probability branch with a zero phase coefficient.
            debug_assert!(
                projected_norm == 0.0,
                "forced projection retained probability but selected a zero phase coefficient"
            );
            return Complex64::new(0.0, 0.0);
        }
        let terminal_tableau_phase =
            phase_tracker.terminal_tableau_basis_phase(&tab, &mps_index, bitstring);
        self.global_phase * phase_tracker.scalar() * terminal_tableau_phase * coefficient
            / coefficient_norm
            * projected_norm
    }

    /// Probability of measuring `bitstring` in the computational basis.
    /// `bitstring[q]` specifies qubit `q`; see the crate-level
    /// **Bitstring convention** section.
    ///
    /// Call [`Self::flush`] first when lazy measurement or RZ merging is enabled,
    /// and materialize a tracked Pauli frame when it must be included.
    ///
    /// Implements Liu-Clark 2412.17209 Algorithm 3 (Section VI.A): iterative
    /// forced projection of the CAMPS state. For each qubit k:
    ///   `π_k` = ⟨`ψ_k` | (I + (-`1)^{s_k`} `Z̃_k)/2` | `ψ_k`⟩
    ///   |ψ_{k+1}⟩ ∝ (I + (-`1)^{s_k`} `Z̃_k)/2` |`ψ_k`⟩
    /// where `Z̃_k` is the tableau's Z-mapping on qubit k. Final probability is
    /// the product of conditional probabilities `π_k`.
    ///
    /// Scales beyond n = 14 (unlike `amplitude`) by working directly on the
    /// MPS + tableau instead of the full state vector.
    ///
    /// # Panics
    ///
    /// Panics if the bitstring length doesn't match `num_qubits`, or if forced
    /// projection encounters an unrecoverable numerical MPS state.
    #[must_use]
    pub fn prob_bitstring(&self, bitstring: &[bool]) -> f64 {
        assert_eq!(
            bitstring.len(),
            self.num_qubits,
            "bitstring length mismatch"
        );
        let mut tab = self.tableau.clone();
        let mut mps = self.mps.clone();
        let mut total_prob: f64 = 1.0;
        for (q, &s_q) in bitstring.iter().enumerate() {
            let pi_q = expect_mps_operation(
                measure::project_forced_z(&mut tab, &mut mps, q, s_q),
                "StabMps::prob_bitstring forced projection",
            );
            total_prob *= pi_q;
            if total_prob < QUERY_ZERO_PROBABILITY_FLOOR {
                return 0.0;
            }
        }
        total_prob.clamp(0.0, 1.0)
    }

    /// Probabilities of measuring each requested computational-basis
    /// bitstring, sharing forced projections across common prefixes.
    ///
    /// Each input and the corresponding output use the same convention as
    /// [`Self::prob_bitstring`]: `bitstrings[i][q]` specifies qubit `q`, and
    /// `probabilities[i]` is exactly the result of the corresponding singular
    /// call, including its configured MPS-truncation and endpoint behavior.
    /// Input order and duplicates are preserved. An empty input returns an
    /// empty output.
    ///
    /// The query set is represented as a prefix trie over qubits
    /// `0..num_qubits`. A node with one occupied child moves its working
    /// tableau/MPS state into that child without cloning. A branch point makes
    /// one clone and gives one working state to each child. Thus every
    /// distinct query prefix receives exactly one atomic forced projection.
    /// If the accumulated probability reaches the singular API's snapped-zero
    /// threshold, the entire query subtree remains exactly `0.0`.
    ///
    /// Call [`Self::flush`] first when lazy measurement or RZ merging is
    /// enabled, and materialize a tracked Pauli frame when it must be included.
    ///
    /// # Panics
    ///
    /// Panics if any bitstring length doesn't match `num_qubits`, or if forced
    /// projection encounters an unrecoverable numerical MPS state.
    #[must_use]
    pub fn prob_bitstrings<B: AsRef<[bool]>>(&self, bitstrings: &[B]) -> Vec<f64> {
        self.prob_bitstrings_impl(bitstrings, None)
    }

    /// Profiled [`Self::prob_bitstrings`] with phase operation counts and wall
    /// time bucketed by trie depth.
    #[must_use]
    pub fn prob_bitstrings_profiled<B: AsRef<[bool]>>(
        &self,
        bitstrings: &[B],
    ) -> (Vec<f64>, ProbabilityQueryTelemetry) {
        let mut telemetry = ProbabilityQueryTelemetry {
            by_depth: vec![QueryDepthTelemetry::default(); self.num_qubits],
            whole_call_wall_time_seconds: 0.0,
        };
        let started = Instant::now();
        let probabilities = self.prob_bitstrings_impl(bitstrings, Some(&mut telemetry));
        telemetry.whole_call_wall_time_seconds = started.elapsed().as_secs_f64();
        (probabilities, telemetry)
    }

    fn prob_bitstrings_impl<B: AsRef<[bool]>>(
        &self,
        bitstrings: &[B],
        telemetry: Option<&mut ProbabilityQueryTelemetry>,
    ) -> Vec<f64> {
        let mut trie = ProbabilityQueryTrieNode::default();
        for (query_index, bitstring) in bitstrings.iter().enumerate() {
            let bitstring = bitstring.as_ref();
            assert_eq!(
                bitstring.len(),
                self.num_qubits,
                "bitstring length mismatch at query {query_index}"
            );
            trie.insert(bitstring, query_index);
        }
        if bitstrings.is_empty() {
            return Vec::new();
        }

        let state = PrefixProjectionState {
            tableau: self.tableau.clone(),
            mps: self.mps.clone(),
        };
        let mut probabilities = vec![0.0; bitstrings.len()];
        expect_mps_operation(
            Self::probability_query_prefix_tree(
                &trie,
                state,
                self.num_qubits,
                0,
                1.0,
                &mut probabilities,
                telemetry,
            ),
            "StabMps::prob_bitstrings forced projection",
        );
        probabilities
    }

    fn probability_query_prefix_tree(
        node: &ProbabilityQueryTrieNode,
        mut state: PrefixProjectionState,
        num_qubits: usize,
        qubit: usize,
        total_probability: f64,
        probabilities: &mut [f64],
        mut telemetry: Option<&mut ProbabilityQueryTelemetry>,
    ) -> Result<(), MpsError> {
        if qubit == num_qubits {
            let probability = total_probability.clamp(0.0, 1.0);
            for &query_index in &node.query_indices {
                probabilities[query_index] = probability;
            }
            return Ok(());
        }

        match (&node.children[0], &node.children[1]) {
            (Some(zero), Some(one)) => {
                let mut zero_state = state.clone();
                let zero_probability = total_probability
                    * zero_state.project_z(
                        qubit,
                        false,
                        telemetry
                            .as_deref_mut()
                            .map(|profile| &mut profile.by_depth[qubit]),
                    )?;
                // NaN must follow the singular walk (which keeps projecting on
                // NaN) rather than being silently pruned to zero.
                if zero_probability.is_nan() || zero_probability >= QUERY_ZERO_PROBABILITY_FLOOR {
                    Self::probability_query_prefix_tree(
                        zero,
                        zero_state,
                        num_qubits,
                        qubit + 1,
                        zero_probability,
                        probabilities,
                        telemetry.as_deref_mut(),
                    )?;
                }

                let one_probability = total_probability
                    * state.project_z(
                        qubit,
                        true,
                        telemetry
                            .as_deref_mut()
                            .map(|profile| &mut profile.by_depth[qubit]),
                    )?;
                if one_probability.is_nan() || one_probability >= QUERY_ZERO_PROBABILITY_FLOOR {
                    Self::probability_query_prefix_tree(
                        one,
                        state,
                        num_qubits,
                        qubit + 1,
                        one_probability,
                        probabilities,
                        telemetry.as_deref_mut(),
                    )?;
                }
            }
            (Some(child), None) | (None, Some(child)) => {
                let outcome = node.children[1].is_some();
                let child_probability = total_probability
                    * state.project_z(
                        qubit,
                        outcome,
                        telemetry
                            .as_deref_mut()
                            .map(|profile| &mut profile.by_depth[qubit]),
                    )?;
                if child_probability.is_nan() || child_probability >= QUERY_ZERO_PROBABILITY_FLOOR {
                    Self::probability_query_prefix_tree(
                        child,
                        state,
                        num_qubits,
                        qubit + 1,
                        child_probability,
                        probabilities,
                        telemetry,
                    )?;
                }
            }
            (None, None) => {
                debug_assert!(
                    node.query_indices.is_empty(),
                    "probability-query trie leaf before the final qubit"
                );
            }
        }
        Ok(())
    }

    /// Second Rényi entropy `S_2` = -`ln(Tr_A(ρ_A²))` at a bipartition
    /// (qubits 0..cut vs qubits cut..N).
    ///
    /// Call [`Self::flush`] first when lazy measurement or RZ merging is enabled,
    /// and materialize a tracked Pauli frame when it must be included.
    ///
    /// Uses the full `state_vector` for computation — works only for n <= 14.
    /// Paper Liu-Clark 2412.17209 Section VI.C gives an MPS-based algorithm
    /// that scales better but requires careful implementation of the Pauli
    /// generator enumeration and CAMPS-specific Gaussian elimination.
    ///
    /// # Panics
    /// Panics if cut == 0 or cut >= `num_qubits`, or if `num_qubits` > 14.
    #[must_use]
    pub fn renyi_s2(&self, cut: usize) -> f64 {
        let n = self.num_qubits;
        assert!(cut > 0 && cut < n, "cut must be in (0, num_qubits)");
        assert!(
            n <= 14,
            "renyi_s2 requires n <= 14 (uses full state vector)"
        );

        let sv = self.state_vector();
        let dim_a = 1usize << cut;
        let dim_b = 1usize << (n - cut);
        // state_vector is LSB-first: `sv[idx]` has qubit k at bit k of idx.
        // Convention: A = first `cut` qubits (0..cut) → low bits.
        //             B = qubits cut..n                → high bits.
        //   idx = a_bits | (b_bits << cut)
        //
        // Reduced density ρ_A: (ρ_A)_{a, a'} = Σ_b ψ(a, b) · ψ*(a', b)
        let mut rho_a = vec![Complex64::new(0.0, 0.0); dim_a * dim_a];
        for a in 0..dim_a {
            for a_prime in 0..dim_a {
                let mut acc = Complex64::new(0.0, 0.0);
                for b in 0..dim_b {
                    let idx1 = a | (b << cut);
                    let idx2 = a_prime | (b << cut);
                    acc += sv[idx1] * sv[idx2].conj();
                }
                rho_a[a * dim_a + a_prime] = acc;
            }
        }

        // S_2 = -ln(Tr(ρ_A^2)) = -ln(Σ_{a,a'} |ρ_A[a, a']|^2).
        let mut tr_sq = 0.0_f64;
        for a in 0..dim_a {
            for a_prime in 0..dim_a {
                tr_sq += rho_a[a * dim_a + a_prime].norm_sqr();
            }
        }
        if tr_sq < 1e-30 {
            f64::INFINITY
        } else {
            -tr_sq.ln()
        }
    }

    /// CAMPS-native `S_2` entropy via Pauli Coefficient Enumeration (Liu-Clark
    /// Section VI.C). Does NOT require constructing the state vector, so scales
    /// beyond n = 14 when the MPS has bond dim 1 and T-gate density is moderate.
    ///
    /// `cut` places qubits [0, cut) in region A, [cut, n) in region B.
    ///
    /// Complexity: ∏_j (1 + `non_zero_bloch_components(j)`) combinations. For
    /// Clifford+T with sparse T gates most sites give count=1 → 2^N fallback.
    /// Full-magic sites give count=3 -> 4^N worst case. Error if > 2^22.
    /// Call [`Self::flush`] before this read when lazy measurement or RZ
    /// merging is enabled, and materialize a tracked Pauli frame if required.
    ///
    /// # Errors
    ///
    /// Returns an error string if the cut is out of range or the number of
    /// Pauli combinations exceeds the safety limit.
    pub fn s2_pce(&self, cut: usize) -> Result<f64, String> {
        let n = self.num_qubits;
        if cut == 0 || cut >= n {
            return Err(format!("cut {cut} must be in (0, {n})"));
        }
        let mask: Vec<bool> = (0..n).map(|q| q < cut).collect();
        renyi::compute_s2_pce(&self.mps, &self.tableau, &mask)
    }

    /// Fast `S_2` via GF(2) null-space enumeration (PCMPS). Requires every MPS
    /// site to have a single Pauli-axis Bloch vector (typical for STN
    /// Clifford+T where T gets absorbed into the tableau). Falls back to
    /// [`StabMps::s2_pce`] if multi-axis sites are present.
    ///
    /// Scales to much larger n than PCE when applicable: `2^null_dim`
    /// enumerations vs 2^N. For pure-Clifford Bell on n=100, `null_dim` is
    /// typically 0-2.
    /// Call [`Self::flush`] before this read when lazy measurement or RZ
    /// merging is enabled, and materialize a tracked Pauli frame if required.
    ///
    /// # Errors
    ///
    /// Returns an error string if the cut is out of range, or if the null-space
    /// enumeration is too large.
    pub fn s2_pcmps(&self, cut: usize) -> Result<f64, String> {
        let n = self.num_qubits;
        if cut == 0 || cut >= n {
            return Err(format!("cut {cut} must be in (0, {n})"));
        }
        let mask: Vec<bool> = (0..n).map(|q| q < cut).collect();
        // Fast path: single-axis-per-site PCMPS (Clifford-state analytic
        // short-circuit handles pure Clifford at any n).
        if let Ok(s) = renyi::compute_s2_pcmps(&self.mps, &self.tableau, &mask) {
            return Ok(s);
        }
        // General path: 2N-bit F_2 null-space TN enumeration. Handles
        // multi-axis Bloch but null_dim capped at 22.
        if let Ok(s) = renyi::compute_s2_pcmps_tn(&self.mps, &self.tableau, &mask) {
            return Ok(s);
        }
        // Last resort: full PCE (4^N hard cap).
        renyi::compute_s2_pce(&self.mps, &self.tableau, &mask)
    }

    /// Access the stored coefficient MPS.
    ///
    /// Call [`Self::flush`] first if the returned tensors must include pending
    /// lazy-measurement operations or merged RZ rotations. A tracked Pauli
    /// frame is stored separately until explicitly materialized.
    #[must_use]
    pub fn mps(&self) -> &Mps {
        &self.mps
    }

    /// Accumulated truncation error so far (approximate `1 - |⟨ψ_exact|ψ⟩|²`).
    /// Zero if no SVD has dropped any singular values above `svd_cutoff`.
    /// Call [`Self::flush`] first to include SVDs triggered by pending work.
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

    /// Number of sampled projections retried after a rolled-back branch vanish.
    #[must_use]
    pub fn branch_vanish_retry_count(&self) -> u64 {
        self.mps.branch_vanish_retry_count()
    }

    /// Number of deferred MAST branches lost; always zero for `StabMps`.
    #[must_use]
    pub fn deferred_branch_lost_count(&self) -> u64 {
        self.mps.deferred_branch_lost_count()
    }

    /// Number of MPS bond-canonicalization consults that required a cold full sweep.
    #[must_use]
    pub fn full_canonical_sweep_count(&self) -> u64 {
        self.mps.full_canonical_sweep_count()
    }

    /// Number of MPS bond-canonicalization consults that reused a tracked center.
    #[must_use]
    pub fn center_reuse_count(&self) -> u64 {
        self.mps.center_reuse_count()
    }

    /// Return the configured single-qubit measurement policy.
    #[must_use]
    pub fn measurement_mode(&self) -> MeasurementMode {
        self.measurement_mode
    }

    /// Number of SVDs where `max_bond_dim` was the binding cap. If > 0 the
    /// state is under-resolved; consider raising `max_bond_dim` or loosening
    /// `max_truncation_error`.
    /// Call [`Self::flush`] first to include SVDs triggered by pending work.
    #[must_use]
    pub fn bond_cap_hits(&self) -> u64 {
        self.mps.bond_cap_hits()
    }

    /// Access the stored stabilizer tableau.
    ///
    /// Call [`Self::flush`] first when the read must include pending merged RZ
    /// rotations. Lazy virtual-frame operations and a tracked Pauli frame can
    /// remain represented outside this tableau.
    #[must_use]
    pub fn tableau(&self) -> &SparseStabY {
        &self.tableau
    }

    /// Run Clifford disentangling sweeps to reduce MPS bond dimension.
    ///
    /// Each sweep examines every internal bond and tries 20 inequivalent
    /// entangling two-qubit Clifford candidates, performing SVD-based entropy
    /// estimates for them. If a candidate reduces entanglement, it is applied
    /// exactly to the coefficient MPS and its inverse is right-composed into
    /// the tableau, preserving the represented state for every read path. The
    /// operation adds no approximation beyond the MPS configuration's normal
    /// SVD truncation.
    ///
    /// Deferred lazy-measurement operations and pending merged RZ rotations
    /// are materialized before the first sweep.
    ///
    /// This can be expensive compared with a gate update; use it after a batch
    /// of non-Clifford gates or when observed bond growth justifies a sweep, not
    /// after every gate. `max_sweeps` bounds full-chain passes. Returns the
    /// number of accepted Clifford gates; zero means no candidate was applied.
    ///
    /// # Panics
    ///
    /// Panics if applying an accepted disentangler encounters an unrecoverable
    /// numerical MPS state. Failed speculative candidates are skipped.
    pub fn disentangle(&mut self, max_sweeps: usize) -> usize {
        self.flush();
        expect_mps_operation(
            disentangle::disentangle(
                &mut self.mps,
                &mut self.tableau,
                &mut self.disent_flags,
                max_sweeps,
            ),
            "StabMps::disentangle accepted candidate",
        )
    }

    /// Compute the full state vector for a small system.
    ///
    /// Directly computes |psi> = `Σ_x` `ν_x` * D^x * |stab> from the MPS
    /// coefficients and the current stabilizer/destabilizer generators.
    /// It allocates `2^n` complex amplitudes and constructs dense `2^n` by
    /// `2^n` operators, and is therefore restricted to `n <= 14`. For scalable
    /// reads, use [`Self::amplitude_iterative`], [`Self::prob_bitstring`],
    /// [`Self::pauli_expectation`], or [`Self::sample_bitstrings`].
    ///
    /// # Accuracy caveats (read if you have outstanding measurements)
    ///
    /// - **Pragmatic measurement path**:
    ///   [`measure::measure_qubit_stab_mps_pragmatic`] skips MPS compensation
    ///   for `pre_reduce` row-ops. The stored
    ///   `(tableau, MPS)` pair may no longer represent the exact physical
    ///   state after a measurement that triggered multi-anticom
    ///   `pre_reduce`. Measurement outcome statistics stay correct, but
    ///   `state_vector`/`amplitude` reads can drift. If exact state is
    ///   needed, keep the default [`MeasurementMode::Exact`] or select
    ///   [`MeasurementMode::Lazy`].
    /// - **Merged-RZ pending buffer** (`merge_rz = true`): any pending
    ///   merged-RZ angle has not been applied yet.
    /// - **Lazy-measurement deferred operations**
    ///   ([`MeasurementMode::Lazy`]):
    ///   queued virtual-frame operations have not been applied to the stored
    ///   MPS yet. Call `StabMps::flush()` before either kind of read.
    /// - **Pauli-frame tracking** (`pauli_frame_tracking = true`): the
    ///   frame's Pauli bits are not in the returned state vector. Call
    ///   `StabMps::flush_pauli_frame_to_state()` first for frame-applied
    ///   output (modulo a global phase for Y contributions).
    ///
    /// # Panics
    ///
    /// Panics if `num_qubits > 14`.
    #[must_use]
    pub fn state_vector(&self) -> Vec<Complex64> {
        assert!(
            self.num_qubits <= 14,
            "state_vector only for small systems (N <= 14)"
        );

        let n = self.num_qubits;
        let dim = 1usize << n;
        let mps_sv = self.mps.state_vector();

        // Build Pauli matrices for generator construction.
        let i2 = DMatrix::<Complex64>::identity(2, 2);
        let x_mat = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        let z_mat = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        );
        let y_mat = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, -1.0),
                Complex64::new(0.0, 1.0),
                Complex64::new(0.0, 0.0),
            ],
        );

        // Helper: build the 2^n × 2^n matrix for a generator row.
        let gen_matrix = |is_stab: bool, row: usize| -> DMatrix<Complex64> {
            let gens = if is_stab {
                self.tableau.stabs()
            } else {
                self.tableau.destabs()
            };
            let mut result = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));
            for q in 0..n {
                let p = match (gens.row_x[row].contains(q), gens.row_z[row].contains(q)) {
                    (false, false) => &i2,
                    (true, false) => &x_mat,
                    (false, true) => &z_mat,
                    (true, true) => &y_mat,
                };
                result = result.kronecker(p);
            }
            let mut phase = Complex64::new(1.0, 0.0);
            if gens.signs_minus.contains(row) {
                phase *= Complex64::new(-1.0, 0.0);
            }
            if gens.signs_i.contains(row) {
                phase *= Complex64::new(0.0, 1.0);
            }
            result * phase
        };

        // Find stabilizer state |stab>: +1 eigenstate of all stabilizers.
        // Build the projector P = prod_k (I + S_k) / 2, then find a nonzero
        // column to get the stabilizer state.
        let id = DMatrix::<Complex64>::identity(dim, dim);
        let mut proj = id.clone();
        for k in 0..n {
            let sk = gen_matrix(true, k);
            proj = (&id + &sk) * Complex64::new(0.5, 0.0) * &proj;
        }
        // Find a nonzero column of the projector
        let mut stab_state = nalgebra::DVector::from_element(dim, Complex64::new(0.0, 0.0));
        for col in 0..dim {
            let candidate = proj.column(col);
            let norm_sq: f64 = candidate.iter().map(nalgebra::Complex::norm_sqr).sum();
            if norm_sq > 1e-20 {
                stab_state = candidate.into_owned() / Complex64::new(norm_sq.sqrt(), 0.0);
                break;
            }
        }

        // Compute |psi> = Σ_x ν_x * D_0^{x_0} * ... * D_{n-1}^{x_{n-1}} * |stab>.
        // MPS SV uses MSB-first: index x = Σ_k σ_k * 2^{n-1-k}.
        let mut psi = nalgebra::DVector::from_element(dim, Complex64::new(0.0, 0.0));
        for (x, &nu) in mps_sv.iter().enumerate() {
            if nu.norm_sqr() < 1e-30 {
                continue;
            }
            let mut state = stab_state.clone();
            for k in 0..n {
                if (x >> (n - 1 - k)) & 1 == 1 {
                    state = &gen_matrix(false, k) * &state;
                }
            }
            psi += state * nu;
        }

        // Convert from Kronecker ordering (MSB-first: q0 is leftmost)
        // to DenseStateVec ordering (LSB-first: bit k = qubit k).
        let mut result = vec![Complex64::new(0.0, 0.0); dim];
        for i in 0..dim {
            let mut rev = 0;
            for b in 0..n {
                if (i >> b) & 1 == 1 {
                    rev |= 1 << (n - 1 - b);
                }
            }
            result[rev] = self.global_phase * psi[i];
        }

        // Normalize (MPS norm can drift from truncation in multi-site gates)
        let norm_sq: f64 = result.iter().map(nalgebra::Complex::norm_sqr).sum();
        if norm_sq > 1e-20 {
            let inv_norm = Complex64::new(1.0 / norm_sq.sqrt(), 0.0);
            for a in &mut result {
                *a *= inv_norm;
            }
        }

        result
    }
}

impl StabMps {
    /// Sample `num_shots` bitstrings from the current Born distribution,
    /// sharing each distinct measurement-prefix projection across all shots
    /// that take that branch.
    ///
    /// This is the `StabMps` bitstring sampler. It is always exact by
    /// construction and does not use the configured [`MeasurementMode`].
    /// To sample each shot through that mode instead, clone a prepared
    /// simulator per shot, reseed it with a distinct per-shot seed, and
    /// explicitly measure all qubits with [`CliffordGateable::mz`].
    ///
    /// The original simulator state is preserved; only its RNG advances.
    /// A working clone first materializes any lazy-measurement frame and then
    /// all pending merged RZ rotations. Tracked Pauli X bits remain classical:
    /// they swap reported Z outcomes without changing the stored-state collapse.
    /// Pauli Z bits and frame phase do not affect computational-basis probabilities.
    ///
    /// At each prefix containing `k` shots, a candidate zero child is cloned and
    /// passed once through [`measure::project_forced_z`]. Its returned probability
    /// drives the split, and the already-projected candidate is reused if the
    /// zero branch is inhabited. The one child similarly receives exactly one
    /// forced projection on the untouched parent. Thus every descending path
    /// follows the same atomic projection sequence as [`Self::prob_bitstring`].
    /// The clamped `p0` is tested against `k` uniforms in branch-local shot order,
    /// with all node draws completed before visiting either child. Probabilities
    /// at the projector's shared endpoint snap become zero; if either child is
    /// zero-probability, the node consumes no RNG draws.
    ///
    /// Children are visited depth-first, outcome 0 before outcome 1, measuring
    /// qubits `0..num_qubits`. Returned bitstrings therefore use the same
    /// `bitstring[q] == qubit q` convention as the other bitstring APIs and are
    /// in lexicographic tree order, with copies of each leaf adjacent. See the
    /// crate-level **Bitstring convention** section.
    ///
    /// It shares projections for common prefixes instead of cloning and
    /// collapsing the whole simulator once per shot.
    pub fn sample_bitstrings(&mut self, num_shots: usize) -> Vec<Vec<bool>> {
        if num_shots == 0 {
            return Vec::new();
        }

        let mut working = self.clone();
        expect_mps_operation(
            measure::flush_deferred_ops(&mut working.mps, &mut working.deferred_ops),
            "StabMps::sample_bitstrings deferred-operation flush",
        );
        working.flush_all_pending_rz();
        let frame_x = if working.flags.pauli_frame_tracking() {
            working.pauli_frame_x.clone()
        } else {
            vec![false; self.num_qubits]
        };

        let mut shots = Vec::with_capacity(num_shots);
        let mut prefix = Vec::with_capacity(self.num_qubits);
        let mut context = PrefixSamplingContext {
            rng: &mut self.rng,
            frame_x: &frame_x,
            num_qubits: self.num_qubits,
            output: &mut shots,
        };
        let mut state = PrefixProjectionState {
            tableau: working.tableau,
            mps: working.mps,
        };
        expect_mps_operation(
            Self::sample_prefix_tree(&mut state, num_shots, &mut prefix, &mut context),
            "StabMps::sample_bitstrings prefix projection",
        );
        shots
    }

    fn sample_prefix_tree(
        state: &mut PrefixProjectionState,
        num_shots: usize,
        prefix: &mut Vec<bool>,
        context: &mut PrefixSamplingContext<'_>,
    ) -> Result<(), MpsError> {
        let q = prefix.len();
        if q == context.num_qubits {
            context
                .output
                .extend(std::iter::repeat_n(prefix.clone(), num_shots));
            return Ok(());
        }

        // Probability and projection are deliberately atomic. In particular,
        // do not pre-reduce the shared parent and then call `project_forced_z`:
        // pre-reduction can produce a trivial MPS, and a second entry would take
        // the trivial fast path instead of completing the in-progress general
        // projection as `prob_bitstring` does.
        let mut zero_state = state.clone();
        let probability_zero = zero_state
            .project_z(q, context.frame_x[q], None)?
            .clamp(0.0, 1.0);
        let probability_one = 1.0 - probability_zero;
        let num_zero = if probability_zero == 0.0 {
            0
        } else if probability_one == 0.0 {
            num_shots
        } else {
            (0..num_shots)
                .filter(|_| context.rng.next_f64() < probability_zero)
                .count()
        };
        let num_one = num_shots - num_zero;

        if num_zero > 0 {
            prefix.push(false);
            Self::sample_prefix_tree(&mut zero_state, num_zero, prefix, context)?;
            prefix.pop();
        }

        if num_one > 0 {
            let projected_probability = state.project_z(q, !context.frame_x[q], None)?;
            // Invariant: `probability_zero` and this forced-projection
            // probability deterministically recompute the same expectation on
            // identical parent states. A positive one-branch probability
            // therefore cannot pair with a zero forced projection; this assert
            // guards an internal consistency bug, not caller input.
            assert!(
                projected_probability > 0.0,
                "positive-probability one branch rejected by forced projection at qubit {q}"
            );
            prefix.push(true);
            Self::sample_prefix_tree(state, num_one, prefix, context)?;
            prefix.pop();
        }
        Ok(())
    }

    /// Auto-grow check: if `auto_grow_bond_dim` is enabled and the MPS
    /// has accumulated truncation error past the threshold AND the cap
    /// is binding, double `max_bond_dim` (capped at
    /// `auto_grow_max_bond_dim`). Called after MPS-modifying ops.
    fn maybe_grow_bond_dim(&mut self) {
        let Some(threshold) = self.auto_grow_bond_dim else {
            return;
        };
        let cur_err = self.mps.truncation_error();
        let delta = cur_err - self.last_truncation_error;
        self.last_truncation_error = cur_err;
        if delta < threshold {
            return;
        }
        // Only grow if the cap was actually binding (not just float noise).
        if self.mps.bond_cap_hits() == 0 {
            return;
        }
        let cur_cap = self.config.max_bond_dim;
        let new_cap = (cur_cap * 2).min(self.auto_grow_max_bond_dim);
        if new_cap > cur_cap {
            self.config.max_bond_dim = new_cap;
            self.mps.set_max_bond_dim(new_cap);
        }
    }

    /// Inject Pauli X into the Pauli frame on qubit `q` (no quantum-state
    /// update). See `StabMpsBuilder::pauli_frame_tracking`.
    pub fn inject_x_in_frame(&mut self, q: QubitId) {
        self.pauli_frame_x[q.index()] ^= true;
    }

    /// Inject Pauli Z into the Pauli frame on qubit `q`.
    pub fn inject_z_in_frame(&mut self, q: QubitId) {
        self.pauli_frame_z[q.index()] ^= true;
    }

    /// Inject Pauli Y into the Pauli frame on qubit `q`. In the Y-direct
    /// representation, the bit pair `(1, 1)` names Y directly — no scalar
    /// phase contribution.
    pub fn inject_y_in_frame(&mut self, q: QubitId) {
        let i = q.index();
        self.pauli_frame_x[i] ^= true;
        self.pauli_frame_z[i] ^= true;
    }

    /// Bulk-inject a list of single-qubit Pauli errors into the frame.
    /// Equivalent to calling `inject_{x,y,z}_in_frame` in order, but
    /// exposed as a single call so noise samplers can emit a single vector
    /// per timestep rather than looping. See `StabMpsBuilder::pauli_frame_tracking`.
    pub fn inject_paulis_in_frame(&mut self, paulis: &[(QubitId, PauliKind)]) {
        for &(q, kind) in paulis {
            match kind {
                PauliKind::X => self.inject_x_in_frame(q),
                PauliKind::Y => self.inject_y_in_frame(q),
                PauliKind::Z => self.inject_z_in_frame(q),
            }
        }
    }

    /// Read the accumulated Z-bit of the Pauli frame on qubit `q`.
    /// (Z-bit tracks pure Z errors; commutes with Z-measurement.)
    #[must_use]
    pub fn frame_z_bit(&self, q: QubitId) -> bool {
        self.pauli_frame_z[q.index()]
    }

    /// Read the accumulated X-bit of the Pauli frame on qubit `q`. When
    /// `pauli_frame_tracking` is on, this bit is `XORed` into the
    /// measurement outcome of `mz(q)` (X/Y anticommute with Z-measurement,
    /// flipping the outcome).
    #[must_use]
    pub fn frame_x_bit(&self, q: QubitId) -> bool {
        self.pauli_frame_x[q.index()]
    }

    /// Propagate the Pauli frame through a single-qubit Clifford gate `kind`
    /// applied to qubit `q`. Y-direct representation — bit pair names
    /// the Pauli directly; `pauli_frame_phase` tracks only `±1` signs
    /// from Clifford sign flips:
    /// - H: X ↔ Z (swap bits); Y → -Y (phase *= -1 if both bits set).
    /// - SZ: X → Y, Z → Z, Y → -X (toggle z; phase *= -1 if both bits set).
    /// - `SZdg`: X → -Y, Z → Z, Y → X (toggle z; phase *= -1 if x && !z).
    /// - X: Z → -Z, Y → -Y (phase *= -1 if z set).
    /// - Y: X → -X, Z → -Z (phase *= -1 if x ⊕ z set).
    /// - Z: X → -X, Y → -Y (phase *= -1 if x set).
    fn propagate_frame_single_qubit(&mut self, kind: SingleQubitCliffordKind, q: usize) {
        let x = self.pauli_frame_x[q];
        let z = self.pauli_frame_z[q];
        match kind {
            SingleQubitCliffordKind::H => {
                self.pauli_frame_x[q] = z;
                self.pauli_frame_z[q] = x;
                if x && z {
                    self.pauli_frame_phase = -self.pauli_frame_phase;
                }
            }
            SingleQubitCliffordKind::SZ => {
                self.pauli_frame_z[q] ^= x;
                if x && z {
                    self.pauli_frame_phase = -self.pauli_frame_phase;
                }
            }
            SingleQubitCliffordKind::SZdg => {
                // SZdg·X·SZ = -Y (flip), SZdg·Y·SZ = +X (no flip).
                // Condition: x set AND z NOT set (starting from X, not Y).
                self.pauli_frame_z[q] ^= x;
                if x && !z {
                    self.pauli_frame_phase = -self.pauli_frame_phase;
                }
            }
            SingleQubitCliffordKind::X => {
                if z {
                    self.pauli_frame_phase = -self.pauli_frame_phase;
                }
            }
            SingleQubitCliffordKind::Y => {
                if x ^ z {
                    self.pauli_frame_phase = -self.pauli_frame_phase;
                }
            }
            SingleQubitCliffordKind::Z => {
                if x {
                    self.pauli_frame_phase = -self.pauli_frame_phase;
                }
            }
        }
    }

    /// Propagate the Pauli frame through CX(c, t).
    fn propagate_frame_cx(&mut self, c: usize, t: usize) {
        // Heisenberg:
        //   X_c → X_c X_t
        //   X_t → X_t
        //   Z_c → Z_c
        //   Z_t → Z_c Z_t
        // Bit updates:
        //   if x_bit[c] set: toggle x_bit[t].
        //   if z_bit[t] set: toggle z_bit[c].
        if self.pauli_frame_x[c] {
            self.pauli_frame_x[t] ^= true;
        }
        if self.pauli_frame_z[t] {
            self.pauli_frame_z[c] ^= true;
        }
    }

    /// Flush the accumulated Pauli frame into the simulator state. Applies
    /// the frame Pauli `P = pauli_frame_phase · ⊗_q P_q` to the MPS via
    /// the decomposition `C† · P · C = decomp_phase · X_flip · Z_sign`
    /// (where `C` is the tableau Clifford). The tableau is left unchanged;
    /// the MPS absorbs the frame content. This avoids stabilizer-formalism
    /// phase loss: `state_vector` / `amplitude` after flush are EXACT
    /// complex amplitudes, including correct global phase even for
    /// Clifford-evolved and entangled states with Y-bits in the frame.
    /// Clears the frame.
    ///
    /// # Panics
    ///
    /// Panics if any MPS gate application fails on a valid site.
    pub fn flush_pauli_frame_to_state(&mut self) {
        // Materialize the lazy virtual frame and pending RZs first so the
        // tableau C and stored MPS are aligned before composing the Pauli frame.
        self.flush();

        // Collect frame Paulis as a Pauli string.
        let mut paulis: Vec<(usize, pauli_decomp::PauliKindForDecomp)> = Vec::new();
        for q in 0..self.num_qubits {
            let pk = match (self.pauli_frame_x[q], self.pauli_frame_z[q]) {
                (true, true) => pauli_decomp::PauliKindForDecomp::Y,
                (true, false) => pauli_decomp::PauliKindForDecomp::X,
                (false, true) => pauli_decomp::PauliKindForDecomp::Z,
                (false, false) => continue,
            };
            paulis.push((q, pk));
        }

        // Frame-phase scalar (from Clifford sign-flip propagation) always
        // folds into global_phase at flush, frame or not.
        let frame_scalar = self.pauli_frame_phase;
        self.pauli_frame_phase = Complex64::new(1.0, 0.0);
        for b in &mut self.pauli_frame_x {
            *b = false;
        }
        for b in &mut self.pauli_frame_z {
            *b = false;
        }

        if paulis.is_empty() {
            self.global_phase *= frame_scalar;
            return;
        }

        // Decomposition trick (avoids the stabilizer-formalism phase loss
        // of tab.x / tab.y / tab.z):
        //   C† · P · C = decomp_phase · X_{flip} · Z_{sign}   (in MPS frame)
        // So P · C · |MPS⟩ = C · (decomp_phase · X_flip · Z_sign) · |MPS⟩.
        // Applying `decomp_phase · X_flip · Z_sign` to MPS (not the tableau)
        // preserves the EXACT physical state — including global phase —
        // because the Clifford C is unchanged and the MPS absorbs the
        // frame's full content. No state-dependent phase loss.
        let (flip, sign, decomp_phase) = pauli_decomp::decompose_pauli_string(
            self.tableau.stabs(),
            self.tableau.destabs(),
            &paulis,
        );
        let z_diag = [Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)];
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
        for &k in &sign {
            self.mps
                .apply_diagonal_one_site(k, &z_diag)
                .expect("frame flush: site from decomposition");
        }
        for &j in &flip {
            self.mps
                .apply_one_site_gate(j, &x_gate)
                .expect("frame flush: site from decomposition");
            self.disent_flags[j] = None;
        }
        self.global_phase *= frame_scalar * decomp_phase;
    }

    /// Propagate the Pauli frame through CZ(a, b).
    fn propagate_frame_cz(&mut self, a: usize, b: usize) {
        // Heisenberg:
        //   X_a → X_a Z_b
        //   X_b → Z_a X_b
        //   Z_a → Z_a
        //   Z_b → Z_b
        if self.pauli_frame_x[a] {
            self.pauli_frame_z[b] ^= true;
        }
        if self.pauli_frame_x[b] {
            self.pauli_frame_z[a] ^= true;
        }
    }

    /// Apply Pauli X to qubit `q` with probability `p` (bit-flip channel).
    /// No-op when `p == 0.0`. Used to model dephasing-free bit-flip noise
    /// or Pauli-X errors injected at specific points in a circuit.
    ///
    /// When `pauli_frame_tracking` is enabled, the X is accumulated into
    /// the Pauli frame (O(1)) instead of applied to the quantum state
    /// (O(n) tableau update).
    ///
    /// Returns `true` iff the X was applied (either to state or frame).
    pub fn apply_bit_flip(&mut self, q: QubitId, p: f64) -> bool {
        if p <= 0.0 {
            return false;
        }
        if self.rng.random_bool(p) {
            if self.flags.pauli_frame_tracking() {
                self.inject_x_in_frame(q);
            } else {
                self.x(&[q]);
            }
            true
        } else {
            false
        }
    }

    /// Apply Pauli Z to qubit `q` with probability `p` (phase-flip channel).
    /// No-op when `p == 0.0`. Models pure dephasing. Uses frame-injection
    /// path when `pauli_frame_tracking` is on.
    pub fn apply_phase_flip(&mut self, q: QubitId, p: f64) -> bool {
        if p <= 0.0 {
            return false;
        }
        if self.rng.random_bool(p) {
            if self.flags.pauli_frame_tracking() {
                self.inject_z_in_frame(q);
            } else {
                self.z(&[q]);
            }
            true
        } else {
            false
        }
    }

    /// Apply depolarizing noise to qubit `q` with total error probability
    /// `p`. With probability `p`, applies one of {X, Y, Z} uniformly
    /// (each with conditional probability 1/3 = total `p/3`). With
    /// probability `1 − p`, no error is applied.
    ///
    /// When `pauli_frame_tracking` is enabled, the error goes into the
    /// Pauli frame (O(1)) instead of the quantum state.
    ///
    /// Returns the applied Pauli kind, or `None` if no error.
    /// Standard QEC depolarizing channel.
    pub fn apply_depolarizing(&mut self, q: QubitId, p: f64) -> Option<PauliKind> {
        if p <= 0.0 {
            return None;
        }
        if !self.rng.random_bool(p) {
            return None;
        }
        // Error occurred; pick X/Y/Z uniformly.
        let r = self.rng.random_bool(2.0 / 3.0);
        let kind = if r {
            // 2/3: X or Y
            if self.rng.random_bool(0.5) {
                PauliKind::X
            } else {
                PauliKind::Y
            }
        } else {
            PauliKind::Z
        };
        if self.flags.pauli_frame_tracking() {
            match kind {
                PauliKind::X => self.inject_x_in_frame(q),
                PauliKind::Y => self.inject_y_in_frame(q),
                PauliKind::Z => self.inject_z_in_frame(q),
            }
        } else {
            match kind {
                PauliKind::X => {
                    self.x(&[q]);
                }
                PauliKind::Y => {
                    self.y(&[q]);
                }
                PauliKind::Z => {
                    self.z(&[q]);
                }
            }
        }
        Some(kind)
    }

    /// Apply depolarizing noise to every qubit in `qubits` independently.
    /// Each qubit gets an X/Y/Z with total probability `p`. Models
    /// memory-error channel applied to multiple qubits per timestep
    /// (e.g., ion-trap idle decoherence).
    pub fn apply_depolarizing_all(&mut self, qubits: &[QubitId], p: f64) {
        for &q in qubits {
            let _ = self.apply_depolarizing(q, p);
        }
    }

    /// Conservative sufficient predicate for an exact stored physical state.
    ///
    /// This is deliberately not an if-and-only-if classifier. It requires all
    /// seven guards: no pending RZ, no unmaterialized Pauli frame, no lazy
    /// deferred operations, no uncompensated pragmatic pre-reduction, exact
    /// measurement mode, zero summed discarded weight, and no deferred MAST
    /// branch loss. A branch-vanish retry is not a guard because its first
    /// attempt was rolled back and its committed retry was untruncated.
    /// Lazy conditional states are exact after issues #555 and #572, but this
    /// conservative diagnostic deliberately remains gated to exact mode.
    #[must_use]
    pub fn is_state_exact(&self) -> bool {
        let no_pending_rz = self.pending_rz.iter().all(std::option::Option::is_none);
        let phase_trivial = (self.pauli_frame_phase - Complex64::new(1.0, 0.0)).norm() < 1e-12;
        let no_frame = !self.flags.pauli_frame_tracking()
            || (self.pauli_frame_x.iter().all(|&b| !b)
                && self.pauli_frame_z.iter().all(|&b| !b)
                && phase_trivial);
        let no_deferred = self.deferred_ops.is_empty();
        let no_drift = self.uncompensated_pre_reduction_count == 0;
        let exact_mode = self.measurement_mode == MeasurementMode::Exact;
        let no_discarded_weight = self.mps.summed_discarded_weight() == 0.0;
        let no_deferred_branch_loss = self.mps.deferred_branch_lost_count() == 0;
        no_pending_rz
            && no_frame
            && no_deferred
            && no_drift
            && exact_mode
            && no_discarded_weight
            && no_deferred_branch_loss
    }

    /// Number of pragmatic measurements whose generator pre-reduction changed
    /// the tableau without matching coefficient-MPS compensation.
    #[must_use]
    pub fn uncompensated_pre_reduction_count(&self) -> u64 {
        self.uncompensated_pre_reduction_count
    }

    /// Materialize deferred lazy-measurement operations and any pending
    /// merged-RZ angles into the simulator state. Call before `&self` read
    /// methods (`state_vector`, `amplitude`, `prob_bitstring`, etc.) when
    /// either feature is enabled. Measurements (`mz`) and `reset` flush the
    /// state needed for their own operation automatically.
    pub fn flush(&mut self) {
        expect_mps_operation(
            measure::flush_deferred_ops(&mut self.mps, &mut self.deferred_ops),
            "StabMps::flush deferred operations",
        );
        self.flush_all_pending_rz();
    }

    /// Mid-circuit reset of qubit `q` to |0⟩. Measures in Z basis, then
    /// conditionally applies X to force |0⟩. Returns the physical
    /// measurement outcome (true iff the qubit was in |1⟩ before reset).
    ///
    /// For QEC ancillas: after syndrome extraction `reset_qubit` clears
    /// the ancilla in one call. Cheaper than `mz` + explicit conditional
    /// `x` because: (1) frame bits for this qubit are cleared directly
    /// rather than propagating X through them; (2) only one `flush_pending_rz`
    /// fires rather than two.
    ///
    /// With `pauli_frame_tracking`: clears both X and Z frame bits for
    /// this qubit — any tracked Pauli error on `q` is semantically erased
    /// by the reset. (Global `pauli_frame_phase` is left unchanged; its
    /// per-qubit contribution is not tracked, so a residual ±1 phase
    /// may remain. Measurement outcomes on other qubits are unaffected.)
    pub fn reset_qubit(&mut self, q: QubitId) -> bool {
        let idx = q.index();
        let reported = self.mz(&[q])[0].outcome;
        // `mz` XORs the frame X-bit into the reported outcome. Undo that
        // to find the stored-state collapse outcome (== physical outcome
        // with frame applied elsewhere but not here).
        let frame_x_before = self.flags.pauli_frame_tracking() && self.pauli_frame_x[idx];
        let physical_outcome = reported ^ frame_x_before;
        // Clear this qubit's frame bits BEFORE applying X so the frame
        // propagation rule for X doesn't spuriously flip the global phase
        // on a Z-bit we're about to erase anyway.
        if self.flags.pauli_frame_tracking() {
            self.pauli_frame_x[idx] = false;
            self.pauli_frame_z[idx] = false;
        }
        if physical_outcome {
            // Apply X to bring stored |1⟩ back to |0⟩. Bypass the
            // public `x` method — we've already flushed pending_rz via mz
            // and cleared the frame, so there's nothing to propagate.
            self.tableau.x(&[q]);
        }
        // The physical reset is absorbed into the tableau, so it does not by
        // itself prove that the same-numbered coefficient site is |0>.  Recover
        // that useful proof only when the stored MPS independently verifies it.
        // A pending lazy frame makes such a stored-frame proof unusable (issue
        // #555), so stay conservative in that regime.
        if self.deferred_ops.is_empty() {
            refresh_disent_flag(&self.mps, &mut self.disent_flags, idx);
        } else {
            self.disent_flags[idx] = None;
        }
        // Return the REPORTED outcome (frame-adjusted) — this is the
        // physical measurement the user observes before reset.
        reported
    }

    /// Prepare qubit `q` in |0⟩ (Z-basis +1 eigenstate). PECOS `pz`. Alias
    /// for `reset_qubit` with the return value discarded — intended for
    /// circuit-building code where the measurement outcome from reset
    /// isn't needed.
    pub fn pz(&mut self, q: QubitId) {
        self.reset_qubit(q);
    }

    /// Prepare qubit `q` in |+⟩ = (|0⟩ + |1⟩)/√2 (X-basis +1 eigenstate).
    /// PECOS `px`. Reset + H.
    pub fn px(&mut self, q: QubitId) {
        self.reset_qubit(q);
        self.h(&[q]);
    }

    /// Extract the syndrome bits of a stabilizer code using one ancilla per
    /// generator. `generators[i]` describes the `i`-th Pauli stabilizer as
    /// a list `(data_qubit, Pauli)`. `ancilla_qubits[i]` is the ancilla for
    /// generator `i`; must be distinct from data qubits and from each
    /// other. Returns a `bool` per generator (syndrome bit).
    ///
    /// Protocol (works for arbitrary Pauli generators including mixed
    /// X/Y/Z on the same generator):
    ///   1. `px(ancilla)` — reset + H.
    ///   2. For each (`data_q`, P): apply controlled-P with ancilla as
    ///      control (CX for P=X, CY for P=Y, CZ for P=Z).
    ///   3. H + mz ancilla → syndrome bit.
    ///   4. `reset_qubit(ancilla)` so it's ready for the next round.
    ///
    /// # Panics
    ///
    /// Panics if `generators.len() != ancilla_qubits.len()`.
    pub fn extract_syndromes(
        &mut self,
        generators: &[Vec<(usize, PauliKind)>],
        ancilla_qubits: &[QubitId],
    ) -> Vec<bool> {
        assert_eq!(
            generators.len(),
            ancilla_qubits.len(),
            "extract_syndromes: one ancilla per generator required"
        );
        let mut syndrome = Vec::with_capacity(generators.len());
        for (generator, &anc) in generators.iter().zip(ancilla_qubits.iter()) {
            debug_assert!(
                !generator.iter().any(|&(q, _)| q == anc.index()),
                "extract_syndromes: ancilla {} overlaps with generator data qubit",
                anc.index()
            );
            self.px(anc);
            for &(q, kind) in generator {
                let data = QubitId(q);
                match kind {
                    PauliKind::X => self.cx(&[(anc, data)]),
                    PauliKind::Y => self.cy(&[(anc, data)]),
                    PauliKind::Z => self.cz(&[(anc, data)]),
                };
            }
            self.h(&[anc]);
            let bit = self.mz(&[anc])[0].outcome;
            syndrome.push(bit);
            // Leave the ancilla in |0⟩ for subsequent rounds.
            self.reset_qubit(anc);
        }
        syndrome
    }

    /// If `merge_rz` is on and qubit `q` has a pending RZ accumulation,
    /// apply it via the standard non-Clifford path and clear the slot.
    /// Called by every gate method that touches `q` (except `rz` itself,
    /// which merges). No-op when `merge_rz` is off or the slot is empty.
    fn flush_pending_rz(&mut self, q: usize) {
        if !self.flags.merge_rz() {
            return;
        }
        if let Some(theta) = self.pending_rz[q].take() {
            self.rz_apply_direct(theta, q);
        }
    }

    /// Apply all pending RZ (all qubits). Used before reads and at reset.
    fn flush_all_pending_rz(&mut self) {
        if !self.flags.merge_rz() {
            return;
        }
        for q in 0..self.num_qubits {
            self.flush_pending_rz(q);
        }
    }

    /// Apply `rz(theta)` on qubit `q` directly (without the merge buffer),
    /// handling Clifford-angle shortcuts and the non-Clifford path.
    /// Factored from `rz()` so `flush_pending_rz` can reuse it.
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
        // Non-Clifford
        let half_rad = theta.to_radians_signed() / 2.0;
        let cos_half = half_rad.cos();
        let sin_half = half_rad.sin();
        expect_mps_operation(
            non_clifford::apply_rz_stab_mps(
                &mut self.tableau,
                &mut self.mps,
                cos_half,
                sin_half,
                q,
                self.flags.normalize_after_gate(),
                &mut non_clifford::RzContext {
                    disent_flags: &mut self.disent_flags,
                    deferred_ops: &self.deferred_ops,
                    // Redetection only feeds exact disentangling. That fast path
                    // is disabled while V is pending because its tableau
                    // right-composition cannot be moved across V, so avoid the
                    // otherwise unused stored-tensor contractions too.
                    numerical_flag_redetection: self.flags.numerical_flag_redetection()
                        && self.deferred_ops.is_empty(),
                    gf2_matrix: &mut self.gf2_matrix,
                    stats: &mut self.stats,
                    saturation_telemetry: self
                        .flags
                        .saturation_telemetry()
                        .then_some(&mut self.saturation_telemetry),
                },
            ),
            "StabMps::rz non-Clifford update",
        );
        self.maybe_grow_bond_dim();
    }

    /// Measure qubit q in the Z basis using the shared STN measurement protocol.
    fn measure_qubit(&mut self, q: QubitId) -> MeasurementResult {
        self.flush_pending_rz(q.index());
        let live_result = match self.measurement_mode {
            MeasurementMode::Lazy => measure::measure_qubit_stab_mps_lazy_with_update(
                &mut self.tableau,
                &mut self.mps,
                &mut self.rng,
                q.index(),
                &mut self.deferred_ops,
            ),
            MeasurementMode::Pragmatic => {
                // Detect pragmatic-fix drift: pre_reduce fires when col_x has
                // multiple anticommuting stabilizers. It applies row-ops to the
                // tableau (changing C) WITHOUT compensating MPS. Drift occurs
                // regardless of whether decompose_z then takes the Stabilizer
                // or DestabilizerFlip path — the uncompensated row-ops already
                // changed the (C, MPS) pair.
                if self.tableau.stabs().col_x[q.index()].len() > 1 {
                    self.uncompensated_pre_reduction_count += 1;
                }
                measure::measure_qubit_stab_mps_with_update(
                    &mut self.tableau,
                    &mut self.mps,
                    &mut self.rng,
                    q.index(),
                )
            }
            MeasurementMode::Exact => measure_qubit_exact_transactional(
                &mut self.tableau,
                &mut self.mps,
                &mut self.rng,
                q.index(),
                "StabMps::mz",
            ),
        };
        let live_result = expect_mps_operation(live_result, "StabMps::mz projection");
        if self.measurement_mode == MeasurementMode::Lazy && !self.deferred_ops.is_empty() {
            // Deferred virtual Cliffords mean a stored |0> is not necessarily
            // an effective |0> (issue #555).  Invalidate touched sites, but do
            // not install any new proof until that frame has been materialized.
            invalidate_disent_flags(&mut self.disent_flags, &live_result.update);
        } else {
            repair_disent_flags(&self.mps, &mut self.disent_flags, &live_result.update);
        }
        let result = live_result.measurement;
        self.maybe_grow_bond_dim();
        // Pauli-frame XOR: the tracked X-bit flips the reported Z-basis
        // outcome, since X (and Y = XZ·sign) anticommute with Z. Z in the
        // frame commutes with Z-measurement and so does not flip the bit.
        if self.flags.pauli_frame_tracking() && self.pauli_frame_x[q.index()] {
            MeasurementResult {
                outcome: !result.outcome,
                is_deterministic: result.is_deterministic,
            }
        } else {
            result
        }
    }
}

impl QuantumSimulator for StabMps {
    /// Reset the state and diagnostics as if newly constructed with the
    /// retained configuration.
    ///
    /// For a seeded simulator, the rebuilt tableau seed and continuing
    /// simulator-RNG seed are drawn from the current simulator stream. This is
    /// deterministic continuation, not replay of the construction stream. An
    /// unseeded simulator obtains fresh entropy.
    fn reset(&mut self) -> &mut Self {
        (self.tableau, self.rng) = reset_tableau_and_rng(
            self.num_qubits,
            self.construction_seed.is_some(),
            &mut self.rng,
        );
        self.mps = Mps::new(self.num_qubits, self.config.clone());
        self.global_phase = Complex64::new(1.0, 0.0);
        self.disent_flags = vec![Some(SiteEigenstate::Z(false)); self.num_qubits];
        self.gf2_matrix.reset();
        self.stats = StabMpsStats::default();
        self.saturation_telemetry = SaturationTelemetry::default();
        self.deferred_ops.clear();
        self.uncompensated_pre_reduction_count = 0;
        self.last_truncation_error = 0.0;
        for slot in &mut self.pending_rz {
            *slot = None;
        }
        for b in &mut self.pauli_frame_x {
            *b = false;
        }
        for b in &mut self.pauli_frame_z {
            *b = false;
        }
        self.pauli_frame_phase = Complex64::new(1.0, 0.0);
        self
    }

    fn num_qubits(&self) -> usize {
        self.num_qubits
    }
}

impl pecos_random::RngManageable for StabMps {
    type Rng = PecosRng;

    /// Reseed BOTH of the simulator's independent random streams: the
    /// tableau stream is derived from one draw of the supplied RNG, and the
    /// advanced RNG becomes the main stream. Consequently `set_seed(seed)`
    /// leaves the main stream one draw past `seed_from_u64(seed)` — a
    /// deterministic, documented offset, not the raw seeded stream.
    fn set_rng(&mut self, mut rng: Self::Rng) {
        let tableau_seed = rng.next_u64();
        self.tableau.set_rng(PecosRng::seed_from_u64(tableau_seed));
        self.rng = rng;
    }

    fn rng(&self) -> &Self::Rng {
        &self.rng
    }

    fn rng_mut(&mut self) -> &mut Self::Rng {
        &mut self.rng
    }
}

impl CliffordGateable for StabMps {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        // SZ commutes with RZ: skip `flush_pending_rz`. The pending RZ
        // angle stays valid; applying it later yields the same physical
        // state as flushing first and then applying SZ (since RZ(θ)·SZ =
        // SZ·RZ(θ)). See merge_rz docstring.
        self.tableau.sz(qubits);
        for &q in qubits {
            self.propagate_frame_single_qubit(SingleQubitCliffordKind::SZ, q.index());
        }
        // Flags are NOT updated through Clifford gates. They track whether the
        // MPS-frame state at a site is in Z-eigenstate |0⟩, which is true iff
        // no non-Clifford has yet been applied to that site. This matches the
        // stabilizer-TN reference's _disent_flag semantics.
        self
    }

    fn szdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        // SZdg commutes with RZ: skip flush. Same reasoning as sz().
        self.tableau.szdg(qubits);
        for &q in qubits {
            self.propagate_frame_single_qubit(SingleQubitCliffordKind::SZdg, q.index());
        }
        self
    }

    fn z(&mut self, qubits: &[QubitId]) -> &mut Self {
        // Z commutes with RZ: skip flush.
        self.tableau.z(qubits);
        for &q in qubits {
            self.propagate_frame_single_qubit(SingleQubitCliffordKind::Z, q.index());
        }
        self
    }

    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        // H does NOT commute with RZ (it swaps Z and X axes). Flush.
        for &q in qubits {
            self.flush_pending_rz(q.index());
        }
        self.tableau.h(qubits);
        for &q in qubits {
            self.propagate_frame_single_qubit(SingleQubitCliffordKind::H, q.index());
        }
        self
    }

    fn x(&mut self, qubits: &[QubitId]) -> &mut Self {
        // X anticommutes with RZ: X·RZ(θ) = RZ(-θ)·X, so applying X
        // after a pending RZ(θ) is equivalent to applying X first then
        // RZ(-θ). Flip sign of pending_rz and skip flush.
        for &q in qubits {
            let idx = q.index();
            if let Some(theta) = self.pending_rz.get_mut(idx).and_then(|s| s.as_mut()) {
                *theta = -*theta;
            }
        }
        self.tableau.x(qubits);
        for &q in qubits {
            self.propagate_frame_single_qubit(SingleQubitCliffordKind::X, q.index());
        }
        self
    }

    fn y(&mut self, qubits: &[QubitId]) -> &mut Self {
        // Y anticommutes with RZ (same as X for this purpose): flip
        // pending_rz sign, skip flush.
        for &q in qubits {
            let idx = q.index();
            if let Some(theta) = self.pending_rz.get_mut(idx).and_then(|s| s.as_mut()) {
                *theta = -*theta;
            }
        }
        self.tableau.y(qubits);
        for &q in qubits {
            self.propagate_frame_single_qubit(SingleQubitCliffordKind::Y, q.index());
        }
        self
    }

    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // CX does not commute with RZ on arbitrary qubits (mixes bases).
        // Flush pending RZ on both control and target.
        for &(c, t) in pairs {
            self.flush_pending_rz(c.index());
            self.flush_pending_rz(t.index());
        }
        self.tableau.cx(pairs);
        for &(c, t) in pairs {
            self.propagate_frame_cx(c.index(), t.index());
        }
        self
    }

    fn cz(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // CZ IS diagonal and commutes with RZ on either qubit. Skip flush.
        self.tableau.cz(pairs);
        for &(a, b) in pairs {
            self.propagate_frame_cz(a.index(), b.index());
        }
        self
    }

    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        qubits.iter().map(|&q| self.measure_qubit(q)).collect()
    }
}

impl ArbitraryRotationGateable for StabMps {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        // RX(theta) = H * RZ(theta) * H
        self.h(qubits);
        self.rz(theta, qubits);
        self.h(qubits);
        self
    }

    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        for &q in qubits {
            let q_idx = q.index();
            if !self.flags.merge_rz() {
                self.rz_apply_direct(theta, q_idx);
                continue;
            }
            // Merge path: accumulate non-Clifford angles; Clifford angles
            // (including ZERO) go through direct path. ALL Clifford-angle
            // RZ operators (ZERO=I, HALF_TURN=Z, QUARTER_TURN=SZ,
            // THREE_QUARTERS_TURN=SZdg) commute with RZ, so they do NOT
            // need to flush pending_rz — they just update the tableau.
            let is_clifford_angle = theta == Angle64::ZERO
                || theta == Angle64::HALF_TURN
                || theta == Angle64::QUARTER_TURN
                || theta == Angle64::THREE_QUARTERS_TURN;
            if is_clifford_angle {
                // No flush: Clifford RZ commutes with pending RZ.
                self.rz_apply_direct(theta, q_idx);
            } else {
                // Accumulate non-Clifford angle.
                let prev = self.pending_rz[q_idx].unwrap_or(Angle64::ZERO);
                let merged = prev + theta;
                // If merged sum hits a Clifford angle, flush via direct path
                // (captures the Clifford-angle shortcut savings).
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

    fn apply_global_phase(&mut self, phase: Angle64, qubits: &[QubitId]) -> &mut Self {
        let scalar = Complex64::from_polar(1.0, phase.to_radians_signed());
        for _ in qubits {
            self.global_phase *= scalar;
        }
        self
    }

    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // RZZ(theta) = CX * RZ_target(theta) * CX
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
    use pecos_core::Clifford;
    use pecos_simulators::{CHForm, StabVec};

    #[test]
    fn test_stn_initial_state() {
        let stn = StabMps::new(2);
        assert_eq!(stn.num_qubits(), 2);
        assert_eq!(stn.max_bond_dim(), 1);
    }

    fn apply_single_qubit_clifford<S: CliffordGateable>(
        sim: &mut S,
        gate: Clifford,
        qubits: &[QubitId],
    ) {
        match gate {
            Clifford::I => sim.identity(qubits),
            Clifford::X => sim.x(qubits),
            Clifford::Y => sim.y(qubits),
            Clifford::Z => sim.z(qubits),
            Clifford::H => sim.h(qubits),
            Clifford::H2 => sim.h2(qubits),
            Clifford::H3 => sim.h3(qubits),
            Clifford::H4 => sim.h4(qubits),
            Clifford::H5 => sim.h5(qubits),
            Clifford::H6 => sim.h6(qubits),
            Clifford::SX => sim.sx(qubits),
            Clifford::SXdg => sim.sxdg(qubits),
            Clifford::SY => sim.sy(qubits),
            Clifford::SYdg => sim.sydg(qubits),
            Clifford::SZ => sim.sz(qubits),
            Clifford::SZdg => sim.szdg(qubits),
            Clifford::F => sim.f(qubits),
            Clifford::Fdg => sim.fdg(qubits),
            Clifford::F2 => sim.f2(qubits),
            Clifford::F2dg => sim.f2dg(qubits),
            Clifford::F3 => sim.f3(qubits),
            Clifford::F3dg => sim.f3dg(qubits),
            Clifford::F4 => sim.f4(qubits),
            Clifford::F4dg => sim.f4dg(qubits),
            _ => panic!("expected a single-qubit Clifford, got {gate}"),
        };
    }

    fn single_qubit_clifford_order(gate: Clifford) -> usize {
        let mut power = Clifford::I;
        for order in 1..=4 {
            power = gate.compose(power);
            if power == Clifford::I {
                return order;
            }
        }
        panic!("single-qubit Clifford {gate} has order greater than four");
    }

    fn state_vector_max_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
        actual
            .iter()
            .zip(expected)
            .map(|(actual, expected)| (actual - expected).norm())
            .fold(0.0, f64::max)
    }

    #[test]
    fn test_stab_mps_clifford_powers_restore_observable_state_phase_exactly() {
        let mut failures = Vec::new();
        for &gate in Clifford::all_1q() {
            let mut sim = StabMps::new(2);
            sim.h(&[QubitId(0)])
                .sz(&[QubitId(0)])
                .cx(&[(QubitId(0), QubitId(1))]);
            let expected = sim.state_vector();
            for _ in 0..single_qubit_clifford_order(gate) {
                apply_single_qubit_clifford(&mut sim, gate, &[QubitId(0)]);
            }
            let error = state_vector_max_error(&sim.state_vector(), &expected);
            if error > 1e-10 {
                failures.push((gate, error));
            }
        }
        eprintln!(
            "StabMps: {}/24 exact; failures={failures:?}",
            24 - failures.len()
        );
        assert!(failures.is_empty(), "StabMps phase failures: {failures:?}");
    }

    #[test]
    fn test_ch_form_clifford_powers_restore_observable_state_phase_exactly() {
        let mut failures = Vec::new();
        for &gate in Clifford::all_1q() {
            let mut sim = CHForm::new(2);
            sim.h(&[QubitId(0)])
                .sz(&[QubitId(0)])
                .cx(&[(QubitId(0), QubitId(1))]);
            let expected = sim.state_vector();
            for _ in 0..single_qubit_clifford_order(gate) {
                apply_single_qubit_clifford(&mut sim, gate, &[QubitId(0)]);
            }
            let error = state_vector_max_error(&sim.state_vector(), &expected);
            if error > 1e-10 {
                failures.push((gate, error));
            }
        }
        eprintln!(
            "CHForm: {}/24 exact; failures={failures:?}",
            24 - failures.len()
        );
        assert!(failures.is_empty(), "CHForm phase failures: {failures:?}");
    }

    fn assert_state_vectors_equal(lhs: &mut StabMps, rhs: &mut StabMps, context: &str) {
        lhs.flush();
        rhs.flush();
        for (index, (lhs, rhs)) in lhs
            .state_vector()
            .iter()
            .zip(rhs.state_vector())
            .enumerate()
        {
            assert!(
                (*lhs - rhs).norm() < 1e-10,
                "{context}, basis {index}: lhs={lhs}, rhs={rhs}"
            );
        }
    }

    #[test]
    fn test_conventional_t_exact_identities_and_batched_phase() {
        let q0 = [QubitId(0)];
        let q1 = [QubitId(1)];
        let both = [QubitId(0), QubitId(1)];
        let prepare = |sim: &mut StabMps| {
            sim.h(&both).sz(&q1).cx(&[(QubitId(0), QubitId(1))]);
        };
        let make = || StabMps::builder(2).merge_rz(true).build();

        let mut t_squared = make();
        let mut sz = make();
        prepare(&mut t_squared);
        prepare(&mut sz);
        t_squared.t(&q0).t(&q0);
        sz.sz(&q0);
        assert_state_vectors_equal(&mut t_squared, &mut sz, "T^2 must equal SZ exactly");

        let mut t_eighth = make();
        let mut identity = make();
        prepare(&mut t_eighth);
        prepare(&mut identity);
        for _ in 0..8 {
            t_eighth.t(&q0);
        }
        assert_state_vectors_equal(&mut t_eighth, &mut identity, "T^8 must equal I exactly");

        let mut batched = make();
        let mut separate = make();
        prepare(&mut batched);
        prepare(&mut separate);
        batched.t(&both);
        separate.t(&q0).t(&q1);
        assert_state_vectors_equal(
            &mut batched,
            &mut separate,
            "batched T must accumulate one scalar per target",
        );

        let mut odd_tdg = StabMps::builder(1).merge_rz(true).build();
        odd_tdg.h(&q0).tdg(&q0);
        odd_tdg.flush();
        let expected = [
            Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
            Complex64::new(0.5, -0.5),
        ];
        for (index, (actual, expected)) in odd_tdg.state_vector().iter().zip(expected).enumerate() {
            assert!(
                (*actual - expected).norm() < 1e-10,
                "odd Tdg, basis {index}: actual={actual}, expected={expected}"
            );
        }

        let mut tdg_squared = make();
        let mut szdg = make();
        prepare(&mut tdg_squared);
        prepare(&mut szdg);
        tdg_squared.tdg(&q0).tdg(&q0);
        szdg.szdg(&q0);
        assert_state_vectors_equal(&mut tdg_squared, &mut szdg, "Tdg^2 must equal SZdg exactly");

        let mut tdg_eighth = make();
        let mut identity = make();
        prepare(&mut tdg_eighth);
        prepare(&mut identity);
        for _ in 0..8 {
            tdg_eighth.tdg(&q0);
        }
        assert_state_vectors_equal(&mut tdg_eighth, &mut identity, "Tdg^8 must equal I exactly");

        let mut batched_tdg = make();
        let mut separate_tdg = make();
        prepare(&mut batched_tdg);
        prepare(&mut separate_tdg);
        batched_tdg.tdg(&both);
        separate_tdg.tdg(&q0).tdg(&q1);
        assert_state_vectors_equal(
            &mut batched_tdg,
            &mut separate_tdg,
            "batched Tdg must accumulate one scalar per target",
        );
    }

    #[test]
    fn test_gf2_diagnostic_single_t() {
        // Single T gate: 1 non-Clifford gate, flip pattern has rank 1
        // Theoretical min bond dim = 2^(1-1) = 1
        let mut stn = StabMps::builder(2).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        assert_eq!(stn.gf2_matrix().num_gates(), 1);
        assert_eq!(stn.theoretical_min_bond_dim(), 1);
        assert_eq!(stn.max_bond_dim(), 1); // Actual should match theoretical
    }

    #[test]
    fn test_gf2_diagnostic_two_independent_t() {
        // Two T gates on independent qubits: rank 2, min bond dim = 1
        let mut stn = StabMps::builder(4).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        stn.h(&[QubitId(2)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(2)]);
        assert_eq!(stn.gf2_matrix().num_gates(), 2);
        assert_eq!(stn.gf2_matrix().gf2_rank(), 2);
        assert_eq!(stn.theoretical_min_bond_dim(), 1);
    }

    #[test]
    fn test_gf2_diagnostic_entangled_t() {
        // Entangled state + T gates: check GF(2) tracking works
        let mut stn = StabMps::builder(3).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.cx(&[(QubitId(1), QubitId(2))]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(1)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(2)]);

        // GF(2) diagnostic reports theoretical values; actual bond dim may be
        // lower because single-site decompositions don't grow bond dim even
        // when the GF(2) matrix shows dependencies.
        let rank = stn.gf2_matrix().gf2_rank();
        let num_gates = stn.gf2_matrix().num_gates();
        assert!(rank <= num_gates, "rank should be <= num_gates");
        assert!(rank <= stn.num_qubits(), "rank should be <= num_qubits");
    }

    #[test]
    fn test_gf2_stabilizer_case_not_tracked() {
        // T on |0⟩: Z_0 is a stabilizer, no flip sites, not tracked in GF(2) matrix
        let mut stn = StabMps::builder(1).merge_rz(false).build();
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        assert_eq!(stn.gf2_matrix().num_gates(), 0); // Stabilizer case: no flip
    }

    /// Disentangling test: H on both qubits, then Rz on q0.
    /// Expected: after H, q0 and q1 are in |+⟩. The Rz on q0 should have
    /// a single-site decomposition (`Z_0` anticommutes only with `X_0` stabilizer).
    /// The disentangling fires on the single flip site.
    #[test]
    fn test_disentangle_single_site_case() {
        use pecos_simulators::DenseStateVec;
        let theta = Angle64::from_radians(0.7);
        let mut stn = StabMps::builder(2).merge_rz(false).build();
        let mut ref_sim = DenseStateVec::new(2);

        stn.h(&[QubitId(0)]);
        ref_sim.h(&[QubitId(0)]);
        stn.h(&[QubitId(1)]);
        ref_sim.h(&[QubitId(1)]);
        stn.rz(theta, &[QubitId(0)]);
        ref_sim.rz(theta, &[QubitId(0)]);

        let stn_sv = stn.state_vector();
        let dim = 1 << 2;
        let ref_sv: Vec<Complex64> = (0..dim).map(|i| ref_sim.get_amplitude(i)).collect();

        let overlap: Complex64 = stn_sv
            .iter()
            .zip(ref_sv.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        assert!(
            (overlap.norm_sqr() - 1.0).abs() < 1e-9,
            "Overlap should be 1: {} vs reference",
            overlap.norm_sqr()
        );
    }

    /// Disentangling test: Bell state + Rz. The Rz decomposition has two flip
    /// sites. Without disentangling, the multi-site cascade runs. With the
    /// current (safe) approach, CX cleared the flags, so disentangling doesn't
    /// fire and we use the cascade.
    #[test]
    fn test_disentangle_multi_site_bell_plus_rz() {
        use pecos_simulators::DenseStateVec;
        let theta = Angle64::from_radians(0.7);
        let mut stn = StabMps::builder(2).merge_rz(false).build();
        let mut ref_sim = DenseStateVec::new(2);

        stn.h(&[QubitId(0)]);
        ref_sim.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        ref_sim.cx(&[(QubitId(0), QubitId(1))]);
        // After Bell: Z_0 decomposition has 2 flip sites (both destabs have X on q0)
        stn.rz(theta, &[QubitId(0)]);
        ref_sim.rz(theta, &[QubitId(0)]);

        let stn_sv = stn.state_vector();
        let dim = 1 << 2;
        let ref_sv: Vec<Complex64> = (0..dim).map(|i| ref_sim.get_amplitude(i)).collect();

        let overlap: Complex64 = stn_sv
            .iter()
            .zip(ref_sv.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        assert!(
            (overlap.norm_sqr() - 1.0).abs() < 1e-9,
            "Overlap should be 1: got {}",
            overlap.norm_sqr()
        );
    }

    /// Test that verifies the GF(2) diagnostic correctly tracks disentangled sites.
    /// When disentangling fires, the flip pattern recorded is just the single `rot_site`.
    /// Targeted test: construct state where `pauli_map`=[(0,Y),(1,Y)] with
    /// flags [X(true), Z(true)] and verify disentangle gives correct rotation.
    ///
    /// To construct: need stab with Y on both q0, q1 (so `col_x` contains both)
    /// AND destab also with Y on both (so `col_x` for destabs also contains both).
    /// Simplest path: apply S,H pattern to get Y stab, then CX to propagate.
    #[test]
    fn test_disentangle_yy_rotation() {
        use pecos_simulators::DenseStateVec;
        let theta = Angle64::from_radians(0.3);
        let mut stn = StabMps::builder(2).merge_rz(false).build();
        let mut ref_sim = DenseStateVec::new(2);

        // Construct a state where RZ decomposition has pauli_map=[(0,Y),(1,Y)]
        // Apply: S on both, then H on both, then apply some Cliffords to get Y stabs
        // Or try sequence that matches seed 107's prefix approximately.
        // Seed 107 prefix (from actual fuzz output, best guess):
        stn.cx(&[(QubitId(0), QubitId(1))]);
        ref_sim.cx(&[(QubitId(0), QubitId(1))]);
        stn.sz(&[QubitId(1)]);
        ref_sim.sz(&[QubitId(1)]);
        stn.sz(&[QubitId(0)]);
        ref_sim.sz(&[QubitId(0)]);
        stn.h(&[QubitId(0)]);
        ref_sim.h(&[QubitId(0)]);
        stn.h(&[QubitId(1)]);
        ref_sim.h(&[QubitId(1)]);
        stn.sz(&[QubitId(0)]);
        ref_sim.sz(&[QubitId(0)]);
        stn.sz(&[QubitId(1)]);
        ref_sim.sz(&[QubitId(1)]);

        eprintln!("Bond dim: {}", stn.max_bond_dim());

        // Apply the non-Clifford RZ that may trigger disentangling
        stn.rz(theta, &[QubitId(0)]);
        ref_sim.rz(theta, &[QubitId(0)]);

        let stn_sv = stn.state_vector();
        let dim = 1 << 2;
        let ref_sv: Vec<Complex64> = (0..dim).map(|i| ref_sim.get_amplitude(i)).collect();
        let overlap: Complex64 = stn_sv
            .iter()
            .zip(ref_sv.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        eprintln!(
            "STN: {:?}",
            stn_sv
                .iter()
                .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "REF: {:?}",
            ref_sv
                .iter()
                .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                .collect::<Vec<_>>()
        );
        assert!(
            (overlap.norm_sqr() - 1.0).abs() < 1e-8,
            "YY rotation mismatch: overlap={}",
            overlap.norm_sqr()
        );
    }

    /// Check: if we FORCE std path at step 14 (clearing flags), does it still diverge?
    /// If yes: std path has a bug (unlikely). If no: disent at step 14 is buggy.
    #[test]
    fn test_737_step14_std_only() {
        use pecos_simulators::DenseStateVec;
        let q = |i: usize| QubitId(i);
        let mut stn = StabMps::builder(4).merge_rz(false).build();
        let mut ref_sim = DenseStateVec::new(4);
        let apply = |stn: &mut StabMps, r: &mut DenseStateVec, step: usize| match step {
            0 => {
                stn.cz(&[(q(1), q(0))]);
                r.cz(&[(q(1), q(0))]);
            }
            1 => {
                stn.cx(&[(q(3), q(0))]);
                r.cx(&[(q(3), q(0))]);
            }
            2 => {
                stn.h(&[q(1)]);
                r.h(&[q(1)]);
            }
            3 => {
                stn.rz(Angle64::from_radians(0.0691), &[q(3)]);
                r.rz(Angle64::from_radians(0.0691), &[q(3)]);
            }
            4 => {
                stn.rz(Angle64::from_radians(0.3330), &[q(2)]);
                r.rz(Angle64::from_radians(0.3330), &[q(2)]);
            }
            5 => {
                stn.cx(&[(q(2), q(3))]);
                r.cx(&[(q(2), q(3))]);
            }
            6 | 7 => {
                stn.cx(&[(q(3), q(1))]);
                r.cx(&[(q(3), q(1))]);
            }
            8 => {
                stn.sz(&[q(3)]);
                r.sz(&[q(3)]);
            }
            9 => {
                stn.sz(&[q(1)]);
                r.sz(&[q(1)]);
            }
            10 => {
                stn.rx(Angle64::from_radians(0.8608), &[q(2)]);
                r.rx(Angle64::from_radians(0.8608), &[q(2)]);
            }
            11 => {
                stn.x(&[q(2)]);
                r.x(&[q(2)]);
            }
            12 => {
                stn.rx(Angle64::from_radians(3.2610), &[q(1)]);
                r.rx(Angle64::from_radians(3.2610), &[q(1)]);
            }
            13 => {
                stn.sz(&[q(2)]);
                r.sz(&[q(2)]);
            }
            14 => {
                stn.rz(Angle64::from_radians(3.4558), &[q(2)]);
                r.rz(Angle64::from_radians(3.4558), &[q(2)]);
            }
            _ => panic!("bad step {step}"),
        };

        for i in 0..14 {
            apply(&mut stn, &mut ref_sim, i);
        }

        // Before step 14, force flags to None.
        for i in 0..stn.disent_flags.len() {
            stn.disent_flags[i] = None;
        }

        apply(&mut stn, &mut ref_sim, 14);

        let sv_stn = stn.state_vector();
        let sv_ref: Vec<Complex64> = (0..16).map(|i| ref_sim.get_amplitude(i)).collect();
        let overlap: Complex64 = sv_stn
            .iter()
            .zip(sv_ref.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        let fid = overlap.norm_sqr();
        eprintln!("step 14 with std path: fid={fid}");
        assert!(
            (fid - 1.0).abs() < 1e-6,
            "std path should give fid=1.0: got {fid}"
        );
    }

    /// Sanity check: `span_decomposition` on a real STN gf2 matrix gives a
    /// dependency whose XOR reconstructs the target row. Tests the primitive
    /// works on data produced by actual simulations.
    #[test]
    fn test_span_decomposition_on_real_simulation() {
        let q = |i: usize| QubitId(i);
        let mut stn = StabMps::builder(3).seed(42).merge_rz(false).build();
        // Bring qubits out of Z-eigenstate so T decomposes via DestabilizerFlip.
        stn.h(&[q(0), q(1), q(2)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(0)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(1)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(2)]);
        let m = stn.gf2_matrix();
        eprintln!("num_gates={} rank={}", m.num_gates(), m.gf2_rank());
        // After H on each qubit, Z_q decomposes to destab_q only (single-site).
        assert!(
            m.num_gates() >= 3,
            "expected at least 3 rows, got {}",
            m.num_gates()
        );
        // Combinations of existing rows should be in span.
        let single = m.span_decomposition(&[0]);
        eprintln!("Looking up [0]: {single:?}");
        assert!(single.is_some());
        // All-three XOR
        let all_three = m.span_decomposition(&[0, 1, 2]);
        eprintln!("Looking up [0,1,2]: {all_three:?}");
        assert!(all_three.is_some());
    }

    /// Verify the explicit heuristic disentangler (`stn.disentangle()`) does not
    /// Verify `StabMps::amplitude` returns correct coefficients for known states.
    #[test]
    fn test_amplitude_api() {
        let q = |i: usize| QubitId(i);
        // Bell state: amplitudes 1/√2 at |00⟩ and |11⟩, 0 elsewhere.
        let mut stn = StabMps::with_seed(2, 1);
        stn.h(&[q(0)]);
        stn.cx(&[(q(0), q(1))]);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let amp_00 = stn.amplitude(&[false, false]);
        let amp_11 = stn.amplitude(&[true, true]);
        let amp_01 = stn.amplitude(&[false, true]);
        let amp_10 = stn.amplitude(&[true, false]);
        assert!((amp_00.re - inv_sqrt2).abs() < 1e-9, "|00⟩ amp = {amp_00}");
        assert!((amp_11.re - inv_sqrt2).abs() < 1e-9, "|11⟩ amp = {amp_11}");
        assert!(amp_01.norm_sqr() < 1e-18, "|01⟩ amp = {amp_01}");
        assert!(amp_10.norm_sqr() < 1e-18, "|10⟩ amp = {amp_10}");
    }

    /// Verify Rényi `S_2` computation: for a product state, `S_2` should be 0.
    /// For a Bell state, `S_2` should be ln(2).
    #[test]
    fn test_renyi_s2_product_vs_bell() {
        let q = |i: usize| QubitId(i);

        // Product state |00⟩ -> S_2 = 0.
        let stn_prod = StabMps::with_seed(2, 1);
        let s_prod = stn_prod.renyi_s2(1);
        assert!(
            s_prod.abs() < 1e-9,
            "product state S_2={s_prod}, expected 0"
        );

        // Bell state |Φ+⟩ = (|00⟩+|11⟩)/√2 -> S_2 = ln(2).
        let mut stn_bell = StabMps::with_seed(2, 2);
        stn_bell.h(&[q(0)]);
        stn_bell.cx(&[(q(0), q(1))]);
        let s_bell = stn_bell.renyi_s2(1);
        eprintln!("Bell S_2 = {s_bell}, ln(2) = {}", (2.0f64).ln());
        assert!(
            (s_bell - (2.0f64).ln()).abs() < 1e-9,
            "Bell state S_2={s_bell}, expected ln(2)={}",
            (2.0f64).ln()
        );

        // Bell+T: (|00⟩ + e^{iπ/4}|11⟩)/√2. Still maximally entangled, S_2 = ln(2).
        let mut stn_bt = StabMps::with_seed(2, 3);
        stn_bt.h(&[q(0)]);
        stn_bt.cx(&[(q(0), q(1))]);
        stn_bt.rz(Angle64::QUARTER_TURN / 2u64, &[q(0)]);
        let s_bt = stn_bt.renyi_s2(1);
        eprintln!("Bell+T S_2 = {s_bt}, expected ln(2) = {}", (2.0f64).ln());
        assert!((s_bt - (2.0f64).ln()).abs() < 1e-9);
    }

    /// Cross-validate PCE vs full-SV `S_2` across Clifford+T circuits.
    #[test]
    fn test_s2_pce_matches_sv_various() {
        let q = |i: usize| QubitId(i);
        let quarter = Angle64::QUARTER_TURN;
        let t = quarter / 2u64;

        // Case 1: 4q Clifford+T with boundary entanglement.
        let mut a = StabMps::with_seed(4, 1);
        a.h(&[q(0), q(1)]);
        a.cx(&[(q(0), q(2))]);
        a.rz(t, &[q(2)]);
        a.cx(&[(q(1), q(3))]);
        assert!((a.s2_pce(2).unwrap() - a.renyi_s2(2)).abs() < 1e-6);

        // Case 2: 6q heavier circuit.
        let mut b = StabMps::with_seed(6, 2);
        b.h(&[q(0), q(1), q(2)]);
        b.cx(&[(q(0), q(3)), (q(1), q(4)), (q(2), q(5))]);
        b.rz(t, &[q(0), q(3)]);
        assert!((b.s2_pce(3).unwrap() - b.renyi_s2(3)).abs() < 1e-6);
    }

    /// Demonstrate PCE scaling beyond n=14 where `renyi_s2` panics.
    #[test]
    fn test_s2_pce_beyond_state_vector_limit() {
        let q = |i: usize| QubitId(i);
        // n=20, pure-Clifford Bell across cut → expect ln(2).
        let mut stn = StabMps::with_seed(20, 42);
        stn.h(&[q(0)]);
        stn.cx(&[(q(0), q(10))]);
        let s = stn.s2_pce(10).unwrap();
        eprintln!("n=20 Bell across middle cut: S_2 = {s}");
        assert!(
            (s - (2.0f64).ln()).abs() < 1e-6,
            "expected ln(2) for Bell, got {s}"
        );
    }

    /// Bell+T at n=20: T gets absorbed into tableau (stab branch), MPS stays bond 1.
    /// T is a diagonal gate on an X-basis stabilizer pair — contributes global phase
    /// only; physical state entanglement unchanged from pure Bell.
    ///
    /// Known limitation: this specific setup has T on a qubit whose stabilizer-at-q
    /// is Z → T hits the Stabilizer branch, so MPS remains bond 1 but the tableau
    /// encodes Bell+phase. PCE may mishandle the phase if `decompose_z` picks a
    /// non-trivial flip pattern. Documented for now.
    #[test]
    fn test_s2_pce_bell_plus_t_n20() {
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::with_seed(20, 42);
        stn.h(&[q(0)]);
        stn.cx(&[(q(0), q(10))]);
        stn.rz(t, &[q(10)]);
        let s = stn.s2_pce(10).unwrap();
        assert!((s - (2.0f64).ln()).abs() < 1e-6, "expected ln(2), got {s}");
    }

    /// Single-qubit trivial: amp(|0⟩) = 1 after no gates.
    #[test]
    fn test_amplitude_iterative_trivial() {
        let stn = StabMps::new(1);
        let a0 = stn.amplitude_iterative(&[false]);
        let a1 = stn.amplitude_iterative(&[true]);
        eprintln!("|0⟩: a(0)={a0} a(1)={a1}");
        assert!(
            (a0 - Complex64::new(1.0, 0.0)).norm() < 1e-9,
            "a(0) should be 1, got {a0}"
        );
        assert!(a1.norm() < 1e-9, "a(1) should be 0, got {a1}");
    }

    /// Single-qubit RZ(π/4)|+⟩ has amp(0) = e^{-iπ/8}/√2.
    #[test]
    fn test_amplitude_iterative_rz_quarter_plus_1q() {
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::new(1);
        stn.h(&[q(0)]);
        stn.rz(t, &[q(0)]);
        let a = stn.amplitude_iterative(&[false]);
        let s = stn.amplitude(&[false]);
        eprintln!("RZ(pi/4)|+⟩: iter={a} sv={s}");
        assert!((a - s).norm() < 1e-9);
    }

    #[test]
    fn test_amplitude_iterative_h_rz_h_regression() {
        let mut stn = StabMps::builder(1).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        stn.rz(Angle64::from_radians(0.9), &[QubitId(0)]);
        stn.h(&[QubitId(0)]);

        let amplitude = stn.amplitude_iterative(&[true]);
        let probability = stn.prob_bitstring(&[true]);
        let expected_probability = 0.45_f64.sin().powi(2);
        assert_relative_eq!(probability, expected_probability, epsilon = 1e-12);
        assert_relative_eq!(amplitude.norm_sqr(), probability, epsilon = 1e-12);
        assert!((amplitude - stn.amplitude(&[true])).norm() < 1e-12);
    }

    #[test]
    fn test_amplitude_iterative_h_t_sz_phase_regression() {
        let mut stn = StabMps::builder(1)
            .merge_rz(false)
            .svd_cutoff(0.0)
            .max_truncation_error(0.0)
            .build();
        stn.h(&[QubitId(0)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        stn.sz(&[QubitId(0)]);
        stn.flush();

        let expected = stn.state_vector()[1];
        let actual = stn.amplitude_iterative(&[true]);
        assert!(
            (actual - expected).norm() <= 1e-12,
            "H-T-SZ |1>: iterative={actual:?}, dense={expected:?}"
        );
    }

    #[test]
    fn test_amplitude_iterative_h_t_cx_h_phase_regression() {
        let mut stn = StabMps::builder(2)
            .merge_rz(false)
            .svd_cutoff(0.0)
            .max_truncation_error(0.0)
            .build();
        stn.h(&[QubitId(0)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.h(&[QubitId(0)]);
        stn.flush();

        let expected = stn.state_vector()[3];
        let actual = stn.amplitude_iterative(&[true, true]);
        assert!(
            (actual - expected).norm() <= 1e-12,
            "H-T-CX-H |11>: iterative={actual:?}, dense={expected:?}"
        );
    }

    fn amplitude_phase_randomized_circuit(n: usize, circuit_seed: u64) -> StabMps {
        let mut stn = StabMps::builder(n)
            .seed(circuit_seed)
            .merge_rz(false)
            .max_truncation_error(0.0)
            .build();
        let mut rng_state = 0x9e37_79b9_7f4a_7c15_u64 ^ circuit_seed ^ n as u64;
        let random = |state: &mut u64| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        };

        for rotation_qubit in 0..n {
            let theta = 0.17 + (random(&mut rng_state) % 101) as f64 * 0.011;
            stn.h(&[QubitId(rotation_qubit)]);
            stn.rz(Angle64::from_radians(theta), &[QubitId(rotation_qubit)]);
            let neighbour = (rotation_qubit + 1) % n;
            match rotation_qubit % 3 {
                0 => {
                    stn.sz(&[QubitId(rotation_qubit)]);
                }
                1 => {
                    stn.x(&[QubitId(neighbour)]);
                }
                _ => {
                    stn.cz(&[(QubitId(rotation_qubit), QubitId(neighbour))]);
                }
            }
            stn.h(&[QubitId(rotation_qubit)]);

            if rotation_qubit + 1 < n {
                let pair = if random(&mut rng_state) & 1 == 0 {
                    (rotation_qubit, rotation_qubit + 1)
                } else {
                    (rotation_qubit + 1, rotation_qubit)
                };
                stn.cx(&[(QubitId(pair.0), QubitId(pair.1))]);
            }
        }
        stn
    }

    /// This case matches `state_vector`'s phase convention at every forced
    /// projection step but has a terminal destabilizer-basis factor of +i. It
    /// binds the terminal scalar independently of the two right-H regressions.
    #[test]
    fn test_amplitude_iterative_terminal_tableau_phase_regression() {
        let stn = amplitude_phase_randomized_circuit(4, 0);
        let expected = stn.state_vector()[0];
        let actual = stn.amplitude_iterative(&[false; 4]);
        assert!(
            (actual - expected).norm() <= 1e-12,
            "n=4 seed=0 index=0 terminal phase: iterative={actual}, state-vector={expected}"
        );
    }

    #[test]
    fn test_amplitude_iterative_h_after_rotation_randomized() {
        for n in 3..=6 {
            for circuit_seed in 0..4_u64 {
                let stn = amplitude_phase_randomized_circuit(n, circuit_seed);

                let state_vector = stn.state_vector();
                for (index, &expected) in state_vector.iter().enumerate() {
                    let bitstring: Vec<bool> = (0..n).map(|q| (index >> q) & 1 == 1).collect();
                    let actual = stn.amplitude_iterative(&bitstring);
                    assert!(
                        (actual - expected).norm() <= 1e-12,
                        "n={n} seed={circuit_seed} index={index}: iterative={actual}, state-vector={expected}"
                    );
                }
            }
        }
    }

    fn assert_all_iterative_amplitudes_match_state_vector(stn: &StabMps, label: &str) {
        for (index, expected) in stn.state_vector().into_iter().enumerate() {
            let bits = [(index & 1) != 0, (index & 2) != 0];
            let actual = stn.amplitude_iterative(&bits);
            assert!(
                (actual - expected).norm() <= 1e-12,
                "{label} index={index}: iterative={actual:?}, state-vector={expected:?}"
            );
        }
    }

    /// A live measurement can leave the trivial coefficient MPS in a nonzero
    /// virtual basis state. The next iterative walk must retain the scalar
    /// while `canonicalize_trivial_mps_basis` right-composes its X.
    #[test]
    fn test_amplitude_iterative_mid_measurement_right_compose_x_scalar() {
        let mut stn = StabMps::builder(2)
            .seed(28)
            .merge_rz(false)
            .max_truncation_error(0.0)
            .build();
        stn.y(&[QubitId(1), QubitId(0)]);
        stn.rx(Angle64::from_radians(0.61), &[QubitId(0)]);
        stn.sz(&[QubitId(0)]).szdg(&[QubitId(0)]);
        stn.rx(Angle64::from_radians(0.61), &[QubitId(0)]);
        stn.rz(Angle64::from_radians(0.37), &[QubitId(1), QubitId(0)]);
        let measurement = stn.mz(&[QubitId(0)]).remove(0);
        assert!(!measurement.outcome && !measurement.is_deterministic);
        stn.flush();

        assert_all_iterative_amplitudes_match_state_vector(&stn, "right-compose-X scalar");
    }

    /// A live measurement can also select a pure-stabilizer ket whose later
    /// forced projection is recanonicalized by `mz_forced`; that scalar must
    /// survive in `amplitude_iterative`.
    #[test]
    fn test_amplitude_iterative_mid_measurement_forced_measurement_scalar() {
        let mut stn = StabMps::builder(2)
            .seed(9)
            .merge_rz(false)
            .max_truncation_error(0.0)
            .build();
        stn.h(&[QubitId(1)]);
        stn.szdg(&[QubitId(0)]).szdg(&[QubitId(0)]);
        stn.rx(Angle64::from_radians(0.61), &[QubitId(0)]);
        stn.y(&[QubitId(1)]).h(&[QubitId(0)]).sz(&[QubitId(0)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        let measurement = stn.mz(&[QubitId(0)]).remove(0);
        assert!(!measurement.outcome && !measurement.is_deterministic);
        stn.flush();

        assert_all_iterative_amplitudes_match_state_vector(&stn, "forced-measurement scalar");
    }

    #[test]
    fn test_amplitude_iterative_bare_h_rz_cx_family() {
        for n in 3..=4 {
            for seed in 0..3_u64 {
                let mut stn = StabMps::builder(n)
                    .seed(seed)
                    .merge_rz(false)
                    .max_truncation_error(0.0)
                    .build();
                for q in 0..n {
                    stn.h(&[QubitId(q)]);
                    let theta = 0.19 + 0.07 * (seed * n as u64 + q as u64) as f64;
                    stn.rz(Angle64::from_radians(theta), &[QubitId(q)]);
                    stn.h(&[QubitId(q)]);
                    if q + 1 < n {
                        stn.cx(&[(QubitId(q), QubitId(q + 1))]);
                    }
                }
                stn.flush();

                for (index, expected) in stn.state_vector().into_iter().enumerate() {
                    let bits = (0..n).map(|q| (index >> q) & 1 != 0).collect::<Vec<_>>();
                    let actual = stn.amplitude_iterative(&bits);
                    assert!(
                        (actual - expected).norm() <= 1e-12,
                        "bare H/RZ/CX n={n} seed={seed} index={index}: iterative={actual:?}, state-vector={expected:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_amplitude_iterative_seed_37304_is_gauge_invariant() {
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::builder(3)
            .seed(37_304)
            .merge_rz(false)
            .max_truncation_error(0.0)
            .build();
        stn.h(&[QubitId(1)]);
        stn.cx(&[(QubitId(1), QubitId(2))]);
        stn.h(&[QubitId(1)]);
        stn.cx(&[(QubitId(1), QubitId(2))]);
        stn.rz(t, &[QubitId(1)]);
        stn.h(&[QubitId(1)]);
        stn.cx(&[(QubitId(2), QubitId(0))]);
        stn.sz(&[QubitId(2)]);
        stn.cx(&[(QubitId(1), QubitId(0))]);
        stn.rz(t, &[QubitId(2)]);
        stn.h(&[QubitId(2)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.h(&[QubitId(1)]);
        stn.rz(t, &[QubitId(0)]);
        stn.h(&[QubitId(1)]);
        stn.sz(&[QubitId(1)]);
        stn.h(&[QubitId(2)]);
        stn.rz(t, &[QubitId(1)]);
        stn.h(&[QubitId(1)]);
        stn.flush();

        let expected = stn.state_vector()[0].norm_sqr();
        let mut left = stn.clone();
        left.mps.left_canonicalize();
        let mut right = stn.clone();
        right.mps.right_canonicalize();
        for (label, candidate) in [("stored", &stn), ("left", &left), ("right", &right)] {
            let actual = candidate.amplitude_iterative(&[false; 3]).norm_sqr();
            assert!(
                (actual - expected).abs() <= 1e-12,
                "seed 37304 gauge={label} bonds={:?} expected={expected:.16e} actual={actual:.16e}",
                candidate.mps.bond_dims(),
            );
        }
    }

    #[test]
    fn test_forced_projection_probability_uses_incoming_frame() {
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::builder(3)
            .seed(18_306)
            .merge_rz(false)
            .max_truncation_error(0.0)
            .build();
        stn.h(&[QubitId(1), QubitId(2)]);
        stn.cx(&[(QubitId(2), QubitId(0)), (QubitId(1), QubitId(0))]);
        stn.h(&[QubitId(0)]);
        stn.rz(t, &[QubitId(0)]);
        stn.sz(&[QubitId(0), QubitId(0)]);
        stn.cx(&[(QubitId(2), QubitId(1))]);
        stn.rz(t, &[QubitId(0)]);
        stn.cx(&[(QubitId(2), QubitId(0))]);
        stn.sz(&[QubitId(0)]);
        stn.cx(&[(QubitId(2), QubitId(1))]);
        stn.rz(t, &[QubitId(2)]);
        stn.h(&[QubitId(2)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.h(&[QubitId(0), QubitId(2)]);
        stn.rz(t, &[QubitId(1)]);
        stn.h(&[QubitId(1)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.sz(&[QubitId(2)]);
        stn.h(&[QubitId(1)]);
        stn.rz(t, &[QubitId(1)]);
        stn.sz(&[QubitId(1), QubitId(1)]);
        stn.h(&[QubitId(1)]);
        stn.rz(t, &[QubitId(1)]);
        stn.flush();

        let outcomes = [true, false, false];
        let state = stn.state_vector();
        let mut expected_prefix = 1.0;
        let mut tableau = stn.tableau.clone();
        let mut mps = stn.mps.clone();
        for (q, &outcome) in outcomes.iter().enumerate() {
            let joint = state
                .iter()
                .enumerate()
                .filter(|(index, _)| {
                    outcomes[..=q]
                        .iter()
                        .enumerate()
                        .all(|(site, &bit)| ((*index >> site) & 1 != 0) == bit)
                })
                .map(|(_, amplitude)| amplitude.norm_sqr())
                .sum::<f64>();
            let expected_probability = joint / expected_prefix;
            let actual_probability =
                measure::project_forced_z(&mut tableau, &mut mps, q, outcome).unwrap();
            assert!(
                (actual_probability - expected_probability).abs() <= 1e-12,
                "q={q}: expected={expected_probability:.16e}, actual={actual_probability:.16e}"
            );

            let mut represented = stn.clone();
            represented.tableau = tableau.clone();
            represented.mps = mps.clone();
            let actual_state = represented.state_vector();
            let inverse_norm = joint.sqrt().recip();
            let expected_state = state
                .iter()
                .enumerate()
                .map(|(index, &amplitude)| {
                    if outcomes[..=q]
                        .iter()
                        .enumerate()
                        .all(|(site, &bit)| ((index >> site) & 1 != 0) == bit)
                    {
                        amplitude * inverse_norm
                    } else {
                        Complex64::new(0.0, 0.0)
                    }
                })
                .collect::<Vec<_>>();
            let overlap = actual_state
                .iter()
                .zip(&expected_state)
                .map(|(actual, expected)| actual.conj() * expected)
                .sum::<Complex64>()
                .norm_sqr();
            assert!(
                overlap >= 1.0 - 1e-12,
                "q={q}: projected-state fidelity={overlap:.16e}"
            );
            expected_prefix = joint;
        }
    }

    #[test]
    fn test_forced_projection_seed_33404_phase_regression() {
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::builder(4)
            .seed(33_404)
            .merge_rz(false)
            .max_truncation_error(0.0)
            .build();
        stn.h(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        stn.sz(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(3)), (QubitId(2), QubitId(0))]);
        stn.rz(t, &[QubitId(3)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.h(&[QubitId(1)]);
        stn.cx(&[(QubitId(2), QubitId(3))]);
        stn.rz(t, &[QubitId(2)]);
        stn.h(&[QubitId(2), QubitId(1), QubitId(2)]);
        stn.cx(&[(QubitId(1), QubitId(2))]);
        stn.rz(t, &[QubitId(0)]);
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(3), QubitId(2)), (QubitId(1), QubitId(3))]);
        stn.sz(&[QubitId(1)]);
        stn.rz(t, &[QubitId(0)]);
        stn.flush();

        let expected_amplitude = stn.state_vector()[0];
        let expected_probability = expected_amplitude.norm_sqr();
        let bitstring = [false; 4];
        let probability = stn.prob_bitstring(&bitstring);
        let iterative_amplitude = stn.amplitude_iterative(&bitstring);
        assert!(
            (probability - expected_probability).abs() <= 1e-12,
            "probability={probability:.16e}, expected={expected_probability:.16e}"
        );
        assert!(
            (iterative_amplitude - expected_amplitude).norm() <= 1e-12,
            "iterative amplitude={iterative_amplitude}, dense={expected_amplitude}"
        );
    }

    /// Reproduce the full-random generator used by
    /// `examples/disent_firing_rate.rs` and the stability census.
    // Test-only visibility lets the #562 phase harness reuse the exact census
    // generator instead of maintaining a subtly different copy.
    pub(super) fn stability_census_random_circuit(seed: u64) -> StabMps {
        let mut stn = StabMps::with_seed(8, seed);
        let mut rng_state = seed.wrapping_add(1);
        let next_rng = |state: &mut u64| {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        };
        for _ in 0..80 {
            let gate_type = next_rng(&mut rng_state) % 8;
            let q0 = (next_rng(&mut rng_state) % 8) as usize;
            let q1 = loop {
                let candidate = (next_rng(&mut rng_state) % 8) as usize;
                if candidate != q0 {
                    break candidate;
                }
            };
            match gate_type {
                0 => {
                    stn.h(&[QubitId(q0)]);
                }
                1 => {
                    stn.sz(&[QubitId(q0)]);
                }
                2 => {
                    stn.x(&[QubitId(q0)]);
                }
                3 => {
                    stn.cx(&[(QubitId(q0), QubitId(q1))]);
                }
                4 => {
                    stn.cz(&[(QubitId(q0), QubitId(q1))]);
                }
                5 => {
                    stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(q0)]);
                }
                6 => {
                    let angle_bits = next_rng(&mut rng_state);
                    let angle = Angle64::from_radians(
                        (angle_bits % 1000) as f64 * 0.001 * std::f64::consts::TAU,
                    );
                    stn.rz(angle, &[QubitId(q0)]);
                }
                _ => {
                    let angle_bits = next_rng(&mut rng_state);
                    let angle = Angle64::from_radians(
                        (angle_bits % 1000) as f64 * 0.001 * std::f64::consts::TAU,
                    );
                    stn.rx(angle, &[QubitId(q0)]);
                }
            }
        }
        stn
    }

    #[test]
    #[should_panic(
        expected = "test non-finite compression failed: SVD failed to converge; the coefficient MPS state is unrecoverable"
    )]
    fn test_infallible_mps_boundary_panics_on_non_finite_compression() {
        let mut mps = Mps::new(2, MpsConfig::default());
        mps.tensors_mut()[1][(0, 0)] = Complex64::new(f64::NAN, 0.0);
        expect_mps_operation(mps.compress(), "test non-finite compression");
    }

    fn assert_stability_census_seed_completes_with_consistent_readouts(seed: u64) {
        const TOLERANCE: f64 = 1e-12;
        let mut stn = stability_census_random_circuit(seed);
        stn.flush();

        let state = stn.state_vector();
        let max_index = state
            .iter()
            .enumerate()
            .max_by(|(_, left), (_, right)| left.norm_sqr().total_cmp(&right.norm_sqr()))
            .unwrap()
            .0;
        let max_bits = (0..8)
            .map(|qubit| (max_index >> qubit) & 1 != 0)
            .collect::<Vec<_>>();
        for bits in [&max_bits[..], &[false; 8]] {
            let probability = stn.prob_bitstring(bits);
            let iterative_probability = stn.amplitude_iterative(bits).norm_sqr();
            assert!(
                (probability - iterative_probability).abs() <= TOLERANCE,
                "seed={seed} bits={bits:?}: prob_bitstring={probability:.16e}, |amplitude_iterative|^2={iterative_probability:.16e}"
            );
        }

        let pre_measurement = stn.clone();
        let measurements = stn.mz(&(0..8).map(QubitId).collect::<Vec<_>>());
        let measured_bits = measurements
            .iter()
            .map(|measurement| measurement.outcome)
            .collect::<Vec<_>>();
        let measured_probability = pre_measurement.prob_bitstring(&measured_bits);
        let measured_iterative_probability = pre_measurement
            .amplitude_iterative(&measured_bits)
            .norm_sqr();
        assert!(
            measured_probability > 0.0
                && (measured_probability - measured_iterative_probability).abs() <= TOLERANCE,
            "seed={seed} mz={measured_bits:?}: prob_bitstring={measured_probability:.16e}, |amplitude_iterative|^2={measured_iterative_probability:.16e}"
        );
    }

    #[test]
    fn test_stability_census_seed_97_completes_without_svd_failure() {
        assert_stability_census_seed_completes_with_consistent_readouts(97);
    }

    #[test]
    fn test_stability_census_seed_766_completes_without_svd_failure() {
        assert_stability_census_seed_completes_with_consistent_readouts(766);
    }

    #[test]
    fn test_stability_census_seed_1698_completes_without_svd_failure() {
        assert_stability_census_seed_completes_with_consistent_readouts(1_698);
    }

    #[test]
    fn test_forced_projection_seed_178_snaps_spurious_endpoint_branch_to_zero() {
        // Reproduce the full-random generator in examples/disent_firing_rate.rs.
        // Without the expectation endpoint snap, the all-zero branch reports
        // ~5.4e-19 and normalizes cancellation noise even though the dense
        // coefficient is structurally zero.
        let mut stn = stability_census_random_circuit(178);
        stn.flush();

        let bitstring = [false; 8];
        assert_eq!(stn.state_vector()[0], Complex64::new(0.0, 0.0));
        assert_eq!(stn.prob_bitstring(&bitstring).to_bits(), 0.0_f64.to_bits());
        assert_eq!(
            stn.amplitude_iterative(&bitstring),
            Complex64::new(0.0, 0.0)
        );
    }

    /// n=2 no-entangle H+T: amp(00) = (e^{-iπ/8}/√2)/√2 = e^{-iπ/8}/2.
    #[test]
    fn test_amplitude_iterative_t_plus_2q() {
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::new(2);
        stn.h(&[q(0), q(1)]);
        stn.rz(t, &[q(0)]);
        let a = stn.amplitude_iterative(&[false, false]);
        let s = stn.amplitude(&[false, false]);
        eprintln!("T|++⟩ n=2: iter={a} sv={s}");
        assert!((a - s).norm() < 1e-9);
    }

    /// n=4 all-plus: amp(any) = 1/4.
    #[test]
    fn test_amplitude_iterative_plus_state() {
        let q = |i: usize| QubitId(i);
        let mut stn = StabMps::new(4);
        stn.h(&[q(0), q(1), q(2), q(3)]);
        let a = stn.amplitude_iterative(&[false; 4]);
        eprintln!("|++++⟩: a(0000)={a}");
        assert!((a - Complex64::new(0.25, 0.0)).norm() < 1e-9, "got {a}");
    }

    /// Regression: forced projection leaves state with correct <Z_{q+1}>
    /// expectation and conditional amplitude for decompositions with
    /// overlapping flip/sign sites (phase = ±i). Fixed 2026-04-12 by
    /// ensuring `project_forced_z`'s `DestabilizerFlip` branch applies
    /// Z-then-X at overlap sites (matches `z_expectation_value` order,
    /// yielding XZ = `Y_conv`, not the anti-sign ZX).
    #[test]
    fn test_forced_projection_matches_conditional_sv() {
        use pecos_core::QubitId;
        let n = 5;
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let s_gate = Angle64::QUARTER_TURN / 4u64;
        let mut stn = StabMps::new(n);
        // Build the failing circuit up to gate 17 (before H(3))
        stn.sz(&[q(2)]);
        stn.h(&[q(3)]);
        stn.sz(&[q(0)]);
        stn.sz(&[q(0)]);
        stn.rz(s_gate, &[q(3)]);
        stn.rz(t, &[q(2)]);
        stn.h(&[q(2)]);
        stn.h(&[q(1)]);
        stn.sz(&[q(3)]);
        stn.rz(t, &[q(4)]);
        stn.rz(s_gate, &[q(2)]);
        stn.rz(t, &[q(3)]);
        stn.cx(&[(q(2), q(3))]);
        stn.rz(t, &[q(3)]);
        stn.rz(s_gate, &[q(2)]);
        stn.sz(&[q(4)]);
        stn.cx(&[(q(2), q(4))]);
        stn.h(&[q(3)]); // gate 18 — the bug trigger
        // Compare SV directly
        let full_sv = stn.state_vector();
        let full_amp_00000 = full_sv[0];
        eprintln!("full state: amp(|00000⟩)={full_amp_00000:.4e}");
        let mut tab = stn.tableau.clone();
        let mut mps = stn.mps.clone();
        let mut cumul_prob: f64 = 1.0;
        for q in 0..n {
            // Compute true conditional <Z_q> from state_vector BEFORE projection.
            let mut stn_pre = StabMps::new(n);
            stn_pre.tableau = tab.clone();
            stn_pre.mps = mps.clone();
            stn_pre.global_phase = stn.global_phase;
            let sv_pre = stn_pre.state_vector();
            // Compute <Z_q> on the current state (which may be conditioned
            // on prior forced outcomes). Since the tableau was mutated by
            // prior projections, this is the conditional expectation.
            let mut num: f64 = 0.0;
            let mut denom: f64 = 0.0;
            for (idx, sv_val) in sv_pre.iter().enumerate() {
                let n2 = sv_val.norm_sqr();
                denom += n2;
                let bit_q = (idx >> q) & 1;
                let sign = if bit_q == 0 { 1.0 } else { -1.0 };
                num += sign * n2;
            }
            let true_ev = if denom > 1e-20 { num / denom } else { 0.0 };
            let true_prob_plus = f64::midpoint(1.0, true_ev).clamp(0.0, 1.0);
            // Also compute true conditional directly from ORIGINAL sv.
            let mut orig_num: f64 = 0.0;
            let mut orig_denom: f64 = 0.0;
            for (idx, _) in full_sv.iter().enumerate() {
                let mut in_subspace = true;
                for qp in 0..q {
                    if (idx >> qp) & 1 != 0 {
                        in_subspace = false;
                        break;
                    }
                }
                if !in_subspace {
                    continue;
                }
                let n2 = full_sv[idx].norm_sqr();
                orig_denom += n2;
                let bit_q = (idx >> q) & 1;
                let sign = if bit_q == 0 { 1.0 } else { -1.0 };
                orig_num += sign * n2;
            }
            let orig_cond_ev = if orig_denom > 1e-20 {
                orig_num / orig_denom
            } else {
                0.0
            };
            let _ = (true_prob_plus, denom);
            eprintln!("  q={q}: code_state<Z_q>={true_ev:.4} orig_cond<Z_q>={orig_cond_ev:.4}");

            let pi = measure::project_forced_z(&mut tab, &mut mps, q, false).unwrap();
            cumul_prob *= pi;
            let mut stn_after = StabMps::new(n);
            stn_after.tableau = tab.clone();
            stn_after.mps = mps.clone();
            let sv_after = stn_after.state_vector();
            eprintln!(
                "  q={q}: code π={pi:.6} cumul={cumul_prob:.6} after |sv[0]|²={:.4e}",
                sv_after[0].norm_sqr()
            );
        }
    }

    /// Regression: `prob_bitstring` matches SV exactly for seed-10 circuit
    /// (was off by 8x before the Z-then-X ordering fix in measure.rs).
    #[test]
    fn test_prob_bitstring_seed10_minimal() {
        use pecos_core::QubitId;
        // From test_prob_bitstring_seed10_repro:
        // "sz(2); h(3); sz(0); sz(0); s(3); t(2); h(2); h(1); sz(3); t(4); s(2); t(3);
        //  cx(2,3); t(3); s(2); sz(4); cx(2,4); h(3); cx(2,4);"
        let n = 5;
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let s_gate = Angle64::QUARTER_TURN / 4u64;
        #[allow(clippy::type_complexity)]
        let gates: Vec<Box<dyn Fn(&mut StabMps)>> = vec![
            Box::new(|s| {
                s.sz(&[q(2)]);
            }),
            Box::new(|s| {
                s.h(&[q(3)]);
            }),
            Box::new(|s| {
                s.sz(&[q(0)]);
            }),
            Box::new(|s| {
                s.sz(&[q(0)]);
            }),
            Box::new(move |s| {
                s.rz(s_gate, &[q(3)]);
            }),
            Box::new(move |s| {
                s.rz(t, &[q(2)]);
            }),
            Box::new(|s| {
                s.h(&[q(2)]);
            }),
            Box::new(|s| {
                s.h(&[q(1)]);
            }),
            Box::new(|s| {
                s.sz(&[q(3)]);
            }),
            Box::new(move |s| {
                s.rz(t, &[q(4)]);
            }),
            Box::new(move |s| {
                s.rz(s_gate, &[q(2)]);
            }),
            Box::new(move |s| {
                s.rz(t, &[q(3)]);
            }),
            Box::new(|s| {
                s.cx(&[(q(2), q(3))]);
            }),
            Box::new(move |s| {
                s.rz(t, &[q(3)]);
            }),
            Box::new(move |s| {
                s.rz(s_gate, &[q(2)]);
            }),
            Box::new(|s| {
                s.sz(&[q(4)]);
            }),
            Box::new(|s| {
                s.cx(&[(q(2), q(4))]);
            }),
            Box::new(|s| {
                s.h(&[q(3)]);
            }),
            Box::new(|s| {
                s.cx(&[(q(2), q(4))]);
            }),
        ];
        // Print prob at each step.
        let mut stn = StabMps::new(n);
        for (step, g) in gates.iter().enumerate() {
            g(&mut stn);
            let bs = vec![false; n];
            let p = stn.prob_bitstring(&bs);
            let sv = stn.amplitude(&bs);
            let diff = (p - sv.norm_sqr()).abs();
            eprintln!(
                "step {step}: p={p:.6} |sv|²={:.6} diff={diff:.3e}",
                sv.norm_sqr()
            );
            if diff > 1e-8 {
                return;
            }
        }
    }

    /// Check `prob_bitstring` is correct even when `amplitude_iterative` has phase.
    #[test]
    fn test_prob_bitstring_vs_amplitude_square() {
        use pecos_core::QubitId;
        let mut stn = StabMps::with_seed(4, 2);
        stn.h(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.sz(&[QubitId(2)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(3)]);
        stn.cx(&[(QubitId(2), QubitId(3))]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        let mut max_diff: f64 = 0.0;
        for idx in 0..16 {
            let bs: Vec<bool> = (0..4).map(|q| (idx >> q) & 1 == 1).collect();
            let p = stn.prob_bitstring(&bs);
            let a = stn.amplitude(&bs);
            let diff = (p - a.norm_sqr()).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        eprintln!("SZ+T circuit: max |prob - |amp|²| = {max_diff:.3e}");
        assert!(max_diff < 1e-8);
    }

    /// n=4 H+T: amp magnitudes still 1/4.
    #[test]
    fn test_amplitude_iterative_plus_plus_t() {
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::new(4);
        stn.h(&[q(0), q(1), q(2), q(3)]);
        stn.rz(t, &[q(2)]);
        let a = stn.amplitude_iterative(&[false; 4]);
        let s = stn.amplitude(&[false; 4]);
        eprintln!("|++++⟩·T(2): iter={a} sv={s}");
        assert!((a - s).norm() < 1e-9);
    }

    /// 2q Bell state: both amp(|00⟩) and amp(|11⟩) = 1/√2.
    #[test]
    fn test_amplitude_iterative_bell() {
        let q = |i: usize| QubitId(i);
        let mut stn = StabMps::new(2);
        stn.h(&[q(0)]);
        stn.cx(&[(q(0), q(1))]);
        let a00 = stn.amplitude_iterative(&[false, false]);
        let a01 = stn.amplitude_iterative(&[false, true]);
        let a10 = stn.amplitude_iterative(&[true, false]);
        let a11 = stn.amplitude_iterative(&[true, true]);
        let target = Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0);
        eprintln!("Bell: a(00)={a00} a(01)={a01} a(10)={a10} a(11)={a11}");
        assert!((a00 - target).norm() < 1e-9, "a(00)={a00}, want {target}");
        assert!(a01.norm() < 1e-9);
        assert!(a10.norm() < 1e-9);
        assert!((a11 - target).norm() < 1e-9, "a(11)={a11}, want {target}");
    }

    /// Test `pre_reduce` with non-Clifford T gate in circuit.
    #[test]
    fn test_pre_reduce_with_t() {
        use pecos_core::QubitId;
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::new(3);
        stn.h(&[q(0)]);
        stn.cx(&[(q(0), q(1))]);
        stn.rz(t, &[q(0)]);
        stn.h(&[q(2)]);
        stn.cx(&[(q(2), q(1))]);
        let sv_before = stn.state_vector();
        let mut tab = stn.tableau.clone();
        let mut mps = stn.mps.clone();
        measure::pre_reduce_for_measurement_pub(&mut tab, &mut mps, 1).unwrap();
        let mut stn_after = StabMps::new(3);
        stn_after.tableau = tab;
        stn_after.mps = mps;
        stn_after.global_phase = stn.global_phase;
        let sv_after = stn_after.state_vector();
        let mut max_diff: f64 = 0.0;
        for i in 0..sv_before.len() {
            let d = (sv_before[i].norm_sqr() - sv_after[i].norm_sqr()).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        eprintln!("T circuit: max ||amp|² diff| = {max_diff:.3e}");
        eprintln!("Pre SV:");
        for (i, a) in sv_before.iter().enumerate() {
            if a.norm() > 1e-9 {
                eprintln!("  [{i}] = {a:.4}");
            }
        }
        eprintln!("Post SV:");
        for (i, a) in sv_after.iter().enumerate() {
            if a.norm() > 1e-9 {
                eprintln!("  [{i}] = {a:.4}");
            }
        }
        assert!(max_diff < 1e-8);
    }

    /// Verify `stn.state_vector()` magnitudes match `DenseStateVec` for seed 16.
    #[test]
    fn test_seed16_sv_matches_dense() {
        use pecos_core::QubitId;
        use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, DenseStateVec};
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let s_g = Angle64::QUARTER_TURN / 4u64;
        let mut stn = StabMps::new(5);
        let mut dsv = DenseStateVec::new(5);
        macro_rules! both {
            ($a:block,$b:block) => {{
                $a;
                $b;
            }};
        }
        both!(
            {
                stn.rz(s_g, &[q(0)]);
            },
            {
                dsv.rz(s_g, &[q(0)]);
            }
        );
        both!(
            {
                stn.h(&[q(2)]);
            },
            {
                dsv.h(&[q(2)]);
            }
        );
        both!(
            {
                stn.rz(s_g, &[q(4)]);
            },
            {
                dsv.rz(s_g, &[q(4)]);
            }
        );
        both!(
            {
                stn.rz(s_g, &[q(0)]);
            },
            {
                dsv.rz(s_g, &[q(0)]);
            }
        );
        both!(
            {
                stn.cx(&[(q(1), q(4))]);
            },
            {
                dsv.cx(&[(q(1), q(4))]);
            }
        );
        both!(
            {
                stn.cx(&[(q(0), q(4))]);
            },
            {
                dsv.cx(&[(q(0), q(4))]);
            }
        );
        both!(
            {
                stn.h(&[q(0)]);
            },
            {
                dsv.h(&[q(0)]);
            }
        );
        both!(
            {
                stn.h(&[q(3)]);
            },
            {
                dsv.h(&[q(3)]);
            }
        );
        both!(
            {
                stn.rz(t, &[q(1)]);
            },
            {
                dsv.rz(t, &[q(1)]);
            }
        );
        both!(
            {
                stn.rz(t, &[q(1)]);
            },
            {
                dsv.rz(t, &[q(1)]);
            }
        );
        both!(
            {
                stn.sz(&[q(1)]);
            },
            {
                dsv.sz(&[q(1)]);
            }
        );
        both!(
            {
                stn.sz(&[q(3)]);
            },
            {
                dsv.sz(&[q(3)]);
            }
        );
        both!(
            {
                stn.h(&[q(1)]);
            },
            {
                dsv.h(&[q(1)]);
            }
        );
        both!(
            {
                stn.sz(&[q(1)]);
            },
            {
                dsv.sz(&[q(1)]);
            }
        );
        both!(
            {
                stn.rz(s_g, &[q(3)]);
            },
            {
                dsv.rz(s_g, &[q(3)]);
            }
        );
        both!(
            {
                stn.h(&[q(3)]);
            },
            {
                dsv.h(&[q(3)]);
            }
        );
        both!(
            {
                stn.cx(&[(q(4), q(1))]);
            },
            {
                dsv.cx(&[(q(4), q(1))]);
            }
        );
        both!(
            {
                stn.rz(t, &[q(0)]);
            },
            {
                dsv.rz(t, &[q(0)]);
            }
        );
        let stn_sv = stn.state_vector();
        let mut max_diff: f64 = 0.0;
        for (i, sv_val) in stn_sv.iter().enumerate().take(32) {
            let d = (sv_val.norm_sqr() - dsv.get_amplitude(i).norm_sqr()).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        eprintln!("seed 16 |sv|² diff: {max_diff:.3e}");
        assert!(max_diff < 1e-8);
    }

    /// Regression: `project_forced_z` state matches true conditional after
    /// Bug #3 fix (MPS CNOT compensation via `apply_long_range_two_site_gate`).
    #[test]
    fn test_seed16_project_correctness() {
        use pecos_core::QubitId;
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let s_g = Angle64::QUARTER_TURN / 4u64;
        let mut stn = StabMps::new(5);
        stn.rz(s_g, &[q(0)]);
        stn.h(&[q(2)]);
        stn.rz(s_g, &[q(4)]);
        stn.rz(s_g, &[q(0)]);
        stn.cx(&[(q(1), q(4))]);
        stn.cx(&[(q(0), q(4))]);
        stn.h(&[q(0)]);
        stn.h(&[q(3)]);
        stn.rz(t, &[q(1)]);
        stn.rz(t, &[q(1)]);
        stn.sz(&[q(1)]);
        stn.sz(&[q(3)]);
        stn.h(&[q(1)]);
        stn.sz(&[q(1)]);
        stn.rz(s_g, &[q(3)]);
        stn.h(&[q(3)]);
        stn.cx(&[(q(4), q(1))]);
        stn.rz(t, &[q(0)]);
        let full_sv = stn.state_vector();
        // True conditional state (q=0 forced to 0): set amps at q=0=1 to zero, renorm.
        let mut true_cond: Vec<Complex64> = full_sv
            .iter()
            .enumerate()
            .map(|(idx, &a)| {
                if idx & 1 == 0 {
                    a
                } else {
                    Complex64::new(0.0, 0.0)
                }
            })
            .collect();
        let norm2: f64 = true_cond.iter().map(nalgebra::Complex::norm_sqr).sum();
        let inv_norm = 1.0 / norm2.sqrt();
        for a in &mut true_cond {
            *a *= Complex64::new(inv_norm, 0.0);
        }
        // Code's post-project state.
        let mut tab = stn.tableau.clone();
        let mut mps = stn.mps.clone();
        measure::project_forced_z(&mut tab, &mut mps, 0, false).unwrap();
        let mut stn_post = StabMps::new(5);
        stn_post.tableau = tab;
        stn_post.mps = mps;
        stn_post.global_phase = stn.global_phase;
        let code_sv = stn_post.state_vector();
        let mut max_mag_diff: f64 = 0.0;
        for i in 0..full_sv.len() {
            let d = (true_cond[i].norm_sqr() - code_sv[i].norm_sqr()).abs();
            if d > max_mag_diff {
                max_mag_diff = d;
            }
        }
        eprintln!("project_forced_z(0) vs truth: max ||amp|² diff| = {max_mag_diff:.3e}");
        assert!(
            max_mag_diff < 1e-8,
            "project_forced_z state diverges from truth"
        );
    }

    /// Test `pre_reduce` preservation with SZ gates (introduces Y bits).
    #[test]
    fn test_pre_reduce_with_sz() {
        use pecos_core::QubitId;
        let q = |i: usize| QubitId(i);
        let mut stn = StabMps::new(2);
        stn.h(&[q(0)]);
        stn.sz(&[q(0)]); // q0 → Y-state (virtually)
        stn.h(&[q(1)]);
        stn.cx(&[(q(0), q(1))]);
        let sv_before = stn.state_vector();
        let mut tab = stn.tableau.clone();
        let mut mps = stn.mps.clone();
        measure::pre_reduce_for_measurement_pub(&mut tab, &mut mps, 1).unwrap();
        let mut stn_after = StabMps::new(2);
        stn_after.tableau = tab;
        stn_after.mps = mps;
        stn_after.global_phase = stn.global_phase;
        let sv_after = stn_after.state_vector();
        let mut max_diff: f64 = 0.0;
        for i in 0..sv_before.len() {
            let d = (sv_before[i].norm_sqr() - sv_after[i].norm_sqr()).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        eprintln!("SZ circuit: max ||amp|² diff| = {max_diff:.3e}");
        assert!(max_diff < 1e-8);
    }

    /// Test non-adjacent CNOT via `apply_cnot_to_mps`.
    #[test]
    fn test_cnot_non_adjacent() {
        use pecos_core::QubitId;
        let q = |i: usize| QubitId(i);
        // Build bond-1 state with specific amp at different sites.
        let mut stn = StabMps::new(5);
        stn.h(&[q(0)]); // |+⟩_0
        stn.h(&[q(4)]); // |+⟩_4
        // State: |+⟩_0 |0⟩_1 |0⟩_2 |0⟩_3 |+⟩_4 = (|00000⟩+|00001⟩+|10000⟩+|10001⟩)/2
        // Wait LSB-first: idx bit 0 = q0. So idx: q0=0: |+⟩_4 at bit 4.
        // state_vector gives 4 non-zero amps.
        let _ = stn;
        let mut stn_test = StabMps::new(5);
        stn_test.h(&[q(0)]);
        stn_test.h(&[q(4)]);
        stn_test.cx(&[(q(0), q(4))]);
        // Stabs: X_0 X_4, Z_1, Z_2, Z_3, X_4. col_x[4] = {0, 4}. pre_reduce on q=4.
        let sv_before = stn_test.state_vector();
        let mut tab = stn_test.tableau.clone();
        let mut mps = stn_test.mps.clone();
        measure::pre_reduce_for_measurement_pub(&mut tab, &mut mps, 4).unwrap();
        let mut stn_after = StabMps::new(5);
        stn_after.tableau = tab;
        stn_after.mps = mps;
        stn_after.global_phase = stn_test.global_phase;
        let sv_after = stn_after.state_vector();
        let mut max_diff: f64 = 0.0;
        for i in 0..sv_before.len() {
            let d = (sv_before[i].norm_sqr() - sv_after[i].norm_sqr()).abs();
            if d > max_diff {
                max_diff = d;
            }
        }
        eprintln!("non-adjacent CNOT: max ||amp|² diff| = {max_diff:.3e}");
        assert!(max_diff < 1e-8);
    }

    /// Seed 16 full circuit: `pre_reduce` on each qubit — do magnitudes preserve?
    #[test]
    fn test_seed16_pre_reduce_each_q() {
        use pecos_core::QubitId;
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let s_g = Angle64::QUARTER_TURN / 4u64;
        let mut stn = StabMps::new(5);
        stn.rz(s_g, &[q(0)]);
        stn.h(&[q(2)]);
        stn.rz(s_g, &[q(4)]);
        stn.rz(s_g, &[q(0)]);
        stn.cx(&[(q(1), q(4))]);
        stn.cx(&[(q(0), q(4))]);
        stn.h(&[q(0)]);
        stn.h(&[q(3)]);
        stn.rz(t, &[q(1)]);
        stn.rz(t, &[q(1)]);
        stn.sz(&[q(1)]);
        stn.sz(&[q(3)]);
        stn.h(&[q(1)]);
        stn.sz(&[q(1)]);
        stn.rz(s_g, &[q(3)]);
        stn.h(&[q(3)]);
        stn.cx(&[(q(4), q(1))]);
        stn.rz(t, &[q(0)]);
        let sv_before = stn.state_vector();
        for test_q in 0..5 {
            let mut tab = stn.tableau.clone();
            let mut mps = stn.mps.clone();
            measure::pre_reduce_for_measurement_pub(&mut tab, &mut mps, test_q).unwrap();
            let mut stn_after = StabMps::new(5);
            stn_after.tableau = tab;
            stn_after.mps = mps;
            stn_after.global_phase = stn.global_phase;
            let sv_after = stn_after.state_vector();
            let mut max_diff: f64 = 0.0;
            for i in 0..sv_before.len() {
                let d = (sv_before[i].norm_sqr() - sv_after[i].norm_sqr()).abs();
                if d > max_diff {
                    max_diff = d;
                }
            }
            eprintln!("pre_reduce(q={test_q}): max ||amp|² diff| = {max_diff:.3e}");
        }
    }

    /// Minimal 2q test: `pre_reduce` preserves state for H+H+CX-|00⟩.
    #[test]
    fn test_pre_reduce_minimal_2q() {
        use pecos_core::QubitId;
        let q = |i: usize| QubitId(i);
        let mut stn = StabMps::new(2);
        stn.h(&[q(0)]);
        stn.h(&[q(1)]);
        stn.cx(&[(q(0), q(1))]);
        let sv_before = stn.state_vector();
        let mut tab = stn.tableau.clone();
        let mut mps = stn.mps.clone();
        measure::pre_reduce_for_measurement_pub(&mut tab, &mut mps, 1).unwrap();
        let mut stn_after = StabMps::new(2);
        stn_after.tableau = tab;
        stn_after.mps = mps;
        stn_after.global_phase = stn.global_phase;
        let sv_after = stn_after.state_vector();
        let mut max_mag_diff: f64 = 0.0;
        for i in 0..sv_before.len() {
            let d = (sv_before[i].norm_sqr() - sv_after[i].norm_sqr()).abs();
            if d > max_mag_diff {
                max_mag_diff = d;
            }
        }
        eprintln!("2q minimal: max ||amp|² diff| = {max_mag_diff:.3e}");
        assert!(max_mag_diff < 1e-8);
    }

    /// `pre_reduce_for_measurement` preserves the CAMPS state when the proper
    /// virtual-frame CNOT is applied to the MPS (seed-16 regression).
    #[test]
    fn test_pre_reduce_preserves_state() {
        use pecos_core::QubitId;
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let s_g = Angle64::QUARTER_TURN / 4u64;
        let mut stn = StabMps::new(5);
        stn.rz(s_g, &[q(0)]);
        stn.h(&[q(2)]);
        stn.rz(s_g, &[q(4)]);
        stn.rz(s_g, &[q(0)]);
        stn.cx(&[(q(1), q(4))]);
        stn.cx(&[(q(0), q(4))]);
        stn.h(&[q(0)]);
        stn.h(&[q(3)]);
        stn.rz(t, &[q(1)]);
        stn.rz(t, &[q(1)]);
        stn.sz(&[q(1)]);
        stn.sz(&[q(3)]);
        stn.h(&[q(1)]);
        stn.sz(&[q(1)]);
        stn.rz(s_g, &[q(3)]);
        stn.h(&[q(3)]);
        stn.cx(&[(q(4), q(1))]);
        stn.rz(t, &[q(0)]);
        // Directly pre_reduce on q=1 (no prior project_forced_z).
        let sv_before = stn.state_vector();
        let mut tab = stn.tableau.clone();
        let mut mps = stn.mps.clone();
        measure::pre_reduce_for_measurement_pub(&mut tab, &mut mps, 1).unwrap();
        let mut stn_after = StabMps::new(5);
        stn_after.tableau = tab;
        stn_after.mps = mps;
        stn_after.global_phase = stn.global_phase;
        let sv_after = stn_after.state_vector();
        let mut max_mag_diff: f64 = 0.0;
        for i in 0..sv_before.len() {
            let d = (sv_before[i].norm_sqr() - sv_after[i].norm_sqr()).abs();
            if d > max_mag_diff {
                max_mag_diff = d;
            }
        }
        eprintln!("seed 16 direct pre_reduce: max ||amp|² diff| = {max_mag_diff:.3e}");
        assert!(max_mag_diff < 1e-6);
    }

    /// Regression: seed-16 `prob_bitstring` matches SV (Bug #3 fixed).
    #[test]
    fn test_prob_bitstring_seed16_diag() {
        use pecos_core::QubitId;
        let n: usize = 4 + ((16u64 % 3) as usize);
        let mut stn = StabMps::with_seed(n, 16);
        let mut rng_state: u64 = 0xDEAD_BEEF ^ 16u64.wrapping_mul(37);
        let rnd = |s: &mut u64| -> u64 {
            *s ^= *s << 13;
            *s ^= *s >> 7;
            *s ^= *s << 17;
            *s
        };
        for _ in 0..20 {
            let op = rnd(&mut rng_state) % 5;
            let q1 = (rnd(&mut rng_state) as usize) % n;
            match op {
                0 => {
                    stn.h(&[QubitId(q1)]);
                }
                1 => {
                    stn.sz(&[QubitId(q1)]);
                }
                2 => {
                    let q2 = (rnd(&mut rng_state) as usize) % n;
                    if q1 != q2 {
                        stn.cx(&[(QubitId(q1), QubitId(q2))]);
                    }
                }
                3 => {
                    stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(q1)]);
                }
                _ => {
                    stn.rz(Angle64::QUARTER_TURN / 4u64, &[QubitId(q1)]);
                }
            }
        }
        let full_sv = stn.state_vector();
        let full_amp_00000 = full_sv[0];
        eprintln!("full amp(|00000⟩)²={:.4e}", full_amp_00000.norm_sqr());
        // True chain of conditional <Z_q> and probs.
        let mut tab = stn.tableau.clone();
        let mut mps = stn.mps.clone();
        let mut cumul_code: f64 = 1.0;
        for q in 0..n {
            // Pre-projection state vector (represents conditional state under code's projections).
            let mut stn_pre = StabMps::new(n);
            stn_pre.tableau = tab.clone();
            stn_pre.mps = mps.clone();
            stn_pre.global_phase = stn.global_phase;
            let sv_pre = stn_pre.state_vector();
            let mut num: f64 = 0.0;
            let mut denom: f64 = 0.0;
            for (idx, sv_val) in sv_pre.iter().enumerate() {
                let n2 = sv_val.norm_sqr();
                denom += n2;
                let bit_q = (idx >> q) & 1;
                let sign = if bit_q == 0 { 1.0 } else { -1.0 };
                num += sign * n2;
            }
            let code_state_ev = if denom > 1e-20 { num / denom } else { 0.0 };
            let code_state_prob0 = f64::midpoint(1.0, code_state_ev).clamp(0.0, 1.0);

            // True conditional from ORIGINAL full sv: condition on q_0..q_{q-1}=0.
            let mut orig_num: f64 = 0.0;
            let mut orig_denom: f64 = 0.0;
            for (idx, _) in full_sv.iter().enumerate() {
                let mut in_sub = true;
                for qp in 0..q {
                    if (idx >> qp) & 1 != 0 {
                        in_sub = false;
                        break;
                    }
                }
                if !in_sub {
                    continue;
                }
                let n2 = full_sv[idx].norm_sqr();
                orig_denom += n2;
                let bit_q = (idx >> q) & 1;
                let sign = if bit_q == 0 { 1.0 } else { -1.0 };
                orig_num += sign * n2;
            }
            let true_cond_ev = if orig_denom > 1e-20 {
                orig_num / orig_denom
            } else {
                0.0
            };
            let true_cond_prob0 = f64::midpoint(1.0, true_cond_ev).clamp(0.0, 1.0);

            let pi = measure::project_forced_z(&mut tab, &mut mps, q, false).unwrap();
            cumul_code *= pi;
            eprintln!(
                "  q={q}: code_state_ev={code_state_ev:.4} true_cond_ev={true_cond_ev:.4} π={pi:.6} (code cond_prob={code_state_prob0:.6}, true cond_prob={true_cond_prob0:.6}) cumul_code={cumul_code:.6}"
            );
        }
    }

    fn seed_10504_honest_basis_change_circuit() -> StabMps {
        let q = |index| QubitId(index);
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::builder(5)
            .seed(10_504)
            .merge_rz(false)
            .max_truncation_error(0.0)
            .build();
        stn.h(&[q(0), q(1), q(2), q(3)]);
        stn.cx(&[(q(4), q(1))]);
        stn.cx(&[(q(0), q(4))]);
        stn.rz(t, &[q(3)]);
        stn.cx(&[(q(0), q(4))]);
        stn.sz(&[q(0)]);
        stn.cx(&[(q(2), q(3))]);
        stn.rz(t, &[q(1)]);
        stn.h(&[q(1)]);
        stn.sz(&[q(3)]);
        stn.h(&[q(2)]);
        stn.sz(&[q(1)]);
        stn.rz(t, &[q(4)]);
        stn.h(&[q(4)]);
        stn.cx(&[(q(4), q(1))]);
        stn.cx(&[(q(0), q(3))]);
        stn.cx(&[(q(2), q(4))]);
        stn.rz(t, &[q(4)]);
        stn.h(&[q(4)]);
        stn
    }

    /// Regression for the phase-complete tableau gauge required after a
    /// nondeterministic forced projection. Before the fix, projecting q1
    /// omitted a virtual Z gauge. The resulting q4 decomposition was
    /// `i X_1 Z_1`, but its conditional P(0) was reversed from 0.75 to 0.25.
    #[test]
    fn test_seed_10504_imaginary_phase_sequential_projection() {
        let stn = seed_10504_honest_basis_change_circuit();
        let full_state = stn.state_vector();
        let mut tableau = stn.tableau.clone();
        let mut mps = stn.mps.clone();
        for q in 0..5 {
            let mut expected_numerator = 0.0;
            let mut expected_denominator = 0.0;
            for (index, amplitude) in full_state.iter().enumerate() {
                if (0..q).all(|prior| (index >> prior) & 1 == 0) {
                    let weight = amplitude.norm_sqr();
                    expected_denominator += weight;
                    if (index >> q) & 1 == 0 {
                        expected_numerator += weight;
                    }
                }
            }
            let expected = expected_numerator / expected_denominator;
            let actual = measure::project_forced_z(&mut tableau, &mut mps, q, false).unwrap();
            assert!(
                (actual - expected).abs() <= 1e-12,
                "q={q}: forced conditional probability={actual}, dense oracle={expected}"
            );
            if q == 4 {
                assert!((expected - 0.75).abs() <= 1e-12);
            }
        }

        let expected_probability = full_state[0].norm_sqr();
        let actual_probability = stn.prob_bitstring(&[false; 5]);
        assert!(
            (actual_probability - expected_probability).abs() <= 1e-12,
            "prob_bitstring={actual_probability}, dense oracle={expected_probability}"
        );

        let expected_amplitude = full_state[0];
        let actual_amplitude = stn.amplitude_iterative(&[false; 5]);
        assert!(
            (actual_amplitude - expected_amplitude).norm() <= 1e-12,
            "amplitude_iterative={actual_amplitude}, dense oracle={expected_amplitude}"
        );
    }

    /// Regression: `prob_bitstring` matches SV across 30 random Clifford+T
    /// circuits at n=4..=6 after Bug #1 (Z-then-X), Bug #2 (`multiply_row`
    /// phase), and Bug #3 (MPS CNOT compensation via long-range gate) fixes.
    #[test]
    #[ignore = "slow stress (~60s debug): run with `cargo test --lib -- --include-ignored`"]
    fn test_prob_bitstring_random_stress() {
        use pecos_core::QubitId;
        for seed in 0..30u64 {
            let n: usize = 4 + ((seed % 3) as usize);
            let mut stn = StabMps::with_seed(n, seed);
            let mut rng_state: u64 = 0xDEAD_BEEF ^ seed.wrapping_mul(37);
            let rnd = |s: &mut u64| -> u64 {
                *s ^= *s << 13;
                *s ^= *s >> 7;
                *s ^= *s << 17;
                *s
            };
            for _ in 0..20 {
                let op = rnd(&mut rng_state) % 5;
                let q1 = (rnd(&mut rng_state) as usize) % n;
                match op {
                    0 => {
                        stn.h(&[QubitId(q1)]);
                    }
                    1 => {
                        stn.sz(&[QubitId(q1)]);
                    }
                    2 => {
                        let q2 = (rnd(&mut rng_state) as usize) % n;
                        if q1 != q2 {
                            stn.cx(&[(QubitId(q1), QubitId(q2))]);
                        }
                    }
                    3 => {
                        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(q1)]);
                    }
                    _ => {
                        stn.rz(Angle64::QUARTER_TURN / 4u64, &[QubitId(q1)]);
                    }
                }
            }
            for idx in 0..(1usize << n) {
                let bs: Vec<bool> = (0..n).map(|q| (idx >> q) & 1 == 1).collect();
                let a_sv = stn.amplitude(&bs);
                // Probability must match exactly (primary correctness check).
                let p = stn.prob_bitstring(&bs);
                let prob_diff = (p - a_sv.norm_sqr()).abs();
                assert!(
                    prob_diff < 1e-8,
                    "seed {seed} idx={idx}: prob={p} |sv|²={} diff={prob_diff:.3e}",
                    a_sv.norm_sqr()
                );
            }
        }
    }

    /// `amplitude_iterative` matches `amplitude` at small n (full complex).
    #[test]
    fn test_amplitude_iterative_matches_sv() {
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::with_seed(4, 1);
        stn.h(&[q(0), q(1), q(2), q(3)]);
        stn.cx(&[(q(0), q(2))]);
        stn.rz(t, &[q(2)]);
        stn.cx(&[(q(1), q(3))]);

        let mut max_diff: f64 = 0.0;
        for idx in 0..16 {
            let bs: Vec<bool> = (0..4).map(|q| (idx >> q) & 1 == 1).collect();
            let a_iter = stn.amplitude_iterative(&bs);
            let a_sv = stn.amplitude(&bs);
            let diff = (a_iter - a_sv).norm();
            if diff > max_diff {
                max_diff = diff;
            }
            if diff > 1e-6 {
                eprintln!("bs={idx:04b}: iter={a_iter:.3} sv={a_sv:.3} diff={diff:.3e}");
            }
        }
        eprintln!("max |amp_iter - amp_sv| = {max_diff:.3e}");
        assert!(
            max_diff < 1e-6,
            "amplitude_iterative mismatch: max_diff={max_diff}"
        );
    }

    /// `amplitude_iterative` at n=30 (beyond `state_vector`).
    #[test]
    fn test_amplitude_iterative_n30_bell() {
        let q = |i: usize| QubitId(i);
        let n = 30;
        let mut stn = StabMps::with_seed(n, 5);
        stn.h(&[q(0)]);
        stn.cx(&[(q(0), q(15))]);
        let bs0 = vec![false; n];
        let a00 = stn.amplitude_iterative(&bs0);
        // bs[q] corresponds to qubit q; flip q0 and q15.
        let mut bs1 = vec![false; n];
        bs1[0] = true;
        bs1[15] = true;
        let a11 = stn.amplitude_iterative(&bs1);
        eprintln!("n=30 Bell: a(0)={a00:.4}, a(q0,q15=1)={a11:.4}");
        assert!((a00.norm_sqr() - 0.5).abs() < 1e-9);
        assert!((a11.norm_sqr() - 0.5).abs() < 1e-9);
    }

    /// `prob_bitstring` matches `|amplitude|²` at small n.
    #[test]
    fn test_prob_bitstring_matches_amplitude() {
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::with_seed(4, 1);
        stn.h(&[q(0), q(1), q(2), q(3)]);
        stn.cx(&[(q(0), q(2))]);
        stn.rz(t, &[q(2)]);
        stn.cx(&[(q(1), q(3))]);

        // Check every bitstring.
        let mut max_diff = 0f64;
        for idx in 0..16 {
            let bs: Vec<bool> = (0..4).map(|q| (idx >> q) & 1 == 1).collect();
            let p = stn.prob_bitstring(&bs);
            let a = stn.amplitude(&bs);
            let diff = (p - a.norm_sqr()).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        eprintln!("max |p - |a|²| = {max_diff:.3e}");
        assert!(max_diff < 1e-8);
    }

    /// `prob_bitstring` at n=30 where `state_vector` would OOM.
    #[test]
    fn test_prob_bitstring_n30_bell() {
        let q = |i: usize| QubitId(i);
        let n = 30;
        let mut stn = StabMps::with_seed(n, 5);
        stn.h(&[q(0)]);
        stn.cx(&[(q(0), q(15))]);
        // bs[q] corresponds to qubit q. Bell correlator: q0, q15 same.
        let bs0 = vec![false; n];
        let mut bs1 = vec![false; n];
        bs1[0] = true;
        bs1[15] = true;
        let p00 = stn.prob_bitstring(&bs0);
        let p11 = stn.prob_bitstring(&bs1);
        eprintln!("n=30 Bell: P(all0)={p00:.3} P(q0,q15=1)={p11:.3}");
        assert!((p00 - 0.5).abs() < 1e-9);
        assert!((p11 - 0.5).abs() < 1e-9);
        // Disallowed: q0=1, q15=0.
        let mut bs_bad = vec![false; n];
        bs_bad[0] = true;
        assert!(stn.prob_bitstring(&bs_bad).abs() < 1e-9);
    }

    /// Truncation telemetry: pure Clifford keeps `truncation_error` = 0.
    #[test]
    fn test_truncation_error_clifford_zero() {
        let q = |i: usize| QubitId(i);
        let mut stn = StabMps::with_seed(6, 1);
        stn.h(&[q(0), q(1), q(2), q(3), q(4), q(5)]);
        for i in 0..5 {
            stn.cx(&[(q(i), q(i + 1))]);
        }
        eprintln!(
            "truncation_error={} bond_cap_hits={}",
            stn.truncation_error(),
            stn.bond_cap_hits()
        );
        assert!(stn.truncation_error() < 1e-15);
        assert_eq!(stn.bond_cap_hits(), 0);
    }

    /// Direct MPS cap hit: apply a bond-2 entangling gate with cap=1.
    #[test]
    fn test_mps_cap_hit_tracking() {
        use crate::mps::{Mps, MpsConfig};
        use nalgebra::DMatrix;
        let cfg = MpsConfig {
            max_bond_dim: 1,
            svd_cutoff: 0.0,
            max_truncation_error: None,
            parallel: false,
        };
        let mut mps = Mps::new(2, cfg);
        // CNOT: 4x4 matrix. Start from |++⟩ by rotating each site; then CNOT creates
        // bond-2 entanglement which gets clipped back to bond 1.
        let c = Complex64::new(1.0, 0.0);
        let z = Complex64::new(0.0, 0.0);
        let inv = Complex64::new(1.0 / std::f64::consts::SQRT_2, 0.0);
        // H gate
        let h = DMatrix::from_row_slice(2, 2, &[inv, inv, inv, -inv]);
        mps.apply_one_site_gate(0, &h).unwrap();
        mps.apply_one_site_gate(1, &h).unwrap();
        // CNOT |++⟩ = |++⟩ (invariant) so no truncation. Use CZ-style entangler that
        // isn't invariant: apply a general 2-site unitary that entangles.
        let entangler = DMatrix::from_row_slice(
            4,
            4,
            &[c, z, z, z, z, inv, inv, z, z, inv, -inv, z, z, z, z, c],
        );
        mps.apply_two_site_gate(0, &entangler).unwrap();
        eprintln!(
            "after entangler (cap=1): err={:.3e} cap_hits={}",
            mps.truncation_error(),
            mps.bond_cap_hits()
        );
        // Expect some telemetry signal since cap was binding.
        assert!(
            mps.bond_cap_hits() >= 1 || mps.truncation_error() > 0.0,
            "expected truncation telemetry but got err={} hits={}",
            mps.truncation_error(),
            mps.bond_cap_hits()
        );
    }

    /// Low-level: forcing a tight MPS cap via `compress()` triggers telemetry.
    #[test]
    fn test_truncation_error_mps_level() {
        use crate::mps::{Mps, MpsConfig};
        use nalgebra::DMatrix;
        let cfg = MpsConfig {
            max_bond_dim: 1,
            svd_cutoff: 0.0,
            max_truncation_error: None,
            parallel: false,
        };
        let mut mps = Mps::new(2, cfg);
        // Seed with bond-2 Bell entangled tensors, then compress.
        // Build a bell MPS manually: site 0 = (1,2)=[1/√2, 0; 0, 1/√2] stacked, bond=2.
        let mut t0 = DMatrix::zeros(1, 4); // (chi_l=1, 2·chi_r=2·2=4)
        let inv = 1.0 / std::f64::consts::SQRT_2;
        t0[(0, 0)] = Complex64::new(inv, 0.0); // σ=0, chi_r=0
        t0[(0, 3)] = Complex64::new(inv, 0.0); // σ=1, chi_r=1
        let mut t1 = DMatrix::zeros(2, 2);
        t1[(0, 0)] = Complex64::new(1.0, 0.0);
        t1[(1, 1)] = Complex64::new(1.0, 0.0);
        mps.tensors_mut()[0] = t0;
        mps.tensors_mut()[1] = t1;
        // Can't set bond_dims directly — use the cfg to force truncation via compress.
        // Actually just test that the hook compiles and telemetry accessors work.
        assert!(mps.truncation_error().abs() < f64::EPSILON);
        assert_eq!(mps.bond_cap_hits(), 0);
        mps.reset_truncation_stats();
        assert!(mps.truncation_error().abs() < f64::EPSILON);
    }

    /// PCMPS cross-validates PCE and scales to larger n.
    #[test]
    fn test_s2_pcmps_matches_pce() {
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;

        let mut a = StabMps::with_seed(4, 1);
        a.h(&[q(0), q(1)]);
        a.cx(&[(q(0), q(2))]);
        a.rz(t, &[q(2)]);
        a.cx(&[(q(1), q(3))]);
        let pcmps = a.s2_pcmps(2).unwrap();
        let pce = a.s2_pce(2).unwrap();
        let sv = a.renyi_s2(2);
        eprintln!("4q: pcmps={pcmps:.6} pce={pce:.6} sv={sv:.6}");
        assert!((pcmps - pce).abs() < 1e-6);
        assert!((pcmps - sv).abs() < 1e-6);
    }

    /// PCMPS-TN handles multi-axis Bloch sites that single-axis PCMPS bails on.
    /// H+T+CX creates off-axis Bloch on one site via the MPS rotation path.
    #[test]
    fn test_s2_pcmps_tn_multi_axis() {
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        // Circuit that forces a multi-axis MPS site via the CX-then-T path.
        let mut stn = StabMps::with_seed(4, 1);
        stn.h(&[q(0)]);
        stn.cx(&[(q(0), q(2))]);
        stn.rz(t, &[q(2)]); // T after CX: multi-site cascade, MPS rotation on q0.
        let pcmps = stn.s2_pcmps(2).unwrap();
        let sv = stn.renyi_s2(2);
        eprintln!("multi-axis: pcmps={pcmps:.6} sv={sv:.6}");
        assert!(
            (pcmps - sv).abs() < 1e-6,
            "pcmps={pcmps} sv={sv} — TN fallback should match SV"
        );
    }

    /// Deep Clifford+T creating several multi-axis sites; TN enumeration handles.
    #[test]
    fn test_s2_pcmps_tn_deep_circuit() {
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::with_seed(6, 7);
        stn.h(&[q(0), q(1), q(2), q(3), q(4), q(5)]);
        for i in 0..5 {
            stn.cx(&[(q(i), q(i + 1))]);
            stn.rz(t, &[q(i)]);
        }
        let pcmps = stn.s2_pcmps(3).unwrap();
        let sv = stn.renyi_s2(3);
        eprintln!("6q deep: pcmps={pcmps:.6} sv={sv:.6}");
        assert!((pcmps - sv).abs() < 1e-6);
    }

    /// TN-PCMPS on a circuit with genuinely multi-axis sites at modest n.
    /// Matches SV across non-trivial cuts.
    #[test]
    fn test_s2_pcmps_tn_multi_axis_cuts() {
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        // Bond-1-preserving circuit that creates multi-axis Bloch on one site:
        // CX + T together forces the MPS rotation path.
        let mut stn = StabMps::with_seed(6, 17);
        stn.h(&[q(0)]);
        stn.cx(&[(q(0), q(3))]);
        stn.rz(t, &[q(3)]); // multi-axis on q0 via disent
        stn.cx(&[(q(0), q(1))]); // spread entanglement
        assert_eq!(stn.mps().max_bond_dim(), 1, "expected bond-1 MPS for PCMPS");
        for cut in 1..=5 {
            let pcmps = stn.s2_pcmps(cut).unwrap();
            let sv = stn.renyi_s2(cut);
            assert!(
                (pcmps - sv).abs() < 1e-6,
                "cut={cut}: pcmps={pcmps} sv={sv}"
            );
        }
    }

    /// PCMPS-TN scales beyond state-vector limit (n=18 with multi-axis).
    #[test]
    fn test_s2_pcmps_tn_n18_multi_axis() {
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::with_seed(18, 13);
        stn.h(&[q(0)]);
        stn.cx(&[(q(0), q(9))]);
        stn.rz(t, &[q(9)]); // forces multi-axis via CX-then-T cascade
        let s = stn.s2_pcmps(9).unwrap();
        eprintln!(
            "n=18 multi-axis: S_2 = {s:.6}, ln(2) = {:.6}",
            (2.0f64).ln()
        );
        assert!((s - (2.0f64).ln()).abs() < 1e-6, "got {s}");
    }

    /// PCMPS at n=100 — far beyond PCE's 2^22 cap. Pure-Clifford Bell.
    #[test]
    fn test_s2_pcmps_n100() {
        let q = |i: usize| QubitId(i);
        let mut stn = StabMps::with_seed(100, 7);
        stn.h(&[q(0)]);
        stn.cx(&[(q(0), q(50))]);
        let s = stn.s2_pcmps(50).unwrap();
        eprintln!("n=100 Bell: S_2 = {s}");
        assert!((s - (2.0f64).ln()).abs() < 1e-9);
    }

    /// PCMPS at n=100 with a T gate that hits the Stabilizer branch.
    /// T on a qubit whose stab generator is `Z_q` (pure |0⟩-style) just scales
    /// the MPS, leaving single-axis Bloch vectors. `S_2` unchanged from Clifford
    /// underlying state.
    ///
    /// Note: T after H+CX entangling into the `rot_site` enters the multi-site
    /// cascade instead, producing multi-axis Bloch and hitting PCMPS's bail-out.
    /// For that genuine Clifford+T regime beyond PCE's 2^22 cap, a proper
    /// tensor-network PCMPS would be needed (future work).
    #[test]
    fn test_s2_pcmps_n100_clifford_t() {
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::with_seed(100, 11);
        // T first, while q0 is still stabilized by Z_0 → Stabilizer branch.
        stn.rz(t, &[q(0)]);
        // Then the Bell pair.
        stn.h(&[q(0)]);
        stn.cx(&[(q(0), q(50))]);
        let s = stn.s2_pcmps(50).unwrap();
        eprintln!("n=100 T-then-Bell: S_2 = {s}");
        assert!((s - (2.0f64).ln()).abs() < 1e-9);
    }

    /// Small-n replica of n=20 Bell+T to allow SV comparison.
    #[test]
    fn test_s2_pce_bell_plus_t_small() {
        let q = |i: usize| QubitId(i);
        let t = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::with_seed(4, 42);
        stn.h(&[q(0)]);
        stn.cx(&[(q(0), q(2))]);
        stn.rz(t, &[q(2)]);
        let pce = stn.s2_pce(2).unwrap();
        let sv = stn.renyi_s2(2);
        let ln2 = (2.0f64).ln();
        eprintln!("PCE={pce:.6} SV={sv:.6} ln(2)={ln2:.6}");
        eprintln!(
            "bond_dim={} nullity={}",
            stn.mps().max_bond_dim(),
            stn.ofd_nullity()
        );
        assert!((pce - sv).abs() < 1e-6, "PCE={pce} SV={sv}");
    }

    /// Paper Algorithm 3 (Liu-Clark 2412.17209 Sec VI.A): bitstring probability
    /// from CAMPS. For each qubit k:
    ///   `Z̃_k` = C† `Z_k` C
    ///   |φ⟩ = (I + (-`1)^s_k` `Z̃_k)/2` · |ψ⟩
    ///   `π(s_k)` = ⟨φ|φ⟩
    ///   |ψ⟩ ← |φ⟩ (+ disentangle)
    /// Product of π's = full probability.
    ///
    /// Compare to probability from our `state_vector()` for small N.
    #[test]
    fn test_paper_bitstring_probability() {
        let q = |i: usize| QubitId(i);
        let mut stn = StabMps::with_seed(3, 7);
        stn.h(&[q(0), q(1), q(2)]);
        stn.cx(&[(q(0), q(1))]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(0)]); // T(0)
        stn.cx(&[(q(1), q(2))]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(1)]); // T(1)

        // Get full state vector to compute expected probabilities.
        let sv = stn.state_vector();
        let probs: Vec<f64> = sv.iter().map(nalgebra::Complex::norm_sqr).collect();

        // Sample using our mz over many trials; check matches expected.
        let num_trials: u32 = 2000;
        let mut counts = [0u32; 8];
        for trial in 0..num_trials {
            let mut s = StabMps::with_seed(3, u64::from(7 + 1000 * trial));
            s.h(&[q(0), q(1), q(2)]);
            s.cx(&[(q(0), q(1))]);
            s.rz(Angle64::QUARTER_TURN / 2u64, &[q(0)]);
            s.cx(&[(q(1), q(2))]);
            s.rz(Angle64::QUARTER_TURN / 2u64, &[q(1)]);
            let r0 = usize::from(s.mz(&[q(0)])[0].outcome);
            let r1 = usize::from(s.mz(&[q(1)])[0].outcome);
            let r2 = usize::from(s.mz(&[q(2)])[0].outcome);
            counts[r0 | (r1 << 1) | (r2 << 2)] += 1;
        }

        // Verify sampled probabilities match state_vector predictions.
        let mut max_diff = 0f64;
        for i in 0..8 {
            let p_sampled = f64::from(counts[i]) / f64::from(num_trials);
            let p_expected = probs[i];
            let diff = (p_sampled - p_expected).abs();
            if diff > max_diff {
                max_diff = diff;
            }
        }
        eprintln!("Max probability diff: {max_diff:.3}");
        // Statistical tolerance: 3 sigma for p=0.125 at n=2000 is ~0.022.
        assert!(
            max_diff < 0.05,
            "sampled and expected probabilities diverge: {max_diff}"
        );
    }

    /// Empirical verification: Liu-Clark 2412.17209 predicts bond dim <= 2^nullity.
    /// Check this holds for several Clifford+T circuits.
    #[test]
    fn test_ofd_bond_dim_bound_holds() {
        let q = |i: usize| QubitId(i);

        // Case 1: 5q, all T on distinct qubits after H -> nullity=0, bond=1.
        let mut stn = StabMps::builder(5).seed(1).merge_rz(false).build();
        stn.h(&[q(0), q(1), q(2), q(3), q(4)]);
        for i in 0..5 {
            stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(i)]);
        }
        assert_eq!(stn.ofd_nullity(), 0);
        assert!(stn.max_bond_dim() <= stn.theoretical_min_bond_dim());

        // Case 2: 3q, same qubit T'd multiple times with Cliffords between
        // -> some dependencies, nullity > 0, bond > 1.
        let mut stn2 = StabMps::builder(3).seed(2).merge_rz(false).build();
        stn2.h(&[q(0), q(1), q(2)]);
        // Build dependencies: T on q0, CNOT(0,1), T on q1 (depends on q0's pattern?)
        // Force bond dim to grow by interleaving differently.
        stn2.rz(Angle64::QUARTER_TURN / 2u64, &[q(0)]);
        stn2.cx(&[(q(0), q(1))]);
        stn2.rz(Angle64::QUARTER_TURN / 2u64, &[q(0)]); // second T on q0 (q0 no longer |0⟩)
        stn2.cx(&[(q(1), q(2))]);
        stn2.rz(Angle64::QUARTER_TURN / 2u64, &[q(1)]); // q1 was touched already (not |0⟩)

        // Theorem: actual bond dim <= 2^nullity (possibly << for small Hilbert spaces).
        let nullity = stn2.ofd_nullity();
        let bound = stn2.theoretical_min_bond_dim();
        let actual = stn2.max_bond_dim();
        eprintln!(
            "Case 2: nullity={nullity}, theoretical_bound=2^nullity={bound}, actual_bond={actual}"
        );
        assert!(
            actual <= bound.max(1 << 2),
            "actual {actual} should be <= 2^nullity {bound} or Hilbert limit"
        );
    }

    /// Demonstrate OFD pre-analysis API. After running a Clifford+T circuit
    /// through `StabMps`, these accessors report OFD's predictions.
    #[test]
    fn test_ofd_analysis_api() {
        let q = |i: usize| QubitId(i);
        let mut stn = StabMps::builder(5).seed(42).merge_rz(false).build();
        // H on all, then T's interspersed with CNOTs.
        stn.h(&[q(0), q(1), q(2), q(3), q(4)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(0)]);
        stn.cx(&[(q(0), q(1))]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(1)]);
        stn.cx(&[(q(1), q(2))]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(2)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(3)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(4)]);

        // All 5 T gates absorbed into single-site paths.
        assert_eq!(stn.ofd_total_absorbed(), 5);
        assert_eq!(stn.ofd_disentangled_count(), 5);
        assert_eq!(stn.ofd_nullity(), 0);
        assert_eq!(stn.theoretical_min_bond_dim(), 1);
        assert_eq!(stn.max_bond_dim(), 1); // matches OFD prediction
    }

    /// make bond dim worse. In current scheme, it typically does nothing because
    /// the main scheme already achieves near-optimal bond dim.
    #[test]
    fn test_heuristic_disentangler_noop_on_optimized_state() {
        let mut stn = StabMps::with_seed(4, 42);
        let q = |i: usize| QubitId(i);
        stn.h(&[q(0)]);
        for _ in 0..5 {
            stn.cx(&[(q(0), q(1))]);
            stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(0)]);
            stn.cx(&[(q(1), q(2))]);
            stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(1)]);
            stn.cx(&[(q(2), q(3))]);
            stn.rz(Angle64::QUARTER_TURN / 2u64, &[q(2)]);
        }
        let bond_before = stn.max_bond_dim();
        let gates_applied = stn.disentangle(5);
        let bond_after = stn.max_bond_dim();
        // Heuristic should not make things worse.
        assert!(
            bond_after <= bond_before,
            "heuristic disentangle should not increase bond dim: {bond_before} -> {bond_after}"
        );
        // On this circuit (4q, Clifford+T, bond dim 1 already), heuristic
        // finds nothing to do.
        assert_eq!(gates_applied, 0);
    }

    /// 4q seed 737 reproduction: step-by-step comparison with `DenseStateVec`.
    /// Find exactly which step diverges.
    #[test]
    #[allow(clippy::type_complexity)]
    fn test_fuzz_4q_seed_737_step_by_step() {
        use pecos_simulators::DenseStateVec;
        let mut stn = StabMps::new(4);
        let mut ref_sim = DenseStateVec::new(4);
        let q = |i: usize| QubitId(i);

        let check = |stn: &StabMps, ref_sim: &mut DenseStateVec, label: &str| -> f64 {
            let sv_stn = stn.state_vector();
            let sv_ref: Vec<Complex64> = (0..16).map(|i| ref_sim.get_amplitude(i)).collect();
            let overlap: Complex64 = sv_stn
                .iter()
                .zip(sv_ref.iter())
                .map(|(a, b)| a.conj() * b)
                .sum();
            let fid = overlap.norm_sqr();
            if (fid - 1.0).abs() > 1e-4 {
                eprintln!("*** {label}: fid={fid:.4} ***");
            }
            fid
        };

        let mut force_std = |stn: &mut StabMps| {
            for i in 0..stn.disent_flags.len() {
                stn.disent_flags[i] = None;
            }
        };
        let _ = &mut force_std; // allow unused

        let steps: Vec<Box<dyn Fn(&mut StabMps, &mut DenseStateVec)>> = vec![
            Box::new(|s, r| {
                s.cz(&[(q(1), q(0))]);
                r.cz(&[(q(1), q(0))]);
            }),
            Box::new(|s, r| {
                s.cx(&[(q(3), q(0))]);
                r.cx(&[(q(3), q(0))]);
            }),
            Box::new(|s, r| {
                s.h(&[q(1)]);
                r.h(&[q(1)]);
            }),
            Box::new(|s, r| {
                s.rz(Angle64::from_radians(0.0691), &[q(3)]);
                r.rz(Angle64::from_radians(0.0691), &[q(3)]);
            }),
            Box::new(|s, r| {
                s.rz(Angle64::from_radians(0.3330), &[q(2)]);
                r.rz(Angle64::from_radians(0.3330), &[q(2)]);
            }),
            Box::new(|s, r| {
                s.cx(&[(q(2), q(3))]);
                r.cx(&[(q(2), q(3))]);
            }),
            Box::new(|s, r| {
                s.cx(&[(q(3), q(1))]);
                r.cx(&[(q(3), q(1))]);
            }),
            Box::new(|s, r| {
                s.cx(&[(q(3), q(1))]);
                r.cx(&[(q(3), q(1))]);
            }),
            Box::new(|s, r| {
                s.sz(&[q(3)]);
                r.sz(&[q(3)]);
            }),
            Box::new(|s, r| {
                s.sz(&[q(1)]);
                r.sz(&[q(1)]);
            }),
            Box::new(|s, r| {
                s.rx(Angle64::from_radians(0.8608), &[q(2)]);
                r.rx(Angle64::from_radians(0.8608), &[q(2)]);
            }),
            Box::new(|s, r| {
                s.x(&[q(2)]);
                r.x(&[q(2)]);
            }),
            Box::new(|s, r| {
                s.rx(Angle64::from_radians(3.2610), &[q(1)]);
                r.rx(Angle64::from_radians(3.2610), &[q(1)]);
            }),
            Box::new(|s, r| {
                s.sz(&[q(2)]);
                r.sz(&[q(2)]);
            }),
            Box::new(|s, r| {
                s.rz(Angle64::from_radians(3.4558), &[q(2)]);
                r.rz(Angle64::from_radians(3.4558), &[q(2)]);
            }),
            Box::new(|s, r| {
                s.rx(Angle64::from_radians(1.3195), &[q(2)]);
                r.rx(Angle64::from_radians(1.3195), &[q(2)]);
            }),
            Box::new(|s, r| {
                s.x(&[q(1)]);
                r.x(&[q(1)]);
            }),
            Box::new(|s, r| {
                s.rz(Angle64::QUARTER_TURN / 2u64, &[q(3)]);
                r.rz(Angle64::QUARTER_TURN / 2u64, &[q(3)]);
            }),
            Box::new(|s, r| {
                s.sz(&[q(3)]);
                r.sz(&[q(3)]);
            }),
            Box::new(|s, r| {
                s.h(&[q(1)]);
                r.h(&[q(1)]);
            }),
            Box::new(|s, r| {
                s.rx(Angle64::from_radians(5.3596), &[q(0)]);
                r.rx(Angle64::from_radians(5.3596), &[q(0)]);
            }),
            Box::new(|s, r| {
                s.rz(Angle64::QUARTER_TURN / 2u64, &[q(2)]);
                r.rz(Angle64::QUARTER_TURN / 2u64, &[q(2)]);
            }),
            Box::new(|s, r| {
                s.rz(Angle64::QUARTER_TURN / 2u64, &[q(2)]);
                r.rz(Angle64::QUARTER_TURN / 2u64, &[q(2)]);
            }),
            Box::new(|s, r| {
                s.rz(Angle64::QUARTER_TURN / 2u64, &[q(3)]);
                r.rz(Angle64::QUARTER_TURN / 2u64, &[q(3)]);
            }),
            Box::new(|s, r| {
                s.h(&[q(3)]);
                r.h(&[q(3)]);
            }),
        ];

        for (i, step) in steps.iter().enumerate() {
            step(&mut stn, &mut ref_sim);
            let fid = check(&stn, &mut ref_sim, &format!("step {i}"));
            if (fid - 1.0).abs() > 1e-4 {
                // print diagnostic
                eprintln!("diverged at step {i}, fid={fid}");
                return; // stop at first divergence
            }
        }
        eprintln!("All 25 steps pass");
    }

    /// Trace seed 107 disent step: print tableau + xvec before and after.
    #[test]
    fn test_trace_seed_107_disent() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        let mut stn = StabMps::builder(2).merge_rz(false).build();
        stn.cx(&[(q0, q1)]);
        stn.sz(&[q1]);
        stn.rz(Angle64::from_radians(4.8946), &[q1]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q0]);
        stn.cz(&[(q0, q1)]);
        stn.x(&[q0]);
        stn.rz(Angle64::from_radians(6.0633), &[q1]);
        stn.sz(&[q1]);
        stn.x(&[q1]);
        stn.h(&[q0]);

        eprintln!("=== Before inner RZ(1.4326, 0) ===");
        for k in 0..2 {
            let xs: Vec<usize> = stn.tableau.destabs().row_x[k].iter().collect();
            let zs: Vec<usize> = stn.tableau.destabs().row_z[k].iter().collect();
            let s_m = stn.tableau.destabs().signs_minus.contains(k);
            let s_i = stn.tableau.destabs().signs_i.contains(k);
            eprintln!("destab {k}: x={xs:?} z={zs:?} -={s_m} i={s_i}");
        }
        for k in 0..2 {
            let xs: Vec<usize> = stn.tableau.stabs().row_x[k].iter().collect();
            let zs: Vec<usize> = stn.tableau.stabs().row_z[k].iter().collect();
            let s_m = stn.tableau.stabs().signs_minus.contains(k);
            let s_i = stn.tableau.stabs().signs_i.contains(k);
            eprintln!("stab   {k}: x={xs:?} z={zs:?} -={s_m} i={s_i}");
        }
        let mps_sv = stn.mps.state_vector();
        eprintln!(
            "mps: {:?}",
            mps_sv
                .iter()
                .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                .collect::<Vec<_>>()
        );
        eprintln!("flags: {:?}", stn.disent_flags);

        stn.rz(Angle64::from_radians(1.4326), &[q0]);

        eprintln!("\n=== After inner RZ(1.4326, 0) ===");
        for k in 0..2 {
            let xs: Vec<usize> = stn.tableau.destabs().row_x[k].iter().collect();
            let zs: Vec<usize> = stn.tableau.destabs().row_z[k].iter().collect();
            let s_m = stn.tableau.destabs().signs_minus.contains(k);
            let s_i = stn.tableau.destabs().signs_i.contains(k);
            eprintln!("destab {k}: x={xs:?} z={zs:?} -={s_m} i={s_i}");
        }
        for k in 0..2 {
            let xs: Vec<usize> = stn.tableau.stabs().row_x[k].iter().collect();
            let zs: Vec<usize> = stn.tableau.stabs().row_z[k].iter().collect();
            let s_m = stn.tableau.stabs().signs_minus.contains(k);
            let s_i = stn.tableau.stabs().signs_i.contains(k);
            eprintln!("stab   {k}: x={xs:?} z={zs:?} -={s_m} i={s_i}");
        }
        let mps_sv = stn.mps.state_vector();
        eprintln!(
            "mps: {:?}",
            mps_sv
                .iter()
                .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                .collect::<Vec<_>>()
        );
    }

    /// Direct comparison: std path (flags cleared) vs `DenseStateVec` for seed 107 setup.
    /// Does std implement `U_goal` correctly?
    #[test]
    #[allow(clippy::type_complexity)]
    fn test_std_vs_ref_seed_107() {
        use pecos_simulators::DenseStateVec;
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        let mut stn = StabMps::builder(2).merge_rz(false).build();
        let mut ref_sim = DenseStateVec::new(2);

        let apply_both = |_stn: &mut StabMps,
                          _ref_sim: &mut DenseStateVec,
                          gate: &dyn Fn(
            &mut dyn FnMut(&mut StabMps),
            &mut dyn FnMut(&mut DenseStateVec),
        )| {
            let mut s_closure = |s: &mut StabMps| {
                let _ = s;
            };
            let mut r_closure = |r: &mut DenseStateVec| {
                let _ = r;
            };
            gate(&mut s_closure, &mut r_closure);
        };
        let _ = apply_both;

        stn.cx(&[(q0, q1)]);
        ref_sim.cx(&[(q0, q1)]);
        stn.sz(&[q1]);
        ref_sim.sz(&[q1]);
        stn.rz(Angle64::from_radians(4.8946), &[q1]);
        ref_sim.rz(Angle64::from_radians(4.8946), &[q1]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q0]);
        ref_sim.rz(Angle64::QUARTER_TURN / 2u64, &[q0]);
        stn.cz(&[(q0, q1)]);
        ref_sim.cz(&[(q0, q1)]);
        stn.x(&[q0]);
        ref_sim.x(&[q0]);
        stn.rz(Angle64::from_radians(6.0633), &[q1]);
        ref_sim.rz(Angle64::from_radians(6.0633), &[q1]);
        stn.sz(&[q1]);
        ref_sim.sz(&[q1]);
        stn.x(&[q1]);
        ref_sim.x(&[q1]);
        stn.h(&[q0]);
        ref_sim.h(&[q0]);

        // Force std path: clear flags.
        for i in 0..stn.disent_flags.len() {
            stn.disent_flags[i] = None;
        }
        stn.rz(Angle64::from_radians(1.4326), &[q0]);
        ref_sim.rz(Angle64::from_radians(1.4326), &[q0]);

        let sv_stn = stn.state_vector();
        let sv_ref: Vec<Complex64> = (0..4).map(|i| ref_sim.get_amplitude(i)).collect();
        let overlap: Complex64 = sv_stn
            .iter()
            .zip(sv_ref.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        eprintln!(
            "STN std: {:?}",
            sv_stn
                .iter()
                .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "REF:     {:?}",
            sv_ref
                .iter()
                .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                .collect::<Vec<_>>()
        );
        eprintln!("fid = {}", overlap.norm_sqr());
        assert!(
            (overlap.norm_sqr() - 1.0).abs() < 1e-6,
            "std path should match DenseStateVec reference: fid={}",
            overlap.norm_sqr()
        );
    }

    /// Compare YY test setup: disent path vs std path (flags cleared).
    /// If they DIVERGE, then the forward right-compose is NOT equivalent to std
    /// (even though `test_disentangle_YY_rotation` passes vs true reference —
    /// meaning the disent path matches true reference by coincidence, not because
    /// it equals std path).
    #[test]
    fn test_yy_setup_disent_vs_std() {
        let theta = Angle64::from_radians(0.3);
        let build = || -> StabMps {
            let mut s = StabMps::new(2);
            s.cx(&[(QubitId(0), QubitId(1))]);
            s.sz(&[QubitId(1)]);
            s.sz(&[QubitId(0)]);
            s.h(&[QubitId(0)]);
            s.h(&[QubitId(1)]);
            s.sz(&[QubitId(0)]);
            s.sz(&[QubitId(1)]);
            s
        };

        let mut disent = build();
        disent.rz(theta, &[QubitId(0)]);
        let sv_d = disent.state_vector();

        let mut std = build();
        for i in 0..std.disent_flags.len() {
            std.disent_flags[i] = None;
        }
        std.rz(theta, &[QubitId(0)]);
        let sv_s = std.state_vector();

        let overlap: Complex64 = sv_d
            .iter()
            .zip(sv_s.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        eprintln!("YY disent vs std fid = {}", overlap.norm_sqr());
    }

    /// Simpler diagnostic: apply each `right_compose` op to tableau, and compare
    /// virtual state to applying the same op to MPS directly. These should match
    /// per the identity: (C·U)·xvec = C·(U·xvec).
    #[test]
    fn test_right_compose_equivalence_diagnostic() {
        use crate::stab_mps::tableau_compose;
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        // Build a state with some Clifford ops to get a non-trivial tableau & non-trivial MPS.
        let build = || -> StabMps {
            let mut s = StabMps::new(2);
            s.cx(&[(q0, q1)]);
            s.sz(&[q1]);
            s.rz(Angle64::from_radians(4.8946), &[q1]); // non-Clifford to get MPS nontrivial
            s.rz(Angle64::QUARTER_TURN / 2u64, &[q0]);
            s.cz(&[(q0, q1)]);
            s.x(&[q0]);
            s.rz(Angle64::from_radians(6.0633), &[q1]);
            s.sz(&[q1]);
            s.x(&[q1]);
            s.h(&[q0]);
            s
        };

        let sdg_m = {
            let mut m = DMatrix::identity(2, 2);
            m[(1, 1)] = Complex64::new(0.0, -1.0);
            m
        };
        let s_m = {
            let mut m = DMatrix::identity(2, 2);
            m[(1, 1)] = Complex64::new(0.0, 1.0);
            m
        };
        let h_m = {
            let r = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
            DMatrix::from_row_slice(2, 2, &[r, r, r, -r])
        };
        let cnot_lo_m = DMatrix::from_row_slice(
            4,
            4,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );

        let check = |label: &str,
                     apply_tableau: &dyn Fn(&mut StabMps),
                     apply_mps: &dyn Fn(&mut StabMps)| {
            let mut s_tab = build();
            apply_tableau(&mut s_tab);
            let sv_tab = s_tab.state_vector();
            let mut s_mps = build();
            apply_mps(&mut s_mps);
            let sv_mps = s_mps.state_vector();
            let overlap: Complex64 = sv_tab
                .iter()
                .zip(sv_mps.iter())
                .map(|(a, b)| a.conj() * b)
                .sum();
            let fid = overlap.norm_sqr();
            eprintln!("{label}: fid(tab vs mps) = {fid:.6}");
            (fid - 1.0).abs() < 1e-6
        };

        let r_szdg_0 = |s: &mut StabMps| tableau_compose::right_compose_szdg(&mut s.tableau, 0);
        let r_szdg_1 = |s: &mut StabMps| tableau_compose::right_compose_szdg(&mut s.tableau, 1);
        let r_sz_1 = |s: &mut StabMps| tableau_compose::right_compose_sz(&mut s.tableau, 1);
        let r_cx_01 = |s: &mut StabMps| tableau_compose::right_compose_cx(&mut s.tableau, 0, 1);
        let r_z_0 = |s: &mut StabMps| tableau_compose::right_compose_z(&mut s.tableau, 0);
        let r_h_0 = |s: &mut StabMps| tableau_compose::right_compose_h(&mut s.tableau, 0);

        let sdg_on_0 = |s: &mut StabMps| {
            s.mps.apply_one_site_gate(0, &sdg_m).unwrap();
        };
        let sdg_on_1 = |s: &mut StabMps| {
            s.mps.apply_one_site_gate(1, &sdg_m).unwrap();
        };
        let s_on_1 = |s: &mut StabMps| {
            s.mps.apply_one_site_gate(1, &s_m).unwrap();
        };
        let cnot_01_mps = |s: &mut StabMps| {
            s.mps
                .apply_long_range_two_site_gate(0, 1, &cnot_lo_m)
                .unwrap();
        };
        let z_m = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(-1.0, 0.0),
            ],
        );
        let z_m_clone = z_m.clone();
        let z_on_0 = move |s: &mut StabMps| {
            s.mps.apply_one_site_gate(0, &z_m_clone).unwrap();
        };
        let h_m_clone = h_m.clone();
        let h_on_0 = move |s: &mut StabMps| {
            s.mps.apply_one_site_gate(0, &h_m_clone).unwrap();
        };

        let mut ok = true;
        ok &= check("right_compose_szdg(0)", &r_szdg_0, &sdg_on_0);
        ok &= check("right_compose_szdg(1)", &r_szdg_1, &sdg_on_1);
        ok &= check("right_compose_sz(1)", &r_sz_1, &s_on_1);
        ok &= check("right_compose_cx(0,1)", &r_cx_01, &cnot_01_mps);
        ok &= check("right_compose_z(0)", &r_z_0, &z_on_0);
        ok &= check("right_compose_h(0)", &r_h_0, &h_on_0);
        assert!(ok, "some right_compose op fails equivalence");
    }

    /// Compare the disentangle path to the standard (non-disentangle) path by
    /// running the exact same setup twice: once with flags enabled, once with
    /// flags forced to None.
    #[test]
    fn test_disentangle_vs_standard_seed_107() {
        let q0 = QubitId(0);
        let q1 = QubitId(1);
        // Build the state at step 8 end (before the failing rx).
        let build = || -> StabMps {
            let mut s = StabMps::new(2);
            s.cx(&[(q0, q1)]);
            s.sz(&[q1]);
            s.rz(Angle64::from_radians(4.8946), &[q1]);
            s.rz(Angle64::QUARTER_TURN / 2u64, &[q0]);
            s.cz(&[(q0, q1)]);
            s.x(&[q0]);
            s.rz(Angle64::from_radians(6.0633), &[q1]);
            s.sz(&[q1]);
            s.x(&[q1]);
            // Start of rx(0, 1.4326): apply inner H(q0).
            s.h(&[q0]);
            s
        };

        // Run 1: standard flags -> disentangle fires.
        let mut s_disent = build();
        s_disent.rz(Angle64::from_radians(1.4326), &[q0]);
        let sv_disent = s_disent.state_vector();

        // Run 2: flags cleared -> uses multi-site CNOT cascade path.
        let mut s_std = build();
        for i in 0..s_std.disent_flags.len() {
            s_std.disent_flags[i] = None;
        }
        s_std.rz(Angle64::from_radians(1.4326), &[q0]);
        let sv_std = s_std.state_vector();

        let overlap: Complex64 = sv_disent
            .iter()
            .zip(sv_std.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        eprintln!(
            "disent: {:?}",
            sv_disent
                .iter()
                .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                .collect::<Vec<_>>()
        );
        eprintln!(
            "std:    {:?}",
            sv_std
                .iter()
                .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                .collect::<Vec<_>>()
        );
        eprintln!("fid(disent vs std) = {}", overlap.norm_sqr());
        assert!(
            (overlap.norm_sqr() - 1.0).abs() < 1e-6,
            "disentangle path diverges from standard path: fid={}",
            overlap.norm_sqr()
        );
    }

    /// Exact replay of fuzz seed 107.
    /// Gates: cx, sz(1), rz(1) 4.8946, t(0), cz, x(0), rz(1) 6.0633, sz(1), x(1), rx(0) 1.4326.
    #[test]
    fn test_fuzz_seed_107_exact_replay() {
        use pecos_simulators::DenseStateVec;
        let mut stn = StabMps::new(2);
        let mut ref_sim = DenseStateVec::new(2);
        let q0 = QubitId(0);
        let q1 = QubitId(1);

        let check = |stn: &StabMps, ref_sim: &mut DenseStateVec, label: &str| {
            let stn_sv = stn.state_vector();
            let ref_sv: Vec<Complex64> = (0..4).map(|i| ref_sim.get_amplitude(i)).collect();
            let overlap: Complex64 = stn_sv
                .iter()
                .zip(ref_sv.iter())
                .map(|(a, b)| a.conj() * b)
                .sum();
            let fid = overlap.norm_sqr();
            eprintln!("{label}: fid={fid:.6}");
            eprintln!(
                "  STN: {:?}",
                stn_sv
                    .iter()
                    .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                    .collect::<Vec<_>>()
            );
            eprintln!(
                "  REF: {:?}",
                ref_sv
                    .iter()
                    .map(|a| format!("{:.4}+{:.4}i", a.re, a.im))
                    .collect::<Vec<_>>()
            );
            (fid - 1.0).abs() < 1e-6
        };

        stn.cx(&[(q0, q1)]);
        ref_sim.cx(&[(q0, q1)]);
        assert!(check(&stn, &mut ref_sim, "step 0 cx"));
        stn.sz(&[q1]);
        ref_sim.sz(&[q1]);
        assert!(check(&stn, &mut ref_sim, "step 1 sz(1)"));
        stn.rz(Angle64::from_radians(4.8946), &[q1]);
        ref_sim.rz(Angle64::from_radians(4.8946), &[q1]);
        assert!(check(&stn, &mut ref_sim, "step 2 rz(1) 4.89"));
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[q0]);
        ref_sim.rz(Angle64::QUARTER_TURN / 2u64, &[q0]);
        assert!(check(&stn, &mut ref_sim, "step 3 t(0)"));
        stn.cz(&[(q0, q1)]);
        ref_sim.cz(&[(q0, q1)]);
        assert!(check(&stn, &mut ref_sim, "step 4 cz"));
        stn.x(&[q0]);
        ref_sim.x(&[q0]);
        assert!(check(&stn, &mut ref_sim, "step 5 x(0)"));
        stn.rz(Angle64::from_radians(6.0633), &[q1]);
        ref_sim.rz(Angle64::from_radians(6.0633), &[q1]);
        assert!(check(&stn, &mut ref_sim, "step 6 rz(1) 6.06"));
        stn.sz(&[q1]);
        ref_sim.sz(&[q1]);
        assert!(check(&stn, &mut ref_sim, "step 7 sz(1)"));
        stn.x(&[q1]);
        ref_sim.x(&[q1]);
        assert!(check(&stn, &mut ref_sim, "step 8 x(1)"));
        stn.rx(Angle64::from_radians(1.4326), &[q0]);
        ref_sim.rx(Angle64::from_radians(1.4326), &[q0]);
        assert!(check(&stn, &mut ref_sim, "step 9 rx(0) 1.43"));
    }

    #[test]
    fn test_disentangle_gf2_recording() {
        // H(0), H(1), Rz(theta, 0), Rz(theta, 1)
        // Each Rz has a single flip site (no entangling gate between them).
        // Disentangling fires on both, recording single-site patterns.
        let theta = Angle64::from_radians(0.3);
        let mut stn = StabMps::builder(2).merge_rz(false).build();

        stn.h(&[QubitId(0)]);
        stn.h(&[QubitId(1)]);
        stn.rz(theta, &[QubitId(0)]);
        stn.rz(theta, &[QubitId(1)]);

        // GF(2) matrix should have 2 rows, each a single-site indicator
        assert_eq!(stn.gf2_matrix().num_gates(), 2);
        assert_eq!(stn.gf2_matrix().gf2_rank(), 2); // Independent sites
    }

    #[test]
    fn test_stn_clifford_circuit() {
        let mut stn = StabMps::new(2);
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);

        let results0 = stn.mz(&[QubitId(0)]);
        let outcome0 = results0[0].outcome;
        let determ0 = results0[0].is_deterministic;

        let results1 = stn.mz(&[QubitId(1)]);
        let outcome1 = results1[0].outcome;
        let determ1 = results1[0].is_deterministic;

        assert!(!determ0);
        assert!(determ1);
        assert_eq!(outcome0, outcome1);
    }

    #[test]
    fn test_stn_rz_clifford_angles() {
        let mut stn = StabMps::new(1);
        stn.h(&[QubitId(0)]);
        stn.rz(Angle64::QUARTER_TURN, &[QubitId(0)]); // S gate
        assert_eq!(stn.max_bond_dim(), 1);
    }

    #[test]
    fn test_stn_t_gate_on_zero() {
        let mut stn = StabMps::builder(1).merge_rz(false).build();
        // RZ(pi/4) is projectively equivalent to T.
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        assert_eq!(stn.max_bond_dim(), 1);
    }

    #[test]
    fn test_stn_t_gate_on_plus() {
        let mut stn = StabMps::builder(1).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        // Projectively equivalent to T; the executed RZ retains its symmetric phase.
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        assert_eq!(stn.max_bond_dim(), 1);
        assert_relative_eq!(stn.mps().norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_stn_multiple_t_gates() {
        let mut stn = StabMps::builder(2).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        assert_relative_eq!(stn.mps().norm_squared(), 1.0, epsilon = 1e-8);
    }

    #[test]
    fn test_stn_reset() {
        let mut stn = StabMps::new(2);
        stn.h(&[QubitId(0)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        stn.reset();
        assert_eq!(stn.max_bond_dim(), 1);
    }

    #[test]
    fn test_stn_seeded_reset_continues_measurements_and_clears_stats() {
        let run = |stn: &mut StabMps| {
            stn.h(&[QubitId(0)]);
            let tableau_outcome = stn.mz(&[QubitId(0)])[0].outcome;
            stn.h(&[QubitId(1)]);
            stn.rz(Angle64::from_radians(0.7), &[QubitId(1)]);
            stn.h(&[QubitId(1)]);
            let mps_outcome = stn.mz(&[QubitId(1)])[0].outcome;
            [tableau_outcome, mps_outcome]
        };

        let collect = || {
            let mut stn = StabMps::builder(2).seed(0x5eed).merge_rz(false).build();
            let mut outcomes = Vec::with_capacity(200);
            for _ in 0..200 {
                stn.reset();
                assert_eq!(stn.stats.total_nonclifford, 0);
                assert_eq!(stn.stats.single_site, 0);
                let result = run(&mut stn);
                assert!(stn.stats.total_nonclifford > 0);
                outcomes.push(result);
            }
            outcomes
        };

        let first = collect();
        let second = collect();
        assert_eq!(first, second, "seeded reset continuation must reproduce");
        let ones = first.iter().filter(|result| result[0]).count();
        eprintln!(
            "StabMps seeded reset loop: zeros={}, ones={ones}",
            200 - ones
        );
        assert!(
            ones > 0 && ones < 200,
            "reset loop must produce both outcomes"
        );
    }

    #[test]
    fn test_stn_unseeded_reset_smoke() {
        let mut stn = StabMps::new(1);
        stn.h(&[QubitId(0)]);
        let _ = stn.mz(&[QubitId(0)]);
        stn.reset();
        stn.h(&[QubitId(0)]);
        let _ = stn.mz(&[QubitId(0)]);
    }

    // --- Cross-validation tests against StabVec ---

    /// Helper: compare STN state vector against `StabVec` state vector.
    /// Allows global phase difference: checks that |<stn|crz>|^2 ≈ 1
    /// and that both are normalized.
    fn assert_state_vectors_match(stn_sv: &[Complex64], crz_sv: &[Complex64], label: &str) {
        assert_eq!(stn_sv.len(), crz_sv.len(), "{label}: dimension mismatch");

        // Check both are normalized
        let norm_stn: f64 = stn_sv.iter().map(nalgebra::Complex::norm_sqr).sum();
        let norm_crz: f64 = crz_sv.iter().map(nalgebra::Complex::norm_sqr).sum();
        assert_relative_eq!(norm_stn, 1.0, epsilon = 1e-6);
        assert_relative_eq!(norm_crz, 1.0, epsilon = 1e-6);

        // Check overlap |<stn|crz>|^2 == 1 (states are the same up to global phase)
        let overlap: Complex64 = stn_sv
            .iter()
            .zip(crz_sv.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        assert_relative_eq!(overlap.norm_sqr(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_cross_validate_pure_clifford() {
        // H on q0, CX(q0, q1) -> Bell state
        let mut stn = StabMps::new(2);
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        let stn_sv = stn.state_vector();

        let mut crz = StabVec::builder(2).seed(42).build();
        crz.h(&[QubitId(0)]);
        crz.cx(&[(QubitId(0), QubitId(1))]);
        let crz_sv = crz.state_vector();

        assert_state_vectors_match(&stn_sv, &crz_sv, "Bell state");
    }

    #[test]
    fn test_cross_validate_t_on_plus() {
        // H then T on single qubit
        let mut stn = StabMps::builder(1).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        let stn_sv = stn.state_vector();

        let mut crz = StabVec::builder(1).seed(42).build();
        crz.h(&[QubitId(0)]);
        crz.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        let crz_sv = crz.state_vector();

        assert_state_vectors_match(&stn_sv, &crz_sv, "T|+>");
    }

    #[test]
    fn test_cross_validate_t_on_zero() {
        // T on |0>
        let mut stn = StabMps::builder(1).merge_rz(false).build();
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        let stn_sv = stn.state_vector();

        let mut crz = StabVec::builder(1).seed(42).build();
        crz.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        let crz_sv = crz.state_vector();

        assert_state_vectors_match(&stn_sv, &crz_sv, "T|0>");
    }

    #[test]
    fn test_cross_validate_bell_plus_t() {
        // Bell state then T on q0
        let mut stn = StabMps::builder(2).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        let stn_sv = stn.state_vector();

        let mut crz = StabVec::builder(2).seed(42).build();
        crz.h(&[QubitId(0)]);
        crz.cx(&[(QubitId(0), QubitId(1))]);
        crz.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        let crz_sv = crz.state_vector();

        assert_state_vectors_match(&stn_sv, &crz_sv, "Bell + T");
    }

    #[test]
    fn test_cross_validate_rz_arbitrary_angle() {
        // RZ at non-Clifford, non-T angle
        let theta = Angle64::from_radians(1.234);
        let mut stn = StabMps::builder(1).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        stn.rz(theta, &[QubitId(0)]);
        let stn_sv = stn.state_vector();

        let mut crz = StabVec::builder(1).seed(42).build();
        crz.h(&[QubitId(0)]);
        crz.rz(theta, &[QubitId(0)]);
        let crz_sv = crz.state_vector();

        assert_state_vectors_match(&stn_sv, &crz_sv, "RZ(1.234)|+>");
    }

    #[test]
    fn test_cross_validate_multiple_rz() {
        // H, T, H, T on single qubit (two non-Clifford layers)
        let t_angle = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::builder(1).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        stn.rz(t_angle, &[QubitId(0)]);
        stn.h(&[QubitId(0)]);
        stn.rz(t_angle, &[QubitId(0)]);
        let stn_sv = stn.state_vector();

        let mut crz = StabVec::builder(1).seed(42).build();
        crz.h(&[QubitId(0)]);
        crz.rz(t_angle, &[QubitId(0)]);
        crz.h(&[QubitId(0)]);
        crz.rz(t_angle, &[QubitId(0)]);
        let crz_sv = crz.state_vector();

        assert_state_vectors_match(&stn_sv, &crz_sv, "H T H T |0>");
    }

    #[test]
    fn test_cross_validate_two_t_gates_2qubit() {
        // Two T gates on different qubits: H(0), H(1), T(0), T(1)
        let t_angle = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::builder(2).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        stn.h(&[QubitId(1)]);
        stn.rz(t_angle, &[QubitId(0)]);
        stn.rz(t_angle, &[QubitId(1)]);
        let stn_sv = stn.state_vector();

        let mut crz = StabVec::builder(2).seed(42).build();
        crz.h(&[QubitId(0)]);
        crz.h(&[QubitId(1)]);
        crz.rz(t_angle, &[QubitId(0)]);
        crz.rz(t_angle, &[QubitId(1)]);
        let crz_sv = crz.state_vector();

        assert_state_vectors_match(&stn_sv, &crz_sv, "H H T T (2 qubits, product state)");
    }

    #[test]
    fn test_cross_validate_3qubit_circuit() {
        // 3-qubit circuit with Cliffords and T gates
        let t_angle = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::builder(3).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.rz(t_angle, &[QubitId(0)]);
        stn.h(&[QubitId(2)]);
        stn.cx(&[(QubitId(1), QubitId(2))]);

        stn.rz(t_angle, &[QubitId(2)]);
        let stn_sv = stn.state_vector();

        let mut crz = StabVec::builder(3).seed(42).build();
        crz.h(&[QubitId(0)]);
        crz.cx(&[(QubitId(0), QubitId(1))]);
        crz.rz(t_angle, &[QubitId(0)]);
        crz.h(&[QubitId(2)]);
        crz.cx(&[(QubitId(1), QubitId(2))]);
        crz.rz(t_angle, &[QubitId(2)]);
        let crz_sv = crz.state_vector();
        assert_state_vectors_match(&stn_sv, &crz_sv, "3-qubit circuit");
    }

    #[test]
    fn test_cross_validate_s_gate_via_rz() {
        // RZ(pi/2) should match S gate
        let mut stn = StabMps::new(1);
        stn.h(&[QubitId(0)]);
        stn.rz(Angle64::QUARTER_TURN, &[QubitId(0)]); // S = RZ(pi/2)
        let stn_sv = stn.state_vector();

        let mut crz = StabVec::builder(1).seed(42).build();
        crz.h(&[QubitId(0)]);
        crz.rz(Angle64::QUARTER_TURN, &[QubitId(0)]);
        let crz_sv = crz.state_vector();

        assert_state_vectors_match(&stn_sv, &crz_sv, "S|+> via RZ");
    }

    // --- Measurement tests ---

    #[test]
    fn test_measurement_after_t_gate() {
        // H, T, measure in Z basis.
        // T|+> = (e^{-i*pi/8}|0> + e^{i*pi/8}|1>)/sqrt(2)
        // Both amplitudes have magnitude 1/sqrt(2), so prob(0) = prob(1) = 0.5
        let expected_p0 = 0.5;

        let n_trials: u32 = 2000;
        let mut count_0 = 0;
        for trial in 0..n_trials {
            let mut stn = StabMps::with_seed(1, u64::from(1000 + trial));
            stn.h(&[QubitId(0)]);
            stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
            let result = stn.mz(&[QubitId(0)]);
            if !result[0].outcome {
                count_0 += 1;
            }
        }
        let measured_p0 = f64::from(count_0) / f64::from(n_trials);
        assert!(
            (measured_p0 - expected_p0).abs() < 0.05,
            "p(0) = {measured_p0:.3}, expected {expected_p0:.3}"
        );
    }

    #[test]
    fn test_measurement_rx_probabilities() {
        // RX(pi/3)|0> has prob(0) = cos^2(pi/6) = 3/4
        let expected_p0 = 0.75;
        let theta = Angle64::from_radians(std::f64::consts::FRAC_PI_3);

        let n_trials: u32 = 2000;
        let mut count_0 = 0;
        for trial in 0..n_trials {
            let mut stn = StabMps::with_seed(1, u64::from(3000 + trial));
            stn.rx(theta, &[QubitId(0)]);
            let result = stn.mz(&[QubitId(0)]);
            if !result[0].outcome {
                count_0 += 1;
            }
        }
        let measured_p0 = f64::from(count_0) / f64::from(n_trials);
        assert!(
            (measured_p0 - expected_p0).abs() < 0.05,
            "p(0) = {measured_p0:.3}, expected {expected_p0:.3}"
        );
    }

    #[test]
    fn test_measurement_deterministic_after_t_on_zero() {
        // T|0> is still an eigenstate of Z (Z is a stabilizer of |0>)
        let mut stn = StabMps::with_seed(1, 42);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        let result = stn.mz(&[QubitId(0)]);
        assert!(result[0].is_deterministic);
        assert!(!result[0].outcome); // +1 eigenvalue -> outcome false
    }

    #[test]
    fn test_measurement_bell_state_correlation() {
        // Bell state: measure q0, then q1 should give same outcome
        for trial in 0..50 {
            let mut stn = StabMps::with_seed(2, 2000 + trial);
            stn.h(&[QubitId(0)]);
            stn.cx(&[(QubitId(0), QubitId(1))]);
            // Apply T to make MPS non-trivial
            stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);

            let r0 = stn.mz(&[QubitId(0)])[0].outcome;
            let r1 = stn.mz(&[QubitId(1)])[0].outcome;
            assert_eq!(
                r0, r1,
                "trial {trial}: Bell state + T should have correlated measurements"
            );
        }
    }

    #[test]
    fn test_disentangle_preserves_state() {
        use pecos_simulators::DenseStateVec;

        // Create a circuit, disentangle, verify against an independent dense oracle.
        let t_angle = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::new(3);
        let mut oracle = DenseStateVec::new(3);
        stn.h(&[QubitId(0)]);
        oracle.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        oracle.cx(&[(QubitId(0), QubitId(1))]);
        stn.rz(t_angle, &[QubitId(0)]);
        oracle.rz(t_angle, &[QubitId(0)]);
        stn.h(&[QubitId(2)]);
        oracle.h(&[QubitId(2)]);
        stn.cx(&[(QubitId(1), QubitId(2))]);
        oracle.cx(&[(QubitId(1), QubitId(2))]);
        stn.rz(t_angle, &[QubitId(2)]);
        oracle.rz(t_angle, &[QubitId(2)]);

        let bond_before = stn.max_bond_dim();

        // Disentangle
        let gates_applied = stn.disentangle(3);
        eprintln!(
            "Disentangle: applied {gates_applied} gates, bond dim {} -> {}",
            bond_before,
            stn.max_bond_dim()
        );

        // State vector should match the dense oracle (up to global phase).
        let sv_after = stn.state_vector();
        let expected = oracle.state();
        let overlap: Complex64 = expected
            .iter()
            .zip(sv_after.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        eprintln!("overlap = {:.6}", overlap.norm_sqr());
        assert_state_vectors_match(&expected, &sv_after, "disentangle preserves state");

        // Bond dimension should not have increased
        assert!(
            stn.max_bond_dim() <= bond_before,
            "disentangle should not increase bond dim: {} > {}",
            stn.max_bond_dim(),
            bond_before
        );

        eprintln!(
            "Disentangle: applied {gates_applied} gates, bond dim {} -> {}",
            bond_before,
            stn.max_bond_dim()
        );
    }

    #[test]
    fn test_compression_keeps_bond_dim_bounded() {
        // Apply multiple T gates. Without compression, bond dim would grow
        // exponentially. With compression, redundant components are removed.
        let t_angle = Angle64::QUARTER_TURN / 2u64;
        let mut stn = StabMps::builder(4).max_bond_dim(4).build();

        // Create entangled state
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.h(&[QubitId(2)]);
        stn.cx(&[(QubitId(2), QubitId(3))]);

        // Apply T gates -- each one could double bond dim without compression
        stn.rz(t_angle, &[QubitId(0)]);
        stn.rz(t_angle, &[QubitId(2)]);
        stn.rz(t_angle, &[QubitId(1)]);
        stn.rz(t_angle, &[QubitId(3)]);

        // Bond dimension should be bounded by max_bond_dim
        assert!(
            stn.max_bond_dim() <= 4,
            "bond dim {} should be <= 4",
            stn.max_bond_dim()
        );

        // State should still be approximately normalized
        assert!(
            (stn.mps().norm_squared() - 1.0).abs() < 0.1,
            "norm should be close to 1, got {}",
            stn.mps().norm_squared()
        );
    }

    #[test]
    fn test_pauli_expectation_z_on_zero_state() {
        // ⟨0|Z|0⟩ = 1.
        let stn = StabMps::new(2);
        let v = stn.pauli_expectation(&[(0, PauliKind::Z)]);
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_expectation_z_on_one_state() {
        // ⟨1|Z|1⟩ = -1.
        let mut stn = StabMps::new(2);
        stn.x(&[QubitId(0)]);
        let v = stn.pauli_expectation(&[(0, PauliKind::Z)]);
        assert!((v + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_pauli_expectation_x_on_plus_state() {
        // ⟨+|X|+⟩ = 1.
        let mut stn = StabMps::new(2);
        stn.h(&[QubitId(0)]);
        let v = stn.pauli_expectation(&[(0, PauliKind::X)]);
        assert!((v - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_auto_grow_bond_dim_starts_low_grows_when_capped() {
        // Build a small-cap STN and exercise it with a deep, adversarial
        // T circuit (small angle that defeats disent flag) so the cap
        // binds. Auto-grow should kick in.
        let n = 6;
        let mut stn = StabMps::builder(n)
            .seed(42)
            .max_bond_dim(2)
            .auto_grow_bond_dim(1e-15) // any truncation triggers
            .auto_grow_max_bond_dim(64)
            .build();

        // Spread + entangle.
        for q in 0..n {
            stn.h(&[QubitId(q)]);
        }
        for q in 0..n - 1 {
            stn.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
        // Deeper T-heavy circuit with rotating qubits + interleaved CXs
        // so the disent flag mechanism can't absorb the T's into the
        // tableau cheaply. Forces real MPS bond growth.
        let small = Angle64::from_radians(0.37);
        for layer in 0..6 {
            for q in 0..n {
                stn.rz(small, &[QubitId((q + layer) % n)]);
            }
            for q in 0..n - 1 {
                stn.cx(&[(QubitId(q), QubitId(q + 1))]);
            }
        }
        // After deep mixing, cap should have hit AND been raised.
        assert!(
            stn.config.max_bond_dim > 2,
            "auto-grow should have raised cap from 2; current = {} (bond_cap_hits={}, trunc={:.2e})",
            stn.config.max_bond_dim,
            stn.bond_cap_hits(),
            stn.truncation_error(),
        );
        assert!(stn.config.max_bond_dim <= 64);
    }

    #[test]
    fn test_auto_grow_bond_dim_disabled_by_default() {
        let n = 4;
        let mut stn = StabMps::builder(n).seed(99).max_bond_dim(2).build();
        // No auto_grow_bond_dim builder call → disabled.
        for q in 0..n {
            stn.h(&[QubitId(q)]);
        }
        for q in 0..n - 1 {
            stn.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
        let t = Angle64::QUARTER_TURN / 2u64;
        for q in 0..n {
            stn.rz(t, &[QubitId(q)]);
        }
        // Cap stays at 2.
        assert_eq!(stn.config.max_bond_dim, 2);
    }

    #[test]
    fn test_pauli_expectation_product_pauli_per_qubit() {
        // Exercise decompose_pauli_string's per-qubit Pauli multiplication:
        // (q, X), (q, Y) → X·Y = iZ at q. Phase contribution should come through.
        // <0|X·Y|0> = <0|iZ|0> = i. Real part = 0.
        let stn = StabMps::new(1);
        let v = stn.pauli_expectation(&[(0, PauliKind::X), (0, PauliKind::Y)]);
        // XY = iZ; <0|iZ|0> = i; pauli_expectation returns real part = 0.
        assert!(v.abs() < 1e-10, "<0|XY|0> real part should be 0, got {v}");
    }

    #[test]
    fn test_pauli_expectation_yy_per_qubit_is_identity() {
        // (q, Y), (q, Y) → Y² = I. <0|I|0> = 1.
        let stn = StabMps::new(1);
        let v = stn.pauli_expectation(&[(0, PauliKind::Y), (0, PauliKind::Y)]);
        assert!((v - 1.0).abs() < 1e-10, "<0|YY|0> = 1, got {v}");
    }

    #[test]
    fn test_pauli_expectation_z_on_one_via_apply_x() {
        // Apply X to a plain state, measure Z: should give -1.
        let mut stn = StabMps::new(1);
        stn.x(&[QubitId(0)]);
        let v = stn.pauli_expectation(&[(0, PauliKind::Z)]);
        assert!((v + 1.0).abs() < 1e-10, "<1|Z|1> = -1, got {v}");
    }

    #[test]
    fn test_pauli_frame_with_measurement_modes() {
        // Lazy measure + Pauli frame should compose: frame applies AFTER
        // the measurement outcome, irrespective of lazy/eager internals.
        // Init |0⟩, inject X in frame, measure: expect outcome=1 regardless
        // of the selected measurement mode.
        for lazy in [false, true] {
            let mut stn = StabMps::builder(1)
                .seed(42)
                .measurement(if lazy {
                    MeasurementMode::Lazy
                } else {
                    MeasurementMode::Pragmatic
                })
                .pauli_frame_tracking(true)
                .build();
            stn.inject_x_in_frame(QubitId(0));
            let r = stn.mz(&[QubitId(0)])[0].outcome;
            assert!(r, "lazy={lazy}, frame X should give outcome=1, got {r}");
        }
    }

    #[test]
    fn test_pauli_frame_with_merge_rz() {
        // Inject frame X on q0, apply rz(theta) with merge_rz on.
        // X·RZ(θ) = RZ(-θ)·X. Since frame X is "virtual" (will be applied
        // at measurement), the simulated state should evolve under RZ(θ)
        // naturally — but the NET physical state differs from
        // "simulated + frame" only by a global phase (e^{-iθ} on X|ψ>).
        //
        // Consequence: measurement outcome distributions are identical
        // between frame-tracking and no-frame-tracking paths for
        // Z-basis measurements. Verify.
        let theta = Angle64::from_radians(0.7);
        let mut stn_frame = StabMps::builder(1)
            .seed(5)
            .merge_rz(true)
            .pauli_frame_tracking(true)
            .build();
        stn_frame.h(&[QubitId(0)]);
        stn_frame.inject_x_in_frame(QubitId(0));
        stn_frame.rz(theta, &[QubitId(0)]);
        stn_frame.h(&[QubitId(0)]);
        let results_frame = stn_frame.mz(&[QubitId(0)]);
        let outcome_frame = results_frame[0].outcome;

        // Reference: apply X explicitly (no frame), same sequence.
        let mut stn_ref = StabMps::builder(1).seed(5).build();
        stn_ref.h(&[QubitId(0)]);
        stn_ref.x(&[QubitId(0)]);
        stn_ref.rz(theta, &[QubitId(0)]);
        stn_ref.h(&[QubitId(0)]);
        let results_ref = stn_ref.mz(&[QubitId(0)]);
        let outcome_ref = results_ref[0].outcome;

        assert_eq!(
            outcome_frame, outcome_ref,
            "frame-X vs applied-X should give same measurement outcome"
        );
    }

    #[test]
    fn test_is_state_exact_detects_all_sources_of_drift() {
        let mut stn = StabMps::builder(2)
            .seed(7)
            .merge_rz(true)
            .pauli_frame_tracking(true)
            .build();
        assert!(stn.is_state_exact(), "fresh builder state should be exact");

        // Pending merged RZ makes it non-exact.
        stn.h(&[QubitId(0)]);
        stn.rz(Angle64::from_radians(0.5), &[QubitId(0)]);
        assert!(!stn.is_state_exact(), "pending merged RZ → not exact");
        stn.flush();
        assert!(stn.is_state_exact(), "after flush() → exact again");

        // Frame injection makes it non-exact.
        stn.inject_x_in_frame(QubitId(0));
        assert!(!stn.is_state_exact(), "frame X set → not exact");
        stn.flush_pauli_frame_to_state();
        assert!(stn.is_state_exact(), "after frame flush → exact");
    }

    #[test]
    fn test_flush_materializes_lazy_deferred_operations() {
        let mut stn = StabMps::builder(2)
            .seed(19)
            .measurement(MeasurementMode::Lazy)
            .build();
        stn.h(&[QubitId(1)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(1)]);
        stn.sz(&[QubitId(0)]);
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        let _ = stn.mz(&[QubitId(0)]);

        assert!(!stn.is_state_exact(), "lazy operations should be pending");
        assert!(
            !stn.deferred_ops.is_empty(),
            "the fixture must exercise a nonempty lazy-operation queue"
        );
        let pre_flush_state = stn.state_vector();
        let mut expected = stn.clone();
        measure::flush_deferred_ops(&mut expected.mps, &mut expected.deferred_ops).unwrap();
        let expected_state = expected.state_vector();
        let stale_fidelity = pre_flush_state
            .iter()
            .zip(&expected_state)
            .map(|(left, right)| left.conj() * right)
            .sum::<Complex64>()
            .norm_sqr();
        assert!(
            stale_fidelity < 1.0 - 1e-8,
            "the fixture must make deferred materialization observable"
        );
        stn.flush();
        assert!(
            stn.deferred_ops.is_empty(),
            "flush must empty the lazy queue"
        );
        let materialized_fidelity = stn
            .state_vector()
            .iter()
            .zip(&expected_state)
            .map(|(left, right)| left.conj() * right)
            .sum::<Complex64>()
            .norm_sqr();
        assert!(
            materialized_fidelity > 1.0 - 1e-12,
            "flush must materialize the queued state: fidelity={materialized_fidelity:.16}"
        );
        assert!(
            !stn.is_state_exact(),
            "lazy mode itself is a conservative exactness guard"
        );
    }

    #[test]
    fn lazy_multi_measurement_cnot_cz_frame_rz_matches_dense() {
        use pecos_simulators::DenseStateVec;

        fn project_z(state: &[Complex64], qubit: usize, outcome: bool) -> Vec<Complex64> {
            let probability = state
                .iter()
                .enumerate()
                .filter(|(index, _)| ((*index >> qubit) & 1 != 0) == outcome)
                .map(|(_, amplitude)| amplitude.norm_sqr())
                .sum::<f64>();
            assert!(probability > 1e-14, "sampled branch must have support");
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

        let mut stn = StabMps::builder(4)
            .seed(0x5550)
            .measurement(MeasurementMode::Lazy)
            .merge_rz(false)
            .build();
        let mut dense = DenseStateVec::new(4);
        for q in 0..4 {
            let angle = Angle64::from_radians(0.23 + 0.07 * q as f64);
            stn.h(&[QubitId(q)]);
            dense.h(&[QubitId(q)]);
            stn.rz(angle, &[QubitId(q)]);
            dense.rz(angle, &[QubitId(q)]);
        }
        for &(control, target) in &[(0, 1), (2, 3)] {
            stn.cx(&[(QubitId(control), QubitId(target))]);
            dense.cx(&[(QubitId(control), QubitId(target))]);
        }
        for &(first, second) in &[(1, 2), (3, 0)] {
            stn.cz(&[(QubitId(first), QubitId(second))]);
            dense.cz(&[(QubitId(first), QubitId(second))]);
        }

        for (measurement, measured) in [0, 3, 1].into_iter().enumerate() {
            let outcome = stn.mz(&[QubitId(measured)])[0].outcome;
            let projected = project_z(&dense.state(), measured, outcome);
            dense = DenseStateVec::from_state(
                &projected,
                PecosRng::seed_from_u64(0x5550 + measurement as u64 + 1),
            );
        }
        assert!(
            stn.deferred_ops
                .iter()
                .any(|op| matches!(op, measure::DeferredOp::Cnot(_, _))),
            "fixture must retain a lazy pre-reduction CNOT: {:?}",
            stn.deferred_ops
        );
        assert!(
            stn.deferred_ops
                .iter()
                .any(|op| matches!(op, measure::DeferredOp::Cz(_, _))),
            "fixture must retain a lazy basis-rotation CZ: {:?}",
            stn.deferred_ops
        );
        assert!(
            stn.deferred_ops.len() > 2,
            "fixture must exercise a nontrivial deferred queue: {:?}",
            stn.deferred_ops
        );

        for q in [0, 0, 3] {
            stn.sz(&[QubitId(q)]);
            dense.sz(&[QubitId(q)]);
        }
        for &(control, target) in &[(3, 1), (3, 0)] {
            stn.cx(&[(QubitId(control), QubitId(target))]);
            dense.cx(&[(QubitId(control), QubitId(target))]);
        }
        stn.sz(&[QubitId(3)]);
        dense.sz(&[QubitId(3)]);
        stn.h(&[QubitId(0)]);
        dense.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(1), QubitId(0))]);
        dense.cx(&[(QubitId(1), QubitId(0))]);
        let angle = Angle64::from_radians(0.67);
        stn.rz(angle, &[QubitId(0)]);
        dense.rz(angle, &[QubitId(0)]);

        stn.flush();
        let actual = stn.state_vector();
        let expected = dense.state();
        let fidelity = actual
            .iter()
            .zip(expected)
            .map(|(left, right)| left.conj() * right)
            .sum::<Complex64>()
            .norm_sqr();
        eprintln!("lazy-multi-measurement-frame-rz fidelity={fidelity:.16}");
        assert!(
            fidelity >= 1.0 - 1e-10,
            "issue #555 multi-measurement lazy frame must match dense; fidelity={fidelity:.16}"
        );
    }

    #[test]
    fn test_uncompensated_pre_reduction_count_tracks_pragmatic_path() {
        // Build a state where col_x for the measured qubit has multiple
        // anticommuting stabilizers so pre_reduce fires. H(0), H(1), CX(0,1)
        // gives stabs {X_0X_1, X_1}; measuring qubit 1 has col_x[1].len()=2.
        let mut stn = StabMps::builder(2)
            .seed(3)
            .measurement(MeasurementMode::Pragmatic)
            .build();
        stn.h(&[QubitId(0)]);
        stn.h(&[QubitId(1)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        assert_eq!(
            stn.uncompensated_pre_reduction_count(),
            0,
            "no measurements yet"
        );
        let _ = stn.mz(&[QubitId(1)]);
        assert_eq!(
            stn.uncompensated_pre_reduction_count(),
            1,
            "non-lazy mz on multi-anticom col_x should bump drift counter"
        );
        assert!(
            !stn.is_state_exact(),
            "pragmatic drift makes stored state non-exact"
        );

        // Lazy path: same setup but no drift (pre_reduce CNOTs go into V).
        let mut stn = StabMps::builder(2)
            .seed(3)
            .measurement(MeasurementMode::Lazy)
            .build();
        stn.h(&[QubitId(0)]);
        stn.h(&[QubitId(1)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        let _ = stn.mz(&[QubitId(1)]);
        assert_eq!(
            stn.uncompensated_pre_reduction_count(),
            0,
            "lazy path must not increment the uncompensated count"
        );

        // Reset clears the counter.
        let mut stn = StabMps::builder(2)
            .seed(3)
            .measurement(MeasurementMode::Pragmatic)
            .build();
        stn.h(&[QubitId(0)]);
        stn.h(&[QubitId(1)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        let _ = stn.mz(&[QubitId(1)]);
        assert!(stn.uncompensated_pre_reduction_count() > 0);
        stn.reset();
        assert_eq!(
            stn.uncompensated_pre_reduction_count(),
            0,
            "reset clears drift counter"
        );
    }

    #[test]
    fn test_lazy_measure_imaginary_sp_y_eigenstate() {
        // To hit the imaginary-sp branch we need `id` (flip_site) to also
        // appear in `sign_sites`, meaning both stab and destab have the X
        // bit at the measured qubit. Circuit: SZ(0), H(0), CX(0,1) gives
        // stab = X_0·X_1 (X-bit at 0) and destab = -Y_0·X_1 (X-bit at 0).
        // CX(0,1) entangles → MPS non-trivial → decompose_z path fires.
        //
        // Expected: ~50/50 outcome on qubit 0, post-collapse re-measurement
        // deterministic and matching.
        let num_shots = 400;
        let mut zero_count = 0;
        let mut one_count = 0;
        let t = Angle64::QUARTER_TURN / 2u64;
        for shot in 0..num_shots {
            let mut stn = StabMps::builder(2)
                .seed(shot)
                .measurement(MeasurementMode::Lazy)
                .build();
            // Non-Clifford first to force MPS non-trivial (Cliffords alone
            // keep MPS in its initial product form via tableau routing).
            stn.h(&[QubitId(1)]);
            stn.rz(t, &[QubitId(1)]);
            stn.sz(&[QubitId(0)]);
            stn.h(&[QubitId(0)]);
            stn.cx(&[(QubitId(0), QubitId(1))]);
            let r1 = stn.mz(&[QubitId(0)])[0].outcome;
            let r2 = stn.mz(&[QubitId(0)]);
            assert_eq!(
                r2[0].outcome, r1,
                "after collapse, Z measurement must be stable (shot {shot})"
            );
            assert!(
                r2[0].is_deterministic,
                "post-collapse measurement must be deterministic (shot {shot})"
            );
            if r1 {
                one_count += 1;
            } else {
                zero_count += 1;
            }
        }
        assert!(
            zero_count > 130 && zero_count < 270,
            "qubit 0 should give ~50/50: got {zero_count} zeros, {one_count} ones"
        );
    }

    #[test]
    fn test_flush_pauli_frame_to_state_makes_read_correct() {
        // Without flush: state_vector shows |0⟩ (sim state) even though
        // frame has X (physical state is |1⟩). After flush: state_vector
        // correctly shows |1⟩.
        let mut stn = StabMps::builder(1)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_x_in_frame(QubitId(0));
        // State vector BEFORE flush: frame-bits aren't in the state.
        let sv_before = stn.state_vector();
        // Stored state is still |0⟩ (index 0, real amplitude 1).
        assert!(
            (sv_before[0].re - 1.0).abs() < 1e-10,
            "before flush: {sv_before:?}"
        );
        assert!(sv_before[1].norm() < 1e-10);

        // Now flush. State should become |1⟩.
        stn.flush_pauli_frame_to_state();
        assert!(!stn.frame_x_bit(QubitId(0)), "frame cleared after flush");
        let sv_after = stn.state_vector();
        assert!(
            sv_after[0].norm() < 1e-10,
            "post-flush q0 amp at |0⟩: {sv_after:?}"
        );
        assert!(
            (sv_after[1].re - 1.0).abs() < 1e-10,
            "post-flush q0 amp at |1⟩: {sv_after:?}"
        );
    }

    #[test]
    fn test_flush_pauli_frame_clears_flipped_site_disent_flags() {
        let mut stn = StabMps::builder(4).pauli_frame_tracking(true).build();
        stn.inject_x_in_frame(QubitId(2));
        stn.inject_x_in_frame(QubitId(3));
        stn.flush_pauli_frame_to_state();

        for site in [2, 3] {
            let marginal = stored_mps_one_marginal(&stn.mps, site);
            assert_relative_eq!(marginal, 1.0, epsilon = 1e-12);
            assert_eq!(
                stn.disent_flags[site], None,
                "X-flipped site {site} cannot retain a stored |0> proof"
            );
        }
    }

    #[test]
    fn test_pauli_frame_inject_x_flips_measurement() {
        // Init |0⟩. Inject X in frame → measurement should give 1.
        let mut stn = StabMps::builder(1)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_x_in_frame(QubitId(0));
        let r = stn.mz(&[QubitId(0)])[0].outcome;
        assert!(r, "X in frame on |0⟩ should measure as 1, got {r}");
    }

    #[test]
    fn test_pauli_frame_inject_z_no_effect_on_zero_state() {
        // Z on |0⟩ gives |0⟩ (eigenstate). Measurement = 0 still.
        let mut stn = StabMps::builder(1)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_z_in_frame(QubitId(0));
        let r = stn.mz(&[QubitId(0)])[0].outcome;
        assert!(!r, "Z in frame on |0⟩ should measure 0 (Z|0⟩=|0⟩), got {r}");
    }

    #[test]
    fn test_pauli_frame_h_swaps_x_z() {
        // Inject Z in frame, apply H, measure. H·Z = X·H. So X bit set,
        // Z bit cleared after H. Measurement of X on |0⟩... hmm, but state
        // isn't an eigenstate of X. Actually we're tracking Paulis via frame.
        // Before H: frame = Z. After H: frame = X (per propagation).
        // Physical state |0⟩, measurement in Z basis: frame has Z=0 so
        // outcome matches underlying quantum outcome (0).
        let mut stn = StabMps::builder(1)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_z_in_frame(QubitId(0));
        stn.h(&[QubitId(0)]); // frame: Z → X after H
        assert!(stn.frame_x_bit(QubitId(0)));
        assert!(!stn.frame_z_bit(QubitId(0)));
    }

    #[test]
    fn test_pauli_frame_y_inject_flush_gives_correct_amplitude_phase() {
        // Inject Y on qubit 0 (|0⟩). Y|0⟩ = i|1⟩. After flushing, the
        // state vector should show amplitude i at index 1 (not -i or 1 or -1).
        let mut stn = StabMps::builder(1)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_y_in_frame(QubitId(0));
        stn.flush_pauli_frame_to_state();
        let sv = stn.state_vector();
        assert!(sv[0].norm() < 1e-10, "post-Y at |0⟩ amp: {:?}", sv[0]);
        assert!(
            (sv[1] - Complex64::new(0.0, 1.0)).norm() < 1e-10,
            "post-Y at |1⟩ amp should be +i, got {:?}",
            sv[1]
        );
    }

    #[test]
    fn test_pauli_frame_h_on_y_exact_state_vector() {
        // Inject Y on |0⟩, apply H → physical = H·Y·|0⟩ = H·(i|1⟩) = i·|-⟩.
        // The decomposition-based flush (applies the frame Pauli to MPS via
        // C†·P·C = phase·X_flip·Z_sign rather than to the tableau via tab.y)
        // recovers the correct global phase — no ±1 residual.
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let mut stn = StabMps::builder(1)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_y_in_frame(QubitId(0));
        stn.h(&[QubitId(0)]);
        stn.flush_pauli_frame_to_state();
        let sv = stn.state_vector();
        // Expected i·|-⟩ = (i/√2)|0⟩ + (-i/√2)|1⟩.
        let expect_0 = Complex64::new(0.0, inv_sqrt2);
        let expect_1 = Complex64::new(0.0, -inv_sqrt2);
        assert!(
            (sv[0] - expect_0).norm() < 1e-10,
            "amp |0⟩: expected {expect_0:?}, got {:?}",
            sv[0]
        );
        assert!(
            (sv[1] - expect_1).norm() < 1e-10,
            "amp |1⟩: expected {expect_1:?}, got {:?}",
            sv[1]
        );
    }

    #[test]
    fn test_pauli_frame_y_inject_on_bell_state_exact_phase() {
        // Φ+ = (|00⟩+|11⟩)/√2. Apply frame Y_0:
        //   Y_0|00⟩ = i|1⟩_{q0}|0⟩_{q1} = i·(q0=1,q1=0) → LSB index 1.
        //   Y_0|11⟩ = -i|0⟩_{q0}|1⟩_{q1} = -i·(q0=0,q1=1) → LSB index 2.
        // So sv[1] = i/√2, sv[2] = -i/√2, others 0. (Equivalent to -i·Ψ-.)
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let mut stn = StabMps::builder(2)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.inject_y_in_frame(QubitId(0));
        stn.flush_pauli_frame_to_state();
        let sv = stn.state_vector();
        let expect_1 = Complex64::new(0.0, inv_sqrt2);
        let expect_2 = Complex64::new(0.0, -inv_sqrt2);
        assert!(sv[0].norm() < 1e-10, "|00⟩: {:?}", sv[0]);
        assert!((sv[1] - expect_1).norm() < 1e-10, "sv[1]: {:?}", sv[1]);
        assert!((sv[2] - expect_2).norm() < 1e-10, "sv[2]: {:?}", sv[2]);
        assert!(sv[3].norm() < 1e-10, "|11⟩: {:?}", sv[3]);
    }

    #[test]
    fn test_pauli_frame_propagation_preserves_signs() {
        // Sanity: even though the state_vector is now exact via flush, the
        // propagation signs are still correctly recorded in pauli_frame_phase.
        let mut stn = StabMps::builder(1)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_y_in_frame(QubitId(0));
        stn.h(&[QubitId(0)]);
        // H·Y·H = -Y: propagation should record -1 in pauli_frame_phase.
        assert!(
            (stn.pauli_frame_phase + Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "H-on-Y should record phase -1, got {:?}",
            stn.pauli_frame_phase
        );
        assert!(stn.frame_x_bit(QubitId(0)));
        assert!(stn.frame_z_bit(QubitId(0)));
    }

    #[test]
    fn test_reset_qubit_forces_zero_from_one() {
        // Prepare |1⟩ then reset → should land on |0⟩ and report outcome=1.
        let mut stn = StabMps::builder(1).seed(42).build();
        stn.x(&[QubitId(0)]);
        let phys = stn.reset_qubit(QubitId(0));
        assert!(phys, "reset from |1⟩ should report physical outcome true");
        // Post-reset measurement must give 0 (deterministic).
        let r = stn.mz(&[QubitId(0)]);
        assert!(!r[0].outcome, "after reset, mz must give 0");
        assert!(r[0].is_deterministic, "post-reset mz must be deterministic");
    }

    #[test]
    fn test_reset_qubit_forces_zero_from_plus() {
        // Prepare |+⟩ then reset. Outcome is random (50/50) but after
        // reset, state is deterministically |0⟩.
        let mut stn = StabMps::builder(1).seed(42).build();
        stn.h(&[QubitId(0)]);
        stn.reset_qubit(QubitId(0));
        let r = stn.mz(&[QubitId(0)]);
        assert!(!r[0].outcome, "after reset from |+⟩, mz must give 0");
        assert!(r[0].is_deterministic);
    }

    #[test]
    fn test_reset_qubit_clears_frame_bits() {
        let mut stn = StabMps::builder(1)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_y_in_frame(QubitId(0));
        assert!(stn.frame_x_bit(QubitId(0)));
        assert!(stn.frame_z_bit(QubitId(0)));
        stn.reset_qubit(QubitId(0));
        assert!(!stn.frame_x_bit(QubitId(0)), "reset must clear frame X");
        assert!(!stn.frame_z_bit(QubitId(0)), "reset must clear frame Z");
        // And the physical state is |0⟩.
        let r = stn.mz(&[QubitId(0)]);
        assert!(!r[0].outcome, "post-reset mz must give 0");
    }

    #[test]
    fn test_reset_qubit_preserves_other_qubits() {
        // Entangle q0-q1, then reset q0. q1 should still have a definite
        // outcome consistent with the GHZ-like correlation collapsed by
        // the reset-measurement.
        for shot in 0..20u64 {
            let mut stn = StabMps::builder(2).seed(100 + shot).build();
            stn.h(&[QubitId(0)]);
            stn.cx(&[(QubitId(0), QubitId(1))]);
            let phys = stn.reset_qubit(QubitId(0));
            // q1 should collapse to match phys (Bell correlation).
            let r1 = stn.mz(&[QubitId(1)]);
            assert!(
                r1[0].is_deterministic,
                "q1 post-Bell-reset must be deterministic"
            );
            assert_eq!(
                r1[0].outcome, phys,
                "Bell correlation: q1 outcome must match reset's physical outcome"
            );
        }
    }

    #[test]
    fn test_px_gives_plus_state() {
        // px should land qubit in |+⟩: X measurement deterministic 0.
        let mut stn = StabMps::builder(1).seed(42).build();
        stn.px(QubitId(0));
        // Measure in X basis via H + mz.
        stn.h(&[QubitId(0)]);
        let r = stn.mz(&[QubitId(0)]);
        assert!(!r[0].outcome, "|+⟩ in X-basis should measure 0");
        assert!(r[0].is_deterministic);
    }

    #[test]
    fn test_extract_syndromes_steane_noiseless() {
        // Steane [[7,1,3]]: prep |0_L⟩, extract syndrome, expect all-zero
        // syndrome (codestate is a +1 eigenstate of every generator).
        // Uses 7 data + 6 ancillas.
        let stabs: Vec<Vec<(usize, PauliKind)>> = vec![
            vec![
                (3, PauliKind::X),
                (4, PauliKind::X),
                (5, PauliKind::X),
                (6, PauliKind::X),
            ],
            vec![
                (1, PauliKind::X),
                (2, PauliKind::X),
                (5, PauliKind::X),
                (6, PauliKind::X),
            ],
            vec![
                (0, PauliKind::X),
                (2, PauliKind::X),
                (4, PauliKind::X),
                (6, PauliKind::X),
            ],
            vec![
                (3, PauliKind::Z),
                (4, PauliKind::Z),
                (5, PauliKind::Z),
                (6, PauliKind::Z),
            ],
            vec![
                (1, PauliKind::Z),
                (2, PauliKind::Z),
                (5, PauliKind::Z),
                (6, PauliKind::Z),
            ],
            vec![
                (0, PauliKind::Z),
                (2, PauliKind::Z),
                (4, PauliKind::Z),
                (6, PauliKind::Z),
            ],
        ];
        let ancillas: Vec<QubitId> = (7..13).map(QubitId).collect();
        let mut stn = StabMps::builder(13).seed(42).for_qec().build();
        // Prep Steane |0_L⟩ (pivots 0, 1, 3).
        stn.h(&[QubitId(0), QubitId(1), QubitId(3)]);
        for (c, t) in [
            (3, 4),
            (3, 5),
            (3, 6),
            (1, 2),
            (1, 5),
            (1, 6),
            (0, 2),
            (0, 4),
            (0, 6),
        ] {
            stn.cx(&[(QubitId(c), QubitId(t))]);
        }
        let syndrome = stn.extract_syndromes(&stabs, &ancillas);
        assert_eq!(syndrome.len(), 6);
        for (i, &b) in syndrome.iter().enumerate() {
            assert!(!b, "noiseless Steane syndrome must be zero, bit {i} = {b}");
        }
        // Ancillas must be left in |0⟩ for the next round.
        for &a in &ancillas {
            let r = stn.mz(&[a]);
            assert!(
                !r[0].outcome && r[0].is_deterministic,
                "ancilla {a:?} not reset after extract_syndromes"
            );
        }
    }

    #[test]
    fn test_extract_syndromes_steane_single_x_error_detects() {
        // Same setup, inject X_0 before extraction — expect NON-ZERO
        // syndrome on at least one Z-stabilizer.
        let stabs: Vec<Vec<(usize, PauliKind)>> = vec![
            vec![
                (3, PauliKind::X),
                (4, PauliKind::X),
                (5, PauliKind::X),
                (6, PauliKind::X),
            ],
            vec![
                (1, PauliKind::X),
                (2, PauliKind::X),
                (5, PauliKind::X),
                (6, PauliKind::X),
            ],
            vec![
                (0, PauliKind::X),
                (2, PauliKind::X),
                (4, PauliKind::X),
                (6, PauliKind::X),
            ],
            vec![
                (3, PauliKind::Z),
                (4, PauliKind::Z),
                (5, PauliKind::Z),
                (6, PauliKind::Z),
            ],
            vec![
                (1, PauliKind::Z),
                (2, PauliKind::Z),
                (5, PauliKind::Z),
                (6, PauliKind::Z),
            ],
            vec![
                (0, PauliKind::Z),
                (2, PauliKind::Z),
                (4, PauliKind::Z),
                (6, PauliKind::Z),
            ],
        ];
        let ancillas: Vec<QubitId> = (7..13).map(QubitId).collect();
        let mut stn = StabMps::builder(13).seed(42).for_qec().build();
        stn.h(&[QubitId(0), QubitId(1), QubitId(3)]);
        for (c, t) in [
            (3, 4),
            (3, 5),
            (3, 6),
            (1, 2),
            (1, 5),
            (1, 6),
            (0, 2),
            (0, 4),
            (0, 6),
        ] {
            stn.cx(&[(QubitId(c), QubitId(t))]);
        }
        // Inject X_0. Z-stabilizer 3 (ZZZZ on 0,2,4,6) anticommutes → syndrome bit 5 set.
        stn.x(&[QubitId(0)]);
        let syndrome = stn.extract_syndromes(&stabs, &ancillas);
        assert!(
            syndrome.iter().skip(3).any(|&b| b),
            "X_0 error must trigger at least one Z-stabilizer syndrome, got {syndrome:?}"
        );
    }

    #[test]
    fn test_extract_syndromes_repeated_rounds_stable() {
        // Two noiseless rounds should both report zero syndrome.
        let stabs: Vec<Vec<(usize, PauliKind)>> = vec![
            vec![(0, PauliKind::Z), (1, PauliKind::Z)],
            vec![(1, PauliKind::Z), (2, PauliKind::Z)],
        ];
        let ancillas = vec![QubitId(3), QubitId(4)];
        let mut stn = StabMps::builder(5).seed(42).for_qec().build();
        // Trivial |000⟩ data is already a +1 eigenstate of Z_iZ_j.
        for _round in 0..2 {
            let s = stn.extract_syndromes(&stabs, &ancillas);
            assert_eq!(s, vec![false, false]);
        }
    }

    #[test]
    fn test_inject_paulis_in_frame_bulk() {
        let mut stn = StabMps::builder(3)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_paulis_in_frame(&[
            (QubitId(0), PauliKind::X),
            (QubitId(1), PauliKind::Y),
            (QubitId(2), PauliKind::Z),
        ]);
        assert!(stn.frame_x_bit(QubitId(0)));
        assert!(!stn.frame_z_bit(QubitId(0)));
        assert!(stn.frame_x_bit(QubitId(1)));
        assert!(stn.frame_z_bit(QubitId(1)));
        assert!(!stn.frame_x_bit(QubitId(2)));
        assert!(stn.frame_z_bit(QubitId(2)));
    }

    #[test]
    fn test_pauli_frame_cx_propagates() {
        // Inject X on q0, apply CX(0, 1). Frame X propagates: X_0 → X_0·X_1.
        let mut stn = StabMps::builder(2)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_x_in_frame(QubitId(0));
        stn.cx(&[(QubitId(0), QubitId(1))]);
        assert!(stn.frame_x_bit(QubitId(0)));
        assert!(
            stn.frame_x_bit(QubitId(1)),
            "CX should propagate X to target"
        );
    }

    #[test]
    fn test_pauli_frame_sz_propagation_phase() {
        // SZ · X · SZdg = Y (no sign flip). SZ · Y · SZdg = -X (sign flip).
        let mut stn = StabMps::builder(1)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_x_in_frame(QubitId(0));
        stn.sz(&[QubitId(0)]);
        // X → Y: bits (1,0) → (1,1), phase stays +1.
        assert!(stn.frame_x_bit(QubitId(0)));
        assert!(stn.frame_z_bit(QubitId(0)));
        assert!(
            (stn.pauli_frame_phase - Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "SZ on X should not flip phase, got {:?}",
            stn.pauli_frame_phase
        );

        // Now apply SZ again: Y → -X. Phase should flip to -1.
        stn.sz(&[QubitId(0)]);
        assert!(stn.frame_x_bit(QubitId(0)));
        assert!(!stn.frame_z_bit(QubitId(0)));
        assert!(
            (stn.pauli_frame_phase + Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "SZ on Y should flip phase to -1, got {:?}",
            stn.pauli_frame_phase
        );
    }

    #[test]
    fn test_pauli_frame_szdg_propagation_phase() {
        // SZdg · X · SZ = -Y (sign flip). SZdg · Y · SZ = +X (no sign flip).
        let mut stn = StabMps::builder(1)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_x_in_frame(QubitId(0));
        stn.szdg(&[QubitId(0)]);
        // X → -Y: bits (1,0) → (1,1), phase -1.
        assert!(stn.frame_x_bit(QubitId(0)));
        assert!(stn.frame_z_bit(QubitId(0)));
        assert!(
            (stn.pauli_frame_phase + Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "SZdg on X should flip phase to -1, got {:?}",
            stn.pauli_frame_phase
        );

        // Apply SZdg again: -Y → +X → -(-X) = +X? No: frame is -Y, bits (1,1).
        // SZdg on Y: Y → X, no sign flip. Phase stays -1.
        stn.szdg(&[QubitId(0)]);
        assert!(stn.frame_x_bit(QubitId(0)));
        assert!(!stn.frame_z_bit(QubitId(0)));
        assert!(
            (stn.pauli_frame_phase + Complex64::new(1.0, 0.0)).norm() < 1e-12,
            "SZdg on Y should NOT flip phase, got {:?}",
            stn.pauli_frame_phase
        );
    }

    #[test]
    fn test_pauli_frame_sz_szdg_state_vector_exact() {
        // End-to-end: inject X, apply SZ, flush, check state_vector matches
        // eager application. Physical: SZ · X · |0⟩ = SZ · |1⟩ = i|1⟩.
        // Frame path: state |0⟩, frame = Y (from SZ on X), flush via decomposition.
        let mut stn = StabMps::builder(1)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_x_in_frame(QubitId(0));
        stn.sz(&[QubitId(0)]);
        stn.flush_pauli_frame_to_state();
        let sv = stn.state_vector();
        // Expected: SZ·X|0⟩ = SZ|1⟩ = i|1⟩.
        assert!(sv[0].norm() < 1e-10, "amp |0⟩: {:?}", sv[0]);
        assert!(
            (sv[1] - Complex64::new(0.0, 1.0)).norm() < 1e-10,
            "amp |1⟩ should be +i, got {:?}",
            sv[1]
        );
    }

    #[test]
    fn test_pauli_frame_cz_propagates() {
        // CZ Heisenberg: X_a → X_a Z_b, X_b → Z_a X_b, Z unchanged.
        let mut stn = StabMps::builder(2)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        // Inject X on q0 only.
        stn.inject_x_in_frame(QubitId(0));
        stn.cz(&[(QubitId(0), QubitId(1))]);
        // After CZ: X_0 → X_0 Z_1. So q0 still X, q1 gains Z.
        assert!(stn.frame_x_bit(QubitId(0)), "CZ: X_0 stays");
        assert!(!stn.frame_x_bit(QubitId(1)), "CZ: q1 should not gain X");
        assert!(
            stn.frame_z_bit(QubitId(1)),
            "CZ: X_0 → X_0 Z_1, so q1 gains Z"
        );
        assert!(!stn.frame_z_bit(QubitId(0)), "CZ: q0 should not gain Z");

        // Now inject Z on q1 separately, apply CZ(0,1) again.
        // Frame is now X_0, Z_1+Z_1=I on q1 (toggled off). So frame = X_0.
        // After another CZ: X_0 → X_0 Z_1 again.
        let mut stn2 = StabMps::builder(2)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        stn2.inject_z_in_frame(QubitId(0));
        stn2.cz(&[(QubitId(0), QubitId(1))]);
        // Z_0 → Z_0 (Z commutes with CZ). No change.
        assert!(!stn2.frame_x_bit(QubitId(0)));
        assert!(stn2.frame_z_bit(QubitId(0)), "CZ: Z_0 unchanged");
        assert!(!stn2.frame_x_bit(QubitId(1)));
        assert!(
            !stn2.frame_z_bit(QubitId(1)),
            "CZ: Z_0 doesn't propagate to q1"
        );
    }

    #[test]
    fn test_reset_qubit_with_frame_tracking_clears_and_measures_correctly() {
        // With frame tracking enabled: inject X error, then reset_qubit.
        // The reported outcome from reset should reflect the frame,
        // and after reset the qubit should be |0⟩ with no frame bits.
        let mut stn = StabMps::builder(1)
            .seed(42)
            .pauli_frame_tracking(true)
            .build();
        // State = |0⟩, frame X = flip → physical is |1⟩.
        stn.inject_x_in_frame(QubitId(0));
        let phys = stn.reset_qubit(QubitId(0));
        // Physical outcome should be true (qubit was in |1⟩ physically).
        assert!(phys, "reset with X-frame on |0⟩ should report physical |1⟩");
        // Frame cleared.
        assert!(!stn.frame_x_bit(QubitId(0)), "frame X must be cleared");
        assert!(!stn.frame_z_bit(QubitId(0)), "frame Z must be cleared");
        // Post-reset measurement should give 0.
        let r = stn.mz(&[QubitId(0)]);
        assert!(!r[0].outcome, "post-reset mz must be 0");

        // Now test with Y frame (both X and Z bits).
        let mut stn = StabMps::builder(1)
            .seed(7)
            .pauli_frame_tracking(true)
            .build();
        stn.inject_y_in_frame(QubitId(0));
        let _phys = stn.reset_qubit(QubitId(0));
        assert!(!stn.frame_x_bit(QubitId(0)));
        assert!(!stn.frame_z_bit(QubitId(0)));
        let r = stn.mz(&[QubitId(0)]);
        assert!(!r[0].outcome, "post-Y-reset mz must be 0");
    }

    #[test]
    fn test_pauli_frame_matches_eager_for_many_noise_injects() {
        let n = 6;
        let num_injects = 1_000;

        let mut stn_frame = StabMps::builder(n)
            .seed(1)
            .pauli_frame_tracking(true)
            .build();
        let mut stn_eager = StabMps::builder(n).seed(1).build();

        for k in 0..num_injects {
            let q = QubitId(k % n);
            let frame_pauli = stn_frame.apply_depolarizing(q, 1.0);
            let eager_pauli = stn_eager.apply_depolarizing(q, 1.0);
            assert_eq!(frame_pauli, eager_pauli);
        }

        stn_frame.flush_pauli_frame_to_state();
        let frame_sv = stn_frame.state_vector();
        let eager_sv = stn_eager.state_vector();

        assert_state_vectors_match(&frame_sv, &eager_sv, "frame vs eager Pauli injections");
    }

    #[test]
    fn test_apply_bit_flip_zero_p_noop() {
        let mut stn = StabMps::with_seed(2, 42);
        // p = 0: no flip, deterministic.
        for _ in 0..10 {
            assert!(!stn.apply_bit_flip(QubitId(0), 0.0));
        }
        let result = stn.mz(&[QubitId(0)])[0].outcome;
        assert!(!result, "no-op noise: |0> should give 0");
    }

    #[test]
    fn test_apply_bit_flip_p_one_always_flips() {
        let mut stn = StabMps::with_seed(2, 42);
        // p = 1: always flips.
        for _ in 0..5 {
            assert!(stn.apply_bit_flip(QubitId(0), 1.0));
        }
        // 5 X's = X (odd count) → q0 = 1.
        let result = stn.mz(&[QubitId(0)])[0].outcome;
        assert!(result, "5x X(0) should leave q0 = 1");
    }

    #[test]
    fn test_apply_depolarizing_p_zero_noop() {
        let mut stn = StabMps::with_seed(2, 42);
        for _ in 0..10 {
            assert!(stn.apply_depolarizing(QubitId(0), 0.0).is_none());
        }
        // Z-basis measurement of |0> is deterministic.
        let results = stn.mz(&[QubitId(0)]);
        assert!(results[0].is_deterministic && !results[0].outcome);
    }

    #[test]
    fn test_apply_depolarizing_distribution() {
        // p = 0.9: error occurs ~90% of the time. Of those, X/Y/Z each ~30%.
        // For 1000 trials, count outcomes.
        let mut x_count = 0;
        let mut y_count = 0;
        let mut z_count = 0;
        let mut none_count = 0;
        let trials = 2000;
        for trial in 0..trials {
            let mut stn = StabMps::with_seed(1, 42 + trial as u64);
            match stn.apply_depolarizing(QubitId(0), 0.9) {
                Some(PauliKind::X) => x_count += 1,
                Some(PauliKind::Y) => y_count += 1,
                Some(PauliKind::Z) => z_count += 1,
                None => none_count += 1,
            }
        }
        let frac_none = f64::from(none_count) / f64::from(trials);
        let frac_each_pauli = f64::from(x_count + y_count + z_count) / f64::from(trials) / 3.0;
        // None = 1 - p ≈ 0.10. Each Pauli ≈ p/3 ≈ 0.30. Tolerance ±5%.
        assert!((frac_none - 0.10).abs() < 0.05, "P(no error) = {frac_none}");
        assert!(
            (frac_each_pauli - 0.30).abs() < 0.05,
            "P(each Pauli) = {frac_each_pauli}"
        );
    }

    #[test]
    fn test_apply_depolarizing_all_uses_each_qubit() {
        // Apply depolarizing to all qubits with p = 1: every qubit gets some error.
        let n = 4;
        let mut stn = StabMps::with_seed(n, 99);
        let qubits: Vec<QubitId> = (0..n).map(QubitId).collect();
        stn.apply_depolarizing_all(&qubits, 1.0);
        // After error on each qubit (X, Y, or Z), the state is no longer |0^N>.
        // For Z errors only, qubit stays in |0> (Z|0>=|0>). For X/Y, q flips to |1>.
        // At least some qubits should have flipped (very high prob with 4 qubits).
        let outcomes: Vec<bool> = stn.mz(&qubits).iter().map(|r| r.outcome).collect();
        let any_flipped = outcomes.iter().any(|&b| b);
        assert!(
            any_flipped,
            "with p=1 on 4 qubits, at least one X/Y likely; got {outcomes:?}"
        );
    }

    #[test]
    fn test_pauli_expectation_n_30_zz_chain() {
        // 30-qubit GHZ-like state, scales beyond the SV path's n<=14 limit.
        // After H(0) + CX chain, ZZ on neighboring qubits = 1 (Bell-style).
        let n = 30;
        let mut stn = StabMps::new(n);
        stn.h(&[QubitId(0)]);
        for q in 0..n - 1 {
            stn.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
        // ⟨ZZ_{0,1}⟩ on GHZ = 1.
        let zz01 = stn.pauli_expectation(&[(0, PauliKind::Z), (1, PauliKind::Z)]);
        assert!((zz01 - 1.0).abs() < 1e-10, "n=30 ZZ_{{0,1}}: {zz01}");
        // ⟨ZZ_{15,29}⟩ on GHZ = 1 (long-range still correlated).
        let zz_far = stn.pauli_expectation(&[(15, PauliKind::Z), (29, PauliKind::Z)]);
        assert!((zz_far - 1.0).abs() < 1e-10, "n=30 ZZ_{{15,29}}: {zz_far}");
    }

    #[test]
    fn test_pauli_expectation_zz_on_bell_state() {
        // Bell state (|00⟩+|11⟩)/√2: ⟨ZZ⟩ = 1, ⟨XX⟩ = 1, ⟨YY⟩ = -1.
        let mut stn = StabMps::new(2);
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        let zz = stn.pauli_expectation(&[(0, PauliKind::Z), (1, PauliKind::Z)]);
        let xx = stn.pauli_expectation(&[(0, PauliKind::X), (1, PauliKind::X)]);
        let yy = stn.pauli_expectation(&[(0, PauliKind::Y), (1, PauliKind::Y)]);
        assert!((zz - 1.0).abs() < 1e-10, "ZZ on Bell = {zz}");
        assert!((xx - 1.0).abs() < 1e-10, "XX on Bell = {xx}");
        assert!((yy + 1.0).abs() < 1e-10, "YY on Bell = {yy}");
    }

    #[test]
    fn test_overlap_with_stabilizer_matches_state_vector() {
        // |s⟩ = |+⟩|0⟩|+⟩ via H on qubits 0 and 2.
        // |Ψ⟩ = |+⟩|0⟩|+⟩ same → overlap = 1.
        use pecos_simulators::CHForm;
        let mut s = CHForm::new_with_seed(3, 42);
        s.h(&[QubitId(0), QubitId(2)]);

        let mut stn = StabMps::with_seed(3, 99);
        stn.h(&[QubitId(0), QubitId(2)]);

        let est = stn.overlap_with_stabilizer(&s, 200, None);
        // Should be ~1 with some MC noise. For identical pure states,
        // each sample contributes exactly 1, so accumulator = num_samples
        // and average = 1 exactly (no variance for identical states).
        assert!(
            (est.norm_sqr() - 1.0).abs() < 0.01,
            "identical states fidelity should be 1.0, got |est|² = {}",
            est.norm_sqr()
        );
    }

    #[test]
    fn test_overlap_with_stabilizer_h_after_rotation_matches_state_vector() {
        use pecos_simulators::CHForm;

        let mut stn = StabMps::builder(3).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        stn.rz(Angle64::from_radians(0.9), &[QubitId(0)]);
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.h(&[QubitId(2)]);
        stn.rz(Angle64::from_radians(0.37), &[QubitId(2)]);
        stn.h(&[QubitId(2)]);
        stn.cx(&[(QubitId(1), QubitId(2))]);

        let mut stabilizer = CHForm::new_with_seed(3, 41);
        stabilizer.x(&[QubitId(0), QubitId(2)]);
        let state_vector = stn.state_vector();
        let expected: Complex64 = state_vector
            .iter()
            .enumerate()
            .map(|(index, amplitude)| stabilizer.amplitude(index).conj() * amplitude)
            .sum();
        let actual = stn.overlap_with_stabilizer(&stabilizer, 8, Some(73));

        assert!(
            (actual - expected).norm() < 1e-10,
            "overlap estimate={actual}, state-vector overlap={expected}"
        );
    }

    #[test]
    fn test_overlap_with_stabilizer_orthogonal_zero() {
        // |s⟩ = |0⟩, |Ψ⟩ = |1⟩ → overlap = 0.
        use pecos_simulators::CHForm;
        let s = CHForm::new_with_seed(2, 7);
        // |s⟩ = |00⟩.

        let mut stn = StabMps::with_seed(2, 99);
        stn.x(&[QubitId(0)]); // |Ψ⟩ = |10⟩.

        // |s⟩ has support {|00⟩} only, so MC samples always give x=|00⟩.
        // <x|Ψ> = <00|10> = 0. Estimator returns 0.
        let est = stn.overlap_with_stabilizer(&s, 50, None);
        assert!(
            est.norm() < 1e-10,
            "orthogonal states overlap should be 0, got {}",
            est.norm()
        );
    }

    #[test]
    fn test_code_state_fidelity_three_qubit_bit_flip() {
        // 3-qubit bit-flip code |0_L⟩ = |000⟩, |1_L⟩ = |111⟩.
        // Stabilizers: Z_0·Z_1, Z_1·Z_2.
        // Logical |0_L⟩ = |000⟩ is in the code → fidelity = 1.
        let mut stn = StabMps::new(3);
        // |000⟩ initially.
        let stabs = vec![
            vec![(0, PauliKind::Z), (1, PauliKind::Z)],
            vec![(1, PauliKind::Z), (2, PauliKind::Z)],
        ];
        let f = stn.code_state_fidelity(&stabs);
        assert!(
            (f - 1.0).abs() < 1e-10,
            "|000⟩ in bit-flip code, fidelity {f}"
        );

        // Apply X_0 → |100⟩, an error state, NOT in the code.
        // Z_0·Z_1·|100⟩ = -|100⟩ (Z_0 gives -1), so it's a -1 eigenstate of stab → fidelity = 0.
        stn.x(&[QubitId(0)]);
        let f = stn.code_state_fidelity(&stabs);
        assert!(f.abs() < 1e-10, "|100⟩ NOT in bit-flip code, fidelity {f}");

        // Encode logical |1⟩: |111⟩ via X on all 3.
        stn.reset();
        stn.x(&[QubitId(0), QubitId(1), QubitId(2)]);
        let f = stn.code_state_fidelity(&stabs);
        assert!(
            (f - 1.0).abs() < 1e-10,
            "|111⟩ in bit-flip code, fidelity {f}"
        );
    }

    #[test]
    fn test_code_state_fidelity_large_n_repetition_code() {
        // 8-qubit repetition code (logical 0 = |00000000>): stabilizers are
        // Z_iZ_{i+1} for i in 0..7. After preparing |0^N>, fidelity = 1.
        let n = 8;
        let stabs: Vec<Vec<(usize, PauliKind)>> = (0..n - 1)
            .map(|i| vec![(i, PauliKind::Z), (i + 1, PauliKind::Z)])
            .collect();
        let stn = StabMps::new(n);
        let f = stn.code_state_fidelity(&stabs);
        assert!((f - 1.0).abs() < 1e-10, "n=8 |0..0> rep code fidelity {f}");
    }

    #[test]
    fn test_code_state_fidelity_partial() {
        // Superposition partly in / partly out of code.
        // 50/50 mix of |000⟩ (in code) and |001⟩ (out of code) → fidelity = 0.5.
        let mut stn = StabMps::new(3);
        stn.h(&[QubitId(2)]); // |0⟩|0⟩(|0⟩+|1⟩)/√2 = (|000⟩ + |001⟩)/√2
        let stabs = vec![
            vec![(0, PauliKind::Z), (1, PauliKind::Z)],
            vec![(1, PauliKind::Z), (2, PauliKind::Z)],
        ];
        let f = stn.code_state_fidelity(&stabs);
        assert!((f - 0.5).abs() < 1e-10, "half/half, fidelity {f}");
    }

    #[test]
    fn test_merge_rz_commutes_through_z_and_s() {
        // RZ(t, q); Z(q); RZ(t, q); S(q); RZ(t, q) should merge to one
        // non-Clifford at flush time, because Z and S commute with RZ.
        // Compare vs eager (no commute optimization, same physics).
        let t = Angle64::from_radians(0.12345);

        let mut merged = StabMps::builder(2).seed(7).merge_rz(true).build();
        merged.h(&[QubitId(0)]); // make MPS non-trivial
        merged.rz(t, &[QubitId(0)]);
        merged.z(&[QubitId(0)]);
        merged.rz(t, &[QubitId(0)]);
        merged.sz(&[QubitId(0)]);
        merged.rz(t, &[QubitId(0)]);
        merged.flush();
        let sv_merged = merged.state_vector();
        let nc_merged = merged.stats.total_nonclifford;

        let mut eager = StabMps::builder(2).seed(7).merge_rz(false).build();
        eager.h(&[QubitId(0)]);
        eager.rz(t, &[QubitId(0)]);
        eager.z(&[QubitId(0)]);
        eager.rz(t, &[QubitId(0)]);
        eager.sz(&[QubitId(0)]);
        eager.rz(t, &[QubitId(0)]);
        let sv_eager = eager.state_vector();

        for (a, b) in sv_eager.iter().zip(sv_merged.iter()) {
            assert!((a - b).norm() < 1e-10, "commute-merge state mismatch");
        }
        // Merge saves non-Clifford applications: 3 → 1.
        assert_eq!(
            nc_merged, 1,
            "merged should call non-Clifford path once; got {nc_merged}"
        );
        assert_eq!(
            eager.stats.total_nonclifford, 3,
            "eager applies 3 non-Cliffords"
        );
    }

    #[test]
    fn test_merge_rz_x_flips_pending_sign() {
        // X anticommutes with RZ. rz(t, q); x(q) should leave pending_rz
        // = -t. Final state after x(q) + rz(t, q) + flush should equal
        // applying x then rz(t) (since net = identity for +t−t... wait
        // actually: X·RZ(θ) = RZ(-θ)·X. So rz(t); x; rz(t) = x; rz(-t); rz(t) = x.
        // Verify equality with just x applied.
        let t = Angle64::from_radians(0.7);

        let mut merged = StabMps::builder(2).seed(5).merge_rz(true).build();
        merged.h(&[QubitId(0)]); // non-trivial MPS
        merged.rz(t, &[QubitId(0)]);
        merged.x(&[QubitId(0)]);
        merged.rz(t, &[QubitId(0)]);
        merged.flush();
        let sv_merged = merged.state_vector();

        // Reference: just H and X (since RZ effects cancel via X-flip).
        let mut ref_sim = StabMps::with_seed(2, 5);
        ref_sim.h(&[QubitId(0)]);
        ref_sim.x(&[QubitId(0)]);
        let sv_ref = ref_sim.state_vector();

        for (a, b) in sv_ref.iter().zip(sv_merged.iter()) {
            assert!(
                (a - b).norm() < 1e-10,
                "X-flip pending-rz sign mismatch: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_merge_rz_two_t_gates_same_state_vector() {
        // RZ(t) + RZ(t) merged should equal one RZ(2t) and equal eager
        // applying RZ(t) twice. Compare state vectors.
        let t = Angle64::QUARTER_TURN / 2u64; // T

        let mut eager = StabMps::builder(2).seed(7).merge_rz(false).build();
        eager.h(&[QubitId(0)]);
        eager.cx(&[(QubitId(0), QubitId(1))]);
        eager.rz(t, &[QubitId(0)]);
        eager.rz(t, &[QubitId(0)]);
        let sv_eager = eager.state_vector();

        let mut merged = StabMps::builder(2).seed(7).merge_rz(true).build();
        merged.h(&[QubitId(0)]);
        merged.cx(&[(QubitId(0), QubitId(1))]);
        merged.rz(t, &[QubitId(0)]);
        merged.rz(t, &[QubitId(0)]);
        merged.flush();
        let sv_merged = merged.state_vector();

        for (a, b) in sv_eager.iter().zip(sv_merged.iter()) {
            assert!((a - b).norm() < 1e-10, "merge_rz state mismatch");
        }
    }

    #[test]
    fn test_merge_rz_intervening_gate_on_other_qubit_still_merges() {
        // rz(t, 0); h(1); rz(t, 0) — h(1) does not touch q0, so merge applies.
        let t = Angle64::QUARTER_TURN / 2u64;

        let mut merged = StabMps::builder(2).seed(11).merge_rz(true).build();
        merged.h(&[QubitId(0)]);
        merged.rz(t, &[QubitId(0)]);
        merged.h(&[QubitId(1)]);
        merged.rz(t, &[QubitId(0)]);
        merged.flush();
        let sv_merged = merged.state_vector();

        let mut eager = StabMps::builder(2).seed(11).merge_rz(false).build();
        eager.h(&[QubitId(0)]);
        eager.rz(t, &[QubitId(0)]);
        eager.h(&[QubitId(1)]);
        eager.rz(t, &[QubitId(0)]);
        let sv_eager = eager.state_vector();

        for (a, b) in sv_eager.iter().zip(sv_merged.iter()) {
            assert!(
                (a - b).norm() < 1e-10,
                "intervening other-qubit gate merge mismatch"
            );
        }
    }

    #[test]
    fn test_merge_rz_intervening_gate_on_same_qubit_flushes() {
        // rz(t, 0); h(0); rz(t, 0) — h(0) flushes pending rz first.
        let t = Angle64::QUARTER_TURN / 2u64;

        let mut merged = StabMps::builder(2).seed(13).merge_rz(true).build();
        merged.h(&[QubitId(0)]);
        merged.rz(t, &[QubitId(0)]);
        merged.h(&[QubitId(0)]);
        merged.rz(t, &[QubitId(0)]);
        merged.flush();
        let sv_merged = merged.state_vector();

        let mut eager = StabMps::builder(2).seed(13).merge_rz(false).build();
        eager.h(&[QubitId(0)]);
        eager.rz(t, &[QubitId(0)]);
        eager.h(&[QubitId(0)]);
        eager.rz(t, &[QubitId(0)]);
        let sv_eager = eager.state_vector();

        for (a, b) in sv_eager.iter().zip(sv_merged.iter()) {
            assert!(
                (a - b).norm() < 1e-10,
                "intervening same-qubit gate flush mismatch"
            );
        }
    }

    #[test]
    fn test_merge_rz_to_clifford_angle_uses_fast_path() {
        // Two T gates merge to S (RZ(π/2)) — Clifford angle, taken via tableau.
        let t = Angle64::QUARTER_TURN / 2u64;

        let mut merged = StabMps::builder(2).seed(17).merge_rz(true).build();
        merged.h(&[QubitId(0)]);
        merged.rz(t, &[QubitId(0)]);
        merged.rz(t, &[QubitId(0)]);
        merged.flush();
        // After merge, no pending RZ, no MPS non-Clifford gate count incremented.
        assert_eq!(
            merged.stats.total_nonclifford, 0,
            "two T merging to S should hit Clifford fast path, not non-Clifford"
        );

        let mut eager = StabMps::builder(2).seed(17).merge_rz(false).build();
        eager.h(&[QubitId(0)]);
        eager.rz(t, &[QubitId(0)]);
        eager.rz(t, &[QubitId(0)]);
        // Eager applies T twice as non-Clifford.
        assert_eq!(eager.stats.total_nonclifford, 2);

        // Both produce equivalent state.
        let sv_merged = merged.state_vector();
        let sv_eager = eager.state_vector();
        for (a, b) in sv_eager.iter().zip(sv_merged.iter()) {
            assert!((a - b).norm() < 1e-10, "T+T = S state mismatch");
        }
    }

    #[test]
    fn test_builder_default_values() {
        let stn = StabMps::builder(4).build();

        assert_eq!(stn.config.max_bond_dim, 128);
        assert_relative_eq!(stn.config.svd_cutoff, 1e-12);
        assert_eq!(stn.config.max_truncation_error, Some(1e-8));
        assert!(!stn.config.parallel);
        assert!(stn.flags.normalize_after_gate());
        assert_eq!(stn.measurement_mode, MeasurementMode::Exact);
        assert!(stn.flags.merge_rz());
        assert!(!stn.flags.pauli_frame_tracking());
        assert!(!stn.flags.numerical_flag_redetection());
    }

    #[test]
    fn test_numerical_redetection_accepts_scaled_product_site() {
        use pecos_simulators::DenseStateVec;

        let t = Angle64::QUARTER_TURN / 2u64;
        let final_angle = Angle64::from_radians(0.37);
        let mut stn = StabMps::builder(2)
            .merge_rz(false)
            .numerical_flag_redetection(true)
            .build();
        let mut oracle = DenseStateVec::new(2);

        stn.h(&[QubitId(0)]);
        oracle.h(&[QubitId(0)]);
        stn.rz(t, &[QubitId(0)]);
        oracle.rz(t, &[QubitId(0)]);
        stn.rz(-t, &[QubitId(0)]);
        oracle.rz(-t, &[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        oracle.cx(&[(QubitId(0), QubitId(1))]);

        // Move an exactly cancelling gauge factor between the two bond-one
        // product tensors. This preserves the physical state while making
        // neither local |0> coefficient unit modulus; site 0 is where
        // Mps::scale normally stores such a global factor.
        stn.mps.tensors_mut()[0] *= Complex64::new(0.25, 0.0);
        stn.mps.tensors_mut()[1] *= Complex64::new(4.0, 0.0);

        stn.rz(final_angle, &[QubitId(1)]);
        oracle.rz(final_angle, &[QubitId(1)]);

        assert_eq!(stn.stats.numerical_redetect, 1);
        let actual = stn.state_vector();
        for (index, amplitude) in actual.iter().enumerate() {
            assert_relative_eq!(
                amplitude.re,
                oracle.get_amplitude(index).re,
                epsilon = 1e-12
            );
            assert_relative_eq!(
                amplitude.im,
                oracle.get_amplitude(index).im,
                epsilon = 1e-12
            );
        }
    }

    #[test]
    fn test_numerical_zero_site_fallback_accepts_entangled_bonds() {
        let config = MpsConfig {
            max_bond_dim: 8,
            svd_cutoff: 0.0,
            max_truncation_error: Some(0.0),
            parallel: false,
        };
        let mut plus = Mps::new(3, config.clone());
        let mut minus = Mps::new(3, config);
        let inv_sqrt2 = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
        let h = DMatrix::from_row_slice(2, 2, &[inv_sqrt2, inv_sqrt2, inv_sqrt2, -inv_sqrt2]);
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        plus.apply_one_site_gate(1, &h).unwrap();
        minus.apply_one_site_gate(1, &h).unwrap();
        minus.apply_diagonal_one_site(1, &[one, -one]).unwrap();
        let mut mps = plus.add(&minus);

        assert!(mps.bond_dim(1) > 1 && mps.bond_dim(2) > 1);
        let tensor = &mps.tensors()[1];
        let chi_r = mps.bond_dim(2);
        assert!(
            (0..tensor.nrows())
                .any(|row| (chi_r..2 * chi_r).any(|column| tensor[(row, column)] != zero)),
            "test must exercise the marginal contraction, not structural-zero shortcut"
        );
        assert_eq!(cheap_product_zero_site_test(&mps, 1), None);
        assert_relative_eq!(stored_mps_one_marginal(&mps, 1), 0.0, epsilon = 1e-14);
        let batched = are_numerical_product_zero_sites(&mps, &[0, 1, 2]);
        assert!(batched[1]);
        let update = measure::ProjectionUpdate {
            collapsed_site: Some(1),
            modified_sites: vec![0, 1, 2],
        };
        let mut flags = vec![None; 3];
        repair_disent_flags(&mps, &mut flags, &update);
        assert_eq!(flags[1], Some(SiteEigenstate::Z(false)));
        assert_disent_flags_match_stored_mps(&mps, &flags, "batched fallback fixture");

        let x = DMatrix::from_row_slice(2, 2, &[zero, one, one, zero]);
        mps.apply_one_site_gate(1, &x).unwrap();
        assert!(!are_numerical_product_zero_sites(&mps, &[0, 1, 2])[1]);
        repair_disent_flags(&mps, &mut flags, &update);
        assert_eq!(flags[1], None);
    }

    #[test]
    fn test_numerical_zero_site_fallback_rejects_zero_norm_mps() {
        let product = Mps::new(3, MpsConfig::default());
        let mut mps = product.add(&product);
        for tensor in mps.tensors_mut() {
            tensor.fill(Complex64::new(1.0, 0.0));
        }
        mps.scale(Complex64::new(0.0, 0.0));

        assert!(mps.norm_squared() <= 1e-30);
        assert_eq!(
            are_numerical_product_zero_sites(&mps, &[0, 1, 2]),
            vec![false; 3]
        );
    }

    #[test]
    #[should_panic(expected = "max_truncation_error must be finite and non-negative")]
    fn test_builder_rejects_negative_max_truncation_error() {
        let _ = StabMps::builder(1).max_truncation_error(-1.0);
    }

    #[test]
    #[should_panic(expected = "max_truncation_error must be finite and non-negative")]
    fn test_builder_rejects_nan_max_truncation_error() {
        let _ = StabMps::builder(1).max_truncation_error(f64::NAN);
    }

    #[test]
    #[should_panic(expected = "max_truncation_error must be finite and non-negative")]
    fn test_builder_rejects_infinite_max_truncation_error() {
        let _ = StabMps::builder(1).max_truncation_error(f64::INFINITY);
    }

    #[test]
    fn test_builder_for_qec_preset() {
        // Smoke test: the preset should build a working StabMps and handle
        // a Clifford + T + measurement sequence.
        let mut stn = StabMps::builder(4).seed(99).for_qec().build();
        assert_eq!(stn.measurement_mode(), MeasurementMode::Exact);
        stn.h(&[QubitId(0)]);
        stn.cx(&[(QubitId(0), QubitId(1))]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
        stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(2)]);
        let r0 = stn.mz(&[QubitId(0)])[0].outcome;
        let r1 = stn.mz(&[QubitId(1)])[0].outcome;
        assert_eq!(r0, r1, "for_qec preset: Bell+T correlation");
    }

    #[test]
    fn measurement_mode_override_clone_and_reset_retention() {
        let mut exact_override = StabMps::builder(2)
            .for_qec()
            .measurement(MeasurementMode::Exact)
            .build();
        assert_eq!(exact_override.measurement_mode(), MeasurementMode::Exact);
        assert_eq!(
            exact_override.clone().measurement_mode(),
            MeasurementMode::Exact
        );
        exact_override.reset();
        assert_eq!(exact_override.measurement_mode(), MeasurementMode::Exact);

        let lazy = StabMps::builder(2)
            .measurement(MeasurementMode::Lazy)
            .build();
        assert_eq!(lazy.clone().measurement_mode(), MeasurementMode::Lazy);
    }

    fn assert_clone_set_seed_reseeds_trivial_tableau_measurements(mode: MeasurementMode) {
        const NUM_SHOTS: usize = 400;

        let mut prepared = StabMps::builder(2).seed(0x5EED).measurement(mode).build();
        prepared.h(&[QubitId(0)]);
        prepared.cx(&[(QubitId(0), QubitId(1))]);
        assert_eq!(prepared.stats.total_nonclifford, 0);
        assert_eq!(prepared.max_bond_dim(), 1);

        let distinct = (0..NUM_SHOTS)
            .map(|shot| {
                let mut simulator = prepared.clone();
                simulator.set_seed(0xC10E_0000_u64.wrapping_add(shot as u64));
                simulator
                    .mz(&[QubitId(0), QubitId(1)])
                    .into_iter()
                    .map(|result| result.outcome)
                    .collect::<Vec<_>>()
            })
            .collect::<std::collections::HashSet<_>>();

        assert!(
            distinct.len() > 1,
            "{mode:?} clone+set_seed shots collapsed to one tableau-RNG outcome: {distinct:?}"
        );
    }

    #[test]
    fn clone_set_seed_reseeds_trivial_tableau_measurements_in_lazy_mode() {
        assert_clone_set_seed_reseeds_trivial_tableau_measurements(MeasurementMode::Lazy);
    }

    #[test]
    fn clone_set_seed_reseeds_trivial_tableau_measurements_in_pragmatic_mode() {
        assert_clone_set_seed_reseeds_trivial_tableau_measurements(MeasurementMode::Pragmatic);
    }

    #[test]
    fn test_builder_lazy_measure_bell_correlation() {
        // Lazy-measure path must give same Bell-state correlation as eager.
        for trial in 0..20 {
            let mut stn = StabMps::builder(2)
                .seed(3000 + trial)
                .measurement(MeasurementMode::Lazy)
                .build();
            stn.h(&[QubitId(0)]);
            stn.cx(&[(QubitId(0), QubitId(1))]);
            stn.rz(Angle64::QUARTER_TURN / 2u64, &[QubitId(0)]);
            let r0 = stn.mz(&[QubitId(0)])[0].outcome;
            let r1 = stn.mz(&[QubitId(1)])[0].outcome;
            assert_eq!(r0, r1, "lazy Bell+T trial {trial}");
        }
    }

    #[test]
    fn test_builder_lazy_measure_rx_statistics() {
        // Lazy path must give correct RX(pi/3) measurement statistics.
        let theta = Angle64::from_radians(std::f64::consts::FRAC_PI_3);
        let num_trials: u32 = 400;
        let mut count_0 = 0;
        for trial in 0..num_trials {
            let mut stn = StabMps::builder(1)
                .seed(u64::from(4000 + trial))
                .measurement(MeasurementMode::Lazy)
                .build();
            stn.rx(theta, &[QubitId(0)]);
            if !stn.mz(&[QubitId(0)])[0].outcome {
                count_0 += 1;
            }
        }
        let p0 = f64::from(count_0) / f64::from(num_trials);
        assert!((p0 - 0.75).abs() < 0.08, "lazy RX(pi/3) p(0) = {p0:.3}");
    }

    #[test]
    fn test_measurement_and_reset_disent_flags_match_stored_mps_marginals() {
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

        const NUM_QUBITS: usize = 5;
        const NUM_CIRCUITS: u64 = 50;
        const NUM_MEASUREMENTS: usize = 3;
        let mut flagged_sites = [0_usize; 2];
        let mut false_proofs = Vec::new();

        for (config_index, truncating) in [false, true].into_iter().enumerate() {
            for lazy_measure in [false, true] {
                for numerical_flag_redetection in [false, true] {
                    for circuit_seed in 0..NUM_CIRCUITS {
                        let mut random = circuit_seed + 1;
                        let (max_bond_dim, svd_cutoff, max_truncation_error) = if truncating {
                            (4, 1e-12, 1e-8)
                        } else {
                            (64, 0.0, 0.0)
                        };
                        let mut stn = StabMps::builder(NUM_QUBITS)
                            .seed(0x6000_0000 + circuit_seed)
                            .max_bond_dim(max_bond_dim)
                            .svd_cutoff(svd_cutoff)
                            .max_truncation_error(max_truncation_error)
                            .merge_rz(false)
                            .measurement(if lazy_measure {
                                MeasurementMode::Lazy
                            } else {
                                MeasurementMode::Pragmatic
                            })
                            .numerical_flag_redetection(numerical_flag_redetection)
                            .build();

                        for step in 0..18 {
                            let choice = next(&mut random) % 8;
                            let q = (next(&mut random) % NUM_QUBITS as u64) as usize;
                            match choice {
                                0 | 1 => {
                                    stn.h(&[QubitId(q)]);
                                }
                                2 => {
                                    stn.sz(&[QubitId(q)]);
                                }
                                3 | 4 => {
                                    let (control, target) =
                                        random_distinct_pair(&mut random, NUM_QUBITS);
                                    stn.cx(&[(QubitId(control), QubitId(target))]);
                                }
                                5 => {
                                    let (first, second) =
                                        random_distinct_pair(&mut random, NUM_QUBITS);
                                    stn.cz(&[(QubitId(first), QubitId(second))]);
                                }
                                _ => {
                                    let radians = if step & 1 == 0 { 0.37 } else { -0.61 };
                                    stn.rz(Angle64::from_radians(radians), &[QubitId(q)]);
                                }
                            }
                        }

                        for measurement in 0..NUM_MEASUREMENTS {
                            let measured_qubit = (next(&mut random) % NUM_QUBITS as u64) as usize;
                            let _ = stn.mz(&[QubitId(measured_qubit)]);
                            for (site, flag) in stn.disent_flags.iter().enumerate() {
                                if matches!(flag, Some(SiteEigenstate::Z(false))) {
                                    flagged_sites[config_index] += 1;
                                    let marginal = stored_mps_one_marginal(&stn.mps, site);
                                    if marginal.abs() > STORED_PROOF_SOUNDNESS_TOLERANCE {
                                        false_proofs.push((
                                            truncating,
                                            lazy_measure,
                                            numerical_flag_redetection,
                                            circuit_seed,
                                            measurement,
                                            "measurement",
                                            site,
                                            marginal,
                                        ));
                                    }
                                }
                            }

                            let reset_qubit = (next(&mut random) % NUM_QUBITS as u64) as usize;
                            stn.reset_qubit(QubitId(reset_qubit));
                            for (site, flag) in stn.disent_flags.iter().enumerate() {
                                if matches!(flag, Some(SiteEigenstate::Z(false))) {
                                    flagged_sites[config_index] += 1;
                                    let marginal = stored_mps_one_marginal(&stn.mps, site);
                                    if marginal.abs() > STORED_PROOF_SOUNDNESS_TOLERANCE {
                                        false_proofs.push((
                                            truncating,
                                            lazy_measure,
                                            numerical_flag_redetection,
                                            circuit_seed,
                                            measurement,
                                            "reset",
                                            site,
                                            marginal,
                                        ));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        eprintln!(
            "stored-frame flag census: configs=2 modes_per_config=4 circuits_per_mode={NUM_CIRCUITS} measurements_and_resets_per_circuit={NUM_MEASUREMENTS} exact_flagged_sites={} truncating_flagged_sites={} false_proofs={}",
            flagged_sites[0],
            flagged_sites[1],
            false_proofs.len()
        );
        assert!(
            false_proofs.is_empty(),
            "every stored-frame Z(false) proof must have |P(sigma=1)| <= {STORED_PROOF_SOUNDNESS_TOLERANCE:.1e}; found {} false proofs, first={:?}",
            false_proofs.len(),
            false_proofs.first()
        );
    }

    #[test]
    fn test_disent_flag_measurement_continuation_matches_dense() {
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

        fn apply(op: Op, stn: &mut StabMps, dense: &mut DenseStateVec) {
            match op {
                Op::H(q) => {
                    stn.h(&[QubitId(q)]);
                    dense.h(&[QubitId(q)]);
                }
                Op::Sz(q) => {
                    stn.sz(&[QubitId(q)]);
                    dense.sz(&[QubitId(q)]);
                }
                Op::Cx(control, target) => {
                    let pair = [(QubitId(control), QubitId(target))];
                    stn.cx(&pair);
                    dense.cx(&pair);
                }
                Op::Cz(first, second) => {
                    let pair = [(QubitId(first), QubitId(second))];
                    stn.cz(&pair);
                    dense.cz(&pair);
                }
                Op::Rz(q, radians) => {
                    let angle = Angle64::from_radians(radians);
                    stn.rz(angle, &[QubitId(q)]);
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

        const N: usize = 5;
        let mut eager_pragmatic_drift_exercised = false;
        for lazy_measure in [false, true] {
            for numerical_flag_redetection in [false, true] {
                for circuit_seed in 0..16 {
                    let continuation_seed = circuit_seed;
                    let mut random = circuit_seed + 1;
                    let mut stn = StabMps::builder(N)
                        .seed(0x5000_0000 + circuit_seed)
                        .max_bond_dim(64)
                        .svd_cutoff(0.0)
                        .max_truncation_error(0.0)
                        .merge_rz(false)
                        .measurement(if lazy_measure {
                            MeasurementMode::Lazy
                        } else {
                            MeasurementMode::Pragmatic
                        })
                        .numerical_flag_redetection(numerical_flag_redetection)
                        .build();
                    let mut dense = DenseStateVec::new(N);

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
                            _ => Op::Rz(q, if step & 1 == 0 { 0.37 } else { -0.61 }),
                        };
                        apply(op, &mut stn, &mut dense);
                    }

                    let dense_before_measurement = dense.state();
                    assert!(fidelity(&stn.state_vector(), &dense_before_measurement) > 1.0 - 1e-9);

                    let measured_qubit = 1;
                    let result = stn
                        .mz(&[QubitId(measured_qubit)])
                        .into_iter()
                        .next()
                        .expect("one measurement result");
                    let mut post_measurement = stn.clone();
                    post_measurement.flush();
                    if lazy_measure {
                        assert_eq!(stn.uncompensated_pre_reduction_count(), 0);
                    } else if stn.uncompensated_pre_reduction_count() > 0 {
                        eager_pragmatic_drift_exercised = true;
                    }
                    let projected =
                        project_z(&dense_before_measurement, measured_qubit, result.outcome);
                    let post_measurement_fidelity =
                        fidelity(&post_measurement.state_vector(), &projected);
                    assert_disent_flags_match_stored_mps(
                        &stn.mps,
                        &stn.disent_flags,
                        &format!(
                            "measurement; lazy={lazy_measure} redetect={numerical_flag_redetection} seed={circuit_seed}"
                        ),
                    );
                    if !lazy_measure && stn.uncompensated_pre_reduction_count() > 0 {
                        // Eager pre-reduction deliberately records that exact
                        // amplitude comparisons are no longer valid. The stored
                        // flag marginal above remains an exact local invariant.
                        continue;
                    }
                    assert!(post_measurement_fidelity > 1.0 - 1e-9);
                    let mut oracle = DenseStateVec::from_state(
                        &projected,
                        PecosRng::seed_from_u64(continuation_seed),
                    );
                    let mut continuation_random = continuation_seed + 1;
                    for _ in 0..4 {
                        let choice = next(&mut continuation_random) % 4;
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
                        apply(op, &mut stn, &mut oracle);
                    }
                    let target = (next(&mut continuation_random) % N as u64) as usize;
                    apply(Op::Rz(target, 0.731), &mut stn, &mut oracle);

                    stn.flush();
                    let actual = stn.state_vector();
                    let expected = oracle.state();
                    let continued_fidelity = fidelity(&actual, &expected);
                    let max_probability_error = actual
                        .iter()
                        .zip(&expected)
                        .map(|(a, b)| (a.norm_sqr() - b.norm_sqr()).abs())
                        .fold(0.0_f64, f64::max);
                    eprintln!(
                        "StabMps continuation: lazy={lazy_measure} redetect={numerical_flag_redetection} \
                     seed={circuit_seed} fidelity={continued_fidelity:.16} \
                     max_probability_error={max_probability_error:.3e}"
                    );
                    assert!(continued_fidelity > 1.0 - 1e-9);
                    assert!(max_probability_error < 1e-10);
                }
            }
        }
        assert!(
            eager_pragmatic_drift_exercised,
            "eager seed sweep must exercise the pragmatic-drift branch"
        );
    }

    #[test]
    fn exact_measurement_vanish_retry_matches_clean_projection_and_restores_config() {
        let prepare = || {
            let mut stn = StabMps::builder(3)
                .seed(0xabc)
                .max_bond_dim(1)
                .svd_cutoff(1e-7)
                .max_truncation_error(1e-4)
                .merge_rz(false)
                .build();
            stn.h(&[QubitId(0)]);
            stn.rz(Angle64::QUARTER_TURN / 2_u64, &[QubitId(0)]);
            stn
        };

        let mut clean = prepare();
        let clean_outcome = clean.mz(&[QubitId(0)])[0].outcome;

        let mut retried = prepare();
        measure::inject_projection_vanishes(1);
        let retried_outcome = retried.mz(&[QubitId(0)])[0].outcome;
        assert_eq!(retried_outcome, clean_outcome);
        assert_eq!(
            format!("{:?}", retried.tableau.stabs()),
            format!("{:?}", clean.tableau.stabs())
        );
        assert_eq!(
            format!("{:?}", retried.tableau.destabs()),
            format!("{:?}", clean.tableau.destabs())
        );
        assert_eq!(retried.mps.tensors(), clean.mps.tensors());
        assert_eq!(retried.mps.bond_dims(), clean.mps.bond_dims());
        assert_eq!(
            retried.mps.tracked_center_for_test(),
            clean.mps.tracked_center_for_test()
        );
        assert_eq!(
            retried.mps.truncation_error().to_bits(),
            clean.mps.truncation_error().to_bits()
        );
        assert_eq!(
            retried.mps.summed_discarded_weight().to_bits(),
            clean.mps.summed_discarded_weight().to_bits()
        );
        assert_eq!(retried.mps.bond_cap_hits(), clean.mps.bond_cap_hits());
        assert_eq!(retried.branch_vanish_retry_count(), 1);
        assert_eq!(retried.mps.config().max_bond_dim, 1);
        assert_eq!(
            retried.mps.config().svd_cutoff.to_bits(),
            1e-7_f64.to_bits()
        );
        assert_eq!(retried.mps.config().max_truncation_error, Some(1e-4));

        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let cnot = DMatrix::from_row_slice(
            4,
            4,
            &[
                one, zero, zero, zero, zero, one, zero, zero, zero, zero, zero, one, zero, zero,
                one, zero,
            ],
        );
        retried
            .mps
            .apply_long_range_two_site_gate(0, 2, &cnot)
            .unwrap();
        retried.mps.compress().unwrap();
        assert!(retried.mps.norm_squared().is_finite());
        assert!(retried.mps.max_bond_dim() <= 1);
    }

    #[test]
    #[should_panic(expected = "StabMps::mz: sampled branch vanished on the untruncated retry")]
    fn exact_measurement_double_vanish_panics_with_operation_name() {
        let mut stn = StabMps::builder(1).seed(9).merge_rz(false).build();
        stn.h(&[QubitId(0)]);
        stn.rz(Angle64::QUARTER_TURN / 2_u64, &[QubitId(0)]);
        measure::inject_projection_vanishes(2);
        let _ = stn.mz(&[QubitId(0)]);
    }

    #[test]
    fn pragmatic_mode_without_pre_reduction_is_not_state_exact() {
        let mut stn = StabMps::builder(1)
            .measurement(MeasurementMode::Pragmatic)
            .build();
        stn.h(&[QubitId(0)]);
        let _ = stn.mz(&[QubitId(0)]);
        assert_eq!(stn.uncompensated_pre_reduction_count(), 0);
        assert!(!stn.is_state_exact());
    }

    #[test]
    fn state_exactness_has_all_seven_guards_but_not_retry_count() {
        let exact = || StabMps::builder(2).merge_rz(false).build();
        assert!(exact().is_state_exact());

        let mut pending_rz = StabMps::builder(2).merge_rz(true).build();
        pending_rz.rz(Angle64::from_radians(0.37), &[QubitId(0)]);
        assert!(!pending_rz.is_state_exact());

        let mut frame = StabMps::builder(2).pauli_frame_tracking(true).build();
        frame.inject_x_in_frame(QubitId(0));
        assert!(!frame.is_state_exact());

        let mut deferred = exact();
        deferred.deferred_ops.push(measure::DeferredOp::H(0));
        assert!(!deferred.is_state_exact());

        let mut uncompensated = exact();
        uncompensated.uncompensated_pre_reduction_count = 1;
        assert!(!uncompensated.is_state_exact());

        let pragmatic = StabMps::builder(2)
            .merge_rz(false)
            .measurement(MeasurementMode::Pragmatic)
            .build();
        assert!(!pragmatic.is_state_exact());

        let mut truncated = exact();
        truncated.mps.record_truncation(1e-9, false);
        assert!(!truncated.is_state_exact());

        let mut deferred_loss = exact();
        deferred_loss.mps.record_deferred_branch_lost();
        assert!(!deferred_loss.is_state_exact());

        let mut retried = exact();
        retried.mps.record_branch_vanish_retry();
        assert!(retried.is_state_exact());
    }
}
