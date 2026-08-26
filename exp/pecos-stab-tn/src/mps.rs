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

//! Matrix Product State (MPS) engine.
//!
//! An MPS represents a quantum state as a chain of tensors:
//!
//! ```text
//! |psi> = sum_{s_0, ..., s_{N-1}} A[0]^{s_0} A[1]^{s_1} ... A[N-1]^{s_{N-1}} |s_0 s_1 ... s_{N-1}>
//! ```
//!
//! Each site tensor `A[i]^{s_i}` is a matrix of shape `(chi_left, chi_right)`.
//! For all physical indices `s_i` together, site `i` is stored as a single
//! `DMatrix<Complex64>` of shape `(chi_left, d * chi_right)`, where columns
//! `[s * chi_right .. (s+1) * chi_right]` correspond to physical index `s`.

pub mod canon;
pub mod svd;
pub mod tensor;

use crate::errors::MpsError;
use nalgebra::DMatrix;
use num_complex::Complex64;
use rayon::prelude::*;
use tensor::{
    contract_two_sites, phys_block, reshape_left_ungroup, reshape_two_site_for_svd, set_phys_block,
};

// Analytic gates are unitary to machine precision. Keep this materially below
// the debug validator's 1e-9 budget so repeated preserved mutations cannot
// consume that entire budget one boundary-passing gate at a time.
const ISOMETRY_PRESERVING_UNITARY_TOLERANCE: f64 = 1e-12;

/// Configuration for MPS truncation.
#[derive(Clone, Debug)]
pub struct MpsConfig {
    /// Maximum bond dimension (hard cap). Singular values beyond this are discarded.
    pub max_bond_dim: usize,
    /// Minimum normalized singular value to keep.
    ///
    /// Each MPS split multiplies this dimensionless threshold by the
    /// Frobenius norm of the matrix being decomposed. During canonical
    /// compression that norm is the state norm, so truncation is invariant
    /// under global rescaling of the state.
    pub svd_cutoff: f64,
    /// Maximum relative truncation error per SVD.
    /// When set, singular values are kept until the discarded weight
    /// (sum of discarded `s_i^2` / sum of all `s_i^2`) exceeds this threshold.
    /// This allows low-entanglement bonds to use small chi (fast) while
    /// high-entanglement bonds grow up to `max_bond_dim` (accurate).
    /// `None` or a value of zero disables adaptive truncation; only
    /// `svd_cutoff` and `max_bond_dim` can then discard singular values.
    pub max_truncation_error: Option<f64>,
    /// Use rayon for parallelizing independent MPS operations.
    pub parallel: bool,
}

impl Default for MpsConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 128,
            svd_cutoff: 1e-12,
            max_truncation_error: Some(1e-8),
            parallel: false,
        }
    }
}

/// Matrix Product State with open boundary conditions.
///
/// Physical dimension is `d` (2 for qubits). Site tensor `i` has shape
/// `(bond_dims[i], d * bond_dims[i+1])`.
pub struct Mps {
    num_sites: usize,
    phys_dim: usize,
    tensors: Vec<DMatrix<Complex64>>,
    /// Claimed mixed-canonical orthogonality center. When this is `Some(k)`,
    /// every site left of `k` is a left-isometry and every site right of `k`
    /// is a right-isometry. `None` makes no canonical-form claim.
    center: Option<usize>,
    /// Bond dimensions: length `num_sites + 1`.
    /// `bond_dims[0] = 1` (left boundary), `bond_dims[num_sites] = 1` (right boundary).
    bond_dims: Vec<usize>,
    config: MpsConfig,
    /// Accumulated truncation error: `1 - ∏(1 - step_discarded_weight)`.
    /// Approximates total 1-fidelity loss from SVD truncations over the lifetime
    /// of this MPS. Each truncated SVD updates this via
    /// `err = err + (1 - err) * step_discarded_weight`.
    truncation_error: f64,
    /// Number of SVDs that were capped by `max_bond_dim` (rank-limited rather
    /// than cutoff-limited). If > 0 the caller may want to raise `max_bond_dim`.
    bond_cap_hits: u64,
    /// Phase-local count of SVD operations. Profiling code clears this before
    /// each measured phase, so clone/add history cannot masquerade as work.
    phase_svd_operations: u64,
    /// Phase-local subset of `phase_svd_operations` at which the bond cap bound.
    phase_capped_svd_operations: u64,
    /// True sum of every relative singular-value weight discarded by an SVD.
    summed_discarded_weight: f64,
    /// Largest bond dimension held by this MPS during its lifetime.
    lifetime_peak_bond: usize,
    /// Number of rolled-back sampled projections retried without truncation.
    branch_vanish_retry_count: u64,
    /// Number of deferred MAST branches replaced by their surviving complement.
    deferred_branch_lost_count: u64,
    /// Number of cold full-chain canonicalization routes taken because no
    /// orthogonality center was available at the canonicalization consult.
    full_canonical_sweep_count: u64,
    /// Number of canonicalization routes that reused a tracked orthogonality
    /// center instead of starting from a cold full-chain sweep.
    center_reuse_count: u64,
}

/// Pass-scoped left and right identity environments for selected MPS sites.
///
/// The cache stores the contractions bordering each candidate, so one-site
/// expectations can be evaluated without recontracting the rest of the chain
/// for each site. It borrows the MPS immutably: callers cannot mutate the
/// tensors while these environments are live, and dropping the cache
/// invalidates the pass.
pub struct MpsEnvironmentCache<'a> {
    mps: &'a Mps,
    sites: Vec<usize>,
    left: Vec<DMatrix<Complex64>>,
    right: Vec<DMatrix<Complex64>>,
    norm_squared: f64,
}

impl MpsEnvironmentCache<'_> {
    /// Return the cached squared norm `<psi|psi>`.
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        self.norm_squared
    }

    /// Evaluate a normalized one-site computational-basis marginal.
    ///
    /// This contracts `|physical_index><physical_index|` between the cached
    /// identity environments. No canonical form is required: both
    /// environments retain the complete gauge-independent contraction.
    ///
    /// # Panics
    ///
    /// Panics if the site was not included when the cache was built or if the
    /// physical index is out of range, or if the cached MPS has numerically
    /// zero norm (for which a normalized marginal is undefined).
    #[must_use]
    pub fn one_site_basis_marginal(&self, site: usize, physical_index: usize) -> f64 {
        let candidate = self
            .sites
            .binary_search(&site)
            .expect("MPS site was not cached");
        assert!(
            physical_index < self.mps.phys_dim,
            "MPS physical index out of range"
        );
        assert!(
            self.norm_squared > 1e-20,
            "cannot evaluate a normalized marginal of a zero-norm MPS"
        );

        let chi_r = self.mps.bond_dims[site + 1];
        let block = phys_block(&self.mps.tensors[site], physical_index, chi_r);
        let local_transfer = block.conjugate().transpose() * &self.left[candidate] * block;
        let weight: Complex64 = local_transfer
            .component_mul(&self.right[candidate])
            .iter()
            .sum();
        weight.re / self.norm_squared
    }
}

fn advance_left_identity_environment(
    environment: &DMatrix<Complex64>,
    tensor: &DMatrix<Complex64>,
    phys_dim: usize,
    chi_r: usize,
) -> DMatrix<Complex64> {
    let mut next = DMatrix::zeros(chi_r, chi_r);
    for sigma in 0..phys_dim {
        let block = phys_block(tensor, sigma, chi_r);
        next += block.conjugate().transpose() * environment * block;
    }
    next
}

fn advance_right_identity_environment(
    environment: &DMatrix<Complex64>,
    tensor: &DMatrix<Complex64>,
    phys_dim: usize,
    chi_r: usize,
) -> DMatrix<Complex64> {
    let mut previous = DMatrix::zeros(tensor.nrows(), tensor.nrows());
    for sigma in 0..phys_dim {
        let block = phys_block(tensor, sigma, chi_r);
        previous += block.conjugate() * environment * block.transpose();
    }
    previous
}

impl Mps {
    /// Create an MPS initialized to |00...0> with bond dimension 1 everywhere.
    #[must_use]
    pub fn new(num_sites: usize, config: MpsConfig) -> Self {
        let d = 2;
        let bond_dims = vec![1; num_sites + 1];
        let mut tensors = Vec::with_capacity(num_sites);
        for _ in 0..num_sites {
            // Each tensor is (1, d*1) = (1, 2), representing [1, 0] (amplitude 1 for |0>)
            let mut t = DMatrix::zeros(1, d);
            t[(0, 0)] = Complex64::new(1.0, 0.0);
            tensors.push(t);
        }
        Self {
            num_sites,
            phys_dim: d,
            tensors,
            center: (num_sites > 0).then_some(0),
            bond_dims,
            config,
            truncation_error: 0.0,
            bond_cap_hits: 0,
            phase_svd_operations: 0,
            phase_capped_svd_operations: 0,
            summed_discarded_weight: 0.0,
            lifetime_peak_bond: 1,
            branch_vanish_retry_count: 0,
            deferred_branch_lost_count: 0,
            full_canonical_sweep_count: 0,
            center_reuse_count: 0,
        }
    }

    /// Accumulated truncation error: `1 - ∏(1 - step_discarded_weight)`.
    /// Zero for exact simulations; bounded above by the sum of per-step
    /// discarded weights. Approximates `1 - |⟨ψ_true|ψ_truncated⟩|²`.
    #[must_use]
    pub fn truncation_error(&self) -> f64 {
        self.truncation_error
    }

    /// Count of SVDs where the `max_bond_dim` cap was binding. If > 0 the
    /// state is under-resolved and the user may want to increase the cap.
    #[must_use]
    pub fn bond_cap_hits(&self) -> u64 {
        self.bond_cap_hits
    }

    /// Sum of the relative discarded weights reported by every SVD.
    ///
    /// Unlike [`Self::truncation_error`], this is a true sum rather than the
    /// product-form estimate `1 - product(1 - weight)`.
    #[must_use]
    pub fn summed_discarded_weight(&self) -> f64 {
        self.summed_discarded_weight
    }

    /// Largest bond dimension held by this MPS since construction or reset.
    #[must_use]
    pub fn lifetime_peak_bond(&self) -> usize {
        self.lifetime_peak_bond
    }

    /// Number of sampled branches whose first, rolled-back projection vanished.
    #[must_use]
    pub fn branch_vanish_retry_count(&self) -> u64 {
        self.branch_vanish_retry_count
    }

    /// Number of lost deferred MAST branches continued on their complement.
    #[must_use]
    pub fn deferred_branch_lost_count(&self) -> u64 {
        self.deferred_branch_lost_count
    }

    /// Number of canonicalization consults that required a cold full-chain sweep.
    ///
    /// Counts only the shared `canonicalize_at` route; the sweeps inside
    /// `compress` and `left_canonicalize` are not routed through it, so this
    /// undercounts total sweep work. The reuse counter likewise counts route
    /// selection, not factorizations saved (a warm walk across the whole
    /// chain does as many local factorizations as a cold sweep).
    #[must_use]
    pub fn full_canonical_sweep_count(&self) -> u64 {
        self.full_canonical_sweep_count
    }

    /// Number of canonicalization consults that reused a tracked center.
    #[must_use]
    pub fn center_reuse_count(&self) -> u64 {
        self.center_reuse_count
    }

    /// Reset truncation and canonical-routing diagnostics (keep state).
    pub fn reset_truncation_stats(&mut self) {
        self.truncation_error = 0.0;
        self.bond_cap_hits = 0;
        self.phase_svd_operations = 0;
        self.phase_capped_svd_operations = 0;
        self.summed_discarded_weight = 0.0;
        self.branch_vanish_retry_count = 0;
        self.deferred_branch_lost_count = 0;
        self.full_canonical_sweep_count = 0;
        self.center_reuse_count = 0;
        self.lifetime_peak_bond = self.max_bond_dim();
    }

    /// Record the outcome of one truncated SVD for telemetry.
    pub(crate) fn record_truncation(&mut self, discarded_weight: f64, hit_cap: bool) {
        self.phase_svd_operations += 1;
        if discarded_weight > 0.0 {
            self.truncation_error += (1.0 - self.truncation_error) * discarded_weight;
            self.summed_discarded_weight += discarded_weight;
        }
        if hit_cap {
            self.bond_cap_hits += 1;
            self.phase_capped_svd_operations += 1;
        }
    }

    /// Clear the operation counters used by query-phase profiling.
    pub(crate) fn reset_phase_svd_operations(&mut self) {
        self.phase_svd_operations = 0;
        self.phase_capped_svd_operations = 0;
    }

    /// Take and clear the operation counters used by query-phase profiling.
    pub(crate) fn take_phase_svd_operations(&mut self) -> (u64, u64) {
        let result = (self.phase_svd_operations, self.phase_capped_svd_operations);
        self.reset_phase_svd_operations();
        result
    }

    pub(crate) fn record_branch_vanish_retry(&mut self) {
        self.branch_vanish_retry_count += 1;
    }

    pub(crate) fn record_deferred_branch_lost(&mut self) {
        self.deferred_branch_lost_count += 1;
    }

    fn record_current_peak_bond(&mut self) {
        self.lifetime_peak_bond = self.lifetime_peak_bond.max(self.max_bond_dim());
    }

    #[must_use]
    /// Return the number of physical sites in the MPS chain.
    pub fn num_sites(&self) -> usize {
        self.num_sites
    }

    #[must_use]
    /// Return the local physical dimension (two for qubit MPS instances).
    pub fn phys_dim(&self) -> usize {
        self.phys_dim
    }

    /// Bond dimension at bond `i` (between sites `i-1` and `i`).
    #[must_use]
    pub fn bond_dim(&self, bond: usize) -> usize {
        self.bond_dims[bond]
    }

    #[must_use]
    /// Return the largest bond dimension currently present in the chain.
    pub fn max_bond_dim(&self) -> usize {
        *self.bond_dims.iter().max().unwrap_or(&1)
    }

    #[must_use]
    /// Return the truncation and parallelism configuration used by this MPS.
    pub fn config(&self) -> &MpsConfig {
        &self.config
    }

    /// Update the max bond dimension cap. Used by adaptive bond-dim
    /// auto-grow logic (e.g., `StabMps::auto_grow_bond_dim_if_needed`).
    /// Does not retroactively change existing tensors; takes effect on
    /// subsequent SVD truncations.
    pub fn set_max_bond_dim(&mut self, new_cap: usize) {
        self.config.max_bond_dim = new_cap;
    }

    /// Replace the truncation configuration while preserving state telemetry.
    pub(crate) fn set_config(&mut self, config: MpsConfig) {
        self.config = config;
    }

    /// Exact Schmidt-rank ceiling over the physical bonds this operation may touch.
    ///
    /// Forced projection can route compensating long-range gates across the
    /// whole chain, so every internal bond is affected in the general case.
    /// The result is clamped below the arithmetic limit used by randomized-SVD
    /// eligibility checks. This includes representable shifts near the top of
    /// `usize`, not only chains whose `2^(n/2)` shift itself overflows.
    #[must_use]
    pub(crate) fn physical_rank_ceiling(&self) -> usize {
        let exponent = self.num_sites / 2;
        1usize
            .checked_shl(u32::try_from(exponent).unwrap_or(u32::MAX))
            .unwrap_or(usize::MAX / 8)
            .min(usize::MAX / 8)
    }

    /// Multiply the entire MPS by a scalar absorbed into site zero.
    ///
    /// The mixed-canonical claim is retained only when site zero is the
    /// orthogonality center. [`Self::normalize`] may instead choose a tracked
    /// nonzero center to avoid invalidating an established canonical gauge.
    pub fn scale(&mut self, scalar: Complex64) {
        if self.tensors.is_empty() {
            return;
        }
        self.scale_tensor(0, scalar);
    }

    /// Scale one tensor and retain the mixed-canonical claim exactly when that
    /// tensor is the orthogonality center. Scaling any other tensor changes an
    /// isometry's Gram matrix and therefore invalidates the claim.
    fn scale_tensor(&mut self, site: usize, scalar: Complex64) {
        self.tensors[site] *= scalar;
        if self.center != Some(site) {
            self.center = None;
        }
    }

    /// Apply a single-site gate (d x d unitary matrix) to site `q`.
    ///
    /// For each pair of physical indices (`sigma_out`, `sigma_in)`:
    ///   A'[`alpha_l`, `sigma_out`, `alpha_r`] = sum_{`sigma_in`} gate[`sigma_out`, `sigma_in`] * A[`alpha_l`, `sigma_in`, `alpha_r`]
    ///
    /// # Errors
    ///
    /// Returns [`MpsError::GateDimMismatch`] if the gate dimensions don't match the
    /// physical dimension, or [`MpsError::SiteOutOfBounds`] if `q` is out of range.
    pub fn apply_one_site_gate(
        &mut self,
        q: usize,
        gate: &DMatrix<Complex64>,
    ) -> Result<(), MpsError> {
        let d = self.phys_dim;
        if gate.nrows() != d || gate.ncols() != d {
            return Err(MpsError::GateDimMismatch {
                expected: d,
                rows: gate.nrows(),
                cols: gate.ncols(),
            });
        }
        if q >= self.num_sites {
            return Err(MpsError::SiteOutOfBounds {
                index: q,
                num_sites: self.num_sites,
            });
        }

        // The public operation historically accepts any square matrix even
        // though its documented gate contract is unitary. Keep that behavior,
        // but retain a canonical claim only when the supplied matrix actually
        // preserves the physical-index inner product.
        let preserves_isometries = self.center.is_some()
            && (0..d).all(|row| {
                (0..d).all(|column| {
                    let inner = (0..d)
                        .map(|index| gate[(index, row)].conj() * gate[(index, column)])
                        .sum::<Complex64>();
                    let expected = if row == column { 1.0 } else { 0.0 };
                    (inner - Complex64::new(expected, 0.0)).norm()
                        <= ISOMETRY_PRESERVING_UNITARY_TOLERANCE
                })
            });

        let chi_r = self.bond_dims[q + 1];

        // Collect old blocks
        let old_blocks: Vec<DMatrix<Complex64>> = (0..d)
            .map(|s| phys_block(&self.tensors[q], s, chi_r))
            .collect();

        // Compute new blocks: new_block[sigma_out] = sum_sigma_in gate[sigma_out, sigma_in] * old_block[sigma_in]
        for sigma_out in 0..d {
            let mut new_block = DMatrix::zeros(self.bond_dims[q], chi_r);
            for (sigma_in, old_block) in old_blocks.iter().enumerate() {
                let coeff = gate[(sigma_out, sigma_in)];
                if coeff != Complex64::new(0.0, 0.0) {
                    new_block += old_block * coeff;
                }
            }
            set_phys_block(&mut self.tensors[q], sigma_out, chi_r, &new_block);
        }
        if !preserves_isometries {
            self.center = None;
        }
        Ok(())
    }

    /// Apply a diagonal single-site gate: diag(c0, c1, ...) to site `q`.
    ///
    /// Just scales each physical block by the corresponding coefficient.
    ///
    /// # Errors
    ///
    /// Returns [`MpsError::GateDimMismatch`] if `coeffs.len()` differs from the
    /// physical dimension, or [`MpsError::SiteOutOfBounds`] if `q` is out of range.
    pub fn apply_diagonal_one_site(
        &mut self,
        q: usize,
        coeffs: &[Complex64],
    ) -> Result<(), MpsError> {
        let d = self.phys_dim;
        if coeffs.len() != d {
            return Err(MpsError::GateDimMismatch {
                expected: d,
                rows: d,
                cols: d,
            });
        }
        if q >= self.num_sites {
            return Err(MpsError::SiteOutOfBounds {
                index: q,
                num_sites: self.num_sites,
            });
        }

        let preserves_isometries = self.center.is_some()
            && coeffs.iter().all(|coefficient| {
                (coefficient.norm() - 1.0).abs() <= ISOMETRY_PRESERVING_UNITARY_TOLERANCE
            });
        let chi_r = self.bond_dims[q + 1];
        for (sigma, &c) in coeffs.iter().enumerate() {
            let start_col = sigma * chi_r;
            for j in 0..chi_r {
                for i in 0..self.bond_dims[q] {
                    self.tensors[q][(i, start_col + j)] *= c;
                }
            }
        }
        if !preserves_isometries {
            self.center = None;
        }
        Ok(())
    }

    /// Apply a two-site gate (d^2 x d^2 matrix) to adjacent sites (q, q+1).
    ///
    /// The gate acts on the combined physical space of both sites.
    /// Row/column index = `sigma_l * d + sigma_r`.
    ///
    /// After applying the gate, the two-site tensor is split via SVD with truncation.
    ///
    /// # Errors
    ///
    /// Returns [`MpsError::GateDimMismatch`] if the gate isn't d^2 x d^2,
    /// [`MpsError::SiteOutOfBounds`] if q+1 exceeds the chain, or
    /// [`MpsError::SvdFailed`] if the SVD decomposition fails.
    pub fn apply_two_site_gate(
        &mut self,
        q: usize,
        gate: &DMatrix<Complex64>,
    ) -> Result<(), MpsError> {
        self.apply_two_site_gate_with_absorption(q, gate, false)
    }

    /// Apply an adjacent two-site gate while leaving the orthogonality center
    /// on the right site.
    pub(crate) fn apply_two_site_gate_right_absorb(
        &mut self,
        q: usize,
        gate: &DMatrix<Complex64>,
    ) -> Result<(), MpsError> {
        self.apply_two_site_gate_with_absorption(q, gate, true)
    }

    fn apply_two_site_gate_with_absorption(
        &mut self,
        q: usize,
        gate: &DMatrix<Complex64>,
        absorb_right: bool,
    ) -> Result<(), MpsError> {
        let d = self.phys_dim;
        let d2 = d * d;
        if gate.nrows() != d2 || gate.ncols() != d2 {
            return Err(MpsError::GateDimMismatch {
                expected: d2,
                rows: gate.nrows(),
                cols: gate.ncols(),
            });
        }
        if q + 1 >= self.num_sites {
            return Err(MpsError::NonAdjacentSites { q0: q, q1: q + 1 });
        }

        // Replacing the two-site center by an SVD preserves the outer
        // canonical environments exactly when the old center lies inside the
        // updated pair. A two-site MPS has no outer environments to preserve.
        let can_track_absorption = self.num_sites == 2
            || self
                .center
                .is_some_and(|center| center == q || center == q + 1);

        let chi_l = self.bond_dims[q];
        let chi_mid = self.bond_dims[q + 1];
        let chi_r = self.bond_dims[q + 2];

        // Contract the two site tensors into a two-site tensor
        let two_site = contract_two_sites(
            &self.tensors[q],
            chi_l,
            chi_mid,
            &self.tensors[q + 1],
            chi_r,
            d,
        );

        // Apply the gate to the physical indices
        // two_site: (chi_l, d * d * chi_r)
        // We need to contract gate[sigma_l_out * d + sigma_r_out, sigma_l_in * d + sigma_r_in]
        // with two_site[alpha_l, sigma_l_in * d * chi_r + sigma_r_in * chi_r + alpha_r]
        let mut gated = DMatrix::zeros(chi_l, d * d * chi_r);
        for alpha_l in 0..chi_l {
            for alpha_r in 0..chi_r {
                for sigma_l_out in 0..d {
                    for sigma_r_out in 0..d {
                        let mut val = Complex64::new(0.0, 0.0);
                        for sigma_l_in in 0..d {
                            for sigma_r_in in 0..d {
                                let gate_val = gate
                                    [(sigma_l_out * d + sigma_r_out, sigma_l_in * d + sigma_r_in)];
                                if gate_val != Complex64::new(0.0, 0.0) {
                                    let in_col = (sigma_l_in * d + sigma_r_in) * chi_r + alpha_r;
                                    val += gate_val * two_site[(alpha_l, in_col)];
                                }
                            }
                        }
                        let out_col = (sigma_l_out * d + sigma_r_out) * chi_r + alpha_r;
                        gated[(alpha_l, out_col)] = val;
                    }
                }
            }
        }

        // Reshape for SVD: (chi_l * d, d * chi_r)
        let svd_matrix = reshape_two_site_for_svd(&gated, chi_l, chi_r, d);
        let scaled_cutoff = self.config.svd_cutoff * svd_matrix.norm();

        // SVD split with truncation
        let (left, right, disc, hit) = if absorb_right {
            svd::truncated_svd_right_absorb_with_error(
                &svd_matrix,
                self.config.max_bond_dim,
                scaled_cutoff,
                self.config.max_truncation_error,
            )?
        } else {
            svd::truncated_svd_left_absorb_with_error(
                &svd_matrix,
                self.config.max_bond_dim,
                scaled_cutoff,
                self.config.max_truncation_error,
            )?
        };
        self.record_truncation(disc, hit);

        let new_chi = left.ncols();

        // U or U_S: (chi_l * d, new_chi) -> reshape to the left site.
        self.tensors[q] = reshape_left_ungroup(&left, chi_l, d, new_chi);

        // Vt or S_Vt is already in right-site tensor format.
        self.tensors[q + 1] = right;

        // Update bond dimension
        self.bond_dims[q + 1] = new_chi;
        self.center = can_track_absorption.then_some(if absorb_right { q + 1 } else { q });
        self.record_current_peak_bond();

        Ok(())
    }

    /// Apply a two-site gate between arbitrary (possibly non-adjacent) sites.
    ///
    /// Uses SWAP gates to bring site `q1` adjacent to `q0`, applies the gate,
    /// then SWAPs back. `q0 < q1` required.
    ///
    /// Before the first split, the MPS is put in mixed-canonical form around
    /// the outermost SWAP bond. Left absorption moves the orthogonality center
    /// inward with the transported site; right absorption moves it outward on
    /// the return path. Every truncating SVD therefore sees physical Schmidt
    /// weights rather than gauge-dependent local singular values.
    ///
    /// On success, an adjacent gate leaves the orthogonality center at `q0`;
    /// a non-adjacent gate leaves it at `q1` after the returning SWAP chain.
    ///
    /// # Errors
    ///
    /// Returns [`MpsError::NonAdjacentSites`] if `q0 >= q1`,
    /// [`MpsError::SiteOutOfBounds`] if `q1` exceeds the chain, or
    /// [`MpsError::SvdFailed`] if any intermediate SVD fails.
    pub fn apply_long_range_two_site_gate(
        &mut self,
        q0: usize,
        q1: usize,
        gate: &DMatrix<Complex64>,
    ) -> Result<(), MpsError> {
        if q0 >= q1 {
            return Err(MpsError::NonAdjacentSites { q0, q1 });
        }
        if q1 >= self.num_sites {
            return Err(MpsError::SiteOutOfBounds {
                index: q1,
                num_sites: self.num_sites,
            });
        }

        let expected_gate_dim = self.phys_dim * self.phys_dim;
        if gate.nrows() != expected_gate_dim || gate.ncols() != expected_gate_dim {
            return Err(MpsError::GateDimMismatch {
                expected: expected_gate_dim,
                rows: gate.nrows(),
                cols: gate.ncols(),
            });
        }

        // Adjacent case: establish physical environments before truncating.
        if q1 == q0 + 1 {
            self.canonicalize_around_bond(q0);
            return self.apply_two_site_gate(q0, gate);
        }

        // Non-adjacent: SWAP chain to bring sites together, apply gate, SWAP back.
        let swap = DMatrix::from_row_slice(
            4,
            4,
            &[
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
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        );

        // Establish a mixed-canonical form once. Left absorption on the
        // inward SWAPs walks the center from q1 - 1 down to q0.
        self.canonicalize_around_bond(q1 - 1);
        for i in (q0 + 1..q1).rev() {
            self.apply_two_site_gate(i, &swap)?;
        }

        // Right absorption starts the center back outward.
        self.apply_two_site_gate_right_absorb(q0, gate)?;

        // Keep moving the center with the transported site on the return path.
        for i in q0 + 1..q1 {
            self.apply_two_site_gate_right_absorb(i, &swap)?;
        }

        Ok(())
    }

    /// Compute the squared norm `<psi|psi>` by contracting the MPS with itself.
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        // Contract from left to right, building the transfer matrix product.
        // E[alpha, beta] = sum_{sigma} A*[alpha, sigma] A[beta, sigma]
        // Start with E = 1x1 identity.
        let mut transfer = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));

        for q in 0..self.num_sites {
            transfer = advance_left_identity_environment(
                &transfer,
                &self.tensors[q],
                self.phys_dim,
                self.bond_dims[q + 1],
            );
        }

        // Final transfer is 1x1
        transfer[(0, 0)].re
    }

    /// Cache the left and right identity environments bordering `sites`.
    ///
    /// This is intended for a pass that evaluates several local observables
    /// against one immutable tensor state. The returned cache borrows `self`,
    /// so tensor mutation cannot occur until the cache is dropped. Construction
    /// contracts the suffix through the leftmost candidate and the prefix
    /// through the rightmost candidate, retaining environments only at the
    /// requested sites.
    ///
    /// # Panics
    ///
    /// Panics if `sites` is empty or contains an out-of-range site.
    #[must_use]
    pub fn environment_cache(&self, sites: &[usize]) -> MpsEnvironmentCache<'_> {
        assert!(!sites.is_empty(), "environment cache requires a site");
        let mut sites = sites.to_vec();
        sites.sort_unstable();
        sites.dedup();
        assert!(
            sites[sites.len() - 1] < self.num_sites,
            "MPS site out of range"
        );

        let first_site = sites[0];
        let last_site = sites[sites.len() - 1];
        let boundary = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));

        let mut right_reversed = Vec::with_capacity(sites.len());
        let mut right_environment = boundary.clone();
        let mut candidate = sites.len();
        for site in (first_site..self.num_sites).rev() {
            if site == sites[candidate - 1] {
                right_reversed.push(right_environment.clone());
                candidate -= 1;
            }
            right_environment = advance_right_identity_environment(
                &right_environment,
                &self.tensors[site],
                self.phys_dim,
                self.bond_dims[site + 1],
            );
        }
        right_reversed.reverse();

        let mut left = Vec::with_capacity(sites.len());
        let mut left_environment = boundary;
        let mut candidate = 0;
        let mut norm_squared = 0.0;
        for site in 0..=last_site {
            if site == sites[candidate] {
                if candidate == 0 {
                    let norm: Complex64 = left_environment
                        .component_mul(&right_environment)
                        .iter()
                        .sum();
                    norm_squared = norm.re;
                }
                left.push(left_environment.clone());
                candidate += 1;
            }
            if site < last_site {
                left_environment = advance_left_identity_environment(
                    &left_environment,
                    &self.tensors[site],
                    self.phys_dim,
                    self.bond_dims[site + 1],
                );
            }
        }

        MpsEnvironmentCache {
            mps: self,
            sites,
            left,
            right: right_reversed,
            norm_squared,
        }
    }

    /// Compute `<mps| O |mps>` where O is a product of per-site 2x2 operators.
    ///
    /// `ops` maps site index -> 2x2 matrix. Sites not in `ops` get identity.
    /// Returns the complex expectation value.
    #[must_use]
    pub fn expectation_product(&self, ops: &[(usize, DMatrix<Complex64>)]) -> Complex64 {
        let d = self.phys_dim;
        let mut transfer = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));

        // Build a lookup for which sites have operators
        let mut site_ops: Vec<Option<&DMatrix<Complex64>>> = vec![None; self.num_sites];
        for (site, op) in ops {
            site_ops[*site] = Some(op);
        }

        for (q, site_op) in site_ops.iter().enumerate() {
            let chi_r = self.bond_dims[q + 1];
            let t = &self.tensors[q];

            let mut new_transfer = DMatrix::zeros(chi_r, chi_r);

            if let Some(op) = site_op {
                // <bra| O_q |ket> at this site
                // new_transfer = sum_{sigma_bra, sigma_ket} conj(A[sigma_bra])^T * transfer * A[sigma_ket] * O[sigma_bra, sigma_ket]
                for sigma_bra in 0..d {
                    let bra_block = phys_block(t, sigma_bra, chi_r);
                    let conj_bra_t = bra_block.conjugate().transpose();
                    for sigma_ket in 0..d {
                        let o_val = op[(sigma_bra, sigma_ket)];
                        if o_val.norm() < 1e-15 {
                            continue;
                        }
                        let ket_block = phys_block(t, sigma_ket, chi_r);
                        let tmp = &conj_bra_t * &transfer * &ket_block;
                        new_transfer += tmp * o_val;
                    }
                }
            } else {
                // Identity at this site (same as norm_squared)
                for sigma in 0..d {
                    let block = phys_block(t, sigma, chi_r);
                    let conj_block_t = block.conjugate().transpose();
                    let tmp = &conj_block_t * &transfer * &block;
                    new_transfer += tmp;
                }
            }

            transfer = new_transfer;
        }

        transfer[(0, 0)]
    }

    /// Normalize the MPS so that `<psi|psi> = 1`.
    pub fn normalize(&mut self) {
        if self.tensors.is_empty() {
            return;
        }
        // In a mixed-canonical gauge every environment contraction outside
        // the center is the identity, so the center tensor's Frobenius norm
        // is the global state norm. Besides avoiding a redundant contraction,
        // this keeps normalization at the invariant-owning site.
        let norm_sq = self.center.map_or_else(
            || self.norm_squared(),
            |center| {
                // This consult consumes the center claim for a VALUE, not
                // just to skip work: a stale claim would silently
                // mis-normalize. Guard it like every other trusting consult.
                #[cfg(debug_assertions)]
                debug_assert!(
                    self.claimed_center_is_valid(center),
                    "tracked MPS orthogonality center {center} is stale"
                );
                self.tensors[center].iter().map(Complex64::norm_sqr).sum()
            },
        );
        debug_assert!(
            norm_sq > 0.0,
            "cannot normalize a zero-norm MPS after projection"
        );
        if norm_sq > 0.0 {
            let inv_norm = Complex64::new(1.0 / norm_sq.sqrt(), 0.0);
            let site = self.center.unwrap_or(0);
            self.scale_tensor(site, inv_norm);
        }
    }

    /// Extract the amplitude for a given computational basis state.
    ///
    /// `basis_state[i]` is the physical index (0 or 1) at site `i`.
    ///
    /// # Panics
    ///
    /// Panics if `basis_state.len() != self.num_sites`.
    #[must_use]
    pub fn amplitude(&self, basis_state: &[u8]) -> Complex64 {
        assert_eq!(basis_state.len(), self.num_sites);

        // Contract: A[0]^{s_0} * A[1]^{s_1} * ... * A[N-1]^{s_{N-1}}
        // Each A[i]^{s_i} is a (chi_l, chi_r) matrix. Product is a 1x1 scalar.
        let mut result = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));
        for (q, &sigma) in basis_state.iter().enumerate() {
            let sigma = sigma as usize;
            let chi_r = self.bond_dims[q + 1];
            let block = phys_block(&self.tensors[q], sigma, chi_r);
            result = &result * &block;
        }
        result[(0, 0)]
    }

    /// Compute the full state vector (`2^N` complex amplitudes).
    ///
    /// This performs `2^N` MPS contractions and allocates `2^N` complex
    /// values, so it is only suitable for testing and other small-system
    /// reads. Prefer [`Self::amplitude`] for selected basis states and
    /// [`Self::expectation_product`] for product-observable expectations.
    /// When `parallel` is enabled in the config, amplitude computations run on
    /// rayon's thread pool.
    ///
    /// # Panics
    ///
    /// Panics if `num_sites > 20`.
    #[must_use]
    pub fn state_vector(&self) -> Vec<Complex64> {
        assert!(
            self.num_sites <= 20,
            "state_vector is only for small systems (N <= 20)"
        );
        let dim = 1 << self.num_sites;
        let n = self.num_sites;

        let to_basis = |idx: usize| -> Vec<u8> {
            (0..n)
                .map(|q| u8::try_from((idx >> (n - 1 - q)) & 1).unwrap())
                .collect()
        };

        if self.config.parallel {
            (0..dim)
                .into_par_iter()
                .map(|idx| self.amplitude(&to_basis(idx)))
                .collect()
        } else {
            (0..dim).map(|idx| self.amplitude(&to_basis(idx))).collect()
        }
    }

    /// Add two MPS of the same structure (direct sum of bond spaces).
    ///
    /// The result has bond dimension `chi_self + chi_other` at each internal bond.
    /// Should be followed by SVD truncation (e.g. via `left_canonicalize` + truncate).
    ///
    /// # Panics
    ///
    /// Panics if `self` and `other` differ in `num_sites` or `phys_dim`.
    #[must_use]
    pub fn add(&self, other: &Self) -> Self {
        assert_eq!(self.num_sites, other.num_sites);
        assert_eq!(self.phys_dim, other.phys_dim);
        let d = self.phys_dim;
        let n = self.num_sites;

        let mut new_bond_dims = vec![1; n + 1];
        for (new_bd, (bd_s, bd_o)) in new_bond_dims[1..n].iter_mut().zip(
            self.bond_dims[1..n]
                .iter()
                .zip(other.bond_dims[1..n].iter()),
        ) {
            *new_bd = bd_s + bd_o;
        }

        let mut new_tensors = Vec::with_capacity(n);
        for q in 0..n {
            let chi_l_s = self.bond_dims[q];
            let chi_r_s = self.bond_dims[q + 1];
            let chi_l_o = other.bond_dims[q];
            let chi_r_o = other.bond_dims[q + 1];
            let chi_l_new = new_bond_dims[q];
            let chi_r_new = new_bond_dims[q + 1];

            let mut t = DMatrix::zeros(chi_l_new, d * chi_r_new);

            for sigma in 0..d {
                // Place self's block in top-left
                let block_s = phys_block(&self.tensors[q], sigma, chi_r_s);
                for i in 0..chi_l_s {
                    for j in 0..chi_r_s {
                        t[(i, sigma * chi_r_new + j)] = block_s[(i, j)];
                    }
                }

                // Place other's block in bottom-right (or add at boundaries)
                let block_o = phys_block(&other.tensors[q], sigma, chi_r_o);
                let row_offset = if q == 0 { 0 } else { chi_l_s };
                let col_offset = if q == n - 1 { 0 } else { chi_r_s };
                for i in 0..chi_l_o {
                    for j in 0..chi_r_o {
                        t[(row_offset + i, sigma * chi_r_new + col_offset + j)] += block_o[(i, j)];
                    }
                }
            }

            new_tensors.push(t);
        }

        let current_peak = new_bond_dims.iter().copied().max().unwrap_or(1);
        Self {
            num_sites: n,
            phys_dim: d,
            tensors: new_tensors,
            center: None,
            bond_dims: new_bond_dims,
            config: self.config.clone(),
            truncation_error: self.truncation_error.max(other.truncation_error),
            bond_cap_hits: self.bond_cap_hits + other.bond_cap_hits,
            phase_svd_operations: self.phase_svd_operations + other.phase_svd_operations,
            phase_capped_svd_operations: self.phase_capped_svd_operations
                + other.phase_capped_svd_operations,
            summed_discarded_weight: self
                .summed_discarded_weight
                .max(other.summed_discarded_weight),
            lifetime_peak_bond: self
                .lifetime_peak_bond
                .max(other.lifetime_peak_bond)
                .max(current_peak),
            branch_vanish_retry_count: self
                .branch_vanish_retry_count
                .max(other.branch_vanish_retry_count),
            deferred_branch_lost_count: self
                .deferred_branch_lost_count
                .max(other.deferred_branch_lost_count),
            full_canonical_sweep_count: self
                .full_canonical_sweep_count
                .max(other.full_canonical_sweep_count),
            center_reuse_count: self.center_reuse_count.max(other.center_reuse_count),
        }
    }

    /// Access the internal tensors.
    #[must_use]
    pub fn tensors(&self) -> &[DMatrix<Complex64>] {
        &self.tensors
    }

    /// Mutable access to the internal tensors.
    pub fn tensors_mut(&mut self) -> &mut [DMatrix<Complex64>] {
        self.center = None;
        &mut self.tensors
    }

    #[cfg(test)]
    pub(crate) fn tracked_center_for_test(&self) -> Option<usize> {
        self.center
    }

    #[cfg(test)]
    pub(crate) fn set_tracked_center_for_test(&mut self, center: Option<usize>) {
        self.center = center;
    }

    /// Replace one physical block of a site tensor.
    ///
    /// This is the owner-mediated path for projection code that must mutate a
    /// tensor without canonicalizing first. Any replacement at the center
    /// preserves the outer left and right isometries; a replacement away from
    /// the center changes an isometry and invalidates the claim.
    pub(crate) fn set_physical_block(
        &mut self,
        site: usize,
        physical_index: usize,
        block: &DMatrix<Complex64>,
    ) {
        let chi_r = self.bond_dims[site + 1];
        set_phys_block(&mut self.tensors[site], physical_index, chi_r, block);
        if self.center != Some(site) {
            self.center = None;
        }
    }

    /// Access the bond dimensions (for testing).
    #[must_use]
    pub fn bond_dims(&self) -> &[usize] {
        &self.bond_dims
    }

    /// Left-canonicalize the entire MPS.
    pub fn left_canonicalize(&mut self) {
        canon::left_canonicalize_all(&mut self.tensors, &mut self.bond_dims, self.phys_dim);
        self.center = self.num_sites.checked_sub(1);
        self.record_current_peak_bond();
    }

    /// Right-canonicalize the MPS by moving the orthogonality center to
    /// site 0. With a tracked center this is a local walk, and a no-op when
    /// the center is already at site 0 — safe because the sites a valid
    /// center claim skips are exact isometries, so the omitted factorizations
    /// are pure unitary gauge moves that cannot change any bond dimension.
    pub fn right_canonicalize(&mut self) {
        if self.num_sites > 0 {
            self.canonicalize_at(0);
        }
    }

    /// Put the environments bordering `(q, q + 1)` in canonical form.
    ///
    /// After this operation, sites through `q` are left-canonical and sites
    /// strictly right of `q + 1` are right-canonical, leaving a one-site
    /// orthogonality center at `q + 1`. Consequently, singular values obtained
    /// by splitting the two-site center have their physical Schmidt weights
    /// even when the input MPS had an arbitrary or rank-redundant gauge.
    pub(crate) fn canonicalize_around_bond(&mut self, q: usize) {
        assert!(q + 1 < self.num_sites, "bond must join two valid sites");
        self.canonicalize_at(q + 1);
    }

    /// Move an established center with exact one-site QR factorizations, or
    /// establish one from a cold gauge by canonicalizing both environments.
    fn canonicalize_at(&mut self, target: usize) {
        assert!(target < self.num_sites, "center must be a valid site");
        if let Some(center) = self.center {
            self.center_reuse_count += 1;
            #[cfg(debug_assertions)]
            debug_assert!(
                self.claimed_center_is_valid(center),
                "tracked MPS orthogonality center {center} is stale"
            );

            if center < target {
                for site in center..target {
                    canon::left_canonicalize_site(
                        &mut self.tensors,
                        &mut self.bond_dims,
                        site,
                        self.phys_dim,
                    );
                    self.center = Some(site + 1);
                }
            } else {
                for site in (target + 1..=center).rev() {
                    canon::right_canonicalize_site(
                        &mut self.tensors,
                        &mut self.bond_dims,
                        site,
                        self.phys_dim,
                    );
                    self.center = Some(site - 1);
                }
            }
        } else {
            self.full_canonical_sweep_count += 1;
            for site in 0..target {
                canon::left_canonicalize_site(
                    &mut self.tensors,
                    &mut self.bond_dims,
                    site,
                    self.phys_dim,
                );
            }
            for site in (target + 1..self.num_sites).rev() {
                canon::right_canonicalize_site(
                    &mut self.tensors,
                    &mut self.bond_dims,
                    site,
                    self.phys_dim,
                );
            }
            self.center = Some(target);
        }
        self.record_current_peak_bond();
    }

    /// Validate a tracked center to an absolute max-entry Gram tolerance of
    /// `1e-9`. This is compiled only when debug assertions are enabled and is
    /// called only before a canonicalization sweep trusts the claim to skip
    /// work.
    #[cfg(debug_assertions)]
    fn claimed_center_is_valid(&self, center: usize) -> bool {
        const TOLERANCE: f64 = 1e-9;

        if center >= self.num_sites {
            return false;
        }
        for site in 0..self.num_sites {
            if site == center {
                continue;
            }
            let gram = if site < center {
                let grouped = tensor::reshape_left_group(
                    &self.tensors[site],
                    self.bond_dims[site],
                    self.phys_dim,
                    self.bond_dims[site + 1],
                );
                grouped.adjoint() * grouped
            } else {
                &self.tensors[site] * self.tensors[site].adjoint()
            };
            for row in 0..gram.nrows() {
                for column in 0..gram.ncols() {
                    let expected = if row == column { 1.0 } else { 0.0 };
                    let error = (gram[(row, column)] - Complex64::new(expected, 0.0)).norm();
                    if !error.is_finite() || error > TOLERANCE {
                        return false;
                    }
                }
            }
        }
        true
    }

    /// Compress the MPS by SVD truncation at each bond.
    ///
    /// Left-canonicalizes first, then sweeps right-to-left performing SVD
    /// truncation at each bond to enforce `max_bond_dim` and `svd_cutoff`.
    /// `svd_cutoff` is relative to the centre tensor's Frobenius norm, which
    /// equals the global state norm in this mixed-canonical sweep. Therefore a
    /// global scalar does not change retained ranks or truncation telemetry.
    ///
    /// # Errors
    ///
    /// Returns [`MpsError::SvdFailed`] if a bond factorization fails. The
    /// failing bond and every bond later in the sweep are left unmodified;
    /// callers must not treat a partial sweep as successful compression.
    pub fn compress(&mut self) -> Result<(), MpsError> {
        if self.num_sites <= 1 {
            self.center = (self.num_sites == 1).then_some(0);
            return Ok(());
        }

        // Left-canonicalize
        self.left_canonicalize();

        // Sweep right to left: retain Vt at site q so it is right-canonical,
        // and absorb U*S into q-1 so the orthogonality center follows the
        // sweep. The singular values at every subsequent bond are therefore
        // physical Schmidt weights.
        let d = self.phys_dim;
        for q in (1..self.num_sites).rev() {
            let chi_l = self.bond_dims[q];

            // Reshape site q from (chi_l, d * chi_r) to (chi_l, d * chi_r) -- already in this form.
            // But we want to split the left bond, so transpose the grouping:
            // Reshape to (chi_l, d * chi_r) and do SVD to split as (chi_l, new_chi) * (new_chi, d * chi_r).
            let matrix = &self.tensors[q];
            let scaled_cutoff = self.config.svd_cutoff * matrix.norm();
            let (us, vt, disc, hit) = svd::truncated_svd_left_absorb_with_error(
                matrix,
                self.config.max_bond_dim,
                scaled_cutoff,
                self.config.max_truncation_error,
            )?;
            self.record_truncation(disc, hit);
            let new_chi = us.ncols();

            // Vt is right-canonical even when the retained rank is
            // unchanged, so always install the factorization.
            self.tensors[q] = vt;
            self.bond_dims[q] = new_chi;

            // Absorb U*S into tensors[q-1].
            let chi_l_prev = self.bond_dims[q - 1];
            let mut new_prev = DMatrix::zeros(chi_l_prev, d * new_chi);
            for sigma in 0..d {
                let prev_block = tensor::phys_block(&self.tensors[q - 1], sigma, chi_l);
                let absorbed = &prev_block * &us;
                for i in 0..chi_l_prev {
                    for j in 0..new_chi {
                        new_prev[(i, sigma * new_chi + j)] = absorbed[(i, j)];
                    }
                }
            }
            self.tensors[q - 1] = new_prev;
            self.center = Some(q - 1);
        }

        // Preserve the established post-compression left-canonical contract
        // for downstream projection code. This exact QR sweep happens only
        // after every truncation has been evaluated in the right-to-left
        // mixed-canonical gauge above.
        self.left_canonicalize();
        self.record_current_peak_bond();
        Ok(())
    }

    /// Compress from an established right-canonical gauge.
    ///
    /// The orthogonality center starts at site zero and follows this
    /// left-to-right SVD sweep. Thus every split sees physical Schmidt
    /// weights while avoiding the cold left-canonical sweep used by
    /// [`Self::compress`]. The configured cutoff, cap, and adaptive error
    /// budget are applied unchanged.
    pub(crate) fn compress_from_right_canonical(&mut self) -> Result<(), MpsError> {
        if self.num_sites <= 1 {
            self.center = (self.num_sites == 1).then_some(0);
            return Ok(());
        }
        debug_assert_eq!(self.center, Some(0));
        #[cfg(debug_assertions)]
        debug_assert!(self.claimed_center_is_valid(0));

        let d = self.phys_dim;
        for q in 0..self.num_sites - 1 {
            let chi_l = self.bond_dims[q];
            let chi_r = self.bond_dims[q + 1];
            let matrix = tensor::reshape_left_group(&self.tensors[q], chi_l, d, chi_r);
            let scaled_cutoff = self.config.svd_cutoff * matrix.norm();
            let (u, svt, disc, hit) = svd::truncated_svd_right_absorb_with_error(
                &matrix,
                self.config.max_bond_dim,
                scaled_cutoff,
                self.config.max_truncation_error,
            )?;
            self.record_truncation(disc, hit);
            let new_chi = u.ncols();

            self.tensors[q] = reshape_left_ungroup(&u, chi_l, d, new_chi);
            self.tensors[q + 1] = &svt * &self.tensors[q + 1];
            self.bond_dims[q + 1] = new_chi;
            self.center = Some(q + 1);
        }
        self.record_current_peak_bond();
        Ok(())
    }
}

impl Clone for Mps {
    fn clone(&self) -> Self {
        Self {
            num_sites: self.num_sites,
            phys_dim: self.phys_dim,
            tensors: self.tensors.clone(),
            center: self.center,
            bond_dims: self.bond_dims.clone(),
            config: self.config.clone(),
            truncation_error: self.truncation_error,
            bond_cap_hits: self.bond_cap_hits,
            phase_svd_operations: self.phase_svd_operations,
            phase_capped_svd_operations: self.phase_capped_svd_operations,
            summed_discarded_weight: self.summed_discarded_weight,
            lifetime_peak_bond: self.lifetime_peak_bond,
            branch_vanish_retry_count: self.branch_vanish_retry_count,
            deferred_branch_lost_count: self.deferred_branch_lost_count,
            full_canonical_sweep_count: self.full_canonical_sweep_count,
            center_reuse_count: self.center_reuse_count,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn normalized_fidelity(first: &[Complex64], second: &[Complex64]) -> f64 {
        let overlap = first
            .iter()
            .zip(second)
            .map(|(a, b)| a.conj() * b)
            .sum::<Complex64>();
        let first_norm = first.iter().map(Complex64::norm_sqr).sum::<f64>();
        let second_norm = second.iter().map(Complex64::norm_sqr).sum::<f64>();
        overlap.norm_sqr() / (first_norm * second_norm)
    }

    #[test]
    fn truncation_telemetry_separates_product_error_sum_and_lifetime_peak() {
        let mut mps = Mps::new(3, MpsConfig::default());
        mps.record_truncation(0.1, false);
        mps.record_truncation(0.2, true);
        assert!((mps.truncation_error() - 0.28).abs() < 1e-15);
        assert!((mps.summed_discarded_weight() - 0.3).abs() < 1e-15);
        assert_eq!(mps.bond_cap_hits(), 1);

        let doubled = mps.add(&mps);
        assert_eq!(doubled.max_bond_dim(), 2);
        assert_eq!(doubled.lifetime_peak_bond(), 2);
        let mut reduced = doubled;
        reduced.compress().unwrap();
        assert_eq!(reduced.lifetime_peak_bond(), 2);
    }

    #[cfg(not(debug_assertions))]
    fn apply_adjacent_dense_gate(
        state: &[Complex64],
        num_sites: usize,
        q: usize,
        gate: &DMatrix<Complex64>,
    ) -> Vec<Complex64> {
        let left_shift = num_sites - 1 - q;
        let right_shift = left_shift - 1;
        let mut result = vec![Complex64::new(0.0, 0.0); state.len()];
        for (input_index, &amplitude) in state.iter().enumerate() {
            let input_left = input_index >> left_shift & 1;
            let input_right = input_index >> right_shift & 1;
            let input = 2 * input_left + input_right;
            let cleared = input_index & !(1 << left_shift) & !(1 << right_shift);
            for output in 0..4 {
                let output_index =
                    cleared | (output >> 1 & 1) << left_shift | (output & 1) << right_shift;
                result[output_index] += gate[(output, input)] * amplitude;
            }
        }
        result
    }

    fn seeded_random_mps(num_sites: usize, bond: usize, seed: u64, config: MpsConfig) -> Mps {
        assert!(bond.is_power_of_two());
        let mut mps = Mps::new(num_sites, config);
        while mps.max_bond_dim() < bond {
            mps = mps.add(&mps);
        }

        let mut random = seed;
        for tensor in mps.tensors_mut() {
            for value in tensor.iter_mut() {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let real = (random as f64 / u64::MAX as f64) - 0.5;
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let imag = (random as f64 / u64::MAX as f64) - 0.5;
                *value = Complex64::new(real, imag);
            }
        }
        let inverse_norm = mps.norm_squared().sqrt().recip();
        mps.scale(Complex64::new(inverse_norm, 0.0));
        mps
    }

    fn legacy_off_center_compress(mps: &mut Mps) {
        if mps.num_sites <= 1 {
            return;
        }
        mps.left_canonicalize();
        let d = mps.phys_dim;
        for q in (1..mps.num_sites).rev() {
            let chi_l = mps.bond_dims[q];
            let (u, svt, _, _) = svd::truncated_svd_right_absorb_with_error(
                &mps.tensors[q],
                mps.config.max_bond_dim,
                mps.config.svd_cutoff,
                mps.config.max_truncation_error,
            )
            .unwrap();
            let new_chi = u.ncols();
            if new_chi < chi_l {
                mps.tensors[q] = svt;
                mps.bond_dims[q] = new_chi;
                let chi_l_prev = mps.bond_dims[q - 1];
                let mut new_prev = DMatrix::zeros(chi_l_prev, d * new_chi);
                for sigma in 0..d {
                    let prev_block = tensor::phys_block(&mps.tensors[q - 1], sigma, chi_l);
                    let absorbed = &prev_block * &u;
                    for i in 0..chi_l_prev {
                        for j in 0..new_chi {
                            new_prev[(i, sigma * new_chi + j)] = absorbed[(i, j)];
                        }
                    }
                }
                mps.tensors[q - 1] = new_prev;
            }
        }
        mps.center = None;
    }

    fn apply_diagonal_bond_gauge(mps: &mut Mps, bond: usize, scales: &[f64]) {
        let chi = mps.bond_dims[bond];
        assert_eq!(scales.len(), chi);
        let gauge = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            chi,
            scales.iter().map(|&scale| Complex64::new(scale, 0.0)),
        ));
        let gauge_inverse = DMatrix::from_diagonal(&nalgebra::DVector::from_iterator(
            chi,
            scales
                .iter()
                .map(|&scale| Complex64::new(scale.recip(), 0.0)),
        ));

        let left_site = bond - 1;
        let chi_l = mps.bond_dims[left_site];
        let mut gauged_left = DMatrix::zeros(chi_l, mps.phys_dim * chi);
        for sigma in 0..mps.phys_dim {
            let block = tensor::phys_block(&mps.tensors[left_site], sigma, chi);
            let gauged_block = block * &gauge;
            gauged_left
                .view_mut((0, sigma * chi), (chi_l, chi))
                .copy_from(&gauged_block);
        }
        mps.tensors[left_site] = gauged_left;
        mps.tensors[bond] = gauge_inverse * &mps.tensors[bond];
        mps.center = None;
    }

    fn legacy_off_center_long_range_gate(
        mps: &mut Mps,
        q0: usize,
        q1: usize,
        gate: &DMatrix<Complex64>,
    ) {
        let zero = Complex64::new(0.0, 0.0);
        let one = Complex64::new(1.0, 0.0);
        let swap = DMatrix::from_row_slice(
            4,
            4,
            &[
                one, zero, zero, zero, zero, zero, one, zero, zero, one, zero, zero, zero, zero,
                zero, one,
            ],
        );
        for site in (q0 + 1..q1).rev() {
            mps.apply_two_site_gate(site, &swap).unwrap();
        }
        mps.apply_two_site_gate(q0, gate).unwrap();
        for site in q0 + 1..q1 {
            mps.apply_two_site_gate(site, &swap).unwrap();
        }
    }

    fn one_site_basis_marginal_reference(mps: &Mps, site: usize, physical_index: usize) -> f64 {
        let mut projector = DMatrix::zeros(mps.phys_dim(), mps.phys_dim());
        projector[(physical_index, physical_index)] = Complex64::new(1.0, 0.0);
        mps.expectation_product(&[(site, projector)]).re / mps.norm_squared()
    }

    #[test]
    fn test_default_config_values() {
        let config = MpsConfig::default();
        assert_eq!(config.max_bond_dim, 128);
        assert_relative_eq!(config.svd_cutoff, 1e-12);
        assert_eq!(config.max_truncation_error, Some(1e-8));
        assert!(!config.parallel);
    }

    #[test]
    fn test_center_tracks_canonical_moves_and_unitary_updates() {
        let mut mps = Mps::new(5, MpsConfig::default());
        assert_eq!(mps.center, Some(0));

        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
            ],
        );
        mps.apply_one_site_gate(3, &h).unwrap();
        assert_eq!(mps.center, Some(0));

        mps.canonicalize_around_bond(3);
        assert_eq!(mps.center, Some(4));
        #[cfg(debug_assertions)]
        assert!(mps.claimed_center_is_valid(4));
        mps.canonicalize_around_bond(1);
        assert_eq!(mps.center, Some(2));
        #[cfg(debug_assertions)]
        assert!(mps.claimed_center_is_valid(2));
        assert_eq!(mps.clone().center, Some(2));
    }

    #[test]
    fn physical_rank_ceiling_is_safe_for_randomized_svd_arithmetic() {
        for num_sites in 124..=127 {
            let mps = Mps::new(num_sites, MpsConfig::default());
            assert!(mps.physical_rank_ceiling() <= usize::MAX / 8);
        }
    }

    #[test]
    fn test_one_site_gate_invalidates_when_unitarity_error_exceeds_preservation_budget() {
        let mut mps = Mps::new(4, MpsConfig::default());
        mps.left_canonicalize();
        let almost_unitary = DMatrix::from_diagonal(&nalgebra::DVector::from_row_slice(&[
            Complex64::new(1.0 + 1e-10, 0.0),
            Complex64::new(1.0, 0.0),
        ]));

        mps.apply_one_site_gate(1, &almost_unitary).unwrap();

        assert_eq!(mps.tracked_center_for_test(), None);
    }

    #[test]
    fn test_unit_modulus_diagonal_preserves_center() {
        let mut mps = Mps::new(4, MpsConfig::default());
        mps.left_canonicalize();
        let center = mps.tracked_center_for_test();
        let phases = [
            Complex64::new(0.0, 1.0),
            Complex64::new(
                std::f64::consts::FRAC_1_SQRT_2,
                -std::f64::consts::FRAC_1_SQRT_2,
            ),
        ];

        mps.apply_diagonal_one_site(1, &phases).unwrap();

        assert_eq!(mps.tracked_center_for_test(), center);
        #[cfg(debug_assertions)]
        assert!(mps.claimed_center_is_valid(center.unwrap()));
    }

    #[test]
    fn test_non_unit_modulus_diagonal_invalidates_center() {
        let mut mps = Mps::new(4, MpsConfig::default());
        mps.left_canonicalize();

        mps.apply_diagonal_one_site(1, &[Complex64::new(1.0, 0.0), Complex64::new(0.5, 0.0)])
            .unwrap();

        assert_eq!(mps.tracked_center_for_test(), None);
    }

    #[test]
    fn test_scale_preserves_only_when_site_zero_is_the_center() {
        let mut mps = Mps::new(4, MpsConfig::default());
        assert_eq!(mps.tracked_center_for_test(), Some(0));

        mps.scale(Complex64::new(2.0, 0.0));
        assert_eq!(mps.tracked_center_for_test(), Some(0));
        mps.canonicalize_around_bond(1);
        assert_eq!(mps.tracked_center_for_test(), Some(2));
        assert_eq!(mps.full_canonical_sweep_count(), 0);
        assert_eq!(mps.center_reuse_count(), 1);
    }

    #[test]
    fn test_normalize_preserves_a_nonzero_center() {
        let config = MpsConfig {
            max_bond_dim: 64,
            svd_cutoff: 0.0,
            max_truncation_error: Some(0.0),
            parallel: false,
        };
        let mut mps = seeded_random_mps(5, 4, 0x5ca1_e000_0000_0001, config);
        mps.canonicalize_around_bond(2);
        assert_eq!(mps.tracked_center_for_test(), Some(3));

        mps.normalize();
        assert_eq!(mps.tracked_center_for_test(), Some(3));
        assert_relative_eq!(mps.norm_squared(), 1.0, epsilon = 1e-12);

        // This is the production consult point: in debug builds it runs the
        // claimed-center validator before trusting the preserved center.
        mps.canonicalize_around_bond(1);
        assert_eq!(mps.tracked_center_for_test(), Some(2));
        assert_eq!(mps.full_canonical_sweep_count(), 1);
        assert_eq!(mps.center_reuse_count(), 1);
    }

    #[test]
    fn test_off_center_scaling_invalidates_before_canonical_routing() {
        let mut mps = Mps::new(4, MpsConfig::default());
        mps.left_canonicalize();
        assert_eq!(mps.tracked_center_for_test(), Some(3));

        mps.scale(Complex64::new(2.0, 0.0));

        mps.canonicalize_around_bond(1);
        assert_eq!(mps.tracked_center_for_test(), Some(2));
        assert_eq!(mps.full_canonical_sweep_count(), 1);
        assert_eq!(mps.center_reuse_count(), 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    fn test_claimed_center_validator_rejects_broken_left_isometry() {
        let mut mps = Mps::new(4, MpsConfig::default());
        mps.left_canonicalize();
        let center = mps.tracked_center_for_test().unwrap();
        mps.tensors[0] *= Complex64::new(2.0, 0.0);

        assert!(!mps.claimed_center_is_valid(center));
    }

    #[test]
    fn test_add_invalidation_guards_canonical_routing() {
        let config = MpsConfig {
            max_bond_dim: 64,
            svd_cutoff: 0.0,
            max_truncation_error: Some(0.0),
            parallel: false,
        };
        let mut first = seeded_random_mps(5, 4, 0xadd0_0001, config.clone());
        let mut second = seeded_random_mps(5, 4, 0xadd0_0002, config);
        first.left_canonicalize();
        second.left_canonicalize();

        let sum = first.add(&second);

        assert_eq!(sum.tracked_center_for_test(), None);
    }

    #[test]
    fn test_mutable_tensor_access_invalidation_guards_canonical_routing() {
        let config = MpsConfig {
            max_bond_dim: 64,
            svd_cutoff: 0.0,
            max_truncation_error: Some(0.0),
            parallel: false,
        };
        let mut mps = seeded_random_mps(5, 4, 0xd1ec_7001, config);
        mps.left_canonicalize();
        assert_eq!(mps.center, Some(4));

        mps.tensors_mut()[0] *= Complex64::new(2.0, 0.0);

        assert_eq!(mps.tracked_center_for_test(), None);
    }

    #[test]
    fn test_stale_center_guard_prevents_gauge_dependent_truncation() {
        let config = MpsConfig {
            max_bond_dim: 2,
            svd_cutoff: 0.0,
            max_truncation_error: Some(0.0),
            parallel: false,
        };
        let mut poisoned = seeded_random_mps(6, 8, 0x57a1_e000_0000_0001, config);
        poisoned.canonicalize_around_bond(2);
        let exact_before = poisoned.state_vector();
        apply_diagonal_bond_gauge(&mut poisoned, 2, &[1e-3, 1e-1, 10.0, 1e3]);
        assert!(normalized_fidelity(&exact_before, &poisoned.state_vector()) > 1.0 - 1e-12);
        poisoned.set_tracked_center_for_test(Some(3));

        #[cfg(debug_assertions)]
        {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                poisoned.canonicalize_around_bond(2);
            }));
            assert!(result.is_err(), "the consult must reject the stale claim");
        }

        #[cfg(not(debug_assertions))]
        {
            let zero = Complex64::new(0.0, 0.0);
            let one = Complex64::new(1.0, 0.0);
            let cnot = DMatrix::from_row_slice(
                4,
                4,
                &[
                    one, zero, zero, zero, zero, one, zero, zero, zero, zero, zero, one, zero,
                    zero, one, zero,
                ],
            );
            let dense_oracle = apply_adjacent_dense_gate(&exact_before, 6, 2, &cnot);

            let mut stale = poisoned.clone();
            stale.canonicalize_around_bond(2);
            stale.apply_two_site_gate(2, &cnot).unwrap();
            assert_eq!(stale.bond_dim(3), 2, "the stale path must truncate");

            let mut correctly_invalidated = poisoned;
            correctly_invalidated.set_tracked_center_for_test(None);
            correctly_invalidated.canonicalize_around_bond(2);
            correctly_invalidated.apply_two_site_gate(2, &cnot).unwrap();
            assert_eq!(
                correctly_invalidated.bond_dim(3),
                2,
                "the guarded path must truncate"
            );

            let stale_fidelity = normalized_fidelity(&dense_oracle, &stale.state_vector());
            let correct_fidelity =
                normalized_fidelity(&dense_oracle, &correctly_invalidated.state_vector());
            eprintln!(
                "release stale-center blast radius: guarded fidelity={correct_fidelity:.16}, stale fidelity={stale_fidelity:.16}"
            );
            assert!(
                1.0 - stale_fidelity > 1e-6,
                "stale truncation unexpectedly matched the dense oracle: {stale_fidelity:.16}"
            );
            assert!(
                correct_fidelity > stale_fidelity + 1e-4,
                "guarded={correct_fidelity:.16}, stale={stale_fidelity:.16}"
            );
        }
    }

    #[test]
    fn test_compress_surfaces_svd_failure() {
        let mut mps = Mps::new(2, MpsConfig::default());
        mps.tensors_mut()[1][(0, 0)] = Complex64::new(f64::NAN, 0.0);

        assert!(matches!(mps.compress(), Err(MpsError::SvdFailed)));
    }

    #[test]
    fn test_compress_sweeps_the_orthogonality_center_with_truncation() {
        let config = MpsConfig {
            max_bond_dim: 3,
            svd_cutoff: 0.0,
            max_truncation_error: Some(0.0),
            parallel: false,
        };
        let mut improved = 0;
        let mut worst_fixed = 1.0_f64;
        let mut worst_legacy = 1.0_f64;

        for seed in 0..40_u64 {
            let original = seeded_random_mps(6, 8, 0xc011_0000_0000_0001 ^ seed, config.clone());
            let exact = original.state_vector();
            let mut fixed = original.clone();
            let mut legacy = original;
            fixed.compress().unwrap();
            legacy_off_center_compress(&mut legacy);
            let fixed_fidelity = normalized_fidelity(&exact, &fixed.state_vector());
            let legacy_fidelity = normalized_fidelity(&exact, &legacy.state_vector());
            worst_fixed = worst_fixed.min(fixed_fidelity);
            worst_legacy = worst_legacy.min(legacy_fidelity);
            if fixed_fidelity > legacy_fidelity + 1e-12 {
                improved += 1;
            }
            assert!(
                fixed_fidelity + 1e-12 >= legacy_fidelity,
                "seed={seed}: fixed={fixed_fidelity:.16}, legacy={legacy_fidelity:.16}"
            );
        }
        eprintln!(
            "compress canonical sweep: improved={improved}/40, worst_fixed={worst_fixed:.16}, worst_legacy={worst_legacy:.16}"
        );
        assert_eq!(improved, 40);
    }

    #[test]
    fn test_long_range_gate_truncation_is_gauge_invariant() {
        let config = MpsConfig {
            max_bond_dim: 4,
            svd_cutoff: 0.0,
            max_truncation_error: Some(0.0),
            parallel: false,
        };
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
        let scales = [1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4];
        let mut worst_fixed_pair = 1.0_f64;
        let mut worst_legacy_pair = 1.0_f64;

        for seed in 0..40_u64 {
            let first = seeded_random_mps(6, 8, 0x6a06_0000_0000_0001 ^ seed, config.clone());
            let mut second = first.clone();
            apply_diagonal_bond_gauge(&mut second, 2, &scales);
            assert!(
                normalized_fidelity(&first.state_vector(), &second.state_vector()) > 1.0 - 1e-12
            );

            let mut fixed_first = first.clone();
            let mut fixed_second = second.clone();
            fixed_first
                .apply_long_range_two_site_gate(0, 5, &cnot)
                .unwrap();
            fixed_second
                .apply_long_range_two_site_gate(0, 5, &cnot)
                .unwrap();
            let fixed_pair =
                normalized_fidelity(&fixed_first.state_vector(), &fixed_second.state_vector());
            worst_fixed_pair = worst_fixed_pair.min(fixed_pair);
            assert!(
                fixed_pair > 1.0 - 1e-12,
                "seed={seed}: gauge-pair fidelity={fixed_pair:.16}"
            );
            assert_relative_eq!(
                fixed_first.truncation_error(),
                fixed_second.truncation_error(),
                epsilon = 1e-12
            );
            assert_eq!(fixed_first.bond_cap_hits(), fixed_second.bond_cap_hits());

            let mut legacy_first = first;
            let mut legacy_second = second;
            legacy_off_center_long_range_gate(&mut legacy_first, 0, 5, &cnot);
            legacy_off_center_long_range_gate(&mut legacy_second, 0, 5, &cnot);
            worst_legacy_pair = worst_legacy_pair.min(normalized_fidelity(
                &legacy_first.state_vector(),
                &legacy_second.state_vector(),
            ));
        }
        eprintln!(
            "long-range gauge pair: worst_fixed={worst_fixed_pair:.16}, worst_legacy={worst_legacy_pair:.16}"
        );
    }

    #[test]
    fn test_long_range_gate_canonical_walk_is_exact_without_truncation() {
        let config = MpsConfig {
            max_bond_dim: 64,
            svd_cutoff: 0.0,
            max_truncation_error: Some(0.0),
            parallel: false,
        };
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
        for seed in 0..16_u64 {
            let original = seeded_random_mps(6, 8, 0xe7ac_0000_0000_0001 ^ seed, config.clone());
            let mut canonical = original.clone();
            let mut reference = original;
            canonical
                .apply_long_range_two_site_gate(0, 5, &cnot)
                .unwrap();
            legacy_off_center_long_range_gate(&mut reference, 0, 5, &cnot);
            let fidelity =
                normalized_fidelity(&canonical.state_vector(), &reference.state_vector());
            assert!(fidelity > 1.0 - 1e-12, "seed={seed}: fidelity={fidelity}");
        }
    }

    #[test]
    fn test_zero_max_truncation_error_disables_adaptive_compression() {
        let mut zero = Mps::new(2, MpsConfig::default());
        let mut small_branch = Mps::new(2, MpsConfig::default());
        let x = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        small_branch.apply_one_site_gate(0, &x).unwrap();
        small_branch.apply_one_site_gate(1, &x).unwrap();
        small_branch.scale(Complex64::new(0.01, 0.0));
        zero = zero.add(&small_branch);
        zero.config.max_bond_dim = 128;
        zero.config.svd_cutoff = 0.0;
        zero.config.max_truncation_error = Some(0.0);

        let mut adaptive = zero.clone();
        adaptive.config.max_truncation_error = Some(1e-3);
        let mut cutoff_limited = zero.clone();
        cutoff_limited.config.svd_cutoff = 0.1;
        let mut cap_limited = zero.clone();
        cap_limited.config.max_bond_dim = 1;

        zero.compress().unwrap();
        adaptive.compress().unwrap();
        cutoff_limited.compress().unwrap();
        cap_limited.compress().unwrap();

        assert_eq!(zero.bond_dim(1), 2, "zero must retain positive weight");
        assert_eq!(adaptive.bond_dim(1), 1, "positive budget may discard it");
        assert_eq!(cutoff_limited.bond_dim(1), 1, "SVD cutoff remains active");
        assert_eq!(cap_limited.bond_dim(1), 1, "bond cap remains active");
    }

    #[test]
    fn test_compress_cutoff_is_invariant_under_global_rescaling() {
        let config = MpsConfig {
            max_bond_dim: 8,
            svd_cutoff: 1e-12,
            max_truncation_error: Some(0.0),
            parallel: false,
        };
        let dominant = Mps::new(6, config.clone());
        let mut small_branch = Mps::new(6, config);
        let x = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        for site in 0..6 {
            small_branch.apply_one_site_gate(site, &x).unwrap();
        }
        small_branch.scale(Complex64::new(1e-10, 0.0));
        let state = dominant.add(&small_branch);

        let mut reference = state.clone();
        let mut scaled_down = state.clone();
        scaled_down.scale(Complex64::new(1e-13, 0.0));
        reference.compress().unwrap();
        scaled_down.compress().unwrap();

        assert_eq!(reference.bond_dims(), scaled_down.bond_dims());
        assert_eq!(reference.bond_dims(), &[1, 2, 2, 2, 2, 2, 1]);
        assert_relative_eq!(
            reference.truncation_error(),
            scaled_down.truncation_error(),
            epsilon = 1e-15
        );
        assert!(
            normalized_fidelity(&reference.state_vector(), &scaled_down.state_vector())
                > 1.0 - 1e-12
        );
    }

    #[test]
    fn test_new_is_all_zeros_state() {
        let mps = Mps::new(3, MpsConfig::default());
        assert_eq!(mps.num_sites(), 3);
        assert_relative_eq!(mps.amplitude(&[0, 0, 0]).re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(mps.amplitude(&[0, 0, 1]).norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(mps.amplitude(&[1, 0, 0]).norm(), 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_norm_of_initial_state() {
        let mps = Mps::new(4, MpsConfig::default());
        assert_relative_eq!(mps.norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_cached_basis_marginals_match_full_contractions_across_bonds_and_scales() {
        fn next(state: &mut u64) -> u64 {
            *state ^= *state << 13;
            *state ^= *state >> 7;
            *state ^= *state << 17;
            *state
        }

        for bond in [2, 4] {
            for seed in 0..16_u64 {
                // Direct sums establish the requested chi; replacing every
                // entry gives a deliberately non-canonical random MPS.
                let product = Mps::new(6, MpsConfig::default());
                let bond_two = product.add(&product);
                let mut base = if bond == 2 {
                    bond_two
                } else {
                    bond_two.add(&bond_two)
                };
                let mut random = 0xeca5_0000_0000_0001_u64 ^ seed ^ (bond as u64) << 32;
                for tensor in base.tensors_mut() {
                    for value in tensor.iter_mut() {
                        let real = (next(&mut random) as f64 / u64::MAX as f64) - 0.5;
                        let imag = (next(&mut random) as f64 / u64::MAX as f64) - 0.5;
                        *value = Complex64::new(real, imag);
                    }
                }
                assert_eq!(base.max_bond_dim(), bond);

                for scale in [1e-8, 1.0, 1e8] {
                    let mut mps = base.clone();
                    mps.scale(Complex64::new(scale, 0.0));
                    let norm_squared = mps.norm_squared();
                    assert!(norm_squared > 1e-20);
                    for sites in [&[0, 1, 2, 3, 4, 5][..], &[4, 1, 5, 2][..], &[3, 0][..]] {
                        let environments = mps.environment_cache(sites);
                        assert_relative_eq!(
                            environments.norm_squared(),
                            norm_squared,
                            max_relative = 1e-14
                        );
                        for &site in sites {
                            let reference = one_site_basis_marginal_reference(&mps, site, 1);
                            let batched = environments.one_site_basis_marginal(site, 1);
                            assert!(
                                (batched - reference).abs() <= 1e-14,
                                "bond={bond}, scale={scale:.1e}, seed={seed}, site={site}, bonds={:?}, batched={batched:.16e}, reference={reference:.16e}",
                                mps.bond_dims(),
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    #[should_panic(expected = "cannot evaluate a normalized marginal of a zero-norm MPS")]
    fn test_zero_norm_cached_marginal_panics() {
        let mut mps = Mps::new(2, MpsConfig::default());
        mps.scale(Complex64::new(0.0, 0.0));
        let environments = mps.environment_cache(&[0]);
        let _ = environments.one_site_basis_marginal(0, 1);
    }

    #[test]
    fn test_single_site_x_gate() {
        let mut mps = Mps::new(2, MpsConfig::default());
        // X gate on site 0: |00> -> |10>
        let x = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        mps.apply_one_site_gate(0, &x).unwrap();
        assert_relative_eq!(mps.amplitude(&[1, 0]).re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(mps.amplitude(&[0, 0]).norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(mps.norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_hadamard_gate() {
        let mut mps = Mps::new(1, MpsConfig::default());
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        );
        mps.apply_one_site_gate(0, &h).unwrap();
        // |+> = (|0> + |1>) / sqrt(2)
        assert_relative_eq!(mps.amplitude(&[0]).re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(mps.amplitude(&[1]).re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(mps.norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_diagonal_gate() {
        let mut mps = Mps::new(1, MpsConfig::default());
        // First apply H to get |+>
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        );
        mps.apply_one_site_gate(0, &h).unwrap();
        // Apply Z = diag(1, -1)
        mps.apply_diagonal_one_site(0, &[Complex64::new(1.0, 0.0), Complex64::new(-1.0, 0.0)])
            .unwrap();
        // Should get |->: (|0> - |1>) / sqrt(2)
        assert_relative_eq!(mps.amplitude(&[0]).re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(mps.amplitude(&[1]).re, -inv_sqrt2, epsilon = 1e-10);
    }

    #[test]
    fn test_cnot_gate() {
        let mut mps = Mps::new(2, MpsConfig::default());
        // Apply X to site 0: |00> -> |10>
        let x = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        mps.apply_one_site_gate(0, &x).unwrap();

        // Apply CNOT (control=0, target=1): |10> -> |11>
        let mut cnot = DMatrix::zeros(4, 4);
        cnot[(0, 0)] = Complex64::new(1.0, 0.0); // |00> -> |00>
        cnot[(1, 1)] = Complex64::new(1.0, 0.0); // |01> -> |01>
        cnot[(3, 2)] = Complex64::new(1.0, 0.0); // |10> -> |11>
        cnot[(2, 3)] = Complex64::new(1.0, 0.0); // |11> -> |10>
        mps.apply_two_site_gate(0, &cnot).unwrap();

        assert_relative_eq!(mps.amplitude(&[1, 1]).re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(mps.amplitude(&[0, 0]).norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(mps.amplitude(&[1, 0]).norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(mps.norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_bell_state() {
        let mut mps = Mps::new(2, MpsConfig::default());
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;

        // H on site 0
        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        );
        mps.apply_one_site_gate(0, &h).unwrap();

        // CNOT
        let mut cnot = DMatrix::zeros(4, 4);
        cnot[(0, 0)] = Complex64::new(1.0, 0.0);
        cnot[(1, 1)] = Complex64::new(1.0, 0.0);
        cnot[(3, 2)] = Complex64::new(1.0, 0.0);
        cnot[(2, 3)] = Complex64::new(1.0, 0.0);
        mps.apply_two_site_gate(0, &cnot).unwrap();

        // Bell state: (|00> + |11>) / sqrt(2)
        assert_relative_eq!(mps.amplitude(&[0, 0]).re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(mps.amplitude(&[1, 1]).re, inv_sqrt2, epsilon = 1e-10);
        assert_relative_eq!(mps.amplitude(&[0, 1]).norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(mps.amplitude(&[1, 0]).norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(mps.norm_squared(), 1.0, epsilon = 1e-10);
        assert_eq!(mps.bond_dim(1), 2); // Bell state needs bond dim 2
    }

    #[test]
    fn test_state_vector() {
        let mut mps = Mps::new(2, MpsConfig::default());
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        );
        mps.apply_one_site_gate(0, &h).unwrap();
        let sv = mps.state_vector();
        // |+0> = (|00> + |10>) / sqrt(2)
        assert_eq!(sv.len(), 4);
        assert_relative_eq!(sv[0].re, inv_sqrt2, epsilon = 1e-10); // |00>
        assert_relative_eq!(sv[1].norm(), 0.0, epsilon = 1e-10); // |01>
        assert_relative_eq!(sv[2].re, inv_sqrt2, epsilon = 1e-10); // |10>
        assert_relative_eq!(sv[3].norm(), 0.0, epsilon = 1e-10); // |11>
    }

    #[test]
    fn test_scale() {
        let mut mps = Mps::new(2, MpsConfig::default());
        mps.scale(Complex64::new(0.0, 1.0)); // multiply by i
        assert_relative_eq!(mps.amplitude(&[0, 0]).im, 1.0, epsilon = 1e-10);
        assert_relative_eq!(mps.norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mps_add() {
        // |00> + |11> (unnormalized)
        let mps0 = Mps::new(2, MpsConfig::default()); // |00>

        let mut mps1 = Mps::new(2, MpsConfig::default());
        let x = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        mps1.apply_one_site_gate(0, &x).unwrap();
        mps1.apply_one_site_gate(1, &x).unwrap();
        // mps1 = |11>

        let sum = mps0.add(&mps1);
        // Should be |00> + |11>
        assert_relative_eq!(sum.amplitude(&[0, 0]).re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(sum.amplitude(&[1, 1]).re, 1.0, epsilon = 1e-10);
        assert_relative_eq!(sum.amplitude(&[0, 1]).norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(sum.amplitude(&[1, 0]).norm(), 0.0, epsilon = 1e-10);
        assert_relative_eq!(sum.norm_squared(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_two_site_gate_preserves_norm() {
        // Build an entangled 4-qubit MPS, then apply a two-site gate.
        // The norm should be preserved.
        let mut mps = Mps::new(4, MpsConfig::default());

        // Create entanglement: H(0), CNOT(0,1), H(2), CNOT(2,3)
        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
            ],
        );
        let cnot = DMatrix::from_row_slice(
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
        let swap = DMatrix::from_row_slice(
            4,
            4,
            &[
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
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
            ],
        );

        mps.apply_one_site_gate(0, &h).unwrap();
        mps.apply_two_site_gate(0, &cnot).unwrap();
        mps.apply_one_site_gate(2, &h).unwrap();
        mps.apply_two_site_gate(2, &cnot).unwrap();

        let norm_before = mps.norm_squared();
        assert_relative_eq!(norm_before, 1.0, epsilon = 1e-10);

        // Apply various two-site gates and check norm
        mps.apply_two_site_gate(1, &cnot).unwrap();
        assert_relative_eq!(mps.norm_squared(), 1.0, epsilon = 1e-10); // "CNOT on (1,2)");

        mps.apply_two_site_gate(0, &swap).unwrap();
        assert_relative_eq!(mps.norm_squared(), 1.0, epsilon = 1e-10); // "SWAP on (0,1)");

        // Long-range CNOT via SWAP chain
        mps.apply_long_range_two_site_gate(0, 3, &cnot).unwrap();
        assert_relative_eq!(mps.norm_squared(), 1.0, epsilon = 1e-10); // "Long-range CNOT(0,3)");

        mps.apply_long_range_two_site_gate(0, 2, &swap).unwrap();
        assert_relative_eq!(mps.norm_squared(), 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_long_range_cnot_state_vector() {
        // Apply CNOT(0, 2) to H(0)|000⟩ via the MPO approach
        // and compare to building the exact state with adjacent gates.
        let c0 = Complex64::new(0.0, 0.0);
        let c1 = Complex64::new(1.0, 0.0);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        );
        let cnot = DMatrix::from_row_slice(
            4,
            4,
            &[
                c1, c0, c0, c0, c0, c1, c0, c0, c0, c0, c0, c1, c0, c0, c1, c0,
            ],
        );

        // Method 1: long-range CNOT(0, 2) via MPO
        let mut mps1 = Mps::new(3, MpsConfig::default());
        mps1.apply_one_site_gate(0, &h).unwrap();
        mps1.apply_long_range_two_site_gate(0, 2, &cnot).unwrap();
        let sv1 = mps1.state_vector();

        // Method 2: build exact state manually
        // H(0)|000⟩ = (|000⟩ + |100⟩) / sqrt(2)
        // CNOT(0,2)(|000⟩ + |100⟩)/sqrt(2) = (|000⟩ + |101⟩)/sqrt(2)
        // State vector ordering: MSB-first, so |000⟩ = idx 0, |101⟩ = idx 5
        assert_relative_eq!(sv1[0].re, inv_sqrt2, epsilon = 1e-8);
        assert_relative_eq!(sv1[5].re, inv_sqrt2, epsilon = 1e-8);
        for (i, amp) in sv1.iter().enumerate().take(8) {
            if i != 0 && i != 5 {
                assert_relative_eq!(amp.norm(), 0.0, epsilon = 1e-8);
            }
        }
    }

    #[test]
    fn test_long_range_cnot_entangled() {
        // Apply CNOT(0, 3) on a 4-qubit state that's already entangled.
        // Compare MPO approach to building reference via adjacent gates only.
        let c0 = Complex64::new(0.0, 0.0);
        let c1 = Complex64::new(1.0, 0.0);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        );
        let cnot = DMatrix::from_row_slice(
            4,
            4,
            &[
                c1, c0, c0, c0, c0, c1, c0, c0, c0, c0, c0, c1, c0, c0, c1, c0,
            ],
        );
        let swap = DMatrix::from_row_slice(
            4,
            4,
            &[
                c1, c0, c0, c0, c0, c0, c1, c0, c0, c1, c0, c0, c0, c0, c0, c1,
            ],
        );

        // Build entangled state: H(0), CNOT(0,1), H(2), CNOT(2,3)
        // Then apply CNOT(0, 3) via MPO
        let mut mps_mpo = Mps::new(4, MpsConfig::default());
        mps_mpo.apply_one_site_gate(0, &h).unwrap();
        mps_mpo.apply_two_site_gate(0, &cnot).unwrap();
        mps_mpo.apply_one_site_gate(2, &h).unwrap();
        mps_mpo.apply_two_site_gate(2, &cnot).unwrap();
        mps_mpo.apply_long_range_two_site_gate(0, 3, &cnot).unwrap();
        let sv_mpo = mps_mpo.state_vector();

        // Reference: same state, CNOT(0, 3) via manual SWAP chain
        let mut mps_ref = Mps::new(4, MpsConfig::default());
        mps_ref.apply_one_site_gate(0, &h).unwrap();
        mps_ref.apply_two_site_gate(0, &cnot).unwrap();
        mps_ref.apply_one_site_gate(2, &h).unwrap();
        mps_ref.apply_two_site_gate(2, &cnot).unwrap();
        // Manual SWAP chain for CNOT(0, 3)
        mps_ref.apply_two_site_gate(2, &swap).unwrap(); // SWAP(2,3)
        mps_ref.apply_two_site_gate(1, &swap).unwrap(); // SWAP(1,2)
        mps_ref.apply_two_site_gate(0, &cnot).unwrap(); // CNOT(0,1) [was q3]
        mps_ref.apply_two_site_gate(1, &swap).unwrap(); // SWAP back
        mps_ref.apply_two_site_gate(2, &swap).unwrap(); // SWAP back
        let sv_ref = mps_ref.state_vector();

        // Check overlap
        let overlap: Complex64 = sv_mpo
            .iter()
            .zip(sv_ref.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        assert_relative_eq!(overlap.norm_sqr(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_long_range_cnot_hi_ctrl() {
        // Test with high-qubit control CNOT (target < control)
        let c0 = Complex64::new(0.0, 0.0);
        let c1 = Complex64::new(1.0, 0.0);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        );
        // CNOT with hi-index qubit as control
        let cnot_hi = DMatrix::from_row_slice(
            4,
            4,
            &[
                c1, c0, c0, c0, c0, c0, c0, c1, c0, c0, c1, c0, c0, c1, c0, c0,
            ],
        );
        let swap = DMatrix::from_row_slice(
            4,
            4,
            &[
                c1, c0, c0, c0, c0, c0, c1, c0, c0, c1, c0, c0, c0, c0, c0, c1,
            ],
        );

        // H(2), CNOT_hi(0, 2) on 3-qubit MPS
        // CNOT_hi: control=qubit 2, target=qubit 0
        let mut mps_mpo = Mps::new(3, MpsConfig::default());
        mps_mpo.apply_one_site_gate(2, &h).unwrap();
        mps_mpo
            .apply_long_range_two_site_gate(0, 2, &cnot_hi)
            .unwrap();
        let sv_mpo = mps_mpo.state_vector();

        // Reference via SWAP chain
        let mut mps_ref = Mps::new(3, MpsConfig::default());
        mps_ref.apply_one_site_gate(2, &h).unwrap();
        mps_ref.apply_two_site_gate(1, &swap).unwrap();
        mps_ref.apply_two_site_gate(0, &cnot_hi).unwrap();
        mps_ref.apply_two_site_gate(1, &swap).unwrap();
        let sv_ref = mps_ref.state_vector();

        let overlap: Complex64 = sv_mpo
            .iter()
            .zip(sv_ref.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        assert_relative_eq!(overlap.norm_sqr(), 1.0, epsilon = 1e-6);
    }

    #[test]
    fn test_long_range_cnot_cascade() {
        // Test the pattern from non_clifford.rs: multiple long-range CNOTs
        let c0 = Complex64::new(0.0, 0.0);
        let c1 = Complex64::new(1.0, 0.0);
        let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(inv_sqrt2, 0.0),
                Complex64::new(-inv_sqrt2, 0.0),
            ],
        );
        let cnot_lo = DMatrix::from_row_slice(
            4,
            4,
            &[
                c1, c0, c0, c0, c0, c1, c0, c0, c0, c0, c0, c1, c0, c0, c1, c0,
            ],
        );
        let rx_gate = {
            let theta = 0.5_f64;
            let c = Complex64::new(theta.cos(), 0.0);
            let s = Complex64::new(0.0, -theta.sin());
            DMatrix::from_row_slice(2, 2, &[c, s, s, c])
        };

        // H on all, then CNOT cascade (0→1, 0→3), RX(0), reverse CNOT
        let mut mps_mpo = Mps::new(4, MpsConfig::default());
        for q in 0..4 {
            mps_mpo.apply_one_site_gate(q, &h).unwrap();
        }
        mps_mpo.apply_two_site_gate(0, &cnot_lo).unwrap();
        mps_mpo
            .apply_long_range_two_site_gate(0, 3, &cnot_lo)
            .unwrap();
        mps_mpo.apply_one_site_gate(0, &rx_gate).unwrap();
        mps_mpo
            .apply_long_range_two_site_gate(0, 3, &cnot_lo)
            .unwrap();
        mps_mpo.apply_two_site_gate(0, &cnot_lo).unwrap();
        let sv_mpo = mps_mpo.state_vector();

        // Reference: same but use SWAP chains for long-range
        let swap = DMatrix::from_row_slice(
            4,
            4,
            &[
                c1, c0, c0, c0, c0, c0, c1, c0, c0, c1, c0, c0, c0, c0, c0, c1,
            ],
        );
        let mut mps_ref = Mps::new(4, MpsConfig::default());
        for q in 0..4 {
            mps_ref.apply_one_site_gate(q, &h).unwrap();
        }
        mps_ref.apply_two_site_gate(0, &cnot_lo).unwrap();
        // SWAP chain for CNOT(0,3)
        mps_ref.apply_two_site_gate(2, &swap).unwrap();
        mps_ref.apply_two_site_gate(1, &swap).unwrap();
        mps_ref.apply_two_site_gate(0, &cnot_lo).unwrap();
        mps_ref.apply_two_site_gate(1, &swap).unwrap();
        mps_ref.apply_two_site_gate(2, &swap).unwrap();
        mps_ref.apply_one_site_gate(0, &rx_gate).unwrap();
        // SWAP chain for CNOT(0,3) again
        mps_ref.apply_two_site_gate(2, &swap).unwrap();
        mps_ref.apply_two_site_gate(1, &swap).unwrap();
        mps_ref.apply_two_site_gate(0, &cnot_lo).unwrap();
        mps_ref.apply_two_site_gate(1, &swap).unwrap();
        mps_ref.apply_two_site_gate(2, &swap).unwrap();
        mps_ref.apply_two_site_gate(0, &cnot_lo).unwrap();
        let sv_ref = mps_ref.state_vector();

        let overlap: Complex64 = sv_mpo
            .iter()
            .zip(sv_ref.iter())
            .map(|(a, b)| a.conj() * b)
            .sum();
        assert_relative_eq!(overlap.norm_sqr(), 1.0, epsilon = 1e-4);
    }

    #[test]
    fn test_multi_site_rotation_preserves_norm() {
        // Reproduce the Stabilizer multi-site rotation:
        // H(0), H(2), CNOT(0,2), RX(0), CNOT(0,2), H(0), H(2)
        let mut mps = Mps::new(4, MpsConfig::default());

        let h = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0),
                Complex64::new(-std::f64::consts::FRAC_1_SQRT_2, 0.0),
            ],
        );
        let cnot = DMatrix::from_row_slice(
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
        let rx = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(0.9239, 0.0),
                Complex64::new(0.0, -0.3827),
                Complex64::new(0.0, -0.3827),
                Complex64::new(0.9239, 0.0),
            ],
        );

        // Build entangled state
        mps.apply_one_site_gate(0, &h).unwrap();
        mps.apply_two_site_gate(0, &cnot).unwrap();
        mps.apply_one_site_gate(2, &h).unwrap();
        mps.apply_two_site_gate(2, &cnot).unwrap();
        assert_relative_eq!(mps.norm_squared(), 1.0, epsilon = 1e-10);

        // Multi-site Z rotation on sites {0, 2}
        mps.apply_one_site_gate(0, &h).unwrap();
        mps.apply_one_site_gate(2, &h).unwrap();
        mps.apply_long_range_two_site_gate(0, 2, &cnot).unwrap();
        let norm_mid = mps.norm_squared();
        mps.apply_one_site_gate(0, &rx).unwrap();
        mps.apply_long_range_two_site_gate(0, 2, &cnot).unwrap();
        mps.apply_one_site_gate(0, &h).unwrap();
        mps.apply_one_site_gate(2, &h).unwrap();

        eprintln!(
            "norm mid-cascade: {norm_mid:.10}, after: {:.10}",
            mps.norm_squared()
        );
        assert_relative_eq!(mps.norm_squared(), 1.0, epsilon = 1e-3);
    }
}
