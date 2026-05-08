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

/// Configuration for MPS truncation.
#[derive(Clone, Debug)]
pub struct MpsConfig {
    /// Maximum bond dimension (hard cap). Singular values beyond this are discarded.
    pub max_bond_dim: usize,
    /// Minimum singular value to keep (absolute cutoff).
    pub svd_cutoff: f64,
    /// Maximum relative truncation error per SVD.
    /// When set, singular values are kept until the discarded weight
    /// (sum of discarded `s_i^2` / sum of all `s_i^2`) exceeds this threshold.
    /// This allows low-entanglement bonds to use small chi (fast) while
    /// high-entanglement bonds grow up to `max_bond_dim` (accurate).
    /// None = disabled (fixed `max_bond_dim` only).
    pub max_truncation_error: Option<f64>,
    /// Use rayon for parallelizing independent MPS operations.
    pub parallel: bool,
}

impl Default for MpsConfig {
    fn default() -> Self {
        Self {
            max_bond_dim: 64,
            svd_cutoff: 1e-12,
            max_truncation_error: None,
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
            bond_dims,
            config,
            truncation_error: 0.0,
            bond_cap_hits: 0,
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

    /// Reset truncation diagnostics (keep state).
    pub fn reset_truncation_stats(&mut self) {
        self.truncation_error = 0.0;
        self.bond_cap_hits = 0;
    }

    /// Record the outcome of one truncated SVD for telemetry.
    pub(crate) fn record_truncation(&mut self, discarded_weight: f64, hit_cap: bool) {
        if discarded_weight > 0.0 {
            self.truncation_error += (1.0 - self.truncation_error) * discarded_weight;
        }
        if hit_cap {
            self.bond_cap_hits += 1;
        }
    }

    #[must_use]
    pub fn num_sites(&self) -> usize {
        self.num_sites
    }

    #[must_use]
    pub fn phys_dim(&self) -> usize {
        self.phys_dim
    }

    /// Bond dimension at bond `i` (between sites `i-1` and `i`).
    #[must_use]
    pub fn bond_dim(&self, bond: usize) -> usize {
        self.bond_dims[bond]
    }

    #[must_use]
    pub fn max_bond_dim(&self) -> usize {
        *self.bond_dims.iter().max().unwrap_or(&1)
    }

    #[must_use]
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

    /// Multiply the entire MPS by a scalar (absorbed into the first tensor).
    pub fn scale(&mut self, scalar: Complex64) {
        if self.tensors.is_empty() {
            return;
        }
        self.tensors[0] *= scalar;
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

        let chi_r = self.bond_dims[q + 1];
        for (sigma, &c) in coeffs.iter().enumerate() {
            let start_col = sigma * chi_r;
            for j in 0..chi_r {
                for i in 0..self.bond_dims[q] {
                    self.tensors[q][(i, start_col + j)] *= c;
                }
            }
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

        // SVD split with truncation
        let (u_s, vt, disc, hit) = svd::truncated_svd_left_absorb_with_error(
            &svd_matrix,
            self.config.max_bond_dim,
            self.config.svd_cutoff,
            self.config.max_truncation_error,
        )?;
        self.record_truncation(disc, hit);

        let new_chi = u_s.ncols();

        // U_S: (chi_l * d, new_chi) -> reshape to (chi_l, d * new_chi)
        self.tensors[q] = reshape_left_ungroup(&u_s, chi_l, d, new_chi);

        // Vt: (new_chi, d * chi_r) -- already in site tensor format
        self.tensors[q + 1] = vt;

        // Update bond dimension
        self.bond_dims[q + 1] = new_chi;

        Ok(())
    }

    /// Apply a two-site gate between arbitrary (possibly non-adjacent) sites.
    ///
    /// Uses SWAP gates to bring site `q1` adjacent to `q0`, applies the gate,
    /// then SWAPs back. `q0 < q1` required.
    ///
    /// SWAP gates are unitary permutations that preserve the Schmidt spectrum,
    /// so SVD truncation after each SWAP introduces minimal numerical drift.
    /// The dominant error comes only from the actual gate application.
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

        // Adjacent case: apply directly
        if q1 == q0 + 1 {
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

        // SWAP q1 leftward until it's adjacent to q0
        for i in (q0 + 1..q1).rev() {
            self.apply_two_site_gate(i, &swap)?;
        }

        // Apply the gate on the now-adjacent pair
        self.apply_two_site_gate(q0, gate)?;

        // SWAP back
        for i in q0 + 1..q1 {
            self.apply_two_site_gate(i, &swap)?;
        }

        Ok(())
    }

    /// Compute the squared norm `<psi|psi>` by contracting the MPS with itself.
    #[must_use]
    pub fn norm_squared(&self) -> f64 {
        // Contract from left to right, building the transfer matrix product.
        // E[alpha, beta] = sum_{sigma} A*[alpha, sigma] A[beta, sigma]
        // Start with E = 1x1 identity.
        let d = self.phys_dim;
        let mut transfer = DMatrix::from_element(1, 1, Complex64::new(1.0, 0.0));

        for q in 0..self.num_sites {
            let chi_r = self.bond_dims[q + 1];
            let t = &self.tensors[q];

            // new_transfer[alpha_r, beta_r] = sum_{alpha_l, beta_l, sigma}
            //   transfer[alpha_l, beta_l] * conj(A[alpha_l, sigma, alpha_r]) * A[beta_l, sigma, beta_r]
            let mut new_transfer = DMatrix::zeros(chi_r, chi_r);
            for sigma in 0..d {
                // block_sigma: (chi_l, chi_r)
                let block = phys_block(t, sigma, chi_r);
                // conj(block)^T * transfer * block
                let conj_block_t = block.conjugate().transpose();
                let tmp = &conj_block_t * &transfer * &block;
                new_transfer += tmp;
            }
            transfer = new_transfer;
        }

        // Final transfer is 1x1
        transfer[(0, 0)].re
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
        let norm_sq = self.norm_squared();
        if norm_sq > 0.0 {
            let inv_norm = Complex64::new(1.0 / norm_sq.sqrt(), 0.0);
            self.tensors[0] *= inv_norm;
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

    /// Compute the full state vector (2^N complex amplitudes).
    ///
    /// Only for testing on small systems.
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

        Self {
            num_sites: n,
            phys_dim: d,
            tensors: new_tensors,
            bond_dims: new_bond_dims,
            config: self.config.clone(),
            truncation_error: self.truncation_error.max(other.truncation_error),
            bond_cap_hits: self.bond_cap_hits + other.bond_cap_hits,
        }
    }

    /// Access the internal tensors.
    #[must_use]
    pub fn tensors(&self) -> &[DMatrix<Complex64>] {
        &self.tensors
    }

    /// Mutable access to the internal tensors.
    pub fn tensors_mut(&mut self) -> &mut [DMatrix<Complex64>] {
        &mut self.tensors
    }

    /// Access the bond dimensions (for testing).
    #[must_use]
    pub fn bond_dims(&self) -> &[usize] {
        &self.bond_dims
    }

    /// Left-canonicalize the entire MPS.
    pub fn left_canonicalize(&mut self) {
        canon::left_canonicalize_all(&mut self.tensors, &mut self.bond_dims, self.phys_dim);
    }

    /// Right-canonicalize the entire MPS.
    pub fn right_canonicalize(&mut self) {
        canon::right_canonicalize_all(&mut self.tensors, &mut self.bond_dims, self.phys_dim);
    }

    /// Compress the MPS by SVD truncation at each bond.
    ///
    /// Left-canonicalizes first, then sweeps right-to-left performing SVD
    /// truncation at each bond to enforce `max_bond_dim` and `svd_cutoff`.
    pub fn compress(&mut self) {
        if self.num_sites <= 1 {
            return;
        }

        // Left-canonicalize
        self.left_canonicalize();

        // Sweep right to left: at each bond, reshape the site tensor into
        // (chi_l * d, chi_r), do truncated SVD, absorb U*S into left neighbor.
        let d = self.phys_dim;
        for q in (1..self.num_sites).rev() {
            let chi_l = self.bond_dims[q];

            // Reshape site q from (chi_l, d * chi_r) to (chi_l, d * chi_r) -- already in this form.
            // But we want to split the left bond, so transpose the grouping:
            // Reshape to (chi_l, d * chi_r) and do SVD to split as (chi_l, new_chi) * (new_chi, d * chi_r).
            let matrix = &self.tensors[q];
            if let Ok((u, svt, disc, hit)) = svd::truncated_svd_right_absorb_with_error(
                matrix,
                self.config.max_bond_dim,
                self.config.svd_cutoff,
                self.config.max_truncation_error,
            ) {
                self.record_truncation(disc, hit);
                let new_chi = u.ncols();
                if new_chi < chi_l {
                    // U: (chi_l, new_chi) -- absorb into left neighbor
                    // SVt: (new_chi, d * chi_r) -- new site q tensor
                    self.tensors[q] = svt;
                    self.bond_dims[q] = new_chi;

                    // Absorb U into tensors[q-1]: multiply each physical block by U
                    let chi_l_prev = self.bond_dims[q - 1];
                    let old_chi_r_prev = chi_l; // was bond_dims[q] before update
                    let mut new_prev = DMatrix::zeros(chi_l_prev, d * new_chi);
                    for sigma in 0..d {
                        let prev_block =
                            tensor::phys_block(&self.tensors[q - 1], sigma, old_chi_r_prev);
                        let absorbed = &prev_block * &u;
                        for i in 0..chi_l_prev {
                            for j in 0..new_chi {
                                new_prev[(i, sigma * new_chi + j)] = absorbed[(i, j)];
                            }
                        }
                    }
                    self.tensors[q - 1] = new_prev;
                }
            }
        }
    }
}

impl Clone for Mps {
    fn clone(&self) -> Self {
        Self {
            num_sites: self.num_sites,
            phys_dim: self.phys_dim,
            tensors: self.tensors.clone(),
            bond_dims: self.bond_dims.clone(),
            config: self.config.clone(),
            truncation_error: self.truncation_error,
            bond_cap_hits: self.bond_cap_hits,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

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
