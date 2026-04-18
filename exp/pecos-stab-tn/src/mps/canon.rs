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

//! MPS canonicalization via QR decomposition.
//!
//! Left-canonical form: each site tensor A[i] satisfies `sum_sigma` A[sigma]^dagger A[sigma] = I.
//! Right-canonical form: each site tensor B[i] satisfies `sum_sigma` B[sigma] B[sigma]^dagger = I.

use super::tensor::{reshape_left_group, reshape_left_ungroup};
use nalgebra::DMatrix;
use num_complex::Complex64;

/// Left-canonicalize a single site by QR decomposition.
///
/// Takes the site tensor at position `q` in `(chi_l, d * chi_r)` format,
/// reshapes to `(chi_l * d, chi_r)`, performs thin QR, stores Q back as the
/// new site tensor, and absorbs R into the next site's tensor.
///
/// Returns the new bond dimension between sites q and q+1.
///
/// # Panics
///
/// Panics if `q >= tensors.len() - 1` (cannot left-canonicalize the last site).
pub fn left_canonicalize_site(
    tensors: &mut [DMatrix<Complex64>],
    bond_dims: &mut [usize],
    q: usize,
    d: usize,
) -> usize {
    let num_sites = tensors.len();
    assert!(q < num_sites - 1, "cannot left-canonicalize the last site");

    let chi_l = bond_dims[q];
    let chi_r = bond_dims[q + 1];

    // Reshape to (chi_l * d, chi_r) for QR
    let grouped = reshape_left_group(&tensors[q], chi_l, d, chi_r);
    let qr = grouped.qr();
    let q_mat = qr.q();
    let r_mat = qr.r();

    // New bond dimension = min(chi_l * d, chi_r) -- rank of R
    let new_chi = q_mat.ncols();
    bond_dims[q + 1] = new_chi;

    // Store Q as new site tensor: reshape (chi_l * d, new_chi) -> (chi_l, d * new_chi)
    tensors[q] = reshape_left_ungroup(&q_mat, chi_l, d, new_chi);

    // Absorb R into next site: new_next = R * old_next
    // R: (new_chi, chi_r), next tensor: (chi_r, d * chi_r_next) -> needs reshape
    let next = &tensors[q + 1];
    // Reshape next to (chi_r, d * chi_r_next) -- it already is in this form
    // but chi_r might have been the old bond dim. The matrix R has chi_r columns.
    // next has chi_r rows, d * chi_r_next columns.
    let absorbed = &r_mat * next;
    tensors[q + 1] = absorbed;

    new_chi
}

/// Right-canonicalize a single site by LQ decomposition.
///
/// Takes the site tensor at position `q`, reshapes to `(chi_l, d * chi_r)`,
/// performs LQ (via QR of transpose), stores Q back as the site tensor,
/// and absorbs L into the previous site's tensor.
///
/// Returns the new bond dimension between sites q-1 and q.
///
/// # Panics
///
/// Panics if `q == 0` (cannot right-canonicalize the first site).
pub fn right_canonicalize_site(
    tensors: &mut [DMatrix<Complex64>],
    bond_dims: &mut [usize],
    q: usize,
    d: usize,
) -> usize {
    assert!(q > 0, "cannot right-canonicalize the first site");

    let chi_l = bond_dims[q];

    // LQ decomposition via QR of transpose: A^T = Q R -> A = R^T Q^T = L Q
    let at = tensors[q].transpose();
    let qr = at.qr();
    // Q^T gives us the right factor, R^T gives us the left factor
    let q_mat_t = qr.q().transpose(); // shape: (new_chi, d * chi_r) -- but need to verify
    let l_mat = qr.r().transpose(); // shape: (chi_l, new_chi)

    let new_chi = q_mat_t.nrows();
    bond_dims[q] = new_chi;

    // Store Q^T as the new site tensor: (new_chi, d * chi_r)
    // We need to reshape this back to proper site tensor format
    tensors[q] = q_mat_t;

    // Absorb L into previous site
    // Previous site: (chi_l_prev, d * chi_l) -- the last chi_l columns per physical block
    // L: (chi_l, new_chi)
    // We need: new_prev[alpha_l_prev, sigma * new_chi + alpha_new] =
    //   sum_{alpha_l} prev[alpha_l_prev, sigma * chi_l + alpha_l] * L[alpha_l, alpha_new]
    let chi_l_prev = bond_dims[q - 1];
    let prev = &tensors[q - 1];
    let mut new_prev = DMatrix::zeros(chi_l_prev, d * new_chi);
    for sigma in 0..d {
        let prev_block = prev.columns(sigma * chi_l, chi_l).clone_owned();
        let absorbed_block = &prev_block * &l_mat;
        for i in 0..chi_l_prev {
            for j in 0..new_chi {
                new_prev[(i, sigma * new_chi + j)] = absorbed_block[(i, j)];
            }
        }
    }
    tensors[q - 1] = new_prev;

    new_chi
}

/// Put the entire MPS in left-canonical form by sweeping left to right.
pub fn left_canonicalize_all(
    tensors: &mut [DMatrix<Complex64>],
    bond_dims: &mut [usize],
    d: usize,
) {
    let n = tensors.len();
    for q in 0..n - 1 {
        left_canonicalize_site(tensors, bond_dims, q, d);
    }
}

/// Put the entire MPS in right-canonical form by sweeping right to left.
pub fn right_canonicalize_all(
    tensors: &mut [DMatrix<Complex64>],
    bond_dims: &mut [usize],
    d: usize,
) {
    let n = tensors.len();
    for q in (1..n).rev() {
        right_canonicalize_site(tensors, bond_dims, q, d);
    }
}
