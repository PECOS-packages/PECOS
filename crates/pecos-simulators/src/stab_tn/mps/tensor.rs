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

//! Tensor reshape and contraction utilities for MPS site tensors.
//!
//! Site tensors are stored as `DMatrix<Complex64>` with shape `(chi_l, d * chi_r)`.
//! The physical index `sigma in {0, ..., d-1}` selects a column block:
//! columns `[sigma * chi_r .. (sigma+1) * chi_r]`.

use nalgebra::DMatrix;
use num_complex::Complex64;

/// Extract the column block for physical index `sigma` from a site tensor.
///
/// Site tensor has shape `(chi_l, d * chi_r)`. Returns a view of columns
/// `[sigma * chi_r .. (sigma+1) * chi_r]`, i.e. shape `(chi_l, chi_r)`.
#[must_use]
pub fn phys_block(tensor: &DMatrix<Complex64>, sigma: usize, chi_r: usize) -> DMatrix<Complex64> {
    let start_col = sigma * chi_r;
    tensor.columns(start_col, chi_r).clone_owned()
}

/// Set the column block for physical index `sigma` in a site tensor.
pub fn set_phys_block(
    tensor: &mut DMatrix<Complex64>,
    sigma: usize,
    chi_r: usize,
    block: &DMatrix<Complex64>,
) {
    let start_col = sigma * chi_r;
    for j in 0..chi_r {
        for i in 0..tensor.nrows() {
            tensor[(i, start_col + j)] = block[(i, j)];
        }
    }
}

/// Reshape a site tensor from `(chi_l, d * chi_r)` to `(chi_l * d, chi_r)`.
///
/// This puts the tensor in "left-grouped" form suitable for SVD when splitting
/// the bond to the right.
#[must_use]
pub fn reshape_left_group(
    tensor: &DMatrix<Complex64>,
    chi_l: usize,
    d: usize,
    chi_r: usize,
) -> DMatrix<Complex64> {
    // Input: (chi_l, d * chi_r), stored as T[alpha_l, sigma * chi_r + alpha_r]
    // Output: (chi_l * d, chi_r), stored as M[alpha_l * d + sigma, alpha_r]
    let mut out = DMatrix::zeros(chi_l * d, chi_r);
    for alpha_l in 0..chi_l {
        for sigma in 0..d {
            for alpha_r in 0..chi_r {
                out[(alpha_l * d + sigma, alpha_r)] = tensor[(alpha_l, sigma * chi_r + alpha_r)];
            }
        }
    }
    out
}

/// Reshape from `(chi_l * d, chi_r)` back to `(chi_l, d * chi_r)`.
#[must_use]
pub fn reshape_left_ungroup(
    matrix: &DMatrix<Complex64>,
    chi_l: usize,
    d: usize,
    chi_r: usize,
) -> DMatrix<Complex64> {
    let mut out = DMatrix::zeros(chi_l, d * chi_r);
    for alpha_l in 0..chi_l {
        for sigma in 0..d {
            for alpha_r in 0..chi_r {
                out[(alpha_l, sigma * chi_r + alpha_r)] = matrix[(alpha_l * d + sigma, alpha_r)];
            }
        }
    }
    out
}

/// Contract two adjacent site tensors into a combined two-site tensor.
///
/// Left tensor: `(chi_l, d * chi_mid)`, right tensor: `(chi_mid, d * chi_r)`.
/// Result: `(chi_l, d * d * chi_r)` -- a "two-site" tensor with two physical indices.
///
/// Layout of result: `T[alpha_l, sigma_l * d * chi_r + sigma_r * chi_r + alpha_r]`
#[must_use]
pub fn contract_two_sites(
    left: &DMatrix<Complex64>,
    chi_l: usize,
    chi_mid: usize,
    right: &DMatrix<Complex64>,
    chi_r: usize,
    d: usize,
) -> DMatrix<Complex64> {
    let mut out = DMatrix::zeros(chi_l, d * d * chi_r);
    for sigma_l in 0..d {
        // left_block: (chi_l, chi_mid) for physical index sigma_l
        let left_block = phys_block(left, sigma_l, chi_mid);
        for sigma_r in 0..d {
            // right_block: (chi_mid, chi_r) for physical index sigma_r
            let right_block = phys_block(right, sigma_r, chi_r);
            // contracted: (chi_l, chi_r) = left_block * right_block
            let contracted = &left_block * &right_block;
            // Place into output at combined physical index (sigma_l, sigma_r)
            let out_col_start = (sigma_l * d + sigma_r) * chi_r;
            for alpha_l in 0..chi_l {
                for alpha_r in 0..chi_r {
                    out[(alpha_l, out_col_start + alpha_r)] = contracted[(alpha_l, alpha_r)];
                }
            }
        }
    }
    out
}

/// Reshape a two-site tensor `(chi_l, d * d * chi_r)` into a matrix
/// `(chi_l * d, d * chi_r)` suitable for SVD splitting.
///
/// Groups the left physical index with `chi_l` and right physical index with `chi_r`.
#[must_use]
pub fn reshape_two_site_for_svd(
    tensor: &DMatrix<Complex64>,
    chi_l: usize,
    chi_r: usize,
    d: usize,
) -> DMatrix<Complex64> {
    let mut out = DMatrix::zeros(chi_l * d, d * chi_r);
    for alpha_l in 0..chi_l {
        for sigma_l in 0..d {
            for sigma_r in 0..d {
                for alpha_r in 0..chi_r {
                    let in_col = (sigma_l * d + sigma_r) * chi_r + alpha_r;
                    let out_row = alpha_l * d + sigma_l;
                    let out_col = sigma_r * chi_r + alpha_r;
                    out[(out_row, out_col)] = tensor[(alpha_l, in_col)];
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_phys_block_roundtrip() {
        // Create a 2x4 tensor (chi_l=2, d=2, chi_r=2)
        let t = DMatrix::from_row_slice(
            2,
            4,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(7.0, 0.0),
                Complex64::new(8.0, 0.0),
            ],
        );
        let b0 = phys_block(&t, 0, 2);
        let b1 = phys_block(&t, 1, 2);
        assert_eq!(b0[(0, 0)], Complex64::new(1.0, 0.0));
        assert_eq!(b0[(0, 1)], Complex64::new(2.0, 0.0));
        assert_eq!(b1[(0, 0)], Complex64::new(3.0, 0.0));
        assert_eq!(b1[(1, 1)], Complex64::new(8.0, 0.0));
    }

    #[test]
    fn test_reshape_roundtrip() {
        let t = DMatrix::from_fn(3, 4, |i, j| {
            Complex64::new(f64::from(u32::try_from(i * 4 + j).unwrap()), 0.0)
        });
        let grouped = reshape_left_group(&t, 3, 2, 2);
        assert_eq!(grouped.nrows(), 6);
        assert_eq!(grouped.ncols(), 2);
        let ungrouped = reshape_left_ungroup(&grouped, 3, 2, 2);
        assert_eq!(ungrouped, t);
    }
}
