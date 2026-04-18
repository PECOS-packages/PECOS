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

//! Truncated SVD for MPS bond compression.
//!
//! Provides both full SVD (via nalgebra) and randomized SVD for large matrices.
//! The randomized variant uses the Halko-Martinsson-Tropp algorithm (2011):
//! random projection -> QR -> small SVD, giving O(mnr) cost instead of
//! O(mn * min(m,n)) for the full SVD.

use crate::errors::MpsError;
use nalgebra::{DMatrix, DVector, SVD};
use num_complex::Complex64;

/// Result of a truncated SVD.
pub struct TruncatedSvd {
    /// Left singular vectors, shape (m, r).
    pub u: DMatrix<Complex64>,
    /// Singular values (r entries, descending order).
    pub singular_values: Vec<f64>,
    /// Right singular vectors (conjugate transpose), shape (r, n).
    pub vt: DMatrix<Complex64>,
    /// Relative weight of discarded singular values:
    /// `sum(discarded_sv²) / sum(all_sv²)`. Zero if no truncation.
    /// Approximates the 1-fidelity cost of this SVD step.
    pub discarded_weight: f64,
    /// True if the kept rank equals `max_rank` (i.e. the bond cap was binding).
    /// Useful for detecting under-resolution in adaptive schemes.
    pub hit_cap: bool,
}

/// Perform truncated SVD on a complex matrix.
///
/// Given matrix M of shape (m, n), computes M = U * diag(S) * V^dagger,
/// then keeps at most `max_rank` singular values that are above `cutoff`.
/// If `max_trunc_error` is Some, also stops when the relative discarded
/// weight (sum of discarded `s_i^2` / total) would exceed the budget.
///
/// # Errors
///
/// Returns [`MpsError::SvdFailed`] if nalgebra's SVD fails to produce U or V^T.
pub fn truncated_svd(
    matrix: &DMatrix<Complex64>,
    max_rank: usize,
    cutoff: f64,
) -> Result<TruncatedSvd, MpsError> {
    truncated_svd_with_error(matrix, max_rank, cutoff, None)
}

/// Perform truncated SVD with optional adaptive error budget.
///
/// # Errors
///
/// Returns [`MpsError::SvdFailed`] if nalgebra's SVD fails to produce U or V^T.
pub fn truncated_svd_with_error(
    matrix: &DMatrix<Complex64>,
    max_rank: usize,
    cutoff: f64,
    max_trunc_error: Option<f64>,
) -> Result<TruncatedSvd, MpsError> {
    let svd = SVD::new(matrix.clone(), true, true);

    let u_full = svd.u.ok_or(MpsError::SvdFailed)?;
    let vt_full = svd.v_t.ok_or(MpsError::SvdFailed)?;
    let svals: &DVector<f64> = &svd.singular_values;

    let rank = compute_rank(svals, max_rank, cutoff, max_trunc_error);

    let u_trunc = u_full.columns(0, rank).clone_owned();
    let vt_trunc = vt_full.rows(0, rank).clone_owned();
    let kept_svals: Vec<f64> = svals.iter().take(rank).copied().collect();
    let total_weight: f64 = svals.iter().map(|s| s * s).sum();
    let kept_weight: f64 = kept_svals.iter().map(|s| s * s).sum();
    let discarded_weight = if total_weight > 0.0 {
        ((total_weight - kept_weight) / total_weight).max(0.0)
    } else {
        0.0
    };

    Ok(TruncatedSvd {
        u: u_trunc,
        singular_values: kept_svals,
        vt: vt_trunc,
        discarded_weight,
        hit_cap: rank >= max_rank && svals.len() > max_rank,
    })
}

/// Determine how many singular values to keep given truncation criteria.
fn compute_rank(
    svals: &DVector<f64>,
    max_rank: usize,
    cutoff: f64,
    max_trunc_error: Option<f64>,
) -> usize {
    let n = svals.len();

    // Start with all singular values that pass the hard criteria
    let mut rank = 0;
    for i in 0..n {
        if i >= max_rank {
            break;
        }
        if svals[i] < cutoff {
            break;
        }
        rank += 1;
    }

    // Apply adaptive error budget: reduce rank if discarded weight is within budget
    if let Some(max_err) = max_trunc_error {
        let total_weight: f64 = svals.iter().map(|s| s * s).sum();
        if total_weight > 0.0 {
            // Walk backwards from rank, checking if we can drop more values
            let mut discarded_weight = 0.0;
            for i in (1..rank).rev() {
                let candidate_discard = discarded_weight + svals[i] * svals[i];
                if candidate_discard / total_weight > max_err {
                    break;
                }
                discarded_weight = candidate_discard;
                rank = i;
            }
        }
    }

    // Keep at least 1 to avoid empty tensors
    rank.max(1)
}

/// Oversampling parameter for randomized SVD.
const RSVD_OVERSAMPLING: usize = 5;

/// Minimum matrix dimension ratio (min(m,n) / `max_rank`) to trigger randomized SVD.
/// When the ratio exceeds this threshold, randomized SVD is used instead of full SVD.
const RSVD_THRESHOLD: usize = 4;

/// Perform truncated SVD, automatically choosing between full and randomized.
///
/// Uses randomized SVD when `max_rank * RSVD_THRESHOLD < min(m, n)`,
/// otherwise uses full SVD.
///
/// # Errors
///
/// Returns [`MpsError::SvdFailed`] if the underlying SVD fails to produce U or V^T.
pub fn truncated_svd_auto(
    matrix: &DMatrix<Complex64>,
    max_rank: usize,
    cutoff: f64,
) -> Result<TruncatedSvd, MpsError> {
    truncated_svd_auto_with_error(matrix, max_rank, cutoff, None)
}

/// Perform truncated SVD with error budget, auto-selecting algorithm.
///
/// # Errors
///
/// Returns [`MpsError::SvdFailed`] if the underlying SVD fails to produce U or V^T.
pub fn truncated_svd_auto_with_error(
    matrix: &DMatrix<Complex64>,
    max_rank: usize,
    cutoff: f64,
    max_trunc_error: Option<f64>,
) -> Result<TruncatedSvd, MpsError> {
    let m = matrix.nrows();
    let n = matrix.ncols();
    let min_dim = m.min(n);

    if max_rank * RSVD_THRESHOLD < min_dim && max_rank + RSVD_OVERSAMPLING < min_dim {
        randomized_truncated_svd_with_error(matrix, max_rank, cutoff, max_trunc_error)
    } else {
        truncated_svd_with_error(matrix, max_rank, cutoff, max_trunc_error)
    }
}

/// Randomized truncated SVD using the Halko-Martinsson-Tropp algorithm.
///
/// For an m×n matrix A with target rank r:
/// 1. Generate random sketch Ω (n × (r+p))
/// 2. Y = A × Ω  (m × (r+p))
/// 3. Q, _ = QR(Y)  (thin QR)
/// 4. B = Q^H × A  ((r+p) × n)
/// 5. SVD(B) = Ũ Σ V^T
/// 6. U = Q × Ũ
///
/// Cost: O(mn(r+p)) vs O(mn·min(m,n)) for full SVD.
fn randomized_truncated_svd_with_error(
    matrix: &DMatrix<Complex64>,
    max_rank: usize,
    cutoff: f64,
    max_trunc_error: Option<f64>,
) -> Result<TruncatedSvd, MpsError> {
    // f64 mantissa is 53 bits, so we extract top 53 bits and convert in two
    // lossless u32->f64 steps to avoid clippy::cast_precision_loss.
    const SCALE: f64 = 2.0 / 9_007_199_254_740_992.0; // 2 / 2^53

    let m = matrix.nrows();
    let n = matrix.ncols();
    let sketch_cols = (max_rank + RSVD_OVERSAMPLING).min(m.min(n));

    // Step 1: Generate random sketch matrix Ω (n × sketch_cols)
    // Using a simple xorshift64 PRNG seeded deterministically from matrix dimensions.
    // Deterministic seed means same matrix always gives same result.
    let mut rng_state: u64 = 0x5DEE_CE66_D1A4_F87D ^ (m as u64 * 31 + n as u64 * 37);
    let next_f64 = |state: &mut u64| -> f64 {
        *state ^= *state << 13;
        *state ^= *state >> 7;
        *state ^= *state << 17;
        // Map to uniform [-1, 1] (sub-Gaussian suffices for randomized SVD).
        let top53 = *state >> 11;
        let hi = (top53 >> 21) as u32; // upper 32 bits
        let lo = (top53 & 0x1F_FFFF) as u32; // lower 21 bits
        (f64::from(hi) * f64::from(1u32 << 21) + f64::from(lo)) * SCALE - 1.0
    };

    let omega = DMatrix::from_fn(n, sketch_cols, |_i, _j| {
        Complex64::new(next_f64(&mut rng_state), next_f64(&mut rng_state))
    });

    // Step 2: Y = A × Ω  (m × sketch_cols)
    let y = matrix * &omega;

    // Step 3: Thin QR of Y
    let qr = y.qr();
    let q = qr.q(); // m × min(m, sketch_cols)
    let q_cols = q.ncols().min(sketch_cols);
    let q_thin = q.columns(0, q_cols).clone_owned();

    // Step 4: B = Q^H × A  (q_cols × n)
    let b = q_thin.adjoint() * matrix;

    // Step 5: Full SVD of the small matrix B
    let svd_b = SVD::new(b, true, true);
    let u_b = svd_b.u.ok_or(MpsError::SvdFailed)?;
    let vt_b = svd_b.v_t.ok_or(MpsError::SvdFailed)?;
    let svals: &DVector<f64> = &svd_b.singular_values;

    // Determine rank using same criteria as full SVD
    let rank = compute_rank(svals, max_rank, cutoff, max_trunc_error);

    // Step 6: U = Q × Ũ_truncated
    let u_b_trunc = u_b.columns(0, rank).clone_owned();
    let u = &q_thin * &u_b_trunc;

    let vt_trunc = vt_b.rows(0, rank).clone_owned();
    let kept_svals: Vec<f64> = svals.iter().take(rank).copied().collect();
    let total_weight: f64 = svals.iter().map(|s| s * s).sum();
    let kept_weight: f64 = kept_svals.iter().map(|s| s * s).sum();
    let discarded_weight = if total_weight > 0.0 {
        ((total_weight - kept_weight) / total_weight).max(0.0)
    } else {
        0.0
    };

    Ok(TruncatedSvd {
        u,
        singular_values: kept_svals,
        vt: vt_trunc,
        discarded_weight,
        hit_cap: rank >= max_rank && svals.len() > max_rank,
    })
}

/// Perform truncated SVD and absorb singular values into the left matrix.
///
/// Returns `(U * diag(S), V^dagger)` after truncation.
/// Automatically uses randomized SVD for large matrices with small target rank.
///
/// # Errors
///
/// Returns [`MpsError::SvdFailed`] if the underlying SVD fails to produce U or V^T.
pub fn truncated_svd_left_absorb(
    matrix: &DMatrix<Complex64>,
    max_rank: usize,
    cutoff: f64,
    max_trunc_error: Option<f64>,
) -> Result<(DMatrix<Complex64>, DMatrix<Complex64>), MpsError> {
    let (us, vt, _, _) =
        truncated_svd_left_absorb_with_error(matrix, max_rank, cutoff, max_trunc_error)?;
    Ok((us, vt))
}

/// Like `truncated_svd_left_absorb` but also returns (`discarded_weight`, `hit_cap`).
///
/// # Errors
///
/// Returns [`MpsError::SvdFailed`] if the underlying SVD fails to produce U or V^T.
pub fn truncated_svd_left_absorb_with_error(
    matrix: &DMatrix<Complex64>,
    max_rank: usize,
    cutoff: f64,
    max_trunc_error: Option<f64>,
) -> Result<(DMatrix<Complex64>, DMatrix<Complex64>, f64, bool), MpsError> {
    let result = truncated_svd_auto_with_error(matrix, max_rank, cutoff, max_trunc_error)?;
    let mut u_scaled = result.u;
    for (j, &sv) in result.singular_values.iter().enumerate() {
        let scale = Complex64::new(sv, 0.0);
        for i in 0..u_scaled.nrows() {
            u_scaled[(i, j)] *= scale;
        }
    }
    Ok((u_scaled, result.vt, result.discarded_weight, result.hit_cap))
}

/// Perform truncated SVD and absorb singular values into the right matrix.
///
/// Returns `(U, diag(S) * V^dagger)` after truncation.
/// Automatically uses randomized SVD for large matrices with small target rank.
///
/// # Errors
///
/// Returns [`MpsError::SvdFailed`] if the underlying SVD fails to produce U or V^T.
pub fn truncated_svd_right_absorb(
    matrix: &DMatrix<Complex64>,
    max_rank: usize,
    cutoff: f64,
    max_trunc_error: Option<f64>,
) -> Result<(DMatrix<Complex64>, DMatrix<Complex64>), MpsError> {
    let (u, svt, _, _) =
        truncated_svd_right_absorb_with_error(matrix, max_rank, cutoff, max_trunc_error)?;
    Ok((u, svt))
}

/// Like `truncated_svd_right_absorb` but also returns (`discarded_weight`, `hit_cap`).
///
/// # Errors
///
/// Returns [`MpsError::SvdFailed`] if the underlying SVD fails to produce U or V^T.
pub fn truncated_svd_right_absorb_with_error(
    matrix: &DMatrix<Complex64>,
    max_rank: usize,
    cutoff: f64,
    max_trunc_error: Option<f64>,
) -> Result<(DMatrix<Complex64>, DMatrix<Complex64>, f64, bool), MpsError> {
    let result = truncated_svd_auto_with_error(matrix, max_rank, cutoff, max_trunc_error)?;
    let mut svt = result.vt;
    for (i, &sv) in result.singular_values.iter().enumerate() {
        let scale = Complex64::new(sv, 0.0);
        for j in 0..svt.ncols() {
            svt[(i, j)] *= scale;
        }
    }
    Ok((result.u, svt, result.discarded_weight, result.hit_cap))
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn test_truncated_svd_identity() {
        let m = DMatrix::from_fn(3, 3, |i, j| {
            if i == j {
                Complex64::new(1.0, 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        let result = truncated_svd(&m, 10, 1e-12).unwrap();
        assert_eq!(result.singular_values.len(), 3);
        for sv in &result.singular_values {
            assert_relative_eq!(*sv, 1.0, epsilon = 1e-10);
        }
    }

    #[test]
    fn test_truncated_svd_rank_1() {
        // Rank-1 matrix: outer product of [1, 0] and [1, 1]
        let m = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        let result = truncated_svd(&m, 10, 1e-12).unwrap();
        // Should have rank 1 (second singular value ~ 0)
        assert_eq!(result.singular_values.len(), 1);
        assert_relative_eq!(result.singular_values[0], 2.0_f64.sqrt(), epsilon = 1e-10);
    }

    #[test]
    fn test_truncated_svd_max_rank() {
        let m = DMatrix::from_fn(4, 4, |i, j| {
            if i == j {
                Complex64::new(f64::from(u32::try_from(4 - i).unwrap()), 0.0)
            } else {
                Complex64::new(0.0, 0.0)
            }
        });
        let result = truncated_svd(&m, 2, 1e-12).unwrap();
        assert_eq!(result.singular_values.len(), 2);
        assert_relative_eq!(result.singular_values[0], 4.0, epsilon = 1e-10);
        assert_relative_eq!(result.singular_values[1], 3.0, epsilon = 1e-10);
    }

    #[test]
    fn test_left_absorb_reconstructs() {
        let m = DMatrix::from_row_slice(
            2,
            3,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
            ],
        );
        let (u_s, vt) = truncated_svd_left_absorb(&m, 10, 1e-12, None).unwrap();
        let reconstructed = &u_s * &vt;
        for i in 0..2 {
            for j in 0..3 {
                assert_relative_eq!(reconstructed[(i, j)].re, m[(i, j)].re, epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_adaptive_truncation() {
        // Build a matrix with known singular value spectrum: 10, 5, 1, 0.1, 0.01
        // Total weight = 100 + 25 + 1 + 0.01 + 0.0001 = 126.0201
        let mut m = DMatrix::zeros(5, 5);
        let spectrum = [10.0_f64, 5.0, 1.0, 0.1, 0.01];
        for (i, &s) in spectrum.iter().enumerate() {
            m[(i, i)] = Complex64::new(s, 0.0);
        }

        // With max_rank=5, cutoff=0, no error budget: keep all 5
        let r1 = truncated_svd_with_error(&m, 5, 0.0, None).unwrap();
        assert_eq!(r1.singular_values.len(), 5);

        // With error budget 1e-4: total=126.02, discarding 0.01^2=0.0001 costs 0.0001/126.02 ~ 8e-7
        // Discarding 0.1^2 + 0.01^2 = 0.0101 costs 0.0101/126.02 ~ 8e-5
        // So error budget 1e-3 should drop the last two (keep 3)
        let r2 = truncated_svd_with_error(&m, 5, 0.0, Some(1e-3)).unwrap();
        assert!(
            r2.singular_values.len() <= 4,
            "should drop small values, got {}",
            r2.singular_values.len()
        );
        assert!(
            r2.singular_values.len() >= 2,
            "should keep large values, got {}",
            r2.singular_values.len()
        );

        // With tight error budget 1e-6: should keep almost all
        let r3 = truncated_svd_with_error(&m, 5, 0.0, Some(1e-6)).unwrap();
        assert!(r3.singular_values.len() >= 4);
    }

    #[test]
    fn test_randomized_svd_low_rank() {
        // Build a rank-2 matrix of size 20x20 (forces randomized path when max_rank=2)
        // A = u * v^T where u is 20x2 and v is 20x2
        let u_col = DMatrix::from_fn(20, 2, |i, j| {
            Complex64::new(
                f64::from(u32::try_from(i * 3 + j * 7 + 1).unwrap()).sin(),
                0.0,
            )
        });
        let v_col = DMatrix::from_fn(20, 2, |i, j| {
            Complex64::new(
                f64::from(u32::try_from(i * 5 + j * 11 + 3).unwrap()).cos(),
                0.0,
            )
        });
        let a = &u_col * &v_col.adjoint();

        // Randomized SVD with max_rank=2 should recover the matrix
        let result = randomized_truncated_svd_with_error(&a, 2, 1e-12, None).unwrap();
        assert!(result.singular_values.len() <= 2);

        // Reconstruct and check
        let mut u_s = result.u.clone();
        for (j, &sv) in result.singular_values.iter().enumerate() {
            for i in 0..u_s.nrows() {
                u_s[(i, j)] *= Complex64::new(sv, 0.0);
            }
        }
        let reconstructed = &u_s * &result.vt;
        let error = (&a - &reconstructed).norm();
        assert!(
            error < 1e-6,
            "reconstruction error {error} should be < 1e-6"
        );
    }

    #[test]
    fn test_randomized_svd_truncation() {
        // Full-rank 20x20 matrix, truncate to rank 3
        let a = DMatrix::from_fn(20, 20, |i, j| {
            Complex64::new(
                f64::from(u32::try_from(i * 7 + j * 13 + 5).unwrap()).sin(),
                f64::from(u32::try_from(i + j).unwrap()).cos(),
            )
        });

        let result_full = truncated_svd(&a, 3, 1e-15).unwrap();
        let result_rand = randomized_truncated_svd_with_error(&a, 3, 1e-15, None).unwrap();

        // Both should return rank 3
        assert_eq!(result_full.singular_values.len(), 3);
        assert_eq!(result_rand.singular_values.len(), 3);

        // Singular values should be close (randomized is approximate)
        for (sf, sr) in result_full
            .singular_values
            .iter()
            .zip(result_rand.singular_values.iter())
        {
            assert_relative_eq!(sf, sr, epsilon = 0.1 * sf);
        }
    }

    #[test]
    fn test_auto_selects_full_for_small() {
        // Small matrix: should use full SVD (same result as truncated_svd)
        let m = DMatrix::from_fn(4, 4, |i, j| {
            Complex64::new(f64::from(u32::try_from(i + j).unwrap()), 0.0)
        });
        let result_auto = truncated_svd_auto(&m, 2, 1e-12).unwrap();
        let result_full = truncated_svd(&m, 2, 1e-12).unwrap();
        assert_eq!(
            result_auto.singular_values.len(),
            result_full.singular_values.len()
        );
        for (sa, sf) in result_auto
            .singular_values
            .iter()
            .zip(result_full.singular_values.iter())
        {
            assert_relative_eq!(sa, sf, epsilon = 1e-10);
        }
    }
}
