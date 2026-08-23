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
use nalgebra::{DMatrix, DVector, SVD, SymmetricEigen};
use num_complex::Complex64;

// Match nalgebra's default deflation threshold. Asking the bidiagonal solver
// to resolve below attainable f64 precision destroys numerical zero singular
// values and can prevent convergence.
const SVD_CONVERGENCE_EPSILON: f64 = 5.0 * f64::EPSILON;

// Nalgebra counts total QR steps, not per-singular-value steps. Sixty-four
// steps per singular direction gives difficult clustered spectra ample room
// to deflate while making failure finite instead of using `0` (unbounded).
const SVD_ITERATIONS_PER_DIMENSION: usize = 64;

// Retained-triplet checks include two matrix-vector products and basis
// orthogonalization. This dimension-scaled backward-error allowance remains
// O(epsilon), far below the O(sqrt(epsilon)) floor of a Gram spectrum.
const SVD_TRIPLET_VALIDATION_MULTIPLIER: f64 = 512.0;

// Fallback-derived factors include a phase-aligned QR repair. Bound the factor
// error of their retained block at this pre-#580 linear-dimension scale.
// Reconstruction alone is insufficient for Gram-derived spectra; independent
// retained-triplet and isometry checks must also pass.
const SVD_FALLBACK_RECONSTRUCTION_MULTIPLIER: f64 = 512.0;

fn iteration_limit(rows: usize, cols: usize) -> usize {
    rows.min(cols)
        .max(1)
        .saturating_mul(SVD_ITERATIONS_PER_DIMENSION)
}

fn numerical_zero_threshold(matrix: &DMatrix<Complex64>) -> f64 {
    // Backward error accumulates with the inner dimension even when the
    // eigensolver uses nalgebra's default per-step deflation tolerance.
    let inner_dimension = matrix.nrows().min(matrix.ncols()).max(1) as f64;
    matrix.norm() * SVD_CONVERGENCE_EPSILON * inner_dimension
}

fn fallback_zero_threshold(matrix: &DMatrix<Complex64>) -> f64 {
    // The fallback's retained-triplet validator uses this same backward-error
    // scale. Directions below it are numerically indistinguishable from the
    // exact null space; the independent real SVD must agree before completion.
    dimension_scaled_backward_error_tolerance(matrix, SVD_TRIPLET_VALIDATION_MULTIPLIER)
}

fn dimension_scaled_backward_error_tolerance(matrix: &DMatrix<Complex64>, multiplier: f64) -> f64 {
    let dimension = matrix.nrows().max(matrix.ncols()).max(1) as f64;
    matrix.norm() * (multiplier * dimension * f64::EPSILON)
}

type SvdFactors = (DMatrix<Complex64>, DVector<f64>, DMatrix<Complex64>);

fn direct_svd_factors(matrix: &DMatrix<Complex64>) -> Result<SvdFactors, MpsError> {
    let mut svd = SVD::try_new(
        matrix.clone(),
        true,
        true,
        SVD_CONVERGENCE_EPSILON,
        iteration_limit(matrix.nrows(), matrix.ncols()),
    )
    .ok_or(MpsError::SvdFailed)?;
    svd.sort_by_singular_values();
    Ok((
        svd.u.ok_or(MpsError::SvdFailed)?,
        svd.singular_values,
        svd.v_t.ok_or(MpsError::SvdFailed)?,
    ))
}

fn adjoint_svd_factors(matrix: &DMatrix<Complex64>) -> Result<SvdFactors, MpsError> {
    let mut svd = SVD::try_new(
        matrix.adjoint(),
        true,
        true,
        SVD_CONVERGENCE_EPSILON,
        iteration_limit(matrix.nrows(), matrix.ncols()),
    )
    .ok_or(MpsError::SvdFailed)?;
    svd.sort_by_singular_values();
    Ok((
        svd.v_t.ok_or(MpsError::SvdFailed)?.adjoint().into_owned(),
        svd.singular_values,
        svd.u.ok_or(MpsError::SvdFailed)?.adjoint().into_owned(),
    ))
}

fn reconstruction_error(
    matrix: &DMatrix<Complex64>,
    u: &DMatrix<Complex64>,
    singular_values: &DVector<f64>,
    vt: &DMatrix<Complex64>,
) -> f64 {
    let mut us = u.clone();
    for (column, &singular_value) in singular_values.iter().enumerate() {
        for row in 0..us.nrows() {
            us[(row, column)] *= singular_value;
        }
    }
    let reconstructed = us * vt;
    matrix
        .iter()
        .zip(reconstructed.iter())
        .map(|(expected, actual)| (*expected - *actual).norm())
        .fold(0.0_f64, f64::max)
}

fn factorization_discarded_weight(singular_values: &DVector<f64>, retained_rank: usize) -> f64 {
    let total_weight: f64 = singular_values.iter().map(|value| value * value).sum();
    if total_weight > 0.0 {
        let discarded_weight: f64 = singular_values
            .iter()
            .skip(retained_rank)
            .map(|value| value * value)
            .sum();
        (discarded_weight / total_weight).clamp(0.0, 1.0)
    } else {
        0.0
    }
}

fn retained_block_reconstruction_error(
    matrix: &DMatrix<Complex64>,
    factors: &SvdFactors,
    retained_rank: usize,
) -> f64 {
    let retained_u = factors.0.columns(0, retained_rank).into_owned();
    let retained_singular_values = factors.1.rows(0, retained_rank).into_owned();
    let retained_vt = factors.2.rows(0, retained_rank).into_owned();
    reconstruction_error(matrix, &retained_u, &retained_singular_values, &retained_vt)
}

fn retained_block_reconstruction_tolerance(
    matrix: &DMatrix<Complex64>,
    factors: &SvdFactors,
    retained_rank: usize,
) -> f64 {
    // The rank-r residual is the genuinely discarded tail plus factor error on
    // the retained block. The factorization's own claimed discarded weight w
    // bounds the tail in Frobenius norm by sqrt(w) * ||A||_F. The retained
    // factor error keeps the pre-#580 O(max(m,n) * epsilon * ||A||_F) bound;
    // its constant already absorbs the sqrt(min(m,n)) composition from
    // normwise triplet residuals to entrywise reconstruction. A factorization
    // that silently hides tail mass cannot charge it to w and is rejected.
    let claimed_discarded_weight = factorization_discarded_weight(&factors.1, retained_rank);
    claimed_discarded_weight.sqrt() * matrix.norm()
        + dimension_scaled_backward_error_tolerance(matrix, SVD_FALLBACK_RECONSTRUCTION_MULTIPLIER)
}

/// Recover an SVD-like factorization from the Hermitian Gram matrix.
///
/// This is a fallback for a nalgebra complex-SVD failure mode on nearly
/// rank-deficient rectangular matrices. The eigenvectors on the smaller side
/// form the canonical factor exactly; deriving the other factor by multiplying
/// with the input also avoids dividing by a squared condition number when
/// reconstructing the retained components.
fn gram_svd_factors(
    matrix: &DMatrix<Complex64>,
    zero_threshold: f64,
) -> Result<SvdFactors, MpsError> {
    let (rows, cols) = matrix.shape();
    let rank = rows.min(cols);
    let mut components = Vec::with_capacity(rank);

    if rows >= cols {
        let gram = matrix.adjoint() * matrix;
        let eigen = SymmetricEigen::try_new(gram, f64::EPSILON, iteration_limit(rank, rank))
            .ok_or(MpsError::SvdFailed)?;
        for column in 0..rank {
            let v = eigen.eigenvectors.column(column).into_owned();
            let av = matrix * &v;
            let singular_value = av.norm();
            let u = if singular_value > 0.0 {
                av / Complex64::new(singular_value, 0.0)
            } else {
                DVector::zeros(rows)
            };
            components.push((singular_value, u, v));
        }
    } else {
        let gram = matrix * matrix.adjoint();
        let eigen = SymmetricEigen::try_new(gram, f64::EPSILON, iteration_limit(rank, rank))
            .ok_or(MpsError::SvdFailed)?;
        for column in 0..rank {
            let u = eigen.eigenvectors.column(column).into_owned();
            let u_adjoint_a = u.adjoint() * matrix;
            let singular_value = u_adjoint_a.norm();
            let v = if singular_value > 0.0 {
                u_adjoint_a.adjoint().into_owned() / Complex64::new(singular_value, 0.0)
            } else {
                DVector::zeros(cols)
            };
            components.push((singular_value, u, v));
        }
    }

    components.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut u = DMatrix::zeros(rows, rank);
    let mut singular_values = DVector::zeros(rank);
    let mut vt = DMatrix::zeros(rank, cols);
    for (column, (singular_value, left, right)) in components.into_iter().enumerate() {
        singular_values[column] = singular_value;
        if singular_value == 0.0 && rows >= cols {
            let completed = orthonormal_complement_column(&u, column)?;
            u.set_column(column, &completed);
        } else {
            u.set_column(column, &left);
        }
        let right = if singular_value == 0.0 && rows < cols {
            orthonormal_complement_column(&vt.adjoint(), column)?
        } else {
            right
        };
        for row in 0..cols {
            vt[(column, row)] = right[row].conj();
        }
    }

    // Gram eigenvectors in a numerical null space can produce O(epsilon)
    // values from `A*v`. Complete only directions within the dimension-scaled
    // default SVD backward error. The much larger sqrt(epsilon) Gram floor
    // remains nonzero and is rejected by retained-spectrum validation.
    for column in 0..rank {
        if singular_values[column] <= zero_threshold {
            singular_values[column] = 0.0;
            if rows >= cols {
                u.set_column(column, &orthonormal_complement_column(&u, column)?);
            } else {
                let right = orthonormal_complement_column(&vt.adjoint(), column)?;
                for row in 0..cols {
                    vt[(column, row)] = right[row].conj();
                }
            }
        }
    }

    Ok(reorthonormalize_derived_factor(
        matrix,
        u,
        singular_values,
        vt,
    ))
}

/// Deterministically complete an orthonormal basis using coordinate vectors.
fn orthonormal_complement_column(
    basis: &DMatrix<Complex64>,
    columns: usize,
) -> Result<DVector<Complex64>, MpsError> {
    for pivot in 0..basis.nrows() {
        let mut candidate = DVector::zeros(basis.nrows());
        candidate[pivot] = Complex64::new(1.0, 0.0);
        // A second modified Gram-Schmidt pass controls loss of orthogonality.
        for _ in 0..2 {
            for column in 0..columns {
                let vector = basis.column(column);
                let projection = vector.dotc(&candidate);
                candidate -= vector * projection;
            }
        }
        let norm = candidate.norm();
        if norm > 64.0 * f64::EPSILON {
            return Ok(candidate / Complex64::new(norm, 0.0));
        }
    }
    Err(MpsError::SvdFailed)
}

fn retained_spectrum_is_trustworthy(
    matrix: &DMatrix<Complex64>,
    factors: &SvdFactors,
    retained_rank: usize,
) -> bool {
    let (u, singular_values, vt) = factors;
    let dimension = matrix.nrows().max(matrix.ncols()).max(1) as f64;
    let residual_tolerance =
        dimension_scaled_backward_error_tolerance(matrix, SVD_TRIPLET_VALIDATION_MULTIPLIER);
    let isometry_tolerance = SVD_TRIPLET_VALIDATION_MULTIPLIER * dimension * f64::EPSILON;

    for column in 0..retained_rank {
        let left = u.column(column).into_owned();
        let right = vt.row(column).adjoint().into_owned();
        let singular_value = Complex64::new(singular_values[column], 0.0);
        let left_residual = (matrix * &right - &left * singular_value).norm();
        let right_residual = (matrix.adjoint() * &left - &right * singular_value).norm();
        if left_residual > residual_tolerance || right_residual > residual_tolerance {
            return false;
        }
    }

    let u_gram = u.columns(0, retained_rank).adjoint() * u.columns(0, retained_rank);
    let v_gram = vt.rows(0, retained_rank) * vt.rows(0, retained_rank).adjoint();
    (0..retained_rank).all(|row| {
        (0..retained_rank).all(|column| {
            let expected = if row == column { 1.0 } else { 0.0 };
            (u_gram[(row, column)] - Complex64::new(expected, 0.0)).norm() <= isometry_tolerance
                && (v_gram[(row, column)] - Complex64::new(expected, 0.0)).norm()
                    <= isometry_tolerance
        })
    })
}

fn svd_factors_are_certified(
    matrix: &DMatrix<Complex64>,
    factors: &SvdFactors,
    retained_rank: usize,
) -> bool {
    retained_block_reconstruction_error(matrix, factors, retained_rank)
        <= retained_block_reconstruction_tolerance(matrix, factors, retained_rank)
        && retained_spectrum_is_trustworthy(matrix, factors, retained_rank)
}

/// Replace the columns by the phase-aligned thin-Q factor of their QR.
///
/// A factor recovered as `A*v/s` or `u^H*A/s` has orthogonality error
/// `O(epsilon * ||A|| / s)`. The fallback is specifically needed for spectra
/// where that amplification is visible, so validate a genuinely orthonormal
/// factor instead of asking the derived columns to meet a fixed epsilon-scale
/// isometry bound. Phase alignment keeps the reconstruction perturbation at
/// the size of the QR correction rather than allowing arbitrary QR phases.
fn thin_qr_columns(matrix: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    let columns = matrix.ncols();
    let q = matrix.clone().qr().q();
    let mut thin_q = q.columns(0, columns).into_owned();
    for column in 0..columns {
        let overlap = thin_q.column(column).dotc(&matrix.column(column));
        let norm = overlap.norm();
        if norm > 0.0 {
            let phase = overlap / Complex64::new(norm, 0.0);
            thin_q
                .column_mut(column)
                .iter_mut()
                .for_each(|x| *x *= phase);
        }
    }
    thin_q
}

/// QR-orthonormalize the factor derived through division by singular values.
fn reorthonormalize_derived_factor(
    matrix: &DMatrix<Complex64>,
    mut u: DMatrix<Complex64>,
    singular_values: DVector<f64>,
    mut vt: DMatrix<Complex64>,
) -> SvdFactors {
    if matrix.nrows() >= matrix.ncols() {
        u = thin_qr_columns(&u);
    } else {
        vt = thin_qr_columns(&vt.adjoint()).adjoint().into_owned();
    }
    (u, singular_values, vt)
}

fn orthogonalize_against(
    mut candidate: DVector<Complex64>,
    basis: &[DVector<Complex64>],
) -> Option<DVector<Complex64>> {
    for _ in 0..2 {
        for vector in basis {
            let projection = vector.dotc(&candidate);
            candidate -= vector * projection;
        }
    }
    let norm = candidate.norm();
    (norm > 64.0 * f64::EPSILON).then(|| candidate / Complex64::new(norm, 0.0))
}

/// Independent fallback through the equivalent doubled real matrix.
///
/// Realification avoids nalgebra's complex bidiagonal-vector failure without
/// squaring the condition number. The doubled spectrum contains each complex
/// singular direction twice; complex Gram-Schmidt removes its `i` multiple.
fn realified_svd_factors(
    matrix: &DMatrix<Complex64>,
    deflation_threshold: f64,
) -> Result<SvdFactors, MpsError> {
    let (rows, cols) = matrix.shape();
    let rank = rows.min(cols);
    let realified = DMatrix::from_fn(2 * rows, 2 * cols, |row, column| {
        let source_row = row % rows;
        let source_column = column % cols;
        let value = matrix[(source_row, source_column)];
        match (row < rows, column < cols) {
            (true, true) | (false, false) => value.re,
            (true, false) => -value.im,
            (false, true) => value.im,
        }
    });
    let svd = SVD::try_new(
        realified,
        true,
        true,
        SVD_CONVERGENCE_EPSILON,
        iteration_limit(2 * rows, 2 * cols),
    )
    .ok_or(MpsError::SvdFailed)?;
    let real_singular_values = svd.singular_values;
    let real_u = svd.u.ok_or(MpsError::SvdFailed)?;
    let real_vt = svd.v_t.ok_or(MpsError::SvdFailed)?;
    let mut basis = Vec::with_capacity(rank);
    let mut components = Vec::with_capacity(rank);
    // Values below the reconstruction validator's backward-error floor are
    // numerical zeros. This remains O(epsilon), rather than the much larger
    // sqrt(epsilon) floor produced by a Gram eigenspectrum.
    for column in 0..2 * rank {
        if rows >= cols {
            let candidate = DVector::from_fn(cols, |row, _| {
                Complex64::new(real_vt[(column, row)], real_vt[(column, cols + row)])
            });
            let Some(right) = orthogonalize_against(candidate, &basis) else {
                continue;
            };
            let av = matrix * &right;
            let actual_singular_value = av.norm();
            let reported_singular_value = real_singular_values[column];
            if (actual_singular_value - reported_singular_value).abs() > deflation_threshold {
                continue;
            }
            let singular_value = if reported_singular_value <= deflation_threshold {
                0.0
            } else {
                actual_singular_value
            };
            let left = if singular_value > 0.0 {
                av / Complex64::new(singular_value, 0.0)
            } else {
                DVector::zeros(rows)
            };
            basis.push(right.clone());
            components.push((singular_value, left, right));
        } else {
            let candidate = DVector::from_fn(rows, |row, _| {
                Complex64::new(real_u[(row, column)], real_u[(rows + row, column)])
            });
            let Some(left) = orthogonalize_against(candidate, &basis) else {
                continue;
            };
            let u_adjoint_a = left.adjoint() * matrix;
            let actual_singular_value = u_adjoint_a.norm();
            let reported_singular_value = real_singular_values[column];
            if (actual_singular_value - reported_singular_value).abs() > deflation_threshold {
                continue;
            }
            let singular_value = if reported_singular_value <= deflation_threshold {
                0.0
            } else {
                actual_singular_value
            };
            let right = if singular_value > 0.0 {
                u_adjoint_a.adjoint().into_owned() / Complex64::new(singular_value, 0.0)
            } else {
                DVector::zeros(cols)
            };
            basis.push(left.clone());
            components.push((singular_value, left, right));
        }
        if components.len() == rank {
            break;
        }
    }
    if components.len() != rank {
        return Err(MpsError::SvdFailed);
    }

    components.sort_by(|left, right| right.0.total_cmp(&left.0));
    let mut u = DMatrix::zeros(rows, rank);
    let mut singular_values = DVector::zeros(rank);
    let mut vt = DMatrix::zeros(rank, cols);
    for (column, (singular_value, left, right)) in components.into_iter().enumerate() {
        singular_values[column] = singular_value;
        if singular_value == 0.0 && rows >= cols {
            u.set_column(column, &orthonormal_complement_column(&u, column)?);
        } else {
            u.set_column(column, &left);
        }
        let right = if singular_value == 0.0 && rows < cols {
            orthonormal_complement_column(&vt.adjoint(), column)?
        } else {
            right
        };
        for row in 0..cols {
            vt[(column, row)] = right[row].conj();
        }
    }
    Ok(reorthonormalize_derived_factor(
        matrix,
        u,
        singular_values,
        vt,
    ))
}

fn stable_svd_factors(
    matrix: &DMatrix<Complex64>,
    max_rank: usize,
    cutoff: f64,
    max_trunc_error: Option<f64>,
) -> Result<SvdFactors, MpsError> {
    if matrix
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err(MpsError::SvdFailed);
    }
    let max_element = matrix
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    // Keep the cheap, unvalidated primary path on its deliberately stricter
    // max-element, dimension-free gate. If that gauge-sensitive check fails,
    // the same tail-budgeted retained-block reconstruction, retained-triplet,
    // and isometry checks used for the derived fallbacks can still certify the
    // nalgebra factors.
    // Accepting those already-computed factors trades up to roughly four
    // orders of max-entry reconstruction residual (while remaining inside the
    // certified backward-error bound) for skipping the more accurate fallback
    // recomputation.
    let reconstruction_tolerance = max_element * (256.0 * f64::EPSILON);
    if let Ok(adjoint) = adjoint_svd_factors(matrix) {
        let full_reconstruction_error =
            reconstruction_error(matrix, &adjoint.0, &adjoint.1, &adjoint.2);
        let retained_rank = compute_rank(&adjoint.1, max_rank, cutoff, max_trunc_error);
        if full_reconstruction_error <= reconstruction_tolerance
            || svd_factors_are_certified(matrix, &adjoint, retained_rank)
        {
            return Ok(adjoint);
        }
    }

    if let Ok(direct) = direct_svd_factors(matrix) {
        let full_reconstruction_error =
            reconstruction_error(matrix, &direct.0, &direct.1, &direct.2);
        let retained_rank = compute_rank(&direct.1, max_rank, cutoff, max_trunc_error);
        if full_reconstruction_error <= reconstruction_tolerance
            || svd_factors_are_certified(matrix, &direct, retained_rank)
        {
            return Ok(direct);
        }
    }

    if let Ok(gram) = gram_svd_factors(matrix, numerical_zero_threshold(matrix)) {
        let retained_rank = compute_rank(&gram.1, max_rank, cutoff, max_trunc_error);
        if svd_factors_are_certified(matrix, &gram, retained_rank) {
            return Ok(gram);
        }
    }

    // Do not accept a reconstruction-only Gram result: its small singular
    // values can sit at the sqrt(epsilon) floor. Retry with an independent
    // nonsquaring formulation, and propagate failure if that spectrum also
    // cannot satisfy the caller's requested truncation policy.
    let realified = realified_svd_factors(matrix, fallback_zero_threshold(matrix))?;
    let retained_rank = compute_rank(&realified.1, max_rank, cutoff, max_trunc_error);
    if svd_factors_are_certified(matrix, &realified, retained_rank) {
        Ok(realified)
    } else {
        Err(MpsError::SvdFailed)
    }
}

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
    let (u_full, svals, vt_full) = stable_svd_factors(matrix, max_rank, cutoff, max_trunc_error)?;

    let rank = compute_rank(&svals, max_rank, cutoff, max_trunc_error);

    let u_trunc = u_full.columns(0, rank).clone_owned();
    let vt_trunc = vt_full.rows(0, rank).clone_owned();
    let kept_svals: Vec<f64> = svals.iter().take(rank).copied().collect();
    let discarded_weight = factorization_discarded_weight(&svals, rank);

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
    let (u_b, svals, vt_b) = stable_svd_factors(&b, max_rank, cutoff, max_trunc_error)?;

    // Determine rank using same criteria as full SVD
    let rank = compute_rank(&svals, max_rank, cutoff, max_trunc_error);

    // Step 6: U = Q × Ũ_truncated
    let u_b_trunc = u_b.columns(0, rank).clone_owned();
    let u = &q_thin * &u_b_trunc;

    let vt_trunc = vt_b.rows(0, rank).clone_owned();
    let kept_svals: Vec<f64> = svals.iter().take(rank).copied().collect();
    let discarded_weight = factorization_discarded_weight(&svals, rank);

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

    fn assert_complex_identity(matrix: &DMatrix<Complex64>, tolerance: f64) {
        assert_eq!(matrix.nrows(), matrix.ncols());
        let max_error = (0..matrix.nrows())
            .flat_map(|row| {
                (0..matrix.ncols()).map(move |column| {
                    let expected = if row == column { 1.0 } else { 0.0 };
                    (matrix[(row, column)] - Complex64::new(expected, 0.0)).norm()
                })
            })
            .fold(0.0_f64, f64::max);
        assert!(
            max_error <= tolerance,
            "matrix is not identity: max error={max_error:.3e}"
        );
    }

    fn deterministic_isometry(rows: usize, columns: usize, offset: usize) -> DMatrix<Complex64> {
        let raw = DMatrix::from_fn(rows, columns, |row, column| {
            let index = u32::try_from((row + 1) * (column + offset + 2)).unwrap();
            Complex64::new(
                (f64::from(index) * 0.73).sin(),
                (f64::from(index) * 1.17).cos(),
            )
        });
        raw.qr().q().columns(0, columns).into_owned()
    }

    fn matrix_with_spectrum(rows: usize, cols: usize, spectrum: &[f64]) -> DMatrix<Complex64> {
        let rank = rows.min(cols);
        assert_eq!(spectrum.len(), rank);
        let left = deterministic_isometry(rows, rank, 1);
        let right = deterministic_isometry(cols, rank, 7);
        let diagonal = DMatrix::from_diagonal(&DVector::from_iterator(
            rank,
            spectrum.iter().map(|&value| Complex64::new(value, 0.0)),
        ));
        left * diagonal * right.adjoint()
    }

    fn decode_base64_fixture(encoded: &str) -> Vec<u8> {
        let mut decoded = Vec::with_capacity(encoded.len() * 3 / 4);
        let mut accumulator = 0_u32;
        let mut bits = 0_u32;
        for byte in encoded.bytes().filter(|byte| !byte.is_ascii_whitespace()) {
            let value = match byte {
                b'A'..=b'Z' => u32::from(byte - b'A'),
                b'a'..=b'z' => u32::from(byte - b'a') + 26,
                b'0'..=b'9' => u32::from(byte - b'0') + 52,
                b'+' => 62,
                b'/' => 63,
                b'=' => break,
                _ => panic!("invalid base64 fixture byte"),
            };
            accumulator = (accumulator << 6) | value;
            bits += 6;
            if bits >= 8 {
                bits -= 8;
                decoded.push((accumulator >> bits) as u8);
                accumulator &= (1_u32 << bits) - 1;
            }
        }
        decoded
    }

    fn issue_580_seed_143_matrix() -> (DMatrix<Complex64>, usize, f64, Option<f64>) {
        let bytes = decode_base64_fixture(include_str!("fixtures/issue_580_seed_143_svd.b64"));
        assert_eq!(&bytes[..8], b"PECOSSVD");
        let mut offset = 8;
        let mut read_u64 = || {
            let value = u64::from_le_bytes(bytes[offset..offset + 8].try_into().unwrap());
            offset += 8;
            value
        };
        let rows = usize::try_from(read_u64()).unwrap();
        let columns = usize::try_from(read_u64()).unwrap();
        let max_rank = usize::try_from(read_u64()).unwrap();
        let cutoff = f64::from_bits(read_u64());
        let max_trunc_error = f64::from_bits(read_u64());
        let (value_bytes, remainder) = bytes[offset..].as_chunks::<16>();
        assert!(remainder.is_empty());
        let values = value_bytes
            .iter()
            .map(|chunk| {
                let real = f64::from_bits(u64::from_le_bytes(chunk[..8].try_into().unwrap()));
                let imaginary = f64::from_bits(u64::from_le_bytes(chunk[8..].try_into().unwrap()));
                Complex64::new(real, imaginary)
            })
            .collect::<Vec<_>>();
        assert_eq!(values.len(), rows * columns);
        (
            DMatrix::from_column_slice(rows, columns, &values),
            max_rank,
            cutoff,
            (!max_trunc_error.is_nan()).then_some(max_trunc_error),
        )
    }

    fn assert_realified_factors(matrix: &DMatrix<Complex64>, expected_singular_values: &[f64]) {
        const FACTOR_ISOMETRY_TOLERANCE: f64 = 1e-12;
        let factors = realified_svd_factors(matrix, fallback_zero_threshold(matrix)).unwrap();
        assert_eq!(factors.1.len(), expected_singular_values.len());
        for (index, (&actual, &expected)) in
            factors.1.iter().zip(expected_singular_values).enumerate()
        {
            // Keep an absolute O(epsilon) floor for exact zeros, while making
            // the graded-spectrum check relative so 1e-12 cannot pass as 0.
            let tolerance = 1e-3 * expected + 1e-15 * matrix.norm().max(1.0);
            assert!(
                (actual - expected).abs() <= tolerance,
                "singular value {index}: actual={actual:.16e}, expected={expected:.16e}, tolerance={tolerance:.3e}"
            );
        }
        assert_complex_identity(
            &(factors.0.adjoint() * &factors.0),
            FACTOR_ISOMETRY_TOLERANCE,
        );
        assert_complex_identity(
            &(&factors.2 * factors.2.adjoint()),
            FACTOR_ISOMETRY_TOLERANCE,
        );
        let error = reconstruction_error(matrix, &factors.0, &factors.1, &factors.2);
        let tolerance = matrix
            .iter()
            .map(|value| value.norm())
            .fold(0.0_f64, f64::max)
            * 2e-12;
        assert!(
            error <= tolerance.max(f64::EPSILON),
            "realified reconstruction error={error:.3e}, tolerance={tolerance:.3e}"
        );
    }

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
    fn test_nearly_rank_deficient_complex_svd_reconstructs() {
        let c = Complex64::new;
        let matrix = DMatrix::from_row_slice(
            4,
            4,
            &[
                c(0.924_751_335_947_476_8, 1.149_464_534_439_411_6e-18),
                c(0.005_588_521_570_544_361, 0.001_862_840_523_514_787_4),
                c(1.113_638_002_453_152e-17, -1.001_691_566_199_797_2e-17),
                c(-9.900_226_694_310_336e-18, 2.045_446_567_427_834e-17),
                c(4.174_238_067_903_684e-32, -2.107_447_139_670_980_2e-32),
                c(-4.561_859_616_215_51e-32, 3.030_573_543_716_516_5e-32),
                c(7.649_815_290_319_717e-16, 2.976_525_575_056_578e-16),
                c(-1.782_373_947_207_878e-16, -1.251_720_721_791_526e-16),
                c(-0.010_225_154_846_847_47, 0.010_033_920_008_989_889),
                c(0.341_701_797_119_702_33, -0.166_837_985_536_791_08),
                c(3.884_325_578_198_641e-19, 5.748_278_930_752_762e-17),
                c(4.512_116_547_715_041e-19, -6.004_905_012_027_449e-17),
                c(5.033_965_804_543_656e-32, -1.227_951_147_885_866_2e-32),
                c(-5.445_650_955_783_522e-32, 1.741_442_486_895_124e-32),
                c(4.515_259_889_376_597e-16, 2.417_433_227_383_755e-16),
                c(-1.809_225_712_298_247e-16, -2.213_247_773_922_222_7e-16),
            ],
        );
        let (us, vt) = truncated_svd_left_absorb(&matrix, 4, 0.0, None).unwrap();
        let reconstructed = us * vt;
        let max_error = matrix
            .iter()
            .zip(reconstructed.iter())
            .map(|(expected, actual)| (*expected - *actual).norm())
            .fold(0.0_f64, f64::max);
        assert!(max_error <= 1e-14, "reconstruction error={max_error:.3e}");
    }

    #[test]
    fn test_complex_svd_default_epsilon_reconstructs_regression() {
        // Captured from a rank-reducing return SWAP in the issue #557 sweep.
        // Nalgebra's default convergence tolerance decomposes this rank-one
        // matrix accurately; the former 1e-18 tolerance manufactured the
        // O(1e-1) reconstruction failure that sent it to the Gram path.
        let c = Complex64::new;
        let matrix = DMatrix::from_column_slice(
            4,
            2,
            &[
                c(0.736_532_729_158_897_1, -0.135_156_274_059_290_76),
                c(0.135_156_274_059_290_98, -0.466_220_181_040_314_65),
                c(0.130_066_723_459_075_8, 0.000_104_181_257_041_555_5),
                c(-0.089_876_514_560_921_97, 0.179_393_656_944_288_47),
                c(0.055_983_561_755_173_154, 0.305_081_845_549_284_8),
                c(0.193_114_722_038_938_23, 0.055_983_561_755_173_245),
                c(-0.000_043_153_289_611_682_14, 0.053_875_400_870_180_04),
                c(-0.074_307_285_710_030_7, -0.037_228_071_269_956_88),
            ],
        );
        let (us, vt) = truncated_svd_left_absorb(&matrix, 2, 0.0, None).unwrap();
        let reconstructed = us * vt;
        let max_error = matrix
            .iter()
            .zip(reconstructed.iter())
            .map(|(expected, actual)| (*expected - *actual).norm())
            .fold(0.0_f64, f64::max);
        assert!(max_error <= 1e-14, "reconstruction error={max_error:.3e}");
    }

    #[test]
    fn test_zero_singular_value_retains_isometric_factors_in_exact_mode() {
        // With no nonzero total weight, exact mode retains the structural
        // columns. They must still form isometries.
        let matrix = DMatrix::zeros(3, 3);
        let result = truncated_svd_with_error(&matrix, 3, 0.0, Some(0.0)).unwrap();

        assert_eq!(result.singular_values, &[0.0, 0.0, 0.0]);
        let u_gram = result.u.adjoint() * &result.u;
        let v_gram = &result.vt * result.vt.adjoint();
        assert_complex_identity(&u_gram, 1e-14);
        assert_complex_identity(&v_gram, 1e-14);
    }

    #[test]
    fn test_exact_mode_drops_exact_zero_tail() {
        let mut diagonal = DVector::zeros(8);
        diagonal[0] = Complex64::new(3.0, 0.0);
        diagonal[1] = Complex64::new(2.0, 0.0);
        diagonal[2] = Complex64::new(1.0, 0.0);
        let matrix = DMatrix::from_diagonal(&diagonal);
        let result = truncated_svd_with_error(&matrix, 8, 0.0, Some(0.0)).unwrap();

        assert_eq!(result.singular_values.len(), 3);
    }

    #[test]
    fn test_gram_zero_singular_value_columns_are_completed_as_isometries() {
        let matrix = DMatrix::from_row_slice(
            4,
            3,
            &[
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
            ],
        );
        let (u, singular_values, vt) =
            gram_svd_factors(&matrix, numerical_zero_threshold(&matrix)).unwrap();

        assert!(singular_values[2].abs() < f64::MIN_POSITIVE);
        assert_complex_identity(&(u.adjoint() * &u), 1e-14);
        assert_complex_identity(&(&vt * vt.adjoint()), 1e-14);
    }

    #[test]
    fn test_retained_spectrum_is_trustworthy_accepts_valid_and_rejects_bad_triplets() {
        let matrix = matrix_with_spectrum(6, 4, &[4.0, 2.0, 0.5, 0.125]);
        let factors = direct_svd_factors(&matrix).unwrap();
        assert!(retained_spectrum_is_trustworthy(&matrix, &factors, 4));

        let mut corrupted = factors;
        corrupted.1[3] *= 2.0;
        assert!(!retained_spectrum_is_trustworthy(&matrix, &corrupted, 4));
    }

    #[test]
    fn test_issue_580_deep20_gram_factors_need_tail_budgeted_reconstruction_bound() {
        // Captured bit-for-bit at index 143 by a temporary custom deep(20, 2n)
        // census harness configured with cap 128. The shipped
        // canonical_cost_workloads example was not the capture harness.
        // The primary and realified factorizations fail retained-triplet
        // validation. The Gram fallback's retained factors are accurate, but
        // its numerical-null tail exceeds the old factor-error-only bound.
        let (matrix, max_rank, cutoff, max_trunc_error) = issue_580_seed_143_matrix();
        assert_eq!(matrix.shape(), (76, 64));
        let gram = gram_svd_factors(&matrix, numerical_zero_threshold(&matrix)).unwrap();
        let retained_rank = compute_rank(&gram.1, max_rank, cutoff, max_trunc_error);
        assert_eq!(retained_rank, 38);

        let retained_error = retained_block_reconstruction_error(&matrix, &gram, retained_rank);
        let factor_error_tolerance = dimension_scaled_backward_error_tolerance(
            &matrix,
            SVD_FALLBACK_RECONSTRUCTION_MULTIPLIER,
        );
        let tail_budgeted_tolerance =
            retained_block_reconstruction_tolerance(&matrix, &gram, retained_rank);
        assert!(
            retained_error > factor_error_tolerance,
            "mutation guard: a factor-error-only bound must reject the captured factors"
        );
        assert!(
            retained_error <= tail_budgeted_tolerance,
            "the factorization's claimed tail must explain the retained-block residual"
        );
        assert!(
            svd_factors_are_certified(&matrix, &gram, retained_rank),
            "captured retained Gram factors must pass both independent certificates"
        );

        // Reconstruction alone is deliberately insufficient. This corruption
        // stays inside the tail-budgeted reconstruction allowance but violates a
        // retained singular triplet, so removing/loosening that gate is caught.
        let mut corrupted = gram;
        corrupted.1[0] += 1e-10;
        assert!(
            retained_block_reconstruction_error(&matrix, &corrupted, retained_rank)
                <= retained_block_reconstruction_tolerance(&matrix, &corrupted, retained_rank)
        );
        assert!(!svd_factors_are_certified(
            &matrix,
            &corrupted,
            retained_rank
        ));

        truncated_svd_with_error(&matrix, max_rank, cutoff, max_trunc_error).unwrap();
    }

    #[test]
    fn test_issue_580_silent_tail_loss_is_not_certified() {
        // Model a factorization that silently reports a 1e-8-amplitude
        // direction as exactly zero. Exact mode consequently retains rank one
        // and the factorization claims no discarded weight. The retained
        // triplet and isometry certificates alone cannot see the missing tail;
        // the retained-block reconstruction certificate must reject it.
        let mut matrix = DMatrix::zeros(1024, 2);
        matrix[(0, 0)] = Complex64::new(1.0, 0.0);
        matrix[(1, 1)] = Complex64::new(1e-8, 0.0);

        let mut u = DMatrix::zeros(1024, 2);
        u[(0, 0)] = Complex64::new(1.0, 0.0);
        u[(1, 1)] = Complex64::new(1.0, 0.0);
        let factors = (
            u,
            DVector::from_vec(vec![1.0, 0.0]),
            DMatrix::identity(2, 2),
        );
        let retained_rank = compute_rank(&factors.1, 2, 0.0, Some(0.0));
        assert_eq!(retained_rank, 1);
        assert_eq!(
            factorization_discarded_weight(&factors.1, retained_rank).to_bits(),
            0.0_f64.to_bits()
        );
        assert!(retained_spectrum_is_trustworthy(
            &matrix,
            &factors,
            retained_rank
        ));

        let error = retained_block_reconstruction_error(&matrix, &factors, retained_rank);
        let allowance = retained_block_reconstruction_tolerance(&matrix, &factors, retained_rank);
        assert!(
            error > allowance,
            "a zero claimed tail must not explain silently missing tail mass"
        );
        assert!(
            error <= 100.0 * allowance,
            "mutation guard: widening the reconstruction allowance 100x must accept this corruption"
        );
        assert!(!svd_factors_are_certified(&matrix, &factors, retained_rank));
    }

    #[test]
    fn test_realified_svd_matrix_families_have_accurate_orthonormal_factors() {
        // Rectangular in both orientations and square.
        for (rows, cols, spectrum) in [
            (6, 4, vec![4.0, 2.0, 0.5, 0.125]),
            (4, 6, vec![4.0, 2.0, 0.5, 0.125]),
            (4, 4, vec![3.0, 1.5, 0.75, 0.25]),
            // Exactly degenerate and a cluster with a ~1e-13 relative gap.
            (5, 3, vec![2.0, 2.0, 0.25]),
            (5, 3, vec![2.0, 2.0 * (1.0 - 1e-13), 0.25]),
            // A retained spectrum spanning twelve decades.
            (7, 4, vec![1.0, 1e-4, 1e-8, 1e-12]),
            // Exact-zero tail.
            (5, 4, vec![3.0, 1.0, 0.0, 0.0]),
        ] {
            let matrix = matrix_with_spectrum(rows, cols, &spectrum);
            assert_realified_factors(&matrix, &spectrum);
        }

        // The one-dimensional shapes exercise selection and completion without
        // relying on a multi-column Gram-Schmidt basis.
        for (rows, cols, spectrum) in [(1, 6, vec![2.5]), (6, 1, vec![2.5]), (1, 1, vec![2.5])] {
            let matrix = matrix_with_spectrum(rows, cols, &spectrum);
            assert_realified_factors(&matrix, &spectrum);
        }

        let all_zero = DMatrix::zeros(4, 6);
        assert_realified_factors(&all_zero, &[0.0; 4]);

        let real = DMatrix::from_fn(5, 3, |row, column| {
            Complex64::new(
                f64::from(u32::try_from(row * 3 + column + 1).unwrap()).sin(),
                0.0,
            )
        });
        let real_oracle = direct_svd_factors(&real).unwrap().1;
        assert_realified_factors(&real, real_oracle.as_slice());

        let imaginary = real.map(|value| Complex64::new(0.0, value.re));
        assert_realified_factors(&imaginary, real_oracle.as_slice());
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
