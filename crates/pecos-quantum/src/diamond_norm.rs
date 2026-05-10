// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Internal linear-algebra helpers for a future Rust diamond-norm solver.
//!
//! This module deliberately does not expose a public `diamond_norm` routine or
//! add an SDP solver dependency yet. It contains the convention-sensitive pieces
//! needed before a feature-gated conic-solver integration is reviewable:
//! Clarabel-style scaled triangular vectorization for real PSD cones and the
//! standard complex-Hermitian to real-symmetric embedding.

use std::error::Error;
use std::fmt;

use nalgebra::DMatrix;
use num_complex::Complex64;

const DEFAULT_TOLERANCE: f64 = 1e-12;

/// Error returned by diamond-norm linear-algebra helpers.
#[derive(Debug, Clone, PartialEq)]
pub enum DiamondNormError {
    /// A matrix was not square.
    NonSquareMatrix {
        /// Row count.
        rows: usize,
        /// Column count.
        cols: usize,
    },
    /// A scaled-vector input had the wrong length for the requested matrix.
    InvalidSvecLength {
        /// Expected triangular-vector length.
        expected: usize,
        /// Actual length.
        actual: usize,
    },
    /// A matrix expected to be symmetric or Hermitian was not within tolerance.
    NonHermitian {
        /// Maximum observed entrywise difference from the adjoint/symmetric
        /// counterpart.
        max_difference: f64,
        /// Allowed tolerance.
        tolerance: f64,
    },
    /// A matrix entry was not finite.
    NonFiniteEntry,
}

impl fmt::Display for DiamondNormError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NonSquareMatrix { rows, cols } => {
                write!(f, "matrix must be square, got {rows}x{cols}")
            }
            Self::InvalidSvecLength { expected, actual } => write!(
                f,
                "invalid scaled-triangle vector length {actual}; expected {expected}"
            ),
            Self::NonHermitian {
                max_difference,
                tolerance,
            } => write!(
                f,
                "matrix is not Hermitian/symmetric within tolerance {tolerance}; max difference {max_difference}"
            ),
            Self::NonFiniteEntry => write!(f, "matrix contains a non-finite entry"),
        }
    }
}

impl Error for DiamondNormError {}

/// Returns the length of the scaled upper-triangular vector for an `n x n`
/// symmetric matrix.
#[must_use]
pub const fn scaled_psd_triangle_len(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Converts a real symmetric matrix to Clarabel-style scaled upper-triangular
/// vector form.
///
/// Diagonal entries are stored unchanged. Strict upper-triangular entries are
/// multiplied by `sqrt(2)`, preserving Frobenius inner products under vector
/// dot products.
///
/// # Errors
///
/// Returns an error when `matrix` is not square, contains non-finite values, or
/// is not symmetric within the default tolerance.
pub fn svec_real_symmetric(matrix: &DMatrix<f64>) -> Result<Vec<f64>, DiamondNormError> {
    svec_real_symmetric_with_tolerance(matrix, DEFAULT_TOLERANCE)
}

/// Converts a real symmetric matrix to scaled upper-triangular vector form
/// with explicit symmetry tolerance.
///
/// # Errors
///
/// Returns an error when `matrix` is not square, contains non-finite values, or
/// is not symmetric within `tolerance`.
pub fn svec_real_symmetric_with_tolerance(
    matrix: &DMatrix<f64>,
    tolerance: f64,
) -> Result<Vec<f64>, DiamondNormError> {
    validate_real_symmetric(matrix, tolerance)?;
    let n = matrix.nrows();
    let sqrt2 = 2.0_f64.sqrt();
    let mut out = Vec::with_capacity(scaled_psd_triangle_len(n));
    for col in 0..n {
        for row in 0..=col {
            let scale = if row == col { 1.0 } else { sqrt2 };
            out.push(matrix[(row, col)] * scale);
        }
    }
    Ok(out)
}

/// Converts Clarabel-style scaled upper-triangular vector form back to a real
/// symmetric matrix.
///
/// # Errors
///
/// Returns an error when `data.len()` is not `n * (n + 1) / 2` or a data entry
/// is not finite.
pub fn smat_real_symmetric(n: usize, data: &[f64]) -> Result<DMatrix<f64>, DiamondNormError> {
    let expected = scaled_psd_triangle_len(n);
    if data.len() != expected {
        return Err(DiamondNormError::InvalidSvecLength {
            expected,
            actual: data.len(),
        });
    }
    let sqrt2 = 2.0_f64.sqrt();
    let mut out = DMatrix::zeros(n, n);
    let mut idx = 0;
    for col in 0..n {
        for row in 0..=col {
            let value = data[idx];
            if !value.is_finite() {
                return Err(DiamondNormError::NonFiniteEntry);
            }
            let unscaled = if row == col { value } else { value / sqrt2 };
            out[(row, col)] = unscaled;
            out[(col, row)] = unscaled;
            idx += 1;
        }
    }
    Ok(out)
}

/// Embeds a complex Hermitian matrix `A = X + iY` as a real symmetric matrix:
///
/// ```text
/// [ X  -Y ]
/// [ Y   X ]
/// ```
///
/// This embedding maps complex PSD constraints into real PSD constraints and is
/// the representation needed before modeling the diamond-norm SDP in a real
/// conic solver.
///
/// # Errors
///
/// Returns an error when `matrix` is not square, contains non-finite values, or
/// is not Hermitian within the default tolerance.
pub fn hermitian_to_real_symmetric(
    matrix: &DMatrix<Complex64>,
) -> Result<DMatrix<f64>, DiamondNormError> {
    hermitian_to_real_symmetric_with_tolerance(matrix, DEFAULT_TOLERANCE)
}

/// Hermitian-to-real-symmetric embedding with explicit tolerance.
///
/// # Errors
///
/// Returns an error when `matrix` is not square, contains non-finite values, or
/// is not Hermitian within `tolerance`.
pub fn hermitian_to_real_symmetric_with_tolerance(
    matrix: &DMatrix<Complex64>,
    tolerance: f64,
) -> Result<DMatrix<f64>, DiamondNormError> {
    validate_hermitian(matrix, tolerance)?;
    let n = matrix.nrows();
    let mut out = DMatrix::zeros(2 * n, 2 * n);
    for row in 0..n {
        for col in 0..n {
            let value = matrix[(row, col)];
            out[(row, col)] = value.re;
            out[(row, col + n)] = -value.im;
            out[(row + n, col)] = value.im;
            out[(row + n, col + n)] = value.re;
        }
    }
    Ok(out)
}

fn validate_real_symmetric(matrix: &DMatrix<f64>, tolerance: f64) -> Result<(), DiamondNormError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DiamondNormError::NonSquareMatrix {
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }
    let mut max_difference: f64 = 0.0;
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            let value = matrix[(row, col)];
            if !value.is_finite() {
                return Err(DiamondNormError::NonFiniteEntry);
            }
            max_difference = max_difference.max((value - matrix[(col, row)]).abs());
        }
    }
    if max_difference > tolerance {
        return Err(DiamondNormError::NonHermitian {
            max_difference,
            tolerance,
        });
    }
    Ok(())
}

fn validate_hermitian(matrix: &DMatrix<Complex64>, tolerance: f64) -> Result<(), DiamondNormError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(DiamondNormError::NonSquareMatrix {
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }
    let mut max_difference: f64 = 0.0;
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            let value = matrix[(row, col)];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(DiamondNormError::NonFiniteEntry);
            }
            max_difference = max_difference.max((value - matrix[(col, row)].conj()).norm());
        }
    }
    if max_difference > tolerance {
        return Err(DiamondNormError::NonHermitian {
            max_difference,
            tolerance,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-12, "{left} != {right}");
    }

    fn frobenius_inner(left: &DMatrix<f64>, right: &DMatrix<f64>) -> f64 {
        left.iter().zip(right.iter()).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn scaled_triangle_round_trips_real_symmetric_matrix() {
        let matrix =
            DMatrix::from_row_slice(3, 3, &[1.0, 2.0, -3.0, 2.0, 5.0, 7.0, -3.0, 7.0, 11.0]);
        let packed = svec_real_symmetric(&matrix).unwrap();
        assert_eq!(packed.len(), 6);
        let recovered = smat_real_symmetric(3, &packed).unwrap();
        for row in 0..3 {
            for col in 0..3 {
                assert_close(recovered[(row, col)], matrix[(row, col)]);
            }
        }
    }

    #[test]
    fn scaled_triangle_preserves_frobenius_inner_product() {
        let a = DMatrix::from_row_slice(2, 2, &[1.0, 3.0, 3.0, 2.0]);
        let b = DMatrix::from_row_slice(2, 2, &[5.0, -7.0, -7.0, 11.0]);
        let a_vec = svec_real_symmetric(&a).unwrap();
        let b_vec = svec_real_symmetric(&b).unwrap();
        let vector_inner: f64 = a_vec.iter().zip(b_vec.iter()).map(|(x, y)| x * y).sum();

        assert_close(vector_inner, frobenius_inner(&a, &b));
    }

    #[test]
    fn hermitian_embedding_is_real_symmetric_and_trace_scaled() {
        let i = Complex64::new(0.0, 1.0);
        let matrix = DMatrix::from_row_slice(
            2,
            2,
            &[Complex64::new(2.0, 0.0), i, -i, Complex64::new(3.0, 0.0)],
        );
        let embedded = hermitian_to_real_symmetric(&matrix).unwrap();

        assert_eq!(embedded.shape(), (4, 4));
        for row in 0..4 {
            for col in 0..4 {
                assert_close(embedded[(row, col)], embedded[(col, row)]);
            }
        }
        assert_close(embedded.trace(), 10.0);
    }

    #[test]
    fn helper_validation_rejects_invalid_inputs() {
        assert!(matches!(
            svec_real_symmetric(&DMatrix::zeros(2, 3)).unwrap_err(),
            DiamondNormError::NonSquareMatrix { .. }
        ));

        let nonsymmetric = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        assert!(matches!(
            svec_real_symmetric(&nonsymmetric).unwrap_err(),
            DiamondNormError::NonHermitian { .. }
        ));

        assert!(matches!(
            smat_real_symmetric(3, &[1.0, 2.0]).unwrap_err(),
            DiamondNormError::InvalidSvecLength { .. }
        ));

        let nonhermitian = DMatrix::from_row_slice(
            2,
            2,
            &[
                Complex64::new(1.0, 0.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(1.0, 1.0),
                Complex64::new(1.0, 0.0),
            ],
        );
        assert!(matches!(
            hermitian_to_real_symmetric(&nonhermitian).unwrap_err(),
            DiamondNormError::NonHermitian { .. }
        ));
    }
}
