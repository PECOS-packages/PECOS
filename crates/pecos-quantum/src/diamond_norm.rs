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

//! Diamond-norm utilities.
//!
//! General channel diamond norm requires solving a semidefinite program. PECOS
//! does not add an external SDP dependency for that. This module exposes exact
//! dependency-free cases that are mathematically closed-form today, plus the
//! linear-algebra pieces needed by a future PECOS-owned general solver.

use std::error::Error;
use std::fmt;

use nalgebra::DMatrix;
use num_complex::Complex64;

use crate::channel::{PauliChannel, basis_bitmask, pauli_basis_len};

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
    /// A Choi matrix does not match the expected input/output dimensions.
    InvalidChoiShape {
        /// Expected row count.
        expected_rows: usize,
        /// Expected column count.
        expected_cols: usize,
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
    },
    /// Input/output dimensions overflowed a `usize` matrix dimension.
    DimensionOverflow {
        /// Input Hilbert-space dimension.
        dim_in: usize,
        /// Output Hilbert-space dimension.
        dim_out: usize,
    },
    /// A matrix entry was not finite.
    NonFiniteEntry,
    /// Two channel representations act on different Hilbert spaces.
    QubitCountMismatch {
        /// Left channel qubit count.
        left: usize,
        /// Right channel qubit count.
        right: usize,
    },
    /// Failed to enumerate the Pauli basis for a channel.
    PauliBasis {
        /// Underlying reason.
        reason: String,
    },
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
            Self::InvalidChoiShape {
                expected_rows,
                expected_cols,
                rows,
                cols,
            } => write!(
                f,
                "invalid Choi matrix shape {rows}x{cols}; expected {expected_rows}x{expected_cols}"
            ),
            Self::DimensionOverflow { dim_in, dim_out } => write!(
                f,
                "Choi input/output dimensions overflow usize: dim_in={dim_in}, dim_out={dim_out}"
            ),
            Self::NonFiniteEntry => write!(f, "matrix contains a non-finite entry"),
            Self::QubitCountMismatch { left, right } => write!(
                f,
                "channels must act on the same number of qubits, got {left} and {right}"
            ),
            Self::PauliBasis { reason } => {
                write!(f, "failed to enumerate Pauli basis: {reason}")
            }
        }
    }
}

impl Error for DiamondNormError {}

/// Returns `||left - right||_diamond` for two Pauli channels.
///
/// For Pauli channels, the diamond norm of the channel difference is exactly
/// the L1 distance between the two Pauli probability vectors. Applying the
/// channel difference to half of a maximally entangled state produces
/// orthogonal Pauli-labelled Bell states, so no SDP is needed.
///
/// # Errors
///
/// Returns an error if the channels act on different numbers of qubits or the
/// Pauli basis size overflows.
pub fn pauli_channel_diamond_norm(
    left: &PauliChannel,
    right: &PauliChannel,
) -> Result<f64, DiamondNormError> {
    if left.num_qubits() != right.num_qubits() {
        return Err(DiamondNormError::QubitCountMismatch {
            left: left.num_qubits(),
            right: right.num_qubits(),
        });
    }
    let num_qubits = left.num_qubits();
    let basis_len = pauli_basis_len(num_qubits).map_err(|err| DiamondNormError::PauliBasis {
        reason: err.to_string(),
    })?;
    let mut total = 0.0;
    for basis_index in 0..basis_len {
        let pauli =
            basis_bitmask(num_qubits, basis_index).map_err(|err| DiamondNormError::PauliBasis {
                reason: err.to_string(),
            })?;
        total += (left.probability(&pauli) - right.probability(&pauli)).abs();
    }
    Ok(total)
}

/// Returns the diamond distance between two Pauli channels.
///
/// The diamond distance is `0.5 * ||left - right||_diamond`, matching the
/// standard trace-distance normalization.
///
/// # Errors
///
/// Returns an error if [`pauli_channel_diamond_norm`] fails.
pub fn pauli_channel_diamond_distance(
    left: &PauliChannel,
    right: &PauliChannel,
) -> Result<f64, DiamondNormError> {
    Ok(0.5 * pauli_channel_diamond_norm(left, right)?)
}

/// Returns the length of the scaled upper-triangular vector for an `n x n`
/// symmetric matrix.
#[must_use]
pub const fn scaled_psd_triangle_len(n: usize) -> usize {
    n * (n + 1) / 2
}

/// Converts a real symmetric matrix to scaled upper-triangular
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

/// Converts scaled upper-triangular vector form back to a real
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
/// This embedding maps complex PSD constraints into real PSD constraints, which
/// is the representation needed by a real-valued PECOS SDP implementation.
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

/// Converts PECOS's column-stacked Choi convention to the transposed
/// row-vector convention used by the Watrous diamond-norm SDP objective.
///
/// PECOS indexes Choi rows and columns as `output + input * dim_output`.
/// This helper performs the convention transform used when assembling the
/// row-vector form of the Watrous SDP:
///
/// ```text
/// reshape(J, (dim_in, dim_out, dim_in, dim_out))
/// transpose axes (3, 2, 1, 0)
/// reshape back to a matrix
/// ```
///
/// The result is not a public diamond-norm implementation. It is a tested
/// convention helper for the future PECOS-owned SDP assembly.
///
/// # Errors
///
/// Returns an error if `choi` is not `(dim_in * dim_out) x (dim_in * dim_out)`
/// or if it contains a non-finite entry.
pub fn choi_to_watrous_row_transpose(
    choi: &DMatrix<Complex64>,
    dim_in: usize,
    dim_out: usize,
) -> Result<DMatrix<Complex64>, DiamondNormError> {
    let size = dim_in
        .checked_mul(dim_out)
        .ok_or(DiamondNormError::DimensionOverflow { dim_in, dim_out })?;
    if choi.nrows() != size || choi.ncols() != size {
        return Err(DiamondNormError::InvalidChoiShape {
            expected_rows: size,
            expected_cols: size,
            rows: choi.nrows(),
            cols: choi.ncols(),
        });
    }

    let mut out = DMatrix::zeros(size, size);
    for input_row in 0..dim_in {
        for output_row in 0..dim_out {
            let src_row = output_row + input_row * dim_out;
            for input_col in 0..dim_in {
                for output_col in 0..dim_out {
                    let src_col = output_col + input_col * dim_out;
                    let value = choi[(src_row, src_col)];
                    if !value.re.is_finite() || !value.im.is_finite() {
                        return Err(DiamondNormError::NonFiniteEntry);
                    }
                    let dst_row = input_col + output_col * dim_in;
                    let dst_col = input_row + output_row * dim_in;
                    out[(dst_row, dst_col)] = value;
                }
            }
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
    use std::collections::BTreeMap;

    fn assert_close(left: f64, right: f64) {
        assert!((left - right).abs() < 1e-12, "{left} != {right}");
    }

    #[test]
    fn pauli_channel_diamond_norm_is_l1_probability_distance() {
        let left = PauliChannel::one_qubit(0.1, 0.2, 0.0).unwrap();
        let right = PauliChannel::one_qubit(0.0, 0.2, 0.3).unwrap();

        assert_close(pauli_channel_diamond_norm(&left, &right).unwrap(), 0.6);
        assert_close(pauli_channel_diamond_distance(&left, &right).unwrap(), 0.3);
    }

    #[test]
    fn pauli_channel_diamond_norm_includes_absent_terms_as_zero() {
        let mut left_probs = BTreeMap::new();
        left_probs.insert(basis_bitmask(2, 0).unwrap(), 0.9);
        left_probs.insert(basis_bitmask(2, 5).unwrap(), 0.1);
        let left = PauliChannel::try_new(2, left_probs).unwrap();

        let mut right_probs = BTreeMap::new();
        right_probs.insert(basis_bitmask(2, 0).unwrap(), 0.8);
        right_probs.insert(basis_bitmask(2, 10).unwrap(), 0.2);
        let right = PauliChannel::try_new(2, right_probs).unwrap();

        assert_close(pauli_channel_diamond_norm(&left, &right).unwrap(), 0.4);
    }

    #[test]
    fn pauli_channel_diamond_norm_handles_sparse_three_qubit_channels() {
        let mut left_probs = BTreeMap::new();
        left_probs.insert(basis_bitmask(3, 0).unwrap(), 0.7);
        left_probs.insert(basis_bitmask(3, 1).unwrap(), 0.1);
        left_probs.insert(basis_bitmask(3, 17).unwrap(), 0.2);
        let left = PauliChannel::try_new(3, left_probs).unwrap();

        let mut right_probs = BTreeMap::new();
        right_probs.insert(basis_bitmask(3, 0).unwrap(), 0.6);
        right_probs.insert(basis_bitmask(3, 17).unwrap(), 0.1);
        right_probs.insert(basis_bitmask(3, 63).unwrap(), 0.3);
        let right = PauliChannel::try_new(3, right_probs).unwrap();

        assert_close(pauli_channel_diamond_norm(&left, &right).unwrap(), 0.6);
        assert_close(pauli_channel_diamond_distance(&left, &right).unwrap(), 0.3);
    }

    #[test]
    fn pauli_channel_diamond_norm_rejects_qubit_count_mismatch() {
        let left = PauliChannel::one_qubit(0.1, 0.0, 0.0).unwrap();
        let mut right_probs = BTreeMap::new();
        right_probs.insert(basis_bitmask(2, 0).unwrap(), 1.0);
        let right = PauliChannel::try_new(2, right_probs).unwrap();

        assert!(matches!(
            pauli_channel_diamond_norm(&left, &right).unwrap_err(),
            DiamondNormError::QubitCountMismatch { left: 1, right: 2 }
        ));
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
    fn choi_to_watrous_row_transpose_matches_reference_axis_permutation() {
        let choi = DMatrix::from_row_slice(
            4,
            4,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(7.0, 0.0),
                Complex64::new(8.0, 0.0),
                Complex64::new(9.0, 0.0),
                Complex64::new(10.0, 0.0),
                Complex64::new(11.0, 0.0),
                Complex64::new(12.0, 0.0),
                Complex64::new(13.0, 0.0),
                Complex64::new(14.0, 0.0),
                Complex64::new(15.0, 0.0),
            ],
        );
        let converted = choi_to_watrous_row_transpose(&choi, 2, 2).unwrap();
        let expected = DMatrix::from_row_slice(
            4,
            4,
            &[
                Complex64::new(0.0, 0.0),
                Complex64::new(8.0, 0.0),
                Complex64::new(4.0, 0.0),
                Complex64::new(12.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(10.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(14.0, 0.0),
                Complex64::new(1.0, 0.0),
                Complex64::new(9.0, 0.0),
                Complex64::new(5.0, 0.0),
                Complex64::new(13.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(11.0, 0.0),
                Complex64::new(7.0, 0.0),
                Complex64::new(15.0, 0.0),
            ],
        );

        assert_eq!(converted, expected);
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

        assert!(matches!(
            choi_to_watrous_row_transpose(&DMatrix::zeros(2, 2), 2, 2).unwrap_err(),
            DiamondNormError::InvalidChoiShape { .. }
        ));
    }
}
