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

//! Standalone quantum information measures.
//!
//! Measures are free functions so they can be shared across simulator and
//! representation types without forcing every backend into a single state API.

use std::error::Error;
use std::fmt;

use nalgebra::{DMatrix, DVector, SVD};
use num_complex::Complex64;

use crate::channel::Ptm;

const DEFAULT_TOLERANCE: f64 = 1e-12;

/// Error returned by quantum-information measure functions.
#[derive(Debug, Clone, PartialEq)]
pub enum MeasureError {
    /// The requested Hilbert-space dimension would overflow `usize`.
    DimensionOverflow {
        /// Number of qubits supplied by the caller.
        num_qubits: usize,
    },
    /// Two vectors have incompatible lengths.
    VectorLengthMismatch {
        /// Left vector length.
        left: usize,
        /// Right vector length.
        right: usize,
    },
    /// A matrix is not square.
    NonSquareMatrix {
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
    },
    /// A matrix does not have the expected shape.
    InvalidMatrixShape {
        /// Expected row count.
        expected_rows: usize,
        /// Expected column count.
        expected_cols: usize,
        /// Actual row count.
        rows: usize,
        /// Actual column count.
        cols: usize,
    },
    /// A value is not finite.
    NonFiniteValue {
        /// Offending value.
        value: Complex64,
    },
    /// A state vector is not normalized.
    InvalidStateNorm {
        /// Observed squared norm.
        norm_sqr: f64,
        /// Allowed absolute tolerance.
        tolerance: f64,
    },
    /// A density matrix is not Hermitian within tolerance.
    NonHermitianMatrix {
        /// Row index where the mismatch was observed.
        row: usize,
        /// Column index where the mismatch was observed.
        col: usize,
        /// Observed entry.
        value: Complex64,
        /// Conjugate-transposed entry.
        adjoint_value: Complex64,
        /// Allowed absolute tolerance.
        tolerance: f64,
    },
    /// A density matrix does not have trace one.
    InvalidDensityTrace {
        /// Observed trace.
        trace: Complex64,
        /// Allowed absolute tolerance.
        tolerance: f64,
    },
    /// The requested logarithm base is invalid for entropy.
    InvalidEntropyBase {
        /// Invalid base.
        base: f64,
    },
}

impl fmt::Display for MeasureError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionOverflow { num_qubits } => {
                write!(
                    f,
                    "Hilbert-space dimension overflows usize for {num_qubits} qubits"
                )
            }
            Self::VectorLengthMismatch { left, right } => {
                write!(f, "vector length mismatch: {left} != {right}")
            }
            Self::NonSquareMatrix { rows, cols } => {
                write!(f, "matrix must be square, got {rows}x{cols}")
            }
            Self::InvalidMatrixShape {
                expected_rows,
                expected_cols,
                rows,
                cols,
            } => write!(
                f,
                "invalid matrix shape {rows}x{cols}; expected {expected_rows}x{expected_cols}"
            ),
            Self::NonFiniteValue { value } => write!(f, "non-finite value: {value}"),
            Self::InvalidStateNorm {
                norm_sqr,
                tolerance,
            } => write!(
                f,
                "state vector squared norm must be 1 within tolerance {tolerance}, got {norm_sqr}"
            ),
            Self::NonHermitianMatrix {
                row,
                col,
                value,
                adjoint_value,
                tolerance,
            } => write!(
                f,
                "matrix is not Hermitian within tolerance {tolerance} at ({row}, {col}): {value} != {adjoint_value}"
            ),
            Self::InvalidDensityTrace { trace, tolerance } => write!(
                f,
                "density matrix trace must be 1 within tolerance {tolerance}, got {trace}"
            ),
            Self::InvalidEntropyBase { base } => {
                write!(
                    f,
                    "entropy logarithm base must be finite, positive, and not 1; got {base}"
                )
            }
        }
    }
}

impl Error for MeasureError {}

/// Returns pure-state fidelity `|<left|right>|^2`.
///
/// Both state vectors must have the same length and be normalized.
///
/// # Errors
///
/// Returns an error when lengths differ, entries are non-finite, or either
/// vector is not normalized within tolerance.
pub fn state_fidelity(
    left: &DVector<Complex64>,
    right: &DVector<Complex64>,
) -> Result<f64, MeasureError> {
    if left.len() != right.len() {
        return Err(MeasureError::VectorLengthMismatch {
            left: left.len(),
            right: right.len(),
        });
    }
    validate_state_vector(left)?;
    validate_state_vector(right)?;
    let overlap: Complex64 = left
        .iter()
        .zip(right.iter())
        .map(|(left, right)| left.conj() * right)
        .sum();
    Ok(overlap.norm_sqr())
}

/// Returns fidelity `<psi|rho|psi>` between a density matrix and a pure state.
///
/// `rho` must be a trace-one Hermitian density matrix and `psi` must be a
/// normalized state vector with matching dimension. Positive-semidefinite
/// validation is intentionally not part of this cheap structural check.
///
/// # Errors
///
/// Returns an error when dimensions differ or either input is structurally
/// invalid.
pub fn state_fidelity_with_density_matrix(
    rho: &DMatrix<Complex64>,
    psi: &DVector<Complex64>,
) -> Result<f64, MeasureError> {
    validate_density_matrix(rho)?;
    validate_state_vector(psi)?;
    if rho.nrows() != psi.len() {
        return Err(MeasureError::InvalidMatrixShape {
            expected_rows: psi.len(),
            expected_cols: psi.len(),
            rows: rho.nrows(),
            cols: rho.ncols(),
        });
    }
    let evolved = rho * psi;
    let value: Complex64 = psi
        .iter()
        .zip(evolved.iter())
        .map(|(left, right)| left.conj() * right)
        .sum();
    if value.im.abs() > DEFAULT_TOLERANCE {
        return Err(MeasureError::NonFiniteValue { value });
    }
    Ok(value.re)
}

/// Returns density-matrix purity `Tr(rho^2)`.
///
/// # Errors
///
/// Returns an error when `rho` is not square, finite, Hermitian, and trace one.
pub fn purity(rho: &DMatrix<Complex64>) -> Result<f64, MeasureError> {
    validate_density_matrix(rho)?;
    let value = trace(&(rho * rho));
    if value.im.abs() > DEFAULT_TOLERANCE {
        return Err(MeasureError::NonFiniteValue { value });
    }
    Ok(value.re)
}

/// Returns the von Neumann entropy `-Tr(rho log_2 rho)`.
///
/// `rho` must be a positive-semidefinite density matrix. This function
/// validates the cheap structural conditions (square, finite, Hermitian,
/// trace one) and computes the entropy from singular values, which equal the
/// eigenvalues for valid density matrices.
///
/// # Errors
///
/// Returns an error when `rho` is structurally invalid.
pub fn entropy(rho: &DMatrix<Complex64>) -> Result<f64, MeasureError> {
    entropy_with_base(rho, 2.0)
}

/// Returns the von Neumann entropy `-Tr(rho log_base rho)`.
///
/// # Errors
///
/// Returns an error when `rho` is structurally invalid or `base` is not finite,
/// positive, and different from one.
pub fn entropy_with_base(rho: &DMatrix<Complex64>, base: f64) -> Result<f64, MeasureError> {
    validate_density_matrix(rho)?;
    validate_entropy_base(base)?;
    let svd = SVD::new(rho.clone(), false, false);
    let log_base = base.ln();
    Ok(svd
        .singular_values
        .iter()
        .copied()
        .filter(|lambda| *lambda > DEFAULT_TOLERANCE)
        .map(|lambda| -lambda * lambda.ln() / log_base)
        .sum())
}

/// Returns normalized process fidelity between two PTMs.
///
/// With PECOS's normalized Pauli basis convention, this is
/// `Tr(R_left^T R_right) / 4^n`. Identity compared with identity gives 1.
///
/// # Errors
///
/// Returns an error when the PTMs have different qubit counts.
pub fn process_fidelity(left: &Ptm, right: &Ptm) -> Result<f64, MeasureError> {
    if left.num_qubits() != right.num_qubits() {
        return Err(MeasureError::InvalidMatrixShape {
            expected_rows: left.matrix().nrows(),
            expected_cols: left.matrix().ncols(),
            rows: right.matrix().nrows(),
            cols: right.matrix().ncols(),
        });
    }
    #[allow(clippy::cast_precision_loss)]
    let basis_len = left.matrix().nrows() as f64;
    let value: f64 = left
        .matrix()
        .iter()
        .zip(right.matrix().iter())
        .map(|(left, right)| left * right)
        .sum::<f64>()
        / basis_len;
    Ok(value)
}

/// Returns average gate fidelity between two PTMs.
///
/// This uses `F_avg = (d F_process + 1) / (d + 1)` for Hilbert-space
/// dimension `d = 2^n`.
///
/// # Errors
///
/// Returns an error when the PTMs have different qubit counts or the Hilbert
/// dimension overflows.
pub fn average_gate_fidelity(left: &Ptm, right: &Ptm) -> Result<f64, MeasureError> {
    let process = process_fidelity(left, right)?;
    let dim = hilbert_dim(left.num_qubits())?;
    #[allow(clippy::cast_precision_loss)]
    let dim = dim as f64;
    Ok((dim * process + 1.0) / (dim + 1.0))
}

/// Returns average gate error `1 - average_gate_fidelity`.
///
/// # Errors
///
/// Returns an error when [`average_gate_fidelity`] fails.
pub fn gate_error(left: &Ptm, right: &Ptm) -> Result<f64, MeasureError> {
    Ok(1.0 - average_gate_fidelity(left, right)?)
}

fn validate_state_vector(vector: &DVector<Complex64>) -> Result<(), MeasureError> {
    let mut norm_sqr = 0.0;
    for value in vector.iter() {
        validate_complex(*value)?;
        norm_sqr += value.norm_sqr();
    }
    if (norm_sqr - 1.0).abs() > DEFAULT_TOLERANCE {
        return Err(MeasureError::InvalidStateNorm {
            norm_sqr,
            tolerance: DEFAULT_TOLERANCE,
        });
    }
    Ok(())
}

fn validate_density_matrix(matrix: &DMatrix<Complex64>) -> Result<(), MeasureError> {
    if matrix.nrows() != matrix.ncols() {
        return Err(MeasureError::NonSquareMatrix {
            rows: matrix.nrows(),
            cols: matrix.ncols(),
        });
    }
    for value in matrix.iter() {
        validate_complex(*value)?;
    }
    for row in 0..matrix.nrows() {
        for col in 0..matrix.ncols() {
            let value = matrix[(row, col)];
            let adjoint_value = matrix[(col, row)].conj();
            if (value - adjoint_value).norm() > DEFAULT_TOLERANCE {
                return Err(MeasureError::NonHermitianMatrix {
                    row,
                    col,
                    value,
                    adjoint_value,
                    tolerance: DEFAULT_TOLERANCE,
                });
            }
        }
    }
    let trace = trace(matrix);
    if trace.im.abs() > DEFAULT_TOLERANCE || (trace.re - 1.0).abs() > DEFAULT_TOLERANCE {
        return Err(MeasureError::InvalidDensityTrace {
            trace,
            tolerance: DEFAULT_TOLERANCE,
        });
    }
    Ok(())
}

fn validate_complex(value: Complex64) -> Result<(), MeasureError> {
    if value.re.is_finite() && value.im.is_finite() {
        Ok(())
    } else {
        Err(MeasureError::NonFiniteValue { value })
    }
}

fn validate_entropy_base(base: f64) -> Result<(), MeasureError> {
    if base.is_finite() && base > 0.0 && (base - 1.0).abs() > DEFAULT_TOLERANCE {
        Ok(())
    } else {
        Err(MeasureError::InvalidEntropyBase { base })
    }
}

fn trace(matrix: &DMatrix<Complex64>) -> Complex64 {
    let n = matrix.nrows().min(matrix.ncols());
    (0..n).map(|idx| matrix[(idx, idx)]).sum()
}

fn hilbert_dim(num_qubits: usize) -> Result<usize, MeasureError> {
    2usize
        .checked_pow(
            num_qubits
                .try_into()
                .map_err(|_| MeasureError::DimensionOverflow { num_qubits })?,
        )
        .ok_or(MeasureError::DimensionOverflow { num_qubits })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channel::Ptm;
    use pecos_core::{Op, op};

    fn assert_close(a: f64, b: f64) {
        assert!((a - b).abs() < 1e-10, "{a} != {b}");
    }

    fn ket(values: &[Complex64]) -> DVector<Complex64> {
        DVector::from_column_slice(values)
    }

    fn pure_density(psi: &DVector<Complex64>) -> DMatrix<Complex64> {
        psi * psi.adjoint()
    }

    #[test]
    fn pure_state_fidelity_matches_known_values() {
        let zero = ket(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let one = ket(&[Complex64::new(0.0, 0.0), Complex64::new(1.0, 0.0)]);
        let plus = ket(&[
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]);

        assert_close(state_fidelity(&zero, &zero).unwrap(), 1.0);
        assert_close(state_fidelity(&zero, &one).unwrap(), 0.0);
        assert_close(state_fidelity(&zero, &plus).unwrap(), 0.5);
    }

    #[test]
    fn state_fidelity_rejects_unnormalized_vectors() {
        let zero = ket(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let bad = ket(&[Complex64::new(1.0, 0.0), Complex64::new(1.0, 0.0)]);

        assert!(matches!(
            state_fidelity(&zero, &bad).unwrap_err(),
            MeasureError::InvalidStateNorm { .. }
        ));
    }

    #[test]
    fn density_matrix_purity_and_entropy_match_known_states() {
        let zero = ket(&[Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]);
        let pure = pure_density(&zero);
        let half = Complex64::new(0.5, 0.0);
        let mixed = DMatrix::from_diagonal_element(2, 2, half);

        assert_close(purity(&pure).unwrap(), 1.0);
        assert_close(entropy(&pure).unwrap(), 0.0);
        assert_close(purity(&mixed).unwrap(), 0.5);
        assert_close(entropy(&mixed).unwrap(), 1.0);
        assert_close(
            state_fidelity_with_density_matrix(&mixed, &zero).unwrap(),
            0.5,
        );
    }

    #[test]
    fn density_matrix_measures_reject_invalid_matrices() {
        let non_square = DMatrix::from_element(2, 3, Complex64::new(0.0, 0.0));
        assert!(matches!(
            purity(&non_square).unwrap_err(),
            MeasureError::NonSquareMatrix { .. }
        ));

        let mut non_hermitian = DMatrix::zeros(2, 2);
        non_hermitian[(0, 0)] = Complex64::new(1.0, 0.0);
        non_hermitian[(0, 1)] = Complex64::new(0.1, 0.0);
        assert!(matches!(
            purity(&non_hermitian).unwrap_err(),
            MeasureError::NonHermitianMatrix { .. }
        ));
    }

    #[test]
    fn process_and_average_gate_fidelity_match_depolarizing_channel() {
        let identity = Ptm::identity(1).unwrap();
        let Op::Channel(expr) = op::Depolarizing(0.3, 0) else {
            panic!("expected channel");
        };
        let depolarizing = Ptm::from_channel_expr(&expr).unwrap();

        assert_close(process_fidelity(&identity, &identity).unwrap(), 1.0);
        assert_close(process_fidelity(&depolarizing, &identity).unwrap(), 0.7);
        assert_close(
            average_gate_fidelity(&depolarizing, &identity).unwrap(),
            0.8,
        );
        assert_close(gate_error(&depolarizing, &identity).unwrap(), 0.2);
    }
}
