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

use nalgebra::{DMatrix, DVector, SVD, Schur};
use num_complex::Complex64;

use crate::channel::Ptm;

const DEFAULT_TOLERANCE: f64 = 1e-12;

/// One term in a Schmidt decomposition.
///
/// The tuple is `(coefficient, left_vector, right_vector)`.
pub type SchmidtTerm = (f64, Vec<Complex64>, Vec<Complex64>);

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
    /// Two channel/process representations have incompatible qubit counts.
    QubitCountMismatch {
        /// Expected qubit count.
        expected: usize,
        /// Actual qubit count.
        actual: usize,
    },
    /// A value is not finite.
    NonFiniteValue {
        /// Offending value.
        value: Complex64,
    },
    /// A finite complex value was expected to be real within tolerance.
    NonRealValue {
        /// Offending value.
        value: Complex64,
        /// Allowed imaginary-part tolerance.
        tolerance: f64,
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
    /// A probability is negative or non-finite.
    InvalidProbability {
        /// Index of the invalid probability.
        index: usize,
        /// Invalid probability value.
        probability: f64,
    },
    /// A probability distribution does not sum to one.
    InvalidProbabilitySum {
        /// Observed probability sum.
        sum: f64,
        /// Allowed absolute tolerance.
        tolerance: f64,
    },
    /// Subsystem dimensions are invalid for a multipartite measure.
    InvalidSubsystemDimensions {
        /// Subsystem dimensions supplied by the caller.
        dims: Vec<usize>,
        /// Actual Hilbert-space dimension of the density matrix.
        matrix_dim: usize,
    },
    /// A subsystem index is outside the supplied tensor-factor list.
    SubsystemOutOfRange {
        /// Number of subsystems supplied by the caller.
        num_subsystems: usize,
        /// Invalid subsystem index.
        subsystem: usize,
    },
    /// A subsystem was listed more than once.
    DuplicateSubsystem {
        /// Repeated subsystem index.
        subsystem: usize,
    },
    /// An eigendecomposition did not converge.
    EigenDecompositionFailed,
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
            Self::QubitCountMismatch { expected, actual } => {
                write!(f, "qubit count mismatch: expected {expected}, got {actual}")
            }
            Self::NonFiniteValue { value } => write!(f, "non-finite value: {value}"),
            Self::NonRealValue { value, tolerance } => write!(
                f,
                "value must be real within tolerance {tolerance}, got {value}"
            ),
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
            Self::InvalidProbability { index, probability } => write!(
                f,
                "probability at index {index} must be finite and non-negative, got {probability}"
            ),
            Self::InvalidProbabilitySum { sum, tolerance } => write!(
                f,
                "probability distribution must sum to 1 within tolerance {tolerance}, got {sum}"
            ),
            Self::InvalidSubsystemDimensions { dims, matrix_dim } => write!(
                f,
                "invalid subsystem dimensions {dims:?} for density matrix dimension {matrix_dim}"
            ),
            Self::SubsystemOutOfRange {
                num_subsystems,
                subsystem,
            } => write!(
                f,
                "subsystem {subsystem} is outside the {num_subsystems}-subsystem tensor product"
            ),
            Self::DuplicateSubsystem { subsystem } => {
                write!(f, "duplicate subsystem index: {subsystem}")
            }
            Self::EigenDecompositionFailed => write!(f, "eigendecomposition failed"),
        }
    }
}

impl Error for MeasureError {}

/// Method-style partial trace operations for state density matrices.
///
/// This trait is implemented for `DMatrix<Complex64>` because PECOS currently
/// represents state density matrices as dense complex matrices. The methods
/// validate that the matrix is a trace-one Hermitian density matrix before
/// reducing it.
pub trait DensityMatrixPartialTrace {
    /// Returns the reduced density matrix after tracing out selected
    /// tensor-product subsystems.
    ///
    /// `dims[i]` is the Hilbert-space dimension of subsystem `i`. Subsystem 0
    /// is the fastest-varying factor in the computational-basis index.
    ///
    /// # Errors
    ///
    /// Returns an error when the matrix is not a structurally valid density
    /// matrix, when `dims` do not match its shape, or when `traced_subsystems`
    /// contains an out-of-range or repeated subsystem.
    fn partial_trace(
        &self,
        dims: &[usize],
        traced_subsystems: &[usize],
    ) -> Result<DMatrix<Complex64>, MeasureError>;

    /// Returns the reduced density matrix after tracing out selected qubits.
    ///
    /// Qubit indexing is little-endian: qubit 0 is the least-significant bit
    /// of the computational-basis index.
    ///
    /// # Errors
    ///
    /// Returns an error when the matrix is not a structurally valid density
    /// matrix, when its shape is not `2^num_qubits x 2^num_qubits`, or when
    /// `traced_qubits` contains an out-of-range or repeated qubit.
    fn partial_trace_qubits(
        &self,
        num_qubits: usize,
        traced_qubits: &[usize],
    ) -> Result<DMatrix<Complex64>, MeasureError>;
}

impl DensityMatrixPartialTrace for DMatrix<Complex64> {
    fn partial_trace(
        &self,
        dims: &[usize],
        traced_subsystems: &[usize],
    ) -> Result<DMatrix<Complex64>, MeasureError> {
        partial_trace_subsystems(self, dims, traced_subsystems)
    }

    fn partial_trace_qubits(
        &self,
        num_qubits: usize,
        traced_qubits: &[usize],
    ) -> Result<DMatrix<Complex64>, MeasureError> {
        partial_trace_qubits(self, num_qubits, traced_qubits)
    }
}

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
        return Err(MeasureError::NonRealValue {
            value,
            tolerance: DEFAULT_TOLERANCE,
        });
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
        return Err(MeasureError::NonRealValue {
            value,
            tolerance: DEFAULT_TOLERANCE,
        });
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
    shannon_entropy(svd.singular_values.as_slice(), base)
}

/// Returns the Shannon entropy of a probability distribution.
///
/// The result is `-sum_i p_i log_base(p_i)`. Zero-probability entries
/// contribute zero.
///
/// # Errors
///
/// Returns an error when `base` is invalid, when a probability is negative or
/// non-finite, or when the probabilities do not sum to one.
///
/// # Examples
///
/// ```
/// use pecos_quantum::shannon_entropy;
///
/// let entropy = shannon_entropy(&[0.5, 0.5], 2.0).unwrap();
/// assert!((entropy - 1.0).abs() < 1e-12);
/// ```
pub fn shannon_entropy(probabilities: &[f64], base: f64) -> Result<f64, MeasureError> {
    validate_entropy_base(base)?;
    validate_probability_distribution(probabilities)?;
    let log_base = base.ln();
    Ok(probabilities
        .iter()
        .copied()
        .filter(|probability| *probability > DEFAULT_TOLERANCE)
        .map(|probability| -probability * probability.ln() / log_base)
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
        return Err(MeasureError::QubitCountMismatch {
            expected: left.num_qubits(),
            actual: right.num_qubits(),
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

/// Returns the two-qubit concurrence of a density matrix.
///
/// For a two-qubit state `rho`, this computes Wootters' concurrence using the
/// spin-flipped matrix `rho_tilde = (Y ⊗ Y) rho* (Y ⊗ Y)` and the square roots
/// of the eigenvalues of `rho rho_tilde`.
///
/// # Errors
///
/// Returns an error when `rho` is not a structurally valid 4x4 density matrix,
/// or when the eigendecomposition fails.
pub fn concurrence(rho: &DMatrix<Complex64>) -> Result<f64, MeasureError> {
    validate_density_matrix(rho)?;
    if rho.nrows() != 4 || rho.ncols() != 4 {
        return Err(MeasureError::InvalidMatrixShape {
            expected_rows: 4,
            expected_cols: 4,
            rows: rho.nrows(),
            cols: rho.ncols(),
        });
    }

    let yy = pauli_y_tensor_pauli_y();
    let rho_conj = rho.map(|value| value.conj());
    let rho_tilde = &yy * rho_conj * yy;
    let product = rho * rho_tilde;
    let eigenvalues = Schur::try_new(product, DEFAULT_TOLERANCE, 0)
        .and_then(|schur| schur.eigenvalues())
        .ok_or(MeasureError::EigenDecompositionFailed)?;

    let mut roots: Vec<f64> = eigenvalues
        .iter()
        .map(|lambda| {
            if lambda.im.abs() <= 1e-8 {
                lambda.re.max(0.0).sqrt()
            } else {
                lambda.norm().sqrt()
            }
        })
        .collect();
    roots.sort_by(|a, b| b.total_cmp(a));
    let value = roots[0] - roots[1] - roots[2] - roots[3];
    Ok(value.clamp(0.0, 1.0))
}

/// Returns the two-qubit entanglement of formation.
///
/// This is derived from [`concurrence`] as
/// `h((1 + sqrt(1 - C^2)) / 2)`, where `h` is binary entropy.
///
/// # Errors
///
/// Returns an error when [`concurrence`] fails.
pub fn entanglement_of_formation(rho: &DMatrix<Complex64>) -> Result<f64, MeasureError> {
    let concurrence = concurrence(rho)?;
    let argument = f64::midpoint(1.0, (1.0 - concurrence * concurrence).max(0.0).sqrt());
    Ok(binary_entropy(argument))
}

/// Returns the entanglement negativity of a bipartite or multipartite state.
///
/// This computes `(||rho^T_s||_1 - 1) / 2`, where `rho^T_s` is the partial
/// transpose with respect to `subsystem`.
///
/// # Errors
///
/// Returns an error when `rho` is not a structurally valid density matrix,
/// when `dims` do not match its Hilbert-space dimension, or when `subsystem`
/// is out of range.
pub fn negativity(
    rho: &DMatrix<Complex64>,
    dims: &[usize],
    subsystem: usize,
) -> Result<f64, MeasureError> {
    let partial_transpose = partial_transpose_subsystem(rho, dims, subsystem)?;
    let trace_norm: f64 = SVD::new(partial_transpose, false, false)
        .singular_values
        .iter()
        .sum();
    Ok(((trace_norm - 1.0) / 2.0).max(0.0))
}

/// Returns logarithmic negativity `log2(2 * negativity + 1)`.
///
/// # Errors
///
/// Returns an error when [`negativity`] fails.
pub fn logarithmic_negativity(
    rho: &DMatrix<Complex64>,
    dims: &[usize],
    subsystem: usize,
) -> Result<f64, MeasureError> {
    Ok((2.0 * negativity(rho, dims, subsystem)? + 1.0).log2())
}

/// Returns the Schmidt decomposition of a pure state across a bipartition.
///
/// `dims[i]` is the dimension of subsystem `i`; subsystem 0 is the
/// fastest-varying factor. `left_subsystems` selects the left side of the
/// bipartition. The right side is the sorted complement. Returned terms are
/// `(coefficient, left_vector, right_vector)` and omit numerically-zero
/// coefficients.
///
/// # Errors
///
/// Returns an error when the state is not normalized, when `dims` do not match
/// the state-vector length, or when `left_subsystems` contains an invalid or
/// repeated subsystem.
pub fn schmidt_decomposition(
    state: &DVector<Complex64>,
    dims: &[usize],
    left_subsystems: &[usize],
) -> Result<Vec<SchmidtTerm>, MeasureError> {
    validate_state_vector(state)?;
    validate_state_subsystem_dimensions(state.len(), dims)?;
    let left = validated_sorted_subsystems(dims, left_subsystems)?;
    let right: Vec<usize> = (0..dims.len())
        .filter(|subsystem| left.binary_search(subsystem).is_err())
        .collect();
    let left_dim = subsystem_product(dims, &left)?;
    let right_dim = subsystem_product(dims, &right)?;
    let strides = subsystem_strides(dims)?;

    let mut matrix = DMatrix::zeros(left_dim, right_dim);
    for basis_index in 0..state.len() {
        let left_index = project_subsystem_index(dims, &strides, &left, basis_index);
        let right_index = project_subsystem_index(dims, &strides, &right, basis_index);
        matrix[(left_index, right_index)] = state[basis_index];
    }

    let svd = SVD::new(matrix, true, true);
    let left_vectors = svd.u.ok_or(MeasureError::EigenDecompositionFailed)?;
    let right_vectors_adjoint = svd.v_t.ok_or(MeasureError::EigenDecompositionFailed)?;

    Ok(svd
        .singular_values
        .iter()
        .enumerate()
        .filter(|(_, coefficient)| **coefficient > DEFAULT_TOLERANCE)
        .map(|(idx, &coefficient)| {
            let left_vector = left_vectors.column(idx).iter().copied().collect();
            let right_vector = right_vectors_adjoint
                .row(idx)
                .iter()
                .copied()
                .map(|value| value.conj())
                .collect();
            (coefficient, left_vector, right_vector)
        })
        .collect())
}

/// Returns the reduced density matrix after tracing out selected subsystems.
///
/// `dims[i]` is the Hilbert-space dimension of subsystem `i`. Subsystem 0 is
/// the fastest-varying factor in the computational-basis index, matching PECOS
/// little-endian qubit ordering. The returned density matrix keeps untraced
/// subsystems in ascending subsystem-index order.
///
/// # Errors
///
/// Returns an error when `rho` is not a structurally valid density matrix,
/// when the product of `dims` does not match `rho`, or when
/// `traced_subsystems` contains an out-of-range or repeated subsystem.
pub fn partial_trace_subsystems(
    rho: &DMatrix<Complex64>,
    dims: &[usize],
    traced_subsystems: &[usize],
) -> Result<DMatrix<Complex64>, MeasureError> {
    validate_density_matrix(rho)?;
    validate_subsystem_dimensions(rho, dims)?;

    let mut traced = traced_subsystems.to_vec();
    traced.sort_unstable();
    for window in traced.windows(2) {
        if window[0] == window[1] {
            return Err(MeasureError::DuplicateSubsystem {
                subsystem: window[0],
            });
        }
    }
    for &subsystem in &traced {
        if subsystem >= dims.len() {
            return Err(MeasureError::SubsystemOutOfRange {
                num_subsystems: dims.len(),
                subsystem,
            });
        }
    }

    let kept: Vec<usize> = (0..dims.len())
        .filter(|subsystem| traced.binary_search(subsystem).is_err())
        .collect();
    let out_dim = subsystem_product(dims, &kept)?;
    let traced_dim = subsystem_product(dims, &traced)?;
    let strides = subsystem_strides(dims)?;

    let mut out = DMatrix::zeros(out_dim, out_dim);
    for kept_row in 0..out_dim {
        for kept_col in 0..out_dim {
            let mut value = Complex64::new(0.0, 0.0);
            for traced_idx in 0..traced_dim {
                let row =
                    embed_subsystem_index(dims, &strides, &kept, kept_row, &traced, traced_idx);
                let col =
                    embed_subsystem_index(dims, &strides, &kept, kept_col, &traced, traced_idx);
                value += rho[(row, col)];
            }
            out[(kept_row, kept_col)] = value;
        }
    }
    Ok(out)
}

/// Returns the reduced density matrix after tracing out selected qubits.
///
/// Qubit indexing is little-endian: qubit 0 is the least-significant bit of
/// the computational-basis index. The returned density matrix keeps untraced
/// qubits in ascending qubit-index order.
///
/// # Errors
///
/// Returns an error when `rho` is not a structurally valid density matrix, when
/// its shape is not `2^num_qubits x 2^num_qubits`, or when `traced_qubits`
/// contains an out-of-range or repeated qubit.
pub fn partial_trace_qubits(
    rho: &DMatrix<Complex64>,
    num_qubits: usize,
    traced_qubits: &[usize],
) -> Result<DMatrix<Complex64>, MeasureError> {
    let dims = vec![2; num_qubits];
    partial_trace_subsystems(rho, &dims, traced_qubits)
}

/// Returns bipartite quantum mutual information.
///
/// `dims` is `(dim_a, dim_b)`, and `rho` must have shape
/// `(dim_a * dim_b) x (dim_a * dim_b)`. Subsystem `A` is the fastest-varying
/// factor in the computational-basis index, matching
/// [`partial_trace_subsystems`].
///
/// # Errors
///
/// Returns an error when `rho` is not a structurally valid density matrix, when
/// `dims` are invalid, or when entropy evaluation fails on a reduced state.
pub fn mutual_information(
    rho: &DMatrix<Complex64>,
    dims: (usize, usize),
) -> Result<f64, MeasureError> {
    validate_density_matrix(rho)?;
    let (dim_a, dim_b) = dims;
    let Some(total_dim) = dim_a.checked_mul(dim_b) else {
        return Err(MeasureError::InvalidSubsystemDimensions {
            dims: vec![dim_a, dim_b],
            matrix_dim: rho.nrows(),
        });
    };
    if dim_a == 0 || dim_b == 0 || rho.nrows() != total_dim || rho.ncols() != total_dim {
        return Err(MeasureError::InvalidSubsystemDimensions {
            dims: vec![dim_a, dim_b],
            matrix_dim: rho.nrows(),
        });
    }

    let rho_a = partial_trace_subsystems(rho, &[dim_a, dim_b], &[1])?;
    let rho_b = partial_trace_subsystems(rho, &[dim_a, dim_b], &[0])?;
    Ok(entropy(&rho_a)? + entropy(&rho_b)? - entropy(rho)?)
}

/// Returns the Hellinger distance between classical probability distributions.
///
/// `H(p, q) = sqrt(1 - sum_i sqrt(p_i q_i))`.
///
/// # Errors
///
/// Returns an error when the vectors have different lengths or either vector is
/// not a probability distribution.
pub fn hellinger_distance(left: &[f64], right: &[f64]) -> Result<f64, MeasureError> {
    if left.len() != right.len() {
        return Err(MeasureError::VectorLengthMismatch {
            left: left.len(),
            right: right.len(),
        });
    }
    validate_probability_distribution(left)?;
    validate_probability_distribution(right)?;
    let affinity: f64 = left
        .iter()
        .zip(right.iter())
        .map(|(&left, &right)| (left * right).sqrt())
        .sum();
    Ok((1.0 - affinity.clamp(0.0, 1.0)).sqrt())
}

/// Returns Hellinger fidelity between classical probability distributions.
///
/// The value is `(1 - H(p, q)^2)^2`, where `H` is
/// [`hellinger_distance`].
///
/// # Errors
///
/// Returns an error when [`hellinger_distance`] fails.
pub fn hellinger_fidelity(left: &[f64], right: &[f64]) -> Result<f64, MeasureError> {
    let distance = hellinger_distance(left, right)?;
    Ok((1.0 - distance * distance).powi(2))
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

fn validate_probability_distribution(probabilities: &[f64]) -> Result<(), MeasureError> {
    let mut sum = 0.0;
    for (index, &probability) in probabilities.iter().enumerate() {
        if !probability.is_finite() || probability < -DEFAULT_TOLERANCE {
            return Err(MeasureError::InvalidProbability { index, probability });
        }
        sum += probability.max(0.0);
    }
    if (sum - 1.0).abs() > DEFAULT_TOLERANCE {
        return Err(MeasureError::InvalidProbabilitySum {
            sum,
            tolerance: DEFAULT_TOLERANCE,
        });
    }
    Ok(())
}

fn trace(matrix: &DMatrix<Complex64>) -> Complex64 {
    let n = matrix.nrows().min(matrix.ncols());
    (0..n).map(|idx| matrix[(idx, idx)]).sum()
}

fn pauli_y_tensor_pauli_y() -> DMatrix<Complex64> {
    let i = Complex64::new(0.0, 1.0);
    let minus_i = Complex64::new(0.0, -1.0);
    let y = DMatrix::from_row_slice(
        2,
        2,
        &[
            Complex64::new(0.0, 0.0),
            minus_i,
            i,
            Complex64::new(0.0, 0.0),
        ],
    );
    kronecker(&y, &y)
}

fn kronecker(left: &DMatrix<Complex64>, right: &DMatrix<Complex64>) -> DMatrix<Complex64> {
    let rows = left.nrows() * right.nrows();
    let cols = left.ncols() * right.ncols();
    let mut out = DMatrix::zeros(rows, cols);
    for left_row in 0..left.nrows() {
        for left_col in 0..left.ncols() {
            let scale = left[(left_row, left_col)];
            for right_row in 0..right.nrows() {
                for right_col in 0..right.ncols() {
                    out[(
                        left_row * right.nrows() + right_row,
                        left_col * right.ncols() + right_col,
                    )] = scale * right[(right_row, right_col)];
                }
            }
        }
    }
    out
}

fn validate_subsystem_dimensions(
    rho: &DMatrix<Complex64>,
    dims: &[usize],
) -> Result<(), MeasureError> {
    let Some(total_dim) =
        dims.iter().try_fold(
            1usize,
            |acc, &dim| {
                if dim == 0 { None } else { acc.checked_mul(dim) }
            },
        )
    else {
        return Err(MeasureError::InvalidSubsystemDimensions {
            dims: dims.to_vec(),
            matrix_dim: rho.nrows(),
        });
    };

    if rho.nrows() == total_dim && rho.ncols() == total_dim {
        Ok(())
    } else {
        Err(MeasureError::InvalidSubsystemDimensions {
            dims: dims.to_vec(),
            matrix_dim: rho.nrows(),
        })
    }
}

fn validate_state_subsystem_dimensions(
    state_len: usize,
    dims: &[usize],
) -> Result<(), MeasureError> {
    let Some(total_dim) =
        dims.iter().try_fold(
            1usize,
            |acc, &dim| {
                if dim == 0 { None } else { acc.checked_mul(dim) }
            },
        )
    else {
        return Err(MeasureError::InvalidSubsystemDimensions {
            dims: dims.to_vec(),
            matrix_dim: state_len,
        });
    };

    if state_len == total_dim {
        Ok(())
    } else {
        Err(MeasureError::InvalidSubsystemDimensions {
            dims: dims.to_vec(),
            matrix_dim: state_len,
        })
    }
}

fn validated_sorted_subsystems(
    dims: &[usize],
    subsystems: &[usize],
) -> Result<Vec<usize>, MeasureError> {
    let mut sorted = subsystems.to_vec();
    sorted.sort_unstable();
    for window in sorted.windows(2) {
        if window[0] == window[1] {
            return Err(MeasureError::DuplicateSubsystem {
                subsystem: window[0],
            });
        }
    }
    for &subsystem in &sorted {
        if subsystem >= dims.len() {
            return Err(MeasureError::SubsystemOutOfRange {
                num_subsystems: dims.len(),
                subsystem,
            });
        }
    }
    Ok(sorted)
}

fn subsystem_product(dims: &[usize], subsystems: &[usize]) -> Result<usize, MeasureError> {
    subsystems.iter().try_fold(1usize, |acc, &subsystem| {
        acc.checked_mul(dims[subsystem])
            .ok_or_else(|| MeasureError::InvalidSubsystemDimensions {
                dims: dims.to_vec(),
                matrix_dim: usize::MAX,
            })
    })
}

fn subsystem_strides(dims: &[usize]) -> Result<Vec<usize>, MeasureError> {
    let mut strides = Vec::with_capacity(dims.len());
    let mut stride = 1usize;
    for &dim in dims {
        strides.push(stride);
        stride =
            stride
                .checked_mul(dim)
                .ok_or_else(|| MeasureError::InvalidSubsystemDimensions {
                    dims: dims.to_vec(),
                    matrix_dim: usize::MAX,
                })?;
    }
    Ok(strides)
}

fn embed_subsystem_index(
    dims: &[usize],
    strides: &[usize],
    kept_subsystems: &[usize],
    kept_index: usize,
    traced_subsystems: &[usize],
    traced_index: usize,
) -> usize {
    let mut index = 0usize;
    let mut kept_remaining = kept_index;
    for &subsystem in kept_subsystems {
        let coord = kept_remaining % dims[subsystem];
        kept_remaining /= dims[subsystem];
        index += coord * strides[subsystem];
    }
    let mut traced_remaining = traced_index;
    for &subsystem in traced_subsystems {
        let coord = traced_remaining % dims[subsystem];
        traced_remaining /= dims[subsystem];
        index += coord * strides[subsystem];
    }
    index
}

fn project_subsystem_index(
    dims: &[usize],
    strides: &[usize],
    subsystems: &[usize],
    basis_index: usize,
) -> usize {
    let mut out = 0usize;
    let mut out_stride = 1usize;
    for &subsystem in subsystems {
        let coord = (basis_index / strides[subsystem]) % dims[subsystem];
        out += coord * out_stride;
        out_stride *= dims[subsystem];
    }
    out
}

fn partial_transpose_subsystem(
    rho: &DMatrix<Complex64>,
    dims: &[usize],
    subsystem: usize,
) -> Result<DMatrix<Complex64>, MeasureError> {
    validate_density_matrix(rho)?;
    validate_subsystem_dimensions(rho, dims)?;
    if subsystem >= dims.len() {
        return Err(MeasureError::SubsystemOutOfRange {
            num_subsystems: dims.len(),
            subsystem,
        });
    }

    let strides = subsystem_strides(dims)?;
    let mut out = DMatrix::zeros(rho.nrows(), rho.ncols());
    for row in 0..rho.nrows() {
        for col in 0..rho.ncols() {
            let row_coord = (row / strides[subsystem]) % dims[subsystem];
            let col_coord = (col / strides[subsystem]) % dims[subsystem];
            let transposed_row =
                row - row_coord * strides[subsystem] + col_coord * strides[subsystem];
            let transposed_col =
                col - col_coord * strides[subsystem] + row_coord * strides[subsystem];
            out[(transposed_row, transposed_col)] = rho[(row, col)];
        }
    }
    Ok(out)
}

fn binary_entropy(probability: f64) -> f64 {
    let p = probability.clamp(0.0, 1.0);
    if p <= DEFAULT_TOLERANCE || (1.0 - p) <= DEFAULT_TOLERANCE {
        0.0
    } else {
        -p * p.log2() - (1.0 - p) * (1.0 - p).log2()
    }
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

    fn werner_state(p: f64) -> DMatrix<Complex64> {
        let bell = ket(&[
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]);
        pure_density(&bell) * Complex64::new(p, 0.0)
            + DMatrix::identity(4, 4) * Complex64::new((1.0 - p) / 4.0, 0.0)
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
    fn two_qubit_entanglement_measures_match_known_states() {
        let bell = ket(&[
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]);
        let bell_rho = pure_density(&bell);
        assert_close(concurrence(&bell_rho).unwrap(), 1.0);
        assert_close(entanglement_of_formation(&bell_rho).unwrap(), 1.0);
        assert_close(mutual_information(&bell_rho, (2, 2)).unwrap(), 2.0);

        let zero_zero = ket(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        let product = pure_density(&zero_zero);
        assert_close(concurrence(&product).unwrap(), 0.0);
        assert_close(entanglement_of_formation(&product).unwrap(), 0.0);
        assert_close(mutual_information(&product, (2, 2)).unwrap(), 0.0);
    }

    #[test]
    fn concurrence_matches_werner_state_threshold_formula() {
        assert_close(concurrence(&werner_state(0.5)).unwrap(), 0.25);
        assert_close(concurrence(&werner_state(0.3)).unwrap(), 0.0);
    }

    #[test]
    fn entanglement_of_formation_matches_intermediate_werner_state() {
        let rho = werner_state(0.5);
        assert_close(concurrence(&rho).unwrap(), 0.25);
        assert_close(
            entanglement_of_formation(&rho).unwrap(),
            0.117_618_873_770_917_81,
        );
    }

    #[test]
    fn negativity_matches_bell_and_product_states() {
        let bell = ket(&[
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]);
        let bell_rho = pure_density(&bell);
        assert_close(negativity(&bell_rho, &[2, 2], 1).unwrap(), 0.5);
        assert_close(logarithmic_negativity(&bell_rho, &[2, 2], 1).unwrap(), 1.0);

        let product = pure_density(&ket(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]));
        assert_close(negativity(&product, &[2, 2], 1).unwrap(), 0.0);
        assert_close(logarithmic_negativity(&product, &[2, 2], 1).unwrap(), 0.0);
    }

    #[test]
    fn schmidt_decomposition_matches_bell_and_product_states() {
        let bell = ket(&[
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]);
        let bell_terms = schmidt_decomposition(&bell, &[2, 2], &[0]).unwrap();
        assert_eq!(bell_terms.len(), 2);
        assert_close(bell_terms[0].0, 1.0 / 2.0_f64.sqrt());
        assert_close(bell_terms[1].0, 1.0 / 2.0_f64.sqrt());

        let product = ket(&[
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
        ]);
        let product_terms = schmidt_decomposition(&product, &[2, 2], &[0]).unwrap();
        assert_eq!(product_terms.len(), 1);
        assert_close(product_terms[0].0, 1.0);
    }

    #[test]
    fn schmidt_decomposition_supports_unequal_bipartition() {
        let mut ghz = DVector::zeros(8);
        ghz[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        ghz[7] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);

        let terms = schmidt_decomposition(&ghz, &[2, 4], &[0]).unwrap();
        assert_eq!(terms.len(), 2);
        assert_close(terms[0].0, 1.0 / 2.0_f64.sqrt());
        assert_close(terms[1].0, 1.0 / 2.0_f64.sqrt());
        assert_eq!(terms[0].1.len(), 2);
        assert_eq!(terms[0].2.len(), 4);
    }

    #[test]
    fn mutual_information_accepts_non_qubit_subsystem_dims() {
        let mut rho = DMatrix::zeros(6, 6);
        rho[(0, 0)] = Complex64::new(0.5, 0.0);
        rho[(5, 5)] = Complex64::new(0.5, 0.0);

        assert_close(mutual_information(&rho, (2, 3)).unwrap(), 1.0);
    }

    #[test]
    fn shannon_entropy_is_public_distribution_entropy() {
        assert_close(shannon_entropy(&[0.5, 0.5], 2.0).unwrap(), 1.0);
        assert_close(shannon_entropy(&[1.0, 0.0], 2.0).unwrap(), 0.0);
        assert!(matches!(
            shannon_entropy(&[0.25, 0.25], 2.0).unwrap_err(),
            MeasureError::InvalidProbabilitySum { .. }
        ));
    }

    #[test]
    fn hellinger_distance_and_fidelity_match_classical_cases() {
        assert_close(
            hellinger_distance(&[0.25, 0.75], &[0.25, 0.75]).unwrap(),
            0.0,
        );
        assert_close(
            hellinger_fidelity(&[0.25, 0.75], &[0.25, 0.75]).unwrap(),
            1.0,
        );

        assert_close(hellinger_distance(&[1.0, 0.0], &[0.0, 1.0]).unwrap(), 1.0);
        assert_close(hellinger_fidelity(&[1.0, 0.0], &[0.0, 1.0]).unwrap(), 0.0);
    }

    #[test]
    fn partial_trace_supports_method_and_qubit_forms() {
        let bell = ket(&[
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0),
        ]);
        let bell_rho = pure_density(&bell);

        let reduced_from_method = bell_rho.partial_trace(&[2, 2], &[1]).unwrap();
        let reduced_from_qubits = partial_trace_qubits(&bell_rho, 2, &[1]).unwrap();
        let expected = DMatrix::from_diagonal_element(2, 2, Complex64::new(0.5, 0.0));

        assert_close((&reduced_from_method - &expected).norm(), 0.0);
        assert_close((&reduced_from_qubits - expected).norm(), 0.0);
    }

    #[test]
    fn partial_trace_subsystems_accepts_arbitrary_tensor_factors() {
        let mut state = DVector::zeros(12);
        state[2] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        state[9] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let rho = pure_density(&state);

        let reduced = partial_trace_subsystems(&rho, &[2, 3, 2], &[1]).unwrap();
        let mut expected = DMatrix::zeros(4, 4);
        expected[(0, 0)] = Complex64::new(0.5, 0.0);
        expected[(0, 3)] = Complex64::new(0.5, 0.0);
        expected[(3, 0)] = Complex64::new(0.5, 0.0);
        expected[(3, 3)] = Complex64::new(0.5, 0.0);

        assert_close((reduced - expected).norm(), 0.0);
    }

    #[test]
    fn partial_trace_subsystems_can_trace_noncontiguous_factors() {
        let mut state = DVector::zeros(12);
        state[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        state[11] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let rho = pure_density(&state);

        let reduced = partial_trace_subsystems(&rho, &[2, 3, 2], &[0, 2]).unwrap();
        let mut expected = DMatrix::zeros(3, 3);
        expected[(0, 0)] = Complex64::new(0.5, 0.0);
        expected[(2, 2)] = Complex64::new(0.5, 0.0);

        assert_close((reduced - expected).norm(), 0.0);
    }

    #[test]
    fn three_qubit_ghz_reductions_have_expected_information() {
        let mut ghz = DVector::zeros(8);
        ghz[0] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        ghz[7] = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);
        let rho = pure_density(&ghz);

        let two_qubit_reduction = partial_trace_qubits(&rho, 3, &[2]).unwrap();
        let mut expected_two_qubit = DMatrix::zeros(4, 4);
        expected_two_qubit[(0, 0)] = Complex64::new(0.5, 0.0);
        expected_two_qubit[(3, 3)] = Complex64::new(0.5, 0.0);
        assert_close((&two_qubit_reduction - &expected_two_qubit).norm(), 0.0);
        assert_close(entropy(&two_qubit_reduction).unwrap(), 1.0);
        assert_close(
            mutual_information(&two_qubit_reduction, (2, 2)).unwrap(),
            1.0,
        );

        let one_qubit_reduction = partial_trace_qubits(&rho, 3, &[1, 2]).unwrap();
        let expected_one_qubit = DMatrix::from_diagonal_element(2, 2, Complex64::new(0.5, 0.0));
        assert_close((&one_qubit_reduction - expected_one_qubit).norm(), 0.0);
        assert_close(entropy(&one_qubit_reduction).unwrap(), 1.0);
    }

    #[test]
    fn partial_trace_rejects_repeated_or_out_of_range_subsystems() {
        let mixed = DMatrix::from_diagonal_element(4, 4, Complex64::new(0.25, 0.0));
        assert!(matches!(
            partial_trace_subsystems(&mixed, &[2, 2], &[0, 0]).unwrap_err(),
            MeasureError::DuplicateSubsystem { subsystem: 0 }
        ));
        assert!(matches!(
            partial_trace_subsystems(&mixed, &[2, 2], &[2]).unwrap_err(),
            MeasureError::SubsystemOutOfRange {
                num_subsystems: 2,
                subsystem: 2
            }
        ));
    }

    #[test]
    fn entanglement_measures_reject_invalid_shapes() {
        let mixed = DMatrix::from_diagonal_element(2, 2, Complex64::new(0.5, 0.0));
        assert!(matches!(
            concurrence(&mixed).unwrap_err(),
            MeasureError::InvalidMatrixShape { .. }
        ));
        assert!(matches!(
            mutual_information(&mixed, (2, 2)).unwrap_err(),
            MeasureError::InvalidSubsystemDimensions { .. }
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

    #[test]
    fn process_fidelity_reports_qubit_count_mismatch() {
        let one_qubit = Ptm::identity(1).unwrap();
        let two_qubit = Ptm::identity(2).unwrap();

        assert_eq!(
            process_fidelity(&one_qubit, &two_qubit).unwrap_err(),
            MeasureError::QubitCountMismatch {
                expected: 1,
                actual: 2
            }
        );
        assert_eq!(
            average_gate_fidelity(&one_qubit, &two_qubit).unwrap_err(),
            MeasureError::QubitCountMismatch {
                expected: 1,
                actual: 2
            }
        );
        assert_eq!(
            gate_error(&one_qubit, &two_qubit).unwrap_err(),
            MeasureError::QubitCountMismatch {
                expected: 1,
                actual: 2
            }
        );
    }
}
