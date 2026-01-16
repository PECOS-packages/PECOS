// Copyright 2025 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Matrix representations and operations for quantum operators.
//!
//! This module provides functions to convert operators to dense matrices
//! and perform matrix-level operations like exponential and logarithm.
//!
//! # Extension Trait
//!
//! The [`ToMatrix`] trait provides a method-style API for converting operators:
//!
//! ```
//! use pecos_quantum::operator_matrix::ToMatrix;
//! use pecos_core::operator::X;
//!
//! let x = X(0);
//! let matrix = x.to_matrix();  // Method style
//! ```

use nalgebra::DMatrix;
use num_complex::Complex64;
use pecos_core::gate_type::GateType;
use pecos_core::operator::{Operator, RotationType};
use pecos_core::{Pauli, PauliString, Phase};

/// Extension trait for converting quantum operators to matrix representations.
///
/// This trait is implemented for [`Operator`] and [`PauliString`], providing
/// a method-style API for matrix conversion.
///
/// # Example
///
/// ```
/// use pecos_quantum::operator_matrix::ToMatrix;
/// use pecos_core::operator::{X, H, CX, Is};
///
/// // Single qubit gate
/// let x_matrix = X(0).to_matrix();
/// assert_eq!(x_matrix.nrows(), 2);
///
/// // Two qubit gate
/// let cnot_matrix = CX(0, 1).to_matrix();
/// assert_eq!(cnot_matrix.nrows(), 4);
///
/// // For larger matrices, tensor with identities using Is()
/// let x_extended = X(0) & Is(1..3);  // X on qubit 0 in 3-qubit space
/// let mat = x_extended.to_matrix();
/// assert_eq!(mat.nrows(), 8);
/// ```
pub trait ToMatrix {
    /// Converts to a dense matrix representation.
    ///
    /// The matrix size is 2^n where n is determined by the maximum qubit index + 1.
    fn to_matrix(&self) -> DMatrix<Complex64>;
}

impl ToMatrix for Operator {
    fn to_matrix(&self) -> DMatrix<Complex64> {
        to_matrix(self)
    }
}

impl ToMatrix for PauliString {
    fn to_matrix(&self) -> DMatrix<Complex64> {
        let num_qubits = self.qubits().into_iter().max().map_or(1, |q| q + 1);
        pauli_string_to_matrix_impl(self, num_qubits)
    }
}

/// Converts an `Operator` to its dense matrix representation.
///
/// The matrix size is 2^n where n is the number of qubits (determined by
/// the maximum qubit index + 1).
///
/// # Example
///
/// ```
/// use pecos_quantum::operator_matrix::to_matrix;
/// use pecos_core::operator::X;
/// use num_complex::Complex64;
///
/// let x = X(0);
/// let matrix = to_matrix(&x);
///
/// // X gate matrix: [[0, 1], [1, 0]]
/// assert_eq!(matrix.nrows(), 2);
/// assert!((matrix[(0, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
/// ```
#[must_use]
pub fn to_matrix(op: &Operator) -> DMatrix<Complex64> {
    let num_qubits = op.qubits().into_iter().max().map_or(1, |q| q + 1);
    to_matrix_with_size(op, num_qubits)
}

/// Converts an `Operator` to its dense matrix representation with a specified size.
///
/// # Arguments
/// * `op` - The operator to convert
/// * `num_qubits` - The number of qubits (matrix will be `2^num_qubits` x `2^num_qubits`)
#[must_use]
pub fn to_matrix_with_size(op: &Operator, num_qubits: usize) -> DMatrix<Complex64> {
    let dim = 1 << num_qubits; // 2^num_qubits

    match op {
        Operator::Pauli(ps) => pauli_string_to_matrix_impl(ps, num_qubits),

        Operator::Rotation {
            rotation_type,
            angle,
            qubits,
        } => rotation_to_matrix(*rotation_type, angle.to_radians(), qubits, num_qubits),

        Operator::Gate { gate_type, qubits } => gate_to_matrix(*gate_type, qubits, num_qubits),

        Operator::Tensor(parts) => {
            // Start with identity, combine each part
            let mut result = DMatrix::identity(dim, dim);
            for part in parts {
                let part_matrix = to_matrix_with_size(part, num_qubits);
                result = combine_disjoint_operators(&result, &part_matrix);
            }
            result
        }

        Operator::Compose(parts) => {
            // Matrix multiplication in reverse order (last part applied first)
            let mut result = DMatrix::identity(dim, dim);
            for part in parts {
                let part_matrix = to_matrix_with_size(part, num_qubits);
                result = part_matrix * result;
            }
            result
        }

        Operator::Adjoint(inner) => {
            let inner_matrix = to_matrix_with_size(inner, num_qubits);
            inner_matrix.adjoint()
        }

        Operator::Phase { phase, inner } => {
            let inner_matrix = to_matrix_with_size(inner, num_qubits);
            let phase_factor = Complex64::new(0.0, phase.to_radians()).exp();
            inner_matrix * phase_factor
        }
    }
}

/// Computes the matrix exponential of an operator: exp(i * op).
///
/// This is useful for generating unitaries from Hermitian generators.
///
/// # Example
///
/// ```
/// use pecos_quantum::operator_matrix::operator_exp;
/// use pecos_core::operator::Z;
/// use num_complex::Complex64;
/// use std::f64::consts::PI;
///
/// // exp(i * pi * Z) = -I
/// let z = Z(0);
/// let result = operator_exp(&z, PI);
/// // Result should be approximately -I
/// ```
#[must_use]
pub fn operator_exp(op: &Operator, theta: f64) -> DMatrix<Complex64> {
    let matrix = to_matrix(op);
    let scaled = matrix * Complex64::new(0.0, theta);
    pecos_num::matrix_exp(&scaled)
}

/// Computes the matrix logarithm of an operator.
///
/// Returns `Some(generator)` where `exp(i * generator) = op`, or `None` if
/// the computation fails (e.g., for singular matrices).
///
/// # Example
///
/// ```
/// use pecos_quantum::operator_matrix::{operator_log, to_matrix};
/// use pecos_core::operator::X;
///
/// let x = X(0);
/// if let Some(log_x) = operator_log(&x) {
///     // log_x is the generator such that exp(i * log_x) = X
/// }
/// ```
#[must_use]
pub fn operator_log(op: &Operator) -> Option<DMatrix<Complex64>> {
    let matrix = to_matrix(op);
    let log_matrix = pecos_num::matrix_log(&matrix)?;
    // Divide by i to get the Hermitian generator
    Some(log_matrix / Complex64::new(0.0, 1.0))
}

/// Checks if two operators are equivalent up to a global phase.
///
/// Returns `true` if A = e^{i*phi} * B for some real phi.
///
/// # Example
///
/// ```
/// use pecos_quantum::operator_matrix::operators_equiv;
/// use pecos_core::operator::{X, Y, Z};
///
/// let x = X(0);
/// let x2 = X(0);
/// assert!(operators_equiv(&x, &x2));
///
/// let y = Y(0);
/// assert!(!operators_equiv(&x, &y));
/// ```
#[must_use]
pub fn operators_equiv(a: &Operator, b: &Operator) -> bool {
    operators_equiv_with_tolerance(a, b, 1e-10)
}

/// Checks if two operators are equivalent up to a global phase, with custom tolerance.
#[must_use]
pub fn operators_equiv_with_tolerance(a: &Operator, b: &Operator, tol: f64) -> bool {
    let num_qubits_a = a.qubits().into_iter().max().map_or(1, |q| q + 1);
    let num_qubits_b = b.qubits().into_iter().max().map_or(1, |q| q + 1);
    let num_qubits = num_qubits_a.max(num_qubits_b);

    let mat_a = to_matrix_with_size(a, num_qubits);
    let mat_b = to_matrix_with_size(b, num_qubits);

    matrices_equiv_up_to_phase(&mat_a, &mat_b, tol)
}

/// Checks if two matrices are equal up to a global phase factor.
fn matrices_equiv_up_to_phase(a: &DMatrix<Complex64>, b: &DMatrix<Complex64>, tol: f64) -> bool {
    if a.nrows() != b.nrows() || a.ncols() != b.ncols() {
        return false;
    }

    // Find the first non-zero element to determine the phase
    let mut phase: Option<Complex64> = None;

    for i in 0..a.nrows() {
        for j in 0..a.ncols() {
            let a_val = a[(i, j)];
            let b_val = b[(i, j)];

            // Skip near-zero elements
            if a_val.norm() < tol && b_val.norm() < tol {
                continue;
            }

            // If one is zero but not the other, not equivalent
            if a_val.norm() < tol || b_val.norm() < tol {
                return false;
            }

            // Compute the ratio a/b
            let ratio = a_val / b_val;

            match phase {
                None => {
                    // First non-zero element sets the phase
                    phase = Some(ratio);
                }
                Some(p) => {
                    // Check if this ratio matches the established phase
                    if (ratio - p).norm() > tol {
                        return false;
                    }
                }
            }
        }
    }

    // Also verify the phase has unit magnitude (global phase factor)
    if let Some(p) = phase {
        (p.norm() - 1.0).abs() < tol
    } else {
        // Both matrices are zero
        true
    }
}

// ============================================================================
// Helper functions for matrix construction
// ============================================================================

/// Converts a [`PauliString`] to a dense matrix (implementation).
fn pauli_string_to_matrix_impl(ps: &PauliString, num_qubits: usize) -> DMatrix<Complex64> {
    let dim = 1 << num_qubits;
    let mut result = DMatrix::identity(dim, dim);

    // Get the phase
    let phase = ps.phase().to_complex();

    // Apply each single-qubit Pauli
    for (pauli, qubit) in ps.iter_pairs() {
        let q = usize::from(qubit);
        let pauli_matrix = single_pauli_matrix(pauli);
        let full_matrix = embed_single_qubit_gate(&pauli_matrix, q, num_qubits);
        result = full_matrix * result;
    }

    result * phase
}

/// Returns the 2x2 matrix for a single Pauli operator.
fn single_pauli_matrix(pauli: Pauli) -> DMatrix<Complex64> {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let neg_i = Complex64::new(0.0, -1.0);
    let neg_one = Complex64::new(-1.0, 0.0);

    match pauli {
        Pauli::I => DMatrix::from_row_slice(2, 2, &[one, zero, zero, one]),
        Pauli::X => DMatrix::from_row_slice(2, 2, &[zero, one, one, zero]),
        Pauli::Y => DMatrix::from_row_slice(2, 2, &[zero, neg_i, i, zero]),
        Pauli::Z => DMatrix::from_row_slice(2, 2, &[one, zero, zero, neg_one]),
    }
}

/// Embeds a single-qubit gate into a larger Hilbert space.
fn embed_single_qubit_gate(
    gate: &DMatrix<Complex64>,
    qubit: usize,
    num_qubits: usize,
) -> DMatrix<Complex64> {
    let dim = 1 << num_qubits;
    let mut result = DMatrix::from_element(dim, dim, Complex64::new(0.0, 0.0));

    for i in 0..dim {
        for j in 0..dim {
            // Check if all qubits except `qubit` match
            let mask = !(1 << qubit);
            if (i & mask) == (j & mask) {
                let i_bit = (i >> qubit) & 1;
                let j_bit = (j >> qubit) & 1;
                result[(i, j)] = gate[(i_bit, j_bit)];
            }
        }
    }

    result
}

/// Converts a rotation to a matrix.
fn rotation_to_matrix(
    rotation_type: RotationType,
    angle: f64,
    qubits: &[usize],
    num_qubits: usize,
) -> DMatrix<Complex64> {
    let half_angle = angle / 2.0;
    let cos_half = Complex64::new(half_angle.cos(), 0.0);
    let sin_half = Complex64::new(half_angle.sin(), 0.0);
    let i = Complex64::new(0.0, 1.0);
    let neg_i = Complex64::new(0.0, -1.0);

    match rotation_type {
        RotationType::RX => {
            // RX(θ) = cos(θ/2)I - i*sin(θ/2)X
            let gate = DMatrix::from_row_slice(
                2,
                2,
                &[cos_half, neg_i * sin_half, neg_i * sin_half, cos_half],
            );
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        RotationType::RY => {
            // RY(θ) = cos(θ/2)I - i*sin(θ/2)Y
            let gate = DMatrix::from_row_slice(2, 2, &[cos_half, -sin_half, sin_half, cos_half]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        RotationType::RZ => {
            // RZ(θ) = cos(θ/2)I - i*sin(θ/2)Z = diag(e^{-iθ/2}, e^{iθ/2})
            let exp_neg = (neg_i * Complex64::new(half_angle, 0.0)).exp();
            let exp_pos = (i * Complex64::new(half_angle, 0.0)).exp();
            let zero = Complex64::new(0.0, 0.0);
            let gate = DMatrix::from_row_slice(2, 2, &[exp_neg, zero, zero, exp_pos]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        RotationType::RXX | RotationType::RYY | RotationType::RZZ => {
            // For two-qubit rotations, use matrix exponential
            let dim = 1 << num_qubits;
            let generator = match rotation_type {
                RotationType::RXX => {
                    two_qubit_pauli_matrix(Pauli::X, Pauli::X, qubits[0], qubits[1], num_qubits)
                }
                RotationType::RYY => {
                    two_qubit_pauli_matrix(Pauli::Y, Pauli::Y, qubits[0], qubits[1], num_qubits)
                }
                RotationType::RZZ => {
                    two_qubit_pauli_matrix(Pauli::Z, Pauli::Z, qubits[0], qubits[1], num_qubits)
                }
                _ => DMatrix::identity(dim, dim),
            };
            let scaled = generator * Complex64::new(0.0, -half_angle);
            pecos_num::matrix_exp(&scaled)
        }
    }
}

/// Constructs a two-qubit Pauli tensor product matrix.
fn two_qubit_pauli_matrix(
    p1: Pauli,
    p2: Pauli,
    q1: usize,
    q2: usize,
    num_qubits: usize,
) -> DMatrix<Complex64> {
    let m1 = single_pauli_matrix(p1);
    let m2 = single_pauli_matrix(p2);
    let e1 = embed_single_qubit_gate(&m1, q1, num_qubits);
    let e2 = embed_single_qubit_gate(&m2, q2, num_qubits);
    e1 * e2
}

/// Converts a gate type to a matrix.
fn gate_to_matrix(gate_type: GateType, qubits: &[usize], num_qubits: usize) -> DMatrix<Complex64> {
    let zero = Complex64::new(0.0, 0.0);
    let one = Complex64::new(1.0, 0.0);
    let i = Complex64::new(0.0, 1.0);
    let neg_i = Complex64::new(0.0, -1.0);
    let neg_one = Complex64::new(-1.0, 0.0);
    let sqrt2_inv = Complex64::new(1.0 / 2.0_f64.sqrt(), 0.0);

    match gate_type {
        GateType::I => {
            let gate = DMatrix::from_row_slice(2, 2, &[one, zero, zero, one]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::X => {
            let gate = DMatrix::from_row_slice(2, 2, &[zero, one, one, zero]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::Y => {
            let gate = DMatrix::from_row_slice(2, 2, &[zero, neg_i, i, zero]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::Z => {
            let gate = DMatrix::from_row_slice(2, 2, &[one, zero, zero, neg_one]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::H => {
            let gate =
                DMatrix::from_row_slice(2, 2, &[sqrt2_inv, sqrt2_inv, sqrt2_inv, -sqrt2_inv]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::SX => {
            // SX = (1+i)/2 * [[1, -i], [-i, 1]]
            let factor = Complex64::new(0.5, 0.5);
            let gate = DMatrix::from_row_slice(
                2,
                2,
                &[factor * one, factor * neg_i, factor * neg_i, factor * one],
            );
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::SXdg => {
            let factor = Complex64::new(0.5, -0.5);
            let gate = DMatrix::from_row_slice(
                2,
                2,
                &[factor * one, factor * i, factor * i, factor * one],
            );
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::SZ => {
            // S = diag(1, i)
            let gate = DMatrix::from_row_slice(2, 2, &[one, zero, zero, i]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::SZdg => {
            let gate = DMatrix::from_row_slice(2, 2, &[one, zero, zero, neg_i]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::T => {
            // T = diag(1, e^{i*pi/4})
            let exp_pi_4 = Complex64::from_polar(1.0, std::f64::consts::FRAC_PI_4);
            let gate = DMatrix::from_row_slice(2, 2, &[one, zero, zero, exp_pi_4]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::Tdg => {
            let exp_neg_pi_4 = Complex64::from_polar(1.0, -std::f64::consts::FRAC_PI_4);
            let gate = DMatrix::from_row_slice(2, 2, &[one, zero, zero, exp_neg_pi_4]);
            embed_single_qubit_gate(&gate, qubits[0], num_qubits)
        }
        GateType::CX => controlled_gate(
            &single_pauli_matrix(Pauli::X),
            qubits[0],
            qubits[1],
            num_qubits,
        ),
        GateType::CY => controlled_gate(
            &single_pauli_matrix(Pauli::Y),
            qubits[0],
            qubits[1],
            num_qubits,
        ),
        GateType::CZ => controlled_gate(
            &single_pauli_matrix(Pauli::Z),
            qubits[0],
            qubits[1],
            num_qubits,
        ),
        GateType::SWAP => swap_matrix(qubits[0], qubits[1], num_qubits),
        _ => {
            // Gates not yet implemented: SY, SYdg, U, R1XY, SZZ, SZZdg, CRZ, CCX
            // Rotation gates (RX, RY, RZ, RXX, RYY, RZZ) should use Operator::Rotation
            // Prep/Measure gates are not unitary and shouldn't be converted to matrices
            log::warn!("Gate type {gate_type:?} not implemented in to_matrix, returning identity");
            let dim = 1 << num_qubits;
            DMatrix::identity(dim, dim)
        }
    }
}

/// Constructs a controlled gate matrix.
fn controlled_gate(
    target_gate: &DMatrix<Complex64>,
    control: usize,
    target: usize,
    num_qubits: usize,
) -> DMatrix<Complex64> {
    let dim = 1 << num_qubits;
    let mut result = DMatrix::identity(dim, dim);

    for i in 0..dim {
        for j in 0..dim {
            // Only apply gate when control qubit is 1
            let control_bit_i = (i >> control) & 1;
            let control_bit_j = (j >> control) & 1;

            if control_bit_i == 1 && control_bit_j == 1 {
                // Check if all qubits except target match
                let mask = !(1 << target);
                if (i & mask) == (j & mask) {
                    let i_bit = (i >> target) & 1;
                    let j_bit = (j >> target) & 1;
                    result[(i, j)] = target_gate[(i_bit, j_bit)];
                } else {
                    result[(i, j)] = Complex64::new(0.0, 0.0);
                }
            } else if control_bit_i == control_bit_j && i == j {
                result[(i, j)] = Complex64::new(1.0, 0.0);
            } else if control_bit_i != control_bit_j {
                result[(i, j)] = Complex64::new(0.0, 0.0);
            }
        }
    }

    result
}

/// Constructs a SWAP gate matrix.
fn swap_matrix(q1: usize, q2: usize, num_qubits: usize) -> DMatrix<Complex64> {
    let dim = 1 << num_qubits;
    let mut result = DMatrix::from_element(dim, dim, Complex64::new(0.0, 0.0));

    for i in 0..dim {
        // Swap bits at positions q1 and q2
        let bit1 = (i >> q1) & 1;
        let bit2 = (i >> q2) & 1;

        let j = if bit1 == bit2 {
            i
        } else {
            // Swap the bits
            i ^ (1 << q1) ^ (1 << q2)
        };

        result[(i, j)] = Complex64::new(1.0, 0.0);
    }

    result
}

/// Combines two matrices representing operators on disjoint qubits.
///
/// When operators act on disjoint qubits, the tensor product in the full Hilbert space
/// is equivalent to matrix multiplication (since disjoint operators commute).
fn combine_disjoint_operators(
    a: &DMatrix<Complex64>,
    b: &DMatrix<Complex64>,
) -> DMatrix<Complex64> {
    a * b
}

#[cfg(test)]
mod tests {
    use super::*;
    use pecos_core::Angle64;
    use pecos_core::operator::{CX, H, I, Is, RX, RZ, SWAP, SZ, T, X, Y, Z};
    use std::f64::consts::PI;

    // ========================================================================
    // Basic to_matrix tests
    // ========================================================================

    #[test]
    fn test_pauli_matrices() {
        let x = X(0);
        let mat = to_matrix(&x);
        assert_eq!(mat.nrows(), 2);
        assert!((mat[(0, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);

        let z = Z(0);
        let mat = to_matrix(&z);
        assert!((mat[(0, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 1)] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_pauli_y() {
        let y = Y(0);
        let mat = to_matrix(&y);
        let i = Complex64::new(0.0, 1.0);
        assert!((mat[(0, 1)] - (-i)).norm() < 1e-10);
        assert!((mat[(1, 0)] - i).norm() < 1e-10);
    }

    #[test]
    fn test_hadamard() {
        let h = H(0);
        let mat = to_matrix(&h);
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        assert!((mat[(0, 0)] - Complex64::new(sqrt2_inv, 0.0)).norm() < 1e-10);
        assert!((mat[(0, 1)] - Complex64::new(sqrt2_inv, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 0)] - Complex64::new(sqrt2_inv, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 1)] - Complex64::new(-sqrt2_inv, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_cnot() {
        let cx = CX(0, 1);
        let mat = to_matrix(&cx);
        assert_eq!(mat.nrows(), 4);
        // CX with control=0, target=1
        // |q1 q0> indexing: index = q1*2 + q0
        // When q0=0 (control off): do nothing
        // When q0=1 (control on): flip q1
        // |00> -> |00> (mat[0,0] = 1)
        // |01> -> |11> (mat[3,1] = 1)
        // |10> -> |10> (mat[2,2] = 1)
        // |11> -> |01> (mat[1,3] = 1)
        assert!((mat[(0, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(3, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(2, 2)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 3)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_identity() {
        let id = I(0);
        let mat = to_matrix(&id);
        assert!((mat[(0, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!(mat[(0, 1)].norm() < 1e-10);
        assert!(mat[(1, 0)].norm() < 1e-10);
    }

    // ========================================================================
    // Rotation matrix tests
    // ========================================================================

    #[test]
    fn test_t_gate_matrix() {
        let t = T(0);
        let mat = to_matrix(&t);
        // T = RZ(π/4) = diag(e^{-iπ/8}, e^{iπ/8})
        let exp_neg = Complex64::from_polar(1.0, -PI / 8.0);
        let exp_pos = Complex64::from_polar(1.0, PI / 8.0);
        assert!((mat[(0, 0)] - exp_neg).norm() < 1e-10);
        assert!((mat[(1, 1)] - exp_pos).norm() < 1e-10);
    }

    #[test]
    fn test_s_gate_matrix() {
        let s = SZ(0);
        let mat = to_matrix(&s);
        // S = RZ(π/2) = diag(e^{-iπ/4}, e^{iπ/4})
        let exp_neg = Complex64::from_polar(1.0, -PI / 4.0);
        let exp_pos = Complex64::from_polar(1.0, PI / 4.0);
        assert!((mat[(0, 0)] - exp_neg).norm() < 1e-10);
        assert!((mat[(1, 1)] - exp_pos).norm() < 1e-10);
    }

    #[test]
    fn test_rx_matrix() {
        // RX(π) should give X (up to global phase)
        let rx_pi = RX(Angle64::HALF_TURN, 0);
        let mat = to_matrix(&rx_pi);
        let x_mat = to_matrix(&X(0));
        // RX(π) = -iX, so matrices differ by global phase -i
        assert!(matrices_equiv_up_to_phase(&mat, &x_mat, 1e-10));
    }

    #[test]
    fn test_rz_matrix() {
        // RZ(π) should give Z (up to global phase)
        let rz_pi = RZ(Angle64::HALF_TURN, 0);
        let mat = to_matrix(&rz_pi);
        let z_mat = to_matrix(&Z(0));
        assert!(matrices_equiv_up_to_phase(&mat, &z_mat, 1e-10));
    }

    // ========================================================================
    // Tensor product and composition tests
    // ========================================================================

    #[test]
    fn test_tensor_product() {
        // X ⊗ Z should give a 4x4 matrix
        let xz = X(0) & Z(1);
        let mat = to_matrix(&xz);
        assert_eq!(mat.nrows(), 4);

        // Verify it's the product of embedded X and Z
        let x_embedded = to_matrix_with_size(&X(0), 2);
        let z_embedded = to_matrix_with_size(&Z(1), 2);
        let expected = &x_embedded * &z_embedded;
        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_composition() {
        // H * X = XH (matrix multiplication order)
        let hx = H(0) * X(0);
        let mat = to_matrix(&hx);

        let h_mat = to_matrix(&H(0));
        let x_mat = to_matrix(&X(0));
        let expected = &h_mat * &x_mat;
        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_adjoint_matrix() {
        // T† matrix should be conjugate transpose of T
        let t = T(0);
        let t_dg = t.dg();
        let mat_t = to_matrix(&t);
        let mat_t_dg = to_matrix(&t_dg);

        let expected = mat_t.adjoint();
        assert!(matrices_equiv_up_to_phase(&mat_t_dg, &expected, 1e-10));
    }

    #[test]
    fn test_swap_gate() {
        let swap = SWAP(0, 1);
        let mat = to_matrix(&swap);
        assert_eq!(mat.nrows(), 4);

        // SWAP|00> = |00>, SWAP|01> = |10>, SWAP|10> = |01>, SWAP|11> = |11>
        assert!((mat[(0, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(2, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 2)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(3, 3)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    // ========================================================================
    // operators_equiv tests
    // ========================================================================

    #[test]
    fn test_operators_equiv_same() {
        let x1 = X(0);
        let x2 = X(0);
        assert!(operators_equiv(&x1, &x2));
    }

    #[test]
    fn test_operators_equiv_different() {
        let x = X(0);
        let y = Y(0);
        assert!(!operators_equiv(&x, &y));
    }

    #[test]
    fn test_operators_equiv_global_phase() {
        // X and -X differ by global phase -1
        let x = X(0);
        let neg_x = pecos_core::operator::phase(Angle64::HALF_TURN) * X(0);
        assert!(operators_equiv(&x, &neg_x));
    }

    #[test]
    fn test_operators_equiv_i_phase() {
        // X and iX differ by global phase i
        let x = X(0);
        let i_x = pecos_core::operator::i * X(0);
        assert!(operators_equiv(&x, &i_x));
    }

    // ========================================================================
    // operator_exp tests
    // ========================================================================

    #[test]
    fn test_operator_exp_identity() {
        // exp(i * 0 * X) = I
        let x = X(0);
        let result = operator_exp(&x, 0.0);
        let identity = DMatrix::identity(2, 2);
        assert!(matrices_equiv_up_to_phase(&result, &identity, 1e-10));
    }

    #[test]
    fn test_operator_exp_pauli_pi() {
        // exp(i * π * Z) = -I
        let z = Z(0);
        let result = operator_exp(&z, PI);
        let neg_identity: DMatrix<Complex64> = DMatrix::identity(2, 2) * Complex64::new(-1.0, 0.0);
        assert!(matrices_equiv_up_to_phase(&result, &neg_identity, 1e-10));
    }

    #[test]
    fn test_operator_exp_pauli_half_pi() {
        // exp(i * π/2 * X) = i*X = [[0, i], [i, 0]]
        let x = X(0);
        let result = operator_exp(&x, PI / 2.0);
        let i = Complex64::new(0.0, 1.0);
        let expected = to_matrix(&x) * i;
        assert!(matrices_equiv_up_to_phase(&result, &expected, 1e-10));
    }

    // ========================================================================
    // operator_log tests
    // ========================================================================

    #[test]
    fn test_operator_log_identity() {
        // log(I) = 0
        let id = I(0);
        let result = operator_log(&id);
        assert!(result.is_some());
        let log_mat = result.unwrap();
        // All elements should be near zero
        for i in 0..log_mat.nrows() {
            for j in 0..log_mat.ncols() {
                assert!(log_mat[(i, j)].norm() < 1e-8);
            }
        }
    }

    #[test]
    fn test_operator_log_returns_matrix() {
        // log(T) should exist (T is close to identity)
        let t = T(0);
        let result = operator_log(&t);
        assert!(result.is_some());

        // log(S) should exist
        let s = SZ(0);
        let result = operator_log(&s);
        assert!(result.is_some());
    }

    // ========================================================================
    // to_matrix_with_size tests
    // ========================================================================

    #[test]
    fn test_to_matrix_with_size_embedding() {
        // X(0) in 3-qubit space should be 8x8
        let x = X(0);
        let mat = to_matrix_with_size(&x, 3);
        assert_eq!(mat.nrows(), 8);

        // Should act as X on qubit 0, identity on others
        // Check that |000> -> |001>, |001> -> |000>
        assert!((mat[(1, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(0, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
    }

    #[test]
    fn test_to_matrix_preserves_unitarity() {
        // Verify U * U† = I for various operators
        let operators = vec![X(0), Y(0), Z(0), H(0), T(0), CX(0, 1)];

        for op in operators {
            let mat = to_matrix(&op);
            let product = &mat * mat.adjoint();
            let identity: DMatrix<Complex64> = DMatrix::identity(mat.nrows(), mat.ncols());

            for i in 0..mat.nrows() {
                for j in 0..mat.ncols() {
                    assert!(
                        (product[(i, j)] - identity[(i, j)]).norm() < 1e-10,
                        "Unitarity failed for operator at ({i}, {j})"
                    );
                }
            }
        }
    }

    // ========================================================================
    // Conjugation matrix verification tests
    // ========================================================================

    #[test]
    fn test_conj_matrix_verification() {
        // Verify A.conj(U) = U * A * U† via matrices
        let a = X(0);
        let u = H(0);

        let conj_result = a.conj(&u);
        let conj_mat = to_matrix(&conj_result);

        // Compute U * A * U† directly
        let u_mat = to_matrix(&u);
        let a_mat = to_matrix(&a);
        let expected = &u_mat * &a_mat * u_mat.adjoint();

        assert!(matrices_equiv_up_to_phase(&conj_mat, &expected, 1e-10));
    }

    #[test]
    fn test_conjdg_matrix_verification() {
        // Verify A.conjdg(U) = U† * A * U via matrices
        let a = X(0);
        let u = H(0);

        let conjdg_result = a.conjdg(&u);
        let conjdg_mat = to_matrix(&conjdg_result);

        // Compute U† * A * U directly
        let u_mat = to_matrix(&u);
        let a_mat = to_matrix(&a);
        let expected = u_mat.adjoint() * &a_mat * &u_mat;

        assert!(matrices_equiv_up_to_phase(&conjdg_mat, &expected, 1e-10));
    }

    #[test]
    fn test_conj_sz_gives_y() {
        // X.conj(SZ) = SZ * X * SZ† should equal Y (up to phase)
        let x = X(0);
        let sz = SZ(0);

        let conj_result = x.conj(&sz);
        let conj_mat = to_matrix(&conj_result);

        let y_mat = to_matrix(&Y(0));

        assert!(matrices_equiv_up_to_phase(&conj_mat, &y_mat, 1e-10));
    }

    #[test]
    fn test_conj_conjdg_inverse_via_matrix() {
        // A.conj(U).conjdg(U) should equal A
        let a = X(0);
        let u = T(0);

        let forward = a.clone().conj(&u);
        let back = forward.conjdg(&u);
        let back_mat = to_matrix(&back);

        let a_mat = to_matrix(&a);

        assert!(matrices_equiv_up_to_phase(&back_mat, &a_mat, 1e-10));
    }

    // ========================================================================
    // Multi-qubit conjugation tests
    // ========================================================================

    #[test]
    fn test_conj_multi_qubit_stabilizer() {
        // Two-qubit stabilizer X⊗Z conjugated by CNOT
        let stabilizer = X(0) & Z(1);
        let cnot = CX(0, 1);

        let updated = stabilizer.conj(&cnot);
        let updated_mat = to_matrix(&updated);

        // Compute CNOT * (X⊗Z) * CNOT† directly
        let cnot_mat = to_matrix(&cnot);
        let stab_mat = to_matrix(&stabilizer);
        let expected = &cnot_mat * &stab_mat * cnot_mat.adjoint();

        assert!(matrices_equiv_up_to_phase(&updated_mat, &expected, 1e-10));
    }

    #[test]
    fn test_conj_by_two_qubit_gate() {
        // Single-qubit Pauli conjugated by two-qubit gate
        let x = X(0);
        let cnot = CX(0, 1);

        let result = x.conj(&cnot);
        let result_mat = to_matrix(&result);

        // CNOT * X(0) * CNOT† = X(0) ⊗ X(1) (CNOT propagates X from control to target)
        let xx = X(0) & X(1);
        let expected = to_matrix(&xx);

        assert!(matrices_equiv_up_to_phase(&result_mat, &expected, 1e-10));
    }

    // ========================================================================
    // More two-qubit gate tests
    // ========================================================================

    #[test]
    fn test_cz_gate() {
        // CZ = |0><0| ⊗ I + |1><1| ⊗ Z
        use pecos_core::operator::CZ;
        let cz = CZ(0, 1);
        let mat = to_matrix(&cz);

        // CZ matrix: diag(1, 1, 1, -1)
        assert!((mat[(0, 0)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(1, 1)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(2, 2)] - Complex64::new(1.0, 0.0)).norm() < 1e-10);
        assert!((mat[(3, 3)] - Complex64::new(-1.0, 0.0)).norm() < 1e-10);

        // Off-diagonal should be zero
        assert!(mat[(0, 1)].norm() < 1e-10);
        assert!(mat[(1, 2)].norm() < 1e-10);
    }

    #[test]
    fn test_cz_symmetric() {
        // CZ(0,1) should equal CZ(1,0)
        use pecos_core::operator::CZ;
        let cz_01 = CZ(0, 1);
        let cz_10 = CZ(1, 0);

        let mat_01 = to_matrix(&cz_01);
        let mat_10 = to_matrix(&cz_10);

        assert!(matrices_equiv_up_to_phase(&mat_01, &mat_10, 1e-10));
    }

    // ========================================================================
    // Algebraic identity tests
    // ========================================================================

    #[test]
    fn test_adjoint_of_product() {
        // (AB)† = B†A†
        let a = H(0);
        let b = T(0);

        let ab = a.clone() * b.clone();
        let ab_dagger = ab.dg();
        let ab_dagger_mat = to_matrix(&ab_dagger);

        let b_dagger_a_dagger = b.dg() * a.dg();
        let expected = to_matrix(&b_dagger_a_dagger);

        assert!(matrices_equiv_up_to_phase(&ab_dagger_mat, &expected, 1e-10));
    }

    #[test]
    fn test_double_adjoint_identity() {
        // (A†)† = A
        let ops = vec![X(0), Y(0), H(0), T(0), CX(0, 1)];

        for op in ops {
            let double_dagger = op.dg().dg();
            let original_mat = to_matrix(&op);
            let double_mat = to_matrix(&double_dagger);

            assert!(matrices_equiv_up_to_phase(
                &original_mat,
                &double_mat,
                1e-10
            ));
        }
    }

    #[test]
    fn test_tensor_adjoint() {
        // (A ⊗ B)† = A† ⊗ B†
        let a = H(0);
        let b = T(1);

        let tensor = a.clone() & b.clone();
        let tensor_dagger = tensor.dg();
        let tensor_dagger_mat = to_matrix(&tensor_dagger);

        let a_dagger_tensor_b_dagger = a.dg() & b.dg();
        let expected = to_matrix(&a_dagger_tensor_b_dagger);

        assert!(matrices_equiv_up_to_phase(
            &tensor_dagger_mat,
            &expected,
            1e-10
        ));
    }

    // ========================================================================
    // ToMatrix trait tests
    // ========================================================================

    #[test]
    fn test_to_matrix_trait_method() {
        // Test that trait method gives same result as standalone function
        let h = H(0);

        let via_function = to_matrix(&h);
        let via_trait = h.to_matrix();

        assert_eq!(via_function.nrows(), via_trait.nrows());
        assert_eq!(via_function.ncols(), via_trait.ncols());
        for i in 0..via_function.nrows() {
            for j in 0..via_function.ncols() {
                assert!((via_function[(i, j)] - via_trait[(i, j)]).norm() < 1e-10);
            }
        }
    }

    #[test]
    fn test_to_matrix_with_identity_tensor() {
        // Test using Is() to get larger matrix
        let x_extended = X(0) & Is(1..3); // X on qubit 0 in 3-qubit space

        let mat = x_extended.to_matrix();
        assert_eq!(mat.nrows(), 8); // 2^3 = 8

        // Should match the standalone function
        let expected = to_matrix_with_size(&X(0), 3);
        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_to_matrix_trait_chaining() {
        // Verify trait works well with operator chaining
        let circuit = H(0) * CX(0, 1) * H(0);
        let mat = circuit.to_matrix();

        assert_eq!(mat.nrows(), 4); // 2 qubits

        // Verify unitarity
        let product = &mat * mat.adjoint();
        let identity: DMatrix<Complex64> = DMatrix::identity(4, 4);
        assert!(matrices_equiv_up_to_phase(&product, &identity, 1e-10));
    }

    // ========================================================================
    // Identity operator ToMatrix tests
    // ========================================================================

    #[test]
    fn test_identity_to_matrix_single_qubit() {
        // I(0).to_matrix() should be 2x2 identity
        let mat = I(0).to_matrix();
        let expected: DMatrix<Complex64> = DMatrix::identity(2, 2);

        assert_eq!(mat.nrows(), 2);
        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_identity_to_matrix_two_qubits() {
        // Is(0..=1).to_matrix() should be 4x4 identity
        let mat = Is(0..=1).to_matrix();
        let expected: DMatrix<Complex64> = DMatrix::identity(4, 4);

        assert_eq!(mat.nrows(), 4);
        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_identity_tensor_with_gate() {
        // X(0) & I(1) should give X tensor I = 4x4 matrix
        let op = X(0) & I(1);
        let mat = op.to_matrix();

        assert_eq!(mat.nrows(), 4);

        // Should equal X(0) extended to 2 qubits
        let expected = to_matrix_with_size(&X(0), 2);

        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_simplify_preserves_tensor_dimension() {
        // (X(0) & I(1)).simplify() should preserve the 2-qubit space
        let op = X(0) & I(1);
        let simplified = op.simplify();

        // Both should produce equivalent 4x4 matrices
        let orig_mat = op.to_matrix();
        let simp_mat = simplified.to_matrix();

        assert_eq!(orig_mat.nrows(), 4);
        assert_eq!(simp_mat.nrows(), 4);
        assert!(matrices_equiv_up_to_phase(&orig_mat, &simp_mat, 1e-10));
    }

    // ========================================================================
    // PauliString ToMatrix tests
    // ========================================================================

    #[test]
    fn test_pauli_string_to_matrix_single() {
        use pecos_core::PauliString;

        // Single X Pauli
        let ps = PauliString::x(0);
        let mat = ps.to_matrix();

        // Should match X(0).to_matrix()
        let x_mat = X(0).to_matrix();
        assert!(matrices_equiv_up_to_phase(&mat, &x_mat, 1e-10));
    }

    #[test]
    fn test_pauli_string_to_matrix_multi() {
        use pecos_core::{Pauli, PauliString};

        // X on qubit 0, Z on qubit 1
        let ps = PauliString::from_paulis(&[Pauli::X, Pauli::Z]);
        let mat = ps.to_matrix();

        // Should match (X(0) & Z(1)).to_matrix()
        let xz = X(0) & Z(1);
        let expected = xz.to_matrix();
        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_pauli_string_to_matrix_with_phase() {
        use pecos_core::{Pauli, PauliString, QuarterPhase};

        // -i * X
        let ps = PauliString::from_paulis_with_phase(QuarterPhase::MinusI, &[Pauli::X]);
        let mat = ps.to_matrix();

        // Should be -i times X matrix
        let x_mat = X(0).to_matrix();
        let neg_i = Complex64::new(0.0, -1.0);
        let expected = x_mat * neg_i;

        assert!(matrices_equiv_up_to_phase(&mat, &expected, 1e-10));
    }

    #[test]
    fn test_pauli_string_to_matrix_identity() {
        use pecos_core::PauliString;

        // Identity PauliString - returns 1x1 identity (no qubits)
        let ps = PauliString::identity();
        let mat = ps.to_matrix();

        // Identity with no qubits defaults to 1 qubit -> 2x2
        let identity: DMatrix<Complex64> = DMatrix::identity(2, 2);
        assert!(matrices_equiv_up_to_phase(&mat, &identity, 1e-10));
    }

    #[test]
    fn test_pauli_string_matches_operator_pauli() {
        use pecos_core::{Pauli, PauliString};

        // Verify PauliString.to_matrix() matches Operator::Pauli.to_matrix()
        let ps = PauliString::from_paulis(&[Pauli::Y, Pauli::Z]);

        // Convert to Operator::Pauli
        let op = pecos_core::operator::Operator::Pauli(ps.clone());

        let ps_mat = ps.to_matrix();
        let op_mat = op.to_matrix();

        assert!(matrices_equiv_up_to_phase(&ps_mat, &op_mat, 1e-10));
    }
}
