// Copyright 2026 The PECOS Developers
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

//! Symplectic representation of Pauli operators and Clifford unitaries.
//!
//! This module provides dense GF(2) matrix representations that bridge
//! [`CliffordRep`](pecos_core::clifford_rep::CliffordRep) (which stores
//! Clifford actions as Pauli string images) with
//! [`Mat2`](quizx::linalg::Mat2) (which provides GF(2) linear algebra).
//!
//! # Types
//!
//! - [`SymplecticVector`] -- a Pauli operator on n qubits as a 2n-bit vector
//! - [`SymplecticMatrix`] -- a Clifford unitary on n qubits as a 2n x 2n binary matrix
//!
//! # Layout
//!
//! Bit vectors use the convention `[x_0, ..., x_{n-1}, z_0, ..., z_{n-1}]`.
//! - X = (1, 0), Z = (0, 1), Y = (1, 1), I = (0, 0)
//!
//! Matrix rows: rows 0..n are images of X_0..X_{n-1}, rows n..2n are images
//! of Z_0..Z_{n-1}.

use pecos_core::clifford_rep::CliffordRep;
use pecos_core::{Pauli, PauliString, QuarterPhase};
use quizx::linalg::Mat2;

// --- SymplecticVector ---

/// A Pauli operator on n qubits represented as a 2n-bit symplectic vector.
///
/// Layout: `[x_0, ..., x_{n-1}, z_0, ..., z_{n-1}]` with a [`QuarterPhase`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymplecticVector {
    n: usize,
    bits: Vec<u8>,
    phase: QuarterPhase,
}

impl SymplecticVector {
    /// Identity operator on n qubits (all zero bits, phase +1).
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self {
            n,
            bits: vec![0; 2 * n],
            phase: QuarterPhase::PlusOne,
        }
    }

    /// Construct from separate x and z bit slices plus phase.
    ///
    /// # Panics
    ///
    /// Panics if `x_bits` or `z_bits` have length != `n`.
    #[must_use]
    pub fn new(n: usize, x_bits: &[u8], z_bits: &[u8], phase: QuarterPhase) -> Self {
        assert_eq!(x_bits.len(), n);
        assert_eq!(z_bits.len(), n);
        let mut bits = Vec::with_capacity(2 * n);
        bits.extend_from_slice(x_bits);
        bits.extend_from_slice(z_bits);
        Self { n, bits, phase }
    }

    /// Construct from a flat 2n-bit row plus phase.
    ///
    /// # Panics
    ///
    /// Panics if `row.len() != 2 * n`.
    #[must_use]
    pub fn from_row(n: usize, row: &[u8], phase: QuarterPhase) -> Self {
        assert_eq!(row.len(), 2 * n);
        Self {
            n,
            bits: row.to_vec(),
            phase,
        }
    }

    /// Number of qubits.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// The phase of this Pauli operator.
    #[must_use]
    pub fn phase(&self) -> QuarterPhase {
        self.phase
    }

    /// The full 2n-bit vector.
    #[must_use]
    pub fn bits(&self) -> &[u8] {
        &self.bits
    }

    /// The x-part of the bit vector (first n bits).
    #[must_use]
    pub fn x_bits(&self) -> &[u8] {
        &self.bits[..self.n]
    }

    /// The z-part of the bit vector (last n bits).
    #[must_use]
    pub fn z_bits(&self) -> &[u8] {
        &self.bits[self.n..]
    }

    /// The single-qubit Pauli at position `i`.
    #[must_use]
    pub fn pauli_at(&self, i: usize) -> Pauli {
        match (self.bits[i], self.bits[self.n + i]) {
            (0, 0) => Pauli::I,
            (1, 0) => Pauli::X,
            (0, 1) => Pauli::Z,
            (1, 1) => Pauli::Y,
            _ => unreachable!("bits should be 0 or 1"),
        }
    }

    /// Number of non-identity single-qubit Paulis (Hamming weight).
    #[must_use]
    pub fn weight(&self) -> usize {
        (0..self.n)
            .filter(|&i| self.bits[i] != 0 || self.bits[self.n + i] != 0)
            .count()
    }

    /// Convert to a [`PauliString`].
    #[must_use]
    pub fn to_pauli_string(&self) -> PauliString {
        let mut x_qubits = Vec::new();
        let mut y_qubits = Vec::new();
        let mut z_qubits = Vec::new();

        for i in 0..self.n {
            match self.pauli_at(i) {
                Pauli::X => x_qubits.push(i),
                Pauli::Y => y_qubits.push(i),
                Pauli::Z => z_qubits.push(i),
                Pauli::I => {}
            }
        }

        PauliString::from_decomposed(self.phase, x_qubits, y_qubits, z_qubits)
    }

    /// Convert from a [`PauliString`], embedding into `n` qubits.
    ///
    /// Qubits beyond the PauliString's range are treated as identity.
    #[must_use]
    pub fn from_pauli_string(ps: &PauliString, n: usize) -> Self {
        let mut bits = vec![0u8; 2 * n];

        for (pauli, qubit_id) in ps.iter_pairs() {
            let q = usize::from(qubit_id);
            if q < n {
                match pauli {
                    Pauli::X => bits[q] = 1,
                    Pauli::Z => bits[n + q] = 1,
                    Pauli::Y => {
                        bits[q] = 1;
                        bits[n + q] = 1;
                    }
                    Pauli::I => {}
                }
            }
        }

        Self {
            n,
            bits,
            phase: ps.phase(),
        }
    }
}

// --- Free functions ---

/// The 2n x 2n symplectic form matrix: `[[0, I], [I, 0]]`.
#[must_use]
pub fn omega(n: usize) -> Mat2 {
    Mat2::build(
        2 * n,
        2 * n,
        |i, j| {
            if i < n { j == i + n } else { j == i - n }
        },
    )
}

/// GF(2) symplectic inner product: `sum_i (a_x[i]*b_z[i] + a_z[i]*b_x[i]) mod 2`.
///
/// Returns 0 if commuting, 1 if anticommuting.
#[must_use]
pub fn symplectic_inner_product(a: &[u8], b: &[u8], n: usize) -> u8 {
    assert_eq!(a.len(), 2 * n);
    assert_eq!(b.len(), 2 * n);

    let mut acc = 0u8;
    for i in 0..n {
        acc ^= a[i] & b[n + i]; // a_x[i] * b_z[i]
        acc ^= a[n + i] & b[i]; // a_z[i] * b_x[i]
    }
    acc
}

/// Check whether a 2n x 2n binary matrix is symplectic: `M^T * omega * M == omega`.
#[must_use]
pub fn is_symplectic(mat: &Mat2, n: usize) -> bool {
    assert_eq!(mat.num_rows(), 2 * n);
    assert_eq!(mat.num_cols(), 2 * n);
    let om = omega(n);
    let product = &mat.transpose() * &(&om * mat);
    product == om
}

// --- SymplecticMatrix ---

/// A Clifford unitary on n qubits as a 2n x 2n binary matrix with sign bits.
///
/// Row layout: rows 0..n = images of X_0..X_{n-1}, rows n..2n = images of
/// Z_0..Z_{n-1}.
///
/// `signs[j]` is `true` when the image of the j-th basis generator has a
/// minus sign.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SymplecticMatrix {
    n: usize,
    mat: Mat2,
    signs: Vec<bool>,
}

impl SymplecticMatrix {
    /// Identity Clifford on n qubits.
    #[must_use]
    pub fn identity(n: usize) -> Self {
        Self {
            n,
            mat: Mat2::id(2 * n),
            signs: vec![false; 2 * n],
        }
    }

    /// Construct from a matrix and sign vector.
    ///
    /// # Panics
    ///
    /// Panics if dimensions are inconsistent.
    #[must_use]
    pub fn new(n: usize, mat: Mat2, signs: Vec<bool>) -> Self {
        assert_eq!(mat.num_rows(), 2 * n);
        assert_eq!(mat.num_cols(), 2 * n);
        assert_eq!(signs.len(), 2 * n);
        Self { n, mat, signs }
    }

    /// Number of qubits.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.n
    }

    /// The underlying 2n x 2n binary matrix.
    #[must_use]
    pub fn binary_matrix(&self) -> &Mat2 {
        &self.mat
    }

    /// The sign vector (length 2n).
    #[must_use]
    pub fn signs(&self) -> &[bool] {
        &self.signs
    }

    /// The image of the j-th basis generator as a `SymplecticVector`.
    #[must_use]
    pub fn image(&self, j: usize) -> SymplecticVector {
        let row: Vec<u8> = (0..2 * self.n).map(|c| self.mat[(j, c)]).collect();
        let phase = if self.signs[j] {
            QuarterPhase::MinusOne
        } else {
            QuarterPhase::PlusOne
        };
        SymplecticVector::from_row(self.n, &row, phase)
    }

    /// The image of X on qubit q.
    #[must_use]
    pub fn x_image(&self, q: usize) -> SymplecticVector {
        self.image(q)
    }

    /// The image of Z on qubit q.
    #[must_use]
    pub fn z_image(&self, q: usize) -> SymplecticVector {
        self.image(self.n + q)
    }

    /// Check whether the binary matrix is symplectic.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        is_symplectic(&self.mat, self.n)
    }

    /// Compose two Cliffords: `self * other` (apply other first, then self).
    ///
    /// Delegates to [`CliffordRep`] for correct sign handling.
    #[must_use]
    pub fn compose(&self, other: &SymplecticMatrix) -> SymplecticMatrix {
        let self_cliff = self.to_clifford_rep();
        let other_cliff = other.to_clifford_rep();
        let composed = self_cliff.compose(&other_cliff);
        SymplecticMatrix::from_clifford_rep(&composed)
    }

    /// Compute the inverse Clifford.
    ///
    /// Binary part: `M^{-1} = omega * M^T * omega`.
    /// Signs: recovered by checking forward-map of each unsigned inverse image.
    #[must_use]
    pub fn inverse(&self) -> SymplecticMatrix {
        let om = omega(self.n);
        let inv_mat = &om * &(&self.mat.transpose() * &om);

        // Recover signs: for each generator g_k, compute the unsigned inverse
        // image P_k (from inv_mat row k, with sign=false). Apply the forward
        // Clifford. If the result is -g_k, the sign should be true.
        let forward = self.to_clifford_rep();
        let mut inv_signs = vec![false; 2 * self.n];

        for k in 0..2 * self.n {
            let row: Vec<u8> = (0..2 * self.n).map(|c| inv_mat[(k, c)]).collect();
            let unsigned_image = SymplecticVector::from_row(self.n, &row, QuarterPhase::PlusOne);
            let ps = unsigned_image.to_pauli_string();
            let mapped = forward.apply(&ps);

            // The mapped result should be +/- g_k (the k-th generator).
            // g_k: for k < n it's X_k, for k >= n it's Z_{k-n}.
            inv_signs[k] = mapped.phase() == QuarterPhase::MinusOne;
        }

        SymplecticMatrix::new(self.n, inv_mat, inv_signs)
    }

    /// Rank of the binary matrix (delegates to [`Mat2::rank`]).
    #[must_use]
    pub fn rank(&self) -> usize {
        self.mat.rank()
    }

    /// Convert to a [`CliffordRep`].
    #[must_use]
    pub fn to_clifford_rep(&self) -> CliffordRep {
        let mut rep = CliffordRep::identity(self.n);
        for q in 0..self.n {
            let x_vec = self.image(q);
            rep.set_x_image(q, x_vec.to_pauli_string());

            let z_vec = self.image(self.n + q);
            rep.set_z_image(q, z_vec.to_pauli_string());
        }
        rep
    }

    /// Convert from a [`CliffordRep`].
    ///
    /// # Panics
    ///
    /// Panics if any generator image has a phase other than +1 or -1.
    #[must_use]
    pub fn from_clifford_rep(rep: &CliffordRep) -> Self {
        let n = rep.num_qubits();
        let dim = 2 * n;
        let mut mat = Mat2::zeros(dim, dim);
        let mut signs = vec![false; dim];

        for q in 0..n {
            // X image -> row q
            let x_img = rep.x_image(q);
            assert!(
                x_img.phase() == QuarterPhase::PlusOne || x_img.phase() == QuarterPhase::MinusOne,
                "X image for qubit {q} has non-real phase {:?}",
                x_img.phase()
            );
            signs[q] = x_img.phase() == QuarterPhase::MinusOne;

            for i in 0..n {
                let p = x_img.get(i);
                // x-bit
                if matches!(p, Pauli::X | Pauli::Y) {
                    mat[(q, i)] = 1;
                }
                // z-bit
                if matches!(p, Pauli::Z | Pauli::Y) {
                    mat[(q, n + i)] = 1;
                }
            }

            // Z image -> row n+q
            let z_img = rep.z_image(q);
            assert!(
                z_img.phase() == QuarterPhase::PlusOne || z_img.phase() == QuarterPhase::MinusOne,
                "Z image for qubit {q} has non-real phase {:?}",
                z_img.phase()
            );
            signs[n + q] = z_img.phase() == QuarterPhase::MinusOne;

            for i in 0..n {
                let p = z_img.get(i);
                if matches!(p, Pauli::X | Pauli::Y) {
                    mat[(n + q, i)] = 1;
                }
                if matches!(p, Pauli::Z | Pauli::Y) {
                    mat[(n + q, n + i)] = 1;
                }
            }
        }

        Self { n, mat, signs }
    }

    // --- Gate factories ---

    /// Hadamard gate on the given qubit (in an n-qubit system).
    #[must_use]
    pub fn h(qubit: usize, n: usize) -> Self {
        Self::from_clifford_rep(&padded_clifford(CliffordRep::h(qubit), n))
    }

    /// S gate on the given qubit.
    #[must_use]
    pub fn s(qubit: usize, n: usize) -> Self {
        Self::from_clifford_rep(&padded_clifford(CliffordRep::sz(qubit), n))
    }

    /// S-dagger gate on the given qubit.
    #[must_use]
    pub fn sdg(qubit: usize, n: usize) -> Self {
        Self::from_clifford_rep(&padded_clifford(CliffordRep::szdg(qubit), n))
    }

    /// Pauli X gate on the given qubit.
    #[must_use]
    pub fn x(qubit: usize, n: usize) -> Self {
        Self::from_clifford_rep(&CliffordRep::x_on(qubit, n))
    }

    /// Pauli Y gate on the given qubit.
    #[must_use]
    pub fn y(qubit: usize, n: usize) -> Self {
        Self::from_clifford_rep(&CliffordRep::y_on(qubit, n))
    }

    /// Pauli Z gate on the given qubit.
    #[must_use]
    pub fn z(qubit: usize, n: usize) -> Self {
        Self::from_clifford_rep(&CliffordRep::z_on(qubit, n))
    }

    /// SX (sqrt-X) gate on the given qubit.
    #[must_use]
    pub fn sx(qubit: usize, n: usize) -> Self {
        Self::from_clifford_rep(&padded_clifford(CliffordRep::sx(qubit), n))
    }

    /// SY (sqrt-Y) gate on the given qubit.
    #[must_use]
    pub fn sy(qubit: usize, n: usize) -> Self {
        Self::from_clifford_rep(&padded_clifford(CliffordRep::sy(qubit), n))
    }

    /// SZ (sqrt-Z, same as S) gate on the given qubit.
    #[must_use]
    pub fn sz(qubit: usize, n: usize) -> Self {
        Self::s(qubit, n)
    }

    /// CX (CNOT) gate.
    #[must_use]
    pub fn cx(control: usize, target: usize, n: usize) -> Self {
        Self::from_clifford_rep(&padded_clifford(CliffordRep::cx(control, target), n))
    }

    /// CZ gate.
    #[must_use]
    pub fn cz(q0: usize, q1: usize, n: usize) -> Self {
        Self::from_clifford_rep(&padded_clifford(CliffordRep::cz(q0, q1), n))
    }

    /// SWAP gate.
    #[must_use]
    pub fn swap(q0: usize, q1: usize, n: usize) -> Self {
        Self::from_clifford_rep(&padded_clifford(CliffordRep::swap(q0, q1), n))
    }
}

/// Pad a CliffordRep to n qubits by composing with identity.
///
/// CliffordRep gate factories produce representations sized to the largest
/// qubit index + 1. This pads to exactly n qubits.
pub(crate) fn padded_clifford(rep: CliffordRep, n: usize) -> CliffordRep {
    if rep.num_qubits() >= n {
        return rep;
    }
    let id = CliffordRep::identity(n);
    // The gate rep acts on qubits 0..rep.num_qubits().
    // We need to build a new CliffordRep on n qubits with the same images
    // for qubits the gate touches, and identity for the rest.
    let mut result = id;
    for q in 0..rep.num_qubits() {
        result.set_x_image(q, rep.x_image(q).clone());
        result.set_z_image(q, rep.z_image(q).clone());
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- SymplecticVector tests ---

    #[test]
    fn vector_identity() {
        let v = SymplecticVector::identity(3);
        assert_eq!(v.num_qubits(), 3);
        assert_eq!(v.phase(), QuarterPhase::PlusOne);
        assert_eq!(v.weight(), 0);
        for i in 0..3 {
            assert_eq!(v.pauli_at(i), Pauli::I);
        }
    }

    #[test]
    fn vector_single_x() {
        let ps = PauliString::x(1);
        let v = SymplecticVector::from_pauli_string(&ps, 3);
        assert_eq!(v.pauli_at(0), Pauli::I);
        assert_eq!(v.pauli_at(1), Pauli::X);
        assert_eq!(v.pauli_at(2), Pauli::I);
        assert_eq!(v.weight(), 1);
    }

    #[test]
    fn vector_single_z() {
        let ps = PauliString::z(0);
        let v = SymplecticVector::from_pauli_string(&ps, 2);
        assert_eq!(v.pauli_at(0), Pauli::Z);
        assert_eq!(v.pauli_at(1), Pauli::I);
        assert_eq!(v.x_bits(), &[0, 0]);
        assert_eq!(v.z_bits(), &[1, 0]);
    }

    #[test]
    fn vector_single_y() {
        let ps = PauliString::y(0);
        let v = SymplecticVector::from_pauli_string(&ps, 1);
        assert_eq!(v.pauli_at(0), Pauli::Y);
        assert_eq!(v.x_bits(), &[1]);
        assert_eq!(v.z_bits(), &[1]);
    }

    #[test]
    fn vector_multi_qubit() {
        // X on 0, Y on 1, Z on 2
        let ps = PauliString::from_paulis(&[Pauli::X, Pauli::Y, Pauli::Z]);
        let v = SymplecticVector::from_pauli_string(&ps, 3);
        assert_eq!(v.pauli_at(0), Pauli::X);
        assert_eq!(v.pauli_at(1), Pauli::Y);
        assert_eq!(v.pauli_at(2), Pauli::Z);
        assert_eq!(v.weight(), 3);
        assert_eq!(v.x_bits(), &[1, 1, 0]);
        assert_eq!(v.z_bits(), &[0, 1, 1]);
    }

    #[test]
    fn vector_roundtrip() {
        let ps = PauliString::from_paulis_with_phase(
            QuarterPhase::MinusOne,
            &[Pauli::X, Pauli::Y, Pauli::Z],
        );
        let v = SymplecticVector::from_pauli_string(&ps, 3);
        let ps2 = v.to_pauli_string();
        assert_eq!(ps2.phase(), ps.phase());
        for q in 0..3 {
            assert_eq!(ps2.get(q), ps.get(q));
        }
    }

    #[test]
    fn vector_phase_propagation() {
        let ps = PauliString::from_paulis_with_phase(QuarterPhase::MinusOne, &[Pauli::X]);
        let v = SymplecticVector::from_pauli_string(&ps, 1);
        assert_eq!(v.phase(), QuarterPhase::MinusOne);
        let back = v.to_pauli_string();
        assert_eq!(back.phase(), QuarterPhase::MinusOne);
    }

    // --- Symplectic form tests ---

    #[test]
    fn inner_product_commuting() {
        // X_0 and X_0 commute
        let a = [1, 0, 0, 0]; // X on qubit 0 in 2-qubit system
        let b = [1, 0, 0, 0];
        assert_eq!(symplectic_inner_product(&a, &b, 2), 0);
    }

    #[test]
    fn inner_product_anticommuting() {
        // X_0 and Z_0 anticommute
        let a = [1, 0, 0, 0]; // X on qubit 0
        let b = [0, 0, 1, 0]; // Z on qubit 0
        assert_eq!(symplectic_inner_product(&a, &b, 2), 1);
    }

    #[test]
    fn inner_product_different_qubits() {
        // X_0 and Z_1 commute
        let a = [1, 0, 0, 0]; // X on qubit 0
        let b = [0, 0, 0, 1]; // Z on qubit 1
        assert_eq!(symplectic_inner_product(&a, &b, 2), 0);
    }

    #[test]
    fn omega_structure() {
        let om = omega(2);
        // Should be [[0,0,1,0],[0,0,0,1],[1,0,0,0],[0,1,0,0]]
        let expected = Mat2::new(vec![
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 1],
            vec![1, 0, 0, 0],
            vec![0, 1, 0, 0],
        ]);
        assert_eq!(om, expected);
    }

    #[test]
    fn identity_is_symplectic() {
        assert!(is_symplectic(&Mat2::id(4), 2));
        assert!(is_symplectic(&Mat2::id(6), 3));
    }

    #[test]
    fn non_symplectic_rejection() {
        // A random non-symplectic matrix
        let m = Mat2::new(vec![
            vec![1, 1, 0, 0],
            vec![0, 1, 0, 0],
            vec![0, 0, 1, 0],
            vec![0, 0, 0, 1],
        ]);
        assert!(!is_symplectic(&m, 2));
    }

    // --- SymplecticMatrix basics ---

    #[test]
    fn matrix_identity_structure() {
        let id = SymplecticMatrix::identity(2);
        assert_eq!(id.num_qubits(), 2);
        assert_eq!(*id.binary_matrix(), Mat2::id(4));
        assert_eq!(id.signs(), &[false, false, false, false]);
    }

    #[test]
    fn matrix_identity_is_valid() {
        let id = SymplecticMatrix::identity(3);
        assert!(id.is_valid());
    }

    #[test]
    fn clifford_rep_roundtrip() {
        // Build a CliffordRep, convert to SymplecticMatrix, convert back
        let cliff = CliffordRep::h(0);
        let padded = padded_clifford(cliff, 2);
        let sym = SymplecticMatrix::from_clifford_rep(&padded);
        let cliff2 = sym.to_clifford_rep();

        // Verify they produce the same images
        for q in 0..2 {
            assert_eq!(
                padded.x_image(q).phase(),
                cliff2.x_image(q).phase(),
                "X image phase mismatch on qubit {q}"
            );
            assert_eq!(
                padded.z_image(q).phase(),
                cliff2.z_image(q).phase(),
                "Z image phase mismatch on qubit {q}"
            );
            for i in 0..2 {
                assert_eq!(
                    padded.x_image(q).get(i),
                    cliff2.x_image(q).get(i),
                    "X image Pauli mismatch at qubit {q}, position {i}"
                );
                assert_eq!(
                    padded.z_image(q).get(i),
                    cliff2.z_image(q).get(i),
                    "Z image Pauli mismatch at qubit {q}, position {i}"
                );
            }
        }
    }

    // --- Gate validation ---

    #[test]
    fn all_single_qubit_gates_are_symplectic() {
        let n = 2;
        let gates = [
            SymplecticMatrix::h(0, n),
            SymplecticMatrix::s(0, n),
            SymplecticMatrix::sdg(0, n),
            SymplecticMatrix::x(0, n),
            SymplecticMatrix::y(0, n),
            SymplecticMatrix::z(0, n),
            SymplecticMatrix::sx(0, n),
            SymplecticMatrix::sy(0, n),
            SymplecticMatrix::sz(0, n),
        ];
        for (i, g) in gates.iter().enumerate() {
            assert!(
                g.is_valid(),
                "Single-qubit gate index {i} is not symplectic"
            );
        }
    }

    #[test]
    fn all_two_qubit_gates_are_symplectic() {
        let n = 2;
        let gates = [
            SymplecticMatrix::cx(0, 1, n),
            SymplecticMatrix::cz(0, 1, n),
            SymplecticMatrix::swap(0, 1, n),
        ];
        for (i, g) in gates.iter().enumerate() {
            assert!(g.is_valid(), "Two-qubit gate index {i} is not symplectic");
        }
    }

    #[test]
    fn gates_match_clifford_rep() {
        let n = 2;

        // For each gate, verify from_clifford_rep -> to_clifford_rep roundtrip
        let gate_pairs: Vec<(SymplecticMatrix, CliffordRep)> = vec![
            (
                SymplecticMatrix::h(0, n),
                padded_clifford(CliffordRep::h(0), n),
            ),
            (
                SymplecticMatrix::s(0, n),
                padded_clifford(CliffordRep::sz(0), n),
            ),
            (
                SymplecticMatrix::sdg(0, n),
                padded_clifford(CliffordRep::szdg(0), n),
            ),
            (
                SymplecticMatrix::cx(0, 1, n),
                padded_clifford(CliffordRep::cx(0, 1), n),
            ),
            (
                SymplecticMatrix::cz(0, 1, n),
                padded_clifford(CliffordRep::cz(0, 1), n),
            ),
        ];

        for (sym, cliff) in gate_pairs {
            let cliff2 = sym.to_clifford_rep();
            for q in 0..n {
                assert_eq!(cliff.x_image(q).phase(), cliff2.x_image(q).phase());
                assert_eq!(cliff.z_image(q).phase(), cliff2.z_image(q).phase());
                for i in 0..n {
                    assert_eq!(cliff.x_image(q).get(i), cliff2.x_image(q).get(i));
                    assert_eq!(cliff.z_image(q).get(i), cliff2.z_image(q).get(i));
                }
            }
        }
    }

    #[test]
    fn all_gate_ranks_are_full() {
        let n = 2;
        let gates = [
            SymplecticMatrix::h(0, n),
            SymplecticMatrix::s(0, n),
            SymplecticMatrix::cx(0, 1, n),
            SymplecticMatrix::cz(0, 1, n),
            SymplecticMatrix::swap(0, 1, n),
            SymplecticMatrix::identity(n),
        ];
        for (i, g) in gates.iter().enumerate() {
            assert_eq!(g.rank(), 2 * n, "Gate index {i} does not have full rank");
        }
    }

    // --- Composition ---

    #[test]
    fn h_squared_is_identity() {
        let n = 2;
        let h = SymplecticMatrix::h(0, n);
        let hh = h.compose(&h);
        let id = SymplecticMatrix::identity(n);
        assert_eq!(hh, id);
    }

    #[test]
    fn s_squared_is_z() {
        let n = 2;
        let s = SymplecticMatrix::s(0, n);
        let ss = s.compose(&s);
        let z = SymplecticMatrix::z(0, n);
        assert_eq!(ss, z);
    }

    #[test]
    fn cx_squared_is_identity() {
        let n = 2;
        let cx = SymplecticMatrix::cx(0, 1, n);
        let cx2 = cx.compose(&cx);
        let id = SymplecticMatrix::identity(n);
        assert_eq!(cx2, id);
    }

    #[test]
    fn compose_matches_clifford_rep() {
        let n = 2;
        // H then S then CX -- build via both paths
        let h = SymplecticMatrix::h(0, n);
        let s = SymplecticMatrix::s(1, n);
        let cx = SymplecticMatrix::cx(0, 1, n);

        let sym_composed = cx.compose(&s.compose(&h));

        let h_c = padded_clifford(CliffordRep::h(0), n);
        let s_c = padded_clifford(CliffordRep::sz(1), n);
        let cx_c = padded_clifford(CliffordRep::cx(0, 1), n);

        let cliff_composed = cx_c.compose(&s_c.compose(&h_c));
        let sym_from_cliff = SymplecticMatrix::from_clifford_rep(&cliff_composed);

        assert_eq!(sym_composed, sym_from_cliff);
    }

    // --- Inverse ---

    #[test]
    fn identity_inverse() {
        let id = SymplecticMatrix::identity(2);
        assert_eq!(id.inverse(), id);
    }

    #[test]
    fn h_self_inverse() {
        let n = 2;
        let h = SymplecticMatrix::h(0, n);
        let h_inv = h.inverse();
        assert_eq!(h, h_inv);
    }

    #[test]
    fn s_inverse_is_sdg() {
        let n = 2;
        let s = SymplecticMatrix::s(0, n);
        let s_inv = s.inverse();
        let sdg = SymplecticMatrix::sdg(0, n);
        assert_eq!(s_inv, sdg);
    }

    #[test]
    fn cx_times_inverse_is_identity() {
        let n = 2;
        let cx = SymplecticMatrix::cx(0, 1, n);
        let cx_inv = cx.inverse();
        let product = cx.compose(&cx_inv);
        assert_eq!(product, SymplecticMatrix::identity(n));
    }

    #[test]
    fn cz_times_inverse_is_identity() {
        let n = 2;
        let cz = SymplecticMatrix::cz(0, 1, n);
        let cz_inv = cz.inverse();
        let product = cz.compose(&cz_inv);
        assert_eq!(product, SymplecticMatrix::identity(n));
    }

    // --- Rank ---

    #[test]
    fn identity_rank_is_2n() {
        assert_eq!(SymplecticMatrix::identity(3).rank(), 6);
    }

    #[test]
    fn gate_rank_is_2n() {
        let n = 3;
        assert_eq!(SymplecticMatrix::h(0, n).rank(), 6);
        assert_eq!(SymplecticMatrix::cx(0, 1, n).rank(), 6);
    }
}
