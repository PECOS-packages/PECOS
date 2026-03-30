// Copyright 2024 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Clifford gate representation via Heisenberg picture (generator propagation).
//!
//! A Clifford gate is fully specified by how it transforms the Pauli generators.
//! For n qubits, we track 2n generators (`X_i` and `Z_i` for each qubit i).
//!
//! # Example
//!
//! ```
//! use pecos_core::clifford_rep::CliffordRep;
//! use pecos_core::unitary_rep::{X, Z};
//!
//! // Hadamard swaps X <-> Z
//! let h = CliffordRep::h(0);
//! let stabilizer = X(0) & Z(1);
//! let transformed = h.apply_to(&stabilizer).unwrap();
//! // H transforms X(0) -> Z(0), Z(1) unchanged
//! ```

use crate::pauli::algebra::i;
use crate::unitary_rep::UnitaryRep;
use crate::{Pauli, PauliString, Phase, QuarterPhase};
use rand::RngExt;
use std::fmt;
use std::ops::{BitAnd, Mul};

/// Clifford gate representation via generator propagation (Heisenberg picture).
///
/// Stores how each input generator (`X_i`, `Z_i`) maps to an output `PauliString`.
/// This representation allows efficient composition and Pauli transformation.
#[derive(Debug, Clone, PartialEq)]
pub struct CliffordRep {
    /// Number of qubits this Clifford acts on
    num_qubits: usize,
    /// How `X_i` transforms: `x_images`[i] = image of X on qubit i
    x_images: Vec<PauliString>,
    /// How `Z_i` transforms: `z_images`[i] = image of Z on qubit i
    z_images: Vec<PauliString>,
}

impl CliffordRep {
    /// Creates a new Clifford representation for the given number of qubits.
    ///
    /// Initially the identity: `X_i` -> `X_i`, `Z_i` -> `Z_i`.
    #[must_use]
    pub fn identity(num_qubits: usize) -> Self {
        let x_images: Vec<PauliString> = (0..num_qubits).map(PauliString::x).collect();
        let z_images: Vec<PauliString> = (0..num_qubits).map(PauliString::z).collect();
        Self {
            num_qubits,
            x_images,
            z_images,
        }
    }

    /// Returns the number of qubits.
    #[must_use]
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }

    /// Returns how X on the given qubit transforms.
    #[must_use]
    pub fn x_image(&self, qubit: usize) -> &PauliString {
        &self.x_images[qubit]
    }

    /// Returns how Z on the given qubit transforms.
    #[must_use]
    pub fn z_image(&self, qubit: usize) -> &PauliString {
        &self.z_images[qubit]
    }

    /// Sets how X on the given qubit transforms.
    pub fn set_x_image(&mut self, qubit: usize, image: PauliString) {
        self.x_images[qubit] = image;
    }

    /// Sets how Z on the given qubit transforms.
    pub fn set_z_image(&mut self, qubit: usize, image: PauliString) {
        self.z_images[qubit] = image;
    }

    /// Composes two Clifford representations: self * other.
    ///
    /// This means: apply other first, then self.
    /// In terms of generator images: (A * B)(P) = A(B(P))
    ///
    /// If the two representations have different numbers of qubits,
    /// the smaller one is implicitly extended with identity on the extra qubits.
    #[must_use]
    pub fn compose(&self, other: &CliffordRep) -> CliffordRep {
        let n = self.num_qubits.max(other.num_qubits);

        // Extend both to n qubits if needed
        let a = self.extended_to(n);
        let b = other.extended_to(n);

        let mut result = CliffordRep::identity(n);
        for q in 0..n {
            result.x_images[q] = a.apply(&b.x_images[q]);
            result.z_images[q] = a.apply(&b.z_images[q]);
        }

        result
    }

    /// Returns a copy of this Clifford extended to `n` qubits.
    /// Extra qubits are identity (`X_q` -> `X_q`, `Z_q` -> `Z_q`).
    /// If `n <= self.num_qubits`, returns a clone.
    #[must_use]
    pub fn extended_to(&self, n: usize) -> CliffordRep {
        if n <= self.num_qubits {
            return self.clone();
        }
        let mut result = CliffordRep::identity(n);
        for q in 0..self.num_qubits {
            result.x_images[q] = self.x_images[q].clone();
            result.z_images[q] = self.z_images[q].clone();
        }
        result
    }

    /// Applies this Clifford transformation to a `PauliString`.
    ///
    /// For each single-qubit Pauli in the input:
    /// - `X_q` -> `x_images[q]`
    /// - `Z_q` -> `z_images[q]`
    /// - `Y_q` = iXZ -> i * `x_images[q]` * `z_images[q]`
    ///
    /// The result is the product of all transformed single-qubit terms.
    #[must_use]
    pub fn apply(&self, pauli: &PauliString) -> PauliString {
        // Start with the input phase
        let mut result = PauliString::with_phase_and_paulis(pauli.phase(), vec![]);

        for (p, qubit_id) in pauli.iter_pairs() {
            let qubit = usize::from(qubit_id);

            if qubit >= self.num_qubits {
                // Qubit outside our range - pass through unchanged
                let paulis: Vec<_> = result.iter_pairs().collect();
                let mut new_paulis = paulis;
                new_paulis.push((p, qubit_id));
                result = PauliString::with_phase_and_paulis(result.phase(), new_paulis);
                continue;
            }

            match p {
                Pauli::I => {}
                Pauli::X => {
                    result = result * &self.x_images[qubit];
                }
                Pauli::Z => {
                    result = result * &self.z_images[qubit];
                }
                Pauli::Y => {
                    // Y = iXZ, so Y_q -> i * X_q_image * Z_q_image
                    result = i * result;
                    result = result * &self.x_images[qubit];
                    result = result * &self.z_images[qubit];
                }
            }
        }

        result
    }

    /// Applies this Clifford transformation to an `UnitaryRep`.
    ///
    /// This works seamlessly with Pauli operators created via `X(n)`, `Y(n)`, `Z(n)`.
    /// Returns `None` for non-Pauli operators.
    ///
    /// # Example
    ///
    /// ```
    /// use pecos_core::clifford_rep::CliffordRep;
    /// use pecos_core::unitary_rep::{X, Z};
    ///
    /// let h = CliffordRep::h(0);
    /// let stabilizer = X(0) & Z(1);
    /// let transformed = h.apply_to(&stabilizer).unwrap();
    /// ```
    #[must_use]
    pub fn apply_to(&self, op: &UnitaryRep) -> Option<UnitaryRep> {
        match op {
            UnitaryRep::Pauli(ps) => Some(UnitaryRep::Pauli(self.apply(ps))),
            _ => None,
        }
    }

    /// Returns the inverse of this Clifford.
    ///
    /// Uses the symplectic identity: for a symplectic matrix `M` over GF(2),
    /// `M^{-1} = Omega^T M^T Omega` where `Omega = [[0, I], [I, 0]]`.
    /// Phases are recovered by applying the forward map to each inverse generator
    /// image and matching against the expected output.
    #[must_use]
    pub fn inverse(&self) -> CliffordRep {
        let n = self.num_qubits;

        // Build the 2n x 2n symplectic matrix from generator images.
        // Row layout: rows 0..n are X-generator images, rows n..2n are Z-generator images.
        // Column layout: columns 0..n are X-bits, columns n..2n are Z-bits.
        let mut mat = vec![vec![0u8; 2 * n]; 2 * n];
        for q in 0..n {
            Self::pauli_string_to_symplectic_row(&self.x_images[q], n, &mut mat[q]);
            Self::pauli_string_to_symplectic_row(&self.z_images[q], n, &mut mat[n + q]);
        }

        // Compute M^{-1} = Omega^T M^T Omega over GF(2).
        // Since Omega = [[0, I], [I, 0]], applying Omega swaps the X and Z halves
        // of both rows and columns. The result: inv[r][c] = mat[swap(c)][swap(r)]
        // where swap(k) = k+n if k<n, else k-n.
        let swap = |k: usize| if k < n { k + n } else { k - n };
        let mut inv = vec![vec![0u8; 2 * n]; 2 * n];
        for (r, inv_row) in inv.iter_mut().enumerate() {
            let sr = swap(r);
            for (c, inv_cell) in inv_row.iter_mut().enumerate() {
                *inv_cell = mat[swap(c)][sr];
            }
        }

        // Reconstruct PauliStrings from inverse matrix rows and recover phases.
        let mut result = CliffordRep::identity(n);
        for q in 0..n {
            // X_q inverse image (from row q of inv)
            let x_inv_body = Self::symplectic_row_to_pauli_string(&inv[q], n);
            // Apply forward map to the body to find what phase correction is needed
            let forward = self.apply(&x_inv_body);
            // forward should equal (some_phase) * X_q
            // So C^{-1}(X_q) = conjugate(some_phase) * x_inv_body
            let phase = forward.phase().conjugate();
            result.x_images[q] =
                PauliString::with_phase_and_paulis(phase, x_inv_body.iter_pairs().collect());

            // Z_q inverse image (from row n+q of inv)
            let z_inv_body = Self::symplectic_row_to_pauli_string(&inv[n + q], n);
            let forward = self.apply(&z_inv_body);
            let phase = forward.phase().conjugate();
            result.z_images[q] =
                PauliString::with_phase_and_paulis(phase, z_inv_body.iter_pairs().collect());
        }

        result
    }

    /// Checks whether this Clifford representation is valid (symplectic).
    ///
    /// A valid Clifford must preserve the Pauli commutation relations:
    /// - `[X_i, Z_i] != 0` (anticommute on same qubit)
    /// - `[X_i, X_j] = [Z_i, Z_j] = [X_i, Z_j] = 0` for `i != j`
    /// - `[X_i, X_i] = [Z_i, Z_i] = 0` (self-commute)
    ///
    /// Additionally, all images must have `Sign` phases (`{+1, -1}`), not `{+i, -i}`.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        let n = self.num_qubits;
        use crate::PauliOperator;

        for qi in 0..n {
            // Check phases are real (Sign subset of QuarterPhase)
            if !self.x_images[qi].phase().is_real() || !self.z_images[qi].phase().is_real() {
                return false;
            }

            // X_qi and Z_qi must anticommute
            if self.x_images[qi].commutes_with(&self.z_images[qi]) {
                return false;
            }

            for qj in (qi + 1)..n {
                // X_qi, X_qj must commute
                if !self.x_images[qi].commutes_with(&self.x_images[qj]) {
                    return false;
                }
                // Z_qi, Z_qj must commute
                if !self.z_images[qi].commutes_with(&self.z_images[qj]) {
                    return false;
                }
                // X_qi, Z_qj must commute
                if !self.x_images[qi].commutes_with(&self.z_images[qj]) {
                    return false;
                }
                // Z_qi, X_qj must commute
                if !self.z_images[qi].commutes_with(&self.x_images[qj]) {
                    return false;
                }
            }
        }
        true
    }

    /// Converts a `PauliString` to a symplectic row vector `(x_0..x_{n-1} | z_0..z_{n-1})`.
    fn pauli_string_to_symplectic_row(ps: &PauliString, n: usize, row: &mut [u8]) {
        for bit in row.iter_mut() {
            *bit = 0;
        }
        for (p, qubit_id) in ps.iter_pairs() {
            let q = usize::from(qubit_id);
            if q < n {
                match p {
                    Pauli::X => row[q] = 1,
                    Pauli::Z => row[n + q] = 1,
                    Pauli::Y => {
                        row[q] = 1;
                        row[n + q] = 1;
                    }
                    Pauli::I => {}
                }
            }
        }
    }

    /// Converts a symplectic row vector to a `PauliString` with phase `+1`.
    fn symplectic_row_to_pauli_string(row: &[u8], n: usize) -> PauliString {
        let mut paulis = Vec::new();
        for q in 0..n {
            let x = row[q];
            let z = row[n + q];
            let pauli = match (x, z) {
                (1, 0) => Pauli::X,
                (0, 1) => Pauli::Z,
                (1, 1) => Pauli::Y,
                _ => continue,
            };
            paulis.push((pauli, crate::QubitId::new(q)));
        }
        PauliString::with_phase_and_paulis(QuarterPhase::PlusOne, paulis)
    }

    // ========================================================================
    // Standard single-qubit Clifford gates
    // ========================================================================

    /// Hadamard gate on qubit q: X <-> Z
    #[must_use]
    pub fn h(qubit: usize) -> Self {
        let num_qubits = qubit + 1;
        let mut cliff = Self::identity(num_qubits);
        // H: X -> Z, Z -> X
        cliff.x_images[qubit] = PauliString::z(qubit);
        cliff.z_images[qubit] = PauliString::x(qubit);
        cliff
    }

    /// SZ gate (sqrt Z, also known as S) on qubit q: X -> Y, Z -> Z
    #[must_use]
    pub fn sz(qubit: usize) -> Self {
        let num_qubits = qubit + 1;
        let mut cliff = Self::identity(num_qubits);
        cliff.x_images[qubit] = PauliString::y(qubit);
        cliff
    }

    /// SZ† gate (S†) on qubit q: X -> -Y, Z -> Z
    #[must_use]
    pub fn szdg(qubit: usize) -> Self {
        let num_qubits = qubit + 1;
        let mut cliff = Self::identity(num_qubits);
        cliff.x_images[qubit] = -PauliString::y(qubit);
        cliff
    }

    /// X gate on qubit q: X -> X, Z -> -Z
    #[must_use]
    pub fn x(qubit: usize) -> Self {
        Self::x_on(qubit, qubit + 1)
    }

    /// X gate on qubit q with specified total qubits: X -> X, Z -> -Z
    #[must_use]
    pub fn x_on(qubit: usize, num_qubits: usize) -> Self {
        let mut cliff = Self::identity(num_qubits);
        // X: X -> X, Z -> -Z
        cliff.z_images[qubit] = -PauliString::z(qubit);
        cliff
    }

    /// Y gate on qubit q: X -> -X, Z -> -Z
    #[must_use]
    pub fn y(qubit: usize) -> Self {
        Self::y_on(qubit, qubit + 1)
    }

    /// Y gate on qubit q with specified total qubits: X -> -X, Z -> -Z
    #[must_use]
    pub fn y_on(qubit: usize, num_qubits: usize) -> Self {
        let mut cliff = Self::identity(num_qubits);
        // Y: X -> -X, Z -> -Z
        cliff.x_images[qubit] = -PauliString::x(qubit);
        cliff.z_images[qubit] = -PauliString::z(qubit);
        cliff
    }

    /// Z gate on qubit q: X -> -X, Z -> Z
    #[must_use]
    pub fn z(qubit: usize) -> Self {
        Self::z_on(qubit, qubit + 1)
    }

    /// Z gate on qubit q with specified total qubits: X -> -X, Z -> Z
    #[must_use]
    pub fn z_on(qubit: usize, num_qubits: usize) -> Self {
        let mut cliff = Self::identity(num_qubits);
        // Z: X -> -X, Z -> Z
        cliff.x_images[qubit] = -PauliString::x(qubit);
        cliff
    }

    /// SX gate (sqrt X) on qubit q: X -> X, Z -> -Y
    #[must_use]
    pub fn sx(qubit: usize) -> Self {
        let num_qubits = qubit + 1;
        let mut cliff = Self::identity(num_qubits);
        // SX: X -> X, Z -> -Y
        cliff.z_images[qubit] = -PauliString::y(qubit);
        cliff
    }

    /// SY gate (sqrt Y) on qubit q: X -> -Z, Z -> X
    #[must_use]
    pub fn sy(qubit: usize) -> Self {
        let num_qubits = qubit + 1;
        let mut cliff = Self::identity(num_qubits);
        // SY: X -> -Z, Z -> X
        cliff.x_images[qubit] = -PauliString::z(qubit);
        cliff.z_images[qubit] = PauliString::x(qubit);
        cliff
    }

    /// SX† gate on qubit q.
    /// Decomposition: X · SX (`CliffordGateable`: self.x(&[q]).sx(&[q]))
    #[must_use]
    pub fn sxdg(qubit: usize) -> Self {
        Self::sx(qubit).compose(&Self::x(qubit))
    }

    /// SY† gate on qubit q.
    /// Decomposition: Y · SY (`CliffordGateable`: self.y(&[q]).sy(&[q]))
    #[must_use]
    pub fn sydg(qubit: usize) -> Self {
        Self::sy(qubit).compose(&Self::y(qubit))
    }

    /// H2 gate on qubit q.
    /// Decomposition: SY · Z (`CliffordGateable`: self.sy(&[q]).z(&[q]))
    #[must_use]
    pub fn h2(qubit: usize) -> Self {
        Self::z(qubit).compose(&Self::sy(qubit))
    }

    /// H3 gate on qubit q.
    /// Decomposition: SZ · Y (`CliffordGateable`: self.sz(&[q]).y(&[q]))
    #[must_use]
    pub fn h3(qubit: usize) -> Self {
        Self::y(qubit).compose(&Self::sz(qubit))
    }

    /// H4 gate on qubit q.
    /// Decomposition: SZ · X (`CliffordGateable`: self.sz(&[q]).x(&[q]))
    #[must_use]
    pub fn h4(qubit: usize) -> Self {
        Self::x(qubit).compose(&Self::sz(qubit))
    }

    /// H5 gate on qubit q.
    /// Decomposition: SX · Z (`CliffordGateable`: self.sx(&[q]).z(&[q]))
    #[must_use]
    pub fn h5(qubit: usize) -> Self {
        Self::z(qubit).compose(&Self::sx(qubit))
    }

    /// H6 gate on qubit q.
    /// Decomposition: SX · Y (`CliffordGateable`: self.sx(&[q]).y(&[q]))
    #[must_use]
    pub fn h6(qubit: usize) -> Self {
        Self::y(qubit).compose(&Self::sx(qubit))
    }

    /// F (Face) gate on qubit q.
    /// Decomposition: SX · SZ (`CliffordGateable`: self.sx(&[q]).sz(&[q]))
    #[must_use]
    pub fn f(qubit: usize) -> Self {
        Self::sz(qubit).compose(&Self::sx(qubit))
    }

    /// F† gate on qubit q.
    /// Decomposition: SZ† · SX† (`CliffordGateable`: self.szdg(&[q]).sxdg(&[q]))
    #[must_use]
    pub fn fdg(qubit: usize) -> Self {
        Self::sxdg(qubit).compose(&Self::szdg(qubit))
    }

    /// F2 gate on qubit q.
    /// Decomposition: SX† · SY (`CliffordGateable`: self.sxdg(&[q]).sy(&[q]))
    #[must_use]
    pub fn f2(qubit: usize) -> Self {
        Self::sy(qubit).compose(&Self::sxdg(qubit))
    }

    /// F2† gate on qubit q.
    /// Decomposition: SY† · SX (`CliffordGateable`: self.sydg(&[q]).sx(&[q]))
    #[must_use]
    pub fn f2dg(qubit: usize) -> Self {
        Self::sx(qubit).compose(&Self::sydg(qubit))
    }

    /// F3 gate on qubit q.
    /// Decomposition: SX† · SZ (`CliffordGateable`: self.sxdg(&[q]).sz(&[q]))
    #[must_use]
    pub fn f3(qubit: usize) -> Self {
        Self::sz(qubit).compose(&Self::sxdg(qubit))
    }

    /// F3† gate on qubit q.
    /// Decomposition: SZ† · SX (`CliffordGateable`: self.szdg(&[q]).sx(&[q]))
    #[must_use]
    pub fn f3dg(qubit: usize) -> Self {
        Self::sx(qubit).compose(&Self::szdg(qubit))
    }

    /// F4 gate on qubit q.
    /// Decomposition: SZ · SX (`CliffordGateable`: self.sz(&[q]).sx(&[q]))
    #[must_use]
    pub fn f4(qubit: usize) -> Self {
        Self::sx(qubit).compose(&Self::sz(qubit))
    }

    /// F4† gate on qubit q.
    /// Decomposition: SX† · SZ† (`CliffordGateable`: self.sxdg(&[q]).szdg(&[q]))
    #[must_use]
    pub fn f4dg(qubit: usize) -> Self {
        Self::szdg(qubit).compose(&Self::sxdg(qubit))
    }

    // ========================================================================
    // Two-qubit Clifford gates
    // ========================================================================

    /// CNOT (CX) gate: control -> target
    ///
    /// CX: `X_c` -> `X_c` `X_t`, `Z_c` -> `Z_c`
    ///     `X_t` -> `X_t`,     `Z_t` -> `Z_c` `Z_t`
    #[must_use]
    pub fn cx(control: usize, target: usize) -> Self {
        let num_qubits = control.max(target) + 1;
        let mut cliff = Self::identity(num_qubits);

        // X_control -> X_control * X_target (tensor product)
        cliff.x_images[control] = PauliString::x(control) & PauliString::x(target);

        // Z_target -> Z_control * Z_target (tensor product)
        cliff.z_images[target] = PauliString::z(control) & PauliString::z(target);

        cliff
    }

    /// CZ gate: controlled-Z
    ///
    /// CZ: `X_0` -> `X_0` `Z_1`, `Z_0` -> `Z_0`
    ///     `X_1` -> `Z_0` `X_1`, `Z_1` -> `Z_1`
    #[must_use]
    pub fn cz(q0: usize, q1: usize) -> Self {
        let num_qubits = q0.max(q1) + 1;
        let mut cliff = Self::identity(num_qubits);

        // X_0 -> X_0 * Z_1 (tensor product)
        cliff.x_images[q0] = PauliString::x(q0) & PauliString::z(q1);

        // X_1 -> Z_0 * X_1 (tensor product)
        cliff.x_images[q1] = PauliString::z(q0) & PauliString::x(q1);

        cliff
    }

    /// SWAP gate
    ///
    /// SWAP: `X_0` -> `X_1`, `Z_0` -> `Z_1`
    ///       `X_1` -> `X_0`, `Z_1` -> `Z_0`
    #[must_use]
    pub fn swap(q0: usize, q1: usize) -> Self {
        let num_qubits = q0.max(q1) + 1;
        let mut cliff = Self::identity(num_qubits);

        cliff.x_images[q0] = PauliString::x(q1);
        cliff.x_images[q1] = PauliString::x(q0);
        cliff.z_images[q0] = PauliString::z(q1);
        cliff.z_images[q1] = PauliString::z(q0);

        cliff
    }

    /// CY gate: controlled-Y
    ///
    /// CY = (I ⊗ S) CX (I ⊗ S†)
    ///
    /// CY: `X_c` -> `X_c` `Y_t`, `Z_c` -> `Z_c`
    ///     `X_t` -> `Z_c` `X_t`, `Z_t` -> `Z_c` `Z_t`
    #[must_use]
    pub fn cy(control: usize, target: usize) -> Self {
        let num_qubits = control.max(target) + 1;
        let mut cliff = Self::identity(num_qubits);

        // X_control -> X_control * Y_target
        cliff.x_images[control] = PauliString::x(control) & PauliString::y(target);

        // X_target -> Z_control * X_target
        cliff.x_images[target] = PauliString::z(control) & PauliString::x(target);

        // Z_target -> Z_control * Z_target
        cliff.z_images[target] = PauliString::z(control) & PauliString::z(target);

        cliff
    }

    /// SXX gate (sqrt XX): XI -> XI, IX -> IX, ZI -> -YX, IZ -> -XY
    #[must_use]
    pub fn sxx(q0: usize, q1: usize) -> Self {
        let num_qubits = q0.max(q1) + 1;
        let mut cliff = Self::identity(num_qubits);
        cliff.z_images[q0] = -(PauliString::y(q0) & PauliString::x(q1));
        cliff.z_images[q1] = -(PauliString::x(q0) & PauliString::y(q1));
        cliff
    }

    /// SXX† gate: XI -> XI, IX -> IX, ZI -> YX, IZ -> XY
    #[must_use]
    pub fn sxxdg(q0: usize, q1: usize) -> Self {
        let num_qubits = q0.max(q1) + 1;
        let mut cliff = Self::identity(num_qubits);
        cliff.z_images[q0] = PauliString::y(q0) & PauliString::x(q1);
        cliff.z_images[q1] = PauliString::x(q0) & PauliString::y(q1);
        cliff
    }

    /// SYY gate (sqrt YY): XI -> -ZY, IX -> -YZ, ZI -> XY, IZ -> YX
    #[must_use]
    pub fn syy(q0: usize, q1: usize) -> Self {
        let num_qubits = q0.max(q1) + 1;
        let mut cliff = Self::identity(num_qubits);
        cliff.x_images[q0] = -(PauliString::z(q0) & PauliString::y(q1));
        cliff.x_images[q1] = -(PauliString::y(q0) & PauliString::z(q1));
        cliff.z_images[q0] = PauliString::x(q0) & PauliString::y(q1);
        cliff.z_images[q1] = PauliString::y(q0) & PauliString::x(q1);
        cliff
    }

    /// SYY† gate: XI -> ZY, IX -> YZ, ZI -> -XY, IZ -> -YX
    #[must_use]
    pub fn syydg(q0: usize, q1: usize) -> Self {
        let num_qubits = q0.max(q1) + 1;
        let mut cliff = Self::identity(num_qubits);
        cliff.x_images[q0] = PauliString::z(q0) & PauliString::y(q1);
        cliff.x_images[q1] = PauliString::y(q0) & PauliString::z(q1);
        cliff.z_images[q0] = -(PauliString::x(q0) & PauliString::y(q1));
        cliff.z_images[q1] = -(PauliString::y(q0) & PauliString::x(q1));
        cliff
    }

    /// SZZ gate (sqrt ZZ): XI -> YZ, IX -> ZY, ZI -> ZI, IZ -> IZ
    #[must_use]
    pub fn szz(q0: usize, q1: usize) -> Self {
        let num_qubits = q0.max(q1) + 1;
        let mut cliff = Self::identity(num_qubits);
        cliff.x_images[q0] = PauliString::y(q0) & PauliString::z(q1);
        cliff.x_images[q1] = PauliString::z(q0) & PauliString::y(q1);
        cliff
    }

    /// SZZ† gate: XI -> -YZ, IX -> -ZY, ZI -> ZI, IZ -> IZ
    #[must_use]
    pub fn szzdg(q0: usize, q1: usize) -> Self {
        let num_qubits = q0.max(q1) + 1;
        let mut cliff = Self::identity(num_qubits);
        cliff.x_images[q0] = -(PauliString::y(q0) & PauliString::z(q1));
        cliff.x_images[q1] = -(PauliString::z(q0) & PauliString::y(q1));
        cliff
    }

    /// iSWAP gate: XI -> ZY, IX -> YZ, ZI -> IZ, IZ -> ZI
    #[must_use]
    pub fn iswap(q0: usize, q1: usize) -> Self {
        let num_qubits = q0.max(q1) + 1;
        let mut cliff = Self::identity(num_qubits);
        cliff.x_images[q0] = PauliString::z(q0) & PauliString::y(q1);
        cliff.x_images[q1] = PauliString::y(q0) & PauliString::z(q1);
        cliff.z_images[q0] = PauliString::z(q1);
        cliff.z_images[q1] = PauliString::z(q0);
        cliff
    }

    /// G gate: XI -> IX, IX -> XI, ZI -> XZ, IZ -> ZX
    #[must_use]
    pub fn g(q0: usize, q1: usize) -> Self {
        let num_qubits = q0.max(q1) + 1;
        let mut cliff = Self::identity(num_qubits);
        cliff.x_images[q0] = PauliString::x(q1);
        cliff.x_images[q1] = PauliString::x(q0);
        cliff.z_images[q0] = PauliString::x(q0) & PauliString::z(q1);
        cliff.z_images[q1] = PauliString::z(q0) & PauliString::x(q1);
        cliff
    }

    /// iSWAP† gate: XI -> -ZY, IX -> -YZ, ZI -> IZ, IZ -> ZI
    ///
    /// Both X images have opposite signs from iSWAP (the Z images are the same).
    #[must_use]
    pub fn iswapdg(q0: usize, q1: usize) -> Self {
        let num_qubits = q0.max(q1) + 1;
        let mut cliff = Self::identity(num_qubits);
        cliff.x_images[q0] = -(PauliString::z(q0) & PauliString::y(q1));
        cliff.x_images[q1] = -(PauliString::y(q0) & PauliString::z(q1));
        cliff.z_images[q0] = PauliString::z(q1);
        cliff.z_images[q1] = PauliString::z(q0);
        cliff
    }

    /// G† gate (inverse of G)
    ///
    /// G is self-inverse at the stabilizer level (G^2 = I for all generators),
    /// so Gdg has the same `CliffordRep` as G. The two gates differ only by a
    /// global phase at the unitary level.
    #[must_use]
    pub fn gdg(q0: usize, q1: usize) -> Self {
        Self::g(q0, q1)
    }
}

impl CliffordRep {
    // ========================================================================
    // Single-qubit Clifford enumeration
    // ========================================================================

    /// All 24 single-qubit Cliffords as generator sequences (H=0, S=1).
    /// Applied left-to-right: [a, b, c] means C = a * b * c.
    const SINGLE_QUBIT_SEQUENCES: &'static [&'static [u8]] = &[
        &[],                    //  0: I
        &[0, 1, 1, 0],          //  1: X = HSSH
        &[1, 1, 0, 1, 1, 0],    //  2: Y = SSHSSH
        &[1, 1],                //  3: Z = SS
        &[1],                   //  4: S
        &[1, 1, 1],             //  5: Sdg
        &[0],                   //  6: H
        &[1, 0],                //  7: SH
        &[0, 1],                //  8: HS
        &[1, 1, 0],             //  9: S²H
        &[0, 1, 1],             // 10: HS²
        &[1, 1, 1, 0],          // 11: S³H
        &[1, 0, 1],             // 12: SHS
        &[0, 1, 0],             // 13: HSH
        &[1, 0, 1, 0],          // 14: SHSH
        &[1, 1, 0, 1],          // 15: S²HS
        &[1, 0, 1, 1],          // 16: SHS²
        &[1, 1, 1, 0, 1],       // 17: S³HS
        &[1, 1, 0, 1, 1],       // 18: S²HS²
        &[1, 1, 0, 1, 0],       // 19: S²HSH
        &[0, 1, 1, 0, 1],       // 20: HS²HS
        &[1, 1, 1, 0, 1, 1],    // 21: S³HS²
        &[1, 1, 1, 0, 1, 0],    // 22: S³HSH
        &[0, 1, 1, 0, 1, 1, 1], // 23: HS²HS³
    ];

    /// Returns all 24 single-qubit Clifford gates on the given qubit.
    ///
    /// The 24 elements form the single-qubit Clifford group (modulo global phase).
    /// Index 0 is the identity.
    #[must_use]
    pub fn single_qubit_cliffords(qubit: usize) -> Vec<CliffordRep> {
        Self::SINGLE_QUBIT_SEQUENCES
            .iter()
            .map(|seq| {
                let mut cliff = CliffordRep::identity(qubit + 1);
                for &gate in *seq {
                    let g = if gate == 0 {
                        CliffordRep::h(qubit)
                    } else {
                        CliffordRep::sz(qubit)
                    };
                    cliff = cliff.compose(&g);
                }
                cliff
            })
            .collect()
    }

    /// Returns a random single-qubit Clifford on the given qubit.
    ///
    /// Uses the provided RNG to select uniformly from the 24 elements.
    #[must_use]
    pub fn random_single_qubit(qubit: usize, rng: &mut impl rand::Rng) -> CliffordRep {
        let idx = rng.random_range(0..24usize);
        let seq = Self::SINGLE_QUBIT_SEQUENCES[idx];
        let mut cliff = CliffordRep::identity(qubit + 1);
        for &gate in seq {
            let g = if gate == 0 {
                CliffordRep::h(qubit)
            } else {
                CliffordRep::sz(qubit)
            };
            cliff = cliff.compose(&g);
        }
        cliff
    }

    /// Returns a random n-qubit Clifford by composing random gate layers.
    ///
    /// Uses `depth` layers, each consisting of random single-qubit Cliffords
    /// on every qubit followed by CZ gates on random pairs. With sufficient
    /// depth (typically `depth >= 2*n`), this generates a distribution that
    /// covers the full Clifford group (though not perfectly uniform).
    ///
    /// For exact uniform sampling, use specialized algorithms (e.g., Koenig & Smolin).
    #[must_use]
    pub fn random(num_qubits: usize, depth: usize, rng: &mut impl rand::Rng) -> CliffordRep {
        let mut cliff = CliffordRep::identity(num_qubits);

        for _ in 0..depth {
            // Layer of random single-qubit Cliffords
            for q in 0..num_qubits {
                let single = CliffordRep::random_single_qubit(q, rng);
                cliff = cliff.compose(&single.extended_to(num_qubits));
            }

            // Layer of random CZ gates (each pair independently with 50% probability)
            for q0 in 0..num_qubits {
                for q1 in (q0 + 1)..num_qubits {
                    if rng.random_bool(0.5) {
                        cliff = cliff.compose(&CliffordRep::cz(q0, q1).extended_to(num_qubits));
                    }
                }
            }
        }

        cliff
    }
}

impl fmt::Display for CliffordRep {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        writeln!(f, "CliffordRep({} qubits):", self.num_qubits)?;
        for q in 0..self.num_qubits {
            writeln!(f, "  X_{q} -> {}", self.x_images[q].to_sparse_str())?;
            writeln!(f, "  Z_{q} -> {}", self.z_images[q].to_sparse_str())?;
        }
        Ok(())
    }
}

// ============================================================================
// Mul trait: * operator for composition
// ============================================================================

impl Mul for CliffordRep {
    type Output = CliffordRep;

    /// Composes two Cliffords: `self * rhs` means "apply self, then rhs".
    fn mul(self, rhs: CliffordRep) -> CliffordRep {
        self.compose(&rhs)
    }
}

impl Mul<&CliffordRep> for CliffordRep {
    type Output = CliffordRep;

    fn mul(self, rhs: &CliffordRep) -> CliffordRep {
        self.compose(rhs)
    }
}

impl Mul<CliffordRep> for &CliffordRep {
    type Output = CliffordRep;

    fn mul(self, rhs: CliffordRep) -> CliffordRep {
        self.compose(&rhs)
    }
}

impl Mul<&CliffordRep> for &CliffordRep {
    type Output = CliffordRep;

    fn mul(self, rhs: &CliffordRep) -> CliffordRep {
        self.compose(rhs)
    }
}

// ============================================================================
// BitAnd trait: & operator for tensor product
// ============================================================================

impl BitAnd for CliffordRep {
    type Output = CliffordRep;

    /// Tensor product of two `CliffordReps` acting on disjoint qubits.
    fn bitand(self, rhs: CliffordRep) -> CliffordRep {
        // Since compose auto-extends with identity on extra qubits,
        // and these CliffordReps act on disjoint qubits, composing gives the tensor.
        self.compose(&rhs)
    }
}

impl BitAnd<&CliffordRep> for CliffordRep {
    type Output = CliffordRep;

    fn bitand(self, rhs: &CliffordRep) -> CliffordRep {
        self.compose(rhs)
    }
}

impl BitAnd<CliffordRep> for &CliffordRep {
    type Output = CliffordRep;

    fn bitand(self, rhs: CliffordRep) -> CliffordRep {
        self.compose(&rhs)
    }
}

impl BitAnd<&CliffordRep> for &CliffordRep {
    type Output = CliffordRep;

    fn bitand(self, rhs: &CliffordRep) -> CliffordRep {
        self.compose(rhs)
    }
}

// ============================================================================
// From<PauliString>: Paulis are Cliffords
// ============================================================================

impl From<PauliString> for CliffordRep {
    /// Converts a `PauliString` into a `CliffordRep`.
    ///
    /// A Pauli P acts on generators by conjugation:
    /// - `P X_q P†` depends on whether P commutes or anticommutes with `X_q`
    /// - If P commutes with `X_q`: `X_q` -> `X_q`
    /// - If P anticommutes with `X_q`: `X_q` -> `-X_q`
    ///   (Same logic for `Z_q`.)
    fn from(pauli: PauliString) -> CliffordRep {
        let n = pauli.qubits().into_iter().max().map_or(1, |m| m + 1);
        let mut cliff = CliffordRep::identity(n);

        for q in 0..n {
            let p_at_q = pauli.get(q);
            // P conjugation: P * X_q * P† = ±X_q
            // Anticommutes iff both are non-identity and different (and neither is I)
            // X anticommutes with Z and Y; Z anticommutes with X and Y; Y anticommutes with X and Z
            let x_sign = match p_at_q {
                Pauli::I | Pauli::X => QuarterPhase::PlusOne,
                Pauli::Y | Pauli::Z => QuarterPhase::MinusOne,
            };
            let z_sign = match p_at_q {
                Pauli::I | Pauli::Z => QuarterPhase::PlusOne,
                Pauli::X | Pauli::Y => QuarterPhase::MinusOne,
            };

            if x_sign == QuarterPhase::MinusOne {
                let mut img = cliff.x_image(q).clone();
                img.set_phase(img.phase().multiply(&QuarterPhase::MinusOne));
                cliff.set_x_image(q, img);
            }
            if z_sign == QuarterPhase::MinusOne {
                let mut img = cliff.z_image(q).clone();
                img.set_phase(img.phase().multiply(&QuarterPhase::MinusOne));
                cliff.set_z_image(q, img);
            }
        }

        // Also need to account for the global phase of the PauliString
        // But Clifford conjugation P * G * P† cancels the phase of P, so
        // the global phase of the PauliString doesn't affect the Clifford action.
        cliff
    }
}

impl From<&PauliString> for CliffordRep {
    fn from(pauli: &PauliString) -> CliffordRep {
        CliffordRep::from(pauli.clone())
    }
}

// ============================================================================
// Constructor functions for ergonomic Clifford algebra
// ============================================================================

/// Free-standing constructor functions for Clifford gates.
///
/// These mirror the `pecos_core::pauli::constructors` module, providing
/// ergonomic gate creation and composition via the `*` operator.
///
/// # Examples
///
/// ```
/// use pecos_core::clifford_rep::constructors::*;
///
/// // H * SZ * H = SX (sqrt-X)
/// let sx = H(0) * SZ(0) * H(0);
/// assert!(sx.is_valid());
///
/// // Bell state prep circuit
/// let bell = H(0) * CX(0, 1);
/// assert_eq!(bell.num_qubits(), 2);
/// ```
#[allow(non_snake_case)]
pub mod constructors {
    use super::CliffordRep;

    /// Hadamard gate on qubit `q`.
    #[must_use]
    pub fn H(q: usize) -> CliffordRep {
        CliffordRep::h(q)
    }

    // Single-qubit sqrt gates and their daggers

    #[must_use]
    pub fn SX(q: usize) -> CliffordRep {
        CliffordRep::sx(q)
    }
    #[must_use]
    pub fn SXdg(q: usize) -> CliffordRep {
        CliffordRep::sxdg(q)
    }
    #[must_use]
    pub fn SY(q: usize) -> CliffordRep {
        CliffordRep::sy(q)
    }
    #[must_use]
    pub fn SYdg(q: usize) -> CliffordRep {
        CliffordRep::sydg(q)
    }
    #[must_use]
    pub fn SZ(q: usize) -> CliffordRep {
        CliffordRep::sz(q)
    }
    #[must_use]
    pub fn SZdg(q: usize) -> CliffordRep {
        CliffordRep::szdg(q)
    }

    // Hadamard variants

    #[must_use]
    pub fn H2(q: usize) -> CliffordRep {
        CliffordRep::h2(q)
    }
    #[must_use]
    pub fn H3(q: usize) -> CliffordRep {
        CliffordRep::h3(q)
    }
    #[must_use]
    pub fn H4(q: usize) -> CliffordRep {
        CliffordRep::h4(q)
    }
    #[must_use]
    pub fn H5(q: usize) -> CliffordRep {
        CliffordRep::h5(q)
    }
    #[must_use]
    pub fn H6(q: usize) -> CliffordRep {
        CliffordRep::h6(q)
    }

    // Face gates

    #[must_use]
    pub fn F(q: usize) -> CliffordRep {
        CliffordRep::f(q)
    }
    #[must_use]
    pub fn Fdg(q: usize) -> CliffordRep {
        CliffordRep::fdg(q)
    }
    #[must_use]
    pub fn F2(q: usize) -> CliffordRep {
        CliffordRep::f2(q)
    }
    #[must_use]
    pub fn F2dg(q: usize) -> CliffordRep {
        CliffordRep::f2dg(q)
    }
    #[must_use]
    pub fn F3(q: usize) -> CliffordRep {
        CliffordRep::f3(q)
    }
    #[must_use]
    pub fn F3dg(q: usize) -> CliffordRep {
        CliffordRep::f3dg(q)
    }
    #[must_use]
    pub fn F4(q: usize) -> CliffordRep {
        CliffordRep::f4(q)
    }
    #[must_use]
    pub fn F4dg(q: usize) -> CliffordRep {
        CliffordRep::f4dg(q)
    }

    // Two-qubit gates

    #[must_use]
    pub fn CX(c: usize, t: usize) -> CliffordRep {
        CliffordRep::cx(c, t)
    }
    #[must_use]
    pub fn CY(c: usize, t: usize) -> CliffordRep {
        CliffordRep::cy(c, t)
    }
    #[must_use]
    pub fn CZ(a: usize, b: usize) -> CliffordRep {
        CliffordRep::cz(a, b)
    }
    #[must_use]
    pub fn SWAP(a: usize, b: usize) -> CliffordRep {
        CliffordRep::swap(a, b)
    }
    #[must_use]
    pub fn SXX(a: usize, b: usize) -> CliffordRep {
        CliffordRep::sxx(a, b)
    }
    #[must_use]
    pub fn SXXdg(a: usize, b: usize) -> CliffordRep {
        CliffordRep::sxxdg(a, b)
    }
    #[must_use]
    pub fn SYY(a: usize, b: usize) -> CliffordRep {
        CliffordRep::syy(a, b)
    }
    #[must_use]
    pub fn SYYdg(a: usize, b: usize) -> CliffordRep {
        CliffordRep::syydg(a, b)
    }
    #[must_use]
    pub fn SZZ(a: usize, b: usize) -> CliffordRep {
        CliffordRep::szz(a, b)
    }
    #[must_use]
    pub fn SZZdg(a: usize, b: usize) -> CliffordRep {
        CliffordRep::szzdg(a, b)
    }
    #[must_use]
    pub fn ISWAP(a: usize, b: usize) -> CliffordRep {
        CliffordRep::iswap(a, b)
    }
    #[must_use]
    pub fn ISWAPdg(a: usize, b: usize) -> CliffordRep {
        CliffordRep::iswapdg(a, b)
    }
    #[must_use]
    pub fn G(a: usize, b: usize) -> CliffordRep {
        CliffordRep::g(a, b)
    }
    #[must_use]
    pub fn Gdg(a: usize, b: usize) -> CliffordRep {
        CliffordRep::gdg(a, b)
    }

    /// Identity Clifford on `n` qubits.
    #[must_use]
    pub fn Id(n: usize) -> CliffordRep {
        CliffordRep::identity(n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PauliOperator;

    // ========================================================================
    // Helper: verify C * C^{-1} = identity on all generators
    // ========================================================================

    fn assert_inverse_correct(cliff: &CliffordRep) {
        let inv = cliff.inverse();
        let n = cliff.num_qubits();

        for q in 0..n {
            // C(C^{-1}(X_q)) should equal X_q
            let x_q = PauliString::x(q);
            let round_trip = cliff.apply(&inv.apply(&x_q));
            assert_eq!(
                round_trip, x_q,
                "C(C^-1(X_{q})) != X_{q}, got {round_trip:?}"
            );

            // C(C^{-1}(Z_q)) should equal Z_q
            let z_q = PauliString::z(q);
            let round_trip = cliff.apply(&inv.apply(&z_q));
            assert_eq!(
                round_trip, z_q,
                "C(C^-1(Z_{q})) != Z_{q}, got {round_trip:?}"
            );

            // C^{-1}(C(X_q)) should equal X_q
            let round_trip = inv.apply(&cliff.apply(&x_q));
            assert_eq!(
                round_trip, x_q,
                "C^-1(C(X_{q})) != X_{q}, got {round_trip:?}"
            );

            // C^{-1}(C(Z_q)) should equal Z_q
            let round_trip = inv.apply(&cliff.apply(&z_q));
            assert_eq!(
                round_trip, z_q,
                "C^-1(C(Z_{q})) != Z_{q}, got {round_trip:?}"
            );
        }
    }

    // ========================================================================
    // Basic gate tests
    // ========================================================================

    #[test]
    fn test_identity() {
        let id = CliffordRep::identity(2);
        let x0 = PauliString::x(0);
        let z1 = PauliString::z(1);

        assert_eq!(id.apply(&x0), x0);
        assert_eq!(id.apply(&z1), z1);
    }

    #[test]
    fn test_hadamard_swaps_x_z() {
        let h = CliffordRep::h(0);

        let x0 = PauliString::x(0);
        let z0 = PauliString::z(0);

        // H: X -> Z
        let hx = h.apply(&x0);
        assert_eq!(hx.paulis().len(), 1);
        assert_eq!(hx.paulis()[0].0, Pauli::Z);

        // H: Z -> X
        let hz = h.apply(&z0);
        assert_eq!(hz.paulis().len(), 1);
        assert_eq!(hz.paulis()[0].0, Pauli::X);
    }

    #[test]
    fn test_hadamard_squared_is_identity() {
        let h = CliffordRep::h(0);
        let hh = h.compose(&h);

        let x0 = PauliString::x(0);
        let z0 = PauliString::z(0);

        assert_eq!(hh.apply(&x0), x0);
        assert_eq!(hh.apply(&z0), z0);
    }

    #[test]
    fn test_s_transforms_x_to_y() {
        let s = CliffordRep::sz(0);
        let x0 = PauliString::x(0);

        let sx = s.apply(&x0);
        assert_eq!(sx.paulis().len(), 1);
        assert_eq!(sx.paulis()[0].0, Pauli::Y);
    }

    #[test]
    fn test_composition_h_s() {
        let h = CliffordRep::h(0);
        let s = CliffordRep::sz(0);

        // HS: apply S first, then H
        let hs = h.compose(&s);
        let x0 = PauliString::x(0);
        let z0 = PauliString::z(0);

        // S(X) = Y, then H(Y): Y = iXZ so H(Y) = i * Z * X = i * iY = -Y
        let result = hs.apply(&x0);
        assert_eq!(result.get(0), Pauli::Y);
        assert_eq!(result.phase(), QuarterPhase::MinusOne);

        // S(Z) = Z, then H(Z) = X
        let result = hs.apply(&z0);
        assert_eq!(result.get(0), Pauli::X);
        assert_eq!(result.phase(), QuarterPhase::PlusOne);
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_cx_propagation() {
        let cx = CliffordRep::cx(0, 1);

        // X_0 -> X_0 X_1
        let cx_x0 = cx.apply(&PauliString::x(0));
        assert_eq!(cx_x0.weight(), 2);
        assert_eq!(cx_x0.get(0), Pauli::X);
        assert_eq!(cx_x0.get(1), Pauli::X);

        // Z_1 -> Z_0 Z_1
        let cx_z1 = cx.apply(&PauliString::z(1));
        assert_eq!(cx_z1.weight(), 2);
        assert_eq!(cx_z1.get(0), Pauli::Z);
        assert_eq!(cx_z1.get(1), Pauli::Z);

        // X_1 -> X_1
        let cx_x1 = cx.apply(&PauliString::x(1));
        assert_eq!(cx_x1, PauliString::x(1));

        // Z_0 -> Z_0
        let cx_z0 = cx.apply(&PauliString::z(0));
        assert_eq!(cx_z0, PauliString::z(0));
    }

    #[test]
    #[allow(clippy::similar_names)]
    fn test_cz_symmetric() {
        let cz = CliffordRep::cz(0, 1);

        // X on either qubit picks up Z on the other
        let cz_x0 = cz.apply(&PauliString::x(0));
        assert_eq!(cz_x0.weight(), 2);

        let cz_x1 = cz.apply(&PauliString::x(1));
        assert_eq!(cz_x1.weight(), 2);

        // Z stays unchanged
        assert_eq!(cz.apply(&PauliString::z(0)), PauliString::z(0));
        assert_eq!(cz.apply(&PauliString::z(1)), PauliString::z(1));
    }

    #[test]
    fn test_swap() {
        let swap = CliffordRep::swap(0, 1);

        assert_eq!(swap.apply(&PauliString::x(0)), PauliString::x(1));
        assert_eq!(swap.apply(&PauliString::x(1)), PauliString::x(0));
        assert_eq!(swap.apply(&PauliString::z(0)), PauliString::z(1));
        assert_eq!(swap.apply(&PauliString::z(1)), PauliString::z(0));
    }

    // ========================================================================
    // Inverse tests
    // ========================================================================

    #[test]
    fn test_inverse_identity() {
        let id = CliffordRep::identity(2);
        assert_inverse_correct(&id);
    }

    #[test]
    fn test_inverse_hadamard() {
        // H is self-inverse
        let h = CliffordRep::h(0);
        assert_inverse_correct(&h);
        let inv = h.inverse();
        assert_eq!(h.apply(&PauliString::x(0)), inv.apply(&PauliString::x(0)));
    }

    #[test]
    fn test_inverse_s() {
        // S^{-1} = S^dag: X -> -Y (instead of X -> Y)
        let s = CliffordRep::sz(0);
        assert_inverse_correct(&s);

        let inv = s.inverse();
        let result = inv.apply(&PauliString::x(0));
        assert_eq!(result.get(0), Pauli::Y);
        assert_eq!(result.phase(), QuarterPhase::MinusOne);
    }

    #[test]
    fn test_inverse_cx() {
        // CX is self-inverse
        let cx = CliffordRep::cx(0, 1);
        assert_inverse_correct(&cx);
    }

    #[test]
    fn test_inverse_cz() {
        // CZ is self-inverse
        let cz = CliffordRep::cz(0, 1);
        assert_inverse_correct(&cz);
    }

    #[test]
    fn test_inverse_swap() {
        // SWAP is self-inverse
        let swap = CliffordRep::swap(0, 1);
        assert_inverse_correct(&swap);
    }

    #[test]
    fn test_inverse_sdg() {
        let sdg = CliffordRep::szdg(0);
        assert_inverse_correct(&sdg);

        // Sdg^{-1} = S
        let inv = sdg.inverse();
        let result = inv.apply(&PauliString::x(0));
        assert_eq!(result.get(0), Pauli::Y);
        assert_eq!(result.phase(), QuarterPhase::PlusOne);
    }

    #[test]
    fn test_inverse_composed_gate() {
        // Compose H*S*CX and verify inverse
        let h = CliffordRep::h(0);
        let s = CliffordRep::sz(0);
        let cx = CliffordRep::cx(0, 1);
        let composed = h.compose(&s).compose(&cx);
        assert_inverse_correct(&composed);
    }

    #[test]
    fn test_inverse_pauli_gates() {
        // X, Y, Z gates are self-inverse
        assert_inverse_correct(&CliffordRep::x(0));
        assert_inverse_correct(&CliffordRep::y(0));
        assert_inverse_correct(&CliffordRep::z(0));
    }

    #[test]
    fn test_inverse_sx_sy() {
        assert_inverse_correct(&CliffordRep::sx(0));
        assert_inverse_correct(&CliffordRep::sy(0));
    }

    #[test]
    fn test_inverse_cy() {
        assert_inverse_correct(&CliffordRep::cy(0, 1));
    }

    // ========================================================================
    // Validity tests
    // ========================================================================

    #[test]
    fn test_valid_gates() {
        assert!(CliffordRep::identity(2).is_valid());
        assert!(CliffordRep::h(0).is_valid());
        assert!(CliffordRep::sz(0).is_valid());
        assert!(CliffordRep::szdg(0).is_valid());
        assert!(CliffordRep::x(0).is_valid());
        assert!(CliffordRep::y(0).is_valid());
        assert!(CliffordRep::z(0).is_valid());
        assert!(CliffordRep::cx(0, 1).is_valid());
        assert!(CliffordRep::cz(0, 1).is_valid());
        assert!(CliffordRep::swap(0, 1).is_valid());
        assert!(CliffordRep::cy(0, 1).is_valid());
        assert!(CliffordRep::sx(0).is_valid());
        assert!(CliffordRep::sy(0).is_valid());
    }

    #[test]
    fn test_invalid_non_symplectic() {
        // Construct an invalid Clifford: X_0 -> X_0, Z_0 -> X_0
        // This violates anticommutativity: [X_0, X_0] = 0 but should be != 0
        let mut bad = CliffordRep::identity(1);
        bad.z_images[0] = PauliString::x(0);
        assert!(!bad.is_valid());
    }

    #[test]
    fn test_invalid_imaginary_phase() {
        // Construct a Clifford with imaginary phase on a generator image
        let mut bad = CliffordRep::identity(1);
        bad.x_images[0] = i * PauliString::z(0);
        assert!(!bad.is_valid());
    }

    // ========================================================================
    // Display test
    // ========================================================================

    #[test]
    fn test_display() {
        let h = CliffordRep::h(0);
        let s = h.to_string();
        assert!(s.contains("X_0 ->"));
        assert!(s.contains("Z_0 ->"));
    }

    // ========================================================================
    // Composition properties
    // ========================================================================

    #[test]
    fn test_composed_cliffords_are_valid() {
        let h = CliffordRep::h(0);
        let s = CliffordRep::sz(0);
        let cx = CliffordRep::cx(0, 1);
        assert!(h.compose(&s).is_valid());
        assert!(s.compose(&h).is_valid());
        assert!(h.compose(&cx).is_valid());
    }

    #[test]
    fn test_inverse_is_valid() {
        let gates = vec![
            CliffordRep::h(0),
            CliffordRep::sz(0),
            CliffordRep::cx(0, 1),
            CliffordRep::cz(0, 1),
            CliffordRep::cy(0, 1),
        ];
        for gate in &gates {
            assert!(
                gate.inverse().is_valid(),
                "inverse of {gate:?} is not valid"
            );
        }
    }

    // ========================================================================
    // Direct gate transformation tests
    // ========================================================================

    #[test]
    fn test_sdg_transforms() {
        let sdg = CliffordRep::szdg(0);
        // Sdg: X -> -Y, Z -> Z
        let result = sdg.apply(&PauliString::x(0));
        assert_eq!(result.get(0), Pauli::Y);
        assert_eq!(result.phase(), QuarterPhase::MinusOne);

        assert_eq!(sdg.apply(&PauliString::z(0)), PauliString::z(0));
    }

    #[test]
    fn test_x_gate_transforms() {
        let x = CliffordRep::x(0);
        // X: X -> X, Z -> -Z
        assert_eq!(x.apply(&PauliString::x(0)), PauliString::x(0));
        let result = x.apply(&PauliString::z(0));
        assert_eq!(result.get(0), Pauli::Z);
        assert_eq!(result.phase(), QuarterPhase::MinusOne);
    }

    #[test]
    fn test_y_gate_transforms() {
        let y = CliffordRep::y(0);
        // Y: X -> -X, Z -> -Z
        let rx = y.apply(&PauliString::x(0));
        assert_eq!(rx.get(0), Pauli::X);
        assert_eq!(rx.phase(), QuarterPhase::MinusOne);

        let rz = y.apply(&PauliString::z(0));
        assert_eq!(rz.get(0), Pauli::Z);
        assert_eq!(rz.phase(), QuarterPhase::MinusOne);
    }

    #[test]
    fn test_z_gate_transforms() {
        let z = CliffordRep::z(0);
        // Z: X -> -X, Z -> Z
        let rx = z.apply(&PauliString::x(0));
        assert_eq!(rx.get(0), Pauli::X);
        assert_eq!(rx.phase(), QuarterPhase::MinusOne);

        assert_eq!(z.apply(&PauliString::z(0)), PauliString::z(0));
    }

    #[test]
    fn test_sx_transforms() {
        let sx = CliffordRep::sx(0);
        // SX: X -> X, Z -> -Y
        assert_eq!(sx.apply(&PauliString::x(0)), PauliString::x(0));
        let rz = sx.apply(&PauliString::z(0));
        assert_eq!(rz.get(0), Pauli::Y);
        assert_eq!(rz.phase(), QuarterPhase::MinusOne);
    }

    #[test]
    fn test_sy_transforms() {
        let sy = CliffordRep::sy(0);
        // SY: X -> -Z, Z -> X
        let rx = sy.apply(&PauliString::x(0));
        assert_eq!(rx.get(0), Pauli::Z);
        assert_eq!(rx.phase(), QuarterPhase::MinusOne);

        assert_eq!(sy.apply(&PauliString::z(0)), PauliString::x(0));
    }

    #[test]
    fn test_cy_transforms() {
        let cy = CliffordRep::cy(0, 1);

        // X_c -> X_c Y_t
        let rx0 = cy.apply(&PauliString::x(0));
        assert_eq!(rx0.get(0), Pauli::X);
        assert_eq!(rx0.get(1), Pauli::Y);
        assert_eq!(rx0.phase(), QuarterPhase::PlusOne);

        // Z_c -> Z_c
        assert_eq!(cy.apply(&PauliString::z(0)), PauliString::z(0));

        // X_t -> Z_c X_t
        let rx1 = cy.apply(&PauliString::x(1));
        assert_eq!(rx1.get(0), Pauli::Z);
        assert_eq!(rx1.get(1), Pauli::X);
        assert_eq!(rx1.phase(), QuarterPhase::PlusOne);

        // Z_t -> Z_c Z_t
        let rz1 = cy.apply(&PauliString::z(1));
        assert_eq!(rz1.get(0), Pauli::Z);
        assert_eq!(rz1.get(1), Pauli::Z);
        assert_eq!(rz1.phase(), QuarterPhase::PlusOne);
    }

    // ========================================================================
    // extended_to tests
    // ========================================================================

    #[test]
    fn test_extended_to_larger() {
        let h = CliffordRep::h(0);
        assert_eq!(h.num_qubits(), 1);

        let h3 = h.extended_to(3);
        assert_eq!(h3.num_qubits(), 3);

        // Qubit 0 still swaps X <-> Z
        assert_eq!(h3.apply(&PauliString::x(0)), PauliString::z(0));
        assert_eq!(h3.apply(&PauliString::z(0)), PauliString::x(0));

        // Qubits 1,2 are identity
        assert_eq!(h3.apply(&PauliString::x(1)), PauliString::x(1));
        assert_eq!(h3.apply(&PauliString::z(2)), PauliString::z(2));
    }

    #[test]
    fn test_extended_to_same_size() {
        let cx = CliffordRep::cx(0, 1);
        let cx2 = cx.extended_to(2);
        assert_eq!(cx, cx2);
    }

    #[test]
    fn test_extended_to_smaller() {
        let cx = CliffordRep::cx(0, 1);
        let cx_shrunk = cx.extended_to(1);
        // Should return a clone when n <= num_qubits
        assert_eq!(cx_shrunk, cx);
    }

    // ========================================================================
    // apply edge cases
    // ========================================================================

    #[test]
    fn test_apply_identity_pauli() {
        let h = CliffordRep::h(0);
        let id = PauliString::identity();
        let result = h.apply(&id);
        assert_eq!(result.weight(), 0);
        assert_eq!(result.phase(), QuarterPhase::PlusOne);
    }

    #[test]
    fn test_apply_out_of_bounds_qubit() {
        // H on qubit 0 (1-qubit Clifford), applied to X on qubit 5
        let h = CliffordRep::h(0);
        let x5 = PauliString::x(5);
        let result = h.apply(&x5);
        // Out-of-bounds qubits pass through unchanged
        assert_eq!(result.get(5), Pauli::X);
        assert_eq!(result.phase(), QuarterPhase::PlusOne);
    }

    #[test]
    fn test_apply_mixed_in_and_out_of_bounds() {
        // H on qubit 0, applied to X(0) & Z(3)
        let h = CliffordRep::h(0);
        let p = PauliString::x(0) & PauliString::z(3);
        let result = h.apply(&p);
        // X(0) -> Z(0) by H, Z(3) passes through unchanged
        assert_eq!(result.get(0), Pauli::Z);
        assert_eq!(result.get(3), Pauli::Z);
    }

    // ========================================================================
    // Composition edge cases
    // ========================================================================

    #[test]
    fn test_compose_different_qubit_counts() {
        let h = CliffordRep::h(0); // 1 qubit
        let cx = CliffordRep::cx(0, 1); // 2 qubits

        // Should not panic, auto-extends
        let composed = h.compose(&cx);
        assert_eq!(composed.num_qubits(), 2);
        assert!(composed.is_valid());
    }

    #[test]
    fn test_compose_associativity() {
        let h = CliffordRep::h(0);
        let s = CliffordRep::sz(0);
        let cx = CliffordRep::cx(0, 1);

        let lhs = h.compose(&s).compose(&cx); // (H * S) * CX
        let rhs = h.compose(&s.compose(&cx)); // H * (S * CX)

        // Both should produce the same Clifford
        let x0 = PauliString::x(0);
        let z0 = PauliString::z(0);
        let x1 = PauliString::x(1);
        let z1 = PauliString::z(1);

        assert_eq!(lhs.apply(&x0), rhs.apply(&x0));
        assert_eq!(lhs.apply(&z0), rhs.apply(&z0));
        assert_eq!(lhs.apply(&x1), rhs.apply(&x1));
        assert_eq!(lhs.apply(&z1), rhs.apply(&z1));
    }

    // ========================================================================
    // Inverse properties
    // ========================================================================

    #[test]
    fn test_inverse_of_inverse() {
        let s = CliffordRep::sz(0);
        let inv = s.inverse();
        let inv_inv = inv.inverse();

        // (S^{-1})^{-1} = S
        let x0 = PauliString::x(0);
        let z0 = PauliString::z(0);
        assert_eq!(inv_inv.apply(&x0), s.apply(&x0));
        assert_eq!(inv_inv.apply(&z0), s.apply(&z0));
    }

    #[test]
    fn test_inverse_on_multi_qubit_pauli() {
        let cx = CliffordRep::cx(0, 1);
        let inv = cx.inverse();

        // Apply inverse to a multi-qubit Pauli
        let p = PauliString::x(0) & PauliString::z(1);
        let round_trip = cx.apply(&inv.apply(&p));
        assert_eq!(round_trip, p);
    }

    // ========================================================================
    // Enumeration and random Clifford tests
    // ========================================================================

    #[test]
    fn test_single_qubit_cliffords_count() {
        let cliffords = CliffordRep::single_qubit_cliffords(0);
        assert_eq!(cliffords.len(), 24);
    }

    #[test]
    fn test_single_qubit_cliffords_all_valid() {
        let cliffords = CliffordRep::single_qubit_cliffords(0);
        for (idx, c) in cliffords.iter().enumerate() {
            assert!(c.is_valid(), "Clifford {idx} is not valid");
        }
    }

    #[test]
    fn test_single_qubit_cliffords_all_distinct() {
        let cliffords = CliffordRep::single_qubit_cliffords(0);
        for a in 0..24usize {
            for b in (a + 1)..24usize {
                assert_ne!(
                    cliffords[a], cliffords[b],
                    "Cliffords {a} and {b} are equal"
                );
            }
        }
    }

    #[test]
    fn test_single_qubit_cliffords_identity_first() {
        let cliffords = CliffordRep::single_qubit_cliffords(0);
        assert_eq!(cliffords[0], CliffordRep::identity(1));
    }

    #[test]
    fn test_random_single_qubit_is_valid() {
        let mut rng = rand::rng();
        for _ in 0..50 {
            let c = CliffordRep::random_single_qubit(0, &mut rng);
            assert!(c.is_valid());
        }
    }

    #[test]
    fn test_random_multi_qubit_is_valid() {
        let mut rng = rand::rng();
        for n in 1..=4 {
            let c = CliffordRep::random(n, 2 * n, &mut rng);
            assert!(c.is_valid(), "random {n}-qubit Clifford is invalid");
            assert_eq!(c.num_qubits(), n);
        }
    }

    // ========================================================================
    // Ergonomic API tests (constructors + Mul)
    // ========================================================================

    #[test]
    fn test_mul_operator() {
        // H * SZ * H = SX (sqrt-X)
        use super::constructors::*;

        let sx_composed = H(0) * SZ(0) * H(0);
        let sx_direct = CliffordRep::sx(0);

        // Both should transform generators the same way
        let x0 = PauliString::x(0);
        let z0 = PauliString::z(0);
        assert_eq!(sx_composed.apply(&x0), sx_direct.apply(&x0));
        assert_eq!(sx_composed.apply(&z0), sx_direct.apply(&z0));
    }

    #[test]
    fn test_mul_multi_qubit() {
        use super::constructors::*;

        // Bell state circuit: H(0) * CX(0,1)
        // Convention: compose(other) means apply other first, then self
        // So H(0) * CX(0,1) = "apply CX first, then H"
        let bell = H(0) * CX(0, 1);
        assert_eq!(bell.num_qubits(), 2);
        assert!(bell.is_valid());

        // CX maps X(0) -> X(0)X(1), then H maps X(0) -> Z(0)
        // So bell maps X(0) -> Z(0)X(1)
        let result = bell.apply(&PauliString::x(0));
        assert_eq!(result.get(0), Pauli::Z);
        assert_eq!(result.get(1), Pauli::X);
        assert_eq!(result.phase(), QuarterPhase::PlusOne);

        // CX maps Z(1) -> Z(0)Z(1), then H maps Z(0) -> X(0)
        // So bell maps Z(1) -> X(0)Z(1)
        let result = bell.apply(&PauliString::z(1));
        assert_eq!(result.get(0), Pauli::X);
        assert_eq!(result.get(1), Pauli::Z);
        assert_eq!(result.phase(), QuarterPhase::PlusOne);
    }

    #[test]
    fn test_mul_with_references() {
        let h = CliffordRep::h(0);
        let s = CliffordRep::sz(0);

        // All four Mul variants should work
        let _ = h.clone() * s.clone();
        let _ = h.clone() * &s;
        let _ = &h * s.clone();
        let _ = &h * &s;
    }

    #[test]
    fn test_from_pauli_string() {
        // X gate as Clifford: X * Z * X† = -Z, X * X * X† = X
        let x_cliff = CliffordRep::from(PauliString::x(0));
        assert!(x_cliff.is_valid());

        let z0 = PauliString::z(0);
        let result = x_cliff.apply(&z0);
        assert_eq!(result.get(0), Pauli::Z);
        assert_eq!(result.phase(), QuarterPhase::MinusOne); // X flips Z sign
    }

    #[test]
    fn test_from_pauli_string_matches_gate() {
        // The PauliString X(0) as Clifford should match CliffordRep::x(0)
        let from_pauli = CliffordRep::from(PauliString::x(0));
        let from_gate = CliffordRep::x(0);

        let x0 = PauliString::x(0);
        let z0 = PauliString::z(0);
        assert_eq!(from_pauli.apply(&x0), from_gate.apply(&x0));
        assert_eq!(from_pauli.apply(&z0), from_gate.apply(&z0));
    }

    #[test]
    fn test_from_pauli_string_y() {
        let y_cliff = CliffordRep::from(PauliString::y(0));
        let y_gate = CliffordRep::y(0);

        let x0 = PauliString::x(0);
        let z0 = PauliString::z(0);
        assert_eq!(y_cliff.apply(&x0), y_gate.apply(&x0));
        assert_eq!(y_cliff.apply(&z0), y_gate.apply(&z0));
    }

    #[test]
    fn test_from_pauli_string_multi_qubit() {
        // X(0)Z(1) as Clifford
        let xz = PauliString::x(0) & PauliString::z(1);
        let cliff = CliffordRep::from(xz);
        assert!(cliff.is_valid());
        assert_eq!(cliff.num_qubits(), 2);

        // Should match composing X(0) and Z(1) gates
        let composed = CliffordRep::x(0).compose(&CliffordRep::z(1));
        let x0 = PauliString::x(0);
        let z0 = PauliString::z(0);
        let x1 = PauliString::x(1);
        let z1 = PauliString::z(1);
        assert_eq!(cliff.apply(&x0), composed.apply(&x0));
        assert_eq!(cliff.apply(&z0), composed.apply(&z0));
        assert_eq!(cliff.apply(&x1), composed.apply(&x1));
        assert_eq!(cliff.apply(&z1), composed.apply(&z1));
    }

    #[test]
    fn test_clifford_enum_on_qubit() {
        use crate::clifford::Clifford;

        let c = Clifford::H.on_qubit(0);
        assert!(c.is_valid());
        assert_eq!(c, CliffordRep::h(0));
    }

    #[test]
    fn test_compose_matches_simulator_convention() {
        // CliffordGateable: H3 = self.sz(&[q]).y(&[q]) -- SZ first, then Y
        // H3 docs: X → Y, Z → -Z
        //
        // sim.gate1(q).gate2(q) applies gate1 first, then gate2 to state.
        // CliffordRep: cliff(P) = C·P·C† (stabilizer convention)
        // compose(other) = "apply other's transform first, then self's"
        //
        // For U = gate2·gate1 (gate1 first in state):
        //   P → U·P·U† = gate2·gate1·P·gate1†·gate2†
        //   = gate2_cliff(gate1_cliff(P))
        //   = gate2_cliff.compose(gate1_cliff)
        //
        // So sim.sz(&[q]).y(&[q]) → Y.compose(SZ)
        let h3_composed = CliffordRep::y(0).compose(&CliffordRep::sz(0));
        let x0 = PauliString::x(0);
        let z0 = PauliString::z(0);
        assert_eq!(
            h3_composed.apply(&x0),
            PauliString::y(0),
            "H3(X) should be Y"
        );
        assert_eq!(
            h3_composed.apply(&z0),
            -PauliString::z(0),
            "H3(Z) should be -Z"
        );
    }

    #[test]
    fn test_all_single_qubit_gates_valid() {
        let gates = [
            CliffordRep::h(0),
            CliffordRep::h2(0),
            CliffordRep::h3(0),
            CliffordRep::h4(0),
            CliffordRep::h5(0),
            CliffordRep::h6(0),
            CliffordRep::sz(0),
            CliffordRep::szdg(0),
            CliffordRep::sx(0),
            CliffordRep::sxdg(0),
            CliffordRep::sy(0),
            CliffordRep::sydg(0),
            CliffordRep::sz(0),
            CliffordRep::szdg(0),
            CliffordRep::x(0),
            CliffordRep::y(0),
            CliffordRep::z(0),
            CliffordRep::f(0),
            CliffordRep::fdg(0),
            CliffordRep::f2(0),
            CliffordRep::f2dg(0),
            CliffordRep::f3(0),
            CliffordRep::f3dg(0),
            CliffordRep::f4(0),
            CliffordRep::f4dg(0),
        ];
        for (idx, gate) in gates.iter().enumerate() {
            assert!(gate.is_valid(), "Single-qubit gate {idx} is not valid");
        }
    }

    #[test]
    fn test_all_two_qubit_gates_valid() {
        let gates = [
            CliffordRep::cx(0, 1),
            CliffordRep::cy(0, 1),
            CliffordRep::cz(0, 1),
            CliffordRep::swap(0, 1),
            CliffordRep::iswap(0, 1),
            CliffordRep::g(0, 1),
            CliffordRep::sxx(0, 1),
            CliffordRep::sxxdg(0, 1),
            CliffordRep::syy(0, 1),
            CliffordRep::syydg(0, 1),
            CliffordRep::szz(0, 1),
            CliffordRep::szzdg(0, 1),
        ];
        for (idx, gate) in gates.iter().enumerate() {
            assert!(gate.is_valid(), "Two-qubit gate {idx} is not valid");
        }
    }

    #[test]
    fn test_dagger_pairs_are_inverses() {
        let pairs: Vec<(CliffordRep, CliffordRep)> = vec![
            (CliffordRep::sx(0), CliffordRep::sxdg(0)),
            (CliffordRep::sy(0), CliffordRep::sydg(0)),
            (CliffordRep::sz(0), CliffordRep::szdg(0)),
            (CliffordRep::f(0), CliffordRep::fdg(0)),
            (CliffordRep::f2(0), CliffordRep::f2dg(0)),
            (CliffordRep::f3(0), CliffordRep::f3dg(0)),
            (CliffordRep::f4(0), CliffordRep::f4dg(0)),
            (CliffordRep::sxx(0, 1), CliffordRep::sxxdg(0, 1)),
            (CliffordRep::syy(0, 1), CliffordRep::syydg(0, 1)),
            (CliffordRep::szz(0, 1), CliffordRep::szzdg(0, 1)),
        ];
        for (idx, (gate, gate_dg)) in pairs.iter().enumerate() {
            let product = gate.compose(gate_dg);
            let n = product.num_qubits();
            let identity = CliffordRep::identity(n);
            for q in 0..n {
                assert_eq!(
                    product.x_image(q),
                    identity.x_image(q),
                    "Pair {idx}: G * G† x_image({q}) mismatch"
                );
                assert_eq!(
                    product.z_image(q),
                    identity.z_image(q),
                    "Pair {idx}: G * G† z_image({q}) mismatch"
                );
            }
        }
    }

    #[test]
    fn test_face_gate_cyclic() {
        // F: X -> Y -> Z -> X (cyclic permutation)
        let f = CliffordRep::f(0);
        let x0 = PauliString::x(0);
        let y0 = PauliString::y(0);
        let z0 = PauliString::z(0);
        assert_eq!(f.apply(&x0), y0);
        assert_eq!(f.apply(&z0), x0);
        // F^3 = identity (up to global phase)
        let f3 = f.compose(&f).compose(&f);
        // F^3 should act as identity on generators
        assert_eq!(f3.apply(&x0), x0);
        assert_eq!(f3.apply(&z0), z0);
    }

    #[test]
    fn test_hadamard_variants_are_involutions() {
        // All H variants are self-inverse (H^2 = I)
        for gate in [
            CliffordRep::h(0),
            CliffordRep::h2(0),
            CliffordRep::h3(0),
            CliffordRep::h4(0),
            CliffordRep::h5(0),
            CliffordRep::h6(0),
        ] {
            let product = gate.compose(&gate);
            let identity = CliffordRep::identity(1);
            assert_eq!(
                product.x_image(0),
                identity.x_image(0),
                "Hadamard variant not involution on X"
            );
            assert_eq!(
                product.z_image(0),
                identity.z_image(0),
                "Hadamard variant not involution on Z"
            );
        }
    }

    #[test]
    fn test_constructors_complete() {
        use super::constructors::*;
        // Verify all constructors compile and are valid
        let singles = [
            H(0),
            H2(0),
            H3(0),
            H4(0),
            H5(0),
            H6(0),
            SX(0),
            SXdg(0),
            SY(0),
            SYdg(0),
            SZ(0),
            SZdg(0),
            F(0),
            Fdg(0),
            F2(0),
            F2dg(0),
            F3(0),
            F3dg(0),
            F4(0),
            F4dg(0),
        ];
        for s in &singles {
            assert!(s.is_valid());
        }

        let doubles = [
            CX(0, 1),
            CY(0, 1),
            CZ(0, 1),
            SWAP(0, 1),
            SXX(0, 1),
            SXXdg(0, 1),
            SYY(0, 1),
            SYYdg(0, 1),
            SZZ(0, 1),
            SZZdg(0, 1),
            ISWAP(0, 1),
            G(0, 1),
        ];
        for d in &doubles {
            assert!(d.is_valid());
        }

        assert!(Id(3).is_valid());
    }

    // ====== CliffordRep::inverse() direct verification ======

    #[test]
    fn inverse_equals_known_dagger_all_1q() {
        use crate::clifford::Clifford;
        // For every 1q Clifford, CliffordRep::inverse() should match the
        // known dagger gate's CliffordRep.
        for &cliff in Clifford::all_1q() {
            let rep = cliff.on_qubit(0);
            let inv = rep.inverse();
            let dagger_rep = cliff.inverse().on_qubit(0);
            assert_eq!(
                inv, dagger_rep,
                "CliffordRep::inverse() disagrees with known dagger for {cliff}"
            );
        }
    }

    #[test]
    fn inverse_equals_known_dagger_all_2q() {
        use crate::clifford::Clifford;
        // For every 2q Clifford, CliffordRep::inverse() should match the
        // known dagger gate's CliffordRep.
        for &cliff in Clifford::all_2q() {
            let rep = cliff.on_qubits(0, 1);
            let inv = rep.inverse();
            let dagger_rep = cliff.inverse().on_qubits(0, 1);
            assert_eq!(
                inv, dagger_rep,
                "CliffordRep::inverse() disagrees with known dagger for {cliff}"
            );
        }
    }

    #[test]
    fn inverse_is_involutory() {
        use crate::clifford::Clifford;
        // (C^{-1})^{-1} == C for all Cliffords
        for &cliff in Clifford::all_1q() {
            let rep = cliff.on_qubit(0);
            let double_inv = rep.inverse().inverse();
            assert_eq!(rep, double_inv, "inverse(inverse({cliff})) != {cliff}");
        }
        for &cliff in Clifford::all_2q() {
            let rep = cliff.on_qubits(0, 1);
            let double_inv = rep.inverse().inverse();
            assert_eq!(rep, double_inv, "inverse(inverse({cliff})) != {cliff}");
        }
    }

    // ====== Validity preservation ======

    #[test]
    fn inverse_preserves_validity_all_gates() {
        use crate::clifford::Clifford;
        for &cliff in Clifford::all_1q() {
            let inv = cliff.on_qubit(0).inverse();
            assert!(inv.is_valid(), "inverse of {cliff} is not valid");
        }
        for &cliff in Clifford::all_2q() {
            let inv = cliff.on_qubits(0, 1).inverse();
            assert!(inv.is_valid(), "inverse of {cliff} is not valid");
        }
    }

    #[test]
    fn compose_preserves_validity_1q() {
        use crate::clifford::Clifford;
        let gates_1q: Vec<CliffordRep> = Clifford::all_1q().iter().map(|c| c.on_qubit(0)).collect();
        // Test a representative sample of compositions (all pairs would be 24*24=576)
        for a in &gates_1q {
            for b in &gates_1q {
                let composed = a.compose(b);
                assert!(
                    composed.is_valid(),
                    "compose of two 1q Cliffords is not valid"
                );
            }
        }
    }

    #[test]
    fn compose_preserves_validity_2q() {
        use crate::clifford::Clifford;
        let gates_2q: Vec<CliffordRep> = Clifford::all_2q()
            .iter()
            .map(|c| c.on_qubits(0, 1))
            .collect();
        // Test all pairs (14*14=196)
        for a in &gates_2q {
            for b in &gates_2q {
                let composed = a.compose(b);
                assert!(
                    composed.is_valid(),
                    "compose of two 2q Cliffords is not valid"
                );
            }
        }
    }

    // ====== single_qubit_cliffords matches named Cliffords ======

    #[test]
    fn single_qubit_cliffords_matches_named_set() {
        use crate::clifford::Clifford;

        let from_sequences = CliffordRep::single_qubit_cliffords(0);
        let from_named: Vec<CliffordRep> =
            Clifford::all_1q().iter().map(|c| c.on_qubit(0)).collect();

        assert_eq!(from_sequences.len(), 24);
        assert_eq!(from_named.len(), 24);

        // Every sequence-generated Clifford appears in the named set
        for (idx, seq_cliff) in from_sequences.iter().enumerate() {
            assert!(
                from_named.contains(seq_cliff),
                "sequence Clifford #{idx} not found in named set"
            );
        }

        // Every named Clifford appears in the sequence-generated set
        for named_cliff in &from_named {
            assert!(
                from_sequences.contains(named_cliff),
                "named Clifford not found in sequence set"
            );
        }

        // All 24 are distinct
        for idx_a in 0..from_sequences.len() {
            for idx_b in (idx_a + 1)..from_sequences.len() {
                assert_ne!(
                    from_sequences[idx_a], from_sequences[idx_b],
                    "single_qubit_cliffords() produced duplicates at indices {idx_a} and {idx_b}"
                );
            }
        }
    }

    // ====== inverse on composed (non-named) 2q Cliffords ======

    #[test]
    fn inverse_correct_on_composed_2q_cliffords() {
        use crate::clifford::Clifford;
        let gates_2q: Vec<CliffordRep> = Clifford::all_2q()
            .iter()
            .map(|c| c.on_qubits(0, 1))
            .collect();

        // Test inverse on all pairwise compositions (14*14 = 196 composed gates)
        for (idx_a, a) in gates_2q.iter().enumerate() {
            for (idx_b, b) in gates_2q.iter().enumerate() {
                let composed = a.compose(b);
                assert_inverse_correct(&composed);
                let _ = (idx_a, idx_b);
            }
        }
    }

    // ====== apply() on PauliStrings containing Y ======

    #[test]
    fn apply_y_uses_correct_phase() {
        // Y = iXZ, so applying a Clifford to Y_q should give i * X_image * Z_image.
        // Test on several gates where the result is known.

        // H: X -> Z, Z -> X. So Y -> i*Z*X = i*(iY) = -Y.
        let h = CliffordRep::h(0);
        let y0 = PauliString::y(0);
        let result = h.apply(&y0);
        assert_eq!(result.get(0), Pauli::Y);
        assert_eq!(
            result.phase(),
            QuarterPhase::MinusOne,
            "H: Y should map to -Y"
        );

        // SZ: X -> Y, Z -> Z. So Y -> i*Y*Z = i*(iX) = -X.
        // (Y*Z = iX because Pauli product YZ = iX.)
        let sz = CliffordRep::sz(0);
        let result = sz.apply(&y0);
        assert_eq!(result.get(0), Pauli::X);
        assert_eq!(
            result.phase(),
            QuarterPhase::MinusOne,
            "SZ: Y should map to -X"
        );

        // SX: X -> X, Z -> -Y. So Y -> i*X*(-Y) = -i*X*Y = -i*(iZ) = Z.
        let sx = CliffordRep::sx(0);
        let result = sx.apply(&y0);
        assert_eq!(result.get(0), Pauli::Z);
        assert_eq!(
            result.phase(),
            QuarterPhase::PlusOne,
            "SX: Y should map to Z"
        );
    }

    #[test]
    fn apply_y_on_2q_gates() {
        // CX: X0 -> X0X1, Z0 -> Z0, X1 -> X1, Z1 -> Z0Z1.
        // Y0 = iX0Z0 -> i*(X0X1)*(Z0) = i*X0*Z0*X1 = Y0*X1.
        let cx = CliffordRep::cx(0, 1);
        let y0 = PauliString::y(0);
        let result = cx.apply(&y0);
        assert_eq!(result.get(0), Pauli::Y);
        assert_eq!(result.get(1), Pauli::X);
        assert_eq!(
            result.phase(),
            QuarterPhase::PlusOne,
            "CX: Y0 should map to Y0X1"
        );

        // Y1 = iX1Z1 -> i*(X1)*(Z0Z1) = i*Z0*X1*Z1 = Z0*(iX1Z1) = Z0*Y1.
        let y1 = PauliString::y(1);
        let result = cx.apply(&y1);
        assert_eq!(result.get(0), Pauli::Z);
        assert_eq!(result.get(1), Pauli::Y);
        assert_eq!(
            result.phase(),
            QuarterPhase::PlusOne,
            "CX: Y1 should map to Z0Y1"
        );
    }

    #[test]
    fn apply_y_matches_matrix_conjugation_all_1q() {
        // For every 1q Clifford, verify that apply(Y) matches the derived Y image
        // computed from X and Z images: Y_image = i * X_image * Z_image.
        use crate::clifford::Clifford;

        for &cliff in Clifford::all_1q() {
            let rep = cliff.on_qubit(0);
            let y0 = PauliString::y(0);
            let result = rep.apply(&y0);

            // Derive expected: i * X_image * Z_image
            let x_img = rep.x_image(0).clone();
            let z_img = rep.z_image(0).clone();
            let expected = i * (x_img * &z_img);

            assert_eq!(
                result, expected,
                "{cliff}: apply(Y) disagrees with i * X_image * Z_image"
            );
        }
    }

    #[test]
    fn apply_y_matches_matrix_conjugation_all_2q() {
        // For every 2q Clifford, verify apply(Y0) and apply(Y1).
        use crate::clifford::Clifford;

        for &cliff in Clifford::all_2q() {
            let rep = cliff.on_qubits(0, 1);

            for q in 0..2 {
                let y_q = PauliString::y(q);
                let result = rep.apply(&y_q);

                let x_img = rep.x_image(q).clone();
                let z_img = rep.z_image(q).clone();
                let expected = i * (x_img * &z_img);

                assert_eq!(
                    result, expected,
                    "{cliff}: apply(Y{q}) disagrees with i * X{q}_image * Z{q}_image"
                );
            }
        }
    }

    // ====== apply() on multi-qubit PauliStrings ======

    #[test]
    fn apply_multiqubit_pauli_strings() {
        // Test apply() on 2-qubit PauliStrings like X0Z1, Y0Y1, etc.
        // The rule: apply(P0 * P1) should equal apply(P0) * apply(P1)
        // because Clifford conjugation distributes over Pauli products
        // (up to phase from reordering, but apply handles this).
        use crate::clifford::Clifford;

        let paulis_1q = [PauliString::x(0), PauliString::y(0), PauliString::z(0)];
        let paulis_1q_q1 = [PauliString::x(1), PauliString::y(1), PauliString::z(1)];

        for &cliff in Clifford::all_2q() {
            let rep = cliff.on_qubits(0, 1);

            for p0 in &paulis_1q {
                for p1 in &paulis_1q_q1 {
                    // Build 2-qubit Pauli string P0 * P1
                    let combined = p0 * p1;

                    // Apply the Clifford to the combined string
                    let result = rep.apply(&combined);

                    // Apply separately and multiply
                    let img_p0 = rep.apply(p0);
                    let img_p1 = rep.apply(p1);
                    let expected = img_p0 * &img_p1;

                    assert_eq!(
                        result,
                        expected,
                        "{cliff}: apply({}) != apply({}) * apply({})",
                        combined.to_sparse_str(),
                        p0.to_sparse_str(),
                        p1.to_sparse_str(),
                    );
                }
            }
        }
    }

    // ====== CliffordRep for 2q gates on non-adjacent qubits in 3-qubit register ======

    #[test]
    fn clifford_rep_nonadjacent_inverse_correct() {
        // Test that CliffordRep for 2q gates on qubits (0, 2) has correct inverse.
        use crate::clifford::Clifford;

        for &cliff in Clifford::all_2q() {
            let rep = cliff.on_qubits(0, 2);
            assert_inverse_correct(&rep);
        }
    }

    #[test]
    fn clifford_rep_nonadjacent_compose_with_adjacent() {
        // Compose a gate on (0, 1) with a gate on (0, 2) and verify inverse.
        let gates_01: Vec<CliffordRep> = vec![
            CliffordRep::cx(0, 1),
            CliffordRep::cz(0, 1),
            CliffordRep::iswap(0, 1),
            CliffordRep::g(0, 1),
        ];

        let gates_02: Vec<CliffordRep> = vec![
            CliffordRep::cx(0, 2),
            CliffordRep::cz(0, 2),
            CliffordRep::iswap(0, 2),
            CliffordRep::g(0, 2),
        ];

        for a in &gates_01 {
            for b in &gates_02 {
                let composed = a.compose(b);
                assert_eq!(composed.num_qubits(), 3);
                assert_inverse_correct(&composed);
            }
        }
    }

    #[test]
    fn clifford_rep_nonadjacent_apply_identity_on_spectator() {
        // A 2q gate on (0, 2) should act as identity on qubit 1.
        // Verify: apply(X1) = X1 and apply(Z1) = Z1.
        use crate::clifford::Clifford;

        let x1 = PauliString::x(1);
        let z1 = PauliString::z(1);

        for &cliff in Clifford::all_2q() {
            let rep = cliff.on_qubits(0, 2);

            let x1_img = rep.apply(&x1);
            assert_eq!(
                x1_img,
                x1,
                "{cliff} on (0,2): X1 should be unchanged but got {}",
                x1_img.to_sparse_str()
            );

            let z1_img = rep.apply(&z1);
            assert_eq!(
                z1_img,
                z1,
                "{cliff} on (0,2): Z1 should be unchanged but got {}",
                z1_img.to_sparse_str()
            );
        }
    }

    #[test]
    fn clifford_rep_nonadjacent_matches_adjacent_pauli_images() {
        // For each 2q Clifford, the Pauli images on qubits (0, 2) should
        // be the "same" as on (0, 1) but with qubit 1 remapped to qubit 2.
        use crate::clifford::Clifford;

        for &cliff in Clifford::all_2q() {
            let rep_01 = cliff.on_qubits(0, 1);
            let rep_02 = cliff.on_qubits(0, 2);

            // X0 image: in rep_01 the result involves qubits {0,1}.
            // In rep_02 the result should be the same but with qubit 1 -> qubit 2.
            let x0_01 = rep_01.x_image(0);
            let x0_02 = rep_02.x_image(0);

            // Compare: same Pauli on qubit 0, and qubit 1's Pauli in rep_01
            // should appear on qubit 2 in rep_02.
            assert_eq!(
                x0_01.get(0),
                x0_02.get(0),
                "{cliff}: X0 image qubit-0 component differs between (0,1) and (0,2)"
            );
            assert_eq!(
                x0_01.get(1),
                x0_02.get(2),
                "{cliff}: X0 image: qubit-1 component in (0,1) should match qubit-2 in (0,2)"
            );
            assert_eq!(
                x0_01.phase(),
                x0_02.phase(),
                "{cliff}: X0 image phase differs between (0,1) and (0,2)"
            );

            // Z0 image
            let z0_01 = rep_01.z_image(0);
            let z0_02 = rep_02.z_image(0);
            assert_eq!(z0_01.get(0), z0_02.get(0), "{cliff}: Z0 q0 mismatch");
            assert_eq!(z0_01.get(1), z0_02.get(2), "{cliff}: Z0 q1->q2 mismatch");
            assert_eq!(z0_01.phase(), z0_02.phase(), "{cliff}: Z0 phase mismatch");
        }
    }
}
