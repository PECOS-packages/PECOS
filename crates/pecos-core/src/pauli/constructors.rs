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

//! Ergonomic shorthand constructors for [`PauliString`].
//!
//! These free functions provide concise syntax for constructing Pauli strings,
//! mirroring the mathematical notation used in quantum error correction.
//!
//! # Examples
//!
//! ```
//! use pecos_core::pauli::constructors::*;
//! use pecos_core::PauliOperator;
//!
//! // Single-qubit Paulis
//! let p = X(0) & Z(1);
//! assert_eq!(p.weight(), 2);
//!
//! // Multi-qubit batch constructors
//! let stab = Zs(&[0, 1]);  // ZZ on qubits 0 and 1
//! assert_eq!(stab.weight(), 2);
//!
//! // Mixed Pauli strings via tensor product
//! let check = Xs(&[0, 1]) & Zs(&[2, 3]);  // XXZZ
//! assert_eq!(check.weight(), 4);
//!
//! // Phase multiplication
//! let p = -X(0);  // -X on qubit 0
//! ```

use crate::PauliString;

/// A trait for types that can be used as qubit index arguments.
///
/// This allows `Xs`, `Ys`, `Zs` to accept slices, arrays, `Vec`, and ranges.
pub trait QubitArgs {
    /// Collects the qubit indices into a `Vec<usize>`.
    fn collect_qubits(self) -> Vec<usize>;
}

impl QubitArgs for &[usize] {
    fn collect_qubits(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> QubitArgs for [usize; N] {
    fn collect_qubits(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const N: usize> QubitArgs for &[usize; N] {
    fn collect_qubits(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl QubitArgs for Vec<usize> {
    fn collect_qubits(self) -> Vec<usize> {
        self
    }
}

impl QubitArgs for std::ops::Range<usize> {
    fn collect_qubits(self) -> Vec<usize> {
        self.collect()
    }
}

impl QubitArgs for std::ops::RangeInclusive<usize> {
    fn collect_qubits(self) -> Vec<usize> {
        self.collect()
    }
}

/// Pauli X on a single qubit.
///
/// # Examples
///
/// ```
/// use pecos_core::pauli::constructors::X;
/// use pecos_core::{Pauli, PauliOperator};
///
/// let p = X(0);
/// assert_eq!(p.get(0), Pauli::X);
/// assert_eq!(p.weight(), 1);
/// ```
#[must_use]
#[allow(non_snake_case)]
pub fn X(qubit: usize) -> PauliString {
    PauliString::x(qubit)
}

/// Pauli Y on a single qubit.
#[must_use]
#[allow(non_snake_case)]
pub fn Y(qubit: usize) -> PauliString {
    PauliString::y(qubit)
}

/// Pauli Z on a single qubit.
#[must_use]
#[allow(non_snake_case)]
pub fn Z(qubit: usize) -> PauliString {
    PauliString::z(qubit)
}

/// Identity (empty Pauli string).
#[must_use]
#[allow(non_snake_case)]
pub fn I() -> PauliString {
    PauliString::identity()
}

/// Pauli X on multiple qubits.
///
/// `Xs([0, 2, 5])` is equivalent to `X(0) & X(2) & X(5)`.
///
/// Accepts arrays, slices, `Vec<usize>`, and ranges.
///
/// # Examples
///
/// ```
/// use pecos_core::pauli::constructors::Xs;
/// use pecos_core::PauliOperator;
///
/// let p = Xs([0, 1, 2]);
/// assert_eq!(p.weight(), 3);
///
/// let p = Xs(0..3);
/// assert_eq!(p.weight(), 3);
/// ```
#[must_use]
#[allow(non_snake_case)]
pub fn Xs(qubits: impl QubitArgs) -> PauliString {
    PauliString::xs(&qubits.collect_qubits())
}

/// Pauli Y on multiple qubits.
///
/// `Ys([0, 2, 5])` is equivalent to `Y(0) & Y(2) & Y(5)`.
///
/// Accepts arrays, slices, `Vec<usize>`, and ranges.
#[must_use]
#[allow(non_snake_case)]
pub fn Ys(qubits: impl QubitArgs) -> PauliString {
    PauliString::ys(&qubits.collect_qubits())
}

/// Pauli Z on multiple qubits.
///
/// `Zs([0, 2, 5])` is equivalent to `Z(0) & Z(2) & Z(5)`.
///
/// Accepts arrays, slices, `Vec<usize>`, and ranges.
///
/// # Examples
///
/// ```
/// use pecos_core::pauli::constructors::Zs;
/// use pecos_core::PauliOperator;
///
/// let stab = Zs([0, 1]);
/// assert_eq!(stab.weight(), 2);
///
/// let stab = Zs(0..=1);
/// assert_eq!(stab.weight(), 2);
/// ```
#[must_use]
#[allow(non_snake_case)]
pub fn Zs(qubits: impl QubitArgs) -> PauliString {
    PauliString::zs(&qubits.collect_qubits())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pauli::algebra::i;
    use crate::{Pauli, PauliOperator, QuarterPhase};

    #[test]
    fn test_single_qubit() {
        let x = X(0);
        assert_eq!(x.get(0), Pauli::X);
        assert_eq!(x.weight(), 1);

        let y = Y(3);
        assert_eq!(y.get(3), Pauli::Y);

        let z = Z(1);
        assert_eq!(z.get(1), Pauli::Z);
    }

    #[test]
    fn test_identity() {
        let id = I();
        assert_eq!(id.weight(), 0);
        assert!(id.is_identity());
    }

    #[test]
    fn test_tensor_product() {
        let p = X(0) & Z(1);
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.get(1), Pauli::Z);
        assert_eq!(p.weight(), 2);
    }

    #[test]
    fn test_batch() {
        let p = Xs([0, 1, 2]);
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.get(1), Pauli::X);
        assert_eq!(p.get(2), Pauli::X);
        assert_eq!(p.weight(), 3);
    }

    #[test]
    fn test_batch_array() {
        // Array literal without &
        let p = Xs([0, 1, 2]);
        assert_eq!(p.weight(), 3);

        let p = Zs([0, 1]);
        assert_eq!(p.weight(), 2);
    }

    #[test]
    fn test_batch_range() {
        let p = Xs(0..3);
        assert_eq!(p.weight(), 3);
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.get(1), Pauli::X);
        assert_eq!(p.get(2), Pauli::X);

        let p = Zs(0..=2);
        assert_eq!(p.weight(), 3);
    }

    #[test]
    fn test_batch_vec() {
        let qubits = vec![0, 2, 4];
        let p = Xs(qubits);
        assert_eq!(p.weight(), 3);
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.get(2), Pauli::X);
        assert_eq!(p.get(4), Pauli::X);
    }

    #[test]
    fn test_mixed_batch_tensor() {
        // XXZZ
        let p = Xs([0, 1]) & Zs([2, 3]);
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.get(1), Pauli::X);
        assert_eq!(p.get(2), Pauli::Z);
        assert_eq!(p.get(3), Pauli::Z);
        assert_eq!(p.weight(), 4);
    }

    #[test]
    fn test_multiplication() {
        // X * Y = iZ
        let p = X(0) * Y(0);
        assert_eq!(p.get(0), Pauli::Z);
        assert_eq!(p.phase(), QuarterPhase::PlusI);
    }

    #[test]
    fn test_negation() {
        let p = -X(0);
        assert_eq!(p.get(0), Pauli::X);
        assert_eq!(p.phase(), QuarterPhase::MinusOne);
    }

    #[test]
    fn test_imaginary_phase() {
        let p = i * X(0);
        assert_eq!(p.phase(), QuarterPhase::PlusI);

        let p = -i * Z(1);
        assert_eq!(p.phase(), QuarterPhase::MinusI);
    }

    #[test]
    fn test_commutation() {
        // X and Z anticommute on same qubit
        assert!(!X(0).commutes_with(&Z(0)));
        // X and Z commute on different qubits
        assert!(X(0).commutes_with(&Z(1)));
    }

    #[test]
    fn test_stabilizer_like_usage() {
        // Repetition code stabilizers
        let s1 = Zs([0, 1]);
        let s2 = Zs([1, 2]);
        // Stabilizers commute
        assert!(s1.commutes_with(&s2));

        // Logical X anticommutes with stabilizers? No, XXXXX commutes with ZZ checks.
        let lx = Xs([0, 1, 2]);
        // X on all vs ZZ on 0,1: X0 anticommutes with Z0, X1 anticommutes with Z1 => 2 anticommutations => commutes
        assert!(lx.commutes_with(&s1));
    }
}
