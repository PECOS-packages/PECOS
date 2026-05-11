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

//! `UnitaryRep` algebra for Pauli strings with ergonomic syntax.
//!
//! This module extends `PauliString` with operator overloading for natural
//! mathematical syntax.
//!
//! # Operators
//!
//! - `&` - Tensor product (operators on different qubits)
//! - `*` - Multiplication (Pauli algebra on same qubit, tensor on different)
//! - `-` - Negation (multiply phase by -1)
//! - `i *` - Phase multiplication
//!
//! # Examples
//!
//! ```
//! use pecos_core::pauli::algebra::i;
//! use pecos_core::PauliString;
//!
//! // Single qubit operators
//! let x0 = PauliString::x(0);
//! let z1 = PauliString::z(1);
//!
//! // Tensor products using & operator
//! let ps = PauliString::x(0) & PauliString::z(1);  // X on qubit 0, Z on qubit 1
//!
//! // Multiplication using * operator
//! let ps = PauliString::x(0) * PauliString::y(0);  // X * Y = iZ (same qubit)
//! let ps = PauliString::x(0) * PauliString::z(1);  // X ⊗ Z (different qubits)
//!
//! // With phase
//! let ps = -i * (PauliString::x(0) & PauliString::y(1));  // -i(X ⊗ Y)
//! let ps = i * PauliString::x(0);                         // iX
//! let ps = -PauliString::x(0);                            // -X
//! ```

use crate::qubit_support::overlapping_qubits;
use crate::{Pauli, PauliString, Phase, QuarterPhase, QubitId};
use std::ops::{BitAnd, Mul, Neg};

// ============================================================================
// Phase types
// ============================================================================

/// Imaginary unit constant for phase specification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImaginaryUnit;

/// The imaginary unit `i`.
#[allow(non_upper_case_globals)]
pub const i: ImaginaryUnit = ImaginaryUnit;

impl Neg for ImaginaryUnit {
    type Output = NegImaginaryUnit;

    fn neg(self) -> NegImaginaryUnit {
        NegImaginaryUnit
    }
}

/// Negative imaginary unit (-i).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NegImaginaryUnit;

// ============================================================================
// Tensor product: & operator
// ============================================================================

impl BitAnd for PauliString {
    type Output = PauliString;

    fn bitand(self, rhs: PauliString) -> PauliString {
        let overlap = overlapping_qubits(self.qubits(), rhs.qubits());
        assert!(
            overlap.is_empty(),
            "tensor product requires disjoint Pauli support; overlapping qubits: {overlap:?}"
        );

        // Combine phases
        let new_phase = self.phase().multiply(&rhs.phase());

        // Combine paulis.
        let mut paulis: Vec<(Pauli, QubitId)> = self.iter_pairs().collect();
        paulis.extend(rhs.iter_pairs());

        // Sort by qubit for canonical form
        paulis.sort_by_key(|(_, q)| *q);

        PauliString::with_phase_and_paulis(new_phase, paulis)
    }
}

impl BitAnd<&PauliString> for PauliString {
    type Output = PauliString;

    fn bitand(self, rhs: &PauliString) -> PauliString {
        self & rhs.clone()
    }
}

impl BitAnd<PauliString> for &PauliString {
    type Output = PauliString;

    fn bitand(self, rhs: PauliString) -> PauliString {
        self.clone() & rhs
    }
}

impl BitAnd<&PauliString> for &PauliString {
    type Output = PauliString;

    fn bitand(self, rhs: &PauliString) -> PauliString {
        self.clone() & rhs.clone()
    }
}

// ============================================================================
// Multiplication: * operator (Pauli algebra)
// ============================================================================

/// Multiply two Paulis on the same qubit.
/// Returns (phase, result) where result may be I.
fn multiply_paulis(a: Pauli, b: Pauli) -> (QuarterPhase, Pauli) {
    use Pauli::{I, X, Y, Z};
    match (a, b) {
        // Identity
        (I, p) | (p, I) => (QuarterPhase::PlusOne, p),

        // Self-inverse: P * P = I
        (X, X) | (Y, Y) | (Z, Z) => (QuarterPhase::PlusOne, I),

        // XY = iZ, YX = -iZ
        (X, Y) => (QuarterPhase::PlusI, Z),
        (Y, X) => (QuarterPhase::MinusI, Z),

        // YZ = iX, ZY = -iX
        (Y, Z) => (QuarterPhase::PlusI, X),
        (Z, Y) => (QuarterPhase::MinusI, X),

        // ZX = iY, XZ = -iY
        (Z, X) => (QuarterPhase::PlusI, Y),
        (X, Z) => (QuarterPhase::MinusI, Y),
    }
}

impl Mul for PauliString {
    type Output = PauliString;

    fn mul(self, rhs: PauliString) -> PauliString {
        // Start with combined phase
        let mut phase = self.phase().multiply(&rhs.phase());

        // Build result paulis, handling overlaps
        let mut result: Vec<(Pauli, QubitId)> = Vec::new();

        // Collect all qubits
        let mut all_qubits: Vec<QubitId> = self
            .iter_pairs()
            .map(|(_, q)| q)
            .chain(rhs.iter_pairs().map(|(_, q)| q))
            .collect();
        all_qubits.sort();
        all_qubits.dedup();

        for qubit in all_qubits {
            let p1 = self.get(usize::from(qubit));
            let p2 = rhs.get(usize::from(qubit));

            let (mul_phase, result_pauli) = multiply_paulis(p1, p2);
            phase = phase.multiply(&mul_phase);

            if result_pauli != Pauli::I {
                result.push((result_pauli, qubit));
            }
        }

        PauliString::with_phase_and_paulis(phase, result)
    }
}

impl Mul<&PauliString> for PauliString {
    type Output = PauliString;

    fn mul(self, rhs: &PauliString) -> PauliString {
        self * rhs.clone()
    }
}

impl Mul<PauliString> for &PauliString {
    type Output = PauliString;

    fn mul(self, rhs: PauliString) -> PauliString {
        self.clone() * rhs
    }
}

impl Mul<&PauliString> for &PauliString {
    type Output = PauliString;

    fn mul(self, rhs: &PauliString) -> PauliString {
        self.clone() * rhs.clone()
    }
}

// ============================================================================
// Negation: - operator
// ============================================================================

impl Neg for PauliString {
    type Output = PauliString;

    fn neg(self) -> PauliString {
        let new_phase = self.phase().multiply(&QuarterPhase::MinusOne);
        PauliString::with_phase_and_paulis(new_phase, self.iter_pairs().collect())
    }
}

impl Neg for &PauliString {
    type Output = PauliString;

    fn neg(self) -> PauliString {
        -self.clone()
    }
}

// ============================================================================
// Phase multiplication: i * PauliString, -i * PauliString
// ============================================================================

impl Mul<PauliString> for ImaginaryUnit {
    type Output = PauliString;

    fn mul(self, rhs: PauliString) -> PauliString {
        let new_phase = rhs.phase().multiply(&QuarterPhase::PlusI);
        PauliString::with_phase_and_paulis(new_phase, rhs.iter_pairs().collect())
    }
}

impl Mul<&PauliString> for ImaginaryUnit {
    type Output = PauliString;

    fn mul(self, rhs: &PauliString) -> PauliString {
        self * rhs.clone()
    }
}

impl Mul<PauliString> for NegImaginaryUnit {
    type Output = PauliString;

    fn mul(self, rhs: PauliString) -> PauliString {
        let new_phase = rhs.phase().multiply(&QuarterPhase::MinusI);
        PauliString::with_phase_and_paulis(new_phase, rhs.iter_pairs().collect())
    }
}

impl Mul<&PauliString> for NegImaginaryUnit {
    type Output = PauliString;

    fn mul(self, rhs: &PauliString) -> PauliString {
        self * rhs.clone()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PauliOperator;

    #[test]
    fn test_single_pauli() {
        let x = PauliString::x(0);
        assert_eq!(x.get(0), Pauli::X);
        assert_eq!(x.weight(), 1);
    }

    #[test]
    fn test_tensor_product() {
        let ps = PauliString::x(0) & PauliString::z(1);
        assert_eq!(ps.get(0), Pauli::X);
        assert_eq!(ps.get(1), Pauli::Z);
        assert_eq!(ps.weight(), 2);
    }

    #[test]
    #[should_panic(expected = "tensor product requires disjoint Pauli support")]
    fn test_tensor_product_rejects_overlapping_qubits() {
        let _ = PauliString::x(0) & PauliString::z(0);
    }

    #[test]
    fn test_triple_tensor() {
        let ps = PauliString::x(0) & PauliString::y(2) & PauliString::z(5);
        assert_eq!(ps.get(0), Pauli::X);
        assert_eq!(ps.get(2), Pauli::Y);
        assert_eq!(ps.get(5), Pauli::Z);
        assert_eq!(ps.weight(), 3);
    }

    #[test]
    fn test_mul_same_qubit() {
        // X * Y = iZ
        let ps = PauliString::x(0) * PauliString::y(0);
        assert_eq!(ps.get(0), Pauli::Z);
        assert_eq!(ps.phase(), QuarterPhase::PlusI);
    }

    #[test]
    fn test_mul_self_inverse() {
        // X * X = I
        let ps = PauliString::x(0) * PauliString::x(0);
        assert_eq!(ps.weight(), 0); // Identity
        assert_eq!(ps.phase(), QuarterPhase::PlusOne);
    }

    #[test]
    fn test_mul_different_qubits() {
        // X(0) * Z(1) = X(0) tensor Z(1)
        let ps = PauliString::x(0) * PauliString::z(1);
        assert_eq!(ps.get(0), Pauli::X);
        assert_eq!(ps.get(1), Pauli::Z);
        assert_eq!(ps.weight(), 2);
    }

    #[test]
    fn test_mul_multi_qubit() {
        // (X(0) & Y(1)) * (X(0) & Z(1)) = I(0) & iX(1) = iX(1)
        let op1 = PauliString::x(0) & PauliString::y(1);
        let op2 = PauliString::x(0) & PauliString::z(1);
        let ps = op1 * op2;

        assert_eq!(ps.get(0), Pauli::I);
        assert_eq!(ps.get(1), Pauli::X);
        assert_eq!(ps.phase(), QuarterPhase::PlusI);
    }

    #[test]
    fn test_negation() {
        let ps = -PauliString::x(0);
        assert_eq!(ps.get(0), Pauli::X);
        assert_eq!(ps.phase(), QuarterPhase::MinusOne);
    }

    #[test]
    fn test_imaginary_phase() {
        let ps = i * PauliString::x(0);
        assert_eq!(ps.phase(), QuarterPhase::PlusI);

        let ps = -i * PauliString::x(0);
        assert_eq!(ps.phase(), QuarterPhase::MinusI);
    }

    #[test]
    fn test_complex_expression() {
        // -i * (X(0) & Y(2) & Z(5))
        let ps = -i * (PauliString::x(0) & PauliString::y(2) & PauliString::z(5));
        assert_eq!(ps.phase(), QuarterPhase::MinusI);
        assert_eq!(ps.get(0), Pauli::X);
        assert_eq!(ps.get(2), Pauli::Y);
        assert_eq!(ps.get(5), Pauli::Z);
        assert_eq!(ps.weight(), 3);
    }

    #[test]
    fn test_pauli_algebra_xy() {
        // X * Y = iZ
        let ps = PauliString::x(0) * PauliString::y(0);
        assert_eq!(ps.get(0), Pauli::Z);
        assert_eq!(ps.phase(), QuarterPhase::PlusI);
    }

    #[test]
    fn test_pauli_algebra_yx() {
        // Y * X = -iZ
        let ps = PauliString::y(0) * PauliString::x(0);
        assert_eq!(ps.get(0), Pauli::Z);
        assert_eq!(ps.phase(), QuarterPhase::MinusI);
    }

    #[test]
    fn test_pauli_algebra_yz() {
        // Y * Z = iX
        let ps = PauliString::y(0) * PauliString::z(0);
        assert_eq!(ps.get(0), Pauli::X);
        assert_eq!(ps.phase(), QuarterPhase::PlusI);
    }

    #[test]
    fn test_pauli_algebra_zx() {
        // Z * X = iY
        let ps = PauliString::z(0) * PauliString::x(0);
        assert_eq!(ps.get(0), Pauli::Y);
        assert_eq!(ps.phase(), QuarterPhase::PlusI);
    }

    #[test]
    fn test_anticommutes_with() {
        let x = PauliString::x(0);
        let z = PauliString::z(0);
        let z1 = PauliString::z(1);

        assert!(x.anticommutes_with(&z));
        assert!(!x.anticommutes_with(&z1));
        assert!(!x.anticommutes_with(&x));
    }

    // ========================================================================
    // Algebraic property tests for operator overloading
    // ========================================================================

    #[test]
    fn test_mul_associativity() {
        // (X * Y) * Z == X * (Y * Z) on same qubit
        let x = PauliString::x(0);
        let y = PauliString::y(0);
        let z = PauliString::z(0);

        let lhs = (x.clone() * y.clone()) * z.clone();
        let rhs = x * (y * z);
        assert_eq!(lhs, rhs);
    }

    #[test]
    fn test_mul_identity_neutral() {
        let x = PauliString::x(0);
        let id = PauliString::identity();

        let result = x.clone() * id.clone();
        assert_eq!(result, x.clone());

        let result = id * x.clone();
        assert_eq!(result, x);
    }

    #[test]
    fn test_double_negation() {
        let x = PauliString::x(0);
        let result = -(-x.clone());
        assert_eq!(result, x);
    }

    #[test]
    fn test_i_times_i_is_minus_one() {
        // i * (i * X) should equal -X
        let x = PauliString::x(0);
        let ix = i * x.clone();
        let iix = i * ix;
        let neg_x = -x;
        assert_eq!(iix, neg_x);
    }

    #[test]
    fn test_neg_i_times_i_is_plus_one() {
        // (-i) * (i * X) should equal X
        let x = PauliString::x(0);
        let ix = i * x.clone();
        let result = -i * ix;
        assert_eq!(result, x);
    }

    #[test]
    fn test_tensor_commutativity() {
        // X(0) & Z(1) == Z(1) & X(0) (tensor product is commutative for different qubits)
        let lhs = PauliString::x(0) & PauliString::z(1);
        let rhs = PauliString::z(1) & PauliString::x(0);
        // Both should represent the same operator
        assert_eq!(lhs.get(0), rhs.get(0));
        assert_eq!(lhs.get(1), rhs.get(1));
        assert_eq!(lhs.phase(), rhs.phase());
    }

    #[test]
    fn test_mul_reference_variants() {
        // Verify all reference combinations produce same result
        let x = PauliString::x(0);
        let y = PauliString::y(0);

        let result1 = x.clone() * y.clone();
        let result2 = &x * y.clone();
        let result3 = x.clone() * &y;
        let result4 = &x * &y;

        assert_eq!(result1, result2);
        assert_eq!(result1, result3);
        assert_eq!(result1, result4);
    }

    #[test]
    fn test_tensor_reference_variants() {
        let x = PauliString::x(0);
        let z = PauliString::z(1);

        let result1 = x.clone() & z.clone();
        let result2 = &x & z.clone();
        let result3 = x.clone() & &z;
        let result4 = &x & &z;

        assert_eq!(result1, result2);
        assert_eq!(result1, result3);
        assert_eq!(result1, result4);
    }

    #[test]
    fn test_neg_reference() {
        let x = PauliString::x(0);
        let result1 = -x.clone();
        let result2 = -&x;
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_i_reference() {
        let x = PauliString::x(0);
        let result1 = i * x.clone();
        let result2 = i * &x;
        assert_eq!(result1, result2);
    }

    #[test]
    fn test_neg_i_reference() {
        let x = PauliString::x(0);
        let result1 = -i * x.clone();
        let result2 = -i * &x;
        assert_eq!(result1, result2);
    }
}
