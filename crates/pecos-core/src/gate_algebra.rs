// Copyright 2026 The PECOS Developers
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

//! Cross-type algebraic operators for quantum gate primitives.
//!
//! Implements `*` (composition) and `&` (tensor product) between [`Pauli`],
//! [`Clifford`], and [`Unitary`] base types with automatic type promotion:
//!
//! | `*` / `&`      | Pauli         | Clifford      | Unitary       |
//! |----------------|---------------|---------------|---------------|
//! | **Pauli**      | `PauliString` | `CliffordRep` | `UnitaryRep`  |
//! | **Clifford**   | `CliffordRep` | `CliffordRep` | `UnitaryRep`  |
//! | **Unitary**    | `UnitaryRep`  | `UnitaryRep`  | `UnitaryRep`  |

use crate::clifford::Clifford;
use crate::clifford_rep::CliffordRep;
use crate::unitary_rep::{Unitary, UnitaryRep};
use crate::{Pauli, PauliString};
use std::ops::{BitAnd, Mul};

// ============================================================================
// Pauli: embedding helpers
// ============================================================================

impl Pauli {
    /// Embeds this Pauli on a specific qubit, returning a `PauliString`.
    #[must_use]
    pub fn on_qubit(self, qubit: usize) -> PauliString {
        match self {
            Pauli::I => PauliString::new(),
            Pauli::X => PauliString::x(qubit),
            Pauli::Y => PauliString::y(qubit),
            Pauli::Z => PauliString::z(qubit),
        }
    }
}

// ============================================================================
// Pauli * Pauli -> PauliString
// ============================================================================

impl Mul for Pauli {
    type Output = PauliString;

    /// Composes two Paulis on qubit 0, producing a `PauliString`.
    fn mul(self, rhs: Pauli) -> PauliString {
        self.on_qubit(0) * rhs.on_qubit(0)
    }
}

// ============================================================================
// Pauli & Pauli -> PauliString
// ============================================================================

impl BitAnd for Pauli {
    type Output = PauliString;

    /// Tensor product of two Paulis on consecutive qubits.
    fn bitand(self, rhs: Pauli) -> PauliString {
        self.on_qubit(0) & rhs.on_qubit(1)
    }
}

// ============================================================================
// Clifford: embedding to UnitaryRep
// ============================================================================

impl Clifford {
    /// Embeds this Clifford gate on default qubits as a `UnitaryRep`.
    fn to_unitary_rep_default(self) -> UnitaryRep {
        if self.is_1q() {
            self.to_unitary_rep_on_qubit(0)
        } else {
            self.to_unitary_rep_on_qubits(0, 1)
        }
    }

    /// Embeds this Clifford gate on qubits offset by `n` as a `UnitaryRep`.
    fn to_unitary_rep_offset(self, offset: usize) -> UnitaryRep {
        if self.is_1q() {
            self.to_unitary_rep_on_qubit(offset)
        } else {
            self.to_unitary_rep_on_qubits(offset, offset + 1)
        }
    }
}

// ============================================================================
// Clifford & Clifford -> CliffordRep
// ============================================================================

impl BitAnd for Clifford {
    type Output = CliffordRep;

    /// Tensor product of two Cliffords on consecutive qubits.
    fn bitand(self, rhs: Clifford) -> CliffordRep {
        let lhs_nq = self.num_qubits();
        let total_nq = lhs_nq + rhs.num_qubits();

        let lhs_rep = if self.is_1q() {
            self.on_qubit(0)
        } else {
            self.on_qubits(0, 1)
        };

        let rhs_rep = if rhs.is_1q() {
            rhs.on_qubit(lhs_nq)
        } else {
            rhs.on_qubits(lhs_nq, lhs_nq + 1)
        };

        // Extend both to the combined qubit count and compose.
        // Since they act on disjoint qubits, composition = tensor product.
        lhs_rep
            .extended_to(total_nq)
            .compose(&rhs_rep.extended_to(total_nq))
    }
}

// ============================================================================
// Pauli <-> Clifford cross-type operators
// ============================================================================

impl Mul<Clifford> for Pauli {
    type Output = CliffordRep;

    /// Compose Pauli (applied second) with Clifford (applied first) on qubit 0.
    fn mul(self, rhs: Clifford) -> CliffordRep {
        let total_nq = rhs.num_qubits();
        let lhs_rep = CliffordRep::from(self.on_qubit(0)).extended_to(total_nq);
        let rhs_rep = if rhs.is_1q() {
            rhs.on_qubit(0)
        } else {
            rhs.on_qubits(0, 1)
        };
        lhs_rep.compose(&rhs_rep)
    }
}

impl Mul<Pauli> for Clifford {
    type Output = CliffordRep;

    /// Compose Clifford (applied second) with Pauli (applied first) on qubit 0.
    fn mul(self, rhs: Pauli) -> CliffordRep {
        let total_nq = self.num_qubits();
        let lhs_rep = if self.is_1q() {
            self.on_qubit(0)
        } else {
            self.on_qubits(0, 1)
        };
        let rhs_rep = CliffordRep::from(rhs.on_qubit(0)).extended_to(total_nq);
        lhs_rep.compose(&rhs_rep)
    }
}

impl BitAnd<Clifford> for Pauli {
    type Output = CliffordRep;

    /// Tensor product: Pauli on qubit 0, Clifford on subsequent qubits.
    fn bitand(self, rhs: Clifford) -> CliffordRep {
        let rhs_offset = 1; // Pauli is always 1 qubit
        let total_nq = 1 + rhs.num_qubits();
        let lhs_rep = CliffordRep::from(self.on_qubit(0)).extended_to(total_nq);
        let rhs_rep = if rhs.is_1q() {
            rhs.on_qubit(rhs_offset)
        } else {
            rhs.on_qubits(rhs_offset, rhs_offset + 1)
        };
        lhs_rep.compose(&rhs_rep.extended_to(total_nq))
    }
}

#[allow(clippy::suspicious_arithmetic_impl)]
impl BitAnd<Pauli> for Clifford {
    type Output = CliffordRep;

    /// Tensor product: Clifford on initial qubits, Pauli on next qubit.
    fn bitand(self, rhs: Pauli) -> CliffordRep {
        let rhs_offset = self.num_qubits();
        let total_nq = rhs_offset + 1;
        let lhs_rep = if self.is_1q() {
            self.on_qubit(0)
        } else {
            self.on_qubits(0, 1)
        };
        let rhs_rep = CliffordRep::from(rhs.on_qubit(rhs_offset)).extended_to(total_nq);
        lhs_rep.extended_to(total_nq).compose(&rhs_rep)
    }
}

// ============================================================================
// Pauli <-> Unitary cross-type operators
// ============================================================================

impl Mul<Unitary> for Pauli {
    type Output = UnitaryRep;

    fn mul(self, rhs: Unitary) -> UnitaryRep {
        UnitaryRep::from(self.on_qubit(0)) * rhs.on_default_qubits()
    }
}

impl Mul<Pauli> for Unitary {
    type Output = UnitaryRep;

    fn mul(self, rhs: Pauli) -> UnitaryRep {
        self.on_default_qubits() * UnitaryRep::from(rhs.on_qubit(0))
    }
}

impl BitAnd<Unitary> for Pauli {
    type Output = UnitaryRep;

    fn bitand(self, rhs: Unitary) -> UnitaryRep {
        UnitaryRep::from(self.on_qubit(0)) & rhs.on_default_qubits_offset(1)
    }
}

impl BitAnd<Pauli> for Unitary {
    type Output = UnitaryRep;

    fn bitand(self, rhs: Pauli) -> UnitaryRep {
        let offset = self.num_qubits();
        self.on_default_qubits() & UnitaryRep::from(rhs.on_qubit(offset))
    }
}

// ============================================================================
// Clifford <-> Unitary cross-type operators
// ============================================================================

impl Mul<Unitary> for Clifford {
    type Output = UnitaryRep;

    fn mul(self, rhs: Unitary) -> UnitaryRep {
        self.to_unitary_rep_default() * rhs.on_default_qubits()
    }
}

impl Mul<Clifford> for Unitary {
    type Output = UnitaryRep;

    fn mul(self, rhs: Clifford) -> UnitaryRep {
        self.on_default_qubits() * rhs.to_unitary_rep_default()
    }
}

impl BitAnd<Unitary> for Clifford {
    type Output = UnitaryRep;

    fn bitand(self, rhs: Unitary) -> UnitaryRep {
        let offset = self.num_qubits();
        self.to_unitary_rep_default() & rhs.on_default_qubits_offset(offset)
    }
}

impl BitAnd<Clifford> for Unitary {
    type Output = UnitaryRep;

    fn bitand(self, rhs: Clifford) -> UnitaryRep {
        let offset = self.num_qubits();
        self.on_default_qubits() & rhs.to_unitary_rep_offset(offset)
    }
}

// ============================================================================
// Rep-level cross-type operators
// ============================================================================
//
// | `*` / `&`      | PauliString   | CliffordRep   | UnitaryRep    |
// |----------------|---------------|---------------|---------------|
// | **PauliString** | PauliString   | CliffordRep   | UnitaryRep    |
// | **CliffordRep** | CliffordRep   | CliffordRep   | UnitaryRep    |
// | **UnitaryRep**  | UnitaryRep    | UnitaryRep    | UnitaryRep    |
//
// Same-type ops already exist (PauliString*PauliString, CliffordRep*CliffordRep,
// UnitaryRep*UnitaryRep, etc.). Below are the cross-type promotions.

// --- PauliString <-> CliffordRep ---

impl Mul<CliffordRep> for PauliString {
    type Output = CliffordRep;
    fn mul(self, rhs: CliffordRep) -> CliffordRep {
        CliffordRep::from(self) * rhs
    }
}

impl Mul<PauliString> for CliffordRep {
    type Output = CliffordRep;
    fn mul(self, rhs: PauliString) -> CliffordRep {
        self * CliffordRep::from(rhs)
    }
}

impl BitAnd<CliffordRep> for PauliString {
    type Output = CliffordRep;
    fn bitand(self, rhs: CliffordRep) -> CliffordRep {
        CliffordRep::from(self) & rhs
    }
}

impl BitAnd<PauliString> for CliffordRep {
    type Output = CliffordRep;
    fn bitand(self, rhs: PauliString) -> CliffordRep {
        self & CliffordRep::from(rhs)
    }
}

// --- PauliString <-> UnitaryRep ---

impl Mul<UnitaryRep> for PauliString {
    type Output = UnitaryRep;
    fn mul(self, rhs: UnitaryRep) -> UnitaryRep {
        UnitaryRep::from(self) * rhs
    }
}

impl Mul<PauliString> for UnitaryRep {
    type Output = UnitaryRep;
    fn mul(self, rhs: PauliString) -> UnitaryRep {
        self * UnitaryRep::from(rhs)
    }
}

impl BitAnd<UnitaryRep> for PauliString {
    type Output = UnitaryRep;
    fn bitand(self, rhs: UnitaryRep) -> UnitaryRep {
        UnitaryRep::from(self) & rhs
    }
}

impl BitAnd<PauliString> for UnitaryRep {
    type Output = UnitaryRep;
    fn bitand(self, rhs: PauliString) -> UnitaryRep {
        self & UnitaryRep::from(rhs)
    }
}

// --- CliffordRep <-> UnitaryRep ---

impl Mul<UnitaryRep> for CliffordRep {
    type Output = UnitaryRep;
    fn mul(self, rhs: UnitaryRep) -> UnitaryRep {
        // Convert CliffordRep stabilizer images to UnitaryRep
        // by reconstructing from the generator images
        clifford_rep_to_unitary_rep(&self) * rhs
    }
}

impl Mul<CliffordRep> for UnitaryRep {
    type Output = UnitaryRep;
    fn mul(self, rhs: CliffordRep) -> UnitaryRep {
        self * clifford_rep_to_unitary_rep(&rhs)
    }
}

impl BitAnd<UnitaryRep> for CliffordRep {
    type Output = UnitaryRep;
    fn bitand(self, rhs: UnitaryRep) -> UnitaryRep {
        clifford_rep_to_unitary_rep(&self) & rhs
    }
}

impl BitAnd<CliffordRep> for UnitaryRep {
    type Output = UnitaryRep;
    fn bitand(self, rhs: CliffordRep) -> UnitaryRep {
        self & clifford_rep_to_unitary_rep(&rhs)
    }
}

/// Convert a [`CliffordRep`] to a [`UnitaryRep`] by matching against known
/// Clifford gates.
///
/// Tries to identify the `CliffordRep` as a tensor product of single-qubit
/// Cliffords first, then falls back to matching known 2-qubit gates.
/// Works for all 38 Clifford variants, including those without `GateType`
/// entries.
///
/// # Panics
///
/// Panics if the `CliffordRep` cannot be matched to any known gate
/// decomposition (e.g. Cliffords on 3+ qubits that are not a tensor
/// product of 1q and 2q gates).
#[must_use]
pub fn clifford_rep_to_unitary_rep(cr: &CliffordRep) -> UnitaryRep {
    use crate::clifford::Clifford;

    let nq = cr.num_qubits();
    if nq == 0 {
        return crate::unitary_rep::I(0);
    }

    // Try matching as a tensor product of 1q Cliffords
    let mut parts: Vec<UnitaryRep> = Vec::new();
    let mut all_matched = true;

    for q in 0..nq {
        let mut found = false;
        for &cliff in Clifford::all_1q() {
            let cliff_rep = cliff.on_qubit(q).extended_to(nq);
            if *cliff_rep.x_image(q) == *cr.x_image(q) && *cliff_rep.z_image(q) == *cr.z_image(q) {
                parts.push(cliff.to_unitary_rep_on_qubit(q));
                found = true;
                break;
            }
        }
        if !found {
            all_matched = false;
            break;
        }
    }

    if all_matched && !parts.is_empty() {
        return parts
            .into_iter()
            .reduce(|a, b| a & b)
            .expect("parts is non-empty");
    }

    // Fallback: try known 2q Cliffords
    if nq >= 2 {
        for q0 in 0..nq {
            for q1 in (q0 + 1)..nq {
                for &cliff in Clifford::all_2q() {
                    let cliff_rep = cliff.on_qubits(q0, q1).extended_to(nq);
                    if cliff_rep == *cr {
                        return cliff.to_unitary_rep_on_qubits(q0, q1);
                    }
                }
            }
        }
    }

    panic!(
        "Cannot convert {nq}-qubit CliffordRep to UnitaryRep: \
         no matching gate decomposition found"
    );
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::QuarterPhase;
    use crate::pauli::PauliOperator;

    // Pauli * Pauli -> PauliString
    #[test]
    fn pauli_mul_same() {
        let result = Pauli::X * Pauli::X;
        // X * X = I
        assert_eq!(result.weight(), 0);
    }

    #[test]
    fn pauli_mul_different() {
        let result = Pauli::X * Pauli::Z;
        // X * Z = -iY (on qubit 0)
        assert_eq!(result.get(0), Pauli::Y);
        assert_eq!(result.phase(), QuarterPhase::MinusI);
    }

    #[test]
    fn pauli_mul_phase_yz() {
        let result = Pauli::Y * Pauli::Z;
        // Y * Z = iX
        assert_eq!(result.get(0), Pauli::X);
        assert_eq!(result.phase(), QuarterPhase::PlusI);
    }

    #[test]
    fn pauli_mul_identity() {
        let result = Pauli::I * Pauli::X;
        assert_eq!(result.get(0), Pauli::X);
        assert_eq!(result.phase(), QuarterPhase::PlusOne);
    }

    // Pauli & Pauli -> PauliString
    #[test]
    fn pauli_tensor() {
        let result = Pauli::X & Pauli::Z;
        assert_eq!(result.get(0), Pauli::X);
        assert_eq!(result.get(1), Pauli::Z);
        assert_eq!(result.phase(), QuarterPhase::PlusOne);
    }

    #[test]
    fn pauli_tensor_identity() {
        let result = Pauli::I & Pauli::X;
        // I on q0 is not stored, X on q1
        assert_eq!(result.get(0), Pauli::I);
        assert_eq!(result.get(1), Pauli::X);
    }

    // Clifford & Clifford -> CliffordRep
    #[test]
    fn clifford_tensor_1q() {
        let result = Clifford::H & Clifford::SZ;
        // H on q0, SZ on q1 -> 2-qubit CliffordRep
        assert_eq!(result.num_qubits(), 2);
        assert!(result.is_valid());
    }

    #[test]
    fn clifford_tensor_2q_1q() {
        let result = Clifford::CX & Clifford::H;
        // CX on q0,q1, H on q2 -> 3-qubit CliffordRep
        assert_eq!(result.num_qubits(), 3);
        assert!(result.is_valid());
    }

    #[test]
    fn clifford_tensor_2q_2q() {
        let result = Clifford::CX & Clifford::CZ;
        // CX on q0,q1, CZ on q2,q3 -> 4-qubit CliffordRep
        assert_eq!(result.num_qubits(), 4);
        assert!(result.is_valid());
    }

    #[test]
    fn clifford_tensor_stabilizer_correctness() {
        // H & SZ: H on q0, SZ on q1
        // X0 -> Z0, Z0 -> X0 (from H)
        // X1 -> Y1, Z1 -> Z1 (from SZ)
        let result = Clifford::H & Clifford::SZ;

        let x0 = PauliString::x(0);
        let z1 = PauliString::z(1);

        let tx0 = result.apply(&x0);
        assert_eq!(tx0.get(0), Pauli::Z);

        let tz1 = result.apply(&z1);
        assert_eq!(tz1.get(1), Pauli::Z);
    }

    #[test]
    fn pauli_tensor_clifford_stabilizer_correctness() {
        // X & H: X on q0 (Pauli), H on q1 (Clifford)
        // X0 -> X0 (X gate is identity on X), Z0 -> -Z0 (X gate negates Z)
        // X1 -> Z1 (from H), Z1 -> X1 (from H)
        let result = Pauli::X & Clifford::H;

        let z0 = PauliString::z(0);
        let tz0 = result.apply(&z0);
        assert_eq!(tz0.get(0), Pauli::Z);
        assert_eq!(tz0.phase(), QuarterPhase::MinusOne);

        let x1 = PauliString::x(1);
        let tx1 = result.apply(&x1);
        assert_eq!(tx1.get(1), Pauli::Z);
        assert_eq!(tx1.phase(), QuarterPhase::PlusOne);
    }

    // Pauli * Clifford -> CliffordRep
    #[test]
    fn pauli_mul_clifford() {
        let result = Pauli::X * Clifford::H;
        // Apply H first, then X, on qubit 0
        assert_eq!(result.num_qubits(), 1);
        assert!(result.is_valid());
    }

    #[test]
    fn clifford_mul_pauli() {
        let result = Clifford::H * Pauli::X;
        // Apply X first, then H, on qubit 0
        assert_eq!(result.num_qubits(), 1);
        assert!(result.is_valid());
    }

    #[test]
    fn composition_order_matters() {
        // X * H: apply H first, then X
        // H * X: apply X first, then H
        // These should differ since H and X don't commute in general
        let xh = Pauli::X * Clifford::H;
        let hx = Clifford::H * Pauli::X;
        assert_ne!(xh, hx);
    }

    #[test]
    fn pauli_mul_clifford_stabilizer_correctness() {
        // X * H on qubit 0:
        //   H sends X->Z, Z->X
        //   X sends X->X, Z->-Z
        //   Combined (X * H): X -> X(H(X)) = X(Z) = -Z, Z -> X(H(Z)) = X(X) = X
        let result = Pauli::X * Clifford::H;
        let z0 = PauliString::z(0);
        let x0 = PauliString::x(0);

        let tz = result.apply(&z0);
        assert_eq!(tz.get(0), Pauli::X);
        assert_eq!(tz.phase(), QuarterPhase::PlusOne);

        let tx = result.apply(&x0);
        assert_eq!(tx.get(0), Pauli::Z);
        assert_eq!(tx.phase(), QuarterPhase::MinusOne);
    }

    #[test]
    fn identity_mul_clifford() {
        // I * H should equal H
        let result = Pauli::I * Clifford::H;
        let just_h = Clifford::H.on_qubit(0);
        assert_eq!(result, just_h);
    }

    // Pauli & Clifford -> CliffordRep
    #[test]
    fn pauli_tensor_clifford() {
        let result = Pauli::X & Clifford::H;
        // X on q0, H on q1
        assert_eq!(result.num_qubits(), 2);
        assert!(result.is_valid());
    }

    #[test]
    fn clifford_tensor_pauli() {
        let result = Clifford::CX & Pauli::Z;
        // CX on q0,q1, Z on q2
        assert_eq!(result.num_qubits(), 3);
        assert!(result.is_valid());
    }

    // Pauli <-> Unitary -> UnitaryRep
    #[test]
    fn pauli_mul_unitary() {
        let result = Pauli::X * Unitary::Named(crate::gate_type::GateType::H);
        assert_eq!(result.qubits(), vec![0]);
    }

    #[test]
    fn unitary_mul_pauli() {
        let result = Unitary::Named(crate::gate_type::GateType::H) * Pauli::X;
        assert_eq!(result.qubits(), vec![0]);
    }

    #[test]
    fn pauli_tensor_unitary() {
        let result = Pauli::X & Unitary::Named(crate::gate_type::GateType::H);
        let qubits = result.qubits();
        assert!(qubits.contains(&0));
        assert!(qubits.contains(&1));
    }

    #[test]
    fn unitary_tensor_pauli() {
        let result = Unitary::Named(crate::gate_type::GateType::CX) & Pauli::Z;
        let qubits = result.qubits();
        assert_eq!(qubits, vec![0, 1, 2]);
    }

    // Clifford <-> Unitary -> UnitaryRep
    #[test]
    fn clifford_mul_unitary() {
        use crate::Angle64;
        use crate::unitary_rep::RotationType;
        let rz = Unitary::Rotation {
            rotation_type: RotationType::RZ,
            angle: Angle64::from_turn_ratio(1, 8),
        };
        let result = Clifford::H * rz;
        assert_eq!(result.qubits(), vec![0]);
    }

    #[test]
    fn unitary_mul_clifford() {
        use crate::Angle64;
        use crate::unitary_rep::RotationType;
        let rz = Unitary::Rotation {
            rotation_type: RotationType::RZ,
            angle: Angle64::from_turn_ratio(1, 8),
        };
        let result = rz * Clifford::H;
        assert_eq!(result.qubits(), vec![0]);
    }

    #[test]
    fn clifford_tensor_unitary() {
        use crate::Angle64;
        use crate::unitary_rep::RotationType;
        let rz = Unitary::Rotation {
            rotation_type: RotationType::RZ,
            angle: Angle64::from_turn_ratio(1, 8),
        };
        let result = Clifford::H & rz;
        let qubits = result.qubits();
        assert!(qubits.contains(&0));
        assert!(qubits.contains(&1));
    }

    #[test]
    fn unitary_tensor_clifford() {
        let result = Unitary::Named(crate::gate_type::GateType::H) & Clifford::CX;
        let qubits = result.qubits();
        assert_eq!(qubits, vec![0, 1, 2]);
    }

    // Pauli::on_qubit
    #[test]
    fn pauli_on_qubit() {
        let ps = Pauli::X.on_qubit(5);
        assert_eq!(ps.get(5), Pauli::X);
        assert_eq!(ps.weight(), 1);

        let id = Pauli::I.on_qubit(3);
        assert_eq!(id.weight(), 0);
    }

    // Clifford::to_gate_type
    #[test]
    fn clifford_to_gate_type() {
        use crate::gate_type::GateType;
        assert_eq!(Clifford::H.to_gate_type(), Some(GateType::H));
        assert_eq!(Clifford::CX.to_gate_type(), Some(GateType::CX));
        assert_eq!(Clifford::SWAP.to_gate_type(), Some(GateType::SWAP));
        assert_eq!(Clifford::SZ.to_gate_type(), Some(GateType::SZ));
        // Variants without GateType return None
        assert_eq!(Clifford::H2.to_gate_type(), None);
        assert_eq!(Clifford::ISWAP.to_gate_type(), None);
        assert_eq!(Clifford::G.to_gate_type(), None);
    }

    #[test]
    fn clifford_rep_inverse_correctness_1q() {
        use crate::clifford_rep::CliffordRep;

        // SZ and SZdg should be inverses
        let sz = CliffordRep::sz(0);
        let szdg = CliffordRep::szdg(0);
        let sz_inv = sz.inverse();
        assert_eq!(sz_inv, szdg, "SZ.inverse() should equal SZdg");

        // SX and SXdg
        let sx = CliffordRep::sx(0);
        let sxdg = CliffordRep::sxdg(0);
        let sx_inv = sx.inverse();
        assert_eq!(sx_inv, sxdg, "SX.inverse() should equal SXdg");
    }

    #[test]
    fn clifford_rep_inverse_correctness_2q() {
        use crate::clifford_rep::CliffordRep;

        // SZZ and SZZdg
        let szz = CliffordRep::szz(0, 1);
        let szzdg = CliffordRep::szzdg(0, 1);
        assert_ne!(szz, szzdg, "SZZ and SZZdg should be distinct");

        // ISWAP and ISWAPdg differ in their X images (signs are flipped)
        let iswap = CliffordRep::iswap(0, 1);
        let iswapdg = CliffordRep::iswapdg(0, 1);
        assert_ne!(iswap, iswapdg, "ISWAP and ISWAPdg should be distinct");

        // G is self-inverse at the stabilizer level (G^2 = I for all generators)
        let g = CliffordRep::g(0, 1);
        let gdg = CliffordRep::gdg(0, 1);
        assert_eq!(g, gdg, "G is self-inverse at the stabilizer level");
    }
}
