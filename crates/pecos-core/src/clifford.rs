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

//! Named Clifford gate primitives.
//!
//! The base Clifford gates (single-qubit and two-qubit), analogous to [`Pauli`]
//! for the Pauli group. The 24 single-qubit elements form a closed group with
//! fast composition via lookup. Two-qubit gates are the standard entangling primitives.
//!
//! # Example
//!
//! ```
//! use pecos_core::clifford::Clifford;
//! use pecos_core::Pauli;
//! use pecos_core::Sign;
//!
//! let h = Clifford::H;
//! let (sign, p) = h.conjugate(Pauli::X);
//! assert_eq!(p, Pauli::Z);
//! assert_eq!(sign, Sign::PlusOne);
//!
//! // Two-qubit gates
//! let cx_rep = Clifford::CX.on_qubits(0, 1);
//! assert!(cx_rep.is_valid());
//! ```

use crate::clifford_rep::CliffordRep;
use crate::gate_type::GateType;
use crate::unitary_rep::UnitaryRep;
use crate::{Angle64, Pauli, QubitId, Sign};
use std::fmt;
use std::ops::Mul;

/// Named Clifford gate primitive.
///
/// Includes all 24 single-qubit Clifford gates and the standard two-qubit gates.
/// The single-qubit subset forms a closed group with [`compose`](Clifford::compose)
/// and [`inverse`](Clifford::inverse). Two-qubit gates support
/// [`inverse`](Clifford::inverse) and embedding via [`on_qubits`](Clifford::on_qubits).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
#[derive(Default)]
pub enum Clifford {
    // === Single-qubit gates (24 elements) ===

    // Identity and Paulis
    #[default]
    I = 0,
    X = 1,
    Y = 2,
    Z = 3,

    // Hadamard variants (involutions)
    H = 4,
    H2 = 5,
    H3 = 6,
    H4 = 7,
    H5 = 8,
    H6 = 9,

    // Square root gates and daggers
    SX = 10,
    SXdg = 11,
    SY = 12,
    SYdg = 13,
    SZ = 14,
    SZdg = 15,

    // Face gates
    F = 16,
    Fdg = 17,
    F2 = 18,
    F2dg = 19,
    F3 = 20,
    F3dg = 21,
    F4 = 22,
    F4dg = 23,

    // === Two-qubit gates ===
    CX = 24,
    CY = 25,
    CZ = 26,
    SWAP = 27,
    SXX = 28,
    SXXdg = 29,
    SYY = 30,
    SYYdg = 31,
    SZZ = 32,
    SZZdg = 33,
    ISWAP = 34,
    ISWAPdg = 35,
    G = 36,
    Gdg = 37,
}

/// All 24 single-qubit variants.
const ALL_1Q: [Clifford; 24] = [
    Clifford::I,
    Clifford::X,
    Clifford::Y,
    Clifford::Z,
    Clifford::H,
    Clifford::H2,
    Clifford::H3,
    Clifford::H4,
    Clifford::H5,
    Clifford::H6,
    Clifford::SX,
    Clifford::SXdg,
    Clifford::SY,
    Clifford::SYdg,
    Clifford::SZ,
    Clifford::SZdg,
    Clifford::F,
    Clifford::Fdg,
    Clifford::F2,
    Clifford::F2dg,
    Clifford::F3,
    Clifford::F3dg,
    Clifford::F4,
    Clifford::F4dg,
];

/// All 14 two-qubit variants.
const ALL_2Q: [Clifford; 14] = [
    Clifford::CX,
    Clifford::CY,
    Clifford::CZ,
    Clifford::SWAP,
    Clifford::SXX,
    Clifford::SXXdg,
    Clifford::SYY,
    Clifford::SYYdg,
    Clifford::SZZ,
    Clifford::SZZdg,
    Clifford::ISWAP,
    Clifford::ISWAPdg,
    Clifford::G,
    Clifford::Gdg,
];

/// All 38 variants.
const ALL_VARIANTS: [Clifford; 38] = [
    Clifford::I,
    Clifford::X,
    Clifford::Y,
    Clifford::Z,
    Clifford::H,
    Clifford::H2,
    Clifford::H3,
    Clifford::H4,
    Clifford::H5,
    Clifford::H6,
    Clifford::SX,
    Clifford::SXdg,
    Clifford::SY,
    Clifford::SYdg,
    Clifford::SZ,
    Clifford::SZdg,
    Clifford::F,
    Clifford::Fdg,
    Clifford::F2,
    Clifford::F2dg,
    Clifford::F3,
    Clifford::F3dg,
    Clifford::F4,
    Clifford::F4dg,
    Clifford::CX,
    Clifford::CY,
    Clifford::CZ,
    Clifford::SWAP,
    Clifford::SXX,
    Clifford::SXXdg,
    Clifford::SYY,
    Clifford::SYYdg,
    Clifford::SZZ,
    Clifford::SZZdg,
    Clifford::ISWAP,
    Clifford::ISWAPdg,
    Clifford::G,
    Clifford::Gdg,
];

/// Pauli image: (negated, `target_pauli`) where `target_pauli` is in {X, Y, Z}.
/// `negated = true` means the image is -P, `false` means +P.
type SignedPauli = (bool, Pauli);

/// Single-qubit Pauli images table (indexed by discriminant, only valid for 0..24).
/// Entry: (`neg_x`, `pauli_x`, `neg_z`, `pauli_z`)
const IMAGES_1Q: [(bool, Pauli, bool, Pauli); 24] = [
    //                     X image     Z image
    (false, Pauli::X, false, Pauli::Z), //  0: I
    (false, Pauli::X, true, Pauli::Z),  //  1: X
    (true, Pauli::X, true, Pauli::Z),   //  2: Y
    (true, Pauli::X, false, Pauli::Z),  //  3: Z
    (false, Pauli::Z, false, Pauli::X), //  4: H
    (true, Pauli::Z, true, Pauli::X),   //  5: H2
    (false, Pauli::Y, true, Pauli::Z),  //  6: H3
    (true, Pauli::Y, true, Pauli::Z),   //  7: H4
    (true, Pauli::X, false, Pauli::Y),  //  8: H5
    (true, Pauli::X, true, Pauli::Y),   //  9: H6
    (false, Pauli::X, true, Pauli::Y),  // 10: SX
    (false, Pauli::X, false, Pauli::Y), // 11: SXdg
    (true, Pauli::Z, false, Pauli::X),  // 12: SY
    (false, Pauli::Z, true, Pauli::X),  // 13: SYdg
    (false, Pauli::Y, false, Pauli::Z), // 14: SZ
    (true, Pauli::Y, false, Pauli::Z),  // 15: SZdg
    (false, Pauli::Y, false, Pauli::X), // 16: F
    (false, Pauli::Z, false, Pauli::Y), // 17: Fdg
    (true, Pauli::Z, false, Pauli::Y),  // 18: F2
    (true, Pauli::Y, true, Pauli::X),   // 19: F2dg
    (false, Pauli::Y, true, Pauli::X),  // 20: F3
    (true, Pauli::Z, true, Pauli::Y),   // 21: F3dg
    (false, Pauli::Z, true, Pauli::Y),  // 22: F4
    (true, Pauli::Y, false, Pauli::X),  // 23: F4dg
];

impl Clifford {
    /// Returns all 38 Clifford gate variants (24 single-qubit + 14 two-qubit).
    #[must_use]
    pub fn all() -> &'static [Clifford; 38] {
        &ALL_VARIANTS
    }

    /// Returns all 24 single-qubit Clifford gates.
    #[must_use]
    pub fn all_1q() -> &'static [Clifford; 24] {
        &ALL_1Q
    }

    /// Returns all 14 two-qubit Clifford gates.
    #[must_use]
    pub fn all_2q() -> &'static [Clifford; 14] {
        &ALL_2Q
    }

    /// Returns the number of qubits this gate acts on.
    #[must_use]
    pub fn num_qubits(self) -> usize {
        if (self as u8) < 24 { 1 } else { 2 }
    }

    /// Returns whether this is a single-qubit gate.
    #[must_use]
    pub fn is_1q(self) -> bool {
        self.num_qubits() == 1
    }

    /// Returns whether this is a two-qubit gate.
    #[must_use]
    pub fn is_2q(self) -> bool {
        self.num_qubits() == 2
    }

    /// Returns the corresponding `GateType` for this Clifford gate, if one exists.
    ///
    /// Returns `None` for Clifford variants that don't have a matching `GateType`
    /// (H2-H6, F2-F4 and daggers, ISWAP, G).
    #[must_use]
    pub fn to_gate_type(self) -> Option<GateType> {
        match self {
            Self::I => Some(GateType::I),
            Self::X => Some(GateType::X),
            Self::Y => Some(GateType::Y),
            Self::Z => Some(GateType::Z),
            Self::H => Some(GateType::H),
            Self::SX => Some(GateType::SX),
            Self::SXdg => Some(GateType::SXdg),
            Self::SY => Some(GateType::SY),
            Self::SYdg => Some(GateType::SYdg),
            Self::SZ => Some(GateType::SZ),
            Self::SZdg => Some(GateType::SZdg),
            Self::F => Some(GateType::F),
            Self::Fdg => Some(GateType::Fdg),
            Self::CX => Some(GateType::CX),
            Self::CY => Some(GateType::CY),
            Self::CZ => Some(GateType::CZ),
            Self::SWAP => Some(GateType::SWAP),
            Self::SXX => Some(GateType::SXX),
            Self::SXXdg => Some(GateType::SXXdg),
            Self::SYY => Some(GateType::SYY),
            Self::SYYdg => Some(GateType::SYYdg),
            Self::SZZ => Some(GateType::SZZ),
            Self::SZZdg => Some(GateType::SZZdg),
            // These Clifford variants don't have matching GateType entries yet
            Self::H2
            | Self::H3
            | Self::H4
            | Self::H5
            | Self::H6
            | Self::F2
            | Self::F2dg
            | Self::F3
            | Self::F3dg
            | Self::F4
            | Self::F4dg
            | Self::ISWAP
            | Self::ISWAPdg
            | Self::G
            | Self::Gdg => None,
        }
    }

    // ========================================================================
    // Single-qubit operations (only valid for 1q gates)
    // ========================================================================

    /// Returns how this single-qubit Clifford transforms the X generator.
    ///
    /// # Panics
    /// Panics if called on a two-qubit gate.
    #[must_use]
    pub fn x_image(self) -> (Sign, Pauli) {
        assert!(self.is_1q(), "x_image only valid for single-qubit gates");
        let (neg, p, _, _) = IMAGES_1Q[self as usize];
        (if neg { Sign::MinusOne } else { Sign::PlusOne }, p)
    }

    /// Returns how this single-qubit Clifford transforms the Z generator.
    ///
    /// # Panics
    /// Panics if called on a two-qubit gate.
    #[must_use]
    pub fn z_image(self) -> (Sign, Pauli) {
        assert!(self.is_1q(), "z_image only valid for single-qubit gates");
        let (_, _, neg, p) = IMAGES_1Q[self as usize];
        (if neg { Sign::MinusOne } else { Sign::PlusOne }, p)
    }

    /// Returns how this single-qubit Clifford transforms a given Pauli by conjugation.
    ///
    /// `C * P * C†` = `sign * result_pauli`
    ///
    /// # Panics
    /// Panics if called on a two-qubit gate.
    #[must_use]
    pub fn conjugate(self, p: Pauli) -> (Sign, Pauli) {
        assert!(self.is_1q(), "conjugate only valid for single-qubit gates");
        match p {
            Pauli::I => (Sign::PlusOne, Pauli::I),
            Pauli::X => self.x_image(),
            Pauli::Z => self.z_image(),
            Pauli::Y => self.y_image(),
        }
    }

    /// Composes two single-qubit Cliffords: `self * other`.
    ///
    /// The result represents applying `other` first, then `self`:
    /// `(self * other)(P) = self(other(P))`.
    ///
    /// Only works for the 24 single-qubit gates (which form a closed group).
    ///
    /// # Panics
    /// Panics if either operand is a two-qubit gate.
    #[must_use]
    pub fn compose(self, other: Clifford) -> Clifford {
        assert!(
            self.is_1q() && other.is_1q(),
            "compose only valid for single-qubit gates (got {self}, {other})"
        );
        let (onx, opx, onz, opz) = IMAGES_1Q[other as usize];
        let new_x = self.conjugate_signed(onx, opx);
        let new_z = self.conjugate_signed(onz, opz);
        Self::from_images(new_x, new_z)
    }

    // ========================================================================
    // Operations valid for all gates
    // ========================================================================

    /// Returns the inverse of this Clifford gate.
    #[must_use]
    pub fn inverse(self) -> Clifford {
        if self.is_2q() {
            return self.inverse_2q();
        }
        let (nx, px, nz, pz) = IMAGES_1Q[self as usize];
        let (ny, py) = self.y_image_raw();

        let mut inv_map_x = (false, Pauli::I);
        let mut inv_map_z = (false, Pauli::I);

        set_inverse_entry(px, nx, Pauli::X, &mut inv_map_x, &mut inv_map_z);
        set_inverse_entry(pz, nz, Pauli::Z, &mut inv_map_x, &mut inv_map_z);
        set_inverse_entry(py, ny, Pauli::Y, &mut inv_map_x, &mut inv_map_z);

        Self::from_images(inv_map_x, inv_map_z)
    }

    /// Embeds a single-qubit gate on a specific qubit, returning a `CliffordRep`.
    ///
    /// # Panics
    /// Panics if called on a two-qubit gate. Use [`on_qubits`](Clifford::on_qubits) instead.
    #[must_use]
    pub fn on_qubit(self, qubit: usize) -> CliffordRep {
        match self {
            Clifford::I => CliffordRep::identity(qubit + 1),
            Clifford::X => CliffordRep::x(qubit),
            Clifford::Y => CliffordRep::y(qubit),
            Clifford::Z => CliffordRep::z(qubit),
            Clifford::H => CliffordRep::h(qubit),
            Clifford::H2 => CliffordRep::h2(qubit),
            Clifford::H3 => CliffordRep::h3(qubit),
            Clifford::H4 => CliffordRep::h4(qubit),
            Clifford::H5 => CliffordRep::h5(qubit),
            Clifford::H6 => CliffordRep::h6(qubit),
            Clifford::SX => CliffordRep::sx(qubit),
            Clifford::SXdg => CliffordRep::sxdg(qubit),
            Clifford::SY => CliffordRep::sy(qubit),
            Clifford::SYdg => CliffordRep::sydg(qubit),
            Clifford::SZ => CliffordRep::sz(qubit),
            Clifford::SZdg => CliffordRep::szdg(qubit),
            Clifford::F => CliffordRep::f(qubit),
            Clifford::Fdg => CliffordRep::fdg(qubit),
            Clifford::F2 => CliffordRep::f2(qubit),
            Clifford::F2dg => CliffordRep::f2dg(qubit),
            Clifford::F3 => CliffordRep::f3(qubit),
            Clifford::F3dg => CliffordRep::f3dg(qubit),
            Clifford::F4 => CliffordRep::f4(qubit),
            Clifford::F4dg => CliffordRep::f4dg(qubit),
            _ => panic!("on_qubit called on two-qubit gate {self}; use on_qubits instead"),
        }
    }

    /// Embeds a two-qubit gate on specific qubits, returning a `CliffordRep`.
    ///
    /// For single-qubit gates, the second qubit is ignored and
    /// [`on_qubit`](Clifford::on_qubit) is preferred.
    #[must_use]
    pub fn on_qubits(self, q0: usize, q1: usize) -> CliffordRep {
        match self {
            // Single-qubit gates: use q0, ignore q1
            g if g.is_1q() => g.on_qubit(q0),

            // Two-qubit gates
            Clifford::CX => CliffordRep::cx(q0, q1),
            Clifford::CY => CliffordRep::cy(q0, q1),
            Clifford::CZ => CliffordRep::cz(q0, q1),
            Clifford::SWAP => CliffordRep::swap(q0, q1),
            Clifford::SXX => CliffordRep::sxx(q0, q1),
            Clifford::SXXdg => CliffordRep::sxxdg(q0, q1),
            Clifford::SYY => CliffordRep::syy(q0, q1),
            Clifford::SYYdg => CliffordRep::syydg(q0, q1),
            Clifford::SZZ => CliffordRep::szz(q0, q1),
            Clifford::SZZdg => CliffordRep::szzdg(q0, q1),
            Clifford::ISWAP => CliffordRep::iswap(q0, q1),
            Clifford::ISWAPdg => CliffordRep::iswapdg(q0, q1),
            Clifford::G => CliffordRep::g(q0, q1),
            Clifford::Gdg => CliffordRep::gdg(q0, q1),
            _ => unreachable!(),
        }
    }

    /// Returns a [`UnitaryRep`] expression for this single-qubit Clifford on a given qubit.
    ///
    /// This provides a direct decomposition for ALL 24 single-qubit Clifford variants,
    /// including those without a `GateType` entry (H2-H6, F2-F4dg).
    ///
    /// # Panics
    /// Panics if called on a two-qubit gate.
    #[must_use]
    pub fn to_unitary_rep_on_qubit(self, q: impl Into<QubitId>) -> UnitaryRep {
        assert!(
            self.is_1q(),
            "to_unitary_rep_on_qubit called on two-qubit gate {self}"
        );
        let q = q.into();
        use crate::unitary_rep;
        match self {
            // Paulis and identity
            Clifford::I => unitary_rep::I(q),
            Clifford::X => unitary_rep::X(q),
            Clifford::Y => unitary_rep::Y(q),
            Clifford::Z => unitary_rep::Z(q),
            // Standard Cliffords
            Clifford::H => unitary_rep::H(q),
            Clifford::SX => unitary_rep::SX(q),
            Clifford::SXdg => unitary_rep::SX(q).dg(),
            Clifford::SY => unitary_rep::SY(q),
            Clifford::SYdg => unitary_rep::SY(q).dg(),
            Clifford::SZ => unitary_rep::SZ(q),
            Clifford::SZdg => unitary_rep::SZ(q).dg(),
            // Hadamard variants (A * B means "apply B first, then A")
            Clifford::H2 => unitary_rep::Z(q) * unitary_rep::SY(q),
            Clifford::H3 => unitary_rep::Y(q) * unitary_rep::SZ(q),
            Clifford::H4 => unitary_rep::X(q) * unitary_rep::SZ(q),
            Clifford::H5 => unitary_rep::Z(q) * unitary_rep::SX(q),
            Clifford::H6 => unitary_rep::Y(q) * unitary_rep::SX(q),
            // Face gates
            Clifford::F => unitary_rep::SZ(q) * unitary_rep::SX(q),
            Clifford::Fdg => unitary_rep::SX(q).dg() * unitary_rep::SZ(q).dg(),
            Clifford::F2 => unitary_rep::SY(q) * unitary_rep::SX(q).dg(),
            Clifford::F2dg => unitary_rep::SX(q) * unitary_rep::SY(q).dg(),
            Clifford::F3 => unitary_rep::SZ(q) * unitary_rep::SX(q).dg(),
            Clifford::F3dg => unitary_rep::SX(q) * unitary_rep::SZ(q).dg(),
            Clifford::F4 => unitary_rep::SX(q) * unitary_rep::SZ(q),
            Clifford::F4dg => unitary_rep::SZ(q).dg() * unitary_rep::SX(q).dg(),
            _ => unreachable!(),
        }
    }

    /// Returns a [`UnitaryRep`] expression for this two-qubit Clifford on given qubits.
    ///
    /// This provides a direct decomposition for ALL 14 two-qubit Clifford variants,
    /// including those without a `GateType` entry (ISWAP/ISWAPdg, G/Gdg).
    ///
    /// # Panics
    /// Panics if called on a single-qubit gate.
    #[must_use]
    pub fn to_unitary_rep_on_qubits(
        self,
        q0: impl Into<QubitId>,
        q1: impl Into<QubitId>,
    ) -> UnitaryRep {
        assert!(
            self.is_2q(),
            "to_unitary_rep_on_qubits called on single-qubit gate {self}"
        );
        let a = q0.into();
        let b = q1.into();
        use crate::unitary_rep;
        match self {
            Clifford::CX => unitary_rep::CX(a, b),
            Clifford::CY => unitary_rep::CY(a, b),
            Clifford::CZ => unitary_rep::CZ(a, b),
            Clifford::SWAP => unitary_rep::SWAP(a, b),
            Clifford::SXX => unitary_rep::RXX(Angle64::QUARTER_TURN, a, b),
            Clifford::SXXdg => unitary_rep::RXX(Angle64::THREE_QUARTERS_TURN, a, b),
            Clifford::SYY => unitary_rep::RYY(Angle64::QUARTER_TURN, a, b),
            Clifford::SYYdg => unitary_rep::RYY(Angle64::THREE_QUARTERS_TURN, a, b),
            Clifford::SZZ => unitary_rep::SZZ(a, b),
            Clifford::SZZdg => unitary_rep::SZZ(a, b).dg(),
            // iSWAP = exp(+i*pi/4*(XX+YY)) = RXX(-pi/2) * RYY(-pi/2)
            // where RXX(theta) = exp(-i*theta/2*XX), so -pi/2 = THREE_QUARTERS_TURN
            Clifford::ISWAP => {
                unitary_rep::RXX(Angle64::THREE_QUARTERS_TURN, a, b)
                    * unitary_rep::RYY(Angle64::THREE_QUARTERS_TURN, a, b)
            }
            Clifford::ISWAPdg => (unitary_rep::RXX(Angle64::THREE_QUARTERS_TURN, a, b)
                * unitary_rep::RYY(Angle64::THREE_QUARTERS_TURN, a, b))
            .dg(),
            // G = CZ * H(q0) * H(q1) * CZ (apply CZ, then H on both, then CZ)
            Clifford::G => {
                unitary_rep::CZ(a, b)
                    * unitary_rep::H(a)
                    * unitary_rep::H(b)
                    * unitary_rep::CZ(a, b)
            }
            Clifford::Gdg => (unitary_rep::CZ(a, b)
                * unitary_rep::H(a)
                * unitary_rep::H(b)
                * unitary_rep::CZ(a, b))
            .dg(),
            _ => unreachable!(),
        }
    }

    /// Attempts to identify which single-qubit Clifford a 1-qubit `CliffordRep` represents.
    ///
    /// Returns `None` if the `CliffordRep` is not a single-qubit Clifford or
    /// doesn't match any of the 24 elements.
    #[must_use]
    pub fn from_clifford_rep(rep: &CliffordRep) -> Option<Clifford> {
        if rep.num_qubits() != 1 {
            return None;
        }

        let x_img = rep.x_image(0);
        let z_img = rep.z_image(0);

        let x_pauli = x_img.get(0);
        let x_neg = x_img.phase() == crate::QuarterPhase::MinusOne;
        let z_pauli = z_img.get(0);
        let z_neg = z_img.phase() == crate::QuarterPhase::MinusOne;

        for &variant in &ALL_1Q {
            let (enx, epx, enz, epz) = IMAGES_1Q[variant as usize];
            if enx == x_neg && epx == x_pauli && enz == z_neg && epz == z_pauli {
                return Some(variant);
            }
        }
        None
    }

    // ========================================================================
    // Internal helpers
    // ========================================================================

    /// Compute the Y image from X and Z images (1q only).
    fn y_image(self) -> (Sign, Pauli) {
        let (neg, pauli) = self.y_image_raw();
        (if neg { Sign::MinusOne } else { Sign::PlusOne }, pauli)
    }

    /// Raw Y image as (negated, Pauli). Only valid for 1q gates.
    fn y_image_raw(self) -> SignedPauli {
        let (nx, px, nz, pz) = IMAGES_1Q[self as usize];
        // C(Y) = i * ((-1)^nx * Px) * ((-1)^nz * Pz)
        //      = i * (-1)^(nx+nz) * Px * Pz
        // For distinct non-identity Paulis: Px * Pz = (-1)^epsilon * i * Pr
        // So C(Y) = (-1)^(nx+nz+epsilon) * i^2 * Pr = (-1)^(1+nx+nz+epsilon) * Pr
        let (epsilon, pr) = pauli_product_sign(px, pz);
        let neg_y = !(nx ^ nz ^ epsilon);
        (neg_y, pr)
    }

    /// Apply this 1q Clifford to a signed Pauli.
    fn conjugate_signed(self, neg: bool, p: Pauli) -> SignedPauli {
        match p {
            Pauli::I => (neg, Pauli::I),
            Pauli::X => {
                let (nx, px, _, _) = IMAGES_1Q[self as usize];
                (neg ^ nx, px)
            }
            Pauli::Z => {
                let (_, _, nz, pz) = IMAGES_1Q[self as usize];
                (neg ^ nz, pz)
            }
            Pauli::Y => {
                let (ny, py) = self.y_image_raw();
                (neg ^ ny, py)
            }
        }
    }

    /// Look up a 1q Clifford from its X and Z images.
    fn from_images(x_img: SignedPauli, z_img: SignedPauli) -> Clifford {
        let (nx, px) = x_img;
        let (nz, pz) = z_img;
        for &variant in &ALL_1Q {
            let (enx, epx, enz, epz) = IMAGES_1Q[variant as usize];
            if enx == nx && epx == px && enz == nz && epz == pz {
                return variant;
            }
        }
        unreachable!(
            "Invalid Clifford images: X->({}, {:?}), Z->({}, {:?})",
            nx, px, nz, pz
        )
    }

    /// Inverse for two-qubit gates (hardcoded dagger pairs and self-inverses).
    fn inverse_2q(self) -> Clifford {
        match self {
            // Self-inverse gates
            Clifford::CX => Clifford::CX,
            Clifford::CY => Clifford::CY,
            Clifford::CZ => Clifford::CZ,
            Clifford::SWAP => Clifford::SWAP,
            // Dagger pairs
            Clifford::SXX => Clifford::SXXdg,
            Clifford::SXXdg => Clifford::SXX,
            Clifford::SYY => Clifford::SYYdg,
            Clifford::SYYdg => Clifford::SYY,
            Clifford::SZZ => Clifford::SZZdg,
            Clifford::SZZdg => Clifford::SZZ,
            Clifford::ISWAP => Clifford::ISWAPdg,
            Clifford::ISWAPdg => Clifford::ISWAP,
            Clifford::G => Clifford::Gdg,
            Clifford::Gdg => Clifford::G,
            _ => unreachable!(),
        }
    }
}

/// Multiply two distinct non-identity Paulis.
/// Returns (epsilon, `third_pauli`) where epsilon encodes the sign:
/// - `false` for cyclic (X*Y, Y*Z, Z*X)
/// - `true` for anti-cyclic (Y*X, Z*Y, X*Z)
fn pauli_product_sign(p1: Pauli, p2: Pauli) -> (bool, Pauli) {
    match (p1, p2) {
        (Pauli::X, Pauli::Y) => (false, Pauli::Z),
        (Pauli::Y, Pauli::Z) => (false, Pauli::X),
        (Pauli::Z, Pauli::X) => (false, Pauli::Y),
        (Pauli::Y, Pauli::X) => (true, Pauli::Z),
        (Pauli::Z, Pauli::Y) => (true, Pauli::X),
        (Pauli::X, Pauli::Z) => (true, Pauli::Y),
        _ => unreachable!("pauli_product_sign called with equal or identity Paulis"),
    }
}

/// Helper for building the inverse map.
fn set_inverse_entry(
    target: Pauli,
    neg: bool,
    source: Pauli,
    inv_x: &mut SignedPauli,
    inv_z: &mut SignedPauli,
) {
    match target {
        Pauli::X => *inv_x = (neg, source),
        Pauli::Z => *inv_z = (neg, source),
        Pauli::Y => {}
        Pauli::I => unreachable!(),
    }
}

// ============================================================================
// Mul trait: * operator for 1q composition
// ============================================================================

impl Mul for Clifford {
    type Output = Clifford;

    /// Composes two single-qubit Cliffords. Panics for two-qubit gates.
    fn mul(self, rhs: Clifford) -> Clifford {
        self.compose(rhs)
    }
}

// ============================================================================
// Display
// ============================================================================

impl fmt::Display for Clifford {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Clifford::I => "I",
            Clifford::X => "X",
            Clifford::Y => "Y",
            Clifford::Z => "Z",
            Clifford::H => "H",
            Clifford::H2 => "H2",
            Clifford::H3 => "H3",
            Clifford::H4 => "H4",
            Clifford::H5 => "H5",
            Clifford::H6 => "H6",
            Clifford::SX => "SX",
            Clifford::SXdg => "SXdg",
            Clifford::SY => "SY",
            Clifford::SYdg => "SYdg",
            Clifford::SZ => "SZ",
            Clifford::SZdg => "SZdg",
            Clifford::F => "F",
            Clifford::Fdg => "Fdg",
            Clifford::F2 => "F2",
            Clifford::F2dg => "F2dg",
            Clifford::F3 => "F3",
            Clifford::F3dg => "F3dg",
            Clifford::F4 => "F4",
            Clifford::F4dg => "F4dg",
            Clifford::CX => "CX",
            Clifford::CY => "CY",
            Clifford::CZ => "CZ",
            Clifford::SWAP => "SWAP",
            Clifford::SXX => "SXX",
            Clifford::SXXdg => "SXXdg",
            Clifford::SYY => "SYY",
            Clifford::SYYdg => "SYYdg",
            Clifford::SZZ => "SZZ",
            Clifford::SZZdg => "SZZdg",
            Clifford::ISWAP => "ISWAP",
            Clifford::ISWAPdg => "ISWAPdg",
            Clifford::G => "G",
            Clifford::Gdg => "Gdg",
        };
        write!(f, "{name}")
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ====== Collection tests ======

    #[test]
    fn test_variant_counts() {
        assert_eq!(Clifford::all().len(), 38);
        assert_eq!(Clifford::all_1q().len(), 24);
        assert_eq!(Clifford::all_2q().len(), 14);
    }

    #[test]
    fn test_num_qubits() {
        for &c in Clifford::all_1q() {
            assert_eq!(c.num_qubits(), 1, "{c} should be 1q");
        }
        for &c in Clifford::all_2q() {
            assert_eq!(c.num_qubits(), 2, "{c} should be 2q");
        }
    }

    #[test]
    fn test_all_distinct() {
        let all = Clifford::all();
        for i in 0..all.len() {
            for j in (i + 1)..all.len() {
                assert_ne!(all[i], all[j], "{} and {} are equal", all[i], all[j]);
            }
        }
    }

    // ====== 1q images and group tests ======

    #[test]
    fn test_1q_images_match_clifford_rep() {
        for &c in Clifford::all_1q() {
            let rep = c.on_qubit(0);
            let x_img = rep.x_image(0);
            let z_img = rep.z_image(0);

            let (sx, px) = c.x_image();
            let (sz, pz) = c.z_image();

            let expected_x_sign = if x_img.phase() == crate::QuarterPhase::MinusOne {
                Sign::MinusOne
            } else {
                Sign::PlusOne
            };
            assert_eq!(px, x_img.get(0), "{c}: X Pauli mismatch");
            assert_eq!(sx, expected_x_sign, "{c}: X sign mismatch");

            let expected_z_sign = if z_img.phase() == crate::QuarterPhase::MinusOne {
                Sign::MinusOne
            } else {
                Sign::PlusOne
            };
            assert_eq!(pz, z_img.get(0), "{c}: Z Pauli mismatch");
            assert_eq!(sz, expected_z_sign, "{c}: Z sign mismatch");
        }
    }

    #[test]
    fn test_1q_compose_matches_clifford_rep() {
        for &a in Clifford::all_1q() {
            for &b in Clifford::all_1q() {
                let composed_enum = a.compose(b);
                let composed_rep = a.on_qubit(0).compose(&b.on_qubit(0));
                let expected = Clifford::from_clifford_rep(&composed_rep)
                    .unwrap_or_else(|| panic!("Failed to identify {a} * {b}"));
                assert_eq!(composed_enum, expected, "{a} * {b} mismatch");
            }
        }
    }

    #[test]
    fn test_1q_group_closure() {
        for &a in Clifford::all_1q() {
            for &b in Clifford::all_1q() {
                let _ = a.compose(b);
            }
        }
    }

    #[test]
    fn test_1q_identity() {
        for &c in Clifford::all_1q() {
            assert_eq!(Clifford::I.compose(c), c);
            assert_eq!(c.compose(Clifford::I), c);
        }
    }

    #[test]
    fn test_1q_associativity() {
        let a = Clifford::H;
        let b = Clifford::SZ;
        let c = Clifford::F;
        assert_eq!((a * b) * c, a * (b * c));
    }

    // ====== Inverse tests (all gates) ======

    #[test]
    fn test_1q_inverse() {
        for &c in Clifford::all_1q() {
            let inv = c.inverse();
            assert_eq!(c.compose(inv), Clifford::I, "{c} * {c}^-1 != I");
            assert_eq!(inv.compose(c), Clifford::I, "{c}^-1 * {c} != I");
        }
    }

    #[test]
    fn test_2q_inverse_via_clifford_rep() {
        for &c in Clifford::all_2q() {
            let inv = c.inverse();
            let rep = c.on_qubits(0, 1);
            let inv_rep = inv.on_qubits(0, 1);
            let product = rep.compose(&inv_rep);
            let identity = CliffordRep::identity(2);
            for q in 0..2 {
                assert_eq!(
                    product.x_image(q),
                    identity.x_image(q),
                    "{c} * {c}^-1: x_image({q}) mismatch"
                );
                assert_eq!(
                    product.z_image(q),
                    identity.z_image(q),
                    "{c} * {c}^-1: z_image({q}) mismatch"
                );
            }
        }
    }

    // ====== Specific gate properties ======

    #[test]
    fn test_paulis_are_involutions() {
        assert_eq!(Clifford::X * Clifford::X, Clifford::I);
        assert_eq!(Clifford::Y * Clifford::Y, Clifford::I);
        assert_eq!(Clifford::Z * Clifford::Z, Clifford::I);
    }

    #[test]
    fn test_hadamards_are_involutions() {
        for &h in &[
            Clifford::H,
            Clifford::H2,
            Clifford::H3,
            Clifford::H4,
            Clifford::H5,
            Clifford::H6,
        ] {
            assert_eq!(h.compose(h), Clifford::I, "{h}^2 != I");
        }
    }

    #[test]
    fn test_face_gate_order_3() {
        assert_eq!(Clifford::F * Clifford::F * Clifford::F, Clifford::I);
    }

    #[test]
    fn test_1q_dagger_pairs() {
        let pairs = [
            (Clifford::SX, Clifford::SXdg),
            (Clifford::SY, Clifford::SYdg),
            (Clifford::SZ, Clifford::SZdg),
            (Clifford::F, Clifford::Fdg),
            (Clifford::F2, Clifford::F2dg),
            (Clifford::F3, Clifford::F3dg),
            (Clifford::F4, Clifford::F4dg),
        ];
        for (gate, dagger) in pairs {
            assert_eq!(gate.compose(dagger), Clifford::I, "{gate} * {gate}dg != I");
            assert_eq!(gate.inverse(), dagger, "{gate}^-1 != {gate}dg");
        }
    }

    #[test]
    fn test_2q_dagger_pairs() {
        let pairs = [
            (Clifford::SXX, Clifford::SXXdg),
            (Clifford::SYY, Clifford::SYYdg),
            (Clifford::SZZ, Clifford::SZZdg),
            (Clifford::ISWAP, Clifford::ISWAPdg),
            (Clifford::G, Clifford::Gdg),
        ];
        for (gate, dagger) in pairs {
            assert_eq!(gate.inverse(), dagger, "{gate}^-1 != {dagger}");
            assert_eq!(dagger.inverse(), gate, "{dagger}^-1 != {gate}");
        }
    }

    #[test]
    fn test_2q_self_inverse() {
        for &g in &[Clifford::CX, Clifford::CY, Clifford::CZ, Clifford::SWAP] {
            assert_eq!(g.inverse(), g, "{g} should be self-inverse");
        }
    }

    // ====== Embedding tests ======

    #[test]
    fn test_2q_on_qubits_valid() {
        for &c in Clifford::all_2q() {
            let rep = c.on_qubits(0, 1);
            assert!(rep.is_valid(), "{c} on_qubits(0,1) is not valid");
        }
    }

    #[test]
    fn test_on_qubits_noncontiguous() {
        let rep = Clifford::CX.on_qubits(0, 3);
        assert!(rep.is_valid());
        assert_eq!(rep.num_qubits(), 4);
    }

    // ====== Conjugate tests ======

    #[test]
    fn test_conjugate_identity() {
        for p in [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z] {
            let (sign, result) = Clifford::I.conjugate(p);
            assert_eq!(result, p);
            assert_eq!(sign, Sign::PlusOne);
        }
    }

    #[test]
    fn test_conjugate_hadamard() {
        assert_eq!(Clifford::H.conjugate(Pauli::X), (Sign::PlusOne, Pauli::Z));
        assert_eq!(Clifford::H.conjugate(Pauli::Z), (Sign::PlusOne, Pauli::X));
    }

    // ====== Roundtrip tests ======

    #[test]
    fn test_from_clifford_rep_roundtrip() {
        for &c in Clifford::all_1q() {
            let rep = c.on_qubit(0);
            let back = Clifford::from_clifford_rep(&rep).unwrap();
            assert_eq!(c, back, "Roundtrip failed for {c}");
        }
    }

    // ====== Display / default ======

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", Clifford::H), "H");
        assert_eq!(format!("{}", Clifford::CX), "CX");
        assert_eq!(format!("{}", Clifford::ISWAPdg), "ISWAPdg");
    }

    #[test]
    fn test_default_is_identity() {
        assert_eq!(Clifford::default(), Clifford::I);
    }
}
