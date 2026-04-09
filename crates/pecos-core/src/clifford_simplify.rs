//! Rotation-to-Clifford simplification using `Angle64` fixed-point comparison.
//!
//! When a rotation gate is applied at a special angle (multiples of pi/4 or pi/2),
//! it is equivalent to a named Clifford gate. This module provides a single source
//! of truth for those simplifications so that both PHIR-level passes and engine-level
//! dispatch can reuse the same logic.

use crate::angle::Angle;
use crate::gate_type::GateType;

/// Type alias -- all comparisons use 64-bit fixed-point angles.
type A64 = Angle<u64>;

/// Eighth-turn (pi/4): `QUARTER_TURN` / 2.
fn eighth_turn() -> A64 {
    A64::QUARTER_TURN / 2u64
}

/// Negative eighth-turn (7*pi/4): -`QUARTER_TURN` / 2.
fn neg_eighth_turn() -> A64 {
    -(A64::QUARTER_TURN / 2u64)
}

/// Try to simplify a single-angle rotation gate to a named Clifford gate.
///
/// Supports `RZ`, `RX`, `RY`, `RZZ`, `RXX`, `RYY`.
/// Returns `Some(clifford_gate)` when the angle matches a known Clifford, or
/// `None` if the angle is not a special Clifford angle.
///
/// For `RZ(0)`, `RX(0)`, etc. returns `Some(GateType::I)` (identity).
///
/// # Special cases
///
/// | Rotation | Angle        | Simplifies to  |
/// |----------|-------------|----------------|
/// | RZ(0)    | 0           | I              |
/// | RZ(pi)   | HALF_TURN   | Z              |
/// | RZ(pi/2) | QUARTER_TURN| SZ             |
/// | RZ(-pi/2)| 3/4 TURN    | SZdg           |
/// | RZ(pi/4) | EIGHTH_TURN | T              |
/// | RZ(-pi/4)| NEG_EIGHTH  | Tdg            |
/// | RX(0)    | 0           | I              |
/// | RX(pi)   | HALF_TURN   | X              |
/// | RX(pi/2) | QUARTER_TURN| SX             |
/// | RX(-pi/2)| 3/4 TURN    | SXdg           |
/// | RY(0)    | 0           | I              |
/// | RY(pi)   | HALF_TURN   | Y              |
/// | RY(pi/2) | QUARTER_TURN| SY             |
/// | RY(-pi/2)| 3/4 TURN    | SYdg           |
/// | RZZ(0)   | 0           | I (per qubit)  |
/// | RZZ(pi/2)| QUARTER_TURN| SZZ            |
/// | RZZ(-pi/2)| 3/4 TURN  | SZZdg          |
/// | RXX(pi/2)| QUARTER_TURN| SXX            |
/// | RXX(-pi/2)| 3/4 TURN  | SXXdg          |
/// | RYY(pi/2)| QUARTER_TURN| SYY            |
/// | RYY(-pi/2)| 3/4 TURN  | SYYdg          |
#[must_use]
pub fn try_simplify_rotation(gate: GateType, angle: A64) -> Option<GateType> {
    match gate {
        GateType::RZ => simplify_rz(angle),
        GateType::RX => simplify_rx(angle),
        GateType::RY => simplify_ry(angle),
        GateType::RZZ => simplify_rzz(angle),
        GateType::RXX => simplify_rxx(angle),
        GateType::RYY => simplify_ryy(angle),
        _ => None,
    }
}

/// Try to simplify an R1XY(theta, phi) gate to a named Clifford.
///
/// R1XY(theta, phi) is a rotation by `theta` about the axis
/// `cos(phi)*X + sin(phi)*Y` in the XY plane.
///
/// R1XY has two angle parameters, so it is handled separately from the
/// single-angle rotations.
///
/// | theta     | phi            | Simplifies to |
/// |-----------|---------------|---------------|
/// | 0         | any           | I             |
/// | pi/2      | 0             | SX            |
/// | pi/2      | pi            | `SXdg`        |
/// | pi/2      | pi/2          | SY            |
/// | pi/2      | 3pi/2         | `SYdg`        |
/// | pi        | 0 or pi       | X             |
/// | pi        | pi/2 or 3pi/2 | Y             |
/// | 3pi/2     | 0             | `SXdg`        |
/// | 3pi/2     | pi            | SX            |
/// | 3pi/2     | pi/2          | `SYdg`        |
/// | 3pi/2     | 3pi/2         | SY            |
///
/// A negative axis flips the sign of the rotation angle. That only collapses to
/// the same Clifford up to global phase for half-turns (`pi`), not for the
/// quarter-turn sqrt gates.
#[must_use]
pub fn try_simplify_r1xy(theta: A64, phi: A64) -> Option<GateType> {
    if theta == A64::ZERO {
        return Some(GateType::I);
    }

    match phi {
        A64::ZERO => simplify_rx(theta),
        A64::HALF_TURN => simplify_rx(-theta),
        A64::QUARTER_TURN => simplify_ry(theta),
        A64::THREE_QUARTERS_TURN => simplify_ry(-theta),
        _ => None,
    }
}

// -------------------------------------------------------------------------
// Internal helpers
// -------------------------------------------------------------------------

/// Negate an angle.
fn neg(a: A64) -> A64 {
    -a
}

fn simplify_rz(angle: A64) -> Option<GateType> {
    if angle == A64::ZERO {
        Some(GateType::I)
    } else if angle == A64::HALF_TURN || angle == neg(A64::HALF_TURN) {
        Some(GateType::Z)
    } else if angle == A64::QUARTER_TURN {
        Some(GateType::SZ)
    } else if angle == A64::THREE_QUARTERS_TURN || angle == neg(A64::QUARTER_TURN) {
        Some(GateType::SZdg)
    } else if angle == eighth_turn() {
        Some(GateType::T)
    } else if angle == neg_eighth_turn() {
        Some(GateType::Tdg)
    } else {
        None
    }
}

fn simplify_rx(angle: A64) -> Option<GateType> {
    if angle == A64::ZERO {
        Some(GateType::I)
    } else if angle == A64::HALF_TURN || angle == neg(A64::HALF_TURN) {
        Some(GateType::X)
    } else if angle == A64::QUARTER_TURN {
        Some(GateType::SX)
    } else if angle == A64::THREE_QUARTERS_TURN || angle == neg(A64::QUARTER_TURN) {
        Some(GateType::SXdg)
    } else {
        None
    }
}

fn simplify_ry(angle: A64) -> Option<GateType> {
    if angle == A64::ZERO {
        Some(GateType::I)
    } else if angle == A64::HALF_TURN || angle == neg(A64::HALF_TURN) {
        Some(GateType::Y)
    } else if angle == A64::QUARTER_TURN {
        Some(GateType::SY)
    } else if angle == A64::THREE_QUARTERS_TURN || angle == neg(A64::QUARTER_TURN) {
        Some(GateType::SYdg)
    } else {
        None
    }
}

fn simplify_rzz(angle: A64) -> Option<GateType> {
    if angle == A64::ZERO {
        Some(GateType::I)
    } else if angle == A64::QUARTER_TURN {
        Some(GateType::SZZ)
    } else if angle == A64::THREE_QUARTERS_TURN || angle == neg(A64::QUARTER_TURN) {
        Some(GateType::SZZdg)
    } else {
        None
        // Note: RZZ(pi) = Z tensor Z is a *decomposition* (two separate gates),
        // not a single GateType, so the caller must handle it.
    }
}

fn simplify_rxx(angle: A64) -> Option<GateType> {
    if angle == A64::ZERO {
        Some(GateType::I)
    } else if angle == A64::QUARTER_TURN {
        Some(GateType::SXX)
    } else if angle == A64::THREE_QUARTERS_TURN || angle == neg(A64::QUARTER_TURN) {
        Some(GateType::SXXdg)
    } else {
        None
    }
}

fn simplify_ryy(angle: A64) -> Option<GateType> {
    if angle == A64::ZERO {
        Some(GateType::I)
    } else if angle == A64::QUARTER_TURN {
        Some(GateType::SYY)
    } else if angle == A64::THREE_QUARTERS_TURN || angle == neg(A64::QUARTER_TURN) {
        Some(GateType::SYYdg)
    } else {
        None
    }
}

/// Check whether a two-qubit rotation at half turn decomposes into two
/// single-qubit Pauli gates.
///
/// Returns `Some(pauli)` when the gate should be replaced by applying
/// `pauli` to each qubit independently:
///
/// | Gate    | Angle | Decomposition |
/// |---------|-------|---------------|
/// | RZZ(pi) | pi    | Z + Z         |
/// | RXX(pi) | pi    | X + X         |
/// | RYY(pi) | pi    | Y + Y         |
///
/// This is separate from `try_simplify_rotation` because the result is a
/// *decomposition* into two single-qubit gates, not a single gate replacement.
#[must_use]
pub fn half_turn_decomposition(gate: GateType, angle: A64) -> Option<GateType> {
    if angle != A64::HALF_TURN && angle != neg(A64::HALF_TURN) {
        return None;
    }
    match gate {
        GateType::RZZ => Some(GateType::Z),
        GateType::RXX => Some(GateType::X),
        GateType::RYY => Some(GateType::Y),
        _ => None,
    }
}

/// Check whether RZZ at the given angle decomposes to Z tensor Z (i.e. angle = pi).
///
/// Convenience wrapper around [`half_turn_decomposition`] for the common
/// RZZ-only case.
#[must_use]
pub fn is_rzz_z_tensor_z(angle: A64) -> bool {
    half_turn_decomposition(GateType::RZZ, angle).is_some()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Angle64;

    #[test]
    fn rz_simplifications() {
        assert_eq!(
            try_simplify_rotation(GateType::RZ, Angle64::ZERO),
            Some(GateType::I)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RZ, Angle64::HALF_TURN),
            Some(GateType::Z)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RZ, Angle64::QUARTER_TURN),
            Some(GateType::SZ)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RZ, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SZdg)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RZ, eighth_turn()),
            Some(GateType::T)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RZ, neg_eighth_turn()),
            Some(GateType::Tdg)
        );
        // Non-Clifford angle
        assert_eq!(
            try_simplify_rotation(GateType::RZ, Angle64::from_radians(0.123)),
            None
        );
    }

    #[test]
    fn rx_simplifications() {
        assert_eq!(
            try_simplify_rotation(GateType::RX, Angle64::ZERO),
            Some(GateType::I)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RX, Angle64::HALF_TURN),
            Some(GateType::X)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RX, Angle64::QUARTER_TURN),
            Some(GateType::SX)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RX, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SXdg)
        );
    }

    #[test]
    fn ry_simplifications() {
        assert_eq!(
            try_simplify_rotation(GateType::RY, Angle64::ZERO),
            Some(GateType::I)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RY, Angle64::HALF_TURN),
            Some(GateType::Y)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RY, Angle64::QUARTER_TURN),
            Some(GateType::SY)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RY, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SYdg)
        );
    }

    #[test]
    fn rzz_simplifications() {
        assert_eq!(
            try_simplify_rotation(GateType::RZZ, Angle64::ZERO),
            Some(GateType::I)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RZZ, Angle64::QUARTER_TURN),
            Some(GateType::SZZ)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RZZ, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SZZdg)
        );
        // RZZ(pi) is not a single gate -- returns None
        assert_eq!(
            try_simplify_rotation(GateType::RZZ, Angle64::HALF_TURN),
            None
        );
        assert!(is_rzz_z_tensor_z(Angle64::HALF_TURN));
    }

    #[test]
    fn rxx_simplifications() {
        assert_eq!(
            try_simplify_rotation(GateType::RXX, Angle64::ZERO),
            Some(GateType::I)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RXX, Angle64::QUARTER_TURN),
            Some(GateType::SXX)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RXX, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SXXdg)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RXX, Angle64::HALF_TURN),
            None
        );
    }

    #[test]
    fn ryy_simplifications() {
        assert_eq!(
            try_simplify_rotation(GateType::RYY, Angle64::ZERO),
            Some(GateType::I)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RYY, Angle64::QUARTER_TURN),
            Some(GateType::SYY)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RYY, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SYYdg)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RYY, Angle64::HALF_TURN),
            None
        );
    }

    #[test]
    fn r1xy_identity() {
        // theta=0 with any phi is identity
        assert_eq!(
            try_simplify_r1xy(Angle64::ZERO, Angle64::ZERO),
            Some(GateType::I)
        );
        assert_eq!(
            try_simplify_r1xy(Angle64::ZERO, Angle64::QUARTER_TURN),
            Some(GateType::I)
        );
        assert_eq!(
            try_simplify_r1xy(Angle64::ZERO, Angle64::HALF_TURN),
            Some(GateType::I)
        );
    }

    #[test]
    fn r1xy_half_turn_pauli_gates() {
        // theta=pi, phi=0: X
        assert_eq!(
            try_simplify_r1xy(Angle64::HALF_TURN, Angle64::ZERO),
            Some(GateType::X)
        );
        // theta=pi, phi=pi/2: Y
        assert_eq!(
            try_simplify_r1xy(Angle64::HALF_TURN, Angle64::QUARTER_TURN),
            Some(GateType::Y)
        );
        // theta=-pi also works
        assert_eq!(
            try_simplify_r1xy(-Angle64::HALF_TURN, Angle64::ZERO),
            Some(GateType::X)
        );
        assert_eq!(
            try_simplify_r1xy(-Angle64::HALF_TURN, Angle64::QUARTER_TURN),
            Some(GateType::Y)
        );
    }

    #[test]
    fn r1xy_half_turn_negated_axis() {
        // phi=pi (-X axis) is equivalent to X for stabilizer
        assert_eq!(
            try_simplify_r1xy(Angle64::HALF_TURN, Angle64::HALF_TURN),
            Some(GateType::X)
        );
        // phi=3pi/2 (-Y axis) is equivalent to Y for stabilizer
        assert_eq!(
            try_simplify_r1xy(Angle64::HALF_TURN, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::Y)
        );
    }

    #[test]
    fn r1xy_quarter_turn_sqrt_gates() {
        // theta=pi/2, phi=0: SX
        assert_eq!(
            try_simplify_r1xy(Angle64::QUARTER_TURN, Angle64::ZERO),
            Some(GateType::SX)
        );
        // theta=pi/2, phi=pi/2: SY
        assert_eq!(
            try_simplify_r1xy(Angle64::QUARTER_TURN, Angle64::QUARTER_TURN),
            Some(GateType::SY)
        );
        // theta=pi/2, phi=pi: rotation about -X is SXdg
        assert_eq!(
            try_simplify_r1xy(Angle64::QUARTER_TURN, Angle64::HALF_TURN),
            Some(GateType::SXdg)
        );
        // theta=pi/2, phi=3pi/2: rotation about -Y is SYdg
        assert_eq!(
            try_simplify_r1xy(Angle64::QUARTER_TURN, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SYdg)
        );
    }

    #[test]
    fn r1xy_three_quarter_turn_sqrt_dagger_gates() {
        // theta=3pi/2, phi=0: SXdg
        assert_eq!(
            try_simplify_r1xy(Angle64::THREE_QUARTERS_TURN, Angle64::ZERO),
            Some(GateType::SXdg)
        );
        // theta=3pi/2, phi=pi/2: SYdg
        assert_eq!(
            try_simplify_r1xy(Angle64::THREE_QUARTERS_TURN, Angle64::QUARTER_TURN),
            Some(GateType::SYdg)
        );
        // theta=3pi/2, phi=pi: rotation about -X is SX
        assert_eq!(
            try_simplify_r1xy(Angle64::THREE_QUARTERS_TURN, Angle64::HALF_TURN),
            Some(GateType::SX)
        );
        // theta=3pi/2, phi=3pi/2: rotation about -Y is SY
        assert_eq!(
            try_simplify_r1xy(Angle64::THREE_QUARTERS_TURN, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SY)
        );
        // theta=-pi/2 wraps to 3pi/2
        assert_eq!(
            try_simplify_r1xy(-Angle64::QUARTER_TURN, Angle64::ZERO),
            Some(GateType::SXdg)
        );
        assert_eq!(
            try_simplify_r1xy(-Angle64::QUARTER_TURN, Angle64::QUARTER_TURN),
            Some(GateType::SYdg)
        );
    }

    #[test]
    fn r1xy_non_clifford_angles() {
        // Non-Clifford theta
        assert_eq!(
            try_simplify_r1xy(Angle64::from_radians(0.123), Angle64::ZERO),
            None
        );
        // Non-axis phi (pi/4 is not along X or Y axis)
        assert_eq!(
            try_simplify_r1xy(Angle64::HALF_TURN, Angle64::QUARTER_TURN / 2u64),
            None
        );
    }

    #[test]
    fn negative_angles_via_wrapping_rz() {
        let neg_pi = Angle64::from_radians(-std::f64::consts::PI);
        assert_eq!(
            try_simplify_rotation(GateType::RZ, neg_pi),
            Some(GateType::Z)
        );

        let neg_half_pi = Angle64::from_radians(-std::f64::consts::FRAC_PI_2);
        assert_eq!(
            try_simplify_rotation(GateType::RZ, neg_half_pi),
            Some(GateType::SZdg)
        );

        let neg_quarter_pi = Angle64::from_radians(-std::f64::consts::FRAC_PI_4);
        assert_eq!(
            try_simplify_rotation(GateType::RZ, neg_quarter_pi),
            Some(GateType::Tdg)
        );
    }

    #[test]
    fn negative_angles_via_wrapping_rx_ry() {
        use std::f64::consts::{FRAC_PI_2, PI};
        // RX
        assert_eq!(
            try_simplify_rotation(GateType::RX, Angle64::from_radians(-PI)),
            Some(GateType::X)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RX, Angle64::from_radians(-FRAC_PI_2)),
            Some(GateType::SXdg)
        );
        // RY
        assert_eq!(
            try_simplify_rotation(GateType::RY, Angle64::from_radians(-PI)),
            Some(GateType::Y)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RY, Angle64::from_radians(-FRAC_PI_2)),
            Some(GateType::SYdg)
        );
    }

    #[test]
    fn negative_angles_via_wrapping_two_qubit() {
        use std::f64::consts::FRAC_PI_2;
        // RZZ
        assert_eq!(
            try_simplify_rotation(GateType::RZZ, Angle64::from_radians(-FRAC_PI_2)),
            Some(GateType::SZZdg)
        );
        // RXX
        assert_eq!(
            try_simplify_rotation(GateType::RXX, Angle64::from_radians(-FRAC_PI_2)),
            Some(GateType::SXXdg)
        );
        // RYY
        assert_eq!(
            try_simplify_rotation(GateType::RYY, Angle64::from_radians(-FRAC_PI_2)),
            Some(GateType::SYYdg)
        );
    }

    #[test]
    fn half_turn_decompositions() {
        assert_eq!(
            half_turn_decomposition(GateType::RZZ, Angle64::HALF_TURN),
            Some(GateType::Z)
        );
        assert_eq!(
            half_turn_decomposition(GateType::RXX, Angle64::HALF_TURN),
            Some(GateType::X)
        );
        assert_eq!(
            half_turn_decomposition(GateType::RYY, Angle64::HALF_TURN),
            Some(GateType::Y)
        );
        // Negative pi
        let neg_pi = Angle64::from_radians(-std::f64::consts::PI);
        assert_eq!(
            half_turn_decomposition(GateType::RZZ, neg_pi),
            Some(GateType::Z)
        );
        assert_eq!(
            half_turn_decomposition(GateType::RXX, neg_pi),
            Some(GateType::X)
        );
        assert_eq!(
            half_turn_decomposition(GateType::RYY, neg_pi),
            Some(GateType::Y)
        );
        // Non-half-turn returns None
        assert_eq!(
            half_turn_decomposition(GateType::RZZ, Angle64::QUARTER_TURN),
            None
        );
        assert_eq!(half_turn_decomposition(GateType::RZZ, Angle64::ZERO), None);
        // Non-rotation gate returns None
        assert_eq!(
            half_turn_decomposition(GateType::H, Angle64::HALF_TURN),
            None
        );
    }

    #[test]
    fn is_rzz_z_tensor_z_wraps_half_turn_decomposition() {
        assert!(is_rzz_z_tensor_z(Angle64::HALF_TURN));
        assert!(is_rzz_z_tensor_z(Angle64::from_radians(
            -std::f64::consts::PI
        )));
        assert!(!is_rzz_z_tensor_z(Angle64::ZERO));
        assert!(!is_rzz_z_tensor_z(Angle64::QUARTER_TURN));
    }

    #[test]
    fn crz_not_in_simplify_rotation() {
        // CRZ cannot be simplified to a single gate (CRZ(pi) != CZ).
        // It is handled via decomposition in the CliffordRotation trait instead.
        assert_eq!(try_simplify_rotation(GateType::CRZ, Angle64::ZERO), None);
        assert_eq!(
            try_simplify_rotation(GateType::CRZ, Angle64::HALF_TURN),
            None
        );
    }

    #[test]
    fn non_rotation_gate_returns_none() {
        assert_eq!(try_simplify_rotation(GateType::H, Angle64::ZERO), None);
        assert_eq!(
            try_simplify_rotation(GateType::CX, Angle64::HALF_TURN),
            None
        );
    }

    #[test]
    fn from_radians_round_trip() {
        use std::f64::consts::{FRAC_PI_2, PI};
        assert_eq!(
            try_simplify_rotation(GateType::RZ, Angle64::from_radians(PI)),
            Some(GateType::Z)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RZ, Angle64::from_radians(FRAC_PI_2)),
            Some(GateType::SZ)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RX, Angle64::from_radians(PI)),
            Some(GateType::X)
        );
        assert_eq!(
            try_simplify_rotation(GateType::RY, Angle64::from_radians(PI)),
            Some(GateType::Y)
        );
    }
}
