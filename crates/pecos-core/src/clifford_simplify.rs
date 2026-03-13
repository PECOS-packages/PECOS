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

/// Eighth-turn (pi/4): QUARTER_TURN / 2.
fn eighth_turn() -> A64 {
    A64::QUARTER_TURN / 2u64
}

/// Negative eighth-turn (7*pi/4): -QUARTER_TURN / 2.
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
/// R1XY has two angle parameters, so it is handled separately from the
/// single-angle rotations.
///
/// | theta     | phi          | Simplifies to |
/// |-----------|-------------|---------------|
/// | 0         | any         | I             |
/// | pi        | 0           | X             |
/// | pi        | pi/2        | Y             |
#[must_use]
pub fn try_simplify_r1xy(theta: A64, phi: A64) -> Option<GateType> {
    if theta == A64::ZERO {
        return Some(GateType::I);
    }
    if theta == A64::HALF_TURN || theta == neg(A64::HALF_TURN) {
        if phi == A64::ZERO {
            return Some(GateType::X);
        }
        if phi == A64::QUARTER_TURN {
            return Some(GateType::Y);
        }
    }
    None
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

/// Check whether RZZ at the given angle decomposes to Z tensor Z (i.e. angle = pi).
///
/// This is separate from `try_simplify_rotation` because the result is a *decomposition*
/// into two single-qubit gates, not a single two-qubit gate replacement.
#[must_use]
pub fn is_rzz_z_tensor_z(angle: A64) -> bool {
    angle == A64::HALF_TURN || angle == neg(A64::HALF_TURN)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Angle64;

    #[test]
    fn rz_simplifications() {
        assert_eq!(try_simplify_rotation(GateType::RZ, Angle64::ZERO), Some(GateType::I));
        assert_eq!(try_simplify_rotation(GateType::RZ, Angle64::HALF_TURN), Some(GateType::Z));
        assert_eq!(try_simplify_rotation(GateType::RZ, Angle64::QUARTER_TURN), Some(GateType::SZ));
        assert_eq!(try_simplify_rotation(GateType::RZ, Angle64::THREE_QUARTERS_TURN), Some(GateType::SZdg));
        assert_eq!(try_simplify_rotation(GateType::RZ, eighth_turn()), Some(GateType::T));
        assert_eq!(try_simplify_rotation(GateType::RZ, neg_eighth_turn()), Some(GateType::Tdg));
        // Non-Clifford angle
        assert_eq!(try_simplify_rotation(GateType::RZ, Angle64::from_radians(0.123)), None);
    }

    #[test]
    fn rx_simplifications() {
        assert_eq!(try_simplify_rotation(GateType::RX, Angle64::ZERO), Some(GateType::I));
        assert_eq!(try_simplify_rotation(GateType::RX, Angle64::HALF_TURN), Some(GateType::X));
        assert_eq!(try_simplify_rotation(GateType::RX, Angle64::QUARTER_TURN), Some(GateType::SX));
        assert_eq!(try_simplify_rotation(GateType::RX, Angle64::THREE_QUARTERS_TURN), Some(GateType::SXdg));
    }

    #[test]
    fn ry_simplifications() {
        assert_eq!(try_simplify_rotation(GateType::RY, Angle64::ZERO), Some(GateType::I));
        assert_eq!(try_simplify_rotation(GateType::RY, Angle64::HALF_TURN), Some(GateType::Y));
        assert_eq!(try_simplify_rotation(GateType::RY, Angle64::QUARTER_TURN), Some(GateType::SY));
        assert_eq!(try_simplify_rotation(GateType::RY, Angle64::THREE_QUARTERS_TURN), Some(GateType::SYdg));
    }

    #[test]
    fn rzz_simplifications() {
        assert_eq!(try_simplify_rotation(GateType::RZZ, Angle64::ZERO), Some(GateType::I));
        assert_eq!(try_simplify_rotation(GateType::RZZ, Angle64::QUARTER_TURN), Some(GateType::SZZ));
        assert_eq!(try_simplify_rotation(GateType::RZZ, Angle64::THREE_QUARTERS_TURN), Some(GateType::SZZdg));
        // RZZ(pi) is not a single gate -- returns None
        assert_eq!(try_simplify_rotation(GateType::RZZ, Angle64::HALF_TURN), None);
        assert!(is_rzz_z_tensor_z(Angle64::HALF_TURN));
    }

    #[test]
    fn rxx_ryy_simplifications() {
        assert_eq!(try_simplify_rotation(GateType::RXX, Angle64::QUARTER_TURN), Some(GateType::SXX));
        assert_eq!(try_simplify_rotation(GateType::RXX, Angle64::THREE_QUARTERS_TURN), Some(GateType::SXXdg));
        assert_eq!(try_simplify_rotation(GateType::RYY, Angle64::QUARTER_TURN), Some(GateType::SYY));
        assert_eq!(try_simplify_rotation(GateType::RYY, Angle64::THREE_QUARTERS_TURN), Some(GateType::SYYdg));
    }

    #[test]
    fn r1xy_simplifications() {
        assert_eq!(try_simplify_r1xy(Angle64::ZERO, Angle64::ZERO), Some(GateType::I));
        assert_eq!(try_simplify_r1xy(Angle64::ZERO, Angle64::QUARTER_TURN), Some(GateType::I));
        assert_eq!(try_simplify_r1xy(Angle64::HALF_TURN, Angle64::ZERO), Some(GateType::X));
        assert_eq!(try_simplify_r1xy(Angle64::HALF_TURN, Angle64::QUARTER_TURN), Some(GateType::Y));
        // Non-Clifford
        assert_eq!(try_simplify_r1xy(Angle64::QUARTER_TURN, Angle64::ZERO), None);
    }

    #[test]
    fn negative_angles_via_wrapping() {
        // -pi should wrap to same as +pi for half-turn
        let neg_pi = Angle64::from_radians(-std::f64::consts::PI);
        assert_eq!(try_simplify_rotation(GateType::RZ, neg_pi), Some(GateType::Z));

        let neg_half_pi = Angle64::from_radians(-std::f64::consts::FRAC_PI_2);
        assert_eq!(try_simplify_rotation(GateType::RZ, neg_half_pi), Some(GateType::SZdg));

        let neg_quarter_pi = Angle64::from_radians(-std::f64::consts::FRAC_PI_4);
        assert_eq!(try_simplify_rotation(GateType::RZ, neg_quarter_pi), Some(GateType::Tdg));
    }

    #[test]
    fn non_rotation_gate_returns_none() {
        assert_eq!(try_simplify_rotation(GateType::H, Angle64::ZERO), None);
        assert_eq!(try_simplify_rotation(GateType::CX, Angle64::HALF_TURN), None);
    }

    #[test]
    fn from_radians_round_trip() {
        use std::f64::consts::{FRAC_PI_2, PI};
        assert_eq!(try_simplify_rotation(GateType::RZ, Angle64::from_radians(PI)), Some(GateType::Z));
        assert_eq!(try_simplify_rotation(GateType::RZ, Angle64::from_radians(FRAC_PI_2)), Some(GateType::SZ));
        assert_eq!(try_simplify_rotation(GateType::RX, Angle64::from_radians(PI)), Some(GateType::X));
        assert_eq!(try_simplify_rotation(GateType::RY, Angle64::from_radians(PI)), Some(GateType::Y));
    }
}
