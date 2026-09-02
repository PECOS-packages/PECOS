//! Rotation-to-Clifford simplification using `Angle64` fixed-point comparison.
//!
//! When a rotation gate is applied at a special Clifford angle,
//! it is equivalent up to global phase to a named gate. This module provides a single source
//! of truth for those simplifications so that both PHIR-level passes and engine-level
//! dispatch can reuse the same logic. The unsuffixed single-angle helpers match exactly;
//! Clifford-only consumers that accept numerically lowered angles must opt into the
//! corresponding `*_snapped` helper.

use crate::angle::Angle;
use crate::gate_type::GateType;

/// Type alias -- all comparisons use 64-bit fixed-point angles.
type A64 = Angle<u64>;

/// Numerical lowering pipelines can produce angles that are a few fixed-point
/// units away from canonical Clifford quarter-turns. Clifford-only entry points
/// snap within this tolerance so genuine non-Clifford rotations still fail loudly.
/// Shared rewriting and propagation consumers use exact matching instead.
const CLIFFORD_SNAP_EPSILON_TURNS: f64 = 1e-9;

/// Try to simplify a single-angle rotation gate to a named Clifford gate.
///
/// Supports `RZ`, `RX`, `RY`, `RZZ`, `RXX`, `RYY`.
/// Returns `Some(clifford_gate)` when the angle matches a known Clifford, or
/// `None` if the angle is not a special Clifford angle.
/// Matching is exact; use [`try_simplify_rotation_snapped`] only at a
/// Clifford-only boundary that explicitly accepts numerical tolerance.
///
/// For `RZ(0)`, `RX(0)`, etc. returns `Some(GateType::I)` (identity).
///
/// # Special cases
///
/// | Rotation | Angle        | Simplifies to (up to global phase) |
/// |----------|-------------|----------------|
/// | RZ(0)    | 0           | I              |
/// | RZ(pi)   | HALF_TURN   | Z              |
/// | RZ(pi/2) | QUARTER_TURN| SZ             |
/// | RZ(-pi/2)| 3/4 TURN    | SZdg           |
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

/// Try to simplify a single-angle rotation after applying the Clifford-only
/// numerical snap policy.
///
/// This is intended for entry points that cannot execute arbitrary rotations.
/// General circuit rewriting and propagation should use the exact
/// [`try_simplify_rotation`] helper.
#[must_use]
pub fn try_simplify_rotation_snapped(gate: GateType, angle: A64) -> Option<GateType> {
    try_simplify_rotation(gate, snap_clifford_angle(angle))
}

/// Try to simplify an RXY1Q(theta, phi) gate to a named Clifford.
///
/// RXY1Q(theta, phi) is a rotation by `theta` about the axis
/// `cos(phi)*X + sin(phi)*Y` in the XY plane.
///
/// RXY1Q has two angle parameters, so it is handled separately from the
/// single-angle rotations.
/// It retains its pre-existing numerical snap policy for both angles.
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
pub fn try_simplify_rxy1q(theta: A64, phi: A64) -> Option<GateType> {
    let theta = snap_clifford_angle(theta);
    if theta == A64::ZERO {
        return Some(GateType::I);
    }

    let phi = snap_clifford_angle(phi);
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

fn snap_clifford_angle(angle: A64) -> A64 {
    [
        A64::ZERO,
        A64::QUARTER_TURN,
        A64::HALF_TURN,
        A64::THREE_QUARTERS_TURN,
    ]
    .into_iter()
    .find(|target| angle.abs_diff_eq_turns(target, CLIFFORD_SNAP_EPSILON_TURNS))
    .unwrap_or(angle)
}

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
/// Matching is exact; use [`half_turn_decomposition_snapped`] only at a
/// Clifford-only boundary that explicitly accepts numerical tolerance.
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

/// Check for a two-qubit half-turn decomposition after applying the
/// Clifford-only numerical snap policy.
///
/// General circuit rewriting and propagation should use the exact
/// [`half_turn_decomposition`] helper.
#[must_use]
pub fn half_turn_decomposition_snapped(gate: GateType, angle: A64) -> Option<GateType> {
    half_turn_decomposition(gate, snap_clifford_angle(angle))
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
        // T/Tdg use the conventional phase-fixed matrices, so symmetric
        // RZ(+/-pi/4) rotations cannot be replaced by those named gates.
        let eighth_turn = Angle64::QUARTER_TURN / 2u64;
        assert_eq!(try_simplify_rotation(GateType::RZ, eighth_turn), None);
        assert_eq!(try_simplify_rotation(GateType::RZ, -eighth_turn), None);
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
    fn rxy1q_identity() {
        // theta=0 with any phi is identity
        assert_eq!(
            try_simplify_rxy1q(Angle64::ZERO, Angle64::ZERO),
            Some(GateType::I)
        );
        assert_eq!(
            try_simplify_rxy1q(Angle64::ZERO, Angle64::QUARTER_TURN),
            Some(GateType::I)
        );
        assert_eq!(
            try_simplify_rxy1q(Angle64::ZERO, Angle64::HALF_TURN),
            Some(GateType::I)
        );
    }

    #[test]
    fn rxy1q_half_turn_pauli_gates() {
        // theta=pi, phi=0: X
        assert_eq!(
            try_simplify_rxy1q(Angle64::HALF_TURN, Angle64::ZERO),
            Some(GateType::X)
        );
        // theta=pi, phi=pi/2: Y
        assert_eq!(
            try_simplify_rxy1q(Angle64::HALF_TURN, Angle64::QUARTER_TURN),
            Some(GateType::Y)
        );
        // theta=-pi also works
        assert_eq!(
            try_simplify_rxy1q(-Angle64::HALF_TURN, Angle64::ZERO),
            Some(GateType::X)
        );
        assert_eq!(
            try_simplify_rxy1q(-Angle64::HALF_TURN, Angle64::QUARTER_TURN),
            Some(GateType::Y)
        );
    }

    #[test]
    fn rxy1q_half_turn_negated_axis() {
        // phi=pi (-X axis) is equivalent to X for stabilizer
        assert_eq!(
            try_simplify_rxy1q(Angle64::HALF_TURN, Angle64::HALF_TURN),
            Some(GateType::X)
        );
        // phi=3pi/2 (-Y axis) is equivalent to Y for stabilizer
        assert_eq!(
            try_simplify_rxy1q(Angle64::HALF_TURN, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::Y)
        );
    }

    #[test]
    fn rxy1q_quarter_turn_sqrt_gates() {
        // theta=pi/2, phi=0: SX
        assert_eq!(
            try_simplify_rxy1q(Angle64::QUARTER_TURN, Angle64::ZERO),
            Some(GateType::SX)
        );
        // theta=pi/2, phi=pi/2: SY
        assert_eq!(
            try_simplify_rxy1q(Angle64::QUARTER_TURN, Angle64::QUARTER_TURN),
            Some(GateType::SY)
        );
        // theta=pi/2, phi=pi: rotation about -X is SXdg
        assert_eq!(
            try_simplify_rxy1q(Angle64::QUARTER_TURN, Angle64::HALF_TURN),
            Some(GateType::SXdg)
        );
        // theta=pi/2, phi=3pi/2: rotation about -Y is SYdg
        assert_eq!(
            try_simplify_rxy1q(Angle64::QUARTER_TURN, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SYdg)
        );
    }

    #[test]
    fn rxy1q_near_quarter_turn_sqrt_gates() {
        assert_eq!(
            try_simplify_rxy1q(
                Angle64::from_turns(0.25 + 1e-12),
                Angle64::from_turns(0.75 - 1e-12),
            ),
            Some(GateType::SYdg)
        );
    }

    #[test]
    fn rxy1q_three_quarter_turn_sqrt_dagger_gates() {
        // theta=3pi/2, phi=0: SXdg
        assert_eq!(
            try_simplify_rxy1q(Angle64::THREE_QUARTERS_TURN, Angle64::ZERO),
            Some(GateType::SXdg)
        );
        // theta=3pi/2, phi=pi/2: SYdg
        assert_eq!(
            try_simplify_rxy1q(Angle64::THREE_QUARTERS_TURN, Angle64::QUARTER_TURN),
            Some(GateType::SYdg)
        );
        // theta=3pi/2, phi=pi: rotation about -X is SX
        assert_eq!(
            try_simplify_rxy1q(Angle64::THREE_QUARTERS_TURN, Angle64::HALF_TURN),
            Some(GateType::SX)
        );
        // theta=3pi/2, phi=3pi/2: rotation about -Y is SY
        assert_eq!(
            try_simplify_rxy1q(Angle64::THREE_QUARTERS_TURN, Angle64::THREE_QUARTERS_TURN),
            Some(GateType::SY)
        );
        // theta=-pi/2 wraps to 3pi/2
        assert_eq!(
            try_simplify_rxy1q(-Angle64::QUARTER_TURN, Angle64::ZERO),
            Some(GateType::SXdg)
        );
        assert_eq!(
            try_simplify_rxy1q(-Angle64::QUARTER_TURN, Angle64::QUARTER_TURN),
            Some(GateType::SYdg)
        );
    }

    #[test]
    fn rxy1q_non_clifford_angles() {
        // Non-Clifford theta
        assert_eq!(
            try_simplify_rxy1q(Angle64::from_radians(0.123), Angle64::ZERO),
            None
        );
        // Non-axis phi (pi/4 is not along X or Y axis)
        assert_eq!(
            try_simplify_rxy1q(Angle64::HALF_TURN, Angle64::QUARTER_TURN / 2u64),
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
        assert_eq!(try_simplify_rotation(GateType::RZ, neg_quarter_pi), None);
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

    #[test]
    fn single_angle_rotation_tables_reject_near_clifford_turns() {
        let one_qubit = [GateType::RZ, GateType::RX, GateType::RY];
        for rotation in one_qubit {
            assert_eq!(
                try_simplify_rotation(rotation, Angle64::from_turns(0.75 - 1e-12)),
                None
            );
            assert_eq!(
                try_simplify_rotation(rotation, Angle64::from_turns(0.25 + 1e-12)),
                None
            );
            assert_eq!(
                try_simplify_rotation(rotation, Angle64::from_turns(0.5 + 1e-12)),
                None
            );
        }

        let two_qubit = [GateType::RZZ, GateType::RXX, GateType::RYY];
        for rotation in two_qubit {
            assert_eq!(
                try_simplify_rotation(rotation, Angle64::from_turns(0.75 - 1e-12)),
                None
            );
            assert_eq!(
                try_simplify_rotation(rotation, Angle64::from_turns(0.25 + 1e-12)),
                None
            );
            assert_eq!(
                half_turn_decomposition(rotation, Angle64::from_turns(0.5 + 1e-12)),
                None
            );
        }
    }

    #[test]
    fn python_float_radians_snap_to_clifford_turns() {
        use std::f64::consts::{FRAC_PI_2, PI};

        let rotations = [
            (GateType::RZ, GateType::SZ, GateType::SZdg),
            (GateType::RX, GateType::SX, GateType::SXdg),
            (GateType::RY, GateType::SY, GateType::SYdg),
            (GateType::RZZ, GateType::SZZ, GateType::SZZdg),
            (GateType::RXX, GateType::SXX, GateType::SXXdg),
            (GateType::RYY, GateType::SYY, GateType::SYYdg),
        ];
        let dagger_angles = [1.5 * PI, 3.0 * FRAC_PI_2, -FRAC_PI_2, 3.5 * PI];
        let sqrt_angles = [2.5 * PI, 1e6 * PI + FRAC_PI_2];

        for (rotation, sqrt, sqrt_dg) in rotations {
            for radians in dagger_angles {
                assert_eq!(
                    try_simplify_rotation_snapped(rotation, Angle64::from_radians(radians)),
                    Some(sqrt_dg)
                );
            }
            for radians in sqrt_angles {
                assert_eq!(
                    try_simplify_rotation_snapped(rotation, Angle64::from_radians(radians)),
                    Some(sqrt)
                );
            }
        }
    }
}
