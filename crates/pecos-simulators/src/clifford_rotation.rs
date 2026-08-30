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

//! Rotation gates restricted to Clifford angles.
//!
//! The [`CliffordRotation`] trait extends [`CliffordGateable`] with rotation
//! methods (`try_rz`, `try_rx`, etc.) that succeed only when the angle is a
//! Clifford angle (0, pi/2, pi, or 3pi/2).
//!
//! A blanket implementation is provided for every [`CliffordGateable`] type,
//! so stabilizer simulators get these methods for free.

use crate::CliffordGateable;
use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, QubitId};

/// Rotation gates that only succeed at Clifford angles.
///
/// This trait sits between [`CliffordGateable`] (no angles at all) and
/// [`ArbitraryRotationGateable`](crate::ArbitraryRotationGateable) (any angle).
/// It lets callers pass rotation angles while getting a clear error when
/// the angle is not a Clifford angle that the stabilizer simulator can handle.
///
/// A blanket implementation is provided for all `CliffordGateable` types.
///
/// # Example
/// ```
/// use pecos_simulators::sparse_stab::SparseStab;
/// use pecos_simulators::clifford_rotation::CliffordRotation;
/// use pecos_core::{Angle64, qid};
///
/// let mut sim = SparseStab::new(1);
/// // pi/2 rotation is a Clifford angle -- succeeds
/// assert!(sim.try_rz(Angle64::QUARTER_TURN, &qid(0)).is_ok());
/// // Arbitrary angle -- fails
/// assert!(sim.try_rz(Angle64::from_radians(0.123), &qid(0)).is_err());
/// ```
pub trait CliffordRotation: CliffordGateable {
    /// Try to apply RZ(angle). Succeeds for Clifford angles.
    ///
    /// # Errors
    /// Returns `Err` with a message if the angle is not a recognized Clifford angle.
    fn try_rz(&mut self, angle: Angle64, qubits: &[QubitId]) -> Result<&mut Self, String>;

    /// Try to apply RX(angle). Succeeds for Clifford angles.
    ///
    /// # Errors
    /// Returns `Err` with a message if the angle is not a recognized Clifford angle.
    fn try_rx(&mut self, angle: Angle64, qubits: &[QubitId]) -> Result<&mut Self, String>;

    /// Try to apply RY(angle). Succeeds for Clifford angles.
    ///
    /// # Errors
    /// Returns `Err` with a message if the angle is not a recognized Clifford angle.
    fn try_ry(&mut self, angle: Angle64, qubits: &[QubitId]) -> Result<&mut Self, String>;

    /// Try to apply RZZ(angle). Succeeds for Clifford angles and pi (Z+Z decomposition).
    ///
    /// # Errors
    /// Returns `Err` with a message if the angle is not a recognized Clifford angle.
    fn try_rzz(
        &mut self,
        angle: Angle64,
        pairs: &[(QubitId, QubitId)],
    ) -> Result<&mut Self, String>;

    /// Try to apply RXX(angle). Succeeds for Clifford angles and pi (X+X decomposition).
    ///
    /// # Errors
    /// Returns `Err` with a message if the angle is not a recognized Clifford angle.
    fn try_rxx(
        &mut self,
        angle: Angle64,
        pairs: &[(QubitId, QubitId)],
    ) -> Result<&mut Self, String>;

    /// Try to apply RYY(angle). Succeeds for Clifford angles and pi (Y+Y decomposition).
    ///
    /// # Errors
    /// Returns `Err` with a message if the angle is not a recognized Clifford angle.
    fn try_ryy(
        &mut self,
        angle: Angle64,
        pairs: &[(QubitId, QubitId)],
    ) -> Result<&mut Self, String>;

    /// Try to apply RXY1Q(theta, phi). Succeeds when the combination maps to a
    /// named Clifford (identity, X, Y, SX, `SXdg`, SY, `SYdg`).
    ///
    /// # Errors
    /// Returns `Err` with a message if the angle combination is not a recognized Clifford.
    fn try_rxy1q(
        &mut self,
        theta: Angle64,
        phi: Angle64,
        qubits: &[QubitId],
    ) -> Result<&mut Self, String>;

    /// Try to apply CRZ(angle). CRZ(0) is identity, while CRZ(pi) is the
    /// Clifford `(SZdg ⊗ I) . CZ`.
    ///
    /// # Errors
    /// Returns `Err` with a message if the angle is not 0 or pi.
    fn try_crz(
        &mut self,
        angle: Angle64,
        pairs: &[(QubitId, QubitId)],
    ) -> Result<&mut Self, String>;

    /// Try to apply U(theta, phi, lambda). Succeeds when the decomposition
    /// `RZ(phi) * RY(theta) * RZ(lambda)` consists entirely of Clifford gates.
    ///
    /// # Errors
    /// Returns `Err` with a message if any component is not a Clifford angle.
    fn try_u(
        &mut self,
        theta: Angle64,
        phi: Angle64,
        lambda: Angle64,
        qubits: &[QubitId],
    ) -> Result<&mut Self, String>;

    /// Try to apply RXXRYYRZZ(alpha, beta, gamma) = RXX(alpha) * RYY(beta) * RZZ(gamma).
    /// Succeeds when each component is a Clifford angle.
    ///
    /// # Errors
    /// Returns `Err` with a message if any component angle is not a Clifford angle.
    fn try_rxxryyrzz(
        &mut self,
        alpha: Angle64,
        beta: Angle64,
        gamma: Angle64,
        pairs: &[(QubitId, QubitId)],
    ) -> Result<&mut Self, String>;

    /// Try to apply U2q(before, interaction, after). Succeeds when all component
    /// U3 gates and the interaction (RXXRYYRZZ) are Clifford.
    ///
    /// # Errors
    /// Returns `Err` with a message if any component is not a Clifford angle.
    fn try_u2q(
        &mut self,
        before: [[Angle64; 3]; 2],
        interaction: [Angle64; 3],
        after: [[Angle64; 3]; 2],
        pairs: &[(QubitId, QubitId)],
    ) -> Result<&mut Self, String>;
}

impl<T: CliffordGateable> CliffordRotation for T {
    fn try_rz(&mut self, angle: Angle64, qubits: &[QubitId]) -> Result<&mut Self, String> {
        apply_rotation(self, GateType::RZ, angle, qubits)
    }

    fn try_rx(&mut self, angle: Angle64, qubits: &[QubitId]) -> Result<&mut Self, String> {
        apply_rotation(self, GateType::RX, angle, qubits)
    }

    fn try_ry(&mut self, angle: Angle64, qubits: &[QubitId]) -> Result<&mut Self, String> {
        apply_rotation(self, GateType::RY, angle, qubits)
    }

    fn try_rzz(
        &mut self,
        angle: Angle64,
        pairs: &[(QubitId, QubitId)],
    ) -> Result<&mut Self, String> {
        apply_two_qubit_rotation(self, GateType::RZZ, angle, pairs)
    }

    fn try_rxx(
        &mut self,
        angle: Angle64,
        pairs: &[(QubitId, QubitId)],
    ) -> Result<&mut Self, String> {
        apply_two_qubit_rotation(self, GateType::RXX, angle, pairs)
    }

    fn try_ryy(
        &mut self,
        angle: Angle64,
        pairs: &[(QubitId, QubitId)],
    ) -> Result<&mut Self, String> {
        apply_two_qubit_rotation(self, GateType::RYY, angle, pairs)
    }

    fn try_crz(
        &mut self,
        angle: Angle64,
        pairs: &[(QubitId, QubitId)],
    ) -> Result<&mut Self, String> {
        // CRZ(theta) = CX * RZ(-theta/2)_target * CX * RZ(theta/2)_target
        // Each RZ(theta/2) must be Clifford for this to work.
        let half = angle / 2u64;
        for &(_, target) in pairs {
            let target = &[target];
            self.try_rz(half, target)
                .map_err(|_| format!("CRZ({angle}) is not a Clifford rotation"))?;
        }
        self.cx(pairs);
        for &(_, target) in pairs {
            let target = &[target];
            self.try_rz(-half, target)
                .map_err(|_| format!("CRZ({angle}) is not a Clifford rotation"))?;
        }
        self.cx(pairs);
        Ok(self)
    }

    fn try_rxy1q(
        &mut self,
        theta: Angle64,
        phi: Angle64,
        qubits: &[QubitId],
    ) -> Result<&mut Self, String> {
        match pecos_core::try_simplify_rxy1q(theta, phi) {
            Some(clifford) => {
                dispatch_single_qubit_clifford(self, clifford, qubits)?;
                Ok(self)
            }
            None => Err(format!(
                "RXY1Q(theta={theta}, phi={phi}) is not a Clifford rotation"
            )),
        }
    }

    fn try_u(
        &mut self,
        theta: Angle64,
        phi: Angle64,
        lambda: Angle64,
        qubits: &[QubitId],
    ) -> Result<&mut Self, String> {
        let error =
            || format!("U(theta={theta}, phi={phi}, lambda={lambda}) is not a Clifford rotation");
        let decomposition = simplify_u_clifford([theta, phi, lambda]).ok_or_else(&error)?;

        dispatch_u_decomposition(self, decomposition, qubits).map_err(|_| error())?;
        Ok(self)
    }

    fn try_rxxryyrzz(
        &mut self,
        alpha: Angle64,
        beta: Angle64,
        gamma: Angle64,
        pairs: &[(QubitId, QubitId)],
    ) -> Result<&mut Self, String> {
        let error = || {
            format!(
                "RXXRYYRZZ(alpha={alpha}, beta={beta}, gamma={gamma}) is not a Clifford rotation"
            )
        };
        let decomposition = simplify_rxxryyrzz_clifford([alpha, beta, gamma]).ok_or_else(&error)?;

        dispatch_rxxryyrzz_decomposition(self, decomposition, pairs).map_err(|_| error())?;
        Ok(self)
    }

    fn try_u2q(
        &mut self,
        before: [[Angle64; 3]; 2],
        interaction: [Angle64; 3],
        after: [[Angle64; 3]; 2],
        pairs: &[(QubitId, QubitId)],
    ) -> Result<&mut Self, String> {
        // U2q = (U3(before[0]) x U3(before[1])) * RXXRYYRZZ(interaction) * (U3(after[0]) x U3(after[1]))
        // Applied right-to-left: after first, then interaction, then before.
        let err = || "U2q is not a Clifford rotation".to_string();
        let after = [
            simplify_u_clifford(after[0]).ok_or_else(&err)?,
            simplify_u_clifford(after[1]).ok_or_else(&err)?,
        ];
        let interaction = simplify_rxxryyrzz_clifford(interaction).ok_or_else(&err)?;
        let before = [
            simplify_u_clifford(before[0]).ok_or_else(&err)?,
            simplify_u_clifford(before[1]).ok_or_else(&err)?,
        ];

        for &(q0, q1) in pairs {
            dispatch_u_decomposition(self, after[0], &[q0]).map_err(|_| err())?;
            dispatch_u_decomposition(self, after[1], &[q1]).map_err(|_| err())?;
            dispatch_rxxryyrzz_decomposition(self, interaction, &[(q0, q1)]).map_err(|_| err())?;
            dispatch_u_decomposition(self, before[0], &[q0]).map_err(|_| err())?;
            dispatch_u_decomposition(self, before[1], &[q1]).map_err(|_| err())?;
        }
        Ok(self)
    }
}

fn simplify_u_clifford(angles: [Angle64; 3]) -> Option<[GateType; 3]> {
    let [theta, phi, lambda] = angles;
    Some([
        pecos_core::try_simplify_rotation(GateType::RZ, lambda)?,
        pecos_core::try_simplify_rotation(GateType::RY, theta)?,
        pecos_core::try_simplify_rotation(GateType::RZ, phi)?,
    ])
}

fn dispatch_u_decomposition<T: CliffordGateable>(
    sim: &mut T,
    decomposition: [GateType; 3],
    qubits: &[QubitId],
) -> Result<(), String> {
    for gate in decomposition {
        dispatch_single_qubit_clifford(sim, gate, qubits)?;
    }
    Ok(())
}

#[derive(Clone, Copy)]
enum TwoQubitCliffordDecomposition {
    Named(GateType),
    TensorPauli(GateType),
}

fn simplify_two_qubit_clifford(
    gate: GateType,
    angle: Angle64,
) -> Option<TwoQubitCliffordDecomposition> {
    pecos_core::try_simplify_rotation(gate, angle)
        .map(TwoQubitCliffordDecomposition::Named)
        .or_else(|| {
            pecos_core::half_turn_decomposition(gate, angle)
                .map(TwoQubitCliffordDecomposition::TensorPauli)
        })
}

fn simplify_rxxryyrzz_clifford(angles: [Angle64; 3]) -> Option<[TwoQubitCliffordDecomposition; 3]> {
    let [alpha, beta, gamma] = angles;
    Some([
        simplify_two_qubit_clifford(GateType::RXX, alpha)?,
        simplify_two_qubit_clifford(GateType::RYY, beta)?,
        simplify_two_qubit_clifford(GateType::RZZ, gamma)?,
    ])
}

fn dispatch_rxxryyrzz_decomposition<T: CliffordGateable>(
    sim: &mut T,
    decomposition: [TwoQubitCliffordDecomposition; 3],
    pairs: &[(QubitId, QubitId)],
) -> Result<(), String> {
    for component in decomposition {
        dispatch_two_qubit_decomposition(sim, component, pairs)?;
    }
    Ok(())
}

fn dispatch_two_qubit_decomposition<T: CliffordGateable>(
    sim: &mut T,
    decomposition: TwoQubitCliffordDecomposition,
    pairs: &[(QubitId, QubitId)],
) -> Result<(), String> {
    match decomposition {
        TwoQubitCliffordDecomposition::Named(clifford) => {
            dispatch_two_qubit_clifford(sim, clifford, pairs)
        }
        TwoQubitCliffordDecomposition::TensorPauli(pauli) => {
            for &(q0, q1) in pairs {
                dispatch_single_qubit_clifford(sim, pauli, &[q0])?;
                dispatch_single_qubit_clifford(sim, pauli, &[q1])?;
            }
            Ok(())
        }
    }
}

/// Single-qubit rotation dispatch: simplification to named Clifford.
fn apply_rotation<'a, T: CliffordGateable>(
    sim: &'a mut T,
    gate: GateType,
    angle: Angle64,
    qubits: &[QubitId],
) -> Result<&'a mut T, String> {
    if let Some(clifford) = pecos_core::try_simplify_rotation(gate, angle) {
        dispatch_single_qubit_clifford(sim, clifford, qubits)?;
        return Ok(sim);
    }
    Err(format!("{gate}({angle}) is not a Clifford rotation"))
}

/// Two-qubit rotation dispatch: simplification + half-turn decomposition.
fn apply_two_qubit_rotation<'a, T: CliffordGateable>(
    sim: &'a mut T,
    gate: GateType,
    angle: Angle64,
    pairs: &[(QubitId, QubitId)],
) -> Result<&'a mut T, String> {
    if let Some(decomposition) = simplify_two_qubit_clifford(gate, angle) {
        dispatch_two_qubit_decomposition(sim, decomposition, pairs)?;
        return Ok(sim);
    }
    Err(format!("{gate}({angle}) is not a Clifford rotation"))
}

/// Dispatch a named single-qubit Clifford `GateType` to the corresponding method.
fn dispatch_single_qubit_clifford<T: CliffordGateable>(
    sim: &mut T,
    gate: GateType,
    qubits: &[QubitId],
) -> Result<(), String> {
    match gate {
        GateType::I => {
            sim.identity(qubits);
        }
        GateType::X => {
            sim.x(qubits);
        }
        GateType::Y => {
            sim.y(qubits);
        }
        GateType::Z => {
            sim.z(qubits);
        }
        GateType::H => {
            sim.h(qubits);
        }
        GateType::SZ => {
            sim.sz(qubits);
        }
        GateType::SZdg => {
            sim.szdg(qubits);
        }
        GateType::SX => {
            sim.sx(qubits);
        }
        GateType::SXdg => {
            sim.sxdg(qubits);
        }
        GateType::SY => {
            sim.sy(qubits);
        }
        GateType::SYdg => {
            sim.sydg(qubits);
        }
        // T/Tdg are not Clifford gates -- they require ArbitraryRotationGateable
        GateType::T | GateType::Tdg => {
            return Err(format!(
                "{gate} is not a Clifford gate (requires ArbitraryRotationGateable)"
            ));
        }
        _ => {
            return Err(format!("{gate} is not supported by CliffordRotation"));
        }
    }
    Ok(())
}

/// Dispatch a named two-qubit Clifford `GateType` to the corresponding method.
fn dispatch_two_qubit_clifford<T: CliffordGateable>(
    sim: &mut T,
    gate: GateType,
    pairs: &[(QubitId, QubitId)],
) -> Result<(), String> {
    match gate {
        // Identity at angle=0 can appear for any rotation
        GateType::I => {
            // no-op
        }
        GateType::CX => {
            sim.cx(pairs);
        }
        GateType::CZ => {
            sim.cz(pairs);
        }
        GateType::SZZ => {
            sim.szz(pairs);
        }
        GateType::SZZdg => {
            sim.szzdg(pairs);
        }
        GateType::SXX => {
            sim.sxx(pairs);
        }
        GateType::SXXdg => {
            sim.sxxdg(pairs);
        }
        GateType::SYY => {
            sim.syy(pairs);
        }
        GateType::SYYdg => {
            sim.syydg(pairs);
        }
        _ => {
            return Err(format!("{gate} is not supported by CliffordRotation"));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sparse_stab::SparseStab;
    use pecos_core::{QubitId, qid};

    #[test]
    fn try_rz_clifford_angles() {
        let mut sim = SparseStab::new(1);
        assert!(sim.try_rz(Angle64::ZERO, &qid(0)).is_ok());
        assert!(sim.try_rz(Angle64::QUARTER_TURN, &qid(0)).is_ok());
        assert!(sim.try_rz(Angle64::HALF_TURN, &qid(0)).is_ok());
        assert!(sim.try_rz(Angle64::THREE_QUARTERS_TURN, &qid(0)).is_ok());
    }

    #[test]
    fn try_rz_non_clifford_fails() {
        let mut sim = SparseStab::new(1);
        assert!(sim.try_rz(Angle64::from_radians(0.123), &qid(0)).is_err());
    }

    #[test]
    fn try_rz_t_gate_fails_on_clifford_sim() {
        let mut sim = SparseStab::new(1);
        let eighth = Angle64::QUARTER_TURN / 2u64;
        // RZ(pi/4), projectively equivalent to T, is not Clifford and should fail.
        assert!(sim.try_rz(eighth, &qid(0)).is_err());
    }

    #[test]
    fn try_rx_clifford_angles() {
        let mut sim = SparseStab::new(1);
        assert!(sim.try_rx(Angle64::ZERO, &qid(0)).is_ok());
        assert!(sim.try_rx(Angle64::QUARTER_TURN, &qid(0)).is_ok());
        assert!(sim.try_rx(Angle64::HALF_TURN, &qid(0)).is_ok());
    }

    #[test]
    fn try_rx_non_clifford_fails() {
        let mut sim = SparseStab::new(1);
        assert!(sim.try_rx(Angle64::from_radians(0.5), &qid(0)).is_err());
    }

    #[test]
    fn try_ry_clifford_angles() {
        let mut sim = SparseStab::new(1);
        assert!(sim.try_ry(Angle64::ZERO, &qid(0)).is_ok());
        assert!(sim.try_ry(Angle64::HALF_TURN, &qid(0)).is_ok());
    }

    #[test]
    fn try_rzz_clifford_angles() {
        let mut sim = SparseStab::new(2);
        assert!(
            sim.try_rzz(Angle64::ZERO, &[(QubitId(0), QubitId(1))])
                .is_ok()
        );
        assert!(
            sim.try_rzz(Angle64::QUARTER_TURN, &[(QubitId(0), QubitId(1))])
                .is_ok()
        );
        assert!(
            sim.try_rzz(Angle64::THREE_QUARTERS_TURN, &[(QubitId(0), QubitId(1))])
                .is_ok()
        );
    }

    #[test]
    fn try_rzz_half_turn_decomposes() {
        let mut sim = SparseStab::new(2);
        // RZZ(pi) decomposes to Z+Z
        assert!(
            sim.try_rzz(Angle64::HALF_TURN, &[(QubitId(0), QubitId(1))])
                .is_ok()
        );
    }

    #[test]
    fn try_rxx_half_turn_decomposes() {
        let mut sim = SparseStab::new(2);
        assert!(
            sim.try_rxx(Angle64::HALF_TURN, &[(QubitId(0), QubitId(1))])
                .is_ok()
        );
    }

    #[test]
    fn try_ryy_half_turn_decomposes() {
        let mut sim = SparseStab::new(2);
        assert!(
            sim.try_ryy(Angle64::HALF_TURN, &[(QubitId(0), QubitId(1))])
                .is_ok()
        );
    }

    #[test]
    fn try_rzz_non_clifford_fails() {
        let mut sim = SparseStab::new(2);
        assert!(
            sim.try_rzz(Angle64::from_radians(0.5), &[(QubitId(0), QubitId(1))])
                .is_err()
        );
    }

    #[test]
    fn try_rxy1q_identity() {
        let mut sim = SparseStab::new(1);
        assert!(sim.try_rxy1q(Angle64::ZERO, Angle64::ZERO, &qid(0)).is_ok());
    }

    #[test]
    fn try_rxy1q_x_gate() {
        let mut sim = SparseStab::new(1);
        assert!(
            sim.try_rxy1q(Angle64::HALF_TURN, Angle64::ZERO, &qid(0))
                .is_ok()
        );
    }

    #[test]
    fn try_rxy1q_y_gate() {
        let mut sim = SparseStab::new(1);
        assert!(
            sim.try_rxy1q(Angle64::HALF_TURN, Angle64::QUARTER_TURN, &qid(0))
                .is_ok()
        );
    }

    #[test]
    fn try_rxy1q_sx_gate() {
        let mut sim = SparseStab::new(1);
        assert!(
            sim.try_rxy1q(Angle64::QUARTER_TURN, Angle64::ZERO, &qid(0))
                .is_ok()
        );
    }

    #[test]
    fn try_rxy1q_sxdg_gate() {
        let mut sim = SparseStab::new(1);
        assert!(
            sim.try_rxy1q(Angle64::THREE_QUARTERS_TURN, Angle64::ZERO, &qid(0))
                .is_ok()
        );
    }

    #[test]
    fn try_rxy1q_sy_gate() {
        let mut sim = SparseStab::new(1);
        assert!(
            sim.try_rxy1q(Angle64::QUARTER_TURN, Angle64::QUARTER_TURN, &qid(0))
                .is_ok()
        );
    }

    #[test]
    fn try_rxy1q_sydg_gate() {
        let mut sim = SparseStab::new(1);
        assert!(
            sim.try_rxy1q(Angle64::THREE_QUARTERS_TURN, Angle64::QUARTER_TURN, &qid(0))
                .is_ok()
        );
    }

    #[test]
    fn try_rxy1q_negated_axis() {
        let mut sim = SparseStab::new(1);
        // phi=pi (-X axis): equivalent to X
        assert!(
            sim.try_rxy1q(Angle64::HALF_TURN, Angle64::HALF_TURN, &qid(0))
                .is_ok()
        );
        // phi=3pi/2 (-Y axis): equivalent to Y
        assert!(
            sim.try_rxy1q(Angle64::HALF_TURN, Angle64::THREE_QUARTERS_TURN, &qid(0))
                .is_ok()
        );
    }

    #[test]
    fn try_rxy1q_non_clifford_fails() {
        let mut sim = SparseStab::new(1);
        // Non-Clifford theta
        assert!(
            sim.try_rxy1q(Angle64::from_radians(0.123), Angle64::ZERO, &qid(0))
                .is_err()
        );
        // Non-axis phi (pi/4 is not along X or Y)
        assert!(
            sim.try_rxy1q(Angle64::HALF_TURN, Angle64::QUARTER_TURN / 2u64, &qid(0))
                .is_err()
        );
    }

    // --- CRZ tests ---

    #[test]
    fn try_crz_zero_succeeds() {
        let mut sim = SparseStab::new(2);
        assert!(
            sim.try_crz(Angle64::ZERO, &[(QubitId(0), QubitId(1))])
                .is_ok()
        );
    }

    #[test]
    fn try_crz_pi_succeeds() {
        // CRZ(pi) decomposes to SZ, CX, SZdg, CX -- all Clifford
        let mut sim = SparseStab::new(2);
        assert!(
            sim.try_crz(Angle64::HALF_TURN, &[(QubitId(0), QubitId(1))])
                .is_ok()
        );
    }

    #[test]
    fn try_crz_neg_pi_succeeds() {
        let mut sim = SparseStab::new(2);
        assert!(
            sim.try_crz(-Angle64::HALF_TURN, &[(QubitId(0), QubitId(1))])
                .is_ok()
        );
    }

    #[test]
    fn try_crz_quarter_turn_fails() {
        let mut sim = SparseStab::new(2);
        // CRZ(pi/2) requires an RZ(pi/4), projectively T and not Clifford.
        assert!(
            sim.try_crz(Angle64::QUARTER_TURN, &[(QubitId(0), QubitId(1))])
                .is_err()
        );
    }

    #[test]
    fn try_crz_non_clifford_fails() {
        let mut sim = SparseStab::new(2);
        assert!(
            sim.try_crz(Angle64::from_radians(0.5), &[(QubitId(0), QubitId(1))])
                .is_err()
        );
    }

    // --- U gate tests ---

    #[test]
    fn try_u_identity() {
        // U(0, 0, 0) = I
        let mut sim = SparseStab::new(1);
        assert!(
            sim.try_u(Angle64::ZERO, Angle64::ZERO, Angle64::ZERO, &qid(0))
                .is_ok()
        );
    }

    #[test]
    fn try_u_z_gate() {
        // U(0, 0, pi) = RZ(0) * RY(0) * RZ(pi) = Z
        let mut sim = SparseStab::new(1);
        assert!(
            sim.try_u(Angle64::ZERO, Angle64::ZERO, Angle64::HALF_TURN, &qid(0))
                .is_ok()
        );
    }

    #[test]
    fn try_u_x_gate() {
        // U(pi, 0, pi) = RZ(0) * RY(pi) * RZ(pi) = Y * Z = iX -> X up to phase
        let mut sim = SparseStab::new(1);
        assert!(
            sim.try_u(
                Angle64::HALF_TURN,
                Angle64::ZERO,
                Angle64::HALF_TURN,
                &qid(0)
            )
            .is_ok()
        );
    }

    #[test]
    fn try_u_all_clifford_angles() {
        // U(pi/2, pi/2, pi) should succeed -- all components are Clifford
        let mut sim = SparseStab::new(1);
        assert!(
            sim.try_u(
                Angle64::QUARTER_TURN,
                Angle64::QUARTER_TURN,
                Angle64::HALF_TURN,
                &qid(0),
            )
            .is_ok()
        );
    }

    #[test]
    fn try_u_non_clifford_theta_fails() {
        let mut sim = SparseStab::new(1);
        assert!(
            sim.try_u(
                Angle64::from_radians(0.123),
                Angle64::ZERO,
                Angle64::ZERO,
                &qid(0),
            )
            .is_err()
        );
    }

    #[test]
    fn try_u_non_clifford_lambda_fails() {
        let mut sim = SparseStab::new(1);
        assert!(
            sim.try_u(
                Angle64::ZERO,
                Angle64::ZERO,
                Angle64::from_radians(0.5),
                &qid(0),
            )
            .is_err()
        );
    }

    #[test]
    fn try_u_failure_is_atomic() {
        let mut sim = SparseStab::new(1);
        let before = (sim.stab_tableau(), sim.destab_tableau());

        assert!(
            sim.try_u(
                Angle64::from_radians(0.5),
                Angle64::ZERO,
                Angle64::QUARTER_TURN,
                &qid(0),
            )
            .is_err()
        );
        assert_eq!((sim.stab_tableau(), sim.destab_tableau()), before);
    }

    #[test]
    fn try_rxxryyrzz_failure_is_atomic() {
        let mut sim = SparseStab::new(2);
        let before = (sim.stab_tableau(), sim.destab_tableau());

        assert!(
            sim.try_rxxryyrzz(
                Angle64::QUARTER_TURN,
                Angle64::from_radians(0.5),
                Angle64::ZERO,
                &[(QubitId(0), QubitId(1))],
            )
            .is_err()
        );
        assert_eq!((sim.stab_tableau(), sim.destab_tableau()), before);
    }

    #[test]
    fn try_u2q_failure_is_atomic() {
        let mut sim = SparseStab::new(2);
        let before_tableau = (sim.stab_tableau(), sim.destab_tableau());
        let zero_u = [Angle64::ZERO; 3];

        assert!(
            sim.try_u2q(
                [zero_u; 2],
                [Angle64::ZERO, Angle64::from_radians(0.5), Angle64::ZERO,],
                [
                    [Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN,],
                    zero_u,
                ],
                &[(QubitId(0), QubitId(1))],
            )
            .is_err()
        );
        assert_eq!((sim.stab_tableau(), sim.destab_tableau()), before_tableau);
    }
}
