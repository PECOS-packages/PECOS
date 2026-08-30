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

use crate::CliffordGateable;
use pecos_core::{Angle64, QubitId};
use smallvec::SmallVec;

/// Stack-allocated qubit buffer for small batches (up to 8 qubits).
type QubitBuf = SmallVec<[QubitId; 8]>;

/// A trait for implementing arbitrary rotation gates on a quantum system.
///
/// This trait extends [`CliffordGateable`] and provides methods for applying
/// single-qubit and two-qubit rotation gates around various axes.
///
/// # Slice-based API
/// All methods take `&[QubitId]` slices, allowing both single-qubit and batch operations:
///
/// - Single-qubit gates: `sim.rx(theta, &[QubitId(0)])` or batch: `sim.rx(theta, &[q0, q1, q2])`
/// - Two-qubit gates: `sim.rzz(theta, &[(q0, q1)])` or batch: `sim.rzz(theta, &[(q0, q1), (q2, q3)])`
///
/// # Note
/// Most of the methods in this trait have default implementations. However, the
/// following methods are the minimum methods that must be implemented to utilize the trait:
/// - `rx`: Rotation around the X-axis.
/// - `rz`: Rotation around the Z-axis.
/// - `rzz`: Two-qubit rotation around the ZZ-axis.
pub trait ArbitraryRotationGateable: CliffordGateable {
    /// Applies a rotation around the X-axis by an angle `theta`.
    ///
    /// Gate RX(theta) = exp(-i theta X/2) = cos(theta/2) I - i*sin(theta/2) X
    ///
    /// RX(theta) = [[cos(theta/2), -i*sin(theta/2)],
    ///              [-i*sin(theta/2), cos(theta/2)]]
    ///
    /// # Parameters
    /// - `theta`: The rotation angle.
    /// - `qubits`: The target qubit indices.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self;

    /// Applies a rotation around the Y-axis by an angle `theta`.
    ///
    /// Gate RY(theta) = exp(-i theta Y/2) = cos(theta/2) I - i*sin(theta/2) Y
    ///
    /// RY(theta) = [[cos(theta/2), -sin(theta/2)],
    ///              [sin(theta/2), cos(theta/2)]]
    ///
    /// By default, this is implemented in terms of `szdg`, `rx`, and `sz` gates:
    /// RY(theta) = S * RX(theta) * S^dagger
    ///
    /// # Parameters
    /// - `theta`: The rotation angle.
    /// - `qubits`: The target qubit indices.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn ry(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        self.szdg(qubits).rx(theta, qubits).sz(qubits)
    }

    /// Applies a rotation around the Z-axis by an angle `theta`.
    ///
    /// Gate RZ(theta) = exp(-i theta Z/2) = cos(theta/2) I - i*sin(theta/2) Z
    ///
    /// RZ(theta) = [[cos(theta/2)-i*sin(theta/2), 0],
    ///              [0, cos(theta/2)+i*sin(theta/2)]]
    ///
    /// This is the symmetric rotation convention. The phase gate
    /// `P(theta) = U(0, 0, theta) = diag(1, exp(i*theta))` is related by
    /// `RZ(theta) = exp(-i*theta/2) P(theta)`; the scalar is significant to
    /// amplitude-carrying representations.
    ///
    /// # Parameters
    /// - `theta`: The rotation angle.
    /// - `qubits`: The target qubit indices.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self;

    /// Applies a general single-qubit unitary U(theta, phi, lambda) gate.
    ///
    /// `U1_3` = [[cos(theta/2), -e^(i*lambda)sin(theta/2)],
    ///           [e^(i*phi)sin(theta/2), e^(i(lambda+phi))cos(theta/2)]]
    ///
    /// Its exact relationship to the symmetric rotation family is
    /// `U(theta, phi, lambda) = exp(i*(phi+lambda)/2) * RZ(phi) * RY(theta) * RZ(lambda)`.
    /// The default decomposition carries that scalar through
    /// [`Self::apply_global_phase`].
    ///
    /// # Parameters
    /// - `theta`: The rotation angle around the Y-axis.
    /// - `phi`: The first Z-axis rotation angle.
    /// - `lambda`: The second Z-axis rotation angle.
    /// - `qubits`: The target qubit indices.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn u(
        &mut self,
        theta: Angle64,
        phi: Angle64,
        lambda: Angle64,
        qubits: &[QubitId],
    ) -> &mut Self {
        // Use the signed principal values independently. Adding the fixed-point
        // angles before halving can wrap at 2*pi and lose a compensating pi.
        let phase =
            Angle64::from_radians((lambda.to_radians_signed() + phi.to_radians_signed()) / 2.0);
        self.rz(lambda, qubits)
            .ry(theta, qubits)
            .rz(phi, qubits)
            .apply_global_phase(phase, qubits)
    }

    /// Applies an X-Y plane rotation gate with a specified angle and axis.
    ///
    /// `RXY1Q(theta, phi) = exp(-i*theta*(cos(phi) X + sin(phi) Y)/2)`, with matrix
    /// `[[cos(theta/2), -i*exp(-i*phi)*sin(theta/2)],
    ///   [-i*exp(i*phi)*sin(theta/2), cos(theta/2)]]`.
    ///
    /// The default `RZ(pi/2-phi)`, `RY(theta)`, `RZ(phi-pi/2)` decomposition
    /// is phase-exact because the two Z angles sum to zero.
    ///
    /// # Parameters
    /// - `theta`: The rotation angle.
    /// - `phi`: The axis angle.
    /// - `qubits`: The target qubit indices.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn rxy1q(&mut self, theta: Angle64, phi: Angle64, qubits: &[QubitId]) -> &mut Self {
        self.rz(-phi + Angle64::QUARTER_TURN, qubits)
            .ry(theta, qubits)
            .rz(phi - Angle64::QUARTER_TURN, qubits)
    }

    /// Applies the scalar `exp(i * phase)` once for every target qubit.
    ///
    /// The default is a no-op, which is correct for representations where global
    /// phase is unobservable, such as density matrices, measurement-only mocks,
    /// foreign interfaces without a global-phase operation, and compile-only
    /// resource analyzers. Amplitude-exposing simulators must override this hook.
    ///
    /// # Parameters
    /// - `phase`: The phase angle for one scalar application.
    /// - `qubits`: The targets whose gate applications each contribute the scalar.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn apply_global_phase(&mut self, _phase: Angle64, _qubits: &[QubitId]) -> &mut Self {
        self
    }

    /// Applies the conventional T gate, `diag(1, exp(i*pi/4))`.
    ///
    /// This differs from `RZ(pi/4)` by the global phase `exp(i*pi/8)`.
    ///
    /// # Parameters
    /// - `qubits`: The target qubit indices.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn t(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.rz(Angle64::QUARTER_TURN / 2u64, qubits)
            .apply_global_phase(Angle64::QUARTER_TURN / 4u64, qubits)
    }

    /// Applies the conventional T-dagger gate, `diag(1, exp(-i*pi/4))`.
    ///
    /// This differs from `RZ(-pi/4)` by the global phase `exp(-i*pi/8)`.
    ///
    /// # Parameters
    /// - `qubits`: The target qubit indices.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn tdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.rz(-(Angle64::QUARTER_TURN / 2u64), qubits)
            .apply_global_phase(-(Angle64::QUARTER_TURN / 4u64), qubits)
    }

    /// Applies a two-qubit XX rotation gate.
    ///
    /// Apply RXX(theta) = exp(-i theta XX/2) gate
    ///
    /// By default, this is implemented in terms of Hadamard (`h`) and ZZ rotation (`rzz`) gates.
    ///
    /// # Parameters
    /// - `theta`: The rotation angle.
    /// - `pairs`: Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn rxx(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: QubitBuf = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: QubitBuf = pairs.iter().map(|&(_, q2)| q2).collect();
        self.h(&q1s).h(&q2s).rzz(theta, pairs).h(&q1s).h(&q2s)
    }

    /// Apply RYY(theta) = exp(-i theta YY/2) gate, which implements evolution under the YY coupling
    /// between two qubits.
    ///
    /// The YY coupling generates entanglement between qubits through the Y tensor Y interaction.
    /// For example, RYY(pi/2) transforms basis states as follows:
    /// - |00> -> (|00> - i|11>)/sqrt(2)
    /// - |11> -> (|11> - i|00>)/sqrt(2)
    /// - |01> -> (|01> + i|10>)/sqrt(2)
    /// - |10> -> (|10> + i|01>)/sqrt(2)
    ///
    /// By default, this is implemented in terms of SX and ZZ rotation (`rzz`) gates.
    ///
    /// # Parameters
    /// - `theta`: The rotation angle.
    /// - `pairs`: Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn ryy(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let q1s: QubitBuf = pairs.iter().map(|&(q1, _)| q1).collect();
        let q2s: QubitBuf = pairs.iter().map(|&(_, q2)| q2).collect();
        self.sx(&q1s)
            .sx(&q2s)
            .rzz(theta, pairs)
            .sxdg(&q1s)
            .sxdg(&q2s)
    }

    /// Apply RZZ(theta) = exp(-i theta ZZ/2) gate, implementing evolution under the ZZ coupling
    /// between two qubits.
    ///
    /// The ZZ coupling represents a phase interaction between qubits that is diagonal in the
    /// computational basis. It is a key component in many quantum algorithms and appears naturally
    /// in various physical implementations. The operation adds a theta/2 phase when the qubits have
    /// the same value, and -theta/2 phase when they differ.
    ///
    /// The action on basis states is:
    /// - |00> -> exp(-i*theta/2)|00>
    /// - |11> -> exp(-i*theta/2)|11>
    /// - |01> -> exp(i*theta/2)|01>
    /// - |10> -> exp(i*theta/2)|10>
    ///
    /// The matrix:
    /// ```text
    /// RZZ(theta) = [[e^(-i*theta/2),     0,          0,          0        ],
    ///               [0,          e^(i*theta/2),      0,          0        ],
    ///               [0,             0,       e^(i*theta/2),      0        ],
    ///               [0,             0,          0,       e^(-i*theta/2)   ]]
    /// ```
    ///
    /// # Parameters
    /// - `theta`: The rotation angle.
    /// - `pairs`: Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self;

    /// Applies a composite rotation gate using RXX, RYY, and RZZ gates.
    ///
    /// # Parameters
    /// - `theta`: The rotation angle for the RXX gate.
    /// - `phi`: The rotation angle for the RYY gate.
    /// - `lambda`: The rotation angle for the RZZ gate.
    /// - `pairs`: Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    ///
    #[inline]
    fn rxxryyrzz(
        &mut self,
        theta: Angle64,
        phi: Angle64,
        lambda: Angle64,
        pairs: &[(QubitId, QubitId)],
    ) -> &mut Self {
        self.rxx(theta, pairs).ryy(phi, pairs).rzz(lambda, pairs)
    }

    /// Applies a controlled-RZ rotation: target qubit gets RZ(theta) when control = |1>.
    ///
    /// `CRZ(theta) = block-diag(I, RZ(theta)) = diag(1, 1, exp(-i*theta/2), exp(i*theta/2))`.
    ///
    /// Default 2q-minimal decomposition (1 RZZ + 1 single-qubit RZ on the
    /// target): `CRZ(theta) = (I o RZ(theta/2)) . RZZ(-theta/2)`.
    /// Verified: with the trait's `RZ = exp(-i*theta/2*Z)` and `RZZ =
    /// exp(-i*theta/2*Z*Z)` conventions, the product on the c=0 sector
    /// gives `RZ(theta/2) . exp(i*theta/4*I) = I` up to global phase, and
    /// on c=1 (where ZZ acts as -Z on target) gives `RZ(theta/2) . X .
    /// RZ(theta/2) . X = RZ(theta)` -- i.e. the convention-1 controlled
    /// rotation. The non-PECOS-prefactor convention requires no extra
    /// RZ on the control.
    ///
    /// # Parameters
    /// - `theta`: The rotation angle on the target.
    /// - `pairs`: Pairs of qubit indices `[(control, target), ...]`.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn crz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        // Half-angle first, THEN negate -- `Angle<T>` is a wrapping fraction
        // of a full turn (modulo 2pi), so `-theta / 2` would halve the wrapped
        // 2*pi - theta and produce pi - theta/2, not -theta/2.
        let half = theta / 2u64;
        let targets: QubitBuf = pairs.iter().map(|&(_, t)| t).collect();
        self.rzz(-half, pairs).rz(half, &targets)
    }

    /// Applies a controlled-RX rotation: target qubit gets RX(theta) when control = |1>.
    ///
    /// Default decomposition: `CRX(theta) = (I o H) . CRZ(theta) . (I o H)`,
    /// using `H.Z.H = X` so the c=1 sector applies `H.RZ(theta).H = RX(theta)`.
    /// Same 2q cost as `crz` (1 RZZ).
    ///
    /// # Parameters
    /// - `theta`: The rotation angle on the target.
    /// - `pairs`: Pairs of qubit indices `[(control, target), ...]`.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn crx(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let targets: QubitBuf = pairs.iter().map(|&(_, t)| t).collect();
        self.h(&targets).crz(theta, pairs).h(&targets)
    }

    /// Applies a controlled-RY rotation: target qubit gets RY(theta) when control = |1>.
    ///
    /// Default decomposition: `CRY(theta) = (I o S.H) . CRZ(theta) . (I o H.Sdg)`,
    /// using `S.X.Sdg = Y` (so `S.Rx.Sdg = Ry`) and `H.Rz.H = Rx`, giving
    /// `S.H.RZ.H.Sdg = RY`. Same 2q cost as `crz` (1 RZZ).
    ///
    /// # Parameters
    /// - `theta`: The rotation angle on the target.
    /// - `pairs`: Pairs of qubit indices `[(control, target), ...]`.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn cry(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        let targets: QubitBuf = pairs.iter().map(|&(_, t)| t).collect();
        self.szdg(&targets)
            .h(&targets)
            .crz(theta, pairs)
            .h(&targets)
            .sz(&targets)
    }

    /// Applies a general 2-qubit unitary via KAK decomposition:
    /// U = (U3(before[0]) x U3(before[1])) * RXXRYYRZZ(interaction) * (U3(after[0]) x U3(after[1]))
    ///
    /// # Parameters
    /// - `before`: U3(theta, phi, lambda) parameters for each qubit, applied after the interaction
    /// - `interaction`: [alpha, beta, gamma] for RXXRYYRZZ
    /// - `after`: U3(theta, phi, lambda) parameters for each qubit, applied before the interaction
    /// - `pairs`: Pairs of qubit indices: `[(q0, q1), (q2, q3), ...]`
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn u2q(
        &mut self,
        before: [[Angle64; 3]; 2],
        interaction: [Angle64; 3],
        after: [[Angle64; 3]; 2],
        pairs: &[(QubitId, QubitId)],
    ) -> &mut Self {
        for &(q0, q1) in pairs {
            let q0s = &[q0];
            let q1s = &[q1];
            // Apply after (right-most) single-qubit gates first
            self.u(after[0][0], after[0][1], after[0][2], q0s);
            self.u(after[1][0], after[1][1], after[1][2], q1s);
            // Interaction
            self.rxxryyrzz(interaction[0], interaction[1], interaction[2], &[(q0, q1)]);
            // Apply before (left-most) single-qubit gates last
            self.u(before[0][0], before[0][1], before[0][2], q0s);
            self.u(before[1][0], before[1][1], before[1][2], q1s);
        }
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MeasurementResult, QuantumSimulator};

    #[derive(Default)]
    struct RecordingSimulator {
        rz_calls: Vec<(Angle64, Vec<QubitId>)>,
        global_phase_calls: Vec<(Angle64, Vec<QubitId>)>,
    }

    impl QuantumSimulator for RecordingSimulator {
        fn reset(&mut self) -> &mut Self {
            self
        }

        fn num_qubits(&self) -> usize {
            3
        }
    }

    impl CliffordGateable for RecordingSimulator {
        fn sz(&mut self, _qubits: &[QubitId]) -> &mut Self {
            self
        }

        fn h(&mut self, _qubits: &[QubitId]) -> &mut Self {
            self
        }

        fn cx(&mut self, _pairs: &[(QubitId, QubitId)]) -> &mut Self {
            self
        }

        fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
            qubits
                .iter()
                .map(|_| MeasurementResult {
                    outcome: false,
                    is_deterministic: true,
                })
                .collect()
        }
    }

    impl ArbitraryRotationGateable for RecordingSimulator {
        fn rx(&mut self, _theta: Angle64, _qubits: &[QubitId]) -> &mut Self {
            self
        }

        fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
            self.rz_calls.push((theta, qubits.to_vec()));
            self
        }

        fn apply_global_phase(&mut self, phase: Angle64, qubits: &[QubitId]) -> &mut Self {
            self.global_phase_calls.push((phase, qubits.to_vec()));
            self
        }

        fn rzz(&mut self, _theta: Angle64, _pairs: &[(QubitId, QubitId)]) -> &mut Self {
            self
        }
    }

    #[test]
    fn default_t_and_tdg_each_emit_one_rz_call() {
        let targets = [QubitId(0), QubitId(2)];
        let mut recorder = RecordingSimulator::default();

        recorder.t(&targets);
        assert_eq!(
            recorder.rz_calls,
            vec![(Angle64::QUARTER_TURN / 2u64, targets.to_vec())]
        );

        recorder.rz_calls.clear();
        recorder.tdg(&targets);
        assert_eq!(
            recorder.rz_calls,
            vec![(-(Angle64::QUARTER_TURN / 2u64), targets.to_vec())]
        );
    }

    fn assert_default_u_phase_gate(
        lambda: Angle64,
        expected_half_phase: Angle64,
        expected_diagonal: num_complex::Complex32,
    ) {
        use crate::StateVecSoA32;
        use num_complex::Complex32;

        let target = [QubitId(0)];
        let mut sim = StateVecSoA32::new(1);
        sim.u(Angle64::ZERO, Angle64::ZERO, lambda, &target);
        let zero_column = [sim.get_amplitude(0), sim.get_amplitude(1)];
        assert!((zero_column[0] - Complex32::new(1.0, 0.0)).norm() < 2e-6);
        assert!(zero_column[1].norm() < 2e-6);

        sim.reset().x(&target);
        sim.u(Angle64::ZERO, Angle64::ZERO, lambda, &target);
        let one_column = [sim.get_amplitude(0), sim.get_amplitude(1)];
        assert!(one_column[0].norm() < 2e-6);
        assert!((one_column[1] - expected_diagonal).norm() < 2e-6);

        // Even the zero-angle case must route through the scalar hook. This is
        // the mutation witness distinguishing the exact default decomposition
        // from the old phase-inexact three-rotation implementation.
        let mut recorder = RecordingSimulator::default();
        recorder.u(Angle64::ZERO, Angle64::ZERO, lambda, &target);
        assert_eq!(recorder.global_phase_calls.len(), 1);
        let (actual_phase, actual_targets) = &recorder.global_phase_calls[0];
        assert!(actual_phase.abs_diff_eq_radians(&expected_half_phase, 1e-14));
        assert_eq!(actual_targets, &target);
    }

    #[test]
    fn default_u_000_is_exact_identity() {
        assert_default_u_phase_gate(
            Angle64::ZERO,
            Angle64::ZERO,
            num_complex::Complex32::new(1.0, 0.0),
        );
    }

    #[test]
    fn default_u_00_pi_over_4_is_exact_t() {
        assert_default_u_phase_gate(
            Angle64::QUARTER_TURN / 2u64,
            Angle64::QUARTER_TURN / 4u64,
            num_complex::Complex32::new(
                std::f32::consts::FRAC_1_SQRT_2,
                std::f32::consts::FRAC_1_SQRT_2,
            ),
        );
    }

    #[test]
    fn default_u_00_pi_over_2_is_exact_sz() {
        assert_default_u_phase_gate(
            Angle64::QUARTER_TURN,
            Angle64::QUARTER_TURN / 2u64,
            num_complex::Complex32::new(0.0, 1.0),
        );
    }

    #[test]
    fn default_u_00_pi_is_exact_z() {
        assert_default_u_phase_gate(
            Angle64::HALF_TURN,
            Angle64::QUARTER_TURN,
            num_complex::Complex32::new(-1.0, 0.0),
        );
    }

    #[test]
    fn default_u_matches_documented_matrix() {
        use crate::StateVecSoA32;
        use num_complex::Complex64;

        let theta = Angle64::from_radians(0.73);
        let phi = Angle64::from_radians(-0.41);
        let lambda = Angle64::from_radians(1.17);
        let theta_rad = theta.to_radians_signed();
        let phi_rad = phi.to_radians_signed();
        let lambda_rad = lambda.to_radians_signed();
        let c = (theta_rad / 2.0).cos();
        let s = (theta_rad / 2.0).sin();
        let expected_columns = [
            [Complex64::new(c, 0.0), Complex64::from_polar(s, phi_rad)],
            [
                -Complex64::from_polar(s, lambda_rad),
                Complex64::from_polar(c, lambda_rad + phi_rad),
            ],
        ];

        for (basis, expected) in expected_columns.iter().enumerate() {
            let target = [QubitId(0)];
            let mut sim = StateVecSoA32::new(1);
            if basis == 1 {
                sim.x(&target);
            }
            sim.u(theta, phi, lambda, &target);
            for (row, expected_entry) in expected.iter().enumerate() {
                let actual = sim.get_amplitude(row);
                let actual = Complex64::new(f64::from(actual.re), f64::from(actual.im));
                assert!(
                    (actual - expected_entry).norm() < 3e-6,
                    "column={basis}, row={row}: expected {expected_entry}, got {actual}"
                );
            }
        }
    }

    #[test]
    fn default_rxy1q_matches_documented_matrix_exactly() {
        use crate::StateVecSoA32;
        use num_complex::Complex64;

        let theta = Angle64::from_radians(0.73);
        let phi = Angle64::from_radians(0.31);
        let theta_rad = theta.to_radians_signed();
        let phi_rad = phi.to_radians_signed();
        let c = (theta_rad / 2.0).cos();
        let s = (theta_rad / 2.0).sin();
        let minus_i = Complex64::new(0.0, -1.0);
        let expected_columns = [
            [
                Complex64::new(c, 0.0),
                minus_i * Complex64::from_polar(s, phi_rad),
            ],
            [
                minus_i * Complex64::from_polar(s, -phi_rad),
                Complex64::new(c, 0.0),
            ],
        ];

        for (basis, expected) in expected_columns.iter().enumerate() {
            let target = [QubitId(0)];
            let mut sim = StateVecSoA32::new(1);
            if basis == 1 {
                sim.x(&target);
            }
            sim.rxy1q(theta, phi, &target);
            for (row, expected_entry) in expected.iter().enumerate() {
                let actual = sim.get_amplitude(row);
                let actual = Complex64::new(f64::from(actual.re), f64::from(actual.im));
                assert!(
                    (actual - expected_entry).norm() < 3e-6,
                    "column={basis}, row={row}: expected {expected_entry}, got {actual}"
                );
            }
        }
    }
}
