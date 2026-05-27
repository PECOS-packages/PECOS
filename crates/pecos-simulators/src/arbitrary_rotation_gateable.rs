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
    /// By default, this is implemented in terms of `rz` and `ry` gates.
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
        self.rz(lambda, qubits).ry(theta, qubits).rz(phi, qubits)
    }

    /// Applies an X-Y plane rotation gate with a specified angle and axis.
    ///
    /// By default, this is implemented in terms of `rz` and `ry` gates.
    ///
    /// # Parameters
    /// - `theta`: The rotation angle.
    /// - `phi`: The axis angle.
    /// - `qubits`: The target qubit indices.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn r1xy(&mut self, theta: Angle64, phi: Angle64, qubits: &[QubitId]) -> &mut Self {
        self.rz(-phi + Angle64::QUARTER_TURN, qubits)
            .ry(theta, qubits)
            .rz(phi - Angle64::QUARTER_TURN, qubits)
    }

    /// Applies the T gate (pi/8 rotation around Z-axis).
    ///
    /// # Parameters
    /// - `qubits`: The target qubit indices.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn t(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.rz(Angle64::QUARTER_TURN / 2u64, qubits)
    }

    /// Applies the T^dagger (T-dagger) gate (-pi/8 rotation around Z-axis).
    ///
    /// # Parameters
    /// - `qubits`: The target qubit indices.
    ///
    /// # Returns
    /// A mutable reference to `Self` for method chaining.
    #[inline]
    fn tdg(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.rz(-(Angle64::QUARTER_TURN / 2u64), qubits)
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
