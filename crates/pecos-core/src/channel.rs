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

//! Symbolic quantum-channel namespace.
//!
//! Constructors in this module return [`ChannelExpr`]. Use this namespace for
//! physical noise, open-system maps, and other CPTP processes:
//!
//! ```
//! use pecos_core::channel::*;
//!
//! let noise = Depolarizing(0.001, 0) & BitFlip(0.01, 1);
//! ```
//!
//! Ideal gates can be lifted into this level with [`from_gate`]. Unitary
//! operations and ideal gates are not noise, but they are valid channels when a
//! channel-level expression is needed.

use crate::op::Op;
use crate::qubit_support::overlapping_qubits;
use crate::{GateExpr, PauliString, QubitId, UnitaryRep, op};
use std::ops::{BitAnd, Mul};

pub use crate::op::ChannelExpr;

fn channel_from_op(op: Op) -> ChannelExpr {
    op.into_channel()
}

/// Lifts a unitary expression to the channel level.
#[must_use]
pub fn from_unitary(unitary: impl Into<UnitaryRep>) -> ChannelExpr {
    ChannelExpr::Unitary(unitary.into())
}

/// Lifts an ideal gate expression to the channel level.
#[must_use]
pub fn from_gate(gate: impl Into<GateExpr>) -> ChannelExpr {
    ChannelExpr::Gate(gate.into())
}

impl From<UnitaryRep> for ChannelExpr {
    fn from(unitary: UnitaryRep) -> Self {
        ChannelExpr::Unitary(unitary)
    }
}

impl From<PauliString> for ChannelExpr {
    fn from(pauli: PauliString) -> Self {
        ChannelExpr::Unitary(UnitaryRep::from(pauli))
    }
}

impl From<GateExpr> for ChannelExpr {
    fn from(gate: GateExpr) -> Self {
        ChannelExpr::Gate(gate)
    }
}

/// Single-qubit depolarizing channel.
#[allow(non_snake_case)]
#[must_use]
pub fn Depolarizing(p: f64, qubit: impl Into<QubitId>) -> ChannelExpr {
    channel_from_op(op::Depolarizing(p, qubit))
}

/// Dephasing channel.
#[allow(non_snake_case)]
#[must_use]
pub fn Dephasing(p: f64, qubit: impl Into<QubitId>) -> ChannelExpr {
    channel_from_op(op::Dephasing(p, qubit))
}

/// Bit-flip channel.
#[allow(non_snake_case)]
#[must_use]
pub fn BitFlip(p: f64, qubit: impl Into<QubitId>) -> ChannelExpr {
    channel_from_op(op::BitFlip(p, qubit))
}

/// Bit-phase-flip channel.
#[allow(non_snake_case)]
#[must_use]
pub fn BitPhaseFlip(p: f64, qubit: impl Into<QubitId>) -> ChannelExpr {
    channel_from_op(op::BitPhaseFlip(p, qubit))
}

/// General single-qubit Pauli channel.
#[allow(non_snake_case)]
#[must_use]
pub fn PauliChannel(px: f64, py: f64, pz: f64, qubit: impl Into<QubitId>) -> ChannelExpr {
    channel_from_op(op::PauliChannel(px, py, pz, qubit))
}

/// Two-qubit depolarizing channel.
#[allow(non_snake_case)]
#[must_use]
pub fn Depolarizing2(p: f64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> ChannelExpr {
    channel_from_op(op::Depolarizing2(p, q0, q1))
}

/// Amplitude damping channel.
#[allow(non_snake_case)]
#[must_use]
pub fn AmplitudeDamping(gamma: f64, qubit: impl Into<QubitId>) -> ChannelExpr {
    channel_from_op(op::AmplitudeDamping(gamma, qubit))
}

/// Phase damping channel.
#[allow(non_snake_case)]
#[must_use]
pub fn PhaseDamping(lambda: f64, qubit: impl Into<QubitId>) -> ChannelExpr {
    channel_from_op(op::PhaseDamping(lambda, qubit))
}

/// Erasure channel.
#[allow(non_snake_case)]
#[must_use]
pub fn Erasure(prob: f64, qubit: impl Into<QubitId>) -> ChannelExpr {
    channel_from_op(op::Erasure(prob, qubit))
}

/// Leakage channel.
#[allow(non_snake_case)]
#[must_use]
pub fn Leakage(rate: f64, qubit: impl Into<QubitId>) -> ChannelExpr {
    channel_from_op(op::Leakage(rate, qubit))
}

impl BitAnd for ChannelExpr {
    type Output = ChannelExpr;

    fn bitand(self, rhs: ChannelExpr) -> ChannelExpr {
        let overlap = overlapping_qubits(self.qubits(), rhs.qubits());
        assert!(
            overlap.is_empty(),
            "tensor product requires disjoint channel support; overlapping qubits: {overlap:?}"
        );
        ChannelExpr::Tensor(vec![self, rhs])
    }
}

impl BitAnd<&ChannelExpr> for ChannelExpr {
    type Output = ChannelExpr;

    fn bitand(self, rhs: &ChannelExpr) -> ChannelExpr {
        self & rhs.clone()
    }
}

impl BitAnd<ChannelExpr> for &ChannelExpr {
    type Output = ChannelExpr;

    fn bitand(self, rhs: ChannelExpr) -> ChannelExpr {
        self.clone() & rhs
    }
}

impl BitAnd<&ChannelExpr> for &ChannelExpr {
    type Output = ChannelExpr;

    fn bitand(self, rhs: &ChannelExpr) -> ChannelExpr {
        self.clone() & rhs.clone()
    }
}

impl BitAnd<GateExpr> for ChannelExpr {
    type Output = ChannelExpr;

    fn bitand(self, rhs: GateExpr) -> ChannelExpr {
        self & ChannelExpr::Gate(rhs)
    }
}

impl BitAnd<ChannelExpr> for GateExpr {
    type Output = ChannelExpr;

    fn bitand(self, rhs: ChannelExpr) -> ChannelExpr {
        ChannelExpr::Gate(self) & rhs
    }
}

impl Mul for ChannelExpr {
    type Output = ChannelExpr;

    fn mul(self, rhs: ChannelExpr) -> ChannelExpr {
        ChannelExpr::Compose(vec![self, rhs])
    }
}

impl Mul<&ChannelExpr> for ChannelExpr {
    type Output = ChannelExpr;

    fn mul(self, rhs: &ChannelExpr) -> ChannelExpr {
        self * rhs.clone()
    }
}

impl Mul<ChannelExpr> for &ChannelExpr {
    type Output = ChannelExpr;

    fn mul(self, rhs: ChannelExpr) -> ChannelExpr {
        self.clone() * rhs
    }
}

impl Mul<&ChannelExpr> for &ChannelExpr {
    type Output = ChannelExpr;

    fn mul(self, rhs: &ChannelExpr) -> ChannelExpr {
        self.clone() * rhs.clone()
    }
}

impl Mul<GateExpr> for ChannelExpr {
    type Output = ChannelExpr;

    fn mul(self, rhs: GateExpr) -> ChannelExpr {
        self * ChannelExpr::Gate(rhs)
    }
}

impl Mul<ChannelExpr> for GateExpr {
    type Output = ChannelExpr;

    fn mul(self, rhs: ChannelExpr) -> ChannelExpr {
        ChannelExpr::Gate(self) * rhs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gate;

    #[test]
    fn noise_constructors_return_channel_expr() {
        assert!(matches!(Depolarizing(0.1, 0), ChannelExpr::MixedUnitary(_)));
        assert!(matches!(Dephasing(0.1, 0), ChannelExpr::MixedUnitary(_)));
        assert!(matches!(BitFlip(0.1, 0), ChannelExpr::MixedUnitary(_)));
        assert!(matches!(BitPhaseFlip(0.1, 0), ChannelExpr::MixedUnitary(_)));
        assert!(matches!(
            PauliChannel(0.1, 0.2, 0.3, 0),
            ChannelExpr::MixedUnitary(_)
        ));
        assert!(matches!(
            Depolarizing2(0.1, 0, 1),
            ChannelExpr::MixedUnitary(_)
        ));
        assert!(matches!(
            AmplitudeDamping(0.1, 0),
            ChannelExpr::AmplitudeDamping { .. }
        ));
        assert!(matches!(
            PhaseDamping(0.1, 0),
            ChannelExpr::PhaseDamping { .. }
        ));
        assert!(matches!(Erasure(0.1, 0), ChannelExpr::Erasure { .. }));
        assert!(matches!(Leakage(0.1, 0), ChannelExpr::Leakage { .. }));
    }

    #[test]
    #[should_panic(expected = "Depolarizing2 requires distinct qubits")]
    fn channel_namespace_two_qubit_channel_rejects_repeated_qubit() {
        let _ = Depolarizing2(0.1, 0, 0);
    }

    #[test]
    fn ideal_gate_lifts_to_channel_expr() {
        let channel = from_gate(gate::MZ(0));
        assert!(matches!(
            channel,
            ChannelExpr::Gate(GateExpr::Measure { .. })
        ));
    }

    #[test]
    fn channel_tensor_and_composition_stay_channel_level() {
        let tensor = Depolarizing(0.1, 0) & BitFlip(0.2, 1);
        assert!(matches!(tensor, ChannelExpr::Tensor(parts) if parts.len() == 2));

        let sequence = AmplitudeDamping(0.1, 0) * PhaseDamping(0.2, 0);
        assert!(matches!(sequence, ChannelExpr::Compose(parts) if parts.len() == 2));
    }

    #[test]
    #[should_panic(expected = "tensor product requires disjoint channel support")]
    fn channel_tensor_rejects_overlapping_qubits() {
        let _ = Depolarizing(0.1, 0) & BitFlip(0.2, 0);
    }

    #[test]
    #[should_panic(expected = "tensor product requires disjoint channel support")]
    fn channel_tensor_rejects_partial_overlap_with_multi_qubit_support() {
        let _ = Depolarizing2(0.1, 0, 2) & BitFlip(0.2, 2);
    }

    #[test]
    fn channel_tensor_uses_sparse_support_not_dense_span() {
        let tensor = Depolarizing2(0.1, 0, 2) & BitFlip(0.2, 1);
        assert!(matches!(tensor, ChannelExpr::Tensor(ref parts) if parts.len() == 2));
        assert_eq!(tensor.qubits(), vec![0, 1, 2]);
    }

    #[test]
    fn gate_channel_combinations_promote_to_channel_level() {
        let tensor = gate::H(0) & Depolarizing(0.1, 1);
        assert!(matches!(tensor, ChannelExpr::Tensor(parts) if parts.len() == 2));

        let sequence = Depolarizing(0.1, 0) * gate::MZ(0);
        assert!(matches!(sequence, ChannelExpr::Compose(parts) if parts.len() == 2));
    }
}
