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

//! Ideal circuit-operation namespace.
//!
//! Constructors in this module return [`GateExpr`]. Use this namespace when
//! the operation is an intended ideal circuit operation, including unitary
//! gates, preparation, measurement, and reset:
//!
//! ```
//! use pecos_core::gate::*;
//!
//! let layer = H(0) & MZ(1);
//! let sequence = PZ(0) * H(0) * MZ(0);
//! ```
//!
//! For automatic promotion across all levels, use [`crate::op`]. For physical
//! noise and open-system maps, use [`crate::channel`].

use crate::op::Op;
use crate::qubit_support::overlapping_qubits;
use crate::unitary_rep::{QubitPairs, Qubits};
use crate::{Angle64, PauliString, QubitId, UnitaryRep, op, unitary_rep};
use std::ops::{BitAnd, Mul};

pub use crate::op::{Basis, GateExpr};

fn gate_from_op(op: Op) -> GateExpr {
    op.into_gate()
        .expect("gate namespace constructors are gate-convertible")
}

/// Lifts a unitary expression to the ideal gate level.
#[must_use]
pub fn from_unitary(unitary: impl Into<UnitaryRep>) -> GateExpr {
    GateExpr::Unitary(unitary.into())
}

impl From<UnitaryRep> for GateExpr {
    fn from(unitary: UnitaryRep) -> Self {
        GateExpr::Unitary(unitary)
    }
}

impl From<PauliString> for GateExpr {
    fn from(pauli: PauliString) -> Self {
        GateExpr::Unitary(UnitaryRep::from(pauli))
    }
}

macro_rules! unitary_1q {
    ($name:ident) => {
        #[allow(non_snake_case)]
        #[must_use]
        pub fn $name(qubit: impl Into<QubitId>) -> GateExpr {
            from_unitary(unitary_rep::$name(qubit))
        }
    };
}

macro_rules! unitary_1q_plural {
    ($name:ident) => {
        #[allow(non_snake_case)]
        #[must_use]
        pub fn $name(qubits: impl Into<Qubits>) -> GateExpr {
            from_unitary(unitary_rep::$name(qubits))
        }
    };
}

macro_rules! op_1q {
    ($name:ident) => {
        #[allow(non_snake_case)]
        #[must_use]
        pub fn $name(qubit: impl Into<QubitId>) -> GateExpr {
            gate_from_op(op::$name(qubit))
        }
    };
}

macro_rules! op_2q {
    ($name:ident) => {
        #[allow(non_snake_case)]
        #[must_use]
        pub fn $name(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> GateExpr {
            gate_from_op(op::$name(q0, q1))
        }
    };
}

macro_rules! unitary_2q {
    ($name:ident) => {
        #[allow(non_snake_case)]
        #[must_use]
        pub fn $name(q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> GateExpr {
            from_unitary(unitary_rep::$name(q0, q1))
        }
    };
}

macro_rules! unitary_2q_plural {
    ($name:ident) => {
        #[allow(non_snake_case)]
        #[must_use]
        pub fn $name(pairs: impl Into<QubitPairs>) -> GateExpr {
            from_unitary(unitary_rep::$name(pairs))
        }
    };
}

unitary_1q!(I);
unitary_1q_plural!(Is);
unitary_1q!(X);
unitary_1q_plural!(Xs);
unitary_1q!(Y);
unitary_1q_plural!(Ys);
unitary_1q!(Z);
unitary_1q_plural!(Zs);
unitary_1q!(H);
unitary_1q_plural!(Hs);
unitary_1q!(SX);
unitary_1q_plural!(SXs);
op_1q!(SXdg);
unitary_1q!(SY);
unitary_1q_plural!(SYs);
op_1q!(SYdg);
unitary_1q!(SZ);
unitary_1q_plural!(SZs);
op_1q!(SZdg);
op_1q!(H2);
op_1q!(H3);
op_1q!(H4);
op_1q!(H5);
op_1q!(H6);
op_1q!(F);
op_1q!(Fdg);
op_1q!(F2);
op_1q!(F2dg);
op_1q!(F3);
op_1q!(F3dg);
op_1q!(F4);
op_1q!(F4dg);
unitary_1q!(T);
unitary_1q_plural!(Ts);
op_1q!(Tdg);

unitary_2q!(CX);
unitary_2q_plural!(CXs);
unitary_2q!(CY);
unitary_2q_plural!(CYs);
unitary_2q!(CZ);
unitary_2q_plural!(CZs);
unitary_2q!(SWAP);
unitary_2q_plural!(SWAPs);
op_2q!(SXX);
op_2q!(SXXdg);
op_2q!(SYY);
op_2q!(SYYdg);
unitary_2q!(SZZ);
unitary_2q_plural!(SZZs);
op_2q!(SZZdg);
op_2q!(ISWAP);
op_2q!(ISWAPdg);
op_2q!(G);
op_2q!(Gdg);

/// Rotation around X axis: exp(-i theta/2 X).
#[allow(non_snake_case)]
#[must_use]
pub fn RX(angle: Angle64, qubit: impl Into<QubitId>) -> GateExpr {
    from_unitary(unitary_rep::RX(angle, qubit))
}

/// Rotations around X on multiple qubits.
#[allow(non_snake_case)]
#[must_use]
pub fn RXs(angle: Angle64, qubits: impl Into<Qubits>) -> GateExpr {
    from_unitary(unitary_rep::RXs(angle, qubits))
}

/// Rotation around Y axis: exp(-i theta/2 Y).
#[allow(non_snake_case)]
#[must_use]
pub fn RY(angle: Angle64, qubit: impl Into<QubitId>) -> GateExpr {
    from_unitary(unitary_rep::RY(angle, qubit))
}

/// Rotations around Y on multiple qubits.
#[allow(non_snake_case)]
#[must_use]
pub fn RYs(angle: Angle64, qubits: impl Into<Qubits>) -> GateExpr {
    from_unitary(unitary_rep::RYs(angle, qubits))
}

/// Rotation around Z axis: exp(-i theta/2 Z).
#[allow(non_snake_case)]
#[must_use]
pub fn RZ(angle: Angle64, qubit: impl Into<QubitId>) -> GateExpr {
    from_unitary(unitary_rep::RZ(angle, qubit))
}

/// Rotations around Z on multiple qubits.
#[allow(non_snake_case)]
#[must_use]
pub fn RZs(angle: Angle64, qubits: impl Into<Qubits>) -> GateExpr {
    from_unitary(unitary_rep::RZs(angle, qubits))
}

/// Two-qubit XX rotation: exp(-i theta/2 XX).
#[allow(non_snake_case)]
#[must_use]
pub fn RXX(angle: Angle64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> GateExpr {
    from_unitary(unitary_rep::RXX(angle, q0, q1))
}

/// XX rotations on multiple qubit pairs.
#[allow(non_snake_case)]
#[must_use]
pub fn RXXs(angle: Angle64, pairs: impl Into<QubitPairs>) -> GateExpr {
    from_unitary(unitary_rep::RXXs(angle, pairs))
}

/// Two-qubit YY rotation: exp(-i theta/2 YY).
#[allow(non_snake_case)]
#[must_use]
pub fn RYY(angle: Angle64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> GateExpr {
    from_unitary(unitary_rep::RYY(angle, q0, q1))
}

/// YY rotations on multiple qubit pairs.
#[allow(non_snake_case)]
#[must_use]
pub fn RYYs(angle: Angle64, pairs: impl Into<QubitPairs>) -> GateExpr {
    from_unitary(unitary_rep::RYYs(angle, pairs))
}

/// Two-qubit ZZ rotation: exp(-i theta/2 ZZ).
#[allow(non_snake_case)]
#[must_use]
pub fn RZZ(angle: Angle64, q0: impl Into<QubitId>, q1: impl Into<QubitId>) -> GateExpr {
    from_unitary(unitary_rep::RZZ(angle, q0, q1))
}

/// ZZ rotations on multiple qubit pairs.
#[allow(non_snake_case)]
#[must_use]
pub fn RZZs(angle: Angle64, pairs: impl Into<QubitPairs>) -> GateExpr {
    from_unitary(unitary_rep::RZZs(angle, pairs))
}

/// Toffoli gate (CCX).
#[allow(non_snake_case)]
#[must_use]
pub fn CCX(c0: impl Into<QubitId>, c1: impl Into<QubitId>, target: impl Into<QubitId>) -> GateExpr {
    from_unitary(unitary_rep::CCX(c0, c1, target))
}

/// Prepare a qubit in the requested basis eigenstate.
#[must_use]
pub fn prep(basis: Basis, qubit: impl Into<QubitId>) -> GateExpr {
    GateExpr::Prep {
        basis,
        qubit: qubit.into().0,
    }
}

/// Measure a qubit in the requested basis.
#[must_use]
pub fn measure(basis: Basis, qubit: impl Into<QubitId>) -> GateExpr {
    GateExpr::Measure {
        basis,
        qubit: qubit.into().0,
    }
}

/// Reset a qubit to the requested basis eigenstate.
#[must_use]
pub fn reset(basis: Basis, qubit: impl Into<QubitId>) -> GateExpr {
    GateExpr::Reset {
        basis,
        qubit: qubit.into().0,
    }
}

/// Prepare qubit in the |0> state.
#[allow(non_snake_case)]
#[must_use]
pub fn PZ(qubit: impl Into<QubitId>) -> GateExpr {
    prep(Basis::Z, qubit)
}

/// Prepare qubit in the |+> state.
#[allow(non_snake_case)]
#[must_use]
pub fn PX(qubit: impl Into<QubitId>) -> GateExpr {
    prep(Basis::X, qubit)
}

/// Prepare qubit in the Y-basis +1 eigenstate.
#[allow(non_snake_case)]
#[must_use]
pub fn PY(qubit: impl Into<QubitId>) -> GateExpr {
    prep(Basis::Y, qubit)
}

/// Measure qubit in the Z basis.
#[allow(non_snake_case)]
#[must_use]
pub fn MZ(qubit: impl Into<QubitId>) -> GateExpr {
    measure(Basis::Z, qubit)
}

/// Measure qubit in the X basis.
#[allow(non_snake_case)]
#[must_use]
pub fn MX(qubit: impl Into<QubitId>) -> GateExpr {
    measure(Basis::X, qubit)
}

/// Measure qubit in the Y basis.
#[allow(non_snake_case)]
#[must_use]
pub fn MY(qubit: impl Into<QubitId>) -> GateExpr {
    measure(Basis::Y, qubit)
}

/// Reset qubit to |0>.
#[allow(non_snake_case)]
#[must_use]
pub fn Reset(qubit: impl Into<QubitId>) -> GateExpr {
    reset(Basis::Z, qubit)
}

impl BitAnd for GateExpr {
    type Output = GateExpr;

    fn bitand(self, rhs: GateExpr) -> GateExpr {
        let overlap = overlapping_qubits(self.qubits(), rhs.qubits());
        assert!(
            overlap.is_empty(),
            "tensor product requires disjoint gate support; overlapping qubits: {overlap:?}"
        );
        GateExpr::Tensor(vec![self, rhs])
    }
}

impl BitAnd<&GateExpr> for GateExpr {
    type Output = GateExpr;

    fn bitand(self, rhs: &GateExpr) -> GateExpr {
        self & rhs.clone()
    }
}

impl BitAnd<GateExpr> for &GateExpr {
    type Output = GateExpr;

    fn bitand(self, rhs: GateExpr) -> GateExpr {
        self.clone() & rhs
    }
}

impl BitAnd<&GateExpr> for &GateExpr {
    type Output = GateExpr;

    fn bitand(self, rhs: &GateExpr) -> GateExpr {
        self.clone() & rhs.clone()
    }
}

impl Mul for GateExpr {
    type Output = GateExpr;

    fn mul(self, rhs: GateExpr) -> GateExpr {
        GateExpr::Compose(vec![self, rhs])
    }
}

impl Mul<&GateExpr> for GateExpr {
    type Output = GateExpr;

    fn mul(self, rhs: &GateExpr) -> GateExpr {
        self * rhs.clone()
    }
}

impl Mul<GateExpr> for &GateExpr {
    type Output = GateExpr;

    fn mul(self, rhs: GateExpr) -> GateExpr {
        self.clone() * rhs
    }
}

impl Mul<&GateExpr> for &GateExpr {
    type Output = GateExpr;

    fn mul(self, rhs: &GateExpr) -> GateExpr {
        self.clone() * rhs.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn namespace_constructors_return_gate_expr() {
        assert!(matches!(
            MZ(3),
            GateExpr::Measure {
                basis: Basis::Z,
                qubit: 3
            }
        ));
        assert!(matches!(
            MX(4),
            GateExpr::Measure {
                basis: Basis::X,
                qubit: 4
            }
        ));
        assert!(matches!(
            MY(5),
            GateExpr::Measure {
                basis: Basis::Y,
                qubit: 5
            }
        ));
        assert!(matches!(
            PZ(0),
            GateExpr::Prep {
                basis: Basis::Z,
                qubit: 0
            }
        ));
        assert!(matches!(
            PX(1),
            GateExpr::Prep {
                basis: Basis::X,
                qubit: 1
            }
        ));
        assert!(matches!(
            PY(2),
            GateExpr::Prep {
                basis: Basis::Y,
                qubit: 2
            }
        ));
    }

    #[test]
    fn unitary_constructor_lifts_to_gate_level() {
        assert!(matches!(H(0), GateExpr::Unitary(_)));
        assert!(matches!(T(0), GateExpr::Unitary(_)));
        assert!(matches!(CX(0, 1), GateExpr::Unitary(_)));
        assert_eq!(I(7).qubits(), vec![7]);
    }

    #[test]
    #[should_panic(expected = "CX requires distinct qubits")]
    fn gate_namespace_two_qubit_gate_rejects_repeated_qubit() {
        let _ = CX(0, 0);
    }

    #[test]
    #[should_panic(expected = "RZZ requires distinct qubits")]
    fn gate_namespace_two_qubit_rotation_rejects_repeated_qubit() {
        let _ = RZZ(Angle64::QUARTER_TURN, 1, 1);
    }

    #[test]
    #[should_panic(expected = "CCX requires distinct qubits")]
    fn gate_namespace_three_qubit_gate_rejects_repeated_qubit() {
        let _ = CCX(0, 1, 1);
    }

    #[test]
    fn gate_tensor_and_composition_stay_gate_level() {
        let tensor = H(0) & MZ(1);
        assert!(matches!(tensor, GateExpr::Tensor(parts) if parts.len() == 2));

        let sequence = PZ(0) * H(0) * MZ(0);
        assert!(matches!(sequence, GateExpr::Compose(parts) if parts.len() == 2));
    }

    #[test]
    #[should_panic(expected = "tensor product requires disjoint gate support")]
    fn gate_tensor_rejects_overlapping_qubits() {
        let _ = H(0) & MZ(0);
    }

    #[test]
    #[should_panic(expected = "tensor product requires disjoint gate support")]
    fn gate_tensor_rejects_partial_overlap_with_multi_qubit_support() {
        let _ = CX(0, 2) & H(2);
    }

    #[test]
    fn gate_tensor_uses_sparse_support_not_dense_span() {
        let tensor = CX(0, 2) & MZ(1);
        assert!(matches!(tensor, GateExpr::Tensor(ref parts) if parts.len() == 2));
        assert_eq!(tensor.qubits(), vec![0, 1, 2]);
    }

    #[test]
    fn gate_namespace_plural_helpers_match_tensor_forms() {
        let cxs = CXs([(0, 1), (2, 3)]);
        assert!(matches!(cxs, GateExpr::Unitary(_)));
        assert_eq!(cxs.qubits(), vec![0, 1, 2, 3]);

        let rzzs = RZZs(Angle64::QUARTER_TURN, [(0, 1), (2, 3)]);
        assert!(matches!(rzzs, GateExpr::Unitary(_)));
        assert_eq!(rzzs.qubits(), vec![0, 1, 2, 3]);

        let tensor = CX(0, 1) & CX(2, 3);
        assert!(matches!(tensor, GateExpr::Tensor(_)));
        assert_eq!(tensor.qubits(), vec![0, 1, 2, 3]);
    }

    #[test]
    fn gate_namespace_plural_helpers_reject_overlapping_support() {
        fn assert_tensor_overlap_panic(f: impl FnOnce() + std::panic::UnwindSafe) {
            let err = std::panic::catch_unwind(f).expect_err("expected tensor overlap panic");
            let message = err
                .downcast_ref::<String>()
                .map(String::as_str)
                .or_else(|| err.downcast_ref::<&str>().copied())
                .unwrap_or("<non-string panic>");
            assert!(
                message.contains("tensor product requires disjoint"),
                "unexpected panic message: {message}"
            );
        }

        assert_tensor_overlap_panic(|| {
            let _ = CXs([(0, 1), (1, 2)]);
        });
        assert_tensor_overlap_panic(|| {
            let _ = RZZs(Angle64::QUARTER_TURN, [(0, 2), (2, 3)]);
        });
    }
}
