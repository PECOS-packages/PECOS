// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations
// under the License.

//! Rotation algebra must preserve amplitudes, including global phase.

use pecos_core::unitary_rep::{RotationType, rotation_sum_wraps};
use pecos_core::{Angle64, Unitary, UnitaryRep};
use pecos_quantum::unitary_matrix::{ToMatrix, UnitaryMatrix, to_matrix_with_size};
use std::f64::consts::{FRAC_PI_2, PI, TAU};

const AXES: [RotationType; 6] = [
    RotationType::RX,
    RotationType::RY,
    RotationType::RZ,
    RotationType::RXX,
    RotationType::RYY,
    RotationType::RZZ,
];

fn rotation(axis: RotationType, angle: Angle64) -> UnitaryRep {
    Unitary::Rotation {
        rotation_type: axis,
        angle,
    }
    .on_qubits(0, 1)
}

// Compare the complex amplitudes themselves, with only floating-point roundoff
// tolerance. No phase normalization or quotient is permitted.
fn assert_same_matrix(actual: &UnitaryMatrix, expected: &UnitaryMatrix) {
    assert_eq!(actual.shape(), expected.shape());
    let error = (actual.inner() - expected.inner()).norm();
    assert!(
        error < 1e-12,
        "amplitude error {error}:\nactual={actual:?}\nexpected={expected:?}"
    );
}

fn assert_simplification(rep: &UnitaryRep, size: usize) {
    let expected = to_matrix_with_size(rep, size);
    let simplified = rep.simplify();
    assert_same_matrix(&to_matrix_with_size(&simplified, size), &expected);
    assert_same_matrix(
        &to_matrix_with_size(&simplified.simplify(), size),
        &expected,
    );
}

#[test]
fn rotation_fusion_preserves_minus_identity() {
    for axis in AXES {
        let u = rotation(axis, Angle64::HALF_TURN);
        let size = u.to_matrix().num_qubits();
        let original = u.clone() * u;
        let expected = UnitaryMatrix::identity(1 << size) * num_complex::Complex64::new(-1.0, 0.0);
        assert_same_matrix(&original.to_matrix(), &expected);
        assert_simplification(&original, size);
    }
}

#[test]
fn rotation_adjoint_is_inverse() {
    for axis in AXES {
        for theta in [0.0, PI, -PI, FRAC_PI_2, TAU, 0.37] {
            let u = rotation(axis, Angle64::from_radians(theta));
            let matrix = u.to_matrix();
            assert_same_matrix(&u.dg().to_matrix(), &matrix.adjoint());
            let product = u.dg() * u;
            assert_same_matrix(
                &product.to_matrix(),
                &UnitaryMatrix::identity(matrix.nrows()),
            );
            assert_simplification(&product, matrix.num_qubits());
        }
    }
}

#[test]
fn rotation_double_adjoint_preserves_amplitudes() {
    for axis in AXES {
        for theta in [0.0, PI, -PI, FRAC_PI_2, TAU, 0.37] {
            let u = rotation(axis, Angle64::from_radians(theta));
            assert_same_matrix(&u.dg().dg().to_matrix(), &u.to_matrix());
            assert_simplification(&u.dg().dg(), u.to_matrix().num_qubits());
        }
    }
}

#[test]
fn rotation_fusion_pi_over_four_lattice() {
    for axis in AXES {
        for i in -8..=8 {
            for j in -8..=8 {
                let a = Angle64::from_turn_ratio(i, 8);
                let b = Angle64::from_turn_ratio(j, 8);
                let original = rotation(axis, a) * rotation(axis, b);
                assert_simplification(&original, original.to_matrix().num_qubits());
            }
        }
    }
}

#[test]
fn rotation_fusion_fixed_point_boundaries() {
    let half = Angle64::HALF_TURN.fraction();
    let angles = [0, 1, half - 1, half, half + 1, u64::MAX];
    for axis in AXES {
        for a in angles.map(Angle64::new) {
            for b in angles.map(Angle64::new) {
                let original = rotation(axis, a) * rotation(axis, b);
                assert_simplification(&original, original.to_matrix().num_qubits());
            }
        }
    }
}

#[test]
fn rotation_fusion_chains_and_nested_consumers() {
    use pecos_core::unitary_rep::{H, X};
    for axis in AXES {
        let u = rotation(axis, Angle64::from_turn_ratio(3, 8));
        for count in 2..=12 {
            let chain = u.pow(count);
            assert_simplification(&chain, 2);
            let nested = (chain.simplify() * H(0)) & X(2);
            assert_simplification(&nested, 3);
            assert_same_matrix(&nested.dg().to_matrix(), &nested.to_matrix().adjoint());
            let adjoint = UnitaryRep::Adjoint(Box::new(chain));
            assert_simplification(&adjoint, 2);
        }
    }
}

#[test]
fn rotation_adjoint_parameter_slots() {
    for theta in [0.0, PI, -PI, FRAC_PI_2, TAU, 0.37] {
        for phi in [0.0, PI, -PI, 0.23] {
            for lambda in [0.0, PI, -PI, -0.41] {
                for unitary in [
                    Unitary::RXY1Q {
                        theta: Angle64::from_radians(theta),
                        phi: Angle64::from_radians(phi),
                    },
                    Unitary::U3 {
                        theta: Angle64::from_radians(theta),
                        phi: Angle64::from_radians(phi),
                        lambda: Angle64::from_radians(lambda),
                    },
                ] {
                    let u = unitary.on_qubit(0);
                    let matrix = u.to_matrix();
                    assert_same_matrix(&u.dg().to_matrix(), &matrix.adjoint());
                    assert_same_matrix(
                        &(u.dg() * u.clone()).to_matrix(),
                        &UnitaryMatrix::identity(2),
                    );
                    assert_same_matrix(&u.dg().dg().to_matrix(), &matrix);
                }
            }
        }
    }
}

#[test]
fn rotation_adjoint_compound_slot_parity() {
    // All subsets of the four U3 rotation slots and three interaction slots.
    // Phase slots deliberately include pi and must not affect sign parity.
    for mask in 0..128 {
        let angle = |bit| {
            if mask & (1 << bit) != 0 {
                Angle64::HALF_TURN
            } else {
                Angle64::from_radians(0.37)
            }
        };
        let interaction = [angle(4), angle(5), angle(6)];
        for unitary in [
            Unitary::RXXRYYRZZ {
                alpha: interaction[0],
                beta: interaction[1],
                gamma: interaction[2],
            },
            Unitary::U2q {
                before: [
                    [angle(0), Angle64::HALF_TURN, Angle64::QUARTER_TURN],
                    [angle(1), Angle64::ZERO, Angle64::HALF_TURN],
                ],
                interaction,
                after: [
                    [angle(2), Angle64::HALF_TURN, Angle64::ZERO],
                    [angle(3), Angle64::QUARTER_TURN, Angle64::HALF_TURN],
                ],
            },
        ] {
            let u = unitary.on_qubits(0, 1);
            assert_same_matrix(&u.dg().to_matrix(), &u.to_matrix().adjoint());
            assert_same_matrix(
                &(u.dg() * u.clone()).to_matrix(),
                &UnitaryMatrix::identity(4),
            );
            assert_same_matrix(&u.dg().dg().to_matrix(), &u.to_matrix());
        }
    }
}

#[test]
fn rotation_pauli_consumers_preserve_amplitudes() {
    for axis in AXES {
        let u = rotation(axis, Angle64::HALF_TURN);
        for rep in [
            u.clone(),
            u.dg(),
            u.dg().simplify(),
            (u.clone() * u).simplify(),
        ] {
            let expected = to_matrix_with_size(&rep, 2);
            let pauli = rep.clone().try_to_pauli().expect("Pauli conversion");
            assert_same_matrix(&to_matrix_with_size(&pauli, 2), &expected);
            let string =
                UnitaryRep::Pauli(rep.try_to_pauli_string().expect("Pauli string conversion"));
            assert_same_matrix(&to_matrix_with_size(&string, 2), &expected);
        }
    }
}

#[test]
fn rotation_phase_packet_reproduction() {
    let u = rotation(RotationType::RZ, Angle64::HALF_TURN);
    let original = (u.clone() * u.clone()).to_matrix();
    let naive = rotation(RotationType::RZ, Angle64::ZERO).to_matrix();
    println!(
        "fusion errors: corrected={:.4}, naive={:.4}",
        (to_matrix_with_size(&(u.clone() * u.clone()).simplify(), 1).inner() - original.inner())
            .norm(),
        (naive.inner() - original.inner()).norm()
    );
    println!(
        "adjoint errors: corrected={:.4}, naive={:.4}",
        (u.dg().to_matrix().inner() - u.to_matrix().adjoint().inner()).norm(),
        (u.to_matrix().inner() - u.to_matrix().adjoint().inner()).norm()
    );
    for i in -8..=8 {
        for j in -8..=8 {
            let a = Angle64::from_turn_ratio(i, 8);
            let b = Angle64::from_turn_ratio(j, 8);
            let fused = rotation(RotationType::RZ, a + b).with_phase(if rotation_sum_wraps(a, b) {
                Angle64::HALF_TURN
            } else {
                Angle64::ZERO
            });
            assert_same_matrix(
                &fused.to_matrix(),
                &(rotation(RotationType::RZ, a) * rotation(RotationType::RZ, b)).to_matrix(),
            );
        }
    }
    println!("wrap rule: 289 pairs, 0 mismatches");
}

#[test]
fn rotation_op_adjoint_preserves_amplitudes() {
    for axis in AXES {
        let u = rotation(axis, Angle64::HALF_TURN);
        let expected = u.to_matrix().adjoint();
        let op = pecos_core::Op::from(u);
        assert_same_matrix(&op.dg().to_matrix(), &expected);
        assert_same_matrix(
            &op.try_dg().expect("unitary adjoint").to_matrix(),
            &expected,
        );
    }
}
