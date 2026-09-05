// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use
// this file except in compliance with the License. You may obtain a copy of the
// License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

//! Dense unitary matrices must agree with what the simulators do entrywise.
//!
//! `pecos-quantum`'s `to_matrix` and `pecos-simulators`' `CliffordGateable` are
//! independent implementations of the same gate set, and nothing compared them.
//! Issue #379 was the consequence: `GateType::F`'s dense matrix was built as
//! `SX * SZ` instead of `SZ * SX`, making it the `F4` face gate. Every other
//! layer agreed with every other layer, so no existing test could see it.
//!
//! This walks every named Clifford and every parameterised [`Unitary`] kind,
//! reconstructs the simulator's unitary one basis state at a time, and compares
//! it exactly to the dense matrix.

use pecos_core::unitary_rep::RotationType;
use pecos_core::{Angle64, Clifford, QubitId, Unitary, UnitaryRep};
use pecos_quantum::GateType;
use pecos_quantum::unitary_matrix::to_matrix_with_size;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVecSoA};

/// Apply a named gate through the `CliffordGateable` API.
fn apply_named(sim: &mut StateVecSoA, gate_type: GateType, qubits: &[QubitId]) {
    let one = &[qubits[0]];
    match gate_type {
        GateType::I => sim.identity(one),
        GateType::X => sim.x(one),
        GateType::Y => sim.y(one),
        GateType::Z => sim.z(one),
        GateType::H => sim.h(one),
        GateType::F => sim.f(one),
        GateType::Fdg => sim.fdg(one),
        GateType::SX => sim.sx(one),
        GateType::SXdg => sim.sxdg(one),
        GateType::SY => sim.sy(one),
        GateType::SYdg => sim.sydg(one),
        GateType::SZ => sim.sz(one),
        GateType::SZdg => sim.szdg(one),
        GateType::CX => sim.cx(&[(qubits[0], qubits[1])]),
        GateType::CY => sim.cy(&[(qubits[0], qubits[1])]),
        GateType::CZ => sim.cz(&[(qubits[0], qubits[1])]),
        GateType::SXX => sim.sxx(&[(qubits[0], qubits[1])]),
        GateType::SXXdg => sim.sxxdg(&[(qubits[0], qubits[1])]),
        GateType::SYY => sim.syy(&[(qubits[0], qubits[1])]),
        GateType::SYYdg => sim.syydg(&[(qubits[0], qubits[1])]),
        GateType::SZZ => sim.szz(&[(qubits[0], qubits[1])]),
        GateType::SZZdg => sim.szzdg(&[(qubits[0], qubits[1])]),
        GateType::SWAP => sim.swap(&[(qubits[0], qubits[1])]),
        other => panic!("gate {other:?} is not covered by this test's dispatch"),
    };
}

/// Apply a two-qubit Clifford, including variants with no `GateType` spelling.
fn apply_two_qubit_clifford(sim: &mut StateVecSoA, clifford: Clifford) {
    let pair = &[(QubitId(0), QubitId(1))];
    match clifford {
        Clifford::CX => sim.cx(pair),
        Clifford::CY => sim.cy(pair),
        Clifford::CZ => sim.cz(pair),
        Clifford::SXX => sim.sxx(pair),
        Clifford::SXXdg => sim.sxxdg(pair),
        Clifford::SYY => sim.syy(pair),
        Clifford::SYYdg => sim.syydg(pair),
        Clifford::SZZ => sim.szz(pair),
        Clifford::SZZdg => sim.szzdg(pair),
        Clifford::SWAP => sim.swap(pair),
        Clifford::G => sim.g(pair),
        Clifford::Gdg => sim.gdg(pair),
        Clifford::ISWAP => sim.iswap(pair),
        Clifford::ISWAPdg => sim.iswapdg(pair),
        other => panic!("Clifford {other:?} is not a two-qubit gate covered by this test"),
    };
}

/// Rebuild the simulator's unitary column by column: column `j` is the state
/// produced by applying the gate to basis state `|j>`.
fn simulator_unitary(
    num_qubits: usize,
    apply: &dyn Fn(&mut StateVecSoA),
) -> Vec<Vec<num_complex::Complex64>> {
    let dim = 1usize << num_qubits;
    let qubits: Vec<QubitId> = (0..num_qubits).map(QubitId).collect();
    let mut columns = Vec::with_capacity(dim);
    for basis in 0..dim {
        let mut sim = StateVecSoA::new(num_qubits);
        for (bit, &q) in qubits.iter().enumerate() {
            if basis & (1 << bit) != 0 {
                sim.x(&[q]);
            }
        }
        apply(&mut sim);
        columns.push(sim.state());
    }
    columns
}

/// Compare two matrices entrywise. A global phase difference is a defect here.
fn assert_exactly_equal(
    name: &str,
    dense: &[Vec<num_complex::Complex64>],
    sim: &[Vec<num_complex::Complex64>],
) {
    for (col_idx, (d_col, s_col)) in dense.iter().zip(sim).enumerate() {
        for (row_idx, (d, s)) in d_col.iter().zip(s_col).enumerate() {
            assert!(
                (d - s).norm() < 1e-12,
                "{name}: dense matrix disagrees with the simulator at \
                 (row {row_idx}, col {col_idx}): dense {d} vs simulator {s}. \
                 A global phase difference is a defect."
            );
        }
    }
}

fn matrix_columns(rep: &UnitaryRep, num_qubits: usize) -> Vec<Vec<num_complex::Complex64>> {
    let matrix = to_matrix_with_size(rep, num_qubits).into_inner();
    let dim = 1usize << num_qubits;
    // to_matrix is column-major-agnostic; take explicit columns to match the
    // simulator reconstruction.
    (0..dim)
        .map(|col| (0..dim).map(|row| matrix[(row, col)]).collect())
        .collect()
}

fn check_named(gate_type: GateType, num_qubits: usize) {
    let named = Unitary::Named(gate_type);
    let rep = if num_qubits == 1 {
        named.on_qubit(0)
    } else {
        named.on_qubits(0, 1)
    };
    let qubits: Vec<QubitId> = (0..num_qubits).map(QubitId).collect();
    assert_exactly_equal(
        &format!("{gate_type:?}"),
        &matrix_columns(&rep, num_qubits),
        &simulator_unitary(num_qubits, &|sim| {
            apply_named(sim, gate_type, &qubits);
        }),
    );
}

fn check_two_qubit_clifford(clifford: Clifford) {
    let rep = clifford.to_unitary_rep_on_qubits(0, 1);
    assert_exactly_equal(
        &format!("{clifford:?}"),
        &matrix_columns(&rep, 2),
        &simulator_unitary(2, &|sim| {
            apply_two_qubit_clifford(sim, clifford);
        }),
    );
}

fn check_parameterised(name: &str, unitary: Unitary, apply: &dyn Fn(&mut StateVecSoA)) {
    let num_qubits = unitary.num_qubits();
    let rep = if num_qubits == 1 {
        unitary.on_qubit(0)
    } else {
        unitary.on_qubits(0, 1)
    };
    assert_exactly_equal(
        name,
        &matrix_columns(&rep, num_qubits),
        &simulator_unitary(num_qubits, apply),
    );
}

#[test]
fn dense_matrices_match_the_simulator_for_one_qubit_cliffords() {
    for gate_type in [
        GateType::I,
        GateType::X,
        GateType::Y,
        GateType::Z,
        GateType::H,
        GateType::F,
        GateType::Fdg,
        GateType::SX,
        GateType::SXdg,
        GateType::SY,
        GateType::SYdg,
        GateType::SZ,
        GateType::SZdg,
    ] {
        check_named(gate_type, 1);
    }
}

#[test]
fn dense_matrices_match_the_simulator_for_two_qubit_cliffords() {
    for clifford in [
        Clifford::CX,
        Clifford::CY,
        Clifford::CZ,
        Clifford::SXX,
        Clifford::SXXdg,
        Clifford::SYY,
        Clifford::SYYdg,
        Clifford::SZZ,
        Clifford::SZZdg,
        Clifford::SWAP,
        Clifford::G,
        Clifford::Gdg,
        Clifford::ISWAP,
        Clifford::ISWAPdg,
    ] {
        check_two_qubit_clifford(clifford);
    }
}

#[test]
fn dense_matrices_match_the_simulator_for_parameterised_unitaries() {
    let q = [QubitId(0)];
    let pair = [(QubitId(0), QubitId(1))];
    for sign in [1.0, -1.0] {
        let angle = |magnitude| Angle64::from_radians(sign * magnitude);

        for (name, rotation_type) in [
            ("RX", RotationType::RX),
            ("RY", RotationType::RY),
            ("RZ", RotationType::RZ),
        ] {
            let theta = angle(0.37);
            let unitary = Unitary::Rotation {
                rotation_type,
                angle: theta,
            };
            check_parameterised(&format!("{name}({:+.2})", sign * 0.37), unitary, &|sim| {
                match rotation_type {
                    RotationType::RX => sim.rx(theta, &q),
                    RotationType::RY => sim.ry(theta, &q),
                    RotationType::RZ => sim.rz(theta, &q),
                    _ => unreachable!("the one-qubit table contains only RX, RY, and RZ"),
                };
            });
        }

        for (name, rotation_type) in [
            ("RXX", RotationType::RXX),
            ("RYY", RotationType::RYY),
            ("RZZ", RotationType::RZZ),
        ] {
            let theta = angle(0.41);
            let unitary = Unitary::Rotation {
                rotation_type,
                angle: theta,
            };
            check_parameterised(&format!("{name}({:+.2})", sign * 0.41), unitary, &|sim| {
                match rotation_type {
                    RotationType::RXX => sim.rxx(theta, &pair),
                    RotationType::RYY => sim.ryy(theta, &pair),
                    RotationType::RZZ => sim.rzz(theta, &pair),
                    _ => unreachable!("the two-qubit table contains only RXX, RYY, and RZZ"),
                };
            });
        }

        let theta = angle(0.43);
        let phi = angle(0.67);
        check_parameterised(
            &format!("RXY1Q(sign={sign:+.0})"),
            Unitary::RXY1Q { theta, phi },
            &|sim| {
                sim.rxy1q(theta, phi, &q);
            },
        );

        let lambda = angle(0.89);
        check_parameterised(
            &format!("U3(sign={sign:+.0})"),
            Unitary::U3 { theta, phi, lambda },
            &|sim| {
                sim.u(theta, phi, lambda, &q);
            },
        );

        let interaction = [angle(0.31), angle(0.47), angle(0.59)];
        check_parameterised(
            &format!("RXXRYYRZZ(sign={sign:+.0})"),
            Unitary::RXXRYYRZZ {
                alpha: interaction[0],
                beta: interaction[1],
                gamma: interaction[2],
            },
            &|sim| {
                sim.rxxryyrzz(interaction[0], interaction[1], interaction[2], &pair);
            },
        );

        let before = [
            [angle(0.11), angle(0.13), angle(0.17)],
            [angle(0.19), angle(0.23), angle(0.29)],
        ];
        let after = [
            [angle(0.61), angle(0.71), angle(0.73)],
            [angle(0.79), angle(0.83), angle(0.97)],
        ];
        check_parameterised(
            &format!("U2q(sign={sign:+.0})"),
            Unitary::U2q {
                before,
                interaction,
                after,
            },
            &|sim| {
                sim.u2q(before, interaction, after, &pair);
            },
        );
    }
}

/// `F = i SZ * SX` is the `SX`-then-`SZ` face rotation cycling
/// `X -> Y -> Z -> X`, and `Fdg = -i SXdg * SZdg` is its reverse,
/// `X -> Z -> Y -> X`. Pinned explicitly because the composition order is the
/// exact thing #379 got backwards, and because `SX * SZ` is the projective
/// representative of `F4` -- a real, separately implemented gate, so the
/// mistake produces a valid Clifford rather than anything obviously broken.
#[test]
fn face_gate_composition_order_is_pinned() {
    for (face, decomposition, phase) in [
        (
            GateType::F,
            [GateType::SX, GateType::SZ],
            num_complex::Complex64::new(0.0, 1.0),
        ),
        (
            GateType::Fdg,
            [GateType::SZdg, GateType::SXdg],
            num_complex::Complex64::new(0.0, -1.0),
        ),
    ] {
        let composed = {
            let mut sim = StateVecSoA::new(1);
            sim.h(&[QubitId(0)]);
            sim.sz(&[QubitId(0)]);
            for gate_type in decomposition {
                apply_named(&mut sim, gate_type, &[QubitId(0)]);
            }
            sim.state()
        };
        let native = {
            let mut sim = StateVecSoA::new(1);
            sim.h(&[QubitId(0)]);
            sim.sz(&[QubitId(0)]);
            apply_named(&mut sim, face, &[QubitId(0)]);
            sim.state()
        };
        for (a, b) in composed.iter().zip(&native) {
            assert!(
                (phase * a - b).norm() < 1e-9,
                "{face:?} must equal {phase} times {decomposition:?} applied in order"
            );
        }
    }
}
