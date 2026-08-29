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

//! The dense named-unitary matrices must agree with what the simulators do.
//!
//! `pecos-quantum`'s `to_matrix` and `pecos-simulators`' `CliffordGateable` are
//! independent implementations of the same gate set, and nothing compared them.
//! Issue #379 was the consequence: `GateType::F`'s dense matrix was built as
//! `SX * SZ` instead of `SZ * SX`, making it the `F4` face gate. Every other
//! layer agreed with every other layer, so no existing test could see it.
//!
//! This walks every named Clifford, reconstructs the simulator's unitary one
//! basis state at a time, and compares it to the dense matrix up to global
//! phase.

use pecos_core::{QubitId, Unitary};
use pecos_quantum::GateType;
use pecos_quantum::unitary_matrix::to_matrix_with_size;
use pecos_simulators::{CliffordGateable, StateVecSoA};

/// Apply a named gate through the `CliffordGateable` API.
fn apply(sim: &mut StateVecSoA, gate_type: GateType, qubits: &[QubitId]) {
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

/// Rebuild the simulator's unitary column by column: column `j` is the state
/// produced by applying the gate to basis state `|j>`.
fn simulator_unitary(gate_type: GateType, num_qubits: usize) -> Vec<Vec<num_complex::Complex64>> {
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
        apply(&mut sim, gate_type, &qubits);
        columns.push(sim.state());
    }
    columns
}

/// Compare two matrices up to a single global phase.
fn assert_equal_up_to_phase(
    gate_type: GateType,
    dense: &[Vec<num_complex::Complex64>],
    sim: &[Vec<num_complex::Complex64>],
) {
    let phase = dense
        .iter()
        .flatten()
        .zip(sim.iter().flatten())
        .find(|(d, _)| d.norm() > 1e-9)
        .map(|(d, s)| s / d)
        .expect("a unitary has at least one non-zero entry");
    assert!(
        (phase.norm() - 1.0).abs() < 1e-9,
        "{gate_type:?}: ratio between dense and simulator matrices is not a phase ({phase})"
    );
    for (col_idx, (d_col, s_col)) in dense.iter().zip(sim).enumerate() {
        for (row_idx, (d, s)) in d_col.iter().zip(s_col).enumerate() {
            assert!(
                (d * phase - s).norm() < 1e-9,
                "{gate_type:?}: dense matrix disagrees with the simulator at \
                 (row {row_idx}, col {col_idx}): dense {d} (x phase {phase}) vs simulator {s}. \
                 The dense named-unitary and CliffordGateable have drifted apart."
            );
        }
    }
}

fn check(gate_type: GateType, num_qubits: usize) {
    let named = Unitary::Named(gate_type);
    let rep = if num_qubits == 1 {
        named.on_qubit(0)
    } else {
        named.on_qubits(0, 1)
    };
    let matrix = to_matrix_with_size(&rep, num_qubits).into_inner();
    let dim = 1usize << num_qubits;
    // to_matrix is column-major-agnostic; take explicit columns to match the
    // simulator reconstruction.
    let dense: Vec<Vec<num_complex::Complex64>> = (0..dim)
        .map(|col| (0..dim).map(|row| matrix[(row, col)]).collect())
        .collect();
    assert_equal_up_to_phase(gate_type, &dense, &simulator_unitary(gate_type, num_qubits));
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
        check(gate_type, 1);
    }
}

#[test]
fn dense_matrices_match_the_simulator_for_two_qubit_cliffords() {
    for gate_type in [
        GateType::CX,
        GateType::CY,
        GateType::CZ,
        GateType::SXX,
        GateType::SXXdg,
        GateType::SYY,
        GateType::SYYdg,
        GateType::SZZ,
        GateType::SZZdg,
        GateType::SWAP,
    ] {
        check(gate_type, 2);
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
                apply(&mut sim, gate_type, &[QubitId(0)]);
            }
            sim.state()
        };
        let native = {
            let mut sim = StateVecSoA::new(1);
            sim.h(&[QubitId(0)]);
            sim.sz(&[QubitId(0)]);
            apply(&mut sim, face, &[QubitId(0)]);
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
