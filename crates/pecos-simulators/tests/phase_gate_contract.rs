// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! The `Phase(gamma)` contract.
//!
//! One rule: `Phase(gamma)` on an operand set `S` multiplies the amplitude by `e^{i gamma}` on the
//! subspace where every qubit in `S` is `|1>`, and leaves every other amplitude alone.
//!
//! - `S = {}`   -> a global phase: the zero-qubit identity with `with_phase(gamma)` applied;
//!   the hardware lowering emits no gates for it (hardware cannot see a scalar)
//! - `S = {q}`  -> `diag(1, e^{i gamma})`, which is exactly `U(0, 0, gamma)`
//! - `S = {c,t}` -> `diag(1, 1, 1, e^{i gamma})`
//! - `|S| > 2`  -> unsupported; the constructor panics like every other arity-checked constructor
//!
//! `control()` is structural: controlling `Phase` on `S` by `c` is `Phase` on `S + {c}`.
//!
//! Three properties are pinned here, and each one names the API it expects. This file is the
//! contract; the implementation is judged against it, not the other way round.
//!
//! 1. The matrix comes from the rule, directly. A phase is 2pi-periodic, so unlike a rotation it
//!    has no half-angle and no representative problem: negative angles need no special case.
//! 2. The hardware lowering (`lower_phase`) produces the SAME matrix, entrywise, not up to a
//!    global phase. Layer 2 agrees with layer 1 exactly.
//! 3. `control()` composes with both, exactly, and refuses what it cannot represent.

use num_complex::Complex64;
use pecos_core::controlled_rotations::lower_phase;
use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, QubitId, Unitary, UnitaryRep};
use pecos_quantum::unitary_matrix::to_matrix_with_size;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVecSoA};
use smallvec::smallvec;
use std::f64::consts::PI;

/// `lower_phase` may emit exactly three gate kinds: RZ, RZZ, and U. Anything else is a contract
/// violation, so both mappers below fail loud on it.
fn lowered_as_rep(gate: &pecos_core::Gate) -> UnitaryRep {
    use pecos_core::unitary_rep::RotationType;
    let qubits: smallvec::SmallVec<[usize; 3]> =
        gate.qubits.iter().map(|q| usize::from(*q)).collect();
    match gate.gate_type {
        GateType::RZ => UnitaryRep::rotation(RotationType::RZ, gate.angles[0], qubits),
        GateType::RZZ => UnitaryRep::rotation(RotationType::RZZ, gate.angles[0], qubits),
        GateType::U => UnitaryRep::Gate(
            Unitary::U3 {
                theta: gate.angles[0],
                phi: gate.angles[1],
                lambda: gate.angles[2],
            },
            qubits,
        ),
        other => panic!("lower_phase emitted {other:?}; the contract allows only RZ, RZZ, U"),
    }
}

fn apply_lowered(sim: &mut StateVecSoA, gate: &pecos_core::Gate) {
    match gate.gate_type {
        GateType::RZ => {
            sim.rz(gate.angles[0], &gate.qubits);
        }
        GateType::RZZ => {
            sim.rzz(gate.angles[0], &[(gate.qubits[0], gate.qubits[1])]);
        }
        GateType::U => {
            sim.u(gate.angles[0], gate.angles[1], gate.angles[2], &gate.qubits);
        }
        other => panic!("lower_phase emitted {other:?}; the contract allows only RZ, RZZ, U"),
    }
}

const ANGLES: [f64; 9] = [
    0.37,
    -0.37,
    2.9,
    -2.9,
    PI,
    1.5 * PI,
    -PI / 2.0,
    2.0 * PI,
    -2.0 * PI,
];

/// The rule, written as a literal: `e^{i gamma}` at the all-ones index of `targets`, 1 elsewhere.
fn expected(gamma: f64, targets: &[usize], num_qubits: usize) -> Vec<Vec<Complex64>> {
    let dim = 1usize << num_qubits;
    let all_ones = targets.iter().fold(0usize, |m, &q| m | (1 << q));
    (0..dim)
        .map(|col| {
            (0..dim)
                .map(|row| {
                    if row != col {
                        Complex64::new(0.0, 0.0)
                    } else if col & all_ones == all_ones {
                        Complex64::new(0.0, gamma).exp()
                    } else {
                        Complex64::new(1.0, 0.0)
                    }
                })
                .collect()
        })
        .collect()
}

fn columns(rep: &UnitaryRep, num_qubits: usize) -> Vec<Vec<Complex64>> {
    let m = to_matrix_with_size(rep, num_qubits);
    let m = m.inner();
    let dim = 1usize << num_qubits;
    (0..dim)
        .map(|col| (0..dim).map(|row| m[(row, col)]).collect())
        .collect()
}

fn assert_columns_exact(name: &str, got: &[Vec<Complex64>], want: &[Vec<Complex64>]) {
    for (col, (g, w)) in got.iter().zip(want).enumerate() {
        for (row, (a, b)) in g.iter().zip(w).enumerate() {
            assert!(
                (a - b).norm() < 1e-12,
                "{name}: entry ({row}, {col}) is {a:.6}, expected {b:.6}"
            );
        }
    }
}

#[test]
fn phase_matrix_follows_the_rule_directly() {
    for &g in &ANGLES {
        let gamma = Angle64::from_radians(g);
        // Single qubit, both positions in a two-qubit register.
        for q in 0..2usize {
            let rep = UnitaryRep::phase_gate(gamma, smallvec![q]);
            assert_columns_exact(
                &format!("Phase({g:+.3}) on {{{q}}}"),
                &columns(&rep, 2),
                &expected(g, &[q], 2),
            );
        }
        // Two qubits, both orders: the rule is symmetric in S, so these must be identical.
        let ab = UnitaryRep::phase_gate(gamma, smallvec![0usize, 1]);
        let ba = UnitaryRep::phase_gate(gamma, smallvec![1usize, 0]);
        assert_columns_exact(
            &format!("Phase({g:+.3}) on {{0,1}}"),
            &columns(&ab, 2),
            &expected(g, &[0, 1], 2),
        );
        assert_columns_exact(
            &format!("Phase({g:+.3}) on {{1,0}}"),
            &columns(&ba, 2),
            &expected(g, &[0, 1], 2),
        );
        // Global: the zero-qubit identity carrying the phase, embedded in one qubit.
        let global = UnitaryRep::phase_gate(gamma, smallvec![]);
        assert_columns_exact(
            &format!("Phase({g:+.3}) on {{}}"),
            &columns(&global, 1),
            &expected(g, &[], 1),
        );
    }
}

#[test]
fn single_qubit_phase_is_exactly_u_0_0_gamma() {
    for &g in &ANGLES {
        let gamma = Angle64::from_radians(g);
        let phase = UnitaryRep::phase_gate(gamma, smallvec![0usize]);
        let u = UnitaryRep::Gate(
            Unitary::U3 {
                theta: Angle64::ZERO,
                phi: Angle64::ZERO,
                lambda: gamma,
            },
            smallvec![0usize],
        );
        assert_columns_exact(
            &format!("Phase({g:+.3}) vs U(0,0,gamma)"),
            &columns(&phase, 1),
            &columns(&u, 1),
        );
    }
}

#[test]
#[should_panic(expected = "Phase")]
fn phase_on_three_qubits_is_refused() {
    let _ = UnitaryRep::phase_gate(Angle64::from_radians(0.37), smallvec![0usize, 1, 2]);
}

/// Layer 2: the hardware lowering, composed as a dense matrix, equals layer 1 exactly.
#[test]
fn lowering_matches_the_rule_exactly_in_the_dense_path() {
    for &g in &ANGLES {
        for targets in [vec![0usize], vec![1usize], vec![0usize, 1], vec![1usize, 0]] {
            let qubits: Vec<QubitId> = targets.iter().map(|&q| QubitId(q)).collect();
            let lowered = lower_phase(g, &qubits);
            let composed = UnitaryRep::Compose(lowered.iter().map(lowered_as_rep).collect());
            let rule = UnitaryRep::phase_gate(
                Angle64::from_radians(g),
                smallvec::SmallVec::from_slice(&targets),
            );
            assert_columns_exact(
                &format!("lower_phase({g:+.3}, {targets:?})"),
                &columns(&composed, 2),
                &columns(&rule, 2),
            );
        }
        assert!(
            lower_phase(g, &[]).is_empty(),
            "a bare global phase lowers to no gates"
        );
    }
}

/// Layer 2, executed: applying the lowered gates on a simulator equals the rule's matrix.
#[test]
fn lowering_matches_the_rule_exactly_when_executed() {
    for &g in &ANGLES {
        for targets in [vec![0usize], vec![0usize, 1], vec![1usize, 0]] {
            let qubits: Vec<QubitId> = targets.iter().map(|&q| QubitId(q)).collect();
            let lowered = lower_phase(g, &qubits);
            let dim = 4usize;
            let got: Vec<Vec<Complex64>> = (0..dim)
                .map(|basis| {
                    let mut sim = StateVecSoA::new(2);
                    for bit in 0..2 {
                        if basis & (1 << bit) != 0 {
                            sim.x(&[QubitId(bit)]);
                        }
                    }
                    for gate in &lowered {
                        apply_lowered(&mut sim, gate);
                    }
                    sim.state()
                })
                .collect();
            assert_columns_exact(
                &format!("executed lower_phase({g:+.3}, {targets:?})"),
                &got,
                &expected(g, &targets, 2),
            );
        }
    }
}

#[test]
fn control_is_structural() {
    for &g in &ANGLES {
        let gamma = Angle64::from_radians(g);
        // control(Phase on {1}) by 0  ==  Phase on {0, 1}
        let controlled = UnitaryRep::phase_gate(gamma, smallvec![1usize])
            .control(0)
            .unwrap();
        assert_columns_exact(
            &format!("control(Phase({g:+.3}) on {{1}}, 0)"),
            &columns(&controlled, 2),
            &expected(g, &[0, 1], 2),
        );
        // control(global phase) by c  ==  Phase on {c}
        let promoted = UnitaryRep::phase_gate(gamma, smallvec![])
            .control(1)
            .unwrap();
        assert_columns_exact(
            &format!("control(Phase({g:+.3}) on {{}}, 1)"),
            &columns(&promoted, 2),
            &expected(g, &[1], 2),
        );
        // A control that is already in S, or a third qubit, is refused.
        assert!(
            UnitaryRep::phase_gate(gamma, smallvec![1usize])
                .control(1)
                .is_err(),
            "control already in S"
        );
        assert!(
            UnitaryRep::phase_gate(gamma, smallvec![0usize, 1])
                .control(2)
                .is_err()
        );
        // Anything that is not a Phase is refused rather than guessed at.
        assert!(
            UnitaryRep::gate(GateType::H, smallvec![0usize])
                .control(1)
                .is_err()
        );
    }
}

#[test]
fn clifford_membership_follows_the_angle_table() {
    let quarter = |k: u32| Angle64::from_radians(f64::from(k) * PI / 2.0);
    // One qubit: multiples of pi/2 are Cliffords and identify as I, SZ, Z, SZdg.
    for (k, want) in [
        (0, GateType::I),
        (1, GateType::SZ),
        (2, GateType::Z),
        (3, GateType::SZdg),
    ] {
        let rep = UnitaryRep::phase_gate(quarter(k), smallvec![0usize]);
        assert!(
            rep.is_clifford(),
            "Phase({k}·pi/2) on one qubit is Clifford"
        );
        assert_eq!(
            rep.to_named_gate(),
            Some(want),
            "Phase({k}·pi/2) on one qubit"
        );
    }
    assert!(!UnitaryRep::phase_gate(Angle64::from_radians(0.37), smallvec![0usize]).is_clifford());
    assert!(
        !UnitaryRep::phase_gate(Angle64::from_radians(PI / 4.0), smallvec![0usize]).is_clifford(),
        "T is not Clifford"
    );
    // Two qubits: only 0 and pi are Cliffords; pi is CZ.
    assert!(UnitaryRep::phase_gate(quarter(2), smallvec![0usize, 1]).is_clifford());
    assert_eq!(
        UnitaryRep::phase_gate(quarter(2), smallvec![0usize, 1]).to_named_gate(),
        Some(GateType::CZ)
    );
    assert!(
        !UnitaryRep::phase_gate(quarter(1), smallvec![0usize, 1]).is_clifford(),
        "Phase(pi/2) on two qubits is not Clifford"
    );
}
