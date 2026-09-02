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

//! The dense matrix of a parameterised gate must equal the operator the simulator applies,
//! entrywise -- not up to a global phase.
//!
//! `Angle64` stores an angle mod 2pi, so `RZ(theta)` and `RZ(theta + 2pi) = -RZ(theta)` are
//! indistinguishable once stored. Both the simulators and the dense-matrix path therefore have
//! to choose a representative, and they must choose the same one. The simulators halve the
//! signed representative (`to_radians_signed`, then `/ 2`). This test pins the dense path to the
//! same choice for every angle, including negative ones and the boundary at exactly pi.
//!
//! This comparison is deliberately exact. A projective comparison cannot see a global `-1`, and
//! a global `-1` on a block becomes a relative phase the moment that block is controlled.

use num_complex::Complex64;
use pecos_core::unitary_rep::RotationType;
use pecos_core::{Angle64, QubitId, Unitary, UnitaryRep};
use pecos_quantum::ToMatrix;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVecSoA};
use smallvec::smallvec;
use std::f64::consts::PI;

/// Column `b` is the simulator's image of basis state `|b>`, with qubit 0 the least-significant bit.
fn simulator_unitary(num_qubits: usize, apply: &dyn Fn(&mut StateVecSoA)) -> Vec<Vec<Complex64>> {
    let dim = 1usize << num_qubits;
    (0..dim)
        .map(|basis| {
            let mut sim = StateVecSoA::new(num_qubits);
            for bit in 0..num_qubits {
                if basis & (1 << bit) != 0 {
                    sim.x(&[QubitId(bit)]);
                }
            }
            apply(&mut sim);
            sim.state()
        })
        .collect()
}

fn assert_exactly_equal(
    name: &str,
    dense: &UnitaryRep,
    num_qubits: usize,
    apply: &dyn Fn(&mut StateVecSoA),
) {
    let dense = dense.to_matrix();
    let dense = dense.inner();
    let sim = simulator_unitary(num_qubits, apply);
    let dim = 1usize << num_qubits;
    let mut worst = 0.0f64;
    let mut worst_at = (0, 0);
    for col in 0..dim {
        for row in 0..dim {
            let err = (dense[(row, col)] - sim[col][row]).norm();
            if err > worst {
                worst = err;
                worst_at = (row, col);
            }
        }
    }
    assert!(
        worst < 1e-12,
        "{name}: dense matrix differs from the simulator by {worst:.3e} at {worst_at:?} \
         (dense {:.6}, simulator {:.6}); a global -1 here is a defect, not a convention",
        dense[worst_at],
        sim[worst_at.1][worst_at.0]
    );
}

/// Every angle class that matters: positive, negative, past +-pi/2, exactly pi, and 3pi/2 (== -pi/2).
const ANGLES: [f64; 7] = [0.37, -0.37, 2.9, -2.9, PI, 1.5 * PI, -PI / 2.0];

#[test]
fn single_qubit_rotations_match_exactly() {
    for &rad in &ANGLES {
        let a = Angle64::from_radians(rad);
        let q = [QubitId(0)];
        for (name, rt, apply) in [
            (
                "RX",
                RotationType::RX,
                Box::new(move |s: &mut StateVecSoA| {
                    s.rx(a, &q);
                }) as Box<dyn Fn(&mut StateVecSoA)>,
            ),
            (
                "RY",
                RotationType::RY,
                Box::new(move |s: &mut StateVecSoA| {
                    s.ry(a, &q);
                }),
            ),
            (
                "RZ",
                RotationType::RZ,
                Box::new(move |s: &mut StateVecSoA| {
                    s.rz(a, &q);
                }),
            ),
        ] {
            let dense = UnitaryRep::rotation(rt, a, smallvec![0usize]);
            assert_exactly_equal(&format!("{name}({rad:+.3})"), &dense, 1, &*apply);
        }
    }
}

#[test]
fn two_qubit_rotations_match_exactly() {
    for &rad in &ANGLES {
        let a = Angle64::from_radians(rad);
        let pair = [(QubitId(0), QubitId(1))];
        for (name, rt, apply) in [
            (
                "RXX",
                RotationType::RXX,
                Box::new(move |s: &mut StateVecSoA| {
                    s.rxx(a, &pair);
                }) as Box<dyn Fn(&mut StateVecSoA)>,
            ),
            (
                "RYY",
                RotationType::RYY,
                Box::new(move |s: &mut StateVecSoA| {
                    s.ryy(a, &pair);
                }),
            ),
            (
                "RZZ",
                RotationType::RZZ,
                Box::new(move |s: &mut StateVecSoA| {
                    s.rzz(a, &pair);
                }),
            ),
        ] {
            let dense = UnitaryRep::rotation(rt, a, smallvec![0usize, 1]);
            assert_exactly_equal(&format!("{name}({rad:+.3})"), &dense, 2, &*apply);
        }
    }
}

#[test]
fn multi_angle_single_qubit_gates_match_exactly() {
    let q = [QubitId(0)];
    for &t in &ANGLES {
        for &p in &[0.0, 1.1, -1.1] {
            let (theta, phi) = (Angle64::from_radians(t), Angle64::from_radians(p));
            let dense = UnitaryRep::Gate(Unitary::RXY1Q { theta, phi }, smallvec![0usize]);
            assert_exactly_equal(&format!("RXY1Q({t:+.3}, {p:+.3})"), &dense, 1, &move |s| {
                s.rxy1q(theta, phi, &q);
            });
            for &l in &[0.0, -0.7] {
                let lambda = Angle64::from_radians(l);
                let dense = UnitaryRep::Gate(Unitary::U3 { theta, phi, lambda }, smallvec![0usize]);
                assert_exactly_equal(
                    &format!("U({t:+.3}, {p:+.3}, {l:+.3})"),
                    &dense,
                    1,
                    &move |s| {
                        s.u(theta, phi, lambda, &q);
                    },
                );
            }
        }
    }
}

#[test]
fn named_negative_angle_cliffords_match_exactly() {
    // SZZdg is built from a three-quarter-turn RZZ; it is the named gate most exposed to the
    // choice of representative.
    let pair = [(QubitId(0), QubitId(1))];
    let dense = UnitaryRep::gate(pecos_core::gate_type::GateType::SZZdg, smallvec![0usize, 1]);
    assert_exactly_equal("SZZdg", &dense, 2, &move |s| {
        s.szzdg(&pair);
    });
    let dense = UnitaryRep::gate(pecos_core::gate_type::GateType::SZZ, smallvec![0usize, 1]);
    assert_exactly_equal("SZZ", &dense, 2, &move |s| {
        s.szz(&pair);
    });
}
