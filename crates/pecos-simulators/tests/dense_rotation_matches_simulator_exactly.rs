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
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, MeasurementResult, QuantumSimulator, StateVecSoA,
};
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
    // SXXdg and SYYdg are built the same way and were found to carry the same defect.
    let pair = [(QubitId(0), QubitId(1))];
    for (name, gate, apply) in [
        (
            "SXXdg",
            pecos_core::gate_type::GateType::SXXdg,
            Box::new(move |s: &mut StateVecSoA| {
                s.sxxdg(&pair);
            }) as Box<dyn Fn(&mut StateVecSoA)>,
        ),
        (
            "SYYdg",
            pecos_core::gate_type::GateType::SYYdg,
            Box::new(move |s: &mut StateVecSoA| {
                s.syydg(&pair);
            }),
        ),
    ] {
        let dense = UnitaryRep::gate(gate, smallvec![0usize, 1]);
        assert_exactly_equal(name, &dense, 2, &*apply);
    }

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

/// A backend that implements ONLY the trait's required methods, delegating them to a real
/// state vector, so every default decomposition in `ArbitraryRotationGateable` and
/// `CliffordGateable` runs on real amplitudes. `StateVecSoA` and `StateVecAoS` override
/// `rxy1q` directly and so cannot exhibit a defect in the shared default; the sparse, f32,
/// stabilizer-vector and GPU backends inherit it.
struct DefaultsOnly(StateVecSoA);

impl QuantumSimulator for DefaultsOnly {
    fn reset(&mut self) -> &mut Self {
        self.0.reset();
        self
    }
    fn num_qubits(&self) -> usize {
        self.0.num_qubits()
    }
}

impl CliffordGateable for DefaultsOnly {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.0.sz(qubits);
        self
    }
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.0.h(qubits);
        self
    }
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.0.cx(pairs);
        self
    }
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.0.mz(qubits)
    }
}

impl ArbitraryRotationGateable for DefaultsOnly {
    fn rx(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        self.0.rx(theta, qubits);
        self
    }
    fn rz(&mut self, theta: Angle64, qubits: &[QubitId]) -> &mut Self {
        self.0.rz(theta, qubits);
        self
    }
    fn rzz(&mut self, theta: Angle64, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.0.rzz(theta, pairs);
        self
    }
}

fn defaults_only_unitary(
    num_qubits: usize,
    apply: &dyn Fn(&mut DefaultsOnly),
) -> Vec<Vec<Complex64>> {
    let dim = 1usize << num_qubits;
    (0..dim)
        .map(|basis| {
            let mut sim = DefaultsOnly(StateVecSoA::new(num_qubits));
            for bit in 0..num_qubits {
                if basis & (1 << bit) != 0 {
                    sim.x(&[QubitId(bit)]);
                }
            }
            apply(&mut sim);
            sim.0.state()
        })
        .collect()
}

fn assert_default_matches_dense(
    name: &str,
    dense: &UnitaryRep,
    num_qubits: usize,
    apply: &dyn Fn(&mut DefaultsOnly),
) {
    let dense = dense.to_matrix();
    let dense = dense.inner();
    let got = defaults_only_unitary(num_qubits, apply);
    let dim = 1usize << num_qubits;
    for col in 0..dim {
        for row in 0..dim {
            let err = (dense[(row, col)] - got[col][row]).norm();
            assert!(
                err < 1e-12,
                "{name}: the trait DEFAULT decomposition differs from the dense matrix by {err:.3e} at ({row}, {col}) \
                 (dense {:.6}, default {:.6}); a stored angle of +pi and -pi are the same Angle64, so a \
                 decomposition whose exactness relies on two angles cancelling as reals is not exact",
                dense[(row, col)],
                got[col][row]
            );
        }
    }
}

#[test]
fn trait_default_decompositions_match_the_dense_matrices_exactly() {
    let q = [QubitId(0)];
    // The axis angles that matter: the two where pi/2 - phi and phi - pi/2 both land on a stored
    // half turn, plus ordinary ones.
    for &p in &[-PI / 2.0, PI / 2.0, 0.0, PI, 0.67, -1.1] {
        for &t in &ANGLES {
            let (theta, phi) = (Angle64::from_radians(t), Angle64::from_radians(p));
            let dense = UnitaryRep::Gate(Unitary::RXY1Q { theta, phi }, smallvec![0usize]);
            assert_default_matches_dense(
                &format!("default rxy1q({t:+.3}, {p:+.3})"),
                &dense,
                1,
                &move |s| {
                    s.rxy1q(theta, phi, &q);
                },
            );
            let lambda = Angle64::from_radians(-0.7);
            let dense = UnitaryRep::Gate(Unitary::U3 { theta, phi, lambda }, smallvec![0usize]);
            assert_default_matches_dense(
                &format!("default u({t:+.3}, {p:+.3}, -0.7)"),
                &dense,
                1,
                &move |s| {
                    s.u(theta, phi, lambda, &q);
                },
            );
        }
    }
    let pair = [(QubitId(0), QubitId(1))];
    for &t in &ANGLES {
        let theta = Angle64::from_radians(t);
        assert_default_matches_dense(
            &format!("default ry({t:+.3})"),
            &UnitaryRep::rotation(RotationType::RY, theta, smallvec![0usize]),
            1,
            &move |s| {
                s.ry(theta, &q);
            },
        );
        assert_default_matches_dense(
            &format!("default rxx({t:+.3})"),
            &UnitaryRep::rotation(RotationType::RXX, theta, smallvec![0usize, 1]),
            2,
            &move |s| {
                s.rxx(theta, &pair);
            },
        );
        assert_default_matches_dense(
            &format!("default ryy({t:+.3})"),
            &UnitaryRep::rotation(RotationType::RYY, theta, smallvec![0usize, 1]),
            2,
            &move |s| {
                s.ryy(theta, &pair);
            },
        );
    }
}

/// `to_radians_signed` must decide the sign from the stored fraction, not from an `f64` that has
/// already rounded away the difference between a half turn and one tick above it.
#[test]
fn signed_radians_are_decided_on_the_raw_fraction() {
    let one_tick_above_half = Angle64::new(Angle64::HALF_TURN.fraction() + 1);
    assert!(
        one_tick_above_half.to_radians_signed() < 0.0,
        "one tick above a half turn is a negative signed angle, got {}",
        one_tick_above_half.to_radians_signed()
    );
    assert_eq!(
        Angle64::HALF_TURN.to_radians_signed(),
        PI,
        "exactly a half turn is +pi"
    );
    let one_tick_below_half = Angle64::new(Angle64::HALF_TURN.fraction() - 1);
    assert!(one_tick_below_half.to_radians_signed() > 0.0);
}
