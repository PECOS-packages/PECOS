// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
// either express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Frame rotations use the projective `StabMps` oracle; frame scalars use a dense oracle.

use num_complex::Complex64;
use pecos_core::{Angle64, QubitId};
use pecos_random::PecosRng;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, DenseStateVec};
use pecos_stab_tn::stab_mps::{PauliKind, StabMps};

#[derive(Clone, Copy, Debug)]
enum Op {
    H,
    S,
    Sdg,
    X,
    Y,
    Z,
    Cx,
    Cz,
    Rz,
    Rx,
    Ry,
    Rzz,
    Rxx,
    Ryy,
    T,
    Tdg,
    U,
    Rxy,
    Rxyxy,
    Rxxryyrzz,
    U2q,
}

fn apply(sim: &mut impl ArbitraryRotationGateable, op: Op, q: usize, r: usize, theta: Angle64) {
    let qs = &[QubitId(q)];
    let pairs = &[(QubitId(q), QubitId(r))];
    let phi = Angle64::from_radians(0.23);
    let lambda = Angle64::from_radians(-0.41);
    match op {
        Op::H => sim.h(qs),
        Op::S => sim.sz(qs),
        Op::Sdg => sim.szdg(qs),
        Op::X => sim.x(qs),
        Op::Y => sim.y(qs),
        Op::Z => sim.z(qs),
        Op::Cx => sim.cx(pairs),
        Op::Cz => sim.cz(pairs),
        Op::Rz => sim.rz(theta, qs),
        Op::Rx => sim.rx(theta, qs),
        Op::Ry => sim.ry(theta, qs),
        Op::Rzz => sim.rzz(theta, pairs),
        Op::Rxx => sim.rxx(theta, pairs),
        Op::Ryy => sim.ryy(theta, pairs),
        Op::T => sim.t(qs),
        Op::Tdg => sim.tdg(qs),
        Op::U => sim.u(theta, phi, lambda, qs),
        Op::Rxy => sim.rxy1q(theta, phi, qs),
        Op::Rxyxy => sim.rxyxy2q(theta, phi, pairs),
        Op::Rxxryyrzz => sim.rxxryyrzz(theta, phi, lambda, pairs),
        Op::U2q => sim.u2q(
            [[theta, phi, lambda]; 2],
            [theta, phi, lambda],
            [[phi, lambda, theta]; 2],
            pairs,
        ),
    };
}

fn inject(sim: &mut StabMps, kind: PauliKind, q: usize) {
    sim.inject_paulis_in_frame(&[(QubitId(q), kind)]);
}

fn pauli(sim: &mut impl CliffordGateable, kind: PauliKind, q: usize) {
    match kind {
        PauliKind::X => sim.x(&[QubitId(q)]),
        PauliKind::Y => sim.y(&[QubitId(q)]),
        PauliKind::Z => sim.z(&[QubitId(q)]),
    };
}

fn pair(n: usize, merge: bool, seed: u64) -> (StabMps, StabMps) {
    let builder = || {
        StabMps::builder(n)
            .seed(seed)
            .merge_rz(merge)
            .svd_cutoff(0.0)
            .max_truncation_error(0.0)
    };
    (
        builder().pauli_frame_tracking(true).build(),
        builder().build(),
    )
}

fn states(on: &StabMps, off: &StabMps) -> (Vec<Complex64>, Vec<Complex64>) {
    let mut on = on.clone();
    let mut off = off.clone();
    on.flush_pauli_frame_to_state();
    off.flush();
    (on.state_vector(), off.state_vector())
}

fn assert_projective(a: &[Complex64], b: &[Complex64], label: &str) {
    let overlap: Complex64 = a.iter().zip(b).map(|(x, y)| x * y.conj()).sum();
    assert!(overlap.norm() > 1e-12, "{label}: orthogonal states");
    let phase = overlap / overlap.norm();
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(
            (*x - phase * y).norm() < 2e-10,
            "{label}: amplitude {i}: {x} vs {y}, phase {phase}"
        );
    }
}

fn rotation_case(op: Op, kind: PauliKind, both: bool) {
    let n = if matches!(op, Op::Rzz) { 2 } else { 1 };
    let qubits: Vec<_> = (0..n).map(QubitId).collect();
    for merge in [false, true] {
        let (mut on, mut off) = pair(n, merge, 946);
        for sim in [&mut on, &mut off] {
            sim.h(&qubits);
            apply(sim, Op::Ry, 0, 1, Angle64::from_radians(0.17));
            apply(sim, op, 0, 1, Angle64::from_radians(0.3));
        }
        inject(&mut on, kind, 0);
        pauli(&mut off, kind, 0);
        if both {
            inject(&mut on, kind, 1);
            pauli(&mut off, kind, 1);
        }
        for sim in [&mut on, &mut off] {
            apply(sim, op, 0, 1, Angle64::from_radians(0.7));
            sim.h(&qubits);
        }
        let (a, b) = states(&on, &off);
        assert_projective(&a, &b, &format!("{op:?}, merge={merge}, both={both}"));
    }
}

#[test]
fn rz_reproducer() {
    let (mut on, mut off) = pair(1, false, 946);
    for sim in [&mut on, &mut off] {
        sim.h(&[QubitId(0)])
            .rz(Angle64::from_radians(0.3), &[QubitId(0)]);
    }
    on.inject_x_in_frame(QubitId(0));
    off.x(&[QubitId(0)]);
    for sim in [&mut on, &mut off] {
        sim.rz(Angle64::from_radians(0.7), &[QubitId(0)])
            .h(&[QubitId(0)]);
    }
    let (a, b) = states(&on, &off);
    assert!((b[1].norm_sqr() - 0.2_f64.sin().powi(2)).abs() < 1e-12);
    assert_projective(&a, &b, "RZ reproducer");
}

#[test]
fn rx_z_frame() {
    rotation_case(Op::Rx, PauliKind::Z, false);
}
#[test]
fn ry_x_frame() {
    rotation_case(Op::Ry, PauliKind::X, false);
}
#[test]
fn rzz_one_x_frame() {
    rotation_case(Op::Rzz, PauliKind::X, false);
}
#[test]
fn rzz_two_x_frames() {
    rotation_case(Op::Rzz, PauliKind::X, true);
}

#[test]
fn pending_rz_injection_and_physical_x() {
    let (mut on, mut off) = pair(1, true, 946);
    for sim in [&mut on, &mut off] {
        sim.h(&[QubitId(0)])
            .rz(Angle64::from_radians(0.3), &[QubitId(0)]);
    }
    on.inject_x_in_frame(QubitId(0));
    off.x(&[QubitId(0)]);
    for sim in [&mut on, &mut off] {
        sim.rz(Angle64::from_radians(0.7), &[QubitId(0)]);
        sim.x(&[QubitId(0)]);
        sim.rz(Angle64::from_radians(0.19), &[QubitId(0)]);
    }
    let (a, b) = states(&on, &off);
    assert_projective(&a, &b, "pending RZ with injected and physical X");
}

#[test]
fn all_rotation_entries_and_clifford_angles() {
    let ops = [
        Op::Rz,
        Op::Rx,
        Op::Ry,
        Op::Rzz,
        Op::Rxx,
        Op::Ryy,
        Op::T,
        Op::Tdg,
        Op::U,
        Op::Rxy,
        Op::Rxyxy,
        Op::Rxxryyrzz,
        Op::U2q,
    ];
    for op in ops {
        for theta in [
            Angle64::ZERO,
            Angle64::QUARTER_TURN,
            Angle64::HALF_TURN,
            Angle64::THREE_QUARTERS_TURN,
            Angle64::from_radians(0.43),
        ] {
            for merge in [false, true] {
                let (mut on, mut off) = pair(2, merge, 946);
                for sim in [&mut on, &mut off] {
                    sim.h(&[QubitId(0), QubitId(1)])
                        .rz(Angle64::from_radians(0.31), &[QubitId(0)]);
                }
                on.inject_x_in_frame(QubitId(0));
                off.x(&[QubitId(0)]);
                for sim in [&mut on, &mut off] {
                    apply(sim, op, 0, 1, theta);
                }
                let (a, b) = states(&on, &off);
                assert_projective(&a, &b, &format!("{op:?}, {theta:?}, merge={merge}"));
            }
        }
    }
}

#[test]
fn randomized_rotations_and_measurement_probabilities() {
    let mut rng = PecosRng::seed_from_u64(946);
    let ops = [
        Op::H,
        Op::S,
        Op::Sdg,
        Op::X,
        Op::Y,
        Op::Z,
        Op::Cx,
        Op::Cz,
        Op::Rz,
        Op::Rx,
        Op::Rzz,
    ];
    for circuit in 0..240 {
        let n = 1 + circuit % 6;
        let (mut on, mut off) = pair(n, (circuit / 6) % 2 == 0, circuit as u64);
        let depth = 1 + rng.next_u64() as usize % 40;
        for step in 0..depth {
            let q = rng.next_u64() as usize % n;
            let r = (q + 1 + rng.next_u64() as usize % n.max(2).saturating_sub(1)) % n;
            if rng.next_u64().is_multiple_of(4) {
                let kind = [PauliKind::X, PauliKind::Y, PauliKind::Z][rng.next_u64() as usize % 3];
                inject(&mut on, kind, q);
                pauli(&mut off, kind, q);
            } else {
                let mut op = ops[rng.next_u64() as usize % ops.len()];
                if n == 1 && matches!(op, Op::Cx | Op::Cz | Op::Rzz) {
                    op = Op::Rz;
                }
                let theta = match rng.next_u64() % 5 {
                    0 => Angle64::QUARTER_TURN,
                    1 => Angle64::HALF_TURN,
                    2 => Angle64::THREE_QUARTERS_TURN,
                    _ => Angle64::from_radians((rng.next_u64() % 6001) as f64 / 1000.0 - 3.0),
                };
                apply(&mut on, op, q, r, theta);
                apply(&mut off, op, q, r, theta);
            }
            if step % 7 == 0 {
                // Compare the two MZ branch probabilities without sampling, so
                // stochastic branch selection cannot mask a rotation error.
                let (a, b) = states(&on, &off);
                for outcome in [false, true] {
                    let probability = |sv: &[Complex64]| -> f64 {
                        sv.iter()
                            .enumerate()
                            .filter(|(i, _)| ((i >> q) & 1 != 0) == outcome)
                            .map(|(_, a)| a.norm_sqr())
                            .sum()
                    };
                    assert!(
                        (probability(&a) - probability(&b)).abs() < 2e-10,
                        "circuit {circuit}, step {step}, MZ {q}"
                    );
                }
            }
        }
        let (a, b) = states(&on, &off);
        assert_projective(&a, &b, &format!("circuit {circuit}, n={n}, depth={depth}"));
    }
}

fn assert_dense(on: &mut StabMps, dense: &mut DenseStateVec, label: &str) {
    on.flush_pauli_frame_to_state();
    for (i, a) in on.state_vector().iter().enumerate() {
        let b = dense.get_amplitude(i);
        assert!(
            (*a - b).norm() < 2e-10,
            "{label}: amplitude {i}: {a} vs {b}"
        );
    }
}

#[test]
fn frame_cx_exact_phase() {
    let mut on = StabMps::builder(2).pauli_frame_tracking(true).build();
    let mut dense = DenseStateVec::new(2);
    on.inject_x_in_frame(QubitId(0));
    on.inject_z_in_frame(QubitId(1));
    dense.x(&[QubitId(0)]).z(&[QubitId(1)]);
    on.cx(&[(QubitId(0), QubitId(1))]);
    dense.cx(&[(QubitId(0), QubitId(1))]);
    assert_dense(&mut on, &mut dense, "CX(Xc Zt)");
}

fn sequential_injection_case(first: PauliKind, second: PauliKind) {
    let mut on = StabMps::builder(1).pauli_frame_tracking(true).build();
    let mut dense = DenseStateVec::new(1);
    inject(&mut on, first, 0);
    inject(&mut on, second, 0);
    pauli(&mut dense, first, 0);
    pauli(&mut dense, second, 0);
    assert_dense(&mut on, &mut dense, &format!("{first:?} then {second:?}"));
}

#[test]
fn frame_x_then_z_exact_phase() {
    sequential_injection_case(PauliKind::X, PauliKind::Z);
}

#[test]
fn frame_y_then_x_exact_phase() {
    sequential_injection_case(PauliKind::Y, PauliKind::X);
}

#[test]
fn sequential_injections_exact_phase() {
    for first in [PauliKind::X, PauliKind::Y, PauliKind::Z] {
        for second in [PauliKind::X, PauliKind::Y, PauliKind::Z] {
            sequential_injection_case(first, second);
        }
    }
}

#[test]
fn noise_channels_before_rotations() {
    for channel in 0..3 {
        let (mut on, mut off) = pair(2, true, 946);
        for round in 0..8 {
            let q = QubitId(round % 2);
            for sim in [&mut on, &mut off] {
                sim.h(&[q]).rz(Angle64::from_radians(0.31), &[q]);
            }
            match channel {
                0 => assert_eq!(on.apply_bit_flip(q, 1.0), off.apply_bit_flip(q, 1.0)),
                1 => assert_eq!(on.apply_phase_flip(q, 1.0), off.apply_phase_flip(q, 1.0)),
                _ => assert_eq!(
                    on.apply_depolarizing(q, 1.0),
                    off.apply_depolarizing(q, 1.0)
                ),
            }
            for sim in [&mut on, &mut off] {
                sim.rz(Angle64::from_radians(0.71), &[q]);
                sim.rzz(Angle64::from_radians(0.19), &[(QubitId(0), QubitId(1))]);
            }
        }
        let (a, b) = states(&on, &off);
        assert_projective(&a, &b, &format!("noise channel {channel}"));
    }
}

#[test]
fn pending_rz_y_and_phase_exact_u() {
    for tail in [
        Angle64::ZERO,
        Angle64::QUARTER_TURN,
        Angle64::HALF_TURN,
        Angle64::from_radians(0.19),
    ] {
        let (mut on, mut off) = pair(1, true, 946);
        let first = Angle64::from_radians(0.3);
        for sim in [&mut on, &mut off] {
            sim.h(&[QubitId(0)]).rz(first, &[QubitId(0)]);
        }
        on.inject_y_in_frame(QubitId(0));
        off.y(&[QubitId(0)]);
        for sim in [&mut on, &mut off] {
            sim.rz(first + tail, &[QubitId(0)]);
            sim.y(&[QubitId(0)]);
            sim.u(
                Angle64::QUARTER_TURN,
                Angle64::HALF_TURN,
                first,
                &[QubitId(0)],
            );
        }
        let (a, b) = states(&on, &off);
        assert_projective(&a, &b, "pending Y and U, including merged Clifford angles");
    }
}

#[test]
fn every_frame_clifford_conjugation_exact_phase() {
    // These preparations keep the represented tableau's canonical scalar
    // unchanged under the tested gate, isolating the frame's exact operator.
    for op in [Op::H, Op::S, Op::Sdg, Op::X, Op::Y, Op::Z, Op::Cx, Op::Cz] {
        for first in [
            None,
            Some(PauliKind::X),
            Some(PauliKind::Y),
            Some(PauliKind::Z),
        ] {
            for second in [
                None,
                Some(PauliKind::X),
                Some(PauliKind::Y),
                Some(PauliKind::Z),
            ] {
                let mut on = StabMps::builder(2).pauli_frame_tracking(true).build();
                let mut dense = DenseStateVec::new(2);
                if matches!(op, Op::X | Op::Y) {
                    on.h(&[QubitId(0)]);
                    dense.h(&[QubitId(0)]);
                }
                if matches!(op, Op::Y) {
                    on.sz(&[QubitId(0)]);
                    dense.sz(&[QubitId(0)]);
                }
                for (q, kind) in [first, second].into_iter().enumerate() {
                    if let Some(kind) = kind {
                        inject(&mut on, kind, q);
                        pauli(&mut dense, kind, q);
                    }
                }
                apply(&mut on, op, 0, 1, Angle64::ZERO);
                apply(&mut dense, op, 0, 1, Angle64::ZERO);
                assert_dense(
                    &mut on,
                    &mut dense,
                    &format!("{op:?} on frame {first:?}, {second:?}"),
                );
            }
        }
    }
}

#[test]
fn scalar_only_frame_flush() {
    let mut on = StabMps::builder(1).pauli_frame_tracking(true).build();
    let mut dense = DenseStateVec::new(1);
    for kind in [PauliKind::X, PauliKind::Z, PauliKind::X, PauliKind::Z] {
        inject(&mut on, kind, 0);
        pauli(&mut dense, kind, 0);
    }
    assert!(!on.frame_x_bit(QubitId(0)) && !on.frame_z_bit(QubitId(0)));
    assert_dense(&mut on, &mut dense, "scalar -1 with identity support");
    assert_dense(
        &mut on,
        &mut dense,
        "second flush does not reapply the scalar",
    );
}

#[test]
fn half_turn_after_x_frame_exact_phase() {
    // Angle64 identifies RZ(pi) and RZ(-pi), so conjugation by the frame
    // shows up only in the global phase: X RZ(pi) X = +iZ, not -iZ.
    for merge in [false, true] {
        let mut on = StabMps::builder(1)
            .merge_rz(merge)
            .pauli_frame_tracking(true)
            .build();
        let mut dense = DenseStateVec::new(1);
        on.h(&[QubitId(0)]);
        dense.h(&[QubitId(0)]);
        on.inject_x_in_frame(QubitId(0));
        dense.x(&[QubitId(0)]);
        on.rz(Angle64::HALF_TURN, &[QubitId(0)]);
        dense.rz(Angle64::HALF_TURN, &[QubitId(0)]);
        assert_dense(
            &mut on,
            &mut dense,
            &format!("X frame then RZ(pi), merge={merge}"),
        );
    }
}
