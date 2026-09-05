// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use
// this file except in compliance with the License. You may obtain a copy of the
// License at https://www.apache.org/licenses/LICENSE-2.0

//! Differential coverage for pending RZ rotations paired with deferred Clifford frames.

use num_complex::Complex64;
use pecos_core::{Angle64, QubitId};
use pecos_simulators::state_vector_test_utils::assert_phase_exact_state_matches;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StabVec, StateVecSoA};
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};

const TOLERANCE: f64 = 1e-12;

fn qid(index: usize) -> [QubitId; 1] {
    [QubitId(index)]
}

#[test]
fn cx_materializes_x_negated_pending_rz_with_its_frame() {
    let mut stab = StabVec::builder(5).pruning_threshold(0.0).seed(7).build();
    let mut dense = StateVecSoA::with_seed(5, 7);

    stab.h(&qid(3));
    dense.h(&qid(3));
    stab.cx(&[(QubitId(3), QubitId(1))]);
    dense.cx(&[(QubitId(3), QubitId(1))]);
    stab.cx(&[(QubitId(1), QubitId(2))]);
    dense.cx(&[(QubitId(1), QubitId(2))]);
    let angle = Angle64::from_radians(std::f64::consts::FRAC_PI_4);
    stab.rz(angle, &qid(2)).x(&qid(2));
    dense.rz(angle, &qid(2)).x(&qid(2));

    // Observe the state without disturbing the deferred frame/rotation in the
    // simulator used below. Control qubit 4 is |0>, so the final CX is identity.
    let before_final_cx = stab.clone().state_vector();
    stab.cx(&[(QubitId(4), QubitId(2))]);
    dense.cx(&[(QubitId(4), QubitId(2))]);

    let actual = stab.state_vector();
    let expected = dense.state();
    assert_phase_exact_state_matches(
        &actual,
        &before_final_cx,
        TOLERANCE,
        "CX with a zero control changed the StabVec state",
    );
    assert_phase_exact_state_matches(
        &actual,
        &expected,
        TOLERANCE,
        "StabVec pending-RZ/frame CX regression",
    );
}

fn prepare_negated_pending_rotations(stab: &mut StabVec, dense: &mut StateVecSoA) {
    stab.h(&qid(0)).sz(&qid(0)).h(&qid(1)).szdg(&qid(1));
    dense.h(&qid(0)).sz(&qid(0)).h(&qid(1)).szdg(&qid(1));
    stab.cx(&[(QubitId(0), QubitId(1))]);
    dense.cx(&[(QubitId(0), QubitId(1))]);

    // Start the scenario from a materialized, phase-exact entangled state.
    let actual_input = stab.state_vector();
    let expected_input = dense.state();
    assert_phase_exact_state_matches(
        &actual_input,
        &expected_input,
        TOLERANCE,
        "two-qubit pending-RZ test input",
    );

    let x_angle = Angle64::from_radians(0.37);
    let y_angle = Angle64::from_radians(-0.53);
    stab.rz(x_angle, &qid(0))
        .x(&qid(0))
        .rz(y_angle, &qid(1))
        .y(&qid(1));
    dense
        .rz(x_angle, &qid(0))
        .x(&qid(0))
        .rz(y_angle, &qid(1))
        .y(&qid(1));
}

#[test]
fn public_pending_rz_flush_keeps_each_negated_angle_with_its_frame() {
    let mut stab = StabVec::builder(2)
        .pruning_threshold(0.0)
        .seed(0x720)
        .build();
    let mut dense = StateVecSoA::with_seed(2, 0x720);
    prepare_negated_pending_rotations(&mut stab, &mut dense);

    stab.flush_all_pending_rz();
    assert_phase_exact_state_matches(
        &stab.state_vector(),
        &dense.state(),
        TOLERANCE,
        "public pending-RZ flush",
    );
}

#[test]
fn cy_materializes_target_rz_even_with_an_identity_or_z_frame() {
    for z_frame in [false, true] {
        let mut stab = StabVec::builder(2).pruning_threshold(0.0).seed(7).build();
        let mut dense = StateVecSoA::with_seed(2, 7);
        stab.h(&qid(0)).h(&qid(1));
        dense.h(&qid(0)).h(&qid(1));
        let angle = Angle64::from_radians(0.37);
        stab.rz(angle, &qid(1));
        dense.rz(angle, &qid(1));
        if z_frame {
            stab.z(&qid(1));
            dense.z(&qid(1));
        }
        stab.cy(&[(QubitId(0), QubitId(1))]);
        dense.cy(&[(QubitId(0), QubitId(1))]);
        assert_phase_exact_state_matches(
            &stab.state_vector(),
            &dense.state(),
            TOLERANCE,
            &format!("CY with pending target RZ, Z frame={z_frame}"),
        );
    }
}

fn apply_z_axis_frame<S: CliffordGateable>(sim: &mut S, frame: usize, q: usize) {
    for _ in 0..frame % 4 {
        sim.sz(&qid(q));
    }
    if frame >= 4 {
        sim.x(&qid(q));
    }
}

#[test]
fn public_flush_and_measurement_preserve_all_z_axis_frames() {
    // S^k and X S^k exhaust the eight frames compatible with a pending RZ.
    for frame in 0..8 {
        for seed in 0..16 {
            let mut stab = StabVec::builder(2)
                .pruning_threshold(0.0)
                .seed(seed)
                .build();
            let mut dense = StateVecSoA::with_seed(2, seed);
            prepare_negated_pending_rotations(&mut stab, &mut dense);
            apply_z_axis_frame(&mut stab, frame, 0);
            apply_z_axis_frame(&mut dense, frame, 0);

            let label = format!("Z-axis frame {frame}, seed {seed}");
            let mut flushed = stab.clone();
            flushed.flush_all_pending_rz();
            assert_phase_exact_state_matches(
                &flushed.state_vector(),
                &dense.state(),
                TOLERANCE,
                &label,
            );

            // Compare with the dense state's normalized projection onto the
            // sampled branch. The other qubit's RZ/frame remains deferred.
            let outcome = stab.mz(&qid(0))[0].outcome;
            let mut projected = dense.state();
            for (index, amplitude) in projected.iter_mut().enumerate() {
                if (index & 1 != 0) != outcome {
                    *amplitude = Complex64::new(0.0, 0.0);
                }
            }
            let norm = projected
                .iter()
                .map(Complex64::norm_sqr)
                .sum::<f64>()
                .sqrt();
            assert!(norm > TOLERANCE, "{label}: sampled an impossible outcome");
            for amplitude in &mut projected {
                *amplitude /= norm;
            }
            assert_phase_exact_state_matches(&stab.state_vector(), &projected, TOLERANCE, &label);
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum TwoQubitGate {
    Cx,
    Cy,
    Cz,
    Swap,
    Sxx,
    Sxxdg,
    Syy,
    Syydg,
    Szz,
    Szzdg,
    Iswap,
    Iswapdg,
    G,
    Gdg,
    Rxx,
    Ryy,
    Rzz,
    Rxxryyrzz,
    U2q,
}

fn apply_two_qubit_gate<S>(simulator: &mut S, gate: TwoQubitGate)
where
    S: ArbitraryRotationGateable + CliffordGateable,
{
    let pair = [(QubitId(0), QubitId(1))];
    match gate {
        TwoQubitGate::Cx => {
            simulator.cx(&pair);
        }
        TwoQubitGate::Cy => {
            simulator.cy(&pair);
        }
        TwoQubitGate::Cz => {
            simulator.cz(&pair);
        }
        TwoQubitGate::Swap => {
            simulator.swap(&pair);
        }
        TwoQubitGate::Sxx => {
            simulator.sxx(&pair);
        }
        TwoQubitGate::Sxxdg => {
            simulator.sxxdg(&pair);
        }
        TwoQubitGate::Syy => {
            simulator.syy(&pair);
        }
        TwoQubitGate::Syydg => {
            simulator.syydg(&pair);
        }
        TwoQubitGate::Szz => {
            simulator.szz(&pair);
        }
        TwoQubitGate::Szzdg => {
            simulator.szzdg(&pair);
        }
        TwoQubitGate::Iswap => {
            simulator.iswap(&pair);
        }
        TwoQubitGate::Iswapdg => {
            simulator.iswapdg(&pair);
        }
        TwoQubitGate::G => {
            simulator.g(&pair);
        }
        TwoQubitGate::Gdg => {
            simulator.gdg(&pair);
        }
        TwoQubitGate::Rxx => {
            simulator.rxx(Angle64::from_radians(0.31), &pair);
        }
        TwoQubitGate::Ryy => {
            simulator.ryy(Angle64::from_radians(0.31), &pair);
        }
        TwoQubitGate::Rzz => {
            simulator.rzz(Angle64::from_radians(0.31), &pair);
        }
        TwoQubitGate::Rxxryyrzz => {
            simulator.rxxryyrzz(
                Angle64::from_radians(0.31),
                Angle64::from_radians(-0.47),
                Angle64::from_radians(0.23),
                &pair,
            );
        }
        TwoQubitGate::U2q => {
            let angles = [0.31, -0.47, 0.23].map(Angle64::from_radians);
            simulator.u2q([angles; 2], angles, [angles; 2], &pair);
        }
    }
}

#[test]
fn two_qubit_paths_preserve_x_and_y_negated_pending_rz_pairings() {
    for gate in [
        TwoQubitGate::Cx,
        TwoQubitGate::Cy,
        TwoQubitGate::Cz,
        TwoQubitGate::Swap,
        TwoQubitGate::Sxx,
        TwoQubitGate::Sxxdg,
        TwoQubitGate::Syy,
        TwoQubitGate::Syydg,
        TwoQubitGate::Szz,
        TwoQubitGate::Szzdg,
        TwoQubitGate::Iswap,
        TwoQubitGate::Iswapdg,
        TwoQubitGate::G,
        TwoQubitGate::Gdg,
        TwoQubitGate::Rxx,
        TwoQubitGate::Ryy,
        TwoQubitGate::Rzz,
        TwoQubitGate::Rxxryyrzz,
        TwoQubitGate::U2q,
    ] {
        let mut stab = StabVec::builder(2)
            .pruning_threshold(0.0)
            .seed(0x720)
            .build();
        let mut dense = StateVecSoA::with_seed(2, 0x720);
        prepare_negated_pending_rotations(&mut stab, &mut dense);

        apply_two_qubit_gate(&mut stab, gate);
        apply_two_qubit_gate(&mut dense, gate);
        assert_phase_exact_state_matches(
            &stab.state_vector(),
            &dense.state(),
            TOLERANCE,
            &format!("{gate:?} with X/Y-negated pending rotations"),
        );
    }
}

#[test]
fn yy_roots_propagate_unflushed_partner_frames_once() {
    for gate in [TwoQubitGate::Syy, TwoQubitGate::Syydg] {
        for rotated in 0..2 {
            for pauli in [
                CliffordTGate::X(1 - rotated),
                CliffordTGate::Y(1 - rotated),
                CliffordTGate::Z(1 - rotated),
            ] {
                let mut stab = StabVec::builder(2).pruning_threshold(0.0).seed(7).build();
                let mut dense = StateVecSoA::with_seed(2, 7);
                stab.h(&qid(0)).h(&qid(1));
                dense.h(&qid(0)).h(&qid(1));
                // Materialize preparation, then leave a pending RZ on just one
                // qubit and an independent Pauli frame on the other.
                assert_phase_exact_state_matches(
                    &stab.state_vector(),
                    &dense.state(),
                    TOLERANCE,
                    "YY root input",
                );
                let angle = Angle64::from_radians(0.37);
                stab.rz(angle, &qid(rotated)).x(&qid(rotated));
                dense.rz(angle, &qid(rotated)).x(&qid(rotated));
                apply_gate(&mut stab, pauli);
                apply_gate(&mut dense, pauli);
                apply_two_qubit_gate(&mut stab, gate);
                apply_two_qubit_gate(&mut dense, gate);
                assert_phase_exact_state_matches(
                    &stab.state_vector(),
                    &dense.state(),
                    TOLERANCE,
                    &format!("{gate:?}, RZ on {rotated}, partner {pauli:?}"),
                );
            }
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum CliffordTGate {
    H(usize),
    S(usize),
    Sdg(usize),
    Z(usize),
    X(usize),
    Y(usize),
    Sx(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    T(usize),
    Tdg(usize),
}

fn random_gate(rng: &mut StdRng, num_qubits: usize) -> CliffordTGate {
    let q = rng.random_range(0..num_qubits);
    match rng.random_range(0..11) {
        0 => CliffordTGate::H(q),
        1 => CliffordTGate::S(q),
        2 => CliffordTGate::Sdg(q),
        3 => CliffordTGate::Z(q),
        4 => CliffordTGate::X(q),
        5 => CliffordTGate::Y(q),
        6 => CliffordTGate::Sx(q),
        7 => {
            let mut target = rng.random_range(0..num_qubits - 1);
            if target >= q {
                target += 1;
            }
            CliffordTGate::Cx(q, target)
        }
        8 => {
            let mut other = rng.random_range(0..num_qubits - 1);
            if other >= q {
                other += 1;
            }
            CliffordTGate::Cz(q, other)
        }
        9 => CliffordTGate::T(q),
        _ => CliffordTGate::Tdg(q),
    }
}

fn apply_gate<S>(simulator: &mut S, gate: CliffordTGate)
where
    S: ArbitraryRotationGateable + CliffordGateable,
{
    match gate {
        CliffordTGate::H(q) => {
            simulator.h(&qid(q));
        }
        CliffordTGate::S(q) => {
            simulator.sz(&qid(q));
        }
        CliffordTGate::Sdg(q) => {
            simulator.szdg(&qid(q));
        }
        CliffordTGate::Z(q) => {
            simulator.z(&qid(q));
        }
        CliffordTGate::X(q) => {
            simulator.x(&qid(q));
        }
        CliffordTGate::Y(q) => {
            simulator.y(&qid(q));
        }
        CliffordTGate::Sx(q) => {
            simulator.sx(&qid(q));
        }
        CliffordTGate::Cx(control, target) => {
            simulator.cx(&[(QubitId(control), QubitId(target))]);
        }
        CliffordTGate::Cz(q0, q1) => {
            simulator.cz(&[(QubitId(q0), QubitId(q1))]);
        }
        CliffordTGate::T(q) => {
            simulator.t(&qid(q));
        }
        CliffordTGate::Tdg(q) => {
            simulator.tdg(&qid(q));
        }
    }
}

fn max_amplitude_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    assert_eq!(actual.len(), expected.len());
    assert!(!expected.is_empty());
    assert!(
        actual
            .iter()
            .chain(expected)
            .all(|z| z.re.is_finite() && z.im.is_finite())
    );
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (actual - expected).norm())
        .fold(0.0, f64::max)
}

fn max_phase_aligned_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    let (reference_index, _) = expected
        .iter()
        .enumerate()
        .max_by(|(_, lhs), (_, rhs)| lhs.norm_sqr().total_cmp(&rhs.norm_sqr()))
        .expect("a nonempty reference state");
    if actual[reference_index].norm() <= TOLERANCE {
        return f64::INFINITY;
    }
    let ratio = expected[reference_index] / actual[reference_index];
    // Remove only a unit scalar: a norm error must remain a failing error.
    let phase = ratio / ratio.norm();
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (phase * actual - expected).norm())
        .fold(0.0, f64::max)
}

#[test]
fn phase_alignment_removes_only_global_phase() {
    let a = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
    let expected = [a, a];
    let global = [a * Complex64::i(), a * Complex64::i()];
    assert!(max_amplitude_error(&global, &expected) > TOLERANCE);
    assert!(max_phase_aligned_error(&global, &expected) <= TOLERANCE);
    for wrong in [[a, -a], [2.0 * a, 2.0 * a], [a, Complex64::new(0.0, 0.0)]] {
        assert!(max_phase_aligned_error(&wrong, &expected) > TOLERANCE);
    }
}

#[test]
fn seeded_random_clifford_t_circuits_match_after_global_phase_alignment() {
    const NUM_QUBITS: usize = 5;
    const DEPTH: usize = 30;
    const CIRCUITS: usize = 200;

    let mut rng = StdRng::seed_from_u64(0x720_C11F_F04D);
    let mut raw_mismatches = 0;
    let mut global_phase_only_mismatches = 0;
    let mut aligned_mismatches = 0;
    let mut worst_aligned_error = 0.0_f64;
    let mut worst_circuit = 0;
    let mut worst_gates = Vec::new();

    for circuit_index in 0..CIRCUITS {
        let gates: Vec<_> = (0..DEPTH)
            .map(|_| random_gate(&mut rng, NUM_QUBITS))
            .collect();
        let mut stab = StabVec::builder(NUM_QUBITS)
            .pruning_threshold(0.0)
            .seed(circuit_index as u64)
            .build();
        let mut dense = StateVecSoA::with_seed(NUM_QUBITS, circuit_index as u64);
        for &gate in &gates {
            apply_gate(&mut stab, gate);
            apply_gate(&mut dense, gate);
        }

        let actual = stab.state_vector();
        let expected = dense.state();
        let raw_error = max_amplitude_error(&actual, &expected);
        let aligned_error = max_phase_aligned_error(&actual, &expected);
        if raw_error > TOLERANCE {
            raw_mismatches += 1;
            if aligned_error <= TOLERANCE {
                global_phase_only_mismatches += 1;
            }
        }
        if aligned_error > TOLERANCE {
            aligned_mismatches += 1;
        }
        if aligned_error > worst_aligned_error {
            worst_aligned_error = aligned_error;
            worst_circuit = circuit_index;
            worst_gates = gates;
        }
    }

    eprintln!(
        "random Clifford+T differential: raw={raw_mismatches}/{CIRCUITS}, \
         global-phase-only={global_phase_only_mismatches}/{CIRCUITS}, \
         aligned={aligned_mismatches}/{CIRCUITS}, \
         worst-aligned={worst_aligned_error:e} at circuit {worst_circuit}"
    );
    assert_eq!(
        aligned_mismatches, 0,
        "{aligned_mismatches}/{CIRCUITS} random circuits still differ after removing global phase; \
         worst error {worst_aligned_error:e} at circuit {worst_circuit} \
         ({raw_mismatches} raw mismatches, {global_phase_only_mismatches} global-phase-only); \
         seed=0x720_C11F_F04D, gates={worst_gates:?}"
    );
}
