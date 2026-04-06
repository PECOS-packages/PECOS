// Copyright 2025 The PECOS Developers
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

//! Shared measurement-based test utilities for any `ArbitraryRotationGateable` simulator.
//!
//! These tests verify rotation gate contracts using only measurement outcomes,
//! making them usable by any simulator that implements [`ArbitraryRotationGateable`].
//!
//! For simulator-specific tests (exact amplitudes, etc.), see [`state_vector_test_utils`].

#![allow(clippy::missing_panics_doc)]

use crate::ArbitraryRotationGateable;
use pecos_core::{Angle64, QubitId, qid};
use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI};

// --- Helper: deterministic measurement assertion ---

fn assert_mz<S: ArbitraryRotationGateable>(sim: &mut S, q: usize, expected: bool, msg: &str) {
    let result = sim.mz(&qid(q));
    assert!(result[0].is_deterministic, "{msg}: should be deterministic");
    assert_eq!(result[0].outcome, expected, "{msg}: wrong outcome");
}

fn assert_mx<S: ArbitraryRotationGateable>(sim: &mut S, q: usize, expected: bool, msg: &str) {
    let result = sim.mx(&qid(q));
    assert!(result[0].is_deterministic, "{msg}: should be deterministic");
    assert_eq!(result[0].outcome, expected, "{msg}: wrong outcome");
}

fn assert_my<S: ArbitraryRotationGateable>(sim: &mut S, q: usize, expected: bool, msg: &str) {
    let result = sim.my(&qid(q));
    assert!(result[0].is_deterministic, "{msg}: should be deterministic");
    assert_eq!(result[0].outcome, expected, "{msg}: wrong outcome");
}

fn assert_mz_superposition<S: ArbitraryRotationGateable>(sim: &mut S, q: usize, msg: &str) {
    let result = sim.mz(&qid(q));
    assert!(
        !result[0].is_deterministic,
        "{msg}: should be non-deterministic"
    );
}

// --- Clifford-Angle Equivalences ---

/// Verify RX(pi) = X (up to global phase, invisible to measurement).
///
/// RX(pi)|0> = -i|1>, X|0> = |1>. Both measure 1 in Z basis.
pub fn verify_rx_pi_equals_x<S: ArbitraryRotationGateable>(sim: &mut S) {
    // On |0>: both give |1>
    sim.reset();
    sim.rx(Angle64::from_radians(PI), &qid(0));
    assert_mz(sim, 0, true, "RX(pi)|0>");

    // On |1>: both give |0>
    sim.reset();
    sim.x(&qid(0));
    sim.rx(Angle64::from_radians(PI), &qid(0));
    assert_mz(sim, 0, false, "RX(pi)|1>");

    // On |+>: RX(pi)|+> = -i|+>, X|+> = |+>. Both measure 0 in X basis.
    sim.reset();
    sim.h(&qid(0));
    sim.rx(Angle64::from_radians(PI), &qid(0));
    assert_mx(sim, 0, false, "RX(pi)|+>");
}

/// Verify RY(pi) = Y (up to global phase, invisible to measurement).
pub fn verify_ry_pi_equals_y<S: ArbitraryRotationGateable>(sim: &mut S) {
    // On |0>: Y|0> = i|1>, RY(pi)|0> = |1>. Both measure 1.
    sim.reset();
    sim.ry(Angle64::from_radians(PI), &qid(0));
    assert_mz(sim, 0, true, "RY(pi)|0>");

    // On |1>: Y|1> = -i|0>, RY(pi)|1> = -|0>. Both measure 0.
    sim.reset();
    sim.x(&qid(0));
    sim.ry(Angle64::from_radians(PI), &qid(0));
    assert_mz(sim, 0, false, "RY(pi)|1>");
}

/// Verify RZ(pi) = Z (up to global phase, invisible to measurement).
pub fn verify_rz_pi_equals_z<S: ArbitraryRotationGateable>(sim: &mut S) {
    // On |0>: both give |0> (phase invisible)
    sim.reset();
    sim.rz(Angle64::from_radians(PI), &qid(0));
    assert_mz(sim, 0, false, "RZ(pi)|0>");

    // On |+>: Z|+> = |->, RZ(pi)|+> = -i|->. Both measure 1 in X basis.
    sim.reset();
    sim.h(&qid(0));
    sim.rz(Angle64::from_radians(PI), &qid(0));
    assert_mx(sim, 0, true, "RZ(pi)|+>");
}

// --- Identity at Zero Angle ---

/// Verify R(0) = I for all single-qubit rotations.
pub fn verify_rotation_identity_at_zero<S: ArbitraryRotationGateable>(sim: &mut S) {
    // RX(0)|0> = |0>
    sim.reset();
    sim.rx(Angle64::from_radians(0.0), &qid(0));
    assert_mz(sim, 0, false, "RX(0)|0>");

    // RX(0)|+> = |+>
    sim.reset();
    sim.h(&qid(0));
    sim.rx(Angle64::from_radians(0.0), &qid(0));
    assert_mx(sim, 0, false, "RX(0)|+>");

    // RY(0)|0> = |0>
    sim.reset();
    sim.ry(Angle64::from_radians(0.0), &qid(0));
    assert_mz(sim, 0, false, "RY(0)|0>");

    // RY(0)|+> = |+>
    sim.reset();
    sim.h(&qid(0));
    sim.ry(Angle64::from_radians(0.0), &qid(0));
    assert_mx(sim, 0, false, "RY(0)|+>");

    // RZ(0)|0> = |0>
    sim.reset();
    sim.rz(Angle64::from_radians(0.0), &qid(0));
    assert_mz(sim, 0, false, "RZ(0)|0>");

    // RZ(0)|+> = |+>
    sim.reset();
    sim.h(&qid(0));
    sim.rz(Angle64::from_radians(0.0), &qid(0));
    assert_mx(sim, 0, false, "RZ(0)|+>");
}

// --- Inverse Rotations ---

/// Verify R(theta) * R(-theta) = I for various rotations.
pub fn verify_rotation_inverse<S: ArbitraryRotationGateable>(sim: &mut S) {
    let angles = [0.3, 1.0, PI / 3.0, PI / 7.0, 2.5];

    for &theta in &angles {
        let ang = Angle64::from_radians(theta);
        let neg_ang = Angle64::from_radians(-theta);

        // RX(theta)*RX(-theta)|+> = |+>
        sim.reset();
        sim.h(&qid(0));
        sim.rx(ang, &qid(0)).rx(neg_ang, &qid(0));
        assert_mx(sim, 0, false, &format!("RX({theta})*RX(-{theta})|+>"));

        // RY(theta)*RY(-theta)|+> = |+>
        sim.reset();
        sim.h(&qid(0));
        sim.ry(ang, &qid(0)).ry(neg_ang, &qid(0));
        assert_mx(sim, 0, false, &format!("RY({theta})*RY(-{theta})|+>"));

        // RZ(theta)*RZ(-theta)|+> = |+>
        sim.reset();
        sim.h(&qid(0));
        sim.rz(ang, &qid(0)).rz(neg_ang, &qid(0));
        assert_mx(sim, 0, false, &format!("RZ({theta})*RZ(-{theta})|+>"));
    }
}

/// Verify two-qubit rotation inverses: RZZ(theta)*RZZ(-theta) = I.
pub fn verify_two_qubit_rotation_inverse<S: ArbitraryRotationGateable>(sim: &mut S) {
    let angles = [0.3, 1.0, PI / 3.0];

    for &theta in &angles {
        let ang = Angle64::from_radians(theta);
        let neg_ang = Angle64::from_radians(-theta);

        // On Bell state
        sim.reset();
        sim.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
        sim.rzz(ang, &[(QubitId(0), QubitId(1))])
            .rzz(neg_ang, &[(QubitId(0), QubitId(1))]);

        // Should still be a Bell state
        let r0 = sim.mz(&qid(0));
        let r1 = sim.mz(&qid(1));
        assert!(
            r1[0].is_deterministic,
            "RZZ({theta})*RZZ(-{theta}) Bell: q1 deterministic"
        );
        assert_eq!(
            r0[0].outcome, r1[0].outcome,
            "RZZ({theta})*RZZ(-{theta}) Bell: correlated"
        );

        // RXX(theta)*RXX(-theta) = I
        sim.reset();
        sim.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
        sim.rxx(ang, &[(QubitId(0), QubitId(1))])
            .rxx(neg_ang, &[(QubitId(0), QubitId(1))]);
        let r0 = sim.mz(&qid(0));
        let r1 = sim.mz(&qid(1));
        assert!(
            r1[0].is_deterministic,
            "RXX({theta})*RXX(-{theta}) Bell: q1 deterministic"
        );
        assert_eq!(
            r0[0].outcome, r1[0].outcome,
            "RXX({theta})*RXX(-{theta}) Bell: correlated"
        );

        // RYY(theta)*RYY(-theta) = I
        sim.reset();
        sim.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
        sim.ryy(ang, &[(QubitId(0), QubitId(1))])
            .ryy(neg_ang, &[(QubitId(0), QubitId(1))]);
        let r0 = sim.mz(&qid(0));
        let r1 = sim.mz(&qid(1));
        assert!(
            r1[0].is_deterministic,
            "RYY({theta})*RYY(-{theta}) Bell: q1 deterministic"
        );
        assert_eq!(
            r0[0].outcome, r1[0].outcome,
            "RYY({theta})*RYY(-{theta}) Bell: correlated"
        );
    }
}

// --- T Gate Tests ---

/// Verify T^8 = I (up to global phase, invisible to measurement).
///
/// T = RZ(pi/4), so T^8 = RZ(2*pi) = e^{-i*pi}*I = -I.
/// Global phase is invisible to measurement.
pub fn verify_t_eighth_power<S: ArbitraryRotationGateable>(sim: &mut S) {
    // On |0>: T^8|0> = -|0>, measures 0
    sim.reset();
    for _ in 0..8 {
        sim.t(&qid(0));
    }
    assert_mz(sim, 0, false, "T^8|0>");

    // On |+>: T^8|+> = -|+>, measures 0 in X
    sim.reset();
    sim.h(&qid(0));
    for _ in 0..8 {
        sim.t(&qid(0));
    }
    assert_mx(sim, 0, false, "T^8|+>");
}

/// Verify T * Tdg = I.
pub fn verify_t_adjoint<S: ArbitraryRotationGateable>(sim: &mut S) {
    // On |+>
    sim.reset();
    sim.h(&qid(0));
    sim.t(&qid(0)).tdg(&qid(0));
    assert_mx(sim, 0, false, "T*Tdg|+>");

    // On |0>
    sim.reset();
    sim.t(&qid(0)).tdg(&qid(0));
    assert_mz(sim, 0, false, "T*Tdg|0>");
}

// --- Rotation Composition ---

/// Verify RZ(a) * RZ(b) = RZ(a+b) at Clifford angles.
///
/// Tests that sequential rotations compose correctly.
pub fn verify_rz_composition<S: ArbitraryRotationGateable>(sim: &mut S) {
    // RZ(pi/2)*RZ(pi/2) should equal RZ(pi) = Z (up to global phase)
    // On |+>: Z|+> = |->, so both should measure 1 in X basis
    sim.reset();
    sim.h(&qid(0));
    sim.rz(Angle64::from_radians(FRAC_PI_2), &qid(0))
        .rz(Angle64::from_radians(FRAC_PI_2), &qid(0));
    assert_mx(sim, 0, true, "RZ(pi/2)+RZ(pi/2)|+> = RZ(pi)|+> = |->");

    sim.reset();
    sim.h(&qid(0));
    sim.rz(Angle64::from_radians(PI), &qid(0));
    assert_mx(sim, 0, true, "RZ(pi)|+> = |->");

    // RZ(pi/4)*RZ(pi/4) = RZ(pi/2) = S (up to global phase)
    // S|+> = |+y>: measure 0 in Y basis
    sim.reset();
    sim.h(&qid(0));
    sim.rz(Angle64::from_radians(FRAC_PI_4), &qid(0))
        .rz(Angle64::from_radians(FRAC_PI_4), &qid(0));
    let result = sim.my(&qid(0));
    assert!(
        result[0].is_deterministic,
        "RZ(pi/4)+RZ(pi/4)|+>: Y-measurement should be deterministic"
    );

    sim.reset();
    sim.h(&qid(0));
    sim.rz(Angle64::from_radians(FRAC_PI_2), &qid(0));
    let result2 = sim.my(&qid(0));
    assert!(
        result2[0].is_deterministic,
        "RZ(pi/2)|+>: Y-measurement should be deterministic"
    );
    assert_eq!(
        result[0].outcome, result2[0].outcome,
        "RZ(pi/4)*2 should equal RZ(pi/2)"
    );
}

// --- Two-Qubit Rotation Tests ---

/// Verify RZZ at special angles.
///
/// RZZ(0) = I (identity).
/// RZZ on computational basis states only adds phase (invisible to Z measurement).
pub fn verify_rzz_special_angles<S: ArbitraryRotationGateable>(sim: &mut S) {
    // RZZ(0) = I
    sim.reset();
    sim.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
    sim.rzz(Angle64::from_radians(0.0), &[(QubitId(0), QubitId(1))]);
    let r0 = sim.mz(&qid(0));
    let r1 = sim.mz(&qid(1));
    assert!(r1[0].is_deterministic, "RZZ(0) Bell: q1 deterministic");
    assert_eq!(
        r0[0].outcome, r1[0].outcome,
        "RZZ(0) Bell: still correlated"
    );

    // RZZ on |00>, |01>, |10>, |11> - phase only, outcomes unchanged
    for state in 0..4u8 {
        let q0_val = (state & 1) != 0;
        let q1_val = (state >> 1) != 0;

        sim.reset();
        if q0_val {
            sim.x(&qid(0));
        }
        if q1_val {
            sim.x(&qid(1));
        }
        sim.rzz(Angle64::from_radians(1.0), &[(QubitId(0), QubitId(1))]);
        assert_mz(sim, 0, q0_val, &format!("RZZ(1.0)|{state:02b}>: q0"));
        assert_mz(sim, 1, q1_val, &format!("RZZ(1.0)|{state:02b}>: q1"));
    }
}

/// Verify RXX(0) = I and RYY(0) = I.
pub fn verify_rxx_ryy_identity<S: ArbitraryRotationGateable>(sim: &mut S) {
    // RXX(0)|Bell> = |Bell>
    sim.reset();
    sim.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
    sim.rxx(Angle64::from_radians(0.0), &[(QubitId(0), QubitId(1))]);
    let r0 = sim.mz(&qid(0));
    let r1 = sim.mz(&qid(1));
    assert!(r1[0].is_deterministic, "RXX(0) Bell: q1 deterministic");
    assert_eq!(
        r0[0].outcome, r1[0].outcome,
        "RXX(0) Bell: still correlated"
    );

    // RYY(0)|Bell> = |Bell>
    sim.reset();
    sim.h(&qid(0)).cx(&[(QubitId(0), QubitId(1))]);
    sim.ryy(Angle64::from_radians(0.0), &[(QubitId(0), QubitId(1))]);
    let r0 = sim.mz(&qid(0));
    let r1 = sim.mz(&qid(1));
    assert!(r1[0].is_deterministic, "RYY(0) Bell: q1 deterministic");
    assert_eq!(
        r0[0].outcome, r1[0].outcome,
        "RYY(0) Bell: still correlated"
    );
}

/// Verify RX(pi/2) creates superposition from |0>.
pub fn verify_rx_half_pi_superposition<S: ArbitraryRotationGateable>(sim: &mut S) {
    sim.reset();
    sim.rx(Angle64::from_radians(FRAC_PI_2), &qid(0));
    assert_mz_superposition(sim, 0, "RX(pi/2)|0>");
}

/// Verify RY(pi/2) creates superposition from |0>.
pub fn verify_ry_half_pi_superposition<S: ArbitraryRotationGateable>(sim: &mut S) {
    sim.reset();
    sim.ry(Angle64::from_radians(FRAC_PI_2), &qid(0));
    assert_mz_superposition(sim, 0, "RY(pi/2)|0>");
}

// --- Circuit Inverse Test ---

/// Verify that a mixed rotation circuit and its reverse compose to identity.
pub fn verify_rotation_circuit_inverse<S: ArbitraryRotationGateable>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0));

    // Forward circuit
    sim.rx(Angle64::from_radians(0.7), &qid(0));
    sim.ry(Angle64::from_radians(1.3), &qid(0));
    sim.rz(Angle64::from_radians(0.5), &qid(0));
    sim.rx(Angle64::from_radians(2.1), &qid(0));

    // Reverse circuit (each gate's inverse, in reverse order)
    sim.rx(Angle64::from_radians(-2.1), &qid(0));
    sim.rz(Angle64::from_radians(-0.5), &qid(0));
    sim.ry(Angle64::from_radians(-1.3), &qid(0));
    sim.rx(Angle64::from_radians(-0.7), &qid(0));

    // Should be back to |+>
    assert_mx(sim, 0, false, "Rotation circuit inverse: back to |+>");
}

// --- U Gate Tests ---

/// Verify U(0, 0, 0) = I.
pub fn verify_u_identity<S: ArbitraryRotationGateable>(sim: &mut S) {
    // On |0>
    sim.reset();
    sim.u(
        Angle64::from_radians(0.0),
        Angle64::from_radians(0.0),
        Angle64::from_radians(0.0),
        &qid(0),
    );
    assert_mz(sim, 0, false, "U(0,0,0)|0>");

    // On |1>
    sim.reset();
    sim.x(&qid(0));
    sim.u(
        Angle64::from_radians(0.0),
        Angle64::from_radians(0.0),
        Angle64::from_radians(0.0),
        &qid(0),
    );
    assert_mz(sim, 0, true, "U(0,0,0)|1>");

    // On |+>
    sim.reset();
    sim.h(&qid(0));
    sim.u(
        Angle64::from_radians(0.0),
        Angle64::from_radians(0.0),
        Angle64::from_radians(0.0),
        &qid(0),
    );
    assert_mx(sim, 0, false, "U(0,0,0)|+>");
}

/// Verify U(pi, 0, pi) = X (up to global phase).
pub fn verify_u_as_x<S: ArbitraryRotationGateable>(sim: &mut S) {
    // U(pi, 0, pi)|0> should give |1>
    sim.reset();
    sim.u(
        Angle64::from_radians(PI),
        Angle64::from_radians(0.0),
        Angle64::from_radians(PI),
        &qid(0),
    );
    assert_mz(sim, 0, true, "U(pi,0,pi)|0> = X|0> = |1>");

    // U(pi, 0, pi)|1> should give |0>
    sim.reset();
    sim.x(&qid(0));
    sim.u(
        Angle64::from_radians(PI),
        Angle64::from_radians(0.0),
        Angle64::from_radians(PI),
        &qid(0),
    );
    assert_mz(sim, 0, false, "U(pi,0,pi)|1> = X|1> = |0>");
}

/// Verify U(pi/2, 0, pi) = H (up to global phase).
pub fn verify_u_as_h<S: ArbitraryRotationGateable>(sim: &mut S) {
    // U(pi/2, 0, pi)|0> should create |+> (superposition)
    sim.reset();
    sim.u(
        Angle64::from_radians(FRAC_PI_2),
        Angle64::from_radians(0.0),
        Angle64::from_radians(PI),
        &qid(0),
    );
    assert_mx(sim, 0, false, "U(pi/2,0,pi)|0> = H|0> = |+>");

    // U(pi/2, 0, pi) applied twice should give identity
    sim.reset();
    sim.u(
        Angle64::from_radians(FRAC_PI_2),
        Angle64::from_radians(0.0),
        Angle64::from_radians(PI),
        &qid(0),
    );
    sim.u(
        Angle64::from_radians(FRAC_PI_2),
        Angle64::from_radians(0.0),
        Angle64::from_radians(PI),
        &qid(0),
    );
    assert_mz(sim, 0, false, "U(pi/2,0,pi)^2|0> = H^2|0> = |0>");
}

/// Verify U(theta, phi, lambda) * U(-theta, -lambda, -phi) = I.
pub fn verify_u_inverse<S: ArbitraryRotationGateable>(sim: &mut S) {
    let cases: &[(f64, f64, f64)] = &[
        (0.7, 1.2, 0.3),
        (PI, FRAC_PI_2, FRAC_PI_4),
        (1.5, -0.8, 2.1),
    ];

    for &(theta, phi, lambda) in cases {
        // On |0>
        sim.reset();
        sim.u(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            Angle64::from_radians(lambda),
            &qid(0),
        );
        sim.u(
            Angle64::from_radians(-theta),
            Angle64::from_radians(-lambda),
            Angle64::from_radians(-phi),
            &qid(0),
        );
        assert_mz(sim, 0, false, &format!("U*U_inv|0> theta={theta}"));

        // On |+>
        sim.reset();
        sim.h(&qid(0));
        sim.u(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            Angle64::from_radians(lambda),
            &qid(0),
        );
        sim.u(
            Angle64::from_radians(-theta),
            Angle64::from_radians(-lambda),
            Angle64::from_radians(-phi),
            &qid(0),
        );
        assert_mx(sim, 0, false, &format!("U*U_inv|+> theta={theta}"));
    }
}

/// Regression test for gate fusion ordering: a queued Clifford (H) followed by U
/// must apply H before U, not after. This catches bugs where U bypasses
/// the fusion queue and operates on stale state.
pub fn verify_u_after_clifford_ordering<S: ArbitraryRotationGateable>(sim: &mut S) {
    // H|0> = |+>, then U(pi,0,pi)=X on |+> gives X|+> = |+>
    // X-basis measurement of |+> is deterministic 0.
    sim.reset();
    sim.h(&qid(0));
    sim.u(
        Angle64::from_radians(PI),
        Angle64::from_radians(0.0),
        Angle64::from_radians(PI),
        &qid(0),
    );
    assert_mx(
        sim,
        0,
        false,
        "H then U(pi,0,pi): X|+> = |+>, mx should be 0",
    );

    // H|0> = |+>, then U(pi/2,0,pi)=H on |+> gives H|+> = |0>
    // Z-basis measurement of |0> is deterministic 0.
    sim.reset();
    sim.h(&qid(0));
    sim.u(
        Angle64::from_radians(FRAC_PI_2),
        Angle64::from_radians(0.0),
        Angle64::from_radians(PI),
        &qid(0),
    );
    assert_mz(
        sim,
        0,
        false,
        "H then U(pi/2,0,pi): H|+> = |0>, mz should be 0",
    );

    // X|0> = |1>, then U(pi,0,pi)=X on |1> gives X|1> = |0>
    sim.reset();
    sim.x(&qid(0));
    sim.u(
        Angle64::from_radians(PI),
        Angle64::from_radians(0.0),
        Angle64::from_radians(PI),
        &qid(0),
    );
    assert_mz(
        sim,
        0,
        false,
        "X then U(pi,0,pi): X*X|0> = |0>, mz should be 0",
    );
}

// --- R1XY Gate Tests ---

/// Verify R1XY(0, phi) = I for any phi.
pub fn verify_r1xy_identity<S: ArbitraryRotationGateable>(sim: &mut S) {
    let phi_values: &[f64] = &[0.0, FRAC_PI_4, FRAC_PI_2, PI, -1.3];

    for &phi in phi_values {
        sim.reset();
        sim.r1xy(
            Angle64::from_radians(0.0),
            Angle64::from_radians(phi),
            &qid(0),
        );
        assert_mz(sim, 0, false, &format!("R1XY(0, {phi})|0>"));

        sim.reset();
        sim.h(&qid(0));
        sim.r1xy(
            Angle64::from_radians(0.0),
            Angle64::from_radians(phi),
            &qid(0),
        );
        assert_mx(sim, 0, false, &format!("R1XY(0, {phi})|+>"));
    }
}

/// Verify R1XY(theta, pi/2) = RX(theta) (up to global phase).
pub fn verify_r1xy_as_rx<S: ArbitraryRotationGateable>(sim: &mut S) {
    // R1XY(pi, pi/2) should act like RX(pi) = X (up to phase)
    sim.reset();
    sim.r1xy(
        Angle64::from_radians(PI),
        Angle64::from_radians(FRAC_PI_2),
        &qid(0),
    );
    assert_mz(sim, 0, true, "R1XY(pi, pi/2)|0> = RX(pi)|0> = |1>");

    sim.reset();
    sim.x(&qid(0));
    sim.r1xy(
        Angle64::from_radians(PI),
        Angle64::from_radians(FRAC_PI_2),
        &qid(0),
    );
    assert_mz(sim, 0, false, "R1XY(pi, pi/2)|1> = RX(pi)|1> = |0>");
}

/// Verify R1XY(theta, 0) = RY(theta) (up to global phase).
pub fn verify_r1xy_as_ry<S: ArbitraryRotationGateable>(sim: &mut S) {
    // R1XY(pi, 0) should act like RY(pi) = Y (up to phase)
    sim.reset();
    sim.r1xy(
        Angle64::from_radians(PI),
        Angle64::from_radians(0.0),
        &qid(0),
    );
    assert_mz(sim, 0, true, "R1XY(pi, 0)|0> = RY(pi)|0> = |1>");

    sim.reset();
    sim.x(&qid(0));
    sim.r1xy(
        Angle64::from_radians(PI),
        Angle64::from_radians(0.0),
        &qid(0),
    );
    assert_mz(sim, 0, false, "R1XY(pi, 0)|1> = RY(pi)|1> = |0>");
}

/// Verify R1XY(theta, phi) * R1XY(-theta, phi) = I.
pub fn verify_r1xy_inverse<S: ArbitraryRotationGateable>(sim: &mut S) {
    let cases: &[(f64, f64)] = &[(0.7, 0.3), (PI, FRAC_PI_2), (1.5, -0.8)];

    for &(theta, phi) in cases {
        sim.reset();
        sim.r1xy(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            &qid(0),
        );
        sim.r1xy(
            Angle64::from_radians(-theta),
            Angle64::from_radians(phi),
            &qid(0),
        );
        assert_mz(sim, 0, false, &format!("R1XY*R1XY_inv|0> theta={theta}"));

        sim.reset();
        sim.h(&qid(0));
        sim.r1xy(
            Angle64::from_radians(theta),
            Angle64::from_radians(phi),
            &qid(0),
        );
        sim.r1xy(
            Angle64::from_radians(-theta),
            Angle64::from_radians(phi),
            &qid(0),
        );
        assert_mx(sim, 0, false, &format!("R1XY*R1XY_inv|+> theta={theta}"));
    }
}

// --- RXXRYYRZZ Composite Gate Tests ---

/// Verify RXXRYYRZZ(0, 0, 0) = I.
pub fn verify_rxxryyrzz_identity<S: ArbitraryRotationGateable>(sim: &mut S) {
    // On |00>
    sim.reset();
    sim.rxxryyrzz(
        Angle64::from_radians(0.0),
        Angle64::from_radians(0.0),
        Angle64::from_radians(0.0),
        &[(QubitId(0), QubitId(1))],
    );
    assert_mz(sim, 0, false, "RXXRYYRZZ(0,0,0)|00> q0");
    assert_mz(sim, 1, false, "RXXRYYRZZ(0,0,0)|00> q1");

    // On |10>
    sim.reset();
    sim.x(&qid(0));
    sim.rxxryyrzz(
        Angle64::from_radians(0.0),
        Angle64::from_radians(0.0),
        Angle64::from_radians(0.0),
        &[(QubitId(0), QubitId(1))],
    );
    assert_mz(sim, 0, true, "RXXRYYRZZ(0,0,0)|10> q0");
    assert_mz(sim, 1, false, "RXXRYYRZZ(0,0,0)|10> q1");

    // On |+0>
    sim.reset();
    sim.h(&qid(0));
    sim.rxxryyrzz(
        Angle64::from_radians(0.0),
        Angle64::from_radians(0.0),
        Angle64::from_radians(0.0),
        &[(QubitId(0), QubitId(1))],
    );
    assert_mx(sim, 0, false, "RXXRYYRZZ(0,0,0)|+0> q0");
    assert_mz(sim, 1, false, "RXXRYYRZZ(0,0,0)|+0> q1");
}

/// Verify RXXRYYRZZ(a,b,c) * RXXRYYRZZ(-a,-b,-c) = I.
pub fn verify_rxxryyrzz_inverse<S: ArbitraryRotationGateable>(sim: &mut S) {
    let cases: &[(f64, f64, f64)] = &[(0.5, 0.3, 0.7), (FRAC_PI_4, FRAC_PI_2, PI)];

    for &(a, b, c) in cases {
        // On |01>
        sim.reset();
        sim.x(&qid(1));
        sim.rxxryyrzz(
            Angle64::from_radians(a),
            Angle64::from_radians(b),
            Angle64::from_radians(c),
            &[(QubitId(0), QubitId(1))],
        );
        sim.rxxryyrzz(
            Angle64::from_radians(-a),
            Angle64::from_radians(-b),
            Angle64::from_radians(-c),
            &[(QubitId(0), QubitId(1))],
        );
        assert_mz(sim, 0, false, &format!("RXXRYYRZZ inv|01> q0 a={a}"));
        assert_mz(sim, 1, true, &format!("RXXRYYRZZ inv|01> q1 a={a}"));

        // On |+0>
        sim.reset();
        sim.h(&qid(0));
        sim.rxxryyrzz(
            Angle64::from_radians(a),
            Angle64::from_radians(b),
            Angle64::from_radians(c),
            &[(QubitId(0), QubitId(1))],
        );
        sim.rxxryyrzz(
            Angle64::from_radians(-a),
            Angle64::from_radians(-b),
            Angle64::from_radians(-c),
            &[(QubitId(0), QubitId(1))],
        );
        assert_mx(sim, 0, false, &format!("RXXRYYRZZ inv|+0> q0 a={a}"));
        assert_mz(sim, 1, false, &format!("RXXRYYRZZ inv|+0> q1 a={a}"));
    }
}

/// Verify RXXRYYRZZ equals the explicit RXX * RYY * RZZ sequence by checking
/// that applying the composite followed by the inverse of each component gives identity.
pub fn verify_rxxryyrzz_decomposition<S: ArbitraryRotationGateable>(sim: &mut S) {
    let (a, b, c) = (0.5, 0.3, 0.7);

    // Apply RXXRYYRZZ(a,b,c) then undo with RZZ(-c) * RYY(-b) * RXX(-a)
    // If RXXRYYRZZ = RXX(a) * RYY(b) * RZZ(c), then the inverse is
    // RZZ(-c) * RYY(-b) * RXX(-a).
    sim.reset();
    sim.x(&qid(1)); // |01>
    sim.rxxryyrzz(
        Angle64::from_radians(a),
        Angle64::from_radians(b),
        Angle64::from_radians(c),
        &[(QubitId(0), QubitId(1))],
    );
    sim.rzz(Angle64::from_radians(-c), &[(QubitId(0), QubitId(1))]);
    sim.ryy(Angle64::from_radians(-b), &[(QubitId(0), QubitId(1))]);
    sim.rxx(Angle64::from_radians(-a), &[(QubitId(0), QubitId(1))]);
    assert_mz(sim, 0, false, "RXXRYYRZZ decomp|01> q0");
    assert_mz(sim, 1, true, "RXXRYYRZZ decomp|01> q1");

    // Also on |+0>
    sim.reset();
    sim.h(&qid(0));
    sim.rxxryyrzz(
        Angle64::from_radians(a),
        Angle64::from_radians(b),
        Angle64::from_radians(c),
        &[(QubitId(0), QubitId(1))],
    );
    sim.rzz(Angle64::from_radians(-c), &[(QubitId(0), QubitId(1))]);
    sim.ryy(Angle64::from_radians(-b), &[(QubitId(0), QubitId(1))]);
    sim.rxx(Angle64::from_radians(-a), &[(QubitId(0), QubitId(1))]);
    assert_mx(sim, 0, false, "RXXRYYRZZ decomp|+0> q0");
    assert_mz(sim, 1, false, "RXXRYYRZZ decomp|+0> q1");
}

// --- U2q General 2-Qubit Gate Tests ---

/// Verify U2q with identity parameters = I.
pub fn verify_u2q_identity<S: ArbitraryRotationGateable>(sim: &mut S) {
    let zero = [Angle64::ZERO; 3];
    let id_params = [zero; 2];

    // On |00>
    sim.reset();
    sim.u2q(
        id_params,
        [Angle64::ZERO; 3],
        id_params,
        &[(QubitId(0), QubitId(1))],
    );
    assert_mz(sim, 0, false, "U2q(I)|00> q0");
    assert_mz(sim, 1, false, "U2q(I)|00> q1");

    // On |10>
    sim.reset();
    sim.x(&qid(0));
    sim.u2q(
        id_params,
        [Angle64::ZERO; 3],
        id_params,
        &[(QubitId(0), QubitId(1))],
    );
    assert_mz(sim, 0, true, "U2q(I)|10> q0");
    assert_mz(sim, 1, false, "U2q(I)|10> q1");

    // On |+0>
    sim.reset();
    sim.h(&qid(0));
    sim.u2q(
        id_params,
        [Angle64::ZERO; 3],
        id_params,
        &[(QubitId(0), QubitId(1))],
    );
    assert_mx(sim, 0, false, "U2q(I)|+0> q0");
    assert_mz(sim, 1, false, "U2q(I)|+0> q1");
}

/// Verify U2q * `U2q_inverse` = I for a non-trivial decomposition.
pub fn verify_u2q_inverse<S: ArbitraryRotationGateable>(sim: &mut S) {
    // Use non-trivial parameters: single-qubit rotations + interaction
    let before = [
        [
            Angle64::from_radians(0.5),
            Angle64::from_radians(0.3),
            Angle64::from_radians(0.7),
        ],
        [
            Angle64::from_radians(1.0),
            Angle64::from_radians(0.2),
            Angle64::from_radians(0.4),
        ],
    ];
    let interaction = [
        Angle64::from_radians(0.6),
        Angle64::from_radians(0.3),
        Angle64::from_radians(0.8),
    ];
    let after = [
        [
            Angle64::from_radians(0.9),
            Angle64::from_radians(0.1),
            Angle64::from_radians(0.5),
        ],
        [
            Angle64::from_radians(0.4),
            Angle64::from_radians(0.7),
            Angle64::from_radians(0.2),
        ],
    ];

    // U2q_inverse: swap before/after and negate+swap phi/lambda, negate interaction
    let inv_before = [
        [-after[0][0], -after[0][2], -after[0][1]],
        [-after[1][0], -after[1][2], -after[1][1]],
    ];
    let inv_interaction = [-interaction[0], -interaction[1], -interaction[2]];
    let inv_after = [
        [-before[0][0], -before[0][2], -before[0][1]],
        [-before[1][0], -before[1][2], -before[1][1]],
    ];

    // On |01>
    sim.reset();
    sim.x(&qid(1));
    sim.u2q(before, interaction, after, &[(QubitId(0), QubitId(1))]);
    sim.u2q(
        inv_before,
        inv_interaction,
        inv_after,
        &[(QubitId(0), QubitId(1))],
    );
    assert_mz(sim, 0, false, "U2q*U2q_inv|01> q0");
    assert_mz(sim, 1, true, "U2q*U2q_inv|01> q1");

    // On |+0>
    sim.reset();
    sim.h(&qid(0));
    sim.u2q(before, interaction, after, &[(QubitId(0), QubitId(1))]);
    sim.u2q(
        inv_before,
        inv_interaction,
        inv_after,
        &[(QubitId(0), QubitId(1))],
    );
    assert_mx(sim, 0, false, "U2q*U2q_inv|+0> q0");
    assert_mz(sim, 1, false, "U2q*U2q_inv|+0> q1");
}

/// Verify U2q with only interaction (no single-qubit gates) matches RXXRYYRZZ.
pub fn verify_u2q_matches_rxxryyrzz<S: ArbitraryRotationGateable>(sim: &mut S) {
    let zero = [Angle64::ZERO; 3];
    let id_params = [zero; 2];
    let interaction = [
        Angle64::from_radians(0.5),
        Angle64::from_radians(0.3),
        Angle64::from_radians(0.7),
    ];

    // Apply U2q(I, interaction, I) then undo with RXXRYYRZZ(-a,-b,-c)
    // They should be equivalent since u2q with identity single-qubit gates
    // is just rxxryyrzz.
    sim.reset();
    sim.x(&qid(1)); // |01>
    sim.u2q(
        id_params,
        interaction,
        id_params,
        &[(QubitId(0), QubitId(1))],
    );
    sim.rxxryyrzz(
        -interaction[0],
        -interaction[1],
        -interaction[2],
        &[(QubitId(0), QubitId(1))],
    );
    assert_mz(sim, 0, false, "U2q(I,int,I)*RXXRYYRZZ_inv|01> q0");
    assert_mz(sim, 1, true, "U2q(I,int,I)*RXXRYYRZZ_inv|01> q1");

    // On |+0>
    sim.reset();
    sim.h(&qid(0));
    sim.u2q(
        id_params,
        interaction,
        id_params,
        &[(QubitId(0), QubitId(1))],
    );
    sim.rxxryyrzz(
        -interaction[0],
        -interaction[1],
        -interaction[2],
        &[(QubitId(0), QubitId(1))],
    );
    assert_mx(sim, 0, false, "U2q(I,int,I)*RXXRYYRZZ_inv|+0> q0");
    assert_mz(sim, 1, false, "U2q(I,int,I)*RXXRYYRZZ_inv|+0> q1");
}

// --- Half-Pi Clifford Equivalences ---

/// Verify RZ(pi/2) = SZ (up to global phase).
pub fn verify_rz_half_pi_is_sz<S: ArbitraryRotationGateable>(sim: &mut S) {
    // Both should map |+> to |+Y>
    sim.reset();
    sim.h(&qid(0));
    sim.rz(Angle64::from_radians(FRAC_PI_2), &qid(0));
    assert_my(sim, 0, false, "RZ(pi/2)|+> should be |+Y>");

    sim.reset();
    sim.h(&qid(0));
    sim.sz(&qid(0));
    assert_my(sim, 0, false, "SZ|+> should be |+Y>");

    // RZ(pi/2)^2 = RZ(pi) = Z, same as SZ^2 = Z
    sim.reset();
    sim.h(&qid(0));
    sim.rz(Angle64::from_radians(FRAC_PI_2), &qid(0));
    sim.rz(Angle64::from_radians(FRAC_PI_2), &qid(0));
    assert_mx(sim, 0, true, "RZ(pi/2)^2|+> = Z|+> = |->");
}

/// Verify RX(pi/2) = SX (up to global phase).
pub fn verify_rx_half_pi_is_sx<S: ArbitraryRotationGateable>(sim: &mut S) {
    // Both should map |0> to a state where my gives a specific result
    // SX|0> has my = false (as tested in clifford tests). Check RX(pi/2)|0> matches.
    sim.reset();
    sim.rx(Angle64::from_radians(FRAC_PI_2), &qid(0));
    let rx_my = sim.my(&qid(0));

    sim.reset();
    sim.sx(&qid(0));
    let sx_my = sim.my(&qid(0));

    assert_eq!(
        rx_my[0].is_deterministic, sx_my[0].is_deterministic,
        "RX(pi/2) vs SX on |0>: determinism should match"
    );
    if rx_my[0].is_deterministic {
        assert_eq!(
            rx_my[0].outcome, sx_my[0].outcome,
            "RX(pi/2) vs SX on |0>: my outcome should match"
        );
    }

    // RX(pi/2)^2 = RX(pi) = X, same as SX^2 = X
    sim.reset();
    sim.rx(Angle64::from_radians(FRAC_PI_2), &qid(0));
    sim.rx(Angle64::from_radians(FRAC_PI_2), &qid(0));
    assert_mz(sim, 0, true, "RX(pi/2)^2|0> = X|0> = |1>");
}

/// Verify RY(pi/2) = SY (up to global phase).
pub fn verify_ry_half_pi_is_sy<S: ArbitraryRotationGateable>(sim: &mut S) {
    // Both should map |0> to the same state
    sim.reset();
    sim.ry(Angle64::from_radians(FRAC_PI_2), &qid(0));
    let ry_mx = sim.mx(&qid(0));

    sim.reset();
    sim.sy(&qid(0));
    let sy_mx = sim.mx(&qid(0));

    assert_eq!(
        ry_mx[0].is_deterministic, sy_mx[0].is_deterministic,
        "RY(pi/2) vs SY on |0>: determinism should match"
    );
    if ry_mx[0].is_deterministic {
        assert_eq!(
            ry_mx[0].outcome, sy_mx[0].outcome,
            "RY(pi/2) vs SY on |0>: mx outcome should match"
        );
    }

    // RY(pi/2)^2 = RY(pi) = Y, same as SY^2 = Y
    sim.reset();
    sim.ry(Angle64::from_radians(FRAC_PI_2), &qid(0));
    sim.ry(Angle64::from_radians(FRAC_PI_2), &qid(0));
    assert_mz(sim, 0, true, "RY(pi/2)^2|0> = Y|0> = |1>");
}

/// Verify RZ(pi/4) = T (up to global phase).
pub fn verify_rz_quarter_pi_is_t<S: ArbitraryRotationGateable>(sim: &mut S) {
    // T^8 = I, so RZ(pi/4)^8 should also be identity
    sim.reset();
    sim.h(&qid(0)); // |+>
    for _ in 0..8 {
        sim.rz(Angle64::from_radians(FRAC_PI_4), &qid(0));
    }
    assert_mx(sim, 0, false, "RZ(pi/4)^8|+> = |+>");

    // RZ(pi/4) * RZ(-pi/4) = I, same as T * Tdg = I
    sim.reset();
    sim.h(&qid(0));
    sim.rz(Angle64::from_radians(FRAC_PI_4), &qid(0));
    sim.tdg(&qid(0));
    assert_mx(sim, 0, false, "RZ(pi/4)*Tdg|+> = |+>");
}

// --- Aggregator ---

/// Run all measurement-based rotation gate tests.
///
/// These tests verify rotation gate contracts using only measurement outcomes,
/// suitable for any simulator implementing `ArbitraryRotationGateable`.
///
/// # Arguments
/// * `sim` - A mutable reference to any `ArbitraryRotationGateable` simulator
/// * `num_qubits` - Number of qubits available
pub fn run_rotation_gate_tests<S: ArbitraryRotationGateable>(sim: &mut S, num_qubits: usize) {
    assert!(num_qubits >= 1, "Need at least 1 qubit");

    // -- Clifford-angle equivalences --
    verify_rx_pi_equals_x(sim);
    verify_ry_pi_equals_y(sim);
    verify_rz_pi_equals_z(sim);

    // -- Identity at zero --
    verify_rotation_identity_at_zero(sim);

    // -- Inverse rotations --
    verify_rotation_inverse(sim);

    // -- T gate --
    verify_t_eighth_power(sim);
    verify_t_adjoint(sim);

    // -- Rotation composition --
    verify_rz_composition(sim);

    // -- Superposition from rotations --
    verify_rx_half_pi_superposition(sim);
    verify_ry_half_pi_superposition(sim);

    // -- U gate --
    verify_u_identity(sim);
    verify_u_as_x(sim);
    verify_u_as_h(sim);
    verify_u_inverse(sim);
    verify_u_after_clifford_ordering(sim);

    // -- R1XY gate --
    verify_r1xy_identity(sim);
    verify_r1xy_as_rx(sim);
    verify_r1xy_as_ry(sim);
    verify_r1xy_inverse(sim);

    // -- Half-pi Clifford equivalences --
    verify_rz_half_pi_is_sz(sim);
    verify_rx_half_pi_is_sx(sim);
    verify_ry_half_pi_is_sy(sim);
    verify_rz_quarter_pi_is_t(sim);

    // -- Circuit inverse --
    verify_rotation_circuit_inverse(sim);

    if num_qubits >= 2 {
        // -- Two-qubit rotation tests --
        verify_rzz_special_angles(sim);
        verify_rxx_ryy_identity(sim);
        verify_two_qubit_rotation_inverse(sim);

        // -- RXXRYYRZZ composite gate --
        verify_rxxryyrzz_identity(sim);
        verify_rxxryyrzz_inverse(sim);
        verify_rxxryyrzz_decomposition(sim);

        // -- U2q general 2-qubit gate --
        verify_u2q_identity(sim);
        verify_u2q_inverse(sim);
        verify_u2q_matches_rxxryyrzz(sim);
    }
}

// --- Test Suite Macro ---

/// Generates a measurement-based rotation test suite for any `ArbitraryRotationGateable` simulator.
///
/// # Arguments
///
/// * `$sim_type` - The simulator type
/// * `$num_qubits` - Number of qubits
/// * `$constructor` - Expression to create the simulator
///
/// # Example
///
/// ```text
/// use pecos_simulators::rotation_test_suite;
///
/// rotation_test_suite!(MySimType, 4, MySimType::new(num_qubits));
/// ```
#[macro_export]
macro_rules! rotation_test_suite {
    ($sim_type:ty, $num_qubits_val:expr, $constructor:expr) => {
        paste::paste! {
            #[test]
            fn [<test_ $sim_type:snake _rotation_suite>]() {
                let num_qubits = $num_qubits_val;
                let mut sim = $constructor;
                $crate::rotation_test_utils::run_rotation_gate_tests(&mut sim, num_qubits);
            }
        }
    };
}
