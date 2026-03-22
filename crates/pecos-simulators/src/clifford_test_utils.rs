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

//! Shared measurement-based test utilities for any `CliffordGateable` simulator.
//!
//! These tests verify gate contracts using only measurement outcomes (`mz`, `mx`, `my`),
//! making them usable by stabilizer simulators, state vector simulators, and any future
//! simulator type that implements [`CliffordGateable`].
//!
//! For simulator-specific tests (exact amplitudes, stabilizer tableau comparisons, etc.),
//! see [`state_vector_test_utils`] and [`stabilizer_test_utils`].
//!
//! # Example
//!
//! ```ignore
//! use pecos_simulators::clifford_test_utils::run_clifford_gate_tests;
//! use pecos_simulators::{CliffordGateable, QuantumSimulator};
//!
//! fn test_my_simulator<S: CliffordGateable>(sim: &mut S, num_qubits: usize) {
//!     run_clifford_gate_tests(sim, num_qubits);
//! }
//! ```

#![allow(clippy::missing_panics_doc)]

use crate::CliffordGateable;
use pecos_core::{QubitId, qid, qid2};

// ============================================================================
// Helper: deterministic measurement assertion
// ============================================================================

/// Assert that measuring qubit `q` in Z basis gives a deterministic result with the expected outcome.
fn assert_mz<S: CliffordGateable>(sim: &mut S, q: usize, expected: bool, msg: &str) {
    let result = sim.mz(&qid(q));
    assert!(result[0].is_deterministic, "{msg}: should be deterministic");
    assert_eq!(result[0].outcome, expected, "{msg}: wrong outcome");
}

/// Assert that measuring qubit `q` in Z basis gives a non-deterministic result (superposition).
fn assert_mz_superposition<S: CliffordGateable>(sim: &mut S, q: usize, msg: &str) {
    let result = sim.mz(&qid(q));
    assert!(
        !result[0].is_deterministic,
        "{msg}: should be non-deterministic"
    );
}

/// Assert that measuring qubit `q` in X basis gives a deterministic result with the expected outcome.
fn assert_mx<S: CliffordGateable>(sim: &mut S, q: usize, expected: bool, msg: &str) {
    let result = sim.mx(&qid(q));
    assert!(result[0].is_deterministic, "{msg}: should be deterministic");
    assert_eq!(result[0].outcome, expected, "{msg}: wrong outcome");
}

/// Assert that measuring qubit `q` in Y basis gives a deterministic result with the expected outcome.
fn assert_my<S: CliffordGateable>(sim: &mut S, q: usize, expected: bool, msg: &str) {
    let result = sim.my(&qid(q));
    assert!(result[0].is_deterministic, "{msg}: should be deterministic");
    assert_eq!(result[0].outcome, expected, "{msg}: wrong outcome");
}

// ============================================================================
// Single-Qubit Gate Identity Tests
// ============================================================================

/// Verify H^2 = I by testing on |0>, |1>, and |+>.
pub fn verify_h_squared<S: CliffordGateable>(sim: &mut S) {
    // On |0>: H^2|0> = |0>
    sim.reset();
    sim.h(&qid(0)).h(&qid(0));
    assert_mz(sim, 0, false, "H^2|0>");

    // On |1>: H^2|1> = |1>
    sim.reset();
    sim.x(&qid(0));
    sim.h(&qid(0)).h(&qid(0));
    assert_mz(sim, 0, true, "H^2|1>");

    // On |+>: H^2|+> = |+>
    sim.reset();
    sim.h(&qid(0));
    sim.h(&qid(0)).h(&qid(0));
    assert_mx(sim, 0, false, "H^2|+>");
}

/// Verify X^2 = I.
pub fn verify_x_squared<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.x(&qid(0)).x(&qid(0));
    assert_mz(sim, 0, false, "X^2|0>");

    sim.reset();
    sim.x(&qid(0));
    sim.x(&qid(0)).x(&qid(0));
    assert_mz(sim, 0, true, "X^2|1>");
}

/// Verify Y^2 = I.
pub fn verify_y_squared<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.y(&qid(0)).y(&qid(0));
    assert_mz(sim, 0, false, "Y^2|0>");

    sim.reset();
    sim.x(&qid(0));
    sim.y(&qid(0)).y(&qid(0));
    assert_mz(sim, 0, true, "Y^2|1>");
}

/// Verify Z^2 = I.
pub fn verify_z_squared<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.z(&qid(0)).z(&qid(0));
    assert_mz(sim, 0, false, "Z^2|0>");

    // Also test that Z^2 preserves superposition
    sim.reset();
    sim.h(&qid(0));
    sim.z(&qid(0)).z(&qid(0));
    assert_mx(sim, 0, false, "Z^2|+>");
}

/// Verify S^4 = I (SZ applied four times).
pub fn verify_s_fourth<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0));
    for _ in 0..4 {
        sim.sz(&qid(0));
    }
    sim.h(&qid(0));
    assert_mz(sim, 0, false, "H S^4 H|0>");
}

/// Verify S^2 = Z.
///
/// Tests that applying S twice has the same measurement effect as Z.
pub fn verify_s_squared_is_z<S: CliffordGateable>(sim: &mut S) {
    // S^2|+> should give |-> (same as Z|+>)
    sim.reset();
    sim.h(&qid(0));
    sim.sz(&qid(0)).sz(&qid(0));
    assert_mx(sim, 0, true, "S^2|+> should be |->");

    // Z|+> should also give |->
    sim.reset();
    sim.h(&qid(0));
    sim.z(&qid(0));
    assert_mx(sim, 0, true, "Z|+> should be |->");
}

/// Verify SX^2 = X.
pub fn verify_sx_squared_is_x<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.sx(&qid(0)).sx(&qid(0));
    assert_mz(sim, 0, true, "SX^2|0> = X|0> = |1>");
}

/// Verify SY^2 = Y.
pub fn verify_sy_squared_is_y<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.sy(&qid(0)).sy(&qid(0));
    // Y|0> = i|1>, measurement gives 1
    assert_mz(sim, 0, true, "SY^2|0> = Y|0> should measure 1");
}

// ============================================================================
// Single-Qubit Adjoint Pair Tests
// ============================================================================

/// Helper: verify G * Gdg = I on |0>, |1>, and |+>.
fn verify_adjoint_pair<S: CliffordGateable>(
    sim: &mut S,
    apply_g: fn(&mut S),
    apply_gdg: fn(&mut S),
    name: &str,
) {
    // On |0>
    sim.reset();
    apply_g(sim);
    apply_gdg(sim);
    assert_mz(sim, 0, false, &format!("{name}*{name}dg|0>"));

    // On |1>
    sim.reset();
    sim.x(&qid(0));
    apply_g(sim);
    apply_gdg(sim);
    assert_mz(sim, 0, true, &format!("{name}*{name}dg|1>"));

    // On |+>
    sim.reset();
    sim.h(&qid(0));
    apply_g(sim);
    apply_gdg(sim);
    assert_mx(sim, 0, false, &format!("{name}*{name}dg|+>"));
}

/// Verify SZ * `SZdg` = I.
pub fn verify_sz_adjoint<S: CliffordGateable>(sim: &mut S) {
    verify_adjoint_pair(
        sim,
        |s| {
            s.sz(&qid(0));
        },
        |s| {
            s.szdg(&qid(0));
        },
        "SZ",
    );
}

/// Verify SX * `SXdg` = I.
pub fn verify_sx_adjoint<S: CliffordGateable>(sim: &mut S) {
    verify_adjoint_pair(
        sim,
        |s| {
            s.sx(&qid(0));
        },
        |s| {
            s.sxdg(&qid(0));
        },
        "SX",
    );
}

/// Verify SY * `SYdg` = I.
pub fn verify_sy_adjoint<S: CliffordGateable>(sim: &mut S) {
    verify_adjoint_pair(
        sim,
        |s| {
            s.sy(&qid(0));
        },
        |s| {
            s.sydg(&qid(0));
        },
        "SY",
    );
}

/// Verify F * Fdg = I.
pub fn verify_f_adjoint<S: CliffordGateable>(sim: &mut S) {
    verify_adjoint_pair(
        sim,
        |s| {
            s.f(&qid(0));
        },
        |s| {
            s.fdg(&qid(0));
        },
        "F",
    );
}

/// Verify F2 * F2dg = I.
pub fn verify_f2_adjoint<S: CliffordGateable>(sim: &mut S) {
    verify_adjoint_pair(
        sim,
        |s| {
            s.f2(&qid(0));
        },
        |s| {
            s.f2dg(&qid(0));
        },
        "F2",
    );
}

/// Verify F3 * F3dg = I.
pub fn verify_f3_adjoint<S: CliffordGateable>(sim: &mut S) {
    verify_adjoint_pair(
        sim,
        |s| {
            s.f3(&qid(0));
        },
        |s| {
            s.f3dg(&qid(0));
        },
        "F3",
    );
}

/// Verify F4 * F4dg = I.
pub fn verify_f4_adjoint<S: CliffordGateable>(sim: &mut S) {
    verify_adjoint_pair(
        sim,
        |s| {
            s.f4(&qid(0));
        },
        |s| {
            s.f4dg(&qid(0));
        },
        "F4",
    );
}

// ============================================================================
// Hadamard Variant Tests
// ============================================================================

/// Verify Hi^2 = I (up to global phase, invisible to measurement) for H2..H6.
pub fn verify_hadamard_variant_involutions<S: CliffordGateable>(sim: &mut S) {
    type GateEntry<T> = (&'static str, fn(&mut T));
    let variants: &[GateEntry<S>] = &[
        ("H2", |s: &mut S| {
            s.h2(&qid(0));
        }),
        ("H3", |s: &mut S| {
            s.h3(&qid(0));
        }),
        ("H4", |s: &mut S| {
            s.h4(&qid(0));
        }),
        ("H5", |s: &mut S| {
            s.h5(&qid(0));
        }),
        ("H6", |s: &mut S| {
            s.h6(&qid(0));
        }),
    ];

    for &(name, apply) in variants {
        // On |0>: Hi^2|0> should measure 0 (phase invisible)
        sim.reset();
        apply(sim);
        apply(sim);
        assert_mz(sim, 0, false, &format!("{name}^2|0>"));

        // On |1>: Hi^2|1> should measure 1
        sim.reset();
        sim.x(&qid(0));
        apply(sim);
        apply(sim);
        assert_mz(sim, 0, true, &format!("{name}^2|1>"));

        // On |+>: Hi^2|+> should preserve |+> (up to phase)
        sim.reset();
        sim.h(&qid(0));
        apply(sim);
        apply(sim);
        assert_mx(sim, 0, false, &format!("{name}^2|+>"));
    }
}

// ============================================================================
// Face Gate Tests
// ============================================================================

/// Verify F^3 = I (up to global phase, invisible to measurement).
///
/// The face gate cyclically permutes X -> Y -> Z -> X, so F^3 should be identity.
pub fn verify_face_gate_cube<S: CliffordGateable>(sim: &mut S) {
    // On |0>
    sim.reset();
    sim.f(&qid(0)).f(&qid(0)).f(&qid(0));
    assert_mz(sim, 0, false, "F^3|0>");

    // On |1>
    sim.reset();
    sim.x(&qid(0));
    sim.f(&qid(0)).f(&qid(0)).f(&qid(0));
    assert_mz(sim, 0, true, "F^3|1>");

    // On |+>
    sim.reset();
    sim.h(&qid(0));
    sim.f(&qid(0)).f(&qid(0)).f(&qid(0));
    assert_mx(sim, 0, false, "F^3|+>");
}

/// Verify F maps Z eigenstates to X eigenstates.
///
/// F: X -> Y -> Z -> X, so F maps Z eigenstates to X eigenstates.
/// F|0> (Z+ eigenstate) should be an X eigenstate.
pub fn verify_face_gate_axis_rotation<S: CliffordGateable>(sim: &mut S) {
    // F maps Z -> X, so |0> (Z+ eigenstate) -> X+ eigenstate
    sim.reset();
    sim.f(&qid(0));
    assert_mx(sim, 0, false, "F|0> should be X+ eigenstate");

    // F maps |1> (Z- eigenstate) -> X- eigenstate
    sim.reset();
    sim.x(&qid(0));
    sim.f(&qid(0));
    assert_mx(sim, 0, true, "F|1> should be X- eigenstate");

    // F maps X -> Y eigenstates: |+> -> Y+ eigenstate
    sim.reset();
    sim.h(&qid(0));
    sim.f(&qid(0));
    assert_my(sim, 0, false, "F|+> should be Y+ eigenstate");
}

// ============================================================================
// Basic Gate Behavior Tests
// ============================================================================

/// Verify initial state is |0...0>.
pub fn verify_initial_state<S: CliffordGateable>(sim: &mut S, num_qubits: usize) {
    sim.reset();
    for q in 0..num_qubits {
        assert_mz(sim, q, false, &format!("initial state qubit {q}"));
    }
}

/// Verify X flips |0> to |1>.
pub fn verify_x_gate<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.x(&qid(0));
    assert_mz(sim, 0, true, "X|0>");
}

/// Verify H creates superposition.
pub fn verify_h_superposition<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0));
    assert_mz_superposition(sim, 0, "H|0>");
}

/// Verify Z on |0> (no visible effect on Z measurement).
pub fn verify_z_on_zero<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.z(&qid(0));
    assert_mz(sim, 0, false, "Z|0>");
}

/// Verify Z on |1> (no visible effect on Z measurement, just phase).
pub fn verify_z_on_one<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.x(&qid(0));
    sim.z(&qid(0));
    assert_mz(sim, 0, true, "ZX|0>");
}

/// Verify SX creates superposition.
pub fn verify_sx_superposition<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.sx(&qid(0));
    assert_mz_superposition(sim, 0, "SX|0>");
}

/// Verify SY creates superposition.
pub fn verify_sy_superposition<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.sy(&qid(0));
    assert_mz_superposition(sim, 0, "SY|0>");
}

/// Verify SZ maps |+> to |+Y> (Y+ eigenstate).
pub fn verify_sz_maps_plus_to_plus_y<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0)); // |+>
    sim.sz(&qid(0));
    assert_my(sim, 0, false, "SZ|+> should be |+Y>");
}

/// Verify `SZdg` maps |+> to |-Y>.
pub fn verify_szdg_maps_plus_to_minus_y<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0)); // |+>
    sim.szdg(&qid(0));
    assert_my(sim, 0, true, "SZdg|+> should be |-Y>");
}

// ============================================================================
// Two-Qubit Gate Tests
// ============================================================================

/// Verify CX^2 = I.
pub fn verify_cx_squared<S: CliffordGateable>(sim: &mut S) {
    // On |10>: CX^2 should return to |10>
    sim.reset();
    sim.x(&qid(0));
    sim.cx(&qid2(0, 1)).cx(&qid2(0, 1));
    assert_mz(sim, 0, true, "CX^2|10> q0");
    assert_mz(sim, 1, false, "CX^2|10> q1");

    // On Bell state: CX^2 should preserve
    sim.reset();
    sim.h(&qid(0));
    sim.cx(&qid2(0, 1));
    sim.cx(&qid2(0, 1)).cx(&qid2(0, 1));
    // Still a Bell state: q0 non-deterministic, q1 correlated
    let r0 = sim.mz(&qid(0));
    let r1 = sim.mz(&qid(1));
    assert!(
        r1[0].is_deterministic,
        "CX^2 on Bell: q1 should be deterministic after q0 measurement"
    );
    assert_eq!(
        r0[0].outcome, r1[0].outcome,
        "CX^2 on Bell: should remain correlated"
    );
}

/// Verify CY^2 = I.
pub fn verify_cy_squared<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0)); // Superposition on control
    sim.cy(&qid2(0, 1)).cy(&qid2(0, 1));
    assert_mz(sim, 1, false, "CY^2: target should be |0>");
}

/// Verify CX behavior: control=0 no action, control=1 flips target.
pub fn verify_cx_behavior<S: CliffordGateable>(sim: &mut S) {
    // Control = 0: no action
    sim.reset();
    sim.cx(&qid2(0, 1));
    assert_mz(sim, 1, false, "CX control=0: target unchanged");

    // Control = 1: flips target
    sim.reset();
    sim.x(&qid(0));
    sim.cx(&qid2(0, 1));
    assert_mz(sim, 0, true, "CX control=1: control still 1");
    assert_mz(sim, 1, true, "CX control=1: target flipped to 1");
}

/// Verify CY behavior: control=0 no action, control=1 applies Y.
pub fn verify_cy_behavior<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.cy(&qid2(0, 1));
    assert_mz(sim, 1, false, "CY control=0: target unchanged");

    sim.reset();
    sim.x(&qid(0));
    sim.cy(&qid2(0, 1));
    // Y|0> = i|1>
    assert_mz(sim, 1, true, "CY control=1: target flipped to 1");
}

/// Verify CZ behavior: control=0 no action, control=1 applies Z.
pub fn verify_cz_behavior<S: CliffordGateable>(sim: &mut S) {
    // Control = 0: no action on target (test in X basis to detect Z)
    sim.reset();
    sim.h(&qid(1)); // |+> on target
    sim.cz(&qid2(0, 1));
    sim.h(&qid(1)); // Convert back
    assert_mz(sim, 1, false, "CZ control=0: H CZ H target = |0>");

    // Control = 1: applies Z to target
    sim.reset();
    sim.x(&qid(0));
    sim.h(&qid(1)); // |+> on target
    sim.cz(&qid2(0, 1));
    sim.h(&qid(1)); // H|-> = |1>
    assert_mz(sim, 1, true, "CZ control=1: H CZ H target = |1>");
}

/// Verify SWAP exchanges qubit states.
pub fn verify_swap_behavior<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.x(&qid(0)); // q0=|1>, q1=|0>
    sim.swap(&qid2(0, 1));
    assert_mz(sim, 0, false, "SWAP: q0 should be 0");
    assert_mz(sim, 1, true, "SWAP: q1 should be 1");
}

/// Verify SWAP^2 = I.
pub fn verify_swap_squared<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.x(&qid(0));
    sim.swap(&qid2(0, 1)).swap(&qid2(0, 1));
    assert_mz(sim, 0, true, "SWAP^2: q0 back to 1");
    assert_mz(sim, 1, false, "SWAP^2: q1 back to 0");
}

/// Verify SWAP is symmetric: SWAP(0,1) = SWAP(1,0).
pub fn verify_swap_symmetric<S: CliffordGateable>(sim: &mut S) {
    // SWAP(0,1)
    sim.reset();
    sim.x(&qid(0));
    sim.swap(&qid2(0, 1));
    let r0_a = sim.mz(&qid(0));
    let r1_a = sim.mz(&qid(1));

    // SWAP(1,0)
    sim.reset();
    sim.x(&qid(0));
    sim.swap(&[QubitId(1), QubitId(0)]);
    let r0_b = sim.mz(&qid(0));
    let r1_b = sim.mz(&qid(1));

    assert_eq!(
        r0_a[0].outcome, r0_b[0].outcome,
        "SWAP symmetry: q0 should match"
    );
    assert_eq!(
        r1_a[0].outcome, r1_b[0].outcome,
        "SWAP symmetry: q1 should match"
    );
}

/// Verify iSWAP swaps computational basis states (phase invisible to Z measurement).
pub fn verify_iswap_behavior<S: CliffordGateable>(sim: &mut S) {
    // iSWAP|10> = i|01>: q0 -> 0, q1 -> 1 (phase invisible)
    sim.reset();
    sim.x(&qid(0));
    sim.iswap(&qid2(0, 1));
    assert_mz(sim, 0, false, "iSWAP|10>: q0 = 0");
    assert_mz(sim, 1, true, "iSWAP|10>: q1 = 1");

    // iSWAP|00> = |00>
    sim.reset();
    sim.iswap(&qid2(0, 1));
    assert_mz(sim, 0, false, "iSWAP|00>: q0 = 0");
    assert_mz(sim, 1, false, "iSWAP|00>: q1 = 0");
}

// ============================================================================
// Two-Qubit Adjoint Pair Tests
// ============================================================================

/// Verify SXX * `SXXdg` = I on a Bell state.
pub fn verify_sxx_adjoint<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0)).cx(&qid2(0, 1)); // Bell state
    sim.sxx(&qid2(0, 1)).sxxdg(&qid2(0, 1));

    // Should still be a Bell state
    let r0 = sim.mz(&qid(0));
    let r1 = sim.mz(&qid(1));
    assert!(
        r1[0].is_deterministic,
        "SXX*SXXdg Bell: q1 deterministic after q0"
    );
    assert_eq!(
        r0[0].outcome, r1[0].outcome,
        "SXX*SXXdg Bell: still correlated"
    );
}

/// Verify SYY * `SYYdg` = I on a Bell state.
pub fn verify_syy_adjoint<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0)).cx(&qid2(0, 1));
    sim.syy(&qid2(0, 1)).syydg(&qid2(0, 1));

    let r0 = sim.mz(&qid(0));
    let r1 = sim.mz(&qid(1));
    assert!(
        r1[0].is_deterministic,
        "SYY*SYYdg Bell: q1 deterministic after q0"
    );
    assert_eq!(
        r0[0].outcome, r1[0].outcome,
        "SYY*SYYdg Bell: still correlated"
    );
}

/// Verify SZZ * `SZZdg` = I on a Bell state.
pub fn verify_szz_adjoint<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0)).cx(&qid2(0, 1));
    sim.szz(&qid2(0, 1)).szzdg(&qid2(0, 1));

    let r0 = sim.mz(&qid(0));
    let r1 = sim.mz(&qid(1));
    assert!(
        r1[0].is_deterministic,
        "SZZ*SZZdg Bell: q1 deterministic after q0"
    );
    assert_eq!(
        r0[0].outcome, r1[0].outcome,
        "SZZ*SZZdg Bell: still correlated"
    );
}

// ============================================================================
// Entanglement Tests
// ============================================================================

/// Verify Bell state |Phi+> = (|00> + |11>)/sqrt(2) has correct correlations.
pub fn verify_bell_state_correlations<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0)).cx(&qid2(0, 1));

    let r0 = sim.mz(&qid(0));
    assert!(
        !r0[0].is_deterministic,
        "Bell q0 should be non-deterministic"
    );

    let r1 = sim.mz(&qid(1));
    assert!(
        r1[0].is_deterministic,
        "Bell q1 should be deterministic after q0"
    );
    assert_eq!(r0[0].outcome, r1[0].outcome, "Bell pair should correlate");
}

/// Verify GHZ state has correct correlations.
pub fn verify_ghz_correlations<S: CliffordGateable>(sim: &mut S, num_qubits: usize) {
    assert!(
        num_qubits >= 2,
        "GHZ state requires at least 2 qubits, got {num_qubits}"
    );

    sim.reset();
    sim.h(&qid(0));
    for i in 0..(num_qubits - 1) {
        sim.cx(&qid2(i, i + 1));
    }

    let r0 = sim.mz(&qid(0));
    assert!(
        !r0[0].is_deterministic,
        "GHZ q0 should be non-deterministic"
    );

    for i in 1..num_qubits {
        let ri = sim.mz(&qid(i));
        assert!(
            ri[0].is_deterministic,
            "GHZ q{i} should be deterministic after q0"
        );
        assert_eq!(
            r0[0].outcome, ri[0].outcome,
            "GHZ q{i} should correlate with q0"
        );
    }
}

// ============================================================================
// Gate Decomposition Tests (deterministic inputs)
// ============================================================================

/// Verify SWAP = CX(0,1) CX(1,0) CX(0,1) on all 4 basis states.
pub fn verify_swap_decomposition<S: CliffordGateable>(sim: &mut S) {
    for state in 0..4u8 {
        let q0_val = (state & 1) != 0;
        let q1_val = (state >> 1) != 0;

        // Apply SWAP directly
        sim.reset();
        if q0_val {
            sim.x(&qid(0));
        }
        if q1_val {
            sim.x(&qid(1));
        }
        sim.swap(&qid2(0, 1));
        let swap_r0 = sim.mz(&qid(0));
        let swap_r1 = sim.mz(&qid(1));

        // Apply CX decomposition
        sim.reset();
        if q0_val {
            sim.x(&qid(0));
        }
        if q1_val {
            sim.x(&qid(1));
        }
        sim.cx(&qid2(0, 1))
            .cx(&[QubitId(1), QubitId(0)])
            .cx(&qid2(0, 1));
        let cx_r0 = sim.mz(&qid(0));
        let cx_r1 = sim.mz(&qid(1));

        assert_eq!(
            swap_r0[0].outcome, cx_r0[0].outcome,
            "SWAP=CX^3 on |{state:02b}>: q0 mismatch"
        );
        assert_eq!(
            swap_r1[0].outcome, cx_r1[0].outcome,
            "SWAP=CX^3 on |{state:02b}>: q1 mismatch"
        );
    }
}

/// Verify CZ = H(target) CX H(target) on all 4 basis states.
pub fn verify_cz_decomposition<S: CliffordGateable>(sim: &mut S) {
    for state in 0..4u8 {
        let q0_val = (state & 1) != 0;
        let q1_val = (state >> 1) != 0;

        // Apply CZ directly
        sim.reset();
        if q0_val {
            sim.x(&qid(0));
        }
        if q1_val {
            sim.x(&qid(1));
        }
        sim.cz(&qid2(0, 1));
        // CZ only applies phase, so Z-measurement outcomes are unchanged
        let cz_r0 = sim.mz(&qid(0));
        let cz_r1 = sim.mz(&qid(1));

        // Apply H CX H decomposition
        sim.reset();
        if q0_val {
            sim.x(&qid(0));
        }
        if q1_val {
            sim.x(&qid(1));
        }
        sim.h(&qid(1)).cx(&qid2(0, 1)).h(&qid(1));
        let decomp_r0 = sim.mz(&qid(0));
        let decomp_r1 = sim.mz(&qid(1));

        assert_eq!(
            cz_r0[0].outcome, decomp_r0[0].outcome,
            "CZ=HCXh on |{state:02b}>: q0 mismatch"
        );
        assert_eq!(
            cz_r1[0].outcome, decomp_r1[0].outcome,
            "CZ=HCXH on |{state:02b}>: q1 mismatch"
        );
    }
}

/// Verify X = HZH on both basis states.
pub fn verify_x_hzh_decomposition<S: CliffordGateable>(sim: &mut S) {
    // On |0>: X|0> = |1>, HZH|0> = |1>
    sim.reset();
    sim.x(&qid(0));
    let x_r = sim.mz(&qid(0));

    sim.reset();
    sim.h(&qid(0)).z(&qid(0)).h(&qid(0));
    let decomp_r = sim.mz(&qid(0));

    assert_eq!(
        x_r[0].outcome, decomp_r[0].outcome,
        "X=HZH on |0>: mismatch"
    );

    // On |1>
    sim.reset();
    sim.x(&qid(0));
    sim.x(&qid(0));
    let x_r = sim.mz(&qid(0));

    sim.reset();
    sim.x(&qid(0));
    sim.h(&qid(0)).z(&qid(0)).h(&qid(0));
    let decomp_r = sim.mz(&qid(0));

    assert_eq!(
        x_r[0].outcome, decomp_r[0].outcome,
        "X=HZH on |1>: mismatch"
    );
}

/// Verify Z = S^2 via measurement on both |0> and |+>.
pub fn verify_z_from_s_squared<S: CliffordGateable>(sim: &mut S) {
    // On |+>: Z|+> = |->, S^2|+> should also be |->
    sim.reset();
    sim.h(&qid(0));
    sim.z(&qid(0));
    let z_r = sim.mx(&qid(0));

    sim.reset();
    sim.h(&qid(0));
    sim.sz(&qid(0)).sz(&qid(0));
    let s2_r = sim.mx(&qid(0));

    assert_eq!(z_r[0].outcome, s2_r[0].outcome, "Z=S^2 on |+>: mismatch");
}

// ============================================================================
// Commutativity Tests
// ============================================================================

/// Verify that gates on different qubits commute.
///
/// Tests X(0)H(1) vs H(1)X(0) on |00>. Both should give the same result
/// since they act on independent qubits.
pub fn verify_different_qubit_commutativity<S: CliffordGateable>(sim: &mut S) {
    // X(0)H(1) on |00>
    sim.reset();
    sim.x(&qid(0)).h(&qid(1));
    let r0_a = sim.mz(&qid(0));
    let r1_a = sim.mx(&qid(1)); // Measure q1 in X basis since it's in |+>

    // H(1)X(0) on |00>
    sim.reset();
    sim.h(&qid(1)).x(&qid(0));
    let r0_b = sim.mz(&qid(0));
    let r1_b = sim.mx(&qid(1));

    assert_eq!(
        r0_a[0].outcome, r0_b[0].outcome,
        "Different-qubit commutativity: q0"
    );
    assert_eq!(
        r1_a[0].outcome, r1_b[0].outcome,
        "Different-qubit commutativity: q1"
    );
    assert_eq!(
        r0_a[0].is_deterministic, r0_b[0].is_deterministic,
        "Different-qubit commutativity: q0 determinism"
    );
    assert_eq!(
        r1_a[0].is_deterministic, r1_b[0].is_deterministic,
        "Different-qubit commutativity: q1 determinism"
    );
}

/// Verify that H and Z do not commute on the same qubit.
///
/// HZ|0> produces |-> (X- eigenstate), while ZH|0> produces |+> (X+ eigenstate).
pub fn verify_same_qubit_non_commutativity<S: CliffordGateable>(sim: &mut S) {
    // ZH|0> = Z|+> = |->
    sim.reset();
    sim.h(&qid(0)).z(&qid(0));
    assert_mx(sim, 0, true, "ZH|0> should be |->");

    // HZ|0> = H|0> = |+> (Z|0> = |0>)
    sim.reset();
    sim.z(&qid(0)).h(&qid(0));
    assert_mx(sim, 0, false, "HZ|0> should be |+>");
}

// ============================================================================
// Measurement Property Tests
// ============================================================================

/// Verify measurement idempotence: measuring twice gives same result.
pub fn verify_measurement_idempotence<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0));

    let r1 = sim.mz(&qid(0));
    let r2 = sim.mz(&qid(0));

    assert!(
        r2[0].is_deterministic,
        "Second measurement should be deterministic"
    );
    assert_eq!(
        r1[0].outcome, r2[0].outcome,
        "Second measurement should match first"
    );
}

/// Verify measurement idempotence on an entangled state.
pub fn verify_measurement_idempotence_entangled<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0)).cx(&qid2(0, 1));

    let r1 = sim.mz(&qid(0));
    let r2 = sim.mz(&qid(0));

    assert!(
        r2[0].is_deterministic,
        "Second Bell measurement should be deterministic"
    );
    assert_eq!(
        r1[0].outcome, r2[0].outcome,
        "Bell: second measurement should match first"
    );

    let r3 = sim.mz(&qid(1));
    assert!(
        r3[0].is_deterministic,
        "Bell partner should be deterministic"
    );
    assert_eq!(
        r1[0].outcome, r3[0].outcome,
        "Bell partner should correlate"
    );
}

/// Verify that X flips a measured qubit's outcome.
pub fn verify_measurement_then_gate<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.h(&qid(0));

    let r1 = sim.mz(&qid(0));
    sim.x(&qid(0));
    let r2 = sim.mz(&qid(0));

    assert!(
        r2[0].is_deterministic,
        "Post-X measurement should be deterministic"
    );
    assert_ne!(
        r1[0].outcome, r2[0].outcome,
        "X should flip the measurement outcome"
    );
}

// ============================================================================
// Identity Gate
// ============================================================================

/// Verify that the identity gate does not change the state.
pub fn verify_identity_gate<S: CliffordGateable>(sim: &mut S) {
    // On |0>
    sim.reset();
    sim.identity(&qid(0));
    assert_mz(sim, 0, false, "identity|0>");

    // On |1>
    sim.reset();
    sim.x(&qid(0));
    sim.identity(&qid(0));
    assert_mz(sim, 0, true, "identity|1>");

    // On |+>
    sim.reset();
    sim.h(&qid(0));
    sim.identity(&qid(0));
    assert_mx(sim, 0, false, "identity|+>");
}

// ============================================================================
// G Gate Tests
// ============================================================================

/// Verify G^2 = I on all computational basis states and a superposition.
///
/// G = CZ * H(q0) * H(q1) * CZ, so
/// G^2 = CZ * H*H * CZ * CZ * H*H * CZ = CZ * I * I * CZ = I.
pub fn verify_g_squared<S: CliffordGateable>(sim: &mut S) {
    let basis_states: &[(bool, bool)] =
        &[(false, false), (false, true), (true, false), (true, true)];

    for &(q0_one, q1_one) in basis_states {
        sim.reset();
        if q0_one {
            sim.x(&qid(0));
        }
        if q1_one {
            sim.x(&qid(1));
        }
        sim.g(&qid2(0, 1)).g(&qid2(0, 1));

        let label = format!("G^2|{}{}>", u8::from(q0_one), u8::from(q1_one));
        assert_mz(sim, 0, q0_one, &format!("{label} q0"));
        assert_mz(sim, 1, q1_one, &format!("{label} q1"));
    }

    // Also test on |+0>
    sim.reset();
    sim.h(&qid(0));
    sim.g(&qid2(0, 1)).g(&qid2(0, 1));
    assert_mx(sim, 0, false, "G^2|+0> q0 should be |+>");
    assert_mz(sim, 1, false, "G^2|+0> q1 should be |0>");
}

/// Verify that G on |00> creates a non-trivial state (cluster state).
///
/// G|00> = CZ * H * H * CZ|00> = CZ|++>, which is a cluster state
/// where both qubits are in superposition under Z-basis measurement.
pub fn verify_g_gate_creates_superposition<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.g(&qid2(0, 1));

    // Both qubits should be in superposition (non-deterministic under mz)
    assert_mz_superposition(sim, 0, "G|00>: q0 should be in superposition");
}

// ============================================================================
// Preparation Gate Tests
// ============================================================================

/// Verify `px` prepares the |+> state.
pub fn verify_px_prepares_plus<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.px(&qid(0));
    assert_mx(sim, 0, false, "px prepares |+>");
}

/// Verify `pnx` prepares the |-> state.
pub fn verify_pnx_prepares_minus<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.pnx(&qid(0));
    assert_mx(sim, 0, true, "pnx prepares |->");
}

/// Verify `py` prepares the |+Y> state.
pub fn verify_py_prepares_plus_y<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.py(&qid(0));
    assert_my(sim, 0, false, "py prepares |+Y>");
}

/// Verify `pny` prepares the |-Y> state.
pub fn verify_pny_prepares_minus_y<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.pny(&qid(0));
    assert_my(sim, 0, true, "pny prepares |-Y>");
}

/// Verify `pz` prepares the |0> state.
pub fn verify_pz_prepares_zero<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.pz(&qid(0));
    assert_mz(sim, 0, false, "pz prepares |0>");
}

/// Verify `pnz` prepares the |1> state.
pub fn verify_pnz_prepares_one<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.pnz(&qid(0));
    assert_mz(sim, 0, true, "pnz prepares |1>");
}

// ============================================================================
// Measure-and-Prepare Gate Tests
// ============================================================================

/// Verify `mpx` always prepares |+> regardless of initial state.
pub fn verify_mpx_prepares_plus<S: CliffordGateable>(sim: &mut S) {
    // From |0>
    sim.reset();
    sim.mpx(&qid(0));
    assert_mx(sim, 0, false, "mpx from |0> prepares |+>");

    // From |1>
    sim.reset();
    sim.x(&qid(0));
    sim.mpx(&qid(0));
    assert_mx(sim, 0, false, "mpx from |1> prepares |+>");

    // From |+> (superposition, exercises correction path stochastically)
    sim.reset();
    sim.h(&qid(0));
    sim.mpx(&qid(0));
    assert_mx(sim, 0, false, "mpx from |+> prepares |+>");
}

/// Verify `mpnx` always prepares |->.
pub fn verify_mpnx_prepares_minus<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.mpnx(&qid(0));
    assert_mx(sim, 0, true, "mpnx from |0> prepares |->");

    sim.reset();
    sim.x(&qid(0));
    sim.mpnx(&qid(0));
    assert_mx(sim, 0, true, "mpnx from |1> prepares |->");
}

/// Verify `mpy` always prepares |+Y>.
pub fn verify_mpy_prepares_plus_y<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.mpy(&qid(0));
    assert_my(sim, 0, false, "mpy from |0> prepares |+Y>");

    sim.reset();
    sim.x(&qid(0));
    sim.mpy(&qid(0));
    assert_my(sim, 0, false, "mpy from |1> prepares |+Y>");
}

/// Verify `mpny` always prepares |-Y>.
pub fn verify_mpny_prepares_minus_y<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.mpny(&qid(0));
    assert_my(sim, 0, true, "mpny from |0> prepares |-Y>");

    sim.reset();
    sim.x(&qid(0));
    sim.mpny(&qid(0));
    assert_my(sim, 0, true, "mpny from |1> prepares |-Y>");
}

/// Verify `mpz` always prepares |0>.
pub fn verify_mpz_prepares_zero<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.mpz(&qid(0));
    assert_mz(sim, 0, false, "mpz from |0> prepares |0>");

    sim.reset();
    sim.x(&qid(0));
    sim.mpz(&qid(0));
    assert_mz(sim, 0, false, "mpz from |1> prepares |0>");

    // From superposition
    sim.reset();
    sim.h(&qid(0));
    sim.mpz(&qid(0));
    assert_mz(sim, 0, false, "mpz from |+> prepares |0>");
}

/// Verify `mpnz` always prepares |1>.
pub fn verify_mpnz_prepares_one<S: CliffordGateable>(sim: &mut S) {
    sim.reset();
    sim.mpnz(&qid(0));
    assert_mz(sim, 0, true, "mpnz from |0> prepares |1>");

    sim.reset();
    sim.x(&qid(0));
    sim.mpnz(&qid(0));
    assert_mz(sim, 0, true, "mpnz from |1> prepares |1>");
}

// ============================================================================
// Negative Measurement Tests
// ============================================================================

/// Verify `mnx` on eigenstates gives the flipped outcome compared to `mx`.
pub fn verify_mnx_on_eigenstates<S: CliffordGateable>(sim: &mut S) {
    // mx on |+> = false (deterministic), so mnx on |+> = true (deterministic)
    sim.reset();
    sim.h(&qid(0)); // |+>
    let result = sim.mnx(&qid(0));
    assert!(
        result[0].is_deterministic,
        "mnx|+>: should be deterministic"
    );
    assert!(
        result[0].outcome,
        "mnx|+>: should be true (flipped from mx)"
    );

    // mx on |-> = true (deterministic), so mnx on |-> = false (deterministic)
    sim.reset();
    sim.x(&qid(0));
    sim.h(&qid(0)); // |->
    let result = sim.mnx(&qid(0));
    assert!(
        result[0].is_deterministic,
        "mnx|->: should be deterministic"
    );
    assert!(
        !result[0].outcome,
        "mnx|->: should be false (flipped from mx)"
    );
}

/// Verify `mny` on eigenstates gives the flipped outcome compared to `my`.
pub fn verify_mny_on_eigenstates<S: CliffordGateable>(sim: &mut S) {
    // Prepare |+Y>: SZ|+> has my = false, so mny should give true
    sim.reset();
    sim.h(&qid(0));
    sim.sz(&qid(0)); // |+Y>
    let result = sim.mny(&qid(0));
    assert!(
        result[0].is_deterministic,
        "mny|+Y>: should be deterministic"
    );
    assert!(
        result[0].outcome,
        "mny|+Y>: should be true (flipped from my)"
    );

    // Prepare |-Y>: SZdg|+> has my = true, so mny should give false
    sim.reset();
    sim.h(&qid(0));
    sim.szdg(&qid(0)); // |-Y>
    let result = sim.mny(&qid(0));
    assert!(
        result[0].is_deterministic,
        "mny|-Y>: should be deterministic"
    );
    assert!(
        !result[0].outcome,
        "mny|-Y>: should be false (flipped from my)"
    );
}

/// Verify `mnz` on eigenstates gives the flipped outcome compared to `mz`.
pub fn verify_mnz_on_eigenstates<S: CliffordGateable>(sim: &mut S) {
    // mz on |0> = false, so mnz on |0> = true
    sim.reset();
    let result = sim.mnz(&qid(0));
    assert!(
        result[0].is_deterministic,
        "mnz|0>: should be deterministic"
    );
    assert!(
        result[0].outcome,
        "mnz|0>: should be true (flipped from mz)"
    );

    // mz on |1> = true, so mnz on |1> = false
    sim.reset();
    sim.x(&qid(0));
    let result = sim.mnz(&qid(0));
    assert!(
        result[0].is_deterministic,
        "mnz|1>: should be deterministic"
    );
    assert!(
        !result[0].outcome,
        "mnz|1>: should be false (flipped from mz)"
    );
}

// ============================================================================
// Measurement Statistics
// ============================================================================

/// Verify that measuring a superposition state produces non-trivial statistics.
pub fn verify_measurement_statistics<S: CliffordGateable>(sim: &mut S) {
    let n = 1000;
    let mut count_one = 0;

    for _ in 0..n {
        sim.reset();
        sim.h(&qid(0)); // |+> superposition
        let result = sim.mz(&qid(0));
        if result[0].outcome {
            count_one += 1;
        }
    }

    // Expect approximately 50/50 split. Allow wide margin (40%-60%) for statistical noise.
    assert!(
        count_one > 400 && count_one < 600,
        "measurement statistics: expected ~500 ones out of {n}, got {count_one}"
    );
}

// ============================================================================
// Y Gate Behavior
// ============================================================================

/// Verify explicit Y gate behavior on computational and non-computational bases.
pub fn verify_y_gate_behavior<S: CliffordGateable>(sim: &mut S) {
    // Y|0> = i|1>, which measures as 1
    sim.reset();
    sim.y(&qid(0));
    assert_mz(sim, 0, true, "Y|0> should measure 1");

    // Y|1> = -i|0>, which measures as 0
    sim.reset();
    sim.x(&qid(0));
    sim.y(&qid(0));
    assert_mz(sim, 0, false, "Y|1> should measure 0");

    // Y maps +X to -X (since YXY† = -X)
    // Y|+> measured in X basis: should give outcome true (|->) since Y flips X eigenvalue
    sim.reset();
    sim.h(&qid(0)); // |+>
    sim.y(&qid(0));
    assert_mx(sim, 0, true, "Y|+> should be |-> in X basis");
}

// ============================================================================
// Controlled Gate Truth Tables
// ============================================================================

/// Verify the CX truth table on all 4 computational basis states.
pub fn verify_cx_truth_table<S: CliffordGateable>(sim: &mut S) {
    // CX: |00> -> |00>, |01> -> |01>, |10> -> |11>, |11> -> |10>
    let truth_table: &[(bool, bool, bool, bool)] = &[
        // (ctrl_in, tgt_in, ctrl_out, tgt_out)
        (false, false, false, false),
        (false, true, false, true),
        (true, false, true, true),
        (true, true, true, false),
    ];

    for &(ctrl_in, tgt_in, ctrl_out, tgt_out) in truth_table {
        sim.reset();
        if ctrl_in {
            sim.x(&qid(0));
        }
        if tgt_in {
            sim.x(&qid(1));
        }
        sim.cx(&qid2(0, 1));

        let label = format!("CX|{}{}>", u8::from(ctrl_in), u8::from(tgt_in));
        assert_mz(sim, 0, ctrl_out, &format!("{label} ctrl"));
        assert_mz(sim, 1, tgt_out, &format!("{label} tgt"));
    }
}

/// Verify the CY truth table on all 4 computational basis states.
pub fn verify_cy_truth_table<S: CliffordGateable>(sim: &mut S) {
    // CY: control=0 does nothing, control=1 applies Y to target
    // Y|0> = i|1> (measures 1), Y|1> = -i|0> (measures 0)
    let truth_table: &[(bool, bool, bool, bool)] = &[
        (false, false, false, false),
        (false, true, false, true),
        (true, false, true, true), // CY|10> -> ctrl=1, Y|0>=i|1>
        (true, true, true, false), // CY|11> -> ctrl=1, Y|1>=-i|0>
    ];

    for &(ctrl_in, tgt_in, ctrl_out, tgt_out) in truth_table {
        sim.reset();
        if ctrl_in {
            sim.x(&qid(0));
        }
        if tgt_in {
            sim.x(&qid(1));
        }
        sim.cy(&qid2(0, 1));

        let label = format!("CY|{}{}>", u8::from(ctrl_in), u8::from(tgt_in));
        assert_mz(sim, 0, ctrl_out, &format!("{label} ctrl"));
        assert_mz(sim, 1, tgt_out, &format!("{label} tgt"));
    }
}

/// Verify CZ truth table. CZ adds a phase flip on |11> -- invisible to mz,
/// so we test via X-basis measurement on H-prepared states.
pub fn verify_cz_truth_table<S: CliffordGateable>(sim: &mut S) {
    // CZ on computational basis states: only |11> gets a phase flip.
    // Since phase is invisible to mz, test that CZ doesn't change Z-basis outcomes:
    let basis_states: &[(bool, bool)] =
        &[(false, false), (false, true), (true, false), (true, true)];

    for &(q0_one, q1_one) in basis_states {
        sim.reset();
        if q0_one {
            sim.x(&qid(0));
        }
        if q1_one {
            sim.x(&qid(1));
        }
        sim.cz(&qid2(0, 1));

        let label = format!("CZ|{}{}>", u8::from(q0_one), u8::from(q1_one));
        assert_mz(sim, 0, q0_one, &format!("{label} q0 unchanged"));
        assert_mz(sim, 1, q1_one, &format!("{label} q1 unchanged"));
    }

    // Test the phase effect: CZ on |+1> should flip |+> to |-> on q0
    // because CZ|+1> = CZ(|01>+|11>)/sqrt(2) = (|01>-|11>)/sqrt(2) = |->|1>
    sim.reset();
    sim.h(&qid(0)); // |+>
    sim.x(&qid(1)); // |1>
    sim.cz(&qid2(0, 1));
    assert_mx(sim, 0, true, "CZ|+1> q0 should be |->");
    assert_mz(sim, 1, true, "CZ|+1> q1 should be |1>");

    // CZ on |+0> should NOT flip: CZ|+0> = (|00>+|10>)/sqrt(2) = |+>|0>
    sim.reset();
    sim.h(&qid(0)); // |+>
    // q1 stays |0>
    sim.cz(&qid2(0, 1));
    assert_mx(sim, 0, false, "CZ|+0> q0 should still be |+>");
    assert_mz(sim, 1, false, "CZ|+0> q1 should be |0>");
}

// ============================================================================
// iSWAP Identity
// ============================================================================

/// Verify iSWAP^2 behavior via its effect on entangled states.
///
/// iSWAP^2 acts as -SWAP on the |01>,|10> subspace (global phase on each component),
/// but global phase is invisible. We test that iSWAP^4 = I on all basis states.
pub fn verify_iswap_fourth<S: CliffordGateable>(sim: &mut S) {
    let basis_states: &[(bool, bool)] =
        &[(false, false), (false, true), (true, false), (true, true)];

    for &(q0_one, q1_one) in basis_states {
        sim.reset();
        if q0_one {
            sim.x(&qid(0));
        }
        if q1_one {
            sim.x(&qid(1));
        }
        sim.iswap(&qid2(0, 1))
            .iswap(&qid2(0, 1))
            .iswap(&qid2(0, 1))
            .iswap(&qid2(0, 1));

        let label = format!("iSWAP^4|{}{}>", u8::from(q0_one), u8::from(q1_one));
        assert_mz(sim, 0, q0_one, &format!("{label} q0"));
        assert_mz(sim, 1, q1_one, &format!("{label} q1"));
    }

    // Also test on |+0> to cover superposition
    sim.reset();
    sim.h(&qid(0));
    sim.iswap(&qid2(0, 1))
        .iswap(&qid2(0, 1))
        .iswap(&qid2(0, 1))
        .iswap(&qid2(0, 1));
    assert_mx(sim, 0, false, "iSWAP^4|+0> q0 should be |+>");
    assert_mz(sim, 1, false, "iSWAP^4|+0> q1 should be |0>");
}

// ============================================================================
// Additional Decompositions
// ============================================================================

/// Verify Y = iZX (global phase invisible to measurement) on computational basis.
pub fn verify_y_from_zx<S: CliffordGateable>(sim: &mut S) {
    // Y|0> = i|1>, ZX|0> = Z|1> = -|1>. Same measurement outcome (1).
    sim.reset();
    sim.y(&qid(0));
    assert_mz(sim, 0, true, "Y|0> should measure 1");

    sim.reset();
    sim.x(&qid(0));
    sim.z(&qid(0));
    assert_mz(sim, 0, true, "ZX|0> should measure 1");

    // Y|1> = -i|0>, ZX|1> = Z|0> = |0>. Same measurement outcome (0).
    sim.reset();
    sim.x(&qid(0));
    sim.y(&qid(0));
    assert_mz(sim, 0, false, "Y|1> should measure 0");

    sim.reset();
    sim.x(&qid(0));
    sim.x(&qid(0));
    sim.z(&qid(0));
    assert_mz(sim, 0, false, "ZX|1> should measure 0");

    // Test that (ZX)^2 = Y^2 = I (both are involutions up to global phase)
    sim.reset();
    sim.h(&qid(0)); // |+>
    sim.z(&qid(0));
    sim.x(&qid(0));
    sim.z(&qid(0));
    sim.x(&qid(0));
    assert_mx(sim, 0, false, "(ZX)^2|+> should be |+>");
}

// ============================================================================
// Aggregator
// ============================================================================

/// Run all measurement-based Clifford gate tests.
///
/// These tests verify gate contracts using only measurement outcomes, making them
/// suitable for any simulator that implements `CliffordGateable`.
///
/// # Arguments
/// * `sim` - A mutable reference to any `CliffordGateable` simulator
/// * `num_qubits` - Number of qubits available (determines which tests run)
pub fn run_clifford_gate_tests<S: CliffordGateable>(sim: &mut S, num_qubits: usize) {
    assert!(num_qubits >= 1, "Need at least 1 qubit");

    // -- Initial state --
    verify_initial_state(sim, num_qubits);

    // -- Identity gate --
    verify_identity_gate(sim);

    // -- Basic gate behavior --
    verify_x_gate(sim);
    verify_y_gate_behavior(sim);
    verify_h_superposition(sim);
    verify_z_on_zero(sim);
    verify_z_on_one(sim);
    verify_sx_superposition(sim);
    verify_sy_superposition(sim);
    verify_sz_maps_plus_to_plus_y(sim);
    verify_szdg_maps_plus_to_minus_y(sim);

    // -- Single-qubit identities --
    verify_h_squared(sim);
    verify_x_squared(sim);
    verify_y_squared(sim);
    verify_z_squared(sim);
    verify_s_fourth(sim);
    verify_s_squared_is_z(sim);
    verify_sx_squared_is_x(sim);
    verify_sy_squared_is_y(sim);

    // -- Single-qubit adjoint pairs --
    verify_sz_adjoint(sim);
    verify_sx_adjoint(sim);
    verify_sy_adjoint(sim);
    verify_f_adjoint(sim);
    verify_f2_adjoint(sim);
    verify_f3_adjoint(sim);
    verify_f4_adjoint(sim);

    // -- Hadamard variants --
    verify_hadamard_variant_involutions(sim);

    // -- Face gates --
    verify_face_gate_cube(sim);
    verify_face_gate_axis_rotation(sim);

    // -- Preparation gates --
    verify_px_prepares_plus(sim);
    verify_pnx_prepares_minus(sim);
    verify_py_prepares_plus_y(sim);
    verify_pny_prepares_minus_y(sim);
    verify_pz_prepares_zero(sim);
    verify_pnz_prepares_one(sim);

    // -- Measure-and-prepare gates --
    verify_mpx_prepares_plus(sim);
    verify_mpnx_prepares_minus(sim);
    verify_mpy_prepares_plus_y(sim);
    verify_mpny_prepares_minus_y(sim);
    verify_mpz_prepares_zero(sim);
    verify_mpnz_prepares_one(sim);

    // -- Negative measurement gates --
    verify_mnx_on_eigenstates(sim);
    verify_mny_on_eigenstates(sim);
    verify_mnz_on_eigenstates(sim);

    // -- Measurement properties --
    verify_measurement_idempotence(sim);
    verify_measurement_then_gate(sim);
    verify_measurement_statistics(sim);

    // -- Decompositions (single-qubit) --
    verify_y_from_zx(sim);

    if num_qubits >= 2 {
        // -- Two-qubit gate behavior --
        verify_cx_behavior(sim);
        verify_cx_squared(sim);
        verify_cy_behavior(sim);
        verify_cy_squared(sim);
        verify_cz_behavior(sim);
        verify_swap_behavior(sim);
        verify_swap_squared(sim);
        verify_swap_symmetric(sim);
        verify_iswap_behavior(sim);
        verify_iswap_fourth(sim);

        // -- Controlled gate truth tables --
        verify_cx_truth_table(sim);
        verify_cy_truth_table(sim);
        verify_cz_truth_table(sim);

        // -- G gate --
        verify_g_squared(sim);
        verify_g_gate_creates_superposition(sim);

        // -- Two-qubit adjoint pairs --
        verify_sxx_adjoint(sim);
        verify_syy_adjoint(sim);
        verify_szz_adjoint(sim);

        // -- Entanglement --
        verify_bell_state_correlations(sim);
        verify_measurement_idempotence_entangled(sim);

        // -- Gate decompositions --
        verify_swap_decomposition(sim);
        verify_cz_decomposition(sim);
        verify_x_hzh_decomposition(sim);
        verify_z_from_s_squared(sim);

        // -- Commutativity --
        verify_different_qubit_commutativity(sim);
        verify_same_qubit_non_commutativity(sim);
    }

    if num_qubits >= 3 {
        verify_ghz_correlations(sim, num_qubits.min(5));
    }
}

// ============================================================================
// Test Suite Macro
// ============================================================================

/// Generates a measurement-based Clifford test suite for any `CliffordGateable` simulator.
///
/// This macro creates a single test that runs all shared Clifford gate tests using
/// only measurement outcomes. It works for any simulator type.
///
/// # Arguments
///
/// * `$sim_type` - The simulator type
/// * `$num_qubits` - Number of qubits to use
/// * `$constructor` - Expression to create the simulator (receives `$num_qubits` as identifier)
///
/// # Example
///
/// ```ignore
/// use pecos_simulators::clifford_test_suite;
///
/// clifford_test_suite!(MySimType, 4, MySimType::new(num_qubits));
/// ```
#[macro_export]
macro_rules! clifford_test_suite {
    ($sim_type:ty, $num_qubits_val:expr, $constructor:expr) => {
        paste::paste! {
            #[test]
            fn [<test_ $sim_type:snake _clifford_suite>]() {
                let num_qubits = $num_qubits_val;
                let mut sim = $constructor;
                $crate::clifford_test_utils::run_clifford_gate_tests(&mut sim, num_qubits);
            }
        }
    };
}
