// Copyright 2026 The PECOS Developers
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

//! Verify that `StateVec` gate application matches matrix multiplication.
//!
//! For each gate, we apply it to |0...0> via the simulator, then compare the
//! resulting statevector against the gate's unitary matrix applied to |0...0>.

mod helpers;

use helpers::assert_states_equal;
use num_complex::Complex64;
use pecos_core::Angle64;
use pecos_core::clifford::Clifford;
use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, StateVec, qid, qid2};
use pecos_quantum::unitary_matrix::{ToMatrix, UnitaryMatrix};

type GateAction = (Clifford, Box<dyn Fn(&mut StateVec)>);
type NamedAction = (&'static str, Box<dyn Fn(&mut StateVec)>);
type NamedGatePair = (
    &'static str,
    Box<dyn Fn(&mut StateVec)>,
    Box<dyn Fn(&mut StateVec)>,
);

/// Applies a unitary matrix to |0...0> and returns the resulting state vector.
/// |0...0> is the first column of the matrix.
fn matrix_times_zero_state(mat: &UnitaryMatrix) -> Vec<Complex64> {
    let dim = mat.nrows();
    (0..dim).map(|r| mat[(r, 0)]).collect()
}

/// Applies a unitary matrix to an arbitrary state vector.
fn matrix_times_state(mat: &UnitaryMatrix, state: &[Complex64]) -> Vec<Complex64> {
    let dim = mat.nrows();
    (0..dim)
        .map(|r| (0..dim).map(|c| mat[(r, c)] * state[c]).sum::<Complex64>())
        .collect()
}

// ============================================================================
// 1-qubit Cliffords: simulator vs matrix
// ============================================================================

#[test]
fn sim_matches_matrix_1q_cliffords() {
    // Map each 1q Clifford to the corresponding StateVec method
    for &cliff in Clifford::all_1q() {
        let mat = cliff.to_matrix();
        let expected = matrix_times_zero_state(&mat);

        let mut sim = StateVec::new(1);
        apply_1q_clifford(&mut sim, cliff);
        let actual = sim.state();

        // Compare with tolerance, report gate name on failure
        let tolerance = 1e-10;
        let matches = if actual[0].norm() < tolerance && expected[0].norm() < tolerance {
            actual
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| (a.norm() - b.norm()).abs() < tolerance)
        } else if let Some((a, b)) = actual
            .iter()
            .zip(expected.iter())
            .find(|(a, b)| a.norm() > tolerance && b.norm() > tolerance)
        {
            let ratio = b / a;
            actual
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| (a * ratio - b).norm() < tolerance)
        } else {
            false
        };
        assert!(
            matches,
            "Simulator disagrees with matrix for 1q gate {cliff}"
        );
    }
}

fn apply_1q_clifford(sim: &mut StateVec, cliff: Clifford) {
    match cliff {
        Clifford::I => {}
        Clifford::X => {
            sim.x(&qid(0));
        }
        Clifford::Y => {
            sim.y(&qid(0));
        }
        Clifford::Z => {
            sim.z(&qid(0));
        }
        Clifford::H => {
            sim.h(&qid(0));
        }
        Clifford::SX => {
            sim.sx(&qid(0));
        }
        Clifford::SXdg => {
            sim.sxdg(&qid(0));
        }
        Clifford::SY => {
            sim.sy(&qid(0));
        }
        Clifford::SYdg => {
            sim.sydg(&qid(0));
        }
        Clifford::SZ => {
            sim.sz(&qid(0));
        }
        Clifford::SZdg => {
            sim.szdg(&qid(0));
        }
        Clifford::H2 => {
            sim.h2(&qid(0));
        }
        Clifford::H3 => {
            sim.h3(&qid(0));
        }
        Clifford::H4 => {
            sim.h4(&qid(0));
        }
        Clifford::H5 => {
            sim.h5(&qid(0));
        }
        Clifford::H6 => {
            sim.h6(&qid(0));
        }
        Clifford::F => {
            sim.f(&qid(0));
        }
        Clifford::Fdg => {
            sim.fdg(&qid(0));
        }
        Clifford::F2 => {
            sim.f2(&qid(0));
        }
        Clifford::F2dg => {
            sim.f2dg(&qid(0));
        }
        Clifford::F3 => {
            sim.f3(&qid(0));
        }
        Clifford::F3dg => {
            sim.f3dg(&qid(0));
        }
        Clifford::F4 => {
            sim.f4(&qid(0));
        }
        Clifford::F4dg => {
            sim.f4dg(&qid(0));
        }
        _ => panic!("unexpected 2q gate in 1q test"),
    }
}

// ============================================================================
// 2-qubit Cliffords: simulator vs matrix
// ============================================================================

#[test]
fn sim_matches_matrix_2q_cliffords() {
    // Gates that have direct CliffordGateable methods
    let gates: Vec<GateAction> = vec![
        (
            Clifford::CX,
            Box::new(|s: &mut StateVec| {
                s.cx(&qid2(0, 1));
            }),
        ),
        (
            Clifford::CY,
            Box::new(|s: &mut StateVec| {
                s.cy(&qid2(0, 1));
            }),
        ),
        (
            Clifford::CZ,
            Box::new(|s: &mut StateVec| {
                s.cz(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SWAP,
            Box::new(|s: &mut StateVec| {
                s.swap(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SXX,
            Box::new(|s: &mut StateVec| {
                s.sxx(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SXXdg,
            Box::new(|s: &mut StateVec| {
                s.sxxdg(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SYY,
            Box::new(|s: &mut StateVec| {
                s.syy(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SYYdg,
            Box::new(|s: &mut StateVec| {
                s.syydg(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SZZ,
            Box::new(|s: &mut StateVec| {
                s.szz(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SZZdg,
            Box::new(|s: &mut StateVec| {
                s.szzdg(&qid2(0, 1));
            }),
        ),
        (
            Clifford::ISWAP,
            Box::new(|s: &mut StateVec| {
                s.iswap(&qid2(0, 1));
            }),
        ),
        (
            Clifford::G,
            Box::new(|s: &mut StateVec| {
                s.g(&qid2(0, 1));
            }),
        ),
    ];

    for (cliff, apply_fn) in &gates {
        let mat = cliff.to_matrix();
        let expected = matrix_times_zero_state(&mat);

        let mut sim = StateVec::new(2);
        apply_fn(&mut sim);
        let actual = sim.state();

        // Report gate name on failure
        let tolerance = 1e-10;
        let matches = if let Some((a, b)) = actual
            .iter()
            .zip(expected.iter())
            .find(|(a, b)| a.norm() > tolerance && b.norm() > tolerance)
        {
            let ratio = b / a;
            actual
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| (a * ratio - b).norm() < tolerance)
        } else {
            actual
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| (a.norm() - b.norm()).abs() < tolerance)
        };
        assert!(
            matches,
            "Simulator disagrees with matrix for 2q gate {cliff}"
        );
    }
}

// ============================================================================
// 2-qubit Cliffords on non-trivial input state
// ============================================================================

#[test]
fn sim_matches_matrix_2q_on_superposition() {
    // Apply 2q gates to H|0> x H|0> = |++> and compare with matrix path.
    // First, get |++> from the simulator.
    let gates: Vec<GateAction> = vec![
        (
            Clifford::CX,
            Box::new(|s: &mut StateVec| {
                s.cx(&qid2(0, 1));
            }),
        ),
        (
            Clifford::CY,
            Box::new(|s: &mut StateVec| {
                s.cy(&qid2(0, 1));
            }),
        ),
        (
            Clifford::CZ,
            Box::new(|s: &mut StateVec| {
                s.cz(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SWAP,
            Box::new(|s: &mut StateVec| {
                s.swap(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SXX,
            Box::new(|s: &mut StateVec| {
                s.sxx(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SXXdg,
            Box::new(|s: &mut StateVec| {
                s.sxxdg(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SYY,
            Box::new(|s: &mut StateVec| {
                s.syy(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SYYdg,
            Box::new(|s: &mut StateVec| {
                s.syydg(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SZZ,
            Box::new(|s: &mut StateVec| {
                s.szz(&qid2(0, 1));
            }),
        ),
        (
            Clifford::SZZdg,
            Box::new(|s: &mut StateVec| {
                s.szzdg(&qid2(0, 1));
            }),
        ),
        (
            Clifford::ISWAP,
            Box::new(|s: &mut StateVec| {
                s.iswap(&qid2(0, 1));
            }),
        ),
        (
            Clifford::G,
            Box::new(|s: &mut StateVec| {
                s.g(&qid2(0, 1));
            }),
        ),
    ];

    // Get |++> state from simulator (ground truth)
    let plus_plus = {
        let mut sim = StateVec::new(2);
        sim.h(&qid(0));
        sim.h(&qid(1));
        sim.state()
    };

    for (cliff, apply_fn) in &gates {
        let mat = cliff.to_matrix();
        // Matrix path: gate * |++>
        let expected = matrix_times_state(&mat, &plus_plus);

        // Simulator path: H(0), H(1), gate(0,1)
        let mut sim = StateVec::new(2);
        sim.h(&qid(0));
        sim.h(&qid(1));
        apply_fn(&mut sim);
        let actual = sim.state();

        let tolerance = 1e-10;
        let matches = if let Some((a, b)) = actual
            .iter()
            .zip(expected.iter())
            .find(|(a, b)| a.norm() > tolerance && b.norm() > tolerance)
        {
            let ratio = b / a;
            actual
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| (a * ratio - b).norm() < tolerance)
        } else {
            actual
                .iter()
                .zip(expected.iter())
                .all(|(a, b)| (a.norm() - b.norm()).abs() < tolerance)
        };
        assert!(
            matches,
            "Simulator disagrees with matrix on |++> for 2q gate {cliff}"
        );
    }
}

// ============================================================================
// Rotation gates: simulator vs matrix
// ============================================================================

#[test]
fn sim_matches_matrix_1q_rotations() {
    use pecos_core::unitary_rep;

    let angles = [
        Angle64::ZERO,
        Angle64::QUARTER_TURN,
        Angle64::HALF_TURN,
        Angle64::THREE_QUARTERS_TURN,
        Angle64::from_radians(0.7),
        Angle64::from_radians(2.3),
    ];

    for &angle in &angles {
        // RX
        let mat = unitary_rep::RX(angle, 0).to_matrix();
        let expected = matrix_times_zero_state(&mat);
        let mut sim = StateVec::new(1);
        sim.rx(angle, &qid(0));
        assert_states_equal(sim.state(), &expected);

        // RY
        let mat = unitary_rep::RY(angle, 0).to_matrix();
        let expected = matrix_times_zero_state(&mat);
        let mut sim = StateVec::new(1);
        sim.ry(angle, &qid(0));
        assert_states_equal(sim.state(), &expected);

        // RZ
        let mat = unitary_rep::RZ(angle, 0).to_matrix();
        let expected = matrix_times_zero_state(&mat);
        let mut sim = StateVec::new(1);
        sim.rz(angle, &qid(0));
        assert_states_equal(sim.state(), &expected);
    }
}

#[test]
fn sim_matches_matrix_2q_rotations() {
    use pecos_core::unitary_rep;

    let angles = [
        Angle64::QUARTER_TURN,
        Angle64::HALF_TURN,
        Angle64::from_radians(1.1),
    ];

    for &angle in &angles {
        // RXX
        let mat = unitary_rep::RXX(angle, 0, 1).to_matrix();
        let expected = matrix_times_zero_state(&mat);
        let mut sim = StateVec::new(2);
        sim.rxx(angle, &qid2(0, 1));
        assert_states_equal(sim.state(), &expected);

        // RYY
        let mat = unitary_rep::RYY(angle, 0, 1).to_matrix();
        let expected = matrix_times_zero_state(&mat);
        let mut sim = StateVec::new(2);
        sim.ryy(angle, &qid2(0, 1));
        assert_states_equal(sim.state(), &expected);

        // RZZ
        let mat = unitary_rep::RZZ(angle, 0, 1).to_matrix();
        let expected = matrix_times_zero_state(&mat);
        let mut sim = StateVec::new(2);
        sim.rzz(angle, &qid2(0, 1));
        assert_states_equal(sim.state(), &expected);
    }
}

// ============================================================================
// Rotation gates on superposition states
// ============================================================================

#[test]
fn sim_matches_matrix_1q_rotations_on_plus() {
    use pecos_core::unitary_rep;

    let angles = [
        Angle64::QUARTER_TURN,
        Angle64::HALF_TURN,
        Angle64::THREE_QUARTERS_TURN,
        Angle64::from_radians(0.7),
        Angle64::from_radians(2.3),
    ];

    // Get |+> from simulator
    let plus_state = {
        let mut sim = StateVec::new(1);
        sim.h(&qid(0));
        sim.state()
    };

    for &angle in &angles {
        // RX on |+>
        let mat = unitary_rep::RX(angle, 0).to_matrix();
        let expected = matrix_times_state(&mat, &plus_state);
        let mut sim = StateVec::new(1);
        sim.h(&qid(0));
        sim.rx(angle, &qid(0));
        assert_states_equal(sim.state(), &expected);

        // RY on |+>
        let mat = unitary_rep::RY(angle, 0).to_matrix();
        let expected = matrix_times_state(&mat, &plus_state);
        let mut sim = StateVec::new(1);
        sim.h(&qid(0));
        sim.ry(angle, &qid(0));
        assert_states_equal(sim.state(), &expected);

        // RZ on |+>
        let mat = unitary_rep::RZ(angle, 0).to_matrix();
        let expected = matrix_times_state(&mat, &plus_state);
        let mut sim = StateVec::new(1);
        sim.h(&qid(0));
        sim.rz(angle, &qid(0));
        assert_states_equal(sim.state(), &expected);
    }
}

#[test]
fn sim_matches_matrix_2q_rotations_on_superposition() {
    use pecos_core::unitary_rep;

    let angles = [
        Angle64::QUARTER_TURN,
        Angle64::HALF_TURN,
        Angle64::from_radians(1.1),
        Angle64::from_radians(0.3),
    ];

    // Get |++> from simulator
    let plus_plus = {
        let mut sim = StateVec::new(2);
        sim.h(&qid(0));
        sim.h(&qid(1));
        sim.state()
    };

    for &angle in &angles {
        // RXX on |++>
        let mat = unitary_rep::RXX(angle, 0, 1).to_matrix();
        let expected = matrix_times_state(&mat, &plus_plus);
        let mut sim = StateVec::new(2);
        sim.h(&qid(0));
        sim.h(&qid(1));
        sim.rxx(angle, &qid2(0, 1));
        assert_states_equal(sim.state(), &expected);

        // RYY on |++>
        let mat = unitary_rep::RYY(angle, 0, 1).to_matrix();
        let expected = matrix_times_state(&mat, &plus_plus);
        let mut sim = StateVec::new(2);
        sim.h(&qid(0));
        sim.h(&qid(1));
        sim.ryy(angle, &qid2(0, 1));
        assert_states_equal(sim.state(), &expected);

        // RZZ on |++>
        let mat = unitary_rep::RZZ(angle, 0, 1).to_matrix();
        let expected = matrix_times_state(&mat, &plus_plus);
        let mut sim = StateVec::new(2);
        sim.h(&qid(0));
        sim.h(&qid(1));
        sim.rzz(angle, &qid2(0, 1));
        assert_states_equal(sim.state(), &expected);
    }
}

// ============================================================================
// SXX^2 = XX, SYY^2 = YY, SZZ^2 = ZZ at the simulator level
// ============================================================================

#[test]
fn sxx_squared_is_xx() {
    let input_states: Vec<NamedAction> = vec![
        ("|00>", Box::new(|_s: &mut StateVec| {})),
        (
            "|10>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(0));
            }),
        ),
        (
            "|01>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(1));
            }),
        ),
        (
            "|++>",
            Box::new(|s: &mut StateVec| {
                s.h(&qid(0));
                s.h(&qid(1));
            }),
        ),
    ];

    for (name, prepare) in &input_states {
        // Reference: prepare then X(0)X(1)
        let mut ref_sim = StateVec::new(2);
        prepare(&mut ref_sim);
        ref_sim.x(&qid(0));
        ref_sim.x(&qid(1));
        let ref_state = ref_sim.state();

        // Test: prepare then SXX twice
        let mut sim = StateVec::new(2);
        prepare(&mut sim);
        sim.sxx(&qid2(0, 1));
        sim.sxx(&qid2(0, 1));
        assert_states_equal(sim.state(), &ref_state);
        let _ = name;
    }
}

#[test]
fn syy_squared_is_yy() {
    let input_states: Vec<NamedAction> = vec![
        ("|00>", Box::new(|_s: &mut StateVec| {})),
        (
            "|10>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(0));
            }),
        ),
        (
            "|01>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(1));
            }),
        ),
        (
            "|++>",
            Box::new(|s: &mut StateVec| {
                s.h(&qid(0));
                s.h(&qid(1));
            }),
        ),
    ];

    for (name, prepare) in &input_states {
        // Reference: prepare then Y(0)Y(1)
        let mut ref_sim = StateVec::new(2);
        prepare(&mut ref_sim);
        ref_sim.y(&qid(0));
        ref_sim.y(&qid(1));
        let ref_state = ref_sim.state();

        // Test: prepare then SYY twice
        let mut sim = StateVec::new(2);
        prepare(&mut sim);
        sim.syy(&qid2(0, 1));
        sim.syy(&qid2(0, 1));
        assert_states_equal(sim.state(), &ref_state);
        let _ = name;
    }
}

#[test]
fn szz_squared_is_zz() {
    let input_states: Vec<NamedAction> = vec![
        ("|00>", Box::new(|_s: &mut StateVec| {})),
        (
            "|10>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(0));
            }),
        ),
        (
            "|01>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(1));
            }),
        ),
        (
            "|++>",
            Box::new(|s: &mut StateVec| {
                s.h(&qid(0));
                s.h(&qid(1));
            }),
        ),
    ];

    for (name, prepare) in &input_states {
        // Reference: prepare then Z(0)Z(1)
        let mut ref_sim = StateVec::new(2);
        prepare(&mut ref_sim);
        ref_sim.z(&qid(0));
        ref_sim.z(&qid(1));
        let ref_state = ref_sim.state();

        // Test: prepare then SZZ twice
        let mut sim = StateVec::new(2);
        prepare(&mut sim);
        sim.szz(&qid2(0, 1));
        sim.szz(&qid2(0, 1));
        assert_states_equal(sim.state(), &ref_state);
        let _ = name;
    }
}

// ============================================================================
// ISWAPdg and Gdg: verify via iswap*iswap = iswapdg*iswap = identity
// ============================================================================

#[test]
fn iswapdg_times_iswap_is_identity_matrix() {
    // ISWAPdg * ISWAP = I at the matrix level.
    // Note: iSWAP is NOT unitarily self-inverse (iSWAP^2 = ZZ), so there is no
    // simulator-level iswapdg() to test. We verify the matrix product here.
    let iswap_mat = Clifford::ISWAP.to_matrix();
    let iswapdg_mat = Clifford::ISWAPdg.to_matrix();
    let product = &iswapdg_mat * &iswap_mat;
    let identity = UnitaryMatrix::identity(4);
    let diff = (&product - &identity).norm();
    assert!(
        diff < 1e-10,
        "ISWAPdg * ISWAP should be identity, diff = {diff}"
    );

    // Also verify ISWAP * ISWAPdg = I
    let product2 = &iswap_mat * &iswapdg_mat;
    let diff2 = (&product2 - &identity).norm();
    assert!(
        diff2 < 1e-10,
        "ISWAP * ISWAPdg should be identity, diff = {diff2}"
    );
}

#[test]
fn iswap_squared_is_zz() {
    // iSWAP^2 = Z tensor Z (not identity). Verify at simulator level.
    let input_states: Vec<NamedAction> = vec![
        ("|00>", Box::new(|_s: &mut StateVec| {})),
        (
            "|10>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(0));
            }),
        ),
        (
            "|01>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(1));
            }),
        ),
        (
            "|++>",
            Box::new(|s: &mut StateVec| {
                s.h(&qid(0));
                s.h(&qid(1));
            }),
        ),
    ];

    for (name, prepare) in &input_states {
        // Reference: prepare state then apply Z(0)Z(1)
        let mut ref_sim = StateVec::new(2);
        prepare(&mut ref_sim);
        ref_sim.z(&qid(0));
        ref_sim.z(&qid(1));
        let ref_state = ref_sim.state();

        // Test: prepare state then apply iswap twice
        let mut sim = StateVec::new(2);
        prepare(&mut sim);
        sim.iswap(&qid2(0, 1));
        sim.iswap(&qid2(0, 1));
        assert_states_equal(sim.state(), &ref_state);
        let _ = name;
    }
}

#[test]
fn g_squared_returns_to_original_state() {
    // G is self-inverse: G * G = I. Test on several input states in the simulator.
    let input_states: Vec<NamedAction> = vec![
        ("|00>", Box::new(|_s: &mut StateVec| {})),
        (
            "|10>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(0));
            }),
        ),
        (
            "|01>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(1));
            }),
        ),
        (
            "|++>",
            Box::new(|s: &mut StateVec| {
                s.h(&qid(0));
                s.h(&qid(1));
            }),
        ),
    ];

    for (name, prepare) in &input_states {
        let mut ref_sim = StateVec::new(2);
        prepare(&mut ref_sim);
        let ref_state = ref_sim.state();

        let mut sim = StateVec::new(2);
        prepare(&mut sim);
        sim.g(&qid2(0, 1));
        sim.g(&qid2(0, 1));
        assert_states_equal(sim.state(), &ref_state);

        // Also verify Gdg matrix * G matrix = I
        let g_mat = Clifford::G.to_matrix();
        let gdg_mat = Clifford::Gdg.to_matrix();
        let product = &gdg_mat * &g_mat;
        let identity = UnitaryMatrix::identity(4);
        let diff = (&product - &identity).norm();
        assert!(
            diff < 1e-10,
            "Gdg * G should be identity on {name}, diff = {diff}"
        );
    }
}

// ============================================================================
// Gate-then-dagger identity: apply gate then its inverse, verify state unchanged
// ============================================================================

#[test]
fn gate_then_dagger_1q_identity() {
    // For each 1q gate that has a dagger, apply gate then dagger and check state unchanged.
    let pairs: Vec<NamedGatePair> = vec![
        (
            "SX/SXdg",
            Box::new(|s: &mut StateVec| {
                s.sx(&qid(0));
            }),
            Box::new(|s: &mut StateVec| {
                s.sxdg(&qid(0));
            }),
        ),
        (
            "SY/SYdg",
            Box::new(|s: &mut StateVec| {
                s.sy(&qid(0));
            }),
            Box::new(|s: &mut StateVec| {
                s.sydg(&qid(0));
            }),
        ),
        (
            "SZ/SZdg",
            Box::new(|s: &mut StateVec| {
                s.sz(&qid(0));
            }),
            Box::new(|s: &mut StateVec| {
                s.szdg(&qid(0));
            }),
        ),
        (
            "F/Fdg",
            Box::new(|s: &mut StateVec| {
                s.f(&qid(0));
            }),
            Box::new(|s: &mut StateVec| {
                s.fdg(&qid(0));
            }),
        ),
        (
            "F2/F2dg",
            Box::new(|s: &mut StateVec| {
                s.f2(&qid(0));
            }),
            Box::new(|s: &mut StateVec| {
                s.f2dg(&qid(0));
            }),
        ),
        (
            "F3/F3dg",
            Box::new(|s: &mut StateVec| {
                s.f3(&qid(0));
            }),
            Box::new(|s: &mut StateVec| {
                s.f3dg(&qid(0));
            }),
        ),
        (
            "F4/F4dg",
            Box::new(|s: &mut StateVec| {
                s.f4(&qid(0));
            }),
            Box::new(|s: &mut StateVec| {
                s.f4dg(&qid(0));
            }),
        ),
    ];

    // Self-inverse 1q gates (H variants are all self-inverse)
    let self_inverse: Vec<NamedAction> = vec![
        (
            "H",
            Box::new(|s: &mut StateVec| {
                s.h(&qid(0));
            }),
        ),
        (
            "X",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(0));
            }),
        ),
        (
            "Y",
            Box::new(|s: &mut StateVec| {
                s.y(&qid(0));
            }),
        ),
        (
            "Z",
            Box::new(|s: &mut StateVec| {
                s.z(&qid(0));
            }),
        ),
        (
            "H2",
            Box::new(|s: &mut StateVec| {
                s.h2(&qid(0));
            }),
        ),
        (
            "H3",
            Box::new(|s: &mut StateVec| {
                s.h3(&qid(0));
            }),
        ),
        (
            "H4",
            Box::new(|s: &mut StateVec| {
                s.h4(&qid(0));
            }),
        ),
        (
            "H5",
            Box::new(|s: &mut StateVec| {
                s.h5(&qid(0));
            }),
        ),
        (
            "H6",
            Box::new(|s: &mut StateVec| {
                s.h6(&qid(0));
            }),
        ),
    ];

    let input_states: Vec<NamedAction> = vec![
        ("|0>", Box::new(|_s: &mut StateVec| {})),
        (
            "|1>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(0));
            }),
        ),
        (
            "|+>",
            Box::new(|s: &mut StateVec| {
                s.h(&qid(0));
            }),
        ),
    ];

    for (input_name, prepare) in &input_states {
        let mut ref_sim = StateVec::new(1);
        prepare(&mut ref_sim);
        let ref_state = ref_sim.state();

        for (name, gate, dagger) in &pairs {
            let mut sim = StateVec::new(1);
            prepare(&mut sim);
            gate(&mut sim);
            dagger(&mut sim);
            assert_states_equal(sim.state(), &ref_state);
            // Suppress unused variable warning with explicit naming in assertion
            let _ = (name, input_name);
        }

        for (name, gate) in &self_inverse {
            let mut sim = StateVec::new(1);
            prepare(&mut sim);
            gate(&mut sim);
            gate(&mut sim);
            assert_states_equal(sim.state(), &ref_state);
            let _ = (name, input_name);
        }
    }
}

#[test]
fn gate_then_dagger_2q_identity() {
    // For each 2q gate with a dagger, apply gate then dagger and verify state unchanged.
    let pairs: Vec<NamedGatePair> = vec![
        (
            "SXX/SXXdg",
            Box::new(|s: &mut StateVec| {
                s.sxx(&qid2(0, 1));
            }),
            Box::new(|s: &mut StateVec| {
                s.sxxdg(&qid2(0, 1));
            }),
        ),
        (
            "SYY/SYYdg",
            Box::new(|s: &mut StateVec| {
                s.syy(&qid2(0, 1));
            }),
            Box::new(|s: &mut StateVec| {
                s.syydg(&qid2(0, 1));
            }),
        ),
        (
            "SZZ/SZZdg",
            Box::new(|s: &mut StateVec| {
                s.szz(&qid2(0, 1));
            }),
            Box::new(|s: &mut StateVec| {
                s.szzdg(&qid2(0, 1));
            }),
        ),
        (
            "ISWAP/ISWAPdg",
            Box::new(|s: &mut StateVec| {
                s.iswap(&qid2(0, 1));
            }),
            Box::new(|s: &mut StateVec| {
                s.iswapdg(&qid2(0, 1));
            }),
        ),
    ];

    let self_inverse: Vec<NamedAction> = vec![
        (
            "CX",
            Box::new(|s: &mut StateVec| {
                s.cx(&qid2(0, 1));
            }),
        ),
        (
            "CY",
            Box::new(|s: &mut StateVec| {
                s.cy(&qid2(0, 1));
            }),
        ),
        (
            "CZ",
            Box::new(|s: &mut StateVec| {
                s.cz(&qid2(0, 1));
            }),
        ),
        (
            "SWAP",
            Box::new(|s: &mut StateVec| {
                s.swap(&qid2(0, 1));
            }),
        ),
        (
            "G",
            Box::new(|s: &mut StateVec| {
                s.g(&qid2(0, 1));
            }),
        ),
    ];

    let input_states: Vec<NamedAction> = vec![
        ("|00>", Box::new(|_s: &mut StateVec| {})),
        (
            "|10>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(0));
            }),
        ),
        (
            "|01>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(1));
            }),
        ),
        (
            "|++>",
            Box::new(|s: &mut StateVec| {
                s.h(&qid(0));
                s.h(&qid(1));
            }),
        ),
    ];

    for (input_name, prepare) in &input_states {
        let mut ref_sim = StateVec::new(2);
        prepare(&mut ref_sim);
        let ref_state = ref_sim.state();

        for (name, gate, dagger) in &pairs {
            let mut sim = StateVec::new(2);
            prepare(&mut sim);
            gate(&mut sim);
            dagger(&mut sim);
            assert_states_equal(sim.state(), &ref_state);
            let _ = (name, input_name);
        }

        for (name, gate) in &self_inverse {
            let mut sim = StateVec::new(2);
            prepare(&mut sim);
            gate(&mut sim);
            gate(&mut sim);
            assert_states_equal(sim.state(), &ref_state);
            let _ = (name, input_name);
        }
    }
}

// ============================================================================
// Non-adjacent qubit tests: 2q gates on qubits (0, 2) in a 3-qubit system
// ============================================================================

/// Applies a unitary matrix to an arbitrary state vector.
fn matrix_times_state_3q(mat: &UnitaryMatrix, state: &[Complex64]) -> Vec<Complex64> {
    let dim = mat.nrows();
    assert_eq!(dim, state.len());
    (0..dim)
        .map(|r| (0..dim).map(|c| mat[(r, c)] * state[c]).sum::<Complex64>())
        .collect()
}

#[test]
fn sim_matches_matrix_2q_nonadjacent_on_zero_state() {
    // Apply 2q gates to qubits (0, 2) in a 3-qubit register starting from |000>.
    // Compare simulator vs matrix multiplication.
    use pecos_quantum::unitary_matrix::to_matrix;

    let gates: Vec<GateAction> = vec![
        (
            Clifford::CX,
            Box::new(|s: &mut StateVec| {
                s.cx(&qid2(0, 2));
            }),
        ),
        (
            Clifford::CY,
            Box::new(|s: &mut StateVec| {
                s.cy(&qid2(0, 2));
            }),
        ),
        (
            Clifford::CZ,
            Box::new(|s: &mut StateVec| {
                s.cz(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SWAP,
            Box::new(|s: &mut StateVec| {
                s.swap(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SXX,
            Box::new(|s: &mut StateVec| {
                s.sxx(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SXXdg,
            Box::new(|s: &mut StateVec| {
                s.sxxdg(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SYY,
            Box::new(|s: &mut StateVec| {
                s.syy(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SYYdg,
            Box::new(|s: &mut StateVec| {
                s.syydg(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SZZ,
            Box::new(|s: &mut StateVec| {
                s.szz(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SZZdg,
            Box::new(|s: &mut StateVec| {
                s.szzdg(&qid2(0, 2));
            }),
        ),
        (
            Clifford::ISWAP,
            Box::new(|s: &mut StateVec| {
                s.iswap(&qid2(0, 2));
            }),
        ),
        (
            Clifford::ISWAPdg,
            Box::new(|s: &mut StateVec| {
                s.iswapdg(&qid2(0, 2));
            }),
        ),
        (
            Clifford::G,
            Box::new(|s: &mut StateVec| {
                s.g(&qid2(0, 2));
            }),
        ),
        (
            Clifford::Gdg,
            Box::new(|s: &mut StateVec| {
                s.gdg(&qid2(0, 2));
            }),
        ),
    ];

    for (cliff, apply_fn) in &gates {
        let ur = cliff.to_unitary_rep_on_qubits(0usize, 2usize);
        let mat = to_matrix(&ur);
        assert_eq!(
            mat.nrows(),
            8,
            "Non-adjacent gate should produce 8x8 matrix"
        );
        let expected: Vec<Complex64> = (0..8).map(|r| mat[(r, 0)]).collect();

        let mut sim = StateVec::new(3);
        apply_fn(&mut sim);
        let actual = sim.state();

        assert_states_equal(actual, &expected);
    }
}

#[test]
fn sim_matches_matrix_2q_nonadjacent_on_superposition() {
    // Apply 2q gates to qubits (0, 2) starting from H(0)H(2)|000>.
    // Qubit 1 stays |0> throughout.
    use pecos_quantum::unitary_matrix::to_matrix;

    let gates: Vec<GateAction> = vec![
        (
            Clifford::CX,
            Box::new(|s: &mut StateVec| {
                s.cx(&qid2(0, 2));
            }),
        ),
        (
            Clifford::CY,
            Box::new(|s: &mut StateVec| {
                s.cy(&qid2(0, 2));
            }),
        ),
        (
            Clifford::CZ,
            Box::new(|s: &mut StateVec| {
                s.cz(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SWAP,
            Box::new(|s: &mut StateVec| {
                s.swap(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SXX,
            Box::new(|s: &mut StateVec| {
                s.sxx(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SXXdg,
            Box::new(|s: &mut StateVec| {
                s.sxxdg(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SYY,
            Box::new(|s: &mut StateVec| {
                s.syy(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SYYdg,
            Box::new(|s: &mut StateVec| {
                s.syydg(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SZZ,
            Box::new(|s: &mut StateVec| {
                s.szz(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SZZdg,
            Box::new(|s: &mut StateVec| {
                s.szzdg(&qid2(0, 2));
            }),
        ),
        (
            Clifford::ISWAP,
            Box::new(|s: &mut StateVec| {
                s.iswap(&qid2(0, 2));
            }),
        ),
        (
            Clifford::ISWAPdg,
            Box::new(|s: &mut StateVec| {
                s.iswapdg(&qid2(0, 2));
            }),
        ),
        (
            Clifford::G,
            Box::new(|s: &mut StateVec| {
                s.g(&qid2(0, 2));
            }),
        ),
        (
            Clifford::Gdg,
            Box::new(|s: &mut StateVec| {
                s.gdg(&qid2(0, 2));
            }),
        ),
    ];

    // Prepare H(0) H(2) |000> via simulator
    let input_state = {
        let mut sim = StateVec::new(3);
        sim.h(&qid(0));
        sim.h(&qid(2));
        sim.state()
    };

    for (cliff, apply_fn) in &gates {
        let ur = cliff.to_unitary_rep_on_qubits(0usize, 2usize);
        let mat = to_matrix(&ur);
        let expected = matrix_times_state_3q(&mat, &input_state);

        let mut sim = StateVec::new(3);
        sim.h(&qid(0));
        sim.h(&qid(2));
        apply_fn(&mut sim);
        let actual = sim.state();

        assert_states_equal(actual, &expected);
    }
}

#[test]
fn sim_matches_matrix_2q_nonadjacent_with_entangled_spectator() {
    // Entangle qubit 1 with qubit 0 first (CX(0,1)), then apply 2q gate on (0, 2).
    // This tests non-adjacent gates when the "spectator" qubit 1 is entangled.
    use pecos_quantum::unitary_matrix::to_matrix;

    let gates: Vec<GateAction> = vec![
        (
            Clifford::CX,
            Box::new(|s: &mut StateVec| {
                s.cx(&qid2(0, 2));
            }),
        ),
        (
            Clifford::CY,
            Box::new(|s: &mut StateVec| {
                s.cy(&qid2(0, 2));
            }),
        ),
        (
            Clifford::CZ,
            Box::new(|s: &mut StateVec| {
                s.cz(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SWAP,
            Box::new(|s: &mut StateVec| {
                s.swap(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SXX,
            Box::new(|s: &mut StateVec| {
                s.sxx(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SXXdg,
            Box::new(|s: &mut StateVec| {
                s.sxxdg(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SYY,
            Box::new(|s: &mut StateVec| {
                s.syy(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SYYdg,
            Box::new(|s: &mut StateVec| {
                s.syydg(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SZZ,
            Box::new(|s: &mut StateVec| {
                s.szz(&qid2(0, 2));
            }),
        ),
        (
            Clifford::SZZdg,
            Box::new(|s: &mut StateVec| {
                s.szzdg(&qid2(0, 2));
            }),
        ),
        (
            Clifford::ISWAP,
            Box::new(|s: &mut StateVec| {
                s.iswap(&qid2(0, 2));
            }),
        ),
        (
            Clifford::ISWAPdg,
            Box::new(|s: &mut StateVec| {
                s.iswapdg(&qid2(0, 2));
            }),
        ),
        (
            Clifford::G,
            Box::new(|s: &mut StateVec| {
                s.g(&qid2(0, 2));
            }),
        ),
        (
            Clifford::Gdg,
            Box::new(|s: &mut StateVec| {
                s.gdg(&qid2(0, 2));
            }),
        ),
    ];

    // Prepare H(0) then CX(0,1) |000> = (|000> + |110>) / sqrt(2)
    let input_state = {
        let mut sim = StateVec::new(3);
        sim.h(&qid(0));
        sim.cx(&qid2(0, 1));
        sim.state()
    };

    for (cliff, apply_fn) in &gates {
        let ur = cliff.to_unitary_rep_on_qubits(0usize, 2usize);
        let mat = to_matrix(&ur);
        let expected = matrix_times_state_3q(&mat, &input_state);

        let mut sim = StateVec::new(3);
        sim.h(&qid(0));
        sim.cx(&qid2(0, 1));
        apply_fn(&mut sim);
        let actual = sim.state();

        assert_states_equal(actual, &expected);
    }
}

#[test]
fn gate_then_dagger_2q_nonadjacent_identity() {
    // Apply gate(0,2) then dagger(0,2) on a 3-qubit system. State should be unchanged.
    let pairs: Vec<NamedGatePair> = vec![
        (
            "SXX/SXXdg",
            Box::new(|s: &mut StateVec| {
                s.sxx(&qid2(0, 2));
            }),
            Box::new(|s: &mut StateVec| {
                s.sxxdg(&qid2(0, 2));
            }),
        ),
        (
            "SYY/SYYdg",
            Box::new(|s: &mut StateVec| {
                s.syy(&qid2(0, 2));
            }),
            Box::new(|s: &mut StateVec| {
                s.syydg(&qid2(0, 2));
            }),
        ),
        (
            "SZZ/SZZdg",
            Box::new(|s: &mut StateVec| {
                s.szz(&qid2(0, 2));
            }),
            Box::new(|s: &mut StateVec| {
                s.szzdg(&qid2(0, 2));
            }),
        ),
        (
            "ISWAP/ISWAPdg",
            Box::new(|s: &mut StateVec| {
                s.iswap(&qid2(0, 2));
            }),
            Box::new(|s: &mut StateVec| {
                s.iswapdg(&qid2(0, 2));
            }),
        ),
    ];

    let self_inverse: Vec<NamedAction> = vec![
        (
            "CX",
            Box::new(|s: &mut StateVec| {
                s.cx(&qid2(0, 2));
            }),
        ),
        (
            "CZ",
            Box::new(|s: &mut StateVec| {
                s.cz(&qid2(0, 2));
            }),
        ),
        (
            "SWAP",
            Box::new(|s: &mut StateVec| {
                s.swap(&qid2(0, 2));
            }),
        ),
        (
            "G",
            Box::new(|s: &mut StateVec| {
                s.g(&qid2(0, 2));
            }),
        ),
    ];

    let input_states: Vec<NamedAction> = vec![
        ("|000>", Box::new(|_s: &mut StateVec| {})),
        (
            "|100>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(0));
            }),
        ),
        (
            "|001>",
            Box::new(|s: &mut StateVec| {
                s.x(&qid(2));
            }),
        ),
        (
            "H(0)H(2)|000>",
            Box::new(|s: &mut StateVec| {
                s.h(&qid(0));
                s.h(&qid(2));
            }),
        ),
        (
            "Bell(0,1)",
            Box::new(|s: &mut StateVec| {
                s.h(&qid(0));
                s.cx(&qid2(0, 1));
            }),
        ),
    ];

    for (input_name, prepare) in &input_states {
        let mut ref_sim = StateVec::new(3);
        prepare(&mut ref_sim);
        let ref_state = ref_sim.state();

        for (name, gate, dagger) in &pairs {
            let mut sim = StateVec::new(3);
            prepare(&mut sim);
            gate(&mut sim);
            dagger(&mut sim);
            assert_states_equal(sim.state(), &ref_state);
            let _ = (name, input_name);
        }

        for (name, gate) in &self_inverse {
            let mut sim = StateVec::new(3);
            prepare(&mut sim);
            gate(&mut sim);
            gate(&mut sim);
            assert_states_equal(sim.state(), &ref_state);
            let _ = (name, input_name);
        }
    }
}
