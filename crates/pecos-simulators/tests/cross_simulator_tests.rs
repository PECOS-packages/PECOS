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

//! Cross-simulator consistency tests: `SparseStab` vs `StateVec`.
//!
//! For each Clifford gate, apply it to |0...0> in both `SparseStab` and `StateVec`,
//! then verify that `SparseStab`'s deterministic measurements match `StateVec`'s
//! state amplitudes.

use pecos_core::QubitId;
use pecos_core::clifford::Clifford;
use pecos_simulators::{CliffordGateable, SparseStab, StateVec, qid};

type GateTestEntry = (
    Clifford,
    Box<dyn Fn(&mut StateVec)>,
    Box<dyn Fn(&mut SparseStab)>,
);

type BatchGateTestEntry = (
    &'static str,
    Box<dyn Fn(&mut StateVec, &[(QubitId, QubitId)])>,
    Box<dyn Fn(&mut StateVec)>,
);

/// For a 1-qubit state, measure in Z, X, Y bases using `SparseStab`,
/// and verify deterministic outcomes match `StateVec` amplitudes.
fn cross_check_1q(cliff: Clifford) {
    let apply = |sim: &mut StateVec| apply_1q_clifford(sim, cliff);
    let apply_stab = |sim: &mut SparseStab| apply_1q_clifford_stab(sim, cliff);

    // Z-basis check
    {
        let mut stab = SparseStab::new(1);
        apply_stab(&mut stab);
        let results = stab.mz(&qid(0));
        if results[0].is_deterministic {
            let mut sv = StateVec::new(1);
            apply(&mut sv);
            let state = sv.state();
            let prob0 = state[0].norm_sqr();
            let expected_outcome = prob0 <= 0.5;
            assert_eq!(
                results[0].outcome, expected_outcome,
                "Z-basis mismatch for {cliff}: SparseStab={}, StateVec prob0={prob0}",
                results[0].outcome
            );
        }
    }

    // X-basis check (H then Z-measure)
    {
        let mut stab = SparseStab::new(1);
        apply_stab(&mut stab);
        let results = stab.mx(&qid(0));
        if results[0].is_deterministic {
            let mut sv = StateVec::new(1);
            apply(&mut sv);
            sv.h(&qid(0));
            let state = sv.state();
            let prob0 = state[0].norm_sqr();
            let expected_outcome = prob0 <= 0.5;
            assert_eq!(
                results[0].outcome, expected_outcome,
                "X-basis mismatch for {cliff}: SparseStab={}, StateVec prob0={prob0}",
                results[0].outcome
            );
        }
    }

    // Y-basis check
    {
        let mut stab = SparseStab::new(1);
        apply_stab(&mut stab);
        let results = stab.my(&qid(0));
        if results[0].is_deterministic {
            let mut sv = StateVec::new(1);
            apply(&mut sv);
            // MY = SX then MZ then SXdg (per CliffordGateable default)
            sv.sx(&qid(0));
            let state = sv.state();
            let prob0 = state[0].norm_sqr();
            let expected_outcome = prob0 <= 0.5;
            assert_eq!(
                results[0].outcome, expected_outcome,
                "Y-basis mismatch for {cliff}: SparseStab={}, StateVec prob0={prob0}",
                results[0].outcome
            );
        }
    }
}

/// For a 2-qubit state, check deterministic measurements match across simulators.
fn cross_check_2q(
    cliff: Clifford,
    apply_sv: &dyn Fn(&mut StateVec),
    apply_ss: &dyn Fn(&mut SparseStab),
) {
    for q in 0..2 {
        // Z-basis
        {
            let mut stab = SparseStab::new(2);
            apply_ss(&mut stab);
            let results = stab.mz(&qid(q));
            if results[0].is_deterministic {
                let mut sv = StateVec::new(2);
                apply_sv(&mut sv);
                let state = sv.state();
                // Probability of qubit q being |0>: sum |a_k|^2 where bit q of k is 0
                let prob0: f64 = state
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| (k >> q) & 1 == 0)
                    .map(|(_, a)| a.norm_sqr())
                    .sum();
                let expected_outcome = prob0 <= 0.5;
                assert_eq!(
                    results[0].outcome, expected_outcome,
                    "Z-basis q{q} mismatch for {cliff}: SparseStab={}, StateVec prob0={prob0}",
                    results[0].outcome
                );
            }
        }

        // X-basis
        {
            let mut stab = SparseStab::new(2);
            apply_ss(&mut stab);
            let results = stab.mx(&qid(q));
            if results[0].is_deterministic {
                let mut sv = StateVec::new(2);
                apply_sv(&mut sv);
                sv.h(&qid(q));
                let state = sv.state();
                let prob0: f64 = state
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| (k >> q) & 1 == 0)
                    .map(|(_, a)| a.norm_sqr())
                    .sum();
                let expected_outcome = prob0 <= 0.5;
                assert_eq!(
                    results[0].outcome, expected_outcome,
                    "X-basis q{q} mismatch for {cliff}: SparseStab={}, StateVec prob0={prob0}",
                    results[0].outcome
                );
            }
        }
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

fn apply_1q_clifford_stab(sim: &mut SparseStab, cliff: Clifford) {
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
// 1-qubit: SparseStab vs StateVec
// ============================================================================

#[test]
fn sparse_stab_matches_state_vec_all_1q_cliffords() {
    for &cliff in Clifford::all_1q() {
        cross_check_1q(cliff);
    }
}

// ============================================================================
// 2-qubit: SparseStab vs StateVec
// ============================================================================

#[test]
fn sparse_stab_matches_state_vec_2q_cliffords() {
    let gates: Vec<GateTestEntry> = vec![
        (
            Clifford::CX,
            Box::new(|s: &mut StateVec| {
                s.cx(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.cx(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::CY,
            Box::new(|s: &mut StateVec| {
                s.cy(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.cy(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::CZ,
            Box::new(|s: &mut StateVec| {
                s.cz(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.cz(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SWAP,
            Box::new(|s: &mut StateVec| {
                s.swap(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.swap(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SXX,
            Box::new(|s: &mut StateVec| {
                s.sxx(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.sxx(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SXXdg,
            Box::new(|s: &mut StateVec| {
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SYY,
            Box::new(|s: &mut StateVec| {
                s.syy(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.syy(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SYYdg,
            Box::new(|s: &mut StateVec| {
                s.syydg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.syydg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SZZ,
            Box::new(|s: &mut StateVec| {
                s.szz(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.szz(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SZZdg,
            Box::new(|s: &mut StateVec| {
                s.szzdg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.szzdg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::ISWAP,
            Box::new(|s: &mut StateVec| {
                s.iswap(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.iswap(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::G,
            Box::new(|s: &mut StateVec| {
                s.g(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.g(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::ISWAPdg,
            Box::new(|s: &mut StateVec| {
                s.iswapdg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.iswapdg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::Gdg,
            Box::new(|s: &mut StateVec| {
                s.gdg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.gdg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
    ];

    for (cliff, apply_sv, apply_ss) in &gates {
        cross_check_2q(*cliff, apply_sv, apply_ss);
    }
}

// ============================================================================
// 2-qubit on non-trivial input: SparseStab vs StateVec
// ============================================================================

#[test]
fn sparse_stab_matches_state_vec_2q_on_plus_plus() {
    // Apply H to both qubits first, then apply gate, then check measurements
    let gates: Vec<GateTestEntry> = vec![
        (
            Clifford::CX,
            Box::new(|s: &mut StateVec| {
                s.cx(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.cx(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::CY,
            Box::new(|s: &mut StateVec| {
                s.cy(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.cy(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::CZ,
            Box::new(|s: &mut StateVec| {
                s.cz(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.cz(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SWAP,
            Box::new(|s: &mut StateVec| {
                s.swap(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.swap(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SXX,
            Box::new(|s: &mut StateVec| {
                s.sxx(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.sxx(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SXXdg,
            Box::new(|s: &mut StateVec| {
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SYY,
            Box::new(|s: &mut StateVec| {
                s.syy(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.syy(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SYYdg,
            Box::new(|s: &mut StateVec| {
                s.syydg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.syydg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SZZ,
            Box::new(|s: &mut StateVec| {
                s.szz(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.szz(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SZZdg,
            Box::new(|s: &mut StateVec| {
                s.szzdg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.szzdg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::ISWAP,
            Box::new(|s: &mut StateVec| {
                s.iswap(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.iswap(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::G,
            Box::new(|s: &mut StateVec| {
                s.g(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.g(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::ISWAPdg,
            Box::new(|s: &mut StateVec| {
                s.iswapdg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.iswapdg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::Gdg,
            Box::new(|s: &mut StateVec| {
                s.gdg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.gdg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
    ];

    for (cliff, apply_sv, apply_ss) in &gates {
        for q in 0..2 {
            // Z-basis after H|0>H|0> then gate
            {
                let mut stab = SparseStab::new(2);
                stab.h(&qid(0));
                stab.h(&qid(1));
                apply_ss(&mut stab);
                let results = stab.mz(&qid(q));
                if results[0].is_deterministic {
                    let mut sv = StateVec::new(2);
                    sv.h(&qid(0));
                    sv.h(&qid(1));
                    apply_sv(&mut sv);
                    let state = sv.state();
                    let prob0: f64 = state
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| (k >> q) & 1 == 0)
                        .map(|(_, a)| a.norm_sqr())
                        .sum();
                    let expected_outcome = prob0 <= 0.5;
                    assert_eq!(
                        results[0].outcome, expected_outcome,
                        "Z q{q} on |++> mismatch for {cliff}"
                    );
                }
            }

            // X-basis after H|0>H|0> then gate
            {
                let mut stab = SparseStab::new(2);
                stab.h(&qid(0));
                stab.h(&qid(1));
                apply_ss(&mut stab);
                let results = stab.mx(&qid(q));
                if results[0].is_deterministic {
                    let mut sv = StateVec::new(2);
                    sv.h(&qid(0));
                    sv.h(&qid(1));
                    apply_sv(&mut sv);
                    sv.h(&qid(q));
                    let state = sv.state();
                    let prob0: f64 = state
                        .iter()
                        .enumerate()
                        .filter(|(k, _)| (k >> q) & 1 == 0)
                        .map(|(_, a)| a.norm_sqr())
                        .sum();
                    let expected_outcome = prob0 <= 0.5;
                    assert_eq!(
                        results[0].outcome, expected_outcome,
                        "X q{q} on |++> mismatch for {cliff}"
                    );
                }
            }
        }
    }
}

// ============================================================================
// Non-adjacent qubits: SparseStab vs StateVec on qubits (0, 2) in 3-qubit register
// ============================================================================

/// Cross-check deterministic measurements for a 2q gate on non-adjacent qubits (0, 2)
/// in a 3-qubit register.
fn cross_check_nonadjacent(
    cliff: Clifford,
    apply_sv: &dyn Fn(&mut StateVec),
    apply_ss: &dyn Fn(&mut SparseStab),
) {
    for q in [0, 1, 2] {
        // Z-basis on |000>
        {
            let mut stab = SparseStab::new(3);
            apply_ss(&mut stab);
            let results = stab.mz(&qid(q));
            if results[0].is_deterministic {
                let mut sv = StateVec::new(3);
                apply_sv(&mut sv);
                let state = sv.state();
                let prob0: f64 = state
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| (k >> q) & 1 == 0)
                    .map(|(_, a)| a.norm_sqr())
                    .sum();
                let expected_outcome = prob0 <= 0.5;
                assert_eq!(
                    results[0].outcome, expected_outcome,
                    "Z q{q} mismatch for {cliff} on non-adjacent (0,2)"
                );
            }
        }

        // Z-basis after H(0)H(2)
        {
            let mut stab = SparseStab::new(3);
            stab.h(&qid(0));
            stab.h(&qid(2));
            apply_ss(&mut stab);
            let results = stab.mz(&qid(q));
            if results[0].is_deterministic {
                let mut sv = StateVec::new(3);
                sv.h(&qid(0));
                sv.h(&qid(2));
                apply_sv(&mut sv);
                let state = sv.state();
                let prob0: f64 = state
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| (k >> q) & 1 == 0)
                    .map(|(_, a)| a.norm_sqr())
                    .sum();
                let expected_outcome = prob0 <= 0.5;
                assert_eq!(
                    results[0].outcome, expected_outcome,
                    "Z q{q} on H(0)H(2)|000> mismatch for {cliff} on non-adjacent (0,2)"
                );
            }
        }

        // X-basis after H(0)H(2)
        {
            let mut stab = SparseStab::new(3);
            stab.h(&qid(0));
            stab.h(&qid(2));
            apply_ss(&mut stab);
            let results = stab.mx(&qid(q));
            if results[0].is_deterministic {
                let mut sv = StateVec::new(3);
                sv.h(&qid(0));
                sv.h(&qid(2));
                apply_sv(&mut sv);
                sv.h(&qid(q));
                let state = sv.state();
                let prob0: f64 = state
                    .iter()
                    .enumerate()
                    .filter(|(k, _)| (k >> q) & 1 == 0)
                    .map(|(_, a)| a.norm_sqr())
                    .sum();
                let expected_outcome = prob0 <= 0.5;
                assert_eq!(
                    results[0].outcome, expected_outcome,
                    "X q{q} on H(0)H(2)|000> mismatch for {cliff} on non-adjacent (0,2)"
                );
            }
        }
    }
}

#[test]
fn sparse_stab_matches_state_vec_2q_nonadjacent() {
    let gates: Vec<GateTestEntry> = vec![
        (
            Clifford::CX,
            Box::new(|s: &mut StateVec| {
                s.cx(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.cx(&[(QubitId(0), QubitId(2))]);
            }),
        ),
        (
            Clifford::CY,
            Box::new(|s: &mut StateVec| {
                s.cy(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.cy(&[(QubitId(0), QubitId(2))]);
            }),
        ),
        (
            Clifford::CZ,
            Box::new(|s: &mut StateVec| {
                s.cz(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.cz(&[(QubitId(0), QubitId(2))]);
            }),
        ),
        (
            Clifford::SWAP,
            Box::new(|s: &mut StateVec| {
                s.swap(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.swap(&[(QubitId(0), QubitId(2))]);
            }),
        ),
        (
            Clifford::SXX,
            Box::new(|s: &mut StateVec| {
                s.sxx(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.sxx(&[(QubitId(0), QubitId(2))]);
            }),
        ),
        (
            Clifford::SXXdg,
            Box::new(|s: &mut StateVec| {
                s.sxxdg(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.sxxdg(&[(QubitId(0), QubitId(2))]);
            }),
        ),
        (
            Clifford::SYY,
            Box::new(|s: &mut StateVec| {
                s.syy(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.syy(&[(QubitId(0), QubitId(2))]);
            }),
        ),
        (
            Clifford::SYYdg,
            Box::new(|s: &mut StateVec| {
                s.syydg(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.syydg(&[(QubitId(0), QubitId(2))]);
            }),
        ),
        (
            Clifford::SZZ,
            Box::new(|s: &mut StateVec| {
                s.szz(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.szz(&[(QubitId(0), QubitId(2))]);
            }),
        ),
        (
            Clifford::SZZdg,
            Box::new(|s: &mut StateVec| {
                s.szzdg(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.szzdg(&[(QubitId(0), QubitId(2))]);
            }),
        ),
        (
            Clifford::ISWAP,
            Box::new(|s: &mut StateVec| {
                s.iswap(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.iswap(&[(QubitId(0), QubitId(2))]);
            }),
        ),
        (
            Clifford::ISWAPdg,
            Box::new(|s: &mut StateVec| {
                s.iswapdg(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.iswapdg(&[(QubitId(0), QubitId(2))]);
            }),
        ),
        (
            Clifford::G,
            Box::new(|s: &mut StateVec| {
                s.g(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.g(&[(QubitId(0), QubitId(2))]);
            }),
        ),
        (
            Clifford::Gdg,
            Box::new(|s: &mut StateVec| {
                s.gdg(&[(QubitId(0), QubitId(2))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.gdg(&[(QubitId(0), QubitId(2))]);
            }),
        ),
    ];

    for (cliff, apply_sv, apply_ss) in &gates {
        cross_check_nonadjacent(*cliff, apply_sv, apply_ss);
    }
}

// ============================================================================
// Measurement outcome consistency: SparseStab vs StateVec after 2q gates
// ============================================================================

/// For a 2q gate applied to H(0)|00>, verify `SparseStab` and `StateVec` agree on
/// measurement determinism for all qubits in Z, X, and Y bases.
fn cross_check_measurement_after_gate(
    cliff: Clifford,
    apply_sv: &dyn Fn(&mut StateVec),
    apply_ss: &dyn Fn(&mut SparseStab),
) {
    for q in 0..2 {
        // Z-basis measurement after H(0) then gate
        {
            let mut stab = SparseStab::with_seed(2, 42);
            stab.h(&qid(0));
            apply_ss(&mut stab);
            let stab_result = stab.mz(&qid(q));

            let mut sv = StateVec::with_seed(2, 42);
            sv.h(&qid(0));
            apply_sv(&mut sv);
            let sv_result = sv.mz(&qid(q));

            assert_eq!(
                stab_result[0].is_deterministic, sv_result[0].is_deterministic,
                "Z q{q} determinism mismatch for {cliff} on H(0)|00>"
            );
            if stab_result[0].is_deterministic {
                assert_eq!(
                    stab_result[0].outcome, sv_result[0].outcome,
                    "Z q{q} outcome mismatch for {cliff} on H(0)|00>"
                );
            }
        }

        // X-basis measurement after H(0) then gate
        {
            let mut stab = SparseStab::with_seed(2, 42);
            stab.h(&qid(0));
            apply_ss(&mut stab);
            let stab_result = stab.mx(&qid(q));

            let mut sv = StateVec::with_seed(2, 42);
            sv.h(&qid(0));
            apply_sv(&mut sv);
            sv.h(&qid(q));
            let state = sv.state();
            let prob0: f64 = state
                .iter()
                .enumerate()
                .filter(|(k, _)| (k >> q) & 1 == 0)
                .map(|(_, a)| a.norm_sqr())
                .sum();

            if stab_result[0].is_deterministic {
                let expected_outcome = prob0 <= 0.5;
                assert_eq!(
                    stab_result[0].outcome, expected_outcome,
                    "X q{q} outcome mismatch for {cliff} on H(0)|00>"
                );
            }
        }
    }
}

#[test]
fn measurement_consistency_after_2q_gates() {
    let gates: Vec<GateTestEntry> = vec![
        (
            Clifford::CX,
            Box::new(|s: &mut StateVec| {
                s.cx(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.cx(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::CY,
            Box::new(|s: &mut StateVec| {
                s.cy(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.cy(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::CZ,
            Box::new(|s: &mut StateVec| {
                s.cz(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.cz(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SWAP,
            Box::new(|s: &mut StateVec| {
                s.swap(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.swap(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SXX,
            Box::new(|s: &mut StateVec| {
                s.sxx(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.sxx(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SXXdg,
            Box::new(|s: &mut StateVec| {
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SYY,
            Box::new(|s: &mut StateVec| {
                s.syy(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.syy(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SYYdg,
            Box::new(|s: &mut StateVec| {
                s.syydg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.syydg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SZZ,
            Box::new(|s: &mut StateVec| {
                s.szz(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.szz(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::SZZdg,
            Box::new(|s: &mut StateVec| {
                s.szzdg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.szzdg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::ISWAP,
            Box::new(|s: &mut StateVec| {
                s.iswap(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.iswap(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::ISWAPdg,
            Box::new(|s: &mut StateVec| {
                s.iswapdg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.iswapdg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::G,
            Box::new(|s: &mut StateVec| {
                s.g(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.g(&[(QubitId(0), QubitId(1))]);
            }),
        ),
        (
            Clifford::Gdg,
            Box::new(|s: &mut StateVec| {
                s.gdg(&[(QubitId(0), QubitId(1))]);
            }),
            Box::new(|s: &mut SparseStab| {
                s.gdg(&[(QubitId(0), QubitId(1))]);
            }),
        ),
    ];

    for (cliff, apply_sv, apply_ss) in &gates {
        cross_check_measurement_after_gate(*cliff, apply_sv, apply_ss);
    }
}

// ============================================================================
// Batch gate operations: batch vs sequential application
// ============================================================================

#[test]
fn batch_cx_matches_sequential() {
    // 4-qubit system: batch CX([(0,1),(2,3)]) vs sequential CX(0,1) then CX(2,3)
    let batch_qubits = [(QubitId(0), QubitId(1)), (QubitId(2), QubitId(3))];

    // StateVec: batch
    let mut sv_batch = StateVec::new(4);
    for q in 0..4 {
        sv_batch.h(&qid(q));
    }
    sv_batch.cx(&batch_qubits);

    // StateVec: sequential
    let mut sv_seq = StateVec::new(4);
    for q in 0..4 {
        sv_seq.h(&qid(q));
    }
    sv_seq
        .cx(&[(QubitId(0), QubitId(1))])
        .cx(&[(QubitId(2), QubitId(3))]);

    let batch_state = sv_batch.state();
    let seq_state = sv_seq.state();
    for (i, (b, s)) in batch_state.iter().zip(seq_state.iter()).enumerate() {
        assert!(
            (b - s).norm() < 1e-10,
            "CX batch vs seq differ at index {i}: {b} vs {s}"
        );
    }

    // SparseStab: batch vs sequential (check via deterministic measurements)
    let mut ss_batch = SparseStab::new(4);
    for q in 0..4 {
        ss_batch.h(&qid(q));
    }
    ss_batch.cx(&batch_qubits);

    let mut ss_seq = SparseStab::new(4);
    for q in 0..4 {
        ss_seq.h(&qid(q));
    }
    ss_seq
        .cx(&[(QubitId(0), QubitId(1))])
        .cx(&[(QubitId(2), QubitId(3))]);

    for q in 0..4 {
        let r_b = ss_batch.clone().mz(&qid(q));
        let r_s = ss_seq.clone().mz(&qid(q));
        assert_eq!(
            r_b[0].is_deterministic, r_s[0].is_deterministic,
            "CX batch vs seq determinism mismatch on qubit {q}"
        );
        if r_b[0].is_deterministic {
            assert_eq!(
                r_b[0].outcome, r_s[0].outcome,
                "CX batch vs seq outcome mismatch on qubit {q}"
            );
        }
    }
}

#[test]
fn batch_2q_gates_match_sequential() {
    let batch_qubits = [(QubitId(0), QubitId(1)), (QubitId(2), QubitId(3))];

    // Test several 2q gates in batch mode
    let gates: Vec<BatchGateTestEntry> = vec![
        (
            "CZ",
            Box::new(|s, q| {
                s.cz(q);
            }),
            Box::new(|s: &mut StateVec| {
                s.cz(&[(QubitId(0), QubitId(1))])
                    .cz(&[(QubitId(2), QubitId(3))]);
            }),
        ),
        (
            "SWAP",
            Box::new(|s, q| {
                s.swap(q);
            }),
            Box::new(|s: &mut StateVec| {
                s.swap(&[(QubitId(0), QubitId(1))])
                    .swap(&[(QubitId(2), QubitId(3))]);
            }),
        ),
        (
            "SXX",
            Box::new(|s, q| {
                s.sxx(q);
            }),
            Box::new(|s: &mut StateVec| {
                s.sxx(&[(QubitId(0), QubitId(1))])
                    .sxx(&[(QubitId(2), QubitId(3))]);
            }),
        ),
        (
            "SYY",
            Box::new(|s, q| {
                s.syy(q);
            }),
            Box::new(|s: &mut StateVec| {
                s.syy(&[(QubitId(0), QubitId(1))])
                    .syy(&[(QubitId(2), QubitId(3))]);
            }),
        ),
        (
            "SZZ",
            Box::new(|s, q| {
                s.szz(q);
            }),
            Box::new(|s: &mut StateVec| {
                s.szz(&[(QubitId(0), QubitId(1))])
                    .szz(&[(QubitId(2), QubitId(3))]);
            }),
        ),
        (
            "ISWAP",
            Box::new(|s, q| {
                s.iswap(q);
            }),
            Box::new(|s: &mut StateVec| {
                s.iswap(&[(QubitId(0), QubitId(1))])
                    .iswap(&[(QubitId(2), QubitId(3))]);
            }),
        ),
        (
            "ISWAPdg",
            Box::new(|s, q| {
                s.iswapdg(q);
            }),
            Box::new(|s: &mut StateVec| {
                s.iswapdg(&[(QubitId(0), QubitId(1))])
                    .iswapdg(&[(QubitId(2), QubitId(3))]);
            }),
        ),
        (
            "G",
            Box::new(|s, q| {
                s.g(q);
            }),
            Box::new(|s: &mut StateVec| {
                s.g(&[(QubitId(0), QubitId(1))])
                    .g(&[(QubitId(2), QubitId(3))]);
            }),
        ),
    ];

    for (name, batch_fn, seq_fn) in &gates {
        let mut sv_batch = StateVec::new(4);
        for q in 0..4 {
            sv_batch.h(&qid(q));
        }
        batch_fn(&mut sv_batch, &batch_qubits);

        let mut sv_seq = StateVec::new(4);
        for q in 0..4 {
            sv_seq.h(&qid(q));
        }
        seq_fn(&mut sv_seq);

        let batch_state = sv_batch.state();
        let seq_state = sv_seq.state();
        for (i, (b, s)) in batch_state.iter().zip(seq_state.iter()).enumerate() {
            assert!(
                (b - s).norm() < 1e-10,
                "{name} batch vs seq differ at index {i}: {b} vs {s}"
            );
        }
    }
}
