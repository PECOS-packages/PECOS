// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Cross-validation tests for SPP gates (SXX, SYY, SZZ and their daggers).
//!
//! Tests validate agreement across six representations:
//! 1. Unitary matrix (`CliffordRep` -> `UnitaryMatrix`)
//! 2. `StateVec` (state vector simulator)
//! 3. `DensityMatrix`
//! 4. `SparseStab` (W-convention stabilizer)
//! 5. `SparseStabY` (Y-convention stabilizer)
//! 6. `CliffordRep` (Pauli image tracking)

mod helpers;

use helpers::assert_states_equal;
use pecos_core::PauliString;
use pecos_core::QubitId;
use pecos_core::Set;
use pecos_core::clifford::Clifford;
use pecos_quantum::unitary_matrix::{ToMatrix, UnitaryMatrix};
use pecos_simulators::{
    CliffordGateable, DenseStateVec, DensityMatrix, SparseStab, SparseStabHybrid, SparseStabY,
    StateVec, qid,
};

type NamedPrep = (&'static str, fn(&mut StateVec));
type CircuitTestEntry = (
    &'static str,
    fn(&mut StateVec),
    fn(&mut DensityMatrix),
    fn(&mut SparseStab),
    fn(&mut SparseStabY),
);

// ============================================================================
// Helpers
// ============================================================================

/// Apply a 2q SPP gate to `StateVec`.
fn apply_sv(sim: &mut StateVec, gate: Clifford) {
    let q = [(QubitId(0), QubitId(1))];
    match gate {
        Clifford::SXX => {
            sim.sxx(&q);
        }
        Clifford::SXXdg => {
            sim.sxxdg(&q);
        }
        Clifford::SYY => {
            sim.syy(&q);
        }
        Clifford::SYYdg => {
            sim.syydg(&q);
        }
        Clifford::SZZ => {
            sim.szz(&q);
        }
        Clifford::SZZdg => {
            sim.szzdg(&q);
        }
        _ => panic!("unexpected gate {gate:?}"),
    }
}

/// Apply a 2q SPP gate to `DensityMatrix`.
fn apply_dm(sim: &mut DensityMatrix, gate: Clifford) {
    let q = [(QubitId(0), QubitId(1))];
    match gate {
        Clifford::SXX => {
            sim.sxx(&q);
        }
        Clifford::SXXdg => {
            sim.sxxdg(&q);
        }
        Clifford::SYY => {
            sim.syy(&q);
        }
        Clifford::SYYdg => {
            sim.syydg(&q);
        }
        Clifford::SZZ => {
            sim.szz(&q);
        }
        Clifford::SZZdg => {
            sim.szzdg(&q);
        }
        _ => panic!("unexpected gate {gate:?}"),
    }
}

/// Apply a 2q SPP gate to `SparseStab`.
fn apply_ss(sim: &mut SparseStab, gate: Clifford) {
    let q = [(QubitId(0), QubitId(1))];
    match gate {
        Clifford::SXX => {
            sim.sxx(&q);
        }
        Clifford::SXXdg => {
            sim.sxxdg(&q);
        }
        Clifford::SYY => {
            sim.syy(&q);
        }
        Clifford::SYYdg => {
            sim.syydg(&q);
        }
        Clifford::SZZ => {
            sim.szz(&q);
        }
        Clifford::SZZdg => {
            sim.szzdg(&q);
        }
        _ => panic!("unexpected gate {gate:?}"),
    }
}

/// Apply a 2q SPP gate to `SparseStabY`.
fn apply_sy(sim: &mut SparseStabY, gate: Clifford) {
    let q = [(QubitId(0), QubitId(1))];
    match gate {
        Clifford::SXX => {
            sim.sxx(&q);
        }
        Clifford::SXXdg => {
            sim.sxxdg(&q);
        }
        Clifford::SYY => {
            sim.syy(&q);
        }
        Clifford::SYYdg => {
            sim.syydg(&q);
        }
        Clifford::SZZ => {
            sim.szz(&q);
        }
        Clifford::SZZdg => {
            sim.szzdg(&q);
        }
        _ => panic!("unexpected gate {gate:?}"),
    }
}

/// Apply a 2q SPP gate to `SparseStabHybrid`.
fn apply_sh(sim: &mut SparseStabHybrid, gate: Clifford) {
    let q = [(QubitId(0), QubitId(1))];
    match gate {
        Clifford::SXX => {
            sim.sxx(&q);
        }
        Clifford::SXXdg => {
            sim.sxxdg(&q);
        }
        Clifford::SYY => {
            sim.syy(&q);
        }
        Clifford::SYYdg => {
            sim.syydg(&q);
        }
        Clifford::SZZ => {
            sim.szz(&q);
        }
        Clifford::SZZdg => {
            sim.szzdg(&q);
        }
        _ => panic!("unexpected gate {gate:?}"),
    }
}

/// Apply a 2q SPP gate to `DenseStateVec` (`StateVecSoA`).
fn apply_dsv(sim: &mut DenseStateVec, gate: Clifford) {
    let q = [(QubitId(0), QubitId(1))];
    match gate {
        Clifford::SXX => {
            sim.sxx(&q);
        }
        Clifford::SXXdg => {
            sim.sxxdg(&q);
        }
        Clifford::SYY => {
            sim.syy(&q);
        }
        Clifford::SYYdg => {
            sim.syydg(&q);
        }
        Clifford::SZZ => {
            sim.szz(&q);
        }
        Clifford::SZZdg => {
            sim.szzdg(&q);
        }
        _ => panic!("unexpected gate {gate:?}"),
    }
}

const SPP_GATES: [Clifford; 6] = [
    Clifford::SXX,
    Clifford::SXXdg,
    Clifford::SYY,
    Clifford::SYYdg,
    Clifford::SZZ,
    Clifford::SZZdg,
];

/// Apply a unitary matrix to a state vector.
fn matrix_times_state(
    mat: &UnitaryMatrix,
    state: &[num_complex::Complex64],
) -> Vec<num_complex::Complex64> {
    let dim = mat.nrows();
    (0..dim)
        .map(|r| (0..dim).map(|c| mat[(r, c)] * state[c]).sum())
        .collect()
}

/// Compare two probability distributions (from `DensityMatrix` or `StateVec`).
fn assert_probs_close(p1: f64, p2: f64, context: &str) {
    assert!(
        (p1 - p2).abs() < 1e-10,
        "Probability mismatch ({context}): {p1} vs {p2}"
    );
}

/// Extract probabilities from a stabilizer sim by forced measurement.
/// Returns probability for each computational basis state.
fn stab_probabilities(stab: &SparseStab, num_qubits: usize) -> Vec<f64> {
    let dim = 1 << num_qubits;
    let mut probs = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut stab_copy = stab.clone();
        let mut prob = 1.0;
        let mut zero = false;
        for q in 0..num_qubits {
            let bit = (i >> q) & 1 == 1;
            let result = stab_copy.mz_forced(q, bit);
            if !result.is_deterministic {
                prob *= 0.5;
            } else if result.outcome != bit {
                zero = true;
                break;
            }
        }
        probs.push(if zero { 0.0 } else { prob });
    }
    probs
}

/// Extract probabilities from `SparseStabY` by forced measurement.
fn stab_y_probabilities(stab: &SparseStabY, num_qubits: usize) -> Vec<f64> {
    let dim = 1 << num_qubits;
    let mut probs = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut stab_copy = stab.clone();
        let mut prob = 1.0;
        let mut zero = false;
        for q in 0..num_qubits {
            let bit = (i >> q) & 1 == 1;
            let result = stab_copy.mz_forced(q, bit);
            if !result.is_deterministic {
                prob *= 0.5;
            } else if result.outcome != bit {
                zero = true;
                break;
            }
        }
        probs.push(if zero { 0.0 } else { prob });
    }
    probs
}

/// Extract probabilities from `SparseStabHybrid` by forced measurement.
fn stab_hybrid_probabilities(stab: &SparseStabHybrid, num_qubits: usize) -> Vec<f64> {
    let dim = 1 << num_qubits;
    let mut probs = Vec::with_capacity(dim);
    for i in 0..dim {
        let mut stab_copy = stab.clone();
        let mut prob = 1.0;
        let mut zero = false;
        for q in 0..num_qubits {
            let bit = (i >> q) & 1 == 1;
            let result = stab_copy.mz_forced(q, bit);
            if !result.is_deterministic {
                prob *= 0.5;
            } else if result.outcome != bit {
                zero = true;
                break;
            }
        }
        probs.push(if zero { 0.0 } else { prob });
    }
    probs
}

type StatePrep = (
    &'static str,
    fn(&mut StateVec),
    fn(&mut DensityMatrix),
    fn(&mut SparseStab),
    fn(&mut SparseStabY),
    fn(&mut SparseStabHybrid),
);

/// Input states that exercise different Pauli sectors.
/// Includes Z-basis, X-basis, Y-basis, and entangled states to cover all
/// sign branches (especially z-parity for SYY).
fn input_states() -> Vec<StatePrep> {
    vec![
        (
            "|00>",
            |_: &mut StateVec| {},
            |_: &mut DensityMatrix| {},
            |_: &mut SparseStab| {},
            |_: &mut SparseStabY| {},
            |_: &mut SparseStabHybrid| {},
        ),
        (
            "|10>",
            |s: &mut StateVec| {
                s.x(&qid(0));
            },
            |s: &mut DensityMatrix| {
                s.x(&qid(0));
            },
            |s: &mut SparseStab| {
                s.x(&qid(0));
            },
            |s: &mut SparseStabY| {
                s.x(&qid(0));
            },
            |s: &mut SparseStabHybrid| {
                s.x(&qid(0));
            },
        ),
        (
            "|01>",
            |s: &mut StateVec| {
                s.x(&qid(1));
            },
            |s: &mut DensityMatrix| {
                s.x(&qid(1));
            },
            |s: &mut SparseStab| {
                s.x(&qid(1));
            },
            |s: &mut SparseStabY| {
                s.x(&qid(1));
            },
            |s: &mut SparseStabHybrid| {
                s.x(&qid(1));
            },
        ),
        (
            "|11>",
            |s: &mut StateVec| {
                s.x(&qid(0));
                s.x(&qid(1));
            },
            |s: &mut DensityMatrix| {
                s.x(&qid(0));
                s.x(&qid(1));
            },
            |s: &mut SparseStab| {
                s.x(&qid(0));
                s.x(&qid(1));
            },
            |s: &mut SparseStabY| {
                s.x(&qid(0));
                s.x(&qid(1));
            },
            |s: &mut SparseStabHybrid| {
                s.x(&qid(0));
                s.x(&qid(1));
            },
        ),
        (
            "|++>",
            |s: &mut StateVec| {
                s.h(&qid(0));
                s.h(&qid(1));
            },
            |s: &mut DensityMatrix| {
                s.h(&qid(0));
                s.h(&qid(1));
            },
            |s: &mut SparseStab| {
                s.h(&qid(0));
                s.h(&qid(1));
            },
            |s: &mut SparseStabY| {
                s.h(&qid(0));
                s.h(&qid(1));
            },
            |s: &mut SparseStabHybrid| {
                s.h(&qid(0));
                s.h(&qid(1));
            },
        ),
        (
            "|+->",
            |s: &mut StateVec| {
                s.h(&qid(0));
                s.x(&qid(1));
                s.h(&qid(1));
            },
            |s: &mut DensityMatrix| {
                s.h(&qid(0));
                s.x(&qid(1));
                s.h(&qid(1));
            },
            |s: &mut SparseStab| {
                s.h(&qid(0));
                s.x(&qid(1));
                s.h(&qid(1));
            },
            |s: &mut SparseStabY| {
                s.h(&qid(0));
                s.x(&qid(1));
                s.h(&qid(1));
            },
            |s: &mut SparseStabHybrid| {
                s.h(&qid(0));
                s.x(&qid(1));
                s.h(&qid(1));
            },
        ),
        (
            "Bell |00>+|11>",
            |s: &mut StateVec| {
                s.h(&qid(0));
                s.cx(&[(QubitId(0), QubitId(1))]);
            },
            |s: &mut DensityMatrix| {
                s.h(&qid(0));
                s.cx(&[(QubitId(0), QubitId(1))]);
            },
            |s: &mut SparseStab| {
                s.h(&qid(0));
                s.cx(&[(QubitId(0), QubitId(1))]);
            },
            |s: &mut SparseStabY| {
                s.h(&qid(0));
                s.cx(&[(QubitId(0), QubitId(1))]);
            },
            |s: &mut SparseStabHybrid| {
                s.h(&qid(0));
                s.cx(&[(QubitId(0), QubitId(1))]);
            },
        ),
        (
            "|0,+i> (Y eigenstate at q1)",
            |s: &mut StateVec| {
                s.sx(&qid(1));
            },
            |s: &mut DensityMatrix| {
                s.sx(&qid(1));
            },
            |s: &mut SparseStab| {
                s.sx(&qid(1));
            },
            |s: &mut SparseStabY| {
                s.sx(&qid(1));
            },
            |s: &mut SparseStabHybrid| {
                s.sx(&qid(1));
            },
        ),
        // Y-eigenstates on both qubits: exercises both z-parity branches for SYY
        (
            "|+i,+i> (Y eigenstates both)",
            |s: &mut StateVec| {
                s.sx(&qid(0));
                s.sx(&qid(1));
            },
            |s: &mut DensityMatrix| {
                s.sx(&qid(0));
                s.sx(&qid(1));
            },
            |s: &mut SparseStab| {
                s.sx(&qid(0));
                s.sx(&qid(1));
            },
            |s: &mut SparseStabY| {
                s.sx(&qid(0));
                s.sx(&qid(1));
            },
            |s: &mut SparseStabHybrid| {
                s.sx(&qid(0));
                s.sx(&qid(1));
            },
        ),
        (
            "|+i,-i> (opposite Y eigenstates)",
            |s: &mut StateVec| {
                s.sx(&qid(0));
                s.sxdg(&qid(1));
            },
            |s: &mut DensityMatrix| {
                s.sx(&qid(0));
                s.sxdg(&qid(1));
            },
            |s: &mut SparseStab| {
                s.sx(&qid(0));
                s.sxdg(&qid(1));
            },
            |s: &mut SparseStabY| {
                s.sx(&qid(0));
                s.sxdg(&qid(1));
            },
            |s: &mut SparseStabHybrid| {
                s.sx(&qid(0));
                s.sxdg(&qid(1));
            },
        ),
        // Mixed X/Y: generators with different (x,z) patterns
        (
            "|+,+i> (X at q0, Y at q1)",
            |s: &mut StateVec| {
                s.h(&qid(0));
                s.sx(&qid(1));
            },
            |s: &mut DensityMatrix| {
                s.h(&qid(0));
                s.sx(&qid(1));
            },
            |s: &mut SparseStab| {
                s.h(&qid(0));
                s.sx(&qid(1));
            },
            |s: &mut SparseStabY| {
                s.h(&qid(0));
                s.sx(&qid(1));
            },
            |s: &mut SparseStabHybrid| {
                s.h(&qid(0));
                s.sx(&qid(1));
            },
        ),
    ]
}

// ============================================================================
// 0a. StateVec vs Unitary Matrix for all 1q Clifford gates
// ============================================================================

#[test]
fn statevec_matches_unitary_matrix_all_1q_cliffords() {
    // 1-qubit input state preparation functions
    let preps: Vec<NamedPrep> = vec![
        ("|0>", |_: &mut StateVec| {}),
        ("|1>", |s: &mut StateVec| {
            s.x(&qid(0));
        }),
        ("|+>", |s: &mut StateVec| {
            s.h(&qid(0));
        }),
        ("|->", |s: &mut StateVec| {
            s.x(&qid(0));
            s.h(&qid(0));
        }),
        ("|+i>", |s: &mut StateVec| {
            s.sx(&qid(0));
        }),
        ("|-i>", |s: &mut StateVec| {
            s.sxdg(&qid(0));
        }),
    ];

    for &gate in Clifford::all_1q() {
        let mat = gate.to_matrix();

        for (name, prep) in &preps {
            let input_state = {
                let mut sim = StateVec::new(1);
                prep(&mut sim);
                sim.state()
            };
            let expected = matrix_times_state(&mat, &input_state);

            let mut sim = StateVec::new(1);
            prep(&mut sim);
            apply_1q_sv(&mut sim, gate);

            assert_states_equal(sim.state(), &expected);
            let _ = name;
        }
    }
}

// ============================================================================
// 0b. StateVec vs Unitary Matrix for all 2q Clifford gates
// ============================================================================

#[test]
fn statevec_matches_unitary_matrix_all_2q_cliffords() {
    for &gate in Clifford::all_2q() {
        let mat = gate.to_matrix();

        for (name, prep_sv, _, _, _, _) in input_states() {
            let input_state = {
                let mut sim = StateVec::new(2);
                prep_sv(&mut sim);
                sim.state()
            };
            let expected = matrix_times_state(&mat, &input_state);

            let mut sim = StateVec::new(2);
            prep_sv(&mut sim);
            apply_2q_sv(&mut sim, gate);

            assert_states_equal(sim.state(), &expected);
            let _ = name;
        }
    }
}

// ============================================================================
// 1. StateVec vs Unitary Matrix for all SPP gates
// ============================================================================

#[test]
fn statevec_matches_unitary_matrix_spp_gates() {
    for gate in SPP_GATES {
        let mat = gate.to_matrix();

        for (name, prep_sv, _, _, _, _) in input_states() {
            let input_state = {
                let mut sim = StateVec::new(2);
                prep_sv(&mut sim);
                sim.state()
            };
            let expected = matrix_times_state(&mat, &input_state);

            let mut sim = StateVec::new(2);
            prep_sv(&mut sim);
            apply_sv(&mut sim, gate);

            assert_states_equal(sim.state(), &expected);
            let _ = name;
        }
    }
}

// ============================================================================
// 2. DensityMatrix vs StateVec for all SPP gates
// ============================================================================

#[test]
fn density_matrix_matches_statevec_spp_gates() {
    for gate in SPP_GATES {
        for (name, prep_sv, prep_dm, _, _, _) in input_states() {
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            apply_sv(&mut sv, gate);

            let mut dm = DensityMatrix::new(2);
            prep_dm(&mut dm);
            apply_dm(&mut dm, gate);

            for i in 0..4 {
                let sv_prob = sv.probability(i);
                let dm_prob = dm.probability(i);
                assert_probs_close(
                    sv_prob,
                    dm_prob,
                    &format!("{gate:?} on {name}, basis state {i}"),
                );
            }
        }
    }
}

// ============================================================================
// 3. SparseStab (W-convention) vs StateVec for all SPP gates
// ============================================================================

#[test]
fn sparse_stab_matches_statevec_spp_gates() {
    for gate in SPP_GATES {
        for (name, prep_sv, _, prep_ss, _, _) in input_states() {
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            apply_sv(&mut sv, gate);

            let mut ss = SparseStab::new(2);
            prep_ss(&mut ss);
            apply_ss(&mut ss, gate);

            let ss_probs = stab_probabilities(&ss, 2);
            for (i, &ss_prob) in ss_probs.iter().enumerate() {
                assert_probs_close(
                    sv.probability(i),
                    ss_prob,
                    &format!("{gate:?} on {name}, basis state {i}: SparseStab vs StateVec"),
                );
            }
        }
    }
}

// ============================================================================
// 4. SparseStabY (Y-convention) vs StateVec for all SPP gates
// ============================================================================

#[test]
fn sparse_stab_y_matches_statevec_spp_gates() {
    for gate in SPP_GATES {
        for (name, prep_sv, _, _, prep_sy, _) in input_states() {
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            apply_sv(&mut sv, gate);

            let mut sy = SparseStabY::new(2);
            prep_sy(&mut sy);
            apply_sy(&mut sy, gate);

            let sy_probs = stab_y_probabilities(&sy, 2);
            for (i, &sy_prob) in sy_probs.iter().enumerate() {
                assert_probs_close(
                    sv.probability(i),
                    sy_prob,
                    &format!("{gate:?} on {name}, basis state {i}: SparseStabY vs StateVec"),
                );
            }
        }
    }
}

// ============================================================================
// 5. SparseStab vs SparseStabY: direct comparison via probabilities
// ============================================================================

#[test]
fn sparse_stab_matches_sparse_stab_y_spp_gates() {
    for gate in SPP_GATES {
        for (name, _, _, prep_ss, prep_sy, _) in input_states() {
            let mut ss = SparseStab::new(2);
            prep_ss(&mut ss);
            apply_ss(&mut ss, gate);

            let mut sy = SparseStabY::new(2);
            prep_sy(&mut sy);
            apply_sy(&mut sy, gate);

            let ss_probs = stab_probabilities(&ss, 2);
            let sy_probs = stab_y_probabilities(&sy, 2);
            for i in 0..4 {
                assert_probs_close(
                    ss_probs[i],
                    sy_probs[i],
                    &format!("{gate:?} on {name}, basis state {i}: SparseStab vs SparseStabY"),
                );
            }
        }
    }
}

// ============================================================================
// 6. SparseStab vs SparseStabY: deterministic measurement outcomes agree
// ============================================================================

#[test]
fn deterministic_measurements_agree_all_stab_sims_spp_gates() {
    for gate in SPP_GATES {
        for (name, _, _, prep_ss, prep_sy, prep_sh) in input_states() {
            let mut ss = SparseStab::new(2);
            prep_ss(&mut ss);
            apply_ss(&mut ss, gate);

            let mut sy = SparseStabY::new(2);
            prep_sy(&mut sy);
            apply_sy(&mut sy, gate);

            let mut sh = SparseStabHybrid::new(2);
            prep_sh(&mut sh);
            apply_sh(&mut sh, gate);

            // Check Z-basis measurements on each qubit
            for q in 0..2 {
                let mut ss_copy = ss.clone();
                let mut sy_copy = sy.clone();
                let mut sh_copy = sh.clone();
                let ss_result = ss_copy.mz(&qid(q));
                let sy_result = sy_copy.mz(&qid(q));
                let sh_result = sh_copy.mz(&qid(q));

                assert_eq!(
                    ss_result[0].is_deterministic, sy_result[0].is_deterministic,
                    "{gate:?} on {name}: Z-basis determinism mismatch SS vs SY at q{q}"
                );
                assert_eq!(
                    ss_result[0].is_deterministic, sh_result[0].is_deterministic,
                    "{gate:?} on {name}: Z-basis determinism mismatch SS vs SH at q{q}"
                );
                if ss_result[0].is_deterministic {
                    assert_eq!(
                        ss_result[0].outcome, sy_result[0].outcome,
                        "{gate:?} on {name}: Z-basis outcome mismatch SS vs SY at q{q}"
                    );
                    assert_eq!(
                        ss_result[0].outcome, sh_result[0].outcome,
                        "{gate:?} on {name}: Z-basis outcome mismatch SS vs SH at q{q}"
                    );
                }
            }

            // Check X-basis measurements
            for q in 0..2 {
                let mut ss_copy = ss.clone();
                let mut sy_copy = sy.clone();
                let mut sh_copy = sh.clone();
                let ss_result = ss_copy.mx(&qid(q));
                let sy_result = sy_copy.mx(&qid(q));
                let sh_result = sh_copy.mx(&qid(q));

                assert_eq!(
                    ss_result[0].is_deterministic, sy_result[0].is_deterministic,
                    "{gate:?} on {name}: X-basis determinism mismatch SS vs SY at q{q}"
                );
                assert_eq!(
                    ss_result[0].is_deterministic, sh_result[0].is_deterministic,
                    "{gate:?} on {name}: X-basis determinism mismatch SS vs SH at q{q}"
                );
                if ss_result[0].is_deterministic {
                    assert_eq!(
                        ss_result[0].outcome, sy_result[0].outcome,
                        "{gate:?} on {name}: X-basis outcome mismatch SS vs SY at q{q}"
                    );
                    assert_eq!(
                        ss_result[0].outcome, sh_result[0].outcome,
                        "{gate:?} on {name}: X-basis outcome mismatch SS vs SH at q{q}"
                    );
                }
            }

            // Check Y-basis measurements
            for q in 0..2 {
                let mut ss_copy = ss.clone();
                let mut sy_copy = sy.clone();
                let mut sh_copy = sh.clone();
                let ss_result = ss_copy.my(&qid(q));
                let sy_result = sy_copy.my(&qid(q));
                let sh_result = sh_copy.my(&qid(q));

                assert_eq!(
                    ss_result[0].is_deterministic, sy_result[0].is_deterministic,
                    "{gate:?} on {name}: Y-basis determinism mismatch SS vs SY at q{q}"
                );
                assert_eq!(
                    ss_result[0].is_deterministic, sh_result[0].is_deterministic,
                    "{gate:?} on {name}: Y-basis determinism mismatch SS vs SH at q{q}"
                );
                if ss_result[0].is_deterministic {
                    assert_eq!(
                        ss_result[0].outcome, sy_result[0].outcome,
                        "{gate:?} on {name}: Y-basis outcome mismatch SS vs SY at q{q}"
                    );
                    assert_eq!(
                        ss_result[0].outcome, sh_result[0].outcome,
                        "{gate:?} on {name}: Y-basis outcome mismatch SS vs SH at q{q}"
                    );
                }
            }
        }
    }
}

// ============================================================================
// 7. CliffordRep Pauli images: all four generators, all stabilizer sims
// ============================================================================

/// Extract Pauli bit pattern from a `CliffordRep` image.
/// Returns (`x_bits`, `z_bits`, `num_ys`) -- the sign encoding depends on convention.
fn image_pauli_bits(image: &PauliString, num_qubits: usize) -> (Vec<bool>, Vec<bool>, u32) {
    use pecos_core::Pauli;

    let mut x_bits = vec![false; num_qubits];
    let mut z_bits = vec![false; num_qubits];
    let mut num_ys = 0u32;

    for (p, qid) in image.iter_pairs() {
        let q = usize::from(qid);
        match p {
            Pauli::I => {}
            Pauli::X => {
                x_bits[q] = true;
            }
            Pauli::Z => {
                z_bits[q] = true;
            }
            Pauli::Y => {
                x_bits[q] = true;
                z_bits[q] = true;
                num_ys += 1;
            }
        }
    }
    (x_bits, z_bits, num_ys)
}

/// Convert `CliffordRep` image to W-convention (SparseStab/SparseStabHybrid) sign bits.
/// W = XZ, so Y = iW means each Y factor absorbs an extra i into the phase.
fn image_to_w_signs(image: &PauliString, num_ys: u32) -> (bool, bool) {
    let base = image.phase() as u8;
    let w_phase = (base + 2 * (num_ys % 4) as u8) % 4;
    (w_phase & 1 != 0, w_phase & 2 != 0)
}

/// Convert `CliffordRep` image to Y-convention (`SparseStabY`) sign bits.
/// Y is stored directly, so `CliffordRep` phase maps with no conversion.
fn image_to_y_signs(image: &PauliString) -> (bool, bool) {
    let phase = image.phase() as u8;
    (phase & 1 != 0, phase & 2 != 0)
}

#[test]
fn clifford_rep_matches_all_stab_sims_spp_gates() {
    let inputs: [(PauliString, usize, bool); 4] = [
        (PauliString::x(0), 0, true),
        (PauliString::z(0), 0, false),
        (PauliString::x(1), 1, true),
        (PauliString::z(1), 1, false),
    ];

    for gate in SPP_GATES {
        let rep = gate.on_qubits(0, 1);

        for (input_ps, input_q, init_x) in &inputs {
            let image = rep.apply(input_ps);
            let (exp_x, exp_z, num_ys) = image_pauli_bits(&image, 2);
            let (w_minus, w_i) = image_to_w_signs(&image, num_ys);
            let (y_minus, y_i) = image_to_y_signs(&image);
            let gen_id = *input_q;

            // SparseStab (W-convention)
            {
                let mut ss = SparseStab::new(2);
                if *init_x {
                    ss.h(&qid(*input_q));
                }
                apply_ss(&mut ss, gate);

                for qubit in 0..2 {
                    assert_eq!(
                        ss.stabs().col_x[qubit].contains(gen_id),
                        exp_x[qubit],
                        "{gate:?} on {input_ps:?}: SparseStab qubit {qubit} X-bit mismatch (image: {image:?})"
                    );
                    assert_eq!(
                        ss.stabs().col_z[qubit].contains(gen_id),
                        exp_z[qubit],
                        "{gate:?} on {input_ps:?}: SparseStab qubit {qubit} Z-bit mismatch (image: {image:?})"
                    );
                }
                assert_eq!(
                    ss.stabs().signs_minus.contains(gen_id),
                    w_minus,
                    "{gate:?} on {input_ps:?}: SparseStab signs_minus mismatch (image: {image:?})"
                );
                assert_eq!(
                    ss.stabs().signs_i.contains(gen_id),
                    w_i,
                    "{gate:?} on {input_ps:?}: SparseStab signs_i mismatch (image: {image:?})"
                );
            }

            // SparseStabY (Y-convention)
            {
                let mut sy = SparseStabY::new(2);
                if *init_x {
                    sy.h(&qid(*input_q));
                }
                apply_sy(&mut sy, gate);

                for qubit in 0..2 {
                    assert_eq!(
                        sy.stabs().col_x[qubit].contains(gen_id),
                        exp_x[qubit],
                        "{gate:?} on {input_ps:?}: SparseStabY qubit {qubit} X-bit mismatch (image: {image:?})"
                    );
                    assert_eq!(
                        sy.stabs().col_z[qubit].contains(gen_id),
                        exp_z[qubit],
                        "{gate:?} on {input_ps:?}: SparseStabY qubit {qubit} Z-bit mismatch (image: {image:?})"
                    );
                }
                assert_eq!(
                    sy.stabs().signs_minus.contains(gen_id),
                    y_minus,
                    "{gate:?} on {input_ps:?}: SparseStabY signs_minus mismatch (image: {image:?})"
                );
                assert_eq!(
                    sy.stabs().signs_i.contains(gen_id),
                    y_i,
                    "{gate:?} on {input_ps:?}: SparseStabY signs_i mismatch (image: {image:?})"
                );
            }

            // SparseStabHybrid (W-convention, hybrid storage)
            {
                let mut sh = SparseStabHybrid::new(2);
                if *init_x {
                    sh.h(&qid(*input_q));
                }
                apply_sh(&mut sh, gate);

                for qubit in 0..2 {
                    assert_eq!(
                        sh.stabs().col_x[qubit].contains(&gen_id),
                        exp_x[qubit],
                        "{gate:?} on {input_ps:?}: SparseStabHybrid qubit {qubit} X-bit mismatch (image: {image:?})"
                    );
                    assert_eq!(
                        sh.stabs().col_z[qubit].contains(&gen_id),
                        exp_z[qubit],
                        "{gate:?} on {input_ps:?}: SparseStabHybrid qubit {qubit} Z-bit mismatch (image: {image:?})"
                    );
                }
                assert_eq!(
                    sh.stabs().signs_minus.contains(gen_id),
                    w_minus,
                    "{gate:?} on {input_ps:?}: SparseStabHybrid signs_minus mismatch (image: {image:?})"
                );
                assert_eq!(
                    sh.stabs().signs_i.contains(gen_id),
                    w_i,
                    "{gate:?} on {input_ps:?}: SparseStabHybrid signs_i mismatch (image: {image:?})"
                );
            }
        }
    }
}

// ============================================================================
// 8. SPP^2 = PP property across all simulators
// ============================================================================

#[test]
fn spp_squared_is_pp_all_simulators() {
    for (name, prep_sv, prep_dm, prep_ss, prep_sy, _prep_sh) in input_states() {
        // SXX^2 = XX
        {
            // StateVec reference: X(0)X(1)
            let mut ref_sv = StateVec::new(2);
            prep_sv(&mut ref_sv);
            ref_sv.x(&qid(0));
            ref_sv.x(&qid(1));

            // StateVec: SXX twice
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            sv.sxx(&[(QubitId(0), QubitId(1))]);
            sv.sxx(&[(QubitId(0), QubitId(1))]);
            assert_states_equal(sv.state(), ref_sv.state());

            // DensityMatrix
            let mut dm = DensityMatrix::new(2);
            prep_dm(&mut dm);
            dm.sxx(&[(QubitId(0), QubitId(1))]);
            dm.sxx(&[(QubitId(0), QubitId(1))]);
            for i in 0..4 {
                assert_probs_close(
                    ref_sv.probability(i),
                    dm.probability(i),
                    &format!("SXX^2=XX DensityMatrix on {name}, state {i}"),
                );
            }

            // SparseStab
            let mut ss = SparseStab::new(2);
            prep_ss(&mut ss);
            ss.sxx(&[(QubitId(0), QubitId(1))]);
            ss.sxx(&[(QubitId(0), QubitId(1))]);
            let ss_probs = stab_probabilities(&ss, 2);
            for (i, &ss_prob) in ss_probs.iter().enumerate() {
                assert_probs_close(
                    ref_sv.probability(i),
                    ss_prob,
                    &format!("SXX^2=XX SparseStab on {name}, state {i}"),
                );
            }

            // SparseStabY
            let mut sy = SparseStabY::new(2);
            prep_sy(&mut sy);
            sy.sxx(&[(QubitId(0), QubitId(1))]);
            sy.sxx(&[(QubitId(0), QubitId(1))]);
            let sy_probs = stab_y_probabilities(&sy, 2);
            for (i, &sy_prob) in sy_probs.iter().enumerate() {
                assert_probs_close(
                    ref_sv.probability(i),
                    sy_prob,
                    &format!("SXX^2=XX SparseStabY on {name}, state {i}"),
                );
            }
        }

        // SYY^2 = YY
        {
            let mut ref_sv = StateVec::new(2);
            prep_sv(&mut ref_sv);
            ref_sv.y(&qid(0));
            ref_sv.y(&qid(1));

            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            sv.syy(&[(QubitId(0), QubitId(1))]);
            sv.syy(&[(QubitId(0), QubitId(1))]);
            assert_states_equal(sv.state(), ref_sv.state());

            let mut dm = DensityMatrix::new(2);
            prep_dm(&mut dm);
            dm.syy(&[(QubitId(0), QubitId(1))]);
            dm.syy(&[(QubitId(0), QubitId(1))]);
            for i in 0..4 {
                assert_probs_close(
                    ref_sv.probability(i),
                    dm.probability(i),
                    &format!("SYY^2=YY DensityMatrix on {name}, state {i}"),
                );
            }

            let mut ss = SparseStab::new(2);
            prep_ss(&mut ss);
            ss.syy(&[(QubitId(0), QubitId(1))]);
            ss.syy(&[(QubitId(0), QubitId(1))]);
            let ss_probs = stab_probabilities(&ss, 2);
            for (i, &ss_prob) in ss_probs.iter().enumerate() {
                assert_probs_close(
                    ref_sv.probability(i),
                    ss_prob,
                    &format!("SYY^2=YY SparseStab on {name}, state {i}"),
                );
            }

            let mut sy = SparseStabY::new(2);
            prep_sy(&mut sy);
            sy.syy(&[(QubitId(0), QubitId(1))]);
            sy.syy(&[(QubitId(0), QubitId(1))]);
            let sy_probs = stab_y_probabilities(&sy, 2);
            for (i, &sy_prob) in sy_probs.iter().enumerate() {
                assert_probs_close(
                    ref_sv.probability(i),
                    sy_prob,
                    &format!("SYY^2=YY SparseStabY on {name}, state {i}"),
                );
            }
        }

        // SZZ^2 = ZZ
        {
            let mut ref_sv = StateVec::new(2);
            prep_sv(&mut ref_sv);
            ref_sv.z(&qid(0));
            ref_sv.z(&qid(1));

            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            sv.szz(&[(QubitId(0), QubitId(1))]);
            sv.szz(&[(QubitId(0), QubitId(1))]);
            assert_states_equal(sv.state(), ref_sv.state());

            let mut dm = DensityMatrix::new(2);
            prep_dm(&mut dm);
            dm.szz(&[(QubitId(0), QubitId(1))]);
            dm.szz(&[(QubitId(0), QubitId(1))]);
            for i in 0..4 {
                assert_probs_close(
                    ref_sv.probability(i),
                    dm.probability(i),
                    &format!("SZZ^2=ZZ DensityMatrix on {name}, state {i}"),
                );
            }

            let mut ss = SparseStab::new(2);
            prep_ss(&mut ss);
            ss.szz(&[(QubitId(0), QubitId(1))]);
            ss.szz(&[(QubitId(0), QubitId(1))]);
            let ss_probs = stab_probabilities(&ss, 2);
            for (i, &ss_prob) in ss_probs.iter().enumerate() {
                assert_probs_close(
                    ref_sv.probability(i),
                    ss_prob,
                    &format!("SZZ^2=ZZ SparseStab on {name}, state {i}"),
                );
            }

            let mut sy = SparseStabY::new(2);
            prep_sy(&mut sy);
            sy.szz(&[(QubitId(0), QubitId(1))]);
            sy.szz(&[(QubitId(0), QubitId(1))]);
            let sy_probs = stab_y_probabilities(&sy, 2);
            for (i, &sy_prob) in sy_probs.iter().enumerate() {
                assert_probs_close(
                    ref_sv.probability(i),
                    sy_prob,
                    &format!("SZZ^2=ZZ SparseStabY on {name}, state {i}"),
                );
            }
        }
    }
}

// ============================================================================
// 9. Gate-then-dagger = identity across all simulators
// ============================================================================

#[test]
fn gate_then_dagger_identity_all_simulators() {
    let pairs = [
        (Clifford::SXX, Clifford::SXXdg),
        (Clifford::SYY, Clifford::SYYdg),
        (Clifford::SZZ, Clifford::SZZdg),
    ];

    for (name, prep_sv, prep_dm, prep_ss, prep_sy, _prep_sh) in input_states() {
        for (gate, dagger) in &pairs {
            // StateVec
            {
                let mut ref_sim = StateVec::new(2);
                prep_sv(&mut ref_sim);

                let mut sim = StateVec::new(2);
                prep_sv(&mut sim);
                apply_sv(&mut sim, *gate);
                apply_sv(&mut sim, *dagger);
                assert_states_equal(sim.state(), ref_sim.state());
            }

            // DensityMatrix
            {
                let mut ref_dm = DensityMatrix::new(2);
                prep_dm(&mut ref_dm);

                let mut dm = DensityMatrix::new(2);
                prep_dm(&mut dm);
                apply_dm(&mut dm, *gate);
                apply_dm(&mut dm, *dagger);
                for i in 0..4 {
                    assert_probs_close(
                        ref_dm.probability(i),
                        dm.probability(i),
                        &format!("{gate:?}*{dagger:?}=I DensityMatrix on {name}, state {i}"),
                    );
                }
            }

            // SparseStab
            {
                let mut ref_ss = SparseStab::new(2);
                prep_ss(&mut ref_ss);
                let ref_probs = stab_probabilities(&ref_ss, 2);

                let mut ss = SparseStab::new(2);
                prep_ss(&mut ss);
                apply_ss(&mut ss, *gate);
                apply_ss(&mut ss, *dagger);
                let probs = stab_probabilities(&ss, 2);
                for i in 0..4 {
                    assert_probs_close(
                        ref_probs[i],
                        probs[i],
                        &format!("{gate:?}*{dagger:?}=I SparseStab on {name}, state {i}"),
                    );
                }
            }

            // SparseStabY
            {
                let mut ref_sy = SparseStabY::new(2);
                prep_sy(&mut ref_sy);
                let ref_probs = stab_y_probabilities(&ref_sy, 2);

                let mut sy = SparseStabY::new(2);
                prep_sy(&mut sy);
                apply_sy(&mut sy, *gate);
                apply_sy(&mut sy, *dagger);
                let probs = stab_y_probabilities(&sy, 2);
                for i in 0..4 {
                    assert_probs_close(
                        ref_probs[i],
                        probs[i],
                        &format!("{gate:?}*{dagger:?}=I SparseStabY on {name}, state {i}"),
                    );
                }
            }
        }
    }
}

// ============================================================================
// 10. Gate sequences: SPP gates composed with other Cliffords
// ============================================================================

#[test]
fn spp_in_circuit_sequences_all_simulators() {
    // Test SPP gates within larger circuit sequences to catch
    // interactions with other gate updates.
    let circuits: Vec<CircuitTestEntry> = vec![
        (
            "H(0).SXX.H(1).SZZ",
            |s: &mut StateVec| {
                s.h(&qid(0));
                s.sxx(&[(QubitId(0), QubitId(1))]);
                s.h(&qid(1));
                s.szz(&[(QubitId(0), QubitId(1))]);
            },
            |s: &mut DensityMatrix| {
                s.h(&qid(0));
                s.sxx(&[(QubitId(0), QubitId(1))]);
                s.h(&qid(1));
                s.szz(&[(QubitId(0), QubitId(1))]);
            },
            |s: &mut SparseStab| {
                s.h(&qid(0));
                s.sxx(&[(QubitId(0), QubitId(1))]);
                s.h(&qid(1));
                s.szz(&[(QubitId(0), QubitId(1))]);
            },
            |s: &mut SparseStabY| {
                s.h(&qid(0));
                s.sxx(&[(QubitId(0), QubitId(1))]);
                s.h(&qid(1));
                s.szz(&[(QubitId(0), QubitId(1))]);
            },
        ),
        (
            "SYY.CX.SZZdg",
            |s: &mut StateVec| {
                s.syy(&[(QubitId(0), QubitId(1))]);
                s.cx(&[(QubitId(0), QubitId(1))]);
                s.szzdg(&[(QubitId(0), QubitId(1))]);
            },
            |s: &mut DensityMatrix| {
                s.syy(&[(QubitId(0), QubitId(1))]);
                s.cx(&[(QubitId(0), QubitId(1))]);
                s.szzdg(&[(QubitId(0), QubitId(1))]);
            },
            |s: &mut SparseStab| {
                s.syy(&[(QubitId(0), QubitId(1))]);
                s.cx(&[(QubitId(0), QubitId(1))]);
                s.szzdg(&[(QubitId(0), QubitId(1))]);
            },
            |s: &mut SparseStabY| {
                s.syy(&[(QubitId(0), QubitId(1))]);
                s.cx(&[(QubitId(0), QubitId(1))]);
                s.szzdg(&[(QubitId(0), QubitId(1))]);
            },
        ),
        (
            "H(0).H(1).SXXdg.SZ(0).SYY.H(0)",
            |s: &mut StateVec| {
                s.h(&qid(0));
                s.h(&qid(1));
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
                s.sz(&qid(0));
                s.syy(&[(QubitId(0), QubitId(1))]);
                s.h(&qid(0));
            },
            |s: &mut DensityMatrix| {
                s.h(&qid(0));
                s.h(&qid(1));
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
                s.sz(&qid(0));
                s.syy(&[(QubitId(0), QubitId(1))]);
                s.h(&qid(0));
            },
            |s: &mut SparseStab| {
                s.h(&qid(0));
                s.h(&qid(1));
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
                s.sz(&qid(0));
                s.syy(&[(QubitId(0), QubitId(1))]);
                s.h(&qid(0));
            },
            |s: &mut SparseStabY| {
                s.h(&qid(0));
                s.h(&qid(1));
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
                s.sz(&qid(0));
                s.syy(&[(QubitId(0), QubitId(1))]);
                s.h(&qid(0));
            },
        ),
        (
            "X(0).SZZ.SXX.SYYdg.H(1)",
            |s: &mut StateVec| {
                s.x(&qid(0));
                s.szz(&[(QubitId(0), QubitId(1))]);
                s.sxx(&[(QubitId(0), QubitId(1))]);
                s.syydg(&[(QubitId(0), QubitId(1))]);
                s.h(&qid(1));
            },
            |s: &mut DensityMatrix| {
                s.x(&qid(0));
                s.szz(&[(QubitId(0), QubitId(1))]);
                s.sxx(&[(QubitId(0), QubitId(1))]);
                s.syydg(&[(QubitId(0), QubitId(1))]);
                s.h(&qid(1));
            },
            |s: &mut SparseStab| {
                s.x(&qid(0));
                s.szz(&[(QubitId(0), QubitId(1))]);
                s.sxx(&[(QubitId(0), QubitId(1))]);
                s.syydg(&[(QubitId(0), QubitId(1))]);
                s.h(&qid(1));
            },
            |s: &mut SparseStabY| {
                s.x(&qid(0));
                s.szz(&[(QubitId(0), QubitId(1))]);
                s.sxx(&[(QubitId(0), QubitId(1))]);
                s.syydg(&[(QubitId(0), QubitId(1))]);
                s.h(&qid(1));
            },
        ),
        (
            "Bell.SYY.SXXdg.SZZ",
            |s: &mut StateVec| {
                s.h(&qid(0));
                s.cx(&[(QubitId(0), QubitId(1))]);
                s.syy(&[(QubitId(0), QubitId(1))]);
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
                s.szz(&[(QubitId(0), QubitId(1))]);
            },
            |s: &mut DensityMatrix| {
                s.h(&qid(0));
                s.cx(&[(QubitId(0), QubitId(1))]);
                s.syy(&[(QubitId(0), QubitId(1))]);
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
                s.szz(&[(QubitId(0), QubitId(1))]);
            },
            |s: &mut SparseStab| {
                s.h(&qid(0));
                s.cx(&[(QubitId(0), QubitId(1))]);
                s.syy(&[(QubitId(0), QubitId(1))]);
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
                s.szz(&[(QubitId(0), QubitId(1))]);
            },
            |s: &mut SparseStabY| {
                s.h(&qid(0));
                s.cx(&[(QubitId(0), QubitId(1))]);
                s.syy(&[(QubitId(0), QubitId(1))]);
                s.sxxdg(&[(QubitId(0), QubitId(1))]);
                s.szz(&[(QubitId(0), QubitId(1))]);
            },
        ),
    ];

    for (circ_name, run_sv, run_dm, run_ss, run_sy) in &circuits {
        let mut sv = StateVec::new(2);
        run_sv(&mut sv);

        let mut dm = DensityMatrix::new(2);
        run_dm(&mut dm);

        let mut ss = SparseStab::new(2);
        run_ss(&mut ss);

        let mut sy = SparseStabY::new(2);
        run_sy(&mut sy);

        let ss_probs = stab_probabilities(&ss, 2);
        let sy_probs = stab_y_probabilities(&sy, 2);

        for i in 0..4 {
            let sv_prob = sv.probability(i);
            let dm_prob = dm.probability(i);
            assert_probs_close(
                sv_prob,
                dm_prob,
                &format!("{circ_name} SV vs DM, state {i}"),
            );
            assert_probs_close(
                sv_prob,
                ss_probs[i],
                &format!("{circ_name} SV vs SS, state {i}"),
            );
            assert_probs_close(
                sv_prob,
                sy_probs[i],
                &format!("{circ_name} SV vs SY, state {i}"),
            );
            assert_probs_close(
                ss_probs[i],
                sy_probs[i],
                &format!("{circ_name} SS vs SY, state {i}"),
            );
        }
    }
}

// ============================================================================
// 11. Non-adjacent qubits: SPP gates on qubits (0, 2) in 3-qubit system
// ============================================================================

#[test]
fn spp_nonadjacent_qubits_all_simulators() {
    let q02 = [(pecos_core::QubitId(0), pecos_core::QubitId(2))];

    for gate in SPP_GATES {
        // Prepare |+0+> then apply gate on (0,2)
        let mut sv = StateVec::new(3);
        sv.h(&qid(0));
        sv.h(&qid(2));
        match gate {
            Clifford::SXX => {
                sv.sxx(&q02);
            }
            Clifford::SXXdg => {
                sv.sxxdg(&q02);
            }
            Clifford::SYY => {
                sv.syy(&q02);
            }
            Clifford::SYYdg => {
                sv.syydg(&q02);
            }
            Clifford::SZZ => {
                sv.szz(&q02);
            }
            Clifford::SZZdg => {
                sv.szzdg(&q02);
            }
            _ => unreachable!(),
        }

        let mut ss = SparseStab::new(3);
        ss.h(&qid(0));
        ss.h(&qid(2));
        match gate {
            Clifford::SXX => {
                ss.sxx(&q02);
            }
            Clifford::SXXdg => {
                ss.sxxdg(&q02);
            }
            Clifford::SYY => {
                ss.syy(&q02);
            }
            Clifford::SYYdg => {
                ss.syydg(&q02);
            }
            Clifford::SZZ => {
                ss.szz(&q02);
            }
            Clifford::SZZdg => {
                ss.szzdg(&q02);
            }
            _ => unreachable!(),
        }

        let mut sy = SparseStabY::new(3);
        sy.h(&qid(0));
        sy.h(&qid(2));
        match gate {
            Clifford::SXX => {
                sy.sxx(&q02);
            }
            Clifford::SXXdg => {
                sy.sxxdg(&q02);
            }
            Clifford::SYY => {
                sy.syy(&q02);
            }
            Clifford::SYYdg => {
                sy.syydg(&q02);
            }
            Clifford::SZZ => {
                sy.szz(&q02);
            }
            Clifford::SZZdg => {
                sy.szzdg(&q02);
            }
            _ => unreachable!(),
        }

        let ss_probs = stab_probabilities(&ss, 3);
        let sy_probs = stab_y_probabilities(&sy, 3);

        for i in 0..8 {
            let sv_prob = sv.probability(i);
            assert_probs_close(
                sv_prob,
                ss_probs[i],
                &format!("{gate:?} nonadj SV vs SS, state {i}"),
            );
            assert_probs_close(
                sv_prob,
                sy_probs[i],
                &format!("{gate:?} nonadj SV vs SY, state {i}"),
            );
        }
    }
}

// ============================================================================
// 12. Unitary matrix properties: gate * dagger = identity, SPP^2 = PP
// ============================================================================

#[test]
fn unitary_matrix_spp_properties() {
    let identity = UnitaryMatrix::identity(4);
    let tolerance = 1e-10;

    let pairs = [
        (Clifford::SXX, Clifford::SXXdg),
        (Clifford::SYY, Clifford::SYYdg),
        (Clifford::SZZ, Clifford::SZZdg),
    ];

    for (gate, dagger) in &pairs {
        let g_mat = gate.to_matrix();
        let d_mat = dagger.to_matrix();

        // gate * dagger = scalar * I (up to global phase)
        // Note: RXX(3π/2) differs from RXX(π/2)† by a global phase of -1,
        // so we check proportionality to identity rather than exact identity.
        let product = &g_mat * &d_mat;
        let phase = product[(0, 0)];
        let diff = (&product - &(&identity * phase)).norm();
        assert!(
            diff < tolerance,
            "{gate:?} * {dagger:?} should be proportional to identity, diff = {diff}"
        );
        assert!(
            (phase.norm() - 1.0).abs() < tolerance,
            "{gate:?} * {dagger:?} phase should have unit magnitude"
        );

        // dagger * gate = scalar * I
        let product2 = &d_mat * &g_mat;
        let phase2 = product2[(0, 0)];
        let diff2 = (&product2 - &(&identity * phase2)).norm();
        assert!(
            diff2 < tolerance,
            "{dagger:?} * {gate:?} should be proportional to identity, diff = {diff2}"
        );

        // gate^4 = scalar * I
        let g_squared = &g_mat * &g_mat;
        let g_fourth = &g_squared * &g_squared;
        let phase4 = g_fourth[(0, 0)];
        let diff3 = (&g_fourth - &(&identity * phase4)).norm();
        assert!(
            diff3 < tolerance,
            "{gate:?}^4 should be proportional to identity, diff = {diff3}"
        );
    }
}

// ============================================================================
// 13. SparseStabHybrid vs StateVec for all SPP gates
// ============================================================================

#[test]
fn sparse_stab_hybrid_matches_statevec_spp_gates() {
    for gate in SPP_GATES {
        for (name, prep_sv, _, _, _, prep_sh) in input_states() {
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            apply_sv(&mut sv, gate);

            let mut sh = SparseStabHybrid::new(2);
            prep_sh(&mut sh);
            apply_sh(&mut sh, gate);

            let sh_probs = stab_hybrid_probabilities(&sh, 2);
            for (i, &sh_prob) in sh_probs.iter().enumerate() {
                assert_probs_close(
                    sv.probability(i),
                    sh_prob,
                    &format!("{gate:?} on {name}, basis state {i}: SparseStabHybrid vs StateVec"),
                );
            }
        }
    }
}

// ============================================================================
// 14. SparseStabHybrid: gate*dagger=identity and SPP^2=PP
// ============================================================================

#[test]
fn sparse_stab_hybrid_roundtrip_properties() {
    let pairs = [
        (Clifford::SXX, Clifford::SXXdg),
        (Clifford::SYY, Clifford::SYYdg),
        (Clifford::SZZ, Clifford::SZZdg),
    ];

    for (name, _, _, _, _, prep_sh) in input_states() {
        // gate * dagger = identity
        for (gate, dagger) in &pairs {
            let mut ref_sh = SparseStabHybrid::new(2);
            prep_sh(&mut ref_sh);
            let ref_probs = stab_hybrid_probabilities(&ref_sh, 2);

            let mut sh = SparseStabHybrid::new(2);
            prep_sh(&mut sh);
            apply_sh(&mut sh, *gate);
            apply_sh(&mut sh, *dagger);
            let probs = stab_hybrid_probabilities(&sh, 2);
            for i in 0..4 {
                assert_probs_close(
                    ref_probs[i],
                    probs[i],
                    &format!("{gate:?}*{dagger:?}=I SparseStabHybrid on {name}, state {i}"),
                );
            }
        }

        // SPP^2 = PP
        {
            // SXX^2 = XX
            let mut ref_sv = StateVec::new(2);
            let prep_sv = input_states()
                .into_iter()
                .find(|(n, ..)| *n == name)
                .unwrap()
                .1;
            prep_sv(&mut ref_sv);
            ref_sv.x(&qid(0));
            ref_sv.x(&qid(1));

            let mut sh = SparseStabHybrid::new(2);
            prep_sh(&mut sh);
            sh.sxx(&[(QubitId(0), QubitId(1))]);
            sh.sxx(&[(QubitId(0), QubitId(1))]);
            let sh_probs = stab_hybrid_probabilities(&sh, 2);
            for (i, &sh_prob) in sh_probs.iter().enumerate() {
                assert_probs_close(
                    ref_sv.probability(i),
                    sh_prob,
                    &format!("SXX^2=XX SparseStabHybrid on {name}, state {i}"),
                );
            }
        }
        {
            // SYY^2 = YY
            let mut ref_sv = StateVec::new(2);
            let prep_sv = input_states()
                .into_iter()
                .find(|(n, ..)| *n == name)
                .unwrap()
                .1;
            prep_sv(&mut ref_sv);
            ref_sv.y(&qid(0));
            ref_sv.y(&qid(1));

            let mut sh = SparseStabHybrid::new(2);
            prep_sh(&mut sh);
            sh.syy(&[(QubitId(0), QubitId(1))]);
            sh.syy(&[(QubitId(0), QubitId(1))]);
            let sh_probs = stab_hybrid_probabilities(&sh, 2);
            for (i, &sh_prob) in sh_probs.iter().enumerate() {
                assert_probs_close(
                    ref_sv.probability(i),
                    sh_prob,
                    &format!("SYY^2=YY SparseStabHybrid on {name}, state {i}"),
                );
            }
        }
        {
            // SZZ^2 = ZZ
            let mut ref_sv = StateVec::new(2);
            let prep_sv = input_states()
                .into_iter()
                .find(|(n, ..)| *n == name)
                .unwrap()
                .1;
            prep_sv(&mut ref_sv);
            ref_sv.z(&qid(0));
            ref_sv.z(&qid(1));

            let mut sh = SparseStabHybrid::new(2);
            prep_sh(&mut sh);
            sh.szz(&[(QubitId(0), QubitId(1))]);
            sh.szz(&[(QubitId(0), QubitId(1))]);
            let sh_probs = stab_hybrid_probabilities(&sh, 2);
            for (i, &sh_prob) in sh_probs.iter().enumerate() {
                assert_probs_close(
                    ref_sv.probability(i),
                    sh_prob,
                    &format!("SZZ^2=ZZ SparseStabHybrid on {name}, state {i}"),
                );
            }
        }
    }
}

// ============================================================================
// 15. DenseStateVec (StateVecSoA) vs StateVec: validates direct SPP overrides
// ============================================================================

#[test]
fn dense_statevec_matches_statevec_spp_gates() {
    for gate in SPP_GATES {
        for (name, prep_sv, _, _, _, _) in input_states() {
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            apply_sv(&mut sv, gate);

            let mut dsv = DenseStateVec::new(2);
            prep_dsv(&mut dsv, name);
            apply_dsv(&mut dsv, gate);

            for i in 0..4 {
                assert_probs_close(
                    sv.probability(i),
                    dsv.probability(i),
                    &format!("{gate:?} on {name}, basis state {i}: DenseStateVec vs StateVec"),
                );
            }
        }
    }
}

// ============================================================================
// 1q Clifford cross-validation helpers
// ============================================================================

/// All 1q single-qubit input states for testing (on a 2-qubit system, applied to q0).
/// Using 2 qubits lets us verify the gate doesn't corrupt qubit 1.
type StatePrep1q = (
    &'static str,
    fn(&mut StateVec),
    fn(&mut SparseStab),
    fn(&mut SparseStabY),
    fn(&mut SparseStabHybrid),
);

fn input_states_1q() -> Vec<StatePrep1q> {
    vec![
        (
            "|0>",
            |_: &mut StateVec| {},
            |_: &mut SparseStab| {},
            |_: &mut SparseStabY| {},
            |_: &mut SparseStabHybrid| {},
        ),
        (
            "|1>",
            |s: &mut StateVec| {
                s.x(&qid(0));
            },
            |s: &mut SparseStab| {
                s.x(&qid(0));
            },
            |s: &mut SparseStabY| {
                s.x(&qid(0));
            },
            |s: &mut SparseStabHybrid| {
                s.x(&qid(0));
            },
        ),
        (
            "|+>",
            |s: &mut StateVec| {
                s.h(&qid(0));
            },
            |s: &mut SparseStab| {
                s.h(&qid(0));
            },
            |s: &mut SparseStabY| {
                s.h(&qid(0));
            },
            |s: &mut SparseStabHybrid| {
                s.h(&qid(0));
            },
        ),
        (
            "|->",
            |s: &mut StateVec| {
                s.x(&qid(0));
                s.h(&qid(0));
            },
            |s: &mut SparseStab| {
                s.x(&qid(0));
                s.h(&qid(0));
            },
            |s: &mut SparseStabY| {
                s.x(&qid(0));
                s.h(&qid(0));
            },
            |s: &mut SparseStabHybrid| {
                s.x(&qid(0));
                s.h(&qid(0));
            },
        ),
        (
            "|+i>",
            |s: &mut StateVec| {
                s.sx(&qid(0));
            },
            |s: &mut SparseStab| {
                s.sx(&qid(0));
            },
            |s: &mut SparseStabY| {
                s.sx(&qid(0));
            },
            |s: &mut SparseStabHybrid| {
                s.sx(&qid(0));
            },
        ),
        (
            "|-i>",
            |s: &mut StateVec| {
                s.sxdg(&qid(0));
            },
            |s: &mut SparseStab| {
                s.sxdg(&qid(0));
            },
            |s: &mut SparseStabY| {
                s.sxdg(&qid(0));
            },
            |s: &mut SparseStabHybrid| {
                s.sxdg(&qid(0));
            },
        ),
    ]
}

/// Apply a 1q Clifford gate on qubit 0 to each simulator type.
fn apply_1q_sv(sim: &mut StateVec, gate: Clifford) {
    let q = qid(0);
    match gate {
        Clifford::I => {}
        Clifford::X => {
            sim.x(&q);
        }
        Clifford::Y => {
            sim.y(&q);
        }
        Clifford::Z => {
            sim.z(&q);
        }
        Clifford::H => {
            sim.h(&q);
        }
        Clifford::H2 => {
            sim.h2(&q);
        }
        Clifford::H3 => {
            sim.h3(&q);
        }
        Clifford::H4 => {
            sim.h4(&q);
        }
        Clifford::H5 => {
            sim.h5(&q);
        }
        Clifford::H6 => {
            sim.h6(&q);
        }
        Clifford::SX => {
            sim.sx(&q);
        }
        Clifford::SXdg => {
            sim.sxdg(&q);
        }
        Clifford::SY => {
            sim.sy(&q);
        }
        Clifford::SYdg => {
            sim.sydg(&q);
        }
        Clifford::SZ => {
            sim.sz(&q);
        }
        Clifford::SZdg => {
            sim.szdg(&q);
        }
        Clifford::F => {
            sim.f(&q);
        }
        Clifford::Fdg => {
            sim.fdg(&q);
        }
        Clifford::F2 => {
            sim.f2(&q);
        }
        Clifford::F2dg => {
            sim.f2dg(&q);
        }
        Clifford::F3 => {
            sim.f3(&q);
        }
        Clifford::F3dg => {
            sim.f3dg(&q);
        }
        Clifford::F4 => {
            sim.f4(&q);
        }
        Clifford::F4dg => {
            sim.f4dg(&q);
        }
        _ => panic!("not a 1q gate: {gate:?}"),
    }
}

fn apply_1q_ss(sim: &mut SparseStab, gate: Clifford) {
    let q = qid(0);
    match gate {
        Clifford::I => {}
        Clifford::X => {
            sim.x(&q);
        }
        Clifford::Y => {
            sim.y(&q);
        }
        Clifford::Z => {
            sim.z(&q);
        }
        Clifford::H => {
            sim.h(&q);
        }
        Clifford::H2 => {
            sim.h2(&q);
        }
        Clifford::H3 => {
            sim.h3(&q);
        }
        Clifford::H4 => {
            sim.h4(&q);
        }
        Clifford::H5 => {
            sim.h5(&q);
        }
        Clifford::H6 => {
            sim.h6(&q);
        }
        Clifford::SX => {
            sim.sx(&q);
        }
        Clifford::SXdg => {
            sim.sxdg(&q);
        }
        Clifford::SY => {
            sim.sy(&q);
        }
        Clifford::SYdg => {
            sim.sydg(&q);
        }
        Clifford::SZ => {
            sim.sz(&q);
        }
        Clifford::SZdg => {
            sim.szdg(&q);
        }
        Clifford::F => {
            sim.f(&q);
        }
        Clifford::Fdg => {
            sim.fdg(&q);
        }
        Clifford::F2 => {
            sim.f2(&q);
        }
        Clifford::F2dg => {
            sim.f2dg(&q);
        }
        Clifford::F3 => {
            sim.f3(&q);
        }
        Clifford::F3dg => {
            sim.f3dg(&q);
        }
        Clifford::F4 => {
            sim.f4(&q);
        }
        Clifford::F4dg => {
            sim.f4dg(&q);
        }
        _ => panic!("not a 1q gate: {gate:?}"),
    }
}

fn apply_1q_sy(sim: &mut SparseStabY, gate: Clifford) {
    let q = qid(0);
    match gate {
        Clifford::I => {}
        Clifford::X => {
            sim.x(&q);
        }
        Clifford::Y => {
            sim.y(&q);
        }
        Clifford::Z => {
            sim.z(&q);
        }
        Clifford::H => {
            sim.h(&q);
        }
        Clifford::H2 => {
            sim.h2(&q);
        }
        Clifford::H3 => {
            sim.h3(&q);
        }
        Clifford::H4 => {
            sim.h4(&q);
        }
        Clifford::H5 => {
            sim.h5(&q);
        }
        Clifford::H6 => {
            sim.h6(&q);
        }
        Clifford::SX => {
            sim.sx(&q);
        }
        Clifford::SXdg => {
            sim.sxdg(&q);
        }
        Clifford::SY => {
            sim.sy(&q);
        }
        Clifford::SYdg => {
            sim.sydg(&q);
        }
        Clifford::SZ => {
            sim.sz(&q);
        }
        Clifford::SZdg => {
            sim.szdg(&q);
        }
        Clifford::F => {
            sim.f(&q);
        }
        Clifford::Fdg => {
            sim.fdg(&q);
        }
        Clifford::F2 => {
            sim.f2(&q);
        }
        Clifford::F2dg => {
            sim.f2dg(&q);
        }
        Clifford::F3 => {
            sim.f3(&q);
        }
        Clifford::F3dg => {
            sim.f3dg(&q);
        }
        Clifford::F4 => {
            sim.f4(&q);
        }
        Clifford::F4dg => {
            sim.f4dg(&q);
        }
        _ => panic!("not a 1q gate: {gate:?}"),
    }
}

fn apply_1q_sh(sim: &mut SparseStabHybrid, gate: Clifford) {
    let q = qid(0);
    match gate {
        Clifford::I => {}
        Clifford::X => {
            sim.x(&q);
        }
        Clifford::Y => {
            sim.y(&q);
        }
        Clifford::Z => {
            sim.z(&q);
        }
        Clifford::H => {
            sim.h(&q);
        }
        Clifford::H2 => {
            sim.h2(&q);
        }
        Clifford::H3 => {
            sim.h3(&q);
        }
        Clifford::H4 => {
            sim.h4(&q);
        }
        Clifford::H5 => {
            sim.h5(&q);
        }
        Clifford::H6 => {
            sim.h6(&q);
        }
        Clifford::SX => {
            sim.sx(&q);
        }
        Clifford::SXdg => {
            sim.sxdg(&q);
        }
        Clifford::SY => {
            sim.sy(&q);
        }
        Clifford::SYdg => {
            sim.sydg(&q);
        }
        Clifford::SZ => {
            sim.sz(&q);
        }
        Clifford::SZdg => {
            sim.szdg(&q);
        }
        Clifford::F => {
            sim.f(&q);
        }
        Clifford::Fdg => {
            sim.fdg(&q);
        }
        Clifford::F2 => {
            sim.f2(&q);
        }
        Clifford::F2dg => {
            sim.f2dg(&q);
        }
        Clifford::F3 => {
            sim.f3(&q);
        }
        Clifford::F3dg => {
            sim.f3dg(&q);
        }
        Clifford::F4 => {
            sim.f4(&q);
        }
        Clifford::F4dg => {
            sim.f4dg(&q);
        }
        _ => panic!("not a 1q gate: {gate:?}"),
    }
}

// ============================================================================
// 1q Clifford cross-validation: SparseStab vs StateVec
// ============================================================================

#[test]
fn sparse_stab_matches_statevec_all_1q_cliffords() {
    for &gate in Clifford::all_1q() {
        for (name, prep_sv, prep_ss, _, _) in input_states_1q() {
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            apply_1q_sv(&mut sv, gate);

            let mut ss = SparseStab::new(2);
            prep_ss(&mut ss);
            apply_1q_ss(&mut ss, gate);

            let ss_probs = stab_probabilities(&ss, 2);
            for (i, &ss_prob) in ss_probs.iter().enumerate() {
                assert_probs_close(
                    sv.probability(i),
                    ss_prob,
                    &format!("{gate:?} on {name}, basis state {i}: SparseStab vs StateVec"),
                );
            }
        }
    }
}

// ============================================================================
// 1q Clifford cross-validation: SparseStabY vs StateVec
// ============================================================================

#[test]
fn sparse_stab_y_matches_statevec_all_1q_cliffords() {
    for &gate in Clifford::all_1q() {
        for (name, prep_sv, _, prep_sy, _) in input_states_1q() {
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            apply_1q_sv(&mut sv, gate);

            let mut sy = SparseStabY::new(2);
            prep_sy(&mut sy);
            apply_1q_sy(&mut sy, gate);

            let sy_probs = stab_y_probabilities(&sy, 2);
            for (i, &sy_prob) in sy_probs.iter().enumerate() {
                assert_probs_close(
                    sv.probability(i),
                    sy_prob,
                    &format!("{gate:?} on {name}, basis state {i}: SparseStabY vs StateVec"),
                );
            }
        }
    }
}

// ============================================================================
// 1q Clifford cross-validation: SparseStab vs SparseStabY
// ============================================================================

#[test]
fn sparse_stab_matches_sparse_stab_y_all_1q_cliffords() {
    for &gate in Clifford::all_1q() {
        for (name, _, prep_ss, prep_sy, _) in input_states_1q() {
            let mut ss = SparseStab::new(2);
            prep_ss(&mut ss);
            apply_1q_ss(&mut ss, gate);

            let mut sy = SparseStabY::new(2);
            prep_sy(&mut sy);
            apply_1q_sy(&mut sy, gate);

            let ss_probs = stab_probabilities(&ss, 2);
            let sy_probs = stab_y_probabilities(&sy, 2);
            for i in 0..4 {
                assert_probs_close(
                    ss_probs[i],
                    sy_probs[i],
                    &format!("{gate:?} on {name}, basis state {i}: SparseStab vs SparseStabY"),
                );
            }
        }
    }
}

// ============================================================================
// 1q Clifford cross-validation: SparseStabHybrid vs StateVec
// ============================================================================

#[test]
fn sparse_stab_hybrid_matches_statevec_all_1q_cliffords() {
    for &gate in Clifford::all_1q() {
        for (name, prep_sv, _, _, prep_sh) in input_states_1q() {
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            apply_1q_sv(&mut sv, gate);

            let mut sh = SparseStabHybrid::new(2);
            prep_sh(&mut sh);
            apply_1q_sh(&mut sh, gate);

            let sh_probs = stab_hybrid_probabilities(&sh, 2);
            for (i, &sh_prob) in sh_probs.iter().enumerate() {
                assert_probs_close(
                    sv.probability(i),
                    sh_prob,
                    &format!("{gate:?} on {name}, basis state {i}: SparseStabHybrid vs StateVec"),
                );
            }
        }
    }
}

// ============================================================================
// 1q Clifford: deterministic measurements agree across all stab sims
// ============================================================================

#[test]
fn deterministic_measurements_agree_all_stab_sims_1q_cliffords() {
    for &gate in Clifford::all_1q() {
        for (name, _, prep_ss, prep_sy, prep_sh) in input_states_1q() {
            let mut ss = SparseStab::new(2);
            prep_ss(&mut ss);
            apply_1q_ss(&mut ss, gate);

            let mut sy = SparseStabY::new(2);
            prep_sy(&mut sy);
            apply_1q_sy(&mut sy, gate);

            let mut sh = SparseStabHybrid::new(2);
            prep_sh(&mut sh);
            apply_1q_sh(&mut sh, gate);

            for q in 0..2 {
                let mut ss_copy = ss.clone();
                let mut sy_copy = sy.clone();
                let mut sh_copy = sh.clone();
                let ss_result = ss_copy.mz(&qid(q));
                let sy_result = sy_copy.mz(&qid(q));
                let sh_result = sh_copy.mz(&qid(q));

                assert_eq!(
                    ss_result[0].is_deterministic, sy_result[0].is_deterministic,
                    "{gate:?} on {name}, qubit {q}: determinism mismatch (SS vs SY)"
                );
                assert_eq!(
                    ss_result[0].is_deterministic, sh_result[0].is_deterministic,
                    "{gate:?} on {name}, qubit {q}: determinism mismatch (SS vs SH)"
                );

                if ss_result[0].is_deterministic {
                    assert_eq!(
                        ss_result[0].outcome, sy_result[0].outcome,
                        "{gate:?} on {name}, qubit {q}: deterministic outcome mismatch (SS vs SY)"
                    );
                    assert_eq!(
                        ss_result[0].outcome, sh_result[0].outcome,
                        "{gate:?} on {name}, qubit {q}: deterministic outcome mismatch (SS vs SH)"
                    );
                }
            }
        }
    }
}

// ============================================================================
// 1q Clifford: gate then dagger is identity
// ============================================================================

#[test]
fn gate_then_dagger_identity_all_1q_cliffords() {
    // Pairs of (gate, dagger)
    let pairs = [
        (Clifford::SX, Clifford::SXdg),
        (Clifford::SY, Clifford::SYdg),
        (Clifford::SZ, Clifford::SZdg),
        (Clifford::H, Clifford::H),   // H is self-inverse
        (Clifford::H2, Clifford::H2), // H2 is self-inverse
        (Clifford::H3, Clifford::H3),
        (Clifford::H4, Clifford::H4),
        (Clifford::H5, Clifford::H5),
        (Clifford::H6, Clifford::H6),
        (Clifford::F, Clifford::Fdg),
        (Clifford::F2, Clifford::F2dg),
        (Clifford::F3, Clifford::F3dg),
        (Clifford::F4, Clifford::F4dg),
    ];

    for (name, prep_sv, prep_ss, prep_sy, prep_sh) in input_states_1q() {
        for (gate, dagger) in &pairs {
            // StateVec
            let mut sv_ref = StateVec::new(2);
            prep_sv(&mut sv_ref);
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            apply_1q_sv(&mut sv, *gate);
            apply_1q_sv(&mut sv, *dagger);
            for i in 0..4 {
                assert_probs_close(
                    sv_ref.probability(i),
                    sv.probability(i),
                    &format!("{gate:?}*{dagger:?} on {name}: StateVec not identity"),
                );
            }

            // SparseStab
            let ss_probs_ref = {
                let mut ss = SparseStab::new(2);
                prep_ss(&mut ss);
                stab_probabilities(&ss, 2)
            };
            let ss_probs = {
                let mut ss = SparseStab::new(2);
                prep_ss(&mut ss);
                apply_1q_ss(&mut ss, *gate);
                apply_1q_ss(&mut ss, *dagger);
                stab_probabilities(&ss, 2)
            };
            for i in 0..4 {
                assert_probs_close(
                    ss_probs_ref[i],
                    ss_probs[i],
                    &format!("{gate:?}*{dagger:?} on {name}: SparseStab not identity"),
                );
            }

            // SparseStabY
            let sy_probs_ref = {
                let mut sy = SparseStabY::new(2);
                prep_sy(&mut sy);
                stab_y_probabilities(&sy, 2)
            };
            let sy_probs = {
                let mut sy = SparseStabY::new(2);
                prep_sy(&mut sy);
                apply_1q_sy(&mut sy, *gate);
                apply_1q_sy(&mut sy, *dagger);
                stab_y_probabilities(&sy, 2)
            };
            for i in 0..4 {
                assert_probs_close(
                    sy_probs_ref[i],
                    sy_probs[i],
                    &format!("{gate:?}*{dagger:?} on {name}: SparseStabY not identity"),
                );
            }

            // SparseStabHybrid
            let sh_probs_ref = {
                let mut sh = SparseStabHybrid::new(2);
                prep_sh(&mut sh);
                stab_hybrid_probabilities(&sh, 2)
            };
            let sh_probs = {
                let mut sh = SparseStabHybrid::new(2);
                prep_sh(&mut sh);
                apply_1q_sh(&mut sh, *gate);
                apply_1q_sh(&mut sh, *dagger);
                stab_hybrid_probabilities(&sh, 2)
            };
            for i in 0..4 {
                assert_probs_close(
                    sh_probs_ref[i],
                    sh_probs[i],
                    &format!("{gate:?}*{dagger:?} on {name}: SparseStabHybrid not identity"),
                );
            }
        }
    }
}

// ============================================================================
// 2q Clifford cross-validation helpers (all 14 gates)
// ============================================================================

fn apply_2q_sv(sim: &mut StateVec, gate: Clifford) {
    let q = [(QubitId(0), QubitId(1))];
    match gate {
        Clifford::CX => {
            sim.cx(&q);
        }
        Clifford::CY => {
            sim.cy(&q);
        }
        Clifford::CZ => {
            sim.cz(&q);
        }
        Clifford::SWAP => {
            sim.swap(&q);
        }
        Clifford::SXX => {
            sim.sxx(&q);
        }
        Clifford::SXXdg => {
            sim.sxxdg(&q);
        }
        Clifford::SYY => {
            sim.syy(&q);
        }
        Clifford::SYYdg => {
            sim.syydg(&q);
        }
        Clifford::SZZ => {
            sim.szz(&q);
        }
        Clifford::SZZdg => {
            sim.szzdg(&q);
        }
        Clifford::ISWAP => {
            sim.iswap(&q);
        }
        Clifford::ISWAPdg => {
            sim.iswapdg(&q);
        }
        Clifford::G => {
            sim.g(&q);
        }
        Clifford::Gdg => {
            sim.gdg(&q);
        }
        _ => panic!("not a 2q gate: {gate:?}"),
    }
}

fn apply_2q_ss(sim: &mut SparseStab, gate: Clifford) {
    let q = [(QubitId(0), QubitId(1))];
    match gate {
        Clifford::CX => {
            sim.cx(&q);
        }
        Clifford::CY => {
            sim.cy(&q);
        }
        Clifford::CZ => {
            sim.cz(&q);
        }
        Clifford::SWAP => {
            sim.swap(&q);
        }
        Clifford::SXX => {
            sim.sxx(&q);
        }
        Clifford::SXXdg => {
            sim.sxxdg(&q);
        }
        Clifford::SYY => {
            sim.syy(&q);
        }
        Clifford::SYYdg => {
            sim.syydg(&q);
        }
        Clifford::SZZ => {
            sim.szz(&q);
        }
        Clifford::SZZdg => {
            sim.szzdg(&q);
        }
        Clifford::ISWAP => {
            sim.iswap(&q);
        }
        Clifford::ISWAPdg => {
            sim.iswapdg(&q);
        }
        Clifford::G => {
            sim.g(&q);
        }
        Clifford::Gdg => {
            sim.gdg(&q);
        }
        _ => panic!("not a 2q gate: {gate:?}"),
    }
}

fn apply_2q_sy_all(sim: &mut SparseStabY, gate: Clifford) {
    let q = [(QubitId(0), QubitId(1))];
    match gate {
        Clifford::CX => {
            sim.cx(&q);
        }
        Clifford::CY => {
            sim.cy(&q);
        }
        Clifford::CZ => {
            sim.cz(&q);
        }
        Clifford::SWAP => {
            sim.swap(&q);
        }
        Clifford::SXX => {
            sim.sxx(&q);
        }
        Clifford::SXXdg => {
            sim.sxxdg(&q);
        }
        Clifford::SYY => {
            sim.syy(&q);
        }
        Clifford::SYYdg => {
            sim.syydg(&q);
        }
        Clifford::SZZ => {
            sim.szz(&q);
        }
        Clifford::SZZdg => {
            sim.szzdg(&q);
        }
        Clifford::ISWAP => {
            sim.iswap(&q);
        }
        Clifford::ISWAPdg => {
            sim.iswapdg(&q);
        }
        Clifford::G => {
            sim.g(&q);
        }
        Clifford::Gdg => {
            sim.gdg(&q);
        }
        _ => panic!("not a 2q gate: {gate:?}"),
    }
}

fn apply_2q_sh_all(sim: &mut SparseStabHybrid, gate: Clifford) {
    let q = [(QubitId(0), QubitId(1))];
    match gate {
        Clifford::CX => {
            sim.cx(&q);
        }
        Clifford::CY => {
            sim.cy(&q);
        }
        Clifford::CZ => {
            sim.cz(&q);
        }
        Clifford::SWAP => {
            sim.swap(&q);
        }
        Clifford::SXX => {
            sim.sxx(&q);
        }
        Clifford::SXXdg => {
            sim.sxxdg(&q);
        }
        Clifford::SYY => {
            sim.syy(&q);
        }
        Clifford::SYYdg => {
            sim.syydg(&q);
        }
        Clifford::SZZ => {
            sim.szz(&q);
        }
        Clifford::SZZdg => {
            sim.szzdg(&q);
        }
        Clifford::ISWAP => {
            sim.iswap(&q);
        }
        Clifford::ISWAPdg => {
            sim.iswapdg(&q);
        }
        Clifford::G => {
            sim.g(&q);
        }
        Clifford::Gdg => {
            sim.gdg(&q);
        }
        _ => panic!("not a 2q gate: {gate:?}"),
    }
}

// ============================================================================
// 2q Clifford cross-validation: SparseStab vs StateVec (all 14 gates)
// ============================================================================

#[test]
fn sparse_stab_matches_statevec_all_2q_cliffords() {
    for &gate in Clifford::all_2q() {
        for (name, prep_sv, _, prep_ss, _, _) in input_states() {
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            apply_2q_sv(&mut sv, gate);

            let mut ss = SparseStab::new(2);
            prep_ss(&mut ss);
            apply_2q_ss(&mut ss, gate);

            let ss_probs = stab_probabilities(&ss, 2);
            for (i, &ss_prob) in ss_probs.iter().enumerate() {
                assert_probs_close(
                    sv.probability(i),
                    ss_prob,
                    &format!("{gate:?} on {name}, basis state {i}: SparseStab vs StateVec"),
                );
            }
        }
    }
}

// ============================================================================
// 2q Clifford cross-validation: SparseStabY vs StateVec (all 14 gates)
// ============================================================================

#[test]
fn sparse_stab_y_matches_statevec_all_2q_cliffords() {
    for &gate in Clifford::all_2q() {
        for (name, prep_sv, _, _, prep_sy, _) in input_states() {
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            apply_2q_sv(&mut sv, gate);

            let mut sy = SparseStabY::new(2);
            prep_sy(&mut sy);
            apply_2q_sy_all(&mut sy, gate);

            let sy_probs = stab_y_probabilities(&sy, 2);
            for (i, &sy_prob) in sy_probs.iter().enumerate() {
                assert_probs_close(
                    sv.probability(i),
                    sy_prob,
                    &format!("{gate:?} on {name}, basis state {i}: SparseStabY vs StateVec"),
                );
            }
        }
    }
}

// ============================================================================
// 2q Clifford cross-validation: SparseStab vs SparseStabY (all 14 gates)
// ============================================================================

#[test]
fn sparse_stab_matches_sparse_stab_y_all_2q_cliffords() {
    for &gate in Clifford::all_2q() {
        for (name, _, _, prep_ss, prep_sy, _) in input_states() {
            let mut ss = SparseStab::new(2);
            prep_ss(&mut ss);
            apply_2q_ss(&mut ss, gate);

            let mut sy = SparseStabY::new(2);
            prep_sy(&mut sy);
            apply_2q_sy_all(&mut sy, gate);

            let ss_probs = stab_probabilities(&ss, 2);
            let sy_probs = stab_y_probabilities(&sy, 2);
            for i in 0..4 {
                assert_probs_close(
                    ss_probs[i],
                    sy_probs[i],
                    &format!("{gate:?} on {name}, basis state {i}: SparseStab vs SparseStabY"),
                );
            }
        }
    }
}

// ============================================================================
// 2q Clifford cross-validation: SparseStabHybrid vs StateVec (all 14 gates)
// ============================================================================

#[test]
fn sparse_stab_hybrid_matches_statevec_all_2q_cliffords() {
    for &gate in Clifford::all_2q() {
        for (name, prep_sv, _, _, _, prep_sh) in input_states() {
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            apply_2q_sv(&mut sv, gate);

            let mut sh = SparseStabHybrid::new(2);
            prep_sh(&mut sh);
            apply_2q_sh_all(&mut sh, gate);

            let sh_probs = stab_hybrid_probabilities(&sh, 2);
            for (i, &sh_prob) in sh_probs.iter().enumerate() {
                assert_probs_close(
                    sv.probability(i),
                    sh_prob,
                    &format!("{gate:?} on {name}, basis state {i}: SparseStabHybrid vs StateVec"),
                );
            }
        }
    }
}

// ============================================================================
// 2q Clifford: deterministic measurements agree across all stab sims
// ============================================================================

#[test]
fn deterministic_measurements_agree_all_stab_sims_2q_cliffords() {
    for &gate in Clifford::all_2q() {
        for (name, _, _, prep_ss, prep_sy, prep_sh) in input_states() {
            let mut ss = SparseStab::new(2);
            prep_ss(&mut ss);
            apply_2q_ss(&mut ss, gate);

            let mut sy = SparseStabY::new(2);
            prep_sy(&mut sy);
            apply_2q_sy_all(&mut sy, gate);

            let mut sh = SparseStabHybrid::new(2);
            prep_sh(&mut sh);
            apply_2q_sh_all(&mut sh, gate);

            for q in 0..2 {
                let mut ss_copy = ss.clone();
                let mut sy_copy = sy.clone();
                let mut sh_copy = sh.clone();
                let ss_result = ss_copy.mz(&qid(q));
                let sy_result = sy_copy.mz(&qid(q));
                let sh_result = sh_copy.mz(&qid(q));

                assert_eq!(
                    ss_result[0].is_deterministic, sy_result[0].is_deterministic,
                    "{gate:?} on {name}, qubit {q}: determinism mismatch (SS vs SY)"
                );
                assert_eq!(
                    ss_result[0].is_deterministic, sh_result[0].is_deterministic,
                    "{gate:?} on {name}, qubit {q}: determinism mismatch (SS vs SH)"
                );

                if ss_result[0].is_deterministic {
                    assert_eq!(
                        ss_result[0].outcome, sy_result[0].outcome,
                        "{gate:?} on {name}, qubit {q}: deterministic outcome mismatch (SS vs SY)"
                    );
                    assert_eq!(
                        ss_result[0].outcome, sh_result[0].outcome,
                        "{gate:?} on {name}, qubit {q}: deterministic outcome mismatch (SS vs SH)"
                    );
                }
            }
        }
    }
}

// ============================================================================
// 2q Clifford: gate then dagger is identity (all 14 gates)
// ============================================================================

#[test]
fn gate_then_dagger_identity_all_2q_cliffords() {
    let pairs = [
        (Clifford::CX, Clifford::CX),     // CX is self-inverse
        (Clifford::CY, Clifford::CY),     // CY is self-inverse
        (Clifford::CZ, Clifford::CZ),     // CZ is self-inverse
        (Clifford::SWAP, Clifford::SWAP), // SWAP is self-inverse
        (Clifford::SXX, Clifford::SXXdg),
        (Clifford::SYY, Clifford::SYYdg),
        (Clifford::SZZ, Clifford::SZZdg),
        (Clifford::ISWAP, Clifford::ISWAPdg),
        (Clifford::G, Clifford::Gdg),
    ];

    for (name, prep_sv, _, prep_ss, prep_sy, prep_sh) in input_states() {
        for (gate, dagger) in &pairs {
            // StateVec
            let mut sv_ref = StateVec::new(2);
            prep_sv(&mut sv_ref);
            let mut sv = StateVec::new(2);
            prep_sv(&mut sv);
            apply_2q_sv(&mut sv, *gate);
            apply_2q_sv(&mut sv, *dagger);
            for i in 0..4 {
                assert_probs_close(
                    sv_ref.probability(i),
                    sv.probability(i),
                    &format!("{gate:?}*{dagger:?} on {name}: StateVec not identity"),
                );
            }

            // SparseStab
            let ss_probs_ref = {
                let mut ss = SparseStab::new(2);
                prep_ss(&mut ss);
                stab_probabilities(&ss, 2)
            };
            let ss_probs = {
                let mut ss = SparseStab::new(2);
                prep_ss(&mut ss);
                apply_2q_ss(&mut ss, *gate);
                apply_2q_ss(&mut ss, *dagger);
                stab_probabilities(&ss, 2)
            };
            for i in 0..4 {
                assert_probs_close(
                    ss_probs_ref[i],
                    ss_probs[i],
                    &format!("{gate:?}*{dagger:?} on {name}: SparseStab not identity"),
                );
            }

            // SparseStabY
            let sy_probs_ref = {
                let mut sy = SparseStabY::new(2);
                prep_sy(&mut sy);
                stab_y_probabilities(&sy, 2)
            };
            let sy_probs = {
                let mut sy = SparseStabY::new(2);
                prep_sy(&mut sy);
                apply_2q_sy_all(&mut sy, *gate);
                apply_2q_sy_all(&mut sy, *dagger);
                stab_y_probabilities(&sy, 2)
            };
            for i in 0..4 {
                assert_probs_close(
                    sy_probs_ref[i],
                    sy_probs[i],
                    &format!("{gate:?}*{dagger:?} on {name}: SparseStabY not identity"),
                );
            }

            // SparseStabHybrid
            let sh_probs_ref = {
                let mut sh = SparseStabHybrid::new(2);
                prep_sh(&mut sh);
                stab_hybrid_probabilities(&sh, 2)
            };
            let sh_probs = {
                let mut sh = SparseStabHybrid::new(2);
                prep_sh(&mut sh);
                apply_2q_sh_all(&mut sh, *gate);
                apply_2q_sh_all(&mut sh, *dagger);
                stab_hybrid_probabilities(&sh, 2)
            };
            for i in 0..4 {
                assert_probs_close(
                    sh_probs_ref[i],
                    sh_probs[i],
                    &format!("{gate:?}*{dagger:?} on {name}: SparseStabHybrid not identity"),
                );
            }
        }
    }
}

/// Apply the named state prep to a `DenseStateVec`.
fn prep_dsv(sim: &mut DenseStateVec, name: &str) {
    match name {
        "|00>" => {}
        "|10>" => {
            sim.x(&qid(0));
        }
        "|01>" => {
            sim.x(&qid(1));
        }
        "|11>" => {
            sim.x(&qid(0));
            sim.x(&qid(1));
        }
        "|++>" => {
            sim.h(&qid(0));
            sim.h(&qid(1));
        }
        "|+->" => {
            sim.h(&qid(0));
            sim.x(&qid(1));
            sim.h(&qid(1));
        }
        "Bell |00>+|11>" => {
            sim.h(&qid(0));
            sim.cx(&[(QubitId(0), QubitId(1))]);
        }
        "|0,+i> (Y eigenstate at q1)" => {
            sim.sx(&qid(1));
        }
        "|+i,+i> (Y eigenstates both)" => {
            sim.sx(&qid(0));
            sim.sx(&qid(1));
        }
        "|+i,-i> (opposite Y eigenstates)" => {
            sim.sx(&qid(0));
            sim.sxdg(&qid(1));
        }
        "|+,+i> (X at q0, Y at q1)" => {
            sim.h(&qid(0));
            sim.sx(&qid(1));
        }
        _ => panic!("unknown state prep: {name}"),
    }
}

// ============================================================================
// Unitary conjugation: U * P * U† via matrix algebra vs stabilizer sims
// ============================================================================

/// Build a Pauli matrix on a specific qubit in an n-qubit system.
/// Little-endian: qubit 0 is the rightmost (LSB) tensor factor.
fn pauli_on_qubit(p: pecos_core::Pauli, qubit: usize, num_qubits: usize) -> UnitaryMatrix {
    use pecos_quantum::unitary_matrix::ToMatrix;
    let mut mat = UnitaryMatrix::identity(1);
    for q in (0..num_qubits).rev() {
        if q == qubit {
            mat = &mat & &p.to_matrix();
        } else {
            mat = &mat & &UnitaryMatrix::identity(2);
        }
    }
    mat
}

/// Given a matrix that is a Pauli operator (up to global phase),
/// identify its XZ-decomposition: matrix = phase * `X^{x_bits`} `Z^{z_bits`}.
fn identify_pauli_xz(
    mat: &UnitaryMatrix,
    num_qubits: usize,
) -> (Vec<bool>, Vec<bool>, num_complex::Complex64) {
    let dim = 1usize << num_qubits;
    assert_eq!(mat.nrows(), dim);

    // Find x_mask from row 0's nonzero column
    let mut x_mask = 0;
    for j in 0..dim {
        if mat[(0, j)].norm() > 1e-10 {
            x_mask = j;
            break;
        }
    }

    // phase = M[x_mask, 0] (since P|0> = phase * |x_mask> and (-1)^{z AND 0} = 1)
    let phase = mat[(x_mask, 0)];
    assert!(
        (phase.norm() - 1.0).abs() < 1e-10,
        "phase should have unit magnitude, got {phase:?}"
    );

    let mut x_bits = vec![false; num_qubits];
    let mut z_bits = vec![false; num_qubits];

    for q in 0..num_qubits {
        x_bits[q] = (x_mask >> q) & 1 == 1;

        // P|(1<<q)> has nonzero entry at row (1<<q) XOR x_mask.
        // The value is phase * (-1)^{z_q}.
        let row = (1 << q) ^ x_mask;
        let col = 1 << q;
        let ratio = mat[(row, col)] / phase;
        if (ratio.re + 1.0).abs() < 1e-10 && ratio.im.abs() < 1e-10 {
            z_bits[q] = true;
        } else {
            assert!(
                (ratio.re - 1.0).abs() < 1e-10 && ratio.im.abs() < 1e-10,
                "expected ratio +1 or -1, got {ratio:?} for qubit {q}"
            );
        }
    }

    (x_bits, z_bits, phase)
}

/// Map a Complex64 phase (+1, -1, +i, or -i) to (`signs_minus`, `signs_i`).
fn phase_to_sign_bits(phase: num_complex::Complex64) -> (bool, bool) {
    let minus = phase.re < -0.5 || phase.im < -0.5;
    let has_i = phase.im.abs() > 0.5;
    (minus, has_i)
}

#[test]
fn unitary_conjugation_matches_all_stab_sims_all_cliffords() {
    let minus_i = num_complex::Complex64::new(0.0, -1.0);

    // -- 1q Cliffords --
    for &gate in Clifford::all_1q() {
        let u_mat = gate.to_matrix();
        let u_dag = u_mat.adjoint();
        let num_qubits = 1;

        for (input_pauli, init_x) in [(pecos_core::Pauli::X, true), (pecos_core::Pauli::Z, false)] {
            let p_mat = pauli_on_qubit(input_pauli, 0, num_qubits);
            let conjugated = &u_mat * &p_mat * &u_dag;
            let (x_bits, z_bits, xz_phase) = identify_pauli_xz(&conjugated, num_qubits);

            let gen_id = 0usize;

            // W-convention signs (for SparseStab and SparseStabHybrid)
            let (w_minus, w_i) = phase_to_sign_bits(xz_phase);

            // Y-convention signs: y_phase = xz_phase * (-i)^{num_ys}
            let num_ys = (0..num_qubits).filter(|&q| x_bits[q] && z_bits[q]).count();
            let mut y_phase = xz_phase;
            for _ in 0..num_ys {
                y_phase *= minus_i;
            }
            let (y_minus, y_i) = phase_to_sign_bits(y_phase);

            // SparseStab (W-convention)
            {
                let mut ss = SparseStab::new(num_qubits);
                if init_x {
                    ss.h(&qid(0));
                }
                apply_1q_ss(&mut ss, gate);

                for q in 0..num_qubits {
                    assert_eq!(
                        ss.stabs().col_x[q].contains(gen_id),
                        x_bits[q],
                        "1q {gate:?} on {input_pauli:?}: SS qubit {q} X-bit"
                    );
                    assert_eq!(
                        ss.stabs().col_z[q].contains(gen_id),
                        z_bits[q],
                        "1q {gate:?} on {input_pauli:?}: SS qubit {q} Z-bit"
                    );
                }
                assert_eq!(
                    ss.stabs().signs_minus.contains(gen_id),
                    w_minus,
                    "1q {gate:?} on {input_pauli:?}: SS signs_minus (xz_phase={xz_phase:?})"
                );
                assert_eq!(
                    ss.stabs().signs_i.contains(gen_id),
                    w_i,
                    "1q {gate:?} on {input_pauli:?}: SS signs_i (xz_phase={xz_phase:?})"
                );
            }

            // SparseStabY (Y-convention)
            {
                let mut sy = SparseStabY::new(num_qubits);
                if init_x {
                    sy.h(&qid(0));
                }
                apply_1q_sy(&mut sy, gate);

                for q in 0..num_qubits {
                    assert_eq!(
                        sy.stabs().col_x[q].contains(gen_id),
                        x_bits[q],
                        "1q {gate:?} on {input_pauli:?}: SY qubit {q} X-bit"
                    );
                    assert_eq!(
                        sy.stabs().col_z[q].contains(gen_id),
                        z_bits[q],
                        "1q {gate:?} on {input_pauli:?}: SY qubit {q} Z-bit"
                    );
                }
                assert_eq!(
                    sy.stabs().signs_minus.contains(gen_id),
                    y_minus,
                    "1q {gate:?} on {input_pauli:?}: SY signs_minus (y_phase={y_phase:?})"
                );
                assert_eq!(
                    sy.stabs().signs_i.contains(gen_id),
                    y_i,
                    "1q {gate:?} on {input_pauli:?}: SY signs_i (y_phase={y_phase:?})"
                );
            }

            // SparseStabHybrid (W-convention)
            {
                let mut sh = SparseStabHybrid::new(num_qubits);
                if init_x {
                    sh.h(&qid(0));
                }
                apply_1q_sh(&mut sh, gate);

                for q in 0..num_qubits {
                    assert_eq!(
                        sh.stabs().col_x[q].contains(&gen_id),
                        x_bits[q],
                        "1q {gate:?} on {input_pauli:?}: SH qubit {q} X-bit"
                    );
                    assert_eq!(
                        sh.stabs().col_z[q].contains(&gen_id),
                        z_bits[q],
                        "1q {gate:?} on {input_pauli:?}: SH qubit {q} Z-bit"
                    );
                }
                assert_eq!(
                    sh.stabs().signs_minus.contains(gen_id),
                    w_minus,
                    "1q {gate:?} on {input_pauli:?}: SH signs_minus (xz_phase={xz_phase:?})"
                );
                assert_eq!(
                    sh.stabs().signs_i.contains(gen_id),
                    w_i,
                    "1q {gate:?} on {input_pauli:?}: SH signs_i (xz_phase={xz_phase:?})"
                );
            }
        }
    }

    // -- 2q Cliffords --
    for &gate in Clifford::all_2q() {
        let u_mat = gate.to_matrix();
        let u_dag = u_mat.adjoint();
        let num_qubits = 2;

        let inputs: [(pecos_core::Pauli, usize, bool); 4] = [
            (pecos_core::Pauli::X, 0, true),
            (pecos_core::Pauli::Z, 0, false),
            (pecos_core::Pauli::X, 1, true),
            (pecos_core::Pauli::Z, 1, false),
        ];

        for (input_pauli, input_q, init_x) in &inputs {
            let p_mat = pauli_on_qubit(*input_pauli, *input_q, num_qubits);
            let conjugated = &u_mat * &p_mat * &u_dag;
            let (x_bits, z_bits, xz_phase) = identify_pauli_xz(&conjugated, num_qubits);

            let gen_id = *input_q;

            let (w_minus, w_i) = phase_to_sign_bits(xz_phase);

            let num_ys = (0..num_qubits).filter(|&q| x_bits[q] && z_bits[q]).count();
            let mut y_phase = xz_phase;
            for _ in 0..num_ys {
                y_phase *= minus_i;
            }
            let (y_minus, y_i) = phase_to_sign_bits(y_phase);

            // SparseStab (W-convention)
            {
                let mut ss = SparseStab::new(num_qubits);
                if *init_x {
                    ss.h(&qid(*input_q));
                }
                apply_2q_ss(&mut ss, gate);

                for q in 0..num_qubits {
                    assert_eq!(
                        ss.stabs().col_x[q].contains(gen_id),
                        x_bits[q],
                        "2q {gate:?} on {input_pauli:?}_{input_q}: SS qubit {q} X-bit"
                    );
                    assert_eq!(
                        ss.stabs().col_z[q].contains(gen_id),
                        z_bits[q],
                        "2q {gate:?} on {input_pauli:?}_{input_q}: SS qubit {q} Z-bit"
                    );
                }
                assert_eq!(
                    ss.stabs().signs_minus.contains(gen_id),
                    w_minus,
                    "2q {gate:?} on {input_pauli:?}_{input_q}: SS signs_minus (xz_phase={xz_phase:?})"
                );
                assert_eq!(
                    ss.stabs().signs_i.contains(gen_id),
                    w_i,
                    "2q {gate:?} on {input_pauli:?}_{input_q}: SS signs_i (xz_phase={xz_phase:?})"
                );
            }

            // SparseStabY (Y-convention)
            {
                let mut sy = SparseStabY::new(num_qubits);
                if *init_x {
                    sy.h(&qid(*input_q));
                }
                apply_2q_sy_all(&mut sy, gate);

                for q in 0..num_qubits {
                    assert_eq!(
                        sy.stabs().col_x[q].contains(gen_id),
                        x_bits[q],
                        "2q {gate:?} on {input_pauli:?}_{input_q}: SY qubit {q} X-bit"
                    );
                    assert_eq!(
                        sy.stabs().col_z[q].contains(gen_id),
                        z_bits[q],
                        "2q {gate:?} on {input_pauli:?}_{input_q}: SY qubit {q} Z-bit"
                    );
                }
                assert_eq!(
                    sy.stabs().signs_minus.contains(gen_id),
                    y_minus,
                    "2q {gate:?} on {input_pauli:?}_{input_q}: SY signs_minus (y_phase={y_phase:?})"
                );
                assert_eq!(
                    sy.stabs().signs_i.contains(gen_id),
                    y_i,
                    "2q {gate:?} on {input_pauli:?}_{input_q}: SY signs_i (y_phase={y_phase:?})"
                );
            }

            // SparseStabHybrid (W-convention)
            {
                let mut sh = SparseStabHybrid::new(num_qubits);
                if *init_x {
                    sh.h(&qid(*input_q));
                }
                apply_2q_sh_all(&mut sh, gate);

                for q in 0..num_qubits {
                    assert_eq!(
                        sh.stabs().col_x[q].contains(&gen_id),
                        x_bits[q],
                        "2q {gate:?} on {input_pauli:?}_{input_q}: SH qubit {q} X-bit"
                    );
                    assert_eq!(
                        sh.stabs().col_z[q].contains(&gen_id),
                        z_bits[q],
                        "2q {gate:?} on {input_pauli:?}_{input_q}: SH qubit {q} Z-bit"
                    );
                }
                assert_eq!(
                    sh.stabs().signs_minus.contains(gen_id),
                    w_minus,
                    "2q {gate:?} on {input_pauli:?}_{input_q}: SH signs_minus (xz_phase={xz_phase:?})"
                );
                assert_eq!(
                    sh.stabs().signs_i.contains(gen_id),
                    w_i,
                    "2q {gate:?} on {input_pauli:?}_{input_q}: SH signs_i (xz_phase={xz_phase:?})"
                );
            }
        }
    }
}
