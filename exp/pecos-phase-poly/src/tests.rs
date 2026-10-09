// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Independent dense evolution, projection, and 3PP-membership oracles.

use super::*;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec};
use std::sync::OnceLock;

const TOL: f64 = 1e-10;

pub(crate) fn index(rng: &mut PecosRng, n: usize) -> usize {
    usize::try_from(rng.next_u64() % u64::try_from(n).unwrap()).unwrap()
}

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < TOL, "{a} != {b}");
}

fn assert_invariants(sim: &PhasePoly) {
    let n = sim.num_qubits();
    assert!(sim.r <= n);
    assert_eq!(binary::rank(&sim.rows, sim.r), sim.r);
    for mask in sim.rows.iter().chain(sim.phase.keys()) {
        assert_eq!(mask.0.len(), n.div_ceil(64));
        assert!(mask.ones().all(|bit| bit < sim.r));
    }
    for (mask, &coefficient) in &sim.phase {
        assert!(!mask.is_zero());
        assert!((1..8).contains(&coefficient));
    }
}

// PecosRng's PartialEq compares only generator states, not cached bits/words.
// Debug includes all fields, so this checks that the buffered stream is intact.
fn assert_rng_unchanged(actual: &PecosRng, expected: &PecosRng) {
    assert_eq!(format!("{actual:?}"), format!("{expected:?}"));
}

fn assert_unchanged(sim: &PhasePoly, before: &PhasePoly) {
    assert_eq!(sim.rows, before.rows);
    assert_eq!(sim.x0, before.x0);
    assert_eq!(sim.r, before.r);
    assert_eq!(sim.phase, before.phase);
    assert_rng_unchanged(&sim.rng, &before.rng);
}

pub(crate) fn assert_state(sim: &PhasePoly, reference: &[Complex64]) {
    assert_invariants(sim);
    let state = sim.state_vector();
    close(state.iter().map(Complex64::norm_sqr).sum(), 1.0);
    let anchor = reference.iter().position(|a| a.norm() > TOL).unwrap();
    let phase = state[anchor] / reference[anchor];
    close(phase.norm(), 1.0);
    for (i, (a, b)) in state.iter().zip(reference).enumerate() {
        assert!(
            (*a - phase * b).norm() < TOL,
            "basis {i}: {a} != {phase} * {b}"
        );
    }
}

fn arity(gate: usize) -> usize {
    match gate {
        7..=10 => 2,
        11 => 3,
        _ => 1,
    }
}

// Gates 0..11: X,Y,Z,S,Sdg,T,Tdg,CX,CZ,CS,CSdg,CCZ; 12/13: PZ/PX.
fn apply_gate(sim: &mut PhasePoly, gate: usize, qubits: &[QubitId]) {
    match gate {
        0 => sim.x(qubits),
        1 => sim.y(qubits),
        2 => sim.z(qubits),
        3 => sim.sz(qubits),
        4 => sim.szdg(qubits),
        5 => sim.t(qubits),
        6 => sim.tdg(qubits),
        7 => sim.cx(&[(qubits[0], qubits[1])]),
        8 => sim.cz(&[(qubits[0], qubits[1])]),
        9 => sim.cs(&[(qubits[0], qubits[1])]),
        10 => sim.csdg(&[(qubits[0], qubits[1])]),
        11 => sim.ccz(&[(qubits[0], qubits[1], qubits[2])]),
        12 => sim.pz(qubits),
        13 => sim.px(qubits),
        _ => unreachable!(),
    };
}

// Build small gate matrices from StateVec and act on a plain vector, which
// allows independent forced projections between gates. CS/CSdg/CCZ are explicit
// computational-basis diagonal matrices. Gate 12 is H, for the reset oracle.
fn matrices() -> &'static Vec<Vec<Vec<Complex64>>> {
    static MATRICES: OnceLock<Vec<Vec<Vec<Complex64>>>> = OnceLock::new();
    MATRICES.get_or_init(|| {
        (0..13)
            .map(|gate| {
                let width = arity(gate);
                (0..1 << width)
                    .map(|input| {
                        let mut sim = StateVec::new(width);
                        for bit in 0..width {
                            if input & (1 << bit) != 0 {
                                sim.x(&[QubitId(bit)]);
                            }
                        }
                        let one = &[QubitId(0)];
                        let pair = &[(QubitId(0), QubitId(1))];
                        match gate {
                            0 => {
                                sim.x(one);
                            }
                            1 => {
                                sim.y(one);
                            }
                            2 => {
                                sim.z(one);
                            }
                            3 => {
                                sim.sz(one);
                            }
                            4 => {
                                sim.szdg(one);
                            }
                            5 => {
                                sim.t(one);
                            }
                            6 => {
                                sim.tdg(one);
                            }
                            7 => {
                                sim.cx(pair);
                            }
                            8 => {
                                sim.cz(pair);
                            }
                            9..=11 => {}
                            12 => {
                                sim.h(one);
                            }
                            _ => unreachable!(),
                        }
                        let mut state = sim.state();
                        if (9..=11).contains(&gate) && input == (1 << width) - 1 {
                            state[input] *= match gate {
                                9 => Complex64::new(0.0, 1.0),
                                10 => Complex64::new(0.0, -1.0),
                                _ => Complex64::new(-1.0, 0.0),
                            };
                        }
                        state
                    })
                    .collect()
            })
            .collect()
    })
}

pub(crate) fn dense_gate(state: &mut [Complex64], gate: usize, qubits: &[QubitId]) {
    let columns = &matrices()[gate];
    let affected = qubits.iter().fold(0, |mask, q| mask | (1 << q.0));
    let old = state.to_vec();
    state.fill(Complex64::new(0.0, 0.0));
    for (input, amplitude) in old.into_iter().enumerate() {
        let local = qubits
            .iter()
            .enumerate()
            .fold(0, |mask, (j, q)| mask | (((input >> q.0) & 1) << j));
        for (output, coefficient) in columns[local].iter().enumerate() {
            let destination = qubits
                .iter()
                .enumerate()
                .fold(input & !affected, |mask, (j, q)| {
                    mask | (((output >> j) & 1) << q.0)
                });
            state[destination] += amplitude * coefficient;
        }
    }
}

pub(crate) fn dense_projection(
    state: &[Complex64],
    qubits: &[QubitId],
    is_x: bool,
    outcome: bool,
) -> (f64, Vec<Complex64>) {
    let mask = qubits.iter().fold(0_usize, |mask, q| mask | (1 << q.0));
    let sign = if outcome { -1.0 } else { 1.0 };
    let mut projected: Vec<_> = state
        .iter()
        .enumerate()
        .map(|(i, a)| {
            let transformed = if is_x {
                state[i ^ mask]
            } else {
                *a * if (i & mask).count_ones() % 2 == 0 {
                    1.0
                } else {
                    -1.0
                }
            };
            (*a + sign * transformed) / 2.0
        })
        .collect();
    let probability: f64 = projected.iter().map(Complex64::norm_sqr).sum();
    if probability > TOL {
        for amplitude in &mut projected {
            *amplitude /= probability.sqrt();
        }
    }
    (probability, projected)
}

fn shifted_qubits(qubits: &[QubitId], offset: usize) -> Vec<QubitId> {
    qubits.iter().map(|q| QubitId(q.0 + offset)).collect()
}

fn synchronized_gate(
    sim: &mut PhasePoly,
    state: &mut Vec<Complex64>,
    gate: usize,
    qubits: &[QubitId],
) {
    synchronized_gate_at(sim, state, gate, qubits, 0);
    assert_state(sim, state);
}

fn synchronized_gate_at(
    sim: &mut PhasePoly,
    state: &mut Vec<Complex64>,
    gate: usize,
    qubits: &[QubitId],
    offset: usize,
) {
    let physical = shifted_qubits(qubits, offset);
    if gate < 12 {
        apply_gate(sim, gate, &physical);
        dense_gate(state, gate, qubits);
    } else {
        let mut predicted = sim.clone();
        let outcome = predicted.mz(&physical)[0].outcome;
        let (probability, projection) = dense_projection(state, qubits, false, outcome);
        assert!(probability > TOL);
        *state = projection;
        if outcome {
            dense_gate(state, 0, qubits);
        }
        if gate == 13 {
            dense_gate(state, 12, qubits);
        }
        apply_gate(sim, gate, &physical);
        assert_rng_unchanged(&sim.rng, &predicted.rng);
    }
    assert_invariants(sim);
}

fn random_gate(rng: &mut PecosRng, n: usize) -> (usize, Vec<QubitId>) {
    let mut gate = index(rng, 17);
    if gate >= 14 {
        gate = 13;
    } // Keep support nontrivial despite mid-circuit resets.
    while arity(gate) > n {
        gate = index(rng, 14);
    }
    let mut qubits = Vec::new();
    while qubits.len() < arity(gate) {
        let q = QubitId(index(rng, n));
        if !qubits.contains(&q) {
            qubits.push(q);
        }
    }
    (gate, qubits)
}

fn random_string(rng: &mut PecosRng, n: usize) -> Vec<QubitId> {
    let mut qubits: Vec<_> = (0..n)
        .filter(|_| rng.next_bool_fast())
        .map(QubitId)
        .collect();
    if qubits.is_empty() {
        qubits.push(QubitId(index(rng, n)));
    }
    qubits
}

fn dense_expectation(state: &[Complex64], paulis: &[Pauli], negative: bool) -> f64 {
    let mut value = Complex64::new(0.0, 0.0);
    for (input, amplitude) in state.iter().enumerate() {
        let mut output = input;
        let mut phase = Complex64::new(if negative { -1.0 } else { 1.0 }, 0.0);
        for (q, pauli) in paulis.iter().enumerate() {
            let bit = input & (1 << q) != 0;
            match pauli {
                Pauli::I => {}
                Pauli::X => output ^= 1 << q,
                Pauli::Y => {
                    output ^= 1 << q;
                    phase *= Complex64::new(0.0, if bit { -1.0 } else { 1.0 });
                }
                Pauli::Z => {
                    if bit {
                        phase = -phase;
                    }
                }
            }
        }
        value += state[output].conj() * phase * amplitude;
    }
    close(value.im, 0.0);
    value.re
}

fn check_paulis(sim: &PhasePoly, state: &[Complex64], rng: &mut PecosRng) {
    for _ in 0..8 {
        let paulis: Vec<_> = (0..sim.num_qubits())
            .map(|_| [Pauli::I, Pauli::X, Pauli::Y, Pauli::Z][index(rng, 4)])
            .collect();
        for negative in [false, true] {
            let sign = if negative {
                QuarterPhase::MinusOne
            } else {
                QuarterPhase::PlusOne
            };
            let pauli = PauliString::from_paulis_with_phase(sign, &paulis);
            close(
                sim.expectation(&pauli),
                dense_expectation(state, &paulis, negative),
            );
        }
    }
}

#[test]
fn unitary_reset_and_pauli_sweep() {
    for n in 1..=8 {
        for seed in 0..40 {
            let mut rng = PecosRng::seed_from_u64(1234 + seed);
            let mut sim = PhasePoly::with_seed(n, 9876 + seed);
            let mut state = StateVec::new(n).state();
            for q in 0..n {
                synchronized_gate(&mut sim, &mut state, 13, &[QubitId(q)]);
            }
            for step in 0..100 {
                let (gate, qubits) = random_gate(&mut rng, n);
                synchronized_gate(&mut sim, &mut state, gate, &qubits);
                if step % 10 == 0 {
                    check_paulis(&sim, &state, &mut rng);
                }
            }
            for (basis, amplitude) in sim.state_vector().iter().enumerate() {
                let bits: Vec<_> = (0..n).map(|q| basis & (1 << q) != 0).collect();
                assert!((sim.amplitude(&bits) - amplitude).norm() < TOL);
            }
        }
    }
}

// This oracle uses only a normalized dense vector. Recover an affine basis,
// read relative eighth-root phases, and perform a Boolean Moebius transform.
// No masks, phase terms, derivatives, or decisions from PhasePoly are consulted.
fn is_3pp(state: &[Complex64]) -> bool {
    if (state.iter().map(Complex64::norm_sqr).sum::<f64>() - 1.0).abs() > TOL {
        return false;
    }
    let support: Vec<_> = state
        .iter()
        .enumerate()
        .filter(|(_, a)| a.norm() > TOL)
        .map(|(i, _)| i)
        .collect();
    if !support.len().is_power_of_two() {
        return false;
    }
    let offset = support[0];
    let magnitude = state[offset].norm();
    if support
        .iter()
        .any(|&i| (state[i].norm() - magnitude).abs() > TOL)
    {
        return false;
    }
    let mut basis = Vec::new();
    let mut span = vec![0];
    for &i in &support {
        let v = i ^ offset;
        if !span.contains(&v) {
            basis.push(v);
            let extension: Vec<_> = span.iter().map(|&x| x ^ v).collect();
            span.extend(extension);
            if span.len() > support.len() {
                return false;
            }
        }
    }
    if span.iter().any(|&x| !support.contains(&(x ^ offset))) {
        return false;
    }
    let phases: Vec<_> = (0..8)
        .map(|i| Complex64::from_polar(1.0, f64::from(i) * std::f64::consts::FRAC_PI_4))
        .collect();
    let mut coefficients = Vec::new();
    for x in span {
        let ratio = state[x ^ offset] / state[offset];
        let Some(phase) = phases.iter().position(|w| (ratio - w).norm() < TOL) else {
            return false;
        };
        coefficients.push(i32::try_from(phase).unwrap());
    }
    for bit in 0..basis.len() {
        for subset in 0..coefficients.len() {
            if subset & (1 << bit) != 0 {
                coefficients[subset] =
                    (coefficients[subset] - coefficients[subset ^ (1 << bit)]).rem_euclid(8);
            }
        }
    }
    coefficients.iter().enumerate().skip(1).all(|(mask, c)| {
        let degree = mask.count_ones();
        if degree >= 4 {
            *c == 0
        } else {
            c % (1 << (degree - 1)) == 0
        }
    })
}

fn outcome_choice(probability_one: f64, rng: &mut PecosRng) -> bool {
    if probability_one < TOL {
        false
    } else if probability_one > 1.0 - TOL {
        true
    } else {
        rng.next_bool_fast()
    }
}

fn attempt_x(
    sim: &mut PhasePoly,
    state: &mut Vec<Complex64>,
    qubits: &[QubitId],
    rng: &mut PecosRng,
    counts: &mut [usize; 6],
) {
    attempt_x_at(sim, state, qubits, rng, counts, 0);
    assert_state(sim, state);
}

fn attempt_x_at(
    sim: &mut PhasePoly,
    state: &mut Vec<Complex64>,
    qubits: &[QubitId],
    rng: &mut PecosRng,
    counts: &mut [usize; 6],
    offset: usize,
) {
    let physical = shifted_qubits(qubits, offset);
    let branches = [
        dense_projection(state, qubits, true, false),
        dense_projection(state, qubits, true, true),
    ];
    let compatible = branches
        .iter()
        .all(|(p, branch)| *p < TOL || is_3pp(branch));
    let before = sim.clone();
    let query = sim.x_probabilities(&physical);
    assert_unchanged(sim, &before);
    assert_eq!(
        query.is_ok(),
        compatible,
        "classification mismatch: {query:?}, string {qubits:?}"
    );
    if let Ok((probabilities, case)) = query {
        for i in 0..2 {
            close(probabilities[i], branches[i].0);
        }
        let outcome = outcome_choice(branches[1].0, rng);
        let result = if qubits.len() == 1 {
            sim.mx_forced(physical[0], outcome)
        } else {
            sim.mx_string_forced(&physical, outcome)
        }
        .unwrap();
        assert_eq!(result.outcome, outcome);
        let deterministic = branches.iter().any(|(p, _)| *p < TOL);
        assert_eq!(result.is_deterministic, deterministic);
        counts[match case {
            XMeasurementCase::Case1 => 0,
            XMeasurementCase::Case2a => {
                if deterministic {
                    1
                } else {
                    2
                }
            }
            XMeasurementCase::Case2b => 3,
            XMeasurementCase::Case2c => 4,
        }] += 1;
        *state = branches[usize::from(outcome)].1.clone();
        assert_rng_unchanged(&sim.rng, &before.rng);
    } else {
        counts[5] += 1;
        assert!(sim.mx_string(&physical).is_err());
        assert_unchanged(sim, &before);
        for outcome in [false, true] {
            assert!(sim.mx_string_forced(&physical, outcome).is_err());
            assert_unchanged(sim, &before);
        }
    }
    assert_invariants(sim);
}

fn attempt_z(
    sim: &mut PhasePoly,
    state: &mut Vec<Complex64>,
    qubits: &[QubitId],
    rng: &mut PecosRng,
) {
    attempt_z_at(sim, state, qubits, rng, 0);
    assert_state(sim, state);
}

fn attempt_z_at(
    sim: &mut PhasePoly,
    state: &mut Vec<Complex64>,
    qubits: &[QubitId],
    rng: &mut PecosRng,
    offset: usize,
) {
    let physical = shifted_qubits(qubits, offset);
    let branches = [
        dense_projection(state, qubits, false, false),
        dense_projection(state, qubits, false, true),
    ];
    let before = sim.clone();
    let probabilities = sim.z_probabilities(&physical);
    assert_unchanged(sim, &before);
    for i in 0..2 {
        close(probabilities[i], branches[i].0);
    }
    let outcome = outcome_choice(branches[1].0, rng);
    let result = if qubits.len() == 1 {
        sim.mz_forced(physical[0].0, outcome)
    } else {
        sim.mz_string_forced(&physical, outcome)
    };
    assert_eq!(result.outcome, outcome);
    assert_eq!(
        result.is_deterministic,
        branches.iter().any(|(p, _)| *p < TOL)
    );
    *state = branches[usize::from(outcome)].1.clone();
    assert_rng_unchanged(&sim.rng, &before.rng);
    assert_invariants(sim);
}

fn biased_prefix(sim: &mut PhasePoly, state: &mut Vec<Complex64>, kind: usize) {
    if kind == 0 {
        return;
    }
    let n = sim.num_qubits();
    for q in 0..n {
        synchronized_gate(sim, state, 13, &[QubitId(q)]);
    }
    match kind {
        2 if n >= 2 => synchronized_gate(sim, state, 8, &[QubitId(0), QubitId(1)]),
        3 => {
            synchronized_gate(sim, state, 3, &[QubitId(0)]);
            if n >= 3 {
                synchronized_gate(sim, state, 11, &[QubitId(0), QubitId(1), QubitId(2)]);
            }
        }
        4 => synchronized_gate(sim, state, 5, &[QubitId(0)]),
        5 if n >= 3 => synchronized_gate(sim, state, 11, &[QubitId(0), QubitId(1), QubitId(2)]),
        _ => {}
    }
}

#[test]
fn measurement_and_bidirectional_classification_sweep() {
    let mut counts = [0; 6];
    for n in 1..=8 {
        for seed in 0..48 {
            let mut rng = PecosRng::seed_from_u64(4433 + seed);
            let mut sim = PhasePoly::with_seed(n, 98765 + seed);
            let mut state = StateVec::new(n).state();
            biased_prefix(&mut sim, &mut state, usize::try_from(seed % 6).unwrap());
            attempt_x(&mut sim, &mut state, &[QubitId(0)], &mut rng, &mut counts);
            for _ in 0..80 {
                let (gate, qubits) = random_gate(&mut rng, n);
                synchronized_gate(&mut sim, &mut state, gate, &qubits);
                let qubits = if rng.next_bool_fast() {
                    vec![QubitId(index(&mut rng, n))]
                } else {
                    random_string(&mut rng, n)
                };
                if rng.next_bool_fast() {
                    attempt_x(&mut sim, &mut state, &qubits, &mut rng, &mut counts);
                } else {
                    attempt_z(&mut sim, &mut state, &qubits, &mut rng);
                }
            }
        }
    }
    println!("X case coverage [1, 2a deterministic, 2a random, 2b, 2c, incompatible]: {counts:?}");
    assert!(counts.iter().all(|&count| count >= 20));
}

#[test]
fn targeted_x_cases() {
    let q = QubitId(0);
    let mut sim = PhasePoly::with_seed(1, 1);
    assert_eq!(
        sim.x_probabilities(&[q]).unwrap().1,
        XMeasurementCase::Case1
    );
    sim.px(&[q]).t(&[q]);
    let (p, case) = sim.x_probabilities(&[q]).unwrap();
    assert_eq!(case, XMeasurementCase::Case2c);
    close(p[0], (2.0 + std::f64::consts::SQRT_2) / 4.0);
    close(p[1], (2.0 - std::f64::consts::SQRT_2) / 4.0);
    let mut sim = PhasePoly::with_seed(3, 1);
    sim.px(&[q, QubitId(1), QubitId(2)])
        .ccz(&[(q, QubitId(1), QubitId(2))]);
    assert!(
        sim.mx(q)
            .err()
            .unwrap()
            .to_string()
            .contains("quadratic matrix is nonzero")
    );
    sim.sz(&[q]);
    assert_eq!(
        sim.x_probabilities(&[q]).unwrap().1,
        XMeasurementCase::Case2b
    );
    let mut sim = PhasePoly::with_seed(5, 1);
    sim.px(&(0..5).map(QubitId).collect::<Vec<_>>())
        .sz(&[q])
        .ccz(&[(q, QubitId(1), QubitId(2))])
        .ccz(&[(q, QubitId(3), QubitId(4))]);
    assert!(sim.mx(q).err().unwrap().to_string().contains("rank 4"));
}

#[test]
fn gauss_sums_against_enumeration() {
    let mut rng = PecosRng::seed_from_u64(88_889_999);
    let mut zeros = 0;
    for r in 0..=10 {
        for _ in 0..100 {
            let mut polynomial = Clifford::zero(r, r);
            polynomial.k = u8::try_from(index(&mut rng, 8)).unwrap();
            for i in 0..r {
                polynomial.s[i] = u8::try_from(index(&mut rng, 4)).unwrap();
                for j in i + 1..r {
                    if rng.next_bool_fast() {
                        polynomial.m[i].toggle(j);
                        polynomial.m[j].toggle(i);
                    }
                }
            }
            let brute: Complex64 = (0_usize..1 << r)
                .map(|y| {
                    let mut exponent = u32::from(polynomial.k);
                    for i in 0..r {
                        if y & (1 << i) != 0 {
                            exponent += 2 * u32::from(polynomial.s[i]);
                            for j in i + 1..r {
                                if y & (1 << j) != 0 && polynomial.m[i].get(j) {
                                    exponent += 4;
                                }
                            }
                        }
                    }
                    Complex64::from_polar(
                        1.0,
                        f64::from(exponent % 8) * std::f64::consts::FRAC_PI_4,
                    )
                })
                .sum();
            let exact = polynomial.gauss_sum();
            if exact.is_none() {
                zeros += 1;
            }
            let actual = exact.map_or(Complex64::new(0.0, 0.0), |sum| sum.value(0));
            assert!((actual - brute).norm() < TOL, "r={r}, {actual} != {brute}");
            assert_eq!(exact.is_none(), brute.norm() < TOL);
        }
    }
    assert!(zeros >= 20);
}

#[test]
fn scale_smoke_256_qubits() {
    let mut rng = PecosRng::seed_from_u64(2026);
    let mut sim = PhasePoly::with_seed(256, 17);
    let qubits: Vec<_> = (0..256).map(QubitId).collect();
    sim.px(&qubits).t(&qubits);
    for _ in 0..3000 {
        let a = index(&mut rng, 256);
        let b = (a + 1 + index(&mut rng, 255)) % 256;
        sim.cx(&[(QubitId(a), QubitId(b))]);
    }
    for _ in 0..320 {
        let mut triple = Vec::new();
        while triple.len() < 3 {
            let q = QubitId(index(&mut rng, 256));
            if !triple.contains(&q) {
                triple.push(q);
            }
        }
        sim.ccz(&[(triple[0], triple[1], triple[2])]);
    }
    assert_invariants(&sim);
    assert_eq!(sim.mz(&qubits).len(), 256);
    assert_invariants(&sim);
    assert_eq!(sim.support_dimension(), 0);
    assert_eq!(sim.num_phase_terms(), 0);
}

#[test]
fn empty_inputs_global_phase_and_forced_determinism() {
    let mut sim = PhasePoly::with_seed(0, 8);
    assert_eq!(sim.state_vector(), vec![Complex64::new(1.0, 0.0)]);
    close(sim.amplitude(&[]).re, 1.0);
    close(sim.expectation(&PauliString::identity()), 1.0);
    assert!(!sim.mx_string_forced(&[], true).unwrap().outcome);
    assert!(!sim.mz_string_forced(&[], true).outcome);
    assert_invariants(&sim);
    let q = QubitId(0);
    let mut sim = PhasePoly::with_seed(1, 8);
    sim.x(&[q]).t(&[q]);
    close(sim.amplitude(&[true]).re, 1.0); // T contributes only omitted global phase.
    assert!(sim.mz_forced(0, false).outcome);
    sim.px(&[q]);
    assert!(!sim.mx_forced(q, true).unwrap().outcome);
    sim.z(&[q]);
    assert!(sim.mx_forced(q, false).unwrap().outcome);
    let rng = sim.rng.clone();
    sim.reset();
    assert_rng_unchanged(&sim.rng, &rng);
    assert_eq!(sim.num_qubits(), 1);
    assert_eq!(
        sim.state_vector(),
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)]
    );
}

#[test]
fn coordinate_deletion_crosses_word_boundaries() {
    let n = 130;
    let mut sim = PhasePoly::with_seed(n, 0);
    let qubits: Vec<_> = (0..n).map(QubitId).collect();
    sim.px(&qubits).t(&qubits);
    for q in [63, 64, 0, 129, 65] {
        let result = sim.mz_forced(q, true);
        assert!(result.outcome);
        assert_invariants(&sim);
        close(sim.expectation(&PauliString::z(q)), -1.0);
        sim.px(&[QubitId(q)]).t(&[QubitId(q)]);
        assert_invariants(&sim);
        close(
            sim.expectation(&PauliString::x(q)),
            std::f64::consts::FRAC_1_SQRT_2,
        );
    }
}

#[test]
fn membership_oracle_rejects_each_violation() {
    let mut state = StateVec::new(4);
    state.h(&(0..4).map(QubitId).collect::<Vec<_>>());
    let uniform = state.state();
    assert!(is_3pp(&uniform));
    for (mask, angle) in [(3, 1.0), (7, 2.0), (15, 4.0), (1, 0.5)] {
        let mut invalid = uniform.clone();
        for (i, a) in invalid.iter_mut().enumerate() {
            if i & mask == mask {
                *a *= Complex64::from_polar(1.0, angle * std::f64::consts::FRAC_PI_4);
            }
        }
        assert!(!is_3pp(&invalid));
    }
    let mut invalid = vec![Complex64::new(0.0, 0.0); 8];
    for i in [0, 1, 2, 4] {
        invalid[i] = Complex64::new(0.5, 0.0);
    }
    assert!(!is_3pp(&invalid));
    let unequal = [Complex64::new(0.6, 0.0), Complex64::new(0.8, 0.0)];
    assert!(!is_3pp(&unequal));
}

#[test]
fn sampled_x_branches_and_seed_reproducibility() {
    let q = QubitId(0);
    for power in [1, 3, 5, 7] {
        let mut ones = 0;
        let mut expected = 0.0;
        for seed in 0..256 {
            let mut sim = PhasePoly::with_seed(1, seed);
            let mut state = StateVec::new(1).state();
            synchronized_gate(&mut sim, &mut state, 13, &[q]);
            for _ in 0..power {
                synchronized_gate(&mut sim, &mut state, 5, &[q]);
            }
            let probabilities = sim.x_probabilities(&[q]).unwrap().0;
            expected = probabilities[1];
            let mut replay = sim.clone();
            let result = sim.mx(q).unwrap();
            let repeated = replay.mx(q).unwrap();
            assert_eq!(result.outcome, repeated.outcome);
            assert_unchanged(&sim, &replay);
            ones += u32::from(result.outcome);
            let (probability, projected) = dense_projection(&state, &[q], true, result.outcome);
            close(probability, probabilities[usize::from(result.outcome)]);
            assert_state(&sim, &projected);
            assert!(sim.mx_forced(q, !result.outcome).unwrap().is_deterministic);
        }
        assert!((f64::from(ones) / 256.0 - expected).abs() < 0.1);
    }
    // Exercise random support extension and random case-2a/2b updates as well.
    for kind in [0, 2, 3] {
        for seed in 0..32 {
            let mut sim = PhasePoly::with_seed(3, seed);
            let mut state = StateVec::new(3).state();
            biased_prefix(&mut sim, &mut state, kind);
            let result = sim.mx(q).unwrap();
            let (_, projected) = dense_projection(&state, &[q], true, result.outcome);
            assert_state(&sim, &projected);
        }
    }
}

#[test]
fn batches_and_support_disjoint_expectations() {
    let mut sim = PhasePoly::with_seed(8, 123);
    let mut reference = StateVec::new(8);
    let qubits: Vec<_> = (0..8).map(QubitId).collect();
    sim.px(&qubits).t(&qubits).x(&[QubitId(1), QubitId(5)]);
    reference.h(&qubits).t(&qubits).x(&[QubitId(1), QubitId(5)]);
    let pairs = [(QubitId(0), QubitId(7)), (QubitId(2), QubitId(6))];
    sim.cx(&pairs).cz(&pairs);
    reference.cx(&pairs).cz(&pairs);
    assert_state(&sim, &reference.state());
    sim.reset();
    close(sim.expectation(&PauliString::xs(&[0, 7])), 0.0);
    sim.px(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(7))]);
    close(sim.expectation(&PauliString::xs(&[0, 7])), 1.0);
    close(sim.expectation(&PauliString::ys(&[0, 7])), -1.0);
    close(sim.expectation(&PauliString::x(0)), 0.0);
}

#[test]
fn invalid_inputs_fail_before_mutation() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let mut sim = PhasePoly::with_seed(3, 1);
    let before = sim.clone();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            sim.px(&[QubitId(0), QubitId(0)]);
        }))
        .is_err()
    );
    assert_unchanged(&sim, &before);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            sim.cx(&[(QubitId(0), QubitId(1)), (QubitId(1), QubitId(2))]);
        }))
        .is_err()
    );
    assert_unchanged(&sim, &before);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            sim.mx_string(&[QubitId(0), QubitId(3)]).ok();
        }))
        .is_err()
    );
    assert_unchanged(&sim, &before);
    let pauli = PauliString::from_paulis_with_phase(QuarterPhase::PlusI, &[Pauli::Y]);
    assert!(catch_unwind(AssertUnwindSafe(|| sim.expectation(&pauli))).is_err());
    assert_unchanged(&sim, &before);
}

// Condition on a nonzero dense anchor in every other independent block. Divide
// by the global anchor before comparing, so amplitudes of order 2^(-n/2) cannot
// pass merely because their absolute values are below TOL.
fn assert_block_oracles(sim: &PhasePoly, states: &[Vec<Complex64>]) {
    assert_invariants(sim);
    let width = 7;
    let anchors: Vec<_> = states
        .iter()
        .map(|state| state.iter().position(|a| a.norm() > TOL).unwrap())
        .collect();
    let mut bits: Vec<_> = anchors
        .iter()
        .flat_map(|&anchor| (0..width).map(move |q| anchor & (1 << q) != 0))
        .collect();
    let global_anchor = sim.amplitude(&bits);
    let expected_magnitude: f64 = states
        .iter()
        .zip(&anchors)
        .map(|(state, &anchor)| state[anchor].norm())
        .product();
    close(global_anchor.norm() / expected_magnitude, 1.0);
    for (block, state) in states.iter().enumerate() {
        let offset = block * width;
        for (basis, expected) in state.iter().enumerate() {
            for q in 0..width {
                bits[offset + q] = basis & (1 << q) != 0;
            }
            let actual = sim.amplitude(&bits) / global_anchor * state[anchors[block]];
            assert!(
                (actual - expected).norm() < TOL,
                "block {block}, basis {basis}: {actual} != {expected}"
            );
        }
        for q in 0..width {
            bits[offset + q] = anchors[block] & (1 << q) != 0;
            let mut paulis = vec![Pauli::I; width];
            paulis[q] = Pauli::Z;
            close(
                sim.expectation(&PauliString::z(offset + q)),
                dense_expectation(state, &paulis, false),
            );
        }
    }
}

#[test]
fn interleaved_blocks_cross_word_boundary_against_dense_oracles() {
    const BLOCKS: usize = 12;
    const WIDTH: usize = 7;
    let mut sim = PhasePoly::with_seed(BLOCKS * WIDTH, 9241);
    let mut states: Vec<_> = (0..BLOCKS).map(|_| StateVec::new(WIDTH).state()).collect();
    let mut rng = PecosRng::seed_from_u64(3019);
    let mut max_r = 0;
    let mut counts = [0; 6];
    // Coordinate q*BLOCKS+b belongs to physical qubit b*WIDTH+q.
    for q in 0..WIDTH {
        for (block, state) in states.iter_mut().enumerate() {
            synchronized_gate_at(&mut sim, state, 13, &[QubitId(q)], block * WIDTH);
            max_r = max_r.max(sim.support_dimension());
        }
        assert_block_oracles(&sim, &states);
    }
    assert!(max_r > 64);
    for round in 0..12 {
        for (block, state) in states.iter_mut().enumerate() {
            let offset = block * WIDTH;
            let q = (round + block) % WIDTH;
            let next = (q + 1) % WIDTH;
            // Entangle within blocks and vary phases before deleting coordinates.
            synchronized_gate_at(&mut sim, state, 7, &[QubitId(q), QubitId(next)], offset);
            synchronized_gate_at(&mut sim, state, 5, &[QubitId(q)], offset);
            let (gate, qubits) = random_gate(&mut rng, WIDTH);
            synchronized_gate_at(&mut sim, state, gate, &qubits, offset);
            max_r = max_r.max(sim.support_dimension());
        }
        assert_block_oracles(&sim, &states);
        for (block, state) in states.iter_mut().enumerate() {
            let offset = block * WIDTH;
            let qubits = random_string(&mut rng, WIDTH);
            if (round + block) % 2 == 0 {
                attempt_x_at(&mut sim, state, &qubits, &mut rng, &mut counts, offset);
            } else {
                attempt_z_at(&mut sim, state, &qubits, &mut rng, offset);
            }
            max_r = max_r.max(sim.support_dimension());
        }
        assert_block_oracles(&sim, &states);
        // Reset and append in a different block order after the restrictions.
        for block in (0..BLOCKS).rev() {
            let q = QubitId((round + block) % WIDTH);
            synchronized_gate_at(&mut sim, &mut states[block], 12, &[q], block * WIDTH);
            synchronized_gate_at(&mut sim, &mut states[block], 13, &[q], block * WIDTH);
            max_r = max_r.max(sim.support_dimension());
        }
        assert_block_oracles(&sim, &states);
    }
    assert!(counts[..5].iter().sum::<usize>() > 0);
    assert!(counts[5] > 0);
    println!("Interleaved block oracle maximum support dimension: {max_r}");
}

#[test]
fn unforced_x_string_sampling_matches_probability_and_dense_projection() {
    const SHOTS: u32 = 4096;
    let qubits = [QubitId(0), QubitId(1)];
    let mut ones = 0;
    let mut expected = 0.0;
    for seed in 0..SHOTS {
        let mut sim = PhasePoly::with_seed(3, u64::from(seed));
        let mut state = StateVec::new(3).state();
        for q in 0..3 {
            synchronized_gate(&mut sim, &mut state, 13, &[QubitId(q)]);
        }
        synchronized_gate(&mut sim, &mut state, 5, &[QubitId(0)]);
        synchronized_gate(&mut sim, &mut state, 7, &[QubitId(2), QubitId(1)]);
        let (probabilities, case) = sim.x_probabilities(&qubits).unwrap();
        assert_eq!(case, XMeasurementCase::Case2c);
        expected = probabilities[1];
        close(expected, (2.0 - std::f64::consts::SQRT_2) / 4.0);
        let result = sim.mx_string(&qubits).unwrap();
        assert!(!result.is_deterministic);
        ones += u32::from(result.outcome);
        let (probability, projection) = dense_projection(&state, &qubits, true, result.outcome);
        close(probability, probabilities[usize::from(result.outcome)]);
        assert_state(&sim, &projection);
    }
    // For independent Bernoulli shots, Hoeffding gives
    // Pr(|frequency-p| >= 0.05) <= 2 exp(-2*4096*0.05^2) < 2.6e-9.
    // Distinct fixed seeds make this statistical regression reproducible.
    let frequency = f64::from(ones) / f64::from(SHOTS);
    assert!(
        (frequency - expected).abs() < 0.05,
        "frequency {frequency}, expected {expected}"
    );
}
