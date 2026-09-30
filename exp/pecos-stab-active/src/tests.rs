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

use super::*;
use pecos_core::IndexSet;
use pecos_simulators::state_vector_test_utils::normalized_z_projection;
use pecos_simulators::{GensGeneric, StateVec};

// A dense Pauli matrix represented by its one nonzero entry per column. Its
// construction reads only physical Y-convention row bits and both sign bits.
fn dense_row<S: IndexSet>(gens: &GensGeneric<S>, row: usize) -> Vec<(usize, Complex64)> {
    let n = gens.get_num_qubits();
    (0..1 << n)
        .map(|input| {
            let mut output = input;
            let mut coefficient = Complex64::new(1.0, 0.0);
            if gens.signs_minus.contains(row) {
                coefficient = -coefficient;
            }
            if gens.signs_i.contains(row) {
                coefficient *= Complex64::new(0.0, 1.0);
            }
            for q in 0..n {
                match (gens.row_x[row].contains(q), gens.row_z[row].contains(q)) {
                    (true, false) => output ^= 1 << q,
                    (false, true) => {
                        if input & (1 << q) != 0 {
                            coefficient = -coefficient;
                        }
                    }
                    (true, true) => {
                        output ^= 1 << q;
                        coefficient *= if input & (1 << q) == 0 {
                            Complex64::new(0.0, 1.0)
                        } else {
                            Complex64::new(0.0, -1.0)
                        };
                    }
                    (false, false) => {}
                }
            }
            (output, coefficient)
        })
        .collect()
}

fn apply_dense(matrix: &[(usize, Complex64)], vector: &[Complex64]) -> Vec<Complex64> {
    let mut output = vec![Complex64::new(0.0, 0.0); vector.len()];
    for (input, &(row, coefficient)) in matrix.iter().enumerate() {
        output[row] = coefficient * vector[input];
    }
    output
}

// Independent H1 oracle: no simulator methods or decomposition/basis-change
// helpers are used. The random seed only selects a nonorthogonal projector input.
fn reconstruct(sim: &StabActive) -> Vec<Complex64> {
    let n = sim.tableau.num_qubits();
    let mut rng = PecosRng::seed_from_u64(937);
    let mut phi: Vec<_> = (0..1 << n)
        .map(|_| Complex64::new(rng.next_f64() - 0.5, rng.next_f64() - 0.5))
        .collect();
    for row in 0..n {
        let image = apply_dense(&dense_row(sim.tableau.stabs(), row), &phi);
        for (a, b) in phi.iter_mut().zip(image) {
            *a = (*a + b) / 2.0;
        }
    }
    let norm = phi.iter().map(Complex64::norm_sqr).sum::<f64>().sqrt();
    assert!(norm > 1e-12);
    for a in &mut phi {
        *a /= norm;
    }
    let matrices: Vec<_> = sim
        .active
        .iter()
        .map(|&row| dense_row(sim.tableau.destabs(), row))
        .collect();
    let mut basis = vec![phi];
    for x in 1usize..sim.amplitudes.len() {
        let bit = x.trailing_zeros() as usize;
        // The least present bit is the leftmost factor in the representation.
        basis.push(apply_dense(&matrices[bit], &basis[x ^ (1 << bit)]));
    }
    let mut state = vec![Complex64::new(0.0, 0.0); 1 << n];
    for (coefficient, vector) in sim.amplitudes.iter().zip(basis) {
        for (output, value) in state.iter_mut().zip(vector) {
            *output += coefficient * value;
        }
    }
    state
}

fn assert_state(sim: &StabActive, reference: &[Complex64], label: &str) {
    let state = reconstruct(sim);
    let norm: f64 = state.iter().map(Complex64::norm_sqr).sum();
    assert!((norm - 1.0).abs() <= 1e-10, "{label}: norm {norm}");
    let overlap: Complex64 = state.iter().zip(reference).map(|(a, b)| a.conj() * b).sum();
    assert!(
        overlap.norm_sqr() >= 1.0 - 1e-10,
        "{label}: fidelity {}",
        overlap.norm_sqr()
    );
}

fn marginal(state: &[Complex64], q: usize) -> f64 {
    state
        .iter()
        .enumerate()
        .filter(|(x, _)| x & (1 << q) != 0)
        .map(|(_, a)| a.norm_sqr())
        .sum()
}

fn synchronized_measure(
    sim: &mut StabActive,
    reference: &mut Vec<Complex64>,
    q: usize,
    choice: bool,
) -> bool {
    let state = reference.clone();
    let expected = marginal(&state, q);
    let actual = sim.probability_one(&sim.parts(&[(q, PauliKindForDecomp::Z)]));
    assert!(
        (actual - expected).abs() <= 1e-10,
        "probability: {actual} != {expected}"
    );
    let outcome = if expected <= 1e-6 {
        false
    } else if expected >= 1.0 - 1e-6 {
        true
    } else {
        choice
    };
    assert_eq!(sim.mz_forced(q, outcome).outcome, outcome);
    let projected = normalized_z_projection(&state, q, outcome, "dense oracle");
    *reference = projected;
    outcome
}

fn index(rng: &mut PecosRng, n: usize) -> usize {
    usize::try_from(rng.next_u64() % u64::try_from(n).unwrap()).unwrap()
}

fn unitary<S: ArbitraryRotationGateable>(
    sim: &mut S,
    gate: usize,
    q: usize,
    r: usize,
    theta: Angle64,
) {
    let qubits = [QubitId(q)];
    let pairs = [(QubitId(q), QubitId(r))];
    match gate {
        1 => {
            sim.sz(&qubits);
        }
        2 => {
            sim.szdg(&qubits);
        }
        3 => {
            sim.x(&qubits);
        }
        4 => {
            sim.y(&qubits);
        }
        5 => {
            sim.z(&qubits);
        }
        6 if q != r => {
            sim.cx(&pairs);
        }
        7 if q != r => {
            sim.cz(&pairs);
        }
        8 => {
            sim.rz(theta, &qubits);
        }
        9 => {
            sim.rx(theta, &qubits);
        }
        10 if q != r => {
            sim.rzz(theta, &pairs);
        }
        _ => {
            sim.h(&qubits);
        }
    }
}

// Evolve the plain dense reference with small physical gate matrices computed
// by StateVec. Keeping a plain vector permits forced dense projection without
// depending on a measurement or state-loading API in StateVec.
fn reference_unitary(state: &mut [Complex64], gate: usize, q: usize, r: usize, angle: Angle64) {
    let two = matches!(gate, 6 | 7 | 10) && q != r;
    let width = if two { 2 } else { 1 };
    let columns: Vec<_> = (0..1 << width)
        .map(|input| {
            let mut simulator = StateVec::new(width);
            for bit in 0..width {
                if input & (1 << bit) != 0 {
                    simulator.x(&[QubitId(bit)]);
                }
            }
            unitary(&mut simulator, gate, 0, usize::from(two), angle);
            simulator.state()
        })
        .collect();
    let affected = (1 << q) | if two { 1 << r } else { 0 };
    let old = state.to_vec();
    state.fill(Complex64::new(0.0, 0.0));
    for (input, amplitude) in old.into_iter().enumerate() {
        let local = ((input >> q) & 1) | if two { ((input >> r) & 1) << 1 } else { 0 };
        for (output, &coefficient) in columns[local].iter().enumerate() {
            let destination = (input & !affected)
                | ((output & 1) << q)
                | if two { ((output >> 1) & 1) << r } else { 0 };
            state[destination] += amplitude * coefficient;
        }
    }
}

#[test]
fn dense_random_circuits() {
    let mut rng = PecosRng::seed_from_u64(19087);
    let mut peak = 0;
    for n in 1..=10 {
        for circuit in 0..20 {
            let mut sim = StabActive::with_seed(n, 91);
            let mut reference = StateVec::new(n).state();
            for depth in 0..60 {
                let q = index(&mut rng, n);
                let r = (q + 1 + index(&mut rng, n.saturating_sub(1).max(1))) % n;
                let gate = if circuit == 0 && depth < n {
                    9
                } else {
                    index(&mut rng, 13)
                };
                let q = if circuit == 0 && depth < n { depth } else { q };
                let angle = Angle64::from_radians(6.0 * rng.next_f64() - 3.0);
                if gate >= 11 {
                    let outcome =
                        synchronized_measure(&mut sim, &mut reference, q, rng.next_bool_fast());
                    if gate == 12 && outcome {
                        sim.x(&[QubitId(q)]);
                        reference_unitary(&mut reference, 3, q, q, Angle64::ZERO);
                    }
                } else {
                    unitary(&mut sim, gate, q, r, angle);
                    reference_unitary(&mut reference, gate, q, r, angle);
                }
                assert_state(
                    &sim,
                    &reference,
                    &format!("n={n}, circuit={circuit}, depth={depth}, gate={gate}"),
                );
                peak = peak.max(sim.active_width());
            }
        }
    }
    assert!(peak >= 8);
}

#[test]
fn repeated_support_bounds_width() {
    let angle = Angle64::from_radians(0.37);
    let mut sim = StabActive::with_seed(10, 12);
    let mut reference = StateVec::new(10).state();
    for step in 0..60 {
        let q = step % 3;
        sim.rx(angle, &[QubitId(q)]);
        reference_unitary(&mut reference, 9, q, q, angle);
        assert!(sim.active_width() <= 3);
        assert_state(&sim, &reference, "bounded independent support");
    }
    sim.reset();
    for q in 0..10 {
        sim.rx(angle, &[QubitId(q)]);
        assert_eq!(sim.active_width(), q + 1);
    }
}

#[test]
fn active_certain_measurements_ignore_impossible_force() {
    for pair in [false, true] {
        for outcome in [false, true] {
            let mut sim = StabActive::with_seed(1, 34);
            let theta = Angle64::from_radians(0.37);
            if pair {
                let t = Angle64::QUARTER_TURN / 2u64;
                sim.rx(t, &[QubitId(0)])
                    .rx(t, &[QubitId(0)])
                    .sxdg(&[QubitId(0)]);
            } else {
                sim.rx(theta, &[QubitId(0)]).rx(-theta, &[QubitId(0)]);
            }
            if outcome {
                sim.x(&[QubitId(0)]);
            }
            let parts = sim.parts(&[(0, PauliKindForDecomp::Z)]);
            assert_eq!(parts.measurement_case(), MeasurementCase::Active);
            assert_eq!(!parts.active_flips.is_empty(), pair);
            let result = sim.mz_forced(0, !outcome);
            assert!(result.is_deterministic, "pair={pair}, outcome={outcome}");
            assert_eq!(result.outcome, outcome);
            assert_eq!(sim.active_width(), 0);
            let mut reference = vec![Complex64::new(0.0, 0.0); 2];
            reference[usize::from(outcome)] = Complex64::new(1.0, 0.0);
            assert_state(&sim, &reference, "certain active measurement");
        }
    }
}

#[test]
fn exact_angle_boundaries() {
    for quarter in 0..4u64 {
        let exact = Angle64::QUARTER_TURN * quarter;
        for theta in [exact - Angle64::new(1), exact, exact + Angle64::new(1)] {
            for gate in [8, 9, 10] {
                let mut sim = StabActive::new(2);
                if gate != 9 {
                    sim.h(&[QubitId(0)]);
                }
                unitary(&mut sim, gate, 0, 1, theta);
                assert_eq!(
                    sim.active_width(),
                    usize::from(theta != exact),
                    "quarter={quarter}, gate={gate}, theta={theta:?}"
                );
            }
        }
    }
}

#[test]
fn smallest_signed_rotations_retain_relative_phase() {
    let unit = Angle64::new(1);
    for theta in [unit, -unit] {
        let mut sim = StabActive::new(1);
        sim.rx(theta, &[QubitId(0)]);
        let expected = if theta == unit { -1.0 } else { 1.0 } * unit.to_radians() / 2.0;
        assert!(expected.abs() > 0.0);
        assert!((sim.amplitudes[1].im / expected - 1.0).abs() < 1e-12);
        assert_eq!(sim.active_width(), 1);
    }
}

#[test]
fn pure_cliffords_never_promote() {
    let mut rng = PecosRng::seed_from_u64(243);
    let mut sim = StabActive::with_seed(5, 26);
    let mut reference = StateVec::new(5).state();
    for step in 0..300 {
        let q = index(&mut rng, 5);
        let r = (q + 1 + index(&mut rng, 4)) % 5;
        if step % 5 == 0 {
            synchronized_measure(&mut sim, &mut reference, q, rng.next_bool_fast());
        } else {
            let gate = index(&mut rng, 11);
            let angle = Angle64::QUARTER_TURN * (rng.next_u64() % 4);
            unitary(&mut sim, gate, q, r, angle);
            reference_unitary(&mut reference, gate, q, r, angle);
        }
        assert_eq!(sim.active_width(), 0);
        assert_state(&sim, &reference, "Clifford circuit");
    }
}

#[test]
#[should_panic(expected = "active width 2 exceeds limit 1")]
fn width_limit() {
    StabActive::new(2)
        .with_max_active_width(1)
        .rx(Angle64::from_radians(0.37), &[QubitId(0), QubitId(1)]);
}

#[test]
fn seed_reset_and_preparation() {
    let mut first = StabActive::with_seed(4, 64);
    let mut second = StabActive::with_seed(4, 64);
    for _ in 0..100 {
        for sim in [&mut first, &mut second] {
            sim.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
            sim.rx(Angle64::from_radians(0.79), &[QubitId(2), QubitId(3)]);
        }
        let a = first.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        let b = second.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
        assert_eq!(
            a.iter().map(|m| m.outcome).collect::<Vec<_>>(),
            b.iter().map(|m| m.outcome).collect::<Vec<_>>()
        );
        first.reset();
        second.reset();
        assert_eq!(first.active_width(), 0);
        assert_eq!(first.peak_active_width(), 0);
        assert_state(&first, &StateVec::new(4).state(), "reset");
    }
    first.rx(
        Angle64::from_radians(0.89),
        &[QubitId(0), QubitId(1), QubitId(2), QubitId(3)],
    );
    assert_eq!(first.peak_active_width(), 4);
    first.pz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
    assert_eq!(first.peak_active_width(), 4);
    assert_eq!(first.active_width(), 0);
    assert_state(&first, &StateVec::new(4).state(), "pz");
}

pecos_simulators::rotation_test_suite!(StabActive, 4, StabActive::with_seed(4, 42));
pecos_simulators::measurement_stress_test_suite!(StabActive, 4, StabActive::with_seed(4, 42));

#[test]
fn full_stabilizer_suite() {
    pecos_simulators::stabilizer_test_utils::run_full_stabilizer_test_suite(
        &mut StabActive::with_seed(4, 42),
        4,
    );
}

#[test]
fn mixed_support_and_odd_overlap() {
    let angle = Angle64::from_radians(0.37);
    // X in the S-rotated frame has odd overlap of its D and S factors.
    let mut sim = StabActive::new(1);
    let mut reference = StateVec::new(1).state();
    for gate in [9, 1, 9] {
        unitary(&mut sim, gate, 0, 0, angle);
        reference_unitary(&mut reference, gate, 0, 0, angle);
        assert_state(&sim, &reference, "odd D/S overlap");
    }
    let mut overlap_sim = StabActive::new(1);
    overlap_sim.rx(angle, &[QubitId(0)]).sz(&[QubitId(0)]);
    let parts = overlap_sim.parts(&[(0, PauliKindForDecomp::X)]);
    assert_eq!(parts.active_flips, vec![0]);
    assert_eq!(parts.active_signs, vec![0]);
    assert!(parts.phase.im.abs() > 0.5);

    for negative_dormant in [false, true] {
        for promotion in [false, true] {
            for outcome in [false, true] {
                let mut sim = StabActive::new(2);
                let mut reference = StateVec::new(2).state();
                let mut circuit = vec![(9, 0, 0)];
                if negative_dormant {
                    circuit.push((3, 1, 1));
                }
                if promotion {
                    circuit.push((0, 0, 0));
                }
                circuit.extend([(0, 1, 1), (6, 0, 1)]);
                for (gate, q, r) in circuit {
                    unitary(&mut sim, gate, q, r, angle);
                    reference_unitary(&mut reference, gate, q, r, angle);
                    assert_state(&sim, &reference, "mixed support preparation");
                }
                let parts = sim.parts(&[(1, PauliKindForDecomp::Z)]);
                assert_eq!(parts.dormant_flips, vec![1]);
                if promotion {
                    assert_eq!(parts.active_flips, vec![0]);
                    sim.rz(angle, &[QubitId(1)]);
                    reference_unitary(&mut reference, 8, 1, 1, angle);
                    assert_eq!(sim.active_width(), 2);
                } else {
                    assert_eq!(parts.active_signs, vec![0]);
                    synchronized_measure(&mut sim, &mut reference, 1, outcome);
                    assert_eq!(sim.active_width(), 1);
                }
                assert_state(&sim, &reference, "mixed active/dormant update");
            }
        }
    }
}

fn check_roundoff_endpoint(pair: bool) {
    use pecos_stab_tn::stab_mps::measure::EXPECTATION_ENDPOINT_TOLERANCE;

    for outcome in [false, true] {
        let mut sim = StabActive::with_seed(1, 34);
        let mut reference = StateVec::new(1);
        let (a, b) = if pair { (0.37, 0.83) } else { (0.79, 2.91) };
        let a = Angle64::from_radians(a);
        let b = Angle64::from_radians(b);
        let target = if pair {
            Angle64::QUARTER_TURN
        } else {
            Angle64::ZERO
        };
        // A nontrivial global phase leaves both real and imaginary components
        // in the endpoint amplitude, exposing roundoff in its squared norm.
        let phase = Angle64::from_radians(1.23);
        sim.rz(phase, &[QubitId(0)]);
        reference.rz(phase, &[QubitId(0)]);
        for angle in [a, b, target - (a + b)] {
            sim.rx(angle, &[QubitId(0)]);
            reference.rx(angle, &[QubitId(0)]);
        }
        if pair {
            sim.sxdg(&[QubitId(0)]);
            reference.sxdg(&[QubitId(0)]);
        }
        if outcome {
            sim.x(&[QubitId(0)]);
            reference.x(&[QubitId(0)]);
        }
        let parts = sim.parts(&[(0, PauliKindForDecomp::Z)]);
        assert_eq!(parts.measurement_case(), MeasurementCase::Active);
        assert_eq!(!parts.active_flips.is_empty(), pair);
        assert_eq!(sim.active_width(), 1);
        let raw = sim.active_expectation(&parts).re;
        assert!(
            raw.abs() < 1.0,
            "fixture must have a non-exact interior endpoint: {raw}"
        );
        assert!(1.0 - raw.abs() <= EXPECTATION_ENDPOINT_TOLERANCE);
        assert_eq!(raw.is_sign_negative(), outcome);
        let reference = reference.state();
        assert_state(&sim, &reference, "roundoff endpoint before measurement");
        let projected = normalized_z_projection(&reference, 0, outcome, "roundoff endpoint");

        for forced in [Some(!outcome), None] {
            let mut measured = sim.clone();
            let mut untouched_rng = measured.rng().clone();
            let result = match forced {
                Some(value) => measured.mz_forced(0, value),
                None => measured.mz(&[QubitId(0)]).remove(0),
            };
            assert!(
                result.is_deterministic,
                "pair={pair}, outcome={outcome}, raw={raw}"
            );
            assert_eq!(result.outcome, outcome);
            assert_eq!(measured.active_width(), 0);
            assert_state(&measured, &projected, "roundoff endpoint after measurement");
            assert_eq!(measured.rng_mut().next_u64(), untouched_rng.next_u64());
        }
        assert_eq!(
            sim.probability_one(&parts).to_bits(),
            f64::from(outcome).to_bits()
        );
    }
}

#[test]
fn diagonal_roundoff_endpoint_ignores_impossible_force() {
    check_roundoff_endpoint(false);
}

#[test]
fn pair_roundoff_endpoint_ignores_impossible_force() {
    check_roundoff_endpoint(true);
}
