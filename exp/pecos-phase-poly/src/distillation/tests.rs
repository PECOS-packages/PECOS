// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::*;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec};
use std::collections::BTreeSet;
use std::time::Instant;

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a} != {b}");
}

fn bits(rows: &[&str]) -> Vec<Vec<bool>> {
    rows.iter()
        .map(|row| row.chars().map(|c| c == '1').collect())
        .collect()
}

fn paper_example() -> TriorthogonalMatrix {
    TriorthogonalMatrix::new(
        &bits(&[
            "11111110000000",
            "00000001111111",
            "10101011010101",
            "01100110110011",
            "00011110001111",
        ]),
        2,
    )
    .unwrap()
}

#[test]
fn matrices_and_validation() {
    for k in (2..=12).step_by(2) {
        let matrix = TriorthogonalMatrix::bravyi_haah(k).unwrap();
        assert_eq!((matrix.m(), matrix.n(), matrix.k()), (k + 3, 3 * k + 8, k));
        let weights: Vec<_> = matrix.rows.iter().map(|r| r.ones().count()).collect();
        assert_eq!(&weights[..k], vec![7; k]);
        assert_eq!(&weights[k..], [4 + 2 * k, 4 + 2 * k, 8]);
    }
    assert_eq!(TriorthogonalMatrix::rm15().m(), 5);
    assert_eq!(paper_example().n(), 14);
    let error = TriorthogonalMatrix::new(&bits(&["110", "101"]), 0).unwrap_err();
    assert!(error.contains("odd pair overlap"), "{error}");
    let error = TriorthogonalMatrix::new(&bits(&["1010101", "0110011", "0001111"]), 0).unwrap_err();
    assert!(error.contains("odd triple overlap"), "{error}");
    for (rows, k, message) in [
        (bits(&[""]), 0, "dependent"),
        (bits(&["11", "1"]), 0, "length"),
        (bits(&["1"]), 2, "exceeds"),
        (bits(&["11"]), 1, "odd weight"),
        (bits(&["1"]), 0, "even weight"),
        (bits(&["11", "11"]), 0, "dependent"),
        (bits(&["00"]), 0, "dependent"),
    ] {
        assert!(
            TriorthogonalMatrix::new(&rows, k)
                .unwrap_err()
                .contains(message)
        );
    }
    for k in [0, 1, 3, usize::MAX - 1] {
        assert!(TriorthogonalMatrix::bravyi_haah(k).is_err());
    }
}

fn polynomial(n: usize, terms: &[(usize, u64)]) -> Vec<u64> {
    let mut coefficients = vec![0; n + 1];
    for &(degree, coefficient) in terms {
        coefficients[degree] += coefficient;
    }
    coefficients
}

// Coefficient of p^degree in W(1-2p), using integer binomial coefficients.
fn coefficient(weights: &[u64], degree: usize) -> i128 {
    weights
        .iter()
        .enumerate()
        .filter(|&(w, _)| w >= degree)
        .map(|(w, &count)| {
            let mut choose = 1_i128;
            for j in 0..degree {
                choose = choose * i128::try_from(w - j).unwrap() / i128::try_from(j + 1).unwrap();
            }
            i128::from(count) * choose * (-2_i128).pow(u32::try_from(degree).unwrap())
        })
        .sum()
}

#[test]
fn enumerators_and_exact_leading_coefficients() {
    for k in (2..=12).step_by(2) {
        let matrix = TriorthogonalMatrix::bravyi_haah(k).unwrap();
        let weights = matrix.weight_enumerators().unwrap();
        assert_eq!(
            weights.even,
            polynomial(matrix.n(), &[(0, 1), (8, 1), (4 + 2 * k, 6)])
        );
        let coset = polynomial(matrix.n(), &[(7, 2), (3 + 2 * k, 6)]);
        for c in &weights.cosets {
            assert_eq!(*c, coset);
            assert_eq!(coefficient(c, 0), 8);
            assert_eq!(coefficient(&weights.even, 1) - coefficient(c, 1), 0);
            assert_eq!(
                coefficient(&weights.even, 2) - coefficient(c, 2),
                16 * i128::try_from(1 + 3 * k).unwrap()
            );
        }
        assert_eq!(
            coefficient(&weights.even, 1),
            -8 * i128::try_from(matrix.n()).unwrap()
        );
        close(weights.oracles(0.0).unwrap().p_s, 1.0);
        close(weights.oracles(0.5).unwrap().p_s, 0.125);
        for q in weights.oracles(0.5).unwrap().q_a {
            close(q, 0.5);
        }
        for q in weights.oracles(1.0).unwrap().q_a {
            close(q, 1.0);
        }
        for p in [f64::NAN, f64::INFINITY, -0.01, 1.01] {
            assert!(weights.oracles(p).is_err());
            assert!(matrix.circuit(p).is_err());
        }
    }
    let weights = TriorthogonalMatrix::rm15().weight_enumerators().unwrap();
    assert_eq!(weights.even, polynomial(15, &[(0, 1), (8, 15)]));
    assert_eq!(weights.cosets[0], polynomial(15, &[(7, 15), (15, 1)]));
    for degree in 0..3 {
        assert_eq!(
            coefficient(&weights.even, degree),
            coefficient(&weights.cosets[0], degree)
        );
    }
    assert_eq!(
        coefficient(&weights.even, 3) - coefficient(&weights.cosets[0], 3),
        32 * 35
    );
}

fn check_pattern(
    matrix: &TriorthogonalMatrix,
    sim: &mut PhasePoly,
    ops: &[Op],
    pattern: &[bool],
) -> bool {
    let expected = matrix.pattern_oracle(pattern).unwrap();
    let mut calls = Vec::new();
    let result = run_shot(sim, ops, &mut |index, q, p| {
        assert!(matches!(ops[index], Op::ZError(qubit, rate) if qubit.0 == q && rate.to_bits() == p.to_bits()));
        calls.push(q);
        pattern[q]
    })
    .unwrap();
    assert_eq!(calls, (0..matrix.n()).collect::<Vec<_>>());
    assert_eq!(result.accepted, expected.accepted);
    for (actual, expected) in result.syndrome.iter().zip(&expected.syndrome) {
        assert!(actual.is_deterministic);
        assert_eq!(actual.outcome, *expected);
    }
    assert_eq!(result.logical_w.is_some(), expected.logical_w.is_some());
    if let Some(actual) = result.logical_w {
        let expected = expected.logical_w.unwrap();
        assert_eq!(actual.len(), matrix.k());
        for (&a, &b) in actual.iter().zip(&expected) {
            close(a, b);
        }
        expected.contains(&-1.0)
    } else {
        false
    }
}

#[test]
fn ideal_circuits_and_changed_row_bases() {
    let mut matrices = vec![TriorthogonalMatrix::rm15(), paper_example()];
    matrices.extend(
        (2..=10)
            .step_by(2)
            .map(|k| TriorthogonalMatrix::bravyi_haah(k).unwrap()),
    );
    // Preserve triorthogonality while making the original coefficient map
    // nontrivial: add even rows to logical rows and change the even-row basis.
    let mut rows = paper_example().rows();
    for (a, b) in [(0, 2), (1, 3), (2, 4)] {
        let source = rows[b].clone();
        for (target, bit) in rows[a].iter_mut().zip(source) {
            *target ^= bit;
        }
    }
    for row in &mut rows {
        row.rotate_left(3);
    }
    matrices.push(TriorthogonalMatrix::new(&rows, 2).unwrap());
    // Degenerate boundary cases: no logical rows and no syndrome rows.
    matrices.push(TriorthogonalMatrix::new(&[], 0).unwrap());
    matrices.push(TriorthogonalMatrix::new(&bits(&["11"]), 0).unwrap());
    matrices.push(TriorthogonalMatrix::new(&bits(&["100", "010"]), 2).unwrap());
    for matrix in matrices {
        let ops = matrix.circuit(0.0).unwrap();
        let mut sim = PhasePoly::with_seed(matrix.n(), 7);
        check_pattern(&matrix, &mut sim, &ops, &vec![false; matrix.n()]);
        // Reuse the same simulator after an arbitrary prior shot.
        check_pattern(&matrix, &mut sim, &ops, &vec![true; matrix.n()]);
        check_pattern(&matrix, &mut sim, &ops, &vec![false; matrix.n()]);
    }
}

#[test]
fn dense_statevec_oracle() {
    for matrix in [
        TriorthogonalMatrix::bravyi_haah(2).unwrap(),
        TriorthogonalMatrix::rm15(),
        paper_example(),
    ] {
        let ops = matrix.circuit(0.0).unwrap();
        let mut dense = StateVec::new(matrix.n());
        for op in &ops {
            match op {
                Op::PZ(q) => {
                    dense.pz(&[*q]);
                }
                Op::PX(q) => {
                    dense.px(&[*q]);
                }
                Op::CX(a, b) => {
                    dense.cx(&[(*a, *b)]);
                }
                Op::CZ(a, b) => {
                    dense.cz(&[(*a, *b)]);
                }
                Op::S(q) => {
                    dense.sz(&[*q]);
                }
                Op::Sdg(q) => {
                    dense.szdg(&[*q]);
                }
                Op::Z(q) => {
                    dense.z(&[*q]);
                }
                Op::T(q) => {
                    dense.t(&[*q]);
                }
                Op::ZError(..) => {}
                Op::MeasureX(_) | Op::ExpectW(_) => break,
            }
        }
        // Independent target amplitudes from original row coefficients.
        let mut target = vec![num_complex::Complex64::new(0.0, 0.0); 1 << matrix.n()];
        let magnitude = 1.0 / f64::from(1_u32 << matrix.m()).sqrt();
        for assignment in 0_usize..1 << matrix.m() {
            let mut word = Mask::zero(matrix.n());
            let mut logical_weight = 0_u32;
            for a in 0..matrix.m() {
                if assignment & (1 << a) != 0 {
                    word.xor(&matrix.rows[a]);
                    logical_weight += u32::from(a < matrix.k());
                }
            }
            let index = word.ones().fold(0, |index, q| index | (1 << q));
            target[index] = num_complex::Complex64::from_polar(
                magnitude,
                f64::from(logical_weight) * std::f64::consts::FRAC_PI_4,
            );
        }
        let anchor = target.iter().position(|a| a.norm() > 0.0).unwrap();
        let dense_state = dense.state();
        let phase = dense_state[anchor] / target[anchor];
        for (a, b) in dense_state.iter().zip(&target) {
            assert!((*a - phase * b).norm() < 1e-12);
        }
        let mut sim = PhasePoly::with_seed(matrix.n(), 1);
        run_shot(&mut sim, &ops, &mut |_, _, _| false).unwrap();
        let actual = sim.state_vector();
        let phase = actual[anchor] / target[anchor];
        for (a, b) in actual.iter().zip(&target) {
            assert!((*a - phase * b).norm() < 1e-12);
        }
    }
}

fn random_index(rng: &mut PecosRng, n: usize) -> usize {
    usize::try_from(rng.next_u64() % u64::try_from(n).unwrap()).unwrap()
}

#[test]
fn exact_error_patterns() {
    for (label, matrix) in [
        ("rm15", TriorthogonalMatrix::rm15()),
        ("G(2)", TriorthogonalMatrix::bravyi_haah(2).unwrap()),
        ("G(4)", TriorthogonalMatrix::bravyi_haah(4).unwrap()),
        ("G(6)", TriorthogonalMatrix::bravyi_haah(6).unwrap()),
    ] {
        let n = matrix.n();
        let ops = matrix.circuit(0.05).unwrap();
        let mut sim = PhasePoly::with_seed(n, 123);
        let mut rng = PecosRng::seed_from_u64(456);
        let mut patterns = BTreeSet::new();
        patterns.insert(vec![false; n]);
        for a in 0..n {
            let mut e = vec![false; n];
            e[a] = true;
            patterns.insert(e);
            for b in a + 1..n {
                let mut e = vec![false; n];
                e[a] = true;
                e[b] = true;
                patterns.insert(e);
            }
        }
        // 300 distinct patterns at each of weights 3, 4, and 5.
        for weight in 3..=5 {
            let target = patterns.len() + 300;
            while patterns.len() < target {
                let mut e = vec![false; n];
                while e.iter().filter(|&&b| b).count() < weight {
                    e[random_index(&mut rng, n)] = true;
                }
                patterns.insert(e);
            }
        }
        let mut flipped = [0; 6];
        for e in &patterns {
            if check_pattern(&matrix, &mut sim, &ops, e) {
                flipped[e.iter().filter(|&&b| b).count()] += 1;
            }
        }
        println!(
            "{label}: patterns={}, accepted with any logical flip by weight 0..5: {flipped:?}",
            patterns.len()
        );
        assert!(flipped[if matrix.k() == 1 { 3 } else { 2 }] > 0);
    }
}

#[test]
fn statistical_oracles() {
    // Two-sided Bernstein bound for Bernoulli means: with L=ln(2/delta),
    // error <= sqrt(2 q(1-q)L/N) + 2L/(3N). Union bound over 18 comparisons
    // gives failure probability <= 18e-8 < 1e-6. Conditional accepted samples
    // remain iid; condition on their count before applying the bound.
    let log = (2.0_f64 / 1e-8).ln();
    let bound = |q: f64, n: f64| (2.0 * q * (1.0 - q) * log / n).sqrt() + 2.0 * log / (3.0 * n);
    for (matrix, p, seed) in [
        (TriorthogonalMatrix::bravyi_haah(4).unwrap(), 0.05, 2026),
        (TriorthogonalMatrix::rm15(), 0.05, 2027),
        (TriorthogonalMatrix::bravyi_haah(10).unwrap(), 0.02, 2028),
    ] {
        let oracle = matrix.weight_enumerators().unwrap().oracles(p).unwrap();
        let ops = matrix.circuit(p).unwrap();
        let mut sim = PhasePoly::with_seed(matrix.n(), seed);
        let mut rng = PecosRng::seed_from_u64(seed);
        let mut accepted = 0_u32;
        let mut flips = vec![0_u32; matrix.k()];
        let shots = 20_000_u32;
        for _ in 0..shots {
            let shot = run_shot_sampled(&mut sim, &ops, &mut rng).unwrap();
            if let Some(w) = shot.logical_w {
                accepted += 1;
                for (count, value) in flips.iter_mut().zip(w) {
                    *count += u32::from(value < 0.0);
                }
            }
        }
        let rate = f64::from(accepted) / f64::from(shots);
        assert!((rate - oracle.p_s).abs() <= bound(oracle.p_s, f64::from(shots)));
        assert!(accepted > 0);
        for (&count, &q) in flips.iter().zip(&oracle.q_a) {
            let rate = f64::from(count) / f64::from(accepted);
            assert!(
                (rate - q).abs() <= bound(q, f64::from(accepted)),
                "k={} observed={rate}, oracle={q}",
                matrix.k()
            );
        }
        println!(
            "statistical k={} p={p}: accepted={accepted}/{shots}, flips={flips:?}; joint failure <= 1.8e-7",
            matrix.k()
        );
    }
}

#[test]
fn scale_smoke() {
    let matrix = TriorthogonalMatrix::bravyi_haah(40).unwrap();
    assert_eq!(matrix.n(), 128);
    let ops = matrix.circuit(0.0).unwrap();
    let mut sim = PhasePoly::with_seed(matrix.n(), 7);
    let start = Instant::now();
    for _ in 0..3 {
        check_pattern(&matrix, &mut sim, &ops, &vec![false; matrix.n()]);
    }
    println!(
        "G(40), n=128: {:.3} us/ideal shot",
        start.elapsed().as_secs_f64() * 1e6 / 3.0
    );
}

#[test]
fn emitted_dialect() {
    let matrix = TriorthogonalMatrix::bravyi_haah(4).unwrap();
    let text = to_stim(&matrix.circuit(0.05).unwrap());
    assert_eq!(text.lines().filter(|l| l.starts_with("T ")).count(), 20);
    assert_eq!(
        text.lines()
            .filter(|l| l.starts_with("Z_ERROR(0.05)"))
            .count(),
        20
    );
    assert_eq!(text.matches("DETECTOR rec[-1]").count(), 3);
    assert_eq!(
        text.lines().filter(|l| l.starts_with("EXP_VAL ")).count(),
        4
    );
    assert_eq!(text.matches(" !Y").count(), 4);
    let one = TriorthogonalMatrix::new(&bits(&["1"]), 1).unwrap();
    assert!(to_stim(&one.circuit(0.0).unwrap()).contains("EXP_VAL X0 Y0"));
    assert!(one.pattern_oracle(&[]).is_err());
}
