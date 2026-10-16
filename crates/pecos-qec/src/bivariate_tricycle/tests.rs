// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::*;
use crate::connected_cluster_code_distance;
use std::collections::BTreeSet;

fn codes() -> [BivariateTricycleCode; 3] {
    [
        (3, [(2, 1), (2, 2)], [(0, 0), (1, 2)], [(1, 0), (2, 1)]),
        (5, [(1, 0), (0, 2)], [(0, 0), (1, 2)], [(1, 0), (1, 3)]),
        (9, [(1, 3), (2, 1)], [(0, 0), (1, 8)], [(2, 4), (2, 6)]),
    ]
    .map(|(m, a, b, c)| BivariateTricycleCode::new(3, m, &a, &b, &c).unwrap())
}

#[test]
fn parameters_checks_and_meta_checks() {
    for (code, n) in codes().iter().zip([27, 45, 81]) {
        assert_eq!(code.num_qubits(), n);
        assert_eq!(code.num_logical_qubits(), 3);
        assert_eq!(code.hx().num_checks(), n / 3);
        assert_eq!(code.hz().num_checks(), n);
        assert_eq!(code.meta_z().num_checks(), n / 3);
        for (matrix, weight) in [(code.hx(), 6), (code.hz(), 4)] {
            assert!(
                matrix
                    .rows()
                    .iter()
                    .all(|row| row.iter().copied().map(usize::from).sum::<usize>() == weight)
            );
        }
        assert_eq!(
            code.hx().matrix().mul(&code.hz().matrix().transpose()),
            F2Matrix::zeros(n / 3, n)
        );
        assert_eq!(
            code.meta_z().matrix().mul(code.hz().matrix()),
            F2Matrix::zeros(n / 3, n)
        );
    }
}

#[test]
fn exact_distances() {
    for (code, distance) in codes().iter().zip([3, 4, 5]) {
        let result =
            connected_cluster_code_distance(code.hx(), code.logical_x(), distance).unwrap();
        assert_eq!(result.distance, distance);
        assert!(
            connected_cluster_code_distance(code.hz(), code.logical_z(), distance - 1).is_none()
        );
    }
}

#[test]
fn ccz_layers() {
    for code in codes() {
        let n = code.num_qubits();
        let layers = code.cup_product_ccz();
        let mut triples = BTreeSet::new();
        for layer in &layers {
            assert_eq!(layer.len(), n);
            for block in 0..3 {
                let mut indices = layer.iter().map(|gate| gate[block]).collect::<Vec<_>>();
                indices.sort_unstable();
                assert_eq!(indices, (0..n).collect::<Vec<_>>());
            }
            for &gate in layer {
                assert!(triples.insert(gate));
                assert_eq!(
                    gate.map(|q| q / (n / 3))
                        .into_iter()
                        .collect::<BTreeSet<_>>(),
                    BTreeSet::from([0, 1, 2])
                );
            }
        }
        assert_eq!(triples.len(), 2 * n);
    }
}

fn evaluate(gates: &[[usize; 3]], vectors: [&[u8]; 3]) -> u8 {
    gates.iter().fold(0, |parity, &[a, b, c]| {
        parity ^ (vectors[0][a] & vectors[1][b] & vectors[2][c])
    })
}

fn invariant(code: &BivariateTricycleCode, gates: &[[usize; 3]]) -> bool {
    let kernel = code.hz().matrix().kernel();
    code.hx().rows().iter().all(|check| {
        kernel.iter().all(|left| {
            kernel.iter().all(|right| {
                [
                    [check.as_slice(), left, right],
                    [left, check, right],
                    [left, right, check],
                ]
                .into_iter()
                .all(|vectors| evaluate(gates, vectors) == 0)
            })
        })
    })
}

#[test]
fn coboundary_invariance_and_missing_gate_control() {
    for code in codes() {
        let mut gates = code.cup_product_ccz().concat();
        assert!(invariant(&code, &gates));
        gates.remove(0);
        assert!(!invariant(&code, &gates));
    }
}

#[test]
fn sector_uniform_logical_action() {
    for code in codes() {
        let n = code.num_qubits();
        let uniform = (0..3)
            .map(|s| {
                (0..n)
                    .map(|q| u8::from(q / (n / 3) == s))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        for row in &uniform {
            assert!(code.hz().rows().iter().all(|check| {
                check
                    .iter()
                    .zip(row)
                    .fold(0, |parity, (&a, &b)| parity ^ (a & b))
                    == 0
            }));
        }
        let mut extended = code.hx().rows();
        extended.extend(uniform.clone());
        assert_eq!(
            F2Matrix::from_rows(extended).row_reduce().1.len(),
            code.hx().rank() + 3
        );
        let gates = code.cup_product_ccz().concat();
        for r in 0..3 {
            for s in 0..3 {
                for t in 0..3 {
                    assert_eq!(
                        evaluate(&gates, [&uniform[r], &uniform[s], &uniform[t]]),
                        u8::from(r != s && s != t && r != t)
                    );
                }
            }
        }
    }
}

#[test]
fn invalid_constructors() {
    let valid = [(0, 0), (1, 1)];
    for polynomial in 0..3 {
        for count in [0, 1, 3] {
            let bad = vec![(0, 0); count];
            let mut terms: [&[BbMonomial]; 3] = [&valid; 3];
            terms[polynomial] = &bad;
            assert!(
                matches!(BivariateTricycleCode::new(3, 3, terms[0], terms[1], terms[2]), Err(BivariateTricycleError::WrongTermCount { actual, .. }) if actual == count)
            );
        }
        for (bad, duplicate) in [
            ([(0, 0), (0, 0)], true),
            ([(3, 0), (0, 0)], false),
            ([(0, 0), (0, 3)], false),
        ] {
            let mut terms: [&[BbMonomial]; 3] = [&valid; 3];
            terms[polynomial] = &bad;
            let error = BivariateTricycleCode::new(3, 3, terms[0], terms[1], terms[2]).unwrap_err();
            if duplicate {
                assert!(matches!(
                    error,
                    BivariateTricycleError::DuplicateMonomial { .. }
                ));
            } else {
                assert!(matches!(
                    error,
                    BivariateTricycleError::ExponentOutOfRange { .. }
                ));
            }
        }
    }
    for (l, m) in [(0, 3), (3, 0), (0, 0)] {
        assert_eq!(
            BivariateTricycleCode::new(l, m, &valid, &valid, &valid),
            Err(BivariateTricycleError::ZeroDimension { l, m })
        );
    }
    for (l, m) in [(usize::MAX, 2), (usize::MAX / 2, 1)] {
        assert_eq!(
            BivariateTricycleCode::new(l, m, &valid, &valid, &valid),
            Err(BivariateTricycleError::SizeOverflow { l, m })
        );
    }
}
