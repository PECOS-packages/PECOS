// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use nalgebra::DMatrix;
use num_complex::Complex64;
use pecos_core::Pauli;
use pecos_core::clifford::Clifford;
use pecos_core::{Angle64, BitSet, QubitId};
use pecos_quantum::ToMatrix;
use pecos_random::PecosRng;
use pecos_simulators::{CliffordGateable, Gens, MeasurementResult, QuantumSimulator, SparseStabY};
use rand::RngExt;
use std::collections::BTreeSet;
use std::fmt::Write;

/// Only the four required operations are forwarded. Every optional operation
/// must exercise the trait decomposition, including its recursive calls.
struct Decomposed(SparseStabY);

impl QuantumSimulator for Decomposed {
    fn num_qubits(&self) -> usize {
        self.0.num_qubits()
    }
    fn reset(&mut self) -> &mut Self {
        self.0.reset();
        self
    }
}

impl CliffordGateable for Decomposed {
    fn sz(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.0.sz(qubits);
        self
    }
    fn h(&mut self, qubits: &[QubitId]) -> &mut Self {
        self.0.h(qubits);
        self
    }
    fn cx(&mut self, pairs: &[(QubitId, QubitId)]) -> &mut Self {
        self.0.cx(pairs);
        self
    }
    fn mz(&mut self, qubits: &[QubitId]) -> Vec<MeasurementResult> {
        self.0.mz(qubits)
    }
}

// One inventory drives dispatch and independent per-method tests. The source
// coverage test below catches newly added trait methods as well as omissions.
macro_rules! gates {
    (single: [$($single:ident),*]; pair: [$($pair:ident),*]; measure: [$($measure:ident),*]) => {
        const SINGLE: &[&str] = &[$(stringify!($single)),*];
        const PAIR: &[&str] = &[$(stringify!($pair)),*];
        const MEASURE: &[&str] = &[$(stringify!($measure)),*];
        fn apply<S: CliffordGateable>(sim: &mut S, name: &str, targets: &[QubitId]) -> Vec<(bool, bool)> {
            match name {
                $(stringify!($single) => { sim.$single(targets); Vec::new() },)*
                $(stringify!($pair) => {
                    let pairs: Vec<_> = targets.as_chunks::<2>().0.iter().map(|p| (p[0], p[1])).collect();
                    sim.$pair(&pairs); Vec::new()
                },)*
                $(stringify!($measure) => sim.$measure(targets).into_iter()
                    .map(|r| (r.outcome, r.is_deterministic)).collect(),)*
                "apply_global_phase" => {
                    sim.apply_global_phase(Angle64::QUARTER_TURN / 2u64, targets);
                    Vec::new()
                }
                _ => panic!("unlisted CliffordGateable method: {name}"),
            }
        }
        mod oracle {
            use super::*;
            $(#[test] fn $single() { check_gate(stringify!($single), false); })*
            $(#[test] fn $pair() { check_gate(stringify!($pair), true); })*
            $(#[test] fn $measure() { check_gate(stringify!($measure), false); })*
            #[test] fn apply_global_phase() { check_gate("apply_global_phase", false); }
        }
    }
}

gates! {
    single: [identity, x, y, z, sx, sxdg, sy, sydg, sz, szdg,
        h, h2, h3, h4, h5, h6, f, fdg, f2, f2dg, f3, f3dg, f4, f4dg,
        px, pnx, py, pny, pz, pnz];
    pair: [cx, cy, cz, sxx, sxxdg, syy, syydg, szz, szzdg, swap, iswap, g, iswapdg, gdg];
    measure: [mx, mnx, my, mny, mz, mnz, mpx, mpnx, mpy, mpny, mpz, mpnz]
}

fn all_methods() -> Vec<&'static str> {
    SINGLE
        .iter()
        .chain(PAIR)
        .chain(MEASURE)
        .copied()
        .chain(["apply_global_phase"])
        .collect()
}

#[test]
fn inventory_covers_entire_trait() {
    let declared: BTreeSet<_> = include_str!("../src/clifford_gateable.rs")
        .lines()
        .filter_map(|line| line.strip_prefix("    fn "))
        .map(|line| line.split('(').next().unwrap())
        .collect();
    let listed: BTreeSet<_> = all_methods().into_iter().collect();
    assert_eq!(
        listed, declared,
        "update the oracle inventory for every trait method"
    );
}

fn prefix(num_qubits: usize, seed: u64) -> SparseStabY {
    let mut sim = SparseStabY::with_seed(num_qubits, seed).with_destab_sign_tracking();
    let mut rng = PecosRng::seed_from_u64(seed);
    // Required gates only: a mutation of a tested gate cannot contaminate the
    // inputs of other gate tests. Vary depth, including the identity tableau.
    for _ in 0..seed % 97 {
        let first = QubitId(rng.random_range(0..num_qubits));
        match rng.random_range(0..3) {
            0 => {
                sim.h(&[first]);
            }
            1 => {
                sim.sz(&[first]);
            }
            _ if num_qubits > 1 => {
                let second =
                    QubitId((first.index() + rng.random_range(1..num_qubits)) % num_qubits);
                sim.cx(&[(first, second)]);
            }
            _ => {
                sim.h(&[first]);
            }
        }
    }
    sim
}

fn assert_set(actual: &BitSet, expected: &BitSet, label: &str) {
    // BitSet's derived Eq compares backing storage: [] and [0] represent
    // identical bits. Compare all members, without dropping any sign bits.
    assert!(
        actual.iter().eq(expected.iter()),
        "{label}: {actual:?} != {expected:?}"
    );
}

fn assert_gens(actual: &Gens, expected: &Gens) {
    for (left, right, label) in [
        (&actual.row_x, &expected.row_x, "row X"),
        (&actual.row_z, &expected.row_z, "row Z"),
        (&actual.col_x, &expected.col_x, "column X"),
        (&actual.col_z, &expected.col_z, "column Z"),
    ] {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert_set(left, right, label);
        }
    }
    assert_set(&actual.signs_minus, &expected.signs_minus, "minus signs");
    assert_set(&actual.signs_i, &expected.signs_i, "i signs");
}

fn placements(num_qubits: usize, pair: bool) -> Vec<Vec<QubitId>> {
    let mut targets = vec![Vec::new()];
    for first in 0..num_qubits {
        if pair {
            for second in 0..num_qubits {
                if first != second {
                    targets.push(vec![QubitId(first), QubitId(second)]);
                }
            }
        } else {
            targets.push(vec![QubitId(first)]);
        }
    }
    // Valid disjoint batches, both orientations.
    if !pair || num_qubits >= 4 {
        let count = if pair { num_qubits / 2 * 2 } else { num_qubits };
        targets.push((0..count).map(QubitId).collect());
        targets.push((0..count).rev().map(QubitId).collect());
    }
    targets
}

fn check_gate(name: &str, pair: bool) {
    for num_qubits in 1..=5 {
        for seed in 0..64 {
            let before = prefix(num_qubits, seed);
            for targets in placements(num_qubits, pair) {
                let mut native = before.clone();
                let mut decomposed = Decomposed(before.clone());
                assert_eq!(
                    apply(&mut native, name, &targets),
                    apply(&mut decomposed, name, &targets),
                    "{name}, n={num_qubits}, seed={seed}, targets={targets:?}"
                );
                assert_gens(native.stabs(), decomposed.0.stabs());
                assert_gens(native.destabs(), decomposed.0.destabs());
            }
        }
    }
}

// Lossless packing, not a hash: n <= 5 needs 4*n*n + 2*n <= 110 bits.
fn tableau_bits(gens: &Gens, num_qubits: usize) -> u128 {
    let mut bits = 0u128;
    for sets in [&gens.row_x, &gens.row_z, &gens.col_x, &gens.col_z] {
        for set in sets {
            for qubit in 0..num_qubits {
                bits = (bits << 1) | u128::from(set.contains(qubit));
            }
        }
    }
    for set in [&gens.signs_minus, &gens.signs_i] {
        for row in 0..num_qubits {
            bits = (bits << 1) | u128::from(set.contains(row));
        }
    }
    bits
}

fn untracked_transcript() -> String {
    let mut transcript = String::new();
    for num_qubits in 1..=5 {
        for seed in [0, 0xdead_beef] {
            let mut sim = SparseStabY::with_seed(num_qubits, seed);
            let mut rng = PecosRng::seed_from_u64(seed);
            for _ in 0..2 {
                for name in all_methods() {
                    if num_qubits == 1 && PAIR.contains(&name) {
                        continue;
                    }
                    let first = rng.random_range(0..num_qubits);
                    let mut targets = vec![QubitId(first)];
                    if PAIR.contains(&name) {
                        targets.push(QubitId(
                            (first + rng.random_range(1..num_qubits)) % num_qubits,
                        ));
                    }
                    let outcomes = apply(&mut sim, name, &targets);
                    writeln!(
                        transcript,
                        "{num_qubits} {seed:x} {name} {:x} {:x} {outcomes:?}",
                        tableau_bits(sim.stabs(), num_qubits),
                        tableau_bits(sim.destabs(), num_qubits)
                    )
                    .unwrap();
                }
            }
        }
    }
    transcript
}

#[test]
fn tracking_off_matches_dev() {
    // Captured from dev 500769bcf4696922182a04464213d4ebefded623 with this
    // exact fixed-seed circuit generator; every line preserves all tableau bits.
    assert_eq!(
        untracked_transcript(),
        include_str!("data/sparse_stab_y_untracked.txt")
    );
}

fn gate_matrix(gate: Clifford, targets: &[QubitId], num_qubits: usize) -> DMatrix<Complex64> {
    let entries = if gate.is_1q() {
        gate.canonical_1q_matrix().unwrap().to_vec()
    } else {
        gate.canonical_2q_matrix().unwrap().to_vec()
    };
    let local_dim = 1 << targets.len();
    let local = DMatrix::from_row_slice(
        local_dim,
        local_dim,
        &entries
            .as_chunks::<2>()
            .0
            .iter()
            .map(|entry| Complex64::new(entry[0], entry[1]))
            .collect::<Vec<_>>(),
    );
    let mask = targets
        .iter()
        .fold(0, |mask, qubit| mask | (1 << qubit.index()));
    DMatrix::from_fn(1 << num_qubits, 1 << num_qubits, |row, col| {
        if row & !mask != col & !mask {
            return Complex64::ZERO;
        }
        // Canonical two-qubit matrices put the first target in the high bit.
        let extract = |basis: usize| {
            targets.iter().fold(0, |index, qubit| {
                (index << 1) | ((basis >> qubit.index()) & 1)
            })
        };
        local[(extract(row), extract(col))]
    })
}

fn row_matrix(gens: &Gens, row: usize, num_qubits: usize) -> DMatrix<Complex64> {
    let mut matrix = DMatrix::identity(1, 1);
    for qubit in (0..num_qubits).rev() {
        let pauli = match (
            gens.row_x[row].contains(qubit),
            gens.row_z[row].contains(qubit),
        ) {
            (false, false) => Pauli::I,
            (true, false) => Pauli::X,
            (false, true) => Pauli::Z,
            (true, true) => Pauli::Y,
        };
        matrix = matrix.kronecker(pauli.to_matrix().inner());
    }
    let mut phase = if gens.signs_i.contains(row) {
        Complex64::I
    } else {
        Complex64::ONE
    };
    if gens.signs_minus.contains(row) {
        phase = -phase;
    }
    matrix * phase
}

fn dense_prefix(num_qubits: usize, seed: u64) -> (SparseStabY, DMatrix<Complex64>) {
    let mut sim = SparseStabY::with_seed(num_qubits, seed).with_destab_sign_tracking();
    let mut unitary = DMatrix::identity(1 << num_qubits, 1 << num_qubits);
    let mut rng = PecosRng::seed_from_u64(seed);
    for _ in 0..seed % 31 {
        let first = QubitId(rng.random_range(0..num_qubits));
        let (name, gate, targets) = match rng.random_range(0..3) {
            0 => ("h", Clifford::H, vec![first]),
            1 => ("sz", Clifford::SZ, vec![first]),
            _ if num_qubits > 1 => (
                "cx",
                Clifford::CX,
                vec![
                    first,
                    QubitId((first.index() + rng.random_range(1..num_qubits)) % num_qubits),
                ],
            ),
            _ => ("h", Clifford::H, vec![first]),
        };
        apply(&mut sim, name, &targets);
        unitary = gate_matrix(gate, &targets, num_qubits) * unitary;
    }
    (sim, unitary)
}

#[test]
fn fixed_gates_match_dense_conjugation() {
    let fixed = [
        ("h2", Clifford::H2),
        ("h3", Clifford::H3),
        ("h4", Clifford::H4),
        ("h5", Clifford::H5),
        ("h6", Clifford::H6),
        ("f2", Clifford::F2),
        ("f2dg", Clifford::F2dg),
        ("f3", Clifford::F3),
        ("f3dg", Clifford::F3dg),
        ("f4", Clifford::F4),
        ("f4dg", Clifford::F4dg),
        ("szz", Clifford::SZZ),
        ("szzdg", Clifford::SZZdg),
    ];
    for num_qubits in 1..=3 {
        let initial = SparseStabY::with_seed(num_qubits, 0);
        for seed in 0..32 {
            let (before, prefix_matrix) = dense_prefix(num_qubits, seed);
            for (name, gate) in fixed {
                for targets in placements(num_qubits, !gate.is_1q())
                    .into_iter()
                    .filter(|targets| targets.len() == if gate.is_1q() { 1 } else { 2 })
                {
                    let mut after = before.clone();
                    apply(&mut after, name, &targets);
                    let circuit = gate_matrix(gate, &targets, num_qubits) * &prefix_matrix;
                    for (gens, initial_gens) in [
                        (after.stabs(), initial.stabs()),
                        (after.destabs(), initial.destabs()),
                    ] {
                        for row in 0..num_qubits {
                            let expected = &circuit
                                * row_matrix(initial_gens, row, num_qubits)
                                * circuit.adjoint();
                            let actual = row_matrix(gens, row, num_qubits);
                            assert!(
                                (actual - expected).norm() < 1e-9,
                                "{name}, n={num_qubits}, seed={seed}, targets={targets:?}, row={row}"
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn forced_preparation_matches_measurement_then_x() {
    for num_qubits in 1..=5 {
        for seed in 0..64 {
            for qubit in 0..num_qubits {
                for outcome in [false, true] {
                    let mut native = prefix(num_qubits, seed);
                    let mut expected = native.clone();
                    let result = expected.mz_forced(qubit, outcome);
                    if result.outcome {
                        expected.x(&[QubitId(qubit)]);
                    }
                    native.pz_forced(qubit, outcome);
                    assert_gens(native.stabs(), expected.stabs());
                    assert_gens(native.destabs(), expected.destabs());
                }
            }
        }
    }
}
