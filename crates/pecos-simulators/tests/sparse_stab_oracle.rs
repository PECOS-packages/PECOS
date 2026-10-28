// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use pecos_core::{AdaptiveIndexSet, BitSet, IndexSet, PauliBitmask, QubitId, SortedVecSet, VecSet};
use pecos_random::PecosRng;
use pecos_simulators::{CliffordGateable, GensGeneric, SparseStabGeneric, SparseStabVecSet};
use rand::RngExt;

#[derive(Clone, Copy, Debug)]
struct SignedPauli {
    pauli: PauliBitmask,
    phase: u8,
}

impl SignedPauli {
    fn multiply(self, rhs: Self) -> Self {
        let (pauli, phase) = self.pauli.multiply_with_phase(&rhs.pauli);
        Self {
            pauli,
            phase: (self.phase + rhs.phase + phase) % 4,
        }
    }

    fn pivot(self) -> Option<usize> {
        if self.pauli.x_bits != 0 {
            Some(self.pauli.x_bits.trailing_zeros() as usize)
        } else if self.pauli.z_bits != 0 {
            Some(128 + self.pauli.z_bits.trailing_zeros() as usize)
        } else {
            None
        }
    }
}

fn bits<S: IndexSet>(set: &S) -> u128 {
    set.iter().fold(0, |mask, q| mask | (1u128 << q))
}

fn rows<S: IndexSet>(gens: &GensGeneric<S>) -> Vec<SignedPauli> {
    // Stored W convention: (-1)^m i^s prod_q X_q^x Z_q^z, so XZ = -iY.
    // Evidence (pre-tie-break lines): gens.rs:399-430 (generator's -i per Y),
    // sparse_stab.rs:703-735 (SZ: X -> iXZ), 749-750 (H: XZ -> -XZ),
    // and 474, 534 (product sign from the Z-left/X-right intersection).
    // PauliBitmask uses Y, hence subtract one quarter turn per x-and-z site.
    gens.row_x
        .iter()
        .zip(&gens.row_z)
        .enumerate()
        .map(|(i, (x, z))| {
            let pauli = PauliBitmask {
                x_bits: bits(x),
                z_bits: bits(z),
            };
            let ys = u8::try_from((pauli.x_bits & pauli.z_bits).count_ones() % 4).unwrap();
            SignedPauli {
                pauli,
                phase: (2 * u8::from(gens.signs_minus.contains(i))
                    + u8::from(gens.signs_i.contains(i))
                    + 4
                    - ys)
                    % 4,
            }
        })
        .collect()
}

struct Basis([Option<SignedPauli>; 256]);

impl Basis {
    fn new(rows: &[SignedPauli]) -> Self {
        let mut basis = Self([None; 256]);
        for &row in rows {
            let mut remainder = row;
            while let Some(pivot) = remainder.pivot() {
                if let Some(existing) = basis.0[pivot] {
                    remainder = remainder.multiply(existing);
                } else {
                    basis.0[pivot] = Some(remainder);
                    break;
                }
            }
            assert!(
                remainder.pivot().is_some() || remainder.phase == 0,
                "inconsistent signed stabilizer group"
            );
        }
        basis
    }

    fn contains(&self, mut row: SignedPauli) -> bool {
        while let Some(pivot) = row.pivot() {
            let Some(existing) = self.0[pivot] else {
                return false;
            };
            row = row.multiply(existing);
        }
        row.phase == 0
    }
}

fn same_group(left: &[SignedPauli], right: &[SignedPauli]) -> bool {
    let left_basis = Basis::new(left);
    let right_basis = Basis::new(right);
    left.iter().all(|&r| right_basis.contains(r)) && right.iter().all(|&r| left_basis.contains(r))
}

fn same_gens<S: IndexSet, T: IndexSet>(left: &GensGeneric<S>, right: &GensGeneric<T>) -> bool {
    let mut a = Vec::new();
    let mut b = Vec::new();
    let mut equal = |x: &S, y: &T| {
        // Compare the public cardinality too: equal elements must not conceal
        // an incorrect wide-mode cache used by measurement pivot selection.
        if x.len() != y.len() {
            return false;
        }
        a.clear();
        b.clear();
        a.extend(x.iter());
        b.extend(y.iter());
        a.sort_unstable();
        b.sort_unstable();
        a == b
    };
    for (x, y) in [
        (&left.row_x, &right.row_x),
        (&left.row_z, &right.row_z),
        (&left.col_x, &right.col_x),
        (&left.col_z, &right.col_z),
    ] {
        if x.len() != y.len() || !x.iter().zip(y).all(|(a, b)| equal(a, b)) {
            return false;
        }
    }
    equal(&left.signs_minus, &right.signs_minus) && equal(&left.signs_i, &right.signs_i)
}

fn same_tableau<S: IndexSet, T: IndexSet>(
    a: &SparseStabGeneric<S>,
    b: &SparseStabGeneric<T>,
) -> bool {
    same_gens(a.stabs(), b.stabs()) && same_gens(a.destabs(), b.destabs())
}

fn valid_views<S: IndexSet>(gens: &GensGeneric<S>) -> bool {
    let n = gens.row_x.len();
    for (rows, cols) in [(&gens.row_x, &gens.col_x), (&gens.row_z, &gens.col_z)] {
        if rows.len() != n || cols.len() != n {
            return false;
        }
        let mut transpose = vec![0u128; n];
        for (i, row) in rows.iter().enumerate() {
            for q in row.iter() {
                if q >= n {
                    return false;
                }
                transpose[q] |= 1u128 << i;
            }
        }
        for (col, expected) in cols.iter().zip(transpose) {
            if col.iter().any(|i| i >= n) || bits(col) != expected {
                return false;
            }
        }
    }
    true
}

fn anticommutes(a: SignedPauli, b: SignedPauli) -> bool {
    ((a.pauli.x_bits & b.pauli.z_bits) ^ (a.pauli.z_bits & b.pauli.x_bits)).count_ones() % 2 == 1
}

fn valid_structure<S: IndexSet>(sim: &SparseStabGeneric<S>) -> bool {
    if !valid_views(sim.stabs()) || !valid_views(sim.destabs()) {
        return false;
    }
    let stabs = rows(sim.stabs());
    let destabs = rows(sim.destabs());
    for (i, &s) in stabs.iter().enumerate() {
        for (j, &d) in destabs.iter().enumerate() {
            if anticommutes(s, d) != (i == j)
                || anticommutes(s, stabs[j])
                || anticommutes(destabs[i], d)
            {
                return false;
            }
        }
    }
    true
}

#[derive(Clone, Copy, Debug)]
enum Op {
    H(usize),
    Sz(usize),
    Cx(usize, usize),
    Cz(usize, usize),
    X(usize),
    Z(usize),
    Measure(usize, bool),
    Prepare(usize, bool),
    RandomMeasure(usize),
}

fn apply<S: IndexSet>(sim: &mut SparseStabGeneric<S>, op: Op) -> Option<(bool, bool)> {
    match op {
        Op::H(q) => {
            sim.h(&[QubitId(q)]);
        }
        Op::Sz(q) => {
            sim.sz(&[QubitId(q)]);
        }
        Op::Cx(a, b) => {
            sim.cx(&[(QubitId(a), QubitId(b))]);
        }
        Op::Cz(a, b) => {
            sim.cz(&[(QubitId(a), QubitId(b))]);
        }
        Op::X(q) => {
            sim.x(&[QubitId(q)]);
        }
        Op::Z(q) => {
            sim.z(&[QubitId(q)]);
        }
        Op::Prepare(q, coin) => {
            sim.pz_forced(q, coin);
        }
        Op::Measure(q, coin) => {
            let result = sim.mz_forced(q, coin);
            return Some((result.is_deterministic, result.outcome));
        }
        Op::RandomMeasure(q) => {
            let result = sim.mz(&[QubitId(q)]).remove(0);
            return Some((result.is_deterministic, result.outcome));
        }
    }
    None
}

fn simulator<S: IndexSet>(n: usize, seed: u64, tracking: bool) -> SparseStabGeneric<S> {
    let sim = SparseStabGeneric::with_rng(n, PecosRng::seed_from_u64(seed));
    if tracking {
        sim.with_destab_sign_tracking()
    } else {
        sim
    }
}

trait Candidate {
    fn apply(&mut self, op: Op) -> Option<(bool, bool)>;
    fn check(&self, reference: &SparseStabVecSet, deep: bool, context: &str);
}

impl<S: IndexSet> Candidate for SparseStabGeneric<S> {
    fn apply(&mut self, op: Op) -> Option<(bool, bool)> {
        apply(self, op)
    }

    fn check(&self, reference: &SparseStabVecSet, deep: bool, context: &str) {
        let name = std::any::type_name::<S>();
        assert!(
            same_tableau(reference, self),
            "(b) tableau: {name}; {context}"
        );
        if deep {
            assert!(valid_structure(self), "(c) structure: {name}; {context}");
            assert!(
                same_group(&rows(reference.stabs()), &rows(self.stabs())),
                "(d) signed group: {name}; {context}"
            );
        }
    }
}

struct Harness {
    reference: SparseStabVecSet,
    candidates: Vec<Box<dyn Candidate>>,
    context: String,
    step: usize,
    deep_gates: bool,
}

impl Harness {
    fn new(n: usize, seed: u64, tracking: bool, deep_gates: bool) -> Self {
        let mut candidates: Vec<Box<dyn Candidate>> = vec![
            Box::new(simulator::<BitSet>(n, seed, tracking)),
            Box::new(simulator::<AdaptiveIndexSet>(n, seed, tracking)),
        ];
        // The large-ID regression targets adaptive mode transitions; keep the
        // slower unsorted candidate in the existing small-circuit corpus.
        if n <= 128 {
            candidates.push(Box::new(simulator::<VecSet<usize>>(n, seed, tracking)));
        }
        Self {
            reference: simulator(n, seed, tracking),
            candidates,
            context: format!("seed={seed}, n={n}, tracking={tracking}"),
            step: 0,
            deep_gates,
        }
    }

    fn step(&mut self, op: Op) -> Option<(bool, bool)> {
        let deep = self.deep_gates
            || matches!(
                op,
                Op::Measure(..) | Op::Prepare(..) | Op::RandomMeasure(..)
            );
        let result = apply(&mut self.reference, op);
        let context = format!("{}, step={}, {op:?}", self.context, self.step);
        for candidate in &mut self.candidates {
            assert_eq!(candidate.apply(op), result, "(a) measurement: {context}");
        }
        if deep || self.reference.num_qubits() <= 128 {
            self.check(deep, &context);
        }
        self.step += 1;
        result
    }

    fn check(&self, deep: bool, context: &str) {
        // Checks (c) and (d) encode qubit and generator IDs in u128 masks,
        // so explicitly omit them above 128 qubits. Full tableau equality (b)
        // remains enabled at every large-circuit measurement/reset checkpoint.
        let deep = deep && self.reference.num_qubits() <= 128;
        if deep {
            assert!(valid_structure(&self.reference), "(c) reference: {context}");
        }
        for candidate in &self.candidates {
            candidate.check(&self.reference, deep, context);
        }
    }

    fn reset(&mut self, q: usize, coin: bool) -> (bool, bool) {
        // Observe the forced branch before applying the same correction as pz_forced.
        let result = self.step(Op::Measure(q, coin)).unwrap();
        if result.1 {
            let deep_gates = self.deep_gates;
            self.deep_gates = true;
            self.step(Op::X(q));
            self.deep_gates = deep_gates;
        }
        result
    }
}

#[test]
fn random_small_circuits() {
    for seed in 0..500 {
        for tracking in [false, true] {
            let mut rng = PecosRng::seed_from_u64(seed);
            let n = rng.random_range(3..=10);
            let mut harness = Harness::new(n, seed, tracking, true);
            for _ in 0..rng.random_range(20..=220) {
                let q = rng.random_range(0..n);
                let other = (q + rng.random_range(1..n)) % n;
                let coin = rng.random();
                let op = match rng.random_range(0..8) {
                    0 => Op::H(q),
                    1 => Op::Sz(q),
                    2 => Op::Cx(q, other),
                    3 => Op::Cz(q, other),
                    4 => Op::X(q),
                    5 => Op::Measure(q, coin),
                    6 => {
                        harness.reset(q, coin);
                        continue;
                    }
                    _ => Op::Prepare(q, coin),
                };
                harness.step(op);
            }
        }
    }
}

struct Check {
    ancilla: usize,
    x_type: bool,
    corners: Vec<usize>,
}

fn surface_checks() -> Vec<Check> {
    let mut checks = Vec::new();
    for r in -1i32..=4 {
        for c in -1i32..=4 {
            let interior = (0..4).contains(&r) && (0..4).contains(&c);
            let boundary = (r == -1 && [1, 3].contains(&c))
                || (r == 4 && [0, 2].contains(&c))
                || (c == -1 && [0, 2].contains(&r))
                || (c == 4 && [1, 3].contains(&r));
            if !interior && !boundary {
                continue;
            }
            let x_type = (r + c) % 2 == 0;
            let order = if x_type {
                [(0, 0), (0, 1), (1, 0), (1, 1)]
            } else {
                [(0, 0), (1, 0), (0, 1), (1, 1)]
            };
            let corners = order
                .iter()
                .filter_map(|&(dr, dc)| {
                    let (row, col) = (r + dr, c + dc);
                    ((0..5).contains(&row) && (0..5).contains(&col))
                        .then(|| usize::try_from(5 * row + col).unwrap())
                })
                .collect();
            checks.push(Check {
                ancilla: 25 + checks.len(),
                x_type,
                corners,
            });
        }
    }
    assert_eq!(checks.len(), 24);
    assert_eq!(checks.iter().filter(|c| c.x_type).count(), 12);
    checks
}

fn surface_run(seed: u64, tracking: bool, noisy: bool) {
    let mut harness = Harness::new(49, seed, tracking, false);
    surface_rounds(&mut harness, seed, noisy, |q| q, false);
}

fn surface_rounds(
    harness: &mut Harness,
    seed: u64,
    noisy: bool,
    id: impl Fn(usize) -> usize,
    random_measurements: bool,
) {
    let checks = surface_checks();
    let mut rng = PecosRng::seed_from_u64(seed);
    for round in 0..8 {
        if noisy {
            for q in 0..25 {
                let q = id(q);
                if rng.random_bool(0.05) {
                    harness.step(if rng.random() { Op::X(q) } else { Op::Z(q) });
                }
                if rng.random_bool(0.05) {
                    harness.reset(q, rng.random());
                }
            }
        }
        for check in &checks {
            if check.x_type {
                harness.step(Op::H(id(check.ancilla)));
            }
        }
        for layer in 0..4 {
            for check in &checks {
                if let Some(&data) = check.corners.get(layer) {
                    harness.step(if check.x_type {
                        Op::Cx(id(check.ancilla), id(data))
                    } else {
                        Op::Cx(id(data), id(check.ancilla))
                    });
                }
            }
        }
        for check in &checks {
            if check.x_type {
                harness.step(Op::H(id(check.ancilla)));
            }
        }
        for check in &checks {
            let coin = noisy && rng.random();
            let result = harness.step(Op::Measure(id(check.ancilla), coin)).unwrap();
            if !noisy {
                assert_eq!(
                    result,
                    (round > 0 || !check.x_type, false),
                    "round={round}, ancilla={}, X={}",
                    check.ancilla,
                    check.x_type
                );
            }
            let reset_result = harness.reset(id(check.ancilla), noisy && rng.random());
            if !noisy {
                assert_eq!(reset_result, (true, false), "reset in round={round}");
            }
        }
        if random_measurements {
            // Use the simulator's seeded RNG without changing the noise/reset
            // coin stream. Repeating a measurement also exercises determinism.
            let q = id((7 * round) % 25);
            harness.step(Op::RandomMeasure(q));
            assert!(harness.step(Op::RandomMeasure(q)).unwrap().0);
        }
        harness.check(true, &format!("{}, end of round={round}", harness.context));
    }
}

#[test]
fn rotated_surface_code_noise_and_resets() {
    for seed in 0..50 {
        for tracking in [false, true] {
            surface_run(seed, tracking, true);
        }
    }
}

#[test]
fn rotated_surface_code_noiseless_schedule() {
    for tracking in [false, true] {
        surface_run(0, tracking, false);
    }
}

#[test]
fn rotated_surface_code_spread_over_large_ids() {
    for seed in 0..3 {
        for tracking in [false, true] {
            // Leave the other qubits idle while both qubit and generator IDs
            // span well beyond the 2048-bit narrow dense limit.
            let id = |q| 43 * q;
            let mut harness = Harness::new(2100, seed, tracking, false);
            // A d=5 check has too few entries to guarantee wide dense storage
            // just by spreading IDs. Build a 49-element X row spanning 33 words
            // so measurement exercises its cached length and take_clearing.
            // Only the same 49 mapped qubits participate; all others stay idle.
            harness.step(Op::H(id(0)));
            for q in 1..49 {
                harness.step(Op::Cx(id(0), id(q)));
            }
            assert!(!harness.step(Op::RandomMeasure(id(0))).unwrap().0);
            for q in 0..49 {
                harness.reset(id(q), false);
            }
            surface_rounds(&mut harness, seed, true, id, true);
        }
    }
}

#[test]
fn seeded_unforced_measurements() {
    for tracking in [false, true] {
        let mut harness = Harness::new(7, 0x5eed, tracking, true);
        let mut rng = PecosRng::seed_from_u64(314);
        let mut deterministic = 0;
        let mut random = 0;
        for _ in 0..500 {
            let q = rng.random_range(0..7);
            harness.step(Op::H(q));
            harness.step(Op::Sz((q + 1) % 7));
            harness.step(Op::Cx(q, (q + 1) % 7));
            for target in [q, q, (q + 2) % 7] {
                if harness.step(Op::RandomMeasure(target)).unwrap().0 {
                    deterministic += 1;
                } else {
                    random += 1;
                }
            }
        }
        assert!(deterministic > 0 && random > 0);
    }
}

// Mutation helpers intentionally use the stored XZ product, independently of
// both PauliBitmask multiplication and the production GensGeneric helpers.
fn multiply_row<S: IndexSet>(
    target: &mut GensGeneric<S>,
    i: usize,
    source: &GensGeneric<S>,
    j: usize,
) {
    let phase = |g: &GensGeneric<S>, k| {
        2 * u8::from(g.signs_minus.contains(k)) + u8::from(g.signs_i.contains(k))
    };
    let crossings = (bits(&target.row_z[i]) & bits(&source.row_x[j])).count_ones() % 2;
    let product_phase =
        (phase(target, i) + phase(source, j) + 2 * u8::try_from(crossings).unwrap()) % 4;
    for q in source.row_x[j].iter() {
        target.row_x[i].toggle(q);
        target.col_x[q].toggle(i);
    }
    for q in source.row_z[j].iter() {
        target.row_z[i].toggle(q);
        target.col_z[q].toggle(i);
    }
    target.signs_minus.remove(i);
    target.signs_i.remove(i);
    if product_phase >= 2 {
        target.signs_minus.insert(i);
    }
    if product_phase % 2 == 1 {
        target.signs_i.insert(i);
    }
}

fn fixture() -> SparseStabVecSet {
    let mut sim = simulator(3, 0, true);
    sim.h(&[QubitId(0), QubitId(1)])
        .sz(&[QubitId(0), QubitId(1)])
        .cx(&[(QubitId(0), QubitId(2))]);
    assert!(valid_structure(&sim));
    sim
}

fn change_basis(sim: &mut SparseStabVecSet) {
    let old = sim.clone();
    let (stabs, destabs) = sim.stabs_and_destabs_mut();
    multiply_row(stabs, 0, old.stabs(), 1);
    // The OTHER generator's partner must change: S0 <- S0 S1, D1 <- D1 D0.
    multiply_row(destabs, 1, old.destabs(), 0);
}

fn signed(phase: u8, paulis: &[(char, usize)]) -> SignedPauli {
    let mut pauli = PauliBitmask {
        x_bits: 0,
        z_bits: 0,
    };
    for &(p, q) in paulis {
        if matches!(p, 'X' | 'Y') {
            pauli.x_bits |= 1 << q;
        }
        if matches!(p, 'Z' | 'Y') {
            pauli.z_bits |= 1 << q;
        }
    }
    SignedPauli { pauli, phase }
}

#[test]
fn oracle_group_membership_pins_phases() {
    // Bell state: XX and ZZ generate a group containing -YY, not +YY.
    let mut bell = simulator::<SortedVecSet>(2, 0, false);
    bell.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
    let group = Basis::new(&rows(bell.stabs()));
    assert!(group.contains(signed(0, &[('X', 0), ('X', 1)])));
    assert!(group.contains(signed(0, &[('Z', 0), ('Z', 1)])));
    assert!(group.contains(signed(2, &[('Y', 0), ('Y', 1)])));
    assert!(!group.contains(signed(0, &[('Y', 0), ('Y', 1)])));

    // SZ on qubit 0 maps XX to YX, stored as i (XZ)_0 X_1. The group then holds
    // YX * ZZ = (YZ)(XZ) = (iX)(-iY) = +XY.
    bell.sz(&[QubitId(0)]);
    let group = Basis::new(&rows(bell.stabs()));
    assert!(group.contains(signed(0, &[('Y', 0), ('X', 1)])));
    assert!(!group.contains(signed(2, &[('Y', 0), ('X', 1)])));
    assert!(group.contains(signed(0, &[('X', 0), ('Y', 1)])));
    assert!(!group.contains(signed(2, &[('X', 0), ('Y', 1)])));
}

#[test]
fn oracle_rejects_stabilizer_sign_flip() {
    let original = fixture();
    let mut changed = original.clone();
    changed.stabs_mut().signs_minus.toggle(0);
    assert!(!same_tableau(&original, &changed));
    assert!(!same_group(&rows(original.stabs()), &rows(changed.stabs())));
}

#[test]
fn oracle_rejects_missing_column_entry() {
    let mut changed = fixture();
    let q = changed.stabs().row_x[0].iter().next().unwrap();
    changed.stabs_mut().col_x[q].remove(0);
    assert!(!valid_structure(&changed));
}

#[test]
fn oracle_rejects_unpaired_destabilizer_product() {
    let mut changed = fixture();
    let source = changed.stabs().clone();
    multiply_row(changed.destabs_mut(), 0, &source, 1);
    assert!(!valid_structure(&changed));
}

#[test]
fn oracle_accepts_signed_basis_change_but_rejects_tableau_change() {
    let original = fixture();
    let mut changed = original.clone();
    change_basis(&mut changed);
    assert!(valid_structure(&changed));
    assert!(same_group(&rows(original.stabs()), &rows(changed.stabs())));
    assert!(!same_tableau(&original, &changed));
}

#[test]
fn oracle_rejects_wrong_basis_change_phase() {
    let original = fixture();
    let mut changed = original.clone();
    change_basis(&mut changed);
    changed.stabs_mut().signs_minus.toggle(0);
    assert!(valid_structure(&changed));
    assert!(!same_group(&rows(original.stabs()), &rows(changed.stabs())));
}

#[test]
fn default_sparse_exports_match_bitset_order() {
    use pecos_simulators::{SparseStab, SparseStabBitSet};

    let mut adaptive = SparseStab::with_seed(4097, 1434);
    let mut bitset = SparseStabBitSet::with_seed(4097, 1434);
    for q in [4096, 2048, 7, 0] {
        adaptive.h(&[QubitId(q)]);
        bitset.h(&[QubitId(q)]);
    }
    for (control, target) in [(4096, 0), (7, 4096), (2048, 7), (0, 2048)] {
        adaptive.cx(&[(QubitId(control), QubitId(target))]);
        bitset.cx(&[(QubitId(control), QubitId(target))]);
    }
    for is_stab in [true, false] {
        assert_eq!(adaptive.gens_data(is_stab), bitset.gens_data(is_stab));
    }
    assert_eq!(
        adaptive.stabs().anticommuting_indices([4096, 7], [2048, 0]),
        bitset.stabs().anticommuting_indices([4096, 7], [2048, 0])
    );
    assert_eq!(
        adaptive
            .stabs()
            .classify_pauli(adaptive.destabs(), [4096, 7], [2048, 0]),
        bitset
            .stabs()
            .classify_pauli(bitset.destabs(), [4096, 7], [2048, 0])
    );
}
