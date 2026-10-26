// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use pecos_core::{AdaptiveIndexSet, BitSet, IndexSet, PauliBitmask, QubitId, SortedVecSet, VecSet};
use pecos_random::PecosRng;
use pecos_simulators::{
    CliffordGateable, GensGeneric, MeasurementBatchOrder, MeasurementRule, SparseStabGeneric,
    SparseStabVecSet,
};
use rand::RngExt;
use rand::seq::SliceRandom;

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
        // Check both directions without a fixed-width mask, including n > 128.
        for (r, row) in rows.iter().enumerate() {
            if row.iter().any(|q| q >= n || !cols[q].contains(r)) {
                return false;
            }
        }
        for (q, col) in cols.iter().enumerate() {
            if col.iter().any(|r| r >= n || !rows[r].contains(q)) {
                return false;
            }
        }
    }
    true
}

fn assert_hermitian<S: IndexSet>(sim: &SparseStabGeneric<S>, context: &str) {
    for (name, gens) in [("stabilizer", sim.stabs()), ("destabilizer", sim.destabs())] {
        if name == "destabilizer" && !sim.tracks_destab_signs() {
            continue;
        }
        for (r, (x, z)) in gens.row_x.iter().zip(&gens.row_z).enumerate() {
            // In W convention, (XZ)^dagger = -XZ. Thus i flag = |X & Z| mod 2
            // is necessary and sufficient for a real phase in the Hermitian Pauli basis.
            assert_eq!(
                gens.signs_i.contains(r),
                x.intersection_count(z) % 2 == 1,
                "Hermiticity: {name} row {r}; {context}"
            );
        }
    }
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
    fn check(
        &self,
        reference: &SparseStabVecSet,
        refactored: &SparseStabVecSet,
        deep: bool,
        context: &str,
    );
    fn check_decoupled(&self, q: usize, context: &str);
}

impl<S: IndexSet> Candidate for SparseStabGeneric<S> {
    fn apply(&mut self, op: Op) -> Option<(bool, bool)> {
        apply(self, op)
    }

    fn check(
        &self,
        reference: &SparseStabVecSet,
        refactored: &SparseStabVecSet,
        deep: bool,
        context: &str,
    ) {
        let name = format!(
            "{} / {:?}",
            std::any::type_name::<S>(),
            self.measurement_rule()
        );
        assert_hermitian(self, context);
        if deep {
            assert!(valid_structure(self), "(c) structure: {name}; {context}");
            assert!(
                same_group(&rows(reference.stabs()), &rows(self.stabs())),
                "(d) signed group: {name}; {context}"
            );
        }
        let same_rule = match self.measurement_rule() {
            MeasurementRule::AaronsonGottesman => reference,
            MeasurementRule::RefactorDecouple => refactored,
        };
        assert!(
            same_tableau(same_rule, self),
            "(b) tableau: {name}; {context}"
        );
    }

    fn check_decoupled(&self, q: usize, context: &str) {
        if self.measurement_rule() == MeasurementRule::RefactorDecouple {
            assert_decoupled(self, q, context);
        }
    }
}

fn assert_decoupled<S: IndexSet>(sim: &SparseStabGeneric<S>, q: usize, context: &str) {
    let (s, d) = (sim.stabs(), sim.destabs());
    assert_eq!(d.col_x[q].len(), 1, "self-check: unique partner; {context}");
    let r0 = d.col_x[q].iter().next().unwrap();
    assert_eq!(
        d.row_x[r0].iter().collect::<Vec<_>>(),
        [q],
        "self-check: X_q partner; {context}"
    );
    assert!(
        d.row_z[r0].is_empty(),
        "self-check: partner has no Z; {context}"
    );
    assert!(
        s.row_x[r0].is_empty() && !s.signs_i.contains(r0),
        "self-check: real Z partner; {context}"
    );
    assert_eq!(
        s.row_z[r0].iter().collect::<Vec<_>>(),
        [q],
        "self-check: Z_q support; {context}"
    );
    assert!(
        valid_views(s) && valid_views(d),
        "self-check: row/column agreement; {context}"
    );
    // The checked columns prove no other
    // row touches q: the measured qubit is an exact tensor factor, including
    // every deterministic check measurement in noiseless surface rounds 1 onward.
    assert!(
        s.col_x[q].is_empty() && d.col_z[q].is_empty(),
        "self-check: no other support on q; {context}"
    );
    assert_eq!(
        s.col_z[q].iter().collect::<Vec<_>>(),
        [r0],
        "self-check: isolated Z_q; {context}"
    );
}

struct Harness {
    reference: SparseStabVecSet,
    refactored: SparseStabVecSet,
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
            Box::new(
                simulator::<BitSet>(n, seed, tracking)
                    .with_measurement_rule(MeasurementRule::RefactorDecouple),
            ),
            Box::new(
                simulator::<AdaptiveIndexSet>(n, seed, tracking)
                    .with_measurement_rule(MeasurementRule::RefactorDecouple),
            ),
        ];
        // The large-ID regression targets adaptive mode transitions; keep the
        // slower unsorted candidates for both rules in the small-circuit corpus.
        if n <= 128 {
            candidates.push(Box::new(simulator::<VecSet<usize>>(n, seed, tracking)));
            candidates.push(Box::new(
                simulator::<VecSet<usize>>(n, seed, tracking)
                    .with_measurement_rule(MeasurementRule::RefactorDecouple),
            ));
        }
        Self {
            reference: simulator(n, seed, tracking),
            refactored: simulator(n, seed, tracking)
                .with_measurement_rule(MeasurementRule::RefactorDecouple),
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
        assert_eq!(
            apply(&mut self.refactored, op),
            result,
            "(a) refactor measurement: {context}"
        );
        for candidate in &mut self.candidates {
            assert_eq!(candidate.apply(op), result, "(a) measurement: {context}");
        }
        if deep || self.reference.num_qubits() <= 128 {
            self.check(deep, &context);
        }
        if let Op::Measure(q, _) | Op::RandomMeasure(q) | Op::Prepare(q, _) = op {
            assert_decoupled(&self.refactored, q, &context);
            for candidate in &self.candidates {
                candidate.check_decoupled(q, &context);
            }
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
        assert_hermitian(&self.reference, context);
        assert_hermitian(&self.refactored, context);
        if deep {
            assert!(
                valid_structure(&self.refactored),
                "(c) refactored: {context}"
            );
            assert!(
                same_group(
                    &rows(self.reference.stabs()),
                    &rows(self.refactored.stabs())
                ),
                "(d) refactored signed group: {context}"
            );
        }
        for candidate in &self.candidates {
            candidate.check(&self.reference, &self.refactored, deep, context);
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
fn deterministic_refactor_dual_update_and_sign() {
    for tracking in [false, true] {
        let mut harness = Harness::new(2, 0, tracking, true);
        // S=(Z0,Z0 Z1), D=(X0 X1,X1). R={0,1}, stabs.col_z[1]={1},
        // and destabs.col_z[1] is empty: this exercises refactor_two_sources.
        // Scores (-1,1) select row 1; the shortcut applies D0 <- D0 D1.
        // X1 below makes the deterministic measurement outcome 1.
        harness.step(Op::Cx(0, 1));
        harness.step(Op::X(1));
        assert_eq!(harness.step(Op::Measure(1, false)), Some((true, true)));
        assert!(harness.refactored.stabs().signs_minus.contains(1));
        assert_eq!(
            harness.refactored.destabs().row_x[0]
                .iter()
                .collect::<Vec<_>>(),
            [0]
        );
        assert!(harness.refactored.destabs().col_x[1].contains(1));
    }
}

// Assert the complete CSS supports used to establish each branch's entry state
// and the resulting dual basis. Harness still checks signed groups and phases.
fn assert_css_supports(sim: &SparseStabVecSet, stabs: &[u128], destabs: &[u128]) {
    assert!(sim.stabs().row_x.iter().all(IndexSet::is_empty));
    assert!(sim.destabs().row_z.iter().all(IndexSet::is_empty));
    assert_eq!(
        sim.stabs().row_z.iter().map(bits).collect::<Vec<_>>(),
        stabs
    );
    assert_eq!(
        sim.destabs().row_x.iter().map(bits).collect::<Vec<_>>(),
        destabs
    );
}

#[test]
fn deterministic_refactor_general_two_sources() {
    for tracking in [false, true] {
        let mut harness = Harness::new(3, 0, tracking, true);
        harness.step(Op::Cx(0, 1));
        harness.step(Op::Cx(1, 2));
        assert_css_supports(
            &harness.refactored,
            &[0b001, 0b011, 0b110],
            &[0b111, 0b110, 0b100],
        );
        assert_eq!(bits(&harness.refactored.destabs().col_x[1]), 0b011);
        assert_eq!(bits(&harness.refactored.stabs().col_z[1]), 0b110);
        // Inspection: R={0,1}, but stabs.col_z[1]={1,2}, so the two-source
        // shortcut declines. The asserted supports give scores (-2,0), hence
        // r0=1 with |D1|=2: refactor_measured_z takes the fused column branch.
        // Z1 is deterministic with outcome 0; the forced true coin is ignored.
        assert_eq!(harness.step(Op::Measure(1, true)), Some((true, false)));
        // The dual update and stripping leave D=(X0,X1,X2), S=(Z0,Z1,Z2).
        assert_css_supports(
            &harness.refactored,
            &[0b001, 0b010, 0b100],
            &[0b001, 0b010, 0b100],
        );
    }
}

#[test]
fn deterministic_refactor_general_three_sources_and_strip() {
    for tracking in [false, true] {
        let mut harness = Harness::new(4, 0, tracking, true);
        harness.step(Op::Cx(0, 1));
        harness.step(Op::Cx(1, 2));
        harness.step(Op::Cx(2, 3));
        assert_css_supports(
            &harness.refactored,
            &[0b0001, 0b0011, 0b0110, 0b1100],
            &[0b1111, 0b1110, 0b1100, 0b1000],
        );
        assert_eq!(bits(&harness.refactored.destabs().col_x[2]), 0b0111);
        assert_eq!(bits(&harness.refactored.stabs().col_z[2]), 0b1100);
        // Inspection: |R|=3 bypasses both shortcuts. Scores (-7,-4,-2) select
        // r0=2 with |D2|=2, so the fused column branch updates partners {0,1}.
        // S3=Z2 Z3 must then be stripped to Z3 by decoupling.
        assert_eq!(harness.step(Op::Measure(2, true)), Some((true, false)));
        assert_css_supports(
            &harness.refactored,
            &[0b0001, 0b0011, 0b0100, 0b1000],
            &[0b0011, 0b0010, 0b0100, 0b1000],
        );
    }
}

#[test]
fn deterministic_refactor_general_weight_one_partner() {
    for tracking in [false, true] {
        let mut harness = Harness::new(3, 0, tracking, true);
        harness.step(Op::Cx(0, 1));
        harness.step(Op::Cx(1, 2));
        assert_css_supports(
            &harness.refactored,
            &[0b001, 0b011, 0b110],
            &[0b111, 0b110, 0b100],
        );
        assert_eq!(bits(&harness.refactored.destabs().col_x[2]), 0b111);
        assert_eq!(bits(&harness.refactored.stabs().col_z[2]), 0b100);
        // Inspection: |R|=3 bypasses both shortcuts even though only S2 has Z2.
        // Scores (-5,-2,0) select r0=2 with D2=X2: refactor_measured_z takes
        // the weight-one branch, toggling X2 in both partners {0,1}.
        assert_eq!(harness.step(Op::Measure(2, true)), Some((true, false)));
        assert_css_supports(
            &harness.refactored,
            &[0b001, 0b011, 0b100],
            &[0b011, 0b010, 0b100],
        );
    }
}

#[test]
fn random_measurement_decouples_bell_pair() {
    let mut harness = Harness::new(2, 0, false, true);
    harness.step(Op::H(0));
    harness.step(Op::Cx(0, 1));
    assert_eq!(harness.step(Op::Measure(0, true)), Some((false, true)));
    assert_eq!(harness.step(Op::Measure(1, false)), Some((true, true)));
}

#[test]
fn measurement_rule_survives_clone_and_reset() {
    let sim = simulator::<SortedVecSet>(2, 0, false);
    assert_eq!(sim.measurement_rule(), MeasurementRule::AaronsonGottesman);
    let sim = sim.with_measurement_rule(MeasurementRule::RefactorDecouple);
    let mut cloned = sim.clone();
    cloned.reset();
    assert_eq!(cloned.measurement_rule(), MeasurementRule::RefactorDecouple);
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

// Exact signed W rows: (X mask, Z mask, minus, i).
fn assert_signed_destabs(sim: &SparseStabVecSet, expected: &[(u128, u128, bool, bool)]) {
    let d = sim.destabs();
    let actual: Vec<_> = (0..sim.num_qubits())
        .map(|r| {
            (
                bits(&d.row_x[r]),
                bits(&d.row_z[r]),
                d.signs_minus.contains(r),
                d.signs_i.contains(r),
            )
        })
        .collect();
    assert_eq!(actual, expected);
    assert_hermitian(sim, "signed fixture");
}

#[test]
fn tracked_two_source_negative_dual_product() {
    let mut h = Harness::new(2, 0, true, true);
    h.step(Op::Z(1));
    h.step(Op::Cx(0, 1));
    // D=(X0 X1,-X1), S=(Z0,Z0 Z1). Scores (-1,1) select r0=1.
    // D0*(D1) = (X0 X1)*(-X1) = -X0; the fresh D1 is +X1.
    assert_eq!(h.step(Op::Measure(1, false)), Some((true, false)));
    assert_signed_destabs(&h.refactored, &[(1, 0, true, false), (2, 0, false, false)]);
}

#[test]
fn tracked_single_source_clears_imaginary_phase() {
    let mut h = Harness::new(1, 0, true, true);
    h.step(Op::Sz(0));
    // SZ maps X to iXZ=Y. S0=Z0, R={0}, but D0 has Z0, so decouple
    // replaces that entire iXZ row by +X, including clearing the i flag.
    assert_eq!(h.step(Op::Measure(0, true)), Some((true, false)));
    assert_signed_destabs(&h.refactored, &[(1, 0, false, false)]);
}

#[test]
fn tracked_default_random_dual_product_phase() {
    let mut h = Harness::new(2, 0, true, true);
    h.step(Op::H(0));
    h.step(Op::Cx(0, 1));
    h.step(Op::Sz(1));
    // S0=i X0 X1 Z1, S1=Z0 Z1; D0=Z0, D1=i X1 Z1.
    // MZ1 selects S0. D1*S0 = (Y1)*(X0 Y1)=X0:
    // the two i factors and the Z1/X1 crossing each contribute -1.
    assert_eq!(h.step(Op::Measure(1, true)), Some((false, true)));
    assert_signed_destabs(&h.reference, &[(3, 2, false, true), (1, 0, false, false)]);
}

#[test]
fn tracked_default_random_carries_pivot_minus() {
    let mut h = Harness::new(2, 0, true, true);
    for op in [Op::H(0), Op::Z(0), Op::Cx(0, 1)] {
        h.step(op);
    }
    // S0=-X0 X1, S1=Z0 Z1; D0=Z0, D1=X1. MZ1 pivots on S0:
    // D1*S0 = X1*(-X0 X1) = -X0, and D0 takes the old S0 = -X0 X1.
    assert_eq!(h.step(Op::Measure(1, false)), Some((false, false)));
    assert_signed_destabs(&h.reference, &[(3, 0, true, false), (1, 0, true, false)]);
}

#[test]
fn tracked_isolated_measurement_resets_partner_phase() {
    let mut h = Harness::new(1, 0, true, true);
    h.step(Op::Z(0));
    // S0=Z0, D0=-X0. MZ0 is deterministic and already isolated; the partner
    // is rewritten as a fresh +X0.
    assert_eq!(h.step(Op::Measure(0, false)), Some((true, false)));
    assert_signed_destabs(&h.refactored, &[(1, 0, false, false)]);
}

#[test]
fn general_three_sources_outcome_one() {
    let mut h = Harness::new(4, 0, true, true);
    for op in [Op::Cx(0, 1), Op::Cx(1, 2), Op::Cx(2, 3), Op::X(2)] {
        h.step(op);
    }
    assert_eq!(bits(&h.refactored.destabs().col_x[2]), 7);
    // R={0,1,2}, scores (-7,-4,-2): general path selects r0=2.
    // Z0*(Z0 Z1)*(-Z1 Z2)=-Z2, so outcome=1; S3=-Z2 Z3 strips to +Z3.
    assert_eq!(h.step(Op::Measure(2, false)), Some((true, true)));
    assert_eq!(bits(&h.refactored.destabs().col_x[2]), 4);
    assert_eq!(bits(&h.refactored.stabs().signs_minus), 4);
}

#[test]
fn general_non_css_crossings_and_i_phases() {
    let mut h = Harness::new(4, 0, true, true);
    for op in [
        Op::Cx(0, 1),
        Op::Cx(1, 2),
        Op::Cx(2, 3),
        Op::H(0),
        Op::Sz(0),
        Op::X(2),
    ] {
        h.step(op);
    }
    assert_eq!(bits(&h.refactored.destabs().col_x[2]), 7);
    // S0=iX0Z0, S1=iX0Z0 Z1, S2=-Z1 Z2. Their product is -Z2:
    // i*i=-1 cancels the Z0/X0 crossing. Three sources force the general
    // path; scores (-6,-3,-2) still select r0=2.
    assert!(h.refactored.stabs().signs_i.contains(0));
    assert!(h.refactored.stabs().signs_i.contains(1));
    assert_eq!(h.step(Op::Measure(2, false)), Some((true, true)));
    assert_eq!(bits(&h.refactored.destabs().col_x[2]), 4);
    assert_eq!(bits(&h.refactored.stabs().signs_minus), 4);
}

#[test]
fn general_fused_destabilizer_z_loop() {
    let mut h = Harness::new(3, 0, true, true);
    for op in [Op::Cx(0, 1), Op::Cx(1, 2), Op::Sz(2)] {
        h.step(op);
    }
    assert_eq!(bits(&h.refactored.stabs().col_z[1]), 6);
    assert_eq!(bits(&h.refactored.destabs().col_x[1]), 3);
    // Two S rows carry Z1, so the shortcut declines. D=(iX0X1X2Z2,
    // iX1X2Z2,iX2Z2), scores (-3,-1), r0=1 has nonempty row_z.
    // D0*D1=X0 (i*i and one crossing cancel); D1 is replaced by X1.
    assert_eq!(bits(&h.refactored.destabs().row_z[1]), 4);
    assert_eq!(h.step(Op::Measure(1, true)), Some((true, false)));
    assert_signed_destabs(
        &h.refactored,
        &[
            (1, 0, false, false),
            (2, 0, false, false),
            (4, 4, false, true),
        ],
    );
}

#[test]
fn two_source_shortcut_declines_destabilizer_z() {
    let mut h = Harness::new(2, 0, true, true);
    h.step(Op::Cx(0, 1));
    h.step(Op::Sz(1));
    assert_eq!(bits(&h.refactored.destabs().col_x[1]), 3);
    assert_eq!(bits(&h.refactored.stabs().col_z[1]), 2);
    assert_eq!(bits(&h.refactored.destabs().col_z[1]), 3);
    // The sole failed shortcut condition is D's Z1 column. Scores (-2,0)
    // select r0=1. (iX0X1Z1)*(iX1Z1)=X0, then D1 becomes X1.
    assert_eq!(h.step(Op::Measure(1, true)), Some((true, false)));
    assert_signed_destabs(&h.refactored, &[(1, 0, false, false), (2, 0, false, false)]);
}

#[test]
fn single_source_not_isolated() {
    let mut h = Harness::new(2, 0, true, true);
    h.step(Op::Cx(0, 1));
    assert_eq!(bits(&h.refactored.destabs().col_x[0]), 1);
    assert_eq!(bits(&h.refactored.stabs().col_z[0]), 3);
    // R={0}: no refactor, but S1=Z0Z1 must strip to Z1 and D0=X0X1
    // must become X0. The nonempty extra Z0 entry rules out the isolated shortcut.
    assert_eq!(h.step(Op::Measure(0, true)), Some((true, false)));
    assert_css_supports(&h.refactored, &[1, 2], &[1, 2]);
    assert_signed_destabs(&h.refactored, &[(1, 0, false, false), (2, 0, false, false)]);
}

#[test]
fn two_source_sign_differs_from_pivot_sign() {
    let mut h = Harness::new(2, 0, true, true);
    h.step(Op::X(0));
    h.step(Op::Cx(0, 1));
    // |10> becomes |11>. S=(-Z0,+Z0Z1), so the two-source product is -Z1
    // although the score-selected S1 has positive sign.
    assert_eq!(bits(&h.refactored.stabs().signs_minus), 1);
    assert_eq!(bits(&h.refactored.destabs().col_x[1]), 3);
    assert_eq!(bits(&h.refactored.stabs().col_z[1]), 2);
    assert!(h.refactored.destabs().col_z[1].is_empty());
    assert_eq!(h.step(Op::Measure(1, false)), Some((true, true)));
    assert_eq!(bits(&h.refactored.destabs().col_x[1]), 2);
}

#[test]
#[should_panic(expected = "Hermiticity: destabilizer")]
fn oracle_rejects_stale_destabilizer_i() {
    let mut sim = simulator::<SortedVecSet>(1, 0, true);
    sim.destabs_mut().signs_i.insert(0);
    assert_hermitian(&sim, "stale iX");
}

#[test]
fn general_score_tie_selects_lowest_id() {
    let mut h = Harness::new(2, 0, true, true);
    h.step(Op::Sz(0));
    h.step(Op::Cx(1, 0));
    // S=(Z0Z1,Z1), D=(iX0Z0Z1,X0X1). R={0,1}; both scores are -1.
    // D's nonempty Z0 column excludes the shortcut; lowest id selects r0=0.
    assert_eq!(bits(&h.refactored.destabs().col_x[0]), 3);
    assert_eq!(bits(&h.refactored.destabs().col_z[0]), 1);
    for r in 0..2 {
        assert_eq!(
            h.refactored.stabs().generator_weight(r) as i128
                - h.refactored.destabs().generator_weight(r) as i128,
            -1
        );
    }
    assert_eq!(h.step(Op::Measure(0, true)), Some((true, false)));
    assert_eq!(bits(&h.refactored.destabs().col_x[0]), 1);
    // D1*D0=iX1Z0Z1; removing +Z0 leaves iX1Z1=Y1.
    assert_signed_destabs(&h.refactored, &[(1, 0, false, false), (2, 2, false, true)]);
}

#[test]
fn tracked_strip_multiplies_by_negative_measured_z() {
    let mut h = Harness::new(2, 0, true, true);
    for op in [Op::Sz(0), Op::Cx(1, 0), Op::X(0)] {
        h.step(op);
    }
    // The tied scores select r0=0 as above, now S0=-Z0Z1 and D0=-iX0Z0Z1.
    // D1*D0=-iX1Z0Z1; multiplying by measured -Z0 gives +iX1Z1.
    assert_eq!(h.step(Op::Measure(0, false)), Some((true, true)));
    assert_signed_destabs(&h.refactored, &[(1, 0, false, false), (2, 2, false, true)]);
}

#[test]
fn tracked_default_replacement_copies_full_source_phase() {
    for (ops, expected) in [
        (vec![Op::Sz(0), Op::H(0)], (1, 0, false, false)),
        (vec![Op::H(0), Op::Sz(0)], (1, 1, false, true)),
        (vec![Op::H(0), Op::Sz(0), Op::Z(0)], (1, 1, true, true)),
    ] {
        let mut h = Harness::new(1, 0, true, true);
        for op in ops {
            h.step(op);
        }
        // SZ,H leaves S0=+X and D0=-Y: copying S0 clears both stale flags.
        // H,SZ leaves S0=+Y=iXZ; a final Z makes it -Y=-iXZ. Copying
        // either Y row must keep its i flag to remain Hermitian in W convention.
        assert_eq!(h.step(Op::Measure(0, false)), Some((false, false)));
        assert_signed_destabs(&h.reference, &[expected]);
    }
}

#[test]
fn general_weight_one_partner_matches_signed_basis_product() {
    let mut h = Harness::new(3, 0, true, true);
    for op in [Op::Z(2), Op::Cx(0, 1), Op::Cx(1, 2)] {
        h.step(op);
    }
    assert_eq!(bits(&h.refactored.destabs().col_x[2]), 7);
    // R={0,1,2}, S=(Z0,Z0Z1,Z1Z2), D=(X0X1X2,X1X2,-X2).
    // Scores (-5,-2,0) select r0=2 with weight one. Independent full products
    // D0*D2=-X0X1 and D1*D2=-X1, S0*S1*S2=+Z2; fresh D2=+X2.
    let scores: Vec<_> = (0..3)
        .map(|r| {
            h.refactored.stabs().generator_weight(r) as i128
                - 2 * h.refactored.destabs().generator_weight(r) as i128
        })
        .collect();
    assert_eq!(scores, [-5, -2, 0]);
    assert_eq!(h.step(Op::Measure(2, true)), Some((true, false)));
    assert_css_supports(&h.refactored, &[1, 3, 4], &[3, 2, 4]);
    assert!(h.refactored.stabs().signs_minus.is_empty());
    assert!(h.refactored.stabs().signs_i.is_empty());
    assert_signed_destabs(
        &h.refactored,
        &[
            (3, 0, true, false),
            (2, 0, true, false),
            (4, 0, false, false),
        ],
    );
}

// Batch oracle: the AG reference remains AsGiven and receives single-qubit
// measurements in the independently replayed order, with the same coin stream.
struct BatchHarness<S: IndexSet> {
    candidate: SparseStabGeneric<S>,
    reference: SparseStabVecSet,
    context: String,
}

impl<S: IndexSet> BatchHarness<S> {
    fn new(n: usize, seed: u64, rule: MeasurementRule, tracking: bool) -> Self {
        Self {
            candidate: simulator(n, seed, tracking)
                .with_measurement_rule(rule)
                .with_measurement_batch_order(MeasurementBatchOrder::BitReversed),
            reference: simulator(n, seed, tracking),
            context: format!(
                "{} seed={seed} rule={rule:?} order=BitReversed tracking={tracking}",
                std::any::type_name::<S>()
            ),
        }
    }

    fn gate(&mut self, op: Op) {
        assert_eq!(
            apply(&mut self.candidate, op),
            apply(&mut self.reference, op)
        );
    }

    fn batch(&mut self, qubits: &[QubitId], prepare: bool) {
        let distinct = qubits
            .iter()
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == qubits.len();
        let order = if distinct {
            self.candidate.measurement_batch_order()
        } else {
            MeasurementBatchOrder::AsGiven
        };
        let positions = order.permutation(qubits.len());
        let mut expected = vec![(false, false); qubits.len()];
        for i in positions {
            let r = self.reference.mz(&qubits[i..=i]).remove(0);
            expected[i] = (r.outcome, r.is_deterministic);
        }
        let actual = if prepare {
            // Observe PZ's measurement outcomes on a clone; exercise the actual
            // default PZ path on the candidate, including caller-order corrections.
            let results = self.candidate.clone().mz(qubits);
            self.candidate.pz(qubits);
            let corrections: Vec<_> = qubits
                .iter()
                .zip(&expected)
                .filter_map(|(&q, &(outcome, _))| outcome.then_some(q))
                .collect();
            self.reference.x(&corrections);
            results
        } else {
            self.candidate.mz(qubits)
        };
        let actual: Vec<_> = actual
            .into_iter()
            .map(|r| (r.outcome, r.is_deterministic))
            .collect();
        assert_eq!(
            actual, expected,
            "(a) batch outcomes: {}; prepare={prepare}, qubits={qubits:?}",
            self.context
        );
        assert!(
            valid_structure(&self.candidate),
            "(c) candidate: {}",
            self.context
        );
        assert!(
            valid_structure(&self.reference),
            "(c) reference: {}",
            self.context
        );
        assert!(
            same_group(&rows(self.candidate.stabs()), &rows(self.reference.stabs())),
            "(d) batch signed group: {}",
            self.context
        );
        assert_hermitian(&self.candidate, &self.context);
        if self.candidate.measurement_rule() == MeasurementRule::AaronsonGottesman {
            // Same rule, same single-qubit sequence: the tableaux, including any
            // tracked destabilizer signs, must agree exactly.
            assert!(
                same_tableau(&self.candidate, &self.reference),
                "(b) batch tableau: {}",
                self.context
            );
        }
    }
}

fn batch_workloads<S: IndexSet>() {
    for rule in [
        MeasurementRule::AaronsonGottesman,
        MeasurementRule::RefactorDecouple,
    ] {
        for tracking in [false, true] {
            for seed in 0..12 {
                let mut rng = PecosRng::seed_from_u64(seed);
                let n = rng.random_range(3..=10);
                let mut h = BatchHarness::<S>::new(n, seed, rule, tracking);
                for _ in 0..40 {
                    for _ in 0..8 {
                        let q = rng.random_range(0..n);
                        let other = (q + rng.random_range(1..n)) % n;
                        h.gate(match rng.random_range(0..5) {
                            0 => Op::H(q),
                            1 => Op::Sz(q),
                            2 => Op::Cx(q, other),
                            3 => Op::Cz(q, other),
                            _ => Op::X(q),
                        });
                    }
                    let mut qubits: Vec<_> = (0..n).map(QubitId).collect();
                    qubits.shuffle(&mut rng);
                    qubits.truncate(rng.random_range(0..=n));
                    if !qubits.is_empty() && rng.random_bool(0.2) {
                        qubits.push(qubits[0]);
                    }
                    h.batch(&qubits, rng.random());
                }
                let checks = surface_checks();
                let ancillas: Vec<_> = checks.iter().map(|c| QubitId(c.ancilla)).collect();
                let mut h = BatchHarness::<S>::new(49, seed, rule, tracking);
                for round in 0..4 {
                    h.gate(Op::X(round));
                    h.gate(Op::Z(round + 1));
                    for check in checks.iter().filter(|c| c.x_type) {
                        h.gate(Op::H(check.ancilla));
                    }
                    for layer in 0..4 {
                        for check in &checks {
                            if let Some(&q) = check.corners.get(layer) {
                                h.gate(if check.x_type {
                                    Op::Cx(check.ancilla, q)
                                } else {
                                    Op::Cx(q, check.ancilla)
                                });
                            }
                        }
                    }
                    for check in checks.iter().filter(|c| c.x_type) {
                        h.gate(Op::H(check.ancilla));
                    }
                    h.batch(&ancillas, false);
                    h.batch(&ancillas, true);
                    // PZ also sees fresh nondeterministic batches.
                    h.gate(Op::H(round));
                    h.batch(
                        &[QubitId(round), QubitId(round + 1), QubitId(round + 2)],
                        true,
                    );
                }
            }
        }
    }
}

#[test]
fn batch_measurement_oracle() {
    batch_workloads::<AdaptiveIndexSet>();
    batch_workloads::<BitSet>();
    batch_workloads::<SortedVecSet>();
    batch_workloads::<VecSet<usize>>();
}

#[test]
fn batch_permutations_and_setting_lifecycle() {
    for (len, expected) in [
        (0, vec![]),
        (1, vec![0]),
        (2, vec![0, 1]),
        (5, vec![0, 4, 2, 1, 3]),
        (8, vec![0, 4, 2, 6, 1, 5, 3, 7]),
    ] {
        assert_eq!(
            MeasurementBatchOrder::BitReversed.permutation(len),
            expected
        );
    }
    for len in 0..100 {
        assert_eq!(
            MeasurementBatchOrder::AsGiven.permutation(len),
            (0..len).collect::<Vec<_>>()
        );
        let mut positions = MeasurementBatchOrder::BitReversed.permutation(len);
        positions.sort_unstable();
        assert_eq!(positions, (0..len).collect::<Vec<_>>());
    }
    let qubits: Vec<_> = (0..9).map(QubitId).collect();
    let mut sim = SparseStabVecSet::with_seed(9, 7);
    assert_eq!(
        sim.measurement_batch_order(),
        MeasurementBatchOrder::AsGiven
    );
    sim = sim.with_measurement_batch_order(MeasurementBatchOrder::BitReversed);
    let expected = MeasurementBatchOrder::BitReversed.permutation(qubits.len());
    assert_eq!(sim.measurement_batch_permutation(&qubits), expected);
    let mut cloned = sim.clone();
    cloned.reset();
    assert_eq!(
        cloned.measurement_batch_order(),
        MeasurementBatchOrder::BitReversed
    );
    assert_eq!(cloned.measurement_batch_permutation(&qubits), expected);
    assert_eq!(
        sim.measurement_batch_permutation(&[QubitId(8), QubitId(0), QubitId(8)]),
        [0, 1, 2]
    );
    // A batch with a duplicate does not change the order of later batches.
    assert_eq!(sim.measurement_batch_permutation(&qubits), expected);
    assert_eq!(sim.measurement_batch_permutation(&[]), Vec::<usize>::new());
    assert_eq!(sim.measurement_batch_permutation(&[QubitId(8)]), [0]);
}

#[test]
fn batch_bell_correlations_and_marginals() {
    for rule in [
        MeasurementRule::AaronsonGottesman,
        MeasurementRule::RefactorDecouple,
    ] {
        for order in [
            MeasurementBatchOrder::AsGiven,
            MeasurementBatchOrder::BitReversed,
        ] {
            let mut ones = 0;
            for seed in 0..256 {
                let mut sim = SparseStabVecSet::with_seed(3, seed)
                    .with_measurement_rule(rule)
                    .with_measurement_batch_order(order);
                sim.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
                // Three positions give order [0, 2, 1], reversing which Bell
                // qubit is measured first while the spectator stays deterministic.
                let batch = [QubitId(2), QubitId(1), QubitId(0)];
                let results = sim.mz(&batch);
                assert!(!results[0].outcome && results[0].is_deterministic);
                assert_eq!(results[1].outcome, results[2].outcome);
                assert_ne!(results[1].is_deterministic, results[2].is_deterministic);
                ones += usize::from(results[1].outcome);
                let mut given = SparseStabVecSet::with_seed(3, seed).with_measurement_rule(rule);
                given.h(&[QubitId(0)]).cx(&[(QubitId(0), QubitId(1))]);
                assert_eq!(results[1].outcome, given.mz(&batch)[1].outcome);
            }
            assert!(
                (80..176).contains(&ones),
                "both Bell marginals should be balanced: {ones}"
            );
        }
    }
}
