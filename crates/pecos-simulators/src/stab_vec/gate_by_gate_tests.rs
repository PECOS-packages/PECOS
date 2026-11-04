//! G: gate-by-gate contracts. C: compatibility with the unchanged single-term path.
use super::*;
use pecos_core::RngManageable;
use pecos_num::stats::jeffreys_interval;

fn bits(value: usize, n: usize) -> BitSet {
    let mut result = BitSet::with_capacity(n);
    for q in 0..n {
        if value & (1 << q) != 0 {
            result.insert(q);
        }
    }
    result
}

fn assert_support(sim: &StabVec) {
    assert_eq!(sim.x.is_some(), sim.num_terms() > 1 && !sim.is_dense());
    if let Some(x) = &sim.x {
        let amplitude = sim.sampled_amplitude().unwrap();
        assert!(
            amplitude.norm_sqr() > 0.0,
            "sampled x support invariant: {x:?}"
        );
        assert!(amplitude.re.is_finite() && amplitude.im.is_finite());
    }
}

#[derive(Clone, Copy, Debug)]
enum Op {
    H(usize),
    S(usize),
    X(usize),
    Y(usize),
    Cx(usize, usize),
    Cy(usize, usize),
    Sxx(usize, usize),
    Syy(usize, usize),
    Rz(usize, f64),
    T(usize),
    Mz(usize),
    Mpz(usize),
    Mx(usize),
    Reset,
}

fn apply<S: CliffordGateable + ArbitraryRotationGateable>(sim: &mut S, op: Op) {
    match op {
        Op::H(q) => {
            sim.h(&[QubitId(q)]);
        }
        Op::S(q) => {
            sim.sz(&[QubitId(q)]);
        }
        Op::X(q) => {
            sim.x(&[QubitId(q)]);
        }
        Op::Y(q) => {
            sim.y(&[QubitId(q)]);
        }
        Op::Cx(c, t) => {
            sim.cx(&[(QubitId(c), QubitId(t))]);
        }
        Op::Cy(c, t) => {
            sim.cy(&[(QubitId(c), QubitId(t))]);
        }
        Op::Sxx(c, t) => {
            sim.sxx(&[(QubitId(c), QubitId(t))]);
        }
        Op::Syy(c, t) => {
            sim.syy(&[(QubitId(c), QubitId(t))]);
        }
        Op::Rz(q, a) => {
            sim.rz(Angle64::try_from_radians(a).unwrap(), &[QubitId(q)]);
        }
        Op::T(q) => {
            sim.t(&[QubitId(q)]);
        }
        Op::Reset => {
            sim.reset();
        }
        _ => panic!("measurement must use the outcome-history oracle"),
    }
}

fn measured_qubit(op: Op) -> Option<usize> {
    match op {
        Op::Mz(q) | Op::Mpz(q) | Op::Mx(q) => Some(q),
        _ => None,
    }
}

fn measure(sim: &mut StabVec, op: Op) -> MeasurementResult {
    match op {
        Op::Mz(q) => sim.mz(&[QubitId(q)]).remove(0),
        Op::Mpz(q) => sim.mpz(&[QubitId(q)]).remove(0),
        Op::Mx(q) => sim.mx(&[QubitId(q)]).remove(0),
        _ => panic!("measurement helper requires a measurement"),
    }
}

fn dense_branch(mut dense: StateVecSoA, op: Op, outcome: bool) -> (f64, StateVecSoA) {
    let q = measured_qubit(op).unwrap();
    if matches!(op, Op::Mx(_)) {
        dense.h(&[QubitId(q)]);
    }
    let mut state = dense.state();
    for (basis, amplitude) in state.iter_mut().enumerate() {
        if (basis & (1 << q) != 0) != outcome {
            *amplitude = Complex64::new(0.0, 0.0);
        }
    }
    let probability = state.iter().map(Complex64::norm_sqr).sum::<f64>();
    if probability > 0.0 {
        for amplitude in &mut state {
            *amplitude /= probability.sqrt();
        }
        dense = StateVecSoA::from_complex_state(&state, PecosRng::seed_from_u64(0));
        if matches!(op, Op::Mx(_)) {
            dense.h(&[QubitId(q)]);
        }
        if matches!(op, Op::Mpz(_)) && outcome {
            dense.x(&[QubitId(q)]);
        }
    }
    (probability, dense)
}

fn lockstep(ops: &[Op], n: usize, seed: u64, allow_dense: bool) -> bool {
    let mut sim = StabVec::builder(n)
        .seed(seed)
        .pruning_threshold(0.0)
        .unwrap()
        .build();
    if !allow_dense {
        sim.max_dense_bytes = 0;
    }
    let mut crossed = false;
    let mut dense = StateVecSoA::with_seed(n, seed);
    for (step, &op) in ops.iter().enumerate() {
        if measured_qubit(op).is_some() {
            let result = measure(&mut sim, op);
            let (probability, projected) = dense_branch(dense, op, result.outcome);
            assert!(
                probability > 1e-24,
                "seed {seed}, step {step}, {op:?}: impossible outcome"
            );
            if result.is_deterministic {
                assert!(probability >= 1.0 - 1e-12, "{op:?}: {probability}");
            }
            dense = projected;
        } else {
            apply(&mut sim, op);
            apply(&mut dense, op);
        }
        crossed |= sim.is_dense();
        assert_support(&sim);
        let actual = sim.state_vector();
        let expected = dense.state();
        for (a, b) in actual.iter().zip(expected) {
            assert!(
                (a - b).norm() < 1e-10,
                "seed {seed}, step {step}, {op:?}: {a} != {b}"
            );
        }
    }
    crossed
}

// G: new behavior.
#[test]
fn g_random_lockstep_and_determinism() {
    let mut crossed = false;
    for allow_dense in [false, true] {
        for seed in 0..32 {
            let n = 2 + usize::try_from(seed % if allow_dense { 3 } else { 7 }).unwrap();
            let mut rng = PecosRng::seed_from_u64(seed + 987);
            let mut ops = vec![Op::H(0), Op::Rz(0, 0.37), Op::H(0)];
            let mut rotations = 1;
            for _ in 0..70 {
                let q = rng.random_range(0..n);
                let t = (q + rng.random_range(1..n)) % n;
                let op = match rng.random_range(0..13) {
                    0 => Op::H(q),
                    1 => Op::S(q),
                    2 => Op::X(q),
                    3 => Op::Y(q),
                    4 => Op::Cx(q, t),
                    5 => Op::Cy(q, t),
                    6 => Op::Sxx(q, t),
                    7 => Op::Syy(q, t),
                    8 if rotations < 7 => {
                        rotations += 1;
                        Op::Rz(q, 0.61)
                    }
                    9 if rotations < 7 => {
                        rotations += 1;
                        Op::T(q)
                    }
                    10 => Op::Mpz(q),
                    11 => Op::Mx(q),
                    _ => Op::Mz(q),
                };
                ops.push(op);
            }
            crossed |= lockstep(&ops, n, seed, allow_dense);
        }
    }
    assert!(
        crossed,
        "random lockstep must exercise the growth crossover"
    );
    let mut sim = StabVec::new_with_seed(2, 7);
    sim.h(&[QubitId(0)])
        .rz(pecos_core::radians!(0.37), &[QubitId(0)])
        .h(&[QubitId(0)]);
    assert_eq!(sim.terms[0].1.expectation_value_zq(0), 1.0);
    assert_eq!(sim.terms[1].1.expectation_value_zq(0), -1.0);
    assert!(!sim.mz(&[QubitId(0)])[0].is_deterministic);
}

fn exact_histories(ops: &[Op], n: usize) -> Vec<f64> {
    let mut branches = vec![(0usize, 1.0, StateVecSoA::with_seed(n, 0))];
    let mut measurements = 0;
    for &op in ops {
        if measured_qubit(op).is_some() {
            let mut next = Vec::new();
            for (history, weight, state) in branches {
                for outcome in [false, true] {
                    let (p, projected) = dense_branch(state.clone(), op, outcome);
                    if p > 1e-24 {
                        next.push((
                            history | (usize::from(outcome) << measurements),
                            weight * p,
                            projected,
                        ));
                    }
                }
            }
            branches = next;
            measurements += 1;
        } else {
            for (_, _, state) in &mut branches {
                apply(state, op);
            }
        }
    }
    let mut probabilities = vec![0.0; 1 << measurements];
    for (history, p, _) in branches {
        probabilities[history] += p;
    }
    probabilities
}

// G: new behavior.
#[test]
fn g_joint_distributions() {
    // Bonferroni across 4 * 8 + 4 possible histories: family-wise alpha = 1e-5.
    let fixtures = [
        vec![
            Op::H(0),
            Op::Rz(0, 0.37),
            Op::H(0),
            Op::H(0),
            Op::Rz(1, 0.41),
            Op::H(1),
            Op::Rz(0, -0.37),
            Op::H(0),
            Op::Mz(0),
            Op::Mz(1),
            Op::Mz(2),
        ],
        vec![
            Op::H(0),
            Op::T(0),
            Op::Cx(0, 1),
            Op::Mz(0),
            Op::Rz(1, 0.61),
            Op::H(1),
            Op::Mz(1),
            Op::Mx(2),
        ],
        vec![
            Op::H(0),
            Op::T(0),
            Op::H(0),
            Op::Sxx(0, 1),
            Op::Mz(1),
            Op::Rz(0, 0.7),
            Op::H(0),
            Op::Mz(0),
            Op::Mx(2),
        ],
        vec![
            Op::H(0),
            Op::T(0),
            Op::Cx(0, 1),
            Op::Mpz(0),
            Op::H(0),
            Op::Rz(0, 0.53),
            Op::H(0),
            Op::Mz(0),
            Op::Reset,
            Op::H(1),
            Op::T(1),
            Op::H(1),
            Op::Mz(1),
        ],
        vec![
            Op::H(0),
            Op::Rz(0, 0.37),
            Op::H(0),
            Op::H(1),
            Op::Rz(1, 0.41),
            Op::H(1),
            Op::X(0),
            Op::Rz(0, 0.5),
            Op::Y(0),
            Op::Mz(0),
            Op::Mz(1),
        ],
    ];
    for (fixture, ops) in fixtures.iter().enumerate() {
        let n = if fixture == 4 { 2 } else { 3 };
        let expected = exact_histories(ops, n);
        let mut counts = vec![0u64; expected.len()];
        let shots = 6000u64;
        for seed in 0..shots {
            let mut sim = StabVec::builder(n)
                .seed(seed)
                .pruning_threshold(0.0)
                .unwrap()
                .build();
            if fixture != 4 {
                sim.max_dense_bytes = 0;
            }
            let mut history = 0;
            let mut position = 0;
            for &op in ops {
                if measured_qubit(op).is_some() {
                    if fixture == 0 && position == 0 {
                        assert!(sim.num_terms() > 1);
                    }
                    if fixture == 4 && position == 0 {
                        assert!(!sim.is_dense());
                        assert_eq!(sim.num_terms(), 4);
                        assert!(!sim.cliff_frame[0].is_diagonal());
                    }
                    let result = measure(&mut sim, op);
                    if fixture == 4 && position == 0 {
                        assert!(sim.is_dense(), "measurement frame flush must cross over");
                    }
                    if fixture == 0 && position == 0 {
                        assert!(!result.outcome, "interference fixture must measure zero");
                        assert!(
                            !result.is_deterministic,
                            "interference alone does not set the flag"
                        );
                    }
                    history |= usize::from(result.outcome) << position;
                    position += 1;
                } else {
                    apply(&mut sim, op);
                }
                assert_support(&sim);
            }
            counts[history] += 1;
        }
        for (history, (&count, &p)) in counts.iter().zip(&expected).enumerate() {
            let interval = jeffreys_interval(count, shots, 1e-5 / 36.0).unwrap();
            assert!(
                p >= interval.lo - 1e-14 && p <= interval.hi + 1e-14,
                "fixture {fixture}, history {history}: exact {p}, count {count}, interval {interval:?}"
            );
        }
    }
}

// G: new behavior.
#[test]
fn g_bitset_amplitude_matches_word_and_wide_support() {
    let mut rng = PecosRng::seed_from_u64(9);
    for n in [1, 3, 8, 31, 63] {
        let mut ch = CHFormGeneric::<BitSet>::with_seed(n, 7);
        for _ in 0..100 {
            let q = rng.random_range(0..n);
            ch.h(&[QubitId(q)]).sz(&[QubitId(q)]);
            if n > 1 {
                ch.cx(&[(QubitId(q), QubitId((q + 1) % n))]);
            }
            let value = rng.random_range(0..(1usize << n));
            assert_eq!(ch.amplitude_bits(&bits(value, n)), ch.amplitude(value));
            let sampled = ch.sample_basis(&mut rng);
            let value = sampled.iter().fold(0, |value, q| value | (1 << q));
            assert_eq!(ch.amplitude_bits(&sampled), ch.amplitude(value));
            assert!(ch.amplitude(value).norm_sqr() > 0.0);
        }
    }
    let mut ch = CHFormGeneric::<BitSet>::with_seed(100, 9);
    ch.h(&[QubitId(97), QubitId(98)])
        .cx(&[(QubitId(97), QubitId(99)), (QubitId(98), QubitId(97))]);
    // These identity rows/columns prove that any of the 97 low bits being
    // nonzero violates the CH support constraint, for every high-bit pattern.
    let f = ch.arc_f();
    for q in 0..97 {
        let mut singleton = BitSet::with_capacity(100);
        singleton.insert(q);
        assert_eq!(f.row(q), &singleton);
        assert_eq!(f.col(q), &singleton);
        assert!(!ch.arc_v().contains(q));
        assert!(!ch.arc_s().contains(q));
    }
    for tail in 0..8 {
        let mut x = BitSet::with_capacity(100);
        for q in 0..3 {
            if tail & (1 << q) != 0 {
                x.insert(97 + q);
            }
        }
        // x97 = a xor b, x98 = b, x99 = a.
        let supported = x.contains(97) == (x.contains(98) ^ x.contains(99));
        let expected = if supported { 0.5 } else { 0.0 };
        assert!((ch.amplitude_bits(&x) - Complex64::new(expected, 0.0)).norm() < 1e-15);
        for q in 0..97 {
            x.insert(q);
            assert_eq!(ch.amplitude_bits(&x), Complex64::new(0.0, 0.0));
            x.remove(q);
        }
    }
    for _ in 0..100 {
        let mut x = BitSet::with_capacity(100);
        for q in 0..100 {
            if rng.random::<bool>() {
                x.insert(q);
            }
        }
        assert_eq!(ch.amplitude_bits(&x), Complex64::new(0.0, 0.0));
    }
}

// G: new behavior.
#[test]
fn g_entry_sampler_support_and_uniformity() {
    let mut rng = PecosRng::seed_from_u64(178);
    for circuit in 0..12 {
        let n = 3 + circuit % 4;
        let mut ch = CHFormGeneric::<BitSet>::with_seed(n, 7);
        ch.h(&[QubitId(0)])
            .cx(&[(QubitId(0), QubitId(1)), (QubitId(1), QubitId(2))]);
        if circuit == 0 {
            let f = ch.arc_f();
            let mut square_row = BitSet::with_capacity(n);
            for j in f.row(0) {
                square_row.xor_assign(f.row(j));
            }
            assert_ne!(
                square_row,
                bits(1, n),
                "F squared must differ from identity"
            );
        } else {
            for _ in 0..20 {
                let q = rng.random_range(0..n);
                match rng.random_range(0..4) {
                    0 => {
                        ch.h(&[QubitId(q)]);
                    }
                    1 => {
                        ch.sz(&[QubitId(q)]);
                    }
                    2 => {
                        ch.x(&[QubitId(q)]);
                    }
                    _ => {
                        ch.cx(&[(QubitId(q), QubitId((q + 1) % n))]);
                    }
                }
            }
        }
        let mut counts = vec![0u64; 1 << n];
        for _ in 0..6000 {
            let x = ch.sample_basis(&mut rng);
            assert!(ch.amplitude_bits(&x).norm_sqr() > 0.0);
            let value = x.iter().fold(0, |value, q| value | (1 << q));
            counts[value] += 1;
        }
        let support_size = 1u32 << ch.arc_v().len();
        assert_eq!(
            (0..1 << n)
                .filter(|&x| ch.amplitude(x).norm_sqr() > 0.0)
                .count(),
            usize::try_from(support_size).unwrap()
        );
        for (x, count) in counts.into_iter().enumerate() {
            let p = if ch.amplitude(x).norm_sqr() > 0.0 {
                1.0 / f64::from(support_size)
            } else {
                0.0
            };
            let interval = jeffreys_interval(count, 6000, 1e-5 / (12.0 * 64.0)).unwrap();
            assert!(
                p >= interval.lo && p <= interval.hi,
                "circuit {circuit}, x {x}, p {p}, {interval:?}"
            );
        }
    }
}

// G: new behavior.
#[test]
fn g_joint_h_fixture() {
    let mut reached = false;
    for seed in 0..64 {
        let mut sim = StabVec::new_with_seed(3, seed);
        sim.rz(pecos_core::radians!(0.37), &[QubitId(2)])
            .h(&[QubitId(2)]);
        sim.x(&[QubitId(0), QubitId(1)])
            .h(&[QubitId(0), QubitId(1)]);
        // Match SXX's frame preparation, then inspect the stored input to its Hs.
        sim.propagate_cliff_frames(0, 1, PauliFrameGate::Sxx);
        for (_, ch) in &sim.terms {
            for c in 0..2 {
                let base = ch.amplitude_bits(&bits(c << 2, 3));
                for ab in 0usize..4 {
                    let sign = if ab.count_ones() & 1 == 0 { 1.0 } else { -1.0 };
                    assert!(
                        (ch.amplitude_bits(&bits((c << 2) | ab, 3)) - base * sign).norm() < 1e-15
                    );
                }
            }
        }
        if !sim.x.as_ref().unwrap().contains(1) {
            reached = true;
        }
        sim.apply_xx_root(&[(QubitId(0), QubitId(1))], false);
        assert_support(&sim);
    }
    assert!(reached, "fixture must reach x_b=0 before the first term H");
}

// G: new behavior.
#[test]
fn g_pruning_guard_preserves_rare_branch() {
    let mut sim = StabVec::builder(2)
        .seed(9)
        .pruning_threshold(1e-2)
        .unwrap()
        .build();
    sim.h(&[QubitId(0)])
        .rz(pecos_core::radians!(0.1), &[QubitId(0)])
        .h(&[QubitId(0)]);
    let mut common = sim.clone();
    let mut common_x = common.x.clone().unwrap();
    common_x.remove(0);
    common.set_sampled_basis_for_test(common_x);
    common
        .rz(pecos_core::radians!(0.37), &[QubitId(1)])
        .h(&[QubitId(1)]);
    assert_eq!(
        common.num_terms(),
        2,
        "pruning to one term must re-enter at the split"
    );
    assert_support(&common);
    // Test-only assignment selects the rare supported branch without a seed search.
    let mut rare = sim.x.clone().unwrap();
    rare.insert(0);
    sim.set_sampled_basis_for_test(rare);
    assert_support(&sim);
    sim.rz(pecos_core::radians!(0.37), &[QubitId(1)])
        .h(&[QubitId(1)]);
    assert_eq!(
        sim.num_terms(),
        4,
        "guard must skip pruning of the rare branch"
    );
    assert_support(&sim);
    assert!(
        sim.state_vector()
            .iter()
            .all(|a| a.re.is_finite() && a.im.is_finite())
    );
}

// G: new behavior.
#[test]
fn g_projection_rescale_survives_2300_measurements() {
    let mut sim = StabVec::new_with_seed(3, 7);
    sim.h(&[QubitId(1)])
        .rz(pecos_core::radians!(0.37), &[QubitId(1)])
        .h(&[QubitId(1)]);
    for _ in 0..2300 {
        sim.h(&[QubitId(0)]);
        assert!(!sim.mz(&[QubitId(0)])[0].is_deterministic);
        assert!(sim.num_terms() > 1);
        let max = sim.max_coefficient_magnitude();
        assert!((0.5..=1.0).contains(&max), "coefficient scale {max}");
        assert_support(&sim);
    }
}

// G: new behavior.
#[test]
fn g_observation_and_reseed_preserve_sample() {
    let mut a = StabVec::new_with_seed(4, 9);
    a.h(&[QubitId(0)])
        .rz(pecos_core::radians!(0.37), &[QubitId(0)])
        .h(&[QubitId(0)]);
    a.h(&[QubitId(1)])
        .sz(&[QubitId(2)])
        .rz(pecos_core::radians!(0.61), &[QubitId(3)]);
    let mut b = a.clone();
    let original_x = a.x.clone();
    let _ = a.state_vector();
    assert_eq!(a.x, original_x);
    assert_eq!(a.cliff_frame, b.cliff_frame);
    assert_eq!(a.pending_rz, b.pending_rz);
    for q in [1, 3, 0, 2, 1, 0] {
        a.rz(pecos_core::radians!(0.23), &[QubitId(q)])
            .h(&[QubitId(q)]);
        b.rz(pecos_core::radians!(0.23), &[QubitId(q)])
            .h(&[QubitId(q)]);
        let ra = a.mz(&[QubitId(q)]).remove(0);
        let rb = b.mz(&[QubitId(q)]).remove(0);
        assert_eq!(
            (ra.outcome, ra.is_deterministic),
            (rb.outcome, rb.is_deterministic)
        );
    }
    assert_eq!(a.rng.random::<u64>(), b.rng.random::<u64>());
    a.set_rng(PecosRng::seed_from_u64(200));
    let x = a.x.clone();
    a.set_rng(PecosRng::seed_from_u64(300));
    assert_eq!(a.x, x);
}

// G: observing a pending decomposition preserves both state and future draws.
#[test]
fn g_single_term_pending_rz_observation_preserves_rng_and_outcomes() {
    for seed in 0..32 {
        let mut input = StabVec::new_with_seed(1, seed);
        input
            .h(&[QubitId(0)])
            .rz(pecos_core::radians!(0.37), &[QubitId(0)]);
        assert_eq!(input.num_terms(), 1);
        let mut observed = input.clone();
        let mut untouched = input.clone();
        let state = observed.state_vector();
        let mut dense = StateVecSoA::with_seed(1, seed);
        dense
            .h(&[QubitId(0)])
            .rz(pecos_core::radians!(0.37), &[QubitId(0)]);
        for (actual, expected) in state.iter().zip(dense.state()) {
            assert!((actual - expected).norm() < 1e-14);
        }
        assert_eq!(observed.num_terms(), 1);
        assert_eq!(observed.x, input.x);
        assert_eq!(observed.pending_rz, input.pending_rz);
        assert_eq!(observed.cliff_frame, input.cliff_frame);
        let mut before_rng = input.rng().clone();
        let mut observed_rng = observed.rng().clone();
        let mut untouched_rng = untouched.rng().clone();
        for _ in 0..8 {
            let expected = before_rng.random::<u64>();
            assert_eq!(observed_rng.random::<u64>(), expected);
            assert_eq!(untouched_rng.random::<u64>(), expected);
        }
        for _ in 0..16 {
            observed
                .h(&[QubitId(0)])
                .rz(pecos_core::radians!(0.23), &[QubitId(0)])
                .h(&[QubitId(0)]);
            untouched
                .h(&[QubitId(0)])
                .rz(pecos_core::radians!(0.23), &[QubitId(0)])
                .h(&[QubitId(0)]);
            let actual = observed.mz(&[QubitId(0)]).remove(0);
            let expected = untouched.mz(&[QubitId(0)]).remove(0);
            assert_eq!(
                (actual.outcome, actual.is_deterministic),
                (expected.outcome, expected.is_deterministic)
            );
        }
        assert_eq!(
            observed.rng().clone().random::<u64>(),
            untouched.rng().clone().random::<u64>()
        );
    }
}

// C: compatibility.
#[test]
fn c_single_term_observation_draws_nothing() {
    let mut a = StabVec::new_with_seed(3, 42);
    a.h(&[QubitId(0)])
        .sz(&[QubitId(1)])
        .cx(&[(QubitId(0), QubitId(2))]);
    let mut rng = a.rng.clone();
    let _ = a.state_vector();
    assert!(a.x.is_none());
    assert_eq!(a.rng.random::<u64>(), rng.random::<u64>());
}

// G: new behavior.
#[test]
fn g_deterministic_resample_and_measurement_draw_nothing() {
    let mut sim = StabVec::new_with_seed(3, 42);
    sim.h(&[QubitId(0)])
        .rz(pecos_core::radians!(0.37), &[QubitId(0)])
        .h(&[QubitId(0)]);
    sim.h_term(1);
    let mut rng = sim.rng.clone();
    sim.h_term(1); // |+> -> |0>: exactly one conditional amplitude is nonzero.
    assert_support(&sim);
    assert!(!sim.x.as_ref().unwrap().contains(1));
    sim.mz(&[QubitId(0)]);
    assert_eq!(sim.rng.random::<u64>(), rng.random::<u64>());
}

// G: new behavior.
#[test]
fn g_dense_transfer_clears_deferred_state_without_draws() {
    let mut sim = StabVec::builder(2)
        .seed(7)
        .pruning_threshold(0.0)
        .unwrap()
        .build();
    sim.max_dense_bytes = 0;
    for _ in 0..3 {
        sim.rz(pecos_core::radians!(0.37), &[QubitId(0)])
            .h(&[QubitId(0)]);
    }
    sim.h(&[QubitId(1)])
        .rz(pecos_core::radians!(0.61), &[QubitId(1)]);
    sim.h(&[QubitId(0)])
        .apply_global_phase(pecos_core::radians!(0.31), &[QubitId(0)]);
    let expected = sim.state_vector();
    let mut rng = sim.rng.clone();
    sim.max_dense_bytes = DEFAULT_MAX_DENSE_BYTES;
    assert!(sim.should_switch_to_dense(sim.terms.len()));
    sim.switch_to_dense(sim.terms.len());
    assert!(sim.x.is_none());
    assert!(sim.cliff_frame.iter().all(|f| f.is_identity()));
    assert!(sim.pending_rz.iter().all(|&a| a == Angle64::ZERO));
    assert_eq!(sim.frame_phase, 0);
    assert_eq!(sim.global_phase, Angle64::ZERO);
    assert_eq!(sim.rng().clone().random::<u64>(), rng.random::<u64>());
    for (a, b) in sim.state_vector().iter().zip(expected) {
        assert!((a - b).norm() < 1e-14);
    }
    sim.reset();
    assert!(sim.x.is_none());
    assert_eq!(sim.num_terms(), 1);
}

// G: growth crosses the same budgeted threshold before any measurement.
#[test]
fn g_rz_growth_switches_immediately_with_exact_distribution() {
    let ops = [
        Op::H(0),
        Op::H(1),
        Op::Rz(0, 0.37),
        Op::H(0),
        Op::Rz(1, 0.53),
        Op::H(1),
        Op::Cx(0, 1),
        Op::Rz(0, 0.71),
        Op::H(0),
        Op::Mz(0),
        Op::Mz(1),
    ];
    let expected = exact_histories(&ops, 2);
    let shots = 6000u64;
    let mut counts = vec![0u64; expected.len()];
    for seed in 0..shots {
        let mut sim = StabVec::builder(2)
            .seed(seed)
            .pruning_threshold(0.0)
            .unwrap()
            .build();
        let mut dense = StateVecSoA::with_seed(2, seed);
        let mut history = 0;
        let mut position = 0;
        for (step, &op) in ops.iter().enumerate() {
            if measured_qubit(op).is_some() {
                history |= usize::from(measure(&mut sim, op).outcome) << position;
                position += 1;
            } else {
                apply(&mut sim, op);
                apply(&mut dense, op);
                assert_eq!(sim.is_dense(), step >= 8, "step {step}");
                crate::state_vector_test_utils::assert_phase_exact_state_matches(
                    &sim.state_vector(),
                    &dense.state(),
                    1e-12,
                    "RZ growth crossover",
                );
            }
            assert_support(&sim);
        }
        counts[history] += 1;
    }
    // Bonferroni over four histories: family-wise alpha = 1e-5.
    for (&count, &probability) in counts.iter().zip(&expected) {
        let interval = jeffreys_interval(count, shots, 1e-5 / 4.0).unwrap();
        assert!(
            probability >= interval.lo - 1e-14 && probability <= interval.hi + 1e-14,
            "exact {probability}, count {count}, interval {interval:?}"
        );
    }
}

// G: a flush may switch inside a gate, including partway through a batch.
#[test]
fn g_growth_crossover_preserves_enclosing_gates_and_frames() {
    fn trigger<T: CliffordGateable + ArbitraryRotationGateable>(sim: &mut T, gate: &str) {
        let qubits = [QubitId(0), QubitId(1), QubitId(2)];
        let pairs = [(QubitId(0), QubitId(1)), (QubitId(1), QubitId(2))];
        match gate {
            "H" => {
                sim.h(&qubits);
            }
            "H3" => {
                sim.h3(&qubits);
            }
            "H4" => {
                sim.h4(&qubits);
            }
            "CX" => {
                sim.cx(&pairs);
            }
            "CY" => {
                sim.cy(&pairs);
            }
            "CZ" => {
                sim.cz(&pairs);
            }
            "SXX" => {
                sim.sxx(&pairs);
            }
            "SXXdg" => {
                sim.sxxdg(&pairs);
            }
            "SYY" => {
                sim.syy(&pairs);
            }
            "SYYdg" => {
                sim.syydg(&pairs);
            }
            "SZZ" => {
                sim.szz(&pairs);
            }
            "SZZdg" => {
                sim.szzdg(&pairs);
            }
            "RZ" => {
                sim.rz(pecos_core::radians!(0.17), &qubits);
            }
            "RX" => {
                sim.rx(pecos_core::radians!(0.17), &qubits);
            }
            "RY" => {
                sim.ry(pecos_core::radians!(0.17), &qubits);
            }
            "RXX" => {
                sim.rxx(pecos_core::radians!(0.17), &pairs);
            }
            "RYY" => {
                sim.ryy(pecos_core::radians!(0.17), &pairs);
            }
            "RZZ" => {
                sim.rzz(pecos_core::radians!(0.17), &pairs);
            }
            _ => panic!("crossover fixture requires a known trigger gate"),
        }
    }
    for frame in ["I", "X", "Y", "S", "XS", "YS"] {
        for gate in [
            "H", "H3", "H4", "CX", "CY", "CZ", "SXX", "SXXdg", "SYY", "SYYdg", "SZZ", "SZZdg",
            "RZ", "RX", "RY", "RXX", "RYY", "RZZ",
        ] {
            let mut sim = StabVec::builder(3)
                .seed(9)
                .pruning_threshold(0.0)
                .unwrap()
                .build();
            let mut dense = StateVecSoA::with_seed(3, 9);
            for op in [
                Op::H(0),
                Op::H(1),
                Op::H(2),
                Op::Rz(0, 0.37),
                Op::H(0),
                Op::Rz(0, 0.53),
                Op::H(0),
                Op::Rz(0, 0.71),
                Op::H(0),
                Op::Rz(1, 0.41),
                Op::Rz(2, 0.67),
            ] {
                apply(&mut sim, op);
                apply(&mut dense, op);
            }
            assert_eq!(sim.num_terms(), 8);
            if frame.contains('X') {
                sim.x(&[QubitId(1)]);
                dense.x(&[QubitId(1)]);
            }
            if frame.contains('Y') {
                sim.y(&[QubitId(1)]);
                dense.y(&[QubitId(1)]);
            }
            if frame.contains('S') {
                sim.sz(&[QubitId(1)]);
                dense.sz(&[QubitId(1)]);
            }
            sim.apply_global_phase(pecos_core::radians!(0.31), &[QubitId(0)]);
            dense.apply_global_phase(pecos_core::radians!(0.31), &[QubitId(0)]);
            trigger(&mut sim, gate);
            trigger(&mut dense, gate);
            sim.flush_all_pending_rz();
            assert!(sim.is_dense(), "{frame}; {gate}");
            assert!(sim.cliff_frame.iter().all(|f| f.is_identity()));
            assert!(sim.pending_rz.iter().all(|&a| a == Angle64::ZERO));
            assert_eq!(sim.frame_phase, 0);
            assert_eq!(sim.global_phase, Angle64::ZERO);
            crate::state_vector_test_utils::assert_phase_exact_state_matches(
                &sim.state_vector(),
                &dense.state(),
                1e-12,
                &format!("{frame}; {gate}"),
            );
        }
    }
}

// G: invalid caller configuration is rejected before simulation begins.
#[test]
fn g_pruning_threshold_rejects_invalid_values() {
    for (threshold, value) in [
        (1.0, "1"),
        (2.0, "2"),
        (-1e-3, "-0.001"),
        (f64::NAN, "NaN"),
        (f64::INFINITY, "inf"),
        (f64::NEG_INFINITY, "-inf"),
    ] {
        let error = StabVec::builder(2)
            .pruning_threshold(threshold)
            .err()
            .unwrap();
        assert_eq!(
            error.to_string(),
            format!(
                "invalid pruning threshold {value}: expected a finite value with 0.0 <= threshold < 1.0"
            )
        );
    }
}

// G: a high valid threshold preserves the largest term at decomposition.
#[test]
fn g_high_pruning_threshold_preserves_largest_term_and_dense_reference() {
    for seed in 0..64 {
        let mut sim = StabVec::builder(2)
            .seed(seed)
            .pruning_threshold(0.999)
            .unwrap()
            .build();
        let mut dense = StateVecSoA::with_seed(2, seed);
        for op in [Op::H(0), Op::Rz(0, 0.37), Op::H(0)] {
            apply(&mut sim, op);
            apply(&mut dense, op);
        }
        assert_eq!(sim.num_terms(), 2);
        assert!(sim.max_coefficient_magnitude() > 0.98);
        assert_support(&sim);
        crate::state_vector_test_utils::assert_phase_exact_state_matches(
            &sim.state_vector(),
            &dense.state(),
            1e-12,
            "threshold 0.999 before measurement",
        );
        let outcome = sim.mz(&[QubitId(0)])[0].outcome;
        let (probability, mut projected) = dense_branch(dense, Op::Mz(0), outcome);
        assert!(probability > 0.0);
        crate::state_vector_test_utils::assert_phase_exact_state_matches(
            &sim.state_vector(),
            &projected.state(),
            1e-12,
            "threshold 0.999 after measurement",
        );
    }
}

// G: the pruning boundary is strict, including the support guard's survivor sum.
#[test]
fn g_pruning_retains_terms_equal_to_cutoff() {
    let mut sim = StabVec::builder(2).pruning_threshold(0.25).unwrap().build();
    let ch = sim.terms[0].1.clone();
    sim.terms = vec![
        (Complex64::new(0.5, 0.0), ch.clone()),
        (Complex64::new(0.25, 0.0), ch),
    ];
    sim.set_sampled_basis_for_test(bits(0, 2));
    sim.apply_rz_immediate(pecos_core::radians!(0.37), 1);
    assert_eq!(
        sim.num_terms(),
        4,
        "a term exactly at the cutoff must survive"
    );
    assert_support(&sim);
}
