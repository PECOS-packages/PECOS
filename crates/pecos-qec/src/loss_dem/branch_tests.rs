use super::branch_replay::ReplayOptions;
use super::branches::{correlated_probability, span_elements};
use super::*;
use pecos_core::{BitSet, Gate, gate_type::GateType};
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::noise::general::{LossSite, QubitLoss};
use pecos_quantum::{TickCircuit, TickMeasRef};
use std::collections::{BTreeMap, BTreeSet};

fn gate(c: &mut TickCircuit, kind: GateType, qs: &[usize]) {
    c.tick()
        .try_add_gate(Gate::simple(
            kind,
            qs.iter().copied().map(Into::into).collect::<Vec<_>>(),
        ))
        .unwrap();
}
fn measure(c: &mut TickCircuit, qs: &[usize]) {
    let records = c.tick().mz(qs);
    for record in records {
        c.detector(&[record]).unwrap();
    }
}
fn probe(c: &mut TickCircuit, qs: &[usize]) {
    gate(c, GateType::DetectLost, qs);
}
fn model(loss: QubitLoss) -> GeneralNoiseModel {
    GeneralNoiseModel::builder().with_loss(loss).build()
}
fn catalog(c: &TickCircuit, noise: &GeneralNoiseModel) -> LossCatalog {
    LossCatalog::build(c, noise, 0.0, 12, 1000).unwrap()
}
fn singleton(label: &str) -> BTreeMap<String, f64> {
    BTreeMap::from([(label.into(), 1.0)])
}
fn idle_model(p: f64) -> GeneralNoiseModel {
    model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(p)
            .build()
            .unwrap(),
    )
}
fn mask(bits: &BitSet) -> usize {
    bits.iter().fold(0, |m, i| m ^ (1 << i))
}
fn branch_cell(catalog: &LossCatalog, shot: &Shot) -> usize {
    let d = shot
        .detectors
        .iter()
        .enumerate()
        .fold(0, |m, (i, v)| m | (usize::from(*v) << i));
    let o = shot
        .observables
        .iter_set_bits()
        .fold(0, |m, i| m | (1 << (catalog.num_detectors + i)));
    let offset = catalog.num_detectors + catalog.num_observables;
    let f = shot
        .flags
        .iter()
        .enumerate()
        .fold(0, |m, (i, v)| m | (usize::from(*v) << (offset + i)));
    d | o | f
}

// Joint cells include ALL flags, detectors, and observables, including zero cells.
// Bonferroni family alpha=.001; N=100000; both engines seeded with 824109.
fn engine_joint(name: &str, c: &TickCircuit, noise: GeneralNoiseModel) -> LossCatalog {
    engine_joint_with_effective_sites(name, c, noise, 1)
}

fn engine_joint_with_effective_sites(
    name: &str,
    c: &TickCircuit,
    noise: GeneralNoiseModel,
    observable_sites: usize,
) -> LossCatalog {
    let description = noise.description();
    assert!(
        [
            description.p1,
            description.p2,
            description.p_meas_0,
            description.p_meas_1,
            description.p_prep
        ]
        .iter()
        .all(|p| *p == 0.0)
    );
    let loss = description.loss.as_ref().unwrap();
    assert!(!loss.default_recovery_added);
    for channel in &loss.channels {
        if let pecos_engines::noise::general::LossChannelKind::Recovery { false_positive, .. } =
            channel.kind
        {
            assert_eq!(false_positive, 0.0);
        }
    }
    let catalog = LossCatalog::build_with_options(
        c,
        &noise,
        0.0,
        12,
        1000,
        ReplayOptions {
            convention_coins: false,
            ..ReplayOptions::default()
        },
    )
    .unwrap();
    let width = catalog.num_detectors + catalog.num_observables;
    let cells = 1 << (width + catalog.outputs.probes.len());
    let mut baseline_runner = TickRunner::new(c, GeneralNoiseModel::default(), 824_109).unwrap();
    let baseline = branch_cell(&catalog, &baseline_runner.shot().unwrap());
    let mut expected = vec![0.0; cells];
    expected[baseline] = 1.0;
    let mut effective_sites = BTreeSet::new();
    for root in &catalog.roots {
        let mut effective = false;
        expected[baseline] -= root.prior;
        for branch in &root.branches {
            let flags = branch.flags.iter().fold(0, |m, p| m | (1 << (width + p.0)));
            let vectors = span_elements(&branch.normal_form.span).unwrap();
            let weight = branch.weight / f64::from(u32::try_from(vectors.len()).unwrap());
            for v in vectors {
                let cell = baseline ^ flags ^ mask(&branch.normal_form.constant) ^ mask(&v);
                expected[cell] += weight;
                effective |= cell != baseline;
            }
        }
        if effective {
            effective_sites.insert((
                root.key.kind,
                root.key.channel,
                root.key.location,
                root.key.phase,
            ));
        }
    }
    let mut runner = TickRunner::new(c, noise, 824_109).unwrap();
    let mut counts = vec![0; cells];
    let shots = 100_000;
    for _ in 0..shots {
        counts[branch_cell(&catalog, &runner.shot().unwrap())] += 1;
    }
    let alpha = 0.001 / f64::from(u32::try_from(cells).unwrap());
    for (cell, (&count, &p)) in counts.iter().zip(&expected).enumerate() {
        let interval = pecos_num::stats::jeffreys_interval(count, shots, alpha).unwrap();
        assert!(
            p >= interval.lo - 1e-14 && p <= interval.hi + 1e-14,
            "{name}, {cells} cells, cell {cell}: expected {p}, count {count}, Jeffreys {interval:?}"
        );
    }
    assert_eq!(
        effective_sites.len(),
        observable_sites,
        "{name}: observable nonzero source sites"
    );
    eprintln!("engine joint {name}: N={shots}, cells={cells}, seed=824109, family alpha=.001");
    catalog
}

#[test]
fn engine_joint_constant_and_own_event() {
    let mut c = TickCircuit::new();
    c.tick().x(&[0]);
    probe(&mut c, &[0]);
    let read = c.tick().mz(&[0]);
    c.detector(&read).unwrap();
    c.observable(&read).unwrap();
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .add_events_channel(LossSite::BeforeSingleQubit, 0.37, &singleton("L"))
            .build()
            .unwrap(),
    );
    engine_joint("X on lost path", &c, noise);

    let mut c = TickCircuit::new();
    c.tick().cx(&[(0, 1)]);
    probe(&mut c, &[0, 1]);
    measure(&mut c, &[0, 1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_lost_partner_loss_probability(1.0)
            .add_events_channel(LossSite::BeforeTwoQubit, 0.37, &singleton("LI"))
            .build()
            .unwrap(),
    );
    engine_joint("own event pair", &c, noise);
}

#[test]
fn engine_joint_lz_and_partner_models() {
    let mut c = TickCircuit::new();
    c.tick().h(&[1]);
    c.tick().cx(&[(0, 1)]);
    probe(&mut c, &[0, 1]);
    c.tick().h(&[1]);
    measure(&mut c, &[0, 1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .add_events_channel(LossSite::BeforeTwoQubit, 0.37, &singleton("LZ"))
            .build()
            .unwrap(),
    );
    engine_joint("joint LZ", &c, noise);
    for weights in [
        singleton("Z"),
        BTreeMap::from([("X".into(), 0.5), ("Z".into(), 0.5)]),
    ] {
        let mut c = TickCircuit::new();
        c.tick().h(&[1]);
        c.tick().idle(1, &[0]);
        c.tick().cx(&[(0, 1)]);
        probe(&mut c, &[0, 1]);
        c.tick().h(&[1]);
        measure(&mut c, &[0, 1]);
        let noise = model(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_idle_rate(0.37)
                .with_lost_partner_loss_probability(0.0)
                .with_lost_partner_pauli_model(&weights)
                .build()
                .unwrap(),
        );
        engine_joint(&format!("partner {weights:?}"), &c, noise);
    }
}

#[test]
fn engine_joint_recovery_and_order() {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    c.tick().cx(&[(0, 1)]);
    probe(&mut c, &[0, 1]);
    measure(&mut c, &[0, 1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.37)
            .add_recovery_channel(LossSite::BeforeTwoQubit, 0.3, 0.0)
            .build()
            .unwrap(),
    );
    engine_joint("recovery FN and executed recovered gate", &c, noise);

    let mut c = TickCircuit::new();
    c.tick().x(&[0]);
    probe(&mut c, &[0]);
    measure(&mut c, &[0]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .add_events_channel(LossSite::AfterSingleQubit, 0.37, &singleton("L"))
            .add_recovery_channel(LossSite::AfterSingleQubit, 0.3, 0.0)
            .build()
            .unwrap(),
    );
    engine_joint("listed after events then recovery", &c, noise);
}

#[test]
fn engine_joint_chained_spread_and_noiseless_pair() {
    for noiseless in [false, true] {
        let mut c = TickCircuit::new();
        c.tick().idle(1, &[0]);
        c.tick().cx(&[(0, 1)]);
        c.tick().cz(&[(1, 2)]);
        probe(&mut c, &[0, 1, 2]);
        measure(&mut c, &[0, 1, 2]);
        let mut builder = GeneralNoiseModel::builder().with_loss(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_idle_rate(0.37)
                .with_lost_partner_loss_probability(0.6)
                .build()
                .unwrap(),
        );
        if noiseless {
            builder = builder.with_noiseless_gate(GateType::CX);
        }
        engine_joint(
            if noiseless {
                "noiseless dropped pair"
            } else {
                "chained spread"
            },
            &c,
            builder.build(),
        );
    }
}

#[test]
fn engine_joint_prep_idle_and_after_pair_idle() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0]);
    probe(&mut c, &[0]);
    measure(&mut c, &[0]);
    engine_joint(
        "prep",
        &c,
        model(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_prep(0.37)
                .build()
                .unwrap(),
        ),
    );

    let mut c = TickCircuit::new();
    c.tick().idle(2, &[0]);
    probe(&mut c, &[0]);
    measure(&mut c, &[0]);
    engine_joint("idle duration clamp", &c, idle_model(0.8));

    // An own before-event loses leg 0 with certainty. Only the implicit idle on
    // leg 1 has a variable draw; leg 0's idle is blocked. The deterministic root
    // is invisible (no probe/record on leg 0), so one effective root remains.
    let mut c = TickCircuit::new();
    c.tick().cx(&[(0, 1)]);
    probe(&mut c, &[1]);
    measure(&mut c, &[1]);
    let noise = GeneralNoiseModel::builder()
        .with_idle_after_2q(1.0)
        .with_loss(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_idle_rate(0.37)
                .add_events_channel(LossSite::BeforeTwoQubit, 1.0, &singleton("LI"))
                .build()
                .unwrap(),
        )
        .build();
    engine_joint("after pair idle on dropped partner", &c, noise);
}

#[test]
fn engine_joint_reload_mpz_and_dropped_pz() {
    for reload in [false, true] {
        let mut c = TickCircuit::new();
        c.tick().idle(1, &[0]);
        probe(&mut c, &[0]);
        c.tick().pz(&[0]);
        c.tick().x(&[0]);
        probe(&mut c, &[0]);
        measure(&mut c, &[0]);
        let noise = model(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_idle_rate(0.37)
                .with_reload_on_prep(reload)
                .build()
                .unwrap(),
        );
        if reload {
            let cat = engine_joint("PZ reload", &c, noise);
            assert!(
                cat.roots[0].branches[0]
                    .lost_over_time
                    .last()
                    .unwrap()
                    .lost
                    .is_empty()
            );
        } else {
            engine_joint("PZ dropped without reload", &c, noise);
        }
    }
    let mut c = TickCircuit::new();
    c.tick().x(&[0]);
    c.tick().idle(1, &[0]);
    let read = c.tick().mpz(&[0]);
    c.detector(&read).unwrap();
    probe(&mut c, &[0]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.37)
            .with_reload_on_prep(true)
            .build()
            .unwrap(),
    );
    let cat = engine_joint("MPZ read while lost before reload", &c, noise);
    assert_eq!(cat.roots[0].branches[0].lost_records, BTreeSet::from([0]));

    let mut c = TickCircuit::new();
    c.tick().mpz(&[0]);
    probe(&mut c, &[0]);
    measure(&mut c, &[0]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_prep(0.37)
            .with_reload_on_prep(true)
            .build()
            .unwrap(),
    );
    engine_joint("MPZ prep loss follows reload", &c, noise);
}

#[test]
fn engine_joint_measurement_and_z_herald_loss() {
    for kind in [
        GateType::MZ,
        GateType::MeasureZLost,
        GateType::MeasureZLeakedLost,
    ] {
        let mut c = TickCircuit::new();
        gate(&mut c, kind, &[0]);
        probe(&mut c, &[0]);
        let noise = model(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_measurement(0.37)
                .build()
                .unwrap(),
        );
        engine_joint(&format!("measurement source {kind}"), &c, noise);
    }
}

fn dependent_fixture() -> TickCircuit {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    let a = c.tick().mz(&[0])[0];
    c.tick().x(&[0]);
    let b = c.tick().mz(&[0])[0];
    let d = c.tick().mz(&[0])[0];
    c.detector(&[a, b]).unwrap();
    c.detector(&[b, d]).unwrap();
    c
}
#[test]
fn normal_form_rank_and_constant_reduction() {
    for w in [0.5, 1.0] {
        let cat = catalog(&dependent_fixture(), &idle_model(w));
        let form = &cat.roots[0].branches[0].normal_form;
        assert_eq!(
            form.span.rank(),
            2,
            "three dependent visible random signatures have GF(2) rank two"
        );
        assert!(form.constant.is_empty(), "c in S must reduce to zero");
        let terms = cat.background.to_mechanisms().0;
        assert_eq!(terms.len(), 3);
        for (p, _, _) in terms {
            assert!((p - correlated_probability(w, 2)).abs() < 1e-12);
        }
    }
}

#[test]
fn normal_form_character_expectations() {
    for w in [1e-3, 0.5, 1.0] {
        for rank in [1, 2, 3] {
            let q = correlated_probability(w, rank);
            for character in 1usize..1 << rank {
                let expectation = (1usize..1 << rank)
                    .filter(|v| (v & character).count_ones() % 2 == 1)
                    .fold(1.0, |e, _| e * (1.0 - 2.0 * q));
                assert!(
                    (expectation - (1.0 - w)).abs() < 1e-12,
                    "w={w}, rank={rank}, character={character}, q={q}"
                );
            }
        }
    }
}

fn recovery_fixture(w: f64, fp: f64) -> (TickCircuit, GeneralNoiseModel) {
    let mut c = TickCircuit::new();
    c.tick().h(&[0]);
    c.tick().cx(&[(0, 1)]);
    c.tick().idle(1, &[0]);
    c.tick().cx(&[(0, 1)]);
    c.tick().h(&[0]);
    measure(&mut c, &[0, 1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(w)
            .add_recovery_channel(LossSite::BeforeTwoQubit, 0.0, fp)
            .build()
            .unwrap(),
    );
    (c, noise)
}
#[test]
fn correlated_background_has_y_at_absolute_weight() {
    for w in [1e-3, 0.5] {
        let (c, noise) = recovery_fixture(w, 0.0);
        let cat = catalog(&c, &noise);
        assert_eq!(cat.roots.len(), 1);
        let branch = &cat.roots[0].branches[0];
        assert!((branch.weight - w).abs() < 1e-15);
        assert_eq!(branch.normal_form.span.rank(), 2);
        let terms = cat.background.to_mechanisms().0;
        assert_eq!(
            terms.len(),
            3,
            "X, Z and the joint Y signature are all required"
        );
        assert!(terms.iter().any(|(_, d, _)| d == &[0, 1]));
        for (p, _, _) in terms {
            assert!((p - correlated_probability(w, 2)).abs() < 1e-12);
        }
    }
}

#[test]
fn cutoff_accounts_mass_and_cap_names_root() {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    c.tick().cx(&[(0, 1)]);
    probe(&mut c, &[0]);
    measure(&mut c, &[0, 1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.4)
            .add_recovery_channel(LossSite::BeforeTwoQubit, 0.25, 0.0)
            .build()
            .unwrap(),
    );
    let cat = LossCatalog::build(&c, &noise, 0.15, 10, 10).unwrap();
    assert!((cat.roots[0].dropped_mass - 0.1).abs() < 1e-15);
    assert!((cat.report.dropped_mass - 0.1).abs() < 1e-15);
    assert_eq!(
        cat.roots[0]
            .dropped_mass_by_flags
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        [BTreeSet::from([ProbeId(0)])]
    );
    assert_eq!(cat.roots[0].branches.len(), 1);
    let error = match LossCatalog::build(&c, &noise, 0.0, 10, 1) {
        Ok(_) => panic!("cap ignored"),
        Err(e) => e.to_string(),
    };
    assert!(
        error.contains("Idle") && error.contains("count 2") && error.contains("max_branches"),
        "{error}"
    );
    for invalid in [f64::NAN, f64::INFINITY, -0.1, 1.1] {
        assert!(LossCatalog::build(&c, &noise, invalid, 10, 10).is_err());
    }
}

#[test]
fn catalog_loss_off_and_zero_loss_bytes() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0]);
    c.tick().idle(1, &[0]);
    c.tick().x(&[0]);
    measure(&mut c, &[0]);
    for noise in [
        GeneralNoiseModel::default(),
        model(
            QubitLoss::builder()
                .with_default_recovery(false)
                .add_events_channel(LossSite::BeforeSingleQubit, 0.0, &singleton("L"))
                .build()
                .unwrap(),
        ),
    ] {
        let expected = background_dem(&c, &noise).unwrap();
        let cat = catalog(&c, &noise);
        assert_eq!(cat.background.to_string(), expected.to_string());
        assert_eq!(cat.background.to_mechanisms(), expected.to_mechanisms());
        assert!(cat.roots.is_empty());
    }
}

#[test]
fn false_alarm_tags_are_unmerged_and_removal_matches_omission() {
    for fp in [0.2, 1.0] {
        let mut c = TickCircuit::new();
        c.tick().idle(1, &[0]);
        c.tick().cx(&[(0, 1)]);
        measure(&mut c, &[0, 1]);
        let noise = model(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_idle_rate(0.4)
                .add_recovery_channel(LossSite::BeforeTwoQubit, 0.0, fp)
                .build()
                .unwrap(),
        );
        let cat = catalog(&c, &noise);
        let tag = FalseAlarmTag {
            channel: 0,
            location: Location {
                tick: 1,
                batch: 0,
                operand: 0,
                side: Side::Before,
            },
        };
        assert_eq!(
            cat.roots[0].branches[0].false_alarm_removals,
            BTreeSet::from([tag])
        );
        let channel = |location| LocationChannel {
            location,
            probabilities: ["X", "Y", "Z"]
                .into_iter()
                .map(|s| (s.into(), fp / 4.0))
                .collect(),
        };
        let isolated = crate::DemBuilder::try_from_tick_circuit_with_location_channels(
            &c,
            crate::NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
            &[channel(tag.location)],
        )
        .unwrap();
        let unmerged: Vec<_> = isolated
            .contributions()
            .iter()
            .map(|c| (c.effect.clone(), c.probability))
            .collect();
        assert_eq!(contribution_pairs(&cat.false_alarm_tags()[&tag]), unmerged);
        let remaining: Vec<_> = cat
            .false_alarm_tags()
            .keys()
            .filter(|&&t| t != tag)
            .map(|t| channel(t.location))
            .collect();
        let mut omitted = crate::DemBuilder::try_from_tick_circuit_with_location_channels(
            &c,
            crate::NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
            &remaining,
        )
        .unwrap();
        // In this fixture base contains only the unflagged loss branch.
        assert_eq!(cat.base().len(), 1);
        for contribution in cat.base() {
            omitted.push_contribution(contribution.clone());
        }
        let removed = cat.background_without(&BTreeSet::from([tag]));
        assert_eq!(removed.to_mechanisms(), omitted.to_mechanisms());
        assert_eq!(removed.to_string(), omitted.to_string());
        let unchanged = cat.background_without(&BTreeSet::new());
        assert_eq!(unchanged.to_mechanisms(), cat.background.to_mechanisms());
        assert_eq!(unchanged.to_string(), cat.background.to_string());
    }
}

// Rotated d=3 planar Z memory: nine data, four Z checks then four X
// checks. Sequential PZ/H/CX/MZ syndrome extraction avoids unsupported MX.
const Z_CHECKS: [&[usize]; 4] = [&[0, 1], &[1, 2, 4, 5], &[3, 4, 6, 7], &[7, 8]];
const X_CHECKS: [&[usize]; 4] = [&[0, 1, 3, 4], &[2, 5], &[3, 6], &[4, 5, 7, 8]];
fn surface_memory(lost: usize) -> (TickCircuit, Vec<Vec<TickMeasRef>>) {
    let mut c = TickCircuit::new();
    c.tick().pz(&(0..9).collect::<Vec<_>>());
    let mut rounds = Vec::new();
    for round in 0..4 {
        c.tick().pz(&(9..17).collect::<Vec<_>>());
        if round == 1 {
            c.tick().idle(1, &[lost]);
        }
        let mut results = Vec::new();
        for (check, support) in Z_CHECKS.iter().chain(&X_CHECKS).enumerate() {
            let ancilla = 9 + check;
            if check >= 4 {
                c.tick().h(&[ancilla]);
            }
            for &q in *support {
                let pair = if check >= 4 {
                    (ancilla, q)
                } else {
                    (q, ancilla)
                };
                c.tick().cx(&[pair]);
            }
            if check >= 4 {
                c.tick().h(&[ancilla]);
            }
            results.push(c.tick().mz(&[ancilla])[0]);
        }
        rounds.push(results);
    }
    for round in 1..4 {
        for check in 0..8 {
            c.detector(&[rounds[round - 1][check], rounds[round][check]])
                .unwrap();
        }
    }
    (c, rounds)
}
#[test]
fn gauge_surface_d3_supercheck_and_punctured_checks() {
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(1.0)
            .with_reload_on_prep(true)
            .with_lost_partner_loss_probability(0.0)
            .build()
            .unwrap(),
    );
    let (c, _) = surface_memory(9);
    let cat = catalog(&c, &noise);
    let span = &cat.roots[0].branches[0].normal_form.span;
    for v in span.basis() {
        assert_eq!(
            v.contains(0),
            v.contains(8),
            "measure-qubit supercheck D0 D8 is deterministic"
        );
        assert!(!v.contains(16));
    }
    assert!(
        span.basis().iter().any(|v| v.contains(0)),
        "missing read convention makes individual detectors random"
    );
    let (c, _) = surface_memory(4);
    let cat = catalog(&c, &noise);
    let span = &cat.roots[0].branches[0].normal_form.span;
    for round in 1..3 {
        for check in [1, 2, 4, 7] {
            assert!(
                span.basis().iter().any(|v| v.contains(round * 8 + check)),
                "punctured check {check}, post-loss round {round} must be random"
            );
        }
    }
}

#[test]
fn pauli_outcomes_follow_dropped_gates_and_compose() {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    c.tick().cx(&[(0, 1)]);
    c.tick().cx(&[(1, 0)]);
    probe(&mut c, &[0]);
    measure(&mut c, &[0, 1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.4)
            .with_lost_partner_loss_probability(0.0)
            .with_lost_partner_pauli_model(&BTreeMap::from([
                ("I".into(), 0.8),
                ("X".into(), 0.2),
                ("Y".into(), 0.0),
            ]))
            .build()
            .unwrap(),
    );
    let cat = catalog(&c, &noise);
    let branches = &cat.roots[0].branches;
    assert_eq!(
        branches.len(),
        4,
        "two successive binary Pauli choices; zero outcomes excluded"
    );
    let mut weights = branches.iter().map(|b| b.weight).collect::<Vec<_>>();
    weights.sort_by(f64::total_cmp);
    for (actual, expected) in weights.iter().zip([0.016, 0.064, 0.064, 0.256]) {
        assert!((actual - expected).abs() < 1e-14);
    }
    let flipped_weight: f64 = branches
        .iter()
        .filter(|b| b.normal_form.constant.contains(1))
        .map(|b| b.weight)
        .sum();
    assert!((flipped_weight - 0.128).abs() < 1e-14);
    for branch in branches {
        assert_eq!(branch.flags, BTreeSet::from([ProbeId(0)]));
        assert!(
            !branch.normal_form.constant.contains(0),
            "the second CX is dropped, so partner X never propagates onto D0"
        );
        assert_eq!(branch.normal_form.span.basis(), &[BitSet::single(0)]);
    }
    assert_eq!(cat.report.approximated, BTreeMap::new());
}

#[test]
fn weighted_coset_rank_limit_and_random_observable_reports() {
    for w in [0.4, 1.0] {
        let mut c = TickCircuit::new();
        c.tick().h(&[1]);
        c.tick().cx(&[(0, 1)]);
        c.tick().h(&[1]);
        measure(&mut c, &[1]);
        let noise = model(
            QubitLoss::builder()
                .with_default_recovery(false)
                .add_events_channel(LossSite::BeforeTwoQubit, w, &singleton("LZ"))
                .build()
                .unwrap(),
        );
        let cat = catalog(&c, &noise);
        let branch = &cat.roots[0].branches[0];
        assert_eq!(branch.normal_form.span.rank(), 0);
        assert_eq!(branch.normal_form.constant, BitSet::single(0));
        let terms = cat.background.to_mechanisms().0;
        assert_eq!(terms.len(), 1);
        assert!((terms[0].0 - w).abs() < 1e-14);
        assert!(
            cat.report.approximated.is_empty(),
            "rank-zero constant is exact at every weight"
        );
    }
    let (c, noise) = recovery_fixture(0.4, 0.0);
    let cat = LossCatalog::build(&c, &noise, 0.0, 1, 10).unwrap();
    assert_eq!(cat.background.to_mechanisms().0.len(), 2);
    assert_eq!(
        cat.report.approximated[&BranchId { root: 0, branch: 0 }],
        BTreeSet::from([Approximation::RankLimit])
    );

    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    let read = c.tick().mz(&[0]);
    c.observable(&read).unwrap();
    let cat = catalog(&c, &idle_model(0.4));
    assert_eq!(cat.num_detectors, 0);
    assert_eq!(cat.num_observables, 1);
    assert_eq!(cat.report.random_observable_count, 1);
    assert!((cat.report.random_observable_weight - 0.4).abs() < 1e-15);
    assert_eq!(cat.background.to_mechanisms().0[0].2, [0]);
}

#[test]
fn rejects_random_lossless_definitions_by_name_and_handles_duplicate_records() {
    let mut c = TickCircuit::new();
    c.tick().h(&[0]);
    let read = c.tick().mz(&[0]);
    c.detector(&read).unwrap();
    let error = match LossCatalog::build(&c, &idle_model(0.0), 0.0, 5, 10) {
        Ok(_) => panic!("random detector accepted"),
        Err(e) => e.to_string(),
    };
    assert!(
        error.contains("lossless detector 0") && error.contains("nondeterministic"),
        "{error}"
    );
    let mut c = TickCircuit::new();
    c.tick().h(&[0]);
    let read = c.tick().mz(&[0]);
    c.observable(&read).unwrap();
    let error = match LossCatalog::build(&c, &idle_model(0.0), 0.0, 5, 10) {
        Ok(_) => panic!("random observable accepted"),
        Err(e) => e.to_string(),
    };
    assert!(error.contains("lossless observable 0"), "{error}");
    let mut c = TickCircuit::new();
    c.tick().h(&[0]);
    c.tick().mz(&[0]);
    c.add_detector_metadata(&[0, 0], None, None, None, None)
        .unwrap();
    assert_eq!(catalog(&c, &idle_model(0.0)).roots.len(), 0);
}

#[test]
fn joint_and_spread_partners_all_get_false_alarm_removals() {
    let mut c = TickCircuit::new();
    c.tick().cx(&[(0, 1)]);
    probe(&mut c, &[0, 1]);
    c.tick().cz(&[(1, 2)]);
    c.tick().h(&[0, 1, 2]);
    probe(&mut c, &[0, 1, 2]);
    let noise = GeneralNoiseModel::builder()
        .with_noiseless_gate(GateType::CZ)
        .with_loss(
            QubitLoss::builder()
                .with_default_recovery(false)
                .add_events_channel(LossSite::BeforeTwoQubit, 0.4, &singleton("LL"))
                .add_recovery_channel(LossSite::BeforeSingleQubit, 0.5, 0.2)
                .build()
                .unwrap(),
        )
        .build();
    let cat = catalog(&c, &noise);
    assert_eq!(cat.roots.len(), 1);
    assert_eq!(cat.roots[0].branches.len(), 4);
    for branch in &cat.roots[0].branches {
        assert_eq!(branch.lost_over_time[0].lost, BTreeSet::from([0, 1]));
        assert_eq!(
            branch
                .lost_over_time
                .iter()
                .filter(|m| m.location == cat.roots[0].key.location
                    && m.phase == cat.roots[0].key.phase)
                .count(),
            1
        );
        assert!((branch.weight - 0.1).abs() < 1e-15);
        assert_eq!(
            branch
                .false_alarm_removals
                .iter()
                .map(|t| t.location.operand)
                .collect::<Vec<_>>(),
            [0, 1]
        );
    }
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    c.tick().cx(&[(0, 1)]);
    c.tick().cx(&[(1, 2)]);
    c.tick().h(&[0, 1, 2]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.4)
            .with_lost_partner_loss_probability(1.0)
            .add_recovery_channel(LossSite::BeforeSingleQubit, 0.0, 0.2)
            .build()
            .unwrap(),
    );
    let cat = catalog(&c, &noise);
    assert_eq!(
        cat.roots[0].branches[0]
            .false_alarm_removals
            .iter()
            .map(|t| t.location.operand)
            .collect::<Vec<_>>(),
        [0, 1, 2]
    );
}

#[test]
fn roots_use_scaled_values_hook_order_and_phases() {
    let mut c = TickCircuit::new();
    c.tick().x(&[0]);
    c.tick().cx(&[(0, 1)]);
    c.tick().mpz(&[0]);
    let noise = GeneralNoiseModel::builder()
        .with_p1_scale(0.5)
        .with_idle_after_2q(2.0)
        .with_loss(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_measurement(0.2)
                .with_prep(0.3)
                .with_idle_rate(0.1)
                .add_events_channel(
                    LossSite::BeforeSingleQubit,
                    0.5,
                    &BTreeMap::from([("L".into(), 0.4), ("X".into(), 0.6)]),
                )
                .add_recovery_channel(LossSite::BeforeSingleQubit, 0.2, 0.0)
                .add_events_channel(LossSite::AfterTwoQubit, 0.1, &singleton("LL"))
                .add_recovery_channel(LossSite::AfterTwoQubit, 0.4, 0.0)
                .with_reload_on_prep(true)
                .build()
                .unwrap(),
        )
        .build();
    let cat = catalog(&c, &noise);
    let root = &cat.roots[0];
    assert_eq!(root.key.channel, Some(0));
    assert_eq!(
        root.key.phase,
        RootPhase::BeforeHook(1),
        "recovery is composed before the earlier-listed event"
    );
    assert!((root.prior - 0.1).abs() < 1e-15);
    let pair = cat.roots.iter().find(|r| r.key.label == "LL").unwrap();
    assert_eq!(pair.key.phase, RootPhase::AfterHook(0));
    assert_eq!(pair.branches.len(), 4);
    let phases: Vec<_> = cat
        .roots
        .iter()
        .filter(|r| r.key.location.tick == 2)
        .map(|r| r.key.phase)
        .collect();
    assert_eq!(
        phases,
        [RootPhase::MeasurementLoss, RootPhase::PreparationLoss]
    );
    assert_eq!(
        cat.roots
            .iter()
            .filter(|r| r.key.kind == RootKind::AfterTwoQubitIdle)
            .count(),
        2
    );
}

#[test]
fn noiseless_preparations_reload_without_draws() {
    for kind in [GateType::PZ, GateType::MPZ] {
        let mut c = TickCircuit::new();
        c.tick().idle(1, &[0]);
        probe(&mut c, &[0]);
        if kind == GateType::MPZ {
            c.tick().mpz(&[0]);
        } else {
            c.tick().pz(&[0]);
        }
        probe(&mut c, &[0]);
        let noise = GeneralNoiseModel::builder()
            .with_noiseless_gate(kind)
            .with_loss(
                QubitLoss::builder()
                    .with_default_recovery(false)
                    .with_reload_on_prep(true)
                    .with_idle_rate(0.4)
                    .with_prep(1.0)
                    .with_measurement(1.0)
                    .build()
                    .unwrap(),
            )
            .build();
        let cat = engine_joint(&format!("noiseless reload {kind}"), &c, noise);
        assert_eq!(cat.roots.len(), 1);
        assert_eq!(cat.roots[0].branches[0].flags, BTreeSet::from([ProbeId(0)]));
    }
}

fn bell_partner_fixture() -> TickCircuit {
    let mut c = TickCircuit::new();
    c.tick().h(&[1]);
    c.tick().cx(&[(1, 2)]);
    c.tick().idle(1, &[0]);
    c.tick().cx(&[(0, 1)]);
    probe(&mut c, &[0]);
    c.tick().cx(&[(1, 2)]);
    c.tick().h(&[1]);
    let read = c.tick().mz(&[1, 2]);
    for r in &read {
        c.detector(&[*r]).unwrap();
    }
    c.observable(&[read[0]]).unwrap();
    c
}

#[test]
fn engine_joint_partner_cosets_in_bell_basis() {
    let c = bell_partner_fixture();
    for support in [
        vec!["Y"],
        vec!["I", "X"],
        vec!["I", "Y"],
        vec!["I", "Z"],
        vec!["X", "Y"],
        vec!["X", "Z"],
        vec!["Y", "Z"],
        vec!["I", "X", "Y", "Z"],
    ] {
        let p = 1.0 / f64::from(u32::try_from(support.len()).unwrap());
        let weights = support.iter().map(|s| ((*s).into(), p)).collect();
        let noise = model(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_idle_rate(0.37)
                .with_lost_partner_loss_probability(0.0)
                .with_lost_partner_pauli_model(&weights)
                .build()
                .unwrap(),
        );
        let cat = engine_joint(&format!("Bell partner {support:?}"), &c, noise);
        assert_eq!(
            cat.roots[0].branches.len(),
            1,
            "single Paulis and cosets retain one coin-form branch"
        );
    }
}

#[test]
fn batch_emissions_and_inserted_ticks_keep_original_record_definitions() {
    let mut c = TickCircuit::new();
    c.tick().pz(&[0, 1, 2]);
    let mut tick = c.tick();
    tick.idle(1, &[0]);
    tick.h(&[1]);
    c.tick().h(&[1]);
    let before = c.tick().mz(&[1, 0, 2]);
    let after = c.tick().mz(&[0, 2, 1]);
    c.detector(&[before[1], after[0]]).unwrap();
    c.detector(&[before[0], after[2]]).unwrap();
    c.observable(&[before[2], after[1]]).unwrap();
    let cat = catalog(&c, &idle_model(0.4));
    let branch = &cat.roots[0].branches[0];
    assert_eq!(branch.lost_records, BTreeSet::from([1, 3]));
    assert_eq!(branch.normal_form.span.basis(), &[BitSet::single(0)]);
    assert_eq!(
        cat.outputs
            .records
            .iter()
            .map(|r| r.qubit.index())
            .collect::<Vec<_>>(),
        [1, 0, 2, 0, 2, 1]
    );
}

fn survivor_model(zero_character: bool) -> BTreeMap<String, f64> {
    if zero_character {
        BTreeMap::from([
            ("I".into(), 3.0 / 8.0),
            ("Z".into(), 3.0 / 8.0),
            ("X".into(), 1.0 / 8.0),
            ("Y".into(), 1.0 / 8.0),
        ])
    } else {
        ["X", "Y", "Z"]
            .into_iter()
            .map(|s| (s.into(), 1.0 / 3.0))
            .collect()
    }
}

fn assert_partner_outcomes(cat: &LossCatalog, weights: &BTreeMap<String, f64>) {
    assert_eq!(cat.roots.len(), 1);
    let branches = &cat.roots[0].branches;
    assert_eq!(branches.len(), weights.len());
    for (label, p) in weights {
        // Bell decoding: X -> D1; Z -> D0 and its duplicate observable L0.
        let expected = match label.as_str() {
            "X" => 2,
            "Y" => 7,
            "Z" => 5,
            _ => 0,
        };
        let branch = branches
            .iter()
            .find(|b| mask(&b.normal_form.constant) == expected)
            .unwrap();
        assert!((branch.weight - 0.37 * p).abs() < 1e-14);
        assert_eq!(
            branch.normal_form.span.rank(),
            0,
            "each outcome is a certain gate"
        );
        assert_eq!(branch.flags, BTreeSet::from([ProbeId(0)]));
    }
    assert_eq!(
        cat.report.approximated,
        BTreeMap::new(),
        "flagged Pauli sub-branches have no extra approximation"
    );
}

#[test]
fn engine_joint_partner_negative_characters() {
    let weights = survivor_model(false);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.37)
            .with_lost_partner_loss_probability(0.0)
            .with_lost_partner_pauli_model(&weights)
            .build()
            .unwrap(),
    );
    let cat = engine_joint(
        "Bell negative-character XYZ",
        &bell_partner_fixture(),
        noise,
    );
    assert_partner_outcomes(&cat, &weights);
}

#[test]
fn engine_joint_partner_zero_character() {
    let weights = survivor_model(true);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.37)
            .with_lost_partner_loss_probability(0.0)
            .with_lost_partner_pauli_model(&weights)
            .build()
            .unwrap(),
    );
    let cat = engine_joint(
        "Bell zero-character Perrin-Roger",
        &bell_partner_fixture(),
        noise,
    );
    assert_partner_outcomes(&cat, &weights);
}

#[test]
fn engine_joint_mixed_event_pauli_outcomes() {
    let mut c = TickCircuit::new();
    c.tick().h(&[1]);
    c.tick().cx(&[(1, 2)]);
    c.tick().cz(&[(0, 1)]);
    probe(&mut c, &[0]);
    c.tick().cx(&[(1, 2)]);
    c.tick().h(&[1]);
    let read = c.tick().mz(&[1, 2]);
    for r in &read {
        c.detector(&[*r]).unwrap();
    }
    c.observable(&[read[0]]).unwrap();
    for zero_character in [false, true] {
        let weights = survivor_model(zero_character);
        let events = weights
            .iter()
            .map(|(label, p)| (format!("L{label}"), *p))
            .collect();
        let noise = GeneralNoiseModel::builder()
            .with_noiseless_gate(GateType::CX)
            .with_loss(
                QubitLoss::builder()
                    .with_default_recovery(false)
                    .add_events_channel(LossSite::BeforeTwoQubit, 0.37, &events)
                    .build()
                    .unwrap(),
            )
            .build();
        let cat = engine_joint(&format!("Bell mixed events {events:?}"), &c, noise);
        assert_eq!(
            cat.roots.len(),
            weights.len(),
            "mixed labels retain their root identity"
        );
        for root in &cat.roots {
            assert_eq!(root.branches.len(), 1);
            assert_eq!(root.branches[0].normal_form.span.rank(), 0);
            assert!((root.prior - 0.37 * weights[&root.key.label[1..]]).abs() < 1e-14);
        }
        assert_eq!(cat.report.approximated, BTreeMap::new());
    }
}

#[test]
fn pauli_subbranches_inherit_metadata_and_obey_limits() {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    c.tick().z(&[0]);
    c.tick().cx(&[(0, 1)]);
    probe(&mut c, &[0, 1]);
    measure(&mut c, &[0, 1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.4)
            .add_recovery_channel(LossSite::BeforeSingleQubit, 0.4, 0.0)
            .with_lost_partner_loss_probability(0.25)
            .with_lost_partner_pauli_model(&survivor_model(true))
            .build()
            .unwrap(),
    );
    let tag = FalseAlarmTag {
        channel: 0,
        location: Location {
            tick: 2,
            batch: 0,
            operand: 0,
            side: Side::Before,
        },
    };
    let full = catalog(&c, &noise);
    assert_eq!(
        full.roots[0].branches.len(),
        6,
        "one recovered, one spread, four Pauli outcomes"
    );
    let flags = BTreeSet::from([ProbeId(0), ProbeId(1)]);
    let paulis = full.roots[0]
        .branches
        .iter()
        .filter(|b| b.flags == flags)
        .collect::<Vec<_>>();
    assert_eq!(paulis.len(), 4);
    for child in paulis {
        assert_eq!(child.false_alarm_removals, BTreeSet::from([tag]));
        assert_eq!(
            child.lost_over_time.len(),
            1,
            "a Pauli does not change the lost set"
        );
        assert_eq!(child.lost_over_time[0].lost, BTreeSet::from([0]));
    }
    assert!((full.roots[0].branches.iter().map(|b| b.weight).sum::<f64>() - 0.4).abs() < 1e-14);
    let cut = LossCatalog::build(&c, &noise, 0.02, 12, 5).unwrap();
    assert_eq!(cut.roots[0].branches.len(), 4);
    assert_eq!(cut.roots[0].explored_paths, 5);
    assert!(
        (cut.roots[0].dropped_mass - 0.03).abs() < 1e-14,
        "two Pauli children at .4*.4*.75*.125 are pruned"
    );
    assert_eq!(
        cut.roots[0]
            .dropped_mass_by_flags
            .keys()
            .cloned()
            .collect::<Vec<_>>(),
        [BTreeSet::from([ProbeId(0)])]
    );
    assert!((cut.report.dropped_mass - 0.03).abs() < 1e-14);
    for (cutoff, cap, count) in [(0.0, 6, 7), (0.02, 4, 5)] {
        let error = match LossCatalog::build(&c, &noise, cutoff, 12, cap) {
            Ok(_) => panic!("Pauli sub-branches ignored by cap"),
            Err(e) => e.to_string(),
        };
        assert!(
            error.contains("Idle") && error.contains(&format!("count {count}")),
            "{error}"
        );
    }
}

#[test]
fn false_alarm_build_is_shared_and_zero_fp_needs_no_channel_build() {
    use super::channels::LOCATION_CHANNEL_BUILDS;
    for fp in [0.0, 0.2, 1.0] {
        let (c, noise) = recovery_fixture(0.0, fp);
        LOCATION_CHANNEL_BUILDS.with(|count| count.set(0));
        let cat = catalog(&c, &noise);
        assert_eq!(
            LOCATION_CHANNEL_BUILDS.with(std::cell::Cell::get),
            usize::from(fp > 0.0)
        );
        assert_eq!(cat.false_alarm_tags().len(), 4);
        if fp == 0.0 {
            assert!(cat.false_alarm_tags().values().all(Vec::is_empty));
        }
        assert_eq!(
            cat.background.to_string(),
            background_dem(&c, &noise).unwrap().to_string()
        );
    }
}

#[test]
fn tagged_background_preserves_base_and_interleaved_order() {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    c.tick().x(&[0]);
    c.tick().z(&[0]);
    measure(&mut c, &[0]);
    let loss = QubitLoss::builder()
        .with_default_recovery(false)
        .with_idle_rate(0.23)
        .add_recovery_channel(LossSite::BeforeSingleQubit, 0.0, 0.17)
        .add_events_channel(LossSite::BeforeSingleQubit, 0.13, &singleton("X"))
        .add_recovery_channel(LossSite::BeforeSingleQubit, 0.0, 0.31)
        .build()
        .unwrap();
    let noise = GeneralNoiseModel::builder()
        .with_p1(0.07)
        .with_loss(loss)
        .build();
    let cat = catalog(&c, &noise);
    let no_fp = GeneralNoiseModel::builder()
        .with_p1(0.07)
        .with_loss(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_idle_rate(0.23)
                .add_recovery_channel(LossSite::BeforeSingleQubit, 0.0, 0.0)
                .add_events_channel(LossSite::BeforeSingleQubit, 0.13, &singleton("X"))
                .add_recovery_channel(LossSite::BeforeSingleQubit, 0.0, 0.0)
                .build()
                .unwrap(),
        )
        .build();
    let baseline = catalog(&c, &no_fp);
    let contributions = baseline
        .background
        .contributions()
        .iter()
        .map(|c| (c.effect.clone(), c.probability))
        .collect::<Vec<_>>();
    assert_eq!(
        contribution_pairs(cat.base()),
        contributions,
        "bridge, pure events and unflagged branches stay in base"
    );
    // X and Y can have the same effect: the tag must retain both independent terms.
    assert!(cat.false_alarm_tags().values().any(|terms| {
        terms.len()
            > terms
                .iter()
                .map(|c| &c.effect)
                .collect::<BTreeSet<_>>()
                .len()
    }));
    let removed = cat.background_without(&cat.false_alarm_tags().keys().copied().collect());
    assert_eq!(removed.to_mechanisms(), baseline.background.to_mechanisms());
    assert_eq!(removed.to_string(), baseline.background.to_string());
    let merged = cat.background_without(&BTreeSet::new());
    let contributions = |dem: &crate::DetectorErrorModel| {
        dem.contributions()
            .iter()
            .map(|c| (c.effect.clone(), c.probability))
            .collect::<Vec<_>>()
    };
    assert_eq!(
        contributions(&merged),
        contributions(&cat.background),
        "preserve ordinary insertion order even when regrouping happens to round identically"
    );
    assert_eq!(merged.to_mechanisms(), cat.background.to_mechanisms());
    assert_eq!(merged.to_string(), cat.background.to_string());
    let mut ordinary = background_dem(&c, &noise).unwrap();
    for branch in &cat.roots[0].branches {
        super::branches::emit_normal_form(
            &mut ordinary,
            &branch.normal_form,
            branch.weight,
            12,
            cat.num_detectors,
        )
        .unwrap();
    }
    assert_eq!(cat.background.to_string(), ordinary.to_string());
}

#[test]
fn gauge_keep_dropped_switch_leaves_punctured_checks_deterministic() {
    let (c, _) = surface_memory(4);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(1.0)
            .with_reload_on_prep(true)
            .with_lost_partner_loss_probability(0.0)
            .build()
            .unwrap(),
    );
    let cat = LossCatalog::build_with_options(
        &c,
        &noise,
        0.0,
        12,
        1000,
        ReplayOptions {
            keep_dropped: true,
            ..ReplayOptions::default()
        },
    )
    .unwrap();
    let span = &cat.roots[0].branches[0].normal_form.span;
    for round in 1..3 {
        for check in [1, 2, 4, 7] {
            assert!(
                !span.basis().iter().any(|v| v.contains(round * 8 + check)),
                "kept interactions make punctured check {check}, round {round} deterministic; oracle 2 must reject this"
            );
        }
    }
}

#[test]
fn engine_joint_after_two_qubit_idle_follows_recovery() {
    let mut c = TickCircuit::new();
    c.tick().cx(&[(0, 1)]);
    probe(&mut c, &[0]);
    measure(&mut c, &[0]);
    // Leg 1's idle is unobservable; leg 0 is the only effective source site.
    let noise = GeneralNoiseModel::builder()
        .with_idle_after_2q(1.0)
        .with_loss(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_idle_rate(0.37)
                .add_recovery_channel(LossSite::AfterTwoQubit, 0.0, 0.0)
                .build()
                .unwrap(),
        )
        .build();
    let cat = engine_joint("idle after after-recovery", &c, noise);
    assert_eq!(cat.roots[0].key.kind, RootKind::AfterTwoQubitIdle);
    assert_eq!(cat.roots[0].branches[0].flags, BTreeSet::from([ProbeId(0)]));
}

#[test]
fn engine_joint_mpz_measurement_loss_precedes_reload() {
    let mut c = TickCircuit::new();
    let read = c.tick().mpz(&[0]);
    c.detector(&read).unwrap();
    probe(&mut c, &[0]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_measurement(0.37)
            .with_reload_on_prep(true)
            .build()
            .unwrap(),
    );
    // The sole nonzero source is erased before it can affect any output.
    let cat = engine_joint_with_effective_sites("MPZ measurement loss before reload", &c, noise, 0);
    assert_eq!(cat.roots.len(), 1);
    let branch = &cat.roots[0].branches[0];
    assert!(branch.flags.is_empty());
    assert!(branch.lost_over_time.last().unwrap().lost.is_empty());
}

#[test]
fn branch_cap_counts_explored_paths_even_when_all_leaves_prune() {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    for partner in 1..=8 {
        c.tick().cx(&[(0, partner)]);
    }
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(1.0)
            .with_lost_partner_loss_probability(0.5)
            .build()
            .unwrap(),
    );
    // Each of the 256 leaves weighs 1/256. All prune at the final fork,
    // after a substantial number of path segments have already been explored.
    let cutoff = 1.0 / 256.0 + f64::EPSILON;
    let cat = LossCatalog::build(&c, &noise, cutoff, 12, 1000).unwrap();
    assert!(cat.roots[0].branches.is_empty());
    assert_eq!(cat.roots[0].dropped_mass, 1.0);
    assert!(cat.roots[0].explored_paths > 16);
    let error = LossCatalog::build(&c, &noise, cutoff, 12, 16)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("Idle") && error.contains("count 17 explored paths"),
        "{error}"
    );

    // Even a root pruned immediately is one explored segment.
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    let noise = idle_model(0.4);
    let cat = LossCatalog::build(&c, &noise, 0.5, 12, 1).unwrap();
    assert_eq!(cat.roots[0].explored_paths, 1);
    assert!(cat.roots[0].branches.is_empty());
    let error = LossCatalog::build(&c, &noise, 0.5, 12, 0)
        .unwrap_err()
        .to_string();
    assert!(error.contains("count 1 explored paths"), "{error}");
}

#[test]
fn indexed_actions_skip_unrelated_qubits_and_follow_lost_set_changes() {
    use super::trajectory::{ActionKind, Program};
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    for _ in 0..128 {
        c.tick().x(&[9]);
    }
    c.tick().cx(&[(0, 1)]);
    c.tick().h(&[0]);
    probe(&mut c, &[1]);
    measure(&mut c, &[1]);
    c.tick().pz(&[1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.4)
            .with_lost_partner_loss_probability(1.0)
            .with_reload_on_prep(true)
            .add_recovery_channel(LossSite::BeforeSingleQubit, 0.0, 0.0)
            .build()
            .unwrap(),
    );
    let program = Program::new(&c, &noise.description(), &OutputTable::new(&c).unwrap()).unwrap();
    let root = &program.roots[0];
    let next =
        |cursor, lost: &BTreeSet<usize>| program.next_action(cursor, lost).map(|(index, _)| index);
    let pair = next(root.action + 1, &BTreeSet::from([0])).unwrap();
    assert_eq!(program.actions[pair].location.tick, 129);
    assert_eq!(
        next(root.action, &BTreeSet::from([0])),
        Some(pair),
        "a source slot cannot be returned as a traversable action"
    );
    assert!(
        matches!(&program.actions[pair].kind, Some(ActionKind::Gate { gate, .. }) if gate.gate_type == GateType::CX)
    );
    let recovery = next(pair + 1, &BTreeSet::from([0, 1])).unwrap();
    assert!(matches!(
        program.actions[recovery].kind,
        Some(ActionKind::Recovery { qubit: 0, .. })
    ));
    let probe = next(recovery + 1, &BTreeSet::from([1])).unwrap();
    assert_eq!(program.actions[probe].location.tick, 131);
    let record = next(probe + 1, &BTreeSet::from([1])).unwrap();
    assert!(matches!(
        program.actions[record].kind,
        Some(ActionKind::Gate {
            record: Some(0),
            ..
        })
    ));
    let reload = next(record + 1, &BTreeSet::from([1])).unwrap();
    assert!(matches!(
        program.actions[reload].kind,
        Some(ActionKind::Reload { qubit: 1 })
    ));
    assert_eq!(next(reload + 1, &BTreeSet::new()), None);
    super::trajectory::ACTION_VISITS.with(|count| count.set(0));
    let cat = catalog(&c, &noise);
    assert_eq!(
        super::trajectory::ACTION_VISITS.with(std::cell::Cell::get),
        5,
        "walk visits only the pair, recovery, probe, record and reload"
    );
    let branch = &cat.roots[0].branches[0];
    assert_eq!(branch.flags, BTreeSet::from([ProbeId(0)]));
    assert_eq!(branch.lost_records, BTreeSet::from([0]));
    assert_eq!(
        branch
            .lost_over_time
            .iter()
            .map(|m| m.lost.clone())
            .collect::<Vec<_>>(),
        [
            BTreeSet::from([0]),
            BTreeSet::from([0, 1]),
            BTreeSet::from([1]),
            BTreeSet::new()
        ]
    );
}

#[test]
fn weighted_positive_rank_coset_still_reports_approximation() {
    let form = BranchNormalForm {
        constant: BitSet::single(1),
        span: Span::new(2, &[BitSet::single(0)]).unwrap(),
    };
    let mut dem = crate::DetectorErrorModel::new();
    let reports = super::branches::emit_normal_form(&mut dem, &form, 0.4, 12, 2).unwrap();
    assert_eq!(reports, BTreeSet::from([Approximation::WeightedCoset]));
    assert_eq!(
        dem.to_mechanisms().0,
        vec![(0.2, vec![0, 1], vec![]), (0.2, vec![1], vec![])]
    );
}

fn contribution_pairs(
    contributions: &[crate::fault_tolerance::dem_builder::FaultContribution],
) -> Vec<(crate::FaultMechanism, f64)> {
    contributions
        .iter()
        .map(|c| (c.effect.clone(), c.probability))
        .collect()
}

pub(super) fn assert_same_sources(
    actual: &[crate::fault_tolerance::dem_builder::FaultContribution],
    expected: &[crate::fault_tolerance::dem_builder::FaultContribution],
) {
    assert_eq!(actual.len(), expected.len());
    for (a, b) in actual.iter().zip(expected) {
        assert_eq!(a.effect, b.effect);
        assert_eq!(a.probability.to_bits(), b.probability.to_bits());
        assert_eq!(a.source_type, b.source_type);
        assert_eq!(a.location_indices, b.location_indices);
        assert_eq!(a.paulis, b.paulis);
        assert_eq!(a.source_gate_types, b.source_gate_types);
        assert_eq!(a.source_before_flags, b.source_before_flags);
        assert_eq!(a.direct_source_family, b.direct_source_family);
        assert_eq!(a.direct_component_effects, b.direct_component_effects);
        assert_eq!(a.source_component_effects, b.source_component_effects);
        assert_eq!(a.replacement_branch, b.replacement_branch);
    }
}

pub(super) fn assert_same_dem_renderings(
    actual: &crate::DetectorErrorModel,
    expected: &crate::DetectorErrorModel,
) {
    use crate::fault_tolerance::dem_builder::TwoDetectorDirectRenderPolicy;
    assert_eq!(actual.to_string(), expected.to_string());
    assert_eq!(
        actual.to_string_decomposed(),
        expected.to_string_decomposed()
    );
    assert_eq!(
        actual.to_string_source_decomposed(),
        expected.to_string_source_decomposed()
    );
    assert_eq!(
        actual.to_string_source_graphlike_decomposed(),
        expected.to_string_source_graphlike_decomposed()
    );
    assert_eq!(
        actual.to_string_graphlike_search_decomposed(),
        expected.to_string_graphlike_search_decomposed()
    );
    assert_eq!(
        actual.to_string_decomposed_maximally(),
        expected.to_string_decomposed_maximally()
    );
    for policy in [
        TwoDetectorDirectRenderPolicy::KeepDirect,
        TwoDetectorDirectRenderPolicy::PreferRecordedComponents,
    ] {
        assert_eq!(
            actual.to_string_decomposed_with_two_detector_direct_policy(policy),
            expected.to_string_decomposed_with_two_detector_direct_policy(policy)
        );
        assert_eq!(
            actual.to_string_decomposed_maximally_with_two_detector_direct_policy(policy),
            expected.to_string_decomposed_maximally_with_two_detector_direct_policy(policy)
        );
    }
    assert_eq!(
        actual.to_string_terminal_graphlike_decomposed().unwrap(),
        expected.to_string_terminal_graphlike_decomposed().unwrap()
    );
    assert_eq!(actual.to_mechanisms(), expected.to_mechanisms());
    assert_eq!(
        actual.idle_noise_residuals(),
        expected.idle_noise_residuals()
    );
    for d0 in 0..4 {
        for d1 in d0 + 1..4 {
            assert_eq!(
                actual.graphlike_decomposable_count(d0, d1),
                expected.graphlike_decomposable_count(d0, d1)
            );
        }
    }
    assert_same_sources(actual.contributions(), expected.contributions());
}

#[test]
fn catalog_background_preserves_all_renderings_and_metadata() {
    let mut c = TickCircuit::new();
    c.tick().h(&[0]);
    c.tick().cx(&[(0, 1)]);
    c.tick().sz(&[0]);
    c.tick().szdg(&[0]);
    c.tick().idle(1, &[0]);
    c.tick().cx(&[(0, 1)]);
    c.tick().h(&[0]);
    c.tick().cx(&[(2, 3)]);
    measure(&mut c, &[0, 1, 2, 3]);
    for fp in [0.2, 1.0] {
        let noise = GeneralNoiseModel::builder()
            .with_p1(0.04)
            .with_p2(0.03)
            .with_loss(
                QubitLoss::builder()
                    .with_default_recovery(false)
                    .with_idle_rate(0.23)
                    .add_recovery_channel(LossSite::BeforeTwoQubit, 0.0, fp)
                    .add_events_channel(
                        LossSite::AfterTwoQubit,
                        0.2,
                        &BTreeMap::from([("XI".into(), 0.5), ("IX".into(), 0.5)]),
                    )
                    .build()
                    .unwrap(),
            )
            .build();
        let cat = catalog(&c, &noise);
        assert!(
            !cat.background.idle_noise_residuals().is_empty(),
            "fixture must exercise an inexact events fit"
        );
        assert!(cat.base().iter().any(|c| !c.location_indices.is_empty()));
        // Ordinary p1/p2 use exclusive-signature Direct contributions in this
        // builder. Nonzero Y/component/count cases have a structured-source
        // catalog fixture in branches.rs, without changing E1 emission semantics.
        assert_same_dem_renderings(&cat.background_without(&BTreeSet::new()), &cat.background);

        let tag = FalseAlarmTag {
            channel: 0,
            location: Location {
                tick: 5,
                batch: 0,
                operand: 0,
                side: Side::Before,
            },
        };
        let mut channels = Vec::new();
        for tick in [1, 5, 7] {
            for operand in 0..2 {
                let location = Location {
                    tick,
                    batch: 0,
                    operand,
                    side: Side::Before,
                };
                if location != tag.location {
                    channels.push(LocationChannel {
                        location,
                        probabilities: ["X", "Y", "Z"]
                            .into_iter()
                            .map(|s| (s.into(), fp / 4.0))
                            .collect(),
                    });
                }
            }
            channels.push(LocationChannel {
                location: Location {
                    tick,
                    batch: 0,
                    operand: 0,
                    side: Side::After,
                },
                probabilities: BTreeMap::from([("XI".into(), 0.1), ("IX".into(), 0.1)]),
            });
        }
        let config = pauli_noise_config(&c, &noise.description()).unwrap();
        let mut omitted =
            crate::DemBuilder::try_from_tick_circuit_with_location_channels(&c, config, &channels)
                .unwrap();
        for root in &cat.roots {
            for branch in &root.branches {
                assert!(branch.flags.is_empty());
                super::branches::emit_normal_form(
                    &mut omitted,
                    &branch.normal_form,
                    branch.weight,
                    12,
                    cat.num_detectors,
                )
                .unwrap();
            }
        }
        assert_same_dem_renderings(&cat.background_without(&BTreeSet::from([tag])), &omitted);
        let isolated = crate::DemBuilder::try_from_tick_circuit_with_location_channels(
            &c,
            crate::NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
            &[LocationChannel {
                location: tag.location,
                probabilities: ["X", "Y", "Z"]
                    .into_iter()
                    .map(|s| (s.into(), fp / 4.0))
                    .collect(),
            }],
        )
        .unwrap();
        assert_same_sources(&cat.false_alarm_tags()[&tag], isolated.contributions());
    }
}

#[test]
fn location_channel_targets_cannot_cross_gate_instances() {
    let mut c = TickCircuit::new();
    c.tick().h(&[0, 1]);
    let error = crate::DemBuilder::try_from_tick_circuit_with_location_channels(
        &c,
        crate::NoiseConfig::new(0.0, 0.0, 0.0, 0.0),
        &[LocationChannel {
            location: Location {
                tick: 0,
                batch: 0,
                operand: 0,
                side: Side::Before,
            },
            probabilities: singleton("XX"),
        }],
    )
    .unwrap_err()
    .to_string();
    assert!(
        error.contains("crosses a gate instance; keep its operands in one instance"),
        "{error}"
    );
    assert!(
        error.contains("tick: 0") && error.contains("operand: 0"),
        "{error}"
    );
}
