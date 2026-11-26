use super::*;
use crate::fault_tolerance::dem_builder::{DemOutput, DetectorDef, FaultContribution};
use crate::{DetectorErrorModel, FaultMechanism};
use pecos_core::{BitSet, Gate, gate_type::GateType};
use pecos_decoder_core::ObservableDecoder;
use pecos_decoders::DecoderSpec;
use pecos_engines::noise::GeneralNoiseModel;
use pecos_engines::noise::general::{LossSite, QubitLoss};
use pecos_quantum::TickCircuit;
use std::collections::{BTreeMap, BTreeSet};

fn bits(mask: usize) -> BitSet {
    (0..usize::BITS as usize)
        .filter(|i| mask & (1 << i) != 0)
        .collect()
}
fn mask(bits: &BitSet) -> usize {
    bits.iter().fold(0, |m, i| m | (1 << i))
}
fn blank(width: usize) -> ShotBackground {
    let mut dem = DetectorErrorModel::new();
    for i in 0..width {
        dem.add_detector(DetectorDef::new(u32::try_from(i).unwrap()));
    }
    ShotBackground {
        dem,
        removed: BTreeMap::new(),
    }
}
fn alt(width: usize, pi: f64, constant: usize, span: &[usize]) -> ShotAlternative {
    ShotAlternative {
        branches: Vec::new(),
        weight: pi,
        pi,
        normal_form: BranchNormalForm {
            constant: bits(constant),
            span: Span::new(width, &span.iter().map(|&v| bits(v)).collect::<Vec<_>>()).unwrap(),
        },
        removals: BTreeSet::new(),
    }
}
fn effect_mask(effect: &FaultMechanism, detectors: usize) -> usize {
    effect.detectors.iter().fold(0, |m, &i| m ^ (1 << i))
        ^ effect
            .dem_outputs
            .iter()
            .fold(0, |m, &i| m ^ (1 << (detectors + i as usize)))
}
fn mechanisms(shot: &ShotDem) -> BTreeMap<usize, f64> {
    shot.dem
        .to_mechanisms()
        .0
        .into_iter()
        .map(|(p, d, o)| {
            (
                effect_mask(
                    &FaultMechanism::from_unsorted(d, o),
                    shot.detector_bias.len(),
                ),
                p,
            )
        })
        .collect()
}
fn bias(shot: &ShotDem) -> usize {
    shot.detector_bias
        .iter()
        .enumerate()
        .fold(0, |m, (i, &b)| m ^ (usize::from(b) << i))
        ^ shot
            .observable_bias
            .iter_set_bits()
            .fold(0, |m, i| m ^ (1 << (shot.detector_bias.len() + i)))
}
fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-12, "{a:.16} != {b:.16}");
}

// Independent Walsh transform, including the bias. No production emission helpers.
fn emitted(shot: &ShotDem, width: usize) -> Vec<f64> {
    let mechanisms = mechanisms(shot);
    let characters: Vec<_> = (0usize..1 << width)
        .map(|h| {
            let sign = if (h & bias(shot)).count_ones().is_multiple_of(2) {
                1.0
            } else {
                -1.0
            };
            mechanisms.iter().fold(sign, |v, (&m, &p)| {
                if (h & m).count_ones() % 2 == 1 {
                    v * (1.0 - 2.0 * p)
                } else {
                    v
                }
            })
        })
        .collect();
    inverse(&characters)
}
fn inverse(characters: &[f64]) -> Vec<f64> {
    (0..characters.len())
        .map(|x| {
            characters
                .iter()
                .enumerate()
                .map(|(h, &v)| {
                    if (h & x).count_ones().is_multiple_of(2) {
                        v
                    } else {
                        -v
                    }
                })
                .sum::<f64>()
                / f64::from(u32::try_from(characters.len()).unwrap())
        })
        .collect()
}
fn exact(groups: &[Vec<ShotAlternative>], width: usize) -> Vec<f64> {
    inverse(
        &(0usize..1 << width)
            .map(|h| {
                groups
                    .iter()
                    .map(|g| {
                        g.iter()
                            .map(|a| {
                                if a.normal_form
                                    .span
                                    .basis()
                                    .iter()
                                    .any(|v| (h & mask(v)).count_ones() % 2 == 1)
                                {
                                    0.0
                                } else if (h & mask(&a.normal_form.constant))
                                    .count_ones()
                                    .is_multiple_of(2)
                                {
                                    a.pi
                                } else {
                                    -a.pi
                                }
                            })
                            .sum::<f64>()
                    })
                    .product()
            })
            .collect::<Vec<_>>(),
    )
}
fn xor(a: &[f64], b: &[f64]) -> Vec<f64> {
    (0..a.len())
        .map(|x| a.iter().enumerate().map(|(y, p)| p * b[x ^ y]).sum())
        .collect()
}
fn tv(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(a, b)| (a - b).abs()).sum::<f64>() / 2.0
}
fn bound(groups: &[Vec<ShotAlternative>]) -> f64 {
    groups
        .iter()
        .map(|g| {
            let r = g
                .iter()
                .enumerate()
                .max_by(|(i, a), (j, b)| a.pi.total_cmp(&b.pi).then_with(|| j.cmp(i)))
                .unwrap()
                .0;
            1.0 - g[r].pi
                * g.iter()
                    .enumerate()
                    .filter(|(i, _)| *i != r)
                    .map(|(_, a)| 1.0 - a.pi)
                    .product::<f64>()
        })
        .sum()
}

#[test]
fn oracle1_mixing_mechanisms_and_exact_cases() {
    let q3 = (1.0 - 0.7f64.sqrt()) / 2.0;
    let q5 = (1.0 - 0.5f64.sqrt()) / 2.0;
    let fixtures = [
        (
            "shared half",
            3,
            vec![vec![alt(3, 0.6, 0, &[1, 2]), alt(3, 0.4, 0, &[1, 4])]],
            vec![(1, 0.5), (2, 0.3), (4, 0.2)],
            0,
            false,
        ),
        (
            "shared certain",
            3,
            vec![vec![alt(3, 0.7, 4, &[]), alt(3, 0.3, 4, &[])]],
            vec![],
            4,
            true,
        ),
        (
            "differing coset",
            2,
            vec![vec![alt(2, 0.7, 0, &[]), alt(2, 0.3, 2, &[1])]],
            vec![(2, 0.15), (3, 0.15)],
            0,
            false,
        ),
        (
            "delta in rank two span",
            2,
            vec![vec![alt(2, 0.7, 1, &[]), alt(2, 0.3, 0, &[1, 2])]],
            vec![(1, q3), (2, q3), (3, q3)],
            1,
            true,
        ),
        (
            "nonzero reference",
            3,
            vec![vec![alt(3, 0.9, 4, &[1]), alt(3, 0.1, 0, &[])]],
            vec![(1, 0.45), (4, 0.1)],
            4,
            false,
        ),
        (
            "independent groups",
            2,
            vec![
                vec![alt(2, 0.6, 0, &[]), alt(2, 0.4, 1, &[])],
                vec![alt(2, 0.6, 0, &[]), alt(2, 0.4, 2, &[])],
            ],
            vec![(1, 0.4), (2, 0.4)],
            0,
            true,
        ),
        (
            "rank two half",
            4,
            vec![vec![alt(4, 0.5, 0, &[1, 2]), alt(4, 0.5, 0, &[4, 8])]],
            vec![(1, q5), (2, q5), (3, q5), (4, q5), (8, q5), (12, q5)],
            0,
            false,
        ),
    ];
    for (name, width, groups, expected, expected_bias, is_exact) in fixtures {
        let shot = mix_shot_dem(&groups, &blank(width), 12).unwrap();
        let actual = mechanisms(&shot);
        assert_eq!(actual.len(), expected.len(), "{name}: {actual:?}");
        for (m, p) in expected {
            close(actual[&m], p);
        }
        assert_eq!(bias(&shot), expected_bias, "{name}");
        if is_exact {
            close(tv(&emitted(&shot, width), &exact(&groups, width)), 0.0);
        }
        if name == "differing coset" {
            let p = emitted(&shot, width);
            close(p[0] - p[1] - p[2] + p[3], 0.7);
        }
    }
}

#[test]
fn oracle6a_walsh_bounds() {
    for (name, width, groups) in [
        (
            "two covers",
            3,
            vec![vec![alt(3, 0.55, 0, &[1, 2]), alt(3, 0.45, 4, &[1])]],
        ),
        (
            "unequal differing constants",
            2,
            vec![vec![alt(2, 0.7, 0, &[]), alt(2, 0.3, 2, &[1])]],
        ),
        (
            "rank two counterexample",
            3,
            vec![vec![alt(3, 0.9, 0, &[1, 2]), alt(3, 0.1, 4, &[])]],
        ),
        (
            "independent bounds",
            3,
            vec![
                vec![alt(3, 0.7, 0, &[]), alt(3, 0.3, 2, &[1])],
                vec![alt(3, 0.6, 0, &[]), alt(3, 0.4, 4, &[])],
            ],
        ),
    ] {
        let measured = tv(
            &emitted(&mix_shot_dem(&groups, &blank(width), 12).unwrap(), width),
            &exact(&groups, width),
        );
        let limit = bound(&groups);
        assert!(measured <= limit + 1e-12, "{name}: {measured} > {limit}");
        if name == "rank two counterexample" {
            close(measured, 0.135);
            close(limit, 0.19);
        }
        eprintln!("oracle6a {name}: TV={measured:.15}, bound={limit:.15}");
    }
}

fn tag(index: usize) -> FalseAlarmTag {
    FalseAlarmTag {
        channel: index,
        location: Location {
            tick: index,
            batch: 0,
            operand: 0,
            side: Side::Before,
        },
    }
}
fn uniform_channel(width: usize, generators: [usize; 2], p: f64) -> Vec<f64> {
    let mut dem = blank(width).dem;
    for m in [generators[0], generators[1], generators[0] ^ generators[1]] {
        dem.add_direct_contribution(super::branches::effect(&bits(m), width).unwrap(), p);
    }
    emitted(
        &super::shot::normalise(&dem, &BitSet::new()).unwrap(),
        width,
    )
}
#[test]
fn oracle6b_false_alarm_mixture() {
    let width = 4;
    let mut reference = alt(width, 0.95, 0, &[]);
    reference.removals.insert(tag(0));
    let mut other = alt(width, 0.05, 0, &[]);
    other.removals.insert(tag(1));
    let groups = vec![vec![reference, other]];
    let mut background = blank(width);
    for (index, generators) in [[1, 2], [4, 8]].into_iter().enumerate() {
        background.removed.insert(
            tag(index),
            [generators[0], generators[1], generators[0] ^ generators[1]]
                .map(|m| {
                    FaultContribution::direct(
                        super::branches::effect(&bits(m), width).unwrap(),
                        0.5,
                    )
                })
                .to_vec(),
        );
    }
    let shot = mix_shot_dem(&groups, &background, 12).unwrap();
    let mut expected_background = vec![0.0; 1 << width];
    expected_background[0] = 1.0;
    let actual = mechanisms(&shot);
    for (keep, generators) in [(0.05f64, [1, 2]), (0.95f64, [4, 8])] {
        let p = (1.0 - (1.0 - keep).sqrt()) / 2.0;
        for m in [generators[0], generators[1], generators[0] ^ generators[1]] {
            close(actual[&m], p);
        }
        expected_background = xor(&expected_background, &uniform_channel(width, generators, p));
    }
    let b_ref = uniform_channel(width, [4, 8], 0.5);
    let b_other = uniform_channel(width, [1, 2], 0.5);
    let exact: Vec<_> = b_ref
        .iter()
        .zip(&b_other)
        .map(|(a, b)| 0.95 * a + 0.05 * b)
        .collect();
    let limit = bound(&groups)
        + 0.95 * tv(&b_ref, &expected_background)
        + 0.05 * tv(&b_other, &expected_background);
    let measured = tv(&emitted(&shot, width), &exact);
    assert!(measured <= limit + 1e-12);
    let mut point = vec![0.0; 16];
    point[0] = 1.0;
    let both = xor(&b_ref, &b_other);
    assert!(tv(&point, &exact) > 2.0 * limit);
    assert!(tv(&both, &exact) > 2.0 * limit);
    eprintln!(
        "oracle6b two FP=1 tags: TV={measured:.15}, bound={limit:.15}; skip={}, intersection={}",
        tv(&point, &exact),
        tv(&both, &exact)
    );
}

fn gate(c: &mut TickCircuit, kind: GateType, qs: &[usize]) {
    c.tick()
        .try_add_gate(Gate::simple(
            kind,
            qs.iter().copied().map(Into::into).collect::<Vec<_>>(),
        ))
        .unwrap();
}
fn probe(c: &mut TickCircuit, qs: &[usize]) {
    gate(c, GateType::DetectLost, qs);
}
fn model(loss: QubitLoss) -> GeneralNoiseModel {
    GeneralNoiseModel::builder().with_loss(loss).build()
}
fn limits() -> ShotLimits {
    ShotLimits {
        max_assignments: 1000,
        max_search_nodes: 1_000_000,
        max_correlated_rank: 12,
    }
}
fn build(c: &TickCircuit, noise: &GeneralNoiseModel) -> LossCatalog {
    LossCatalog::build(c, noise, 0.0, 12, 10000).unwrap()
}
fn flags(values: &[usize]) -> BTreeSet<ProbeId> {
    values.iter().map(|&p| ProbeId(p)).collect()
}

#[test]
fn oracle2_assignments() {
    // Two separated draws and a recovery: one persistent loss or two losses.
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    c.tick().x(&[0]);
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    let r = c.tick().mz(&[0]);
    c.detector(&r).unwrap();
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.2)
            .add_recovery_channel(LossSite::BeforeSingleQubit, 0.3, 0.0)
            .build()
            .unwrap(),
    );
    let catalog = build(&c, &noise);
    let shot = ShotDemBuilder::new(&catalog)
        .unwrap()
        .build(&flags(&[0, 1]), limits())
        .unwrap();
    let assignments: Vec<_> = shot.assignments.iter().flatten().collect();
    assert_eq!(assignments.len(), 2);
    let one = assignments.iter().find(|a| a.branches.len() == 1).unwrap();
    let two = assignments.iter().find(|a| a.branches.len() == 2).unwrap();
    close(one.weight, 0.2 * 0.3 / 0.8);
    close(two.weight, (0.2 * 0.7 / 0.8) * (0.2 / 0.8));
    close(one.pi, 0.075 / 0.11875);
    close(two.pi, 0.04375 / 0.11875);
    // One categorical draw's LI and IL labels must never be paired.
    let mut c = TickCircuit::new();
    c.tick().cx(&[(0, 1)]);
    probe(&mut c, &[0, 1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .add_events_channel(
                LossSite::BeforeTwoQubit,
                0.3,
                &BTreeMap::from([("LI".into(), 0.5), ("IL".into(), 0.5)]),
            )
            .build()
            .unwrap(),
    );
    let catalog = build(&c, &noise);
    let builder = ShotDemBuilder::new(&catalog).unwrap();
    assert!(matches!(
        builder.assignments(&flags(&[0, 1]), 100, 1_000_000),
        Err(AssignmentError::NoAssignment { .. })
    ));
    assert!(matches!(
        builder.assignments(&flags(&[2]), 100, 1_000_000),
        Err(AssignmentError::InvalidProbe { .. })
    ));
    assert_eq!(
        builder.assignments(&BTreeSet::new(), 1, 1_000_000).unwrap(),
        vec![vec![vec![]]]
    );
    // A spread is a single indivisible part.
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    c.tick().cx(&[(0, 1)]);
    probe(&mut c, &[0, 1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.2)
            .with_lost_partner_loss_probability(1.0)
            .build()
            .unwrap(),
    );
    let catalog = build(&c, &noise);
    let builder = ShotDemBuilder::new(&catalog).unwrap();
    assert_eq!(
        builder.assignments(&flags(&[0, 1]), 1, 1_000_000).unwrap()[0][0].len(),
        1
    );
    assert!(matches!(
        builder.assignments(&flags(&[0]), 10, 1_000_000),
        Err(AssignmentError::NoAssignment { .. })
    ));
}

#[test]
fn assignments_factor_independence_and_limits() {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0, 1]);
    probe(&mut c, &[0, 1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.2)
            .build()
            .unwrap(),
    );
    let mut catalog = build(&c, &noise);
    // Multiple trajectories in each independent component; Cartesian product is 4,
    // but a cap of 2 must succeed. Rebuild the public index after adding leaves.
    for (i, root) in catalog.roots.iter_mut().enumerate() {
        root.branches.push(root.branches[0].clone());
        catalog
            .by_flags
            .get_mut(&root.branches[0].flags)
            .unwrap()
            .push(BranchId { root: i, branch: 1 });
    }
    let builder = ShotDemBuilder::new(&catalog).unwrap();
    assert_eq!(
        builder
            .assignments(&flags(&[0, 1]), 2, 1_000_000)
            .unwrap()
            .iter()
            .map(Vec::len)
            .collect::<Vec<_>>(),
        vec![2, 2]
    );
    let error = builder
        .assignments(&flags(&[0, 1]), 1, 1_000_000)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("at least max + 1")
            && error.contains("ProbeId(0)")
            && !error.contains("ProbeId(1)")
    );
    catalog.report.dropped_mass_by_flags = BTreeMap::from([
        (BTreeSet::new(), 0.01),
        (flags(&[0]), 0.02),
        (flags(&[1]), 0.04),
    ]);
    close(
        ShotDemBuilder::new(&catalog)
            .unwrap()
            .build(&flags(&[0]), limits())
            .unwrap()
            .pruned_mass_upper_bound,
        0.03,
    );
}

#[test]
fn oracle4_backends_and_normalisation() {
    let specs = [
        DecoderSpec::PyMatching(pecos_decoders::spec::PyMatchingConfig::default()),
        DecoderSpec::FusionBlossom(pecos_decoders::spec::FusionBlossomConfig::default()),
        DecoderSpec::PerturbedFusionBlossomCorrelated(
            pecos_decoders::spec::PerturbedFusionBlossomConfig::default(),
        ),
        DecoderSpec::Tesseract(pecos_decoders::spec::TesseractConfig::default()),
        DecoderSpec::TesseractTrellis(pecos_decoders::spec::TesseractTrellisConfig::default()),
        DecoderSpec::KMwpm(pecos_decoders::spec::KMwpmConfig::default()),
        DecoderSpec::BpOsd(pecos_decoders::spec::BpOsdConfig::default()),
        DecoderSpec::BpLsd(pecos_decoders::spec::BpLsdConfig::default()),
        DecoderSpec::BeliefFind,
        DecoderSpec::UnionFind,
    ];
    for p in [1.0, 0.7, 0.5] {
        let mut bg = blank(2);
        bg.dem.add_observable(DemOutput::new(0));
        bg.dem
            .add_direct_contribution(FaultMechanism::from_unsorted(vec![0], vec![0]), p);
        // D0 is touched only by the folded mechanism at p=1. D1 retains an
        // ordinary edge so backends requiring a nonempty reduced model can build.
        bg.dem
            .add_direct_contribution(FaultMechanism::from_unsorted(vec![1], vec![]), 0.01);
        let shot = mix_shot_dem(&[], &bg, 12).unwrap();
        assert!(
            shot.dem
                .contributions()
                .iter()
                .all(|c| c.probability <= 0.5)
        );
        assert_eq!(shot.detector_bias, vec![u8::from(p > 0.5), 0]);
        assert_eq!(shot.observable_bias.get(0), p > 0.5);
        let text = shot.dem.to_string();
        assert!(text.contains("detector D0") && text.contains("logical_observable L0"));
        for spec in &specs {
            let mut decoder =
                BiasedDecoder::build(&shot, spec).unwrap_or_else(|e| panic!("{spec:?} p={p}: {e}"));
            assert_eq!(decoder.num_detectors(), Some(2));
            let prediction = decoder
                .decode_obs(&shot.detector_bias)
                .unwrap_or_else(|e| panic!("{spec:?} p={p}: {e}"));
            assert_eq!(prediction, shot.observable_bias, "{spec:?}, p={p}");
        }
    }
    // The requested 0.7/0.3 alternatives with differing constants, and a wide
    // observable, exercise the public pure function and bias wrapper together.
    let mut bg = blank(2);
    bg.dem.add_observable(DemOutput::new(64));
    bg.dem
        .add_direct_contribution(FaultMechanism::from_unsorted(vec![1], vec![]), 0.01);
    let mut a = alt(67, 0.7, 0, &[]);
    a.normal_form.constant = [0, 66].into_iter().collect();
    let b = alt(67, 0.3, 0, &[]);
    let shot = mix_shot_dem(&[vec![a, b]], &bg, 12).unwrap();
    assert!(shot.observable_bias.get(64));
    let mut decoder = BiasedDecoder::build(
        &shot,
        &DecoderSpec::BpOsd(pecos_decoders::spec::BpOsdConfig::default()),
    )
    .unwrap();
    assert_eq!(decoder.decode_obs(&[1, 0]).unwrap(), shot.observable_bias);
}

#[test]
fn oracle5_errors_count_as_failures_and_cache() {
    use pecos_engines::noise::general::ReadoutMatrix;
    let mut c = TickCircuit::new();
    probe(&mut c, &[0]);
    let r = c.tick().mz(&[0]);
    c.detector(&r).unwrap();
    c.observable(&r).unwrap();
    let readout = ReadoutMatrix::new(
        GateType::DetectLost,
        &BTreeMap::from([("present".into(), BTreeMap::from([("3".into(), 1.0)]))]),
    )
    .unwrap();
    let noise = GeneralNoiseModel::builder().with_readout(readout).build();
    let catalog = build(&c, &noise);
    let error = ShotDemBuilder::new(&catalog)
        .unwrap()
        .build(&flags(&[0]), limits())
        .unwrap_err();
    assert!(error.to_string().contains("ProbeId(0)"));
    let mut runner = TickRunner::new(&c, noise, 311).unwrap();
    let report = LossDecoder::run(
        &catalog,
        &mut runner,
        &DecoderSpec::PyMatching(pecos_decoders::spec::PyMatchingConfig::default()),
        10,
        limits(),
    )
    .unwrap();
    assert_eq!(report.loss_aware.shots, 10);
    assert_eq!(report.loss_aware.logical_errors, 10);
    assert_eq!(
        report.loss_aware.errored_shots[&ErroredShotReason::NoAssignment],
        10
    );
    assert_eq!(report.loss_aware.cache_hits, 9);
    assert_eq!(report.loss_blind.shots, 10);
    assert_eq!(report.loss_blind.cache_hits, 9);
    let mut other = TickCircuit::new();
    other.tick().mz(&[1]);
    let mut runner = TickRunner::new(&other, GeneralNoiseModel::default(), 311).unwrap();
    assert!(
        LossDecoder::run(&catalog, &mut runner, &DecoderSpec::AStarFull, 0, limits())
            .unwrap_err()
            .to_string()
            .contains("catalog.outputs")
    );
}

fn engine_conditional(
    name: &str,
    c: &TickCircuit,
    noise: GeneralNoiseModel,
    check_frequency: bool,
) {
    let description = noise.description();
    assert!(
        [
            description.p1,
            description.p2,
            description.p_prep,
            description.p_meas_0,
            description.p_meas_1
        ]
        .iter()
        .all(|&p| p == 0.0)
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
        10000,
        super::branch_replay::ReplayOptions {
            convention_coins: false,
            ..super::branch_replay::ReplayOptions::default()
        },
    )
    .unwrap();
    let width = catalog.num_detectors + catalog.num_observables;
    let cells = 1 << width;
    let mut baseline = TickRunner::new(c, GeneralNoiseModel::default(), 824_109).unwrap();
    let baseline = baseline.shot().unwrap();
    let baseline_mask = baseline
        .detectors
        .iter()
        .enumerate()
        .fold(0, |m, (i, &v)| m ^ (usize::from(v) << i))
        ^ baseline
            .observables
            .iter_set_bits()
            .fold(0, |m, i| m ^ (1 << (catalog.num_detectors + i)));
    let mut runner = TickRunner::new(c, noise, 824_109).unwrap();
    let n = 100_000usize;
    let mut counts: BTreeMap<BTreeSet<ProbeId>, Vec<usize>> = BTreeMap::new();
    for _ in 0..n {
        let s = runner.shot().unwrap();
        let f = s
            .flags
            .iter()
            .enumerate()
            .filter_map(|(i, &v)| v.then_some(ProbeId(i)))
            .collect();
        let value = s
            .detectors
            .iter()
            .enumerate()
            .fold(0, |m, (i, &v)| m ^ (usize::from(v) << i))
            ^ s.observables
                .iter_set_bits()
                .fold(0, |m, i| m ^ (1 << (catalog.num_detectors + i)));
        counts.entry(f).or_insert_with(|| vec![0; cells])[value ^ baseline_mask] += 1;
    }
    let builder = ShotDemBuilder::new(&catalog).unwrap();
    for (f, observed) in &counts {
        let expected = if f.is_empty() && check_frequency {
            // Exact conditional reference, before the stated unflagged-background
            // approximation: include no-loss mass and all unflagged trajectories.
            let mut distribution = vec![0.0; cells];
            distribution[0] = 1.0 - catalog.roots.iter().map(|r| r.prior).sum::<f64>();
            for branch in catalog
                .roots
                .iter()
                .flat_map(|r| &r.branches)
                .filter(|b| b.flags.is_empty())
            {
                let alternative = ShotAlternative {
                    branches: Vec::new(),
                    weight: branch.weight,
                    pi: 1.0,
                    normal_form: branch.normal_form.clone(),
                    removals: BTreeSet::new(),
                };
                for (cell, probability) in distribution
                    .iter_mut()
                    .zip(exact(&[vec![alternative]], width))
                {
                    *cell += branch.weight * probability;
                }
            }
            let total: f64 = distribution.iter().sum();
            for probability in &mut distribution {
                *probability /= total;
            }
            distribution
        } else {
            let shot = builder.build(f, limits()).unwrap();
            exact(&shot.assignments, width)
        };
        let total: usize = observed.iter().sum();
        for (&count, p) in observed.iter().zip(expected) {
            let interval = pecos_num::stats::jeffreys_interval(
                count as u64,
                total as u64,
                0.001 / f64::from(u32::try_from(cells * counts.len()).unwrap()),
            )
            .unwrap();
            assert!(
                p >= interval.lo - 1e-12 && p <= interval.hi + 1e-12,
                "{name} flags={f:?}, p={p}, count={count}/{total}, {interval:?}"
            );
        }
    }
    if check_frequency {
        let mut probabilities = BTreeMap::from([(BTreeSet::new(), 1.0)]);
        for root in &catalog.roots {
            *probabilities.get_mut(&BTreeSet::new()).unwrap() -= root.prior;
            for branch in &root.branches {
                *probabilities.entry(branch.flags.clone()).or_default() += branch.weight;
            }
        }
        for (f, p) in probabilities {
            let count = counts.get(&f).map_or(0, |v| v.iter().sum());
            let interval =
                pecos_num::stats::jeffreys_interval(count as u64, n as u64, 0.001 / 16.0).unwrap();
            assert!(
                p >= interval.lo - 1e-12 && p <= interval.hi + 1e-12,
                "frequency {name} {f:?}: {p} {count}/{n} {interval:?}"
            );
        }
    }
    eprintln!(
        "oracle3 {name}: N={n}, seed=824109, conditional passed, frequencies={check_frequency}"
    );
}

#[test]
fn oracle3_engine_conditionals() {
    let mut c = TickCircuit::new();
    c.tick().h(&[1]);
    c.tick().cx(&[(0, 1)]);
    probe(&mut c, &[0, 1]);
    c.tick().h(&[1]);
    for r in c.tick().mz(&[0, 1]) {
        c.detector(&[r]).unwrap();
    }
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .add_events_channel(
                LossSite::BeforeTwoQubit,
                0.37,
                &BTreeMap::from([("LI".into(), 0.8), ("LZ".into(), 0.2)]),
            )
            .build()
            .unwrap(),
    );
    engine_conditional("unequal LI/LZ visible partner Z", &c, noise, true);
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    c.tick().x(&[0]);
    probe(&mut c, &[0]);
    let r = c.tick().mz(&[0]);
    c.detector(&r).unwrap();
    c.observable(&r).unwrap();
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.37)
            .add_recovery_channel(LossSite::BeforeSingleQubit, 0.3, 0.0)
            .build()
            .unwrap(),
    );
    engine_conditional("one site recovery FN", &c, noise, true);
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0, 1]);
    probe(&mut c, &[0, 1]);
    for r in c.tick().mz(&[0, 1]) {
        c.detector(&[r]).unwrap();
    }
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.37)
            .build()
            .unwrap(),
    );
    engine_conditional("two independent sites", &c, noise, false);
}

#[test]
fn oracle7_surface_flags_help() {
    let start = std::time::Instant::now();
    let z_checks: [&[usize]; 4] = [&[0, 1], &[1, 2, 4, 5], &[3, 4, 6, 7], &[7, 8]];
    let x_checks: [&[usize]; 4] = [&[0, 1, 3, 4], &[2, 5], &[3, 6], &[4, 5, 7, 8]];
    let mut c = TickCircuit::new();
    c.tick().pz(&(0..9).collect::<Vec<_>>());
    let mut rounds: Vec<Vec<pecos_quantum::TickMeasRef>> = Vec::new();
    for round in 0..4 {
        c.tick().pz(&(9..17).collect::<Vec<_>>());
        // One explicit loss opportunity per qubit, after a reference
        // round. Ancilla reloads on the next PZ; data are never re-prepared.
        if round == 1 {
            c.tick().idle(1, &(0..17).collect::<Vec<_>>());
        }
        let mut results = Vec::new();
        for (check, support) in z_checks.iter().chain(&x_checks).enumerate() {
            let ancilla = 9 + check;
            if check >= 4 {
                c.tick().h(&[ancilla]);
            }
            for &q in *support {
                c.tick().cx(&[if check >= 4 {
                    (ancilla, q)
                } else {
                    (q, ancilla)
                }]);
            }
            if check >= 4 {
                c.tick().h(&[ancilla]);
            }
            probe(&mut c, &[ancilla]);
            results.push(c.tick().mz(&[ancilla])[0]);
        }
        for check in 0..8 {
            if round > 0 {
                c.detector(&[rounds[round - 1][check], results[check]])
                    .unwrap();
            } else if check < 4 {
                c.detector(&[results[check]]).unwrap();
            }
        }
        rounds.push(results);
    }
    probe(&mut c, &(0..9).collect::<Vec<_>>());
    let data = c.tick().mz(&(0..9).collect::<Vec<_>>());
    for (check, support) in z_checks.iter().enumerate() {
        let mut parity = vec![rounds[3][check]];
        parity.extend(support.iter().map(|&q| data[q]));
        c.detector(&parity).unwrap();
    }
    c.observable(&[data[0], data[3], data[6]]).unwrap();
    let loss = QubitLoss::builder()
        .with_default_recovery(false)
        .with_reload_on_prep(true)
        .with_lost_partner_loss_probability(0.0)
        .with_idle_rate(0.08)
        .build()
        .unwrap();
    let noise = GeneralNoiseModel::builder()
        .with_loss(loss)
        .with_p1(0.0001)
        .with_p2(0.0001)
        .build();
    let catalog = LossCatalog::build(&c, &noise, 0.0, 12, 10000).unwrap();
    let mut runner = TickRunner::new(&c, noise, 923_471).unwrap();
    let n = 2000;
    let report = LossDecoder::run(
        &catalog,
        &mut runner,
        &DecoderSpec::BpOsd(pecos_decoders::spec::BpOsdConfig::default()),
        n,
        ShotLimits {
            max_assignments: 1000,
            max_search_nodes: 1_000_000,
            max_correlated_rank: 12,
        },
    )
    .unwrap();
    let aware = &report.loss_aware;
    let blind = &report.loss_blind;
    let ai =
        pecos_num::stats::jeffreys_interval(aware.logical_errors as u64, n as u64, 0.01).unwrap();
    let bi =
        pecos_num::stats::jeffreys_interval(blind.logical_errors as u64, n as u64, 0.01).unwrap();
    eprintln!(
        "oracle7 d=3 Z memory: N={n}, seed=923471, loss idle=0.08 at 17 round-1 sites, p1=p2=0.0001, prep=measurement=spread=0, partner_paulis=none, default_recovery=false, reload_on_prep=true (ancillas only re-prepared), cutoff=0, max_branches=10000, rank=12, assignments=1000, search_nodes=1000000; aware={}/{n} {ai:?}, errors={:?}; blind={}/{n} {bi:?}, errors={:?}; runtime={:?}",
        aware.logical_errors,
        aware.errored_shots,
        blind.logical_errors,
        blind.errored_shots,
        start.elapsed()
    );
    assert!(
        ai.hi + 0.02 < bi.lo,
        "flags must help with disjoint 99% Jeffreys intervals and a two-percentage-point margin"
    );
    assert!(aware.errored_shots.is_empty());
    assert!(blind.errored_shots.is_empty());
}

#[test]
fn cover_common_components_and_same_root_exclusion() {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0, 1, 2]);
    probe(&mut c, &[0, 1, 2]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.2)
            .build()
            .unwrap(),
    );
    let mut catalog = build(&c, &noise);
    let template = catalog.roots[0].clone();
    catalog.roots.clear();
    catalog.by_flags.clear();
    for (index, (probes, weight)) in [
        (vec![0, 1], 0.4),
        (vec![0], 0.3),
        (vec![1], 0.2),
        (vec![2], 0.6),
        (vec![2], 0.4),
    ]
    .into_iter()
    .enumerate()
    {
        let mut root = template.clone();
        root.key.location.operand = index;
        root.branches[0].flags = flags(&probes);
        root.branches[0].weight = weight;
        catalog
            .by_flags
            .entry(flags(&probes))
            .or_default()
            .push(BranchId {
                root: index,
                branch: 0,
            });
        catalog.roots.push(root);
    }
    let shot = ShotDemBuilder::new(&catalog)
        .unwrap()
        .build(
            &flags(&[0, 1, 2]),
            ShotLimits {
                max_assignments: 2,
                max_search_nodes: 1_000_000,
                max_correlated_rank: 12,
            },
        )
        .unwrap();
    assert_eq!(shot.assignments.len(), 2);
    close(
        shot.assignments[0][0].pi,
        (0.3 / 0.7 * 0.2 / 0.8) / (0.4 / 0.6 + 0.3 / 0.7 * 0.2 / 0.8),
    );
    close(
        shot.assignments[0][1].pi,
        (0.4 / 0.6) / (0.4 / 0.6 + 0.3 / 0.7 * 0.2 / 0.8),
    );
    close(
        shot.assignments[1][0].pi,
        (0.6 / 0.4) / (0.6 / 0.4 + 0.4 / 0.6),
    );
    close(
        shot.assignments[1][1].pi,
        (0.4 / 0.6) / (0.6 / 0.4 + 0.4 / 0.6),
    );
    // Two disjoint trajectories belonging to the very same root draw.
    let mut root = template;
    root.branches[0].flags = flags(&[0]);
    root.branches.push(root.branches[0].clone());
    root.branches[1].flags = flags(&[1]);
    catalog.roots = vec![root];
    catalog.by_flags = BTreeMap::from([
        (flags(&[0]), vec![BranchId { root: 0, branch: 0 }]),
        (flags(&[1]), vec![BranchId { root: 0, branch: 1 }]),
    ]);
    assert!(matches!(
        ShotDemBuilder::new(&catalog)
            .unwrap()
            .assignments(&flags(&[0, 1]), 100, 1_000_000),
        Err(AssignmentError::NoAssignment { .. })
    ));
}

#[test]
fn decoder_build_and_assignment_limit_errors_count() {
    let mut c = TickCircuit::new();
    for r in c.tick().mz(&[0, 1, 2]) {
        c.detector(&[r]).unwrap();
    }
    let pauli_model = GeneralNoiseModel::builder().with_p_meas(0.01).build();
    let mut catalog = build(&c, &pauli_model);
    catalog
        .background
        .add_direct_contribution(FaultMechanism::from_unsorted(vec![0, 1, 2], vec![]), 0.1);
    // loss_blind_dem uses background; aware reconstructs base. Make both routes
    // reject by selecting a backend spec requiring a HybridDem instead.
    let spec = DecoderSpec::BeliefMatching(pecos_decoders::spec::BeliefMatchingConfig {
        mode: pecos_decoders::spec::BeliefMatchingMode::Hybrid,
        ..pecos_decoders::spec::BeliefMatchingConfig::default()
    });
    let mut runner = TickRunner::new(&c, GeneralNoiseModel::default(), 7).unwrap();
    let report = LossDecoder::run(&catalog, &mut runner, &spec, 3, limits()).unwrap();
    for counts in [&report.loss_aware, &report.loss_blind] {
        assert_eq!(counts.logical_errors, 3);
        assert_eq!(counts.errored_shots[&ErroredShotReason::DecoderBuild], 3);
        assert_eq!(counts.cache_hits, 2);
    }
    // Also check the caller's Refuse policy on a real irreducible hyperedge.
    let mut hyperedge = blank(3).dem;
    hyperedge.add_direct_contribution(FaultMechanism::from_unsorted(vec![0, 1, 2], vec![]), 0.1);
    let shot = super::shot::normalise(&hyperedge, &BitSet::new()).unwrap();
    assert!(matches!(
        BiasedDecoder::build(
            &shot,
            &DecoderSpec::PyMatching(pecos_decoders::spec::PyMatchingConfig::default())
        ),
        Err(ShotError::DecoderBuild(_))
    ));
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    c.tick().mz(&[0]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.5)
            .build()
            .unwrap(),
    );
    let catalog = build(&c, &noise);
    let mut runner = TickRunner::new(&c, noise, 7).unwrap();
    let report = LossDecoder::run(
        &catalog,
        &mut runner,
        &DecoderSpec::PyMatching(pecos_decoders::spec::PyMatchingConfig::default()),
        3,
        ShotLimits {
            max_assignments: 0,
            max_search_nodes: 1_000_000,
            max_correlated_rank: 12,
        },
    )
    .unwrap();
    assert_eq!(report.loss_aware.logical_errors, 3);
    assert_eq!(
        report.loss_aware.errored_shots[&ErroredShotReason::TooManyAssignments],
        3
    );
}

#[test]
fn false_alarm_groups_multiply_keep_mass_and_preserve_metadata() {
    let mut c = TickCircuit::new();
    c.tick().h(&[0]);
    let r = c.tick().mz(&[0]);
    // Use a deterministic lossless circuit while leaving recovery X/Y visible.
    c.tick().pz(&[0]);
    c.tick().x(&[0]);
    let r2 = c.tick().mz(&[0]);
    c.detector(&r2).unwrap();
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .add_recovery_channel(LossSite::BeforeSingleQubit, 0.0, 1.0)
            .build()
            .unwrap(),
    );
    let catalog = build(&c, &noise);
    assert_eq!(r.len(), 1);
    let (&tag, sources) = catalog
        .false_alarm_tags()
        .iter()
        .find(|(_, v)| !v.is_empty())
        .unwrap();
    let mut removed = alt(1, 0.6, 0, &[]);
    removed.removals.insert(tag);
    let kept = alt(1, 0.4, 0, &[]);
    let groups = vec![vec![removed.clone(), kept.clone()], vec![removed, kept]];
    let background = catalog.shot_background(&BTreeSet::from([tag])).unwrap();
    let shot = mix_shot_dem(&groups, &background, 12).unwrap();
    let expected = (1.0 - (1.0 - 0.16f64).sqrt()) / 2.0;
    let start = background.dem.num_contributions();
    for (actual, original) in shot.dem.contributions()[start..].iter().zip(sources) {
        close(actual.probability, expected);
        let mut restored = actual.clone();
        restored.probability = original.probability;
        super::branch_tests::assert_same_sources(&[restored], std::slice::from_ref(original));
    }
    assert!(
        shot.approximations
            .contains(&Approximation::FalseAlarmMixture)
    );
}

#[test]
fn sparse_and_implicit_dimensions_survive_folding() {
    // No declarations: a folded source is the only mention of D7 and L64.
    let mut dem = DetectorErrorModel::new();
    dem.add_direct_contribution(FaultMechanism::from_unsorted(vec![7], vec![64]), 1.0);
    dem.add_direct_contribution(FaultMechanism::from_unsorted(vec![0], vec![]), 0.1);
    let shot = super::shot::normalise(&dem, &BitSet::new()).unwrap();
    assert_eq!(shot.detector_bias.len(), 8);
    assert_eq!(shot.detector_bias[7], 1);
    assert!(shot.observable_bias.get(64));
    assert_eq!(
        pecos_decoder_core::dem::utils::parse_dem_metadata(&shot.dem.to_string()).unwrap(),
        (8, 65)
    );
    let mut decoder = BiasedDecoder::build(
        &shot,
        &DecoderSpec::BpOsd(pecos_decoders::spec::BpOsdConfig::default()),
    )
    .unwrap();
    assert_eq!(
        decoder.decode_obs(&shot.detector_bias).unwrap(),
        shot.observable_bias
    );
    // A real catalog with sparse metadata IDs must also mix at its ambient width.
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    c.tick().mz(&[0]);
    c.add_detector_metadata(&[-1], None, None, Some(7), None)
        .unwrap();
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.5)
            .build()
            .unwrap(),
    );
    let catalog = build(&c, &noise);
    assert_eq!(catalog.num_detectors, 8);
    let shot = ShotDemBuilder::new(&catalog)
        .unwrap()
        .build(&flags(&[0]), limits())
        .unwrap();
    assert_eq!(shot.detector_bias.len(), 8);
    let mut runner = TickRunner::new(&c, noise, 7).unwrap();
    let report = LossDecoder::run(
        &catalog,
        &mut runner,
        &DecoderSpec::PyMatching(pecos_decoders::spec::PyMatchingConfig::default()),
        10,
        limits(),
    )
    .unwrap();
    assert!(report.loss_aware.errored_shots.is_empty());
}

#[test]
fn r1_reference_selection_and_three_way_intersection() {
    let q = (1.0 - 0.5f64.sqrt()) / 2.0;
    for (name, groups, expected, expected_bias) in [
        (
            "heaviest last",
            vec![vec![alt(3, 0.3, 1, &[2]), alt(3, 0.7, 4, &[])]],
            vec![(5, 0.15), (7, 0.15)],
            4,
        ),
        (
            "lowest-index tie",
            vec![vec![alt(3, 0.5, 1, &[]), alt(3, 0.5, 2, &[])]],
            vec![(3, 0.5)],
            1,
        ),
        // First pair intersects in <1>; the third span excludes 1. The full
        // intersection is zero. Signatures 1,2,4 merge independent contributions.
        (
            "three-way intersection",
            vec![vec![
                alt(3, 0.5, 0, &[1, 2]),
                alt(3, 0.3, 0, &[1, 4]),
                alt(3, 0.2, 0, &[2, 4]),
            ]],
            vec![
                (1, (1.0 - (0.5f64 * 0.7).sqrt()) / 2.0),
                (2, (1.0 - (0.5f64 * 0.8).sqrt()) / 2.0),
                (3, q),
                (4, (1.0 - (0.7f64 * 0.8).sqrt()) / 2.0),
                (5, (1.0 - 0.7f64.sqrt()) / 2.0),
                (6, (1.0 - 0.8f64.sqrt()) / 2.0),
            ],
            0,
        ),
    ] {
        let shot = mix_shot_dem(&groups, &blank(3), 12).unwrap();
        let actual = mechanisms(&shot);
        assert_eq!(actual.len(), expected.len(), "{name}: {actual:?}");
        for (signature, p) in expected {
            close(actual[&signature], p);
        }
        assert_eq!(bias(&shot), expected_bias, "{name}");
    }
}

#[test]
fn r1_false_alarm_nonunit_fp() {
    let mut removed = alt(2, 0.7, 0, &[]);
    removed.removals.insert(tag(0));
    let groups = vec![vec![removed, alt(2, 0.3, 0, &[])]];
    let p = (1.0 - 0.7f64.sqrt()) / 2.0; // FP=0.3, not FP=1.
    let mut background = blank(2);
    background.removed.insert(
        tag(0),
        [1, 2, 3]
            .map(|m| FaultContribution::direct(super::branches::effect(&bits(m), 2).unwrap(), p))
            .to_vec(),
    );
    let shot = mix_shot_dem(&groups, &background, 12).unwrap();
    let expected = (1.0 - (1.0 - 0.3f64 * 0.3).sqrt()) / 2.0;
    assert_eq!(mechanisms(&shot).len(), 3);
    for probability in mechanisms(&shot).values() {
        close(*probability, expected);
    }
}

#[test]
fn r1_blocks_root_while_operand_lost() {
    // Review's exact ordering: B occurs BEFORE recovery, so an A trajectory
    // cannot coexist with B even if A is recovered later.
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    c.tick().idle(1, &[0]);
    c.tick().x(&[0]);
    probe(&mut c, &[0]);
    let r = c.tick().mz(&[0]);
    c.detector(&r).unwrap();
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.2)
            .add_recovery_channel(LossSite::BeforeSingleQubit, 0.3, 0.0)
            .build()
            .unwrap(),
    );
    let catalog = build(&c, &noise);
    let shot = ShotDemBuilder::new(&catalog)
        .unwrap()
        .build(&flags(&[0, 1]), limits())
        .unwrap();
    assert_eq!(shot.assignments.len(), 1);
    assert_eq!(shot.assignments[0].len(), 1);
    let assignment = &shot.assignments[0][0];
    assert_eq!(assignment.branches, vec![BranchId { root: 0, branch: 0 }]);
    close(assignment.pi, 1.0);
    close(assignment.weight, 0.2 * 0.3 / 0.8);
}

fn ambiguous_regions(n: usize) -> LossCatalog {
    let mut c = TickCircuit::new();
    let qs = (0..n).collect::<Vec<_>>();
    c.tick().idle(1, &qs);
    probe(&mut c, &qs);
    c.tick().x(&qs);
    c.tick().idle(1, &qs);
    probe(&mut c, &qs);
    for r in c.tick().mz(&qs) {
        c.detector(&[r]).unwrap();
    }
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.2)
            .add_recovery_channel(LossSite::BeforeSingleQubit, 0.3, 0.0)
            .build()
            .unwrap(),
    );
    build(&c, &noise)
}

#[test]
fn r1_twelve_independent_ambiguous_regions() {
    let catalog = ambiguous_regions(12);
    super::shot::COVER_VISITS.with(|c| c.set(0));
    let shot = ShotDemBuilder::new(&catalog)
        .unwrap()
        .build(
            &flags(&(0..24).collect::<Vec<_>>()),
            ShotLimits {
                max_assignments: 4,
                max_search_nodes: 1_000_000,
                max_correlated_rank: 12,
            },
        )
        .unwrap();
    assert_eq!(shot.assignments.len(), 12);
    assert_eq!(super::shot::COVER_VISITS.with(std::cell::Cell::get), 24);
    for group in &shot.assignments {
        assert_eq!(group.len(), 2);
        close(
            group.iter().find(|a| a.branches.len() == 1).unwrap().pi,
            0.075 / 0.11875,
        );
        close(
            group.iter().find(|a| a.branches.len() == 2).unwrap().pi,
            0.04375 / 0.11875,
        );
    }
    // Every alternative leaves that qubit's final lost record uniformly random.
    // Thus the exact product is uniform on all 12 output bits, and the emitted
    // list is exactly twelve independent half-probability singleton mechanisms.
    let emitted = mechanisms(&shot);
    assert_eq!(emitted.len(), 12);
    assert_eq!(bias(&shot), 0);
    for i in 0..12 {
        close(emitted[&(1 << i)], 0.5);
    }
}

#[test]
fn r1_cover_cap_stops_enumeration() {
    // All 2^9-1 nonempty parts produce Bell(9)=21147 covers in one region.
    // Each part has one distinct draw and all covers are feasible.
    let mut catalog = ambiguous_regions(9);
    let template = catalog.roots[0].clone();
    catalog.roots.clear();
    catalog.by_flags.clear();
    for subset in 1usize..1 << 9 {
        let mut root = template.clone();
        root.key.location.tick = subset;
        root.qubits = vec![subset];
        root.branches[0].lost_over_time.clear();
        root.branches[0].flags = flags(
            &(0..9)
                .filter(|i| subset & (1 << i) != 0)
                .collect::<Vec<_>>(),
        );
        root.branches[0].weight = 0.001;
        let id = BranchId {
            root: catalog.roots.len(),
            branch: 0,
        };
        catalog
            .by_flags
            .insert(root.branches[0].flags.clone(), vec![id]);
        catalog.roots.push(root);
    }
    super::shot::COVER_VISITS.with(|c| c.set(0));
    let error = ShotDemBuilder::new(&catalog)
        .unwrap()
        .assignments(&flags(&(0..9).collect::<Vec<_>>()), 4, 1_000_000)
        .unwrap_err();
    assert!(matches!(
        error,
        AssignmentError::TooManyAssignments { max: 4, .. }
    ));
    assert_eq!(super::shot::COVER_VISITS.with(std::cell::Cell::get), 5);
    assert!(error.to_string().contains("ProbeId(8)"));
}

#[test]
fn r1_empty_models_use_bias_without_factory() {
    use pecos_decoders::DecodeModel;
    let spec = DecoderSpec::BpOsd(pecos_decoders::spec::BpOsdConfig::default());
    for decoder in [
        &spec,
        &DecoderSpec::BpLsd(pecos_decoders::spec::BpLsdConfig::default()),
        &DecoderSpec::BeliefFind,
    ] {
        let error = decoder
            .build(&DecodeModel::SingleDem(
                "detector D0\nlogical_observable L0".into(),
            ))
            .err()
            .unwrap()
            .to_string();
        assert!(error.contains("no uncertain error mechanism"), "{error}");
    }
    let mut c = TickCircuit::new();
    c.tick().x(&[0, 1]);
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    c.tick().cx(&[(0, 1)]);
    let r = c.tick().mz(&[1]);
    c.detector(&r).unwrap();
    c.observable(&r).unwrap();
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.2)
            .with_lost_partner_loss_probability(0.0)
            .build()
            .unwrap(),
    );
    let catalog = build(&c, &noise);
    let shot = ShotDemBuilder::new(&catalog)
        .unwrap()
        .build(&flags(&[0]), limits())
        .unwrap();
    assert!(shot.dem.contributions().is_empty());
    assert_eq!(shot.detector_bias, vec![1]);
    assert!(shot.observable_bias.get(0));
    let mut decoder = BiasedDecoder::build(&shot, &spec).unwrap();
    assert_eq!(decoder.decode_obs(&[1]).unwrap(), shot.observable_bias);
    assert_eq!(decoder.decode_obs(&[0]).unwrap(), shot.observable_bias);
    let mut runner = TickRunner::new(&c, noise, 82341).unwrap();
    let report = LossDecoder::run(&catalog, &mut runner, &spec, 200, limits()).unwrap();
    assert_eq!(report.loss_aware.logical_errors, 0);
    assert!(report.loss_aware.errored_shots.is_empty());
    assert_eq!(report.loss_aware.cache_hits, 198);
    assert!(
        ShotDemBuilder::new(&catalog)
            .unwrap()
            .build(&BTreeSet::new(), limits())
            .unwrap()
            .approximations
            .contains(&Approximation::UnflaggedBackground)
    );
}

#[test]
fn r1_other_draw_odds_engine() {
    // A: idle loss before p0; recovery separates it from B. B and C are equal-
    // rate LL draws on (0,1) and (1,2). C would add the excluded flag p2.
    // A-FN blocks B but leaves C active; A-recovered+B blocks C. Equal blocked
    // masses make the shared no-flag factor genuinely common to these covers.
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    c.tick().x(&[0]);
    let r = c.tick().mz(&[0]);
    c.detector(&r).unwrap();
    c.tick().cx(&[(0, 1)]);
    c.tick().cx(&[(1, 2)]);
    probe(&mut c, &[0, 2]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.2)
            .with_lost_partner_loss_probability(0.0)
            .add_recovery_channel(LossSite::BeforeSingleQubit, 0.3, 0.0)
            .add_events_channel(
                LossSite::BeforeTwoQubit,
                0.2,
                &BTreeMap::from([("LL".into(), 1.0)]),
            )
            .build()
            .unwrap(),
    );
    let catalog = LossCatalog::build_with_options(
        &c,
        &noise,
        0.0,
        12,
        10000,
        super::branch_replay::ReplayOptions {
            convention_coins: false,
            ..super::branch_replay::ReplayOptions::default()
        },
    )
    .unwrap();
    let shot = ShotDemBuilder::new(&catalog)
        .unwrap()
        .build(&flags(&[0, 1]), limits())
        .unwrap();
    assert_eq!(shot.assignments.len(), 1);
    assert_eq!(shot.assignments[0].len(), 2);
    // Engine event masses: A-FN and no C = .2*.3*.8 = .048;
    // A recovered, then B = .2*.7*.2 = .028. The latter's record is fair.
    let expected = exact(&shot.assignments, 1)[1];
    let mut runner = TickRunner::new(&c, noise, 824_109).unwrap();
    let mut n = 0u64;
    let mut flips = 0u64;
    for _ in 0..100_000 {
        let shot = runner.shot().unwrap();
        if shot.flags == [true, true, false] {
            n += 1;
            flips += u64::from(shot.detectors[0] ^ 1);
        }
    }
    let interval = pecos_num::stats::jeffreys_interval(flips, n, 0.001).unwrap();
    eprintln!(
        "r1 odds engine: N=100000, seed=824109, flags={{p0,p1}}, p2=false, selected={n}, flipped={flips}, expected={expected}, Jeffreys={interval:?}; omit-factor=0.840909090909091"
    );
    assert!(expected >= interval.lo && expected <= interval.hi);
    close(expected, (0.048 + 0.028 / 2.0) / 0.076);
    assert!(0.840_909_090_909_091 > interval.hi);
}

#[test]
fn r1_errors_and_large_posterior_groups() {
    let error = mix_shot_dem(&[vec![alt(2, 0.7, 0, &[])]], &blank(2), 12)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("group 0") && error.contains("0.7"),
        "{error}"
    );
    let error = mix_shot_dem(&[vec![alt(3, 1.0, 0, &[])]], &blank(2), 12)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("width 3") && error.contains("width 2"),
        "{error}"
    );
    let mut a = alt(2, 1.0, 0, &[]);
    a.removals.insert(tag(7));
    let error = mix_shot_dem(&[vec![a]], &blank(2), 12)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("channel: 7") && error.contains("supplied tags []"),
        "{error}"
    );
    let large = vec![alt(1, 1e-5, 0, &[]); 100_000];
    assert!((large.iter().map(|a| a.pi).sum::<f64>() - 1.0).abs() > 1e-12);
    assert!(
        mix_shot_dem(&[large], &blank(1), 12)
            .unwrap()
            .dem
            .contributions()
            .is_empty()
    );
    let shot = mix_shot_dem(&[], &blank(2), 12).unwrap();
    let mut decoder = BiasedDecoder::build(
        &shot,
        &DecoderSpec::BpOsd(pecos_decoders::spec::BpOsdConfig::default()),
    )
    .unwrap();
    let error = decoder.decode_obs(&[0]).unwrap_err().to_string();
    assert!(
        error.contains("length 1") && error.contains("width 2"),
        "{error}"
    );
    let error = decoder.decode_obs(&[0, 3]).unwrap_err().to_string();
    assert!(error.contains("(1, 3)"), "{error}");
    let catalog = ambiguous_regions(1);
    let mut c = TickCircuit::new();
    c.tick().mz(&[3]);
    let mut runner = TickRunner::new(&c, GeneralNoiseModel::default(), 7).unwrap();
    let error = LossDecoder::run(
        &catalog,
        &mut runner,
        &DecoderSpec::BpOsd(pecos_decoders::spec::BpOsdConfig::default()),
        0,
        limits(),
    )
    .unwrap_err()
    .to_string();
    for field in ["slots", "probes", "records", "record_locations"] {
        assert!(error.contains(field), "{error}");
    }
    let mut catalog = ambiguous_regions(1);
    for branch in &mut catalog.roots[0].branches {
        branch.weight = 0.5;
    }
    let error = ShotDemBuilder::new(&catalog).unwrap_err();
    assert!(matches!(
        error,
        LossDemError::InvalidDrawMass { mass: 1.0, .. }
    ));
    let error = error.to_string();
    assert!(
        error.contains("draw (Idle") && error.contains("flagging mass 1"),
        "{error}"
    );
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0, 1]);
    probe(&mut c, &[0, 1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.2)
            .build()
            .unwrap(),
    );
    let mut tiny = build(&c, &noise);
    for root in &mut tiny.roots {
        root.branches[0].weight = 1e-200;
    }
    let mut alternative = tiny.roots[0].branches[0].clone();
    alternative.flags = flags(&[1]);
    tiny.roots[0].branches.push(alternative);
    tiny.by_flags
        .get_mut(&flags(&[1]))
        .unwrap()
        .push(BranchId { root: 0, branch: 1 });
    let error = ShotDemBuilder::new(&tiny)
        .unwrap()
        .build(&flags(&[0, 1]), limits())
        .unwrap_err()
        .to_string();
    assert!(
        error.contains("weight total 0")
            && error.contains("ProbeId(0)")
            && error.contains("ProbeId(1)")
            && error.contains("correct the catalog branch weights"),
        "{error}"
    );
}

#[test]
fn r1_zero_weight_candidates_do_not_join_regions() {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0, 1]);
    probe(&mut c, &[0, 1]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.2)
            .build()
            .unwrap(),
    );
    let mut catalog = build(&c, &noise);
    let mut zero = catalog.roots[0].branches[0].clone();
    zero.weight = 0.0;
    zero.flags = flags(&[1]);
    catalog.roots[0].branches.push(zero);
    catalog
        .by_flags
        .get_mut(&flags(&[1]))
        .unwrap()
        .push(BranchId { root: 0, branch: 1 });
    let groups = ShotDemBuilder::new(&catalog)
        .unwrap()
        .assignments(&flags(&[0, 1]), 1, 1_000_000)
        .unwrap();
    assert_eq!(groups.len(), 2);
    assert_eq!(groups[0][0].len(), 1);
    assert_eq!(groups[1][0].len(), 1);
}

#[test]
fn followup_blocked_draw_odds_marker() {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    c.tick().x(&[0]);
    let r = c.tick().mz(&[0]);
    c.detector(&r).unwrap();
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.2)
            .add_recovery_channel(LossSite::BeforeSingleQubit, 0.3, 0.0)
            .build()
            .unwrap(),
    );
    let catalog = LossCatalog::build_with_options(
        &c,
        &noise,
        0.0,
        12,
        10000,
        super::branch_replay::ReplayOptions {
            convention_coins: false,
            ..super::branch_replay::ReplayOptions::default()
        },
    )
    .unwrap();
    let shot = ShotDemBuilder::new(&catalog)
        .unwrap()
        .build(&flags(&[0, 1]), limits())
        .unwrap();
    assert_eq!(shot.assignments.len(), 1);
    assert_eq!(shot.assignments[0].len(), 2);
    assert!(
        shot.approximations
            .contains(&Approximation::BlockedDrawOdds)
    );
    // Keeping only p0 selects A-recovered and excludes B: one assignment.
    let single = ShotDemBuilder::new(&catalog)
        .unwrap()
        .build(&flags(&[0]), limits())
        .unwrap();
    assert_eq!(single.assignments.len(), 1);
    assert_eq!(single.assignments[0].len(), 1);
    assert!(
        !single
            .approximations
            .contains(&Approximation::BlockedDrawOdds)
    );
    let empty = ShotDemBuilder::new(&catalog)
        .unwrap()
        .build(&BTreeSet::new(), limits())
        .unwrap();
    assert!(
        !empty
            .approximations
            .contains(&Approximation::BlockedDrawOdds)
    );
    // Separate singleton regions must not look ambiguous merely because there
    // are multiple groups. Conversely, ambiguous independent regions are marked.
    let independent = ambiguous_regions(2);
    let ambiguous = ShotDemBuilder::new(&independent)
        .unwrap()
        .build(&flags(&[0, 1, 2, 3]), limits())
        .unwrap();
    assert_eq!(ambiguous.assignments.len(), 2);
    assert!(
        ambiguous
            .approximations
            .contains(&Approximation::BlockedDrawOdds)
    );
    // The first probes alone each have only their recovered-A branch.
    let singles = ShotDemBuilder::new(&independent)
        .unwrap()
        .build(&flags(&[0, 1]), limits())
        .unwrap();
    assert_eq!(singles.assignments.len(), 2);
    assert!(singles.assignments.iter().all(|group| group.len() == 1));
    assert!(
        !singles
            .approximations
            .contains(&Approximation::BlockedDrawOdds)
    );
}

fn r3_il_catalog() -> LossCatalog {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    c.tick().cx(&[(0, 1)]);
    probe(&mut c, &[1]);
    build(
        &c,
        &model(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_idle_rate(0.2)
                .with_lost_partner_loss_probability(0.0)
                .add_events_channel(
                    LossSite::BeforeTwoQubit,
                    0.2,
                    &BTreeMap::from([("IL".into(), 1.0)]),
                )
                .build()
                .unwrap(),
        ),
    )
}

#[test]
fn r3_present_operand_blocks_region() {
    let catalog = r3_il_catalog();
    let event = catalog.roots.iter().find(|r| r.key.label == "IL").unwrap();
    assert_eq!(event.qubits, vec![0, 1]);
    assert!(catalog.by_flags.contains_key(&flags(&[0])));
    assert!(catalog.by_flags.contains_key(&flags(&[1])));
    assert!(!catalog.by_flags.contains_key(&flags(&[0, 1])));
    assert!(matches!(
        ShotDemBuilder::new(&catalog)
            .unwrap()
            .assignments(&flags(&[0, 1]), 100, 100),
        Err(AssignmentError::NoAssignment { .. })
    ));
}

#[test]
fn r3_same_action_is_not_before() {
    let mut catalog = r3_il_catalog();
    let a = catalog.by_flags[&flags(&[0])][0];
    let b = catalog.by_flags[&flags(&[1])][0];
    // Pin the boundary with a loss transition exactly at the target root.
    catalog.roots[b.root].action_index =
        catalog.roots[a.root].branches[a.branch].lost_over_time[0].action_index;
    let groups = ShotDemBuilder::new(&catalog)
        .unwrap()
        .assignments(&flags(&[0, 1]), 100, 100)
        .unwrap();
    assert_eq!(groups.len(), 2);
    assert!(groups.iter().all(|g| g.len() == 1));
}

#[test]
fn r3_spread_partner_blocks_later_root() {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    c.tick().cx(&[(0, 1)]);
    c.tick().idle(1, &[1]);
    c.tick().x(&[1]);
    probe(&mut c, &[1]);
    let catalog = build(
        &c,
        &model(
            QubitLoss::builder()
                .with_default_recovery(false)
                .with_idle_rate(0.2)
                .with_lost_partner_loss_probability(0.5)
                .add_recovery_channel(LossSite::BeforeSingleQubit, 0.5, 0.0)
                .build()
                .unwrap(),
        ),
    );
    let a = catalog.by_flags[&flags(&[0])]
        .iter()
        .copied()
        .find(|id| {
            catalog.roots[id.root].branches[id.branch]
                .lost_over_time
                .iter()
                .any(|m| m.lost.contains(&1))
        })
        .unwrap();
    let b = catalog.by_flags[&flags(&[1])][0];
    assert_eq!(catalog.roots[a.root].qubits, vec![0]);
    assert_eq!(catalog.roots[b.root].qubits, vec![1]);
    let shot = ShotDemBuilder::new(&catalog)
        .unwrap()
        .build(&flags(&[0, 1]), limits())
        .unwrap();
    assert!(
        shot.assignments
            .iter()
            .flatten()
            .any(|alt| alt.branches.contains(&b))
    );
    assert!(
        !shot
            .assignments
            .iter()
            .flatten()
            .any(|alt| alt.branches.contains(&a) && alt.branches.contains(&b))
    );
}

#[test]
fn r3_odd_pair_cover_memo() {
    let start = std::time::Instant::now();
    let n = 23;
    let mut catalog = ambiguous_regions(n);
    let template = catalog.roots[0].clone();
    catalog.roots.clear();
    catalog.by_flags.clear();
    for a in 0..n {
        for b in a + 1..n {
            let mut root = template.clone();
            let index = catalog.roots.len();
            root.key.location.tick = index;
            root.qubits = vec![index];
            root.branches.truncate(1);
            root.branches[0].lost_over_time.clear();
            root.branches[0].flags = flags(&[a, b]);
            root.branches[0].weight = 0.001;
            catalog.by_flags.insert(
                flags(&[a, b]),
                vec![BranchId {
                    root: index,
                    branch: 0,
                }],
            );
            catalog.roots.push(root);
        }
    }
    super::shot::SEARCH_VISITS.with(|c| c.set(0));
    let error = ShotDemBuilder::new(&catalog)
        .unwrap()
        .assignments(&flags(&(0..n).collect::<Vec<_>>()), 4, 450_000)
        .unwrap_err();
    assert!(
        matches!(error, AssignmentError::NoAssignment { .. }),
        "{error}"
    );
    let nodes = super::shot::SEARCH_VISITS.with(std::cell::Cell::get);
    assert_eq!(nodes, 442_033);
    eprintln!(
        "round3 odd pairs: n={n}, nodes={nodes}, runtime={:?}; unmemoised leaves=81749606400",
        start.elapsed()
    );
}

#[test]
fn r3_search_node_limit_and_accounting() {
    let mut c = TickCircuit::new();
    c.tick().idle(1, &[0]);
    probe(&mut c, &[0]);
    let noise = model(
        QubitLoss::builder()
            .with_default_recovery(false)
            .with_idle_rate(0.2)
            .build()
            .unwrap(),
    );
    let catalog = build(&c, &noise);
    let builder = ShotDemBuilder::new(&catalog).unwrap();
    // Two cover nodes and two candidate nodes. Check both traversal budgets.
    for (max, visited) in [(1, 2), (3, 4)] {
        let error = builder.assignments(&flags(&[0]), 10, max).unwrap_err();
        assert_eq!(
            error,
            AssignmentError::SearchNodeLimit {
                probes: flags(&[0]),
                visited,
                max
            }
        );
        let message = error.to_string();
        assert!(
            message.contains("ProbeId(0)")
                && message.contains(&format!("visited {visited}"))
                && message.contains("increase max_search_nodes")
        );
    }
    assert!(builder.assignments(&flags(&[0]), 10, 4).is_ok());
    let independent = ambiguous_regions(2);
    assert_eq!(
        ShotDemBuilder::new(&independent)
            .unwrap()
            .assignments(&flags(&[0, 1]), 10, 4)
            .unwrap()
            .len(),
        2
    );
    let mut runner = TickRunner::new(&c, noise, 82341).unwrap();
    let report = LossDecoder::run(
        &catalog,
        &mut runner,
        &DecoderSpec::BpOsd(pecos_decoders::spec::BpOsdConfig::default()),
        64,
        ShotLimits {
            max_search_nodes: 0,
            ..limits()
        },
    )
    .unwrap();
    assert_eq!(report.loss_aware.shots, 64);
    assert_eq!(report.loss_aware.logical_errors, 64);
    assert_eq!(
        report.loss_aware.errored_shots[&ErroredShotReason::SearchNodeLimit],
        64
    );
    assert_eq!(report.loss_aware.cache_hits, 62);
}

#[test]
fn r3_empty_effect_and_unflagged_background() {
    let mut shot = mix_shot_dem(&[vec![alt(1, 1.0, 1, &[])]], &blank(1), 12).unwrap();
    // push_contribution retains an empty effect although rendering omits it.
    shot.dem.push_contribution(FaultContribution::direct(
        FaultMechanism::from_unsorted(vec![], vec![]),
        0.25,
    ));
    assert!(!shot.dem.contributions().is_empty());
    assert!(
        !shot
            .dem
            .to_string()
            .lines()
            .any(|line| line.starts_with("error("))
    );
    let mut decoder = BiasedDecoder::build(
        &shot,
        &DecoderSpec::BpOsd(pecos_decoders::spec::BpOsdConfig::default()),
    )
    .unwrap();
    assert_eq!(decoder.decode_obs(&[1]).unwrap(), shot.observable_bias);
    let catalog = ambiguous_regions(1);
    let builder = ShotDemBuilder::new(&catalog).unwrap();
    for f in [BTreeSet::new(), flags(&[0, 1])] {
        assert!(
            builder
                .build(&f, limits())
                .unwrap()
                .approximations
                .contains(&Approximation::UnflaggedBackground)
        );
    }
}

#[test]
fn r3_catalog_caches() {
    let catalog = r3_il_catalog();
    super::shot::PAIR_EVALUATIONS.with(|c| c.set(0));
    super::shot::DRAW_MASS_EVALUATIONS.with(|c| c.set(0));
    let builder = ShotDemBuilder::new(&catalog).unwrap();
    assert_eq!(
        super::shot::DRAW_MASS_EVALUATIONS.with(std::cell::Cell::get),
        1
    );
    assert!(matches!(
        builder.build(&flags(&[0, 1]), limits()),
        Err(ShotBuildError::Assignment(
            AssignmentError::NoAssignment { .. }
        ))
    ));
    let pairs = super::shot::PAIR_EVALUATIONS.with(std::cell::Cell::get);
    assert!(pairs > 0);
    assert!(matches!(
        builder.build(&flags(&[0, 1]), limits()),
        Err(ShotBuildError::Assignment(
            AssignmentError::NoAssignment { .. }
        ))
    ));
    assert_eq!(
        super::shot::PAIR_EVALUATIONS.with(std::cell::Cell::get),
        pairs
    );
    assert_eq!(
        super::shot::DRAW_MASS_EVALUATIONS.with(std::cell::Cell::get),
        1
    );
}
