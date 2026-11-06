// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

use pecos_core::gate_type::GateType;
use pecos_core::{Gate, PauliString, QubitId};
use pecos_qec::fault_tolerance::dem_builder::{
    DemBoundaryKind, DemBuilder, DemSampler, DemSamplerBuilder, DemSliceRoundSchedule,
    DemWindowSpec, DetectorErrorModel, FaultContribution, MeasurementCrosstalkDemMode,
    MeasurementCrosstalkTransitionModel, MemBuilder, NoiseConfig, PauliWeights, PerGateTypeNoise,
    ReplacementBranchApproximation, SamplingEngine,
};
use pecos_qec::fault_tolerance::influence_builder::InfluenceBuilder;
use pecos_qec::fault_tolerance::lookup_decoder::LookupDecoder;
use pecos_qec::fault_tolerance::propagator::{DagFaultAnalyzer, DagFaultInfluenceMap};
use pecos_quantum::{Attribute, DagCircuit, TickCircuit};
use rand::{SeedableRng, rngs::StdRng};
use std::collections::BTreeSet;

const DETECTORS: &str = r#"[{"id":0,"records":[-4],"coords":[0,0,0]},{"id":1,"records":[-3],"coords":[1,0,0]},{"id":2,"records":[-2],"coords":[2,0,0]},{"id":3,"records":[-1],"coords":[3,0,0]}]"#;
const OBSERVABLES: &str = r#"[{"id":0,"records":[-1],"label":"logical"}]"#;

fn circuit() -> TickCircuit {
    let mut tc = TickCircuit::new();
    tc.tick().pz(&[0, 1, 2, 3]);
    tc.tick().h(&[0, 1, 2, 3]);
    tc.tick().h(&[0, 1, 2, 3]);
    tc.tick().x(&[0, 1, 2, 3]);
    tc.tick().x(&[0, 1, 2, 3]);
    tc.tick().cx(&[(0, 1), (2, 3)]);
    tc.tick().idle(1, &[0, 1, 2, 3]);
    tc.tick().mz(&[0, 1, 2, 3]);
    tc.set_meta("detectors", Attribute::String(DETECTORS.into()));
    tc.set_meta("observables", Attribute::String(OBSERVABLES.into()));
    tc
}

fn noise() -> NoiseConfig {
    NoiseConfig::with_idle(0.03, 0.06, 0.07, 0.08, 0.04)
}

fn per_gate_noise() -> PerGateTypeNoise {
    PerGateTypeNoise::from_base_noise(noise())
        .with_1q_rates(GateType::H, [0.01, 0.02, 0.03])
        .with_2q_rates(GateType::CX, [0.002; 15])
        .with_1q_rates(GateType::Idle, [0.02, 0.01, 0.03])
        .with_init_rate(QubitId(0), 0.11)
        .with_measurement_rate(QubitId(1), 0.12)
}

fn mark_tick(tc: &mut TickCircuit, tick: usize, value: &Attribute) {
    let tick = tc.get_tick_mut(tick).expect("fixture tick exists");
    for gate in 0..tick.len() {
        tick.set_gate_attr(gate, "noiseless", value.clone());
    }
}

fn mark_all(tc: &mut TickCircuit, value: &Attribute) {
    let count = tc.iter_ticks().count();
    for tick in 0..count {
        mark_tick(tc, tick, value);
    }
}

fn map(dag: &DagCircuit) -> DagFaultInfluenceMap {
    DagFaultAnalyzer::new(dag)
        .build_influence_map()
        .expect("Clifford fixture")
}

fn build(tc: &TickCircuit, per_gate: bool) -> DetectorErrorModel {
    let dag = DagCircuit::try_from(tc).expect("valid fixture");
    let influence = map(&dag);
    let mut builder = DemBuilder::new(&influence)
        .with_noise_config(noise())
        .with_exact_branch_replay_context(&dag)
        .with_detectors_json(DETECTORS)
        .expect("valid detectors")
        .with_observables_json(OBSERVABLES)
        .expect("valid observable");
    if per_gate {
        builder = builder.with_per_gate_noise(per_gate_noise());
    }
    builder.try_build().expect("valid DEM")
}

// Compare every source's effect, probability, Pauli labels and ownership, not counts.
fn contributions<'a>(items: impl Iterator<Item = &'a FaultContribution>) -> Vec<String> {
    let mut rows: Vec<_> = items.map(|item| format!("{item:?}")).collect();
    rows.sort();
    rows
}

fn declarations(dem: &DetectorErrorModel) -> String {
    format!(
        "{:?} {:?} {:?}",
        dem.detectors, dem.observables, dem.tracked_paulis
    )
}

#[test]
fn noiseless_batches_remove_exactly_their_sources() {
    for per_gate in [false, true] {
        let base = circuit();
        let baseline = build(&base, per_gate);
        let base_dag = DagCircuit::try_from(&base).expect("valid fixture");
        let base_map = map(&base_dag);
        for tick in [0, 1, 5, 6, 7] {
            let mut marked = base.clone();
            mark_tick(&mut marked, tick, &Attribute::Bool(true));
            let dag = DagCircuit::try_from(&marked).expect("valid fixture");
            let marked_map = map(&dag);
            assert_eq!(base_map.measurements, marked_map.measurements);
            assert_eq!(base_map.meas_ids, marked_map.meas_ids);
            assert_eq!(base_map.locations.len(), marked_map.locations.len());
            let removed: BTreeSet<_> = marked_map
                .locations
                .iter()
                .enumerate()
                .filter(|(_, loc)| {
                    dag.get_gate_attr(loc.node, "noiseless") == Some(&Attribute::Bool(true))
                })
                .map(|(idx, _)| u32::try_from(idx).expect("small fixture"))
                .collect();
            assert_eq!(
                removed.len(),
                4,
                "every split batch application, including both CX locations"
            );
            let expected = contributions(
                baseline
                    .contributions()
                    .iter()
                    .filter(|c| c.location_indices.iter().all(|idx| !removed.contains(idx))),
            );
            assert!(
                expected.len() < baseline.contributions().len(),
                "fixture must expose this source kind"
            );
            let actual = build(&marked, per_gate);
            assert_eq!(
                contributions(actual.contributions().iter()),
                expected,
                "tick {tick}, per_gate={per_gate}"
            );
            assert_eq!(declarations(&actual), declarations(&baseline));
        }
    }
}

#[test]
fn noiseless_true_is_empty_false_is_unchanged() {
    for per_gate in [false, true] {
        let base = circuit();
        let baseline = build(&base, per_gate);
        let mut marked = base.clone();
        mark_all(&mut marked, &Attribute::Bool(false));
        let false_dem = build(&marked, per_gate);
        assert_eq!(
            contributions(false_dem.contributions().iter()),
            contributions(baseline.contributions().iter())
        );
        assert_eq!(false_dem.to_string(), baseline.to_string());
        mark_all(&mut marked, &Attribute::Bool(true));
        let empty = build(&marked, per_gate);
        assert!(empty.contributions().is_empty());
        assert_eq!(declarations(&empty), declarations(&baseline));
    }
}

#[test]
fn noiseless_non_boolean_is_rejected_at_every_dem_boundary() {
    for value in [
        Attribute::Int(1),
        Attribute::Float(0.0),
        Attribute::String("true".into()),
    ] {
        let mut tc = circuit();
        mark_tick(&mut tc, 1, &value);
        let dag = DagCircuit::try_from(&tc).expect("valid fixture");
        let maps = [
            map(&dag),
            InfluenceBuilder::new(&dag)
                .expect("Clifford")
                .build()
                .expect("influence"),
        ];
        let mut messages = vec![
            DemBuilder::from_tick_circuit(&tc, 0.0, 0.0, 0.0, 0.0)
                .expect_err("invalid noiseless")
                .to_string(),
            DemSampler::from_tick_circuit(&tc, &noise())
                .expect_err("invalid noiseless")
                .to_string(),
            DemSampler::from_circuit(&dag, &noise())
                .expect_err("invalid noiseless")
                .to_string(),
        ];
        for message in &messages[..2] {
            assert!(
                message.contains("gate H at tick 1 gate 0 must be a boolean"),
                "{message}"
            );
        }
        for influence in &maps {
            messages.extend([
                LookupDecoder::build(influence, &noise(), 1)
                    .expect_err("invalid noiseless")
                    .to_string(),
                DemBuilder::new(influence)
                    .build()
                    .expect_err("invalid noiseless")
                    .to_string(),
                DemBuilder::new(influence)
                    .try_build()
                    .expect_err("invalid noiseless")
                    .to_string(),
                MemBuilder::new(influence)
                    .build()
                    .expect_err("invalid noiseless")
                    .to_string(),
                DemSamplerBuilder::new(influence)
                    .build()
                    .expect_err("invalid noiseless")
                    .to_string(),
                DemSamplerBuilder::new(influence)
                    .raw_measurements()
                    .build()
                    .expect_err("invalid noiseless")
                    .to_string(),
                SamplingEngine::from_influence_map(
                    influence,
                    &vec![0.1; influence.locations.len()],
                    &noise(),
                )
                .expect_err("invalid noiseless")
                .to_string(),
            ]);
        }
        for message in messages {
            assert!(
                message.contains("noiseless")
                    && message.contains("gate H at")
                    && message.contains("must be a boolean"),
                "{message}"
            );
        }
    }
}

#[test]
fn noiseless_samplers_and_mem_have_no_events() {
    let mut tc = circuit();
    mark_all(&mut tc, &Attribute::Bool(true));
    let dag = DagCircuit::try_from(&tc).expect("valid fixture");
    let mut samplers = vec![
        DemSampler::from_tick_circuit(&tc, &noise()).expect("tick sampler"),
        DemSampler::from_circuit(&dag, &noise()).expect("DAG sampler"),
    ];
    for influence in [
        map(&dag),
        InfluenceBuilder::new(&dag)
            .expect("Clifford")
            .build()
            .expect("influence"),
    ] {
        assert_eq!(
            LookupDecoder::build(&influence, &noise(), 1)
                .expect("lookup")
                .num_syndromes(),
            1
        );
        samplers.push(
            DemSampler::from_influence_map(&influence, &vec![0.1; influence.locations.len()])
                .expect("map sampler"),
        );
        samplers.push(
            DemSamplerBuilder::new(&influence)
                .with_noise_config(noise())
                .raw_measurements()
                .build()
                .expect("raw sampler"),
        );
        for per_gate in [false, true] {
            let mut builder = DemSamplerBuilder::new(&influence)
                .with_noise_config(noise())
                .with_detector_records(vec![vec![-4], vec![-3], vec![-2], vec![-1]])
                .with_observable_records(vec![vec![-1]]);
            if per_gate {
                builder = builder.with_per_gate_noise(per_gate_noise());
            }
            samplers.push(builder.build().expect("detector sampler"));
        }
        assert_eq!(
            MemBuilder::new(&influence)
                .with_noise_config(noise())
                .build()
                .expect("MEM")
                .num_mechanisms(),
            0
        );
    }
    let mut rng = StdRng::seed_from_u64(19);
    for sampler in samplers {
        assert_eq!(sampler.num_mechanisms(), 0);
        for _ in 0..64 {
            let (detectors, observables) = sampler.sample(&mut rng);
            assert!(!detectors.into_iter().chain(observables).any(|event| event));
        }
    }
}

#[test]
fn noiseless_crosstalk_payloads_remove_only_their_sources() {
    for global in [false, true] {
        let mut dag = DagCircuit::new();
        dag.pz(&[0, 1]);
        // Order both preparations before a global payload on qubit 1.
        dag.cx(&[(0, 1)]);
        let payload = if global {
            Gate::meas_crosstalk_global_payload(&[1])
        } else {
            Gate::meas_crosstalk_local_payload(&[0])
        };
        let node = dag.add_gate_auto_wire(payload);
        dag.mz(&[0]);
        dag.set_attr(
            "detectors",
            Attribute::String(r#"[{"id":0,"records":[-1]}]"#.into()),
        );
        let cfg = NoiseConfig::new(0.0, 0.0, 0.07, 0.08)
            .set_measurement_crosstalk_local_rate(if global { 0.0 } else { 0.2 })
            .set_measurement_crosstalk_global_rate(if global { 0.2 } else { 0.0 })
            .set_measurement_crosstalk_transition_model(MeasurementCrosstalkTransitionModel {
                p_0_to_1: 0.4,
                ..Default::default()
            })
            .set_measurement_crosstalk_dem_mode(MeasurementCrosstalkDemMode::ExactDeterministic);
        let baseline = DemBuilder::try_from_circuit_with_noise_config(&dag, cfg.clone())
            .expect("crosstalk DEM");
        let influence = map(&dag);
        let removed: BTreeSet<_> = influence
            .locations
            .iter()
            .enumerate()
            .filter(|(_, loc)| loc.node == node)
            .map(|(idx, _)| u32::try_from(idx).expect("small fixture"))
            .collect();
        let expected = contributions(
            baseline
                .contributions()
                .iter()
                .filter(|c| c.location_indices.iter().all(|idx| !removed.contains(idx))),
        );
        assert!(expected.len() < baseline.contributions().len());
        dag.set_gate_attr(node, "noiseless", Attribute::Bool(true));
        let actual = DemBuilder::try_from_circuit_with_noise_config(&dag, cfg.clone())
            .expect("marked crosstalk DEM");
        assert_eq!(contributions(actual.contributions().iter()), expected);
        assert_eq!(declarations(&actual), declarations(&baseline));
        for node in dag.topological_order() {
            dag.set_gate_attr(node, "noiseless", Attribute::Bool(true));
        }
        let empty = DemBuilder::try_from_circuit_with_noise_config(&dag, cfg)
            .expect("circuit-aware crosstalk DEM");
        assert!(empty.contributions().is_empty());
    }
}

#[test]
fn noiseless_exact_replacement_cache_preserves_active_sources() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1, 2, 3]);
    dag.x(&[0, 2]);
    let suppressed = dag.add_gate_auto_wire(Gate::cx(&[(0, 1)]));
    dag.cx(&[(2, 3)]);
    dag.mz(&[1, 3]);
    let detectors = r#"[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]"#;
    let cfg = NoiseConfig::new(0.0, 0.1, 0.0, 0.0)
        .set_p2_weights(PauliWeights::with_replacement(
            [],
            [(PauliString::identity(), 1.0)],
        ))
        .set_p2_replacement_approximation(ReplacementBranchApproximation::ExactBranchReplay);
    let baseline_map = map(&dag);
    let baseline = DemBuilder::new(&baseline_map)
        .with_noise_config(cfg.clone())
        .with_exact_branch_replay_context(&dag)
        .with_detectors_json(detectors)
        .expect("detectors")
        .try_build()
        .expect("baseline replay");
    let expected = contributions(baseline.contributions().iter().filter(|source| {
        source
            .location_indices
            .iter()
            .all(|&index| baseline_map.locations[index as usize].node != suppressed)
    }));
    assert_ne!(expected, Vec::<String>::new());
    assert!(expected.len() < baseline.contributions().len());
    dag.set_gate_attr(suppressed, "noiseless", Attribute::Bool(true));
    let influence = map(&dag);
    let builder = DemBuilder::new(&influence)
        .with_noise_config(cfg)
        .with_exact_branch_replay_context(&dag)
        .with_detectors_json(detectors)
        .expect("detectors");
    // The active CX populates exact replay analyses during validation. Source
    // generation and the second build read that same builder's populated cache.
    for _ in 0..2 {
        let actual = builder.try_build().expect("cached replay");
        assert_eq!(contributions(actual.contributions().iter()), expected);
        assert_eq!(declarations(&actual), declarations(&baseline));
        assert!(
            actual
                .contributions()
                .iter()
                .all(|source| source.replacement_branch)
        );
    }
}

#[test]
fn noiseless_slice_window_has_no_sources() {
    let mut tc = circuit();
    mark_all(&mut tc, &Attribute::Bool(true));
    let dag = DagCircuit::try_from(&tc).expect("valid fixture");
    let influence = map(&dag);
    let dem = build(&tc, false);
    let schedule = DemSliceRoundSchedule::from_location_rounds(
        "marked",
        &dem,
        &vec![0; influence.locations.len()],
    )
    .expect("slice schedule");
    let composed = schedule
        .compose(DemWindowSpec::new(0, 1, 0, DemBoundaryKind::Hard))
        .expect("window");
    assert!(composed.model.contributions().is_empty());
}

fn mixed_circuit() -> TickCircuit {
    let mut tc = TickCircuit::new();
    tc.tick().pz(&[0, 1]);
    tc.tick().h(&[0]).x(&[1]);
    tc.tick().h(&[0]).x(&[1]);
    tc.tick().mz(&[0, 1]);
    tc.set_meta("detectors", Attribute::String(r#"[{"id":0,"records":[-2],"coords":[0,0,0]},{"id":1,"records":[-1],"coords":[1,0,0]}]"#.into()));
    tc.set_meta(
        "observables",
        Attribute::String(r#"[{"id":0,"records":[-1]}]"#.into()),
    );
    tc
}

fn mixed_noise() -> NoiseConfig {
    NoiseConfig::new(0.09, 0.0, 0.0, 0.0)
}

fn mixed_marked() -> TickCircuit {
    let mut tc = mixed_circuit();
    for tick in [1, 2] {
        tc.get_tick_mut(tick).expect("mixed tick").set_gate_attr(
            0,
            "noiseless",
            Attribute::Bool(true),
        );
    }
    tc
}

fn mixed_reference_noise() -> NoiseConfig {
    mixed_noise().set_p1_gate_rate(GateType::H, 0.0)
}

fn mixed_samplers(tc: &TickCircuit, noise: &NoiseConfig) -> Vec<DemSampler> {
    let dag = DagCircuit::try_from(tc).expect("fixture");
    let influence = map(&dag);
    // Explicit caller probabilities intentionally ignore metadata: the map
    // constructor must still suppress marked locations.
    let probabilities: Vec<_> = influence
        .locations
        .iter()
        .map(|loc| match loc.gate_type {
            GateType::H | GateType::X => {
                noise.p1_rate_for_operation(loc.gate_type, loc.noise_gate_type)
            }
            _ => 0.0,
        })
        .collect();
    vec![
        DemSampler::from_tick_circuit(tc, noise).expect("tick sampler"),
        DemSampler::from_circuit(&dag, noise).expect("dag sampler"),
        DemSampler::from_influence_map(&influence, &probabilities).expect("explicit probabilities"),
        DemSamplerBuilder::new(&influence)
            .with_noise_config(noise.clone())
            .raw_measurements()
            .build()
            .expect("raw sampler"),
        DemSamplerBuilder::new(&influence)
            .with_noise_config(noise.clone())
            .with_detector_records(vec![vec![-2], vec![-1]])
            .build()
            .expect("detector sampler"),
        DemSamplerBuilder::new(&influence)
            .with_per_gate_noise(PerGateTypeNoise::from_base_noise(noise.clone()))
            .with_detector_records(vec![vec![-2], vec![-1]])
            .build()
            .expect("per-gate sampler"),
    ]
}

#[test]
fn noiseless_mixed_samplers_retain_active_channels() {
    let actual = mixed_samplers(&mixed_marked(), &mixed_noise());
    let expected = mixed_samplers(&mixed_circuit(), &mixed_reference_noise());
    for (actual, expected) in actual.iter().zip(&expected) {
        assert!(actual.num_mechanisms() > 0);
        assert_eq!(
            actual.to_detector_error_model().to_mechanisms(),
            expected.to_detector_error_model().to_mechanisms()
        );
        let mut actual_rng = StdRng::seed_from_u64(41);
        let mut expected_rng = StdRng::seed_from_u64(41);
        let mut saw_active = false;
        for _ in 0..256 {
            let shot = actual.sample(&mut actual_rng);
            assert_eq!(shot, expected.sample(&mut expected_rng));
            assert!(!shot.0[0], "marked H gates cannot cause events on qubit 0");
            saw_active |= shot.0[1];
        }
        assert!(saw_active, "unmarked X gates must retain noise");
    }
}

#[test]
fn noiseless_mixed_mem_and_lookup_retain_active_channels() {
    let marked = DagCircuit::try_from(&mixed_marked()).expect("fixture");
    let baseline = DagCircuit::try_from(&mixed_circuit()).expect("fixture");
    let actual_map = map(&marked);
    let reference_map = map(&baseline);
    let actual = MemBuilder::new(&actual_map)
        .with_noise_config(mixed_noise())
        .build()
        .expect("MEM");
    let expected = MemBuilder::new(&reference_map)
        .with_noise_config(mixed_reference_noise())
        .build()
        .expect("reference MEM");
    assert!(!actual.mechanisms.is_empty());
    assert_eq!(actual.mechanisms, expected.mechanisms);
    let lookup = LookupDecoder::build(&actual_map, &mixed_noise(), 1).expect("lookup");
    let reference = LookupDecoder::build(&reference_map, &mixed_reference_noise(), 1)
        .expect("reference lookup");
    assert_eq!(lookup.num_syndromes(), reference.num_syndromes());
    assert_eq!(
        lookup.accounted_probability().to_bits(),
        reference.accounted_probability().to_bits()
    );
    // Weight-one accounting excludes simultaneous faults at the two active X
    // gates. Dropping all noise would incorrectly give probability one.
    assert!(lookup.accounted_probability() < 1.0);
    for syndrome in [vec![], vec![0], vec![1], vec![0, 1]] {
        assert_eq!(
            format!("{:?}", lookup.decode(&syndrome)),
            format!("{:?}", reference.decode(&syndrome))
        );
    }
}

#[test]
fn noiseless_mixed_slice_window_retains_active_sources() {
    let marked = mixed_marked();
    let baseline = mixed_circuit();
    let actual =
        DemBuilder::try_from_tick_circuit_with_noise_config(&marked, mixed_noise()).expect("DEM");
    let expected =
        DemBuilder::try_from_tick_circuit_with_noise_config(&baseline, mixed_reference_noise())
            .expect("reference DEM");
    assert!(!actual.contributions().is_empty());
    assert_eq!(
        contributions(actual.contributions().iter()),
        contributions(expected.contributions().iter())
    );
    let location_count = map(&DagCircuit::try_from(&marked).expect("fixture"))
        .locations
        .len();
    let compose = |dem: &DetectorErrorModel| {
        DemSliceRoundSchedule::from_location_rounds("mixed", dem, &vec![0; location_count])
            .expect("schedule")
            .compose(DemWindowSpec::new(0, 1, 0, DemBoundaryKind::Hard))
            .expect("window")
            .model
    };
    let composed = compose(&actual);
    assert!(!composed.contributions().is_empty());
    assert_eq!(composed.to_mechanisms(), compose(&expected).to_mechanisms());
}

#[test]
fn noiseless_catalog_and_targeted_lookup_preserve_only_active_sources() {
    use pecos_qec::fault_tolerance::fault_sampler::{
        FaultCatalog, StochasticNoiseParams, build_fault_table,
    };
    use pecos_qec::fault_tolerance::targeted_lookup_decoder::TargetedLookupDecoder;
    let base = circuit();
    let baseline = FaultCatalog::from_circuit(&base).expect("catalog");
    for tick in [0, 1, 5, 7] {
        let mut marked = base.clone();
        mark_tick(&mut marked, tick, &Attribute::Bool(true));
        let actual = FaultCatalog::from_circuit(&marked).expect("marked catalog");
        let mut reference = baseline.clone();
        reference.locations.retain(|loc| loc.tick != tick);
        assert!(!actual.locations.is_empty());
        assert!(reference.locations.len() < baseline.locations.len());
        assert_eq!(
            format!("{:?}", actual.locations),
            format!("{:?}", reference.locations)
        );
        for rate in [0.03, 0.09] {
            let params = StochasticNoiseParams {
                p1: rate,
                p2: rate,
                p_prep: rate,
                p_meas: rate,
            };
            let actual = actual.parameterized(&params);
            let expected = reference.parameterized(&params);
            assert_eq!(actual.to_mechanisms(), expected.to_mechanisms());
            assert_eq!(
                build_fault_table(&marked, &params).expect("fault table"),
                expected.to_mechanisms()
            );
            let decoder = TargetedLookupDecoder::new(&actual).max_faults(2);
            let expected_decoder = TargetedLookupDecoder::new(&expected).max_faults(2);
            assert_eq!(
                decoder.base_probability().to_bits(),
                expected_decoder.base_probability().to_bits()
            );
            for mask in 0..16 {
                let syndrome: Vec<_> = (0..4).filter(|bit| mask & (1 << bit) != 0).collect();
                assert_eq!(
                    decoder.decode(&syndrome).logical_weights,
                    expected_decoder.decode(&syndrome).logical_weights
                );
            }
        }
    }
    let mut fully_marked = base;
    mark_all(&mut fully_marked, &Attribute::Bool(true));
    let catalog = FaultCatalog::from_circuit(&fully_marked).expect("fully marked");
    assert!(catalog.locations.is_empty());
    assert!(
        TargetedLookupDecoder::new(&catalog)
            .decode(&[0])
            .logical_weights
            .is_empty()
    );
    mark_tick(&mut fully_marked, 1, &Attribute::String("true".into()));
    let error = FaultCatalog::from_circuit(&fully_marked)
        .expect_err("malformed noiseless")
        .to_string();
    assert!(
        error.contains("noiseless")
            && error.contains("gate H at")
            && error.contains("must be a boolean")
    );
}

struct ExportedRows {
    dimensions: [u32; 3],
    channels: Vec<Vec<Vec<u32>>>,
}

fn exported_rows(map: &DagFaultInfluenceMap) -> ExportedRows {
    let (n, d, o, dx, dxdata, dy, dydata, dz, dzdata, ox, oxdata, oy, oydata, oz, ozdata) =
        map.export_csr().expect("valid export");
    let channels = [
        (dx, dxdata),
        (dy, dydata),
        (dz, dzdata),
        (ox, oxdata),
        (oy, oydata),
        (oz, ozdata),
    ]
    .into_iter()
    .map(|(offsets, data)| {
        assert_eq!(offsets.len(), map.locations.len() + 1);
        offsets
            .windows(2)
            .map(|range| data[range[0] as usize..range[1] as usize].to_vec())
            .collect()
    })
    .collect();
    ExportedRows {
        dimensions: [n, d, o],
        channels,
    }
}

#[test]
fn noiseless_csr_export_suppresses_rows_without_renumbering() {
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    dag.h(&[0]);
    dag.h(&[0]);
    dag.x(&[1]);
    let meas = dag.mz(&[0, 1]);
    dag.observable_labeled("parity", &meas).expect("observable");
    let build = |dag: &DagCircuit| {
        InfluenceBuilder::new(dag)
            .expect("Clifford")
            .with_circuit_annotations()
            .expect("annotations")
            .build()
            .expect("map")
    };
    let base = build(&dag);
    let baseline = exported_rows(&base);
    let node = base
        .locations
        .iter()
        .find(|loc| loc.gate_type == GateType::H)
        .expect("H gate")
        .node;
    dag.set_gate_attr(node, "noiseless", Attribute::Bool(true));
    let marked = build(&dag);
    let actual = exported_rows(&marked);
    assert_eq!(actual.dimensions, baseline.dimensions);
    assert_eq!(marked.measurements, base.measurements);
    assert_eq!(marked.meas_ids, base.meas_ids);
    let mut removed_effects = 0;
    let mut active_effects = 0;
    for (channel, reference) in actual.channels.iter().zip(&baseline.channels) {
        for (idx, loc) in base.locations.iter().enumerate() {
            if loc.node == node {
                removed_effects += reference[idx].len();
                assert_eq!(channel[idx], Vec::<u32>::new());
            } else {
                active_effects += channel[idx].len();
                assert_eq!(channel[idx], reference[idx]);
            }
        }
    }
    assert!(removed_effects > 0 && active_effects > 0);
    assert_eq!(
        marked.influences.detectors_x.data,
        base.influences.detectors_x.data
    );
    assert_eq!(
        marked.influences.dem_outputs_x.data,
        base.influences.dem_outputs_x.data
    );
    for node in dag.topological_order() {
        dag.set_gate_attr(node, "noiseless", Attribute::Bool(true));
    }
    assert!(
        exported_rows(&build(&dag))
            .channels
            .iter()
            .flatten()
            .all(Vec::is_empty)
    );
    dag.set_gate_attr(node, "noiseless", Attribute::Int(1));
    let Err(error) = build(&dag).export_csr() else {
        panic!("bad attribute must fail export");
    };
    assert!(
        error.contains("noiseless")
            && error.contains("gate H at")
            && error.contains("must be a boolean")
    );
}

#[test]
fn noiseless_preserves_configuration_errors() {
    use pecos_core::pauli::X;
    let invalid = NoiseConfig::new(0.0, 0.1, 0.0, 0.0)
        .set_p2_weights(PauliWeights::with_replacement([], [(X(2), 1.0)]))
        .set_p2_replacement_approximation(ReplacementBranchApproximation::ExactBranchReplay);
    let errors = |tc: &TickCircuit| {
        let dag = DagCircuit::try_from(tc).expect("fixture");
        let influence = map(&dag);
        vec![
            DemBuilder::new(&influence)
                .with_noise_config(invalid.clone())
                .build()
                .expect_err("provider required")
                .to_string(),
            DemBuilder::try_from_circuit_with_noise_config(&dag, invalid.clone())
                .expect_err("invalid replay label")
                .to_string(),
            DemSampler::from_circuit(&dag, &invalid)
                .expect_err("provider required")
                .to_string(),
            DemSamplerBuilder::new(&influence)
                .with_noise_config(invalid.clone())
                .build()
                .expect_err("provider required")
                .to_string(),
            SamplingEngine::from_influence_map(
                &influence,
                &vec![0.1; influence.locations.len()],
                &invalid,
            )
            .expect_err("provider required")
            .to_string(),
            MemBuilder::new(&influence)
                .with_noise_config(invalid.clone())
                .build()
                .expect_err("MEM cannot represent branches")
                .to_string(),
        ]
    };
    let mut tc = circuit();
    let baseline = errors(&tc);
    mark_all(&mut tc, &Attribute::Bool(true));
    assert_eq!(errors(&tc), baseline);
    // Suppression remains observable even though validation is independent.
    assert!(build(&tc, false).contributions().is_empty());
}

#[test]
fn noiseless_mem_attribute_error_precedes_configuration_error() {
    use pecos_core::pauli::X;
    let mut tc = circuit();
    let invalid = NoiseConfig::new(0.0, 0.1, 0.0, 0.0)
        .set_p2_weights(PauliWeights::with_replacement([], [(X(0), 1.0)]))
        .set_p2_replacement_approximation(ReplacementBranchApproximation::BranchImpact);
    mark_tick(&mut tc, 1, &Attribute::Int(1));
    let dag = DagCircuit::try_from(&tc).expect("fixture");
    let influence = map(&dag);
    let error = MemBuilder::new(&influence)
        .with_noise_config(invalid)
        .build()
        .expect_err("attribute takes precedence")
        .to_string();
    assert!(error.contains("noiseless"), "{error}");
}

#[test]
fn noiseless_tick_errors_keep_batch_provenance() {
    let mut tc = mixed_circuit();
    tc.get_tick_mut(2).expect("mixed tick").set_gate_attr(
        1,
        "noiseless",
        Attribute::String("true".into()),
    );
    let messages = [
        DemBuilder::try_from_tick_circuit(&tc, 0.0, 0.0, 0.0, 0.0)
            .expect_err("invalid attribute")
            .to_string(),
        DemBuilder::try_from_tick_circuit_with_noise_config(&tc, noise())
            .expect_err("invalid attribute")
            .to_string(),
        DemSampler::from_tick_circuit(&tc, &noise())
            .expect_err("invalid attribute")
            .to_string(),
        pecos_qec::fault_tolerance::fault_sampler::FaultCatalog::from_circuit(&tc)
            .expect_err("invalid attribute")
            .to_string(),
    ];
    for message in messages {
        assert!(
            message.contains("noiseless")
                && message.contains("gate X")
                && message.contains("tick 2 gate 1")
                && message.contains("must be a boolean"),
            "{message}"
        );
    }
    let dag = DagCircuit::try_from(&tc).expect("fixture");
    let error = DemBuilder::try_from_circuit_with_noise_config(&dag, noise())
        .expect_err("invalid attribute")
        .to_string();
    assert!(error.contains("DAG node"), "{error}");
}

#[test]
fn noiseless_only_marks_one_batch_in_a_tick() {
    use pecos_qec::fault_tolerance::fault_sampler::FaultCatalog;
    let tc = mixed_circuit();
    let marked = mixed_marked();
    let baseline = FaultCatalog::from_circuit(&tc).expect("baseline");
    let mut expected = baseline.clone();
    expected
        .locations
        .retain(|loc| !([1, 2].contains(&loc.tick) && loc.gate_index == 0));
    assert!(expected.locations.len() < baseline.locations.len());
    assert!(
        expected
            .locations
            .iter()
            .any(|loc| loc.tick == 1 && loc.gate_index == 1)
    );
    assert_eq!(
        format!(
            "{:?}",
            FaultCatalog::from_circuit(&marked)
                .expect("marked")
                .locations
        ),
        format!("{:?}", expected.locations)
    );
    let actual = DemBuilder::try_from_tick_circuit_with_noise_config(&marked, mixed_noise())
        .expect("marked DEM");
    let expected =
        DemBuilder::try_from_tick_circuit_with_noise_config(&tc, mixed_reference_noise())
            .expect("reference DEM");
    assert!(!actual.contributions().is_empty());
    assert_eq!(
        contributions(actual.contributions().iter()),
        contributions(expected.contributions().iter())
    );
}

#[test]
fn noiseless_preserves_crosstalk_configuration_errors() {
    for global in [false, true] {
        let mut dag = DagCircuit::new();
        dag.pz(&[0]);
        dag.h(&[0]);
        dag.add_gate_auto_wire(if global {
            Gate::meas_crosstalk_global_payload(&[0])
        } else {
            Gate::meas_crosstalk_local_payload(&[0])
        });
        dag.mz(&[0]);
        let config = NoiseConfig::new(0.0, 0.0, 0.0, 0.0)
            .set_measurement_crosstalk_local_rate(if global { 0.0 } else { 0.2 })
            .set_measurement_crosstalk_global_rate(if global { 0.2 } else { 0.0 })
            .set_measurement_crosstalk_dem_mode(MeasurementCrosstalkDemMode::ExactDeterministic);
        let errors = |dag: &DagCircuit| {
            let influence = map(dag);
            vec![
                DemBuilder::new(&influence)
                    .with_noise_config(config.clone())
                    .build()
                    .expect_err("context required")
                    .to_string(),
                DemBuilder::new(&influence)
                    .with_noise_config(config.clone())
                    .with_exact_branch_replay_context(dag)
                    .build()
                    .expect_err("random hidden measurement")
                    .to_string(),
                DemBuilder::new(&influence)
                    .with_noise_config(config.clone().set_measurement_crosstalk_transition_model(
                        MeasurementCrosstalkTransitionModel {
                            p_0_to_leak: 0.1,
                            ..Default::default()
                        },
                    ))
                    .with_exact_branch_replay_context(dag)
                    .build()
                    .expect_err("leakage unsupported")
                    .to_string(),
            ]
        };
        let expected = errors(&dag);
        for node in dag.topological_order() {
            dag.set_gate_attr(node, "noiseless", Attribute::Bool(true));
        }
        assert_eq!(errors(&dag), expected);
        let influence = map(&dag);
        assert!(
            DemBuilder::new(&influence)
                .with_noise_config(NoiseConfig::new(0.03, 0.06, 0.07, 0.08))
                .with_detectors_json(r#"[{"id":0,"records":[-1]}]"#)
                .expect("detector")
                .build()
                .expect("ordinary noise")
                .contributions()
                .is_empty()
        );
    }
}

#[test]
fn noiseless_preserves_channel_validation() {
    let mut marked = circuit();
    mark_all(&mut marked, &Attribute::Bool(true));
    for config in [
        NoiseConfig::new(2.0, 0.0, 0.0, 0.0),
        NoiseConfig::new(0.0, 2.0, 0.0, 0.0),
        NoiseConfig::default().set_idle(2.0),
    ] {
        let errors = |tc: &TickCircuit| {
            let dag = DagCircuit::try_from(tc).expect("fixture");
            let influence = map(&dag);
            vec![
                DemBuilder::try_from_tick_circuit_with_noise_config(tc, config.clone())
                    .expect_err("invalid channel")
                    .to_string(),
                MemBuilder::new(&influence)
                    .with_noise_config(config.clone())
                    .build()
                    .expect_err("invalid channel")
                    .to_string(),
                DemSamplerBuilder::new(&influence)
                    .with_noise_config(config.clone())
                    .with_detector_records(vec![vec![-4], vec![-3], vec![-2], vec![-1]])
                    .build()
                    .expect_err("invalid channel")
                    .to_string(),
                SamplingEngine::from_influence_map(
                    &influence,
                    &influence
                        .locations
                        .iter()
                        .map(|loc| match loc.gate_type {
                            GateType::H => config.p1,
                            GateType::CX => config.p2,
                            GateType::Idle => config.p_idle,
                            _ => 0.0,
                        })
                        .collect::<Vec<_>>(),
                    &config,
                )
                .expect_err("invalid channel")
                .to_string(),
            ]
        };
        assert_eq!(errors(&marked), errors(&circuit()));
    }
    assert!(build(&marked, false).contributions().is_empty());
}

#[test]
fn noiseless_independent_idle_channels_preserve_active_measurements() {
    use pecos_qec::fault_tolerance::dem_builder::IdleNoiseFamily;
    let mut tc = circuit();
    let reference_noise = NoiseConfig::new(0.0, 0.0, 0.07, 0.0);
    let config = reference_noise
        .clone()
        .set_idle_quadratic(IdleNoiseFamily::new(
            0.03,
            [("X".to_owned(), 1.0)].into_iter().collect(),
        ));
    let expected =
        DemBuilder::try_from_tick_circuit_with_noise_config(&tc, reference_noise.clone())
            .expect("reference");
    mark_tick(&mut tc, 6, &Attribute::Bool(true));
    let actual = DemBuilder::try_from_tick_circuit_with_noise_config(&tc, config.clone())
        .expect("marked idle");
    assert!(!actual.contributions().is_empty());
    assert_eq!(
        contributions(actual.contributions().iter()),
        contributions(expected.contributions().iter())
    );
    let dag = DagCircuit::try_from(&tc).expect("fixture");
    let influence = map(&dag);
    let mem = |cfg: NoiseConfig| {
        MemBuilder::new(&influence)
            .with_noise_config(cfg)
            .build()
            .expect("MEM")
    };
    assert_eq!(
        mem(config.clone()).mechanisms,
        mem(reference_noise.clone()).mechanisms
    );
    for raw in [false, true] {
        let sampler = |cfg: NoiseConfig| {
            let builder = DemSamplerBuilder::new(&influence).with_noise_config(cfg);
            if raw {
                builder.raw_measurements().build()
            } else {
                builder
                    .with_detector_records(vec![vec![-4], vec![-3], vec![-2], vec![-1]])
                    .build()
            }
            .expect("sampler")
        };
        assert_eq!(
            format!(
                "{:?}",
                sampler(config.clone())
                    .to_detector_error_model()
                    .to_mechanisms()
            ),
            format!(
                "{:?}",
                sampler(reference_noise.clone())
                    .to_detector_error_model()
                    .to_mechanisms()
            )
        );
    }
}

#[test]
fn noiseless_preserves_unfittable_channel_errors() {
    let mut tc = TickCircuit::new();
    tc.tick().pz(&[0, 1]);
    tc.tick().h(&[0]);
    tc.tick().cx(&[(0, 1)]);
    tc.tick().x(&[0]);
    tc.tick().x(&[0]);
    tc.tick().cx(&[(0, 1)]);
    tc.tick().h(&[0]);
    tc.tick().mz(&[0, 1]);
    tc.set_meta(
        "detectors",
        Attribute::String(r#"[{"id":0,"records":[-2]},{"id":1,"records":[-1]}]"#.into()),
    );
    // Prepare then unprepare a Bell pair, with two X gates that ideally cancel.
    // Each Pauli has probability 0.3 (total 0.9): valid probabilities, but faults
    // between the entanglers reach independent outputs and have negative
    // signature characters. Ideal detector outcomes are both zero.
    let config = NoiseConfig::new(0.9, 0.0, 0.0, 0.0);
    let errors = |tc: &TickCircuit| {
        let dag = DagCircuit::try_from(tc).expect("fixture");
        let influence = map(&dag);
        let probabilities: Vec<_> = influence
            .locations
            .iter()
            .map(|loc| {
                if matches!(loc.gate_type, GateType::H | GateType::X) && !loc.before {
                    0.9
                } else {
                    0.0
                }
            })
            .collect();
        vec![
            DemBuilder::try_from_tick_circuit_with_noise_config(tc, config.clone())
                .expect_err("unfittable DEM")
                .to_string(),
            MemBuilder::new(&influence)
                .with_noise_config(config.clone())
                .build()
                .expect_err("unfittable MEM")
                .to_string(),
            SamplingEngine::from_influence_map(&influence, &probabilities, &config)
                .expect_err("unfittable map sampler")
                .to_string(),
            DemSamplerBuilder::new(&influence)
                .with_noise_config(config.clone())
                .with_detector_records(vec![vec![-2], vec![-1]])
                .build()
                .expect_err("unfittable builder sampler")
                .to_string(),
            DemSampler::from_tick_circuit(tc, &config)
                .expect_err("unfittable circuit sampler")
                .to_string(),
        ]
    };
    let expected = errors(&tc);
    for error in &expected {
        assert!(
            error.contains("signature channel") && error.contains("must all be positive"),
            "{error}"
        );
    }
    mark_all(&mut tc, &Attribute::Bool(true));
    assert_eq!(errors(&tc), expected);
}

#[test]
fn noiseless_branch_impact_retains_active_y_decomposed_sources() {
    use pecos_core::pauli::{X, Y};
    use pecos_qec::fault_tolerance::dem_builder::FaultSourceType;
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1, 2, 3]);
    dag.x(&[0, 2]);
    let suppressed = dag.add_gate_auto_wire(Gate::cx(&[(0, 1)]));
    dag.cx(&[(2, 3)]);
    // Extend the exact-replay fixture's readout to expose both X and Z effects.
    dag.h(&[0, 2]);
    dag.mz(&[0, 1, 2, 3]);
    let config = NoiseConfig::new(0.0, 0.1, 0.0, 0.0)
        .set_p2_weights(PauliWeights::with_replacement([], [(Y(0) * X(1), 1.0)]))
        .set_p2_replacement_approximation(ReplacementBranchApproximation::BranchImpact);
    let baseline_map = map(&dag);
    let baseline = DemBuilder::new(&baseline_map)
        .with_noise_config(config.clone())
        .with_detectors_json(DETECTORS)
        .expect("detectors")
        .try_build()
        .expect("branch impacts");
    let kept: Vec<_> = baseline
        .contributions()
        .iter()
        .filter(|source| {
            source
                .location_indices
                .iter()
                .all(|&index| baseline_map.locations[index as usize].node != suppressed)
        })
        .collect();
    let removed: Vec<_> = baseline
        .contributions()
        .iter()
        .filter(|source| {
            source
                .location_indices
                .iter()
                .any(|&index| baseline_map.locations[index as usize].node == suppressed)
        })
        .collect();
    for sources in [&kept, &removed] {
        assert!(sources.iter().all(|source| source.replacement_branch));
        assert!(
            sources
                .iter()
                .any(|source| matches!(source.source_type, FaultSourceType::YDecomposed { .. }))
        );
    }
    dag.set_gate_attr(suppressed, "noiseless", Attribute::Bool(true));
    let influence = map(&dag);
    let actual = DemBuilder::new(&influence)
        .with_noise_config(config)
        .with_detectors_json(DETECTORS)
        .expect("detectors")
        .try_build()
        .expect("marked branch impacts");
    assert_eq!(
        contributions(actual.contributions().iter()),
        contributions(kept.into_iter())
    );
    assert_eq!(declarations(&actual), declarations(&baseline));
}

#[test]
fn noiseless_structural_combinations_keep_indices_and_validate_repeated_qubits() {
    let tc = circuit();
    let dag = DagCircuit::try_from(&tc).expect("fixture");
    let baseline = map(&dag);
    let mut marked_tc = tc;
    mark_all(&mut marked_tc, &Attribute::Bool(true));
    let marked_dag = DagCircuit::try_from(&marked_tc).expect("fixture");
    let mut marked = map(&marked_dag);
    let combos = |influence: &DagFaultInfluenceMap| {
        let mut result = Vec::new();
        influence
            .for_each_fault_combo(1, |combo| result.push(format!("{combo:?}")))
            .expect("structural combinations");
        result
    };
    assert!(!baseline.gate_fault_locations().is_empty());
    assert_eq!(combos(&marked), combos(&baseline));
    let cx_node = marked
        .locations
        .iter()
        .find(|loc| loc.gate_type == GateType::CX)
        .expect("CX fixture")
        .node;
    for loc in &mut marked.locations {
        if loc.node == cx_node {
            loc.qubits = vec![QubitId(0)];
        }
    }
    let marked_error = LookupDecoder::build(&marked, &noise(), 1)
        .expect_err("marked repeated qubits")
        .to_string();
    assert!(marked_error.contains("duplicate"), "{marked_error}");
    for loc in &mut marked.locations {
        loc.noiseless = false;
    }
    assert_eq!(
        LookupDecoder::build(&marked, &noise(), 1)
            .expect_err("repeated qubits")
            .to_string(),
        marked_error
    );
}

#[test]
fn noiseless_attribute_precedes_unsupported_gate_errors() {
    use pecos_qec::fault_tolerance::fault_sampler::FaultCatalog;
    let mut tc = TickCircuit::new();
    tc.tick().t(&[0]);
    mark_tick(&mut tc, 0, &Attribute::String("true".into()));
    let dag = DagCircuit::try_from(&tc).expect("fixture");
    let errors = [
        DemBuilder::try_from_tick_circuit_with_noise_config(&tc, NoiseConfig::default())
            .expect_err("invalid attribute")
            .to_string(),
        DemBuilder::try_from_circuit_with_noise_config(&dag, NoiseConfig::default())
            .expect_err("invalid attribute")
            .to_string(),
        FaultCatalog::from_circuit(&tc)
            .expect_err("invalid attribute")
            .to_string(),
        DemSampler::from_tick_circuit(&tc, &NoiseConfig::default())
            .expect_err("invalid attribute")
            .to_string(),
        DemSampler::from_circuit(&dag, &NoiseConfig::default())
            .expect_err("invalid attribute")
            .to_string(),
    ];
    for error in errors {
        assert!(
            error.contains("noiseless")
                && error.contains("gate T at")
                && error.contains("must be a boolean"),
            "{error}"
        );
    }
}

#[test]
fn noiseless_dem_build_validates_before_circuit_passes() {
    let mut tc = TickCircuit::new();
    tc.tick().cx(&[(0, 1)]);
    tc.tick().h(&[0]);
    mark_tick(&mut tc, 1, &Attribute::Int(1));
    let Err(error) = DetectorErrorModel::builder()
        .with_circuit(tc)
        .with_detectors(vec![])
        .with_observables(vec![])
        .with_idle_after_2q(1.0)
        .build()
    else {
        panic!("invalid input attribute must be rejected")
    };
    let error = error.to_string();
    assert!(
        error.contains("gate H at tick 1 gate 0 must be a boolean"),
        "{error}"
    );

    let mut tc = TickCircuit::new();
    tc.tick().idle(1, &[0]);
    mark_tick(&mut tc, 0, &Attribute::String("true".into()));
    let Err(error) = DetectorErrorModel::builder()
        .with_circuit(tc)
        .with_detectors(vec![])
        .with_observables(vec![])
        .with_strip_traced_idles(true)
        .build()
    else {
        panic!("stripping must not delete invalid metadata")
    };
    let error = error.to_string();
    assert!(
        error.contains("gate Idle at tick 0 gate 0 must be a boolean"),
        "{error}"
    );
}

struct NoiselessProgram(TickCircuit);

impl pecos_qec::DemProgram for NoiselessProgram {
    fn prepare(
        &mut self,
        _request: &pecos_qec::DemProgramRequest<'_>,
    ) -> Result<pecos_qec::DemBuildInput, pecos_qec::DemBuildError> {
        Ok(pecos_qec::DemBuildInput {
            circuit: self.0.clone(),
            probes: None,
            resolved_detector_outputs: None,
            resolved_observable_outputs: None,
        })
    }
}

#[test]
fn noiseless_dem_program_validates_before_circuit_passes() {
    let mut insertion = TickCircuit::new();
    insertion.tick().cx(&[(0, 1)]);
    insertion.tick().h(&[0]);
    mark_tick(&mut insertion, 1, &Attribute::Int(1));
    let mut stripping = TickCircuit::new();
    stripping.tick().idle(1, &[0]);
    mark_tick(&mut stripping, 0, &Attribute::String("true".into()));

    for (circuit, insert, expected) in [
        (insertion, true, "gate H at tick 1 gate 0 must be a boolean"),
        (
            stripping,
            false,
            "gate Idle at tick 0 gate 0 must be a boolean",
        ),
    ] {
        let builder = DetectorErrorModel::builder()
            .with_program(NoiselessProgram(circuit))
            .with_qubits(2)
            .with_detectors(vec![])
            .with_observables(vec![]);
        let builder = if insert {
            builder.with_idle_after_2q(1.0)
        } else {
            builder.with_strip_traced_idles(true)
        };
        let Err(error) = builder.build() else {
            panic!("program metadata must be validated before circuit passes")
        };
        let error = error.to_string();
        assert!(error.contains(expected), "{error}");
    }
}

#[test]
fn noiseless_csr_preserves_hand_built_row_counts() {
    use pecos_qec::fault_tolerance::propagator::dag::CsrArray;
    let export = |map: &DagFaultInfluenceMap| {
        let (_, _, _, dx, dxd, dy, dyd, dz, dzd, ox, oxd, oy, oyd, oz, ozd) =
            map.export_csr().expect("export");
        vec![
            (dx, dxd),
            (dy, dyd),
            (dz, dzd),
            (ox, oxd),
            (oy, oyd),
            (oz, ozd),
        ]
    };
    let mut dag = DagCircuit::new();
    dag.pz(&[0, 1]);
    for offsets in [
        vec![],
        vec![0],
        vec![0, 1],
        vec![0, 1, 2],
        vec![0, 1, 2, 3, 4],
    ] {
        let mut influence = map(&dag);
        influence.locations.truncate(2);
        let data: Vec<u32> = (0..offsets.last().copied().unwrap_or(0)).collect();
        let csr = CsrArray {
            offsets: offsets.clone(),
            data: data.clone(),
        };
        influence.influences.detectors_x = csr.clone();
        influence.influences.detectors_y = csr.clone();
        influence.influences.detectors_z = csr.clone();
        influence.influences.dem_outputs_x = csr.clone();
        influence.influences.dem_outputs_y = csr.clone();
        influence.influences.dem_outputs_z = csr;
        assert_eq!(export(&influence), vec![(offsets.clone(), data.clone()); 6]);
        influence.locations[1].noiseless = true;
        let expected = if data.len() >= 2 {
            (
                offsets
                    .iter()
                    .map(|&offset| if offset >= 2 { offset - 1 } else { offset })
                    .collect(),
                data.iter().copied().filter(|&value| value != 1).collect(),
            )
        } else {
            (offsets, data)
        };
        assert_eq!(export(&influence), vec![expected; 6]);
    }
}
