use pecos_core::{Angle64, errors::PecosError};
use pecos_engines::noise::{GeneralNoiseModelBuilder, IntoNoiseModel};
use pecos_engines::runtime_frame::ShotContext;
use pecos_engines::scheduled_events::{
    ScheduledBatchAdapter, ScheduledEventBatch, ScheduledEventIdleNoise, ScheduledEventOp,
    ScheduledGateBuffer, ScheduledResult, encode_event_batches,
};
use pecos_engines::scheduled_frame::{
    ScheduledIdleNoise, ScheduledLocalNoise, TimedBatch, encode_timed_batches,
};
use pecos_engines::{
    ByteMessage, Engine, EngineSystem, Gate, GateType, StateVecEngine,
    quantum_system::QuantumSystem,
};
use std::collections::BTreeMap;

fn model(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
    entries.iter().map(|(k, v)| ((*k).to_owned(), *v)).collect()
}
fn batch(index: u64, start: u64, duration: u64, gates: Vec<Gate>) -> TimedBatch {
    TimedBatch {
        runtime_shot_id: 7,
        batch_index: index,
        start_nanos: start,
        duration_nanos: duration,
        gates,
    }
}
fn context(shot: usize) -> ShotContext {
    ShotContext {
        run: 1,
        worker: 0,
        shot,
    }
}
struct PassThrough;
impl ScheduledBatchAdapter for PassThrough {
    fn validate(&self, batch: &ScheduledEventBatch) -> Result<(), PecosError> {
        if batch.operations.iter().any(|op| matches!(op, ScheduledEventOp::Custom { tag, payload } if *tag != 42 || !payload.is_empty())) {
            return Err(PecosError::Input("unsupported synthetic metadata".into()));
        }
        Ok(())
    }
    fn translate(
        &mut self,
        batch: &ScheduledEventBatch,
        out: &mut ScheduledGateBuffer<'_>,
    ) -> Result<(), PecosError> {
        for op in &batch.operations {
            if let ScheduledEventOp::Gate(g) = op {
                out.push(*g.clone())?;
            }
        }
        Ok(())
    }
}
fn message(batches: &[TimedBatch], events: bool) -> ByteMessage {
    if !events {
        return encode_timed_batches(batches).unwrap();
    }
    let batches: Vec<_> = batches
        .iter()
        .map(|b| {
            let measurements = b
                .gates
                .iter()
                .enumerate()
                .filter_map(|(i, g)| {
                    matches!(g.gate_type, GateType::MZ | GateType::MeasureLeaked).then_some(
                        ScheduledResult {
                            operation_index: i,
                            runtime_result: b.batch_index * 10 + i as u64,
                            program_result: b.batch_index * 10 + i as u64 + 100,
                        },
                    )
                })
                .collect();
            let mut operations: Vec<_> = b
                .gates
                .iter()
                .cloned()
                .map(|g| ScheduledEventOp::Gate(Box::new(g)))
                .collect();
            {
                operations.push(ScheduledEventOp::Custom {
                    tag: 42,
                    payload: vec![],
                });
            }
            ScheduledEventBatch {
                runtime_shot_id: b.runtime_shot_id,
                batch_index: b.batch_index,
                start_nanos: b.start_nanos,
                duration_nanos: b.duration_nanos,
                operations,
                measurements,
            }
        })
        .collect();
    encode_event_batches(&batches).unwrap()
}
fn system(profile: ScheduledIdleNoise, events: bool, seed: u64) -> QuantumSystem {
    let noise = if events {
        ScheduledEventIdleNoise::new(profile, |_| Ok(Box::new(PassThrough)))
            .into_noise_model()
            .unwrap()
    } else {
        profile.into_noise_model().unwrap()
    };
    let mut system = QuantumSystem::new(noise, Box::new(StateVecEngine::new(2)));
    system.set_seed(seed);
    system.begin_shot(context(0)).unwrap();
    system
}
fn run(system: &mut QuantumSystem, batches: &[TimedBatch], events: bool) -> Vec<u32> {
    system
        .process(message(batches, events))
        .unwrap()
        .outcomes()
        .unwrap()
}
fn profile(p1: f64, p2: f64, prep: f64, meas0: f64, meas1: f64) -> ScheduledIdleNoise {
    ScheduledLocalNoise::new(
        ScheduledIdleNoise::new(2).unwrap(),
        p1,
        p2,
        prep,
        meas0,
        meas1,
    )
    .unwrap()
    .into()
}
#[test]
fn invalid_probabilities_reject_before_model_construction() {
    for bad in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        for index in 0..5 {
            let mut p = [0.0; 5];
            p[index] = bad;
            assert!(
                ScheduledLocalNoise::new(
                    ScheduledIdleNoise::new(2).unwrap(),
                    p[0],
                    p[1],
                    p[2],
                    p[3],
                    p[4]
                )
                .is_err()
            );
        }
    }
}
#[test]
fn preparation_and_asymmetric_readout_are_applied_once() {
    for events in [false, true] {
        for (prep, m0, m1, want) in [(1.0, 0.0, 0.0, 1), (0.0, 1.0, 0.0, 1), (1.0, 0.0, 1.0, 0)] {
            for measurement in [
                Gate::mz(&[0]),
                Gate::simple(GateType::MeasureLeaked, vec![0.into()]),
            ] {
                let mut sim = system(profile(0.0, 0.0, prep, m0, m1), events, 9);
                let batches = [
                    batch(0, 0, 0, vec![Gate::pz(&[0])]),
                    batch(1, 0, 0, vec![measurement]),
                ];
                assert_eq!(run(&mut sim, &batches, events), vec![want]);
                sim.reset().unwrap();
                sim.begin_shot(context(1)).unwrap();
                assert_eq!(run(&mut sim, &batches, events), vec![want]);
            }
        }
    }
}
#[test]
fn local_gate_faults_match_explicit_general_noise_and_rng_per_native_batch() {
    let gates = vec![
        Gate::rxy1q(Angle64::from_radians(0.7), Angle64::ZERO, &[0]),
        Gate::rzz(Angle64::from_radians(-0.3), &[(0, 1)]),
        Gate::rz(Angle64::from_radians(0.4), &[0]),
    ];
    let batches = [
        batch(0, 0, 0, vec![Gate::pz(&[0]), Gate::pz(&[1])]),
        batch(1, 0, 0, gates),
        batch(2, 0, 0, vec![Gate::mz(&[0]), Gate::mz(&[1])]),
    ];
    for events in [false, true] {
        for (seed, ratio) in (0..32).flat_map(|seed| [0.0, 0.4, 1.0].map(|ratio| (seed, ratio))) {
            let local = ScheduledLocalNoise::new(
                ScheduledIdleNoise::new(2).unwrap(),
                0.8,
                0.7,
                0.3,
                0.4,
                0.2,
            )
            .unwrap()
            .with_prep_leak_ratio(ratio)
            .unwrap();
            let mut actual = system(local.into(), events, seed);
            let mut expected = QuantumSystem::new(
                Box::new(
                    GeneralNoiseModelBuilder::new()
                        .with_p1(0.8)
                        .with_p2(0.7)
                        .with_p_prep(0.3)
                        .with_prep_leak_ratio(ratio)
                        .with_p_meas_0(0.4)
                        .with_p_meas_1(0.2)
                        .build(),
                ),
                Box::new(StateVecEngine::new(2)),
            );
            expected.set_seed(seed);
            for b in &batches {
                let want = expected
                    .process(
                        ByteMessage::quantum_operations_builder()
                            .add_gate_commands(&b.gates)
                            .build(),
                    )
                    .unwrap();
                assert_eq!(
                    run(&mut actual, std::slice::from_ref(b), events),
                    want.outcomes().unwrap()
                );
                assert_eq!(
                    actual.controller().rng().clone().next_u64(),
                    expected.controller().rng().clone().next_u64()
                );
            }
        }
    }
}
#[test]
fn zero_local_faults_preserve_idle_results_rng_and_input_grouping() {
    let idle = ScheduledIdleNoise::new(2)
        .unwrap()
        .with_linear(0.4, model(&[("Z", 0.5), ("L", 0.5)]))
        .unwrap()
        .with_coherent(0.3, None)
        .unwrap();
    let batches = [
        batch(0, 0, 0, vec![Gate::pz(&[0])]),
        batch(
            1,
            1_000_000_000,
            0,
            vec![Gate::rxy1q(Angle64::from_radians(0.7), Angle64::ZERO, &[0])],
        ),
        batch(2, 2_000_000_000, 0, vec![Gate::mz(&[0])]),
    ];
    for events in [false, true] {
        for seed in 0..32 {
            let local: ScheduledIdleNoise =
                ScheduledLocalNoise::new(idle.clone(), 0.0, 0.0, 0.0, 0.0, 0.0)
                    .unwrap()
                    .into();
            let mut a = system(local.clone(), events, seed);
            let mut b = system(idle.clone(), events, seed);
            let mut split = system(local, events, seed);
            let want = run(&mut b, &batches, events);
            assert_eq!(run(&mut a, &batches, events), want);
            let got: Vec<_> = batches
                .iter()
                .flat_map(|b| run(&mut split, std::slice::from_ref(b), events))
                .collect();
            assert_eq!(got, want);
            assert_eq!(
                a.controller().rng().clone().next_u64(),
                b.controller().rng().clone().next_u64()
            );
            assert_eq!(
                a.controller().rng().clone().next_u64(),
                split.controller().rng().clone().next_u64()
            );
        }
    }
}
#[test]
fn leakage_readout_precedes_later_noisy_preparation() {
    let idle = ScheduledIdleNoise::new(2)
        .unwrap()
        .with_linear(1.0, model(&[("L", 1.0)]))
        .unwrap();
    for events in [false, true] {
        let local = ScheduledLocalNoise::new(idle.clone(), 0.0, 0.0, 1.0, 0.0, 1.0).unwrap();
        let mut sim = system(local.into(), events, 7);
        let leaked = || Gate::simple(GateType::MeasureLeaked, vec![0.into()]);
        let b = [
            batch(0, 0, 0, vec![Gate::pz(&[0])]),
            batch(
                1,
                1_000_000_000,
                0,
                vec![leaked(), Gate::mz(&[0]), Gate::pz(&[0]), leaked()],
            ),
        ];
        // Ternary leakage bypasses readout flips. Binary leaked=1 is flipped;
        // the reset clears leakage, then prep makes 1 and readout flips it to 0.
        assert_eq!(run(&mut sim, &b, events), vec![2, 0, 0]);
    }
}
#[test]
fn same_batch_reuse_preserves_gate_order_readout_and_rng() {
    // The legacy model is a valid oracle here because these faults cannot leak.
    // Comparing the RNG catches an accidental split into multiple lifecycles.
    let gates = vec![
        Gate::pz(&[0]),
        Gate::mz(&[0]),
        Gate::mz(&[0]),
        Gate::rxy1q(Angle64::from_radians(0.7), Angle64::ZERO, &[0]),
        Gate::rzz(Angle64::from_radians(-0.3), &[(0, 1)]),
        Gate::mz(&[1]),
        Gate::pz(&[0]),
        Gate::measure_leaked(&[0]),
    ];
    for events in [false, true] {
        for seed in 0..32 {
            let mut actual = system(profile(0.8, 0.7, 0.3, 0.4, 0.2), events, seed);
            let mut expected = QuantumSystem::new(
                Box::new(
                    GeneralNoiseModelBuilder::new()
                        .with_p1(0.8)
                        .with_p2(0.7)
                        .with_p_prep(0.3)
                        .with_p_meas_0(0.4)
                        .with_p_meas_1(0.2)
                        .build(),
                ),
                Box::new(StateVecEngine::new(2)),
            );
            expected.set_seed(seed);
            let want = expected
                .process(
                    ByteMessage::quantum_operations_builder()
                        .add_gate_commands(&gates)
                        .build(),
                )
                .unwrap();
            assert_eq!(
                run(&mut actual, &[batch(0, 0, 0, gates.clone())], events),
                want.outcomes().unwrap()
            );
            assert_eq!(
                actual.controller().rng().clone().next_u64(),
                expected.controller().rng().clone().next_u64()
            );
        }
    }
}

#[test]
fn oversized_batch_rejected_before_execution() {
    let mut sim = system(profile(1.0, 1.0, 1.0, 1.0, 1.0), false, 0);
    let invalid = [batch(0, 0, 0, vec![Gate::pz(&[0]); 4097])];
    assert!(sim.process(message(&invalid, false)).is_err());
    assert_eq!(
        run(&mut sim, &[batch(0, 0, 0, vec![Gate::mz(&[0])])], false),
        vec![1]
    );
}

#[test]
fn normalized_capacity_violation_poison_clone_and_reset() {
    struct AppendOutsideCapacity;
    impl ScheduledBatchAdapter for AppendOutsideCapacity {
        fn validate(&self, _: &ScheduledEventBatch) -> Result<(), PecosError> {
            Ok(())
        }
        fn translate(
            &mut self,
            b: &ScheduledEventBatch,
            out: &mut ScheduledGateBuffer<'_>,
        ) -> Result<(), PecosError> {
            for op in &b.operations {
                if let ScheduledEventOp::Gate(g) = op {
                    out.push(*g.clone())?;
                }
            }
            if b.batch_index == 0
                && matches!(&b.operations[0], ScheduledEventOp::Gate(g) if g.gate_type == GateType::MZ)
            {
                out.push(Gate::pz(&[2]))?;
            }
            Ok(())
        }
    }
    // A certain preparation fault reads 1 when it applies X and 2 when it leaks.
    for (ratio, expected) in [(0.0, 1), (1.0, 2)] {
        let local =
            ScheduledLocalNoise::new(ScheduledIdleNoise::new(2).unwrap(), 0.0, 0.0, 1.0, 0.0, 0.0)
                .unwrap()
                .with_prep_leak_ratio(ratio)
                .unwrap();
        let noise = ScheduledEventIdleNoise::new(local, |_| Ok(Box::new(AppendOutsideCapacity)));
        let mut sim = QuantumSystem::new(
            noise.into_noise_model().unwrap(),
            Box::new(StateVecEngine::new(2)),
        );
        sim.begin_shot(context(0)).unwrap();
        let bad = message(&[batch(0, 0, 0, vec![Gate::mz(&[0])])], true);
        assert!(sim.process(bad).is_err());
        let mut cloned = sim.clone();
        for owner in [&mut sim, &mut cloned] {
            assert!(owner.process(ByteMessage::create_empty()).is_err());
            owner.reset().unwrap();
            owner.begin_shot(context(1)).unwrap();
            assert_eq!(
                run(
                    owner,
                    &[
                        batch(0, 0, 0, vec![Gate::pz(&[0])]),
                        batch(1, 0, 0, vec![Gate::measure_leaked(&[0])])
                    ],
                    true
                ),
                vec![expected]
            );
        }
    }
}

#[test]
fn preparation_leakage_ratio_is_checked_even_without_preparation_faults() {
    for ratio in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let local =
            ScheduledLocalNoise::new(ScheduledIdleNoise::new(2).unwrap(), 0.0, 0.0, 0.0, 0.0, 0.0)
                .unwrap();
        assert!(local.with_prep_leak_ratio(ratio).is_err());
    }
}

#[test]
fn preparation_leakage_is_conditional_and_readout_preserves_the_ternary_result() {
    for events in [false, true] {
        for (prep, ratio, expected) in [
            (0.0, 1.0, vec![0, 0]),
            (1.0, 0.0, vec![0, 0]),
            (1.0, 1.0, vec![2, 0]),
        ] {
            let local = ScheduledLocalNoise::new(
                ScheduledIdleNoise::new(2).unwrap(),
                0.0,
                0.0,
                prep,
                0.0,
                1.0,
            )
            .unwrap()
            .with_prep_leak_ratio(ratio)
            .unwrap();
            let mut sim = system(local.into(), events, 4);
            let b = [batch(
                0,
                0,
                0,
                vec![Gate::pz(&[0]), Gate::measure_leaked(&[0]), Gate::mz(&[0])],
            )];
            // Binary 1 (whether leaked or an X fault) flips to zero; ternary 2 does not.
            assert_eq!(run(&mut sim, &b, events), expected);
        }
    }
}

#[test]
fn preparation_leakage_same_batch_reset_snapshots_match_general_readout() {
    let gates = vec![
        Gate::pz(&[0]),
        Gate::measure_leaked(&[0]),
        Gate::mz(&[0]),
        Gate::pz(&[0]),
        Gate::measure_leaked(&[0]),
        Gate::pz(&[0]),
        Gate::measure_leaked(&[0]),
    ];
    for events in [false, true] {
        let mut saw_clear = false;
        let mut saw_leak = false;
        for seed in 0..32 {
            let local = ScheduledLocalNoise::new(
                ScheduledIdleNoise::new(2).unwrap(),
                0.0,
                0.0,
                0.5,
                0.0,
                0.0,
            )
            .unwrap()
            .with_prep_leak_ratio(1.0)
            .unwrap();
            let mut actual = system(local.into(), events, seed);
            let mut expected = QuantumSystem::new(
                Box::new(
                    GeneralNoiseModelBuilder::new()
                        .with_p_prep(0.5)
                        .with_prep_leak_ratio(1.0)
                        .with_p_meas_1(0.0)
                        .build(),
                ),
                Box::new(StateVecEngine::new(2)),
            );
            expected.set_seed(seed);
            // Preserve the complete native lifecycle, including the order of
            // preparation sampling and deferred readout RNG draws. The general
            // model records leakage separately at each measurement position.
            let want = expected
                .process(
                    ByteMessage::quantum_operations_builder()
                        .add_gate_commands(&gates)
                        .build(),
                )
                .unwrap()
                .outcomes()
                .unwrap();
            let got = run(&mut actual, &[batch(0, 0, 0, gates.clone())], events);
            assert_eq!(got, want);
            assert_eq!(
                actual.controller().rng().clone().next_u64(),
                expected.controller().rng().clone().next_u64()
            );
            saw_clear |= got[0] == 2 && got[2] == 0;
            saw_leak |= got[0] == 0 && got[2] == 2;
        }
        assert!(
            saw_clear && saw_leak,
            "exercise both reset clearing and new leakage"
        );
    }
}

#[test]
fn whole_shot_reset_clears_preparation_leakage_in_cloned_hosts() {
    for events in [false, true] {
        let local =
            ScheduledLocalNoise::new(ScheduledIdleNoise::new(2).unwrap(), 0.0, 0.0, 1.0, 0.0, 0.0)
                .unwrap()
                .with_prep_leak_ratio(1.0)
                .unwrap();
        let mut sim = system(local.into(), events, 3);
        assert_eq!(
            run(
                &mut sim,
                &[batch(
                    0,
                    0,
                    0,
                    vec![Gate::pz(&[0]), Gate::measure_leaked(&[0])]
                )],
                events
            ),
            vec![2]
        );
        let mut cloned = sim.clone();
        for host in [&mut sim, &mut cloned] {
            host.reset().unwrap();
            host.begin_shot(context(1)).unwrap();
            assert_eq!(
                run(
                    host,
                    &[batch(0, 0, 0, vec![Gate::measure_leaked(&[0])])],
                    events
                ),
                vec![0]
            );
            assert_eq!(
                run(
                    host,
                    &[batch(
                        1,
                        0,
                        0,
                        vec![Gate::pz(&[0]), Gate::measure_leaked(&[0])]
                    )],
                    events
                ),
                vec![2]
            );
        }
    }
}

#[test]
fn replacing_local_faults_clears_old_preparation_leakage_but_retains_idle_noise() {
    for events in [false, true] {
        let idle = ScheduledIdleNoise::new(2)
            .unwrap()
            .with_linear(1e9, model(&[("X", 1.0)]))
            .unwrap();
        let original = ScheduledLocalNoise::new(idle, 0.0, 0.0, 1.0, 0.0, 0.0)
            .unwrap()
            .with_prep_leak_ratio(1.0)
            .unwrap();
        let replacement =
            ScheduledLocalNoise::new(original.clone().into(), 0.0, 0.0, 1.0, 0.0, 0.0).unwrap();
        let batches = [batch(
            0,
            1_000_000_000,
            0,
            vec![Gate::pz(&[0]), Gate::measure_leaked(&[0]), Gate::mz(&[1])],
        )];
        // Preparation replaces qubit 0's idle history. Qubit 1 still receives
        // the retained idle channel, even after replacing the local profile.
        let mut old = system(original.into(), events, 9);
        let mut new = system(replacement.into(), events, 9);
        assert_eq!(run(&mut old, &batches, events), vec![2, 1]);
        assert_eq!(run(&mut new, &batches, events), vec![1, 1]);
    }
}

#[test]
fn invalid_two_qubit_distributions_reject_even_with_zero_event_probability() {
    let mut invalid = vec![
        BTreeMap::new(),
        model(&[("XI", 0.0)]),
        model(&[("XI", 0.5)]),
    ];
    for key in ["II", "XL", "LX", "xx", "X", "XYZ", "é", " X"] {
        invalid.push(model(&[(key, 0.0), ("XX", 1.0)]));
    }
    for weight in [-0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, f64::MAX] {
        invalid.push(model(&[("XI", weight), ("IX", 1.0)]));
    }
    // Sums to one, so only the negative-weight check rejects it.
    invalid.push(model(&[("XI", -0.1), ("IX", 1.1)]));
    for distribution in invalid {
        let local =
            ScheduledLocalNoise::new(ScheduledIdleNoise::new(2).unwrap(), 0.0, 0.0, 0.0, 0.0, 0.0)
                .unwrap();
        assert!(local.with_p2_pauli_model(distribution).is_err());
    }
}

#[test]
fn ordered_pauli_pairs_act_on_emitted_operands_in_all_measurement_bases() {
    for events in [false, true] {
        for reverse in [false, true] {
            let pair = if reverse { (1, 0) } else { (0, 1) };
            for interaction in [
                Gate::rzz(Angle64::ZERO, &[pair]),
                Gate::rxyxy2q(Angle64::ZERO, Angle64::from_radians(0.4), &[pair]),
            ] {
                for a in ['I', 'X', 'Y', 'Z'] {
                    for b in ['I', 'X', 'Y', 'Z'] {
                        if a == 'I' && b == 'I' {
                            continue;
                        }
                        for basis in ['X', 'Y', 'Z'] {
                            let local = ScheduledLocalNoise::new(
                                ScheduledIdleNoise::new(2).unwrap(),
                                0.0,
                                1.0,
                                0.0,
                                0.0,
                                0.0,
                            )
                            .unwrap()
                            .with_p2_pauli_model(model(&[(&format!("{a}{b}"), 1.0)]))
                            .unwrap();
                            let mut sim = system(local.into(), events, 9);
                            let mut gates = vec![Gate::pz(&[0]), Gate::pz(&[1])];
                            let phi = if basis == 'X' {
                                std::f64::consts::FRAC_PI_2
                            } else {
                                0.0
                            };
                            if basis != 'Z' {
                                for q in [0, 1] {
                                    gates.push(Gate::rxy1q(
                                        Angle64::from_radians(std::f64::consts::FRAC_PI_2),
                                        Angle64::from_radians(phi),
                                        &[q],
                                    ));
                                }
                            }
                            gates.push(interaction.clone());
                            if basis != 'Z' {
                                for q in [0, 1] {
                                    gates.push(Gate::rxy1q(
                                        Angle64::from_radians(-std::f64::consts::FRAC_PI_2),
                                        Angle64::from_radians(phi),
                                        &[q],
                                    ));
                                }
                            }
                            gates.extend([Gate::mz(&[0]), Gate::mz(&[1])]);
                            // A Pauli flips the measured eigenvalue exactly when it
                            // anticommutes with the chosen measurement basis.
                            let mut want = vec![
                                u32::from(a != 'I' && a != basis),
                                u32::from(b != 'I' && b != basis),
                            ];
                            if reverse {
                                want.reverse();
                            }
                            assert_eq!(
                                run(&mut sim, &[batch(0, 0, 0, gates)], events),
                                want,
                                "pair={a}{b}, basis={basis}, reversed={reverse}, gate={:?}",
                                interaction.gate_type
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn nonuniform_pairs_match_whole_batch_general_model_results_and_rng() {
    let distribution = model(&[("IX", 0.15), ("XI", 0.25), ("YZ", 0.6), ("ZZ", 0.0)]);
    let batches = [
        batch(0, 0, 0, vec![Gate::pz(&[0]), Gate::pz(&[1])]),
        batch(
            1,
            0,
            0,
            vec![
                Gate::rxy1q(Angle64::from_radians(0.7), Angle64::ZERO, &[0]),
                Gate::rzz(Angle64::from_radians(0.3), &[(1, 0)]),
                Gate::mz(&[0]),
                Gate::pz(&[0]),
                Gate::rxyxy2q(
                    Angle64::from_radians(-0.6),
                    Angle64::from_radians(0.2),
                    &[(0, 1)],
                ),
                Gate::measure_leaked(&[1]),
                Gate::mz(&[0]),
            ],
        ),
    ];
    for events in [false, true] {
        for seed in 0..32 {
            let local = ScheduledLocalNoise::new(
                ScheduledIdleNoise::new(2).unwrap(),
                0.1,
                0.8,
                0.2,
                0.1,
                0.3,
            )
            .unwrap()
            .with_prep_leak_ratio(0.4)
            .unwrap()
            .with_p2_pauli_model(distribution.clone())
            .unwrap();
            let mut actual = system(local.into(), events, seed);
            let mut expected = QuantumSystem::new(
                Box::new(
                    GeneralNoiseModelBuilder::new()
                        .with_p1(0.1)
                        .with_p2(0.8)
                        .with_p_prep(0.2)
                        .with_prep_leak_ratio(0.4)
                        .with_p_meas_0(0.1)
                        .with_p_meas_1(0.3)
                        .with_p2_pauli_model(&distribution)
                        .build(),
                ),
                Box::new(StateVecEngine::new(2)),
            );
            expected.set_seed(seed);
            for b in &batches {
                let want = expected
                    .process(
                        ByteMessage::quantum_operations_builder()
                            .add_gate_commands(&b.gates)
                            .build(),
                    )
                    .unwrap()
                    .outcomes()
                    .unwrap();
                assert_eq!(run(&mut actual, std::slice::from_ref(b), events), want);
                assert_eq!(
                    actual.controller().rng().clone().next_u64(),
                    expected.controller().rng().clone().next_u64()
                );
            }
        }
    }
}

#[test]
fn replacing_pair_distribution_preserves_clones_and_other_faults() {
    for events in [false, true] {
        let original =
            ScheduledLocalNoise::new(ScheduledIdleNoise::new(2).unwrap(), 0.0, 1.0, 1.0, 0.0, 0.0)
                .unwrap()
                .with_prep_leak_ratio(1.0)
                .unwrap()
                .with_p2_pauli_model(model(&[("XI", 1.0)]))
                .unwrap();
        let changed = original
            .clone()
            .with_p2_pauli_model(model(&[("IX", 1.0)]))
            .unwrap();
        let batches = [batch(
            0,
            0,
            0,
            vec![
                Gate::rzz(Angle64::ZERO, &[(0, 1)]),
                Gate::mz(&[0]),
                Gate::mz(&[1]),
                Gate::pz(&[0]),
                Gate::measure_leaked(&[0]),
            ],
        )];
        assert_eq!(
            run(&mut system(original.into(), events, 9), &batches, events),
            vec![1, 0, 2]
        );
        assert_eq!(
            run(&mut system(changed.into(), events, 9), &batches, events),
            vec![0, 1, 2]
        );
    }
}

#[test]
fn replacing_local_profile_restores_default_two_qubit_sampling() {
    let custom =
        ScheduledLocalNoise::new(ScheduledIdleNoise::new(2).unwrap(), 0.0, 1.0, 0.0, 0.0, 0.0)
            .unwrap()
            .with_p2_pauli_model(model(&[("XI", 1.0)]))
            .unwrap();
    let replaced = ScheduledLocalNoise::new(custom.into(), 0.0, 0.8, 0.0, 0.0, 0.0).unwrap();
    let batches = [batch(
        0,
        0,
        0,
        vec![
            Gate::pz(&[0]),
            Gate::pz(&[1]),
            Gate::rzz(Angle64::ZERO, &[(0, 1)]),
            Gate::mz(&[0]),
            Gate::mz(&[1]),
        ],
    )];
    for events in [false, true] {
        for seed in 0..32 {
            let mut actual = system(replaced.clone().into(), events, seed);
            let mut expected = system(profile(0.0, 0.8, 0.0, 0.0, 0.0), events, seed);
            assert_eq!(
                run(&mut actual, &batches, events),
                run(&mut expected, &batches, events)
            );
            assert_eq!(
                actual.controller().rng().clone().next_u64(),
                expected.controller().rng().clone().next_u64()
            );
        }
    }
}
