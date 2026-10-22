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
        Gate::rxy1q(pecos_core::radians!(0.7), Angle64::ZERO, &[0]),
        Gate::rzz(pecos_core::radians!(-0.3), &[(0, 1)]),
        Gate::rz(pecos_core::radians!(0.4), &[0]),
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
            vec![Gate::rxy1q(pecos_core::radians!(0.7), Angle64::ZERO, &[0])],
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
        Gate::rxy1q(pecos_core::radians!(0.7), Angle64::ZERO, &[0]),
        Gate::rzz(pecos_core::radians!(-0.3), &[(0, 1)]),
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
                Gate::rxyxy2q(Angle64::ZERO, pecos_core::radians!(0.4), &[pair]),
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
                                        Angle64::QUARTER_TURN,
                                        Angle64::try_from_radians(phi).unwrap(),
                                        &[q],
                                    ));
                                }
                            }
                            gates.push(interaction.clone());
                            if basis != 'Z' {
                                for q in [0, 1] {
                                    gates.push(Gate::rxy1q(
                                        Angle64::THREE_QUARTERS_TURN,
                                        Angle64::try_from_radians(phi).unwrap(),
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
                Gate::rxy1q(pecos_core::radians!(0.7), Angle64::ZERO, &[0]),
                Gate::rzz(pecos_core::radians!(0.3), &[(1, 0)]),
                Gate::mz(&[0]),
                Gate::pz(&[0]),
                Gate::rxyxy2q(
                    pecos_core::radians!(-0.6),
                    pecos_core::radians!(0.2),
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
            .unwrap()
            .with_p2_angle_params(0.0, 0.0, 0.0, 0.0)
            .unwrap()
            .with_p2_angle_power(2.0)
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

#[test]
fn angle_scaling_preserves_mixed_faults_whole_batches_and_rng() {
    let distribution = model(&[("IX", 0.15), ("XI", 0.25), ("YZ", 0.6), ("ZZ", 0.0)]);
    let batches = [
        batch(0, 0, 0, vec![Gate::pz(&[0]), Gate::pz(&[1])]),
        batch(
            1,
            0,
            0,
            vec![
                Gate::rxy1q(pecos_core::radians!(0.7), Angle64::ZERO, &[0]),
                Gate::rzz(pecos_core::radians!(0.3), &[(1, 0)]),
                Gate::rzz(Angle64::ZERO, &[(0, 1)]),
                Gate::rzz(pecos_core::radians!(4.3), &[(1, 0)]),
                Gate::mz(&[0]),
                Gate::pz(&[0]),
                Gate::rxyxy2q(
                    pecos_core::radians!(-0.6),
                    pecos_core::radians!(0.2),
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
            .unwrap()
            .with_p2_angle_params(0.7, 0.1, -0.3, 0.8)
            .unwrap()
            .with_p2_angle_power(1.7)
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
                        .with_p2_angle_params(0.7, 0.1, -0.3, 0.8)
                        .with_p2_angle_power(1.7)
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
fn angle_admission_reuses_combined_general_model_validation() {
    let local = |p2| {
        ScheduledLocalNoise::new(ScheduledIdleNoise::new(2).unwrap(), 0.0, p2, 0.0, 0.0, 0.0)
            .unwrap()
    };
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        for p2 in [0.0, 1.0] {
            for slot in 0..4 {
                let mut coefficients = [0.0, 1.0, 0.0, 1.0];
                coefficients[slot] = bad;
                let [a, b, c, d] = coefficients;
                assert!(local(p2).with_p2_angle_params(a, b, c, d).is_err());
            }
        }
    }
    for power in [0.0, -1.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(local(0.0).with_p2_angle_power(power).is_err());
    }
    for (a, b, c, d) in [
        (0.0, -0.1, 0.0, 1.0),
        (1.0, 0.1, 0.0, 1.0),
        (0.0, 0.0, -1.0, 0.5),
        (0.0, 0.0, 0.0, 1.1),
    ] {
        assert!(local(1.0).with_p2_angle_params(a, b, c, d).is_err());
    }
    // Coefficients need not themselves be probabilities: scaled endpoints are.
    assert!(local(0.5).with_p2_angle_params(-1.0, 2.0, 1.5, 0.5).is_ok());
    assert!(
        local(0.0)
            .with_p2_angle_params(f64::MAX, f64::MAX, 0.0, 0.0)
            .is_err()
    );
}

#[test]
fn angle_branches_follow_emitted_signed_angles_for_both_transports() {
    use std::f64::consts::PI;
    for events in [false, true] {
        for xy in [false, true] {
            for (angle, negative) in [
                (-1.5 * PI, false),
                (-PI, false),
                (-PI / 2.0, true),
                (PI / 2.0, false),
                (PI, false),
                (1.5 * PI, true),
            ] {
                for negative_only in [false, true] {
                    let (b, d) = if negative_only {
                        (1.0, 0.0)
                    } else {
                        (0.0, 1.0)
                    };
                    let local = ScheduledLocalNoise::new(
                        ScheduledIdleNoise::new(2).unwrap(),
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                    .unwrap()
                    .with_p2_angle_params(0.0, b, 0.0, d)
                    .unwrap()
                    .with_p2_angle_power(2.0)
                    .unwrap()
                    .with_p2_pauli_model(model(&[("XI", 1.0)]))
                    .unwrap();
                    let a = Angle64::try_from_radians(angle).unwrap();
                    let gate = if xy {
                        Gate::rxyxy2q(a, Angle64::ZERO, &[(0, 1)])
                    } else {
                        Gate::rzz(a, &[(0, 1)])
                    };
                    let mut sim = system(local.into(), events, 17);
                    let values = run(
                        &mut sim,
                        &[batch(
                            0,
                            0,
                            0,
                            vec![
                                Gate::pz(&[0]),
                                Gate::pz(&[1]),
                                gate,
                                Gate::mz(&[0]),
                                Gate::mz(&[1]),
                            ],
                        )],
                        events,
                    );
                    assert_eq!(values[0] ^ values[1], u32::from(negative == negative_only));
                }
            }
        }
    }
}

#[test]
fn explicit_constant_angle_defaults_preserve_seeded_behavior() {
    for events in [false, true] {
        for seed in 0..32 {
            let make = || {
                ScheduledLocalNoise::new(
                    ScheduledIdleNoise::new(2).unwrap(),
                    0.2,
                    0.6,
                    0.1,
                    0.2,
                    0.3,
                )
                .unwrap()
            };
            let mut original = system(make().into(), events, seed);
            let explicit = make()
                .with_p2_angle_params(0.0, 1.0, 0.0, 1.0)
                .unwrap()
                .with_p2_angle_power(1.0)
                .unwrap();
            let mut changed = system(explicit.clone().into(), events, seed);
            let mut cloned = system(explicit.into(), events, seed);
            let batches = [batch(
                0,
                0,
                0,
                vec![
                    Gate::pz(&[0]),
                    Gate::pz(&[1]),
                    Gate::rzz(Angle64::ZERO, &[(0, 1)]),
                    Gate::rzz(pecos_core::radians!(-0.7), &[(1, 0)]),
                    Gate::mz(&[0]),
                    Gate::mz(&[1]),
                ],
            )];
            let want = run(&mut original, &batches, events);
            for host in [&mut changed, &mut cloned] {
                assert_eq!(run(host, &batches, events), want);
                assert_eq!(
                    host.controller().rng().clone().next_u64(),
                    original.controller().rng().clone().next_u64()
                );
            }
        }
    }
}

#[test]
fn emission_configuration_is_checked_even_when_gate_faults_are_disabled() {
    let fresh = || {
        ScheduledLocalNoise::new(ScheduledIdleNoise::new(2).unwrap(), 0.0, 0.0, 0.0, 0.0, 0.0)
            .unwrap()
    };
    for value in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert!(fresh().with_p1_emission_ratio(value).is_err());
        assert!(fresh().with_p2_emission_ratio(value).is_err());
        assert!(fresh().with_p1_seepage_prob(value).is_err());
        assert!(fresh().with_p2_seepage_prob(value).is_err());
    }
    for entries in [
        vec![],
        vec![("I", 1.0)],
        vec![("XX", 1.0)],
        vec![("L", 0.0)],
        vec![("L", 0.7)],
        vec![("L", f64::NAN)],
        vec![("X", -0.5), ("L", 1.5)],
    ] {
        assert!(fresh().with_p1_emission_model(model(&entries)).is_err());
    }
    for entries in [
        vec![],
        vec![("II", 1.0)],
        vec![("L", 1.0)],
        vec![("LZQ", 1.0)],
        vec![("LL", 0.0)],
        vec![("LL", 1.1)],
        vec![("LL", f64::INFINITY)],
        vec![("XI", -0.5), ("IL", 1.5)],
    ] {
        assert!(fresh().with_p2_emission_model(model(&entries)).is_err());
    }
    fresh()
        .with_p1_emission_model(model(&[("L", 1.0)]))
        .unwrap()
        .with_p2_emission_model(model(&[("LL", 1.0)]))
        .unwrap()
        .with_p1_seepage_prob(1.0)
        .unwrap()
        .with_p2_seepage_prob(1.0)
        .unwrap();
}

#[test]
fn single_qubit_emission_replaces_the_gate_and_preserves_measurement_time_leakage() {
    for events in [false, true] {
        for gate in [
            Gate::rxy1q(Angle64::HALF_TURN, Angle64::ZERO, &[0]),
            Gate::rz(Angle64::HALF_TURN, &[0]),
        ] {
            for (key, expected) in [("Z", vec![0, 0, 0]), ("L", vec![2, 1, 0])] {
                let local = ScheduledLocalNoise::new(
                    ScheduledIdleNoise::new(2).unwrap(),
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
                .unwrap()
                .with_p1_emission_ratio(1.0)
                .unwrap()
                .with_p1_emission_model(model(&[(key, 1.0)]))
                .unwrap();
                let mut sim = system(local.into(), events, 7);
                let b = [batch(
                    0,
                    0,
                    0,
                    vec![
                        Gate::pz(&[0]),
                        gate.clone(),
                        Gate::measure_leaked(&[0]),
                        Gate::mz(&[0]),
                        Gate::pz(&[0]),
                        Gate::measure_leaked(&[0]),
                    ],
                )];
                assert_eq!(run(&mut sim, &b, events), expected);
                sim.reset().unwrap();
                sim.begin_shot(context(1)).unwrap();
                assert_eq!(run(&mut sim, &b, events), expected);
            }
        }
    }
}

#[test]
fn two_qubit_emission_uses_operand_order_and_seepage_is_conditional() {
    for events in [false, true] {
        for reversed in [false, true] {
            for (key, first) in [("LX", vec![2, 1]), ("XL", vec![1, 2]), ("LL", vec![2, 2])] {
                for seepage in [0.0, 1.0] {
                    let (a, b) = if reversed { (1, 0) } else { (0, 1) };
                    let local = ScheduledLocalNoise::new(
                        ScheduledIdleNoise::new(2).unwrap(),
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                    .unwrap()
                    .with_p2_emission_ratio(1.0)
                    .unwrap()
                    .with_p2_emission_model(model(&[(key, 1.0)]))
                    .unwrap()
                    .with_p2_seepage_prob(seepage)
                    .unwrap();
                    let mut sim = system(local.into(), events, 17);
                    let batches = [
                        batch(
                            0,
                            0,
                            0,
                            vec![
                                Gate::pz(&[0]),
                                Gate::pz(&[1]),
                                Gate::rzz(Angle64::ZERO, &[(a, b)]),
                                Gate::measure_leaked(&[a]),
                                Gate::measure_leaked(&[b]),
                            ],
                        ),
                        batch(
                            1,
                            0,
                            0,
                            vec![
                                Gate::rxyxy2q(Angle64::ZERO, Angle64::ZERO, &[(a, b)]),
                                Gate::measure_leaked(&[a]),
                                Gate::measure_leaked(&[b]),
                            ],
                        ),
                    ];
                    assert_eq!(run(&mut sim, &batches[..1], events), first);
                    let second = run(&mut sim, &batches[1..], events);
                    for i in 0..2 {
                        if first[i] == 2 && seepage > 0.0 {
                            assert!(second[i] < 2);
                        } else {
                            assert_eq!(second[i], first[i]);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn single_qubit_seepage_observes_idle_leakage_and_measurement_reset_order() {
    for events in [false, true] {
        for seepage in [0.0, 1.0] {
            let idle = ScheduledIdleNoise::new(2)
                .unwrap()
                .with_linear(1.0, model(&[("L", 1.0)]))
                .unwrap();
            let local = ScheduledLocalNoise::new(idle, 1.0, 0.0, 0.0, 0.0, 0.0)
                .unwrap()
                .with_p1_emission_ratio(1.0)
                .unwrap()
                .with_p1_emission_model(model(&[("L", 1.0)]))
                .unwrap()
                .with_p1_seepage_prob(seepage)
                .unwrap();
            let mut sim = system(local.into(), events, 32);
            let batches = [
                batch(0, 0, 0, vec![Gate::pz(&[0])]),
                batch(
                    1,
                    1_000_000_000,
                    0,
                    vec![
                        Gate::measure_leaked(&[0]),
                        Gate::rxy1q(Angle64::ZERO, Angle64::ZERO, &[0]),
                        Gate::measure_leaked(&[0]),
                        Gate::pz(&[0]),
                        Gate::measure_leaked(&[0]),
                    ],
                ),
            ];
            let rows = run(&mut sim, &batches, events);
            assert_eq!(rows[0], 2);
            assert_eq!(rows[2], 0);
            if seepage == 0.0 {
                assert_eq!(rows[1], 2);
            } else {
                assert!(rows[1] < 2);
            }
        }
    }
}

#[test]
fn mixed_emission_and_angle_scaling_match_general_noise_results_and_rng() {
    let one = model(&[("X", 0.2), ("L", 0.8)]);
    let two = model(&[("LX", 0.3), ("ZL", 0.4), ("LL", 0.3)]);
    let batches = [
        batch(0, 0, 0, vec![Gate::pz(&[0]), Gate::pz(&[1])]),
        batch(
            1,
            0,
            0,
            vec![
                Gate::rxy1q(pecos_core::radians!(0.7), Angle64::ZERO, &[0]),
                Gate::rzz(pecos_core::radians!(-0.3), &[(1, 0)]),
                Gate::measure_leaked(&[0]),
                Gate::mz(&[1]),
                Gate::rxyxy2q(pecos_core::radians!(0.5), Angle64::ZERO, &[(0, 1)]),
                Gate::rz(pecos_core::radians!(0.9), &[1]),
                Gate::measure_leaked(&[1]),
                Gate::pz(&[0]),
                Gate::measure_leaked(&[0]),
            ],
        ),
    ];
    for events in [false, true] {
        for seed in 0..64 {
            let local = ScheduledLocalNoise::new(
                ScheduledIdleNoise::new(2).unwrap(),
                0.8,
                0.7,
                0.3,
                0.1,
                0.2,
            )
            .unwrap()
            .with_prep_leak_ratio(0.5)
            .unwrap()
            .with_p2_angle_params(0.4, 0.3, 0.6, 0.2)
            .unwrap()
            .with_p2_angle_power(2.0)
            .unwrap()
            .with_p1_emission_ratio(0.8)
            .unwrap()
            .with_p1_emission_model(one.clone())
            .unwrap()
            .with_p1_seepage_prob(0.7)
            .unwrap()
            .with_p2_emission_ratio(0.9)
            .unwrap()
            .with_p2_emission_model(two.clone())
            .unwrap()
            .with_p2_seepage_prob(0.6)
            .unwrap();
            let mut actual = system(local.into(), events, seed);
            let mut expected = QuantumSystem::new(
                Box::new(
                    GeneralNoiseModelBuilder::new()
                        .with_p1(0.8)
                        .with_p2(0.7)
                        .with_p_prep(0.3)
                        .with_prep_leak_ratio(0.5)
                        .with_p_meas_0(0.1)
                        .with_p_meas_1(0.2)
                        .with_p2_angle_params(0.4, 0.3, 0.6, 0.2)
                        .with_p2_angle_power(2.0)
                        .with_p1_emission_ratio(0.8)
                        .with_p1_emission_model(&one)
                        .with_p1_seepage_prob(0.7)
                        .with_p2_emission_ratio(0.9)
                        .with_p2_emission_model(&two)
                        .with_p2_seepage_prob(0.6)
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
fn emission_defaults_clones_and_local_replacement_preserve_behavior() {
    for events in [false, true] {
        let idle = ScheduledIdleNoise::new(2).unwrap();
        let plain = ScheduledLocalNoise::new(idle.clone(), 0.4, 0.6, 0.0, 0.0, 0.0).unwrap();
        let explicit = plain
            .clone()
            .with_p1_emission_ratio(0.0)
            .unwrap()
            .with_p2_emission_ratio(0.0)
            .unwrap()
            .with_p1_seepage_prob(0.0)
            .unwrap()
            .with_p2_seepage_prob(0.0)
            .unwrap();
        let emitting = plain
            .clone()
            .with_p1_emission_ratio(1.0)
            .unwrap()
            .with_p1_emission_model(model(&[("L", 1.0)]))
            .unwrap()
            .with_p2_emission_ratio(1.0)
            .unwrap()
            .with_p2_emission_model(model(&[("LL", 1.0)]))
            .unwrap()
            .with_p1_seepage_prob(1.0)
            .unwrap()
            .with_p2_seepage_prob(1.0)
            .unwrap();
        let replacement =
            ScheduledLocalNoise::new(emitting.into(), 0.4, 0.6, 0.0, 0.0, 0.0).unwrap();
        let batches = [batch(
            0,
            0,
            0,
            vec![
                Gate::rxy1q(Angle64::HALF_TURN, Angle64::ZERO, &[0]),
                Gate::rzz(Angle64::QUARTER_TURN, &[(0, 1)]),
                Gate::measure_leaked(&[0]),
                Gate::measure_leaked(&[1]),
            ],
        )];
        for seed in 0..32 {
            let mut a = system(plain.clone().into(), events, seed);
            let want = run(&mut a, &batches, events);
            for profile in [explicit.clone(), replacement.clone()] {
                let mut b = system(profile.into(), events, seed);
                assert_eq!(run(&mut b, &batches, events), want);
                assert_eq!(
                    a.controller().rng().clone().next_u64(),
                    b.controller().rng().clone().next_u64()
                );
            }
        }
    }
}

#[test]
fn replacing_emission_models_preserves_profile_clones_and_host_reset() {
    for events in [false, true] {
        let original =
            ScheduledLocalNoise::new(ScheduledIdleNoise::new(2).unwrap(), 1.0, 0.0, 0.0, 0.0, 0.0)
                .unwrap()
                .with_p1_emission_ratio(1.0)
                .unwrap()
                .with_p1_emission_model(model(&[("Z", 1.0)]))
                .unwrap();
        let leaking = original
            .clone()
            .with_p1_emission_model(model(&[("L", 1.0)]))
            .unwrap();
        let gate = batch(
            0,
            0,
            0,
            vec![Gate::rxy1q(Angle64::HALF_TURN, Angle64::ZERO, &[0])],
        );
        let read = batch(1, 0, 0, vec![Gate::measure_leaked(&[0])]);
        assert_eq!(
            run(
                &mut system(original.into(), events, 3),
                &[gate.clone(), read.clone()],
                events
            ),
            vec![0]
        );
        let mut active = system(leaking.into(), events, 3);
        run(&mut active, &[gate], events);
        let mut cloned = active.clone();
        assert!(
            cloned
                .process(message(std::slice::from_ref(&read), events))
                .is_err()
        );
        assert_eq!(
            run(&mut active, std::slice::from_ref(&read), events),
            vec![2]
        );
        for host in [&mut active, &mut cloned] {
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
                        vec![
                            Gate::rxy1q(Angle64::HALF_TURN, Angle64::ZERO, &[0]),
                            Gate::measure_leaked(&[0])
                        ]
                    )],
                    events
                ),
                vec![2]
            );
        }
    }
}

#[test]
fn seepage_requires_both_a_gate_fault_and_its_emission_branch() {
    for events in [false, true] {
        for two_qubit in [false, true] {
            for (probability, emission) in [(0.0, 1.0), (1.0, 0.0), (1.0, 1.0)] {
                let local = ScheduledLocalNoise::new(
                    ScheduledIdleNoise::new(2).unwrap(),
                    probability,
                    probability,
                    1.0,
                    0.0,
                    0.0,
                )
                .unwrap()
                .with_prep_leak_ratio(1.0)
                .unwrap()
                .with_p1_emission_ratio(emission)
                .unwrap()
                .with_p2_emission_ratio(emission)
                .unwrap()
                .with_p1_seepage_prob(1.0)
                .unwrap()
                .with_p2_seepage_prob(1.0)
                .unwrap();
                let gate = if two_qubit {
                    Gate::rzz(Angle64::QUARTER_TURN, &[(0, 1)])
                } else {
                    Gate::rxy1q(Angle64::HALF_TURN, Angle64::ZERO, &[0])
                };
                let mut sim = system(local.into(), events, 11);
                let rows = run(
                    &mut sim,
                    &[batch(
                        0,
                        0,
                        0,
                        vec![
                            Gate::pz(&[0]),
                            Gate::pz(&[1]),
                            gate,
                            Gate::measure_leaked(&[0]),
                            Gate::measure_leaked(&[1]),
                        ],
                    )],
                    events,
                );
                if probability > 0.0 && emission > 0.0 {
                    assert!(rows[0] < 2);
                    if two_qubit {
                        assert!(rows[1] < 2);
                    } else {
                        assert_eq!(rows[1], 2);
                    }
                } else {
                    assert_eq!(rows, vec![2, 2]);
                }
            }
        }
    }
}

#[test]
fn preparation_crosstalk_rejects_invalid_probabilities_even_without_prep_faults() {
    for bad in [-0.1, 1.1, f64::INFINITY, f64::NEG_INFINITY, f64::NAN] {
        assert!(
            ScheduledLocalNoise::new(ScheduledIdleNoise::new(2).unwrap(), 0.0, 0.0, 0.0, 0.0, 0.0)
                .unwrap()
                .with_prep_crosstalk(bad)
                .is_err()
        );
    }
}

fn prep_crosstalk_profile(probability: f64) -> ScheduledIdleNoise {
    ScheduledLocalNoise::new(ScheduledIdleNoise::new(2).unwrap(), 0.0, 0.0, 0.0, 0.0, 0.0)
        .unwrap()
        .with_prep_crosstalk(probability)
        .unwrap()
        .into()
}

#[test]
fn preparation_crosstalk_collapses_spectator_and_discards_internal_results() {
    let angle = pecos_core::radians!(std::f64::consts::FRAC_PI_2);
    let gates = vec![
        Gate::pz(&[0]),
        Gate::rxy1q(angle, Angle64::ZERO, &[0]),
        Gate::pz(&[1]),
        Gate::rxy1q(-angle, Angle64::ZERO, &[0]),
        Gate::mz(&[0]),
        Gate::mz(&[1]),
    ];
    for events in [false, true] {
        for rate in [0.0, 0.37, 1.0] {
            let mut ones = 0;
            for seed in 0..128 {
                let mut actual = system(prep_crosstalk_profile(rate), events, seed);
                let mut expected = QuantumSystem::new(
                    Box::new(
                        GeneralNoiseModelBuilder::new()
                            .with_p_prep_crosstalk(rate)
                            .build(),
                    ),
                    Box::new(StateVecEngine::new(2)),
                );
                expected.set_seed(seed);
                let got = run(&mut actual, &[batch(0, 0, 0, gates.clone())], events);
                let mut input = ByteMessage::quantum_operations_builder();
                for g in &gates {
                    input.add_gate_command(g);
                }
                let want = expected.process(input.build()).unwrap().outcomes().unwrap();
                assert_eq!(got, want);
                assert_eq!(got.len(), 2, "internal collapse is not a user result");
                assert_eq!(got[1], 0);
                assert_eq!(
                    actual.noise_model().rng().clone().next_u64(),
                    expected.noise_model().rng().clone().next_u64()
                );
                ones += got[0];
            }
            if rate == 0.0 {
                assert_eq!(ones, 0);
            } else {
                assert!(ones > 0 && ones < 128, "coherence loss must be observable");
            }
        }
    }
}

#[test]
fn preparation_crosstalk_preserves_whole_spectator_idle_gap() {
    let angle = pecos_core::radians!(std::f64::consts::FRAC_PI_2);
    for events in [false, true] {
        for rate in [0.0, 0.7, 1.0] {
            for seed in 0..32 {
                let idle = ScheduledIdleNoise::new(2)
                    .unwrap()
                    .with_sine(0.31, model(&[("X", 1.0)]))
                    .unwrap()
                    .with_coherent(0.27, model(&[("RY", 1.0)]))
                    .unwrap();
                let config = ScheduledLocalNoise::new(idle, 0.0, 0.0, 0.0, 0.0, 0.0)
                    .unwrap()
                    .with_prep_crosstalk(rate)
                    .unwrap();
                let mut actual = system(config.into(), events, seed);
                let mut expected = QuantumSystem::new(
                    Box::new(
                        GeneralNoiseModelBuilder::new()
                            .with_p_prep_crosstalk(rate)
                            .with_p_idle_sin_squared(0.31, &model(&[("X", 1.0)]))
                            .with_p_idle_coherent(0.27, &model(&[("RY", 1.0)]))
                            .build(),
                    ),
                    Box::new(StateVecEngine::new(2)),
                );
                expected.set_seed(seed);
                let got = run(
                    &mut actual,
                    &[
                        batch(
                            0,
                            0,
                            0,
                            vec![Gate::pz(&[0]), Gate::rxy1q(angle, Angle64::ZERO, &[0])],
                        ),
                        // Busy time belongs only to qubit 1, not its spectator.
                        batch(1, 1_000_000_000, 99_000_000_000, vec![Gate::pz(&[1])]),
                        batch(
                            2,
                            4_000_000_000,
                            0,
                            vec![Gate::rxy1q(-angle, Angle64::ZERO, &[0]), Gate::mz(&[0])],
                        ),
                    ],
                    events,
                );
                expected
                    .process(
                        ByteMessage::quantum_operations_builder()
                            .pz(&[0])
                            .rxy1q(angle, Angle64::ZERO, &[0])
                            .build(),
                    )
                    .unwrap();
                expected
                    .process(ByteMessage::quantum_operations_builder().pz(&[1]).build())
                    .unwrap();
                let want = expected
                    .process(
                        ByteMessage::quantum_operations_builder()
                            .idle(4.0, &[0])
                            .rxy1q(-angle, Angle64::ZERO, &[0])
                            .mz(&[0])
                            .build(),
                    )
                    .unwrap()
                    .outcomes()
                    .unwrap();
                assert_eq!(got, want);
                assert_eq!(
                    actual.noise_model().rng().clone().next_u64(),
                    expected.noise_model().rng().clone().next_u64()
                );
            }
        }
    }
}

#[test]
fn preparation_crosstalk_prepared_pool_is_cleared_on_host_reset_and_clone() {
    let angle = pecos_core::radians!(std::f64::consts::FRAC_PI_2);
    for events in [false, true] {
        let mut active = system(prep_crosstalk_profile(1.0), events, 42);
        run(&mut active, &[batch(0, 0, 0, vec![Gate::pz(&[0])])], events);
        let mut cloned = active.clone();
        for host in [&mut active, &mut cloned] {
            host.reset().unwrap();
            host.begin_shot(context(1)).unwrap();
            // Qubit 0 has a state but was not prepared in this shot. The other
            // qubit's reset must not collapse it via a stale prepared pool.
            assert_eq!(
                run(
                    host,
                    &[batch(
                        0,
                        0,
                        0,
                        vec![
                            Gate::rxy1q(angle, Angle64::ZERO, &[0]),
                            Gate::pz(&[1]),
                            Gate::rxy1q(-angle, Angle64::ZERO, &[0]),
                            Gate::mz(&[0]),
                        ]
                    )],
                    events
                ),
                vec![0]
            );
        }
    }
}

#[test]
fn preparation_crosstalk_preserves_leakage_flags_across_repeated_preparations() {
    for events in [false, true] {
        let config =
            ScheduledLocalNoise::new(ScheduledIdleNoise::new(2).unwrap(), 0.0, 0.0, 1.0, 0.0, 0.0)
                .unwrap()
                .with_prep_leak_ratio(1.0)
                .unwrap()
                .with_prep_crosstalk(1.0)
                .unwrap();
        let mut host = system(config.into(), events, 42);
        assert_eq!(
            run(
                &mut host,
                &[batch(
                    0,
                    0,
                    0,
                    vec![
                        Gate::pz(&[0]),
                        Gate::measure_leaked(&[0]),
                        Gate::pz(&[1]),
                        Gate::measure_leaked(&[0]),
                        Gate::pz(&[1]),
                        Gate::measure_leaked(&[0]),
                        Gate::measure_leaked(&[1]),
                    ]
                )],
                events
            ),
            vec![2, 2, 2, 2]
        );
    }
}

#[test]
fn preparation_crosstalk_zero_default_and_local_replacement_preserve_rng() {
    let angle = pecos_core::radians!(std::f64::consts::FRAC_PI_2);
    for events in [false, true] {
        let plain = ScheduledLocalNoise::new(
            ScheduledIdleNoise::new(2)
                .unwrap()
                .with_linear(0.2, None)
                .unwrap(),
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        .unwrap();
        let explicit = plain.clone().with_prep_crosstalk(0.0).unwrap();
        let replaced = ScheduledLocalNoise::new(
            plain.clone().with_prep_crosstalk(1.0).unwrap().into(),
            0.1,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        .unwrap();
        for seed in 0..32 {
            let batches = [
                batch(
                    0,
                    0,
                    0,
                    vec![
                        Gate::pz(&[0]),
                        Gate::rxy1q(angle, Angle64::ZERO, &[0]),
                        Gate::pz(&[1]),
                    ],
                ),
                batch(
                    1,
                    1_000_000_000,
                    0,
                    vec![Gate::rxy1q(-angle, Angle64::ZERO, &[0]), Gate::mz(&[0])],
                ),
            ];
            let mut original = system(plain.clone().into(), events, seed);
            let expected = run(&mut original, &batches, events);
            for config in [explicit.clone(), replaced.clone()] {
                let mut host = system(config.into(), events, seed);
                assert_eq!(run(&mut host, &batches, events), expected);
                assert_eq!(
                    host.noise_model().rng().clone().next_u64(),
                    original.noise_model().rng().clone().next_u64()
                );
            }
        }
    }
}
