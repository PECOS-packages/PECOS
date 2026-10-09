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
        ScheduledEventIdleNoise::new(profile, |_| Ok(Box::new(PassThrough))).into_noise_model()
    } else {
        profile.into_noise_model()
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
    let noise = ScheduledEventIdleNoise::new(profile(0.0, 0.0, 1.0, 0.0, 0.0), |_| {
        Ok(Box::new(AppendOutsideCapacity))
    });
    let mut sim = QuantumSystem::new(noise.into_noise_model(), Box::new(StateVecEngine::new(2)));
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
                    batch(1, 0, 0, vec![Gate::mz(&[0])])
                ],
                true
            ),
            vec![1]
        );
    }
}
