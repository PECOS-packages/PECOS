use pecos_core::{Angle64, errors::PecosError};
use pecos_engines::noise::{GeneralNoiseModelBuilder, IntoNoiseModel};
use pecos_engines::runtime_frame::ShotContext;
use pecos_engines::scheduled_events::{
    ScheduledBatchAdapter, ScheduledEventBatch, ScheduledEventIdleNoise, ScheduledEventOp,
    ScheduledGateBuffer, ScheduledResult, encode_event_batches,
};
use pecos_engines::scheduled_frame::{ScheduledIdleNoise, TimedBatch, encode_timed_batches};
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
            if operations.is_empty() {
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
fn leakage(sine: bool) -> ScheduledIdleNoise {
    let profile = ScheduledIdleNoise::new(2).unwrap();
    if sine {
        profile
            .with_sine(std::f64::consts::FRAC_PI_2, model(&[("L", 1.0)]))
            .unwrap()
    } else {
        profile.with_linear(1.0, model(&[("L", 1.0)])).unwrap()
    }
}

#[test]
fn leakage_survives_batches_measurement_and_gates_until_preparation() {
    for events in [false, true] {
        for sine in [false, true] {
            let mut s = system(leakage(sine), events, 3);
            // The first measurement occupies a full second. The next gap is one
            // second, not two; the sine case distinguishes these deterministically.
            assert_eq!(
                run(
                    &mut s,
                    &[batch(0, 0, 1_000_000_000, vec![Gate::mz(&[0])])],
                    events
                ),
                vec![0]
            );
            assert_eq!(
                run(
                    &mut s,
                    &[batch(
                        1,
                        2_000_000_000,
                        0,
                        vec![Gate::mz(&[0]), Gate::measure_leaked(&[0])]
                    )],
                    events
                ),
                vec![1, 2]
            );
            s.process(ByteMessage::create_empty()).unwrap();
            assert_eq!(
                run(
                    &mut s,
                    &[batch(
                        2,
                        2_000_000_000,
                        0,
                        vec![
                            Gate::rxy1q(
                                Angle64::from_radians(std::f64::consts::PI),
                                Angle64::ZERO,
                                &[0]
                            ),
                            Gate::measure_leaked(&[0])
                        ]
                    )],
                    events
                ),
                vec![2]
            );
            // Preparation both clears existing leakage and omits its preceding idle.
            assert_eq!(
                run(
                    &mut s,
                    &[
                        batch(
                            3,
                            4_000_000_000,
                            1_000_000_000,
                            vec![Gate::pz(&[0]), Gate::measure_leaked(&[0])]
                        ),
                        batch(
                            4,
                            5_000_000_000,
                            0,
                            vec![Gate::measure_leaked(&[0]), Gate::measure_leaked(&[1])]
                        )
                    ],
                    events
                ),
                vec![0, 0, 2]
            );
        }
    }
}
#[test]
fn nonlinear_idle_is_not_split_by_metadata_or_host_input_boundaries() {
    let p = ScheduledIdleNoise::new(2)
        .unwrap()
        .with_sine(std::f64::consts::PI, model(&[("L", 1.0)]))
        .unwrap();
    for events in [false, true] {
        let mut together = system(p.clone(), events, 8);
        let mut split = system(p.clone(), events, 8);
        let batches = [
            batch(0, 0, 0, vec![Gate::pz(&[0])]),
            batch(1, 500_000_000, 0, vec![]),
            batch(2, 1_000_000_000, 0, vec![Gate::measure_leaked(&[0])]),
        ];
        assert_eq!(run(&mut together, &batches, events), vec![0]);
        for b in &batches[..2] {
            assert!(run(&mut split, std::slice::from_ref(b), events).is_empty());
        }
        split.process(ByteMessage::create_empty()).unwrap();
        assert_eq!(run(&mut split, &batches[2..], events), vec![0]);
        assert_eq!(
            together.controller().rng().clone().next_u64(),
            split.controller().rng().clone().next_u64()
        );
    }
}
#[test]
fn leakage_is_cleared_by_whole_system_reset_and_cannot_escape_live_clone() {
    for events in [false, true] {
        let mut original = system(leakage(false), events, 6);
        assert_eq!(
            run(
                &mut original,
                &[batch(0, 1_000_000_000, 0, vec![Gate::measure_leaked(&[0])])],
                events
            ),
            vec![2]
        );
        let mut cloned = original.clone();
        let next = [batch(1, 1_000_000_000, 0, vec![Gate::measure_leaked(&[0])])];
        assert!(cloned.process(message(&next, events)).is_err());
        assert_eq!(run(&mut original, &next, events), vec![2]);
        for s in [&mut original, &mut cloned] {
            s.reset().unwrap();
            s.begin_shot(context(1)).unwrap();
            assert_eq!(
                run(
                    s,
                    &[batch(0, 0, 0, vec![Gate::measure_leaked(&[0])])],
                    events
                ),
                vec![0]
            );
        }
    }
}
#[test]
fn all_idle_families_match_explicit_general_noise_outcomes_and_rng() {
    let linear = model(&[("X", 0.2), ("Y", 0.1), ("Z", 0.3), ("L", 0.4)]);
    let sine = model(&[("X", 0.7), ("Y", 0.4), ("Z", 0.8), ("L", 0.6)]);
    let coherent = model(&[("RX", 0.3), ("RY", 1.2), ("RZ", 0.8)]);
    let profile = ScheduledIdleNoise::new(2)
        .unwrap()
        .with_linear(0.2, linear.clone())
        .unwrap()
        .with_sine(0.3, sine.clone())
        .unwrap()
        .with_coherent(0.4, coherent.clone())
        .unwrap();
    let batches = [
        batch(
            0,
            1_000_000_000,
            100_000_000,
            vec![Gate::pz(&[0]), Gate::pz(&[1])],
        ),
        batch(
            1,
            1_300_000_000,
            200_000_000,
            vec![
                Gate::rxy1q(Angle64::from_radians(0.9), Angle64::ZERO, &[0]),
                Gate::rzz(Angle64::from_radians(0.7), &[(0, 1)]),
            ],
        ),
        batch(
            2,
            2_000_000_000,
            0,
            vec![Gate::measure_leaked(&[0]), Gate::mz(&[1])],
        ),
        batch(
            3,
            3_000_000_000,
            0,
            vec![
                Gate::pz(&[0]),
                Gate::measure_leaked(&[0]),
                Gate::measure_leaked(&[1]),
            ],
        ),
    ];
    for events in [false, true] {
        for seed in 0..32 {
            let mut actual = system(profile.clone(), events, seed);
            let mut expected = QuantumSystem::new(
                Box::new(
                    GeneralNoiseModelBuilder::new()
                        .with_p_idle_linear(0.2, &linear)
                        .with_p_idle_sin_squared(0.3, &sine)
                        .with_p_idle_coherent(0.4, &coherent)
                        .build(),
                ),
                Box::new(StateVecEngine::new(2)),
            );
            expected.set_seed(seed);
            // Independent explicit gaps: reset omissions, 0.2 seconds on both,
            // 0.5 seconds on both, then 1 second on qubit 1 only.
            for (i, b) in batches.iter().enumerate() {
                let mut explicit = ByteMessage::quantum_operations_builder();
                for (j, g) in b.gates.iter().enumerate() {
                    match (i, j) {
                        (1, 0) => {
                            explicit.idle(0.2, &[0]);
                        }
                        (1, 1) => {
                            explicit.idle(0.2, &[1]);
                        }
                        (2, 0) => {
                            explicit.idle(0.5, &[0]);
                        }
                        (2, 1) => {
                            explicit.idle(0.5, &[1]);
                        }
                        (3, 2) => {
                            explicit.idle(1.0, &[1]);
                        }
                        _ => {}
                    }
                    explicit.add_gate_command(g);
                }
                let want = expected
                    .process(explicit.build())
                    .unwrap()
                    .outcomes()
                    .unwrap();
                assert_eq!(
                    run(&mut actual, std::slice::from_ref(b), events),
                    want,
                    "events={events} seed={seed} batch={i}"
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
fn checked_profile_rejects_invalid_models_without_panicking() {
    assert!(ScheduledIdleNoise::new(0).is_err());
    assert!(ScheduledIdleNoise::new(17).is_err());
    let p = ScheduledIdleNoise::new(1).unwrap();
    for rate in [-1.0, f64::NAN, f64::INFINITY] {
        assert!(p.clone().with_linear(rate, model(&[("L", 1.0)])).is_err());
        assert!(p.clone().with_sine(rate, model(&[("L", 1.0)])).is_err());
        assert!(
            p.clone()
                .with_coherent(rate, model(&[("RX", 1.0)]))
                .is_err()
        );
    }
    for weights in [
        model(&[]),
        model(&[("X", 0.5)]),
        model(&[("X", -0.5), ("Y", 1.5)]),
        model(&[("bad", 1.0)]),
        model(&[("X", f64::NAN)]),
        model(&[("X", f64::INFINITY)]),
    ] {
        assert!(p.clone().with_linear(0.0, weights).is_err());
    }
    for weights in [
        model(&[("X", -1.0)]),
        model(&[("bad", 1.0)]),
        model(&[("X", f64::NAN)]),
        model(&[("L", f64::INFINITY)]),
    ] {
        assert!(p.clone().with_sine(0.0, weights).is_err());
    }
    assert!(p.clone().with_coherent(1.0, model(&[("L", 1.0)])).is_err());
    assert!(p.clone().with_sine(f64::MAX, model(&[("Z", 2.0)])).is_err());
    assert!(p.with_coherent(f64::MAX, model(&[("RZ", 2.0)])).is_err());
}
#[test]
fn overflowing_later_gap_rejects_before_execution_or_rng_mutation() {
    for events in [false, true] {
        let p = ScheduledIdleNoise::new(2)
            .unwrap()
            .with_coherent(1e307, model(&[("RX", 2.0)]))
            .unwrap();
        let mut s = system(p, events, 9);
        let before = s.controller().rng().clone().next_u64();
        let invalid = [
            batch(
                0,
                0,
                0,
                vec![Gate::rxy1q(
                    Angle64::from_radians(std::f64::consts::PI),
                    Angle64::ZERO,
                    &[0],
                )],
            ),
            batch(1, 10_000_000_000, 0, vec![Gate::mz(&[0])]),
        ];
        assert!(s.process(message(&invalid, events)).is_err());
        assert_eq!(before, s.controller().rng().clone().next_u64());
        // v4 adapter normalization failures poison the owner, even before gate
        // execution. Reset is required there; v3 admission permits retry.
        if events {
            s.reset().unwrap();
            s.begin_shot(context(1)).unwrap();
        }
        assert_eq!(
            run(&mut s, &[batch(0, 0, 0, vec![Gate::mz(&[0])])], events),
            vec![0]
        );
    }
}
