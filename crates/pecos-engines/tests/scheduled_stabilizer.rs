use pecos_core::{Angle64, errors::PecosError};
use pecos_engines::noise::IntoNoiseModel;
use pecos_engines::runtime_frame::ShotContext;
use pecos_engines::scheduled_events::{
    ScheduledBatchAdapter, ScheduledEventBatch, ScheduledEventIdleNoise, ScheduledEventOp,
    ScheduledGateBuffer, ScheduledResult, encode_event_batches,
};
use pecos_engines::scheduled_frame::{
    ScheduledIdleNoise, ScheduledLocalNoise, TimedBatch, encode_timed_batches,
};
use pecos_engines::{
    ByteMessage, Engine, EngineSystem, Gate, GateType, QuantumEngine, StabilizerEngine,
    StateVecEngine, quantum_system::QuantumSystem,
};

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
fn host(
    profile: ScheduledIdleNoise,
    events: bool,
    engine: Box<dyn QuantumEngine>,
    seed: u64,
) -> QuantumSystem {
    let noise = if events {
        ScheduledEventIdleNoise::new(profile, |_| Ok(Box::new(PassThrough)))
            .into_noise_model()
            .unwrap()
    } else {
        profile.into_noise_model().unwrap()
    };
    let mut host = QuantumSystem::new(noise, engine);
    host.set_seed(seed);
    host.begin_shot(context(0)).unwrap();
    host
}
fn run(host: &mut QuantumSystem, batches: &[TimedBatch], events: bool) -> Vec<u32> {
    host.process(message(batches, events))
        .unwrap()
        .outcomes()
        .unwrap()
}
fn local(n: usize, prep: f64, leak: f64, crosstalk: f64) -> ScheduledIdleNoise {
    ScheduledLocalNoise::new(ScheduledIdleNoise::new(n).unwrap(), 0., 0., prep, 0., 0.)
        .unwrap()
        .with_prep_leak_ratio(leak)
        .unwrap()
        .with_prep_crosstalk(crosstalk)
        .unwrap()
        .into()
}
#[test]
fn full_capacity_and_internal_measurements_do_not_escape() {
    for events in [false, true] {
        let mut s = host(
            local(512, 0., 0., 1.),
            events,
            Box::new(StabilizerEngine::new(512)),
            1,
        );
        let mut gates: Vec<_> = (0..512).map(|q| Gate::pz(&[q])).collect();
        gates.extend((0..512).map(|q| Gate::mz(&[q])));
        assert_eq!(run(&mut s, &[batch(0, 0, 0, gates)], events), vec![0; 512]);
        // Retain state and highest qubit identity across separate process inputs.
        assert_eq!(
            run(
                &mut s,
                &[batch(
                    1,
                    0,
                    0,
                    vec![
                        Gate::rxy1q(Angle64::HALF_TURN, Angle64::ZERO, &[511]),
                        Gate::mz(&[511])
                    ]
                )],
                events
            ),
            vec![1]
        );
    }
}
#[test]
fn whole_gap_stochastic_dephasing_matches_state_vector() {
    for events in [false, true] {
        for (case, rate) in [0.0, 0.7, std::f64::consts::FRAC_PI_2]
            .into_iter()
            .enumerate()
        {
            let config = ScheduledIdleNoise::new(2)
                .unwrap()
                .with_sine(rate, None)
                .unwrap();
            let input = [
                batch(
                    0,
                    0,
                    0,
                    vec![
                        Gate::pz(&[0]),
                        Gate::rxy1q(Angle64::QUARTER_TURN, Angle64::ZERO, &[0]),
                    ],
                ),
                batch(
                    1,
                    1_000_000_000,
                    0,
                    vec![
                        Gate::rxy1q(-Angle64::QUARTER_TURN, Angle64::ZERO, &[0]),
                        Gate::mz(&[0]),
                    ],
                ),
            ];
            let mut ones = 0;
            for seed in 0..64 {
                let mut a = host(
                    config.clone(),
                    events,
                    Box::new(StabilizerEngine::new(2)),
                    seed,
                );
                let mut b = host(
                    config.clone(),
                    events,
                    Box::new(StateVecEngine::new(2)),
                    seed,
                );
                let result = run(&mut a, &input, events);
                assert_eq!(result, run(&mut b, &input, events));
                ones += result[0];
                assert_eq!(
                    a.controller().rng().clone().next_u64(),
                    b.controller().rng().clone().next_u64()
                );
            }
            if case == 0 {
                assert_eq!(ones, 0);
            } else if case == 1 {
                assert!((1..64).contains(&ones));
            } else {
                assert_eq!(ones, 64);
            }
        }
    }
}
#[test]
fn late_non_clifford_rejects_before_noise_rng_or_state_mutation() {
    for events in [false, true] {
        for bad in [
            Gate::rz(Angle64::QUARTER_TURN / 2u64, &[0]),
            Gate::rzz(Angle64::QUARTER_TURN / 2u64, &[(0, 1)]),
            Gate::rxy1q(Angle64::QUARTER_TURN, Angle64::QUARTER_TURN / 2u64, &[0]),
            Gate::rxyxy2q(
                Angle64::QUARTER_TURN,
                Angle64::QUARTER_TURN / 2u64,
                &[(0, 1)],
            ),
        ] {
            let mut s = host(
                local(2, 1., 0., 0.),
                events,
                Box::new(StabilizerEngine::new(2)),
                42,
            );
            let before = s.controller().rng().clone().next_u64();
            let input = [
                batch(0, 0, 0, vec![Gate::pz(&[0])]),
                batch(1, 0, 0, vec![bad]),
            ];
            assert!(
                s.process(message(&input, events))
                    .err()
                    .unwrap()
                    .to_string()
                    .contains("Clifford")
            );
            assert_eq!(s.controller().rng().clone().next_u64(), before);
            let (_, mut engine) = s.into_components();
            assert_eq!(
                engine
                    .process(ByteMessage::quantum_operations_builder().mz(&[0]).build())
                    .unwrap()
                    .outcomes()
                    .unwrap(),
                vec![0]
            );
        }
    }
}
#[test]
fn coherent_profile_rejects_before_any_callback_even_with_empty_input() {
    let profile = ScheduledIdleNoise::new(1)
        .unwrap()
        .with_coherent(0.1, None)
        .unwrap();
    let noise =
        ScheduledEventIdleNoise::new(profile, |_| panic!("must reject before adapter factory"));
    let mut s = QuantumSystem::new(
        noise.into_noise_model().unwrap(),
        Box::new(StabilizerEngine::new(1)),
    );
    s.begin_shot(context(0)).unwrap();
    assert!(
        s.process(ByteMessage::builder().build())
            .err()
            .unwrap()
            .to_string()
            .contains("coherent")
    );
}
#[test]
fn state_vector_resource_limit_and_unsupported_engines_remain_closed() {
    let mut s = host(
        ScheduledIdleNoise::new(17).unwrap(),
        false,
        Box::new(StateVecEngine::new(17)),
        0,
    );
    assert!(
        s.process(ByteMessage::builder().build())
            .err()
            .unwrap()
            .to_string()
            .contains("at most 16")
    );
    let mut s = host(
        ScheduledIdleNoise::new(2).unwrap(),
        false,
        Box::new(StabilizerEngine::new(1)),
        0,
    );
    assert!(
        s.process(ByteMessage::builder().build())
            .err()
            .unwrap()
            .to_string()
            .contains("capacity")
    );
    let mut s = host(
        ScheduledIdleNoise::new(2).unwrap(),
        false,
        Box::new(pecos_engines::CoinTossEngine::new(2)),
        0,
    );
    assert!(s.process(ByteMessage::builder().build()).is_err());
}
#[test]
fn leakage_readout_reset_and_clone_preserve_the_contract() {
    for events in [false, true] {
        let mut s = host(
            local(65, 1., 1., 1.),
            events,
            Box::new(StabilizerEngine::new(65)),
            7,
        );
        let input = [batch(
            0,
            0,
            0,
            vec![
                Gate::pz(&[64]),
                Gate::simple(GateType::MeasureLeaked, vec![64.into()]),
            ],
        )];
        assert_eq!(run(&mut s, &input, events), vec![2]);
        let mut cloned = s.clone();
        for h in [&mut s, &mut cloned] {
            h.reset().unwrap();
            h.begin_shot(context(1)).unwrap();
            assert_eq!(run(h, &input, events), vec![2]);
        }
    }
}
#[test]
fn preparation_crosstalk_is_visible_with_a_stabilizer_spectator() {
    for events in [false, true] {
        let mut ones = 0;
        for seed in 0..64 {
            let mut s = host(
                local(33, 0., 0., 1.),
                events,
                Box::new(StabilizerEngine::new(33)),
                seed,
            );
            let input = [batch(
                0,
                0,
                0,
                vec![
                    Gate::pz(&[0]),
                    Gate::rxy1q(Angle64::QUARTER_TURN, Angle64::ZERO, &[0]),
                    Gate::pz(&[32]),
                    Gate::rxy1q(-Angle64::QUARTER_TURN, Angle64::ZERO, &[0]),
                    Gate::mz(&[0]),
                ],
            )];
            ones += run(&mut s, &input, events)[0];
        }
        assert!((1..64).contains(&ones));
    }
}

#[test]
fn rejected_adapter_frame_and_its_clone_require_reset() {
    let mut system = host(
        ScheduledIdleNoise::new(1).unwrap(),
        true,
        Box::new(StabilizerEngine::new(1)),
        42,
    );
    let bad = [batch(
        0,
        0,
        0,
        vec![Gate::rz(Angle64::QUARTER_TURN / 2u64, &[0])],
    )];
    assert!(system.process(message(&bad, true)).is_err());
    let mut cloned = system.clone();
    let good = [batch(0, 0, 0, vec![Gate::pz(&[0]), Gate::mz(&[0])])];
    for host in [&mut system, &mut cloned] {
        assert!(host.process(message(&good, true)).is_err());
        assert!(host.begin_shot(context(1)).is_err());
        host.reset().unwrap();
        host.begin_shot(context(1)).unwrap();
        assert_eq!(run(host, &good, true), vec![0]);
    }
}

struct EmptyClassical;
impl pecos_engines::ClassicalControlEngineBuilder for EmptyClassical {
    type Engine = pecos_engines::monte_carlo::engine::ExternalClassicalEngine;
    fn build(self) -> Result<Self::Engine, PecosError> {
        Ok(Self::Engine::new())
    }
}

#[test]
fn sim_builder_checks_backend_before_allocating_a_large_state_vector() {
    use pecos_engines::{sim, stabilizer, state_vector};
    for profile_qubits in [1, 512] {
        let profile = ScheduledIdleNoise::new(profile_qubits).unwrap();
        let error = sim(EmptyClassical)
            .qubits(512)
            .quantum(state_vector())
            .noise(profile)
            .build()
            .err()
            .expect("state-vector allocation must not be attempted");
        assert!(error.to_string().contains("at most 16"));
    }
    let profile = ScheduledIdleNoise::new(512).unwrap();
    assert!(
        sim(EmptyClassical)
            .qubits(512)
            .quantum(stabilizer())
            .noise(profile)
            .build()
            .is_ok()
    );
    let profile = ScheduledIdleNoise::new(1)
        .unwrap()
        .with_coherent(0.1, None)
        .unwrap();
    assert!(
        sim(EmptyClassical)
            .qubits(1)
            .quantum(stabilizer())
            .noise(profile)
            .build()
            .err()
            .unwrap()
            .to_string()
            .contains("coherent")
    );
    let profile = ScheduledIdleNoise::new(1).unwrap();
    assert!(
        sim(EmptyClassical)
            .qubits(512)
            .quantum(state_vector().parallel(true))
            .noise(profile)
            .build()
            .err()
            .unwrap()
            .to_string()
            .contains("parallel/dense")
    );
}
