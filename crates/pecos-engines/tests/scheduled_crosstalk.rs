//! Production v7 admission and the existing noise/quantum exchange, with invented events.
use pecos_core::{Angle64, errors::PecosError};
use pecos_engines::noise::{GeneralNoiseModel, IntoNoiseModel};
use pecos_engines::runtime_frame::ShotContext;
use pecos_engines::scheduled_events::*;
use pecos_engines::scheduled_frame::{ScheduledLocalNoise, ScheduledNoise};
use pecos_engines::{ByteMessage, Engine, Gate, GateType, QuantumSystem, StateVecEngine};
use std::collections::BTreeMap;

fn flips() -> BTreeMap<String, f64> {
    BTreeMap::from([("0->1".into(), 1.0), ("1->0".into(), 1.0)])
}
fn g(gate: Gate) -> ScheduledEventOp {
    ScheduledEventOp::Gate(Box::new(gate))
}
fn e(tag: u64, ids: &[u8]) -> ScheduledEventOp {
    ScheduledEventOp::Custom {
        tag,
        payload: ids.to_vec(),
    }
}
fn x(q: usize) -> Gate {
    Gate::rxy1q(
        pecos_core::radians!(std::f64::consts::PI),
        Angle64::ZERO,
        &[q],
    )
}
fn batch(
    index: u64,
    start: u64,
    duration: u64,
    operations: Vec<ScheduledEventOp>,
) -> ScheduledEventBatch {
    let measurements = operations
        .iter()
        .enumerate()
        .filter_map(|(i, op)| {
            if let ScheduledEventOp::Gate(g) = op
                && matches!(g.gate_type, GateType::MZ | GateType::MeasureLeaked)
            {
                Some(ScheduledResult {
                    operation_index: i,
                    runtime_result: index * 8192 + i as u64,
                    program_result: 99,
                })
            } else {
                None
            }
        })
        .collect();
    ScheduledEventBatch {
        runtime_shot_id: 0,
        batch_index: index,
        start_nanos: start,
        duration_nanos: duration,
        operations,
        measurements,
    }
}
struct Adapter;
impl ScheduledBatchAdapter for Adapter {
    fn validate(&self, batch: &ScheduledEventBatch) -> Result<(), PecosError> {
        if batch
            .operations
            .iter()
            .any(|op| matches!(op, ScheduledEventOp::Custom { tag, .. } if *tag > 1))
        {
            return Err(PecosError::Input("invented unsupported schema".into()));
        }
        Ok(())
    }
    fn translate(
        &mut self,
        batch: &ScheduledEventBatch,
        out: &mut ScheduledGateBuffer<'_>,
    ) -> Result<(), PecosError> {
        for op in &batch.operations {
            match op {
                ScheduledEventOp::Gate(g) => out.push(g.as_ref().clone())?,
                ScheduledEventOp::Custom { tag, payload } => {
                    let ids = payload.iter().map(|v| usize::from(*v)).collect();
                    out.push_crosstalk(if *tag == 0 {
                        ScheduledCrosstalk::Local(ids)
                    } else {
                        ScheduledCrosstalk::GlobalExcluding(ids)
                    })?;
                }
            }
        }
        Ok(())
    }
}
fn profile(rate: f64) -> ScheduledNoise {
    ScheduledNoise::new(2)
        .unwrap()
        .with_measurement_crosstalk(rate, rate, flips())
        .unwrap()
}
fn host(profile: ScheduledNoise) -> QuantumSystem {
    let model = ScheduledEventNoise::new(profile, |_| Ok(Box::new(Adapter)))
        .into_noise_model()
        .unwrap();
    let mut host = QuantumSystem::new(model, Box::new(StateVecEngine::new(2)));
    host.begin_shot(ShotContext {
        run: 1,
        worker: 0,
        shot: 0,
    })
    .unwrap();
    host
}
fn run(host: &mut QuantumSystem, batches: &[ScheduledEventBatch]) -> Vec<u32> {
    host.process(encode_event_batches(batches).unwrap())
        .unwrap()
        .outcomes()
        .unwrap()
}

#[test]
fn production_exchanges_internal_measurements_and_preserves_user_order() {
    let mut system = host(profile(1.0));
    let operations = vec![
        g(Gate::pz(&[0])),
        g(Gate::pz(&[1])),
        e(0, &[0]),
        g(Gate::mz(&[0])),
        e(0, &[0]),
        g(Gate::mz(&[0])),
        e(1, &[0]),
        g(Gate::mz(&[1])),
        g(x(0)),
    ];
    assert_eq!(
        run(&mut system, &[batch(0, 0, 0, operations)]),
        vec![1, 0, 1]
    );
    assert_eq!(
        run(
            &mut system,
            &[batch(1, 0, 0, vec![g(Gate::mz(&[0])), g(Gate::mz(&[1]))])]
        ),
        vec![1, 1]
    );
    let mut clone = system.clone();
    assert!(
        clone
            .begin_shot(ShotContext {
                run: 1,
                worker: 1,
                shot: 0
            })
            .is_err()
    );
    clone.reset().unwrap();
    clone
        .begin_shot(ShotContext {
            run: 1,
            worker: 1,
            shot: 0,
        })
        .unwrap();
    assert_eq!(
        run(
            &mut clone,
            &[batch(0, 0, 0, vec![e(1, &[]), g(Gate::mz(&[0]))])]
        ),
        vec![0]
    );
}

#[test]
fn exposure_does_not_reserve_busy_time_even_during_a_native_operation() {
    let mut system = host(profile(1.0));
    assert_eq!(
        run(
            &mut system,
            &[
                batch(0, 0, 10, vec![g(Gate::pz(&[0]))]),
                batch(1, 5, 100, vec![e(0, &[0])]),
                batch(2, 10, 0, vec![g(Gate::mz(&[0]))]),
            ]
        ),
        vec![1]
    );
}

#[test]
fn exposures_preserve_whole_nonlinear_idle_and_general_rng() {
    let z = BTreeMap::from([("Z".into(), 1.0)]);
    let rz = BTreeMap::from([("RZ".into(), 1.0)]);
    for rate in [0.0, 0.7, 1.0] {
        let idle = profile(rate)
            .with_sine(0.31, None)
            .unwrap()
            .with_coherent(0.27, None)
            .unwrap();
        for seed in 0..32 {
            let mut actual = host(idle.clone());
            let model = GeneralNoiseModel::builder()
                .with_p_meas_crosstalk(rate)
                .with_p_meas_crosstalk_model(&flips())
                .with_p_idle_sin_squared(0.31, &z)
                .with_p_idle_coherent(0.27, &rz)
                .build();
            let mut expected =
                QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(2)));
            actual.set_seed(seed);
            expected.set_seed(seed);
            let theta = pecos_core::radians!(std::f64::consts::FRAC_PI_2);
            let first = vec![
                g(Gate::pz(&[0])),
                g(Gate::rxy1q(theta, Angle64::ZERO, &[0])),
            ];
            let last = vec![
                g(Gate::rxy1q(-theta, Angle64::ZERO, &[0])),
                g(Gate::mz(&[0])),
            ];
            let a = run(
                &mut actual,
                &[
                    batch(0, 0, 0, first),
                    batch(1, 1_000_000_000, 99_000_000_000, vec![e(0, &[0])]),
                    batch(2, 4_000_000_000, 0, last),
                ],
            );
            expected
                .process(
                    ByteMessage::quantum_operations_builder()
                        .pz(&[0])
                        .rxy1q(theta, Angle64::ZERO, &[0])
                        .build(),
                )
                .unwrap();
            expected
                .process(
                    ByteMessage::quantum_operations_builder()
                        .meas_crosstalk_local_payload(&[0])
                        .build(),
                )
                .unwrap();
            // Exactly one four-second idle at the next physical operation. The
            // intermediate exposure does not close or split this interval.
            let b = expected
                .process(
                    ByteMessage::quantum_operations_builder()
                        .idle(4.0, &[0])
                        .rxy1q(-theta, Angle64::ZERO, &[0])
                        .mz(&[0])
                        .build(),
                )
                .unwrap()
                .outcomes()
                .unwrap();
            assert_eq!(a, b);
            // Explicit exposure probability draws follow GeneralNoiseModel even
            // at rate zero; they are not inert metadata acknowledgements.
            assert_eq!(
                actual.noise_model().rng().clone().next_u64(),
                expected.noise_model().rng().clone().next_u64()
            );
        }
    }
}

#[test]
fn later_bad_effect_rejects_whole_input_before_quantum_mutation() {
    for (profile, ids) in [
        (profile(1.0), vec![2]),
        (profile(0.0), vec![0, 0]),
        (ScheduledNoise::new(2).unwrap(), vec![0]),
    ] {
        let mut system = host(profile);
        let input = encode_event_batches(&[
            batch(0, 0, 0, vec![g(Gate::pz(&[0])), g(x(0))]),
            batch(1, 0, 0, vec![e(0, &ids)]),
        ])
        .unwrap();
        assert!(system.process(input).is_err());
        assert!(
            system
                .process(encode_event_batches(&[batch(0, 0, 0, vec![g(Gate::mz(&[0]))])]).unwrap())
                .is_err()
        );
        let (_, mut engine) = system.into_components();
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

#[test]
fn invalid_profile_parameters_reject_even_when_disabled() {
    for rate in [f64::NAN, -0.1, 1.1, f64::INFINITY] {
        assert!(
            ScheduledNoise::new(2)
                .unwrap()
                .with_measurement_crosstalk(rate, 0.0, flips())
                .is_err()
        );
        assert!(
            ScheduledNoise::new(2)
                .unwrap()
                .with_measurement_crosstalk(0.0, rate, flips())
                .is_err()
        );
    }
    assert!(
        ScheduledNoise::new(2)
            .unwrap()
            .with_measurement_crosstalk(0.0, 0.0, BTreeMap::new())
            .is_err()
    );
}

#[test]
fn production_matches_general_controller_for_ordered_exposures() {
    for seed in 0..64 {
        let local =
            ScheduledLocalNoise::new(ScheduledNoise::new(2).unwrap(), 0.2, 0.0, 0.15, 0.3, 0.4)
                .unwrap();
        let config = ScheduledNoise::from(local)
            .with_measurement_crosstalk(0.6, 0.6, flips())
            .unwrap();
        let mut actual = host(config);
        let model = GeneralNoiseModel::builder()
            .with_p1(0.2)
            .with_p_prep(0.15)
            .with_p_meas_0(0.3)
            .with_p_meas_1(0.4)
            .with_p_meas_crosstalk(0.6)
            .with_p_meas_crosstalk_model(&flips())
            .build();
        let mut expected = QuantumSystem::new(Box::new(model), Box::new(StateVecEngine::new(2)));
        actual.set_seed(seed);
        expected.set_seed(seed);
        let input = batch(
            0,
            0,
            0,
            vec![
                g(Gate::pz(&[0])),
                g(Gate::pz(&[1])),
                g(x(0)),
                e(0, &[1, 0]),
                g(Gate::mz(&[0])),
                e(1, &[]),
                g(Gate::mz(&[0])),
                g(Gate::mz(&[1])),
            ],
        );
        let reference = ByteMessage::quantum_operations_builder()
            .pz(&[0])
            .pz(&[1])
            .rxy1q(
                pecos_core::radians!(std::f64::consts::PI),
                Angle64::ZERO,
                &[0],
            )
            .meas_crosstalk_local_payload(&[1, 0])
            .mz(&[0])
            .meas_crosstalk_global_payload(&[])
            .mz(&[0])
            .mz(&[1])
            .build();
        assert_eq!(
            run(&mut actual, &[input]),
            expected.process(reference).unwrap().outcomes().unwrap()
        );
        assert_eq!(
            actual.noise_model().rng().clone().next_u64(),
            expected.noise_model().rng().clone().next_u64()
        );
    }
}

#[test]
fn leakage_and_unprepared_victims_follow_general_semantics() {
    let config = ScheduledNoise::new(2)
        .unwrap()
        .with_measurement_crosstalk(
            1.0,
            1.0,
            BTreeMap::from([("0->L".into(), 1.0), ("1->L".into(), 1.0)]),
        )
        .unwrap();
    let mut system = host(config);
    let ml = |q| {
        Gate::new(
            GateType::MeasureLeaked,
            vec![],
            vec![],
            vec![pecos_core::QubitId(q)],
        )
    };
    assert_eq!(
        run(
            &mut system,
            &[batch(
                0,
                0,
                0,
                vec![
                    e(0, &[0]),
                    g(ml(0)),
                    g(Gate::pz(&[0])),
                    e(0, &[0]),
                    g(ml(0))
                ]
            )]
        ),
        vec![0, 2]
    );
    assert_eq!(
        run(
            &mut system,
            &[batch(
                1,
                0,
                0,
                vec![e(0, &[0]), g(ml(0)), g(Gate::pz(&[0])), g(ml(0))]
            )]
        ),
        vec![2, 0]
    );
}

#[test]
fn maximum_spectator_fanout_and_latched_output_rejection() {
    struct Fanout(usize);
    impl ScheduledBatchAdapter for Fanout {
        fn validate(&self, _: &ScheduledEventBatch) -> Result<(), PecosError> {
            Ok(())
        }
        fn translate(
            &mut self,
            _: &ScheduledEventBatch,
            out: &mut ScheduledGateBuffer<'_>,
        ) -> Result<(), PecosError> {
            for _ in 0..self.0 {
                // Even swallowing an admission failure cannot authorize the batch.
                let _ = out.push_crosstalk(ScheduledCrosstalk::GlobalExcluding(vec![]));
            }
            Ok(())
        }
    }
    for count in [1024, 1025] {
        let config = ScheduledNoise::new(16)
            .unwrap()
            .with_measurement_crosstalk(0.0, 0.0, flips())
            .unwrap();
        let model = ScheduledEventNoise::new(config, move |_| Ok(Box::new(Fanout(count))))
            .into_noise_model()
            .unwrap();
        let mut system = QuantumSystem::new(model, Box::new(StateVecEngine::new(16)));
        system
            .begin_shot(ShotContext {
                run: 0,
                worker: 0,
                shot: 0,
            })
            .unwrap();
        let result =
            system.process(encode_event_batches(&[batch(0, 0, 0, vec![e(1, &[])])]).unwrap());
        assert_eq!(result.is_ok(), count == 1024);
    }
}
