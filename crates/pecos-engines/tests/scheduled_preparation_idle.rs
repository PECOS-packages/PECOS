use pecos_core::Angle64;
use pecos_engines::noise::{GeneralNoiseModelBuilder, IntoNoiseModel};
use pecos_engines::runtime_frame::ShotContext;
use pecos_engines::scheduled_frame::{ScheduledIdleZ, TimedBatch, encode_timed_batches};
use pecos_engines::{
    ByteMessage, Engine, EngineSystem, Gate, StateVecEngine, quantum_system::QuantumSystem,
};
use std::collections::BTreeMap;

fn batches() -> Vec<TimedBatch> {
    let rotation = |sign| {
        Gate::rxy1q(
            Angle64::from_radians(sign * std::f64::consts::FRAC_PI_2),
            Angle64::ZERO,
            &[0],
        )
    };
    [
        (1, vec![Gate::pz(&[0])]),
        (1, vec![rotation(1.0)]),
        (2, vec![rotation(-1.0), Gate::mz(&[0])]),
    ]
    .into_iter()
    .enumerate()
    .map(|(index, (start, gates))| TimedBatch {
        runtime_shot_id: 7,
        batch_index: index as u64,
        start_nanos: start * 1_000_000_000,
        duration_nanos: 0,
        gates,
    })
    .collect()
}
fn scheduled(include: Option<bool>, seed: u64) -> QuantumSystem {
    let mut profile = ScheduledIdleZ::new(1, 0.2, 0.3, 0.4).unwrap();
    if let Some(include) = include {
        profile = profile.with_idle_before_preparation(include);
    }
    let mut system =
        QuantumSystem::new(profile.into_noise_model(), Box::new(StateVecEngine::new(1)));
    system.set_seed(seed);
    system
        .begin_shot(ShotContext {
            run: 1,
            worker: 0,
            shot: 0,
        })
        .unwrap();
    system
}
#[test]
fn preparation_policy_matches_explicit_noise_and_rng_history() {
    let batches = batches();
    for include in [false, true] {
        for seed in 0..32 {
            let mut actual = scheduled(Some(include), seed);
            let z = BTreeMap::from([("Z".into(), 1.0)]);
            let rz = BTreeMap::from([("RZ".into(), 1.0)]);
            let mut expected = QuantumSystem::new(
                Box::new(
                    GeneralNoiseModelBuilder::new()
                        .with_p_idle_linear(0.2, &z)
                        .with_p_idle_sin_squared(0.3, &z)
                        .with_p_idle_coherent(0.4, &rz)
                        .build(),
                ),
                Box::new(StateVecEngine::new(1)),
            );
            expected.set_seed(seed);
            for (i, batch) in batches.iter().enumerate() {
                let mut explicit = ByteMessage::quantum_operations_builder();
                // Independent schedule: prep at t=1, rotation at t=1, inverse
                // rotation/readout at t=2. Only two gaps can produce idle noise.
                if (i == 0 && include) || i == 2 {
                    explicit.idle(1.0, &[0]);
                }
                explicit.add_gate_commands(&batch.gates);
                let want = expected.process(explicit.build()).unwrap();
                let got = actual
                    .process(encode_timed_batches(std::slice::from_ref(batch)).unwrap())
                    .unwrap();
                assert_eq!(got.outcomes().unwrap(), want.outcomes().unwrap());
                assert_eq!(
                    actual.controller().rng().clone().next_u64(),
                    expected.controller().rng().clone().next_u64()
                );
            }
        }
    }
}
#[test]
fn default_keeps_existing_rng_and_omission_changes_rng_even_after_reset() {
    for seed in 0..16 {
        let input = encode_timed_batches(&batches()[..1]).unwrap();
        let mut default = scheduled(None, seed);
        let mut included = scheduled(Some(true), seed);
        let mut omitted = scheduled(Some(false), seed);
        for system in [&mut default, &mut included, &mut omitted] {
            system.process(input.clone()).unwrap();
        }
        assert_eq!(
            default.controller().rng().clone().next_u64(),
            included.controller().rng().clone().next_u64()
        );
        assert_ne!(
            included.controller().rng().clone().next_u64(),
            omitted.controller().rng().clone().next_u64()
        );
    }
}
#[test]
fn policy_preserves_rng_across_input_grouping_and_empty_waits() {
    for include in [false, true] {
        for seed in 0..32 {
            let mut grouped = scheduled(Some(include), seed);
            let mut split = scheduled(Some(include), seed);
            let a = grouped
                .process(encode_timed_batches(&batches()).unwrap())
                .unwrap();
            let mut b = ByteMessage::create_empty();
            for batch in batches() {
                split.process(ByteMessage::create_empty()).unwrap();
                b = split
                    .process(encode_timed_batches(&[batch]).unwrap())
                    .unwrap();
            }
            assert_eq!(a.outcomes().unwrap(), b.outcomes().unwrap());
            assert_eq!(
                grouped.controller().rng().clone().next_u64(),
                split.controller().rng().clone().next_u64()
            );
        }
    }
}

#[test]
fn event_policy_uses_normalized_gates_and_keeps_source_timing_admission() {
    use pecos_core::errors::PecosError;
    use pecos_engines::scheduled_events::{
        ScheduledBatchAdapter, ScheduledEventBatch, ScheduledEventIdleZ, ScheduledEventOp,
        ScheduledGateBuffer, encode_event_batches,
    };
    struct Replace(Gate);
    impl ScheduledBatchAdapter for Replace {
        fn validate(&self, _: &ScheduledEventBatch) -> Result<(), PecosError> {
            Ok(())
        }
        fn translate(
            &mut self,
            _: &ScheduledEventBatch,
            output: &mut ScheduledGateBuffer<'_>,
        ) -> Result<(), PecosError> {
            output.push(self.0.clone())
        }
    }
    for normalized_prep in [false, true] {
        let prep = Gate::pz(&[0]);
        let rotation = Gate::rz(Angle64::ZERO, &[0]);
        let (source, normalized) = if normalized_prep {
            (rotation, prep)
        } else {
            (prep, rotation)
        };
        let replacement = normalized.clone();
        let mut actual = QuantumSystem::new(
            ScheduledEventIdleZ::new(
                ScheduledIdleZ::new(1, 0.2, 0.3, 0.4)
                    .unwrap()
                    .with_idle_before_preparation(false),
                move |_| Ok(Box::new(Replace(replacement.clone()))),
            )
            .into_noise_model(),
            Box::new(StateVecEngine::new(1)),
        );
        actual.set_seed(42);
        actual
            .begin_shot(ShotContext {
                run: 1,
                worker: 0,
                shot: 0,
            })
            .unwrap();
        let input = ScheduledEventBatch {
            runtime_shot_id: 7,
            batch_index: 0,
            start_nanos: 1_000_000_000,
            duration_nanos: 1_000_000_000,
            operations: vec![ScheduledEventOp::Gate(Box::new(source))],
            measurements: vec![],
        };
        actual
            .process(encode_event_batches(std::slice::from_ref(&input)).unwrap())
            .unwrap();
        let mut expected = scheduled(Some(false), 42);
        expected
            .process(
                encode_timed_batches(&[TimedBatch {
                    runtime_shot_id: 7,
                    batch_index: 0,
                    start_nanos: input.start_nanos,
                    duration_nanos: input.duration_nanos,
                    gates: vec![normalized],
                }])
                .unwrap(),
            )
            .unwrap();
        assert_eq!(
            actual.controller().rng().clone().next_u64(),
            expected.controller().rng().clone().next_u64()
        );
        let mut overlap = input;
        overlap.batch_index = 1;
        overlap.start_nanos = 1_500_000_000;
        assert!(
            actual
                .process(encode_event_batches(&[overlap]).unwrap())
                .is_err()
        );
    }
}
