use pecos_core::{Angle64, errors::PecosError};
use pecos_engines::noise::IntoNoiseModel;
use pecos_engines::quantum_system::QuantumSystem;
use pecos_engines::runtime_frame::ShotContext;
use pecos_engines::scheduled_frame::{ScheduledIdleZ, TimedBatch, encode_timed_batches};
use pecos_engines::{ByteMessage, Engine, Gate, StateVecEngine};
fn system(rate: f64) -> QuantumSystem {
    let mut s = QuantumSystem::new(
        ScheduledIdleZ::new(2, 0.0, 0.0, rate)
            .unwrap()
            .into_noise_model(),
        Box::new(StateVecEngine::new(2)),
    );
    s.set_seed(4);
    s.begin_shot(ShotContext {
        run: 1,
        worker: 0,
        shot: 0,
    })
    .unwrap();
    s
}
fn ramsey(gap: u64) -> Vec<TimedBatch> {
    vec![
        TimedBatch {
            runtime_shot_id: 7,
            batch_index: 0,
            start_nanos: 0,
            duration_nanos: 100,
            gates: vec![Gate::rxy1q(
                Angle64::from_radians(std::f64::consts::FRAC_PI_2),
                Angle64::from_radians(std::f64::consts::FRAC_PI_2),
                &[0],
            )],
        },
        TimedBatch {
            runtime_shot_id: 7,
            batch_index: 1,
            start_nanos: 100 + gap,
            duration_nanos: 10,
            gates: vec![
                Gate::rxy1q(
                    Angle64::from_radians(-std::f64::consts::FRAC_PI_2),
                    Angle64::from_radians(std::f64::consts::FRAC_PI_2),
                    &[0],
                ),
                Gate::mz(&[0]),
            ],
        },
    ]
}
#[test]
fn production_timing_changes_ramsey_and_preserves_submission_boundaries() {
    for gap in [0, 1_000_000_000] {
        let mut together = system(std::f64::consts::PI);
        let mut split = system(std::f64::consts::PI);
        let batches = ramsey(gap);
        let a = together
            .process(encode_timed_batches(&batches).unwrap())
            .unwrap()
            .outcomes()
            .unwrap();
        assert!(
            split
                .process(encode_timed_batches(&batches[..1]).unwrap())
                .unwrap()
                .outcomes()
                .unwrap()
                .is_empty()
        );
        let b = split
            .process(encode_timed_batches(&batches[1..]).unwrap())
            .unwrap()
            .outcomes()
            .unwrap();
        assert_eq!(a, b);
        assert_eq!(a, vec![u32::from(gap != 0)]);
    }
}
#[test]
fn unsupported_consumers_and_legacy_bypass_reject() {
    let input = encode_timed_batches(&ramsey(1)).unwrap();
    let mut plain = QuantumSystem::new_without_noise(Box::new(StateVecEngine::new(2)));
    assert!(plain.process(input).is_err());
    let mut scheduled = system(0.0);
    assert!(
        scheduled
            .process(ByteMessage::quantum_operations_builder().x(&[0]).build())
            .is_err()
    );
    let unsupported = TimedBatch {
        gates: vec![Gate::x(&[0])],
        ..ramsey(0).remove(0)
    };
    assert!(encode_timed_batches(&[unsupported]).is_err());
}
#[test]
fn malformed_transport_and_unknown_records_return_errors_without_panics() {
    let input = encode_timed_batches(&ramsey(1)).unwrap();
    let original = input.as_bytes();
    for length in 0..original.len() {
        assert!(
            system(0.0)
                .process(ByteMessage::new(&original[..length]))
                .is_err(),
            "prefix {length}"
        );
    }
    for (offset, bytes) in [
        (4, vec![9]),
        (8, u32::MAX.to_le_bytes().to_vec()),
        (48, u64::MAX.to_le_bytes().to_vec()),
        (56, 99_u64.to_le_bytes().to_vec()),
        (80, f64::NAN.to_bits().to_le_bytes().to_vec()),
    ] {
        let mut mutated = original.to_vec();
        mutated[offset..offset + bytes.len()].copy_from_slice(&bytes);
        assert!(
            system(0.0).process(ByteMessage::new(&mutated)).is_err(),
            "offset {offset}"
        );
    }
}
#[test]
fn later_invalid_batch_rejects_before_state_or_cursor_mutation() {
    let mut input = ramsey(1);
    input[0].gates = vec![Gate::rxy1q(
        Angle64::from_radians(std::f64::consts::PI),
        Angle64::ZERO,
        &[0],
    )];
    input[1].gates = vec![Gate::mz(&[2])];
    let mut s = system(0.0);
    assert!(s.process(encode_timed_batches(&input).unwrap()).is_err());
    let probe = TimedBatch {
        gates: vec![Gate::mz(&[0])],
        ..ramsey(0).remove(0)
    };
    assert_eq!(
        s.process(encode_timed_batches(&[probe]).unwrap())
            .unwrap()
            .outcomes()
            .unwrap(),
        vec![0]
    );
}
#[test]
fn identities_overlaps_and_overflow_reject_without_consuming_state() {
    for case in 0..4 {
        let mut input = ramsey(1);
        match case {
            0 => input[1].batch_index = 2,
            1 => input[1].runtime_shot_id = 8,
            2 => input[1].start_nanos = 0,
            _ => input[1].duration_nanos = u64::MAX,
        }
        let mut s = system(0.0);
        assert!(s.process(encode_timed_batches(&input).unwrap()).is_err());
        assert!(s.process(encode_timed_batches(&ramsey(0)).unwrap()).is_ok());
    }
}
#[test]
fn cloned_live_owner_requires_reset_and_context() {
    let mut s = system(0.0);
    s.process(encode_timed_batches(&ramsey(0)[..1]).unwrap())
        .unwrap();
    let mut cloned = s.clone();
    assert!(
        cloned
            .begin_shot(ShotContext {
                run: 2,
                worker: 0,
                shot: 0
            })
            .is_err()
    );
    cloned.reset().unwrap();
    cloned
        .begin_shot(ShotContext {
            run: 2,
            worker: 0,
            shot: 0,
        })
        .unwrap();
    assert_eq!(
        cloned
            .process(encode_timed_batches(&ramsey(0)).unwrap())
            .unwrap()
            .outcomes()
            .unwrap(),
        vec![0]
    );
}
#[test]
fn capacity_and_profile_admission_are_explicit() {
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        assert!(ScheduledIdleZ::new(2, bad, 0.0, 0.0).is_err());
    }
    assert!(ScheduledIdleZ::new(17, 0.0, 0.0, 0.0).is_err());
    let mut undersized = QuantumSystem::new(
        ScheduledIdleZ::new(2, 0.0, 0.0, 0.0)
            .unwrap()
            .into_noise_model(),
        Box::new(StateVecEngine::new(1)),
    );
    undersized
        .begin_shot(ShotContext {
            run: 1,
            worker: 0,
            shot: 0,
        })
        .unwrap();
    assert!(matches!(
        undersized.process(encode_timed_batches(&ramsey(0)).unwrap()),
        Err(PecosError::Input(_))
    ));
    assert!(
        system(f64::MAX)
            .process(encode_timed_batches(&ramsey(3_000_000_000)).unwrap())
            .is_err()
    );
}

#[test]
fn stochastic_profile_keeps_rng_and_whole_idle_across_inputs() {
    // A nontrivial sine rate detects accidental per-input idle restart/splitting.
    let batches = ramsey(900_000_000);
    let make = |seed| {
        let mut s = QuantumSystem::new(
            ScheduledIdleZ::new(2, 0.1, 0.8, 0.3)
                .unwrap()
                .into_noise_model(),
            Box::new(StateVecEngine::new(2)),
        );
        s.set_seed(seed);
        s.begin_shot(ShotContext {
            run: 1,
            worker: 0,
            shot: usize::try_from(seed).unwrap(),
        })
        .unwrap();
        s
    };
    let mut observed = [false; 2];
    for seed in 0..64 {
        let mut together = make(seed);
        let mut split = make(seed);
        let a = together
            .process(encode_timed_batches(&batches).unwrap())
            .unwrap();
        split
            .process(encode_timed_batches(&batches[..1]).unwrap())
            .unwrap();
        // Empty extraction must not reset timing or advance the RNG.
        split.process(encode_timed_batches(&[]).unwrap()).unwrap();
        let b = split
            .process(encode_timed_batches(&batches[1..]).unwrap())
            .unwrap();
        assert_eq!(a.as_bytes(), b.as_bytes());
        observed[a.outcomes().unwrap()[0] as usize] = true;
    }
    assert_eq!(observed, [true, true]);
}
