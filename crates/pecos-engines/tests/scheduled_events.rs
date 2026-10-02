use pecos_core::{Angle64, errors::PecosError};
use pecos_engines::EngineSystem;
use pecos_engines::noise::{IntoNoiseModel, PassThroughNoiseModel};
use pecos_engines::runtime_frame::ShotContext;
use pecos_engines::scheduled_events::*;
use pecos_engines::scheduled_frame::{ScheduledIdleZ, encode_timed_batches};
use pecos_engines::{ByteMessage, Engine, Gate, StateVecEngine, quantum_system::QuantumSystem};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

fn err() -> PecosError {
    PecosError::Input("synthetic unsupported event".into())
}
fn rotation(theta: f64) -> Gate {
    Gate::rxy1q(
        Angle64::from_radians(theta),
        Angle64::from_radians(0.0),
        &[0],
    )
}
fn event(code: u8) -> ScheduledEventOp {
    ScheduledEventOp::Custom {
        tag: 42,
        payload: vec![code],
    }
}
fn gate(g: Gate) -> ScheduledEventOp {
    ScheduledEventOp::Gate(Box::new(g))
}
fn batch(index: u64, operations: Vec<ScheduledEventOp>) -> ScheduledEventBatch {
    let measurements = operations
        .iter()
        .enumerate()
        .filter_map(|(i, op)| match op {
            ScheduledEventOp::Gate(g)
                if matches!(
                    g.gate_type,
                    pecos_engines::GateType::MZ | pecos_engines::GateType::MeasureLeaked
                ) =>
            {
                Some(ScheduledResult {
                    operation_index: i,
                    runtime_result: i as u64 + index * 10,
                    program_result: i as u64 + index * 10 + 91,
                })
            }
            _ => None,
        })
        .collect();
    ScheduledEventBatch {
        runtime_shot_id: 12,
        batch_index: index,
        start_nanos: index * 1_000_000_000,
        duration_nanos: 0,
        operations,
        measurements,
    }
}
#[derive(Default)]
struct Synthetic {
    armed: bool,
}
impl ScheduledBatchAdapter for Synthetic {
    fn validate(&self, b: &ScheduledEventBatch) -> Result<(), PecosError> {
        for op in &b.operations {
            if let ScheduledEventOp::Custom { tag, payload } = op
                && (*tag != 42 || payload.len() != 1 || payload[0] > 3)
            {
                return Err(err());
            }
        }
        Ok(())
    }
    fn translate(
        &mut self,
        b: &ScheduledEventBatch,
        out: &mut ScheduledGateBuffer<'_>,
    ) -> Result<(), PecosError> {
        for op in &b.operations {
            match op {
                ScheduledEventOp::Gate(g) => out.push(g.as_ref().clone())?,
                ScheduledEventOp::Custom { payload, .. } => match payload[0] {
                    0 => {}
                    1 => out.push(rotation(std::f64::consts::PI))?,
                    2 => self.armed = true,
                    3 => {
                        if self.armed {
                            out.push(rotation(std::f64::consts::PI))?;
                        }
                        self.armed = false;
                    }
                    _ => return Err(err()),
                },
            }
        }
        Ok(())
    }
}
fn system() -> QuantumSystem {
    QuantumSystem::new(
        ScheduledEventIdleZ::new(ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(), |_| {
            Ok(Box::<Synthetic>::default())
        })
        .into_noise_model(),
        Box::new(StateVecEngine::new(1)),
    )
}
fn begin(q: &mut QuantumSystem) {
    q.reset().unwrap();
    q.begin_shot(ShotContext {
        run: 9,
        worker: 2,
        shot: 3,
    })
    .unwrap();
}
fn run(q: &mut QuantumSystem, batches: &[ScheduledEventBatch]) -> Vec<u32> {
    q.process(encode_event_batches(batches).unwrap())
        .unwrap()
        .outcomes()
        .unwrap()
}
#[test]
fn wire_preserves_timing_positions_payload_and_both_result_ids() {
    let mut b = batch(
        0,
        vec![
            event(0),
            gate(Gate::pz(&[0])),
            event(1),
            gate(Gate::mz(&[0])),
            event(0),
        ],
    );
    b.start_nanos = 17;
    b.duration_nanos = 22;
    let decoded = decode_event_batches(&encode_event_batches(&[b.clone()]).unwrap())
        .unwrap()
        .remove(0);
    assert_eq!(decoded.measurements, b.measurements);
    assert_eq!((decoded.start_nanos, decoded.duration_nanos), (17, 22));
    assert_eq!(
        format!("{:?}", decoded.operations),
        format!("{:?}", b.operations)
    );
    let mut q = system();
    begin(&mut q);
    assert_eq!(run(&mut q, &[b]), vec![1]);
}
#[test]
fn state_persists_across_inputs_but_reset_and_clones_cannot_reuse_a_live_session() {
    let mut q = system();
    begin(&mut q);
    run(&mut q, &[batch(0, vec![event(2)])]);
    let mut clone = q.clone();
    assert!(
        clone
            .begin_shot(ShotContext {
                run: 9,
                worker: 3,
                shot: 3
            })
            .is_err()
    );
    assert_eq!(
        run(&mut q, &[batch(1, vec![event(3), gate(Gate::mz(&[0]))])]),
        vec![1]
    );
    begin(&mut clone);
    assert_eq!(
        run(
            &mut clone,
            &[batch(0, vec![event(3), gate(Gate::mz(&[0]))])]
        ),
        vec![0]
    );
    begin(&mut q);
    assert_eq!(
        run(&mut q, &[batch(0, vec![event(3), gate(Gate::mz(&[0]))])]),
        vec![0]
    );
}
#[test]
fn unsupported_late_event_rejects_before_translation_and_poison_is_latched() {
    struct Counting(Arc<AtomicUsize>);
    impl ScheduledBatchAdapter for Counting {
        fn validate(&self, b: &ScheduledEventBatch) -> Result<(), PecosError> {
            Synthetic::default().validate(b)
        }
        fn translate(
            &mut self,
            _: &ScheduledEventBatch,
            _: &mut ScheduledGateBuffer<'_>,
        ) -> Result<(), PecosError> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(())
        }
    }
    let count = Arc::new(AtomicUsize::new(0));
    let c = count.clone();
    let mut q = QuantumSystem::new(
        ScheduledEventIdleZ::new(ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(), move |_| {
            Ok(Box::new(Counting(c.clone())))
        })
        .into_noise_model(),
        Box::new(StateVecEngine::new(1)),
    );
    begin(&mut q);
    let bad = encode_event_batches(&[batch(0, vec![event(1)]), batch(1, vec![event(9)])]).unwrap();
    assert!(q.process(bad).is_err());
    assert_eq!(count.load(Ordering::Relaxed), 0);
    assert!(
        q.process(encode_event_batches(&[batch(0, vec![])]).unwrap())
            .is_err()
    );
    let mut clone = q.clone();
    assert!(
        clone
            .begin_shot(ShotContext {
                run: 0,
                worker: 0,
                shot: 0
            })
            .is_err()
    );
    begin(&mut q);
    run(&mut q, &[batch(0, vec![])]);
}
#[test]
fn malformed_wire_rejects_before_factory_and_never_reaches_legacy_consumers() {
    let count = Arc::new(AtomicUsize::new(0));
    let c = count.clone();
    let mut q = QuantumSystem::new(
        ScheduledEventIdleZ::new(ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(), move |_| {
            c.fetch_add(1, Ordering::Relaxed);
            Ok(Box::<Synthetic>::default())
        })
        .into_noise_model(),
        Box::new(StateVecEngine::new(1)),
    );
    begin(&mut q);
    let good = encode_event_batches(&[batch(0, vec![event(0)])]).unwrap();
    for n in 0..good.as_bytes().len() {
        let mut truncated = good.as_bytes()[..n].to_vec();
        // Keep the declared total honest, so inner decoder bounds are exercised.
        if n >= 16 {
            truncated[12..16].copy_from_slice(&u32::try_from(n).unwrap().to_le_bytes());
        }
        assert!(q.process(ByteMessage::new(&truncated)).is_err());
    }
    assert_eq!(count.load(Ordering::Relaxed), 0);
    let mut old = QuantumSystem::new(
        Box::new(PassThroughNoiseModel::builder().build()),
        Box::new(StateVecEngine::new(1)),
    );
    assert!(old.process(good.clone()).is_err());
    let mut v3 = QuantumSystem::new(
        ScheduledIdleZ::new(1, 0.0, 0.0, 0.0)
            .unwrap()
            .into_noise_model(),
        Box::new(StateVecEngine::new(1)),
    );
    begin(&mut v3);
    assert!(v3.process(good).is_err());
    let mut too_large = batch(
        0,
        vec![ScheduledEventOp::Custom {
            tag: 42,
            payload: vec![0; MAX_BATCH_PAYLOAD + 1],
        }],
    );
    assert!(encode_event_batches(&[too_large.clone()]).is_err());
    too_large.operations = vec![event(0); MAX_BATCH_OPERATIONS + 1];
    assert!(encode_event_batches(&[too_large]).is_err());
}
#[test]
fn source_capacity_and_identity_reject_before_callbacks() {
    let count = Arc::new(AtomicUsize::new(0));
    let c = count.clone();
    let mut q = QuantumSystem::new(
        ScheduledEventIdleZ::new(ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(), move |_| {
            c.fetch_add(1, Ordering::Relaxed);
            Ok(Box::<Synthetic>::default())
        })
        .into_noise_model(),
        Box::new(StateVecEngine::new(1)),
    );
    begin(&mut q);
    for b in [batch(1, vec![]), batch(0, vec![gate(Gate::pz(&[1]))])] {
        assert!(q.process(encode_event_batches(&[b]).unwrap()).is_err());
    }
    assert_eq!(count.load(Ordering::Relaxed), 0);
}
#[test]
fn metadata_insertion_preserves_seeded_idle_behavior_and_empty_batch_lifecycles() {
    for (linear, sine, coherent) in [(0.3, 0.0, 0.0), (0.0, 0.7, 0.0), (0.0, 0.0, 0.4)] {
        let profile = ScheduledIdleZ::new(1, linear, sine, coherent).unwrap();
        let mut old = QuantumSystem::new(
            profile.clone().into_noise_model(),
            Box::new(StateVecEngine::new(1)),
        );
        let mut new = QuantumSystem::new(
            ScheduledEventIdleZ::new(profile, |_| Ok(Box::<Synthetic>::default()))
                .into_noise_model(),
            Box::new(StateVecEngine::new(1)),
        );
        let batches = vec![
            batch(
                0,
                vec![gate(rotation(std::f64::consts::FRAC_PI_2)), event(0)],
            ),
            batch(1, vec![event(0)]),
            batch(
                2,
                vec![
                    event(0),
                    gate(rotation(-std::f64::consts::FRAC_PI_2)),
                    event(0),
                    gate(Gate::mz(&[0])),
                ],
            ),
        ];
        let legacy: Vec<_> = batches
            .iter()
            .map(ScheduledEventBatch::source_gates)
            .collect();
        for seed in 0..32 {
            begin(&mut old);
            begin(&mut new);
            old.set_seed(seed);
            new.set_seed(seed);
            assert_eq!(
                old.process(encode_timed_batches(&legacy).unwrap())
                    .unwrap()
                    .outcomes()
                    .unwrap(),
                run(&mut new, &batches)
            );
            // Compare the next model RNG draw as well as observed outcomes.
            assert_eq!(
                old.controller().rng().clone().next_u64(),
                new.controller().rng().clone().next_u64()
            );
        }
    }
}
#[test]
fn callback_panic_cannot_be_recovered_without_whole_host_reset() {
    let factory = Arc::new(AtomicUsize::new(0));
    let f = factory.clone();
    let mut q = QuantumSystem::new(
        ScheduledEventIdleZ::new(ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(), move |_| {
            assert_ne!(
                f.fetch_add(1, Ordering::Relaxed),
                0,
                "synthetic factory failure"
            );
            Ok(Box::<Synthetic>::default())
        })
        .into_noise_model(),
        Box::new(StateVecEngine::new(1)),
    );
    begin(&mut q);
    let msg = encode_event_batches(&[batch(0, vec![event(0)])]).unwrap();
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| q.process(msg.clone()))).is_err()
    );
    assert!(q.process(msg.clone()).is_err());
    begin(&mut q);
    q.process(msg).unwrap();
}
#[test]
fn dropped_measurements_poison_before_quantum_execution() {
    struct DropMeasurements;
    impl ScheduledBatchAdapter for DropMeasurements {
        fn validate(&self, _: &ScheduledEventBatch) -> Result<(), PecosError> {
            Ok(())
        }
        fn translate(
            &mut self,
            _: &ScheduledEventBatch,
            _: &mut ScheduledGateBuffer<'_>,
        ) -> Result<(), PecosError> {
            Ok(())
        }
    }
    let mut q = QuantumSystem::new(
        ScheduledEventIdleZ::new(ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(), |_| {
            Ok(Box::new(DropMeasurements))
        })
        .into_noise_model(),
        Box::new(StateVecEngine::new(1)),
    );
    begin(&mut q);
    let msg = encode_event_batches(&[batch(0, vec![gate(Gate::mz(&[0]))])]).unwrap();
    assert!(q.process(msg.clone()).is_err());
    assert!(q.process(msg).is_err());
}

#[test]
fn event_position_relative_to_measurements_is_preserved() {
    let mut q = QuantumSystem::new(
        ScheduledEventIdleZ::new(ScheduledIdleZ::new(2, 0.0, 0.0, 0.0).unwrap(), |_| {
            Ok(Box::<Synthetic>::default())
        })
        .into_noise_model(),
        Box::new(StateVecEngine::new(2)),
    );
    begin(&mut q);
    assert_eq!(
        run(
            &mut q,
            &[batch(
                0,
                vec![gate(Gate::mz(&[1])), event(1), gate(Gate::mz(&[0]))]
            )]
        ),
        vec![0, 1]
    );
}
#[test]
fn caught_output_rejection_stays_latched() {
    struct Swallow;
    impl ScheduledBatchAdapter for Swallow {
        fn validate(&self, _: &ScheduledEventBatch) -> Result<(), PecosError> {
            Ok(())
        }
        fn translate(
            &mut self,
            _: &ScheduledEventBatch,
            out: &mut ScheduledGateBuffer<'_>,
        ) -> Result<(), PecosError> {
            assert!(out.push(Gate::x(&[0])).is_err());
            assert!(out.push(Gate::pz(&[0])).is_err());
            Ok(())
        }
    }
    let mut q = QuantumSystem::new(
        ScheduledEventIdleZ::new(ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(), |_| {
            Ok(Box::new(Swallow))
        })
        .into_noise_model(),
        Box::new(StateVecEngine::new(1)),
    );
    begin(&mut q);
    assert!(
        q.process(encode_event_batches(&[batch(0, vec![event(0)])]).unwrap())
            .is_err()
    );
}

#[test]
fn factory_sessions_receive_distinct_worker_context_and_never_share_state() {
    let contexts = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen = contexts.clone();
    let template = QuantumSystem::new(
        ScheduledEventIdleZ::new(
            ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(),
            move |context| {
                seen.lock().unwrap().push(context);
                Ok(Box::<Synthetic>::default())
            },
        )
        .into_noise_model(),
        Box::new(StateVecEngine::new(1)),
    );
    let mut first = template.clone();
    let mut second = template.clone();
    for (worker, q) in [(0, &mut first), (1, &mut second)] {
        q.reset().unwrap();
        q.begin_shot(ShotContext {
            run: 5,
            worker,
            shot: 6,
        })
        .unwrap();
    }
    run(&mut first, &[batch(0, vec![event(2)])]);
    assert_eq!(
        run(
            &mut second,
            &[batch(0, vec![event(3), gate(Gate::mz(&[0]))])]
        ),
        vec![0]
    );
    assert_eq!(
        run(
            &mut first,
            &[batch(1, vec![event(3), gate(Gate::mz(&[0]))])]
        ),
        vec![1]
    );
    let seen = contexts.lock().unwrap();
    assert_eq!(seen.len(), 2);
    assert_eq!(seen[0].worker, 0);
    assert_eq!(seen[1].worker, 1);
}

#[test]
fn removed_gates_still_reserve_source_timing_across_inputs() {
    struct DropRz(Arc<AtomicUsize>);
    impl ScheduledBatchAdapter for DropRz {
        fn validate(&self, _: &ScheduledEventBatch) -> Result<(), PecosError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn translate(
            &mut self,
            b: &ScheduledEventBatch,
            out: &mut ScheduledGateBuffer<'_>,
        ) -> Result<(), PecosError> {
            self.0.fetch_add(1, Ordering::SeqCst);
            for op in &b.operations {
                match op {
                    ScheduledEventOp::Gate(g) if g.gate_type != pecos_engines::GateType::RZ => {
                        out.push(g.as_ref().clone())?;
                    }
                    ScheduledEventOp::Custom { .. } => out.push(Gate::pz(&[0]))?,
                    ScheduledEventOp::Gate(_) => {}
                }
            }
            Ok(())
        }
    }
    let calls = Arc::new(AtomicUsize::new(0));
    let seen = calls.clone();
    let mut q = QuantumSystem::new(
        ScheduledEventIdleZ::new(ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(), move |_| {
            Ok(Box::new(DropRz(seen.clone())))
        })
        .into_noise_model(),
        Box::new(StateVecEngine::new(1)),
    );
    let mut first = batch(0, vec![gate(Gate::rz(Angle64::ZERO, &[0]))]);
    first.duration_nanos = 10;
    let mut overlap = batch(1, vec![gate(Gate::pz(&[0]))]);
    overlap.start_nanos = 5;
    begin(&mut q);
    assert!(
        q.process(encode_event_batches(&[first.clone(), overlap.clone()]).unwrap())
            .is_err()
    );
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    run(&mut q, &[first]);
    assert_eq!(calls.load(Ordering::SeqCst), 2);
    assert!(
        q.process(encode_event_batches(&[overlap.clone()]).unwrap())
            .is_err()
    );
    assert_eq!(
        calls.load(Ordering::SeqCst),
        2,
        "overlap must reject before callbacks"
    );
    // Rejection must not commit the ordinal. The corrected input can still run.
    overlap.start_nanos = 10;
    run(&mut q, &[overlap]);
    let mut clone = q.clone();
    for host in [&mut q, &mut clone] {
        begin(host);
        let mut restarted = batch(0, vec![gate(Gate::pz(&[0]))]);
        restarted.start_nanos = 5;
        run(host, &[restarted]);
    }
    // Expanded gates reserve the normalized timeline, not the source timeline.
    begin(&mut q);
    let mut generated = batch(0, vec![event(0)]);
    generated.duration_nanos = 10;
    run(&mut q, &[generated]);
    let mut removed = batch(1, vec![gate(Gate::rz(Angle64::ZERO, &[0]))]);
    removed.start_nanos = 5;
    run(&mut q, &[removed]);
    let mut normalized_overlap = batch(2, vec![gate(Gate::pz(&[0]))]);
    normalized_overlap.start_nanos = 5;
    assert!(
        q.process(encode_event_batches(&[normalized_overlap]).unwrap())
            .is_err()
    );
    assert!(
        q.process(encode_event_batches(&[batch(2, vec![])]).unwrap())
            .is_err()
    );
}

#[test]
fn component_clones_cannot_restart_an_adapter_in_a_live_native_shot() {
    let mut original = system();
    begin(&mut original);
    run(&mut original, &[batch(0, vec![event(2)])]);
    let mut copy = QuantumSystem::new(original.controller().clone(), original.engine().clone());
    copy.begin_shot(ShotContext {
        run: 10,
        worker: 0,
        shot: 0,
    })
    .unwrap();
    let continuation = batch(1, vec![event(3), gate(Gate::mz(&[0]))]);
    assert!(matches!(
        copy.process(encode_event_batches(std::slice::from_ref(&continuation)).unwrap()),
        Err(PecosError::Processing(message)) if message.contains("cloned event session requires reset")
    ));
    assert_eq!(run(&mut original, &[continuation]), vec![1]);
    begin(&mut copy);
    assert_eq!(
        run(&mut copy, &[batch(0, vec![event(3), gate(Gate::mz(&[0]))])]),
        vec![0]
    );
}

#[test]
fn changed_measurement_target_or_kind_rejects_and_poisons() {
    struct Rewrite(Gate);
    impl ScheduledBatchAdapter for Rewrite {
        fn validate(&self, _: &ScheduledEventBatch) -> Result<(), PecosError> {
            Ok(())
        }
        fn translate(
            &mut self,
            _: &ScheduledEventBatch,
            out: &mut ScheduledGateBuffer<'_>,
        ) -> Result<(), PecosError> {
            out.push(self.0.clone())
        }
    }
    for replacement in [Gate::mz(&[1]), Gate::measure_leaked(&[0])] {
        let mut q = QuantumSystem::new(
            ScheduledEventIdleZ::new(ScheduledIdleZ::new(2, 0.0, 0.0, 0.0).unwrap(), move |_| {
                Ok(Box::new(Rewrite(replacement.clone())))
            })
            .into_noise_model(),
            Box::new(StateVecEngine::new(2)),
        );
        begin(&mut q);
        let error = q
            .process(encode_event_batches(&[batch(0, vec![gate(Gate::mz(&[0]))])]).unwrap())
            .err()
            .expect("expected rejection");
        assert!(
            error
                .to_string()
                .contains("measurement order, kind or targets")
        );
        assert!(
            q.process(encode_event_batches(&[batch(0, vec![])]).unwrap())
                .is_err()
        );
    }
}

#[test]
fn output_operation_bound_accepts_4096_and_rejects_4097_with_measurements_preserved() {
    struct Expand(usize);
    impl ScheduledBatchAdapter for Expand {
        fn validate(&self, _: &ScheduledEventBatch) -> Result<(), PecosError> {
            Ok(())
        }
        fn translate(
            &mut self,
            b: &ScheduledEventBatch,
            out: &mut ScheduledGateBuffer<'_>,
        ) -> Result<(), PecosError> {
            // Fill the normalized batch while preserving its original readout.
            for _ in 1..self.0 {
                out.push(Gate::pz(&[0]))?;
            }
            for op in &b.operations {
                if let ScheduledEventOp::Gate(g) = op {
                    out.push(g.as_ref().clone())?;
                }
            }
            Ok(())
        }
    }
    for count in [MAX_BATCH_OPERATIONS, MAX_BATCH_OPERATIONS + 1] {
        let mut q = QuantumSystem::new(
            ScheduledEventIdleZ::new(ScheduledIdleZ::new(1, 0.0, 0.0, 0.0).unwrap(), move |_| {
                Ok(Box::new(Expand(count)))
            })
            .into_noise_model(),
            Box::new(StateVecEngine::new(1)),
        );
        begin(&mut q);
        let result =
            q.process(encode_event_batches(&[batch(0, vec![gate(Gate::mz(&[0]))])]).unwrap());
        if count == MAX_BATCH_OPERATIONS {
            assert_eq!(result.unwrap().outcomes().unwrap(), vec![0]);
        } else {
            assert!(
                result
                    .err()
                    .expect("expected rejection")
                    .to_string()
                    .contains("event expansion limit")
            );
        }
    }
}

#[test]
fn wire_measurement_positions_and_each_identity_namespace_are_checked() {
    let first = batch(0, vec![event(0), gate(Gate::mz(&[0]))]);
    let second = batch(1, vec![gate(Gate::mz(&[0]))]);
    let one = encode_event_batches(std::slice::from_ref(&first)).unwrap();
    let good = encode_event_batches(&[first, second]).unwrap();
    // The final 24 bytes of each batch hold (position, native ID, program ID).
    for (offset, value, expected) in [
        (one.as_bytes().len() - 24, 0, "measurement positions"),
        (
            good.as_bytes().len() - 16,
            1,
            "duplicate scheduled event measurement identity",
        ),
        (
            good.as_bytes().len() - 8,
            92,
            "duplicate scheduled event measurement identity",
        ),
    ] {
        let mut bytes = good.as_bytes().to_vec();
        bytes[offset..offset + 8].copy_from_slice(&u64::to_le_bytes(value));
        let mut q = system();
        begin(&mut q);
        assert!(
            q.process(ByteMessage::new(&bytes))
                .err()
                .expect("expected rejection")
                .to_string()
                .contains(expected)
        );
        // Malformed wire has not invoked the adapter or consumed batch 0.
        assert_eq!(
            run(&mut q, &decode_event_batches(&good).unwrap()),
            vec![0, 0]
        );
    }
}

#[test]
fn wire_counts_and_event_positions_reject_before_record_allocation() {
    fn patch(bytes: &[u8], offset: usize, value: u64, message: &str) {
        let mut wire = bytes.to_vec();
        wire[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        assert!(
            decode_event_batches(&ByteMessage::new(&wire))
                .expect_err("reject malformed wire")
                .to_string()
                .contains(message)
        );
    }
    let good =
        encode_event_batches(&[batch(0, vec![event(0), event(0), gate(Gate::mz(&[0]))])]).unwrap();
    let bytes = good.as_bytes();
    let mut bad_count = bytes.to_vec();
    bad_count[8..12].copy_from_slice(&100u32.to_le_bytes());
    assert!(
        decode_event_batches(&ByteMessage::new(&bad_count))
            .err()
            .unwrap()
            .to_string()
            .contains("batch count")
    );
    let nested_bytes = 56 + 40;
    let events = 24 + nested_bytes;
    patch(
        bytes,
        16,
        (56 + MAX_BATCH_OPERATIONS * 40 + 1) as u64,
        "gate count limit",
    );
    patch(
        bytes,
        events,
        (MAX_BATCH_OPERATIONS + 1) as u64,
        "operation count limit",
    );
    patch(bytes, events + 8 + 25, 3, "invalid event position");
    patch(bytes, events + 8 + 25, 0, "invalid event position");
    patch(bytes, bytes.len() - 32, 4, "measurement count limit");
    let max_ops = batch(0, vec![event(0); MAX_BATCH_OPERATIONS]);
    assert_eq!(
        decode_event_batches(&encode_event_batches(&[max_ops]).unwrap()).unwrap()[0]
            .operations
            .len(),
        MAX_BATCH_OPERATIONS
    );
    let max_payload = batch(
        0,
        vec![ScheduledEventOp::Custom {
            tag: 42,
            payload: vec![0; MAX_BATCH_PAYLOAD],
        }],
    );
    assert!(decode_event_batches(&encode_event_batches(&[max_payload]).unwrap()).is_ok());
}
