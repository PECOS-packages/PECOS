//! Regression coverage for the after-gate operand partition and handler marks.
use super::*;
use crate::command::CommandBuilder;
use crate::extensible::{
    DecompEntry, DecompOp, Decomposition, GateCategory, GateSpec, GateSupportSet,
};
use crate::gates;
use crate::noise::*;
use pecos_simulators::SparseStab;
use std::sync::Mutex;

#[derive(Clone, Debug, PartialEq)]
struct Shape {
    executed: Vec<QubitId>,
    suppressed: Vec<QubitId>,
    size: usize,
}

#[derive(Clone, Default)]
struct Fixture {
    skip: Vec<QubitId>,
    after: Arc<Mutex<Vec<Shape>>>,
    unleak_child: bool,
    seep_before: bool,
}
impl NoiseChannel for Fixture {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        event.is_gate_event()
    }
    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> NoiseResponse {
        match event {
            NoiseEvent::BeforeGate { gate_type, .. }
                if !self.unleak_child || *gate_type == GateType::RZZ =>
            {
                let response = NoiseResponse::SkipGate(self.skip.iter().copied().collect());
                if self.seep_before {
                    response.combine(NoiseResponse::MarkUnleaked(smallvec::smallvec![QubitId(0)]))
                } else {
                    response
                }
            }
            NoiseEvent::AfterGate {
                gate_type: GateType::X,
                ..
            } if self.unleak_child => NoiseResponse::MarkUnleaked(smallvec::smallvec![QubitId(0)]),
            NoiseEvent::AfterGate {
                qubits,
                leaked_suppressed,
                group_size,
                ..
            } => {
                self.after.lock().unwrap().push(Shape {
                    executed: qubits.to_vec(),
                    suppressed: leaked_suppressed.to_vec(),
                    size: group_size.get(),
                });
                NoiseResponse::None
            }
            _ => NoiseResponse::None,
        }
    }
    fn name(&self) -> &'static str {
        "partition fixture"
    }
    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}
fn execute(
    runner: &mut CircuitRunner<SparseStab>,
    sim: &mut SparseStab,
    adapted: bool,
    gate: GateType,
    qubits: &[QubitId],
    angles: &[Angle64],
) {
    if adapted {
        runner
            .execute_gate(sim, gate.to_gate_id(), qubits, angles, 0)
            .unwrap();
    } else {
        runner.apply_gate(sim, gate, qubits, angles).unwrap();
    }
}

#[test]
fn suppressed_real_gate_error_channels() {
    use crate::noise::composite::prelude::*;
    let channels: Vec<Box<dyn NoiseChannel>> = vec![
        Box::new(
            GateDependentChannel::new().with_gate(
                GateType::CX,
                GateNoiseConfig::new(1.0)
                    .unwrap()
                    .with_pauli_weights(crate::noise::PauliWeights::x_only())
                    .unwrap(),
            ),
        ),
        Box::new(
            GateIdDependentChannel::new().with_gate(
                gates::CX,
                GateIdNoiseConfig::new(1.0)
                    .unwrap()
                    .with_pauli_weights(crate::noise::PauliWeights::x_only())
                    .unwrap(),
            ),
        ),
        Box::new(
            CategoryBasedChannel::new()
                .with_category_weights(
                    GateCategory::TwoQubitUnitary,
                    1.0,
                    crate::noise::PauliWeights::x_only(),
                )
                .unwrap(),
        ),
        Box::new(
            CorrelatedNoiseChannel::new(1.0, 0.0)
                .unwrap()
                .with_pauli_weights(crate::noise::PauliWeights::x_only())
                .unwrap(),
        ),
        Box::new(
            CompositeChannel::new(
                "gate error",
                prob(
                    1.0,
                    Pauli::new(
                        crate::noise::composite::PauliWeights::custom(1.0, 0.0, 0.0).unwrap(),
                    )
                    .unwrap(),
                )
                .unwrap(),
            )
            .with_filter(CompositeEventFilter::TwoQubitGate),
        ),
        Box::new(
            CompositeChannel::new(
                "wrapped gate error",
                channel_action(
                    GateDependentChannel::new().with_gate(
                        GateType::CX,
                        GateNoiseConfig::new(1.0)
                            .unwrap()
                            .with_pauli_weights(crate::noise::PauliWeights::x_only())
                            .unwrap(),
                    ),
                ),
            )
            .with_filter(CompositeEventFilter::TwoQubitGate),
        ),
    ];
    let mut failures = Vec::new();
    for channel in channels {
        for adapted in [false, true] {
            let mut noise = ComposableNoiseModel::new()
                .add_channel(Fixture {
                    skip: vec![QubitId(0)],
                    ..Default::default()
                })
                .add_boxed_channel(channel.clone_box());
            noise.context_mut().mark_leaked(QubitId(0));
            let mut runner = CircuitRunner::new().with_noise(noise);
            let mut sim = SparseStab::with_seed(2, 7);
            execute(
                &mut runner,
                &mut sim,
                adapted,
                GateType::CX,
                &[QubitId(0), QubitId(1)],
                &[],
            );
            if sim.mz(&[QubitId(1)])[0].outcome {
                failures.push(format!("{} adapted={adapted}", channel.name()));
            }
        }
    }
    assert!(
        failures.is_empty(),
        "gate errors reached suppressed partners: {failures:?}"
    );
}

#[test]
fn suppressed_mid_command_unleak() {
    let mut weights = [0.0; 24];
    weights[13] = 1.0;
    let mut disturbances = Vec::new();
    for adapted in [false, true] {
        let mut disturbed = 0;
        for seed in 0..40 {
            let mut definitions = GateDefinitions::new();
            definitions.set_decomposition(
                gates::RZZ,
                DecompEntry {
                    requires: GateSupportSet::from_iter([gates::X]),
                    decomposition: Decomposition::Dynamic(Arc::new(vec![DecompOp::gate1(
                        gates::X,
                        0,
                    )])),
                },
            );
            let mut noise = ComposableNoiseModel::new()
                .add_channel(Fixture {
                    skip: vec![QubitId(0)],
                    unleak_child: true,
                    ..Default::default()
                })
                .add_channel(
                    TwoQubitChannel::depolarizing(1.0)
                        .unwrap()
                        .with_emission_weights(
                            1.0,
                            TwoQubitEmissionWeights::custom(weights).unwrap(),
                        )
                        .unwrap(),
                );
            noise.context_mut().mark_leaked(QubitId(0));
            let mut runner = CircuitRunner::with_definitions(definitions)
                .with_noise(noise)
                .with_seed(seed);
            let mut sim = SparseStab::with_seed(4, seed);
            sim.h(&[QubitId(0), QubitId(1)]);
            execute(
                &mut runner,
                &mut sim,
                adapted,
                GateType::RZZ,
                &[QubitId(0), QubitId(1), QubitId(2), QubitId(3)],
                &[Angle64::QUARTER_TURN],
            );
            assert!(
                !runner
                    .noise
                    .as_ref()
                    .unwrap()
                    .context()
                    .is_leaked(QubitId(0))
            );
            sim.h(&[QubitId(0), QubitId(1)]);
            if sim.mz(&[QubitId(0), QubitId(1)]).iter().any(|m| m.outcome) {
                disturbed += 1;
            }
        }
        disturbances.push(disturbed);
    }
    assert_eq!(
        disturbances,
        vec![0, 0],
        "disturbed seeds out of 40, queue then adapted"
    );
}

#[test]
fn suppressed_event_shape_and_handler_split() {
    for adapted in [false, true] {
        for (skip, leaked, expected) in [
            (
                vec![0, 2],
                vec![0],
                Some(Shape {
                    executed: vec![QubitId(4), QubitId(5)],
                    suppressed: vec![QubitId(0), QubitId(1)],
                    size: 2,
                }),
            ),
            (
                vec![0, 2, 4],
                vec![0, 2, 4],
                Some(Shape {
                    executed: vec![],
                    suppressed: (0..6).map(QubitId).collect(),
                    size: 2,
                }),
            ),
            (vec![0, 2, 4], vec![], None),
        ] {
            let fixture = Fixture {
                skip: skip.into_iter().map(QubitId).collect(),
                ..Default::default()
            };
            let recorded = Arc::clone(&fixture.after);
            let mut noise = ComposableNoiseModel::new().add_channel(fixture);
            for q in leaked {
                noise.context_mut().mark_leaked(QubitId(q));
            }
            let mut runner = CircuitRunner::new().with_noise(noise);
            let handlers = Arc::new(Mutex::new(vec![]));
            let seen = Arc::clone(&handlers);
            runner.on_after_gate(move |ctx| {
                seen.lock()
                    .unwrap()
                    .push((ctx.qubits.to_vec(), ctx.leaked_suppressed.to_vec()));
                NoiseResponse::None
            });
            execute(
                &mut runner,
                &mut SparseStab::with_seed(6, 42),
                adapted,
                GateType::CX,
                &(0..6).map(QubitId).collect::<Vec<_>>(),
                &[],
            );
            assert_eq!(
                *recorded.lock().unwrap(),
                expected.clone().into_iter().collect::<Vec<_>>()
            );
            let expected_handlers: Vec<_> = expected
                .filter(|_| !adapted)
                .into_iter()
                .map(|s| (s.executed, s.suppressed))
                .collect();
            assert_eq!(*handlers.lock().unwrap(), expected_handlers);
        }
    }
}

#[test]
fn same_before_emit_seep_drops_group() {
    for adapted in [false, true] {
        let fixture = Fixture {
            skip: vec![QubitId(0)],
            seep_before: true,
            ..Default::default()
        };
        let recorded = Arc::clone(&fixture.after);
        let mut noise = ComposableNoiseModel::new().add_channel(fixture);
        noise.context_mut().mark_leaked(QubitId(0));
        let mut runner = CircuitRunner::new().with_noise(noise);
        execute(
            &mut runner,
            &mut SparseStab::with_seed(2, 42),
            adapted,
            GateType::CX,
            &[QubitId(0), QubitId(1)],
            &[],
        );
        assert!(recorded.lock().unwrap().is_empty());
        assert!(
            !runner
                .noise
                .as_ref()
                .unwrap()
                .context()
                .is_leaked(QubitId(0))
        );
    }
}

#[derive(Clone)]
struct MarkObserver(Arc<Mutex<Vec<bool>>>);
impl ContextObserver for MarkObserver {
    fn on_leaked(&self, _q: QubitId, _ctx: &NoiseContext, _rng: &mut PecosRng) -> NoiseResponse {
        self.0.lock().unwrap().push(true);
        NoiseResponse::inject_gate(GateCommand::x(QubitId(2)))
            .combine(NoiseResponse::MarkLeaked(smallvec::smallvec![QubitId(3)]))
    }
    fn on_unleaked(&self, _q: QubitId, _ctx: &NoiseContext, _rng: &mut PecosRng) -> NoiseResponse {
        self.0.lock().unwrap().push(false);
        NoiseResponse::None
    }
    fn clone_box(&self) -> Box<dyn ContextObserver> {
        Box::new(self.clone())
    }
}
#[derive(Clone, Copy, Debug)]
struct MarkSignal;
pecos_core::impl_signal!(MarkSignal);

#[test]
fn handler_marks_all_seven_sites() {
    for site in 0..7 {
        for leaked in [true, false] {
            for with_noise in [true, false] {
                let observed = Arc::new(Mutex::new(vec![]));
                let mut noise =
                    ComposableNoiseModel::new().add_observer(MarkObserver(Arc::clone(&observed)));
                if !leaked {
                    noise.context_mut().mark_leaked(QubitId(0));
                }
                let mut runner = CircuitRunner::new();
                if with_noise {
                    runner = runner.with_noise(noise);
                }
                let handler = move |_: &DispatchContext<'_>| {
                    NoiseResponse::Multiple(vec![NoiseResponse::Multiple(vec![if leaked {
                        NoiseResponse::MarkLeaked(smallvec::smallvec![QubitId(0)])
                    } else {
                        NoiseResponse::MarkUnleaked(smallvec::smallvec![QubitId(0)])
                    }])])
                };
                match site {
                    0 => {
                        runner.on_before_gate(handler);
                    }
                    1 => {
                        runner.on_after_gate(handler);
                    }
                    2 => {
                        runner.on_before_measurement(handler);
                    }
                    3 => {
                        runner.on_after_measurement(handler);
                    }
                    4 => {
                        runner.on_after_preparation(handler);
                    }
                    5 => {
                        runner.on_idle(handler);
                    }
                    _ => {
                        runner.on_signal_with_response(move |_: &MarkSignal, ctx| handler(ctx));
                    }
                }
                let mut sim = SparseStab::with_seed(4, 42);
                let result = match site {
                    0 | 1 => runner.apply_gate(&mut sim, GateType::X, &[QubitId(1)], &[]),
                    2 | 3 => runner.apply_gate(&mut sim, GateType::MZ, &[QubitId(1)], &[]),
                    4 => runner.apply_gate(&mut sim, GateType::PZ, &[QubitId(1)], &[]),
                    5 => runner.execute_queue_command(
                        &mut sim,
                        &GateCommand::idle(QubitId(1), TimeUnits::new(1)),
                    ),
                    _ => runner.execute_queue_with_signals(
                        &mut sim,
                        &CommandBuilder::new().signal(MarkSignal).build(),
                    ),
                };
                if with_noise {
                    result.unwrap();
                    let ctx = runner.noise.as_ref().unwrap().context();
                    assert_eq!(ctx.is_leaked(QubitId(0)), leaked, "site={site}");
                    assert!(
                        !ctx.is_leaked(QubitId(3)),
                        "observer marks must not recurse"
                    );
                    assert_eq!(*observed.lock().unwrap(), vec![leaked], "site={site}");
                    assert_eq!(
                        sim.mz(&[QubitId(2)])[0].outcome,
                        leaked,
                        "observer gates are combined"
                    );
                } else {
                    let error = result.unwrap_err();
                    assert!(matches!(
                        error,
                        ExecutionError::HandlerLeakageWithoutNoise { .. }
                    ));
                    let text = error.to_string();
                    assert!(text.contains(if leaked { "MarkLeaked" } else { "MarkUnleaked" }));
                    assert!(text.contains("QubitId(0)") && text.contains("add a noise model"));
                }
            }
        }
    }
}

#[test]
fn before_handler_marks_reach_noise_and_partition() {
    for skip in [false, true] {
        let fixture = Fixture::default();
        let recorded = Arc::clone(&fixture.after);
        let observed = Arc::new(Mutex::new(vec![]));
        let noise = ComposableNoiseModel::new()
            .add_channel(fixture)
            .add_channel(LeakageChannel::new())
            .add_observer(MarkObserver(Arc::clone(&observed)));
        let mut runner = CircuitRunner::new().with_noise(noise);
        runner.on_before_gate(move |_| {
            let mark = NoiseResponse::MarkLeaked(smallvec::smallvec![QubitId(0)]);
            if skip {
                mark.combine(NoiseResponse::SkipGate(smallvec::smallvec![QubitId(0)]))
            } else {
                mark
            }
        });
        runner
            .apply_gate(
                &mut SparseStab::with_seed(4, 42),
                GateType::CX,
                &[QubitId(0), QubitId(1)],
                &[],
            )
            .unwrap();
        assert_eq!(
            *recorded.lock().unwrap(),
            vec![Shape {
                executed: vec![],
                suppressed: vec![QubitId(0), QubitId(1)],
                size: 2
            }]
        );
        assert_eq!(*observed.lock().unwrap(), vec![true]);
    }
}

#[test]
fn composite_filters_use_registered_group_size() {
    use crate::noise::composite::BatchCompositeChannel;
    use crate::noise::composite::prelude::*;
    for batch in [false, true] {
        for dropped in [false, true] {
            for (gate, size, filter, matches) in [
                (GateType::X, 1, CompositeEventFilter::SingleQubitGate, true),
                (GateType::CX, 2, CompositeEventFilter::TwoQubitGate, true),
                (GateType::H, 1, CompositeEventFilter::TwoQubitGate, false),
                (GateType::I, 2, CompositeEventFilter::TwoQubitGate, true),
            ] {
                let mut definitions = GateDefinitions::new();
                let gate_id = if gate == GateType::I {
                    definitions.register(GateSpec::new("pair").with_quantum_arity(2))
                } else {
                    gate.to_gate_id()
                };
                let overrides =
                    GateOverrides::new().register(gate_id, |_: &mut SparseStab, _, _| true);
                let channel: Box<dyn NoiseChannel> = if batch {
                    Box::new(
                        BatchCompositeChannel::new("filter", 1.0, inject_x())
                            .unwrap()
                            .with_filter(filter),
                    )
                } else {
                    Box::new(CompositeChannel::new("filter", inject_x()).with_filter(filter))
                };
                let noise = ComposableNoiseModel::new()
                    .add_boxed_channel(channel)
                    .add_channel(Fixture {
                        skip: if dropped { vec![QubitId(0)] } else { vec![] },
                        ..Default::default()
                    });
                let mut runner = CircuitRunner::with_definitions(definitions)
                    .with_overrides(overrides)
                    .with_noise(noise);
                let qubits: Vec<_> = (0..size * 2).map(QubitId).collect();
                let mut sim = SparseStab::with_seed(size * 2, 42);
                runner
                    .execute_gate(&mut sim, gate_id, &qubits, &[], 0)
                    .unwrap();
                for (i, m) in sim.mz(&qubits).iter().enumerate() {
                    assert_eq!(
                        m.outcome,
                        matches && (!dropped || i >= size),
                        "batch={batch} dropped={dropped} gate={gate:?} i={i}"
                    );
                }
            }
        }
    }
}

#[test]
fn noise_marks_notify_once_with_handlers() {
    let observed = Arc::new(Mutex::new(vec![]));
    let noise = ComposableNoiseModel::new()
        .add_channel(
            SingleQubitChannel::depolarizing(1.0)
                .unwrap()
                .with_emission(1.0, 1.0)
                .unwrap(),
        )
        .add_observer(MarkObserver(Arc::clone(&observed)));
    let mut runner = CircuitRunner::new().with_noise(noise);
    runner.on_after_gate(|_| NoiseResponse::None);
    runner
        .apply_gate(
            &mut SparseStab::with_seed(4, 42),
            GateType::X,
            &[QubitId(0)],
            &[],
        )
        .unwrap();
    assert_eq!(*observed.lock().unwrap(), vec![true]);
    assert!(
        !runner
            .noise
            .as_ref()
            .unwrap()
            .context()
            .is_leaked(QubitId(3))
    );
}

#[test]
fn suppressed_adapter_seepage() {
    use crate::noise::composite::prelude::*;
    for adapted in [false, true] {
        let channel = CompositeChannel::new(
            "wrapped seepage",
            channel_action(
                TwoQubitChannel::depolarizing(1.0)
                    .unwrap()
                    .with_leakage(1.0)
                    .unwrap()
                    .with_seepage(1.0)
                    .unwrap(),
            ),
        )
        .with_filter(CompositeEventFilter::TwoQubitGate);
        let mut noise = ComposableNoiseModel::new()
            .add_channel(Fixture {
                skip: vec![QubitId(0)],
                ..Default::default()
            })
            .add_channel(channel);
        noise.context_mut().mark_leaked(QubitId(0));
        let mut runner = CircuitRunner::new().with_noise(noise);
        let mut sim = SparseStab::with_seed(2, 7);
        execute(
            &mut runner,
            &mut sim,
            adapted,
            GateType::CX,
            &[QubitId(0), QubitId(1)],
            &[],
        );
        assert!(
            !runner
                .noise
                .as_ref()
                .unwrap()
                .context()
                .is_leaked(QubitId(0))
        );
        assert!(!sim.mz(&[QubitId(1)])[0].outcome);
    }
}

#[test]
fn channel_adapter_marks_every_named_qubit() {
    use crate::noise::composite::prelude::*;
    #[derive(Clone)]
    struct MarkBoth;
    impl NoiseChannel for MarkBoth {
        fn responds_to(&self, _: &NoiseEvent<'_>) -> bool {
            true
        }
        fn apply(
            &self,
            event: &NoiseEvent<'_>,
            _: &mut NoiseContext,
            _: &mut PecosRng,
        ) -> NoiseResponse {
            NoiseResponse::MarkLeaked(event.qubits().iter().copied().collect())
        }
        fn name(&self) -> &'static str {
            "mark both"
        }
        fn clone_box(&self) -> Box<dyn NoiseChannel> {
            Box::new(self.clone())
        }
    }
    let noise = ComposableNoiseModel::new().add_channel(
        CompositeChannel::new("marks", channel_action(MarkBoth))
            .with_filter(CompositeEventFilter::TwoQubitGate),
    );
    let mut runner = CircuitRunner::new().with_noise(noise);
    let mut sim = SparseStab::with_seed(2, 42);
    runner
        .apply_gate(&mut sim, GateType::CX, &[QubitId(0), QubitId(1)], &[])
        .unwrap();
    let ctx = runner.noise.as_ref().unwrap().context();
    assert!(ctx.is_leaked(QubitId(0)));
    assert!(ctx.is_leaked(QubitId(1)));
}
