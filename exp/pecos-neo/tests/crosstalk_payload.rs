use pecos_core::QubitId;
use pecos_neo::command::{CommandBuilder, GateCommand, GateType};
use pecos_neo::extensible::{
    CircuitValidator, CliffordValidator, GateCategory, GateDefinitions, GateRegistry, OpBuilder,
    ResultId,
};
use pecos_neo::noise::composite::prelude::*;
use pecos_neo::noise::{
    ComposableNoiseModel, CrosstalkChannel, CrosstalkTransitions, GeneralNoiseModelBuilder,
    NoiseChannel, NoiseContext, NoiseEvent, NoiseEventKind, NoiseResponse,
};
use pecos_neo::runner::{CircuitRunner, ExecutionError};
use pecos_neo::sampling::{ImportanceSamplingError, ImportanceSamplingRunner, PathExplorer};
use pecos_random::PecosRng;
use pecos_simulators::{CliffordGateable, SparseStab};
use smallvec::smallvec;
use std::sync::{Arc, Mutex};

fn channels(local: bool, rate: f64) -> Vec<Box<dyn NoiseChannel>> {
    vec![
        Box::new(
            CrosstalkChannel::measurement()
                .with_global_rate(if local { 0.0 } else { rate })
                .unwrap()
                .with_local_rate(if local { rate } else { 0.0 })
                .unwrap(),
        ),
        Box::new(
            CompositeCrosstalkChannel::measurement(
                "test",
                if local {
                    PayloadKind::Local
                } else {
                    PayloadKind::Global
                },
                rate,
                nothing(),
            )
            .unwrap(),
        ),
    ]
}

#[test]
fn payload_victims_noiseless_empty_and_no_fired_response() {
    for local in [false, true] {
        let gate_type = if local {
            GateType::MeasCrosstalkLocalPayload
        } else {
            GateType::MeasCrosstalkGlobalPayload
        };
        let operands = if local {
            vec![QubitId(1), QubitId(2), QubitId(3)]
        } else {
            vec![QubitId(0)]
        };
        let event = NoiseEvent::CrosstalkPayload {
            gate_type,
            qubits: &operands,
        };
        for channel in channels(local, 1.0) {
            let mut ctx = NoiseContext::new();
            for q in 0..3 {
                ctx.mark_prepared(QubitId(q));
            }
            ctx.mark_measured(QubitId(1));
            ctx.mark_leaked(QubitId(2));
            let mut rng = PecosRng::seed_from_u64(42);
            let response = channel.apply(&event, &mut ctx, &mut rng);
            assert!(
                matches!(response, NoiseResponse::MeasureHidden { gate_type: actual, ref qubits }
                if actual == gate_type && qubits.as_slice() == [QubitId(1)]),
                "{response:?}"
            );
            ctx.add_noiseless_gate(gate_type);
            assert!(channel.apply(&event, &mut ctx, &mut rng).is_none());
            ctx.clear_noiseless_gates();
            ctx.mark_leaked(QubitId(1));
            assert!(channel.apply(&event, &mut ctx, &mut rng).is_none());
            ctx.reset();
            assert!(channel.apply(&event, &mut ctx, &mut rng).is_none());
        }
        for channel in channels(local, 0.0) {
            let mut ctx = NoiseContext::new();
            ctx.mark_prepared(QubitId(1));
            assert!(
                channel
                    .apply(&event, &mut ctx, &mut PecosRng::seed_from_u64(42))
                    .is_none()
            );
        }
    }
    for channel in channels(false, 1.0) {
        let mut ctx = NoiseContext::new();
        ctx.mark_prepared(QubitId(0));
        ctx.mark_prepared(QubitId(1));
        let response = channel.apply(
            &NoiseEvent::CrosstalkPayload {
                gate_type: GateType::MeasCrosstalkGlobalPayload,
                qubits: &[],
            },
            &mut ctx,
            &mut PecosRng::seed_from_u64(42),
        );
        assert!(
            matches!(response, NoiseResponse::MeasureHidden { ref qubits, .. }
            if qubits.as_slice() == [QubitId(0), QubitId(1)])
        );
    }
}

#[derive(Clone)]
struct Observe(Arc<Mutex<Vec<NoiseEventKind>>>);
impl NoiseChannel for Observe {
    fn responds_to(&self, _: &NoiseEvent<'_>) -> bool {
        true
    }
    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        _: &mut PecosRng,
    ) -> NoiseResponse {
        self.0.lock().unwrap().push(event.kind());
        if let NoiseEvent::AfterHiddenMeasurement { outcomes, .. } = event {
            assert_eq!(*outcomes, [false]);
            assert!(ctx.exists(QubitId(1)));
            assert!(ctx.is_active(QubitId(1)));
            assert!(!ctx.is_leaked(QubitId(1)));
        }
        NoiseResponse::None
    }
    fn name(&self) -> &'static str {
        "Observe"
    }
    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

#[test]
fn hidden_measurement_is_private_and_adapted_payload_matches_queue() {
    for adapted in [false, true] {
        let log = Arc::new(Mutex::new(Vec::new()));
        let noise = GeneralNoiseModelBuilder::new()
            .with_p_meas_crosstalk(1.0, 0.0)
            .with_p_meas_crosstalk_transitions(
                CrosstalkTransitions::custom(0.0, 1.0, 0.0, 1.0, 0.0, 0.0).unwrap(),
            )
            .unwrap()
            .with_p_meas(1.0, 1.0)
            .build()
            .unwrap()
            .add_channel(Observe(log.clone()));
        let mut runner = CircuitRunner::new().with_noise(noise).with_seed(42);
        runner
            .on_before_gate(|_| panic!("payload reached a user gate handler"))
            .on_after_gate(|_| panic!("payload reached a user gate handler"));
        let mut state = SparseStab::with_seed(2, 42);
        let outcomes = if adapted {
            runner
                .apply_gate(&mut state, GateType::PZ, &[QubitId(0), QubitId(1)], &[])
                .unwrap();
            let circuit = OpBuilder::new()
                .gate(
                    GateType::MeasCrosstalkGlobalPayload.to_gate_id(),
                    &[QubitId(0)],
                    &[],
                )
                .mz(QubitId(1), ResultId(0))
                .build();
            runner.apply_adapted_circuit(&mut state, &circuit).unwrap()
        } else {
            let circuit = CommandBuilder::new()
                .pz(&[0, 1])
                .meas_crosstalk_global_payload(&[0])
                .mz(&[1])
                .build();
            runner.apply_circuit(&mut state, &circuit).unwrap()
        };
        assert_eq!(outcomes.get_all(QubitId(1)).len(), 1);
        assert_eq!(outcomes.get(QubitId(1)).unwrap().outcome, adapted);
        assert!(state.mz(&[QubitId(1)])[0].outcome);
        let events = log.lock().unwrap();
        assert_eq!(
            events
                .iter()
                .filter(|&&kind| kind == NoiseEventKind::AfterMeasurement)
                .count(),
            usize::from(!adapted)
        );
        assert_eq!(
            events
                .iter()
                .filter(|&&kind| kind == NoiseEventKind::AfterHiddenMeasurement)
                .count(),
            1
        );
        assert!(!events.contains(&NoiseEventKind::BeforeGate));
        assert!(!events.contains(&NoiseEventKind::AfterGate));
    }
}

#[test]
fn payloads_round_trip_and_remain_metadata() {
    use pecos_engines::ByteMessage;
    let definitions = GateDefinitions::new();
    let registry = GateRegistry::new();
    for gate_type in [
        GateType::MeasCrosstalkGlobalPayload,
        GateType::MeasCrosstalkLocalPayload,
    ] {
        assert!(!gate_type.is_measurement());
        assert!(!gate_type.is_unitary_gate());
        let id = gate_type.to_gate_id();
        assert_eq!(id.try_to_gate_type(), Some(gate_type));
        assert!(CliffordValidator::new().is_gate_allowed(id, &[], &registry));
        assert_eq!(
            definitions.spec(id).unwrap().category,
            GateCategory::Metadata
        );
        assert_eq!(registry.get(id).unwrap().category, GateCategory::Metadata);
        for qubits in [vec![], vec![QubitId(0), QubitId(2)]] {
            let command = GateCommand::new(gate_type, qubits.clone());
            let core = command.try_to_core_gate().unwrap();
            let restored = GateCommand::try_from(core).unwrap();
            assert_eq!(restored.gate_type, gate_type);
            assert_eq!(restored.qubits.as_slice(), qubits);
            let mut message = ByteMessage::quantum_operations_builder();
            let ids: Vec<_> = qubits.iter().map(|q| q.0).collect();
            if gate_type == GateType::MeasCrosstalkGlobalPayload {
                message.meas_crosstalk_global_payload(&ids);
            } else {
                message.meas_crosstalk_local_payload(&ids);
            }
            let queue =
                pecos_neo::adapter::byte_message_to_command_queue(&message.build()).unwrap();
            assert_eq!(queue.len(), 1);
            let converted = queue.iter().next().unwrap();
            assert_eq!(converted.gate_type, gate_type);
            assert_eq!(converted.qubits.as_slice(), qubits);
        }
        let duplicate = GateCommand::new(gate_type, vec![QubitId(0), QubitId(0)]);
        assert!(duplicate.validate().is_err());
        let circuit = OpBuilder::new()
            .gate(id, &[QubitId(0), QubitId(0)], &[])
            .build();
        assert!(
            CircuitRunner::new()
                .apply_adapted_circuit(&mut SparseStab::with_seed(1, 42), &circuit)
                .is_err()
        );
    }
}

#[derive(Clone)]
struct RequestHidden;
impl NoiseChannel for RequestHidden {
    fn responds_to(&self, _: &NoiseEvent<'_>) -> bool {
        true
    }
    fn apply(&self, _: &NoiseEvent<'_>, _: &mut NoiseContext, _: &mut PecosRng) -> NoiseResponse {
        NoiseResponse::Multiple(vec![NoiseResponse::None, hidden()])
    }
    fn name(&self) -> &'static str {
        "RequestHidden"
    }
    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}
fn hidden() -> NoiseResponse {
    NoiseResponse::MeasureHidden {
        gate_type: GateType::MeasCrosstalkGlobalPayload,
        qubits: smallvec![QubitId(0)],
    }
}

#[test]
fn importance_rejects_payload_before_emission_and_hidden_responses() {
    for local in [false, true] {
        let queue = if local {
            CommandBuilder::new().meas_crosstalk_local_payload(&[0])
        } else {
            CommandBuilder::new().meas_crosstalk_global_payload(&[] as &[usize])
        }
        .build();
        let log = Arc::new(Mutex::new(Vec::new()));
        let mut runner = ImportanceSamplingRunner::new(SparseStab::with_seed(1, 42))
            .with_noise(ComposableNoiseModel::new().add_channel(Observe(log.clone())));
        let error = runner.try_run_shot(&queue).unwrap_err();
        assert!(matches!(
            error,
            ImportanceSamplingError::UnsupportedGate { .. }
        ));
        assert!(error.to_string().contains("measurement crosstalk"));
        assert!(log.lock().unwrap().is_empty());
        assert!(matches!(
            runner.try_run_shot_biased(&queue),
            Err(ImportanceSamplingError::UnsupportedGate { .. })
        ));
        assert!(log.lock().unwrap().is_empty());
        let recorded = PathExplorer::new(SparseStab::with_seed(1, 42)).run_and_record(&queue);
        assert_eq!(recorded.outcomes.get_all(QubitId(0)).len(), 0);
    }
    let mut runner = ImportanceSamplingRunner::new(SparseStab::with_seed(1, 42))
        .with_noise(ComposableNoiseModel::new().add_channel(RequestHidden));
    let error = runner
        .try_run_shot(&CommandBuilder::new().h(&[0]).build())
        .unwrap_err();
    assert!(matches!(
        error,
        ImportanceSamplingError::HiddenMeasurementUnsupported
    ));
    assert!(error.to_string().contains("CircuitRunner"));
}

#[test]
fn handler_hidden_measurement_requires_noise_model() {
    let mut runner = CircuitRunner::new();
    runner.on_before_gate(|_| NoiseResponse::Multiple(vec![hidden()]));
    let error = runner
        .apply_circuit(
            &mut SparseStab::with_seed(1, 42),
            &CommandBuilder::new().h(&[0]).build(),
        )
        .unwrap_err();
    assert!(matches!(
        error,
        ExecutionError::HandlerHiddenMeasurementWithoutNoise
    ));
}

#[test]
fn hidden_measurement_is_applied_at_its_multiple_position() {
    let noise = GeneralNoiseModelBuilder::new()
        .with_p_meas_crosstalk(1.0, 0.0)
        .with_p_meas_crosstalk_transitions(
            CrosstalkTransitions::custom(1.0, 0.0, 0.0, 0.0, 1.0, 0.0).unwrap(),
        )
        .unwrap()
        .build()
        .unwrap();
    let mut runner = CircuitRunner::new().with_noise(noise).with_seed(42);
    runner.on_after_preparation(|_| {
        NoiseResponse::Multiple(vec![
            NoiseResponse::inject_gate(GateCommand::x(QubitId(0))),
            hidden(),
            NoiseResponse::inject_gate(GateCommand::x(QubitId(0))),
        ])
    });
    let results = runner
        .apply_circuit(
            &mut SparseStab::with_seed(1, 42),
            &CommandBuilder::new().pz(&[0]).mz(&[0]).build(),
        )
        .unwrap();
    assert!(results.get(QubitId(0)).unwrap().outcome);
    assert_eq!(results.get_all(QubitId(0)).len(), 1);
}

#[test]
fn hidden_outcomes_are_scoped_to_the_owning_channel_and_cleared() {
    let mut ctx = NoiseContext::new();
    ctx.mark_prepared(QubitId(0));
    ctx.mark_prepared(QubitId(1));
    let table = CrosstalkTransitions::custom(0.0, 1.0, 0.0, 0.0, 0.0, 1.0).unwrap();
    let channel = CompositeCrosstalkChannel::measurement(
        "local",
        PayloadKind::Local,
        1.0,
        crosstalk_transitions(table).unwrap(),
    )
    .unwrap();
    let mut rng = PecosRng::seed_from_u64(42);
    let wrong_kind = NoiseEvent::AfterHiddenMeasurement {
        gate_type: GateType::MeasCrosstalkGlobalPayload,
        qubits: &[QubitId(0), QubitId(1)],
        outcomes: &[false, true],
    };
    assert!(channel.apply(&wrong_kind, &mut ctx, &mut rng).is_none());
    assert_eq!(ctx.leaked_count(), 0);
    let event = NoiseEvent::AfterHiddenMeasurement {
        gate_type: GateType::MeasCrosstalkLocalPayload,
        qubits: &[QubitId(0), QubitId(1)],
        outcomes: &[false, true],
    };
    let response = channel.apply(&event, &mut ctx, &mut rng);
    let NoiseResponse::Multiple(parts) = response else {
        panic!("{response:?}");
    };
    assert!(
        matches!(&parts[0], NoiseResponse::InjectGates(gates) if gates[0] == GateCommand::x(QubitId(0)))
    );
    assert!(
        matches!(&parts[1], NoiseResponse::MarkLeaked(qubits) if qubits.as_slice() == [QubitId(1)])
    );
    assert!(ctx.is_leaked(QubitId(1)));
    assert_eq!(ctx.current_outcome(), None);
}

#[test]
fn wrapped_channel_preserves_hidden_measurement_response() {
    let channel = CompositeChannelBuilder::single_qubit("wrapped", channel_action(RequestHidden));
    let response = channel.apply(
        &NoiseEvent::after_gate(GateType::H, &[QubitId(0)], &[]),
        &mut NoiseContext::new(),
        &mut PecosRng::seed_from_u64(42),
    );
    assert!(
        matches!(response, NoiseResponse::MeasureHidden { gate_type: GateType::MeasCrosstalkGlobalPayload, ref qubits }
        if qubits.as_slice() == [QubitId(0)])
    );
}
