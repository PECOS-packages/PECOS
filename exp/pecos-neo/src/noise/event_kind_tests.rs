// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

use super::*;
use crate::sampling::importance::{ImportanceConfig, ImportanceSamplingChannel};

pub(super) fn representative(kind: NoiseEventKind) -> NoiseEvent<'static> {
    match kind {
        NoiseEventKind::BeforeGate => NoiseEvent::before_gate(GateType::H, &[QubitId(0)], &[]),
        NoiseEventKind::AfterGate => NoiseEvent::after_gate(GateType::H, &[QubitId(0)], &[]),
        NoiseEventKind::BeforeMeasurement => NoiseEvent::BeforeMeasurement {
            qubits: &[QubitId(0)],
        },
        NoiseEventKind::AfterMeasurement => NoiseEvent::AfterMeasurement {
            qubits: &[QubitId(0)],
            outcomes: &[true],
        },
        NoiseEventKind::AfterPreparation => NoiseEvent::AfterPreparation {
            qubits: &[QubitId(0)],
        },
        NoiseEventKind::IdleTime => NoiseEvent::IdleTime {
            qubits: &[QubitId(0)],
            duration: TimeUnits::new(2),
        },
        NoiseEventKind::AfterReset => NoiseEvent::AfterReset {
            qubits: &[QubitId(0)],
        },
        NoiseEventKind::BeforeCircuit => NoiseEvent::BeforeCircuit { num_qubits: 4 },
        NoiseEventKind::AfterCircuit => NoiseEvent::AfterCircuit { num_qubits: 4 },
        NoiseEventKind::BetweenLayers => NoiseEvent::BetweenLayers {
            qubits: &[QubitId(0)],
            layer_index: 1,
        },
        NoiseEventKind::Signal => NoiseEvent::Signal {
            type_id: TypeId::of::<u64>(),
            data: &7_u64,
        },
    }
}

pub(super) fn witnesses() -> Vec<NoiseEvent<'static>> {
    let mut events: Vec<_> = NoiseEventKind::ALL
        .into_iter()
        .map(representative)
        .collect();
    events.push(NoiseEvent::before_gate(
        GateType::CX,
        &[QubitId(0), QubitId(1)],
        &[],
    ));
    events.push(NoiseEvent::after_gate(
        GateType::CX,
        &[QubitId(0), QubitId(1)],
        &[],
    ));
    events
}

#[test]
fn kinds_round_trip_and_bitset_operations() {
    let mut accumulated = EventKinds::NONE;
    for (index, kind) in NoiseEventKind::ALL.into_iter().enumerate() {
        assert_eq!(kind.index(), index);
        assert_eq!(representative(kind).kind(), kind);
        assert!(EventKinds::ALL.contains(kind));
        assert!(!EventKinds::NONE.contains(kind));
        for other in NoiseEventKind::ALL {
            assert_eq!(EventKinds::of(kind).contains(other), kind == other);
        }
        accumulated = accumulated.with(kind);
        assert_eq!(accumulated.union(EventKinds::of(kind)), accumulated);
    }
    assert_eq!(accumulated, EventKinds::ALL);
    assert_eq!(NoiseEventKind::ALL.len(), EventKinds::COUNT);
}

pub(super) fn assert_channel_kinds(channel: &dyn NoiseChannel, expected: EventKinds) {
    let events = witnesses();
    for kind in NoiseEventKind::ALL {
        if expected.contains(kind) {
            let witness = events
                .iter()
                .find(|event| event.kind() == kind && channel.responds_to(event))
                .unwrap_or_else(|| {
                    panic!("{} has no positive witness for {kind:?}", channel.name())
                });
            assert!(channel.responds_to(witness));
            assert!(
                channel.event_kinds().contains(kind),
                "{} missing {kind:?}",
                channel.name()
            );
            let mut ctx = NoiseContext::new();
            for q in 0..4 {
                ctx.mark_prepared(QubitId(q));
            }
            assert!(
                channel
                    .try_apply(witness, &mut ctx, &mut PecosRng::seed_from_u64(42))
                    .is_some(),
                "{} rejected its positive {kind:?} witness in try_apply",
                channel.name()
            );
        } else {
            assert!(
                !channel.event_kinds().contains(kind),
                "{} unexpectedly declares {kind:?}",
                channel.name()
            );
            for event in events.iter().filter(|event| event.kind() == kind) {
                assert!(
                    !channel.responds_to(event),
                    "{} responds to excluded {kind:?}",
                    channel.name()
                );
                assert!(
                    channel
                        .try_apply(
                            event,
                            &mut NoiseContext::new(),
                            &mut PecosRng::seed_from_u64(42)
                        )
                        .is_none(),
                    "{} try_apply responds to excluded {kind:?}",
                    channel.name()
                );
            }
        }
    }
}

#[test]
fn builtin_channel_kind_declarations_have_positive_witnesses() {
    use NoiseEventKind::{
        AfterGate, AfterMeasurement, AfterPreparation, BeforeGate, BeforeMeasurement, IdleTime,
    };
    let gates = EventKinds::of(BeforeGate).with(AfterGate);
    let after_gate = EventKinds::of(AfterGate);
    let channels: Vec<(Box<dyn NoiseChannel>, EventKinds)> = vec![
        (Box::new(SingleQubitChannel::depolarizing(0.2)), gates),
        (Box::new(TwoQubitChannel::depolarizing(0.2)), gates),
        (
            Box::new(LeakageChannel::new()),
            gates.with(BeforeMeasurement),
        ),
        (
            Box::new(MeasurementChannel::symmetric(0.2)),
            EventKinds::of(AfterMeasurement),
        ),
        (
            Box::new(MeasurementStateFlipChannel::new(0.2)),
            EventKinds::of(BeforeMeasurement),
        ),
        (
            Box::new(PreparationChannel::new(0.2)),
            EventKinds::of(AfterPreparation),
        ),
        (
            Box::new(IdleChannel::linear(0.2).with_idle_after_2q(1.0)),
            EventKinds::of(IdleTime).with(AfterGate),
        ),
        (
            Box::new(CrosstalkChannel::global_only(0.2)),
            after_gate.with(AfterPreparation).with(AfterMeasurement),
        ),
        (Box::new(CorrelatedNoiseChannel::new(0.2, 0.5)), after_gate),
        (
            Box::new(CategoryBasedChannel::new().with_default(0.2)),
            after_gate,
        ),
        (
            Box::new(GateDependentChannel::new().with_gate_error(GateType::H, 0.2)),
            after_gate,
        ),
        (
            Box::new(GateIdDependentChannel::new().with_gate_type_error(GateType::H, 0.2)),
            after_gate,
        ),
        (
            Box::new(
                PerGatePauliChannel::new()
                    .with_base(0.2, 0.2)
                    .with_meas_init(0.2, 0.2),
            ),
            after_gate.with(BeforeMeasurement).with(AfterPreparation),
        ),
    ];
    for (channel, kinds) in channels {
        assert_channel_kinds(&*channel, kinds);
    }

    // The wrapper conservatively declares ALL: its default try_apply consults
    // responds_to, not the inner channel's potentially optimized try_apply.
    let inner = SingleQubitChannel::depolarizing(0.2);
    let wrapper =
        ImportanceSamplingChannel::new(inner.clone(), ImportanceConfig::with_boost(0.02, 10.0));
    assert_eq!(wrapper.event_kinds(), EventKinds::ALL);
    for event in witnesses() {
        assert_eq!(wrapper.responds_to(&event), inner.responds_to(&event));
        assert_eq!(
            wrapper
                .try_apply(
                    &event,
                    &mut NoiseContext::new(),
                    &mut PecosRng::seed_from_u64(42),
                )
                .is_some(),
            inner.responds_to(&event)
        );
    }
}

#[test]
fn per_gate_optional_kind_declarations_follow_configuration() {
    let base = PerGatePauliChannel::new().with_base(0.2, 0.2);
    let after_gate = EventKinds::of(NoiseEventKind::AfterGate);
    assert_channel_kinds(&base, after_gate);
    assert_channel_kinds(
        &base.clone().with_meas_rate_for_qubit(QubitId(0), 0.2),
        after_gate.with(NoiseEventKind::BeforeMeasurement),
    );
    assert_channel_kinds(
        &base.with_init_rate_for_qubit(QubitId(0), 0.2),
        after_gate.with(NoiseEventKind::AfterPreparation),
    );
}

#[test]
fn core_handler_kind_declarations_have_positive_witnesses() {
    let mut config = NoiseModelConfig::new();
    plugins::CorePlugin.build(&mut config);
    assert_eq!(config.event_handlers.len(), 2);
    for (handler, expected) in config.event_handlers.iter().zip([
        NoiseEventKind::AfterPreparation,
        NoiseEventKind::AfterMeasurement,
    ]) {
        for kind in NoiseEventKind::ALL {
            let event = representative(kind);
            assert_eq!(
                handler.handles(&event),
                kind == expected,
                "{} {kind:?}",
                handler.name()
            );
            assert_eq!(
                handler.event_kinds().contains(kind),
                kind == expected,
                "{} {kind:?}",
                handler.name()
            );
        }
    }
}
