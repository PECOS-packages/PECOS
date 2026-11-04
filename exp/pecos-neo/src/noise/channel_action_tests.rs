//! Regression tests for channel adapters, operand groups, response targets and event state.
use super::*;
use crate::noise::composite::batch_composite::{BatchState, FastBatchProcessor};
use crate::noise::composite::prelude::*;
use crate::noise::composite::{BatchCompositeChannel, CompositeCrosstalkChannel, Condition};
use std::sync::{Arc, Mutex};

const PAIR: [QubitId; 2] = [QubitId(0), QubitId(1)];

type View = (usize, Vec<QubitId>, Vec<QubitId>);
#[derive(Clone, Default)]
struct Probe {
    views: Arc<Mutex<Vec<View>>>,
    response: NoiseResponse,
}
impl NoiseChannel for Probe {
    fn responds_to(&self, _: &NoiseEvent<'_>) -> bool {
        false // The adapter deliberately bypasses this filter.
    }
    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        ctx: &mut NoiseContext,
        _: &mut PecosRng,
    ) -> NoiseResponse {
        assert!(matches!(event, NoiseEvent::AfterGate { .. }));
        self.views.lock().unwrap().push((
            ctx.current_qubit_index(),
            event.qubits().to_vec(),
            event.leaked_suppressed().to_vec(),
        ));
        self.response.clone()
    }
    fn name(&self) -> &'static str {
        "group probe"
    }
    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

fn event(qubits: &[QubitId], suppressed: bool) -> NoiseEvent<'_> {
    NoiseEvent::AfterGate {
        gate_type: if qubits.len() == 1 {
            GateType::H
        } else {
            GateType::CX
        },
        gate_id: None,
        qubits: if suppressed { &[] } else { qubits },
        leaked_suppressed: if suppressed { qubits } else { &[] },
        angles: &[],
        group_size: gate_group_size(2, qubits.len()),
    }
}

fn assert_rate(channel: &dyn NoiseChannel, qubits: &[QubitId], expected: f64) {
    const EVENTS: u32 = 100_000;
    let mut rng = PecosRng::seed_from_u64(1568);
    let mut ctx = NoiseContext::new();
    let event = event(qubits, false);
    let count = (0..EVENTS)
        .filter(|_| !channel.apply(&event, &mut ctx, &mut rng).is_none())
        .count();
    let rate = f64::from(u32::try_from(count).unwrap()) / f64::from(EVENTS);
    let bound = 5.0 * (expected * (1.0 - expected) / f64::from(EVENTS)).sqrt();
    assert!(
        (rate - expected).abs() <= bound,
        "rate {rate}, expected {expected} +/- {bound}"
    );
}

#[test]
fn adapter_runs_a_pair_channel_at_its_configured_rate() {
    let channel = CompositeChannel::new(
        "rate",
        channel_action(TwoQubitChannel::depolarizing(0.1).unwrap()),
    );
    assert_rate(&channel, &PAIR, 0.1);
}

#[test]
fn adapter_prob_draws_once_per_pair() {
    let channel = CompositeChannel::new(
        "prob",
        prob(
            0.3,
            channel_action(TwoQubitChannel::depolarizing(1.0).unwrap()),
        )
        .unwrap(),
    );
    assert_rate(&channel, &PAIR, 0.3);
}

fn assert_channel_probability(geometric: bool) {
    let channel = CompositeChannel::new(
        "sampled",
        channel_action(TwoQubitChannel::depolarizing(1.0).unwrap()),
    )
    .with_probability(0.3)
    .unwrap()
    .with_geometric_threshold(if geometric { 2.0 } else { 0.0 })
    .with_min_qubits(0);
    assert_rate(&channel, &PAIR, 0.3);
}

#[test]
fn adapter_channel_probability_is_per_pair_linear() {
    assert_channel_probability(false);
}

#[test]
fn adapter_channel_probability_is_per_pair_geometric() {
    assert_channel_probability(true);
}

fn assert_batch_probability(geometric: bool) {
    let channel = BatchCompositeChannel::new(
        "sampled",
        0.3,
        channel_action(TwoQubitChannel::depolarizing(1.0).unwrap()),
    )
    .unwrap()
    .with_geometric_threshold(if geometric { 2.0 } else { 0.0 })
    .with_min_qubits(0);
    assert_rate(&channel, &PAIR, 0.3);
}

#[test]
fn adapter_batch_probability_is_per_pair_linear() {
    assert_batch_probability(false);
}

#[test]
fn adapter_batch_probability_is_per_pair_geometric() {
    assert_batch_probability(true);
}

#[test]
fn adapter_keeps_single_qubit_rate() {
    let channel = CompositeChannel::new(
        "single",
        prob(
            0.5,
            channel_action(SingleQubitChannel::depolarizing(1.0).unwrap()),
        )
        .unwrap(),
    );
    assert_rate(&channel, &PAIR[..1], 0.5);
}

#[test]
fn adapter_condition_on_leader_decides_for_the_group() {
    let probe = Probe::default();
    let channel = CompositeChannel::new(
        "condition",
        seq![skip_if_leaked(), channel_action(probe.clone())],
    );
    let mut ctx = NoiseContext::new();
    ctx.mark_leaked(PAIR[0]);
    channel.apply(
        &event(&PAIR, false),
        &mut ctx,
        &mut PecosRng::seed_from_u64(42),
    );
    assert!(probe.views.lock().unwrap().is_empty());
}

#[test]
fn adapter_suppressed_group_runs_on_first_leaked_member() {
    let probe = Probe::default();
    let channel = CompositeChannel::new("suppressed", channel_action(probe.clone()));
    let mut ctx = NoiseContext::new();
    ctx.mark_leaked(PAIR[1]);
    channel.apply(
        &event(&PAIR, true),
        &mut ctx,
        &mut PecosRng::seed_from_u64(42),
    );
    assert_eq!(
        *probe.views.lock().unwrap(),
        vec![(1, vec![], PAIR.to_vec())]
    );
    assert_eq!(ctx.group_leader(), None);
}

#[test]
fn adapter_leader_stays_fixed_after_unleak() {
    let probe = Probe::default();
    let channel = CompositeChannel::new("unleak", seq![channel_action(probe.clone()), unleak()]);
    let mut ctx = NoiseContext::new();
    for q in PAIR {
        ctx.mark_leaked(q);
    }
    channel.apply(
        &event(&PAIR, true),
        &mut ctx,
        &mut PecosRng::seed_from_u64(42),
    );
    assert_eq!(
        *probe.views.lock().unwrap(),
        vec![(0, vec![], PAIR.to_vec())]
    );
    assert!(!ctx.any_leaked(&PAIR));
}

#[derive(Clone, Default)]
struct Members(Arc<Mutex<Vec<usize>>>);
impl Primitive for Members {
    fn apply(&self, _: QubitId, ctx: &mut NoiseContext, _: &mut PecosRng) -> CompositeResponse {
        self.0.lock().unwrap().push(ctx.current_qubit_index());
        CompositeResponse::None
    }
    fn describe(&self) -> String {
        "members".into()
    }
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

#[test]
fn adapter_skip_stops_the_sequence() {
    let probe = Probe {
        response: NoiseResponse::SkipGate(PAIR.into_iter().collect()),
        ..Probe::default()
    };
    let members = Members::default();
    let channel = CompositeChannel::new("skip", seq![channel_action(probe), members.clone()]);
    channel.apply(
        &event(&PAIR, false),
        &mut NoiseContext::new(),
        &mut PecosRng::seed_from_u64(42),
    );
    assert_eq!(*members.0.lock().unwrap(), vec![1]);
}

#[test]
fn adapter_non_gate_event_runs_once_with_all_operands() {
    let qubits = [QubitId(0), QubitId(1), QubitId(2)];
    for geometric in [false, true] {
        let probe = Probe::default();
        let members = Members::default();
        let primitive = seq![channel_action(probe.clone()), members.clone()];
        let threshold = if geometric { 2.0 } else { 0.0 };
        let channels: Vec<Box<dyn NoiseChannel>> = vec![
            Box::new(
                CompositeChannel::new("non-gate", primitive.clone())
                    .with_probability(0.999_999)
                    .unwrap()
                    .with_geometric_threshold(threshold)
                    .with_min_qubits(0),
            ),
            Box::new(
                BatchCompositeChannel::new("non-gate batch", 0.999_999, primitive)
                    .unwrap()
                    .with_geometric_threshold(threshold)
                    .with_min_qubits(0),
            ),
        ];
        for channel in channels {
            probe.views.lock().unwrap().clear();
            members.0.lock().unwrap().clear();
            channel.apply(
                &NoiseEvent::AfterPreparation {
                    gate_type: GateType::PZ,
                    qubits: &qubits,
                },
                &mut NoiseContext::new(),
                &mut PecosRng::seed_from_u64(42),
            );
            assert_eq!(
                *probe.views.lock().unwrap(),
                vec![(0, qubits.to_vec(), vec![])]
            );
            assert_eq!(*members.0.lock().unwrap(), vec![0, 1, 2]);
        }
    }
}

#[test]
fn adapter_runs_standalone_without_a_group() {
    let probe = Probe::default();
    let adapter = channel_action(probe.clone());
    let mut ctx = NoiseContext::new();
    let mut rng = PecosRng::seed_from_u64(42);
    for q in PAIR {
        assert!(Primitive::apply(&adapter, q, &mut ctx, &mut rng).is_none());
    }
    assert_eq!(
        *probe.views.lock().unwrap(),
        vec![(0, vec![PAIR[0]], vec![]), (0, vec![PAIR[1]], vec![])]
    );
}

#[test]
fn adapter_unleak_reaches_fast_batch_result() {
    let probe = Probe {
        response: NoiseResponse::MarkUnleaked(smallvec::smallvec![PAIR[0]]),
        ..Probe::default()
    };
    let processor = FastBatchProcessor::new(1.0, channel_action(probe)).unwrap();
    let result = processor.process_range(
        0,
        1,
        &BatchState::new(1),
        &mut NoiseContext::new(),
        &mut PecosRng::seed_from_u64(42),
    );
    assert_eq!(result.unleaked.as_slice(), &PAIR[..1]);
}

#[test]
fn adapter_sites_and_stages_each_run_once() {
    let probe = Probe::default();
    let primitive = two_stage(
        channel_action(probe.clone()),
        seq![channel_action(probe.clone()), channel_action(probe.clone())],
    );
    for geometric in [false, true] {
        let threshold = if geometric { 2.0 } else { 0.0 };
        let channels: Vec<Box<dyn NoiseChannel>> = vec![
            Box::new(
                CompositeChannel::new("stages", primitive.clone())
                    .with_probability(0.3)
                    .unwrap()
                    .with_geometric_threshold(threshold)
                    .with_min_qubits(0),
            ),
            Box::new(
                BatchCompositeChannel::new("stages", 0.3, primitive.clone())
                    .unwrap()
                    .with_geometric_threshold(threshold)
                    .with_min_qubits(0),
            ),
        ];
        for channel in channels {
            let mut rng = PecosRng::seed_from_u64(1568);
            let mut ctx = NoiseContext::new();
            let mut selected = 0_u32;
            for _ in 0..100_000 {
                probe.views.lock().unwrap().clear();
                channel.apply(&event(&PAIR, false), &mut ctx, &mut rng);
                let calls = probe.views.lock().unwrap().len();
                assert!(
                    (2..=3).contains(&calls),
                    "stage 2 must run both sites once, got {calls}"
                );
                selected += u32::from(calls == 3);
            }
            let rate = f64::from(selected) / 100_000.0;
            assert!(
                (rate - 0.3).abs() < 5.0 * (0.3_f64 * 0.7 / 100_000.0).sqrt(),
                "stage 1 rate {rate}"
            );
        }
    }
}

#[test]
fn adapter_leader_cleared_with_operand_state() {
    for clear in [
        NoiseContext::clear_correlation,
        NoiseContext::clear_current_gate,
        NoiseContext::reset,
    ] {
        let mut ctx = NoiseContext::new();
        ctx.set_group_leader(Some(1));
        clear(&mut ctx);
        assert_eq!(ctx.group_leader(), None);
    }
}

#[test]
fn adapter_crosstalk_keeps_each_target_with_or_without_enclosing_group() {
    for nested in [false, true] {
        let probe = Probe::default();
        let crosstalk = CompositeCrosstalkChannel::new("targets", channel_action(probe.clone()))
            .local(|_| vec![QubitId(2), QubitId(3)]);
        let channel: Box<dyn NoiseChannel> = if nested {
            Box::new(CompositeChannel::new(
                "nested targets",
                channel_action(crosstalk),
            ))
        } else {
            Box::new(crosstalk)
        };
        let mut ctx = NoiseContext::new();
        ctx.mark_prepared(QubitId(2));
        ctx.mark_prepared(QubitId(3));
        channel.apply(
            &event(&PAIR, false),
            &mut ctx,
            &mut PecosRng::seed_from_u64(42),
        );
        assert_eq!(
            *probe.views.lock().unwrap(),
            vec![(0, vec![QubitId(2)], vec![]), (0, vec![QubitId(3)], vec![])],
        );
    }
}

#[derive(Clone)]
struct UninstalledOperands;
impl Primitive for UninstalledOperands {
    fn apply(&self, qubit: QubitId, ctx: &mut NoiseContext, _: &mut PecosRng) -> CompositeResponse {
        assert_eq!(ctx.group_leader(), Some(0));
        assert_eq!(ctx.current_qubit_index(), qubit.0);
        assert_ne!(ctx.current_qubit_index(), 0);
        assert_eq!(ctx.current_gate_qubits(), &[]);
        assert_eq!(ctx.other_qubit(), None);
        assert!(!ctx.partner_fired());
        assert!(!partner_leaked().evaluate(qubit, ctx));
        assert!(!partner_fired().evaluate(qubit, ctx));
        assert!(!partner_only_fired().evaluate(qubit, ctx));
        assert!(!i_fired().evaluate(qubit, ctx));
        ctx.set_fired(usize::MAX, true);
        assert!(!ctx.is_fired(usize::MAX));
        CompositeResponse::None
    }
    fn describe(&self) -> String {
        "uninstalled operands".into()
    }
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

#[test]
fn geometric_non_gate_skips_operand_installation_when_leader_is_not_sampled() {
    let qubits: Vec<_> = (0..1_000_000).map(QubitId).collect();
    let seed = 42;
    let indices = crate::noise::composite::batch::GeometricSampler::new(1e-4)
        .unwrap()
        .sample_range(0, qubits.len(), &mut PecosRng::seed_from_u64(seed));
    assert!(indices.first().is_some_and(|&index| index > 0));
    assert!(!indices.contains(&0));

    let probe = Probe::default();
    let members = Members::default();
    let primitive = seq![
        UninstalledOperands,
        channel_action(probe.clone()),
        members.clone(),
        two_qubit_pauli(),
        two_qubit_emission(),
        partner_depolarize(),
    ];
    let channels: Vec<Box<dyn NoiseChannel>> = vec![
        Box::new(
            CompositeChannel::new("sparse preparation", primitive.clone())
                .with_probability(1e-4)
                .unwrap(),
        ),
        Box::new(BatchCompositeChannel::new("sparse preparation", 1e-4, primitive).unwrap()),
    ];
    for channel in channels {
        members.0.lock().unwrap().clear();
        channel.apply(
            &NoiseEvent::AfterPreparation {
                gate_type: GateType::PZ,
                qubits: &qubits,
            },
            &mut NoiseContext::new(),
            &mut PecosRng::seed_from_u64(seed),
        );
        assert_eq!(*probe.views.lock().unwrap(), Vec::<View>::new());
        assert_eq!(*members.0.lock().unwrap(), indices);
    }
}
