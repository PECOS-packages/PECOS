//! Regression tests for channel adapters, operand groups, response targets and event state.
use super::*;
use crate::noise::composite::batch_composite::{BatchState, FastBatchProcessor};
use crate::noise::composite::prelude::*;
use crate::noise::composite::{BatchCompositeChannel, CompositeCrosstalkChannel, Condition};
use std::sync::{Arc, Mutex};

const PAIR: [QubitId; 2] = [QubitId(0), QubitId(1)];

#[derive(Debug, PartialEq, Eq)]
struct View {
    member: usize,
    operands: Vec<QubitId>,
    suppressed: Vec<QubitId>,
}
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
        self.views.lock().unwrap().push(View {
            member: ctx.current_qubit_index(),
            operands: event.qubits().to_vec(),
            suppressed: event.leaked_suppressed().to_vec(),
        });
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
        vec![View {
            member: 1,
            operands: vec![],
            suppressed: PAIR.to_vec()
        }]
    );
    assert_eq!(ctx.group_leader(), None);
}

#[test]
fn adapter_single_pass_leader_stays_fixed_after_unleak() {
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
        vec![View {
            member: 0,
            operands: vec![],
            suppressed: PAIR.to_vec()
        }]
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
    assert_eq!(*members.0.lock().unwrap(), Vec::<usize>::new());
}

#[test]
fn adapter_non_gate_event_runs_once_with_all_operands() {
    let qubits = [QubitId(0), QubitId(1), QubitId(2)];
    let sampled = sampled_indices(42, 0.999_999, qubits.len());
    assert!(sampled.contains(&0));
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
                vec![View {
                    member: 0,
                    operands: qubits.to_vec(),
                    suppressed: vec![]
                }]
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
        vec![
            View {
                member: 0,
                operands: vec![PAIR[0]],
                suppressed: vec![]
            },
            View {
                member: 0,
                operands: vec![PAIR[1]],
                suppressed: vec![]
            }
        ]
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
fn fast_batch_skip_list_contains_only_invoked_qubits_that_are_skipped() {
    for target in PAIR {
        let probe = Probe {
            response: NoiseResponse::SkipGate(smallvec::smallvec![target]),
            ..Probe::default()
        };
        let processor = FastBatchProcessor::new(1.0, channel_action(probe)).unwrap();
        let result = processor.process_range(
            0,
            1,
            &BatchState::new(2),
            &mut NoiseContext::new(),
            &mut PecosRng::seed_from_u64(42),
        );
        let expected = if target == PAIR[0] {
            vec![PAIR[0]]
        } else {
            vec![]
        };
        assert_eq!(result.skip.as_slice(), expected);
        assert_eq!(result.responses.len(), 1);
        assert_eq!(result.responses[0].0, PAIR[0]);
        assert_eq!(
            result.responses[0].1.channel_skipped_qubits().as_slice(),
            &[target]
        );
    }
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
            vec![
                View {
                    member: 0,
                    operands: vec![QubitId(2)],
                    suppressed: vec![]
                },
                View {
                    member: 0,
                    operands: vec![QubitId(3)],
                    suppressed: vec![]
                }
            ],
        );
    }
}

#[derive(Clone)]
struct UninstalledOperands;
impl Primitive for UninstalledOperands {
    fn apply(&self, qubit: QubitId, ctx: &mut NoiseContext, _: &mut PecosRng) -> CompositeResponse {
        assert_eq!(ctx.group_leader(), None);
        assert_eq!(ctx.current_qubit_index(), qubit.0);
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

fn sampled_indices(seed: u64, probability: f64, count: usize) -> Vec<usize> {
    crate::noise::composite::batch::GeometricSampler::new(probability)
        .unwrap()
        .sample_range(0, count, &mut PecosRng::seed_from_u64(seed))
}

#[test]
fn non_readers_never_install_geometric_non_gate_operands() {
    for (count, probability) in [(1_000_000, 1e-4), (3, 0.999_999)] {
        let qubits: Vec<_> = (0..count).map(QubitId).collect();
        let indices = sampled_indices(42, probability, count);
        assert_ne!(indices, Vec::<usize>::new());
        assert_eq!(indices.contains(&0), count == 3);
        let members = Members::default();
        let primitive = seq![
            UninstalledOperands,
            members.clone(),
            two_qubit_pauli(),
            two_qubit_emission()
        ];
        let channels: Vec<Box<dyn NoiseChannel>> = vec![
            Box::new(
                CompositeChannel::new("non-reader", primitive.clone())
                    .with_probability(probability)
                    .unwrap()
                    .with_geometric_threshold(2.0)
                    .with_min_qubits(0),
            ),
            Box::new(
                BatchCompositeChannel::new("non-reader", probability, primitive)
                    .unwrap()
                    .with_geometric_threshold(2.0)
                    .with_min_qubits(0),
            ),
        ];
        for channel in channels {
            members.0.lock().unwrap().clear();
            channel.apply(
                &NoiseEvent::AfterPreparation {
                    gate_type: GateType::PZ,
                    qubits: &qubits,
                },
                &mut NoiseContext::new(),
                &mut PecosRng::seed_from_u64(42),
            );
            assert_eq!(*members.0.lock().unwrap(), indices);
        }
    }
}

#[derive(Clone)]
struct OperandReader;
impl Primitive for OperandReader {
    fn reads_operands(&self) -> bool {
        true
    }
    fn apply(&self, qubit: QubitId, ctx: &mut NoiseContext, _: &mut PecosRng) -> CompositeResponse {
        assert_eq!(ctx.current_gate_qubits(), PAIR);
        assert_eq!(ctx.current_qubit_index(), qubit.0);
        assert_eq!(ctx.other_qubit(), Some(PAIR[1 - qubit.0]));
        assert_eq!(partner_leaked().evaluate(qubit, ctx), qubit == PAIR[1]);
        assert!(any_qubit_leaked().evaluate(qubit, ctx));
        ctx.set_fired(0, true);
        assert_eq!(partner_fired().evaluate(qubit, ctx), qubit == PAIR[1]);
        assert_eq!(partner_only_fired().evaluate(qubit, ctx), qubit == PAIR[1]);
        CompositeResponse::None
    }
    fn describe(&self) -> String {
        "operand reader".into()
    }
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

#[test]
fn geometric_readers_see_operands_independently_of_leader_sampling() {
    for (seed, expected_indices) in [(1, vec![1]), (2, vec![0, 1])] {
        let indices = sampled_indices(seed, 0.5, 2);
        assert_eq!(indices, expected_indices);
        assert_eq!(indices.contains(&0), seed == 2);
        let members = Members::default();
        let primitive = seq![OperandReader, members.clone()];
        let channels: Vec<Box<dyn NoiseChannel>> = vec![
            Box::new(
                CompositeChannel::new("reader", primitive.clone())
                    .with_probability(0.5)
                    .unwrap()
                    .with_geometric_threshold(2.0)
                    .with_min_qubits(0),
            ),
            Box::new(
                BatchCompositeChannel::new("reader", 0.5, primitive)
                    .unwrap()
                    .with_geometric_threshold(2.0)
                    .with_min_qubits(0),
            ),
        ];
        for channel in channels {
            members.0.lock().unwrap().clear();
            let mut ctx = NoiseContext::new();
            ctx.mark_leaked(PAIR[0]);
            channel.apply(
                &NoiseEvent::AfterPreparation {
                    gate_type: GateType::PZ,
                    qubits: &PAIR,
                },
                &mut ctx,
                &mut PecosRng::seed_from_u64(seed),
            );
            assert_eq!(*members.0.lock().unwrap(), indices);
        }
    }
}

#[derive(Clone)]
struct UnleakFirst;
impl Primitive for UnleakFirst {
    fn apply(&self, qubit: QubitId, ctx: &mut NoiseContext, _: &mut PecosRng) -> CompositeResponse {
        if qubit == PAIR[0] {
            ctx.mark_unleaked(qubit);
            CompositeResponse::Unleak
        } else {
            CompositeResponse::None
        }
    }
    fn describe(&self) -> String {
        "unleak first".into()
    }
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

#[test]
fn stage_two_chooses_the_first_still_leaked_member() {
    let probe = Probe::default();
    for geometric in [false, true] {
        let primitive = two_stage(UnleakFirst, channel_action(probe.clone()));
        let threshold = if geometric { 2.0 } else { 0.0 };
        let channel = CompositeChannel::new("two pass", primitive.clone())
            .with_probability(0.999_999)
            .unwrap()
            .with_geometric_threshold(threshold)
            .with_min_qubits(0);
        let batch = BatchCompositeChannel::new("two pass", 0.999_999, primitive)
            .unwrap()
            .with_geometric_threshold(threshold)
            .with_min_qubits(0);
        let channels: Vec<Box<dyn NoiseChannel>> = vec![
            Box::new(channel.clone()),
            Box::new(batch.clone()),
            Box::new(CompositeChannel::new("nested", channel_action(channel))),
            Box::new(CompositeChannel::new("nested batch", channel_action(batch))),
        ];
        for channel in channels {
            probe.views.lock().unwrap().clear();
            let mut ctx = NoiseContext::new();
            for qubit in PAIR {
                ctx.mark_leaked(qubit);
            }
            channel.apply(
                &event(&PAIR, true),
                &mut ctx,
                &mut PecosRng::seed_from_u64(42),
            );
            assert_eq!(
                *probe.views.lock().unwrap(),
                vec![View {
                    member: 1,
                    operands: vec![],
                    suppressed: PAIR.to_vec()
                }]
            );
        }
    }
}

fn skip_targets(response: &NoiseResponse) -> Vec<QubitId> {
    match response {
        NoiseResponse::SkipGate(qubits) => qubits.to_vec(),
        NoiseResponse::Multiple(responses) => responses.iter().flat_map(skip_targets).collect(),
        _ => vec![],
    }
}

fn gates(response: &NoiseResponse) -> Vec<GateCommand> {
    match response {
        NoiseResponse::InjectGates(gates) => gates.to_vec(),
        NoiseResponse::Multiple(responses) => responses.iter().flat_map(gates).collect(),
        _ => vec![],
    }
}

#[test]
fn adapter_skips_stop_exactly_the_named_members_without_duplicate_marks() {
    for skipped in [&PAIR[..], &PAIR[..1], &PAIR[1..]] {
        for suppressed in [false, true] {
            let probe = Probe {
                response: NoiseResponse::SkipGate(skipped.iter().copied().collect()),
                ..Probe::default()
            };
            let inner = CompositeChannel::new("inner", channel_action(probe.clone()));
            let adapters = [channel_action(probe.clone()), channel_action(inner)];
            for adapter in adapters {
                probe.views.lock().unwrap().clear();
                let channel = CompositeChannel::new("skip", seq![adapter, inject_x()]);
                let mut ctx = NoiseContext::new();
                if suppressed {
                    for q in PAIR {
                        ctx.mark_leaked(q);
                    }
                }
                let response = channel.apply(
                    &event(&PAIR, suppressed),
                    &mut ctx,
                    &mut PecosRng::seed_from_u64(42),
                );
                assert_eq!(probe.views.lock().unwrap().len(), 1);
                assert_eq!(skip_targets(&response), skipped);
                let expected: Vec<_> = PAIR
                    .into_iter()
                    .filter(|q| !skipped.contains(q))
                    .map(GateCommand::x)
                    .collect();
                assert_eq!(gates(&response), expected);
            }
        }
    }
}

fn sequence(primitives: Vec<Box<dyn Primitive>>, boxed: bool) -> Box<dyn Primitive> {
    use crate::noise::composite::{BoxSeq, Seq};
    if boxed {
        Box::new(BoxSeq::new(primitives))
    } else {
        Box::new(Seq::new(primitives))
    }
}

fn skipping_probe() -> Probe {
    Probe {
        response: NoiseResponse::SkipGate(PAIR.into_iter().collect()),
        ..Probe::default()
    }
}

fn assert_guarded_sequence_stops(primitive: Box<dyn Primitive>, probe: &Probe, seed: u64) {
    let response = CompositeChannel::new("guarded skip", seq![primitive]).apply(
        &event(&PAIR, false),
        &mut NoiseContext::new(),
        &mut PecosRng::seed_from_u64(seed),
    );
    assert_eq!(probe.views.lock().unwrap().len(), 1);
    assert_eq!(skip_targets(&response), PAIR);
    assert_eq!(gates(&response), vec![]);
}

fn leader_hit_partner_miss_seed() -> u64 {
    use rand::RngExt;
    let seed = 1;
    let mut rng = PecosRng::seed_from_u64(seed);
    // Probe and gate injection consume no RNG draws, so these are the two guard draws.
    assert!(rng.random::<f64>() < 0.5, "leader must hit the guard");
    assert!(rng.random::<f64>() >= 0.5, "non-leader must miss the guard");
    seed
}

struct GuardEntry<P> {
    inner: P,
    members: Members,
}
impl<P: Primitive> Primitive for GuardEntry<P> {
    fn reads_operands(&self) -> bool {
        self.inner.reads_operands()
    }
    fn apply(
        &self,
        qubit: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        self.members.apply(qubit, ctx, rng);
        self.inner.apply(qubit, ctx, rng)
    }
    fn describe(&self) -> String {
        "guard entry".into()
    }
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(GuardEntry {
            inner: self.inner.clone_box(),
            members: self.members.clone(),
        })
    }
}

#[test]
fn sequence_remembers_skip_when_probability_guard_misses_partner() {
    let seed = leader_hit_partner_miss_seed();
    for boxed in [false, true] {
        let probe = skipping_probe();
        let entered = Members::default();
        let primitive = sequence(
            vec![
                Box::new(
                    prob(
                        0.5,
                        GuardEntry {
                            inner: channel_action(probe.clone()),
                            members: entered.clone(),
                        },
                    )
                    .unwrap(),
                ),
                Box::new(inject_x()),
            ],
            boxed,
        );
        assert_guarded_sequence_stops(primitive, &probe, seed);
        assert_eq!(
            *entered.0.lock().unwrap(),
            vec![0],
            "non-leader must miss the guard"
        );
    }
}

#[test]
fn sequence_remembers_skip_when_condition_excludes_partner() {
    use crate::noise::composite::FnCondition;
    for boxed in [false, true] {
        let probe = skipping_probe();
        let evaluated = Arc::new(Mutex::new(vec![]));
        let captured = Arc::clone(&evaluated);
        let condition = FnCondition::new("leader only", move |q, _: &NoiseContext| {
            let hit = q == PAIR[0];
            captured.lock().unwrap().push((q, hit));
            hit
        });
        let primitive = sequence(
            vec![
                Box::new(when(condition, channel_action(probe.clone()), nothing())),
                Box::new(inject_x()),
            ],
            boxed,
        );
        assert_guarded_sequence_stops(primitive, &probe, 42);
        assert_eq!(
            *evaluated.lock().unwrap(),
            vec![(PAIR[0], true), (PAIR[1], false)]
        );
    }
}

#[test]
fn outer_sequence_remembers_skip_when_guard_bypasses_inner_sequence() {
    let seed = leader_hit_partner_miss_seed();
    for outer_boxed in [false, true] {
        for inner_boxed in [false, true] {
            let probe = skipping_probe();
            let entered = Members::default();
            let inner = sequence(
                vec![
                    Box::new(channel_action(probe.clone())),
                    Box::new(inject_x()),
                ],
                inner_boxed,
            );
            let outer = sequence(
                vec![
                    Box::new(
                        prob(
                            0.5,
                            GuardEntry {
                                inner,
                                members: entered.clone(),
                            },
                        )
                        .unwrap(),
                    ),
                    Box::new(inject_z()),
                ],
                outer_boxed,
            );
            assert_guarded_sequence_stops(outer, &probe, seed);
            assert_eq!(
                *entered.0.lock().unwrap(),
                vec![0],
                "non-leader must bypass the inner sequence"
            );
        }
    }
}

#[test]
fn sequence_remembers_skip_through_function_and_choice_guards() {
    use crate::noise::composite::BoxSample;
    let seed = leader_hit_partner_miss_seed();
    for boxed in [false, true] {
        let probe = skipping_probe();
        let entered = Members::default();
        let guards: Vec<Box<dyn Primitive>> = vec![
            Box::new(prob_fn(
                |_| Probability::new(0.5).unwrap(),
                GuardEntry {
                    inner: channel_action(probe.clone()),
                    members: entered.clone(),
                },
            )),
            Box::new(
                BoxSample::new(vec![
                    (
                        0.5,
                        Box::new(GuardEntry {
                            inner: channel_action(probe.clone()),
                            members: entered.clone(),
                        }),
                    ),
                    (0.5, Box::new(nothing())),
                ])
                .unwrap(),
            ),
        ];
        for guard in guards {
            probe.views.lock().unwrap().clear();
            entered.0.lock().unwrap().clear();
            assert_guarded_sequence_stops(
                sequence(vec![guard, Box::new(inject_x())], boxed),
                &probe,
                seed,
            );
            assert_eq!(
                *entered.0.lock().unwrap(),
                vec![0],
                "non-leader must miss the guard"
            );
        }
    }
}

#[test]
fn sequence_stops_at_the_recorded_step_and_keeps_earlier_effects() {
    let seed = leader_hit_partner_miss_seed();
    for boxed in [false, true] {
        let probe = skipping_probe();
        let primitive = sequence(
            vec![
                Box::new(inject_z()),
                Box::new(prob(0.5, channel_action(probe.clone())).unwrap()),
                Box::new(inject_x()),
            ],
            boxed,
        );
        let response = CompositeChannel::new("step index", seq![primitive]).apply(
            &event(&PAIR, false),
            &mut NoiseContext::new(),
            &mut PecosRng::seed_from_u64(seed),
        );
        assert_eq!(skip_targets(&response), PAIR);
        assert_eq!(
            gates(&response),
            vec![GateCommand::z(PAIR[0]), GateCommand::z(PAIR[1])]
        );
        assert_eq!(probe.views.lock().unwrap().len(), 1);
    }
}

const OPERAND_BUFFER_TEST_SIZE: usize = 1024;

#[derive(Clone, Default)]
struct OperandBufferProbe(Arc<Mutex<Vec<usize>>>);
impl Primitive for OperandBufferProbe {
    fn reads_operands(&self) -> bool {
        true
    }
    fn apply(&self, qubit: QubitId, ctx: &mut NoiseContext, _: &mut PecosRng) -> CompositeResponse {
        assert!(
            ctx.current_gate_qubits_spilled(),
            "buffer must be heap-backed for address reuse to be meaningful"
        );
        assert_eq!(ctx.current_gate_qubits().len(), OPERAND_BUFFER_TEST_SIZE);
        assert_eq!(ctx.current_qubit_index(), qubit.0);
        assert_eq!(ctx.current_gate_qubits()[ctx.current_qubit_index()], qubit);
        self.0
            .lock()
            .unwrap()
            .push(ctx.current_gate_qubits().as_ptr().addr());
        CompositeResponse::None
    }
    fn describe(&self) -> String {
        "operand buffer".into()
    }
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

#[test]
fn two_pass_non_gate_reader_reuses_one_operand_buffer() {
    let qubits: Vec<_> = (0..OPERAND_BUFFER_TEST_SIZE).map(QubitId).collect();
    for geometric in [false, true] {
        let probe = OperandBufferProbe::default();
        let primitive = two_stage(probe.clone(), probe.clone());
        let threshold = if geometric { 2.0 } else { 0.0 };
        let channels: Vec<Box<dyn NoiseChannel>> = vec![
            Box::new(
                CompositeChannel::new("two pass", primitive.clone())
                    .with_probability(0.999_999)
                    .unwrap()
                    .with_geometric_threshold(threshold)
                    .with_min_qubits(0),
            ),
            Box::new(
                BatchCompositeChannel::new("two pass", 0.999_999, primitive)
                    .unwrap()
                    .with_geometric_threshold(threshold)
                    .with_min_qubits(0),
            ),
        ];
        for channel in channels {
            probe.0.lock().unwrap().clear();
            channel.apply(
                &NoiseEvent::AfterPreparation {
                    gate_type: GateType::PZ,
                    qubits: &qubits,
                },
                &mut NoiseContext::new(),
                &mut PecosRng::seed_from_u64(42),
            );
            let buffers = probe.0.lock().unwrap();
            assert_eq!(buffers.len(), 2 * OPERAND_BUFFER_TEST_SIZE);
            assert!(
                buffers.iter().all(|address| *address == buffers[0]),
                "each stage must reuse the group's operand buffer"
            );
        }
    }
}

#[test]
fn sequence_skip_records_are_scoped_to_site_step_group_stage_and_event() {
    let mut ctx = NoiseContext::new();
    ctx.set_current_qubit_index(0, &PAIR);
    ctx.set_group_leader(Some(0));
    ctx.record_sequence_skips(7, 0, PAIR.into_iter().collect());
    assert!(ctx.sequence_skipped(7, 0, PAIR[1]));
    assert!(!ctx.sequence_skipped(8, 0, PAIR[1]));
    assert!(!ctx.sequence_skipped(7, 1, PAIR[1]));
    let saved = ctx.enter_event();
    assert!(!ctx.sequence_skipped(7, 0, PAIR[1]));
    ctx.record_sequence_skips(8, 0, PAIR.into_iter().collect());
    ctx.exit_event(saved);
    assert!(ctx.sequence_skipped(7, 0, PAIR[1]));
    assert!(!ctx.sequence_skipped(8, 0, PAIR[1]));
    ctx.set_current_qubit_index(0, &[QubitId(2), QubitId(3)]);
    assert!(!ctx.sequence_skipped(7, 0, PAIR[1]));
    ctx.record_sequence_skips(7, 0, PAIR.into_iter().collect());
    ctx.set_group_leader(Some(1));
    assert!(!ctx.sequence_skipped(7, 0, PAIR[1]));
    for clear in [
        NoiseContext::clear_correlation,
        NoiseContext::clear_current_gate,
        NoiseContext::reset,
    ] {
        ctx.record_sequence_skips(7, 0, PAIR.into_iter().collect());
        clear(&mut ctx);
        assert!(!ctx.sequence_skipped(7, 0, PAIR[1]));
    }
}

#[derive(Clone)]
struct Capability(bool);
impl Primitive for Capability {
    fn reads_operands(&self) -> bool {
        self.0
    }
    fn apply(&self, _: QubitId, _: &mut NoiseContext, _: &mut PecosRng) -> CompositeResponse {
        CompositeResponse::None
    }
    fn describe(&self) -> String {
        "capability".into()
    }
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}
#[derive(Clone)]
struct ConditionCapability(bool);
impl Condition for ConditionCapability {
    fn reads_operands(&self) -> bool {
        self.0
    }
    fn evaluate(&self, _: QubitId, _: &NoiseContext) -> bool {
        false
    }
    fn name(&self) -> &'static str {
        "condition capability"
    }
}

#[test]
fn probability_combinators_forward_operand_reads() {
    for reads in [false, true] {
        let child = Capability(reads);
        let wrappers: Vec<Box<dyn Primitive>> = vec![
            Box::new(prob(0.5, child.clone()).unwrap()),
            Box::new(prob_fn(|_| Probability::new(0.5).unwrap(), child.clone())),
            Box::new(prob_linear(0.1, child.clone()).unwrap()),
            Box::new(prob_quadratic(0.1, child).unwrap()),
        ];
        for wrapper in wrappers {
            assert_eq!(wrapper.reads_operands(), reads, "{}", wrapper.describe());
        }
    }
}

#[test]
fn sequences_and_choices_forward_operand_reads_from_each_child() {
    use crate::noise::composite::{BoxSample, Sample, Seq};
    for selected in [None, Some(0), Some(1)] {
        let children = [
            Capability(selected == Some(0)),
            Capability(selected == Some(1)),
        ];
        let wrappers: Vec<Box<dyn Primitive>> = vec![
            Box::new(Seq::new(children.to_vec())),
            Box::new(seq![children[0].clone(), children[1].clone()]),
            Box::new(
                Sample::new(vec![(0.5, children[0].clone()), (0.5, children[1].clone())]).unwrap(),
            ),
            Box::new(
                BoxSample::new(vec![
                    (0.5, Box::new(children[0].clone())),
                    (0.5, Box::new(children[1].clone())),
                ])
                .unwrap(),
            ),
            Box::new(two_stage(children[0].clone(), children[1].clone())),
        ];
        for wrapper in wrappers {
            assert_eq!(
                wrapper.reads_operands(),
                selected.is_some(),
                "{}",
                wrapper.describe()
            );
            assert_eq!(wrapper.clone_box().reads_operands(), selected.is_some());
        }
    }
}

#[test]
fn conditional_combinators_forward_conditions_and_both_branches() {
    for selected in [None, Some(0), Some(1), Some(2)] {
        let conditional = when(
            ConditionCapability(selected == Some(0)),
            Capability(selected == Some(1)),
            Capability(selected == Some(2)),
        );
        assert_eq!(conditional.reads_operands(), selected.is_some());
    }
    for reads in [false, true] {
        assert_eq!(skip_if(ConditionCapability(reads)).reads_operands(), reads);
        assert_eq!(
            if_then(ConditionCapability(reads), nothing()).reads_operands(),
            reads
        );
        assert_eq!(
            if_then(ConditionCapability(false), Capability(reads)).reads_operands(),
            reads
        );
        let boxed: Box<dyn Primitive> = Box::new(Capability(reads));
        assert_eq!(Primitive::reads_operands(&boxed), reads);
    }
}

#[test]
fn operand_actions_conditions_and_callbacks_declare_their_reads() {
    use crate::noise::composite::FnCondition;
    let readers: Vec<Box<dyn Primitive>> = vec![
        Box::new(channel_action(Probe::default())),
        Box::new(partner_depolarize()),
        Box::new(two_qubit_emission_with_partner_depolarize()),
        Box::new(independent_emission(0.1).unwrap()),
        Box::new(zz_dephasing(0.1).unwrap()),
        Box::new(zz_dephasing_rate(0.1).unwrap()),
        Box::new(if_then(partner_leaked(), nothing())),
        Box::new(if_then(any_qubit_leaked(), nothing())),
        Box::new(if_then(partner_fired(), nothing())),
        Box::new(if_then(partner_only_fired(), nothing())),
    ];
    for reader in readers {
        assert!(reader.reads_operands(), "{}", reader.describe());
    }
    assert!(!Primitive::reads_operands(&pauli()));
    assert!(!Primitive::reads_operands(&two_qubit_pauli()));
    assert!(!Primitive::reads_operands(&two_qubit_emission()));
    assert!(!i_fired().reads_operands());
    let callback = FnCondition::new("callback", |_: QubitId, ctx: &NoiseContext| {
        !ctx.current_gate_qubits().is_empty()
    });
    assert!(!callback.reads_operands());
    assert!(if_then(callback.with_operand_reads(), nothing()).reads_operands());
}
