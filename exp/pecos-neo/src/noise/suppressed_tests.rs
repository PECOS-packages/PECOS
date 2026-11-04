//! Consumer regression tests for executed and leaked-suppressed operands.
use super::*;
use crate::noise::composite::prelude::*;
use crate::noise::composite::{BatchCompositeChannel, CompositeNoiseModelBuilder};

fn injected(response: NoiseResponse) -> Vec<GateCommand> {
    match response {
        NoiseResponse::InjectGates(g) => g.to_vec(),
        NoiseResponse::Multiple(rs) => rs.into_iter().flat_map(injected).collect(),
        _ => vec![],
    }
}
fn unmarked(response: &NoiseResponse) -> Vec<QubitId> {
    match response {
        NoiseResponse::MarkUnleaked(qs) => qs.to_vec(),
        NoiseResponse::Multiple(rs) => rs.iter().flat_map(unmarked).collect(),
        _ => vec![],
    }
}

/// A leaked or suppressed pair may receive a seep twirl, never gate error.
fn assert_only_seep_injections(response: &NoiseResponse) {
    let seeping = unmarked(response);
    let gates = injected(response.clone());
    if seeping.is_empty() {
        assert_eq!(gates, vec![], "nothing seeped, so nothing may be injected");
    }
    let mut targets = std::collections::BTreeSet::new();
    for gate in gates {
        assert!(
            matches!(gate.gate_type, GateType::X | GateType::Y | GateType::Z),
            "seepage must inject only X, Y or Z, never a gate dagger: {gate:?}"
        );
        assert_eq!(
            gate.qubits.len(),
            1,
            "seepage must be single-qubit: {gate:?}"
        );
        let qubit = gate.qubits[0];
        assert!(
            seeping.contains(&qubit),
            "injected on non-seeping qubit {qubit:?}; MarkUnleaked={seeping:?}"
        );
        assert!(
            targets.insert(qubit),
            "more than one injection on seeping qubit {qubit:?}"
        );
    }
}

fn after<'a>(qubits: &'a [QubitId], suppressed: &'a [QubitId]) -> NoiseEvent<'a> {
    NoiseEvent::AfterGate {
        gate_type: GateType::CX,
        gate_id: None,
        qubits,
        leaked_suppressed: suppressed,
        group_size: std::num::NonZeroUsize::new(2).unwrap(),
        angles: &[],
    }
}

#[test]
fn suppressed_idle_after_two_qubit() {
    let channel = IdleChannel::linear(1.0)
        .unwrap()
        .with_linear_weights(super::PauliWeights::x_only())
        .unwrap()
        .with_idle_after_2q(1.0)
        .unwrap();
    for all_suppressed in [false, true] {
        let qubits: Vec<_> = (0..6).map(QubitId).collect();
        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0));
        let skip = if all_suppressed {
            vec![QubitId(0), QubitId(2), QubitId(4)]
        } else {
            vec![QubitId(0), QubitId(2)]
        };
        let partition =
            partition_gate_qubits(&qubits, gate_group_size(2, qubits.len()), &skip, Some(&ctx));
        let gates = injected(channel.apply(
            &after(&partition.executed, &partition.leaked_suppressed),
            &mut ctx,
            &mut PecosRng::seed_from_u64(42),
        ));
        let targets: Vec<_> = gates
            .iter()
            .flat_map(|g| g.qubits.iter().copied())
            .collect();
        assert!(targets.contains(&QubitId(1)));
        assert!(
            !targets.contains(&QubitId(0))
                && !targets.contains(&QubitId(2))
                && !targets.contains(&QubitId(3))
        );
        assert_eq!(targets.len(), if all_suppressed { 1 } else { 3 });
    }
    // A child can leak an executed member after partitioning.
    let mut ctx = NoiseContext::new();
    ctx.mark_leaked(QubitId(0));
    let gates = injected(channel.apply(
        &after(&[QubitId(0), QubitId(1)], &[]),
        &mut ctx,
        &mut PecosRng::seed_from_u64(42),
    ));
    assert_eq!(gates.len(), 1);
    assert_eq!(gates[0].qubits.as_slice(), &[QubitId(1)]);
}

#[test]
fn suppressed_crosstalk_sources_and_exclusions() {
    fn neighbors(sources: &[QubitId]) -> Vec<QubitId> {
        sources.iter().map(|q| QubitId(q.0 + 6)).collect()
    }
    for composite in [false, true] {
        for local in [false, true] {
            let channel: Box<dyn NoiseChannel> = if composite {
                let c = CompositeCrosstalkChannel::new("crosstalk", inject_x()).responds_to_gates();
                Box::new(if local {
                    c.local(neighbors)
                } else {
                    c.global()
                })
            } else {
                let c = CrosstalkChannel::new();
                Box::new(if local {
                    c.with_local_rate(1.0)
                        .unwrap()
                        .with_neighbor_function(neighbors)
                } else {
                    c.with_global_rate(1.0).unwrap()
                })
            };
            let mut ctx = NoiseContext::new();
            for q in 0..12 {
                ctx.mark_prepared(QubitId(q));
            }
            ctx.mark_leaked(QubitId(0));
            // (0,1) was suppressed, (2,3) dropped, (4,5) executed.
            let event = after(&[QubitId(4), QubitId(5)], &[QubitId(0), QubitId(1)]);
            let targets: Vec<_> =
                injected(channel.apply(&event, &mut ctx, &mut PecosRng::seed_from_u64(42)))
                    .into_iter()
                    .flat_map(|g| g.qubits)
                    .collect();
            for q in [0, 1, 4, 5] {
                assert!(
                    !targets.contains(&QubitId(q)),
                    "composite={composite} local={local}"
                );
            }
            if local {
                for q in [6, 7, 10, 11] {
                    assert!(targets.contains(&QubitId(q)));
                }
                for q in [8, 9] {
                    assert!(!targets.contains(&QubitId(q)));
                }
            } else {
                for q in [2, 3] {
                    assert!(targets.contains(&QubitId(q)));
                }
            }
        }
    }
}

#[test]
fn suppressed_leakage_partner_noise() {
    for mixed in [false, true] {
        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0));
        let executed = if mixed {
            vec![QubitId(2), QubitId(3)]
        } else {
            vec![]
        };
        let event = after(&executed, &[QubitId(0), QubitId(1)]);
        let gates = injected(
            LeakageChannel::new()
                .with_partner_pauli_probability(1.0)
                .unwrap()
                .apply(&event, &mut ctx, &mut PecosRng::seed_from_u64(42)),
        );
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].qubits.as_slice(), &[QubitId(1)]);
    }
}

fn within_four_sigma(count: u32, samples: u32, rate: f64) {
    let n = f64::from(samples);
    let mean = n * rate;
    let bound = 4.0 * (n * rate * (1.0 - rate)).sqrt();
    assert!(
        (f64::from(count) - mean).abs() <= bound,
        "count={count}, expected={mean}, bound={bound}"
    );
}

#[test]
fn suppressed_two_qubit_seepage_rate() {
    const N: u32 = 100_000;
    for try_apply in [false, true] {
        for suppressed in [false, true] {
            for both in [false, true] {
                for (p, r) in [(0.5, 0.5), (0.5, 0.0), (0.0, 0.5)] {
                    let channel = TwoQubitChannel::depolarizing(p)
                        .unwrap()
                        .with_leakage(r)
                        .unwrap()
                        .with_seepage(0.5)
                        .unwrap();
                    let mut ctx = NoiseContext::new();
                    ctx.mark_leaked(QubitId(0));
                    if both {
                        ctx.mark_leaked(QubitId(1));
                    }
                    let pair = [QubitId(0), QubitId(1)];
                    let event = if suppressed {
                        after(&[], &pair)
                    } else {
                        after(&pair, &[])
                    };
                    let mut rng = PecosRng::seed_from_u64(19);
                    let mut first = 0;
                    let mut joint = 0;
                    for _ in 0..N {
                        let response = if try_apply {
                            channel
                                .try_apply(&event, &mut ctx, &mut rng)
                                .unwrap_or_default()
                        } else {
                            channel.apply(&event, &mut ctx, &mut rng)
                        };
                        let seeping = unmarked(&response);
                        first += u32::from(seeping.contains(&QubitId(0)));
                        joint += u32::from(seeping.len() == 2);
                        assert_only_seep_injections(&response);
                    }
                    within_four_sigma(first, N, p * r * 0.5);
                    within_four_sigma(joint, N, if both { p * r * 0.25 } else { 0.0 });
                }
            }
        }
    }
    // Angle scaling participates in seepage, including its zero endpoint.
    for angle in [Angle64::ZERO, Angle64::QUARTER_TURN] {
        let channel = TwoQubitChannel::depolarizing(0.5)
            .unwrap()
            .with_angle_scaling(AngleScaling::linear())
            .unwrap()
            .with_leakage(0.5)
            .unwrap()
            .with_seepage(0.5)
            .unwrap();
        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0));
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::RZZ,
            gate_id: None,
            qubits: &[],
            leaked_suppressed: &[QubitId(0), QubitId(1)],
            group_size: std::num::NonZeroUsize::new(2).unwrap(),
            angles: &[angle],
        };
        let mut count = 0;
        let mut rng = PecosRng::seed_from_u64(41);
        for _ in 0..N {
            let response = channel.apply(&event, &mut ctx, &mut rng);
            count += u32::from(!unmarked(&response).is_empty());
            assert_only_seep_injections(&response);
        }
        within_four_sigma(count, N, if angle == Angle64::ZERO { 0.0 } else { 0.0625 });
    }
}

#[test]
fn suppressed_composite_builder_seepage_rate() {
    const N: u32 = 100_000;
    for gate in [GateType::X, GateType::CX, GateType::RZZ] {
        for r in [0.0, 0.5] {
            for scale in [0.0, 1.0] {
                let mut noise = CompositeNoiseModelBuilder::new()
                    .with_p1(0.5)
                    .unwrap()
                    .with_p2(0.5)
                    .unwrap()
                    .with_p1_emission_ratio(r)
                    .unwrap()
                    .with_p2_emission_ratio(r)
                    .unwrap()
                    .with_p1_seepage(0.5)
                    .unwrap()
                    .with_p2_seepage(0.5)
                    .unwrap()
                    .with_p2_angle_scaling(AngleScaling::linear())
                    .with_leakage_scale(scale)
                    .unwrap()
                    .build()
                    .unwrap();
                let operands: Vec<_> = (0..gate.quantum_arity()).map(QubitId).collect();
                let angles = if gate == GateType::RZZ {
                    vec![Angle64::QUARTER_TURN]
                } else {
                    vec![]
                };
                let event = NoiseEvent::AfterGate {
                    gate_type: gate,
                    gate_id: Some(gate.to_gate_id()),
                    qubits: &[],
                    leaked_suppressed: &operands,
                    group_size: std::num::NonZeroUsize::new(gate.quantum_arity()).unwrap(),
                    angles: &angles,
                };
                let mut rng = PecosRng::seed_from_u64(37);
                let mut count = 0;
                for _ in 0..N {
                    noise.context_mut().mark_leaked(QubitId(0));
                    count += u32::from(!unmarked(&noise.emit(&event, &mut rng)).is_empty());
                }
                within_four_sigma(
                    count,
                    N,
                    0.5 * r * 0.5 * if gate == GateType::RZZ { 0.5 } else { 1.0 },
                );
            }
        }
    }
}

#[test]
fn suppressed_single_qubit_seepage() {
    let channel = SingleQubitChannel::depolarizing(1.0)
        .unwrap()
        .with_emission(1.0, 1.0)
        .unwrap()
        .with_seepage(1.0)
        .unwrap();
    let mut ctx = NoiseContext::new();
    ctx.mark_leaked(QubitId(0));
    let event = NoiseEvent::AfterGate {
        gate_type: GateType::X,
        gate_id: Some(GateType::X.to_gate_id()),
        qubits: &[],
        leaked_suppressed: &[QubitId(0)],
        group_size: std::num::NonZeroUsize::new(1).unwrap(),
        angles: &[],
    };
    for use_try in [false, true] {
        let response = if use_try {
            channel
                .try_apply(&event, &mut ctx, &mut PecosRng::seed_from_u64(42))
                .unwrap()
        } else {
            channel.apply(&event, &mut ctx, &mut PecosRng::seed_from_u64(42))
        };
        assert_eq!(unmarked(&response), vec![QubitId(0)]);
        assert_eq!(injected(response).len(), 0);
    }
}

#[test]
fn suppressed_noiseless_channels_do_not_draw() {
    for two in [false, true] {
        let (gate_type, channel): (_, Box<dyn NoiseChannel>) = if two {
            (
                GateType::CX,
                Box::new(
                    TwoQubitChannel::depolarizing(1.0)
                        .unwrap()
                        .with_leakage(1.0)
                        .unwrap()
                        .with_seepage(1.0)
                        .unwrap(),
                ),
            )
        } else {
            (
                GateType::X,
                Box::new(
                    SingleQubitChannel::depolarizing(1.0)
                        .unwrap()
                        .with_emission(1.0, 1.0)
                        .unwrap()
                        .with_seepage(1.0)
                        .unwrap(),
                ),
            )
        };
        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(QubitId(0));
        ctx.add_noiseless_gate(gate_type);
        let operands: Vec<_> = (0..gate_type.quantum_arity()).map(QubitId).collect();
        let event = NoiseEvent::AfterGate {
            gate_type,
            gate_id: None,
            qubits: &[],
            leaked_suppressed: &operands,
            group_size: std::num::NonZeroUsize::new(operands.len()).unwrap(),
            angles: &[],
        };
        for use_try in [false, true] {
            let mut rng = PecosRng::seed_from_u64(17);
            let mut control = rng.clone();
            let response = if use_try {
                channel.try_apply(&event, &mut ctx, &mut rng).unwrap()
            } else {
                channel.apply(&event, &mut ctx, &mut rng)
            };
            assert!(response.is_none());
            assert_eq!(rng.next_u64(), control.next_u64());
        }
    }
}

#[derive(Clone)]
struct GroupProbe {
    calls: std::sync::Arc<std::sync::Mutex<Vec<(usize, QubitId)>>>,
    two_pass: bool,
    suppressed: bool,
}
impl Primitive for GroupProbe {
    fn apply(&self, q: QubitId, ctx: &mut NoiseContext, _rng: &mut PecosRng) -> CompositeResponse {
        use crate::noise::composite::{AnyQubitLeaked, Condition, PartnerLeaked};
        let first = q.0 / 2 * 2;
        assert_eq!(ctx.is_current_gate_suppressed(), self.suppressed && q.0 < 2);
        assert_eq!(ctx.current_gate().unwrap().num_qubits, 2);
        assert_eq!(
            ctx.current_gate_qubits(),
            &[QubitId(first), QubitId(first + 1)]
        );
        assert_eq!(ctx.current_qubit_index(), q.0 % 2);
        assert_eq!(ctx.other_qubit(), Some(QubitId(q.0 ^ 1)));
        assert_eq!(PartnerLeaked.evaluate(q, ctx), q == QubitId(1));
        assert_eq!(AnyQubitLeaked.evaluate(q, ctx), q.0 < 2);
        self.calls.lock().unwrap().push((0, q));
        CompositeResponse::None
    }
    fn needs_two_pass(&self) -> bool {
        self.two_pass
    }
    fn apply_stage1(
        &self,
        q: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        self.apply(q, ctx, rng);
        self.calls.lock().unwrap().push((1, q));
        CompositeResponse::None
    }
    fn apply_stage2(
        &self,
        q: QubitId,
        ctx: &mut NoiseContext,
        rng: &mut PecosRng,
    ) -> CompositeResponse {
        self.apply(q, ctx, rng);
        self.calls.lock().unwrap().push((2, q));
        CompositeResponse::None
    }
    fn describe(&self) -> String {
        "group probe".into()
    }
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

#[test]
fn composite_group_context_all_paths() {
    for batch in [false, true] {
        for geometric in [false, true] {
            for two_pass in [false, true] {
                for suppressed in [false, true] {
                    let calls = std::sync::Arc::new(std::sync::Mutex::new(vec![]));
                    let probe = GroupProbe {
                        calls: std::sync::Arc::clone(&calls),
                        two_pass,
                        suppressed,
                    };
                    let threshold = if geometric { 2.0 } else { 0.0 };
                    let channel: Box<dyn NoiseChannel> = if batch {
                        Box::new(
                            BatchCompositeChannel::new("probe", 0.5, probe)
                                .unwrap()
                                .with_filter(CompositeEventFilter::TwoQubitGate)
                                .with_min_qubits(0)
                                .with_geometric_threshold(threshold),
                        )
                    } else {
                        Box::new(
                            CompositeChannel::new("probe", probe)
                                .with_probability(0.5)
                                .unwrap()
                                .with_filter(CompositeEventFilter::TwoQubitGate)
                                .with_min_qubits(0)
                                .with_geometric_threshold(threshold),
                        )
                    };
                    let qubits: Vec<_> = (0..4).map(QubitId).collect();
                    let event = if suppressed {
                        after(&qubits[2..], &qubits[..2])
                    } else {
                        after(&qubits, &[])
                    };
                    let mut ctx = NoiseContext::new();
                    ctx.mark_leaked(QubitId(0));
                    let mut rng = PecosRng::seed_from_u64(42);
                    for _ in 0..100 {
                        channel.apply(&event, &mut ctx, &mut rng);
                    }
                    assert!(!ctx.is_current_gate_suppressed());
                    let calls = calls.lock().unwrap();
                    for q in [0, 2, 3] {
                        assert!(calls.contains(&(0, QubitId(q))));
                    }
                    assert_eq!(calls.contains(&(0, QubitId(1))), !suppressed);
                    if two_pass {
                        for stage in [1, 2] {
                            assert!(calls.contains(&(stage, QubitId(0))));
                        }
                    }
                }
            }
        }
    }
}

#[derive(Clone, Copy)]
struct FiredGroupProbe;
impl Primitive for FiredGroupProbe {
    fn apply(
        &self,
        _q: QubitId,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        panic!("two-stage primitive requires both stages");
    }
    fn needs_two_pass(&self) -> bool {
        true
    }
    fn apply_stage1(
        &self,
        q: QubitId,
        ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        ctx.set_fired(ctx.current_qubit_index(), q == QubitId(0));
        CompositeResponse::None
    }
    fn apply_stage2(
        &self,
        q: QubitId,
        ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> CompositeResponse {
        if ctx.partner_fired() {
            CompositeResponse::InjectGates(vec![GateCommand::x(q)])
        } else {
            CompositeResponse::None
        }
    }
    fn describe(&self) -> String {
        "fired group probe".into()
    }
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(*self)
    }
}

#[test]
fn composite_two_stage_flags_stay_in_group() {
    for batch in [false, true] {
        for geometric in [false, true] {
            let threshold = if geometric { 2.0 } else { 0.0 };
            let channel: Box<dyn NoiseChannel> = if batch {
                Box::new(
                    BatchCompositeChannel::new("fired", 0.999_999, FiredGroupProbe)
                        .unwrap()
                        .with_geometric_threshold(threshold)
                        .with_min_qubits(0),
                )
            } else {
                Box::new(
                    CompositeChannel::new("fired", FiredGroupProbe)
                        .with_probability(0.999_999)
                        .unwrap()
                        .with_geometric_threshold(threshold)
                        .with_min_qubits(0),
                )
            };
            let gates = injected(channel.apply(
                &after(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)], &[]),
                &mut NoiseContext::new(),
                &mut PecosRng::seed_from_u64(42),
            ));
            assert_eq!(gates.len(), 1, "batch={batch} geometric={geometric}");
            assert_eq!(gates[0].qubits.as_slice(), &[QubitId(1)]);
        }
    }
}

#[test]
fn composite_sampled_correlation_stays_in_group() {
    let mut ctx = NoiseContext::new();
    ctx.set_current_gate(GateType::CX, &[], std::num::NonZeroUsize::new(2).unwrap());
    ctx.set_current_qubit_index(0, &[QubitId(0), QubitId(1)]);
    ctx.set_sampled_correlation(7);
    ctx.set_current_qubit_index(1, &[QubitId(0), QubitId(1)]);
    assert_eq!(ctx.sampled_correlation(), Some(7));
    // Geometric sampling can first visit the second member of the next group.
    ctx.set_current_qubit_index(1, &[QubitId(2), QubitId(3)]);
    assert_eq!(ctx.sampled_correlation(), None);
    // A later event on the same pair may start at its second member.
    ctx.set_current_qubit_index(0, &[QubitId(2), QubitId(3)]);
    ctx.set_sampled_correlation(7);
    ctx.clear_current_gate();
    assert_eq!(ctx.current_gate_qubits(), &[]);
    ctx.set_current_gate(GateType::CX, &[], std::num::NonZeroUsize::new(2).unwrap());
    ctx.set_current_qubit_index(1, &[QubitId(2), QubitId(3)]);
    assert_eq!(ctx.sampled_correlation(), None);
}

#[derive(Clone)]
struct AdapterProbe;
impl NoiseChannel for AdapterProbe {
    fn responds_to(&self, _: &NoiseEvent<'_>) -> bool {
        true
    }
    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        _: &mut NoiseContext,
        _: &mut PecosRng,
    ) -> NoiseResponse {
        let NoiseEvent::AfterGate {
            gate_type,
            qubits,
            group_size,
            leaked_suppressed,
            ..
        } = event
        else {
            panic!("expected synthesized gate");
        };
        assert_eq!(*gate_type, GateType::I);
        assert_eq!(group_size.get(), 2);
        assert_eq!(qubits.len(), 2);
        assert_eq!(*leaked_suppressed, &[]);
        NoiseResponse::inject_gate(GateCommand::x(qubits[0]))
    }
    fn name(&self) -> &'static str {
        "adapter probe"
    }
    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

#[test]
fn channel_adapter_preserves_custom_group_size() {
    let channel = CompositeChannel::new("adapter", channel_action(AdapterProbe));
    let event = NoiseEvent::AfterGate {
        gate_type: GateType::I,
        gate_id: None,
        qubits: &[QubitId(0), QubitId(1), QubitId(2), QubitId(3)],
        leaked_suppressed: &[],
        group_size: std::num::NonZeroUsize::new(2).unwrap(),
        angles: &[],
    };
    let mut gates = injected(channel.apply(
        &event,
        &mut NoiseContext::new(),
        &mut PecosRng::seed_from_u64(42),
    ));
    // Each group reaches the probe whole; how often it is repeated is not the contract here.
    gates.dedup();
    assert_eq!(
        gates,
        vec![GateCommand::x(QubitId(0)), GateCommand::x(QubitId(2))]
    );
}

/// Records what a composite nested through `channel_action` sees for each member.
#[derive(Clone, Default)]
struct NestedProbe(std::sync::Arc<std::sync::Mutex<Vec<NestedView>>>);
type NestedView = (QubitId, bool, GateType, Vec<QubitId>);
impl Primitive for NestedProbe {
    fn apply(&self, q: QubitId, ctx: &mut NoiseContext, _: &mut PecosRng) -> CompositeResponse {
        self.0.lock().unwrap().push((
            q,
            ctx.is_current_gate_suppressed(),
            ctx.current_gate().expect("adapter emits a gate").gate_type,
            ctx.current_gate_qubits().to_vec(),
        ));
        CompositeResponse::None
    }
    fn describe(&self) -> String {
        "nested probe".into()
    }
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

fn nested_channels(probe: &NestedProbe) -> Vec<Box<dyn NoiseChannel>> {
    let inner = CompositeChannel::new("inner", probe.clone());
    let batch_inner = BatchCompositeChannel::new("batch inner", 0.999_999, probe.clone())
        .unwrap()
        .with_min_qubits(0)
        .with_geometric_threshold(0.0);
    vec![
        Box::new(CompositeChannel::new(
            "outer",
            channel_action(inner.clone()),
        )),
        Box::new(CompositeChannel::new("outer", channel_action(batch_inner))),
        Box::new(
            BatchCompositeChannel::new("outer", 0.999_999, channel_action(inner))
                .unwrap()
                .with_min_qubits(0)
                .with_geometric_threshold(0.0),
        ),
    ]
}

#[test]
fn nested_composite_keeps_the_enclosing_gate_for_every_member() {
    let pair = [QubitId(0), QubitId(1)];
    for suppressed in [false, true] {
        let probe = NestedProbe::default();
        for channel in nested_channels(&probe) {
            probe.0.lock().unwrap().clear();
            let mut ctx = NoiseContext::new();
            let mut rng = PecosRng::seed_from_u64(42);
            let event = if suppressed {
                ctx.mark_leaked(pair[0]);
                ctx.mark_leaked(pair[1]);
                after(&[], &pair)
            } else {
                after(&pair, &[])
            };
            channel.apply(&event, &mut ctx, &mut rng);
            let views = probe.0.lock().unwrap().clone();
            assert_ne!(views, Vec::<NestedView>::new());
            for (_, seen_suppressed, gate, qubits) in views {
                assert_eq!(seen_suppressed, suppressed);
                assert_eq!(gate, GateType::CX);
                assert_eq!(qubits, pair);
            }
            assert_eq!(ctx.current_gate_qubits(), &[]);
            assert!(ctx.current_gate().is_none());
        }
    }
}

#[test]
fn nested_composite_keeps_non_gate_operands_for_every_member() {
    let qubits = [QubitId(0), QubitId(1), QubitId(2)];
    let probe = NestedProbe::default();
    for channel in nested_channels(&probe) {
        probe.0.lock().unwrap().clear();
        channel.apply(
            &NoiseEvent::AfterPreparation {
                gate_type: GateType::PZ,
                qubits: &qubits,
            },
            &mut NoiseContext::new(),
            &mut PecosRng::seed_from_u64(42),
        );
        let views = probe.0.lock().unwrap().clone();
        assert_ne!(views, Vec::<NestedView>::new());
        for (_, _, _, seen) in views {
            assert_eq!(seen, qubits);
        }
    }
}

#[test]
fn nested_composite_restores_all_enclosing_event_state() {
    let pair = [QubitId(0), QubitId(1)];
    let inner = CompositeChannel::new("inner", nothing());
    let batch_inner = BatchCompositeChannel::new("batch inner", 0.999_999, nothing())
        .unwrap()
        .with_min_qubits(0)
        .with_geometric_threshold(0.0);
    for adapter in [channel_action(inner), channel_action(batch_inner)] {
        let mut ctx = NoiseContext::new();
        ctx.mark_leaked(pair[1]);
        ctx.set_current_gate(GateType::CX, &[], std::num::NonZeroUsize::new(2).unwrap());
        ctx.set_current_idle(pecos_core::TimeUnits::new(5));
        // The inner channel ends on member 1, so restoring member 0 is observable.
        ctx.set_current_qubit_index(0, &pair);
        ctx.set_current_gate_suppressed(true);
        ctx.set_sampled_correlation(7);
        ctx.set_fired(0, true);
        Primitive::apply(
            &adapter,
            pair[1],
            &mut ctx,
            &mut PecosRng::seed_from_u64(42),
        );
        assert_eq!(ctx.current_gate().map(|g| g.gate_type), Some(GateType::CX));
        assert!(ctx.current_idle().is_some());
        assert_eq!(ctx.current_qubit_index(), 0);
        assert_eq!(ctx.current_gate_qubits(), pair);
        assert!(ctx.is_current_gate_suppressed());
        assert_eq!(ctx.sampled_correlation(), Some(7));
        assert!(ctx.is_fired(0));
    }
}

#[derive(Clone)]
struct SelectionProbe(std::sync::Arc<std::sync::atomic::AtomicUsize>);
impl Primitive for SelectionProbe {
    fn describe(&self) -> String {
        "selection probe".into()
    }
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
    fn apply(&self, _: QubitId, _: &mut NoiseContext, _: &mut PecosRng) -> CompositeResponse {
        CompositeResponse::None
    }
    fn needs_two_pass(&self) -> bool {
        self.0.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        true
    }
}

#[test]
fn composite_skips_empty_selections() {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    for batch in [false, true] {
        for geometric in [false, true] {
            let calls = Arc::new(AtomicUsize::new(0));
            let probe = SelectionProbe(Arc::clone(&calls));
            let threshold = if geometric { 2.0 } else { 0.0 };
            let channel: Box<dyn NoiseChannel> = if batch {
                Box::new(
                    BatchCompositeChannel::new("empty", 0.5, probe)
                        .unwrap()
                        .with_min_qubits(0)
                        .with_geometric_threshold(threshold),
                )
            } else {
                Box::new(
                    CompositeChannel::new("empty", probe)
                        .with_probability(0.5)
                        .unwrap()
                        .with_min_qubits(0)
                        .with_geometric_threshold(threshold),
                )
            };
            let mut ctx = NoiseContext::new();
            ctx.set_current_gate(GateType::H, &[], std::num::NonZeroUsize::MIN);
            ctx.set_fired(0, true);
            let mut rng = PecosRng::seed_from_u64(42);
            let mut control_rng = rng.clone();
            for event in [
                after(&[], &[]),
                NoiseEvent::BeforeMeasurement {
                    gate_type: GateType::MZ,
                    qubits: &[],
                },
            ] {
                assert!(matches!(
                    channel.apply(&event, &mut ctx, &mut rng),
                    NoiseResponse::None
                ));
                assert_eq!(ctx.current_gate().unwrap().gate_type, GateType::H);
                assert!(ctx.is_fired(0));
                assert_eq!(calls.load(Ordering::SeqCst), 0);
            }
            assert_eq!(rng.next_u64(), control_rng.next_u64());
            let pair = [QubitId(0), QubitId(1)];
            ctx.mark_leaked(pair[0]);
            for event in [after(&pair, &[]), after(&[], &pair)] {
                calls.store(0, Ordering::SeqCst);
                channel.apply(&event, &mut ctx, &mut rng);
                assert_eq!(calls.load(Ordering::SeqCst), 1);
            }
        }
    }
}

#[test]
fn angle_scaled_zero_emission_preserves_leaked_branch() {
    for suppressed in [false, true] {
        let pair = [QubitId(0), QubitId(1)];
        let event = NoiseEvent::AfterGate {
            gate_type: GateType::RZZ,
            gate_id: None,
            qubits: if suppressed { &[] } else { &pair },
            leaked_suppressed: if suppressed { &pair } else { &[] },
            group_size: gate_group_size(2, pair.len()),
            angles: &[Angle64::HALF_TURN],
        };
        for scaled in [false, true] {
            let builder = CompositeNoiseModelBuilder::new()
                .with_p2(1.0)
                .unwrap()
                .with_p2_emission_ratio(0.0)
                .unwrap()
                .with_p2_seepage(1.0)
                .unwrap();
            let mut noise = if scaled {
                builder.with_p2_angle_scaling(AngleScaling::linear())
            } else {
                builder
            }
            .build()
            .unwrap();
            // Both members leaked: neither branch may inject depolarizing noise.
            noise.context_mut().mark_leaked(pair[0]);
            noise.context_mut().mark_leaked(pair[1]);
            let response = noise.emit(&event, &mut PecosRng::seed_from_u64(7));
            assert!(
                injected(response.clone()).is_empty(),
                "scaled={scaled}, suppressed={suppressed}"
            );
            assert_eq!(unmarked(&response), Vec::<QubitId>::new());
        }
    }
}

#[test]
fn two_qubit_handles_complete_pairs_only() {
    let channel = TwoQubitChannel::depolarizing(1.0)
        .unwrap()
        .with_leakage(1.0)
        .unwrap()
        .with_seepage(1.0)
        .unwrap();
    let mut weights = [0.0; 15];
    weights[14] = 1.0;
    let gate_error = TwoQubitChannel::depolarizing(1.0)
        .unwrap()
        .with_pauli_weights(TwoQubitPauliWeights::custom(weights).unwrap())
        .unwrap();
    for len in 0..=5 {
        let qubits: Vec<_> = (0..len).map(QubitId).collect();
        for suppressed in [false, true] {
            for try_apply in [false, true] {
                let event = if suppressed {
                    after(&[], &qubits)
                } else {
                    after(&qubits, &[])
                };
                let mut ctx = NoiseContext::new();
                for &q in &qubits {
                    ctx.mark_leaked(q);
                }
                let mut rng = PecosRng::seed_from_u64(42);
                let response = if try_apply {
                    channel.try_apply(&event, &mut ctx, &mut rng).unwrap()
                } else {
                    channel.apply(&event, &mut ctx, &mut rng)
                };
                assert_eq!(unmarked(&response), qubits[..len / 2 * 2]);
                assert_only_seep_injections(&response);
                let mut healthy = NoiseContext::new();
                let response = if try_apply {
                    gate_error
                        .try_apply(&event, &mut healthy, &mut rng)
                        .unwrap()
                } else {
                    gate_error.apply(&event, &mut healthy, &mut rng)
                };
                if suppressed {
                    assert_only_seep_injections(&response);
                }
                let targets: Vec<_> = injected(response)
                    .into_iter()
                    .flat_map(|g| g.qubits.into_vec())
                    .collect();
                assert_eq!(
                    targets,
                    if suppressed {
                        vec![]
                    } else {
                        qubits[..len / 2 * 2].to_vec()
                    }
                );
            }
        }
    }
}

#[test]
fn composite_linear_two_stage_seeded_draw_order() {
    let channel = CompositeChannel::new(
        "seeded stages",
        two_stage(prob(0.43, pauli()).unwrap(), prob(0.61, pauli()).unwrap()),
    )
    .with_probability(0.57)
    .unwrap()
    .with_geometric_threshold(0.0);
    let observe = |event: &NoiseEvent<'_>| -> Vec<(u64, u64)> {
        [0, 1, 7, 42, 100]
            .map(|seed| {
                let mut rng = PecosRng::seed_from_u64(seed);
                let gates = injected(channel.apply(event, &mut NoiseContext::new(), &mut rng));
                let signature = gates.iter().fold(0_u64, |signature, gate| {
                    let pauli = match gate.gate_type {
                        GateType::X => 1,
                        GateType::Y => 2,
                        GateType::Z => 3,
                        _ => panic!("expected a Pauli fault"),
                    };
                    (signature << 4) | (4 * u64::try_from(gate.qubits[0].0).unwrap() + pauli)
                });
                (signature, rng.next_u64())
            })
            .to_vec()
    };
    let qubits = [QubitId(0), QubitId(1), QubitId(2), QubitId(3)];
    // Captured before per-group processing: one group keeps its seeded results.
    assert_eq!(
        observe(&after(&qubits[..2], &[])),
        vec![
            (3, 17_356_855_043_271_732),
            (0, 5_295_051_070_016_473_251),
            (7, 5_768_340_367_737_899_501),
            (103, 3_088_116_801_777_502_571),
            (3, 6_666_168_914_398_614_217),
        ]
    );
    assert_eq!(
        observe(&NoiseEvent::AfterMeasurement {
            gate_type: GateType::MZ,
            qubits: &qubits,
            outcomes: &[true, false, true, false],
        }),
        vec![
            (1437, 16_722_455_084_393_394_041),
            (633, 12_783_689_275_883_468_267),
            (41, 10_898_401_604_247_839_105),
            (1575, 11_803_152_480_074_679_081),
            (382, 17_518_749_168_294_015_972),
        ]
    );
    // Pins the intended change: each group finishes both stages before the next group starts.
    assert_eq!(
        observe(&after(&qubits, &[])),
        vec![
            (941, 1_213_719_370_718_833_293),
            (9, 12_783_689_275_883_468_267),
            (121, 10_898_401_604_247_839_105),
            (103, 11_803_152_480_074_679_081),
            (942, 17_518_749_168_294_015_972),
        ]
    );
}

#[derive(Clone)]
struct EventCorrelationProbe(std::sync::Arc<std::sync::atomic::AtomicBool>);
impl Primitive for EventCorrelationProbe {
    fn apply(&self, q: QubitId, ctx: &mut NoiseContext, _: &mut PecosRng) -> CompositeResponse {
        if q == QubitId(0) && self.0.load(std::sync::atomic::Ordering::SeqCst) {
            ctx.set_sampled_correlation(7);
        }
        if q == QubitId(1) && ctx.sampled_correlation() == Some(7) {
            CompositeResponse::InjectGates(vec![GateCommand::x(q)])
        } else {
            CompositeResponse::None
        }
    }
    fn describe(&self) -> String {
        "event correlation probe".into()
    }
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

#[test]
fn composite_correlation_clears_between_repeated_pairs() {
    use std::sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    };
    for batch in [false, true] {
        for geometric in [false, true] {
            let sets_correlation = Arc::new(AtomicBool::new(true));
            let probe = EventCorrelationProbe(Arc::clone(&sets_correlation));
            let threshold = if geometric { 2.0 } else { 0.0 };
            let channel: Box<dyn NoiseChannel> = if batch {
                Box::new(
                    BatchCompositeChannel::new("correlation", 0.999_999, probe)
                        .unwrap()
                        .with_min_qubits(0)
                        .with_geometric_threshold(threshold),
                )
            } else {
                Box::new(
                    CompositeChannel::new("correlation", probe)
                        .with_probability(0.999_999)
                        .unwrap()
                        .with_min_qubits(0)
                        .with_geometric_threshold(threshold),
                )
            };
            let pair = [QubitId(0), QubitId(1)];
            let event = after(&pair, &[]);
            let mut ctx = NoiseContext::new();
            let mut rng = PecosRng::seed_from_u64(42);
            assert_eq!(injected(channel.apply(&event, &mut ctx, &mut rng)).len(), 1);
            // The same pair repeats, but this event does not reach the correlation setter.
            sets_correlation.store(false, Ordering::SeqCst);
            assert_eq!(
                injected(channel.apply(&event, &mut ctx, &mut rng)),
                Vec::<GateCommand>::new()
            );
            assert_eq!(ctx.current_gate_qubits(), &[]);
            assert_eq!(ctx.sampled_correlation(), None);
        }
    }
}

#[test]
fn composite_stage_two_runs_on_unsampled_groups() {
    for batch in [false, true] {
        for geometric in [false, true] {
            let primitive = two_stage(inject_x(), inject_z());
            let threshold = if geometric { 2.0 } else { 0.0 };
            let channel: Box<dyn NoiseChannel> = if batch {
                Box::new(
                    BatchCompositeChannel::new("stage two", 0.000_001, primitive)
                        .unwrap()
                        .with_min_qubits(0)
                        .with_geometric_threshold(threshold),
                )
            } else {
                Box::new(
                    CompositeChannel::new("stage two", primitive)
                        .with_probability(0.000_001)
                        .unwrap()
                        .with_min_qubits(0)
                        .with_geometric_threshold(threshold),
                )
            };
            let qubits = [QubitId(0), QubitId(1), QubitId(2), QubitId(3)];
            let gates = injected(channel.apply(
                &after(&qubits, &[]),
                &mut NoiseContext::new(),
                &mut PecosRng::seed_from_u64(42),
            ));
            assert_eq!(gates, qubits.map(GateCommand::z));
        }
    }
}

#[derive(Clone, Default)]
struct MemberIndexProbe(std::sync::Arc<std::sync::Mutex<Vec<usize>>>);
impl Primitive for MemberIndexProbe {
    fn apply(&self, _: QubitId, ctx: &mut NoiseContext, _: &mut PecosRng) -> CompositeResponse {
        self.0.lock().unwrap().push(ctx.current_qubit_index());
        CompositeResponse::None
    }
    fn describe(&self) -> String {
        "member index probe".into()
    }
    fn clone_box(&self) -> Box<dyn Primitive> {
        Box::new(self.clone())
    }
}

#[test]
fn composite_non_gate_context_linear_and_two_pass() {
    fn make_channel(
        primitive: impl Primitive + Clone + 'static,
        batch: bool,
        threshold: f64,
    ) -> Box<dyn NoiseChannel> {
        if batch {
            Box::new(
                BatchCompositeChannel::new("non gate", 0.999_999, primitive)
                    .unwrap()
                    .with_min_qubits(0)
                    .with_geometric_threshold(threshold),
            )
        } else {
            Box::new(
                CompositeChannel::new("non gate", primitive)
                    .with_probability(0.999_999)
                    .unwrap()
                    .with_min_qubits(0)
                    .with_geometric_threshold(threshold),
            )
        }
    }
    for batch in [false, true] {
        for two_pass in [false, true] {
            for geometric in [false, true] {
                let primitive = when(any_qubit_leaked(), inject_x(), nothing());
                let threshold = if geometric { 2.0 } else { 0.0 };
                let channel = if two_pass {
                    make_channel(two_stage(nothing(), primitive), batch, threshold)
                } else {
                    make_channel(primitive, batch, threshold)
                };
                let qubits = [QubitId(0), QubitId(1), QubitId(2)];
                let mut ctx = NoiseContext::new();
                ctx.mark_leaked(qubits[0]);
                let gates = injected(channel.apply(
                    &NoiseEvent::AfterPreparation {
                        gate_type: GateType::PZ,
                        qubits: &qubits,
                    },
                    &mut ctx,
                    &mut PecosRng::seed_from_u64(42),
                ));
                if two_pass || !geometric {
                    assert_eq!(gates, qubits.map(GateCommand::x));
                    let probe = MemberIndexProbe::default();
                    let channel = if two_pass {
                        make_channel(two_stage(nothing(), probe.clone()), batch, threshold)
                    } else {
                        make_channel(probe.clone(), batch, threshold)
                    };
                    channel.apply(
                        &NoiseEvent::AfterPreparation {
                            gate_type: GateType::PZ,
                            qubits: &qubits,
                        },
                        &mut ctx,
                        &mut PecosRng::seed_from_u64(42),
                    );
                    assert_eq!(*probe.0.lock().unwrap(), [0, 1, 2]);
                } else {
                    // Single-pass geometric processing retains its existing non-gate context policy.
                    assert_eq!(gates, Vec::<GateCommand>::new());
                }
            }
        }
    }
}
