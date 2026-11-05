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

//! Circuit and channel regressions for #1573, #1574 and #1575.
use pecos_core::{QubitId, TimeUnits};
use pecos_neo::command::{CommandBuilder, CommandQueue, GateCommand, GateType};
use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
use pecos_neo::noise::composite::prelude::*;
use pecos_neo::noise::plugins::CorePlugin;
use pecos_neo::noise::{
    ComposableNoiseModel, GeneralNoiseModelBuilder, LeakageChannel, MeasurementChannel,
    MeasurementStateFlipChannel, NoiseChannel, NoiseContext, NoiseEvent, NoiseModelBuilder,
    NoiseResponse, PerGatePauliChannel, PreparationChannel, SingleQubitChannel, TwoQubitChannel,
};
use pecos_neo::outcome::MeasurementOutcomes;
use pecos_neo::runner::CircuitRunner;
use pecos_random::PecosRng;
use pecos_simulators::SparseStab;
use rand::RngExt;
use smallvec::smallvec;
use std::collections::BTreeMap;

fn run(model: ComposableNoiseModel, circuit: &CommandQueue, seed: u64) -> MeasurementOutcomes {
    CircuitRunner::<SparseStab>::new()
        .with_noise(model)
        .with_seed(seed)
        .apply_circuit(&mut SparseStab::with_seed(3, seed), circuit)
        .unwrap()
}

#[derive(Clone)]
struct LeakAfterZ;

impl NoiseChannel for LeakAfterZ {
    fn responds_to(&self, event: &NoiseEvent<'_>) -> bool {
        matches!(
            event,
            NoiseEvent::AfterGate {
                gate_type: GateType::Z,
                ..
            }
        )
    }

    fn apply(
        &self,
        event: &NoiseEvent<'_>,
        _ctx: &mut NoiseContext,
        _rng: &mut PecosRng,
    ) -> NoiseResponse {
        // Intentionally independent of the noisy-gate selection: only X/CZ
        // receives gate-channel faults in the issue's circuits.
        NoiseResponse::MarkLeaked(event.qubits().iter().copied().collect())
    }

    fn name(&self) -> &'static str {
        "leak_after_z"
    }
    fn clone_box(&self) -> Box<dyn NoiseChannel> {
        Box::new(self.clone())
    }
}

#[derive(Clone, Copy, Debug)]
enum Builder {
    General,
    Unified,
    Composite,
}
const BUILDERS: [Builder; 3] = [Builder::General, Builder::Unified, Builder::Composite];

fn model(
    kind: Builder,
    p1: f64,
    p2: f64,
    emission: f64,
    noiseless: &[GateType],
) -> ComposableNoiseModel {
    match kind {
        Builder::General => GeneralNoiseModelBuilder::new()
            .with_p1(p1)
            .with_p2(p2)
            .with_p2_emission_ratio(emission)
            .with_noiseless_gates(noiseless)
            .build()
            .unwrap(),
        Builder::Unified => NoiseModelBuilder::new()
            .with_p1(p1)
            .with_p2(p2)
            .with_p2_emission_ratio(emission)
            .with_noiseless_gates(noiseless.iter().copied())
            .build()
            .unwrap(),
        Builder::Composite => CompositeNoiseModelBuilder::new()
            .with_p1(p1)
            .unwrap()
            .with_p2(p2)
            .unwrap()
            .with_p2_emission_ratio(emission)
            .unwrap()
            .with_noiseless_gates(noiseless)
            .build()
            .unwrap(),
    }
}

fn ones(model: &ComposableNoiseModel, circuit: &CommandQueue, q: usize) -> usize {
    (0..1000)
        .filter(|&seed| {
            run(model.clone(), circuit, seed)
                .get_bit(QubitId(q))
                .unwrap()
        })
        .count()
}

#[test]
fn noiseless_gate_faults_all_builders() {
    let x = CommandBuilder::new()
        .pz(&[0, 1])
        .x(&[1])
        .x(&[1])
        .mz(&[1])
        .build();
    let cx = CommandBuilder::new()
        .pz(&[0, 1])
        .cx(&[(0, 1)])
        .mz(&[1])
        .build();
    for kind in BUILDERS {
        for (gate, circuit, p1, p2, minimum) in [
            (GateType::X, &x, 0.5, 0.0, 300),
            (GateType::CX, &cx, 0.0, 0.5, 150),
        ] {
            assert_eq!(
                ones(&model(kind, p1, p2, 0.0, &[gate]), circuit, 1),
                0,
                "{kind:?} {gate:?}"
            );
            assert!(
                ones(&model(kind, p1, p2, 0.0, &[]), circuit, 1) > minimum,
                "{kind:?} {gate:?} control"
            );
        }
    }
}

fn prep_readout(composite: bool, prep: bool) -> ComposableNoiseModel {
    let p = if prep { 0.5 } else { 0.0 };
    let m = if prep { 0.0 } else { 0.5 };
    if composite {
        CompositeNoiseModelBuilder::new()
            .with_p_prep(p)
            .unwrap()
            .with_p_meas(m, m)
            .unwrap()
            .build()
            .unwrap()
    } else {
        GeneralNoiseModelBuilder::new()
            .with_p_prep(p)
            .with_p_meas(m, m)
            .build()
            .unwrap()
    }
}

#[test]
fn preparation_and_measurement_use_their_actual_gate_types() {
    let ordinary = CommandBuilder::new().pz(&[0]).mz(&[0]).build();
    let alloc = CommandBuilder::new().qalloc(&[0]).mz(&[0]).build();
    let leaked_measure = CommandBuilder::new()
        .pz(&[0])
        .gate(GateCommand::new(
            GateType::MeasureLeaked,
            smallvec![QubitId(0)],
        ))
        .build();
    for composite in [false, true] {
        for prep in [false, true] {
            let gate = if prep { GateType::PZ } else { GateType::MZ };
            let control = prep_readout(composite, prep);
            let quiet = control.clone().with_noiseless_gate(gate);
            assert_eq!(ones(&quiet, &ordinary, 0), 0);
            assert!(ones(&control, &ordinary, 0) > 300);
            let distinct = if prep { &alloc } else { &leaked_measure };
            assert!(ones(&quiet, distinct, 0) > 300);
            assert!(ones(&control, distinct, 0) > 300);
        }
    }
}

#[test]
fn composite_channels_directly_honour_noiseless() {
    let circuit = CommandBuilder::new()
        .pz(&[0, 1])
        .x(&[1])
        .x(&[1])
        .mz(&[1])
        .build();
    let channels: [Box<dyn NoiseChannel>; 2] = [
        Box::new(CompositeChannelBuilder::single_qubit(
            "direct",
            prob(0.5, pauli()).unwrap(),
        )),
        Box::new(CompositeChannelBuilder::batch_single_qubit("batch", 0.5, pauli()).unwrap()),
    ];
    for channel in channels {
        let control = ComposableNoiseModel::new()
            .add_plugin(&CorePlugin)
            .add_boxed_channel(channel);
        assert!(ones(&control, &circuit, 1) > 300);
        assert_eq!(
            ones(&control.with_noiseless_gate(GateType::X), &circuit, 1),
            0
        );
    }
    let mut model = ComposableNoiseModel::new()
        .with_noiseless_gate(GateType::X)
        .add_channel(CompositeChannelBuilder::before_gate(
            "skip",
            skip_if_leaked(),
        ));
    model.context_mut().mark_leaked(QubitId(0));
    let response = model.emit(
        &NoiseEvent::before_gate(GateType::X, &[QubitId(0)], &[]),
        &mut PecosRng::seed_from_u64(0),
    );
    assert_eq!(response.skipped_qubits().as_slice(), &[QubitId(0)]);
}

#[test]
fn noiseless_preparation_still_clears_leakage() {
    let circuit = CommandBuilder::new()
        .pz(&[0])
        .z(&[0])
        .pz(&[0])
        .mz(&[0])
        .build();
    let control = CommandBuilder::new().pz(&[0]).z(&[0]).mz(&[0]).build();
    for kind in BUILDERS {
        let model = model(kind, 0.0, 0.0, 0.0, &[GateType::PZ]).add_channel(LeakAfterZ);
        assert_eq!(ones(&model, &circuit, 0), 0, "{kind:?}");
        assert_eq!(ones(&model, &control, 0), 1000, "{kind:?} control");
    }
}

#[test]
fn noiseless_leaked_readout_keeps_its_result() {
    for gate in [GateType::MZ, GateType::MeasureLeaked] {
        let circuit = CommandBuilder::new()
            .pz(&[0])
            .z(&[0])
            .gate(GateCommand::new(gate, smallvec![QubitId(0)]))
            .build();
        let mut models: Vec<_> = BUILDERS
            .into_iter()
            .map(|kind| model(kind, 0.0, 0.0, 0.0, &[gate]))
            .collect();
        for handling in [false, true] {
            models.push(
                CompositeNoiseModelBuilder::new()
                    .with_measure_leaked_handling(handling)
                    .with_noiseless_gate(gate)
                    .build()
                    .unwrap(),
            );
        }
        for model in models {
            let model = model.add_channel(LeakAfterZ);
            for seed in 0..1000 {
                let result = run(model.clone(), &circuit, seed);
                let outcome = result.get(QubitId(0)).unwrap();
                if gate == GateType::MeasureLeaked {
                    assert_eq!(outcome.as_int_leaked(), 2);
                } else {
                    assert!(outcome.outcome);
                }
            }
        }
    }
}

#[test]
fn noiseless_operations_are_not_crosstalk_sources() {
    for composite in [false, true] {
        for prep in [false, true] {
            let control = match (composite, prep) {
                (false, true) => GeneralNoiseModelBuilder::new()
                    .with_p_prep_crosstalk(1.0)
                    .build()
                    .unwrap(),
                (false, false) => GeneralNoiseModelBuilder::new()
                    .with_p_meas_crosstalk(1.0, 0.0)
                    .build()
                    .unwrap(),
                (true, true) => CompositeNoiseModelBuilder::new()
                    .with_p_prep_crosstalk(1.0)
                    .unwrap()
                    .build()
                    .unwrap(),
                (true, false) => CompositeNoiseModelBuilder::new()
                    .with_p_meas_crosstalk(1.0, 0.0)
                    .unwrap()
                    .build()
                    .unwrap(),
            };
            let start = CommandBuilder::new().pz(&[0, 1]).h(&[1]);
            let circuit = if prep { start.pz(&[0]) } else { start.mz(&[0]) }
                .h(&[1])
                .mz(&[1])
                .build();
            assert!((421..=579).contains(&ones(&control, &circuit, 1)));
            let quiet = control.with_noiseless_gate(if prep { GateType::PZ } else { GateType::MZ });
            assert_eq!(ones(&quiet, &circuit, 1), 0);
        }
    }
}

#[test]
fn noiseless_cx_suppresses_its_idle_but_not_scheduled_idle() {
    for composite in [false, true] {
        let control = if composite {
            CompositeNoiseModelBuilder::new()
                .with_p_idle_linear(1.0)
                .unwrap()
                .with_idle_after_2q(1.0)
                .unwrap()
                .build()
                .unwrap()
        } else {
            GeneralNoiseModelBuilder::new()
                .with_p_idle_linear(1.0, &BTreeMap::from([("X".to_owned(), 1.0)]))
                .unwrap()
                .with_idle_after_2q(1.0)
                .unwrap()
                .build()
                .unwrap()
        };
        let circuit = CommandBuilder::new()
            .pz(&[0, 1])
            .cx(&[(0, 1)])
            .mz(&[1])
            .build();
        let quiet = control.clone().with_noiseless_gate(GateType::CX);
        assert_eq!(ones(&quiet, &circuit, 1), 0);
        let count = ones(&control, &circuit, 1);
        if composite {
            assert!((600..=730).contains(&count), "{count}");
        } else {
            assert_eq!(count, 1000);
        }
        let scheduled = CommandBuilder::new()
            .pz(&[0, 1])
            .cx(&[(0, 1)])
            .idle(&[1], TimeUnits::new(1))
            .mz(&[1])
            .build();
        let count = ones(&quiet, &scheduled, 1);
        if composite {
            assert!((600..=730).contains(&count), "{count}");
        } else {
            assert_eq!(count, 1000);
        }
    }
}

fn leaked_pair(control_one: bool) -> CommandQueue {
    let circuit = CommandBuilder::new().pz(&[0, 1]);
    let circuit = if control_one {
        circuit.x(&[0])
    } else {
        circuit
    };
    circuit.z(&[0]).cx(&[(0, 1)]).mz(&[1]).build()
}

#[test]
fn leaked_gates_never_run_in_any_builder() {
    for kind in BUILDERS {
        for control in [false, true] {
            for quiet in [false, true] {
                for (p2, emission) in [(0.5, 1.0), (0.0, 0.0)] {
                    let noiseless = if quiet { vec![GateType::CX] } else { vec![] };
                    let model = model(kind, 0.0, p2, emission, &noiseless).add_channel(LeakAfterZ);
                    assert_eq!(
                        ones(&model, &leaked_pair(control), 1),
                        0,
                        "{kind:?} control={control} quiet={quiet} p2={p2}"
                    );
                }
            }
        }
    }
}

#[test]
fn bare_gate_channels_suppress_noiseless_leaked_operands_on_both_paths() {
    for pair in [false, true] {
        let channel: Box<dyn NoiseChannel> = if pair {
            Box::new(TwoQubitChannel::depolarizing(0.5).unwrap())
        } else {
            Box::new(SingleQubitChannel::depolarizing(0.5).unwrap())
        };
        let gate = if pair { GateType::CX } else { GateType::X };
        let qubits = if pair {
            vec![QubitId(0), QubitId(1)]
        } else {
            vec![QubitId(0)]
        };
        let event = NoiseEvent::before_gate(gate, &qubits, &[]);
        let mut model = ComposableNoiseModel::new()
            .with_noiseless_gate(gate)
            .add_boxed_channel(channel.clone_box());
        model.context_mut().mark_leaked(QubitId(0));
        let mut rng = PecosRng::seed_from_u64(0);
        assert_eq!(
            model.emit(&event, &mut rng).skipped_qubits().as_slice(),
            &[QubitId(0)]
        );
        for use_try in [false, true] {
            let response = if use_try {
                channel
                    .try_apply(&event, model.context_mut(), &mut rng)
                    .unwrap()
            } else {
                channel.apply(&event, model.context_mut(), &mut rng)
            };
            assert_eq!(response.skipped_qubits().as_slice(), &[QubitId(0)]);
        }
    }
}

#[test]
fn partner_pauli_is_opt_in_and_respects_noiseless() {
    for (p, bounds) in [(0.5, 270..=400), (1.0, 600..=730)] {
        for control in [false, true] {
            let model = ComposableNoiseModel::new()
                .add_plugin(&CorePlugin)
                .add_channel(LeakAfterZ)
                .add_channel(
                    LeakageChannel::new()
                        .with_partner_pauli_probability(p)
                        .unwrap(),
                );
            let count = ones(&model, &leaked_pair(control), 1);
            assert!(bounds.contains(&count), "p={p} control={control}: {count}");
            assert_eq!(
                ones(
                    &model.with_noiseless_gate(GateType::CX),
                    &leaked_pair(control),
                    1
                ),
                0
            );
        }
    }
    for p in [-0.1, 1.1, f64::NAN, f64::INFINITY] {
        assert!(
            LeakageChannel::new()
                .with_partner_pauli_probability(p)
                .is_err()
        );
    }
}

#[test]
fn default_partner_noise_consumes_no_rng() {
    let mut ctx = NoiseContext::new();
    ctx.mark_leaked(QubitId(0));
    let event = NoiseEvent::after_gate(GateType::CX, &[QubitId(0), QubitId(1)], &[]);
    for channel in [LeakageChannel::new(), LeakageChannel::no_leakage()] {
        assert!(!channel.responds_to(&event));
        assert!(
            !channel
                .event_kinds()
                .contains(pecos_neo::noise::NoiseEventKind::AfterGate)
        );
        let mut rng = PecosRng::seed_from_u64(42);
        let mut expected = rng.clone();
        assert!(channel.apply(&event, &mut ctx, &mut rng).is_none());
        assert_eq!(rng.random::<u64>(), expected.random::<u64>());
    }
    ctx.add_noiseless_gate(GateType::CX);
    let mut rng = PecosRng::seed_from_u64(42);
    let mut expected = rng.clone();
    assert!(
        LeakageChannel::new()
            .with_partner_pauli_probability(1.0)
            .unwrap()
            .apply(&event, &mut ctx, &mut rng)
            .is_none()
    );
    assert_eq!(rng.random::<u64>(), expected.random::<u64>());
}

#[test]
fn leakage_does_not_flip_an_earlier_measurement() {
    let circuit = CommandBuilder::new()
        .pz(&[0])
        .x(&[0])
        .mz(&[0])
        .z(&[0])
        .mz(&[0])
        .build();
    for kind in BUILDERS {
        let model = model(kind, 0.0, 0.5, 1.0, &[]).add_channel(LeakAfterZ);
        for seed in 0..1000 {
            let outcomes = run(model.clone(), &circuit, seed);
            let bits: Vec<_> = outcomes
                .get_all(QubitId(0))
                .iter()
                .map(|o| o.outcome)
                .collect();
            assert_eq!(bits, [true, true], "{kind:?} seed={seed}");
        }
    }
}

#[test]
fn always_registered_suppression_preserves_healthy_gate_rates_and_rng() {
    let circuit = CommandBuilder::new()
        .pz(&[0, 1])
        .x(&[1])
        .x(&[1])
        .mz(&[1])
        .build();
    let general = model(Builder::General, 0.5, 0.0, 0.0, &[]);
    let bare = ComposableNoiseModel::new()
        .add_plugin(&CorePlugin)
        .add_channel(SingleQubitChannel::depolarizing(0.5).unwrap());
    let count = ones(&general, &circuit, 1);
    assert!((400..=550).contains(&count), "{count}");
    for seed in 0..1000 {
        assert_eq!(
            run(general.clone(), &circuit, seed).get_bit(QubitId(1)),
            run(bare.clone(), &circuit, seed).get_bit(QubitId(1))
        );
    }
}

#[test]
fn preparation_and_readout_channels_guard_apply_and_try_apply() {
    let qubits = [QubitId(0)];
    let rows: Vec<(Box<dyn NoiseChannel>, NoiseEvent<'_>, GateType)> = vec![
        (
            Box::new(
                PreparationChannel::new(1.0)
                    .unwrap()
                    .with_leakage(1.0)
                    .unwrap(),
            ),
            NoiseEvent::AfterPreparation {
                gate_type: GateType::PZ,
                qubits: &qubits,
            },
            GateType::PZ,
        ),
        (
            Box::new(MeasurementChannel::symmetric(1.0).unwrap()),
            NoiseEvent::AfterMeasurement {
                gate_type: GateType::MZ,
                qubits: &qubits,
                outcomes: &[false],
            },
            GateType::MZ,
        ),
        (
            Box::new(MeasurementStateFlipChannel::new(1.0).unwrap()),
            NoiseEvent::BeforeMeasurement {
                gate_type: GateType::MZ,
                qubits: &qubits,
            },
            GateType::MZ,
        ),
        (
            Box::new(PerGatePauliChannel::new().with_meas_init(1.0, 1.0).unwrap()),
            NoiseEvent::BeforeMeasurement {
                gate_type: GateType::MZ,
                qubits: &qubits,
            },
            GateType::MZ,
        ),
        (
            Box::new(PerGatePauliChannel::new().with_meas_init(1.0, 1.0).unwrap()),
            NoiseEvent::AfterPreparation {
                gate_type: GateType::PZ,
                qubits: &qubits,
            },
            GateType::PZ,
        ),
    ];
    for (channel, event, gate) in rows {
        for use_try in [false, true] {
            let mut ctx = NoiseContext::new();
            let mut rng = PecosRng::seed_from_u64(42);
            assert!(
                !channel.apply(&event, &mut ctx, &mut rng).is_none(),
                "{} control",
                channel.name()
            );
            ctx.add_noiseless_gate(gate);
            let mut expected = rng.clone();
            let response = if use_try {
                channel
                    .try_apply(&event, &mut ctx, &mut rng)
                    .unwrap_or(NoiseResponse::None)
            } else {
                channel.apply(&event, &mut ctx, &mut rng)
            };
            assert!(response.is_none(), "{} try={use_try}", channel.name());
            assert_eq!(rng.random::<u64>(), expected.random::<u64>());
        }
    }
}

#[test]
fn composite_noiseless_guard_precedes_context_mutating_primitives() {
    let channels: Vec<Box<dyn NoiseChannel>> = vec![
        Box::new(CompositeChannelBuilder::single_qubit("leak", leak())),
        Box::new(CompositeChannelBuilder::batch_single_qubit("leak_batch", 1.0, leak()).unwrap()),
        Box::new(CompositeChannelBuilder::crosstalk("leak_other", leak())),
    ];
    for channel in channels {
        for quiet in [false, true] {
            let mut ctx = NoiseContext::new();
            ctx.mark_prepared(QubitId(0));
            ctx.mark_prepared(QubitId(1));
            if quiet {
                ctx.add_noiseless_gate(GateType::X);
            }
            channel.apply(
                &NoiseEvent::after_gate(GateType::X, &[QubitId(0)], &[]),
                &mut ctx,
                &mut PecosRng::seed_from_u64(0),
            );
            assert_eq!(
                ctx.leaked_count(),
                usize::from(!quiet),
                "{}",
                channel.name()
            );
        }
    }
}

#[test]
fn noiseless_lifecycle_and_event_type_classification() {
    let q = QubitId(0);
    for kind in BUILDERS {
        let mut model = model(kind, 0.0, 0.0, 0.0, &[GateType::PZ, GateType::MZ]);
        let mut rng = PecosRng::seed_from_u64(42);
        model.context_mut().mark_leaked(q);
        model.emit(
            &NoiseEvent::AfterPreparation {
                gate_type: GateType::PZ,
                qubits: &[q],
            },
            &mut rng,
        );
        assert!(!model.context().is_leaked(q));
        assert!(model.context().is_active(q));
        model.emit(
            &NoiseEvent::AfterMeasurement {
                gate_type: GateType::MZ,
                qubits: &[q],
                outcomes: &[false],
            },
            &mut rng,
        );
        assert!(!model.context().is_active(q));
        assert!(
            !model
                .context()
                .is_noiseless_operation(&NoiseEvent::AfterReset { qubits: &[q] })
        );
        assert!(
            !model
                .context()
                .is_noiseless_operation(&NoiseEvent::IdleTime {
                    qubits: &[q],
                    duration: TimeUnits::new(1)
                })
        );
        assert!(
            !model
                .context()
                .is_noiseless_operation(&NoiseEvent::AfterPreparation {
                    gate_type: GateType::QAlloc,
                    qubits: &[q]
                })
        );
        assert!(
            !model
                .context()
                .is_noiseless_operation(&NoiseEvent::BeforeMeasurement {
                    gate_type: GateType::MeasureLeaked,
                    qubits: &[q]
                })
        );
    }
}

#[test]
fn before_gate_fault_primitives_are_intentionally_not_silenced() {
    let channel = CompositeChannelBuilder::before_gate("before_fault", leak());
    let mut ctx = NoiseContext::new();
    ctx.add_noiseless_gate(GateType::X);
    channel.apply(
        &NoiseEvent::before_gate(GateType::X, &[QubitId(0)], &[]),
        &mut ctx,
        &mut PecosRng::seed_from_u64(0),
    );
    assert!(ctx.is_leaked(QubitId(0)));
}

#[test]
fn noiseless_composites_do_not_unleak() {
    let channels: Vec<Box<dyn NoiseChannel>> = vec![
        Box::new(CompositeChannelBuilder::single_qubit("unleak", unleak())),
        Box::new(
            CompositeChannelBuilder::batch_single_qubit("unleak_batch", 1.0, unleak()).unwrap(),
        ),
    ];
    for channel in channels {
        for quiet in [false, true] {
            let mut ctx = NoiseContext::new();
            ctx.mark_leaked(QubitId(0));
            if quiet {
                ctx.add_noiseless_gate(GateType::X);
            }
            channel.apply(
                &NoiseEvent::after_gate(GateType::X, &[QubitId(0)], &[]),
                &mut ctx,
                &mut PecosRng::seed_from_u64(0),
            );
            assert_eq!(ctx.is_leaked(QubitId(0)), quiet);
        }
    }
}

fn preparation_crosstalk_model(composite: bool) -> ComposableNoiseModel {
    if composite {
        CompositeNoiseModelBuilder::new()
            .with_p_prep_crosstalk(1.0)
            .unwrap()
            .build()
            .unwrap()
    } else {
        GeneralNoiseModelBuilder::new()
            .with_p_prep_crosstalk(1.0)
            .build()
            .unwrap()
    }
}

// Each row uses the last public q1 outcome; hidden engine readouts are excluded.
fn check_preparation_crosstalk_parity(row: usize) {
    use pecos_engines::noise::GeneralNoiseModel;
    use pecos_engines::{ByteMessage, Engine, QuantumSystem, StateVecEngine};
    let mut circuit = CommandBuilder::new().pz(&[0, 1]);
    let mut message = ByteMessage::quantum_operations_builder();
    message.pz(&[0, 1]);
    if row == 3 {
        circuit = circuit.mz(&[1]);
        message.mz(&[1]);
    }
    if row != 0 {
        circuit = circuit.h(&[1]);
        message.h(&[1]);
    }
    if row == 2 {
        circuit = circuit.x(&[0]);
        message.x(&[0]);
    } else {
        circuit = circuit.pz(&[0]);
        message.pz(&[0]);
    }
    if row != 0 {
        circuit = circuit.h(&[1]);
        message.h(&[1]);
    }
    let circuit = circuit.mz(&[1]).build();
    let message = message.mz(&[1]).build();
    let engines = (0..1000)
        .filter(|&seed| {
            let model = GeneralNoiseModel::builder()
                .with_p_prep_crosstalk(1.0)
                .with_seed(seed)
                .build();
            let mut system = QuantumSystem::new(
                Box::new(model),
                Box::new(StateVecEngine::with_seed(2, seed)),
            );
            // Match run_general_noise_model: derive independent component seeds.
            system.set_seed(seed);
            *system
                .process(message.clone())
                .unwrap()
                .outcomes()
                .unwrap()
                .last()
                .unwrap()
                == 1
        })
        .count();
    let counts = [
        engines,
        ones(&preparation_crosstalk_model(false), &circuit, 1),
        ones(&preparation_crosstalk_model(true), &circuit, 1),
    ];
    eprintln!("preparation parity row {row}: engines/general/composite = {counts:?}");
    for count in counts {
        if row == 0 || row == 2 {
            assert_eq!(count, 0, "row {row}: {counts:?}");
        } else {
            // Five standard deviations for Binomial(1000, 0.5): 79.06 shots.
            assert!((421..=579).contains(&count), "row {row}: {counts:?}");
        }
    }
}

#[test]
fn preparation_crosstalk_dephases_zero_state() {
    check_preparation_crosstalk_parity(0);
}

#[test]
fn preparation_crosstalk_dephases_superposition() {
    check_preparation_crosstalk_parity(1);
}

#[test]
fn preparation_crosstalk_has_no_gate_sources() {
    check_preparation_crosstalk_parity(2);
}

#[test]
fn preparation_crosstalk_keeps_measured_targets() {
    check_preparation_crosstalk_parity(3);
}

#[test]
fn crosstalk_rates_are_separate() {
    for preparation_source in [false, true] {
        let mut circuit = CommandBuilder::new().pz(&[0, 1]).h(&[1]);
        circuit = if preparation_source {
            circuit.pz(&[0])
        } else {
            circuit.mz(&[0])
        };
        let circuit = circuit.h(&[1]).mz(&[1]).build();
        for source_rate in [0.0, 0.25] {
            let model = if preparation_source {
                GeneralNoiseModelBuilder::new()
                    .with_p_prep_crosstalk(source_rate)
                    .with_p_meas_crosstalk(1.0, 0.0)
            } else {
                GeneralNoiseModelBuilder::new()
                    .with_p_prep_crosstalk(1.0)
                    .with_p_meas_crosstalk(source_rate, 0.0)
            }
            .build()
            .unwrap();
            let count = ones(&model, &circuit, 1);
            if source_rate == 0.0 {
                assert_eq!(count, 0, "preparation_source={preparation_source}");
            } else {
                // Five sigma around 1000 * 0.25 * 0.5 = 125 flips.
                assert!(
                    (73..=177).contains(&count),
                    "preparation_source={preparation_source}, source_rate={source_rate}: {count}"
                );
            }
        }
    }
}

#[test]
fn qalloc_is_not_a_preparation_crosstalk_source() {
    let circuit = CommandBuilder::new()
        .pz(&[0, 1])
        .h(&[1])
        .qalloc(&[0])
        .h(&[1])
        .mz(&[1])
        .build();
    for composite in [false, true] {
        assert_eq!(
            ones(&preparation_crosstalk_model(composite), &circuit, 1),
            0
        );
    }
}

#[test]
fn crosstalk_targets_include_qalloc_and_noiseless_preparations() {
    // Deliberate engine deviation: every preparation establishes existence.
    for composite in [false, true] {
        let alloc = CommandBuilder::new()
            .qalloc(&[1])
            .h(&[1])
            .pz(&[0])
            .h(&[1])
            .mz(&[1])
            .build();
        assert!((421..=579).contains(&ones(&preparation_crosstalk_model(composite), &alloc, 1)));
        let circuit = CommandBuilder::new()
            .pz(&[0, 1])
            .h(&[1])
            .mz(&[0])
            .h(&[1])
            .mz(&[1])
            .build();
        let model = if composite {
            CompositeNoiseModelBuilder::new()
                .with_p_meas_crosstalk(1.0, 0.0)
                .unwrap()
                .build()
                .unwrap()
        } else {
            GeneralNoiseModelBuilder::new()
                .with_p_meas_crosstalk(1.0, 0.0)
                .build()
                .unwrap()
        }
        .with_noiseless_gate(GateType::PZ);
        assert!((421..=579).contains(&ones(&model, &circuit, 1)));
    }
}

#[test]
fn measure_free_keeps_crosstalk_target_after_measurement() {
    let circuit = CommandBuilder::new()
        .pz(&[0, 1])
        .gate(GateCommand::new(
            GateType::MeasureFree,
            smallvec![QubitId(1)],
        ))
        .h(&[1])
        .pz(&[0])
        .h(&[1])
        .mz(&[1])
        .build();
    for composite in [false, true] {
        assert!((421..=579).contains(&ones(&preparation_crosstalk_model(composite), &circuit, 1)));
    }
}
