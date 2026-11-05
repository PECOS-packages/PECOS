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

//! Regressions for #1570 and #1571 through the circuit and response APIs.
use pecos_core::QubitId;
use pecos_neo::command::{CommandBuilder, CommandQueue, GateCommand, GateType};
use pecos_neo::noise::composite::CompositeNoiseModelBuilder;
use pecos_neo::noise::composite::prelude::*;
use pecos_neo::noise::plugins::CorePlugin;
use pecos_neo::noise::{
    ComposableNoiseModel, GeneralNoiseModelBuilder, NoiseChannel, NoiseContext, NoiseEvent,
    NoiseModelBuilder, NoiseResponse, SingleQubitChannel, TwoQubitChannel,
};
use pecos_neo::outcome::MeasurementOutcomes;
use pecos_neo::runner::CircuitRunner;
use pecos_random::PecosRng;
use pecos_simulators::SparseStab;

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

fn leak_after_z(model: ComposableNoiseModel) -> ComposableNoiseModel {
    model
        .add_channel(LeakAfterZ)
        .add_channel(CompositeChannelBuilder::before_gate(
            "skip",
            skip_if_leaked(),
        ))
}

fn injected(response: NoiseResponse) -> Vec<GateCommand> {
    match response {
        NoiseResponse::InjectGates(gates) => gates.to_vec(),
        NoiseResponse::Multiple(responses) => responses.into_iter().flat_map(injected).collect(),
        _ => vec![],
    }
}

fn crosstalk_builder(kind: &str) -> CompositeNoiseModelBuilder {
    let builder = CompositeNoiseModelBuilder::new();
    match kind {
        "prep" => builder.with_p_prep_crosstalk(1.0).unwrap(),
        "shorthand" => builder.with_p_crosstalk(1.0).unwrap(),
        "global" => builder.with_p_meas_crosstalk(1.0, 0.0).unwrap(),
        "local" => builder.with_p_meas_crosstalk(0.0, 1.0).unwrap(),
        _ => unreachable!(),
    }
}

fn check_crosstalk(kind: &str) {
    for measured in [false, true] {
        let mut circuit = CommandBuilder::new().pz(&[0, 1]);
        if measured {
            circuit = circuit.mz(&[1]);
        }
        circuit = circuit.h(&[1]);
        circuit = if kind == "prep" {
            circuit.pz(&[0])
        } else if kind == "local" {
            circuit.mz(&[0]).meas_crosstalk_local_payload(&[1])
        } else {
            circuit.mz(&[0]).meas_crosstalk_global_payload(&[0])
        };
        let circuit = circuit.h(&[1]).mz(&[1]).build();
        let ones = (0..1000)
            .filter(|&seed| {
                let outcomes = run(crosstalk_builder(kind).build().unwrap(), &circuit, seed);
                outcomes.get_all(QubitId(1)).last().unwrap().outcome
            })
            .count();
        assert!(
            (421..=579).contains(&ones),
            "{kind}, measured={measured}: {ones}/1000 flips"
        );
    }

    // Aggregate direct emissions, checking the effect and source exclusions too.
    let mut model = crosstalk_builder(kind).build().unwrap();
    let mut rng = PecosRng::seed_from_u64(42);
    model.context_mut().mark_prepared(QubitId(0));
    model.context_mut().mark_prepared(QubitId(1));
    let source = [QubitId(0)];
    let trigger = if kind == "prep" {
        NoiseEvent::AfterPreparation {
            gate_type: GateType::PZ,
            qubits: &source,
        }
    } else {
        NoiseEvent::CrosstalkPayload {
            gate_type: if kind == "local" {
                GateType::MeasCrosstalkLocalPayload
            } else {
                GateType::MeasCrosstalkGlobalPayload
            },
            qubits: if kind == "local" {
                &[QubitId(1)]
            } else {
                &source
            },
        }
    };
    for measured in [false, true] {
        if measured {
            model.context_mut().mark_measured(QubitId(1));
        }
        let mut count = 0;
        for _ in 0..1000 {
            let response = model.emit(&trigger, &mut rng);
            if kind != "prep" {
                assert!(
                    matches!(response, NoiseResponse::MeasureHidden { ref qubits, .. } if qubits.as_slice() == [QubitId(1)])
                );
                continue;
            }
            for gate in injected(response) {
                assert_eq!(gate.qubits.as_slice(), &[QubitId(1)]);
                assert_eq!(gate.gate_type, GateType::Z);
                count += 1;
            }
        }
        assert!(
            kind != "prep" || (421..=579).contains(&count),
            "{kind}, measured={measured}: {count}"
        );
    }
    model.context_mut().mark_leaked(QubitId(1));
    for _ in 0..1000 {
        assert!(model.emit(&trigger, &mut rng).is_none());
    }
}

#[test]
fn composite_preparation_crosstalk_reaches_prepared_qubit() {
    check_crosstalk("prep");
}
#[test]
fn composite_shorthand_crosstalk_reaches_prepared_qubit() {
    check_crosstalk("shorthand");
}
#[test]
fn composite_global_crosstalk_reaches_prepared_qubit() {
    check_crosstalk("global");
    // The newly reachable transition action must propagate both injected
    // flips and leakage marks through the runner to the spectator.
    let circuit = CommandBuilder::new()
        .pz(&[0, 1])
        .mz(&[0])
        .meas_crosstalk_global_payload(&[0])
        .mz(&[1])
        .build();
    for leaks in [false, true] {
        let flip = if leaks { 0.0 } else { 1.0 };
        let leak = 1.0 - flip;
        let transitions =
            pecos_neo::noise::CrosstalkTransitions::custom(0.0, flip, leak, 0.0, flip, leak)
                .unwrap();
        let model = crosstalk_builder("global")
            .with_p_meas_crosstalk_model(transitions)
            .unwrap()
            .build()
            .unwrap();
        let result = run(model, &circuit, 42);
        let spectator = result.get(QubitId(1)).unwrap();
        assert!(spectator.outcome);
        assert_eq!(spectator.is_leaked, leaks);
    }
}
#[test]
fn composite_local_crosstalk_reaches_prepared_qubit() {
    check_crosstalk("local");
}

#[test]
fn preparation_clears_leakage_issue_table() {
    let circuit = CommandBuilder::new()
        .pz(&[0])
        .z(&[0])
        .pz(&[0])
        .x(&[0])
        .mz(&[0])
        .build();
    let models = [
        ("core", ComposableNoiseModel::new().add_plugin(&CorePlugin)),
        ("general", GeneralNoiseModelBuilder::new().build().unwrap()),
        ("unified", NoiseModelBuilder::new().build().unwrap()),
        (
            "composite",
            CompositeNoiseModelBuilder::new().build().unwrap(),
        ),
        // A prep fault on each PZ makes the final X yield zero after reset.
        (
            "composite_prep",
            CompositeNoiseModelBuilder::new()
                .with_p_prep(1.0)
                .unwrap()
                .build()
                .unwrap(),
        ),
    ];
    let mut rows = Vec::new();
    for (name, model) in models {
        let model = leak_after_z(model);
        let mut correct = 0;
        for seed in 0..1000 {
            let result = run(model.clone(), &circuit, seed);
            let outcome = result.get(QubitId(0)).unwrap();
            correct +=
                usize::from(!outcome.is_leaked && outcome.outcome == (name != "composite_prep"));
        }
        rows.push((name, correct));
    }
    assert!(rows.iter().all(|(_, count)| *count == 1000), "{rows:?}");
}

#[test]
fn same_event_preparation_leak_sticks() {
    let mut model = CompositeNoiseModelBuilder::new()
        .with_p_prep(1.0)
        .unwrap()
        .with_p_prep_leak_ratio(1.0)
        .unwrap()
        .build()
        .unwrap();
    model.context_mut().mark_leaked(QubitId(0));
    model.emit(
        &NoiseEvent::AfterPreparation {
            gate_type: GateType::PZ,
            qubits: &[QubitId(0)],
        },
        &mut PecosRng::seed_from_u64(42),
    );
    assert!(model.context().is_active(QubitId(0)));
    assert!(model.context().is_leaked(QubitId(0)));
    let circuit = CommandBuilder::new()
        .pz(&[0])
        .pz(&[0])
        .x(&[0])
        .mz(&[0])
        .build();
    assert!(run(model, &circuit, 42).get(QubitId(0)).unwrap().is_leaked);
}

#[test]
fn composite_active_condition_and_queries_follow_lifecycle() {
    let mut model = CompositeNoiseModelBuilder::new()
        .build()
        .unwrap()
        .add_channel(CompositeChannelBuilder::single_qubit(
            "active_only",
            when(active(), inject_x(), nothing()),
        ));
    let mut rng = PecosRng::seed_from_u64(42);
    let qubits = [QubitId(0)];
    let gate = NoiseEvent::AfterGate {
        gate_type: GateType::I,
        qubits: &qubits,
        leaked_suppressed: &[],
        group_size: std::num::NonZeroUsize::MIN,
        gate_id: None,
        angles: &[],
    };
    assert_eq!(injected(model.emit(&gate, &mut rng)), vec![]);
    model.emit(
        &NoiseEvent::AfterPreparation {
            gate_type: GateType::PZ,
            qubits: &qubits,
        },
        &mut rng,
    );
    assert_eq!(model.context().active_count(), 1);
    assert_eq!(model.context().other_active_qubits(&[]), qubits);
    assert!(model.context().exists(QubitId(0)));
    assert_eq!(model.context().other_qubits(&[]), qubits);
    assert_eq!(
        model.context().qubit_state(QubitId(0)),
        Some(pecos_neo::noise::context::QubitState::Active)
    );
    assert_eq!(injected(model.emit(&gate, &mut rng)).len(), 1);
    model.emit(
        &NoiseEvent::AfterMeasurement {
            gate_type: GateType::MZ,
            qubits: &qubits,
            outcomes: &[false],
        },
        &mut rng,
    );
    assert_eq!(model.context().active_count(), 0);
    assert_eq!(
        model.context().qubit_state(QubitId(0)),
        Some(pecos_neo::noise::context::QubitState::Inactive)
    );
    assert_eq!(model.context().other_qubits(&[]), qubits);
    assert_eq!(model.context().other_active_qubits(&[]), vec![]);
    assert_eq!(injected(model.emit(&gate, &mut rng)), vec![]);
}

fn single_seep_channel() -> SingleQubitChannel {
    SingleQubitChannel::depolarizing(1.0)
        .unwrap()
        .with_emission(1.0, 1.0)
        .unwrap()
        .with_seepage(1.0)
        .unwrap()
}

fn pair_seep_channel() -> TwoQubitChannel {
    TwoQubitChannel::depolarizing(1.0)
        .unwrap()
        .with_leakage(1.0)
        .unwrap()
        .with_seepage(1.0)
        .unwrap()
}

fn check_circuit_a(general: bool) {
    let model = if general {
        GeneralNoiseModelBuilder::new()
            .with_p1(1.0)
            .with_p1_emission_ratio(1.0)
            .with_p1_seepage(1.0)
            .build()
            .unwrap()
    } else {
        ComposableNoiseModel::new()
            .add_plugin(&CorePlugin)
            .add_channel(single_seep_channel())
    };
    let model = leak_after_z(model).with_noiseless_gates(&[GateType::H, GateType::Z]);
    let circuit = CommandBuilder::new()
        .pz(&[0])
        .h(&[0])
        .z(&[0])
        .x(&[0])
        .h(&[0])
        .mz(&[0])
        .build();
    let mut counts = [0; 2];
    for seed in 0..1000 {
        let result = run(model.clone(), &circuit, seed);
        let outcome = result.get(QubitId(0)).unwrap();
        assert!(!outcome.is_leaked);
        counts[usize::from(outcome.outcome)] += 1;
    }
    assert!(
        counts.iter().all(|count| (421..=579).contains(count)),
        "general={general}: {counts:?}"
    );
}

#[test]
fn circuit_a_single_qubit_seep_discards_phase() {
    check_circuit_a(false);
}
#[test]
fn circuit_a_general_builder_seep_discards_phase() {
    check_circuit_a(true);
}

#[test]
fn circuit_b_pair_seep_discards_correlations() {
    let model = leak_after_z(
        ComposableNoiseModel::new()
            .add_plugin(&CorePlugin)
            .add_channel(pair_seep_channel()),
    )
    .with_noiseless_gate(GateType::CX);
    let circuit = CommandBuilder::new()
        .pz(&[0, 1])
        .h(&[0])
        .cx(&[(0, 1)])
        .z(&[0])
        .cz(&[(0, 1)])
        .cx(&[(0, 1)])
        .h(&[0])
        .mz(&[0, 1])
        .build();
    let mut counts = [0; 4];
    for seed in 0..1000 {
        let result = run(model.clone(), &circuit, seed);
        assert!(result.iter().all(|outcome| !outcome.is_leaked));
        counts[2 * usize::from(result.get_bit(QubitId(0)).unwrap())
            + usize::from(result.get_bit(QubitId(1)).unwrap())] += 1;
    }
    // The old state-preserving seep gives 10 every time. Require both q0
    // values near 1/2 AND all four decoded Bell outcomes near 1/4, so neither
    // the pre-leak phase nor the Bell correlations can survive the channel.
    assert!((421..=579).contains(&(counts[2] + counts[3])), "{counts:?}");
    // Five sigma around 250 for each of 1000 samples.
    assert!(
        counts.iter().all(|count| (181..=319).contains(count)),
        "{counts:?}"
    );
}

fn record_seep(response: NoiseResponse, gates: &mut Vec<GateCommand>, unleaked: &mut Vec<QubitId>) {
    match response {
        NoiseResponse::None => {}
        NoiseResponse::InjectGates(injected) => gates.extend(*injected),
        NoiseResponse::MarkUnleaked(qubits) => unleaked.extend(qubits),
        NoiseResponse::Multiple(responses) => {
            for response in responses {
                record_seep(response, gates, unleaked);
            }
        }
        other => panic!("unexpected seep response: {other:?}"),
    }
}

fn check_seep_distribution(pair: bool) {
    let channel: Box<dyn NoiseChannel> = if pair {
        Box::new(pair_seep_channel())
    } else {
        Box::new(single_seep_channel())
    };
    let qubits = if pair {
        vec![QubitId(0), QubitId(1)]
    } else {
        vec![QubitId(0)]
    };
    let mut failures = Vec::new();
    for suppressed in [false, true] {
        for use_try in [false, true] {
            let mut ctx = NoiseContext::new();
            for &qubit in &qubits {
                ctx.mark_leaked(qubit);
            }
            let event = NoiseEvent::AfterGate {
                gate_type: if pair { GateType::CZ } else { GateType::X },
                qubits: if suppressed { &[] } else { &qubits },
                leaked_suppressed: if suppressed { &qubits } else { &[] },
                group_size: std::num::NonZeroUsize::new(qubits.len()).unwrap(),
                gate_id: None,
                angles: &[],
            };
            let mut rng = PecosRng::seed_from_u64(1570);
            let mut marginal = [[0; 4]; 2];
            let mut joint = [0; 16];
            for _ in 0..16_000 {
                let response = if use_try {
                    channel.try_apply(&event, &mut ctx, &mut rng).unwrap()
                } else {
                    channel.apply(&event, &mut ctx, &mut rng)
                };
                let mut gates = Vec::new();
                let mut unleaked = Vec::new();
                record_seep(response, &mut gates, &mut unleaked);
                assert_eq!(unleaked, qubits);
                let mut paulis = [0; 2]; // Absence of an injected gate is identity.
                let mut seen = [false; 2];
                for gate in gates {
                    assert_eq!(gate.qubits.len(), 1);
                    let q = gate.qubits[0].0;
                    assert!(q < qubits.len() && !seen[q]);
                    seen[q] = true;
                    paulis[q] = match gate.gate_type {
                        GateType::I => 0,
                        GateType::X => 1,
                        GateType::Y => 2,
                        GateType::Z => 3,
                        other => panic!("seep injected {other:?}"),
                    };
                }
                for q in 0..qubits.len() {
                    marginal[q][paulis[q]] += 1;
                }
                joint[4 * paulis[0] + paulis[1]] += 1;
            }
            // Five-sigma bounds: 4000 +/- 274 per Pauli, 1000 +/- 154 per
            // ordered Pauli pair. The joint test rejects shared pair draws.
            if !marginal[..qubits.len()]
                .iter()
                .flatten()
                .all(|n| (3726..=4274).contains(n))
                || (pair && !joint.iter().all(|n| (846..=1154).contains(n)))
            {
                failures.push((suppressed, use_try, marginal, joint));
            }
        }
    }
    assert!(failures.is_empty(), "{failures:?}");
}

#[test]
fn single_seep_responses_are_uniform_on_every_path() {
    check_seep_distribution(false);
}
#[test]
fn pair_seep_responses_are_uniform_and_independent_on_every_path() {
    check_seep_distribution(true);
}
