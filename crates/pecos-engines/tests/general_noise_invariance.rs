//! Golden coverage of the legacy `ControlEngine` path only.
//! Capture deliberately with:
//! ```sh
//! PECOS_UPDATE_GENERAL_NOISE_FIXTURE=1 cargo test -p pecos-engines --test general_noise_invariance
//! ```
//! Missing files otherwise fail.
//! Runtime-frame and scheduled coverage remains in `runtime_frame_production.rs`
//! and `scheduled_*.rs`. Keep transcendental idle behavior in its own fixture.
use pecos_core::{Angle64, Gate, RngManageable, gate_type::GateType};
use pecos_engines::noise::{GeneralNoiseModel, PauliLeakageChannel, StateTransitionChannel};
use pecos_engines::quantum::StateVecEngine;
use pecos_engines::{ByteMessage, ByteMessageBuilder, ControlEngine, Engine, EngineStage};
use std::collections::BTreeMap;
use std::fmt::Write;
use std::path::Path;

const SEED: u64 = 1907;
const SHOTS: usize = 16;

struct Input {
    leaked: Vec<usize>,
    message: ByteMessage,
}

struct Scenario {
    name: &'static str,
    model: GeneralNoiseModel,
    inputs: Vec<Input>,
}

fn weights(entries: &[(&str, f64)]) -> BTreeMap<String, f64> {
    entries.iter().map(|&(k, v)| (k.to_owned(), v)).collect()
}

fn input(leaked: &[usize], build: impl FnOnce(&mut ByteMessageBuilder)) -> Input {
    let mut builder = ByteMessage::quantum_operations_builder();
    build(&mut builder);
    Input {
        leaked: leaked.to_vec(),
        message: builder.build(),
    }
}

fn leakage() -> Scenario {
    Scenario {
        name: "leakage",
        model: GeneralNoiseModel::builder()
            .with_seed(SEED)
            .with_p_prep(0.7)
            .with_prep_leak_ratio(1.0)
            .with_p1(1.0)
            .with_p1_emission_ratio(1.0)
            .with_p1_emission_model(&weights(&[("L", 1.0)]))
            .with_p2(1.0)
            .with_p2_emission_ratio(1.0)
            .with_p2_emission_model(&weights(&[("LL", 1.0)]))
            .with_seepage_prob(1.0)
            .with_leakage_scale(0.5)
            .build(),
        inputs: vec![input(&[3], |b| {
            b.measure_z_leaked(&[0, 3]).pz(&[0, 1, 2]);
            b.measure_z_leaked(&[0, 1, 2, 3]);
            b.h(&[0, 1, 2, 3]).measure_z_leaked(&[0, 1, 2, 3]);
            b.cx(&[(0, 1), (2, 3)]).measure_z_leaked(&[0, 1, 2, 3]);
            b.h(&[0, 1, 2, 3]).cx(&[(0, 1), (2, 3)]);
            b.measure_z_leaked(&[0, 1, 2, 3]);
        })],
    }
}

fn readout(name: &'static str, noiseless: Option<GateType>) -> Scenario {
    let mut builder = GeneralNoiseModel::builder()
        .with_seed(SEED)
        .with_p_meas_0(0.25)
        .with_p_meas_1(0.75);
    if let Some(gate) = noiseless {
        builder = builder.with_noiseless_gate(gate);
    }
    Scenario {
        name,
        model: builder.build(),
        inputs: vec![input(&[0], |b| {
            b.x(&[2]).measure_z_leaked(&[0, 1, 2]);
            b.mz(&[0, 1, 2]).mpz(&[0, 1, 2]);
            b.measure_z_leaked(&[0, 1, 2]).mz(&[0, 1, 2]);
        })],
    }
}

fn emission() -> Scenario {
    Scenario {
        name: "emission",
        model: GeneralNoiseModel::builder()
            .with_seed(SEED)
            .with_p1(0.75)
            .with_p1_emission_ratio(1.0)
            .with_p1_emission_model(&weights(&[("X", 0.25), ("Y", 0.25), ("Z", 0.5)]))
            .with_p2(0.75)
            .with_p2_emission_ratio(1.0)
            .with_p2_emission_model(&weights(&[("XZ", 0.5), ("YX", 0.5)]))
            .build(),
        inputs: vec![input(&[], |b| {
            b.h(&[0, 1, 2, 3]).cx(&[(0, 1), (2, 3)]).mz(&[0, 1, 2, 3]);
        })],
    }
}

fn crosstalk(prep: bool) -> Scenario {
    let builder = GeneralNoiseModel::builder().with_seed(SEED);
    let model = if prep {
        builder.with_p_prep_crosstalk(1.0)
    } else {
        builder
            .with_p_meas_crosstalk_global(1.0)
            .with_p_meas_crosstalk_local(1.0)
            .with_p_meas_crosstalk_model(&weights(&[("0->1", 1.0), ("1->0", 1.0)]))
    }
    .build();
    Scenario {
        name: if prep {
            "prep_crosstalk"
        } else {
            "measurement_crosstalk"
        },
        model,
        inputs: vec![
            input(&[], |b| {
                b.pz(&[0, 1, 2, 3]);
            }),
            input(&[2], |b| {
                if prep {
                    b.pz(&[0]);
                } else {
                    b.mz(&[0]).meas_crosstalk_global_payload(&[0]);
                    b.meas_crosstalk_local_payload(&[1, 2, 3]);
                }
                b.measure_z_leaked(&[0, 1, 2, 3]);
            }),
        ],
    }
}

fn idle(transcendental: bool) -> Scenario {
    let builder = GeneralNoiseModel::builder()
        .with_seed(SEED)
        .with_idle_after_2q(0.5);
    let model = if transcendental {
        builder
            .with_p_idle_sin_squared(1.25, &weights(&[("X", 1.0), ("Y", 0.5), ("Z", 0.75)]))
            .with_p_idle_coherent(0.5, &weights(&[("RX", 1.0), ("RY", 0.5), ("RZ", 0.25)]))
    } else {
        builder.with_p_idle_linear(0.75, &weights(&[("X", 0.5), ("Z", 0.5)]))
    }
    .build();
    Scenario {
        name: if transcendental {
            "idle_sin_squared_coherent"
        } else {
            "idle_linear"
        },
        model,
        inputs: vec![input(&[3], |b| {
            b.h(&[0, 1, 2]).idle(0.75, &[0, 1, 2, 3]);
            b.cx(&[(0, 1)]).idle(1.25, &[0, 1, 2, 3]).mz(&[0, 1, 2, 3]);
        })],
    }
}

fn hooks() -> Scenario {
    let flip =
        StateTransitionChannel::from_transitions(1.0, &weights(&[("0->1", 1.0), ("1->0", 1.0)]))
            .unwrap();
    let recovery = StateTransitionChannel::leak_recovery(1.0, 1.0).unwrap();
    let pauli = PauliLeakageChannel::new(1.0, &weights(&[("Z", 1.0)])).unwrap();
    Scenario {
        name: "transition_channel_hooks",
        model: GeneralNoiseModel::builder()
            .with_seed(SEED)
            .add_p1_transition_channel_before_gate(&flip)
            .add_p1_pauli_leakage_channel_before_gate(&pauli)
            .add_p1_transition_channel_after_gate(&flip)
            .add_p1_pauli_leakage_channel_after_gate(&pauli)
            .add_p2_transition_channel_before_gate(&recovery)
            .add_p2_transition_channel_before_gate(&flip)
            .add_p2_pauli_leakage_channel_before_gate(&pauli)
            .add_p2_transition_channel_after_gate(&flip)
            .add_p2_pauli_leakage_channel_after_gate(&pauli)
            .build(),
        inputs: vec![input(&[2], |b| {
            // User readout and the first deferring transition share an input.
            b.mz(&[0]).h(&[0, 1]).cx(&[(1, 2)]).mz(&[0, 1, 2]);
        })],
    }
}

fn noiseless_gates() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "noiseless_2q",
            model: GeneralNoiseModel::builder()
                .with_seed(SEED)
                .with_p2(1.0)
                .with_p2_pauli_model(&weights(&[("XZ", 1.0)]))
                .with_noiseless_gate(GateType::CX)
                .build(),
            inputs: vec![input(&[], |b| {
                b.h(&[0]).cx(&[(0, 1)]).cz(&[(0, 1)]).mz(&[0, 1]);
            })],
        },
        Scenario {
            name: "noiseless_pz",
            model: GeneralNoiseModel::builder()
                .with_seed(SEED)
                .with_p_prep(1.0)
                .with_noiseless_gate(GateType::PZ)
                .build(),
            inputs: vec![input(&[0], |b| {
                b.measure_z_leaked(&[0])
                    .pz(&[0])
                    .h(&[0])
                    .measure_z_leaked(&[0]);
            })],
        },
        Scenario {
            name: "noiseless_rz",
            model: GeneralNoiseModel::builder()
                .with_seed(SEED)
                .with_p1(1.0)
                .with_p1_pauli_model(&weights(&[("X", 1.0)]))
                .with_noiseless_gate(GateType::RZ)
                .build(),
            inputs: vec![input(&[], |b| {
                b.h(&[0]).rz(Angle64::QUARTER_TURN, &[0]);
                b.u(Angle64::ZERO, Angle64::ZERO, Angle64::QUARTER_TURN, &[0]);
                b.mz(&[0]);
            })],
        },
    ]
}

fn pairs_and_scales() -> Vec<Scenario> {
    vec![
        Scenario {
            name: "multiple_pairs",
            model: GeneralNoiseModel::builder()
                .with_seed(SEED)
                .with_p2(0.75)
                .with_p2_pauli_model(&weights(&[("XZ", 0.5), ("YX", 0.5)]))
                .build(),
            inputs: vec![input(&[], |b| {
                // Overlapping pairs in one command are rejected by Gate payload
                // validation; retain the publicly reachable multi-pair case.
                b.h(&[0, 2]).cx(&[(0, 1), (2, 3)]).mz(&[0, 1, 2, 3]);
            })],
        },
        Scenario {
            name: "scales",
            model: GeneralNoiseModel::builder()
                .with_seed(SEED)
                .with_scale(0.5)
                .with_prep_scale(0.75)
                .with_p1_scale(0.5)
                .with_p2_scale(0.75)
                .with_meas_scale(0.5)
                .with_idle_scale(0.5)
                .with_p_prep(0.5)
                .with_p1(0.75)
                .with_p2(0.75)
                .with_p_meas_0(0.5)
                .with_p_meas_1(0.75)
                .with_p_idle_linear(0.75, &weights(&[("Z", 1.0)]))
                .build(),
            inputs: vec![input(&[], |b| {
                b.pz(&[0, 1, 2, 3]).h(&[0, 1, 2, 3]).cx(&[(0, 1), (2, 3)]);
                b.idle(1.0, &[0, 1, 2, 3]).mz(&[0, 1, 2, 3]);
            })],
        },
    ]
}

fn scenarios() -> Vec<Scenario> {
    let mut scenarios = vec![
        leakage(),
        readout("leaked_readout", None),
        emission(),
        crosstalk(false),
        crosstalk(true),
        idle(false),
        idle(true),
        hooks(),
        readout("noiseless_mz", Some(GateType::MZ)),
        readout("noiseless_mpz", Some(GateType::MPZ)),
        readout("noiseless_measure_z_leaked", Some(GateType::MeasureZLeaked)),
    ];
    scenarios.extend(noiseless_gates());
    scenarios.push(Scenario {
        name: "asymmetric_readout",
        model: GeneralNoiseModel::builder()
            .with_seed(SEED)
            .with_p_meas_0(0.25)
            .with_p_meas_1(0.75)
            .build(),
        inputs: vec![input(&[], |b| {
            b.x(&[1, 3]).mz(&[0, 1, 2, 3]);
        })],
    });
    scenarios.extend(pairs_and_scales());
    scenarios
}

fn gate_text(gate: &Gate) -> String {
    assert!(gate.channel.is_none());
    let qubits: Vec<_> = gate.qubits.iter().map(|q| usize::from(*q)).collect();
    let angles: Vec<_> = gate.angles.iter().map(Angle64::fraction).collect();
    let params: Vec<_> = gate
        .params
        .iter()
        .map(|p| format!("0x{:016x}", p.to_bits()))
        .collect();
    format!(
        "{:?} qubits={qubits:?} angles={angles:?} params=[{}]",
        gate.gate_type,
        params.join(", ")
    )
}

// All source commands in these types are preserved, unsplit, by these profiles.
// H/CX/CZ can be removed or rewritten; Idle/payloads are consumed, so none are
// counted here. Subtracting preserved source counts gives injected *commands*,
// not qubit operations, including internal measurements and leakage resets.
fn countable(gate: &Gate) -> bool {
    matches!(
        gate.gate_type,
        GateType::X
            | GateType::Y
            | GateType::Z
            | GateType::PZ
            | GateType::MZ
            | GateType::RX
            | GateType::RY
            | GateType::RZ
    )
}

fn run(scenario: &Scenario) -> String {
    let mut model = scenario.model.clone();
    let mut engine = StateVecEngine::with_seed(4, SEED);
    let mut text = String::new();
    let mut injected = 0;
    for shot in 0..SHOTS {
        if shot > 0 {
            ControlEngine::reset(&mut model).unwrap();
            Engine::reset(&mut engine).unwrap();
        }
        let mut round = 0;
        for (message, input) in scenario.inputs.iter().enumerate() {
            for &qubit in &input.leaked {
                model.mark_as_leaked(qubit);
            }
            let source_count = input
                .message
                .quantum_ops()
                .unwrap()
                .iter()
                .filter(|g| countable(g))
                .count();
            let mut emitted_count = 0;
            let mut stage = model.start(input.message.clone()).unwrap();
            loop {
                match stage {
                    EngineStage::NeedsProcessing(commands) => {
                        let gates = commands.quantum_ops().unwrap();
                        writeln!(
                            text,
                            "shot={shot} message={message} round={round} commands={}",
                            gates.len()
                        )
                        .unwrap();
                        for (entry, gate) in gates.iter().enumerate() {
                            writeln!(
                                text,
                                "shot={shot} message={message} round={round} entry={entry} {}",
                                gate_text(gate)
                            )
                            .unwrap();
                            emitted_count += usize::from(countable(gate));
                        }
                        let reply = engine.process(commands).unwrap();
                        stage = model.continue_processing(reply).unwrap();
                        round += 1;
                    }
                    EngineStage::Complete(out) => {
                        writeln!(
                            text,
                            "shot={shot} message={message} round={round} outcomes={:?}",
                            out.outcomes().unwrap()
                        )
                        .unwrap();
                        break;
                    }
                }
            }
            injected += emitted_count
                .checked_sub(source_count)
                .expect("counted source commands must be preserved");
        }
        let mut rng = model.rng().clone();
        let tail: [u64; 4] = std::array::from_fn(|_| rng.next_u64());
        writeln!(text, "shot={shot} round={round} rng={tail:?}").unwrap();
    }
    writeln!(text, "shots={SHOTS} injected_commands={injected}").unwrap();
    text
}

fn compare(name: &str, context: &str, expected: &str, actual: &str) {
    if expected == actual {
        return;
    }
    let expected: Vec<_> = expected.split('\n').collect();
    let actual: Vec<_> = actual.split('\n').collect();
    let index = (0..expected.len().max(actual.len()))
        .find(|&i| expected.get(i) != actual.get(i))
        .unwrap();
    panic!(
        "{name}: {context}, first differing entry at line {}\nexpected: {}\nactual:   {}",
        index + 1,
        expected.get(index).unwrap_or(&"<missing>"),
        actual.get(index).unwrap_or(&"<missing>")
    );
}

#[test]
fn general_noise_invariance() {
    let directory =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/general_noise_invariance");
    let capture = std::env::var("PECOS_UPDATE_GENERAL_NOISE_FIXTURE").as_deref() == Ok("1");
    if capture {
        std::fs::create_dir_all(&directory).unwrap();
    }
    for scenario in scenarios() {
        let actual = run(&scenario);
        compare(
            scenario.name,
            "nondeterministic repeat",
            &actual,
            &run(&scenario),
        );
        assert!(actual.len() < 200_000);
        assert!(actual.ends_with('\n') && !actual.ends_with("\n\n"));
        assert!(
            actual
                .lines()
                .all(|line| line.trim_end() == line && !line.contains('\r'))
        );
        let path = directory.join(format!("{}.txt", scenario.name));
        if capture {
            std::fs::write(&path, &actual).unwrap();
        } else {
            let expected = std::fs::read_to_string(&path).unwrap_or_else(|error| {
                panic!("{}: cannot read {}: {error}", scenario.name, path.display())
            });
            compare(scenario.name, "fixture mismatch", &expected, &actual);
        }
        println!("{}: {}", scenario.name, actual.lines().last().unwrap());
    }
}
