use pecos_core::{Angle64, Gate};
use pecos_engines::{ByteMessage, ClassicalEngine, DenseStateVecEngine, Engine};
use pecos_phir::PhirEngine;
use pecos_phir_json::{PhirJsonEngine, phir_json_to_module};
use serde_json::json;

fn source(name: &str, arity: usize, angles: &[f64], unit: &str) -> String {
    let args = if arity == 1 {
        json!([["q", 3], ["q", 1]])
    } else {
        json!([[["q", 3], ["q", 1]], [["q", 2], ["q", 0]]])
    };
    let mut op = json!({"qop":name,"args":args});
    if !angles.is_empty() {
        op["angles"] = json!([angles, unit]);
    }
    if name == "Measure" {
        op["returns"] = json!([["m", 0], ["m", 1]]);
    }
    let mut ops = vec![
        json!({"data":"qvar_define","data_type":"qubits","variable":"q","size":4}),
        op,
    ];
    if name == "Measure" {
        ops.push(json!({"data":"cvar_define","data_type":"u32","variable":"m","size":2}));
    }
    json!({"format":"PHIR/JSON","version":"0.1.0","ops":ops}).to_string()
}

fn individual_gates(message: &ByteMessage) -> Vec<Gate> {
    message
        .quantum_ops()
        .unwrap()
        .into_iter()
        .flat_map(|gate| {
            gate.qubits
                .chunks_exact(gate.gate_type.quantum_arity())
                .map(|qubits| {
                    let mut individual = gate.clone();
                    individual.qubits = qubits.iter().copied().collect();
                    individual
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

#[test]
fn every_spec_name_and_alias_executes_in_both_paths() {
    let spec = include_str!("../specification/v0.1/spec.md");
    let table = spec
        .split("### Table II - Quantum operations")
        .nth(1)
        .unwrap()
        .split("## Machine operations")
        .next()
        .unwrap();
    for row in table.lines().filter(|line| line.starts_with("| `\"")) {
        let columns: Vec<_> = row.split('|').collect();
        let angles: usize = columns[3].trim().parse().unwrap();
        let arity: usize = columns[4].trim().parse().unwrap();
        for field in [columns[1], columns[2]] {
            for name in field.split('"').skip(1).step_by(2) {
                for (unit, scale) in [
                    ("rad", 1.0),
                    ("pi", std::f64::consts::PI),
                    ("deg", std::f64::consts::PI / 180.0),
                ] {
                    let values: Vec<_> = [0.37, -0.29, 0.13]
                        .into_iter()
                        .take(angles)
                        .map(|a| a / scale)
                        .collect();
                    let source = source(name, arity, &values, unit);
                    let native = individual_gates(
                        &PhirJsonEngine::from_json(&source)
                            .unwrap()
                            .generate_commands()
                            .unwrap_or_else(|e| panic!("{name}: {e}")),
                    );
                    let converted = individual_gates(
                        &PhirEngine::new(
                            phir_json_to_module(&source).unwrap_or_else(|e| panic!("{name}: {e}")),
                        )
                        .unwrap()
                        .generate_commands()
                        .unwrap(),
                    );
                    assert_eq!(native.len(), 2, "{name}");
                    assert_eq!(native, converted, "{name}, {unit}");
                    assert!(
                        native.iter().all(|gate| gate.angles.len() == angles),
                        "{name}"
                    );
                }
            }
        }
    }
}

// Independent decompositions in existing primitive gates. These do not call the
// PHIR gate mapper; the comparison includes global phase on every basis column.
fn reference(name: &str, angles: &[f64], targets: usize) -> ByteMessage {
    let mut b = ByteMessage::quantum_operations_builder();
    let q = &[3, 1][..targets];
    let pairs = &[(3, 1), (2, 0)][..targets];
    let theta = Angle64::try_from_radians(0.37).unwrap();
    match name {
        "I" => {}
        "RX" => {
            b.h(q).rz(theta, q).h(q);
        }
        "RY" => {
            b.sx(q).rz(theta, q).sxdg(q);
        }
        "U1q" => {
            let phi = Angle64::try_from_radians(angles[1]).unwrap();
            b.rz(-phi, q).rx(theta, q).rz(phi, q);
        }
        "SY" => {
            b.szdg(q).sx(q).sz(q);
        }
        "SYdg" => {
            b.szdg(q).sxdg(q).sz(q);
        }
        "SZ" | "S" => {
            b.t(q).t(q);
        }
        "SZdg" | "Sdg" => {
            b.tdg(q).tdg(q);
        }
        "F" => {
            b.sx(q).sz(q).z(q).y(q).x(q);
        }
        "Fdg" => {
            b.szdg(q).sxdg(q).x(q).y(q).z(q);
        }
        "T" => {
            b.u(
                Angle64::ZERO,
                Angle64::ZERO,
                Angle64::from_turn_ratio(1, 8),
                q,
            );
        }
        "Tdg" => {
            b.u(
                Angle64::ZERO,
                Angle64::ZERO,
                -Angle64::from_turn_ratio(1, 8),
                q,
            );
        }
        "CY" => {
            for &(c, t) in pairs {
                b.szdg(&[t]).cx(&[(c, t)]).sz(&[t]);
            }
        }
        "CZ" => {
            for &(c, t) in pairs {
                b.h(&[t]).cx(&[(c, t)]).h(&[t]);
            }
        }
        "RXX" | "RYY" | "RZZ" | "ZZPhase" => {
            for &(c, t) in pairs {
                match name {
                    "RXX" => {
                        b.h(&[c, t]);
                    }
                    "RYY" => {
                        b.sx(&[c, t]);
                    }
                    _ => {}
                }
                b.cx(&[(c, t)]).rz(theta, &[t]).cx(&[(c, t)]);
                match name {
                    "RXX" => {
                        b.h(&[c, t]);
                    }
                    "RYY" => {
                        b.sxdg(&[c, t]);
                    }
                    _ => {}
                }
            }
        }
        "ZZMax" => {
            for &(c, t) in pairs {
                b.cx(&[(c, t)]).sz(&[t]).cx(&[(c, t)]);
            }
        }
        "SWAP" => {
            for &(c, t) in pairs {
                b.cx(&[(c, t)]).cx(&[(t, c)]).cx(&[(c, t)]);
            }
        }
        _ => panic!("missing independent reference for {name}"),
    }
    b.build()
}

#[test]
fn newly_supported_gates_match_independent_state_vectors() {
    for (name, arity, angle_count) in [
        ("I", 1, 0),
        ("RX", 1, 1),
        ("RY", 1, 1),
        ("U1q", 1, 2),
        ("SY", 1, 0),
        ("SYdg", 1, 0),
        ("SZ", 1, 0),
        ("S", 1, 0),
        ("SZdg", 1, 0),
        ("Sdg", 1, 0),
        ("F", 1, 0),
        ("Fdg", 1, 0),
        ("T", 1, 0),
        ("Tdg", 1, 0),
        ("CY", 2, 0),
        ("CZ", 2, 0),
        ("RXX", 2, 1),
        ("RYY", 2, 1),
        ("RZZ", 2, 1),
        ("ZZPhase", 2, 1),
        ("ZZMax", 2, 0),
        ("SWAP", 2, 0),
    ] {
        for targets in 1..=2 {
            let angles = &[0.37, -0.29][..angle_count];
            let mut program: serde_json::Value =
                serde_json::from_str(&source(name, arity, angles, "rad")).unwrap();
            program["ops"][1]["args"]
                .as_array_mut()
                .unwrap()
                .truncate(targets);
            let program = program.to_string();
            let actual = PhirJsonEngine::from_json(&program)
                .unwrap()
                .generate_commands()
                .unwrap_or_else(|e| panic!("{name}: {e}"));
            let expected = reference(name, angles, targets);
            for basis in 0..16 {
                let mut prep = ByteMessage::quantum_operations_builder();
                for qubit in 0..4 {
                    if basis & (1 << qubit) != 0 {
                        prep.x(&[qubit]);
                    }
                }
                let prep = prep.build();
                let mut direct = DenseStateVecEngine::new(4);
                let mut oracle = DenseStateVecEngine::new(4);
                direct.process(prep.clone()).unwrap();
                oracle.process(prep).unwrap();
                direct.process(actual.clone()).unwrap();
                oracle.process(expected.clone()).unwrap();
                for (a, b) in direct
                    .simulator_mut()
                    .state()
                    .iter()
                    .zip(oracle.simulator_mut().state().iter())
                {
                    assert!(
                        (*a - *b).norm() < 1e-12,
                        "{name}, {targets} targets, basis {basis}: {a} != {b}"
                    );
                }
            }
        }
    }
}

#[test]
fn quantum_validation_rejects_bad_counts_and_repeated_targets() {
    use pecos_phir_json::v0_1::operations::OperationProcessor;
    let processor = OperationProcessor::new();
    for (name, qubits, angles, message) in [
        ("CZ", vec![0, 1, 2], vec![], "3 qubits"),
        ("RZZ", vec![0, 1], vec![], "0 angles"),
        ("RZZ", vec![0, 1], vec![0.1, 0.2], "2 angles"),
        ("H", vec![0], vec![0.1], "1 angles"),
        ("RZ", vec![0], vec![0.1, 0.2], "2 angles"),
        ("H", vec![0, 0], vec![], "[0]"),
        ("RZZ", vec![0, 1, 1, 2], vec![0.1], "[1]"),
        ("Measure", vec![0, 0], vec![], "[0]"),
        ("Init", vec![0, 0], vec![], "[0]"),
    ] {
        let mut builder = ByteMessage::quantum_operations_builder();
        let error = processor
            .add_quantum_operation_to_builder(&mut builder, name, &qubits, &angles)
            .unwrap_err()
            .to_string();
        assert!(error.contains(name) && error.contains(message), "{error}");
        assert!(
            error.contains("supply") || error.contains("use distinct"),
            "{error}"
        );
        let mut source: serde_json::Value =
            serde_json::from_str(&source(name, 1, &angles, "rad")).unwrap();
        source["ops"][1]["args"] =
            json!(qubits.iter().map(|q| json!(["q", q])).collect::<Vec<_>>());
        let source = source.to_string();
        assert!(
            PhirJsonEngine::from_json(&source)
                .unwrap()
                .generate_commands()
                .is_err(),
            "{source}"
        );
        assert!(phir_json_to_module(&source).is_err(), "{source}");
    }
}

#[test]
fn large_broadcasts_preserve_all_targets_in_both_paths() {
    for (name, arity, angles) in [
        ("H", 1, vec![]),
        ("RZZ", 2, vec![0.37]),
        ("Init", 1, vec![]),
        ("Measure", 1, vec![]),
    ] {
        let mut source: serde_json::Value =
            serde_json::from_str(&source(name, arity, &angles, "rad")).unwrap();
        source["ops"][0]["size"] = json!(300);
        let qubits: Vec<_> = (0..300).rev().map(|q| json!(["q", q])).collect();
        source["ops"][1]["args"] = if arity == 1 {
            json!(qubits)
        } else {
            json!(qubits.chunks_exact(arity).collect::<Vec<_>>())
        };
        if name == "Measure" {
            source["ops"][1]["returns"] =
                json!((0..300).map(|_| json!(["m", 0])).collect::<Vec<_>>());
        }
        let source = source.to_string();
        let native = drain_commands(&mut PhirJsonEngine::from_json(&source).unwrap());
        let converted =
            drain_commands(&mut PhirEngine::new(phir_json_to_module(&source).unwrap()).unwrap());
        assert_eq!(native.len(), 300 / arity, "{name}");
        assert_eq!(native, converted, "{name}");
    }
}

#[test]
fn cx_broadcast_with_128_pairs_is_one_command_and_runs() {
    let pairs: Vec<_> = (0..128).map(|i| (2 * i, 2 * i + 1)).collect();
    let args: Vec<_> = pairs
        .iter()
        .map(|&(c, t)| json!([["q", c], ["q", t]]))
        .collect();
    let source = json!({"format":"PHIR/JSON","version":"0.1.0","ops":[
        {"data":"qvar_define","data_type":"qubits","variable":"q","size":256},
        {"qop":"CX","args":args}
    ]})
    .to_string();
    let mut engine = PhirJsonEngine::from_json(&source).unwrap();
    let commands = engine.generate_commands().unwrap();
    assert_eq!(commands.quantum_ops().unwrap(), [Gate::cx(&pairs)]);
    let mut quantum = pecos_engines::StabilizerEngine::new(256);
    let controls: Vec<_> = pairs.iter().map(|&(c, _)| c).collect();
    quantum
        .process(ByteMessage::builder().x(&controls).build())
        .unwrap();
    let outcomes = quantum.process(commands).unwrap();
    engine.handle_measurements(outcomes).unwrap();
    assert!(engine.generate_commands().unwrap().is_empty().unwrap());
    let measured = quantum
        .process(
            ByteMessage::builder()
                .mz(&(0..256).collect::<Vec<_>>())
                .build(),
        )
        .unwrap();
    assert_eq!(measured.outcomes().unwrap(), vec![1; 256]);
}

fn drain_commands(engine: &mut dyn ClassicalEngine) -> Vec<Gate> {
    let mut commands = Vec::new();
    for _ in 0..10 {
        let batch = engine.generate_commands().unwrap();
        if batch.is_empty().unwrap() {
            return commands;
        }
        let batch = individual_gates(&batch);
        let measurements = batch
            .iter()
            .filter(|gate| gate.gate_type == pecos_core::gate_type::GateType::MZ)
            .count();
        if measurements > 0 {
            let mut outcomes = ByteMessage::outcomes_builder();
            outcomes.add_outcomes(&vec![0; measurements]);
            engine.handle_measurements(outcomes.build()).unwrap();
        }
        commands.extend(batch);
    }
    panic!("engine did not finish the broadcast");
}
