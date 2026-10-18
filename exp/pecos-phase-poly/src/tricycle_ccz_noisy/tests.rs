// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::*;
use pecos_core::gate_type::GateType;
use std::collections::BTreeSet;

fn codes() -> [BivariateTricycleCode; 3] {
    [
        (3, [(2, 1), (2, 2)], [(0, 0), (1, 2)], [(1, 0), (2, 1)]),
        (5, [(1, 0), (0, 2)], [(0, 0), (1, 2)], [(1, 0), (1, 3)]),
        (9, [(1, 3), (2, 1)], [(0, 0), (1, 8)], [(2, 4), (2, 6)]),
    ]
    .map(|(m, a, b, c)| BivariateTricycleCode::new(3, m, &a, &b, &c).unwrap())
}

fn noiseless(model: CczModel) {
    for meta_checks in [MetaChecks::Complete, MetaChecks::Paper] {
        // [contract] Random Z projection, followed by deterministic corrected readout.
        // [kill: skip the X correction] Diagnostics must also remain zero.
        // [kill: use H_Z^T or M_Z^T] Orientation is essential to repair and correction.
        for code in codes() {
            let protocol =
                NoisyCcz::new(&code, 0.0, model, NoiseModel::CircuitLevel, meta_checks).unwrap();
            let result = protocol.run(16, 8108).unwrap();
            let residual = result
                .outcomes
                .iter()
                .filter(|o| o.diagnostics.iter().any(|&b| b))
                .count();
            let detector_events = result
                .outcomes
                .iter()
                .filter(|o| {
                    o.readout
                        .as_ref()
                        .is_some_and(|r| r.detectors.iter().any(|&b| b))
                })
                .count();
            let observable_events = result
                .outcomes
                .iter()
                .filter(|o| {
                    o.readout
                        .as_ref()
                        .is_some_and(|r| r.observables.iter().any(|&b| b))
                })
                .count();
            assert_eq!(
                residual, 0,
                "model={model:?}, residual={residual}, detector_events={detector_events}, observable_events={observable_events}"
            );
            assert_eq!(result.counts.repair_failures, 0);
            assert_eq!(result.counts.accepted, 16);
            assert_eq!(result.counts.logical_failures, 0);
            assert_eq!(result.counts.accepted_with_residual_x, 0);
            assert!(
                result
                    .outcomes
                    .iter()
                    .map(|o| &o.syndromes)
                    .collect::<BTreeSet<_>>()
                    .len()
                    >= 2
            );
            for outcome in result.outcomes {
                assert_eq!(outcome.decoder_calls, 0);
                let readout = outcome.readout.unwrap();
                assert!(readout.detectors.iter().all(|b| !b));
                assert!(readout.observables.iter().all(|b| !b));
                assert!(outcome.diagnostics.iter().all(|b| !b));
                assert_eq!(readout.detectors.len(), 3 * code.hx().rank());
                assert_eq!(outcome.diagnostics.len(), 3 * code.hz().rank());
            }
        }
    }
}

#[test]
fn noiseless_exact() {
    noiseless(CczModel::Exact);
}
#[test]
fn noiseless_surrogate() {
    noiseless(CczModel::PaperSurrogate);
}

#[test]
fn forced_measurement_and_data_faults() {
    for meta_checks in [MetaChecks::Complete, MetaChecks::Paper] {
        // [contract] Single recorded-syndrome errors invoke relay-BP and are repaired.
        for code in codes() {
            let n = code.num_qubits();
            for model in [CczModel::Exact, CczModel::PaperSurrogate] {
                for index in [0, 5, n + 1, n + 7, 2 * n + 3, 3 * n - 1] {
                    let faults = [Fault {
                        point: FaultPoint::BeforeAncillaMeasurement,
                        qubit: 3 * n + index,
                        kind: GateType::X,
                    }];
                    let result = NoisyCcz::build(
                        &code,
                        0.0,
                        model,
                        NoiseModel::CircuitLevel,
                        meta_checks,
                        &faults,
                    )
                    .unwrap()
                    .run(4, 71)
                    .unwrap();
                    assert_eq!(
                        result.counts.accepted, 4,
                        "n={n}, ancilla={index}, model={model:?}"
                    );
                    assert_eq!(result.counts.logical_failures, 0);
                    assert_eq!(result.counts.repair_failures, 0);
                    for outcome in result.outcomes {
                        assert_eq!(outcome.decoder_calls, 1);
                        assert!(outcome.diagnostics.iter().all(|b| !b));
                    }
                }
                // [contract] Z data faults always trigger an X-check detector.
                let faults = [Fault {
                    point: FaultPoint::BeforeCcz,
                    qubit: n + 2,
                    kind: GateType::Z,
                }];
                let result = NoisyCcz::build(
                    &code,
                    0.0,
                    model,
                    NoiseModel::CircuitLevel,
                    meta_checks,
                    &faults,
                )
                .unwrap()
                .run(16, 72)
                .unwrap();
                assert_eq!(result.counts.repair_failures, 0);
                assert_eq!(result.counts.accepted, 0);
                assert!(
                    result.outcomes.iter().all(|o| o
                        .readout
                        .as_ref()
                        .unwrap()
                        .detectors
                        .iter()
                        .any(|&b| b))
                );
            }
        }
    }
}

#[test]
fn image_check_even_for_zero_meta_syndrome() {
    // [contract] ker M_Z can strictly contain im H_Z. Reject without decoding.
    for code in codes() {
        let solver = SyndromeSolver::new(&F2Matrix::from_rows(code.hz().rows()));
        let outside = F2Matrix::from_rows(code.meta_z().rows())
            .kernel()
            .into_iter()
            .find(|s| solver.solve(s).is_none())
            .unwrap();
        let faults = outside
            .iter()
            .enumerate()
            .filter_map(|(q, &bit)| {
                (bit != 0).then_some(Fault {
                    point: FaultPoint::BeforeAncillaMeasurement,
                    qubit: 3 * code.num_qubits() + q,
                    kind: GateType::X,
                })
            })
            .collect::<Vec<_>>();
        for model in [CczModel::Exact, CczModel::PaperSurrogate] {
            let result = NoisyCcz::build(
                &code,
                0.0,
                model,
                NoiseModel::CircuitLevel,
                MetaChecks::Paper,
                &faults,
            )
            .unwrap()
            .run(8, 77)
            .unwrap();
            assert_eq!(result.counts.repair_failures, 8);
            assert_eq!(result.counts.accepted, 0);
            for outcome in result.outcomes {
                assert_eq!(outcome.decoder_calls, 0);
                assert!(outcome.readout.is_none());
                assert_eq!(outcome.diagnostics, [] as [bool; 0]);
            }
        }
    }
}

#[test]
fn supports_and_probabilities() {
    for meta_checks in [MetaChecks::Complete, MetaChecks::Paper] {
        // [contract] Each independently sampled surrogate event has five qubits.
        for code in codes() {
            let supports = surrogate_supports(&code);
            // [kill: exchange the other blocks in surrogate_supports]
            let n = code.num_qubits();
            for [a, b, c] in code.cup_product_ccz().concat() {
                let triple = [a, n + b, 2 * n + c];
                for q in triple {
                    assert!(triple.iter().all(|member| supports[q].contains(member)));
                }
            }
            for (q, support) in supports.iter().enumerate() {
                assert_eq!(support.len(), 5);
                assert!(support.contains(&q));
            }
        }
        // [contract] Boundary errors retain the ideal-encoder benchmark error text.
        let code = &codes()[0];
        for p in [-0.1, 1.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                NoisyCcz::new(
                    code,
                    p,
                    CczModel::Exact,
                    NoiseModel::CircuitLevel,
                    meta_checks
                )
                .err()
                .unwrap(),
                "p must be finite and in [0,1]"
            );
        }
        assert!(
            NoisyCcz::new(
                code,
                1.0,
                CczModel::Exact,
                NoiseModel::CircuitLevel,
                meta_checks
            )
            .is_ok()
        );
    }
}

#[test]
fn reproducibility_and_statistical_smoke() {
    for meta_checks in [MetaChecks::Complete, MetaChecks::Paper] {
        let code = &codes()[0];
        for model in [CczModel::Exact, CczModel::PaperSurrogate] {
            // [contract] Decoder state and all RNG streams restart on each run.
            let protocol =
                NoisyCcz::new(code, 0.01, model, NoiseModel::CircuitLevel, meta_checks).unwrap();
            assert_eq!(
                protocol.run(32, 2026).unwrap(),
                protocol.run(32, 2026).unwrap()
            );
            // [contract] Counts-only execution uses the same loop and random streams.
            assert_eq!(
                protocol.run_counts(32, 2026).unwrap(),
                protocol.run(32, 2026).unwrap().counts
            );
            assert_eq!(protocol.run_counts(0, 2026).unwrap(), Counts::default());
            // [contract] Nontrivial acceptance and internally consistent counts.
            let result = NoisyCcz::new(code, 0.001, model, NoiseModel::CircuitLevel, meta_checks)
                .unwrap()
                .run(256, 2027)
                .unwrap();
            let counts = &result.counts;
            assert_eq!(counts.attempted, 256);
            assert!(counts.accepted > 0 && counts.accepted < counts.attempted);
            assert_eq!(
                counts.repair_failures,
                result.outcomes.iter().filter(|o| o.repair_failure).count()
            );
            assert_eq!(
                counts.accepted,
                result.outcomes.iter().filter(|o| o.accepted()).count()
            );
            assert_eq!(
                counts.logical_failures,
                result
                    .outcomes
                    .iter()
                    .filter(|o| o.accepted()
                        && o.readout.as_ref().unwrap().observables.iter().any(|&b| b))
                    .count()
            );
            assert_eq!(
                counts.accepted_with_residual_x,
                result
                    .outcomes
                    .iter()
                    .filter(|o| o.accepted() && o.diagnostics.iter().any(|&b| b))
                    .count()
            );
            assert_eq!(
                counts.accepted_z_clean,
                result
                    .outcomes
                    .iter()
                    .filter(|o| o.accepted() && o.diagnostics.iter().all(|b| !b))
                    .count()
            );
            assert_eq!(
                counts.logical_failures_z_clean,
                result
                    .outcomes
                    .iter()
                    .filter(|o| o.accepted()
                        && o.diagnostics.iter().all(|b| !b)
                        && o.readout.as_ref().unwrap().observables.iter().any(|&b| b))
                    .count()
            );
            assert_eq!(
                counts.accepted_z_clean + counts.accepted_with_residual_x,
                counts.accepted
            );
            assert!(counts.logical_failures_z_clean <= counts.accepted_z_clean);
            assert!(counts.logical_failures_z_clean <= counts.logical_failures);
            assert!(counts.accepted + counts.repair_failures <= counts.attempted);
        }
    }
}

#[test]
fn wide_pauli_channels() {
    // [contract] Generic three- and five-qubit mixtures above global index 255.
    use pecos_quantum::channel::pauli_mixture;
    let qubits = [257, 300, 485];
    let channel = depolarizing3(0.63, qubits);
    let mut full_mixture = TickCircuit::new();
    full_mixture.tick().channel(channel.clone());
    full_mixture.tick().mz(&qubits);
    assert_eq!(
        Program::compile(&full_mixture)
            .unwrap()
            .run_shots(32, 18)
            .unwrap()
            .len(),
        32
    );
    let alternatives = pauli_mixture(&channel).unwrap();
    assert_eq!(alternatives.len(), 64);
    // [kill: reuse the first Pauli digit for the second qubit]
    assert_eq!(
        alternatives
            .iter()
            .map(|(_, pauli)| qubits.map(|q| (pauli.has_x(q), pauli.has_z(q))))
            .collect::<BTreeSet<_>>()
            .len(),
        64
    );
    assert!((alternatives[0].0 - 0.37).abs() < 1e-14);
    assert!(
        alternatives[1..]
            .iter()
            .all(|(p, _)| (*p - 0.01).abs() < 1e-14)
    );
    // Force every alternative through generic compilation, checking both axes.
    let ChannelExpr::MixedUnitary(terms) = channel else {
        unreachable!()
    };
    for ((_, term), (_, pauli)) in terms.into_iter().zip(alternatives) {
        for x_basis in [false, true] {
            let mut circuit = TickCircuit::new();
            if x_basis {
                circuit.tick().px(&qubits);
            } else {
                circuit.tick().pz(&qubits);
            }
            circuit
                .tick()
                .channel(ChannelExpr::MixedUnitary(vec![(1.0, term.clone())]));
            if x_basis {
                circuit.tick().mx(&qubits);
            } else {
                circuit.tick().mz(&qubits);
            }
            let shot = Program::compile(&circuit)
                .unwrap()
                .run_shots(1, 19)
                .unwrap()
                .remove(0);
            assert_eq!(
                shot.records,
                qubits.map(|q| if x_basis {
                    pauli.has_z(q)
                } else {
                    pauli.has_x(q)
                })
            );
        }
    }
    let support = [256, 257, 300, 400, 485];
    for (p, expected) in [(0.0, false), (1.0, true)] {
        let mut circuit = TickCircuit::new();
        circuit.tick().px(&support);
        circuit.tick().channel(correlated_z(p, &support));
        circuit.tick().mx(&support);
        assert_eq!(
            Program::compile(&circuit)
                .unwrap()
                .run_shots(1, 20)
                .unwrap()[0]
                .records,
            vec![expected; 5]
        );
    }
}

#[test]
fn residual_x_is_diagnostic_only() {
    for meta_checks in [MetaChecks::Complete, MetaChecks::Paper] {
        // [contract] An accepted surrogate shot can retain a residual X syndrome.
        let code = &codes()[0];
        let faults = [Fault {
            point: FaultPoint::BeforeCcz,
            qubit: 0,
            kind: GateType::X,
        }];
        let result = NoisyCcz::build(
            code,
            0.0,
            CczModel::PaperSurrogate,
            NoiseModel::CircuitLevel,
            meta_checks,
            &faults,
        )
        .unwrap()
        .run(16, 90)
        .unwrap();
        assert_eq!(result.counts.accepted, 16);
        assert_eq!(result.counts.accepted_with_residual_x, 16);
        assert_eq!(result.counts.accepted_z_clean, 0);
        assert_eq!(result.counts.logical_failures, 0);
    }
}

#[test]
fn exact_logical_pivot_x_is_not_z_clean() {
    for meta_checks in [MetaChecks::Complete, MetaChecks::Paper] {
        // [contract] A physical pivot X can produce tensor-undo readout artifacts;
        // zero Z diagnostics exclude every accepted shot carrying this fault.
        let code = &codes()[0];
        let pivot = Encoder::new(code).logical_pivots[0];
        let faults = [Fault {
            point: FaultPoint::BeforeCcz,
            qubit: pivot,
            kind: GateType::X,
        }];
        let result = NoisyCcz::build(
            code,
            0.0,
            CczModel::Exact,
            NoiseModel::CircuitLevel,
            meta_checks,
            &faults,
        )
        .unwrap()
        .run(500, 2026)
        .unwrap();
        assert_eq!(result.counts.repair_failures, 0);
        assert_eq!(result.outcomes.len(), 500);
        assert!(result.counts.accepted > 0);
        assert!(
            result.counts.logical_failures > 0,
            "logical pivot={pivot}, counts={:?}",
            result.counts
        );
        assert!(
            result
                .outcomes
                .iter()
                .all(|o| o.diagnostics.iter().any(|&b| b))
        );
        assert!(
            result
                .outcomes
                .iter()
                .filter(|o| o.accepted())
                .all(|o| o.diagnostics.iter().any(|&b| b))
        );
        assert_eq!(result.counts.accepted_z_clean, 0);
        assert_eq!(result.counts.logical_failures_z_clean, 0);
    }
}

#[test]
fn preparation_schedule_and_noise() {
    // [contract] One noisy preparation per qubit, four disjoint CX layers, and
    // no idle noise. Every Tanner edge is present in each of the three blocks.
    for code in codes() {
        let n = code.num_qubits();
        let m_z = code.hz().num_checks();
        let total = 3 * (n + m_z);
        let circuit = preparation(&code, 0.01, total, NoiseModel::CircuitLevel).unwrap();
        let mut prepared = vec![0; total];
        let mut pairs = BTreeSet::new();
        let mut cx_ticks = BTreeSet::new();
        let mut channels = 0;
        for (tick, row) in circuit.iter_ticks() {
            let mut touched = BTreeSet::new();
            for gate in row.iter_gate_batches() {
                let gate = gate.as_gate();
                match gate.gate_type {
                    GateType::PX | GateType::PZ => {
                        for q in &gate.qubits {
                            prepared[q.0] += 1;
                            assert_eq!(
                                gate.gate_type,
                                if q.0 < 3 * n {
                                    GateType::PX
                                } else {
                                    GateType::PZ
                                }
                            );
                        }
                    }
                    GateType::CX => {
                        cx_ticks.insert(tick);
                        for pair in gate.qubits.as_chunks::<2>().0 {
                            assert!(touched.insert(pair[0]));
                            assert!(touched.insert(pair[1]));
                            assert!(pairs.insert((pair[0].0, pair[1].0)));
                        }
                    }
                    GateType::Channel => channels += 1,
                    kind => panic!("unexpected preparation gate {kind:?}"),
                }
            }
        }
        assert_eq!(prepared, vec![1; total]);
        assert_eq!(cx_ticks.len(), 4);
        assert_eq!(pairs.len(), 12 * n);
        assert_eq!(channels, total + 12 * n + 3 * m_z);
        for block in 0..3 {
            for (check, row) in code.hz().rows().iter().enumerate() {
                for (data, &bit) in row.iter().enumerate() {
                    assert_eq!(
                        pairs.contains(&(block * n + data, 3 * n + block * m_z + check)),
                        bit != 0
                    );
                }
            }
        }
    }
}

#[test]
fn meta_check_homology_dimension() {
    // [contract] Boundary-map meta checks omit k image constraints.
    for code in codes() {
        let rows = meta_check_rows(&code, MetaChecks::Paper);
        let kernel_dimension =
            code.hz().num_checks() - F2Matrix::from_rows(rows).row_reduce().1.len();
        assert_eq!(
            kernel_dimension - code.hz().rank(),
            code.num_logical_qubits()
        );
        assert_eq!(kernel_dimension - code.hz().rank(), 3);
    }
}

fn gates_at(circuit: &TickCircuit, tick: usize) -> Vec<&pecos_core::Gate> {
    circuit
        .get_tick(tick)
        .unwrap()
        .gate_batches()
        .iter()
        .collect()
}

fn assert_channel_support(gate: &pecos_core::Gate, support: &[usize]) {
    assert_eq!(gate.gate_type, GateType::Channel);
    let mut actual = gate.qubits.iter().map(|q| q.0).collect::<Vec<_>>();
    actual.sort_unstable();
    let mut expected = support.to_vec();
    expected.sort_unstable();
    assert_eq!(actual, expected);
}

fn assert_binary_channel(gate: &pecos_core::Gate, support: &[usize], p: f64, x: bool, z: bool) {
    use pecos_quantum::channel::pauli_mixture;
    assert_channel_support(gate, support);
    let alternatives = pauli_mixture(gate.channel.as_ref().unwrap()).unwrap();
    assert_eq!(alternatives.len(), 2);
    let mut seen = BTreeSet::new();
    for (weight, pauli) in alternatives {
        let mask = support
            .iter()
            .map(|&q| (pauli.has_x(q), pauli.has_z(q)))
            .collect::<Vec<_>>();
        let identity = mask.iter().all(|&(a, b)| !a && !b);
        if !identity {
            assert_eq!(mask, vec![(x, z); support.len()]);
        }
        assert!((weight - if identity { 1.0 - p } else { p }).abs() < 1e-14);
        assert!(seen.insert(mask));
    }
}

fn assert_depolarizing_channel(gate: &pecos_core::Gate, support: &[usize], p: f64) {
    use pecos_quantum::channel::pauli_mixture;
    assert_channel_support(gate, support);
    let alternatives = pauli_mixture(gate.channel.as_ref().unwrap()).unwrap();
    let cardinality = 1_usize << (2 * support.len());
    assert_eq!(alternatives.len(), cardinality);
    let mut masks = BTreeSet::new();
    let nonidentity_count = u32::try_from(cardinality - 1).unwrap();
    for (weight, pauli) in alternatives {
        let mask = support
            .iter()
            .map(|&q| (pauli.has_x(q), pauli.has_z(q)))
            .collect::<Vec<_>>();
        let identity = mask.iter().all(|&(x, z)| !x && !z);
        let expected = if identity {
            1.0 - p
        } else {
            p / f64::from(nonidentity_count)
        };
        assert!((weight - expected).abs() < 1e-14);
        assert!(masks.insert(mask), "duplicate depolarizing Pauli");
    }
    // There are only 4^k possible X/Z masks on k qubits, so distinctness plus
    // cardinality proves this is the complete Pauli set, independently of digits.
    assert_eq!(masks.len(), cardinality);
}

#[test]
fn prefix_noise_content_and_placement() {
    // [contract] Full Pauli content and exact timing, including prep and readout.
    // [kill] Data-prep Z -> X; pre-MZ X -> Z; Depolarizing2 -> 1q Depolarizing.
    let p = 0.037;
    for noise_model in [NoiseModel::CircuitLevel, NoiseModel::Paper] {
        for code in codes() {
            let n = code.num_qubits();
            let m_z = code.hz().num_checks();
            let total = 3 * (n + m_z);
            let mut circuit = preparation(&code, p, total, noise_model).unwrap();
            circuit.tick().mz(&(3 * n..total).collect::<Vec<_>>());
            let prep = gates_at(&circuit, 0);
            assert_eq!(prep.len(), 2);
            for gate in prep {
                let expected = match gate.gate_type {
                    GateType::PX => (0..3 * n).collect::<Vec<_>>(),
                    GateType::PZ => (3 * n..total).collect::<Vec<_>>(),
                    kind => panic!("unexpected prep {kind:?}"),
                };
                assert_eq!(
                    gate.qubits.iter().map(|q| q.0).collect::<Vec<_>>(),
                    expected
                );
            }
            let noise = gates_at(&circuit, 1);
            match noise_model {
                NoiseModel::CircuitLevel => {
                    assert_eq!(noise.len(), total);
                    for (q, gate) in noise.into_iter().enumerate() {
                        assert_binary_channel(gate, &[q], p, q >= 3 * n, q < 3 * n);
                    }
                }
                NoiseModel::Paper => {
                    assert_eq!(noise.len(), total - 3 * n);
                    for (q, gate) in (3 * n..total).zip(noise) {
                        assert_depolarizing_channel(gate, &[q], p);
                    }
                }
            }
            assert_eq!(circuit.num_ticks(), 12);
            for color in 0..4 {
                let cx_tick = 2 + 2 * color;
                let cx = gates_at(&circuit, cx_tick);
                assert!(cx.iter().all(|g| g.gate_type == GateType::CX));
                let pairs = cx
                    .iter()
                    .flat_map(|g| g.qubits.as_chunks::<2>().0)
                    .map(|pair| [pair[0].0, pair[1].0])
                    .collect::<Vec<_>>();
                let noise = gates_at(&circuit, cx_tick + 1);
                assert_eq!(noise.len(), pairs.len());
                for (gate, pair) in noise.into_iter().zip(pairs) {
                    assert_depolarizing_channel(gate, &pair, p);
                }
            }
            let noise = gates_at(&circuit, 10);
            assert_eq!(noise.len(), 3 * m_z);
            for (q, gate) in (3 * n..total).zip(noise) {
                match noise_model {
                    NoiseModel::CircuitLevel => assert_binary_channel(gate, &[q], p, true, false),
                    NoiseModel::Paper => assert_depolarizing_channel(gate, &[q], p),
                }
            }
            let measurement = gates_at(&circuit, 11);
            assert_eq!(measurement.len(), 1);
            assert_eq!(measurement[0].gate_type, GateType::MZ);
            assert_eq!(
                measurement[0]
                    .qubits
                    .iter()
                    .map(|q| q.0)
                    .collect::<Vec<_>>(),
                (3 * n..total).collect::<Vec<_>>()
            );
        }
    }
}

#[test]
fn suffix_noise_content_and_placement() {
    // [contract] Exact CCZ depolarization, surrogate weights, and final dephasing.
    // [kill] Remove depolarizing3; remove/change readout Z; set surrogate p=0.
    let p = 0.037;
    for noise_model in [NoiseModel::CircuitLevel, NoiseModel::Paper] {
        for code in codes() {
            let n = code.num_qubits();
            for model in [CczModel::Exact, CczModel::PaperSurrogate] {
                let mut circuit = TickCircuit::new();
                ccz_stage(&mut circuit, &code, p, model);
                match model {
                    CczModel::Exact => {
                        assert_eq!(circuit.num_ticks(), 4);
                        for (layer, triples) in code.cup_product_ccz().iter().enumerate() {
                            let gates = gates_at(&circuit, 2 * layer);
                            assert!(gates.iter().all(|g| g.gate_type == GateType::CCZ));
                            let expected = triples
                                .iter()
                                .map(|&[a, b, c]| [a, n + b, 2 * n + c])
                                .collect::<Vec<_>>();
                            let actual = gates
                                .iter()
                                .flat_map(|g| g.qubits.as_chunks::<3>().0)
                                .map(|qs| qs.map(|q| q.0))
                                .collect::<Vec<_>>();
                            assert_eq!(actual, expected);
                            let noise = gates_at(&circuit, 2 * layer + 1);
                            assert_eq!(noise.len(), triples.len());
                            for (gate, triple) in noise.into_iter().zip(expected) {
                                assert_depolarizing_channel(gate, &triple, p);
                            }
                        }
                    }
                    CczModel::PaperSurrogate => {
                        assert_eq!(circuit.num_ticks(), 3 * n);
                        for (tick, support) in surrogate_supports(&code).iter().enumerate() {
                            let noise = gates_at(&circuit, tick);
                            assert_eq!(noise.len(), 1);
                            assert_binary_channel(noise[0], support, p, false, true);
                        }
                    }
                }
                let readout_tick = circuit.num_ticks();
                readout(&mut circuit, &code, p, model, noise_model).unwrap();
                let noise = gates_at(&circuit, readout_tick);
                assert_eq!(noise.len(), 3 * n);
                for (q, gate) in noise.into_iter().enumerate() {
                    match noise_model {
                        NoiseModel::CircuitLevel => {
                            assert_binary_channel(gate, &[q], p, false, true);
                        }
                        NoiseModel::Paper => assert_depolarizing_channel(gate, &[q], p),
                    }
                }
                // The rest of readout is ideal; no further channels or preparations.
                for tick in readout_tick + 1..circuit.num_ticks() {
                    assert!(gates_at(&circuit, tick).iter().all(|g| !matches!(
                        g.gate_type,
                        GateType::Channel | GateType::PX | GateType::PZ
                    )));
                }
            }
        }
    }
}

#[test]
fn complete_meta_checks_span_left_kernel() {
    for code in codes() {
        let protocol = NoisyCcz::new(
            &code,
            0.0,
            CczModel::Exact,
            NoiseModel::CircuitLevel,
            MetaChecks::Complete,
        )
        .unwrap();
        let meta = protocol.meta_matrix();
        let paper = code.meta_z().rows();
        let rows = meta
            .rows()
            .into_iter()
            .map(|row| row.to_vec())
            .collect::<Vec<_>>();
        assert_eq!(&rows[..paper.len()], paper);
        assert_eq!(rows.len(), paper.len() + 3);
        assert_eq!(
            F2Matrix::from_rows(rows.clone()).row_reduce().1.len(),
            code.hz().num_checks() - code.hz().rank()
        );
        let hz = Array2::from_shape_vec(
            (code.hz().num_checks(), code.num_qubits()),
            code.hz().rows().concat(),
        )
        .unwrap();
        assert!(meta.dot(&hz).iter().all(|bit| bit % 2 == 0));
    }
}

#[test]
fn complete_meta_checks_detect_kernel_but_not_image_error() {
    for code in codes() {
        let solver = SyndromeSolver::new(&F2Matrix::from_rows(code.hz().rows()));
        let outside = F2Matrix::from_rows(code.meta_z().rows())
            .kernel()
            .into_iter()
            .find(|s| solver.solve(s).is_none())
            .unwrap();
        for meta_checks in [MetaChecks::Paper, MetaChecks::Complete] {
            let protocol = NoisyCcz::new(
                &code,
                0.0,
                CczModel::Exact,
                NoiseModel::CircuitLevel,
                meta_checks,
            )
            .unwrap();
            let sigma = protocol
                .meta_matrix()
                .dot(&Array1::from_vec(outside.clone()));
            assert_eq!(
                sigma.iter().any(|bit| bit % 2 != 0),
                meta_checks == MetaChecks::Complete
            );
        }
    }
}
