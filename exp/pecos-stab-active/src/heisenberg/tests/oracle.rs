// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.
use super::*;
use pecos_simulators::state_vector_test_utils::normalized_z_projection;

fn rotation<S: ArbitraryRotationGateable>(state: &mut S, gate: &Gate) {
    let angle = gate.angles[0];
    let qs = &gate.qubits;
    let pairs: Vec<_> = qs.as_chunks::<2>().0.iter().map(|p| (p[0], p[1])).collect();
    match gate.gate_type {
        GateType::RX => {
            state.rx(angle, qs);
        }
        GateType::RY => {
            state.ry(angle, qs);
        }
        GateType::RZ => {
            state.rz(angle, qs);
        }
        GateType::RXX => {
            state.rxx(angle, &pairs);
        }
        GateType::RYY => {
            state.ryy(angle, &pairs);
        }
        GateType::RZZ => {
            state.rzz(angle, &pairs);
        }
        _ => panic!("not a physical rotation"),
    }
}

fn physical_noise<S: CliffordGateable>(state: &mut S, qubits: &[usize], bits: &[bool]) {
    for (i, &q) in qubits.iter().enumerate() {
        if bits[2 * i] {
            state.x(&[QubitId(q)]);
        }
        if bits[2 * i + 1] {
            state.z(&[QubitId(q)]);
        }
    }
}

struct PhysicalTrace {
    state: StabActive,
    dense: Option<DenseStateVec>,
    probabilities: Vec<f64>,
    outcomes: Vec<bool>,
    records: Vec<bool>,
}

impl PhysicalTrace {
    fn measure(&mut self, gate: &Gate, rng: &mut PecosRng) {
        let x_basis = matches!(gate.gate_type, GateType::MX | GateType::PX);
        for q in &gate.qubits {
            if x_basis {
                self.state.h(&[*q]);
                if let Some(dense) = &mut self.dense {
                    dense.h(&[*q]);
                }
            }
            let probability = self
                .state
                .probability_one(&self.state.parts(&[(q.index(), PauliKindForDecomp::Z)]));
            let outcome = if probability <= 1e-6 {
                false
            } else if probability >= 1.0 - 1e-6 {
                true
            } else {
                rng.next_bool_fast()
            };
            self.probabilities.push(probability);
            self.outcomes.push(outcome);
            assert_eq!(self.state.mz_forced(q.index(), outcome).outcome, outcome);
            if let Some(dense) = &mut self.dense {
                let vector = dense.state();
                let expected: f64 = vector
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i & (1 << q.index()) != 0)
                    .map(|(_, a)| a.norm_sqr())
                    .sum();
                assert!(
                    (expected - probability).abs() < 1e-10,
                    "dense={expected} physical={probability}"
                );
                let projected = normalized_z_projection(&vector, q.index(), outcome, "HIR oracle");
                *dense = DenseStateVec::from_complex_state(&projected, PecosRng::seed_from_u64(0));
            }
            if x_basis {
                self.state.h(&[*q]);
                if let Some(dense) = &mut self.dense {
                    dense.h(&[*q]);
                }
            }
            if gate.gate_type.consumes_measurement_record() {
                self.records.push(outcome);
            }
            if outcome && matches!(gate.gate_type, GateType::PZ | GateType::PX | GateType::MPZ) {
                if x_basis {
                    self.state.z(&[*q]);
                    if let Some(dense) = &mut self.dense {
                        dense.z(&[*q]);
                    }
                } else {
                    self.state.x(&[*q]);
                    if let Some(dense) = &mut self.dense {
                        dense.x(&[*q]);
                    }
                }
            }
        }
    }
}

pub(super) fn compare(circuit: &TickCircuit, seed: u64) {
    let program = HeisenbergProgram::compile(circuit).unwrap();

    let n = program.num_qubits;
    let mut rng = PecosRng::seed_from_u64(seed);
    let mut noise = vec![false; program.num_noise_symbols];
    for channel in &program.noise_channels {
        channel.sample(&mut rng, &mut noise);
    }
    let mut trace = PhysicalTrace {
        state: StabActive::with_seed(n, seed),
        dense: (n <= 6).then(|| DenseStateVec::with_seed(n, seed)),
        probabilities: Vec::new(),
        outcomes: Vec::new(),
        records: Vec::new(),
    };
    let mut noise_index = 0;
    for gate in circuit.iter_gate_batches() {
        if dispatch::named_clifford(gate.gate_type) {
            dispatch::apply_clifford(&mut trace.state, gate.gate_type, &gate.qubits);
            if let Some(dense) = &mut trace.dense {
                dispatch::apply_clifford(dense, gate.gate_type, &gate.qubits);
            }
        } else if dispatch::rotation_axis(gate.gate_type).is_some() {
            rotation(&mut trace.state, gate.as_gate());
            if let Some(dense) = &mut trace.dense {
                rotation(dense, gate.as_gate());
            }
        } else {
            match gate.gate_type {
                GateType::Channel => {
                    let channel = &program.noise_channels[noise_index];
                    let bits = &noise
                        [channel.first_symbol..channel.first_symbol + 2 * channel.qubits.len()];
                    physical_noise(&mut trace.state, &channel.qubits, bits);
                    if let Some(dense) = &mut trace.dense {
                        physical_noise(dense, &channel.qubits, bits);
                    }
                    noise_index += 1;
                }
                GateType::MX | GateType::MZ | GateType::MPZ | GateType::PZ | GateType::PX => {
                    trace.measure(gate.as_gate(), &mut rng);
                }
                GateType::I | GateType::Idle | GateType::TrackedPauliMeta => {}
                _ => panic!("unexpected gate"),
            }
        }
    }
    let shot = program.execute(
        StabActive::with_seed(n, seed),
        &noise,
        |state, pauli, negative, symbol, _| {
            let parts = pecos_stab_tn::stab_mps::coordinate_tableau::decompose(
                state.structure.tableau(),
                state.structure.active(),
                pauli.factors(),
                negative,
            );
            let probability = state.probability_one(&parts);
            assert!(
                (probability - trace.probabilities[symbol]).abs() < 1e-10,
                "seed {seed} symbol {symbol}: compiled {probability}, physical {}",
                trace.probabilities[symbol]
            );
            Some(trace.outcomes[symbol])
        },
    );
    assert_eq!(shot.records, trace.records, "seed {seed}");
    assert_eq!(
        shot.peak_active_width,
        trace.state.peak_active_width(),
        "seed {seed}"
    );
    let plan = program.plan(26).unwrap();
    let planned = plan.sampler().fixed_noise_observed(
        seed,
        &noise,
        |probability, symbol, _| {
            assert!((probability - trace.probabilities[symbol]).abs() < 1e-10);
            Some(trace.outcomes[symbol])
        },
        |_, _, _| {},
    );
    assert_eq!(planned, shot);
}

fn index(rng: &mut PecosRng, n: usize) -> usize {
    usize::try_from(rng.next_u64() % u64::try_from(n).unwrap()).unwrap()
}

#[test]
fn random_physical_records_probabilities_dense_and_width() {
    let kinds = [
        GateType::H,
        GateType::F,
        GateType::Fdg,
        GateType::X,
        GateType::Y,
        GateType::Z,
        GateType::SX,
        GateType::SXdg,
        GateType::SY,
        GateType::SYdg,
        GateType::SZ,
        GateType::SZdg,
        GateType::CX,
        GateType::CY,
        GateType::CZ,
        GateType::SXX,
        GateType::SXXdg,
        GateType::SYY,
        GateType::SYYdg,
        GateType::SZZ,
        GateType::SZZdg,
        GateType::SWAP,
        GateType::RX,
        GateType::RY,
        GateType::RZ,
        GateType::RXX,
        GateType::RYY,
        GateType::RZZ,
        GateType::MZ,
        GateType::MX,
        GateType::MPZ,
        GateType::PZ,
        GateType::PX,
        GateType::Channel,
    ];
    let mut rng = PecosRng::seed_from_u64(19381);
    for n in 1..=8 {
        for trial in 0..30 {
            let mut circuit = TickCircuit::new();
            circuit.tick().pz(&(0..n).collect::<Vec<_>>());
            for _ in 0..58 {
                let kind = loop {
                    let k = kinds[index(&mut rng, kinds.len())];
                    if k.quantum_arity() <= n {
                        break k;
                    }
                };
                let q = index(&mut rng, n);
                let r = (q + 1 + index(&mut rng, n.saturating_sub(1).max(1))) % n;
                let gate = if kind == GateType::Channel {
                    if n > 1 && rng.next_bool_fast() {
                        Gate::channel(channel::Depolarizing2(rng.next_f64(), q, r))
                    } else {
                        Gate::channel(channel::PauliChannel(
                            rng.next_f64() / 3.0,
                            rng.next_f64() / 3.0,
                            rng.next_f64() / 3.0,
                            q,
                        ))
                    }
                } else {
                    let qs = if kind.quantum_arity() == 2 {
                        vec![q, r]
                    } else {
                        vec![q]
                    };
                    let angles = if kind.angle_arity() == 0 {
                        vec![]
                    } else {
                        vec![Angle64::from_radians(6.0 * rng.next_f64() - 3.0)]
                    };
                    make_gate(kind, &qs, &angles)
                };
                append(&mut circuit, gate);
            }
            circuit.tick().mz(&(0..n).collect::<Vec<_>>());
            // Raw append deliberately used nonordinal IDs; assign final batch
            // disjoint IDs too (the TickCircuit builder's counter is independent).
            let tick = circuit.num_ticks() - 1;
            let mut gate = circuit.get_tick(tick).unwrap().gate_batches()[0].clone();
            gate.meas_ids = (0..n).map(|q| MeasId::from_raw(1000 + q)).collect();
            circuit
                .get_tick_mut(tick)
                .unwrap()
                .replace_gate_batch(0, gate)
                .unwrap();
            compare(&circuit, u64::try_from(n * 100 + trial).unwrap());
        }
    }
}
