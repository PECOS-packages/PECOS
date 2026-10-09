// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0.

use super::*;
use num_complex::Complex64;
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StateVec};
use std::collections::BTreeSet;

fn apply_native(sim: &mut StateVec, kind: GateType, angles: &[Angle64], qs: &[QubitId]) {
    let pair = if qs.len() == 2 {
        vec![(qs[0], qs[1])]
    } else {
        vec![]
    };
    match kind {
        GateType::I | GateType::Idle | GateType::TrackedPauliMeta => {}
        GateType::X => {
            sim.x(qs);
        }
        GateType::Y => {
            sim.y(qs);
        }
        GateType::Z => {
            sim.z(qs);
        }
        GateType::SZ => {
            sim.sz(qs);
        }
        GateType::SZdg => {
            sim.szdg(qs);
        }
        GateType::T => {
            sim.t(qs);
        }
        GateType::Tdg => {
            sim.tdg(qs);
        }
        GateType::CX => {
            sim.cx(&pair);
        }
        GateType::CZ => {
            sim.cz(&pair);
        }
        GateType::RZ => {
            sim.rz(angles[0], qs);
        }
        GateType::RZZ => {
            sim.rzz(angles[0], &pair);
        }
        GateType::SZZ => {
            sim.szz(&pair);
        }
        GateType::SZZdg => {
            sim.szzdg(&pair);
        }
        _ => panic!("not a native unitary: {kind:?}"),
    }
}

// Independently build native gate matrices from StateVec, including actual
// RZ/RZZ/SZZ implementations (never the runner's parity decomposition).
fn native_gate(state: &mut [Complex64], gate: &Gate, qs: &[QubitId]) {
    let width = qs.len();
    let local_qubits: Vec<_> = (0..width).map(QubitId).collect();
    let columns: Vec<_> = (0..1 << width)
        .map(|input| {
            let mut sim = StateVec::new(width);
            for q in 0..width {
                if input & (1 << q) != 0 {
                    sim.x(&[QubitId(q)]);
                }
            }
            apply_native(&mut sim, gate.gate_type, &gate.angles, &local_qubits);
            sim.state()
        })
        .collect();
    let old = state.to_vec();
    let mask = qs.iter().fold(0, |m, q| m | (1 << q.0));
    state.fill(Complex64::new(0.0, 0.0));
    for (i, amplitude) in old.into_iter().enumerate() {
        let local = qs
            .iter()
            .enumerate()
            .fold(0, |m, (j, q)| m | (((i >> q.0) & 1) << j));
        for (output, coefficient) in columns[local].iter().enumerate() {
            let destination = qs
                .iter()
                .enumerate()
                .fold(i & !mask, |m, (j, q)| m | (((output >> j) & 1) << q.0));
            state[destination] += amplitude * coefficient;
        }
    }
}

fn project(state: &mut Vec<Complex64>, q: QubitId, is_x: bool, outcome: bool) {
    let (probability, projected) = dense_projection(state, &[q], is_x, outcome);
    assert!(probability > 1e-10, "recorded impossible outcome on {q:?}");
    *state = projected;
}

fn replay(circuit: &TickCircuit, records: &[bool], n: usize) -> Vec<Complex64> {
    let mut state = vec![Complex64::new(0.0, 0.0); 1 << n];
    state[0] = Complex64::new(1.0, 0.0);
    let mut record = records.iter();
    for batch in circuit.iter_gate_batches() {
        let gate = batch.as_gate();
        match gate.gate_type {
            GateType::MZ | GateType::MPZ | GateType::MX => {
                for &q in &gate.qubits {
                    let outcome = *record.next().unwrap();
                    project(&mut state, q, gate.gate_type == GateType::MX, outcome);
                    if gate.gate_type == GateType::MPZ && outcome {
                        dense_gate(&mut state, 0, &[q]);
                    }
                }
            }
            GateType::PZ | GateType::PX => {
                for &q in &gate.qubits {
                    // Mid-circuit resets follow a recorded MZ, so their internal
                    // Z measurements are deterministic. Initial resets are too.
                    let (p, _) = dense_projection(&state, &[q], false, false);
                    assert!(p < 1e-10 || p > 1.0 - 1e-10);
                    let outcome = p < 0.5;
                    project(&mut state, q, false, outcome);
                    if outcome {
                        dense_gate(&mut state, 0, &[q]);
                    }
                    if gate.gate_type == GateType::PX {
                        dense_gate(&mut state, 12, &[q]);
                    }
                }
            }
            GateType::Channel => {
                // Generated channels have one probability-one symbolic unitary.
                // Use the core unitary conversion, independently of pauli_mixture.
                let pecos_core::ChannelExpr::MixedUnitary(ops) = gate.channel.as_ref().unwrap()
                else {
                    panic!("expected mixture")
                };
                let pauli = ops
                    .iter()
                    .find(|(p, _)| *p > 0.5)
                    .unwrap()
                    .1
                    .clone()
                    .try_to_pauli_string()
                    .unwrap();
                for (p, q) in pauli.iter_pairs() {
                    match p {
                        pecos_core::Pauli::I => {}
                        pecos_core::Pauli::X => dense_gate(&mut state, 0, &[q]),
                        pecos_core::Pauli::Y => dense_gate(&mut state, 1, &[q]),
                        pecos_core::Pauli::Z => dense_gate(&mut state, 2, &[q]),
                    }
                }
            }
            _ => {
                for qs in gate.qubits.chunks_exact(gate.gate_type.quantum_arity()) {
                    native_gate(&mut state, gate, qs);
                }
            }
        }
    }
    assert!(record.next().is_none());
    state
}

const KINDS: [GateType; 22] = [
    GateType::I,
    GateType::Idle,
    GateType::TrackedPauliMeta,
    GateType::X,
    GateType::Y,
    GateType::Z,
    GateType::SZ,
    GateType::SZdg,
    GateType::T,
    GateType::Tdg,
    GateType::CX,
    GateType::CZ,
    GateType::RZ,
    GateType::RZZ,
    GateType::SZZ,
    GateType::SZZdg,
    GateType::PZ,
    GateType::PX,
    GateType::MZ,
    GateType::MX,
    GateType::MPZ,
    GateType::Channel,
];

#[test]
fn random_tick_circuits_match_dense_oracle() {
    let mut rng = PecosRng::seed_from_u64(0x2610_0681);
    let mut skipped = 0;
    let mut shots = 0;
    let mut covered = BTreeSet::new();
    let mut rotations = BTreeSet::new();
    for n in 1..=6 {
        for trial in 0..64 {
            let mut circuit = TickCircuit::new();
            circuit.tick().px(&(0..n).collect::<Vec<_>>());
            for _ in 0..48 {
                let kind = loop {
                    let kind = KINDS[index(&mut rng, KINDS.len())];
                    if kind.quantum_arity() <= n {
                        break kind;
                    }
                };
                let mut qs: Vec<_> = (0..n).map(QubitId).collect();
                for j in (1..n).rev() {
                    qs.swap(j, index(&mut rng, j + 1));
                }
                let arity = kind.quantum_arity();
                let batch_len = arity * (1 + index(&mut rng, n / arity));
                qs.truncate(batch_len);
                let k = u8::try_from(index(&mut rng, 8)).unwrap();
                let angles = if kind.angle_arity() == 1 {
                    vec![Angle64::from_turns(f64::from(k) / 8.0)]
                } else {
                    vec![]
                };
                if matches!(kind, GateType::PZ | GateType::PX) {
                    append(&mut circuit, Gate::mz(&qs));
                }
                let gate = if kind == GateType::Channel {
                    let q = qs[0];
                    Gate::channel(match index(&mut rng, 3) {
                        0 => channel::BitFlip(1.0, q),
                        1 => channel::BitPhaseFlip(1.0, q),
                        _ => channel::Dephasing(1.0, q),
                    })
                } else if kind == GateType::Idle {
                    Gate::new(kind, angles, vec![2.0], qs)
                } else {
                    Gate::with_angles(kind, angles, qs)
                };
                append(&mut circuit, gate);
            }
            let program = compile(&circuit).unwrap();
            let mut sim = PhasePoly::with_seed(n, trial);
            let mut noise = PecosRng::seed_from_u64(123);
            shots += 1;
            let result = match program.run_shot(&mut sim, &mut noise) {
                Ok(result) => result,
                Err(RunError::Measurement { .. }) => {
                    skipped += 1;
                    continue;
                }
                Err(e) => panic!("unexpected failure: {e}"),
            };
            assert_state(&sim, &replay(&circuit, &result.records, n));
            for gate in circuit.iter_gate_batches() {
                covered.insert(gate.gate_type);
                if matches!(gate.gate_type, GateType::RZ | GateType::RZZ) {
                    rotations.insert((gate.gate_type, gate.angles[0]));
                }
            }
        }
    }
    assert!(skipped < shots / 2, "compatible shots must be the majority");
    for kind in KINDS {
        assert!(covered.contains(&kind), "uncovered {kind:?}");
    }
    for kind in [GateType::RZ, GateType::RZZ] {
        for k in 0_u8..8 {
            assert!(rotations.contains(&(kind, Angle64::from_turns(f64::from(k) / 8.0))));
        }
    }
    println!(
        "TickCircuit dense oracle: shots={shots}, incompatible skipped={skipped}, compatible={}",
        shots - skipped
    );
}

#[test]
fn all_phase_rotation_identities_match_statevec() {
    for kind in [GateType::RZ, GateType::RZZ, GateType::SZZ, GateType::SZZdg] {
        for k in 0_u8..8 {
            let mut circuit = TickCircuit::new();
            circuit.tick().px(&[0, 1, 2, 3]);
            circuit.tick().t(&[0, 2]);
            circuit.tick().cx(&[(0, 1), (2, 3)]);
            let angles = if kind.angle_arity() == 1 {
                vec![Angle64::from_turns(f64::from(k) / 8.0)]
            } else {
                vec![]
            };
            append(
                &mut circuit,
                Gate::with_angles(kind, angles, (0..4).map(QubitId).collect::<Vec<_>>()),
            );
            let mut sim = PhasePoly::with_seed(4, 0);
            let shot = compile(&circuit)
                .unwrap()
                .run_shot(&mut sim, &mut PecosRng::seed_from_u64(1))
                .unwrap();
            assert_state(&sim, &replay(&circuit, &shot.records, 4));
        }
    }
}

#[test]
fn unrecorded_resets_on_entangled_states() {
    for kind in [GateType::PZ, GateType::PX] {
        let mut circuit = TickCircuit::new();
        circuit.tick().px(&[0]);
        circuit.tick().cx(&[(0, 1)]);
        append(&mut circuit, Gate::simple(kind, vec![QubitId(0)]));
        let program = compile(&circuit).unwrap();
        let mut seen = BTreeSet::new();
        for seed in 0..100 {
            let mut sim = PhasePoly::with_seed(2, seed);
            let shot = program
                .run_shot(&mut sim, &mut PecosRng::seed_from_u64(0))
                .unwrap();
            assert_eq!(shot.records, Vec::<bool>::new());
            let branch = sim.mz(&[QubitId(1)])[0].outcome;
            seen.insert(branch);
            let mut dense = vec![Complex64::new(0.0, 0.0); 4];
            dense[0] = Complex64::new(std::f64::consts::FRAC_1_SQRT_2, 0.0);
            dense[3] = dense[0];
            project(&mut dense, QubitId(0), false, branch);
            if branch {
                dense_gate(&mut dense, 0, &[QubitId(0)]);
            }
            if kind == GateType::PX {
                dense_gate(&mut dense, 12, &[QubitId(0)]);
            }
            assert_state(&sim, &dense);
        }
        assert_eq!(seen.len(), 2);
    }
}
