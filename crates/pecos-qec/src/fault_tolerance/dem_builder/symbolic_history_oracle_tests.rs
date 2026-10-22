// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0

//! Exact joint distributions from dense amplitudes and explicit measurement branches.

use super::DemBuilder;
use crate::fault_tolerance::fault_sampler::symbolic_measurement_history;
use crate::fault_tolerance::influence_builder::InfluenceBuilder;
use pecos_core::gate_type::GateType;
use pecos_core::{Angle64, BitSet, Gate, QubitId, Unitary, UnitaryRep};
use pecos_quantum::{TickCircuit, unitary_matrix::to_matrix_with_size};
use pecos_random::PecosRng;
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, DenseStateVec, SymbolicSparseStab,
    SymbolicSparseStabVecSet,
};
use rand::RngExt;

fn unitary(sim: &mut DenseStateVec, gate: &Gate, n: usize) {
    let qs = &gate.qubits;
    let a = &gate.angles;
    match gate.gate_type {
        GateType::RX => {
            sim.rx(a[0], qs);
        }
        GateType::RY => {
            sim.ry(a[0], qs);
        }
        GateType::RZ => {
            sim.rz(a[0], qs);
        }
        GateType::RXX => {
            sim.rxx(a[0], &[(qs[0], qs[1])]);
        }
        GateType::RYY => {
            sim.ryy(a[0], &[(qs[0], qs[1])]);
        }
        GateType::RZZ => {
            sim.rzz(a[0], &[(qs[0], qs[1])]);
        }
        GateType::RXY1Q => {
            sim.rxy1q(a[0], a[1], qs);
        }
        GateType::RXYXY2Q => {
            sim.rxyxy2q(a[0], a[1], &[(qs[0], qs[1])]);
        }
        GateType::U => {
            sim.u(a[0], a[1], a[2], qs);
        }
        GateType::QFree
        | GateType::Idle
        | GateType::MeasCrosstalkGlobalPayload
        | GateType::MeasCrosstalkLocalPayload
        | GateType::TrackedPauliMeta => {}
        _ => {
            for operands in qs.chunks(gate.gate_type.quantum_arity()) {
                let rep = UnitaryRep::Gate(
                    Unitary::named(gate.gate_type),
                    operands.iter().map(QubitId::index).collect(),
                );
                let matrix = to_matrix_with_size(&rep, n);
                let amplitudes = sim.state();
                for row in 0..1 << n {
                    sim.set_amplitude(
                        row,
                        (0..1 << n)
                            .map(|col| matrix[(row, col)] * amplitudes[col])
                            .sum(),
                    );
                }
            }
        }
    }
}

fn distribution(gates: &[Gate], n: usize) -> Vec<f64> {
    let mut branches = vec![(DenseStateVec::new(n), 0usize, 1.0)];
    let mut records = 0;
    for gate in gates {
        let measured = matches!(
            gate.gate_type,
            GateType::MX
                | GateType::MZ
                | GateType::MPZ
                | GateType::MeasureFree
                | GateType::MeasureLeaked
        );
        let prepared = matches!(
            gate.gate_type,
            GateType::PZ | GateType::PX | GateType::QAlloc
        );
        if !measured && !prepared {
            for (sim, _, _) in &mut branches {
                unitary(sim, gate, n);
            }
            continue;
        }
        for &q in &gate.qubits {
            let mut next = Vec::new();
            for (mut sim, record, weight) in branches {
                if gate.gate_type == GateType::MX {
                    sim.h(&[q]);
                }
                let amplitudes = sim.state();
                for outcome in 0..2 {
                    let probability: f64 = amplitudes
                        .iter()
                        .enumerate()
                        .filter(|(basis, _)| (basis >> q.index()) & 1 == outcome)
                        .map(|(_, amplitude)| amplitude.norm_sqr())
                        .sum();
                    if probability < 1e-14 {
                        continue;
                    }
                    let mut projected = sim.clone();
                    for (basis, &amplitude) in amplitudes.iter().enumerate() {
                        projected.set_amplitude(
                            basis,
                            if (basis >> q.index()) & 1 == outcome {
                                amplitude / probability.sqrt()
                            } else {
                                0.0.into()
                            },
                        );
                    }
                    if gate.gate_type == GateType::MX {
                        projected.h(&[q]);
                    }
                    if (prepared || gate.gate_type == GateType::MPZ) && outcome == 1 {
                        projected.x(&[q]);
                    }
                    if gate.gate_type == GateType::PX {
                        projected.h(&[q]);
                    }
                    next.push((
                        projected,
                        record
                            | if measured && gate.gate_type != GateType::MeasureLeaked {
                                outcome << records
                            } else {
                                0
                            },
                        weight * probability,
                    ));
                }
            }
            branches = next;
            records += usize::from(measured && gate.gate_type != GateType::MeasureLeaked);
        }
    }
    let mut probabilities = vec![0.0; 1 << records];
    for (_, record, weight) in branches {
        probabilities[record] += weight;
    }
    assert!((probabilities.iter().sum::<f64>() - 1.0).abs() < 1e-10);
    probabilities
}

fn apply_vecset_pairs(
    sim: &mut SymbolicSparseStabVecSet,
    gate_type: GateType,
    pairs: &[(usize, usize)],
) {
    for &(a, b) in pairs {
        let steps = match gate_type {
            GateType::CY => vec![Gate::szdg(&[b]), Gate::cx(&[(a, b)]), Gate::sz(&[b])],
            GateType::CZ => vec![Gate::h(&[b]), Gate::cx(&[(a, b)]), Gate::h(&[b])],
            GateType::SZZ => vec![Gate::cx(&[(a, b)]), Gate::sz(&[b]), Gate::cx(&[(a, b)])],
            GateType::SZZdg => vec![Gate::cx(&[(a, b)]), Gate::szdg(&[b]), Gate::cx(&[(a, b)])],
            GateType::SWAP => vec![
                Gate::cx(&[(a, b)]),
                Gate::cx(&[(b, a)]),
                Gate::cx(&[(a, b)]),
            ],
            GateType::SXX | GateType::SXXdg => vec![
                Gate::h(&[a, b]),
                Gate::simple(
                    if gate_type == GateType::SXX {
                        GateType::SZZ
                    } else {
                        GateType::SZZdg
                    },
                    vec![QubitId(a), QubitId(b)],
                ),
                Gate::h(&[a, b]),
            ],
            GateType::SYY | GateType::SYYdg => vec![
                Gate::szdg(&[a, b]),
                Gate::simple(
                    if gate_type == GateType::SYY {
                        GateType::SXX
                    } else {
                        GateType::SXXdg
                    },
                    vec![QubitId(a), QubitId(b)],
                ),
                Gate::sz(&[a, b]),
            ],
            _ => unreachable!(),
        };
        for step in steps {
            apply_vecset(sim, &step);
        }
    }
}

fn apply_vecset(sim: &mut SymbolicSparseStabVecSet, gate: &Gate) {
    if let Some(lowering) = pecos_core::try_lower_rotation_to_clifford(gate) {
        match lowering {
            pecos_core::CliffordLowering::Named(named) => {
                apply_vecset(sim, &Gate::simple(named, gate.qubits.clone()));
            }
            pecos_core::CliffordLowering::PerQubit(pauli) => {
                for &q in &gate.qubits {
                    apply_vecset(sim, &Gate::simple(pauli, vec![q]));
                }
            }
        }
        return;
    }
    let qs: Vec<_> = gate.qubits.iter().map(QubitId::index).collect();
    let pairs: Vec<_> = qs.as_chunks::<2>().0.iter().map(|p| (p[0], p[1])).collect();
    // VecSet exposes H, SZ, CX and Paulis. Use their Clifford decompositions
    // for the other named unitaries; the dense oracle still uses matrices.
    let sequence = match gate.gate_type {
        GateType::SX => Some(&[GateType::H, GateType::SZ, GateType::H][..]),
        GateType::SXdg => Some(&[GateType::H, GateType::SZdg, GateType::H][..]),
        GateType::SY => Some(&[GateType::Z, GateType::H][..]),
        GateType::SYdg => Some(&[GateType::H, GateType::Z][..]),
        GateType::SZdg => Some(&[GateType::Z, GateType::SZ][..]),
        GateType::F => Some(&[GateType::SX, GateType::SZ][..]),
        GateType::Fdg => Some(&[GateType::SZdg, GateType::SXdg][..]),
        _ => None,
    };
    if let Some(sequence) = sequence {
        for &gt in sequence {
            apply_vecset(sim, &Gate::simple(gt, gate.qubits.clone()));
        }
        return;
    }
    match gate.gate_type {
        GateType::H => {
            sim.h(&qs);
        }
        GateType::X => {
            sim.x(&qs);
        }
        GateType::Y => {
            sim.y(&qs);
        }
        GateType::Z => {
            sim.z(&qs);
        }
        GateType::SZ => {
            sim.sz(&qs);
        }
        GateType::CX => {
            sim.cx(&pairs);
        }
        GateType::CY
        | GateType::CZ
        | GateType::SZZ
        | GateType::SZZdg
        | GateType::SWAP
        | GateType::SXX
        | GateType::SXXdg
        | GateType::SYY
        | GateType::SYYdg => {
            apply_vecset_pairs(sim, gate.gate_type, &pairs);
        }
        GateType::PX | GateType::PZ | GateType::QAlloc => {
            for &q in &qs {
                sim.pz(q);
            }
            if gate.gate_type == GateType::PX {
                sim.h(&qs);
            }
        }
        GateType::MX => {
            sim.h(&qs);
            sim.mz(&qs);
            sim.h(&qs);
        }
        GateType::MZ | GateType::MeasureFree => {
            sim.mz(&qs);
        }
        GateType::MPZ => {
            sim.mz(&qs);
            for &q in &qs {
                sim.pz(q);
            }
        }
        GateType::MeasureLeaked => {
            sim.mz_unrecorded(&qs);
        }
        GateType::I
        | GateType::Idle
        | GateType::QFree
        | GateType::MeasCrosstalkGlobalPayload
        | GateType::MeasCrosstalkLocalPayload
        | GateType::TrackedPauliMeta => {}
        _ => panic!("unsupported oracle gate: {gate:?}"),
    }
}

fn check_record_flags(
    gates: &[Gate],
    probabilities: &[f64],
    histories: &[&pecos_simulators::MeasurementHistory],
) {
    for column in 0..histories[0].len() {
        let mut possible = vec![[false; 2]; 1 << column];
        for (record, &probability) in probabilities.iter().enumerate() {
            if probability > 1e-12 {
                possible[record & ((1 << column) - 1)][(record >> column) & 1] = true;
            }
        }
        let determined_by_prior_records = !possible.contains(&[true, true]);
        for history in histories {
            let result = &history[column];
            assert_eq!(
                result.is_deterministic, determined_by_prior_records,
                "record {column} of {gates:?}"
            );
            if result.is_deterministic {
                assert!(result.outcome.iter().all(|dep| dep < column));
            } else {
                assert_eq!(result.outcome.iter().collect::<Vec<_>>(), vec![column]);
                assert!(!result.flip);
            }
        }
    }
}

fn check(gates: &[Gate], n: usize) {
    let probabilities = distribution(gates, n);
    let mut circuit = TickCircuit::new();
    for gate in gates {
        circuit.tick().try_add_gate(gate.clone()).unwrap();
    }
    let tick_history = symbolic_measurement_history(&circuit).unwrap();
    let dag = pecos_quantum::DagCircuit::try_from(&circuit).unwrap();
    let dag_info = InfluenceBuilder::new(&dag)
        .unwrap()
        .run_symbolic_simulation()
        .unwrap();
    let mut crosstalk = SymbolicSparseStab::new(n);
    let mut vecset = SymbolicSparseStabVecSet::new(n);
    for (node, gate) in gates.iter().enumerate() {
        DemBuilder::apply_symbolic_gate_for_crosstalk_hidden_mz(&mut crosstalk, node, gate)
            .unwrap();
        apply_vecset(&mut vecset, gate);
    }
    let histories = [
        &tick_history,
        vecset.measurement_history(),
        &dag_info.history,
        crosstalk.measurement_history(),
    ];
    for history in histories {
        assert_eq!(probabilities.len(), 1 << history.len());
        for (result, expected) in history.iter().zip(tick_history.iter()) {
            assert_eq!(
                result.outcome.iter().collect::<Vec<_>>(),
                expected.outcome.iter().collect::<Vec<_>>(),
                "{gates:?}"
            );
            assert_eq!(
                (result.flip, result.is_deterministic, result.index),
                (expected.flip, expected.is_deterministic, expected.index),
                "{gates:?}"
            );
        }
    }
    check_record_flags(gates, &probabilities, &histories);
    for subset in 0usize..1 << tick_history.len() {
        let mut values = [0.0; 2];
        for (record, &probability) in probabilities.iter().enumerate() {
            values[((record & subset).count_ones() & 1) as usize] += probability;
        }
        let exact = if values[1] < 1e-12 {
            Some(false)
        } else if values[0] < 1e-12 {
            Some(true)
        } else {
            assert!((values[0] - 0.5).abs() < 1e-10);
            None
        };
        for history in histories {
            let mut dependencies = BitSet::new();
            let mut flip = false;
            for (column, result) in history.iter().enumerate() {
                if subset & (1 << column) != 0 {
                    dependencies ^= &result.outcome;
                    flip ^= result.flip;
                }
            }
            let symbolic = dependencies.is_empty().then_some(flip);
            assert_eq!(symbolic, exact, "parity {subset:b} of {gates:?}: {history}");
        }
    }
}

#[test]
fn exact_oracle_entangled_reset() {
    check(
        &[
            Gate::pz(&[0, 1]),
            Gate::h(&[0]),
            Gate::cx(&[(0, 1)]),
            Gate::pz(&[0]),
            Gate::mz(&[1]),
        ],
        2,
    );
}

#[test]
fn exact_oracle_mx_post_state() {
    check(&[Gate::pz(&[0]), Gate::mx(&[0]), Gate::mz(&[0])], 1);
    check(
        &[
            Gate::mx(&[0]),
            Gate::sz(&[0]),
            Gate::h(&[0]),
            Gate::mx(&[0]),
            Gate::mz(&[0]),
        ],
        1,
    );
}

#[test]
fn exact_oracle_hidden_cancellation() {
    check(
        &[
            Gate::h(&[0]),
            Gate::cx(&[(0, 1)]),
            Gate::measure_leaked(&[0]),
            Gate::mz(&[0, 1]),
            Gate::mz(&[2]),
        ],
        3,
    );
    check(
        &[
            Gate::h(&[0]),
            Gate::cx(&[(0, 1)]),
            Gate::measure_leaked(&[0]),
            Gate::mz(&[0]),
            Gate::x(&[1]),
            Gate::mz(&[1]),
            Gate::h(&[2]),
            Gate::cx(&[(2, 3)]),
            Gate::pz(&[2]),
            Gate::mz(&[3]),
        ],
        4,
    );
    check(
        &[
            Gate::h(&[0]),
            Gate::measure_leaked(&[0]),
            Gate::h(&[1]),
            Gate::mz(&[1]),
            Gate::cx(&[(0, 1)]),
            Gate::mz(&[1, 0]),
        ],
        2,
    );
}

#[test]
fn exact_oracle_hidden_substitution_chain() {
    check(
        &[
            Gate::h(&[0, 1, 2]),
            Gate::measure_leaked(&[0, 1, 2]),
            Gate::cx(&[(0, 3)]),
            Gate::cx(&[(1, 3)]),
            Gate::x(&[3]),
            Gate::mz(&[3]),
            Gate::cx(&[(1, 3)]),
            Gate::cx(&[(2, 3)]),
            Gate::mz(&[3]),
            Gate::mz(&[0, 1, 2]),
        ],
        4,
    );
}

#[test]
fn exact_oracle_bell_repeated_mpz() {
    check(
        &[
            Gate::h(&[0]),
            Gate::cx(&[(0, 1)]),
            Gate::mz(&[0, 1]),
            Gate::mz(&[0]),
            Gate::mpz(&[1]),
            Gate::mz(&[0, 1]),
        ],
        2,
    );
}

fn gate_cases() -> Vec<Gate> {
    use GateType::{
        CX, CY, CZ, F, Fdg, H, I, Idle, MPZ, MX, MZ, MeasureFree, MeasureLeaked, PX, PZ, QAlloc,
        QFree, SWAP, SX, SXX, SXXdg, SXdg, SY, SYY, SYYdg, SYdg, SZ, SZZ, SZZdg, SZdg, X, Y, Z,
    };
    let mut gates: Vec<_> = [
        H,
        F,
        Fdg,
        SX,
        SXdg,
        SY,
        SYdg,
        SZ,
        SZdg,
        X,
        Y,
        Z,
        CX,
        CY,
        CZ,
        SXX,
        SXXdg,
        SYY,
        SYYdg,
        SZZ,
        SZZdg,
        SWAP,
        PX,
        PZ,
        QAlloc,
        MX,
        MZ,
        MPZ,
        MeasureFree,
        MeasureLeaked,
        I,
        Idle,
        QFree,
        GateType::MeasCrosstalkGlobalPayload,
        GateType::MeasCrosstalkLocalPayload,
        GateType::TrackedPauliMeta,
    ]
    .into_iter()
    .map(|g| {
        Gate::new(
            g,
            vec![],
            vec![0.0; g.classical_arity()],
            (0..g.quantum_arity()).map(QubitId).collect::<Vec<_>>(),
        )
    })
    .collect();
    for angle in [
        Angle64::ZERO,
        Angle64::QUARTER_TURN,
        Angle64::HALF_TURN,
        Angle64::THREE_QUARTERS_TURN,
    ] {
        for g in [
            GateType::RX,
            GateType::RY,
            GateType::RZ,
            GateType::RXX,
            GateType::RYY,
            GateType::RZZ,
        ] {
            gates.push(Gate::with_angles(
                g,
                vec![angle],
                (0..g.quantum_arity()).map(QubitId).collect::<Vec<_>>(),
            ));
        }
        for axis in [
            Angle64::ZERO,
            Angle64::QUARTER_TURN,
            Angle64::HALF_TURN,
            Angle64::THREE_QUARTERS_TURN,
        ] {
            gates.push(Gate::rxy1q(angle, axis, &[0]));
            gates.push(Gate::rxyxy2q(angle, axis, &[(0, 1)]));
        }
        gates.push(Gate::u(Angle64::ZERO, Angle64::ZERO, angle, &[0]));
    }
    gates
}

#[test]
fn exact_oracle_gate_catalog() {
    for gate in gate_cases() {
        check(
            &[
                Gate::h(&[0, 1]),
                Gate::sz(&[0]),
                Gate::cx(&[(0, 1)]),
                gate,
                Gate::h(&[0, 1]),
                Gate::mz(&[0, 1]),
            ],
            2,
        );
    }
}

#[test]
fn exact_oracle_seeded_clifford_circuits() {
    let cases = gate_cases();
    let mut rng = PecosRng::seed_from_u64(0x5045_434f_535f_5030);
    for trial in 0..6000 {
        let n = rng.random_range(2..=4);
        let mut gates = Vec::new();
        for step in 0..8 {
            let mut gate = if step % 3 == 0 {
                [
                    Gate::h(&[0]),
                    Gate::mx(&[0]),
                    Gate::mz(&[0]),
                    Gate::pz(&[0]),
                    Gate::measure_leaked(&[0]),
                ][rng.random_range(0..5)]
                .clone()
            } else {
                cases[(trial + rng.random_range(0..cases.len())) % cases.len()].clone()
            };
            let offset = rng.random_range(0..n);
            for q in &mut gate.qubits {
                *q = QubitId((q.index() + offset) % n);
            }
            gates.push(gate);
        }
        gates.push(Gate::mz(&(0..n).collect::<Vec<_>>()));
        check(&gates, n);
    }
}

/// The highest-pivot elimination passes every other fixture but mislabels
/// record 5 here: a substitution naming a lower source revisits it.
#[test]
fn exact_oracle_pivot_ordering() {
    check(
        &[
            Gate::x(&[1]),
            Gate::cx(&[(3, 2)]),
            Gate::h(&[2]),
            Gate::h(&[3]),
            Gate::cx(&[(3, 1)]),
            Gate::x(&[2]),
            Gate::pz(&[1]),
            Gate::cx(&[(2, 3)]),
            Gate::pz(&[1]),
            Gate::x(&[1]),
            Gate::h(&[1]),
            Gate::pz(&[2]),
            Gate::mz(&[3]),
            Gate::mx(&[0]),
            Gate::mz(&[0, 1, 2, 3]),
        ],
        4,
    );
}

#[test]
fn exact_oracle_seeded_reset_and_leak_heavy_circuits() {
    let mut rng = PecosRng::seed_from_u64(0xdead_beef);
    for _ in 0..3000 {
        let n = rng.random_range(3..=5);
        let mut gates = Vec::new();
        for _ in 0..14 {
            let a = rng.random_range(0..n);
            let mut b = rng.random_range(0..n);
            if b == a {
                b = (a + 1) % n;
            }
            gates.push(match rng.random_range(0..9) {
                0 | 1 => Gate::h(&[a]),
                2 | 3 => Gate::cx(&[(a, b)]),
                4 => Gate::measure_leaked(&[a]),
                5 => Gate::pz(&[a]),
                6 => Gate::mz(&[a]),
                7 => Gate::x(&[a]),
                _ => Gate::mx(&[a]),
            });
        }
        gates.push(Gate::mz(&(0..n).collect::<Vec<_>>()));
        check(&gates, n);
    }
}
