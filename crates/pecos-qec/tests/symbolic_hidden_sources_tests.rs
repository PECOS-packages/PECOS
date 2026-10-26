// Copyright 2026 The PECOS Developers
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy at https://www.apache.org/licenses/LICENSE-2.0

use pecos_core::gate_type::GateType;
use pecos_core::{BitSet, Gate};
use pecos_num::z2_linalg::z2_rank;
use pecos_qec::fault_tolerance::fault_sampler::symbolic_measurement_history;
use pecos_quantum::TickCircuit;
use pecos_random::{PecosRng, RngExt};
use pecos_simulators::{CircuitExecutor, SparseStab};

#[test]
fn reset_entangled_partner_history_is_random() {
    let mut circuit = TickCircuit::new();
    circuit.tick().h(&[1]);
    circuit.tick().cx(&[(1, 0)]);
    circuit.tick().pz(&[1]);
    circuit.tick().mz(&[0]);
    let history = symbolic_measurement_history(&circuit).unwrap();
    assert_eq!(history.len(), 1);
    assert!(!history.get(0).unwrap().is_deterministic);
    assert_eq!(history.get(0).unwrap().outcome, BitSet::single(0));
}

#[test]
fn leaked_bell_measurement_keeps_two_correlated_records() {
    let mut circuit = TickCircuit::new();
    circuit.tick().pz(&[0, 1]);
    circuit.tick().h(&[0]);
    circuit.tick().cx(&[(0, 1)]);
    circuit
        .tick()
        .try_add_gate(Gate::measure_leaked(&[0]))
        .unwrap();
    circuit.tick().mz(&[1]);
    circuit.tick().mz(&[0]);
    let history = symbolic_measurement_history(&circuit).unwrap();
    assert_eq!(history.len(), 2);
    let first = history.get(0).unwrap();
    assert_eq!(first.index, 0);
    assert!(!first.is_deterministic);
    assert_eq!(first.outcome, BitSet::single(0));
    assert!(!first.flip);
    let second = history.get(1).unwrap();
    assert_eq!(second.index, 1);
    assert!(second.is_deterministic);
    assert_eq!(second.outcome, BitSet::single(0));
    assert!(!second.flip);
}

fn random_circuit(seed: u64) -> (TickCircuit, usize) {
    let mut rng = PecosRng::seed_from_u64(seed);
    let num_qubits = rng.random_range(2..=6);
    let num_ticks = rng.random_range(10..=40);
    let mut circuit = TickCircuit::new();
    while circuit.num_ticks() < num_ticks {
        let q = rng.random_range(0..num_qubits);
        let r = (q + rng.random_range(1..num_qubits)) % num_qubits;
        // Frequently prepare an entangled pair immediately before a reset.
        if num_ticks - circuit.num_ticks() >= 5 && rng.random_bool(0.3) {
            for gate in [
                Gate::pz(&[q, r]),
                Gate::h(&[q]),
                Gate::cx(&[(q, r)]),
                if rng.random_bool(0.5) {
                    Gate::pz(&[q])
                } else {
                    Gate::px(&[q])
                },
                Gate::mz(&[r]),
            ] {
                circuit.tick().try_add_gate(gate).unwrap();
            }
            continue;
        }
        let gate = match rng.random_range(0..14) {
            0 => Gate::h(&[q]),
            1 => Gate::sz(&[q]),
            2 | 3 => Gate::cx(&[(q, r)]),
            4 => Gate::cz(&[(q, r)]),
            5 => Gate::x(&[q]),
            6 | 7 => Gate::pz(&[q]),
            8 => Gate::px(&[q]),
            9 | 10 => Gate::mz(&[q]),
            11 => Gate::mx(&[q]),
            12 => Gate::mpz(&[q]),
            _ => Gate::measure_leaked(&[q]),
        };
        circuit.tick().try_add_gate(gate).unwrap();
    }
    (circuit, num_qubits)
}

fn sampled_twin(circuit: &TickCircuit) -> (TickCircuit, Vec<usize>) {
    let mut twin = TickCircuit::new();
    let mut record_columns = Vec::new();
    let mut column = 0;
    for (_, tick) in circuit.iter_ticks() {
        let mut twin_tick = twin.tick();
        for batch in tick.iter_gate_batches() {
            let mut gate = batch.as_gate().clone();
            if gate.gate_type.consumes_measurement_record() {
                record_columns.extend(column..column + gate.qubits.len());
            } else if gate.gate_type == GateType::MeasureLeaked {
                gate = Gate::mz(&gate.qubits);
            }
            if gate.gate_type.consumes_measurement_record() {
                column += gate.qubits.len();
            }
            twin_tick.try_add_gate(gate).unwrap();
        }
    }
    (twin, record_columns)
}

fn vecset_history(
    circuit: &TickCircuit,
    num_qubits: usize,
) -> pecos_simulators::MeasurementHistory {
    let mut sim = pecos_simulators::SymbolicSparseStabVecSet::new(num_qubits);
    for gate in circuit.iter_gate_batches() {
        let qubits: Vec<usize> = gate.qubits.iter().map(pecos_core::QubitId::index).collect();
        match gate.gate_type {
            GateType::H => {
                sim.h(&qubits);
            }
            GateType::SZ => {
                sim.sz(&qubits);
            }
            GateType::X => {
                sim.x(&qubits);
            }
            GateType::CX => {
                sim.cx(&[(qubits[0], qubits[1])]);
            }
            GateType::CZ => {
                sim.h(&[qubits[1]])
                    .cx(&[(qubits[0], qubits[1])])
                    .h(&[qubits[1]]);
            }
            GateType::PZ | GateType::PX => {
                for &q in &qubits {
                    sim.pz(q);
                }
                if gate.gate_type == GateType::PX {
                    sim.h(&qubits);
                }
            }
            GateType::MZ | GateType::MX | GateType::MPZ => {
                if gate.gate_type == GateType::MX {
                    sim.h(&qubits);
                }
                sim.mz(&qubits);
                if gate.gate_type == GateType::MX {
                    sim.h(&qubits);
                }
                if gate.gate_type == GateType::MPZ {
                    for &q in &qubits {
                        sim.pz(q);
                    }
                }
            }
            GateType::MeasureLeaked => sim.mz_unrecorded(&qubits),
            other => panic!("unexpected generated gate {other:?}"),
        }
    }
    sim.measurement_history().clone()
}

#[test]
fn seeded_histories_match_sparse_stab_soundness_and_completeness() {
    for seed in 0..300 {
        let (circuit, num_qubits) = random_circuit(seed);
        let history = symbolic_measurement_history(&circuit)
            .unwrap_or_else(|err| panic!("seed {seed}, circuit {circuit:?}: {err}"));
        assert_eq!(
            history.as_slice(),
            vecset_history(&circuit, num_qubits).as_slice(),
            "simulator agreement: seed {seed}, circuit {circuit:?}"
        );
        let (twin, record_columns) = sampled_twin(&circuit);
        assert_eq!(
            history.len(),
            record_columns.len(),
            "seed {seed}, circuit {circuit:?}"
        );
        let mut sim = SparseStab::with_seed(num_qubits, seed ^ 0x5eed);
        let executor = CircuitExecutor::new(&twin);
        let shots = 256.max(2 * history.len() + 64);
        let mut rows = Vec::with_capacity(shots);
        for shot in 0..shots {
            sim.reset();
            let outcomes = executor
                .run(&mut sim)
                .unwrap_or_else(|err| panic!("seed {seed}, circuit {circuit:?}: {err}"));
            let records: Vec<bool> = record_columns
                .iter()
                .map(|&i| outcomes[i].outcome)
                .collect();
            for result in history.iter().filter(|result| result.is_deterministic) {
                let expected = result
                    .outcome
                    .iter()
                    .fold(result.flip, |value, i| value ^ records[i]);
                assert_eq!(
                    records[result.index], expected,
                    "soundness: seed {seed}, shot {shot}, record {}, circuit {circuit:?}",
                    result.index
                );
            }
            let mut row: Vec<usize> = records
                .iter()
                .enumerate()
                .filter_map(|(i, &value)| value.then_some(i))
                .collect();
            row.push(history.len()); // Constant column for affine relations.
            rows.push(row);
        }
        let nullity = history.len() + 1 - z2_rank(&rows);
        assert_eq!(
            nullity,
            history.deterministic().len(),
            "completeness: seed {seed}, circuit {circuit:?}"
        );
    }
}
