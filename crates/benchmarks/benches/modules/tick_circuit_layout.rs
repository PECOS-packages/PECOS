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

//! `TickCircuit` batched layout benchmarks.
//!
//! These benchmarks measure the current batched `TickCircuit` access patterns:
//! - direct `TickCircuit` traversal,
//! - explicit batched `TickCircuit` traversal, and
//! - direct vs `CircuitExecutor` simulator execution.

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos_core::gate_type::GateType;
use pecos_quantum::{Gate, QubitId, TickCircuit};
use pecos_simulators::{CircuitExecutor, CliffordGateable, SparseStab};
use std::hint::black_box;

const DISTANCES: &[usize] = &[3, 5, 7, 9, 11];
const AMORTIZED_SHOTS: usize = 64;

#[derive(Clone)]
struct LayoutSpec {
    label: String,
    num_qubits: usize,
    gate_count: usize,
    tick_circuit: TickCircuit,
}

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    let specs = DISTANCES
        .iter()
        .map(|&distance| {
            let rounds = distance;
            let tick_circuit = build_surface_like_tick_circuit(distance, rounds);
            let num_qubits = surface_like_num_qubits(distance);
            let gate_count = tick_circuit.gate_count();
            LayoutSpec {
                label: format!("d{distance}_r{rounds}"),
                num_qubits,
                gate_count,
                tick_circuit,
            }
        })
        .collect::<Vec<_>>();

    bench_traversal(c, &specs);
    bench_execution(c, &specs);
    bench_amortized_execution(c, &specs);
}

fn bench_traversal<M: Measurement>(c: &mut Criterion<M>, specs: &[LayoutSpec]) {
    let mut group = c.benchmark_group("tick_circuit_layout/traversal");

    for spec in specs {
        group.throughput(Throughput::Elements(spec.gate_count as u64));
        group.bench_with_input(
            BenchmarkId::new("tick_circuit_iter_gates", &spec.label),
            spec,
            |b, spec| {
                b.iter(|| black_box(traverse_tick_circuit(black_box(&spec.tick_circuit))));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("tick_circuit_gate_batches", &spec.label),
            spec,
            |b, spec| {
                b.iter(|| black_box(traverse_tick_circuit_batched(black_box(&spec.tick_circuit))));
            },
        );
    }

    group.finish();
}

fn bench_execution<M: Measurement>(c: &mut Criterion<M>, specs: &[LayoutSpec]) {
    let mut group = c.benchmark_group("tick_circuit_layout/execution_one_shot");

    for spec in specs {
        group.throughput(Throughput::Elements(spec.gate_count as u64));
        group.bench_with_input(
            BenchmarkId::new("tick_circuit_direct", &spec.label),
            spec,
            |b, spec| {
                b.iter(|| {
                    black_box(run_tick_circuit_direct(
                        black_box(&spec.tick_circuit),
                        spec.num_qubits,
                    ))
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("circuit_executor", &spec.label),
            spec,
            |b, spec| {
                b.iter(|| {
                    black_box(run_tick_circuit_executor(
                        black_box(&spec.tick_circuit),
                        spec.num_qubits,
                    ))
                });
            },
        );
    }

    group.finish();
}

fn bench_amortized_execution<M: Measurement>(c: &mut Criterion<M>, specs: &[LayoutSpec]) {
    let mut group = c.benchmark_group("tick_circuit_layout/execution_amortized_64_shots");

    for spec in specs {
        let throughput = spec.gate_count.saturating_mul(AMORTIZED_SHOTS);
        group.throughput(Throughput::Elements(throughput as u64));
        group.bench_with_input(
            BenchmarkId::new("tick_circuit_direct", &spec.label),
            spec,
            |b, spec| {
                b.iter(|| {
                    let mut total = 0usize;
                    for _ in 0..AMORTIZED_SHOTS {
                        total = total.wrapping_add(run_tick_circuit_direct(
                            black_box(&spec.tick_circuit),
                            spec.num_qubits,
                        ));
                    }
                    black_box(total)
                });
            },
        );
        group.bench_with_input(
            BenchmarkId::new("circuit_executor", &spec.label),
            spec,
            |b, spec| {
                b.iter(|| {
                    let mut total = 0usize;
                    for _ in 0..AMORTIZED_SHOTS {
                        total = total.wrapping_add(run_tick_circuit_executor(
                            black_box(&spec.tick_circuit),
                            spec.num_qubits,
                        ));
                    }
                    black_box(total)
                });
            },
        );
    }

    group.finish();
}

fn build_surface_like_tick_circuit(distance: usize, rounds: usize) -> TickCircuit {
    let num_data = distance * distance;
    let plaquettes = (distance - 1) * (distance - 1);
    let x_ancilla_start = num_data;
    let z_ancilla_start = x_ancilla_start + plaquettes;
    let total_qubits = surface_like_num_qubits(distance);

    let mut circuit = TickCircuit::new();
    let data_qubits = (0..num_data).collect::<Vec<_>>();
    let ancilla_qubits = (num_data..total_qubits).collect::<Vec<_>>();
    let x_ancillas = (x_ancilla_start..z_ancilla_start).collect::<Vec<_>>();
    let all_qubits = (0..total_qubits).collect::<Vec<_>>();

    circuit.tick().pz(&all_qubits);
    circuit.tick().h(&data_qubits);

    for _ in 0..rounds {
        circuit.tick().pz(&ancilla_qubits);
        circuit.tick().h(&x_ancillas);

        for neighbor in 0..4 {
            let pairs = surface_like_pairs_for_neighbor(distance, neighbor);
            add_disjoint_cx_layers(&mut circuit, total_qubits, pairs);
        }

        circuit.tick().h(&x_ancillas);
        circuit.tick().mz(&ancilla_qubits);
    }

    circuit.tick().mz(&data_qubits);
    circuit
}

fn surface_like_num_qubits(distance: usize) -> usize {
    let num_data = distance * distance;
    let plaquettes = (distance - 1) * (distance - 1);
    num_data + 2 * plaquettes
}

fn surface_like_pairs_for_neighbor(distance: usize, neighbor: usize) -> Vec<(usize, usize)> {
    let num_data = distance * distance;
    let plaquettes_per_type = (distance - 1) * (distance - 1);
    let x_ancilla_start = num_data;
    let z_ancilla_start = x_ancilla_start + plaquettes_per_type;
    let mut pairs = Vec::with_capacity(2 * plaquettes_per_type);

    for row in 0..(distance - 1) {
        for col in 0..(distance - 1) {
            let plaquette = row * (distance - 1) + col;
            let x_ancilla = x_ancilla_start + plaquette;
            let z_ancilla = z_ancilla_start + plaquette;
            let data = match neighbor {
                0 => row * distance + col,
                1 => (row + 1) * distance + col,
                2 => row * distance + col + 1,
                3 => (row + 1) * distance + col + 1,
                _ => unreachable!("neighbor index is in 0..4"),
            };

            pairs.push((x_ancilla, data));
            pairs.push((data, z_ancilla));
        }
    }

    pairs
}

fn add_disjoint_cx_layers(
    circuit: &mut TickCircuit,
    num_qubits: usize,
    mut remaining: Vec<(usize, usize)>,
) {
    while !remaining.is_empty() {
        let mut used = vec![false; num_qubits];
        let mut layer = Vec::new();
        let mut next = Vec::new();

        for (control, target) in remaining {
            if !used[control] && !used[target] {
                used[control] = true;
                used[target] = true;
                layer.push((control, target));
            } else {
                next.push((control, target));
            }
        }

        circuit.tick().cx(&layer);
        remaining = next;
    }
}

fn traverse_tick_circuit(circuit: &TickCircuit) -> usize {
    let mut total = 0usize;
    for (tick_idx, tick) in circuit.iter_ticks() {
        total = total.wrapping_add(tick_idx);
        for gate in tick.gate_batches() {
            total = total.wrapping_add(gate.num_gates());
            total = total.wrapping_add(gate.qubits.len());
        }
    }
    total
}

fn traverse_tick_circuit_batched(circuit: &TickCircuit) -> usize {
    let mut total = 0usize;
    for (tick_idx, batch) in circuit.iter_gate_batches_with_tick() {
        total = total.wrapping_add(tick_idx);
        total = total.wrapping_add(batch.num_gates());
        total = total.wrapping_add(batch.qubits.len());
    }
    total
}

fn run_tick_circuit_direct(circuit: &TickCircuit, num_qubits: usize) -> usize {
    let mut sim = SparseStab::new(num_qubits);
    let mut measurement_count = 0usize;

    for (_tick_idx, tick) in circuit.iter_ticks() {
        for gate in tick.gate_batches() {
            measurement_count += execute_gate_direct(&mut sim, gate);
        }
    }

    measurement_count
}

fn execute_gate_direct<S: CliffordGateable>(sim: &mut S, gate: &Gate) -> usize {
    match gate.gate_type {
        GateType::PZ | GateType::QAlloc => {
            sim.pz(&gate.qubits);
            0
        }
        GateType::H => {
            sim.h(&gate.qubits);
            0
        }
        GateType::CX => {
            let pairs = gate
                .qubits
                .chunks_exact(2)
                .map(|pair| (pair[0], pair[1]))
                .collect::<Vec<(QubitId, QubitId)>>();
            sim.cx(&pairs);
            0
        }
        GateType::MZ | GateType::MeasureFree => sim.mz(&gate.qubits).len(),
        other => panic!("unsupported benchmark gate type: {other:?}"),
    }
}

fn run_tick_circuit_executor(circuit: &TickCircuit, num_qubits: usize) -> usize {
    let mut sim = SparseStab::new(num_qubits);
    CircuitExecutor::new(circuit).run(&mut sim).len()
}
