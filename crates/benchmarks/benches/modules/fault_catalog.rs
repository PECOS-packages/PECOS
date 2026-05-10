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

//! Parameterized fault-catalog benchmarks.
//!
//! These benchmarks cover the Rust-side sweep path before doing packed
//! performance work:
//! - structural catalog construction from a `TickCircuit`,
//! - applying a concrete noise point with `with_noise`,
//! - projecting the richer catalog into raw-measurement mechanisms, and
//! - amortized noise sweeps compared with direct concrete catalog builds.

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos_qec::SurfaceCode;
use pecos_qec::fault_tolerance::fault_sampler::{
    FaultCatalog, StochasticNoiseParams, build_fault_catalog, build_fault_table,
};
use pecos_quantum::{Attribute, TickCircuit, TickMeasRef};
use std::hint::black_box;

const DISTANCES: &[usize] = &[3, 5, 7, 9, 11];
const SWEEP_NOISES: &[StochasticNoiseParams] = &[
    StochasticNoiseParams {
        p1: 0.00005,
        p2: 0.0005,
        p_meas: 0.0005,
        p_prep: 0.0005,
    },
    StochasticNoiseParams {
        p1: 0.0001,
        p2: 0.001,
        p_meas: 0.001,
        p_prep: 0.001,
    },
    StochasticNoiseParams {
        p1: 0.0002,
        p2: 0.002,
        p_meas: 0.002,
        p_prep: 0.002,
    },
    StochasticNoiseParams {
        p1: 0.0005,
        p2: 0.005,
        p_meas: 0.005,
        p_prep: 0.005,
    },
];

#[derive(Debug)]
struct MemoryCircuit {
    circuit: TickCircuit,
    distance: usize,
    rounds: usize,
    num_measurements: usize,
    num_detectors: usize,
    num_observables: usize,
}

#[derive(Debug)]
struct CatalogShape {
    locations: usize,
    alternatives: usize,
}

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_from_circuit(c);
    bench_with_noise(c);
    bench_to_mechanisms(c);
    bench_noise_sweeps(c);
}

fn bench_from_circuit<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("fault_catalog/from_circuit");

    for memory in surface_memory_circuits() {
        let shape = catalog_shape(&memory.circuit);
        group.throughput(Throughput::Elements(as_u64(shape.locations)));
        group.bench_with_input(bench_id(&memory, &shape), &memory, |b, memory| {
            b.iter(|| {
                black_box(FaultCatalog::from_circuit(black_box(&memory.circuit)))
                    .expect("surface memory circuit should be supported")
            });
        });
    }

    group.finish();
}

fn bench_with_noise<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("fault_catalog/with_noise");
    let noise = representative_noise();

    for memory in surface_memory_circuits() {
        let shape = catalog_shape(&memory.circuit);
        group.throughput(Throughput::Elements(as_u64(shape.locations)));
        group.bench_with_input(bench_id(&memory, &shape), &memory, |b, memory| {
            b.iter_batched(
                || {
                    FaultCatalog::from_circuit(&memory.circuit)
                        .expect("surface memory circuit should be supported")
                },
                |mut catalog| {
                    catalog.with_noise(black_box(&noise));
                    black_box(catalog)
                },
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_to_mechanisms<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("fault_catalog/to_mechanisms");
    let noise = representative_noise();

    for memory in surface_memory_circuits() {
        let shape = catalog_shape(&memory.circuit);
        group.throughput(Throughput::Elements(as_u64(shape.alternatives)));
        group.bench_with_input(bench_id(&memory, &shape), &memory, |b, memory| {
            b.iter_batched(
                || {
                    let mut catalog = FaultCatalog::from_circuit(&memory.circuit)
                        .expect("surface memory circuit should be supported");
                    catalog.with_noise(&noise);
                    catalog
                },
                |catalog| black_box(catalog.to_mechanisms()),
                BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

fn bench_noise_sweeps<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("fault_catalog/noise_sweep");

    for memory in surface_memory_circuits() {
        let shape = catalog_shape(&memory.circuit);
        let id = bench_label(&memory, &shape);
        group.throughput(Throughput::Elements(as_u64(
            shape.locations * SWEEP_NOISES.len(),
        )));

        group.bench_with_input(
            BenchmarkId::new("direct_catalog_sweep", id.clone()),
            &memory,
            |b, memory| {
                b.iter(|| {
                    let mut total_locations = 0usize;
                    let mut total_alternatives = 0usize;
                    for noise in SWEEP_NOISES {
                        let catalog = build_fault_catalog(black_box(&memory.circuit), noise)
                            .expect("surface memory circuit should be supported");
                        total_locations += catalog.locations.len();
                        total_alternatives += count_alternatives(&catalog);
                    }
                    black_box((total_locations, total_alternatives))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parameterized_catalog_sweep", id.clone()),
            &memory,
            |b, memory| {
                b.iter(|| {
                    let structural = FaultCatalog::from_circuit(black_box(&memory.circuit))
                        .expect("surface memory circuit should be supported");
                    let mut total_locations = 0usize;
                    let mut total_alternatives = 0usize;
                    for noise in SWEEP_NOISES {
                        let catalog = structural.parameterized(noise);
                        total_locations += catalog.locations.len();
                        total_alternatives += count_alternatives(&catalog);
                    }
                    black_box((total_locations, total_alternatives))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mutable_catalog_sweep", id.clone()),
            &memory,
            |b, memory| {
                b.iter(|| {
                    let mut catalog = FaultCatalog::from_circuit(black_box(&memory.circuit))
                        .expect("surface memory circuit should be supported");
                    let mut total_locations = 0usize;
                    let mut total_alternatives = 0usize;
                    for noise in SWEEP_NOISES {
                        catalog.with_noise(black_box(noise));
                        total_locations += catalog.locations.len();
                        total_alternatives += count_alternatives(&catalog);
                    }
                    black_box((total_locations, total_alternatives))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("direct_raw_mechanism_sweep", id.clone()),
            &memory,
            |b, memory| {
                b.iter(|| {
                    let mut total_mechanisms = 0usize;
                    let mut total_alternatives = 0usize;
                    for noise in SWEEP_NOISES {
                        let mechanisms = build_fault_table(black_box(&memory.circuit), noise)
                            .expect("surface memory circuit should be supported");
                        total_mechanisms += mechanisms.len();
                        total_alternatives += mechanisms
                            .iter()
                            .map(|mechanism| mechanism.alternatives.len())
                            .sum::<usize>();
                    }
                    black_box((total_mechanisms, total_alternatives))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("parameterized_raw_mechanism_sweep", id.clone()),
            &memory,
            |b, memory| {
                b.iter(|| {
                    let structural = FaultCatalog::from_circuit(black_box(&memory.circuit))
                        .expect("surface memory circuit should be supported");
                    let mut total_mechanisms = 0usize;
                    let mut total_alternatives = 0usize;
                    for noise in SWEEP_NOISES {
                        let catalog = structural.parameterized(noise);
                        let mechanisms = catalog.to_mechanisms();
                        total_mechanisms += mechanisms.len();
                        total_alternatives += mechanisms
                            .iter()
                            .map(|mechanism| mechanism.alternatives.len())
                            .sum::<usize>();
                    }
                    black_box((total_mechanisms, total_alternatives))
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("mutable_raw_mechanism_sweep", id),
            &memory,
            |b, memory| {
                b.iter(|| {
                    let mut catalog = FaultCatalog::from_circuit(black_box(&memory.circuit))
                        .expect("surface memory circuit should be supported");
                    let mut total_mechanisms = 0usize;
                    let mut total_alternatives = 0usize;
                    for noise in SWEEP_NOISES {
                        catalog.with_noise(black_box(noise));
                        let mechanisms = catalog.to_mechanisms();
                        total_mechanisms += mechanisms.len();
                        total_alternatives += mechanisms
                            .iter()
                            .map(|mechanism| mechanism.alternatives.len())
                            .sum::<usize>();
                    }
                    black_box((total_mechanisms, total_alternatives))
                });
            },
        );
    }

    group.finish();
}

fn surface_memory_circuits() -> Vec<MemoryCircuit> {
    DISTANCES
        .iter()
        .map(|&distance| {
            build_rotated_z_memory_circuit(distance, distance)
                .expect("surface memory circuit should build")
        })
        .collect()
}

fn representative_noise() -> StochasticNoiseParams {
    StochasticNoiseParams {
        p1: 0.0001,
        p2: 0.001,
        p_meas: 0.001,
        p_prep: 0.001,
    }
}

fn bench_id(memory: &MemoryCircuit, shape: &CatalogShape) -> BenchmarkId {
    BenchmarkId::from_parameter(bench_label(memory, shape))
}

fn bench_label(memory: &MemoryCircuit, shape: &CatalogShape) -> String {
    format!(
        "d{}_r{}_m{}_det{}_obs{}_loc{}_alt{}",
        memory.distance,
        memory.rounds,
        memory.num_measurements,
        memory.num_detectors,
        memory.num_observables,
        shape.locations,
        shape.alternatives,
    )
}

fn catalog_shape(circuit: &TickCircuit) -> CatalogShape {
    let catalog =
        FaultCatalog::from_circuit(circuit).expect("surface memory circuit should be supported");
    CatalogShape {
        locations: catalog.locations.len(),
        alternatives: count_alternatives(&catalog),
    }
}

fn count_alternatives(catalog: &FaultCatalog) -> usize {
    catalog
        .locations
        .iter()
        .map(|location| location.faults.len())
        .sum()
}

fn build_rotated_z_memory_circuit(distance: usize, rounds: usize) -> Result<MemoryCircuit, String> {
    let code = SurfaceCode::rotated(distance)?;
    let num_data = code.num_data_qubits();
    let x_ancilla_offset = num_data;
    let z_ancilla_offset = x_ancilla_offset + code.num_x_stabilizers();

    let x_ancilla = |idx: usize| x_ancilla_offset + idx;
    let z_ancilla = |idx: usize| z_ancilla_offset + idx;

    let mut circuit = TickCircuit::new();
    let data_qubits: Vec<usize> = (0..num_data).collect();
    circuit.tick().pz(&data_qubits);

    let mut x_round_measurements: Vec<Vec<TickMeasRef>> = Vec::with_capacity(rounds);
    let mut z_round_measurements: Vec<Vec<TickMeasRef>> = Vec::with_capacity(rounds);

    for _round in 0..rounds {
        let x_ancillas: Vec<usize> = (0..code.num_x_stabilizers()).map(x_ancilla).collect();
        let z_ancillas: Vec<usize> = (0..code.num_z_stabilizers()).map(z_ancilla).collect();

        circuit.tick().pz(&x_ancillas);
        circuit.tick().pz(&z_ancillas);
        circuit.tick().h(&x_ancillas);

        for check in code.x_stabilizers() {
            let ancilla = x_ancilla(check.index);
            for data in check.qubits() {
                circuit.tick().cx(&[(ancilla, data)]);
            }
        }

        for check in code.z_stabilizers() {
            let ancilla = z_ancilla(check.index);
            for data in check.qubits() {
                circuit.tick().cx(&[(data, ancilla)]);
            }
        }

        circuit.tick().h(&x_ancillas);

        let x_refs = circuit.tick().mz(&x_ancillas);
        let z_refs = circuit.tick().mz(&z_ancillas);
        x_round_measurements.push(x_refs);
        z_round_measurements.push(z_refs);
    }

    let final_data_measurements = circuit.tick().mz(&data_qubits);
    let num_measurements = circuit.num_measurements();

    let mut detectors: Vec<Vec<i32>> = Vec::new();

    for &meas_ref in z_round_measurements[0]
        .iter()
        .take(code.num_z_stabilizers())
    {
        detectors.push(relative_records(num_measurements, &[meas_ref]));
    }

    for round in 1..rounds {
        for (&current, &previous) in x_round_measurements[round]
            .iter()
            .zip(x_round_measurements[round - 1].iter())
            .take(code.num_x_stabilizers())
        {
            detectors.push(relative_records(num_measurements, &[current, previous]));
        }
        for (&current, &previous) in z_round_measurements[round]
            .iter()
            .zip(z_round_measurements[round - 1].iter())
            .take(code.num_z_stabilizers())
        {
            detectors.push(relative_records(num_measurements, &[current, previous]));
        }
    }

    let last_round = rounds - 1;
    for check in code.z_stabilizers() {
        let mut refs = vec![z_round_measurements[last_round][check.index]];
        refs.extend(
            check
                .qubits()
                .into_iter()
                .map(|q| final_data_measurements[q]),
        );
        detectors.push(relative_records(num_measurements, &refs));
    }

    let logical_z_refs: Vec<TickMeasRef> = code
        .logical_z()
        .data_qubits
        .iter()
        .map(|&q| final_data_measurements[q])
        .collect();
    let observables = vec![relative_records(num_measurements, &logical_z_refs)];

    circuit.set_meta(
        "num_measurements",
        Attribute::String(num_measurements.to_string()),
    );
    circuit.set_meta("detectors", Attribute::String(records_json(&detectors)));
    circuit.set_meta("observables", Attribute::String(records_json(&observables)));

    Ok(MemoryCircuit {
        circuit,
        distance,
        rounds,
        num_measurements,
        num_detectors: detectors.len(),
        num_observables: observables.len(),
    })
}

fn relative_records(num_measurements: usize, refs: &[TickMeasRef]) -> Vec<i32> {
    let num_measurements = i32::try_from(num_measurements).expect("measurement count fits in i32");
    refs.iter()
        .map(|meas_ref| {
            i32::try_from(meas_ref.record_idx).expect("measurement record index fits in i32")
                - num_measurements
        })
        .collect()
}

fn records_json(records: &[Vec<i32>]) -> String {
    let entries: Vec<String> = records
        .iter()
        .map(|records| {
            let values = records
                .iter()
                .map(i32::to_string)
                .collect::<Vec<_>>()
                .join(",");
            format!(r#"{{"records":[{values}]}}"#)
        })
        .collect();
    format!("[{}]", entries.join(","))
}

fn as_u64(value: usize) -> u64 {
    u64::try_from(value).expect("benchmark size fits in u64")
}
