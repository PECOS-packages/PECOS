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

//! Comprehensive CPU stabilizer simulator comparison benchmark.
//!
//! Compares all CPU-based stabilizer simulator implementations on surface code
//! syndrome extraction to determine relative performance. This helps inform
//! which implementation `Stab` should wrap as its default backend.
//!
//! Simulators compared:
//! - `DenseStab` (row+column dual representation)
//! - `DenseStabColOnly` (column-only dense)
//! - `DenseStabRowOnly` (row-only dense)
//! - `SparseColOnly` (column-only sparse `SmallVec`)
//! - `SparseRowOnly` (row-only sparse `SmallVec`)
//! - `SparseStab` (BitSet-based sparse, row+column)
//! - `GpuStab` (column-only dense, u32 words)
//! - `GpuStabOpt` (optimized `GpuStab` variant)
//! - `GpuStabParallel` (parallel `GpuStab` variant)

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos::prelude::*;
use pecos::qsim::{
    DenseStab, DenseStabColOnly, DenseStabRowOnly, GpuStab, GpuStabOpt, GpuStabParallel,
    SparseColOnly, SparseRowOnly, SparseStab,
};
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_cpu_stabilizer_surface_code(c);
}

/// Surface code parameters for a given distance.
struct SurfaceCodeParams {
    distance: usize,
    num_qubits: usize,
    num_data: usize,
    num_ancillas: usize,
    ancilla_start: usize,
}

impl SurfaceCodeParams {
    fn new(distance: usize) -> Self {
        let num_data = distance * distance;
        let num_ancillas = num_data - 1;
        let num_qubits = num_data + num_ancillas;
        Self {
            distance,
            num_qubits,
            num_data,
            num_ancillas,
            ancilla_start: num_data,
        }
    }

    fn ancilla_neighbors(&self, ancilla_idx: usize) -> Vec<usize> {
        let d = self.distance;
        let ancilla_local = ancilla_idx;

        let mut neighbors = Vec::with_capacity(4);
        let base = ancilla_local % self.num_data;
        neighbors.push(base);

        if ancilla_local + 1 < self.num_data {
            neighbors.push((base + 1) % self.num_data);
        }

        if ancilla_local < self.num_ancillas / 2 {
            if base + d < self.num_data {
                neighbors.push(base + d);
            }
            if ancilla_local > d && base >= d {
                neighbors.push(base - d);
            }
        } else {
            if base + d < self.num_data {
                neighbors.push(base + d);
            }
            if base + d + 1 < self.num_data {
                neighbors.push((base + d + 1) % self.num_data);
            }
        }

        neighbors
    }
}

/// Run surface code syndrome extraction on any `CliffordGateable + QuantumSimulator`.
fn run_circuit<S: CliffordGateable + QuantumSimulator>(
    sim: &mut S,
    params: &SurfaceCodeParams,
    rounds: usize,
) {
    // Initialize data qubits in |+> state
    for i in 0..params.num_data {
        sim.h(&[QubitId::from(i)]);
    }

    for _round in 0..rounds {
        for a in 0..params.num_ancillas {
            let ancilla = QubitId::from(params.ancilla_start + a);
            let neighbors = params.ancilla_neighbors(a);

            if a < params.num_ancillas / 2 {
                for &data in &neighbors {
                    sim.cx(&[ancilla, QubitId::from(data)]);
                }
            } else {
                for &data in &neighbors {
                    sim.cx(&[QubitId::from(data), ancilla]);
                }
            }
        }

        for a in 0..params.num_ancillas {
            let ancilla = QubitId::from(params.ancilla_start + a);
            sim.mz(&[ancilla]);
        }
    }
}

/// Compare all CPU stabilizer simulators on surface code syndrome extraction.
fn bench_cpu_stabilizer_surface_code<M: Measurement>(c: &mut Criterion<M>) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("CPU Stabilizer Comparison");

    for distance in [5, 11, 17] {
        let params = SurfaceCodeParams::new(distance);

        for rounds in [1, 5, 10] {
            let label = format!("d{distance}_r{rounds}");

            let ops_per_run = rounds * (params.num_ancillas * 3 + params.num_ancillas);
            group.throughput(Throughput::Elements(ops_per_run as u64));

            // --- DenseStab (row+col dual, what Stab currently wraps) ---
            group.bench_with_input(BenchmarkId::new("DenseStab", &label), &(), |b, ()| {
                b.iter_batched(
                    || {
                        let mut sim = DenseStab::new(params.num_qubits);
                        sim.reset();
                        sim
                    },
                    |mut sim| {
                        run_circuit(&mut sim, &params, rounds);
                        black_box(sim)
                    },
                    BatchSize::SmallInput,
                );
            });

            // --- DenseStabColOnly ---
            group.bench_with_input(
                BenchmarkId::new("DenseStabColOnly", &label),
                &(),
                |b, ()| {
                    b.iter_batched(
                        || {
                            let mut sim = DenseStabColOnly::new(params.num_qubits);
                            sim.reset();
                            sim
                        },
                        |mut sim| {
                            run_circuit(&mut sim, &params, rounds);
                            black_box(sim)
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            // --- DenseStabRowOnly ---
            group.bench_with_input(
                BenchmarkId::new("DenseStabRowOnly", &label),
                &(),
                |b, ()| {
                    b.iter_batched(
                        || {
                            let mut sim = DenseStabRowOnly::new(params.num_qubits);
                            sim.reset();
                            sim
                        },
                        |mut sim| {
                            run_circuit(&mut sim, &params, rounds);
                            black_box(sim)
                        },
                        BatchSize::SmallInput,
                    );
                },
            );

            // --- SparseColOnly ---
            group.bench_with_input(BenchmarkId::new("SparseColOnly", &label), &(), |b, ()| {
                b.iter_batched(
                    || {
                        let mut sim = SparseColOnly::new(params.num_qubits);
                        sim.reset();
                        sim
                    },
                    |mut sim| {
                        run_circuit(&mut sim, &params, rounds);
                        black_box(sim)
                    },
                    BatchSize::SmallInput,
                );
            });

            // --- SparseRowOnly ---
            group.bench_with_input(BenchmarkId::new("SparseRowOnly", &label), &(), |b, ()| {
                b.iter_batched(
                    || {
                        let mut sim = SparseRowOnly::new(params.num_qubits);
                        sim.reset();
                        sim
                    },
                    |mut sim| {
                        run_circuit(&mut sim, &params, rounds);
                        black_box(sim)
                    },
                    BatchSize::SmallInput,
                );
            });

            // --- SparseStab (BitSet-based, row+col) ---
            group.bench_with_input(BenchmarkId::new("SparseStab", &label), &(), |b, ()| {
                b.iter_batched(
                    || {
                        let mut sim = SparseStab::new(params.num_qubits);
                        sim.reset();
                        sim
                    },
                    |mut sim| {
                        run_circuit(&mut sim, &params, rounds);
                        black_box(sim)
                    },
                    BatchSize::SmallInput,
                );
            });

            // --- GpuStab (column-only, u32 words) ---
            group.bench_with_input(BenchmarkId::new("GpuStab", &label), &(), |b, ()| {
                b.iter_batched(
                    || {
                        let mut sim = GpuStab::new(params.num_qubits);
                        sim.reset();
                        sim
                    },
                    |mut sim| {
                        run_circuit(&mut sim, &params, rounds);
                        black_box(sim)
                    },
                    BatchSize::SmallInput,
                );
            });

            // --- GpuStabOpt ---
            group.bench_with_input(BenchmarkId::new("GpuStabOpt", &label), &(), |b, ()| {
                b.iter_batched(
                    || {
                        let mut sim = GpuStabOpt::new(params.num_qubits);
                        sim.reset();
                        sim
                    },
                    |mut sim| {
                        run_circuit(&mut sim, &params, rounds);
                        black_box(sim)
                    },
                    BatchSize::SmallInput,
                );
            });

            // --- GpuStabParallel ---
            group.bench_with_input(BenchmarkId::new("GpuStabParallel", &label), &(), |b, ()| {
                b.iter_batched(
                    || {
                        let mut sim = GpuStabParallel::new(params.num_qubits);
                        sim.reset();
                        sim
                    },
                    |mut sim| {
                        run_circuit(&mut sim, &params, rounds);
                        black_box(sim)
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}
