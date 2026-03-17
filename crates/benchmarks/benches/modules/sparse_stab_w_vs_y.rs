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

//! Benchmark comparing W-convention (`SparseStab`) vs Y-convention (`SparseStabY`)
//! sparse stabilizer simulators on surface code syndrome extraction.

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos::prelude::*;
use pecos::qsim::{SparseStab, SparseStabY};
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_w_vs_y_surface_code(c);
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
        let mut neighbors = Vec::with_capacity(4);
        let base = ancilla_idx % self.num_data;
        neighbors.push(base);

        if ancilla_idx + 1 < self.num_data {
            neighbors.push((base + 1) % self.num_data);
        }

        if ancilla_idx < self.num_ancillas / 2 {
            if base + d < self.num_data {
                neighbors.push(base + d);
            }
            if ancilla_idx > d && base >= d {
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

/// Compare W-convention (`SparseStab`) vs Y-convention (`SparseStabY`) on surface code.
fn bench_w_vs_y_surface_code<M: Measurement>(c: &mut Criterion<M>) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("SparseStab W vs Y Convention");

    for distance in [5, 11, 17] {
        let params = SurfaceCodeParams::new(distance);

        for rounds in [1, 5, 10] {
            let label = format!("d{distance}_r{rounds}");

            let ops_per_run = rounds * (params.num_ancillas * 3 + params.num_ancillas);
            group.throughput(Throughput::Elements(ops_per_run as u64));

            // --- SparseStab (W-convention) ---
            group.bench_with_input(BenchmarkId::new("SparseStab_W", &label), &(), |b, ()| {
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

            // --- SparseStabY (Y-convention) ---
            group.bench_with_input(BenchmarkId::new("SparseStabY", &label), &(), |b, ()| {
                b.iter_batched(
                    || {
                        let mut sim = SparseStabY::new(params.num_qubits);
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
