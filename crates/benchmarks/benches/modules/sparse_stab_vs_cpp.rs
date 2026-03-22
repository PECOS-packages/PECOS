// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
// in compliance with the License.You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the License
// is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
// or implied. See the License for the specific language governing permissions and limitations under
// the License.

//! Performance comparison: Pure Rust `SparseStab` vs C++ `CppSparseStab` vs `Stab` (`DenseStab`).
//!
//! Benchmarks surface code syndrome extraction at various distances and round counts
//! to compare the three stabilizer simulator backends:
//!
//! - `SparseStab` (pure Rust, BitSet-based sparse representation)
//! - `CppSparseStab` (C++ implementation via cxx FFI)
//! - `Stab` (pure Rust, `DenseStab` with row+column bit-matrix layout)

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos::prelude::*;
use pecos::simulators::{SparseStab, Stab};
use pecos_cppsparsesim::CppSparseStab;
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_rust_vs_cpp_surface_code(c);
}

/// Surface code parameters for a given distance.
///
/// Uses the same layout as the main `surface_code` benchmarks for consistency.
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

    /// Get the neighbors of an ancilla (simplified model).
    /// Matches the pattern in the main `surface_code` benchmarks.
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

/// Run surface code syndrome extraction on `SparseStab` (pure Rust, BitSet-based).
fn run_circuit_sparse_stab(sim: &mut SparseStab, params: &SurfaceCodeParams, rounds: usize) {
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

/// Run surface code syndrome extraction on `CppSparseStab` (C++ via FFI).
fn run_circuit_cpp_sparse_stab(sim: &mut CppSparseStab, params: &SurfaceCodeParams, rounds: usize) {
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

/// Run surface code syndrome extraction on Stab (`DenseStab`, pure Rust).
fn run_circuit_stab(sim: &mut Stab, params: &SurfaceCodeParams, rounds: usize) {
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

/// Compare `SparseStab` (Rust) vs `CppSparseStab` (C++) vs Stab (`DenseStab`) on surface code
/// syndrome extraction across distances and round counts.
fn bench_rust_vs_cpp_surface_code<M: Measurement>(c: &mut Criterion<M>) {
    use criterion::BatchSize;

    let mut group = c.benchmark_group("Rust vs C++ vs DenseStab - Surface Code");

    for distance in [5, 11, 17] {
        let params = SurfaceCodeParams::new(distance);

        for rounds in [1, 5, 10, 20] {
            let label = format!("d{distance}_r{rounds}");

            // Throughput: ~3 CNOTs + 1 measurement per ancilla per round
            let ops_per_run = rounds * (params.num_ancillas * 3 + params.num_ancillas);
            group.throughput(Throughput::Elements(ops_per_run as u64));

            // --- Pure Rust SparseStab (BitSet-based) ---
            group.bench_with_input(BenchmarkId::new("SparseStab_Rust", &label), &(), |b, ()| {
                b.iter_batched(
                    || {
                        let mut sim = SparseStab::new(params.num_qubits);
                        sim.reset();
                        sim
                    },
                    |mut sim| {
                        run_circuit_sparse_stab(&mut sim, &params, rounds);
                        black_box(sim)
                    },
                    BatchSize::SmallInput,
                );
            });

            // --- C++ SparseStab (via cxx FFI) ---
            group.bench_with_input(BenchmarkId::new("SparseStab_Cpp", &label), &(), |b, ()| {
                b.iter_batched(
                    || {
                        let mut sim = CppSparseStab::with_seed(params.num_qubits, 42);
                        sim.reset();
                        sim
                    },
                    |mut sim| {
                        run_circuit_cpp_sparse_stab(&mut sim, &params, rounds);
                        black_box(sim)
                    },
                    BatchSize::SmallInput,
                );
            });

            // --- DenseStab (Stab, pure Rust) ---
            group.bench_with_input(BenchmarkId::new("Stab_DenseRust", &label), &(), |b, ()| {
                b.iter_batched(
                    || {
                        let mut sim = Stab::new(params.num_qubits);
                        sim.reset();
                        sim
                    },
                    |mut sim| {
                        run_circuit_stab(&mut sim, &params, rounds);
                        black_box(sim)
                    },
                    BatchSize::SmallInput,
                );
            });
        }
    }

    group.finish();
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_all_three_complete_without_panic() {
        let params = SurfaceCodeParams::new(5);
        let rounds = 3;

        let mut rust_sim = SparseStab::new(params.num_qubits);
        rust_sim.reset();
        run_circuit_sparse_stab(&mut rust_sim, &params, rounds);

        let mut cpp_sim = CppSparseStab::with_seed(params.num_qubits, 42);
        cpp_sim.reset();
        run_circuit_cpp_sparse_stab(&mut cpp_sim, &params, rounds);

        let mut stab_sim = Stab::new(params.num_qubits);
        stab_sim.reset();
        run_circuit_stab(&mut stab_sim, &params, rounds);
    }
}
