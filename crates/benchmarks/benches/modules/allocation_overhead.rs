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

//! Microbenchmarks for allocation overhead in trait default implementations.
//!
//! Compares the overhead of Vec vs `SmallVec` for temporary qubit buffers.

use criterion::{BenchmarkId, Criterion, measurement::Measurement};
use pecos_core::QubitId;
use pecos_simulators::{CliffordGateable, SparseStab};
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_gate_allocation_overhead(c);
}

/// Benchmark gates that use temporary allocations in their default implementations.
fn bench_gate_allocation_overhead<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Allocation Overhead");
    group.sample_size(100);

    // Test different batch sizes (number of qubit pairs)
    let pair_counts = [1, 2, 4, 8, 16];

    for num_pairs in pair_counts {
        let num_qubits = num_pairs * 2 + 2; // Extra qubits to avoid overlap
        let label = format!("{num_pairs}_pairs");

        // Build qubit pairs for the batch
        let qubits: Vec<(QubitId, QubitId)> = (0..num_pairs)
            .map(|i| (QubitId(i * 2), QubitId(i * 2 + 1)))
            .collect();

        // Benchmark SZZ (uses default impl with SmallVec extraction)
        group.bench_with_input(
            BenchmarkId::new("SparseStab_SZZ", &label),
            &(num_qubits, &qubits),
            |b, &(nq, qs)| {
                let mut sim = SparseStab::new(nq);
                b.iter(|| {
                    sim.szz(qs);
                    black_box(());
                });
            },
        );

        // Benchmark CZ for comparison (commonly overridden but uses default in SparseStab)
        group.bench_with_input(
            BenchmarkId::new("SparseStab_CZ", &label),
            &(num_qubits, &qubits),
            |b, &(nq, qs)| {
                let mut sim = SparseStab::new(nq);
                b.iter(|| {
                    sim.cz(qs);
                    black_box(());
                });
            },
        );

        // Benchmark iSWAP (uses 3 temporary allocations in default impl)
        group.bench_with_input(
            BenchmarkId::new("SparseStab_iSWAP", &label),
            &(num_qubits, &qubits),
            |b, &(nq, qs)| {
                let mut sim = SparseStab::new(nq);
                b.iter(|| {
                    sim.iswap(qs);
                    black_box(());
                });
            },
        );
    }

    group.finish();
}
