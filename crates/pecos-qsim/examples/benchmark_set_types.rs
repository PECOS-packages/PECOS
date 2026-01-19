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

//! Benchmark comparing different set implementations for stabilizer simulation.
//!
//! Run with:
//!   cargo run --release --example benchmark_set_types -p pecos-qsim
//!
//! Options:
//!   cargo run --release --example benchmark_set_types -p pecos-qsim -- <distance> <rounds> <iterations>

use pecos_core::QubitId;
use pecos_qsim::{
    CliffordGateable, DenseStab, DenseStabColOnly, DenseStabRowOnly, SparseColOnly, SparseStab,
    SparseStabHybrid, SparseStabUnsortedVecSet, SparseStabVecSet,
};
use std::hint::black_box;

/// Surface code parameters
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

/// Run surface code syndrome extraction
fn run_surface_code<S: CliffordGateable>(sim: &mut S, params: &SurfaceCodeParams, rounds: usize) {
    // Initialize data qubits in |+> state
    for i in 0..params.num_data {
        sim.h(&[QubitId(i)]);
    }

    // Syndrome extraction rounds
    for _round in 0..rounds {
        // CX gates for syndrome extraction
        for a in 0..params.num_ancillas {
            let ancilla = QubitId(params.ancilla_start + a);
            let neighbors = params.ancilla_neighbors(a);

            if a < params.num_ancillas / 2 {
                for &data in &neighbors {
                    sim.cx(&[ancilla, QubitId(data)]);
                }
            } else {
                for &data in &neighbors {
                    sim.cx(&[QubitId(data), ancilla]);
                }
            }
        }

        // Measure all ancillas
        for a in 0..params.num_ancillas {
            let ancilla = QubitId(params.ancilla_start + a);
            black_box(sim.mz(&[ancilla]));
        }
    }
}

fn benchmark<S, F>(name: &str, params: &SurfaceCodeParams, rounds: usize, iterations: usize, mut make_sim: F) -> f64
where
    S: CliffordGateable,
    F: FnMut() -> S,
{
    // Warmup
    for _ in 0..10 {
        let mut sim = make_sim();
        run_surface_code(&mut sim, params, rounds);
    }

    // Benchmark
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let mut sim = make_sim();
        run_surface_code(&mut sim, params, rounds);
    }
    let elapsed = start.elapsed();
    let per_iter = elapsed.as_micros() as f64 / iterations as f64;

    println!("  {:<12} {:>8.1} us/iter", name, per_iter);
    per_iter
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let distance = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(7);
    let rounds = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let iterations = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let params = SurfaceCodeParams::new(distance);

    println!("Benchmark: Set Type Comparison");
    println!("  Distance: {}", distance);
    println!(
        "  Qubits: {} ({} data + {} ancilla)",
        params.num_qubits, params.num_data, params.num_ancillas
    );
    println!("  Rounds: {}", rounds);
    println!("  Iterations: {}", iterations);
    println!();

    println!("Running benchmarks...");

    let bitset = benchmark("BitSet", &params, rounds, iterations, || {
        SparseStab::new(params.num_qubits)
    });

    let vecset = benchmark("VecSet", &params, rounds, iterations, || {
        SparseStabVecSet::new(params.num_qubits)
    });

    let unsorted = benchmark("Unsorted", &params, rounds, iterations, || {
        SparseStabUnsortedVecSet::new(params.num_qubits)
    });

    let hybrid = benchmark("Hybrid", &params, rounds, iterations, || {
        SparseStabHybrid::new(params.num_qubits)
    });

    let dense = benchmark("Dense", &params, rounds, iterations, || {
        DenseStab::new(params.num_qubits)
    });

    let col_only = benchmark("ColOnly", &params, rounds, iterations, || {
        DenseStabColOnly::new(params.num_qubits)
    });

    let row_only = benchmark("RowOnly", &params, rounds, iterations, || {
        DenseStabRowOnly::new(params.num_qubits)
    });

    let sparse_col = benchmark("SparseCol", &params, rounds, iterations, || {
        SparseColOnly::new(params.num_qubits)
    });

    println!();
    println!("Summary:");
    let best = bitset
        .min(vecset)
        .min(unsorted)
        .min(hybrid)
        .min(dense)
        .min(col_only)
        .min(row_only)
        .min(sparse_col);
    println!("  BitSet:    {:>8.1} us ({:.2}x)", bitset, bitset / best);
    println!("  VecSet:    {:>8.1} us ({:.2}x)", vecset, vecset / best);
    println!("  Unsorted:  {:>8.1} us ({:.2}x)", unsorted, unsorted / best);
    println!("  Hybrid:    {:>8.1} us ({:.2}x)", hybrid, hybrid / best);
    println!("  Dense:     {:>8.1} us ({:.2}x)", dense, dense / best);
    println!("  ColOnly:   {:>8.1} us ({:.2}x)", col_only, col_only / best);
    println!("  RowOnly:   {:>8.1} us ({:.2}x)", row_only, row_only / best);
    println!("  SparseCol: {:>8.1} us ({:.2}x)", sparse_col, sparse_col / best);
}
