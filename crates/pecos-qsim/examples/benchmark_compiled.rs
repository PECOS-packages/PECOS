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

//! Benchmark comparing CompiledRunner to ad-hoc circuit execution.
//!
//! Run with:
//!   cargo run --release --example benchmark_compiled -p pecos-qsim

use pecos_core::QubitId;
use pecos_qsim::{CliffordGateable, CompiledRunner, CompiledSurfaceCode, SparseStab};
use std::hint::black_box;

/// Surface code parameters (same as profiling example)
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

/// Run surface code using ad-hoc approach (same as profiling example)
fn run_adhoc(sim: &mut SparseStab, params: &SurfaceCodeParams, rounds: usize) {
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

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let distance = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(11);
    let rounds = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let iterations = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let params = SurfaceCodeParams::new(distance);

    println!("Benchmark: CompiledRunner vs Ad-hoc execution");
    println!("  Distance: {}", distance);
    println!(
        "  Qubits: {} ({} data + {} ancilla)",
        params.num_qubits, params.num_data, params.num_ancillas
    );
    println!("  Rounds: {}", rounds);
    println!("  Iterations: {}", iterations);
    println!();

    // Warmup
    println!("Warming up...");
    for _ in 0..10 {
        let mut sim = SparseStab::new(params.num_qubits);
        run_adhoc(&mut sim, &params, rounds);
    }

    // Benchmark ad-hoc approach
    println!("Running ad-hoc benchmark...");
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let mut sim = SparseStab::new(params.num_qubits);
        run_adhoc(&mut sim, &params, rounds);
    }
    let adhoc_elapsed = start.elapsed();
    let adhoc_per_iter = adhoc_elapsed.as_micros() as f64 / iterations as f64;

    // Create compiled runner
    let compiled = CompiledSurfaceCode::rotated(distance, rounds);
    let runner = CompiledRunner::new(compiled);

    // Warmup compiled
    for _ in 0..10 {
        let mut sim = SparseStab::new(runner.num_qubits());
        black_box(runner.run(&mut sim));
    }

    // Benchmark compiled approach
    println!("Running compiled benchmark...");
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let mut sim = SparseStab::new(runner.num_qubits());
        black_box(runner.run(&mut sim));
    }
    let compiled_elapsed = start.elapsed();
    let compiled_per_iter = compiled_elapsed.as_micros() as f64 / iterations as f64;

    // Benchmark compiled with simulator reuse
    println!("Running compiled+reuse benchmark...");
    let mut sim = SparseStab::new(runner.num_qubits());
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        sim.reset();
        black_box(runner.run(&mut sim));
    }
    let compiled_reuse_elapsed = start.elapsed();
    let compiled_reuse_per_iter = compiled_reuse_elapsed.as_micros() as f64 / iterations as f64;

    println!();
    println!("Results:");
    println!("  Ad-hoc (new sim):      {:.1} us/iter", adhoc_per_iter);
    println!("  Compiled (new sim):    {:.1} us/iter", compiled_per_iter);
    println!("  Compiled (reuse sim):  {:.1} us/iter", compiled_reuse_per_iter);

    let speedup_compiled = adhoc_per_iter / compiled_per_iter;
    let speedup_reuse = adhoc_per_iter / compiled_reuse_per_iter;
    println!();
    println!("  Compiled vs Ad-hoc:    {:.2}x faster", speedup_compiled);
    println!("  Compiled+Reuse vs Ad-hoc: {:.2}x faster", speedup_reuse);
}
