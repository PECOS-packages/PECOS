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

//! Profile CompiledRunner specifically.
//!
//! Run with:
//!   cargo build --release --example profile_compiled -p pecos-qsim
//!   perf record -g ./target/release/examples/profile_compiled

use pecos_qsim::{CompiledRunner, CompiledSurfaceCode, SparseStab};
use std::hint::black_box;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let distance = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(11);
    let rounds = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(5);
    let iterations = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(2000);

    let compiled = CompiledSurfaceCode::rotated(distance, rounds);
    let runner = CompiledRunner::new(compiled);

    println!("Profiling CompiledRunner:");
    println!("  Distance: {}", distance);
    println!("  Qubits: {}", runner.num_qubits());
    println!("  Rounds: {}", rounds);
    println!("  Iterations: {}", iterations);
    println!();

    // Warmup
    println!("Warming up...");
    for _ in 0..10 {
        let mut sim = SparseStab::new(runner.num_qubits());
        black_box(runner.run(&mut sim));
    }

    // Main profiling loop with simulator reuse
    println!("Running {} iterations with simulator reuse...", iterations);
    let mut sim = SparseStab::new(runner.num_qubits());
    let start = std::time::Instant::now();

    for _ in 0..iterations {
        sim.reset();
        black_box(runner.run(&mut sim));
    }

    let elapsed = start.elapsed();
    let per_iter = elapsed.as_micros() as f64 / iterations as f64;

    println!();
    println!("Results:");
    println!("  Total time: {:?}", elapsed);
    println!("  Per iteration: {:.1} us", per_iter);
    println!("  Throughput: {:.0} iter/sec", 1_000_000.0 / per_iter);
}
