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

//! Large-scale simulation example demonstrating O(1) context operations.
//!
//! This example shows how `NoiseContext` scales efficiently to large qubit counts
//! using bit vector operations internally.
//!
//! The context provides:
//! - O(1) qubit state lookups (`is_leaked`, `is_active`, exists)
//! - O(1) state modifications (`mark_leaked`, `mark_prepared`, etc.)
//! - Efficient bitwise operations for crosstalk queries
//!
//! Run with: cargo run --example `large_scale` --release

use pecos_core::QubitId;
use pecos_neo::noise::NoiseContext;
use std::time::Instant;

fn main() {
    println!("=== Large-Scale Simulation Context Benchmark ===\n");

    // Test different scales
    for &num_qubits in &[1_000, 10_000, 100_000, 1_000_000] {
        println!("--- {} qubits ---", format_number(num_qubits));
        let time = benchmark_context(num_qubits);
        println!("  Time: {time:>10.3} ms\n");
    }
}

fn format_number(n: usize) -> String {
    if n >= 1_000_000 {
        format!("{}M", n / 1_000_000)
    } else if n >= 1_000 {
        format!("{}K", n / 1_000)
    } else {
        n.to_string()
    }
}

fn benchmark_context(num_qubits: usize) -> f64 {
    let start = Instant::now();

    // Create context with capacity hint for large simulations
    let mut ctx = NoiseContext::with_capacity(num_qubits);

    // Prepare all qubits
    for i in 0..num_qubits {
        ctx.mark_prepared(QubitId(i));
    }

    // Simulate some gates with leakage checks (O(1) each)
    let mut leaked_count = 0;
    for i in 0..num_qubits {
        if ctx.is_leaked(QubitId(i)) {
            leaked_count += 1;
        }
    }

    // Leak some qubits (every 1000th)
    for i in (0..num_qubits).step_by(1000) {
        ctx.mark_leaked(QubitId(i));
    }

    // Check crosstalk targets for a sample of qubits
    let sample_size = (num_qubits / 100).clamp(100, 1000);
    let mut total_targets = 0;
    for i in (0..num_qubits).step_by(num_qubits / sample_size) {
        let targets = ctx.crosstalk_targets(&[QubitId(i)]);
        total_targets += targets.len();
    }

    // Measure some qubits
    for i in (0..num_qubits).step_by(2) {
        ctx.mark_measured(QubitId(i));
    }

    // Final leaked count check
    let final_leaked = ctx.leaked_count();

    // Use values to prevent optimization
    std::hint::black_box((leaked_count, total_targets, final_leaked));

    start.elapsed().as_secs_f64() * 1000.0
}
