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

//! Benchmark comparing StateVecSoA (f64) vs StateVecSoA32 (f32).
//!
//! Run with: cargo run --release --example f32_vs_f64_bench -p pecos-qsim

use pecos_core::{qid, qid2};
use pecos_qsim::{CliffordGateable, QuantumSimulator, StateVecSoA, StateVecSoA32};
use std::hint::black_box;
use std::time::Instant;

fn main() {
    println!("StateVecSoA (f64) vs StateVecSoA32 (f32) Benchmark");
    println!("==================================================\n");

    // H-gates only benchmark
    println!("--- H Gates Only ---\n");
    for num_qubits in [16, 20, 24] {
        let iterations = match num_qubits {
            16 => 500,
            20 => 100,
            _ => 10,
        };

        println!("{} qubits (H only):", num_qubits);

        let mut sim64 = StateVecSoA::with_seed(num_qubits, 42);
        let start = Instant::now();
        for _ in 0..iterations {
            for q in 0..num_qubits {
                black_box(&mut sim64).h(&qid(q));
            }
            sim64.reset();
        }
        let f64_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        let mut sim32 = StateVecSoA32::with_seed(num_qubits, 42);
        let start = Instant::now();
        for _ in 0..iterations {
            for q in 0..num_qubits {
                black_box(&mut sim32).h(&qid(q));
            }
            sim32.reset();
        }
        let f32_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        let speedup = f64_time / f32_time;
        println!("  f64: {:>8.3} ms, f32: {:>8.3} ms ({:.2}x speedup)\n", f64_time, f32_time, speedup);
    }

    println!("\n--- Full Circuit (H + CX) ---\n");

    for num_qubits in [16, 18, 20, 22, 24] {
        let size_mb_f64 = (1usize << num_qubits) * 16 / 1_000_000;
        let size_mb_f32 = (1usize << num_qubits) * 8 / 1_000_000;

        let iterations = match num_qubits {
            16 => 500,
            18 => 200,
            20 => 80,
            22 => 20,
            _ => 5,
        };

        println!(
            "{} qubits (f64: {} MB, f32: {} MB)",
            num_qubits, size_mb_f64, size_mb_f32
        );

        // f64 benchmark
        let mut sim64 = StateVecSoA::with_seed(num_qubits, 42);

        // Warmup
        for q in 0..num_qubits {
            sim64.h(&qid(q));
        }
        sim64.reset();

        let start = Instant::now();
        for _ in 0..iterations {
            // Apply H to all qubits
            for q in 0..num_qubits {
                black_box(&mut sim64).h(&qid(q));
            }
            // Apply some CX gates
            for q in 0..num_qubits - 1 {
                black_box(&mut sim64).cx(&qid2(q, q + 1));
            }
        }
        let f64_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        // f32 benchmark
        let mut sim32 = StateVecSoA32::with_seed(num_qubits, 42);

        // Warmup
        for q in 0..num_qubits {
            sim32.h(&qid(q));
        }
        sim32.reset();

        let start = Instant::now();
        for _ in 0..iterations {
            // Apply H to all qubits
            for q in 0..num_qubits {
                black_box(&mut sim32).h(&qid(q));
            }
            // Apply some CX gates
            for q in 0..num_qubits - 1 {
                black_box(&mut sim32).cx(&qid2(q, q + 1));
            }
        }
        let f32_time = start.elapsed().as_secs_f64() * 1000.0 / iterations as f64;

        let speedup = f64_time / f32_time;
        println!(
            "  f64 (StateVecSoA):   {:>8.3} ms/circuit",
            f64_time
        );
        println!(
            "  f32 (StateVecSoA32): {:>8.3} ms/circuit ({:.2}x speedup)\n",
            f32_time, speedup
        );
    }

    println!("Summary:");
    println!("--------");
    println!("f32 should be ~1.6-1.8x faster due to:");
    println!("  - Half the memory bandwidth (8 bytes vs 16 bytes per amplitude)");
    println!("  - Wider SIMD (f32x8 vs f64x4 - 8 vs 4 elements per vector)");
}
