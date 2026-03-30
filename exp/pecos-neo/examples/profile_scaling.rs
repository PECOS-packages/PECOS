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

//! Scaling profiling for pecos-neo: measures per-operation cost at different qubit counts.
//!
//! Run with: `cargo run --release --example profile_scaling -p pecos-neo`

use pecos_neo::prelude::{
    CircuitRunner, CommandBuilder, ComposableNoiseModel, CorePlugin, SingleQubitChannel,
    TwoQubitChannel,
};
use pecos_simulators::SparseStab;
use std::time::Instant;

fn build_circuit(num_qubits: usize) -> pecos_neo::command::CommandQueue {
    let mut builder = CommandBuilder::new();

    // Prepare all qubits
    for q in 0..num_qubits {
        builder = builder.pz(&[q]);
    }

    // Layer of Hadamards
    for q in 0..num_qubits {
        builder = builder.h(&[q]);
    }

    // Layer of CX gates (nearest-neighbor)
    for q in (0..num_qubits - 1).step_by(2) {
        builder = builder.cx(&[(q, q + 1)]);
    }

    // Another layer of single-qubit gates
    for q in 0..num_qubits {
        builder = builder.sz(&[q]);
    }

    // Second CX layer (offset)
    for q in (1..num_qubits - 1).step_by(2) {
        builder = builder.cx(&[(q, q + 1)]);
    }

    // Measure all
    for q in 0..num_qubits {
        builder = builder.mz(&[q]);
    }

    builder.build()
}

fn bench_no_noise(num_qubits: usize, shots: usize) -> f64 {
    let commands = build_circuit(num_qubits);
    let num_gates = commands.len();

    let mut state = SparseStab::new(num_qubits);
    let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);

    // Warmup
    for _ in 0..10 {
        state.reset();
        runner.apply_circuit(&mut state, &commands).unwrap();
    }

    let start = Instant::now();
    for _ in 0..shots {
        state.reset();
        runner.apply_circuit(&mut state, &commands).unwrap();
    }
    let elapsed = start.elapsed();

    let per_shot = elapsed / shots as u32;
    let per_gate_ns = elapsed.as_nanos() as f64 / (shots * num_gates) as f64;
    println!(
        "  no-noise:    {num_qubits:>5} qubits, {num_gates:>6} gates, {shots:>6} shots | \
         {per_shot:>10.1?}/shot, {per_gate_ns:>8.1} ns/gate"
    );
    per_gate_ns
}

fn bench_with_noise(num_qubits: usize, shots: usize) -> f64 {
    let commands = build_circuit(num_qubits);
    let num_gates = commands.len();

    let noise = ComposableNoiseModel::new()
        .add_plugin(CorePlugin)
        .add_channel(SingleQubitChannel::depolarizing(0.001))
        .add_channel(TwoQubitChannel::depolarizing(0.01));

    let mut state = SparseStab::new(num_qubits);
    let mut runner = CircuitRunner::<SparseStab>::new()
        .with_noise(noise)
        .with_seed(42);

    // Warmup
    for _ in 0..10 {
        state.reset();
        runner.apply_circuit(&mut state, &commands).unwrap();
    }

    let start = Instant::now();
    for _ in 0..shots {
        state.reset();
        runner.apply_circuit(&mut state, &commands).unwrap();
    }
    let elapsed = start.elapsed();

    let per_shot = elapsed / shots as u32;
    let per_gate_ns = elapsed.as_nanos() as f64 / (shots * num_gates) as f64;
    println!(
        "  with-noise:  {num_qubits:>5} qubits, {num_gates:>6} gates, {shots:>6} shots | \
         {per_shot:>10.1?}/shot, {per_gate_ns:>8.1} ns/gate"
    );
    per_gate_ns
}

fn main() {
    println!("=== pecos-neo Scaling Profile ===\n");

    let configs: &[(usize, usize)] = &[
        (2, 100_000),
        (10, 50_000),
        (50, 10_000),
        (100, 5_000),
        (500, 1_000),
        (1_000, 500),
        (5_000, 50),
        (10_000, 20),
    ];

    println!("--- Per-qubit-count results ---\n");
    let mut results = Vec::new();
    for &(num_qubits, shots) in configs {
        let no_noise = bench_no_noise(num_qubits, shots);
        let with_noise = bench_with_noise(num_qubits, shots);
        let overhead = with_noise - no_noise;
        let overhead_pct = (overhead / no_noise) * 100.0;
        println!("  => noise overhead: {overhead:.1} ns/gate ({overhead_pct:.0}%)\n");
        results.push((num_qubits, no_noise, with_noise, overhead));
    }

    println!("\n--- Summary ---\n");
    println!(
        "{:>8} {:>14} {:>14} {:>14} {:>8}",
        "Qubits", "No-noise", "With-noise", "Overhead", "Pct"
    );
    for (q, nn, wn, oh) in &results {
        let pct = (oh / nn) * 100.0;
        println!("{q:>8} {nn:>12.1} ns {wn:>12.1} ns {oh:>12.1} ns {pct:>6.1}%");
    }

    println!("\n--- Scaling Analysis ---\n");

    // Check if per-gate cost scales linearly with qubits (expected for stabilizer)
    if results.len() >= 2 {
        let (q1, nn1, _, _) = results[0];
        let (q2, nn2, _, _) = results[results.len() - 1];
        let ratio = nn2 / nn1;
        let qubit_ratio = q2 as f64 / q1 as f64;
        println!(
            "Simulator scaling: {q1} -> {q2} qubits ({qubit_ratio:.0}x), \
             per-gate cost: {nn1:.1} -> {nn2:.1} ns ({ratio:.1}x)"
        );
        let (_, _, _, oh1) = results[0];
        let (_, _, _, oh2) = results[results.len() - 1];
        println!(
            "Noise overhead scaling: {oh1:.1} -> {oh2:.1} ns ({:.1}x)",
            oh2 / oh1
        );
    }
}
