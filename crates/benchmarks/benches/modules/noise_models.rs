// Copyright 2025 The PECOS Developers
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

//! Noise model benchmarks.
//!
//! This module benchmarks the noise model implementations, measuring
//! the overhead of noise application on quantum gates.

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos::prelude::*;
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_depolarizing_noise(c);
}

/// Benchmark the depolarizing noise model
#[allow(clippy::too_many_lines)]
fn bench_depolarizing_noise<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Noise Model: Depolarizing");

    // Test different gate counts
    for &num_gates in &[100, 1_000, 10_000] {
        group.throughput(Throughput::Elements(num_gates as u64));

        // Benchmark single-qubit gates with noise
        group.bench_with_input(
            BenchmarkId::new("single_qubit", num_gates),
            &num_gates,
            |b, &n| {
                let mut noise = DepolarizingNoiseModel::builder()
                    .with_uniform_probability(0.001)
                    .with_seed(42)
                    .build();

                // Create a message with n single-qubit gates
                let mut builder = ByteMessageBuilder::new();
                let _ = builder.for_quantum_operations();
                for i in 0..n {
                    builder.add_h(&[i % 100]); // Cycle through 100 qubits
                }
                let input = builder.build();

                b.iter(|| {
                    // Recreate with seed for reproducibility (no set_seed in public API)
                    noise = DepolarizingNoiseModel::builder()
                        .with_uniform_probability(0.001)
                        .with_seed(42)
                        .build();
                    let result = noise.start(input.clone()).unwrap();
                    black_box(result)
                });
            },
        );

        // Benchmark two-qubit gates with noise
        group.bench_with_input(
            BenchmarkId::new("two_qubit", num_gates),
            &num_gates,
            |b, &n| {
                let mut noise = DepolarizingNoiseModel::builder()
                    .with_uniform_probability(0.001)
                    .with_seed(42)
                    .build();

                // Create a message with n two-qubit gates
                let mut builder = ByteMessageBuilder::new();
                let _ = builder.for_quantum_operations();
                for i in 0..n {
                    let q0 = (i * 2) % 100;
                    let q1 = (i * 2 + 1) % 100;
                    builder.add_cx(&[q0], &[q1]);
                }
                let input = builder.build();

                b.iter(|| {
                    // Recreate with seed for reproducibility
                    noise = DepolarizingNoiseModel::builder()
                        .with_uniform_probability(0.001)
                        .with_seed(42)
                        .build();
                    let result = noise.start(input.clone()).unwrap();
                    black_box(result)
                });
            },
        );

        // Benchmark mixed gate set (more realistic)
        group.bench_with_input(BenchmarkId::new("mixed", num_gates), &num_gates, |b, &n| {
            let mut noise = DepolarizingNoiseModel::builder()
                .with_prep_probability(0.001)
                .with_meas_probability(0.001)
                .with_single_qubit_probability(0.0005)
                .with_two_qubit_probability(0.002)
                .with_seed(42)
                .build();

            // Create a mixed message
            let mut builder = ByteMessageBuilder::new();
            let _ = builder.for_quantum_operations();
            for i in 0..n {
                match i % 4 {
                    0 => {
                        builder.add_prep(&[i % 100]);
                    }
                    1 => {
                        builder.add_h(&[i % 100]);
                    }
                    2 => {
                        let q0 = (i * 2) % 100;
                        let q1 = (i * 2 + 1) % 100;
                        builder.add_cx(&[q0], &[q1]);
                    }
                    _ => {
                        builder.add_measurements(&[i % 100]);
                    }
                }
            }
            let input = builder.build();

            b.iter(|| {
                // Recreate with seed for reproducibility
                noise = DepolarizingNoiseModel::builder()
                    .with_prep_probability(0.001)
                    .with_meas_probability(0.001)
                    .with_single_qubit_probability(0.0005)
                    .with_two_qubit_probability(0.002)
                    .with_seed(42)
                    .build();
                let result = noise.start(input.clone()).unwrap();
                black_box(result)
            });
        });
    }

    // Benchmark with different error rates
    group.throughput(Throughput::Elements(10_000));
    for &error_rate in &[0.0001, 0.001, 0.01, 0.1] {
        group.bench_with_input(
            BenchmarkId::new("error_rate", format!("{error_rate}")),
            &error_rate,
            |b, &rate| {
                let mut noise = DepolarizingNoiseModel::builder()
                    .with_uniform_probability(rate)
                    .with_seed(42)
                    .build();

                // Create a message with 10k single-qubit gates
                let mut builder = ByteMessageBuilder::new();
                let _ = builder.for_quantum_operations();
                for i in 0..10_000 {
                    builder.add_h(&[i % 100]);
                }
                let input = builder.build();

                b.iter(|| {
                    // Recreate with seed for reproducibility
                    noise = DepolarizingNoiseModel::builder()
                        .with_uniform_probability(rate)
                        .with_seed(42)
                        .build();
                    let result = noise.start(input.clone()).unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}
