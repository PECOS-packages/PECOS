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

//! Comparison benchmarks between pecos-neo and pecos-engines noise models.
//!
//! This module benchmarks the composable noise model (pecos-neo) against
//! the traditional noise models (pecos-engines) to measure any performance
//! differences.

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos::prelude::*;
use pecos_core::QubitId;
use pecos_neo::ecs::{ParallelConfig, ParallelCoordinator};
use pecos_neo::noise::GeneralNoiseModelBuilder as NeoNoiseModelBuilder;
use pecos_neo::prelude::{
    CommandBuilder, ComposableNoiseModel, CorePlugin, MeasurementChannel, PreparationChannel,
    SingleQubitChannel, TwoQubitChannel,
};
use pecos_neo::runner::CircuitRunner;
use pecos_neo::sampling::{MonteCarloConfig, MonteCarloRunner};
use pecos_simulators::SparseStab;
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_noise_application(c);
    bench_shot_execution(c);
    bench_monte_carlo_comparison(c);
}

/// Benchmark noise event emission and response processing
fn bench_noise_application<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pecos-neo: Noise Application");

    for &num_gates in &[100, 1_000, 10_000] {
        group.throughput(Throughput::Elements(num_gates as u64));

        // Benchmark pecos-engines DepolarizingNoiseModel
        group.bench_with_input(
            BenchmarkId::new("pecos-engines/depolarizing", num_gates),
            &num_gates,
            |b, &n| {
                let mut builder = ByteMessageBuilder::new();
                let _ = builder.for_quantum_operations();
                for i in 0..n {
                    builder.add_h(&[i % 100]);
                }
                let input = builder.build();

                b.iter(|| {
                    let mut noise = DepolarizingNoiseModel::builder()
                        .with_uniform_probability(0.001)
                        .with_seed(42)
                        .build();
                    let result = noise.start(input.clone()).unwrap();
                    black_box(result)
                });
            },
        );

        // Benchmark pecos-neo ComposableNoiseModel (channel-based)
        group.bench_with_input(
            BenchmarkId::new("pecos-neo/composable", num_gates),
            &num_gates,
            |b, &n| {
                // Build command queue once
                let mut builder = CommandBuilder::new();
                for i in 0..n {
                    builder = builder.pz(i % 100).h(i % 100);
                }
                let commands = builder.build();

                b.iter(|| {
                    let noise = ComposableNoiseModel::new()
                        .add_plugin(CorePlugin)
                        .add_channel(SingleQubitChannel::depolarizing(0.001));
                    let mut runner = CircuitRunner::<SparseStab>::new()
                        .with_noise(noise)
                        .with_seed(42);
                    let mut sim = SparseStab::new(100);
                    let result = runner.apply_circuit(&mut sim, &commands).unwrap();
                    black_box(result)
                });
            },
        );

        // Benchmark pecos-neo GeneralNoiseModelBuilder
        group.bench_with_input(
            BenchmarkId::new("pecos-neo/builder", num_gates),
            &num_gates,
            |b, &n| {
                let mut builder = CommandBuilder::new();
                for i in 0..n {
                    builder = builder.pz(i % 100).h(i % 100);
                }
                let commands = builder.build();

                b.iter(|| {
                    let noise = NeoNoiseModelBuilder::new().with_p1(0.001).build();
                    let mut runner = CircuitRunner::<SparseStab>::new()
                        .with_noise(noise)
                        .with_seed(42);
                    let mut sim = SparseStab::new(100);
                    let result = runner.apply_circuit(&mut sim, &commands).unwrap();
                    black_box(result)
                });
            },
        );
    }

    group.finish();
}

/// Benchmark full shot execution (circuit + noise + simulation)
fn bench_shot_execution<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pecos-neo: Shot Execution");

    // Benchmark Bell state circuit
    group.throughput(Throughput::Elements(100)); // 100 shots

    // pecos-engines with QuantumSystem
    group.bench_function("pecos-engines/bell_state", |b| {
        b.iter(|| {
            let noise = DepolarizingNoiseModel::builder()
                .with_uniform_probability(0.001)
                .with_seed(42)
                .build();
            let quantum = Box::new(pecos_engines::quantum::StateVecEngine::new(2));
            let mut system = QuantumSystem::new(Box::new(noise), quantum);

            for _ in 0..100 {
                let mut builder = ByteMessageBuilder::new();
                let _ = builder.for_quantum_operations();
                builder.add_h(&[0]);
                builder.add_cx(&[0], &[1]);
                builder.add_measurements(&[0, 1]);
                let circ = builder.build();

                system.reset().unwrap();
                let result = system.process(circ).unwrap();
                black_box(result);
            }
        });
    });

    // pecos-neo with CircuitRunner
    group.bench_function("pecos-neo/bell_state", |b| {
        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .h(0)
            .cx(0, 1)
            .mz(0)
            .mz(1)
            .build();

        b.iter(|| {
            let noise = ComposableNoiseModel::new()
                .add_plugin(CorePlugin)
                .add_channel(SingleQubitChannel::depolarizing(0.001))
                .add_channel(TwoQubitChannel::depolarizing(0.001));
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(noise)
                .with_seed(42);
            let mut sim = SparseStab::new(2);

            for _ in 0..100 {
                sim.reset();
                let result = runner.apply_circuit(&mut sim, &commands).unwrap();
                black_box(result);
            }
        });
    });

    // Benchmark with multiple noise channels
    group.bench_function("pecos-neo/multi_channel", |b| {
        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .h(0)
            .cx(0, 1)
            .mz(0)
            .mz(1)
            .build();

        b.iter(|| {
            let noise = ComposableNoiseModel::new()
                .add_plugin(CorePlugin)
                .add_channel(PreparationChannel::new(0.001))
                .add_channel(SingleQubitChannel::depolarizing(0.001))
                .add_channel(TwoQubitChannel::depolarizing(0.01))
                .add_channel(MeasurementChannel::symmetric(0.005));
            let mut runner = CircuitRunner::<SparseStab>::new()
                .with_noise(noise)
                .with_seed(42);
            let mut sim = SparseStab::new(2);

            for _ in 0..100 {
                sim.reset();
                let result = runner.apply_circuit(&mut sim, &commands).unwrap();
                black_box(result);
            }
        });
    });

    // Benchmark without noise (baseline)
    group.bench_function("pecos-neo/no_noise", |b| {
        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .h(0)
            .cx(0, 1)
            .mz(0)
            .mz(1)
            .build();

        b.iter(|| {
            let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
            let mut sim = SparseStab::new(2);

            for _ in 0..100 {
                sim.reset();
                let result = runner.apply_circuit(&mut sim, &commands).unwrap();
                black_box(result);
            }
        });
    });

    group.finish();
}

/// Benchmark Monte Carlo execution comparing different runners
fn bench_monte_carlo_comparison<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("pecos-neo: Monte Carlo Comparison");

    // Test with different shot counts
    for &num_shots in &[100, 1_000, 10_000] {
        group.throughput(Throughput::Elements(num_shots as u64));

        // pecos-neo MonteCarloRunner
        group.bench_with_input(
            BenchmarkId::new("MonteCarloRunner", num_shots),
            &num_shots,
            |b, &n| {
                let commands = CommandBuilder::new()
                    .pz(0)
                    .pz(1)
                    .h(0)
                    .cx(0, 1)
                    .mz(0)
                    .mz(1)
                    .build();

                b.iter(|| {
                    let config = MonteCarloConfig::new()
                        .with_shots(n)
                        .with_workers(4)
                        .with_seed(42);

                    let result = MonteCarloRunner::run(
                        &commands,
                        config,
                        || (CircuitRunner::new(), SparseStab::new(2)),
                        |outcomes| {
                            let b0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
                            let b1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);
                            (b0, b1)
                        },
                    );
                    black_box(result)
                });
            },
        );

        // pecos-neo ParallelCoordinator
        group.bench_with_input(
            BenchmarkId::new("ParallelCoordinator", num_shots),
            &num_shots,
            |b, &n| {
                b.iter(|| {
                    let config = ParallelConfig::new()
                        .with_workers(4)
                        .with_entities_per_worker(n / 4)
                        .with_seed(42);

                    let coordinator: ParallelCoordinator<SparseStab> =
                        ParallelCoordinator::new(config);

                    let result = coordinator.run(
                        || SparseStab::new(2),
                        |world| {
                            let commands = CommandBuilder::new()
                                .pz(0)
                                .pz(1)
                                .h(0)
                                .cx(0, 1)
                                .mz(0)
                                .mz(1)
                                .build();

                            world
                                .entities()
                                .map(|entity| {
                                    let sim = world.simulators.get(entity).unwrap();
                                    let rng = world.rngs.get(entity).unwrap();

                                    let mut runner = CircuitRunner::new().with_rng(rng.rng.clone());
                                    let mut sim_clone = sim.simulator.clone();
                                    let outcomes =
                                        runner.apply_circuit(&mut sim_clone, &commands).unwrap();
                                    let b0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
                                    let b1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);
                                    (b0, b1)
                                })
                                .collect()
                        },
                    );
                    black_box(result)
                });
            },
        );
    }

    group.finish();

    // Benchmark with noise
    let mut noisy_group = c.benchmark_group("pecos-neo: Monte Carlo with Noise");
    let num_shots = 1000;
    noisy_group.throughput(Throughput::Elements(num_shots as u64));

    // pecos-neo MonteCarloRunner with noise
    noisy_group.bench_function("MonteCarloRunner/depolarizing", |b| {
        let commands = CommandBuilder::new()
            .pz(0)
            .pz(1)
            .h(0)
            .cx(0, 1)
            .mz(0)
            .mz(1)
            .build();

        b.iter(|| {
            let config = MonteCarloConfig::new()
                .with_shots(num_shots as usize)
                .with_workers(4)
                .with_seed(42);

            let result = MonteCarloRunner::run(
                &commands,
                config,
                || {
                    let noise = NeoNoiseModelBuilder::new()
                        .with_p1(0.01)
                        .with_p2(0.01)
                        .build();
                    (CircuitRunner::new().with_noise(noise), SparseStab::new(2))
                },
                |outcomes| {
                    let b0 = outcomes.get_bit(QubitId(0)).unwrap_or(false);
                    let b1 = outcomes.get_bit(QubitId(1)).unwrap_or(false);
                    (b0, b1)
                },
            );
            black_box(result)
        });
    });

    noisy_group.finish();
}
