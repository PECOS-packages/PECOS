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

//! Benchmarks for hot path operations in pecos-neo.
//!
//! These benchmarks focus on the performance-critical paths:
//! - Noise channel emission
//! - Shot execution
//! - Entity operations (spawn, clone, split)
//! - Redistribution operations

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use pecos_core::QubitId;
use pecos_neo::command::{CommandBuilder, GateType};
use pecos_neo::ecs::{WorkerState, World, redistribute_by_weight};
use pecos_neo::noise::{
    ComposableNoiseModel, MeasurementChannel, NoiseEvent, SingleQubitChannel, TwoQubitChannel,
};
use pecos_neo::runner::CircuitRunner;
use pecos_random::PecosRng;
use pecos_simulators::{CliffordGateable, SparseStab};
use rand::RngExt;
use std::hint::black_box;

/// Helper to create a noise model (since it doesn't implement Clone)
fn make_noise_model() -> ComposableNoiseModel {
    ComposableNoiseModel::new()
        .add_channel(SingleQubitChannel::depolarizing(0.001))
        .add_channel(TwoQubitChannel::depolarizing(0.01))
        .add_channel(MeasurementChannel::symmetric(0.01))
}

// ============================================================================
// Noise Channel Benchmarks
// ============================================================================

fn bench_noise_emission(c: &mut Criterion) {
    let mut group = c.benchmark_group("noise_emission");

    let mut noise = make_noise_model();
    let mut rng = PecosRng::seed_from_u64(42);
    let qubits = vec![QubitId(0)];
    let qubits_2q = vec![QubitId(0), QubitId(1)];

    group.throughput(Throughput::Elements(1));

    group.bench_function("after_gate_1q", |b| {
        b.iter(|| {
            let event = NoiseEvent::AfterGate {
                gate_type: GateType::H,
                qubits: &qubits,
                angles: &[],
                gate_id: None,
            };
            black_box(noise.emit(event, &mut rng))
        });
    });

    group.bench_function("after_gate_2q", |b| {
        b.iter(|| {
            let event = NoiseEvent::AfterGate {
                gate_type: GateType::CX,
                qubits: &qubits_2q,
                angles: &[],
                gate_id: None,
            };
            black_box(noise.emit(event, &mut rng))
        });
    });

    group.bench_function("after_measurement", |b| {
        b.iter(|| {
            let event = NoiseEvent::AfterMeasurement {
                qubits: &qubits,
                outcomes: &[false],
            };
            black_box(noise.emit(event, &mut rng))
        });
    });

    group.finish();
}

// ============================================================================
// Shot Execution Benchmarks
// ============================================================================

fn bench_shot_execution(c: &mut Criterion) {
    let mut group = c.benchmark_group("shot_execution");

    // Bell state circuit
    let bell_circuit = CommandBuilder::new()
        .pz_all(0..2)
        .h(0)
        .cx(0, 1)
        .mz_all(0..2)
        .build();

    // Larger circuit: 10 qubits, many gates
    let large_circuit = {
        let mut builder = CommandBuilder::new();
        builder = builder.pz_all(0..10);
        for i in 0..10 {
            builder = builder.h(i);
        }
        for i in 0..9 {
            builder = builder.cx(i, i + 1);
        }
        builder = builder.mz_all(0..10);
        builder.build()
    };

    // No noise
    group.bench_function("bell_no_noise", |b| {
        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new();
        b.iter(|| {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &bell_circuit).unwrap();
            black_box(outcomes)
        });
    });

    // With noise
    group.bench_function("bell_with_noise", |b| {
        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new().with_noise(make_noise_model());
        b.iter(|| {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &bell_circuit).unwrap();
            black_box(outcomes)
        });
    });

    // Larger circuit
    group.bench_function("10q_circuit_no_noise", |b| {
        let mut state = SparseStab::new(10);
        let mut runner = CircuitRunner::<SparseStab>::new();
        b.iter(|| {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &large_circuit).unwrap();
            black_box(outcomes)
        });
    });

    group.bench_function("10q_circuit_with_noise", |b| {
        let mut state = SparseStab::new(10);
        let mut runner = CircuitRunner::<SparseStab>::new().with_noise(make_noise_model());
        b.iter(|| {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &large_circuit).unwrap();
            black_box(outcomes)
        });
    });

    group.finish();
}

// ============================================================================
// Multi-Shot Benchmarks
// ============================================================================

fn bench_multi_shot(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_shot");

    let bell_circuit = CommandBuilder::new()
        .pz_all(0..2)
        .h(0)
        .cx(0, 1)
        .mz_all(0..2)
        .build();

    for shots in [100, 1000] {
        group.throughput(Throughput::Elements(shots as u64));

        group.bench_function(BenchmarkId::new("bell_no_noise", shots), |b| {
            b.iter(|| {
                let mut state = SparseStab::new(2);
                let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
                for _ in 0..shots {
                    state.reset();
                    black_box(runner.apply_circuit(&mut state, &bell_circuit).unwrap());
                }
            });
        });

        group.bench_function(BenchmarkId::new("bell_with_noise", shots), |b| {
            b.iter(|| {
                let mut state = SparseStab::new(2);
                let mut runner = CircuitRunner::<SparseStab>::new()
                    .with_noise(make_noise_model())
                    .with_seed(42);
                for _ in 0..shots {
                    state.reset();
                    black_box(runner.apply_circuit(&mut state, &bell_circuit).unwrap());
                }
            });
        });
    }

    // Compare clone-per-shot vs reset-per-shot for larger qubit counts
    let large_circuit = {
        let mut builder = CommandBuilder::new();
        builder = builder.pz_all(0..50);
        for i in 0..50 {
            builder = builder.h(i);
        }
        for i in 0..49 {
            builder = builder.cx(i, i + 1);
        }
        builder = builder.mz_all(0..50);
        builder.build()
    };

    // Clone-per-shot pattern (old approach - what coordinator does)
    group.bench_function("50q_clone_per_shot_100", |b| {
        let base_sim = SparseStab::new(50);
        b.iter(|| {
            for _ in 0..100 {
                let mut state = base_sim.clone();
                let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
                black_box(runner.apply_circuit(&mut state, &large_circuit).unwrap());
            }
        });
    });

    // reset-per-shot pattern (optimized approach)
    group.bench_function("50q_fresh_shot_100", |b| {
        b.iter(|| {
            let mut state = SparseStab::new(50);
            let mut runner = CircuitRunner::<SparseStab>::new().with_seed(42);
            for _ in 0..100 {
                state.reset();
                black_box(runner.apply_circuit(&mut state, &large_circuit).unwrap());
            }
        });
    });

    group.finish();
}

// ============================================================================
// Entity Operation Benchmarks
// ============================================================================

fn bench_entity_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("entity_operations");

    // Spawn benchmark
    group.bench_function("spawn_100_entities", |b| {
        b.iter(|| {
            let mut world: World<SparseStab> = World::new(42);
            for _ in 0..100 {
                black_box(world.spawn_with_simulator(SparseStab::new(1)));
            }
        });
    });

    // Clone benchmark
    group.bench_function("clone_entity_10q", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut world: World<SparseStab> = World::new(42);
                let entity = world.spawn_with_simulator(SparseStab::new(10));

                let start = std::time::Instant::now();
                black_box(world.clone_entity(entity));
                total += start.elapsed();
            }
            total
        });
    });

    // Split benchmark
    group.bench_function("split_entity_4", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for _ in 0..iters {
                let mut world: World<SparseStab> = World::new(42);
                let entity = world.spawn_with_simulator(SparseStab::new(10));

                let start = std::time::Instant::now();
                black_box(world.split_entity(entity, 4));
                total += start.elapsed();
            }
            total
        });
    });

    // Resample benchmark
    group.bench_function("resample_100_to_100", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for i in 0..iters {
                let mut world: World<SparseStab> = World::new(42);
                for _ in 0..100 {
                    world.spawn_with_simulator(SparseStab::new(1));
                }
                let mut rng = PecosRng::seed_from_u64(i);

                let start = std::time::Instant::now();
                black_box(world.resample_by_weight(100, &mut rng));
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

// ============================================================================
// Redistribution Benchmarks
// ============================================================================

fn bench_redistribution(c: &mut Criterion) {
    let mut group = c.benchmark_group("redistribution");

    for (num_workers, entities_per_worker) in [(2, 50), (4, 50), (4, 100)] {
        let name = format!("{num_workers}w_{entities_per_worker}e");

        group.bench_function(BenchmarkId::new("redistribute", &name), |b| {
            b.iter_custom(|iters| {
                let mut total = std::time::Duration::ZERO;
                for i in 0..iters {
                    let mut workers: Vec<WorkerState<SparseStab>> = (0..num_workers)
                        .map(|id| {
                            let mut worker = WorkerState::new(id, 42);
                            for j in 0..entities_per_worker {
                                let e = worker.world.spawn_with_simulator(SparseStab::new(1));
                                worker.world.weights.get_mut(e).unwrap().weight =
                                    pecos_neo::sampling::SampleWeight::from_linear((j + 1) as f64);
                            }
                            worker
                        })
                        .collect();
                    let mut rng = PecosRng::seed_from_u64(i);

                    let start = std::time::Instant::now();
                    black_box(redistribute_by_weight(
                        &mut workers,
                        entities_per_worker,
                        &mut rng,
                    ));
                    total += start.elapsed();
                }
                total
            });
        });
    }

    group.finish();
}

// ============================================================================
// RNG Benchmarks
// ============================================================================

fn bench_rng_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("rng");

    group.bench_function("seed_from_u64", |b| {
        b.iter(|| black_box(PecosRng::seed_from_u64(black_box(42))));
    });

    group.bench_function("random_f64", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| black_box(rng.random::<f64>()));
    });

    group.bench_function("random_bool", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| black_box(rng.random::<bool>()));
    });

    // Probability threshold check (used in noise channels)
    group.bench_function("probability_check", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| black_box(rng.random::<f64>() < 0.01));
    });

    group.finish();
}

// ============================================================================
// Simulator Benchmarks
// ============================================================================

fn bench_simulator_ops(c: &mut Criterion) {
    let mut group = c.benchmark_group("simulator");

    // Gate application
    group.bench_function("h_gate", |b| {
        let mut sim = SparseStab::new(10);
        b.iter(|| {
            sim.h(&[QubitId(0)]);
        });
    });

    group.bench_function("cx_gate", |b| {
        let mut sim = SparseStab::new(10);
        b.iter(|| {
            sim.cx(&[QubitId(0), QubitId(1)]);
        });
    });

    // Clone (important for splitting)
    for num_qubits in [10, 50, 100] {
        group.bench_function(BenchmarkId::new("clone", num_qubits), |b| {
            let sim = SparseStab::new(num_qubits);
            b.iter(|| black_box(sim.clone()));
        });
    }

    // Reset (alternative to clone for shot execution)
    for num_qubits in [10, 50, 100] {
        group.bench_function(BenchmarkId::new("reset", num_qubits), |b| {
            let mut sim = SparseStab::new(num_qubits);
            // Modify state first to simulate realistic scenario
            sim.h(&[QubitId(0)]);
            sim.cx(&[QubitId(0), QubitId(1 % num_qubits)]);
            b.iter(|| {
                sim.reset();
                black_box(&sim);
            });
        });
    }

    // Compare clone vs reset for "dirty" simulators (after some gates)
    for num_qubits in [10, 50, 100] {
        group.bench_function(BenchmarkId::new("clone_dirty", num_qubits), |b| {
            let mut sim = SparseStab::new(num_qubits);
            // Apply some gates to create a more complex state
            for i in 0..num_qubits {
                sim.h(&[QubitId(i)]);
            }
            for i in 0..num_qubits.saturating_sub(1) {
                sim.cx(&[QubitId(i), QubitId(i + 1)]);
            }
            b.iter(|| black_box(sim.clone()));
        });
    }

    group.finish();
}

// ============================================================================
// Memory Usage Benchmarks
// ============================================================================

fn bench_memory_usage(c: &mut Criterion) {
    use pecos_core::Angle64;
    use pecos_neo::command::GateCommand;
    use smallvec::SmallVec;

    let mut group = c.benchmark_group("memory");

    // Print sizes of key types (these are constant, so we measure allocation patterns)
    println!("\n=== Type Sizes ===");
    println!(
        "SparseStab (stack): {} bytes",
        std::mem::size_of::<SparseStab>()
    );
    println!(
        "World<SparseStab> (stack): {} bytes",
        std::mem::size_of::<World<SparseStab>>()
    );
    println!(
        "ComposableNoiseModel (stack): {} bytes",
        std::mem::size_of::<ComposableNoiseModel>()
    );
    println!(
        "CircuitRunner<SparseStab> (stack): {} bytes",
        std::mem::size_of::<pecos_neo::runner::CircuitRunner<SparseStab>>()
    );
    println!(
        "WorkerState<SparseStab> (stack): {} bytes",
        std::mem::size_of::<WorkerState<SparseStab>>()
    );

    // Detailed breakdown of NoiseResponse size
    println!("\n=== NoiseResponse Size Breakdown ===");
    println!("QubitId: {} bytes", std::mem::size_of::<QubitId>());
    println!("Angle64: {} bytes", std::mem::size_of::<Angle64>());
    println!("GateType: {} bytes", std::mem::size_of::<GateType>());
    println!(
        "SmallVec<[QubitId; 4]>: {} bytes",
        std::mem::size_of::<SmallVec<[QubitId; 4]>>()
    );
    println!(
        "SmallVec<[Angle64; 2]>: {} bytes",
        std::mem::size_of::<SmallVec<[Angle64; 2]>>()
    );
    println!("GateCommand: {} bytes", std::mem::size_of::<GateCommand>());
    println!(
        "SmallVec<[GateCommand; 4]>: {} bytes",
        std::mem::size_of::<SmallVec<[GateCommand; 4]>>()
    );
    println!(
        "SmallVec<[GateCommand; 2]>: {} bytes",
        std::mem::size_of::<SmallVec<[GateCommand; 2]>>()
    );
    println!(
        "SmallVec<[GateCommand; 1]>: {} bytes",
        std::mem::size_of::<SmallVec<[GateCommand; 1]>>()
    );
    println!(
        "NoiseResponse: {} bytes",
        std::mem::size_of::<pecos_neo::noise::NoiseResponse>()
    );
    println!(
        "NoiseEvent: {} bytes",
        std::mem::size_of::<pecos_neo::noise::NoiseEvent>()
    );

    // Measure allocation patterns by creating many entities
    group.bench_function("world_spawn_1000_entities", |b| {
        b.iter(|| {
            let mut world: World<SparseStab> = World::new(42);
            for _ in 0..1000 {
                world.spawn_with_simulator(SparseStab::new(10));
            }
            black_box(world)
        });
    });

    // Measure redistribution memory overhead
    group.bench_function("redistribution_memory_4w_100e", |b| {
        b.iter_custom(|iters| {
            let mut total = std::time::Duration::ZERO;
            for i in 0..iters {
                let mut workers: Vec<WorkerState<SparseStab>> = (0..4)
                    .map(|id| {
                        let mut worker = WorkerState::new(id, 42);
                        for _ in 0..100 {
                            worker.world.spawn_with_simulator(SparseStab::new(10));
                        }
                        worker
                    })
                    .collect();
                let mut rng = PecosRng::seed_from_u64(i);

                let start = std::time::Instant::now();
                black_box(redistribute_by_weight(&mut workers, 100, &mut rng));
                total += start.elapsed();
            }
            total
        });
    });

    group.finish();
}

// ============================================================================
// Monte Carlo Engine Comparison Benchmarks
// ============================================================================

/// Compare pecos-neo `MonteCarloRunner` against pecos-engines `MonteCarloEngine`
///
/// Both systems use `QASMEngine` for parsing, with QASM parsing done once in setup
/// (outside the timed section) to ensure a fair comparison of Monte Carlo execution.
fn bench_monte_carlo_comparison(c: &mut Criterion) {
    use criterion::BatchSize;
    use pecos_engines::{MonteCarloEngine, PassThroughNoiseModel};
    use pecos_neo::noise::GeneralNoiseModelBuilder;
    use pecos_neo::sampling::{MonteCarloConfig, MonteCarloRunner};
    use pecos_qasm::QASMEngine;
    use std::str::FromStr;

    let mut group = c.benchmark_group("monte_carlo_comparison");
    group.sample_size(20); // Reduce sample size since these are slower

    // Bell state circuit QASM
    let bell_qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[2];
        creg c[2];
        h q[0];
        cx q[0], q[1];
        measure q[0] -> c[0];
        measure q[1] -> c[1];
    "#;

    // Pre-build commands for pecos-neo (equivalent to pre-parsing QASM)
    let bell_commands = CommandBuilder::new()
        .pz_all(0..2)
        .h(0)
        .cx(0, 1)
        .mz_all(0..2)
        .build();

    for num_shots in [100, 1000] {
        group.throughput(Throughput::Elements(num_shots as u64));

        // pecos-engines MonteCarloEngine (no noise)
        // QASM parsing done in setup, only Monte Carlo execution is timed
        group.bench_function(BenchmarkId::new("engines_bell_no_noise", num_shots), |b| {
            b.iter_batched(
                || {
                    // Setup: parse QASM once (not timed)
                    QASMEngine::from_str(bell_qasm).unwrap()
                },
                |engine| {
                    // Timed: Monte Carlo execution only
                    let results = MonteCarloEngine::run_with_noise_model(
                        Box::new(engine),
                        Box::new(PassThroughNoiseModel::builder().build()),
                        num_shots,
                        1,
                        Some(42),
                    );
                    black_box(results)
                },
                BatchSize::SmallInput,
            );
        });

        // pecos-neo MonteCarloRunner (no noise)
        group.bench_function(BenchmarkId::new("neo_bell_no_noise", num_shots), |b| {
            b.iter(|| {
                let config = MonteCarloConfig::new()
                    .with_shots(num_shots)
                    .with_workers(1)
                    .with_seed(42);

                let results = MonteCarloRunner::run(
                    &bell_commands,
                    config,
                    || (CircuitRunner::new(), SparseStab::new(2)),
                    |outcomes| {
                        black_box((
                            outcomes.get_bit(QubitId(0)).unwrap_or(false),
                            outcomes.get_bit(QubitId(1)).unwrap_or(false),
                        ))
                    },
                );
                black_box(results)
            });
        });
    }

    // With noise - compare at 1000 shots
    let num_shots = 1000;
    group.throughput(Throughput::Elements(num_shots as u64));

    // pecos-engines MonteCarloEngine (with noise)
    group.bench_function(
        BenchmarkId::new("engines_bell_with_noise", num_shots),
        |b| {
            use pecos_engines::noise::GeneralNoiseModel;
            b.iter_batched(
                || {
                    // Setup: parse QASM and create noise model (not timed)
                    let engine = QASMEngine::from_str(bell_qasm).unwrap();
                    let noise = GeneralNoiseModel::builder()
                        .with_average_p1_probability(0.001)
                        .with_average_p2_probability(0.01)
                        .build();
                    (engine, noise)
                },
                |(engine, noise)| {
                    // Timed: Monte Carlo execution only
                    let results = MonteCarloEngine::run_with_noise_model(
                        Box::new(engine),
                        Box::new(noise),
                        num_shots,
                        1,
                        Some(42),
                    );
                    black_box(results)
                },
                BatchSize::SmallInput,
            );
        },
    );

    // pecos-neo MonteCarloRunner (with noise)
    group.bench_function(BenchmarkId::new("neo_bell_with_noise", num_shots), |b| {
        b.iter(|| {
            let config = MonteCarloConfig::new()
                .with_shots(num_shots)
                .with_workers(1)
                .with_seed(42);

            let results = MonteCarloRunner::run(
                &bell_commands,
                config,
                || {
                    let noise = GeneralNoiseModelBuilder::new()
                        .with_p1(0.001)
                        .with_p2(0.01)
                        .build();
                    (CircuitRunner::new().with_noise(noise), SparseStab::new(2))
                },
                |outcomes| {
                    black_box((
                        outcomes.get_bit(QubitId(0)).unwrap_or(false),
                        outcomes.get_bit(QubitId(1)).unwrap_or(false),
                    ))
                },
            );
            black_box(results)
        });
    });

    // Larger circuit comparison (10 qubits)
    let large_qasm = r#"
        OPENQASM 2.0;
        include "qelib1.inc";
        qreg q[10];
        creg c[10];
        h q[0]; h q[1]; h q[2]; h q[3]; h q[4];
        h q[5]; h q[6]; h q[7]; h q[8]; h q[9];
        cx q[0], q[1]; cx q[1], q[2]; cx q[2], q[3]; cx q[3], q[4];
        cx q[4], q[5]; cx q[5], q[6]; cx q[6], q[7]; cx q[7], q[8]; cx q[8], q[9];
        measure q[0] -> c[0]; measure q[1] -> c[1]; measure q[2] -> c[2];
        measure q[3] -> c[3]; measure q[4] -> c[4]; measure q[5] -> c[5];
        measure q[6] -> c[6]; measure q[7] -> c[7]; measure q[8] -> c[8];
        measure q[9] -> c[9];
    "#;

    let large_commands = {
        let mut builder = CommandBuilder::new();
        builder = builder.pz_all(0..10);
        for i in 0..10 {
            builder = builder.h(i);
        }
        for i in 0..9 {
            builder = builder.cx(i, i + 1);
        }
        builder = builder.mz_all(0..10);
        builder.build()
    };

    // pecos-engines (10 qubits, 1000 shots)
    group.bench_function(BenchmarkId::new("engines_10q_1000shots", 1000), |b| {
        b.iter_batched(
            || QASMEngine::from_str(large_qasm).unwrap(),
            |engine| {
                let results = MonteCarloEngine::run_with_noise_model(
                    Box::new(engine),
                    Box::new(PassThroughNoiseModel::builder().build()),
                    1000,
                    1,
                    Some(42),
                );
                black_box(results)
            },
            BatchSize::SmallInput,
        );
    });

    // pecos-neo (10 qubits, 1000 shots)
    group.bench_function(BenchmarkId::new("neo_10q_1000shots", 1000), |b| {
        b.iter(|| {
            let config = MonteCarloConfig::new()
                .with_shots(1000)
                .with_workers(1)
                .with_seed(42);

            let results = MonteCarloRunner::run(
                &large_commands,
                config,
                || (CircuitRunner::new(), SparseStab::new(10)),
                |outcomes| {
                    let mut sum = 0u32;
                    for i in 0..10 {
                        if outcomes.get_bit(QubitId(i)).unwrap_or(false) {
                            sum |= 1 << i;
                        }
                    }
                    black_box(sum)
                },
            );
            black_box(results)
        });
    });

    group.finish();
}

// ============================================================================
// Flow vs Channel Noise System Comparison
// ============================================================================

/// Compare composite-based noise (`CompositeNoiseModelBuilder`) vs channel-based (`GeneralNoiseModelBuilder`)
fn bench_composite_vs_channel_noise(c: &mut Criterion) {
    use pecos_neo::noise::GeneralNoiseModelBuilder;
    use pecos_neo::noise::composite::CompositeNoiseModelBuilder;

    let mut group = c.benchmark_group("composite_vs_channel");

    // Simple depolarizing noise
    let p1 = 0.001;
    let p2 = 0.01;

    // Bell circuit
    let bell_circuit = CommandBuilder::new()
        .pz_all(0..2)
        .h(0)
        .cx(0, 1)
        .mz_all(0..2)
        .build();

    // Larger circuit for more realistic comparison
    let large_circuit = {
        let mut builder = CommandBuilder::new();
        builder = builder.pz_all(0..20);
        for i in 0..20 {
            builder = builder.h(i);
        }
        for i in 0..19 {
            builder = builder.cx(i, i + 1);
        }
        for i in 0..20 {
            builder = builder.h(i);
        }
        builder = builder.mz_all(0..20);
        builder.build()
    };

    // Compare noise emission overhead
    group.bench_function("channel_noise_emission_1q", |b| {
        let mut noise = GeneralNoiseModelBuilder::new()
            .with_p1(p1)
            .with_p2(p2)
            .build();
        let mut rng = PecosRng::seed_from_u64(42);
        let qubits = vec![QubitId(0)];

        b.iter(|| {
            let event = NoiseEvent::AfterGate {
                gate_type: GateType::H,
                qubits: &qubits,
                angles: &[],
                gate_id: None,
            };
            black_box(noise.emit(event, &mut rng))
        });
    });

    group.bench_function("composite_noise_emission_1q", |b| {
        let mut noise = CompositeNoiseModelBuilder::new()
            .with_p1(p1)
            .with_p2(p2)
            .build();
        let mut rng = PecosRng::seed_from_u64(42);
        let qubits = vec![QubitId(0)];

        b.iter(|| {
            let event = NoiseEvent::AfterGate {
                gate_type: GateType::H,
                qubits: &qubits,
                angles: &[],
                gate_id: None,
            };
            black_box(noise.emit(event, &mut rng))
        });
    });

    group.bench_function("channel_noise_emission_2q", |b| {
        let mut noise = GeneralNoiseModelBuilder::new()
            .with_p1(p1)
            .with_p2(p2)
            .build();
        let mut rng = PecosRng::seed_from_u64(42);
        let qubits = vec![QubitId(0), QubitId(1)];

        b.iter(|| {
            let event = NoiseEvent::AfterGate {
                gate_type: GateType::CX,
                qubits: &qubits,
                angles: &[],
                gate_id: None,
            };
            black_box(noise.emit(event, &mut rng))
        });
    });

    group.bench_function("composite_noise_emission_2q", |b| {
        let mut noise = CompositeNoiseModelBuilder::new()
            .with_p1(p1)
            .with_p2(p2)
            .build();
        let mut rng = PecosRng::seed_from_u64(42);
        let qubits = vec![QubitId(0), QubitId(1)];

        b.iter(|| {
            let event = NoiseEvent::AfterGate {
                gate_type: GateType::CX,
                qubits: &qubits,
                angles: &[],
                gate_id: None,
            };
            black_box(noise.emit(event, &mut rng))
        });
    });

    // Full shot execution comparison
    group.bench_function("channel_bell_shot", |b| {
        let noise = GeneralNoiseModelBuilder::new()
            .with_p1(p1)
            .with_p2(p2)
            .build();
        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new().with_noise(noise);
        b.iter(|| {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &bell_circuit).unwrap();
            black_box(outcomes)
        });
    });

    group.bench_function("composite_bell_shot", |b| {
        let noise = CompositeNoiseModelBuilder::new()
            .with_p1(p1)
            .with_p2(p2)
            .build();
        let mut state = SparseStab::new(2);
        let mut runner = CircuitRunner::<SparseStab>::new().with_noise(noise);
        b.iter(|| {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &bell_circuit).unwrap();
            black_box(outcomes)
        });
    });

    // Larger circuit comparison
    group.bench_function("channel_20q_shot", |b| {
        let noise = GeneralNoiseModelBuilder::new()
            .with_p1(p1)
            .with_p2(p2)
            .build();
        let mut state = SparseStab::new(20);
        let mut runner = CircuitRunner::<SparseStab>::new().with_noise(noise);
        b.iter(|| {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &large_circuit).unwrap();
            black_box(outcomes)
        });
    });

    group.bench_function("composite_20q_shot", |b| {
        let noise = CompositeNoiseModelBuilder::new()
            .with_p1(p1)
            .with_p2(p2)
            .build();
        let mut state = SparseStab::new(20);
        let mut runner = CircuitRunner::<SparseStab>::new().with_noise(noise);
        b.iter(|| {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &large_circuit).unwrap();
            black_box(outcomes)
        });
    });

    // With leakage (more complex noise model)
    group.bench_function("channel_20q_with_leakage", |b| {
        let noise = GeneralNoiseModelBuilder::new()
            .with_p1(p1)
            .with_p1_emission_ratio(0.1)
            .with_p1_seepage(0.05)
            .with_p2(p2)
            .with_p2_emission_ratio(0.1)
            .with_p2_seepage(0.05)
            .build();
        let mut state = SparseStab::new(20);
        let mut runner = CircuitRunner::<SparseStab>::new().with_noise(noise);
        b.iter(|| {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &large_circuit).unwrap();
            black_box(outcomes)
        });
    });

    group.bench_function("composite_20q_with_leakage", |b| {
        let noise = CompositeNoiseModelBuilder::new()
            .with_p1(p1)
            .with_p1_emission_ratio(0.1)
            .with_p1_seepage(0.05)
            .with_p2(p2)
            .with_p2_emission_ratio(0.1)
            .with_p2_seepage(0.05)
            .build();
        let mut state = SparseStab::new(20);
        let mut runner = CircuitRunner::<SparseStab>::new().with_noise(noise);
        b.iter(|| {
            state.reset();
            let outcomes = runner.apply_circuit(&mut state, &large_circuit).unwrap();
            black_box(outcomes)
        });
    });

    group.finish();
}

// ============================================================================
// Batch Filtering Benchmarks
// ============================================================================

/// Benchmark batch probability filtering at scale
fn bench_batch_filtering(c: &mut Criterion) {
    use pecos_neo::noise::composite::batch::{filter_by_probability, filter_range_by_probability};
    use pecos_neo::noise::composite::batch_composite::BatchState;

    let mut group = c.benchmark_group("batch_filtering");

    // Test at various scales
    for num_qubits in [1_000, 10_000, 100_000, 1_000_000] {
        let qubits: Vec<_> = (0..num_qubits).map(QubitId).collect();

        // Low probability (p=0.001) - geometric should shine here
        group.bench_function(BenchmarkId::new("filter_p0.001", num_qubits), |b| {
            let mut rng = PecosRng::seed_from_u64(42);
            b.iter(|| black_box(filter_by_probability(&qubits, 0.001, &mut rng)));
        });

        // Medium probability (p=0.01)
        group.bench_function(BenchmarkId::new("filter_p0.01", num_qubits), |b| {
            let mut rng = PecosRng::seed_from_u64(42);
            b.iter(|| black_box(filter_by_probability(&qubits, 0.01, &mut rng)));
        });

        // Higher probability (p=0.1) - linear should be fine here
        group.bench_function(BenchmarkId::new("filter_p0.1", num_qubits), |b| {
            let mut rng = PecosRng::seed_from_u64(42);
            b.iter(|| black_box(filter_by_probability(&qubits, 0.1, &mut rng)));
        });
    }

    // Compare filter_range (no pre-allocated vec) at 1M
    group.bench_function("filter_range_1M_p0.001", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| black_box(filter_range_by_probability(0, 1_000_000, 0.001, &mut rng)));
    });

    // Compare with RngProbabilityExt optimized version
    use pecos_neo::noise::composite::batch::{
        ProbabilityThreshold, filter_by_probability_fast, sample_noise_1q_batch,
    };

    group.bench_function("rng_ext_filter_1M_p0.001", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        let threshold = ProbabilityThreshold::new(0.001, &rng);
        b.iter(|| black_box(filter_by_probability_fast(1_000_000, &threshold, &mut rng)));
    });

    group.bench_function("rng_ext_noise_1q_1M_p0.001", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        let threshold = ProbabilityThreshold::new(0.001, &rng);
        b.iter(|| black_box(sample_noise_1q_batch(1_000_000, &threshold, &mut rng)));
    });

    // ========================================================================
    // Low probability benchmarks (realistic fault rates: 1e-4 to 1e-5)
    // ========================================================================

    // p = 1e-4 (0.0001) - typical gate error rate
    for num_qubits in [100_000, 1_000_000] {
        group.bench_function(BenchmarkId::new("geometric_p1e-4", num_qubits), |b| {
            let qubits: Vec<_> = (0..num_qubits).map(QubitId).collect();
            let mut rng = PecosRng::seed_from_u64(42);
            b.iter(|| black_box(filter_by_probability(&qubits, 1e-4, &mut rng)));
        });
    }

    // p = 1e-5 (0.00001) - very low error rate
    for num_qubits in [100_000, 1_000_000] {
        group.bench_function(BenchmarkId::new("geometric_p1e-5", num_qubits), |b| {
            let qubits: Vec<_> = (0..num_qubits).map(QubitId).collect();
            let mut rng = PecosRng::seed_from_u64(42);
            b.iter(|| black_box(filter_by_probability(&qubits, 1e-5, &mut rng)));
        });
    }

    // Compare: filter_range (avoids vec allocation) at low probabilities
    group.bench_function("filter_range_1M_p1e-4", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| black_box(filter_range_by_probability(0, 1_000_000, 1e-4, &mut rng)));
    });

    group.bench_function("filter_range_1M_p1e-5", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| black_box(filter_range_by_probability(0, 1_000_000, 1e-5, &mut rng)));
    });

    // Benchmark BatchState operations at scale
    group.bench_function("batch_state_mark_leaked_1000", |b| {
        let mut state = BatchState::new(1_000_000);
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            // Mark 1000 random qubits as leaked
            for _ in 0..1000 {
                let idx = (rng.random::<f64>() * 1_000_000.0) as usize;
                state.mark_leaked(QubitId(idx));
            }
            black_box(&state);
        });
    });

    group.bench_function("batch_state_is_leaked_1000", |b| {
        let mut state = BatchState::new(1_000_000);
        // Pre-mark some qubits
        for i in (0..1_000_000).step_by(1000) {
            state.mark_leaked(QubitId(i));
        }
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let mut count = 0;
            for _ in 0..1000 {
                let idx = (rng.random::<f64>() * 1_000_000.0) as usize;
                if state.is_leaked(QubitId(idx)) {
                    count += 1;
                }
            }
            black_box(count);
        });
    });

    group.finish();
}

// ============================================================================
// Batch Processor Comparison
// ============================================================================

/// Compare `FastBatchProcessor` vs `BatchCompositeChannel` vs `CompositeChannel` at scale
fn bench_batch_processor(c: &mut Criterion) {
    use pecos_neo::noise::composite::batch_composite::{BatchState, fast_depolarizing};
    use pecos_neo::noise::composite::prelude::*;
    use pecos_neo::noise::{NoiseChannel, NoiseContext, NoiseEvent};

    let mut group = c.benchmark_group("batch_processor");

    // Test at various scales with low probability
    for num_qubits in [10_000, 100_000, 1_000_000] {
        let qubits: Vec<_> = (0..num_qubits).map(QubitId).collect();
        let angles = [];

        // FastBatchProcessor (geometric + lazy filter) - OPTIMAL
        group.bench_function(BenchmarkId::new("fast_processor_p1e-4", num_qubits), |b| {
            let mut state = BatchState::new(num_qubits);
            state.mark_all_active();
            // Mark 1% as leaked for realistic scenario
            for i in (0..num_qubits).step_by(100) {
                state.mark_leaked(QubitId(i));
            }

            let mut ctx = NoiseContext::new();
            let mut rng = PecosRng::seed_from_u64(42);
            let processor = fast_depolarizing(1e-4, pauli());

            b.iter(|| black_box(processor.process_all(&state, &mut ctx, &mut rng)));
        });

        // BatchCompositeChannel (geometric sampling via NoiseChannel interface)
        let batch_channel = CompositeChannelBuilder::batch_single_qubit("batch", 1e-4, pauli());
        group.bench_function(BenchmarkId::new("batch_channel_p1e-4", num_qubits), |b| {
            let mut ctx = NoiseContext::new();
            let mut rng = PecosRng::seed_from_u64(42);
            let event = NoiseEvent::AfterGate {
                gate_type: GateType::H,
                qubits: &qubits,
                angles: &angles,
                gate_id: None,
            };
            b.iter(|| black_box(batch_channel.apply(&event, &mut ctx, &mut rng)));
        });

        // CompositeChannel with probability (geometric sampling - new consolidated API)
        let composite_with_prob = CompositeChannel::new("composite_fast", pauli())
            .with_probability(1e-4)
            .with_filter(CompositeEventFilter::SingleQubitGate);
        group.bench_function(
            BenchmarkId::new("composite_with_prob_p1e-4", num_qubits),
            |b| {
                let mut ctx = NoiseContext::new();
                let mut rng = PecosRng::seed_from_u64(42);
                let event = NoiseEvent::AfterGate {
                    gate_type: GateType::H,
                    qubits: &qubits,
                    angles: &angles,
                    gate_id: None,
                };
                b.iter(|| black_box(composite_with_prob.apply(&event, &mut ctx, &mut rng)));
            },
        );

        // CompositeChannel (linear, no probability) - skip for 1M (too slow)
        if num_qubits <= 100_000 {
            let composite_channel =
                CompositeChannelBuilder::single_qubit("composite", prob(1e-4, pauli()));
            group.bench_function(
                BenchmarkId::new("composite_linear_p1e-4", num_qubits),
                |b| {
                    let mut ctx = NoiseContext::new();
                    let mut rng = PecosRng::seed_from_u64(42);
                    let event = NoiseEvent::AfterGate {
                        gate_type: GateType::H,
                        qubits: &qubits,
                        angles: &angles,
                        gate_id: None,
                    };
                    b.iter(|| black_box(composite_channel.apply(&event, &mut ctx, &mut rng)));
                },
            );
        }
    }

    // Test FastBatchProcessor with varying leaked percentages at 100K qubits
    let num_qubits = 100_000;
    for leaked_pct in [0, 1, 10, 50] {
        group.bench_function(
            BenchmarkId::new(format!("fast_{leaked_pct}pct_leaked"), num_qubits),
            |b| {
                let mut state = BatchState::new(num_qubits);
                state.mark_all_active();
                // Mark leaked_pct% as leaked
                let step = if leaked_pct > 0 {
                    100 / leaked_pct
                } else {
                    num_qubits + 1
                };
                for i in (0..num_qubits).step_by(step) {
                    state.mark_leaked(QubitId(i));
                }

                let mut ctx = NoiseContext::new();
                let mut rng = PecosRng::seed_from_u64(42);
                let processor = fast_depolarizing(1e-4, pauli());

                b.iter(|| black_box(processor.process_all(&state, &mut ctx, &mut rng)));
            },
        );
    }

    group.finish();
}

/// Compare trait object dispatch vs compiled enum dispatch for composite primitives
fn bench_dispatch_comparison(c: &mut Criterion) {
    use pecos_neo::noise::NoiseContext;
    use pecos_neo::noise::composite::prelude::*;
    use pecos_neo::noise::composite::{CompiledAction, CompiledCondition, CompiledPrimitive};

    let mut group = c.benchmark_group("dispatch_comparison");

    // Create equivalent noise trees: trait-object and compiled versions

    // Simple probability + pauli
    let trait_simple: Box<dyn Primitive> = Box::new(prob(0.01, pauli()));
    let compiled_simple = CompiledPrimitive::prob(
        0.01,
        CompiledPrimitive::action(CompiledAction::Pauli(PauliWeights::uniform())),
    );

    // More complex tree with conditions
    let trait_complex = seq![skip_if_leaked(), prob(0.01, when_leaked(seep(), pauli())),];
    let compiled_complex = CompiledPrimitive::seq(vec![
        CompiledPrimitive::skip_if(CompiledCondition::Leaked),
        CompiledPrimitive::prob(
            0.01,
            CompiledPrimitive::when(
                CompiledCondition::Leaked,
                CompiledPrimitive::action(CompiledAction::Seep(PauliWeights::uniform())),
                CompiledPrimitive::action(CompiledAction::Pauli(PauliWeights::uniform())),
            ),
        ),
    ]);

    // Benchmark simple tree
    group.bench_function("trait_simple_pauli", |b| {
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| black_box(trait_simple.apply(QubitId(0), &mut ctx, &mut rng)));
    });

    group.bench_function("compiled_simple_pauli", |b| {
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| black_box(compiled_simple.apply(QubitId(0), &mut ctx, &mut rng)));
    });

    // Benchmark complex tree
    group.bench_function("trait_complex_tree", |b| {
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| black_box(trait_complex.apply(QubitId(0), &mut ctx, &mut rng)));
    });

    group.bench_function("compiled_complex_tree", |b| {
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| black_box(compiled_complex.apply(QubitId(0), &mut ctx, &mut rng)));
    });

    // Benchmark many iterations to amortize setup
    group.bench_function("trait_1000_iterations", |b| {
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            for _ in 0..1000 {
                black_box(trait_complex.apply(QubitId(0), &mut ctx, &mut rng));
            }
        });
    });

    group.bench_function("compiled_1000_iterations", |b| {
        let mut ctx = NoiseContext::new();
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            for _ in 0..1000 {
                black_box(compiled_complex.apply(QubitId(0), &mut ctx, &mut rng));
            }
        });
    });

    group.finish();
}

criterion_group!(
    benches,
    bench_noise_emission,
    bench_shot_execution,
    bench_multi_shot,
    bench_entity_operations,
    bench_redistribution,
    bench_rng_operations,
    bench_simulator_ops,
    bench_memory_usage,
    bench_monte_carlo_comparison,
    bench_composite_vs_channel_noise,
    bench_batch_filtering,
    bench_batch_processor,
    bench_dispatch_comparison,
);

criterion_main!(benches);
