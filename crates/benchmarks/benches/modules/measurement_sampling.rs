// Copyright 2025 The PECOS Developers
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

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos::prelude::*;
use pecos::simulators::measurement_sampler::{
    MeasurementKind, MeasurementSampler, SequentialMeasurementSampler,
};
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_bell_state(c);
    bench_ghz_state(c);
    bench_many_random_measurements(c);
    bench_scaling_shots(c);
    bench_scaling_measurements(c);
    bench_realistic_qec(c);
    bench_multi_round_qec(c);
}

/// Benchmark sampling from a Bell state (2 qubits, 2 measurements, 1 random + 1 computed)
fn bench_bell_state<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Measurement Sampling - Bell State");

    // Create the Bell state measurement history once
    let mut sim = SymbolicSparseStab::new(2);
    sim.h(0).cx(0, 1);
    sim.mz(0);
    sim.mz(1);
    let history = sim.measurement_history().clone();

    let sequential_sampler = SequentialMeasurementSampler::new(&history);
    let sampler = MeasurementSampler::new(&history);

    for shots in [100, 1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(shots as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential_sampler", shots),
            &shots,
            |b, &shots| b.iter(|| black_box(sequential_sampler.sample(shots))),
        );

        group.bench_with_input(BenchmarkId::new("sampler", shots), &shots, |b, &shots| {
            b.iter(|| black_box(sampler.sample(shots)));
        });
    }

    group.finish();
}

/// Benchmark sampling from a GHZ state (3 qubits, 3 measurements)
fn bench_ghz_state<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Measurement Sampling - GHZ State");

    let mut sim = SymbolicSparseStab::new(3);
    sim.h(0).cx(0, 1).cx(1, 2);
    sim.mz(0);
    sim.mz(1);
    sim.mz(2);
    let history = sim.measurement_history().clone();

    let sequential_sampler = SequentialMeasurementSampler::new(&history);
    let sampler = MeasurementSampler::new(&history);

    for shots in [100, 1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(shots as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential_sampler", shots),
            &shots,
            |b, &shots| b.iter(|| black_box(sequential_sampler.sample(shots))),
        );

        group.bench_with_input(BenchmarkId::new("sampler", shots), &shots, |b, &shots| {
            b.iter(|| black_box(sampler.sample(shots)));
        });
    }

    group.finish();
}

/// Benchmark sampling many independent random measurements
fn bench_many_random_measurements<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Measurement Sampling - Many Random");

    // Create many independent random measurements (all |+> states)
    let mut sim = SymbolicSparseStab::new(20);
    for i in 0..20 {
        sim.h(i);
    }
    for i in 0..20 {
        sim.mz(i);
    }
    let history = sim.measurement_history().clone();

    let sequential_sampler = SequentialMeasurementSampler::new(&history);
    let sampler = MeasurementSampler::new(&history);

    for shots in [100, 1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(shots as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential_sampler", shots),
            &shots,
            |b, &shots| b.iter(|| black_box(sequential_sampler.sample(shots))),
        );

        group.bench_with_input(BenchmarkId::new("sampler", shots), &shots, |b, &shots| {
            b.iter(|| black_box(sampler.sample(shots)));
        });
    }

    group.finish();
}

/// Benchmark how performance scales with number of shots
fn bench_scaling_shots<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Measurement Sampling - Scaling Shots");

    // A medium complexity circuit: 10 qubits, entangled
    let mut sim = SymbolicSparseStab::new(10);
    sim.h(0);
    for i in 0..9 {
        sim.cx(i, i + 1);
    }
    for i in 0..10 {
        sim.mz(i);
    }
    let history = sim.measurement_history().clone();

    let sequential_sampler = SequentialMeasurementSampler::new(&history);
    let sampler = MeasurementSampler::new(&history);

    for shots in [1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(shots as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential_sampler", shots),
            &shots,
            |b, &shots| b.iter(|| black_box(sequential_sampler.sample(shots))),
        );

        group.bench_with_input(BenchmarkId::new("sampler", shots), &shots, |b, &shots| {
            b.iter(|| black_box(sampler.sample(shots)));
        });
    }

    group.finish();
}

/// Benchmark how performance scales with number of measurements
fn bench_scaling_measurements<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Measurement Sampling - Scaling Measurements");
    let shots = 100_000;

    for num_measurements in [10, 50, 100, 200, 500, 1000] {
        // Create a GHZ-like state with all qubits entangled
        let mut sim = SymbolicSparseStab::new(num_measurements);
        sim.h(0);
        for i in 0..(num_measurements - 1) {
            sim.cx(i, i + 1);
        }
        for i in 0..num_measurements {
            sim.mz(i);
        }
        let history = sim.measurement_history().clone();

        let sequential_sampler = SequentialMeasurementSampler::new(&history);
        let sampler = MeasurementSampler::new(&history);

        group.throughput(Throughput::Elements(num_measurements as u64 * shots as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential_sampler", num_measurements),
            &num_measurements,
            |b, _| b.iter(|| black_box(sequential_sampler.sample(shots))),
        );

        group.bench_with_input(
            BenchmarkId::new("sampler", num_measurements),
            &num_measurements,
            |b, _| b.iter(|| black_box(sampler.sample(shots))),
        );
    }

    group.finish();
}

/// Benchmark realistic QEC-like measurement patterns
///
/// Realistic QEC circuits have:
/// - ~10% truly random measurements (non-deterministic syndrome measurements)
/// - ~5% fixed values (initialized ancillas)
/// - Mostly computed measurements with 1-4 dependencies
fn bench_realistic_qec<M: Measurement>(c: &mut Criterion<M>) {
    use pecos::random;

    let mut group = c.benchmark_group("Measurement Sampling - Realistic QEC");

    // Test different circuit sizes
    for num_measurements in [100, 500, 1000, 5000] {
        // Generate realistic QEC-like measurement pattern using seeded RNG for reproducibility
        random::seed(42);
        let measurements = generate_qec_like_measurements(num_measurements);

        let sequential_sampler =
            SequentialMeasurementSampler::from_measurements(measurements.clone());
        let sampler = MeasurementSampler::from_measurements(measurements);

        let shots = 100_000;
        group.throughput(Throughput::Elements(num_measurements as u64 * shots as u64));

        group.bench_with_input(
            BenchmarkId::new("sequential_sampler", num_measurements),
            &num_measurements,
            |b, _| b.iter(|| black_box(sequential_sampler.sample(shots))),
        );

        group.bench_with_input(
            BenchmarkId::new("sampler", num_measurements),
            &num_measurements,
            |b, _| b.iter(|| black_box(sampler.sample(shots))),
        );
    }

    group.finish();
}

/// Benchmark multi-round QEC with sparse cross-round dependencies.
///
/// This tests the realistic scenario where:
/// - Multiple syndrome extraction rounds are performed
/// - Dependencies can span from early rounds to late rounds
/// - Creates sparse `BitSet` patterns like {0, 100, 500, 900} for measurement 950
fn bench_multi_round_qec<M: Measurement>(c: &mut Criterion<M>) {
    use pecos::random;

    let mut group = c.benchmark_group("Measurement Sampling - Multi-Round QEC");

    // Test: 1000 measurements over 10 rounds (100 per round)
    // Dependencies span across rounds, creating sparse patterns
    let num_measurements = 1000;
    let num_rounds = 10;

    random::seed(42);
    let measurements = generate_qec_measurements_with_rounds(num_measurements, num_rounds);

    let sequential_sampler = SequentialMeasurementSampler::from_measurements(measurements.clone());
    let sampler = MeasurementSampler::from_measurements(measurements);

    let shots = 100_000;
    group.throughput(Throughput::Elements(num_measurements as u64 * shots as u64));

    group.bench_function("sequential_sampler/10_rounds", |b| {
        b.iter(|| black_box(sequential_sampler.sample(shots)));
    });

    group.bench_function("sampler/10_rounds", |b| {
        b.iter(|| black_box(sampler.sample(shots)));
    });

    // Also test with more rounds (sparser dependencies)
    let num_rounds_50 = 50;
    random::seed(42);
    let measurements_50 = generate_qec_measurements_with_rounds(num_measurements, num_rounds_50);

    let sequential_sampler_50 =
        SequentialMeasurementSampler::from_measurements(measurements_50.clone());
    let sampler_50 = MeasurementSampler::from_measurements(measurements_50);

    group.bench_function("sequential_sampler/50_rounds", |b| {
        b.iter(|| black_box(sequential_sampler_50.sample(shots)));
    });

    group.bench_function("sampler/50_rounds", |b| {
        b.iter(|| black_box(sampler_50.sample(shots)));
    });

    group.finish();
}

/// Generate QEC-like measurement patterns manually.
///
/// Pattern: 10% random, 5% fixed, rest computed with 1-3 deps
fn generate_qec_like_measurements(num_measurements: usize) -> Vec<MeasurementKind> {
    generate_qec_measurements_with_rounds(num_measurements, 1)
}

/// Generate multi-round QEC measurement patterns.
///
/// Simulates `num_rounds` of syndrome extraction where:
/// - Each round has `measurements_per_round` measurements
/// - ~10% of each round's measurements are non-deterministic (random)
/// - Later rounds can depend on measurements from ANY earlier round
///   (simulating stabilizer measurements that correlate across rounds)
///
/// This creates realistic sparse dependency patterns where measurement 950
/// might depend on measurements {0, 100, 200, ...} spanning many rounds.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)]
fn generate_qec_measurements_with_rounds(
    num_measurements: usize,
    num_rounds: usize,
) -> Vec<MeasurementKind> {
    use pecos::random;

    let measurements_per_round = num_measurements / num_rounds.max(1);
    let mut measurements = Vec::with_capacity(num_measurements);

    for i in 0..num_measurements {
        let current_round = i / measurements_per_round.max(1);
        let r: f64 = random::random(1)[0];

        let kind = if r < 0.10 {
            // 10% random (non-deterministic syndrome measurements)
            MeasurementKind::Random
        } else if i == 0 || r < 0.15 {
            // 5% fixed
            let flip: bool = random::random(1)[0] > 0.5;
            MeasurementKind::Fixed(flip)
        } else {
            // Computed from earlier measurements
            // Key insight: dependencies can span multiple rounds
            let max_deps = 3.min(i);
            // Generate random number of dependencies (1 to max_deps inclusive)
            let rand_val: f64 = random::random(1)[0];
            let num_deps = 1 + (rand_val * max_deps as f64) as usize % max_deps;

            let mut deps: Vec<usize> = Vec::with_capacity(num_deps);

            // For multi-round scenarios, prefer dependencies from:
            // 1. Same position in previous rounds (simulating repeated stabilizer measurement)
            // 2. Random earlier measurements
            for d in 0..num_deps {
                let dep = if current_round > 0 && d == 0 && measurements_per_round > 0 {
                    // First dep: same stabilizer from a previous round
                    let rand_val: f64 = random::random(1)[0];
                    let prev_round = (rand_val * current_round as f64) as usize;
                    let pos_in_round = i % measurements_per_round;
                    (prev_round * measurements_per_round + pos_in_round).min(i.saturating_sub(1))
                } else {
                    // Other deps: random earlier measurement
                    let rand_val: f64 = random::random(1)[0];
                    (rand_val * i as f64) as usize
                };

                if !deps.contains(&dep) {
                    deps.push(dep);
                }
            }
            deps.sort_unstable();

            let flip: bool = random::random(1)[0] > 0.5;
            MeasurementKind::Computed { deps, flip }
        };
        measurements.push(kind);
    }

    measurements
}
