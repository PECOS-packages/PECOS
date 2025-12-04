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
use pecos::qsim::measurement_sampler::{ColumnarSampler, MeasurementKind, ShotSampler};
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_bell_state(c);
    bench_ghz_state(c);
    bench_many_random_measurements(c);
    bench_scaling_shots(c);
    bench_scaling_measurements(c);
    bench_realistic_qec(c);
}

/// Benchmark sampling from a Bell state (2 qubits, 2 measurements, 1 random + 1 computed)
fn bench_bell_state<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Measurement Sampling - Bell State");

    // Create the Bell state measurement history once
    let mut sim = StdSymbolicSparseStab::new(2);
    sim.h(0).cx(0, 1);
    sim.mz(0);
    sim.mz(1);
    let history = sim.measurement_history().clone();

    let shot_sampler = ShotSampler::new(&history);
    let columnar_sampler = ColumnarSampler::new(&history);

    for shots in [100, 1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(shots as u64));

        group.bench_with_input(BenchmarkId::new("shot_sampler", shots), &shots, |b, &shots| {
            b.iter(|| black_box(shot_sampler.sample_raw_with_thread_rng(shots)))
        });

        group.bench_with_input(
            BenchmarkId::new("columnar_sampler", shots),
            &shots,
            |b, &shots| b.iter(|| black_box(columnar_sampler.sample_raw_with_thread_rng(shots))),
        );
    }

    group.finish();
}

/// Benchmark sampling from a GHZ state (3 qubits, 3 measurements)
fn bench_ghz_state<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Measurement Sampling - GHZ State");

    let mut sim = StdSymbolicSparseStab::new(3);
    sim.h(0).cx(0, 1).cx(1, 2);
    sim.mz(0);
    sim.mz(1);
    sim.mz(2);
    let history = sim.measurement_history().clone();

    let shot_sampler = ShotSampler::new(&history);
    let columnar_sampler = ColumnarSampler::new(&history);

    for shots in [100, 1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(shots as u64));

        group.bench_with_input(BenchmarkId::new("shot_sampler", shots), &shots, |b, &shots| {
            b.iter(|| black_box(shot_sampler.sample_raw_with_thread_rng(shots)))
        });

        group.bench_with_input(
            BenchmarkId::new("columnar_sampler", shots),
            &shots,
            |b, &shots| b.iter(|| black_box(columnar_sampler.sample_raw_with_thread_rng(shots))),
        );
    }

    group.finish();
}

/// Benchmark sampling many independent random measurements
fn bench_many_random_measurements<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Measurement Sampling - Many Random");

    // Create many independent random measurements (all |+> states)
    let mut sim = StdSymbolicSparseStab::new(20);
    for i in 0..20 {
        sim.h(i);
    }
    for i in 0..20 {
        sim.mz(i);
    }
    let history = sim.measurement_history().clone();

    let shot_sampler = ShotSampler::new(&history);
    let columnar_sampler = ColumnarSampler::new(&history);

    for shots in [100, 1_000, 10_000, 100_000] {
        group.throughput(Throughput::Elements(shots as u64));

        group.bench_with_input(BenchmarkId::new("shot_sampler", shots), &shots, |b, &shots| {
            b.iter(|| black_box(shot_sampler.sample_raw_with_thread_rng(shots)))
        });

        group.bench_with_input(
            BenchmarkId::new("columnar_sampler", shots),
            &shots,
            |b, &shots| b.iter(|| black_box(columnar_sampler.sample_raw_with_thread_rng(shots))),
        );
    }

    group.finish();
}

/// Benchmark how performance scales with number of shots
fn bench_scaling_shots<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Measurement Sampling - Scaling Shots");

    // A medium complexity circuit: 10 qubits, entangled
    let mut sim = StdSymbolicSparseStab::new(10);
    sim.h(0);
    for i in 0..9 {
        sim.cx(i, i + 1);
    }
    for i in 0..10 {
        sim.mz(i);
    }
    let history = sim.measurement_history().clone();

    let shot_sampler = ShotSampler::new(&history);
    let columnar_sampler = ColumnarSampler::new(&history);

    for shots in [1_000, 10_000, 100_000, 1_000_000] {
        group.throughput(Throughput::Elements(shots as u64));

        group.bench_with_input(BenchmarkId::new("shot_sampler", shots), &shots, |b, &shots| {
            b.iter(|| black_box(shot_sampler.sample_raw_with_thread_rng(shots)))
        });

        group.bench_with_input(
            BenchmarkId::new("columnar_sampler", shots),
            &shots,
            |b, &shots| b.iter(|| black_box(columnar_sampler.sample_raw_with_thread_rng(shots))),
        );
    }

    group.finish();
}

/// Benchmark how performance scales with number of measurements
fn bench_scaling_measurements<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Measurement Sampling - Scaling Measurements");
    let shots = 100_000;

    for num_measurements in [10, 50, 100, 200, 500, 1000] {
        // Create a GHZ-like state with all qubits entangled
        let mut sim = StdSymbolicSparseStab::new(num_measurements);
        sim.h(0);
        for i in 0..(num_measurements - 1) {
            sim.cx(i, i + 1);
        }
        for i in 0..num_measurements {
            sim.mz(i);
        }
        let history = sim.measurement_history().clone();

        let shot_sampler = ShotSampler::new(&history);
        let columnar_sampler = ColumnarSampler::new(&history);

        group.throughput(Throughput::Elements(num_measurements as u64 * shots as u64));

        group.bench_with_input(
            BenchmarkId::new("shot_sampler", num_measurements),
            &num_measurements,
            |b, _| b.iter(|| black_box(shot_sampler.sample_raw_with_thread_rng(shots))),
        );

        group.bench_with_input(
            BenchmarkId::new("columnar_sampler", num_measurements),
            &num_measurements,
            |b, _| b.iter(|| black_box(columnar_sampler.sample_raw_with_thread_rng(shots))),
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

        let shot_sampler = ShotSampler::from_measurements(measurements.clone());
        let columnar_sampler = ColumnarSampler::from_measurements(measurements);

        let shots = 100_000;
        group.throughput(Throughput::Elements(num_measurements as u64 * shots as u64));

        group.bench_with_input(
            BenchmarkId::new("shot_sampler", num_measurements),
            &num_measurements,
            |b, _| b.iter(|| black_box(shot_sampler.sample_raw_with_thread_rng(shots))),
        );

        group.bench_with_input(
            BenchmarkId::new("columnar_sampler", num_measurements),
            &num_measurements,
            |b, _| b.iter(|| black_box(columnar_sampler.sample_raw_with_thread_rng(shots))),
        );

        group.bench_with_input(
            BenchmarkId::new("columnar_fast", num_measurements),
            &num_measurements,
            |b, _| b.iter(|| black_box(columnar_sampler.sample_raw_fast(shots))),
        );
    }

    group.finish();
}

/// Generate QEC-like measurement patterns manually.
///
/// Pattern: 10% random, 5% fixed, rest computed with 1-3 deps
fn generate_qec_like_measurements(num_measurements: usize) -> Vec<MeasurementKind> {
    use pecos::random;

    let mut measurements = Vec::with_capacity(num_measurements);

    for i in 0..num_measurements {
        let r: f64 = random::random(1)[0];
        let kind = if r < 0.10 {
            // 10% random
            MeasurementKind::Random
        } else if i == 0 || r < 0.15 {
            // 5% fixed (of total) - 0.10 to 0.15
            let flip: bool = random::random(1)[0] > 0.5;
            MeasurementKind::Fixed(flip)
        } else {
            // Computed from earlier measurements (1-3 deps)
            let max_deps = 3.min(i);
            let num_deps = 1 + (random::randint(0, Some(max_deps as i64), 1)[0] as usize % max_deps);

            // Pick random earlier indices
            let mut deps: Vec<usize> = Vec::with_capacity(num_deps);
            for _ in 0..num_deps {
                let dep = random::randint(0, Some(i as i64), 1)[0] as usize;
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
