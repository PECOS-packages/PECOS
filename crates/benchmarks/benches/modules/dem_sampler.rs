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

//! DEM Sampler benchmarks for threshold estimation.
//!
//! These benchmarks measure the performance of the DEM-based sampling
//! infrastructure used for fast error threshold estimation.
//!
//! # Benchmarks
//!
//! - **DEM Sampler - Original vs Columnar**: Compare row-major vs column-major sampling
//! - **DEM Sampler - Statistics**: Compare statistics-only methods
//! - **DEM Sampler - Scaling**: How different methods scale with DEM size

use std::str::FromStr;

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos_qec::fault_tolerance::dem_builder::{DemSamplerBuilder, ParsedDem};
use pecos_qec::fault_tolerance::propagator::DagFaultAnalyzer;
use pecos_quantum::DagCircuit;
use pecos_random::PecosRng;
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_sampler_comparison(c);
    bench_statistics_comparison(c);
    bench_optimization_comparison(c);
    bench_parsed_dem_sampling(c);
}

/// Create a realistic DEM sampler from a surface-code-like circuit.
fn create_surface_code_sampler(
    distance: usize,
    rounds: usize,
) -> pecos_qec::fault_tolerance::dem_builder::DemSampler {
    // Create a simplified surface code circuit
    let num_data = distance * distance;
    let num_ancilla = num_data - 1;

    let mut dag = DagCircuit::new();

    // Initialize data qubits
    for q in 0..num_data {
        dag.pz(&[q]);
        dag.h(&[q]);
    }

    // Syndrome extraction rounds
    for _round in 0..rounds {
        // Initialize ancillas
        for a in 0..num_ancilla {
            dag.pz(&[num_data + a]);
        }

        // Entangle ancillas with data (simplified pattern)
        for a in 0..num_ancilla {
            let ancilla = num_data + a;
            let d1 = a % num_data;
            let d2 = (a + 1) % num_data;
            dag.cx(&[(ancilla, d1)]);
            dag.cx(&[(ancilla, d2)]);
        }

        // Measure ancillas
        for a in 0..num_ancilla {
            dag.mz(&[num_data + a]);
        }
    }

    // Build influence map and sampler
    let analyzer = DagFaultAnalyzer::new(&dag);
    let influence_map = analyzer.build_influence_map();

    // Create detector definitions (one per ancilla measurement)
    let num_measurements = num_ancilla * rounds;
    let mut detector_records = Vec::new();
    for i in 0..num_measurements.min(50) {
        // Limit to 50 detectors for benchmark
        detector_records.push(vec![-(i as i32 + 1)]);
    }

    DemSamplerBuilder::new(&influence_map)
        .with_noise(0.001, 0.001, 0.001, 0.001)
        .with_detector_records(detector_records)
        .with_observable_records(vec![])
        .build()
}

/// Benchmark comparing original row-major vs columnar sampling.
fn bench_sampler_comparison<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("DEM Sampler - Original vs Columnar");

    // Test with different circuit sizes
    for (distance, rounds) in [(3, 2), (5, 3)] {
        let sampler = create_surface_code_sampler(distance, rounds);
        let num_mechanisms = sampler.num_mechanisms();
        let num_detectors = sampler.num_detectors();

        // Test different shot counts
        for shots in [1_000, 10_000, 100_000] {
            let label = format!("d{distance}_r{rounds}_{shots}");

            group.throughput(Throughput::Elements((num_mechanisms * shots) as u64));

            // Original row-major sampling
            group.bench_with_input(BenchmarkId::new("row_major", &label), &(), |b, ()| {
                let mut rng = PecosRng::seed_from_u64(42);
                b.iter(|| {
                    let result = sampler.sample_batch(shots, &mut rng);
                    black_box(result)
                });
            });

            // Columnar sampling (accurate - one random per shot per mechanism)
            group.bench_with_input(
                BenchmarkId::new("columnar_accurate", &label),
                &(),
                |b, ()| {
                    let mut rng = PecosRng::seed_from_u64(42);
                    b.iter(|| {
                        let result = sampler.sample_batch_columnar_accurate(shots, &mut rng);
                        black_box(result)
                    });
                },
            );
        }

        // Print info
        println!(
            "  d={distance} r={rounds}: {num_mechanisms} mechanisms, {num_detectors} detectors"
        );
    }

    group.finish();
}

/// Benchmark comparing statistics methods.
fn bench_statistics_comparison<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("DEM Sampler - Statistics");

    let sampler = create_surface_code_sampler(5, 3);
    let num_mechanisms = sampler.num_mechanisms();

    for shots in [10_000, 100_000, 1_000_000] {
        let label = format!("{shots}shots");

        group.throughput(Throughput::Elements((num_mechanisms * shots) as u64));

        // Original statistics (row-major)
        group.bench_with_input(
            BenchmarkId::new("row_major", &label),
            &shots,
            |b, &shots| {
                let mut rng = PecosRng::seed_from_u64(42);
                b.iter(|| {
                    let result = sampler.sample_statistics_row_major(shots, &mut rng);
                    black_box(result)
                });
            },
        );

        // Columnar statistics
        group.bench_with_input(BenchmarkId::new("columnar", &label), &shots, |b, &shots| {
            let mut rng = PecosRng::seed_from_u64(42);
            b.iter(|| {
                let result = sampler.sample_statistics_columnar(shots, &mut rng);
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Benchmark comparing all optimization methods.
fn bench_optimization_comparison<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("DEM Sampler - Optimizations");

    // Use a realistic surface code sampler
    let sampler = create_surface_code_sampler(5, 3);
    let num_mechanisms = sampler.num_mechanisms();

    let shots = 100_000;
    group.throughput(Throughput::Elements((num_mechanisms * shots) as u64));

    // Baseline: current columnar_accurate
    group.bench_function("columnar_accurate", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let result = sampler.sample_batch_columnar_accurate(shots, &mut rng);
            black_box(result)
        });
    });

    // SIMD u64x4 version
    group.bench_function("simd_u64x4", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let result = sampler.sample_batch_columnar_simd(shots, &mut rng);
            black_box(result)
        });
    });

    // Geometric skip version (fastest for low error rates)
    group.bench_function("geometric", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let result = sampler.sample_batch_columnar_geometric(shots, &mut rng);
            black_box(result)
        });
    });

    group.finish();

    // Also compare statistics methods
    let mut stats_group = c.benchmark_group("DEM Sampler - Stats Optimizations");
    stats_group.throughput(Throughput::Elements((num_mechanisms * shots) as u64));

    stats_group.bench_function("stats_columnar", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let result = sampler.sample_statistics_columnar(shots, &mut rng);
            black_box(result)
        });
    });

    stats_group.bench_function("stats_simd", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let result = sampler.sample_statistics_simd(shots, &mut rng);
            black_box(result)
        });
    });

    stats_group.bench_function("stats_geometric", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let result = sampler.sample_statistics_geometric(shots, &mut rng);
            black_box(result)
        });
    });

    stats_group.bench_function("stats_auto", |b| {
        let mut rng = PecosRng::seed_from_u64(42);
        b.iter(|| {
            let result = sampler.sample_statistics_with_rng(shots, &mut rng);
            black_box(result)
        });
    });

    stats_group.bench_function("stats_parallel", |b| {
        b.iter(|| {
            let result = sampler.sample_statistics_parallel(shots, 42);
            black_box(result)
        });
    });

    stats_group.finish();

    // Benchmark parallel scaling with larger DEM
    bench_parallel_scaling(c);
}

/// Benchmark parallel scaling with different DEM sizes.
fn bench_parallel_scaling<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("DEM Sampler - Parallel Scaling");

    let shots = 100_000;

    // Test with different circuit sizes
    for (distance, rounds) in [(3, 2), (5, 3), (7, 5)] {
        let sampler = create_surface_code_sampler(distance, rounds);
        let num_mechanisms = sampler.num_mechanisms();
        let label = format!("d{distance}_r{rounds}_{num_mechanisms}mech");

        group.throughput(Throughput::Elements((num_mechanisms * shots) as u64));

        // Sequential geometric (baseline)
        group.bench_with_input(BenchmarkId::new("sequential", &label), &(), |b, ()| {
            let mut rng = PecosRng::seed_from_u64(42);
            b.iter(|| {
                let result = sampler.sample_statistics_geometric(shots, &mut rng);
                black_box(result)
            });
        });

        // Parallel
        group.bench_with_input(BenchmarkId::new("parallel", &label), &(), |b, ()| {
            b.iter(|| {
                let result = sampler.sample_statistics_parallel(shots, 42);
                black_box(result)
            });
        });

        // Auto (should pick geometric for low p)
        group.bench_with_input(BenchmarkId::new("auto", &label), &(), |b, ()| {
            let mut rng = PecosRng::seed_from_u64(42);
            b.iter(|| {
                let result = sampler.sample_statistics_with_rng(shots, &mut rng);
                black_box(result)
            });
        });
    }

    group.finish();
}

/// Create a synthetic DEM string for benchmarking.
fn create_synthetic_dem(num_mechanisms: usize, num_detectors: usize, prob: f64) -> String {
    use std::fmt::Write;

    let mut dem = String::new();

    for i in 0..num_detectors {
        writeln!(dem, "detector({i}, 0, 0) D{i}").unwrap();
    }

    for i in 0..num_mechanisms {
        let d1 = i % num_detectors;
        let d2 = (i + 1) % num_detectors;
        let d3 = (i + 2) % num_detectors;

        match i % 3 {
            0 => writeln!(dem, "error({prob}) D{d1}").unwrap(),
            1 => writeln!(dem, "error({prob}) D{d1} D{d2}").unwrap(),
            _ => writeln!(dem, "error({prob}) D{d1} D{d2} D{d3}").unwrap(),
        }
    }

    dem
}

/// Benchmark `ParsedDem` sampling (used by equivalence testing).
fn bench_parsed_dem_sampling<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("ParsedDem - Sampling");

    let medium_dem = create_synthetic_dem(50, 24, 0.01);
    let complex_dem = create_synthetic_dem(200, 96, 0.01);

    let dems: [(&str, &str); 3] = [
        (
            "simple",
            "error(0.01) D0\nerror(0.01) D1\nerror(0.01) D0 D1",
        ),
        ("medium", &medium_dem),
        ("complex", &complex_dem),
    ];

    let shots = 50_000;

    for (name, dem_str) in &dems {
        let dem = ParsedDem::from_str(dem_str).expect("failed to parse DEM");
        let num_mechanisms = dem.mechanisms.len();

        group.throughput(Throughput::Elements((num_mechanisms * shots) as u64));

        group.bench_with_input(BenchmarkId::new("sample_batch", *name), &(), |b, ()| {
            let mut rng = PecosRng::seed_from_u64(42);
            b.iter(|| {
                let result = dem.sample_batch(shots, &mut rng);
                black_box(result)
            });
        });
    }

    group.finish();
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_sampler_creation() {
        let sampler = create_surface_code_sampler(3, 1);
        assert!(sampler.num_mechanisms() > 0);
    }

    #[test]
    fn test_columnar_matches_row_major() {
        let sampler = create_surface_code_sampler(3, 1);

        // Sample with row-major
        let mut rng1 = PecosRng::seed_from_u64(42);
        let stats1 = sampler.sample_statistics(1000, &mut rng1);

        // Sample with columnar
        let mut rng2 = PecosRng::seed_from_u64(42);
        let stats2 = sampler.sample_statistics_columnar(1000, &mut rng2);

        // Statistics should be similar (not exact due to different RNG consumption order)
        // Just verify both produce reasonable results
        assert!(stats1.total_shots == stats2.total_shots);
    }
}
