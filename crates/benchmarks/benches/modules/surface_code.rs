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

//! Surface code benchmarks for `SymbolicSparseStab` and `MeasurementSampler`.
//!
//! These benchmarks simulate realistic QEC workloads:
//! - Distance 5, 11, 17 surface codes
//! - Multiple rounds of syndrome extraction
//! - Both simulation and sampling phases

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use pecos::prelude::*;
use pecos::qsim::measurement_sampler::{MeasurementSampler, SequentialMeasurementSampler};
use rand::Rng;
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_surface_code_simulation(c);
    bench_surface_code_sampling(c);
    bench_surface_code_shot_scaling(c);
    bench_simd_vs_scalar(c);
}

/// Surface code parameters for a given distance.
struct SurfaceCodeParams {
    distance: usize,
    /// Total qubits: data + ancillas
    num_qubits: usize,
    /// Number of data qubits: d^2
    num_data: usize,
    /// Number of ancilla qubits (X and Z stabilizers): d^2 - 1
    num_ancillas: usize,
    /// Data qubit indices: `0..num_data`
    #[allow(dead_code)]
    data_start: usize,
    /// Ancilla qubit indices: `num_data..num_qubits`
    ancilla_start: usize,
}

impl SurfaceCodeParams {
    fn new(distance: usize) -> Self {
        let num_data = distance * distance;
        let num_ancillas = num_data - 1; // (d^2-1)/2 X-type + (d^2-1)/2 Z-type
        let num_qubits = num_data + num_ancillas;
        Self {
            distance,
            num_qubits,
            num_data,
            num_ancillas,
            data_start: 0,
            ancilla_start: num_data,
        }
    }

    /// Get the neighbors of an ancilla (simplified model).
    /// In a real surface code, ancillas connect to 2-4 data qubits.
    /// We model this as: bulk ancillas have 4 neighbors, boundary have 2.
    fn ancilla_neighbors(&self, ancilla_idx: usize) -> Vec<usize> {
        // Simplified: map ancilla to a position and find its data qubit neighbors
        // For benchmarking purposes, we just need realistic connectivity patterns
        let d = self.distance;
        let ancilla_local = ancilla_idx; // 0..num_ancillas

        // Arrange ancillas in a (d-1) x d + d x (d-1) pattern approximately
        // Simplified: just use modular arithmetic to get 2-4 neighbors
        let mut neighbors = Vec::with_capacity(4);

        // Each ancilla connects to some data qubits based on its position
        // Use a deterministic pattern that gives 2-4 neighbors
        let base = ancilla_local % self.num_data;
        neighbors.push(base);

        if ancilla_local + 1 < self.num_data {
            neighbors.push((base + 1) % self.num_data);
        }

        // Bulk ancillas get more neighbors
        if ancilla_local < self.num_ancillas / 2 {
            // X-type stabilizers (roughly half)
            if base + d < self.num_data {
                neighbors.push(base + d);
            }
            if ancilla_local > d && base >= d {
                neighbors.push(base - d);
            }
        } else {
            // Z-type stabilizers (roughly half)
            if base + d < self.num_data {
                neighbors.push(base + d);
            }
            if base + d + 1 < self.num_data {
                neighbors.push((base + d + 1) % self.num_data);
            }
        }

        neighbors
    }
}

/// Simulate surface code syndrome extraction using `SymbolicSparseStab`.
///
/// This creates realistic measurement patterns where:
/// - Each ancilla is entangled with 2-4 data qubits via CNOT gates
/// - Ancillas are measured, creating non-deterministic outcomes in round 1
/// - Subsequent rounds create computed measurements (XOR with previous round)
fn simulate_surface_code(params: &SurfaceCodeParams, rounds: usize) -> StdSymbolicSparseStab {
    let mut sim = StdSymbolicSparseStab::new(params.num_qubits);

    // Initialize data qubits in |+> state (typical for X-error detection)
    for i in 0..params.num_data {
        sim.h(i);
    }

    // Perform syndrome extraction rounds
    for _round in 0..rounds {
        // Reset ancillas to |0>
        // In practice we'd do a reset, but for symbolic sim ancillas start in |0>
        // and measurements handle the state update

        // Entangle ancillas with their data qubit neighbors
        for a in 0..params.num_ancillas {
            let ancilla = params.ancilla_start + a;
            let neighbors = params.ancilla_neighbors(a);

            // Apply CNOT gates: ancilla as target for Z-stabilizers, as control for X-stabilizers
            // Simplified: alternate pattern
            if a < params.num_ancillas / 2 {
                // X-type: CNOT with ancilla as control
                for &data in &neighbors {
                    sim.cx(ancilla, data);
                }
            } else {
                // Z-type: CNOT with ancilla as target
                for &data in &neighbors {
                    sim.cx(data, ancilla);
                }
            }
        }

        // Measure all ancillas
        for a in 0..params.num_ancillas {
            let ancilla = params.ancilla_start + a;
            sim.mz(ancilla);
        }
    }

    sim
}

/// Benchmark the simulation phase (running circuits through `SymbolicSparseStab`).
fn bench_surface_code_simulation<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Surface Code - Simulation");

    // Test different distances and round counts
    for distance in [5, 11, 17] {
        let params = SurfaceCodeParams::new(distance);

        for rounds in [1, 3, 5, 10] {
            let label = format!("d{distance}_r{rounds}");

            // Throughput: number of gates + measurements
            // Rough estimate: rounds * (ancillas * avg_neighbors CNOTs + ancillas measurements)
            let ops_per_run = rounds * (params.num_ancillas * 3 + params.num_ancillas); // ~3 CNOTs + 1 meas per ancilla
            group.throughput(Throughput::Elements(ops_per_run as u64));

            group.bench_with_input(BenchmarkId::new("symbolic_sim", &label), &(), |b, ()| {
                b.iter(|| black_box(simulate_surface_code(&params, rounds)));
            });
        }
    }

    group.finish();
}

/// Benchmark the sampling phase with pre-computed measurement histories.
fn bench_surface_code_sampling<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Surface Code - Sampling");

    let shots = 100_000;

    for distance in [5, 11, 17] {
        let params = SurfaceCodeParams::new(distance);

        for rounds in [1, 5, 10, 20] {
            let label = format!("d{distance}_r{rounds}");

            // Pre-compute the measurement history
            let sim = simulate_surface_code(&params, rounds);
            let history = sim.measurement_history().clone();
            let num_measurements = history.len();

            // Create samplers
            let sequential_sampler = SequentialMeasurementSampler::new(&history);
            let sampler = MeasurementSampler::new(&history);

            group.throughput(Throughput::Elements(num_measurements as u64 * shots as u64));

            group.bench_with_input(BenchmarkId::new("sequential", &label), &(), |b, ()| {
                b.iter(|| black_box(sequential_sampler.sample(shots)));
            });

            group.bench_with_input(BenchmarkId::new("columnar", &label), &(), |b, ()| {
                b.iter(|| black_box(sampler.sample(shots)));
            });
        }
    }

    group.finish();
}

/// Benchmark how sampling scales with shot count.
///
/// Tests a fixed circuit (d=11, 5 rounds = 600 measurements) with varying shot counts
/// from 1K to 1B to understand shot scaling behavior.
fn bench_surface_code_shot_scaling<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Surface Code - Shot Scaling");

    // Use d=11, 5 rounds as a representative workload (600 measurements)
    let params = SurfaceCodeParams::new(11);
    let rounds = 5;
    let sim = simulate_surface_code(&params, rounds);
    let history = sim.measurement_history().clone();
    let num_measurements = history.len();

    let sequential_sampler = SequentialMeasurementSampler::new(&history);
    let sampler = MeasurementSampler::new(&history);

    // Test shot counts from 1K to 1B (powers of 10)
    // Note: 1B shots may take a while, so we include it but it can be skipped
    for shots in [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000] {
        let label = format!("{shots}shots");

        group.throughput(Throughput::Elements(num_measurements as u64 * shots as u64));

        // Only run sequential for smaller shot counts (it's too slow otherwise)
        if shots <= 1_000_000 {
            group.bench_with_input(
                BenchmarkId::new("sequential", &label),
                &shots,
                |b, &shots| b.iter(|| black_box(sequential_sampler.sample(shots))),
            );
        }

        group.bench_with_input(BenchmarkId::new("columnar", &label), &shots, |b, &shots| {
            b.iter(|| black_box(sampler.sample(shots)));
        });
    }

    group.finish();
}

/// Benchmark comparing SIMD-native vs scalar APIs.
///
/// This isolates the SIMD processing time from the conversion overhead.
fn bench_simd_vs_scalar<M: Measurement>(c: &mut Criterion<M>) {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    let mut group = c.benchmark_group("SIMD vs Scalar");

    // Use d=11, 5 rounds as a representative workload
    let params = SurfaceCodeParams::new(11);
    let rounds = 5;
    let sim = simulate_surface_code(&params, rounds);
    let history = sim.measurement_history().clone();
    let num_measurements = history.len();

    let sampler = MeasurementSampler::new(&history);
    let shots = 100_000;

    group.throughput(Throughput::Elements(num_measurements as u64 * shots as u64));

    // Regular API: sample() returns SampleResult (uses SmallRng internally)
    group.bench_with_input(BenchmarkId::new("sample", "d11_r5"), &(), |b, ()| {
        b.iter(|| black_box(sampler.sample(shots)));
    });

    // sample_with_rng: should be identical to sample() but with explicit RNG
    group.bench_with_input(
        BenchmarkId::new("sample_with_rng", "d11_r5"),
        &(),
        |b, ()| {
            b.iter(|| {
                let mut rng = SmallRng::from_rng(&mut rand::rng());
                black_box(sampler.sample_with_rng(shots, &mut rng))
            });
        },
    );

    // Raw API with SmallRng seeded from ThreadRng (exactly like sample())
    group.bench_with_input(
        BenchmarkId::new("sample_raw_from_threadrng", "d11_r5"),
        &(),
        |b, ()| {
            b.iter(|| {
                let mut rng = SmallRng::from_rng(&mut rand::rng());
                black_box(sampler.sample_raw(shots, &mut rng))
            });
        },
    );

    // Raw API with SmallRng seeded from u64
    group.bench_with_input(
        BenchmarkId::new("sample_raw_seed_u64", "d11_r5"),
        &(),
        |b, ()| {
            b.iter(|| {
                let mut rng = SmallRng::seed_from_u64(42);
                black_box(sampler.sample_raw(shots, &mut rng))
            });
        },
    );

    // Also test with ThreadRng for comparison
    group.bench_with_input(
        BenchmarkId::new("sample_raw_threadrng", "d11_r5"),
        &(),
        |b, ()| {
            b.iter(|| black_box(sampler.sample_raw(shots, &mut rand::rng())));
        },
    );

    group.finish();

    // Micro-benchmark to isolate XOR operation costs
    bench_xor_operations(c);
}

/// Micro-benchmark: Pure XOR operation cost.
///
/// This isolates the cost of XOR operations without RNG overhead.
fn bench_xor_operations<M: Measurement>(c: &mut Criterion<M>) {
    use rand::SeedableRng;
    use rand::rngs::SmallRng;
    use wide::u64x4;

    let mut group = c.benchmark_group("XOR Operations");

    let shots: usize = 100_000;
    let num_words = shots.div_ceil(64);
    let num_simd_words = num_words.div_ceil(4);

    // Pre-generate some random columns
    let mut rng = SmallRng::seed_from_u64(42);
    let col_a: Vec<u64x4> = (0..num_simd_words)
        .map(|_| u64x4::new([rng.random(), rng.random(), rng.random(), rng.random()]))
        .collect();
    let col_b: Vec<u64x4> = (0..num_simd_words)
        .map(|_| u64x4::new([rng.random(), rng.random(), rng.random(), rng.random()]))
        .collect();
    let col_c: Vec<u64x4> = (0..num_simd_words)
        .map(|_| u64x4::new([rng.random(), rng.random(), rng.random(), rng.random()]))
        .collect();

    group.throughput(Throughput::Elements(shots as u64));

    // XOR 2 columns
    group.bench_function("xor_2_columns", |b| {
        b.iter(|| {
            let mut result = col_a.clone();
            for (r, b) in result.iter_mut().zip(col_b.iter()) {
                *r ^= *b;
            }
            black_box(result)
        });
    });

    // XOR 3 columns
    group.bench_function("xor_3_columns", |b| {
        b.iter(|| {
            let mut result = col_a.clone();
            for (r, b) in result.iter_mut().zip(col_b.iter()) {
                *r ^= *b;
            }
            for (r, c) in result.iter_mut().zip(col_c.iter()) {
                *r ^= *c;
            }
            black_box(result)
        });
    });

    // Clone column (simulating Copy operation)
    group.bench_function("clone_column", |b| {
        b.iter(|| black_box(col_a.clone()));
    });

    // Flip column (simulating CopyFlipped)
    group.bench_function("flip_column", |b| {
        b.iter(|| {
            let result: Vec<u64x4> = col_a.iter().map(|v| !*v).collect();
            black_box(result)
        });
    });

    group.finish();
}

#[cfg(test)]
mod tests {

    #[test]
    fn test_surface_code_params() {
        let p5 = SurfaceCodeParams::new(5);
        assert_eq!(p5.num_data, 25);
        assert_eq!(p5.num_ancillas, 24);
        assert_eq!(p5.num_qubits, 49);

        let p11 = SurfaceCodeParams::new(11);
        assert_eq!(p11.num_data, 121);
        assert_eq!(p11.num_ancillas, 120);
        assert_eq!(p11.num_qubits, 241);

        let p17 = SurfaceCodeParams::new(17);
        assert_eq!(p17.num_data, 289);
        assert_eq!(p17.num_ancillas, 288);
        assert_eq!(p17.num_qubits, 577);
    }

    #[test]
    fn test_surface_code_simulation_runs() {
        // Just verify simulation completes without panic
        let params = SurfaceCodeParams::new(5);
        let sim = simulate_surface_code(&params, 3);

        // Should have 3 rounds * 24 ancillas = 72 measurements
        assert_eq!(sim.measurement_count(), 72);
    }

    #[test]
    fn test_surface_code_sampling() {
        let params = SurfaceCodeParams::new(5);
        let sim = simulate_surface_code(&params, 2);
        let history = sim.measurement_history();

        let sampler = MeasurementSampler::new(history);
        let result = sampler.sample(1000);

        assert_eq!(result.shots(), 1000);
        assert_eq!(result.num_measurements(), 48); // 2 rounds * 24 ancillas
    }

    #[test]
    fn test_ancilla_neighbors() {
        let params = SurfaceCodeParams::new(5);

        // Each ancilla should have 2-4 neighbors
        for a in 0..params.num_ancillas {
            let neighbors = params.ancilla_neighbors(a);
            assert!(
                neighbors.len() >= 1 && neighbors.len() <= 4,
                "Ancilla {} has {} neighbors",
                a,
                neighbors.len()
            );

            // All neighbors should be valid data qubit indices
            for &n in &neighbors {
                assert!(n < params.num_data, "Invalid neighbor index {}", n);
            }
        }
    }

    #[test]
    fn test_measurement_type_distribution() {
        // Analyze what types of measurements we're generating
        let params = SurfaceCodeParams::new(11);
        let sim = simulate_surface_code(&params, 5);
        let history = sim.measurement_history();
        let kinds = MeasurementKind::from_history(history);

        let mut fixed = 0;
        let mut random = 0;
        let mut copy = 0;
        let mut copy_flipped = 0;
        let mut computed = 0;
        let mut computed_deps_sum = 0;

        for kind in &kinds {
            match kind {
                MeasurementKind::Fixed(_) => fixed += 1,
                MeasurementKind::Random => random += 1,
                MeasurementKind::Copy(_) => copy += 1,
                MeasurementKind::CopyFlipped(_) => copy_flipped += 1,
                MeasurementKind::Computed { deps, .. } => {
                    computed += 1;
                    computed_deps_sum += deps.len();
                }
            }
        }

        let total = kinds.len();
        let avg_deps = if computed > 0 {
            computed_deps_sum as f64 / computed as f64
        } else {
            0.0
        };

        println!("\n=== Measurement Type Distribution (d=11, 5 rounds) ===");
        println!("Total measurements: {total}");
        println!(
            "  Fixed:       {fixed:4} ({:.1}%)",
            100.0 * fixed as f64 / total as f64
        );
        println!(
            "  Random:      {random:4} ({:.1}%)",
            100.0 * random as f64 / total as f64
        );
        println!(
            "  Copy:        {copy:4} ({:.1}%)",
            100.0 * copy as f64 / total as f64
        );
        println!(
            "  CopyFlipped: {copy_flipped:4} ({:.1}%)",
            100.0 * copy_flipped as f64 / total as f64
        );
        println!(
            "  Computed:    {computed:4} ({:.1}%) - avg {avg_deps:.1} deps",
            100.0 * computed as f64 / total as f64
        );
    }
}
