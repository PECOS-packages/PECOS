// Copyright 2026 The PECOS Developers
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file
// except in compliance with the License. You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software distributed under the
// License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
// express or implied. See the License for the specific language governing permissions and
// limitations under the License.

//! Performance benchmarks: STN vs `StabVec`.
//!
//! Measures wall time for the same circuits on both simulators to find
//! the crossover point where STN becomes faster.
//!
//! Run: `cargo bench -p pecos-stab-tn`

use criterion::{BenchmarkId, Criterion, measurement::Measurement};
use nalgebra::DMatrix;
use num_complex::Complex64;
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{ArbitraryRotationGateable, CliffordGateable, StabVec};
use pecos_stab_tn::mps::svd;
use pecos_stab_tn::stab_mps::StabMps;
use pecos_stab_tn::stab_mps::mast::Mast;
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_vary_t_count(c);
    bench_vary_qubits(c);
    bench_measurement(c);
    bench_stab_mps_at_scale(c);
    bench_mast_vs_stn(c);
    bench_disentangling(c);
    bench_svd_comparison(c);
    bench_adaptive_truncation(c);
}

/// Build a random Clifford+T circuit and run it.
fn run_circuit<S: ArbitraryRotationGateable>(sim: &mut S, num_qubits: usize, num_t_gates: usize) {
    let t = Angle64::QUARTER_TURN / 2u64;

    // Clifford entangling layer
    for q in 0..num_qubits {
        sim.h(&[QubitId(q)]);
    }
    for q in 0..num_qubits - 1 {
        sim.cx(&[(QubitId(q), QubitId(q + 1))]);
    }

    // T gates on rotating qubits
    for i in 0..num_t_gates {
        let q = i % num_qubits;
        sim.rz(t, &[QubitId(q)]);
    }

    // Another Clifford layer
    for q in (0..num_qubits - 1).rev() {
        sim.cx(&[(QubitId(q + 1), QubitId(q))]);
    }
    for q in 0..num_qubits {
        sim.h(&[QubitId(q)]);
    }
}

/// Benchmark: vary T-gate count at fixed qubit count.
fn bench_vary_t_count<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("STN vs CRZ: vary T-count (10 qubits)");
    group.sample_size(10);

    let num_qubits = 10;
    for &num_t in &[2, 4, 8, 12, 16, 20] {
        group.bench_with_input(BenchmarkId::new("StabVec", num_t), &num_t, |b, &nt| {
            b.iter(|| {
                let mut sim = StabVec::builder(num_qubits).seed(42).build();
                run_circuit(&mut sim, num_qubits, nt);
                black_box(&sim);
            });
        });

        group.bench_with_input(BenchmarkId::new("STN", num_t), &num_t, |b, &nt| {
            b.iter(|| {
                let mut sim = StabMps::builder(num_qubits).seed(42).build();
                run_circuit(&mut sim, num_qubits, nt);
                black_box(&sim);
            });
        });
    }
    group.finish();
}

/// Benchmark: vary qubit count at fixed T-gate count.
fn bench_vary_qubits<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("STN vs CRZ: vary qubits (8 T gates)");
    group.sample_size(10);

    let num_t = 8;
    for &num_qubits in &[4, 8, 16, 32, 64] {
        group.bench_with_input(
            BenchmarkId::new("StabVec", num_qubits),
            &num_qubits,
            |b, &nq| {
                b.iter(|| {
                    let mut sim = StabVec::builder(nq).seed(42).build();
                    run_circuit(&mut sim, nq, num_t);
                    black_box(&sim);
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("STN", num_qubits),
            &num_qubits,
            |b, &nq| {
                b.iter(|| {
                    let mut sim = StabMps::builder(nq).seed(42).build();
                    run_circuit(&mut sim, nq, num_t);
                    black_box(&sim);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark: measurement cost comparison.
fn bench_measurement<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("STN vs CRZ: measurement (10 qubits, 8 T)");
    group.sample_size(10);

    let num_qubits = 10;
    let num_t = 8;

    group.bench_function("StabVec", |b| {
        b.iter(|| {
            let mut sim = StabVec::builder(num_qubits).seed(42).build();
            run_circuit(&mut sim, num_qubits, num_t);
            let results = sim.mz(&(0..num_qubits).map(QubitId).collect::<Vec<_>>());
            black_box(&results);
        });
    });

    group.bench_function("STN", |b| {
        b.iter(|| {
            let mut sim = StabMps::builder(num_qubits).seed(42).build();
            run_circuit(&mut sim, num_qubits, num_t);
            let results = sim.mz(&(0..num_qubits).map(QubitId).collect::<Vec<_>>());
            black_box(&results);
        });
    });

    group.finish();
}

/// Benchmark: MAST vs STN at varying T-count.
fn bench_mast_vs_stn<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("MAST vs STN: vary T-count (20 qubits)");
    group.sample_size(10);

    let num_qubits = 20;
    for &num_t in &[4, 8, 16, 20] {
        group.bench_with_input(BenchmarkId::new("STN", num_t), &num_t, |b, &nt| {
            b.iter(|| {
                let mut sim = StabMps::builder(num_qubits).seed(42).build();
                run_circuit(&mut sim, num_qubits, nt);
                black_box(&sim);
            });
        });

        group.bench_with_input(BenchmarkId::new("MAST", num_t), &num_t, |b, &nt| {
            b.iter(|| {
                let mut sim = Mast::with_seed(num_qubits, nt + 2, 42);
                run_circuit(&mut sim, num_qubits, nt);
                sim.mz(&[QubitId(0)]); // Force projection
                black_box(&sim);
            });
        });
    }
    group.finish();
}

/// Benchmark: STN at scale (CRZ can't do this).
fn bench_stab_mps_at_scale<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("STN at scale");
    group.sample_size(10);

    for &num_qubits in &[50, 100, 200] {
        let num_t = num_qubits / 2;
        group.bench_with_input(
            BenchmarkId::new("STN circuit", num_qubits),
            &num_qubits,
            |b, &nq| {
                b.iter(|| {
                    let mut sim = StabMps::builder(nq).seed(42).build();
                    run_circuit(&mut sim, nq, num_t);
                    black_box(&sim);
                });
            },
        );
    }
    group.finish();
}

/// Build a "hard" circuit: interleaved Clifford + T layers that create real MPS entanglement.
/// Each layer: CX chain + T on all qubits. Repeats `depth` times.
fn run_interleaved_circuit<S: ArbitraryRotationGateable>(
    sim: &mut S,
    num_qubits: usize,
    depth: usize,
) {
    let t = Angle64::QUARTER_TURN / 2u64;

    for layer in 0..depth {
        // Clifford entangling: H + CX chain (alternating direction)
        for q in 0..num_qubits {
            sim.h(&[QubitId(q)]);
        }
        if layer % 2 == 0 {
            for q in 0..num_qubits - 1 {
                sim.cx(&[(QubitId(q), QubitId(q + 1))]);
            }
        } else {
            for q in (0..num_qubits - 1).rev() {
                sim.cx(&[(QubitId(q + 1), QubitId(q))]);
            }
        }

        // T gate on every qubit
        for q in 0..num_qubits {
            sim.rz(t, &[QubitId(q)]);
        }
    }
}

/// Benchmark: disentangling cost and effectiveness.
///
/// Measures wall time and bond dim reduction from heuristic disentangling.
/// This provides the baseline for comparing against OFD.
fn bench_disentangling<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("STN disentangling");
    group.sample_size(10);

    // Use the interleaved circuit which creates real MPS entanglement
    let num_qubits = 10;
    for &depth in &[1, 2, 3] {
        let label = format!("{num_qubits}q/depth{depth}");

        group.bench_with_input(BenchmarkId::new("no_disent", &label), &depth, |b, &d| {
            b.iter(|| {
                let mut sim = StabMps::builder(num_qubits).seed(42).build();
                run_interleaved_circuit(&mut sim, num_qubits, d);
                black_box(sim.max_bond_dim());
            });
        });

        group.bench_with_input(
            BenchmarkId::new("heuristic_1sweep", &label),
            &depth,
            |b, &d| {
                b.iter(|| {
                    let mut sim = StabMps::builder(num_qubits).seed(42).build();
                    run_interleaved_circuit(&mut sim, num_qubits, d);
                    sim.disentangle(1);
                    black_box(sim.max_bond_dim());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("heuristic_3sweeps", &label),
            &depth,
            |b, &d| {
                b.iter(|| {
                    let mut sim = StabMps::builder(num_qubits).seed(42).build();
                    run_interleaved_circuit(&mut sim, num_qubits, d);
                    sim.disentangle(3);
                    black_box(sim.max_bond_dim());
                });
            },
        );
    }
    group.finish();

    // Report bond dims and GF(2) diagnostic for reference
    eprintln!("\n=== Bond dimension report (interleaved circuits) ===");
    for &depth in &[1, 2, 3] {
        let mut sim = StabMps::builder(num_qubits).seed(42).build();
        run_interleaved_circuit(&mut sim, num_qubits, depth);
        let before = sim.max_bond_dim();
        let gf2_rank = sim.gf2_matrix().gf2_rank();
        let gf2_gates = sim.gf2_matrix().num_gates();
        let gf2_min = sim.theoretical_min_bond_dim();
        let applied = sim.disentangle(3);
        let after = sim.max_bond_dim();
        eprintln!(
            "  {num_qubits}q/depth{depth}: bond_dim {before} -> {after} ({applied} gates), GF2: {gf2_gates} gates rank {gf2_rank} (theory min {gf2_min})"
        );
    }
}

/// Benchmark: full SVD vs randomized SVD at various matrix sizes.
///
/// The randomized SVD (Halko-Martinsson-Tropp) gives O(mnr) cost vs
/// O(mn*min(m,n)) for full SVD. This benchmark measures the crossover.
fn bench_svd_comparison<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("SVD full vs randomized");
    group.sample_size(10);

    // Generate a test matrix of given size with known rank structure
    let make_matrix = |rows: usize, cols: usize| -> DMatrix<Complex64> {
        DMatrix::from_fn(rows, cols, |i, j| {
            let r = (i * 7 + j * 13 + 5) & 0xFFFF;
            let c = (i * 3 + j * 11 + 2) & 0xFFFF;
            Complex64::new(f64::from(r as u16).sin(), f64::from(c as u16).cos())
        })
    };

    for &(rows, cols, max_rank) in &[
        (64, 64, 8),    // Small matrix, heavy truncation
        (128, 128, 16), // Medium matrix
        (256, 256, 32), // Large matrix -- rSVD should win here
        (512, 512, 32), // Very large -- rSVD should win clearly
    ] {
        let label = format!("{rows}x{cols}/r{max_rank}");
        let m = make_matrix(rows, cols);

        group.bench_with_input(BenchmarkId::new("full_svd", &label), &m, |b, matrix| {
            b.iter(|| {
                let result = svd::truncated_svd(matrix, max_rank, 1e-12).unwrap();
                black_box(result.singular_values.len());
            });
        });

        group.bench_with_input(BenchmarkId::new("auto_svd", &label), &m, |b, matrix| {
            b.iter(|| {
                let result = svd::truncated_svd_auto(matrix, max_rank, 1e-12).unwrap();
                black_box(result.singular_values.len());
            });
        });
    }
    group.finish();
}

/// Benchmark: adaptive truncation (error-budget) vs fixed `max_bond_dim`.
fn bench_adaptive_truncation<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("STN adaptive truncation");
    group.sample_size(10);

    let num_qubits = 20;
    let depth = 2;

    // Fixed max_bond_dim = 64 (default)
    group.bench_function("fixed_chi64", |b| {
        b.iter(|| {
            let mut sim = StabMps::builder(num_qubits).seed(42).build();
            run_interleaved_circuit(&mut sim, num_qubits, depth);
            black_box(sim.max_bond_dim());
        });
    });

    // Adaptive with error budget 1e-6, cap 64
    group.bench_function("adaptive_1e-6_cap64", |b| {
        b.iter(|| {
            let mut sim = StabMps::builder(num_qubits)
                .seed(42)
                .max_truncation_error(1e-6)
                .build();
            run_interleaved_circuit(&mut sim, num_qubits, depth);
            black_box(sim.max_bond_dim());
        });
    });

    // Adaptive with error budget 1e-3, cap 64
    group.bench_function("adaptive_1e-3_cap64", |b| {
        b.iter(|| {
            let mut sim = StabMps::builder(num_qubits)
                .seed(42)
                .max_truncation_error(1e-3)
                .build();
            run_interleaved_circuit(&mut sim, num_qubits, depth);
            black_box(sim.max_bond_dim());
        });
    });

    group.finish();

    // Report bond dims
    eprintln!("\n=== Adaptive truncation report ({num_qubits}q, depth {depth}) ===");
    for &(label, err) in &[
        ("fixed", -1.0),
        ("adaptive_1e-6", 1e-6),
        ("adaptive_1e-3", 1e-3),
    ] {
        let mut builder = StabMps::builder(num_qubits).seed(42);
        if err > 0.0 {
            builder = builder.max_truncation_error(err);
        }
        let mut sim = builder.build();
        run_interleaved_circuit(&mut sim, num_qubits, depth);
        eprintln!("  {label}: max_bond_dim = {}", sim.max_bond_dim());
    }
}
