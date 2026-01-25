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

//! Benchmarks comparing sparse vs dense state vector simulators.

use criterion::{BenchmarkId, Criterion, Throughput};
use pecos_core::QubitId;
use pecos_qsim::{CliffordGateable, QuantumSimulator, SparseStateVec, SparseStateVecSoA, StateVec};

/// Benchmark sparse state vector on sparse-friendly circuits (X, Z, CX only)
fn bench_sparse_friendly(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vs_dense/sparse_friendly");

    for num_qubits in [10, 14, 18, 20] {
        group.throughput(Throughput::Elements(num_qubits as u64));

        // Sparse version
        group.bench_with_input(
            BenchmarkId::new("sparse", num_qubits),
            &num_qubits,
            |b, &n| {
                let mut sim = SparseStateVec::new(n);
                b.iter(|| {
                    for q in 0..n {
                        sim.x(&[QubitId(q)]);
                        sim.z(&[QubitId(q)]);
                    }
                    for q in 0..n - 1 {
                        sim.cx(&[QubitId(q), QubitId(q + 1)]);
                    }
                });
            },
        );

        // Dense version
        group.bench_with_input(
            BenchmarkId::new("dense", num_qubits),
            &num_qubits,
            |b, &n| {
                let mut sim = StateVec::new(n);
                b.iter(|| {
                    for q in 0..n {
                        sim.x(&[QubitId(q)]);
                        sim.z(&[QubitId(q)]);
                    }
                    for q in 0..n - 1 {
                        sim.cx(&[QubitId(q), QubitId(q + 1)]);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark sparse state vector with varying superposition levels
fn bench_varying_superposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_vs_dense/varying_superposition");
    let num_qubits = 16;

    for h_qubits in [0, 4, 8, 10, 12, 14, 16] {
        let expected_amps = 1usize << h_qubits;
        group.throughput(Throughput::Elements(expected_amps as u64));

        // Sparse version
        group.bench_with_input(
            BenchmarkId::new("sparse", h_qubits),
            &h_qubits,
            |b, &h| {
                let mut sim = SparseStateVec::new(num_qubits);
                b.iter(|| {
                    sim.reset();
                    for q in 0..h {
                        sim.h(&[QubitId(q)]);
                    }
                });
            },
        );

        // Dense version
        group.bench_with_input(
            BenchmarkId::new("dense", h_qubits),
            &h_qubits,
            |b, &h| {
                let mut sim = StateVec::new(num_qubits);
                b.iter(|| {
                    sim.reset();
                    for q in 0..h {
                        sim.h(&[QubitId(q)]);
                    }
                });
            },
        );
    }

    group.finish();
}

/// Benchmark individual operations on sparse state vector
fn bench_sparse_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_ops");

    // Test with different amplitude counts
    for h_qubits in [0, 4, 8, 10] {
        let label = format!("{}amps", 1usize << h_qubits);

        // H gate (doubles amplitude count)
        group.bench_function(BenchmarkId::new("h_gate", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            // Set up initial state with 2^h_qubits amplitudes
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.h(&[QubitId(h_qubits)]); // Apply H to next qubit
                sim.h(&[QubitId(h_qubits)]); // Apply H again to restore
            });
        });

        // X gate (permutes amplitudes, count stays same)
        group.bench_function(BenchmarkId::new("x_gate", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.x(&[QubitId(0)]);
            });
        });

        // Z gate (in-place phase flip)
        group.bench_function(BenchmarkId::new("z_gate", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.z(&[QubitId(0)]);
            });
        });

        // CX gate
        group.bench_function(BenchmarkId::new("cx_gate", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[QubitId(0), QubitId(1)]);
            });
        });

        // CZ gate (in-place)
        group.bench_function(BenchmarkId::new("cz_gate", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cz(&[QubitId(0), QubitId(1)]);
            });
        });

        // Batched Z gates: compare individual vs batched
        group.bench_function(BenchmarkId::new("z_gate_x4_individual", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.z(&[QubitId(10)]);
                sim.z(&[QubitId(11)]);
                sim.z(&[QubitId(12)]);
                sim.z(&[QubitId(13)]);
            });
        });

        group.bench_function(BenchmarkId::new("z_gate_x4_batched", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.z(&[QubitId(10), QubitId(11), QubitId(12), QubitId(13)]);
            });
        });

        // Batched X gates
        group.bench_function(BenchmarkId::new("x_gate_x4_individual", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.x(&[QubitId(10)]);
                sim.x(&[QubitId(11)]);
                sim.x(&[QubitId(12)]);
                sim.x(&[QubitId(13)]);
            });
        });

        group.bench_function(BenchmarkId::new("x_gate_x4_batched", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.x(&[QubitId(10), QubitId(11), QubitId(12), QubitId(13)]);
            });
        });

        // Batched CZ gates
        group.bench_function(BenchmarkId::new("cz_gate_x4_individual", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cz(&[QubitId(10), QubitId(11)]);
                sim.cz(&[QubitId(12), QubitId(13)]);
                sim.cz(&[QubitId(14), QubitId(15)]);
                sim.cz(&[QubitId(10), QubitId(12)]);
            });
        });

        group.bench_function(BenchmarkId::new("cz_gate_x4_batched", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cz(&[
                    QubitId(10), QubitId(11),
                    QubitId(12), QubitId(13),
                    QubitId(14), QubitId(15),
                    QubitId(10), QubitId(12),
                ]);
            });
        });

        // Batched CX gates
        group.bench_function(BenchmarkId::new("cx_gate_x4_individual", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[QubitId(10), QubitId(11)]);
                sim.cx(&[QubitId(12), QubitId(13)]);
                sim.cx(&[QubitId(14), QubitId(15)]);
                sim.cx(&[QubitId(10), QubitId(12)]);
            });
        });

        group.bench_function(BenchmarkId::new("cx_gate_x4_batched", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[
                    QubitId(10), QubitId(11),
                    QubitId(12), QubitId(13),
                    QubitId(14), QubitId(15),
                    QubitId(10), QubitId(12),
                ]);
            });
        });

        // Batched measurement (measure then restore with H gates)
        group.bench_function(BenchmarkId::new("mz_x4_individual", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVec::new(16);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim
                },
                |mut sim| {
                    sim.mz(&[QubitId(0)]);
                    sim.mz(&[QubitId(1)]);
                    sim.mz(&[QubitId(2)]);
                    sim.mz(&[QubitId(3)]);
                },
                criterion::BatchSize::SmallInput,
            );
        });

        group.bench_function(BenchmarkId::new("mz_x4_batched", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVec::new(16);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim
                },
                |mut sim| {
                    sim.mz(&[QubitId(0), QubitId(1), QubitId(2), QubitId(3)]);
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // Batched H gates (H doubles amplitude count, so use H^2=I pattern)
        group.bench_function(BenchmarkId::new("h_gate_x2_individual", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                // Apply H to 2 qubits, then reverse to restore state
                sim.h(&[QubitId(10)]);
                sim.h(&[QubitId(11)]);
                sim.h(&[QubitId(10)]);
                sim.h(&[QubitId(11)]);
            });
        });

        group.bench_function(BenchmarkId::new("h_gate_x2_batched", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                // Apply H to 2 qubits batched, then reverse
                sim.h(&[QubitId(10), QubitId(11)]);
                sim.h(&[QubitId(10), QubitId(11)]);
            });
        });
    }

    group.finish();
}

/// Benchmark comparing original sparse (AoS) vs optimized SoA version
fn bench_sparse_aos_vs_soa(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_aos_vs_soa");

    for h_qubits in [0, 4, 8, 10] {
        let label = format!("{}amps", 1usize << h_qubits);

        // H gate - AoS version (original)
        group.bench_function(BenchmarkId::new("h_aos", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.h(&[QubitId(h_qubits)]);
                sim.h(&[QubitId(h_qubits)]);
            });
        });

        // H gate - SoA version (optimized)
        group.bench_function(BenchmarkId::new("h_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.h(&[QubitId(h_qubits)]);
                sim.h(&[QubitId(h_qubits)]);
            });
        });

        // X gate - AoS
        group.bench_function(BenchmarkId::new("x_aos", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.x(&[QubitId(0)]);
            });
        });

        // X gate - SoA
        group.bench_function(BenchmarkId::new("x_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.x(&[QubitId(0)]);
            });
        });

        // CX gate - AoS
        group.bench_function(BenchmarkId::new("cx_aos", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[QubitId(0), QubitId(1)]);
            });
        });

        // CX gate - SoA
        group.bench_function(BenchmarkId::new("cx_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[QubitId(0), QubitId(1)]);
            });
        });

        // Z gate - AoS (in-place operation)
        group.bench_function(BenchmarkId::new("z_aos", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.z(&[QubitId(0)]);
            });
        });

        // Z gate - SoA (SIMD optimized)
        group.bench_function(BenchmarkId::new("z_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.z(&[QubitId(0)]);
            });
        });

        // CZ gate - AoS (in-place operation)
        group.bench_function(BenchmarkId::new("cz_aos", &label), |b| {
            let mut sim = SparseStateVec::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cz(&[QubitId(0), QubitId(1)]);
            });
        });

        // CZ gate - SoA
        group.bench_function(BenchmarkId::new("cz_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cz(&[QubitId(0), QubitId(1)]);
            });
        });
    }

    group.finish();
}

/// Benchmark realistic circuits that stay sparse
fn bench_realistic_circuits(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_realistic");

    // GHZ state preparation: H on q0, then CX chain
    // Stays at 2 amplitudes throughout
    for num_qubits in [10, 20, 30, 50] {
        group.bench_function(BenchmarkId::new("ghz_aos", num_qubits), |b| {
            let mut sim = SparseStateVec::new(num_qubits);
            b.iter(|| {
                sim.reset();
                sim.h(&[QubitId(0)]);
                for q in 0..num_qubits - 1 {
                    sim.cx(&[QubitId(q), QubitId(q + 1)]);
                }
            });
        });

        group.bench_function(BenchmarkId::new("ghz_soa", num_qubits), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            b.iter(|| {
                sim.reset();
                sim.h(&[QubitId(0)]);
                for q in 0..num_qubits - 1 {
                    sim.cx(&[QubitId(q), QubitId(q + 1)]);
                }
            });
        });
    }

    // Random Clifford on sparse state: X, Z, CX, CZ gates
    // These keep the state sparse
    for num_qubits in [10, 20, 30] {
        let gates_per_iter = 100;

        group.bench_function(BenchmarkId::new("clifford_sparse_aos", num_qubits), |b| {
            let mut sim = SparseStateVec::new(num_qubits);
            b.iter(|| {
                for i in 0..gates_per_iter {
                    let q = i % num_qubits;
                    let q2 = (i + 1) % num_qubits;
                    match i % 4 {
                        0 => { sim.x(&[QubitId(q)]); }
                        1 => { sim.z(&[QubitId(q)]); }
                        2 => { sim.cx(&[QubitId(q), QubitId(q2)]); }
                        _ => { sim.cz(&[QubitId(q), QubitId(q2)]); }
                    }
                }
            });
        });

        group.bench_function(BenchmarkId::new("clifford_sparse_soa", num_qubits), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            b.iter(|| {
                for i in 0..gates_per_iter {
                    let q = i % num_qubits;
                    let q2 = (i + 1) % num_qubits;
                    match i % 4 {
                        0 => { sim.x(&[QubitId(q)]); }
                        1 => { sim.z(&[QubitId(q)]); }
                        2 => { sim.cx(&[QubitId(q), QubitId(q2)]); }
                        _ => { sim.cz(&[QubitId(q), QubitId(q2)]); }
                    }
                }
            });
        });
    }

    // Incremental superposition: start sparse, progressively add H gates
    // Tests performance as state grows from 1 to 2^n amplitudes
    for final_h_count in [4, 6, 8] {
        let num_qubits = 16;

        group.bench_function(BenchmarkId::new("incremental_h_aos", final_h_count), |b| {
            let mut sim = SparseStateVec::new(num_qubits);
            b.iter(|| {
                sim.reset();
                for q in 0..final_h_count {
                    sim.h(&[QubitId(q)]);
                    // Intersperse with CZ gates
                    if q > 0 {
                        sim.cz(&[QubitId(q - 1), QubitId(q)]);
                    }
                }
            });
        });

        group.bench_function(BenchmarkId::new("incremental_h_soa", final_h_count), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            b.iter(|| {
                sim.reset();
                for q in 0..final_h_count {
                    sim.h(&[QubitId(q)]);
                    if q > 0 {
                        sim.cz(&[QubitId(q - 1), QubitId(q)]);
                    }
                }
            });
        });
    }

    group.finish();
}

pub fn benchmarks(c: &mut Criterion) {
    bench_sparse_friendly(c);
    bench_varying_superposition(c);
    bench_sparse_operations(c);
    bench_sparse_aos_vs_soa(c);
    bench_realistic_circuits(c);
}
