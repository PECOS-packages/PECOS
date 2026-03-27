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
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, SparseStateVecSoA, StateVecSoA,
};

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
                let mut sim = SparseStateVecSoA::new(n);
                b.iter(|| {
                    for q in 0..n {
                        sim.x(&[QubitId(q)]);
                        sim.z(&[QubitId(q)]);
                    }
                    for q in 0..n - 1 {
                        sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                    }
                });
            },
        );

        // Sparse SoA version
        group.bench_with_input(
            BenchmarkId::new("sparse_soa", num_qubits),
            &num_qubits,
            |b, &n| {
                let mut sim = SparseStateVecSoA::new(n);
                b.iter(|| {
                    for q in 0..n {
                        sim.x(&[QubitId(q)]);
                        sim.z(&[QubitId(q)]);
                    }
                    for q in 0..n - 1 {
                        sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                    }
                });
            },
        );

        // Dense version
        group.bench_with_input(
            BenchmarkId::new("dense", num_qubits),
            &num_qubits,
            |b, &n| {
                let mut sim = StateVecSoA::new(n);
                b.iter(|| {
                    for q in 0..n {
                        sim.x(&[QubitId(q)]);
                        sim.z(&[QubitId(q)]);
                    }
                    for q in 0..n - 1 {
                        sim.cx(&[(QubitId(q), QubitId(q + 1))]);
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
        group.bench_with_input(BenchmarkId::new("sparse", h_qubits), &h_qubits, |b, &h| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            b.iter(|| {
                sim.reset();
                for q in 0..h {
                    sim.h(&[QubitId(q)]);
                }
            });
        });

        // Sparse SoA version
        group.bench_with_input(
            BenchmarkId::new("sparse_soa", h_qubits),
            &h_qubits,
            |b, &h| {
                let mut sim = SparseStateVecSoA::new(num_qubits);
                b.iter(|| {
                    sim.reset();
                    for q in 0..h {
                        sim.h(&[QubitId(q)]);
                    }
                    sim.flush_all_frames();
                });
            },
        );

        // Dense version
        group.bench_with_input(BenchmarkId::new("dense", h_qubits), &h_qubits, |b, &h| {
            let mut sim = StateVecSoA::new(num_qubits);
            b.iter(|| {
                sim.reset();
                for q in 0..h {
                    sim.h(&[QubitId(q)]);
                }
            });
        });
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
            let mut sim = SparseStateVecSoA::new(16);
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
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.x(&[QubitId(0)]);
            });
        });

        // Z gate (in-place phase flip)
        group.bench_function(BenchmarkId::new("z_gate", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.z(&[QubitId(0)]);
            });
        });

        // CX gate
        group.bench_function(BenchmarkId::new("cx_gate", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[(QubitId(0), QubitId(1))]);
            });
        });

        // CZ gate (in-place)
        group.bench_function(BenchmarkId::new("cz_gate", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cz(&[(QubitId(0), QubitId(1))]);
            });
        });

        // Batched Z gates: compare individual vs batched
        group.bench_function(BenchmarkId::new("z_gate_x4_individual", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
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
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.z(&[QubitId(10), QubitId(11), QubitId(12), QubitId(13)]);
            });
        });

        // Batched X gates
        group.bench_function(BenchmarkId::new("x_gate_x4_individual", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
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
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.x(&[QubitId(10), QubitId(11), QubitId(12), QubitId(13)]);
            });
        });

        // Batched CZ gates
        group.bench_function(BenchmarkId::new("cz_gate_x4_individual", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cz(&[(QubitId(10), QubitId(11))]);
                sim.cz(&[(QubitId(12), QubitId(13))]);
                sim.cz(&[(QubitId(14), QubitId(15))]);
                sim.cz(&[(QubitId(10), QubitId(12))]);
            });
        });

        group.bench_function(BenchmarkId::new("cz_gate_x4_batched", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cz(&[
                    (QubitId(10), QubitId(11)),
                    (QubitId(12), QubitId(13)),
                    (QubitId(14), QubitId(15)),
                    (QubitId(10), QubitId(12)),
                ]);
            });
        });

        // Batched CX gates
        group.bench_function(BenchmarkId::new("cx_gate_x4_individual", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[(QubitId(10), QubitId(11))]);
                sim.cx(&[(QubitId(12), QubitId(13))]);
                sim.cx(&[(QubitId(14), QubitId(15))]);
                sim.cx(&[(QubitId(10), QubitId(12))]);
            });
        });

        group.bench_function(BenchmarkId::new("cx_gate_x4_batched", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[
                    (QubitId(10), QubitId(11)),
                    (QubitId(12), QubitId(13)),
                    (QubitId(14), QubitId(15)),
                    (QubitId(10), QubitId(12)),
                ]);
            });
        });

        // Batched CX gates - SoA
        group.bench_function(BenchmarkId::new("cx_gate_x4_individual_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[(QubitId(10), QubitId(11))]);
                sim.cx(&[(QubitId(12), QubitId(13))]);
                sim.cx(&[(QubitId(14), QubitId(15))]);
                sim.cx(&[(QubitId(8), QubitId(9))]);
            });
        });

        group.bench_function(BenchmarkId::new("cx_gate_x4_batched_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[
                    (QubitId(10), QubitId(11)),
                    (QubitId(12), QubitId(13)),
                    (QubitId(14), QubitId(15)),
                    (QubitId(8), QubitId(9)),
                ]);
            });
        });

        // Batched CZ gates - SoA
        group.bench_function(BenchmarkId::new("cz_gate_x4_individual_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cz(&[(QubitId(10), QubitId(11))]);
                sim.cz(&[(QubitId(12), QubitId(13))]);
                sim.cz(&[(QubitId(14), QubitId(15))]);
                sim.cz(&[(QubitId(8), QubitId(9))]);
            });
        });

        group.bench_function(BenchmarkId::new("cz_gate_x4_batched_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cz(&[
                    (QubitId(10), QubitId(11)),
                    (QubitId(12), QubitId(13)),
                    (QubitId(14), QubitId(15)),
                    (QubitId(8), QubitId(9)),
                ]);
            });
        });

        // SWAP x4 individual (SoA) with frames (physical state stays small)
        group.bench_function(
            BenchmarkId::new("swap_gate_x4_individual_soa", &label),
            |b| {
                let mut sim = SparseStateVecSoA::new(16);
                for q in 0..h_qubits {
                    sim.h(&[QubitId(q)]);
                }
                b.iter(|| {
                    sim.swap(&[(QubitId(10), QubitId(11))]);
                    sim.swap(&[(QubitId(12), QubitId(13))]);
                    sim.swap(&[(QubitId(14), QubitId(15))]);
                    sim.swap(&[(QubitId(8), QubitId(9))]);
                });
            },
        );

        // SWAP x4 batched (SoA) with frames (physical state stays small)
        group.bench_function(BenchmarkId::new("swap_gate_x4_batched_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.swap(&[
                    (QubitId(10), QubitId(11)),
                    (QubitId(12), QubitId(13)),
                    (QubitId(14), QubitId(15)),
                    (QubitId(8), QubitId(9)),
                ]);
            });
        });

        // SWAP x4 individual (SoA) with flushed frames (expanded physical state)
        group.bench_function(
            BenchmarkId::new("swap_x4_individual_soa_flushed", &label),
            |b| {
                let mut sim = SparseStateVecSoA::new(16);
                for q in 0..h_qubits {
                    sim.h(&[QubitId(q)]);
                }
                sim.flush_all_frames();
                b.iter(|| {
                    sim.swap(&[(QubitId(0), QubitId(1))]);
                    sim.swap(&[(QubitId(2), QubitId(3))]);
                    sim.swap(&[(QubitId(4), QubitId(5))]);
                    sim.swap(&[(QubitId(6), QubitId(7))]);
                });
            },
        );

        // SWAP x4 batched (SoA) with flushed frames (expanded physical state)
        group.bench_function(
            BenchmarkId::new("swap_x4_batched_soa_flushed", &label),
            |b| {
                let mut sim = SparseStateVecSoA::new(16);
                for q in 0..h_qubits {
                    sim.h(&[QubitId(q)]);
                }
                sim.flush_all_frames();
                b.iter(|| {
                    sim.swap(&[
                        (QubitId(0), QubitId(1)),
                        (QubitId(2), QubitId(3)),
                        (QubitId(4), QubitId(5)),
                        (QubitId(6), QubitId(7)),
                    ]);
                });
            },
        );

        // SZZ x4 individual (SoA) with frames
        group.bench_function(BenchmarkId::new("szz_x4_individual_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.szz(&[(QubitId(10), QubitId(11))]);
                sim.szz(&[(QubitId(12), QubitId(13))]);
                sim.szz(&[(QubitId(14), QubitId(15))]);
                sim.szz(&[(QubitId(8), QubitId(9))]);
            });
        });

        // SZZ x4 batched (SoA) with frames
        group.bench_function(BenchmarkId::new("szz_x4_batched_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.szz(&[
                    (QubitId(10), QubitId(11)),
                    (QubitId(12), QubitId(13)),
                    (QubitId(14), QubitId(15)),
                    (QubitId(8), QubitId(9)),
                ]);
            });
        });

        // SZZ x4 individual (SoA) with flushed frames
        group.bench_function(
            BenchmarkId::new("szz_x4_individual_soa_flushed", &label),
            |b| {
                let mut sim = SparseStateVecSoA::new(16);
                for q in 0..h_qubits {
                    sim.h(&[QubitId(q)]);
                }
                sim.flush_all_frames();
                b.iter(|| {
                    sim.szz(&[(QubitId(0), QubitId(1))]);
                    sim.szz(&[(QubitId(2), QubitId(3))]);
                    sim.szz(&[(QubitId(4), QubitId(5))]);
                    sim.szz(&[(QubitId(6), QubitId(7))]);
                });
            },
        );

        // SZZ x4 batched (SoA) with flushed frames
        group.bench_function(
            BenchmarkId::new("szz_x4_batched_soa_flushed", &label),
            |b| {
                let mut sim = SparseStateVecSoA::new(16);
                for q in 0..h_qubits {
                    sim.h(&[QubitId(q)]);
                }
                sim.flush_all_frames();
                b.iter(|| {
                    sim.szz(&[
                        (QubitId(0), QubitId(1)),
                        (QubitId(2), QubitId(3)),
                        (QubitId(4), QubitId(5)),
                        (QubitId(6), QubitId(7)),
                    ]);
                });
            },
        );

        // iSWAP x4 individual (SoA) with frames
        group.bench_function(BenchmarkId::new("iswap_x4_individual_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.iswap(&[(QubitId(10), QubitId(11))]);
                sim.iswap(&[(QubitId(12), QubitId(13))]);
                sim.iswap(&[(QubitId(14), QubitId(15))]);
                sim.iswap(&[(QubitId(8), QubitId(9))]);
            });
        });

        // iSWAP x4 batched (SoA) with frames
        group.bench_function(BenchmarkId::new("iswap_x4_batched_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.iswap(&[
                    (QubitId(10), QubitId(11)),
                    (QubitId(12), QubitId(13)),
                    (QubitId(14), QubitId(15)),
                    (QubitId(8), QubitId(9)),
                ]);
            });
        });

        // iSWAP x4 individual (SoA) with flushed frames
        group.bench_function(
            BenchmarkId::new("iswap_x4_individual_soa_flushed", &label),
            |b| {
                let mut sim = SparseStateVecSoA::new(16);
                for q in 0..h_qubits {
                    sim.h(&[QubitId(q)]);
                }
                sim.flush_all_frames();
                b.iter(|| {
                    sim.iswap(&[(QubitId(0), QubitId(1))]);
                    sim.iswap(&[(QubitId(2), QubitId(3))]);
                    sim.iswap(&[(QubitId(4), QubitId(5))]);
                    sim.iswap(&[(QubitId(6), QubitId(7))]);
                });
            },
        );

        // iSWAP x4 batched (SoA) with flushed frames
        group.bench_function(
            BenchmarkId::new("iswap_x4_batched_soa_flushed", &label),
            |b| {
                let mut sim = SparseStateVecSoA::new(16);
                for q in 0..h_qubits {
                    sim.h(&[QubitId(q)]);
                }
                sim.flush_all_frames();
                b.iter(|| {
                    sim.iswap(&[
                        (QubitId(0), QubitId(1)),
                        (QubitId(2), QubitId(3)),
                        (QubitId(4), QubitId(5)),
                        (QubitId(6), QubitId(7)),
                    ]);
                });
            },
        );

        // Batched measurement (measure then restore with H gates)
        group.bench_function(BenchmarkId::new("mz_x4_individual", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVecSoA::new(16);
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
                    let mut sim = SparseStateVecSoA::new(16);
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
            let mut sim = SparseStateVecSoA::new(16);
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
            let mut sim = SparseStateVecSoA::new(16);
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

/// Benchmark comparing original sparse (`AoS`) vs optimized `SoA` version
fn bench_sparse_aos_vs_soa(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_aos_vs_soa");

    for h_qubits in [0, 4, 8, 10] {
        let label = format!("{}amps", 1usize << h_qubits);

        // H gate - AoS version (original)
        group.bench_function(BenchmarkId::new("h_aos", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
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
            let mut sim = SparseStateVecSoA::new(16);
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
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[(QubitId(0), QubitId(1))]);
            });
        });

        // CX gate - SoA
        group.bench_function(BenchmarkId::new("cx_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[(QubitId(0), QubitId(1))]);
            });
        });

        // Z gate - AoS (in-place operation)
        group.bench_function(BenchmarkId::new("z_aos", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
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
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cz(&[(QubitId(0), QubitId(1))]);
            });
        });

        // CZ gate - SoA
        group.bench_function(BenchmarkId::new("cz_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cz(&[(QubitId(0), QubitId(1))]);
            });
        });

        // SZZ gate - AoS (trait default: H.H.SXX.H.H decomposition)
        group.bench_function(BenchmarkId::new("szz_aos", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.szz(&[(QubitId(0), QubitId(1))]);
            });
        });

        // SZZ gate - SoA (direct diagonal + push-through, flushed for fair comparison)
        group.bench_function(BenchmarkId::new("szz_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.szz(&[(QubitId(0), QubitId(1))]);
            });
        });

        // SZZdg gate - AoS (trait default: Z.Z.SZZ decomposition)
        group.bench_function(BenchmarkId::new("szzdg_aos", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.szzdg(&[(QubitId(0), QubitId(1))]);
            });
        });

        // SZZdg gate - SoA (direct diagonal + push-through, flushed)
        group.bench_function(BenchmarkId::new("szzdg_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.szzdg(&[(QubitId(0), QubitId(1))]);
            });
        });

        // iSWAP gate - AoS (trait default: SZ.SZ.H.CX.CX.H decomposition)
        group.bench_function(BenchmarkId::new("iswap_aos", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.iswap(&[(QubitId(0), QubitId(1))]);
            });
        });

        // iSWAP gate - SoA (direct bit-swap + push-through, flushed)
        group.bench_function(BenchmarkId::new("iswap_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.iswap(&[(QubitId(0), QubitId(1))]);
            });
        });

        // SXX gate - AoS (trait default decomposition)
        group.bench_function(BenchmarkId::new("sxx_aos", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.sxx(&[(QubitId(0), QubitId(1))]);
            });
        });

        // SXX gate - SoA (trait default decomposition, flushed)
        group.bench_function(BenchmarkId::new("sxx_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.sxx(&[(QubitId(0), QubitId(1))]);
            });
        });

        // G gate - AoS (trait default)
        group.bench_function(BenchmarkId::new("g_aos", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.g(&[(QubitId(0), QubitId(1))]);
            });
        });

        // G gate - SoA (trait default, flushed)
        group.bench_function(BenchmarkId::new("g_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(16);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.g(&[(QubitId(0), QubitId(1))]);
            });
        });
    }

    group.finish();
}

/// Benchmark realistic circuits comparing all three simulators
fn bench_realistic_circuits(c: &mut Criterion) {
    let mut group = c.benchmark_group("sparse_realistic");

    // GHZ state preparation: H on q0, then CX chain
    // Stays at 2 amplitudes throughout (sparse-friendly)
    for num_qubits in [10, 20, 30, 50] {
        group.bench_function(BenchmarkId::new("ghz_aos", num_qubits), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            b.iter(|| {
                sim.reset();
                sim.h(&[QubitId(0)]);
                for q in 0..num_qubits - 1 {
                    sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                }
            });
        });

        group.bench_function(BenchmarkId::new("ghz_soa", num_qubits), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            b.iter(|| {
                sim.reset();
                sim.h(&[QubitId(0)]);
                for q in 0..num_qubits - 1 {
                    sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                }
            });
        });

        // Dense only feasible for <= 20 qubits (2^20 = 1M amplitudes)
        if num_qubits <= 20 {
            group.bench_function(BenchmarkId::new("ghz_dense", num_qubits), |b| {
                let mut sim = StateVecSoA::new(num_qubits);
                b.iter(|| {
                    sim.reset();
                    sim.h(&[QubitId(0)]);
                    for q in 0..num_qubits - 1 {
                        sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                    }
                });
            });
        }
    }

    // Random Clifford on sparse state: X, Z, CX, CZ gates
    // These keep the state sparse (1 amplitude throughout)
    for num_qubits in [10, 20, 30] {
        let gates_per_iter = 100;

        group.bench_function(BenchmarkId::new("clifford_sparse_aos", num_qubits), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            b.iter(|| {
                for i in 0..gates_per_iter {
                    let q = i % num_qubits;
                    let q2 = (i + 1) % num_qubits;
                    match i % 4 {
                        0 => {
                            sim.x(&[QubitId(q)]);
                        }
                        1 => {
                            sim.z(&[QubitId(q)]);
                        }
                        2 => {
                            sim.cx(&[(QubitId(q), QubitId(q2))]);
                        }
                        _ => {
                            sim.cz(&[(QubitId(q), QubitId(q2))]);
                        }
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
                        0 => {
                            sim.x(&[QubitId(q)]);
                        }
                        1 => {
                            sim.z(&[QubitId(q)]);
                        }
                        2 => {
                            sim.cx(&[(QubitId(q), QubitId(q2))]);
                        }
                        _ => {
                            sim.cz(&[(QubitId(q), QubitId(q2))]);
                        }
                    }
                }
            });
        });

        // Dense only feasible for <= 20 qubits
        if num_qubits <= 20 {
            group.bench_function(BenchmarkId::new("clifford_sparse_dense", num_qubits), |b| {
                let mut sim = StateVecSoA::new(num_qubits);
                b.iter(|| {
                    for i in 0..gates_per_iter {
                        let q = i % num_qubits;
                        let q2 = (i + 1) % num_qubits;
                        match i % 4 {
                            0 => {
                                sim.x(&[QubitId(q)]);
                            }
                            1 => {
                                sim.z(&[QubitId(q)]);
                            }
                            2 => {
                                sim.cx(&[(QubitId(q), QubitId(q2))]);
                            }
                            _ => {
                                sim.cz(&[(QubitId(q), QubitId(q2))]);
                            }
                        }
                    }
                });
            });
        }
    }

    // Incremental superposition: start sparse, progressively add H gates
    // Tests performance as state grows from 1 to 2^n amplitudes
    for final_h_count in [4, 6, 8] {
        let num_qubits = 16;

        group.bench_function(BenchmarkId::new("incremental_h_aos", final_h_count), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            b.iter(|| {
                sim.reset();
                for q in 0..final_h_count {
                    sim.h(&[QubitId(q)]);
                    // Intersperse with CZ gates
                    if q > 0 {
                        sim.cz(&[(QubitId(q - 1), QubitId(q))]);
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
                        sim.cz(&[(QubitId(q - 1), QubitId(q))]);
                    }
                }
            });
        });

        group.bench_function(
            BenchmarkId::new("incremental_h_dense", final_h_count),
            |b| {
                let mut sim = StateVecSoA::new(num_qubits);
                b.iter(|| {
                    sim.reset();
                    for q in 0..final_h_count {
                        sim.h(&[QubitId(q)]);
                        if q > 0 {
                            sim.cz(&[(QubitId(q - 1), QubitId(q))]);
                        }
                    }
                });
            },
        );
    }

    group.finish();
}

/// 3-way benchmark comparing Sparse `AoS`, Sparse `SoA`, and Dense `StateVec` on individual gates
fn bench_three_statevecs_gates(c: &mut Criterion) {
    let mut group = c.benchmark_group("three_statevecs");
    let num_qubits = 16;

    for h_qubits in [0usize, 8, 10] {
        let amps = 1usize << h_qubits;
        let label = format!("{amps}amps");

        // --- SZZ ---
        group.bench_function(BenchmarkId::new("szz_aos", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.szz(&[(QubitId(0), QubitId(1))]);
            });
        });

        group.bench_function(BenchmarkId::new("szz_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.szz(&[(QubitId(0), QubitId(1))]);
            });
        });

        group.bench_function(BenchmarkId::new("szz_dense", &label), |b| {
            let mut sim = StateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.szz(&[(QubitId(0), QubitId(1))]);
            });
        });

        // --- SZZdg ---
        group.bench_function(BenchmarkId::new("szzdg_aos", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.szzdg(&[(QubitId(0), QubitId(1))]);
            });
        });

        group.bench_function(BenchmarkId::new("szzdg_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.szzdg(&[(QubitId(0), QubitId(1))]);
            });
        });

        group.bench_function(BenchmarkId::new("szzdg_dense", &label), |b| {
            let mut sim = StateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.szzdg(&[(QubitId(0), QubitId(1))]);
            });
        });

        // --- iSWAP ---
        group.bench_function(BenchmarkId::new("iswap_aos", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.iswap(&[(QubitId(0), QubitId(1))]);
            });
        });

        group.bench_function(BenchmarkId::new("iswap_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.iswap(&[(QubitId(0), QubitId(1))]);
            });
        });

        group.bench_function(BenchmarkId::new("iswap_dense", &label), |b| {
            let mut sim = StateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.iswap(&[(QubitId(0), QubitId(1))]);
            });
        });

        // --- SXX ---
        group.bench_function(BenchmarkId::new("sxx_aos", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.sxx(&[(QubitId(0), QubitId(1))]);
            });
        });

        group.bench_function(BenchmarkId::new("sxx_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.sxx(&[(QubitId(0), QubitId(1))]);
            });
        });

        group.bench_function(BenchmarkId::new("sxx_dense", &label), |b| {
            let mut sim = StateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.sxx(&[(QubitId(0), QubitId(1))]);
            });
        });

        // --- CX ---
        group.bench_function(BenchmarkId::new("cx_aos", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[(QubitId(0), QubitId(1))]);
            });
        });

        group.bench_function(BenchmarkId::new("cx_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.cx(&[(QubitId(0), QubitId(1))]);
            });
        });

        group.bench_function(BenchmarkId::new("cx_dense", &label), |b| {
            let mut sim = StateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.cx(&[(QubitId(0), QubitId(1))]);
            });
        });

        // --- H (apply twice: H^2 = I to avoid state growth) ---
        group.bench_function(BenchmarkId::new("h_aos", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.h(&[QubitId(h_qubits)]);
                sim.h(&[QubitId(h_qubits)]);
            });
        });

        group.bench_function(BenchmarkId::new("h_soa", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.h(&[QubitId(h_qubits)]);
                sim.h(&[QubitId(h_qubits)]);
            });
        });

        group.bench_function(BenchmarkId::new("h_dense", &label), |b| {
            let mut sim = StateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.h(&[QubitId(h_qubits)]);
                sim.h(&[QubitId(h_qubits)]);
            });
        });
    }

    group.finish();
}

/// Benchmark rotation gates (T, RZ, RZZ) on sparse `SoA` with Pauli push-through optimization.
///
/// All sparse cases start with flushed frames to ensure the same physical amplitude count.
/// Compares:
/// - `pauli_frame`: target qubit has Pauli frame (X); RZ pushes through without flushing
/// - `identity_frame`: target qubit has identity frame; RZ applies directly
/// - `non_pauli_frame`: target qubit has non-Pauli frame (H); RZ must flush first
/// - `dense`: dense `StateVecSoA` baseline for reference
fn bench_rotation_pauli_pushthrough(c: &mut Criterion) {
    let mut group = c.benchmark_group("rotation_pauli_pushthrough");
    let num_qubits = 16;

    for h_qubits in [0usize, 4, 8, 10] {
        let amps = 1usize << h_qubits;
        let label = format!("{amps}amps");

        // --- T gate ---

        // T with Pauli frame: flush first to expand amps, then X creates Pauli frame
        group.bench_function(BenchmarkId::new("t_pauli_frame", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            // X composes into frame (O(1)), doesn't change physical amps
            sim.x(&[QubitId(h_qubits)]);
            b.iter(|| {
                sim.t(&[QubitId(h_qubits)]);
                sim.tdg(&[QubitId(h_qubits)]);
            });
        });

        // T on identity frame (flushed, no frame overhead)
        group.bench_function(BenchmarkId::new("t_identity_frame", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.t(&[QubitId(h_qubits)]);
                sim.tdg(&[QubitId(h_qubits)]);
            });
        });

        // T needing flush: fresh state each iteration since H flush doubles amp count
        group.bench_function(BenchmarkId::new("t_non_pauli_frame", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim.flush_all_frames();
                    sim.h(&[QubitId(h_qubits)]); // H frame (non-Pauli)
                    sim
                },
                |mut sim| {
                    sim.t(&[QubitId(h_qubits)]); // must flush H, then apply T
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // T on dense baseline
        group.bench_function(BenchmarkId::new("t_dense", &label), |b| {
            let mut sim = StateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.t(&[QubitId(h_qubits)]);
                sim.tdg(&[QubitId(h_qubits)]);
            });
        });

        // --- RZ gate ---

        // RZ with Pauli frame
        group.bench_function(BenchmarkId::new("rz_pauli_frame", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            sim.x(&[QubitId(h_qubits)]);
            b.iter(|| {
                sim.rz(Angle64::from_radians(0.123), &[QubitId(h_qubits)]);
                sim.rz(Angle64::from_radians(-0.123), &[QubitId(h_qubits)]);
            });
        });

        // RZ on identity frame
        group.bench_function(BenchmarkId::new("rz_identity_frame", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.rz(Angle64::from_radians(0.123), &[QubitId(h_qubits)]);
                sim.rz(Angle64::from_radians(-0.123), &[QubitId(h_qubits)]);
            });
        });

        // RZ needing flush
        group.bench_function(BenchmarkId::new("rz_non_pauli_frame", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim.flush_all_frames();
                    sim.h(&[QubitId(h_qubits)]);
                    sim
                },
                |mut sim| {
                    sim.rz(Angle64::from_radians(0.123), &[QubitId(h_qubits)]);
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // RZ on dense
        group.bench_function(BenchmarkId::new("rz_dense", &label), |b| {
            let mut sim = StateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.rz(Angle64::from_radians(0.123), &[QubitId(h_qubits)]);
                sim.rz(Angle64::from_radians(-0.123), &[QubitId(h_qubits)]);
            });
        });

        // --- RZZ gate ---

        // RZZ with both Pauli frames
        group.bench_function(BenchmarkId::new("rzz_both_pauli", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            sim.x(&[QubitId(h_qubits)]);
            sim.x(&[QubitId(h_qubits + 1)]);
            b.iter(|| {
                sim.rzz(
                    Angle64::from_radians(0.123),
                    &[(QubitId(h_qubits), QubitId(h_qubits + 1))],
                );
                sim.rzz(
                    Angle64::from_radians(-0.123),
                    &[(QubitId(h_qubits), QubitId(h_qubits + 1))],
                );
            });
        });

        // RZZ on identity frames
        group.bench_function(BenchmarkId::new("rzz_identity_frame", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            sim.flush_all_frames();
            b.iter(|| {
                sim.rzz(
                    Angle64::from_radians(0.123),
                    &[(QubitId(h_qubits), QubitId(h_qubits + 1))],
                );
                sim.rzz(
                    Angle64::from_radians(-0.123),
                    &[(QubitId(h_qubits), QubitId(h_qubits + 1))],
                );
            });
        });

        // RZZ needing flush on both qubits
        group.bench_function(BenchmarkId::new("rzz_both_non_pauli", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim.flush_all_frames();
                    sim.h(&[QubitId(h_qubits)]);
                    sim.h(&[QubitId(h_qubits + 1)]);
                    sim
                },
                |mut sim| {
                    sim.rzz(
                        Angle64::from_radians(0.123),
                        &[(QubitId(h_qubits), QubitId(h_qubits + 1))],
                    );
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // RZZ on dense
        group.bench_function(BenchmarkId::new("rzz_dense", &label), |b| {
            let mut sim = StateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                sim.rzz(
                    Angle64::from_radians(0.123),
                    &[(QubitId(h_qubits), QubitId(h_qubits + 1))],
                );
                sim.rzz(
                    Angle64::from_radians(-0.123),
                    &[(QubitId(h_qubits), QubitId(h_qubits + 1))],
                );
            });
        });
    }

    group.finish();
}

/// Benchmark non-diagonal rotation gates: RX, RY, RXX, RYY.
/// Tests Pauli push-through and direct implementations vs decomposition baseline.
fn bench_nondiag_rotation_pushthrough(c: &mut Criterion) {
    let mut group = c.benchmark_group("nondiag_rotation_pushthrough");
    let num_qubits = 16;

    for h_qubits in [0usize, 4, 8, 10] {
        let amps = 1usize << h_qubits;
        let label = format!("{amps}amps");

        // --- RX gate ---

        // RX with Pauli frame (Z frame -> push through)
        group.bench_function(BenchmarkId::new("rx_pauli_frame", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim.flush_all_frames();
                    sim.z(&[QubitId(h_qubits)]); // Z frame (Pauli)
                    sim
                },
                |mut sim| {
                    sim.rx(Angle64::from_radians(0.123), &[QubitId(h_qubits)]);
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // RX needing flush (H frame, non-Pauli)
        group.bench_function(BenchmarkId::new("rx_non_pauli_frame", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim.flush_all_frames();
                    sim.h(&[QubitId(h_qubits)]); // H frame (non-Pauli)
                    sim
                },
                |mut sim| {
                    sim.rx(Angle64::from_radians(0.123), &[QubitId(h_qubits)]);
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // RX on dense
        group.bench_function(BenchmarkId::new("rx_dense", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = StateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim
                },
                |mut sim| {
                    sim.rx(Angle64::from_radians(0.123), &[QubitId(h_qubits)]);
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // --- RY gate ---

        // RY with Pauli frame (X frame -> push through)
        group.bench_function(BenchmarkId::new("ry_pauli_frame", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim.flush_all_frames();
                    sim.x(&[QubitId(h_qubits)]); // X frame (Pauli)
                    sim
                },
                |mut sim| {
                    sim.ry(Angle64::from_radians(0.123), &[QubitId(h_qubits)]);
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // RY needing flush
        group.bench_function(BenchmarkId::new("ry_non_pauli_frame", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim.flush_all_frames();
                    sim.h(&[QubitId(h_qubits)]);
                    sim
                },
                |mut sim| {
                    sim.ry(Angle64::from_radians(0.123), &[QubitId(h_qubits)]);
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // RY on dense
        group.bench_function(BenchmarkId::new("ry_dense", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = StateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim
                },
                |mut sim| {
                    sim.ry(Angle64::from_radians(0.123), &[QubitId(h_qubits)]);
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // --- RXX gate ---

        // RXX with both Pauli frames (Z frames -> push through)
        group.bench_function(BenchmarkId::new("rxx_pauli_frame", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim.flush_all_frames();
                    sim.z(&[QubitId(h_qubits)]);
                    sim.z(&[QubitId(h_qubits + 1)]);
                    sim
                },
                |mut sim| {
                    sim.rxx(
                        Angle64::from_radians(0.123),
                        &[(QubitId(h_qubits), QubitId(h_qubits + 1))],
                    );
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // RXX needing flush
        group.bench_function(BenchmarkId::new("rxx_non_pauli_frame", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim.flush_all_frames();
                    sim.h(&[QubitId(h_qubits)]);
                    sim.h(&[QubitId(h_qubits + 1)]);
                    sim
                },
                |mut sim| {
                    sim.rxx(
                        Angle64::from_radians(0.123),
                        &[(QubitId(h_qubits), QubitId(h_qubits + 1))],
                    );
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // RXX on dense
        group.bench_function(BenchmarkId::new("rxx_dense", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = StateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim
                },
                |mut sim| {
                    sim.rxx(
                        Angle64::from_radians(0.123),
                        &[(QubitId(h_qubits), QubitId(h_qubits + 1))],
                    );
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // --- RYY gate ---

        // RYY with both Pauli frames (X frames -> push through since has_x != has_z)
        group.bench_function(BenchmarkId::new("ryy_pauli_frame", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim.flush_all_frames();
                    sim.x(&[QubitId(h_qubits)]);
                    sim.x(&[QubitId(h_qubits + 1)]);
                    sim
                },
                |mut sim| {
                    sim.ryy(
                        Angle64::from_radians(0.123),
                        &[(QubitId(h_qubits), QubitId(h_qubits + 1))],
                    );
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // RYY needing flush
        group.bench_function(BenchmarkId::new("ryy_non_pauli_frame", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = SparseStateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim.flush_all_frames();
                    sim.h(&[QubitId(h_qubits)]);
                    sim.h(&[QubitId(h_qubits + 1)]);
                    sim
                },
                |mut sim| {
                    sim.ryy(
                        Angle64::from_radians(0.123),
                        &[(QubitId(h_qubits), QubitId(h_qubits + 1))],
                    );
                },
                criterion::BatchSize::SmallInput,
            );
        });

        // RYY on dense
        group.bench_function(BenchmarkId::new("ryy_dense", &label), |b| {
            b.iter_batched(
                || {
                    let mut sim = StateVecSoA::new(num_qubits);
                    for q in 0..h_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    sim
                },
                |mut sim| {
                    sim.ryy(
                        Angle64::from_radians(0.123),
                        &[(QubitId(h_qubits), QubitId(h_qubits + 1))],
                    );
                },
                criterion::BatchSize::SmallInput,
            );
        });
    }

    group.finish();
}

/// Benchmark a QEC-like circuit with Cliffords interspersed with T/RZ gates.
/// This represents the realistic use case where Pauli push-through matters.
fn bench_qec_rotation_circuit(c: &mut Criterion) {
    let mut group = c.benchmark_group("qec_rotation_circuit");
    let num_qubits = 16;
    let rounds = 10;

    for h_qubits in [0usize, 4, 8] {
        let amps = 1usize << h_qubits;
        let label = format!("{amps}amps");

        // Clifford + T circuit on sparse SoA
        // Pattern: X/Z gates (create Pauli frames) then T gates (push through)
        group.bench_function(BenchmarkId::new("clifford_t_sparse", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                for _ in 0..rounds {
                    // Clifford layer: X, Z, CX create Pauli frames
                    for q in (0..num_qubits).step_by(2) {
                        sim.x(&[QubitId(q)]);
                        sim.z(&[QubitId(q + 1)]);
                    }
                    for q in (0..num_qubits - 1).step_by(2) {
                        sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                    }
                    // T layer: should push through Pauli frames
                    for q in 0..num_qubits {
                        sim.t(&[QubitId(q)]);
                    }
                }
            });
        });

        // Same circuit on dense SoA
        group.bench_function(BenchmarkId::new("clifford_t_dense", &label), |b| {
            let mut sim = StateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                for _ in 0..rounds {
                    for q in (0..num_qubits).step_by(2) {
                        sim.x(&[QubitId(q)]);
                        sim.z(&[QubitId(q + 1)]);
                    }
                    for q in (0..num_qubits - 1).step_by(2) {
                        sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                    }
                    for q in 0..num_qubits {
                        sim.t(&[QubitId(q)]);
                    }
                }
            });
        });

        // Clifford + RZZ circuit on sparse SoA
        group.bench_function(BenchmarkId::new("clifford_rzz_sparse", &label), |b| {
            let mut sim = SparseStateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                for _ in 0..rounds {
                    for q in (0..num_qubits).step_by(2) {
                        sim.x(&[QubitId(q)]);
                        sim.z(&[QubitId(q + 1)]);
                    }
                    for q in (0..num_qubits - 1).step_by(2) {
                        sim.rzz(Angle64::from_radians(0.1), &[(QubitId(q), QubitId(q + 1))]);
                    }
                }
            });
        });

        // Same circuit on dense SoA
        group.bench_function(BenchmarkId::new("clifford_rzz_dense", &label), |b| {
            let mut sim = StateVecSoA::new(num_qubits);
            for q in 0..h_qubits {
                sim.h(&[QubitId(q)]);
            }
            b.iter(|| {
                for _ in 0..rounds {
                    for q in (0..num_qubits).step_by(2) {
                        sim.x(&[QubitId(q)]);
                        sim.z(&[QubitId(q + 1)]);
                    }
                    for q in (0..num_qubits - 1).step_by(2) {
                        sim.rzz(Angle64::from_radians(0.1), &[(QubitId(q), QubitId(q + 1))]);
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
    bench_three_statevecs_gates(c);
    bench_rotation_pauli_pushthrough(c);
    bench_nondiag_rotation_pushthrough(c);
    bench_qec_rotation_circuit(c);
}
