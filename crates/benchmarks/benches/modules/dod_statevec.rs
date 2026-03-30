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

//! DOD (Data-Oriented Design) micro-benchmarks for `StateVec`.
//!
//! These benchmarks focus on measuring specific gate operations and data layout
//! impacts for the `StateVec` simulator. The goal is to:
//!
//! 1. Establish baselines for individual gate performance
//! 2. Measure the impact of different qubit counts (cache pressure)
//! 3. Compare current implementation vs DOD-optimized versions
//!
//! Run with:
//! ```
//! cargo bench -p benchmarks -- "DOD StateVec"
//! ```

use criterion::{BenchmarkId, Criterion, Throughput, measurement::Measurement};
use num_complex::Complex64;
use pecos_core::QubitId;
use pecos_simulators::{CliffordGateable, StateVecAoS, StateVecSoA};
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_single_qubit_gates(c);
    bench_two_qubit_gates(c);
    bench_cx_scaling(c);
    bench_allocation_overhead(c);
    bench_dod_comparison(c);
    bench_fused_gates(c);
}

/// Benchmark individual single-qubit gates (H, X, Z, S) at various qubit counts.
///
/// This measures the raw gate performance without circuit overhead.
fn bench_single_qubit_gates<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("DOD StateVec Single-Qubit Gates");
    group.sample_size(50);

    // Test at different qubit counts to see cache effects
    // 10 qubits = 16 KB state, 15 = 512 KB, 20 = 16 MB, 24 = 256 MB
    let qubit_counts = [10, 12, 14, 16, 18, 20];

    for num_qubits in qubit_counts {
        let state_size = 1usize << num_qubits;
        group.throughput(Throughput::Elements(state_size as u64));

        // H gate - involves both real/imag components
        group.bench_with_input(BenchmarkId::new("H", num_qubits), &num_qubits, |b, &nq| {
            let mut sim = StateVecSoA::new(nq);
            let target = nq / 2; // Middle qubit
            b.iter(|| {
                sim.h(&[QubitId(target)]);
                black_box(());
            });
        });

        // X gate - pure swap operation
        group.bench_with_input(BenchmarkId::new("X", num_qubits), &num_qubits, |b, &nq| {
            let mut sim = StateVecSoA::new(nq);
            let target = nq / 2;
            b.iter(|| {
                sim.x(&[QubitId(target)]);
                black_box(());
            });
        });

        // Z gate - phase only, no swaps
        group.bench_with_input(BenchmarkId::new("Z", num_qubits), &num_qubits, |b, &nq| {
            let mut sim = StateVecSoA::new(nq);
            let target = nq / 2;
            b.iter(|| {
                sim.z(&[QubitId(target)]);
                black_box(());
            });
        });

        // S gate - phase only
        group.bench_with_input(BenchmarkId::new("S", num_qubits), &num_qubits, |b, &nq| {
            let mut sim = StateVecSoA::new(nq);
            let target = nq / 2;
            b.iter(|| {
                sim.sz(&[QubitId(target)]);
                black_box(());
            });
        });
    }

    group.finish();
}

/// Benchmark two-qubit gates (CX, CZ, SWAP) at various qubit counts.
///
/// These are particularly interesting for DOD because:
/// - CX/CY use branching iteration (only 25% of iterations do work)
/// - CZ only touches indices where both qubits are 1
/// - SWAP has specific access patterns
fn bench_two_qubit_gates<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("DOD StateVec Two-Qubit Gates");
    group.sample_size(50);

    let qubit_counts = [10, 12, 14, 16, 18, 20];

    for num_qubits in qubit_counts {
        let state_size = 1usize << num_qubits;
        group.throughput(Throughput::Elements(state_size as u64));

        // CX gate - adjacent qubits
        group.bench_with_input(
            BenchmarkId::new("CX_adjacent", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                let q1 = nq / 2;
                let q2 = q1 + 1;
                b.iter(|| {
                    sim.cx(&[(QubitId(q1), QubitId(q2))]);
                    black_box(());
                });
            },
        );

        // CX gate - distant qubits (worse cache locality)
        group.bench_with_input(
            BenchmarkId::new("CX_distant", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                let q1 = 0;
                let q2 = nq - 1;
                b.iter(|| {
                    sim.cx(&[(QubitId(q1), QubitId(q2))]);
                    black_box(());
                });
            },
        );

        // CZ gate - diagonal, only phase changes
        group.bench_with_input(BenchmarkId::new("CZ", num_qubits), &num_qubits, |b, &nq| {
            let mut sim = StateVecSoA::new(nq);
            let q1 = nq / 2;
            let q2 = q1 + 1;
            b.iter(|| {
                sim.cz(&[(QubitId(q1), QubitId(q2))]);
                black_box(());
            });
        });

        // SWAP gate
        group.bench_with_input(
            BenchmarkId::new("SWAP", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                let q1 = nq / 2;
                let q2 = q1 + 1;
                b.iter(|| {
                    sim.swap(&[(QubitId(q1), QubitId(q2))]);
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark CX gate scaling to understand iteration overhead.
///
/// The current CX iterates over all 2^n amplitudes but only 25% do work.
/// This benchmark measures whether that overhead matters at scale.
fn bench_cx_scaling<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("DOD StateVec CX Scaling");
    group.sample_size(30);

    // Focus on larger qubit counts where the inefficiency matters more
    let qubit_counts = [14, 16, 18, 20, 22];

    for num_qubits in qubit_counts {
        let state_size = 1usize << num_qubits;
        group.throughput(Throughput::Elements(state_size as u64));

        // Low qubit index (large stride)
        group.bench_with_input(
            BenchmarkId::new("CX_low_qubit", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                // Control on qubit 0, target on qubit 1
                b.iter(|| {
                    sim.cx(&[(QubitId(0), QubitId(1))]);
                    black_box(());
                });
            },
        );

        // High qubit index (small stride)
        group.bench_with_input(
            BenchmarkId::new("CX_high_qubit", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                // Control on high qubit, target on highest
                b.iter(|| {
                    sim.cx(&[(QubitId(nq - 2), QubitId(nq - 1))]);
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark the overhead of temporary allocations in `two_qubit_unitary`.
///
/// This measures the cost of allocating a full state vector copy per gate.
fn bench_allocation_overhead<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("DOD StateVec Allocation Overhead");
    group.sample_size(30);

    let qubit_counts = [14, 16, 18, 20];

    for num_qubits in qubit_counts {
        let state_size = 1usize << num_qubits;
        let memory_bytes = state_size * std::mem::size_of::<Complex64>();
        group.throughput(Throughput::Bytes(memory_bytes as u64));

        // Measure just allocation + deallocation
        group.bench_with_input(
            BenchmarkId::new("alloc_only", num_qubits),
            &num_qubits,
            |b, &nq| {
                let size = 1usize << nq;
                b.iter(|| {
                    let v: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); size];
                    black_box(v);
                });
            },
        );

        // Measure allocation + zeroing + iteration
        group.bench_with_input(
            BenchmarkId::new("alloc_and_iterate", num_qubits),
            &num_qubits,
            |b, &nq| {
                let size = 1usize << nq;
                b.iter(|| {
                    let mut v: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); size];
                    // Touch every element
                    for (i, vi) in v.iter_mut().enumerate() {
                        *vi = Complex64::new(i as f64, 0.0);
                    }
                    black_box(v);
                });
            },
        );

        // Compare to reusing a buffer
        group.bench_with_input(
            BenchmarkId::new("reuse_buffer", num_qubits),
            &num_qubits,
            |b, &nq| {
                let size = 1usize << nq;
                let mut buffer: Vec<Complex64> = vec![Complex64::new(0.0, 0.0); size];
                b.iter(|| {
                    // Just clear and iterate (no allocation)
                    for (i, bi) in buffer.iter_mut().enumerate() {
                        *bi = Complex64::new(i as f64, 0.0);
                    }
                    black_box(&buffer);
                });
            },
        );
    }

    group.finish();
}

/// Compare `StateVec` (baseline) vs `StateVecAoS` (optimized) for CX gate.
///
/// This directly measures the impact of strided iteration.
fn bench_dod_comparison<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("DOD Comparison: CX");
    group.sample_size(30);

    let qubit_counts = [14, 16, 18, 20];

    for num_qubits in qubit_counts {
        let state_size = 1usize << num_qubits;
        group.throughput(Throughput::Elements(state_size as u64));

        // Baseline: StateVec (branching iteration)
        group.bench_with_input(
            BenchmarkId::new("StateVec", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                let q1 = nq / 2;
                let q2 = q1 + 1;
                b.iter(|| {
                    sim.cx(&[(QubitId(q1), QubitId(q2))]);
                    black_box(());
                });
            },
        );

        // DOD optimized: StateVecAoS (strided iteration)
        group.bench_with_input(
            BenchmarkId::new("StateVecAoS", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecAoS::new(nq);
                let q1 = nq / 2;
                let q2 = q1 + 1;
                b.iter(|| {
                    sim.cx(&[(QubitId(q1), QubitId(q2))]);
                    black_box(());
                });
            },
        );
    }

    group.finish();

    // Also compare H gate (should be similar since both use strided iteration)
    let mut group = c.benchmark_group("DOD Comparison: H");
    group.sample_size(30);

    for num_qubits in qubit_counts {
        let state_size = 1usize << num_qubits;
        group.throughput(Throughput::Elements(state_size as u64));

        group.bench_with_input(
            BenchmarkId::new("StateVec", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                let target = nq / 2;
                b.iter(|| {
                    sim.h(&[QubitId(target)]);
                    black_box(());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("StateVecAoS", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecAoS::new(nq);
                let target = nq / 2;
                b.iter(|| {
                    sim.h(&[QubitId(target)]);
                    black_box(());
                });
            },
        );
    }

    group.finish();

    // Compare CZ gate (strided vs branching)
    let mut group = c.benchmark_group("DOD Comparison: CZ");
    group.sample_size(30);

    for num_qubits in qubit_counts {
        let state_size = 1usize << num_qubits;
        group.throughput(Throughput::Elements(state_size as u64));

        group.bench_with_input(
            BenchmarkId::new("StateVec", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                let q1 = nq / 2;
                let q2 = q1 + 1;
                b.iter(|| {
                    sim.cz(&[(QubitId(q1), QubitId(q2))]);
                    black_box(());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("StateVecAoS", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecAoS::new(nq);
                let q1 = nq / 2;
                let q2 = q1 + 1;
                b.iter(|| {
                    sim.cz(&[(QubitId(q1), QubitId(q2))]);
                    black_box(());
                });
            },
        );
    }

    group.finish();

    // Compare SWAP gate (strided vs branching)
    let mut group = c.benchmark_group("DOD Comparison: SWAP");
    group.sample_size(30);

    for num_qubits in qubit_counts {
        let state_size = 1usize << num_qubits;
        group.throughput(Throughput::Elements(state_size as u64));

        group.bench_with_input(
            BenchmarkId::new("StateVec", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                let q1 = nq / 2;
                let q2 = q1 + 1;
                b.iter(|| {
                    sim.swap(&[(QubitId(q1), QubitId(q2))]);
                    black_box(());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("StateVecAoS", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecAoS::new(nq);
                let q1 = nq / 2;
                let q2 = q1 + 1;
                b.iter(|| {
                    sim.swap(&[(QubitId(q1), QubitId(q2))]);
                    black_box(());
                });
            },
        );
    }

    group.finish();
}

/// Benchmark fused gates vs separate gate applications.
///
/// Fused gates combine two consecutive gates into a single memory pass,
/// reducing memory bandwidth by ~50% and avoiding intermediate state writes.
fn bench_fused_gates<M: Measurement>(c: &mut Criterion<M>) {
    let qubit_counts = [14, 16, 18, 20];

    // H-Z fusion comparison
    let mut group = c.benchmark_group("Fused Gates: H-Z");
    group.sample_size(50);

    for num_qubits in qubit_counts {
        let state_size = 1usize << num_qubits;
        group.throughput(Throughput::Elements(state_size as u64));

        // Separate H then Z
        group.bench_with_input(
            BenchmarkId::new("separate", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                let target = nq / 2;
                b.iter(|| {
                    sim.h(&[QubitId(target)]);
                    sim.z(&[QubitId(target)]);
                    black_box(());
                });
            },
        );

        // Fused H-Z
        group.bench_with_input(
            BenchmarkId::new("fused", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                let target = nq / 2;
                b.iter(|| {
                    sim.hz(&[QubitId(target)]);
                    black_box(());
                });
            },
        );
    }

    group.finish();

    // H-S fusion comparison
    let mut group = c.benchmark_group("Fused Gates: H-S");
    group.sample_size(50);

    for num_qubits in qubit_counts {
        let state_size = 1usize << num_qubits;
        group.throughput(Throughput::Elements(state_size as u64));

        // Separate H then S
        group.bench_with_input(
            BenchmarkId::new("separate", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                let target = nq / 2;
                b.iter(|| {
                    sim.h(&[QubitId(target)]);
                    sim.sz(&[QubitId(target)]);
                    black_box(());
                });
            },
        );

        // Fused H-S
        group.bench_with_input(
            BenchmarkId::new("fused", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                let target = nq / 2;
                b.iter(|| {
                    sim.hs(&[QubitId(target)]);
                    black_box(());
                });
            },
        );
    }

    group.finish();

    // H-X fusion comparison
    let mut group = c.benchmark_group("Fused Gates: H-X");
    group.sample_size(50);

    for num_qubits in qubit_counts {
        let state_size = 1usize << num_qubits;
        group.throughput(Throughput::Elements(state_size as u64));

        // Separate H then X
        group.bench_with_input(
            BenchmarkId::new("separate", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                let target = nq / 2;
                b.iter(|| {
                    sim.h(&[QubitId(target)]);
                    sim.x(&[QubitId(target)]);
                    black_box(());
                });
            },
        );

        // Fused H-X
        group.bench_with_input(
            BenchmarkId::new("fused", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                let target = nq / 2;
                b.iter(|| {
                    sim.hx(&[QubitId(target)]);
                    black_box(());
                });
            },
        );
    }

    group.finish();
}
