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

//! State vector simulator benchmarks comparing GPU and CPU implementations.
//!
//! Compares performance of:
//! - `GpuStateVec` (GPU via wgpu/Vulkan/Metal/DX12)
//! - `CuStateVec` (GPU via NVIDIA cuQuantum/CUDA)
//! - `QuestStateVec` (`QuEST` - CPU or CUDA)
//! - `QulacsStateVec` (Qulacs - CPU)
//! - `StateVec` (pecos-simulators pure Rust CPU)
//!
//! Run with specific features:
//! ```
//! cargo bench -p benchmarks --features gpu-sims        # GpuStateVec only
//! cargo bench -p benchmarks --features cuquantum       # CuStateVec (NVIDIA CUDA)
//! cargo bench -p benchmarks --features quest-cuda      # QuEST with CUDA
//! cargo bench -p benchmarks --features all-sims        # All simulators
//! ```

use criterion::{BenchmarkId, Criterion, measurement::Measurement};
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, StateVecSoA,
};
use std::hint::black_box;

#[cfg(feature = "gpu-sims")]
use pecos_gpu_sims::GpuStateVec;

#[cfg(feature = "cuquantum")]
use pecos_cuquantum::CuStateVec;

#[cfg(all(feature = "quest", not(feature = "quest-cuda")))]
use pecos_quest::QuestStateVec;

#[cfg(feature = "quest-cuda")]
use pecos_quest::QuestCudaStateVecEngine;

#[cfg(feature = "qulacs")]
use pecos_qulacs::QulacsStateVec;

/// Run a benchmark circuit: layers of H + RZ + CX gates.
fn benchmark_circuit<S>(sim: &mut S, num_qubits: usize, num_layers: usize)
where
    S: CliffordGateable + ArbitraryRotationGateable,
{
    for _layer in 0..num_layers {
        // Single-qubit layer: H and RZ on all qubits
        for q in 0..num_qubits {
            sim.h(&[QubitId(q)]);
            sim.rz(Angle64::from_radians(0.1), &[QubitId(q)]);
        }
        // Two-qubit layer: CX between adjacent qubits
        for q in 0..(num_qubits - 1) {
            sim.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
    }
}

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_state_vec_scaling(c);
    bench_individual_gates(c);
    bench_measurement_scaling(c);
    bench_subset_measurement(c);
    bench_flush_scaling(c);
    #[cfg(feature = "parallel")]
    bench_parallel_execution(c);
}

/// Benchmark individual gate performance on `StateVec`.
/// Tests gates commonly used in QEC circuits: H, SZ, SX, CX, SZZ, SXX, and prep.
fn bench_individual_gates<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Individual Gates");
    group.sample_size(50);

    let num_qubits = 18;
    let gates_per_iter = 100;

    // Single-qubit gates
    group.bench_function("H_18q", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        b.iter(|| {
            for _ in 0..gates_per_iter {
                for q in 0..num_qubits {
                    sim.h(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("SZ_18q", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        b.iter(|| {
            for _ in 0..gates_per_iter {
                for q in 0..num_qubits {
                    sim.sz(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("SX_18q", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        b.iter(|| {
            for _ in 0..gates_per_iter {
                for q in 0..num_qubits {
                    sim.sx(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("X_18q", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        b.iter(|| {
            for _ in 0..gates_per_iter {
                for q in 0..num_qubits {
                    sim.x(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("Y_18q", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        b.iter(|| {
            for _ in 0..gates_per_iter {
                for q in 0..num_qubits {
                    sim.y(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("Z_18q", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        b.iter(|| {
            for _ in 0..gates_per_iter {
                for q in 0..num_qubits {
                    sim.z(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    // Two-qubit gates
    group.bench_function("CX_18q", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        b.iter(|| {
            for _ in 0..gates_per_iter {
                for q in 0..num_qubits - 1 {
                    sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("CZ_18q", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        b.iter(|| {
            for _ in 0..gates_per_iter {
                for q in 0..num_qubits - 1 {
                    sim.cz(&[(QubitId(q), QubitId(q + 1))]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("SZZ_18q", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        b.iter(|| {
            for _ in 0..gates_per_iter {
                for q in 0..num_qubits - 1 {
                    sim.szz(&[(QubitId(q), QubitId(q + 1))]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("SXX_18q", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        b.iter(|| {
            for _ in 0..gates_per_iter {
                for q in 0..num_qubits - 1 {
                    sim.sxx(&[(QubitId(q), QubitId(q + 1))]);
                }
            }
            black_box(());
        });
    });

    // Prep operation (measure + prepare |0⟩)
    group.bench_function("pz_18q", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        // Put in superposition first so prep has work to do
        for q in 0..num_qubits {
            sim.h(&[QubitId(q)]);
        }
        b.iter(|| {
            for _ in 0..gates_per_iter {
                for q in 0..num_qubits {
                    sim.pz(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    // Measurement
    group.bench_function("mz_18q", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        b.iter(|| {
            // Reset to known state, apply H, then measure
            sim.reset();
            for q in 0..num_qubits {
                sim.h(&[QubitId(q)]);
            }
            for q in 0..num_qubits {
                black_box(sim.mz(&[QubitId(q)]));
            }
        });
    });

    group.finish();
}

/// Benchmark measurement performance: sequential (per-qubit) vs batch (all at once).
///
/// Sequential: `for q in 0..n { sim.mz(&[QubitId(q)]); }` — 2n passes over state vector.
/// Batch: `sim.mz(&all_qubits)` — uses joint sampling, 2 passes over state vector.
fn bench_measurement_scaling<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Measurement Scaling");
    group.sample_size(20);

    let qubit_counts = [10, 14, 18, 20, 22];

    for &nq in &qubit_counts {
        // Sequential: one mz() call per qubit (2n passes)
        group.bench_with_input(BenchmarkId::new("mz_sequential", nq), &nq, |b, &nq| {
            let mut sim = StateVecSoA::new(nq);
            b.iter(|| {
                sim.reset();
                for q in 0..nq {
                    sim.h(&[QubitId(q)]);
                }
                for q in 0..nq {
                    black_box(sim.mz(&[QubitId(q)]));
                }
            });
        });

        // Batch: one mz() call with all qubits (2 passes via joint sampling)
        group.bench_with_input(BenchmarkId::new("mz_batch", nq), &nq, |b, &nq| {
            let mut sim = StateVecSoA::new(nq);
            let all_qubits: Vec<QubitId> = (0..nq).map(QubitId).collect();
            b.iter(|| {
                sim.reset();
                for q in 0..nq {
                    sim.h(&[QubitId(q)]);
                }
                black_box(sim.mz(&all_qubits));
            });
        });

        // GPU (sequential per-qubit for comparison)
        #[cfg(feature = "gpu-sims")]
        {
            #[allow(clippy::cast_possible_truncation)]
            if let Ok(mut sim) = GpuStateVec::new(nq as u32) {
                group.bench_with_input(BenchmarkId::new("GpuStateVec_wgpu", nq), &nq, |b, &nq| {
                    b.iter(|| {
                        sim.reset();
                        for q in 0..nq {
                            sim.h(&[QubitId(q)]);
                        }
                        for q in 0..nq {
                            black_box(sim.mz(&[QubitId(q)]));
                        }
                    });
                });
            }
        }
    }

    group.finish();
}

/// Benchmark subset measurement: measure half the qubits (even-indexed).
/// Tests the `mz_joint_subset` path (QEC-realistic: measure ancillas, not data qubits).
fn bench_subset_measurement<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Subset Measurement");
    group.sample_size(20);

    let qubit_counts = [10, 14, 18, 20, 22];

    for &nq in &qubit_counts {
        let half: Vec<QubitId> = (0..nq).step_by(2).map(QubitId).collect();
        let half_count = half.len();

        // Sequential: one mz() per qubit
        group.bench_with_input(
            BenchmarkId::new("mz_sequential", format!("{nq}q_{half_count}m")),
            &nq,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                b.iter(|| {
                    sim.reset();
                    for q in 0..nq {
                        sim.h(&[QubitId(q)]);
                    }
                    for &q in &half {
                        black_box(sim.mz(&[q]));
                    }
                });
            },
        );

        // Batch: one mz() with all measured qubits
        group.bench_with_input(
            BenchmarkId::new("mz_batch_subset", format!("{nq}q_{half_count}m")),
            &nq,
            |b, &nq| {
                let mut sim = StateVecSoA::new(nq);
                b.iter(|| {
                    sim.reset();
                    for q in 0..nq {
                        sim.h(&[QubitId(q)]);
                    }
                    black_box(sim.mz(&half));
                });
            },
        );
    }

    group.finish();
}

/// Benchmark flush performance: H on all qubits then flush.
/// Isolates the cache-blocked flush optimization from measurement.
fn bench_flush_scaling<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Flush Scaling");
    group.sample_size(20);

    let qubit_counts = [14, 18, 20, 22];

    for &nq in &qubit_counts {
        group.bench_with_input(BenchmarkId::new("h_all_flush", nq), &nq, |b, &nq| {
            let mut sim = StateVecSoA::new(nq);
            b.iter(|| {
                sim.reset();
                for q in 0..nq {
                    sim.h(&[QubitId(q)]);
                }
                sim.flush();
                black_box(());
            });
        });
    }

    group.finish();
}

/// Benchmark parallel vs sequential execution for large state vectors.
/// Only runs when the `parallel` feature is enabled on pecos-simulators.
#[cfg(feature = "parallel")]
fn bench_parallel_execution<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Parallel Execution");
    group.sample_size(20);

    // Test with 18 qubits - above the parallel threshold (14)
    let num_qubits = 18;
    let gates_per_iter = 50;

    // Sequential H gates
    group.bench_function("H_18q_sequential", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        sim.set_parallel(false);
        b.iter(|| {
            for _ in 0..gates_per_iter {
                for q in 0..num_qubits {
                    sim.h(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    // Parallel H gates
    group.bench_function("H_18q_parallel", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        sim.set_parallel(true);
        b.iter(|| {
            for _ in 0..gates_per_iter {
                for q in 0..num_qubits {
                    sim.h(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    // Test with 20 qubits for even more parallelism benefit
    let num_qubits_large = 20;
    let gates_per_iter_large = 20;

    group.bench_function("H_20q_sequential", |b| {
        let mut sim = StateVecSoA::new(num_qubits_large);
        sim.set_parallel(false);
        b.iter(|| {
            for _ in 0..gates_per_iter_large {
                for q in 0..num_qubits_large {
                    sim.h(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("H_20q_parallel", |b| {
        let mut sim = StateVecSoA::new(num_qubits_large);
        sim.set_parallel(true);
        b.iter(|| {
            for _ in 0..gates_per_iter_large {
                for q in 0..num_qubits_large {
                    sim.h(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.finish();
}

/// Benchmark state vector simulators across different qubit counts.
fn bench_state_vec_scaling<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("State Vector Simulators");

    // Use fewer samples for GPU benchmarks (they're more consistent)
    group.sample_size(20);

    // Test configurations: (num_qubits, num_layers)
    // Note: 26+ qubits exceeds wgpu's max_buffer_binding_size limit (128 MB default)
    let configs = [
        (10, 20),
        (14, 20),
        (18, 20),
        (20, 20),
        (22, 10), // Fewer layers for larger qubit counts
        (24, 5),  // Large qubit count - GPU should dominate here
    ];

    for (num_qubits, num_layers) in configs {
        let label = format!("{num_qubits}q_{num_layers}l");

        // Benchmark pecos-simulators StateVec (CPU baseline)
        group.bench_with_input(
            BenchmarkId::new("StateVec_CPU", &label),
            &(num_qubits, num_layers),
            |b, &(nq, nl)| {
                let mut sim = StateVecSoA::new(nq);
                b.iter(|| {
                    sim.reset();
                    benchmark_circuit(&mut sim, nq, nl);
                    black_box(());
                });
            },
        );

        // Benchmark GpuStateVec (wgpu GPU)
        #[cfg(feature = "gpu-sims")]
        {
            // Safe: num_qubits comes from configs array with small values (10-22)
            #[allow(clippy::cast_possible_truncation)]
            if let Ok(mut sim) = GpuStateVec::new(num_qubits as u32) {
                group.bench_with_input(
                    BenchmarkId::new("GpuStateVec_wgpu", &label),
                    &(num_qubits, num_layers),
                    |b, &(nq, nl)| {
                        b.iter(|| {
                            sim.reset();
                            benchmark_circuit(&mut sim, nq, nl);
                            black_box(());
                        });
                    },
                );
            }
        }

        // Benchmark CuStateVec (NVIDIA cuQuantum/CUDA)
        #[cfg(feature = "cuquantum")]
        {
            match CuStateVec::new(num_qubits) {
                Ok(mut sim) => {
                    group.bench_with_input(
                        BenchmarkId::new("CuStateVec_CUDA", &label),
                        &(num_qubits, num_layers),
                        |b, &(nq, nl)| {
                            b.iter(|| {
                                sim.reset();
                                benchmark_circuit(&mut sim, nq, nl);
                                black_box(());
                            });
                        },
                    );
                }
                Err(e) => {
                    eprintln!("Warning: Failed to create CuStateVec({num_qubits}): {e}");
                }
            }
        }

        // Benchmark QuEST (CPU mode - when quest feature is enabled but not quest-cuda)
        #[cfg(all(feature = "quest", not(feature = "quest-cuda")))]
        {
            let mut sim = QuestStateVec::new(num_qubits);
            group.bench_with_input(
                BenchmarkId::new("QuestStateVec_CPU", &label),
                &(num_qubits, num_layers),
                |b, &(nq, nl)| {
                    b.iter(|| {
                        sim.reset();
                        benchmark_circuit(&mut sim, nq, nl);
                        black_box(())
                    });
                },
            );
        }

        // NOTE: QuEST CUDA benchmarks are disabled in the loop due to a QuEST bug:
        // 1. QuEST CUDA only supports ONE qureg at a time
        // 2. After destroying a qureg, subsequent qureg creation fails
        // The CUDA benchmark is run separately below for a single configuration.

        // Benchmark Qulacs
        #[cfg(feature = "qulacs")]
        {
            let mut sim = QulacsStateVec::new(num_qubits);
            group.bench_with_input(
                BenchmarkId::new("QulacsStateVec_CPU", &label),
                &(num_qubits, num_layers),
                |b, &(nq, nl)| {
                    b.iter(|| {
                        sim.reset();
                        benchmark_circuit(&mut sim, nq, nl);
                        black_box(());
                    });
                },
            );
        }
    }

    // QuEST CUDA benchmark - run separately due to QuEST bugs:
    // 1. Only one qureg can exist at a time
    // 2. After destroying a qureg, subsequent creations fail
    // 3. Creating quregs with 12+ qubits fails (QuEST CUDA configuration limit?)
    // We run a single configuration (10 qubits) to compare against CPU implementations.
    #[cfg(feature = "quest-cuda")]
    {
        let cuda_config = (10, 20); // 10 qubits, 20 layers - max reliable size
        let (num_qubits, num_layers) = cuda_config;
        let label = format!("{num_qubits}q_{num_layers}l");

        match QuestCudaStateVecEngine::new(num_qubits) {
            Ok(mut sim) => {
                group.bench_with_input(
                    BenchmarkId::new("QuestCuda_GPU", &label),
                    &(num_qubits, num_layers),
                    |b, &(nq, nl)| {
                        b.iter(|| {
                            sim.reset();
                            benchmark_circuit(&mut sim, nq, nl);
                            black_box(());
                        });
                    },
                );
            }
            Err(e) => {
                eprintln!("Warning: Failed to create QuestCudaStateVecEngine: {e}");
            }
        }
    }

    group.finish();
}
