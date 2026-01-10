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
//! - `QuestStateVec` (`QuEST` - CPU or CUDA)
//! - `QulacsStateVec` (Qulacs - CPU)
//! - `StateVec` (pecos-qsim pure Rust CPU)
//!
//! Run with specific features:
//! ```
//! cargo bench -p benchmarks --features gpu-sims        # GpuStateVec only
//! cargo bench -p benchmarks --features quest-cuda      # QuEST with CUDA
//! cargo bench -p benchmarks --features all-sims        # All simulators
//! ```

use criterion::{BenchmarkId, Criterion, measurement::Measurement};
use pecos_core::QubitId;
use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, StateVec};
use std::hint::black_box;

#[cfg(feature = "gpu-sims")]
use pecos_gpu_sims::GpuStateVec;

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
            sim.rz(0.1, &[QubitId(q)]);
        }
        // Two-qubit layer: CX between adjacent qubits
        for q in 0..(num_qubits - 1) {
            sim.cx(&[QubitId(q), QubitId(q + 1)]);
        }
    }
}

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_state_vec_scaling(c);
}

/// Benchmark state vector simulators across different qubit counts.
fn bench_state_vec_scaling<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("State Vector Simulators");

    // Use fewer samples for GPU benchmarks (they're more consistent)
    group.sample_size(20);

    // Test configurations: (num_qubits, num_layers)
    let configs = [
        (10, 20),
        (14, 20),
        (18, 20),
        (20, 20),
        (22, 10), // Fewer layers for larger qubit counts
        (24, 5),  // Large qubit count - GPU should dominate here
        (26, 3),  // Very large - 512 MB state vector
    ];

    for (num_qubits, num_layers) in configs {
        let label = format!("{num_qubits}q_{num_layers}l");

        // Benchmark pecos-qsim StateVec (CPU baseline)
        group.bench_with_input(
            BenchmarkId::new("StateVec_CPU", &label),
            &(num_qubits, num_layers),
            |b, &(nq, nl)| {
                let mut sim = StateVec::new(nq);
                b.iter(|| {
                    sim.reset();
                    benchmark_circuit(&mut sim, nq, nl);
                    black_box(());
                });
            },
        );

        // Benchmark GpuStateVec (GPU)
        #[cfg(feature = "gpu-sims")]
        {
            // Safe: num_qubits comes from configs array with small values (10-22)
            #[allow(clippy::cast_possible_truncation)]
            if let Ok(mut sim) = GpuStateVec::new(num_qubits as u32) {
                group.bench_with_input(
                    BenchmarkId::new("GpuStateVec_GPU", &label),
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
