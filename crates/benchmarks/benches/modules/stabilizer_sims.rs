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

//! Stabilizer simulator benchmarks comparing GPU implementations.
//!
//! Compares performance of:
//! - `GpuStabMulti` (GPU via wgpu/Vulkan/Metal/DX12) - multi-shot stabilizer
//! - `CuFrameSimulator` (GPU via NVIDIA cuQuantum/CUDA) - frame-based stabilizer
//! - `SparseStab` (CPU) - baseline stabilizer simulator
//!
//! Run with specific features:
//! ```
//! cargo bench -p benchmarks --features gpu-sims        # GpuStabMulti only
//! cargo bench -p benchmarks --features cuquantum       # CuFrameSimulator (NVIDIA CUDA)
//! cargo bench -p benchmarks --features all-sims        # All simulators
//! ```

use criterion::{Criterion, measurement::Measurement};

#[cfg(any(feature = "gpu-sims", feature = "cuquantum"))]
use criterion::BenchmarkId;

#[cfg(any(feature = "gpu-sims", feature = "cuquantum"))]
use std::hint::black_box;

#[cfg(feature = "gpu-sims")]
use pecos_gpu_sims::GpuStabMulti;

#[cfg(feature = "gpu-sims")]
use pecos_core::QubitId;

#[cfg(feature = "cuquantum")]
use pecos_cuquantum::CuFrameSimulator;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_stabilizer_circuit_execution(c);
    bench_stabilizer_scaling(c);
}

/// Benchmark running a fixed circuit across many shots.
///
/// This is the primary use case for both `GpuStabMulti` and `CuFrameSimulator`:
/// running the same Clifford circuit on many independent shots in parallel.
fn bench_stabilizer_circuit_execution<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Stabilizer Circuit Execution");
    group.sample_size(20);

    // Test configurations: (num_qubits, num_shots, circuit_depth)
    let configs = [
        (10, 1000, 50),
        (20, 1000, 50),
        (50, 1000, 50),
        (100, 1000, 50),
        (100, 10000, 20),
        (200, 1000, 20),
    ];

    for (num_qubits, num_shots, depth) in configs {
        // Benchmark GpuStabMulti (wgpu)
        #[cfg(feature = "gpu-sims")]
        {
            let label = format!("{num_qubits}q_{num_shots}s_{depth}d");
            if let Ok(mut sim) = GpuStabMulti::new(num_qubits, num_shots) {
                group.bench_with_input(
                    BenchmarkId::new("GpuStabMulti_wgpu", &label),
                    &(num_qubits, depth),
                    |b, &(nq, d)| {
                        b.iter(|| {
                            sim.reset();
                            run_benchmark_circuit_gpustab(&mut sim, nq, d);
                            black_box(());
                        });
                    },
                );
            }
        }

        // Benchmark CuFrameSimulator (NVIDIA cuQuantum/CUDA)
        #[cfg(feature = "cuquantum")]
        {
            let label = format!("{num_qubits}q_{num_shots}s_{depth}d");
            let circuit_string = build_benchmark_circuit_string(num_qubits, depth);
            let num_measurements = num_qubits;
            if let Ok(mut sim) = CuFrameSimulator::new(num_qubits, num_shots, num_measurements) {
                let circuit = circuit_string.clone();
                group.bench_with_input(
                    BenchmarkId::new("CuFrameSimulator_CUDA", &label),
                    &(),
                    |b, ()| {
                        b.iter(|| {
                            let result = sim.run_circuit(&circuit, 42);
                            let _ = black_box(result);
                        });
                    },
                );
            }
        }

        // Suppress unused variable warnings when neither feature is enabled
        #[cfg(not(any(feature = "gpu-sims", feature = "cuquantum")))]
        {
            let _ = (num_qubits, num_shots, depth);
        }
    }

    group.finish();
}

/// Benchmark stabilizer simulation scaling with qubit count.
fn bench_stabilizer_scaling<M: Measurement>(c: &mut Criterion<M>) {
    #[cfg(any(feature = "gpu-sims", feature = "cuquantum"))]
    const NUM_SHOTS: usize = 1000;
    #[cfg(any(feature = "gpu-sims", feature = "cuquantum"))]
    const DEPTH: usize = 20;

    let mut group = c.benchmark_group("Stabilizer Qubit Scaling");
    group.sample_size(20);

    // Test scaling from small to large qubit counts
    let qubit_counts = [10, 25, 50, 100, 200, 500, 1000];

    for num_qubits in qubit_counts {
        // Benchmark GpuStabMulti (wgpu)
        #[cfg(feature = "gpu-sims")]
        {
            let label = format!("{num_qubits}q");
            if let Ok(mut sim) = GpuStabMulti::new(num_qubits, NUM_SHOTS) {
                group.bench_with_input(
                    BenchmarkId::new("GpuStabMulti_wgpu", &label),
                    &(num_qubits, DEPTH),
                    |b, &(nq, d)| {
                        b.iter(|| {
                            sim.reset();
                            run_benchmark_circuit_gpustab(&mut sim, nq, d);
                            black_box(());
                        });
                    },
                );
            }
        }

        // Benchmark CuFrameSimulator (NVIDIA cuQuantum/CUDA)
        #[cfg(feature = "cuquantum")]
        {
            let label = format!("{num_qubits}q");
            let circuit_string = build_benchmark_circuit_string(num_qubits, DEPTH);
            let num_measurements = num_qubits;
            if let Ok(mut sim) = CuFrameSimulator::new(num_qubits, NUM_SHOTS, num_measurements) {
                let circuit = circuit_string.clone();
                group.bench_with_input(
                    BenchmarkId::new("CuFrameSimulator_CUDA", &label),
                    &(),
                    |b, ()| {
                        b.iter(|| {
                            let result = sim.run_circuit(&circuit, 42);
                            let _ = black_box(result);
                        });
                    },
                );
            }
        }

        // Suppress unused variable warnings when neither feature is enabled
        #[cfg(not(any(feature = "gpu-sims", feature = "cuquantum")))]
        let _ = num_qubits;
    }

    group.finish();
}

/// Build a Stim-format circuit string for benchmarking.
///
/// Creates a circuit with:
/// - Layers of H gates on all qubits
/// - Layers of CNOT gates between adjacent qubits
/// - Final measurement of all qubits
#[cfg(feature = "cuquantum")]
fn build_benchmark_circuit_string(num_qubits: usize, depth: usize) -> String {
    let mut lines = Vec::new();

    for _layer in 0..depth {
        // H on all qubits
        for q in 0..num_qubits {
            lines.push(format!("H {q}"));
        }
        // S on half the qubits
        for q in (0..num_qubits).step_by(2) {
            lines.push(format!("S {q}"));
        }
        // CNOT chain
        for q in 0..(num_qubits - 1) {
            lines.push(format!("CNOT {q} {}", q + 1));
        }
    }

    // Final measurement
    let measure_qubits: Vec<String> = (0..num_qubits).map(|q| q.to_string()).collect();
    lines.push(format!("M {}", measure_qubits.join(" ")));

    lines.join("\n")
}

/// Run a benchmark circuit on `GpuStabMulti` using its native API.
#[cfg(feature = "gpu-sims")]
fn run_benchmark_circuit_gpustab(sim: &mut GpuStabMulti, num_qubits: usize, depth: usize) {
    for _layer in 0..depth {
        // H on all qubits
        for q in 0..num_qubits {
            sim.h(&[QubitId(q)]);
        }
        // S on half the qubits
        for q in (0..num_qubits).step_by(2) {
            sim.sz(&[QubitId(q)]);
        }
        // CNOT chain
        for q in 0..(num_qubits - 1) {
            sim.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
    }

    // Final measurement
    let qubits: Vec<QubitId> = (0..num_qubits).map(QubitId).collect();
    let _results = sim.mz(&qubits);
}
