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

//! Native state vector comparison benchmarks.
//!
//! Compares raw gate computation performance across PECOS's internal state vector simulators
//! (`StateVecSoA`, `StateVecSoA32`, `StateVecAoS`) at the trait layer, plus GPU simulators
//! (`GpuStateVec32` via wgpu, `CuStateVec` via cuQuantum) when their respective features
//! (`gpu-sims`, `cuquantum`) are enabled.

use criterion::{BenchmarkId, Criterion, measurement::Measurement};
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{
    ArbitraryRotationGateable, CliffordGateable, QuantumSimulator, StateVecAoS, StateVecSoA,
    StateVecSoA32,
};
use std::hint::black_box;

#[cfg(feature = "gpu-sims")]
use pecos_gpu_sims::{GpuStateVec32, gates as gpu_gates};

#[cfg(feature = "cuquantum")]
use pecos_cuquantum::CuStateVec;

// ---------------------------------------------------------------------------
// Helpers for PECOS simulators (trait-based calls)
// ---------------------------------------------------------------------------

fn pecos_circuit<S: CliffordGateable + ArbitraryRotationGateable>(
    sim: &mut S,
    num_qubits: usize,
    num_layers: usize,
) {
    for _layer in 0..num_layers {
        for q in 0..num_qubits {
            sim.h(&[QubitId(q)]);
            sim.rz(Angle64::from_radians(0.1), &[QubitId(q)]);
        }
        for q in 0..num_qubits - 1 {
            sim.cx(&[(QubitId(q), QubitId(q + 1))]);
        }
    }
}

// ---------------------------------------------------------------------------
// GpuStateVec32 direct helpers (bypasses trait layer, calls wgpu dispatch directly)
// ---------------------------------------------------------------------------

#[cfg(feature = "gpu-sims")]
fn gpu_circuit(sim: &mut GpuStateVec32, num_qubits: usize, num_layers: usize) {
    let rz_matrix = gpu_gates::rz(0.1);
    for _layer in 0..num_layers {
        for q in 0..num_qubits {
            sim.apply_single_gate(q as u32, gpu_gates::H);
            sim.apply_single_gate(q as u32, rz_matrix);
        }
        for q in 0..num_qubits - 1 {
            sim.apply_cx(q as u32, (q + 1) as u32);
        }
    }
}

// ---------------------------------------------------------------------------
// CuStateVec direct helpers (bypasses trait layer, calls custatevecApplyMatrix directly)
// ---------------------------------------------------------------------------

#[cfg(feature = "cuquantum")]
mod cuquantum_matrices {
    use std::f64::consts::FRAC_1_SQRT_2;

    pub const H: [[f64; 2]; 4] = [
        [FRAC_1_SQRT_2, 0.0],
        [FRAC_1_SQRT_2, 0.0],
        [FRAC_1_SQRT_2, 0.0],
        [-FRAC_1_SQRT_2, 0.0],
    ];

    pub const X: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [0.0, 0.0]];

    pub const CX: [[f64; 2]; 16] = [
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 0.0],
    ];

    pub fn rz(theta: f64) -> [[f64; 2]; 4] {
        let c = (theta / 2.0).cos();
        let s = (theta / 2.0).sin();
        [[c, -s], [0.0, 0.0], [0.0, 0.0], [c, s]]
    }
}

#[cfg(feature = "cuquantum")]
fn cuquantum_circuit(sim: &mut CuStateVec, num_qubits: usize, num_layers: usize) {
    let rz_matrix = cuquantum_matrices::rz(0.1);
    for _layer in 0..num_layers {
        for q in 0..num_qubits {
            sim.apply_matrix_1q(q, &cuquantum_matrices::H);
            sim.apply_matrix_1q(q, &rz_matrix);
        }
        for q in 0..num_qubits - 1 {
            sim.apply_matrix_2q(q, q + 1, &cuquantum_matrices::CX);
        }
    }
}

// ---------------------------------------------------------------------------
// Benchmark group 1: Layered circuit scaling
// ---------------------------------------------------------------------------

fn bench_native_statevec_comparison<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Native StateVec Comparison");
    group.sample_size(20);

    let configs = [(10, 20), (14, 20), (18, 20), (20, 20), (22, 10), (24, 5)];

    for (num_qubits, num_layers) in configs {
        let label = format!("{num_qubits}q_{num_layers}l");

        // -- StateVecSoA (default: fusion on) --
        group.bench_with_input(
            BenchmarkId::new("StateVecSoA", &label),
            &(num_qubits, num_layers),
            |b, &(nq, nl)| {
                let mut sim = StateVecSoA::new(nq);
                sim.set_parallel(false);
                b.iter(|| {
                    sim.reset();
                    pecos_circuit(&mut sim, nq, nl);
                    black_box(());
                });
            },
        );

        // -- StateVecSoA (no fusion) --
        group.bench_with_input(
            BenchmarkId::new("StateVecSoA/no_fusion", &label),
            &(num_qubits, num_layers),
            |b, &(nq, nl)| {
                let mut sim = StateVecSoA::new(nq);
                sim.set_parallel(false);
                sim.set_fusion(false);
                b.iter(|| {
                    sim.reset();
                    pecos_circuit(&mut sim, nq, nl);
                    black_box(());
                });
            },
        );

        // -- StateVecAoS --
        group.bench_with_input(
            BenchmarkId::new("StateVecAoS", &label),
            &(num_qubits, num_layers),
            |b, &(nq, nl)| {
                let mut sim = StateVecAoS::new(nq);
                b.iter(|| {
                    sim.reset();
                    pecos_circuit(&mut sim, nq, nl);
                    black_box(());
                });
            },
        );

        // -- StateVecSoA32 (default: fusion on) --
        group.bench_with_input(
            BenchmarkId::new("StateVecSoA32", &label),
            &(num_qubits, num_layers),
            |b, &(nq, nl)| {
                let mut sim = StateVecSoA32::new(nq);
                b.iter(|| {
                    sim.reset();
                    pecos_circuit(&mut sim, nq, nl);
                    black_box(());
                });
            },
        );

        // -- StateVecSoA32 (no fusion) --
        group.bench_with_input(
            BenchmarkId::new("StateVecSoA32/no_fusion", &label),
            &(num_qubits, num_layers),
            |b, &(nq, nl)| {
                let mut sim = StateVecSoA32::new(nq);
                sim.set_fusion(false);
                b.iter(|| {
                    sim.reset();
                    pecos_circuit(&mut sim, nq, nl);
                    black_box(());
                });
            },
        );

        // -- GpuStateVec32 direct (wgpu) --
        #[cfg(feature = "gpu-sims")]
        if let Ok(mut sim) = GpuStateVec32::new(num_qubits as u32) {
            group.bench_with_input(
                BenchmarkId::new("GpuStateVec32_direct", &label),
                &(num_qubits, num_layers),
                |b, &(nq, nl)| {
                    b.iter(|| {
                        sim.reset();
                        gpu_circuit(&mut sim, nq, nl);
                        black_box(());
                    });
                },
            );
        }

        // -- CuStateVec direct (cuQuantum) --
        #[cfg(feature = "cuquantum")]
        match CuStateVec::new(num_qubits) {
            Ok(mut sim) => {
                group.bench_with_input(
                    BenchmarkId::new("CuStateVec_direct", &label),
                    &(num_qubits, num_layers),
                    |b, &(nq, nl)| {
                        b.iter(|| {
                            sim.reset();
                            cuquantum_circuit(&mut sim, nq, nl);
                            black_box(());
                        });
                    },
                );
            }
            Err(e) => eprintln!("CuStateVec not available: {e}"),
        }
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Benchmark group 2: Individual gate comparison
// ---------------------------------------------------------------------------

fn bench_native_individual_gates<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Native Individual Gates");
    group.sample_size(50);

    let num_qubits: usize = 18;
    let iters: usize = 100;

    // ---- H gate ----

    group.bench_function("H/StateVecSoA", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        sim.set_parallel(false);
        sim.set_fusion(false);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.h(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("H/StateVecSoA_fused", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        sim.set_parallel(false);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.h(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("H/StateVecAoS", |b| {
        let mut sim = StateVecAoS::new(num_qubits);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.h(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("H/StateVecSoA32", |b| {
        let mut sim = StateVecSoA32::new(num_qubits);
        sim.set_fusion(false);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.h(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("H/StateVecSoA32_fused", |b| {
        let mut sim = StateVecSoA32::new(num_qubits);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.h(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    #[cfg(feature = "gpu-sims")]
    if let Ok(mut sim) = GpuStateVec32::new(num_qubits as u32) {
        group.bench_function("H/GpuStateVec32_direct", |b| {
            b.iter(|| {
                for _ in 0..iters {
                    for q in 0..num_qubits {
                        sim.apply_single_gate(q as u32, gpu_gates::H);
                    }
                }
                black_box(());
            });
        });
    }

    #[cfg(feature = "cuquantum")]
    if let Ok(mut sim) = CuStateVec::new(num_qubits) {
        group.bench_function("H/CuStateVec_direct", |b| {
            b.iter(|| {
                for _ in 0..iters {
                    for q in 0..num_qubits {
                        sim.apply_matrix_1q(q, &cuquantum_matrices::H);
                    }
                }
                black_box(());
            });
        });
    }

    // ---- X gate ----

    group.bench_function("X/StateVecSoA", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        sim.set_parallel(false);
        sim.set_fusion(false);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.x(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("X/StateVecSoA_fused", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        sim.set_parallel(false);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.x(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("X/StateVecAoS", |b| {
        let mut sim = StateVecAoS::new(num_qubits);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.x(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("X/StateVecSoA32", |b| {
        let mut sim = StateVecSoA32::new(num_qubits);
        sim.set_fusion(false);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.x(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("X/StateVecSoA32_fused", |b| {
        let mut sim = StateVecSoA32::new(num_qubits);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.x(&[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    #[cfg(feature = "gpu-sims")]
    if let Ok(mut sim) = GpuStateVec32::new(num_qubits as u32) {
        group.bench_function("X/GpuStateVec32_direct", |b| {
            b.iter(|| {
                for _ in 0..iters {
                    for q in 0..num_qubits {
                        sim.apply_single_gate(q as u32, gpu_gates::X);
                    }
                }
                black_box(());
            });
        });
    }

    #[cfg(feature = "cuquantum")]
    if let Ok(mut sim) = CuStateVec::new(num_qubits) {
        group.bench_function("X/CuStateVec_direct", |b| {
            b.iter(|| {
                for _ in 0..iters {
                    for q in 0..num_qubits {
                        sim.apply_matrix_1q(q, &cuquantum_matrices::X);
                    }
                }
                black_box(());
            });
        });
    }

    // ---- CX gate ----

    group.bench_function("CX/StateVecSoA", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        sim.set_parallel(false);
        sim.set_fusion(false);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits - 1 {
                    sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("CX/StateVecSoA_fused", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        sim.set_parallel(false);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits - 1 {
                    sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("CX/StateVecAoS", |b| {
        let mut sim = StateVecAoS::new(num_qubits);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits - 1 {
                    sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("CX/StateVecSoA32", |b| {
        let mut sim = StateVecSoA32::new(num_qubits);
        sim.set_fusion(false);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits - 1 {
                    sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("CX/StateVecSoA32_fused", |b| {
        let mut sim = StateVecSoA32::new(num_qubits);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits - 1 {
                    sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                }
            }
            black_box(());
        });
    });

    #[cfg(feature = "gpu-sims")]
    if let Ok(mut sim) = GpuStateVec32::new(num_qubits as u32) {
        group.bench_function("CX/GpuStateVec32_direct", |b| {
            b.iter(|| {
                for _ in 0..iters {
                    for q in 0..num_qubits - 1 {
                        sim.apply_cx(q as u32, (q + 1) as u32);
                    }
                }
                black_box(());
            });
        });
    }

    #[cfg(feature = "cuquantum")]
    if let Ok(mut sim) = CuStateVec::new(num_qubits) {
        group.bench_function("CX/CuStateVec_direct", |b| {
            b.iter(|| {
                for _ in 0..iters {
                    for q in 0..num_qubits - 1 {
                        sim.apply_matrix_2q(q, q + 1, &cuquantum_matrices::CX);
                    }
                }
                black_box(());
            });
        });
    }

    // ---- RZ gate ----

    group.bench_function("RZ/StateVecSoA", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        sim.set_parallel(false);
        sim.set_fusion(false);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.rz(Angle64::from_radians(0.1), &[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("RZ/StateVecSoA_fused", |b| {
        let mut sim = StateVecSoA::new(num_qubits);
        sim.set_parallel(false);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.rz(Angle64::from_radians(0.1), &[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("RZ/StateVecAoS", |b| {
        let mut sim = StateVecAoS::new(num_qubits);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.rz(Angle64::from_radians(0.1), &[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("RZ/StateVecSoA32", |b| {
        let mut sim = StateVecSoA32::new(num_qubits);
        sim.set_fusion(false);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.rz(Angle64::from_radians(0.1), &[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    group.bench_function("RZ/StateVecSoA32_fused", |b| {
        let mut sim = StateVecSoA32::new(num_qubits);
        b.iter(|| {
            for _ in 0..iters {
                for q in 0..num_qubits {
                    sim.rz(Angle64::from_radians(0.1), &[QubitId(q)]);
                }
            }
            black_box(());
        });
    });

    #[cfg(feature = "gpu-sims")]
    if let Ok(mut sim) = GpuStateVec32::new(num_qubits as u32) {
        let rz_matrix = gpu_gates::rz(0.1);
        group.bench_function("RZ/GpuStateVec32_direct", |b| {
            b.iter(|| {
                for _ in 0..iters {
                    for q in 0..num_qubits {
                        sim.apply_single_gate(q as u32, rz_matrix);
                    }
                }
                black_box(());
            });
        });
    }

    #[cfg(feature = "cuquantum")]
    if let Ok(mut sim) = CuStateVec::new(num_qubits) {
        let rz_matrix = cuquantum_matrices::rz(0.1);
        group.bench_function("RZ/CuStateVec_direct", |b| {
            b.iter(|| {
                for _ in 0..iters {
                    for q in 0..num_qubits {
                        sim.apply_matrix_1q(q, &rz_matrix);
                    }
                }
                black_box(());
            });
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_native_statevec_comparison(c);
    bench_native_individual_gates(c);
}
