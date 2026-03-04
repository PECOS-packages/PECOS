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

//! Benchmarks for pecos-cuquantum GPU simulators.
//!
//! Benchmarks cuQuantum state vector and stabilizer simulation performance.
//!
//! Run with: `cargo bench -p benchmarks --features cuquantum`
//!
//! **Requires cuQuantum to be installed.**

use criterion::{BenchmarkId, Criterion, Throughput};
use pecos_core::Angle64;
use pecos_cuquantum::{CuStabilizer, CuStateVec, QubitId, TryClone, is_cuquantum_available};
use pecos_qsim::{ArbitraryRotationGateable, CliffordGateable, QuantumSimulator};
use std::f64::consts::PI;
use std::hint::black_box;

/// Benchmark state vector simulation for different qubit counts
fn bench_statevec_gates(c: &mut Criterion) {
    if !is_cuquantum_available() {
        eprintln!("Skipping statevec benchmarks: cuQuantum not available");
        return;
    }

    let mut group = c.benchmark_group("custatevec_gates");

    for num_qubits in [4, 8, 12, 16, 20] {
        let mut sim = CuStateVec::new(num_qubits).expect("Failed to create simulator");

        group.throughput(Throughput::Elements(num_qubits as u64));

        // Benchmark H gates
        group.bench_with_input(
            BenchmarkId::new("hadamard", num_qubits),
            &num_qubits,
            |b, &n| {
                b.iter(|| {
                    for i in 0..n {
                        sim.h(&[QubitId(i)]);
                    }
                    black_box(&sim);
                });
            },
        );

        // Benchmark CX gates (linear chain)
        if num_qubits >= 2 {
            group.bench_with_input(
                BenchmarkId::new("cx_chain", num_qubits),
                &num_qubits,
                |b, &n| {
                    b.iter(|| {
                        for i in 0..n - 1 {
                            sim.cx(&[QubitId(i), QubitId(i + 1)]);
                        }
                        black_box(&sim);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark stabilizer simulation for large qubit counts
fn bench_stabilizer_gates(c: &mut Criterion) {
    if !is_cuquantum_available() {
        eprintln!("Skipping stabilizer benchmarks: cuQuantum not available");
        return;
    }

    let mut group = c.benchmark_group("custabilizer_gates");

    // Stabilizer can handle many more qubits
    for num_qubits in [100, 200, 500, 1000] {
        let mut sim = CuStabilizer::new(num_qubits).expect("Failed to create simulator");

        group.throughput(Throughput::Elements(num_qubits as u64));

        // Benchmark H gates
        group.bench_with_input(
            BenchmarkId::new("hadamard", num_qubits),
            &num_qubits,
            |b, &n| {
                b.iter(|| {
                    for i in 0..n {
                        sim.h(&[QubitId(i)]);
                    }
                    black_box(&sim);
                });
            },
        );

        // Benchmark CX chain
        if num_qubits >= 2 {
            group.bench_with_input(
                BenchmarkId::new("cx_chain", num_qubits),
                &num_qubits,
                |b, &n| {
                    b.iter(|| {
                        for i in 0..n - 1 {
                            sim.cx(&[QubitId(i), QubitId(i + 1)]);
                        }
                        black_box(&sim);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark Bell state creation and measurement
fn bench_bell_state(c: &mut Criterion) {
    if !is_cuquantum_available() {
        eprintln!("Skipping bell state benchmarks: cuQuantum not available");
        return;
    }

    let mut group = c.benchmark_group("bell_state");

    for num_qubits in [2, 10, 20] {
        // State vector benchmark
        group.bench_with_input(
            BenchmarkId::new("statevec", num_qubits),
            &num_qubits,
            |b, &n| {
                let mut sim = CuStateVec::new(n).expect("Failed to create simulator");
                b.iter(|| {
                    sim.reset();
                    // Create GHZ state
                    sim.h(&[QubitId(0)]);
                    for i in 0..n - 1 {
                        sim.cx(&[QubitId(i), QubitId(i + 1)]);
                    }
                    let qubits: Vec<_> = (0..n).map(QubitId).collect();
                    let results = sim.mz(&qubits);
                    black_box(results);
                });
            },
        );

        // Stabilizer benchmark
        group.bench_with_input(
            BenchmarkId::new("stabilizer", num_qubits),
            &num_qubits,
            |b, &n| {
                let mut sim = CuStabilizer::new(n).expect("Failed to create simulator");
                b.iter(|| {
                    sim.reset();
                    // Create GHZ state
                    sim.h(&[QubitId(0)]);
                    for i in 0..n - 1 {
                        sim.cx(&[QubitId(i), QubitId(i + 1)]);
                    }
                    let qubits: Vec<_> = (0..n).map(QubitId).collect();
                    let results = sim.mz(&qubits);
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark surface code syndrome extraction
fn bench_surface_code_syndrome(c: &mut Criterion) {
    if !is_cuquantum_available() {
        eprintln!("Skipping surface code benchmarks: cuQuantum not available");
        return;
    }

    let mut group = c.benchmark_group("surface_code_syndrome");

    // Distance 3, 5, 7 surface codes
    for distance in [3, 5, 7] {
        let num_data_qubits = distance * distance;
        let num_ancilla = (distance - 1) * distance; // Z stabilizers
        let total_qubits = num_data_qubits + num_ancilla;

        group.bench_with_input(
            BenchmarkId::new("stabilizer", format!("d{distance}")),
            &(total_qubits, num_data_qubits, num_ancilla),
            |b, &(total, _data, ancilla)| {
                let mut sim = CuStabilizer::new(total).expect("Failed to create simulator");
                b.iter(|| {
                    sim.reset();
                    // Simplified syndrome extraction - just H and CX gates
                    for i in 0..ancilla {
                        sim.h(&[QubitId(i)]);
                    }
                    // Some CX gates to simulate stabilizer measurement
                    for i in 0..ancilla {
                        if i + ancilla < total {
                            sim.cx(&[QubitId(i), QubitId(i + ancilla)]);
                        }
                    }
                    for i in 0..ancilla {
                        sim.h(&[QubitId(i)]);
                    }
                    // Measure ancillas
                    let qubits: Vec<_> = (0..ancilla).map(QubitId).collect();
                    let results = sim.mz(&qubits);
                    black_box(results);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark rotation gates on state vectors
fn bench_rotation_gates(c: &mut Criterion) {
    if !is_cuquantum_available() {
        eprintln!("Skipping rotation benchmarks: cuQuantum not available");
        return;
    }

    let mut group = c.benchmark_group("rotation_gates");

    for num_qubits in [4, 8, 12, 16] {
        let mut sim = CuStateVec::new(num_qubits).expect("Failed to create simulator");

        group.throughput(Throughput::Elements(num_qubits as u64));

        // Benchmark RX gates
        group.bench_with_input(BenchmarkId::new("rx", num_qubits), &num_qubits, |b, &n| {
            b.iter(|| {
                for i in 0..n {
                    sim.rx(Angle64::from_radians(PI / 4.0), &[QubitId(i)]);
                }
                black_box(&sim);
            });
        });

        // Benchmark RZ gates
        group.bench_with_input(BenchmarkId::new("rz", num_qubits), &num_qubits, |b, &n| {
            b.iter(|| {
                for i in 0..n {
                    sim.rz(Angle64::from_radians(PI / 4.0), &[QubitId(i)]);
                }
                black_box(&sim);
            });
        });

        // Benchmark T gates
        group.bench_with_input(BenchmarkId::new("t", num_qubits), &num_qubits, |b, &n| {
            b.iter(|| {
                for i in 0..n {
                    sim.t(&[QubitId(i)]);
                }
                black_box(&sim);
            });
        });
    }

    group.finish();
}

/// Benchmark two-qubit rotation gates on state vectors
fn bench_two_qubit_rotation_gates(c: &mut Criterion) {
    if !is_cuquantum_available() {
        eprintln!("Skipping two-qubit rotation benchmarks: cuQuantum not available");
        return;
    }

    let mut group = c.benchmark_group("two_qubit_rotation_gates");

    for num_qubits in [4, 8, 12, 16] {
        if num_qubits < 2 {
            continue;
        }

        let mut sim = CuStateVec::new(num_qubits).expect("Failed to create simulator");
        let num_pairs = (num_qubits - 1) as u64;
        group.throughput(Throughput::Elements(num_pairs));

        // Benchmark RZZ gates (chain)
        group.bench_with_input(
            BenchmarkId::new("rzz_chain", num_qubits),
            &num_qubits,
            |b, &n| {
                b.iter(|| {
                    for i in 0..n - 1 {
                        sim.rzz(
                            Angle64::from_radians(PI / 4.0),
                            &[QubitId(i), QubitId(i + 1)],
                        );
                    }
                    black_box(&sim);
                });
            },
        );

        // Benchmark RXX gates (chain)
        group.bench_with_input(
            BenchmarkId::new("rxx_chain", num_qubits),
            &num_qubits,
            |b, &n| {
                b.iter(|| {
                    for i in 0..n - 1 {
                        sim.rxx(
                            Angle64::from_radians(PI / 4.0),
                            &[QubitId(i), QubitId(i + 1)],
                        );
                    }
                    black_box(&sim);
                });
            },
        );

        // Benchmark RYY gates (chain)
        group.bench_with_input(
            BenchmarkId::new("ryy_chain", num_qubits),
            &num_qubits,
            |b, &n| {
                b.iter(|| {
                    for i in 0..n - 1 {
                        sim.ryy(
                            Angle64::from_radians(PI / 4.0),
                            &[QubitId(i), QubitId(i + 1)],
                        );
                    }
                    black_box(&sim);
                });
            },
        );
    }

    group.finish();
}

/// Benchmark sampling from state vectors
fn bench_sampling(c: &mut Criterion) {
    if !is_cuquantum_available() {
        eprintln!("Skipping sampling benchmarks: cuQuantum not available");
        return;
    }

    let mut group = c.benchmark_group("sampling");

    for num_qubits in [4, 8, 12, 16] {
        let mut sim = CuStateVec::new(num_qubits).expect("Failed to create simulator");

        // Create a superposition state
        for i in 0..num_qubits {
            sim.h(&[QubitId(i)]);
        }

        // Benchmark sampling different numbers of shots
        for num_samples in [100, 1000, 10000] {
            group.throughput(Throughput::Elements(num_samples as u64));
            group.bench_with_input(
                BenchmarkId::new(format!("q{num_qubits}"), num_samples),
                &num_samples,
                |b, &n| {
                    b.iter(|| {
                        let samples = sim.sample(n);
                        black_box(samples);
                    });
                },
            );
        }
    }

    group.finish();
}

/// Benchmark Clone and `TryClone` operations
fn bench_clone(c: &mut Criterion) {
    if !is_cuquantum_available() {
        eprintln!("Skipping clone benchmarks: cuQuantum not available");
        return;
    }

    let mut group = c.benchmark_group("clone");

    for num_qubits in [4, 8, 12, 16, 20] {
        let sim = CuStateVec::new(num_qubits).expect("Failed to create simulator");

        // Benchmark Clone
        group.bench_with_input(
            BenchmarkId::new("statevec_clone", num_qubits),
            &num_qubits,
            |b, _| {
                b.iter(|| {
                    let cloned = sim.clone();
                    black_box(cloned);
                });
            },
        );

        // Benchmark TryClone
        group.bench_with_input(
            BenchmarkId::new("statevec_try_clone", num_qubits),
            &num_qubits,
            |b, _| {
                b.iter(|| {
                    let cloned = sim.try_clone().expect("try_clone failed");
                    black_box(cloned);
                });
            },
        );
    }

    // Stabilizer clone is different (creates new instance, doesn't copy state)
    for num_qubits in [100, 500, 1000] {
        let sim = CuStabilizer::new(num_qubits).expect("Failed to create simulator");

        group.bench_with_input(
            BenchmarkId::new("stabilizer_clone", num_qubits),
            &num_qubits,
            |b, _| {
                b.iter(|| {
                    let cloned = sim.clone();
                    black_box(cloned);
                });
            },
        );
    }

    group.finish();
}

pub fn benchmarks(c: &mut Criterion) {
    bench_statevec_gates(c);
    bench_stabilizer_gates(c);
    bench_bell_state(c);
    bench_surface_code_syndrome(c);
    bench_rotation_gates(c);
    bench_two_qubit_rotation_gates(c);
    bench_sampling(c);
    bench_clone(c);
}
