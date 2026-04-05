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

//! Benchmarks for the CH-form and Clifford+RZ simulators.
//!
//! Measures:
//! - `CHForm` vs `SparseStab` Clifford gate throughput
//! - RZ gate cost (term doubling)
//! - State vector / amplitude computation
//! - Measurement cost

use criterion::{BenchmarkId, Criterion, measurement::Measurement};
use pecos_core::{Angle64, QubitId};
use pecos_simulators::{
    ArbitraryRotationGateable, CHForm, CliffordGateable, CliffordRz, QuantumSimulator, SparseStab,
};
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    bench_clifford_gate_throughput(c);
    bench_rz_term_growth(c);
    bench_state_vector(c);
    bench_measurement(c);
    bench_amplitude(c);
    bench_inner_product(c);
    bench_clone_cost(c);
    bench_measurement_only(c);
    bench_realistic_circuit(c);
    bench_rz_commutation_opportunity(c);
    bench_t_gate_patterns(c);
    bench_rz_fusion_opportunity(c);
    bench_small_angle_rz(c);
    bench_end_to_end(c);
    bench_measurement_scaling(c);
}

/// Compare `CHForm` vs `SparseStab` on a Clifford-only circuit.
fn bench_clifford_gate_throughput<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Clifford Gate Throughput");
    group.sample_size(50);

    for &num_qubits in &[4, 8, 16, 32] {
        // Build a fixed circuit: H on all, then CX chain, then S on all, then H on all
        group.bench_with_input(
            BenchmarkId::new("SparseStab", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = SparseStab::with_seed(nq, 42);
                b.iter(|| {
                    sim.reset();
                    run_clifford_circuit(&mut sim, nq);
                    black_box(());
                });
            },
        );

        group.bench_with_input(
            BenchmarkId::new("CHForm", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = CHForm::new_with_seed(nq, 42);
                b.iter(|| {
                    sim.reset();
                    run_clifford_circuit(&mut sim, nq);
                    black_box(());
                });
            },
        );
    }
    group.finish();
}

/// Measure the cost of RZ gates (term doubling) at various term counts.
#[allow(clippy::cast_precision_loss)] // small loop index as f64
fn bench_rz_term_growth<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("RZ Term Growth");
    group.sample_size(20);

    // 2 qubits, varying number of RZ gates
    let num_qubits = 2;
    for &num_rz in &[1, 2, 4, 6, 8] {
        group.bench_with_input(
            BenchmarkId::new("CliffordRz", format!("{num_rz}_rz")),
            &num_rz,
            |b, &nrz| {
                b.iter(|| {
                    let mut sim = CliffordRz::new_with_seed(num_qubits, 42);
                    sim.h(&[QubitId(0)]).h(&[QubitId(1)]);
                    for i in 0..nrz {
                        let theta = Angle64::from_radians(0.3 + 0.1 * i as f64);
                        sim.rz(theta, &[QubitId(i % num_qubits)]);
                    }
                    black_box(sim.num_terms());
                });
            },
        );
    }
    group.finish();
}

/// Benchmark state vector computation at various qubit counts and term counts.
#[allow(clippy::cast_precision_loss)] // small loop index as f64
fn bench_state_vector<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("State Vector Computation");
    group.sample_size(20);

    // Vary qubit count with 1 RZ gate (2 terms)
    for &num_qubits in &[2, 4, 6, 8, 10] {
        group.bench_with_input(
            BenchmarkId::new("2_terms", format!("{num_qubits}q")),
            &num_qubits,
            |b, &nq| {
                let mut sim = CliffordRz::new_with_seed(nq, 42);
                sim.h(&[QubitId(0)]);
                if nq > 1 {
                    sim.cx(&[(QubitId(0), QubitId(1))]);
                }
                sim.rz(Angle64::from_radians(0.5), &[QubitId(0)]);
                b.iter(|| {
                    black_box(sim.state_vector());
                });
            },
        );
    }

    // Vary term count with 4 qubits
    let nq = 4;
    for &num_rz in &[1, 2, 3, 4, 5] {
        let terms = 1 << num_rz;
        group.bench_with_input(
            BenchmarkId::new(format!("{terms}_terms"), format!("{nq}q_vary_terms")),
            &num_rz,
            |b, &nrz| {
                let mut sim = CliffordRz::new_with_seed(nq, 42);
                for q in 0..nq {
                    sim.h(&[QubitId(q)]);
                }
                for i in 0..nrz {
                    sim.rz(
                        Angle64::from_radians(0.3 + 0.1 * i as f64),
                        &[QubitId(i % nq)],
                    );
                }
                b.iter(|| {
                    black_box(sim.state_vector());
                });
            },
        );
    }
    group.finish();
}

/// Benchmark measurement cost.
#[allow(clippy::cast_precision_loss)] // small loop index as f64
fn bench_measurement<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("CliffordRz Measurement");
    group.sample_size(20);

    for &num_rz in &[1, 2, 3, 4] {
        let terms = 1 << num_rz;
        group.bench_with_input(
            BenchmarkId::new(format!("{terms}_terms"), "4q"),
            &num_rz,
            |b, &nrz| {
                b.iter(|| {
                    let mut sim = CliffordRz::new_with_seed(4, 42);
                    for q in 0..4 {
                        sim.h(&[QubitId(q)]);
                    }
                    for i in 0..nrz {
                        sim.rz(
                            Angle64::from_radians(0.3 + 0.1 * i as f64),
                            &[QubitId(i % 4)],
                        );
                    }
                    let result = sim.mz(&[QubitId(0)]);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark single amplitude computation.
fn bench_amplitude<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("CHForm Amplitude");
    group.sample_size(50);

    for &num_qubits in &[4, 8, 16, 32] {
        group.bench_with_input(
            BenchmarkId::new("single_amplitude", num_qubits),
            &num_qubits,
            |b, &nq| {
                let mut sim = CHForm::new_with_seed(nq, 42);
                run_clifford_circuit(&mut sim, nq);
                b.iter(|| {
                    black_box(sim.amplitude(0));
                });
            },
        );
    }
    group.finish();
}

/// Benchmark just the inner product computation (no circuit building).
fn bench_inner_product<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("CHForm Inner Product");
    group.sample_size(10);

    for &num_qubits in &[4, 8, 14, 22, 60, 62, 63, 64, 100, 126, 127, 128, 150] {
        group.bench_with_input(
            BenchmarkId::new("pairwise", format!("{num_qubits}q")),
            &num_qubits,
            |b, &nq| {
                // Build two CH-form states that differ by Z on q0 (simulates RZ decomposition)
                let mut ch1 = CHForm::new_with_seed(nq, 42);
                for q in 0..nq {
                    ch1.h(&[QubitId(q)]);
                }
                if nq > 1 {
                    ch1.cx(&[(QubitId(0), QubitId(1))]);
                }
                let mut ch2 = ch1.clone();
                ch2.z(&[QubitId(0)]);

                b.iter(|| {
                    black_box(ch1.inner_product(&ch2));
                });
            },
        );
    }
    group.finish();
}

/// Benchmark ONLY the measurement call (circuit pre-built).
fn bench_measurement_only<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("CliffordRz Measurement Only");
    group.sample_size(10);

    for &num_qubits in &[4, 8, 14, 18, 22] {
        group.bench_with_input(
            BenchmarkId::new("4_terms", format!("{num_qubits}q")),
            &num_qubits,
            |b, &nq| {
                // Pre-build the circuit state
                let mut template = CliffordRz::new_with_seed(nq, 42);
                for q in 0..nq {
                    template.h(&[QubitId(q)]);
                }
                if nq > 1 {
                    template.cx(&[(QubitId(0), QubitId(1))]);
                }
                template.rz(Angle64::from_radians(0.5), &[QubitId(0)]);
                template.rz(Angle64::from_radians(0.8), &[QubitId(nq.min(2) - 1)]);

                b.iter(|| {
                    let mut sim = template.clone();
                    let result = sim.mz(&[QubitId(0)]);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark clone cost for CH-form states.
fn bench_clone_cost<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("CHForm Clone Cost");
    group.sample_size(50);

    for &num_qubits in &[4, 14, 22, 30] {
        group.bench_with_input(
            BenchmarkId::new("clone", format!("{num_qubits}q")),
            &num_qubits,
            |b, &nq| {
                let mut ch = CHForm::new_with_seed(nq, 42);
                for q in 0..nq {
                    ch.h(&[QubitId(q)]);
                }
                b.iter(|| {
                    black_box(ch.clone());
                });
            },
        );
    }
    group.finish();
}

/// Benchmark measurement at higher qubit counts to show where O(2^n) becomes the bottleneck.
fn bench_measurement_scaling<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("CliffordRz Measurement Scaling");
    group.sample_size(10);

    // Fixed 2 RZ gates (4 terms), vary qubit count
    for &num_qubits in &[4, 8, 10, 12, 14, 18, 22] {
        group.bench_with_input(
            BenchmarkId::new("4_terms", format!("{num_qubits}q")),
            &num_qubits,
            |b, &nq| {
                b.iter(|| {
                    let mut sim = CliffordRz::new_with_seed(nq, 42);
                    for q in 0..nq {
                        sim.h(&[QubitId(q)]);
                    }
                    if nq > 1 {
                        sim.cx(&[(QubitId(0), QubitId(1))]);
                    }
                    sim.rz(Angle64::from_radians(0.5), &[QubitId(0)]);
                    sim.rz(Angle64::from_radians(0.8), &[QubitId(nq.min(2) - 1)]);
                    let result = sim.mz(&[QubitId(0)]);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

/// Benchmark a realistic interleaved Clifford+RZ circuit.
/// Measures total circuit execution time and isolates Clifford vs RZ costs.
fn bench_realistic_circuit<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Realistic Clifford+RZ Circuit");
    group.sample_size(10);

    let num_qubits = 20;
    let theta = Angle64::from_radians(0.3);

    // Benchmark: Clifford cost at various term counts
    for &num_rz in &[2, 4, 6] {
        let terms = 1 << num_rz;
        let theta = Angle64::from_radians(0.3);
        group.bench_with_input(
            BenchmarkId::new("20_H_gates", format!("{terms}_terms")),
            &num_rz,
            |b, &nrz| {
                let mut template = CliffordRz::new_with_seed(num_qubits, 42);
                for q in 0..num_qubits {
                    template.h(&[QubitId(q)]);
                }
                for i in 0..nrz {
                    template.rz(theta, &[QubitId(i % num_qubits)]);
                }
                b.iter(|| {
                    let mut sim = template.clone();
                    for q in 0..num_qubits {
                        sim.h(&[QubitId(q)]);
                    }
                    black_box(sim.num_terms());
                });
            },
        );
    }

    // Benchmark 1: Clifford-only portion (no term doubling)
    group.bench_function("20q_clifford_only_4terms", |b| {
        // Pre-build a 4-term state
        let mut template = CliffordRz::new_with_seed(num_qubits, 42);
        for q in 0..num_qubits {
            template.h(&[QubitId(q)]);
        }
        template.rz(theta, &[QubitId(0)]);
        template.rz(theta, &[QubitId(1)]);
        assert_eq!(template.num_terms(), 4);

        b.iter(|| {
            let mut sim = template.clone();
            // Apply 20 Clifford gates to all 4 terms
            for q in 0..num_qubits {
                sim.h(&[QubitId(q)]);
            }
            black_box(sim.num_terms());
        });
    });

    // Benchmark 2: Single RZ (doubles terms from 4 to 8)
    group.bench_function("20q_single_rz_4to8terms", |b| {
        let mut template = CliffordRz::new_with_seed(num_qubits, 42);
        for q in 0..num_qubits {
            template.h(&[QubitId(q)]);
        }
        template.rz(theta, &[QubitId(0)]);
        template.rz(theta, &[QubitId(1)]);

        b.iter(|| {
            let mut sim = template.clone();
            sim.rz(theta, &[QubitId(2)]);
            black_box(sim.num_terms());
        });
    });

    // Benchmark 3: Measurement on 4-term, 20-qubit state
    group.bench_function("20q_measurement_4terms", |b| {
        let mut template = CliffordRz::new_with_seed(num_qubits, 42);
        for q in 0..num_qubits {
            template.h(&[QubitId(q)]);
        }
        template.rz(theta, &[QubitId(0)]);
        template.rz(theta, &[QubitId(1)]);

        b.iter(|| {
            let mut sim = template.clone();
            let result = sim.mz(&[QubitId(0)]);
            black_box(result);
        });
    });

    group.finish();
}

/// Show impact of RZ fusion through commuting Cliffords.
fn bench_rz_commutation_opportunity<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("RZ Commutation Opportunity");
    group.sample_size(10);

    let nq = 20;
    let theta = Angle64::from_radians(0.3);

    // Pattern: RZ(q) - S(q) - RZ(q). S commutes with RZ, so these could fuse.
    group.bench_function("rz_S_rz_same_qubit", |b| {
        b.iter(|| {
            let mut sim = CliffordRz::new_with_seed(nq, 42);
            for q in 0..nq {
                sim.h(&[QubitId(q)]);
            }
            sim.rz(theta, &[QubitId(0)]);
            sim.sz(&[QubitId(0)]); // S commutes with RZ -- should NOT break fusion
            sim.rz(theta, &[QubitId(0)]);
            let result = sim.mz(&[QubitId(0)]);
            black_box(result);
        });
    });

    // Pattern: RZ(q) - CZ(q,r) - RZ(q). CZ is diagonal, commutes with RZ.
    group.bench_function("rz_CZ_rz_same_qubit", |b| {
        b.iter(|| {
            let mut sim = CliffordRz::new_with_seed(nq, 42);
            for q in 0..nq {
                sim.h(&[QubitId(q)]);
            }
            sim.rz(theta, &[QubitId(0)]);
            sim.cz(&[(QubitId(0), QubitId(1))]); // CZ commutes with RZ
            sim.rz(theta, &[QubitId(0)]);
            let result = sim.mz(&[QubitId(0)]);
            black_box(result);
        });
    });

    // Ideal: manually fused (what we'd get with perfect commutation)
    group.bench_function("rz_fused_ideal", |b| {
        let fused = Angle64::from_radians(0.6);
        b.iter(|| {
            let mut sim = CliffordRz::new_with_seed(nq, 42);
            for q in 0..nq {
                sim.h(&[QubitId(q)]);
            }
            sim.rz(fused, &[QubitId(0)]);
            sim.sz(&[QubitId(0)]);
            let result = sim.mz(&[QubitId(0)]);
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark T-gate circuit patterns (realistic QEC-like).
fn bench_t_gate_patterns<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("T-Gate Patterns");
    group.sample_size(10);

    let nq = 20;
    // T gate = RZ(pi/4)
    let t_angle = Angle64::from_radians(std::f64::consts::FRAC_PI_4);

    // Pattern: T gates on same qubit fuse (T*T = S = Clifford)
    group.bench_function("4T_same_qubit_fuses_to_Z", |b| {
        b.iter(|| {
            let mut sim = CliffordRz::new_with_seed(nq, 42);
            for q in 0..nq {
                sim.h(&[QubitId(q)]);
            }
            // 4 T gates = Z (Clifford angle) -> 0 extra terms
            sim.rz(t_angle, &[QubitId(0)]);
            sim.rz(t_angle, &[QubitId(0)]);
            sim.rz(t_angle, &[QubitId(0)]);
            sim.rz(t_angle, &[QubitId(0)]);
            let result = sim.mz(&[QubitId(0)]);
            black_box(result);
        });
    });

    // Pattern: T on different qubits (no fusion possible)
    group.bench_function("4T_different_qubits", |b| {
        b.iter(|| {
            let mut sim = CliffordRz::new_with_seed(nq, 42);
            for q in 0..nq {
                sim.h(&[QubitId(q)]);
            }
            sim.rz(t_angle, &[QubitId(0)]);
            sim.rz(t_angle, &[QubitId(1)]);
            sim.rz(t_angle, &[QubitId(2)]);
            sim.rz(t_angle, &[QubitId(3)]);
            let result = sim.mz(&[QubitId(0)]);
            black_box(result);
        });
    });

    // Pattern: T-Clifford-T interleaved on same qubit
    group.bench_function("T_CZ_T_same_qubit", |b| {
        b.iter(|| {
            let mut sim = CliffordRz::new_with_seed(nq, 42);
            for q in 0..nq {
                sim.h(&[QubitId(q)]);
            }
            sim.rz(t_angle, &[QubitId(0)]);
            sim.cz(&[(QubitId(0), QubitId(1))]); // diagonal, commutes
            sim.rz(t_angle, &[QubitId(0)]); // fuses with first T: 2T = S
            let result = sim.mz(&[QubitId(0)]);
            black_box(result);
        });
    });

    group.finish();
}

/// Show impact of RZ cancellation: same qubit hit twice.
fn bench_rz_fusion_opportunity<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("RZ Fusion Opportunity");
    group.sample_size(10);

    let nq = 20;
    let theta = Angle64::from_radians(0.3);

    // Without fusion: 4 separate RZ gates on same qubit = 16 terms
    group.bench_function("4_separate_rz_same_qubit", |b| {
        b.iter(|| {
            let mut sim = CliffordRz::new_with_seed(nq, 42);
            for q in 0..nq {
                sim.h(&[QubitId(q)]);
            }
            sim.rz(theta, &[QubitId(0)]);
            sim.rz(theta, &[QubitId(0)]);
            sim.rz(theta, &[QubitId(0)]);
            sim.rz(theta, &[QubitId(0)]);
            let result = sim.mz(&[QubitId(0)]);
            black_box(result);
        });
    });

    // With manual fusion: 1 RZ gate with 4*theta = 2 terms
    group.bench_function("1_fused_rz_same_qubit", |b| {
        let fused_theta = Angle64::from_radians(0.3 * 4.0);
        b.iter(|| {
            let mut sim = CliffordRz::new_with_seed(nq, 42);
            for q in 0..nq {
                sim.h(&[QubitId(q)]);
            }
            sim.rz(fused_theta, &[QubitId(0)]);
            let result = sim.mz(&[QubitId(0)]);
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark small-angle RZ circuits (where term pruning matters).
fn bench_small_angle_rz<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("Small Angle RZ");
    group.sample_size(10);

    let nq = 20;

    // 10 small-angle RZ gates (5 degrees each)
    group.bench_function("10_rz_5deg_different_qubits", |b| {
        let theta = Angle64::from_radians(5.0f64.to_radians());
        b.iter(|| {
            let mut sim = CliffordRz::new_with_seed(nq, 42);
            for q in 0..nq {
                sim.h(&[QubitId(q)]);
            }
            for i in 0..10 {
                sim.rz(theta, &[QubitId(i % nq)]);
            }
            let result = sim.mz(&[QubitId(0)]);
            black_box(result);
        });
    });

    // Same but with 1 degree rotations (even more prunable)
    group.bench_function("10_rz_1deg_different_qubits", |b| {
        let theta = Angle64::from_radians(1.0f64.to_radians());
        b.iter(|| {
            let mut sim = CliffordRz::new_with_seed(nq, 42);
            for q in 0..nq {
                sim.h(&[QubitId(q)]);
            }
            for i in 0..10 {
                sim.rz(theta, &[QubitId(i % nq)]);
            }
            let result = sim.mz(&[QubitId(0)]);
            black_box(result);
        });
    });

    group.finish();
}

/// Benchmark end-to-end: build circuit + measure.
fn bench_end_to_end<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("End-to-End Clifford+RZ");
    group.sample_size(10);

    // Vary number of RZ gates (= log2 of term count)
    for &num_rz in &[2, 4, 6, 8] {
        let terms = 1 << num_rz;
        let nq = 20;
        group.bench_with_input(
            BenchmarkId::new(format!("{terms}_terms"), format!("{nq}q")),
            &num_rz,
            |b, &nrz| {
                let theta = Angle64::from_radians(0.3);
                b.iter(|| {
                    let mut sim = CliffordRz::new_with_seed(nq, 42);
                    // Initial Clifford layer
                    for q in 0..nq {
                        sim.h(&[QubitId(q)]);
                    }
                    for q in 0..nq - 1 {
                        sim.cx(&[(QubitId(q), QubitId(q + 1))]);
                    }
                    // RZ layers
                    for i in 0..nrz {
                        sim.rz(theta, &[QubitId(i % nq)]);
                    }
                    // Measure one qubit
                    let result = sim.mz(&[QubitId(0)]);
                    black_box(result);
                });
            },
        );
    }
    group.finish();
}

/// A fixed Clifford circuit for benchmarking.
fn run_clifford_circuit(sim: &mut impl CliffordGateable, num_qubits: usize) {
    // H on all qubits
    for q in 0..num_qubits {
        sim.h(&[QubitId(q)]);
    }
    // CX chain
    for q in 0..num_qubits - 1 {
        sim.cx(&[(QubitId(q), QubitId(q + 1))]);
    }
    // S on all
    for q in 0..num_qubits {
        sim.sz(&[QubitId(q)]);
    }
    // H on all
    for q in 0..num_qubits {
        sim.h(&[QubitId(q)]);
    }
}
