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

//! Evaluate `QuiZX` circuit simplification for T-count reduction.
//!
//! Tests whether ZX-calculus simplification meaningfully reduces the number
//! of non-Clifford gates in circuits relevant to the `CliffordRz` simulator.

use criterion::{BenchmarkId, Criterion, measurement::Measurement};
use quizx::circuit::Circuit;
use quizx::extract::ToCircuit;
use quizx::gate::GType;
use quizx::phase::Phase;
use quizx::simplify;
use quizx::vec_graph::Graph;
use std::hint::black_box;

pub fn benchmarks<M: Measurement>(c: &mut Criterion<M>) {
    eval_t_count_reduction();
    bench_simplification_time(c);
}

/// Print T-count reduction for various circuit patterns (not a timed benchmark).
fn eval_t_count_reduction() {
    println!("\n=== QuiZX T-count Reduction Evaluation ===\n");

    // Pattern 1: T gates on different qubits (no cancellation expected)
    {
        let nq = 10;
        let mut circ = Circuit::new(nq);
        for q in 0..nq {
            circ.add_gate("h", vec![q]);
        }
        for q in 0..nq {
            circ.add_gate("t", vec![q]);
        }
        let t_before = count_non_clifford(&circ);
        let simplified = simplify_circuit(&circ);
        let t_after = count_non_clifford(&simplified);
        println!("Pattern 1 ({nq} T on different qubits): {t_before} -> {t_after} non-Clifford");
    }

    // Pattern 2: T-T on same qubit = S (should be eliminated)
    {
        let nq = 5;
        let mut circ = Circuit::new(nq);
        for q in 0..nq {
            circ.add_gate("h", vec![q]);
        }
        for q in 0..nq {
            circ.add_gate("t", vec![q]);
            circ.add_gate("t", vec![q]); // T*T = S (Clifford)
        }
        let t_before = count_non_clifford(&circ);
        let simplified = simplify_circuit(&circ);
        let t_after = count_non_clifford(&simplified);
        println!("Pattern 2 ({nq} T-T pairs = S): {t_before} -> {t_after} non-Clifford");
    }

    // Pattern 3: T gates interleaved with CX (some cancellation through gadget fusion)
    {
        let nq = 6;
        let mut circ = Circuit::new(nq);
        for q in 0..nq {
            circ.add_gate("h", vec![q]);
        }
        for q in 0..nq - 1 {
            circ.add_gate("t", vec![q]);
            circ.add_gate("cx", vec![q, q + 1]);
        }
        circ.add_gate("t", vec![nq - 1]);
        // Second layer
        for q in 0..nq - 1 {
            circ.add_gate("t", vec![q]);
            circ.add_gate("cx", vec![q, q + 1]);
        }
        let t_before = count_non_clifford(&circ);
        let simplified = simplify_circuit(&circ);
        let t_after = count_non_clifford(&simplified);
        println!("Pattern 3 ({nq}q T-CX layers): {t_before} -> {t_after} non-Clifford");
    }

    // Pattern 4: Random Clifford + T circuit
    {
        let nq = 10;
        let mut circ = Circuit::new(nq);
        // Repeat: H-layer, CX-chain, T-layer
        for _layer in 0..3 {
            for q in 0..nq {
                circ.add_gate("h", vec![q]);
            }
            for q in 0..nq - 1 {
                circ.add_gate("cx", vec![q, q + 1]);
            }
            for q in 0..nq {
                circ.add_gate("t", vec![q]);
            }
        }
        let t_before = count_non_clifford(&circ);
        let simplified = simplify_circuit(&circ);
        let t_after = count_non_clifford(&simplified);
        println!("Pattern 4 (10q, 3 layers H-CX-T): {t_before} -> {t_after} non-Clifford");
    }

    // Pattern 5: Small RZ rotations (rational angle approximation)
    {
        let nq = 6;
        let mut circ = Circuit::new(nq);
        for q in 0..nq {
            circ.add_gate("h", vec![q]);
        }
        // RZ(pi/8) on each qubit -- can QuiZX simplify these?
        for q in 0..nq {
            // Phase in half-turns: 1/8 of a half-turn = pi/8 radians
            circ.add_gate_with_phase("rz", vec![q], Phase::new(::num::Rational64::new(1, 8)));
        }
        let t_before = count_non_clifford(&circ);
        let simplified = simplify_circuit(&circ);
        let t_after = count_non_clifford(&simplified);
        println!("Pattern 5 ({nq}q RZ(pi/8)): {t_before} -> {t_after} non-Clifford");
    }

    println!();
}

/// Benchmark the time cost of `QuiZX` simplification.
fn bench_simplification_time<M: Measurement>(c: &mut Criterion<M>) {
    let mut group = c.benchmark_group("QuiZX Simplification Time");
    group.sample_size(20);

    for &nq in &[10, 20, 50] {
        group.bench_with_input(
            BenchmarkId::new("3_layers_H_CX_T", format!("{nq}q")),
            &nq,
            |b, &nq| {
                let mut circ = Circuit::new(nq);
                for _layer in 0..3 {
                    for q in 0..nq {
                        circ.add_gate("h", vec![q]);
                    }
                    for q in 0..nq - 1 {
                        circ.add_gate("cx", vec![q, q + 1]);
                    }
                    for q in 0..nq {
                        circ.add_gate("t", vec![q]);
                    }
                }
                b.iter(|| {
                    let simplified = simplify_circuit(&circ);
                    black_box(count_non_clifford(&simplified));
                });
            },
        );
    }
    group.finish();
}

/// Simplify a circuit via ZX-calculus and extract back.
fn simplify_circuit(circ: &Circuit) -> Circuit {
    let mut g: Graph = circ.to_graph();
    simplify::full_simp(&mut g);
    match g.to_circuit() {
        Ok(c) => c,
        Err(_) => {
            // Extraction failed -- return original
            circ.clone()
        }
    }
}

/// Count non-Clifford gates in a circuit.
fn count_non_clifford(circ: &Circuit) -> usize {
    circ.gates
        .iter()
        .filter(|g| match g.t {
            GType::ZPhase | GType::XPhase => !g.phase.is_clifford(),
            GType::T | GType::Tdg => true,
            _ => false,
        })
        .count()
}
